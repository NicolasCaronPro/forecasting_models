# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from .mesh_graph_mlp import MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum, MeshGraphMLP
from .utils import aggregate_and_concat


class MeshGraphEncoder(nn.Module):
    """Encoder used e.g. in GraphCast
       which acts on the bipartite graph connecting a mostly
       regular grid (e.g. representing the input domain) to a mesh
       (e.g. representing a latent space).

    Parameters
    ----------
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    input_dim_src_nodes : int, optional
        Input dimensionality of the source node features, by default 512
    input_dim_dst_nodes : int, optional
        Input dimensionality of the destination node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim_src_nodes : int, optional
        Output dimensionality of the source node features, by default 512
    output_dim_dst_nodes : int, optional
        Output dimensionality of the destination node features, by default 512
    output_dim_edges : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: int = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        # edge MLP
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # src node MLP
        self.src_node_mlp = MeshGraphMLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # dst node MLP
        self.dst_node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    @torch.jit.ignore()
    def forward(
        self,
        g2m_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: DGLGraph,
    ) -> Tuple[Tensor, Tensor]:

        has_time_dim = len(grid_nfeat.shape) == 3
        if has_time_dim:
            timesteps = grid_nfeat.size(0)
            mesh_nfeat_new = []
            grid_nfeat_new = []
            for i in range(timesteps):
                # update edge features by concatenating node features (both mesh and grid)
                # and existing edge features (or applying the concat trick instead)
                efeat = self.edge_mlp(g2m_efeat, (grid_nfeat[i], mesh_nfeat), graph)
                # aggregate messages (edge features) to obtain updated node features
                cat_feat = aggregate_and_concat(
                    efeat, mesh_nfeat, graph, self.aggregation
                )
                # update src, dst node features + residual connections
                mesh_nfeat_new.append(mesh_nfeat + self.dst_node_mlp(cat_feat))
                grid_nfeat_new.append(grid_nfeat[i] + self.src_node_mlp(grid_nfeat[i]))
            return torch.stack(grid_nfeat_new), torch.stack(mesh_nfeat_new)
        else:
            # update edge features by concatenating node features (both mesh and grid)
            # and existing edge features (or applying the concat trick instead)
            efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
            # aggregate messages (edge features) to obtain updated node features
            cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
            # update src, dst node features + residual connections
            mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
            grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
            return grid_nfeat, mesh_nfeat

class MeshGraphEncoder(nn.Module):
    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: int = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        use_lstm: bool = True,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.use_lstm = use_lstm

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat

        # LSTM for temporal encoding (only for grid_nfeat here, but could be extended)
        if self.use_lstm:
            self.grid_lstm = nn.LSTM(
                input_size=input_dim_src_nodes,
                hidden_size=input_dim_src_nodes,
                batch_first=False
            )

        # edge MLP
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # src node MLP
        self.src_node_mlp = MeshGraphMLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        
        # dst node MLP
        self.dst_node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    @torch.jit.ignore()
    def forward(
        self,
        g2m_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: DGLGraph,
    ) -> Tuple[Tensor, Tensor]:

        has_time_dim = len(grid_nfeat.shape) == 3
        if has_time_dim:
            timesteps, num_nodes, feat_dim = grid_nfeat.size()

            if self.use_lstm:
                # Process grid_nfeat through LSTM
                grid_encoded, _ = self.grid_lstm(grid_nfeat)  # shape: (T, N, D)

            mesh_nfeat_new = []
            grid_nfeat_new = []

            for i in range(timesteps):
                grid_input = grid_encoded[i] if self.use_lstm else grid_nfeat[i]

                efeat = self.edge_mlp(g2m_efeat, (grid_input, mesh_nfeat), graph)
                cat_feat = aggregate_and_concat(
                    efeat, mesh_nfeat, graph, self.aggregation
                )
                mesh_nfeat_new.append(mesh_nfeat + self.dst_node_mlp(cat_feat))
                grid_nfeat_new.append(grid_input + self.src_node_mlp(grid_input))

            return torch.stack(grid_nfeat_new), torch.stack(mesh_nfeat_new)

        else:
            efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
            cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
            mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
            grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
            return grid_nfeat, mesh_nfeat