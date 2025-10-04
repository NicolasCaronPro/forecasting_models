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
import dgl.function as fn

from .mesh_graph_mlp import MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum, MeshGraphMLP, MeshGraphMLPAttention, MeshGraphMLPGAT
from .utils import aggregate_and_concat, aggregate_and_concat_with_attention, EdgeScoreDotProductTransformer, EdgeScoreDotProductGAT

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
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        use_lstm: bool = True,
        attention: str = None
    ):
        super().__init__()
        self.aggregation = aggregation
        self.use_lstm = use_lstm
        self.attention = attention

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat

        if attention == "Transformer":
            MLP_PROCESS = MeshGraphMLPAttention
        elif attention == "GAT":
            MLP_PROCESS = MeshGraphMLPGAT
        else:
            MLP_PROCESS = MeshGraphMLP

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
        
        if attention == "GAT":
            self.attention_score = EdgeScoreDotProductGAT(input_dim_src_nodes, input_dim_dst_nodes, output_dim_edges, num_heads=4, head_dim=32)
        elif attention == "Transformer":
            self.attention_score = EdgeScoreDotProductTransformer(input_dim_src_nodes, input_dim_dst_nodes, output_dim_edges, num_heads=4, head_dim=32)

        # src node MLP
        self.src_node_mlp = MLP_PROCESS(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        
        # dst node MLP
        self.dst_node_mlp = MLP_PROCESS(
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
                if self.attention:
                    attention_score = self.attention_score(graph, grid_input, mesh_nfeat, efeat)
                    cat_feat = aggregate_and_concat_with_attention(
                        efeat, mesh_nfeat, graph, attention_score
                    )
                else:  
                    cat_feat = aggregate_and_concat(
                        efeat, mesh_nfeat, graph, self.aggregation
                    )
                mesh_nfeat_new.append(mesh_nfeat + self.dst_node_mlp(cat_feat), graph)
                grid_nfeat_new.append(grid_input + self.src_node_mlp(grid_input), graph)

            return torch.stack(grid_nfeat_new), torch.stack(mesh_nfeat_new)
        else:
            efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
            if self.aggregation:
                attention_score = self.attention_score(graph, g2m_efeat, mesh_nfeat, efeat)
                cat_feat = aggregate_and_concat_with_attention(grid_nfeat, mesh_nfeat, graph, attention_score)
            else:
                cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
            mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat, graph)
            grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat, graph)
            return grid_nfeat, mesh_nfeat