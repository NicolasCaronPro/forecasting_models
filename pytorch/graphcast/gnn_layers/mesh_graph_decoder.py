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

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from .mesh_graph_mlp import MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum, MeshGraphMLP, MeshGraphMLPAttention, MeshGraphMLPGAT
from .utils import aggregate_and_concat, aggregate_and_concat_with_attention, EdgeScoreDotProductGAT, EdgeScoreDotProductTransformer


class MeshGraphDecoder(nn.Module):
    """Decoder used e.g. in GraphCast
       which acts on the bipartite graph connecting a mesh
       (e.g. representing a latent space) to a mostly regular
       grid (e.g. representing the output domain).

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
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        attention: str = None
    ):
        super().__init__()
        self.aggregation = aggregation

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        
        if attention == "Transformer":
            MLP_PROCESS = MeshGraphMLPAttention
        elif attention == "GAT":
            MLP_PROCESS = MeshGraphMLPGAT
        else:
            MLP_PROCESS = MeshGraphMLP
            
        MLP_PROCESS = MeshGraphMLP

        self.attention = attention

        if attention == "GAT":
            self.attention_score = EdgeScoreDotProductGAT(input_dim_src_nodes, input_dim_dst_nodes, output_dim_edges, num_heads=4, head_dim=32)
        elif attention == "Transformer":
            self.attention_score = EdgeScoreDotProductTransformer(input_dim_src_nodes, input_dim_dst_nodes, output_dim_edges, num_heads=4, head_dim=32)

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

        # dst node MLP
        self.node_mlp = MLP_PROCESS(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        if attention == "GAT":
            self.node_mlp.op = 'Decoder'

    @torch.jit.ignore()
    def forward(
        self,
        m2g_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: DGLGraph,
    ) -> Tensor:
        has_time_dim = len(mesh_nfeat.shape) == 3
        if has_time_dim:
            timesteps = mesh_nfeat.size(0)
            grid_nfeat_new = []
            for i in range(timesteps):
                # update edge features
                efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat[i], grid_nfeat[i]), graph)
                # aggregate messages (edge features) to obtain updated node features
                if self.attention:
                    attention_score = self.attention_score(graph, mesh_nfeat[i], grid_nfeat[i], efeat)
                    cat_feat = aggregate_and_concat_with_attention(
                        efeat, grid_nfeat[i], graph, attention_score
                    )
                else:
                    cat_feat = aggregate_and_concat(
                        efeat, grid_nfeat[i], graph, self.aggregation
                    )
                # transformation and residual connection
<<<<<<< HEAD
                #if self.attention == 'GAT':
                if False:
                    inputs = (mesh_nfeat[i], cat_feat)
                else:
                    inputs = cat_feat
                    
=======
                if self.attention == 'GAT':
                    inputs = (mesh_nfeat[i], cat_feat)
                else:
                    inputs = cat_feat
>>>>>>> 8e0e38145e8fcf485b70ae715672f2544c9b71aa
                grid_nfeat_new.append(self.node_mlp(inputs, graph) + grid_nfeat[i])
            return torch.stack(grid_nfeat_new)
        
        else:
            # update edge features
            efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)
            # aggregate messages (edge features) to obtain updated node features
            
            if self.attention:
                attention_score = self.attention_score(graph, m2g_efeat, grid_nfeat, efeat)
                cat_feat = aggregate_and_concat_with_attention(efeat, grid_nfeat, graph, attention_score)
            else:
                cat_feat = aggregate_and_concat(efeat, grid_nfeat, graph, self.aggregation)
    
            # transformation and residual connection
<<<<<<< HEAD
            #if self.attention == 'GAT':
            if False:
                inputs = (cat_feat, grid_nfeat[i])
            else:
                inputs = cat_feat
                
            dst_feat = self.node_mlp(inputs, graph) + grid_nfeat
            
=======
            if self.attention == 'GAT':
                inputs = (cat_feat, grid_nfeat[i])
            else:
                inputs = grid_nfeat
            dst_feat = self.node_mlp(inputs, graph) + grid_nfeat
>>>>>>> 8e0e38145e8fcf485b70ae715672f2544c9b71aa
            return dst_feat