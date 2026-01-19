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

from typing import Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv
from torch import Tensor

try:
    from apex.normalization import FusedLayerNorm

    apex_imported = True
except ImportError:
    apex_imported = False

from .utils import concat_efeat, sum_efeat

import logging

logger = logging.getLogger(__name__)

class MeshGraphMLP(nn.Module):
    """MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes. It
    consists of a number of linear layers followed by an activation
    and a norm layer following the last linear layer.

    Parameters
    ----------
    input_dim : int
        dimensionality of the input features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : Union[int, None], optional
        number of hidden layers, by default 1
        if None is provided, the MLP will collapse to a Identity function
    activation_fn : nn.Module, optional
        , by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Union[int, None] = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()

        if hidden_layers is not None:
            layers = [nn.Linear(input_dim, hidden_dim), copy.deepcopy(activation_fn)]
            self.hidden_layers = hidden_layers
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), copy.deepcopy(activation_fn)]
            layers.append(nn.Linear(hidden_dim, output_dim))

            self.norm_type = norm_type
            if norm_type is not None:
                if norm_type not in [
                    "LayerNorm",
                    "GraphNorm",
                    "InstanceNorm",
                    "BatchNorm",
                    "MessageNorm",
                ]:
                    raise ValueError(norm_type)
                if norm_type == "LayerNorm" and apex_imported:
                    norm_layer = FusedLayerNorm
                    logger.info("Found apex, using FusedLayerNorm")
                else:
                    norm_layer = getattr(nn, norm_type)
                layers.append(norm_layer(output_dim))

            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

    def forward(self, x: Tensor, graph=None) -> Tensor:
        self.model = self.model.to(x)
        return self.model(x)
    
class MeshGraphMLPAttention(MeshGraphMLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Union[int, None] = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        nhead: int = 4
    ):
        super(MeshGraphMLPAttention, self).__init__(
            input_dim,
            output_dim,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
        )
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)

        self.attention_layer = nn.MultiheadAttention(input_dim, nhead, dropout=0.03, batch_first=True)

    def forward(self, x: Tensor, graph=None) -> Tensor:
        q = self.q_proj(self.norm1(x)).unsqueeze(0)  # (1, L, E)
        k = self.k_proj(self.norm1(x)).unsqueeze(0)  # (1, L, E)
        v = self.v_proj(self.norm1(x)).unsqueeze(0)  # (1, L, E)

        attention, _ = self.attention_layer(q, k, v)      # (1, L, E)
        attention = attention.squeeze(0)   

        attention = attention + x
        self.model = self.model.to(attention)
        return self.model(x)
    
class MeshGraphMLPGAT(MeshGraphMLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Union[int, None] = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        nhead: int = 4,
        aggregate: str = "concat",   # <-- par défaut 'concat' pour matcher l'entrée MLP
        op: str = 'Encoder'
    ):
        self.op = op
        # L’entrée de l’MLP dépend de l’agrégation choisie
        mlp_in = input_dim * nhead if aggregate == "concat" else input_dim
        super().__init__(
            mlp_in,
            output_dim,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
        )

        self.input_dim  = input_dim
        self.nhead      = nhead
        self.aggregate  = aggregate

        self.norm1 = nn.LayerNorm(input_dim)

        self.proj_grid = nn.Linear(input_dim // 2, input_dim) 

        # GAT : sortie par tête = input_dim -> shape (N, H, input_dim)
        self.attention_layer = GATConv(
            in_feats=input_dim,
            out_feats=input_dim,
            num_heads=nhead,
            allow_zero_in_degree=True
        )
        # Skip/residual adaptés à l’agrégation
        self.residual_cat  = nn.Linear(input_dim, input_dim * nhead)

    def _aggregate_heads(self, att: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        att: (N, H, D) avec D = input_dim (sortie de GATConv)
        x  : (N, D)      features d'entrée
        """
        if self.aggregate == "concat":
            # (N, H*D)
            att_agg = att.flatten(1, 2)
            att_agg = att_agg + self.residual_cat(x)          # skip compatible
        elif self.aggregate == "mean":
            # (N, D)
            att_agg = att.mean(dim=1)
            att_agg = att_agg + x
        elif self.aggregate == "sum":
            # (N, D)
            att_agg = att.sum(dim=1)
            att_agg = att_agg + x
        elif self.aggregate == "max":
            # (N, D)
            att_agg, _ = att.max(dim=1)
            att_agg = att_agg + x
        else:
            raise ValueError(f"aggregate must be one of ['concat','mean','sum','max'], got {self.aggregate}")
        return att_agg

    def forward(self, x: torch.Tensor, graph: DGLGraph) -> torch.Tensor:
        # Normalisation (optionnelle mais utile pour la stabilité)
        # GAT : (N, H, D) avec D = input_dim
        x = (self.proj_grid(x[0]), x[1])
        
        att = self.attention_layer(graph, x)

        # Agrégation des têtes + skip connection
        att_agg = self._aggregate_heads(att, x[1])

        # S'assurer que le MLP est sur le bon device
        self.model = self.model.to(att_agg.device)

        # On envoie l'attention agrégée dans l'MLP
        return self.model(att_agg)

class MeshGraphEdgeMLPConcat(MeshGraphMLP):
    """MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes. It
    consists of a number of linear layers followed by an activation
    and a norm layer following the last linear layer. It first
    concatenates the input edge features and the node features of the
    corresponding source and destination nodes of the corresponding edge
    to create new edge features. These then are transformed through the
    transformations mentioned above.

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        efeat_dim: int = 512,
        src_dim: int = 512,
        dst_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        cat_dim = efeat_dim + src_dim + dst_dim
        super(MeshGraphEdgeMLPConcat, self).__init__(
            cat_dim,
            output_dim,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: DGLGraph,
    ) -> Tensor:
        self.model = self.model.to(efeat.device)
        efeat = concat_efeat(efeat, nfeat, graph)
        efeat = self.model(efeat)
        return efeat

class MeshGraphEdgeMLPSum(nn.Module):
    """MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes. It
    consists of a number of linear layers followed by an activation
    and a norm layer following the last linear layer. It transform
    edge features - which originally are intended to be a concatenation
    of previous edge features, and the node features of the corresponding
    source and destinationn nodes - by transorming these three features
    individually through separate linear transformations and then sums
    them for each edge accordingly. The result of this is transformed
    through the remaining linear layers and activation or norm functions.

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    bias : bool, optional
        whether to use bias in the MLP, by default True
    """

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
    ):
        super().__init__()

        self.efeat_dim = efeat_dim
        self.src_dim = src_dim
        self.dst_dim = dst_dim

        # this should ensure the same sequence of initializations
        # as the original MLP-Layer in combination with a concat operation
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim, bias=bias)
        # orig_weight has shape (hidden_dim, efeat_dim + src_dim + dst_dim)
        orig_weight = tmp_lin.weight
        w_efeat, w_src, w_dst = torch.split(
            orig_weight, [efeat_dim, src_dim, dst_dim], dim=1
        )
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)

        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None

        layers = [copy.deepcopy(activation_fn)]
        self.hidden_layers = hidden_layers
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), copy.deepcopy(activation_fn)]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.norm_type = norm_type
        if norm_type is not None:
            if norm_type not in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]:
                raise ValueError(norm_type)
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

    def forward_truncated_sum(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: DGLGraph,
    ) -> Tensor:
        """forward pass of the truncated MLP. This uses separate linear layers without
        bias. Bias is added to one MLP, as we sum afterwards. This adds the bias to the
         total sum, too. Having it in one F.linear should allow a fusion of the bias
         addition while avoiding adding the bias to the "edge-level" result.
        """
        if isinstance(nfeat, Tensor):
            src_feat, dst_feat = nfeat, nfeat
        else:
            src_feat, dst_feat = nfeat
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = sum_efeat(mlp_efeat, (mlp_src, mlp_dst), graph)
        return mlp_sum

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor]],
        graph: DGLGraph,
    ) -> Tensor:
        """Default forward pass of the truncated MLP."""
        mlp_sum = self.forward_truncated_sum(
            efeat,
            nfeat,
            graph,
        )
        self.model = self.model.to(efeat.device)
        return self.model(mlp_sum)
