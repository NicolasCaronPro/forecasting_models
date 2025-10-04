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

from typing import Any, Callable, Dict, Tuple, Union

import dgl.function as fn
import torch
from dgl import DGLGraph
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from dgl.nn.functional import edge_softmax
import torch.nn as nn

import torch
import torch.nn as nn
import dgl.function as fn
from dgl import DGLGraph

# ---------------------------
# 1) Scores de type Transformer
# ---------------------------
class EdgeScoreDotProductTransformer(nn.Module):
    def __init__(self, input_dim_node_src: int, input_dim_node_dst, input_dim_edge: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.H, self.Dh = num_heads, head_dim
        self.Wq = nn.Linear(input_dim_node_src,  num_heads * head_dim, bias=False)
        self.Wk = nn.Linear(input_dim_node_dst,  num_heads * head_dim, bias=False)
        self.Be = nn.Linear(input_dim_edge, num_heads, bias=False) if input_dim_edge > 0 else None  # biais par arête
        self.scale = head_dim ** 0.5

    def forward(self, g: DGLGraph, h_src: torch.Tensor,
                h_dst: torch.Tensor, efeat: torch.Tensor) -> torch.Tensor:
        """
        Retourne des scores bruts (E, H) = (u·v / sqrt(Dh)) + biais d'arête (optionnel)
        """
        with g.local_scope():
            Q = self.Wq(h_src).view(-1, self.H, self.Dh)   # (N_src, H, Dh)
            K = self.Wk(h_dst).view(-1, self.H, self.Dh)   # (N_dst, H, Dh)
            g.srcdata['Q'] = Q
            g.dstdata['K'] = K
            g.apply_edges(fn.u_dot_v('Q', 'K', 'score'))   # 'score': (E, H)
            score = g.edata['score'] / self.scale          # (E, H)
            if self.Be is not None and efeat is not None:
                score = score + self.Be(efeat)             # (E, H)
            return score


# ---------------------------
# 2) Scores de type GAT
# ---------------------------
class EdgeScoreDotProductGAT(nn.Module):
    def __init__(self, input_dim_node_src: int, input_dim_node_dst, input_dim_edge: int, num_heads: int = 4, head_dim: int = 32, negative_slope: float = 0.2):
        super().__init__()
        self.H, self.Dh = num_heads, head_dim
        self.Wq = nn.Linear(input_dim_node_src, num_heads * head_dim, bias=False)
        self.Wk = nn.Linear(input_dim_node_dst, num_heads * head_dim, bias=False)

        self.a = nn.Parameter(torch.empty((num_heads, 2 * head_dim)))
        nn.init.xavier_uniform_(self.Wq.weight, gain=1.414)
        nn.init.xavier_uniform_(self.Wk.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a,        gain=1.414)
        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, g: DGLGraph, h_src: torch.Tensor,
                h_dst: torch.Tensor, efeat: torch.Tensor = None) -> torch.Tensor:
        """
        Retourne des scores bruts e_ij^(h) de GAT (E, H) = LeakyReLU(a_l^T z_i + a_r^T z_j)
        (Pas de biais d'arête par défaut ; ajouter si souhaité via une autre tête/MLP.)
        """
        with g.local_scope():
            # 1) Projections par tête
            z_src = self.Wq(h_src).view(-1, self.H, self.Dh)   # (N_src, H, Dh)
            z_dst = self.Wk(h_dst).view(-1, self.H, self.Dh)   # (N_dst, H, Dh)

            # 2) Séparer a = [a_l | a_r]
            a_l, a_r = self.a[:, :self.Dh], self.a[:, self.Dh:]  # (H, Dh), (H, Dh)

            # 3) Demi-scores par nœud
            el = (z_src * a_l.unsqueeze(0)).sum(dim=-1)  # (N_src, H)
            er = (z_dst * a_r.unsqueeze(0)).sum(dim=-1)  # (N_dst, H)

            # 4) Combinaison par arête et activation
            g.srcdata['el'] = el
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))        # (E, H)
            e = self.leakyrelu(g.edata['e'])                  # (E, H)
            return e
            
def checkpoint_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
    """Applies the identity function for checkpointing.

    This function serves as an identity function for use with model layers
    when checkpointing is not enabled. It simply forwards the input arguments
    to the specified layer and returns its output.

    Parameters
    ----------
    layer : Callable
        The model layer or function to apply to the input arguments.
    *args
        Positional arguments to be passed to the layer.
    **kwargs
        Keyword arguments to be passed to the layer.

    Returns
    -------
    Any
        The output of the specified layer after processing the input arguments.
    """
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: bool) -> Callable:
    """Sets checkpoint function.

    This function returns the appropriate checkpoint function based on the
    provided `do_checkpointing` flag. If `do_checkpointing` is True, the
    function returns the checkpoint function from PyTorch's
    `torch.utils.checkpoint`. Otherwise, it returns an identity function
    that simply passes the inputs through the given layer.

    Parameters
    ----------
    do_checkpointing : bool
        Whether to use checkpointing for gradient computation. Checkpointing
        can reduce memory usage during backpropagation at the cost of
        increased computation time.

    Returns
    -------
    Callable
        The selected checkpoint function to use for gradient computation.
    """
    if do_checkpointing:
        return checkpoint
    else:
        return checkpoint_identity


def concat_message_function(edges: Tensor) -> Dict[str, Tensor]:
    """Concatenates source node, destination node, and edge features.

    Parameters
    ----------
    edges : Tensor
        Edges.

    Returns
    -------
    Dict[Tensor]
        Concatenated source node, destination node, and edge features.
    """
    # concats src node , dst node, and edge features
    cat_feat = torch.cat((edges.data["x"], edges.src["x"], edges.dst["x"]), dim=1)
    return {"cat_feat": cat_feat}


@torch.jit.ignore()
def concat_efeat_dgl(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[torch.Tensor, torch.Tensor]],
    graph: DGLGraph,
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor, Tensor]
        Node features.
    graph : DGLGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    if isinstance(nfeat, Tuple):
        src_feat, dst_feat = nfeat
        with graph.local_scope():
            graph.srcdata["x"] = src_feat
            graph.dstdata["x"] = dst_feat
            graph.edata["x"] = efeat
            graph.apply_edges(concat_message_function)
            return graph.edata["cat_feat"]

    with graph.local_scope():
        graph.ndata["x"] = nfeat
        graph.edata["x"] = efeat
        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


def concat_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: DGLGraph,
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features.
    graph : DGLGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    if isinstance(nfeat, Tensor):
        efeat = concat_efeat_dgl(efeat, nfeat, graph)
    else:
        src_feat, dst_feat = nfeat
        # update edge features through concatenating edge and node features
        efeat = concat_efeat_dgl(efeat, (src_feat, dst_feat), graph)
    return efeat

@torch.jit.script
def sum_efeat_dgl(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
) -> Tensor:
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    src_feat : Tensor
        Source node features.
    dst_feat : Tensor
        Destination node features.
    src_idx : Tensor
        Source node indices.
    dst_idx : Tensor
        Destination node indices.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """

    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def sum_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: DGLGraph,
):
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features (static setting) or tuple of node features of
        source and destination nodes (bipartite setting).
    graph : DGLGraph
        The underlying graph.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """
    if isinstance(nfeat, Tensor):
        src_feat, dst_feat = nfeat, nfeat
        src, dst = (item.long() for item in graph.edges())
        sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)
    else:
        src_feat, dst_feat = nfeat
        src, dst = (item.long() for item in graph.edges())
        sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

    return sum_efeat


@torch.jit.ignore()
def agg_concat_dgl(
    efeat: Tensor, dst_nfeat: Tensor, graph: DGLGraph, aggregation: str
) -> Tensor:
    """Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """
    with graph.local_scope():
        # populate features on graph edges
        graph.edata["x"] = efeat

        # aggregate edge features
        if aggregation == "sum":
            graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
        elif aggregation == "mean":
            graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
        else:
            raise RuntimeError("Not a valid aggregation!")

        # concat dst-node & edge features
        cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), -1)
        return cat_feat


def aggregate_and_concat(
    efeat: Tensor,
    nfeat: Tensor,
    graph: DGLGraph,
    aggregation: str,
):
    """
    Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """

    cat_feat = agg_concat_dgl(efeat, nfeat, graph, aggregation)

    return cat_feat

@torch.jit.ignore()
def aggregate_and_concat_with_attention(
    efeat: Tensor,
    dst_nfeat: Tensor,
    graph: DGLGraph,
    edge_scores: Tensor,
    norm_by: str = "dst",
) -> Tensor:
    """
    Version attention : normalise des scores par arête via edge_softmax, pèse
    les messages d'arête, agrège (somme) puis concatène avec dst_nfeat.

    Paramètres
    ----------
    efeat : Tensor
        Features des arêtes, shape (E, De).
    dst_nfeat : Tensor
        Features des nœuds destination, shape (N_dst, Dd).
    graph : DGLGraph
        Graphe bipartite/grid->mesh.
    edge_scores : Tensor
        Scores d'attention par arête avant softmax. Shape (E,) ou (E, H).
        Si (E, H), on fait une attention multi-têtes et on agrège par somme
        sur H après pondération.
    norm_by : str
        "dst" (par défaut) ou "src" pour la normalisation dans edge_softmax.

    Retour
    ------
    Tensor
        cat_feat de shape (N_dst, De + Dd) si (E,) ; ou (N_dst, De + Dd) après
        somme des têtes si (E, H).
    """
    with graph.local_scope():
        # 1) Normalisation softmax sur les arêtes (par dst par défaut)
        #    edge_softmax accepte (E,) -> (E,1) ou (E,H).
        if edge_scores.dim() == 1:
            edge_scores = edge_scores.unsqueeze(-1)  # (E, 1)

        alpha = edge_softmax(graph, edge_scores, norm_by=norm_by)  # (E, 1) ou (E, H)

        # 2) Pondération des messages d'arête
        if alpha.size(-1) == 1:
            graph.edata["m"] = efeat * alpha  # (E, De)
        else:
            # multi-head : on étend efeat sur H, pèse, puis somme sur H
            # efeat_h : (E, H, De), m : (E, H, De) -> sum over H -> (E, De)
            efeat_h = efeat.unsqueeze(1).expand(-1, alpha.size(-1), -1)
            m = efeat_h * alpha.unsqueeze(-1)
            graph.edata["m"] = m.sum(dim=1)

        # 3) Agrégation vers les nœuds destinataires
        graph.update_all(fn.copy_e("m", "m"), fn.sum("m", "h_dest"))

        # 4) Concat avec l'état courant des nœuds dst
        cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), dim=-1)
        return cat_feat
    
@torch.jit.ignore()
def aggregate_and_concat_with_attention(
    efeat: Tensor,
    dst_nfeat: Tensor,
    graph: DGLGraph,
    edge_scores: Tensor,
    norm_by: str = "dst",
) -> Tensor:
    """
    Version attention : normalise des scores par arête via edge_softmax, pèse
    les messages d'arête, agrège (somme) puis concatène avec dst_nfeat.

    Paramètres
    ----------
    efeat : Tensor
        Features des arêtes, shape (E, De).
    dst_nfeat : Tensor
        Features des nœuds destination, shape (N_dst, Dd).
    graph : DGLGraph
        Graphe bipartite/grid->mesh.
    edge_scores : Tensor
        Scores d'attention par arête avant softmax. Shape (E,) ou (E, H).
        Si (E, H), on fait une attention multi-têtes et on agrège par somme
        sur H après pondération.
    norm_by : str
        "dst" (par défaut) ou "src" pour la normalisation dans edge_softmax.

    Retour
    ------
    Tensor
        cat_feat de shape (N_dst, De + Dd) si (E,) ; ou (N_dst, De + Dd) après
        somme des têtes si (E, H).
    """
    with graph.local_scope():
        # 1) Normalisation softmax sur les arêtes (par dst par défaut)
        #    edge_softmax accepte (E,) -> (E,1) ou (E,H).
        if edge_scores.dim() == 1:
            edge_scores = edge_scores.unsqueeze(-1)  # (E, 1)

        alpha = edge_softmax(graph, edge_scores, norm_by=norm_by)  # (E, 1) ou (E, H)

        # 2) Pondération des messages d'arête
        if alpha.size(-1) == 1:
            graph.edata["m"] = efeat * alpha  # (E, De)
        else:
            # multi-head : on étend efeat sur H, pèse, puis somme sur H
            # efeat_h : (E, H, De), m : (E, H, De) -> sum over H -> (E, De)
            efeat_h = efeat.unsqueeze(1).expand(-1, alpha.size(-1), -1)
            m = efeat_h * alpha.unsqueeze(-1)
            graph.edata["m"] = m.sum(dim=1)

        # 3) Agrégation vers les nœuds destinataires
        graph.update_all(fn.copy_e("m", "m"), fn.sum("m", "h_dest"))

        # 4) Concat avec l'état courant des nœuds dst
        cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), dim=-1)
        return cat_feat