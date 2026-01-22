from forecasting_models.pytorch.utils import *
from forecasting_models.pytorch.graphcast.graph_cast_net import *
from dgl.nn.pytorch.conv import GraphConv, GATConv
from torch_geometric.nn import GraphNorm, global_mean_pool, global_max_pool
from torch.nn import ReLU, GELU
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from blitz.losses import kl_divergence_from_nn
import dgl
import dgl.function as fn
import math
import numpy as np
import scipy.sparse

from forecasting_models.pytorch.tfn import TemporalFusionTransformerClassifier

# Monkeypatch for older scipy versions
if not hasattr(scipy.sparse, 'random_array'):
    def random_array_wrapper(*args, **kwargs):
        # Map 'data_sampler' to 'data_rvs' if present
        if 'data_sampler' in kwargs:
            kwargs['data_rvs'] = kwargs.pop('data_sampler')
        # Map 'rng' to 'random_state' if present
        if 'rng' in kwargs:
            kwargs['random_state'] = kwargs.pop('rng')
            
        # Handle shape argument: random_array(shape, ...) vs random(m, n, ...)
        if len(args) > 0:
            if isinstance(args[0], (tuple, list, np.ndarray)) and len(args[0]) == 2:
                shape = args[0]
                args = (shape[0], shape[1]) + args[1:]
                
        return scipy.sparse.random(*args, **kwargs)
    scipy.sparse.random_array = random_array_wrapper

from reservoirpy.nodes import Reservoir


import torch
import torch.nn as nn
import torch.nn.functional as F


##################################### SIMPLE GRAPH #####################################
class NetGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim_2, output_channels, end_channels, n_sequences, graph_or_node, device, task_type, return_hidden=False, horizon=0):
        super(NetGCN, self).__init__()
        self.layer1 = GraphConv(in_dim * n_sequences, hidden_dim).to(device)
        self.layer2 = GraphConv(hidden_dim, hidden_dim_2).to(device)
        self.is_graph_or_node = graph_or_node == 'graph'
        self.n_sequences = n_sequences
        self.device = device
        self.task_type = task_type
        self.soft = torch.nn.Softmax(dim=1)
        self.return_hidden = return_hidden
        self.horizon = horizon
        self.end_channels = end_channels

        self.output_layer = OutputLayer(
            in_channels=hidden_dim_2,
            end_channels=end_channels,
            n_steps=1,
            device=device,
            act_func='relu',
            task_type=task_type,
            out_channels=output_channels
        )

    def forward(self, features, g, z_prev=None):

        if z_prev is None:
            z_prev = torch.zeros((features.shape[0], self.end_channels, self.n_sequences), device=features.device, dtype=features.dtype)

        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)

        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)

        hidden = x
        logits = self.output_layer(hidden)
        output = logits

        return output, logits, hidden
    
class GAT(torch.nn.Module):
    def __init__(self, n_sequences, in_dim,
                 hidden_channels,
                 end_channels,
                 heads,
                 dropout,
                 bias,
                 device,
                 act_func,
                 task_type,
                 out_channels,
                 graph_or_node='node',
                 return_hidden=False,
                 horizon=0):

        super(GAT, self).__init__()

        num_of_layers = len(hidden_channels) - 1
        self.gat_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activation_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.return_hidden = return_hidden
        self.out_channels = out_channels
        self.horizon = horizon
        self.n_sequences = n_sequences
        self.end_channels = end_channels

        # Couche d'entrée linéaire pour projeter les dimensions d'entrée
        self.input = nn.Linear(in_channels=in_dim + (end_channels * n_sequences if horizon > 0 else 0), out_channels=hidden_channels[0] * heads[0], weight_initializer='glorot').to(device)

        for i in range(num_of_layers):
            concat = True if i < num_of_layers - 1 else False
            gat_layer = GATConv(
                in_feats=hidden_channels[i] * heads[i],
                out_feats=hidden_channels[i + 1],
                num_heads=heads[i + 1],
                concat=concat,
                dropout=dropout,
                bias=bias,
                add_self_loops=False,
            ).to(device)
            self.gat_layers.append(gat_layer)

            # Normalization layer
            norm_layer = nn.BatchNorm(hidden_channels[i + 1] * heads[i + 1] if concat else hidden_channels[i + 1]).to(device)
            self.norm_layers.append(norm_layer)

            # Activation layer
            if i < num_of_layers - 1:
                if act_func == 'relu':
                    self.activation_layers.append(ReLU())
                elif act_func == 'gelu':
                    self.activation_layers.append(GELU())
            else:
                self.activation_layers.append(None)

            # Dropout layer
            dropout_layer = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
            self.dropout_layers.append(dropout_layer)

        self.is_graph_or_node = graph_or_node == 'graph'

        layer_for_output = hidden_channels[-1]
        if graph_or_node == 'graph':
            layer_for_output *= 3

        self.output = OutputLayer(
            in_channels=layer_for_output,
            end_channels=end_channels,
            n_steps=n_sequences,
            device=device,
            act_func=act_func,
            task_type=task_type,
            out_channels=self.out_channels
        )

    def forward(self, X, edge_index, graphs=None, z_prev=None):
        edge_index = edge_index[:2]

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        X = X[:, :, -1]
        
        if self.horizon > 0:
            z_prev = z_prev.reshape(X.shape[0], -1)
            X = torch.cat((X, z_prev), dim=1)

        x = self.input(X)  # Projeter les dimensions d'entrée
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply GAT layer
            x = gat_layer(x, edge_index)

            # Apply normalization
            x = self.norm_layers[i](x)

            # Apply activation if available
            if self.activation_layers[i] is not None:
                x = self.activation_layers[i](x)

            # Apply dropout
            x = self.dropout_layers[i](x)

        # Graph pooling if needed
        if self.is_graph_or_node:
            x_mean = global_mean_pool(x, graphs)
            x_max = global_max_pool(x, graphs)
            x_add = global_add_pool(x, graphs)
            x = torch.cat([x_mean, x_max, x_add], dim=1)

        hidden = x
        logits = self.output(hidden)
        output = logits
        return output, logits, hidden



class LSTM_GNN_Feedback(torch.nn.Module):
    def __init__(
        self,
        lstm_hidden=64,
        gnn_hidden=64,
        end_channels=64,
        out_channels=1,
        n_sequences=1,
        act_func='relu',
        task_type='classification',
        device=None,
        static_idx=None,
        num_lstm_layers=1,
        temporal_idx=None,
        use_layernorm=False,
        dropout=0.03,
        return_hidden=False,
        horizon=0,
    ):
        super(LSTM_GNN_Feedback, self).__init__()

        self.lstm_hidden = lstm_hidden
        self.device = device
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx
        self.task_type = task_type
        self.return_hidden = return_hidden
        self.horizon = horizon
        self.end_channels = end_channels
        self.n_sequences = n_sequences

        # LSTMCell: traite séquentiellement les pas de temps
        self.lstm_cell = torch.nn.LSTMCell(
            input_size=len(temporal_idx) + gnn_hidden,
            hidden_size=lstm_hidden,
        )

        # Encodeur des données statiques
        self.static_encoder = torch.nn.Linear(len(static_idx), gnn_hidden)

        # GNN appliqué aux états LSTM
        self.gnn = GraphConv(lstm_hidden, gnn_hidden)
        
        # Dropout after process
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output linear layer
        self.output_layer = torch.nn.Linear(gnn_hidden + lstm_hidden, out_channels).to(device)

        # Activation function
        self.act_func = self.act_func = getattr(torch.nn, act_func)()

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gnn_hidden + lstm_hidden).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gnn_hidden + lstm_hidden).to(device)

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def separate_variables(self, x):
        is_static = (x == x[:, :, 0:1]).all(dim=2)
        static_mask = is_static.all(dim=0)
        static_idx = torch.where(static_mask)[0]
        temporal_idx = torch.where(~static_mask)[0]

        x_static = x[:, static_idx, 0].unsqueeze(-1)  # (B, S, 1)
        x_temporal = x[:, temporal_idx, :]            # (B, D, T)

        return x_static, x_temporal, static_idx, temporal_idx

    def forward(self, x, edge_index, z_prev=None):
        # x: (B, X, T)
        B, X, T = x.shape

        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)

        # Séparation des variables
        if self.static_idx is None:
            x_static, x_temporal, static_idx, temporal_idx = self.separate_variables(x)
        else:
            x_static = x[:, self.static_idx, 0].unsqueeze(-1)  # (B, S, 1)
            x_temporal = x[:, self.temporal_idx, :]            # (B, D, T)

        D_temporal = x_temporal.shape[1]
        D_static = x_static.shape[1]

        # --- Encodage statique ---
        static_input = x_static.squeeze(-1)  # (B, S)
        static_embed = self.static_encoder(static_input)  # (B, gnn_hidden)

        # --- Init LSTM ---
        h_t = torch.zeros(B, static_embed.size(1), device=x.device)
        c_t = torch.zeros_like(h_t)

        # --- Boucle temporelle avec feedback GNN ---
        for t in range(T):
            x_t = x_temporal[:, :, t]  # (B, D_temporal)

            gnn_out = self.gnn(h_t, edge_index)  # (B, gnn_hidden)

            lstm_input = torch.cat([x_t, gnn_out], dim=1)
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))

        # --- Fusion finale et prédiction ---
        fused = torch.cat([h_t, gnn_out], dim=1)
        x = self.dropout(fused)
        x = self.norm(x)

       # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden

class Sep_GRU_GNN(torch.nn.Module):
    def __init__(
        self,
        gru_hidden=64,
        gnn_hidden_list=[32, 64],
        lin_channels=64,
        end_channels=64,
        out_channels=1,
        n_sequences=1,
        task_type='classification',
        device=None,
        act_func='relu',
        static_idx=None,
        temporal_idx=None,
        num_lstm_layers=1,
        use_layernorm=False,
        dropout=0.03,
        horizon=0,
    ):
        super(Sep_GRU_GNN, self).__init__()

        self.gru_hidden = gru_hidden
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx
        self.is_graph_or_node = False
        self.device = device
        self.return_hidden = False
        self.horizon = horizon
        self.end_channels = end_channels
        self.n_sequences = n_sequences

        # LSTM
        input_size = len(temporal_idx)
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Multi-layer GCN
        self.gnn_layers = torch.nn.ModuleList()
        in_feats = len(static_idx)
        for out_feats in gnn_hidden_list:
            self.gnn_layers.append(GraphConv(in_feats, out_feats))
            in_feats = out_feats  # pour la prochaine couche

        self.gnn_output_dim = gnn_hidden_list[-1]

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        
        # Output linear layer
        print(f'Spatial {in_feats}')
        print(f'Temporal {input_size}')
        print(f'Sum {gru_hidden} + {self.gnn_output_dim}')
        self.linear1 = torch.nn.Linear(gru_hidden + self.gnn_output_dim, lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gru_hidden + self.gnn_output_dim).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gru_hidden + self.gnn_output_dim).to(device)
            
        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def separate_variables(self, x):
        # x: (B, X, T)
        is_static = (x == x[:, :, 0:1]).all(dim=2)
        static_mask = is_static.all(dim=0)
        static_idx = torch.where(static_mask)[0]
        temporal_idx = torch.where(~static_mask)[0]

        x_static = x[:, static_idx, 0].unsqueeze(-1)  # (B, S, 1)
        x_temporal = x[:, temporal_idx, :]            # (B, D, T)

        return x_static, x_temporal, static_idx, temporal_idx
    
    def forward(self, x, graph, z_prev=None):
        # x: (B, X, T)
        B, X, T = x.shape

        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)

        # Séparation statique/temporelle
        if self.static_idx is None:
            x_static, x_temporal, static_idx, temporal_idx = self.separate_variables(x)
        else:
            x_static = x[:, self.static_idx, 0].unsqueeze(-1)  # (B, S, 1)
            x_temporal = x[:, self.temporal_idx, :]            # (B, D, T)

        # --- LSTM ---
        D = x_temporal.shape[1]
        if D == 0:
            lstm_out = torch.zeros(B, self.lstm_hidden, device=x.device)
        else:
            x_lstm_input = x_temporal.permute(0, 2, 1)  # (B, T, D)
            lstm_out, _ = self.gru(x_lstm_input)       # (B, T, H)
            lstm_out = lstm_out[:, -1, :]               # (B, H)
            
        # --- GCN ---
        S = x_static.shape[1]
        if S == 0:
            gnn_out = torch.zeros(B, self.gnn_output_dim, device=x.device)
        else:
            h = x_static.squeeze(-1)  # (B, S)false
            for layer in self.gnn_layers:
                h = layer(graph, h)
                h = torch.relu(h)
            gnn_out = h               # (B, out_dim)

        # --- Fusion ---
        x = torch.cat([lstm_out, gnn_out], dim=1)  # (B, total)
        x = self.dropout(x)
        x = self.norm(x)
        
        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden


class Sep_LSTM_GNN(torch.nn.Module):
    def __init__(
        self,
        lstm_hidden=64,
        gnn_hidden_list=[32, 64],
        lin_channels=64,
        end_channels=64,
        out_channels=1,
        n_sequences=1,
        task_type='classification',
        device=None,
        act_func='relu',
        static_idx=None,
        temporal_idx=None,
        num_lstm_layers=1,
        use_layernorm=False,
        dropout=0.03,
        horizon=0,
    ):
        super(Sep_LSTM_GNN, self).__init__()

        self.lstm_hidden = lstm_hidden
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx
        self.is_graph_or_node = False
        self.device = device
        self.horizon = horizon
        self.end_channels = end_channels
        self.n_sequences = n_sequences

        # LSTM
        input_size = len(temporal_idx)
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Multi-layer GCN
        self.gnn_layers = torch.nn.ModuleList()
        in_feats = len(static_idx)
        for out_feats in gnn_hidden_list:
            self.gnn_layers.append(GraphConv(in_feats, out_feats))
            in_feats = out_feats  # pour la prochaine couche

        self.gnn_output_dim = gnn_hidden_list[-1]

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output linear layer
        print(f'Spatial {in_feats}')
        print(f'Temporal {input_size}')
        print(f'Sum {lstm_hidden} + {self.gnn_output_dim}')
        self.linear1 = torch.nn.Linear(lstm_hidden + self.gnn_output_dim, lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(lstm_hidden + self.gnn_output_dim).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(lstm_hidden + self.gnn_output_dim).to(device)
            
        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def separate_variables(self, x):
        # x: (B, X, T)
        is_static = (x == x[:, :, 0:1]).all(dim=2)
        static_mask = is_static.all(dim=0)
        static_idx = torch.where(static_mask)[0]
        temporal_idx = torch.where(~static_mask)[0]

        x_static = x[:, static_idx, 0].unsqueeze(-1)  # (B, S, 1)
        x_temporal = x[:, temporal_idx, :]            # (B, D, T)

        return x_static, x_temporal, static_idx, temporal_idx
    
    def forward(self, x, graph, z_prev=None):
        # x: (B, X, T)
        B, X, T = x.shape

        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)

        # Séparation statique/temporelle
        if self.static_idx is None:
            x_static, x_temporal, static_idx, temporal_idx = self.separate_variables(x)
        else:
            x_static = x[:, self.static_idx, 0].unsqueeze(-1)  # (B, S, 1)
            x_temporal = x[:, self.temporal_idx, :]            # (B, D, T)

        # --- LSTM ---
        D = x_temporal.shape[1]
        if D == 0:
            lstm_out = torch.zeros(B, self.lstm_hidden, device=x.device)
        else:
            x_lstm_input = x_temporal.permute(0, 2, 1)  # (B, T, D)
            lstm_out, _ = self.lstm(x_lstm_input)       # (B, T, H)
            lstm_out = lstm_out[:, -1, :]               # (B, H)
            
        # --- GCN ---
        S = x_static.shape[1]
        if S == 0:
            gnn_out = torch.zeros(B, self.gnn_output_dim, device=x.device)
        else:
            h = x_static.squeeze(-1)  # (B, S)
            for layer in self.gnn_layers:
                h = layer(graph, h)
                h = torch.relu(h)
            gnn_out = h               # (B, out_dim)

        # --- Fusion ---
        x = torch.cat([lstm_out, gnn_out], dim=1)  # (B, total)
        x = self.dropout(x)
        x = self.norm(x)
        
        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden

class ST_GATLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, end_channels, n_sequences, device, act_func, heads,
                 dropout, num_layers, task_type, concat, graph_or_node='node', return_hidden=False, out_channels=None, horizon=0):
        super(ST_GATLSTM, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = num_layers - 1
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        self.concat = concat
        self.horizon = horizon
        self.end_channels = end_channels

        # LSTM layers with different hidden channels per layer
        self.lstm_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.lstm_layers.append(torch.nn.LSTM(input_size=hidden_channels_list[i],
                                                   hidden_size=hidden_channels_list[i+1],
                                                   num_layers=1,  # Each layer is a single LSTM layer
                                                   dropout=dropout, batch_first=True).to(device))

        # GAT layer
        self.gat = GATConv(in_feats=hidden_channels_list[-1], out_feats=hidden_channels_list[-1],
                           num_heads=heads).to(device)

        self.graph_norm = torch.nn.BatchNorm1d(hidden_channels_list[-1]).to(device)

        # Adjusted output layer input size
        pooled_output_dim = hidden_channels_list[-1] * 3 if self.is_graph_or_node else hidden_channels_list[-1]
        pooled_output_dim = pooled_output_dim * heads if concat else pooled_output_dim
        self.output = OutputLayer(in_channels=pooled_output_dim, end_channels=end_channels,
                                  n_steps=n_sequences, device=device, act_func=act_func, 
                                  task_type=task_type, out_channels=out_channels)

    def forward(self, X, graphs, graphs_ids=None, z_prev=None):
        batch_size = X.size(0)

        # Rearrange dimensions for LSTM input
        x = X.permute(0, 2, 1)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        # Initialisation des états cachés et des cellules pour chaque couche
        h0 = torch.zeros(1, batch_size, self.hidden_channels_list[0]).to(self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_channels_list[0]).to(self.device)

        # Boucle sur les couches LSTM
        for i in range(self.num_layers):
            # Passer l'entrée à travers la couche LSTM
            x, (h0, c0) = self.lstm_layers[i](x, (h0, c0))

        # Extract the last output of LSTM
        x = x[:, -1, :]  # Shape: (batch_size, hidden_channels)

        # Apply Batch Normalization
        x = self.graph_norm(x)

        # Pass through GAT layer
        x = self.gat(graphs, x)
        
        if self.concat:
            x = x.view(x.shape[0], -1)
        else:
            x = torch.mean(x, dim=1)

        # Apply pooling if working with graph-level predictions
        if self.is_graph_or_node:
            x_mean = global_mean_pool(xg, graphs_ids)
            x_max = global_max_pool(xg, graphs_ids)
            x_sum = global_add_pool(xg, graphs_ids)
            xg = torch.cat([x_mean, x_max, x_sum], dim=1)

        # Final output
        hidden = x
        logits = self.output(hidden)
        output = logits

        return output, logits, hidden
    
class STGCN(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, hidden_channels, end_channels, dropout, act_func, device, task_type, out_channels, graph_or_node='node', return_hidden=False, horizon=0):
        super(STGCN, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        self.horizon = horizon
        self.end_channels = end_channels

        # Initial input projection layer
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels[0], kernel_size=1, device=device)

        # Sandwich layers
        self.layers = torch.nn.ModuleList()
        num_of_layers = len(hidden_channels) - 1
        for i in range(num_of_layers):
            self.layers.append(SandwichLayerGCN(n_sequences=n_sequences,
                                                in_channels=hidden_channels[i],
                                                out_channels=hidden_channels[i + 1],
                                                dropout=dropout,
                                                act_func=act_func,
                                                last=i==num_of_layers-1).to(device))
            
        # Output layer with concatenated pooling dimensions
        pooled_output_dim = hidden_channels[-1] * 3 * n_sequences if self.is_graph_or_node else hidden_channels[-1] * n_sequences

        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=1,
                                  device=device, act_func=act_func,
                                  task_type=task_type,
                                  out_channels=out_channels)

    def forward(self, X, graphs, graphs_id=None, z_prev=None):

        graphs = graphs.to(X.device)

        # Initial input projection
        x = self.input(X)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        # Apply each SandwichLayerGCN
        for layer in self.layers:
            x = layer(x, graphs)
            
        # If graph-level pooling is needed, apply all three poolings and concatenate
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs_id)
            x_max = global_max_pool(x, graphs_id)
            x_sum = global_add_pool(x, graphs_id)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)
            
        # Output layer
        hidden = x
        logits = self.output(hidden)
        output = logits
        return output, logits, hidden

class SandwichLayerGCN(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, out_channels, dropout, act_func, last):
        super(SandwichLayerGCN, self).__init__()

        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.gated_conv1 = Temporal_Gated_Conv(in_channels, out_channels, kernel_size=3)

        self.gcn = GraphConv(in_feats=out_channels * n_sequences, out_feats=out_channels * n_sequences)

        self.batch_norm = nn.BatchNorm(out_channels)
        self.batch_norm_1 = nn.BatchNorm(out_channels * n_sequences)
        self.batch_norm_2 = nn.BatchNorm(out_channels)

        self.gated_conv2 = Temporal_Gated_Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3)

        self.n_sequences = n_sequences
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        if act_func == 'gelu':
            self.activation = GELU()
        elif act_func == 'relu': 
            self.activation = ReLU()
        elif act_func == 'silu':
            self.activation = SiLU()

        self.last = last

    def forward(self, X, graphs):
        
        residual = self.residual_proj(X)

        x = self.gated_conv1(X)

        x = self.batch_norm(x)

        x = x.view(X.shape[0], self.n_sequences * x.shape[1])

        x = self.gcn(graphs, x)
        
        x = self.batch_norm_1(x)

        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)
        
        x = self.gated_conv2(x)

        x = self.batch_norm_2(x)

        x = x + residual

        if not self.last:
            x = self.activation(x)

        x = self.drop(x)
        
        return x

class STGAT(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, hidden_channels, end_channels, dropout, heads, act_func, device, task_type, out_channels, graph_or_node='node', return_hidden=False, horizon=0):
        super(STGAT, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.is_graph_or_node = graph_or_node == 'graph'
        self.n_sequences = n_sequences
        self.horizon = horizon
        self.end_channels = end_channels

        self.input = Conv1d(in_channels=in_channels, out_channels=hidden_channels[0], kernel_size=1, device=device)

        self.layers = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()

        num_of_layers = len(hidden_channels) - 1

        for i in range(num_of_layers):
            concat = True if i < num_of_layers - 1 else False
            self.layers.append(SandwichLayer(n_sequences=n_sequences,
                                             in_channels=hidden_channels[i],
                                             out_channels=hidden_channels[i+1],
                                             heads=heads[i],
                                             dropout=dropout,
                                             concat=concat).to(device))
 
        pooled_output_dim = hidden_channels[-1] * 3 * n_sequences if self.is_graph_or_node else hidden_channels[-1] * n_sequences
        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=1,
                                  device=device, act_func=act_func,
                                  task_type=task_type,
                                  out_channels=out_channels)
        
    def forward(self, X, graphs, graphs_id=None, z_prev=None):

        # Initial input projection
        x = self.input(X)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        # Apply each SandwichLayerGCN
        for layer in self.layers:
            x = layer(x, graphs)
            
        # If graph-level pooling is needed, apply all three poolings and concatenate
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs_id)
            x_max = global_max_pool(x, graphs_id)
            x_sum = global_add_pool(x, graphs_id)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)
            
        # Output layer
        hidden = x
        logits = self.output(hidden)
        output = logits
        return output, logits, hidden


class SandwichLayer(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, out_channels, dropout, heads, concat):
        super(SandwichLayer, self).__init__()

        self.concat = concat
        self.residual_proj = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        
        self.gated_conv1 = Temporal_Gated_Conv(in_channels, in_channels, kernel_size=3)
        self.gat = GATConv(in_feats=out_channels * n_sequences, out_feats=out_channels * n_sequences, num_heads=heads, feat_drop=0.0, attn_drop=0.0, activation=ReLU())

        self.gated_conv2 = Temporal_Gated_Conv(in_channels=out_channels * heads if concat else out_channels, out_channels=out_channels)
        
        self.batch_norm = nn.BatchNorm(out_channels)
        self.batch_norm_1 = nn.BatchNorm(out_channels * n_sequences * heads if concat else out_channels * n_sequences)
        self.batch_norm_2 = nn.BatchNorm(out_channels)
        
        self.n_sequences = n_sequences
        self.drop = Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.activation = ReLU()
        self.last = True

    def forward(self, X, graphs):

        residual = self.residual_proj(X)
        
        x = self.gated_conv1(X)

        x = self.batch_norm(x)
        
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])

        x = self.gat(graphs, x)

        if self.concat:
            x = x.view(x.shape[0], -1)
        else:
            x = torch.mean(x, dim=1)

        x = self.batch_norm_1(x)
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)
        
        x = self.gated_conv2(x)

        x = self.batch_norm_2(x)

        x = x + residual
        if self.last:
            x = self.activation(x)

        x = self.drop(x)
        
        return x

#####################################################



################################ GCN ################################################

class GCN(torch.nn.Module):
    def __init__(self, n_sequences, in_dim,
                 hidden_channels,
                 end_channels,
                 dropout,
                 bias,
                 device,
                 act_func,
                 task_type,
                 out_channels,
                 graph_or_node='node',
                 return_hidden=False,
                 horizon=0):
        
        super(GCN, self).__init__()
        
        num_of_layers = len(hidden_channels) - 1
        self.gcn_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activation_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.return_hidden = return_hidden
        self.out_channels = out_channels
        self.horizon = horizon
        self.n_sequences = n_sequences
        self.end_channels = end_channels

        self.input = nn.Linear(in_channels=in_dim + (end_channels * n_sequences if horizon > 0 else 0), out_channels=hidden_channels[0], weight_initializer='glorot').to(device)

        for i in range(num_of_layers):
            # GCN layer
            gcn_layer = GCNLayer(
                in_channels=hidden_channels[i],
                out_channels=hidden_channels[i + 1],
            ).to(device)
            self.gcn_layers.append(gcn_layer)

            # Normalization layer
            norm_layer = nn.BatchNorm(hidden_channels[i + 1]).to(device)
            self.norm_layers.append(norm_layer)

            # Activation layer (optional, only add if not last layer)
            if i < num_of_layers - 1:
                if act_func == 'relu':
                    self.activation_layers.append(ReLU())
                elif act_func == 'gelu':
                    self.activation_layers.append(GELU())
            else:
                # If no activation layer for the last layer, append None
                self.activation_layers.append(None)
            
            # Dropout layer
            dropout_layer = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
            self.dropout_layers.append(dropout_layer)

        self.is_graph_or_node = graph_or_node == 'graph'

        layer_for_output = hidden_channels[-1]
        if graph_or_node == 'graph':
            layer_for_output *= 3

        self.output = OutputLayer(
            in_channels=layer_for_output,
            end_channels=end_channels,
            n_steps=n_sequences,
            device=device,
            act_func=act_func,
            task_type=task_type,
            out_channels=self.out_channels
        )

    def forward(self, X, edge_index, graphs=None, z_prev=None):

        edge_index = edge_index[:2]

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        X = X[:, :, -1]

        if self.horizon > 0:
            z_prev = z_prev.reshape(X.shape[0], -1)
            X = torch.cat((X, z_prev), dim=1)

        x = self.input(X)
        for i, gcn_layer in enumerate(self.gcn_layers):
            # Apply GCN layer
            x = gcn_layer(x, edge_index)
            
            # Apply normalization
            x = self.norm_layers[i](x)
            
            # Apply activation if available
            if self.activation_layers[i] is not None:
                x = self.activation_layers[i](x)
            
            # Apply dropout
            x = self.dropout_layers[i](x)

        if self.is_graph_or_node:
            x_mean = global_mean_pool(x, graphs)
            x_max = global_max_pool(x, graphs)
            x_add = global_add_pool(x, graphs)
            x = torch.cat([x_mean, x_max, x_add], dim=1)

        hidden = x
        logits = self.output(hidden)
        output = logits
        return output, logits, hidden

################################### DSTGCN ######################################
    
# See code from https://github.com/SakastLord/STGAT and https://github.com/jswang/stgat_traffic_prediction
# https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py

class GatedDilatedConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(GatedDilatedConvolution, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=3, padding='same')
        self.conv2 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=3, padding='same')
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        
    def forward(self, X):
        x1 = self.tanh(self.conv1(X))
        x2 = self.sigmoid(self.conv2(X))
        x = torch.mul(x1, x2)
        return x

class SpatioTemporalLayer(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, out_channels, dilation, dropout, last):
        super(SpatioTemporalLayer, self).__init__()
        self.tcn = GatedDilatedConvolution(in_channels=in_channels, out_channels=out_channels, dilation=dilation)
        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.gcn = GraphConv(out_channels * n_sequences, out_channels * n_sequences)

        #self.graph_norm = nn.GraphNorm(in_channels=out_channels * n_sequences)
        self.batch_norm = torch.nn.BatchNorm1d(out_channels)
        self.batch_norm_2 = torch.nn.BatchNorm1d(out_channels)

        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.n_sequences = n_sequences
        self.activation = ReLU()
        self.last = last
    
    def forward(self, X, graphs):

        residual = self.residual_proj(X)

        x = self.tcn(X)

        x = self.batch_norm(x)

        x = x.view(X.shape[0], self.n_sequences * x.shape[1])

        x = self.gcn(graphs, x)

        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)

        x = self.batch_norm_2(x)

        x = x + residual

        if not self.last:
            x = self.activation(x)
        
        x = self.drop(x)

        return x

class DSTGCN(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, end_channels, dilation_channels, dilations, dropout, act_func, device, task_type, out_channels, graph_or_node='node', return_hidden=False, horizon=0):
        super(DSTGCN, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        self.out_channels = out_channels
        self.horizon = horizon
        self.end_channels = end_channels
        self.end_channels = end_channels
        
        # Initial layer
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=dilation_channels[0], kernel_size=1, device=device)
        
        # Spatio-temporal layers
        self.layers = torch.nn.ModuleList()
        num_of_layers = len(dilation_channels) - 1
        for i in range(num_of_layers):
            self.layers.append(SpatioTemporalLayer(n_sequences=n_sequences,
                                                   in_channels=dilation_channels[i],
                                                   out_channels=dilation_channels[i + 1],
                                                   dilation=dilations[i],
                                                   dropout=dropout,
                                                   last= i == num_of_layers - 1).to(device))
        
        # Output layer, adapted for graph pooling with concatenation (mean + max + sum pooling)
        pooled_output_dim = dilation_channels[-1] * 3 * n_sequences if self.is_graph_or_node else dilation_channels[-1] * n_sequences
        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device,
                                  act_func=act_func,
                                  task_type=task_type,
                                  out_channels=self.out_channels)
        
    def forward(self, X, graphs, graphs_id=None, z_prev=None):
        # Initial projection
        x = self.input(X)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        # Apply each SpatioTemporalLayer
        for layer in self.layers:
            x = layer(x, graphs)
        
        # Apply global pooling if graph-level representation is needed
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs_id)
            x_max = global_max_pool(x, graphs_id)
            x_sum = global_add_pool(x, graphs_id)
            
            # Concatenate pooled representations (mean, max, sum)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)

        # Final output layer
        hidden = x
        logits = self.output(hidden)
        output = logits
        return output, logits, hidden
        
################################################### DST GAT ##########################################################

class SpatioTemporalLayerGAT(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, out_channels, dilation, dropout, concat, last, heads):
        super(SpatioTemporalLayerGAT, self).__init__()
        
        # Temporal convolutional network part
        self.tcn = GatedDilatedConvolution(in_channels=in_channels, out_channels=out_channels, dilation=dilation)
        
        # Graph Attention Convolution (GAT) for spatial information
        factor = heads if concat else 1
        self.concat = concat
        self.heads = heads
        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels * factor, kernel_size=1)
        self.gcn = GATConv(in_feats=out_channels * n_sequences, out_feats=out_channels * n_sequences, num_heads=heads, feat_drop=0.0, attn_drop=0.0, activation=ReLU())
        
        # Batch normalization layers
        self.batch_norm = torch.nn.BatchNorm1d(out_channels)
        self.batch_norm_2 = torch.nn.BatchNorm1d(out_channels * factor)
        
        # Dropout layer
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        
        # Other properties
        self.n_sequences = n_sequences
        self.activation = ReLU()
        self.last = last

    def forward(self, X, graph):
        # Residual connection
        residual = self.residual_proj(X)
        
        # Apply temporal gated convolution
        x = self.tcn(X)
        x = self.batch_norm(x)
        
        # Flatten for GAT convolution and apply GAT
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gcn(graph, x)
        if self.concat:
            x = x.view(x.shape[0], -1)
        else:
            x = torch.mean(x, dim=1)

        # Reshape back and apply batch normalization
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)
        x = self.batch_norm_2(x)
        
        # Add residual connection
        x = x + residual
        if not self.last:
            x = self.activation(x)
        
        # Apply dropout
        x = self.drop(x)
        return x

class DSTGAT(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, end_channels, dilation_channels, dilations, dropout, act_func, device, task_type, heads, out_channels, graph_or_node='node', return_hidden=False, horizon=0):
        super(DSTGAT, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        self.out_channels = out_channels
        self.horizon = horizon
        
        # Initial layer
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=dilation_channels[0], kernel_size=1, device=device)

        input_channels = dilation_channels[0]
        # Spatio-temporal layers
        self.layers = torch.nn.ModuleList()
        num_of_layers = len(dilation_channels) - 1
        heads_mul = 1
        for i in range(num_of_layers):
            concat = True if i < num_of_layers - 1 else False
            factor = heads[i] if concat else 1
            self.layers.append(SpatioTemporalLayerGAT(n_sequences=n_sequences,
                                                   in_channels=input_channels,
                                                   out_channels=dilation_channels[i + 1],
                                                   dilation=dilations[i],
                                                   heads=heads[i],
                                                   dropout=dropout,
                                                   concat=concat,
                                                   last = i == num_of_layers - 1).to(device))
            if concat:
                input_channels = dilation_channels[i + 1] * factor
                heads_mul = heads_mul + heads[i]
                
        # Output layer, adapted for graph pooling with concatenation (mean + max + sum pooling)
        pooled_output_dim = dilation_channels[-1] * 3 * n_sequences if self.is_graph_or_node else dilation_channels[-1] * n_sequences

        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device,
                                  act_func=act_func,
                                  task_type=task_type,
                                  out_channels=self.out_channels)
        
    def forward(self, X, graph, graphs_id=None, z_prev=None):
        # Initial projection
        x = self.input(X)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        # Apply each SpatioTemporalLayer
        for layer in self.layers:
            x = layer(x, graph)
        
        # Apply global pooling if graph-level representation is needed
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs_id)
            x_max = global_max_pool(x, graphs_id)
            x_sum = global_add_pool(x, graphs_id)
            
            # Concatenate pooled representations (mean, max, sum)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)

        # Final output layer
        hidden = x
        logits = self.output(hidden)
        output = logits
        return output, logits, hidden

################################# ST-GATCONV ###################################
    
# See code https://github.com/hazdzz/STGCN/blob/main/model/layers.py

class Temporal_Gated_Conv(torch.nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(Temporal_Gated_Conv, self).__init__()
        self.conv_1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_2 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_3 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size * num_nodes, in_channels, input_time_steps).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size * num_nodes, in_channels, input_time_steps).
        """
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        return H
    
class ESN(torch.nn.Module):
    """
    Echo State Network (ESN) avec réservoir fixe + tête MLP entraînable,
    calqué sur l'API/structure de ta classe GRU.

    Entrée attendue: X de forme (batch, features, seq_len)
    - Si horizon > 0, concatène z_prev (état précédent de la tête) aux features (même logique que ta GRU).
    - Le réservoir est mis à jour itérativement sur la dimension temporelle.
    - La tête lit l'état final (ou pooling) et produit la prédiction.
    """
    def __init__(
        self,
        in_channels,
        reservoir_size,
        hidden_channels,
        end_channels,
        n_sequences,
        device,
        act_func='ReLU',
        task_type='regression',
        dropout=0.0,
        return_hidden=False,
        out_channels=None,
        use_layernorm=False,
        horizon=0,
        # Hyperparamètres ESN
        spectral_radius=0.9,
        sparsity=0.1,
        leak_rate=1.0,           # 1.0 => ESN classique (pas de fuite), <1.0 => leaky-ESN
        input_scale=1.0,
        bias_scale=0.0,
        reservoir_activation='Tanh',
        readout_from='last',     # 'last' ou 'mean' (pooling temporel)
        reservoir_noise_std=0.0,  # bruit optionnel ajouté à l'état
        temporal_idx = None,
        static_idx = None,
    ):
        super().__init__()

        # --------- méta ---------
        self.device = device
        self.return_hidden = return_hidden
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.reservoir_size = reservoir_size
        self.end_channels = end_channels
        self.n_sequences = n_sequences
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels
        self.readout_from = readout_from

        # --------- dimensions d'entrée du réservoir ---------
        self.input_dim = in_channels + (self.end_channels if horizon > 0 else 0)

        # --------- réservoir (poids fixes) ---------
        # W_in: (N, input_dim)
        Win = torch.randn(reservoir_size, self.input_dim) * input_scale
        # W_res: (N, N) sparse-ish, redimensionnée pour spectral_radius
        W = torch.zeros(reservoir_size, reservoir_size)
        mask = (torch.rand_like(W) < sparsity)
        W[mask] = torch.randn(mask.sum())
        # normalisation au rayon spectral
        # on approxime via norme spéctrale par itérations de puissance (quelques steps)
        with torch.no_grad():
            v = torch.randn(reservoir_size)
            for _ in range(10):
                v = torch.matmul(W, v)
                v = v / (v.norm() + 1e-8)
            lambda_max = torch.matmul(v, torch.matmul(W, v)) / (torch.matmul(v, v) + 1e-8)
            # si lambda_max ~ 0 (sparsité élevée), éviter /0
            scale = spectral_radius / (lambda_max.abs() + 1e-8)
            W = W * scale

        # b (biais du réservoir)
        b = torch.randn(reservoir_size) * bias_scale

        # On enregistre comme buffers (fixes, déplacés avec .to(device))
        self.register_buffer('W_in', Win)
        self.register_buffer('W_res', W)
        self.register_buffer('b_res', b)

        # --------- activation réservoir ---------
        self.reservoir_act = getattr(torch.nn, reservoir_activation)() if hasattr(torch.nn, reservoir_activation) else torch.nn.Tanh()
        self.leak_rate = leak_rate
        self.reservoir_noise_std = reservoir_noise_std

        # --------- normalisation & dropout (sur la lecture de l'état agrégé) ---------
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(reservoir_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(reservoir_size).to(device)

        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # --------- tête MLP + output ---------
        self.linear1 = torch.nn.Linear(reservoir_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        self.act_func = getattr(torch.nn, act_func)()

        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

        # On s'assure que seules les couches de tête sont entraînables (optionnel mais recommandé)
        for p in [self.W_in, self.W_res, self.b_res]:
            p.requires_grad = False
        # Les layers linéaires restent entraînables (default True)

    def _reservoir_step(self, h, x_t):
        """
        h: (B, N)  état courant
        x_t: (B, D) entrée au temps t (après concat éventuelle)
        retour: h_next (B, N)
        """
        # affinité
        pre = torch.matmul(x_t, self.W_in.T) + torch.matmul(h, self.W_res.T) + self.b_res
        h_tilde = self.reservoir_act(pre)
        # leaky update
        if self.leak_rate < 1.0:
            h_next = (1.0 - self.leak_rate) * h + self.leak_rate * h_tilde
        else:
            h_next = h_tilde
        if self.reservoir_noise_std > 0:
            h_next = h_next + self.reservoir_noise_std * torch.randn_like(h_next)
        return h_next

    def forward(self, X, edge_index=None, graphs=None, z_prev=None):
        """
        X: (batch, features, seq_len)
        Retour:
            output: prédiction finale
            logits: avant activation de sortie
            hidden: représentation (end_channels) juste avant la couche de sortie
        """
        B = X.size(0)
        # z_prev: même logique que ta GRU (injecter la dernière prédiction comme feature)
        if z_prev is None:
            z_prev = torch.zeros((B, self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(B, self.end_channels, self.n_sequences)

        if self.horizon > 0:
            X = torch.cat((X, z_prev), dim=1)  # (B, in_channels+end_channels, T)

        # permute -> (B, T, D)
        x_seq = X.permute(0, 2, 1).contiguous()
        T = x_seq.size(1)
        D = x_seq.size(2)

        # état réservoir initial
        h = torch.zeros(B, self.reservoir_size, device=self.device, dtype=X.dtype)

        # on peut accumuler les états si pooling 'mean'
        if self.readout_from == 'mean':
            h_acc = torch.zeros_like(h)

        # itération temporelle
        for t in range(T):
            h = self._reservoir_step(h, x_seq[:, t, :])
            if self.readout_from == 'mean':
                h_acc = h_acc + h

        if self.readout_from == 'mean':
            h_read = h_acc / max(T, 1)
        else:
            h_read = h  # dernier état

        # normalisation + dropout (attention BatchNorm1d attend (B, C))
        h_norm = self.norm(h_read)
        h_drop = self.dropout(h_norm)

        # tête MLP + sortie
        x = self.act_func(self.linear1(h_drop))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)

        return output, logits, hidden