import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

from forecasting_models.pytorch.utils import *
from forecasting_models.pytorch.graphcast.graph_cast_net import *
from dgl.nn.pytorch.conv import GraphConv, GATConv
from torch_geometric.nn import GraphNorm, global_mean_pool, global_max_pool
from torch.nn import ReLU, GELU
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import dgl
import dgl.function as fn
import torch
import math

##################################### SIMPLE GRAPH #####################################
class NetGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim_2, output_channels, end_channels, n_sequences, graph_or_node, device, task_type, return_hidden=False):
        super(NetGCN, self).__init__()
        self.layer1 = GraphConv(in_dim * n_sequences, hidden_dim).to(device)
        self.layer2 = GraphConv(hidden_dim, hidden_dim_2).to(device)
        self.is_graph_or_node = graph_or_node == 'graph'
        self.n_sequences = n_sequences
        self.device = device
        self.task_type = task_type
        self.soft = torch.nn.Softmax(dim=1)
        self.return_hidden = return_hidden

        self.output_layer = OutputLayer(
            in_channels=hidden_dim_2,
            end_channels=end_channels,
            n_steps=1,
            device=device,
            act_func='relu',
            task_type=task_type,
            out_channels=output_channels
        )

    def forward(self, features, g):

        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)

        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)

        hidden = x
        output = self.output_layer(hidden)

        if self.return_hidden:
            return output, hidden
        else:
            return output
    
class MLPLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, device):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(in_feats, hidden_dim, weight_initializer='glorot', bias=True, bias_initializer='zeros').to(device)
        #self.mlp = torch.nn.Linear(in_feats, hidden_dim).to(device)
    def forward(self, x):
        return self.mlp(x)
    
class NetMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, end_channels, output_channels, n_sequences, device, task_type, return_hidden=False):
        super(NetMLP, self).__init__()
        self.layer1 = MLPLayer(in_dim * n_sequences, hidden_dim, device)
        self.layer3 = MLPLayer(hidden_dim, hidden_dim, device)
        self.layer4 = MLPLayer(hidden_dim, end_channels, device)
        self.layer2 = MLPLayer(end_channels, output_channels, device)
        self.task_type = task_type
        self.n_sequences = n_sequences
        self.soft = torch.nn.Softmax(dim=1)
        self.return_hidden = return_hidden

    def forward(self, features, edges=None):
        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)
        x = F.relu(self.layer1(features))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        hidden = self.layer2(x)
        output = hidden
        if self.task_type == 'classification':
            output = self.soft(hidden)
        if self.return_hidden:
            return output, hidden
        else:
            return output
    
#####################################################""""

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
                 return_hidden=False):
        
        super(GAT, self).__init__()

        num_of_layers = len(hidden_channels) - 1
        self.gat_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activation_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.return_hidden = return_hidden
        self.out_channels = out_channels

        # Couche d'entrée linéaire pour projeter les dimensions d'entrée
        self.input = nn.Linear(in_channels=in_dim, out_channels=hidden_channels[0] * heads[0], weight_initializer='glorot').to(device)

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

    def forward(self, X, edge_index, graphs=None):
        edge_index = edge_index[:2]
        
        X = X[:, :, -1]
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

        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output


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
                 return_hidden=False):
        
        super(GCN, self).__init__()
        
        num_of_layers = len(hidden_channels) - 1
        self.gcn_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activation_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.return_hidden = return_hidden
        self.out_channels = out_channels

        self.input = nn.Linear(in_channels=in_dim, out_channels=hidden_channels[0], weight_initializer='glorot').to(device)

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

    def forward(self, X, edge_index, graphs=None):

        edge_index = edge_index[:2]
        
        X = X[:, :, -1]

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

        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output

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
    def __init__(self, n_sequences, in_channels, end_channels, dilation_channels, dilations, dropout, act_func, device, task_type, out_channels, graph_or_node='node', return_hidden=False):
        super(DSTGCN, self).__init__()
        
        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        self.out_channels = out_channels
        
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
        
    def forward(self, X, graphs, graphs_id=None):
        # Initial projection
        x = self.input(X)

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
        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output
        
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
    def __init__(self, n_sequences, in_channels, end_channels, dilation_channels, dilations, dropout, act_func, device, task_type, heads, out_channels, graph_or_node='node', return_hidden=False):
        super(DSTGAT, self).__init__()
        
        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        self.out_channels = out_channels
        
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
        
    def forward(self, X, graph, graphs_id=None):
        # Initial projection
        x = self.input(X)

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
        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output

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

class STGAT(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, hidden_channels, end_channels, dropout, heads, act_func, device, task_type, out_channels, graph_or_node='node', return_hidden=False):
        super(STGAT, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.is_graph_or_node = graph_or_node == 'graph'
        self.n_sequences = n_sequences

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
        
    def forward(self, X, graphs, graphs_id=None):
        
        # Initial input projection
        x = self.input(X)

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
        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output

###################################### ST-GCN #####################################################

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

class STGCN(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, hidden_channels, end_channels, dropout, act_func, device, task_type, out_channels, graph_or_node='node', return_hidden=False):
        super(STGCN, self).__init__()
        
        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'

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

    def forward(self, X, graphs, graphs_id=None):

        graphs = graphs.to(X.device)

        # Initial input projection
        x = self.input(X)

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
        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output

################################### ST_LSTM ######################################

class ST_GATLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, end_channels, n_sequences, device, act_func, heads,
                 dropout, num_layers, task_type, concat, graph_or_node='node', return_hidden=False, out_channels=None):
        super(ST_GATLSTM, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = num_layers - 1
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        self.concat = concat

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

    def forward(self, X, graphs, graphs_ids=None):
        batch_size = X.size(0)

        # Rearrange dimensions for LSTM input
        x = X.permute(0, 2, 1)

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
        output = self.output(x)
        
        return (output, x) if self.return_hidden else output

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
    ):
        super(Sep_LSTM_GNN, self).__init__()

        self.lstm_hidden = lstm_hidden
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx
        self.is_graph_or_node = False
        self.device = device

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
            self.output_activation = torch.nn.Sigmoid().to(device)
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
    
    def forward(self, x, graph):
        # x: (B, X, T)
        B, X, T = x.shape

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
        x = self.output_layer(hidden)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, hidden
        else:
            return output

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
    ):
        super(Sep_GRU_GNN, self).__init__()

        self.gru_hidden = gru_hidden
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx
        self.is_graph_or_node = False
        self.device = device

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
            self.output_activation = torch.nn.Sigmoid().to(device)
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
    
    def forward(self, x, graph):
        # x: (B, X, T)
        B, X, T = x.shape

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
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        return output

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
    ):
        super(LSTM_GNN_Feedback, self).__init__()
        
        self.lstm_hidden = lstm_hidden
        self.device = device
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx
        self.task_type = task_type
        self.return_hidden = return_hidden

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
            self.output_activation = torch.nn.Sigmoid().to(device)
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

    def forward(self, x, edge_index):
        # x: (B, X, T)
        B, X, T = x.shape

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
        x = self.output_layer(hidden)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, hidden
        else:
            return output

class GRU(torch.nn.Module):
    def __init__(self, in_channels, gru_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.0, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False):
        super(GRU, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.gru_size = gru_size
        
        # GRU layer
        self.gru = torch.nn.GRU(
            input_size=in_channels,
            hidden_size=gru_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gru_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gru_size).to(device)

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output linear layer
        self.linear1 = torch.nn.Linear(gru_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, X, edge_index=None, graphs=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        batch_size = X.size(0)

        # Reshape to (batch, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.gru_size).to(self.device)

        # GRU forward
        x, _ = self.gru(x, h0)

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(hidden)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, hidden
        else:
            return output

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, lstm_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.03, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False):
        super(LSTM, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.lstm_size = lstm_size

        # LSTM block
        self.lstm = torch.nn.LSTM(
            input_size=in_channels,
            hidden_size=self.lstm_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(self.lstm_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(self.lstm_size).to(device)

        # Dropout after LSTM
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output layer
        self.linear1 = torch.nn.Linear(self.lstm_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def forward(self, X, edge_index=None, graphs=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        batch_size = X.size(0)

        # (batch_size, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)

        # LSTM forward
        x, _ = self.lstm(x, (h0, c0))

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, x
        else:
            return output
        
class DilatedCNN(torch.nn.Module):
    def __init__(self, channels, dilations, lin_channels, end_channels, n_sequences, device, act_func, dropout, out_channels, task_type, use_layernorm=False, return_hidden=False):
        super(DilatedCNN, self).__init__()

        # Initialisation des listes pour les convolutions et les BatchNorm
        self.cnn_layer_list = []
        self.batch_norm_list = []
        self.num_layer = len(channels) - 1
        
        # Initialisation des couches convolutives et BatchNorm
        for i in range(self.num_layer):
            self.cnn_layer_list.append(torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            if use_layernorm:
                self.batch_norm_list.append(torch.nn.LayerNorm(channels[i + 1]).to(device))
            else:
                self.batch_norm_list.append(torch.nn.BatchNorm1d(channels[i + 1]).to(device))

        self.dropout = torch.nn.Dropout(dropout)
        
        # Convertir les listes en ModuleList pour être compatible avec PyTorch
        self.cnn_layer_list = torch.nn.ModuleList(self.cnn_layer_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        
        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output layer
        self.linear1 = torch.nn.Linear(channels[-1], lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()
        
        self.return_hidden = return_hidden

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, x, edges=None):
        # Couche d'entrée

        # Couches convolutives dilatées avec BatchNorm, activation et dropout
        for cnn_layer, batch_norm in zip(self.cnn_layer_list, self.batch_norm_list):
            x = cnn_layer(x)
            x = batch_norm(x)  # Batch Normalization
            x = self.act_func(x)
            x = self.dropout(x)
        
        # Garder uniquement le dernier élément des séquences
        x = x[:, :, -1]

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, x
        else:
            return output
        
class GraphCast(torch.nn.Module):
    def __init__(self,
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        end_channels = 64,
        lin_channels = 64,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        out_channels = 4,
        task_type = 'classification',
        do_concat_trick: bool = False,
        has_time_dim: bool = False,
        n_sequences = 1,
        act_func='ReLU',
        is_graph_or_node=False,
        return_hidden=False):
        super(GraphCast, self).__init__()

        self.net = GraphCastNet(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim_grid_nodes=output_dim_grid_nodes,
            processor_layers=processor_layers,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            has_time_dim=has_time_dim)
        
        # Output layer
        self.linear1 = torch.nn.Linear(output_dim_grid_nodes, lin_channels)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels)
        self.output_layer = torch.nn.Linear(end_channels, out_channels)
        
        self.is_graph_or_node = is_graph_or_node == 'graph'
        
        self.act_func = getattr(torch.nn, act_func)()
        self.return_hidden = return_hidden
        
        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()  # For regression or custom handling

    def forward(self, X, graph, graph2mesh, mesh2graph):
        #X = X.view(X.shape[0], -1)
        #print(X.device)
        #print(X.shape)
        X = X.permute(2, 0, 1)
        x = self.net(X, graph, graph2mesh, mesh2graph)[-1]
        
        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        x = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        x = self.output_layer(x)
        output = self.output_activation(x)
        return output

class GraphCastGRU(torch.nn.Module):
    def __init__(
        self,
        *,
        # --- GRU specific parameters ---
        in_channels: int = 16,
        num_gru_layers: int = 1,
        # --- GraphCast parameters (unchanged) ---
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        end_channels: int = 64,
        lin_channels: int = 64,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        out_channels: int = 4,
        task_type: str = "classification",
        do_concat_trick: bool = False,
        has_time_dim: bool = False,
        n_sequences: int = 1,
        act_func: str = "ReLU",
        is_graph_or_node: bool = False,
        return_hidden: bool = False,
    ):
        """GraphCast‐based model preceded by a GRU that encodes the temporal dimension.

        Args:
            in_channels: Dimension of the temporal features fed to the GRU (== input_size).
            num_gru_layers: Number of stacked GRU layers.
            input_dim_grid_nodes: Size of the embedding produced by the GRU for each node.  Must
                match *hidden_size* of the GRU.
            All other parameters are identical to the original GraphCastGRU.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # GRU — encodes the temporal axis and outputs an embedding per node
        # ------------------------------------------------------------------
        self.gru = torch.nn.GRU(
            input_size=in_channels,
            hidden_size=input_dim_grid_nodes,
            num_layers=num_gru_layers,
            dropout=0.03 if num_gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.gru_size = input_dim_grid_nodes
        self.num_gru_layers = num_gru_layers

        # ------------------------------------------------------------------
        # GraphCast core network (unchanged)
        # ------------------------------------------------------------------
        self.net = GraphCastNet(
            input_dim_grid_nodes,
            input_dim_mesh_nodes,
            input_dim_edges,
            output_dim_grid_nodes,
            processor_layers,
            hidden_layers,
            hidden_dim,
            aggregation,
            norm_type,
            do_concat_trick,
            has_time_dim,
        )

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        self.linear1 = torch.nn.Linear(output_dim_grid_nodes, lin_channels)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels)
        self.output_layer = torch.nn.Linear(end_channels, out_channels)

        self.is_graph_or_node = is_graph_or_node == "graph"

        self.act_func = getattr(torch.nn, act_func)()
        self.return_hidden = return_hidden

        if task_type == "classification":
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == "binary":
            self.output_activation = torch.nn.Sigmoid()
        else:  # regression or custom
            self.output_activation = torch.nn.Identity()

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(self, X, graph, graph2mesh, mesh2graph):
        """Args:
            X: Tensor shaped (batch, seq_len, in_channels, n_nodes).
        """
        # Bring node dimension next to batch for GRU: (batch * n_nodes, seq_len, in_channels)
        B, C_in, T = X.shape
        X_for_gru = X.permute(0, 2, 1)
        h0 = torch.zeros(self.num_gru_layers, B, self.gru_size).to(X.device)

        gru_out, _ = self.gru(X_for_gru, h0)  # shape: (B*N, T, hidden)
        # Keep the last hidden state for each sequence
        gru_last = gru_out[:, -1, :]  # (B*N, hidden == input_dim_grid_nodes)
        X_graphcast = gru_last[None,: ,:]

        # GraphCast processing
        x = self.net(X_graphcast, graph, graph2mesh, mesh2graph)[-1]

        # Head
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        x = self.output_layer(hidden)
        output = self.output_activation(x)
        if self.return_hidden:
            return output, hidden
        else:
            return output

"""class MultiScaleGraph(torch.nn.Module):
    def __init__(self, input_channels, graph_input_channels, graph_output_channels, device, graph_or_node, task_type,
                 num_output_scale=1, num_sequence=1, out_channels=5):
        
        super(MultiScaleGraph, self).__init__()
        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device=device

        ### Embedding Layer
        self.embedding = torch.nn.Linear(input_channels * num_sequence, graph_input_channels).to(device)

        ### One GCN per scale
        self.gcn_layers = torch.nn.ModuleList([
            GraphConv(graph_input_channels, graph_output_channels).to(device) for _ in range(num_output_scale)
        ])

        ### Encoder: upscale to next scale (Linear for feature transfer)
        self.encoder = torch.nn.Linear(graph_input_channels, graph_input_channels).to(device)

        ### Decoder: downscale from higher scale
        self.decoder = torch.nn.Linear(graph_output_channels, graph_output_channels).to(device)

        ### Output layers (per scale)
        self.output_heads = torch.nn.ModuleList([
            nn.Linear(graph_output_channels * 2, out_channels).to(device) for _ in range(num_output_scale)
        ])

    def forward(self, X, graph_scale_list: list, increase_scale: list, decrease_scale: list):

        #print(graph_scale_list)
        #print(decrease_scale)
        #print(increase_scale)
        assert self.num_output_scale == len(graph_scale_list)
        batch_size = X.shape[0]
        
        X = X.view(batch_size, -1)

        ### Step 1: Embed input features at scale 0
        features_per_scale = [self.embedding(X)]  # scale 0 only

        ### Step 2: Encode to upper scales using increase_scale
        for i in range(self.num_output_scale - 1):
            g = increase_scale[i].to(X.device)
            src_feat = features_per_scale[i]
            g.srcdata["h"] = self.encoder(src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            features_per_scale.append(dst_feat)

        ### Step 3: Apply GCN at each scale
        gcn_outputs = []
        for i in range(self.num_output_scale):
            g = graph_scale_list[i].to(X.device)
            h = self.gcn_layers[i](g, features_per_scale[i])
            gcn_outputs.append(h)

        ### Step 4: Decode from top to bottom using decrease_scale

        decoded_features = [gcn_outputs[-1]]  # start from top
        for i in range(self.num_output_scale - 1):
            g = decrease_scale[i].to(X.device)
            src_feat = decoded_features[0]
            g.srcdata["h"] = self.decoder(src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            decoded_features.insert(0, dst_feat)

        ### Step 5: Final prediction per scale
        outputs = []
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            outputs.append(F.softmax(logits, dim=-1))

        outputs = torch.cat(outputs, dim=0)
        return outputs"""

class MultiScaleGraph(torch.nn.Module):
    def __init__(
        self, input_channels, features_per_scale, device, num_output_scale,
        graph_or_node, task_type, num_sequence=1, out_channels=5, return_hidden=False
    ):
        super(MultiScaleGraph, self).__init__()
        
        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device = device
        self.features_per_scale = features_per_scale
        self.return_hidden = return_hidden

        ### Embedding Layer (for scale 0 only)
        self.embedding = nn.Linear(input_channels * num_sequence, features_per_scale[0]).to(device)

        ### GCN layers (one per scale)
        self.gcn_layers = torch.nn.ModuleList([
            GraphConv(features_per_scale[i], features_per_scale[i]).to(device)
            for i in range(self.num_output_scale)
        ])

        ### Encoder: separate encoder per scale (except top scale)
        self.encoders = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i], features_per_scale[i + 1]).to(device)
            for i in range(self.num_output_scale - 1)
        ])

        ### Decoder: separate decoder per scale (except top scale)
        self.decoders = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i + 1], features_per_scale[i]).to(device)
            for i in reversed(range(self.num_output_scale - 1))
        ])

        ### Output heads: one per scale
        self.output_heads = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i] * 2, out_channels).to(device)
            for i in range(self.num_output_scale)
        ])

    def forward(self, X, graph_scale_list: list, increase_scale: list, decrease_scale: list):
        assert self.num_output_scale == len(graph_scale_list)
        batch_size = X.shape[0]

        X = X.view(batch_size, -1)

        ### Step 1: Embed input at scale 0
        features_per_scale = [self.embedding(X)]

        ### Step 2: Encode to higher scales
        for i, g in enumerate(increase_scale):
            g = g.to(X.device)
            src_feat = features_per_scale[i]
            g.srcdata["h"] = self.encoders[i](src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            features_per_scale.append(dst_feat)

        ### Step 3: GCN at each scale
        gcn_outputs = []
        for i, g in enumerate(graph_scale_list):
            g = g.to(X.device)
            h = self.gcn_layers[i](g, features_per_scale[i])
            gcn_outputs.append(h)

        ### Step 4: Decode from top to bottom
        decoded_features = [gcn_outputs[-1]]
        for i, g in zip(reversed(range(self.num_output_scale - 1)), decrease_scale):
            g = g.to(X.device)
            src_feat = decoded_features[0]
            g.srcdata["h"] = self.decoders[self.num_output_scale - 2 - i](src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            decoded_features.insert(0, dst_feat)

        ### Step 5: Final prediction at each scale
        outputs = []
        hidden = None
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            outputs.append(F.softmax(logits, dim=-1))
            if i == self.num_output_scale - 1:
                hidden = combined

        outputs = torch.cat(outputs, dim=0)
        if self.return_hidden:
            return outputs, hidden
        else:
            return outputs

class CrossScaleAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(CrossScaleAttention, self).__init__()
        self.query_proj = nn.Linear(output_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.num_heads = num_heads
        self.scale = (output_dim // num_heads) ** 0.5

    def forward(self, g, src_feat, dst_feat):
        Q = self.query_proj(dst_feat)  # (N_dst, d)
        K = self.key_proj(src_feat)    # (N_src, d)
        V = self.value_proj(src_feat)  # (N_src, d)

        g.srcdata["K"] = K
        g.srcdata["V"] = V
        g.dstdata["Q"] = Q

        def message_func(edges):
            score = (edges.dst["Q"] * edges.src["K"]).sum(dim=-1) / self.scale
            return {"score": score, "V": edges.src["V"]}

        def reduce_func(nodes):
            attn = F.softmax(nodes.mailbox["score"], dim=1)  # (N_dst, num_neighbors)
            V = nodes.mailbox["V"]  # (N_dst, num_neighbors, d)
            out = (attn.unsqueeze(-1) * V).sum(dim=1)  # (N_dst, d)
            return {"h_dst": out}

        g.apply_edges(message_func)
        g.update_all(message_func, reduce_func)
        return g.dstdata["h_dst"]


class MultiScaleAttentionGraph(torch.nn.Module):
    def __init__(
        self, input_channels, features_per_scale, device, num_output_scale,
        graph_or_node='graph', task_type='classification',
        num_sequence=1, out_channels=5, num_heads=4, return_hidden=False
    ):
        super(MultiScaleAttentionGraph, self).__init__()

        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device = device
        self.features_per_scale = features_per_scale
        self.return_hidden = return_hidden

        # Embedding Layer for scale 0
        self.embedding = nn.Linear(input_channels * num_sequence, features_per_scale[0]).to(device)

        # GCN layers per scale
        self.gcn_layers = torch.nn.ModuleList([
            GraphConv(features_per_scale[i], features_per_scale[i]).to(device)
            for i in range(self.num_output_scale)
        ])

        # Attention Encoders between scales
        self.encoder_attn_blocks = torch.nn.ModuleList([
            CrossScaleAttention(features_per_scale[i], features_per_scale[i + 1], num_heads).to(device)
            for i in range(self.num_output_scale - 1)
        ])

        # Attention Decoders between scales (reverse order)
        self.decoder_attn_blocks = torch.nn.ModuleList([
            CrossScaleAttention(features_per_scale[i + 1], features_per_scale[i], num_heads).to(device)
            for i in reversed(range(self.num_output_scale - 1))
        ])

        # Output layers per scale
        self.output_heads = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i] * 2, out_channels).to(device)
            for i in range(self.num_output_scale)
        ])

    def forward(self, X, graph_scale_list, increase_scale, decrease_scale):
        assert self.num_output_scale == len(graph_scale_list)
        batch_size = X.shape[0]

        X = X.view(batch_size, -1)

        # Step 1: Initial embedding (scale 0)
        features_per_scale = [self.embedding(X)]

        # Step 2: Encoding to upper scales with attention
        for i, g in enumerate(increase_scale):
            g = g.to(X.device)
            src_feat = features_per_scale[i]
            dst_feat = torch.zeros(g.num_dst_nodes(), self.features_per_scale[i + 1], device=X.device)
            encoded = self.encoder_attn_blocks[i](g, src_feat, dst_feat)
            features_per_scale.append(encoded)

        # Step 3: GCN on each scale
        gcn_outputs = []
        for i, g in enumerate(graph_scale_list):
            g = g.to(X.device)
            h = self.gcn_layers[i](g, features_per_scale[i])
            gcn_outputs.append(h)

        # Step 4: Decoding back down with attention
        decoded_features = [gcn_outputs[-1]]
        for i, g in zip(reversed(range(self.num_output_scale - 1)), decrease_scale):
            g = g.to(X.device)
            src_feat = decoded_features[0]
            dst_feat = torch.zeros(g.num_dst_nodes(), self.features_per_scale[i], device=X.device)
            decoded = self.decoder_attn_blocks[self.num_output_scale - 2 - i](g, src_feat, dst_feat)
            decoded_features.insert(0, decoded)

        # Step 5: Output heads for predictions
        outputs = []
        hidden = None
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            outputs.append(F.softmax(logits, dim=-1))
            if i == self.num_output_scale - 1:
                hidden = combined

        outputs = torch.cat(outputs, dim=0)
        if self.return_hidden:
            return outputs, hidden
        else:
            return outputs
    
##############################################################################################################
#                                                                                                            #
#                                       TRANSFORMER                                                          #
#                                                                                                            #
##############################################################################################################

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int = 256, dropout: float = 0.1, max_len: int = 30):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerNet(torch.nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
            self,
            seq_len=30,
            input_dim=24,
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            num_layers=4,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
            channel_attention=False,
            graph_or_node='node',
            out_channels=2,
            task_type='binary',
            return_hidden=False,
    ):

        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"
        
        self.graph_or_node = graph_or_node == 'graph'

        # self.emb = torch.nn.Embedding(input_dim, d_model)
        self.channel_attention = channel_attention

        self.lin_time = torch.nn.Linear(input_dim, d_model)
        self.lin_channel = torch.nn.Linear(seq_len, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer_time = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_time = torch.nn.TransformerEncoder(
            encoder_layer_time,
            num_layers=num_layers,
        )

        encoder_layer_channel = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_channel = torch.nn.TransformerEncoder(
            encoder_layer_channel,
            num_layers=num_layers,
        )

        self.out_time = torch.nn.Linear(d_model, d_model)
        self.out_channel = torch.nn.Linear(d_model, d_model)

        self.lin = torch.nn.Linear(d_model * 2, 2)

        if self.channel_attention:
            self.classifier = torch.nn.Linear(d_model * 2, out_channels)
        else:
            self.classifier = torch.nn.Linear(d_model, out_channels)

        self.d_model = d_model
        self.return_hidden = return_hidden

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()  # For regression or custom handling

    def resh(self, x, y):
        return x.unsqueeze(1).expand(y.size(0), -1)

    def forward(self, x_, edge_index=None):
        x_ = x_.permute(2, 0, 1)
        x = torch.tanh(self.lin_time(x_))
        x = self.pos_encoder(x)
        x = self.transformer_encoder_time(x)
        x = x[0, :, :]

        if self.channel_attention:
            y = torch.transpose(x_, 0, 2)
            y = torch.tanh(self.lin_channel(y))
            y = self.transformer_encoder_channel(y)

            x = torch.tanh(self.out_time(x))
            y = torch.tanh(self.out_channel(y[0, :, :]))

            h = self.lin(torch.cat([x, y], dim=1))

            m = torch.nn.Softmax(dim=1)
            g = m(h)

            g1 = g[:, 0]
            g2 = g[:, 1]

            x = torch.cat([self.resh(g1, x) * x, self.resh(g2, x) * y], dim=1)

        hidden = self.classifier(x)
        output = self.output_activation(hidden)
        if self.return_hidden:
            return output, hidden
        else:
            return output




@variational_estimator
class BayesianMLP(torch.nn.Module):
    """Minimal Bayesian MLP implemented with blitz."""
    def __init__(self, in_dim, hidden_dim, out_channels, task_type='regression',
                 device='cpu', graph_or_node='node', return_hidden=False):
        super().__init__()
        self.fc1 = BayesianLinear(in_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, out_channels)
        self.to(device)

        self.graph_or_node = graph_or_node
        self.return_hidden = return_hidden

        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()

    def forward(self, x, edge_index=None):
        x = torch.relu(self.fc1(x))
        hidden = self.fc2(x)
        output = self.output_activation(hidden)
        if self.return_hidden:
            return output, hidden
        return output

    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss()
