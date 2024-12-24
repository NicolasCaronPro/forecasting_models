import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

from forecasting_models.pytorch.utils import *

################################ GAT ###########################################
import torch
from torch_geometric.nn import GATConv, GraphNorm, global_mean_pool, global_max_pool
from torch.nn import ReLU, GELU

class GAT(torch.nn.Module):
    def __init__(self, n_sequences, in_dim,
                 hidden_channels,
                 end_channels,
                 heads,
                 dropout, 
                 bias,
                 device,
                 act_func,
                 binary,
                 graph_or_node='node',
                 return_hidden=False):
        
        super(GAT, self).__init__()

        num_of_layers = len(hidden_channels) - 1
        self.gat_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activation_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.return_hidden = return_hidden

        # Couche d'entrée linéaire pour projeter les dimensions d'entrée
        self.input = torch.nn.Linear(in_features=in_dim, out_features=hidden_channels[0] * heads[0]).to(device)

        for i in range(num_of_layers):
            concat = True if i < num_of_layers - 1 else False
            gat_layer = GATConv(
                in_channels=hidden_channels[i] * heads[i],
                out_channels=hidden_channels[i + 1],
                heads=heads[i + 1],
                concat=concat,
                dropout=dropout,
                bias=bias,
                add_self_loops=False,
            ).to(device)
            self.gat_layers.append(gat_layer)

            # Normalization layer
            #norm_layer = GraphNorm(hidden_channels[i + 1] * heads[i + 1] if concat else hidden_channels[i + 1]).to(device)
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
            binary=binary
        )

    def forward(self, X, edge_index, graphs=None):
        edge_index = edge_index[:2]
        
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
                 binary,
                 graph_or_node='node',
                 return_hidden=False):
        
        super(GCN, self).__init__()
        
        num_of_layers = len(hidden_channels) - 1
        self.gcn_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activation_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.return_hidden = return_hidden

        self.input = torch.nn.Linear(in_features=in_dim, out_features=hidden_channels[0]).to(device)

        for i in range(num_of_layers):
            # GCN layer
            gcn_layer = GCNConv(
                in_channels=hidden_channels[i],
                out_channels=hidden_channels[i + 1],
                bias=bias,
            ).to(device)
            self.gcn_layers.append(gcn_layer)

            # Normalization layer
            #norm_layer = nn.GraphNorm(hidden_channels[i + 1]).to(device)
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
            binary=binary
        )

    def forward(self, X, edge_index, graphs=None):

        edge_index = edge_index[:2]

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
        self.gcn = GCNConv(in_channels=out_channels * n_sequences, out_channels=out_channels * n_sequences)

        #self.graph_norm = nn.GraphNorm(in_channels=out_channels * n_sequences)
        self.batch_norm = torch.nn.BatchNorm1d(out_channels)
        self.batch_norm_2 = torch.nn.BatchNorm1d(out_channels)

        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.n_sequences = n_sequences
        self.activation = ReLU()
        self.last = last
    
    def forward(self, X, edge_index, graphs):

        residual = self.residual_proj(X)

        x = self.tcn(X)

        x = self.batch_norm(x)

        x = x.view(X.shape[0], self.n_sequences * x.shape[1])

        x = self.gcn(x, edge_index)

        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)

        x = self.batch_norm_2(x)

        x = x + residual

        if not self.last:
            x = self.activation(x)
        
        x = self.drop(x)

        return x

class DSTGCN(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, end_channels, dilation_channels, dilations, dropout, act_func, device, binary, graph_or_node='node', return_hidden=False):
        super(DSTGCN, self).__init__()
        
        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        
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
        pooled_output_dim = dilation_channels[-1] * 3 if self.is_graph_or_node else dilation_channels[-1]
        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device,
                                  act_func=act_func,
                                  binary=binary)
        
    def forward(self, X, edge_index, graphs=None):
        # Initial projection
        x = self.input(X)

        # Apply each SpatioTemporalLayer
        for layer in self.layers:
            x = layer(x, edge_index, graphs)
        
        # Apply global pooling if graph-level representation is needed
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs)
            x_max = global_max_pool(x, graphs)
            x_sum = global_add_pool(x, graphs)
            
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
        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels * factor, kernel_size=1)
        self.gcn = GATConv(in_channels=out_channels * n_sequences, out_channels=out_channels * n_sequences, heads=heads, concat=concat, add_self_loops=False)
        
        # Batch normalization layers
        self.batch_norm = torch.nn.BatchNorm1d(out_channels)
        self.batch_norm_2 = torch.nn.BatchNorm1d(out_channels * factor)
        
        # Dropout layer
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        
        # Other properties
        self.n_sequences = n_sequences
        self.activation = ReLU()
        self.last = last

    def forward(self, X, edge_index, graphs):
        # Residual connection

        residual = self.residual_proj(X)
        
        # Apply temporal gated convolution
        x = self.tcn(X)
        x = self.batch_norm(x)
        
        # Flatten for GAT convolution and apply GAT
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gcn(x, edge_index)
        
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
    def __init__(self, n_sequences, in_channels, end_channels, dilation_channels, dilations, dropout, act_func, device, binary, heads, graph_or_node='node', return_hidden=False):
        super(DSTGAT, self).__init__()
        
        self.return_hidden = return_hidden
        self.device = device
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'
        
        # Initial layer
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=dilation_channels[0], kernel_size=1, device=device)

        input_channels = dilation_channels[0]
        # Spatio-temporal layers
        self.layers = torch.nn.ModuleList()
        num_of_layers = len(dilation_channels) - 1
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
        # Output layer, adapted for graph pooling with concatenation (mean + max + sum pooling)
        pooled_output_dim = dilation_channels[-1] * 3 if self.is_graph_or_node else dilation_channels[-1]
        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device,
                                  act_func=act_func,
                                  binary=binary)
        
    def forward(self, X, edge_index, graphs=None):
        # Initial projection
        x = self.input(X)

        # Apply each SpatioTemporalLayer
        for layer in self.layers:
            x = layer(x, edge_index, graphs)
        
        # Apply global pooling if graph-level representation is needed
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs)
            x_max = global_max_pool(x, graphs)
            x_sum = global_add_pool(x, graphs)
            
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
    def __init__(self, n_sequences, in_channels, out_channels, dropout, heads, concat, last):
        super(SandwichLayer, self).__init__()

        self.concat = concat
        self.residual_proj = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        
        self.gated_conv1 = Temporal_Gated_Conv(in_channels, in_channels, kernel_size=3)
        
        self.gat = GATConv(in_channels=in_channels * n_sequences,
                           out_channels=out_channels * n_sequences,
                           heads=heads, concat=concat,
                           add_self_loops=False,
                           dropout=dropout)
        
        self.gated_conv2 = Temporal_Gated_Conv(in_channels=out_channels * heads if concat else out_channels, out_channels=out_channels)
        
        self.batch_norm = nn.BatchNorm(out_channels)
        self.batch_norm_1 = nn.BatchNorm(out_channels * n_sequences * heads if concat else out_channels * n_sequences)
        self.batch_norm_2 = nn.BatchNorm(out_channels)
        
        self.n_sequences = n_sequences
        self.drop = Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.activation = ReLU()
        self.last = True

    def forward(self, X, edge_index, graphs=None):
        residual = self.residual_proj(X)
        
        x = self.gated_conv1(X)

        x = self.batch_norm(x)
        
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])

        x = self.gat(x, edge_index)

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
    def __init__(self, n_sequences, in_channels, hidden_channels, end_channels, dropout, heads, act_func, device, binary, graph_or_node='node', return_hidden=False):
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
                                             heads=heads,
                                             dropout=dropout,
                                             concat=concat).to(device))
 
        pooled_output_dim = hidden_channels[-1] * 3 if self.is_graph_or_node else hidden_channels[-1]
        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=1,
                                  device=device, act_func=act_func,
                                  binary=binary)
        
    def forward(self, X, edge_index, graphs=None):
        
        # Initial input projection
        x = self.input(X)

        # Apply each SandwichLayerGCN
        for layer in self.layers:
            x = layer(x, edge_index, graphs)
            
        # If graph-level pooling is needed, apply all three poolings and concatenate
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs)
            x_max = global_max_pool(x, graphs)
            x_sum = global_add_pool(x, graphs)
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

        self.gcn = GCNConv(in_channels=out_channels * n_sequences, out_channels=out_channels * n_sequences)

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

    def forward(self, X, edge_index, graph=None):
        
        residual = self.residual_proj(X)

        x = self.gated_conv1(X)

        x = self.batch_norm(x)

        x = x.view(X.shape[0], self.n_sequences * x.shape[1])

        x = self.gcn(x, edge_index)
        
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
    def __init__(self, n_sequences, in_channels, hidden_channels, end_channels, dropout, act_func, device, binary, graph_or_node='node', return_hidden=False):
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
        pooled_output_dim = hidden_channels[-1] * 3 if self.is_graph_or_node else hidden_channels[-1]

        self.output = OutputLayer(in_channels=pooled_output_dim,
                                  end_channels=end_channels,
                                  n_steps=1,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index, graphs=None):

        # Initial input projection
        x = self.input(X)

        # Apply each SandwichLayerGCN
        for layer in self.layers:
            x = layer(x, edge_index, graphs)
            
        # If graph-level pooling is needed, apply all three poolings and concatenate
        if self.is_graph_or_node:
            x = x[:, :, -1]
            x_mean = global_mean_pool(x, graphs)
            x_max = global_max_pool(x, graphs)
            x_sum = global_add_pool(x, graphs)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)
            
        # Output layer
        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output

################################### ST_LSTM ######################################

class ST_GATLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list,
                 end_channels, n_sequences, device, act_func, heads,
                 dropout, num_layers, binary, concat, graph_or_node='node', return_hidden=False):
        super(ST_GATLSTM, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = num_layers - 1
        self.n_sequences = n_sequences
        self.is_graph_or_node = graph_or_node == 'graph'

        # LSTM layers with different hidden channels per layer
        self.lstm_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.lstm_layers.append(torch.nn.LSTM(input_size=hidden_channels_list[i],
                                                   hidden_size=hidden_channels_list[i+1],
                                                   num_layers=1,  # Each layer is a single LSTM layer
                                                   dropout=dropout, batch_first=True).to(device))

        # GAT layer
        self.gat = GATConv(in_channels=hidden_channels_list[-1], out_channels=hidden_channels_list[-1],
                           heads=heads, dropout=dropout, concat=concat).to(device)

        self.graph_norm = torch.nn.BatchNorm1d(hidden_channels_list[-1]).to(device)

        # Adjusted output layer input size
        pooled_output_dim = hidden_channels_list[-1] * 3 if self.is_graph_or_node else hidden_channels_list[-1]
        pooled_output_dim = pooled_output_dim * heads if concat else pooled_output_dim
        self.output = OutputLayer(in_channels=pooled_output_dim, end_channels=end_channels,
                                  n_steps=n_sequences, device=device, act_func=act_func, binary=binary)

    def forward(self, X, edge_index, graphs=None):
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

        # Apply Batch Normalization
        x = self.batch_norm(x)

        # Extract the last output of LSTM
        x = x[:, :, -1]  # Shape: (batch_size, hidden_channels)

        # Pass through GAT layer
        xg = self.gat(x, edge_index)
        
        # Apply pooling if working with graph-level predictions
        if self.is_graph_or_node:
            x_mean = global_mean_pool(xg, graphs)
            x_max = global_max_pool(xg, graphs)
            x_sum = global_add_pool(xg, graphs)
            xg = torch.cat([x_mean, x_max, x_sum], dim=1)

        # Final output
        output = self.output(xg)
        
        return (output, xg) if self.return_hidden else output
    
################################################### LSTM #######################################################""

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list,
                 end_channels, n_sequences, device, act_func, binary, dropout, num_layers, return_hidden=False):
        super(LSTM, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = num_layers - 1

        # LSTM layers with different hidden channels per layer
        self.lstm_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.lstm_layers.append(torch.nn.LSTM(input_size=hidden_channels_list[i],
                                                   hidden_size=hidden_channels_list[i+1],
                                                   num_layers=1,
                                                   dropout=dropout,
                                                   batch_first=True).to(device))

        # Output layer
        self.output = OutputLayer(in_channels=hidden_channels_list[-1], end_channels=end_channels,
                                  n_steps=n_sequences, device=device, act_func=act_func,
                                  binary=binary)

        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels_list[-1]).to(device)

        self.is_graph_or_node = False

    def forward(self, X, edge_index=None, graphs=None):
        batch_size = X.size(0)

        # Rearrange dimensions for LSTM input
        x = X.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, residual_channels)

       # Initialisation des états cachés et des cellules pour chaque couche
        h0 = torch.zeros(1, batch_size, self.hidden_channels_list[0]).to(self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_channels_list[0]).to(self.device)

        # Boucle sur les couches LSTM
        for i in range(self.num_layers):
            # Passer l'entrée à travers la couche LSTM
            x, (h0, c0) = self.lstm_layers[i](x, (h0, c0))

        x = x.permute(0, 2, 1)  # Shape: (batch_size, residual_channels, sequence_length)

        # Apply Batch Normalization
        x = self.batch_norm(x)

        # Extract the last output of LSTM
        x = x[:, :, -1]  # Shape: (batch_size, hidden_channels)

        # Generate the final output
        output = self.output(x)

        if self.return_hidden:
            return output, x
        else:
            return output
