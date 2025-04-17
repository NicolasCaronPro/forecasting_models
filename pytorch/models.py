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
import dgl
import dgl.function as fn

##################################### SIMPLE GRAPH #####################################
class NetGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim_2, output_channels, end_channels, n_sequences, graph_or_node, device, task_type):
        super(NetGCN, self).__init__()
        self.layer1 = GraphConv(in_dim * n_sequences, hidden_dim).to(device)
        self.layer2 = GraphConv(hidden_dim, hidden_dim_2).to(device)
        self.is_graph_or_node = graph_or_node == 'graph'
        self.n_sequences = n_sequences
        self.device = device
        self.task_type = task_type
        self.soft = torch.nn.Softmax(dim=1)

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

        x = self.output_layer(x)

        return x
    
class MLPLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, device):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(in_feats, hidden_dim, weight_initializer='glorot', bias=True, bias_initializer='zeros').to(device)
        #self.mlp = torch.nn.Linear(in_feats, hidden_dim).to(device)
    def forward(self, x):
        return self.mlp(x)
    
class NetMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, output_channels, n_sequences, device, task_type):
        super(NetMLP, self).__init__()
        self.layer1 = MLPLayer(in_dim * n_sequences, hidden_dim, device)
        self.layer3 = MLPLayer(hidden_dim, hidden_dim * 2, device)
        self.layer4 = MLPLayer(hidden_dim * 2, hidden_dim * 4, device)
        self.layer2 = MLPLayer(hidden_dim * 4, output_channels, device)
        self.task_type = task_type
        self.n_sequences = n_sequences
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, features, edges=None):
        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)
        x = F.relu(self.layer1(features))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer2(x)
        if self.task_type == 'classification':
            x = self.soft(x)
        return x
    
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
        end_channels=64,
        out_channels=1,
        n_sequences=1,
        task_type='classification',
        device=None,
        act_func='relu',
        static_idx=None,
        temporal_idx=None,
        num_lstm_layers=1
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

        # Output layer
        self.output_layer = OutputLayer(
            in_channels=lstm_hidden + self.gnn_output_dim,
            end_channels=end_channels,
            n_steps=n_sequences,
            device=device,
            act_func=act_func,
            task_type=task_type,
            out_channels=out_channels
        )

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
        fused = torch.cat([lstm_out, gnn_out], dim=1)  # (B, total)
        output = self.output_layer(fused)
        return output

class GRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, end_channels, n_sequences, device, act_func, task_type, dropout, 
                 num_layers, return_hidden=False, out_channels=None):
        super(GRU, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = len(hidden_channels_list)

        # GRU layers with different hidden channels per layer
        self.gru_layers = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.gru_layers.append(torch.nn.GRU(input_size=hidden_channels_list[i],
                                                hidden_size=hidden_channels_list[i+1],
                                                num_layers=1,
                                                dropout=dropout,
                                                batch_first=True).to(device))

        # Output layer (same as in LSTM)
        self.output = OutputLayer(in_channels=hidden_channels_list[-1], end_channels=end_channels,
                                  n_steps=n_sequences, device=device, act_func=act_func,
                                  task_type=task_type, out_channels=out_channels)

        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels_list[-1]).to(device)

        self.is_graph_or_node = False

    def forward(self, X, edge_index=None, graphs=None):
        batch_size = X.size(0)

        # Rearrange dimensions for GRU input
        x = X.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, features)

        # Initial hidden state
        h0 = torch.zeros(1, batch_size, self.hidden_channels_list[0]).to(self.device)

        # Pass through each GRU layer
        for i in range(self.num_layers - 1):
            x, h0 = self.gru_layers[i](x, h0)

        x = x.permute(0, 2, 1)  # Back to (batch_size, features, sequence_length)

        # Batch Normalization
        x = self.batch_norm(x)

        # Take last time step output
        x = x[:, :, -1]

        # Output
        output = self.output(x)

        if self.return_hidden:
            return output, x
        else:
            return output
    
class LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, end_channels, n_sequences, device, act_func, task_type, dropout, 
                 num_layers, return_hidden=False, out_channels=None):
        super(LSTM, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = len(hidden_channels_list)

        # LSTM layers with different hidden channels per layer
        self.lstm_layers = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.lstm_layers.append(torch.nn.LSTM(input_size=hidden_channels_list[i],
                                                   hidden_size=hidden_channels_list[i+1],
                                                   num_layers=1,
                                                   dropout=dropout,
                                                   batch_first=True).to(device))

        """self.lstm = torch.nn.LSTM(input_size=hidden_channels_list,
                                                   hidden_size=hidden_channels_list,
                                                   num_layers=self.num_layers,
                                                   dropout=dropout,
                                                   batch_first=True)"""

        # Output layer
        self.output = OutputLayer(in_channels=hidden_channels_list[-1], end_channels=end_channels,
                                  n_steps=n_sequences, device=device, act_func=act_func,
                                  task_type=task_type, out_channels=out_channels)
        
        #self.output = torch.nn.Linear(hidden_channels_list[-1], out_channels)
        #self.sofmax = torch.nn.Softmax(dim=-1)

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
        for i in range(self.num_layers - 1):
            # Passer l'entrée à travers la couche LSTM
            x, (h0, c0) = self.lstm_layers[i](x, (h0, c0))
        
        #x, _ = self.lstm(x, (h0, c0))

        x = x.permute(0, 2, 1)  # Shape: (batch_size, residual_channels, sequence_length)

        # Apply Batch Normalization
        x = self.batch_norm(x)

        # Extract the last output of LSTM
        x = x[:, :, -1]  # Shape: (batch_size, hidden_channels)

        # Generate the final output
        #output = self.sofmax(self.output(x))
        output = self.output(x)

        if self.return_hidden:
            return output, x
        else:
            return output
        
class DilatedCNN(torch.nn.Module):
    def __init__(self, in_channels, channels, dilations, end_channels, n_sequences, device, act_func, dropout, out_channels, task_type):
        super(DilatedCNN, self).__init__()

        # Initialisation des listes pour les convolutions et les BatchNorm
        self.cnn_layer_list = []
        self.batch_norm_list = []
        self.num_layer = len(channels) - 1
        
        # Couche d'entrée
        self.input = torch.nn.Conv1d(in_channels, channels[0], kernel_size=1, padding='same', dilation=1, padding_mode='zeros').to(device)

        # Initialisation des couches convolutives et BatchNorm
        for i in range(self.num_layer):
            self.cnn_layer_list.append(torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            self.batch_norm_list.append(torch.nn.BatchNorm1d(channels[i + 1]).to(device))
        
        # Fonction d'activation
        if act_func == 'relu':
            self.activation = torch.nn.ReLU()
        elif act_func == 'gelu':
            self.activation = torch.nn.GELU()
        else:
            print(f'WARNING : {act_func} not implemented. Use Identity')
            self.activation = torch.nn.Identity()
        
        self.dropout = torch.nn.Dropout(dropout)
        
        # Convertir les listes en ModuleList pour être compatible avec PyTorch
        self.cnn_layer_list = torch.nn.ModuleList(self.cnn_layer_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        
        # Couche de sortie
        self.output = OutputLayer(
            in_channels=channels[-1],
            end_channels=end_channels,
            n_steps=n_sequences,
            device=device,
            act_func=act_func,
            task_type=task_type,
            out_channels=out_channels
        )

    def forward(self, x, edges=None):
        # Couche d'entrée

        x = self.input(x)
        x = self.activation(x)
        
        # Couches convolutives dilatées avec BatchNorm, activation et dropout
        for cnn_layer, batch_norm in zip(self.cnn_layer_list, self.batch_norm_list):
            x = cnn_layer(x)
            x = batch_norm(x)  # Batch Normalization
            x = self.activation(x)
            x = self.dropout(x)
        
        # Garder uniquement le dernier élément des séquences
        x = x[:, :, -1]
        
        # Couche de sortie
        x = self.output(x)
        
        return x
    
class GraphCast(torch.nn.Module):
    def __init__(self,
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        has_time_dim: bool = False,
        n_sequences = 1,
        is_graph_or_node=False):
        super(GraphCast, self).__init__()

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
            has_time_dim)
        
        self.is_graph_or_node = is_graph_or_node == 'graph'
        
        self.softmax_layer = torch.nn.Softmax(dim=-1)

    def forward(self, X, graph, graph2mesh, mesh2graph):
        #X = X.view(X.shape[0], -1)
        X = X.permute(2, 0, 1)
        x = self.net(X, graph, graph2mesh, mesh2graph)
        return self.softmax_layer(x)[-1]

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
        graph_or_node, task_type, num_sequence=1, out_channels=5
    ):
        super(MultiScaleGraph, self).__init__()
        
        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device = device
        self.features_per_scale = features_per_scale

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
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            outputs.append(F.softmax(logits, dim=-1))

        outputs = torch.cat(outputs, dim=0)
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
        num_sequence=1, out_channels=5, num_heads=4
    ):
        super(MultiScaleAttentionGraph, self).__init__()

        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device = device
        self.features_per_scale = features_per_scale

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
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            outputs.append(F.softmax(logits, dim=-1))

        outputs = torch.cat(outputs, dim=0)
        return outputs