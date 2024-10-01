import sys
#sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/forecasting_models')
sys.path.insert(0, '/Home/Users/ncaron/WORK/ST-GNN-for-wildifre-prediction/Prediction/GNN/')
from forecasting_models.pytorch.utils import *

################################ GAT ###########################################
class GAT(torch.nn.Module):
    def __init__(self, n_sequences, in_dim,
                 heads,
                 dropout, 
                 bias,
                 device,
                 act_func,
                 binary, return_hidden=False):
            super(GAT, self).__init__()

            heads = [1] + heads
            num_of_layers = len(in_dim) - 1
            gat_layers = []
            self.return_hidden = return_hidden

            self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()

            for i in range(num_of_layers):
                layer = GATConv(
                    in_channels=in_dim[i] * heads[i],
                    out_channels=in_dim[i+1],
                    heads=heads[i+1],
                    concat=True if i < num_of_layers - 1 else False,
                    dropout=dropout,
                    bias=bias,
                ).to(device)
                gat_layers.append((layer, "x, edge_index -> x"))
                if i < num_of_layers - 1:
                    if act_func == 'relu':
                        gat_layers.append((ReLU(), "x -> x"))
                    elif act_func == 'gelu':
                        gat_layers.append((GELU(), 'x -> x'))

            self.net = Sequential("x, edge_index", gat_layers)

            self.output = OutputLayer(in_channels=in_dim[-1],
                                  end_channels=in_dim[-1],
                                  n_steps=n_sequences,
                                  device=device,
                                  act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        edge_index = edge_index[:2]
        x = self.net(X, edge_index)
        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output
        
################################ GCN ################################################

class GCN(torch.nn.Module):
    def __init__(self, n_sequences, in_dim,
                 dropout, 
                 bias,
                 device,
                 act_func,
                 binary, return_hidden=False):
        super(GCN, self).__init__()

        num_of_layers = len(in_dim) - 1
        gcn_layers = []
        self.return_hidden = return_hidden

        self.dropout_layer = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()

        for i in range(num_of_layers):
            layer = GCNConv(
                in_channels=in_dim[i],
                out_channels=in_dim[i+1],
                bias=bias,
            ).to(device)
            gcn_layers.append((layer, "x, edge_index -> x"))
            if i < num_of_layers - 1:
                if act_func == 'relu':
                    gcn_layers.append((ReLU(), "x -> x"))
                elif act_func == 'gelu':
                    gcn_layers.append((GELU(), "x -> x"))
                if dropout > 0.0:
                    gcn_layers.append((self.dropout_layer, "x -> x"))

        self.net = Sequential("x, edge_index", gcn_layers)

        self.output = OutputLayer(
            in_channels=in_dim[-1],
            end_channels=in_dim[-1],
            n_steps=n_sequences,
            device=device,
            act_func=act_func,
            binary=binary
        )

    def forward(self, X, edge_index):
        edge_index = edge_index[:2]
        x = self.net(X, edge_index)
        output = self.output(x)
        if self.return_hidden:
            return output, x
        else:
            return output


################################### ST_GATCN ######################################
    
# See code from https://github.com/SakastLord/STGAT and https://github.com/jswang/stgat_traffic_prediction
# https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py

class GatedDilatedConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(GatedDilatedConvolution, self).__init__()
        #self.residual = [in_channels // i for i in range(1, n_residual + 1)]
        #self.tcn1 = TCN(num_inputs=in_channels, num_channels=[out_channels], kernel_size=3, activation='tanh', input_shape='NCL', dilations=None)
        #self.tcn2 = TCN(num_inputs=in_channels, num_channels=[out_channels], kernel_size=3, activation='relu', input_shape='NCL', dilations=None)

        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=3, padding='same')
        self.conv2 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=3, padding='same')
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, X):
        x1 = self.conv1(X)
        x1 = self.tanh(x1)

        x2 = self.conv2(X)
        x2 = self.sigmoid(x2)

        x = torch.mul(x1, x2)

        #x1 = self.tcn1(X)
        #x2 = self.tcn2(X)
        #x = torch.mul(x1, x2)
        return x

class SpatioTemporalLayer(torch.nn.Module):
    def __init__(self, n_sequences,
                 in_channels,
                 out_channels,
                 dilation,
                 dropout):
        
        super(SpatioTemporalLayer, self).__init__()
        
        self.tcn = GatedDilatedConvolution(in_channels=in_channels, out_channels=out_channels, dilation=dilation)

        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.gcn = GCNConv(in_channels=out_channels * n_sequences, out_channels=out_channels * n_sequences)

        self.bn = nn.BatchNorm(in_channels=out_channels)

        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.n_sequences = n_sequences
        self.activation = torch.nn.GELU()
    
    def forward(self, X, edge_index):
        residual = self.residual_proj(X)

        x = self.tcn(X)

        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gcn(x, edge_index)
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)

        x = self.bn(x)
        x = self.activation(x + residual)
        x = self.drop(x)

        return x

class DSTGCN(torch.nn.Module):
    def __init__(self, n_sequences,
                 in_channels,
                 end_channels,
                 dilation_channels,
                 dilations,
                 dropout,
                 act_func,
                 device,
                 binary,
                 return_hidden=False):
        
        super(DSTGCN, self).__init__()
        
        self.return_hidden = return_hidden
        self.device = device

        # Parameters for Time2Vec
        self.time2vec_kernel_size = 1  # Adjust as needed
        self.time2vec_output_size = self.time2vec_kernel_size + 1
        self.time2vec = Time2Vec(kernel_size=self.time2vec_kernel_size).to(device)

        # Update input channels
        total_in_channels = in_channels + self.time2vec_output_size

        self.input = torch.nn.Conv1d(in_channels=total_in_channels, out_channels=dilation_channels[0], kernel_size=1, device=device)
        self.n_sequences = n_sequences
        self.layers = []

        num_of_layers = len(dilation_channels) - 1

        for i in range(num_of_layers):
            self.layers.append(SpatioTemporalLayer(n_sequences=n_sequences,
                                                   in_channels=dilation_channels[i],
                                                   out_channels=dilation_channels[i + 1],
                                                   dilation=dilations[i],
                                                   dropout=dropout).to(device))
            
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = OutputLayer(in_channels=dilation_channels[-1] * n_sequences,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device,
                                  act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        # Extract time indices (assuming time is the first channel)
        time_steps = X[:, 0, :].unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)
        # Remove time from features
        X = X[:, 1:, :]  # Shape: (batch_size, in_channels - 1, sequence_length)

        # Apply Time2Vec to the time indices
        t2v_output = self.time2vec(time_steps)  # Shape: (batch_size, sequence_length, time2vec_output_size)
        t2v_output = t2v_output.permute(0, 2, 1)  # Shape: (batch_size, time2vec_output_size, sequence_length)
        # Concatenate the input features with Time2Vec output
        x = torch.cat([X, t2v_output], dim=1)  # Shape: (batch_size, total_in_channels, sequence_length)

        x = self.input(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i == 0:
                skip = x
            else:
                skip = x + skip
        
        x = self.output(skip)
        if self.return_hidden:
            return x, skip
        else:
            return x


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

class SandiwchLayer(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, out_channels, dropout, heads, concat):
        super(SandiwchLayer, self).__init__()

        self.concat = concat
        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.gated_conv1 = Temporal_Gated_Conv(in_channels, in_channels, kernel_size=3)

        self.gat = GATConv(in_channels=in_channels * n_sequences,
                           out_channels=out_channels * n_sequences,
                           heads=heads, concat=concat,
                           dropout=dropout)

        self.gated_conv2 = Temporal_Gated_Conv(in_channels=out_channels, out_channels=out_channels)

        self.bn = nn.BatchNorm(in_channels=out_channels)

        self.n_sequences = n_sequences
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.activation = torch.nn.GELU()

    def forward(self, X, edge_index):
        residual = self.residual_proj(X)
        x = self.gated_conv1(X)
        
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gat(x, edge_index)
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)
        
        x = self.gated_conv2(x)

        x = self.bn(x)
        x = self.activation(x + residual)
        x = self.drop(x)
        
        return x
    
class STGAT(torch.nn.Module):
    def __init__(self, n_sequences,
                 in_channels,
                 hidden_channels,
                 end_channels,
                 dropout, heads, act_func, device,
                 binary, return_hidden=False):
        super(STGAT, self).__init__()

        self.return_hidden = return_hidden
        self.device = device

        # Parameters for Time2Vec
        self.time2vec_kernel_size = 1  # Adjust as needed
        self.time2vec_output_size = self.time2vec_kernel_size + 1
        self.time2vec = Time2Vec(kernel_size=self.time2vec_kernel_size).to(device)

        # Update input channels
        total_in_channels = in_channels + self.time2vec_output_size

        self.input = torch.nn.Conv1d(in_channels=total_in_channels, out_channels=hidden_channels[0], kernel_size=1, device=device)

        self.n_sequences = n_sequences
        self.layers = []
        self.skip_layers = []

        num_of_layers = len(hidden_channels) - 1

        for i in range(num_of_layers):
            concat = True if i < num_of_layers -1 else False
            self.layers.append(SandiwchLayer(n_sequences=n_sequences,
                                             in_channels=hidden_channels[i],
                                             out_channels=hidden_channels[i+1], 
                                             heads=heads,
                                             dropout=dropout,
                                             concat=concat).to(device))
            
        for i in range(num_of_layers):
            self.skip_layers.append(torch.nn.Conv1d(in_channels=hidden_channels[i+1],
                                              out_channels=hidden_channels[-1],
                                              padding='same',
                                              kernel_size=1,
                                              device=device))
                
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = OutputLayer(in_channels=hidden_channels[-1] * self.n_sequences,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        # Extract time indices
        time_steps = X[:, 0, :].unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)
        # Remove time from features
        X = X[:, 1:, :]  # Shape: (batch_size, in_channels - 1, sequence_length)

        # Apply Time2Vec
        time_steps = time_steps.view(time_steps.shape[0], time_steps.shape[1], 1)
        t2v_output = self.time2vec(time_steps)  # Shape: (batch_size, sequence_length, time2vec_output_size)
        t2v_output = t2v_output.permute(0, 2, 1)  # Shape: (batch_size, time2vec_output_size, sequence_length)
        # Concatenate the input features with Time2Vec output
        
        x = torch.cat([X, t2v_output], dim=1)  # Shape: (batch_size, total_in_channels, sequence_length)
        x = self.input(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            s = self.skip_layers[i](x)
            if i == 0:
                skip = s
            else:
                skip = s + skip
        
        x = self.output(skip)
        if self.return_hidden:
            return x, skip
        else:
            return x

###################################### ST-GCN #####################################################

class SandiwchLayerGCN(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, out_channels, dropout, act_func):
        super(SandiwchLayerGCN, self).__init__()

        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.gated_conv1 = Temporal_Gated_Conv(in_channels, out_channels, kernel_size=3)

        self.gcn = GCNConv(in_channels=out_channels * n_sequences,
                           out_channels=out_channels * n_sequences)

        self.gated_conv2 = Temporal_Gated_Conv(in_channels=out_channels, out_channels=out_channels)

        self.bn = nn.BatchNorm(in_channels=out_channels)

        self.n_sequences = n_sequences
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        if act_func == 'gelu':
            self.activation = torch.nn.GELU()
        elif act_func == 'relu':
            self.activation = torch.nn.ReLU()
        elif self.activation == 'silu':
            self.activation = torch.nn.SiLU()
            
    def forward(self, X, edge_index):
        residual = self.residual_proj(X)
        x = self.gated_conv1(X)
        
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gcn(x, edge_index)
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)
        
        x = self.gated_conv2(x)

        x = self.bn(x)
        x = self.activation(x + residual)
        x = self.drop(x)
        
        return x
    
class STGCN(torch.nn.Module):
    def __init__(self, n_sequences,
                 in_channels,
                 hidden_channels,
                 end_channels,
                 dropout, act_func, device,
                 binary, return_hidden=False):
        super(STGCN, self).__init__()
        self.return_hidden = return_hidden
        self.device = device

        # Parameters for Time2Vec
        self.time2vec_kernel_size = 1
        self.time2vec_output_size = self.time2vec_kernel_size + 1
        self.time2vec = Time2Vec(kernel_size=self.time2vec_kernel_size).to(device)

        # Update input channels
        total_in_channels = in_channels + self.time2vec_output_size

        num_of_layers = len(hidden_channels) - 1
        self.layers = []
        self.input = torch.nn.Conv1d(in_channels=total_in_channels,
                               out_channels=hidden_channels[0],
                               kernel_size=1,
                               device=device)

        self.n_sequences = n_sequences

        for i in range(num_of_layers):
            self.layers.append(SandiwchLayerGCN(n_sequences=n_sequences,
                                                in_channels=hidden_channels[i],
                                                out_channels=hidden_channels[i + 1],
                                                dropout=dropout,
                                                act_func=act_func).to(device))
                
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = OutputLayer(in_channels=hidden_channels[-1] * self.n_sequences,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        # Extract time indices
        time_steps = X[:, 0, :].unsqueeze(2)
        X = X[:, 1:, :]

        # Apply Time2Vec
        t2v_output = self.time2vec(time_steps)
        t2v_output = t2v_output.permute(0, 2, 1)

        x = torch.cat([X, t2v_output], dim=1)
        x = self.input(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i == 0:
                skip = x
            else:
                skip = x + skip
        
        x = self.output(skip)
        if self.return_hidden:
            return x, skip
        else:
            return x

################################## SDSTGCN ###############################

class GCNLAYER(torch.nn.Module):
    def __init__(self, in_dim, end_channels,
                 dropout,
                 act_func):
        super(GCNLAYER, self).__init__()

        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()

        self.gcn = GCNConv(in_channels=in_dim,
                        out_channels=end_channels)
        
        self.activation = None
        if act_func == 'relu':
            self.activation = torch.nn.ReLU()
        if act_func == 'gelu':
            self.activation = torch.nn.GELU()

    def forward(self, X, edge_index):
        X = self.gcn(X, edge_index)
        if self.activation is not None:
            X = self.activation(X)
        X = self.dropout(X)
        return X

class SDSTGCN(torch.nn.Module):
    def __init__(self, n_sequences,
                 in_channels,
                 hidden_channels_temporal,
                 dilations,
                 hidden_channels_spatial,
                 end_channels,
                 dropout, act_func, device,
                 binary, return_hidden=False):
        
        super(SDSTGCN, self).__init__()

        self.return_hidden = return_hidden
        self.device = device

        # Time2Vec for temporal input
        self.time2vec_kernel_size = 1
        self.time2vec_output_size = self.time2vec_kernel_size + 1
        self.time2vec_temporal = Time2Vec(kernel_size=self.time2vec_kernel_size).to(device)

        # Time2Vec for spatial input
        self.time2vec_spatial = Time2Vec(kernel_size=self.time2vec_kernel_size).to(device)

        # Adjust input channels
        total_in_channels_temporal = in_channels + self.time2vec_output_size
        total_in_channels_spatial = in_channels + self.time2vec_output_size

        num_of_temporal_layers = len(hidden_channels_temporal) - 1
        num_of_spatial_layers = len(hidden_channels_spatial) - 1

        self.temporal_layers = []
        self.spatial_layers = []

        self.input_temporal = torch.nn.Conv1d(in_channels=total_in_channels_temporal,
                                        out_channels=hidden_channels_temporal[0],
                                        kernel_size=1,
                                        device=device)

        self.input_spatial = torch.nn.Linear(in_features=total_in_channels_spatial,
                                        out_features=hidden_channels_spatial[0],
                                        bias=True).to(device)

        self.n_sequences = n_sequences

        for ti in range(num_of_temporal_layers):
            self.temporal_layers.append(GatedDilatedConvolution(in_channels=hidden_channels_temporal[ti],
                                                                out_channels=hidden_channels_temporal[ti + 1],
                                                                dilation=dilations[ti]))

        self.temporal_layers = torch.nn.ModuleList(self.temporal_layers)

        for si in range(num_of_spatial_layers):
            self.spatial_layers.append(GCNLAYER(in_dim=hidden_channels_spatial[si],
                                                end_channels=hidden_channels_spatial[si+1],
                                                dropout=dropout,
                                                act_func=act_func).to(device))

        self.spatial_layers = torch.nn.ModuleList(self.spatial_layers)
        self.output = OutputLayer(in_channels=(hidden_channels_spatial[-1] + hidden_channels_temporal[-1]) * 4,
                                  end_channels=end_channels,
                                  n_steps=1,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        # Extract time indices
        time_steps = X[:, 0, :].unsqueeze(2)
        X_features = X[:, 1:, :]

        # Temporal Time2Vec
        t2v_output_temporal = self.time2vec_temporal(time_steps)
        t2v_output_temporal = t2v_output_temporal.permute(0, 2, 1)
        xt_input = torch.cat([X_features, t2v_output_temporal], dim=1)
        xt = self.input_temporal(xt_input)

        # Spatial Time2Vec
        time_steps_spatial = time_steps.squeeze(2)
        t2v_output_spatial = self.time2vec_spatial(time_steps_spatial.unsqueeze(2))
        t2v_output_spatial = t2v_output_spatial.squeeze(1)
        xs_input = torch.cat([X_features[:, :, -1], t2v_output_spatial], dim=1)
        xs = self.input_spatial(xs_input)

        for i, layer in enumerate(self.temporal_layers):
            xt = layer(xt)
        
        for i, layer in enumerate(self.spatial_layers):
            xs = layer(xs, edge_index)

        for band in range(xt.shape[-1]):
            xt[:, :, band] += xs

        x = self.output(xt)

        if self.return_hidden:
            return x, xt
        else:
            return x
    
################################## DGATCONV ####################################
class GATLAYER(torch.nn.Module):
    def __init__(self, in_dim, end_channels,
                 dropout,
                 act_func,
                 concat,
                 heads):
        super(GATLAYER, self).__init__()

        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()

        self.gcn = GATConv(in_channels=in_dim,
                        out_channels=end_channels, heads=heads, concat=concat)
        
        self.activation = None
        if act_func == 'relu':
            self.activation = torch.nn.ReLU()
        if act_func == 'gelu':
            self.activation = torch.nn.GELU()

    def forward(self, X, edge_index):
        X = self.gcn(X, edge_index)
        if self.activation is not None:
            X = self.activation(X)
        X = self.dropout(X)
        return X

################################## TEMPORAL GNN ################################
class TemporalGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_sequences, device, act_func, dropout, binary):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.n_sequences = n_sequences
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1).to(device)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.tgnn = A3TGCN(in_channels=hidden_channels,
                           out_channels=hidden_channels,
                           periods=n_sequences).to(device)
        
        # Equals single-shot prediction
        self.output = OutputLayer(in_channels=hidden_channels,
                                  end_channels=out_channels,
                                  n_steps=n_sequences,
                                  device=device,
                                  act_func=act_func, 
                                  binary=binary)

    def forward(self, X, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        x = self.input(X)
        x = self.tgnn(x, edge_index)
        x = self.dropout(x)
        x = self.output(x)
        return x

################################### ST_LSTM ######################################
class ST_GATLSTM(torch.nn.Module):
    """
    Spatio-Temporal Graph Attention Network with Time2Vec encoding.
    Based on the architecture presented in https://ieeexplore.ieee.org/document/8903252
    """
    def __init__(self, in_channels, hidden_channels,
                 residual_channels, end_channels, n_sequences, device, act_func,
                 heads, dropout, num_layers,
                 binary, concat, return_hidden=False):
        super(ST_GATLSTM, self).__init__()
        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.n_sequences = n_sequences

        # Parameters for Time2Vec
        self.time2vec_kernel_size = 1  # Adjust as needed
        self.time2vec_output_size = self.time2vec_kernel_size + 1
        self.time2vec = Time2Vec(kernel_size=self.time2vec_kernel_size).to(device)

        # Update in_channels to account for Time2Vec output (assuming time is the first feature)
        total_in_channels = in_channels + self.time2vec_output_size

        # Input convolutional layer with updated input channels
        self.input = torch.nn.Conv1d(in_channels=total_in_channels, out_channels=residual_channels, kernel_size=1).to(device)

        self.bn = torch.nn.BatchNorm1d(residual_channels)

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=residual_channels, hidden_size=hidden_channels, num_layers=num_layers,
                                  batch_first=True, dropout=dropout).to(device)

        # Graph Attention Network layer
        self.gat = GATConv(in_channels=hidden_channels, out_channels=hidden_channels,
                           heads=heads, dropout=dropout, concat=concat).to(device)

        # Output layer
        self.output = OutputLayer(in_channels=hidden_channels * heads if concat else hidden_channels,
                                  end_channels=end_channels,
                                  n_steps=n_sequences,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        batch_size = X.size(0)
        sequence_length = X.size(2)

        # Extract time indices (assuming time is the first channel)
        time_steps = X[:, 0, :].unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)

        # Remove time from input features
        X = X[:, 1:, :]  # Shape: (batch_size, in_channels - 1, sequence_length)

        # Apply Time2Vec to the time indices
        t2v_output = self.time2vec(time_steps)  # Shape: (batch_size, sequence_length, time2vec_output_size)
        t2v_output = t2v_output.permute(0, 2, 1)  # Shape: (batch_size, time2vec_output_size, sequence_length)

        # Concatenate Time2Vec output with input features
        x = torch.cat([X, t2v_output], dim=1)  # Shape: (batch_size, total_in_channels, sequence_length)

        # Apply the input convolutional layer
        x = self.input(x)  # Shape: (batch_size, residual_channels, sequence_length)
        x = self.bn(x)

        # Rearrange dimensions for LSTM input
        x = x.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, residual_channels)

        # Initialize hidden states for LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)

        # Pass through LSTM layer
        x, _ = self.lstm(x, (h0, c0))

        # Extract the last output of LSTM
        x = x[:, -1, :]  # Shape: (batch_size, hidden_channels)

        # Pass through GAT layer
        xg = self.gat(x, edge_index)

        # Generate the final output
        output = self.output(xg)

        if self.return_hidden:
            return output, xg
        else:
            return output
############################### LSTM ##################################

# Modified LSTM class with Time2Vec
class LSTM(torch.nn.Module):
    def __init__(self, in_channels, residual_channels, hidden_channels,
                 end_channels, n_sequences, device, act_func, binary, dropout, num_layers, return_hidden=False):
        super(LSTM, self).__init__()

        self.return_hidden = return_hidden
        self.device = device
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Parameters for Time2Vec
        self.time2vec_kernel_size = 1
        self.time2vec_output_size = self.time2vec_kernel_size + 1
        self.time2vec = Time2Vec(kernel_size=self.time2vec_kernel_size).to(device)

        # Update the input channels to account for Time2Vec output
        total_in_channels = in_channels + self.time2vec_output_size

        # First convolutional layer with updated input channels
        self.input = torch.nn.Conv1d(in_channels=total_in_channels, out_channels=residual_channels, kernel_size=1).to(device)

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=residual_channels, hidden_size=hidden_channels, num_layers=num_layers,
                                  dropout=dropout, batch_first=True).to(device)

        # Output layer
        self.output = OutputLayer(in_channels=hidden_channels, end_channels=end_channels,
                                  n_steps=n_sequences, device=device, act_func=act_func,
                                  binary=binary)

        self.batchNorm = torch.nn.BatchNorm1d(hidden_channels).to(device)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X, edge_index=None):
        # edge_index is ignored here
        batch_size = X.size(0)

        # Initialize hidden states for LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)

        # Get time indices
        time_steps = X[:, 0, :]
        time_steps = time_steps.view(time_steps.shape[0], time_steps.shape[1], 1)

        # Remove time from features
        X = X[:, 1:, :]

        # Apply Time2Vec to the time indices
        t2v_output = self.time2vec(time_steps)  # Shape: (batch_size, sequence_length, time2vec_output_size)
        t2v_output = t2v_output.permute(0, 2, 1)  # Shape: (batch_size, time2vec_output_size, sequence_length)

        # Concatenate the input features with Time2Vec output
        x = torch.cat([X, t2v_output], dim=1)  # Shape: (batch_size, total_in_channels, sequence_length)

        # Apply the first convolutional layer
        x = self.input(x)  # Shape: (batch_size, residual_channels, sequence_length)
        original_input = x[:, :, -1]

        # Rearrange dimensions for LSTM input
        x = x.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, residual_channels)

        # Pass through LSTM layer
        x, _ = self.lstm(x, (h0, c0))

        # Extract the last output of LSTM
        x = x[:, -1, :]  # Shape: (batch_size, hidden_channels)

        # Apply Batch Normalization and Dropout
        x = self.batchNorm(x)
        x = self.dropout(x)

        # Generate the final output
        output = self.output(x)

        if self.return_hidden:
            return output, x
        else:
            return output


############################### GraphSAGE ##################################
    
class GRAPH_SAGE(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
