import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/forecasting_models')
#sys.path.insert(0, '/Home/Users/ncaron/WORK/GNN/forecasting_models')
from utils import *

################################ GAT ###########################################
class GAT(torch.nn.Module):
    def __init__(self, n_sequences, in_dim,
                 heads,
                 dropout, 
                 bias,
                 device,
                 act_func,
                 binary):
            super(GAT, self).__init__()

            heads = [1] + heads
            num_of_layers = len(in_dim) - 1
            gat_layers = []

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
        x = X[:,:,-1]
        x = self.net(x, edge_index)
        x = self.dropout(x)
        x = self.output(x)
        return x

################################### ST_GATCN ######################################
    
# See code from https://github.com/SakastLord/STGAT and https://github.com/jswang/stgat_traffic_prediction
# https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py

class TemporalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(TemporalBlock, self).__init__()
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
                 residual_channels,
                 skip_channels,
                 dilation_channels,
                 dilation,
                 dropout,
                 concat,
                 heads):
        
        super(SpatioTemporalLayer, self).__init__()
        
        self.tcn = TemporalBlock(in_channels=residual_channels, out_channels=dilation_channels, dilation=dilation)
        self.concat = concat

        if concat:
            self.residual_proj = torch.nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels * heads, kernel_size=1)
        else:
            self.residual_proj = torch.nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=1)

        self.gat = GATConv(in_channels=dilation_channels * n_sequences, out_channels=dilation_channels * n_sequences,
                           heads=heads, concat=concat, dropout=0.03)

        self.skipconv = torch.nn.Conv1d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=1)

        self.bn = nn.BatchNorm(in_channels=dilation_channels * heads if concat else dilation_channels)

        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.n_sequences = n_sequences
        self.activation = torch.nn.GELU()
    
    def forward(self, X, edge_index):
        residual = self.residual_proj(X)

        x = self.tcn(X)
        s = self.skipconv(x)

        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gat(x, edge_index)
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)

        x = self.bn(x)
        #x =  x + residual
        x = self.activation(x + residual)
        x = self.drop(x)

        return x, s

class STGATCN(torch.nn.Module):
    def __init__(self, n_sequences, num_of_layers,
                 in_channels,
                 end_channels,
                 skip_channels,
                 residual_channels,
                 dilation_channels,
                 dropout,
                 heads,
                 act_func,
                 device,
                 binary):
        
        super(STGATCN, self).__init__()
        

        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=1, device=device)
        dilation = 1
        self.n_sequences = n_sequences
        self.layers = []

        in_channels = residual_channels

        for i in range(num_of_layers):
            concat = True if i < num_of_layers -1 else False
        
            self.layers.append(SpatioTemporalLayer(n_sequences=n_sequences,
                                            residual_channels=in_channels,
                                            skip_channels=skip_channels,
                                            dilation_channels=dilation_channels,
                                            dilation=dilation, dropout=dropout,
                                            concat=concat,
                                            heads=heads).to(device))
        
            in_channels = residual_channels * heads if concat else residual_channels
            
            dilation *= 2
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = OutputLayer(in_channels=skip_channels * n_sequences,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device,
                                  act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        x = self.input(X)
        #x = X
        for i, layer in enumerate(self.layers):
            x, s = layer(x, edge_index)
            if i == 0:
                skip = s
            else:
                skip = s + skip
        
        x = self.output(skip)
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
    def __init__(self, n_sequences, in_channels, hidden_channels, out_channels, dropout, heads, concat):
        super(SandiwchLayer, self).__init__()

        coef = heads if concat else 1
        self.concat = concat
        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels * coef, kernel_size=1)

        self.gated_conv1 = Temporal_Gated_Conv(in_channels, hidden_channels, kernel_size=3)

        self.skipconv = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

        self.gat = GATConv(in_channels=hidden_channels * n_sequences,
                           out_channels=hidden_channels * n_sequences,
                           heads=heads, concat=concat,
                           dropout=0.03)

        self.gated_conv2 = Temporal_Gated_Conv(in_channels=hidden_channels * coef, out_channels=out_channels * coef)

        self.bn = nn.BatchNorm(in_channels=out_channels * coef)

        self.n_sequences = n_sequences
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.activation = torch.nn.GELU()

    def forward(self, X, edge_index):
        residual = self.residual_proj(X)
        #residual = X
        x = self.gated_conv1(X)

        s = self.skipconv(x)
        
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gat(x, edge_index)
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)
        
        x = self.gated_conv2(x)

        x = self.bn(x)
        x = self.activation(x + residual)
        x = self.drop(x)
        
        return x, s
    
class STGATCONV(torch.nn.Module):
    def __init__(self, n_sequences, num_of_layers,
                 in_channels,
                 residual_channels,
                 hidden_channels,
                 end_channels,
                 dropout, heads, act_func, device,
                 binary):
        super(STGATCONV, self).__init__()

        self.layers = []
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=1, device=device)

        self.n_sequences = n_sequences
        in_channels = residual_channels

        for i in range(num_of_layers):
            concat = True if i < num_of_layers -1 else False

            self.layers.append(SandiwchLayer(n_sequences=n_sequences,
                                            in_channels=in_channels,
                                            hidden_channels=hidden_channels,
                                            out_channels=residual_channels, 
                                            heads=heads, dropout=dropout,
                                            concat=concat).to(device))
            
            in_channels = residual_channels * heads if concat else residual_channels
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = OutputLayer(in_channels=residual_channels * self.n_sequences,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        x = self.input(X)
        #x = X
        for i, layer in enumerate(self.layers):
            x, s = layer(x, edge_index)
            if i == 0:
                skip = s
            else:
                skip = s + skip
        
        x = self.output(skip)
        return x


###################################### ST-GCN #####################################################

class SandiwchLayerGCN(torch.nn.Module):
    def __init__(self, n_sequences, in_channels, hidden_channels, out_channels, dropout, act_func):
        super(SandiwchLayerGCN, self).__init__()

        self.residual_proj = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.gated_conv1 = Temporal_Gated_Conv(in_channels, hidden_channels, kernel_size=3)

        self.skipconv = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

        self.gcn = GCNConv(in_channels=hidden_channels * n_sequences,
                           out_channels=hidden_channels * n_sequences)

        self.gated_conv2 = Temporal_Gated_Conv(in_channels=hidden_channels, out_channels=out_channels)

        self.bn = nn.BatchNorm(in_channels=out_channels)

        self.n_sequences = n_sequences
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        if act_func == 'gelu':
            self.activation = torch.nn.GELU()
        elif act_func == 'relu':
            self.activation = torch.nn.ReLU()
            
    def forward(self, X, edge_index):
        residual = self.residual_proj(X)
        x = self.gated_conv1(X)

        s = self.skipconv(x)
        
        x = x.view(X.shape[0], self.n_sequences * x.shape[1])
        x = self.gcn(x, edge_index)
        x = x.reshape(X.shape[0], x.shape[1] // self.n_sequences, self.n_sequences)
        
        x = self.gated_conv2(x)

        x = self.bn(x)
        x = self.activation(x + residual)
        x = self.drop(x)
        
        return x, s
    
class STGCNCONV(torch.nn.Module):
    def __init__(self, n_sequences, num_of_layers,
                 in_channels,
                 residual_channels,
                 hidden_channels,
                 end_channels,
                 dropout, act_func, device,
                 binary):
        super(STGCNCONV, self).__init__()

        self.layers = []
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=1, device=device)

        self.n_sequences = n_sequences
        in_channels = residual_channels

        for i in range(num_of_layers):
            self.layers.append(SandiwchLayerGCN(n_sequences=n_sequences,
                                            in_channels=in_channels,
                                            hidden_channels=hidden_channels,
                                            out_channels=residual_channels, 
                                            dropout=dropout,
                                            act_func=act_func).to(device))
            
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = OutputLayer(in_channels=residual_channels * self.n_sequences,
                                  end_channels=end_channels,
                                  n_steps=self.n_sequences,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, X, edge_index):
        x = self.input(X)
        for i, layer in enumerate(self.layers):
            x, s = layer(x, edge_index)
            if i == 0:
                skip = s
            else:
                skip = s + skip
        
        x = self.output(skip)
        return x
    
################################## MY ST-GCNCONV ###############################

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
        X = self.dropout(X)
        X = self.gcn(X, edge_index)
        if self.activation is not None:
            X = self.activation(X)
        return X

class DSTGCNCONV(torch.nn.Module):
    def __init__(self, n_sequences, num_of_temporal_layers,
                 num_of_spatial_layers,
                 in_channels,
                 residual_channels,
                 hidden_channels,
                 end_channels,
                 dropout, act_func, device,
                 binary):
        
        super(DSTGCNCONV, self).__init__()

        self.temporal_layers = []
        self.spatial_layers = []
        self.input_temporal = torch.nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=1, device=device)
        self.input_spatial = torch.nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=1, device=device)

        self.n_sequences = n_sequences
        in_channels = residual_channels

        for i in range(num_of_temporal_layers):
            self.temporal_layers.append(SandiwchLayerGCN(n_sequences=n_sequences,
                                            in_channels=in_channels,
                                            hidden_channels=hidden_channels,
                                            out_channels=residual_channels, 
                                            dropout=dropout,
                                            act_func=act_func).to(device))
            
        self.temporal_layers = torch.nn.ModuleList(self.temporal_layers)

        self.temporal_output = OutputLayer(in_channels=residual_channels * self.n_sequences,
                                  end_channels=residual_channels,
                                  n_steps=self.n_sequences,
                                  device=device, act_func=act_func,
                                  binary=False)
        
        for i in range(num_of_spatial_layers):
            self.spatial_layers.append(GCNLAYER(in_dim=residual_channels, end_channels=residual_channels, dropout=dropout, act_func=act_func).to(device))

        self.spatial_layers = torch.nn.ModuleList(self.spatial_layers)
        self.spatial_output = OutputLayer(in_channels=residual_channels,
                                  end_channels=residual_channels,
                                  n_steps=1,
                                  device=device, act_func=act_func,
                                  binary=False)

        self.binary = binary
        if not binary:
            self.output = nn.Linear(in_channels=2, out_channels=1, weight_initializer='glorot', bias=True).to(device)
        else:
            self.output = nn.Linear(in_channels=2, out_channels=2, weight_initializer='glorot', bias=True).to(device)
            self.softmax = Softmax()

    def forward(self, X, edge_index):
        xt = self.input_temporal(X)
        for i, layer in enumerate(self.temporal_layers):
            xt, s = layer(xt, edge_index)
            if i == 0:
                skip = s
            else:
                skip = s + skip
        xt = self.temporal_output(skip)

        xs = self.input_spatial(X)
        xs = xs[:,:,-1]
        for i, layer in enumerate(self.spatial_layers):
            xs = layer(xs, edge_index)

        xs = self.spatial_output(xs)

        x = torch.concat((xs, xt), axis=1)

        x = self.output(x)
        if self.binary:
            x = self.softmax(x)
        return x

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
    Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
    """
    def __init__(self, in_channels, hidden_channels,
                 residual_channels, end_channels, n_sequences, device, act_func,
                 heads, dropout, num_layers,
                 binary, concat):
        super(ST_GATLSTM, self).__init__()
        self.n_sequences = n_sequences

        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=1).to(device)

        self.bn = torch.nn.BatchNorm1d(residual_channels)

        self.lstm = torch.nn.LSTM(input_size=residual_channels, hidden_size=hidden_channels, num_layers=num_layers, batch_first=True, dropout=dropout).to(device)

        self.gat = GATConv(in_channels=hidden_channels, out_channels=hidden_channels,
            heads=heads, dropout=dropout, concat=concat).to(device)

        self.output = OutputLayer(in_channels=hidden_channels * heads if concat else hidden_channels,
                                  end_channels=end_channels,
                                  n_steps=n_sequences,
                                  device=device, act_func=act_func,
                                  binary=binary)
        
        self.device = device
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

    def forward(self, X, edge_index):
        batch_size = X.size(0)
        
        x = self.input(X)
        x = torch.movedim(x, 2, 1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        
        x = torch.squeeze(x[:, -1, :])
        x = self.gat(x, edge_index)

        x = self.output(x)

        return x
############################### LSTM ##################################

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, residual_channels, hidden_channels,
                 end_channels, n_sequences, device, act_func, binary, dropout, num_layers):
        super(LSTM, self).__init__()

        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=1).to(device)

        self.lstm = torch.nn.LSTM(input_size=residual_channels, hidden_size=hidden_channels, num_layers=num_layers, dropout=dropout, batch_first=True).to(device)
        
        self.output = OutputLayer(in_channels=hidden_channels, end_channels=end_channels,
                                  n_steps=n_sequences, device=device, act_func=act_func,
                                  binary=binary)
        self.device = device
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

    def forward(self, X, edge_index=None):
        # edge Index is used for api facility but it is ignore
        batch_size = X.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(self.device)
        x = self.input(X)
        x = torch.movedim(x, 2, 1)
        x, _ = self.lstm(x, (h0, c0))
        x = torch.squeeze(x[:, -1, :])
        x = self.output(x)

        return x

############################### GraphSAGE ##################################
    
class GRAPH_SAGE(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
