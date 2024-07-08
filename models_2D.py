import sys

sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/models')

from utils import *
from conv_lstm import ConvLSTM

class Zhang(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, end_channels, dropout, binary, device, n_sequences):
        super(Zhang, self).__init__()
        torch.manual_seed(42)

        self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(1,1)).to(device)

        self.conv1 = torch.nn.Conv2d(hidden_channels, 64, kernel_size=(3,3), padding=1, stride=1).to(device)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=1).to(device)
        self.conv3 = torch.nn.Conv2d(128, end_channels, kernel_size=(3,3), padding=1, stride=1).to(device)

        self.pooling = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2).to(device)
        self.activation = ReLU().to(device)

        self.FC1 = torch.nn.Linear(in_features=end_channels * 6 * 6, out_features=128).to(device)
        self.FC2 = torch.nn.Linear(in_features=128, out_features=64).to(device)
        self.FC3 = torch.nn.Linear(in_features=64, out_features=32).to(device)

        self.FC4 = torch.nn.Linear(in_features=32, out_features=2 if binary else n_sequences).to(device)

        self.output = torch.nn.Softmax(dim=1) if binary else Identity().to(device)
        self.drop = torch.nn.Dropout(dropout) if dropout > 0.0 else Identity().to(device)
    
    def forward(self, X, edge_index):
        # egde index not used, API configuration
        # Zhang model doesn't take in account time series
        # (B, H, W, F, T) -> (B, H, W, F)

        x = X[:,:,:,:,-1]

        # (B, H, W, F) -> (B, F, H, W)
        x = x.permute(0,3,1,2)

        # Bottleneck
        x = self.input(x)

        # Series of convolution
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pooling(x)

        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)

        # Output
        x = self.FC1(x)
        x = self.drop(x)
        x = self.activation(x)

        x = self.FC2(x)
        x = self.drop(x)
        x = self.activation(x)
        
        x = self.FC3(x)
        x = self.drop(x)
        x = self.activation(x)

        x = self.FC4(x)

        output = self.output(x)

        return output
    
########################### ConvLTSM ####################################    

class CONVLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, end_channels, n_sequences, device, act_func, dropout, binary):
        super(CONVLSTM, self).__init__()

        num_layer = len(hidden_dim)
        self.device = device
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.convlstm = ConvLSTM(input_dim=in_channels,
                                hidden_dim=hidden_dim,
                                kernel_size=[3 for i in range(num_layer)] ,
                                num_layers=num_layer,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False).to(device)

        self.conv1 = torch.nn.Conv2d(hidden_dim[-1], 1, kernel_size=(3,3), padding=1, stride=1).to(device)
        
        self.output = OutputLayer(in_channels=hidden_dim[-1] * 25 * 25, end_channels=end_channels,
                        n_steps=n_sequences, device=device, act_func=act_func,
                        binary=binary)

    def forward(self, X, edge_index):
        # edge Index is used for api facility but it is ignore

        X = X.permute(0, 4, 3, 1, 2)
        x, _ = self.convlstm(X)
        x = x[0][:, -1, :, :]
        x = self.dropout(x)
        x = self.output(x)

        return x

########################### ST-GATCONVLSTM ####################################    

class ST_GATCONVLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, end_channels, n_sequences, device, act_func):
        super(ST_GATCONVLSTM, self).__init__()
        
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1).to(device)
        self.bn = torch.nn.BatchNorm1d(in_channels=hidden_channels)

        # TO DO

        self.output = OutputLayer(in_channels=hidden_channels, end_channels=end_channels, n_steps=n_sequences, device=device, act_func=act_func)

    def forward(self, X, edge_index):
        x = self.input(X)
        # TO DO
        x = self.output(x)
    
########################### ST-GATCONV2D ####################################

class ST_GATCONV2D(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, end_channels, n_sequences, device, act_func):
        super(ST_GATCONV2D, self).__init__()

        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1).to(device)
        self.bn = torch.nn.BatchNorm1d(in_channels=hidden_channels)

        # TO DO
        
        self.output = OutputLayer(in_channels=hidden_channels, end_channels=end_channels, n_steps=n_sequences, device=device, act_func=act_func)

    def forward(self, X, edge_index):
        x = self.input(X)
        # TO DO
        x = self.output(x)
        return x