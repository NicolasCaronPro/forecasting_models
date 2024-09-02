import sys

sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/models')

from forecasting_models.utils import *
from conv_lstm import ConvLSTM

class Zhang(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, fc_channels, dropout, binary, device, n_sequences, return_hidden=False):
        super(Zhang, self).__init__()
        torch.manual_seed(42)

        self.return_hidden = return_hidden

        self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=(1,1)).to(device)

        self.conv_list = []
        self.fc_list = []

        for i in range(len(conv_channels) - 1):
            self.conv_list.append(torch.nn.Conv2d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], kernel_size=(3,3), padding='same').to(device))

        self.pooling = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2).to(device)
        self.activation = ReLU().to(device)

        for i in range(len(fc_channels) - 1):
            self.fc_list.append(torch.nn.Linear(in_features=fc_channels[i], out_features=fc_channels[i+1]).to(device))

        self.last_linear = torch.nn.Linear(in_features=fc_channels[-1], out_features=2).to(device) if binary else torch.nn.Linear(in_features=fc_channels[-1], out_features=1).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drop = torch.nn.Dropout(dropout) if dropout > 0.0 else Identity().to(device)
        self.conv_list = torch.nn.ModuleList(self.conv_list)
        self.fc_list = torch.nn.ModuleList(self.fc_list)

        self.num_conv = len(self.conv_list)
        self.num_fc = len(self.fc_list)

        self.binary = binary
    
    def forward(self, X, edge_index):
        # egde index not used, API configuration
        # Zhang model doesn't take in account time series
        # (B, H, W, F, T) -> (B, H, W, F)

        x = X[:,:,:,:,-1]
        # (B, H, W, F) -> (B, F, H, W)
        #x = x.permute(0,3,1,2)

        # Bottleneck
        x = self.input(x)
        for i, layer in enumerate(self.conv_list):
            x = layer(x)
            if i != self.num_conv - 1:
                x = self.activation(x)
                x = self.pooling(x)

        x = x.reshape(x.shape[0], -1)

        for i, layer in enumerate(self.fc_list):
            x = layer(x)
            if i != self.num_fc - 1:
                x = self.drop(x)
                x = self.activation(x)

        x_linear = self.last_linear(x)
        if self.binary:
            output = self.softmax(x_linear)
        else:
            output = x_linear

        if self.return_hidden:
            return output, x
        else:
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
    

#################################### UNET ##########################################

class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

""" Full assembly of the parts to form the complete network """

class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

#################################### ConvGraphNet #######################################

class ConvGraphNet(torch.nn.Module):
    def __init__(self, cnn_model, gnn_model,
                 output_layer_in_channels, output_layer_end_channels, n_sequence, binary, device, act_func):
        super(ConvGraphNet, self).__init__()
        torch.manual_seed(42)

        self.cnn_layer = cnn_model
        self.gnn_layer = gnn_model

        self.output = OutputLayer(in_channels=output_layer_in_channels, end_channels=output_layer_end_channels, n_steps=n_sequence, binary=binary, device=device, act_func=act_func)
    
    def forward(self, gnn_X, cnn_X, edge_index):

        cnn_x = self.cnn_layer(cnn_X)
        gnn_x = self.gnn_layer(gnn_X, edge_index)
        
        x = torch.concat(cnn_x, gnn_x)

        output = self.output(x)

        return output
    

###########################################################################################

class HybridConvGraphNet(torch.nn.Module):
    def __init__(self, cnn_model, gnn_model,
                 output_layer_in_channels, output_layer_end_channels, n_sequence, binary, device, act_func):
        super(HybridConvGraphNet, self).__init__()
        torch.manual_seed(42)

        self.cnn_layer = cnn_model
        self.gnn_layer = gnn_model

        self.output = OutputLayer(in_channels=output_layer_in_channels, end_channels=output_layer_end_channels, n_steps=n_sequence, binary=binary, device=device, act_func=act_func)
    
    def forward(self, gnn_X, cnn_X, edge_index):

        cnn_x = self.cnn_layer(cnn_X)
        gnn_x = self.gnn_layer(cnn_x, edge_index)
        
        output = self.output(gnn_x)

        return output