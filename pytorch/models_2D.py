import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

from forecasting_models.pytorch.utils import *
from forecasting_models.pytorch.conv_lstm import *

import torch
from torch.nn import ReLU, Identity

class Zhang(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, fc_channels, dropout, device, n_sequences, return_hidden=False, out_channels=None, task_type='classification', horizon=0):
        super(Zhang, self).__init__()
        torch.manual_seed(42)

        self.input_batch_norm = torch.nn.BatchNorm2d(in_channels).to(device)

        self.return_hidden = return_hidden
        self.horizon = horizon

        self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=(1,1)).to(device)

        self.conv_list = []
        self.batch_norm_list = []
        self.fc_list = []

        for i in range(len(conv_channels) - 1):
            self.conv_list.append(torch.nn.Conv2d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], kernel_size=(3,3), padding='same').to(device))
            self.batch_norm_list.append(torch.nn.BatchNorm2d(conv_channels[i+1]).to(device))

        self.pooling = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2).to(device)
        self.activation = ReLU().to(device)

        for i in range(len(fc_channels) - 1):
            self.fc_list.append(torch.nn.Linear(in_features=fc_channels[i], out_features=fc_channels[i+1]).to(device))

        self.last_linear = torch.nn.Linear(in_features=fc_channels[-1], out_features=out_channels).to(device) if out_channels is not None else torch.nn.Linear(in_features=fc_channels[-1], out_features=1).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drop = torch.nn.Dropout(dropout) if dropout > 0.0 else Identity().to(device)
        
        self.conv_list = torch.nn.ModuleList(self.conv_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        self.fc_list = torch.nn.ModuleList(self.fc_list)

        self.num_conv = len(self.conv_list)
        self.num_fc = len(self.fc_list)

        self.task_type = task_type
        self.out_channels = out_channels
        self.is_graph_or_node = False
    
    def forward(self, X, edge_index=None, graphs=None):
        # egde index not used, API configuration

        x = X[:,:,:,:,-1]

        # Bottleneck
        x = self.input_batch_norm(x)
        x = self.input(x)
        for i, layer in enumerate(self.conv_list):
            x = layer(x)
            x = self.batch_norm_list[i](x)  # Apply BatchNorm2D after convolution
            if i != self.num_conv - 1:
                x = self.activation(x)
                x = self.pooling(x)
        
        x = x.reshape(x.shape[0], -1)

        for i, layer in enumerate(self.fc_list):
            x = layer(x)
            if i != self.num_fc - 1:
                x = self.drop(x)
                x = self.activation(x)

        logits = self.last_linear(x)

        if self.task_type == 'classification':
            output = self.softmax(logits)
        else:
            output = logits

        hidden = x
        return output, logits, hidden

########################### ConvLTSM ####################################

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1,downsampling=False, expansion = 4, device=None):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False).to(device),
            nn.BatchNorm2d(places).to(device),
            nn.ReLU(inplace=False).to(device),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False).to(device),
            nn.BatchNorm2d(places).to(device),
            nn.ReLU(inplace=False).to(device),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False).to(device),
            nn.BatchNorm2d(places*self.expansion).to(device),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False).to(device),
                nn.BatchNorm2d(places*self.expansion).to(device)
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, fc_channels, dropout, device,
                 n_sequences, avgpooling=1, return_hidden=False,
                 out_channels=None, task_type='classification', horizon=0):

        super(ResNet, self).__init__()
        torch.manual_seed(42)

        self.input_batch_norm = torch.nn.BatchNorm2d(in_channels).to(device)

        self.return_hidden = return_hidden
        self.horizon = horizon

        self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=(1,1)).to(device)

        self.bottleneck = []
        self.batch_norm_list = []
        self.fc_list = []

        for i in range(len(conv_channels) - 1):
            self.bottleneck.append(Bottleneck(conv_channels[i], conv_channels[i + 1], stride=1, downsampling=True, expansion=1, device=device))

        self.activation = ReLU().to(device)
        self.pooling = torch.nn.AdaptiveAvgPool2d(avgpooling)

        for i in range(len(fc_channels) - 1):
            self.fc_list.append(torch.nn.Linear(in_features=fc_channels[i], out_features=fc_channels[i+1]).to(device))

        self.last_linear = torch.nn.Linear(in_features=fc_channels[-1], out_features=out_channels).to(device) if out_channels is not None else torch.nn.Linear(in_features=fc_channels[-1], out_features=1).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drop = torch.nn.Dropout(dropout) if dropout > 0.0 else Identity().to(device)

        self.bottleneck = torch.nn.ModuleList(self.bottleneck)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        self.fc_list = torch.nn.ModuleList(self.fc_list)

        self.num_conv = len(self.bottleneck)
        self.num_fc = len(self.fc_list)

        self.task_type = task_type
        self.out_channels = out_channels
        self.is_graph_or_node = False

    def forward(self, X, edge_index=None, graphs=None):
        # egde index not used, API configuration
        x = X[:,:,:,:,-1]

        x = x.permute(0,3,1,2)

        # Bottleneck
        x = self.input_batch_norm(x)
        x = self.input(x)
        for i, layer in enumerate(self.bottleneck):
            x = layer(x)
        
        x = self.pooling(x)
        x = x.reshape(x.shape[0], -1)

        for i, layer in enumerate(self.fc_list):
            x = layer(x)
            x = self.drop(x)
            x = self.activation(x)

        logits = self.last_linear(x)

        if self.task_type == 'classification':
            output = self.softmax(logits)
        else:
            output = logits

        hidden = x
        return output, logits, hidden

########################### ConvLTSM ####################################    

class CONVLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, end_channels, size, n_sequences, device, act_func, dropout, out_channels=None, task_type='classification', return_hidden=False, horizon=0):
        super(CONVLSTM, self).__init__()

        self.input_batch_norm = torch.nn.BatchNorm3d(in_channels).to(device)
        num_layer = len(hidden_dim)
        self.device = device
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.input = torch.nn.Conv3d(in_channels=in_channels, out_channels=hidden_dim[0], kernel_size=(1, 1, 1)).to(device)

        self.convlstm = ConvLSTM(input_dim=hidden_dim[0],
                                hidden_dim=hidden_dim,
                                kernel_size=[(3, 3, 3) for i in range(num_layer)],
                                num_layers=num_layer,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False).to(device)
        
        self.output = OutputLayer(in_channels=hidden_dim[-1] * size[0] * size[1], end_channels=end_channels,
                        n_steps=n_sequences, device=device, act_func=act_func,
                        task_type=task_type, out_channels=out_channels)

        self.task_type = task_type
        self.return_hidden = return_hidden
        self.horizon = horizon

    def forward(self, X, edge_index=None):
        # edge Index is used for api facility but it is ignore
        x = X.permute(0, 3, 1, 2, 4)
        x = self.input_batch_norm(x)
        x = self.input(x)
        x = x.permute(0, 4, 1, 2, 3)
        x, _ = self.convlstm(x)
        x = x[0][:, -1, :, :]
        x = self.dropout(x)
        hidden = x
        logits = self.output(x)
        output = logits

        return output, logits, hidden

########################### ST-GATCONVLSTM ####################################    

class ST_GATCONVLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, end_channels, n_sequences, device, act_func, return_hidden=False, horizon=0):
        super(ST_GATCONVLSTM, self).__init__()
        
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1).to(device)
        self.bn = torch.nn.BatchNorm1d(in_channels=hidden_channels)

        # TO DO

        self.output = OutputLayer(in_channels=hidden_channels, end_channels=end_channels, n_steps=n_sequences, device=device, act_func=act_func)
        self.horizon = horizon

    def forward(self, X, edge_index=None):
        x = self.input(X)
        # TO DO
        hidden = x
        logits = self.output(x)
        output = logits
        return output, logits, hidden

########################### ST-GATCONV2D ####################################

class ST_GATCONV2D(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, end_channels, n_sequences, device, act_func, return_hidden=False, horizon=0):
        super(ST_GATCONV2D, self).__init__()

        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1).to(device)
        self.bn = torch.nn.BatchNorm1d(in_channels=hidden_channels)

        # TO DO
        
        self.output = OutputLayer(in_channels=hidden_channels, end_channels=end_channels, n_steps=n_sequences, device=device, act_func=act_func)
        self.horizon = horizon
        self.return_hidden = return_hidden

    def forward(self, X, edge_index=None):
        x = self.input(X)
        # TO DO
        hidden = x
        logits = self.output(x)
        output = logits
        return output, logits, hidden
    

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
       
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
      

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_channels, out_channels, conv_channels, bilinear=False, return_hidden=False, horizon=0, task_type='classification'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, conv_channels[0])

        self.downs = torch.nn.ModuleList()
        for idx in range(len(conv_channels) - 1):
            self.downs.append(Down(conv_channels[idx], conv_channels[idx + 1]))

        self.ups = torch.nn.ModuleList()
        for idx in range(len(conv_channels) - 1, 0, -1):
            self.ups.append(Up(conv_channels[idx], conv_channels[idx - 1], bilinear))

        self.outc = OutConv(conv_channels[0], out_channels)
        self.is_graph_or_node = False
        self.return_hidden = return_hidden
        self.horizon = horizon

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()  # For regression or custom handling

    def forward(self, x, edge_index=None, graph=None):
        if len(x.shape) == 5:
            x = x[:, :, :, :, -1]
            
        x = self.inc(x)
        skip_connections = [x]

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        x = skip_connections.pop()

        for up in self.ups:
            x_skip = skip_connections.pop()
            x = up(x, x_skip)

        hidden = x
        logits = self.outc(x)
        output = self.output_activation(logits)

        return output, logits, hidden
    
#################################### ULSTM #############################################

class ULSTM(torch.nn.Module):
    def __init__(self, n_channels, n_classes, n_sequences, num_lstm_layers, features, bilinear=False, return_hidden=False, horizon=0):
        super(ULSTM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input_batch_norm = torch.nn.BatchNorm2d(in_channels)
        self.inc = DoubleConv(n_channels, features[0])

        self.downs = torch.nn.ModuleList()
        for idx in range(len(features) - 1):
            self.downs.append(Down(features[idx], features[idx + 1]))

        self.lstm_layer = ConvLSTM(input_dim=features[-1],
                                hidden_dim=features[-1],
                                kernel_size=[3 for i in range(num_lstm_layers)] ,
                                num_layers=num_lstm_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)

        self.ups = torch.nn.ModuleList()
        for idx in range(len(features) - 1, 0, -1):
            self.ups.append(Up(features[idx], features[idx - 1], bilinear))

        self.outc = OutConv(features[0], n_classes)
        self.return_hidden = return_hidden
        self.horizon = horizon

    def forward(self, x, edge_index=None):

        x = x[:, :, :, :, -1]
        x = self.input_batch_norm(x)
        x = self.inc(x)
        skip_connections = [x]

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        x = skip_connections.pop()
        x = self.lstm_layer(x)

        for up in self.ups:
            x_skip = skip_connections.pop()
            x = up(x, x_skip)

        hidden = x
        logits = self.outc(x)
        output = logits
        return output, logits, hidden

#################################### ConvGraphNet #######################################

class ConvGraphNet(torch.nn.Module):
    def __init__(self, cnn_model, gnn_model,
                 output_layer_in_channels, output_layer_end_channels, n_sequence, binary, device, act_func, return_hidden=False, horizon=0):
        super(ConvGraphNet, self).__init__()
        torch.manual_seed(42)

        self.cnn_layer = cnn_model
        self.gnn_layer = gnn_model

        self.output = OutputLayer(in_channels=output_layer_in_channels, end_channels=output_layer_end_channels,
                                  n_steps=n_sequence, binary=binary, device=device, act_func=act_func)
        self.return_hidden = return_hidden
        self.horizon = horizon
    
    def forward(self, gnn_X, cnn_X, edge_index):

        cnn_x = self.cnn_layer(cnn_X, edge_index)
        gnn_x = self.gnn_layer(gnn_X, edge_index)
        
        x = torch.concat(cnn_x, gnn_x)

        hidden = x
        logits = self.output(x)
        output = logits

        return output, logits, hidden

###########################################################################################

from forecasting_models.pytorch.swin_unet import SwinTransformerSys
from forecasting_models.pytorch.tbconvnet import (
    EncoderBlock, DenseSepBlock, Reduce1x1,
    SkipFusion, UpBlock,
)

###########################################################################################
#################################### SwinUNet #############################################
###########################################################################################

class SwinUnet(nn.Module):
    """
    Wrapper around SwinTransformerSys pour l'interface du pipeline.

    forward(x) -> (output, logits, hidden)
      - output  : (B, num_classes, H, W)  après activation
      - logits  : (B, num_classes, H, W)  avant activation
      - hidden  : (B, L, C)  feature map du bottleneck (avant upsampling final)

    Accepte x de forme (B, C, H, W) ou (B, C, H, W, T) — dernier pas de temps utilisé.
    """

    def __init__(self, img_size=224, patch_size=4, in_channels=3,
                 embed_dim=96, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 depths_decoder=(1, 2, 2, 2), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 num_classes=1, zero_head=False, vis=False,
                 task_type='classification', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.task_type = task_type
        self.is_graph_or_node = False
        self.return_hidden = False
        self.horizon = 0

        self.swin = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        if task_type == 'binary':
            self.output_activation = nn.Sigmoid()
        elif task_type == 'classification':
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = nn.Identity()

        if zero_head:
            nn.init.zeros_(self.swin.output.weight)
            if self.swin.output.bias is not None:
                nn.init.zeros_(self.swin.output.bias)

    def forward(self, x, edge_index=None, graph=None):
        # Gère la dim temporelle (B, C, H, W, T)
        if x.dim() == 5:
            x = x[:, :, :, :, -1]

        # SwinTransformerSys.forward retourne (logits, hidden)
        # logits : (B, num_classes, H, W)
        # hidden : (B, L, C) — tokens du bottleneck avant final expand
        logits, hidden = self.swin(x)

        output = self.output_activation(logits)
        return output, logits, hidden


###########################################################################################
################################# TBConvResNetNet #########################################
###########################################################################################

class TBConvResNetNet(nn.Module):
    """
    U-Net hybride = encodeur ConvNet (EncoderBlock) +
                    skip connections avec attention Swin (SkipFusion) +
                    décodeur (UpBlock).

    Utilise les blocs définis dans tbconvnet.py.

    forward(x) -> (output, logits, hidden)
      - output  : (B, H, W, num_classes)  après activation
      - logits  : (B, H, W, num_classes)  avant activation
      - hidden  : (B, ch*8)  vecteur de bottleneck

    Accepte x de forme (B, C, H, W) ou (B, C, H, W, T).
    """

    def __init__(self, in_chans=3, num_classes=1, base_ch=32, img_size=64,
                 window_size=4, heads=(2, 4, 8), task_type='classification',
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop=0., drop_rate_path=0.,
                 act_layer='GELU', norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.task_type = task_type
        self.is_graph_or_node = False
        self.return_hidden = False
        self.horizon = 0

        ch = base_ch
        # img_size au niveau de chaque état d'encodeur
        h0, h1, h2 = img_size, img_size // 2, img_size // 4

        # -------- Stem --------
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # -------- Encodeur --------
        self.enc1 = EncoderBlock(ch, ch)          # out: ch,  skip: (h0,h0)
        self.enc2 = EncoderBlock(ch, ch * 2)      # out: ch*2, skip: (h1,h1)
        self.enc3 = EncoderBlock(ch * 2, ch * 4)  # out: ch*4, skip: (h2,h2)

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            DenseSepBlock(ch * 4),          # (B, ch*8, h2/2, h2/2)
            Reduce1x1(ch * 8, ch * 8),
        )

        # -------- Skip attention (optionnel, résolution fixe) --------
        self.skip3 = SkipFusion(
            ch=ch * 4, token_hw=(h2, h2),
            num_heads=heads[2], window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=drop_rate_path,
        )
        self.skip2 = SkipFusion(
            ch=ch * 2, token_hw=(h1, h1),
            num_heads=heads[1], window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=drop_rate_path,
        )
        self.skip1 = SkipFusion(
            ch=ch, token_hw=(h0, h0),
            num_heads=heads[0], window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=drop_rate_path,
        )

        # -------- Décodeur --------
        self.up3 = UpBlock(in_ch=ch * 8, skip_ch=ch * 4, out_ch=ch * 4)
        self.up2 = UpBlock(in_ch=ch * 4, skip_ch=ch * 2, out_ch=ch * 2)
        self.up1 = UpBlock(in_ch=ch * 2, skip_ch=ch,     out_ch=ch)

        # -------- Tête de classification/régression --------
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

        if task_type == 'binary':
            self.output_activation = nn.Sigmoid()
        elif task_type == 'classification':
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = nn.Identity()

    def forward(self, x, edge_index=None, graph=None):
        # Gère la dim temporelle (B, C, H, W, T)
        if x.dim() == 5:
            x = x[:, :, :, :, -1]

        # Stem
        x = self.stem(x)

        # Encodeur
        s1, x = self.enc1(x)   # s1: (B, ch,   H, W)
        s2, x = self.enc2(x)   # s2: (B, ch*2, H/2, W/2)
        s3, x = self.enc3(x)   # s3: (B, ch*4, H/4, W/4)

        # Bottleneck
        x = self.bottleneck(x)          # (B, ch*8, H/8, W/8)
        hidden = x.flatten(2).mean(-1)  # (B, ch*8)

        # Skip attention
        s3 = self.skip3(s3)
        s2 = self.skip2(s2)
        s1 = self.skip1(s1)

        # Décodeur
        x = self.up3(x, s3)   # (B, ch*4, H/4, W/4)
        x = self.up2(x, s2)   # (B, ch*2, H/2, W/2)
        x = self.up1(x, s1)   # (B, ch,   H,   W)

        # Tête : (B, num_classes, H, W) -> (B, H, W, num_classes)
        logits = self.head(x).permute(0, 2, 3, 1).contiguous()
        output = self.output_activation(logits)

        return output, logits, hidden


