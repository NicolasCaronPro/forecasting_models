import torch
import torch.nn.functional as F
import math
from forecasting_models.pytorch.utils import *
from forecasting_models.pytorch.models import *
from torch.nn import LSTMCell
#from kan import KANLayer, MultKAN
"""
class KANnetwork(MultKAN):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        end_channels,
        k_days,
        device,
        binary,
        act_func,
        args
    ):
        super(KANnetwork, self).__init__(**args)
        self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels[0], kernel_size=1)
        self.layers = torch.nn.ModuleList()
        
        self.output = OutputLayer(in_channels=hidden_channels[-1],
                                  end_channels=end_channels,
                                  n_steps=k_days,
                                  device=device, act_func=act_func,
                                  binary=binary)

    def forward(self, x: torch.Tensor, edges_index : None):
        
        x = self.input(x)
        
        if len(x.shape) == 3:
            x = x[:, :, -1]
        
        x = super(KANnetwork, self).forward(x)
        x = self.output(x)
              
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
        
###############################################################################################################################
#                                                Temporal KAN                                                                 #
#                                                                                                                             #
#                                          https://github.com/remigenet/TKAN/blob/main/                                       #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################


class TKANCELL(torch.nn.Module):
    def __init__(self, input_size, hidden_size, base_activation, kan_config):
        super(TKANCELL, self).__init__()

        self.weight_ih = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.weight_hh = torch.nn.Parameter(torch.empty((4 * hidden_size, hidden_size)))
        self.weight_ig = torch.nn.Parameter(torch.empty((4 * hidden_size, hidden_size)))
        self.weight_io = torch.nn.Parameter(torch.empty((4 * hidden_size, hidden_size)))
        
        self.b_ih = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.b_hh = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.b_ig = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.b_io = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        
        self.hidden_size = hidden_size
        self.kanlayer = KANLinear(in_features=input_size,
                                  out_features=hidden_size,
                                  base_activation=base_activation, **kan_config)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialisation selon Glorot pour les poids
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.xavier_uniform_(self.weight_hh)
        torch.nn.init.xavier_uniform_(self.weight_ig)
        torch.nn.init.xavier_uniform_(self.weight_io)
        
        # Initialisation des biais à zéro
        torch.nn.init.zeros_(self.b_ih)
        torch.nn.init.zeros_(self.b_hh)
        torch.nn.init.zeros_(self.b_ig)
        torch.nn.init.zeros_(self.b_io)
        
        # Initialisation des paramètres de la couche KAN
        self.kanlayer.reset_parameters()

    def forward(self, X, hx):
        if hx is None:
            hx = torch.zeros(X.size(0), self.hidden_size, device=X.device)
            cx = torch.zeros(X.size(0), self.hidden_size, device=X.device)
        else:
            hx, cx = hx

        forgetgate = F.linear(X, self.weight_ih, self.b_ih)
        ingate = F.linear(hx, self.weight_hh, self.b_hh)
        kan_input = F.linear(X, self.weight_io, self.b_io)
        outgate = self.kanlayer(kan_input)
        cellgate = F.linear(cx, self.weight_ig, self.b_ig)
        
        ingate = torch.nn.Sigmoid(ingate)
        forgetgate = torch.nn.Sigmoid(forgetgate)
        cellgate = torch.nn.Tanh(cellgate)
        outgate = torch.nn.Sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy
    
class TKAN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, end_channels, act_func, dropout, binary, k_days, return_hidden, device, kan_config):
        super(TKAN, self).__init__()

        self.return_hidden = return_hidden
        if act_func == 'gelu':
            base_activation = torch.nn.GELU
        elif act_func == 'relu':
            base_activation = torch.nn.ReLU
        elif act_func == 'silu':
            base_activation = torch.nn.SiLU

        self.hidden_channels = hidden_size[0]

        self.input = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size[0], kernel_size=1)
        self.num_tkan_layer = len(hidden_size)
        self.tkan_layers = torch.nn.ModuleList()
        for i in range(self.num_tkan_layer - 1):
            tkan = TKANCELL(input_size=hidden_size[0], hidden_size=hidden_size[i+1], base_activation=base_activation, kan_config=kan_config)
            self.tkan_layers.append(tkan)

        self.output = OutputLayer(in_channels=hidden_size[-1], end_channels=end_channels,
                                  n_steps=k_days, device=device, act_func=act_func,
                                  binary=binary)
        
        self.batchNorm = nn.BatchNorm(hidden_size[-1]).to(device)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, X, edges):

        batch_size = X.size(0)
        h0 = torch.zeros(self.num_tkan_layer, batch_size, self.hidden_channels, device=X.device)
        c0 = torch.zeros(self.num_tkan_layer, batch_size, self.hidden_channels, device=X.device)
        x = self.input(X)
        x = torch.movedim(x, 2, 1)

        for layer in self.tkan_layers:
            h0, c0 = layer(x, (h0, c0))

        x = torch.squeeze(x[:, -1, :])
        x = self.batchNorm(x)
        x = self.dropout(x)
        output = self.output(x)

        if self.return_hidden:
            return output, x
        else:
            return output
"""