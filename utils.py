import sys

sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/GNN/models')
sys.path.insert(0, '/home/caron/Bureau/pytorch_geometric_temporal')

import sys

sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/GNN/models')
sys.path.insert(0, '/home/caron/Bureau/pytorch_geometric_temporal')
import sys

sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/GNN/models')
sys.path.insert(0, '/home/caron/Bureau/pytorch_geometric_temporal')

import torch
from torch.nn import ELU, ReLU, Sigmoid, Softmax, Tanh, GELU, SiLU, Conv1d, Conv2d, MaxPool2d, Identity, Dropout
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import torch_geometric.nn as nn
from pytorch_tcn import TCN
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, GATv2Conv, GATConv
from torch_geometric.nn.sequential import Sequential
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric_temporal.nn.recurrent import A3TGCN

from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)

############################### Output ########################################

class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels, end_channels, n_steps, device, act_func, binary):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(in_channels=in_channels, out_channels=end_channels, weight_initializer='glorot', bias=True).to(device)
        if act_func == 'relu':
            self.activation = ReLU()
        if act_func == 'gelu':
            self.activation = GELU()

        self.binary = binary

        if binary:
            self.out_channels = 2
            self.softmax = Softmax()
        else:
            self.out_channels = 1

        self.fc2 = nn.Linear(in_channels=end_channels, out_channels=self.out_channels, weight_initializer='glorot', bias=True).to(device)
        self.n_steps = n_steps
        self.output = ReLU()

    def forward(self, x):
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.output(x)
        x = x.view(x.shape[0], self.out_channels)
        if self.binary:
            x = self.softmax(x)
        return x