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

        self.fc = nn.Linear(in_channels=in_channels, out_channels=end_channels, weight_initializer='glorot', bias=True).to(device)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_channels=end_channels, out_channels=self.out_channels, weight_initializer='glorot', bias=True).to(device)
        self.n_steps = n_steps

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], self.out_channels)
        x = torch.clamp(x, min=0)
        if self.binary:
            x = self.softmax(x)
        return x
    

####################################### Output GCN #####################################

class OutputLayerGCN(torch.nn.Module):
    def __init__(self, in_channels, end_channels, n_steps, device, act_func, binary):
        super(OutputLayerGCN, self).__init__()
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

        self.fc = nn.GCNConv(in_channels=in_channels, out_channels=end_channels, bias=True).to(device)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = nn.GCNConv(in_channels=end_channels, out_channels=self.out_channels, bias=True).to(device)
        self.n_steps = n_steps

    def forward(self, x, edge_index):
        x = x.view(x.shape[0], -1)
        x = self.activation(x)
        x = self.fc(x, edge_index)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x, edge_index)
        x = x.view(x.shape[0], self.out_channels)
        x = torch.clamp(x, min=0)
        if self.binary:
            x = self.softmax(x)
        return x
    
############################### Output GAT ##########################################

class OutputLayerGAT(torch.nn.Module):
    def __init__(self, in_channels, end_channels, n_steps, device, act_func, binary):
        super(OutputLayerGAT, self).__init__()
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

        self.fc = nn.GATConv(in_channels=in_channels, out_channels=end_channels, concat=False, heads=6, bias=True).to(device)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = nn.GATConv(in_channels=end_channels, out_channels=self.out_channels, concat=False, heads=6, bias=True).to(device)
        self.n_steps = n_steps

    def forward(self, x, edge_index):
        x = x.view(x.shape[0], -1)
        x = self.activation(x)
        x = self.fc(x, edge_index)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x, edge_index)
        x = x.view(x.shape[0], self.out_channels)
        x = torch.clamp(x, min=0)
        if self.binary:
            x = self.softmax(x)
        return x

######################################## TIME2VEC ######################################

class Time2Vec(torch.nn.Module):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size
        # Linear component parameters
        self.wb = torch.nn.Parameter(torch.Tensor(1))
        self.bb = torch.nn.Parameter(torch.Tensor(1))
        # Periodic component parameters
        self.wa = torch.nn.Parameter(torch.Tensor(kernel_size))
        self.ba = torch.nn.Parameter(torch.Tensor(kernel_size))

    def forward(self, x):
        # x has shape (batch_size, time_steps, 1)
        # Linear component computation
        v_linear = self.wb * x + self.bb  # Shape: (batch_size, time_steps, 1)
        # Periodic components computation
        v_periodic = torch.sin(x * self.wa + self.ba)  # Shape: (batch_size, time_steps, kernel_size)
        # Concatenate linear and periodic components
        return torch.cat([v_linear, v_periodic], dim=-1)  # Shape: (batch_size, time_steps, kernel_size + 1)