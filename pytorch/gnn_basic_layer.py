import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.gcn_msg = fn.copy_u(u="h", out="m")
        self.gcn_reduce = fn.sum(msg="m", out="h")
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata["h"] = feature
            g.update_all(self.gcn_msg, self.gcn_reduce)
            h = g.ndata["h"]
            return self.linear(h)

class MLPLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim=64):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(in_feats, hidden_dim)
    
    def forward(self, x):
        return self.mlp(x)