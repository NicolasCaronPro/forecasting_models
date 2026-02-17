import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

from forecasting_models.pytorch.gnns import *
from forecasting_models.pytorch.utils import corn_class_probs

class SpatialContext(nn.Module):
    """
    Cross-attention: dynamic representation -> static variables

    Inputs:
      h_dyn  : [B, D_dyn]
      x_stat : [B, K, D_stat]  (K static "variables" encodées séparément)

    Outputs:
      c_stat    : [B, d_model]   (static context conditionné par h_dyn)
      attn_mean : [B, K]         (poids d'attention par variable statique)
    """

    def __init__(self, d_model: int, d_dyn: int, d_stat: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Project to shared model dimension
        self.dyn_in = nn.Linear(d_dyn, d_model)
        self.stat_in = nn.Linear(d_stat, d_model)

        # Attention projections in model space
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)

    def forward(self, h_dyn: torch.Tensor, x_stat: torch.Tensor):
        """
        h_dyn  : [B, D_dyn]
        x_stat : [B, K, D_stat]
        """
        B, K, _ = x_stat.shape

        # Map to shared space
        h = self.dyn_in(h_dyn)          # [B, d_model]
        s = self.stat_in(x_stat)        # [B, K, d_model]

        # Optional pre-norm (stabilise training)
        h = self.norm_h(h)
        s = self.norm_s(s)

        # Projections
        q = self.q_proj(h).view(B, self.n_heads, self.d_head)              # [B, H, Dh]
        k = self.k_proj(s).view(B, K, self.n_heads, self.d_head)           # [B, K, H, Dh]
        v = self.v_proj(s).view(B, K, self.n_heads, self.d_head)

        # Reorder for attention
        q = q.unsqueeze(2)                     # [B, H, 1, Dh]
        k = k.permute(0, 2, 1, 3)              # [B, H, K, Dh]
        v = v.permute(0, 2, 1, 3)              # [B, H, K, Dh]

        # Attention weights over K static variables
        attn_logits = (q * k).sum(-1) * self.scale   # [B, H, K]
        attn = F.softmax(attn_logits, dim=-1)        # [B, H, K]
        attn = self.dropout(attn)

        # Weighted sum of static values -> context per head
        c = (attn.unsqueeze(-1) * v).sum(dim=2)      # [B, H, Dh]

        # Merge heads
        c = c.reshape(B, self.d_model)               # [B, d_model]
        c = self.out_proj(c)                         # [B, d_model]

        # Mean over heads for interpretability
        attn_mean = attn.mean(dim=1)                 # [B, K]

        return c, attn_mean
    

class SpatialContextSet(nn.Module):
    """
    Cross-attn/gating: h_dyn (query) -> x_stat (keys/values), BUT we keep per-variable
    representations [B,K,d_model] before pooling with an MLP (DeepSets style).

    Inputs:
      h_dyn  : [B, D_dyn]
      x_stat : [B, K, D_stat]

    Outputs:
      c        : [B, d_model]   pooled static context conditioned on h_dyn
      w_mean   : [B, K]         mean weights over heads (interpretable)
      z_tokens : [B, K, d_model] (optional, can be useful for debugging)
    """

    def __init__(
        self,
        d_model: int,
        d_dyn: int,
        d_stat: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_sigmoid_gating: bool = True,
        renorm_gates: bool = True,   # recommended when K is large (e.g., 61)
        return_tokens: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.use_sigmoid_gating = use_sigmoid_gating
        self.renorm_gates = renorm_gates
        self.return_tokens = return_tokens

        # Project to shared model dimension
        self.dyn_in = nn.Linear(d_dyn, d_model)
        self.stat_in = nn.Linear(d_stat, d_model)

        # Separate norms (often better than sharing one LN for both streams)
        self.norm_dyn = nn.LayerNorm(d_model)
        self.norm_stat = nn.LayerNorm(d_model)

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Per-variable MLP (shared across K) before pooling: DeepSets style
        self.var_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Output projection and dropout
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Optional residual scaling (start small = safer)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h_dyn: torch.Tensor, x_stat: torch.Tensor):
        """
        h_dyn  : [B, D_dyn]
        x_stat : [B, K, D_stat]
        """
        B, K, _ = x_stat.shape

        # Shared space + norms
        h0 = self.dyn_in(h_dyn)          # [B, d_model]
        s = self.stat_in(x_stat)         # [B, K, d_model]
        h = self.norm_dyn(h0)
        s = self.norm_stat(s)

        # Projections
        q = self.q_proj(h).view(B, self.n_heads, self.d_head)              # [B, H, Dh]
        k = self.k_proj(s).view(B, K, self.n_heads, self.d_head)           # [B, K, H, Dh]
        v = self.v_proj(s).view(B, K, self.n_heads, self.d_head)           # [B, K, H, Dh]

        # Reorder
        q = q.unsqueeze(2)                     # [B, H, 1, Dh]
        k = k.permute(0, 2, 1, 3)              # [B, H, K, Dh]
        v = v.permute(0, 2, 1, 3)              # [B, H, K, Dh]

        # Logits
        logits = (q * k).sum(-1) * self.scale  # [B, H, K]

        # Weights/gates
        if self.use_sigmoid_gating:
            w = torch.sigmoid(logits)          # [B, H, K]  (multi-label style)
            if self.renorm_gates:
                w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            w = F.softmax(logits, dim=-1)      # [B, H, K]  (classic attention)

        w_drop = self.dropout(w)

        # Keep per-variable tokens BEFORE pooling:
        # z_tokens_head: [B, H, K, Dh]
        z_tokens_head = w_drop.unsqueeze(-1) * v

        # Merge heads -> [B, K, d_model]
        z_tokens = z_tokens_head.permute(0, 2, 1, 3).contiguous().view(B, K, self.d_model)

        # Per-variable nonlinearity then pool across K
        z_tokens = self.var_mlp(z_tokens)      # [B, K, d_model]
        c = z_tokens.mean(dim=1)               # [B, d_model]  (or .sum / max / attn-pool)

        c = self.out_proj(c)
        c = self.dropout(c)

        # Optional residual update on h_dyn representation (often helps stability)
        c = h0 + self.res_scale * c            # [B, d_model]

        # For interpretability (use w, not w_drop)
        w_mean = w.mean(dim=1)                 # [B, K]

        if self.return_tokens:
            return c, w_mean, z_tokens
        return c, w_mean    

class MLPLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, device):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(in_feats, hidden_dim, weight_initializer='glorot', bias=True, bias_initializer='zeros').to(device)
        #self.mlp = torch.nn.Linear(in_feats, hidden_dim).to(device)
    def forward(self, x):
        return self.mlp(x)
    
class NetMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, end_channels, output_channels, n_sequences, device, task_type, return_hidden=False, horizon=0, **kwargs):
        super(NetMLP, self).__init__()
        self.layer1 = MLPLayer(in_dim * n_sequences + end_channels, hidden_dim[0], device) if horizon > 0 else MLPLayer(in_dim * n_sequences, hidden_dim[0], device)
        self.layer3 = MLPLayer(hidden_dim[0], hidden_dim[1], device)
        self.layer4 = MLPLayer(hidden_dim[1], end_channels, device)
        self.layer2 = MLPLayer(end_channels, output_channels, device)
        self.task_type = task_type
        self.n_sequences = n_sequences
        self.soft = torch.nn.Softmax(dim=1)
        self.return_hidden = return_hidden
        self.device = device
        self.end_channels = end_channels
        self.output_channels = output_channels
        self.in_dim = in_dim
        self.n_sequences = self.n_sequences
        
        #if self.horizon > 0:
        #    self.define_horizon_decodeur()

    def forward(self, features, z_prev=None, edges=None):
        if self.horizon > 0:
            if z_prev is None:
                z_prev = torch.zeros((features.shape[0], self.end_channels * self.n_sequences))

        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)
        
        if self.horizon > 0:
            features = torch.cat((features, z_prev), dim=1)

        x = F.relu(self.layer1(features))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        hidden = x
        logits = self.layer2(x)
        if self.task_type == 'classification':
            output = self.soft(logits)
        elif self.task_type == 'corn':
            output = corn_class_probs(logits)
        else:
            output = logits

        return output, logits, hidden

class GRU(torch.nn.Module):
    def __init__(self, in_channels, gru_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.0, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False, horizon=0,
                 temporal_idx=None, static_idx=None, spatialContext=False, d_channels=16
                 ):
        
        super(GRU, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.gru_size = gru_size
        self.end_channels = end_channels
        self.n_sequences = n_sequences
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels
        self.temporal_idx = temporal_idx
        self.spatial_idx = static_idx
        self.spatialContext = spatialContext
        
        g_cha = len(self.temporal_idx)
        
        # GRU layer
        self.gru = torch.nn.GRU(
            input_size=g_cha + self.end_channels if horizon > 0 else g_cha,
            hidden_size=gru_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gru_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gru_size).to(device)

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        
        if self.spatialContext:
            self.context_layer = SpatialContextSet(d_channels, gru_size, 1, n_heads=4, dropout=dropout, use_sigmoid_gating=True, renorm_gates=True,)

        # Output linear layer
        self.linear1 = torch.nn.Linear(d_channels if self.spatialContext else gru_size + len(self.spatial_idx), hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)
        
        # Custom Init to prevent "Inverted" start:
        # Start with near-zero weights to ensure initial neutrality (deltas ~ 0)
        print("Applying custom neutral initialization to GRU output layer.")
        torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.001)
        if self.output_layer.bias is not None:
            torch.nn.init.zeros_(self.output_layer.bias)


        # Activation functions - separate instances for SHAP compatibility
        self.act_func1 = getattr(torch.nn, act_func)()
        self.act_func2 = getattr(torch.nn, act_func)()

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling
            
    def forward(self, X, edge_index=None, graphs=None, z_prev=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """

        if hasattr(self, 'temporal_idx') and self.temporal_idx is not None and hasattr(self, 'spatial_idx'):
            X_spa = X[:, self.spatial_idx, -1][:, :, None]
            X = X[:, self.temporal_idx, :]
        
        batch_size = X.size(0)
        
        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            X = torch.cat((X, z_prev), dim=1)

        # Reshape to (batch, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.gru_size).to(self.device)

        # GRU forward
        x, _ = self.gru(x, h0)

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        if hasattr(self, 'spatialContext') and self.spatialContext:
            x, a = self.context_layer(x, X_spa)
            self.last_attention_coef = a
        elif hasattr(self, 'spatialContext'):
            x = torch.concat((x, X_spa[:, :, 0]), dim=1)
            self.last_attention_coef = None
            
        # Activation and output - using separate activation instances
        try:
            x = self.act_func1(self.linear1(x))
            hidden = self.act_func2(self.linear2(x))
            logits = self.output_layer(hidden)
            if self.task_type == 'corn':
                output = corn_class_probs(logits)
            else:
                output = self.output_activation(logits)
        except:
            x = self.act_func(self.linear1(x))
            hidden = self.act_func(self.linear2(x))
            logits = self.output_layer(hidden)
            if self.task_type == 'corn':
                output = corn_class_probs(logits)
            else:
                output = self.output_activation(logits)
            
        return output, logits, hidden

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, lstm_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.03, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False, horizon=0,
                 temporal_idx=None, static_idx=None, spatialContext=False, d_channels=16
                 ):
        
        super(LSTM, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.lstm_size = lstm_size
        self.end_channels = end_channels
        self.n_sequences = n_sequences
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels
        self.temporal_idx = temporal_idx
        self.spatial_idx = static_idx
        self.spatialContext = spatialContext
        
        g_cha = len(self.temporal_idx)

        # LSTM block
        self.lstm = torch.nn.LSTM(
            input_size=g_cha + end_channels if horizon > 0 else g_cha,
            hidden_size=self.lstm_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(self.lstm_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(self.lstm_size).to(device)

        # Dropout after LSTM
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        
        if self.spatialContext:
            self.context_layer = SpatialContext(d_channels, self.lstm_size, 1, n_heads=4, dropout=dropout)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output layer
        self.linear1 = torch.nn.Linear(d_channels if self.spatialContext else self.lstm_size + len(self.spatial_idx), hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def forward(self, X, edge_index=None, graphs=None, z_prev=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        
        if hasattr(self, 'temporal_idx') and self.temporal_idx is not None and hasattr(self, 'spatial_idx'):
            X_spa = X[:, self.spatial_idx, -1][:, :, None]
            X = X[:, self.temporal_idx, :]
            
        batch_size = X.size(0)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            X = torch.cat((X, z_prev), dim=1)

        # (batch_size, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)

        # LSTM forward
        x, _ = self.lstm(x, (h0, c0))

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        #x = self.act_func(x)

        if hasattr(self, 'spatialContext') and self.spatialContext:
            x, _ = self.context_layer(x, X_spa)
        elif hasattr(self, 'spatialContext'):
            x = torch.concat((x, X_spa[:, :, 0]), dim=1)
            
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        if self.task_type == 'corn':
            output = corn_class_probs(logits)
        else:
            output = self.output_activation(logits)
        return output, logits, hidden
        
class DilatedCNN(torch.nn.Module):
    def __init__(self, channels, dilations, lin_channels, end_channels, n_sequences, device, act_func,
                 dropout, out_channels, task_type, use_layernorm=False, return_hidden=False, horizon=0,
                 temporal_idx=None, static_idx=None, spatialContext=False, d_channels=16):
        super(DilatedCNN, self).__init__()

        # Initialisation des listes pour les convolutions et les BatchNorm
        self.cnn_layer_list = []
        self.batch_norm_list = []
        self.num_layer = len(channels) - 1
        
        self.temporal_idx = temporal_idx
        self.spatial_idx = static_idx

        channels[0] = len(self.temporal_idx)

        # Initialisation des couches convolutives et BatchNorm
        for i in range(self.num_layer):
            if i == 0:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i] + end_channels if horizon > 0 else channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            else:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            if use_layernorm:
                self.batch_norm_list.append(torch.nn.LayerNorm(channels[i + 1]).to(device))
            else:
                self.batch_norm_list.append(torch.nn.BatchNorm1d(channels[i + 1]).to(device))

        self.dropout = torch.nn.Dropout(dropout)
        
        # Convertir les listes en ModuleList pour être compatible avec PyTorch
        self.cnn_layer_list = torch.nn.ModuleList(self.cnn_layer_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        
        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        
        if spatialContext:
            self.context_layer = SpatialContext(d_channels, channels[-1], 1, n_heads=4, dropout=dropout)

        # Output layer
        self.linear1 = torch.nn.Linear(d_channels if spatialContext else channels[-1] + len(static_idx), lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        self.return_hidden = return_hidden
        self.device = device
        self.end_channels = end_channels
        self.horizon = horizon
        self.out_channels = out_channels
        self.n_sequences = n_sequences
        self.temporal_idx = temporal_idx
        self.spatial_idx = static_idx
        self.spatialContext = spatialContext

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, x, edges=None, z_prev=None):
        # Couche d'entrée

        if hasattr(self, 'temporal_idx') and self.temporal_idx is not None and hasattr(self, 'spatial_idx'):
            X_spa = x[:, self.spatial_idx, -1][:, :, None]
            X = x[:, self.temporal_idx, :]
            x = X

        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)
        else:
            z_prev = z_prev.view(x.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            x = torch.cat((x, z_prev), dim=1)

        # Couches convolutives dilatées avec BatchNorm, activation et dropout
        for cnn_layer, batch_norm in zip(self.cnn_layer_list, self.batch_norm_list):
            x = cnn_layer(x)
            x = batch_norm(x)  # Batch Normalization
            x = self.act_func(x)
            x = self.dropout(x)
        
        # Garder uniquement le dernier élément des séquences
        x = x[:, :, -1]

        # Activation and output
        #x = self.act_func(x)

        if hasattr(self, 'spatialContext') and self.spatialContext:
            x, _ = self.context_layer(x, X_spa)
        else:
            x = torch.concat((x, X_spa[:, :, 0]), dim=1)
            
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        if self.task_type == 'corn':
            output = corn_class_probs(logits)
        else:
            output = self.output_activation(logits)
        return output, logits, hidden

class GraphCastGRU(torch.nn.Module):
    def __init__(
        self,
        *,
        # --- GRU specific parameters ---
        in_channels: int = 16,
        num_gru_layers: int = 1,
        # --- GraphCast parameters (unchanged) ---
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        end_channels: int = 64,
        lin_channels: int = 64,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        out_channels: int = 4,
        task_type: str = "classification",
        do_concat_trick: bool = False,
        has_time_dim: bool = False,
        n_sequences: int = 1,
        act_func: str = "ReLU",
        is_graph_or_node: bool = False,
        return_hidden: bool = False,
        horizon: int = 0,
        temporal_idx=None, static_idx=None, spatialContext=False, d_channels=16
    ):
        """GraphCast‐based model preceded by a GRU that encodes the temporal dimension.

        Args:
            in_channels: Dimension of the temporal features fed to the GRU (== input_size).
            num_gru_layers: Number of stacked GRU layers.
            input_dim_grid_nodes: Size of the embedding produced by the GRU for each node.  Must
                match *hidden_size* of the GRU.
            All other parameters are identical to the original GraphCastGRU.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # GRU — encodes the temporal axis and outputs an embedding per node
        # ------------------------------------------------------------------
        
        g_cha = len(temporal_idx)
        
        self.gru = torch.nn.GRU(
            input_size=g_cha + end_channels if horizon > 0 else g_cha,
            hidden_size=input_dim_grid_nodes,
            num_layers=num_gru_layers,
            dropout=0.03 if num_gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.gru_size = input_dim_grid_nodes
        self.num_gru_layers = num_gru_layers
        self.norm = torch.nn.BatchNorm1d(self.gru_size)
        self.dropout = torch.nn.Dropout(0.03)
        
        self.temporal_idx = temporal_idx
        self.spatial_idx = static_idx
        self.spatialContext = spatialContext
        
        # ------------------------------------------------------------------
        # GraphCast core network (unchanged)
        # ------------------------------------------------------------------
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
            has_time_dim,
        )
        
        if self.spatialContext:
            self.context_layer = SpatialContext(d_channels, output_dim_grid_nodes, 1, n_heads=4, dropout=0.03)

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        self.linear1 = torch.nn.Linear(d_channels if self.spatialContext else output_dim_grid_nodes + len(self.spatial_idx), lin_channels)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels)
        self.output_layer = torch.nn.Linear(end_channels, out_channels)

        self.is_graph_or_node = is_graph_or_node == "graph"

        self.act_func = getattr(torch.nn, act_func)()
        self.return_hidden = return_hidden
        self.end_channels = end_channels
        self.horizon = horizon
        self.out_channels = out_channels
        self.n_sequences = n_sequences

        if task_type == "classification":
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == "binary":
            self.output_activation = torch.nn.Sigmoid()
        else:  # regression or custom
            self.output_activation = torch.nn.Identity()

    # Forward pass
    # ----------------------------------------------------------------------
    def forward(self, X, graph, graph2mesh, mesh2graph, z_prev=None):
        """Args:
            X: Tensor shaped (batch, seq_len, in_channels, n_nodes).
        """

        # Bring node dimension next to batch for GRU: (batch * n_nodes, seq_len, in_channels)
        B, C_in, T = X.shape
        
        if hasattr(self, 'temporal_idx') and self.temporal_idx is not None and hasattr(self, 'spatial_idx'):
            X_spa = X[:, self.spatial_idx, -1][:, :, None]
            X = X[:, self.temporal_idx, :]
            C_in = len(self.temporal_idx)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)

        if self.horizon > 0:            
            X = torch.cat((X, z_prev), dim=1)

        X_for_gru = X.permute(0, 2, 1)
        
        h0 = torch.zeros(self.num_gru_layers, B, self.gru_size).to(X.device)

        gru_out, _ = self.gru(X_for_gru, h0)  # shape: (B*N, T, hidden)
        # Keep the last hidden state for each sequence
        gru_last = self.norm(gru_out[:, -1, :])
        gru_last = self.dropout(gru_last)  # (B*N, hidden == input_dim_grid_nodes)
        
        X_graphcast = gru_last[None, : ,:]

        # GraphCast processing
        x = self.net(X_graphcast, graph, graph2mesh, mesh2graph)[-1]

        # Head
        if hasattr(self, 'spatialContext') and self.spatialContext:
            x, _ = self.context_layer(x, X_spa)
        elif hasattr(self, 'spatialContext'):
            x = torch.concat((x, X_spa[:, :, 0]), dim=1)
            
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)
        if self.task_type == 'corn':
            output = corn_class_probs(logits)
        else:
            output = self.output_activation(logits)
        return output, logits, hidden

class GraphCastGRUWithAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        # --- GRU specific parameters ---
        in_channels: int = 16,
        num_gru_layers: int = 1,
        # --- GraphCast parameters (unchanged) ---
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        end_channels: int = 64,
        lin_channels: int = 64,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        out_channels: int = 4,
        task_type: str = "classification",
        do_concat_trick: bool = False,
        has_time_dim: bool = False,
        n_sequences: int = 1,
        act_func: str = "ReLU",
        is_graph_or_node: bool = False,
        return_hidden: bool = False,
        attention : bool = True,
        horizon: int = 0,
        temporal_idx=None, static_idx=None, spatialContext=False, d_channels=16
    ):
        """GraphCast‐based model preceded by a GRU that encodes the temporal dimension.

        Args:
            in_channels: Dimension of the temporal features fed to the GRU (== input_size).
            num_gru_layers: Number of stacked GRU layers.
            input_dim_grid_nodes: Size of the embedding produced by the GRU for each node.  Must
                match *hidden_size* of the GRU.
            All other parameters are identical to the original GraphCastGRU.
        """
        super().__init__()
        print(f'attention : {attention}')
        # ------------------------------------------------------------------
        # GRU — encodes the temporal axis and outputs an embedding per node
        # ------------------------------------------------------------------
        
        g_cha = len(temporal_idx)
        
        self.gru = torch.nn.GRU(
            input_size=g_cha + end_channels if horizon > 0 else g_cha,
            hidden_size=input_dim_grid_nodes,
            num_layers=num_gru_layers,
            dropout=0.03 if num_gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.gru_size = input_dim_grid_nodes
        self.num_gru_layers = num_gru_layers
        self.norm = torch.nn.BatchNorm1d(self.gru_size)
        self.dropout = torch.nn.Dropout(0.03)
        self.n_sequences = n_sequences
        
        self.temporal_idx = temporal_idx
        self.spatial_idx = static_idx
        self.spatialContext = spatialContext
        
        # ------------------------------------------------------------------
        # GraphCast core network (unchanged)
        # ------------------------------------------------------------------
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
            has_time_dim,
            attention=attention
        )
        
        if self.spatialContext:
            self.context_layer = SpatialContext(d_channels, output_dim_grid_nodes, 1, n_heads=4, dropout=0.03)

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        self.linear1 = torch.nn.Linear(d_channels if self.spatialContext else output_dim_grid_nodes + len(self.spatial_idx), lin_channels)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels)
        self.output_layer = torch.nn.Linear(end_channels, out_channels)

        self.is_graph_or_node = is_graph_or_node == "graph"

        self.act_func = getattr(torch.nn, act_func)()
        self.return_hidden = return_hidden
        self.end_channels = end_channels
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels

        if task_type == "classification":
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == "binary":
            self.output_activation = torch.nn.Sigmoid()
        else:  # regression or custom
            self.output_activation = torch.nn.Identity()

        if self.horizon > 0:
            self.define_horizon_decodeur()

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(self, X, graph, graph2mesh, mesh2graph, z_prev=None):
        """Args:
            X: Tensor shaped (batch, seq_len, in_channels, n_nodes).
        """
        # Bring node dimension next to batch for GRU: (batch * n_nodes, seq_len, in_channels)
        B, C_in, T = X.shape

        if self.temporal_idx is not None and self.spatial_idx is not None:
            X_spa = X[:, self.spatial_idx, -1][:, :, None]
            X = X[:, self.temporal_idx, :]

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            X = torch.cat((X, z_prev), dim=1)

        X_for_gru = X.permute(0, 2, 1)

        h0 = torch.zeros(self.num_gru_layers, B, self.gru_size).to(X.device)

        """gru_out, _ = self.gru(X_for_gru, h0)  # shape: (B*N, T, hidden)
        gru_out = gru_out.permute(0, 2, 1)
        # Keep the last hidden state for each sequence
        gru_last = self.norm(gru_out)
        gru_last = self.dropout(gru_last)  # (B*N, hidden == input_dim_grid_nodes)

        gru_last = gru_out.permute(2, 0, 1)

        X_graphcast = gru_last"""

        gru_out, _ = self.gru(X_for_gru, h0)  # shape: (B*N, T, hidden)
        # Keep the last hidden state for each sequence
        gru_last = self.norm(gru_out[:, -1, :])
        gru_last = self.dropout(gru_last)  # (B*N, hidden == input_dim_grid_nodes)
        
        X_graphcast = gru_last[None,: ,:]

        # GraphCast processing
        x = self.net(X_graphcast, graph, graph2mesh, mesh2graph)[-1]

        # Head

        if hasattr(self, 'spatialContext') and self.spatialContext:
            x, _ = self.context_layer(x, X_spa)
        elif hasattr(self, 'spatialContext'):
            x = torch.concat((x, X_spa[:, :, 0]), dim=1)
                        
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        self._decoder_input = hidden
        return output, logits, hidden