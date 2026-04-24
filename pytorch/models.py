import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

from forecasting_models.pytorch.gnns import *
try:
    from forecasting_models.pytorch import itransformer
except ImportError:
    itransformer = None
    print("Warning: itransformer module could not be loaded due to missing dependencies (e.g., reformer_pytorch).")
from forecasting_models.pytorch.utils import corn_class_probs

def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class TemporalAttentionPooling(nn.Module):
    def __init__(self, in_dim: int, device):
        super().__init__()
        self.score = nn.Linear(in_dim, 1).to(device)

    def forward(self, x):
        """
        x: (B, T, H)
        returns:
            pooled: (B, H)
            attn  : (B, T)
        """
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled, attn

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

        # 1. First, pass values through the per-variable MLP *without* weighting yet
        v_tokens = v.permute(0, 2, 1, 3).reshape(B, K, self.d_model) # [B, K, d_model]
        v_tokens = self.var_mlp(v_tokens)      # [B, K, d_model] - Normalization happens INSIDE here

        # 2. Reshape back to [B, H, K, Dh] to apply attention
        v_tokens_head = v_tokens.view(B, K, self.n_heads, self.d_head).permute(0, 2, 1, 3)

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

        # 3. APPLY ATTENTION WEIGHTS *AFTER* NORMALIZATION/MLP
        z_tokens_head = w_drop.unsqueeze(-1) * v_tokens_head  # [B, H, K, Dh]
        
        # Pool across K
        z_heads = z_tokens_head.sum(dim=2)              # [B, H, Dh]
        z = z_heads.reshape(B, self.d_model)            # [B, d_model]

        # Output projection and residual
        c = self.out_proj(z)
        c = self.dropout(c)
        c = h0 + self.res_scale * c                     # [B, d_model]
        
        # Interpretable attention weights
        w_mean = w.mean(dim=1)                          # [B, K]

        if self.return_tokens:
            return c, w_mean, z_tokens_head
        return c, w_mean
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        device,
        activation: str = "relu",
        dropout: float = 0.0,
        use_bn: bool = True,
    ):
        super().__init__()

        self.linear = nn.Linear(in_feats, out_feats).to(device)
        self.bn = nn.BatchNorm1d(out_feats).to(device) if use_bn else nn.Identity()

        if activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "elu":
            self.act = nn.ELU()
        elif activation.lower() == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class NetMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        end_channels,
        output_channels,
        n_sequences,
        device,
        task_type,
        return_hidden=False,
        horizon=0,
        activation="relu",
        dropout=0.1,
        use_bn=True,
        **kwargs
    ):
        super().__init__()

        self.horizon = horizon
        self.task_type = task_type
        self.n_sequences = n_sequences
        self.return_hidden = return_hidden
        self.device = device
        self.end_channels = end_channels
        self.output_channels = output_channels
        self.in_dim = in_dim

        input_dim = in_dim * n_sequences + end_channels if horizon > 0 else in_dim * n_sequences

        self.layer1 = MLPLayer(
            input_dim,
            hidden_dim[0],
            device=device,
            activation=activation,
            dropout=dropout,
            use_bn=use_bn,
        )
        self.layer3 = MLPLayer(
            hidden_dim[0],
            hidden_dim[1],
            device=device,
            activation=activation,
            dropout=dropout,
            use_bn=use_bn,
        )
        self.layer4 = MLPLayer(
            hidden_dim[1],
            end_channels,
            device=device,
            activation=activation,
            dropout=dropout,
            use_bn=use_bn,
        )

        self.layer2 = nn.Linear(end_channels, output_channels).to(device)
        self.soft = nn.Softmax(dim=1)

    def forward(self, features, z_prev=None, edges=None):
        if features.device != self.device:
            features = features.to(self.device)

        if self.horizon > 0:
            if z_prev is None:
                z_prev = torch.zeros(
                    (features.shape[0], self.end_channels),
                    device=self.device,
                    dtype=features.dtype,
                )
            elif z_prev.device != self.device:
                z_prev = z_prev.to(self.device)

        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)

        if self.horizon > 0:
            features = torch.cat((features, z_prev), dim=1)

        x = self.layer1(features)
        x = self.layer3(x)
        x = self.layer4(x)

        hidden = x
        logits = self.layer2(x)

        if self.task_type == "classification":
            output = self.soft(logits)
        elif self.task_type == "corn":
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
        
        if self.spatial_idx is not None and len(self.spatial_idx) > 0:
            self.spa_norm = torch.nn.BatchNorm1d(len(self.spatial_idx)).to(device)
        
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
            
        self.norm_after_concat = torch.nn.LayerNorm(gru_size + len(self.spatial_idx))
            
        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        
        if self.spatialContext:
            self.context_layer = SpatialContextSet(d_channels, gru_size, 1, n_heads=4, dropout=dropout, use_sigmoid_gating=True, renorm_gates=True,)
        
        self.encoder_spatial = torch.nn.Linear(len(self.spatial_idx), len(self.spatial_idx))

        # Output linear layer
        self.linear1 = torch.nn.Linear(d_channels if self.spatialContext else gru_size + len(self.spatial_idx), hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        
        # Output linear layer
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
        
        X = X.to(self.device) if X.device != self.device else X
        
        if hasattr(self, 'temporal_idx') and self.temporal_idx is not None and hasattr(self, 'spatial_idx'):
            X_spa = X[:, self.spatial_idx, -1]
            if hasattr(self, 'spa_norm') and self.spatial_idx is not None and len(self.spatial_idx) > 0:
                pass
        
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
            X_spa = self.encoder_spatial(X_spa)
            X_spa = self.spa_norm(X_spa)
            x, a = self.context_layer(x, X_spa)
            self.last_attention_coef = a
        elif hasattr(self, 'spatialContext'):
            #X_spa = self.encoder_spatial(X_spa)
            #X_spa = self.spa_norm(X_spa)
            x = torch.concat((x, X_spa), dim=1)
            self.last_attention_coef = None
        
        #x = self.norm_after_concat(x)
        
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
            return X_spa

import torch

class LSTM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        lstm_size,               # gardé exprès pour compatibilité API avec GRU
        hidden_channels,
        end_channels,
        n_sequences,
        device,
        act_func='ReLU',
        task_type='regression',
        dropout=0.0,
        num_layers=1,
        return_hidden=False,
        out_channels=None,
        use_layernorm=False,
        horizon=0,
        temporal_idx=None,
        static_idx=None,
        spatialContext=False,
        d_channels=16,

        # ---- New optional improvements (matching BetterGRU) ----
        use_temporal_conv=False,
        conv_channels=None,
        conv_kernel_size=3,
        conv_layers=1,

        use_full_sequence=True,
        temporal_pool="last",   # {"last", "mean", "max", "meanmax", "attn"}

        use_spatial_mlp=True,
        spatial_hidden_channels=None,
        spatial_mlp_layers=2,
        spatial_mlp_use_bn=True,
    ):
        super(LSTM, self).__init__()

        # ----------------------------
        # Keep original public API fields
        # ----------------------------
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
        if out_channels is None:
            raise ValueError("out_channels must be provided and not None.")
        self.out_channels = out_channels
        self.spatialContext = spatialContext

        # Safer defaults
        self.temporal_idx = list(range(in_channels)) if temporal_idx is None else list(temporal_idx)
        self.spatial_idx = [] if static_idx is None else list(static_idx)

        # New switches
        self.use_temporal_conv = use_temporal_conv
        self.use_full_sequence = use_full_sequence
        self.temporal_pool = temporal_pool.lower()
        self.use_spatial_mlp = use_spatial_mlp

        if self.temporal_pool not in {"last", "mean", "max", "meanmax", "attn"}:
            raise ValueError(
                f"Unsupported temporal_pool='{temporal_pool}'. "
                f"Choose from ['last', 'mean', 'max', 'meanmax', 'attn']."
            )

        if self.use_temporal_conv and conv_kernel_size % 2 == 0:
            raise ValueError(
                "conv_kernel_size must be odd if you want to preserve temporal length cleanly."
            )

        # ----------------------------
        # Dimensions
        # ----------------------------
        g_cha = len(self.temporal_idx)
        self.has_spatial = len(self.spatial_idx) > 0

        temporal_in_channels = g_cha + self.end_channels if horizon > 0 else g_cha

        # ----------------------------
        # Optional temporal Conv1d encoder
        # ----------------------------
        if self.use_temporal_conv:
            conv_channels = temporal_in_channels if conv_channels is None else conv_channels
            conv_blocks = []
            c_in = temporal_in_channels
            c_out = conv_channels

            for _ in range(conv_layers):
                conv_blocks.append(
                    nn.Conv1d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=conv_kernel_size,
                        padding=conv_kernel_size // 2,
                    ).to(device)
                )
                conv_blocks.append(nn.BatchNorm1d(c_out).to(device))
                conv_blocks.append(_make_activation(act_func))
                conv_blocks.append(nn.Dropout(dropout))
                c_in = c_out

            self.temporal_encoder = nn.Sequential(*conv_blocks)
            lstm_input_size = c_out
        else:
            self.temporal_encoder = nn.Identity()
            lstm_input_size = temporal_in_channels

        # ----------------------------
        # LSTM layer
        # ----------------------------
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # ----------------------------
        # Temporal pooling over full sequence
        # ----------------------------
        if not self.use_full_sequence:
            self.temporal_pool = "last"

        if self.temporal_pool == "meanmax":
            self.temporal_repr_dim = 2 * lstm_size
        else:
            self.temporal_repr_dim = lstm_size

        if self.temporal_pool == "attn":
            self.temporal_pool_layer = TemporalAttentionPooling(lstm_size, device=device)
        else:
            self.temporal_pool_layer = None

        # ----------------------------
        # Normalization after temporal readout
        # ----------------------------
        if use_layernorm:
            self.temporal_norm = nn.LayerNorm(self.temporal_repr_dim).to(device)
        else:
            self.temporal_norm = nn.BatchNorm1d(self.temporal_repr_dim).to(device)

        self.dropout = nn.Dropout(p=dropout).to(device)

        # ----------------------------
        # Spatial branch
        # ----------------------------
        self.last_attention_coef = None          # spatial attention coeffs
        self.last_temporal_attention_coef = None # temporal attention coeffs

        if self.has_spatial:
            spatial_in_dim = len(self.spatial_idx)
            spatial_hidden_channels = (
                spatial_in_dim if spatial_hidden_channels is None else spatial_hidden_channels
            )

            if self.use_spatial_mlp:
                spatial_layers = []
                d_in = spatial_in_dim
                for _ in range(max(1, spatial_mlp_layers)):
                    spatial_layers.append(
                        MLPLayer(
                            in_feats=d_in,
                            out_feats=spatial_hidden_channels,
                            device=device,
                            activation=act_func,
                            dropout=dropout,
                            use_bn=spatial_mlp_use_bn,
                        )
                    )
                    d_in = spatial_hidden_channels

                self.spatial_mlp = nn.Sequential(*spatial_layers)
                self.spatial_out_dim = d_in
                self.encoder_spatial = None
                self.spa_norm = None
            else:
                self.spatial_mlp = None
                self.encoder_spatial = nn.Linear(spatial_in_dim, spatial_in_dim).to(device)
                self.spa_norm = nn.BatchNorm1d(spatial_in_dim).to(device)
                self.spatial_out_dim = spatial_in_dim
        else:
            self.spatial_mlp = None
            self.encoder_spatial = None
            self.spa_norm = None
            self.spatial_out_dim = 0

        # ----------------------------
        # Optional spatial context fusion
        # ----------------------------
        if self.spatialContext and self.has_spatial:
            self.context_layer = SpatialContextSet(
                d_channels,
                self.temporal_repr_dim,
                1,
                n_heads=4,
                dropout=dropout,
                use_sigmoid_gating=True,
                renorm_gates=True,
            )
            fusion_dim = d_channels
        else:
            self.context_layer = None
            fusion_dim = self.temporal_repr_dim + self.spatial_out_dim

        # Optional norm after concat / fusion
        if use_layernorm:
            self.norm_after_concat = nn.LayerNorm(fusion_dim).to(device)
        else:
            self.norm_after_concat = nn.BatchNorm1d(fusion_dim).to(device)

        # ----------------------------
        # Head
        # ----------------------------
        self.linear1 = nn.Linear(fusion_dim, hidden_channels).to(device)
        self.linear2 = nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = nn.Linear(end_channels, out_channels).to(device)

        # Neutral init on final output layer
        print("Applying custom neutral initialization to LSTM output layer.")
        torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.001)
        if self.output_layer.bias is not None:
            torch.nn.init.zeros_(self.output_layer.bias)

        # Separate activation instances
        self.act_func1 = _make_activation(act_func)
        self.act_func2 = _make_activation(act_func)

        # Keep original output behavior unchanged
        if task_type == 'classification':
            self.output_activation = nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = nn.Identity().to(device)

    def _pool_temporal_sequence(self, seq_out):
        """
        seq_out: (B, T, H)
        returns x: (B, H') depending on pooling mode
        """
        self.last_temporal_attention_coef = None

        if self.temporal_pool == "last":
            x = seq_out[:, -1, :]
        elif self.temporal_pool == "mean":
            x = seq_out.mean(dim=1)
        elif self.temporal_pool == "max":
            x = seq_out.max(dim=1).values
        elif self.temporal_pool == "meanmax":
            x_mean = seq_out.mean(dim=1)
            x_max = seq_out.max(dim=1).values
            x = torch.cat((x_mean, x_max), dim=1)
        elif self.temporal_pool == "attn":
            x, a_t = self.temporal_pool_layer(seq_out)
            self.last_temporal_attention_coef = a_t
        else:
            raise ValueError(f"Unsupported temporal_pool: {self.temporal_pool}")

        return x

    def _process_spatial(self, X_spa):
        """
        X_spa: (B, S)
        returns spatial representation
        """
        if not self.has_spatial:
            return None

        if self.use_spatial_mlp:
            return self.spatial_mlp(X_spa)
        else:
            X_spa = self.encoder_spatial(X_spa)
            X_spa = self.spa_norm(X_spa)
            return X_spa

    def forward(self, X, edge_index=None, graphs=None, z_prev=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output, logits, hidden
        """
        X = X.to(self.device) if X.device != self.device else X

        # Spatial features from last time step
        X_spa = None
        if self.has_spatial:
            X_spa = X[:, self.spatial_idx, -1]

        # Temporal features
        X_temp = X[:, self.temporal_idx, :]

        batch_size = X_temp.size(0)

        # Previous decoded state for autoregressive horizon input
        if z_prev is None:
            z_prev = torch.zeros(
                (batch_size, self.end_channels, self.n_sequences),
                device=X_temp.device,
                dtype=X_temp.dtype
            )
        else:
            z_prev = z_prev.view(batch_size, self.end_channels, self.n_sequences)

        if self.horizon > 0:
            X_temp = torch.cat((X_temp, z_prev), dim=1)

        # Optional temporal convolution
        # X_temp: (B, C, T)
        X_temp = self.temporal_encoder(X_temp)

        # To LSTM format: (B, T, F)
        x = X_temp.permute(0, 2, 1)

        # Initial hidden states
        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.lstm_size,
            device=self.device,
            dtype=x.dtype
        )
        c0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.lstm_size,
            device=self.device,
            dtype=x.dtype
        )

        # LSTM over full sequence
        seq_out, _ = self.lstm(x, (h0, c0))

        # Temporal readout
        x = self._pool_temporal_sequence(seq_out)

        # Norm + dropout
        x = self.temporal_norm(x)
        x = self.dropout(x)

        # Spatial branch / context branch
        if self.has_spatial:
            X_spa = X[:, self.spatial_idx, -1]
            spa_repr = self._process_spatial(X_spa)

            if self.spatialContext and self.context_layer is not None:
                # SpatialContextSet expects (B, K, D_stat). 
                # Since d_stat is 1, each feature is a variable of dim 1.
                if spa_repr.dim() == 2:
                    spa_repr = spa_repr.unsqueeze(-1)
                x, a_s = self.context_layer(x, spa_repr)
                self.last_attention_coef = a_s
            else:
                x = torch.cat((x, spa_repr), dim=1)
                self.last_attention_coef = None
        else:
            self.last_attention_coef = None

        # Optional norm after fusion
        x = self.norm_after_concat(x)

        # Head
        x = self.act_func1(self.linear1(x))
        hidden = self.act_func2(self.linear2(x))
        logits = self.output_layer(hidden)

        # Output behavior unchanged
        if self.task_type == 'corn':
            output = corn_class_probs(logits)
        else:
            output = self.output_activation(logits)

        return output, logits, hidden
        
import torch
import torch.nn as nn


def _make_activation(act_func: str):
    if isinstance(act_func, str):
        return getattr(nn, act_func)()
    return act_func


def _apply_temporal_norm(x, norm):
    """
    x: (B, C, T)
    BatchNorm1d expects (B, C, T).
    LayerNorm(C) expects the normalized dimension last, so we use (B, T, C).
    """
    if isinstance(norm, nn.LayerNorm):
        return norm(x.transpose(1, 2)).transpose(1, 2)
    return norm(x)


def _build_mlp_block(
    in_dim,
    out_dim,
    device,
    activation="ReLU",
    dropout=0.0,
    use_bn=True,
):
    """
    Compatible fallback if your MLPLayer class does not have the same signature
    as the one used in BetterGRU/LSTM.
    """
    layers = [nn.Linear(in_dim, out_dim).to(device)]

    if use_bn:
        layers.append(nn.BatchNorm1d(out_dim).to(device))

    layers.append(_make_activation(activation))
    layers.append(nn.Dropout(dropout).to(device))

    return nn.Sequential(*layers)


class DilatedCNN(torch.nn.Module):
    def __init__(
        self,
        channels,
        dilations,
        lin_channels,
        end_channels,
        n_sequences,
        device,
        act_func,
        dropout,
        out_channels,
        task_type,
        use_layernorm=False,
        return_hidden=False,
        horizon=0,
        temporal_idx=None,
        static_idx=None,
        spatialContext=False,
        d_channels=16,

        # ---- Encodeur Conv1d temporel optionnel ----
        use_temporal_conv=False,
        conv_channels=None,
        conv_kernel_size=3,
        conv_layers=1,

        # ---- Pooling sur toute la séquence CNN ----
        use_full_sequence=True,
        temporal_pool="attn",  # "last" | "mean" | "max" | "meanmax" | "attn"

        # ---- Branche spatiale ----
        use_spatial_mlp=True,
        spatial_hidden_channels=128,
        spatial_mlp_layers=3,
        spatial_mlp_use_bn=True,
    ):
        super(DilatedCNN, self).__init__()

        self.device = device
        self.task_type = task_type
        self.return_hidden = return_hidden
        self.end_channels = end_channels
        self.horizon = horizon
        self.out_channels = out_channels
        self.n_sequences = n_sequences
        self.spatialContext = spatialContext

        self.use_temporal_conv = use_temporal_conv
        self.use_full_sequence = use_full_sequence
        self.temporal_pool = temporal_pool.lower()
        self.use_spatial_mlp = use_spatial_mlp

        if self.temporal_pool not in {"last", "mean", "max", "meanmax", "attn"}:
            raise ValueError(
                f"Unsupported temporal_pool='{temporal_pool}'. "
                "Choose from ['last', 'mean', 'max', 'meanmax', 'attn']."
            )

        if self.use_temporal_conv and conv_kernel_size % 2 == 0:
            raise ValueError(
                "conv_kernel_size must be odd to preserve temporal length cleanly."
            )

        # Avoid mutating the input list outside the class.
        channels = list(channels)
        dilations = list(dilations)

        self.temporal_idx = (
            list(range(channels[0])) if temporal_idx is None else list(temporal_idx)
        )
        self.spatial_idx = [] if static_idx is None else list(static_idx)
        self.has_spatial = len(self.spatial_idx) > 0

        g_cha = len(self.temporal_idx)
        temporal_in_channels = g_cha + end_channels if horizon > 0 else g_cha

        # ============================================================
        # Optional temporal Conv1d encoder before dilated CNN
        # ============================================================
        if self.use_temporal_conv:
            conv_channels = temporal_in_channels if conv_channels is None else conv_channels

            conv_blocks = []
            c_in = temporal_in_channels
            c_out = conv_channels

            for _ in range(conv_layers):
                conv_blocks.append(
                    nn.Conv1d(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=conv_kernel_size,
                        padding=conv_kernel_size // 2,
                    ).to(device)
                )
                conv_blocks.append(nn.BatchNorm1d(c_out).to(device))
                conv_blocks.append(_make_activation(act_func))
                conv_blocks.append(nn.Dropout(dropout).to(device))
                c_in = c_out

            self.temporal_encoder = nn.Sequential(*conv_blocks)
            first_cnn_in_channels = c_out
        else:
            self.temporal_encoder = nn.Identity()
            first_cnn_in_channels = temporal_in_channels

        # ============================================================
        # Dilated CNN stack
        # ============================================================
        self.num_layer = len(channels) - 1
        if len(dilations) < self.num_layer:
            raise ValueError(
                f"Expected at least {self.num_layer} dilation values, got {len(dilations)}."
            )

        self.cnn_layer_list = nn.ModuleList()
        self.norm_layer_list = nn.ModuleList()

        for i in range(self.num_layer):
            c_in = first_cnn_in_channels if i == 0 else channels[i]
            c_out = channels[i + 1]

            self.cnn_layer_list.append(
                nn.Conv1d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=3,
                    padding="same",
                    dilation=dilations[i],
                    padding_mode="replicate",
                ).to(device)
            )

            if use_layernorm:
                self.norm_layer_list.append(nn.LayerNorm(c_out).to(device))
            else:
                self.norm_layer_list.append(nn.BatchNorm1d(c_out).to(device))

        self.act_func = _make_activation(act_func)
        self.dropout = nn.Dropout(p=dropout).to(device)

        # ============================================================
        # Temporal pooling over CNN output sequence
        # ============================================================
        if not self.use_full_sequence:
            self.temporal_pool = "last"

        cnn_out_channels = channels[-1]

        if self.temporal_pool == "meanmax":
            self.temporal_repr_dim = 2 * cnn_out_channels
        else:
            self.temporal_repr_dim = cnn_out_channels

        if self.temporal_pool == "attn":
            self.temporal_pool_layer = TemporalAttentionPooling(
                cnn_out_channels,
                device=device,
            )
        else:
            self.temporal_pool_layer = None

        if use_layernorm:
            self.temporal_norm = nn.LayerNorm(self.temporal_repr_dim).to(device)
        else:
            self.temporal_norm = nn.BatchNorm1d(self.temporal_repr_dim).to(device)

        self.last_temporal_attention_coef = None
        self.last_attention_coef = None

        # ============================================================
        # Spatial branch
        # ============================================================
        if self.has_spatial:
            spatial_in_dim = len(self.spatial_idx)

            if spatial_hidden_channels is None:
                spatial_hidden_channels = spatial_in_dim

            if self.use_spatial_mlp:
                spatial_layers = []
                d_in = spatial_in_dim

                for _ in range(max(1, spatial_mlp_layers)):
                    try:
                        # If your MLPLayer has the same API as in BetterGRU/LSTM.
                        spatial_layers.append(
                            MLPLayer(
                                in_feats=d_in,
                                out_feats=spatial_hidden_channels,
                                device=device,
                                activation=act_func,
                                dropout=dropout,
                                use_bn=spatial_mlp_use_bn,
                            )
                        )
                    except TypeError:
                        # Fallback if MLPLayer has an older signature.
                        spatial_layers.append(
                            _build_mlp_block(
                                in_dim=d_in,
                                out_dim=spatial_hidden_channels,
                                device=device,
                                activation=act_func,
                                dropout=dropout,
                                use_bn=spatial_mlp_use_bn,
                            )
                        )

                    d_in = spatial_hidden_channels

                self.spatial_mlp = nn.Sequential(*spatial_layers)
                self.encoder_spatial = None
                self.spa_norm = None
                self.spatial_out_dim = d_in
            else:
                self.spatial_mlp = None
                self.encoder_spatial = nn.Linear(spatial_in_dim, spatial_in_dim).to(device)
                self.spa_norm = nn.BatchNorm1d(spatial_in_dim).to(device)
                self.spatial_out_dim = spatial_in_dim
        else:
            self.spatial_mlp = None
            self.encoder_spatial = None
            self.spa_norm = None
            self.spatial_out_dim = 0

        # ============================================================
        # Optional spatial context fusion
        # ============================================================
        if self.spatialContext and self.has_spatial:
            # Prefer the same class as your LSTM if available.
            if "SpatialContextSet" in globals():
                self.context_layer = SpatialContextSet(
                    d_channels,
                    self.temporal_repr_dim,
                    1,
                    n_heads=4,
                    dropout=dropout,
                    use_sigmoid_gating=True,
                    renorm_gates=True,
                )
            else:
                self.context_layer = SpatialContext(
                    d_channels,
                    self.temporal_repr_dim,
                    1,
                    n_heads=4,
                    dropout=dropout,
                )

            self.context_layer = self.context_layer.to(device)
            fusion_dim = d_channels
        else:
            self.context_layer = None
            fusion_dim = self.temporal_repr_dim + self.spatial_out_dim

        if use_layernorm:
            self.norm_after_concat = nn.LayerNorm(fusion_dim).to(device)
        else:
            self.norm_after_concat = nn.BatchNorm1d(fusion_dim).to(device)

        # ============================================================
        # Head
        # ============================================================
        self.linear1 = nn.Linear(fusion_dim, lin_channels).to(device)
        self.linear2 = nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = nn.Linear(end_channels, out_channels).to(device)

        # Same output behavior as before.
        if task_type == "classification":
            self.output_activation = nn.Softmax(dim=-1).to(device)
        elif task_type == "binary":
            self.output_activation = nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = nn.Identity().to(device)

    def _temporal_pooling(self, x):
        """
        x: (B, C, T)
        returns: (B, D)
        """
        if self.temporal_pool == "last":
            pooled = x[:, :, -1]

        elif self.temporal_pool == "mean":
            pooled = x.mean(dim=-1)

        elif self.temporal_pool == "max":
            pooled = x.amax(dim=-1)

        elif self.temporal_pool == "meanmax":
            pooled = torch.cat(
                [
                    x.mean(dim=-1),
                    x.amax(dim=-1),
                ],
                dim=1,
            )

        elif self.temporal_pool == "attn":
            # TemporalAttentionPooling expects (B, T, C).
            x_seq = x.transpose(1, 2)
            pooled, attn = self.temporal_pool_layer(x_seq)
            self.last_temporal_attention_coef = attn

        else:
            raise RuntimeError(f"Unsupported temporal_pool={self.temporal_pool}")

        return pooled

    def forward(self, x, edges=None, z_prev=None):
        x = x.to(self.device) if x.device != self.device else x

        # ------------------------------------------------------------
        # Split temporal and static variables
        # x expected shape: (B, F, T)
        # ------------------------------------------------------------
        if self.temporal_idx is not None:
            X = x[:, self.temporal_idx, :]
        else:
            X = x

        if self.has_spatial:
            X_spa_vec = x[:, self.spatial_idx, -1]  # (B, S)
            X_spa_ctx = X_spa_vec[:, :, None]       # (B, S, 1)
        else:
            X_spa_vec = None
            X_spa_ctx = None

        x = X

        # ------------------------------------------------------------
        # Horizon conditioning
        # ------------------------------------------------------------
        if z_prev is None:
            z_prev = torch.zeros(
                (x.shape[0], self.end_channels, x.shape[-1]),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            z_prev = z_prev.to(x.device)
            z_prev = z_prev.view(x.shape[0], self.end_channels, x.shape[-1])

        if self.horizon > 0:
            x = torch.cat((x, z_prev), dim=1)

        # ------------------------------------------------------------
        # Optional temporal Conv1d encoder
        # ------------------------------------------------------------
        x = self.temporal_encoder(x)

        # ------------------------------------------------------------
        # Dilated CNN stack
        # ------------------------------------------------------------
        for cnn_layer, norm_layer in zip(self.cnn_layer_list, self.norm_layer_list):
            x = cnn_layer(x)
            x = _apply_temporal_norm(x, norm_layer)
            x = self.act_func(x)
            x = self.dropout(x)

        # ------------------------------------------------------------
        # Temporal readout
        # ------------------------------------------------------------
        x = self._temporal_pooling(x)  # (B, temporal_repr_dim)
        x = self.temporal_norm(x)
        x = self.dropout(x)

        # ------------------------------------------------------------
        # Spatial branch and fusion
        # ------------------------------------------------------------
        if self.context_layer is not None:
            x, attn_coef = self.context_layer(x, X_spa_ctx)
            self.last_attention_coef = attn_coef

        else:
            if self.has_spatial:
                if self.use_spatial_mlp:
                    x_spa = self.spatial_mlp(X_spa_vec)
                else:
                    x_spa = self.encoder_spatial(X_spa_vec)
                    x_spa = self.spa_norm(x_spa)
                    x_spa = self.act_func(x_spa)

                x = torch.cat((x, x_spa), dim=1)

            x = self.norm_after_concat(x)

        # ------------------------------------------------------------
        # Prediction head
        # ------------------------------------------------------------
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)

        if self.task_type == "corn":
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
        self.spatial_idx = static_idx
        self.norm = torch.nn.BatchNorm1d(self.gru_size)
        if self.spatial_idx is not None and len(self.spatial_idx) > 0:
            self.spa_norm = torch.nn.BatchNorm1d(len(self.spatial_idx))
        self.dropout = torch.nn.Dropout(0.03)
        self.task_type = task_type
        
        self.temporal_idx = temporal_idx
        self.spatialContext = spatialContext
        
        if self.spatialContext:
            self.context_layer = SpatialContext(input_dim_grid_nodes, input_dim_grid_nodes, 1, n_heads=4, dropout=0.03)
        
        # ------------------------------------------------------------------
        # GraphCast core network (unchanged)
        # ------------------------------------------------------------------
        self.net = GraphCastNet(
            input_dim_grid_nodes if self.spatialContext else input_dim_grid_nodes + len(self.spatial_idx),
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

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        self.linear1 = torch.nn.Linear(output_dim_grid_nodes, lin_channels)
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
        _dev = next(self.parameters()).device
        if X.device != _dev:
            X = X.to(_dev)

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

        # Head
        if hasattr(self, 'spa_norm') and self.spatial_idx is not None and len(self.spatial_idx) > 0:
            X_spa = self.spa_norm(X_spa)
        if hasattr(self, 'spatialContext') and self.spatialContext:
            X_graphcast, _ = self.context_layer(gru_last, X_spa)
        else:
            X_graphcast = torch.concat((gru_last, X_spa[:, :, 0]), dim=1)
        
        X_graphcast = X_graphcast[None, : ,:]
            
        # GraphCast processing
        x = self.net(X_graphcast, graph, graph2mesh, mesh2graph)[-1]
            
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
        
        if self.spatial_idx is not None and len(self.spatial_idx) > 0:
            self.spa_norm = torch.nn.BatchNorm1d(len(self.spatial_idx))
        
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
        _dev = next(self.parameters()).device
        if X.device != _dev:
            X = X.to(_dev)
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
        if hasattr(self, 'spa_norm') and self.spatial_idx is not None and len(self.spatial_idx) > 0:
            X_spa = self.spa_norm(X_spa)

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