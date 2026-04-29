import torch
import torch.nn as nn
from typing import Any, Union, Optional
import torch.nn.functional as F
from forecasting_models.pytorch.utils import corn_class_probs
# Import shared components from models.py
from forecasting_models.pytorch.models import MLPLayer, SpatialContextSet

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


def _make_head_norm(norm_type: str, dim: int, device):
    norm_type = norm_type.lower()

    if norm_type == "batchnorm":
        return nn.BatchNorm1d(dim).to(device)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim).to(device)
    elif norm_type in {"none", "identity"}:
        return nn.Identity().to(device)
    else:
        raise ValueError(f"Unsupported head norm type: {norm_type}")


class TemporalAttentionPooling(nn.Module):
    score: nn.Linear

    def __init__(self, in_dim: int, device: Any):
        super().__init__()
        self.score = nn.Linear(in_dim, 1).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, H)
        returns:
            pooled: (B, H)
            attn  : (B, T)
        """
        attn = F.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = (x * attn.unsqueeze(-1)).sum(dim=1)
        return pooled, attn

class BetterGRU(nn.Module):
    # --- Attributes with type hints ---
    temporal_norm: nn.Module
    spatial_norm: nn.Module
    norm_after_concat: nn.Module
    context_layer: Optional[nn.Module]
    spatial_mlp: Optional[nn.Module]
    temporal_encoder: nn.Module
    gru: nn.Module
    temporal_pool_layer: nn.Module
    dropout: nn.Module
    linear1: nn.Module
    linear2: nn.Module
    output_layer: nn.Module
    output_activation: nn.Module
    last_attention_coef: Optional[torch.Tensor]
    last_temporal_attention_coef: Optional[torch.Tensor]
    temporal_repr_dim: int
    spatial_out_dim: int
    encoder_spatial: Optional[nn.Module]
    spa_norm: Optional[nn.Module]
    head_norm1: nn.Module
    head_act1: nn.Module
    head_dropout1: nn.Module
    concatenation: str
    time_fusion_mlp: nn.Module
    time_fusion_dim: int

    def __init__(
        self,
        in_channels: int,
        gru_size: int,
        hidden_channels: int,
        end_channels: int,
        n_sequences: int,
        device: Union[str, torch.device],
        act_func: str = 'ReLU',
        task_type: str = 'regression',
        dropout: float = 0.0,
        num_layers: int = 1,
        return_hidden: bool = False,
        out_channels: Optional[int] = None,
        use_layernorm: bool = False,
        horizon: int = 0,
        temporal_idx: Optional[list[int]] = None,
        static_idx: Optional[list[int]] = None,
        spatialContext: bool = False,
        d_channels: int = 16,

        # ---- New optional improvements (all removable) ----
        use_temporal_conv: bool = False,
        conv_channels: Optional[int] = None,
        conv_kernel_size: int = 3,
        conv_layers: int = 1,

        use_full_sequence: bool = True,
        temporal_pool: str = "attn",   # {"last", "mean", "max", "meanmax", "attn"}

        use_spatial_mlp: bool = True,
        spatial_hidden_channels: Optional[int] = None,
        spatial_mlp_layers: int = 2,
        spatial_mlp_use_bn: bool = True,

        # ---- Normalization refinement parameters ----
        use_temporal_norm: bool = True,
        temporal_norm_type: str = "layernorm",
        use_spatial_norm: bool = False,
        spatial_norm_type: str = "batchnorm",
        use_fusion_norm: bool = False,

        # ---- Head normalization ----
        use_head_norm: bool = True,
        head_norm_type: str = "batchnorm",

        # ---- Spatial-temporal concatenation mode ----
        concatenation: str = "last",   # "last" | "time"
        time_fusion_dim: Optional[int] = None,
    ):
        super(BetterGRU, self).__init__()

        # ----------------------------
        # Keep original public API fields
        # ----------------------------
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
        self.spatialContext = spatialContext

        # Safer defaults
        self.temporal_idx = list(range(in_channels)) if temporal_idx is None else list(temporal_idx)
        self.spatial_idx = [] if static_idx is None else list(static_idx)

        # New switches
        self.use_temporal_conv = use_temporal_conv
        self.use_full_sequence = use_full_sequence
        self.temporal_pool = temporal_pool.lower()
        self.use_spatial_mlp = use_spatial_mlp
        self.concatenation = concatenation.lower()

        if self.concatenation not in {"last", "time"}:
            raise ValueError(
                f"Unsupported concatenation='{concatenation}'. "
                "Choose from ['last', 'time']."
            )

        if self.concatenation == "time" and spatialContext:
            raise ValueError(
                "concatenation='time' is not compatible with spatialContext=True, "
                "because spatial features are already injected at each time step."
            )

        if self.temporal_pool not in {"last", "mean", "max", "meanmax", "attn", "flatten"}:
            raise ValueError(
                f"Unsupported temporal_pool='{temporal_pool}'. "
                f"Choose from ['last', 'mean', 'max', 'meanmax', 'attn', 'flatten']."
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
        # Input:  (B, C, T)
        # Output: (B, C', T)
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
            gru_input_size = c_out
        else:
            self.temporal_encoder = nn.Identity()
            gru_input_size = temporal_in_channels

        # ----------------------------
        # GRU
        # Input expected by GRU: (B, T, F)
        # ----------------------------
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

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

            # Define spatial_norm
            if use_spatial_norm:
                if spatial_norm_type == "batchnorm":
                    self.spatial_norm = nn.BatchNorm1d(self.spatial_out_dim).to(device)
                elif spatial_norm_type == "layernorm":
                    self.spatial_norm = nn.LayerNorm(self.spatial_out_dim).to(device)
                else:
                    raise ValueError(f"Unsupported spatial_norm_type: {spatial_norm_type}")
            else:
                self.spatial_norm = nn.Identity().to(device)
        else:
            self.spatial_mlp = None
            self.encoder_spatial = None
            self.spa_norm = None
            self.spatial_out_dim = 0
            self.spatial_norm = nn.Identity().to(device)

        # ----------------------------
        # Temporal-spatial fusion before temporal pooling
        # ----------------------------
        if self.concatenation == "time":
            self.gru_output_norm = nn.LayerNorm(gru_size).to(device)
        else:
            self.gru_output_norm = nn.Identity().to(device)
            
        if self.concatenation == "time":
            if not self.has_spatial:
                raise ValueError(
                    "concatenation='time' requires static_idx to be non-empty."
                )

            self.time_fusion_dim = gru_size if time_fusion_dim is None else time_fusion_dim

            self.time_fusion_mlp = nn.Sequential(
                nn.Linear(gru_size + self.spatial_out_dim, self.time_fusion_dim).to(device),
                nn.LayerNorm(self.time_fusion_dim).to(device),
                _make_activation(act_func),
                nn.Dropout(dropout).to(device),
            )

            temporal_base_dim = self.time_fusion_dim

        else:
            self.time_fusion_dim = gru_size
            self.time_fusion_mlp = nn.Identity().to(device)
            temporal_base_dim = gru_size

        # ----------------------------
        # Temporal pooling over full sequence
        # ----------------------------
        if not self.use_full_sequence:
            self.temporal_pool = "last"

        if self.temporal_pool == "meanmax":
            self.temporal_repr_dim = 2 * temporal_base_dim
        elif self.temporal_pool == "flatten":
            self.temporal_repr_dim = temporal_base_dim * n_sequences
        else:
            self.temporal_repr_dim = temporal_base_dim

        if self.temporal_pool == "attn":
            self.temporal_pool_layer = TemporalAttentionPooling(
                temporal_base_dim,
                device=device,
            )
        else:
            self.temporal_pool_layer = nn.Identity()

        # ----------------------------
        # Normalization after temporal readout
        # ----------------------------
        if use_temporal_norm:
            if temporal_norm_type == "layernorm":
                self.temporal_norm = nn.LayerNorm(self.temporal_repr_dim).to(device)
            elif temporal_norm_type == "batchnorm":
                self.temporal_norm = nn.BatchNorm1d(self.temporal_repr_dim).to(device)
            else:
                raise ValueError(f"Unsupported temporal_norm_type: {temporal_norm_type}")
        else:
            self.temporal_norm = nn.Identity().to(device)

        # ----------------------------
        # Optional spatial context fusion / late fusion
        # ----------------------------
        if self.concatenation == "time":
            self.context_layer = None
            fusion_dim = self.temporal_repr_dim

        elif self.spatialContext and self.has_spatial:
            self.context_layer = SpatialContextSet(
                d_channels,
                self.temporal_repr_dim,
                1,
                n_heads=4,
                dropout=dropout,
                use_sigmoid_gating=True,
                renorm_gates=True,
            ).to(device)
            fusion_dim = d_channels

        else:
            self.context_layer = None
            fusion_dim = self.temporal_repr_dim + self.spatial_out_dim

        # Optional norm after concat / fusion
        if use_fusion_norm:
            self.norm_after_concat = nn.LayerNorm(fusion_dim).to(device)
        else:
            self.norm_after_concat = nn.Identity().to(device)

        # ----------------------------
        # Head
        # ----------------------------
        self.linear1 = nn.Linear(fusion_dim, hidden_channels).to(device)

        self.head_norm1 = (
            _make_head_norm(head_norm_type, hidden_channels, device)
            if use_head_norm else nn.Identity().to(device)
        )

        self.head_act1 = _make_activation(act_func)
        self.head_dropout1 = nn.Dropout(dropout).to(device)

        self.linear2 = nn.Linear(hidden_channels, end_channels).to(device)
        # Handle out_channels fallback
        final_out = out_channels if out_channels is not None else 1
        self.output_layer = nn.Linear(end_channels, final_out).to(device)

        # Neutral init on final output layer
        print("Applying custom neutral initialization to BetterGRU output layer.")
        torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.001)
        if self.output_layer.bias is not None:
            torch.nn.init.zeros_(self.output_layer.bias)

        # Keep original output behavior unchanged
        if task_type == 'classification':
            self.output_activation = nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = nn.Softmax(dim=-1).to(device)
        elif task_type == 'uclassification':
            self.output_activation = UClassificationActivation()
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
        elif self.temporal_pool == "flatten":
            x = seq_out.reshape(seq_out.shape[0], -1)
        else:
            raise ValueError(f"Unsupported temporal_pool: {self.temporal_pool}")

        return x

    def _process_spatial(self, X_spa: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.has_spatial:
            return None

        if self.use_spatial_mlp and self.spatial_mlp is not None:
            y = self.spatial_mlp(X_spa)
        elif not self.use_spatial_mlp and self.encoder_spatial is not None:
            y = self.encoder_spatial(X_spa)
        else:
            return None

        y = self.spatial_norm(y)
        return y

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

        # To GRU format: (B, T, F)
        x = X_temp.permute(0, 2, 1)

        # Initial hidden state
        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.gru_size,
            device=self.device,
            dtype=x.dtype
        )
        
        # GRU over full sequence
        seq_out, _ = self.gru(x, h0)
        seq_out = self.gru_output_norm(seq_out)

        # ------------------------------------------------------------
        # Spatial-temporal fusion before temporal aggregation
        # ------------------------------------------------------------
        spa_repr = None

        if self.has_spatial:
            spa_repr = self._process_spatial(X_spa)

        if self.concatenation == "time":
            if spa_repr is None:
                raise RuntimeError(
                    "concatenation='time' requires a valid spatial representation."
                )
            
            spa_seq = spa_repr.unsqueeze(1).expand(-1, seq_out.size(1), -1)
            seq_out = torch.cat((seq_out, spa_seq), dim=-1)
            seq_out = self.time_fusion_mlp(seq_out)
        
        # Temporal readout & Normalization ordering
        if self.temporal_pool == "attn":
            # Recurrent sequence -> LayerNorm -> attention pooling -> Dropout
            # Note: temporal_norm must be LayerNorm or Identity for sequence normalization
            if self.concatenation != "time":
                seq_out = self.temporal_norm(seq_out)
                
            x = self._pool_temporal_sequence(seq_out)
            x = self.dropout(x)
        else:
            # Recurrent sequence -> pooling -> LayerNorm -> Dropout
            x = self._pool_temporal_sequence(seq_out)
            if self.concatenation != "time":
                x = self.temporal_norm(x)
            x = self.dropout(x)

        # ------------------------------------------------------------
        # Late spatial fusion only for concatenation="last"
        # ------------------------------------------------------------
        if self.concatenation == "last" and self.has_spatial:
            if spa_repr is None:
                spa_repr = self._process_spatial(X_spa)

            if self.spatialContext and self.context_layer is not None:
                x, a_s = self.context_layer(x, spa_repr)
                self.last_attention_coef = a_s
            else:
                x = torch.cat((x, spa_repr), dim=1)
                self.last_attention_coef = None
        else:
            self.last_attention_coef = None
            
        x = self.norm_after_concat(x)

        # Head
        x = self.linear1(x)
        x = self.head_norm1(x)
        x = self.head_act1(x)
        x = self.head_dropout1(x)

        hidden = self.linear2(x)
        logits = self.output_layer(hidden)

        # Output behavior unchanged
        if self.task_type == 'corn':
            output = corn_class_probs(logits)
        else:
            output = self.output_activation(logits)

        return output, logits, hidden
