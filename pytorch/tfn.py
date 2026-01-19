import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


# ---------------------------
# Building blocks (TFT paper)
# ---------------------------

class GLU(nn.Module):
    """Gated Linear Unit: (A ⊙ σ(B)) with A,B linear projections."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc_a = nn.Linear(d_in, d_out)
        self.fc_b = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_a(x) * torch.sigmoid(self.fc_b(x))


class GatedResidualNetwork(nn.Module):
    """
    TFT GRN (paper-aligned):
      - optional context
      - two-layer MLP + dropout
      - residual connection
      - GLU gating
      - LayerNorm
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: Optional[int] = None,
        d_context: Optional[int] = None,
        dropout: float = 0.0,
        act_func: str = "ELU",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        self.d_context = d_context

        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = getattr(nn, act_func)()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, self.d_out)

        # optional context projection added before activation (paper-style)
        if d_context is not None:
            self.context_fc = nn.Linear(d_context, d_hidden)
        else:
            self.context_fc = None

        self.skip = nn.Identity() if d_in == self.d_out else nn.Linear(d_in, self.d_out)
        self.glu = GLU(self.d_out, self.d_out)
        self.norm = nn.LayerNorm(self.d_out)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [..., d_in]
        context (optional): [..., d_context] broadcastable to x batch/time dims
        """
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)  # [..., d_out]

        y = h + self.skip(x)
        y = self.glu(y)  # gating (GLU)
        return self.norm(y)


class PositionalEncoding(nn.Module):
    """Classic sinusoidal PE (optional in paper; often used in implementations)."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class VariableSelectionNetwork(nn.Module):
    """
    Paper-aligned VSN:
      - per-variable GRN to produce transformed variable embeddings
      - a "selection" GRN to produce sparse weights over variables
      - weighted sum of transformed variable embeddings

    Here: each variable is scalar at each time step.
    Inputs:
      x: [B, T, F]
      static_context (optional): [B, d_static] to condition weights & transforms
    Outputs:
      v: [B, T, d_model]
      weights: [B, T, F]
    """
    def __init__(
        self,
        num_features: int,
        d_model: int,
        d_hidden: int,
        d_static: Optional[int] = None,
        dropout: float = 0.0,
        act_func: str = "ELU",
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # transform each variable to d_model via GRN (scalar->d_model)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                d_in=1, d_hidden=d_hidden, d_out=d_model, d_context=d_static,
                dropout=dropout, act_func=act_func
            )
            for _ in range(num_features)
        ])

        # selection weights network
        self.selection_grn = GatedResidualNetwork(
            d_in=num_features, d_hidden=d_hidden, d_out=num_features, d_context=d_static,
            dropout=dropout, act_func=act_func
        )

    def forward(self, x: torch.Tensor, static_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F = x.shape
        assert F == self.num_features, f"Expected {self.num_features} features, got {F}"

        # weights: [B,T,F]
        w_logits = self.selection_grn(x, context=static_context.unsqueeze(1) if (static_context is not None) else None)
        w = torch.softmax(w_logits, dim=-1)

        # transformed vars: [B,T,F,d_model]
        transformed = []
        ctx = static_context.unsqueeze(1) if static_context is not None else None  # [B,1,d_static]
        for i in range(F):
            xi = x[:, :, i:i+1]  # [B,T,1]
            transformed.append(self.var_grns[i](xi, context=ctx))  # [B,T,d_model]
        V = torch.stack(transformed, dim=2)  # [B,T,F,d_model]

        # weighted sum -> [B,T,d_model]
        v = (w.unsqueeze(-1) * V).sum(dim=2)
        return v, w


class InterpretableMultiHeadAttention(nn.Module):
    """
    Paper-aligned "interpretable" attention:
      - multi-head attention with shared value projection across heads (common trick),
        and average heads for interpretability.
    Outputs attention weights per head (and optionally averaged).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        # shared V projection for interpretability
        self.w_v = nn.Linear(d_model, self.d_head)
        self.w_o = nn.Linear(self.d_head, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        q: [B, Tq, d_model]
        k,v: [B, Tk, d_model]
        mask (optional): [B, Tq, Tk] with True for allowed positions (or 0/1)
        """
        B, Tq, _ = q.shape
        Tk = k.size(1)

        Q = self.w_q(q).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Tq,Dh]
        K = self.w_k(k).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Tk,Dh]
        # shared V across heads -> expand
        Vh = self.w_v(v)  # [B,Tk,Dh]
        Vh = Vh.unsqueeze(1).expand(B, self.n_heads, Tk, self.d_head)  # [B,H,Tk,Dh]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Tq,Tk]
        if mask is not None:
            # mask: True/1 keeps, False/0 blocks
            scores = scores.masked_fill(~mask.unsqueeze(1).bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # [B,H,Tq,Tk]
        attn = self.dropout(attn)

        out = torch.matmul(attn, Vh)  # [B,H,Tq,Dh]
        out = out.mean(dim=1)         # average heads for interpretability -> [B,Tq,Dh]
        out = self.w_o(out)           # -> [B,Tq,d_model]
        return out, attn


# ---------------------------
# Paper-aligned TFT Module
# ---------------------------

class TemporalFusionTransformerClassifier(nn.Module):
    """
    Paper-aligned TFT backbone + classification head.
    Keeps your GRU-like signature and "MLP head" pattern.

    INPUT expected: x [B, in_channels, T]
    OUTPUT: y_hat [B, out_channels] (softmax if classification)

    Notes:
    - This is the "model-only" piece; your pipeline can handle known/unknown splitting.
    - For alignment with paper, you can pass:
        * static_context: [B, d_static] (e.g. dept embedding already computed)
        * known_future:   [B, F_known, T] if you want to fuse known/unknown separately (optional)
      If you don't have these, just use x and it still runs as a TFT-like model.
    """
    def __init__(
        self,
        in_channels: int,
        tft_size: int,
        hidden_channels: int,
        end_channels: int,
        n_sequences: int,
        device,
        act_func: str = "ELU",
        task_type: str = "classification",
        dropout: float = 0.0,
        num_layers: int = 1,  # for LSTM encoder/decoder
        return_hidden: bool = False,
        out_channels: Optional[int] = None,
        use_layernorm: bool = True,   # paper uses LayerNorm in GRN; keep LN by default
        horizon: int = 1,             # paper does multi-horizon; you can set 1
        n_heads: int = 4,
        d_grn: Optional[int] = None,
        d_static: int = 0,            # 0 means "no static context"
        use_positional_encoding: bool = False,  # paper doesn't require PE; keep optional,
        static_idx = None,
        temporal_idx = None,
    ):
        super().__init__()
        self.device = device
        self.return_hidden = return_hidden
        self.task_type = task_type
        self.is_graph_or_node = False
        self.tft_size = tft_size
        self.end_channels = end_channels
        self.n_sequences = n_sequences
        self.horizon = int(horizon) if horizon is not None else 1
        self.out_channels = out_channels if out_channels is not None else 4
        self.d_static = int(d_static)
        self.static_idx = static_idx
        self.temporal_idx = temporal_idx

        if temporal_idx is not None:
            in_channels = len(temporal_idx)

        d_grn = d_grn or max(64, tft_size)

        # Optional static context processing (paper uses static covariate encoders -> contexts)
        # If you already have a dept embedding, pass it as static_context in forward.
        if self.d_static > 0:
            self.static_encoder = GatedResidualNetwork(
                d_in=self.d_static, d_hidden=d_grn, d_out=self.d_static,
                d_context=None, dropout=dropout, act_func=act_func
            )
        else:
            self.static_encoder = None

        # Variable selection over all features (paper has separate VSNs for past/future/known/unknown;
        # you can do that outside or extend this to multiple VSNs.)
        self.vsn = VariableSelectionNetwork(
            num_features=in_channels,
            d_model=tft_size,
            d_hidden=d_grn,
            d_static=self.d_static if self.d_static > 0 else None,
            dropout=dropout,
            act_func=act_func
        )

        self.pos_enc = PositionalEncoding(tft_size) if use_positional_encoding else nn.Identity()

        # LSTM encoder/decoder (sequence-to-sequence)
        self.encoder = nn.LSTM(
            input_size=tft_size,
            hidden_size=tft_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=tft_size,
            hidden_size=tft_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Static enrichment (paper: enrich decoder outputs with static context via GRN)
        self.static_enrichment = GatedResidualNetwork(
            d_in=tft_size,
            d_hidden=d_grn,
            d_out=tft_size,
            d_context=self.d_static if self.d_static > 0 else None,
            dropout=dropout,
            act_func=act_func
        )

        # Interpretable Multi-head Attention (paper)
        self.imha = InterpretableMultiHeadAttention(d_model=tft_size, n_heads=n_heads, dropout=dropout)
        self.attn_grn = GatedResidualNetwork(
            d_in=tft_size, d_hidden=d_grn, d_out=tft_size,
            d_context=None, dropout=dropout, act_func=act_func
        )

        # Position-wise feed-forward GRN (paper)
        self.pos_ff = GatedResidualNetwork(
            d_in=tft_size, d_hidden=d_grn, d_out=tft_size,
            d_context=None, dropout=dropout, act_func=act_func
        )

        # Normalization in the "head path" (to mimic your GRU style)
        if use_layernorm:
            self.norm = nn.LayerNorm(tft_size).to(device)
        else:
            self.norm = nn.BatchNorm1d(tft_size).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

        # MLP head same pattern as your GRU
        self.linear1 = nn.Linear(tft_size, hidden_channels).to(device)
        self.linear2 = nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = nn.Linear(end_channels, self.out_channels).to(device)

        self.act_func1 = getattr(nn, act_func)()
        self.act_func2 = getattr(nn, act_func)()
        self.act_func = getattr(nn, act_func)()

        if task_type in ("classification", "binary"):
            self.output_activation = nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = nn.Identity().to(device)

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        return_extras: bool = False,
        z_prev = None
    ):
        """
        x: [B, in_channels, T]
        static_context: [B, d_static] (optional, if d_static>0)
        """
        static_context = x[:, self.static_idx, -1]
        x = x[:, self.temporal_idx, :]

        x = x.to(self.device)
        static_context = static_context.to(self.device)
        x = x.permute(0, 2, 1).contiguous()  # -> [B,T,F]
        B, T, F = x.shape

        # static context encoding (paper)
        if self.static_encoder is not None:
            if static_context is None:
                raise ValueError("d_static>0 but static_context was not provided")
            static_context = static_context.to(self.device)
            s = self.static_encoder(static_context)  # [B,d_static]
        else:
            s = None

        # 1) Variable selection (paper)
        v, vsn_w = self.vsn(x, static_context=s)    # v: [B,T,d_model]
        v = self.pos_enc(v)

        # 2) Seq2seq LSTM
        enc_out, (h_n, c_n) = self.encoder(v)       # [B,T,d_model]

        # decoder length = horizon (paper multi-horizon). For single horizon, this is 1.
        pred_len = max(1, self.horizon)

        # Paper uses known future inputs to the decoder. If you don't provide them here,
        # a common aligned fallback is to repeat last observed embedding.
        dec_in = enc_out[:, -1:, :].repeat(1, pred_len, 1)  # [B,pred_len,d_model]
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))       # [B,pred_len,d_model]

        # 3) Static enrichment (paper)
        dec_enriched = self.static_enrichment(
            dec_out,
            context=s.unsqueeze(1) if s is not None else None
        )  # [B,pred_len,d_model]

        # 4) Interpretable attention over time (paper)
        # In TFT, attention typically attends over the *encoder time steps*.
        # queries = enriched decoder steps, keys/values = enriched encoder steps.
        enc_enriched = self.static_enrichment(
            enc_out,
            context=s.unsqueeze(1) if s is not None else None
        )  # [B,T,d_model]

        attn_out, attn_weights = self.imha(dec_enriched, enc_enriched, enc_enriched, mask=None)  # [B,pred_len,d_model]
        # gated skip connection + GRN (paper "gated residual network" after attention)
        attn_out = self.attn_grn(attn_out)  # [B,pred_len,d_model]

        # 5) Position-wise feedforward GRN (paper)
        out = self.pos_ff(attn_out)         # [B,pred_len,d_model]

        # For 1 horizon, take last prediction step
        h = out[:, -1, :]                   # [B,d_model]

        # 6) Head path (GRU-like)
        if isinstance(self.norm, nn.BatchNorm1d):
            h = self.norm(h)
        else:
            h = self.norm(h)

        h = self.dropout(h)
        y = self.act_func1(self.linear1(h))
        y = self.act_func2(self.linear2(y))
        y = self.output_layer(y)

        output = self.output_activation(y)

        if return_extras:
            extras: Dict[str, torch.Tensor] = {
                "vsn_weights": vsn_w,           # [B,T,F]
                "attn_weights": attn_weights,   # [B,H,pred_len,T]
                "hidden": h,                    # [B,d_model]
            }
            return output, y, extras

        return output, y, h