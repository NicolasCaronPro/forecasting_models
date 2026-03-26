import torch
import torch.nn as nn
from types import SimpleNamespace

# import officiel THUML
# adapte ce chemin selon ton projet
from GNN.forecasting_models.pytorch.iTransformer.model.iTransformer import Model as OfficialITransformer


class StaticMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class OfficialITransformerWrapper(nn.Module):
    """
    Wrapper autour du iTransformer officiel THUML.

    Hypothèse d'entrée dans forward:
        x est initialement [B, F, T]
    puis on le convertit en:
        [B, T, F]

    Le backbone officiel ne sert ici QUE de backbone de forecasting:
        dyn_out = [B, pred_len, F_dyn]

    Ensuite:
        - projection dynamique par horizon
        - concaténation avec les features statiques répétées sur chaque horizon
        - 1 head partagé pour tous les horizons
    """

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        temporal_idx,
        static_idx,
        output_dim: int,
        task_type: str = "classification",   # "classification" ou "regression"
        spatial_context: bool = False,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        factor: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        embed: str = "fixed",
        freq: str = "h",
        static_hidden_dim: int = 128,
        use_horizon_embedding: bool = True,
        horizon=0
    ):
        super().__init__()

        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.temporal_idx = list(temporal_idx)
        self.static_idx = list(static_idx)
        self.output_dim = output_dim
        self.horizon = horizon
        self.task_type = task_type
        self.spatial_context = spatial_context
        self.use_horizon_embedding = use_horizon_embedding
        
        print(self.horizon)

        self.n_temporal = len(self.temporal_idx)
        self.n_static = len(self.static_idx)

        print(f"static_idx: {self.n_static}")
        print(f"temporal_idx: {self.n_temporal}")

        if self.n_temporal == 0:
            raise ValueError("temporal_idx ne doit pas être vide.")

        # ------------------------------------------------------------------
        # 1) Backbone iTransformer OFFICIEL
        # ------------------------------------------------------------------
        # Le modèle officiel du repo principal est un forecast model:
        # sortie = [B, pred_len, N]
        # où N = nombre de variables données au backbone.
        configs = SimpleNamespace(
            seq_len=seq_len,
            pred_len=horizon + 1,
            enc_in=self.n_temporal,
            dec_in=self.n_temporal,
            c_out=self.n_temporal,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            activation=activation,
            embed=embed,
            freq=freq,
            output_attention=False,
            use_norm=True,
            class_strategy="projection",

            # champs supplémentaires souvent attendus par l'écosystème THUML
            task_name="short_term_forecast",
            label_len=0,
            d_layers=1,
            features="M",
            moving_avg=25,
            distil=True,
            num_class=output_dim,
        )

        self.dynamic_backbone = OfficialITransformer(configs)

        # ------------------------------------------------------------------
        # 2) Branche statique
        # ------------------------------------------------------------------
        if self.n_static > 0:
            self.static_encoder = StaticMLP(
                in_dim=self.n_static,
                out_dim=d_model,
                hidden_dim=static_hidden_dim,
                dropout=dropout,
            )
        else:
            self.static_encoder = None

        # ------------------------------------------------------------------
        # 3) Modulation statique optionnelle AVANT backbone
        # ------------------------------------------------------------------
        # On module les variables dynamiques brutes [B, T, F_dyn]
        if self.spatial_context and self.n_static > 0:
            self.gamma = nn.Linear(d_model, self.n_temporal)
            self.beta = nn.Linear(d_model, self.n_temporal)
        else:
            self.gamma = None
            self.beta = None

        # ------------------------------------------------------------------
        # 4) Projection de dyn_out par horizon
        # dyn_out = [B, H, F_dyn] -> dyn_repr = [B, H, d_model]
        # ------------------------------------------------------------------
        self.dynamic_projection = nn.Sequential(
            nn.Linear(self.n_temporal, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # ------------------------------------------------------------------
        # 5) Embedding d'horizon optionnel
        # ------------------------------------------------------------------
        if self.use_horizon_embedding:
            self.horizon_embedding = nn.Embedding(horizon + 1, d_model)
        else:
            self.horizon_embedding = None

        # ------------------------------------------------------------------
        # 6) Head partagé sur tous les horizons
        # Entrée par horizon:
        #   dyn_repr[h] + static_repr + optional horizon emb
        # ------------------------------------------------------------------
        fusion_in_dim = d_model
        if self.n_static > 0:
            fusion_in_dim += d_model
        if self.use_horizon_embedding:
            fusion_in_dim += d_model

        hidden_head = max(d_ff, 128)

        self.shared_head = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, output_dim),
        )

        self.softmax = nn.Softmax(dim=-1)

    def _split_inputs(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError("x doit être de forme [B, T, F] après moveaxis")

        B, T, F = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len attendu={self.seq_len}, reçu={T}")
        if F != self.input_dim:
            raise ValueError(f"input_dim attendu={self.input_dim}, reçu={F}")

        x_dyn = x[:, :, self.temporal_idx]  # [B, T, F_dyn]

        x_static = None
        if self.n_static > 0:
            # statiques supposées constantes dans le temps
            x_static = x[:, 0, self.static_idx]  # [B, F_static]

        return x_dyn, x_static

    def forward(self, x: torch.Tensor, z_prev=None):
        # ton pipeline semble fournir [B, F, T]
        # si un jour tu passes déjà [B, T, F], il faudra retirer cette ligne
        x = torch.moveaxis(x, 2, 1)  # [B, T, F]

        x_dyn, x_static = self._split_inputs(x)

        # --------------------------------------------------------------
        # 1) Encodage statique
        # --------------------------------------------------------------
        static_repr = None
        if self.static_encoder is not None and self.n_static > 0:
            static_repr = self.static_encoder(x_static)  # [B, d_model]

        # --------------------------------------------------------------
        # 2) Conditioning statique AVANT backbone
        # --------------------------------------------------------------
        if self.spatial_context and static_repr is not None:
            gamma = self.gamma(static_repr).unsqueeze(1)  # [B, 1, F_dyn]
            beta = self.beta(static_repr).unsqueeze(1)    # [B, 1, F_dyn]
            x_dyn = x_dyn * (1.0 + torch.tanh(gamma)) + beta

        # --------------------------------------------------------------
        # 3) Backbone officiel
        # dyn_out = [B, pred_len, F_dyn]
        # --------------------------------------------------------------
        dyn_out = self.dynamic_backbone(
            x_enc=x_dyn,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None,
            mask=None
        )

        # --------------------------------------------------------------
        # 4) Projection dynamique par horizon
        # dyn_repr = [B, H, d_model]
        # --------------------------------------------------------------
        dyn_repr = self.dynamic_projection(dyn_out)

        # --------------------------------------------------------------
        # 5) Fusion avec static par horizon
        # --------------------------------------------------------------
        fusion_parts = [dyn_repr]

        if static_repr is not None:
            static_rep = static_repr.unsqueeze(1).expand(-1, self.horizon + 1, -1)  # [B, H, d_model]
            fusion_parts.append(static_rep)

        if self.horizon_embedding is not None:
            horizon_ids = torch.arange(self.horizon + 1, device=x.device)  # [H]
            h_emb = self.horizon_embedding(horizon_ids)                 # [H, d_model]
            h_emb = h_emb.unsqueeze(0).expand(x.shape[0], -1, -1)      # [B, H, d_model]
            fusion_parts.append(h_emb)

        fused = torch.cat(fusion_parts, dim=-1)  # [B, H, fusion_in_dim]

        # --------------------------------------------------------------
        # 6) Un seul head partagé sur tous les horizons
        # out = [B, H, output_dim]
        # --------------------------------------------------------------
        logits = self.shared_head(fused)
        
        print(logits.shape)

        if self.task_type == "classification":
            probs = self.softmax(logits)
            return probs, logits, dyn_out

        return logits, logits, dyn_out