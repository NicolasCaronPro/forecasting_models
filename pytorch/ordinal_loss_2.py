import matplotlib
matplotlib.use('Agg')
from typing import Optional, Union, List, Dict, Any, Tuple

import os
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from forecasting_models.pytorch.loss_utils import *

import numpy as np

class OrdinalUncertaintyFocalWKLoss(torch.nn.modules.loss._WeightedLoss):
    """
    Loss ordinale avec incertitude de label.

    Sortie attendue du modèle :
        logits: Tensor de taille (B, 6)

    Colonnes :
        logits[:, 0:5] : logits des classes 0, 1, 2, 3, 4
        logits[:, 5]   : logit d'incertitude

    Convention :
        uncertainty = sigmoid(logits[:, 5])

    Plus uncertainty est grande, plus le label dur y est diffusé
    autour de sa classe observée.

    Loss totale :
        L = L_soft_focal
            + C * L_WK
            + lambda_coverage * L_distribution_coverage
            + lambda_similarity * L_similarity
            + lambda_uncertainty * L_uncertainty

    Coverage :
        Pour chaque cluster g, on maintient une EMA de la distribution moyenne
        prédite par le modèle :

            m_g(t) = beta * m_g(t-1) + (1 - beta) * mean_i p_i

        puis on compare m_g(t) à la distribution empirique cible q_g,
        éventuellement décalée vers le haut ou vers le bas avec
        coverage_risk_bias.

    Important :
        WKLoss doit déjà exister dans ton code.
    """

    def __init__(
        self,
        num_classes: int = 5,
        id: int = 0,

        # Poids de WKLoss
        C: float = 1.0,
        learnedC: bool = False,

        # Focal loss
        gamma: float = 2.0,
        alpha: Union[float, List[float], Tensor] = 1.0,

        # WKLoss existante
        wkpenalizationtype: str = "quadratic",
        uselogits: bool = True,

        # Incertitude / label smoothing ordinal
        sigmasmooth: float = 0.75,
        detachuncertaintyintarget: bool = False,
        lambdauncertainty: float = 0.01,

        # Coverage distributionnelle par cluster
        lambdacoverage: float = 0.1,
        defaultcoverage: Optional[Union[List[float], Tensor]] = None,
        autoregisterunknownclusters: bool = True,

        # Similarité hidden optionnelle
        usesimilarity: bool = True,
        lambdasimilarity: float = 0.5,
        similaritytau: float = 0.85,
        similaritymaxlabelgap: Optional[int] = 1,
        similaritypositivemode: str = "none",  # "none", "both", "at_least_one"
        detachhiddensimilarity: bool = True,

        # Arguments classiques
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        eps: float = 1e-8,

        # Coverage EMA
        coverageemamomentum: float = 0.95,
        coveragewarmupupdates: int = 5,

        riskbias: float = 1.0,
        riskbiastargetstrength: float = 0.5,
        riskbiascoveragestrength: float = 0.75,
        coverageriskweight: float = 0.0,
    ) -> None:

        super(OrdinalUncertaintyFocalWKLoss, self).__init__(
            weight=weight,
            size_average=None,
            reduce=None,
            reduction=reduction,
        )

        self.num_classes = int(num_classes)
        self.id = int(id)
        self.eps = float(eps)

        self.gamma = float(gamma)
        self.sigma_smooth = float(sigmasmooth)
        self.detach_uncertainty_in_target = bool(detachuncertaintyintarget)
        self.lambda_uncertainty = float(lambdauncertainty)

        self.lambda_coverage = float(lambdacoverage)
        self.auto_register_unknown_clusters = bool(autoregisterunknownclusters)

        self.use_similarity = bool(usesimilarity)
        self.lambda_similarity = float(lambdasimilarity)
        self.similarity_tau = float(similaritytau)
        self.similarity_max_label_gap = similaritymaxlabelgap
        self.similarity_positive_mode = similaritypositivemode
        self.detach_hidden_similarity = bool(detachhiddensimilarity)

        self.use_logits = bool(uselogits)
        self.wk_penalization_type = wkpenalizationtype

        if learnedC:
            self.C = torch.nn.Parameter(torch.tensor(float(C)))
        else:
            self.C = torch.nn.Parameter(
                torch.tensor(float(C)),
                requires_grad=False,
            )
            
        # Alpha focal
        if isinstance(alpha, Tensor):
            alpha_tensor = alpha.float()
        elif isinstance(alpha, list) or isinstance(alpha, np.ndarray):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        else:
            alpha_tensor = torch.full(
                (self.num_classes,),
                float(alpha),
                dtype=torch.float32,
            )

        if alpha_tensor.numel() != self.num_classes:
            raise ValueError(
                "alpha doit être un scalaire ou une liste/tensor "
                "de taille {}.".format(self.num_classes)
            )

        self.register_buffer("alpha", alpha_tensor)

        # Distribution cible par défaut pour les clusters inconnus.
        # Ancien nom conservé pour compatibilité API :
        # defaultcoverage signifie maintenant "default target distribution".
        if defaultcoverage is None:
            default_coverage_tensor = torch.ones(
                self.num_classes,
                dtype=torch.float32,
            ) / float(self.num_classes)
        else:
            default_coverage_tensor = torch.as_tensor(
                defaultcoverage,
                dtype=torch.float32,
            )

        if default_coverage_tensor.numel() != self.num_classes:
            raise ValueError(
                "defaultcoverage doit être de taille {}.".format(
                    self.num_classes
                )
            )

        default_coverage_tensor = self._normalize_distribution_static(
            default_coverage_tensor,
            eps=self.eps,
        )

        self.register_buffer("default_coverage", default_coverage_tensor)

        # Dictionnaire dynamique :
        # cluster_id -> distribution cible empirique q_g de taille num_classes.
        #
        # Nom conservé pour limiter les changements dans ton code existant.
        self.coverage_by_cluster = {}

        # EMA des distributions prédites :
        # cluster_id -> Tensor(num_classes)
        self.coverage_distribution_ema = {}

        # Nombre de mises à jour par cluster.
        self.coverage_update_count = {}

        self.coverage_ema_momentum = float(coverageemamomentum)
        self.coverage_warmup_updates = int(coveragewarmupupdates)

        self.risk_bias = float(riskbias)
        self.risk_bias_target_strength = float(riskbiastargetstrength)
        self.risk_bias_coverage_strength = float(riskbiascoveragestrength)
        self.coverage_risk_weight = float(coverageriskweight)
        
        # WKLoss est supposée déjà définie dans ton code.
        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )

    # ============================================================
    # Coverage distributionnelle par cluster
    # ============================================================

    @staticmethod
    def _normalize_distribution_static(x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Normalise une distribution 1D.
        """
        x = torch.clamp(x.float(), min=0.0)
        s = x.sum().clamp_min(eps)
        return x / s

    def calculate_class_coverage(
        self,
        df,
        cluster_col: str,
        target_col: str,
        dir_output=None,

        # Arguments conservés pour compatibilité.
        # Ils ne sont plus utilisés comme rho_min/rho_max.
        rho_min: Optional[Union[List[float], Tensor]] = None,
        rho_max: Optional[Union[List[float], Tensor]] = None,

        # Ici shrinkage devient un lissage de distribution :
        # q_g = (counts_g + shrinkage * q_global) / (n_g + shrinkage)
        shrinkage: float = 30.0,
        reset: bool = True,
    ) -> Dict[Any, Tensor]:
        """
        Calcule la distribution cible empirique par cluster :

            q_g(c) = P(y=c | cluster=g)

        avec shrinkage vers la distribution globale :

            q_g = (counts_g + shrinkage * q_global) / (n_g + shrinkage)

        Cela évite qu'un cluster peu représenté ait une distribution cible
        trop bruitée.

        Cette méthode remplace l'ancien sens de coverage :
            ancien : seuil de fiabilité par cluster/classe ;
            nouveau : distribution cible par cluster.
        """

        if cluster_col not in df.columns:
            raise ValueError("Colonne cluster absente : {}".format(cluster_col))

        if target_col not in df.columns:
            raise ValueError("Colonne target absente : {}".format(target_col))

        if reset:
            self.coverage_by_cluster = {}
            self.coverage_distribution_ema = {}
            self.coverage_update_count = {}

        d = df[[cluster_col, target_col]].dropna().copy()
        d[target_col] = d[target_col].astype(int)

        global_counts = (
            d[target_col]
            .value_counts()
            .reindex(range(self.num_classes), fill_value=0)
            .sort_index()
            .values
        )

        global_counts_tensor = torch.tensor(global_counts, dtype=torch.float32)
        global_dist = self._normalize_distribution_static(
            global_counts_tensor,
            eps=self.eps,
        )

        # Le fallback pour clusters inconnus devient la distribution globale.
        with torch.no_grad():
            self.default_coverage.copy_(
                global_dist.to(
                    device=self.default_coverage.device,
                    dtype=self.default_coverage.dtype,
                )
            )

        for cluster_value, dfg in d.groupby(cluster_col):
            counts = (
                dfg[target_col]
                .value_counts()
                .reindex(range(self.num_classes), fill_value=0)
                .sort_index()
                .values
            )

            counts_tensor = torch.tensor(counts, dtype=torch.float32)

            if shrinkage > 0.0:
                target_dist = (
                    counts_tensor + float(shrinkage) * global_dist
                ) / (
                    counts_tensor.sum() + float(shrinkage)
                ).clamp_min(self.eps)
            else:
                target_dist = self._normalize_distribution_static(
                    counts_tensor,
                    eps=self.eps,
                )

            target_dist = self._normalize_distribution_static(
                target_dist,
                eps=self.eps,
            )

            key = self._safe_cluster_key(cluster_value)
            self.coverage_by_cluster[key] = target_dist.detach().cpu()

        if dir_output is not None:
            try:
                os.makedirs(dir_output, exist_ok=True)

                path = os.path.join(
                    dir_output,
                    "class_coverage_distribution.csv",
                )

                with open(path, "w") as f:
                    f.write("cluster,class,probability\n")
                    for k, v in self.coverage_by_cluster.items():
                        v_np = v.numpy()
                        for c, p_val in enumerate(v_np):
                            f.write("{},{},{:.8f}\n".format(k, c, p_val))

                path_global = os.path.join(
                    dir_output,
                    "class_coverage_global_distribution.csv",
                )

                with open(path_global, "w") as f:
                    f.write("class,probability\n")
                    for c, p_val in enumerate(global_dist.numpy()):
                        f.write("{},{:.8f}\n".format(c, p_val))

                # Visualization of the distributions
                try:
                    import matplotlib.pyplot as plt
                    
                    # 1) Global distribution plot
                    plt.figure(figsize=(8, 4))
                    plt.bar(range(self.num_classes), global_dist.detach().cpu().numpy(), color='skyblue')
                    plt.xlabel('Class')
                    plt.ylabel('Probability')
                    plt.title('Empirical Global Class Distribution')
                    plt.xticks(range(self.num_classes))
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(dir_output, "global_distribution.png"))
                    plt.close()

                    # 2) Cluster-wise heatmap
                    sorted_keys = sorted(self.coverage_by_cluster.keys())
                    if sorted_keys:
                        dist_matrix = torch.stack([self.coverage_by_cluster[k] for k in sorted_keys]).numpy()
                        
                        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_keys) * 0.4)))
                        im = ax.imshow(dist_matrix, aspect='auto', cmap='viridis')
                        
                        plt.colorbar(im, ax=ax, label='Probability')
                        
                        ax.set_xticks(range(self.num_classes))
                        ax.set_xticklabels([str(c) for c in range(self.num_classes)])
                        ax.set_yticks(range(len(sorted_keys)))
                        ax.set_yticklabels([str(k) for k in sorted_keys])
                        
                        ax.set_xlabel('Class Index')
                        ax.set_ylabel('Cluster ID')
                        ax.set_title('Target Distributions by Cluster')

                        # Annotations
                        for i in range(dist_matrix.shape[0]):
                            for j in range(dist_matrix.shape[1]):
                                text_color = "white" if dist_matrix[i, j] < 0.4 else "black"
                                ax.text(j, i, f"{dist_matrix[i, j]:.2f}",
                                        ha="center", va="center", color=text_color, fontsize=8)

                        plt.tight_layout()
                        plt.savefig(os.path.join(dir_output, "class_coverage_heatmap.png"))
                        plt.close()

                    # 3) Risk Bias effect plot
                    plt.figure(figsize=(10, 5))
                    
                    x_axis = np.arange(self.num_classes)
                    width = 0.25
                    
                    global_np = global_dist.detach().cpu().numpy()
                    
                    # Compute shifted versions for visualization
                    # We use the internal self._shift_distribution to show what's happening
                    dist_target = self._shift_distribution(global_dist, self.risk_bias * self.risk_bias_target_strength)
                    target_np = dist_target.detach().cpu().numpy()
                    
                    dist_coverage = self._shift_distribution(global_dist, self.risk_bias * self.risk_bias_coverage_strength)
                    coverage_np = dist_coverage.detach().cpu().numpy()
                    
                    plt.bar(x_axis - width, global_np, width, label='Original Empirical', color='gray', alpha=0.5)
                    plt.bar(x_axis, target_np, width, label='Shifted (Target, strength={})'.format(self.risk_bias_target_strength), color='salmon')
                    plt.bar(x_axis + width, coverage_np, width, label='Shifted (Coverage, strength={})'.format(self.risk_bias_coverage_strength), color='mediumseagreen')
                    
                    plt.xlabel('Class Index')
                    plt.ylabel('Probability')
                    plt.title('Risk Bias Effect (bias={:.2f})'.format(self.risk_bias))
                    plt.xticks(x_axis)
                    plt.legend()
                    plt.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir_output, "risk_bias_effect.png"))
                    plt.close()

                except Exception as e:
                    print(f"Warning: could not generate plots in calculate_class_coverage: {e}")

            except Exception:
                pass

        return self.coverage_by_cluster

    def _safe_cluster_key(self, cluster_value: Any) -> Any:
        """
        Convertit proprement un id cluster en clé de dictionnaire.
        """

        if isinstance(cluster_value, Tensor):
            cluster_value = cluster_value.item()

        try:
            return int(cluster_value)
        except Exception:
            return cluster_value

    def _get_cluster_target_distribution(
        self,
        cluster_id: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Retourne la distribution cible q_g pour un cluster.

        Si le cluster est inconnu, on utilise self.default_coverage.
        """

        key = self._safe_cluster_key(cluster_id)

        if key not in self.coverage_by_cluster:
            if self.auto_register_unknown_clusters:
                self.coverage_by_cluster[key] = (
                    self.default_coverage.detach().cpu()
                )

            q = self.default_coverage
        else:
            q = self.coverage_by_cluster[key]

        q = q.to(device=device, dtype=dtype)
        q = q.clamp_min(0.0)
        q = q / q.sum().clamp_min(self.eps)

        coverage_shift = self.risk_bias * self.risk_bias_coverage_strength

        if coverage_shift != 0.0:
            q = self._shift_distribution(q, coverage_shift)

        return q

    def _shift_distribution(self, dist: Tensor, shift: float) -> Tensor:
        """
        Décale une distribution ordinale vers le haut ou vers le bas.

        shift > 0 :
            déplace la masse vers les classes plus hautes.

        shift < 0 :
            déplace la masse vers les classes plus basses.

        Exemple avec shift = 0.25 :
            une masse placée en classe 2 est redistribuée entre 2 et 3,
            avec 75% sur 2 et 25% sur 3.

        Cette opération conserve la somme des probabilités.
        """

        device = dist.device
        dtype = dist.dtype

        shifted = torch.zeros_like(dist)

        for c in range(self.num_classes):
            mass = dist[c]
            pos = float(c) + float(shift)

            if pos <= 0.0:
                shifted[0] = shifted[0] + mass

            elif pos >= float(self.num_classes - 1):
                shifted[self.num_classes - 1] = (
                    shifted[self.num_classes - 1] + mass
                )

            else:
                low = int(math.floor(pos))
                high = low + 1

                w_high = pos - float(low)
                w_low = 1.0 - w_high

                shifted[low] = shifted[low] + mass * torch.tensor(
                    w_low,
                    device=device,
                    dtype=dtype,
                )

                shifted[high] = shifted[high] + mass * torch.tensor(
                    w_high,
                    device=device,
                    dtype=dtype,
                )

        shifted = shifted.clamp_min(0.0)
        shifted = shifted / shifted.sum().clamp_min(self.eps)

        return shifted

    def _get_distribution_ema_state(
        self,
        cluster_id: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, int]:
        """
        Retourne :
            - EMA de la distribution prédite pour le cluster ;
            - nombre de mises à jour observées.
        """

        key = self._safe_cluster_key(cluster_id)

        if key not in self.coverage_distribution_ema:
            # Initialisation neutre :
            # on initialise l'EMA avec la distribution cible du cluster.
            q_g = self._get_cluster_target_distribution(
                cluster_id=key,
                device=device,
                dtype=dtype,
            )

            self.coverage_distribution_ema[key] = q_g.detach().cpu()

        if key not in self.coverage_update_count:
            self.coverage_update_count[key] = 0

        ema = self.coverage_distribution_ema[key].to(
            device=device,
            dtype=dtype,
        )

        ema = ema / ema.sum().clamp_min(self.eps)

        update_count = int(self.coverage_update_count[key])

        return ema, update_count

    def _update_distribution_ema(
        self,
        updates: List[Tuple[Any, Tensor]],
    ) -> None:
        """
        Met à jour les EMA de distributions prédites par cluster.

        Chaque update contient :
            (cluster_id, batch_pred_distribution)
        """

        beta = self.coverage_ema_momentum

        with torch.no_grad():
            for cluster_id, batch_dist in updates:
                key = self._safe_cluster_key(cluster_id)

                batch_dist_cpu = batch_dist.detach().float().cpu()
                batch_dist_cpu = self._normalize_distribution_static(
                    batch_dist_cpu,
                    eps=self.eps,
                )

                if key not in self.coverage_distribution_ema:
                    self.coverage_distribution_ema[key] = batch_dist_cpu
                else:
                    old = self.coverage_distribution_ema[key]
                    new = beta * old + (1.0 - beta) * batch_dist_cpu
                    new = self._normalize_distribution_static(
                        new,
                        eps=self.eps,
                    )
                    self.coverage_distribution_ema[key] = new

                if key not in self.coverage_update_count:
                    self.coverage_update_count[key] = 1
                else:
                    self.coverage_update_count[key] += 1

    def _ordinal_cdf_distance(
        self,
        pred_dist: Tensor,
        target_dist: Tensor,
    ) -> Tensor:
        """
        Distance ordinale entre distributions.

        On compare les cumulées :

            sum_k (CDF_pred(k) - CDF_target(k))^2

        Cette forme est adaptée à des classes ordonnées.
        """

        pred_dist = pred_dist / pred_dist.sum().clamp_min(self.eps)
        target_dist = target_dist / target_dist.sum().clamp_min(self.eps)

        pred_cdf = torch.cumsum(pred_dist, dim=0)
        target_cdf = torch.cumsum(target_dist, dim=0)

        return (pred_cdf - target_cdf).pow(2).mean()

    def _distribution_mean(self, dist: Tensor) -> Tensor:
        """
        Espérance ordinale d'une distribution sur les classes.
        """

        classes = torch.arange(
            self.num_classes,
            device=dist.device,
            dtype=dist.dtype,
        )

        return (dist * classes).sum()

    def _coverage_loss(
        self,
        class_logits: Tensor,
        clusters_ids: Tensor,
    ) -> Tensor:
        """
        Coverage distributionnelle stabilisée par EMA.

        Pour chaque cluster g présent dans le batch :

            p_batch_g = mean_i softmax(logits_i)

            m_hat_g = beta * EMA_old_g + (1 - beta) * p_batch_g

            L_g = EMD_like_CDF(m_hat_g, q_g_shifted)

        où q_g_shifted est la distribution empirique du cluster,
        éventuellement déplacée vers le haut ou vers le bas avec
        coverage_risk_bias.
        """

        device = class_logits.device
        dtype = class_logits.dtype

        clusters_ids = clusters_ids.detach().view(-1)

        probs = F.softmax(class_logits, dim=1)

        beta = self.coverage_ema_momentum

        losses = []
        updates = []

        unique_clusters = torch.unique(clusters_ids)

        for g in unique_clusters:
            g_key = self._safe_cluster_key(g)
            mask_g = clusters_ids == g

            if not mask_g.any():
                continue

            batch_dist = probs[mask_g].mean(dim=0)
            batch_dist = batch_dist / batch_dist.sum().clamp_min(self.eps)

            old_ema, update_count = self._get_distribution_ema_state(
                cluster_id=g_key,
                device=device,
                dtype=dtype,
            )

            # EMA anticipée, différentiable par rapport au batch courant.
            ema_hat = beta * old_ema.detach() + (1.0 - beta) * batch_dist
            ema_hat = ema_hat / ema_hat.sum().clamp_min(self.eps)

            target_dist = self._get_cluster_target_distribution(
                cluster_id=g_key,
                device=device,
                dtype=dtype,
            )

            dist_loss = self._ordinal_cdf_distance(
                pred_dist=ema_hat,
                target_dist=target_dist,
            )

            total_g_loss = dist_loss

            # Optionnel : renforcer aussi l'alignement de l'espérance ordinale.
            if self.coverage_risk_weight > 0.0:
                pred_mean = self._distribution_mean(ema_hat)
                target_mean = self._distribution_mean(target_dist)

                mean_loss = (pred_mean - target_mean).pow(2)
                total_g_loss = total_g_loss + self.coverage_risk_weight * mean_loss

            # Warmup progressif pour les clusters peu vus.
            if self.coverage_warmup_updates > 0:
                warmup = min(
                    1.0,
                    float(update_count + 1) / float(self.coverage_warmup_updates),
                )
            else:
                warmup = 1.0

            warmup_tensor = torch.tensor(
                warmup,
                device=device,
                dtype=dtype,
            )

            losses.append(warmup_tensor * total_g_loss)

            updates.append((g_key, batch_dist.detach()))

        if len(updates) > 0 and self.training:
            self._update_distribution_ema(updates)

        if len(losses) == 0:
            return torch.zeros((), device=device, dtype=dtype)

        return torch.stack(losses).mean()

    # ============================================================
    # Sorties modèle et cibles ordinales
    # ============================================================

    def _extract_outputs(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sépare :
            - les logits de classification ;
            - le logit d'incertitude.
        """

        if logits.dim() != 2:
            raise ValueError(
                "logits doit être de forme (B, {}), reçu : {}.".format(
                    self.num_classes + 1,
                    tuple(logits.shape),
                )
            )

        expected_dim = self.num_classes + 1

        if logits.shape[1] != expected_dim:
            raise ValueError(
                "logits doit avoir {} colonnes : {} logits + 1 incertitude. "
                "Reçu : {}.".format(
                    expected_dim,
                    self.num_classes,
                    logits.shape[1],
                )
            )

        class_logits = logits[:, :self.num_classes]
        uncertainty_logits = logits[:, self.num_classes]

        return class_logits, uncertainty_logits

    def _to_class_index(self, y: Tensor) -> Tensor:
        """
        Accepte :
            - y sous forme indices : (B,)
            - y sous forme one-hot : (B, C)
        """

        if y.dim() > 1:
            y = torch.argmax(y, dim=1)

        return y.long()

    def _make_soft_ordinal_target(
        self,
        y: Tensor,
        uncertainty: Tensor,
    ) -> Tensor:
        """
        Construit la cible ordinale incertaine :

            q_i(c) = (1 - u_i) * one_hot(y_i)
                    + u_i * smooth_ordinal(y_i + shift, c)

        avec :
            risk_label_shift > 0 : surestimation sample-level
            risk_label_shift < 0 : sous-estimation sample-level

        Attention :
            risk_label_shift agit sur la cible individuelle.
            coverage_risk_bias agit sur la distribution cible par cluster.
        """

        device = y.device
        dtype = uncertainty.dtype

        y = y.long()

        classes = torch.arange(
            self.num_classes,
            device=device,
            dtype=dtype,
        )

        target_shift = self.risk_bias * self.risk_bias_target_strength

        y_center = y.to(dtype).view(-1, 1) + target_shift
        
        y_center = y_center.clamp(0.0, float(self.num_classes - 1))

        dist2 = (classes.view(1, -1) - y_center).pow(2)

        smooth = torch.exp(
            -dist2 / (2.0 * self.sigma_smooth ** 2 + self.eps)
        )

        smooth = smooth / smooth.sum(dim=1, keepdim=True).clamp_min(self.eps)

        one_hot = F.one_hot(
            y,
            num_classes=self.num_classes,
        ).to(dtype=dtype)

        u = uncertainty.view(-1, 1)

        if self.detach_uncertainty_in_target:
            u = u.detach()

        q = (1.0 - u) * one_hot + u * smooth
        q = q / q.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return q

    # ============================================================
    # Composantes de loss
    # ============================================================

    def _soft_focal_loss(
        self,
        class_logits: Tensor,
        soft_target: Tensor,
        sample_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Focal loss compatible avec des cibles soft.

        La focal loss classique utilise un label dur.
        Ici, on utilise q_i(c), la cible ordinale incertaine.
        """

        log_probs = F.log_softmax(class_logits, dim=1)
        probs = log_probs.exp()

        ce = -(soft_target * log_probs).sum(dim=1)

        pt = (soft_target * probs).sum(dim=1).clamp_min(self.eps)

        alpha = self.alpha.to(
            device=class_logits.device,
            dtype=class_logits.dtype,
        )

        alpha_t = (soft_target * alpha.view(1, -1)).sum(dim=1)

        focal = alpha_t * (1.0 - pt).pow(self.gamma) * ce

        return self._reduce(focal, sample_weight)

    def _similarity_loss(
        self,
        hidden: Optional[Tensor],
        class_logits: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Régularisation optionnelle de similarité.

        Si deux hidden states sont très similaires, on rapproche
        leurs espérances ordinales prédites :

            mu_i = sum_c c * p_i(c)

            L_sim = mean_{i,j} w_ij * (mu_i - mu_j)^2
        """

        device = class_logits.device
        dtype = class_logits.dtype

        if hidden is None:
            return torch.zeros((), device=device, dtype=dtype)

        if hidden.shape[0] != class_logits.shape[0]:
            raise ValueError(
                "hidden doit avoir le même batch size que class_logits."
            )

        h = hidden

        if self.detach_hidden_similarity:
            h = h.detach()

        h = F.normalize(h, p=2, dim=1)

        sim = h.matmul(h.t())

        batch_size = sim.shape[0]

        upper_mask = torch.triu(
            torch.ones(
                batch_size,
                batch_size,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        mask = upper_mask & (sim > self.similarity_tau)

        y_i = y.view(-1, 1)
        y_j = y.view(1, -1)

        label_gap = torch.abs(y_i - y_j)

        if self.similarity_max_label_gap is not None:
            mask = mask & (label_gap <= self.similarity_max_label_gap)

        if self.similarity_positive_mode == "both":
            mask = mask & ((y_i > 0) & (y_j > 0))

        elif self.similarity_positive_mode == "at_least_one":
            mask = mask & ((y_i > 0) | (y_j > 0))

        elif self.similarity_positive_mode == "none":
            pass

        else:
            raise ValueError(
                "similarity_positive_mode doit être "
                "'none', 'both' ou 'at_least_one'."
            )

        if not mask.any():
            return torch.zeros((), device=device, dtype=dtype)

        probs = F.softmax(class_logits, dim=1)

        classes = torch.arange(
            self.num_classes,
            device=device,
            dtype=dtype,
        )

        mu = (probs * classes.view(1, -1)).sum(dim=1)

        diff2 = (mu.view(-1, 1) - mu.view(1, -1)).pow(2)

        pair_weight = sim.detach().clamp_min(0.0).to(dtype=dtype)

        loss = pair_weight[mask] * diff2[mask]

        return loss.mean()

    def _reduce(
        self,
        loss_vec: Tensor,
        sample_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Réduction compatible avec sample_weight.
        """

        if sample_weight is not None:
            w = sample_weight.to(
                device=loss_vec.device,
                dtype=loss_vec.dtype,
            ).view(-1)

            return (loss_vec * w).sum() / w.sum().clamp_min(self.eps)

        if self.reduction == "mean":
            return loss_vec.mean()

        if self.reduction == "sum":
            return loss_vec.sum()

        if self.reduction == "none":
            return loss_vec

        raise ValueError("reduction inconnue : {}".format(self.reduction))

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        logits: Tensor,
        y: Tensor,
        clusters_ids: Tensor,
        hidden,
        sample_weight: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        class_logits, uncertainty_logits = self._extract_outputs(logits)

        y = self._to_class_index(y).to(class_logits.device)
        clusters_ids = clusters_ids.to(class_logits.device)

        uncertainty = torch.sigmoid(uncertainty_logits)

        soft_target = self._make_soft_ordinal_target(
            y=y,
            uncertainty=uncertainty,
        )

        focal_loss = self._soft_focal_loss(
            class_logits=class_logits,
            soft_target=soft_target,
            sample_weight=sample_weight,
        )

        wk_loss = self.wk(
            class_logits,
            y,
            sample_weight=sample_weight,
        )

        coverage_loss = self._coverage_loss(
            class_logits=class_logits,
            clusters_ids=clusters_ids,
        )

        if self.use_similarity and self.lambda_similarity > 0.0:
            similarity_loss = self._similarity_loss(
                hidden=hidden,
                class_logits=class_logits,
                y=y,
            )
        else:
            similarity_loss = torch.zeros(
                (),
                device=class_logits.device,
                dtype=class_logits.dtype,
            )

        uncertainty_loss = uncertainty.mean()

        if self.C.requires_grad:
            C_value = torch.sigmoid(self.C)
        else:
            C_value = self.C

        total_loss = (
            focal_loss
            + C_value * wk_loss
            + self.lambda_coverage * coverage_loss
            + self.lambda_similarity * similarity_loss
            + self.lambda_uncertainty * uncertainty_loss
        )

        return {
            "total_loss": total_loss,
            "focal": focal_loss,
            "wk": wk_loss,
            "coverage": coverage_loss,
            "similarity": similarity_loss,
            "uncertainty_penalty": uncertainty_loss,
            "mean_uncertainty": uncertainty.detach().mean(),
            "C": C_value.detach(),
            "id": torch.tensor(
                self.id,
                device=class_logits.device,
                dtype=torch.long,
            ),
        }
        
class ContextualOrdinalUncertaintyFocalWKLoss(OrdinalUncertaintyFocalWKLoss):
    """
    Loss ordinale contextuelle avec incertitude supervisée.

    Idée :
        Pour chaque sample identifié par (graph_id, date), on construit une cible
        contextuelle soft r_{g,t} à partir des labels observés dans la fenêtre
        temporelle [t-J, ..., t+J], uniquement à graph_id fixé.

        Le modèle prédit toujours :
            logits[:, 0:C] : logits de classes
            logits[:, C]   : logit d'incertitude

        La cible finale utilisée dans la focal loss est :

            y_tilde = (1 - u_pred) * one_hot(y)
                    + u_pred * r_context

        avec :
            u_pred = sigmoid(logit_uncertainty)

        L'incertitude est supervisée par :

            u_star = d_ord(one_hot(y), r_context)

        où d_ord est une distance entre CDF ordinales.

    Important :
        - calculate_class_coverage(...) retourne le dataframe modifié.
        - Les colonnes contextuelles sont calculées seulement si absentes.
        - Les dictionnaires internes ne stockent que les samples modifiés,
          pour économiser la mémoire.
        - prevent_downgrade=True interdit de déplacer la masse vers une classe
          strictement inférieure au label officiel.
    """

    def __init__(
        self,
        *args,

        # Colonnes attendues dans le dataframe / forward
        graphcol: str = "graph_id",
        datecol: str = "date",

        # Contexte temporel
        contextradius: int = 5,
        contextmode: str = "gaussian",
        contextlabelmode: str = "onehot",  # "onehot" ou "ordinal"
        contextsigmaordinal: float = 0.75,

        # Règle : ne jamais abaisser un label officiel
        preventdowngrade: bool = True,

        # Persistence comme dans PreprocessorConv :
        # si True, ignore la partie passée du noyau.
        persistence: bool = False,

        # Colonnes ajoutées au df
        contextprefix: str = "context",

        # Stockage mémoire :
        # False = stocke uniquement les samples réellement modifiés.
        storeallcontext: bool = False,

        # Seuil à partir duquel un sample est considéré modifié
        modifiedeps: float = 1e-6,

        # Coverage :
        # True => q_cluster calculé avec (1-u*) onehot + u* r_context
        # False => q_cluster calculé directement avec r_context pour les modifiés
        coverageuseustarmix: bool = True,
        
        diagnosticdepartmentcol: str = "departement",
        savecontextdiagnostics: bool = True,

        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        allowed_modes = {
            "laplace",
            "laplace+mean",
            "laplace+median",
            "mean",
            "sum",
            "max",
            "median",
            "gaussian",
            "cubic",
            "quartic",
            "circular",
        }

        if contextmode not in allowed_modes:
            raise ValueError(
                f"contextmode={contextmode} invalide. "
                f"Valeurs possibles : {sorted(allowed_modes)}"
            )

        if contextlabelmode not in {"onehot", "ordinal"}:
            raise ValueError(
                "contextlabelmode doit être 'onehot' ou 'ordinal'."
            )

        self.graph_col = graphcol
        self.date_col = datecol

        self.context_radius = int(contextradius)
        self.context_mode = contextmode
        self.context_label_mode = contextlabelmode
        self.context_sigma_ordinal = float(contextsigmaordinal)

        self.prevent_downgrade = bool(preventdowngrade)
        self.persistence = bool(persistence)

        self.context_prefix = contextprefix
        self.store_all_context = bool(storeallcontext)
        self.modified_eps = float(modifiedeps)
        self.coverage_use_ustar_mix = bool(coverageuseustarmix)

        # Dictionnaires compacts :
        # clé = (int(graph_id), int(date))
        # valeur = Tensor CPU de taille C, ou float, ou bool
        self.context_target_probs: Dict[Tuple[int, int], Tensor] = {}
        self.context_target_ustar: Dict[Tuple[int, int], float] = {}
        self.context_target_argmax: Dict[Tuple[int, int], int] = {}
        self.context_target_modified: Dict[Tuple[int, int], bool] = {}
        
        
        self.diagnostic_department_col = diagnosticdepartmentcol
        self.save_context_diagnostics = bool(savecontextdiagnostics)

    # ============================================================
    # Colonnes contextuelles
    # ============================================================

    def _context_prob_cols(self) -> List[str]:
        return [
            f"{self.context_prefix}_p{c}"
            for c in range(self.num_classes)
        ]

    def _context_required_cols(self) -> List[str]:
        return (
            self._context_prob_cols()
            + [
                f"{self.context_prefix}_argmax",
                f"{self.context_prefix}_u_star",
                f"{self.context_prefix}_modified",
            ]
        )

    def _has_context_columns(self, df) -> bool:
        return all(col in df.columns for col in self._context_required_cols())

    @staticmethod
    def _safe_int_key_value(x: Any) -> int:
        """
        Convertit proprement date / graph_id en int.
        Compatible avec des entiers stockés en float.
        """
        if isinstance(x, Tensor):
            x = x.item()
        return int(x)

    def _make_key(self, graph_id: Any, date: Any) -> Tuple[int, int]:
        return (
            self._safe_int_key_value(graph_id),
            self._safe_int_key_value(date),
        )

    # ============================================================
    # Noyaux temporels repris de PreprocessorConv
    # ============================================================

    def _kernel_size(self) -> int:
        """
        Fenêtre centrée [t-J, ..., t+J].
        Donc K = 2J + 1.

        Note :
            Ton ancien PreprocessorConv faisait parfois :
                kernel_size = (kernel * 2 + 1) + 2
            Ici on garde explicitement le sens demandé :
                date - J à date + J.
        """
        return 2 * self.context_radius + 1

    def _make_temporal_kernel(self, mode: Optional[str] = None) -> np.ndarray:
        """
        Construit un noyau temporel 1D.

        Les formes laplace, gaussian, cubic, quartic, circular et mean
        reprennent les fonctions de filtre données dans PreprocessorConv,
        mais normalisées ensuite pour produire une distribution de probabilité.
        """

        if mode is None:
            mode = self.context_mode

        K = self._kernel_size()

        if K <= 1:
            kernel = np.ones(1, dtype=np.float64)

        elif mode == "laplace":
            offsets = np.arange(-(K // 2), K // 2 + 1)
            kernel = 1.0 / (np.abs(offsets).astype(np.float64) + 1.0)

        elif mode == "gaussian":
            sigma = (K - 1) / 6.0
            if sigma <= 0:
                kernel = np.ones(K, dtype=np.float64)
            else:
                # Reprise volontaire de la logique PreprocessorConv.
                x = np.linspace(-K // 2, K // 2 + 1, K)
                kernel = np.exp(-0.5 * (x / sigma) ** 2)

        elif mode == "cubic":
            x = np.linspace(-1, 1, K)
            kernel = (1.0 - np.abs(x)) ** 3
            kernel = np.clip(kernel, 0.0, None)

        elif mode == "quartic":
            x = np.linspace(-1, 1, K)
            kernel = (1.0 - x ** 2) ** 2
            kernel = np.clip(kernel, 0.0, None)

        elif mode == "circular":
            x = np.linspace(-1, 1, K)
            kernel = np.sqrt(np.clip(1.0 - x ** 2, 0.0, None))

        elif mode in {"mean", "sum", "median", "max"}:
            kernel = np.ones(K, dtype=np.float64)

        else:
            raise ValueError(f"Mode noyau non supporté ici : {mode}")

        if self.persistence and K > 1:
            # Même idée que dans PreprocessorConv :
            # on ignore la moitié passée.
            kernel[: K // 2] = 0.0

        if np.all(kernel <= 0):
            kernel = np.ones(K, dtype=np.float64)

        kernel = kernel.astype(np.float64)
        kernel = kernel / max(kernel.sum(), self.eps)

        return kernel

    # ============================================================
    # Encodage local des labels
    # ============================================================

    def _label_to_distribution_np(self, y: int) -> np.ndarray:
        """
        Convertit un label en distribution de classe.

        context_label_mode="onehot":
            y -> one-hot

        context_label_mode="ordinal":
            y -> distribution gaussienne ordinale autour de y
        """
        y = int(y)
        y = max(0, min(self.num_classes - 1, y))

        if self.context_label_mode == "onehot":
            out = np.zeros(self.num_classes, dtype=np.float64)
            out[y] = 1.0
            return out

        classes = np.arange(self.num_classes, dtype=np.float64)
        sigma = max(self.context_sigma_ordinal, self.eps)

        dist = np.exp(-((classes - float(y)) ** 2) / (2.0 * sigma ** 2))
        dist = dist / max(dist.sum(), self.eps)

        return dist.astype(np.float64)

    def _one_hot_np(self, y: int) -> np.ndarray:
        y = int(y)
        y = max(0, min(self.num_classes - 1, y))
        out = np.zeros(self.num_classes, dtype=np.float64)
        out[y] = 1.0
        return out

    def _apply_prevent_downgrade_np(
        self,
        dist: np.ndarray,
        y_official: int,
    ) -> np.ndarray:
        """
        Interdit toute masse sur les classes < y_official.

        Exemple :
            y=1 => p0 forcée à 0, mais p2/p3/p4 autorisées.
        """
        dist = np.asarray(dist, dtype=np.float64).copy()
        y_official = int(y_official)

        if not self.prevent_downgrade:
            s = dist.sum()
            if s <= self.eps:
                return self._one_hot_np(y_official)
            return dist / s

        if y_official > 0:
            dist[:y_official] = 0.0

        s = dist.sum()

        if s <= self.eps:
            return self._one_hot_np(y_official)

        return dist / s

    def _ordinal_cdf_distance_np(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """
        Distance ordinale entre deux distributions via leurs CDF.

        Valeur dans [0, 1] environ, car on utilise sqrt(mean(diff^2)).
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        p = p / max(p.sum(), self.eps)
        q = q / max(q.sum(), self.eps)

        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)

        return float(np.sqrt(np.mean((cdf_p - cdf_q) ** 2)))

    # ============================================================
    # Construction des cibles contextuelles
    # ============================================================

    def _compute_context_distribution_for_group(
        self,
        dates: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Calcule r_{g,t} pour un graph_id donné.

        dates:
            array de dates entières.

        labels:
            array de labels officiels associés.

        Retour:
            dict date -> distribution contextuelle r_t.
        """

        dates = np.asarray(dates).astype(int)
        labels = np.asarray(labels).astype(int)

        by_date = {
            int(d): int(y)
            for d, y in zip(dates, labels)
        }

        kernel = self._make_temporal_kernel(self.context_mode)
        offsets = np.arange(-self.context_radius, self.context_radius + 1)

        out: Dict[int, np.ndarray] = {}

        for t in dates:
            t = int(t)

            if self.context_mode in {"laplace+mean", "laplace+median"}:
                r_laplace = self._compute_single_context_distribution(
                    t=t,
                    by_date=by_date,
                    offsets=offsets,
                    kernel=self._make_temporal_kernel("laplace"),
                    reducer="weighted_sum",
                )

                if self.context_mode == "laplace+mean":
                    r_other = self._compute_single_context_distribution(
                        t=t,
                        by_date=by_date,
                        offsets=offsets,
                        kernel=self._make_temporal_kernel("mean"),
                        reducer="weighted_sum",
                    )
                else:
                    r_other = self._compute_single_context_distribution(
                        t=t,
                        by_date=by_date,
                        offsets=offsets,
                        kernel=self._make_temporal_kernel("median"),
                        reducer="median",
                    )

                r = r_laplace + r_other
                r = r / max(r.sum(), self.eps)

            elif self.context_mode == "max":
                r = self._compute_single_context_distribution(
                    t=t,
                    by_date=by_date,
                    offsets=offsets,
                    kernel=kernel,
                    reducer="max",
                )

            elif self.context_mode == "median":
                r = self._compute_single_context_distribution(
                    t=t,
                    by_date=by_date,
                    offsets=offsets,
                    kernel=kernel,
                    reducer="median",
                )

            else:
                r = self._compute_single_context_distribution(
                    t=t,
                    by_date=by_date,
                    offsets=offsets,
                    kernel=kernel,
                    reducer="weighted_sum",
                )

            y_official = by_date[t]
            r = self._apply_prevent_downgrade_np(r, y_official)

            out[t] = r.astype(np.float32)

        return out

    def _compute_single_context_distribution(
        self,
        t: int,
        by_date: Dict[int, int],
        offsets: np.ndarray,
        kernel: np.ndarray,
        reducer: str = "weighted_sum",
    ) -> np.ndarray:
        """
        Calcule la cible contextuelle pour une seule date t.
        """

        vectors = []
        weights = []

        for off, w in zip(offsets, kernel):
            tt = int(t + off)

            if tt not in by_date:
                continue

            y_neighbor = by_date[tt]
            vec = self._label_to_distribution_np(y_neighbor)

            vectors.append(vec)
            weights.append(float(w))

        if len(vectors) == 0:
            return self._one_hot_np(by_date[t])

        V = np.stack(vectors, axis=0).astype(np.float64)
        W = np.asarray(weights, dtype=np.float64)

        if reducer == "weighted_sum":
            W = W / max(W.sum(), self.eps)
            r = (V * W[:, None]).sum(axis=0)

        elif reducer == "max":
            # Version soft du max :
            # une classe reçoit de la masse si elle apparaît dans la fenêtre.
            # Puis normalisation.
            r = V.max(axis=0)

        elif reducer == "median":
            r = np.median(V, axis=0)

        else:
            raise ValueError(f"reducer inconnu : {reducer}")

        r = np.clip(r, 0.0, None)

        if r.sum() <= self.eps:
            return self._one_hot_np(by_date[t])

        return r / r.sum()

    # ============================================================
    # Dictionnaires compacts
    # ============================================================

    def _clear_context_dicts(self) -> None:
        self.context_target_probs = {}
        self.context_target_ustar = {}
        self.context_target_argmax = {}
        self.context_target_modified = {}

    def _register_context_sample(
        self,
        key: Tuple[int, int],
        probs: np.ndarray,
        u_star: float,
        argmax: int,
        modified: bool,
    ) -> None:
        """
        Stocke le sample seulement si nécessaire, sauf store_all_context=True.
        """
        if (not self.store_all_context) and (not modified):
            return

        probs_tensor = torch.tensor(
            probs,
            dtype=torch.float32,
            device="cpu",
        )

        self.context_target_probs[key] = probs_tensor
        self.context_target_ustar[key] = float(u_star)
        self.context_target_argmax[key] = int(argmax)
        self.context_target_modified[key] = bool(modified)

    def _rebuild_context_dicts_from_df(
        self,
        df,
        target_col: str,
    ) -> None:
        """
        Reconstruit les dictionnaires internes depuis les colonnes du dataframe.

        Pour économiser la mémoire, on ne stocke que les samples modifiés,
        sauf si store_all_context=True.
        """
        self._clear_context_dicts()

        prob_cols = self._context_prob_cols()
        arg_col = f"{self.context_prefix}_argmax"
        u_col = f"{self.context_prefix}_u_star"
        mod_col = f"{self.context_prefix}_modified"

        cols = [self.graph_col, self.date_col, target_col] + prob_cols + [arg_col, u_col, mod_col]
        n_probs = len(prob_cols)

        for row in df[cols].itertuples(index=False, name=None):

            graph_id = row[0]
            date = row[1]
            y = int(row[2])

            probs = np.array(
                [float(v) for v in row[3 : 3 + n_probs]],
                dtype=np.float64,
            )
            probs = probs / max(probs.sum(), self.eps)

            argmax = int(row[3 + n_probs])
            u_star = float(row[4 + n_probs])
            modified = bool(row[5 + n_probs])

            # Sécurité : si prevent_downgrade est actif, on ne stocke pas
            # une distribution incohérente.
            if self.prevent_downgrade:
                probs = self._apply_prevent_downgrade_np(probs, y)
                argmax = int(np.argmax(probs))
                u_star = self._ordinal_cdf_distance_np(
                    self._one_hot_np(y),
                    probs,
                )
                modified = bool(u_star > self.modified_eps)

            key = self._make_key(graph_id, date)

            self._register_context_sample(
                key=key,
                probs=probs,
                u_star=u_star,
                argmax=argmax,
                modified=modified,
            )

    # ============================================================
    # Coverage + pré-calcul contextuel
    # ============================================================

    def calculate_class_coverage(
        self,
        df,
        cluster_col: str,
        target_col: str,
        dir_output=None,

        rho_min: Optional[Union[List[float], Tensor]] = None,
        rho_max: Optional[Union[List[float], Tensor]] = None,

        shrinkage: float = 30.0,
        reset: bool = True,
    ):
        """
        Calcule :
            1. les cibles contextuelles soft par (graph_id, date),
               si les colonnes n'existent pas déjà ;
            2. les dictionnaires compacts pour le forward ;
            3. coverage_by_cluster à partir des cibles effectives soft.

        Retourne :
            df enrichi avec :
                context_p0, ..., context_p4,
                context_argmax,
                context_u_star,
                context_modified
        """

        if cluster_col not in df.columns:
            raise ValueError(f"Colonne cluster absente : {cluster_col}")

        if target_col not in df.columns:
            raise ValueError(f"Colonne target absente : {target_col}")

        if self.graph_col not in df.columns:
            raise ValueError(f"Colonne graph absente : {self.graph_col}")

        if self.date_col not in df.columns:
            raise ValueError(f"Colonne date absente : {self.date_col}")

        if reset:
            self.coverage_by_cluster = {}
            self.coverage_distribution_ema = {}
            self.coverage_update_count = {}
            self._clear_context_dicts()

        df = df.copy()

        prob_cols = self._context_prob_cols()
        arg_col = f"{self.context_prefix}_argmax"
        u_col = f"{self.context_prefix}_u_star"
        mod_col = f"{self.context_prefix}_modified"

        # --------------------------------------------------------
        # 1. Calcul des colonnes contextuelles si absentes
        # --------------------------------------------------------
        if not self._has_context_columns(df):
            for col in prob_cols:
                df[col] = np.nan

            df[arg_col] = -1
            df[u_col] = 0.0
            df[mod_col] = False

            work_cols = [self.graph_col, self.date_col, target_col]
            d = df[work_cols].dropna().copy()

            d[self.graph_col] = d[self.graph_col].astype(int)
            d[self.date_col] = d[self.date_col].astype(int)
            d[target_col] = d[target_col].astype(int)

            for graph_id, dfg in d.groupby(self.graph_col):
                dfg = dfg.sort_values(self.date_col)

                dates = dfg[self.date_col].values.astype(int)
                labels = dfg[target_col].values.astype(int)

                context_by_date = self._compute_context_distribution_for_group(
                    dates=dates,
                    labels=labels,
                )

                for idx, row in dfg.iterrows():
                    g = int(row[self.graph_col])
                    t = int(row[self.date_col])
                    y = int(row[target_col])

                    r = context_by_date[t]
                    e = self._one_hot_np(y)

                    u_star = self._ordinal_cdf_distance_np(e, r)
                    argmax = int(np.argmax(r))
                    modified = bool(u_star > self.modified_eps)

                    for c, col in enumerate(prob_cols):
                        df.at[idx, col] = float(r[c])

                    df.at[idx, arg_col] = argmax
                    df.at[idx, u_col] = float(u_star)
                    df.at[idx, mod_col] = modified

        # Si les colonnes existaient, on reconstruit les dictionnaires.
        # Si on vient de les calculer, même chose.
        self._rebuild_context_dicts_from_df(
            df=df,
            target_col=target_col,
        )

        # --------------------------------------------------------
        # 2. Coverage soft par cluster
        # --------------------------------------------------------
        d_cov = df[
            [cluster_col, target_col]
            + prob_cols
            + [u_col, mod_col]
        ].dropna().copy()

        d_cov[target_col] = d_cov[target_col].astype(int)

        # Distribution effective par sample :
        #   m_i = (1-u*) onehot(y_i) + u* r_i
        # ou :
        #   m_i = r_i pour samples modifiés, onehot sinon
        effective_targets = []

        cols = [target_col] + prob_cols + [u_col, mod_col]
        n_probs = len(prob_cols)

        for row in d_cov[cols].itertuples(index=False, name=None):

            y = int(row[0])

            r = np.array(
                [float(v) for v in row[1 : 1 + n_probs]],
                dtype=np.float64,
            )
            r = r / max(r.sum(), self.eps)

            u_star = float(row[1 + n_probs])
            modified = bool(row[2 + n_probs])

            e = self._one_hot_np(y)

            if not modified:
                m = e
            else:
                if self.coverage_use_ustar_mix:
                    m = (1.0 - u_star) * e + u_star * r
                else:
                    m = r

            m = m / max(m.sum(), self.eps)
            effective_targets.append(m.astype(np.float64))

        effective_targets = np.stack(effective_targets, axis=0)

        global_dist_np = effective_targets.mean(axis=0)
        global_dist_np = global_dist_np / max(global_dist_np.sum(), self.eps)

        global_dist = torch.tensor(global_dist_np, dtype=torch.float32)

        with torch.no_grad():
            self.default_coverage.copy_(
                global_dist.to(
                    device=self.default_coverage.device,
                    dtype=self.default_coverage.dtype,
                )
            )

        # Coverage par cluster avec shrinkage vers global_dist
        d_cov["_row_id_tmp_context_loss"] = np.arange(len(d_cov))

        for cluster_value, dfg in d_cov.groupby(cluster_col):
            row_ids = dfg["_row_id_tmp_context_loss"].values.astype(int)

            cluster_targets = effective_targets[row_ids]
            n_g = float(cluster_targets.shape[0])

            cluster_sum = cluster_targets.sum(axis=0)

            if shrinkage > 0.0:
                target_dist_np = (
                    cluster_sum + float(shrinkage) * global_dist_np
                ) / max(n_g + float(shrinkage), self.eps)
            else:
                target_dist_np = cluster_sum / max(cluster_sum.sum(), self.eps)

            target_dist_np = target_dist_np / max(target_dist_np.sum(), self.eps)

            key = self._safe_cluster_key(cluster_value)

            self.coverage_by_cluster[key] = torch.tensor(
                target_dist_np,
                dtype=torch.float32,
                device="cpu",
            )

        # --------------------------------------------------------
        # 3. Sauvegardes optionnelles
        # --------------------------------------------------------
        if dir_output is not None and self.save_context_diagnostics:
            try:
                self._plot_contextual_diagnostics(
                    df=df,
                    target_col=target_col,
                    cluster_col=cluster_col,
                    dir_output=dir_output,
                )
            except Exception as e:
                print(
                    "Warning: could not generate contextual diagnostics plots: "
                    f"{e}"
                )

        return df

    # ============================================================
    # Récupération batch des cibles contextuelles
    # ============================================================

    def _get_context_targets_batch(
        self,
        y: Tensor,
        graph_ids: Tensor,
        dates: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retourne pour le batch :
            context_probs : Tensor(B, C)
            u_star       : Tensor(B,)
            modified     : Tensor(B,)

        Si une clé est absente, fallback :
            context_probs = one_hot(y)
            u_star = 0
            modified = False
        """

        y = y.detach().long().view(-1)
        graph_ids = graph_ids.detach().view(-1)
        dates = dates.detach().view(-1)

        B = y.shape[0]

        context_probs = torch.zeros(
            B,
            self.num_classes,
            device=device,
            dtype=dtype,
        )

        u_star = torch.zeros(B, device=device, dtype=dtype)
        modified = torch.zeros(B, device=device, dtype=torch.bool)

        for i in range(B):
            yi = int(y[i].item())
            key = self._make_key(graph_ids[i], dates[i])

            if key in self.context_target_probs:
                probs_i = self.context_target_probs[key].to(
                    device=device,
                    dtype=dtype,
                )
                u_i = float(self.context_target_ustar.get(key, 0.0))
                m_i = bool(self.context_target_modified.get(key, True))

                probs_i = probs_i.clamp_min(0.0)
                probs_i = probs_i / probs_i.sum().clamp_min(self.eps)

                context_probs[i] = probs_i
                u_star[i] = torch.tensor(u_i, device=device, dtype=dtype)
                modified[i] = m_i

            else:
                context_probs[i] = F.one_hot(
                    y[i].clamp(0, self.num_classes - 1),
                    num_classes=self.num_classes,
                ).to(device=device, dtype=dtype)

        return context_probs, u_star, modified
    
    # ============================================================
    # Diagnostics plots pour les samples contextuellement modifiés
    # ============================================================

    def _plot_contextual_diagnostics(
        self,
        df,
        target_col: str,
        cluster_col: str,
        dir_output,
    ) -> None:
        """
        Génère plusieurs diagnostics visuels pour localiser les samples
        dits "louches", c'est-à-dire les samples dont la cible contextuelle
        diffère du label officiel.

        Plots sauvegardés :
            1. comparaison vraie classe vs classe contextuelle argmax ;
            2. matrice de transition true -> contextual argmax ;
            3. nombre de modifications par département ;
            4. proportion de modifications par département ;
            5. histogramme de u_star ;
            6. u_star par vraie classe ;
            7. modifications par vraie classe ;
            8. modifications par cluster ;
            9. série temporelle globale des modifications par date.
        """

        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        base_dir = os.path.join(dir_output, "contextual_diagnostics")
        os.makedirs(base_dir, exist_ok=True)

        prob_cols = self._context_prob_cols()
        arg_col = f"{self.context_prefix}_argmax"
        u_col = f"{self.context_prefix}_u_star"
        mod_col = f"{self.context_prefix}_modified"

        required_cols = [
            target_col,
            arg_col,
            u_col,
            mod_col,
            self.graph_col,
            self.date_col,
            cluster_col,
        ] + prob_cols

        missing = [c for c in required_cols if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(
                "Impossible de générer les diagnostics contextuels. "
                f"Colonnes manquantes : {missing}"
            )

        d = df[required_cols + [
            c for c in [self.diagnostic_department_col]
            if c in df.columns
        ]].copy()

        d[target_col] = d[target_col].astype(int)
        d[arg_col] = d[arg_col].astype(int)
        d[u_col] = d[u_col].astype(float)
        d[mod_col] = d[mod_col].astype(bool)
        d[self.graph_col] = d[self.graph_col].astype(int)
        d[self.date_col] = d[self.date_col].astype(int)

        # Sécurité : on force une colonne transition utile.
        d["_transition"] = (
            d[target_col].astype(str)
            + " -> "
            + d[arg_col].astype(str)
        )

        d["_delta_class"] = d[arg_col] - d[target_col]

        # --------------------------------------------------------
        # 1. Comparaison vraie classe vs classe contextuelle argmax
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "01_true_vs_context_argmax")
        os.makedirs(out_dir, exist_ok=True)

        counts_true = (
            d[target_col]
            .value_counts()
            .reindex(range(self.num_classes), fill_value=0)
            .sort_index()
        )

        counts_context = (
            d[arg_col]
            .value_counts()
            .reindex(range(self.num_classes), fill_value=0)
            .sort_index()
        )

        x = np.arange(self.num_classes)
        width = 0.38

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, counts_true.values, width, label="Official target")
        plt.bar(x + width / 2, counts_context.values, width, label="Context argmax")
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.title("Official class vs contextual argmax class")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "true_vs_context_argmax_counts.png"), dpi=200)
        plt.close()

        # Version normalisée.
        true_norm = counts_true.values / max(counts_true.values.sum(), self.eps)
        context_norm = counts_context.values / max(counts_context.values.sum(), self.eps)

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, true_norm, width, label="Official target")
        plt.bar(x + width / 2, context_norm, width, label="Context argmax")
        plt.xlabel("Class")
        plt.ylabel("Proportion")
        plt.title("Official class vs contextual argmax class — normalized")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "true_vs_context_argmax_proportions.png"), dpi=200)
        plt.close()

        # --------------------------------------------------------
        # 2. Matrice de transition true -> context argmax
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "02_transition_matrix")
        os.makedirs(out_dir, exist_ok=True)

        transition = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        for y_true, y_ctx in zip(d[target_col].values, d[arg_col].values):
            if 0 <= y_true < self.num_classes and 0 <= y_ctx < self.num_classes:
                transition[int(y_true), int(y_ctx)] += 1.0

        transition_norm = transition / np.maximum(
            transition.sum(axis=1, keepdims=True),
            self.eps,
        )

        self._save_heatmap(
            matrix=transition,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Contextual argmax class",
            ylabel="Official class",
            title="Transition matrix: official class -> contextual argmax",
            path=os.path.join(out_dir, "transition_matrix_counts.png"),
            fmt=".0f",
        )

        self._save_heatmap(
            matrix=transition_norm,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Contextual argmax class",
            ylabel="Official class",
            title="Transition matrix: official class -> contextual argmax — row-normalized",
            path=os.path.join(out_dir, "transition_matrix_row_normalized.png"),
            fmt=".2f",
        )

        # --------------------------------------------------------
        # 3. Histogramme des modifications par département
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "03_modified_by_department")
        os.makedirs(out_dir, exist_ok=True)

        if self.diagnostic_department_col in d.columns:
            dep_col = self.diagnostic_department_col

            by_dep = (
                d.groupby(dep_col)
                .agg(
                    n_samples=(mod_col, "size"),
                    n_modified=(mod_col, "sum"),
                    mean_u_star=(u_col, "mean"),
                )
                .reset_index()
            )

            by_dep["modified_ratio"] = by_dep["n_modified"] / by_dep["n_samples"].clip(lower=1)
            by_dep = by_dep.sort_values("n_modified", ascending=False)

            by_dep.to_csv(
                os.path.join(out_dir, "modified_by_department.csv"),
                index=False,
            )

            self._save_barplot_from_dataframe(
                data=by_dep,
                x_col=dep_col,
                y_col="n_modified",
                title="Number of contextual modifications by department",
                ylabel="Number of modified samples",
                path=os.path.join(out_dir, "modified_count_by_department.png"),
                top_k=40,
            )

            self._save_barplot_from_dataframe(
                data=by_dep.sort_values("modified_ratio", ascending=False),
                x_col=dep_col,
                y_col="modified_ratio",
                title="Ratio of contextual modifications by department",
                ylabel="Modified ratio",
                path=os.path.join(out_dir, "modified_ratio_by_department.png"),
                top_k=40,
            )

            self._save_barplot_from_dataframe(
                data=by_dep.sort_values("mean_u_star", ascending=False),
                x_col=dep_col,
                y_col="mean_u_star",
                title="Mean contextual uncertainty by department",
                ylabel="Mean u_star",
                path=os.path.join(out_dir, "mean_ustar_by_department.png"),
                top_k=40,
            )

        else:
            # Fallback si la colonne département n'existe pas :
            # on utilise graph_id.
            by_graph = (
                d.groupby(self.graph_col)
                .agg(
                    n_samples=(mod_col, "size"),
                    n_modified=(mod_col, "sum"),
                    mean_u_star=(u_col, "mean"),
                )
                .reset_index()
            )

            by_graph["modified_ratio"] = by_graph["n_modified"] / by_graph["n_samples"].clip(lower=1)
            by_graph = by_graph.sort_values("n_modified", ascending=False)

            by_graph.to_csv(
                os.path.join(out_dir, "modified_by_graph_id.csv"),
                index=False,
            )

            self._save_barplot_from_dataframe(
                data=by_graph,
                x_col=self.graph_col,
                y_col="n_modified",
                title="Number of contextual modifications by graph_id",
                ylabel="Number of modified samples",
                path=os.path.join(out_dir, "modified_count_by_graph_id.png"),
                top_k=40,
            )

        # --------------------------------------------------------
        # 4. Histogrammes d'incertitude u_star
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "04_uncertainty_histograms")
        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.hist(d[u_col].values, bins=50)
        plt.xlabel("u_star")
        plt.ylabel("Number of samples")
        plt.title("Distribution of contextual uncertainty u_star")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ustar_hist_all_samples.png"), dpi=200)
        plt.close()

        d_mod = d[d[mod_col]]
        if len(d_mod) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(d_mod[u_col].values, bins=50)
            plt.xlabel("u_star")
            plt.ylabel("Number of modified samples")
            plt.title("Distribution of contextual uncertainty u_star — modified samples only")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ustar_hist_modified_samples.png"), dpi=200)
            plt.close()

        # Histogramme séparé par vraie classe.
        plt.figure(figsize=(9, 5))
        for c in range(self.num_classes):
            vals = d.loc[d[target_col] == c, u_col].values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=40, alpha=0.45, label=f"class {c}")
        plt.xlabel("u_star")
        plt.ylabel("Number of samples")
        plt.title("u_star distribution by official class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ustar_hist_by_true_class.png"), dpi=200)
        plt.close()

        # --------------------------------------------------------
        # 5. Nombre de modifications par vraie classe
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "05_modified_by_true_class")
        os.makedirs(out_dir, exist_ok=True)

        by_class = (
            d.groupby(target_col)
            .agg(
                n_samples=(mod_col, "size"),
                n_modified=(mod_col, "sum"),
                mean_u_star=(u_col, "mean"),
                mean_delta_class=("_delta_class", "mean"),
            )
            .reindex(range(self.num_classes), fill_value=0)
            .reset_index()
        )

        by_class["modified_ratio"] = by_class["n_modified"] / by_class["n_samples"].clip(lower=1)
        by_class.to_csv(
            os.path.join(out_dir, "modified_by_true_class.csv"),
            index=False,
        )

        plt.figure(figsize=(8, 5))
        plt.bar(by_class[target_col].astype(int).values, by_class["n_modified"].values)
        plt.xlabel("Official class")
        plt.ylabel("Number of modified samples")
        plt.title("Number of contextual modifications by official class")
        plt.xticks(range(self.num_classes))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "modified_count_by_true_class.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(by_class[target_col].astype(int).values, by_class["modified_ratio"].values)
        plt.xlabel("Official class")
        plt.ylabel("Modified ratio")
        plt.title("Ratio of contextual modifications by official class")
        plt.xticks(range(self.num_classes))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "modified_ratio_by_true_class.png"), dpi=200)
        plt.close()

        # --------------------------------------------------------
        # 6. Incertitude moyenne par vraie classe
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "06_uncertainty_by_true_class")
        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.bar(by_class[target_col].astype(int).values, by_class["mean_u_star"].values)
        plt.xlabel("Official class")
        plt.ylabel("Mean u_star")
        plt.title("Mean contextual uncertainty by official class")
        plt.xticks(range(self.num_classes))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mean_ustar_by_true_class.png"), dpi=200)
        plt.close()

        # --------------------------------------------------------
        # 7. Modifications par cluster
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "07_modified_by_cluster")
        os.makedirs(out_dir, exist_ok=True)

        by_cluster = (
            d.groupby(cluster_col)
            .agg(
                n_samples=(mod_col, "size"),
                n_modified=(mod_col, "sum"),
                mean_u_star=(u_col, "mean"),
            )
            .reset_index()
        )

        by_cluster["modified_ratio"] = by_cluster["n_modified"] / by_cluster["n_samples"].clip(lower=1)
        by_cluster = by_cluster.sort_values("n_modified", ascending=False)

        by_cluster.to_csv(
            os.path.join(out_dir, "modified_by_cluster.csv"),
            index=False,
        )

        self._save_barplot_from_dataframe(
            data=by_cluster,
            x_col=cluster_col,
            y_col="n_modified",
            title="Number of contextual modifications by cluster",
            ylabel="Number of modified samples",
            path=os.path.join(out_dir, "modified_count_by_cluster.png"),
            top_k=40,
        )

        self._save_barplot_from_dataframe(
            data=by_cluster.sort_values("modified_ratio", ascending=False),
            x_col=cluster_col,
            y_col="modified_ratio",
            title="Ratio of contextual modifications by cluster",
            ylabel="Modified ratio",
            path=os.path.join(out_dir, "modified_ratio_by_cluster.png"),
            top_k=40,
        )

        # --------------------------------------------------------
        # 8. Série temporelle des modifications
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "08_modified_by_date")
        os.makedirs(out_dir, exist_ok=True)

        by_date = (
            d.groupby(self.date_col)
            .agg(
                n_samples=(mod_col, "size"),
                n_modified=(mod_col, "sum"),
                mean_u_star=(u_col, "mean"),
            )
            .reset_index()
            .sort_values(self.date_col)
        )

        by_date["modified_ratio"] = by_date["n_modified"] / by_date["n_samples"].clip(lower=1)

        by_date.to_csv(
            os.path.join(out_dir, "modified_by_date.csv"),
            index=False,
        )

        plt.figure(figsize=(14, 5))
        plt.plot(by_date[self.date_col].values, by_date["n_modified"].values)
        plt.xlabel("Date index")
        plt.ylabel("Number of modified samples")
        plt.title("Number of contextual modifications by date")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "modified_count_by_date.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(14, 5))
        plt.plot(by_date[self.date_col].values, by_date["modified_ratio"].values)
        plt.xlabel("Date index")
        plt.ylabel("Modified ratio")
        plt.title("Ratio of contextual modifications by date")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "modified_ratio_by_date.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(14, 5))
        plt.plot(by_date[self.date_col].values, by_date["mean_u_star"].values)
        plt.xlabel("Date index")
        plt.ylabel("Mean u_star")
        plt.title("Mean contextual uncertainty by date")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mean_ustar_by_date.png"), dpi=200)
        plt.close()

        # --------------------------------------------------------
        # 9. Transitions les plus fréquentes
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "09_transition_counts")
        os.makedirs(out_dir, exist_ok=True)

        transition_counts = (
            d["_transition"]
            .value_counts()
            .reset_index()
        )
        transition_counts.columns = ["transition", "count"]

        transition_counts.to_csv(
            os.path.join(out_dir, "transition_counts.csv"),
            index=False,
        )

        self._save_barplot_from_dataframe(
            data=transition_counts,
            x_col="transition",
            y_col="count",
            title="Most frequent official -> contextual transitions",
            ylabel="Number of samples",
            path=os.path.join(out_dir, "transition_counts.png"),
            top_k=30,
        )

        # --------------------------------------------------------
        # 10. Exemples de samples les plus louches
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "10_top_suspicious_samples")
        os.makedirs(out_dir, exist_ok=True)

        export_cols = [
            self.graph_col,
            self.date_col,
            target_col,
            arg_col,
            u_col,
            mod_col,
            "_delta_class",
            "_transition",
            cluster_col,
        ] + prob_cols

        if self.diagnostic_department_col in d.columns:
            export_cols.insert(2, self.diagnostic_department_col)

        top_suspicious = (
            d.sort_values(u_col, ascending=False)
            .head(5000)
        )

        top_suspicious[export_cols].to_csv(
            os.path.join(out_dir, "top_suspicious_samples.csv"),
            index=False,
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        logits: Tensor,
        y: Tensor,
        clusters_ids: Tensor,
        hidden,
        graph_ids: Tensor,
        dates: Tensor,
        sample_weight: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        class_logits, uncertainty_logits = self._extract_outputs(logits)

        device = class_logits.device
        dtype = class_logits.dtype

        y = self._to_class_index(y).to(device)
        clusters_ids = clusters_ids.to(device)
        graph_ids = graph_ids.to(device)
        dates = dates.to(device)

        uncertainty = torch.sigmoid(uncertainty_logits)

        one_hot = F.one_hot(
            y.clamp(0, self.num_classes - 1),
            num_classes=self.num_classes,
        ).to(device=device, dtype=dtype)

        context_probs, u_star, modified = self._get_context_targets_batch(
            y=y,
            graph_ids=graph_ids,
            dates=dates,
            device=device,
            dtype=dtype,
        )

        # Cible finale :
        #   y_tilde = (1-u_pred) onehot + u_pred r_context
        #
        # Pour les samples non modifiés, context_probs = onehot et u_star=0,
        # donc la cible reste équivalente au label officiel.
        u_pred = uncertainty.view(-1, 1)

        soft_target = (1.0 - u_pred) * one_hot + u_pred * context_probs
        soft_target = soft_target / soft_target.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        focal_loss = self._soft_focal_loss(
            class_logits=class_logits,
            soft_target=soft_target,
            sample_weight=sample_weight,
        )

        wk_loss = self.wk(
            class_logits,
            y,
            sample_weight=sample_weight,
        )

        coverage_loss = self._coverage_loss(
            class_logits=class_logits,
            clusters_ids=clusters_ids,
        )

        if self.use_similarity and self.lambda_similarity > 0.0:
            similarity_loss = self._similarity_loss(
                hidden=hidden,
                class_logits=class_logits,
                y=y,
            )
        else:
            similarity_loss = torch.zeros(
                (),
                device=device,
                dtype=dtype,
            )

        # Incertitude supervisée par u_star.
        # On peut calculer sur tous les samples :
        # - si sample non modifié, u_star = 0.
        uncertainty_loss_vec = (uncertainty - u_star.detach()).pow(2)

        uncertainty_loss = self._reduce(
            uncertainty_loss_vec,
            sample_weight=sample_weight,
        )

        if self.C.requires_grad:
            C_value = torch.sigmoid(self.C)
        else:
            C_value = self.C

        total_loss = (
            focal_loss
            + C_value * wk_loss
            + self.lambda_coverage * coverage_loss
            + self.lambda_similarity * similarity_loss
            + self.lambda_uncertainty * uncertainty_loss
        )

        return {
            "total_loss": total_loss,
            "focal": focal_loss,
            "wk": wk_loss,
            "coverage": coverage_loss,
            "similarity": similarity_loss,
            "uncertainty_penalty": uncertainty_loss,
            "mean_uncertainty": uncertainty.detach().mean(),
            "mean_u_star": u_star.detach().mean(),
            "modified_ratio": modified.float().mean().detach(),
            "C": C_value.detach(),
            "id": torch.tensor(
                self.id,
                device=device,
                dtype=torch.long,
            ),
        }
        
    def _save_heatmap(
        self,
        matrix,
        x_labels,
        y_labels,
        xlabel: str,
        ylabel: str,
        title: str,
        path: str,
        fmt: str = ".2f",
    ) -> None:
        """
        Sauvegarde une heatmap simple avec annotations.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        matrix = np.asarray(matrix)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto")

        plt.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()


    def _save_barplot_from_dataframe(
        self,
        data,
        x_col: str,
        y_col: str,
        title: str,
        ylabel: str,
        path: str,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Sauvegarde un barplot depuis un dataframe.

        Si top_k est donné, garde les top_k plus grandes valeurs de y_col.
        """

        import matplotlib.pyplot as plt

        d = data.copy()

        if top_k is not None and len(d) > top_k:
            d = d.sort_values(y_col, ascending=False).head(top_k)

        d = d.copy()
        d[x_col] = d[x_col].astype(str)

        plt.figure(figsize=(max(10, 0.35 * len(d)), 5))
        plt.bar(d[x_col].values, d[y_col].values)
        plt.xlabel(x_col)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()