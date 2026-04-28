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
    Loss ordinale contextuelle avec détection de samples suspects par distance ordinale.

    Le modèle doit sortir :
        logits[:, 0:C] : logits de classes ordinales
        logits[:, C]   : logit d'incertitude

    Principe :
        1. Construire une cible contextuelle soft r_{g,t} à partir du voisinage
           temporel dans le même graph_id.
        2. Calculer une distance ordinale u_star entre onehot(y_{g,t}) et r_{g,t}.
        3. Détecter les samples suspects par seuil fixe ou quantile local.
        4. Pendant l'entraînement, corriger seulement les samples suspects si
           threshold_affects_training=True.
        5. Superviser l'incertitude du modèle par u_star ou gate * u_star.

    La méthode calculate_class_coverage retourne le dataframe enrichi.
    """

    def __init__(
        self,
        *args,

        graph_col: str = "graph_id",
        date_col: str = "date",

        context_radius: int = 5,
        context_mode: str = "gaussian",
        context_label_mode: str = "onehot",
        context_sigma_ordinal: float = 0.75,

        prevent_downgrade: bool = True,
        persistence: bool = False,

        context_prefix: str = "context",
        store_all_context: bool = False,
        modified_eps: float = 1e-6,

        coverage_use_ustar_mix: bool = True,

        suspicious_distance: str = "cdf_l2",
        suspicious_mode: str = "department_normalized_threshold",
        suspicious_threshold: float = 0.01,
        suspicious_quantile: float = 0.90,
        suspicious_group_col: Optional[str] = None,
        suspicious_normalize_group_col: Optional[str] = "departement",

        threshold_affects_training: bool = True,
        soft_suspicious_gate: bool = False,
        suspicious_gate_temperature: float = 0.05,

        save_context_diagnostics: bool = True,
        diagnostic_department_col: str = "departement",

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

        if context_mode not in allowed_modes:
            raise ValueError(
                f"context_mode={context_mode} invalide. "
                f"Valeurs possibles : {sorted(allowed_modes)}"
            )

        if context_label_mode not in {"onehot", "ordinal"}:
            raise ValueError("context_label_mode doit être 'onehot' ou 'ordinal'.")

        allowed_distances = {"cdf_l2", "cdf_l1", "cdf_linf", "mean_abs"}
        if suspicious_distance not in allowed_distances:
            raise ValueError(
                f"suspicious_distance={suspicious_distance} invalide. "
                f"Valeurs possibles : {sorted(allowed_distances)}"
            )

        allowed_suspicious_modes = {
            "none",
            "threshold",
            "quantile",
            "department_normalized_threshold",
        }
        if suspicious_mode not in allowed_suspicious_modes:
            raise ValueError(
                f"suspicious_mode={suspicious_mode} invalide. "
                f"Valeurs possibles : {sorted(allowed_suspicious_modes)}"
            )

        self.graph_col = graph_col
        self.date_col = date_col

        self.context_radius = int(context_radius)
        self.context_mode = context_mode
        self.context_label_mode = context_label_mode
        self.context_sigma_ordinal = float(context_sigma_ordinal)

        self.prevent_downgrade = bool(prevent_downgrade)
        self.persistence = bool(persistence)

        self.context_prefix = context_prefix
        self.store_all_context = bool(store_all_context)
        self.modified_eps = float(modified_eps)

        self.coverage_use_ustar_mix = bool(coverage_use_ustar_mix)

        self.suspicious_distance = suspicious_distance
        self.suspicious_mode = suspicious_mode
        self.suspicious_threshold = float(suspicious_threshold)
        self.suspicious_quantile = float(suspicious_quantile)
        self.suspicious_group_col = suspicious_group_col
        self.suspicious_normalize_group_col = suspicious_normalize_group_col

        self.threshold_affects_training = bool(threshold_affects_training)
        self.soft_suspicious_gate = bool(soft_suspicious_gate)
        self.suspicious_gate_temperature = float(suspicious_gate_temperature)

        self.save_context_diagnostics = bool(save_context_diagnostics)
        self.diagnostic_department_col = diagnostic_department_col

        self.context_target_probs: Dict[Tuple[int, int], Tensor] = {}
        self.context_target_ustar: Dict[Tuple[int, int], float] = {}
        self.context_target_argmax: Dict[Tuple[int, int], int] = {}
        self.context_target_modified: Dict[Tuple[int, int], bool] = {}
        self.context_target_gate: Dict[Tuple[int, int], float] = {}

    # ============================================================
    # Colonnes
    # ============================================================

    def _context_prob_cols(self) -> List[str]:
        return [f"{self.context_prefix}_p{c}" for c in range(self.num_classes)]

    def _context_required_cols(self) -> List[str]:
        return (
            self._context_prob_cols()
            + [
                f"{self.context_prefix}_argmax",
                f"{self.context_prefix}_u_star",
                f"{self.context_prefix}_u_norm",
                f"{self.context_prefix}_threshold",
                f"{self.context_prefix}_gate",
                f"{self.context_prefix}_modified",
                f"{self.context_prefix}_train_expected_class",
                f"{self.context_prefix}_train_argmax",
            ]
        )

    def _has_context_columns(self, df) -> bool:
        return all(col in df.columns for col in self._context_required_cols())

    @staticmethod
    def _safe_int_key_value(x: Any) -> int:
        if isinstance(x, Tensor):
            x = x.item()
        return int(x)

    def _make_key(self, graph_id: Any, date: Any) -> Tuple[int, int]:
        return (
            self._safe_int_key_value(graph_id),
            self._safe_int_key_value(date),
        )

    # ============================================================
    # Noyaux temporels
    # ============================================================

    def _kernel_size(self) -> int:
        return 2 * self.context_radius + 1

    def _make_temporal_kernel(self, mode: Optional[str] = None) -> np.ndarray:
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
            raise ValueError(f"Mode noyau non supporté : {mode}")

        if self.persistence and K > 1:
            kernel[: K // 2] = 0.0

        if np.all(kernel <= 0):
            kernel = np.ones(K, dtype=np.float64)

        kernel = kernel / max(kernel.sum(), self.eps)
        return kernel.astype(np.float64)

    # ============================================================
    # Encodage et distances
    # ============================================================

    def _one_hot_np(self, y: int) -> np.ndarray:
        y = int(y)
        y = max(0, min(self.num_classes - 1, y))
        out = np.zeros(self.num_classes, dtype=np.float64)
        out[y] = 1.0
        return out

    def _label_to_distribution_np(self, y: int) -> np.ndarray:
        y = int(y)
        y = max(0, min(self.num_classes - 1, y))

        if self.context_label_mode == "onehot":
            return self._one_hot_np(y)

        classes = np.arange(self.num_classes, dtype=np.float64)
        sigma = max(self.context_sigma_ordinal, self.eps)
        dist = np.exp(-((classes - float(y)) ** 2) / (2.0 * sigma ** 2))
        dist = dist / max(dist.sum(), self.eps)
        return dist.astype(np.float64)

    def _apply_prevent_downgrade_np(
        self,
        dist: np.ndarray,
        y_official: int,
    ) -> np.ndarray:
        dist = np.asarray(dist, dtype=np.float64).copy()
        y_official = int(y_official)

        if self.prevent_downgrade and y_official > 0:
            dist[:y_official] = 0.0

        s = dist.sum()
        if s <= self.eps:
            return self._one_hot_np(y_official)

        return dist / s

    def _ordinal_distance_np(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> float:
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        p = np.clip(p, 0.0, None)
        q = np.clip(q, 0.0, None)

        p = p / max(p.sum(), self.eps)
        q = q / max(q.sum(), self.eps)

        if self.suspicious_distance in {"cdf_l2", "cdf_l1", "cdf_linf"}:
            diff = np.cumsum(p) - np.cumsum(q)

            if self.suspicious_distance == "cdf_l2":
                return float(np.sqrt(np.mean(diff ** 2)))

            if self.suspicious_distance == "cdf_l1":
                return float(np.mean(np.abs(diff)))

            if self.suspicious_distance == "cdf_linf":
                return float(np.max(np.abs(diff)))

        if self.suspicious_distance == "mean_abs":
            classes = np.arange(self.num_classes, dtype=np.float64)
            mean_p = float((p * classes).sum())
            mean_q = float((q * classes).sum())
            denom = max(float(self.num_classes - 1), self.eps)
            return float(abs(mean_p - mean_q) / denom)

        raise ValueError(f"Distance ordinale inconnue : {self.suspicious_distance}")

    # ============================================================
    # Cibles contextuelles
    # ============================================================

    def _compute_context_distribution_for_group(
        self,
        dates: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:

        dates = np.asarray(dates).astype(int)
        labels = np.asarray(labels).astype(int)

        by_date = {int(d): int(y) for d, y in zip(dates, labels)}

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

        vectors = []
        weights = []

        for off, w in zip(offsets, kernel):
            tt = int(t + off)

            if tt not in by_date:
                continue

            vec = self._label_to_distribution_np(by_date[tt])
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
            r = V.max(axis=0)

        elif reducer == "median":
            r = np.median(V, axis=0)

        else:
            raise ValueError(f"reducer inconnu : {reducer}")

        r = np.clip(r, 0.0, None)
        if r.sum() <= self.eps:
            return self._one_hot_np(by_date[t])

        return r / r.sum()
    
    def _compute_department_normalized_uncertainty(
            self,
            df,
            cluster_col: str,
        ) -> np.ndarray:
            """
            Calcule u_norm = u_star / max(u_star dans le groupe).

            Par défaut, le groupe est le département si disponible.
            Sinon fallback sur graph_id.
            """

            u_col = f"{self.context_prefix}_u_star"

            if self.suspicious_normalize_group_col is not None:
                group_col = self.suspicious_normalize_group_col
            else:
                group_col = self.diagnostic_department_col

            if group_col == "cluster":
                group_col = cluster_col

            if group_col not in df.columns:
                group_col = self.graph_col

            u_norm = np.zeros(len(df), dtype=np.float64)

            for _, idxs in df.groupby(group_col).groups.items():
                idxs = list(idxs)
                vals = df.loc[idxs, u_col].astype(float).values
                max_val = float(np.nanmax(vals)) if len(vals) > 0 else 0.0

                if max_val <= self.eps:
                    u_norm[idxs] = 0.0
                else:
                    u_norm[idxs] = vals / max_val

            return np.clip(u_norm, 0.0, 1.0)

    # ============================================================
    # Détection par seuil / quantile
    # ============================================================

    def _compute_suspicious_thresholds(
        self,
        df,
        cluster_col: str,
    ) -> Dict[Any, float]:

        u_col = f"{self.context_prefix}_u_star"

        if self.suspicious_mode == "none":
            return {"__global__": -float("inf")}

        if self.suspicious_mode == "threshold":
            return {"__global__": float(self.suspicious_threshold)}

        if self.suspicious_mode == "department_normalized_threshold":
            return {"__global__": float(self.suspicious_threshold)}

        if self.suspicious_mode == "quantile":
            group_col = self.suspicious_group_col

            if group_col is None:
                group_col = self.graph_col

            if group_col == "cluster":
                group_col = cluster_col

            if group_col not in df.columns:
                raise ValueError(
                    f"suspicious_group_col={group_col} absent du dataframe."
                )

            thresholds = {}

            for group_value, dfg in df.groupby(group_col):
                vals = dfg[u_col].dropna().astype(float).values

                if vals.size == 0:
                    threshold = float(self.suspicious_threshold)
                else:
                    threshold = float(np.quantile(vals, self.suspicious_quantile))

                thresholds[self._safe_cluster_key(group_value)] = threshold

            return thresholds

        raise ValueError(f"suspicious_mode inconnu : {self.suspicious_mode}")

    def _get_suspicious_threshold_for_row(
        self,
        row,
        thresholds: Dict[Any, float],
        cluster_col: str,
    ) -> float:

        if "__global__" in thresholds:
            return float(thresholds["__global__"])

        group_col = self.suspicious_group_col

        if group_col is None:
            group_col = self.graph_col

        if group_col == "cluster":
            group_col = cluster_col

        key = self._safe_cluster_key(row[group_col])
        return float(thresholds.get(key, self.suspicious_threshold))

    def _compute_suspicious_gate_np(
        self,
        u_star: float,
        threshold: float,
    ) -> float:

        if self.suspicious_mode == "none":
            return 1.0

        if self.soft_suspicious_gate:
            temp = max(self.suspicious_gate_temperature, self.eps)
            x = (float(u_star) - float(threshold)) / temp
            x = np.clip(x, -60.0, 60.0)
            return float(1.0 / (1.0 + np.exp(-x)))

        return float(float(u_star) >= float(threshold))

    def _effective_training_distribution_np(
        self,
        y: int,
        r: np.ndarray,
        u_star: float,
        gate: float,
    ) -> np.ndarray:
        """
        Distribution effective utilisée pour diagnostics / coverage.

        Attention :
            Pendant le forward réel, l'intensité dépend de u_pred du modèle.
            Ici, pour visualiser le signal arrangé avant entraînement, on utilise
            u_star comme proxy de l'intensité de correction.
        """

        e = self._one_hot_np(y)

        if self.threshold_affects_training:
            correction_strength = float(gate) * float(u_star)
        else:
            correction_strength = float(u_star)

        correction_strength = float(np.clip(correction_strength, 0.0, 1.0))

        if correction_strength <= self.modified_eps:
            m = e
        else:
            if self.coverage_use_ustar_mix:
                m = (1.0 - correction_strength) * e + correction_strength * r
            else:
                m = (1.0 - float(gate)) * e + float(gate) * r

        m = np.clip(m, 0.0, None)
        return m / max(m.sum(), self.eps)

    # ============================================================
    # Dictionnaires
    # ============================================================

    def _clear_context_dicts(self) -> None:
        self.context_target_probs = {}
        self.context_target_ustar = {}
        self.context_target_argmax = {}
        self.context_target_modified = {}
        self.context_target_gate = {}

    def _register_context_sample(
        self,
        key: Tuple[int, int],
        probs: np.ndarray,
        u_star: float,
        argmax: int,
        modified: bool,
        gate: float,
    ) -> None:

        should_store = (
            self.store_all_context
            or bool(modified)
            or float(gate) > self.modified_eps
        )

        if not should_store:
            return

        self.context_target_probs[key] = torch.tensor(
            probs,
            dtype=torch.float32,
            device="cpu",
        )
        self.context_target_ustar[key] = float(u_star)
        self.context_target_argmax[key] = int(argmax)
        self.context_target_modified[key] = bool(modified)
        self.context_target_gate[key] = float(gate)

    def _rebuild_context_dicts_from_df(
        self,
        df,
        target_col: str,
    ) -> None:

        self._clear_context_dicts()

        prob_cols = self._context_prob_cols()
        arg_col = f"{self.context_prefix}_argmax"
        u_col = f"{self.context_prefix}_u_star"
        gate_col = f"{self.context_prefix}_gate"
        mod_col = f"{self.context_prefix}_modified"

        needed = [self.graph_col, self.date_col, target_col] + prob_cols + [
            arg_col,
            u_col,
            gate_col,
            mod_col,
        ]

        d = df[needed].dropna().copy()

        for _, row in d.iterrows():
            graph_id = row[self.graph_col]
            date = row[self.date_col]
            y = int(row[target_col])

            probs = np.array([float(row[col]) for col in prob_cols], dtype=np.float64)
            probs = probs / max(probs.sum(), self.eps)

            if self.prevent_downgrade:
                probs = self._apply_prevent_downgrade_np(probs, y)

            u_star = float(row[u_col])
            gate = float(row[gate_col])
            argmax = int(np.argmax(probs))
            modified = bool(row[mod_col])

            key = self._make_key(graph_id, date)

            self._register_context_sample(
                key=key,
                probs=probs,
                u_star=u_star,
                argmax=argmax,
                modified=modified,
                gate=gate,
            )

    # ============================================================
    # Coverage + pré-calcul
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
        Retourne le dataframe enrichi avec :
            context_p0 ... context_p{C-1}
            context_argmax
            context_u_star
            context_threshold
            context_gate
            context_modified
            context_train_expected_class
            context_train_argmax
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
        thr_col = f"{self.context_prefix}_threshold"
        gate_col = f"{self.context_prefix}_gate"
        mod_col = f"{self.context_prefix}_modified"
        train_exp_col = f"{self.context_prefix}_train_expected_class"
        train_arg_col = f"{self.context_prefix}_train_argmax"
        u_norm_col = f"{self.context_prefix}_u_norm"

        if not self._has_context_columns(df):
            for col in prob_cols:
                df[col] = np.nan

            df[arg_col] = -1
            df[u_col] = 0.0
            df[thr_col] = np.nan
            df[gate_col] = 0.0
            df[mod_col] = False
            df[train_exp_col] = np.nan
            df[train_arg_col] = -1
            df[u_norm_col] = 0.0

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
                    t = int(row[self.date_col])
                    y = int(row[target_col])

                    r = context_by_date[t]
                    e = self._one_hot_np(y)

                    u_star = self._ordinal_distance_np(e, r)
                    argmax = int(np.argmax(r))

                    for c, col in enumerate(prob_cols):
                        df.at[idx, col] = float(r[c])

                    df.at[idx, arg_col] = argmax
                    df.at[idx, u_col] = float(u_star)
                    
        if self.suspicious_mode == "department_normalized_threshold":
            df[u_norm_col] = self._compute_department_normalized_uncertainty(
                df=df,
                cluster_col=cluster_col,
            )

        # Seuils et gates recalculés dans tous les cas.
        thresholds = self._compute_suspicious_thresholds(
            df=df,
            cluster_col=cluster_col,
        )

        classes = np.arange(self.num_classes, dtype=np.float64)

        for idx, row in df.iterrows():
            if np.isnan(row[u_col]):
                continue

            y = int(row[target_col])
            r = np.array([float(row[col]) for col in prob_cols], dtype=np.float64)
            r = r / max(r.sum(), self.eps)

            if self.prevent_downgrade:
                r = self._apply_prevent_downgrade_np(r, y)

            u_star = float(row[u_col])

            threshold = self._get_suspicious_threshold_for_row(
                row=row,
                thresholds=thresholds,
                cluster_col=cluster_col,
            )

            if self.suspicious_mode == "department_normalized_threshold":
                score_for_gate = float(row[u_norm_col])
            else:
                score_for_gate = u_star

            gate = self._compute_suspicious_gate_np(
                u_star=score_for_gate,
                threshold=threshold,
            )

            if self.soft_suspicious_gate:
                modified = bool(gate > 0.5)
            else:
                modified = bool(gate >= 1.0)

            m = self._effective_training_distribution_np(
                y=y,
                r=r,
                u_star=u_star,
                gate=gate,
            )

            for c, col in enumerate(prob_cols):
                df.at[idx, col] = float(r[c])

            df.at[idx, u_norm_col] = float(row[u_norm_col]) if self.suspicious_mode == "department_normalized_threshold" else float(u_star)
            df.at[idx, thr_col] = float(threshold)
            df.at[idx, gate_col] = float(gate)
            df.at[idx, mod_col] = modified
            df.at[idx, train_exp_col] = float((m * classes).sum())
            df.at[idx, train_arg_col] = int(np.argmax(m))

        self._rebuild_context_dicts_from_df(
            df=df,
            target_col=target_col,
        )

        # --------------------------------------------------------
        # Coverage soft
        # --------------------------------------------------------
        d_cov = df[
            [cluster_col, target_col] + prob_cols + [u_col, gate_col, mod_col]
        ].dropna().copy()

        d_cov[target_col] = d_cov[target_col].astype(int)

        effective_targets = []

        for _, row in d_cov.iterrows():
            y = int(row[target_col])
            r = np.array([float(row[col]) for col in prob_cols], dtype=np.float64)
            r = r / max(r.sum(), self.eps)

            u_star = float(row[u_col])
            gate = float(row[gate_col])

            m = self._effective_training_distribution_np(
                y=y,
                r=r,
                u_star=u_star,
                gate=gate,
            )

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

        if dir_output is not None:
            os.makedirs(dir_output, exist_ok=True)

            export_cols = [
                self.graph_col,
                self.date_col,
                target_col,
            ] + prob_cols + [
                arg_col,
                u_col,
                thr_col,
                gate_col,
                mod_col,
                train_exp_col,
                train_arg_col,
            ]

            if cluster_col in df.columns:
                export_cols.append(cluster_col)

            if self.diagnostic_department_col in df.columns:
                export_cols.append(self.diagnostic_department_col)

            df[export_cols].to_csv(
                os.path.join(dir_output, "contextual_targets.csv"),
                index=False,
            )

            with open(
                os.path.join(dir_output, "class_coverage_distribution_contextual.csv"),
                "w",
            ) as f:
                f.write("cluster,class,probability\n")
                for k, v in self.coverage_by_cluster.items():
                    v_np = v.detach().cpu().numpy()
                    for c, p_val in enumerate(v_np):
                        f.write(f"{k},{c},{p_val:.8f}\n")

            with open(
                os.path.join(dir_output, "class_coverage_global_distribution_contextual.csv"),
                "w",
            ) as f:
                f.write("class,probability\n")
                for c, p_val in enumerate(global_dist_np):
                    f.write(f"{c},{p_val:.8f}\n")

        if dir_output is not None and self.save_context_diagnostics:
            self._plot_contextual_diagnostics(
                df=df,
                target_col=target_col,
                cluster_col=cluster_col,
                dir_output=dir_output,
            )

        return df

    # ============================================================
    # Batch contextuel
    # ============================================================

    def _get_context_targets_batch(
        self,
        y: Tensor,
        graph_ids: Tensor,
        dates: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

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
        gate = torch.zeros(B, device=device, dtype=dtype)
        modified = torch.zeros(B, device=device, dtype=torch.bool)

        for i in range(B):
            key = self._make_key(graph_ids[i], dates[i])

            if key in self.context_target_probs:
                probs_i = self.context_target_probs[key].to(
                    device=device,
                    dtype=dtype,
                )
                probs_i = probs_i.clamp_min(0.0)
                probs_i = probs_i / probs_i.sum().clamp_min(self.eps)

                context_probs[i] = probs_i
                u_star[i] = torch.tensor(
                    float(self.context_target_ustar.get(key, 0.0)),
                    device=device,
                    dtype=dtype,
                )
                gate[i] = torch.tensor(
                    float(self.context_target_gate.get(key, 0.0)),
                    device=device,
                    dtype=dtype,
                )
                modified[i] = bool(self.context_target_modified.get(key, False))

            else:
                context_probs[i] = F.one_hot(
                    y[i].clamp(0, self.num_classes - 1),
                    num_classes=self.num_classes,
                ).to(device=device, dtype=dtype)

        return context_probs, u_star, gate, modified

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

        context_probs, u_star, suspicious_gate, modified = self._get_context_targets_batch(
            y=y,
            graph_ids=graph_ids,
            dates=dates,
            device=device,
            dtype=dtype,
        )

        u_pred = uncertainty.view(-1, 1)

        if self.threshold_affects_training:
            effective_uncertainty = u_pred * suspicious_gate.view(-1, 1)
        else:
            effective_uncertainty = u_pred

        soft_target = (
            (1.0 - effective_uncertainty) * one_hot
            + effective_uncertainty * context_probs
        )

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
            similarity_loss = torch.zeros((), device=device, dtype=dtype)

        if self.threshold_affects_training:
            uncertainty_target = (suspicious_gate * u_star).detach()
        else:
            uncertainty_target = u_star.detach()

        uncertainty_loss_vec = (uncertainty - uncertainty_target).pow(2)

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
            "mean_suspicious_gate": suspicious_gate.detach().mean(),
            "mean_effective_uncertainty": effective_uncertainty.detach().mean(),
            "modified_ratio": modified.float().mean().detach(),
            "C": C_value.detach(),
            "id": torch.tensor(
                self.id,
                device=device,
                dtype=torch.long,
            ),
        }

    # ============================================================
    # Diagnostics plots
    # ============================================================

    def _plot_contextual_diagnostics(
        self,
        df,
        target_col: str,
        cluster_col: str,
        dir_output,
    ) -> None:

        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        base_dir = os.path.join(dir_output, "contextual_diagnostics")
        os.makedirs(base_dir, exist_ok=True)

        prob_cols = self._context_prob_cols()
        arg_col = f"{self.context_prefix}_argmax"
        u_col = f"{self.context_prefix}_u_star"
        u_norm_col = f"{self.context_prefix}_u_norm"
        thr_col = f"{self.context_prefix}_threshold"
        gate_col = f"{self.context_prefix}_gate"
        mod_col = f"{self.context_prefix}_modified"
        train_exp_col = f"{self.context_prefix}_train_expected_class"
        train_arg_col = f"{self.context_prefix}_train_argmax"

        required_cols = [
            target_col,
            arg_col,
            u_col,
            thr_col,
            gate_col,
            mod_col,
            train_exp_col,
            train_arg_col,
            self.graph_col,
            self.date_col,
            cluster_col,
        ] + prob_cols

        # Compatibilité si u_norm n'existe pas encore.
        if u_norm_col not in df.columns:
            df = df.copy()
            df[u_norm_col] = df[u_col].astype(float)

        required_cols.append(u_norm_col)

        missing = [c for c in required_cols if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Colonnes manquantes pour diagnostics : {missing}")

        extra_cols = []
        if self.diagnostic_department_col in df.columns:
            extra_cols.append(self.diagnostic_department_col)

        d = df[required_cols + extra_cols].copy()

        d[target_col] = d[target_col].astype(int)
        d[arg_col] = d[arg_col].astype(int)
        d[u_col] = d[u_col].astype(float)
        d[u_norm_col] = d[u_norm_col].astype(float)
        d[thr_col] = d[thr_col].astype(float)
        d[gate_col] = d[gate_col].astype(float)
        d[mod_col] = d[mod_col].astype(bool)
        d[train_exp_col] = d[train_exp_col].astype(float)
        d[train_arg_col] = d[train_arg_col].astype(int)
        d[self.graph_col] = d[self.graph_col].astype(int)
        d[self.date_col] = d[self.date_col].astype(int)
        
        df = df.loc[:,~df.columns.duplicated()].copy()

        d["_transition_context"] = (
            d[target_col].astype(str)
            + " -> "
            + d[arg_col].astype(str)
        )

        d["_transition_train"] = (
            d[target_col].astype(str)
            + " -> "
            + d[train_arg_col].astype(str)
        )

        d["_delta_context"] = d[arg_col] - d[target_col]
        d["_delta_train"] = d[train_arg_col] - d[target_col]
        d["_u_star_modified_only"] = np.where(d[mod_col], d[u_col], 0.0)

        d["_effective_modification"] = (
            d[mod_col].astype(bool)
            & (d[target_col].astype(int) != d[train_arg_col].astype(int))
        )

        d_effective = d[d["_effective_modification"]].copy()
        d_modified = d[d[mod_col]].copy()

        # --------------------------------------------------------
        # 01 true vs context argmax
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
        # 02 transition matrix all samples: official -> context argmax
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "02_transition_matrix_context_all")
        os.makedirs(out_dir, exist_ok=True)

        transition_context = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        for y_true, y_ctx in zip(d[target_col].values, d[arg_col].values):
            if 0 <= y_true < self.num_classes and 0 <= y_ctx < self.num_classes:
                transition_context[int(y_true), int(y_ctx)] += 1.0

        transition_context_norm = transition_context / np.maximum(
            transition_context.sum(axis=1, keepdims=True),
            self.eps,
        )

        self._save_heatmap(
            matrix=transition_context,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Contextual argmax class",
            ylabel="Official class",
            title="Transition matrix: official class -> contextual argmax",
            path=os.path.join(out_dir, "transition_matrix_context_counts.png"),
            fmt=".0f",
        )

        self._save_heatmap(
            matrix=transition_context_norm,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Contextual argmax class",
            ylabel="Official class",
            title="Transition matrix: official class -> contextual argmax — row-normalized",
            path=os.path.join(out_dir, "transition_matrix_context_row_normalized.png"),
            fmt=".2f",
        )

        pd.DataFrame(
            transition_context.astype(int),
            index=[f"official_{c}" for c in range(self.num_classes)],
            columns=[f"context_{c}" for c in range(self.num_classes)],
        ).to_csv(os.path.join(out_dir, "transition_matrix_context_counts.csv"))

        pd.DataFrame(
            transition_context_norm,
            index=[f"official_{c}" for c in range(self.num_classes)],
            columns=[f"context_{c}" for c in range(self.num_classes)],
        ).to_csv(os.path.join(out_dir, "transition_matrix_context_row_normalized.csv"))

        # --------------------------------------------------------
        # 02b transition matrix suspicious samples only
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "02b_transition_matrix_suspicious_samples")
        os.makedirs(out_dir, exist_ok=True)

        transition_modified = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        for y_true, y_mod in zip(
            d_modified[target_col].values,
            d_modified[train_arg_col].values,
        ):
            if 0 <= y_true < self.num_classes and 0 <= y_mod < self.num_classes:
                transition_modified[int(y_true), int(y_mod)] += 1.0

        transition_modified_norm = transition_modified / np.maximum(
            transition_modified.sum(axis=1, keepdims=True),
            self.eps,
        )

        self._save_heatmap(
            matrix=transition_modified,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Training argmax class",
            ylabel="Official class",
            title="Suspicious samples: official class -> training argmax class",
            path=os.path.join(out_dir, "transition_matrix_suspicious_samples_counts.png"),
            fmt=".0f",
        )

        self._save_heatmap(
            matrix=transition_modified_norm,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Training argmax class",
            ylabel="Official class",
            title="Suspicious samples: official class -> training argmax class — row-normalized",
            path=os.path.join(out_dir, "transition_matrix_suspicious_samples_row_normalized.png"),
            fmt=".2f",
        )

        pd.DataFrame(
            transition_modified.astype(int),
            index=[f"official_{c}" for c in range(self.num_classes)],
            columns=[f"training_{c}" for c in range(self.num_classes)],
        ).to_csv(os.path.join(out_dir, "transition_matrix_suspicious_samples_counts.csv"))

        pd.DataFrame(
            transition_modified_norm,
            index=[f"official_{c}" for c in range(self.num_classes)],
            columns=[f"training_{c}" for c in range(self.num_classes)],
        ).to_csv(os.path.join(out_dir, "transition_matrix_suspicious_samples_row_normalized.csv"))

        # --------------------------------------------------------
        # 02c transition matrix effective modifications only
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "02c_transition_matrix_effective_modifications")
        os.makedirs(out_dir, exist_ok=True)

        transition_effective = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        for y_true, y_mod in zip(
            d_effective[target_col].astype(int).values,
            d_effective[train_arg_col].astype(int).values,
        ):
            if 0 <= y_true < self.num_classes and 0 <= y_mod < self.num_classes:
                transition_effective[int(y_true), int(y_mod)] += 1.0

        transition_effective_norm = transition_effective / np.maximum(
            transition_effective.sum(axis=1, keepdims=True),
            self.eps,
        )

        self._save_heatmap(
            matrix=transition_effective,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Modified class used for training",
            ylabel="Official class",
            title="Effective modifications only: official class -> modified class",
            path=os.path.join(out_dir, "transition_matrix_effective_modifications_counts.png"),
            fmt=".0f",
        )

        self._save_heatmap(
            matrix=transition_effective_norm,
            x_labels=[str(c) for c in range(self.num_classes)],
            y_labels=[str(c) for c in range(self.num_classes)],
            xlabel="Modified class used for training",
            ylabel="Official class",
            title="Effective modifications only: official class -> modified class — row-normalized",
            path=os.path.join(out_dir, "transition_matrix_effective_modifications_row_normalized.png"),
            fmt=".2f",
        )

        pd.DataFrame(
            transition_effective.astype(int),
            index=[f"official_{c}" for c in range(self.num_classes)],
            columns=[f"modified_{c}" for c in range(self.num_classes)],
        ).to_csv(os.path.join(out_dir, "transition_matrix_effective_modifications_counts.csv"))

        pd.DataFrame(
            transition_effective_norm,
            index=[f"official_{c}" for c in range(self.num_classes)],
            columns=[f"modified_{c}" for c in range(self.num_classes)],
        ).to_csv(os.path.join(out_dir, "transition_matrix_effective_modifications_row_normalized.csv"))

        # --------------------------------------------------------
        # 02d pie chart effective modifications only
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "02d_effective_modification_pie")
        os.makedirs(out_dir, exist_ok=True)

        if len(d_effective) > 0:
            transition_labels = (
                d_effective[target_col].astype(int).astype(str)
                + " -> "
                + d_effective[train_arg_col].astype(int).astype(str)
            )

            transition_counts = transition_labels.value_counts().sort_values(ascending=False)

            values = transition_counts.values.astype(float)
            labels = transition_counts.index.tolist()
            total = int(values.sum())

            plt.figure(figsize=(10, 10))
            wedges, _ = plt.pie(
                values,
                startangle=90,
                counterclock=False,
            )

            legend_labels = [
                f"{lab}  (n={int(val)}, {100.0 * val / total:.1f}%)"
                for lab, val in zip(labels, values)
            ]

            plt.legend(
                wedges,
                legend_labels,
                title="Effective transitions",
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
            )

            plt.title("Distribution of effective modification transitions")
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, "effective_modification_transitions_pie.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

            pd.DataFrame(
                {
                    "transition": labels,
                    "count": values.astype(int),
                    "percentage": 100.0 * values / total,
                }
            ).to_csv(
                os.path.join(out_dir, "effective_modification_transitions_counts.csv"),
                index=False,
            )

        else:
            self._save_empty_plot(
                title="No effective modification transition",
                path=os.path.join(out_dir, "effective_modification_transitions_pie.png"),
            )

            pd.DataFrame(
                {
                    "transition": [],
                    "count": [],
                    "percentage": [],
                }
            ).to_csv(
                os.path.join(out_dir, "effective_modification_transitions_counts.csv"),
                index=False,
            )

        # --------------------------------------------------------
        # 03 modifications by department / graph_id
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "03_modified_by_department")
        os.makedirs(out_dir, exist_ok=True)

        if self.diagnostic_department_col in d.columns:
            dep_col = self.diagnostic_department_col
            
            d = d.loc[:,~d.columns.duplicated()].copy()

            by_dep = (
                d.groupby(dep_col)
                .agg(
                    n_samples=(mod_col, "size"),
                    n_suspicious=(mod_col, "sum"),
                    n_effective_modifications=("_effective_modification", "sum"),
                    mean_u_star=(u_col, "mean"),
                    mean_u_norm=(u_norm_col, "mean"),
                    mean_gate=(gate_col, "mean"),
                    n_fire=(target_col, lambda x: int((x > 0).sum())),
                )
                .reset_index()
            )

            by_dep["suspicious_ratio"] = by_dep["n_suspicious"] / by_dep["n_samples"].clip(lower=1)
            by_dep["effective_modified_ratio"] = (
                by_dep["n_effective_modifications"] / by_dep["n_samples"].clip(lower=1)
            )

            by_dep = by_dep.sort_values("n_suspicious", ascending=False)
            by_dep.to_csv(os.path.join(out_dir, "modified_by_department.csv"), index=False)

            self._save_barplot_from_dataframe(
                data=by_dep,
                x_col=dep_col,
                y_col="n_suspicious",
                title="Number of suspicious samples by department",
                ylabel="Number of suspicious samples",
                path=os.path.join(out_dir, "suspicious_count_by_department.png"),
                top_k=40,
            )

            self._save_barplot_from_dataframe(
                data=by_dep.sort_values("suspicious_ratio", ascending=False),
                x_col=dep_col,
                y_col="suspicious_ratio",
                title="Ratio of suspicious samples by department",
                ylabel="Suspicious ratio",
                path=os.path.join(out_dir, "suspicious_ratio_by_department.png"),
                top_k=40,
            )

            if by_dep["n_effective_modifications"].sum() > 0:
                self._save_barplot_from_dataframe(
                    data=by_dep.sort_values("n_effective_modifications", ascending=False),
                    x_col=dep_col,
                    y_col="n_effective_modifications",
                    title="Number of effective modifications by department",
                    ylabel="Number of effective modifications",
                    path=os.path.join(out_dir, "effective_modification_count_by_department.png"),
                    top_k=40,
                )

                self._save_barplot_from_dataframe(
                    data=by_dep.sort_values("effective_modified_ratio", ascending=False),
                    x_col=dep_col,
                    y_col="effective_modified_ratio",
                    title="Ratio of effective modifications by department",
                    ylabel="Effective modification ratio",
                    path=os.path.join(out_dir, "effective_modification_ratio_by_department.png"),
                    top_k=40,
                )
            else:
                self._save_empty_plot(
                    title="No effective modification by department",
                    path=os.path.join(out_dir, "effective_modification_count_by_department.png"),
                )
                self._save_empty_plot(
                    title="No effective modification ratio by department",
                    path=os.path.join(out_dir, "effective_modification_ratio_by_department.png"),
                )

            self._save_barplot_from_dataframe(
                data=by_dep.sort_values("mean_u_norm", ascending=False),
                x_col=dep_col,
                y_col="mean_u_norm",
                title="Mean normalized uncertainty by department",
                ylabel="Mean normalized uncertainty",
                path=os.path.join(out_dir, "mean_u_norm_by_department.png"),
                top_k=40,
            )

        else:
            by_graph = (
                d.groupby(self.graph_col)
                .agg(
                    n_samples=(mod_col, "size"),
                    n_suspicious=(mod_col, "sum"),
                    n_effective_modifications=("_effective_modification", "sum"),
                    mean_u_star=(u_col, "mean"),
                    mean_u_norm=(u_norm_col, "mean"),
                    mean_gate=(gate_col, "mean"),
                    n_fire=(target_col, lambda x: int((x > 0).sum())),
                )
                .reset_index()
            )

            by_graph["suspicious_ratio"] = by_graph["n_suspicious"] / by_graph["n_samples"].clip(lower=1)
            by_graph["effective_modified_ratio"] = (
                by_graph["n_effective_modifications"] / by_graph["n_samples"].clip(lower=1)
            )

            by_graph = by_graph.sort_values("n_suspicious", ascending=False)
            by_graph.to_csv(os.path.join(out_dir, "modified_by_graph_id.csv"), index=False)

            self._save_barplot_from_dataframe(
                data=by_graph,
                x_col=self.graph_col,
                y_col="n_suspicious",
                title="Number of suspicious samples by graph_id",
                ylabel="Number of suspicious samples",
                path=os.path.join(out_dir, "suspicious_count_by_graph_id.png"),
                top_k=40,
            )

            if by_graph["n_effective_modifications"].sum() > 0:
                self._save_barplot_from_dataframe(
                    data=by_graph.sort_values("n_effective_modifications", ascending=False),
                    x_col=self.graph_col,
                    y_col="n_effective_modifications",
                    title="Number of effective modifications by graph_id",
                    ylabel="Number of effective modifications",
                    path=os.path.join(out_dir, "effective_modification_count_by_graph_id.png"),
                    top_k=40,
                )
            else:
                self._save_empty_plot(
                    title="No effective modification by graph_id",
                    path=os.path.join(out_dir, "effective_modification_count_by_graph_id.png"),
                )

        # --------------------------------------------------------
        # 04 uncertainty histograms
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

        plt.figure(figsize=(8, 5))
        plt.hist(d[u_norm_col].values, bins=50)
        plt.xlabel("u_norm")
        plt.ylabel("Number of samples")
        plt.title("Distribution of department-normalized uncertainty")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "u_norm_hist_all_samples.png"), dpi=200)
        plt.close()

        if len(d_modified) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(d_modified[u_col].values, bins=50)
            plt.xlabel("u_star")
            plt.ylabel("Number of suspicious samples")
            plt.title("Distribution of u_star — suspicious samples only")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ustar_hist_suspicious_samples.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.hist(d_modified[u_norm_col].values, bins=50)
            plt.xlabel("u_norm")
            plt.ylabel("Number of suspicious samples")
            plt.title("Distribution of normalized uncertainty — suspicious samples only")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "u_norm_hist_suspicious_samples.png"), dpi=200)
            plt.close()

        if len(d_effective) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(d_effective[u_col].values, bins=50)
            plt.xlabel("u_star")
            plt.ylabel("Number of effective modifications")
            plt.title("Distribution of u_star — effective modifications only")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ustar_hist_effective_modifications.png"), dpi=200)
            plt.close()

        plt.figure(figsize=(9, 5))
        for c in range(self.num_classes):
            vals = d.loc[d[target_col] == c, u_norm_col].values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=40, alpha=0.45, label=f"class {c}")
        plt.xlabel("u_norm")
        plt.ylabel("Number of samples")
        plt.title("Normalized uncertainty distribution by official class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "u_norm_hist_by_true_class.png"), dpi=200)
        plt.close()

        # --------------------------------------------------------
        # 05 modifications by true class
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "05_modified_by_true_class")
        os.makedirs(out_dir, exist_ok=True)

        by_class = (
            d.groupby(target_col)
            .agg(
                n_samples=(mod_col, "size"),
                n_suspicious=(mod_col, "sum"),
                n_effective_modifications=("_effective_modification", "sum"),
                mean_u_star=(u_col, "mean"),
                mean_u_norm=(u_norm_col, "mean"),
                mean_gate=(gate_col, "mean"),
                mean_delta_context=("_delta_context", "mean"),
                mean_delta_train=("_delta_train", "mean"),
            )
            .reindex(range(self.num_classes), fill_value=0)
            .reset_index()
        )

        by_class["suspicious_ratio"] = by_class["n_suspicious"] / by_class["n_samples"].clip(lower=1)
        by_class["effective_modified_ratio"] = (
            by_class["n_effective_modifications"] / by_class["n_samples"].clip(lower=1)
        )

        by_class.to_csv(os.path.join(out_dir, "modified_by_true_class.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.bar(by_class[target_col].astype(int).values, by_class["n_suspicious"].values)
        plt.xlabel("Official class")
        plt.ylabel("Number of suspicious samples")
        plt.title("Number of suspicious samples by official class")
        plt.xticks(range(self.num_classes))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "suspicious_count_by_true_class.png"), dpi=200)
        plt.close()

        if by_class["n_effective_modifications"].sum() > 0:
            plt.figure(figsize=(8, 5))
            plt.bar(
                by_class[target_col].astype(int).values,
                by_class["n_effective_modifications"].values,
            )
            plt.xlabel("Official class")
            plt.ylabel("Number of effective modifications")
            plt.title("Number of effective modifications by official class")
            plt.xticks(range(self.num_classes))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "effective_modification_count_by_true_class.png"), dpi=200)
            plt.close()
        else:
            self._save_empty_plot(
                title="No effective modification by true class",
                path=os.path.join(out_dir, "effective_modification_count_by_true_class.png"),
            )

        # --------------------------------------------------------
        # 06 modified by date
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "06_modified_by_date")
        os.makedirs(out_dir, exist_ok=True)

        by_date = (
            d.groupby(self.date_col)
            .agg(
                n_samples=(mod_col, "size"),
                n_suspicious=(mod_col, "sum"),
                n_effective_modifications=("_effective_modification", "sum"),
                mean_u_star=(u_col, "mean"),
                mean_u_norm=(u_norm_col, "mean"),
                mean_gate=(gate_col, "mean"),
            )
            .reset_index()
            .sort_values(self.date_col)
        )

        by_date["suspicious_ratio"] = by_date["n_suspicious"] / by_date["n_samples"].clip(lower=1)
        by_date["effective_modified_ratio"] = (
            by_date["n_effective_modifications"] / by_date["n_samples"].clip(lower=1)
        )

        by_date.to_csv(os.path.join(out_dir, "modified_by_date.csv"), index=False)

        plt.figure(figsize=(14, 5))
        plt.plot(by_date[self.date_col].values, by_date["n_suspicious"].values)
        plt.xlabel("Date index")
        plt.ylabel("Number of suspicious samples")
        plt.title("Number of suspicious samples by date")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "suspicious_count_by_date.png"), dpi=200)
        plt.close()

        if by_date["n_effective_modifications"].sum() > 0:
            plt.figure(figsize=(14, 5))
            plt.plot(by_date[self.date_col].values, by_date["n_effective_modifications"].values)
            plt.xlabel("Date index")
            plt.ylabel("Number of effective modifications")
            plt.title("Number of effective modifications by date")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "effective_modification_count_by_date.png"), dpi=200)
            plt.close()
        else:
            self._save_empty_plot(
                title="No effective modification by date",
                path=os.path.join(out_dir, "effective_modification_count_by_date.png"),
            )

        # --------------------------------------------------------
        # 07 top suspicious samples
        # --------------------------------------------------------
        out_dir = os.path.join(base_dir, "07_top_suspicious_samples")
        os.makedirs(out_dir, exist_ok=True)

        export_cols = [
            self.graph_col,
            self.date_col,
            target_col,
            arg_col,
            u_col,
            u_norm_col,
            thr_col,
            gate_col,
            mod_col,
            train_exp_col,
            train_arg_col,
            "_delta_context",
            "_delta_train",
            "_effective_modification",
            "_transition_context",
            "_transition_train",
            cluster_col,
        ] + prob_cols

        if self.diagnostic_department_col in d.columns:
            export_cols.insert(2, self.diagnostic_department_col)

        d.sort_values(u_norm_col, ascending=False).head(5000)[export_cols].to_csv(
            os.path.join(out_dir, "top_suspicious_samples_by_u_norm.csv"),
            index=False,
        )

        d.sort_values(u_col, ascending=False).head(5000)[export_cols].to_csv(
            os.path.join(out_dir, "top_suspicious_samples_by_u_star.csv"),
            index=False,
        )

        d_effective.sort_values(u_norm_col, ascending=False).head(5000)[export_cols].to_csv(
            os.path.join(out_dir, "top_effective_modifications.csv"),
            index=False,
        )

        # --------------------------------------------------------
        # 08 time series top fire departments
        # --------------------------------------------------------
        if self.diagnostic_department_col in d.columns:
            self._plot_top_fire_departments_timeseries(
                d=d,
                target_col=target_col,
                dir_output=base_dir,
                u_col=u_col,
                mod_col=mod_col,
                train_exp_col=train_exp_col,
                train_arg_col=train_arg_col,
            )
            
    def _plot_top_fire_departments_timeseries(
        self,
        d,
        target_col: str,
        dir_output: str,
        u_col: str,
        mod_col: str,
        train_exp_col: str,
        train_arg_col: str,
    ) -> None:
        """
        Pour les 3 départements avec le plus de feux :
            1. plot de u_star uniquement pour les samples modifiés ;
            2. plot du signal arrangé utilisé pour l'entraînement, sans
            Arranged expected class ;
            3. plot combiné avec les samples modifiés en vert.

        Remarque :
            train_exp_col est conservé dans la signature pour compatibilité,
            mais il n'est plus affiché.
        """

        import matplotlib.pyplot as plt
        import pandas as pd

        dep_col = self.diagnostic_department_col

        out_dir = os.path.join(
            dir_output,
            "08_top_fire_departments_timeseries",
        )
        os.makedirs(out_dir, exist_ok=True)

        fire_counts = (
            d.assign(_is_fire=(d[target_col].astype(int) > 0).astype(int))
            .groupby(dep_col)["_is_fire"]
            .sum()
            .sort_values(ascending=False)
        )

        top_departments = list(fire_counts.head(3).index)

        summary_rows = []

        for dep in top_departments:
            dep_dir_name = str(dep).replace("/", "_")
            dep_dir = os.path.join(out_dir, f"department_{dep_dir_name}")
            os.makedirs(dep_dir, exist_ok=True)

            dd = d[d[dep_col] == dep].copy()
            dd = dd.sort_values([self.graph_col, self.date_col])

            graph_ids = sorted(dd[self.graph_col].dropna().unique())

            for graph_id in graph_ids:
                dg = dd[dd[self.graph_col] == graph_id].copy()
                dg = dg.sort_values(self.date_col)

                if len(dg) == 0:
                    continue

                dates = dg[self.date_col].astype(int).values
                official = dg[target_col].astype(int).values

                modified_mask = dg[mod_col].astype(bool).values

                # u_star uniquement pour les samples modifiés.
                dates_modified = dates[modified_mask]
                u_modified = dg.loc[modified_mask, u_col].astype(float).values

                arranged_argmax = dg[train_arg_col].astype(int).values
                arranged_argmax_modified = arranged_argmax[modified_mask]

                n_fire = int((official > 0).sum())
                n_modified = int(modified_mask.sum())

                summary_rows.append(
                    {
                        dep_col: dep,
                        self.graph_col: graph_id,
                        "n_samples": len(dg),
                        "n_fire": n_fire,
                        "n_modified": n_modified,
                        "mean_u_star_modified_only": float(np.mean(u_modified)) if len(u_modified) > 0 else 0.0,
                        "max_u_star_modified_only": float(np.max(u_modified)) if len(u_modified) > 0 else 0.0,
                    }
                )

                safe_graph = str(graph_id).replace("/", "_")

                # ----------------------------------------------------
                # 1) u_star uniquement pour les samples modifiés
                # ----------------------------------------------------
                plt.figure(figsize=(15, 4))

                if len(dates_modified) > 0:
                    plt.vlines(
                        dates_modified,
                        ymin=0.0,
                        ymax=u_modified,
                        linewidth=1.0,
                        color="green",
                    )
                    plt.scatter(
                        dates_modified,
                        u_modified,
                        s=12,
                        color="green",
                        label="Modified samples",
                        zorder=3,
                    )

                plt.xlabel("Date index")
                plt.ylabel("u_star")
                plt.title(
                    f"Contextual uncertainty of modified samples — "
                    f"department={dep}, graph_id={graph_id}"
                )
                plt.ylim(bottom=0.0)
                if len(dates_modified) > 0:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(dep_dir, f"graph_{safe_graph}_ustar_modified_only.png"),
                    dpi=200,
                )
                plt.close()

                # ----------------------------------------------------
                # 2) Signal arrangé utilisé pour l'entraînement
                #    sans Arranged expected class
                # ----------------------------------------------------
                plt.figure(figsize=(15, 5))

                fire_mask = official > 0

                # Signal officiel discret.
                plt.scatter(
                    dates,
                    official,
                    s=10,
                    label="Official class",
                    zorder=3,
                )

                plt.vlines(
                    dates[fire_mask],
                    ymin=0,
                    ymax=official[fire_mask],
                    linewidth=0.8,
                    alpha=0.45,
                    zorder=2,
                )

                # Classe arrangée uniquement pour les samples modifiés.
                if len(dates_modified) > 0:
                    plt.scatter(
                        dates_modified,
                        arranged_argmax_modified,
                        s=18,
                        marker="x",
                        color="green",
                        label="Arranged argmax class — modified only",
                        zorder=5,
                    )

                plt.xlabel("Date index")
                plt.ylabel("Class")
                plt.title(
                    f"Arranged training signal — modified samples only — "
                    f"department={dep}, graph_id={graph_id}"
                )
                plt.yticks(range(self.num_classes))
                plt.ylim(-0.15, self.num_classes - 1 + 0.25)
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(dep_dir, f"graph_{safe_graph}_arranged_training_signal.png"),
                    dpi=200,
                )
                plt.close()

                # ----------------------------------------------------
                # 3) Plot combiné lisible
                # ----------------------------------------------------
                fig, ax1 = plt.subplots(figsize=(15, 5))

                ax1.scatter(
                    dates,
                    official,
                    s=10,
                    label="Official class",
                    zorder=4,
                )

                ax1.vlines(
                    dates[fire_mask],
                    ymin=0,
                    ymax=official[fire_mask],
                    linewidth=0.8,
                    alpha=0.45,
                    zorder=2,
                )

                if len(dates_modified) > 0:
                    ax1.scatter(
                        dates_modified,
                        arranged_argmax_modified,
                        s=18,
                        marker="x",
                        color="green",
                        label="Arranged argmax class — modified only",
                        zorder=6,
                    )

                ax1.set_xlabel("Date index")
                ax1.set_ylabel("Class")
                ax1.set_yticks(range(self.num_classes))
                ax1.set_ylim(-0.15, self.num_classes - 1 + 0.25)

                # Axe droit : incertitude uniquement pour samples modifiés.
                ax2 = ax1.twinx()

                if len(dates_modified) > 0:
                    ax2.bar(
                        dates_modified,
                        u_modified,
                        width=1.0,
                        alpha=0.25,
                        color="green",
                        label="u_star — modified only",
                        zorder=1,
                    )

                ax2.set_ylabel("Contextual uncertainty")
                if len(u_modified) > 0:
                    ax2.set_ylim(0.0, max(1.0, float(np.max(u_modified)) + 0.05))
                else:
                    ax2.set_ylim(0.0, 1.0)

                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()

                if len(lines_2) > 0:
                    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
                else:
                    ax1.legend(lines_1, labels_1, loc="upper right")

                plt.title(
                    f"Official vs arranged signal with modified samples — "
                    f"department={dep}, graph_id={graph_id}"
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(dep_dir, f"graph_{safe_graph}_combined_signal_uncertainty.png"),
                    dpi=200,
                )
                plt.close()

        if len(summary_rows) > 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, "top_fire_departments_graph_summary.csv"),
                index=False,
            )

    # ============================================================
    # Helpers plots
    # ============================================================

    def _save_empty_plot(
        self,
        title: str,
        path: str,
    ) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()


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
                ax.text(
                    j,
                    i,
                    format(matrix[i, j], fmt),
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