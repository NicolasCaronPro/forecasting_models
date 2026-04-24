from typing import Optional, Union, List, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from forecasting_models.pytorch.loss_utils import *


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
        reliability = 1 - uncertainty

    Plus uncertainty est grande, plus le label dur y est diffusé
    autour de sa classe observée.

    Loss totale :
        L = L_soft_focal
            + C * L_WK
            + lambda_coverage * L_coverage
            + lambda_similarity * L_similarity
            + lambda_uncertainty * L_uncertainty

    Important :
        WKLoss doit déjà exister dans ton code.
    """

    def __init__(
        self,
        numclasses: int = 5,
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

        # Couverture par cluster
        lambdacoverage: float = 0.1,
        defaultcoverage: Optional[Union[List[float], Tensor]] = None,
        autoregisterunknownclusters: bool = True,

        # Similarité hidden optionnelle
        usesimilarity: bool = False,
        lambdasimilarity: float = 0.0,
        similaritytau: float = 0.85,
        similaritymaxlabelgap: Optional[int] = 1,
        similaritypositivemode: str = "none",  # "none", "both", "at_least_one"
        detachhiddensimilarity: bool = True,

        # Arguments classiques
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        eps: float = 1e-8,
        
        # Couverture EMA
        coverageemamomentum: float = 0.95,
        coveragewarmupupdates: int = 5,

        # Biais de risque dans la couverture EMA
        coverageriskbias: float = 0.0,
        coverageriskweight: float = 0.0,

        # Biais direct dans la cible ordinale
        risklabelshift: float = 0.0,
    ) -> None:

        super(OrdinalUncertaintyFocalWKLoss, self).__init__(
            weight=weight,
            size_average=None,
            reduce=None,
            reduction=reduction,
        )

        self.num_classes = int(numclasses)
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
        elif isinstance(alpha, list):
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

        # Couverture par défaut :
        # fiabilité minimale moyenne attendue par classe.
        if defaultcoverage is None:
            default_coverage_tensor = torch.tensor(
                [0.90, 0.75, 0.70, 0.65, 0.60],
                dtype=torch.float32,
            )
        else:
            default_coverage_tensor = torch.as_tensor(
                defaultcoverage,
                dtype=torch.float32,
            )

        if default_coverage_tensor.numel() != self.num_classes:
            raise ValueError(
                "default_coverage doit être de taille {}.".format(
                    self.num_classes
                )
            )

        self.register_buffer("default_coverage", default_coverage_tensor)

        # Dictionnaire dynamique :
        # cluster_id -> Tensor(num_classes)
        self.coverage_by_cluster = {}

        # WKLoss est supposée déjà définie dans ton code.
        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        
        self.coverage_ema_momentum = float(coverageemamomentum)
        self.coverage_warmup_updates = int(coveragewarmupupdates)

        # coverage_risk_bias > 0 : pousse vers la surestimation du risque
        # coverage_risk_bias < 0 : pousse vers la sous-estimation du risque
        # coverage_risk_bias = 0 : pas de biais directionnel
        self.coverage_risk_bias = float(coverageriskbias)
        self.coverage_risk_weight = float(coverageriskweight)

        # Décale directement le centre de la cible ordinale lissée.
        # > 0 : cible plus pessimiste
        # < 0 : cible plus prudente / sous-estimée
        self.risk_label_shift = float(risklabelshift)

        # Mémoires EMA dynamiques :
        # key = (cluster_id, class_id)
        self.coverage_reliability_ema = {}
        self.coverage_risk_ema = {}
        self.coverage_update_count = {}

    # ============================================================
    # Couverture par cluster
    # ============================================================

    def calculate_class_coverage(
        self,
        df,
        cluster_col: str,
        target_col: str,
        dir_output,
        rho_min: Optional[Union[List[float], Tensor]] = None,
        rho_max: Optional[Union[List[float], Tensor]] = None,
        shrinkage: float = 30.0,
        reset: bool = True,
    ) -> Dict[Any, Tensor]:
        """
        Calcule rho_{cluster, class} à partir d'un DataFrame.

        Cette méthode doit être appelée avant l'entraînement :

            criterion.calculate_class_coverage(
                df=train_df,
                cluster_col="cluster",
                target_col="target"
            )

        Principe :
            - si une classe est fréquente dans un cluster,
              la couverture demandée est proche de rho_max ;
            - si une classe est rare ou absente,
              la couverture demandée reste proche de rho_min.

        Cela évite de forcer une fiabilité irréaliste sur les classes
        très rares tout en empêchant le modèle de les ignorer totalement.
        """

        if rho_min is None:
            rho_min_tensor = torch.tensor(
                [0.80, 0.55, 0.50, 0.45, 0.40],
                dtype=torch.float32,
            )
        else:
            rho_min_tensor = torch.as_tensor(rho_min, dtype=torch.float32)

        if rho_max is None:
            rho_max_tensor = torch.tensor(
                [0.95, 0.80, 0.75, 0.70, 0.65],
                dtype=torch.float32,
            )
        else:
            rho_max_tensor = torch.as_tensor(rho_max, dtype=torch.float32)

        if rho_min_tensor.numel() != self.num_classes:
            raise ValueError(
                "rho_min doit être de taille {}.".format(self.num_classes)
            )

        if rho_max_tensor.numel() != self.num_classes:
            raise ValueError(
                "rho_max doit être de taille {}.".format(self.num_classes)
            )

        if cluster_col not in df.columns:
            raise ValueError("Colonne cluster absente : {}".format(cluster_col))

        if target_col not in df.columns:
            raise ValueError("Colonne target absente : {}".format(target_col))

        if reset:
            self.coverage_by_cluster = {}

        for cluster_value, dfg in df.groupby(cluster_col):
            counts = (
                dfg[target_col]
                .value_counts()
                .reindex(range(self.num_classes), fill_value=0)
                .sort_index()
                .values
            )

            counts_tensor = torch.tensor(counts, dtype=torch.float32)

            support = counts_tensor / (
                counts_tensor + float(shrinkage)
            ).clamp_min(self.eps)

            rho = rho_min_tensor + support * (rho_max_tensor - rho_min_tensor)

            key = self._safe_cluster_key(cluster_value)
            self.coverage_by_cluster[key] = rho.detach().cpu()

        if dir_output is not None:
            import os
            try:
                os.makedirs(dir_output, exist_ok=True)
                with open(os.path.join(dir_output, "class_coverage_rho.csv"), "w") as f:
                    f.write("cluster,class,rho\n")
                    for k, v in self.coverage_by_cluster.items():
                        v_np = v.numpy()
                        for c, rho_val in enumerate(v_np):
                            f.write(f"{k},{c},{rho_val:.4f}\n")
            except Exception as e:
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

    def _get_cluster_coverage(
        self,
        cluster_id: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Retourne rho_g pour un cluster.

        Si le cluster est inconnu dans le batch, on l'ajoute
        automatiquement avec self.default_coverage.
        """

        key = self._safe_cluster_key(cluster_id)

        if key not in self.coverage_by_cluster:
            if self.auto_register_unknown_clusters:
                self.coverage_by_cluster[key] = (
                    self.default_coverage.detach().cpu()
                )

            rho = self.default_coverage
        else:
            rho = self.coverage_by_cluster[key]

        return rho.to(device=device, dtype=dtype)

    # ============================================================
    # Sorties modèle et cibles ordinales
    # ============================================================

    def _extract_outputs(self, logits: Tensor) -> Tensor:
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
            shift > 0 : surestimation volontaire du risque
            shift < 0 : sous-estimation volontaire du risque
        """

        device = y.device
        dtype = uncertainty.dtype

        y = y.long()

        classes = torch.arange(
            self.num_classes,
            device=device,
            dtype=dtype,
        )
        
        # Centre ordinal éventuellement décalé.
        # Exemple :
        #   y = 2, risk_label_shift = +0.25 -> centre 2.25
        #   y = 2, risk_label_shift = -0.25 -> centre 1.75
        y_center = y.to(dtype).view(-1, 1) + self.risk_label_shift
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
    
    def _coverage_loss(
        self,
        reliability: Tensor,
        class_logits: Tensor,
        y: Tensor,
        clusters_ids: Tensor,
    ) -> Tensor:
        """
        Couverture stabilisée par EMA.

        Deux contraintes possibles :

        1. Fiabilité minimale EMA :
            EMA(reliability | cluster=g, y=c) >= rho_{g,c}

        2. Biais directionnel du risque moyen prédit :
            si coverage_risk_bias > 0 :
                EMA(mu | g,c) >= c + coverage_risk_bias

            si coverage_risk_bias < 0 :
                EMA(mu | g,c) <= c + coverage_risk_bias

        avec :
            mu_i = sum_k k * p_i(k)
        """

        device = reliability.device
        dtype = reliability.dtype

        y = y.detach().view(-1)
        clusters_ids = clusters_ids.detach().view(-1)

        probs = F.softmax(class_logits, dim=1)

        classes = torch.arange(
            self.num_classes,
            device=device,
            dtype=dtype,
        )

        # Risque ordinal moyen prédit.
        mu = (probs * classes.view(1, -1)).sum(dim=1)

        beta = self.coverage_ema_momentum

        losses = []
        updates = []

        unique_clusters = torch.unique(clusters_ids)

        for g in unique_clusters:
            g_key = self._safe_cluster_key(g)
            mask_g = clusters_ids == g

            rho_g = self._get_cluster_coverage(
                cluster_id=g_key,
                device=device,
                dtype=dtype,
            )

            for c in range(self.num_classes):
                mask_gc = mask_g & (y == c)

                if not mask_gc.any():
                    continue

                batch_rel = reliability[mask_gc].mean()
                batch_risk = mu[mask_gc].mean()

                old_rel_ema, old_risk_ema, update_count = (
                    self._get_coverage_ema_state(
                        cluster_id=g_key,
                        class_id=c,
                        device=device,
                        dtype=dtype,
                    )
                )

                # EMA anticipée, différentiable par rapport au batch courant.
                rel_ema_hat = beta * old_rel_ema.detach() + (1.0 - beta) * batch_rel
                risk_ema_hat = beta * old_risk_ema.detach() + (1.0 - beta) * batch_risk

                # Warmup progressif : évite une contrainte trop forte
                # quand le couple (cluster, classe) vient juste d'apparaître.
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

                # 1. Contrainte de fiabilité minimale EMA.
                target_rel = rho_g[c]
                rel_loss = F.relu(target_rel - rel_ema_hat).pow(2)

                total_gc_loss = rel_loss

                # 2. Biais directionnel du risque moyen.
                if self.coverage_risk_weight > 0.0 and self.coverage_risk_bias != 0.0:
                    raw_target_risk = float(c) + self.coverage_risk_bias
                    target_risk = max(
                        0.0,
                        min(float(self.num_classes - 1), raw_target_risk),
                    )

                    target_risk = torch.tensor(
                        target_risk,
                        device=device,
                        dtype=dtype,
                    )

                    if self.coverage_risk_bias > 0.0:
                        # Surestimation contrôlée :
                        # on pénalise si le risque moyen EMA est trop bas.
                        risk_loss = F.relu(target_risk - risk_ema_hat).pow(2)

                    else:
                        # Sous-estimation contrôlée :
                        # on pénalise si le risque moyen EMA est trop haut.
                        risk_loss = F.relu(risk_ema_hat - target_risk).pow(2)

                    total_gc_loss = (
                        total_gc_loss
                        + self.coverage_risk_weight * risk_loss
                    )

                losses.append(warmup_tensor * total_gc_loss)

                # Mise à jour EMA après la loss.
                updates.append(
                    (
                        g_key,
                        c,
                        batch_rel.detach(),
                        batch_risk.detach(),
                    )
                )

        if len(updates) > 0 and self.training:
            self._update_coverage_ema(updates)

        if len(losses) == 0:
            return torch.zeros((), device=device, dtype=dtype)

        return torch.stack(losses).mean()

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
        hidden: Optional[Tensor] = None,
        sample_weight: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        class_logits, uncertainty_logits = self._extract_outputs(logits)

        y = self._to_class_index(y).to(class_logits.device)
        clusters_ids = clusters_ids.to(class_logits.device)

        uncertainty = torch.sigmoid(uncertainty_logits)
        reliability = 1.0 - uncertainty

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
            reliability=reliability,
            class_logits=class_logits,
            y=y,
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
            "mean_reliability": reliability.detach().mean(),
            "C": C_value.detach(),
            "id": torch.tensor(
                self.id,
                device=class_logits.device,
                dtype=torch.long,
            ),
        }
        
    def _coverage_key(self, cluster_id: Any, class_id: int) -> Any:
        """
        Clé unique pour les mémoires EMA.
        """
        return (self._safe_cluster_key(cluster_id), int(class_id))


    def _get_coverage_ema_state(
        self,
        cluster_id: Any,
        class_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        """
        Retourne :
            - EMA de fiabilité pour (cluster, classe)
            - EMA de risque moyen prédit pour (cluster, classe)
            - nombre de mises à jour déjà observées
        """

        key = self._coverage_key(cluster_id, class_id)

        rho_g = self._get_cluster_coverage(
            cluster_id=cluster_id,
            device=device,
            dtype=dtype,
        )

        if key not in self.coverage_reliability_ema:
            # Initialisation neutre :
            # fiabilité initiale = couverture attendue
            self.coverage_reliability_ema[key] = (
                rho_g[class_id].detach().cpu()
            )

        if key not in self.coverage_risk_ema:
            # Risque initial = classe observée.
            self.coverage_risk_ema[key] = torch.tensor(
                float(class_id),
                dtype=torch.float32,
            )

        if key not in self.coverage_update_count:
            self.coverage_update_count[key] = 0

        reliability_ema = self.coverage_reliability_ema[key].to(
            device=device,
            dtype=dtype,
        )

        risk_ema = self.coverage_risk_ema[key].to(
            device=device,
            dtype=dtype,
        )

        update_count = int(self.coverage_update_count[key])

        return reliability_ema, risk_ema, update_count


    def _update_coverage_ema(
        self,
        updates: List[tuple],
    ) -> None:
        """
        Met à jour les mémoires EMA après le calcul de la loss.

        Chaque élément de updates contient :
            (cluster_id, class_id, batch_reliability, batch_risk)
        """

        beta = self.coverage_ema_momentum

        with torch.no_grad():
            for cluster_id, class_id, batch_rel, batch_risk in updates:
                key = self._coverage_key(cluster_id, class_id)

                batch_rel_cpu = batch_rel.detach().float().cpu()
                batch_risk_cpu = batch_risk.detach().float().cpu()

                if key not in self.coverage_reliability_ema:
                    self.coverage_reliability_ema[key] = batch_rel_cpu
                else:
                    old = self.coverage_reliability_ema[key]
                    self.coverage_reliability_ema[key] = (
                        beta * old + (1.0 - beta) * batch_rel_cpu
                    )

                if key not in self.coverage_risk_ema:
                    self.coverage_risk_ema[key] = batch_risk_cpu
                else:
                    old = self.coverage_risk_ema[key]
                    self.coverage_risk_ema[key] = (
                        beta * old + (1.0 - beta) * batch_risk_cpu
                    )

                if key not in self.coverage_update_count:
                    self.coverage_update_count[key] = 1
                else:
                    self.coverage_update_count[key] += 1