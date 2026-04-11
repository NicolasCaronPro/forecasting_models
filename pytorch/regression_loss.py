import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

import torch.nn.functional as F
from typing import Optional
from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.loss_utils import *

class PoissonLoss(torch.nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        # Assurer que les prédictions sont positives pour éviter log(0) en utilisant torch.clamp
        y_pred = torch.clamp(y_pred, min=1e-8)
        
        # Calcul de la Poisson Loss
        loss = y_pred - y_true * torch.log(y_pred)
        
        if sample_weights is not None:
            # Appliquer les poids d'échantillons
            weighted_loss = loss * sample_weights
            mean_loss = torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            # Si aucun poids n'est fourni, on calcule la moyenne simple
            mean_loss = torch.mean(loss)
        
        return mean_loss

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        # On ajoute 1 aux prédictions et aux vraies valeurs pour éviter les log(0)
        y_pred = torch.clamp(y_pred, min=1e-8)
        y_true = torch.clamp(y_true, min=1e-8)
        
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        
        # Calcul de la différence au carré
        squared_log_error = (log_pred - log_true) ** 2
        
        if sample_weights is not None:
            # Appliquer les poids d'échantillons
            weighted_squared_log_error = squared_log_error * sample_weights
            mean_squared_log_error = torch.sum(weighted_squared_log_error) / torch.sum(sample_weights)
        else:
            # Si aucun poids n'est fourni, on calcule la moyenne simple
            mean_squared_log_error = torch.mean(squared_log_error)
        
        # Racine carrée pour obtenir la RMSLE
        rmsle = torch.sqrt(mean_squared_log_error)
        
        return rmsle

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        # Calcul de l'erreur au carré
        squared_error = (y_pred - y_true) ** 2
        
        if sample_weights is not None:
            # Appliquer les poids d'échantillons
            weighted_squared_error = squared_error * sample_weights
            mean_squared_error = torch.sum(weighted_squared_error) / torch.sum(sample_weights)
        else:
            # Si aucun poids n'est fourni, on calcule la moyenne simple
            mean_squared_error = torch.mean(squared_error)
        
        # Racine carrée pour obtenir la RMSE
        rmse = torch.sqrt(mean_squared_error)
        
        return rmse

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        error = (y_pred - y_true) ** 2
        if sample_weights is not None:
            weighted_error = error * sample_weights
            return torch.sum(weighted_error) / torch.sum(sample_weights)
        else:
            return torch.mean(error)

class HuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true, sample_weights=None):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        quadratic = torch.where(abs_error <= self.delta, 0.5 * error ** 2, self.delta * (abs_error - 0.5 * self.delta))
        if sample_weights is not None:
            weighted_error = quadratic * sample_weights
            return torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            return torch.mean(quadratic)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        error = y_pred - y_true
        log_cosh = torch.log(torch.cosh(error + 1e-12))  # Adding epsilon to avoid log(0)
        if sample_weights is not None:
            weighted_error = log_cosh * sample_weights
            return torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            return torch.mean(log_cosh)

class TukeyBiweightLoss(torch.nn.Module):
    def __init__(self, c=4.685):
        super(TukeyBiweightLoss, self).__init__()
        self.c = c

    def forward(self, y_pred, y_true, sample_weights=None):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        mask = (abs_error <= self.c).float()
        tukey_loss = (1 - (1 - (error / self.c) ** 2) ** 3) * mask
        tukey_loss = (self.c ** 2 / 6) * tukey_loss
        if sample_weights is not None:
            weighted_error = tukey_loss * sample_weights
            return torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            return torch.mean(tukey_loss)

class ExponentialLoss(torch.nn.Module):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        exp_loss = torch.exp(torch.abs(y_pred - y_true))
        if sample_weights is not None:
            weighted_error = exp_loss * sample_weights
            return torch.sum(weighted_error) / torch.sum(sample_weights)
        else:
            return torch.mean(exp_loss)


class ExponentialAbsoluteErrorLoss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(ExponentialAbsoluteErrorLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        # Calcul de l'erreur absolue
        errors = torch.abs(y_true - y_pred)
        # Application de l'exponentielle
        loss = torch.mean(torch.exp(self.alpha * errors))
        return loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEThetaLoss(nn.Module):
    """
    Loss de régression continue avec apprentissage de theta via Lmid uniquement.

    total_loss = wmse * MSE(score, y_cont) + wmid * Lmid

    - Pas de y_ord
    - Pas de transition loss
    - theta est calculé exactement comme dans la classe d'origine :
        theta0 = alpha[0]
        theta_j = theta_{j-1} + softplus(alpha[j]) pour j >= 1
    - Lmid est construit à partir de score-space centers soft, comme dans
      _score_centers_soft + _loss_mid
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-6,
        wmse: float = 1.0,
        wmid: float = 1.0,
        gamma: float = 5.0,
        taugate: float = 0.05,
        gatetemp: float = 0.11,
        alphatype: str = "department",   # "global", "cluster", "department"
        nclusters: int = 1,
        ndepartements: int = 1,
        mid_detach_mu: bool = True,
    ):
        super().__init__()
        
        self.id = 0

        self.C = int(num_classes)
        self.eps = float(eps)

        self.wmse = float(wmse)
        self.wmid = float(wmid)

        self.gamma = float(gamma)
        self.taugate = float(taugate)
        self.gatetemp = float(gatetemp)

        self.alphatype = str(alphatype)
        self.nclusters = int(nclusters)
        self.ndepartements = int(ndepartements)
        self.mid_detach_mu = bool(mid_detach_mu)

        if self.C < 2:
            raise ValueError("num_classes must be >= 2")

        # Même paramétrisation que dans ta classe originale
        if self.alphatype == "cluster":
            self.alpha = nn.Parameter(torch.zeros(self.nclusters, self.C - 1))
        elif self.alphatype == "department":
            self.alpha = nn.Parameter(torch.zeros(self.ndepartements, self.C - 1))
        else:
            self.alpha = nn.Parameter(torch.zeros(self.C - 1))

        # mappings raw ids -> internal slots
        self.cluster_raw_to_slot = {}
        self.departement_raw_to_slot = {}
        self.cluster_next_free_slot = 0
        self.departement_next_free_slot = 0

        self.register_buffer(
            "cluster_slot_to_raw",
            torch.full((self.nclusters,), -1, dtype=torch.long)
        )
        self.register_buffer(
            "departement_slot_to_raw",
            torch.full((self.ndepartements,), -1, dtype=torch.long)
        )

    def _compute_thresholds(self):
        """
        Exactement la même construction que dans ta classe d'origine.
        """
        alpha = self.alpha

        if alpha.dim() == 1:
            theta0 = alpha[0:1]
            if alpha.numel() > 1:
                incr = F.softplus(alpha[1:])
                theta = torch.cat([theta0, incr], dim=0).cumsum(dim=0)
            else:
                theta = theta0
            return theta
        else:
            theta0 = alpha[:, 0:1]
            if alpha.size(1) > 1:
                incr = F.softplus(alpha[:, 1:])
                theta = torch.cat([theta0, incr], dim=1).cumsum(dim=1)
            else:
                theta = theta0
            return theta

    def _remap_ids(self, raw_ids: torch.Tensor, buf_size: int, kind: str):
        if raw_ids.dim() != 1:
            raw_ids = raw_ids.view(-1)

        raw_ids = raw_ids.long()
        device = raw_ids.device

        if kind == "cluster":
            raw_to_slot = self.cluster_raw_to_slot
            slot_to_raw = self.cluster_slot_to_raw
            next_free_attr = "cluster_next_free_slot"
        elif kind == "department":
            raw_to_slot = self.departement_raw_to_slot
            slot_to_raw = self.departement_slot_to_raw
            next_free_attr = "departement_next_free_slot"
        else:
            raise ValueError(f"Unknown kind: {kind}")

        local_ids = torch.empty_like(raw_ids, dtype=torch.long, device=device)
        next_free_slot = getattr(self, next_free_attr)

        for i in range(raw_ids.numel()):
            rid = int(raw_ids[i].item())

            if rid in raw_to_slot:
                slot = raw_to_slot[rid]
            else:
                if next_free_slot >= buf_size:
                    raise ValueError(
                        f"No free slot left for kind='{kind}'. "
                        f"Encountered new raw id {rid}, but buf_size={buf_size}."
                    )
                slot = next_free_slot
                raw_to_slot[rid] = slot
                slot_to_raw[slot] = rid
                next_free_slot += 1

            local_ids[i] = slot

        setattr(self, next_free_attr, next_free_slot)

        valid_mask = torch.ones_like(raw_ids, dtype=torch.bool, device=device)
        return slot_to_raw.clone(), local_ids, valid_mask

    def _class_probs_from_score(self, s, clusters_ids=None, departement_ids=None):
        """
        Convertit le score scalaire en probabilités ordinales via theta.
        """
        theta = self._compute_thresholds().to(s.device, dtype=s.dtype)

        if theta.dim() == 1:
            Fk = torch.sigmoid(theta[None, :] - s[:, None])
        else:
            if self.alphatype == "cluster":
                if clusters_ids is None:
                    raise ValueError("clusters_ids is required when alphatype='cluster'")
                chosen_ids = clusters_ids.clamp(0, theta.shape[0] - 1)
            elif self.alphatype == "department":
                if departement_ids is None:
                    raise ValueError("departement_ids is required when alphatype='department'")
                chosen_ids = departement_ids.clamp(0, theta.shape[0] - 1)
            else:
                raise ValueError(f"Unknown alphatype: {self.alphatype}")

            thr = theta.index_select(0, chosen_ids.long())
            Fk = torch.sigmoid(thr - s[:, None])

        p = s.new_zeros((s.size(0), self.C))
        p[:, 0] = Fk[:, 0]
        if self.C > 2:
            p[:, 1:-1] = Fk[:, 1:] - Fk[:, :-1]
        p[:, -1] = 1.0 - Fk[:, -1]

        p = torch.nan_to_num(p, nan=self.eps, posinf=1.0, neginf=0.0)
        p = p.clamp_min(0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return p

    def _group_centers_from_weights(self, y, weights, group_ids_local, Z):
        device = y.device
        dtype = y.dtype

        m_k = torch.zeros(Z, device=device, dtype=dtype)
        m_k.scatter_add_(0, group_ids_local, weights)

        wy = weights * y
        sum_wy = torch.zeros(Z, device=device, dtype=dtype)
        sum_wy.scatter_add_(0, group_ids_local, wy)

        mu_hat = sum_wy / m_k.clamp_min(self.eps)
        return m_k, mu_hat

    def _score_centers_soft(
        self,
        scores,
        p,
        clusters_ids_local,
        departement_ids_local=None,
        sw=None,
        active_cluster_slots=None,
        active_dept_slots=None,
    ):
        """
        Même logique que ton _score_centers_soft :
        centres de classes dans l'espace score, calculés soft.
        """
        scores = scores.view(-1).to(dtype=p.dtype, device=p.device)
        device = p.device

        gate = torch.sigmoid((p - self.taugate) / max(self.gatetemp, 1e-6))
        p_eff = p * gate

        if self.gamma != 1.0:
            p_eff = p_eff.clamp_min(self.eps).pow(self.gamma)
            p_eff = p_eff / p_eff.sum(dim=1, keepdim=True).clamp_min(self.eps)

        Zc = len(active_cluster_slots) if active_cluster_slots is not None else (
            int(clusters_ids_local.max().item()) + 1 if clusters_ids_local.numel() > 0 else 1
        )

        Zd = len(active_dept_slots) if active_dept_slots is not None else (
            int(departement_ids_local.max().item()) + 1
            if departement_ids_local is not None and departement_ids_local.numel() > 0
            else 0
        )

        score_clusters = torch.zeros(Zc, self.C, device=device, dtype=p.dtype)
        mass_clusters = torch.zeros(Zc, self.C, device=device, dtype=p.dtype)

        score_departements = (
            torch.zeros(Zd, self.C, device=device, dtype=p.dtype) if Zd > 0 else None
        )
        mass_departements = (
            torch.zeros(Zd, self.C, device=device, dtype=p.dtype) if Zd > 0 else None
        )

        sw_eff = sw.to(device=device, dtype=p.dtype).clamp_min(self.eps) if sw is not None else None

        for k in range(self.C):
            weights = p_eff[:, k]
            if sw_eff is not None:
                weights = weights * sw_eff

            m_k_c, s_hat_c = self._group_centers_from_weights(
                y=scores,
                weights=weights,
                group_ids_local=clusters_ids_local,
                Z=Zc,
            )
            score_clusters[:, k] = s_hat_c
            mass_clusters[:, k] = m_k_c

            if Zd > 0 and departement_ids_local is not None:
                m_k_d, s_hat_d = self._group_centers_from_weights(
                    y=scores,
                    weights=weights,
                    group_ids_local=departement_ids_local,
                    Z=Zd,
                )
                score_departements[:, k] = s_hat_d
                mass_departements[:, k] = m_k_d

        return score_clusters, mass_clusters, score_departements, mass_departements

    def _loss_mid(self, mu_ref: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
        """
        Même logique que dans ton _loss_mid :
            theta_k ~ 0.5 * (mu_k + mu_{k+1})
        """
        if mu_ref.dim() == 1:
            mu_ref = mu_ref.unsqueeze(0)
        if theta_ref.dim() == 1:
            theta_ref = theta_ref.unsqueeze(0)

        if mu_ref.shape[-1] != self.C:
            raise ValueError(f"mu_ref last dim must be {self.C}, got {mu_ref.shape}")
        if theta_ref.shape[-1] != self.C - 1:
            raise ValueError(f"theta_ref last dim must be {self.C - 1}, got {theta_ref.shape}")

        if theta_ref.shape[0] == 1 and mu_ref.shape[0] > 1:
            theta_ref = theta_ref.expand(mu_ref.shape[0], -1)
        elif mu_ref.shape[0] == 1 and theta_ref.shape[0] > 1:
            mu_ref = mu_ref.expand(theta_ref.shape[0], -1)

        if mu_ref.shape[0] != theta_ref.shape[0]:
            raise ValueError(
                f"mu_ref and theta_ref are not aligned: {mu_ref.shape} vs {theta_ref.shape}"
            )

        target_mid = 0.5 * (mu_ref[:, :-1] + mu_ref[:, 1:])

        if self.mid_detach_mu:
            target_mid = target_mid.detach()

        valid = torch.isfinite(target_mid) & torch.isfinite(theta_ref)
        if not valid.any():
            return theta_ref.new_tensor(0.0)

        return F.smooth_l1_loss(theta_ref[valid], target_mid[valid], reduction="mean")

    def forward(self, score, y_cont, clusters_ids, departement_ids, sample_weight=None):
        """
        score : (N,) ou (N,1)
        y_cont : (N,)
        """
        s = score.view(-1)
        y_cont = y_cont.view(-1).to(device=s.device, dtype=s.dtype)

        if s.numel() != y_cont.numel():
            raise ValueError(
                f"score and y_cont must have same number of elements, got {s.numel()} and {y_cont.numel()}"
            )

        sw = sample_weight.view(-1).to(device=s.device, dtype=s.dtype) if sample_weight is not None else None
        if sw is not None and sw.numel() != s.numel():
            raise ValueError("sample_weight must have same batch size as score")

        cluster_slot_ids = None
        dept_slot_ids = None
        active_cluster_slots = None
        active_dept_slots = None
        local_cluster_ids = None
        local_dept_ids = None

        if self.alphatype == "cluster":
            if clusters_ids is None:
                raise ValueError("clusters_ids is required when alphatype='cluster'")
            clusters_ids = clusters_ids.view(-1).long().to(device=s.device)
            _, cluster_slot_ids, _ = self._remap_ids(clusters_ids, self.nclusters, kind="cluster")
            active_cluster_slots, local_cluster_ids = torch.unique(cluster_slot_ids, return_inverse=True)

        elif self.alphatype == "department":
            if departement_ids is None:
                raise ValueError("departement_ids is required when alphatype='department'")
            departement_ids = departement_ids.view(-1).long().to(device=s.device)
            _, dept_slot_ids, _ = self._remap_ids(departement_ids, self.ndepartements, kind="department")
            active_dept_slots, local_dept_ids = torch.unique(dept_slot_ids, return_inverse=True)

            # pour conserver le code de _score_centers_soft, on utilise un groupe unique cluster
            local_cluster_ids = torch.zeros_like(local_dept_ids)
            active_cluster_slots = torch.tensor([0], device=s.device, dtype=torch.long)

        else:
            # global
            local_cluster_ids = torch.zeros_like(s, dtype=torch.long)
            active_cluster_slots = torch.tensor([0], device=s.device, dtype=torch.long)

        # 1) MSE
        mse_per_sample = (s - y_cont).pow(2)
        if sw is not None:
            mse_loss = (mse_per_sample * sw).sum() / sw.sum().clamp_min(self.eps)
        else:
            mse_loss = mse_per_sample.mean()

        # 2) probs induites par score + theta
        probs = self._class_probs_from_score(
            s,
            clusters_ids=cluster_slot_ids if self.alphatype == "cluster" else None,
            departement_ids=dept_slot_ids if self.alphatype == "department" else None,
        )

        # 3) centres soft en espace score
        score_centers_cluster, _, score_centers_dept, _ = self._score_centers_soft(
            s,
            probs,
            local_cluster_ids,
            departement_ids_local=local_dept_ids,
            sw=sw,
            active_cluster_slots=active_cluster_slots,
            active_dept_slots=active_dept_slots,
        )

        theta_all = self._compute_thresholds().to(device=s.device, dtype=s.dtype)

        # 4) même logique de choix que dans ton forward initial
        if theta_all.dim() == 1:
            score_ref = score_centers_cluster.mean(dim=0, keepdim=True)
            theta_mid = theta_all.unsqueeze(0)

        elif self.alphatype == "department":
            theta_mid = theta_all.index_select(0, active_dept_slots)

            if score_centers_dept is not None and score_centers_dept.shape[0] > 0:
                score_ref = score_centers_dept
            else:
                score_ref = score_centers_cluster.mean(dim=0, keepdim=True).expand(theta_mid.shape[0], -1)

        elif self.alphatype == "cluster":
            theta_mid = theta_all.index_select(0, active_cluster_slots)
            score_ref = score_centers_cluster

        else:
            raise ValueError(f"Unknown alphatype: {self.alphatype}")

        # 5) apprentissage de theta via les milieux
        Lmid = self._loss_mid(score_ref, theta_mid)

        total_loss = self.wmse * mse_loss + self.wmid * Lmid
        return total_loss

    @torch.no_grad()
    def score_to_class(self, scores, clusters_ids=None, departement_ids=None):
        s = scores.view(-1).to(dtype=self.alpha.dtype)
        device = s.device

        thr = self._compute_thresholds().detach().to(device=device)

        if thr.dim() == 1:
            return torch.bucketize(s, thr, right=True)
        else:
            if self.alphatype == "cluster":
                if clusters_ids is None:
                    raise ValueError("clusters_ids is required when alphatype='cluster'")
                chosen_ids = clusters_ids.view(-1).long().to(device=device)
                _, idx, _ = self._remap_ids(chosen_ids, self.nclusters, kind="cluster")
            elif self.alphatype == "department":
                if departement_ids is None:
                    raise ValueError("departement_ids is required when alphatype='department'")
                chosen_ids = departement_ids.view(-1).long().to(device=device)
                _, idx, _ = self._remap_ids(chosen_ids, self.ndepartements, kind="department")
            else:
                raise ValueError(f"Unknown alphatype: {self.alphatype}")

            thr_s = thr.index_select(0, idx)
            return (s[:, None] > thr_s).sum(dim=1)

    def get_learnable_parameters(self):
        return {"alpha": self.alpha}

    def get_attribute(self):
        return [('ordinal_params', {
            "alpha": self.alpha.detach().cpu().numpy(),
            "thresholds": self._compute_thresholds().detach().cpu().numpy(),
        })]

    def update_params(self, new_dict, epoch=None):
        payload = new_dict

        if isinstance(payload, dict) and "ordinal_params" in payload:
            payload = payload["ordinal_params"]

        if hasattr(payload, "numpy") and not isinstance(payload, dict):
            payload = payload.numpy()

        if not isinstance(payload, dict):
            raise TypeError(f"update_params expected dict-like payload, got {type(payload)}")

        if "alpha" in payload and payload["alpha"] is not None:
            alpha_new = torch.as_tensor(
                payload["alpha"],
                dtype=self.alpha.dtype,
                device=self.alpha.device,
            )
            if alpha_new.shape != self.alpha.shape:
                raise ValueError(
                    f"alpha shape mismatch: got {tuple(alpha_new.shape)}, "
                    f"expected {tuple(self.alpha.shape)}"
                )
            with torch.no_grad():
                self.alpha.copy_(alpha_new)