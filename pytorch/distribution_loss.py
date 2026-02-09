import torch.nn.functional as F
from typing import Optional
from forecasting_models.pytorch.ordinal_loss import *
import math
from pathlib import Path
import numpy as np
from typing import Any

def cdf_egpd_family1(y, sigma, xi, kappa, eps: float = 1e-12):
    """
    eGPD (famille 1)
      F(y) = [ 1 - (1 + xi * y / sigma)^(-1/xi) ]^kappa    pour y >= 0

    Gère exactement :
      - xi > 0  : support infini
      - xi = 0  : limite exponentielle
      - xi < 0  : support fini y < -sigma/xi (CDF=1 au-delà)

    Paramètres:
      y      : Tensor (...), non-négatif (sera clampé)
      sigma  : Tensor/nn.Parameter (...), échelle > 0 (auto-clamp)
      xi     : Tensor/nn.Parameter (...), peut être <0, =0, >0
      kappa  : Tensor/nn.Parameter (...), > 0 (auto-clamp)

    Tous les tenseurs sont broadcastés automatiquement.
    """
    # --- aligne device/dtype SANS casser l'autograd si ce sont déjà des Tensors/Parameters ---
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    device, dtype = y.device, y.dtype

    if torch.is_tensor(sigma):
        sigma = sigma.to(device=device, dtype=dtype)
    else:
        sigma = torch.tensor(sigma, device=device, dtype=dtype)

    if torch.is_tensor(xi):
        xi = xi.to(device=device, dtype=dtype)
    else:
        xi = torch.tensor(xi, device=device, dtype=dtype)

    if torch.is_tensor(kappa):
        kappa = kappa.to(device=device, dtype=dtype)
    else:
        kappa = torch.tensor(kappa, device=device, dtype=dtype)

    # --- clamps de base ---
    y  = torch.clamp(y, min=0.0)
    si = torch.clamp(sigma, min=eps)
    ka = torch.clamp(kappa, min=eps)

    # masque exact xi == 0  (limite exponentielle)
    is_zero = (xi.abs() < 1e-6)

    # pour les éléments xi != 0, on sécurise xi loin de 0 en gardant le signe
    xi_nz = torch.where(is_zero, torch.ones_like(xi), xi)  # placeholder sur les zéros
    xi_safe = torch.where(xi_nz >= 0,
                          torch.clamp(xi_nz, min=eps),
                          torch.clamp(xi_nz, max=-eps))

    # ----- Branche xi == 0 : limite exponentielle -----
    H_exp = 1.0 - torch.exp(-y / si)
    H_exp = torch.clamp(H_exp, min=eps, max=1.0 - eps)
    F_exp = torch.pow(H_exp, ka)

    # ----- Branche xi != 0 -----
    # support fini si xi<0 : y < -sigma/xi
    neg_xi = (xi < 0)
    y_max = -si / xi_safe
    valid_range = torch.where(neg_xi, y < y_max, torch.ones_like(y, dtype=torch.bool))

    z = 1.0 + xi_safe * (y / si)
    z = torch.where(valid_range, torch.clamp(z, min=1.0 + 1e-12), torch.ones_like(z, dtype=dtype))

    H = 1.0 - torch.pow(z, -1.0 / xi_safe)
    H = torch.clamp(H, min=eps, max=1.0 - eps)
    F_nz = torch.pow(H, ka)

    # en dehors du support (xi<0 & y>=y_max) -> F=1
    F_nz = torch.where(neg_xi & (~valid_range), torch.ones_like(F_nz), F_nz)

    # ----- Combinaison finale -----
    F = torch.where(is_zero, F_exp, F_nz)
    return F

class EGPDNLLLoss(torch.nn.Module):
    def __init__(self, kappa: float = 0.831, xi: float = 0.161,
                 eps: float = 1e-8, reduction: str = "mean",
                 force_positive: bool = True,
                 area_exponent: float = 1.0,   # =1.0 si tu modèles BA ; =0.5 si sqrt(BA)
                 num_classes = 5,
                 ):
        super().__init__()
        self.kappa = torch.nn.Parameter(torch.tensor(kappa))
        self.xi    = torch.nn.Parameter(torch.tensor(xi))
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.area_exponent = area_exponent  # contrôle l’offset d’échelle via a**alpha
        self.num_classes = num_classes

    def _pos_params(self):
        if self.force_positive:
            return F.softplus(self.kappa), F.softplus(self.xi)
        return self.kappa, self.xi

    @torch.no_grad()
    def _egpd_icdf(self, u: torch.Tensor, sigma: torch.Tensor,
                   kappa: torch.Tensor, xi: torch.Tensor, eps: float):
        # eGPD inverse CDF (conditionnelle), stable numériquement
        u = torch.clamp(u, eps, 1.0 - eps)
        v = u ** (1.0 / torch.clamp(kappa, min=eps))
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)
        near0 = xi_safe.abs() < 1e-6
        y_near0 = -sigma * torch.log1p(-(v))
        y_xi = (sigma / xi_safe) * ((1.0 - v).clamp(min=eps) ** (-xi_safe) - 1.0)
        y = torch.where(near0, y_near0, y_xi)
        return y.clamp_min(0.0)

    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool,
                      area = None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        if area is not None:
            # applique l’offset d’échelle : sigma <- sigma * a**alpha
            sigma = sigma * torch.clamp(area, min=self.eps) ** self.area_exponent
        return sigma

    # ---------- Entraînement : NLL eGPD (conditionnelle, sur y>0) ----------
    def forward(self, inputs: torch.Tensor, y: torch.Tensor,
                weight: torch.Tensor = None, from_logits: bool = True,
                area = None) -> torch.Tensor:
        """
        inputs: (...,) sigma_raw (ta tête de réseau pour l’échelle)
        y:      cible > 0 (la BA chez toi, car pas de sqrt)
        area:   surface a(s) optionnelle pour l’offset d’échelle
        """
        inputs = inputs[:, 0]
        sigma = self._decode_sigma(inputs, from_logits, area)
        kappa, xi = self._pos_params()

        pos = (y > 0)
        # z = 1 + xi * y / sigma ; calculé seulement où pos
        z = 1.0 + xi * (y / sigma)
        z = torch.where(pos, z.clamp_min(1.0 + 1e-12), torch.ones_like(z))

        # H(y) = 1 - z^{-1/xi} ; h(y) = (1/sigma) * z^{-1/xi - 1}
        log_h = torch.where(pos, -torch.log(sigma) + (-1.0 / xi - 1.0) * torch.log(z), torch.zeros_like(y))
        H = torch.where(pos, 1.0 - torch.pow(z, -1.0 / xi), torch.zeros_like(y))
        H = torch.where(pos, H.clamp(max=1.0 - 1e-12), H)

        # log g(y) = log kappa + log h(y) + (kappa - 1) * log H(y)
        log_g = torch.where(pos, torch.log(kappa) + log_h + (kappa - 1.0) * torch.log(H.clamp_min(self.eps)),
                            torch.zeros_like(y))

        nll = torch.where(pos, -(log_g), torch.zeros_like(y))
        if weight is not None:
            nll = nll * weight

        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll

    @torch.no_grad()
    def transform(self,
                inputs: torch.Tensor, dir_output, from_logits: bool = True,
                area = None,
                u: float = 0.5,
                ):
        """
        Renvoie le quantile conditionnel de la variable que tu modèles.
        - Si tu modèles BA directement (pas de sqrt), c’est un quantile de BA.
        - Si un jour tu repasses à sqrt(BA), passe area_exponent=0.5 et
          élève au carré pour retourner un quantile d’aire.
        """
        inputs = inputs[:, 0]
        sigma = self._decode_sigma(inputs, from_logits, area)
        kappa, xi = self._pos_params()
        if not torch.is_tensor(u):
            u = torch.tensor(u, dtype=sigma.dtype, device=sigma.device)
        y_q = self._egpd_icdf(u, sigma, kappa, xi, self.eps)
        self.plot_final_pdf(inputs, dir_output)
        return y_q
    
    def get_learnable_parameters(self):
        return {"kappa" : self.kappa, "xi" : self.xi}
    
    def get_attribute(self):
        if self.force_positive:
            return [('kappa', F.softplus(self.kappa)), ('xi', F.softplus(self.xi))]
        else:
            return [('kappa', self.kappa), ('xi', self.xi)]
        
    def plot_params(self, egpd_logs, dir_output):
        """Sauvegarde les paramètres EGPD (kappa, xi) et trace leurs évolutions en fonction des epochs."""

        # Extraction directe (car egpd_log est un dict {epoch: {"kappa":..., "xi":...}})
        
        kappas = [egpd_log["kappa"] for egpd_log in egpd_logs]
        xis = [egpd_log["xi"] for egpd_log in egpd_logs]
        epochs = [egpd_log["epoch"] for egpd_log in egpd_logs]

        # Sauvegarde pickle
        egpd_to_save = {"epoch": epochs, "kappa": kappas, "xi": xis}
        save_object(egpd_to_save, 'egpd_kappa_xi.pkl', dir_output)

        # kappa vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, kappas, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('kappa')
        plt.title('EGPD kappa over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'egpd_kappa_over_epochs.png')
        plt.close()

        # xi vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, xis, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('xi')
        plt.title('EGPD xi over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'egpd_xi_over_epochs.png')
        plt.close()
        
    @torch.no_grad()
    def plot_final_pdf(self,
                    sigma: torch.Tensor,        # tenseur des échelles (décodées) du batch
                    dir_output,                 # dossier de sortie (str | Path)
                    num_points: int = 1000,     # points pour la grille
                    which_sigmas: int = 3,      # nb de courbes sigma à tracer (min/med/max)
                    u_max: float = 0.999):      # quantile haut pour fixer l'axe Y
        """
        Trace la densité eGPD conditionnelle sur l'échelle BA (modèle direct BA):
        g(y) = kappa * h(y) * H(y)^{kappa-1},
        avec H(y) = 1 - (1 + xi*y/sigma)^(-1/xi) et h(y) = (1/sigma)*(1 + xi*y/sigma)^(-1/xi - 1).

        - 'sigma' doit être le tenseur d'échelles DÉCODÉES (après softplus et éventuel offset area).
        Ex: sigma = self._decode_sigma(inputs, from_logits=True, area=area)
        - Enregistre un PNG 'egpd_pdf_ba.png' dans dir_output.
        """
        # Prépare sortie
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        # Récupère paramètres positifs
        kappa, xi = self._pos_params()
        kappa = torch.clamp(kappa, min=self.eps)
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)

        # Sélectionne quelques sigmas représentatives (min/med/max)
        sig = sigma.detach().flatten()
        if sig.numel() == 0:
            return
        sig_sorted, _ = torch.sort(sig)
        idxs = torch.linspace(0, sig_sorted.numel()-1, steps=min(which_sigmas, sig_sorted.numel())).round().long()
        sig_list = sig_sorted[idxs].tolist()

        # Détermine la borne supérieure de l'axe BA via l'ICDF(eGPD) au quantile u_max pour la plus grande sigma
        s_ref = torch.tensor(max(sig_list), device=sigma.device, dtype=sigma.dtype)
        y_max = self._egpd_icdf(torch.tensor(float(u_max), device=sigma.device, dtype=sigma.dtype),
                                s_ref, kappa, xi_safe, self.eps).item()
        # Sécurité si xi ~ 0 ou si la borne est trop petite
        y_max = max(y_max, float(s_ref.item()) * 6.0)
        y_max = float(y_max)

        # Grille BA (éviter 0 strict)
        y_grid = torch.linspace(0.0, y_max, num_points, device=sigma.device, dtype=sigma.dtype)

        # Fonction densité (BA direct)
        def egpd_pdf_ba(y: torch.Tensor, s: torch.Tensor, kap: torch.Tensor, xis: torch.Tensor) -> torch.Tensor:
            # z = 1 + xi * y / sigma
            z = 1.0 + xis * (y / torch.clamp(s, min=self.eps))
            z = torch.clamp(z, min=1.0 + 1e-12)

            # H(y), h(y)
            H = 1.0 - torch.pow(z, -1.0 / torch.clamp(xis, min=1e-12))
            H = torch.clamp(H, min=self.eps, max=1.0 - 1e-12)
            h = (1.0 / torch.clamp(s, min=self.eps)) * torch.pow(z, (-1.0 / torch.clamp(xis, min=1e-12)) - 1.0)

            g = torch.clamp(kap, min=self.eps) * h * torch.pow(H, torch.clamp(kap, min=self.eps) - 1.0)
            return g

        # Tracé
        plt.figure(figsize=(7, 5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            pdf_vals = egpd_pdf_ba(y_grid, s_t, kappa, xi_safe)
            plt.plot(y_grid.cpu().numpy(), pdf_vals.cpu().numpy(), label=f"sigma={float(s):.3f}")

        plt.xlabel("BA")
        plt.ylabel("Conditional PDF g(BA)")
        plt.title(f"eGPD PDF on BA  (kappa={float(kappa):.3f}, xi={float(xi_safe):.3f})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(dir_output / "egpd_pdf_ba.png", dpi=200)
        plt.close()
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

class EGPDNLLLossSqrt(torch.nn.Module):
    """
    eGPD NLL (famille 1) APPRISE sur Y = sqrt(BA) pour les échantillons positifs (y > 0).
    - Entraînement (forward): NLL eGPD sur sqrt(y).
    - Inférence (transform): sélection via seuil de PDF (p_thresh) sur sqrt(BA), puis carré pour BA.
    """

    def __init__(self, kappa: float = 0.831, xi: float = 0.161,
                 eps: float = 1e-8, reduction: str = "mean",
                 force_positive: bool = True,
                 area_exponent: float = 0.5,  # =0.5 comme dans l'article (sqrt(BA))
                 num_classes=5):
        super().__init__()
        self.kappa = torch.nn.Parameter(torch.tensor(kappa), requires_grad=True)
        self.xi    = torch.nn.Parameter(torch.tensor(xi),    requires_grad=True)
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.area_exponent = area_exponent
        self.num_classes = num_classes

    # ---------- contraintes de positivité ----------
    def _pos_params(self):
        if self.force_positive:
            return F.softplus(self.kappa), F.softplus(self.xi)
        return self.kappa, self.xi

    # ---------- inverse-CDF eGPD sur l'échelle sqrt(BA) ----------
    @torch.no_grad()
    def _egpd_icdf(self, u: torch.Tensor, sigma: torch.Tensor,
                   kappa: torch.Tensor, xi: torch.Tensor, eps: float):
        u = torch.clamp(u, eps, 1.0 - eps)
        v = u ** (1.0 / torch.clamp(kappa, min=eps))
        xi_safe = torch.clamp(xi, min=-1e-6, max=1e6)
        near0 = xi_safe.abs() < 1e-6
        y_near0 = -sigma * torch.log1p(-v)                                   # xi ~ 0
        y_xi    = (sigma / xi_safe) * ((1.0 - v).clamp(min=eps) ** (-xi_safe) - 1.0)
        y = torch.where(near0, y_near0, y_xi)
        return y.clamp_min(0.0)  # sqrt(BA)

    # ---------- pdf eGPD sur l'échelle sqrt(BA) ----------
    def _egpd_pdf_sqrt(self, y_sqrt: torch.Tensor, sigma: torch.Tensor,
                       kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y_sqrt = torch.clamp(y_sqrt, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        z = 1.0 + xi * (y_sqrt / sigma)
        z = torch.clamp(z, min=1.0 + 1e-12)
        H = 1.0 - torch.pow(z, -1.0 / xi)
        H = torch.clamp(H, min=self.eps, max=1.0 - 1e-12)
        h = (1.0 / sigma) * torch.pow(z, (-1.0 / xi) - 1.0)
        kpos = torch.clamp(kappa, min=self.eps)
        g = kpos * h * torch.pow(H, kpos - 1.0)
        return g

    # ---------- décodage sigma (+ offset surface éventuel) ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, area=None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        if area is not None:
            sigma = sigma * torch.clamp(area, min=self.eps) ** self.area_exponent
        return sigma

    # ---------- Entraînement : NLL eGPD sur Y = sqrt(BA) ----------
    def forward(self, inputs: torch.Tensor, y: torch.Tensor, areas,
                weight: torch.Tensor = None, from_logits: bool = True) -> torch.Tensor:
        inputs = inputs[:, 0]
        sigma = self._decode_sigma(inputs, from_logits, areas)
        kappa, xi = self._pos_params()

        pos = (y > 0)
        y_sqrt = torch.zeros_like(y)
        y_sqrt = torch.where(pos, torch.sqrt(torch.clamp(y, min=self.eps)), y_sqrt)

        z = 1.0 + xi * (y_sqrt / sigma)
        z = torch.where(pos, z.clamp_min(1.0 + 1e-12), torch.ones_like(z))

        log_h = torch.where(pos, -torch.log(sigma) + (-1.0 / xi - 1.0) * torch.log(z),
                            torch.zeros_like(y))
        H = torch.where(pos, 1.0 - torch.pow(z, -1.0 / xi), torch.zeros_like(y))
        H = torch.where(pos, H.clamp(max=1.0 - 1e-12), H)

        kpos = torch.clamp(kappa, min=self.eps)
        log_g = torch.where(pos,
                            torch.log(kpos) + log_h +
                            (kpos - 1.0) * torch.log(H.clamp_min(self.eps)),
                            torch.zeros_like(y))

        nll = torch.where(pos, -(log_g), torch.zeros_like(y))
        if weight is not None:
            nll = nll * weight
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- CDF eGPD sur sqrt(BA) ----------
    def _egpd_cdf_sqrt(self, y_sqrt: torch.Tensor, sigma: torch.Tensor,
                       kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y_sqrt = torch.clamp(y_sqrt, min=0.0)
        sigma  = torch.clamp(sigma,  min=self.eps)
        z = 1.0 + xi * (y_sqrt / sigma)
        z = torch.clamp(z, min=1.0 + 1e-12)
        xi_safe = torch.clamp(xi, min=-1e-6, max=1e6)
        near0   = xi_safe.abs() < 1e-6
        H_xi  = 1.0 - torch.pow(z, -1.0 / xi_safe)
        H_0   = 1.0 - torch.exp(-y_sqrt / sigma)
        H     = torch.where(near0, H_0, H_xi)
        H     = torch.clamp(H, min=self.eps, max=1.0 - 1e-12)
        kpos = torch.clamp(kappa, min=self.eps)
        return torch.pow(H, kpos)

    # ---------- NEW: sélection par seuil de PDF (bord min/max) ----------
    @torch.no_grad()
    def _y_from_pdf_threshold(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                              p_thresh: float = 1e-3, pick: str = "min",
                              num_points: int = 2000, u_max: float = 0.999):
        """
        Cherche un y (sur sqrt(BA)) tel que PDF(y) >= p_thresh.
        - pick="min": bord gauche de l'ensemble {y: g(y) >= p_thresh}
        - pick="max": bord droit
        Repli: mode (argmax PDF) si aucun point ne dépasse p_thresh.
        Retourne un tenseur (N,) si sigma est batch, sinon scalaire tensor.
        """
        assert p_thresh > 0, "p_thresh must be > 0"
        assert pick in ("min", "max")

        if sigma.ndim == 0:
            sigma = sigma.view(1)
        dev, dt = sigma.device, sigma.dtype

        # borne supérieure via quantile u_max
        u = torch.full((sigma.size(0),), float(u_max), device=dev, dtype=dt)
        y_max = self._egpd_icdf(u, sigma, kappa, xi, self.eps).clamp_min(sigma * 6.0)
        ymax_global = y_max.max().item()
        y_grid = torch.linspace(0.0, ymax_global, num_points, device=dev, dtype=dt)  # (G,)

        # PDF sur la grille
        G = torch.stack([self._egpd_pdf_sqrt(y_grid, s, kappa, xi) for s in sigma], dim=0)  # (N,G)

        mask = (G >= p_thresh)
        ys = y_grid.unsqueeze(0).expand_as(G)

        if pick == "min":
            large = torch.full_like(ys, float('inf'))
            cand = torch.where(mask, ys, large)
            y_sel = cand.min(dim=1).values
            # repli: mode
            need = torch.isinf(y_sel)
            if need.any():
                y_mode = y_grid[G.argmax(dim=1)]
                y_sel[need] = y_mode[need]
        else:
            negi = torch.full_like(ys, float('-inf'))
            cand = torch.where(mask, ys, negi)
            y_sel = cand.max(dim=1).values
            need = torch.isinf(y_sel)
            if need.any():
                y_mode = y_grid[G.argmax(dim=1)]
                y_sel[need] = y_mode[need]

        return y_sel  # sur sqrt(BA)

    # ---------- Inférence : seuil PDF -> sqrt(BA) -> BA, + plots ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, areas, dir_output, from_logits: bool = True,
                  p_thresh: float = 0.5, pick: str = "min"):
        """
        Retourne une PRÉDICTION EN BA :
        - trouve y_hat (sur sqrt(BA)) comme bord min/max où PDF(y) >= p_thresh,
        - retourne BA_pred = y_hat**2.
        Si dir_output est fourni, trace PDF/CDF/PPF via plot_final().
        """
        inputs = inputs[:, 0]
        sigma = self._decode_sigma(inputs, from_logits, areas)
        kappa, xi = self._pos_params()

        y_sqrt_sel = self._y_from_pdf_threshold(
            sigma, kappa, xi,
            p_thresh=float(p_thresh),
            pick=pick,
            num_points=2000,
            u_max=0.999
        )
        ba_pred = y_sqrt_sel ** 2

        if dir_output is not None:
            self.plot_final(sigma, dir_output)

        return ba_pred

    # ---------- Plot PDF, CDF & PPF (sqrt et BA) + MAX(sigma) ----------
    @torch.no_grad()
    def plot_final(self, sigma: torch.Tensor, dir_output,
                   num_points: int = 1000, which_sigmas: int = 'max'):
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        kappa, xi = self._pos_params()

        sig = sigma.detach().flatten()
        if sig.numel() == 0:
            return
        sig_sorted, _ = torch.sort(sig)
        sig_max_val = float(sig_sorted.max().item())
        if which_sigmas == 'mean':
            sig_list = [sig_sorted.mean()]
        elif which_sigmas == 'max':
            sig_list = [sig_sorted.max()]
        else:
            idxs = torch.linspace(0, sig_sorted.numel()-1,
                                  steps=min(which_sigmas, sig_sorted.numel())).round().long()
            sig_list = sig_sorted[idxs].tolist()

        # grille en sqrt(BA) basée sur le quantile 0.999 pour la plus grande sigma
        s_ref = torch.tensor(max([float(s) for s in sig_list]), device=sigma.device, dtype=sigma.dtype)
        y_max = self._egpd_icdf(torch.tensor(0.999, device=sigma.device, dtype=sigma.dtype),
                                s_ref, kappa, xi, self.eps).item()
        y_max = max(y_max, float(s_ref.item()) * 6.0)
        y_grid = torch.linspace(0.0, y_max, num_points, device=sigma.device, dtype=sigma.dtype)

        # 1) PDF sqrt(BA)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(float(s), device=sigma.device, dtype=sigma.dtype)
            pdf_y = self._egpd_pdf_sqrt(y_grid, s_t, kappa, xi)
            plt.plot(y_grid.cpu().numpy(), pdf_y.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("y = sqrt(BA)")
        plt.ylabel("Conditional PDF g_Y(y)")
        plt.title(f"eGPD PDF on sqrt(BA)  (kappa={float(kappa):.3f}, xi={float(xi):.3f}, max sigma={sig_max_val:.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / "egpd_pdf_default_sqrt.png", dpi=200); plt.close()

        # 2) CDF sqrt(BA)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(float(s), device=sigma.device, dtype=sigma.dtype)
            cdf_y = self._egpd_cdf_sqrt(y_grid, s_t, kappa, xi)
            plt.plot(y_grid.cpu().numpy(), cdf_y.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("y = sqrt(BA)")
        plt.ylabel("CDF  G_Y(y)")
        plt.title(f"eGPD CDF on sqrt(BA)  (kappa={float(kappa):.3f}, xi={float(xi):.3f}, max sigma={sig_max_val:.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / "egpd_cdf_sqrt.png", dpi=200); plt.close()

        # 3) PDF BA
        z_max = (y_max ** 2)
        z_grid = torch.linspace(1e-12, z_max, num_points, device=sigma.device, dtype=sigma.dtype)
        y_from_z = torch.sqrt(z_grid)

        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(float(s), device=sigma.device, dtype=sigma.dtype)
            g_y = self._egpd_pdf_sqrt(y_from_z, s_t, kappa, xi)
            f_z = g_y / (2.0 * torch.clamp(y_from_z, min=1e-12))
            plt.plot(z_grid.cpu().numpy(), f_z.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("BA")
        plt.ylabel("Conditional PDF on BA,  f_Z(z)")
        plt.title(f"Final PDF on BA via Z=Y^2  (kappa={float(kappa):.3f}, xi={float(xi):.3f}, max sigma={sig_max_val:.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / "egpd_pdf_ba.png", dpi=200); plt.close()

        # 4) CDF BA
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(float(s), device=sigma.device, dtype=sigma.dtype)
            F_z = self._egpd_cdf_sqrt(y_from_z, s_t, kappa, xi)
            plt.plot(z_grid.cpu().numpy(), F_z.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("BA")
        plt.ylabel("CDF  F_Z(z) = G_Y(sqrt(z))")
        plt.title(f"CDF on BA via Z=Y^2  (kappa={float(kappa):.3f}, xi={float(xi):.3f}, max sigma={sig_max_val:.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / "egpd_cdf_ba.png", dpi=200); plt.close()

        # 5) PPF sqrt(BA)
        u_grid = torch.linspace(1e-6, 1.0 - 1e-6, num_points, device=sigma.device, dtype=sigma.dtype)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(float(s), device=sigma.device, dtype=sigma.dtype)
            q_u = self._egpd_icdf(u_grid, s_t, kappa, xi, self.eps)
            plt.plot(u_grid.cpu().numpy(), q_u.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel("q_u (on sqrt(BA))")
        plt.title(f"PPF on sqrt(BA)  (kappa={float(kappa):.3f}, xi={float(xi):.3f}, max sigma={sig_max_val:.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / "egpd_ppf_sqrt.png", dpi=200); plt.close()

        # 6) PPF BA
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(float(s), device=sigma.device, dtype=sigma.dtype)
            q_u = self._egpd_icdf(u_grid, s_t, kappa, xi, self.eps)
            q_ba = q_u ** 2
            plt.plot(u_grid.cpu().numpy(), q_ba.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel("q_u^2 (on BA)")
        plt.title(f"PPF on BA  (kappa={float(kappa):.3f}, xi={float(xi):.3f}, max sigma={sig_max_val:.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / "egpd_ppf_ba.png", dpi=200); plt.close()

    # ---------- Accès params ----------
    def get_learnable_parameters(self):
        return {"kappa": self.kappa, "xi": self.xi}

    def get_attribute(self):
        if self.force_positive:
            return [('kappa', F.softplus(self.kappa)), ('xi', F.softplus(self.xi))]
        else:
            return [('kappa', self.kappa), ('xi', self.xi)]

    def plot_params(self, egpd_logs, dir_output):
        kappas = [egpd_log["kappa"] for egpd_log in egpd_logs]
        xis    = [egpd_log["xi"] for egpd_log in egpd_logs]
        epochs = [egpd_log["epoch"] for egpd_log in egpd_logs]
        egpd_to_save = {"epoch": epochs, "kappa": kappas, "xi": xis}
        save_object(egpd_to_save, 'egpd_kappa_xi.pkl', dir_output)
        plt.figure(figsize=(8, 5)); plt.plot(epochs, kappas, marker='o')
        plt.xlabel('Epoch'); plt.ylabel('kappa'); plt.title('EGPD kappa over epochs')
        plt.grid(True, linestyle='--', alpha=0.4); plt.tight_layout()
        plt.savefig(dir_output / 'egpd_kappa_over_epochs.png'); plt.close()
        plt.figure(figsize=(8, 5)); plt.plot(epochs, xis, marker='o')
        plt.xlabel('Epoch'); plt.ylabel('xi'); plt.title('EGPD xi over epochs')
        plt.grid(True, linestyle='--', alpha=0.4); plt.tight_layout()
        plt.savefig(dir_output / 'egpd_xi_over_epochs.png'); plt.close()

class EGPDNLLLossSqrtClusterIDs(nn.Module):
    """
    eGPD NLL (famille 1) APPRISE sur Y = sqrt(BA) pour y>0,
    avec paramètres kappa/xi dépendant des clusters et optionnellement
    un terme GLOBAL (partial pooling) si `add_global=True`.

    - Entraînement (forward): on applique sqrt à la cible (BA -> sqrt(BA))
      et on maximise la NLL eGPD sur sqrt(BA), pour y>0.
    - Inférence (transform): on calcule un quantile conditionnel sur sqrt(BA),
      puis on LE CARRE pour obtenir une prédiction en BA.

    Paramétrisation:
      G(y) = H(y)^kappa, où H est la CDF de la GPD(sigma, xi) sur sqrt(BA).

    Entrées:
      - areas: (N,) optionnel — facteur multiplicatif d’échelle pour sigma
      - clusters_ids: (N,) long — sélection de kappa, xi par sample
    """

    def __init__(
        self,
        NC: int,                          # nb de clusters
        id: int = None,
        kappa_init: float = 0.831,
        xi_init: float = 0.161,
        eps: float = 1e-8,
        reduction: str = "mean",
        force_positive: bool = False,     # impose kappa>0, xi>0 via softplus
        xi_min: float = 0.0,              # petit plancher additionnel sur xi si force_positive
        G: bool = False,         # = G
        area_exponent: float = 0.5,       # 0.5 pour sqrt(BA)
        num_classes: int = 5,
    ):
        super().__init__()
        self.n_clusters     = NC
        self.id             = id
        self.eps            = eps
        self.reduction      = reduction
        self.force_positive = force_positive
        self.xi_min         = xi_min
        self.add_global     = G
        self.area_exponent  = area_exponent
        self.num_classes    = num_classes
        
        # stocker la dernière série de clusters_ids vus (facultatif, pour inspection)
        self.clusters_ids = None

        if not G:
            # --- Variante "clusters uniquement" ---
            self.kappa_raw = nn.Parameter(torch.full((NC,), float(kappa_init)))
            self.xi_raw    = nn.Parameter(torch.full((NC,), float(xi_init)))
        else:
            # --- Variante "global + delta par cluster" ---
            self.kappa_global = nn.Parameter(torch.tensor(float(kappa_init)))
            self.xi_global    = nn.Parameter(torch.tensor(float(xi_init)))
            self.kappa_delta  = nn.Parameter(torch.zeros(NC))
            self.xi_delta     = nn.Parameter(torch.zeros(NC))

    # ---------- sélecteur de (kappa_raw, xi_raw) par sample ----------
    def _select_raw_params(self, clusters_ids: torch.Tensor):
        if not self.add_global:
            kappa_raw_sel = self.kappa_raw[clusters_ids]                 # (N,)
            xi_raw_sel    = self.xi_raw[clusters_ids]                    # (N,)
        else:
            kappa_raw_sel = self.kappa_global + self.kappa_delta[clusters_ids]  # (N,)
            xi_raw_sel    = self.xi_global    + self.xi_delta[clusters_ids]     # (N,)
        return kappa_raw_sel, xi_raw_sel

    # ---------- contrainte de positivité (appliquée après sélection) ----------
    def _positivize(self, kappa_raw_sel: torch.Tensor, xi_raw_sel: torch.Tensor):
        if self.force_positive:
            kappa = F.softplus(kappa_raw_sel)                   # >0
            xi    = F.softplus(xi_raw_sel) + float(self.xi_min) # > xi_min
        else:
            kappa = kappa_raw_sel
            xi    = xi_raw_sel
        return kappa, xi

    # ---------- inverse-CDF eGPD sur l'échelle sqrt(BA) ----------
    @torch.no_grad()
    def _egpd_icdf(self, u: torch.Tensor, sigma: torch.Tensor,
                   kappa: torch.Tensor, xi: torch.Tensor, eps: float):
        """
        Quantile conditionnel eGPD (sur sqrt(BA)) pour u in (0,1):
          v = u^(1/kappa)
          xi != 0: y = (sigma/xi) * ((1 - v)^(-xi) - 1)
          xi -> 0: y = -sigma * log(1 - v)
        """
        u = torch.clamp(u, eps, 1.0 - eps)
        v = u ** (1.0 / torch.clamp(kappa, min=eps))
        xi_safe = torch.clamp(xi, min=-1e-6, max=1e6)  # protéger xi≈0 et bornes
        near0 = xi_safe.abs() < 1e-6

        y_near0 = -sigma * torch.log1p(-v)                                   # xi ~ 0
        y_xi    = (sigma / xi_safe) * ((1.0 - v).clamp(min=eps) ** (-xi_safe) - 1.0)  # xi != 0
        y = torch.where(near0, y_near0, y_xi)
        return y.clamp_min(0.0)  # sur sqrt(BA)

    # ---------- pdf eGPD sur l'échelle sqrt(BA) ----------
    def _egpd_pdf_sqrt(self, y_sqrt: torch.Tensor, sigma: torch.Tensor,
                       kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        Densité conditionnelle g_Y(y) sur l'échelle sqrt(BA).
        """
        y_sqrt = torch.clamp(y_sqrt, min=0.0)
        sigma  = torch.clamp(sigma, min=self.eps)
        z = 1.0 + xi * (y_sqrt / sigma)
        z = torch.clamp(z, min=1.0 + 1e-12)

        # H(y) = 1 - z^{-1/xi},  h(y) = (1/sigma) * z^{-1/xi - 1}
        H = 1.0 - torch.pow(z, -1.0 / xi)
        H = torch.clamp(H, min=self.eps, max=1.0 - 1e-12)
        h = (1.0 / sigma) * torch.pow(z, (-1.0 / xi) - 1.0)

        g = torch.clamp(kappa, min=self.eps) * h * torch.pow(H, torch.clamp(kappa, min=self.eps) - 1.0)
        return g
    
    # ---------- décodage de sigma (+ offset surface éventuel) ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, areas=None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        if areas is not None:
            sigma = sigma * torch.clamp(areas, min=self.eps) ** self.area_exponent
        return sigma

    # ---------- Entraînement : NLL eGPD conditionnelle sur sqrt(BA) ----------
    def forward(
        self,
        inputs: torch.Tensor,     # (...,) sigma_raw (tête réseau échelle)
        y: torch.Tensor,          # (N,) cible en BA (>=0)
        clusters_ids: torch.Tensor,# (N,) longs, pour kappa/xi
        areas: torch.Tensor = None,# (N,) optionnel, offset d'échelle
        weight: torch.Tensor = None,
        from_logits: bool = True
    ) -> torch.Tensor:
        """
        Calcule la NLL eGPD sur sqrt(BA) pour les échantillons positifs (y>0),
        avec (kappa, xi) dépendants du cluster.
        """

        # sigma conditionnelle (avec offset area éventuel)
        sigma_raw = inputs[:, 0]
        sigma = self._decode_sigma(sigma_raw, from_logits, areas)

        # sélectionner et “positiviser” kappa/xi selon les clusters
        kappa_raw_sel, xi_raw_sel = self._select_raw_params(clusters_ids.long())
        kappa, xi = self._positivize(kappa_raw_sel, xi_raw_sel)

        # positifs: y>0  (travail sur sqrt(y))
        pos = (y > 0)
        y_sqrt = torch.where(pos, torch.sqrt(torch.clamp(y, min=self.eps)), torch.zeros_like(y))

        # z = 1 + xi * y_sqrt / sigma
        z = 1.0 + xi * (y_sqrt / sigma)
        z = torch.where(pos, z.clamp_min(1.0 + 1e-12), torch.ones_like(z))

        # log h, H
        log_h = torch.where(pos, -torch.log(sigma) + (-1.0 / xi - 1.0) * torch.log(z), torch.zeros_like(y))
        H     = torch.where(pos, 1.0 - torch.pow(z, -1.0 / xi), torch.zeros_like(y))
        H     = torch.where(pos, H.clamp(max=1.0 - 1e-12), H)

        # log g(y) = log kappa + log h + (kappa - 1) * log H
        log_g = torch.where(
            pos,
            torch.log(torch.clamp(kappa, min=self.eps)) + log_h +
            (torch.clamp(kappa, min=self.eps) - 1.0) * torch.log(H.clamp_min(self.eps)),
            torch.zeros_like(y)
        )

        nll = torch.where(pos, -(log_g), torch.zeros_like(y))
        if weight is not None:
            nll = nll * weight

        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- Plot PDF par cluster (sqrt et BA) ----------
    @torch.no_grad()
    def _plot_one_cluster_pdfs(self, sigmas_c: torch.Tensor, kappa_c: torch.Tensor, xi_c: torch.Tensor,
                               dir_output, cname: str, num_points: int = 1000):
        """
        Trace les densités pour un cluster donné (liste de sigmas observées).
        Sauvegarde deux figures: sqrt(BA) et BA, suffixées par le nom du cluster.
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        # choisir des sigmas représentatives (min/med/max)
        sig = sigmas_c.flatten()
        if sig.numel() == 0:
            return
        sig_sorted, _ = torch.sort(sig)
        idxs = torch.linspace(0, sig_sorted.numel()-1, steps=min(3, sig_sorted.numel())).round().long()
        sig_list = sig_sorted[idxs].tolist()

        # grille en sqrt(BA) basée sur q0.999 de la plus grande sigma
        s_ref = torch.tensor(max(sig_list), device=sigmas_c.device, dtype=sigmas_c.dtype)
        y_max = self._egpd_icdf(torch.tensor(0.999, device=sigmas_c.device, dtype=sigmas_c.dtype),
                                s_ref, kappa_c, xi_c, self.eps).item()
        y_max = max(y_max, float(s_ref.item()) * 6.0)
        y_grid = torch.linspace(0.0, y_max, num_points, device=sigmas_c.device, dtype=sigmas_c.dtype)

        # --- PDF sur sqrt(BA)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigmas_c.device, dtype=sigmas_c.dtype)
            pdf_y = self._egpd_pdf_sqrt(y_grid, s_t, kappa_c, xi_c)
            plt.plot(y_grid.cpu().numpy(), pdf_y.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("y = sqrt(BA)")
        plt.ylabel("Conditional PDF g_Y(y)")
        plt.title(f"eGPD PDF on sqrt(BA) — cluster {cname}  (kappa={float(kappa_c):.3f}, xi={float(xi_c):.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        fname_sqrt = dir_output / f"egpd_pdf_{cname}_sqrt.png"
        plt.tight_layout(); plt.savefig(fname_sqrt, dpi=200); plt.close()

        # --- PDF transformée sur BA : f_Z(z) = g_Y(sqrt(z)) / (2 sqrt(z))
        z_max = (y_max ** 2)
        z_grid = torch.linspace(1e-12, z_max, num_points, device=sigmas_c.device, dtype=sigmas_c.dtype)
        y_from_z = torch.sqrt(z_grid)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigmas_c.device, dtype=sigmas_c.dtype)
            g_y = self._egpd_pdf_sqrt(y_from_z, s_t, kappa_c, xi_c)
            f_z = g_y / (2.0 * torch.clamp(y_from_z, min=1e-12))
            plt.plot(z_grid.cpu().numpy(), f_z.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("BA"); plt.ylabel("Conditional PDF on BA, f_Z(z)")
        plt.title(f"Final PDF on BA — cluster {cname}  (kappa={float(kappa_c):.3f}, xi={float(xi_c):.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        fname_ba = dir_output / f"egpd_pdf_{cname}_ba.png"
        plt.tight_layout(); plt.savefig(fname_ba, dpi=200); plt.close()

    @torch.no_grad()
    def plot_final_pdf(self, sigma: torch.Tensor, clusters_ids: torch.Tensor, dir_output,
                       num_points: int = 1000, cluster_names=None):
        """
        Trace les PDF **par cluster** (sqrt et BA).
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        clusters_ids = clusters_ids.long()
        C = self.n_clusters
        # reconstruire kappa/xi par cluster (après positivisation)
        if not self.add_global:
            kappa_full = self.kappa_raw
            xi_full    = self.xi_raw
        else:
            kappa_full = self.kappa_global + self.kappa_delta
            xi_full    = self.xi_global    + self.xi_delta

        if self.force_positive:
            kappa_full = F.softplus(kappa_full)
            xi_full    = F.softplus(xi_full) + float(self.xi_min)

        for c in range(C):
            mask = (clusters_ids == c)
            if mask.any():
                sig_c = sigma[mask]
                self._plot_one_cluster_pdfs(
                    sig_c,
                    kappa_full[c],
                    xi_full[c],
                    dir_output,
                    cname=f"cl{c}" if cluster_names is None else str(cluster_names[c]),
                    num_points=num_points
                )
                
    # ---------- Inférence : quantile conditionnel puis retour en BA ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, clusters_ids: torch.Tensor, areas: torch.Tensor = None,
                  dir_output=None, from_logits: bool = True, u: float = 0.5, plot: bool = False,
                  cluster_names=None):
        """
        Retourne une PRÉDICTION EN BA:
          - calcule le quantile conditionnel u sur sqrt(BA) avec (kappa,xi) selon le cluster,
          - BA_pred = (quantile_sqrt)^2.

        Si plot=True (ou si dir_output non None), trace et sauvegarde les PDF par cluster.
        """
        sigma_raw = inputs[:, 0]
        sigma = self._decode_sigma(sigma_raw, from_logits, areas)

        # sélectionner kappa/xi par sample
        kappa_raw_sel, xi_raw_sel = self._select_raw_params(clusters_ids.long())
        kappa, xi = self._positivize(kappa_raw_sel, xi_raw_sel)

        if not torch.is_tensor(u):
            u = torch.tensor(u, dtype=sigma.dtype, device=sigma.device)

        y_sqrt_q = self._egpd_icdf(u, sigma, kappa, xi, self.eps)  # quantile sur sqrt(BA)
        ba_pred = y_sqrt_q ** 2

        if dir_output is not None or plot:
            self.plot_final_pdf(sigma, clusters_ids, dir_output, cluster_names=cluster_names)

        return ba_pred

    # ---------- Utilitaires d'inspection ----------
    def get_learnable_parameters(self):
        if not self.add_global:
            return {"kappa": self.kappa_raw, "xi": self.xi_raw}
        else:
            return {
                "kappa_global": self.kappa_global,
                "xi_global": self.xi_global,
                "kappa_delta": self.kappa_delta,
                "xi_delta": self.xi_delta,
            }

    def get_attribute(self):
        if not self.add_global:
            if self.force_positive:
                return [
                    ('kappa', F.softplus(self.kappa_raw).detach()),
                    ('xi',    (F.softplus(self.xi_raw) + self.xi_min).detach()),
                ]
            else:
                return [
                    ('kappa', self.kappa_raw.detach()),
                    ('xi',    self.xi_raw.detach()),
                ]
        else:
            kappa_raw_full = self.kappa_global + self.kappa_delta
            xi_raw_full    = self.xi_global    + self.xi_delta
            if self.force_positive:
                return [
                    ('kappa_global', F.softplus(self.kappa_global).detach()),
                    ('xi_global',    (F.softplus(self.xi_global) + self.xi_min).detach()),
                    ('kappa_per_cluster', F.softplus(kappa_raw_full).detach()),
                    ('xi_per_cluster',    (F.softplus(xi_raw_full) + self.xi_min).detach()),
                ]
            else:
                return [
                    ('kappa_global', self.kappa_global.detach()),
                    ('xi_global',    self.xi_global.detach()),
                    ('kappa_per_cluster', kappa_raw_full.detach()),
                    ('xi_per_cluster',    xi_raw_full.detach()),
                ]

    def shrinkage_penalty(self, lambda_l2: float = 1e-4) -> torch.Tensor:
        """
        Pénalité L2 sur les déviations par cluster (0 si add_global=False).
        À ajouter à la perte totale : total = nll + loss_fn.shrinkage_penalty(lambda_l2)
        """
        if not self.add_global:
            device = (self.kappa_raw if hasattr(self, "kappa_raw") else self.kappa_global).device
            return torch.zeros((), device=device)
        return lambda_l2 * (self.kappa_delta.pow(2).sum() + self.xi_delta.pow(2).sum())

    # ---------- Plot: un fichier PNG par cluster + (optionnel) global au fil des epochs ----------
    def plot_params(self, egpd_logs, dir_output, cluster_names=None, dpi=120):
        """
        Sauvegarde et trace les paramètres EGPD au fil des epochs, **par cluster**.
        Si add_global=True, trace aussi les paramètres globaux.
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        epochs = [int(log["epoch"]) for log in egpd_logs]

        if not self.add_global:
            kappas_raw = [log["kappa"] for log in egpd_logs]  # (C,)
            xis_raw    = [log["xi"]    for log in egpd_logs]  # (C,)
        else:
            kappas_raw = [log["kappa_per_cluster"] for log in egpd_logs]  # (C,)
            xis_raw    = [log["xi_per_cluster"]    for log in egpd_logs]  # (C,)
            kappa_global = [log["kappa_global"] for log in egpd_logs]     # scalaire
            xi_global    = [log["xi_global"]    for log in egpd_logs]     # scalaire

        # --- Conversion numpy ---
        def to_np_vec(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        K_list = [to_np_vec(k) for k in kappas_raw]
        X_list = [to_np_vec(x) for x in xis_raw]
        K = np.stack(K_list, axis=0)  # (T, C)
        X = np.stack(X_list, axis=0)
        T, C = K.shape

        if cluster_names is not None and len(cluster_names) != C:
            raise ValueError(f"'cluster_names' length ({len(cluster_names)}) must match C ({C}).")

        # Sauvegarde brute
        egpd_to_save = {"epoch": epochs, "kappa": K_list, "xi": X_list}
        if self.add_global:
            egpd_to_save["kappa_global"] = kappa_global
            egpd_to_save["xi_global"] = xi_global
        save_object(egpd_to_save, 'egpd_kappa_xi.pkl', dir_output)

        import re
        def _slug(s):
            s = str(s)
            s = re.sub(r'\s+', '_', s.strip())
            s = re.sub(r'[^\w\-_]+', '', s)
            return s

        # --- Un fichier par cluster ---
        for c in range(C):
            name = cluster_names[c] if cluster_names is not None else f"cl{c}"
            slug = _slug(name)

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, K[:, c], marker='o', label=r'$\kappa$')
            plt.plot(epochs, X[:, c], marker='s', label=r'$\xi$')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title(f'EGPD parameters over epochs — {name}')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(dir_output) / f'egpd_params_{slug}.png', dpi=dpi)
            plt.close()

        # --- Courbes globales (si add_global=True) ---
        if self.add_global:
            kappa_global = np.asarray([to_np_vec(k) for k in kappa_global])
            xi_global    = np.asarray([to_np_vec(x) for x in xi_global])

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, kappa_global, marker='o', label=r'$\kappa_{global}$')
            plt.plot(epochs, xi_global,    marker='s', label=r'$\xi_{global}$')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Global EGPD parameters over epochs')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(dir_output) / f'egpd_params_global.png', dpi=dpi)
            plt.close()

class dEGPDLossTrunc(nn.Module):
    """
    Discrete exponentiated GPD (deGPD) truncated to {0,..., y_max}.
    - Forward: NLL on truncated PMF.
    - Transform: returns a discrete y selected by a PMF threshold (p_thresh) with tie policy.
    PMF (truncated):
        p_trunc(y) = [F_+(y+1) - F_+(y)] / F_+(y_max+1),  for y in {0,...,y_max}
    """

    def __init__(self,
                 kappa: float = 0.8,
                 xi: float = 0.15,
                 eps: float = 1e-8,
                 reduction: str = "mean",
                 force_positive: bool = False,
                 y_max: int = 4):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(kappa), requires_grad=True)
        self.xi    = nn.Parameter(torch.tensor(xi),    requires_grad=True)
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.y_max = int(y_max)

    # ---------- contraintes de positivité sur (kappa, xi) ----------
    def _pos_params(self):
        if self.force_positive:
            return F.softplus(self.kappa), F.softplus(self.xi)
        return self.kappa, self.xi

    # ---------- décodage de sigma (+ offset éventuel) ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, area=None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        # pas d'offset ici (comptes), gardé pour compat API
        return sigma

    # ---------- CDF GPD H(y; sigma, xi) ----------
    def _gpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y     = torch.clamp(y, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)
        near0   = xi_safe.abs() < 1e-6
        z = 1.0 + xi_safe * (y / sigma)
        z = torch.clamp(z, min=self.eps)
        H_xi = 1.0 - torch.pow(z, -1.0 / xi_safe)   # xi != 0
        H_0  = 1.0 - torch.exp(-y / sigma)          # xi ~ 0
        H = torch.where(near0, H_0, H_xi)
        return torch.clamp(H, min=self.eps, max=1.0 - 1e-12)

    # ---------- eGPD CDF: F_+(y) = [H(y)]^kappa ----------
    def _egpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        H = self._gpd_cdf(y, sigma, xi)
        kappa_pos = torch.clamp(kappa, min=self.eps)
        Fp = torch.pow(H, kappa_pos)
        return torch.clamp(Fp, min=self.eps, max=1.0 - 1e-12)

    # ---------- PMF brute: p_raw(y) = F_+(y+1) - F_+(y) ----------
    def _pmf_raw(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y = y_int.to(dtype=sigma.dtype)
        F_y   = self._egpd_cdf(y,   sigma, kappa, xi)
        F_yp1 = self._egpd_cdf(y+1, sigma, kappa, xi)
        p_raw = torch.clamp(F_yp1 - F_y, min=self.eps)
        return p_raw

    # ---------- PMF TRONQUÉE sur {0,...,y_max} ----------
    def _pmf_trunc(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        p_raw = self._pmf_raw(y_int, sigma, kappa, xi)
        Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                               sigma, kappa, xi)  # normalisation: F_+(y_max+1)
        Z = torch.clamp(Z, min=self.eps)
        p = p_raw / Z
        return torch.clamp(p, min=self.eps, max=1.0)  # clamp sup

    # ---------- NLL TRONQUÉE ----------
    def forward(
        self,
        inputs: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        from_logits: bool = True,
    ) -> torch.Tensor:
        sigma_raw = inputs[..., 0] if inputs.ndim > 1 else inputs
        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._pos_params()

        y_int = y.to(torch.long).clamp(min=0, max=self.y_max)
        p_trunc = self._pmf_trunc(y_int, sigma, kappa, xi)

        nll = -torch.log(torch.clamp(p_trunc, min=self.eps))
        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(nll.device, nll.dtype)
            nll = nll * sample_weight
        if self.reduction == "mean":
            if sample_weight is not None:
                return nll.sum() / sample_weight.sum().clamp_min(self.eps)
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- Trouver y tel que PMF(y) >= p_thresh (min/max/all) ----------
    @torch.no_grad()
    def _y_where_p_ge(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                      p_thresh, pick):
        """
        Retourne:
          - si sigma scalaire: int y
          - si sigma batch:    tensor (N,) de y
        Politique si aucun y ne vérifie PMF>=p_thresh : repli sur argmax PMF.
        """
        if isinstance(p_thresh, torch.Tensor):
            assert torch.all(p_thresh >= 0.0) and torch.all(p_thresh <= 1.0), "p_thresh must be in (0,1)"
        else:
            assert p_thresh >=0 and p_thresh <= 1.0, "p_thresh must be in (0,1)"
            
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        
        if sigma.ndim == 0:
            pmf = self._pmf_trunc(y_vals, sigma, kappa, xi)  # (y_max+1,)
            mask = (pmf >= p_thresh)
            if not mask.any():
                return int(torch.argmax(pmf).item())
            idxs = torch.nonzero(mask, as_tuple=False).flatten()
            return int((idxs.min() if pick == "min" else idxs.max()).item())

        # batch
        pmf = self._pmf_trunc(y_vals.unsqueeze(0).expand(sigma.shape[0], -1),
                              sigma.unsqueeze(-1), kappa, xi)  # (N, y_max+1)
        
        if isinstance(p_thresh, torch.Tensor):
            mask = (pmf >= p_thresh.view(-1,1))
        else:
            mask = (pmf >= p_thresh)
        y_index = torch.arange(self.y_max + 1, device=sigma.device).unsqueeze(0).expand_as(pmf)

        if pick == "min":
            large = torch.full_like(y_index, self.y_max + 1)
            cand = torch.where(mask, y_index, large)
            y_hat = cand.min(dim=1).values.clamp_max(self.y_max)
        else:  # "max"
            neg = torch.full_like(y_index, -1)
            cand = torch.where(mask, y_index, neg)
            y_hat = cand.max(dim=1).values.clamp_min(0)

        # repli pour les lignes sans True
        no_hit = ~mask.any(dim=1)
        if no_hit.any():
            fallback = pmf.argmax(dim=1)
            y_hat[no_hit] = fallback[no_hit]
        return y_hat

    # ---------- Transform: sélection par seuil de PMF + tracés optionnels ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output,  p_thresh,
                  from_logits: bool = True,
                  pick: str = "max"):
        """
        Retourne un y discret dans {0,...,y_max} tel que PMF(y) >= p_thresh,
        avec tie-breaking contrôlé par `pick`:
          - "min": plus petit y satisfaisant le seuil,
          - "max": plus grand y satisfaisant le seuil.
        Repli: argmax PMF si aucun y ne dépasse le seuil.
        Si dir_output n'est pas None, trace PMF/CDF/PPF via plot_final().
        """
        
        sigma_raw = inputs[..., 0] if inputs.ndim > 1 else inputs
        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        
        if isinstance(p_thresh, dict):
            p_thresh = torch.sigmoid(p_thresh['a'] + sigma * p_thresh['b'])
            
        kappa, xi = self._pos_params()

        y_hat = self._y_where_p_ge(sigma, kappa, xi, p_thresh=p_thresh, pick=pick)
        
        if dir_output is not None:
            self.plot_final(sigma, dir_output, which_sigmas='mean')
            self.plot_final(sigma, dir_output, which_sigmas='max')
            
        return y_hat
    
     # ---------- Accès params ----------
    def get_learnable_parameters(self):
        return {"kappa": self.kappa, "xi": self.xi}

    def get_attribute(self):
        if self.force_positive:
            return [('kappa', F.softplus(self.kappa)), ('xi', F.softplus(self.xi))]
        else:
            return [('kappa', self.kappa), ('xi', self.xi)]

    # ---------- Plots: PMF/CDF/PPF (tronqués) ----------
    @torch.no_grad()
    def plot_final(self, sigma: torch.Tensor, dir_output,
                   which_sigmas='max'):
        
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        kappa, xi = self._pos_params()

        sig = sigma.detach().flatten()
        if sig.numel() == 0:
            return
        sig_sorted, _ = torch.sort(sig)
        
        if which_sigmas == 'mean':
            sig_list = [sig_sorted.mean().item()]
        elif which_sigmas == 'max':
            sig_list = [sig_sorted.max().item()]
        else:
            idxs = torch.linspace(0, sig_sorted.numel()-1,
                                  steps=min(which_sigmas, sig_sorted.numel())).round().long()
            sig_list = sig_sorted[idxs].tolist()
            
        # 1) PMF tronquée
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            p_raw = self._pmf_raw(y_vals, s_t, kappa, xi)
            Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                   s_t, kappa, xi)
            p     = torch.clamp(p_raw / torch.clamp(Z, min=self.eps), min=self.eps)
            plt.stem(y_vals.cpu().numpy(), p.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated PMF on {{0,...,{self.y_max}}}")
        plt.title(f"deGPD truncated PMF  (kappa={float(kappa):.3f}, xi={float(xi):.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_pmf = dir_output / f"degpd_trunc_pmf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_pmf, dpi=200); plt.close()

        # 2) CDF tronquée
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            F_y = self._egpd_cdf(y_vals, s_t, kappa, xi)
            Z   = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                 s_t, kappa, xi)
            F_tr = torch.clamp(F_y / torch.clamp(Z, min=self.eps), min=self.eps, max=1.0)
            plt.step(y_vals.cpu().numpy(), F_tr.cpu().numpy(), where="post", label=f"sigma={float(s):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated CDF on {{0,...,{self.y_max}}}")
        plt.title(f"deGPD truncated CDF  (kappa={float(kappa):.3f}, xi={float(xi):.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_cdf = dir_output / f"degpd_trunc_cdf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_cdf, dpi=200); plt.close()

        # 3) PPF discrète via balayage de u ∈ (0,1) -> y(u)
        u_grid = torch.linspace(1e-6, 1.0 - 1e-6, 1000, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            Fmax = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                  s_t, kappa, xi)
            u_eff = torch.clamp(u_grid * torch.clamp(Fmax, min=self.eps), min=self.eps, max=1.0 - 1e-12)
            F_tab = self._egpd_cdf(y_vals, s_t, kappa, xi)
            y_idx = torch.searchsorted(F_tab, u_eff, right=False).clamp(0, self.y_max)
            plt.plot(u_grid.cpu().numpy(), y_idx.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel(f"PPF (discrete y on {{0,...,{self.y_max}}})")
        plt.title(f"deGPD truncated PPF  (kappa={float(kappa):.3f}, xi={float(xi):.3f})")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_ppf = dir_output / f"degpd_trunc_ppf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_ppf, dpi=200); plt.close()
        
    def calibrate(self,
                    inputs: torch.Tensor,
                    y_true : torch.Tensor,
                    score_fn,               # callable: preds -> float
                    dir_output,
                    log_sigma: bool = False,
                    from_logtis : bool = True,
                    a_grid=None,
                    b_grid=None,
                    plot: bool = True):
        """
        Calibre u(x) = sigmoid(a + b * phi(sigma)) en maximisant `score_fn`.
        - `sigma`: tensor (N,) ou (N,1), déjà sur l'échelle d'échelle (pas de logits).
        - `score_fn(preds) -> float`: calcule le score final à maximiser à partir des
        prédictions discrètes (par ex. classes après transform avec quantile u).
        NOTE: `score_fn` doit capturer la vérité terrain (closure) si nécessaire.
        - `dir_output`: dossier où sauvegarder la figure (si plot=True).
        - `areas`: offset éventuel passé à `transform` (exposition) ; None pour ignorer.
        - `log_sigma`: si True, phi(sigma) = log(sigma), sinon phi(sigma)=sigma.
        - `a_grid`, `b_grid`: grilles de recherche (torch.Tensor ou list). Par défaut, on crée
        une grille raisonnable.
        - `plot`: si True, trace u vs sigma et sauvegarde un PNG.

        """
        
        sigma_raw = inputs[..., 0] if inputs.ndim > 1 else inputs
        sigma = self._decode_sigma(sigma_raw, from_logtis, None)

        with torch.no_grad():
            s = sigma.detach().flatten()
            # feature phi(sigma)
            x = torch.log(torch.clamp(s, min=self.eps)) if log_sigma else s

            # grilles par défaut (couvrent large, dont la zone "extrêmes")
            if a_grid is None:
                a_grid = torch.linspace(-6.0, 6.0, 25, device=s.device, dtype=s.dtype)
            else:
                a_grid = torch.as_tensor(a_grid, device=s.device, dtype=s.dtype)
            if b_grid is None:
                # on teste pentes positives & négatives
                b_grid = torch.cat([
                    torch.linspace(-8.0, -0.25, 16, device=s.device, dtype=s.dtype),
                    torch.linspace( 0.25,  8.0, 16, device=s.device, dtype=s.dtype)
                ])
            else:
                b_grid = torch.as_tensor(b_grid, device=s.device, dtype=s.dtype)

            calibration = {"a": None, "b": None, "calibration_score": -float("inf"),
                    "u_calibration": None, "preds": None}

            # recherche sur grille (robuste et simple)
            for a in a_grid:
                for b in b_grid:
                    u_vec = torch.sigmoid(a + b * x)              # (N,)
                    # prédictions discrètes via quantile tronqué
                    
                    preds = self.transform(
                        inputs=s,               # on passe sigma "déjà décodé"
                        dir_output=None,
                        from_logits=False,      # IMPORTANT
                        p_thresh=u_vec,                # vecteur par échantillon
                    )
                                        
                    score = float(score_fn(y_true, preds))
                    if score > calibration["calibration_score"]:
                        calibration.update(a=float(a.item()),
                                    b=float(b.item()),
                                    calibration_score=score,
                                    u_calibration=u_vec.clone(),
                                    preds=preds.clone())

            # tracé diagnostique
            if plot and dir_output is not None:
                dir_output = Path(dir_output)
                dir_output.mkdir(parents=True, exist_ok=True)

                # Scatter sigma vs u_calibration + courbe sigmoïde moyenne (sur l'axe x)
                # On ordonne pour une courbe lisible
                order = torch.argsort(s)
                s_ord = s[order].cpu().numpy()
                x_ord = x[order].cpu().numpy()
                u_ord = calibration["u_calibration"][order].cpu().numpy()
                
                plt.figure(figsize=(7,5))
                # points
                plt.scatter(s_ord, u_ord, s=10, alpha=0.5, label="u(sigma) calibrated")
                # courbe "théorique" sur un axe régulier (pour le visuel)
                xx = np.linspace(x.min().cpu(), x.max().cpu(), 400)
                uu = 1.0 / (1.0 + np.exp(-(calibration["a"] + calibration["b"] * xx)))
                # convertir xx -> axe sigma si log_sigma
                if log_sigma:
                    ss = np.exp(xx)
                else:
                    ss = xx
                plt.plot(ss, uu, lw=2, label=f"sigmoid(a + b·phi), a={calibration['a']:.3f}, b={calibration['b']:.3f}")
                plt.xscale("log" if log_sigma else "linear")
                plt.xlabel("sigma" + (" (log-scale axis)" if log_sigma else ""))
                plt.ylabel("u(sigma)")
                plt.title("Calibrated u(sigma) = sigmoid(a + b·phi(sigma))")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                plt.tight_layout()
                plt.savefig(dir_output / f"calibrated_u_sigmoid_.png", dpi=200)
                plt.close()

            return calibration
    
class PredictdEGPDLossTrunc(nn.Module):
    """
    Discrete exponentiated GPD (deGPD) truncated to {0,..., y_max}.
    - Forward: NLL on truncated PMF.
    - Transform: returns a discrete y selected by a PMF threshold (p_thresh) with tie policy.
    PMF (truncated):
        p_trunc(y) = [F_+(y+1) - F_+(y)] / F_+(y_max+1),  for y in {0,...,y_max}

    Inputs layout (last dim):
      - inputs[..., 0] : sigma (raw or decoded depending on from_logits)
      - inputs[..., 1] : kappa (raw if force_positive, decoded with softplus)
      - inputs[..., 2] : xi    (raw if force_positive, decoded with softplus)
    """

    def __init__(self,
                 eps: float = 1e-8,
                 reduction: str = "mean",
                 force_positive: bool = True,
                 y_max: int = 4):
        
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.y_max = int(y_max)

    # ---------- décodage de sigma (+ offset éventuel) ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, area=None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        return sigma

    # ---------- décodage/contrainte sur (kappa, xi) ----------
    def _decode_kappa_xi(self, kappa_raw: torch.Tensor, xi_raw: torch.Tensor):
        if self.force_positive:
            kappa = F.softplus(kappa_raw)
            xi    = F.softplus(xi_raw)
        else:
            kappa = kappa_raw
            xi    = xi_raw
        # bornes de sécurité numériques
        kappa = kappa.clamp_min(self.eps)
        xi    = xi.clamp(min=-1e6, max=1e6)
        return kappa, xi

    # ---------- CDF GPD H(y; sigma, xi) ----------
    def _gpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y     = torch.clamp(y, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)
        near0   = xi_safe.abs() < 1e-6
        z = 1.0 + xi_safe * (y / sigma)
        z = torch.clamp(z, min=self.eps)
        H_xi = 1.0 - torch.pow(z, -1.0 / xi_safe)   # xi != 0
        H_0  = 1.0 - torch.exp(-y / sigma)          # xi ~ 0
        H = torch.where(near0, H_0, H_xi)
        return torch.clamp(H, min=self.eps, max=1.0 - 1e-12)

    # ---------- eGPD CDF: F_+(y) = [H(y)]^kappa ----------
    def _egpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        H = self._gpd_cdf(y, sigma, xi)
        kappa_pos = torch.clamp(kappa, min=self.eps)
        Fp = torch.pow(H, kappa_pos)
        return torch.clamp(Fp, min=self.eps, max=1.0 - 1e-12)

    # ---------- PMF brute: p_raw(y) = F_+(y+1) - F_+(y) ----------
    def _pmf_raw(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y = y_int.to(dtype=sigma.dtype)
        F_y   = self._egpd_cdf(y,   sigma, kappa, xi)
        F_yp1 = self._egpd_cdf(y+1, sigma, kappa, xi)
        p_raw = torch.clamp(F_yp1 - F_y, min=self.eps)
        return p_raw

    # ---------- PMF TRONQUÉE sur {0,...,y_max} ----------
    def _pmf_trunc(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        p_raw = self._pmf_raw(y_int, sigma, kappa, xi)
        Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                               sigma, kappa, xi)  # normalisation: F_+(y_max+1)
        Z = torch.clamp(Z, min=self.eps)
        p = p_raw / Z
        return torch.clamp(p, min=self.eps, max=1.0)

    # ---------- NLL TRONQUÉE ----------
    def forward(
        self,
        inputs: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        from_logits: bool = True,
    ) -> torch.Tensor:
        """
        inputs shape: (..., 3) with [sigma, kappa, xi] on the last dim.
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw  = inputs[..., 0]
        kappa_raw  = inputs[..., 1]
        xi_raw     = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        y_int = y.to(torch.long).clamp(min=0, max=self.y_max)
        p_trunc = self._pmf_trunc(y_int, sigma, kappa, xi)

        nll = -torch.log(torch.clamp(p_trunc, min=self.eps))
        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(nll.device, nll.dtype)
            nll = nll * sample_weight
        if self.reduction == "mean":
            if sample_weight is not None:
                return nll.sum() / sample_weight.sum().clamp_min(self.eps)
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- Trouver y tel que PMF(y) >= p_thresh (min/max/all) ----------
    @torch.no_grad()
    def _y_where_p_ge(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                      p_thresh, pick):
        """
        Retourne:
          - si sigma scalaire: int y
          - si sigma batch:    tensor (N,) de y
        Politique si aucun y ne vérifie PMF>=p_thresh : repli sur argmax PMF.
        """
        if isinstance(p_thresh, torch.Tensor):
            assert torch.all(p_thresh >= 0.0) and torch.all(p_thresh <= 1.0), f"p_thresh must be in (0,1) got {p_thresh}"
        else:
            assert p_thresh >=0 and p_thresh <= 1.0, "p_thresh must be in (0,1)"
            
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        
        if sigma.ndim == 0:
            pmf = self._pmf_trunc(y_vals, sigma, kappa, xi)  # (y_max+1,)
            mask = (pmf >= p_thresh)
            if not mask.any():
                return int(torch.argmax(pmf).item())
            idxs = torch.nonzero(mask, as_tuple=False).flatten()
            return int((idxs.min() if pick == "min" else idxs.max()).item())

        # batch
        pmf = self._pmf_trunc(y_vals.unsqueeze(0).expand(sigma.shape[0], -1),
                              sigma.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1))  # (N, y_max+1)
        
        if isinstance(p_thresh, torch.Tensor):
            mask = (pmf >= p_thresh.view(-1,1))
        else:
            mask = (pmf >= p_thresh)
        y_index = torch.arange(self.y_max + 1, device=sigma.device).unsqueeze(0).expand_as(pmf)

        if pick == "min":
            large = torch.full_like(y_index, self.y_max + 1)
            cand = torch.where(mask, y_index, large)
            y_hat = cand.min(dim=1).values.clamp_max(self.y_max)
        else:  # "max"
            neg = torch.full_like(y_index, -1)
            cand = torch.where(mask, y_index, neg)
            y_hat = cand.max(dim=1).values.clamp_min(0)

        # repli pour les lignes sans True
        no_hit = ~mask.any(dim=1)
        if no_hit.any():
            fallback = pmf.argmax(dim=1)
            y_hat[no_hit] = fallback[no_hit]
        return y_hat

    # ---------- Transform: sélection par seuil de PMF + tracés optionnels ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output, p_thresh,
                  from_logits: bool = True,
                  pick: str = "max"):
        """
        inputs[...,0]=sigma, inputs[...,1]=kappa, inputs[...,2]=xi
        Retourne un y discret dans {0,...,y_max} tel que PMF(y) >= p_thresh,
        avec tie-breaking contrôlé par `pick` ("min"/"max").
        Repli: argmax PMF si aucun y ne dépasse le seuil.
        Si dir_output n'est pas None, trace PMF/CDF/PPF via plot_final().
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw = inputs[..., 0]
        kappa_raw = inputs[..., 1]
        xi_raw    = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        if isinstance(p_thresh, dict):
            # Exemple: p_thresh = {'a': a, 'b': b}
            # u = sigmoid(a + b * sigma) par défaut (ou log sigma si voulu avant l'appel)
            p_thresh = torch.sigmoid(p_thresh['a'] + sigma * p_thresh['b'])

        y_hat = self._y_where_p_ge(sigma, kappa, xi, p_thresh=p_thresh, pick=pick)
        
        if dir_output is not None:
            self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='mean')
            self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='max')
        return y_hat

    # ---------- Plots: PMF/CDF/PPF (tronqués) ----------
    @torch.no_grad()
    def plot_final(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                   dir_output, which_sigmas='max'):
        
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        sig = sigma.detach().flatten()
        kap = kappa.detach().flatten()
        xit = xi.detach().flatten()
        if sig.numel() == 0:
            return

        if which_sigmas == 'mean':
            sig_list = [(sig.mean().item(),
                        kap.mean().item(),
                        xit.mean().item())]

        elif which_sigmas == 'max':
            # max selon sigma ET kappa/xi correspondants (même index)
            idx_max = torch.argmax(sig)          # indice de l'échantillon avec sigma max
            s_max   = sig[idx_max].item()
            k_atmax = kap[idx_max].item()
            xi_atmax= xit[idx_max].item()
            sig_list = [(s_max, k_atmax, xi_atmax)]
            
        # 1) PMF tronquée
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            p_raw = self._pmf_raw(y_vals, s_t, k_t, x_t)
            Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                   s_t, k_t, x_t)
            p     = torch.clamp(p_raw / torch.clamp(Z, min=self.eps), min=self.eps)
            plt.stem(y_vals.cpu().numpy(), p.cpu().numpy(), label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated PMF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated PMF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_pmf = dir_output / f"degpd_trunc_pmf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_pmf, dpi=200); plt.close()

        # 2) CDF tronquée
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            F_y = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            Z   = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                 s_t, k_t, x_t)
            F_tr = torch.clamp(F_y / torch.clamp(Z, min=self.eps), min=self.eps, max=1.0)
            plt.step(y_vals.cpu().numpy(), F_tr.cpu().numpy(), where="post",
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated CDF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated CDF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_cdf = dir_output / f"degpd_trunc_cdf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_cdf, dpi=200); plt.close()

        # 3) PPF discrète via balayage de u ∈ (0,1) -> y(u)
        u_grid = torch.linspace(1e-6, 1.0 - 1e-6, 1000, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            Fmax = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                  s_t, k_t, x_t)
            u_eff = torch.clamp(u_grid * torch.clamp(Fmax, min=self.eps), min=self.eps, max=1.0 - 1e-12)
            F_tab = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            y_idx = torch.searchsorted(F_tab, u_eff, right=False).clamp(0, self.y_max)
            plt.plot(u_grid.cpu().numpy(), y_idx.cpu().numpy(),
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel(f"PPF (discrete y on {{0,...,{self.y_max}}})")
        plt.title("deGPD truncated PPF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_ppf = dir_output / f"degpd_trunc_ppf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_ppf, dpi=200); plt.close()
        
    def calibrate(self,
                  inputs: torch.Tensor,     # (..., 3) : sigma, kappa, xi
                  y_true : torch.Tensor,
                  score_fn,               # callable: (y_true, preds) -> float
                  dir_output,
                  log_sigma: bool = False,
                  from_logits : bool = True,
                  a_grid=None,
                  b_grid=None,
                  plot: bool = True):
        """
        Calibre u(x) = sigmoid(a + b * phi(sigma)) en maximisant `score_fn`.
        Ici kappa/xi sont par-échantillon, pris dans `inputs`.
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw = inputs[..., 0]
        kappa_raw = inputs[..., 1]
        xi_raw    = inputs[..., 2]

        # decode
        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        with torch.no_grad():
            s = sigma.detach().flatten()
            # feature phi(sigma)
            x = torch.log(torch.clamp(s, min=self.eps)) if log_sigma else s

            # grilles par défaut
            if a_grid is None:
                a_grid = torch.linspace(-6.0, 6.0, 25, device=s.device, dtype=s.dtype)
            else:
                a_grid = torch.as_tensor(a_grid, device=s.device, dtype=s.dtype)
            if b_grid is None:
                b_grid = torch.cat([
                    torch.linspace(-8.0, -0.25, 16, device=s.device, dtype=s.dtype),
                    torch.linspace( 0.25,  8.0, 16, device=s.device, dtype=s.dtype)
                ])
            else:
                b_grid = torch.as_tensor(b_grid, device=s.device, dtype=s.dtype)

            calibration = {"a": None, "b": None, "calibration_score": -float("inf"),
                           "u_calibration": None, "preds": None}

            # pour les appels à transform, on prépare les entrées décodées
            sigma_dec = s.view(sigma.shape)
            kappa_dec = kappa.detach()
            xi_dec    = xi.detach()

            for a in a_grid:
                for b in b_grid:
                    u_vec = torch.sigmoid(a + b * x).view_as(sigma_dec)  # (N,) -> shape inputs batch

                    # on passe inputs déjà décodés et from_logits=False
                    inputs_decoded = torch.stack([sigma_dec, kappa_dec, xi_dec], dim=-1)

                    preds = self.transform(
                        inputs=inputs_decoded,
                        dir_output=None,
                        from_logits=False,      # IMPORTANT: déjà décodé
                        p_thresh=u_vec,         # vecteur par échantillon
                    )
                                        
                    score = float(score_fn(y_true, preds))
                    if score > calibration["calibration_score"]:
                        calibration.update(a=float(a.item()),
                                           b=float(b.item()),
                                           calibration_score=score,
                                           u_calibration=u_vec.clone(),
                                           preds=preds.clone())

            # tracé diagnostique
            if plot and dir_output is not None:
                dir_output = Path(dir_output)
                dir_output.mkdir(parents=True, exist_ok=True)

                order = torch.argsort(s)
                s_ord = s[order].cpu().numpy()
                x_ord = x[order].cpu().numpy()
                u_ord = calibration["u_calibration"].detach().flatten()[order].cpu().numpy()
                
                plt.figure(figsize=(7,5))
                plt.scatter(s_ord, u_ord, s=10, alpha=0.5, label="u(sigma) calibrated")
                xx = np.linspace(x.min().cpu(), x.max().cpu(), 400)
                uu = 1.0 / (1.0 + np.exp(-(calibration["a"] + calibration["b"] * xx)))
                # convertir xx -> axe sigma si log_sigma
                if log_sigma:
                    ss = np.exp(xx)
                else:
                    ss = xx
                plt.plot(ss, uu, lw=2, label=f"sigmoid(a + b·phi), a={calibration['a']:.3f}, b={calibration['b']:.3f}")
                plt.xscale("log" if log_sigma else "linear")
                plt.xlabel("sigma" + (" (log-scale axis)" if log_sigma else ""))
                plt.ylabel("u(sigma)")
                plt.title("Calibrated u(sigma) = sigmoid(a + b·phi(sigma))")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                plt.tight_layout()
                plt.savefig(dir_output / f"calibrated_u_sigmoid_.png", dpi=200)
                plt.close()
                
            return calibration
        
class PredictdEGPDLossTruncMostProbable(nn.Module):
    """
    Discrete exponentiated GPD (deGPD) truncated to {0,..., y_max}.
    - Forward: NLL on truncated PMF.
    - Transform: returns a discrete y selected by a PMF threshold (p_thresh) with tie policy.
    PMF (truncated):
        p_trunc(y) = [F_+(y+1) - F_+(y)] / F_+(y_max+1),  for y in {0,...,y_max}

    Inputs layout (last dim):
      - inputs[..., 0] : sigma (raw or decoded depending on from_logits)
      - inputs[..., 1] : kappa (raw if force_positive, decoded with softplus)
      - inputs[..., 2] : xi    (raw if force_positive, decoded with softplus)
    """

    def __init__(self,
                 eps: float = 1e-8,
                 reduction: str = "mean",
                 force_positive: bool = True,
                 y_max: int = 4):
        
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.y_max = int(y_max)

    # ---------- décodage de sigma (+ offset éventuel) ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, area=None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        return sigma

    # ---------- décodage/contrainte sur (kappa, xi) ----------
    def _decode_kappa_xi(self, kappa_raw: torch.Tensor, xi_raw: torch.Tensor):
        if self.force_positive:
            kappa = F.softplus(kappa_raw)
            xi    = F.softplus(xi_raw)
        else:
            kappa = kappa_raw
            xi    = xi_raw
        # bornes de sécurité numériques
        kappa = kappa.clamp_min(self.eps)
        xi    = xi.clamp(self.eps)
        return kappa, xi

    # ---------- CDF GPD H(y; sigma, xi) ----------
    def _gpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y     = torch.clamp(y, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)
        near0   = xi_safe.abs() < 1e-6
        z = 1.0 + xi_safe * (y / sigma)
        z = torch.clamp(z, min=self.eps)
        H_xi = 1.0 - torch.pow(z, -1.0 / xi_safe)   # xi != 0
        H_0  = 1.0 - torch.exp(-y / sigma)          # xi ~ 0
        H = torch.where(near0, H_0, H_xi)
        return torch.clamp(H, min=self.eps, max=1.0 - 1e-12)

    # ---------- eGPD CDF: F_+(y) = [H(y)]^kappa ----------
    def _egpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        H = self._gpd_cdf(y, sigma, xi)
        kappa_pos = torch.clamp(kappa, min=self.eps)
        Fp = torch.pow(H, kappa_pos)
        return torch.clamp(Fp, min=self.eps, max=1.0 - 1e-12)

    # ---------- PMF brute: p_raw(y) = F_+(y+1) - F_+(y) ----------
    def _pmf_raw(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y = y_int.to(dtype=sigma.dtype)
        F_y   = self._egpd_cdf(y,   sigma, kappa, xi)
        F_yp1 = self._egpd_cdf(y+1, sigma, kappa, xi)
        p_raw = torch.clamp(F_yp1 - F_y, min=self.eps)
        return p_raw

    # ---------- PMF TRONQUÉE sur {0,...,y_max} ----------
    def _pmf_trunc(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        p_raw = self._pmf_raw(y_int, sigma, kappa, xi)
        Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                               sigma, kappa, xi)  # normalisation: F_+(y_max+1)
        Z = torch.clamp(Z, min=self.eps)
        p = p_raw / Z
        return torch.clamp(p, min=self.eps, max=1.0)

    # ---------- NLL TRONQUÉE ----------
    def forward(self, inputs: torch.Tensor, y: torch.Tensor,
                weight: torch.Tensor = None, from_logits: bool = True, sample_weight=None) -> torch.Tensor:
        """
        inputs shape: (..., 3) with [sigma, kappa, xi] on the last dim.
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw  = inputs[..., 0]
        kappa_raw  = inputs[..., 1]
        xi_raw     = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)
        
        #print('s', torch.unique(sigma))
        #print('k', torch.unique(kappa))
        #print('xi', torch.unique(xi))

        y_int = y.to(torch.long).clamp(min=0, max=self.y_max)
        p_trunc = self._pmf_trunc(y_int, sigma, kappa, xi)

        nll = -torch.log(torch.clamp(p_trunc, min=self.eps))
        if weight is not None:
            nll = nll * weight
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- Trouver y le plus probable : argmax PMF ----------
    @torch.no_grad()
    def _y_where_p_ge(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor):
        """
        Retourne :
        - si sigma scalaire : int y (classe avec PMF maximale)
        - si sigma batch    : tensor (N,) de y
        """
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)

        if sigma.ndim == 0:
            pmf = self._pmf_trunc(y_vals, sigma, kappa, xi)  # (y_max+1,)
            return int(pmf.argmax().item())

        # batch
        pmf = self._pmf_trunc(
            y_vals.unsqueeze(0).expand(sigma.shape[0], -1),
            sigma.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1)
        )  # (N, y_max+1)

        y_hat = pmf.argmax(dim=1)
        return y_hat

    @torch.no_grad()
    def _y(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor):
        """
        Retourne :
        - si sigma scalaire : int y (classe avec PMF maximale)
        - si sigma batch    : tensor (N,) de y
        """
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)

        if sigma.ndim == 0:
            pmf = self._pmf_trunc(y_vals, sigma, kappa, xi)  # (y_max+1,)
            return int(pmf.argmax().item())

        # batch
        pmf = self._pmf_trunc(
            y_vals.unsqueeze(0).expand(sigma.shape[0], -1),
            sigma.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1)
        )  # (N, y_max+1)
        return pmf

    # ---------- Transform: sélection par seuil de PMF + tracés optionnels ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output, prediction_type,
                  from_logits: bool = True):
        """
        inputs[...,0]=sigma, inputs[...,1]=kappa, inputs[...,2]=xi
        Retourne un y discret dans {0,...,y_max} tel que PMF(y) >= p_thresh,
        avec tie-breaking contrôlé par `pick` ("min"/"max").
        Repli: argmax PMF si aucun y ne dépasse le seuil.
        Si dir_output n'est pas None, trace PMF/CDF/PPF via plot_final().
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw = inputs[..., 0]
        kappa_raw = inputs[..., 1]
        xi_raw    = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        if prediction_type == 'Class':
            y_hat = self._y_where_p_ge(sigma, kappa, xi)
        else:
            y_hat = self._y(sigma, kappa, xi)
            
        if dir_output is not None:
            self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='mean')
            self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='max')
        return y_hat

    # ---------- Plots: PMF/CDF/PPF (tronqués) ----------
    @torch.no_grad()
    def plot_final(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                   dir_output, which_sigmas='max'):
        
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        sig = sigma.detach().flatten()
        kap = kappa.detach().flatten()
        xit = xi.detach().flatten()
        if sig.numel() == 0:
            return

        if which_sigmas == 'mean':
            sig_list = [(sig.mean().item(),
                        kap.mean().item(),
                        xit.mean().item())]

        elif which_sigmas == 'max':
            # max selon sigma ET kappa/xi correspondants (même index)
            idx_max = torch.argmax(sig)          # indice de l'échantillon avec sigma max
            s_max   = sig[idx_max].item()
            k_atmax = kap[idx_max].item()
            xi_atmax= xit[idx_max].item()
            sig_list = [(s_max, k_atmax, xi_atmax)]
            
        # 1) PMF tronquée
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            p_raw = self._pmf_raw(y_vals, s_t, k_t, x_t)
            Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                   s_t, k_t, x_t)
            p     = torch.clamp(p_raw / torch.clamp(Z, min=self.eps), min=self.eps)
            plt.stem(y_vals.cpu().numpy(), p.cpu().numpy(), label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated PMF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated PMF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_pmf = dir_output / f"degpd_trunc_pmf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_pmf, dpi=200); plt.close()

        # 2) CDF tronquée
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            F_y = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            Z   = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                 s_t, k_t, x_t)
            F_tr = torch.clamp(F_y / torch.clamp(Z, min=self.eps), min=self.eps, max=1.0)
            plt.step(y_vals.cpu().numpy(), F_tr.cpu().numpy(), where="post",
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated CDF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated CDF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_cdf = dir_output / f"degpd_trunc_cdf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_cdf, dpi=200); plt.close()

        # 3) PPF discrète via balayage de u ∈ (0,1) -> y(u)
        u_grid = torch.linspace(1e-6, 1.0 - 1e-6, 1000, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            Fmax = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                  s_t, k_t, x_t)
            u_eff = torch.clamp(u_grid * torch.clamp(Fmax, min=self.eps), min=self.eps, max=1.0 - 1e-12)
            F_tab = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            y_idx = torch.searchsorted(F_tab, u_eff, right=False).clamp(0, self.y_max)
            plt.plot(u_grid.cpu().numpy(), y_idx.cpu().numpy(),
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel(f"PPF (discrete y on {{0,...,{self.y_max}}})")
        plt.title("deGPD truncated PPF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_ppf = dir_output / f"degpd_trunc_ppf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_ppf, dpi=200); plt.close()
        
class PredictdBulkTailEGPD(nn.Module):
    """
    Bulk + Tail discret:
      - Bulk: Normal (mu, sigma_b) discrétisée par différence de CDF (Phi(y+1)-Phi(y))
              et tronquée sur {0,...,tau}.
      - Tail: eGPD (sigma_t, kappa, xi) sur les excès (y - tau), discrétisée
              via F_+(e+1) - F_+(e) et tronquée sur {tau+1,...,y_max}.
      - Mélange: p(y) = (1-rho)*p_bulk(y)  si y <= tau
                 p(y) = rho     *p_tail(y)  si y >= tau+1
    Entrées (dernière dimension):
      inputs[..., 0] : sigma_t (tail)   (raw si from_logits, sinon déjà décodé)
      inputs[..., 1] : kappa    (tail)  (raw si force_positive)
      inputs[..., 2] : xi       (tail)  (raw si force_positive)
      inputs[..., 3] : mu_b     (bulk Normal, réel)
      inputs[..., 4] : sigma_b  (bulk Normal, >0 ; raw si from_logits)
      inputs[..., 5] : rho      (poids tail, dans (0,1) via sigmoid si from_logits)

    Args:
      tau (int): seuil séparant bulk (<= tau) et tail (>= tau+1).
      y_max (int): support max du compte discret.
      force_positive (bool): applique softplus à (kappa, xi) si True.
      reduction: "mean" | "sum" | "none"
      eps: petite constante de stabilité.
    """
    def __init__(self,
                 tau: int = 5,
                 y_max: int = 50,
                 force_positive: bool = True,
                 reduction: str = "mean",
                 eps: float = 1e-8):
        super().__init__()
        assert 0 <= tau < y_max, "Exiger 0 <= tau < y_max"
        self.tau = int(tau)
        self.y_max = int(y_max)
        self.force_positive = force_positive
        self.reduction = reduction
        self.eps = eps

    # ---------- décodages ----------
    def _decode_pos(self, t: torch.Tensor, from_logits: bool):
        t = F.softplus(t) if from_logits else t
        return t.clamp_min(self.eps)

    def _decode_tail_params(self, kappa_raw: torch.Tensor, xi_raw: torch.Tensor):
        if self.force_positive:
            kappa = F.softplus(kappa_raw)
            xi    = F.softplus(xi_raw)
        else:
            kappa = kappa_raw
            xi    = xi_raw
        return kappa.clamp_min(self.eps), xi.clamp_min(self.eps)

    def _decode_rho(self, rho_raw: torch.Tensor, from_logits: bool):
        if from_logits:
            rho = torch.sigmoid(rho_raw)
        else:
            rho = rho_raw
        # Serrer dans (eps, 1-eps) pour éviter masses nulles
        return rho.clamp(self.eps, 1.0 - self.eps)

    # ---------- Normal standard CDF ----------
    @staticmethod
    def _phi(x: torch.Tensor):
        # CDF de la N(0,1) via erf
        return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))

    # ---------- PMF bulk: Normal discrétisée et tronquée à {0,...,tau} ----------
    def _pmf_bulk_trunc(self, y_int: torch.Tensor, mu: torch.Tensor, sig_b: torch.Tensor) -> torch.Tensor:
        # PMF(y) = Phi((y+1-mu)/sig_b) - Phi((y-mu)/sig_b), renormalisée sur [0, tau]
        y = y_int.to(dtype=mu.dtype)

        # Z_b = Phi((tau+1 - mu)/sig_b) - Phi((0 - mu)/sig_b)
        z_hi = (self.tau + 1 - mu) / sig_b
        z_lo = (0.0 - mu) / sig_b
        Zb = (self._phi(z_hi) - self._phi(z_lo)).clamp_min(self.eps)

        z1 = (y + 1 - mu) / sig_b
        z0 = (y - mu) / sig_b
        p_raw = (self._phi(z1) - self._phi(z0)).clamp_min(self.eps)

        p = p_raw / Zb
        # Mettre à 0 hors support bulk
        mask = (y_int <= self.tau).to(p.dtype)
        return (p * mask).clamp_min(self.eps)

    # ---------- CDF GPD ----------
    def _gpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(y, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        xi_safe = xi.clamp(min=-1e6, max=1e6)
        near0 = xi_safe.abs() < 1e-6
        z = (1.0 + xi_safe * (y / sigma)).clamp_min(self.eps)
        H_xi = 1.0 - torch.pow(z, -1.0 / xi_safe)
        H_0  = 1.0 - torch.exp(-y / sigma)
        H = torch.where(near0, H_0, H_xi)
        return H.clamp(self.eps, 1.0 - 1e-12)

    # ---------- eGPD CDF: H^kappa ----------
    def _egpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        H = self._gpd_cdf(y, sigma, xi)
        kappa_pos = kappa.clamp_min(self.eps)
        Fp = torch.pow(H, kappa_pos)
        return Fp.clamp(self.eps, 1.0 - 1e-12)

    # ---------- PMF tail: eGPD sur excès e=y-(tau+1)+1 with support {tau+1,...,y_max} ----------
    def _pmf_tail_trunc(self, y_int: torch.Tensor, sigma_t: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        # Excès e = y - (tau+1)  (donc e ∈ {0,..., y_max-(tau+1)})
        e = (y_int - (self.tau + 1)).to(dtype=sigma_t.dtype)
        e = torch.clamp(e, min=0.0)

        # p_raw_tail(y) = F_+(e+1) - F_+(e)
        F_e   = self._egpd_cdf(e,   sigma_t, kappa, xi)
        F_ep1 = self._egpd_cdf(e+1, sigma_t, kappa, xi)
        p_raw = (F_ep1 - F_e).clamp_min(self.eps)
        
        # Normalisation sur {tau+1,...,y_max}  <=>  e ∈ {0,..., y_max-(tau+1)}
        Emax = self._egpd_cdf(torch.tensor(self.y_max - (self.tau + 1) + 1,
                                           dtype=sigma_t.dtype, device=sigma_t.device),
                              sigma_t, kappa, xi).clamp_min(self.eps)
        p = p_raw / Emax

        # Mettre à 0 hors support tail
        mask = (y_int >= self.tau + 1).to(p.dtype)
        return (p * mask).clamp_min(self.eps)

    # ---------- PMF totale ----------
    def _pmf_mixture(self, y_int: torch.Tensor,
                     sigma_t: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                     mu_b: torch.Tensor, sigma_b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        p_b = self._pmf_bulk_trunc(y_int, mu_b, sigma_b)       # support <= tau
        p_t = self._pmf_tail_trunc(y_int, sigma_t, kappa, xi)  # support >= tau+1
        # Mélange
        return ((1.0 - rho) * p_b + rho * p_t).clamp_min(self.eps)

    # ---------- NLL ----------
    def forward(self, inputs: torch.Tensor, y: torch.Tensor,
                weight: torch.Tensor = None, from_logits: bool = True) -> torch.Tensor:
        if inputs.size(-1) < 6:
            raise ValueError("inputs must have 6 channels: [sigma_t, kappa, xi, mu_b, sigma_b, rho]")

        sigma_t_raw = inputs[..., 0]
        kappa_raw   = inputs[..., 1]
        xi_raw      = inputs[..., 2]
        mu_b        = inputs[..., 3]
        sigma_b_raw = inputs[..., 4]
        rho_raw     = inputs[..., 5]

        sigma_t = self._decode_pos(sigma_t_raw, from_logits)
        kappa, xi = self._decode_tail_params(kappa_raw, xi_raw)
        sigma_b = self._decode_pos(sigma_b_raw, from_logits)
        rho = self._decode_rho(rho_raw, from_logits)

        y_int = y.to(torch.long).clamp(min=0, max=self.y_max)
        p = self._pmf_mixture(y_int, sigma_t, kappa, xi, mu_b, sigma_b, rho)

        nll = -torch.log(p.clamp_min(self.eps))
        if weight is not None:
            nll = nll * weight
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- Argmax PMF / PMF row ----------
    @torch.no_grad()
    def _y_argmax(self, sigma_t, kappa, xi, mu_b, sigma_b, rho):
        y_vals = torch.arange(0, self.y_max + 1, device=sigma_t.device)

        if sigma_t.ndim == 0:
            pmf = self._pmf_mixture(y_vals, sigma_t, kappa, xi, mu_b, sigma_b, rho)
            return int(pmf.argmax().item())

        pmf = self._pmf_mixture(
            y_vals.unsqueeze(0).expand(sigma_t.shape[0], -1),
            sigma_t.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1),
            mu_b.unsqueeze(-1), sigma_b.unsqueeze(-1), rho.unsqueeze(-1)
        )
        return pmf.argmax(dim=1)

    @torch.no_grad()
    def _pmf_row(self, sigma_t, kappa, xi, mu_b, sigma_b, rho):
        y_vals = torch.arange(0, self.y_max + 1, device=sigma_t.device)

        if sigma_t.ndim == 0:
            return self._pmf_mixture(y_vals, sigma_t, kappa, xi, mu_b, sigma_b, rho)

        return self._pmf_mixture(
            y_vals.unsqueeze(0).expand(sigma_t.shape[0], -1),
            sigma_t.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1),
            mu_b.unsqueeze(-1), sigma_b.unsqueeze(-1), rho.unsqueeze(-1)
        )

    # ---------- Transform ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output=None, prediction_type='Class',
                  from_logits: bool = True):
        if inputs.size(-1) < 6:
            raise ValueError("inputs must have 6 channels: [sigma_t, kappa, xi, mu_b, sigma_b, rho]")

        sigma_t_raw = inputs[..., 0]
        kappa_raw   = inputs[..., 1]
        xi_raw      = inputs[..., 2]
        mu_b        = inputs[..., 3]
        sigma_b_raw = inputs[..., 4]
        rho_raw     = inputs[..., 5]

        sigma_t = self._decode_pos(sigma_t_raw, from_logits)
        kappa, xi = self._decode_tail_params(kappa_raw, xi_raw)
        sigma_b = self._decode_pos(sigma_b_raw, from_logits)
        rho = self._decode_rho(rho_raw, from_logits)

        if prediction_type == 'Class':
            y_hat = self._y_argmax(sigma_t, kappa, xi, mu_b, sigma_b, rho)
        else:
            y_hat = self._pmf_row(sigma_t, kappa, xi, mu_b, sigma_b, rho)

        if dir_output is not None:
            self.plot_final(sigma_t, kappa, xi, mu_b, sigma_b, rho, dir_output, which='mean')
            self.plot_final(sigma_t, kappa, xi, mu_b, sigma_b, rho, dir_output, which='max')
        return y_hat

    # ---------- Plots ----------
    @torch.no_grad()
    def plot_final(self, sigma_t, kappa, xi, mu_b, sigma_b, rho, dir_output, which='max'):
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        # Aplatir pour extraire (mean) ou (max sigma_t)
        st, kp, xw = sigma_t.detach().flatten(), kappa.detach().flatten(), xi.detach().flatten()
        mu, sb, rh = mu_b.detach().flatten(), sigma_b.detach().flatten(), rho.detach().flatten()
        if st.numel() == 0:
            return

        if which == 'mean':
            params = [(st.mean().item(), kp.mean().item(), xw.mean().item(),
                       mu.mean().item(), sb.mean().item(), rh.mean().item())]
        else:
            idx = torch.argmax(st)
            params = [(float(st[idx]), float(kp[idx]), float(xw[idx]),
                       float(mu[idx]), float(sb[idx]), float(rh[idx]))]

        y_vals = torch.arange(0, self.y_max + 1, device=sigma_t.device)

        # PMF totale + composants
        plt.figure(figsize=(7,5))
        for s_t, k_t, xi_t, mu_t, s_b, r_t in params:
            s_t = torch.tensor(s_t, device=sigma_t.device, dtype=sigma_t.dtype)
            k_t = torch.tensor(k_t, device=sigma_t.device, dtype=sigma_t.dtype)
            xi_t= torch.tensor(xi_t, device=sigma_t.device, dtype=sigma_t.dtype)
            mu_t= torch.tensor(mu_t, device=sigma_t.device, dtype=sigma_t.dtype)
            s_b = torch.tensor(s_b, device=sigma_t.device, dtype=sigma_t.dtype)
            r_t = torch.tensor(r_t, device=sigma_t.device, dtype=sigma_t.dtype)

            p_b = self._pmf_bulk_trunc(y_vals, mu_t, s_b)
            p_t = self._pmf_tail_trunc(y_vals, s_t, k_t, xi_t)
            p   = (1-r_t)*p_b + r_t*p_t

            plt.stem(y_vals.cpu().numpy(), p.cpu().numpy(), linefmt='-', markerfmt='o', basefmt=" ", label="Mixture PMF")
            plt.stem(y_vals.cpu().numpy(), ((1-r_t)*p_b).cpu().numpy(), linefmt='-', markerfmt='x', basefmt=" ", label=f"Bulk (1-rho)")
            plt.stem(y_vals.cpu().numpy(), (r_t*p_t).cpu().numpy(), linefmt='-', markerfmt='^', basefmt=" ", label=f"Tail (rho)")
        plt.axvline(self.tau + 0.5, linestyle="--", alpha=0.5, label=f"tau={self.tau}")
        plt.xlabel("y"); plt.ylabel("PMF")
        plt.title(f"Bulk (Normal) + Tail (eGPD) on {{0,...,{self.y_max}}}")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_p = Path(dir_output) / f"bulk_tail_degpd_pmf_{which}.png"
        plt.tight_layout(); plt.savefig(f_p, dpi=200); plt.close()
        
class PredictdEGPDLossTrunc2(nn.Module):
    """
    Discrete exponentiated GPD (deGPD) truncated to {0,..., y_max}.
    - Forward: NLL on truncated PMF.
    - Transform: returns a discrete y selected by a PMF threshold (p_thresh) with tie policy.
    PMF (truncated):
        p_trunc(y) = [F_+(y+1) - F_+(y)] / F_+(y_max+1),  for y in {0,...,y_max}

    Inputs layout (last dim):
      - inputs[..., 0] : sigma (raw or decoded depending on from_logits)
      - inputs[..., 1] : kappa (raw if force_positive, decoded with softplus)
      - inputs[..., 2] : xi    (raw if force_positive, decoded with softplus)
    """

    def __init__(self,
                 eps: float = 1e-8,
                 reduction: str = "mean",
                 force_positive: bool = True,
                 y_max: int = 4):
        
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.y_max = int(y_max)

    # ---------- décodage de sigma (+ offset éventuel) ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, area=None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        # pas d'offset ici (comptes), gardé pour compat API
        return sigma

    # ---------- décodage/contrainte sur (kappa, xi) ----------
    def _decode_kappa_xi(self, kappa_raw: torch.Tensor, xi_raw: torch.Tensor):
        if self.force_positive:
            kappa = F.softplus(kappa_raw)
            xi    = F.softplus(xi_raw)
        else:
            kappa = kappa_raw
            xi    = xi_raw
        # bornes de sécurité numériques
        kappa = kappa.clamp_min(self.eps)
        xi    = xi.clamp(min=-1e6, max=1e6)
        return kappa, xi

    # ---------- CDF GPD H(y; sigma, xi) ----------
    def _gpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y     = torch.clamp(y, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)
        near0   = xi_safe.abs() < 1e-6
        z = 1.0 + xi_safe * (y / sigma)
        z = torch.clamp(z, min=self.eps)
        H_xi = 1.0 - torch.pow(z, -1.0 / xi_safe)   # xi != 0
        H_0  = 1.0 - torch.exp(-y / sigma)          # xi ~ 0
        H = torch.where(near0, H_0, H_xi)
        return torch.clamp(H, min=self.eps, max=1.0 - 1e-12)

    # ---------- eGPD CDF: F_+(y) = [H(y)]^kappa ----------
    def _egpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        H = self._gpd_cdf(y, sigma, xi)
        kappa_pos = torch.clamp(kappa, min=self.eps)
        Fp = torch.pow(H, kappa_pos)
        return torch.clamp(Fp, min=self.eps, max=1.0 - 1e-12)

    # ---------- PMF brute: p_raw(y) = F_+(y+1) - F_+(y) ----------
    def _pmf_raw(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y = y_int.to(dtype=sigma.dtype)
        F_y   = self._egpd_cdf(y,   sigma, kappa, xi)
        F_yp1 = self._egpd_cdf(y+1, sigma, kappa, xi)
        p_raw = torch.clamp(F_yp1 - F_y, min=self.eps)
        return p_raw

    # ---------- PMF TRONQUÉE sur {0,...,y_max} ----------
    def _pmf_trunc(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        p_raw = self._pmf_raw(y_int, sigma, kappa, xi)
        Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                               sigma, kappa, xi)  # normalisation: F_+(y_max+1)
        Z = torch.clamp(Z, min=self.eps)
        p = p_raw / Z
        return torch.clamp(p, min=self.eps, max=1.0)

    # ---------- NLL TRONQUÉE ----------
    def forward(self, inputs: torch.Tensor, y: torch.Tensor,
                weight: torch.Tensor = None, from_logits: bool = True) -> torch.Tensor:
        """
        inputs shape: (..., 3) with [sigma, kappa, xi] on the last dim.
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw  = inputs[..., 0]
        kappa_raw  = inputs[..., 1]
        xi_raw     = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        y_int = y.to(torch.long).clamp(min=0, max=self.y_max)
        p_trunc = self._pmf_trunc(y_int, sigma, kappa, xi)

        nll = -torch.log(torch.clamp(p_trunc, min=self.eps))
        if weight is not None:
            nll = nll * weight
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- Trouver y tel que PMF(y) >= p_thresh (min/max/all) ----------
    @torch.no_grad()
    def _y_where_p_ge(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                      p_thresh, pick):
        """
        Retourne:
          - si sigma scalaire: int y
          - si sigma batch:    tensor (N,) de y
        Politique si aucun y ne vérifie PMF>=p_thresh : repli sur argmax PMF.
        """
        if isinstance(p_thresh, torch.Tensor):
            assert torch.all(p_thresh >= 0.0) and torch.all(p_thresh <= 1.0), "p_thresh must be in (0,1)"
        else:
            assert p_thresh >=0 and p_thresh <= 1.0, "p_thresh must be in (0,1)"
            
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        
        if sigma.ndim == 0:
            pmf = self._pmf_trunc(y_vals, sigma, kappa, xi)  # (y_max+1,)
            mask = (pmf >= p_thresh)
            if not mask.any():
                return int(torch.argmax(pmf).item())
            idxs = torch.nonzero(mask, as_tuple=False).flatten()
            return int((idxs.min() if pick == "min" else idxs.max()).item())

        # batch
        pmf = self._pmf_trunc(y_vals.unsqueeze(0).expand(sigma.shape[0], -1),
                              sigma.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1))  # (N, y_max+1)
        
        if isinstance(p_thresh, torch.Tensor):
            mask = (pmf >= p_thresh.view(-1,1))
        else:
            mask = (pmf >= p_thresh)
        y_index = torch.arange(self.y_max + 1, device=sigma.device).unsqueeze(0).expand_as(pmf)

        if pick == "min":
            large = torch.full_like(y_index, self.y_max + 1)
            cand = torch.where(mask, y_index, large)
            y_hat = cand.min(dim=1).values.clamp_max(self.y_max)
        else:  # "max"
            neg = torch.full_like(y_index, -1)
            cand = torch.where(mask, y_index, neg)
            y_hat = cand.max(dim=1).values.clamp_min(0)

        # repli pour les lignes sans True
        no_hit = ~mask.any(dim=1)
        if no_hit.any():
            fallback = pmf.argmax(dim=1)
            y_hat[no_hit] = fallback[no_hit]
        return y_hat

    # ---------- Transform: sélection par seuil de PMF + tracés optionnels ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output, p_thresh,
                  from_logits: bool = True,
                  pick: str = "max"):
        """
        inputs[...,0]=sigma, inputs[...,1]=kappa, inputs[...,2]=xi
        Retourne un y discret dans {0,...,y_max} tel que PMF(y) >= p_thresh,
        avec tie-breaking contrôlé par `pick` ("min"/"max").
        Repli: argmax PMF si aucun y ne dépasse le seuil.
        Si dir_output n'est pas None, trace PMF/CDF/PPF via plot_final().
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw = inputs[..., 0]
        kappa_raw = inputs[..., 1]
        xi_raw    = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        if isinstance(p_thresh, dict):
            # Exemple: p_thresh = {'a': a, 'b': b}
            # u = sigmoid(a + b * sigma) par défaut (ou log sigma si voulu avant l'appel)
            l = p_thresh['a'] + p_thresh['b_s']* sigma + p_thresh['b_k']*kappa + p_thresh['b_x']*xi
            p_thresh = torch.sigmoid(l)

        y_hat = self._y_where_p_ge(sigma, kappa, xi, p_thresh=p_thresh, pick=pick)
        
        if dir_output is not None:
            self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='mean')
            self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='max')
        return y_hat

    # ---------- Plots: PMF/CDF/PPF (tronqués) ----------
    @torch.no_grad()
    def plot_final(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                   dir_output, which_sigmas='max'):
        
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        sig = sigma.detach().flatten()
        kap = kappa.detach().flatten()
        xit = xi.detach().flatten()
        if sig.numel() == 0:
            return

        if which_sigmas == 'mean':
            sig_list = [(sig.mean().item(),
                        kap.mean().item(),
                        xit.mean().item())]

        elif which_sigmas == 'max':
            # max selon sigma ET kappa/xi correspondants (même index)
            idx_max = torch.argmax(sig)          # indice de l'échantillon avec sigma max
            s_max   = sig[idx_max].item()
            k_atmax = kap[idx_max].item()
            xi_atmax= xit[idx_max].item()
            sig_list = [(s_max, k_atmax, xi_atmax)]
            
        # 1) PMF tronquée
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            p_raw = self._pmf_raw(y_vals, s_t, k_t, x_t)
            Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                   s_t, k_t, x_t)
            p     = torch.clamp(p_raw / torch.clamp(Z, min=self.eps), min=self.eps)
            plt.stem(y_vals.cpu().numpy(), p.cpu().numpy(), label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated PMF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated PMF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_pmf = dir_output / f"degpd_trunc_pmf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_pmf, dpi=200); plt.close()

        # 2) CDF tronquée
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            F_y = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            Z   = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                 s_t, k_t, x_t)
            F_tr = torch.clamp(F_y / torch.clamp(Z, min=self.eps), min=self.eps, max=1.0)
            plt.step(y_vals.cpu().numpy(), F_tr.cpu().numpy(), where="post",
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated CDF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated CDF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_cdf = dir_output / f"degpd_trunc_cdf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_cdf, dpi=200); plt.close()

        # 3) PPF discrète via balayage de u ∈ (0,1) -> y(u)
        u_grid = torch.linspace(1e-6, 1.0 - 1e-6, 1000, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            Fmax = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                  s_t, k_t, x_t)
            u_eff = torch.clamp(u_grid * torch.clamp(Fmax, min=self.eps), min=self.eps, max=1.0 - 1e-12)
            F_tab = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            y_idx = torch.searchsorted(F_tab, u_eff, right=False).clamp(0, self.y_max)
            plt.plot(u_grid.cpu().numpy(), y_idx.cpu().numpy(),
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel(f"PPF (discrete y on {{0,...,{self.y_max}}})")
        plt.title("deGPD truncated PPF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_ppf = dir_output / f"degpd_trunc_ppf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_ppf, dpi=200); plt.close()
        
    def calibrate(self,
              inputs: torch.Tensor,     # (..., 3) : sigma, kappa, xi (bruts ou logits)
              y_true : torch.Tensor,
              score_fn,                 # callable: (y_true, preds) -> float
              dir_output,
              log_sigma: bool = False,
              log_kappa: bool = False,
              log_xi: bool = False,
              from_logits : bool = True,
              a_grid=None,
              b_grid=None,
              b_s_grid=None,
              b_k_grid=None,
              b_x_grid=None,
              plot: bool = True):
        """
        Calibrates u(x) = sigmoid(a + b_s * phi_s(sigma) + b_k * phi_k(kappa) + b_x * phi_x(xi))
        by maximizing `score_fn`. (force_positive=True attendu pour kappa, xi).
        Uses pick="max" through self.transform (fallback = argmax PMF).
        """
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt

        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        # --- decode (respecte votre implémentation et force_positive=True) ---
        sigma_raw = inputs[..., 0]
        kappa_raw = inputs[..., 1]
        xi_raw    = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        eps = float(self.eps)

        with torch.no_grad():
            # Aplatissement (on garde le shape batch pour transform après)
            s = sigma.detach().flatten()
            k = kappa.detach().flatten()
            x = xi.detach().flatten()

            # Features phi_s / phi_k / phi_x
            phi_s = torch.log(torch.clamp(s, min=eps)) if log_sigma else s
            phi_k = torch.log(torch.clamp(k, min=eps)) if log_kappa else k
            phi_x = torch.log(torch.clamp(x, min=eps)) if log_xi    else x

            # Grilles par défaut
            if a_grid is None:
                a_grid = torch.linspace(-6.0, 6.0, 25, device=s.device, dtype=s.dtype)
            else:
                a_grid = torch.as_tensor(a_grid, device=s.device, dtype=s.dtype)

            if b_grid is None:
                b_grid = torch.cat([
                    torch.linspace(-8.0, -0.25, 16, device=s.device, dtype=s.dtype),
                    torch.linspace( 0.25,  8.0, 16, device=s.device, dtype=s.dtype)
                ])
            else:
                b_grid = torch.as_tensor(b_grid, device=s.device, dtype=s.dtype)

            # Si grilles spécifiques non fournies, on réutilise b_grid
            b_s_grid = b_grid if b_s_grid is None else torch.as_tensor(b_s_grid, device=s.device, dtype=s.dtype)
            b_k_grid = b_grid if b_k_grid is None else torch.as_tensor(b_k_grid, device=s.device, dtype=s.dtype)
            b_x_grid = b_grid if b_x_grid is None else torch.as_tensor(b_x_grid, device=s.device, dtype=s.dtype)

            calibration = {
                "a": None, "b_s": None, "b_k": None, "b_x": None,
                "calibration_score": -float("inf"),
                "u_calibration": None, "preds": None
            }

            # Entrées déjà décodées pour transform (on remet en shape d'origine)
            sigma_dec = s.view(sigma.shape)
            kappa_dec = kappa.detach()
            xi_dec    = xi.detach()
            inputs_decoded = torch.stack([sigma_dec, kappa_dec, xi_dec], dim=-1)

            # Recherche sur (a, b_s, b_k, b_x)
            for a in a_grid:
                for b_s in b_s_grid:
                    for b_k in b_k_grid:
                        for b_x in b_x_grid:
                            logit = a + b_s*phi_s + b_k*phi_k + b_x*phi_x
                            u_vec = torch.sigmoid(logit).clamp(eps, 1.0 - 1e-6).view_as(sigma_dec)

                            preds = self.transform(
                                inputs=inputs_decoded,
                                dir_output=None,
                                from_logits=False,   # IMPORTANT: déjà décodé
                                p_thresh=u_vec,      # vecteur par échantillon
                                pick="max"           # politique imposée
                            )

                            score = float(score_fn(y_true, preds))
                            if score > calibration["calibration_score"]:
                                calibration.update(
                                    a=float(a.item()),
                                    b_s=float(b_s.item()),
                                    b_k=float(b_k.item()),
                                    b_x=float(b_x.item()),
                                    calibration_score=score,
                                    u_calibration=u_vec.clone(),
                                    preds=preds.clone()
                                )

            # --- Visualisation (optionnelle) ---
            if plot and dir_output is not None:
                dir_output = Path(dir_output)
                dir_output.mkdir(parents=True, exist_ok=True)

                order = torch.argsort(s)  # tri selon sigma pour lisibilité
                s_ord   = s[order].cpu().numpy()
                ps_ord  = phi_s[order].cpu().numpy()
                pk_ord  = phi_k[order].cpu().numpy()
                px_ord  = phi_x[order].cpu().numpy()
                u_ord   = calibration["u_calibration"].detach().flatten()[order].cpu().numpy()

                # Courbe u vs. sigma (couleur = phi_k ou phi_x)
                plt.figure(figsize=(7,5))
                sc = plt.scatter(s_ord, u_ord, c=pk_ord, s=10, cmap="viridis", alpha=0.6)
                cbar = plt.colorbar(sc)
                cbar.set_label("phi_k(kappa)" + (" = log(kappa)" if log_kappa else " = kappa"))
                # Ligne lisse (a titre indicatif) : on fige kappa/xi à leur médiane
                a  = calibration["a"]; b_s = calibration["b_s"]; b_k = calibration["b_k"]; b_x = calibration["b_x"]
                k_med = torch.median(k).item()
                x_med = torch.median(x).item()
                fk = (np.log(max(k_med, eps)) if log_kappa else k_med)
                fx = (np.log(max(x_med, eps)) if log_xi    else x_med)
                ss = np.linspace(max(float(s.min().cpu()), eps), float(s.max().cpu()), 400)
                fs = (np.log(np.clip(ss, eps, None)) if log_sigma else ss)
                uu = 1.0/(1.0 + np.exp(-(a + b_s*fs + b_k*fk + b_x*fx)))
                plt.plot(ss, uu, lw=2, label=f"u=σ(a+b_s φ_s + b_k φ_k + b_x φ_x)\n"
                                            f"a={a:.2f}, b_s={b_s:.2f}, b_k={b_k:.2f}, b_x={b_x:.2f}")
                plt.xscale("log" if log_sigma else "linear")
                plt.xlabel("sigma" + (" (log-scale axis)" if log_sigma else ""))
                plt.ylabel("u(sigma,kappa,xi)")
                plt.title("Calibrated threshold u depending on (sigma, kappa, xi)")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                plt.tight_layout()
                plt.savefig(dir_output / "calibrated_u_sigmoid_sigma_kappa_xi.png", dpi=200)
                plt.close()

            return calibration
        
class PredictdEGPDLossTruncClusterIDs(nn.Module):
    """
    Discrete exponentiated GPD (deGPD) TRUNCATED to {0,..., y_max},
    with sigma, kappa, xi provided by the model as inputs[..., 0:3].
    No internal learnable parameters (no global/delta, no per-cluster params).

    - Forward: negative log-likelihood on the TRUNCATED PMF for integer targets in {0,...,y_max}.
    - Transform: returns a discrete y in {0,...,y_max} using a PMF threshold u (pick="max"/"min"),
                 or falls back to argmax PMF if no class reaches the threshold.
    - Optional plotting by cluster for diagnostics (no learnable params involved).

    Inputs layout (last dim):
      inputs[..., 0] = sigma (raw or decoded depending on from_logits)
      inputs[..., 1] = kappa (raw if force_positive, decoded with softplus)
      inputs[..., 2] = xi    (raw if force_positive, decoded with softplus)
    """

    def __init__(self,
                 id: int,
                 num_classes: int = 5,
                 y_max: int = 4,
                 eps: float = 1e-8,
                 reduction: str = "mean",
                 force_positive: bool = True):
        super().__init__()
        self.y_max = int(y_max)
        self.eps = float(eps)
        self.reduction = reduction
        self.force_positive = force_positive
        self.num_classes = num_classes
        self.id = id
        
    # ---------- décodage de sigma ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, areas: Optional[torch.Tensor] = None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        # pas d'offset d'exposition par défaut (gardé pour compat API)
        if areas is not None:
            # si vous souhaitez appliquer un offset d'exposition, adaptez ici
            sigma = sigma  # noop
        return sigma

    # ---------- décodage/contrainte sur (kappa, xi) ----------
    def _decode_kappa_xi(self, kappa_raw: torch.Tensor, xi_raw: torch.Tensor):
        if self.force_positive:
            kappa = F.softplus(kappa_raw)
            xi    = F.softplus(xi_raw)
        else:
            kappa = kappa_raw
            xi    = xi_raw
        # bornes de sécurité numériques
        kappa = kappa.clamp_min(self.eps)
        xi    = xi.clamp(min=-1e6, max=1e6)
        return kappa, xi

    # ---------- GPD CDF H(y; sigma, xi) ----------
    def _gpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y     = torch.clamp(y, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)
        near0   = xi_safe.abs() < 1e-6

        z = 1.0 + xi_safe * (y / sigma)
        z = torch.clamp(z, min=self.eps)

        H_xi = 1.0 - torch.pow(z, -1.0 / xi_safe)   # xi != 0
        H_0  = 1.0 - torch.exp(-y / sigma)          # xi ~ 0
        H = torch.where(near0, H_0, H_xi)
        return torch.clamp(H, min=self.eps, max=1.0 - 1e-12)

    # ---------- eGPD CDF: F_+(y) = [H(y)]^kappa ----------
    def _egpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        H = self._gpd_cdf(y, sigma, xi)
        kappa_pos = torch.clamp(kappa, min=self.eps)
        Fp = torch.pow(H, kappa_pos)
        return torch.clamp(Fp, min=self.eps, max=1.0 - 1e-12)

    # ---------- PMF brute: p_raw(y) = F_+(y+1) - F_+(y) ----------
    def _pmf_raw(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y = y_int.to(dtype=sigma.dtype)
        F_y   = self._egpd_cdf(y,   sigma, kappa, xi)
        F_yp1 = self._egpd_cdf(y+1, sigma, kappa, xi)
        p_raw = torch.clamp(F_yp1 - F_y, min=self.eps)
        return p_raw

    # ---------- PMF TRONQUÉE sur {0,...,y_max} ----------
    def _pmf_trunc(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        p_raw = self._pmf_raw(y_int, sigma, kappa, xi)
        Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                               sigma, kappa, xi)  # normalisation: F_+(y_max+1)
        Z = torch.clamp(Z, min=self.eps)
        p = p_raw / Z
        return torch.clamp(p, min=self.eps, max=1.0)

    # ---------- NLL TRONQUÉE ----------
    def forward(self, inputs: torch.Tensor, y: torch.Tensor,
                weight: Optional[torch.Tensor] = None, from_logits: bool = True) -> torch.Tensor:
        """
        inputs shape: (..., 3) with [sigma, kappa, xi] on the last dim.
        y: integer targets in {0,...,y_max}
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw  = inputs[..., 0]
        kappa_raw  = inputs[..., 1]
        xi_raw     = inputs[..., 2]
        
        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        y_int = y.to(torch.long).clamp(min=0, max=self.y_max)
        p_trunc = self._pmf_trunc(y_int, sigma, kappa, xi)

        nll = -torch.log(torch.clamp(p_trunc, min=self.eps))
        if weight is not None:
            nll = nll * weight
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll  # "none": vector

    # ---------- Trouver y tel que PMF(y) >= p_thresh (min/max/all) ----------
    @torch.no_grad()
    def _y_where_p_ge(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                      p_thresh, pick: str = "max"):
        """
        Retourne:
          - si sigma scalaire: int y
          - si batch: tensor (N,) de y
        Politique si aucun y ne vérifie PMF>=p_thresh : repli sur argmax PMF.
        """
        if isinstance(p_thresh, torch.Tensor):
            assert torch.all(p_thresh >= 0.0) and torch.all(p_thresh <= 1.0), "p_thresh must be in (0,1)"
        else:
            assert 0.0 <= p_thresh <= 1.0, "p_thresh must be in (0,1)"

        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)

        if sigma.ndim == 0:
            pmf = self._pmf_trunc(y_vals, sigma, kappa, xi)  # (y_max+1,)
            mask = (pmf >= p_thresh)
            if not mask.any():
                return int(torch.argmax(pmf).item())
            idxs = torch.nonzero(mask, as_tuple=False).flatten()
            return int((idxs.min() if pick == "min" else idxs.max()).item())

        # batch
        pmf = self._pmf_trunc(
            y_vals.unsqueeze(0).expand(sigma.shape[0], -1),
            sigma.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1)
        )  # (N, y_max+1)
        
        if isinstance(p_thresh, torch.Tensor):
            mask = (pmf >= p_thresh.view(-1, 1))
        else:
            mask = (pmf >= p_thresh)
        y_index = torch.arange(self.y_max + 1, device=sigma.device).unsqueeze(0).expand_as(pmf)

        if pick == "min":
            large = torch.full_like(y_index, self.y_max + 1)
            cand = torch.where(mask, y_index, large)
            y_hat = cand.min(dim=1).values.clamp_max(self.y_max)
        else:  # "max"
            neg = torch.full_like(y_index, -1)
            cand = torch.where(mask, y_index, neg)
            y_hat = cand.max(dim=1).values.clamp_min(0)

        # repli pour les lignes sans True
        no_hit = ~mask.any(dim=1)
        if no_hit.any():
            fallback = pmf.argmax(dim=1)
            y_hat[no_hit] = fallback[no_hit]
        return y_hat

    # ---------- Transform: sélection par seuil de PMF + tracés optionnels ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output: Optional[str], clusters_ids, p_thresh,
                  from_logits: bool = True, pick='max'):
        """
        inputs[...,0]=sigma, inputs[...,1]=kappa, inputs[...,2]=xi
        Retourne un y discret dans {0,...,y_max} tel que PMF(y) >= p_thresh,
        avec tie-breaking contrôlé par `pick` ("min"/"max").
        Repli: argmax PMF si aucun y ne dépasse le seuil.
        Si dir_output n'est pas None, trace PMF/CDF/PPF via plot_final().
        Optionnel: clusters_ids pour des plots par cluster (diagnostic uniquement).
        """
        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        sigma_raw = inputs[..., 0]
        kappa_raw = inputs[..., 1]
        xi_raw    = inputs[..., 2]

        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        if isinstance(p_thresh, dict):
            y_hat = torch.zeros_like(sigma, dtype=torch.int32).long()
            for id in torch.unique(clusters_ids):
                mask = clusters_ids == id
                id = id.item()
                if id in p_thresh.keys():
                    logit = p_thresh[id]['a'] + p_thresh[id]['b_s']*sigma + p_thresh[id]['b_k']*kappa + p_thresh[id]['b_x']*xi
                    u = torch.sigmoid(logit)
                else:
                    print(f'No {id} in calibration dict, taking median -> to do')
                    u = 0.1

                y_hat[mask] = self._y_where_p_ge(sigma[mask], u, kappa[mask], xi[mask], pick=pick)
        else:
            y_hat = self._y_where_p_ge(sigma, p_thresh, kappa, xi, pick=pick)
                
        if dir_output is not None:
            if clusters_ids is None:
                self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='mean')
                self.plot_final(sigma, kappa, xi, dir_output, which_sigmas='max')
            else:
                self.plot_final_by_cluster(sigma, kappa, xi, clusters_ids, dir_output)
        return y_hat

    # ---------- Plots: PMF/CDF/PPF (tronqués) ----------
    @torch.no_grad()
    def plot_final(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                   dir_output: str, which_sigmas: str = 'max'):
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        sig = sigma.detach().flatten()
        kap = kappa.detach().flatten()
        xit = xi.detach().flatten()
        if sig.numel() == 0:
            return

        if which_sigmas == 'mean':
            sig_list = [(sig.mean().item(), kap.mean().item(), xit.mean().item())]
        elif which_sigmas == 'max':
            idx_max = torch.argmax(sig)
            sig_list = [(sig[idx_max].item(), kap[idx_max].item(), xit[idx_max].item())]
        else:
            sig_list = [(sig.mean().item(), kap.mean().item(), xit.mean().item())]

        # 1) PMF tronquée
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            p_raw = self._pmf_raw(y_vals, s_t, k_t, x_t)
            Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                   s_t, k_t, x_t)
            p     = torch.clamp(p_raw / torch.clamp(Z, min=self.eps), min=self.eps)
            plt.stem(y_vals.cpu().numpy(), p.cpu().numpy(), basefmt=" ",
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated PMF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated PMF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_pmf = dir_output / f"degpd_trunc_pmf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_pmf, dpi=200); plt.close()

        # 2) CDF tronquée
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            F_y = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            Z   = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                 s_t, k_t, x_t)
            F_tr = torch.clamp(F_y / torch.clamp(Z, min=self.eps), min=self.eps, max=1.0)
            plt.step(y_vals.cpu().numpy(), F_tr.cpu().numpy(), where="post",
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated CDF on {{0,...,{self.y_max}}}")
        plt.title("deGPD truncated CDF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_cdf = dir_output / f"degpd_trunc_cdf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_cdf, dpi=200); plt.close()

        # 3) PPF discrète via balayage de u ∈ (0,1) -> y(u)
        u_grid = torch.linspace(1e-6, 1.0 - 1e-6, 1000, device=sigma.device)
        plt.figure(figsize=(7,5))
        for s, k, x in sig_list:
            s_t = torch.tensor(s, device=sigma.device, dtype=sigma.dtype)
            k_t = torch.tensor(k, device=sigma.device, dtype=sigma.dtype)
            x_t = torch.tensor(x, device=sigma.device, dtype=sigma.dtype)
            Fmax = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                                  s_t, k_t, x_t)
            u_eff = torch.clamp(u_grid * torch.clamp(Fmax, min=self.eps), min=self.eps, max=1.0 - 1e-12)
            F_tab = self._egpd_cdf(y_vals, s_t, k_t, x_t)
            y_idx = torch.searchsorted(F_tab, u_eff, right=False).clamp(0, self.y_max)
            plt.plot(u_grid.cpu().numpy(), y_idx.cpu().numpy(),
                     label=f"sigma={float(s):.3f}, kappa={float(k):.3f}, xi={float(x):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel(f"PPF (discrete y on {{0,...,{self.y_max}}})")
        plt.title("deGPD truncated PPF")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        f_ppf = dir_output / f"degpd_trunc_ppf_{which_sigmas}.png"
        plt.tight_layout(); plt.savefig(f_ppf, dpi=200); plt.close()

    # ---------- Plots par cluster (diagnostic, aucun paramètre appris) ----------
    @torch.no_grad()
    def plot_final_by_cluster(self, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor,
                              clusters_ids: torch.Tensor, dir_output: str):
        """
        Trace PMF/CDF/PPF par cluster en prenant des sigma représentatifs (mean/max).
        Purement diagnostique : ne crée aucun paramètre appris.
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        clusters_ids = clusters_ids.long()
        for c in torch.unique(clusters_ids):
            mask = (clusters_ids == c)
            if not mask.any():
                continue
            sig_c = sigma[mask]
            kap_c = kappa[mask]
            xi_c  = xi[mask]

            # valeurs représentatives (mean/max) pour (sigma, kappa, xi)
            s_mean = sig_c.mean().item(); s_max = sig_c.max().item()
            k_mean = kap_c.mean().item(); x_mean = xi_c.mean().item()

            name_c = f"cl{int(c.item())}"
            (dir_output / name_c).mkdir(parents=True, exist_ok=True)

            # mean
            self.plot_final(
                sigma=torch.tensor([s_mean], device=sigma.device, dtype=sigma.dtype),
                kappa=torch.tensor([k_mean], device=sigma.device, dtype=sigma.dtype),
                xi=torch.tensor([x_mean], device=sigma.device, dtype=sigma.dtype),
                dir_output=dir_output / name_c,
                which_sigmas="mean"
            )
            # max sigma (avec kappa/xi moyens pour illustrer)
            self.plot_final(
                sigma=torch.tensor([s_max], device=sigma.device, dtype=sigma.dtype),
                kappa=torch.tensor([k_mean], device=sigma.device, dtype=sigma.dtype),
                xi=torch.tensor([x_mean], device=sigma.device, dtype=sigma.dtype),
                dir_output=dir_output / name_c,
                which_sigmas="max"
            )
            
    @torch.no_grad()
    def calibrate(self,
                inputs: torch.Tensor,          # (...,3) : [sigma_raw, kappa_raw, xi_raw] du modèle
                y_true: torch.Tensor,          # (N,)
                clusters_ids: torch.Tensor,    # (N,) ids de cluster
                score_fn,                      # callable: (y_true_sub, preds_sub) -> float
                dir_output: Optional[str] = None,
                from_logits: bool = True,
                log_sigma: bool = False,
                log_kappa: bool = False,
                log_xi:    bool = False,
                a_grid=None,
                b_s_grid=None,
                b_k_grid=None,
                b_x_grid=None,
                plot: bool = True):
        """
        Cluster-wise calibration of u_c(x) = sigmoid(a_c + b_s^c * φ_s(σ) + b_k^c * φ_k(κ) + b_x^c * φ_x(ξ)),
        maximizing `score_fn` on each cluster (prediction via self.transform(..., pick="max")).

        Returns:
            dict { cluster_id: {"a","b_s","b_k","b_x","calibration_score","u_calibration","preds"} }
        """
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt

        if inputs.size(-1) < 3:
            raise ValueError("inputs must have at least 3 channels: sigma, kappa, xi")

        # ---- Decode model outputs -> (σ, κ, ξ) positifs / clampés ----
        sigma_raw, kappa_raw, xi_raw = inputs[..., 0], inputs[..., 1], inputs[..., 2]
        sigma = self._decode_sigma(sigma_raw, from_logits, None)
        kappa, xi = self._decode_kappa_xi(kappa_raw, xi_raw)

        s = sigma.detach().flatten()
        k = kappa.detach().flatten()
        x = xi.detach().flatten()
        N = s.numel()
        if y_true.numel() != N or clusters_ids.numel() != N:
            raise ValueError("y_true and clusters_ids must align with the batch size")

        eps = float(self.eps)
        φs = torch.log(torch.clamp(s, min=eps)) if log_sigma else s
        φk = torch.log(torch.clamp(k, min=eps)) if log_kappa else k
        φx = torch.log(torch.clamp(x, min=eps)) if log_xi    else x

        device, dtype = s.device, s.dtype
        # Grilles par défaut
        if a_grid is None:
            a_grid = torch.linspace(-6.0, 6.0, 25, device=device, dtype=dtype)
        else:
            a_grid = torch.as_tensor(a_grid, device=device, dtype=dtype)

        def _default_b_grid(g):
            if g is not None:
                return torch.as_tensor(g, device=device, dtype=dtype)
            return torch.cat([
                torch.linspace(-8.0, -0.25, 16, device=device, dtype=dtype),
                torch.linspace( 0.25,  8.0, 16, device=device, dtype=dtype)
            ])

        b_s_grid = _default_b_grid(b_s_grid)
        b_k_grid = _default_b_grid(b_k_grid)
        b_x_grid = _default_b_grid(b_x_grid)

        # Entrées décodées (from_logits=False pour transform)
        inputs_decoded = torch.stack([sigma.view_as(sigma),
                                    kappa.detach(),
                                    xi.detach()], dim=-1)

        results = {}
        clusters_ids = clusters_ids.flatten().long()
        unique_cls = torch.unique(clusters_ids)

        for cl in unique_cls:
            mask = (clusters_ids == cl)
            print(f'calibrate {cl}')
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            
            if idx.numel() == 0:
                continue

            y_sub   = y_true[idx]
            φs_sub  = φs[idx]
            φk_sub  = φk[idx]
            φx_sub  = φx[idx]
            in_sub  = inputs_decoded.view(-1, 3)[idx]  # (n_sub,3)

            best = {
                "a": None, "b_s": None, "b_k": None, "b_x": None,
                "calibration_score": -float("inf"),
                "u_calibration": None, "preds": None
            }

            for a in a_grid:
                for bs in b_s_grid:
                    for bk in b_k_grid:
                        for bx in b_x_grid:
                            logit = a + bs*φs_sub + bk*φk_sub + bx*φx_sub
                            u_vec = torch.sigmoid(logit).clamp(eps, 1.0 - 1e-6)  # (n_sub,)

                            preds = self.transform(
                                inputs=in_sub,
                                dir_output=None,
                                p_thresh=u_vec,
                                from_logits=False,
                                pick="max",
                                clusters_ids=clusters_ids,
                            )

                            score = float(score_fn(y_sub, preds))
                            if score > best["calibration_score"]:
                                best.update(a=float(a.item()),
                                            b_s=float(bs.item()),
                                            b_k=float(bk.item()),
                                            b_x=float(bx.item()),
                                            calibration_score=score,
                                            u_calibration=u_vec.clone(),
                                            preds=preds.clone())

            # --- Plot optionnel par cluster ---
            if plot and dir_output is not None and best["u_calibration"] is not None:
                outdir = Path(dir_output) / f"cl{int(cl.item())}"
                outdir.mkdir(parents=True, exist_ok=True)

                # Scatter σ vs u ; courbe lissée en figeant κ, ξ à leur médiane (visuel)
                s_sub = s[idx]
                order = torch.argsort(s_sub)
                σ_axis = s_sub[order].cpu().numpy()
                u_axis = best["u_calibration"][order].cpu().numpy()

                a, bs, bk, bx = best["a"], best["b_s"], best["b_k"], best["b_x"]
                k_med = k[idx].median().item()
                x_med = x[idx].median().item()
                fk = np.log(max(k_med, eps)) if log_kappa else k_med
                fx = np.log(max(x_med, eps)) if log_xi    else x_med
                ss = np.linspace(max(float(s_sub.min().cpu()), eps),
                                float(s_sub.max().cpu()), 400)
                fs = np.log(np.clip(ss, eps, None)) if log_sigma else ss
                uu = 1.0/(1.0 + np.exp(-(a + bs*fs + bk*fk + bx*fx)))

                plt.figure(figsize=(7,5))
                plt.scatter(σ_axis, u_axis, s=10, alpha=0.5, label="u calibrated")
                plt.plot(ss, uu, lw=2, label=f"u=σ(a+bs φ_s+bk φ_k+bx φ_x)\n"
                                            f"a={a:.2f}, bs={bs:.2f}, bk={bk:.2f}, bx={bx:.2f}")
                plt.xscale("log" if log_sigma else "linear")
                plt.xlabel("sigma" + (" (log-scale axis)" if log_sigma else ""))
                plt.ylabel("u")
                plt.title(f"Calibrated u(σ,κ,ξ) — cl{int(cl.item())}")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                plt.tight_layout()
                plt.savefig(outdir / "u_calibration_sigma_kappa_xi.png", dpi=200)
                plt.close()

            results[int(cl.item())] = best

        return results

class dEGPDLossTruncClusterIDs(nn.Module):
    """
    Discrete exponentiated GPD (deGPD) TRUNCATED to {0,..., y_max},
    with cluster-specific parameters (kappa, xi) and optional
    global + per-cluster deltas (partial pooling).

    - Forward: negative log-likelihood on the TRUNCATED PMF.
      For integer targets y in {0,...,y_max}.
    - Transform: returns a (truncated) discrete quantile via u (with F_+(y_max+1) renormalization),
      or the discrete mode (argmax PMF), and can also plot PMF/CDF/PPF by cluster.

    CDF pieces:
        H(y)   = GPD CDF(sigma, xi) on y>=0
        F_+(y) = [H(y)]^kappa  (eGPD CDF, family 1)

    TRUNCATED PMF (on {0,...,y_max}):
        p_trunc(y) = [F_+(y+1) - F_+(y)] / F_+(y_max+1)

    Args
    ----
    NC : int
        Number of clusters.
    y_max : int
        Truncation upper bound (support {0,...,y_max}).
    id : optional
        Identifier used in plot filenames.
    kappa_init, xi_init : float
        Initial values for (kappa, xi).
    eps : float
        Numerical epsilon.
    reduction : {"mean","sum","none"}
        Reduction for the NLL.
    force_positive : bool
        If True, enforce kappa>0 and xi>0 via softplus (and xi>=xi_min).
    xi_min : float
        Lower floor added to xi when force_positive=True.
    G : bool
        If True, use global + per-cluster deltas; else per-cluster params directly.
    area_exponent : float
        Exponent for exposure offset on sigma: sigma <- sigma * areas**area_exponent.
        (Set to 0.0 if you don't want any offset.)
    """

    def __init__(self,
                 NC: int,
                 num_classes: int = 5,
                 id: int = None,
                 kappa_init: float = 0.8,
                 xi_init: float = 0.15,
                 eps: float = 1e-8,
                 reduction: str = "mean",
                 force_positive: bool = False,
                 xi_min: float = 0.0,
                 G: bool = False                 # global + delta:
        ):
        super().__init__()
        self.n_clusters     = NC
        self.y_max          = num_classes - 1
        self.id             = id
        self.eps            = eps
        self.reduction      = reduction
        self.force_positive = force_positive
        self.xi_min         = xi_min
        self.add_global     = G
        self.num_classes = num_classes
        
        if not G:
            # cluster-only parameters
            self.kappa_raw = nn.Parameter(torch.full((NC,), float(kappa_init)))
            self.xi_raw    = nn.Parameter(torch.full((NC,), float(xi_init)))
        else:
            # global + per-cluster deltas
            self.kappa_global = nn.Parameter(torch.tensor(float(kappa_init)))
            self.xi_global    = nn.Parameter(torch.tensor(float(xi_init)))
            self.kappa_delta  = nn.Parameter(torch.zeros(NC))
            self.xi_delta     = nn.Parameter(torch.zeros(NC))

    # ---------- select raw params per sample ----------
    def _select_raw_params(self, clusters_ids: torch.Tensor):
        if isinstance(clusters_ids, torch.Tensor):
            clusters_ids = clusters_ids.long()
        if not self.add_global:
            kappa_raw_sel = self.kappa_raw[clusters_ids]                 # (N,)
            xi_raw_sel    = self.xi_raw[clusters_ids]                    # (N,)
        else:
            kappa_raw_sel = self.kappa_global + self.kappa_delta[clusters_ids]
            xi_raw_sel    = self.xi_global    + self.xi_delta[clusters_ids]
        return kappa_raw_sel, xi_raw_sel

    # ---------- enforce positivity if requested ----------
    def _positivize(self, kappa_raw_sel: torch.Tensor, xi_raw_sel: torch.Tensor):
        if self.force_positive:
            kappa = F.softplus(kappa_raw_sel)                 # >0
            xi    = F.softplus(xi_raw_sel) + float(self.xi_min)  # >xi_min
        else:
            kappa = kappa_raw_sel
            xi    = xi_raw_sel
        return kappa, xi

    # ---------- decode sigma (+ exposure offset if provided) ----------
    def _decode_sigma(self, sigma_raw: torch.Tensor, from_logits: bool, areas=None):
        sigma = F.softplus(sigma_raw) if from_logits else sigma_raw
        sigma = sigma.clamp_min(self.eps)
        area = None
        if areas is not None:
            sigma = sigma * torch.clamp(areas, min=self.eps) ** self.area_exponent
        return sigma

    # ---------- GPD CDF H(y; sigma, xi) ----------
    def _gpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y     = torch.clamp(y, min=0.0)
        sigma = torch.clamp(sigma, min=self.eps)
        xi_safe = torch.clamp(xi, min=-1e6, max=1e6)
        near0   = xi_safe.abs() < 1e-6

        z = 1.0 + xi_safe * (y / sigma)
        z = torch.clamp(z, min=self.eps)

        H_xi = 1.0 - torch.pow(z, -1.0 / xi_safe)   # xi != 0
        H_0  = 1.0 - torch.exp(-y / sigma)          # xi ~ 0
        H = torch.where(near0, H_0, H_xi)
        return torch.clamp(H, min=self.eps, max=1.0 - 1e-12)

    # ---------- eGPD CDF: F_+(y) = [H(y)]^kappa ----------
    def _egpd_cdf(self, y: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        H = self._gpd_cdf(y, sigma, xi)
        kappa_pos = torch.clamp(kappa, min=self.eps)
        Fp = torch.pow(H, kappa_pos)
        return torch.clamp(Fp, min=self.eps, max=1.0 - 1e-12)

    # ---------- NON-truncated discrete PMF: p_raw(y) = F_+(y+1) - F_+(y) ----------
    def _pmf_raw(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        y = y_int.to(dtype=sigma.dtype)
        F_y   = self._egpd_cdf(y,   sigma, kappa, xi)
        F_yp1 = self._egpd_cdf(y+1, sigma, kappa, xi)
        p_raw = torch.clamp(F_yp1 - F_y, min=self.eps)
        return p_raw

    # ---------- TRUNCATED PMF on {0,...,y_max}: normalize by F_+(y_max+1) ----------
    def _pmf_trunc(self, y_int: torch.Tensor, sigma: torch.Tensor, kappa: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        p_raw = self._pmf_raw(y_int, sigma, kappa, xi)
        Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                               sigma, kappa, xi)  # F_+(y_max+1)
        Z = torch.clamp(Z, min=self.eps)
        p = p_raw / Z
        return torch.clamp(p, min=self.eps, max=1.0)

    # ---------- FORWARD: truncated NLL on integer y in {0,...,y_max} ----------
    def forward(self, inputs: torch.Tensor, y: torch.Tensor, clusters_ids: torch.Tensor,
                areas: torch.Tensor = None, weight: torch.Tensor = None,
                from_logits: bool = True) -> torch.Tensor:
        """
        inputs : (...,) sigma_raw (échelle)
        y      : int tensor, targets in {0,...,y_max}
        clusters_ids : (N,) longs
        areas  : (N,) optionnel, offset d'exposition sur sigma
        """
        sigma_raw = inputs[..., 0] if inputs.ndim > 1 else inputs
        sigma = self._decode_sigma(sigma_raw, from_logits, areas)

        kappa_raw_sel, xi_raw_sel = self._select_raw_params(clusters_ids)
        kappa, xi = self._positivize(kappa_raw_sel, xi_raw_sel)

        y_int = y.to(torch.long).clamp(min=0, max=self.y_max)
        p_trunc = self._pmf_trunc(y_int, sigma, kappa, xi)
        
        nll = -torch.log(torch.clamp(p_trunc, min=self.eps))
        if weight is not None:
            nll = nll * weight

        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll
    
    def find_most_probable_value(self, sigma, u, kappa, xi):
        
        # tables sur {0,...,y_max}
        y_vals = torch.arange(0, self.y_max + 1, device=sigma.device, dtype=sigma.dtype)
        
        # quantile tronqué: u_eff = u * F_+(y_max+1)
        if not torch.is_tensor(u):
            u = torch.tensor(u, dtype=sigma.dtype, device=sigma.device)
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)

        Fmax = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigma.dtype, device=sigma.device),
                            sigma, kappa, xi)                  # (N,)
        u_eff = torch.clamp(u * Fmax, min=self.eps, max=1.0 - 1e-12)
        
        # F_+(y) pour y=0..y_max
        ys2 = y_vals.unsqueeze(0).expand(sigma.shape[0], -1)      # (N, y_max+1)
        F_tab = self._egpd_cdf(ys2, sigma.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1))

        # plus petit y tel que F_tab >= u_eff
        idx = torch.searchsorted(F_tab, u_eff.unsqueeze(-1), right=False)
        y_hat = idx.clamp(min=0, max=self.y_max).squeeze(-1)
        return y_hat

    # ---------- TRANSFORM: truncated discrete quantile or mode ----------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, clusters_ids: torch.Tensor, p_thresh: Any,
                  dir_output, from_logits: bool = True,
                  mode: bool = False, plot: bool = False):
        """
        Returns a discrete prediction in {0,...,y_max}.
        - If mode=True: returns argmax_y p_trunc(y).
        - Else: returns the (truncated) quantile:
            u_eff = u * F_+(y_max+1),
            y_hat = min { y : F_+(y) >= u_eff }  clipped to [0, y_max].

        Also plots PMF/CDF/PPF per cluster if dir_output is provided or plot=True.
        """
        sigma_raw = inputs[..., 0] if inputs.ndim > 1 else inputs
        sigma = self._decode_sigma(sigma_raw, from_logits)
        kappa_raw_sel, xi_raw_sel = self._select_raw_params(clusters_ids)
        kappa, xi = self._positivize(kappa_raw_sel, xi_raw_sel)
        if mode:
            # argmax_y p_trunc(y)
            # tables sur {0,...,y_max}
            y_vals = torch.arange(0, self.y_max + 1, device=sigma.device, dtype=sigma.dtype)
            p = self._pmf_trunc(y_vals, sigma.unsqueeze(-1), kappa.unsqueeze(-1), xi.unsqueeze(-1))  # (N, y_max+1)
            y_hat = p.argmax(dim=-1)
        else:
            if isinstance(p_thresh, dict):
                y_hat = torch.zeros_like(sigma, dtype=torch.int32).long()
                for id in torch.unique(clusters_ids):
                    mask = clusters_ids == id
                    id = id.item()
                    kappa_raw_sel, xi_raw_sel = self._select_raw_params(id)
                    kappa, xi = self._positivize(kappa_raw_sel, xi_raw_sel)
                    if id in p_thresh.keys():
                        u = torch.sigmoid(p_thresh[id]['a'] + sigma[mask] * p_thresh[id]['b'])
                    else:
                        print(f'No {id} in calibration dict, taking median -> to do')
                        u = 0.1
                    
                    y_hat[mask] = self.find_most_probable_value(sigma[mask], u, kappa, xi)
            else:
                y_hat = self.find_most_probable_value(sigma, p_thresh, kappa, xi)
                
        if dir_output is not None or plot:
            self.plot_final_pmf(sigma, clusters_ids, dir_output)

        return y_hat

    # ---------- PLOTS: PMF/CDF/PPF per cluster ----------
    @torch.no_grad()
    def _plot_one_cluster(self, sigmas_c: torch.Tensor, kappa_c: torch.Tensor, xi_c: torch.Tensor,
                          dir_output, cname: str, which_sigma : str):
        """
        Save 3 plots for one cluster:
          1) Truncated PMF on {0,...,y_max}
          2) Truncated CDF on {0,...,y_max}
          3) Truncated PPF (discrete y(u)) for u in (0,1)
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        # choose representative sigmas: min/median/max
        sig = sigmas_c.flatten()
        if sig.numel() == 0:
            return

        if which_sigma == 'mean':
            sig_op = torch.mean(sig)
        elif which_sigma == 'max':
            sig_op = torch.max(sig)
        
        sig_list = [sig_op]

        y_vals = torch.arange(0, self.y_max + 1, device=sigmas_c.device)

        # 1) PMF tronquée
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigmas_c.device, dtype=sigmas_c.dtype)
            p_raw = self._pmf_raw(y_vals, s_t, kappa_c, xi_c)
            Z     = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigmas_c.dtype, device=sigmas_c.device),
                                   s_t, kappa_c, xi_c)
            p     = torch.clamp(p_raw / torch.clamp(Z, min=self.eps), min=self.eps)
            plt.stem(y_vals.cpu().numpy(), p.cpu().numpy(), basefmt=" ", label=f"sigma={float(s):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated PMF on {{0,...,{self.y_max}}}")
        plt.title(f"deGPD truncated PMF  (kappa={float(kappa_c):.3f}, xi={float(xi_c):.3f}) — {cname}")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / cname / f"degpd_trunc_pmf_{which_sigma}.png", dpi=200); plt.close()

        # 2) CDF tronquée F_tr(y) = F_+(y)/F_+(y_max+1)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigmas_c.device, dtype=sigmas_c.dtype)
            F_y = self._egpd_cdf(y_vals, s_t, kappa_c, xi_c)
            Z   = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigmas_c.dtype, device=sigmas_c.device),
                                 s_t, kappa_c, xi_c)
            F_tr = torch.clamp(F_y / torch.clamp(Z, min=self.eps), min=self.eps, max=1.0)
            plt.step(y_vals.cpu().numpy(), F_tr.cpu().numpy(), where="post", label=f"sigma={float(s):.3f}")
        plt.xlabel("y (count)")
        plt.ylabel(f"Truncated CDF on {{0,...,{self.y_max}}}")
        plt.title(f"deGPD truncated CDF  (kappa={float(kappa_c):.3f}, xi={float(xi_c):.3f}) — {cname}")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / cname / f"degpd_trunc_cdf_{which_sigma}.png", dpi=200); plt.close()

        # 3) PPF (discrete): y(u) = min{ y : F_+(y) >= u * F_+(y_max+1) }
        u_grid = torch.linspace(1e-6, 1.0 - 1e-6, 1000, device=sigmas_c.device)
        plt.figure(figsize=(7,5))
        for s in sig_list:
            s_t = torch.tensor(s, device=sigmas_c.device, dtype=sigmas_c.dtype)
            Fmax = self._egpd_cdf(torch.tensor(self.y_max + 1, dtype=sigmas_c.dtype, device=sigmas_c.device),
                                  s_t, kappa_c, xi_c)
            u_eff = torch.clamp(u_grid * torch.clamp(Fmax, min=self.eps), min=self.eps, max=1.0 - 1e-12)
            F_tab = self._egpd_cdf(y_vals, s_t, kappa_c, xi_c)  # (y_max+1,)
            y_idx = torch.searchsorted(F_tab, u_eff, right=False).clamp(0, self.y_max)
            plt.plot(u_grid.cpu().numpy(), y_idx.cpu().numpy(), label=f"sigma={float(s):.3f}")
        plt.xlabel("u (quantile level)")
        plt.ylabel(f"PPF (discrete y in {{0,...,{self.y_max}}})")
        plt.title(f"deGPD truncated PPF  (kappa={float(kappa_c):.3f}, xi={float(xi_c):.3f}) — {cname}")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(); plt.savefig(dir_output / cname / f"degpd_trunc_ppf_{which_sigma}.png", dpi=200); plt.close()

    @torch.no_grad()
    def plot_final_pmf(self, sigma: torch.Tensor, clusters_ids: torch.Tensor, dir_output):
        """
        Plot PMF/CDF/PPF per cluster (saves PNG files).
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        clusters_ids = clusters_ids.long()
        C = self.n_clusters

        # params per cluster (after positivity if needed)
        if not self.add_global:
            kappa_full = self.kappa_raw
            xi_full    = self.xi_raw
        else:
            kappa_full = self.kappa_global + self.kappa_delta
            xi_full    = self.xi_global    + self.xi_delta

        if self.force_positive:
            kappa_full = F.softplus(kappa_full)
            xi_full    = F.softplus(xi_full) + float(self.xi_min)

        for c in torch.unique(clusters_ids):
            mask = (clusters_ids == c)
            if mask.any():
                sig_c = sigma[mask]
                name_c = f"cl{c}"
                self._plot_one_cluster(sig_c, kappa_full[c], xi_full[c], dir_output, name_c, 'mean')
                self._plot_one_cluster(sig_c, kappa_full[c], xi_full[c], dir_output, name_c, 'max')

    # ---------- parameters access ----------
    def get_learnable_parameters(self):
        if not self.add_global:
            return {"kappa": self.kappa_raw, "xi": self.xi_raw}
        else:
            return {
                "kappa_global": self.kappa_global,
                "xi_global": self.xi_global,
                "kappa_delta": self.kappa_delta,
                "xi_delta": self.xi_delta,
            }

    def get_attribute(self):
        if not self.add_global:
            if self.force_positive:
                return [
                    ('kappa', F.softplus(self.kappa_raw).detach()),
                    ('xi',    (F.softplus(self.xi_raw) + self.xi_min).detach()),
                ]
            else:
                return [
                    ('kappa', self.kappa_raw.detach()),
                    ('xi',    self.xi_raw.detach()),
                ]
        else:
            kappa_raw_full = self.kappa_global + self.kappa_delta
            xi_raw_full    = self.xi_global    + self.xi_delta
            if self.force_positive:
                return [
                    ('kappa_global', F.softplus(self.kappa_global).detach()),
                    ('xi_global',    (F.softplus(self.xi_global) + self.xi_min).detach()),
                    ('kappa_per_cluster', F.softplus(kappa_raw_full).detach()),
                    ('xi_per_cluster',    (F.softplus(xi_raw_full) + self.xi_min).detach()),
                ]
            else:
                return [
                    ('kappa_global', self.kappa_global.detach()),
                    ('xi_global',    self.xi_global.detach()),
                    ('kappa_per_cluster', kappa_raw_full.detach()),
                    ('xi_per_cluster',    xi_raw_full.detach()),
                ]

    # ---------- optional shrinkage penalty (for global+delta) ----------
    def shrinkage_penalty(self, lambda_l2: float = 1e-4) -> torch.Tensor:
        if not self.add_global:
            device = (self.kappa_raw if hasattr(self, "kappa_raw") else torch.tensor(0.)).device
            return torch.zeros((), device=device)
        return lambda_l2 * (self.kappa_delta.pow(2).sum() + self.xi_delta.pow(2).sum())

    # ---------- plot params over epochs (per cluster, optional global) ----------
    def plot_params(self, egpd_logs, dir_output, clusters_ids=None, dpi=120):
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        epochs = [int(log["epoch"]) for log in egpd_logs]

        if not self.add_global:
            kappas_raw = [log["kappa"] for log in egpd_logs]  # (C,)
            xis_raw    = [log["xi"]    for log in egpd_logs]  # (C,)
        else:
            kappas_raw = [log["kappa_per_cluster"] for log in egpd_logs]  # (C,)
            xis_raw    = [log["xi_per_cluster"]    for log in egpd_logs]  # (C,)
            kappa_global = [log["kappa_global"] for log in egpd_logs]     # scalar
            xi_global    = [log["xi_global"]    for log in egpd_logs]     # scalar

        def to_np_vec(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        K_list = [to_np_vec(k) for k in kappas_raw]
        X_list = [to_np_vec(x) for x in xis_raw]
        K = np.stack(K_list, axis=0)  # (T, C)
        X = np.stack(X_list, axis=0)

        egpd_to_save = {"epoch": epochs, "kappa": K_list, "xi": X_list}
        if self.add_global:
            egpd_to_save["kappa_global"] = kappa_global
            egpd_to_save["xi_global"] = xi_global
        save_object(egpd_to_save, 'degpd_kappa_xi.pkl', dir_output)

        if clusters_ids is not None:
            u_clusters = torch.unique(clusters_ids)
        else:
            u_clusters = np.arange(0, self.n_clusters)
        for c in u_clusters:
            name = f"cl{c}"
            
            check_and_create_path(dir_output / name)

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, K[:, c], marker='o', label=r'$\kappa$')
            plt.plot(epochs, X[:, c], marker='s', label=r'$\xi$')
            plt.xlabel('Epoch'); plt.ylabel('Value')
            plt.title(f'deGPD parameters over epochs — {name}')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(dir_output) / name / f'degpd_params.png', dpi=dpi)
            plt.close()

        if self.add_global:
            kg = np.asarray([to_np_vec(k) for k in kappa_global])
            xg = np.asarray([to_np_vec(x) for x in xi_global])

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, kg, marker='o', label=r'$\kappa_{global}$')
            plt.plot(epochs, xg, marker='s', label=r'$\xi_{global}$')
            plt.xlabel('Epoch'); plt.ylabel('Value')
            plt.title('Global deGPD parameters over epochs')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(dir_output) / f'degpd_params_global.png', dpi=dpi)
            plt.close()
            
    def calibrate(self,
                    inputs: torch.Tensor,
                    y_true : torch.Tensor,
                    clusters_ids : torch.Tensor,
                    score_fn,               # callable: preds -> float
                    dir_output,
                    from_logtis = True,
                    log_sigma: bool = False,
                    a_grid=None,
                    b_grid=None,
                    plot: bool = True):
        """
        Calibre u(x) = sigmoid(a + b * phi(sigma)) en maximisant `score_fn`.
        - `sigma`: tensor (N,) ou (N,1), déjà sur l'échelle d'échelle (pas de logits).
        - `score_fn(preds) -> float`: calcule le score final à maximiser à partir des
        prédictions discrètes (par ex. classes après transform avec quantile u).
        NOTE: `score_fn` doit capturer la vérité terrain (closure) si nécessaire.
        - `dir_output`: dossier où sauvegarder la figure (si plot=True).
        - `areas`: offset éventuel passé à `transform` (exposition) ; None pour ignorer.
        - `log_sigma`: si True, phi(sigma) = log(sigma), sinon phi(sigma)=sigma.
        - `a_grid`, `b_grid`: grilles de recherche (torch.Tensor ou list). Par défaut, on crée
        une grille raisonnable.
        - `plot`: si True, trace u vs sigma et sauvegarde un PNG.

        """
        
        sigma_raw = inputs[..., 0] if inputs.ndim > 1 else inputs
        sigma = self._decode_sigma(sigma_raw, from_logtis, None)
        
        calibration = {}
        
        u_clusters = torch.unique(clusters_ids)
        
        for id in u_clusters:
            mask = clusters_ids == id
            id = id.item()
            name = f"cl{id}"

            with torch.no_grad():
                s = sigma.detach().flatten()[mask]
                # feature phi(sigma)
                x = torch.log(torch.clamp(s, min=self.eps)) if log_sigma else s

                # grilles par défaut (couvrent large, dont la zone "extrêmes")
                if a_grid is None:
                    a_grid = torch.linspace(-6.0, 6.0, 25, device=s.device, dtype=s.dtype)
                else:
                    a_grid = torch.as_tensor(a_grid, device=s.device, dtype=s.dtype)
                if b_grid is None:
                    # on teste pentes positives & négatives
                    b_grid = torch.cat([
                        torch.linspace(-8.0, -0.25, 16, device=s.device, dtype=s.dtype),
                        torch.linspace( 0.25,  8.0, 16, device=s.device, dtype=s.dtype)
                    ])
                else:
                    b_grid = torch.as_tensor(b_grid, device=s.device, dtype=s.dtype)

                calibration_id = {"a": None, "b": None, "calibration_score": -float("inf"),
                        "u_calibration": None, "preds": None}

                # recherche sur grille (robuste et simple)
                for a in a_grid:
                    for b in b_grid:
                        u_vec = torch.sigmoid(a + b * x)              # (N,)
                        # prédictions discrètes via quantile tronqué
                        preds = self.transform(
                            inputs=s,               # on passe sigma "déjà décodé"
                            clusters_ids=clusters_ids[mask],
                            dir_output=None,
                            from_logits=False,      # IMPORTANT
                            mode=False,
                            p_thresh=u_vec
                        )
                        
                        score = float(score_fn(y_true[mask], preds))
                        if score > calibration_id["calibration_score"]:
                            calibration_id.update(a=float(a.item()),
                                        b=float(b.item()),
                                        calibration_score=score,
                                        u_calibration=u_vec.clone(),
                                        preds=preds.clone())

                # tracé diagnostique
                if plot and dir_output is not None:
                    check_and_create_path(dir_output / name)
                    dir_output = Path(dir_output)
                    dir_output.mkdir(parents=True, exist_ok=True)

                    # Scatter sigma vs u_calibration + courbe sigmoïde moyenne (sur l'axe x)
                    # On ordonne pour une courbe lisible
                    order = torch.argsort(s)
                    s_ord = s[order].cpu().numpy()
                    x_ord = x[order].cpu().numpy()
                    u_ord = calibration_id["u_calibration"][order].cpu().numpy()
                    
                    plt.figure(figsize=(7,5))
                    # points
                    plt.scatter(s_ord, u_ord, s=10, alpha=0.5, label="u(sigma) calibrated")
                    # courbe "théorique" sur un axe régulier (pour le visuel)
                    xx = np.linspace(x.min().cpu(), x.max().cpu(), 400)
                    uu = 1.0 / (1.0 + np.exp(-(calibration_id["a"] + calibration_id["b"] * xx)))
                    # convertir xx -> axe sigma si log_sigma
                    if log_sigma:
                        ss = np.exp(xx)
                    else:
                        ss = xx
                    plt.plot(ss, uu, lw=2, label=f"sigmoid(a + b·phi), a={calibration_id['a']:.3f}, b={calibration_id['b']:.3f}")
                    plt.xscale("log" if log_sigma else "linear")
                    plt.xlabel("sigma" + (" (log-scale axis)" if log_sigma else ""))
                    plt.ylabel("u(sigma)")
                    plt.title("Calibrated u(sigma) = sigmoid(a + b·phi(sigma))")
                    plt.grid(True, linestyle="--", alpha=0.4)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(dir_output / name / f"calibrated_u_sigmoid.png", dpi=200)
                    plt.close()
                
                calibration[id] = calibration_id
        
        return calibration
        
##########################################################################################################################

class EGPDIntervalCELoss(nn.Module):
    """
    Interval CE sur une seule distribution choisie par T ∈ {'egpd','log','gamma'}.
    Les probabilités par classe sont obtenues par différences de CDF aux bords appris:
        P0 = F(u0)
        Pj = F(uj) - F(uj-1) pour j=1..K-2
        PK-1 = 1 - F(uK-1)
    Edges TOUJOURS apprenables (u0 + cumsum(softplus(t_steps))).

    Entrées du forward:
        inputs: Tensor [N] — paramètre d'échelle par échantillon selon T:
            - T='egpd'  -> sigma_tail_i
            - T='log'   -> sigma_ln_i
            - T='gamma' -> theta_i (échelle)
        y_class: [N] entiers dans {0..K-1}

    Choix de loss via L:
        - 'entropy' : NLL sur log(P)
        - 'mcewk'   : MCE + Wasserstein-K (nécessite MCEAndWKLoss)
        - 'bceloss' : BCE multi-classes sur P (nécessite BCELoss)
    """

    def __init__(self,
                 num_classes: int = 5,            # classes 0..K-1
                 # eGPD (si T='egpd')
                 kappa: float = 0.831,
                 xi: float = 0.161,
                 # LogNormal (si T='log')
                 mu_log: float = 0.0,             # moyenne du log (globale)
                 # Gamma (si T='gamma')
                 k_shape_init: float = 2.0,       # forme globale k>0
                 # Edges
                 u0_init: float = 0.5,
                 # Divers
                 eps: float = 1e-8,
                 L: str = "entropy",
                 reduction: str = "mean",
                 force_positive: bool = False,
                 T: str = "egpd",                 # 'egpd' | 'log' | 'gamma'
                 xi_min: float = 1e-12):
        super().__init__()
        assert num_classes >= 2, "num_classes>=2 requis"
        assert T in {"egpd","log","gamma"}, f"T must be in {{'egpd','log','gamma'}}, got {T}"
        assert L in {"entropy","mcewk","bceloss"}, f"L must be in {{'entropy','mcewk','bceloss'}}, got {L}"

        self.num_classes   = num_classes
        self.eps           = eps
        self.reduction     = reduction
        self.force_positive= force_positive
        self.L             = L
        self.T             = T
        self.xi_min        = xi_min

        # --- Paramètres globaux selon T ---
        # eGPD
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))
        self.xi    = nn.Parameter(torch.tensor(float(xi)))
        # LogNormal
        self.mu_log = nn.Parameter(torch.tensor(float(mu_log)))
        # Gamma (forme k globale)
        self.k_shape = nn.Parameter(torch.tensor(float(k_shape_init)))

        # --- Edges appris ---
        self.u0_raw  = nn.Parameter(torch.tensor(float(u0_init)))
        # t_steps de taille K-2 -> edges = u0 + cumsum(softplus(t_steps))
        self.t_steps = nn.Parameter(torch.zeros(num_classes - 2))

    # ======== Edges ========
    def _sp(self, x):  # softplus stable
        return F.softplus(x) + 1e-12

    def _u0(self, device, dtype):
        return self._sp(self.u0_raw).to(device=device, dtype=dtype)

    def _edges_tensor(self, device, dtype):
        steps = self._sp(self.t_steps.to(device=device, dtype=dtype))     # (K-2,) > 0
        u0    = self._u0(device, dtype)                                   # scalaire >= 0
        return u0 + torch.cumsum(steps, dim=0)                            # (K-1,)

    def get_edges(self):
        with torch.no_grad():
            steps = self._sp(self.t_steps)
            u0    = self._sp(self.u0_raw)
            edges = (u0 + torch.cumsum(steps, dim=0))
        return tuple(x for x in edges)

    # ======== CDFs ========
    @staticmethod
    def _cdf_lognormal(u: torch.Tensor, mu_log: torch.Tensor, sigma_ln: torch.Tensor, eps: float) -> torch.Tensor:
        # F_LN(u) = Phi((ln u - mu)/sigma), u>0 ; sinon 0
        pos = (u > 0)
        z = (torch.log(u.clamp(min=eps)) - mu_log) / (sigma_ln * math.sqrt(2.0))
        F_ln = 0.5 * (1.0 + torch.erf(z))
        return torch.where(pos, F_ln, torch.zeros_like(F_ln))

    def _cdf_gamma_k_theta(self, u: torch.Tensor, k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        CDF Gamma(k, θ) = P(k, u/θ).
        NOTE: PyTorch n'a pas de gradient w.r.t k dans igamma -> on DÉTACHE k ici.
        Si tu veux apprendre k, remplace par une intégration numérique trapèzes.
        """
        k_det = k.clamp_min(self.eps).detach()
        theta = theta.clamp_min(self.eps)
        x = (u / theta).clamp_min(0.0)
        try:
            Pkx = torch.special.gammainc(k_det, x)  # regularized lower
        except AttributeError:
            Pkx = torch.igamma(k_det, x) / torch.exp(torch.lgamma(k_det))
        return Pkx

    def _cdf_primary(self, u: torch.Tensor, scale: torch.Tensor, device, dtype) -> torch.Tensor:
        """
        Retourne F(u) de la distribution choisie par T, avec 'scale' per-sample.
        """
        if self.T == "egpd":
            kappa = self._sp(self.kappa).to(device=device, dtype=dtype) if self.force_positive else self.kappa.to(device, dtype)
            xi    = (self._sp(self.xi) + self.xi_min).to(device=device, dtype=dtype) if self.force_positive else self.xi.to(device, dtype)
            sigma = scale
            return cdf_egpd_family1(u.clamp_min(0.0), self._sp(sigma), xi, kappa, eps=1e-12)
        elif self.T == "log":
            mu_log = self.mu_log.to(device=device, dtype=dtype)
            sigma_ln = self._sp(scale)  # >0
            return self._cdf_lognormal(u, mu_log, sigma_ln, self.eps)
        else:  # "gamma"
            k = self._sp(self.k_shape) if self.force_positive else self.k_shape
            theta = self._sp(scale)  # >0
            return self._cdf_gamma_k_theta(u, k.to(device=device, dtype=dtype), theta)

    # ======== transform (diagnostic) ========
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output: Path, output_pdf: str) -> torch.Tensor:
        """
        inputs: [N] échelles per-sample, interprétées selon T (voir doc classe).
        Retourne P ∈ [N,K].
        """
        scale = inputs.squeeze(-1) if inputs.ndim > 1 else inputs
        device, dtype = scale.device, scale.dtype
        K = self.num_classes
        
        scale = F.softplus(scale) + 0.05

        # plot diag
        self.plot_final_pdf(scale, dir_output, output_pdf=output_pdf, stat="median")

        # Edges
        u0    = self._u0(device, dtype)
        edges = self._edges_tensor(device, dtype)

        # CDF aux bords
        F_u = [ self._cdf_primary(u0.expand_as(scale), scale, device, dtype) ]
        for u in edges:
            F_u.append(self._cdf_primary(u.expand_as(scale), scale, device, dtype))

        # Probas par différences
        Ps = [F_u[0]] + [torch.clamp(F_u[j+1] - F_u[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_u[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)
        return P

    # ======== Forward (loss) ========
    def forward(self,
                inputs: torch.Tensor,      # [N] échelle per-sample selon T
                y_class: torch.Tensor,     # [N] entiers {0..K-1}
                weight: torch.Tensor = None) -> torch.Tensor:

        scale = inputs.squeeze(-1) if inputs.ndim > 1 else inputs
        device, dtype = scale.device, scale.dtype
        K = self.num_classes

        u0    = self._u0(device, dtype)
        edges = self._edges_tensor(device, dtype)

        # CDF aux bords
        F_u = [ self._cdf_primary(u0.expand_as(scale), scale, device, dtype) ]
        for u in edges:
            F_u.append(self._cdf_primary(u.expand_as(scale), scale, device, dtype))

        # Probas
        Ps = [F_u[0]] + [torch.clamp(F_u[j+1] - F_u[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_u[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)

        # Loss
        if self.L == "entropy":
            logP = torch.log(P.clamp_min(1e-12))
            loss = F.nll_loss(logP, y_class.long(), reduction='none')
        elif self.L == "mcewk":
            loss = MCEAndWKLoss(num_classes=self.num_classes, use_logits=False).forward(P, y_class)
        elif self.L == "bceloss":
            loss = BCELoss(num_classes=self.num_classes).forward(P, y_class)
        else:
            raise ValueError(f"L unknow: {self.L}")

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    # ======== Plot PDF (diag) ========
    def plot_final_pdf(self,
                       scale: torch.Tensor,       # [N] échelle per-sample
                       dir_output,
                       y_max: float = None,
                       num_points: int = 400,
                       dpi: int = 120,
                       stat: str = "median",
                       output_pdf: str = "final_pdf"):
        """
        Trace la PDF de la distribution choisie par T (pas un mélange) avec annotations des edges.
        Le paramètre d'échelle est agrégé (mean/median) sur le batch.
        """
        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)
        device, dtype = scale.device, scale.dtype
        K = self.num_classes

        # Agrégat de l'échelle
        red = torch.median if (stat == "median") else torch.mean
        scale_c = red(self._sp(scale))

        # Paramètres globaux actuels
        if self.force_positive:
            kappa = self._sp(self.kappa); xi = self._sp(self.xi) + self.xi_min
            k     = self._sp(self.k_shape)
        else:
            kappa = self.kappa; xi = self.xi; k = self.k_shape

        # Edges
        u0 = self._sp(self.u0_raw.to(device=device, dtype=dtype))
        if K > 2:
            steps = self._sp(self.t_steps.to(device=device, dtype=dtype))
            edges = u0 + torch.cumsum(steps, dim=0)
            all_edges = torch.cat([u0.view(1), edges], dim=0)
        else:
            all_edges = u0.view(1)

        # y_max heuristique
        if y_max is None:
            ymax = float(all_edges[-1].detach().cpu()) * 2.0
            ymax = max(ymax, float(u0.detach().cpu()) * 1.5 + 1.0, 1.0)
        else:
            ymax = float(y_max)

        y = torch.linspace(0.0, ymax, steps=num_points, device=device, dtype=dtype)

        # PDFs
        def pdf_lognormal(yv, mu_log, sigma_ln, eps=1e-8):
            yv = yv.clamp_min(eps)
            z  = (torch.log(yv) - mu_log) / sigma_ln
            return torch.exp(-0.5 * z * z) / (yv * sigma_ln * math.sqrt(2.0 * math.pi))

        def pdf_gamma_k_theta(yv, k_, theta_, eps=1e-12):
            k_ = k_.clamp_min(eps); theta_ = theta_.clamp_min(eps)
            yv = yv.clamp_min(eps)
            log_pdf = (k_ - 1.0) * torch.log(yv) - (yv / theta_) - torch.lgamma(k_) - k_ * torch.log(theta_)
            return torch.exp(log_pdf)

        def pdf_egpd(yv, sigma_t, xi_v, kappa_v, eps=1e-12):
            yv = yv.clamp_min(0.0)
            si = sigma_t.clamp_min(eps)
            xv = xi_v.clamp_min(eps)
            kv = kappa_v.clamp_min(eps)
            z  = (1.0 + xv * (yv / si)).clamp_min(1.0 + 1e-12)
            H  = (1.0 - torch.pow(z, -1.0 / xv)).clamp(min=eps, max=1.0 - eps)
            h  = (1.0 / si) * torch.pow(z, -1.0 / xv - 1.0)
            return kv * h * torch.pow(H, kv - 1.0)

        if self.T == "egpd":
            f = pdf_egpd(y, scale_c, xi, kappa)
            title = 'EGPD PDF'
        elif self.T == "log":
            f = pdf_lognormal(y, self.mu_log.to(device=device, dtype=dtype), scale_c)
            title = 'LogNormal PDF'
        else:  # gamma
            f = pdf_gamma_k_theta(y, k.to(device=device, dtype=dtype), scale_c)
            title = 'Gamma PDF'

        # Plot
        y_np = y.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(y_np, f_np, label=f'{title}')

        names = [f"u{j}" for j in range(all_edges.numel())]
        for name, uj in zip(names, all_edges):
            xval = float(uj.detach().cpu())
            plt.axvline(x=xval, linestyle='--', alpha=0.55, label=name)
            ymax_txt = (np.nanmax(f_np) * 0.9) if np.isfinite(f_np).any() else 0.9
            plt.text(xval, ymax_txt, f"{name}={xval:.3f}",
                     rotation=90, va='top', ha='right', fontsize=9, alpha=0.8)

        plt.xlim(0, ymax)
        top = (np.nanmax(f_np) * 1.05) if np.isfinite(f_np).any() else 1.0
        plt.ylim(0, top)
        plt.xlabel('y'); plt.ylabel('Density')
        plt.title(title)
        plt.legend(ncol=2)
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.savefig(Path(dir_output) / (output_pdf if output_pdf else "final_pdf.png"), dpi=dpi)
        plt.close()

    # ======== Exposition ========
    def get_learnable_parameters(self):
        """Ne renvoie QUE les paramètres utilisés par la classe (tous sont potentiellement utiles selon T)."""
        params = {
            "u0_raw": self.u0_raw,
            "t_steps": self.t_steps,
        }
        # eGPD (toujours définis; utiles si T='egpd')
        params.update({"kappa": self.kappa, "xi": self.xi})
        # LogNormal
        params.update({"mu_log": self.mu_log})
        # Gamma
        params.update({"k_shape": self.k_shape})
        return params

    def get_attribute(self):
        """Valeurs courantes (contraintes si pertinent) pour affichage/debug."""
        if self.force_positive:
            kappa_v = self._sp(self.kappa)
            xi_v    = self._sp(self.xi) + self.xi_min
            k_v     = self._sp(self.k_shape)
        else:
            kappa_v = self.kappa
            xi_v    = self.xi
            k_v     = self.k_shape
        u0_v = self._sp(self.u0_raw)

        attrs = [('u0', u0_v), ('kappa', kappa_v), ('xi', xi_v), ('mu_log', self.mu_log), ('k_shape', k_v), ('T', self.T), ('L', self.L)]
        for i, u in enumerate(self.get_edges(), start=1):
            attrs.append((f'u{i}', u))
        return attrs

    # ======== MAJ directe (inchangée) ========
    def update_params(self, new_values: dict, strict: bool = False):
        learnables = self.get_learnable_parameters()
        updated = []

        def to_tensor_like(x, ref: torch.Tensor):
            t = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
            if t.ndim == 0 and ref.numel() > 1:
                t = t.expand_as(ref)
            if t.shape != ref.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{name}': got {tuple(t.shape)}, expected {tuple(ref.shape)}")
                try:
                    t = t.expand_as(ref)
                except Exception:
                    return None
            return t

        for name, value in new_values.items():
            if name not in learnables:
                if strict:
                    raise KeyError(f"Unknown learnable parameter '{name}'.")
                else:
                    continue
            param = learnables[name]
            if not isinstance(param, torch.nn.Parameter):
                if strict:
                    raise TypeError(f"Object mapped by '{name}' is not an nn.Parameter.")
                else:
                    continue
            t = to_tensor_like(value, param.data)
            if t is None:
                if strict:
                    raise ValueError(f"Incompatible value for '{name}'.")
                else:
                    continue
            param.data.copy_(t)
            updated.append(name)
        return updated

class IntervalCELoss(nn.Module):
    """
    Interval CE sur une seule distribution choisie par T ∈ {'egpd','log','gamma'}.
    Les probabilités par classe sont obtenues par différences de CDF aux bords appris:
        P0 = F(u0)
        Pj = F(uj) - F(uj-1) pour j=1..K-2
        PK-1 = 1 - F(uK-1)
    Edges TOUJOURS apprenables (u0 + cumsum(softplus(t_steps))).

    Entrées du forward:
        inputs: Tensor [N] — paramètre d'échelle par échantillon selon T:
            - T='egpd'  -> sigma_tail_i
            - T='log'   -> sigma_ln_i
            - T='gamma' -> theta_i (échelle)
        y_class: [N] entiers dans {0..K-1}

    Choix de loss via L:
        - 'entropy' : NLL sur log(P)
        - 'mcewk'   : MCE + Wasserstein-K (nécessite MCEAndWKLoss)
        - 'bceloss' : BCE multi-classes sur P (nécessite BCELoss)
    """

    def __init__(self,
                 num_classes: int = 5,            # classes 0..K-1
                 # eGPD (si T='egpd')
                 kappa: float = 2.5,
                 xi: float = 1.5,
                 # LogNormal (si T='log')
                 mu_log: float = 0.0,             # moyenne du log (globale)
                 # Gamma (si T='gamma')
                 k_shape_init: float = 2.0,       # forme globale k>0
                 # Edges
                 u0_init: float = 0.5,
                 # Divers
                 eps: float = 1e-8,
                 L: str = "entropy",
                 reduction: str = "mean",
                 force_positive: bool = False,
                 T: str = "egpd",                 # 'egpd' | 'log' | 'gamma'
                 xi_min: float = 1e-12):
        super().__init__()
        assert num_classes >= 2, "num_classes>=2 requis"
        assert T in {"egpd","log","gamma"}, f"T must be in {{'egpd','log','gamma'}}, got {T}"
        assert L in {"entropy","mcewk","bceloss", "gwdl"}, f"L must be in {{'entropy','mcewk','bceloss', 'gwdl}}, got {L}"

        self.num_classes   = num_classes
        self.eps           = eps
        self.reduction     = reduction
        self.force_positive= force_positive
        self.L             = L
        self.T             = T
        self.xi_min        = xi_min

        # --- Paramètres globaux selon T ---
        # eGPD
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))
        self.xi    = nn.Parameter(torch.tensor(float(xi)))
        # LogNormal
        self.mu_log = nn.Parameter(torch.tensor(float(mu_log)))
        # Gamma (forme k globale)
        self.k_shape = nn.Parameter(torch.tensor(float(k_shape_init)))

        # --- Edges appris ---
        self.u0_raw  = nn.Parameter(torch.tensor(float(u0_init)))
        # t_steps de taille K-2 -> edges = u0 + cumsum(softplus(t_steps))
        self.t_steps = nn.Parameter(torch.as_tensor(np.arange(1, self.num_classes - 1), dtype=torch.float32))

    # ======== Edges ========
    def _sp(self, x):  # softplus stable
        return F.softplus(x) + 1e-12

    def _u0(self, device, dtype):
        return self._sp(self.u0_raw).to(device=device, dtype=dtype)

    def _edges_tensor(self, device, dtype):
        steps = self._sp(self.t_steps.to(device=device, dtype=dtype))     # (K-2,) > 0
        u0    = self._u0(device, dtype)                                   # scalaire >= 0
        return u0 + torch.cumsum(steps, dim=0)                            # (K-1,)

    def get_edges(self, use_sp=True):
        with torch.no_grad():
            steps = self._sp(self.t_steps) if use_sp else self.t_steps
            u0    = self._sp(self.u0_raw) if use_sp else self.u0_raw
            edges = (u0 + torch.cumsum(steps, dim=0))
        return tuple(x for x in edges)

    # ======== CDFs ========
    @staticmethod
    def _cdf_lognormal(u: torch.Tensor, mu_log: torch.Tensor, sigma_ln: torch.Tensor, eps: float) -> torch.Tensor:
        # F_LN(u) = Phi((ln u - mu)/sigma), u>0 ; sinon 0
        pos = (u > 0)
        z = (torch.log(u.clamp(min=eps)) - mu_log) / (sigma_ln * math.sqrt(2.0))
        F_ln = 0.5 * (1.0 + torch.erf(z))
        return torch.where(pos, F_ln, torch.zeros_like(F_ln))

    def _cdf_gamma_k_theta(self, u: torch.Tensor, k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        CDF Gamma(k, θ) = P(k, u/θ).
        NOTE: PyTorch n'a pas de gradient w.r.t k dans igamma -> on DÉTACHE k ici.
        Si tu veux apprendre k, remplace par une intégration numérique trapèzes.
        """
        k_det = k.clamp_min(self.eps).detach()
        theta = theta.clamp_min(self.eps)
        x = (u / theta).clamp_min(0.0)
        try:
            Pkx = torch.special.gammainc(k_det, x)  # regularized lower
        except AttributeError:
            Pkx = torch.igamma(k_det, x) / torch.exp(torch.lgamma(k_det))
        return Pkx

    def _cdf_primary(self, u: torch.Tensor, scale: torch.Tensor, device, dtype) -> torch.Tensor:
        """
        Retourne F(u) de la distribution choisie par T, avec 'scale' per-sample.
        """
        if self.T == "egpd":
            kappa = self._sp(self.kappa).to(device=device, dtype=dtype)
            xi    = (self._sp(self.xi) + self.xi_min).to(device=device, dtype=dtype) if self.force_positive else self.xi.to(device, dtype)
            return cdf_egpd_family1(u.clamp_min(0.0), scale, xi, kappa, eps=1e-12)
        elif self.T == "log":
            mu_log = self.mu_log.to(device=device, dtype=dtype)
            return self._cdf_lognormal(u, mu_log, scale, self.eps)
        else:  # "gamma"
            k = self._sp(self.k_shape) if self.force_positive else self.k_shape
            return self._cdf_gamma_k_theta(u, k.to(device=device, dtype=dtype), scale)

    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output: Path, output_pdf: str) -> torch.Tensor:
        """
        inputs: [N] échelles per-sample, interprétées selon T (voir doc classe).
        Retourne P ∈ [N,K].
        """
        scale = inputs.squeeze(-1) if inputs.ndim > 1 else inputs
        device, dtype = scale.device, scale.dtype
        K = self.num_classes
        
        scale = self._sp(scale)

        # plot diag
        self.plot_final_pdf(scale, dir_output, output_pdf=output_pdf, stat="median")

        # Edges
        u0    = self._u0(device, dtype)
        edges = self._edges_tensor(device, dtype)

        # CDF aux bords
        F_u = [ self._cdf_primary(u0.expand_as(scale), scale, device, dtype) ]
        for u in edges:
            F_u.append(self._cdf_primary(u.expand_as(scale), scale, device, dtype))

        # Probas par différences
        Ps = [F_u[0]] + [torch.clamp(F_u[j+1] - F_u[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_u[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)
        return P

    # ======== Forward (loss) ========
    def forward(self,
                inputs: torch.Tensor,      # [N] échelle per-sample selon T
                y_class: torch.Tensor,     # [N] entiers {0..K-1}
                weight: torch.Tensor = None) -> torch.Tensor:

        scale = inputs.squeeze(-1) if inputs.ndim > 1 else inputs
        device, dtype = scale.device, scale.dtype
        K = self.num_classes

        scale = self._sp(scale)

        edges = self._edges_tensor(device, dtype)

        # Edges
        u0    = self._u0(device, dtype)
        edges = self._edges_tensor(device, dtype)

        # CDF aux bords
        F_u = [ self._cdf_primary(u0.expand_as(scale), scale, device, dtype) ]
        for u in edges:
            F_u.append(self._cdf_primary(u.expand_as(scale), scale, device, dtype))

        # Probas
        Ps = [F_u[0]] + [torch.clamp(F_u[j+1] - F_u[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_u[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)

        # Loss
        if self.L == "entropy":
            logP = torch.log(P.clamp_min(1e-12))
            loss = F.nll_loss(logP, y_class.long(), reduction='none')
        elif self.L == "mcewk":
            loss = MCEAndWKLoss(num_classes=self.num_classes, use_logits=False).forward(P, y_class)
        elif self.L == "bceloss":
            loss = BCELoss(num_classes=self.num_classes).forward(P, y_class)
        elif self.L == "gwdl":
            loss = GeneralizedWassersteinDiceLoss(type_pen='over').forward(P, y_class)
        else:
            raise ValueError(f"L unknow: {self.L}")

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    # ======== Plot PDF (diag) ========
    def plot_final_pdf(self,
                       scale: torch.Tensor,       # [N] échelle per-sample
                       dir_output,
                       y_max: float = None,
                       num_points: int = 400,
                       dpi: int = 120,
                       stat: str = "median",
                       output_pdf: str = "final_pdf"):
        """
        Trace la PDF de la distribution choisie par T (pas un mélange) avec annotations des edges.
        Le paramètre d'échelle est agrégé (mean/median) sur le batch.
        """
        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)
        device, dtype = scale.device, scale.dtype
        K = self.num_classes

        # Agrégat de l'échelle
        red = torch.median if (stat == "median") else torch.mean
        scale_c = red(self._sp(scale))

        kappa = self._sp(self.kappa); 
        # Paramètres globaux actuels
        if self.force_positive:
            xi = self._sp(self.xi) + self.xi_min
        else:
            xi = self.xi

        # Edges
        u0 = self._sp(self.u0_raw.to(device=device, dtype=dtype))
        if K > 2:
            steps = self._sp(self.t_steps.to(device=device, dtype=dtype))
            edges = u0 + torch.cumsum(steps, dim=0)
            all_edges = torch.cat([u0.view(1), edges], dim=0)
        else:
            all_edges = u0.view(1)

        # y_max heuristique
        if y_max is None:
            ymax = float(all_edges[-1].detach().cpu()) * 2.0
            ymax = max(ymax, float(u0.detach().cpu()) * 1.5 + 1.0, 1.0)
        else:
            ymax = float(y_max)

        y = torch.linspace(0.0, ymax, steps=num_points, device=device, dtype=dtype)

        # PDFs
        def pdf_lognormal(yv, mu_log, sigma_ln, eps=1e-8):
            yv = yv.clamp_min(eps)
            z  = (torch.log(yv) - mu_log) / sigma_ln
            return torch.exp(-0.5 * z * z) / (yv * sigma_ln * math.sqrt(2.0 * math.pi))

        def pdf_gamma_k_theta(yv, k_, theta_, eps=1e-12):
            k_ = k_.clamp_min(eps); theta_ = theta_.clamp_min(eps)
            yv = yv.clamp_min(eps)
            log_pdf = (k_ - 1.0) * torch.log(yv) - (yv / theta_) - torch.lgamma(k_) - k_ * torch.log(theta_)
            return torch.exp(log_pdf)

        def pdf_egpd(yv, sigma_t, xi_v, kappa_v, eps=1e-12):
            yv = yv.clamp_min(0.0)
            si = sigma_t.clamp_min(eps)
            xv = xi_v.clamp_min(eps)
            kv = kappa_v.clamp_min(eps)
            z  = (1.0 + xv * (yv / si)).clamp_min(1.0 + 1e-12)
            H  = (1.0 - torch.pow(z, -1.0 / xv)).clamp(min=eps, max=1.0 - eps)
            h  = (1.0 / si) * torch.pow(z, -1.0 / xv - 1.0)
            return kv * h * torch.pow(H, kv - 1.0)

        if self.T == "egpd":
            f = pdf_egpd(y, scale_c, xi, kappa)
            title = 'EGPD PDF'
        elif self.T == "log":
            f = pdf_lognormal(y, self.mu_log.to(device=device, dtype=dtype), scale_c)
            title = 'LogNormal PDF'
        else:  # gamma
            f = pdf_gamma_k_theta(y, k.to(device=device, dtype=dtype), scale_c)
            title = 'Gamma PDF'

        # Plot
        y_np = y.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(y_np, f_np, label=f'{title}')

        names = [f"u{j}" for j in range(all_edges.numel())]
        for name, uj in zip(names, all_edges):
            xval = float(uj.detach().cpu())
            plt.axvline(x=xval, linestyle='--', alpha=0.55, label=name)
            ymax_txt = (np.nanmax(f_np) * 0.9) if np.isfinite(f_np).any() else 0.9
            plt.text(xval, ymax_txt, f"{name}={xval:.3f}",
                     rotation=90, va='top', ha='right', fontsize=9, alpha=0.8)

        plt.xlim(0, ymax)
        top = (np.nanmax(f_np) * 1.05) if np.isfinite(f_np).any() else 1.0
        plt.ylim(0, top)
        plt.xlabel('y'); plt.ylabel('Density')
        plt.title(title)
        plt.legend(ncol=2)
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.savefig(Path(dir_output) / (output_pdf if output_pdf else "final_pdf.png"), dpi=dpi)
        plt.close()

    # ======== Exposition ========
    def get_learnable_parameters(self):
        """Ne renvoie QUE les paramètres utilisés par la classe (tous sont potentiellement utiles selon T)."""
        params = {
            "u0_raw": self.u0_raw,
            "t_steps": self.t_steps,
        }
        # eGPD (toujours définis; utiles si T='egpd')
        params.update({"kappa": self.kappa, "xi": self.xi})
        # LogNormal
        params.update({"mu_log": self.mu_log})
        # Gamma
        params.update({"k_shape": self.k_shape})
        return params

    def get_attribute(self):
        """Valeurs courantes (contraintes si pertinent) pour affichage/debug."""
        kappa_v = self.kappa
        xi_v    = self.xi
        k_v     = self.k_shape

        attrs = [('kappa', kappa_v), ('xi', xi_v), ('mu_log', self.mu_log), ('k_shape', k_v), ('u0_raw', self.u0_raw), ('t_steps', self.t_steps)]
        
        return attrs

    # ======== MAJ directe (inchangée) ========
    def update_params(self, new_values: dict, strict: bool = False):
        learnables = self.get_learnable_parameters()
        updated = []

        def to_tensor_like(x, ref: torch.Tensor):
            t = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
            if t.ndim == 0 and ref.numel() > 1:
                t = t.expand_as(ref)
            if t.shape != ref.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{name}': got {tuple(t.shape)}, expected {tuple(ref.shape)}")
                try:
                    t = t.expand_as(ref)
                except Exception:
                    return None
            return t

        for name, value in new_values.items():
            if name not in learnables:
                if strict:
                    raise KeyError(f"Unknown learnable parameter '{name}'.")
                else:
                    continue
            param = learnables[name]
            if not isinstance(param, torch.nn.Parameter):
                if strict:
                    raise TypeError(f"Object mapped by '{name}' is not an nn.Parameter.")
                else:
                    continue
            t = to_tensor_like(value, param.data)
            if t is None:
                if strict:
                    raise ValueError(f"Incompatible value for '{name}'.")
                else:
                    continue
            param.data.copy_(t)
            updated.append(name)
        return updated
    
    def plot_params(self, logs, dir_output, dpi=120):
        """
        Trace l’évolution des paramètres (si `logs` fournis) ET superpose la valeur *courante*
        du modèle sur chaque plot existant.

        Attentes (selon T):
        logs: liste de dicts avec au moins 'epoch'.
            - 'u0'                : scalaire
            - 'edges'             : (K-2,)
            - selon T:
                * T='log'   : 'mu_log'
                * T='gamma' : 'k_shape'
                * T='egpd'  : 'kappa', 'xi'
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        out = Path(dir_output); out.mkdir(parents=True, exist_ok=True)

        def to_np(x):
            import torch
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            return np.asarray(x)
        
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # Valeurs courantes
        cur_u0    = float(to_np(self._sp(self.u0_raw)))
        if self.num_classes > 2:
            cur_edges = to_np(self._sp(self.u0_raw) + torch.cumsum(self._sp(self.t_steps.to(device=device, dtype=dtype)), dim=0))
        else:
            cur_edges = np.empty((0,), dtype=float)

        cur_mu_log = None
        cur_k_shape = None
        cur_kappa = None
        cur_xi = None

        if self.T == "log":
            cur_mu_log = float(to_np(self.mu_log))
        elif self.T == "gamma":
            k = self._sp(self.k_shape) if self.force_positive else self.k_shape
            cur_k_shape = float(to_np(k))
        elif self.T == "egpd":
            kappa = self._sp(self.kappa)
            xi    = (self._sp(self.xi) + self.xi_min) if self.force_positive else self.xi
            cur_kappa = float(to_np(kappa))
            cur_xi    = float(to_np(xi))

        # -------- A) Avec logs --------
        if logs:
            epochs = [int(l["epoch"]) for l in logs]

            def seq(key):
                if key not in logs[0]: return None
                vals = [l[key] for l in logs]
                try:
                    return np.asarray([float(v) for v in vals])
                except Exception:
                    arrs = [to_np(v) for v in vals]
                    try:
                        return np.stack(arrs, 0)
                    except Exception:
                        return np.asarray(arrs)

            # u0
            u0_seq = seq("u0")
            if u0_seq is not None:
                u0_seq = F.softplus(torch.as_tensor(u0_seq, dtype=torch.float32)).detach().cpu().numpy()
                plt.figure(figsize=(8.2,4.6))
                plt.plot(epochs, u0_seq, marker='o', label='u0 (train)')
                plt.axhline(cur_u0, linestyle=':', alpha=0.9, label=f'u0 current={cur_u0:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('u0'); plt.title('u0')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(out/'u0.png', dpi=dpi); plt.close()

            # edges
            edges_seq = seq("edges")
            if edges_seq is not None and edges_seq.ndim == 2 and edges_seq.shape[1] > 0:
                plt.figure(figsize=(10,6))
                for j in range(edges_seq.shape[1]):
                    plt.plot(epochs, edges_seq[:, j], marker='^', linestyle='--', label=f'u{j+1} (train)')
                    if j < len(cur_edges):
                        plt.scatter([epochs[-1]], [cur_edges[j]], s=36, marker='o', alpha=0.9)
                plt.xlabel('Epoch'); plt.ylabel('edge'); plt.title('edges')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                plt.tight_layout(); plt.savefig(out/'edges.png', dpi=dpi); plt.close()

            # selon T
            if self.T == "log":
                mu_seq = seq("mu_log")
                if mu_seq is not None:
                    plt.figure(figsize=(8.2,4.6))
                    plt.plot(epochs, mu_seq, marker='o', label='mu_log (train)')
                    plt.axhline(cur_mu_log, linestyle=':', alpha=0.9, label=f'current={cur_mu_log:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('mu_log'); plt.title('bulk log')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(out/'mu_log.png', dpi=dpi); plt.close()
            elif self.T == "gamma":
                k_seq = seq("k_shape")
                if k_seq is not None:
                    if getattr(self, "force_positive", True):
                        k_seq = F.softplus(torch.as_tensor(k_seq, dtype=torch.float32)).detach().cpu().numpy()
                    plt.figure(figsize=(8.2,4.6))
                    plt.plot(epochs, k_seq, marker='o', label='k_shape (train)')
                    plt.axhline(cur_k_shape, linestyle=':', alpha=0.9, label=f'current={cur_k_shape:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('k_shape'); plt.title('bulk gamma')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(out/'k_shape.png', dpi=dpi); plt.close()
            elif self.T == "egpd":
                kappa_seq = seq("kappa"); xi_seq = seq("xi")
                if kappa_seq is not None or xi_seq is not None:
                    plt.figure(figsize=(8.8,5.0))
                    if kappa_seq is not None:
                        kappa_seq = F.softplus(torch.as_tensor(kappa_seq, dtype=torch.float32)).detach().cpu().numpy()
                        plt.plot(epochs, kappa_seq, marker='o', label='kappa (train)')
                        plt.axhline(cur_kappa, linestyle=':', alpha=0.9, label=f'kappa current={cur_kappa:.4g}')
                    if xi_seq is not None:
                        if getattr(self, "force_positive", True):
                            xi_seq = F.softplus(torch.as_tensor(xi_seq, dtype=torch.float32)).detach().cpu().numpy()
                        plt.plot(epochs, xi_seq, marker='s', label='xi (train)')
                        plt.axhline(cur_xi, linestyle=':', alpha=0.9, label=f'xi current={cur_xi:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('value'); plt.title('egpd tail')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(out/'egpd.png', dpi=dpi); plt.close()

        # -------- B) Snapshot seul --------
        else:
            plt.figure(); plt.axhline(cur_u0, linestyle=':', alpha=0.9, label=f'u0={cur_u0:.4g}')
            plt.title('u0 (current)'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout(); plt.savefig(out/'u0_current.png', dpi=dpi); plt.close()

            if cur_edges.size > 0:
                plt.figure(); plt.bar(range(1, cur_edges.size+1), cur_edges)
                plt.xlabel('edge index'); plt.title('edges (current)')
                plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
                plt.savefig(out/'edges_current.png', dpi=dpi); plt.close()

            if self.T == "log" and cur_mu_log is not None:
                plt.figure(); plt.axhline(cur_mu_log, linestyle=':', alpha=0.9, label=f'mu_log={cur_mu_log:.4g}')
                plt.title('mu_log current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'mu_log_current.png', dpi=dpi); plt.close()
            if self.T == "gamma" and cur_k_shape is not None:
                plt.figure(); plt.axhline(cur_k_shape, linestyle=':', alpha=0.9, label=f'k_shape={cur_k_shape:.4g}')
                plt.title('k_shape current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'k_shape_current.png', dpi=dpi); plt.close()
            if self.T == "egpd":
                plt.figure()
                if cur_kappa is not None: plt.axhline(cur_kappa, linestyle=':', alpha=0.9, label=f'kappa={cur_kappa:.4g}')
                if cur_xi is not None:    plt.axhline(cur_xi, linestyle=':', alpha=0.9, label=f'xi={cur_xi:.4g}')
                plt.title('egpd params current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'egpd_current.png', dpi=dpi); plt.close()

class IntervalCEClusterIDs(nn.Module):
    """
    Interval CE via CDF d’une distribution PRIMAIRE (T ∈ {'egpd','log','gamma'})
    avec paramètres (par cluster OU global+delta) et edges par cluster.

    Entrées:
      - forward(scale, y_class, clusters_ids, weight=None)
        * scale: paramètre d'échelle par échantillon
            T='egpd'  -> sigma_tail_i
            T='log'   -> sigma_ln_i (écart-type en log)
            T='gamma' -> theta_i (échelle)
        * y_class ∈ {0..K-1}
        * clusters_ids ∈ {0..C-1}

    P(k) créés par différences de CDF aux bords appris: u0_c + cumsum(softplus(t_steps_c,:))
    """

    def __init__(self,
                 NC: int,
                 num_classes: int = 5,              # classes 0..K-1
                 id: int = 0,

                 # ---- Choix distribution & loss ----
                 T: str = "egpd",                   # 'egpd' | 'log' | 'gamma'
                 L: str = "entropy",                # 'entropy' | 'mcewk' | 'bceloss'

                 # ---- Paramètres init selon T ----
                 # eGPD
                 kappa_init: float = 0.831,
                 xi_init: float    = 0.161,
                 # LogNormal
                 mu_log_init: float = 0.0,
                 # Gamma
                 k_shape_init: float = 2.0,

                 # ---- Partage par cluster vs global+delta ----
                 G: bool = False,                   # False: param par cluster; True: global+delta (par cluster)
                                                    # (applique au jeu de paramètres de T)

                 # ---- Edges ----
                 u0_init: float = 0.5,              # u0 par cluster (softplus)
                 # ---- Divers ----
                 eps: float = 1e-8,
                 xi_min: float = 1e-12,
                 reduction: str = "mean",
                 force_positive: bool = False):
        
        super().__init__()
        assert num_classes >= 2, f"num_classes>=2 requis, got {num_classes}"
        assert T in {"egpd","log","gamma"}
        assert L in {"entropy","mcewk","bceloss"}

        self.n_clusters     = NC
        self.num_classes    = num_classes
        self.T              = T
        self.L              = L
        self.add_global     = G
        self.eps            = eps
        self.xi_min         = xi_min
        self.reduction      = reduction
        self.force_positive = force_positive
        self.id             = id

        # ---- Paramètres du modèle selon T ----
        if T == "egpd":
            if not G:
                self.kappa_raw = nn.Parameter(torch.full((NC,), float(kappa_init)))
                self.xi_raw    = nn.Parameter(torch.full((NC,), float(xi_init)))
            else:
                self.kappa_global = nn.Parameter(torch.tensor(float(kappa_init)))
                self.xi_global    = nn.Parameter(torch.tensor(float(xi_init)))
                self.kappa_delta  = nn.Parameter(torch.zeros(NC))
                self.xi_delta     = nn.Parameter(torch.zeros(NC))
        elif T == "log":
            if not G:
                self.mu_log_raw = nn.Parameter(torch.full((NC,), float(mu_log_init)))
            else:
                self.mu_log_global = nn.Parameter(torch.tensor(float(mu_log_init)))
                self.mu_log_delta  = nn.Parameter(torch.zeros(NC))

        else:  # "gamma"
            if not G:
                self.k_shape_raw = nn.Parameter(torch.full((NC,), float(k_shape_init)))
            else:
                self.k_shape_global = nn.Parameter(torch.tensor(float(k_shape_init)))
                self.k_shape_delta  = nn.Parameter(torch.zeros(NC))

        # ---- u0 & edges par cluster ----
        self.u0_raw  = nn.Parameter(torch.full((NC,), float(u0_init)))      # (C,)
        # --- EDGES PAR CLUSTER ---
        base_steps = torch.arange(1, num_classes - 1, dtype=torch.float32)  # [1, 2, ..., K-2]
        self.t_steps = nn.Parameter(base_steps.unsqueeze(0).repeat(NC, 1))  # (C, K-2)

    # -------------- utils --------------
    def _sp(self, x):
        return F.softplus(x) + 1e-12

    def _u0(self, device=None, dtype=None):
        """u0 contraint >=0, shape (C,)"""
        u0 = self._sp(self.u0_raw) if self.force_positive else self.u0_raw
        if device is not None: u0 = u0.to(device)
        if dtype  is not None: u0 = u0.to(dtype)
        return u0  # (C,)

    def _edges_all_clusters(self, device, dtype):
        """
        Retourne edges pour TOUS les clusters, shape (C, K-1),
        où edges[c] = (u_{c,1},...,u_{c,K-1})
        """
        steps = self._sp(self.t_steps.to(device=device, dtype=dtype)) if self.force_positive else self.t_steps.to(device=device, dtype=dtype)
        u0    = self._u0(device, dtype)                               # (C,)
        edges = u0.unsqueeze(1) + torch.cumsum(steps, dim=1)          # (C, K-1)
        return edges

    def _edges_for_samples(self, clusters_ids: torch.Tensor, device, dtype):
        """Edges du cluster de chaque sample: shape (N, K-1)"""
        all_edges = self._edges_all_clusters(device, dtype)           # (C, K-1)
        return all_edges[clusters_ids.long()]                          # (N, K-1)

    # --- sélection des params par cluster selon T ---
    def _select_primary_params(self, clusters_ids: torch.Tensor):
        cid = clusters_ids.long()
        if self.T == "egpd":
            if not self.add_global:
                k_raw = self.kappa_raw[cid]
                x_raw = self.xi_raw[cid]
            else:
                k_raw = self.kappa_global + self.kappa_delta[cid]
                x_raw = self.xi_global    + self.xi_delta[cid]
            kappa = self._sp(k_raw)
            if self.force_positive:
                xi    = self._sp(x_raw) + self.xi_min
            else:
                xi = x_raw
            return ("egpd", kappa, xi)

        elif self.T == "log":
            if not self.add_global:
                mu = self.mu_log_raw[cid]
            else:
                mu = self.mu_log_global + self.mu_log_delta[cid]
            return ("log", mu, None)

        else:  # gamma
            if not self.add_global:
                k_raw = self.k_shape_raw[cid]
            else:
                k_raw = self.k_shape_global + self.k_shape_delta[cid]
            if self.force_positive:
                k = self._sp(k_raw)
            else:
                k = k_raw
            return ("gamma", k, None)

    # --- CDFs élémentaires ---
    @staticmethod
    def _cdf_lognormal(u: torch.Tensor, mu_log: torch.Tensor, sigma_ln: torch.Tensor, eps: float) -> torch.Tensor:
        pos = (u > 0)
        z = (torch.log(u.clamp(min=eps)) - mu_log) / (sigma_ln * math.sqrt(2.0))
        F_ln = 0.5 * (1.0 + torch.erf(z))
        return torch.where(pos, F_ln, torch.zeros_like(F_ln))

    def _cdf_gamma_k_theta(self, u: torch.Tensor, k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        CDF Gamma(k, θ) = P(k, u/θ). Détache k pour éviter NotImplementedError autograd.
        """
        k_det = k.clamp_min(self.eps).detach()
        theta = theta.clamp_min(self.eps)
        x = (u / theta).clamp_min(0.0)
        try:
            Pkx = torch.special.gammainc(k_det, x)
        except AttributeError:
            Pkx = torch.igamma(k_det, x) / torch.exp(torch.lgamma(k_det))
        return Pkx

    # --- CDF primaire par échantillon ---
    def _cdf_primary(self, u: torch.Tensor, scale: torch.Tensor, clusters_ids: torch.Tensor) -> torch.Tensor:
        tag, a, b = self._select_primary_params(clusters_ids)  # a,b dépendent de T
        clusters_ids = clusters_ids.long()
        if tag == "egpd":
            res = torch.zeros_like(u)
            for cluster in torch.unique(clusters_ids):
                mask = clusters_ids == cluster
                res[mask] = cdf_egpd_family1(u[cluster].clamp_min(0.0), scale[mask], b[mask], a[mask], eps=1e-12)
            return res
        elif tag == "log":
            mu_log   = a
            return self._cdf_lognormal(u, mu_log, scale, self.eps)
        else:  # gamma
            k     = a
            return self._cdf_gamma_k_theta(u, k, scale)

    # -------- API d’inspection --------
    def get_edges_per_cluster(self):
        """Tensor (C, K-1) des edges COURANTS (CPU, float32)"""
        with torch.no_grad():
            edges = self._edges_all_clusters(device='cpu', dtype=torch.float32)
        return edges

    def get_learnable_parameters(self):
        """
        Ne renvoie que les paramètres utilisés par cette classe, selon T et add_global,
        + u0_raw et t_steps.
        """
        out = {"u0_raw": self.u0_raw, "t_steps": self.t_steps}
        if self.T == "egpd":
            if not self.add_global:
                out.update({"kappa_raw": self.kappa_raw, "xi_raw": self.xi_raw})
            else:
                out.update({
                    "kappa_global": self.kappa_global, "xi_global": self.xi_global,
                    "kappa_delta":  self.kappa_delta,  "xi_delta":  self.xi_delta,
                })
        elif self.T == "log":
            if not self.add_global:
                out.update({"mu_log_raw": self.mu_log_raw})
            else:
                out.update({"mu_log_global": self.mu_log_global, "mu_log_delta": self.mu_log_delta})
        else:  # gamma
            if not self.add_global:
                out.update({"k_shape_raw": self.k_shape_raw})
            else:
                out.update({"k_shape_global": self.k_shape_global, "k_shape_delta": self.k_shape_delta})
        return out

    def get_attribute(self):
        """
        Particularité demandée: **NE PAS** appliquer de softplus ici.
        On renvoie les TENSEURS BRUTS (raw), utiles pour du debug direct.
        """
        attrs = []
        # u0_raw & t_steps bruts
        attrs.append(("u0_raw", self.u0_raw.detach().clone()))
        attrs.append(("t_steps", self.t_steps.detach().clone()))

        if self.T == "egpd":
            if not self.add_global:
                attrs.append(("kappa_raw", self.kappa_raw.detach().clone()))
                attrs.append(("xi_raw",    self.xi_raw.detach().clone()))
            else:
                attrs.append(("kappa_global", self.kappa_global.detach().clone()))
                attrs.append(("xi_global",    self.xi_global.detach().clone()))
                attrs.append(("kappa_delta",  self.kappa_delta.detach().clone()))
                attrs.append(("xi_delta",     self.xi_delta.detach().clone()))
        elif self.T == "log":
            if not self.add_global:
                attrs.append(("mu_log_raw", self.mu_log_raw.detach().clone()))
            else:
                attrs.append(("mu_log_global", self.mu_log_global.detach().clone()))
                attrs.append(("mu_log_delta",  self.mu_log_delta.detach().clone()))
        else:  # gamma
            if not self.add_global:
                attrs.append(("k_shape_raw", self.k_shape_raw.detach().clone()))
            else:
                attrs.append(("k_shape_global", self.k_shape_global.detach().clone()))
                attrs.append(("k_shape_delta",  self.k_shape_delta.detach().clone()))
        return attrs

    # -------- forward (loss) --------
    def forward(self,
                scale: torch.Tensor,        # [N] échelle per-sample
                y_class: torch.Tensor,      # [N] in {0..K-1}
                clusters_ids: torch.Tensor,  # [N] in {0..C-1}
                weight: torch.Tensor = None) -> torch.Tensor:

        scale = scale.squeeze(-1) if scale.ndim > 1 else scale
        device, dtype = scale.device, scale.dtype
        K = self.num_classes

        # Edges par sample: (N, K-1) ; u0 par sample: (N,)
        edges_N = self._edges_for_samples(clusters_ids, device=device, dtype=dtype)  # (N, K-1)
        u0_vecC = self._u0(device=device, dtype=dtype)                               # (C,)
        u0_N    = u0_vecC[clusters_ids.long()]                                        # (N,)

        # CDF aux bords (selon T)
        F_list = [ self._cdf_primary(u0_N, scale, clusters_ids) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges_N[:, j], scale, clusters_ids))

        # Probas par différences
        Ps = [F_list[0]] + [torch.clamp(F_list[j+1] - F_list[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_list[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)                                 # (N, K)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)

        # Loss
        if self.L == "entropy":
            logP = torch.log(P.clamp_min(1e-12))
            nll  = F.nll_loss(logP, y_class.long(), reduction='none')
        elif self.L == "mcewk":
            nll  = MCEAndWKLoss(num_classes=self.num_classes, use_logits=False).forward(P, y_class)
        elif self.L == "bceloss":
            nll  = BCELoss(num_classes=self.num_classes).forward(P, y_class)
        else:
            raise ValueError(f"L unknow: {self.L}")

        if weight is not None:
            nll = nll * weight

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll

    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, clusters_ids: torch.Tensor,
                dir_output, output_pdf):
        """
        Retourne P ∈ (N,K) (probabilités par classe) pour inspection/inférence.
        Si dir_output est fourni, trace aussi la PDF primaire par cluster.
        """
        inputs = inputs.squeeze(-1) if inputs.ndim > 1 else inputs
        device, dtype = inputs.device, inputs.dtype
        K = self.num_classes

        inputs = self._sp(inputs) + 1e-12

        # --- Plot per-cluster (diagnostic) ---
        self.plot_final_pdf(inputs, clusters_ids,
                            dir_output=dir_output,
                            output_pdf=(output_pdf or "final_pdf"))

        edges_N = self._edges_for_samples(clusters_ids, device=device, dtype=dtype)  # (N, K-1)
        u0_vecC = self._u0(device=device, dtype=dtype)
        u0_N    = u0_vecC[clusters_ids.long()]

        F_list = [ self._cdf_primary(u0_N, inputs, clusters_ids) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges_N[:, j], inputs, clusters_ids))

        Ps = [F_list[0]] + [torch.clamp(F_list[j+1] - F_list[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_list[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)  # (N, K)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)
        return P

    def plot_params(self, logs, dir_output, cluster_names=None, dpi=120):
        """
        Trace l’évolution des paramètres par cluster selon T, + u0 & edges,
        en superposant les valeurs courantes (constrain si force_positive=True).

        IMPORTANT : si self.force_positive est True, on applique aussi les contraintes
        aux séries issues des logs pour les paramètres qui doivent être positifs :
        - egpd : kappa <- softplus(kappa), xi <- softplus(xi) + xi_min
        - gamma: k_shape <- softplus(k_shape)
        (mu_log n’est pas contraint.)
        """
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        import torch

        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)

        def to_np(x):
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            return np.asarray(x)

        # Valeurs "courantes" (overlay), déjà contraintes si besoin
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        cur_u0 = (self._sp(self.u0_raw) if self.force_positive else self.u0_raw).detach().cpu().numpy()  # (C,)
        cur_edges = (self._edges_all_clusters(device, dtype)).detach().cpu().numpy()  # (C, K-1)

        # Paramètres primaires courants (overlay)
        if self.T == "egpd":
            if not self.add_global:
                cur_kappa = (self._sp(self.kappa_raw) if self.force_positive else self.kappa_raw).detach().cpu().numpy()
                cur_xi    = ((self._sp(self.xi_raw)) if self.force_positive else self.xi_raw).detach().cpu().numpy()
            else:
                cur_kappa = (self._sp(self.kappa_global + self.kappa_delta) if self.force_positive else (self.kappa_global + self.kappa_delta)).detach().cpu().numpy()
                cur_xi    = ((self._sp(self.xi_global + self.xi_delta)) if self.force_positive else (self.xi_global + self.xi_delta)).detach().cpu().numpy()
            cur_tag, cur_a, cur_b = "egpd", cur_kappa, cur_xi
        elif self.T == "log":
            cur_mu = (self.mu_log_raw if not self.add_global else (self.mu_log_global + self.mu_log_delta)).detach().cpu().numpy()
            cur_tag, cur_a, cur_b = "log", cur_mu, None
        else:  # gamma
            if not self.add_global:
                cur_k = (self._sp(self.k_shape_raw) if self.force_positive else self.k_shape_raw).detach().cpu().numpy()
            else:
                cur_k = (self._sp(self.k_shape_global + self.k_shape_delta) if self.force_positive else (self.k_shape_global + self.k_shape_delta)).detach().cpu().numpy()
            cur_tag, cur_a, cur_b = "gamma", cur_k, None

        # ----- Avec logs -----
        if logs:
            epochs = [int(l["epoch"]) for l in logs]

            def seq(key):
                if key not in logs[0]: return None
                vals = [log[key] for log in logs]
                try:
                    return np.asarray([float(v) for v in vals])
                except Exception:
                    arrs = [to_np(v) for v in vals]
                    try:
                        return np.stack(arrs, 0)
                    except Exception:
                        return np.asarray(arrs)

            # Séries de base
            U0_arr = seq("u0_raw")        # (T, C)
            E_arr  = seq("t_steps")     # (T, C, K-1)

            # Séries primaires (brutes)
            if self.T == "egpd":
                if self.add_global:
                    K_arr = seq("kappa_delta") + seq("kappa_global")[:, None] # (T, C)
                    X_arr = seq("xi_delta")  + seq("xi_global")[:, None]  # (T, C)
                else:
                    K_arr = seq("kappa_raw")  # (T, C)
                    X_arr = seq("xi_raw")     # (T, C)
                # >>> Contraintes sur LOGS si demandé
                if K_arr is not None:
                    K_arr = torch.as_tensor(K_arr, dtype=torch.float32)
                    K_arr = (F.softplus(K_arr) if self.force_positive else K_arr).numpy()
                if X_arr is not None:
                    X_arr = torch.as_tensor(X_arr, dtype=torch.float32)
                    X_arr = ((F.softplus(X_arr)+self.xi_min) if self.force_positive else X_arr).numpy()
            elif self.T == "gamma":
                if self.add_global:
                    G_arr = seq("k_shape_delta") + seq("k_shape_global")[:, None] # (T, C)
                else:
                    G_arr = seq("k_shape_raw")  # (T, C)
                if G_arr is not None:
                    G_arr = torch.as_tensor(G_arr, dtype=torch.float32)
                    G_arr = (F.softplus(G_arr) if self.force_positive else G_arr).numpy()
            else:  # log
                if self.add_global:
                    M_arr = seq("mu_log_delta") + seq("mu_log_global")[:, None] # (T, C)
                else:
                    M_arr = seq("mu_log_raw")   # (T, C)  # pas de contrainte

            # Tailles / noms
            C = self.n_clusters
            if cluster_names is not None and len(cluster_names) != C:
                raise ValueError(f"cluster_names length ({len(cluster_names)}) != C ({C})")
            xlabels = [f"cl{c}" if cluster_names is None else str(cluster_names[c]) for c in range(C)]
            x = np.arange(C)

            # 1) Un sous-dossier par cluster
            for c in range(C):
                name = xlabels[c]
                cl_dir = out_dir / name
                cl_dir.mkdir(parents=True, exist_ok=True)

                # u0
                if U0_arr is not None:
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, U0_arr[:, c], marker='o', label='u0 (train)')
                    plt.axhline(cur_u0[c], linestyle=':', alpha=0.9, label=f'u0 current={cur_u0[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('u0'); plt.title(f'u0 — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'u0.png', dpi=dpi); plt.close()

                # edges
                if E_arr is not None and E_arr.ndim == 3 and E_arr.shape[2] > 0:
                    plt.figure(figsize=(10,6))
                    Kminus1 = E_arr.shape[2]
                    for j in range(Kminus1):
                        plt.plot(epochs, E_arr[:, c, j], marker='^', linestyle='--', label=f'u{j+1} (train)')
                        plt.scatter([epochs[-1]], [cur_edges[c, j]], s=36, marker='o', alpha=0.9)
                    plt.xlabel('Epoch'); plt.ylabel('edge'); plt.title(f'edges — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                    plt.tight_layout(); plt.savefig(cl_dir/'edges.png', dpi=dpi); plt.close()

                # params primaires selon T
                if self.T == "egpd" and (K_arr is not None or X_arr is not None):
                    plt.figure(figsize=(9,4.8))
                    if K_arr is not None:
                        plt.plot(epochs, K_arr[:, c], marker='o', label='kappa (train, constrained)')
                        plt.axhline(cur_a[c], linestyle=':', alpha=0.9, label=f'kappa cur={cur_a[c]:.4g}')
                    if X_arr is not None:
                        plt.plot(epochs, X_arr[:, c], color='red', marker='s', label='xi (train, constrained)')
                        plt.axhline(cur_b[c], color='red', linestyle=':', alpha=0.9, label=f'xi cur={cur_b[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('value'); plt.title(f'egpd — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                    plt.tight_layout(); plt.savefig(cl_dir/'egpd.png', dpi=dpi); plt.close()

                if self.T == "log" and ('M_arr' in locals()) and (M_arr is not None):
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, M_arr[:, c], marker='o', label='mu_log (train)')
                    plt.axhline(cur_a[c], linestyle=':', alpha=0.9, label=f'mu_log cur={cur_a[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('mu_log'); plt.title(f'lognorm — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'lognorm.png', dpi=dpi); plt.close()

                if self.T == "gamma" and ('G_arr' in locals()) and (G_arr is not None):
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, G_arr[:, c], marker='o', label='k_shape (train, constrained)')
                    plt.axhline(cur_a[c], linestyle=':', alpha=0.9, label=f'k_shape cur={cur_a[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('k_shape'); plt.title(f'gamma — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'gamma.png', dpi=dpi); plt.close()

            # ----- Snapshot sans logs -----
        else:
            C = self.n_clusters
            xlabels = [f"cl{c}" if cluster_names is None else str(cluster_names[c]) for c in range(C)]
            for c in range(C):
                name = xlabels[c]
                cl_dir = out_dir / name; cl_dir.mkdir(parents=True, exist_ok=True)

                # u0 (current)
                plt.figure(figsize=(8,4.4))
                plt.axhline(cur_u0[c], linestyle=':', alpha=0.9, label=f'u0={cur_u0[c]:.4g}')
                plt.title(f'u0 current — {name}'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(cl_dir/'u0_current.png', dpi=dpi); plt.close()

                # edges (current)
                plt.figure(figsize=(9,4.8))
                plt.bar(range(1, cur_edges.shape[1]+1), cur_edges[c])
                plt.xlabel('edge index'); plt.ylabel('value'); plt.title(f'edges current — {name}')
                plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
                plt.savefig(cl_dir/'edges_current.png', dpi=dpi); plt.close()

                # param primaire (current)
                plt.figure(figsize=(8,4.4))
                if self.T == "egpd":
                    plt.axhline(cur_a[c], color='blue', linestyle=':', alpha=0.9, label=f'kappa={cur_a[c]:.4g}')
                    plt.axhline(cur_b[c], color='red', linestyle=':', alpha=0.9, label=f'xi={cur_b[c]:.4g}')
                    plt.title(f'egpd current — {name}')
                elif self.T == "log":
                    plt.axhline(cur_a[c], linestyle=':', alpha=0.9, label=f'mu_log={cur_a[c]:.4g}')
                    plt.title(f'lognorm current — {name}')
                else:
                    plt.axhline(cur_a[c], linestyle=':', alpha=0.9, label=f'k_shape={cur_a[c]:.4g}')
                    plt.title(f'gamma current — {name}')
                plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(cl_dir/'primary_current.png', dpi=dpi); plt.close()

        # =========================
        # C) COURANTS vs CLUSTERS
        # =========================
        C = self.n_clusters
        xlabels = [f"cl{c}" if cluster_names is None else str(cluster_names[c]) for c in range(C)]
        x = np.arange(C)

        # u0 vs clusters
        plt.figure(figsize=(10,4.8))
        plt.bar(x, cur_u0)
        plt.xticks(x, xlabels, rotation=90, ha='right')
        plt.ylabel('u0'); plt.title('Current u0 vs clusters')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig(Path(dir_output)/'current_u0_vs_clusters.png', dpi=dpi); plt.close()

        # edges vs clusters (une courbe par indice d’edge)
        if cur_edges.shape[1] > 0:
            plt.figure(figsize=(11,6))
            for j in range(cur_edges.shape[1]):
                plt.plot(x, cur_edges[:, j], marker='o', label=f'u{j+1}')
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('edge value'); plt.title('Current edges vs clusters')
            plt.grid(True, linestyle='--', alpha=0.35); plt.legend(ncol=2)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_edges_vs_clusters.png', dpi=dpi); plt.close()

        # primaire vs clusters
        if self.T == "egpd":
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_a)  # kappa
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('kappa'); plt.title('Current kappa vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_kappa_vs_clusters.png', dpi=dpi); plt.close()

            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_b)  # xi
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('xi'); plt.title('Current xi vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_xi_vs_clusters.png', dpi=dpi); plt.close()

        elif self.T == "log":
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_a)  # mu_log
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('mu_log'); plt.title('Current mu_log vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_mu_log_vs_clusters.png', dpi=dpi); plt.close()

        else:  # gamma
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_a)  # k_shape
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('k_shape'); plt.title('Current k_shape vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_kshape_vs_clusters.png', dpi=dpi); plt.close()

        # --------------- mise à jour directe ---------------
    def update_params(self, new_values: dict, strict: bool = False):
        learnables = self.get_learnable_parameters()
        updated = []

        def to_tensor_like(x, ref: torch.Tensor):
            t = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
            if t.ndim == 0 and ref.numel() > 1:
                t = t.expand_as(ref)
            if t.shape != ref.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{name}': got {tuple(t.shape)}, expected {tuple(ref.shape)}")
                try:
                    t = t.expand_as(ref)
                except Exception:
                    return None
            return t

        for name, value in new_values.items():
            if name not in learnables:
                if strict: raise KeyError(f"Unknown learnable parameter '{name}'.")
                else: continue
            param = learnables[name]
            if not isinstance(param, torch.nn.Parameter):
                if strict: raise TypeError(f"Object mapped by '{name}' is not an nn.Parameter.")
                else: continue
            t = to_tensor_like(value, param.data)
            if t is None:
                if strict: raise ValueError(f"Incompatible value for '{name}'.")
                else: continue
            param.data.copy_(t); updated.append(name)

        return updated
    
    def plot_final_pdf(self,
                   scale: torch.Tensor,          # (N,) échelle par sample
                   clusters_ids: torch.Tensor,    # (N,)
                   dir_output,
                   y_max: float = None,
                   num_points: int = 400,
                   dpi: int = 120,
                   stat: str = "median",
                   output_pdf: str = "final_pdf"):
        """
        Trace la PDF de la distribution PRIMAIRE (T) pour CHAQUE CLUSTER et
        sauvegarde dans: dir_output/cl{c}/<output_pdf>.png

        - Agrège par cluster la 'scale' per-sample (median ou mean).
        - Utilise les paramètres du cluster (contraints si force_positive=True).
        - Annote les edges (u0, u1, …).
        """
        from pathlib import Path
        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)

        device, dtype = scale.device, scale.dtype
        K = self.num_classes
        red = torch.median if stat == "median" else torch.mean

        # PDFs élémentaires (stables)
        def pdf_lognormal(yv, mu_log, sigma_ln, eps=1e-8):
            yv = yv.clamp_min(eps)
            z  = (torch.log(yv) - mu_log) / sigma_ln
            return torch.exp(-0.5 * z * z) / (yv * sigma_ln * math.sqrt(2.0 * math.pi))

        def pdf_gamma_k_theta(yv, k_, theta_, eps=1e-12):
            k_ = k_.clamp_min(eps); theta_ = theta_.clamp_min(eps)
            yv = yv.clamp_min(eps)
            # log-pdf pour stabilité
            log_pdf = (k_ - 1.0) * torch.log(yv) - (yv / theta_) - torch.lgamma(k_) - k_ * torch.log(theta_)
            return torch.exp(log_pdf)

        def pdf_egpd(yv, sigma_t, xi_v, kappa_v, eps=1e-12):
            yv = yv.clamp_min(0.0)
            si = sigma_t.clamp_min(eps)
            xv = xi_v.clamp_min(eps)
            kv = kappa_v.clamp_min(eps)
            z  = (1.0 + xv * (yv / si)).clamp_min(1.0 + 1e-12)
            H  = (1.0 - torch.pow(z, -1.0 / xv)).clamp(min=eps, max=1.0 - eps)
            h  = (1.0 / si) * torch.pow(z, -1.0 / xv - 1.0)
            return kv * h * torch.pow(H, kv - 1.0)

        # u0 & edges pour annotation
        u0_all = (self._sp(self.u0_raw) if self.force_positive else self.u0_raw).to(device=device, dtype=dtype)  # (C,)
        E_all  = self._edges_all_clusters(device, dtype)  # (C, K-1)  (cumul sur steps)

        cid = clusters_ids.long()

        for c in range(self.n_clusters):
            cl_dir = out_dir / f"cl{c}"
            cl_dir.mkdir(parents=True, exist_ok=True)

            mask = (cid == c)
            if mask.any():
                sc_c = red(self._sp(scale[mask]) if self.force_positive else scale[mask])
            else:
                # valeurs par défaut si aucun sample de ce cluster
                sc_c = torch.tensor(1.0, device=device, dtype=dtype)

            # paramètres du cluster (contraints si besoin)
            tag, a, b = self._select_primary_params(torch.tensor([c], device=device))
            # a/b sont shapés (1,), squeeze pour scalaires
            a_sc = a.squeeze(0) if isinstance(a, torch.Tensor) else a
            b_sc = b.squeeze(0) if (isinstance(b, torch.Tensor) and b is not None) else b

            # Edges de ce cluster
            u0_c    = u0_all[c]
            edges_c = E_all[c] if E_all.numel() > 0 else torch.empty(0, device=device, dtype=dtype)

            # Heuristique y_max
            if y_max is None:
                if edges_c.numel() > 0:
                    ymax = float(edges_c[-1].detach().cpu()) * 2.0
                else:
                    ymax = float(u0_c.detach().cpu()) * 2.0 + 1.0
                ymax = max(ymax, float(u0_c.detach().cpu()) * 1.5 + 1.0, 1.0)
            else:
                ymax = float(y_max)

            y = torch.linspace(0.0, ymax, steps=num_points, device=device, dtype=dtype)

            # PDF selon T
            if tag == "egpd":
                f_pdf = pdf_egpd(y, sc_c, b_sc, a_sc)           # (xi=b_sc, kappa=a_sc)
            elif tag == "log":
                f_pdf = pdf_lognormal(y, a_sc, sc_c)            # (mu=a_sc, sigma=sc_c)
            else:  # "gamma"
                f_pdf = pdf_gamma_k_theta(y, a_sc, sc_c)        # (k=a_sc, theta=sc_c)

            # --- Plot
            y_np = y.detach().cpu().numpy()
            f_np = f_pdf.detach().cpu().numpy()

            plt.figure(figsize=(8.8, 5.2))
            plt.plot(y_np, f_np, label=f'{self.T} PDF')

            # concat u0 + edges pour annotation u0,u1,…
            if edges_c.numel() > 0:
                all_edges = torch.cat([u0_c.view(1), edges_c], dim=0)
            else:
                all_edges = u0_c.view(1)

            names = [f"u{j}" for j in range(all_edges.numel())]
            ymax_txt = (np.nanmax(f_np[np.isfinite(f_np)]) if np.isfinite(f_np).any() else 1.0)
            for name, uj in zip(names, all_edges):
                xval = float(uj.detach().cpu())
                plt.axvline(x=xval, linestyle='--', alpha=0.55, label=name)
                plt.text(xval, ymax_txt*0.9, f"{name}={xval:.3f}",
                        rotation=90, va='top', ha='right', fontsize=9, alpha=0.8)

            # échelle log utile pour les queues
            f_plot = np.clip(f_np, 1e-12, None)
            plt.xlim(0, ymax)
            plt.yscale('log')
            ymin = max(1e-12, f_plot[f_plot > 0].min()*0.8) if np.any(f_plot > 0) else 1e-12
            ymax_plot = f_plot.max()*1.2
            plt.ylim(ymin, ymax_plot)

            plt.xlabel('y'); plt.ylabel('Density (log scale)')
            plt.title(f'Cluster {c} — T={self.T}')
            plt.legend(ncol=2); plt.grid(True, linestyle='--', alpha=0.35, which='both')
            plt.tight_layout()

            fname = (output_pdf or "final_pdf")
            if not fname.lower().endswith(".png"):
                fname += ".png"
            plt.savefig(cl_dir / fname, dpi=dpi)
            plt.close()

class BulkTailMixtureIntervalCELoss(nn.Module):
    """
    Mélange Bulk + Tail pour classes ordinales :
      F(u) = (1 - pi) * F_BULK(u) + pi * F_TAIL(u)

    - B ∈ {"log", "gamma"}
        * "log"   : Bulk = Log-Normale
            - param global appris : mu_bulk (moyenne du log)
            - param par échantillon : sigma_bulk (écart-type du log) = inputs[:,1]
        * "gamma" : Bulk = Gamma (forme-échelle)
            - param global appris : k_shape (forme k > 0)
            - param par échantillon : theta_bulk (échelle θ > 0) = inputs[:,1]

    - T ∈ {"egpd", "log", "gamma"}
        * "egpd"  : Tail = eGPD
            - params globaux appris : kappa, xi (>0)
            - param par échantillon : sigma_tail (échelle eGPD) = inputs[:,0]
        * "log"   : Tail = Log-Normale
            - param global appris : mu_tail (moyenne du log)
            - param par échantillon : sigma_tail (écart-type du log) = inputs[:,0]
        * "gamma" : Tail = Gamma (forme-échelle)
            - param global appris : k_shape_tail (>0)
            - param par échantillon : theta_tail (échelle θ) = inputs[:,0]

    - Entrées par échantillon (inputs de shape (N, 3)):
        inputs[:,0] = sigma_tail   (interprétation selon T)
        inputs[:,1] = sigma_bulk   (si B="log")  OU  theta_bulk (si B="gamma")
        inputs[:,2] = pi_tail_logit  (logit du poids de queue)

    - Probas de classes = différences de CDF aux bords appris.
    - Loss: 'entropy' | 'mcewk' | 'bceloss'

    y_class ∈ {0..K-1}
    """

    def __init__(self,
                 num_classes: int = 5,
                 # Bulk (log)
                 mu_ln: float = 0.0,          # mu du bulk log; ignoré si B="gamma"
                 # Tail eGPD
                 kappa: float = 0.8,
                 xi: float = 0.15,
                 # Tail log/gamma
                 mu_tail_ln: float = 0.0,     # mu du tail log
                 k_tail_init: float = 2.0,    # forme k du tail gamma
                 # Edges
                 u0_init: float = 0.5,
                 # Divers
                 eps: float = 1e-8,
                 L: str = 'entropy',
                 reduction: str = "mean",
                 force_positive: bool = False,
                 # Choix Bulk/Tail
                 B: str = "log",              # "log" | "gamma"
                 T: str = "egpd",             # "egpd" | "log" | "gamma"
                 # Bulk gamma
                 k_init: float = 2.0          # forme k du bulk gamma
                 ):
        super().__init__()
        assert num_classes >= 2
        assert B in {"log", "gamma"}, f"B must be 'log' or 'gamma', got {B}"
        assert T in {"egpd", "log", "gamma"}, f"T must be 'egpd' | 'log' | 'gamma', got {T}"

        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.L = L
        self.B = B
        self.T = T

        # ----- Bulk (param globaux) -----
        # "log"   : mu_bulk = moyenne du log (utilisé si B="log")
        # "gamma" : k_shape = forme (utilisé si B="gamma")
        self.mu_bulk = nn.Parameter(torch.tensor(float(mu_ln)))    # pour B="log"
        self.k_shape = nn.Parameter(torch.tensor(float(k_init)))   # pour B="gamma"

        # ----- Tail (param globaux) -----
        # eGPD (utilisés si T="egpd")
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))
        self.xi    = nn.Parameter(torch.tensor(float(xi)))
        # Log-normal tail (utilisé si T="log")
        self.mu_tail = nn.Parameter(torch.tensor(float(mu_tail_ln)))
        # Gamma tail (utilisé si T="gamma")
        self.k_shape_tail = nn.Parameter(torch.tensor(float(k_tail_init)))
        self.gamma_cdf_mode = 'trapz'

        # ----- Bords de classes : u0 + cumsum(softplus(t_steps)) -----
        self.u0_raw  = nn.Parameter(torch.tensor(float(u0_init)))
        base_steps = torch.arange(1, num_classes-1, dtype=torch.float32)  # [1..K-2]
        self.t_steps = nn.Parameter(base_steps)

    def _pdf_gamma_k_theta(self, y, k, theta, eps=1e-12):
        # y, k, theta broadcastables
        k = k.clamp_min(eps)
        theta = theta.clamp_min(eps)
        y = y.clamp_min(eps)
        log_pdf = (k - 1.0) * torch.log(y) - (y / theta) - torch.lgamma(k) - k * torch.log(theta)
        return torch.exp(log_pdf)

    def _cdf_gamma_k_theta_trapz(self, u, k, theta, n_grid=512):
        """
        CDF ~ ∫_0^u pdf(y;k,θ) dy via trapèzes.
        - Différentiable w.r.t. k et θ (la grille n’est pas dépendante des params).
        - n_grid: compromis précision/coût.
        """
        # borne de la grille (choix simple & stable)
        # on prend un max commun au batch pour vectoriser
        u_max = torch.amax(u.detach())  # détachée pour éviter un graphe immense
        y_max = torch.maximum(u_max, (k.detach() * theta.detach() * 8.0).amax().to(u_max))  # borne large
        y = torch.linspace(0.0, float(y_max), steps=n_grid, device=u.device, dtype=u.dtype)  # (G,)
        dy = y[1] - y[0]

        # pdf(y) pour chaque sample – on diffuse y -> (N,G)
        # on a besoin de k,theta de shape (N,1) pour broadcast avec y (G,)
        k_ = k.unsqueeze(-1)
        th = theta.unsqueeze(-1)
        pdf = self._pdf_gamma_k_theta(y.unsqueeze(0), k_, th)  # (N,G)

        # CDF discrète par cumsum (trapèzes)
        # integ[0]=0 ; integ[i] = integ[i-1] + 0.5*(pdf[i]+pdf[i-1])*dy
        trap = 0.5 * (pdf[:, 1:] + pdf[:, :-1]) * dy
        cdf_grid = torch.cat([torch.zeros_like(trap[:, :1]), torch.cumsum(trap, dim=1)], dim=1)  # (N,G)

        # Interpolation linéaire de cdf(u)
        # indices dans la grille
        idx = torch.searchsorted(y, u.clamp_max(y[-1]-1e-12)).clamp_max(y.numel()-1)  # (N,)
        idx0 = (idx - 1).clamp_min(0)
        y0, y1 = y[idx0], y[idx]
        c0 = cdf_grid.gather(1, idx0.unsqueeze(1)).squeeze(1)
        c1 = cdf_grid.gather(1, idx.unsqueeze(1)).squeeze(1)
        t = ((u - y0) / (y1 - y0 + 1e-12)).clamp(0, 1)
        return c0 + t * (c1 - c0)

    # -------- utils bords --------
    def _sp(self, x):
        return F.softplus(x) + 1e-12

    def _u0(self, device, dtype):
        return self._sp(self.u0_raw).to(device=device, dtype=dtype)

    def _edges_tensor(self, device, dtype):
        steps = self._sp(self.t_steps.to(device=device, dtype=dtype))  # >0
        u0    = self._u0(device, dtype)
        return u0 + torch.cumsum(steps, dim=0)  # (K-1,)

    def get_edges(self, use_sp=True):
        with torch.no_grad():
            steps = self._sp(self.t_steps) if use_sp else self.t_steps
            u0    = self._sp(self.u0_raw) if use_sp else self.u0_raw
            edges = (u0 + torch.cumsum(steps, dim=0))
        return tuple(x for x in edges)

    # -------- CDFs de base --------
    def _cdf_lognormal(self, u: torch.Tensor, mu_log: torch.Tensor, sigma_ln: torch.Tensor) -> torch.Tensor:
        # F_LN(u) = Phi((ln u - mu)/sigma), u>0 ; sinon 0
        pos = (u > 0)
        z = (torch.log(u.clamp(min=self.eps)) - mu_log) / (sigma_ln * math.sqrt(2.0))
        F_ln = 0.5 * (1.0 + torch.erf(z))
        return torch.where(pos, F_ln, torch.zeros_like(F_ln))

    def _cdf_gamma_k_theta(self, u, k, theta):
        if getattr(self, "gamma_cdf_mode", "igamma") == "trapz":
            return self._cdf_gamma_k_theta_trapz(u, k, theta)   # différentiable w.r.t. k & θ
        else:
            # fast path, mais stop grad sur k
            k_eff = k.detach()
            try:
                return torch.special.gammainc(k_eff, (u/theta).clamp_min(0.0))
            except AttributeError:
                return torch.igamma(k_eff, (u/theta).clamp_min(0.0)) / torch.exp(torch.lgamma(k_eff))

    # -------- CDF Bulk --------
    def _cdf_bulk(self, u: torch.Tensor, sigma_or_theta: torch.Tensor, device, dtype) -> torch.Tensor:
        if self.B == "log":
            mu_log = self.mu_bulk.to(device=device, dtype=dtype)
            sigma_ln = sigma_or_theta
            return self._cdf_lognormal(u, mu_log, sigma_ln)
        else:
            k = (self._sp(self.k_shape) if self.force_positive else self.k_shape).to(device=device, dtype=dtype)
            theta = sigma_or_theta
            return self._cdf_gamma_k_theta(u, k, theta)

    # -------- CDF Tail (selon T) --------
    def _cdf_tail(self, u: torch.Tensor, sigma_tail: torch.Tensor, device, dtype) -> torch.Tensor:
        if self.T == "egpd":
            kappa = self._sp(self.kappa).to(device=device, dtype=dtype)
            if self.force_positive:
                xi    = self._sp(self.xi).to(device=device, dtype=dtype)
            else:
                xi    = self.xi.to(device=device, dtype=dtype)
            return cdf_egpd_family1(u.clamp_min(0.0), sigma_tail, xi, kappa, eps=1e-12)

        elif self.T == "log":
            mu_t = self.mu_tail.to(device=device, dtype=dtype)
            sigma_ln_t = sigma_tail  # per-sample std (log-scale)
            return self._cdf_lognormal(u, mu_t, sigma_ln_t)

        elif self.T == "gamma":
            k_t = (self._sp(self.k_shape_tail) if self.force_positive else self.k_shape_tail).to(device=device, dtype=dtype)
            theta_t = sigma_tail  # per-sample scale θ
            return self._cdf_gamma_k_theta(u, k_t, theta_t)

        else:
            raise ValueError(f"Unknown tail type T={self.T}")

    # -------- CDF du mélange --------
    def _cdf_mixture(self,
                     u: torch.Tensor,
                     sigma_tail: torch.Tensor,
                     sigma_or_theta_bulk: torch.Tensor,  # std(log) si B="log", scale θ si B="gamma"
                     pi_tail_logit: torch.Tensor,
                     device, dtype) -> torch.Tensor:

        pi = torch.sigmoid(pi_tail_logit.to(device=device, dtype=dtype)).clamp(1e-12, 1 - 1e-12)
        F_bulk = self._cdf_bulk(u, sigma_or_theta_bulk, device, dtype)
        F_tail = self._cdf_tail(u, sigma_tail, device, dtype)
        return (1.0 - pi) * F_bulk + pi * F_tail

    # -------- API: transform --------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, output_pdf: str, dir_output: Path):
        with torch.no_grad():
            sigma_tail, sigma_or_theta_bulk, pi_tail_logit = inputs[:, 0], inputs[:, 1], inputs[:, 2]
            if sigma_tail.ndim > 1:    sigma_tail = sigma_tail.squeeze(-1)
            if sigma_or_theta_bulk.ndim > 1: sigma_or_theta_bulk = sigma_or_theta_bulk.squeeze(-1)
            if pi_tail_logit.ndim > 1: pi_tail_logit = pi_tail_logit.squeeze(-1)

            # Positifs (échelles)
            sigma_tail = self._sp(sigma_tail)
            sigma_or_theta_bulk =  self._sp(sigma_or_theta_bulk)

            self.plot_final_pdf(sigma_tail, sigma_or_theta_bulk, pi_tail_logit,
                                dir_output=dir_output, output_pdf=output_pdf)

            device, dtype = sigma_tail.device, sigma_tail.dtype
            K = self.num_classes

            u0    = self._u0(device, dtype)
            edges = self._edges_tensor(device, dtype)

            F_u = [ self._cdf_mixture(u0.expand_as(sigma_tail), sigma_tail, sigma_or_theta_bulk,
                                      pi_tail_logit, device, dtype) ]
            for u in edges:
                F_u.append(self._cdf_mixture(u.expand_as(sigma_tail), sigma_tail, sigma_or_theta_bulk,
                                             pi_tail_logit, device, dtype))

            Ps = [F_u[0]] + [torch.clamp(F_u[j+1] - F_u[j], 1e-12, 1.0) for j in range(K-2)]
            Ps.append(torch.clamp(1.0 - F_u[-1], 1e-12, 1.0))
            P = torch.stack(Ps, dim=-1)
            P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)

            # plots de diag (try/except pour environnements headless)
            try:
                dir_output = Path(dir_output); dir_output.mkdir(parents=True, exist_ok=True)
                meanP = P.mean(dim=0).detach().cpu().numpy()
                fig = plt.figure(figsize=(6, 4))
                plt.bar(range(K), meanP); plt.xlabel("Classe"); plt.ylabel("P moyenne")
                plt.title("Probabilités moyennes par classe"); plt.xticks(range(K))
                fig.tight_layout(); fig.savefig(dir_output / f"{output_pdf}_probas_mean_bar.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[transform] Plot ignoré: {e}")

            return P

    # -------- Loss --------
    def forward(self, inputs: torch.Tensor, y_class: torch.Tensor, weight: torch.Tensor = None):
        assert inputs.ndim == 2 and inputs.shape[1] == 3, \
            f"wrong shape on inputs: {inputs.shape}, expected (N,3)"
        sigma_tail, sigma_or_theta_bulk, pi_tail_logit = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        if sigma_tail.ndim > 1:    sigma_tail = sigma_tail.squeeze(-1)
        if sigma_or_theta_bulk.ndim > 1: sigma_or_theta_bulk = sigma_or_theta_bulk.squeeze(-1)
        if pi_tail_logit.ndim > 1: pi_tail_logit = pi_tail_logit.squeeze(-1)

        # Positifs
        sigma_tail = self._sp(sigma_tail)
        sigma_or_theta_bulk =  self._sp(sigma_or_theta_bulk)

        device, dtype = sigma_tail.device, sigma_tail.dtype
        K = self.num_classes

        u0    = self._u0(device, dtype)
        edges = self._edges_tensor(device, dtype)

        F_u = [ self._cdf_mixture(u0.expand_as(sigma_tail), sigma_tail, sigma_or_theta_bulk,
                                  pi_tail_logit, device, dtype) ]
        for u in edges:
            F_u.append(self._cdf_mixture(u.expand_as(sigma_tail), sigma_tail, sigma_or_theta_bulk,
                                         pi_tail_logit, device, dtype))

        Ps = [F_u[0]] + [torch.clamp(F_u[j+1] - F_u[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_u[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)

        if self.L == 'entropy':
            logP = torch.log(P.clamp_min(1e-12))
            loss = F.nll_loss(logP, y_class, reduction='none')
        elif self.L == 'mcewk':
            loss = MCEAndWKLoss(num_classes=self.num_classes, use_logits=False).forward(P, y_class)
        elif self.L == 'bceloss':
            loss = BCELoss(num_classes=self.num_classes).forward(P, y_class)
        else:
            raise ValueError(f'{self.L} unknow')

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
    
    def get_learnable_parameters(self):
        params = {
            "u0_raw": self.u0_raw,
            "t_steps": self.t_steps,
        }

        # --- Bulk ---
        if self.B == "log":
            params["mu_bulk"] = self.mu_bulk
        elif self.B == "gamma":
            params["k_shape"] = self.k_shape

        # --- Tail ---
        if self.T == "egpd":
            params["kappa"] = self.kappa
            params["xi"]    = self.xi
        elif self.T == "log":
            params["mu_tail"] = self.mu_tail
        elif self.T == "gamma":
            params["k_shape_tail"] = self.k_shape_tail
        else:
            raise ValueError(f"Unknown tail type T={self.T}")

        return params

    def get_attribute(self):
        kappa_v = self.kappa 
        xi_v    = self.xi
        u0_v = self.u0_raw

        attrs = [('u0_raw', u0_v), ('t_steps', self.t_steps)]
        # Bulk visible
        if self.B == "log":
            attrs.append(('mu_bulk', self.mu_bulk))
        else:
            attrs.append(('k_shape', self.k_shape))
        # Tail visible (selon T)
        if self.T == "egpd":
            attrs.append(('kappa', kappa_v))
            attrs.append(('xi', xi_v))
        elif self.T == "log":
            attrs.append(('mu_tail', self.mu_tail))
        elif self.T == "gamma":
            attrs.append(('k_shape_tail', self.k_shape_tail))

        return attrs

    # -------- Plot PDF (diag) --------
    def plot_final_pdf(self,
                       sigma_tail: torch.Tensor,
                       sigma_or_theta_bulk: torch.Tensor,
                       pi_tail_logit: torch.Tensor,
                       dir_output,
                       y_max: float = None,
                       num_points: int = 400,
                       dpi: int = 120,
                       stat: str = "median",
                       output_pdf: str = "final_cdf"):
        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)
        device, dtype = sigma_tail.device, sigma_tail.dtype
        K = self.num_classes

        # réducteurs
        red = torch.median if stat == "median" else torch.mean
        sigma_tail_r = red(sigma_tail)
        sigma_or_theta_bulk_r = red(sigma_or_theta_bulk)
        pi_tail = torch.sigmoid(pi_tail_logit.to(device=device, dtype=dtype)).clamp(1e-12, 1 - 1e-12)
        pi_tail = red(pi_tail)
        kappa = F.softplus(self.kappa) + 1e-12

        # params globaux (eGPD)
        if self.force_positive:
            xi    = F.softplus(self.xi)    + 1e-12
        else:
            xi    = self.xi

        # Edges
        u0 = self._sp(self.u0_raw)
        if K > 2:
            steps = self._sp(self.t_steps)
            edges = u0 + torch.cumsum(steps, dim=0)
            all_edges = torch.cat([u0.view(1), edges], dim=0)
        else:
            all_edges = u0.view(1)

        # y_max heuristique
        if y_max is None:
            ymax = float(all_edges[-1].detach().cpu()) * 2.0
            ymax = max(ymax, float(u0.detach().cpu()) * 1.5 + 1.0, 1.0)
        else:
            ymax = float(y_max)

        y = torch.linspace(0.0, ymax, steps=num_points, device=device, dtype=dtype)

        # PDFs élémentaires
        def pdf_lognormal(yv, mu_log, sigma_ln, eps=1e-8):
            yv = yv.clamp_min(eps)
            z  = (torch.log(yv) - mu_log) / sigma_ln
            return torch.exp(-0.5 * z * z) / (yv * sigma_ln * math.sqrt(2.0 * math.pi))

        def pdf_gamma_k_theta(yv, k, theta, eps=1e-12):
            k     = k.clamp_min(eps); theta = theta.clamp_min(eps)
            return torch.exp((k - 1.0) * torch.log(yv.clamp_min(eps)) - (yv / theta) - torch.lgamma(k)) / (theta ** k)

        def pdf_egpd(yv, sigma_t, xi_v, kappa_v, eps=1e-12):
            yv = yv.clamp_min(0.0)
            si = sigma_t.clamp_min(eps)
            xv = xi_v.clamp_min(eps)
            kv = kappa_v.clamp_min(eps)
            z  = (1.0 + xv * (yv / si)).clamp_min(1.0 + 1e-12)
            H  = (1.0 - torch.pow(z, -1.0 / xv)).clamp(min=eps, max=1.0 - eps)
            h  = (1.0 / si) * torch.pow(z, -1.0 / xv - 1.0)
            return kv * h * torch.pow(H, kv - 1.0)

        # Bulk PDF
        if self.B == "log":
            f_bulk = pdf_lognormal(y, self.mu_bulk.to(device=device, dtype=dtype), sigma_or_theta_bulk_r)
        else:
            k_b = self._sp(self.k_shape) if self.force_positive else self.k_shape
            f_bulk = pdf_gamma_k_theta(y, k_b.to(device=device, dtype=dtype), sigma_or_theta_bulk_r)

        # Tail PDF selon T
        if self.T == "egpd":
            f_tail = pdf_egpd(y, sigma_tail_r, xi, kappa)
        elif self.T == "log":
            f_tail = pdf_lognormal(y, self.mu_tail.to(device=device, dtype=dtype), sigma_tail_r)
        elif self.T == "gamma":
            k_t = self._sp(self.k_shape_tail) if self.force_positive else self.k_shape_tail
            f_tail = pdf_gamma_k_theta(y, k_t.to(device=device, dtype=dtype), sigma_tail_r)
        else:
            raise ValueError(f"Unknown tail type T={self.T}")

        f_mix  = (1.0 - pi_tail) * f_bulk + pi_tail * f_tail

        # Plot
        y_np = y.detach().cpu().numpy()
        f_np = f_mix.detach().cpu().numpy()
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(y_np, f_np, label=f'f_mix(y) [B={self.B}, T={self.T}]')

        names = [f"u{j}" for j in range(all_edges.numel())]
        for name, uj in zip(names, all_edges):
            xval = float(uj.detach().cpu())
            plt.axvline(x=xval, linestyle='--', alpha=0.55, label=name)
            ymax_txt = (np.nanmax(f_np[np.isfinite(f_np)]) if np.isfinite(f_np).any() else 1.0)
            plt.text(xval, ymax_txt*0.9, f"{name}={xval:.3f}", rotation=90, va='top', ha='right', fontsize=9, alpha=0.8)

        f_plot = np.clip(f_np, 1e-12, None)
        plt.xlim(0, ymax)
        plt.yscale('log')
        ymin = max(1e-12, f_plot[f_plot > 0].min()*0.8) if np.any(f_plot > 0) else 1e-12
        ymax_plot = f_plot.max()*1.2
        plt.ylim(ymin, ymax_plot)
        plt.xlabel('y'); plt.ylabel('Density (log scale)')
        plt.title(f'Mixture PDF — pi_tail={float(pi_tail.detach().cpu()):.4f}')
        plt.legend(ncol=2); plt.grid(True, linestyle='--', alpha=0.35, which='both')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / (output_pdf if output_pdf is not None else "final_cdf.png"), dpi=dpi)
        plt.close()

    def update_params(self, new_values: dict, strict: bool = False):
        learnables = self.get_learnable_parameters()
        updated = []

        def to_tensor_like(x, ref: torch.Tensor):
            t = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
            if t.ndim == 0 and ref.numel() > 1:
                t = t.expand_as(ref)
            if t.shape != ref.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{name}': got {tuple(t.shape)}, expected {tuple(ref.shape)}")
                try:
                    t = t.expand_as(ref)
                except Exception:
                    return None
            return t

        for name, value in new_values.items():
            if name not in learnables:
                if strict: raise KeyError(f"Unknown learnable parameter '{name}'.")
                else: continue
            param = learnables[name]
            if not isinstance(param, torch.nn.Parameter):
                if strict: raise TypeError(f"Object mapped by '{name}' is not an nn.Parameter.")
                else: continue
            t = to_tensor_like(value, param.data)
            if t is None:
                if strict: raise ValueError(f"Incompatible value for '{name}'.")
                else: continue
            param.data.copy_(t); updated.append(name)

        print(updated)
        print(new_values)
        return updated

    def plot_params(self, logs, dir_output, dpi=120):
        """
        Trace l’évolution des paramètres (si `logs` fournis) ET superpose la valeur *courante*
        du modèle sur chaque plot existant.

        Attentes (facultatives selon B/T):
        logs: liste de dicts avec au moins 'epoch'.
            - 'u0'                : scalaire
            - 'edges'             : (K-2,)
            - Bulk:
                * si B='log'   : 'mu_bulk'         (scalaire)
                * si B='gamma' : 'k_shape_bulk'    (scalaire)
            - Tail:
                * si T='egpd'  : 'kappa', 'xi'     (scalaires)
                * si T='log'   : 'mu_tail'         (scalaire)
                * si T='gamma' : 'k_shape_tail'    (scalaire)
        Sauvegardes dans dir_output.
        """
        out = Path(dir_output); out.mkdir(parents=True, exist_ok=True)

        def to_np(x):
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            return np.asarray(x)

        # -------- Valeurs "courantes" (snapshot du modèle) --------
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        xi_min = getattr(self, "xi_min", 0.0)

        cur_u0 = float(to_np(self._sp(self.u0_raw)))
        if self.num_classes > 2:
            cur_edges = to_np(self._sp(self.u0_raw) + torch.cumsum(self._sp(self.t_steps.to(device=device, dtype=dtype)), dim=0))
        else:
            cur_edges = np.empty((0,), dtype=float)

        # Bulk courant
        cur_mu_bulk = None
        cur_k_shape_bulk = None
        if self.B == "log" and hasattr(self, "mu_bulk"):
            cur_mu_bulk = float(to_np(self.mu_bulk))
        elif self.B == "gamma" and hasattr(self, "k_shape"):
            k = self._sp(self.k_shape) if getattr(self, "force_positive", True) else self.k_shape
            cur_k_shape_bulk = float(to_np(k))

        # Tail courant
        cur_kappa = cur_xi = None
        cur_mu_tail = None
        cur_k_shape_tail = None
        if self.T == "egpd" and hasattr(self, "kappa") and hasattr(self, "xi"):
            kap = self._sp(self.kappa)
            if getattr(self, "force_positive", True):
                xi  = self._sp(self.xi) + xi_min
            else:
                xi  = self.xi
            cur_kappa = float(to_np(kap)); cur_xi = float(to_np(xi))
        elif self.T == "log" and hasattr(self, "mu_tail"):
            cur_mu_tail = float(to_np(self.mu_tail))
        elif self.T == "gamma" and hasattr(self, "k_shape_tail"):
            kst = self._sp(self.k_shape_tail) if getattr(self, "force_positive", True) else self.k_shape_tail
            cur_k_shape_tail = float(to_np(kst))

        # -------- A) PLOTS TEMPORELS (si logs fournis) + overlays --------
        if logs:
            epochs = [int(l["epoch"]) for l in logs]

            def seq(key):
                if key not in logs[0]: return None
                vals = [l[key] for l in logs]
                try:
                    return np.asarray([float(v) for v in vals])      # (T,)
                except Exception:
                    arrs = [to_np(v) for v in vals]
                    try:
                        return np.stack(arrs, 0)                     # (T, d)
                    except Exception:
                        return np.asarray(arrs)

            # u0
            u0_seq = seq("u0_raw")
            if u0_seq is not None:
                u0_seq = F.softplus(torch.as_tensor(u0_seq, dtype=torch.float32)).detach().cpu().numpy()
                plt.figure(figsize=(8.2,4.6))
                plt.plot(epochs, u0_seq, marker='o', label='u0 (train)')
                plt.axhline(cur_u0, linestyle=':', alpha=0.9, label=f'u0 current={cur_u0:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('u0'); plt.title('u0')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(out/'u0.png', dpi=dpi); plt.close()

            # edges
            edges_seq = seq("edges")
            if edges_seq is not None and edges_seq.ndim == 2 and edges_seq.shape[1] > 0:
                plt.figure(figsize=(10,6))
                Kminus2 = edges_seq.shape[1]
                for j in range(Kminus2):
                    plt.plot(epochs, edges_seq[:, j], marker='^', linestyle='--', label=f'u{j+1} (train)')
                    if j < len(cur_edges):
                        plt.scatter([epochs[-1]], [cur_edges[j]], s=36, marker='o', alpha=0.9)
                plt.xlabel('Epoch'); plt.ylabel('edge'); plt.title('edges')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                plt.tight_layout(); plt.savefig(out/'edges.png', dpi=dpi); plt.close()

            # Bulk
            if self.B == "log":
                mu_bulk_seq = seq("mu_bulk")
                if mu_bulk_seq is not None:
                    plt.figure(figsize=(8.2,4.6))
                    plt.plot(epochs, mu_bulk_seq, marker='o', label='mu_bulk (train)')
                    if cur_mu_bulk is not None:
                        plt.axhline(cur_mu_bulk, linestyle=':', alpha=0.9, label=f'current={cur_mu_bulk:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('mu_bulk'); plt.title('bulk (log)')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(out/'bulk_log.png', dpi=dpi); plt.close()
            elif self.B == "gamma":  # gamma : si force_positive, appliquer softplus aux logs avant plot
                kshape_bulk_seq = seq("k_shape_bulk")
                if kshape_bulk_seq is not None:
                    if getattr(self, "force_positive", True):
                        kshape_bulk_seq = F.softplus(torch.as_tensor(kshape_bulk_seq, dtype=torch.float32)).detach().cpu().numpy()
                    plt.figure(figsize=(8.2,4.6))
                    plt.plot(epochs, kshape_bulk_seq, marker='o', label='k_shape_bulk (train)')
                    if cur_k_shape_bulk is not None:
                        plt.axhline(cur_k_shape_bulk, linestyle=':', alpha=0.9, label=f'current={cur_k_shape_bulk:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('k_shape_bulk'); plt.title('bulk (gamma)')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(out/'bulk_gamma.png', dpi=dpi); plt.close()
            else:
                raise ValueError(f'{self.B} unknow bulk type')
            
            # Tail
            if self.T == "egpd":
                kap_seq = seq("kappa"); xi_seq = seq("xi")
                if kap_seq is not None or xi_seq is not None:
                    if kap_seq is not None:
                        kap_seq = F.softplus(torch.as_tensor(kap_seq, dtype=torch.float32)).detach().cpu().numpy()
                    if getattr(self, "force_positive", True):
                        if xi_seq is not None:
                            xi_seq = (F.softplus(torch.as_tensor(xi_seq, dtype=torch.float32)) + xi_min).detach().cpu().numpy()
                    plt.figure(figsize=(8.8,5.0))
                    if kap_seq is not None:
                        plt.plot(epochs, kap_seq, color='blue', marker='o', label='kappa (train)')
                        if cur_kappa is not None:
                            plt.axhline(cur_kappa, color='blue', linestyle=':', alpha=0.9, label=f'kappa current={cur_kappa:.4g}')
                    if xi_seq is not None:
                        plt.plot(epochs, xi_seq, color='red', marker='s', label='xi (train)')
                        if cur_xi is not None:
                            plt.axhline(cur_xi, color='red', linestyle=':', alpha=0.9, label=f'xi current={cur_xi:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('value'); plt.title('tail (eGPD)')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                    plt.tight_layout(); plt.savefig(out/'tail_egpd.png', dpi=dpi); plt.close()

            elif self.T == "log":
                mu_tail_seq = seq("mu_tail")
                if mu_tail_seq is not None:
                    plt.figure(figsize=(8.2,4.6))
                    plt.plot(epochs, mu_tail_seq, marker='o', label='mu_tail (train)')
                    if cur_mu_tail is not None:
                        plt.axhline(cur_mu_tail, linestyle=':', alpha=0.9, label=f'current={cur_mu_tail:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('mu_tail'); plt.title('tail (lognorm)')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(out/'tail_log.png', dpi=dpi); plt.close()

            else:  # T == "gamma"
                kshape_tail_seq = seq("k_shape_tail")
                if kshape_tail_seq is not None:
                    if getattr(self, "force_positive", True):
                        kshape_tail_seq = F.softplus(torch.as_tensor(kshape_tail_seq, dtype=torch.float32)).detach().cpu().numpy()
                    plt.figure(figsize=(8.2,4.6))
                    plt.plot(epochs, kshape_tail_seq, marker='o', label='k_shape_tail (train)')
                    if cur_k_shape_tail is not None:
                        plt.axhline(cur_k_shape_tail, linestyle=':', alpha=0.9, label=f'current={cur_k_shape_tail:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('k_shape_tail'); plt.title('tail (gamma)')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(out/'tail_gamma.png', dpi=dpi); plt.close()

        # -------- B) Snapshot "current" seul (si pas de logs) --------
        else:
            # u0
            plt.figure(figsize=(8.2,4.6))
            plt.axhline(cur_u0, linestyle=':', alpha=0.9, label=f'u0 current={cur_u0:.4g}')
            plt.title('u0 (current)'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout(); plt.savefig(out/'u0_current.png', dpi=dpi); plt.close()

            # edges
            if cur_edges.size > 0:
                plt.figure(figsize=(9,4.8))
                plt.bar(range(1, cur_edges.size+1), cur_edges)
                plt.xlabel('edge index (u1..uK-2)'); plt.ylabel('value'); plt.title('edges (current)')
                plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
                plt.savefig(out/'edges_current.png', dpi=dpi); plt.close()

            # bulk
            if self.B == "log" and cur_mu_bulk is not None:
                plt.figure(figsize=(7.6,4.2))
                plt.axhline(cur_mu_bulk, linestyle=':', alpha=0.9, label=f'mu_bulk current={cur_mu_bulk:.4g}')
                plt.title('bulk (log) — current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'bulk_log_current.png', dpi=dpi); plt.close()
            if self.B == "gamma" and cur_k_shape_bulk is not None:
                plt.figure(figsize=(7.6,4.2))
                plt.axhline(cur_k_shape_bulk, linestyle=':', alpha=0.9, label=f'k_shape_bulk current={cur_k_shape_bulk:.4g}')
                plt.title('bulk (gamma) — current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'bulk_gamma_current.png', dpi=dpi); plt.close()

            # tail
            if self.T == "egpd" and (cur_kappa is not None or cur_xi is not None):
                plt.figure(figsize=(8.2,4.6))
                if cur_kappa is not None: plt.axhline(cur_kappa, linestyle=':', alpha=0.9, label=f'kappa={cur_kappa:.4g}')
                if cur_xi    is not None: plt.axhline(cur_xi,    linestyle=':', alpha=0.9, label=f'xi={cur_xi:.4g}')
                plt.title('tail (eGPD) — current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'tail_egpd_current.png', dpi=dpi); plt.close()
            if self.T == "log" and cur_mu_tail is not None:
                plt.figure(figsize=(7.6,4.2))
                plt.axhline(cur_mu_tail, linestyle=':', alpha=0.9, label=f'mu_tail current={cur_mu_tail:.4g}')
                plt.title('tail (lognorm) — current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'tail_log_current.png', dpi=dpi); plt.close()
            if self.T == "gamma" and cur_k_shape_tail is not None:
                plt.figure(figsize=(7.6,4.2))
                plt.axhline(cur_k_shape_tail, linestyle=':', alpha=0.9, label=f'k_shape_tail current={cur_k_shape_tail:.4g}')
                plt.title('tail (gamma) — current'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout(); plt.savefig(out/'tail_gamma_current.png', dpi=dpi); plt.close()

class BulkTailMixtureIntervalCELossClusterIDs(nn.Module):
    """
    Bulk + Tail (mixture) *par cluster* pour classes ordinales 0..K-1.

    F_mix(u) = (1 - pi) * F_BULK(u | params_bulk_c, params_bulk_i)
                         + pi * F_TAIL(u | params_tail_c, params_tail_i)

    Choix du BULK (B):
      - "log"   : Log-Normale
          * param cluster (Gm optionnel): mu_c (moyenne du log)
          * param échantillon           : sigma_bulk_i > 0 (inputs[:,1], softplus)
      - "gamma" : Gamma(k, theta)
          * param cluster (Gm optionnel): k_shape_c > 0
          * param échantillon           : theta_bulk_i > 0 (inputs[:,1], softplus)

    Choix du TAIL (T):
      - "egpd"  : eGPD
          * params cluster (Gt optionnel): kappa_c (>0), xi_c (>0)
          * param échantillon            : sigma_tail_i > 0 (inputs[:,0], softplus)
      - "log"   : Log-Normale
          * param cluster (Gt optionnel): mu_tail_c
          * param échantillon           : sigma_tail_i (écart-type log, inputs[:,0], softplus)
      - "gamma" : Gamma(k, theta)
          * param cluster (Gt optionnel): k_shape_tail_c (>0)
          * param échantillon           : theta_tail_i (>0, inputs[:,0], softplus)

    Entrées par échantillon:
      inputs[:,0] = scale du tail (sigma_eGPD OU sigma_log OU theta_gamma)
      inputs[:,1] = scale du bulk (sigma_log OU theta_gamma)
      inputs[:,2] = pi_tail_logit

    Loss L ∈ {'entropy','mcewk','bceloss'}.
    """

    def __init__(self,
                 NC: int,
                 id: int,
                 num_classes: int = 5,
                 # Inits tail eGPD
                 kappa_init: float = 0.831,
                 xi_init: float = 0.161,
                 # Inits bulk log / tail log
                 mu_init_bulk: float = 0.0,
                 mu_init_tail: float = 0.0,
                 # Inits bulk gamma / tail gamma
                 k_shape_init_bulk: float = 2.0,
                 k_shape_init_tail: float = 2.0,
                 # Hiérarchie (global+delta)
                 Gt: bool = False,   # tail params global+delta ?
                 Gm: bool = False,   # bulk params global+delta ?
                 # Edges
                 u0_init: float = 0.5,
                 # Divers
                 eps: float = 1e-8,
                 xi_min: float = 1e-12,
                 reduction: str = "mean",
                 force_positive: bool = False,
                 # Choix de la loss / bulk / tail
                 L: str = "entropy",
                 B: str = "log",         # 'log' | 'gamma'
                 T: str = "egpd"         # 'egpd' | 'log' | 'gamma'
                 ):
        super().__init__()
        assert num_classes >= 2
        assert B in {"log", "gamma"}
        assert T in {"egpd", "log", "gamma"}

        self.C = NC
        self.K = num_classes
        self.id = id
        self.eps = eps
        self.xi_min = xi_min
        self.reduction = reduction
        self.force_positive = force_positive
        self.L = L
        self.B = B
        self.T = T

        # ------------ TAIL (par cluster) ------------
        self.add_global_tail = Gt
        if T == "egpd":
            if not Gt:
                self.kappa_raw = nn.Parameter(torch.full((NC,), float(kappa_init)))
                self.xi_raw    = nn.Parameter(torch.full((NC,), float(xi_init)))
            else:
                self.kappa_global = nn.Parameter(torch.tensor(float(kappa_init)))
                self.xi_global    = nn.Parameter(torch.tensor(float(xi_init)))
                self.kappa_delta  = nn.Parameter(torch.full((NC,), float(kappa_init)))
                self.xi_delta     = nn.Parameter(torch.full((NC,), float(xi_init)))
        elif T == "log":
            if not Gt:
                self.mu_tail_raw = nn.Parameter(torch.full((NC,), float(mu_init_tail)))
            else:
                self.mu_tail_global = nn.Parameter(torch.tensor(float(mu_init_tail)))
                self.mu_tail_delta  = nn.Parameter(torch.full((NC,), float(mu_init_tail)))
        elif T == "gamma":
            if not Gt:
                self.k_shape_tail_raw = nn.Parameter(torch.full((NC,), float(k_shape_init_tail)))
            else:
                self.k_shape_tail_global = nn.Parameter(torch.tensor(float(k_shape_init_tail)))
                self.k_shape_tail_delta  = nn.Parameter(torch.full((NC,), float(k_shape_init_tail)))
        else:
            raise ValueError(f'{T} is a not valid name for Tail')

        # ------------ BULK (par cluster) ------------
        self.add_global_bulk = Gm
        if B == "log":
            if not Gm:
                self.mu_bulk_raw = nn.Parameter(torch.full((NC,), float(mu_init_bulk)))
            else:
                self.mu_bulk_global = nn.Parameter(torch.tensor(float(mu_init_bulk)))
                self.mu_bulk_delta  = nn.Parameter(torch.full((NC,), float(mu_init_bulk)))
        elif B == "gamma":
            if not Gm:
                self.k_shape_bulk_raw = nn.Parameter(torch.full((NC,), float(k_shape_init_bulk)))
            else:
                self.k_shape_bulk_global = nn.Parameter(torch.tensor(float(k_shape_init_bulk)))
                self.k_shape_bulk_delta  = nn.Parameter(torch.full((NC,), float(k_shape_init_bulk)))
        else:
            raise ValueError(f'{B} is a not valid name for Bulk')

        # ------------ Edges par cluster ------------
        self.u0_raw  = nn.Parameter(torch.full((NC,), float(u0_init)))
        base_steps = torch.arange(1, self.K - 1, dtype=torch.float32)   # (K-2,)
        self.t_steps = nn.Parameter(base_steps.unsqueeze(0).repeat(NC, 1), requires_grad=True)  # (C, K-2)

    # --------------- utils bords ---------------
    def _sp(self, x): return F.softplus(x) + 1e-12

    def _u0_all(self, device, dtype, use_sp=True):
        return self._sp(self.u0_raw.to(device=device, dtype=dtype))  if use_sp else self.u0_raw.to(device=device, dtype=dtype)# (C,)

    def _edges_all(self, device, dtype, use_sp=True):
        if self.K <= 2:
            return torch.empty(self.C, 0, device=device, dtype=dtype)
        steps = self._sp(self.t_steps.to(device=device, dtype=dtype)) if use_sp else self.t_steps.to(device=device, dtype=dtype)   # (C,K-2)
        u0 = self._u0_all(device, dtype, use_sp=use_sp).unsqueeze(1)                   # (C,1)
        return u0 + torch.cumsum(steps, dim=1)                          # (C,K-2)

    def _edges_for_samples(self, clusters_ids, device, dtype):
        return self._edges_all(device, dtype)[clusters_ids.long()]       # (N,K-2)

    # --------------- sélecteurs params cluster ---------------
    def _select_bulk_param(self, clusters_ids: torch.Tensor):
        cid = clusters_ids.long()
        if self.B == "log":
            if not self.add_global_bulk:
                mu = self.mu_bulk_raw[cid]
            else:
                mu = self.mu_bulk_global + self.mu_bulk_delta[cid]
            return mu
        else:  # gamma
            if not self.add_global_bulk:
                k = self.k_shape_bulk_raw[cid]
            else:
                k = self.k_shape_bulk_global + self.k_shape_bulk_delta[cid]
            return self._sp(k) if self.force_positive else k

    def _select_tail_params(self, clusters_ids: torch.Tensor):
        """Retourne un dict selon T pour simplifier l'appel en aval."""
        cid = clusters_ids.long()
        if self.T == "egpd":
            if not self.add_global_tail:
                k_raw = self.kappa_raw[cid]
                x_raw = self.xi_raw[cid]
            else:
                k_raw = self.kappa_global + self.kappa_delta[cid]
                x_raw = self.xi_global    + self.xi_delta[cid]
            kappa = self._sp(k_raw)
            xi    = (self._sp(x_raw) + self.xi_min) if self.force_positive else x_raw
            return {"kappa": kappa, "xi": xi}
        elif self.T == "log":
            if not self.add_global_tail:
                mu_t = self.mu_tail_raw[cid]
            else:
                mu_t = self.mu_tail_global + self.mu_tail_delta[cid]
            return {"mu_tail": mu_t}
        else:  # gamma
            if not self.add_global_tail:
                k_t = self.k_shape_tail_raw[cid]
            else:
                k_t = self.k_shape_tail_global + self.k_shape_tail_delta[cid]
            k_t = self._sp(k_t) if self.force_positive else k_t
            return {"k_shape_tail": k_t}

    # --------------- CDF primitives ---------------
    @staticmethod
    def _cdf_lognormal(u, mu, sigma, eps=1e-8):
        pos = (u > 0)
        z = (torch.log(u.clamp(min=eps)) - mu) / (sigma * math.sqrt(2.0))
        F = 0.5 * (1.0 + torch.erf(z))
        return torch.where(pos, F, torch.zeros_like(F))

    @staticmethod
    def _cdf_gamma_k_theta(u, k, theta, eps=1e-12):
        k = k.clamp_min(eps); theta = theta.clamp_min(eps)
        x = (u / theta).clamp_min(0.0)
        try:
            Pkx = torch.special.gammainc(k, x)  # lower regularized
        except AttributeError:
            Pkx = torch.igamma(k, x) / torch.exp(torch.lgamma(k))
        return Pkx

    # --------------- CDF bulk/tail ---------------
    def _cdf_bulk(self, u, scale_bulk, bulk_param_cluster):
        if self.B == "log":
            return self._cdf_lognormal(u, bulk_param_cluster, scale_bulk, eps=self.eps)
        else:
            return self._cdf_gamma_k_theta(u, bulk_param_cluster, scale_bulk, eps=self.eps)

    def _cdf_tail(self, u, scale_tail, tail_params: dict):
        if self.T == "egpd":
            return cdf_egpd_family1(u.clamp_min(0.0),
                                    scale_tail,
                                    tail_params["xi"],
                                    tail_params["kappa"],
                                    eps=1e-12)
        elif self.T == "log":
            return self._cdf_lognormal(u, tail_params["mu_tail"], scale_tail, eps=self.eps)
        else:  # gamma
            return self._cdf_gamma_k_theta(u, tail_params["k_shape_tail"], scale_tail, eps=self.eps)

    # --------------- mélange ---------------
    def _cdf_mixture(self, u, scale_tail, scale_bulk, pi_logit, bulk_param_cluster, tail_params):
        pi = torch.sigmoid(pi_logit).clamp(1e-12, 1 - 1e-12)
        F_b = self._cdf_bulk(u, scale_bulk, bulk_param_cluster)
        F_t = self._cdf_tail(u, scale_tail, tail_params)
        return (1.0 - pi) * F_b + pi * F_t

    # --------------- transform -> P ---------------
    def transform(self, inputs: torch.Tensor, clusters_ids: torch.Tensor,
              dir_output, output_pdf):
        """
        inputs: (N,3) = [scale_tail, scale_bulk, pi_tail_logit]
        return: P (N,K)

        Si dir_output est fourni, génère les PDFs par cluster dans dir_output/cl{c}/...
        """
        assert inputs.ndim == 2 and inputs.shape[1] == 3
        scale_tail, scale_bulk, pi_logit = inputs[:,0], inputs[:,1], inputs[:,2]
        scale_tail = F.softplus(scale_tail) + 1e-12
        scale_bulk = F.softplus(scale_bulk) + 1e-12

        # --- Plot PDFs (diag) ---
        if dir_output is not None:
            self.plot_final_pdf(scale_tail, scale_bulk, pi_logit, clusters_ids,
                                dir_output=dir_output,
                                output_pdf=(output_pdf or "final_pdf"))

        device, dtype = scale_tail.device, scale_tail.dtype
        K = self.K

        bulk_param = self._select_bulk_param(clusters_ids)     # mu_c ou k_shape_c
        tail_params = self._select_tail_params(clusters_ids)   # dict selon T

        u0_all  = self._u0_all(device, dtype)
        u0_N    = u0_all[clusters_ids.long()]
        edges_N = self._edges_for_samples(clusters_ids, device, dtype)

        Fu = [ self._cdf_mixture(u0_N, scale_tail, scale_bulk, pi_logit, bulk_param, tail_params) ]
        for j in range(K - 2):
            Fu.append(self._cdf_mixture(edges_N[:,j], scale_tail, scale_bulk, pi_logit, bulk_param, tail_params))

        Ps = [torch.clamp(Fu[0], 1e-12, 1.0)]
        for j in range(K - 2):
            Ps.append(torch.clamp(Fu[j+1] - Fu[j], 1e-12, 1.0))
        Ps.append(torch.clamp(1.0 - Fu[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)
        return P

    # --------------- forward (loss) ---------------
    def forward(self, inputs: torch.Tensor, y_class: torch.Tensor, clusters_ids: torch.Tensor, weight: torch.Tensor = None):
        """
        inputs: (N,3) = [scale_tail, scale_bulk, pi_tail_logit]
        """
        P = self.transform(inputs, clusters_ids, dir_output=None, output_pdf=None)  # (N,K)

        if self.L == 'entropy':
            logP = torch.log(P.clamp_min(1e-12))
            loss = F.nll_loss(logP, y_class.long(), reduction='none')
        elif self.L == 'mcewk':
            loss = MCEAndWKLoss(num_classes=self.K, use_logits=False).forward(P, y_class)
        elif self.L == 'bceloss':
            loss = BCELoss(num_classes=self.K).forward(P, y_class)
        else:
            raise ValueError(f"{self.L} unknown")

        if weight is not None:
            loss = loss * weight
        return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)

    # --------------- paramètres apprenables (SEULEMENT ceux utilisés) ---------------
    def get_learnable_parameters(self):
        params = {
            "u0_raw": self.u0_raw,
            "t_steps": self.t_steps,
        }

        # --- BULK ---
        if self.B == "log":
            if not self.add_global_bulk:
                params["mu_bulk_raw"] = self.mu_bulk_raw
            else:
                params["mu_bulk_global"] = self.mu_bulk_global
                params["mu_bulk_delta"]  = self.mu_bulk_delta
        else:  # gamma
            if not self.add_global_bulk:
                params["k_shape_bulk_raw"] = self.k_shape_bulk_raw
            else:
                params["k_shape_bulk_global"] = self.k_shape_bulk_global
                params["k_shape_bulk_delta"]  = self.k_shape_bulk_delta

        # --- TAIL ---
        if self.T == "egpd":
            if not self.add_global_tail:
                params["kappa_raw"] = self.kappa_raw
                params["xi_raw"]    = self.xi_raw
            else:
                params["kappa_global"] = self.kappa_global
                params["xi_global"]    = self.xi_global
                params["kappa_delta"]  = self.kappa_delta
                params["xi_delta"]     = self.xi_delta
        elif self.T == "log":
            if not self.add_global_tail:
                params["mu_tail_raw"] = self.mu_tail_raw
            else:
                params["mu_tail_global"] = self.mu_tail_global
                params["mu_tail_delta"]  = self.mu_tail_delta
        else:  # gamma
            if not self.add_global_tail:
                params["k_shape_tail_raw"] = self.k_shape_tail_raw
            else:
                params["k_shape_tail_global"] = self.k_shape_tail_global
                params["k_shape_tail_delta"]  = self.k_shape_tail_delta

        return params

    # --------------- mise à jour directe ---------------
    def update_params(self, new_values: dict, strict: bool = False):
        learnables = self.get_learnable_parameters()
        updated = []

        def to_tensor_like(x, ref: torch.Tensor):
            t = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
            if t.ndim == 0 and ref.numel() > 1:
                t = t.expand_as(ref)
            if t.shape != ref.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{name}': got {tuple(t.shape)}, expected {tuple(ref.shape)}")
                try:
                    t = t.expand_as(ref)
                except Exception:
                    return None
            return t

        for name, value in new_values.items():
            if name not in learnables:
                if strict: raise KeyError(f"Unknown learnable parameter '{name}'.")
                else: continue
            param = learnables[name]
            if not isinstance(param, torch.nn.Parameter):
                if strict: raise TypeError(f"Object mapped by '{name}' is not an nn.Parameter.")
                else: continue
            t = to_tensor_like(value, param.data)
            if t is None:
                if strict: raise ValueError(f"Incompatible value for '{name}'.")
                else: continue
            param.data.copy_(t); updated.append(name)
        return updated
    
    def get_attribute(self):
        """
        Retourne une liste (name, value) d'attributs "lisibles" selon la config courante.
        - Inclut toujours u0 par cluster et les edges par cluster.
        - Pour le BULK: renvoie mu (log) ou k_shape (gamma), en version per-cluster (+ global si Gm=True).
        - Pour le TAIL: renvoie (kappa, xi) si egpd, ou mu_tail (log), ou k_shape_tail (gamma),
        en version per-cluster (+ global si Gt=True).
        - Ajoute aussi ('B','...'), ('T','...'), ('L','...') pour tracer la config.
        """
        attrs = []

        # --- u0 & edges ---
        u0_per = self.u0_raw
        attrs.append(('u0_raw', u0_per))
        edges = self._edges_all(device=u0_per.device, dtype=u0_per.dtype, use_sp=False).detach()
        attrs.append(('t_steps', self.t_steps))

        # --- BULK ---
        if self.B == "log":
            if not self.add_global_bulk:
                attrs.append(('mu_bulk_raw', self.mu_bulk_raw.detach()))
            else:
                attrs.append(('mu_bulk_global', self.mu_bulk_global.detach()))
                attrs.append(('mu_bulk_delta', self.mu_bulk_delta.detach()))
        else:  # B == "gamma"
            if not self.add_global_bulk:
                attrs.append(('k_shape_bulk_raw',
                            (self.k_shape_bulk_raw).detach()))
            else:
                k_global = self.k_shape_bulk_global
                k_per    = self.k_shape_bulk_delta
                attrs.append(('k_shape_bulk_global', k_global.detach()))
                attrs.append(('k_shape_bulk_delta', k_per.detach()))

        # --- TAIL ---
        if self.T == "egpd":
            if not self.add_global_tail:
                kappa_per = self.kappa_raw
                xi_per    = self.xi_raw
                attrs.append(('kappa_raw', kappa_per.detach()))
                attrs.append(('xi_raw',    xi_per.detach()))
            else:
                kappa_g = self.kappa_global
                xi_g    = self.xi_global
                kappa_per = self.kappa_delta
                xi_per    = self.xi_delta
                attrs.append(('kappa_global', kappa_g.detach()))
                attrs.append(('xi_global',    xi_g.detach()))
                attrs.append(('kappa_delta', kappa_per.detach()))
                attrs.append(('xi_delta',    xi_per.detach()))

        elif self.T == "log":
            if not self.add_global_tail:
                attrs.append(('mu_bulk_raw', self.mu_tail_raw.detach()))
            else:
                attrs.append(('mu_bulk_global', self.mu_tail_global.detach()))
                attrs.append(('mu_bulk_delta', self.mu_tail_delta.detach()))

        else:  # T == "gamma"
            if not self.add_global_tail:
                k_t_per = self.k_shape_tail_raw
                attrs.append(('k_shape_bulk_raw', k_t_per.detach()))
            else:
                k_t_g   = self.k_shape_tail_global
                k_t_per = self.k_shape_tail_delta
                attrs.append(('k_shape_bulk_global', k_t_g.detach()))
                attrs.append(('k_shape_bulk_delta', k_t_per.detach()))

        return attrs
    
    def plot_params(self, logs, dir_output, cluster_names=None, dpi=120):
        """
        Trace l’évolution des paramètres par epoch (si `logs` fournis) et
        SUPERPOSE la valeur *courante* du modèle sur chaque plot déjà existant.

        Overlays "current":
        - Par cluster (u0, edges, bulk, tail) : ligne horizontale (axhline) pour les scalaires,
            et pour les 'edges' un scatter au dernier epoch pour chaque u_j (évite l'encombrement).
        - Globaux : ligne horizontale sur la figure globale.
        """

        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)

        def to_np(x):
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            return np.asarray(x)

        # ====== (0) Snapshot "current" (toujours calculé pour overlays) ======
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        cur_u0   = to_np(self._u0_all(device, dtype, use_sp=self.force_positive))                 # (C,)
        cur_edges= to_np(self._edges_all(device, dtype, use_sp=self.force_positive))               # (C, K-2)

        # Bulk courant (par cluster + global si besoin)
        if self.B == "log":
            if not self.add_global_bulk:
                cur_mu_bulk_pc = to_np(self.mu_bulk_raw)                               # (C,)
                cur_mu_bulk_g  = None
            else:
                cur_mu_bulk_pc = to_np(self.mu_bulk_global + self.mu_bulk_delta)       # (C,)
                cur_mu_bulk_g  = float(to_np(self.mu_bulk_global))
        else:
            if not self.add_global_bulk:
                kpc = self._sp(self.k_shape_bulk_raw) if self.force_positive else self.k_shape_bulk_raw
                cur_k_shape_bulk_pc = to_np(kpc)                                       # (C,)
                cur_k_shape_bulk_g  = None
            else:
                kg  = self._sp(self.k_shape_bulk_global) if self.force_positive else self.k_shape_bulk_global
                kpc = self._sp(self.k_shape_bulk_global + self.k_shape_bulk_delta) if self.force_positive else (self.k_shape_bulk_global + self.k_shape_bulk_delta)
                cur_k_shape_bulk_pc = to_np(kpc)                                 # (C,)
                cur_k_shape_bulk_g  = float(to_np(kg))

        # Tail courant (par cluster + global si besoin)
        if self.T == "egpd":
            if not self.add_global_tail:
                kap = self._sp(self.kappa_raw)
                xi  = (self._sp(self.xi_raw) + self.xi_min) if self.force_positive else self.xi_raw
                cur_kappa_pc = to_np(kap); cur_xi_pc = to_np(xi)
                cur_kappa_g = cur_xi_g = None
            else:
                kg  = self._sp(self.kappa_global)
                xg  = (self._sp(self.xi_global) + self.xi_min) if self.force_positive else self.xi_global
                kpc = self._sp(self.kappa_global + self.kappa_delta) if self.force_positive else (self.kappa_global + self.kappa_delta)
                xpc = (self._sp(self.xi_global + self.xi_delta) + self.xi_min) if self.force_positive else (self.xi_global + self.xi_delta)
                cur_kappa_pc = to_np(kpc); cur_xi_pc = to_np(xpc)
                cur_kappa_g  = float(to_np(kg)); cur_xi_g = float(to_np(xg))
        elif self.T == "log":
            if not self.add_global_tail:
                cur_mu_tail_pc = to_np(self.mu_tail_raw); cur_mu_tail_g = None
            else:
                cur_mu_tail_pc = to_np(self.mu_tail_global + self.mu_tail_delta)
                cur_mu_tail_g  = float(to_np(self.mu_tail_global))
        else:  # gamma
            if not self.add_global_tail:
                kt  = self._sp(self.k_shape_tail_raw) if self.force_positive else self.k_shape_tail_raw
                cur_k_shape_tail_pc = to_np(kt); cur_k_shape_tail_g = None
            else:
                ktg = self._sp(self.k_shape_tail_global) if self.force_positive else self.k_shape_tail_global
                ktp = self._sp(self.k_shape_tail_global + self.k_shape_tail_delta) if self.force_positive else (self.k_shape_tail_global + self.k_shape_tail_delta)
                cur_k_shape_tail_pc = to_np(ktp)
                cur_k_shape_tail_g  = float(to_np(ktg))

        # ====== (1) PLOTS TEMPORELS (si logs fournis) AVEC OVERLAYS COURANTS ======
        if logs:
            epochs = [int(l["epoch"]) for l in logs]
            C = self.C

            def stack_series(key):
                if key not in logs[0]: return None
                arrs = [to_np(l[key]) for l in logs]
                try:
                    return np.stack(arrs, 0)
                except Exception:
                    return np.asarray([float(a) for a in arrs])  # scalaires globaux

            # Commun
            U0_arr = stack_series('u0_per_cluster')            # (T,C)
            E_arr  = stack_series('edges_per_cluster')         # (T,C,K-2)

            # Bulk
            if self.B == "log":
                MUc_arr = stack_series('mu_bulk_per_cluster')  # (T,C) ou None
                MUg_arr = stack_series('mu_bulk_global')       # (T,)  ou None
            else:
                KBc_arr = stack_series('k_shape_bulk_per_cluster')
                KBg_arr = stack_series('k_shape_bulk_global')

            # Tail
            if self.T == "egpd":
                Kap_arr = stack_series('kappa_per_cluster')
                Xi_arr  = stack_series('xi_per_cluster')
                Kap_g   = stack_series('kappa_global')
                Xi_g    = stack_series('xi_global')
            elif self.T == "log":
                MUt_arr = stack_series('mu_tail_per_cluster')
                MUt_g   = stack_series('mu_tail_global')
            else:
                Kt_arr  = stack_series('k_shape_tail_per_cluster')
                Kt_g    = stack_series('k_shape_tail_global')

            # Noms
            if cluster_names is not None and len(cluster_names) != self.C:
                raise ValueError(f"cluster_names length ({len(cluster_names)}) != C ({self.C})")
            def cname(i): return f"cl{i}" if cluster_names is None else str(cluster_names[i])

            # --- Par cluster ---
            for c in range(self.C):
                name = cname(c)
                cl_dir = out_dir / name
                cl_dir.mkdir(parents=True, exist_ok=True)

                # u0
                if U0_arr is not None:
                    U0_arr[:, c] = F.softplus(torch.as_tensor(U0_arr[:, c], dtype=torch.float32)).detach().cpu().numpy()
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, U0_arr[:, c], marker='D', label='u0 (train)')
                    # overlay current
                    plt.axhline(cur_u0[c], linestyle=':', alpha=0.8, label=f'u0 current={cur_u0[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('u0'); plt.title(f'u0 — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'u0.png', dpi=dpi); plt.close()

                # edges
                if E_arr is not None and E_arr.ndim == 3 and E_arr.shape[2] > 0:
                    plt.figure(figsize=(10,6))
                    Kminus2 = E_arr.shape[2]
                    for j in range(Kminus2):
                        plt.plot(epochs, E_arr[:, c, j], marker='^', linestyle='--', label=f'u{j+1} (train)')
                        # overlay current point au dernier epoch
                        plt.scatter([epochs[-1]], [cur_edges[c, j]], s=36, marker='o', alpha=0.9)
                    plt.xlabel('Epoch'); plt.ylabel('edge'); plt.title(f'edges — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                    plt.tight_layout(); plt.savefig(cl_dir/'edges.png', dpi=dpi); plt.close()

                # bulk
                if self.B == "log" and MUc_arr is not None:
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, MUc_arr[:, c], marker='o', label='mu_bulk (train)')
                    plt.axhline(cur_mu_bulk_pc[c], linestyle=':', alpha=0.8,
                                label=f'mu_bulk current={cur_mu_bulk_pc[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('mu_bulk'); plt.title(f'mu_bulk — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'mu_bulk.png', dpi=dpi); plt.close()

                if self.B == "gamma" and 'KBc_arr' in locals() and KBc_arr is not None:
                    if getattr(self, "force_positive", True):
                        KBc_arr[:, c] = F.softplus(torch.as_tensor(KBc_arr[:, c], dtype=torch.float32)).detach().cpu().numpy()
                    
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, KBc_arr[:, c], marker='o', label='k_shape_bulk (train)')
                    plt.axhline(cur_k_shape_bulk_pc[c], linestyle=':', alpha=0.8,
                                label=f'k_shape_bulk current={cur_k_shape_bulk_pc[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('k_shape_bulk'); plt.title(f'k_shape_bulk — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'k_shape_bulk.png', dpi=dpi); plt.close()

                # tail
                if self.T == "egpd" and 'Kap_arr' in locals() and Kap_arr is not None and Xi_arr is not None:
                    if getattr(self, "force_positive", True):
                        Kap_arr[:, c] = F.softplus(torch.as_tensor(Kap_arr[:, c], dtype=torch.float32)).detach().cpu().numpy()
                        Xi_arr[:, c] = F.softplus(torch.as_tensor(Xi_arr[:, c], dtype=torch.float32)).detach().cpu().numpy()
                    
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, Kap_arr[:, c], marker='o', label='kappa (train)')
                    plt.axhline(cur_kappa_pc[c], linestyle=':', alpha=0.8,
                                label=f'kappa current={cur_kappa_pc[c]:.4g}')
                    plt.plot(epochs, Xi_arr[:, c],  marker='s', label='xi (train)')
                    plt.axhline(cur_xi_pc[c], linestyle=':', alpha=0.8,
                                label=f'xi current={cur_xi_pc[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('value'); plt.title(f'tail (eGPD) — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                    plt.tight_layout(); plt.savefig(cl_dir/'tail_egpd.png', dpi=dpi); plt.close()

                if self.T == "log" and 'MUt_arr' in locals() and MUt_arr is not None:
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, MUt_arr[:, c], marker='o', label='mu_tail (train)')
                    plt.axhline(cur_mu_tail_pc[c], linestyle=':', alpha=0.8,
                                label=f'mu_tail current={cur_mu_tail_pc[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('mu_tail'); plt.title(f'tail (lognorm) — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'tail_log.png', dpi=dpi); plt.close()

                if self.T == "gamma" and 'Kt_arr' in locals() and Kt_arr is not None:
                    if getattr(self, "force_positive", True):
                        Kt_arr[:, c] = F.softplus(torch.as_tensor(Kt_arr[:, c], dtype=torch.float32)).detach().cpu().numpy()
                    plt.figure(figsize=(9,4.8))
                    plt.plot(epochs, Kt_arr[:, c], marker='o', label='k_shape_tail (train)')
                    plt.axhline(cur_k_shape_tail_pc[c], linestyle=':', alpha=0.8,
                                label=f'k_shape_tail current={cur_k_shape_tail_pc[c]:.4g}')
                    plt.xlabel('Epoch'); plt.ylabel('k_shape_tail'); plt.title(f'tail (gamma) — {name}')
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.tight_layout(); plt.savefig(cl_dir/'tail_gamma.png', dpi=dpi); plt.close()

            # ---- Globaux (si présents) + overlays courants ----
            global_dir = out_dir / 'global'; global_dir.mkdir(parents=True, exist_ok=True)

            if self.B == "log" and 'MUg_arr' in locals() and MUg_arr is not None:
                plt.figure(figsize=(8,5))
                plt.plot(epochs, MUg_arr, marker='o', label='mu_bulk_global (train)')
                if cur_mu_bulk_g is not None:
                    plt.axhline(cur_mu_bulk_g, linestyle=':', alpha=0.9,
                                label=f'current={cur_mu_bulk_g:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('mu_bulk_global'); plt.title('bulk (log) — global')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(global_dir/'bulk_log_global.png', dpi=dpi); plt.close()

            if self.B == "gamma" and 'KBg_arr' in locals() and KBg_arr is not None:
                if getattr(self, "force_positive", True):
                    KBg_arr = F.softplus(torch.as_tensor(KBg_arr, dtype=torch.float32)).detach().cpu().numpy()
                plt.figure(figsize=(8,5))
                plt.plot(epochs, KBg_arr, marker='o', label='k_shape_bulk_global (train)')
                if cur_k_shape_bulk_g is not None:
                    plt.axhline(cur_k_shape_bulk_g, linestyle=':', alpha=0.9,
                                label=f'current={cur_k_shape_bulk_g:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('k_shape_bulk_global'); plt.title('bulk (gamma) — global')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(global_dir/'bulk_gamma_global.png', dpi=dpi); plt.close()

            if self.T == "egpd" and (('Kap_g' in locals() and Kap_g is not None) or ('Xi_g' in locals() and Xi_g is not None)):
                plt.figure(figsize=(8,5))
                if 'Kap_g' in locals() and Kap_g is not None:
                    Kap_g = F.softplus(torch.as_tensor(Kap_g, dtype=torch.float32)).detach().cpu().numpy()
                    plt.plot(epochs, Kap_g, color='blue', marker='o', label='kappa_global (train)')
                    if cur_kappa_g is not None:
                        plt.axhline(cur_kappa_g, colr='blue', linestyle=':', alpha=0.9,
                                    label=f'kappa current={cur_kappa_g:.4g}')
                if 'Xi_g' in locals() and Xi_g is not None:
                    if getattr(self, "force_positive", True):
                        Xi_g = F.softplus(torch.as_tensor(Xi_g, dtype=torch.float32)).detach().cpu().numpy()
                    plt.plot(epochs, Xi_g, color='red', marker='s', label='xi_global (train)')
                    if cur_xi_g is not None:
                        plt.axhline(cur_xi_g, color='red', linestyle=':', alpha=0.9,
                                    label=f'xi current={cur_xi_g:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('value'); plt.title('tail (eGPD) — global')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(global_dir/'tail_egpd_global.png', dpi=dpi); plt.close()

            if self.T == "log" and 'MUt_g' in locals() and MUt_g is not None:
                plt.figure(figsize=(8,5))
                plt.plot(epochs, MUt_g, marker='o', label='mu_tail_global (train)')
                if cur_mu_tail_g is not None:
                    plt.axhline(cur_mu_tail_g, linestyle=':', alpha=0.9,
                                label=f'current={cur_mu_tail_g:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('mu_tail_global'); plt.title('tail (log) — global')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(global_dir/'tail_log_global.png', dpi=dpi); plt.close()

            if self.T == "gamma" and 'Kt_g' in locals() and Kt_g is not None:
                if getattr(self, "force_positive", True):
                    Kt_g = F.softplus(torch.as_tensor(Kt_g, dtype=torch.float32)).detach().cpu().numpy()
                plt.figure(figsize=(8,5))
                plt.plot(epochs, Kt_g, marker='o', label='k_shape_tail_global (train)')
                if cur_k_shape_tail_g is not None:
                    plt.axhline(cur_k_shape_tail_g, linestyle=':', alpha=0.9,
                                label=f'current={cur_k_shape_tail_g:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('k_shape_tail_global'); plt.title('tail (gamma) — global')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(global_dir/'tail_gamma_global.png', dpi=dpi); plt.close()

        # =========================
        # C) COURANTS vs CLUSTERS
        # =========================
        C = self.C
        xlabels = [f"cl{c}" if cluster_names is None else str(cluster_names[c]) for c in range(C)]
        x = np.arange(C)

        # u0 vs clusters
        plt.figure(figsize=(10,4.8))
        plt.bar(x, cur_u0)
        plt.xticks(x, xlabels, rotation=90, ha='right')
        plt.ylabel('u0'); plt.title('Current u0 vs clusters')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig(Path(dir_output)/'current_u0_vs_clusters.png', dpi=dpi); plt.close()

        # edges vs clusters (une courbe par indice d’edge)
        if cur_edges.shape[1] > 0:
            plt.figure(figsize=(11,6))
            for j in range(cur_edges.shape[1]):
                plt.plot(x, cur_edges[:, j], marker='o', label=f'u{j+1}')
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('edge value'); plt.title('Current edges vs clusters')
            plt.grid(True, linestyle='--', alpha=0.35); plt.legend(ncol=2)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_edges_vs_clusters.png', dpi=dpi); plt.close()

        # Bulk courant vs clusters
        if self.B == "log":
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_mu_bulk_pc)
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('mu_bulk'); plt.title('Current mu_bulk vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_mu_bulk_vs_clusters.png', dpi=dpi); plt.close()
        else:  # gamma
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_k_shape_bulk_pc)
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('k_shape_bulk'); plt.title('Current k_shape_bulk vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_kshape_bulk_vs_clusters.png', dpi=dpi); plt.close()

        # Tail courant vs clusters
        if self.T == "egpd":
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_kappa_pc)
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('kappa'); plt.title('Current kappa (tail) vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_kappa_tail_vs_clusters.png', dpi=dpi); plt.close()

            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_xi_pc)
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('xi'); plt.title('Current xi (tail) vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_xi_tail_vs_clusters.png', dpi=dpi); plt.close()

        elif self.T == "log":
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_mu_tail_pc)
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('mu_tail'); plt.title('Current mu_tail vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_mu_tail_vs_clusters.png', dpi=dpi); plt.close()

        else:  # gamma
            plt.figure(figsize=(18,4.8))
            plt.bar(x, cur_k_shape_tail_pc)
            plt.xticks(x, xlabels, rotation=90, ha='right')
            plt.ylabel('k_shape_tail'); plt.title('Current k_shape_tail vs clusters')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(Path(dir_output)/'current_kshape_tail_vs_clusters.png', dpi=dpi); plt.close()

    def plot_final_pdf(self,
                   scale_tail: torch.Tensor,      # (N,)
                   scale_bulk: torch.Tensor,      # (N,)
                   pi_tail_logit: torch.Tensor,   # (N,)
                   clusters_ids: torch.Tensor,     # (N,)
                   dir_output,
                   y_max: float = None,
                   num_points: int = 400,
                   dpi: int = 120,
                   stat: str = "median",
                   output_pdf: str = "final_cdf"):
        """
        Trace la PDF du mélange Bulk+Tail pour CHAQUE CLUSTER et sauvegarde dans:
            dir_output/cl{c}/<output_pdf>.png

        - Agrège par cluster (median ou mean) les échelles per-sample: scale_tail, scale_bulk et pi.
        - Utilise les paramètres de cluster (mu/k_shape pour Bulk, kappa/xi ou mu_tail ou k_shape_tail pour Tail).
        - Annote les edges u0, u1, ...
        """

        from pathlib import Path
        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)
        device, dtype = scale_tail.device, scale_tail.dtype

        red = torch.median if stat == "median" else torch.mean

        # --- petites PDFs
        def pdf_lognormal(yv, mu_log, sigma_ln, eps=1e-8):
            yv = yv.clamp_min(eps)
            z  = (torch.log(yv) - mu_log) / sigma_ln
            return torch.exp(-0.5 * z * z) / (yv * sigma_ln * math.sqrt(2.0 * math.pi))

        def pdf_gamma_k_theta(yv, k_, theta_, eps=1e-12):
            k_ = k_.clamp_min(eps); theta_ = theta_.clamp_min(eps)
            yv = yv.clamp_min(eps)
            # log-pdf stable
            log_pdf = (k_ - 1.0) * torch.log(yv) - (yv / theta_) - torch.lgamma(k_) - k_ * torch.log(theta_)
            return torch.exp(log_pdf)

        def pdf_egpd(yv, sigma_t, xi_v, kappa_v, eps=1e-12):
            yv = yv.clamp_min(0.0)
            si = sigma_t.clamp_min(eps)
            xv = xi_v.clamp_min(eps)
            kv = kappa_v.clamp_min(eps)
            z  = (1.0 + xv * (yv / si)).clamp_min(1.0 + 1e-12)
            H  = (1.0 - torch.pow(z, -1.0 / xv)).clamp(min=eps, max=1.0 - eps)
            h  = (1.0 / si) * torch.pow(z, -1.0 / xv - 1.0)
            return kv * h * torch.pow(H, kv - 1.0)

        # Edges par cluster (pour annotation)
        u0_all  = self._u0_all(device, dtype)                      # (C,)
        E_all   = self._edges_all(device, dtype)                   # (C, K-2)

        cid = clusters_ids.long()

        for c in range(self.C):
            cl_mask = (cid == c)
            cl_dir  = out_dir / f"cl{c}"
            cl_dir.mkdir(parents=True, exist_ok=True)

            # Agrégats représentatifs par cluster (si aucun sample => valeurs par défaut raisonnables)
            if cl_mask.any():
                st_c = red((F.softplus(scale_tail[cl_mask]) + 0.05))
                sb_c = red((F.softplus(scale_bulk[cl_mask]) + 0.05))
                pi_c = red(torch.sigmoid(pi_tail_logit[cl_mask]).clamp(1e-12, 1-1e-12))
            else:
                st_c = torch.tensor(1.0, device=device, dtype=dtype)
                sb_c = torch.tensor(0.5, device=device, dtype=dtype)
                pi_c = torch.tensor(0.1, device=device, dtype=dtype)

            # Paramètres de cluster (BULK & TAIL)
            bulk_param_c = self._select_bulk_param(torch.tensor([c], device=device))  # mu_c ou k_c
            # _select_bulk_param applique déjà softplus pour k si force_positive=True
            bulk_param_c = bulk_param_c.squeeze(0)

            tail_params_c = self._select_tail_params(torch.tensor([c], device=device))
            # pour egpd: dict {"kappa","xi"} (déjà softplus/xi_min si force_positive)
            # pour log : dict {"mu_tail"}
            # pour gamma: dict {"k_shape_tail"} (déjà softplus si force_positive)
            # rien à changer ici

            # Edges pour ce cluster
            u0_c    = u0_all[c]
            edges_c = E_all[c] if E_all.numel() > 0 else torch.empty(0, device=device, dtype=dtype)

            # y_max heuristique
            if y_max is None:
                if edges_c.numel() > 0:
                    ymax = float(edges_c[-1].detach().cpu()) * 2.0
                else:
                    ymax = float(u0_c.detach().cpu()) * 2.0 + 1.0
                ymax = max(ymax, float(u0_c.detach().cpu()) * 1.5 + 1.0, 1.0)
            else:
                ymax = float(y_max)

            y = torch.linspace(0.0, ymax, steps=num_points, device=device, dtype=dtype)

            # PDF bulk
            if self.B == "log":
                f_bulk = pdf_lognormal(y, bulk_param_c, sb_c)
            else:  # gamma
                f_bulk = pdf_gamma_k_theta(y, bulk_param_c, sb_c)

            # PDF tail
            if self.T == "egpd":
                f_tail = pdf_egpd(y, st_c, tail_params_c["xi"].squeeze(0), tail_params_c["kappa"].squeeze(0))
            elif self.T == "log":
                f_tail = pdf_lognormal(y, tail_params_c["mu_tail"].squeeze(0), st_c)
            else:  # gamma
                f_tail = pdf_gamma_k_theta(y, tail_params_c["k_shape_tail"].squeeze(0), st_c)

            f_mix = (1.0 - pi_c) * f_bulk + pi_c * f_tail

            # --- Plot
            y_np = y.detach().cpu().numpy()
            f_np = f_mix.detach().cpu().numpy()
            plt.figure(figsize=(8.8, 5.2))
            plt.plot(y_np, f_np, label=f'f_mix(y) [B={self.B}, T={self.T}]')

            # Concatène u0 et edges pour annoter u0,u1,...
            if edges_c.numel() > 0:
                all_edges = torch.cat([u0_c.view(1), edges_c], dim=0)
            else:
                all_edges = u0_c.view(1)

            names = [f"u{j}" for j in range(all_edges.numel())]
            ymax_txt = (np.nanmax(f_np[np.isfinite(f_np)]) if np.isfinite(f_np).any() else 1.0)
            for name, uj in zip(names, all_edges):
                xval = float(uj.detach().cpu())
                plt.axvline(x=xval, linestyle='--', alpha=0.55, label=name)
                plt.text(xval, ymax_txt*0.9, f"{name}={xval:.3f}",
                        rotation=90, va='top', ha='right', fontsize=9, alpha=0.8)

            f_plot = np.clip(f_np, 1e-12, None)
            plt.xlim(0, ymax)
            plt.yscale('log')  # utile pour voir la queue
            ymin = max(1e-12, f_plot[f_plot > 0].min()*0.8) if np.any(f_plot > 0) else 1e-12
            ymax_plot = f_plot.max()*1.2
            plt.ylim(ymin, ymax_plot)
            plt.xlabel('y'); plt.ylabel('Density (log scale)')
            plt.title(f'Cluster {c} — pi_tail={float(pi_c.detach().cpu()):.4f}')
            plt.legend(ncol=2); plt.grid(True, linestyle='--', alpha=0.35, which='both')
            plt.tight_layout()
            fname = output_pdf if output_pdf is not None else "final_pdf.png"
            if not fname.lower().endswith(".png"):
                fname += ".png"
            plt.savefig(cl_dir / fname, dpi=dpi)
            plt.close()

import math
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- eGPD CDF robuste (xi∈R, gère xi=0) ---
def cdf_egpd_family1(y, sigma, xi, kappa, eps: float = 1e-12):
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    device, dtype = y.device, y.dtype

    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
    xi    = torch.as_tensor(xi,    device=device, dtype=dtype)
    kappa = torch.as_tensor(kappa, device=device, dtype=dtype)

    y  = torch.clamp(y, min=0.0)
    si = torch.clamp(sigma, min=eps)
    ka = torch.clamp(kappa, min=eps)

    is_zero = (xi == 0)
    xi_nz = torch.where(is_zero, torch.ones_like(xi), xi)
    xi_safe = torch.where(xi_nz >= 0,
                          torch.clamp(xi_nz, min=eps),
                          torch.clamp(xi_nz, max=-eps))

    # xi = 0 : limite expo
    H_exp = 1.0 - torch.exp(-y / si)
    H_exp = torch.clamp(H_exp, min=eps, max=1.0 - eps)
    F_exp = torch.pow(H_exp, ka)

    # xi ≠ 0
    neg_xi = (xi < 0)
    y_max = -si / xi_safe
    valid_range = torch.where(neg_xi, y < y_max, torch.ones_like(y, dtype=torch.bool))

    z = 1.0 + xi_safe * (y / si)
    z = torch.where(valid_range, torch.clamp(z, min=1.0 + 1e-12), torch.ones_like(z, dtype=dtype))

    H = 1.0 - torch.pow(z, -1.0 / xi_safe)
    H = torch.clamp(H, min=eps, max=1.0 - eps)
    F_nz = torch.pow(H, ka)

    F_nz = torch.where(neg_xi & (~valid_range), torch.ones_like(F_nz), F_nz)
    F = torch.where(is_zero, F_exp, F_nz)
    return F

class IntervalCELosEdges(nn.Module):
    """
    Interval CE où les *paramètres de loi* sont prédits par le modèle (passés dans `inputs`),
    tandis que les *edges* sont appris dans la loss (u0_raw, t_steps sont des nn.Parameter).

    T ∈ {'egpd','log','gamma'}

    Inputs attendus par échantillon (sortie du modèle):
      - T='egpd'  : [sigma, kappa, xi]
      - T='log'   : [sigma_ln, mu_log]
      - T='gamma' : [theta, k_shape]
    """

    def __init__(self,
                 num_classes: int = 5,
                 u0_init: float = 0.5,
                 force_positive: bool = False,   # n'affecte que xi si True
                 xi_min: float = 1e-12,
                 eps: float = 1e-8,
                 L: str = "entropy",
                 reduction: str = "mean",
                 T: str = "egpd"):
        
        super().__init__()
        assert num_classes >= 2
        assert T in {"egpd","log","gamma"}
        assert L in {"entropy","mcewk","bceloss"}

        self.num_classes    = num_classes
        self.T              = T
        self.L              = L
        self.reduction      = reduction
        self.force_positive = force_positive
        self.xi_min         = xi_min
        self.eps            = eps

        # --- Edges (appris par la loss) ---
        self.u0_raw  = nn.Parameter(torch.tensor(float(u0_init)))
        base_steps   = torch.arange(1, num_classes - 1, dtype=torch.float32)  # (K-2,)
        self.t_steps = nn.Parameter(base_steps.clone())

    # ======== utils edges ========
    @staticmethod
    def _sp(x):  # softplus stable
        return F.softplus(x) + 1e-12

    def _u0(self, device, dtype):
        # Toujours softplus à u0 (exigence)
        return self._sp(self.u0_raw).to(device=device, dtype=dtype)

    def _edges_tensor(self, device, dtype):
        if self.num_classes <= 2:
            return torch.empty(0, device=device, dtype=dtype)
        steps = self._sp(self.t_steps.to(device=device, dtype=dtype))  # (K-2,)
        u0    = self._u0(device, dtype)                                # scalar
        return u0 + torch.cumsum(steps, dim=0)                         # (K-1,)

    # ======== CDF primitives ========
    @staticmethod
    def _cdf_lognormal(u, mu_log, sigma_ln, eps=1e-8):
        pos = (u > 0)
        z = (torch.log(u.clamp(min=eps)) - mu_log) / (sigma_ln * math.sqrt(2.0))
        F_ln = 0.5 * (1.0 + torch.erf(z))
        return torch.where(pos, F_ln, torch.zeros_like(F_ln))

    def _cdf_gamma_k_theta(self, u, k, theta):
        """
        CDF Gamma(k, θ) = P(k, u/θ).  ATTENTION: pas de grad w.r.t k dans PyTorch.
        On détache k pour éviter l'exception (sinon RuntimeError sur igamma).
        """
        k_det = k.clamp_min(self.eps).detach()
        theta = theta.clamp_min(self.eps)
        x = (u / theta).clamp_min(0.0)
        try:
            Pkx = torch.special.gammainc(k_det, x)
        except AttributeError:
            Pkx = torch.igamma(k_det, x) / torch.exp(torch.lgamma(k_det))
        return Pkx

    # ======== CDF principale (selon T) ========
    def _cdf_primary(self, u, inputs_row):
        """
        u:  Tensor (...,)
        inputs_row: Tensor (..., D_in) : paramètres prédits par le modèle
        """
        if self.T == "egpd":
            sigma = self._sp(inputs_row[..., 0])
            kappa = self._sp(inputs_row[..., 1])
            xi    = inputs_row[..., 2]
            if self.force_positive:
                xi = self._sp(xi) + self.xi_min
            return cdf_egpd_family1(u.clamp_min(0.0), sigma, xi, kappa, eps=1e-12)

        elif self.T == "log":
            sigma_ln = self._sp(inputs_row[..., 0])
            mu_log   = inputs_row[..., 1]
            return self._cdf_lognormal(u, mu_log, sigma_ln, self.eps)

        else:  # gamma
            theta = self._sp(inputs_row[..., 0])
            k     = self._sp(inputs_row[..., 1])  # contrainte k>0
            return self._cdf_gamma_k_theta(u, k, theta)

    # ======== transform (inférence/diag) ========
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output: Path, output_pdf: str) -> torch.Tensor:
        """
        inputs: (N, D) — paramètres prédits par le modèle (voir convention).
        Retourne P ∈ (N, K).
        """
        x = inputs
        if x.ndim == 1:
            raise ValueError("inputs must be 2D: (N, D) according to T.")
        device, dtype = x.device, x.dtype
        N, K = x.shape[0], self.num_classes

        # Plot diag (PDF)
        self.plot_final_pdf(x, dir_output, output_pdf=output_pdf, stat="median")

        # Edges
        u0    = self._u0(device, dtype)
        edges = self._edges_tensor(device, dtype)

        # CDF aux bords
        F_list = [ self._cdf_primary(u0.expand(N), x) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges[j].expand(N), x))

        # Probas par différences
        Ps = [F_list[0]] + [torch.clamp(F_list[j+1] - F_list[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_list[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)              # (N, K)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)
        return P

    # ======== forward (loss) ========
    def forward(self,
                inputs: torch.Tensor,   # (N, D) paramètres prédits (selon T)
                y_class: torch.Tensor,  # (N,) entiers 0..K-1
                weight: torch.Tensor = None) -> torch.Tensor:

        if inputs.ndim == 1:
            raise ValueError("inputs must be 2D: (N, D) according to T.")
        device = inputs.device
        N, K = inputs.shape[0], self.num_classes

        # Edges
        u0    = self._u0(device, inputs.dtype)
        edges = self._edges_tensor(device, inputs.dtype)

        # CDF aux bords
        F_list = [ self._cdf_primary(u0.expand(N), inputs) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges[j].expand(N), inputs))

        # Probas
        Ps = [F_list[0]] + [torch.clamp(F_list[j+1] - F_list[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_list[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)

        # Loss
        if self.L == "entropy":
            logP = torch.log(P.clamp_min(1e-12))
            loss = F.nll_loss(logP, y_class.long(), reduction='none')
        elif self.L == "mcewk":
            loss = MCEAndWKLoss(num_classes=self.num_classes, use_logits=False).forward(P, y_class)
        elif self.L == "bceloss":
            loss = BCELoss(num_classes=self.num_classes).forward(P, y_class)
        else:
            raise ValueError(f"L unknown: {self.L}")

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    # ======== Plot PDF (diag) ========
    def plot_final_pdf(self,
                       inputs: torch.Tensor,   # (N, D) paramètres prédits batch
                       dir_output,
                       y_max: float = None,
                       num_points: int = 400,
                       dpi: int = 120,
                       stat: str = "median",
                       output_pdf: str = "final_pdf"):
        """
        Trace la PDF de la distribution choisie par T avec annotations des edges.
        Les paramètres sont agrégés (mean/median) sur le batch.
        """
        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)
        device, dtype = inputs.device, inputs.dtype
        K = self.num_classes

        red = torch.median if (stat == "median") else torch.mean

        # Agrégats des paramètres par T
        if self.T == "egpd":
            sigma = self._sp(red(inputs[:, 0]))
            kappa = self._sp(red(inputs[:, 1]))
            xi    = red(inputs[:, 2])
            if self.force_positive:
                xi = self._sp(xi) + self.xi_min
            title = "EGPD PDF"
        elif self.T == "log":
            sigma = self._sp(red(inputs[:, 0]))
            mu_log= red(inputs[:, 1])
            title = "LogNormal PDF"
        else:  # gamma
            theta = self._sp(red(inputs[:, 0]))
            k     = self._sp(red(inputs[:, 1]))
            title = "Gamma PDF"

        # Edges (toujours softplus)
        u0 = self._sp(self.u0_raw.to(device=device, dtype=dtype))
        if K > 2:
            steps = self._sp(self.t_steps.to(device=device, dtype=dtype))
            edges = u0 + torch.cumsum(steps, dim=0)
            all_edges = torch.cat([u0.view(1), edges], dim=0)
        else:
            all_edges = u0.view(1)

        # y_max heuristique
        if y_max is None:
            ymax = float(all_edges[-1].detach().cpu()) * 2.0
            ymax = max(ymax, float(u0.detach().cpu()) * 1.5 + 1.0, 1.0)
        else:
            ymax = float(y_max)

        y = torch.linspace(0.0, ymax, steps=num_points, device=device, dtype=dtype)

        # PDFs
        def pdf_lognormal(yv, mu_log_, sigma_ln_, eps=1e-8):
            yv = yv.clamp_min(eps)
            z  = (torch.log(yv) - mu_log_) / sigma_ln_
            return torch.exp(-0.5 * z * z) / (yv * sigma_ln_ * math.sqrt(2.0 * math.pi))

        def pdf_gamma_k_theta(yv, k_, theta_, eps=1e-12):
            k_ = k_.clamp_min(eps); theta_ = theta_.clamp_min(eps)
            yv = yv.clamp_min(eps)
            log_pdf = (k_ - 1.0) * torch.log(yv) - (yv / theta_) - torch.lgamma(k_) - k_ * torch.log(theta_)
            return torch.exp(log_pdf)

        def pdf_egpd(yv, sigma_t, xi_v, kappa_v, eps=1e-12):
            yv = yv.clamp_min(0.0)
            si = sigma_t.clamp_min(eps)
            kv = kappa_v.clamp_min(eps)
            xv = xi_v if not self.force_positive else torch.clamp(xi_v, min=self.xi_min)
            z  = torch.clamp(1.0 + xv * (yv / si), min=1.0 + 1e-12)
            H  = (1.0 - torch.pow(z, -1.0 / xv)).clamp(min=eps, max=1.0 - eps)
            h  = (1.0 / si) * torch.pow(z, -1.0 / xv - 1.0)
            return kv * h * torch.pow(H, kv - 1.0)

        if self.T == "egpd":
            f = pdf_egpd(y, sigma, xi, kappa)
        elif self.T == "log":
            f = pdf_lognormal(y, mu_log, sigma)
        else:
            f = pdf_gamma_k_theta(y, k, theta)

        # Plot
        y_np = y.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(y_np, f_np, label=title)

        names = [f"u{j}" for j in range(all_edges.numel())]
        for name, uj in zip(names, all_edges):
            xval = float(uj.detach().cpu())
            plt.axvline(x=xval, linestyle='--', alpha=0.55, label=name)
            ymax_txt = (np.nanmax(f_np) * 0.9) if np.isfinite(f_np).any() else 0.9
            plt.text(xval, ymax_txt, f"{name}={xval:.3f}",
                     rotation=90, va='top', ha='right', fontsize=9, alpha=0.8)

        plt.xlim(0, ymax)
        top = (np.nanmax(f_np) * 1.05) if np.isfinite(f_np).any() else 1.0
        plt.ylim(0, top)
        plt.xlabel('y'); plt.ylabel('Density')
        plt.title(title)
        plt.legend(ncol=2)
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.savefig(Path(dir_output) / (output_pdf if output_pdf else "final_pdf.png"), dpi=dpi)
        plt.close()

    # ======== Fonctions utiles à l'entraînement ========
    def get_learnable_parameters(self):
        """Ne renvoie que les paramètres appris par la loss: u0_raw, t_steps."""
        return {"u0_raw": self.u0_raw, "t_steps": self.t_steps}

    def get_attribute(self):
        """Expose les tenseurs bruts pour debug/trace (sans softplus)."""
        return [
            ("u0_raw", self.u0_raw.detach().clone()),
            ("t_steps", self.t_steps.detach().clone()),
        ]

    def update_params(self, new_values: dict, strict: bool = False):
        learnables = self.get_learnable_parameters()
        updated = []

        def to_tensor_like(x, ref: torch.Tensor):
            t = torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
            if t.ndim == 0 and ref.numel() > 1:
                t = t.expand_as(ref)
            if t.shape != ref.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{name}': got {tuple(t.shape)}, expected {tuple(ref.shape)}")
                try:
                    t = t.expand_as(ref)
                except Exception:
                    return None
            return t

        for name, value in new_values.items():
            if name not in learnables:
                if strict: raise KeyError(f"Unknown learnable parameter '{name}'.")
                else: continue
            param = learnables[name]
            t = to_tensor_like(value, param.data)
            if t is None:
                if strict: raise ValueError(f"Incompatible value for '{name}'.")
                else: continue
            param.data.copy_(t)
            updated.append(name)
        return updated

    def plot_params(self, logs, dir_output, dpi=120):
        """
        Plots d'u0 et des edges.
        Si `logs` fournis: séries par epoch + overlay des valeurs *courantes* (contraintes).
        """
        out = Path(dir_output); out.mkdir(parents=True, exist_ok=True)

        def to_np(x):
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            return np.asarray(x)

        device = self.u0_raw.device
        dtype  = self.u0_raw.dtype

        cur_u0    = float(to_np(self._sp(self.u0_raw)))
        if self.num_classes > 2:
            cur_edges = to_np(self._sp(self.u0_raw) + torch.cumsum(self._sp(self.t_steps.to(device=device, dtype=dtype)), dim=0))
        else:
            cur_edges = np.empty((0,), dtype=float)

        if logs:
            epochs = [int(l["epoch"]) for l in logs]

            def seq(key):
                if key not in logs[0]: return None
                vals = [l[key] for l in logs]
                try:
                    return np.asarray([float(v) for v in vals])
                except Exception:
                    arrs = [to_np(v) for v in vals]
                    try:
                        return np.stack(arrs, 0)
                    except Exception:
                        return np.asarray(arrs)

            # u0
            u0_seq = seq("u0")
            if u0_seq is not None:
                u0_seq = F.softplus(torch.as_tensor(u0_seq, dtype=torch.float32)).detach().cpu().numpy()
                plt.figure(figsize=(8.2,4.6))
                plt.plot(epochs, u0_seq, marker='o', label='u0 (train)')
                plt.axhline(cur_u0, linestyle=':', alpha=0.9, label=f'u0 current={cur_u0:.4g}')
                plt.xlabel('Epoch'); plt.ylabel('u0'); plt.title('u0')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                plt.tight_layout(); plt.savefig(out/'u0.png', dpi=dpi); plt.close()

            # edges
            edges_seq = seq("edges")
            if edges_seq is not None and edges_seq.ndim == 2 and edges_seq.shape[1] > 0:
                plt.figure(figsize=(10,6))
                for j in range(edges_seq.shape[1]):
                    plt.plot(epochs, edges_seq[:, j], marker='^', linestyle='--', label=f'u{j+1} (train)')
                    if j < len(cur_edges):
                        plt.scatter([epochs[-1]], [cur_edges[j]], s=36, marker='o', alpha=0.9)
                plt.xlabel('Epoch'); plt.ylabel('edge'); plt.title('edges')
                plt.grid(True, linestyle='--', alpha=0.4); plt.legend(ncol=2)
                plt.tight_layout(); plt.savefig(out/'edges.png', dpi=dpi); plt.close()
        else:
            # Snapshots courants
            plt.figure(); plt.axhline(cur_u0, linestyle=':', alpha=0.9, label=f'u0={cur_u0:.4g}')
            plt.title('u0 (current)'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout(); plt.savefig(out/'u0_current.png', dpi=dpi); plt.close()

            if cur_edges.size > 0:
                plt.figure(); plt.bar(range(1, cur_edges.size+1), cur_edges)
                plt.xlabel('edge index'); plt.title('edges (current)')
                plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
                plt.savefig(out/'edges_current.png', dpi=dpi); plt.close()

class IntervalCELoss_AllModelParams(nn.Module):
    """
    Interval CE où T ∈ {'egpd','log','gamma'} et
    *tous* les paramètres (loi + edges) sont fournis par le modèle via `inputs`.

    Entrées du modèle, par échantillon (N lignes) :
      - T='egpd'  : [sigma, kappa, xi, u0_raw, t_step1_raw, ..., t_step(K-2)_raw]
      - T='log'   : [sigma_ln, mu_log, u0_raw, t_step1_raw, ..., t_step(K-2)_raw]
      - T='gamma' : [theta, k_shape, u0_raw, t_step1_raw, ..., t_step(K-2)_raw]

    La loss ne contient AUCUN paramètre entraînable.
    """

    def __init__(self,
                 num_classes: int = 5,
                 T: str = "egpd",                 # 'egpd' | 'log' | 'gamma'
                 L: str = "entropy",              # 'entropy' | 'mcewk' | 'bceloss'
                 force_positive: bool = False,    # n'affecte que xi si True
                 xi_min: float = 1e-12,
                 eps: float = 1e-8,
                 reduction: str = "mean"):
        super().__init__()
        assert num_classes >= 2
        assert T in {"egpd","log","gamma"}
        assert L in {"entropy","mcewk","bceloss"}

        self.num_classes    = num_classes
        self.T              = T
        self.L              = L
        self.force_positive = force_positive
        self.xi_min         = xi_min
        self.eps            = eps
        self.reduction      = reduction

    # -------- utils --------
    @staticmethod
    def _sp(x):          # softplus stable
        return F.softplus(x) + 1e-12

    def _check_shape(self, inputs: torch.Tensor):
        K = self.num_classes
        need_steps = max(0, K - 2)
        base = 3 if self.T == "egpd" else 2   # nb de params de loi
        expected_D = base + 1 + need_steps    # + u0_raw + (K-2) t_steps_raw
        if inputs.ndim != 2 or inputs.shape[1] != expected_D:
            raise ValueError(
                f"inputs must be (N,{expected_D}) for T='{self.T}' and K={K}, got {tuple(inputs.shape)}"
            )

    # -------- edges par échantillon à partir d'inputs --------
    def _edges_from_inputs(self, inputs: torch.Tensor):
        """
        inputs: (N, D)
        retourne:
          - u0:    (N,)
          - edges: (N, K-2)   (vide si K<=2)
        """
        N = inputs.shape[0]
        K = self.num_classes
        base = 3 if self.T == "egpd" else 2
        u0_raw = inputs[:, base]                   # (N,)
        u0     = self._sp(u0_raw)                  # u0 >= 0 (toujours)
        if K <= 2:
            edges = inputs.new_zeros((N, 0))
        else:
            t_steps_raw = inputs[:, base+1 : base+1 + (K-2)]  # (N, K-2)
            steps       = self._sp(t_steps_raw)               # > 0
            edges = u0.unsqueeze(1) + torch.cumsum(steps, dim=1)  # (N, K-2)
        return u0, edges

    # -------- CDFs élémentaires --------
    @staticmethod
    def _cdf_lognormal(u, mu_log, sigma_ln, eps=1e-8):
        pos = (u > 0)
        z = (torch.log(u.clamp(min=eps)) - mu_log) / (sigma_ln * math.sqrt(2.0))
        F_ln = 0.5 * (1.0 + torch.erf(z))
        return torch.where(pos, F_ln, torch.zeros_like(F_ln))

    def _cdf_gamma_k_theta(self, u, k, theta):
        """
        CDF Gamma(k, θ) = P(k, u/θ). Détache k (limitation autograd igamma).
        """
        k_det = k.clamp_min(self.eps).detach()
        theta = theta.clamp_min(self.eps)
        x = (u / theta).clamp_min(0.0)
        try:
            Pkx = torch.special.gammainc(k_det, x)
        except AttributeError:
            Pkx = torch.igamma(k_det, x) / torch.exp(torch.lgamma(k_det))
        return Pkx

    # -------- CDF principale selon T, en lot (N,) --------
    def _cdf_primary(self, u: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        u:      (N,)  — valeur de bord par échantillon
        inputs: (N,D) — paramètres prédits par le modèle
        """
        if self.T == "egpd":
            sigma = self._sp(inputs[:, 0])
            kappa = self._sp(inputs[:, 1])
            xi    = inputs[:, 2]
            if self.force_positive:
                xi = self._sp(xi) + self.xi_min
            return cdf_egpd_family1(u.clamp_min(0.0), sigma, xi, kappa, eps=1e-12)

        elif self.T == "log":
            sigma_ln = self._sp(inputs[:, 0])
            mu_log   = inputs[:, 1]
            return self._cdf_lognormal(u, mu_log, sigma_ln, self.eps)

        else:  # gamma
            theta = self._sp(inputs[:, 0])
            k     = self._sp(inputs[:, 1])
            return self._cdf_gamma_k_theta(u, k, theta)

    # -------- transform (inférence/diag) --------
    @torch.no_grad()
    def transform(self, inputs: torch.Tensor, dir_output: Path, output_pdf: str) -> torch.Tensor:
        """
        inputs: (N, D) — paramètres de loi + edges (u0_raw + t_steps_raw...)
        Retourne P ∈ (N, K).
        """
        self._check_shape(inputs)
        device, dtype = inputs.device, inputs.dtype
        N, K = inputs.shape[0], self.num_classes

        # Plot diag (PDF) à partir des paramètres agrégés
        self.plot_final_pdf(inputs, dir_output, output_pdf=output_pdf, stat="median")

        # Edges par échantillon
        u0_vec, edges_mat = self._edges_from_inputs(inputs)  # (N,), (N,K-2)

        # CDF aux bords
        F_list = [ self._cdf_primary(u0_vec, inputs) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges_mat[:, j], inputs))

        # Probas par différences
        Ps = [F_list[0]] + [torch.clamp(F_list[j+1] - F_list[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_list[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)  # (N, K)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)
        return P

    # -------- forward (loss) --------
    def forward(self,
                inputs: torch.Tensor,   # (N,D) paramètres loi + edges (prédits)
                y_class: torch.Tensor,  # (N,) entiers 0..K-1
                weight: torch.Tensor = None) -> torch.Tensor:

        self._check_shape(inputs)
        N, K = inputs.shape[0], self.num_classes

        # Edges (par échantillon)
        u0_vec, edges_mat = self._edges_from_inputs(inputs)  # (N,), (N,K-2)

        # CDF aux bords
        F_list = [ self._cdf_primary(u0_vec, inputs) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges_mat[:, j], inputs))

        # Probas
        Ps = [F_list[0]] + [torch.clamp(F_list[j+1] - F_list[j], 1e-12, 1.0) for j in range(K-2)]
        Ps.append(torch.clamp(1.0 - F_list[-1], 1e-12, 1.0))
        P = torch.stack(Ps, dim=-1)
        P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)

        # Loss
        if self.L == "entropy":
            logP = torch.log(P.clamp_min(1e-12))
            loss = F.nll_loss(logP, y_class.long(), reduction='none')
        elif self.L == "mcewk":
            loss = MCEAndWKLoss(num_classes=self.num_classes, use_logits=False).forward(P, y_class)
        elif self.L == "bceloss":
            loss = BCELoss(num_classes=self.num_classes).forward(P, y_class)
        else:
            raise ValueError(f"L unknown: {self.L}")

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    # -------- Plot PDF (diag) --------
    def plot_final_pdf(self,
                       inputs: torch.Tensor,   # (N, D)
                       dir_output,
                       y_max: float = None,
                       num_points: int = 400,
                       dpi: int = 120,
                       stat: str = "median",
                       output_pdf: str = "final_pdf"):
        """
        Trace la PDF de la distribution choisie par T avec annotations des edges,
        à partir des *paramètres prédits* agrégés (mean/median) sur le batch.
        """
        out_dir = Path(dir_output); out_dir.mkdir(parents=True, exist_ok=True)
        device, dtype = inputs.device, inputs.dtype
        K = self.num_classes

        red = torch.median if (stat == "median") else torch.mean
        base = 3 if self.T == "egpd" else 2

        # ---- Agrégats loi ----
        if self.T == "egpd":
            sigma = self._sp(red(inputs[:, 0]))
            kappa = self._sp(red(inputs[:, 1]))
            xi    = red(inputs[:, 2])
            if self.force_positive:
                xi = self._sp(xi) + self.xi_min
            title = "EGPD PDF"
        elif self.T == "log":
            sigma = self._sp(red(inputs[:, 0]))  # sigma_ln
            mu_log= red(inputs[:, 1])
            title = "LogNormal PDF"
        else:  # gamma
            theta = self._sp(red(inputs[:, 0]))
            k     = self._sp(red(inputs[:, 1]))
            title = "Gamma PDF"

        # ---- Agrégats edges ----
        u0    = self._sp(red(inputs[:, base]))  # u0 >= 0
        if K > 2:
            t_steps_raw = inputs[:, base+1 : base+1 + (K-2)]   # (N, K-2)
            steps = self._sp(red(t_steps_raw, dim=0))          # (K-2,)
            edges = u0 + torch.cumsum(steps.to(device=device, dtype=dtype), dim=0)
            all_edges = torch.cat([u0.view(1), edges], dim=0)  # (K-1,)
        else:
            all_edges = u0.view(1)

        # ---- Grille y ----
        if y_max is None:
            ymax = float(all_edges[-1].detach().cpu()) * 2.0
            ymax = max(ymax, float(u0.detach().cpu()) * 1.5 + 1.0, 1.0)
        else:
            ymax = float(y_max)
        y = torch.linspace(0.0, ymax, steps=num_points, device=device, dtype=dtype)

        # ---- PDFs ----
        def pdf_lognormal(yv, mu_log_, sigma_ln_, eps=1e-8):
            yv = yv.clamp_min(eps)
            z  = (torch.log(yv) - mu_log_) / sigma_ln_
            return torch.exp(-0.5 * z * z) / (yv * sigma_ln_ * math.sqrt(2.0 * math.pi))

        def pdf_gamma_k_theta(yv, k_, theta_, eps=1e-12):
            k_ = k_.clamp_min(eps); theta_ = theta_.clamp_min(eps)
            yv = yv.clamp_min(eps)
            log_pdf = (k_ - 1.0) * torch.log(yv) - (yv / theta_) - torch.lgamma(k_) - k_ * torch.log(theta_)
            return torch.exp(log_pdf)

        def pdf_egpd(yv, sigma_t, xi_v, kappa_v, eps=1e-12):
            yv = yv.clamp_min(0.0)
            si = sigma_t.clamp_min(eps)
            kv = kappa_v.clamp_min(eps)
            # autoriser xi négatif si force_positive=False
            xv = xi_v if not self.force_positive else torch.clamp(xi_v, min=self.xi_min)
            z  = torch.clamp(1.0 + xv * (yv / si), min=1.0 + 1e-12)
            H  = (1.0 - torch.pow(z, -1.0 / xv)).clamp(min=eps, max=1.0 - eps)
            h  = (1.0 / si) * torch.pow(z, -1.0 / xv - 1.0)
            return kv * h * torch.pow(H, kv - 1.0)

        if self.T == "egpd":
            f = pdf_egpd(y, sigma, xi, kappa)
        elif self.T == "log":
            f = pdf_lognormal(y, mu_log, sigma)
        else:
            f = pdf_gamma_k_theta(y, k, theta)

        # ---- Plot ----
        y_np = y.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(y_np, f_np, label=title)

        names = [f"u{j}" for j in range(all_edges.numel())]
        for name, uj in zip(names, all_edges):
            xval = float(uj.detach().cpu())
            plt.axvline(x=xval, linestyle='--', alpha=0.55, label=name)
            ymax_txt = (np.nanmax(f_np) * 0.9) if np.isfinite(f_np).any() else 0.9
            plt.text(xval, ymax_txt, f"{name}={xval:.3f}",
                     rotation=90, va='top', ha='right', fontsize=9, alpha=0.8)

        plt.xlim(0, ymax)
        top = (np.nanmax(f_np) * 1.05) if np.isfinite(f_np).any() else 1.0
        plt.ylim(0, top)
        plt.xlabel('y'); plt.ylabel('Density')
        plt.title(title)
        plt.legend(ncol=2)
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.savefig(Path(dir_output) / (output_pdf if output_pdf else "final_pdf.png"), dpi=dpi)
        plt.close()