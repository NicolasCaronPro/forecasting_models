import torch.nn.functional as F
from typing import Optional
from forecasting_models.pytorch.ordinal_loss import *
import math
from pathlib import Path
import numpy as np

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

###################################### Valeurs Extrêmes ##########################################

class BulkTailNLLLoss(nn.Module):
    """
    Mixture Negative Log-Likelihood pour :
      - Bulk : LogNormal(mu_bulk, sigma_bulk)
      - Tail : eGPD (G(y) = H(y)^kappa)
    Tous les paramètres sont appris dans la loss.
    """

    def __init__(self,
                 init_mu_bulk: float = 0.0,
                 init_pi_tail: float = 0.1,
                 kappa: float = 0.831,
                 xi: float = 0.161,
                 eps: float = 1e-8,
                 reduction: str = "mean",
                 force_positive: bool = False):
        super().__init__()

        # Paramètres du bulk (log-normal)
        self.mu_bulk = nn.Parameter(torch.tensor(init_mu_bulk, dtype=torch.float32))

        # Paramètres de la tail (scale GPD)
        self.pi_tail = nn.Parameter(torch.tensor(init_pi_tail, dtype=torch.float32))  # sigmoid pour [0,1]

        # Paramètres eGPD
        self.kappa = nn.Parameter(torch.tensor(kappa, dtype=torch.float32))
        self.xi = nn.Parameter(torch.tensor(xi, dtype=torch.float32))

        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive

    def forward(self, sigma_pos: torch.Tensor, y_pos: torch.Tensor, weight : torch.Tensor = None) -> torch.Tensor:
        # Contrainte des paramètres positifs
        sigma_bulk = sigma_pos + self.eps
        sigma_tail = sigma_pos + self.eps
        pi_tail = torch.sigmoid(self.pi_tail)  # entre 0 et 1

        if self.force_positive:
            kappa = F.softplus(self.kappa)
            xi = F.softplus(self.xi)
        else:
            kappa = self.kappa
            xi = self.xi

        # --- Bulk log-likelihood (LogNormal) ---
        log_y = torch.log(y_pos.clamp_min(self.eps))
        log_pdf_bulk = -torch.log(y_pos * sigma_bulk * (2 * torch.pi) ** 0.5) \
                       - (log_y - self.mu_bulk) ** 2 / (2 * sigma_bulk ** 2)

        # --- Tail log-likelihood (eGPD) ---
        z = 1.0 + xi * (y_pos / sigma_tail)
        z = z.clamp_min(1.0 + 1e-12)
        h = (1.0 / sigma_tail) * torch.pow(z, -1.0 / xi - 1.0)
        log_h = torch.log(h)

        a = 1.0 - torch.pow(z, -1.0 / xi)
        a = a.clamp(max=1.0 - 1e-12)
        log_H = torch.log(a)

        log_g_tail = torch.log(kappa) + log_h + (kappa - 1.0) * log_H

        # --- Mixture log-likelihood ---
        max_log = torch.maximum(log_pdf_bulk, log_g_tail)  # pour stabilité numérique
        log_mix = torch.log(
            (1 - pi_tail) * torch.exp(log_pdf_bulk - max_log) +
            pi_tail * torch.exp(log_g_tail - max_log)
        ) + max_log

        nll = -log_mix

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll

    def get_learnable_parameters(self):
        return {
            "mu_bulk": self.mu_bulk,
            "pi_tail": self.pi_tail,
            "kappa": self.kappa,
            "xi": self.xi
        }
    
    def get_attributes(self):
        """Retourne les valeurs contraintes (positives) des paramètres."""
        return {
            "mu_bulk": self.mu_bulk.item(),
            "pi_tail": torch.sigmoid(self.pi_tail).item(),
            "kappa": F.softplus(self.kappa).item(),
            "xi": F.softplus(self.xi).item()
        }

    def plot_params(self, logs, dir_output):
        """
        Trace l'évolution des paramètres au cours des epochs.
        
        logs : liste de dicts {epoch: int, param: float, ...} 
               ou dict {'epoch': [...], 'param': [...]}
        dir_output : chemin pour sauvegarder les figures
        """
        import os
        os.makedirs(dir_output, exist_ok=True)

        params = ["mu_bulk", "pi_tail", "kappa", "xi"]

        epochs = [log["epoch"] for log in logs]
        for param in params:
            values = [log[param] for log in logs]
            plt.figure(figsize=(8,5))
            plt.plot(epochs, values, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel(param)
            plt.title(f"Evolution de {param} au cours des epochs")
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_output, f"{param}_over_epochs.png"))
            plt.close()

class BulkTailEGPDNLLLossClusterIDs(nn.Module):
    """
    Mixture Bulk + Tail avec eGPD par cluster et option hiérarchique.

    Bulk : log-normal (mu_bulk, sigma_bulk)
    Tail : eGPD (sigma_tail, kappa, xi)

    Paramètres globaux + delta par cluster pour pooling partiel :
      - add_global=False : paramètres par cluster uniquement
      - add_global=True  : global + delta par cluster (partial pooling)
    """
    def __init__(
        self,
        NC: int,
        kappa_init: float = 0.831,
        xi_init: float = 0.161,
        mu_bulk_init: float = 0.0,
        pi_tail_init: float = 0.1,
        eps: float = 1e-8,
        reduction: str = "mean",
        force_positive: bool = False,
        xi_min: float = 1e-12,
        add_global: bool = False,
    ):
        super().__init__()
        self.n_clusters = NC
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive
        self.xi_min = xi_min
        self.add_global = add_global

        if not add_global:
            # Paramètres par cluster
            self.mu_bulk_raw = nn.Parameter(torch.full((NC,), mu_bulk_init))
            self.pi_tail_raw = nn.Parameter(torch.full((NC,), pi_tail_init))
            self.kappa_raw = nn.Parameter(torch.full((NC,), kappa_init))
            self.xi_raw = nn.Parameter(torch.full((NC,), xi_init))
        else:
            # Paramètres globaux + delta par cluster
            self.mu_bulk_global = nn.Parameter(torch.tensor(mu_bulk_init))
            self.mu_bulk_delta = nn.Parameter(torch.zeros(NC))

            self.pi_tail_global = nn.Parameter(torch.tensor(pi_tail_init))
            self.pi_tail_delta = nn.Parameter(torch.zeros(NC))

            self.kappa_global = nn.Parameter(torch.tensor(kappa_init))
            self.kappa_delta = nn.Parameter(torch.zeros(NC))

            self.xi_global = nn.Parameter(torch.tensor(xi_init))
            self.xi_delta = nn.Parameter(torch.zeros(NC))

    def _select_raw_params(self, cluster_ids: torch.Tensor):
        # Bulk
        if not self.add_global:
            mu_bulk = self.mu_bulk_raw[cluster_ids]
            pi_tail = torch.sigmoid(self.pi_tail_raw[cluster_ids])
            kappa_raw = self.kappa_raw[cluster_ids]
            xi_raw = self.xi_raw[cluster_ids]
        else:
            mu_bulk = self.mu_bulk_global + self.mu_bulk_delta[cluster_ids]
            pi_tail = torch.sigmoid(self.pi_tail_global + self.pi_tail_delta[cluster_ids])
            kappa_raw = self.kappa_global + self.kappa_delta[cluster_ids]
            xi_raw = self.xi_global + self.xi_delta[cluster_ids]
        return mu_bulk, pi_tail, kappa_raw, xi_raw

    def forward(self, sigma_pos: torch.Tensor, y_pos: torch.Tensor, cluster_ids: torch.Tensor, weight : torch.Tensor = None) -> torch.Tensor:
        sigma_bulk = sigma_pos + self.eps
        sigma_tail = sigma_pos + self.eps

        mu_bulk, pi_tail, kappa_raw, xi_raw = self._select_raw_params(cluster_ids)

        if self.force_positive:
            kappa = F.softplus(kappa_raw)
            xi = F.softplus(xi_raw) + self.xi_min
        else:
            kappa = kappa_raw
            xi = xi_raw

        # --- Bulk log-likelihood (LogNormal) ---
        log_y = torch.log(y_pos.clamp_min(self.eps))
        log_pdf_bulk = -torch.log(y_pos * sigma_bulk * (2 * torch.pi) ** 0.5) \
                       - (log_y - mu_bulk) ** 2 / (2 * sigma_bulk ** 2)

        # --- Tail log-likelihood (eGPD) ---
        z = (1.0 + xi * (y_pos / sigma_tail)).clamp_min(1.0 + 1e-12)
        log_h = torch.log((1.0 / sigma_tail) * torch.pow(z, -1.0 / xi - 1.0))
        a = (1.0 - torch.pow(z, -1.0 / xi)).clamp(max=1.0 - 1e-12)
        log_H = torch.log(a)
        log_g_tail = torch.log(kappa) + log_h + (kappa - 1.0) * log_H

        # --- Mixture log-likelihood ---
        max_log = torch.maximum(log_pdf_bulk, log_g_tail)
        log_mix = torch.log((1 - pi_tail) * torch.exp(log_pdf_bulk - max_log) +
                            pi_tail * torch.exp(log_g_tail - max_log)) + max_log

        nll = -log_mix

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll

    # ---------- Utilitaires ----------
    def get_learnable_parameters(self):
        if not self.add_global:
            return {
                "mu_bulk": self.mu_bulk_raw,
                "pi_tail": self.pi_tail_raw,
                "kappa": self.kappa_raw,
                "xi": self.xi_raw
            }
        else:
            return {
                "mu_bulk_global": self.mu_bulk_global,
                "mu_bulk_delta": self.mu_bulk_delta,
                "pi_tail_global": self.pi_tail_global,
                "pi_tail_delta": self.pi_tail_delta,
                "kappa_global": self.kappa_global,
                "xi_global": self.xi_global,
                "kappa_delta": self.kappa_delta,
                "xi_delta": self.xi_delta
            }

    def get_attribute(self):
        if not self.add_global:
            mu_bulk = self.mu_bulk_raw.detach()
            pi_tail = torch.sigmoid(self.pi_tail_raw).detach()
            if self.force_positive:
                kappa = F.softplus(self.kappa_raw).detach()
                xi = (F.softplus(self.xi_raw) + self.xi_min).detach()
            else:
                kappa = self.kappa_raw.detach()
                xi = self.xi_raw.detach()
            return [
                ('mu_bulk', mu_bulk),
                ('pi_tail', pi_tail),
                ('kappa', kappa),
                ('xi', xi)
            ]
        else:
            mu_bulk_full = self.mu_bulk_global + self.mu_bulk_delta
            pi_tail_full = torch.sigmoid(self.pi_tail_global + self.pi_tail_delta)
            kappa_full = F.softplus(self.kappa_global + self.kappa_delta) if self.force_positive else self.kappa_global + self.kappa_delta
            xi_full = (F.softplus(self.xi_global + self.xi_delta) + self.xi_min) if self.force_positive else self.xi_global + self.xi_delta

            return [
                ('mu_bulk_global', self.mu_bulk_global.detach()),
                ('mu_bulk_per_cluster', mu_bulk_full.detach()),
                ('pi_tail_global', self.pi_tail_global.detach()),
                ('pi_tail_per_cluster', pi_tail_full.detach()),
                ('kappa_global', F.softplus(self.kappa_global).detach() if self.force_positive else self.kappa_global.detach()),
                ('xi_global', (F.softplus(self.xi_global) + self.xi_min).detach() if self.force_positive else self.xi_global.detach()),
                ('kappa_per_cluster', kappa_full.detach()),
                ('xi_per_cluster', xi_full.detach())
            ]

    # ---------- Plot ----------
    def plot_params(self, logs, dir_output, cluster_names=None, dpi=120):
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)

        epochs = [int(log["epoch"]) for log in logs]
        C = self.n_clusters

        # Convert logs en arrays
        def to_np(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        mu_bulk_list = [log.get("mu_bulk_per_cluster", log.get("mu_bulk")) for log in logs]
        pi_tail_list = [log.get("pi_tail_per_cluster", log.get("pi_tail")) for log in logs]
        kappa_list = [log.get("kappa_per_cluster", log.get("kappa")) for log in logs]
        xi_list = [log.get("xi_per_cluster", log.get("xi")) for log in logs]

        mu_bulk_arr = np.stack([to_np(x) for x in mu_bulk_list], axis=0)
        pi_tail_arr = np.stack([to_np(x) for x in pi_tail_list], axis=0)
        kappa_arr = np.stack([to_np(x) for x in kappa_list], axis=0)
        xi_arr = np.stack([to_np(x) for x in xi_list], axis=0)

        if cluster_names is None:
            cluster_names = [f"cl{c}" for c in range(C)]
        elif len(cluster_names) != C:
            raise ValueError("cluster_names doit correspondre à n_clusters")

        for c in range(C):
            name = cluster_names[c]
            plt.figure(figsize=(8,5))
            plt.plot(epochs, mu_bulk_arr[:, c], marker='o', label=r'$\mu_{bulk}$')
            plt.plot(epochs, pi_tail_arr[:, c], marker='x', label=r'$\pi_{tail}$')
            plt.plot(epochs, kappa_arr[:, c], marker='d', label=r'$\kappa$')
            plt.plot(epochs, xi_arr[:, c], marker='v', label=r'$\xi$')
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title(f"Cluster {name} parameters over epochs")
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(dir_output / f"bulk_tail_params_{name}.png", dpi=dpi)
            plt.close()

        if self.add_global:
            kappa_global_arr = np.asarray([to_np(log["kappa_global"]) for log in logs])
            xi_global_arr = np.asarray([to_np(log["xi_global"]) for log in logs])
            mu_bulk_global_arr = np.asarray([to_np(log["mu_bulk_global"]) for log in logs])
            pi_tail_global_arr = np.asarray([to_np(log["pi_tail_global"]) for log in logs])

            plt.figure(figsize=(8,5))
            plt.plot(epochs, mu_bulk_global_arr, marker='o', label=r'$\mu_{bulk}^{global}$')
            plt.plot(epochs, pi_tail_global_arr, marker='x', label=r'$\pi_{tail}^{global}$')
            plt.plot(epochs, kappa_global_arr, marker='d', label=r'$\kappa^{global}$')
            plt.plot(epochs, xi_global_arr, marker='v', label=r'$\xi^{global}$')
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title("Global parameters over epochs")
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(dir_output / "bulk_tail_params_global.png", dpi=dpi)
            plt.close()

class EGPDNLLLoss(torch.nn.Module):
    """Negative log-likelihood for the eGPD (first family) when ``y > 0``.
    This loss assumes the parametrisation ``G(y) = H(y)^kappa`` where ``H`` is
    the CDF of the Generalised Pareto Distribution.  The parameters ``kappa``
    and ``xi`` are learnt scalars constrained to be positive via ``softplus``.
    """

    def __init__(self, kappa: float = 0.831, xi: float = 0.161, eps: float = 1e-8, reduction: str = "mean", force_positive = True):
        super(EGPDNLLLoss, self).__init__()
        self.kappa = torch.nn.Parameter(torch.tensor(kappa))
        self.xi = torch.nn.Parameter(torch.tensor(xi))
        
        #self.kappa = torch.nn.parameter.Parameter(0,831, requires_grad=False)
        #self.xi = torch.nn.parameter.Parameter(0,161, requires_grad=False)
        # Register positive scalar buffers (moved automatically across devices)
        # Defaults set near prior values; pass via ctor to override.
        #self.register_buffer('xi', torch.tensor(float(xi), dtype=torch.float32))
        #self.register_buffer('kappa', torch.tensor(float(kappa), dtype=torch.float32))
        self.eps = eps
        self.reduction = reduction
        self.force_positive = force_positive

    def forward(self, sigma_pos: torch.Tensor, y_pos: torch.Tensor, weight : torch.Tensor = None) -> torch.Tensor:
        """Compute the eGPD negative log-likelihood.

        Parameters
        ----------
        y_pos : torch.Tensor
            Observations strictly greater than zero.
        sigma_pos : torch.Tensor
            Positive scale parameter predicted by the network.
        Returns
        -------
        torch.Tensor
            The reduced negative log-likelihood according to ``reduction``.
        """
        if self.force_positive:
            kappa = F.softplus(self.kappa)
            xi = F.softplus(self.xi)
        else:
            kappa = self.kappa
            xi = self.xi

        sigma = sigma_pos.clamp_min(self.eps)

        z = 1.0 + xi * (y_pos / sigma)
        z = z.clamp_min(1.0 + 1e-12)

        h = (1 / sigma) * torch.pow(z, -1.0 / xi - 1.0)
        log_h = torch.log(h)

        a = 1.0 - torch.pow(z, -1.0/xi)
        a = a.clamp(max=1.0 - 1e-12)
        
        log_H = torch.log(a)

        log_g = torch.log(kappa) + log_h + (kappa - 1.0) * log_H
        
        nll = -log_g
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll
    
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

class EGPDNLLLossClusterIDs(nn.Module):
    """
    eGPD NLL (first family, y>0) avec paramètres par cluster, et optionnellement
    un terme GLOBAL (partial pooling) si `add_global=True`.

    Paramétrisation: G(y) = H(y)^kappa, H = CDF de la GPD.

    - add_global=False  -> uniquement des paramètres par cluster:
        * kappa_raw: (C,), xi_raw: (C,)

    - add_global=True   -> hiérarchique (global + delta par cluster):
        * kappa_global, xi_global : scalaires
        * kappa_delta,  xi_delta  : (C,) déviations par cluster (init 0)
        * kappa_raw[cluster] = kappa_global + kappa_delta[cluster]
          xi_raw[cluster]    = xi_global    + xi_delta[cluster]

    Le `forward(...)` prend toujours `cluster_ids: (N,)` (un cluster par sample).
    """

    def __init__(
        self,
        NC: int,
        id : int,
        kappa_init: float = 0.831,
        xi_init: float = 0.161,
        eps: float = 1e-8,
        reduction: str = "mean",
        force_positive: bool = False,
        xi_min: float = 1e-12,
        G: bool = False,
        num_classes: int=5,
    ):
        super().__init__()
        self.n_clusters     = NC
        self.eps            = eps
        self.reduction      = reduction
        self.force_positive = force_positive
        self.xi_min         = xi_min
        self.add_global     = G
        self.id = id
        self.num_classes = num_classes

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

    def _select_raw_params(self, cluster_ids: torch.Tensor):
        """
        Retourne (kappa_raw_sel, xi_raw_sel) de taille (N,)
        selon le mode (clusters only) ou (global + delta).
        """
        if not self.add_global:
            kappa_raw_sel = self.kappa_raw[cluster_ids]                 # (N,)
            xi_raw_sel    = self.xi_raw[cluster_ids]                    # (N,)
        else:
            kappa_raw_sel = self.kappa_global + self.kappa_delta[cluster_ids]  # (N,)
            xi_raw_sel    = self.xi_global    + self.xi_delta[cluster_ids]     # (N,)
        return kappa_raw_sel, xi_raw_sel
    
    def forward(
        self,
        sigma_pos: torch.Tensor,     # (N,) échelle prédite (peut être brute; voir note)
        y_pos: torch.Tensor,         # (N,) y>0
        cluster_ids: torch.Tensor,   # (N,) long
        weight: torch.Tensor = None  # (N,) optionnel
    ) -> torch.Tensor:

        # Sélectionne kappa, xi "bruts" par sample
        kappa_raw_sel, xi_raw_sel = self._select_raw_params(cluster_ids.long())

        # Contraintes de positivité (appliquées APRÈS la combinaison global+delta le cas échéant)
        if self.force_positive:
            kappa = F.softplus(kappa_raw_sel)                # (N,)
            xi    = F.softplus(xi_raw_sel) + self.xi_min     # (N,)
        else:
            kappa = kappa_raw_sel
            xi    = xi_raw_sel

        # Échelle (assainissement numérique)
        sigma = sigma_pos.clamp_min(self.eps)
        # Si votre réseau ne garantit pas sigma>0, préférez:
        # sigma = F.softplus(sigma_pos).clamp_min(self.eps)

        # eGPD (famille 1), y>0
        z = (1.0 + xi * (y_pos / sigma)).clamp_min(1.0 + 1e-12)
        log_h = torch.log((1.0 / sigma) * torch.pow(z, -1.0 / xi - 1.0))
        a = (1.0 - torch.pow(z, -1.0 / xi)).clamp(max=1.0 - 1e-12)
        log_H = torch.log(a)

        log_g = torch.log(kappa) + log_h + (kappa - 1.0) * log_H
        nll = -log_g  # (N,)

        if weight is not None:
            nll = nll * weight

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll

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
            # Reconstruire les valeurs par cluster pour inspection
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
            device = self.kappa_raw.device
            return torch.zeros((), device=device)
        return lambda_l2 * (self.kappa_delta.pow(2).sum() + self.xi_delta.pow(2).sum())

    # ---------- Plot: un fichier PNG par cluster ----------
    def plot_params(self, egpd_logs, dir_output, cluster_names=None, dpi=120):
        """
        Sauvegarde et trace les paramètres EGPD au fil des epochs.
        - add_global=False : un fichier PNG par cluster (kappa, xi).
        - add_global=True  : idem + un fichier 'global' montrant kappa_global et xi_global.
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

            import matplotlib.pyplot as plt
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

        # --- Cas global ---
        if self.add_global:
            kappa_global = np.asarray([to_np_vec(k) for k in kappa_global])
            xi_global    = np.asarray([to_np_vec(x) for x in xi_global])

            import matplotlib.pyplot as plt
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

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# On suppose disponible:
# cdf_egpd_family1(u, sigma_tail, xi, kappa, eps=1e-12) -> Tensor in [0,1]

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
      - forward(scale, y_class, cluster_ids, weight=None)
        * scale: paramètre d'échelle par échantillon
            T='egpd'  -> sigma_tail_i
            T='log'   -> sigma_ln_i (écart-type en log)
            T='gamma' -> theta_i (échelle)
        * y_class ∈ {0..K-1}
        * cluster_ids ∈ {0..C-1}

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

    def _edges_for_samples(self, cluster_ids: torch.Tensor, device, dtype):
        """Edges du cluster de chaque sample: shape (N, K-1)"""
        all_edges = self._edges_all_clusters(device, dtype)           # (C, K-1)
        return all_edges[cluster_ids.long()]                          # (N, K-1)

    # --- sélection des params par cluster selon T ---
    def _select_primary_params(self, cluster_ids: torch.Tensor):
        cid = cluster_ids.long()
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
    def _cdf_primary(self, u: torch.Tensor, scale: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        tag, a, b = self._select_primary_params(cluster_ids)  # a,b dépendent de T
        cluster_ids = cluster_ids.long()
        if tag == "egpd":
            res = torch.zeros_like(u)
            for cluster in torch.unique(cluster_ids):
                mask = cluster_ids == cluster
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
                cluster_ids: torch.Tensor,  # [N] in {0..C-1}
                weight: torch.Tensor = None) -> torch.Tensor:

        scale = scale.squeeze(-1) if scale.ndim > 1 else scale
        device, dtype = scale.device, scale.dtype
        K = self.num_classes

        # Edges par sample: (N, K-1) ; u0 par sample: (N,)
        edges_N = self._edges_for_samples(cluster_ids, device=device, dtype=dtype)  # (N, K-1)
        u0_vecC = self._u0(device=device, dtype=dtype)                               # (C,)
        u0_N    = u0_vecC[cluster_ids.long()]                                        # (N,)

        # CDF aux bords (selon T)
        F_list = [ self._cdf_primary(u0_N, scale, cluster_ids) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges_N[:, j], scale, cluster_ids))

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
    def transform(self, inputs: torch.Tensor, cluster_ids: torch.Tensor,
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
        self.plot_final_pdf(inputs, cluster_ids,
                            dir_output=dir_output,
                            output_pdf=(output_pdf or "final_pdf"))

        edges_N = self._edges_for_samples(cluster_ids, device=device, dtype=dtype)  # (N, K-1)
        u0_vecC = self._u0(device=device, dtype=dtype)
        u0_N    = u0_vecC[cluster_ids.long()]

        F_list = [ self._cdf_primary(u0_N, inputs, cluster_ids) ]
        for j in range(K-2):
            F_list.append(self._cdf_primary(edges_N[:, j], inputs, cluster_ids))

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
                   cluster_ids: torch.Tensor,    # (N,)
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

        cid = cluster_ids.long()

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

    def _edges_for_samples(self, cluster_ids, device, dtype):
        return self._edges_all(device, dtype)[cluster_ids.long()]       # (N,K-2)

    # --------------- sélecteurs params cluster ---------------
    def _select_bulk_param(self, cluster_ids: torch.Tensor):
        cid = cluster_ids.long()
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

    def _select_tail_params(self, cluster_ids: torch.Tensor):
        """Retourne un dict selon T pour simplifier l'appel en aval."""
        cid = cluster_ids.long()
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
    def transform(self, inputs: torch.Tensor, cluster_ids: torch.Tensor,
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
            self.plot_final_pdf(scale_tail, scale_bulk, pi_logit, cluster_ids,
                                dir_output=dir_output,
                                output_pdf=(output_pdf or "final_pdf"))

        device, dtype = scale_tail.device, scale_tail.dtype
        K = self.K

        bulk_param = self._select_bulk_param(cluster_ids)     # mu_c ou k_shape_c
        tail_params = self._select_tail_params(cluster_ids)   # dict selon T

        u0_all  = self._u0_all(device, dtype)
        u0_N    = u0_all[cluster_ids.long()]
        edges_N = self._edges_for_samples(cluster_ids, device, dtype)

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
    def forward(self, inputs: torch.Tensor, y_class: torch.Tensor, cluster_ids: torch.Tensor, weight: torch.Tensor = None):
        """
        inputs: (N,3) = [scale_tail, scale_bulk, pi_tail_logit]
        """
        P = self.transform(inputs, cluster_ids, dir_output=None, output_pdf=None)  # (N,K)

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
                   cluster_ids: torch.Tensor,     # (N,)
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

        cid = cluster_ids.long()

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