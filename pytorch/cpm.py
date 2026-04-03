import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneAlpha(nn.Module):
    """
    Transformation monotone alpha(y).

    alpha(y) = a_pos * y + sum_j w_pos[j] * softplus((y - knot_j)/temp) * temp

    - a_pos > 0  => pente de base strictement positive
    - w_pos[j] >= 0 => contribution monotone positive
    - donc alpha'(y) > 0 partout

    Remarque:
    - on fixe l'intercept implicite de alpha pour limiter les problèmes
      d'identifiabilité avec eta(x).
    """

    def __init__(self, y_min: float, y_max: float, n_knots: int = 15, temp: float = 0.25):
        super().__init__()
        if y_max <= y_min:
            raise ValueError("y_max must be > y_min")
        if n_knots < 1:
            raise ValueError("n_knots must be >= 1")

        self.temp = float(temp)

        knots = torch.linspace(float(y_min), float(y_max), int(n_knots))
        self.register_buffer("knots", knots)

        # paramètres bruts -> rendus positifs par softplus
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.raw_w = nn.Parameter(torch.zeros(n_knots))

    def positive_slope(self):
        # strictement positif
        return F.softplus(self.raw_slope) + 1e-4

    def positive_w(self):
        return F.softplus(self.raw_w)

    def basis(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B]
        return: [B, K]
        """
        z = (y.unsqueeze(-1) - self.knots.unsqueeze(0)) / self.temp
        return F.softplus(z) * self.temp

    def dbasis(self, y: torch.Tensor) -> torch.Tensor:
        """
        dérivée de basis par rapport à y
        d/dy softplus((y-k)/temp)*temp = sigmoid((y-k)/temp)
        """
        z = (y.unsqueeze(-1) - self.knots.unsqueeze(0)) / self.temp
        return torch.sigmoid(z)

    def forward(self, y: torch.Tensor):
        """
        y: [B]
        returns:
            alpha_y: [B]
            dalpha_y: [B]
        """
        if y.dim() != 1:
            raise ValueError("y must be 1D of shape [B]")

        a = self.positive_slope()           # scalaire > 0
        w = self.positive_w()               # [K] >= 0

        B = self.basis(y)                   # [B, K]
        dB = self.dbasis(y)                 # [B, K]
        
        alpha_y = a * y + B @ w            # [B]
        dalpha_y = a + dB @ w              # [B], > 0

        return alpha_y, dalpha_y


class MLPBackbone(nn.Module):
    """
    eta_phi(x): score latent flexible appris par réseau.
    """
    def __init__(self, in_dim: int, hidden_dims=(128, 64), dropout: float = 0.1):
        super().__init__()

        dims = [in_dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(d_in, d_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x must be 2D of shape [B, F]")
        h = self.net(x)
        return self.out(h).squeeze(-1)  # [B]


class ContinuousCPM(nn.Module):
    """
    Continuous CPM / transformation model:
        P(Y <= y | x) = F(alpha(y) - eta(x))

    Ici F = logistique standard.
    """

    def __init__(
        self,
        in_dim: int,
        y_min: float,
        y_max: float,
        hidden_dims=(128, 64),
        n_knots: int = 15,
        temp: float = 0.25,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eta_net = MLPBackbone(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.alpha = MonotoneAlpha(y_min=y_min, y_max=y_max, n_knots=n_knots, temp=temp)
        self.eps = float(eps)

    @staticmethod
    def logistic_log_pdf(s: torch.Tensor) -> torch.Tensor:
        """
        log f(s), f densité logistique standard
        f(s) = sigmoid(s) * sigmoid(-s)

        stable numériquement:
            log f(s) = logsigmoid(s) + logsigmoid(-s)
        """
        return F.logsigmoid(s) + F.logsigmoid(-s)

    @staticmethod
    def logistic_cdf(s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(s)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        returns dict with:
            eta, alpha_y, dalpha_y, s, cdf
        """
        if y.dim() != 1:
            raise ValueError("y must be 1D of shape [B]")
        if x.size(0) != y.size(0):
            raise ValueError("x and y must have same batch size")

        eta = self.eta_net(x)                    # [B]
        alpha_y, dalpha_y = self.alpha(y)        # [B], [B]
        s = alpha_y - eta                        # [B]
        cdf = self.logistic_cdf(s)               # [B]

        return {
            "eta": eta,
            "alpha_y": alpha_y,
            "dalpha_y": dalpha_y,
            "s": s,
            "cdf": cdf,
        }

    def nll(self, x: torch.Tensor, y: torch.Tensor, sample_weight: torch.Tensor = None):
        """
        Negative log-likelihood continue exacte:
            - [ log f(alpha(y)-eta(x)) + log alpha'(y) ]
        """
        out = self.forward(x, y)

        log_pdf = self.logistic_log_pdf(out["s"])                    # [B]
        log_jac = torch.log(out["dalpha_y"] + self.eps)             # [B]

        nll_i = -(log_pdf + log_jac)                                 # [B]

        if sample_weight is not None:
            if sample_weight.dim() != 1 or sample_weight.size(0) != nll_i.size(0):
                raise ValueError("sample_weight must be 1D with same batch size")
            w = sample_weight / (sample_weight.sum() + self.eps)
            return (w * nll_i).sum()

        return nll_i.mean()

    @torch.no_grad()
    def predict_eta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score latent eta(x).
        """
        return self.eta_net(x)

    @torch.no_grad()
    def cdf(self, x: torch.Tensor, y_query: torch.Tensor) -> torch.Tensor:
        """
        Évalue P(Y <= y_query | x).

        x: [B, F]
        y_query:
            - [M] -> retourne [B, M]
            - [B] -> retourne [B]
        """
        eta = self.eta_net(x)  # [B]

        if y_query.dim() == 1 and y_query.size(0) == x.size(0):
            alpha_y, _ = self.alpha(y_query)
            return self.logistic_cdf(alpha_y - eta)

        if y_query.dim() == 1:
            B = x.size(0)
            M = y_query.size(0)
            y_rep = y_query.unsqueeze(0).expand(B, M).reshape(-1)
            alpha_y, _ = self.alpha(y_rep)
            alpha_y = alpha_y.view(B, M)
            return self.logistic_cdf(alpha_y - eta.unsqueeze(1))

        raise ValueError("y_query must be 1D")

    @torch.no_grad()
    def predict_median(self, x: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        """
        Approxime la médiane conditionnelle via recherche sur une grille:
            median(x) = inf{ y : CDF(y|x) >= 0.5 }

        y_grid: [M] croissante
        return: [B]
        """
        cdf_vals = self.cdf(x, y_grid)  # [B, M]
        idx = torch.argmax((cdf_vals >= 0.5).to(torch.int64), dim=1)
        return y_grid[idx]