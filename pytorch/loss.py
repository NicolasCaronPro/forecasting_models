import torch
import torch.nn.functional as F
from typing import Optional

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

class BCELoss(torch.nn.Module):
    """Binomial Cross Entropy loss for ordinal classification.

    In this repository the models return a probability for each class, e.g.
    a tensor of shape ``(N, num_classes)`` when ``num_classes`` is the number of
    ordinal categories.  In order to compute the BCE in an "all-threshold"
    fashion, these class probabilities are converted into ``num_classes - 1``
    binary classification targets representing ``P(y > k)`` for each threshold
    ``k``.
    """

    def __init__(self, num_classes: int):
        super(BCELoss, self).__init__()
        self.num_classes = num_classes

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the BCE loss for ordinal targets.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor of shape ``(N, num_classes)`` containing the predicted
            probabilities for each class. These probabilities will be
            converted internally into ``P(y > k)`` for each threshold ``k``.
        y_true : torch.Tensor
            Tensor of shape ``(N,)`` with integer labels in ``[0, num_classes-1]``.
        sample_weights : Optional[torch.Tensor]
            Optional tensor of shape ``(N,)`` with per-sample weights.

        Returns
        -------
        torch.Tensor
            The averaged binomial cross entropy loss.
        """

        y_pred = F.softmax(y_pred, dim=1)

        # Ensure target is one-dimensional and integer
        y_true = y_true.long().view(-1)

        # Convert class probabilities into P(y > k) for each threshold k
        cumulative = torch.cumsum(y_pred, dim=1)
        y_pred_bin = 1 - cumulative[:, :-1]

        # Create binary targets for each threshold
        thresholds = torch.arange(self.num_classes - 1, device=y_true.device)
        y_true_bin = (y_true.unsqueeze(1) > thresholds).float()

        y_pred_bin = torch.abs(y_pred_bin)

        # BCE for each threshold
        loss_per_thresh = F.binary_cross_entropy(y_pred_bin, y_true_bin, reduction="none")

        # Average loss over thresholds for each sample
        loss = loss_per_thresh.mean(dim=1)

        if sample_weights is not None:
            weighted_loss = loss * sample_weights
            return torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            return torch.mean(loss)

class ForegroundDiceLoss(nn.Module):
    """
    Multiclass Dice sur les classes > 0 (foreground seulement).
    - pénalise X->0 (FN) et 0->X (FP)
    - n'inclut pas 0->0 (TN) car on retire la classe 0 du calcul
    - supporte ignore_index pour exclure des pixels (ex: 255)
    Optionnel: class_weights (longueur C-1) pour pondérer les classes foreground.
    """
    def __init__(self, smooth=1e-5, ignore_index=None, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("w", w / (w.sum() + 1e-12))
        else:
            self.w = None

    def forward(self, probs, target):
        """
        logits: (N, C, H, W)  -- logits multiclasse
        target: (N, H, W)     -- entiers dans [0..C-1], 0 = background
        """
        N, C = probs.shape
        assert C >= 2, "Besoin d'au moins 1 classe foreground (C>=2)."

        probs = probs.view(N, C, -1)           # (N,C,*)
        target  = target.view(N, -1)               # (N,*)

        # Masque des pixels valides (on n'ignore pas la classe 0, seulement ignore_index)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)       # (N,H,W) bool
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        # One-hot du target (on met une valeur sûre pour les pixels ignorés, puis on masquera)
        target_safe = target.clone()
        target_safe[~valid] = 0                         # peu importe: sera masqué
        t_oh = torch.zeros_like(probs).scatter_(1, target_safe.unsqueeze(1), 1.0)  # (N,C,H,W)

        # On retire la classe background (canal 0) -> ne compte pas 0->0
        p_fg = probs[:, 1:]                       # (N,C-1,H,W)
        t_fg = t_oh[:, 1:]                        # (N,C-1,H,W)

        # Appliquer le masque de validité aux deux (exclut totalement les pixels ignore_index)
        valid_ = valid.unsqueeze(1).to(p_fg.dtype)      # (N,1,H,W)
        p = p_fg * valid_
        t = t_fg * valid_

        intersection = (p * t).sum(dim=2)               # (N,C-1)
        denom = p.pow(2).sum(dim=2) + t.sum(dim=2)      # (N,C-1)
        dice_c = (2 * intersection + self.smooth) / (denom + self.smooth)  # (N,C-1)

        # Moyenne sur classes foreground (pondérée éventuelle), puis sur batch
        if self.w is not None:
            # self.w: (C-1,)
            dice_per_n = (dice_c * self.w.view(1, -1)).sum(dim=1)  # (N,)
        else:
            dice_per_n = dice_c.mean(dim=1)                         # (N,)

        loss = 1.0 - dice_per_n.mean()
        return loss

class WeightedCrossEntropyLossLearnable(torch.nn.Module):
    def __init__(self, num_classes=5, min_value=0.5):
        super(WeightedCrossEntropyLossLearnable, self).__init__()
        self.num_classes = num_classes

        self.ratio_matrix = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1],
            [1, 1, 3, 1, 1],
            [1, 1, 1, 4, 1],
            [1, 1, 1, 1, 5]
        ], dtype=torch.float32)

        self.min_value = min_value
        initial_adjustment_rate = torch.ones((num_classes, num_classes), dtype=torch.float32) * 0.01
        self.adjustment_rate = torch.nn.Parameter(initial_adjustment_rate)

    def forward(self, y_pred, y_true, update_matrix=False, sample_weights=None):
        y_true = y_true.long()

        # Prédiction des classes
        pred_class = torch.argmax(y_pred, dim=1)

        if update_matrix:
            self.ratio_matrix = copy.deepcopy(self.adjustment_rate)
            self.ratio_matrix.data.clamp_(min=self.min_value)

        # Calcul de la perte de base
        loss = F.cross_entropy(y_pred, y_true, reduction='none')

        # Ajuster dynamiquement les poids de la ratio_matrix
        if update_matrix:
            for ir in range(y_true.shape[0]):
                true_class = y_true[ir].item()
                predicted_class = pred_class[ir].item()

                #if true_class == predicted_class:
                    # Bonne prédiction : diminuer le poids légèrement
                #    self.ratio_matrix[true_class, predicted_class].data -= self.adjustment_rate[true_class, predicted_class].item()
                #else:
                    # Mauvaise prédiction : augmenter le poids proportionnellement à l'erreur

                # Appliquer la contrainte sur la valeur minimale
                if update_matrix:
                    #if true_class != predicted_class:
                    #    self.ratio_matrix[true_class, predicted_class].data += self.adjustment_rate[true_class, predicted_class].item() 
                    
                    loss[ir] *= self.ratio_matrix[true_class, predicted_class]

        # Appliquer les sample weights si fournis
        if sample_weights is not None:
            weighted_loss = loss * sample_weights
            return torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            return weighted_loss.sum() / y_pred.size(0)

    def get_learnable_parameters(self):
        """Expose the learnable parameters to the external optimizer."""
        return {'adjustment_rate': self.adjustment_rate}

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, y_pred, y_true, update_matrix=False, sample_weights=None):
        y_true = y_true.long()

        #pred_class = torch.argmax(y_pred, dim=1)
        
        """ratio_matrix = [[1,     1,         2,       3,       4],
                        [1,     2,         0.5,     0.75,    1],
                        [2,     0.5,        3,      0.33,   0.5],
                        [3,     0.75,       0.33,      4,   0.33],
                        [4,     1,          0.5,     0.33,     5]                    
                        ]"""

        loss = F.cross_entropy(y_pred, y_true, reduction='none')
        
        #for ir in range(y_true.shape[0]):
        #loss[ir] *= ratio_matrix[y_true[ir]][pred_class[ir]]
        
        if sample_weights is not None:
            weighted_loss = loss * sample_weights
            return torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            return loss.sum() / y_pred.size(0)

class LossPerId(torch.nn.Module):
    def __init__(self, criterion, id, num_classes=5):
        super(LossPerId, self).__init__()
        self.num_classes = num_classes
        self.id = id  # index used to group by department

        self.criterion = criterion

    def forward(self, y_pred, y_true, id_mask, update_matrix=False, sample_weights=None):
        id_mask = id_mask.reshape(y_true.shape[0])
        y_true = y_true.long()
        group_ids = id_mask.unique()
        total_loss = 0.0
        total_samples = 0

        res = []

        for group in group_ids:
            group_mask = (id_mask == group)
            y_pred_group = y_pred[group_mask]
            y_true_group = y_true[group_mask]

            # Optionally apply sample weights per group
            if sample_weights is not None:
                weights_group = sample_weights[group_mask]
                loss = self.criterion(y_pred_group, y_true_group)
                weighted_loss = loss * weights_group
                group_loss = torch.sum(weighted_loss) / torch.sum(weights_group)
            else:
                loss = self.criterion(y_pred_group, y_true_group)
                group_loss = loss.sum() / y_pred_group.size(0)

            res.append(group_loss)

        return torch.tensor(res)  # or total_loss / total_samples if you want average

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

class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes=5, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def dice_coefficient(self, y_true, y_pred, smooth=1e-10):
        """
        Calcul du Dice Coefficient en PyTorch.

        Args:
            y_pred (torch.Tensor): Prédictions de forme (N, C, ...), probabilités.
            y_true (torch.Tensor): Cibles vraies de forme (N, ...), labels.
            smooth (float): Paramètre pour éviter la division par zéro.

        Returns:
            torch.Tensor: La valeur du Dice Coefficient pour chaque échantillon.
        """
        N = y_true.shape[0]

        # Appliquer softmax pour obtenir les probabilités
        prob = F.softmax(y_pred, dim=1)  # (N, C, ...)
        pred_classes = torch.argmax(prob, dim=1)  # (N, ...)

        # Redimensionner les prédictions et cibles
        pred_classes = pred_classes.view(N, -1)  # (N, *)
        y_true = y_true.view(N, -1)  # (N, *)

        # Calcul de l'intersection et de l'union
        intersection = torch.sum(pred_classes * y_true, dim=1).float()  # (N,)
        union = torch.sum(pred_classes**2, dim=1) + torch.sum(y_true**2, dim=1)  # (N,)

        # Calcul du Dice Coefficient
        dice_coef = ((2 * intersection) + smooth) / (union + smooth)  # (N,)

        return dice_coef

    def dice_loss(self, y_pred, y_true, smooth=1e-6):
        """
        Calcul de la Dice Loss pour chaque classe.

        Args:
            y_pred (torch.Tensor): Prédictions de forme (N, C, ...), probabilités.
            y_true (torch.Tensor): Cibles vraies de forme (N, ...), labels.
            smooth (float): Paramètre pour éviter la division par zéro.

        Returns:
            torch.Tensor: La valeur de la Dice Loss.
        """
        dice_coef = self.dice_coefficient(y_true, y_pred, smooth)  # (N,)
        dice_loss = 1 - torch.mean(dice_coef)  # Moyenne sur tous les échantillons
        return dice_loss

    def forward(self, y_pred, y_true):
        """
        Calcul de la Dice Loss.

        :param y_pred: Tenseur contenant les prédictions du modèle (probabilités ou valeurs continues).
        :param y_true: Tenseur contenant les vérités terrain (cibles).
        
        :return: La Dice Loss.
        """
        return self.dice_loss(y_true, y_pred)
    
class DiceLoss2(torch.nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None, ignore_index=None):
        super(DiceLoss2, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

        self.ignore_index = ignore_index

    def forward(self, predict, target):
        N, C = predict.size()[:2]

        # Si 'predict' sont des logits, décommente:
        predict = torch.softmax(predict, dim=1)

        # Aplatir
        predict = predict.view(N, C, -1)           # (N,C,*)
        target  = target.view(N, -1)               # (N,*)

        # Masque des pixels valides
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)  # (N,*)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        # One-hot du target (remplacer les ignorés par une valeur factice, puis on masque)
        target_safe = target.masked_fill(~valid, 0)         # (N,*)
        target_onehot = torch.zeros(N, C, target.size(1), device=predict.device, dtype=predict.dtype)
        target_onehot.scatter_(1, target_safe.unsqueeze(1), 1)  # (N,C,*)

        # Appliquer le masque aux deux (exclut totalement les pixels ignorés)
        valid = valid.unsqueeze(1).to(predict.dtype)        # (N,1,*)
        predict      = predict * valid
        target_onehot = target_onehot * valid

        # Dice
        intersection = (predict * target_onehot).sum(dim=2)                  # (N,C)
        union        = predict.pow(2).sum(dim=2) + target_onehot.sum(dim=2)  # (N,C)
        dice_coef    = (2 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1 - dice_coef.mean()   # moyenne sur N et C
        return dice_loss

class CDWCELoss(torch.nn.Module):
    """Class Distance Weighted Cross-Entropy Loss proposed in :footcite:t:`polat2022class`.
    It respects the order of the classes and takes the distance of the classes into
    account in calculation of the cost.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    alpha : float, default=0.5
        The exponent of the distance between target and predicted class.
    weight : Tensor, optional, default=None
        Weight applied to each class when computing the loss. It is based on the target
        class. Can be used to mitigate class imbalance.
    """

    def __init__(self, num_classes, alpha=0.5, weight=None):
        super(CDWCELoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.weight_ = weight
        self.normalised_weight_ = None
        if self.weight_ is not None:
            self.normalised_weight_ = self.weight_ / self.weight_.sum()

    def forward(self, y_pred, y_true):
        if y_true.dim() > 1:
            y_true_indices = y_true.argmax(dim=1, keepdim=True)
        else:
            y_true_indices = y_true.view(-1, 1)

        N = y_true_indices.size(0)
        J = self.num_classes

        s = torch.exp(y_pred).sum(dim=1, keepdim=True)
        l1 = torch.log(s - torch.exp(y_pred) + 1e-8)
        l2 = torch.log(s + 1e-8)
        l_1_2 = l1 - l2

        i_indices = torch.arange(J).view(1, -1).expand(N, J).to(y_true_indices.device)
        weights = (torch.abs(i_indices - y_true_indices) ** self.alpha).float()

        loss = l_1_2 * weights

        if self.weight_ is not None and self.normalised_weight_ is not None:
            if self.normalised_weight_.device != loss.device:
                self.normalised_weight_ = self.normalised_weight_.to(loss.device)

            tiled_class_weight = self.normalised_weight_.view(1, -1).expand(N, J)
            sample_weights = torch.gather(
                tiled_class_weight, dim=1, index=y_true_indices
            )
            loss = loss * sample_weights

        loss = loss.sum()

        return -loss / N

class MCEAndWKLoss(torch.nn.modules.loss._WeightedLoss):
    """
    The loss function integrates both MCELoss and WKLoss, concurrently minimising
    error distances while preventing the omission of classes from predictions.

    Parameters
    ----------
    num_classes : int
        Number of classes
    C: float, defaul=0.5
        Weights the WK loss (C) and the MCE loss (1-C). Must be between 0 and 1.
    wk_penalization_type : str, default='quadratic'
        The penalization type of WK loss to use (quadratic or linear).
        See WKLoss for more details.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `J`, where `J` is the number of classes.
        Otherwise, it is treated as if having all ones.
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed.
    use_logits : bool, default=False
        If True, the input y_pred will be treated as logits.
        If False, the input y_pred will be treated as probabilities.
    """

    def __init__(
        self,
        num_classes: int,
        C: float = 0.5,
        wk_penalization_type: str = "quadratic",
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=True,
        learned=False
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
        self.learned = learned
        if learned:
            self.C = torch.nn.Parameter(torch.tensor(C))
        else:
            self.C = torch.nn.Parameter(torch.tensor(C, requires_grad=False))

        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )
        
        if C < 0 or C > 1:
            raise ValueError(f"C must be between 0 and 1, but is {C}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.mce = MCELoss(self.num_classes, weight=weight, use_logits=self.use_logits)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            The weighted sum of MCE and QWK loss
        """

        if self.learned:
            C = torch.nn.functional.sigmoid(self.C)
        else:
            C = self.C

        wk_result = self.wk(y_pred, y_true)
        mce_result = self.mce(y_pred, y_true)

        return C * wk_result + (1 - C) * mce_result
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}
        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output):

        Cs = [mcewk_log["C"] for mcewk_log in mcewk_logs]
        epochs = [mcewk_log["epoch"] for mcewk_log in mcewk_logs]

        # Sauvegarde pickle
        egpd_to_save = {"epoch": epochs, "C": Cs}
        save_object(egpd_to_save, 'Cs.pkl', dir_output)

        # kappa vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, Cs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cs')
        plt.title('MCEWK Cs over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'mcewk_Cs_epochs.png')
        plt.close()

class DiceAndWKLoss(torch.nn.modules.loss._WeightedLoss):

    def __init__(
        self,
        num_classes: int,
        C: float = 0.5,
        wk_penalization_type: str = "quadratic",
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=True,
        learned=False
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
        self.learned = learned
        if learned:
            self.C = torch.nn.Parameter(torch.tensor(C))
        else:
            self.C = torch.nn.Parameter(torch.tensor(C), requires_grad=False)

        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if C < 0 or C > 1:
            raise ValueError(f"C must be between 0 and 1, but is {C}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.dice = DiceLoss2()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            The weighted sum of MCE and QWK loss
        """

        if self.learned:
            C = torch.nn.functional.sigmoid(self.C)
        else:
            C = self.C

        wk_result = self.wk(y_pred, y_true)
        dice_result = self.dice(y_pred, y_true)

        return C * wk_result + (1 - C) * dice_result
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}
        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output):

        Cs = [mcewk_log["C"] for mcewk_log in mcewk_logs]
        epochs = [mcewk_log["epoch"] for mcewk_log in mcewk_logs]

        # Sauvegarde pickle
        egpd_to_save = {"epoch": epochs, "C": Cs}
        save_object(egpd_to_save, 'Cs.pkl', dir_output)

        # kappa vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, Cs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cs')
        plt.title('DWK Cs over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'dwk_Cs_epochs.png')
        plt.close()

class ForegroundDiceLossAndWKLoss(torch.nn.modules.loss._WeightedLoss):

    def __init__(
        self,
        num_classes: int,
        C: float = 0.5,
        wk_penalization_type: str = "quadratic",
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=True,
        learned=True
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
        self.learned = learned
        if learned:
            self.C = torch.nn.Parameter(torch.tensor(C))
        else:
            self.C = C

        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if C < 0 or C > 1:
            raise ValueError(f"C must be between 0 and 1, but is {C}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.dice = ForegroundDiceLoss()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            The weighted sum of MCE and QWK loss
        """

        if self.learned:
            C = torch.nn.functional.sigmoid(self.C)
        else:
            C = self.C

        wk_result = self.wk(y_pred, y_true)
        dice_result = self.dice(y_pred, y_true)

        return C * wk_result + (1 - C) * dice_result
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}

        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output):

        Cs = [mcewk_log["C"] for mcewk_log in mcewk_logs]
        epochs = [mcewk_log["epoch"] for mcewk_log in mcewk_logs]

        # Sauvegarde pickle
        egpd_to_save = {"epoch": epochs, "C": Cs}
        save_object(egpd_to_save, 'Cs.pkl', dir_output)

        # kappa vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, Cs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cs')
        plt.title('DWK Cs over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'dwk_Cs_epochs.png')
        plt.close()

class OrdinalDiceLoss(torch.nn.Module):
    """
    Ordinal Dice Loss (multiclasse) sans décalage +1.
    - Cible ordinale 'douce' : target_c = (min(c, t) / max(c, t)) ** gamma, avec conventions:
        * si c==t>0 -> 1
        * si l'un est 0 et l'autre >0 -> 0 (erreur max)
        * si c==t==0 -> ignoré si exclude_background=True (par défaut)
    - Retire le canal 0 (background) du calcul pour ne pas compter 0->0 (vrais négatifs).
    - `ignore_index` exclut totalement ces pixels du numérateur et dénominateur.
    - Option `use_logits`: applique un softmax interne si True.

    Args:
        smooth (float): lissage numérique.
        gamma (float): contrôle la décroissance ordinale (gamma>1 = plus piqué).
        ignore_index (int|None): étiquette à ignorer (ex: 255).
        use_logits (bool): True si `predict` sont des logits.
        exclude_background (bool): True = retire le canal 0 du calcul (par défaut).
    """
    def __init__(self, smooth: float = 1e-5, gamma: float = 1.0,
                 ignore_index = None, use_logits: bool = True,
                 exclude_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.use_logits = use_logits
        self.exclude_background = exclude_background

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        predict: (N, C, H, W) logits ou probabilités
        target : (N, H, W) entiers dans [0..C-1], 0 = background
        """
        N, C = predict.shape
        # Probabilités
        probs = F.softmax(predict, dim=1) if self.use_logits else predict
        probs = probs.clamp_min(0.0)

        # Aplatir
        p = probs.view(N, C, -1)         # (N,C,HW)
        t = target.view(N, -1)           # (N,HW)

        # Masque validité
        if self.ignore_index is not None:
            valid = (t != self.ignore_index)          # (N,HW)
        else:
            valid = torch.ones_like(t, dtype=torch.bool)

        # On prépare t_safe pour éviter de propager des valeurs indésirées
        t_safe = t.masked_fill(~valid, 0)             # (N,HW)

        # Retrait du background si demandé (recommandé pour ignorer 0->0)
        if self.exclude_background:
            # on enlève le canal 0
            p = p[:, 1:, :]                           # (N,C-1,HW)
            C_eff = C - 1
            # indices de classes considérées: 1..C-1
            c_idx = torch.arange(1, C, device=predict.device, dtype=p.dtype).view(1, C_eff, 1)  # (1,C-1,1)
        else:
            C_eff = C
            c_idx = torch.arange(0, C, device=predict.device, dtype=p.dtype).view(1, C, 1)      # (1,C,1)

        # t_idx: vraie classe par pixel
        t_idx = t_safe.unsqueeze(1).to(p.dtype)       # (N,1,HW)

        # Cible ordinale sans +1 : ratio = min(c,t)/max(c,t)
        num = torch.minimum(c_idx, t_idx)             # (N,C_eff,HW) via broadcast
        den = torch.maximum(c_idx, t_idx)

        # Cas exclude_background=True : den >= 1 (car c_idx >=1), donc pas de 0/0
        # Cas exclude_background=False : traiter (0,0) -> ratio = 1
        if self.exclude_background:
            ord_target = (num / den.clamp_min(1e-12)).pow(self.gamma)
        else:
            # là, (c=0,t=0) donne den=0 ; on le pose à 1 (match parfait)
            ratio = torch.zeros_like(den, dtype=p.dtype)
            nonzero = den > 0
            ratio[nonzero] = (num[nonzero] / den[nonzero]).pow(self.gamma)
            # force (0,0) -> 1
            both_zero = (den == 0)                    # implique aussi num==0
            ratio[both_zero] = 1.0
            ord_target = ratio

        # Appliquer le masque validité (exclure ignore_index)
        valid_f = valid.unsqueeze(1).to(p.dtype)      # (N,1,HW)
        p = p * valid_f
        ord_target = ord_target * valid_f

        # Dice (soft) par classe
        intersection = (p * ord_target).sum(dim=2)                    # (N,C_eff)
        denom = p.pow(2).sum(dim=2) + ord_target.sum(dim=2)           # (N,C_eff)
        dice_nc = (2 * intersection + self.smooth) / (denom + self.smooth)

        # Moyenne sur classes puis batch
        dice_n = dice_nc.mean(dim=1)                                   # (N,)
        loss = 1.0 - dice_n.mean()
        return loss

class OrdinalDiceLossAndWKLoss(torch.nn.modules.loss._WeightedLoss):

    def __init__(
        self,
        num_classes: int,
        C: float = 0.5,
        wk_penalization_type: str = "quadratic",
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=True,
        learned=False
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
        self.learned = learned
        if learned:
            self.C = torch.nn.Parameter(torch.tensor(C))
        else:
            self.C = torch.nn.Parameter(torch.tensor(C), requires_grad=False)

        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if C < 0 or C > 1:
            raise ValueError(f"C must be between 0 and 1, but is {C}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.dice = OrdinalDiceLoss()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            The weighted sum of MCE and QWK loss
        """

        if self.learned:
            C = torch.nn.functional.sigmoid(self.C)
        else:
            C = self.C

        wk_result = self.wk(y_pred, y_true)
        dice_result = self.dice(y_pred, y_true)

        return C * wk_result + (1 - C) * dice_result
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}

        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output):

        Cs = [mcewk_log["C"] for mcewk_log in mcewk_logs]
        epochs = [mcewk_log["epoch"] for mcewk_log in mcewk_logs]

        # Sauvegarde pickle
        egpd_to_save = {"epoch": epochs, "C": Cs}
        save_object(egpd_to_save, 'Cs.pkl', dir_output)

        # kappa vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, Cs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cs')
        plt.title('DWK Cs over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'dwk_Cs_epochs.png')
        plt.close()

class GeneralizedWassersteinDiceLoss(torch.nn.Module):

    def __init__(self,
        background: int = 0,           # index of background class
        ignore_index = None,
        use_logits: bool = True,
        eps: float = 1e-6):
        super().__init__()

        self.M = torch.tensor([[0, 1, 2, 3, 4],
                  [1, 0, 0.25, 0.33, 0.5],
                  [2, 0.25, 0, 0.25, 0.33],
                  [3, 0.33, 0.25, 0, 0.25],
                  [4, 0.5, 0.33, 0.25, 0]])
        
        self.background = background
        self.ignore_index = ignore_index
        self.use_logits = use_logits
        self.eps = eps

    def forward(self,
        logits: torch.Tensor,          # (N, C, H, W)  logits or probs
        target: torch.Tensor,          # (N, H, W)      int labels in [0..C-1]
    ) -> torch.Tensor:
        """
        Generalized Wasserstein Dice Loss (GWDL) as in Fidon et al.
        - self.M[l,c] is the cost of confusing l with c (self.M.diag() == 0).
        - 'background' is the index of the background class b.
        - If use_logits, applies softmax over channel dim.
        - Pixels with target == ignore_index are excluded from the loss.
        """
        assert self.M.dim() == 2 and self.M.size(0) == self.M.size(1), "self.M must be square (C x C)"
        N, C = logits.shape
        assert self.M.size(0) == C, "self.M size must match number of classes C"

        # probabilities
        probs = F.softmax(logits, dim=1) if self.use_logits else logits
        probs = probs.clamp_min(0.0)

        # flatten spatial, build validity mask
        P = probs.permute(0, 1).reshape(-1, C)  # (N*H*W, C)
        y = target.reshape(-1)                        # (N*H*W,)
        if self.ignore_index is not None:
            valid_mask = (y != self.ignore_index)
        else:
            valid_mask = torch.ones_like(y, dtype=torch.bool)

        if valid_mask.sum() == 0:
            # no valid pixels -> no contribution
            return logits.new_tensor(0.0)

        P = P[valid_mask]          # (V, C)
        y = y[valid_mask]          # (V,)

        # per-pixel Wasserstein error: delta_i = <self.M[y_i, :], P_i[:]>
        # gather rows of self.M by true labels
        self.M = self.M.to(P.device, P.dtype)
        self.M_y = self.M.index_select(0, y)       # (V, C)
        delta = (self.M_y * P).sum(dim=1)     # (V,)

        # total error
        total_error = delta.sum()        # scalar

        # weights vs background: w_i = self.M[y_i, background]
        w_bg = self.M[:, self.background]          # (C,)
        w = w_bg.index_select(0, y)      # (V,)

        # "semantic TPs" term (weighted), following the canonical GWDL numerator
        # TP_weighted = sum_i w_i * (w_i - delta_i)
        tp_weighted = (w * (w - delta)).sum()

        # Wasserstein-Dice score and loss
        num = 2.0 * tp_weighted
        den = num + total_error + self.eps
        score = num / den
        loss = 1.0 - score
        return loss