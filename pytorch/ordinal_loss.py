import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.loss_utils import *
from forecasting_models.pytorch.classification_loss import WeightedCrossEntropyLoss
from typing import List

###################################### Ordinality ##########################################

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
        sample_weight: Optional[torch.Tensor] = None,
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

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss.device)
            weighted_loss = loss * sample_weight
            return torch.sum(weighted_loss) / torch.sum(sample_weight)

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

    def forward(
        self,
        probs,
        target,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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

        loss_per_sample = 1.0 - dice_per_n

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss_per_sample.device)
            weighted_loss = loss_per_sample * sample_weight
            return torch.sum(weighted_loss) / torch.sum(sample_weight)

        return loss_per_sample.mean()

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

    def dice_loss(
        self,
        y_pred,
        y_true,
        smooth=1e-6,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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
        loss_per_sample = 1 - dice_coef

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss_per_sample.device)
            weighted_loss = loss_per_sample * sample_weight
            return torch.sum(weighted_loss) / torch.sum(sample_weight)

        return loss_per_sample.mean()

    def forward(
        self,
        y_pred,
        y_true,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        Calcul de la Dice Loss.

        :param y_pred: Tenseur contenant les prédictions du modèle (probabilités ou valeurs continues).
        :param y_true: Tenseur contenant les vérités terrain (cibles).
        
        :return: La Dice Loss.
        """
        return self.dice_loss(y_pred, y_true, sample_weight=sample_weight)
    
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
        else:
            self.weight = None
        self.smooth = 1e-5

        self.ignore_index = ignore_index

    def forward(
        self,
        predict,
        target,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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

        if self.weight is not None:
            dice_per_sample = (dice_coef * self.weight.view(1, -1)).sum(dim=1)
        else:
            dice_per_sample = dice_coef.mean(dim=1)

        loss_per_sample = 1 - dice_per_sample

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss_per_sample.device)
            weighted_loss = loss_per_sample * sample_weight
            return torch.sum(weighted_loss) / torch.sum(sample_weight)

        return loss_per_sample.mean()

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

    def forward(
        self,
        y_pred,
        y_true,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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
            class_weight_factors = torch.gather(
                tiled_class_weight, dim=1, index=y_true_indices
            )
            loss = loss * class_weight_factors

        loss_per_sample = loss.sum(dim=1)

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss_per_sample.device)
            weighted_loss = loss_per_sample * sample_weight
            return -torch.sum(weighted_loss) / torch.sum(sample_weight)

        return -loss_per_sample.mean()

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

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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

        wk_result = self.wk(y_pred, y_true, sample_weight=sample_weight)
        mce_result = self.mce(y_pred, y_true, sample_weight=sample_weight)

        res = {
            'total_loss' : C * wk_result + (1 - C) * mce_result,
            'mce' : mce_result,
            "wk" : wk_result
        }

        return res
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}
        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output, best_epoch=None):

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

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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

        wk_result = self.wk(y_pred, y_true, sample_weight=sample_weight)
        dice_result = self.dice(y_pred, y_true, sample_weight=sample_weight)

        res = {
            'total_loss' : C * wk_result + (1 - C) * dice_result,
            'dice' : dice_result,
            "wk" : wk_result
        }

        return res
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}
        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output, best_epoch=None):

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
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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

        wk_result = self.wk(y_pred, y_true, sample_weight=sample_weight)
        dice_result = self.dice(y_pred, y_true, sample_weight=sample_weight)

        res = {
            'total_loss' : C * wk_result + (1 - C) * dice_result,
            'dice' : dice_result,
            "wk" : wk_result
        }

        return res
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}

        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output, best_epoch=None):

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

    def forward(
        self,
        predict: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        loss_per_sample = 1.0 - dice_n

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss_per_sample.device)
            weighted_loss = loss_per_sample * sample_weight
            return torch.sum(weighted_loss) / torch.sum(sample_weight)

        return loss_per_sample.mean()

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

        res = {
            'total_loss' : C * wk_result + (1 - C) * dice_result,
            'dice' : dice_result,
            "wk" : wk_result
        }

        return res
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}

        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output, best_epoch=None):

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

class GeneralizedWassersteinDiceLoss(nn.Module):
    def __init__(self,
                 background: int = 0,
                 M = None,
                 ignore_index = None,
                 use_logits: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        
        """self.M = torch.tensor([ [0, 1, 2, 3, 4], [1, 0, 0.25, 0.33, 0.5], [2, 1.0, 0, 0.25, 0.33], [3, 1.25, 0.75, 0, 0.33], [4, 1.75, 1.0, 0.75, 0]])"""
        """self.M = torch.tensor([ [0, 1, 2, 3, 4], [1, 0, 0.25, 0.33, 0.5], [2, 1.0, 0, 0.25, 0.33], [3, 1.25, 1.0, 0, 0.33], [4, 1.5, 1.25, 0.5, 0]])"""
        if M is None:
            self.M = torch.tensor([ [0, 1.5, 2.5, 3.5, 4.5], [0.75, 0, 0.25, 0.33, 0.5], [1.5, 1.0, 0, 0.25, 0.33], [2, 1.25, 1.0, 0, 0.33], [3.5, 1.5, 1.25, 1.0, 0]])
            """
            self.M = torch.tensor([[0.00, 1.0, 2.0, 3.0, 4.0],
                                   [1.0, 0.00, 1.00, 2.0, 3.0],
                                   [2.0, 1.00, 0.00, 1.00, 2.0],
                                   [3.0, 2.0, 1.00, 0.00, 1.00],
                                   [4.0, 3.0, 2.0, 1.00, 0.00]], dtype=torch.float32)"""
        else:
            self.M = M.clone().detach().to(torch.float32)
            
        self.background = background
        self.ignore_index = ignore_index
        self.use_logits = use_logits
        self.eps = eps

    # ---------------- Confusion matrix (brute) ----------------
    def get_confusion_matrix(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = int(self.M.size(0))
        preds = (torch.softmax(logits, dim=1).argmax(dim=1) if self.use_logits
                 else logits.argmax(dim=1))
        y = target
        if self.ignore_index is not None:
            mask = (y != self.ignore_index)
            preds, y = preds[mask], y[mask]
        y = y.reshape(-1).to(torch.int64)
        preds = preds.reshape(-1).to(torch.int64)
        valid = (y >= 0) & (y < C) & (preds >= 0) & (preds < C)
        if valid.numel() == 0 or valid.sum() == 0:
            return torch.zeros((C, C), dtype=torch.long, device=logits.device)
        y, preds = y[valid], preds[valid]
        cm = torch.zeros((C, C), dtype=torch.long, device=logits.device)
        idx = y * C + preds
        counts = torch.bincount(idx, minlength=C*C)
        cm += counts.view(C, C)
        return cm

    # -------------- Normalisation robuste max(row,col) --------------
    def _normalize_confusion(self, conf_raw: torch.Tensor) -> torch.Tensor:
        conf = conf_raw.to(torch.float32)
        row_sum = conf.sum(dim=1, keepdim=True).clamp_min(1e-12)
        col_sum = conf.sum(dim=0, keepdim=True).clamp_min(1e-12)
        conf_row = conf / row_sum                    # P(pred=j | true=i)
        conf_col = conf / col_sum                    # P(true=i | pred=j)
        conf_norm = torch.maximum(conf_row, conf_col)
        return conf_norm

    # ---------------- API de mise à jour générique ----------------
    def update_after_batch(self, logits, target, **kwargs):
        # Par défaut on fait la version ordinale par ligne
        #return self.confusion_matrix_update(logits, target, **kwargs)
        pass

    # -------------- Mise à jour ordinale par ligne --------------
    def confusion_matrix_update_rows(self,
        logits: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.05,
        normalize: bool = True,
    ):
        """
        Pour chaque vraie classe i, ajoute alpha * max_j!=i conf[i,j] à TOUTES les cases (i,j!=i).
        Diagonale maintenue à 0.
        """
        conf_raw = self.get_confusion_matrix(logits, target).to(self.M.device)
        conf = self._normalize_confusion(conf_raw) if normalize else conf_raw.to(torch.float32)

        C = conf.size(0)
        conf_off = conf.clone()
        conf_off.fill_diagonal_(0.0)
        m_row = conf_off.max(dim=1).values  # (C,)

        per_cell = m_row.unsqueeze(1).expand(C, C).clone()
        per_cell.fill_diagonal_(0.0)

        self.M = self.M.to(per_cell.device, per_cell.dtype)
        self.M = self.M + (alpha * per_cell)
        self.M.fill_diagonal_(0.0)
        return conf
    
    # -------------- Mise à jour par coins (sur/sous-prédictions) --------------
    def confusion_matrix_update_corners(self,
        logits: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.05,
        normalize: bool = True,
        weights = None,   # (C,C) optionnelle pour pondérer le max des coins
    ):
        """
        Ajoute:
          - alpha * u_max à toutes les cases i<j (coin haut-droit, sur-prédictions)
          - alpha * l_max à toutes les cases i>j (coin bas-gauche, sous-prédictions)
        où u_max/l_max sont les max de conf (ou conf*weights) dans chaque coin.
        """
        conf_raw = self.get_confusion_matrix(logits, target).to(self.M.device)
        conf = self._normalize_confusion(conf_raw) if normalize else conf_raw.to(torch.float32)

        C = conf.size(0)
        eff = conf if weights is None else (conf * weights.to(conf.device, conf.dtype))

        upper_mask = torch.triu(torch.ones(C, C, device=conf.device, dtype=torch.bool), diagonal=1)
        lower_mask = torch.tril(torch.ones(C, C, device=conf.device, dtype=torch.bool), diagonal=-1)

        u_max = eff[upper_mask].max().item() if upper_mask.any() else 0.0
        l_max = eff[lower_mask].max().item() if lower_mask.any() else 0.0

        per_cell = torch.zeros_like(conf)
        per_cell[upper_mask] = u_max
        per_cell[lower_mask] = l_max
        per_cell.fill_diagonal_(0.0)

        self.M = self.M.to(per_cell.device, per_cell.dtype)
        self.M = self.M + (alpha * per_cell)
        self.M.fill_diagonal_(0.0)
        return conf
    
    # -------------- Mise à jour par cases (sur/sous-prédictions) --------------
    def confusion_matrix_update(self,
        logits: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.05,
        normalize: bool = True,
        weights=None,   # (C,C) optionnelle pour pondérer chaque case
    ):
        """
        Ajoute, pour chaque case (i,j) != (j,j), la quantité alpha * err_ij
        où err_ij provient directement de la matrice de confusion (éventuellement
        normalisée et/ou pondérée par weights). Contrairement à la version
        précédente, on n'utilise plus le maximum par coin.

        """

        # Matrice de confusion (brute ou normalisée)
        conf_raw = self.get_confusion_matrix(logits, target).to(self.M.device)
        conf = self._normalize_confusion(conf_raw) if normalize else conf_raw.to(torch.float32)

        # Pondération éventuelle case par case
        eff = conf if weights is None else (conf * weights.to(conf.device, conf.dtype))

        C = conf.size(0)
        upper_mask = torch.triu(torch.ones(C, C, device=conf.device, dtype=torch.bool), diagonal=1)
        lower_mask = torch.tril(torch.ones(C, C, device=conf.device, dtype=torch.bool), diagonal=-1)

        # On reprend directement les erreurs par case dans chaque coin
        per_cell = torch.zeros_like(conf)
        per_cell[upper_mask] = eff[upper_mask]  # sur-prédictions
        per_cell[lower_mask] = eff[lower_mask]  # sous-prédictions
        per_cell.fill_diagonal_(0.0)

        # Mise à jour de M
        self.M = self.M.to(per_cell.device, per_cell.dtype)
        self.M = self.M + (alpha * per_cell)
        self.M.fill_diagonal_(0.0)

        return conf

    def forward(
        self,
        logits: torch.Tensor,          # (N, C, H, W)  logits or probs
        target: torch.Tensor,          # (N, H, W)      int labels in [0..C-1]
        sample_weight: Optional[torch.Tensor] = None,
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

        probs = probs.view(N, C, -1)
        targets = target.view(N, -1)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        loss_per_sample = torch.zeros(N, device=logits.device, dtype=probs.dtype)
        valid_sample = valid_mask.any(dim=1)

        self.M = self.M.to(probs.device, probs.dtype)
        w_bg = self.M[:, self.background]

        for idx in range(N):
            if not valid_sample[idx]:
                continue

            mask = valid_mask[idx]
            P_i = probs[idx, :, mask]  # (C, V)
            y_i = targets[idx, mask].to(torch.int64)  # (V,)

            P_i = P_i.transpose(0, 1)  # (V, C)
            M_y = self.M.index_select(0, y_i)  # (V, C)
            delta = (M_y * P_i).sum(dim=1)

            total_error = delta.sum()

            w = w_bg.index_select(0, y_i)
            tp_weighted = (w * (w - delta)).sum()

            num = 2.0 * tp_weighted
            den = num + total_error + self.eps
            score = num / den
            loss_per_sample[idx] = 1.0 - score

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss_per_sample.device)
            effective_weight = sample_weight * valid_sample.to(sample_weight.dtype)
            if torch.sum(effective_weight) == 0:
                return loss_per_sample.new_tensor(0.0)
            weighted_loss = loss_per_sample * sample_weight
            return torch.sum(weighted_loss) / torch.sum(effective_weight)

        if valid_sample.any():
            return loss_per_sample[valid_sample].mean()

        return loss_per_sample.new_tensor(0.0)

    def plot_params(self,
                log,
                dir_output,
                best_epoch=None,
                class_names=None,      # list of class names (length C) or None
                title="Misclassification cost matrix",
                figsize=(6, 5),
                annotate=True,         # if True, write (i,j) and the numeric value in each cell
                cmap="cool",        # matplotlib colormap
                vmin=None, vmax=None,  # colorbar bounds (auto if None)
                savepath=None):        # if provided, save the figure here (png, pdf, ...)
        """
        Display self.M (C x C) as a heatmap.

        - self.M[i,j] = cost of predicting class j when the true class is i.
        - class_names (optional): tick labels for axes.
        - annotate: write "(i,j)" and the numeric value in each cell.
        - savepath: if provided, save the figure to this path; otherwise saves to dir_output/'M.png'.
        """
        # Safely move to CPU and convert to numpy
        M = self.M
        if isinstance(M, torch.Tensor):
            M = M.detach().to("cpu").numpy()
        else:
            M = np.asarray(M)

        C = M.shape[0]
        if class_names is not None:
            assert len(class_names) == C, "class_names must have C elements."

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

        # Axes and titles
        #ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")

        # Ticks + labels
        ax.set_xticks(np.arange(C))
        ax.set_yticks(np.arange(C))
        if class_names is not None:
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
        else:
            ax.set_xticklabels(np.arange(C))
            ax.set_yticklabels(np.arange(C))

        # Light grid for readability
        ax.set_xticks(np.arange(-.5, C, 1), minor=True)
        ax.set_yticks(np.arange(-.5, C, 1), minor=True)
        ax.grid(which="minor", linestyle=":", linewidth=0.5)
        ax.tick_params(top=False, bottom=True, left=True, right=False)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Cost", rotation=90, labelpad=10)

        # Annotations
        if annotate:
            # Choose text color depending on background intensity
            finite_vals = M[np.isfinite(M)]
            if finite_vals.size == 0:
                thresh = 0.0
            else:
                thresh = (np.nanmax(finite_vals) + np.nanmin(finite_vals)) / 2.0

            for i in range(C):
                for j in range(C):
                    val = M[i, j]
                    text_color = "white" if np.isfinite(val) and val > thresh else "black"
                    ax.text(j, i, f"({i},{j})\n{val:.2f}",
                            ha="center", va="center",
                            color=text_color, fontsize=9, linespacing=1.1)

        fig.tight_layout()

        # Save
        if savepath is None:
            savepath = dir_output / "M.png"
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

        return fig, ax

class CEWKLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        C: float = 1.0,              # poids du WKLoss (inchangé)
        C1: float = 0.5,             # trade-off entre wce1 et wce2
        wk_penalization_type: str = "quadratic",
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        use_logits: bool = True,
        learned: bool = False,
        background_class: int = 0,
    ) -> None:
        super().__init__(
        )

        self.num_classes = num_classes
        self.learned = learned

        self.C  = C   # poids pour WKLoss
        self.C1 = C1  # c1: pondère wce1 vs wce2

        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible "
                f"with num_classes {num_classes}"
            )

        for name, val in (("C", C), ("C1", C1)):
            if val < 0 or val > 1:
                raise ValueError(f"{name} must be between 0 and 1, but is {val}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported. Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits
        self.background_class = background_class

        # sous-losses
        self.wk  = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.wce_fp_bg = WeightedCrossEntropyLoss(self.num_classes)
        self.wce_fn_bg = WeightedCrossEntropyLoss(self.num_classes)

    def _flatten_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if y_pred.dim() <= 2:
            return y_pred, y_true.view(-1)

        batch, classes = y_pred.size(0), y_pred.size(1)
        flattened_pred = y_pred.view(batch, classes, -1).permute(0, 2, 1).reshape(-1, classes)
        flattened_true = y_true.view(batch, -1).reshape(-1)
        return flattened_pred, flattened_true

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        # paramètres (sigmoid si apprenables)
        if self.learned:
            C  = torch.sigmoid(self.C)
            c1 = torch.sigmoid(self.C1)
        else:
            C, c1 = self.C, self.C1

        y_pred_flat, y_true_flat = self._flatten_inputs(y_pred, y_true)

        if sample_weight is not None:
            sample_weight_flat = sample_weight.view(-1)
            if sample_weight_flat.numel() != y_true_flat.numel():
                raise ValueError(
                    "Sample weights must have the same number of elements as the targets."
                )
        else:
            sample_weight_flat = None            

        if self.use_logits:
            pred_classes = torch.argmax(torch.nn.functional.softmax(y_pred_flat, dim=1), dim=1)
        else:
            pred_classes = torch.argmax(y_pred_flat, dim=1)

        
        # masques
        mask_true0_wrong = (y_true_flat == self.background_class) & (pred_classes != self.background_class)  # wce1
        mask_pred0_wrong = (pred_classes == self.background_class) & (y_true_flat != self.background_class)  # wce2
        positive_mask    = (y_true_flat != self.background_class) & (pred_classes != self.background_class)# WK
        
        # WCE #1
        if mask_true0_wrong.any():
            wce1_pred = y_pred_flat[mask_true0_wrong]
            wce1_true = y_true_flat[mask_true0_wrong]
            wce1_weight = sample_weight_flat[mask_true0_wrong] if sample_weight_flat is not None else None
            wce1 = self.wce_fp_bg(wce1_pred, wce1_true, sample_weight=wce1_weight)

        # WCE #2
        if mask_pred0_wrong.any():
            wce2_pred = y_pred_flat[mask_pred0_wrong]
            wce2_true = y_true_flat[mask_pred0_wrong]
            wce2_weight = sample_weight_flat[mask_pred0_wrong] if sample_weight_flat is not None else None
            wce2 = self.wce_fn_bg(wce2_pred, wce2_true, sample_weight=wce2_weight)

        # WKLoss
        #if positive_mask.any():
        #    wk_pred = y_pred_flat[positive_mask]
        #    wk_true = y_true_flat[positive_mask]
        #    wk_weight = sample_weight_flat[positive_mask] if sample_weight_flat is not None else None
        #    wk_result = self.wk(wk_pred, wk_true, sample_weight=wk_weight)
        #else:
        #    wk_result = 0.0
        
        wk_result = self.wk(y_pred, y_true)
        
        loss = C * wk_result
        
        if 'wce1' in locals():
            loss += c1 * wce1
        
        if 'wce2' in locals():
            loss += (1 - c1) * wce2
        
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100, num_classes=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if not isinstance(alpha, (float, int)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets, sample_weight=None):
        """
        inputs: (N, C) logits
        targets: (N,) or (N, C)
        """
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,HW
            inputs = inputs.transpose(1, 2)    # N,HW,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # NHW,C
            targets = targets.view(-1)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        # Handle alpha
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Gather alpha for each target
            # targets should be indices here because we used cross_entropy
            # If targets has ignore_index, we need to be careful with indexing
            # We can use a safe gathering or mask
            
            # Create a weight tensor matching the batch size
            alpha_t = self.alpha[targets]
            
            # If ignore_index was used, targets might have invalid indices. 
            # However, ce_loss is already 0 for ignored indices (if reduction='none' handled it correctly? 
            # Actually cross_entropy with reduction='none' returns 0 for ignored index? 
            # Let's check docs or assume standard behavior: usually it returns 0 or we mask it.)
            # But indexing self.alpha[targets] will fail if targets has -100.
            
            if self.ignore_index is not None:
                 # Create a safe target for indexing
                 safe_targets = targets.clone()
                 mask = targets == self.ignore_index
                 safe_targets[mask] = 0 # Dummy index
                 alpha_t = self.alpha[safe_targets]
                 alpha_t[mask] = 0.0 # Zero out alpha for ignored
            else:
                 alpha_t = self.alpha[targets]
                 
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if sample_weight is not None:
            if sample_weight.dim() != 1 or sample_weight.size(0) != focal_loss.size(0):
                 # Try to broadcast or reshape if possible, but for now assume it matches batch size
                 # If inputs were flattened (e.g. segmentation), sample_weight needs to be flattened too
                 pass 
            
            # Simple handling if sample_weight matches batch size and we haven't flattened too much
            # If we flattened inputs (segmentation), we need to be careful. 
            # Assuming standard classification (N, C) for now based on other losses.
            if sample_weight.numel() == focal_loss.numel():
                 focal_loss = focal_loss * sample_weight

        if self.reduction == 'mean':
            if sample_weight is not None:
                 return focal_loss.sum() / sample_weight.sum()
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLossAndWKLoss(torch.nn.modules.loss._WeightedLoss):
    """
    L = CE_focal + lambda * L_QWK
    """
    def __init__(
        self,
        num_classes: int,
        C: float = 1.0, # This is the lambda parameter (weight for QWK)
        gamma: float = 2.0,
        alpha: Union[float, List[float], Tensor] = 1.0,
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
        self.use_logits = use_logits

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='none') # We handle reduction manually

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        if self.learned:
            C = torch.sigmoid(self.C)
        else:
            C = self.C

        # QWK Loss
        wk_result = self.wk(y_pred, y_true, sample_weight=sample_weight)
        
        # Focal Loss
        # FocalLoss expects targets as class indices usually, but let's check what WKLoss expects
        # WKLoss handles both. Our FocalLoss implementation expects indices.
        # If y_true is one-hot, we might need to convert it for FocalLoss if using cross_entropy
        
        y_true_focal = y_true
        if y_true.dim() > 1 and y_true.shape[1] == self.num_classes:
             y_true_focal = torch.argmax(y_true, dim=1)
        
        focal_result = self.focal(y_pred, y_true_focal, sample_weight=sample_weight)
        
        # Reduce Focal Loss
        if sample_weight is not None:
             focal_result = focal_result.sum() / sample_weight.sum()
        else:
             focal_result = focal_result.mean()

        # L = Focal + lambda * QWK
        # The user asked for L = CEfocal + lambda * LQWK
        # Here C is lambda.

        res = {
            'total_loss' : focal_result + C * wk_result,
            'focal' : focal_result,
            "wk" : wk_result
        }
        
        return res

    def get_learnable_parameters(self):
        if not self.learned:
            return {}
        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output, best_epoch=None):
        Cs = [mcewk_log["C"] for mcewk_log in mcewk_logs]
        epochs = [mcewk_log["epoch"] for mcewk_log in mcewk_logs]

        # Sauvegarde pickle
        egpd_to_save = {"epoch": epochs, "C": Cs}
        save_object(egpd_to_save, 'Cs_focal.pkl', dir_output)

        # kappa vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, Cs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cs')
        plt.title('FocalWK Cs over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'focalwk_Cs_epochs.png')
        plt.close()

class ClusterInversionLoss(nn.Module):
    """
    Penalite d'inversions (ranking) *au sein d'un meme departement*.

    Idee:
      - On transforme les logits (N,C) en proba softmax p_{ik}
      - On calcule un score ordinal continu s_i = E[y_hat] = sum_k k * p_{ik}
      - Pour chaque departement d, on considere des paires (i,j) dans d telles que y_i > y_j
      - On penalise si s_i n'est pas au-dessus de s_j (avec marge optionnelle)

    Loss (version logistique, stable):
      l_ij = softplus(-( (s_i - s_j) - margin ))
           = log(1 + exp( -((s_i - s_j) - margin) ))

    Options:
      - weight_by_distance: ponderer par |y_i - y_j|
      - max_pairs_per_dep: echantillonnage pour eviter O(n^2) si batch gros
    """

    def __init__(
        self,
        num_classes: int = 5,
        margin: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        max_pairs_per_dep: int = 50,
        weight_by_distance: bool = True,
        eps: float = 1e-8,
        id = None
    ):
        super().__init__()
        assert id is not None
        self.id = id
        self.num_classes = num_classes
        self.margin = margin
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.max_pairs_per_dep = max_pairs_per_dep
        self.weight_by_distance = weight_by_distance
        self.eps = eps

        # vecteur [0,1,2,3,4] pour calculer l'esperance
        self.register_buffer("class_values", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, inputs, targets, clusters_ids, sample_weight=None):
        """
        inputs:      (N, C) logits
        targets:     (N,) indices de classe (0..C-1) ou ignore_index
        clusters_ids: (N,) identifiant departement (int)
        sample_weight: (N,) optionnel, poids par observation (pas par paire)

        retourne: scalaire si reduction != 'none', sinon une loss scalaire aussi
                  (on ne renvoie pas une loss par echantillon car la loss est par paires)
        """
        if inputs.dim() != 2:
            raise ValueError(f"inputs doit etre (N,C). Recu: {tuple(inputs.shape)}")
        if targets.dim() != 1 or clusters_ids.dim() != 1:
            raise ValueError("targets et clusters_ids doivent etre des tenseurs 1D (N,).")
        if targets.size(0) != inputs.size(0) or clusters_ids.size(0) != inputs.size(0):
            raise ValueError("inputs, targets, clusters_ids doivent avoir le meme N.")

        device = inputs.device
        N, C = inputs.shape
        if C != self.num_classes:
            # pas obligatoire, mais souvent utile pour debug
            pass

        # masque ignore_index
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() < 2:
            # pas assez d'infos pour former des paires
            return inputs.new_tensor(0.0)

        inputs_v = inputs[valid_mask]
        targets_v = targets[valid_mask]
        deps_v = clusters_ids[valid_mask]
        if sample_weight is not None:
            sw_v = sample_weight[valid_mask].to(device)
        else:
            sw_v = None

        # proba et score s = E[y_hat]
        probs = F.softmax(inputs_v, dim=1)  # (Nv,C)
        class_vals = self.class_values.to(device=device, dtype=probs.dtype)  # (C,)
        s = (probs * class_vals.unsqueeze(0)).sum(dim=1)  # (Nv,)

        total_loss = inputs.new_tensor(0.0)
        total_weight = inputs.new_tensor(0.0)

        # boucle par departement present dans le batch
        unique_deps = torch.unique(deps_v)

        for d in unique_deps:
            idx = torch.nonzero(deps_v == d, as_tuple=False).squeeze(1)
            nd = idx.numel()
            if nd < 2:
                continue

            s_d = s[idx]             # (nd,)
            y_d = targets_v[idx]     # (nd,)
            if sw_v is not None:
                sw_d = sw_v[idx]     # (nd,)

            # Construire toutes les paires (i,j) dans ce departement
            # Pour eviter O(nd^2) si nd grand, on peut echantillonner.
            if self.max_pairs_per_dep is None:
                # Toutes les combinaisons i<j
                pairs = torch.combinations(torch.arange(nd, device=device), r=2)  # (P,2)
            else:
                # Echantillonnage de paires: tirage uniforme sur indices
                # On sur-genere un peu, puis on filtre y_i != y_j.
                P = min(self.max_pairs_per_dep, nd * (nd - 1) // 2)
                if P == 0:
                    continue
                i = torch.randint(0, nd, (P,), device=device)
                j = torch.randint(0, nd, (P,), device=device)
                mask_neq = i != j
                i = i[mask_neq]
                j = j[mask_neq]
                if i.numel() == 0:
                    continue
                pairs = torch.stack([i, j], dim=1)

            i = pairs[:, 0]
            j = pairs[:, 1]

            y_i = y_d[i]
            y_j = y_d[j]
            s_i = s_d[i]
            s_j = s_d[j]

            # On veut des paires ordonnees telles que y_pos > y_neg
            # Deux cas: (y_i > y_j) ou (y_j > y_i)
            mask_ij = y_i > y_j
            mask_ji = y_j > y_i
            if not (mask_ij.any() or mask_ji.any()):
                continue

            # Construire deltas pour les paires valides avec ordre correct
            # Pour y_i > y_j : delta = s_i - s_j
            # Pour y_j > y_i : delta = s_j - s_i
            deltas = torch.cat([s_i[mask_ij] - s_j[mask_ij],
                                s_j[mask_ji] - s_i[mask_ji]], dim=0)

            # Marge: on veut delta >= margin
            # loss = softplus(-(delta - margin))
            loss_pairs = F.softplus(-(deltas - self.margin))

            # Pondération optionnelle par distance de classes |y_pos - y_neg|
            if self.weight_by_distance:
                dist_ij = (y_i[mask_ij] - y_j[mask_ij]).abs().to(loss_pairs.dtype)
                dist_ji = (y_j[mask_ji] - y_i[mask_ji]).abs().to(loss_pairs.dtype)
                w_dist = torch.cat([dist_ij, dist_ji], dim=0)
                loss_pairs = loss_pairs * w_dist

            # Pondération optionnelle par sample_weight (poids par observation)
            # Ici, on prend une moyenne des poids des deux elements de la paire.
            if sw_v is not None:
                w_ij = 0.5 * (sw_d[i[mask_ij]] + sw_d[j[mask_ij]]).to(loss_pairs.dtype)
                w_ji = 0.5 * (sw_d[j[mask_ji]] + sw_d[i[mask_ji]]).to(loss_pairs.dtype)
                w_pair = torch.cat([w_ij, w_ji], dim=0)
                loss_pairs = loss_pairs * w_pair
                pair_weight_sum = w_pair.sum()
            else:
                pair_weight_sum = loss_pairs.new_tensor(loss_pairs.numel(), dtype=loss_pairs.dtype)

            total_loss = total_loss + loss_pairs.sum()
            total_weight = total_weight + pair_weight_sum

        if total_weight.abs() < self.eps:
            return inputs.new_tensor(0.0)

        if self.reduction == "sum":
            return total_loss
        elif self.reduction == "mean":
            return total_loss / (total_weight + self.eps)
        else:
            # pour une loss par paire, 'none' n'est pas super defini ici
            # on renvoie quand meme la moyenne (comportement proche de 'mean')
            return total_loss / (total_weight + self.eps)


class FocalWKInversionLoss(torch.nn.modules.loss._WeightedLoss):
    """
    L = (B) * Focal + (A) * WK + (C) * Inversion
    Inversion appliquee uniquement au sein d'un meme departement
    """

    def __init__(
        self,
        num_classes,
        A=0.33,
        B=0.33,
        C=0.34,
        gamma=2.0,
        alpha=1.0,
        id=None,
        wk_penalization_type="quadratic",
        weight=None,
        reduction="mean",
        use_logits=True,
        # paramètres de la pénalité d'inversion
        inv_margin=0.0,
        inv_max_pairs_per_dep=2048,
        inv_weight_by_distance=True,
        inv_ignore_index=-100,
    ):
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )
        assert id is not None
        self.id = id
        self.num_classes = num_classes
        self.use_logits = use_logits
        self.wk_penalization_type = wk_penalization_type

        # --- Coefficients fixes ---
        self.A = torch.tensor(float(A))
        self.B = torch.tensor(float(B))
        self.C = torch.tensor(float(C))

        # --- Losses ---
        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )

        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")

        self.inv = ClusterInversionLoss(
            num_classes=self.num_classes,
            margin=inv_margin,
            ignore_index=inv_ignore_index,
            max_pairs_per_dep=inv_max_pairs_per_dep,
            weight_by_distance=inv_weight_by_distance,
            reduction="mean",
            id=id,
        )

    def forward(self, y_pred, y_true, clusters_ids, sample_weight=None):
        # --- WK ---
        wk_result = self.wk(y_pred, y_true, sample_weight=sample_weight)

        # --- Focal ---
        y_true_focal = y_true
        if y_true.dim() > 1 and y_true.shape[1] == self.num_classes:
            y_true_focal = torch.argmax(y_true, dim=1)

        focal_vec = self.focal(y_pred, y_true_focal, sample_weight=sample_weight)
        if sample_weight is not None:
            focal_result = focal_vec.sum() / (sample_weight.sum() + 1e-8)
        else:
            focal_result = focal_vec.mean()

        # --- Inversion ---
        y_true_inv = y_true
        if y_true.dim() > 1 and y_true.shape[1] == self.num_classes:
            y_true_inv = torch.argmax(y_true, dim=1)

        inv_result = self.inv(
            y_pred,
            y_true_inv,
            clusters_ids,
            sample_weight=sample_weight,
        )

        res = {'total_loss' :self.B * focal_result + self.A * wk_result + self.C * inv_result,
        'focal' : focal_result,
        'wk' : wk_result,
        'inv' : inv_result}

        return res
    
class MonoticRiskLoss(nn.Module):
    """
    MonotonicRiskLoss: combine 4 components inspired by your score framework:
      - AVG : encourages positive mean delta (separation on average)
      - MIN : penalizes worst-case delta (soft-min)
      - NEG : penalizes magnitude of negative deltas
      - VIOL: penalizes violations (delta < margin), weighted by class distance

    Design choices (simple, paper-friendly):
      - Score proxy: s_i = E[y_hat] = sum_k k * softmax(logits)_k
      - Pairs within each cluster (department/zone)
      - Ordered deltas: delta = s_pos - s_neg for pairs where y_pos > y_neg
      - Distance weight: |y_pos - y_neg| (float), applied ONLY to VIOL as requested
      - All component weights set to 1.0

    Notes:
      - Uses sampling of pairs per cluster to avoid O(n^2) when batch is large.
      - Reduction is "mean" over weighted sums, consistent across components.
    """

    def __init__(
        self,
        num_classes: int = 5,
        margin: float = 0.0,
        beta_softmin: float = 10.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        max_pairs_per_cluster: int = 50,
        eps: float = 1e-8,
        id=None,
    ):
        super().__init__()
        self.id = id
        self.num_classes = num_classes
        self.margin = margin
        self.beta_softmin = beta_softmin
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.max_pairs_per_cluster = max_pairs_per_cluster
        self.eps = eps

        # weights for the 4 components (all = 1.0 as requested)
        self.w_avg = 1.0
        self.w_min = 1.0
        self.w_neg = 1.0
        self.w_viol = 1.0

        # [0,1,2,3,4] for expectation
        self.register_buffer("class_values", torch.arange(num_classes, dtype=torch.float32))
        
        #self.ce_loss = WeightedCrossEntropyLoss(num_classes=num_classes)
        #self.ce_loss = FocalLoss(num_classes=num_classes)
        self.ce_loss = FocalLossAndWKLoss(num_classes=num_classes)

    # -----------------------
    # Internal helper methods
    # -----------------------
    def _expected_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute s = E[y_hat] from logits (N,C)."""
        probs = F.softmax(logits, dim=1)
        class_vals = self.class_values.to(device=logits.device, dtype=probs.dtype)
        return (probs * class_vals.unsqueeze(0)).sum(dim=1)

    def _sample_pairs(self, nd: int, device: torch.device) -> torch.Tensor:
        """Return pairs (P,2) indices in [0, nd)."""
        if nd < 2:
            return torch.empty((0, 2), dtype=torch.long, device=device)

        if self.max_pairs_per_cluster is None:
            return torch.combinations(torch.arange(nd, device=device), r=2)

        Pmax = nd * (nd - 1) // 2
        P = min(self.max_pairs_per_cluster, Pmax)
        if P <= 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)

        i = torch.randint(0, nd, (P,), device=device)
        j = torch.randint(0, nd, (P,), device=device)
        m = i != j
        if not m.any():
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.stack([i[m], j[m]], dim=1)

    def _ordered_deltas_and_weights(
        self,
        s_d: torch.Tensor,           # (nd,)
        y_d: torch.Tensor,           # (nd,)
        sw_d: torch.Tensor,   # (nd,) or None
        device: torch.device,
    ):
        """
        Build ordered deltas for a given cluster:
          - delta = s_pos - s_neg where y_pos > y_neg
          - viol_weight = |y_pos - y_neg| (distance) * optional pair sample weight
          - base_pair_weight for AVG/NEG as optional sample-weight average (not distance-weighted)
        Returns:
          deltas: (P,)
          viol_w: (P,)  distance-weighted (and sample-weighted if provided)
          base_w: (P,)  sample-weighted if provided else ones
        """
        nd = y_d.numel()
        pairs = self._sample_pairs(nd, device=device)
        if pairs.numel() == 0:
            return None, None, None

        i = pairs[:, 0]
        j = pairs[:, 1]

        y_i = y_d[i]
        y_j = y_d[j]
        s_i = s_d[i]
        s_j = s_d[j]

        mask_ij = y_i > y_j
        mask_ji = y_j > y_i
        if not (mask_ij.any() or mask_ji.any()):
            return None, None, None

        # deltas with correct orientation
        deltas_ij = s_i[mask_ij] - s_j[mask_ij]
        deltas_ji = s_j[mask_ji] - s_i[mask_ji]
        deltas = torch.cat([deltas_ij, deltas_ji], dim=0)

        # distance weights for VIOL
        dist_ij = (y_i[mask_ij] - y_j[mask_ij]).abs()
        dist_ji = (y_j[mask_ji] - y_i[mask_ji]).abs()
        dist = torch.cat([dist_ij, dist_ji], dim=0).to(dtype=deltas.dtype)

        # sample weights (optional): average of the two samples in the pair
        if sw_d is not None:
            w_ij = 0.5 * (sw_d[i[mask_ij]] + sw_d[j[mask_ij]]).to(dtype=deltas.dtype)
            w_ji = 0.5 * (sw_d[j[mask_ji]] + sw_d[i[mask_ji]]).to(dtype=deltas.dtype)
            base_w = torch.cat([w_ij, w_ji], dim=0)
        else:
            base_w = torch.ones_like(deltas)

        viol_w = dist * base_w  # distance-weighted VIOL as requested
        return deltas, viol_w, base_w

    def _avg_component(self, deltas: torch.Tensor, base_w: torch.Tensor) -> torch.Tensor:
        """
        AVG: encourage mean(delta) to be positive (or above margin).
        We implement as a loss (to minimize): softplus(-(mean_delta - margin)).
        """
        wsum = base_w.sum().clamp_min(self.eps)
        mean_delta = (base_w * deltas).sum() / wsum
        return F.softplus(-(mean_delta - self.margin))

    def _neg_component(self, deltas: torch.Tensor, base_w: torch.Tensor) -> torch.Tensor:
        """
        NEG: penalize magnitude of violations (delta below margin).
        We use relu(margin - delta) as negative mass proxy.
        """
        neg = F.relu(self.margin - deltas)  # 0 if delta>=margin, positive otherwise
        wsum = base_w.sum().clamp_min(self.eps)
        return (base_w * neg).sum() / wsum

    def _viol_component(self, deltas: torch.Tensor, viol_w: torch.Tensor) -> torch.Tensor:
        """
        VIOL: penalize frequency/degree of violations, distance-weighted.
        Use softplus(margin - delta) (logistic hinge), weighted by |y_pos - y_neg|.
        """
        viol_loss = F.softplus(self.margin - deltas)
        wsum = viol_w.sum().clamp_min(self.eps)
        return (viol_w * viol_loss).sum() / wsum

    def _min_component(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        MIN: penalize worst-case delta using soft-min.
          softmin(delta) = -(1/beta) logsumexp(-beta*delta)
        We then penalize if softmin < margin.
        """
        beta = float(self.beta_softmin)
        softmin = -(1.0 / beta) * torch.logsumexp(-beta * deltas, dim=0)
        return F.softplus(-(softmin - self.margin))

    # -------------
    # Forward
    # -------------
    def forward(self, logits, targets, clusters_ids, sample_weight=None):
        """
        logits:       (N, C)
        targets:      (N,) int classes (0..C-1) or ignore_index
        clusters_ids:  (N,) int cluster id (department/zone)
        sample_weight:(N,) optional per-sample weights
        """
        if logits.dim() != 2:
            raise ValueError(f"logits doit etre (N,C). Recu: {tuple(logits.shape)}")
        if targets.dim() != 1 or clusters_ids.dim() != 1:
            raise ValueError("targets et clusters_ids doivent etre des tenseurs 1D (N,).")
        if targets.size(0) != logits.size(0) or clusters_ids.size(0) != logits.size(0):
            raise ValueError("logits, targets, clusters_ids doivent avoir le meme N.")

        device = logits.device

        # valid mask
        valid = targets != self.ignore_index
        if valid.sum() < 2:
            return logits.new_tensor(0.0)

        logits_v = logits[valid]
        y_v = targets[valid]
        c_v = clusters_ids[valid]
        sw_v = sample_weight[valid].to(device) if sample_weight is not None else None

        # expected score
        s_v = self._expected_score(logits_v)  # (Nv,)

        total_avg = logits.new_tensor(0.0)
        total_min = logits.new_tensor(0.0)
        total_neg = logits.new_tensor(0.0)
        total_viol = logits.new_tensor(0.0)

        # weights for aggregating cluster-level components (use sum of base weights)
        total_w = logits.new_tensor(0.0)

        for d in torch.unique(c_v):
            idx = torch.nonzero(c_v == d, as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue

            s_d = s_v[idx]
            y_d = y_v[idx]
            sw_d = sw_v[idx] if sw_v is not None else None

            deltas, viol_w, base_w = self._ordered_deltas_and_weights(s_d, y_d, sw_d, device)
            if deltas is None or deltas.numel() == 0:
                continue

            # cluster aggregation weight
            w_cluster = base_w.sum().clamp_min(self.eps)

            # components (each returns a scalar)
            avg_c = self._avg_component(deltas, base_w)
            neg_c = self._neg_component(deltas, base_w)
            viol_c = self._viol_component(deltas, viol_w)
            min_c = self._min_component(deltas)

            total_avg = total_avg + avg_c * w_cluster
            total_neg = total_neg + neg_c * w_cluster
            total_viol = total_viol + viol_c * w_cluster
            total_min = total_min + min_c * w_cluster
            total_w = total_w + w_cluster

        if total_w.abs() < self.eps:
            return logits.new_tensor(0.0)
        
        # aggregate across clusters
        avg_loss = total_avg / total_w
        neg_loss = total_neg / total_w
        viol_loss = total_viol / total_w
        min_loss = total_min / total_w
        
        # combine (all weights = 1.0)
        mono_loss = (
            self.w_avg * avg_loss
            + self.w_min * min_loss
            + self.w_neg * neg_loss
            + self.w_viol * viol_loss
        )
        
        if isinstance(self.ce_loss, FocalLossAndWKLoss):
            ce_loss = self.ce_loss(logits, targets)['total_loss']
        else:
            ce_loss = self.ce_loss(logits, targets)
            
        loss = ce_loss + 0.75 * mono_loss
        
        if self.reduction == "sum":
            # already scalar; keep for API symmetry
            return loss
        elif self.reduction == "mean":
            return loss
        else:
            # not well-defined for a scalar; return scalar
            return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalMonotonicLoss(nn.Module):
    """
    Differentiable surrogate of the paper's transition-based SCORE_k.

    Implements:
      - mu(s) = E[y | S=s] via soft assignment
      - transitions P_k
      - soft versions of MIN / MED / NEG / VIOL
      - SCORE_k = MED_k + MIN_k - NEG_k * (1 + VIOL_k)

    References:
      Eq(3–5), P_k definition, coverage logic, median preference,
      and evaluation against continuous outcome. 
    """

    def __init__(
        self,
        num_classes=5,
        beta_softmin=10.0,
        t_violation=0.1,
        min_mass=5.0,       # analogue du "<5 occurrences" pour coverage :contentReference[oaicite:4]{index=4}
        gamma_cov=1.0,
        median_iters=8,
        median_lr=0.2,
        wk=None,
        eps=1e-8,
        id=None,

    ):
        super().__init__()
        self.C = num_classes
        self.beta = beta_softmin
        self.t = t_violation
        self.min_mass = float(min_mass)
        self.gamma_cov = float(gamma_cov)
        self.median_iters = int(median_iters)
        self.median_lr = float(median_lr)
        self.eps = eps
        self.id = id

        # P_k construction
        self.P = self._build_Pk(self.C)

        if wk is None:
            self.wk = {k: 1.0 for k in range(1, num_classes)}
        else:
            self.wk = wk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_Pk(self, C):
        P = {}
        for k in range(1, C):
            pairs = []
            for a in range(0, C - k):
                pairs.append((a, a + k))
            P[k] = pairs
        return P

    def _softmin(self, x):
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=0)

    def _soft_violation_prob(self, deltas):
        return torch.sigmoid(-deltas / self.t).mean()

    def _soft_neg(self, deltas):
        return F.softplus(-deltas).mean()

    def _soft_median(self, deltas):
        """
        Differentiable approximation of median:
            argmin_m mean sqrt((d - m)^2 + eps)
        """
        m = deltas.detach().median()
        m = m.to(deltas.dtype).to(deltas.device)
        m = m.requires_grad_(True)

        for _ in range(self.median_iters):
            loss = torch.sqrt((deltas - m) ** 2 + self.eps).mean()
            (g,) = torch.autograd.grad(loss, m, create_graph=True)
            m = m - self.median_lr * g

        return m

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, logits, y_cont, clusters_ids):
        device = logits.device
        probs = F.softmax(logits, dim=1)  # (N,C)

        total_loss = logits.new_tensor(0.0)
        total_w = logits.new_tensor(0.0)

        for d in torch.unique(clusters_ids):
            idx = torch.nonzero(clusters_ids == d, as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue

            p = probs[idx]      # (nd,C)
            y = y_cont[idx]     # (nd,)

            # Soft counts (coverage idea) :contentReference[oaicite:5]{index=5}
            counts = p.sum(dim=0)  # (C,)
            gate = torch.sigmoid(self.gamma_cov * (counts - self.min_mass))

            # mu(s) = E[y | S=s] :contentReference[oaicite:6]{index=6}
            denom = counts.clamp_min(self.eps)
            mu = (p * y.unsqueeze(1)).sum(dim=0) / denom  # (C,)

            cluster_score = logits.new_tensor(0.0)
            cluster_wk = logits.new_tensor(0.0)

            for k, pairs in self.P.items():
                deltas = []
                w_pairs = []

                for (a, b) in pairs:
                    delta = mu[b] - mu[a]
                    w_ab = gate[a] * gate[b]
                    deltas.append(delta)
                    w_pairs.append(w_ab)

                deltas = torch.stack(deltas)
                w_pairs = torch.stack(w_pairs)

                wsum = w_pairs.sum().clamp_min(self.eps)

                MINk = self._softmin(deltas)
                MEDk = self._soft_median(deltas)

                VIOLk = (w_pairs * torch.sigmoid(-deltas / self.t)).sum() / wsum
                NEGk = (w_pairs * F.softplus(-deltas)).sum() / wsum

                SCOREk = MEDk + MINk - NEGk * (1.0 + VIOLk)  # Eq(5) :contentReference[oaicite:7]{index=7}

                Lk = F.softplus(-SCOREk)

                # analogue coverage_k weighting
                support = (wsum.detach() / len(pairs))
                cluster_score += self.wk[k] * support * Lk
                cluster_wk += self.wk[k] * support

            if cluster_wk > 0:
                total_loss += cluster_score
                total_w += cluster_wk

        if total_w.abs() < self.eps:
            return logits.new_tensor(0.0)

        return total_loss / total_w

class OrdinalMonotonicLossNoCoverage(nn.Module):
    """
    Loss différentiable inspirée directement du scoring transitionnel du papier,
    MAIS sans utiliser la coverage pour pondérer / masquer la loss.

    Idée (papier) :
      - Définir les transitions P_k (ordre k) entre niveaux de risque. :contentReference[oaicite:0]{index=0}
      - Calculer Δ_{a→b} = μ(b) - μ(a) avec μ(s)=E[y | S=s]. 
      - Résumer par MIN / MED / NEG / VIOL et SCORE_k = MED + MIN - NEG*(1+VIOL). :contentReference[oaicite:2]{index=2}

    Différences vs scoring papier :
      - Ici on rend MIN/MED/VIOL diff via soft surrogates.
      - Pas de filtrage ">=5 occurrences" : toutes les transitions P_k contribuent. :contentReference[oaicite:3]{index=3}

    Entrées forward:
      logits: (N,C)
      y_cont: (N,) outcome continu (charge op.) :contentReference[oaicite:4]{index=4}
      clusters_ids: (N,) pour calculer μ(s) par cluster (proxy effets fixes zone) :contentReference[oaicite:5]{index=5}
    """

    def __init__(
        self,
        num_classes: int = 5,
        beta_softmin: float = 10.0,
        t_violation: float = 0.1,
        median_iters: int = 8,
        median_lr: float = 0.2,
        mushrinkalpha: float = 1.0,   # stabilise μ(s) si masse ~0, sans coverage
        wk=None,
        eps: float = 1e-8,
        id=None
    ):
        super().__init__()
        self.C = int(num_classes)
        self.beta = float(beta_softmin)
        self.t = float(t_violation)
        self.median_iters = int(median_iters)
        self.median_lr = float(median_lr)
        self.mushrinkalpha = float(mushrinkalpha)
        self.eps = float(eps)
        self.id = id

        self.P = self._build_Pk(self.C)
        if wk is None:
            self.wk = {k: 1.0 for k in range(1, self.C)}
        else:
            self.wk = wk

    # ----------------------
    # Helpers
    # ----------------------

    @staticmethod
    def _build_Pk(C: int):
        """
        P_k = {(a, a+k) for a=0..C-k-1}, k=1..C-1. :contentReference[oaicite:6]{index=6}
        """
        P = {}
        for k in range(1, C):
            P[k] = [(a, a + k) for a in range(0, C - k)]
        return P

    def _softmin(self, x: torch.Tensor) -> torch.Tensor:
        # soft approximation of min
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=0)

    def _soft_median(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Soft-médiane robuste, calculable en train ET en validation (même sous torch.no_grad()).
        Approx : moyenne pondérée des deltas, avec poids plus forts près du centre.
        """
        alpha = 20.0  # + grand -> plus proche d'une médiane, à tuner
        c = deltas.mean()
        w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
        return (w * deltas).sum()

    def _mu_soft(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        μ(s) = E[y | S=s] estimé via affectation soft :
          μ(s) = sum_i p_{i,s} y_i / sum_i p_{i,s}

        Version stabilisée (shrinkage) pour éviter l'instabilité quand sum p_{i,s} ~ 0 :
          μ(s) = (num + alpha * mu0) / (den + alpha)

        Le papier évite ces cas en scoring via la règle coverage; ici on veut une loss
        définie partout, donc on stabilise sans filtrer. 
        """
        counts = p.sum(dim=0)  # (C,)
        num = (p * y.unsqueeze(1)).sum(dim=0)  # (C,)

        if self.mushrinkalpha > 0:
            mu0 = y.mean()  # scalar (par cluster)
            a = self.mushrinkalpha
            mu = (num + a * mu0) / (counts + a).clamp_min(self.eps)
        else:
            mu = num / counts.clamp_min(self.eps)

        return mu  # (C,)

    # ----------------------
    # Forward
    # ----------------------

    def forward(self, logits: torch.Tensor, y_cont: torch.Tensor, clusters_ids: torch.Tensor, sample_weight=None) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits doit etre (N,C). Recu: {tuple(logits.shape)}")
        if y_cont.dim() != 1 or clusters_ids.dim() != 1:
            raise ValueError("y_cont et clusters_ids doivent etre des tenseurs 1D (N,).")
        if logits.size(0) != y_cont.size(0) or logits.size(0) != clusters_ids.size(0):
            raise ValueError("logits, y_cont, clusters_ids doivent avoir le meme N.")

        probs = F.softmax(logits, dim=1)  # (N,C)

        y_cont = y_cont.to(device=logits.device, dtype=probs.dtype)
        total_loss = logits.new_tensor(0.0)
        total_w = logits.new_tensor(0.0)

        for d in torch.unique(clusters_ids):
            idx = torch.nonzero(clusters_ids == d, as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue

            p = probs[idx]       # (nd,C)
            y = y_cont[idx]      # (nd,)

            # μ(s) par cluster (proxy effets fixes zone, sans coverage) :contentReference[oaicite:8]{index=8}
            mu = self._mu_soft(p, y)  # (C,)

            # construire SCORE_k pour k=1..C-1
            cluster_loss = logits.new_tensor(0.0)
            cluster_w = logits.new_tensor(0.0)

            for k, pairs in self.P.items():
                if not pairs:
                    continue

                deltas = torch.stack([mu[b] - mu[a] for (a, b) in pairs], dim=0)  # (|P_k|,)

                # Surrogates diff des métriques (papier) :contentReference[oaicite:9]{index=9}
                MINk = self._softmin(deltas)
                MEDk = self._soft_median(deltas)
                VIOLk = torch.sigmoid(-deltas / self.t).mean()     # approx I[Δ<0]
                NEGk = F.softplus(-deltas).mean()                  # approx mean max(0,-Δ)

                SCOREk = MEDk + MINk - NEGk * (1.0 + VIOLk)        # Eq(5) :contentReference[oaicite:10]{index=10}

                # On veut SCOREk élevé/positif -> loss pénalise SCOREk négatif
                Lk = F.softplus(-SCOREk)

                w = float(self.wk.get(k, 1.0))
                cluster_loss = cluster_loss + w * Lk
                cluster_w = cluster_w + w

            if cluster_w > 0:
                total_loss = total_loss + cluster_loss
                total_w = total_w + cluster_w

        if total_w.abs() < self.eps:
            return logits.new_tensor(0.0)

        return total_loss / total_w

class OrdinalMonotonicLossNoCoverageWithGains(nn.Module):
    """
    Ordinal monotonic loss (transition-based) with per-cluster minimum gains (margins),
    with fallback to "similar clusters" pooling when a cluster cannot define valid bins.

    Key:
      - Class 0: y <= 0
      - Positive classes 1..C-1: bins on y_pos = y[y>0], with (C-2) positive thresholds.
      - Thresholds are "adaptive": if quantiles are equal (discrete y), we push q upward
        until threshold strictly increases; cap at q_max=0.99.
      - Gains are integrated as margin inside deltas:
            delta_{a->b} = (mu[b] - mu[a]) - sum_{j=a..b-1} g[j]
    """

    def __init__(
        self,
        num_classes=5,
        betasoftmin=10.0,
        tviolation=0.01,
        mushrinkalpha=1.0,
        eps=1e-8,
        wk=None,
        # positive quantile targets (length C-2). For C=5 -> 3 thresholds.
        quantileedges=(0.5, 0.8, 0.95),
        minbinn=3,
        gainsalpha=1.0,
        gainsfloorfrac=0.1,
        enforcegainmonotone=True,
        id=None,
        enablelogs=True,
        lambdamu0 = 0.0,
        lambdaentropy = 0.0,
        wmed=1.0,
        wmin=1.0,
        wneg=1.0,
        wviol=1.0,
        lambdadir=0.0, # Default as requested in "Valeurs de départ"
        diralpha=1.05,
        lambdace=0.0,
        class_weights=None,
        lambdagl=0.0,
        cetype='crossentropy'  # 'crossentropy' or 'focal'
    ):
        super().__init__()
        self.lambdadir = float(lambdadir)
        self.diralpha = float(diralpha)
        self.lambdace = float(lambdace)
        self.cetype = str(cetype).lower()

        # CE Loss Setup - choose between focal and cross-entropy
        if self.cetype == 'focal':
            if class_weights is not None:
                if not torch.is_tensor(class_weights):
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)
                self.register_buffer('class_weights', class_weights)
                self.ce_loss = FocalLoss(gamma=2.0, alpha=self.class_weights, reduction='mean')
            else:
                self.ce_loss = FocalLoss(gamma=2.0, alpha=1.0, reduction='mean')
        elif self.cetype == 'crossentropy':
            self.ce_loss = WeightedCrossEntropyLoss(num_classes=num_classes)
        else:
            raise ValueError(f"cetype must be 'focal' or 'crossentropy', got '{cetype}'")

        self.id = id
        self.C = int(num_classes)
        if self.C < 2:
            raise ValueError("num_classes doit être >= 2.")
        if len(quantileedges) != (self.C - 2):
            raise ValueError("quantileedges (positifs) doit avoir longueur C-2=%d (reçu %d)"
                             % (self.C - 2, len(quantileedges)))

        self.beta = float(betasoftmin)
        self.t = float(tviolation)
        self.mushrinkalpha = float(mushrinkalpha)
        self.eps = float(eps)

        self.quantileedges = tuple(float(x) for x in quantileedges)
        self.minbinn = int(minbinn)
        self.gainsalpha = float(gainsalpha)
        self.gainsfloorfrac = float(gainsfloorfrac)
        self.enforcegainmonotone = bool(enforcegainmonotone)

        self.P = self._build_Pk(self.C)
        self.wk = {k: 1.0 for k in range(1, self.C)} if wk is None else wk
        
        self.enablelogs = bool(enablelogs)
        self.lambdamu0 = lambdamu0
        self.lambdaentropy = lambdaentropy
        # cluster_id -> torch.FloatTensor(C-1)
        self.gain_k = {}
        self.register_buffer("global_gains", torch.zeros(self.C - 1, dtype=torch.float32))
        self.register_buffer("global_scale", torch.tensor(1.0, dtype=torch.float32))

        # Component weights
        self.wmed = float(wmed)
        self.wmin = float(wmin)
        self.wneg = float(wneg)
        self.wviol = float(wviol)

        self.lambdagl = float(lambdagl)

        self.call_preprocess = False

        #self._log("Ordinal Loss Config:", self.get_config())

    def get_config(self):
        return {
            "numclasses": self.C,
            "betasoftmin": self.beta,
            "tviolation": self.t,
            "mushrinkalpha": self.mushrinkalpha,
            "eps": self.eps,
            "quantileedges": self.quantileedges,
            "minbinn": self.minbinn,
            "gainsalpha": self.gainsalpha,
            "gainsfloorfrac": self.gainsfloorfrac,
            "enforcegainmonotone": self.enforcegainmonotone,
            "enablelogs": self.enablelogs,
            "lambdamu0": self.lambdamu0,
            "lambdaentropy": self.lambdaentropy,
            "wmed": self.wmed,
            "wmin": self.wmin,
            "wneg": self.wneg,
            "wviol": self.wviol,

            "id": self.id,
            "lambdagl": self.lambdagl,
            "lambdadir": self.lambdadir,
            "diralpha": self.diralpha
        }

    # ------------------------------------------------------------
    # Utilities: logs
    # ------------------------------------------------------------
    def _log(self, *args):
        if self.enablelogs:
            print(*args)

    # ------------------------------------------------------------
    # Adaptive thresholds on y_pos
    # ------------------------------------------------------------
    def _adaptive_positive_thresholds(self, y_pos, qe_pos):
        """
        qe_pos: list/tuple of target quantiles (len = C-2).
        For each target, compute threshold. If threshold does not strictly increase
        vs previous threshold, push quantile upward by q_step until it does (cap q_max).

        Returns:
          thresholds: np.float32 (C-2,)
          used_q: np.float32 (C-2,)
          forced_flat: bool (True if we had to freeze remaining thresholds)
        """
        q_max = 0.99
        q_step = 0.05
        tol = 0.0

        thresholds = []
        used_q = []
        last_t = None
        forced_flat = False

        for q0 in qe_pos:
            q_try = float(q0)
            if q_try <= 0.0 or q_try >= 1.0:
                raise ValueError("quantileedges doit être dans ]0,1[ (reçu %s)" % str(qe_pos))

            if forced_flat:
                thresholds.append(float(last_t))
                used_q.append(float(q_max))
                continue

            t = float(np.quantile(y_pos, q_try))

            if last_t is not None and t <= last_t + tol:
                q = q_try
                t_new = t
                while q < q_max - 1e-12:
                    q = min(q_max, q + q_step)
                    t_new = float(np.quantile(y_pos, q))
                    if t_new > last_t + tol:
                        break

                if t_new > last_t + tol:
                    q_try = q
                    t = t_new
                else:
                    forced_flat = True
                    q_try = q_max
                    t = float(last_t)

            thresholds.append(float(t))
            used_q.append(float(q_try))
            last_t = float(t)

        return (
            np.array(thresholds, dtype=np.float32),
            np.array(used_q, dtype=np.float32),
            bool(forced_flat),
        )
    
    def _dirichlet_barrier(self, pi):
        """
        R(pi) = -(diralpha - 1) * sum_k log(pi_k + eps)
        """
        eps = 1e-9
        pi_safe = pi.clamp(min=eps)
        return -(self.diralpha - 1.0) * pi_safe.log().sum()

    def _compute_single_loss_component(
        self, 
        probs, 
        y, 
        sw, 
        gains, 
        scale, 
        wk_dict, 
        lam0, 
        logits_factory, # tensor to create new tensors on same device
        thresholds=None,
        lambdace=None,
        lambdaentropy=None,
        lambdadir=None
    ):
        """
        Helper to compute loss components (Trans, Mu0) + stats for a given subset (cluster or global).
        Returns:
            loss_val: scalar tensor
            w_val: scalar tensor
            stats_dict: dict with various metrics (mu, pi, entropy, deltas, etc.)
        """
        # Resolve lambdas
        lambdace = lambdace if lambdace is not None else self.lambdace
        lambdaentropy = lambdaentropy if lambdaentropy is not None else self.lambdaentropy
        lambdadir = lambdadir if lambdadir is not None else self.lambdadir

        # --- 5) Masse de proba pi_s ---
        if sw is not None:
            p_weighted = probs * sw.unsqueeze(1)
            pi_s = p_weighted.sum(dim=0) / sw.sum().clamp_min(self.eps)
        else:
            pi_s = probs.mean(dim=0)
        
        mu = self._mu_soft(probs, y, sw=sw)  # (C,)
        
        # Entropy of pi_s
        pi_log_pi = pi_s * torch.log(pi_s.clamp_min(1e-9))
        entropy_pi = -pi_log_pi.sum()

        cluster_loss = logits_factory.new_tensor(0.0)
        cluster_w = logits_factory.new_tensor(0.0)
        
        # --- 8) Terme mu(0) ---
        mu0_val = mu[0]
        mu0_term = F.softplus(mu0_val / (scale + self.eps))

        stats_loss_trans = logits_factory.new_tensor(0.0)
        stats_w_trans = logits_factory.new_tensor(0.0)
        
        Lk_w_batch = {}
        SCORE_k_batch = {}
        VIOL_k_batch = {}
        NEG_k_batch = {}
        
        deltas_batch = {}

        # SCORE_k with margined deltas
        for k, pairs in self.P.items():
            if not pairs:
                continue

            raw = torch.stack([mu[b] - mu[a] for (a, b) in pairs], dim=0)
            margins = torch.stack([gains[a:b].sum() for (a, b) in pairs], dim=0)
            raw = raw / (scale + self.eps)
            deltas = raw - margins
            
            deltas_batch[k] = deltas.detach().cpu().numpy()

            # Surrogates
            MINk = self._softmin(deltas)
            MEDk = self._soft_median(deltas)
            VIOLk = torch.sigmoid(-deltas / self.t).mean()
            NEGk = F.softplus(-deltas).mean()
            
            # === MULTI-CRITERIA INEQUALITY LOSS (ReLU) ===
            # We want MED > 0, MIN > 0, and minimizing magnitude of violations
            # Use ReLU to penalize ONLY when criteria are violated (< 0)
            
            # Criterion 1: Median Penalty (Quadratic)
            loss_med = F.softplus(-MEDk)
            
            # Criterion 2: Minimum Penalty (Quadratic)
            loss_min = F.softplus(-MINk)
            
            # Criterion 3: Magnitude Penalty (Mean of squared violations)
            loss_neg = (F.softplus(-deltas)).mean()
            
            # Weighted Combination
            Lk = (self.wmed * loss_med +
                  self.wmin * loss_min + 
                  self.wneg * loss_neg)

            # SCOREk for logging (kept similar to before for continuity)
            SCOREk = (self.wmed * MEDk + self.wmin * MINk) - self.wneg * NEGk * (1.0 + self.wviol * VIOLk)

            w = float(wk_dict.get(k, 1.0))
            cluster_loss = cluster_loss + w * Lk
            cluster_w = cluster_w + w
            
            stats_loss_trans = stats_loss_trans + w * Lk
            stats_w_trans = stats_w_trans + w

            Lk_w_batch[k] = (w * Lk).detach().item()
            SCORE_k_batch[k] = SCOREk.detach().item()
            VIOL_k_batch[k] = VIOLk.detach().item()
            NEG_k_batch[k] = NEGk.detach().item()

        # Normalize transition loss
        if cluster_w > 0:
            cluster_loss = cluster_loss / cluster_w.clamp_min(self.eps)
            # cluster_w becomes "normalization factor for this cluster" -> 1.0 effectively
            # but we return 1.0 later to tell forward to just average.
        else:
            cluster_loss = logits_factory.new_tensor(0.0)

        # Add Mu0
        if lam0 > 0:
             cluster_loss = cluster_loss + lam0 * mu0_term
        
        # Add CE
        ce_mean = logits_factory.new_tensor(0.0)
        if lambdace > 0:
            logit_tens = logits_factory

            if thresholds is None:
                y_disc = y.long()
            else:
                y_disc = self._discretize_y(y, thresholds)

            ce_mean = self.ce_loss(logit_tens, y_disc, sample_weight=sw)

            ce_mean = ce_mean
            cluster_loss = cluster_loss + lambdace * ce_mean
            
        # Add Entropy
        if lambdaentropy > 0:
            cluster_loss = cluster_loss - lambdaentropy * entropy_pi
            
        # Add Dirichlet Regularization (moved to cluster level)
        dir_reg = logits_factory.new_tensor(0.0)
        if lambdadir > 0:
            dir_reg = self._dirichlet_barrier(pi_s)
            cluster_loss = cluster_loss + lambdadir * dir_reg

        stats_dict = {
            'loss_total': cluster_loss.detach().item(),
            'loss_trans': (stats_loss_trans / stats_w_trans.clamp_min(self.eps)).detach().item() if stats_w_trans > 0 else 0.0,
            'mu0_term': (lam0 * mu0_term).detach().item(),
            'mu': mu.detach().cpu().numpy(),
            'pi': pi_s.detach().cpu().numpy(),
            'entropy_pi': entropy_pi.item(),
            'entropy_weighted': (-lambdaentropy * entropy_pi).detach().item() if lambdaentropy > 0 else 0.0,
            'dirichlet_reg': dir_reg.detach().item() if lambdadir > 0 else 0.0,
            'dirichlet_weighted': (lambdadir * dir_reg).detach().item() if lambdadir > 0 else 0.0,
            'ce_loss': ce_mean.detach().item() if lambdace > 0 else 0.0,
            'ce_weighted': (lambdace * ce_mean).detach().item() if lambdace > 0 else 0.0,
            'deltas': deltas_batch,
            'gains': gains.detach().cpu().numpy(),
            'scale': scale.detach().cpu().numpy(),
            'Lk_weighted': Lk_w_batch,
            'SCORE_k': SCORE_k_batch,
            'VIOL_k': VIOL_k_batch,
            'NEG_k': NEG_k_batch,
            'mu0_stats': {'mu0': mu0_val.detach().item()}
        }
        
        # Stats for scales and margins
        # scale can be scalar or tensor.
        if scale.numel() > 1:
            scale_min = scale.min().item()
            scale_mean = scale.mean().item()
            scale_max = scale.max().item()
        else:
            v_sc = scale.item()
            scale_min = v_sc
            scale_mean = v_sc
            scale_max = v_sc
            
        # raw diffs = diffs_batch (already computed: mu[1:] - mu[:-1])
        # diffs_batch is a list in current code? No, let's see where it comes from.
        # It's calculated inside the loop over k. We want the mean of adjacent diffs.
        # mu is [num_classes] shape.
        diffs_mu = mu[1:] - mu[:-1]
        raw_mean = diffs_mu.mean().item()
        
        # scaled raw
        # If scale is scalar, it's raw_mean / scale. If vector?
        if scale.numel() > 1:
             # Just an approximation if scale varies per class (not the case usually)
             scaled_raw_mean = (diffs_mu / (scale.mean() + 1e-9)).mean().item()
        else:
             scaled_raw_mean = raw_mean / (scale.item() + 1e-9)
             
        # margins (gains)
        margin_mean = gains.mean().item()
        
        stats_dict.update({
             'scale_min': scale_min,
             'scale_mean': scale_mean,
             'scale_max': scale_max,
             'diff_raw_mean': raw_mean,
             'diff_scaled_mean': scaled_raw_mean,
             'margin_mean': margin_mean
        })
        
        return cluster_loss, 1.0, stats_dict

    def _auto_wk_from_gains_np(self, g_np: np.ndarray, eps: float = 1e-8):
        """
        g_np: (C-1,) gains (global ou cluster). Retourne dict wk[k] pour k=1..C-1
        wk[k] = 1 / ( median_{(a,b) in P_k} sum_{j=a..b-1} g[j] + eps )
        """
        C = self.C
        wk = {}
        for k in range(1, C):
            margins_k = []
            for a in range(0, C - k):
                b = a + k
                margins_k.append(float(np.sum(g_np[a:b])))
            mk = float(np.median(margins_k))  # médiane robuste
            
            # Normalization
            wk[k] = (1.0 / (mk + eps)) * self.wk.get(k, 1.0)
            
            # Simple weight
            #wk[k] = self.wk.get(k, 1.0)
        return wk

    def _get_cluster_wk(self, cluster_id):
        # fallback: cluster -> global -> uniform
        if hasattr(self, "wk_k") and (cluster_id in self.wk_k):
            return self.wk_k[cluster_id]
        if hasattr(self, "wk_global"):
            return self.wk_global
        return self.wk

    # -------------------------------- ----------------------------
    # Build bins: 0 + (C-1) positive bins
    # ------------------------------------------------------------
    def _compute_bins_and_ok_positive_quantiles(
        self,
        y_sub,
        qe_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=3,
        tag=""
    ):
        """
        Returns (bin_means, ok, qs_pos, used_q, forced_flat, counts)

        - bin 0: y <= 0
        - bins 1..C-1: defined on y_pos with thresholds qs_pos (len C-2)
          bin1: (0, q1]
          ...
          bin(C-1): (q_last, +inf)
        """
        y_sub = y_sub[np.isfinite(y_sub)]
        if y_sub.size == 0:
            return None, False, None, None, False, None

        y0 = y_sub[y_sub <= 0]
        y_pos = y_sub[y_sub > 0]
        if y_pos.size == 0:
            return None, False, None, None, False, None

        qs_pos, used_q, forced_flat = self._adaptive_positive_thresholds(y_pos, qe_pos)

        # bins
        bins = []
        # bin0
        bins.append(y0)

        # positive bins
        # bin1: y_pos <= q1
        q1 = qs_pos[0]
        bins.append(y_pos[y_pos <= q1])

        # middle: (q_{i-1}, q_i]
        for i in range(1, len(qs_pos)):
            lo = qs_pos[i - 1]
            hi = qs_pos[i]
            bins.append(y_pos[(y_pos > lo) & (y_pos <= hi)])

        # last: > q_last
        q_last = qs_pos[-1]
        bins.append(y_pos[y_pos > q_last])
        
        # means & counts
        bin_means = np.array([float(np.mean(b)) if b.size > 0 else np.nan for b in bins], dtype=np.float32)
        counts = np.array([int(b.size) for b in bins], dtype=np.int32)
        pos_counts = counts.copy()
        pos_counts[0] = 0

        # validity
        ok = True

        # thresholds must be strictly increasing (otherwise bins degenerate)
        if not np.all(np.diff(qs_pos) > 0):
            ok = False

        # positive bins must have enough samples
        if ok and not np.all(counts[1:] >= int(minB)):
            ok = False

        if ok and require_positive_bins:
            if not np.all(pos_counts[1:] >= int(min_pos_per_bin)):
                ok = False

        self._log(
            f"[BINS {tag}] means={bin_means} counts={counts} ok={ok} "
            f"qs_pos={qs_pos} used_q={used_q} forced_flat={forced_flat}"
        )

        return bin_means, bool(ok), qs_pos, used_q, forced_flat, counts

    # ------------------------------------------------------------
    # Gains from bin_means (adjacent)
    # ------------------------------------------------------------
    def _gains_from_bin_means(self, bin_means, y_ref, tag=""):
        diffs = np.diff(bin_means)
        diffs = np.maximum(diffs, 0.0)

        y_ref = y_ref[np.isfinite(y_ref)]
        y_pos = y_ref[y_ref > 0]

        if y_pos.size >= 10:
            q50 = float(np.quantile(y_pos, 0.5))
            q95 = float(np.quantile(y_pos, 0.95))
            #spread = q95 - q50
            spread = 1.0 # No normalization
        else:
            q50 = float(np.quantile(y_pos, 0.5)) if y_pos.size > 0 else 0.0
            q95 = float(np.quantile(y_pos, 0.95)) if y_pos.size > 0 else 0.0
            #spread = float(np.std(y_pos)) if y_pos.size > 1 else 0.0
            spread = 1.0 # No normalization

        #spread = max(spread, 1e-6)
        floor = max(0.0, self.gainsfloorfrac)
        
        # Normailzation
        #base = self.gainsalpha * (diffs / spread)
        
        base = self.gainsalpha * diffs
        gains = np.maximum(base, floor).astype(np.float32)

        if self.enforcegainmonotone:
            gains = np.maximum.accumulate(gains).astype(np.float32)

        self._log(
            f"[GAINS {tag}] diffs={diffs} base={base} q50_pos={q50} q95_pos={q95} "
            f"spread={spread} floor={floor} final={gains}"
        )
        return gains, float(spread)

    # ------------------------------------------------------------
    # Preprocess with similar_cluster_ids fallback
    # ------------------------------------------------------------
    def _preprocess(
        self,
        y_cont,
        clusters_ids,
        similar_cluster_ids=None,   # same length as clusters_ids, defines similarity groups
        quantileedges=None,
        minbinn=None,
        require_positive_bins=True,
        min_pos_per_bin=3,
        enforcegainmonotone=None,
    ):

        self.call_preprocess = True
        y = np.asarray(y_cont)
        c = np.asarray(clusters_ids)

        if y.ndim != 1 or c.ndim != 1 or len(y) != len(c):
            raise ValueError("y_cont et clusters_ids doivent être 1D et de même longueur.")

        if similar_cluster_ids is not None:
            s = np.asarray(similar_cluster_ids)
            if s.ndim != 1 or len(s) != len(c):
                raise ValueError("similar_cluster_ids doit être 1D et de même longueur que clusters_ids.")
        else:
            s = None

        qe_pos = self.quantileedges if quantileedges is None else tuple(float(x) for x in quantileedges)
        if len(qe_pos) != (self.C - 2):
            raise ValueError("quantileedges (positifs) doit avoir longueur C-2=%d" % (self.C - 2))

        minB = self.minbinn if minbinn is None else int(minbinn)
        if enforcegainmonotone is not None:
            self.enforcegainmonotone = bool(enforcegainmonotone)

        # ---- Build group maps (similarity pooling) ----
        idx_by_cluster = {}
        unique_clusters = np.unique(c)
        for cl in unique_clusters:
            idx_by_cluster[cl] = np.where(c == cl)[0]

        idx_by_group = None
        cluster_group = None
        if s is not None:
            idx_by_group = {}
            for g in np.unique(s):
                idx_by_group[g] = np.where(s == g)[0]

            cluster_group = {}
            for cl in unique_clusters:
                idx = idx_by_cluster[cl]
                vals, cnts = np.unique(s[idx], return_counts=True)
                cluster_group[cl] = vals[np.argmax(cnts)] if vals.size > 0 else None
                
        # ---- GLOBAL ----
        self._log("==== GLOBAL GAINS (bins->gains) ====")
        bm_g, ok_g, qs_g, usedq_g, flat_g, cnt_g = self._compute_bins_and_ok_positive_quantiles(
            y, qe_pos, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin, tag="GLOBAL"
        )
        if bm_g is not None and ok_g:
            global_g, global_spread = self._gains_from_bin_means(bm_g, y, tag="GLOBAL")
            self.global_scale = torch.tensor(global_spread, dtype=torch.float32, device=self.global_gains.device)
            self.register_buffer('global_thresholds', torch.tensor(qs_g, dtype=torch.float32))
        else:
            raise ValueError('Can t calcule bin means')
            # robust fallback
            yfin = y[np.isfinite(y)]
            scale = float(np.std(yfin)) if yfin.size > 1 else 1.0
            global_g = np.full(self.C - 1, 0.05 * scale, dtype=np.float32)
            self._log("[GLOBAL] fallback scale-based gains:", global_g)

        self.global_gains = torch.tensor(global_g, dtype=torch.float32, device=self.global_gains.device)
        self._log("[GLOBAL GAINS] tensor:", self.global_gains)

        # ---- PER CLUSTER ----
        self.gain_k = {}
        self.scale_k = {}
        self.thresholds_k = {}
        self._log("==== CLUSTER GAINS ====")

        for cl in unique_clusters:
            idx = idx_by_cluster[cl]
            y_cl = y[idx]

            self._log(f"\n--- cluster {cl} (n={len(idx)}) ---")
            bm, ok, qs, usedq, flat, cnt = self._compute_bins_and_ok_positive_quantiles(
                y_cl, qe_pos, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin,
                tag="CL_%s" % str(cl)
            )
            
            if bm is not None and ok:
                g_cl, spread_cl = self._gains_from_bin_means(bm, y_cl, tag="CL_%s" % str(cl))
            else:
                pooled = False
                if cluster_group is not None and idx_by_group is not None:
                    g_id = cluster_group.get(cl, None)
                    if g_id is not None and g_id in idx_by_group:
                        idx_pool = idx_by_group[g_id]
                        y_pool = y[idx_pool]
                        self._log(f"[CL {cl}] pool by similar group {g_id} (n_pool={len(idx_pool)})")

                        bm2, ok2, qs2, usedq2, flat2, cnt2 = self._compute_bins_and_ok_positive_quantiles(
                            y_pool, qe_pos, minB,
                            require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin,
                            tag="POOL_G%s" % str(g_id)
                        )
                        if bm2 is not None and ok2:
                            g_cl, spread_cl = self._gains_from_bin_means(bm2, y_pool, tag="POOL_G%s" % str(g_id))
                            pooled = True
                        else:
                            self._log(f"[CL {cl}] pooling failed -> fallback global gains")
                            g_cl = global_g
                    else:
                        self._log(f"[CL {cl}] no valid similar group -> fallback global gains")
                        g_cl = global_g
                else:
                    self._log(f"[CL {cl}] no similar_cluster_ids -> fallback global gains")
                    g_cl = global_g

                if pooled:
                    self._log(f"[CL {cl}] used pooled gains")
                else:
                    self._log(f"[CL {cl}] used global gains")

            if self.enforcegainmonotone:
                g_cl = np.maximum.accumulate(np.asarray(g_cl, dtype=np.float32)).astype(np.float32)

            self.gain_k[cl] = torch.tensor(g_cl, dtype=torch.float32, device=self.global_gains.device)
            self.scale_k[cl] = torch.tensor(spread_cl, dtype=torch.float32, device=self.global_gains.device)
            
            # Store per-cluster thresholds for CE discretization
            # If pooling/fallback used, we might not have specific qs.
            # If ok is True, we have qs.
            # If pooled (ok2 is True), we have qs2.
            # Else fallback global -> global_thresholds.
            
            if ok:
                 res_qs = qs
            elif pooled and ok2:
                 res_qs = qs2
            else:
                 res_qs = qs_g # Fallback to global thresholds
            
            if res_qs is not None:
                self.thresholds_k[cl] = torch.tensor(res_qs, dtype=torch.float32, device=self.global_gains.device)
            else:
                # Should correspond to global fallback if qs_g is available
                # If even qs_g is None (which raises ValueError above anyway), we are in trouble.
                self.thresholds_k[cl] = self.global_thresholds

            self._log(f"[CLUSTER GAINS] {cl} -> {g_cl}")
        
        # ---- AUTO wk (global + per-cluster) ----
        self.wk_global = self._auto_wk_from_gains_np(
            self.global_gains.detach().cpu().numpy(),
            eps=self.eps
        )

        self.wk_k = {}
        for cl, g_t in self.gain_k.items():
            self.wk_k[cl] = self._auto_wk_from_gains_np(
                g_t.detach().cpu().numpy(),
                eps=self.eps
            )

        print("[AUTO wk] global:", {k: round(v, 6) for k, v in self.wk_global.items()})
        print("[AUTO wk] per-cluster:", {cl: {k: round(v,6) for k,v in wk.items()} for cl, wk in self.wk_k.items()})

    # ------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------
    @staticmethod
    def _build_Pk(C):
        P = {}
        for k in range(1, C):
            P[k] = [(a, a + k) for a in range(0, C - k)]
        #print("[TRANSITION AND K]:", P)
        return P

    def _get_cluster_scale(self, cluster_id, device, dtype):
        if hasattr(self, "scale_k") and (cluster_id in self.scale_k):
            s = self.scale_k[cluster_id]
        else:
            s = self.global_scale
        return s.to(device=device, dtype=dtype)

    def _softmin(self, x):
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=0)

    def _soft_median(self, deltas):
        alpha = 20.0
        c = deltas.mean()
        w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
        return (w * deltas).sum()

    def _mu_soft(self, p, y, sw=None):
        # ensure y float
        if not torch.is_floating_point(y):
            y = y.to(dtype=p.dtype)

        if sw is not None:
            sw = sw.to(device=p.device, dtype=p.dtype).clamp_min(self.eps)
            p_eff = p * sw.unsqueeze(1)
            counts = p_eff.sum(dim=0)
            num = (p_eff * y.unsqueeze(1)).sum(dim=0)
        else:
            counts = p.sum(dim=0)
            num = (p * y.unsqueeze(1)).sum(dim=0)

        if self.mushrinkalpha > 0:
            mu0 = y.mean()
            a = self.mushrinkalpha
            mu = (num + a * mu0) / (counts + a).clamp_min(self.eps)
        else:
            mu = num / counts.clamp_min(self.eps)

        return mu  # (C,)

    def _get_cluster_gains(self, cluster_id, device, dtype):
        if hasattr(self, "gain_k") and (cluster_id in self.gain_k):
            g = self.gain_k[cluster_id]
        else:
            g = self.global_gains
        return g.to(device=device, dtype=dtype)

    def _get_cluster_thresholds(self, cluster_id, device, dtype):
        if hasattr(self, "thresholds_k") and (cluster_id in self.thresholds_k):
            t = self.thresholds_k[cluster_id]
        else:
            t = self.global_thresholds
        return t.to(device=device, dtype=dtype)

    def _discretize_y(self, y_cont, thresholds):
        # Discretize continuous y into classes based on thresholds
        # Class 0: y <= 0 (handled by y > 0 check generally)
        # Class k: thresholds[k-1] < y <= thresholds[k]
        
        # We start with 0s.
        y_disc = torch.zeros_like(y_cont, dtype=torch.long)
        
        # Positive values
        mask_pos = y_cont > 0
        if not mask_pos.any():
            return y_disc
            
        y_pos = y_cont[mask_pos]
        
        if thresholds.numel() == 0:
            # If no thresholds (C=2 case?), implies only 1 positive class? 
            # If thresholds is empty, bucketize returns 0 for all.
            # We want Class 1. so +1.
            y_disc[mask_pos] = 1
        else:
            # bucketize: 
            # right=True: bins[i-1] < x <= bins[i]
            # output index i.
            # For thresholds [t1, t2] (C=4 classes: 0, 1, 2, 3)
            # y <= 0 -> Class 0 (handled separately)
            # 0 < y <= t1 -> bucket 0 -> Class 1
            # t1 < y <= t2 -> bucket 1 -> Class 2
            # y > t2      -> bucket 2 -> Class 3
            
            buckets = torch.bucketize(y_pos, thresholds, right=True)
            y_disc[mask_pos] = buckets + 1
            
        return y_disc

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.epoch_stats = {}

    def forward(self, logits, y_cont, clusters_ids, sample_weight=None):
        assert self.call_preprocess is True, f'You have to call preprocess before forward'
        if logits.dim() != 2:
            raise ValueError("logits doit etre (N,C). Recu: %s" % (tuple(logits.shape),))
        if y_cont.dim() != 1 or clusters_ids.dim() != 1:
            raise ValueError("y_cont et clusters_ids doivent etre des tenseurs 1D (N,).")
        if logits.size(0) != y_cont.size(0) or logits.size(0) != clusters_ids.size(0):
            raise ValueError("logits, y_cont, clusters_ids doivent avoir le meme N.")

        # Init epoch_stats if not present (e.g. first forward or eval mode used without train call)
        if not hasattr(self, 'epoch_stats'):
            self.epoch_stats = {}

        probs = F.softmax(logits, dim=1)
        y_cont = y_cont.to(device=logits.device, dtype=probs.dtype)

        sw = None
        if sample_weight is not None:
            sw = sample_weight.to(device=logits.device, dtype=probs.dtype)

        total_loss = logits.new_tensor(0.0)
        total_w = logits.new_tensor(0.0)
        
        unique_clusters = torch.unique(clusters_ids)

        for d in unique_clusters:
            idx = torch.nonzero(clusters_ids == d, as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue

            p = probs[idx]
            y = y_cont[idx]
            sw_d = sw[idx] if sw is not None else None

            cl_id = d.item() if torch.is_tensor(d) else d
            # Init stats for this cluster if needed
            if cl_id not in self.epoch_stats:
                self.epoch_stats[cl_id] = {
                    'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                    'mu': [], 'pi': [], 'entropy_pi': [], 'dirichlet_reg': [], 'ce_loss': [],
                    'deltas': [], 'gains': [], 'scale': [],
                    'Lk_weighted': [], 'SCORE_k': [], 'VIOL_k': [],
                    'NEG_k': [], 'mu0_stats': [],
                }
            
            gains_adj = self._get_cluster_gains(cl_id, p.device, p.dtype)
            wk_dict = self._get_cluster_wk(cl_id)
            scale = self._get_cluster_scale(cl_id, p.device, p.dtype)
            lam0 = self.lambdamu0
            thresholds = self._get_cluster_thresholds(cl_id, p.device, p.dtype)
            
            # Slice logits for this cluster!
            l_d = logits[idx]

            c_loss, c_w, stats = self._compute_single_loss_component(
                p, y, sw_d, gains_adj, scale, wk_dict, lam0, l_d, thresholds
            )

            if c_w > 0:
                total_loss = total_loss + c_loss
                total_w = total_w + c_w
            
            # Keep track of everything (detached)
            est = self.epoch_stats[cl_id]
            for k, v in stats.items():
                if k in est:
                    est[k].append(v)

        if total_w.abs() < self.eps:
            final_loss = logits.new_tensor(0.0)
        else:
            final_loss = total_loss / total_w
            
        # --- GLOBAL LOSS COMPONENT ---
        # If more than one cluster, add global loss (on all data).
        # If only one cluster, the above loop effectively computed the global loss already 
        # (and normalized it). 
        # User request: "le cas où on en voit pas qu'un seul cluster, je veux que tu ajoutes la même loss mais calculé sur l'ensemble des logits"
        
        if self.lambdagl > 0:
            if len(unique_clusters) > 1:
                # Global setup
                # gains -> global_gains
                # scale -> global_scale
                # wk -> self.wk
                g_gains = self.global_gains.to(device=logits.device, dtype=probs.dtype)
                g_scale = self.global_scale.to(device=logits.device, dtype=probs.dtype)
                g_wk = self.wk # default / global
                g_thresholds = self.global_thresholds.to(device=logits.device, dtype=probs.dtype)
                
                # We treat the whole batch as one "global cluster"
                # Note: sample_weight is sw
                # lambdace=0.0, lambdaentropy=0.0, lambdadir=0.0 for GLOBAL component as requested
                g_loss, g_w, g_stats = self._compute_single_loss_component(
                    probs, y_cont, sw, g_gains, g_scale, g_wk, self.lambdamu0, logits, None,
                    lambdace=0.0, lambdaentropy=0.0, lambdadir=0.0
                )
                
                if g_w > 0:
                    global_loss_val = g_loss / g_w
                    final_loss = final_loss + self.lambdagl * global_loss_val
                    
                    # Option: log global stats?
                    # The user didn't explicitly ask for logging, but it helps debugging.
                    # We can store it under a special key "global" if we want, or -1.
                    if "global" not in self.epoch_stats:
                        self.epoch_stats["global"] = {
                            'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                            'mu': [], 'pi': [], 'entropy_pi': [], 'dirichlet_reg': [], 'ce_loss': [],
                            'deltas': [], 'gains': [], 'scale': [],
                            'Lk_weighted': [], 'SCORE_k': [], 'VIOL_k': [],
                            'NEG_k': [], 'mu0_stats': [],
                        }
                    est_g = self.epoch_stats["global"]
                    for k, v in g_stats.items():
                         if k in est_g:
                            est_g[k].append(v)

            elif len(unique_clusters) == 1:
                # Single cluster case: copy the stats from that cluster to "global"
                # because the user requested "tout les logs possibles par cluster et au global".
                cl_id = unique_clusters[0].item() if torch.is_tensor(unique_clusters[0]) else unique_clusters[0]
                if cl_id in self.epoch_stats:
                    src_stats = self.epoch_stats[cl_id]
                    if "global" not in self.epoch_stats:
                       self.epoch_stats["global"] = {
                            'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                            'mu': [], 'pi': [], 'entropy_pi': [], 'dirichlet_reg': [], 'ce_loss': [],
                            'deltas': [], 'gains': [], 'scale': [],
                            'Lk_weighted': [], 'SCORE_k': [], 'VIOL_k': [],
                            'NEG_k': [], 'mu0_stats': [],
                        }
                    est_g = self.epoch_stats["global"]
                    # Append last values from src_stats
                    for k in est_g.keys():
                        if k in src_stats and len(src_stats[k]) > 0:
                            est_g[k].append(src_stats[k][-1])
        
        return final_loss

    def get_attribute(self):
        """
        Called by pytorch_model_tools at the end of epoch to retrieve params/stats.
        Returns a list of tuples [('name', value)] or similar.
        Here we want to return a single object containing all our aggregation.
        """
        # Aggregate epoch stats
        aggregated = {}
        
        for cl_id, stats in self.epoch_stats.items():
            agg_cl = {}
            # Means of scalars
            for k in ['loss_total', 'loss_trans', 'mu0_term', 'entropy_pi', 'ce_loss']:
                if k in stats and stats[k]:
                    agg_cl[k] = np.mean(stats[k])
                else:
                    agg_cl[k] = 0.0
            
            # Means of vectors (stack then mean)
            for k in ['mu', 'pi', 'gains', 'scale']:
                if len(stats[k]) > 0:
                    stack = np.stack(stats[k])
                    agg_cl[k] = np.mean(stack, axis=0) # Average over batches
                else:
                    agg_cl[k] = None
                
            # Deltas: dict k->list of arrays. We want to concatenate all batch arrays for K to get distribution
            deltas_cat = {}
            if len(stats['deltas']) > 0:
                all_keys = stats['deltas'][0].keys()
                for key in all_keys:
                    arrays = [d[key] for d in stats['deltas'] if key in d]
                    if arrays:
                         deltas_cat[key] = np.concatenate(arrays)
            agg_cl['deltas'] = deltas_cat
            
            # Lk_weighted, SCORE_k, VIOL_k: list of dicts. Average per k.
            for metric in ['Lk_weighted', 'SCORE_k', 'VIOL_k']:
                avg_dict = {}
                if len(stats[metric]) > 0:
                    all_keys = stats[metric][0].keys()
                    for key in all_keys:
                        vals = [d[key] for d in stats[metric] if key in d]
                        avg_dict[key] = np.mean(vals)
                agg_cl[metric] = avg_dict

            # mu0 stats
            mu0_vals = [d['mu0'] for d in stats['mu0_stats']]
            agg_cl['mu0_mean'] = np.mean(mu0_vals) if mu0_vals else 0.0
            
            aggregated[cl_id] = agg_cl

        class DictWrapper:
            def __init__(self, d):
                self.d = d
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.d

        return [('ordinal_stats', DictWrapper(aggregated))]
        
    def plot_params(self, params_history, log_dir, best_epoch=None):
        """
        Génère des courbes d'évolution des paramètres.
        params_history: list of dicts [{'epoch': E, 'ordinal_stats': aggregated_stats}, ...]
        """
        import matplotlib.pyplot as plt
        import pathlib

        root_dir = pathlib.Path(log_dir) / 'ordinal_params'
        root_dir.mkdir(parents=True, exist_ok=True)

        # Re-organize data: cluster_id -> { metric -> [values over epochs] }
        cluster_series = {}
        
        # Determine if params_history is list or dict
        # Based on previous error, it is a list
        if isinstance(params_history, dict):
            # Sort by epoch if keys are epochs
            iterator = sorted(params_history.items())
        else:
            # Assume list of dicts
            iterator = []
            for entry in params_history:
                if 'epoch' in entry:
                    iterator.append((entry['epoch'], entry))
            iterator.sort(key=lambda x: x[0])
        
        for ep, entry in iterator:
            # Extract ordinal_stats
            # In pytorch_model_tools, it does: dict_params[name] = value
            # define name='ordinal_stats' in get_attribute
            if 'ordinal_stats' in entry:
                stats_container = entry['ordinal_stats']
                # Check if DictWrapper
                if hasattr(stats_container, 'd'):
                    ep_data = stats_container.d
                else:
                    ep_data = stats_container
            else:
                # Maybe params_history was passed as dict {cl_id: stats} if old format?
                # But let's assume new format from get_attribute
                continue

            if not ep_data:
                continue
            
            for cl_id, stats in ep_data.items():
                if cl_id not in cluster_series:
                    cluster_series[cl_id] = {
                        'epochs': [],
                        'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                        'pi': [], 'entropy_pi': [], 'ce_loss': [],
                        'mu': [],
                        'deltas': {}, # k -> {median:[], min:[], viol:[], neg:[]}
                        'gains': [], 'scale': [],
                        'Lk_weighted': {}, 'SCORE_k': {}, 'VIOL_k': {}, 'NEG_k': {},
                        'mu0_mean': [],
                        'delta_min_history': [], 'delta_max_history': []
                    }
                
                s = cluster_series[cl_id]
                s['epochs'].append(ep)
                s['loss_total'].append(stats.get('loss_total', 0.0))
                s['loss_trans'].append(stats.get('loss_trans', 0.0))
                s['mu0_term'].append(stats.get('mu0_term', 0.0))
                s['entropy_pi'].append(stats.get('entropy_pi', 0.0))
                s['ce_loss'].append(stats.get('ce_loss', 0.0))
                
                s['pi'].append(stats.get('pi', None))
                s['mu'].append(stats.get('mu', None))
                s['gains'].append(stats.get('gains', None))
                s['scale'].append(stats.get('scale', 1.0)) # Default scale 1.0
                s['mu0_mean'].append(stats.get('mu0_mean', 0.0))
                
                # Deltas stats
                # stats['deltas'] is dict k->array of deltas for this epoch-cluster
                # We want to compute scalar stats (median, min...) for the plot
                deltas_map = stats.get('deltas', {})
                if deltas_map:
                    for k, d_vals in deltas_map.items():
                        if k not in s['deltas']:
                            s['deltas'][k] = {'median':[], 'min':[], 'viol':[], 'neg':[]}
                        
                        if d_vals is not None and d_vals.size > 0:
                            s['deltas'][k]['median'].append(np.median(d_vals))
                            s['deltas'][k]['min'].append(np.min(d_vals))
                            viol = (d_vals < 0).mean()
                            s['deltas'][k]['viol'].append(viol)
                            neg = np.mean(np.log(1 + np.exp(-d_vals)))
                            s['deltas'][k]['neg'].append(neg)
                            
                            s['delta_min_history'].append(np.min(d_vals))
                            s['delta_max_history'].append(np.max(d_vals))
                        else:
                            s['deltas'][k]['median'].append(0)
                            s['deltas'][k]['min'].append(0)
                            s['deltas'][k]['viol'].append(0)
                            s['deltas'][k]['neg'].append(0)
                
                # Metrics per k
                for mKey in ['Lk_weighted', 'SCORE_k', 'VIOL_k', 'NEG_k']:
                    mDict = stats.get(mKey, {})
                    if mDict:
                        for k, val in mDict.items():
                            if k not in s[mKey]:
                                s[mKey][k] = []
                            s[mKey][k].append(val)
                # Ensure all k have same length (fill missing with nan or 0)
                # (For simplicity we assume strict structure)

        # Now Plot per cluster
        for cl_id, series in cluster_series.items():
            cl_dir = root_dir / str(cl_id)
            cl_dir.mkdir(parents=True, exist_ok=True)
            
            epochs = series['epochs']
            if not epochs:
                continue

            # 1) Loss components
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, series['loss_total'], label='Total Loss', linewidth=2)
                plt.plot(epochs, series['loss_trans'], label='Inequality Loss (Quadratic ReLU)', linestyle='--')
                plt.plot(epochs, series['mu0_term'], label='Mu0 Term', linestyle=':')
                if any(v != 0 for v in series['ce_loss']):
                    plt.plot(epochs, series['ce_loss'], label='CE Loss', color='purple', linestyle='-.')
                if best_epoch is not None:
                    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                plt.title(f'Cluster {cl_id} - Loss Components')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(cl_dir / '1_loss_components.png')
                plt.close()
            except Exception as e:
                print(f"Error plotting loss components for cluster {cl_id}: {e}")
                plt.close()

            # 2) Pi_s (Masses) & Entropy
            try:
                valid_pi = [p for p in series['pi'] if p is not None]
                if valid_pi:
                    pi_stack = np.stack(valid_pi) # (E, C)
                    if pi_stack.ndim == 1:
                         pi_stack = pi_stack.reshape(-1, 1)
                    
                    if pi_stack.ndim >= 2 and pi_stack.shape[0] == len(epochs):
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        ax2 = ax1.twinx()
                        C = pi_stack.shape[1]
                        for c in range(C):
                            ax1.plot(epochs, pi_stack[:, c], label=f'Class {c}')
                        ax2.plot(epochs, series['entropy_pi'], label='Entropy', color='black', linestyle='--')
                        
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Probability Mass')
                        ax2.set_ylabel('Entropy')
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        
                        if best_epoch is not None:
                            ax1.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                            lines1, labels1 = ax1.get_legend_handles_labels()

                        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                        plt.title(f'Cluster {cl_id} - Predicted Mass distribution (pi_s)')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(cl_dir / '2_pi_s_entropy.png')
                        plt.close()
                else:
                    # just plot entropy
                     plt.figure(figsize=(10, 6))
                     plt.plot(epochs, series['entropy_pi'], label='Entropy')
                     plt.title(f'Cluster {cl_id} - Entropy')
                     plt.savefig(cl_dir / '2_entropy_only.png')
                     plt.close()
            except Exception as e:
                print(f"Error plotting pi/entropy for cluster {cl_id}: {e}")
                plt.close()
            
            # 3) Mu(s)
            try:
                valid_mu = [m for m in series['mu'] if m is not None]
                if valid_mu:
                    mu_stack = np.stack(valid_mu)
                    if mu_stack.ndim == 1:
                        mu_stack = mu_stack.reshape(-1, 1)
                    
                    if mu_stack.ndim >= 2 and mu_stack.shape[0] == len(epochs):
                        plt.figure(figsize=(10, 6))
                        C_mu = mu_stack.shape[1]
                        for c in range(C_mu):
                            plt.plot(epochs, mu_stack[:, c], label=f'mu({c})')
                        if best_epoch is not None:
                            plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                        plt.title(f'Cluster {cl_id} - Mu(s) evolution')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(cl_dir / '3_mu_s.png')
                        plt.close()
            except Exception as e:
                print(f"Error plotting mu for cluster {cl_id}: {e}")
                plt.close()

            # 4) Deltas stats per k
            try:
                ks = sorted(series['deltas'].keys())
                if ks:
                    fig, axes = plt.subplots(len(ks), 4, figsize=(20, 3*len(ks)), sharex=True)
                    if len(ks) == 1: axes = axes[None, :] 
                    
                    for i, k in enumerate(ks):
                        dstats = series['deltas'][k]
                        if len(dstats['median']) == len(epochs):
                            axes[i, 0].plot(epochs, dstats['median'], color='blue')
                            axes[i, 0].set_title(f'k={k} Median Delta')
                            axes[i, 0].grid(True)
                            
                            axes[i, 1].plot(epochs, dstats['min'], color='red')
                            axes[i, 1].set_title(f'k={k} Min Delta')
                            axes[i, 1].grid(True)
                            
                            axes[i, 2].plot(epochs, dstats['viol'], color='orange')
                            axes[i, 2].set_title(f'k={k} Violation Rate (<0)')
                            axes[i, 2].set_ylim(-0.1, 1.1)
                            axes[i, 2].grid(True)
                            
                            axes[i, 3].plot(epochs, dstats['neg'], color='purple')
                            axes[i, 3].set_title(f'k={k} Mean NEG (Softplus magnitude)')
                            axes[i, 3].grid(True)
                    
                    if best_epoch is not None:
                        for ax_row in axes:
                            for ax in ax_row:
                                ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5)

                    plt.tight_layout()
                    plt.savefig(cl_dir / '4_deltas_stats.png')
                    plt.close()
            except Exception as e:
                print(f"Error plotting deltas for cluster {cl_id}: {e}")
                plt.close()

            # 5) Metrics per k (Lk, SCORE_k, VIOL_k, NEG_k)
            try:
                fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
                
                # Helper to plot dict of k->vals
                def plot_k_lines(ax, data_dict, title):
                    did_plot = False
                    for k, vals in data_dict.items():
                        if len(vals) == len(epochs):
                             ax.plot(epochs, vals, label=f'k={k}')
                             did_plot = True
                    ax.set_title(title)
                    if did_plot: ax.legend()
                    ax.grid(True)
                
                plot_k_lines(axes[0], series['Lk_weighted'], 'Weighted Level Losses (wk * Lk)')
                plot_k_lines(axes[1], series['SCORE_k'], 'SCORE_k')
                plot_k_lines(axes[2], series['VIOL_k'], 'VIOL_k')
                axes[2].set_ylim(-0.1, 1.1)
                plot_k_lines(axes[3], series['NEG_k'], 'NEG_k (Softplus magnitude)')

                if best_epoch is not None:
                    for ax in axes:
                        ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5)

                plt.tight_layout()
                plt.savefig(cl_dir / '5_metrics_per_k.png')
                plt.close()
            except Exception as e:
                 print(f"Error plotting metrics per k for cluster {cl_id}: {e}")
                 plt.close()

            # 6) Counts/Lambda & Gains
            try:
                valid_gains = [g for g in series['gains'] if g is not None]
                if valid_gains and len(valid_gains) == len(epochs):
                     gains_stack = np.stack(valid_gains) # (E, C-1) ?
                     scale_stack = np.array(series['scale'])
                     
                     # Check shapes
                     if gains_stack.ndim == 1:
                         gains_stack = gains_stack.reshape(-1, 1)
                     
                     if gains_stack.shape[0] == scale_stack.shape[0]:
                         # Convert to absolute scale for easier reading
                         abs_gains_stack = gains_stack * scale_stack[:, None]

                         fig, ax1 = plt.subplots(figsize=(10, 6))
                         ax2 = ax1.twinx()
                        
                         for i in range(abs_gains_stack.shape[1]):
                            ax1.plot(epochs, abs_gains_stack[:, i], label=f'Abs Gain {i}', linestyle='-')
                        
                         ax2.plot(epochs, scale_stack, label='Scale', color='black', linewidth=2, linestyle='--')
                        
                         ax1.set_xlabel('Epoch')
                         ax1.set_ylabel('Absolute Gains (Gain * Scale)')
                         ax2.set_ylabel('Global Scale')
                        
                         lines1, labels1 = ax1.get_legend_handles_labels()
                         lines2, labels2 = ax2.get_legend_handles_labels()
                         if best_epoch is not None:
                            ax1.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                            lines1, labels1 = ax1.get_legend_handles_labels()

                         ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                        
                         plt.title(f'Cluster {cl_id} - Gains & Scale Evolution')
                         plt.grid(True, alpha=0.3)
                         plt.savefig(cl_dir / '6_gains_scale.png')
                         plt.close()
            except Exception as e:
                 print(f"Error plotting gains/scale for cluster {cl_id}: {e}")
                 plt.close()
            
            # 8) Mu0 stats
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, series['mu0_mean'], label='Mu0 Mean')
                plt.title(f'Cluster {cl_id} - Mu(0) Mean')
                plt.xlabel('Epoch')
                plt.ylabel('Mu(0)')
                if best_epoch is not None:
                    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                plt.grid(True, alpha=0.3)
                plt.savefig(cl_dir / '8_mu0_stats.png')
                plt.close()
            except Exception as e:
                print(f"Error plotting mu0 stats for cluster {cl_id}: {e}")
                plt.close()

            # 9) Hyperparameter Effects
            try:
                # Determine range from history
                if len(series['delta_max_history']) > 0 and len(series['delta_min_history']) > 0:
                    global_min = np.min(series['delta_min_history'])
                    global_max = np.max(series['delta_max_history'])
                    # Add 10% margin
                    span = global_max - global_min
                    if span < 1e-6: span = 1.0
                    global_min -= 0.1 * span
                    global_max += 0.1 * span
                else:
                    global_min, global_max = -2.0, 2.0

                self.plot_hyperparams_effects(cl_dir, delta_range=(global_min, global_max))
            except Exception as e:
                 print(f"Error plotting hyperparams effects for cluster {cl_id}: {e}")


    def plot_hyperparams_effects(self, dir_output, delta_range=(-2.0, 2.0)):
        """
        Plots the effect of betasoftmin on SoftMin and tviolation on Violation Penalty.
        Generates 'hyperparams_effects.png'.
        delta_range: (min, max) tuple to define the x-axis.
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        
        # Range of deltas
        dmin, dmax = delta_range
        # Ensure 0 is included if possible or at least cover it if range is close
        if dmax < 0: dmax = 0.5
        if dmin > 0: dmin = -0.5
        
        deltas = np.linspace(dmin, dmax, 400)
        deltas_t = torch.tensor(deltas, dtype=torch.float32)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- 1) SoftMin Effect (beta) ---
        # SoftMin is usually applied to a set. Here we visualize SoftMin([0, delta])
        # which approximates min(0, delta).
        betas = [1.0, 2.0, 5.0, 10.0, 20.0]
        # Include current beta
        if self.beta not in betas:
            betas.append(self.beta)
        betas = sorted(list(set(betas)))
        
        # Reference: true min(0, delta)
        min_vals = np.minimum(0, deltas)
        axes[0].plot(deltas, min_vals, 'k--', label='min(0, x)', linewidth=2, alpha=0.5)
        
        for b in betas:
            # SoftMin([0, x]) = -(1/b) * log( exp(0) + exp(-b*x) ) = -(1/b) * log( 1 + exp(-b*x) )
            # We compute it using torch for stability
            # But wait, self._softmin computes softmin of a vector. 
            # If we pass [0, x], it reduces those 2.
            
            # Vectorized calculation for plotting:
            # y = -(1/b) * log( 1 + exp(-b * delta) )
            # equals -softplus(-b * delta) / b ? No.
            # log(1 + exp(z)) = softplus(z).
            # So -(1/b) * softplus(-b * delta).
            
            y = -(1.0 / b) * F.softplus(-b * deltas_t)
            
            style = '-' if b == self.beta else ':'
            width = 2 if b == self.beta else 1
            label = f'beta={b}' + (' (current)' if b == self.beta else '')
            axes[0].plot(deltas, y.numpy(), style, label=label, linewidth=width)
            
        axes[0].set_title('SoftMin Approximation of min(0, delta)')
        axes[0].set_xlabel('delta')
        axes[0].set_ylabel('SoftMin(0, delta)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # --- 2) Violation Penalty Effect (tviolation) ---
        # Violation = Sigmoid(-delta / t)
        ts = [0.01, 0.05, 0.1, 0.5, 1.0]
        # Include current t
        if self.t not in ts:
            ts.append(self.t)
        ts = sorted(list(set(ts)))
        
        for t in ts:
            # Sigmoid(-delta / t)
            y = torch.sigmoid(-deltas_t / t)
            
            style = '-' if t == self.t else ':'
            width = 2 if t == self.t else 1
            label = f't={t}' + (' (current)' if t == self.t else '')
            axes[1].plot(deltas, y.numpy(), style, label=label, linewidth=width)
            
        axes[1].set_title('Violation Penalty (Sigmoid(-delta/t))')
        axes[1].set_xlabel('delta')
        axes[1].set_ylabel('Penalty')
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(dir_output / 'hyperparams_effects.png')
        plt.close()

class OMMSE(nn.Module):
    """
    Ordinal Margin MSE Loss (OrdinalMargeMSE)
    
    Version simplifiée qui calcule UNIQUEMENT la MSE entre les différences de mu 
    et les gains cibles. Reprend le preprocessing de OrdinalMonotonicLossNoCoverageWithGains
    pour calculer les gains adaptatifs, mais simplifie drastiquement le forward.
    
    Loss: MSE((mu[i+1] - mu[i]) / scale, gains[i])
    
    Permet de tester si le terme MSE seul suffit à résoudre les problèmes.
    """
    
    def __init__(
        self,
        num_classes=5,
        mushrinkalpha=1.0,
        eps=1e-8,
        quantileedges=(0.5, 0.8, 0.95),
        minbinn=3,
        gainsalpha=1.0,
        gainsfloorfrac=0.1,
        enforcegainmonotone=True,
        id=None,
        enablelogs=True,
        addglobal=False,
        lambda_mse=1.0
    ):
        super().__init__()
        self.addglobal = bool(addglobal)
        self.id = id
        self.C = int(num_classes)
        if self.C < 2:
            raise ValueError("num_classes doit être >= 2.")
        if len(quantileedges) != (self.C - 2):
            raise ValueError("quantileedges (positifs) doit avoir longueur C-2=%d (reçu %d)"
                             % (self.C - 2, len(quantileedges)))
        
        self.mushrinkalpha = float(mushrinkalpha)
        self.eps = float(eps)
        
        self.quantileedges = tuple(float(x) for x in quantileedges)
        self.minbinn = int(minbinn)
        self.gainsalpha = float(gainsalpha)
        self.gainsfloorfrac = float(gainsfloorfrac)
        self.enforcegainmonotone = bool(enforcegainmonotone)
        self.enablelogs = bool(enablelogs)
        self.lambda_mse = float(lambda_mse)
        
        # Buffers pour gains et scales
        self.gain_k = {}
        self.register_buffer("global_gains", torch.zeros(self.C - 1, dtype=torch.float32))
        self.register_buffer("global_scale", torch.tensor(1.0, dtype=torch.float32))
        
        self.call_preprocess = False
    
    def get_config(self):
        return {
            "numclasses": self.C,
            "mushrinkalpha": self.mushrinkalpha,
            "eps": self.eps,
            "quantileedges": self.quantileedges,
            "minbinn": self.minbinn,
            "gainsalpha": self.gainsalpha,
            "gainsfloorfrac": self.gainsfloorfrac,
            "enforcegainmonotone": self.enforcegainmonotone,
            "enablelogs": self.enablelogs,
            "id": self.id,
            "addglobal": self.addglobal,
            "lambda_mse": self.lambda_mse
        }
    
    def _log(self, *args):
        if self.enablelogs:
            print(*args)
    
    def _adaptive_positive_thresholds(self, y_pos, qe_pos):
        """Copié de OrdinalMonotonicLossNoCoverageWithGains"""
        q_max = 0.99
        q_step = 0.05
        tol = 0.0
        
        thresholds = []
        used_q = []
        last_t = None
        forced_flat = False
        
        for q0 in qe_pos:
            q_try = float(q0)
            if q_try <= 0.0 or q_try >= 1.0:
                raise ValueError("quantileedges doit être dans ]0,1[ (reçu %s)" % str(qe_pos))
            
            if forced_flat:
                thresholds.append(float(last_t))
                used_q.append(float(q_max))
                continue
            
            t = float(np.quantile(y_pos, q_try))
            
            if last_t is not None and t <= last_t + tol:
                q = q_try
                t_new = t
                while q < q_max - 1e-12:
                    q = min(q_max, q + q_step)
                    t_new = float(np.quantile(y_pos, q))
                    if t_new > last_t + tol:
                        break
                
                if t_new > last_t + tol:
                    q_try = q
                    t = t_new
                else:
                    forced_flat = True
                    q_try = q_max
                    t = float(last_t)
            
            thresholds.append(float(t))
            used_q.append(float(q_try))
            last_t = float(t)
        
        return (
            np.array(thresholds, dtype=np.float32),
            np.array(used_q, dtype=np.float32),
            bool(forced_flat),
        )
    
    def _compute_bins_and_ok_positive_quantiles(
        self,
        y_sub,
        qe_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=3,
        tag=""
    ):
        """Copié de OrdinalMonotonicLossNoCoverageWithGains"""
        y_sub = y_sub[np.isfinite(y_sub)]
        if y_sub.size == 0:
            return None, False, None, None, False, None
        
        y0 = y_sub[y_sub <= 0]
        y_pos = y_sub[y_sub > 0]
        if y_pos.size == 0:
            return None, False, None, None, False, None
        
        qs_pos, used_q, forced_flat = self._adaptive_positive_thresholds(y_pos, qe_pos)
        
        bins = []
        bins.append(y0)
        
        q1 = qs_pos[0]
        bins.append(y_pos[y_pos <= q1])
        
        for i in range(1, len(qs_pos)):
            lo = qs_pos[i - 1]
            hi = qs_pos[i]
            bins.append(y_pos[(y_pos > lo) & (y_pos <= hi)])
        
        q_last = qs_pos[-1]
        bins.append(y_pos[y_pos > q_last])
        
        bin_means = np.array([float(np.mean(b)) if b.size > 0 else np.nan for b in bins], dtype=np.float32)
        counts = np.array([int(b.size) for b in bins], dtype=np.int32)
        pos_counts = counts.copy()
        pos_counts[0] = 0
        
        ok = True
        
        if not np.all(np.diff(qs_pos) > 0):
            ok = False
        
        if ok and not np.all(counts[1:] >= int(minB)):
            ok = False
        
        if ok and require_positive_bins:
            if not np.all(pos_counts[1:] >= int(min_pos_per_bin)):
                ok = False
        
        self._log(
            f"[BINS {tag}] means={bin_means} counts={counts} ok={ok} "
            f"qs_pos={qs_pos} used_q={used_q} forced_flat={forced_flat}"
        )
        
        return bin_means, bool(ok), qs_pos, used_q, forced_flat, counts
    
    def _gains_from_bin_means(self, bin_means, y_ref, tag=""):
        """Copié de OrdinalMonotonicLossNoCoverageWithGains"""
        diffs = np.diff(bin_means)
        diffs = np.maximum(diffs, 0.0)
        
        y_ref = y_ref[np.isfinite(y_ref)]
        y_pos = y_ref[y_ref > 0]
        
        if y_pos.size >= 10:
            spread = 1.0
        else:
            spread = 1.0
        
        floor = max(0.0, self.gainsfloorfrac)
        
        base = self.gainsalpha * diffs
        gains = np.maximum(base, floor).astype(np.float32)
        
        if self.enforcegainmonotone:
            gains = np.maximum.accumulate(gains).astype(np.float32)
        
        self._log(
            f"[GAINS {tag}] diffs={diffs} base={base} floor={floor} final={gains}"
        )
        return gains, float(spread)
    
    def _preprocess(
        self,
        y_cont,
        clusters_ids,
        similar_cluster_ids=None,
        quantileedges=None,
        minbinn=None,
        require_positive_bins=True,
        min_pos_per_bin=3,
        enforcegainmonotone=None,
    ):
        """Preprocessing: calculate bins and gains per cluster"""
        self.call_preprocess = True
        y = np.asarray(y_cont)
        c = np.asarray(clusters_ids)
        
        if y.ndim != 1 or c.ndim != 1 or len(y) != len(c):
            raise ValueError("y_cont et clusters_ids doivent être 1D et de même longueur.")
        
        if similar_cluster_ids is not None:
            s = np.asarray(similar_cluster_ids)
            if s.ndim != 1 or len(s) != len(c):
                raise ValueError("similar_cluster_ids doit être 1D et de même longueur que clusters_ids.")
        else:
            s = None
        
        qe_pos = self.quantileedges if quantileedges is None else tuple(float(x) for x in quantileedges)
        if len(qe_pos) != (self.C - 2):
            raise ValueError("quantileedges (positifs) doit avoir longueur C-2=%d" % (self.C - 2))
        
        minB = self.minbinn if minbinn is None else int(minbinn)
        if enforcegainmonotone is not None:
            self.enforcegainmonotone = bool(enforcegainmonotone)
        
        idx_by_cluster = {}
        unique_clusters = np.unique(c)
        for cl in unique_clusters:
            idx_by_cluster[cl] = np.where(c == cl)[0]
        
        idx_by_group = None
        cluster_group = None
        if s is not None:
            idx_by_group = {}
            for g in np.unique(s):
                idx_by_group[g] = np.where(s == g)[0]
            
            cluster_group = {}
            for cl in unique_clusters:
                idx = idx_by_cluster[cl]
                vals, cnts = np.unique(s[idx], return_counts=True)
                cluster_group[cl] = vals[np.argmax(cnts)] if vals.size > 0 else None
        
        # GLOBAL
        self._log("==== GLOBAL GAINS (bins->gains) ====")
        bm_g, ok_g, qs_g, usedq_g, flat_g, cnt_g = self._compute_bins_and_ok_positive_quantiles(
            y, qe_pos, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin, tag="GLOBAL"
        )
        if bm_g is not None and ok_g:
            global_g, global_spread = self._gains_from_bin_means(bm_g, y, tag="GLOBAL")
            self.global_scale = torch.tensor(global_spread, dtype=torch.float32, device=self.global_gains.device)
            self.register_buffer('global_thresholds', torch.tensor(qs_g, dtype=torch.float32))
        else:
            raise ValueError('Cannot calculate bin means')
        
        self.global_gains = torch.tensor(global_g, dtype=torch.float32, device=self.global_gains.device)
        self._log("[GLOBAL GAINS] tensor:", self.global_gains)
        
        # PER CLUSTER
        self.gain_k = {}
        self.scale_k = {}
        self._log("==== CLUSTER GAINS ====")
        
        for cl in unique_clusters:
            idx = idx_by_cluster[cl]
            y_cl = y[idx]
            
            self._log(f"\n--- cluster {cl} (n={len(idx)}) ---")
            bm, ok, qs, usedq, flat, cnt = self._compute_bins_and_ok_positive_quantiles(
                y_cl, qe_pos, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin,
                tag="CL_%s" % str(cl)
            )
            
            if bm is not None and ok:
                g_cl, spread_cl = self._gains_from_bin_means(bm, y_cl, tag="CL_%s" % str(cl))
            else:
                pooled = False
                if cluster_group is not None and idx_by_group is not None:
                    g_id = cluster_group.get(cl, None)
                    if g_id is not None and g_id in idx_by_group:
                        idx_pool = idx_by_group[g_id]
                        y_pool = y[idx_pool]
                        self._log(f"[CL {cl}] pool by similar group {g_id} (n_pool={len(idx_pool)})")
                        
                        bm2, ok2, qs2, usedq2, flat2, cnt2 = self._compute_bins_and_ok_positive_quantiles(
                            y_pool, qe_pos, minB,
                            require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin,
                            tag="POOL_G%s" % str(g_id)
                        )
                        if bm2 is not None and ok2:
                            g_cl, spread_cl = self._gains_from_bin_means(bm2, y_pool, tag="POOL_G%s" % str(g_id))
                            pooled = True
                        else:
                            self._log(f"[CL {cl}] pooling failed -> fallback global gains")
                            g_cl = global_g
                            spread_cl = global_spread
                    else:
                        self._log(f"[CL {cl}] no valid similar group -> fallback global gains")
                        g_cl = global_g
                        spread_cl = global_spread
                else:
                    self._log(f"[CL {cl}] no similar_cluster_ids -> fallback global gains")
                    g_cl = global_g
                    spread_cl = global_spread
            
            if self.enforcegainmonotone:
                g_cl = np.maximum.accumulate(np.asarray(g_cl, dtype=np.float32)).astype(np.float32)
            
            self.gain_k[cl] = torch.tensor(g_cl, dtype=torch.float32, device=self.global_gains.device)
            self.scale_k[cl] = torch.tensor(spread_cl, dtype=torch.float32, device=self.global_gains.device)
            
            self._log(f"[CLUSTER GAINS] {cl} -> {g_cl}")
    
    def _mu_soft(self, p, y, sw=None):
        """Calcul des mu par classe"""
        if not torch.is_floating_point(y):
            y = y.to(dtype=p.dtype)
        
        if sw is not None:
            sw = sw.to(device=p.device, dtype=p.dtype).clamp_min(self.eps)
            p_eff = p * sw.unsqueeze(1)
            counts = p_eff.sum(dim=0)
            num = (p_eff * y.unsqueeze(1)).sum(dim=0)
        else:
            counts = p.sum(dim=0)
            num = (p * y.unsqueeze(1)).sum(dim=0)
        
        if self.mushrinkalpha > 0:
            mu0 = y.mean()
            a = self.mushrinkalpha
            mu = (num + a * mu0) / (counts + a).clamp_min(self.eps)
        else:
            mu = num / counts.clamp_min(self.eps)
        
        return mu
    
    def _get_cluster_gains(self, cluster_id, device, dtype):
        if hasattr(self, "gain_k") and (cluster_id in self.gain_k):
            g = self.gain_k[cluster_id]
        else:
            g = self.global_gains
        return g.to(device=device, dtype=dtype)
    
    def _get_cluster_scale(self, cluster_id, device, dtype):
        if hasattr(self, "scale_k") and (cluster_id in self.scale_k):
            s = self.scale_k[cluster_id]
        else:
            s = self.global_scale
        return s.to(device=device, dtype=dtype)
    
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.epoch_stats = {}
    
    def forward(self, logits, y_cont, clusters_ids, sample_weight=None):
        """
        Forward simplifié: calcule UNIQUEMENT MSE((mu[i+1] - mu[i]) / scale, gains[i])
        """
        assert self.call_preprocess is True, 'You have to call preprocess before forward'
        if logits.dim() != 2:
            raise ValueError("logits doit etre (N,C). Recu: %s" % (tuple(logits.shape),))
        if y_cont.dim() != 1 or clusters_ids.dim() != 1:
            raise ValueError("y_cont et clusters_ids doivent etre des tenseurs 1D (N,).")
        if logits.size(0) != y_cont.size(0) or logits.size(0) != clusters_ids.size(0):
            raise ValueError("logits, y_cont, clusters_ids doivent avoir le meme N.")
        
        if not hasattr(self, 'epoch_stats'):
            self.epoch_stats = {}
        
        probs = F.softmax(logits, dim=1)
        y_cont = y_cont.to(device=logits.device, dtype=probs.dtype)
        
        sw = None
        if sample_weight is not None:
            sw = sample_weight.to(device=logits.device, dtype=probs.dtype)
        
        total_loss = logits.new_tensor(0.0)
        total_w = logits.new_tensor(0.0)
        
        unique_clusters = torch.unique(clusters_ids)
        
        for d in unique_clusters:
            idx = torch.nonzero(clusters_ids == d, as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue
            
            p = probs[idx]
            y = y_cont[idx]
            sw_d = sw[idx] if sw is not None else None
            
            cl_id = d.item() if torch.is_tensor(d) else d
            
            if cl_id not in self.epoch_stats:
                self.epoch_stats[cl_id] = {
                    'loss_mse': [],
                    'mu': [],
                    'mu_diffs_scaled': [],
                    'target_diffs': [],
                    'scale': []
                }
            
            # Calcul des mu
            mu = self._mu_soft(p, y, sw=sw_d)
            
            # Récupération des gains et scale pour ce cluster
            gains = self._get_cluster_gains(cl_id, p.device, p.dtype)
            scale = self._get_cluster_scale(cl_id, p.device, p.dtype)
            
            # === TERME MSE UNIQUEMENT ===
            mu_diffs = mu[1:] - mu[:-1]
            mu_diffs_scaled = mu_diffs / (scale + self.eps)
            
            # MSE entre différences réelles et cibles (gains)
            loss_mse = F.mse_loss(mu_diffs_scaled, gains)
            
            cluster_loss = self.lambda_mse * loss_mse
            
            total_loss = total_loss + cluster_loss
            total_w = total_w + 1.0
            
            # Stats
            est = self.epoch_stats[cl_id]
            est['loss_mse'].append(loss_mse.detach().item())
            est['mu'].append(mu.detach().cpu().numpy())
            est['mu_diffs_scaled'].append(mu_diffs_scaled.detach().cpu().numpy())
            est['target_diffs'].append(gains.detach().cpu().numpy())
            est['scale'].append(scale.detach().cpu().numpy())
        
        if total_w.abs() < self.eps:
            final_loss = logits.new_tensor(0.0)
        else:
            final_loss = total_loss / total_w
        
        # === GLOBAL LOSS (optionnel) ===
        if self.addglobal and len(unique_clusters) > 1:
            g_gains = self.global_gains.to(device=logits.device, dtype=probs.dtype)
            g_scale = self.global_scale.to(device=logits.device, dtype=probs.dtype)
            
            mu_global = self._mu_soft(probs, y_cont, sw=sw)
            mu_diffs_g = mu_global[1:] - mu_global[:-1]
            mu_diffs_scaled_g = mu_diffs_g / (g_scale + self.eps)
            
            loss_mse_g = F.mse_loss(mu_diffs_scaled_g, g_gains)
            global_loss = self.lambda_mse * loss_mse_g
            
            final_loss = final_loss + global_loss
            
            if "global" not in self.epoch_stats:
                self.epoch_stats["global"] = {
                    'loss_mse': [],
                    'mu': [],
                    'mu_diffs_scaled': [],
                    'target_diffs': [],
                    'scale': []
                }
            est_g = self.epoch_stats["global"]
            est_g['loss_mse'].append(loss_mse_g.detach().item())
            est_g['mu'].append(mu_global.detach().cpu().numpy())
            est_g['mu_diffs_scaled'].append(mu_diffs_scaled_g.detach().cpu().numpy())
            est_g['target_diffs'].append(g_gains.detach().cpu().numpy())
            est_g['scale'].append(g_scale.detach().cpu().numpy())
        
        return final_loss
    
    def get_attribute(self):
        """Agrégation des stats"""
        aggregated = {}
        
        for cl_id, stats in self.epoch_stats.items():
            agg_cl = {}
            
            # Scalaires
            if 'loss_mse' in stats and stats['loss_mse']:
                agg_cl['loss_mse'] = np.mean(stats['loss_mse'])
            else:
                agg_cl['loss_mse'] = 0.0
            
            # Vecteurs
            for k in ['mu', 'mu_diffs_scaled', 'target_diffs', 'scale']:
                if len(stats[k]) > 0:
                    stack = np.stack(stats[k])
                    agg_cl[k] = np.mean(stack, axis=0)
                else:
                    agg_cl[k] = None
            
            aggregated[cl_id] = agg_cl
        
        class DictWrapper:
            def __init__(self, d):
                self.d = d
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.d
        
        return [('ordinal_stats', DictWrapper(aggregated))]

    def plot_params(self, params_history, log_dir, best_epoch=None):
        """
        Génère des courbes d'évolution des paramètres pour OMMSE (version simplifiée).
        params_history: list of dicts [{'epoch': E, 'ordinal_stats': aggregated_stats}, ...]
        """
        import matplotlib.pyplot as plt
        import pathlib

        root_dir = pathlib.Path(log_dir) / 'ommse_params'
        root_dir.mkdir(parents=True, exist_ok=True)

        # Re-organize data: cluster_id -> { metric -> [values over epochs] }
        cluster_series = {}
        
        # Determine if params_history is list or dict
        if isinstance(params_history, dict):
            iterator = sorted(params_history.items())
        else:
            iterator = []
            for entry in params_history:
                if 'epoch' in entry:
                    iterator.append((entry['epoch'], entry))
            iterator.sort(key=lambda x: x[0])
        
        for ep, entry in iterator:
            if 'ordinal_stats' in entry:
                stats_container = entry['ordinal_stats']
                if hasattr(stats_container, 'd'):
                    ep_data = stats_container.d
                else:
                    ep_data = stats_container
            else:
                continue

            if not ep_data:
                continue
            
            for cl_id, stats in ep_data.items():
                if cl_id not in cluster_series:
                    cluster_series[cl_id] = {
                        'epochs': [],
                        'loss_mse': [],
                        'mu': [],
                        'mu_diffs_scaled': [],
                        'target_diffs': [],
                        'scale': []
                    }
                
                s = cluster_series[cl_id]
                s['epochs'].append(ep)
                s['loss_mse'].append(stats.get('loss_mse', 0.0))
                s['mu'].append(stats.get('mu', None))
                s['mu_diffs_scaled'].append(stats.get('mu_diffs_scaled', None))
                s['target_diffs'].append(stats.get('target_diffs', None))
                s['scale'].append(stats.get('scale', 1.0))

        # Plot per cluster
        for cl_id, series in cluster_series.items():
            cl_dir = root_dir / str(cl_id)
            cl_dir.mkdir(parents=True, exist_ok=True)
            
            epochs = series['epochs']
            if not epochs:
                continue

            # 1) MSE Loss Evolution
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, series['loss_mse'], label='MSE Loss', linewidth=2, color='blue')
                if best_epoch is not None:
                    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                plt.title(f'Cluster {cl_id} - MSE Loss Evolution')
                plt.xlabel('Epoch')
                plt.ylabel('MSE Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(cl_dir / '1_mse_loss.png')
                plt.close()
            except Exception as e:
                print(f"Error plotting MSE loss for cluster {cl_id}: {e}")
                plt.close()

            # 2) Mu Evolution
            try:
                valid_mu = [m for m in series['mu'] if m is not None]
                if valid_mu:
                    mu_stack = np.stack(valid_mu)
                    if mu_stack.ndim == 1:
                        mu_stack = mu_stack.reshape(-1, 1)
                    
                    if mu_stack.ndim >= 2 and mu_stack.shape[0] == len(epochs):
                        plt.figure(figsize=(10, 6))
                        C_mu = mu_stack.shape[1]
                        for c in range(C_mu):
                            plt.plot(epochs, mu_stack[:, c], label=f'mu({c})', marker='o' if c == 0 else None)
                        if best_epoch is not None:
                            plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                        plt.title(f'Cluster {cl_id} - Mu Evolution')
                        plt.xlabel('Epoch')
                        plt.ylabel('Mu Values')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(cl_dir / '2_mu_evolution.png')
                        plt.close()
            except Exception as e:
                print(f"Error plotting mu for cluster {cl_id}: {e}")
                plt.close()

            # 3) Mu Diffs vs Target (convergence)
            try:
                valid_diffs = [d for d in series['mu_diffs_scaled'] if d is not None]
                valid_targets = [t for t in series['target_diffs'] if t is not None]
                
                if valid_diffs and valid_targets and len(valid_diffs) == len(epochs):
                    diffs_stack = np.stack(valid_diffs)
                    targets_stack = np.stack(valid_targets)
                    
                    if diffs_stack.ndim == 1:
                        diffs_stack = diffs_stack.reshape(-1, 1)
                    if targets_stack.ndim == 1:
                        targets_stack = targets_stack.reshape(-1, 1)
                    
                    # Plot last epoch comparison
                    plt.figure(figsize=(10, 6))
                    last_diffs = diffs_stack[-1]
                    last_targets = targets_stack[-1]
                    classes = np.arange(len(last_diffs))
                    
                    plt.bar(classes - 0.2, last_diffs, width=0.4, label='Actual (mu[i+1]-mu[i])/scale', alpha=0.7)
                    plt.bar(classes + 0.2, last_targets, width=0.4, label='Target (gains)', alpha=0.7)
                    plt.xlabel('Transition (i -> i+1)')
                    plt.ylabel('Value')
                    plt.title(f'Cluster {cl_id} - Mu Diffs vs Targets (Final Epoch)')
                    plt.legend()
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.savefig(cl_dir / '3_diffs_vs_targets.png')
                    plt.close()
                    
                    # Plot MSE convergence per transition
                    plt.figure(figsize=(12, 6))
                    n_transitions = diffs_stack.shape[1]
                    for i in range(n_transitions):
                        mse_i = (diffs_stack[:, i] - targets_stack[0, i]) ** 2
                        plt.plot(epochs, mse_i, label=f'Transition {i}->{i+1}', alpha=0.7)
                    if best_epoch is not None:
                        plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                    plt.title(f'Cluster {cl_id} - Squared Error per Transition')
                    plt.xlabel('Epoch')
                    plt.ylabel('(mu_diff - target)^2')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.yscale('log')
                    plt.savefig(cl_dir / '4_convergence_per_transition.png')
                    plt.close()
                    
            except Exception as e:
                print(f"Error plotting diffs vs targets for cluster {cl_id}: {e}")
                plt.close()

            # 4) Scale Evolution
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, series['scale'], label='Scale', linewidth=2, color='green')
                if best_epoch is not None:
                    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                plt.title(f'Cluster {cl_id} - Scale Evolution')
                plt.xlabel('Epoch')
                plt.ylabel('Scale')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(cl_dir / '5_scale.png')
                plt.close()
            except Exception as e:
                print(f"Error plotting scale for cluster {cl_id}: {e}")
                plt.close()

        print(f"OMMSE plots saved to {root_dir}")
        
        # Generate mushrinkalpha effect visualization
        try:
            print("Generating mushrinkalpha effect visualization...")
            self.plot_mushrinkalpha_effect(root_dir)
        except Exception as e:
            print(f"Error generating mushrinkalpha visualization: {e}")

    
    def plot_mushrinkalpha_effect(self, dir_output, sample_counts=None, mu_raw=None, mu0=None):
        """
        Visualise l'effet de mushrinkalpha sur le calcul des mu.
        
        Formula: mu = (num + alpha * mu0) / (counts + alpha)
        
        Args:
            dir_output: Répertoire de sortie
            sample_counts: Array de counts par classe (optionnel, sinon utilise [1, 10, 100, 1000])
            mu_raw: Array de mu "raw" = num/counts sans shrinkage (optionnel)
            mu0: Valeur de mu0 (moyenne globale, optionnel, défaut 1.0)
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        
        # Valeurs par défaut
        if sample_counts is None:
            sample_counts = np.array([1, 5, 10, 50, 100, 500, 1000])
        else:
            sample_counts = np.asarray(sample_counts)
        
        if mu_raw is None:
            # Simuler des mu_raw variés
            mu_raw = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])[:len(sample_counts)]
        else:
            mu_raw = np.asarray(mu_raw)
        
        if mu0 is None:
            mu0 = 1.0
        
        # Tester différentes valeurs d'alpha
        alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        # === PLOT 1: Effet sur différentes classes (avec counts variés) ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: Mu en fonction de counts pour différents alphas
        ax = axes[0, 0]
        for alpha in alphas:
            # Simuler: mu_raw fixe, compter effet du count
            mu_shrunk = (mu_raw * sample_counts + alpha * mu0) / (sample_counts + alpha)
            label = f'α={alpha}' + (' (current)' if alpha == self.mushrinkalpha else '')
            style = '-' if alpha == self.mushrinkalpha else '--'
            width = 2 if alpha == self.mushrinkalpha else 1
            ax.semilogx(sample_counts, mu_shrunk, style, label=label, linewidth=width)
        
        ax.axhline(mu0, color='k', linestyle=':', alpha=0.5, label=f'mu0={mu0}')
        ax.plot(sample_counts, mu_raw, 'ko', label='Raw mu (no shrinkage)', markersize=5)
        ax.set_xlabel('Sample Count (log scale)')
        ax.set_ylabel('Mu (after shrinkage)')
        ax.set_title('Effect of mushrinkalpha on Mu vs Sample Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Shrinkage strength vs counts
        ax = axes[0, 1]
        for alpha in alphas:
            # Shrinkage strength = alpha / (counts + alpha)
            shrinkage_weight = alpha / (sample_counts + alpha)
            label = f'α={alpha}' + (' (current)' if alpha == self.mushrinkalpha else '')
            style = '-' if alpha == self.mushrinkalpha else '--'
            width = 2 if alpha == self.mushrinkalpha else 1
            ax.semilogx(sample_counts, shrinkage_weight, style, label=label, linewidth=width)
        
        ax.set_xlabel('Sample Count (log scale)')
        ax.set_ylabel('Shrinkage Weight = α/(count+α)')
        ax.set_title('Shrinkage Strength vs Sample Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Subplot 3: Distance to mu0 reduction
        ax = axes[1, 0]
        for alpha in alphas:
            mu_shrunk = (mu_raw * sample_counts + alpha * mu0) / (sample_counts + alpha)
            distance_reduction = np.abs(mu_raw - mu0) - np.abs(mu_shrunk - mu0)
            label = f'α={alpha}' + (' (current)' if alpha == self.mushrinkalpha else '')
            style = '-' if alpha == self.mushrinkalpha else '--'
            width = 2 if alpha == self.mushrinkalpha else 1
            ax.semilogx(sample_counts, distance_reduction, style, label=label, linewidth=width)
        
        ax.set_xlabel('Sample Count (log scale)')
        ax.set_ylabel('Distance Reduction to mu0')
        ax.set_title('How much closer to mu0 after shrinkage')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        
        # Subplot 4: Effective sample size
        ax = axes[1, 1]
        for alpha in alphas:
            effective_n = sample_counts + alpha
            label = f'α={alpha}' + (' (current)' if alpha == self.mushrinkalpha else '')
            style = '-' if alpha == self.mushrinkalpha else '--'
            width = 2 if alpha == self.mushrinkalpha else 1
            ax.loglog(sample_counts, effective_n, style, label=label, linewidth=width)
        
        ax.loglog(sample_counts, sample_counts, 'k:', alpha=0.5, label='No shrinkage (n=n)')
        ax.set_xlabel('Actual Sample Count (log scale)')
        ax.set_ylabel('Effective Sample Size (count+α)')
        ax.set_title('Effective Sample Size with Shrinkage')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(dir_output / 'mushrinkalpha_effect.png', dpi=150)
        plt.close()
        
        # === PLOT 2: Comparative bar chart for specific scenario ===
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Simuler 5 classes avec différents counts
        n_classes = 5
        counts_scenario = np.array([1000, 500, 100, 10, 1])  # Classes de bien à mal représentées
        mu_raw_scenario = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # Valeurs espacées
        mu0_scenario = 2.0  # Moyenne globale
        
        # Comparer 3 alphas
        alphas_compare = [0.0, 1.0, 10.0]
        
        x = np.arange(n_classes)
        width = 0.25
        
        for i, alpha in enumerate(alphas_compare):
            mu_shrunk = (mu_raw_scenario * counts_scenario + alpha * mu0_scenario) / (counts_scenario + alpha)
            offset = (i - 1) * width
            label = f'α={alpha}' + (' (current)' if alpha == self.mushrinkalpha else '')
            ax.bar(x + offset, mu_shrunk, width, label=label, alpha=0.7)
        
        ax.plot(x, mu_raw_scenario, 'ko-', label='Raw mu (α=0)', markersize=8, linewidth=2, alpha=0.5)
        ax.axhline(mu0_scenario, color='r', linestyle='--', alpha=0.5, label=f'mu0={mu0_scenario}')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Mu Value')
        ax.set_title(f'Mushrinkalpha Effect on Mu Values\n(Counts: {counts_scenario})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i}\n(n={counts_scenario[i]})' for i in range(n_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(dir_output / 'mushrinkalpha_comparison.png', dpi=150)
        plt.close()
        
        print(f"Mushrinkalpha visualization saved to {dir_output}")
        print(f"  - mushrinkalpha_effect.png: 4 subplots showing different aspects")
        print(f"  - mushrinkalpha_comparison.png: Bar chart comparison for specific scenario")
        print(f"\nCurrent mushrinkalpha: {self.mushrinkalpha}")
        print(f"  - α=0: No shrinkage (mu = num/counts)")
        print(f"  - α>0: Shrinks towards mu0, stronger for low counts")
        print(f"  - High α: Strong regularization, all mu closer to mu0")


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CORNLoss(nn.Module):
    """
    CORN loss (Conditional Ordinal Regression for Neural Networks).

    Attendu:
    - y_pred: logits de forme (N, num_classes-1)
      Chaque colonne j correspond à la tâche conditionnelle:
        f_{j+1}(x) = P(y > j+1 | y > j)   pour j>=1
      et pour j=0:
        f_1(x) = P(y > 0)
    - y_true: labels entiers de forme (N,) dans [0, num_classes-1]

    La loss est la somme des BCE (en logits) sur des sous-ensembles conditionnels:
      tâche i (seuil i) est entraînée sur les exemples avec y > i-1
    (voir Section 3.3–3.5) :contentReference[oaicite:1]{index=1}
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes doit être >= 2.")
        self.num_classes = num_classes

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            Logits de forme (N, num_classes-1) pour les tâches CORN.
        y_true : torch.Tensor
            Labels entiers de forme (N,) dans [0, num_classes-1].
        sample_weight : Optional[torch.Tensor]
            Poids par sample (N,).

        Returns
        -------
        torch.Tensor
            Loss scalaire.
        """
        y_true = y_true.long().view(-1)
        N = y_true.shape[0]

        if y_pred.ndim != 2:
            raise ValueError(f"y_pred doit être 2D, obtenu {y_pred.ndim}D.")
        if y_pred.shape[0] != N:
            raise ValueError("y_pred et y_true doivent avoir la même taille batch.")
        if y_pred.shape[1] != self.num_classes - 1:
            raise ValueError(
                f"y_pred doit avoir {self.num_classes-1} colonnes "
                f"(num_classes-1). Obtenu: {y_pred.shape[1]}"
            )

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(y_pred.device, dtype=y_pred.dtype)
            if sample_weight.shape[0] != N:
                raise ValueError("sample_weight doit être de forme (N,).")
        # Accumulateurs
        total_loss = y_pred.new_tensor(0.0)
        total_denom = y_pred.new_tensor(0.0)  # nb d'exemples (ou somme des poids) effectivement utilisés

        # Tâches i = 0..K-2 (seuil i, cible: y > i)
        # Sous-ensemble conditionnel: y > (i-1)
        for i in range(self.num_classes - 1):
            # mask conditionnel (toujours vrai pour i=0 car y > -1)
            cond_mask = (y_true > (i - 1))
            if not torch.any(cond_mask):
                continue

            # cibles binaires pour la tâche i: 1 si y > i sinon 0 (sur le sous-ensemble)
            t = (y_true[cond_mask] > i).to(y_pred.dtype)

            # logits correspondants
            z = y_pred[cond_mask, i]

            # BCE en logits, version numériquement stable (équivalente à Eq. 6)
            # loss_vec = -[ log(sigmoid(z))*t + (log(sigmoid(z)) - z)*(1-t) ]
            log_sig = F.logsigmoid(z)
            loss_vec = -(log_sig * t + (log_sig - z) * (1.0 - t))

            if sample_weight is not None:
                w = sample_weight[cond_mask]
                total_loss = total_loss + torch.sum(loss_vec * w)
                total_denom = total_denom + torch.sum(w)
            else:
                total_loss = total_loss + torch.sum(loss_vec)
                total_denom = total_denom + loss_vec.numel()

        # Sécurité: si rien n'a été accumulé (cas pathologique)
        if total_denom.item() == 0.0:
            return total_loss  # = 0

        return total_loss / total_denom

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence

class CORNFocalLoss(nn.Module):
    """
    CORN + Focal Loss (stable, en logits).

    y_pred: logits (N, K-1)
    y_true: labels (N,) dans [0..K-1]
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = 0.25,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes doit être >= 2.")

        self.num_classes = num_classes
        self.gamma = gamma

        a = torch.as_tensor(alpha, dtype=torch.float32)
        self.alpha = a

    def forward(
        self,
        y_pred: torch.Tensor,  # logits (N, K-1)
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        y_true = y_true.long().view(-1)
        N = y_true.shape[0]

        if y_pred.shape != (N, self.num_classes - 1):
            raise ValueError(
                f"y_pred doit être de forme (N, {self.num_classes-1})"
            )

        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(y_pred.device)

        total_loss = y_pred.new_tensor(0.0)
        total_weight = y_pred.new_tensor(0.0)

        # Boucle sur les seuils CORN
        for i in range(self.num_classes - 1):

            # Sous-ensemble conditionnel
            cond_mask = (y_true > (i - 1))
            if not torch.any(cond_mask):
                continue

            z = y_pred[cond_mask, i]                     # logits
            t = (y_true[cond_mask] > i).float()          # targets 0/1

            # BCE stable en logits
            bce = F.binary_cross_entropy_with_logits(z, t, reduction="none")

            # Probabilité p_t (correct class prob)
            p = torch.sigmoid(z)
            p_t = p * t + (1 - p) * (1 - t)

            focal_factor = (1 - p_t).pow(self.gamma)

            if self.alpha is not None:
                a = self.alpha[0] if self.alpha.numel() == 1 else self.alpha[i]
                alpha_t = a * t + (1 - a) * (1 - t)
                loss_vec = alpha_t * focal_factor * bce
            else:
                loss_vec = focal_factor * bce

            if sample_weight is not None:
                w = sample_weight[cond_mask]
                total_loss += torch.sum(loss_vec * w)
                total_weight += torch.sum(w)
            else:
                total_loss += torch.sum(loss_vec)
                total_weight += loss_vec.numel()

        if total_weight.item() == 0:
            return total_loss

        return total_loss / total_weight

def corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Retourne P(y = k) pour k in [0..K-1]
    logits: (N, K-1)
    """

    cond_probs = torch.sigmoid(logits)          # f_k
    cum_probs = torch.cumprod(cond_probs, dim=1)  # P(y > k)

    N, K_minus_1 = cum_probs.shape
    K = K_minus_1 + 1

    probs = torch.zeros((N, K), device=logits.device)

    # P(y=0)
    probs[:, 0] = 1 - cum_probs[:, 0]

    # P(y=k)
    for k in range(1, K-1):
        probs[:, k] = cum_probs[:, k-1] - cum_probs[:, k]

    # P(y=K-1)
    probs[:, K-1] = cum_probs[:, -1]

    return probs

class CORNWithGains(nn.Module):
    """
    Ordinal monotonic loss (transition-based) with per-cluster minimum gains (margins),
    with fallback to "similar clusters" pooling when a cluster cannot define valid bins.

    Key:
      - Class 0: y <= 0
      - Positive classes 1..C-1: bins on y_pos = y[y>0], with (C-2) positive thresholds.
      - Thresholds are "adaptive": if quantiles are equal (discrete y), we push q upward
        until threshold strictly increases; cap at q_max=0.99.
      - Gains are integrated as margin inside deltas:
            delta_{a->b} = (mu[b] - mu[a]) - sum_{j=a..b-1} g[j]
    """

    def __init__(
        self,
        num_classes=5,
        betasoftmin=10.0,
        tviolation=0.01,
        mushrinkalpha=1.0,
        eps=1e-8,
        wk=None,
        quantileedges=(0.5, 0.8, 0.95),
        minbinn=3,
        gainsalpha=1.0,
        gainsfloorfrac=0.1,
        enforcegainmonotone=True,
        id=None,
        enablelogs=True,
        lambdamu0 = 0.0,
        lambdaentropy = 0.0,
        wmed=1.0,
        wmin=1.0,
        wneg=1.0,
        wviol=1.0,
        lambdadir=0.0,
        diralpha=1.05,
        lambdace=0.0,
        lambdagl=0.0,
        cetype='focal',
        alpha=0.25,
        gamma=2
    ):
        super().__init__()
        self.lambdadir = float(lambdadir)
        self.diralpha = float(diralpha)
        self.lambdace = float(lambdace)
        self.cetype = str(cetype).lower()

        # CE Loss Setup - choose between CORN focal and CORN cross-entropy
        if self.cetype == 'focal':
            self.ce_loss = CORNFocalLoss(num_classes=num_classes, alpha=alpha, gamma=gammma)
        elif self.cetype == 'corn':
            self.ce_loss = CORNLoss(num_classes=num_classes)
        else:
            raise ValueError(f"cetype must be 'focal' or 'corn', got '{cetype}'")

        self.id = id
        self.C = int(num_classes)
        if self.C < 2:
            raise ValueError("num_classes doit être >= 2.")
        if len(quantileedges) != (self.C - 2):
            raise ValueError("quantileedges (positifs) doit avoir longueur C-2=%d (reçu %d)"
                             % (self.C - 2, len(quantileedges)))

        self.beta = float(betasoftmin)
        self.t = float(tviolation)
        self.mushrinkalpha = float(mushrinkalpha)
        self.eps = float(eps)

        self.quantileedges = tuple(float(x) for x in quantileedges)
        self.minbinn = int(minbinn)
        self.gainsalpha = float(gainsalpha)
        self.gainsfloorfrac = float(gainsfloorfrac)
        self.enforcegainmonotone = bool(enforcegainmonotone)

        self.P = self._build_Pk(self.C)
        self.wk = {k: 1.0 for k in range(1, self.C)} if wk is None else wk
        
        self.enablelogs = bool(enablelogs)
        self.lambdamu0 = lambdamu0
        self.lambdaentropy = lambdaentropy
        # cluster_id -> torch.FloatTensor(C-1)
        self.gain_k = {}
        self.register_buffer("global_gains", torch.zeros(self.C - 1, dtype=torch.float32))
        self.register_buffer("global_scale", torch.tensor(1.0, dtype=torch.float32))

        # Component weights
        self.wmed = float(wmed)
        self.wmin = float(wmin)
        self.wneg = float(wneg)
        self.wviol = float(wviol)

        self.lambdagl = float(lambdagl)

        self.call_preprocess = False

        #self._log("Ordinal Loss Config:", self.get_config())

    def get_config(self):
        return {
            "numclasses": self.C,
            "betasoftmin": self.beta,
            "tviolation": self.t,
            "mushrinkalpha": self.mushrinkalpha,
            "eps": self.eps,
            "quantileedges": self.quantileedges,
            "minbinn": self.minbinn,
            "gainsalpha": self.gainsalpha,
            "gainsfloorfrac": self.gainsfloorfrac,
            "enforcegainmonotone": self.enforcegainmonotone,
            "enablelogs": self.enablelogs,
            "lambdamu0": self.lambdamu0,
            "lambdaentropy": self.lambdaentropy,
            "wmed": self.wmed,
            "wmin": self.wmin,
            "wneg": self.wneg,
            "wviol": self.wviol,

            "id": self.id,
            "lambdagl": self.lambdagl,
            "lambdadir": self.lambdadir,
            "diralpha": self.diralpha
        }

    # ------------------------------------------------------------
    # Utilities: logs
    # ------------------------------------------------------------
    def _log(self, *args):
        if self.enablelogs:
            print(*args)

    # ------------------------------------------------------------
    # Adaptive thresholds on y_pos
    # ------------------------------------------------------------
    def _adaptive_positive_thresholds(self, y_pos, qe_pos):
        """
        qe_pos: list/tuple of target quantiles (len = C-2).
        For each target, compute threshold. If threshold does not strictly increase
        vs previous threshold, push quantile upward by q_step until it does (cap q_max).

        Returns:
          thresholds: np.float32 (C-2,)
          used_q: np.float32 (C-2,)
          forced_flat: bool (True if we had to freeze remaining thresholds)
        """
        q_max = 0.99
        q_step = 0.05
        tol = 0.0

        thresholds = []
        used_q = []
        last_t = None
        forced_flat = False

        for q0 in qe_pos:
            q_try = float(q0)
            if q_try <= 0.0 or q_try >= 1.0:
                raise ValueError("quantileedges doit être dans ]0,1[ (reçu %s)" % str(qe_pos))

            if forced_flat:
                thresholds.append(float(last_t))
                used_q.append(float(q_max))
                continue

            t = float(np.quantile(y_pos, q_try))

            if last_t is not None and t <= last_t + tol:
                q = q_try
                t_new = t
                while q < q_max - 1e-12:
                    q = min(q_max, q + q_step)
                    t_new = float(np.quantile(y_pos, q))
                    if t_new > last_t + tol:
                        break

                if t_new > last_t + tol:
                    q_try = q
                    t = t_new
                else:
                    forced_flat = True
                    q_try = q_max
                    t = float(last_t)

            thresholds.append(float(t))
            used_q.append(float(q_try))
            last_t = float(t)

        return (
            np.array(thresholds, dtype=np.float32),
            np.array(used_q, dtype=np.float32),
            bool(forced_flat),
        )
    
    def _dirichlet_barrier(self, pi):
        """
        R(pi) = -(diralpha - 1) * sum_k log(pi_k + eps)
        """
        eps = 1e-9
        pi_safe = pi.clamp(min=eps)
        return -(self.diralpha - 1.0) * pi_safe.log().sum()

    def _compute_single_loss_component(
        self, 
        probs, 
        y, 
        sw, 
        gains, 
        scale, 
        wk_dict,
        lam0, 
        logits_factory, # tensor to create new tensors on same device
        thresholds=None,
        lambdace=None,
        lambdaentropy=None,
        lambdadir=None
    ):
        """
        Helper to compute loss components (Trans, Mu0) + stats for a given subset (cluster or global).
        Returns:
            loss_val: scalar tensor
            w_val: scalar tensor
            stats_dict: dict with various metrics (mu, pi, entropy, deltas, etc.)
        """
        # Resolve lambdas
        lambdace = lambdace if lambdace is not None else self.lambdace
        lambdaentropy = lambdaentropy if lambdaentropy is not None else self.lambdaentropy
        lambdadir = lambdadir if lambdadir is not None else self.lambdadir

        # --- 5) Masse de proba pi_s ---
        if sw is not None:
            p_weighted = probs * sw.unsqueeze(1)
            pi_s = p_weighted.sum(dim=0) / sw.sum().clamp_min(self.eps)
        else:
            pi_s = probs.mean(dim=0)
        
        mu = self._mu_soft(probs, y, sw=sw)  # (C,)
        
        # Entropy of pi_s
        pi_log_pi = pi_s * torch.log(pi_s.clamp_min(1e-9))
        entropy_pi = -pi_log_pi.sum()

        cluster_loss = logits_factory.new_tensor(0.0)
        cluster_w = logits_factory.new_tensor(0.0)
        
        # --- 8) Terme mu(0) ---
        mu0_val = mu[0]
        mu0_term = F.softplus(mu0_val / (scale + self.eps))

        stats_loss_trans = logits_factory.new_tensor(0.0)
        stats_w_trans = logits_factory.new_tensor(0.0)
        
        Lk_w_batch = {}
        SCORE_k_batch = {}
        VIOL_k_batch = {}
        NEG_k_batch = {}
        
        deltas_batch = {}

        # SCORE_k with margined deltas
        for k, pairs in self.P.items():
            if not pairs:
                continue

            raw = torch.stack([mu[b] - mu[a] for (a, b) in pairs], dim=0)
            margins = torch.stack([gains[a:b].sum() for (a, b) in pairs], dim=0)
            raw = raw / (scale + self.eps)
            deltas = raw - margins
            
            deltas_batch[k] = deltas.detach().cpu().numpy()

            # Surrogates
            MINk = self._softmin(deltas)
            MEDk = self._soft_median(deltas)
            VIOLk = torch.sigmoid(-deltas / self.t).mean()
            NEGk = F.softplus(-deltas).mean()
            
            # === MULTI-CRITERIA INEQUALITY LOSS (ReLU) ===
            # We want MED > 0, MIN > 0, and minimizing magnitude of violations
            # Use ReLU to penalize ONLY when criteria are violated (< 0)
            
            # Criterion 1: Median Penalty (Quadratic)
            loss_med = F.softplus(-MEDk)
            
            # Criterion 2: Minimum Penalty (Quadratic)
            loss_min = F.softplus(-MINk)
            
            # Criterion 3: Magnitude Penalty (Mean of squared violations)
            loss_neg = (F.softplus(-deltas)).mean()
            
            # Weighted Combination
            Lk = (self.wmed * loss_med +
                  self.wmin * loss_min + 
                  self.wneg * loss_neg)

            # SCOREk for logging (kept similar to before for continuity)
            SCOREk = (self.wmed * MEDk + self.wmin * MINk) - self.wneg * NEGk * (1.0 + self.wviol * VIOLk)

            w = float(wk_dict.get(k, 1.0))
            cluster_loss = cluster_loss + w * Lk
            cluster_w = cluster_w + w
            
            stats_loss_trans = stats_loss_trans + w * Lk
            stats_w_trans = stats_w_trans + w

            Lk_w_batch[k] = (w * Lk).detach().item()
            SCORE_k_batch[k] = SCOREk.detach().item()
            VIOL_k_batch[k] = VIOLk.detach().item()
            NEG_k_batch[k] = NEGk.detach().item()

        # Normalize transition loss
        if cluster_w > 0:
            cluster_loss = cluster_loss / cluster_w.clamp_min(self.eps)
            # cluster_w becomes "normalization factor for this cluster" -> 1.0 effectively
            # but we return 1.0 later to tell forward to just average.
        else:
            cluster_loss = logits_factory.new_tensor(0.0)

        # Add Mu0
        if lam0 > 0:
             cluster_loss = cluster_loss + lam0 * mu0_term
        
        # Add CE
        ce_mean = logits_factory.new_tensor(0.0)
        if lambdace > 0:
            logit_tens = logits_factory

            if thresholds is None:
                y_disc = y.long()
            else:
                y_disc = self._discretize_y(y, thresholds)

            ce_mean = self.ce_loss(logit_tens, y_disc, sample_weight=sw)

            ce_mean = ce_mean
            cluster_loss = cluster_loss + lambdace * ce_mean
            
        # Add Entropy
        if lambdaentropy > 0:
            cluster_loss = cluster_loss - lambdaentropy * entropy_pi
            
        # Add Dirichlet Regularization (moved to cluster level)
        dir_reg = logits_factory.new_tensor(0.0)
        if lambdadir > 0:
            dir_reg = self._dirichlet_barrier(pi_s)
            cluster_loss = cluster_loss + lambdadir * dir_reg

        stats_dict = {
            'loss_total': cluster_loss.detach().item(),
            'loss_trans': (stats_loss_trans / stats_w_trans.clamp_min(self.eps)).detach().item() if stats_w_trans > 0 else 0.0,
            'mu0_term': (lam0 * mu0_term).detach().item(),
            'mu': mu.detach().cpu().numpy(),
            'pi': pi_s.detach().cpu().numpy(),
            'entropy_pi': entropy_pi.item(),
            'entropy_weighted': (-lambdaentropy * entropy_pi).detach().item() if lambdaentropy > 0 else 0.0,
            'dirichlet_reg': dir_reg.detach().item() if lambdadir > 0 else 0.0,
            'dirichlet_weighted': (lambdadir * dir_reg).detach().item() if lambdadir > 0 else 0.0,
            'ce_loss': ce_mean.detach().item() if lambdace > 0 else 0.0,
            'ce_weighted': (lambdace * ce_mean).detach().item() if lambdace > 0 else 0.0,
            'deltas': deltas_batch,
            'gains': gains.detach().cpu().numpy(),
            'scale': scale.detach().cpu().numpy(),
            'Lk_weighted': Lk_w_batch,
            'SCORE_k': SCORE_k_batch,
            'VIOL_k': VIOL_k_batch,
            'NEG_k': NEG_k_batch,
            'mu0_stats': {'mu0': mu0_val.detach().item()}
        }
        
        # Stats for scales and margins
        # scale can be scalar or tensor.
        if scale.numel() > 1:
            scale_min = scale.min().item()
            scale_mean = scale.mean().item()
            scale_max = scale.max().item()
        else:
            v_sc = scale.item()
            scale_min = v_sc
            scale_mean = v_sc
            scale_max = v_sc
            
        # raw diffs = diffs_batch (already computed: mu[1:] - mu[:-1])
        # diffs_batch is a list in current code? No, let's see where it comes from.
        # It's calculated inside the loop over k. We want the mean of adjacent diffs.
        # mu is [num_classes] shape.
        diffs_mu = mu[1:] - mu[:-1]
        raw_mean = diffs_mu.mean().item()
        
        # scaled raw
        # If scale is scalar, it's raw_mean / scale. If vector?
        if scale.numel() > 1:
             # Just an approximation if scale varies per class (not the case usually)
             scaled_raw_mean = (diffs_mu / (scale.mean() + 1e-9)).mean().item()
        else:
             scaled_raw_mean = raw_mean / (scale.item() + 1e-9)
             
        # margins (gains)
        margin_mean = gains.mean().item()
        
        stats_dict.update({
             'scale_min': scale_min,
             'scale_mean': scale_mean,
             'scale_max': scale_max,
             'diff_raw_mean': raw_mean,
             'diff_scaled_mean': scaled_raw_mean,
             'margin_mean': margin_mean
        })
        
        return cluster_loss, 1.0, stats_dict

    def _auto_wk_from_gains_np(self, g_np: np.ndarray, eps: float = 1e-8):
        """
        g_np: (C-1,) gains (global ou cluster). Retourne dict wk[k] pour k=1..C-1
        wk[k] = 1 / ( median_{(a,b) in P_k} sum_{j=a..b-1} g[j] + eps )
        """
        C = self.C
        wk = {}
        for k in range(1, C):
            margins_k = []
            for a in range(0, C - k):
                b = a + k
                margins_k.append(float(np.sum(g_np[a:b])))
            mk = float(np.median(margins_k))  # médiane robuste
            
            # Normalization
            wk[k] = (1.0 / (mk + eps)) * self.wk.get(k, 1.0)
            
            # Simple weight
            #wk[k] = self.wk.get(k, 1.0)
        return wk

    def _get_cluster_wk(self, cluster_id):
        # fallback: cluster -> global -> uniform
        if hasattr(self, "wk_k") and (cluster_id in self.wk_k):
            return self.wk_k[cluster_id]
        if hasattr(self, "wk_global"):
            return self.wk_global
        return self.wk

    # -------------------------------- ----------------------------
    # Build bins: 0 + (C-1) positive bins
    # ------------------------------------------------------------
    def _compute_bins_and_ok_positive_quantiles(
        self,
        y_sub,
        qe_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=3,
        tag=""
    ):
        """
        Returns (bin_means, ok, qs_pos, used_q, forced_flat, counts)

        - bin 0: y <= 0
        - bins 1..C-1: defined on y_pos with thresholds qs_pos (len C-2)
          bin1: (0, q1]
          ...
          bin(C-1): (q_last, +inf)
        """
        y_sub = y_sub[np.isfinite(y_sub)]
        if y_sub.size == 0:
            return None, False, None, None, False, None

        y0 = y_sub[y_sub <= 0]
        y_pos = y_sub[y_sub > 0]
        if y_pos.size == 0:
            return None, False, None, None, False, None

        qs_pos, used_q, forced_flat = self._adaptive_positive_thresholds(y_pos, qe_pos)

        # bins
        bins = []
        # bin0
        bins.append(y0)

        # positive bins
        # bin1: y_pos <= q1
        q1 = qs_pos[0]
        bins.append(y_pos[y_pos <= q1])

        # middle: (q_{i-1}, q_i]
        for i in range(1, len(qs_pos)):
            lo = qs_pos[i - 1]
            hi = qs_pos[i]
            bins.append(y_pos[(y_pos > lo) & (y_pos <= hi)])

        # last: > q_last
        q_last = qs_pos[-1]
        bins.append(y_pos[y_pos > q_last])
        
        # means & counts
        bin_means = np.array([float(np.mean(b)) if b.size > 0 else np.nan for b in bins], dtype=np.float32)
        counts = np.array([int(b.size) for b in bins], dtype=np.int32)
        pos_counts = counts.copy()
        pos_counts[0] = 0

        # validity
        ok = True

        # thresholds must be strictly increasing (otherwise bins degenerate)
        if not np.all(np.diff(qs_pos) > 0):
            ok = False

        # positive bins must have enough samples
        if ok and not np.all(counts[1:] >= int(minB)):
            ok = False

        if ok and require_positive_bins:
            if not np.all(pos_counts[1:] >= int(min_pos_per_bin)):
                ok = False

        self._log(
            f"[BINS {tag}] means={bin_means} counts={counts} ok={ok} "
            f"qs_pos={qs_pos} used_q={used_q} forced_flat={forced_flat}"
        )

        return bin_means, bool(ok), qs_pos, used_q, forced_flat, counts

    # ------------------------------------------------------------
    # Gains from bin_means (adjacent)
    # ------------------------------------------------------------
    def _gains_from_bin_means(self, bin_means, y_ref, tag=""):
        diffs = np.diff(bin_means)
        diffs = np.maximum(diffs, 0.0)

        y_ref = y_ref[np.isfinite(y_ref)]
        y_pos = y_ref[y_ref > 0]

        if y_pos.size >= 10:
            q50 = float(np.quantile(y_pos, 0.5))
            q95 = float(np.quantile(y_pos, 0.95))
            #spread = q95 - q50
            spread = 1.0 # No normalization
        else:
            q50 = float(np.quantile(y_pos, 0.5)) if y_pos.size > 0 else 0.0
            q95 = float(np.quantile(y_pos, 0.95)) if y_pos.size > 0 else 0.0
            #spread = float(np.std(y_pos)) if y_pos.size > 1 else 0.0
            spread = 1.0 # No normalization

        #spread = max(spread, 1e-6)
        floor = max(0.0, self.gainsfloorfrac)
        
        # Normailzation
        #base = self.gainsalpha * (diffs / spread)
        
        base = self.gainsalpha * diffs
        gains = np.maximum(base, floor).astype(np.float32)

        if self.enforcegainmonotone:
            gains = np.maximum.accumulate(gains).astype(np.float32)

        self._log(
            f"[GAINS {tag}] diffs={diffs} base={base} q50_pos={q50} q95_pos={q95} "
            f"spread={spread} floor={floor} final={gains}"
        )
        return gains, float(spread)

    # ------------------------------------------------------------
    # Preprocess with similar_cluster_ids fallback
    # ------------------------------------------------------------
    def _preprocess(
        self,
        y_cont,
        clusters_ids,
        similar_cluster_ids=None,   # same length as clusters_ids, defines similarity groups
        quantileedges=None,
        minbinn=None,
        require_positive_bins=True,
        min_pos_per_bin=3,
        enforcegainmonotone=None,
    ):

        self.call_preprocess = True
        y = np.asarray(y_cont)
        c = np.asarray(clusters_ids)

        if y.ndim != 1 or c.ndim != 1 or len(y) != len(c):
            raise ValueError("y_cont et clusters_ids doivent être 1D et de même longueur.")

        if similar_cluster_ids is not None:
            s = np.asarray(similar_cluster_ids)
            if s.ndim != 1 or len(s) != len(c):
                raise ValueError("similar_cluster_ids doit être 1D et de même longueur que clusters_ids.")
        else:
            s = None

        qe_pos = self.quantileedges if quantileedges is None else tuple(float(x) for x in quantileedges)
        if len(qe_pos) != (self.C - 2):
            raise ValueError("quantileedges (positifs) doit avoir longueur C-2=%d" % (self.C - 2))

        minB = self.minbinn if minbinn is None else int(minbinn)
        if enforcegainmonotone is not None:
            self.enforcegainmonotone = bool(enforcegainmonotone)

        # ---- Build group maps (similarity pooling) ----
        idx_by_cluster = {}
        unique_clusters = np.unique(c)
        for cl in unique_clusters:
            idx_by_cluster[cl] = np.where(c == cl)[0]

        idx_by_group = None
        cluster_group = None
        if s is not None:
            idx_by_group = {}
            for g in np.unique(s):
                idx_by_group[g] = np.where(s == g)[0]

            cluster_group = {}
            for cl in unique_clusters:
                idx = idx_by_cluster[cl]
                vals, cnts = np.unique(s[idx], return_counts=True)
                cluster_group[cl] = vals[np.argmax(cnts)] if vals.size > 0 else None
                
        # ---- GLOBAL ----
        self._log("==== GLOBAL GAINS (bins->gains) ====")
        bm_g, ok_g, qs_g, usedq_g, flat_g, cnt_g = self._compute_bins_and_ok_positive_quantiles(
            y, qe_pos, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin, tag="GLOBAL"
        )
        if bm_g is not None and ok_g:
            global_g, global_spread = self._gains_from_bin_means(bm_g, y, tag="GLOBAL")
            self.global_scale = torch.tensor(global_spread, dtype=torch.float32, device=self.global_gains.device)
            self.register_buffer('global_thresholds', torch.tensor(qs_g, dtype=torch.float32))
        else:
            raise ValueError('Can t calcule bin means')
            # robust fallback
            yfin = y[np.isfinite(y)]
            scale = float(np.std(yfin)) if yfin.size > 1 else 1.0
            global_g = np.full(self.C - 1, 0.05 * scale, dtype=np.float32)
            self._log("[GLOBAL] fallback scale-based gains:", global_g)

        self.global_gains = torch.tensor(global_g, dtype=torch.float32, device=self.global_gains.device)
        self._log("[GLOBAL GAINS] tensor:", self.global_gains)

        # ---- PER CLUSTER ----
        self.gain_k = {}
        self.scale_k = {}
        self.thresholds_k = {}
        self._log("==== CLUSTER GAINS ====")

        for cl in unique_clusters:
            idx = idx_by_cluster[cl]
            y_cl = y[idx]

            self._log(f"\n--- cluster {cl} (n={len(idx)}) ---")
            bm, ok, qs, usedq, flat, cnt = self._compute_bins_and_ok_positive_quantiles(
                y_cl, qe_pos, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin,
                tag="CL_%s" % str(cl)
            )
            
            if bm is not None and ok:
                g_cl, spread_cl = self._gains_from_bin_means(bm, y_cl, tag="CL_%s" % str(cl))
            else:
                pooled = False
                if cluster_group is not None and idx_by_group is not None:
                    g_id = cluster_group.get(cl, None)
                    if g_id is not None and g_id in idx_by_group:
                        idx_pool = idx_by_group[g_id]
                        y_pool = y[idx_pool]
                        self._log(f"[CL {cl}] pool by similar group {g_id} (n_pool={len(idx_pool)})")

                        bm2, ok2, qs2, usedq2, flat2, cnt2 = self._compute_bins_and_ok_positive_quantiles(
                            y_pool, qe_pos, minB,
                            require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin,
                            tag="POOL_G%s" % str(g_id)
                        )
                        if bm2 is not None and ok2:
                            g_cl, spread_cl = self._gains_from_bin_means(bm2, y_pool, tag="POOL_G%s" % str(g_id))
                            pooled = True
                        else:
                            self._log(f"[CL {cl}] pooling failed -> fallback global gains")
                            g_cl = global_g
                    else:
                        self._log(f"[CL {cl}] no valid similar group -> fallback global gains")
                        g_cl = global_g
                else:
                    self._log(f"[CL {cl}] no similar_cluster_ids -> fallback global gains")
                    g_cl = global_g

                if pooled:
                    self._log(f"[CL {cl}] used pooled gains")
                else:
                    self._log(f"[CL {cl}] used global gains")

            if self.enforcegainmonotone:
                g_cl = np.maximum.accumulate(np.asarray(g_cl, dtype=np.float32)).astype(np.float32)

            self.gain_k[cl] = torch.tensor(g_cl, dtype=torch.float32, device=self.global_gains.device)
            self.scale_k[cl] = torch.tensor(spread_cl, dtype=torch.float32, device=self.global_gains.device)
            
            # Store per-cluster thresholds for CE discretization
            # If pooling/fallback used, we might not have specific qs.
            # If ok is True, we have qs.
            # If pooled (ok2 is True), we have qs2.
            # Else fallback global -> global_thresholds.
            
            if ok:
                 res_qs = qs
            elif pooled and ok2:
                 res_qs = qs2
            else:
                 res_qs = qs_g # Fallback to global thresholds
            
            if res_qs is not None:
                self.thresholds_k[cl] = torch.tensor(res_qs, dtype=torch.float32, device=self.global_gains.device)
            else:
                # Should correspond to global fallback if qs_g is available
                # If even qs_g is None (which raises ValueError above anyway), we are in trouble.
                self.thresholds_k[cl] = self.global_thresholds

            self._log(f"[CLUSTER GAINS] {cl} -> {g_cl}")
        
        # ---- AUTO wk (global + per-cluster) ----
        self.wk_global = self._auto_wk_from_gains_np(
            self.global_gains.detach().cpu().numpy(),
            eps=self.eps
        )

        self.wk_k = {}
        for cl, g_t in self.gain_k.items():
            self.wk_k[cl] = self._auto_wk_from_gains_np(
                g_t.detach().cpu().numpy(),
                eps=self.eps
            )

        print("[AUTO wk] global:", {k: round(v, 6) for k, v in self.wk_global.items()})
        print("[AUTO wk] per-cluster:", {cl: {k: round(v,6) for k,v in wk.items()} for cl, wk in self.wk_k.items()})

    # ------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------
    @staticmethod
    def _build_Pk(C):
        P = {}
        for k in range(1, C):
            P[k] = [(a, a + k) for a in range(0, C - k)]
        #print("[TRANSITION AND K]:", P)
        return P

    def _get_cluster_scale(self, cluster_id, device, dtype):
        if hasattr(self, "scale_k") and (cluster_id in self.scale_k):
            s = self.scale_k[cluster_id]
        else:
            s = self.global_scale
        return s.to(device=device, dtype=dtype)

    def _softmin(self, x):
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=0)

    def _soft_median(self, deltas):
        alpha = 20.0
        c = deltas.mean()
        w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
        return (w * deltas).sum()

    def _mu_soft(self, p, y, sw=None):
        # ensure y float
        if not torch.is_floating_point(y):
            y = y.to(dtype=p.dtype)

        if sw is not None:
            sw = sw.to(device=p.device, dtype=p.dtype).clamp_min(self.eps)
            p_eff = p * sw.unsqueeze(1)
            counts = p_eff.sum(dim=0)
            num = (p_eff * y.unsqueeze(1)).sum(dim=0)
        else:
            counts = p.sum(dim=0)
            num = (p * y.unsqueeze(1)).sum(dim=0)

        if self.mushrinkalpha > 0:
            mu0 = y.mean()
            a = self.mushrinkalpha
            mu = (num + a * mu0) / (counts + a).clamp_min(self.eps)
        else:
            mu = num / counts.clamp_min(self.eps)

        return mu  # (C,)

    def _get_cluster_gains(self, cluster_id, device, dtype):
        if hasattr(self, "gain_k") and (cluster_id in self.gain_k):
            g = self.gain_k[cluster_id]
        else:
            g = self.global_gains
        return g.to(device=device, dtype=dtype)

    def _get_cluster_thresholds(self, cluster_id, device, dtype):
        if hasattr(self, "thresholds_k") and (cluster_id in self.thresholds_k):
            t = self.thresholds_k[cluster_id]
        else:
            t = self.global_thresholds
        return t.to(device=device, dtype=dtype)

    def _discretize_y(self, y_cont, thresholds):
        # Discretize continuous y into classes based on thresholds
        # Class 0: y <= 0 (handled by y > 0 check generally)
        # Class k: thresholds[k-1] < y <= thresholds[k]
        
        # We start with 0s.
        y_disc = torch.zeros_like(y_cont, dtype=torch.long)
        
        # Positive values
        mask_pos = y_cont > 0
        if not mask_pos.any():
            return y_disc
            
        y_pos = y_cont[mask_pos]
        
        if thresholds.numel() == 0:
            # If no thresholds (C=2 case?), implies only 1 positive class? 
            # If thresholds is empty, bucketize returns 0 for all.
            # We want Class 1. so +1.
            y_disc[mask_pos] = 1
        else:
            # bucketize: 
            # right=True: bins[i-1] < x <= bins[i]
            # output index i.
            # For thresholds [t1, t2] (C=4 classes: 0, 1, 2, 3)
            # y <= 0 -> Class 0 (handled separately)
            # 0 < y <= t1 -> bucket 0 -> Class 1
            # t1 < y <= t2 -> bucket 1 -> Class 2
            # y > t2      -> bucket 2 -> Class 3
            
            buckets = torch.bucketize(y_pos, thresholds, right=True)
            y_disc[mask_pos] = buckets + 1
            
        return y_disc

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.epoch_stats = {}

    def forward(self, logits, y_cont, clusters_ids, sample_weight=None):
        assert self.call_preprocess is True, f'You have to call preprocess before forward'
        if logits.dim() != 2:
            raise ValueError("logits doit etre (N,C-1). Recu: %s" % (tuple(logits.shape),))
        if y_cont.dim() != 1 or clusters_ids.dim() != 1:
            raise ValueError("y_cont et clusters_ids doivent etre des tenseurs 1D (N,).")
        if logits.size(0) != y_cont.size(0) or logits.size(0) != clusters_ids.size(0):
            raise ValueError("logits, y_cont, clusters_ids doivent avoir le meme N.")

        # Init epoch_stats if not present (e.g. first forward or eval mode used without train call)
        if not hasattr(self, 'epoch_stats'):
            self.epoch_stats = {}

        probs = corn_class_probs(logits)
        y_cont = y_cont.to(device=logits.device, dtype=probs.dtype)

        sw = None
        if sample_weight is not None:
            sw = sample_weight.to(device=logits.device, dtype=probs.dtype)

        total_loss = logits.new_tensor(0.0)
        total_w = logits.new_tensor(0.0)
        
        unique_clusters = torch.unique(clusters_ids)

        for d in unique_clusters:
            idx = torch.nonzero(clusters_ids == d, as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue

            p = probs[idx]
            y = y_cont[idx]
            sw_d = sw[idx] if sw is not None else None

            cl_id = d.item() if torch.is_tensor(d) else d
            # Init stats for this cluster if needed
            if cl_id not in self.epoch_stats:
                self.epoch_stats[cl_id] = {
                    'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                    'mu': [], 'pi': [], 'entropy_pi': [], 'dirichlet_reg': [], 'ce_loss': [],
                    'deltas': [], 'gains': [], 'scale': [],
                    'Lk_weighted': [], 'SCORE_k': [], 'VIOL_k': [],
                    'NEG_k': [], 'mu0_stats': [],
                }
            
            gains_adj = self._get_cluster_gains(cl_id, p.device, p.dtype)
            wk_dict = self._get_cluster_wk(cl_id)
            scale = self._get_cluster_scale(cl_id, p.device, p.dtype)
            lam0 = self.lambdamu0
            thresholds = self._get_cluster_thresholds(cl_id, p.device, p.dtype)
            
            # Slice logits for this cluster!
            l_d = logits[idx]

            c_loss, c_w, stats = self._compute_single_loss_component(
                p, y, sw_d, gains_adj, scale, wk_dict, lam0, l_d, thresholds
            )

            if c_w > 0:
                total_loss = total_loss + c_loss
                total_w = total_w + c_w
            
            # Keep track of everything (detached)
            est = self.epoch_stats[cl_id]
            for k, v in stats.items():
                if k in est:
                    est[k].append(v)

        if total_w.abs() < self.eps:
            final_loss = logits.new_tensor(0.0)
        else:
            final_loss = total_loss / total_w
            
        # --- GLOBAL LOSS COMPONENT ---
        # If more than one cluster, add global loss (on all data).
        # If only one cluster, the above loop effectively computed the global loss already 
        # (and normalized it). 
        # User request: "le cas où on en voit pas qu'un seul cluster, je veux que tu ajoutes la même loss mais calculé sur l'ensemble des logits"
        
        if self.lambdagl > 0:
            if len(unique_clusters) > 1:
                # Global setup
                # gains -> global_gains
                # scale -> global_scale
                # wk -> self.wk
                g_gains = self.global_gains.to(device=logits.device, dtype=probs.dtype)
                g_scale = self.global_scale.to(device=logits.device, dtype=probs.dtype)
                g_wk = self.wk # default / global
                g_thresholds = self.global_thresholds.to(device=logits.device, dtype=probs.dtype)
                
                # We treat the whole batch as one "global cluster"
                # Note: sample_weight is sw
                # lambdace=0.0, lambdaentropy=0.0, lambdadir=0.0 for GLOBAL component as requested
                g_loss, g_w, g_stats = self._compute_single_loss_component(
                    probs, y_cont, sw, g_gains, g_scale, g_wk, self.lambdamu0, logits, None,
                    lambdace=0.0, lambdaentropy=0.0, lambdadir=0.0
                )
                
                if g_w > 0:
                    global_loss_val = g_loss / g_w
                    final_loss = final_loss + self.lambdagl * global_loss_val
                    
                    # Option: log global stats?
                    # The user didn't explicitly ask for logging, but it helps debugging.
                    # We can store it under a special key "global" if we want, or -1.
                    if "global" not in self.epoch_stats:
                        self.epoch_stats["global"] = {
                            'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                            'mu': [], 'pi': [], 'entropy_pi': [], 'dirichlet_reg': [], 'ce_loss': [],
                            'deltas': [], 'gains': [], 'scale': [],
                            'Lk_weighted': [], 'SCORE_k': [], 'VIOL_k': [],
                            'NEG_k': [], 'mu0_stats': [],
                        }
                    est_g = self.epoch_stats["global"]
                    for k, v in g_stats.items():
                         if k in est_g:
                            est_g[k].append(v)

            elif len(unique_clusters) == 1:
                # Single cluster case: copy the stats from that cluster to "global"
                # because the user requested "tout les logs possibles par cluster et au global".
                cl_id = unique_clusters[0].item() if torch.is_tensor(unique_clusters[0]) else unique_clusters[0]
                if cl_id in self.epoch_stats:
                    src_stats = self.epoch_stats[cl_id]
                    if "global" not in self.epoch_stats:
                       self.epoch_stats["global"] = {
                            'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                            'mu': [], 'pi': [], 'entropy_pi': [], 'dirichlet_reg': [], 'ce_loss': [],
                            'deltas': [], 'gains': [], 'scale': [],
                            'Lk_weighted': [], 'SCORE_k': [], 'VIOL_k': [],
                            'NEG_k': [], 'mu0_stats': [],
                        }
                    est_g = self.epoch_stats["global"]
                    # Append last values from src_stats
                    for k in est_g.keys():
                        if k in src_stats and len(src_stats[k]) > 0:
                            est_g[k].append(src_stats[k][-1])
        
        return final_loss

    def get_attribute(self):
        """
        Called by pytorch_model_tools at the end of epoch to retrieve params/stats.
        Returns a list of tuples [('name', value)] or similar.
        Here we want to return a single object containing all our aggregation.
        """
        # Aggregate epoch stats
        aggregated = {}
        
        for cl_id, stats in self.epoch_stats.items():
            agg_cl = {}
            # Means of scalars
            for k in ['loss_total', 'loss_trans', 'mu0_term', 'entropy_pi', 'ce_loss']:
                if k in stats and stats[k]:
                    agg_cl[k] = np.mean(stats[k])
                else:
                    agg_cl[k] = 0.0
            
            # Means of vectors (stack then mean)
            for k in ['mu', 'pi', 'gains', 'scale']:
                if len(stats[k]) > 0:
                    stack = np.stack(stats[k])
                    agg_cl[k] = np.mean(stack, axis=0) # Average over batches
                else:
                    agg_cl[k] = None
                
            # Deltas: dict k->list of arrays. We want to concatenate all batch arrays for K to get distribution
            deltas_cat = {}
            if len(stats['deltas']) > 0:
                all_keys = stats['deltas'][0].keys()
                for key in all_keys:
                    arrays = [d[key] for d in stats['deltas'] if key in d]
                    if arrays:
                         deltas_cat[key] = np.concatenate(arrays)
            agg_cl['deltas'] = deltas_cat
            
            # Lk_weighted, SCORE_k, VIOL_k: list of dicts. Average per k.
            for metric in ['Lk_weighted', 'SCORE_k', 'VIOL_k']:
                avg_dict = {}
                if len(stats[metric]) > 0:
                    all_keys = stats[metric][0].keys()
                    for key in all_keys:
                        vals = [d[key] for d in stats[metric] if key in d]
                        avg_dict[key] = np.mean(vals)
                agg_cl[metric] = avg_dict

            # mu0 stats
            mu0_vals = [d['mu0'] for d in stats['mu0_stats']]
            agg_cl['mu0_mean'] = np.mean(mu0_vals) if mu0_vals else 0.0
            
            aggregated[cl_id] = agg_cl

        class DictWrapper:
            def __init__(self, d):
                self.d = d
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.d

        return [('ordinal_stats', DictWrapper(aggregated))]
        
    def plot_params(self, params_history, log_dir, best_epoch=None):
        """
        Génère des courbes d'évolution des paramètres.
        params_history: list of dicts [{'epoch': E, 'ordinal_stats': aggregated_stats}, ...]
        """
        import matplotlib.pyplot as plt
        import pathlib

        root_dir = pathlib.Path(log_dir) / 'ordinal_params'
        root_dir.mkdir(parents=True, exist_ok=True)

        # Re-organize data: cluster_id -> { metric -> [values over epochs] }
        cluster_series = {}
        
        # Determine if params_history is list or dict
        # Based on previous error, it is a list
        if isinstance(params_history, dict):
            # Sort by epoch if keys are epochs
            iterator = sorted(params_history.items())
        else:
            # Assume list of dicts
            iterator = []
            for entry in params_history:
                if 'epoch' in entry:
                    iterator.append((entry['epoch'], entry))
            iterator.sort(key=lambda x: x[0])
        
        for ep, entry in iterator:
            # Extract ordinal_stats
            # In pytorch_model_tools, it does: dict_params[name] = value
            # define name='ordinal_stats' in get_attribute
            if 'ordinal_stats' in entry:
                stats_container = entry['ordinal_stats']
                # Check if DictWrapper
                if hasattr(stats_container, 'd'):
                    ep_data = stats_container.d
                else:
                    ep_data = stats_container
            else:
                # Maybe params_history was passed as dict {cl_id: stats} if old format?
                # But let's assume new format from get_attribute
                continue

            if not ep_data:
                continue
            
            for cl_id, stats in ep_data.items():
                if cl_id not in cluster_series:
                    cluster_series[cl_id] = {
                        'epochs': [],
                        'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                        'pi': [], 'entropy_pi': [], 'ce_loss': [],
                        'mu': [],
                        'deltas': {}, # k -> {median:[], min:[], viol:[], neg:[]}
                        'gains': [], 'scale': [],
                        'Lk_weighted': {}, 'SCORE_k': {}, 'VIOL_k': {}, 'NEG_k': {},
                        'mu0_mean': [],
                        'delta_min_history': [], 'delta_max_history': []
                    }
                
                s = cluster_series[cl_id]
                s['epochs'].append(ep)
                s['loss_total'].append(stats.get('loss_total', 0.0))
                s['loss_trans'].append(stats.get('loss_trans', 0.0))
                s['mu0_term'].append(stats.get('mu0_term', 0.0))
                s['entropy_pi'].append(stats.get('entropy_pi', 0.0))
                s['ce_loss'].append(stats.get('ce_loss', 0.0))
                
                s['pi'].append(stats.get('pi', None))
                s['mu'].append(stats.get('mu', None))
                s['gains'].append(stats.get('gains', None))
                s['scale'].append(stats.get('scale', 1.0)) # Default scale 1.0
                s['mu0_mean'].append(stats.get('mu0_mean', 0.0))
                
                # Deltas stats
                # stats['deltas'] is dict k->array of deltas for this epoch-cluster
                # We want to compute scalar stats (median, min...) for the plot
                deltas_map = stats.get('deltas', {})
                if deltas_map:
                    for k, d_vals in deltas_map.items():
                        if k not in s['deltas']:
                            s['deltas'][k] = {'median':[], 'min':[], 'viol':[], 'neg':[]}
                        
                        if d_vals is not None and d_vals.size > 0:
                            s['deltas'][k]['median'].append(np.median(d_vals))
                            s['deltas'][k]['min'].append(np.min(d_vals))
                            viol = (d_vals < 0).mean()
                            s['deltas'][k]['viol'].append(viol)
                            neg = np.mean(np.log(1 + np.exp(-d_vals)))
                            s['deltas'][k]['neg'].append(neg)
                            
                            s['delta_min_history'].append(np.min(d_vals))
                            s['delta_max_history'].append(np.max(d_vals))
                        else:
                            s['deltas'][k]['median'].append(0)
                            s['deltas'][k]['min'].append(0)
                            s['deltas'][k]['viol'].append(0)
                            s['deltas'][k]['neg'].append(0)
                
                # Metrics per k
                for mKey in ['Lk_weighted', 'SCORE_k', 'VIOL_k', 'NEG_k']:
                    mDict = stats.get(mKey, {})
                    if mDict:
                        for k, val in mDict.items():
                            if k not in s[mKey]:
                                s[mKey][k] = []
                            s[mKey][k].append(val)
                # Ensure all k have same length (fill missing with nan or 0)
                # (For simplicity we assume strict structure)

        # Now Plot per cluster
        for cl_id, series in cluster_series.items():
            cl_dir = root_dir / str(cl_id)
            cl_dir.mkdir(parents=True, exist_ok=True)
            
            epochs = series['epochs']
            if not epochs:
                continue

            # 1) Loss components
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, series['loss_total'], label='Total Loss', linewidth=2)
                plt.plot(epochs, series['loss_trans'], label='Inequality Loss (Quadratic ReLU)', linestyle='--')
                plt.plot(epochs, series['mu0_term'], label='Mu0 Term', linestyle=':')
                if any(v != 0 for v in series['ce_loss']):
                    plt.plot(epochs, series['ce_loss'], label='CE Loss', color='purple', linestyle='-.')
                if best_epoch is not None:
                    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                plt.title(f'Cluster {cl_id} - Loss Components')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(cl_dir / '1_loss_components.png')
                plt.close()
            except Exception as e:
                print(f"Error plotting loss components for cluster {cl_id}: {e}")
                plt.close()

            # 2) Pi_s (Masses) & Entropy
            try:
                valid_pi = [p for p in series['pi'] if p is not None]
                if valid_pi:
                    pi_stack = np.stack(valid_pi) # (E, C)
                    if pi_stack.ndim == 1:
                         pi_stack = pi_stack.reshape(-1, 1)
                    
                    if pi_stack.ndim >= 2 and pi_stack.shape[0] == len(epochs):
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        ax2 = ax1.twinx()
                        C = pi_stack.shape[1]
                        for c in range(C):
                            ax1.plot(epochs, pi_stack[:, c], label=f'Class {c}')
                        ax2.plot(epochs, series['entropy_pi'], label='Entropy', color='black', linestyle='--')
                        
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Probability Mass')
                        ax2.set_ylabel('Entropy')
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        
                        if best_epoch is not None:
                            ax1.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                            lines1, labels1 = ax1.get_legend_handles_labels()

                        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                        plt.title(f'Cluster {cl_id} - Predicted Mass distribution (pi_s)')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(cl_dir / '2_pi_s_entropy.png')
                        plt.close()
                else:
                    # just plot entropy
                     plt.figure(figsize=(10, 6))
                     plt.plot(epochs, series['entropy_pi'], label='Entropy')
                     plt.title(f'Cluster {cl_id} - Entropy')
                     plt.savefig(cl_dir / '2_entropy_only.png')
                     plt.close()
            except Exception as e:
                print(f"Error plotting pi/entropy for cluster {cl_id}: {e}")
                plt.close()
            
            # 3) Mu(s)
            try:
                valid_mu = [m for m in series['mu'] if m is not None]
                if valid_mu:
                    mu_stack = np.stack(valid_mu)
                    if mu_stack.ndim == 1:
                        mu_stack = mu_stack.reshape(-1, 1)
                    
                    if mu_stack.ndim >= 2 and mu_stack.shape[0] == len(epochs):
                        plt.figure(figsize=(10, 6))
                        C_mu = mu_stack.shape[1]
                        for c in range(C_mu):
                            plt.plot(epochs, mu_stack[:, c], label=f'mu({c})')
                        if best_epoch is not None:
                            plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                        plt.title(f'Cluster {cl_id} - Mu(s) evolution')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(cl_dir / '3_mu_s.png')
                        plt.close()
            except Exception as e:
                print(f"Error plotting mu for cluster {cl_id}: {e}")
                plt.close()

            # 4) Deltas stats per k
            try:
                ks = sorted(series['deltas'].keys())
                if ks:
                    fig, axes = plt.subplots(len(ks), 4, figsize=(20, 3*len(ks)), sharex=True)
                    if len(ks) == 1: axes = axes[None, :] 
                    
                    for i, k in enumerate(ks):
                        dstats = series['deltas'][k]
                        if len(dstats['median']) == len(epochs):
                            axes[i, 0].plot(epochs, dstats['median'], color='blue')
                            axes[i, 0].set_title(f'k={k} Median Delta')
                            axes[i, 0].grid(True)
                            
                            axes[i, 1].plot(epochs, dstats['min'], color='red')
                            axes[i, 1].set_title(f'k={k} Min Delta')
                            axes[i, 1].grid(True)
                            
                            axes[i, 2].plot(epochs, dstats['viol'], color='orange')
                            axes[i, 2].set_title(f'k={k} Violation Rate (<0)')
                            axes[i, 2].set_ylim(-0.1, 1.1)
                            axes[i, 2].grid(True)
                            
                            axes[i, 3].plot(epochs, dstats['neg'], color='purple')
                            axes[i, 3].set_title(f'k={k} Mean NEG (Softplus magnitude)')
                            axes[i, 3].grid(True)
                    
                    if best_epoch is not None:
                        for ax_row in axes:
                            for ax in ax_row:
                                ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5)

                    plt.tight_layout()
                    plt.savefig(cl_dir / '4_deltas_stats.png')
                    plt.close()
            except Exception as e:
                print(f"Error plotting deltas for cluster {cl_id}: {e}")
                plt.close()

            # 5) Metrics per k (Lk, SCORE_k, VIOL_k, NEG_k)
            try:
                fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
                
                # Helper to plot dict of k->vals
                def plot_k_lines(ax, data_dict, title):
                    did_plot = False
                    for k, vals in data_dict.items():
                        if len(vals) == len(epochs):
                             ax.plot(epochs, vals, label=f'k={k}')
                             did_plot = True
                    ax.set_title(title)
                    if did_plot: ax.legend()
                    ax.grid(True)
                
                plot_k_lines(axes[0], series['Lk_weighted'], 'Weighted Level Losses (wk * Lk)')
                plot_k_lines(axes[1], series['SCORE_k'], 'SCORE_k')
                plot_k_lines(axes[2], series['VIOL_k'], 'VIOL_k')
                axes[2].set_ylim(-0.1, 1.1)
                plot_k_lines(axes[3], series['NEG_k'], 'NEG_k (Softplus magnitude)')

                if best_epoch is not None:
                    for ax in axes:
                        ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5)

                plt.tight_layout()
                plt.savefig(cl_dir / '5_metrics_per_k.png')
                plt.close()
            except Exception as e:
                 print(f"Error plotting metrics per k for cluster {cl_id}: {e}")
                 plt.close()

            # 6) Counts/Lambda & Gains
            try:
                valid_gains = [g for g in series['gains'] if g is not None]
                if valid_gains and len(valid_gains) == len(epochs):
                     gains_stack = np.stack(valid_gains) # (E, C-1) ?
                     scale_stack = np.array(series['scale'])
                     
                     # Check shapes
                     if gains_stack.ndim == 1:
                         gains_stack = gains_stack.reshape(-1, 1)
                     
                     if gains_stack.shape[0] == scale_stack.shape[0]:
                         # Convert to absolute scale for easier reading
                         abs_gains_stack = gains_stack * scale_stack[:, None]

                         fig, ax1 = plt.subplots(figsize=(10, 6))
                         ax2 = ax1.twinx()
                        
                         for i in range(abs_gains_stack.shape[1]):
                            ax1.plot(epochs, abs_gains_stack[:, i], label=f'Abs Gain {i}', linestyle='-')
                        
                         ax2.plot(epochs, scale_stack, label='Scale', color='black', linewidth=2, linestyle='--')
                        
                         ax1.set_xlabel('Epoch')
                         ax1.set_ylabel('Absolute Gains (Gain * Scale)')
                         ax2.set_ylabel('Global Scale')
                        
                         lines1, labels1 = ax1.get_legend_handles_labels()
                         lines2, labels2 = ax2.get_legend_handles_labels()
                         if best_epoch is not None:
                            ax1.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                            lines1, labels1 = ax1.get_legend_handles_labels()

                         ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                        
                         plt.title(f'Cluster {cl_id} - Gains & Scale Evolution')
                         plt.grid(True, alpha=0.3)
                         plt.savefig(cl_dir / '6_gains_scale.png')
                         plt.close()
            except Exception as e:
                 print(f"Error plotting gains/scale for cluster {cl_id}: {e}")
                 plt.close()
            
            # 8) Mu0 stats
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, series['mu0_mean'], label='Mu0 Mean')
                plt.title(f'Cluster {cl_id} - Mu(0) Mean')
                plt.xlabel('Epoch')
                plt.ylabel('Mu(0)')
                if best_epoch is not None:
                    plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
                plt.grid(True, alpha=0.3)
                plt.savefig(cl_dir / '8_mu0_stats.png')
                plt.close()
            except Exception as e:
                print(f"Error plotting mu0 stats for cluster {cl_id}: {e}")
                plt.close()

            # 9) Hyperparameter Effects
            try:
                # Determine range from history
                if len(series['delta_max_history']) > 0 and len(series['delta_min_history']) > 0:
                    global_min = np.min(series['delta_min_history'])
                    global_max = np.max(series['delta_max_history'])
                    # Add 10% margin
                    span = global_max - global_min
                    if span < 1e-6: span = 1.0
                    global_min -= 0.1 * span
                    global_max += 0.1 * span
                else:
                    global_min, global_max = -2.0, 2.0

                self.plot_hyperparams_effects(cl_dir, delta_range=(global_min, global_max))
            except Exception as e:
                 print(f"Error plotting hyperparams effects for cluster {cl_id}: {e}")

    def plot_hyperparams_effects(self, dir_output, delta_range=(-2.0, 2.0)):
        """
        Plots the effect of betasoftmin on SoftMin and tviolation on Violation Penalty.
        Generates 'hyperparams_effects.png'.
        delta_range: (min, max) tuple to define the x-axis.
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        
        # Range of deltas
        dmin, dmax = delta_range
        # Ensure 0 is included if possible or at least cover it if range is close
        if dmax < 0: dmax = 0.5
        if dmin > 0: dmin = -0.5
        
        deltas = np.linspace(dmin, dmax, 400)
        deltas_t = torch.tensor(deltas, dtype=torch.float32)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- 1) SoftMin Effect (beta) ---
        # SoftMin is usually applied to a set. Here we visualize SoftMin([0, delta])
        # which approximates min(0, delta).
        betas = [1.0, 2.0, 5.0, 10.0, 20.0]
        # Include current beta
        if self.beta not in betas:
            betas.append(self.beta)
        betas = sorted(list(set(betas)))
        
        # Reference: true min(0, delta)
        min_vals = np.minimum(0, deltas)
        axes[0].plot(deltas, min_vals, 'k--', label='min(0, x)', linewidth=2, alpha=0.5)
        
        for b in betas:
            # SoftMin([0, x]) = -(1/b) * log( exp(0) + exp(-b*x) ) = -(1/b) * log( 1 + exp(-b*x) )
            # We compute it using torch for stability
            # But wait, self._softmin computes softmin of a vector. 
            # If we pass [0, x], it reduces those 2.
            
            # Vectorized calculation for plotting:
            # y = -(1/b) * log( 1 + exp(-b * delta) )
            # equals -softplus(-b * delta) / b ? No.
            # log(1 + exp(z)) = softplus(z).
            # So -(1/b) * softplus(-b * delta).
            
            y = -(1.0 / b) * F.softplus(-b * deltas_t)
            
            style = '-' if b == self.beta else ':'
            width = 2 if b == self.beta else 1
            label = f'beta={b}' + (' (current)' if b == self.beta else '')
            axes[0].plot(deltas, y.numpy(), style, label=label, linewidth=width)
            
        axes[0].set_title('SoftMin Approximation of min(0, delta)')
        axes[0].set_xlabel('delta')
        axes[0].set_ylabel('SoftMin(0, delta)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # --- 2) Violation Penalty Effect (tviolation) ---
        # Violation = Sigmoid(-delta / t)
        ts = [0.01, 0.05, 0.1, 0.5, 1.0]
        # Include current t
        if self.t not in ts:
            ts.append(self.t)
        ts = sorted(list(set(ts)))
        
        for t in ts:
            # Sigmoid(-delta / t)
            y = torch.sigmoid(-deltas_t / t)
            
            style = '-' if t == self.t else ':'
            width = 2 if t == self.t else 1
            label = f't={t}' + (' (current)' if t == self.t else '')
            axes[1].plot(deltas, y.numpy(), style, label=label, linewidth=width)
            
        axes[1].set_title('Violation Penalty (Sigmoid(-delta/t))')
        axes[1].set_xlabel('delta')
        axes[1].set_ylabel('Penalty')
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(dir_output / 'hyperparams_effects.png')
        plt.close()