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
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100, num_classes=5):
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
        C: float = 0.5, # This is the lambda parameter (weight for QWK)
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
            'total_loss' : (1-C) * focal_result + C * wk_result,
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
    
    def plot_params(self, mcewk_logs, dir_output):
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

    def forward(self, inputs, targets, cluster_ids, sample_weight=None):
        """
        inputs:      (N, C) logits
        targets:     (N,) indices de classe (0..C-1) ou ignore_index
        cluster_ids: (N,) identifiant departement (int)
        sample_weight: (N,) optionnel, poids par observation (pas par paire)

        retourne: scalaire si reduction != 'none', sinon une loss scalaire aussi
                  (on ne renvoie pas une loss par echantillon car la loss est par paires)
        """
        if inputs.dim() != 2:
            raise ValueError(f"inputs doit etre (N,C). Recu: {tuple(inputs.shape)}")
        if targets.dim() != 1 or cluster_ids.dim() != 1:
            raise ValueError("targets et cluster_ids doivent etre des tenseurs 1D (N,).")
        if targets.size(0) != inputs.size(0) or cluster_ids.size(0) != inputs.size(0):
            raise ValueError("inputs, targets, cluster_ids doivent avoir le meme N.")

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
        deps_v = cluster_ids[valid_mask]
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