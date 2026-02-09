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
        tviolation=0.1,
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
        lambdamu0 = 0.1,
        lambdaentropy = 0.0,
        wmed=1.0,
        wmin=1.0,
        wneg=1.0,
        wviol=1.0
    ):
        super().__init__()
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

        # Component weights
        self.wmed = float(wmed)
        self.wmin = float(wmin)
        self.wneg = float(wneg)
        self.wviol = float(wviol)

        self._log("Ordinal Loss Config:", self.get_config())

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
            "id": self.id
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
            mk = float(np.median(margins_k))  # médiane robuste :contentReference[oaicite:0]{index=0}
            #wk[k] = (1.0 / (mk + eps)) * self.wk.get(k, 1.0)
            wk[k] = self.wk.get(k, 1.0)
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
            spread = q95 - q50
        else:
            q50 = float(np.quantile(y_pos, 0.5)) if y_pos.size > 0 else 0.0
            q95 = float(np.quantile(y_pos, 0.95)) if y_pos.size > 0 else 0.0
            spread = float(np.std(y_pos)) if y_pos.size > 1 else 0.0

        spread = max(spread, 1e-6)
        floor = max(0.0, self.gainsfloorfrac)
        base = self.gainsalpha * (diffs / spread)
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
        print("[TRANSITION AND K]:", P)
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

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.epoch_stats = {}

    def forward(self, logits, y_cont, clusters_ids, sample_weight=None):
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

        for d in torch.unique(clusters_ids):
            idx = torch.nonzero(clusters_ids == d, as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue

            p = probs[idx]
            y = y_cont[idx]
            sw_d = sw[idx] if sw is not None else None

            # --- 4) Mu(s) & 5) Pi(s)
            
            # --- 5) Masse de proba pi_s ---
            # --- 5) Masse de proba pi_s ---
            if sw_d is not None:
                p_weighted = p * sw_d.unsqueeze(1)
                pi_s = p_weighted.sum(dim=0) / sw_d.sum().clamp_min(self.eps)
            else:
                pi_s = p.mean(dim=0)
            
            mu = self._mu_soft(p, y, sw=sw_d)  # (C,)
            
            # Entropy of pi_s
            pi_log_pi = pi_s * torch.log(pi_s.clamp_min(1e-9))
            entropy_pi = -pi_log_pi.sum()

            cl_id = d.item() if torch.is_tensor(d) else d
            # Init stats for this cluster if needed
            if cl_id not in self.epoch_stats:
                self.epoch_stats[cl_id] = {
                    'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                    'mu': [],        # list of (C,) arrays
                    'pi': [],        # list of (C,) arrays (masses)
                    'entropy_pi': [],
                    'deltas': [],    # list of dicts {k: array}
                    'gains': [],     # list of (C-1,) values
                    'scale': [],
                    'Lk_weighted': [], # list of dict {k: val}
                    'SCORE_k': [],     # list of dict {k: val}
                    'SCORE_k': [],     # list of dict {k: val}
                    'VIOL_k': [],      # list of dict {k: val}
                    'NEG_k': [],       # list of dict {k: val}
                    'mu0_stats': [],   # list of dict
                }

            gains_adj = self._get_cluster_gains(cl_id, mu.device, mu.dtype)  # (C-1,)
            wk_dict = self._get_cluster_wk(cl_id)
            scale = self._get_cluster_scale(cl_id, mu.device, mu.dtype)

            # Decoupled from wk: use raw lambda_mu0 to allow separation (wk) to grow without strengthening anchor
            lam0 = self.lambdamu0
            cluster_loss = logits.new_tensor(0.0)
            cluster_w = logits.new_tensor(0.0)
            
            # --- 8) Terme mu(0) ---
            mu0_val = mu[0]
            mu0_term = F.softplus(mu0_val / (scale + self.eps))
            cluster_loss = cluster_loss + lam0 * mu0_term
            cluster_w = cluster_w + lam0
            
            # --- Entropy Regularization ---
            # Maximize entropy => Minimize -entropy
            if self.lambdaentropy > 0:
                cluster_loss = cluster_loss - self.lambdaentropy * entropy_pi
                # We do not increase cluster_w to avoid diluting other terms, 
                # or we consider this a regularization term added on top.

            
            # Stats mu0
            stats_loss_trans = logits.new_tensor(0.0)
            stats_w_trans = logits.new_tensor(0.0)
            
            # Store Lk weighted for this batch
            Lk_w_batch = {}
            SCORE_k_batch = {}
            Lk_w_batch = {}
            SCORE_k_batch = {}
            VIOL_k_batch = {}
            NEG_k_batch = {}

            # Save deltas for this batch
            deltas_batch = {}

            # SCORE_k with margined deltas
            for k, pairs in self.P.items():
                if not pairs:
                    continue

                raw = torch.stack([mu[b] - mu[a] for (a, b) in pairs], dim=0)
                margins = torch.stack([gains_adj[a:b].sum() for (a, b) in pairs], dim=0)
                raw = raw / (scale + self.eps)
                deltas = raw - margins
                
                # --- 3) Stats deltas ---
                deltas_batch[k] = deltas.detach().cpu().numpy()

                MINk = self._softmin(deltas)
                MEDk = self._soft_median(deltas)
                VIOLk = torch.sigmoid(-deltas / self.t).mean()
                NEGk = F.softplus(-deltas).mean()

                NEGk = F.softplus(-deltas).mean()

                SCOREk = (self.wmed * MEDk + self.wmin * MINk) - self.wneg * NEGk * (1.0 + self.wviol * VIOLk)
                Lk = F.softplus(-SCOREk)

                w = float(wk_dict.get(k, 1.0))
                cluster_loss = cluster_loss + w * Lk
                cluster_w = cluster_w + w
                
                stats_loss_trans = stats_loss_trans + w * Lk
                stats_w_trans = stats_w_trans + w
                
                Lk_w_batch[k] = (w * Lk).detach().item()
                SCORE_k_batch[k] = SCOREk.detach().item()
                SCORE_k_batch[k] = SCOREk.detach().item()
                VIOL_k_batch[k] = VIOLk.detach().item()
                NEG_k_batch[k] = NEGk.detach().item()

            if cluster_w > 0:
                total_loss = total_loss + cluster_loss
                total_w = total_w + cluster_w
            
            # Keep track of everything (detached)
            est = self.epoch_stats[cl_id]
            
            est['loss_total'].append((cluster_loss / cluster_w.clamp_min(self.eps)).detach().item())
            est['loss_trans'].append((stats_loss_trans / stats_w_trans.clamp_min(self.eps)).detach().item())
            est['mu0_term'].append((lam0 * mu0_term).detach().item())
            
            est['mu'].append(mu.detach().cpu().numpy())
            est['pi'].append(pi_s.detach().cpu().numpy())
            est['entropy_pi'].append(entropy_pi.detach().item())
            
            est['gains'].append(gains_adj.detach().cpu().numpy())
            est['scale'].append(scale.detach().cpu().numpy())
            
            est['Lk_weighted'].append(Lk_w_batch)
            est['SCORE_k'].append(SCORE_k_batch)
            est['Lk_weighted'].append(Lk_w_batch)
            est['SCORE_k'].append(SCORE_k_batch)
            est['VIOL_k'].append(VIOL_k_batch)
            est['NEG_k'].append(NEG_k_batch)
            
            est['deltas'].append(deltas_batch)
            
            # mu0 stats: correlation with y? batch too small?
            # store simple mu0 value
            est['mu0_stats'].append({'mu0': mu0_val.detach().item()})

        if total_w.abs() < self.eps:
            return logits.new_tensor(0.0)

        return total_loss / total_w

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
            for k in ['loss_total', 'loss_trans', 'mu0_term', 'entropy_pi']:
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
        
    def plot_params(self, param_history, dir_output, best_epoch=None):
        """
        param_history: list of dicts {'epoch': e, 'ordinal_stats': aggregated_stats_dict}
        """
        import matplotlib.pyplot as plt
        import os

        # Organiser les données par cluster
        cluster_series = {}

        for entry in param_history:
            epoch = entry['epoch']
            if 'ordinal_stats' not in entry:
                continue
            stats_by_cluster = entry['ordinal_stats'].d if hasattr(entry['ordinal_stats'], 'd') else entry['ordinal_stats']
            
            for cl_id, agg in stats_by_cluster.items():
                if cl_id not in cluster_series:
                    cluster_series[cl_id] = {
                        'epochs': [],
                        'loss_total': [], 'loss_trans': [], 'mu0_term': [],
                        'mu': [], 'pi': [], 'gains': [], 'scale': [],
                        'entropy_pi': [], 'mu0_mean': [],
                        'deltas': {}, 'Lk_weighted': {}, 'SCORE_k': {}, 'VIOL_k': {}, 'NEG_k': {}
                    }
                
                cs = cluster_series[cl_id]
                cs['epochs'].append(epoch)
                cs['loss_total'].append(agg.get('loss_total', 0))
                cs['loss_trans'].append(agg.get('loss_trans', 0))
                cs['mu0_term'].append(agg.get('mu0_term', 0))
                cs['entropy_pi'].append(agg.get('entropy_pi', 0))
                cs['mu0_mean'].append(agg.get('mu0_mean', 0))
                
                cs['mu'].append(agg['mu'])
                cs['pi'].append(agg['pi'])
                cs['gains'].append(agg['gains'])
                cs['scale'].append(agg['scale'])
                
                # Deltas: Compute summaries
                for k, arr in agg['deltas'].items():
                    if k not in cs['deltas']:
                        cs['deltas'][k] = {'median': [], 'min': [], 'viol': [], 'neg': [], 'sigmoid': []}
                    
                    if arr is not None and arr.size > 0:
                        cs['deltas'][k]['median'].append(np.median(arr))
                        cs['deltas'][k]['min'].append(np.min(arr))
                        cs['deltas'][k]['viol'].append(np.mean(arr < 0))
                        cs['deltas'][k]['sigmoid'].append(np.mean(1.0 / (1.0 + np.exp(arr/self.t)))) 
                    else:
                        cs['deltas'][k]['median'].append(np.nan)
                        cs['deltas'][k]['min'].append(np.nan)
                        cs['deltas'][k]['viol'].append(np.nan)
                        cs['deltas'][k]['sigmoid'].append(np.nan)

                # Dict metrics
                for metric in ['Lk_weighted', 'SCORE_k', 'VIOL_k', 'NEG_k']:
                    if metric not in agg: continue
                    for k, val in agg[metric].items():
                        if k not in cs[metric]:
                            cs[metric][k] = []
                        cs[metric][k].append(val)

        # Plotting per cluster
        for cl_id, series in cluster_series.items():
            if len(series['epochs']) == 0:
                continue

            cl_dir = dir_output / f"cluster_{cl_id}"
            os.makedirs(cl_dir, exist_ok=True)
            epochs = series['epochs']

            # 1) Loss components
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, series['loss_total'], label='Total Loss', linewidth=2)
            plt.plot(epochs, series['loss_trans'], label='Transitional Loss', linestyle='--')
            plt.plot(epochs, series['mu0_term'], label='Mu0 Term', linestyle=':')
            if best_epoch is not None:
                plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
            plt.title(f'Cluster {cl_id} - Loss Components')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(cl_dir / '1_loss_components.png')
            plt.close()

            # 2) Pi_s (Masses) & Entropy
            pi_stack = np.stack(series['pi']) # (E, C)
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
                # Re-gather handles to include best epoch if added to ax1
                lines1, labels1 = ax1.get_legend_handles_labels()

            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            plt.title(f'Cluster {cl_id} - Predicted Mass distribution (pi_s)')
            plt.grid(True, alpha=0.3)
            plt.savefig(cl_dir / '2_pi_s_entropy.png')
            plt.close()

            # 3) Mu(s) & Gaps
            mu_stack = np.stack(series['mu']) # (E, C)
            plt.figure(figsize=(10, 6))
            for c in range(C):
                plt.plot(epochs, mu_stack[:, c], label=f'mu({c})')
            if best_epoch is not None:
                plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
            plt.title(f'Cluster {cl_id} - Mu(s) evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(cl_dir / '3_mu_s.png')
            plt.close()

            # 4) Deltas stats per k
            ks = sorted(series['deltas'].keys())
            if ks:
                fig, axes = plt.subplots(len(ks), 4, figsize=(20, 3*len(ks)), sharex=True)
                if len(ks) == 1: axes = axes[None, :] 
                
                for i, k in enumerate(ks):
                    dstats = series['deltas'][k]
                    # Median
                    axes[i, 0].plot(epochs, dstats['median'], color='blue')
                    axes[i, 0].set_title(f'k={k} Median Delta')
                    axes[i, 0].grid(True)
                    # Min
                    axes[i, 1].plot(epochs, dstats['min'], color='red')
                    axes[i, 1].set_title(f'k={k} Min Delta')
                    axes[i, 1].grid(True)
                    # Violation %
                    axes[i, 2].plot(epochs, dstats['viol'], color='orange')
                    axes[i, 2].set_title(f'k={k} Violation Rate (<0)')
                    axes[i, 2].set_ylim(-0.1, 1.1)
                    axes[i, 2].grid(True)
                    # Sigmoid avg
                    axes[i, 3].plot(epochs, dstats['sigmoid'], color='purple')
                    axes[i, 3].set_title(f'k={k} Mean Sigmoid Viol')
                    axes[i, 3].grid(True)
                
                
                if best_epoch is not None:
                    for ax_row in axes:
                        for ax in ax_row:
                            ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5)

                plt.tight_layout()
                plt.savefig(cl_dir / '4_deltas_stats.png')
                plt.close()

            # 5) Metrics per k (Lk, SCORE_k, VIOL_k, NEG_k)
            # Need subplots for these
            fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
            
            for k, vals in series['Lk_weighted'].items():
                axes[0].plot(epochs, vals, label=f'k={k}')
            axes[0].set_title('Weighted Level Losses (wk * Lk)')
            axes[0].legend()
            axes[0].grid(True)
            
            for k, vals in series['SCORE_k'].items():
                axes[1].plot(epochs, vals, label=f'k={k}')
            axes[1].set_title('SCORE_k')
            axes[1].legend()
            axes[1].grid(True)
            
            for k, vals in series['VIOL_k'].items():
                axes[2].plot(epochs, vals, label=f'k={k}')
            axes[2].set_title('VIOL_k')
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].legend()
            axes[2].grid(True)

            axes[2].legend()
            axes[2].grid(True)
            
            for k, vals in series['NEG_k'].items():
                axes[3].plot(epochs, vals, label=f'k={k}')
            axes[3].set_title('NEG_k (Softplus penalty)')
            axes[3].legend()
            axes[3].grid(True)

            if best_epoch is not None:
                for ax in axes:
                    ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(cl_dir / '5_metrics_per_k.png')
            plt.close()

            # 6) Counts/Lambda & Gains
            gains_stack = np.stack(series['gains']) # (E, C-1)
            scale_stack = np.array(series['scale'])
            
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
                # Re-gather handles to include best epoch if added to ax1
                lines1, labels1 = ax1.get_legend_handles_labels()

            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title(f'Cluster {cl_id} - Gains & Scale Evolution')
            plt.grid(True, alpha=0.3)
            plt.savefig(cl_dir / '6_gains_scale.png')
            plt.close()
            
            # 8) Mu0 stats
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