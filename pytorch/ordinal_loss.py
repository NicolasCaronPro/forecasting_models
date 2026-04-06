import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.loss_utils import *
from forecasting_models.pytorch.classification_loss import WeightedCrossEntropyLoss
from typing import List
from forecasting_models.pytorch.distribution_loss import PredictdEGPDLossTrunc

###################################### Ordinality ##########################################

class DictWrapper:
    def __init__(self, d):
        self.d = d
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.d

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
    
    def get_learnable_parameterss(self):
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
    
    def get_learnable_parameterss(self):
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
    
    def get_learnable_parameterss(self):
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
    
    def get_learnable_parameterss(self):
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

    def get_learnable_parameterss(self):
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
        minbinn=1,
        gainsalpha=1.0,
        gainsalpha0=1.0,
        gainsfloorfrac=0.1,
        enforcegainmonotone=False,
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
        cetype='crossentropy',  # 'crossentropy' or 'focal'
        alpha=0.25
    ):
        super().__init__()
        self.lambdadir = float(lambdadir)
        self.diralpha = float(diralpha)
        self.lambdace = float(lambdace)
        self.cetype = str(cetype).lower()

        # CE Loss Setup - choose between focal and cross-entropy
        if self.cetype == 'focal':
            self.ce_loss = FocalLoss(gamma=2.0, alpha=alpha, reduction='mean')
        elif self.cetype == 'crossentropy':
            self.ce_loss = WeightedCrossEntropyLoss(num_classes=num_classes)
        elif self.cetype == "bce":
            self.ce_loss = BCELoss(num_classes=num_classes)
        elif self.cetype == 'wk':
            self.ce_loss = WKLoss(num_classes=num_classes)
        else:
            raise ValueError(f"cetype must be 'focal', 'crossentropy', 'bce', 'wk' or 'pdegpd', got '{cetype}'")
                        
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
        self.gainsalpha0 = float(gainsalpha0)
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
            "gainsalpha0": self.gainsalpha0,
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
    def _adaptive_positive_thresholds(self, y_pos, qe_pos, min_pos=3):
        """
        Adapted from QuantileRiskZerosHandle algorithm.
        
        qe_pos: list/tuple of target quantiles (len = C-2) - used to determine n_thresholds.

        Algorithm:
        - Start at q_start (first value in qe_pos, or 0.5)
        - Increment by q_step (0.05) until we have C-2 distinct thresholds
        - Cap at q_max = dynamic (based on sample size to ensure min_pos samples in last bin)
        - Ensure each threshold is strictly different from the previous

        Returns:
          thresholds: np.float32 (C-2,)
          used_q: np.float32 (C-2,)
          forced_flat: bool (True if we couldn't find enough distinct thresholds)
        """
        n_thresholds = len(qe_pos)  # Should be C-2
        q_start = float(qe_pos[0]) if qe_pos else 0.5
        
        # Dynamic q_max calculation
        n_pos = len(y_pos)
        if n_pos > 0:
            # We want roughly min_pos samples in the tail (> q_max)
            # q_max should be approx 1 - (min_pos / n_pos)
            # We use a slight margin (min_pos - 0.5)
            q_max_limit = 1.0 - (max(1.0, float(min_pos) - 0.5) / (n_pos + 1e-9))
            q_max = min(0.999, q_max_limit)
            q_max = max(0.9, q_max) # Don't go below 0.9
        else:
            q_max = 0.999
            
        q_step = 0.05

        thresholds = []
        used_q = []
        q = q_start

        # First threshold
        if q < 1.0:
            first_val = float(np.quantile(y_pos, q))
            thresholds.append(first_val)
            used_q.append(q)

        # Subsequent thresholds - increment until we have enough distinct values
        while len(thresholds) < n_thresholds and q < 1.0:
            q = min(q + q_step, q_max)
            val = float(np.quantile(y_pos, q))

            # Only add if strictly different from previous
            if val != thresholds[-1]:
                thresholds.append(val)
                used_q.append(q)

            if q >= q_max:
                break

        # Check if we found enough distinct thresholds
        forced_flat = len(thresholds) < n_thresholds

        # If we couldn't find enough thresholds, pad with the last value
        while len(thresholds) < n_thresholds:
            thresholds.append(thresholds[-1] if thresholds else 0.0)
            used_q.append(q_max)

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
    def _compute_bins_given_thresholds(
        self,
        y_sub,
        qs_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=1
    ):
        """
        Calculates bin means and counts for a given set of thresholds.
        """
        y_sub = y_sub[np.isfinite(y_sub)]
        if y_sub.size == 0:
            return None, False, None
            
        y0 = y_sub[y_sub <= 0]
        y_pos = y_sub[y_sub > 0]
        
        bins = []
        bins.append(y0)
        
        if y_pos.size > 0:
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
        else:
            # If no positive values, all positive bins are empty
            for _ in range(len(qs_pos) + 1):
                bins.append(np.array([], dtype=y_sub.dtype))

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
                
        return bin_means, bool(ok), counts

    def _compute_bins_and_ok_positive_quantiles(
        self,
        y_sub,
        qe_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=1,
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
        y_sub_fin = y_sub[np.isfinite(y_sub)]
        if y_sub_fin.size == 0:
            return None, False, None, None, False, None

        y_pos = y_sub_fin[y_sub_fin > 0]
        if y_pos.size == 0:
            return None, False, None, None, False, None

        qs_pos, used_q, forced_flat = self._adaptive_positive_thresholds(y_pos, qe_pos, min_pos=min_pos_per_bin)

        bin_means, ok, counts = self._compute_bins_given_thresholds(
            y_sub_fin, qs_pos, minB, 
            require_positive_bins=require_positive_bins, 
            min_pos_per_bin=min_pos_per_bin
        )

        if bin_means is None:
            return None, False, qs_pos, used_q, forced_flat, None

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

        if gains.size > 0:
            gains[0] *= self.gainsalpha0

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
        min_pos_per_bin=None,
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
        if min_pos_per_bin is None:
            min_pos_per_bin = minB

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
            # robust fallback when bins cannot be calculated
            self._log("[GLOBAL] Cannot calculate bins, using fallback scale-based gains")
            yfin = y[np.isfinite(y)]
            scale = float(np.std(yfin)) if yfin.size > 1 else 1.0
            global_g = np.full(self.C - 1, 0.05 * scale, dtype=np.float32)
            global_spread = float(scale)
            self._log("[GLOBAL] fallback scale-based gains:", global_g)
            # Create dummy thresholds for consistency
            qs_g = np.linspace(0.5, 0.95, self.C - 2, dtype=np.float32)
            self.global_scale = torch.tensor(global_spread, dtype=torch.float32, device=self.global_gains.device)
            self.register_buffer('global_thresholds', torch.tensor(qs_g, dtype=torch.float32))

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
            
            # FORCE GLOBAL DISCRETIZATION: use global thresholds qs_g
            bm, ok, cnt = self._compute_bins_given_thresholds(
                y_cl, qs_g, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin
            )
            qs, usedq, flat = qs_g, usedq_g, flat_g

            if ok:
                g_cl, spread_cl = self._gains_from_bin_means(bm, y_cl, tag="CL_%s" % str(cl))
            else:
                pooled = False
                if cluster_group is not None and idx_by_group is not None:
                    g_id = cluster_group.get(cl, None)
                    if g_id is not None and g_id in idx_by_group:
                        idx_pool = idx_by_group[g_id]
                        y_pool = y[idx_pool]
                        self._log(f"[CL {cl}] pool by similar group {g_id} (n_pool={len(idx_pool)}) (global discr)")

                        bm2, ok2, cnt2 = self._compute_bins_given_thresholds(
                            y_pool, qs_g, minB,
                            require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin
                        )
                        if ok2:
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

                if pooled:
                    self._log(f"[CL {cl}] used pooled gains (global discr)")
                else:
                    self._log(f"[CL {cl}] used global gains (global discr)")

            if self.enforcegainmonotone:
                g_cl = np.maximum.accumulate(np.asarray(g_cl, dtype=np.float32)).astype(np.float32)
            
            self.gain_k[cl] = torch.tensor(g_cl, dtype=torch.float32, device=self.global_gains.device)
            self.scale_k[cl] = torch.tensor(spread_cl, dtype=torch.float32, device=self.global_gains.device)
            
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
        minbinn=1,
        gainsalpha=1.0,
        gainsalpha0=1.0,
        gainsfloorfrac=0.1,
        enforcegainmonotone=False,
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
        self.gainsalpha0 = float(gainsalpha0)
        self.gainsfloorfrac = float(gainsfloorfrac)
        self.enforcegainmonotone = bool(enforcegainmonotone)
        self.enablelogs = bool(enablelogs)
        self.lambda_mse = float(lambda_mse)
        
        # Buffers pour gains et scales
        self.gain_k = {}
        self.register_buffer("global_gains", torch.zeros(self.C - 1, dtype=torch.float32))
        self.register_buffer("global_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("global_thresholds", torch.zeros(self.C - 2, dtype=torch.float32))
        
        self.call_preprocess = False
    
    def get_config(self):
        return {
            "numclasses": self.C,
            "mushrinkalpha": self.mushrinkalpha,
            "eps": self.eps,
            "quantileedges": self.quantileedges,
            "minbinn": self.minbinn,
            "gainsalpha": self.gainsalpha,
            "gainsalpha0": self.gainsalpha0,
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
    
    def _adaptive_positive_thresholds(self, y_pos, qe_pos, min_pos=3):
        """
        Adapted from QuantileRiskZerosHandle algorithm.
        
        qe_pos: list/tuple of target quantiles (len = C-2) - used to determine n_thresholds.
        
        Algorithm:
        - Start at q_start (first value in qe_pos, or 0.5)
        - Increment by q_step (0.05) until we have C-2 distinct thresholds
        - Cap at q_max = dynamic (based on sample size to ensure min_pos samples in last bin)
        - Ensure each threshold is strictly different from the previous

        Returns:
          thresholds: np.float32 (C-2,)
          used_q: np.float32 (C-2,)
          forced_flat: bool (True if we couldn't find enough distinct thresholds)
        """
        n_thresholds = len(qe_pos)  # Should be C-2
        q_start = float(qe_pos[0]) if qe_pos else 0.5
        
        # Dynamic q_max calculation
        n_pos = len(y_pos)
        if n_pos > 0:
            # We want roughly min_pos samples in the tail (> q_max)
            # q_max should be approx 1 - (min_pos / n_pos)
            # We use a slight margin (min_pos - 0.5)
            q_max_limit = 1.0 - (max(1.0, float(min_pos) - 0.5) / (n_pos + 1e-9))
            q_max = min(0.999, q_max_limit)
            q_max = max(0.9, q_max) # Don't go below 0.9
        else:
            q_max = 0.999
            
        q_step = 0.05

        thresholds = []
        used_q = []
        q = q_start

        # First threshold
        if q < 1.0:
            first_val = float(np.quantile(y_pos, q))
            thresholds.append(first_val)
            used_q.append(q)

        # Subsequent thresholds - increment until we have enough distinct values
        while len(thresholds) < n_thresholds and q < 1.0:
            q = min(q + q_step, q_max)
            val = float(np.quantile(y_pos, q))

            # Only add if strictly different from previous
            if val != thresholds[-1]:
                thresholds.append(val)
                used_q.append(q)

            if q >= q_max:
                break

        # Check if we found enough distinct thresholds
        forced_flat = len(thresholds) < n_thresholds

        # If we couldn't find enough thresholds, pad with the last value
        while len(thresholds) < n_thresholds:
            thresholds.append(thresholds[-1] if thresholds else 0.0)
            used_q.append(q_max)

        return (
            np.array(thresholds, dtype=np.float32),
            np.array(used_q, dtype=np.float32),
            bool(forced_flat),
        )
    
    def _compute_bins_given_thresholds(
        self,
        y_sub,
        qs_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=1
    ):
        """Calculates bin means and counts for a given set of thresholds."""
        y_sub = y_sub[np.isfinite(y_sub)]
        if y_sub.size == 0:
            return None, False, None
            
        y0 = y_sub[y_sub <= 0]
        y_pos = y_sub[y_sub > 0]
        
        bins = []
        bins.append(y0)
        
        if y_pos.size > 0:
            q1 = qs_pos[0]
            bins.append(y_pos[y_pos <= q1])
            for i in range(1, len(qs_pos)):
                lo = qs_pos[i - 1]
                hi = qs_pos[i]
                bins.append(y_pos[(y_pos > lo) & (y_pos <= hi)])
            q_last = qs_pos[-1]
            bins.append(y_pos[y_pos > q_last])
        else:
            for _ in range(len(qs_pos) + 1):
                bins.append(np.array([], dtype=y_sub.dtype))

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
                
        return bin_means, bool(ok), counts

    def _compute_bins_and_ok_positive_quantiles(
        self,
        y_sub,
        qe_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=1,
        tag=""
    ):
        """Returns (bin_means, ok, qs_pos, used_q, forced_flat, counts)"""
        y_sub_fin = y_sub[np.isfinite(y_sub)]
        if y_sub_fin.size == 0:
            return None, False, None, None, False, None

        y_pos = y_sub_fin[y_sub_fin > 0]
        if y_pos.size == 0:
            return None, False, None, None, False, None

        qs_pos, used_q, forced_flat = self._adaptive_positive_thresholds(y_pos, qe_pos, min_pos=min_pos_per_bin)

        bin_means, ok, counts = self._compute_bins_given_thresholds(
            y_sub_fin, qs_pos, minB, 
            require_positive_bins=require_positive_bins, 
            min_pos_per_bin=min_pos_per_bin
        )

        if bin_means is None:
            return None, False, qs_pos, used_q, forced_flat, None

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
            
        if gains.size > 0: gains[0] *= self.gainsalpha0
        
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
        min_pos_per_bin=None,
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
        if min_pos_per_bin is None:
            min_pos_per_bin = minB

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
            # robust fallback when bins cannot be calculated
            self._log("[GLOBAL] Cannot calculate bins, using fallback scale-based gains")
            yfin = y[np.isfinite(y)]
            scale = float(np.std(yfin)) if yfin.size > 1 else 1.0
            global_g = np.full(self.C - 1, 0.05 * scale, dtype=np.float32)
            global_spread = float(scale)
            self._log("[GLOBAL] fallback scale-based gains:", global_g)
            # Create dummy thresholds for consistency
            qs_g = np.linspace(0.5, 0.95, self.C - 2, dtype=np.float32)
            self.global_scale = torch.tensor(global_spread, dtype=torch.float32, device=self.global_gains.device)
            self.register_buffer('global_thresholds', torch.tensor(qs_g, dtype=torch.float32))
            usedq_g = np.full(self.C - 2, 0.5, dtype=np.float32) # Dummy
            flat_g = True

        self.global_gains = torch.tensor(global_g, dtype=torch.float32, device=self.global_gains.device)
        self._log("[GLOBAL GAINS] tensor:", self.global_gains)
        
        # PER CLUSTER
        self.gain_k = {}
        self.scale_k = {}
        self.thresholds_k = {}
        self._log("==== CLUSTER GAINS ====")
        
        for cl in unique_clusters:
            idx = idx_by_cluster[cl]
            y_cl = y[idx]
            
            self._log(f"\n--- cluster {cl} (n={len(idx)}) ---")
            
            # FORCE GLOBAL DISCRETIZATION: use global thresholds qs_g
            bm, ok, cnt = self._compute_bins_given_thresholds(
                y_cl, qs_g, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin
            )
            qs, usedq, flat = qs_g, usedq_g, flat_g

            if ok:
                g_cl, spread_cl = self._gains_from_bin_means(bm, y_cl, tag="CL_%s" % str(cl))
            else:
                pooled = False
                if cluster_group is not None and idx_by_group is not None:
                    g_id = cluster_group.get(cl, None)
                    if g_id is not None and g_id in idx_by_group:
                        idx_pool = idx_by_group[g_id]
                        y_pool = y[idx_pool]
                        self._log(f"[CL {cl}] pool by similar group {g_id} (n_pool={len(idx_pool)}) (global discr)")

                        bm2, ok2, cnt2 = self._compute_bins_given_thresholds(
                            y_pool, qs_g, minB,
                            require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin
                        )
                        if ok2:
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

                if pooled:
                    self._log(f"[CL {cl}] used pooled gains (global discr)")
                else:
                    self._log(f"[CL {cl}] used global gains (global discr)")
            
            if self.enforcegainmonotone:
                g_cl = np.maximum.accumulate(np.asarray(g_cl, dtype=np.float32)).astype(np.float32)
            
            self.gain_k[cl] = torch.tensor(g_cl, dtype=torch.float32, device=self.global_gains.device)
            self.scale_k[cl] = torch.tensor(spread_cl, dtype=torch.float32, device=self.global_gains.device)
            self.thresholds_k[cl] = torch.tensor(qs, dtype=torch.float32, device=self.global_gains.device)
            
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
        minbinn=1,
        gainsalpha=1.0,
        gainsalpha0=1.0,
        gainsfloorfrac=0.1,
        enforcegainmonotone=False,
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
        cetype='cornfl',
        alpha=0.25,
        gamma=2
    ):
        super().__init__()
        self.lambdadir = float(lambdadir)
        self.diralpha = float(diralpha)
        self.lambdace = float(lambdace)
        self.cetype = str(cetype).lower()

        # CE Loss Setup - choose between CORN focal and CORN cross-entropy
        if self.cetype == 'cornfl':
            self.ce_loss = CORNFocalLoss(num_classes=num_classes, alpha=alpha, gamma=gamma)
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
        self.gainsalpha0 = float(gainsalpha0)
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
            "gainsalpha0": self.gainsalpha0,
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
    def _adaptive_positive_thresholds(self, y_pos, qe_pos, min_pos=3):
        """
        Adapted from QuantileRiskZerosHandle algorithm.
        
        qe_pos: list/tuple of target quantiles (len = C-2) - used to determine n_thresholds.
        
        Algorithm:
        - Start at q_start (first value in qe_pos, or 0.5)
        - Increment by q_step (0.05) until we have C-2 distinct thresholds
        - Cap at q_max = dynamic (based on sample size to ensure min_pos samples in last bin)
        - Ensure each threshold is strictly different from the previous

        Returns:
          thresholds: np.float32 (C-2,)
          used_q: np.float32 (C-2,)
          forced_flat: bool (True if we couldn't find enough distinct thresholds)
        """
        n_thresholds = len(qe_pos)  # Should be C-2
        q_start = float(qe_pos[0]) if qe_pos else 0.5
        
        # Dynamic q_max calculation
        n_pos = len(y_pos)
        if n_pos > 0:
            # We want roughly min_pos samples in the tail (> q_max)
            # q_max should be approx 1 - (min_pos / n_pos)
            # We use a slight margin (min_pos - 0.5)
            q_max_limit = 1.0 - (max(1.0, float(min_pos) - 0.5) / (n_pos + 1e-9))
            q_max = min(0.999, q_max_limit)
            q_max = max(0.9, q_max) # Don't go below 0.9
        else:
            q_max = 0.999
            
        q_step = 0.05

        thresholds = []
        used_q = []
        q = q_start

        # First threshold
        if q < 1.0:
            first_val = float(np.quantile(y_pos, q))
            thresholds.append(first_val)
            used_q.append(q)

        # Subsequent thresholds - increment until we have enough distinct values
        while len(thresholds) < n_thresholds and q < 1.0:
            q = min(q + q_step, q_max)
            val = float(np.quantile(y_pos, q))

            # Only add if strictly different from previous
            if val != thresholds[-1]:
                thresholds.append(val)
                used_q.append(q)

            if q >= q_max:
                break

        # Check if we found enough distinct thresholds
        forced_flat = len(thresholds) < n_thresholds

        # If we couldn't find enough thresholds, pad with the last value
        while len(thresholds) < n_thresholds:
            thresholds.append(thresholds[-1] if thresholds else 0.0)
            used_q.append(q_max)

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
    def _compute_bins_given_thresholds(
        self,
        y_sub,
        qs_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=1
    ):
        """Calculates bin means and counts for a given set of thresholds."""
        y_sub = y_sub[np.isfinite(y_sub)]
        if y_sub.size == 0:
            return None, False, None
            
        y0 = y_sub[y_sub <= 0]
        y_pos = y_sub[y_sub > 0]
        
        bins = []
        bins.append(y0)
        
        if y_pos.size > 0:
            q1 = qs_pos[0]
            bins.append(y_pos[y_pos <= q1])
            for i in range(1, len(qs_pos)):
                lo = qs_pos[i - 1]
                hi = qs_pos[i]
                bins.append(y_pos[(y_pos > lo) & (y_pos <= hi)])
            q_last = qs_pos[-1]
            bins.append(y_pos[y_pos > q_last])
        else:
            for _ in range(len(qs_pos) + 1):
                bins.append(np.array([], dtype=y_sub.dtype))

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
                
        return bin_means, bool(ok), counts

    def _compute_bins_and_ok_positive_quantiles(
        self,
        y_sub,
        qe_pos,
        minB,
        require_positive_bins=True,
        min_pos_per_bin=1,
        tag=""
    ):
        """Returns (bin_means, ok, qs_pos, used_q, forced_flat, counts)"""
        y_sub_fin = y_sub[np.isfinite(y_sub)]
        if y_sub_fin.size == 0:
            return None, False, None, None, False, None

        y_pos = y_sub_fin[y_sub_fin > 0]
        if y_pos.size == 0:
            return None, False, None, None, False, None

        qs_pos, used_q, forced_flat = self._adaptive_positive_thresholds(y_pos, qe_pos, min_pos=min_pos_per_bin)

        bin_means, ok, counts = self._compute_bins_given_thresholds(
            y_sub_fin, qs_pos, minB, 
            require_positive_bins=require_positive_bins, 
            min_pos_per_bin=min_pos_per_bin
        )

        if bin_means is None:
            return None, False, qs_pos, used_q, forced_flat, None

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
            #spread = 1.0 # No normalization
        else:
            q50 = float(np.quantile(y_pos, 0.5)) if y_pos.size > 0 else 0.0
            q95 = float(np.quantile(y_pos, 0.95)) if y_pos.size > 0 else 0.0
            spread = float(np.std(y_pos)) if y_pos.size > 1 else 0.0
            #spread = 1.0 # No normalization

        spread = max(spread, 1e-6)
        floor = max(0.0, self.gainsfloorfrac)
        
        # Normailzation
        base = self.gainsalpha * (diffs / spread)
        
        #base = self.gainsalpha * diffs
        gains = np.maximum(base, floor).astype(np.float32)
        if self.enforcegainmonotone:
            gains = np.maximum.accumulate(gains).astype(np.float32)

        if gains.size > 0: gains[0] *= self.gainsalpha0

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
        min_pos_per_bin=None,
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
        if min_pos_per_bin is None:
            min_pos_per_bin = minB

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
            # robust fallback when bins cannot be calculated
            self._log("[GLOBAL] Cannot calculate bins, using fallback scale-based gains")
            yfin = y[np.isfinite(y)]
            scale = float(np.std(yfin)) if yfin.size > 1 else 1.0
            global_g = np.full(self.C - 1, 0.05 * scale, dtype=np.float32)
            global_spread = float(scale)
            self._log("[GLOBAL] fallback scale-based gains:", global_g)
            # Create dummy thresholds for consistency
            qs_g = np.linspace(0.5, 0.95, self.C - 2, dtype=np.float32)
            self.global_scale = torch.tensor(global_spread, dtype=torch.float32, device=self.global_gains.device)
            self.register_buffer('global_thresholds', torch.tensor(qs_g, dtype=torch.float32))

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
            
            # FORCE GLOBAL DISCRETIZATION: use global thresholds qs_g
            bm, ok, cnt = self._compute_bins_given_thresholds(
                y_cl, qs_g, minB, require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin
            )
            qs, usedq, flat = qs_g, usedq_g, flat_g

            if ok:
                g_cl, spread_cl = self._gains_from_bin_means(bm, y_cl, tag="CL_%s" % str(cl))
            else:
                pooled = False
                if cluster_group is not None and idx_by_group is not None:
                    g_id = cluster_group.get(cl, None)
                    if g_id is not None and g_id in idx_by_group:
                        idx_pool = idx_by_group[g_id]
                        y_pool = y[idx_pool]
                        self._log(f"[CL {cl}] pool by similar group {g_id} (n_pool={len(idx_pool)}) (global discr)")

                        bm2, ok2, cnt2 = self._compute_bins_given_thresholds(
                            y_pool, qs_g, minB,
                            require_positive_bins=require_positive_bins, min_pos_per_bin=min_pos_per_bin
                        )
                        if ok2:
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

                if pooled:
                    self._log(f"[CL {cl}] used pooled gains (global discr)")
                else:
                    self._log(f"[CL {cl}] used global gains (global discr)")

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
        
        if torch.isnan(final_loss):
            print(f">>> [DEBUG CORNWithGains] NaN Detected in final_loss!")
            print(f"    total_loss: {total_loss.item()} | total_w: {total_w.item()} | eps: {self.eps}")
            print(f"    logits shape: {logits.shape}, y_cont shape: {y_cont.shape}")
            print(f"    logits has NaN: {torch.isnan(logits).any().item()} | y_cont has NaN: {torch.isnan(y_cont).any().item()}")
            
            # Print unique predictions just in case logits exploded
            with torch.no_grad():
                preds = torch.sum(probs > 0.5, dim=1)
                unique_preds, counts = torch.unique(preds, return_counts=True)
                print(f"    Unique predictions: {unique_preds.tolist()} (Counts: {counts.tolist()})")
                
            if hasattr(self, 'epoch_stats'):
                for cl_id, s_d in self.epoch_stats.items():
                    if len(s_d.get('loss_total', [])) > 0:
                        last_loss = s_d['loss_total'][-1]
                        print(f"    Cluster {cl_id} last loss_total: {last_loss}")
                        if math.isnan(last_loss) or math.isinf(last_loss):
                            print(f"      -> BREAKDOWN for {cl_id}:")
                            for k in ['ce_loss', 'mu0_term', 'dirichlet_reg', 'entropy_pi']:
                                if len(s_d.get(k, [])) > 0:
                                    print(f"         {k}: {s_d[k][-1]}")
                        
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

class CumulativeLinkLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.C = num_classes
        
        # raw thresholds parameters (C-1)
        # raw thresholds parameters (C-1)
        self.alpha = nn.Parameter(torch.linspace(-1, 1, num_classes - 1))
        self.thresholds = self._compute_thresholds().detach()
        self.id = 0

    def _compute_thresholds(self):
        # enforce monotonicity
        theta = []
        current = self.alpha[0]
        theta.append(current)
        for i in range(1, len(self.alpha)):
            current = current + F.softplus(self.alpha[i])
            theta.append(current)
        return torch.stack(theta)

    def forward(self, scores, y, sample_weight=None):
        """
        scores: (N,)
        y: (N,) in [0..C-1]
        sample_weight: (N,) or None
        """
        scores = scores.view(-1)
        y = y.view(-1).long()
        theta = self._compute_thresholds()  # (C-1,)

        # CDFs: P(Y <= k | s) for k=0..C-2  (shape: N x (C-1))
        Fk = torch.sigmoid(theta[None, :] - scores[:, None])

        # Convert to class probabilities p(Y=k)
        p = scores.new_zeros((scores.size(0), self.C))
        p[:, 0] = Fk[:, 0]
        if self.C > 2:
            p[:, 1:-1] = Fk[:, 1:] - Fk[:, :-1]
        p[:, -1] = 1.0 - Fk[:, -1]

        # pick true-class probs
        idx = torch.arange(scores.size(0), device=scores.device)
        pt = p[idx, y].clamp_min(1e-9)
        
        nll = -torch.log(pt)
        
        if sample_weight is not None:
            sw = sample_weight.view(-1).to(device=scores.device, dtype=scores.dtype).clamp_min(1e-12)
            return (nll * sw).sum() / sw.sum()
        else:
            return nll.mean()

    def get_learnable_parameters(self):
        return {"alpha": self.alpha}
        
    @torch.no_grad()
    def score_to_class(self, scores: torch.Tensor, clusters_ids: torch.Tensor = None, departement_ids: torch.Tensor = None) -> torch.Tensor:
        """
        scores: (N,) float
        returns: (N,) long in [0..C-1] using self.thresholds
        """
        # bucketize returns index in [0..C-1]
        # right=True => thresholds[i-1] < x <= thresholds[i]
        return torch.bucketize(scores, self.thresholds, right=True)

    def get_attribute(self):
        return [("ordinal_params", {"thresholds": self._compute_thresholds().detach().cpu().numpy()})]

    def update_params(self, epoch):
        self.thresholds = self._compute_thresholds().detach()

    def plot_params(self, params_history, log_dir, best_epoch=None):
        import matplotlib.pyplot as plt
        import pathlib
        import numpy as np

        root_dir = pathlib.Path(log_dir) / 'ordinal_params'
        root_dir.mkdir(parents=True, exist_ok=True)
        
        # history is list of dicts: [{'epoch': E, 'ordinal_params': {...}}, ...]
        epochs = []
        thresholds_list = []
        
        # Handle history format
        iterator = []
        if isinstance(params_history, dict):
             iterator = sorted(params_history.items())
             # map to list of (epoch, dict)
        else:
             # list of dicts
             for entry in params_history:
                 if 'epoch' in entry:
                     iterator.append((entry['epoch'], entry))
             iterator.sort(key=lambda x: x[0])
             
        for ep, entry in iterator:
            if 'ordinal_params' in entry:
                p = entry['ordinal_params']
                if 'thresholds' in p:
                    epochs.append(ep)
                    thresholds_list.append(p['thresholds'])
                    
        if not epochs:
            return

        thresholds_arr = np.array(thresholds_list) # (N_epochs, C-1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(thresholds_arr.shape[1]):
            ax.plot(epochs, thresholds_arr[:, i], label=f'theta_{i}')
            
        ax.set_title(f'{self.__class__.__name__} Thresholds Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Threshold Value')
        ax.grid(True, alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
            
        plt.tight_layout()
        plt.savefig(root_dir / 'thresholds_evolution.png')
        plt.close()

class LargeMarginOrdinalLoss(nn.Module):
    def __init__(self, num_classes, margin=1.0):
        super().__init__()
        self.C = num_classes
        self.margin = margin
        
        self.alpha = nn.Parameter(torch.linspace(-1, 1, num_classes - 1))
        self.thresholds = self._compute_thresholds().detach()

    def _compute_thresholds(self):
        theta = []
        current = self.alpha[0]
        theta.append(current)
        for i in range(1, len(self.alpha)):
            current = current + F.softplus(self.alpha[i])
            theta.append(current)
        return torch.stack(theta)

    def forward(self, scores, y, sample_weight):
        theta = self._compute_thresholds()
        loss = 0.0

        for k in range(self.C):
            mask = (y == k)
            if mask.sum() == 0:
                continue

            s = scores[mask]

            if k > 0:
                lower = theta[k-1] + self.margin
                loss += F.relu(lower - s).mean()

            if k < self.C - 1:
                upper = theta[k] - self.margin
                loss += F.relu(s - upper).mean()

                loss += F.relu(s - upper).mean()

        return loss

    def get_learnable_parameters(self):
        return {"alpha": self.alpha}

    @torch.no_grad()
    def score_to_class(self, scores: torch.Tensor, clusters_ids: torch.Tensor = None, departement_ids: torch.Tensor = None) -> torch.Tensor:
        """
        scores: (N,) float
        returns: (N,) long in [0..C-1] using self.thresholds
        """
        # bucketize returns index in [0..C-1]
        # right=True => thresholds[i-1] < x <= thresholds[i]
        return torch.bucketize(scores, self.thresholds, right=True)

    def get_attribute(self):
        return [("ordinal_params", {"thresholds": self._compute_thresholds().detach().cpu().numpy()})]

    def update_params(self, epoch):
        self.thresholds = self._compute_thresholds().detach()

    def plot_params(self, params_history, log_dir, best_epoch=None):
        import matplotlib.pyplot as plt
        import pathlib
        import numpy as np

        root_dir = pathlib.Path(log_dir) / 'ordinal_params'
        root_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = []
        thresholds_list = []
        
        iterator = []
        if isinstance(params_history, dict):
             iterator = sorted(params_history.items())
        else:
             for entry in params_history:
                 if 'epoch' in entry:
                     iterator.append((entry['epoch'], entry))
             iterator.sort(key=lambda x: x[0])
             
        for ep, entry in iterator:
            if 'ordinal_params' in entry:
                p = entry['ordinal_params']
                if 'thresholds' in p:
                    epochs.append(ep)
                    thresholds_list.append(p['thresholds'])
                    
        if not epochs:
            return

        thresholds_arr = np.array(thresholds_list)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(thresholds_arr.shape[1]):
            ax.plot(epochs, thresholds_arr[:, i], label=f'theta_{i}')
            
        ax.set_title(f'{self.__class__.__name__} Thresholds Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Threshold Value')
        ax.grid(True, alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
            
        plt.tight_layout()
        plt.savefig(root_dir / 'thresholds_evolution.png')
        plt.close()

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


class PairwiseMarginRankingLoss(nn.Module):
    """
    Pairwise Large-Margin Ranking for ordinal bins.

    - forward() utilise directement y_bin
    - score_to_class() utilise des seuils internes appris
    """

    def __init__(
        self,
        num_classes: int,
        margin: float = 0.0,
        num_pairs: Optional[int] = 8192,
    ):
        super().__init__()

        self.C = int(num_classes)
        self.margin = float(margin)
        self.num_pairs = num_pairs

        # Seuils dans l'espace SCORE (pas y)
        if self.C > 1:
            init = torch.linspace(-1.0, 1.0, self.C - 1)
        else:
            init = torch.tensor([])

        self.register_buffer("thresholds", init)

    # --------------------------------------------------
    # Ranking loss
    # --------------------------------------------------

    def forward(self, scores: torch.Tensor, y_bin: torch.Tensor, sample_weight):

        if scores.dim() != 1 or y_bin.dim() != 1:
            raise ValueError("scores and y_bin must be 1D tensors")

        N = scores.numel()
        if N <= 1:
            return scores.new_tensor(0.0)

        # -------- sampled pairs --------
        if self.num_pairs is not None:

            i = torch.randint(0, N, (self.num_pairs,), device=scores.device)
            j = torch.randint(0, N, (self.num_pairs,), device=scores.device)

            mask_neq = i != j
            i, j = i[mask_neq], j[mask_neq]

            yi, yj = y_bin[i], y_bin[j]
            si, sj = scores[i], scores[j]
            
            mask = yi > yj
            if not mask.any():
                return scores.new_tensor(0.0)

            diff = si[mask] - sj[mask]
            return F.relu(self.margin - diff).mean()

        # -------- exact all pairs --------
        ds = scores[:, None] - scores[None, :]
        mask = y_bin[:, None] > y_bin[None, :]
        if not mask.any():
            return scores.new_tensor(0.0)

        return F.relu(self.margin - ds[mask]).mean()

    @torch.no_grad()
    def update_after_batch(self, scores_val: torch.Tensor, y_bin_val: torch.Tensor):
        """
        Calibre self.thresholds (dans l'espace score) pour convertir score -> classe
        en matchant la distribution de y_bin_val.

        scores_val: (N,) float
        y_bin_val: (N,) long in [0..C-1]
        """
        if scores_val.dim() != 1 or y_bin_val.dim() != 1:
            raise ValueError("scores_val et y_bin_val doivent être 1D")
        if scores_val.numel() != y_bin_val.numel():
            raise ValueError("scores_val et y_bin_val doivent avoir la même taille")

        C = self.C
        if C <= 1:
            return

        # proportions cumulées P(y <= k) pour k=0..C-2
        qs = []
        for k in range(C - 1):
            qk = (y_bin_val <= k).float().mean().clamp(1e-4, 1 - 1e-4)
            qs.append(qk)
        qs = torch.stack(qs).to(device=scores_val.device, dtype=scores_val.dtype)

        # thresholds_score[k] = quantile(scores, qs[k])
        t = torch.quantile(scores_val, qs)

        # met à jour self.thresholds (si Parameter ou buffer)
        if isinstance(self.thresholds, torch.nn.Parameter):
            self.thresholds.data.copy_(t)
        else: 
            self.thresholds.copy_(t)

    # --------------------------------------------------
    # Score -> Class
    # --------------------------------------------------

    @torch.no_grad()
    def score_to_class(self, scores: torch.Tensor, clusters_ids: torch.Tensor = None, departement_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Convertit score latent en classe ordinal.

        Utilise self.thresholds (dans l'espace score).
        """
        if scores.dim() != 1:
            raise ValueError("scores must be 1D")

        if self.thresholds.numel() == 0:
            return torch.zeros_like(scores, dtype=torch.long)

        # Garantit monotonie
        thresholds_sorted = torch.sort(self.thresholds)[0]

        return torch.bucketize(scores, thresholds_sorted, right=True)
        
class CLMBinnedTransitionLoss(nn.Module):
    def __init__(self, num_classes: int, beta=3.6167258754794345, t=0.38808363994393985, eps=1e-8,
                 wmed=2.1181338709061728, wmin=0.3090944162785197, wneg=0.010696671944419034, wk=None,
                 learngains=False, gainsfloor=0.5, wkdecay="exp", wkpower=1.0, wklambda=0.3495008616795649,
                 wkmin=0.0026366397418353914, gamma=3.858781740471833, taugate=0.10621789513165877, gatetemp=0.07804890070629288,
                 wfocal=1.5494283305697185, wmu0=0.4200314727662068,
                 fgamma=3.7215798979568424, falpha=0.8563103452989596,
                 mumomentum=0.99, mulambda=1.0):
                 
        super().__init__()
        self.C = int(num_classes)
        self.beta = float(beta)
        self.t = float(t)
        self.eps = float(eps)
        self.wmed = float(wmed)
        self.wmin = float(wmin)
        self.wneg = float(wneg)
        self.wfocal      = float(wfocal)        # weight on focal loss term
        self.wmu0        = float(wmu0)
        self.fgamma = float(fgamma)    # focusing exponent γ (0 = BCE)
        self.falpha = float(falpha)    # class-balance weight α ∈ (0,1)
        self.gamma = float(gamma)
        self.taugate = float(taugate)   # seuil du soft gate sur les probs
        self.gatetemp = float(gatetemp) # température (largeur) du soft gate
        self.mu_momentum = float(mumomentum)
        self.mu_lambda = float(mulambda)
        
        self.register_buffer("mu_prior", torch.zeros(self.C))

        # --- wk schedule (k bigger => smaller weight) ---
        self.wkdecay = wkdecay   # "power" or "exp"
        self.wkpower = wkpower       # p in 1/k^p
        self.wklambda = wklambda     # lambda for exp
        self.wkmin = wkmin        # floor to avoid ~0 weights

        # P_k: transitions (a -> a+k) comme chez toi
        self.P = {k: [(a, a+k) for a in range(0, self.C-k)] for k in range(1, self.C)}
        if wk is None:
            self.wk = self._build_wk_monotone()
        else:
            # If user supplies wk, keep it, but you can optionally enforce monotone decay
            self.wk = wk

        # --- Cutpoints learnables (C-1 seuils) ---
        self.alpha = nn.Parameter(torch.zeros(self.C - 1))
        # --- Optionnel: gains/marges learnables (C-1), positifs ---
        self.learn_gains = bool(learngains)
        self.gains_floor = float(gainsfloor)
        if self.learn_gains:
            self.g_raw = nn.Parameter(torch.zeros(self.C - 1))  # -> softplus pour positiver
            
        self.register_buffer("delta_scale_ema", torch.ones(self.C))
        self.scale_momentum = 0.99   # à tuner
        self.scale_min = 1e-3
        self.scale_max = 1e3
    
    def _build_wk_monotone(self):
        """
        Build wk dict so that wk[k] decreases when k increases.
        """
        wk = {}
        decay = getattr(self, "wkdecay", "power")
        for k in range(1, self.C):
            if decay == "exp":
                w = float(torch.exp(torch.tensor(-self.wklambda * (k - 1))).item())
            elif decay == "None" or decay is None:
                w = 1.0
            else:
                # power decay by default
                w = 1.0 / (float(k) ** float(self.wkpower))
            wk[k] = max(w, float(self.wkmin))
        return wk

    def _compute_thresholds(self):
        # theta monotone via softplus (positif)  :contentReference[oaicite:8]{index=8}
        theta = []
        cur = self.alpha[0]
        theta.append(cur)
        for i in range(1, len(self.alpha)):
            cur = cur + F.softplus(self.alpha[i])
            theta.append(cur)
        return torch.stack(theta)  # (C-1,)

    def _compute_gains(self):
        if hasattr(self, "g_raw"):
            gains = []
            floor = float(getattr(self, "gainsfloor", 0.0))
            cur = F.softplus(self.g_raw[0]) + floor
            gains.append(cur)
            for i in range(1, len(self.g_raw)):
                cur = cur + F.softplus(self.g_raw[i])
                gains.append(cur)
            return torch.stack(gains)  # (C-1,)
        return None

    def _class_probs_from_score(self, s):
        # s: (N,)
        theta = self._compute_thresholds()  # (C-1,)
        # F_k(s)=sigmoid(theta_k - s)  :contentReference[oaicite:9]{index=9}
        Fk = torch.sigmoid(theta[None, :] - s[:, None])  # (N, C-1)

        p = s.new_zeros((s.size(0), self.C))
        p[:, 0] = Fk[:, 0]
        if self.C > 2:
            p[:, 1:-1] = Fk[:, 1:] - Fk[:, :-1]
        p[:, -1] = 1.0 - Fk[:, -1]
        return p  # (N, C)
        
    def _softmin(self, x):
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=0)

    def _soft_median(self, deltas):
        alpha = 20.0
        c = deltas.mean()
        w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
        return (w * deltas).sum()

    def _mu_soft(self, p, y_cont, sw=None):
        """
        p: (N, C)
        y_cont: (N,)
        """
        y = y_cont.to(dtype=p.dtype)

        # --- soft gating by threshold ---
        gate = torch.sigmoid((p - self.taugate) / max(self.gatetemp, 1e-6))
        p = p * gate

        gamma = getattr(self, "gamma", 1.0)
        if gamma != 1.0:
            p = p.clamp_min(self.eps).pow(gamma)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)

        mus = []

        for k in range(p.size(1)):

            pk = p[:, k]

            if sw is not None:
                swk = sw.to(device=p.device, dtype=p.dtype).clamp_min(self.eps)
                weights = pk * swk
            else:
                weights = pk

            # masse effective du bin
            m_k = weights.sum()

            # normalisation des poids
            weights_norm = weights / m_k.clamp_min(self.eps)

            # estimation robuste pondérée (mu_hat_k)
            deltas = y  # vecteur 1D
            alpha = 20.0
            c = (weights_norm * deltas).sum()
            w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
            mu_hat_k = (w * deltas).sum()

            # Mise à jour du prior (EMA) sous no_grad
            with torch.no_grad():
                if m_k > 0.1:  # seuil minimal pour ne pas apprendre sur du bruit pur
                    if self.mu_prior[k] == 0.0:  # initialisation si vide
                        self.mu_prior[k] = mu_hat_k.detach()
                    else:
                        self.mu_prior[k] = self.mu_momentum * self.mu_prior[k] + (1 - self.mu_momentum) * mu_hat_k.detach()

            # Interpolation continue (shrinkage)
            # mu_k ≈ mu_hat_k si mk est grand
            # mu_k ≈ mu_prior si mk est proche de 0
            prior_k = self.mu_prior[k] if self.mu_prior[k] != 0.0 else y.mean()
            mu_k = (m_k * mu_hat_k + self.mu_lambda * prior_k) / (m_k + self.mu_lambda)

            mus.append(mu_k)

        return torch.stack(mus)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.epoch_stats = {}

    def forward(self, score, y_cont, sample_weight=None):
        """
        :contentReference[oaicite:11]{index=11}ore s du modèle
        y_cont: (N,) cible continue (y*)
        """
        s = score.view(-1)
        y = y_cont.view(-1).to(device=s.device)

        sw = sample_weight.view(-1).to(device=s.device) if sample_weight is not None else None

        # 1) bins optimisés => probs ordinales p_k(s)
        probs = self._class_probs_from_score(s)

        # 2) mu par bin, calculé avec y_cont
        mu = self._mu_soft(probs, y, sw=sw)

        # 3) gains (marges) : fixes ou learnables
        if self.learn_gains:
            gains = F.softplus(self.g_raw) + self.gains_floor  # positifs :contentReference[oaicite:12]{index=12}
        else:
            # marge simple: constante (à régler)
            gains = s.new_full((self.C - 1,), 1.0)

        # 4) transition loss = ta logique (deltas + softmin/median + softplus)
        loss = s.new_tensor(0.0)
        wsum = s.new_tensor(0.0)

        for k, pairs in self.P.items():
            raw = torch.stack([mu[b] - mu[a] for (a, b) in pairs], dim=0)

            # marge cumulée entre a..b-1 (comme chez toi) :contentReference[oaicite:13]{index=13}
            margins = torch.stack([gains[a:b].sum() for (a, b) in pairs], dim=0)

            deltas = raw - margins

            batch_scale = deltas.detach().abs().mean().clamp_min(self.eps)

            with torch.no_grad():
                self.delta_scale_ema[k].mul_(self.scale_momentum).add_(
                    batch_scale * (1.0 - self.scale_momentum)
                )

            scale = self.delta_scale_ema[k].clamp(self.scale_min, self.scale_max)
                
            deltas = deltas / scale

            MEDk = self._soft_median(deltas)
            MINk = self._softmin(deltas)

            loss_med = F.softplus(-MEDk)
            loss_min = F.softplus(-MINk)
            loss_neg = F.relu(-deltas).mean()

            Lk = self.wmed * loss_med + self.wmin * loss_min + self.wneg * loss_neg

            if not hasattr(self, 'epoch_stats'):
                self.epoch_stats = {}
            if 'deltas' not in self.epoch_stats:
                self.epoch_stats['deltas'] = {}
            if k not in self.epoch_stats['deltas']:
                self.epoch_stats['deltas'][k] = {'median': [], 'min': [], 'viol': [], 'neg': []}
                
            self.epoch_stats['deltas'][k]['median'].append(MEDk.item())
            self.epoch_stats['deltas'][k]['min'].append(MINk.item())
            self.epoch_stats['deltas'][k]['viol'].append((deltas < 0).float().mean().item())
            self.epoch_stats['deltas'][k]['neg'].append(loss_neg.item())

            w = float(self.wk.get(k, 1.0))
            loss = loss + w * Lk
            wsum = wsum + w

        # Track mu after the loop (mu is global C vector)
        if not hasattr(self, 'epoch_stats'):
            self.epoch_stats = {}
        if 'mu' not in self.epoch_stats:
            self.epoch_stats['mu'] = []
        self.epoch_stats['mu'].append(mu.detach().cpu().numpy())

        # 5) Focal Loss: is-fire term (replaces BCE)
        #    target : (y_cont > 0)  → 1 if any fire
        #    pred   : 1 - probs[:,0] → P(not class-0) = P(fire)
        transition_loss = loss / wsum.clamp_min(self.eps)

        target_bin = (y > 0).to(dtype=s.dtype)           # (N,) ∈ {0,1}
        prob_fire  = (1.0 - probs[:, 0]).clamp(self.eps, 1.0 - self.eps)  # (N,)

        # p_t and alpha_t per sample
        p_t     = torch.where(target_bin > 0.5, prob_fire, 1.0 - prob_fire)  # (N,)
        alpha_t = torch.where(target_bin > 0.5,
                              torch.full_like(p_t, self.falpha),
                              torch.full_like(p_t, 1.0 - self.falpha))  # (N,)
        focal_weight = alpha_t * (1.0 - p_t).pow(self.fgamma)           # (N,)
        focal_ce     = -torch.log(p_t)                                        # (N,)

        if sw is not None:
            sw_norm     = sw / sw.sum().clamp_min(self.eps) * sw.numel()
            focal_loss  = (focal_weight * focal_ce * sw_norm).mean()
        else:
            focal_loss  = (focal_weight * focal_ce).mean()

        if 'focal' not in self.epoch_stats:
            self.epoch_stats['focal'] = []
        self.epoch_stats['focal'].append(focal_loss.item())

        if 'transition' not in self.epoch_stats:
            self.epoch_stats['transition'] = []
        self.epoch_stats['transition'].append(transition_loss.item())

        mu0_val = mu[0]
        mu0_term = F.softplus(mu0_val)

        if 'mu0_term' not in self.epoch_stats:
            self.epoch_stats['mu0_term'] = []
        self.epoch_stats['mu0_term'].append(mu0_term.item())

        return transition_loss + self.wfocal * focal_loss + self.wmu0 * mu0_term
        
    def get_learnable_parameters(self):
        # Always expose alpha (cutpoints params)
        params = {"alpha": self.alpha}

        # If gains are learnable, expose their raw parameters too
        # (Assumes you store raw gains in self.g_raw, then gains = softplus(g_raw) (+ floor))
        if getattr(self, "learn_gains", False):
            if hasattr(self, "g_raw"):
                params["g_raw"] = self.g_raw
            elif hasattr(self, "gain_raw"):
                # fallback naming if you used a different attribute name
                params["gain_raw"] = self.gain_raw
        return params

    @torch.no_grad()
    def score_to_class(self, scores: torch.Tensor, clusters_ids: torch.Tensor = None, departement_ids: torch.Tensor = None) -> torch.Tensor:
        """
        scores: (N,) float
        returns: (N,) long in [0..C-1] using self.thresholds
        torch.bucketize(..., right=True) => thresholds[i-1] < x <= thresholds[i]. :contentReference[oaicite:1]{index=1}
        """
        if not hasattr(self, "thresholds") or self.thresholds is None:
            self.thresholds = self._compute_thresholds().detach()

        thr = self.thresholds.to(device=scores.device)
        return torch.bucketize(scores, thr, right=True)

    def get_attribute(self):
        import numpy as np
        # Always log thresholds; if learn_gains, log gains too.
        payload = {
            "thresholds": self._compute_thresholds().detach().cpu().numpy(),
            "mu_prior": self.mu_prior.detach().cpu().numpy()
        }

        if getattr(self, "learn_gains", False):
            # Compute "current" positive gains for logging
            g = self._compute_gains().detach().cpu().numpy() if hasattr(self, "_compute_gains") else None
            if g is None:
                # fallback: if you keep a tensor self.gains already
                g = self.gains.detach().cpu().numpy() if hasattr(self, "gains") and self.gains is not None else None
            
            if g is None and hasattr(self, "g_raw"):
                floor = float(getattr(self, "gains_floor", 0.0))
                g = (F.softplus(self.g_raw) + floor).detach().cpu().numpy()

            if g is not None:
                payload["gains"] = g

        if hasattr(self, 'epoch_stats') and 'deltas' in self.epoch_stats:
            agg_deltas = {}
            for k, dstats in self.epoch_stats['deltas'].items():
                agg_deltas[k] = {
                    'median': np.mean(dstats['median']) if dstats['median'] else 0.0,
                    'min': np.mean(dstats['min']) if dstats['min'] else 0.0,
                    'viol': np.mean(dstats['viol']) if dstats['viol'] else 0.0,
                    'neg': np.mean(dstats['neg']) if dstats['neg'] else 0.0
                }
            payload['deltas'] = agg_deltas

        if hasattr(self, 'epoch_stats') and 'mu' in self.epoch_stats and len(self.epoch_stats['mu']) > 0:
            mu_stack = np.stack(self.epoch_stats['mu'])  # (N_batches, C)
            payload['mu'] = np.mean(mu_stack, axis=0)  # (C,)

        # Loss components: transition and bce (mean over batches for this epoch)
        if hasattr(self, 'epoch_stats'):
            for _lkey in ('transition', 'focal'):
                vals = self.epoch_stats.get(_lkey, [])
                if vals:
                    payload[_lkey] = [float(np.mean(vals))]  # list so plot_params can use np.mean(vals)
                    
        payload['delta_scale_ema'] = self.delta_scale_ema.detach().cpu().numpy()

        return [("ordinal_params", DictWrapper(payload))]

    def update_params(self, epoch):
        # Cache thresholds for stable usage during an epoch
        self.thresholds = self._compute_thresholds().detach()

        # Cache gains too if learnable
        if getattr(self, "learn_gains", False):
            if hasattr(self, "_compute_gains"):
                self.gains = self._compute_gains().detach()
            else:
                if hasattr(self, "g_raw"):
                    floor = float(getattr(self, "gains_floor", 0.0))
                    self.gains = (F.softplus(self.g_raw) + floor).detach()

        # Reset per-epoch accumulators so get_attribute() exports only the current epoch
        if not hasattr(self, 'epoch_stats'):
            self.epoch_stats = {}
        self.epoch_stats['transition'] = []
        self.epoch_stats['focal']      = []

    def plot_params(self, params_history, log_dir, best_epoch=None):
        import matplotlib.pyplot as plt
        import pathlib
        import numpy as np

        root_dir = pathlib.Path(log_dir) / 'ordinal_params'
        root_dir.mkdir(parents=True, exist_ok=True)

        epochs = []
        thresholds_list = []
        gains_list = []
        deltas_list = []
        mu_list = []
        delta_scale_ema_list = []
        mu_prior_list = []

        iterator = []
        if isinstance(params_history, dict):
            iterator = sorted(params_history.items())
        else:
            for entry in params_history:
                if isinstance(entry, dict) and ('epoch' in entry):
                    iterator.append((entry['epoch'], entry))
            iterator.sort(key=lambda x: x[0])

        for ep, entry in iterator:
            if 'ordinal_params' not in entry:
                continue
                
            stats_container = entry['ordinal_params']
            if hasattr(stats_container, 'd'):
                p = stats_container.d
            else:
                p = stats_container
                
            if not isinstance(p, dict):
                continue

            if 'thresholds' not in p:
                continue
                
            epochs.append(ep)
            thresholds_list.append(p['thresholds'])
            gains_list.append(p.get('gains', None))
            deltas_list.append(p.get('deltas', None))
            mu_list.append(p.get('mu', None))
            delta_scale_ema_list.append(p.get('delta_scale_ema', None))
            mu_prior_list.append(p.get('mu_prior', None))

        if not epochs:
            print(f'Error with epochs {epochs}')
            return

        thresholds_arr = np.array(thresholds_list)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(thresholds_arr.shape[1]):
            ax.plot(epochs, thresholds_arr[:, i], label=f'theta_{i}')
        ax.set_title(f'{self.__class__.__name__} Thresholds Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Threshold Value')
        ax.grid(True, alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
        ax.legend()
        plt.tight_layout()
        plt.savefig(root_dir / 'thresholds_evolution.png')
        plt.close()

        try:
            valid_gains = [(ep, g) for ep, g in zip(epochs, gains_list) if g is not None]
            if valid_gains:
                g_epochs, g_vals = zip(*valid_gains)
                gains_arr = np.array(g_vals)
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(gains_arr.shape[1]):
                    ax.plot(list(g_epochs), gains_arr[:, i], label=f'gain_{i}')
                ax.set_title(f'{self.__class__.__name__} Gains Evolution')
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
                ax.legend()
                plt.tight_layout()
                plt.savefig(root_dir / 'gains_evolution.png')
                plt.close()
        except: plt.close('all')

        try:
            valid_deltas = [(ep, d) for ep, d in zip(epochs, deltas_list) if d is not None]
            if valid_deltas:
                d_epochs, d_vals = zip(*valid_deltas)
                d_epochs = list(d_epochs)
                ks = sorted(d_vals[0].keys())
                if ks:
                    fig, axes = plt.subplots(len(ks), 4, figsize=(20, 3*len(ks)), sharex=True)
                    if len(ks) == 1: axes = axes[None, :] 
                    for i, k in enumerate(ks):
                        axes[i, 0].plot(d_epochs, [d[k]['median'] for d in d_vals if k in d], color='blue')
                        axes[i, 0].set_title(f'k={k} Median Delta')
                        axes[i, 1].plot(d_epochs, [d[k]['min'] for d in d_vals if k in d], color='red')
                        axes[i, 1].set_title(f'k={k} Min Delta')
                        axes[i, 2].plot(d_epochs, [d[k]['viol'] for d in d_vals if k in d], color='orange')
                        axes[i, 2].set_title(f'k={k} Viol Rate')
                        axes[i, 2].set_ylim(-0.1, 1.1)
                        axes[i, 3].plot(d_epochs, [d[k]['neg'] for d in d_vals if k in d], color='purple')
                        axes[i, 3].set_title(f'k={k} Mean NEG')
                    if best_epoch is not None:
                        for ax_row in axes:
                            for ax_cell in ax_row:
                                ax_cell.axvline(best_epoch, color='r', linestyle='--', linewidth=0.8)
                    plt.tight_layout()
                    plt.savefig(root_dir / 'deltas_stats.png')
                    plt.close()
        except: plt.close('all')

        try:
            valid_scales = [(ep, s) for ep, s in zip(epochs, delta_scale_ema_list) if s is not None]
            if valid_scales:
                s_epochs, s_vals = zip(*valid_scales)
                scales_arr = np.array(s_vals)  # (epochs, num_pairs) or (epochs, nclusters, num_pairs)
                pairs = getattr(self, 'all_pairs', None)

                # ── overview : mean/flatten ──────────────────────────────────────
                s_flat = scales_arr.mean(axis=1) if scales_arr.ndim == 3 else scales_arr
                fig, ax = plt.subplots(figsize=(10, 5))
                for i in range(s_flat.shape[1]):
                    lbl = f'({pairs[i][0]},{pairs[i][1]})' if pairs is not None and i < len(pairs) else f'pair_{i}'
                    ax.plot(list(s_epochs), s_flat[:, i], label=lbl)
                ax.set_title(f'{self.__class__.__name__} Delta Scale EMA (mean clusters)')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
                ax.legend(fontsize=6, ncol=4)
                plt.tight_layout()
                plt.savefig(root_dir / 'delta_scale_ema_evolution.png')
                plt.close()

                # ── per-cluster subplots (si 3-D) ────────────────────────────────
                if scales_arr.ndim == 3:
                    n_clusters = scales_arr.shape[1]
                    num_pairs  = scales_arr.shape[2]
                    cols = min(4, n_clusters)
                    rows = (n_clusters + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols,
                                             figsize=(5*cols, 3.5*rows),
                                             sharex=True)
                    axes_flat = np.array(axes).flatten()
                    cmap = plt.cm.tab20
                    for cl in range(n_clusters):
                        ax = axes_flat[cl]
                        for i in range(num_pairs):
                            lbl = f'({pairs[i][0]},{pairs[i][1]})' if pairs is not None and i < len(pairs) else f'p{i}'
                            ax.plot(list(s_epochs), scales_arr[:, cl, i],
                                    color=cmap(i / max(num_pairs-1,1)), label=lbl, alpha=0.8)
                        if best_epoch is not None:
                            ax.axvline(best_epoch, color='r', linestyle='--', linewidth=0.8)
                        ax.set_yscale('log')
                        ax.set_title(f'Cluster {cl}')
                        ax.set_xlabel('Epoch')
                        ax.grid(True, alpha=0.3)
                    for j in range(n_clusters, len(axes_flat)):
                        axes_flat[j].set_visible(False)
                    axes_flat[n_clusters-1].legend(fontsize=5, loc='best', ncol=2)
                    fig.suptitle(f'{self.__class__.__name__} — Delta Scale EMA per cluster', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(root_dir / 'delta_scale_per_cluster.png')
                    plt.close()
        except Exception as _e:
            plt.close('all')
            print(f'[plot_params] scale error: {_e}')

        try:
            valid_mu_priors = [(ep, m) for ep, m in zip(epochs, mu_prior_list) if m is not None]
            if valid_mu_priors:
                m_epochs, m_vals = zip(*valid_mu_priors)
                mu_arr = np.array(m_vals)  # (epochs, classes) or (epochs, nclusters, classes)

                # ── overview : mean over clusters ────────────────────────────────
                mu_avg = mu_arr.mean(axis=1) if mu_arr.ndim == 3 else mu_arr
                fig, ax = plt.subplots(figsize=(10, 5))
                for c in range(mu_avg.shape[1]):
                    ax.plot(list(m_epochs), mu_avg[:, c], label=f'class {c}')
                ax.set_title(f'{self.__class__.__name__} Mu Prior (mean clusters)')
                ax.set_xlabel('Epoch')
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
                ax.legend()
                plt.tight_layout()
                plt.savefig(root_dir / 'mu_prior_evolution.png')
                plt.close()

                # ── per-cluster subplots (si 3-D) ────────────────────────────────
                if mu_arr.ndim == 3:
                    n_clusters = mu_arr.shape[1]
                    n_classes  = mu_arr.shape[2]
                    cols = min(4, n_clusters)
                    rows = (n_clusters + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols,
                                             figsize=(5*cols, 3.5*rows),
                                             sharey=True, sharex=True)
                    axes_flat = np.array(axes).flatten()
                    cmap = plt.cm.viridis
                    for cl in range(n_clusters):
                        ax = axes_flat[cl]
                        for c in range(n_classes):
                            ax.plot(list(m_epochs), mu_arr[:, cl, c],
                                    color=cmap(c / max(n_classes-1,1)), label=f'class {c}')
                        if best_epoch is not None:
                            ax.axvline(best_epoch, color='r', linestyle='--', linewidth=0.8)
                        ax.set_title(f'Cluster {cl}')
                        ax.set_xlabel('Epoch')
                        ax.grid(True, alpha=0.3)
                    for j in range(n_clusters, len(axes_flat)):
                        axes_flat[j].set_visible(False)
                    axes_flat[n_clusters-1].legend(fontsize=7, loc='best')
                    fig.suptitle(f'{self.__class__.__name__} — Mu Prior per cluster', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(root_dir / 'mu_prior_per_cluster.png')
                    plt.close()
        except Exception as _e:
            plt.close('all')
            print(f'[plot_params] mu_prior error: {_e}')

        try:
            valid_mu = [(ep, m) for ep, m in zip(epochs, mu_list) if m is not None]
            if valid_mu:
                m_epochs, m_vals = zip(*valid_mu)
                mu_arr = np.array(m_vals)  # shape: (epochs, classes) or (epochs, nclusters, classes)

                # ── Plot 1: mu evolution per class (averaged over clusters if 3-D) ────
                mu_plot = mu_arr.mean(axis=1) if mu_arr.ndim == 3 else mu_arr
                fig, ax = plt.subplots(figsize=(10, 6))
                for c in range(mu_plot.shape[1]):
                    ax.plot(list(m_epochs), mu_plot[:, c], label=f'mu(class {c})')
                ax.set_title(f'{self.__class__.__name__} - Mu evolution per class (mean over clusters)')
                ax.set_xlabel('Epoch')
                ax.legend()
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
                plt.tight_layout()
                plt.savefig(root_dir / 'mu_s.png')
                plt.close()

                # ── Plot 2: mu par cluster (si 3-D) ─────────────────────────────────
                if mu_arr.ndim == 3:
                    n_clusters = mu_arr.shape[1]
                    n_classes  = mu_arr.shape[2]
                    cols = min(4, n_clusters)
                    rows = (n_clusters + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols,
                                             figsize=(5 * cols, 3.5 * rows),
                                             sharey=True, sharex=True)
                    axes_flat = np.array(axes).flatten()
                    cmap = plt.cm.plasma
                    for cl in range(n_clusters):
                        ax = axes_flat[cl]
                        # mu values over epochs for this cluster
                        cl_mu = mu_arr[:, cl, :]   # (epochs, classes)
                        for c in range(n_classes):
                            color = cmap(c / max(n_classes - 1, 1))
                            ax.plot(list(m_epochs), cl_mu[:, c], color=color, label=f'class {c}')
                        if best_epoch is not None:
                            ax.axvline(best_epoch, color='r', linestyle='--', linewidth=0.8)
                        ax.set_title(f'Cluster {cl}')
                        ax.set_xlabel('Epoch')
                        ax.grid(True, alpha=0.3)
                    for j in range(n_clusters, len(axes_flat)):
                        axes_flat[j].set_visible(False)
                    # legend sur le dernier plot visible
                    axes_flat[n_clusters - 1].legend(fontsize=7, loc='best')
                    fig.suptitle(f'{self.__class__.__name__} — Mu per cluster', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(root_dir / 'mu_per_cluster.png')
                    plt.close()
        except Exception as _e:
            plt.close('all')
            print(f'[plot_params] mu plot error: {_e}')

        try:
            tr_list, bce_list, ep_list = [], [], []
            for ep, entry in iterator:
                p = entry['ordinal_params'].d if hasattr(entry['ordinal_params'], 'd') else entry['ordinal_params']
                if 'transition' in p and 'focal' in p:
                    ep_list.append(ep)
                    tr_list.append(float(np.mean(p['transition'])))
                    bce_list.append(float(np.mean(p['focal'])))
            if ep_list:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ep_list, tr_list,  label='transition loss')
                ax.plot(ep_list, bce_list, label='Focal loss', linestyle='--')
                ax.set_title(f'{self.__class__.__name__} – Loss components')
                ax.legend()
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
                plt.tight_layout()
                plt.savefig(root_dir / 'loss_components.png')
                plt.close()
        except: plt.close('all')

def check_finite(name, x):
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise RuntimeError(f"[NaN ERROR] {name} contains NaN or Inf")
    else:
        import numpy as np
        if not np.isfinite(x):
            raise RuntimeError(f"[NaN ERROR] {name} contains NaN or Inf")

#
#    occurence_01_06_25_summer nbsinister
#

    # "beta": 5.64,
    # "t": 0.14,
    # "wneg": 0.15,
    # "gamma": 4.58,
    # "taugate": 0.09,
    # "gatetemp": 0.05,
    # "wkdecay": "exp",
    # "wklambda": 0.15,
    # "wkmin": 0.0,
    # "wfocal": 0.42,
    # "wmu0": 0.04,
    # "fgamma": 4.02,
    # "falpha": 0.6,
    # "mumomentum": 0.87,
    # "mulambdag": 0.8,
    # "mulambdac": 0.74,
    # "ndepartements": 3,
    # "num_classes": 5
    
#
# occurence_01_06_25 nbsinister
#

   #"beta": 13.34,
   # "t": 0.11,
   # "wneg": 0.89,
   # "gamma": 3.31,
   # "taugate": 0.02,
   # "gatetemp": 0.01,
   # "wkdecay": "None",
   # "wkmin": 0.0,
   # "wfocal": 1.46,
   # "wmu0": 0.22,
   # "fgamma": 1.11,
   # "falpha": 0.8,
   # "mumomentum": 0.89,
   # "mulambdag": 0.3,
   # "mulambdac": 1.81,
   # "ndepartements": 3,
   # "num_classes": 5
   
#
#
#

#"gainsfloor": 1.0,
#"wkdecay": "exp",
#"wklambda": 0.3,
#"gamma": 1.83,
#"taugate": 0.08,
#"gatetemp": 0.98,
#"wfocal": 3.67,
#"wmu0": 2.82,
#"fgamma": 0.6,
#"falpha": 0.79,
#"massupdate": 0.45,
#"mumomentum": 1.0,
#"mulambdag": 4.22,
#"mulambdac": 4.93,
   
#
# occurence 01_06_25 ressource
#

    #"beta": 12.42,
    #"t": 0.0,
    #"wneg": 1.49,
    #"gamma": 2.07,
    #"taugate": 0.36,
    #"gatetemp": 0.02,
    #"wkdecay": "None",
    #"wkmin": 0.0,
    #"wfocal": 1.23,
    #"wmu0": 1.85,
    #"fgamma": 3.19,
    #"falpha": 0.56,
    #"mumomentum": 0.82,
    #"mulambdag": 1.13,
    #"mulambdac": 0.83,
    #"ndepartements": 3,
    #"num_classes": 5
    
#
# occurence_01_06_25 time
#


    #"beta": 7.29,
    #"t": 0.0,
    #"wneg": 2.54,
    #"gamma": 2.48,
    #"taugate": 0.25,
    #"gatetemp": 0.01,
    #"wkdecay": "None",
    #"wkmin": 0.0,
    #"wfocal": 1.93,
    #"wmu0": 0.12,
    #"fgamma": 1.13,
    #"falpha": 0.26,
    #"mumomentum": 0.85,
    #"mulambdag": 0.52,
    #"mulambdac": 0.65,
    #"ndepartements": 3,
    #"num_classes": 5
    
    
#
# bdiff default burnedareaRoot
#

    #"beta": 3.01,
    #"t": 0.23,
    #"wneg": 1.69,
    #"gamma": 1.76,
    #"taugate": 0.35,
    #"gatetemp": 0.44,
    #"wkdecay": "power",
    #"wkpower": 2.42,
    #"wkmin": 0.01,
    #"wfocal": 0.84,
    #"wmu0": 0.22,
    #"fgamma": 3.69,
    #"falpha": 0.22,
    #"mumomentum": 0.92,
    #"mulambdag": 1.19,
    #"mulambdac": 1.47,
    #"ndepartements": 92,
    #"num_classes": 5
    
def check_finite(name, x):
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError(f"[{name}] contains NaN/Inf")
    return True

class ClusterCLMBinnedTransitionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        eps=1e-4,
        wk=None,
        learngains=False,
        gainsfloor=2.5,
        wkdecay="None",
        wkpower=2.06,
        wklambda=0.3495008616795649,
        gamma=5.0,
        taugate=0.05,
        gatetemp=0.11,
        wfocal=1.76,
        wmu0=1.94,
        wmid=0.2,
        wtrans=1.0,
        fgamma=1.03,
        falpha=0.89,
        massupdate=0.5,
        mumomentum=0.99,
        mulambdag=0.18,
        mulambdac=1.61,
        id=0,
        nclusters=1,
        ndepartements=1,
        scaleagg="department",
        alphatype="department",
        muinit=None,
        scaleinit=None,
    ):
        super().__init__()

        self.C = int(num_classes)
        self.id = int(id)
        self.nclusters = int(nclusters)
        self.ndepartements = int(ndepartements)
        self.scaleagg = scaleagg
        self.alphatype = str(alphatype)

        self.beta = 0.0
        self.t = 0.0
        self.eps = float(eps)

        self.wfocal = float(wfocal)
        self.wmu0 = float(wmu0)
        self.wtrans = float(wtrans)

        self.fgamma = float(fgamma)
        self.falpha = float(falpha)

        self.gamma = float(gamma)
        self.taugate = float(taugate)
        self.gatetemp = float(gatetemp)

        self.mu_momentum = float(mumomentum)
        self.mu_lambda_g = float(mulambdag)
        self.mu_lambda_c = float(mulambdac)
        self.massupdate = float(massupdate)
        
        self.wmid = wmid          # poids de Lmid
        self.mid_detach_mu = True

        _buf_size = self.nclusters
        _dept_buf_size = self.ndepartements

        self.cluster_raw_to_slot = {}
        self.departement_raw_to_slot = {}
        self.cluster_next_free_slot = 0
        self.departement_next_free_slot = 0

        self.register_buffer(
            "cluster_slot_to_raw",
            torch.full((_buf_size,), -1, dtype=torch.long)
        )
        self.register_buffer(
            "departement_slot_to_raw",
            torch.full((_dept_buf_size,), -1, dtype=torch.long)
        )

        if muinit is not None:
            _mu = torch.as_tensor(muinit, dtype=torch.float32)
            if _mu.shape == (self.nclusters, self.C):
                self.register_buffer("mu_prior", _mu.clone())
            else:
                raise ValueError(
                    f"muinit shape {tuple(_mu.shape)} != ({self.nclusters}, {self.C})"
                )
        else:
            self.register_buffer(
                "mu_prior",
                torch.full((_buf_size, self.C), float("nan"), dtype=torch.float32)
            )

        self.register_buffer(
            "mu_prior_global",
            torch.full((self.C,), float("nan"), dtype=torch.float32)
        )

        self.wkdecay = wkdecay
        self.wkpower = wkpower
        self.wklambda = wklambda
        self.wkmin = 0.0

        self.P = {k: [(a, a + k) for a in range(0, self.C - k)] for k in range(1, self.C)}
        if wk is None:
            self.wk = self._build_wk_monotone()
        else:
            self.wk = wk
            
        self.all_pairs = [(a, b) for k in range(1, self.C) for (a, b) in self.P[k]]
        self.pair_to_idx = {(a, b): i for i, (a, b) in enumerate(self.all_pairs)}
        num_pairs = len(self.all_pairs)

        if self.alphatype == "cluster":
            self.alpha = nn.Parameter(torch.zeros(_buf_size, self.C - 1))
        elif self.alphatype == "department":
            self.alpha = nn.Parameter(torch.zeros(_dept_buf_size, self.C - 1))
        else:
            self.alpha = nn.Parameter(torch.zeros(self.C - 1))

        self.learn_gains = bool(learngains)
        self.gains_floor = float(gainsfloor)
        if self.learn_gains:
            self.g_raw = nn.Parameter(torch.zeros(self.C - 1))
        else:
            self.g_raw = torch.zeros(self.C - 1)

        if self.scaleagg == "cluster":
            _default_scale = torch.ones(_buf_size, num_pairs)
            self.register_buffer("delta_scale_ema", _default_scale)
        elif self.scaleagg == "department":
            _default_scale = torch.ones(_dept_buf_size, num_pairs)
            self.register_buffer("delta_scale_ema", _default_scale)
        else:
            _default_scale = torch.ones(num_pairs)
            self.register_buffer("delta_scale_ema", _default_scale)

        if scaleinit is not None:
            _sc = torch.as_tensor(scaleinit, dtype=torch.float32)
            if _sc.shape == self.delta_scale_ema.shape:
                self.delta_scale_ema.copy_(_sc)
            else:
                raise ValueError(
                    f"scaleinit shape {tuple(_sc.shape)} != {tuple(self.delta_scale_ema.shape)}"
                )

        self.scale_momentum = 0.99
        self.scale_min = 1e-3
        self.scale_max = 1e3

        self.tau_loss = 1.5
        self.loss_ema_momentum = 0.99
        self.register_buffer(
            "loss_ema",
            torch.zeros(_buf_size, dtype=torch.float32)
        )

    def _build_wk_monotone(self):
        wk = {}
        decay = getattr(self, "wkdecay", "power")
        raw = {}

        for k in range(1, self.C):
            if decay == "exp":
                w = float(math.exp(-self.wklambda * (k - 1)))
            elif decay == "None" or decay is None:
                w = 1.0
            else:
                w = 1.0 / (float(k) ** float(self.wkpower))
            raw[k] = max(w, float(self.wkmin))

        ks = sorted(raw.keys())
        vals = [raw[k] for k in ks]
        for k, v in zip(ks, reversed(vals)):
            wk[k] = v
        return wk

    def _compute_thresholds(self):
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

    def _compute_gains(self):
        if hasattr(self, "g_raw"):
            gains = []
            floor = float(getattr(self, "gains_floor", 0.0))
            cur = F.softplus(self.g_raw[0]) + floor
            gains.append(cur)
            for i in range(1, len(self.g_raw)):
                cur = cur + F.softplus(self.g_raw[i])
                gains.append(cur)
            return torch.stack(gains)
        return None

    def _class_probs_from_score(self, s, clusters_ids=None, departement_ids=None):
        theta = self._compute_thresholds().to(s.device)

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

            thr = theta.index_select(0, chosen_ids.to(device=s.device).long())
            Fk = torch.sigmoid(thr - s[:, None])

        p = s.new_zeros((s.size(0), self.C))
        p[:, 0] = Fk[:, 0]
        if self.C > 2:
            p[:, 1:-1] = Fk[:, 1:] - Fk[:, :-1]
        p[:, -1] = 1.0 - Fk[:, -1]
        return p

    def _softmin(self, x):
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=0)

    def _soft_median(self, deltas):
        alpha = 20.0
        if deltas.dim() == 1:
            c = deltas.mean()
            w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
            return (w * deltas).sum()
        else:
            c = deltas.mean(dim=0, keepdim=True)
            w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
            return (w * deltas).sum(dim=0)

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
                        f"No free slot left for kind='{kind}'. Encountered new raw id {rid}, but buf_size={buf_size}."
                    )
                slot = next_free_slot
                raw_to_slot[rid] = slot
                slot_to_raw[slot] = rid
                next_free_slot += 1

            local_ids[i] = slot

        setattr(self, next_free_attr, next_free_slot)

        valid_mask = torch.ones_like(raw_ids, dtype=torch.bool, device=device)
        return slot_to_raw.clone(), local_ids, valid_mask

    def _mu_soft(self, p, y_cont, clusters_ids_local, sw=None, active_cluster_slots=None):
        y = y_cont.to(dtype=p.dtype)
        device = p.device
        
        
        # Gating to remove small contribution

        gate = torch.sigmoid((p - self.taugate) / max(self.gatetemp, 1e-6))
        p = p * gate

        gamma = getattr(self, "gamma", 1.0)
        if gamma != 1.0:
            p = p.clamp_min(self.eps).pow(gamma)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)

        Z = len(active_cluster_slots) if active_cluster_slots is not None else (
            int(clusters_ids_local.max().item()) + 1 if clusters_ids_local.numel() > 0 else 1
        )

        # Define mu and mass attributes
        mu_clusters = torch.zeros(Z, self.C, device=device, dtype=p.dtype)
        mass_clusters = torch.zeros(Z, self.C, device=device, dtype=p.dtype)

        # Get weights for shrinkage
        lambda_g = float(getattr(self, "mu_lambda_g", 1.0))
        lambda_c = float(getattr(self, "mu_lambda_c", 1.0))
        min_mass_update = self.massupdate * self.taugate

        for k in range(self.C):
            pk = p[:, k]
            if sw is not None:
                swk = sw.to(device=device, dtype=p.dtype).clamp_min(self.eps)
                weights = pk * swk
            else:
                weights = pk

            m_k_global = weights.sum()
            if m_k_global > 0:
                # Calculate the mu_hat_global for class k
                weights_norm_global = weights / m_k_global.clamp_min(self.eps)
                alpha = 20.0
                c_global = (weights_norm_global * y).sum()
                w_global = torch.softmax(-alpha * (y - c_global).abs(), dim=0)
                mu_hat_k_global = (w_global * y).sum()
            else:
                mu_hat_k_global = torch.tensor(0.0, device=device, dtype=p.dtype)

            with torch.no_grad():
                if not torch.isfinite(self.mu_prior_global[k]):
                    self.mu_prior_global[k] = mu_hat_k_global.detach()
                elif m_k_global > min_mass_update:
                    # update global with EMA
                    self.mu_prior_global[k] = (
                        self.mu_momentum * self.mu_prior_global[k]
                        + (1.0 - self.mu_momentum) * mu_hat_k_global.detach()
                    )

            check_finite("p_before_gate", p)
            check_finite("gate", gate)
            check_finite("p_after_gate", p)
            check_finite("weights", weights)
            check_finite("m_k_global", m_k_global)
            check_finite("mu_hat_k_global", mu_hat_k_global)
            check_finite("mu_prior_global", self.mu_prior_global[k])

            eps = self.eps
            alpha = 20.0
            
            # Define mu for cluster and class k

            m_k = torch.zeros(Z, device=device, dtype=p.dtype)
            m_k.scatter_add_(0, clusters_ids_local, weights)
            mass_clusters[:, k] = m_k.detach()

            wy = weights * y
            sum_wy = torch.zeros(Z, device=device, dtype=p.dtype)
            sum_wy.scatter_add_(0, clusters_ids_local, wy)
            c_loc = sum_wy / m_k.clamp_min(eps)

            logits = -alpha * (y - c_loc[clusters_ids_local]).abs()

            max_per_cluster = torch.full((Z,), -float("inf"), device=device, dtype=p.dtype)
            max_per_cluster.scatter_reduce_(0, clusters_ids_local, logits, reduce="amax", include_self=True)

            logits_shift = logits - max_per_cluster[clusters_ids_local]
            exp_logits = torch.exp(logits_shift)

            den = torch.zeros(Z, device=device, dtype=p.dtype)
            den.scatter_add_(0, clusters_ids_local, exp_logits)
            w_loc = exp_logits / den[clusters_ids_local].clamp_min(eps)

            mu_hat = torch.zeros(Z, device=device, dtype=p.dtype)
            mu_hat.scatter_add_(0, clusters_ids_local, w_loc * y)

            valid = m_k > min_mass_update

            with torch.no_grad():
                if active_cluster_slots is not None:
                    for li, slot_t in enumerate(active_cluster_slots):
                        slot = int(slot_t.item())
                        old_val = self.mu_prior[slot, k]
                        if not torch.isfinite(old_val):
                            self.mu_prior[slot, k] = mu_hat[li].detach()
                        elif valid[li]: # update mu cluster with EMA
                            self.mu_prior[slot, k] = (
                                self.mu_momentum * old_val
                                + (1.0 - self.mu_momentum) * mu_hat[li].detach()
                            )

            _pg_raw = self.mu_prior_global[k].to(device=device, dtype=p.dtype)
            prior_global_k = _pg_raw if torch.isfinite(_pg_raw) else y.mean()

            if active_cluster_slots is not None:
                prior_cluster_k = self.mu_prior[active_cluster_slots, k].to(device=device, dtype=p.dtype)
                prior_cluster_k = torch.where(
                    torch.isfinite(prior_cluster_k),
                    prior_cluster_k,
                    torch.full_like(prior_cluster_k, prior_global_k)
                )
            else:
                prior_cluster_k = self.mu_prior[:Z, k].to(device=device, dtype=p.dtype)
                prior_cluster_k = torch.where(
                    torch.isfinite(prior_cluster_k),
                    prior_cluster_k,
                    torch.full_like(prior_cluster_k, prior_global_k)
                )

            mu_k = torch.where(
                m_k <= eps,
                prior_cluster_k,
                (m_k * mu_hat + lambda_c * prior_cluster_k + lambda_g * prior_global_k)
                / (m_k + lambda_c + lambda_g)
            )

            mu_clusters[:, k] = mu_k

        return mu_clusters, mass_clusters

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.epoch_stats = {}
            
    def _loss_mid(self, mu_ref: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
        """
        Align each threshold theta_k with the midpoint between consecutive centers:
            theta_k ~ 0.5 * (mu_k + mu_{k+1})

        Parameters
        ----------
        mu_ref : tensor (..., C)
            Class centers to use for the alignment.
            Typical choice: mu_active or self.mu_prior[active_slots].
        theta_ref : tensor (..., C-1)
            Threshold rows aligned with mu_ref.
            Must have the same leading dimension as mu_ref, or be broadcastable.

        Returns
        -------
        scalar tensor
        """
        if mu_ref.dim() == 1:
            mu_ref = mu_ref.unsqueeze(0)          # (1, C)
        if theta_ref.dim() == 1:
            theta_ref = theta_ref.unsqueeze(0)    # (1, C-1)

        if mu_ref.shape[-1] != self.C:
            raise ValueError(f"mu_ref last dim must be {self.C}, got {mu_ref.shape}")
        if theta_ref.shape[-1] != self.C - 1:
            raise ValueError(f"theta_ref last dim must be {self.C - 1}, got {theta_ref.shape}")

        # Broadcast if one side has one row
        if theta_ref.shape[0] == 1 and mu_ref.shape[0] > 1:
            theta_ref = theta_ref.expand(mu_ref.shape[0], -1)
        elif mu_ref.shape[0] == 1 and theta_ref.shape[0] > 1:
            mu_ref = mu_ref.expand(theta_ref.shape[0], -1)

        if mu_ref.shape[0] != theta_ref.shape[0]:
            raise ValueError(
                f"mu_ref and theta_ref are not aligned: {mu_ref.shape} vs {theta_ref.shape}"
            )

        target_mid = 0.5 * (mu_ref[:, :-1] + mu_ref[:, 1:])

        # Important: detach mu so Lmid mainly calibrates thresholds/bins
        if getattr(self, "mid_detach_mu", True):
            target_mid = target_mid.detach()

        valid = torch.isfinite(target_mid) & torch.isfinite(theta_ref)
        if not valid.any():
            return theta_ref.new_tensor(0.0)

        # Robust regression, better than plain MSE if some centers move abruptly
        return F.smooth_l1_loss(theta_ref[valid], target_mid[valid], reduction="mean")

    def forward(self, score, y_cont, clusters_ids, departement_ids, sample_weight=None):
        s = score.view(-1)
        y = y_cont.view(-1).to(device=s.device).long()

        check_finite("score", s)
        check_finite("y_cont", y_cont)

        clusters_ids = clusters_ids.view(-1).long().to(device=s.device)
        if departement_ids is not None:
            departement_ids = departement_ids.view(-1).long().to(device=s.device)

        sw = sample_weight.view(-1).to(device=s.device) if sample_weight is not None else None
        if sw is not None:
            check_finite("sample_weight", sw)
            assert (sw >= 0).all(), "Negative sample_weight detected"

        cluster_slot_to_raw, cluster_slot_ids, c_valid = self._remap_ids(
            clusters_ids, self.nclusters, kind="cluster"
        )

        if departement_ids is not None:
            dept_slot_to_raw, dept_slot_ids, d_valid = self._remap_ids(
                departement_ids, self.ndepartements, kind="department"
            )
        else:
            dept_slot_to_raw = dept_slot_ids = d_valid = None

        device = s.device

        probs = self._class_probs_from_score(
            s,
            clusters_ids=cluster_slot_ids if self.alphatype == "cluster" else None,
            departement_ids=dept_slot_ids if self.alphatype == "department" else None,
        )
        check_finite("probs", probs)

        probs = torch.nan_to_num(probs, nan=self.eps, posinf=1.0, neginf=0.0)
        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)

        active_cluster_slots, local_cluster_ids = torch.unique(cluster_slot_ids, return_inverse=True)

        if dept_slot_ids is not None:
            active_dept_slots, local_dept_ids = torch.unique(dept_slot_ids, return_inverse=True)
        else:
            active_dept_slots = local_dept_ids = None

        mu, mass = self._mu_soft(
            probs,
            y,
            local_cluster_ids,
            sw=sw,
            active_cluster_slots=active_cluster_slots
        )
        check_finite("mu", mu)
        check_finite("mass", mass)
        check_finite("mu_prior", self.mu_prior_global)

        gains = self._compute_gains()
        active_local_clusters = torch.arange(len(active_cluster_slots), device=device)

        loss = s.new_tensor(0.0)
        wsum = s.new_tensor(0.0)

        mu_active = mu[active_local_clusters]
        mass_active = mass[active_local_clusters]
        check_finite("mu_active", mu_active)
        
        theta_all = self._compute_thresholds().to(device)

        if theta_all.dim() == 1:
            # global thresholds
            theta_mid = theta_all.unsqueeze(0).expand(mu_active.shape[0], -1)

        elif self.alphatype == "cluster":
            # thresholds aligned with active cluster slots
            theta_mid = theta_all.index_select(0, active_cluster_slots)

        elif self.alphatype == "department":
            # simple and safe version:
            # if one active department, broadcast its thresholds to all active clusters
            if active_dept_slots is None:
                theta_mid = theta_all[:1].expand(mu_active.shape[0], -1)
            else:
                theta_dep = theta_all.index_select(0, active_dept_slots)
                if theta_dep.shape[0] == 1:
                    theta_mid = theta_dep.expand(mu_active.shape[0], -1)
                else:
                    # if several departments are active at once, aggregate mu per department
                    # before using Lmid; for now, fallback on the global running centers
                    theta_mid = theta_dep[:1].expand(mu_active.shape[0], -1)

        else:
            raise ValueError(f"Unknown alphatype: {self.alphatype}")

        Lmid = self._loss_mid(mu_active, theta_mid)

        num_active = len(active_local_clusters)
        loss_accum = torch.zeros(num_active, device=device, dtype=s.dtype)
        waccum = 0.0

        tau = float(getattr(self, "tau_loss", 1.5))
        beta = float(getattr(self, "loss_ema_momentum", 0.99))

        loss_ema_dev = self.loss_ema.to(device=device)
        ema_active = loss_ema_dev[active_cluster_slots.clamp(0, loss_ema_dev.shape[0] - 1)].to(dtype=mu_active.dtype)
        w_viol = torch.softmax(tau * ema_active, dim=0)
        w_viol = w_viol.detach() + self.eps

        for k, pairs in self.P.items():
            raw = torch.stack([mu_active[:, b] - mu_active[:, a] for (a, b) in pairs], dim=0)
            margins = torch.stack([gains[a:b].sum() for (a, b) in pairs], dim=0)
            deltas = raw - margins.unsqueeze(1)
            check_finite("deltas_before_scaling", deltas)

            pair_indices = torch.as_tensor(
                [self.pair_to_idx[(a, b)] for (a, b) in pairs],
                device=device,
                dtype=torch.long
            )

            abs_deltas = deltas.detach().abs()
            batch_scale_per_pair = abs_deltas.median(dim=1).values.clamp_min(self.eps)

            with torch.no_grad():
                if self.delta_scale_ema.device != device:
                    self.delta_scale_ema = self.delta_scale_ema.to(device=device)
                if self.scaleagg == "cluster":
                    for ci, slot_t in enumerate(active_cluster_slots):
                        slot = int(slot_t.item())
                        if slot < 0 or slot >= self.delta_scale_ema.shape[0]:
                            continue
                        bs_cid = abs_deltas[:, ci].clamp_min(self.eps)
                        old = self.delta_scale_ema[slot, pair_indices]
                        self.delta_scale_ema[slot, pair_indices] = (
                            old * self.scale_momentum + bs_cid * (1.0 - self.scale_momentum)
                        )
                elif self.scaleagg == "department":
                    if active_dept_slots is not None:
                        for slot_t in active_dept_slots:
                            slot = int(slot_t.item())
                            if slot < 0 or slot >= self.delta_scale_ema.shape[0]:
                                continue
                            old = self.delta_scale_ema[slot, pair_indices]
                            self.delta_scale_ema[slot, pair_indices] = (
                                old * self.scale_momentum + batch_scale_per_pair * (1.0 - self.scale_momentum)
                            )
                else:
                    old = self.delta_scale_ema[pair_indices]
                    self.delta_scale_ema[pair_indices] = (
                        old * self.scale_momentum + batch_scale_per_pair * (1.0 - self.scale_momentum)
                    )

            if self.scaleagg == "cluster":
                sc_raw = self.delta_scale_ema[
                    active_cluster_slots.clamp(0, self.delta_scale_ema.shape[0] - 1)
                ][:, pair_indices].T.to(device=device).clamp(self.scale_min, self.scale_max)
                deltas = deltas / sc_raw
                #deltas = deltas / margins.unsqueeze(1)
            elif self.scaleagg == "department":
                if active_dept_slots is not None:
                    sc_depts = self.delta_scale_ema[
                        active_dept_slots.clamp(0, self.delta_scale_ema.shape[0] - 1).long()
                    ][:, pair_indices].to(device=device).mean(dim=0)
                    sc = sc_depts.clamp(self.scale_min, self.scale_max)
                    deltas = deltas / sc.unsqueeze(1)
                    #deltas = deltas / margins.unsqueeze(1)
            else:
                sc = self.delta_scale_ema[pair_indices].to(device=device).clamp(self.scale_min, self.scale_max)
                deltas = deltas / sc.unsqueeze(1)
                #deltas = deltas / margins.unsqueeze(1)

            check_finite("deltas_after_scaling", deltas)
            check_finite("delta_scale_ema", self.delta_scale_ema)
            
            #print(f'############### Analyse for pair k {pairs} ################')
            #print('margins:', margins)
            
            #print('mu active:', mu_active)
            
            #print('Deltas stats')
            #print(deltas)
            #print(-deltas)
            #print(F.softplus(-deltas))
            
            MEDk = s.new_tensor(0.0)
            MINk = s.new_tensor(0.0)

            loss_med = 0.0
            loss_min = 0.0
            
            loss_neg = F.softplus(-deltas).mean(dim=0)
            
            #print(f'Loss obtained : {loss_neg}')

            check_finite("loss_neg", loss_neg)
            check_finite("MEDk", MEDk)
            check_finite("MINk", MINk)

            Lk_clusters = loss_neg

            w = float(self.wk.get(k, 1.0))
            loss_accum = loss_accum + w * Lk_clusters.detach()
            waccum = waccum + w

            Lk = (Lk_clusters * w_viol).sum() / w_viol.sum()

            #Lk = Lk_clusters
            if not hasattr(self, "epoch_stats"):
                self.epoch_stats = {}
            if "deltas" not in self.epoch_stats:
                self.epoch_stats["deltas"] = {}
            if k not in self.epoch_stats["deltas"]:
                self.epoch_stats["deltas"][k] = {"median": [], "min": [], "viol": [], "neg": []}

            self.epoch_stats["deltas"][k]["median"].append(MEDk.mean().item())
            self.epoch_stats["deltas"][k]["min"].append(MINk.mean().item())
            self.epoch_stats["deltas"][k]["viol"].append((deltas < 0).float().mean().item())
            self.epoch_stats["deltas"][k]["neg"].append(loss_neg.mean().item())

            loss = loss + w * Lk
            wsum = wsum + w

        if waccum > 0:
            loss_mean_per_cluster = loss_accum / waccum
            check_finite("loss_mean_per_cluster", loss_mean_per_cluster)
            with torch.no_grad():
                if self.loss_ema.device != device:
                    self.loss_ema = self.loss_ema.to(device=device)
                for ci, slot_t in enumerate(active_cluster_slots):
                    slot = int(slot_t.item())
                    if 0 <= slot < self.loss_ema.shape[0]:
                        self.loss_ema[slot] = beta * self.loss_ema[slot] + (1.0 - beta) * loss_mean_per_cluster[ci]

        w_mu = mass_active.sum(dim=1).clamp_min(1e-6)
        w_mu_sum = w_mu.sum().clamp_min(1e-6)
        mu_log = (mu_active * w_mu.unsqueeze(1)).sum(dim=0) / w_mu_sum
        if "mu" not in self.epoch_stats:
            self.epoch_stats["mu"] = []
        self.epoch_stats["mu"].append(mu_log.detach().cpu().numpy())

        if not hasattr(self, "epoch_stats"):
            self.epoch_stats = {}
        if "cluster_weights" not in self.epoch_stats:
            self.epoch_stats["cluster_weights"] = []
        cw_full = torch.zeros(self.loss_ema.shape[0], dtype=w_viol.dtype, device=device)
        for ci, slot_t in enumerate(active_cluster_slots):
            slot = int(slot_t.item())
            if 0 <= slot < cw_full.shape[0]:
                cw_full[slot] = w_viol[ci].to(device=device)
        self.epoch_stats["cluster_weights"].append(cw_full.detach().cpu().numpy())

        if "mass_active" not in self.epoch_stats:
            self.epoch_stats["mass_active"] = []
        ma_full = torch.zeros(self.mu_prior.shape[0], self.C, device=device, dtype=mass_active.dtype)
        for ci, slot_t in enumerate(active_cluster_slots):
            slot = int(slot_t.item())
            ma_full[slot] = mass_active[ci].to(device=device).detach()
        self.epoch_stats["mass_active"].append(ma_full.detach().cpu().numpy())

        transition_loss = loss / wsum.clamp_min(1e-6)
        #print('wsum:', wsum.clamp_min(1e-6))
        #print('transition_loss:', transition_loss)
        target_bin = (y > 0).to(dtype=s.dtype)
        prob_fire = (1.0 - probs[:, 0]).clamp(self.eps, 1.0 - self.eps)
        p_t = torch.where(target_bin > 0.5, prob_fire, 1.0 - prob_fire)
        alpha_t = torch.where(
            target_bin > 0.5,
            torch.full_like(p_t, self.falpha),
            torch.full_like(p_t, 1.0 - self.falpha)
        )

        focal_weight = alpha_t * (1.0 - p_t).pow(self.fgamma)
        focal_ce = -torch.log(p_t)

        check_finite("prob_fire", prob_fire)
        check_finite("p_t", p_t)
        check_finite("focal_ce", focal_ce)
        assert (p_t > 0).all(), "p_t contains zeros -> log instability"

        if sw is not None:
            sw_norm = sw / sw.sum().clamp_min(1e-6) * sw.numel()
            focal_loss = (focal_weight * focal_ce * sw_norm).mean()
        else:
            focal_loss = (focal_weight * focal_ce).mean()

        if "focal" not in self.epoch_stats:
            self.epoch_stats["focal"] = []
        self.epoch_stats["focal"].append(focal_loss.item())

        if "transition" not in self.epoch_stats:
            self.epoch_stats["transition"] = []
        self.epoch_stats["transition"].append(transition_loss.item())
        
        if "mid" not in self.epoch_stats:
            self.epoch_stats["mid"] = []
        self.epoch_stats["mid"].append(Lmid.item())

        mu0_val = mu_log[0]
        if not torch.isfinite(mu0_val):
            mu0_val = mu0_val.new_tensor(0.0)
        else:
            mu0_val = mu0_val.clamp(-100.0, 100.0)

        mu0_term = F.softplus(mu0_val)

        if "mu0_term" not in self.epoch_stats:
            self.epoch_stats["mu0_term"] = []
        self.epoch_stats["mu0_term"].append(mu0_term.item())

        check_finite("transition_loss", transition_loss)
        check_finite("focal_loss", focal_loss)
        check_finite("mu0_term", mu0_term)

        try:
            total_loss = \
                self.wtrans * transition_loss \
                + self.wfocal * focal_loss \
                + self.wmu0 * mu0_term \
                + self.wmid * Lmid
                
        except Exception as e:
            print("DEBUG NAN SOURCE:")
            print("score:", s)
            print("probs:", probs)
            print("mu:", mu)
            print("scale:", self.delta_scale_ema)
            raise e

        return total_loss

    def get_learnable_parameters(self):
        params = {"alpha": self.alpha}

        if getattr(self, "learn_gains", False):
            if hasattr(self, "g_raw"):
                params["g_raw"] = self.g_raw
            elif hasattr(self, "gain_raw"):
                params["gain_raw"] = self.gain_raw
        return params

    @torch.no_grad()
    def score_to_class(self, scores: torch.Tensor, clusters_ids: torch.Tensor = None, departement_ids: torch.Tensor = None) -> torch.Tensor:
        s = scores.detach().to(dtype=self.alpha.dtype).flatten().unsqueeze(1)
        device = s.device

        thr = self._compute_thresholds().detach().to(device=device)

        if thr.dim() == 1:
            return torch.bucketize(scores.flatten(), thr, right=True)
        else:
            if self.alphatype == "cluster":
                chosen_ids = clusters_ids
            elif self.alphatype == "department":
                chosen_ids = departement_ids
            else:
                chosen_ids = (clusters_ids if clusters_ids is not None else departement_ids)

            if chosen_ids is None:
                raise ValueError("IDs are required when thresholds are cluster/department-specific.")
            else:
                chosen_ids = chosen_ids.view(-1).long().to(device=device)
                if self.alphatype == "cluster":
                    _, idx, _ = self._remap_ids(chosen_ids, self.nclusters, kind="cluster")
                elif self.alphatype == "department":
                    _, idx, _ = self._remap_ids(chosen_ids, self.ndepartements, kind="department")
                else:
                    raise ValueError(f"Unknown alphatype: {self.alphatype}")

            thr_s = thr.index_select(0, idx)
            return (s > thr_s).sum(dim=1)

    def get_attribute(self):
        payload = {
            "alpha": self.alpha.detach().cpu().numpy(),
            "thresholds": self._compute_thresholds().detach().cpu().numpy(),
            "mu_prior": self.mu_prior.detach().cpu().numpy(),
            "mu_prior_global": self.mu_prior_global.detach().cpu().numpy(),
            "cluster_slot_to_raw": self.cluster_slot_to_raw.detach().cpu().numpy(),
            "departement_slot_to_raw": self.departement_slot_to_raw.detach().cpu().numpy(),
        }

        if getattr(self, "learn_gains", False):
            g = self._compute_gains().detach().cpu().numpy() if hasattr(self, "_compute_gains") else None
            if g is None:
                g = self.gains.detach().cpu().numpy() if hasattr(self, "gains") and self.gains is not None else None
            if g is None and hasattr(self, "g_raw"):
                floor = float(getattr(self, "gains_floor", 0.0))
                g = (F.softplus(self.g_raw) + floor).detach().cpu().numpy()
            if g is not None:
                payload["gains"] = g

        if hasattr(self, "epoch_stats") and "deltas" in self.epoch_stats:
            agg_deltas = {}
            for k, dstats in self.epoch_stats["deltas"].items():
                agg_deltas[k] = {
                    "median": np.mean(dstats["median"]) if dstats["median"] else 0.0,
                    "min": np.mean(dstats["min"]) if dstats["min"] else 0.0,
                    "viol": np.mean(dstats["viol"]) if dstats["viol"] else 0.0,
                    "neg": np.mean(dstats["neg"]) if dstats["neg"] else 0.0,
                }
            payload["deltas"] = agg_deltas

        if hasattr(self, "epoch_stats") and "mu" in self.epoch_stats and len(self.epoch_stats["mu"]) > 0:
            mu_stack = np.stack(self.epoch_stats["mu"])
            payload["mu"] = np.mean(mu_stack, axis=0)

        if hasattr(self, "epoch_stats"):
            for _lkey in ("transition", "focal"):
                vals = self.epoch_stats.get(_lkey, [])
                if vals:
                    payload[_lkey] = [float(np.mean(vals))]

            vals = self.epoch_stats.get("mid", [])
            if vals:
                payload["mid"] = [float(np.mean(vals))]

        payload["delta_scale_ema"] = self.delta_scale_ema.detach().cpu().numpy()

        if hasattr(self, "epoch_stats") and self.epoch_stats.get("cluster_weights"):
            cw_stack = np.stack(self.epoch_stats["cluster_weights"])
            payload["cluster_weights"] = np.mean(cw_stack, axis=0)

        if hasattr(self, "epoch_stats") and self.epoch_stats.get("mass_active"):
            ma_stack = np.stack(self.epoch_stats["mass_active"])
            payload["mass_active"] = np.max(ma_stack, axis=0)

        return [("ordinal_params", DictWrapper(payload))]

    def update_params(self, new_dict, epoch=None):
        """
        Accepte soit :
        1) directement un payload dict
        2) un DictWrapper(payload)
        3) un dict externe du type:
            {"epoch": ..., "ordinal_params": DictWrapper(payload)}
        et met à jour alpha, mu_prior, mu_prior_global si présents.
        """

        # --------------------------------------------------
        # 1) Déplier la structure externe
        # --------------------------------------------------
        payload = new_dict

        # Cas: {"epoch": ..., "ordinal_params": DictWrapper(...)}
        if isinstance(payload, dict) and "ordinal_params" in payload:
            if epoch is None and "epoch" in payload:
                epoch = payload["epoch"]
            payload = payload["ordinal_params"]

        # Cas: DictWrapper(payload)
        if hasattr(payload, "numpy") and not isinstance(payload, dict):
            payload = payload.numpy()

        # Sécurité finale
        if not isinstance(payload, dict):
            raise TypeError(
                f"update_params expected a dict-like payload after unwrapping, got {type(payload)}"
            )

        # --------------------------------------------------
        # 2) alpha
        # --------------------------------------------------
        
        print('Old alpha', self.alpha)
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

        # --------------------------------------------------
        # 3) mu_prior
        # --------------------------------------------------
        if "mu_prior" in payload and payload["mu_prior"] is not None:
            mu_prior_new = torch.as_tensor(
                payload["mu_prior"],
                dtype=self.mu_prior.dtype,
                device=self.mu_prior.device,
            )
            if mu_prior_new.shape != self.mu_prior.shape:
                raise ValueError(
                    f"mu_prior shape mismatch: got {tuple(mu_prior_new.shape)}, "
                    f"expected {tuple(self.mu_prior.shape)}"
                )
            with torch.no_grad():
                self.mu_prior.copy_(mu_prior_new)

        # --------------------------------------------------
        # 4) mu_prior_global
        # --------------------------------------------------
        if "mu_prior_global" in payload and payload["mu_prior_global"] is not None:
            mu_prior_global_new = torch.as_tensor(
                payload["mu_prior_global"],
                dtype=self.mu_prior_global.dtype,
                device=self.mu_prior_global.device,
            )
            if mu_prior_global_new.shape != self.mu_prior_global.shape:
                raise ValueError(
                    f"mu_prior_global shape mismatch: got {tuple(mu_prior_global_new.shape)}, "
                    f"expected {tuple(self.mu_prior_global.shape)}"
                )
            with torch.no_grad():
                self.mu_prior_global.copy_(mu_prior_global_new)

        # --------------------------------------------------
        # 5) resynchronisation des seuils dérivés
        # --------------------------------------------------
        self.thresholds = self._compute_thresholds().detach()

        if getattr(self, "learn_gains", False):
            if hasattr(self, "_compute_gains"):
                self.gains = self._compute_gains().detach()
            elif hasattr(self, "g_raw"):
                floor = float(getattr(self, "gains_floor", 0.0))
                self.gains = (F.softplus(self.g_raw) + floor).detach()

        print('New alpha', self.alpha)
        
    def plot_params(self, params_history, log_dir, best_epoch=None):
        import matplotlib.pyplot as plt

        root_dir = log_dir / "ordinal_params"
        root_dir.mkdir(parents=True, exist_ok=True)

        epochs = []
        thresholds_list = []
        gains_list = []
        deltas_list = []
        mu_list = []
        delta_scale_ema_list = []
        mu_prior_list = [] 
        mu_prior_global_list = []
        cluster_weights_list = []
        mass_active_list = []
        cluster_slot_to_raw_list = []

        iterator = []
        if isinstance(params_history, dict):
            iterator = sorted(params_history.items())
        else:
            for entry in params_history:
                if isinstance(entry, dict) and ("epoch" in entry):
                    iterator.append((entry["epoch"], entry))
            iterator.sort(key=lambda x: x[0])

        for ep, entry in iterator:
            if "ordinal_params" not in entry:
                continue

            stats_container = entry["ordinal_params"]
            p = stats_container.d if hasattr(stats_container, "d") else stats_container

            if not isinstance(p, dict):
                continue

            if "thresholds" not in p:
                continue

            epochs.append(ep)
            thresholds_list.append(p["thresholds"])
            gains_list.append(p.get("gains", None))
            deltas_list.append(p.get("deltas", None))
            mu_list.append(p.get("mu", None))
            delta_scale_ema_list.append(p.get("delta_scale_ema", None))
            mu_prior_list.append(p.get("mu_prior", None))
            mu_prior_global_list.append(p.get("mu_prior_global", None))
            cluster_weights_list.append(p.get("cluster_weights", None))
            mass_active_list.append(p.get("mass_active", None))
            cluster_slot_to_raw_list.append(p.get("cluster_slot_to_raw", None))

        if not epochs:
            return

        thresholds_arr = np.array(thresholds_list)

        fig, ax = plt.subplots(figsize=(8, 6))
        if thresholds_arr.ndim == 2:
            for i in range(thresholds_arr.shape[1]):
                ax.plot(epochs, thresholds_arr[:, i], label=f"theta_{i}")
        else:
            th_mean = thresholds_arr.mean(axis=1)
            for i in range(th_mean.shape[1]):
                ax.plot(epochs, th_mean[:, i], label=f"theta_{i}")
        ax.set_title(f"{self.__class__.__name__} Thresholds Evolution")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Threshold Value")
        ax.grid(True, alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
        ax.legend()
        plt.tight_layout()
        plt.savefig(root_dir / "thresholds_evolution.png")
        plt.close()

        try:
            valid_gains = [(ep, g) for ep, g in zip(epochs, gains_list) if g is not None]
            if valid_gains:
                g_epochs, g_vals = zip(*valid_gains)
                gains_arr = np.array(g_vals)
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(gains_arr.shape[1]):
                    ax.plot(list(g_epochs), gains_arr[:, i], label=f"gain_{i}")
                ax.set_title(f"{self.__class__.__name__} Gains Evolution")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Gain Value")
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                ax.legend()
                plt.tight_layout()
                plt.savefig(root_dir / "gains_evolution.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_deltas = [(ep, d) for ep, d in zip(epochs, deltas_list) if d is not None]
            if valid_deltas:
                d_epochs, d_vals = zip(*valid_deltas)
                d_epochs = list(d_epochs)
                ks = sorted(d_vals[0].keys())
                if ks:
                    fig, axes = plt.subplots(len(ks), 4, figsize=(20, 3 * len(ks)), sharex=True)
                    if len(ks) == 1:
                        axes = axes[None, :]
                    for i, k in enumerate(ks):
                        axes[i, 0].plot(d_epochs, [d[k]["median"] for d in d_vals if k in d], color="blue")
                        axes[i, 0].set_title(f"k={k} Median Delta")
                        axes[i, 1].plot(d_epochs, [d[k]["min"] for d in d_vals if k in d], color="red")
                        axes[i, 1].set_title(f"k={k} Min Delta")
                        axes[i, 2].plot(d_epochs, [d[k]["viol"] for d in d_vals if k in d], color="orange")
                        axes[i, 2].set_title(f"k={k} Viol Rate")
                        axes[i, 2].set_ylim(-0.1, 1.1)
                        axes[i, 3].plot(d_epochs, [d[k]["neg"] for d in d_vals if k in d], color="purple")
                        axes[i, 3].set_title(f"k={k} Mean NEG")
                    plt.tight_layout()
                    plt.savefig(root_dir / "deltas_stats.png")
                    plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_scales = [(ep, s) for ep, s in zip(epochs, delta_scale_ema_list) if s is not None]
            if valid_scales:
                s_epochs, s_vals = zip(*valid_scales)
                scales_arr = np.array(s_vals)
                ep_list = list(s_epochs)

                def _pair_label(i):
                    if i < len(self.all_pairs):
                        a, b = self.all_pairs[i]
                        return f"{a}→{b}"
                    return f"pair_{i}"

                if scales_arr.ndim == 2:
                    num_p = scales_arr.shape[1]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for i in range(num_p):
                        ax.plot(ep_list, scales_arr[:, i], label=_pair_label(i))
                    ax.set_title(f"{self.__class__.__name__} – Delta Scale EMA per pair")
                    ax.set_yscale("log")
                    ax.set_xlabel("Epoch")
                    ax.grid(True, alpha=0.3)
                    if best_epoch is not None:
                        ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                    ax.legend(fontsize=6, ncol=max(1, num_p // 8))
                    plt.tight_layout()
                    plt.savefig(root_dir / "delta_scale_ema_evolution.png")
                    plt.close()
                else:
                    E, ncl, num_p = scales_arr.shape
                    mean_arr = scales_arr.mean(axis=1)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for i in range(num_p):
                        ax.plot(ep_list, mean_arr[:, i], label=_pair_label(i))
                    ax.set_title(f"{self.__class__.__name__} – Delta Scale EMA (mean over groups)")
                    ax.set_yscale("log")
                    ax.set_xlabel("Epoch")
                    ax.grid(True, alpha=0.3)
                    if best_epoch is not None:
                        ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                    ax.legend(fontsize=6, ncol=max(1, num_p // 8))
                    plt.tight_layout()
                    plt.savefig(root_dir / "delta_scale_ema_evolution.png")
                    plt.close()

                    last = scales_arr[-1]
                    pair_labels = [_pair_label(i) for i in range(num_p)]
                    fig, ax = plt.subplots(figsize=(max(8, num_p * 0.4), max(4, ncl * 0.4)))
                    im = ax.imshow(np.log10(last + 1e-12), aspect="auto", origin="upper")
                    ax.set_title(f"log10(Delta Scale EMA) heatmap – epoch {ep_list[-1]}")
                    ax.set_xlabel("(a,b) pair")
                    ax.set_ylabel("Cluster slot")
                    ax.set_xticks(range(num_p))
                    ax.set_xticklabels(pair_labels, rotation=90, fontsize=6)
                    ax.set_yticks(range(ncl))
                    fig.colorbar(im, ax=ax, label="log10(scale)")
                    plt.tight_layout()
                    plt.savefig(root_dir / "delta_scale_ema_heatmap_last.png")
                    plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_mu_priors = [(ep, m) for ep, m in zip(epochs, mu_prior_list) if m is not None]
            if valid_mu_priors:
                m_epochs, m_vals = zip(*valid_mu_priors)
                mu_arr = np.array(m_vals)
                if mu_arr.ndim == 3:
                    mu_arr_mean = np.nanmean(mu_arr, axis=1)
                else:
                    mu_arr_mean = mu_arr
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(mu_arr_mean.shape[1]):
                    ax.plot(list(m_epochs), mu_arr_mean[:, i], label=f"mu_prior_avg_c={i}", alpha=0.4)

                valid_globals = [(ep_g, mg) for ep_g, mg in zip(epochs, mu_prior_global_list) if mg is not None]
                if valid_globals:
                    eg, m_vals_g = zip(*valid_globals)
                    mu_g_arr = np.array(m_vals_g)
                    for i in range(mu_g_arr.shape[1]):
                        ax.plot(list(eg), mu_g_arr[:, i], label=f"mu_prior_global_c={i}", linewidth=2, linestyle="--")
                ax.set_title("Mu Prior Evolution (Global vs Avg-Cluster)")
                ax.legend(fontsize="x-small", ncol=2)
                plt.tight_layout()
                plt.savefig(root_dir / "mu_prior_evolution.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_locals = [(ep, mp) for ep, mp in zip(epochs, mu_prior_list) if mp is not None]
            if valid_locals:
                epl, mp_vals = zip(*valid_locals)
                mp_arr = np.stack(mp_vals)
                fig, ax = plt.subplots(figsize=(10, 6))
                C = mp_arr.shape[2]
                for c in range(C):
                    series = mp_arr[:, :, c]
                    mean_c = np.nanmean(series, axis=1)
                    p10 = np.nanpercentile(series, 10, axis=1)
                    p90 = np.nanpercentile(series, 90, axis=1)
                    ax.plot(list(epl), mean_c, label=f"local_mean_c={c}")
                    ax.fill_between(list(epl), p10, p90, alpha=0.15)
                if best_epoch is not None:
                    ax.axvline(best_epoch, linestyle="--", alpha=0.5, label="Best Epoch")
                ax.set_title("Local mu_prior summary (mean + 10-90% band)")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("mu_prior value")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize="x-small", ncol=2)
                plt.tight_layout()
                plt.savefig(root_dir / "mu_prior_local_summary.png")
                plt.close()

                last = mp_arr[-1]
                fig, ax = plt.subplots(figsize=(8, 6))
                fill_val = np.nanmin(last[np.isfinite(last)]) if np.isfinite(last).any() else 0.0
                im = ax.imshow(np.nan_to_num(last, nan=fill_val), aspect="auto")
                ax.set_title(f"Local mu_prior heatmap (last epoch={epl[-1]})")
                ax.set_xlabel("Class")
                ax.set_ylabel("Cluster slot")
                ax.set_xticks(range(last.shape[1]))
                fig.colorbar(im, ax=ax, shrink=0.8)
                plt.tight_layout()
                plt.savefig(root_dir / "mu_prior_local_heatmap_last.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_mu = [(ep, m) for ep, m in zip(epochs, mu_list) if m is not None]
            if valid_mu:
                m_epochs, m_vals = zip(*valid_mu)
                mu_arr = np.stack(m_vals)
                fig, ax = plt.subplots(figsize=(10, 6))
                for c in range(mu_arr.shape[1]):
                    ax.plot(list(m_epochs), mu_arr[:, c], label=f"mu(class {c})")
                ax.set_title("Mu evolution per class")
                ax.legend()
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                plt.tight_layout()
                plt.savefig(root_dir / "mu_s.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_mup = [(ep, m) for ep, m in zip(epochs, mu_prior_list) if m is not None]
            if valid_mup:
                mp_epochs, mp_vals = zip(*valid_mup)
                mp_arr = np.array(mp_vals)
                slot_maps = [np.asarray(x) if x is not None else None for x in cluster_slot_to_raw_list]

                if mp_arr.ndim == 3:
                    n_buf = mp_arr.shape[1]
                    n_classes = mp_arr.shape[2]
                    cluster_slots_to_plot = [cl for cl in range(n_buf) if np.isfinite(mp_arr[:, cl, :]).any()]
                    n_plot = len(cluster_slots_to_plot)

                    if n_plot > 0:
                        cols = min(4, n_plot)
                        rows = (n_plot + cols - 1) // cols
                        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharey=True, sharex=True)
                        if rows == 1 and cols == 1:
                            axes = np.array([[axes]])
                        elif rows == 1:
                            axes = axes[None, :]
                        axes_flat = axes.flatten()
                        cmap = plt.cm.plasma

                        last_slot_map = slot_maps[-1] if len(slot_maps) > 0 else None

                        for plot_idx, cl in enumerate(cluster_slots_to_plot):
                            ax = axes_flat[plot_idx]
                            for c in range(n_classes):
                                ax.plot(list(mp_epochs), mp_arr[:, cl, c], color=cmap(c / max(n_classes - 1, 1)), label=f"class {c}")
                            if best_epoch is not None:
                                ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)

                            if last_slot_map is not None and cl < len(last_slot_map) and last_slot_map[cl] >= 0:
                                ax.set_title(f"slot {cl} / raw {int(last_slot_map[cl])}")
                            else:
                                ax.set_title(f"slot {cl}")

                            ax.set_xlabel("Epoch")
                            ax.grid(True, alpha=0.3)

                        for j in range(n_plot, len(axes_flat)):
                            axes_flat[j].set_visible(False)

                        axes_flat[n_plot - 1].legend(fontsize=7, loc="best")
                        fig.suptitle(f"{self.__class__.__name__} — Mu Prior per cluster slot", fontsize=12)
                        plt.tight_layout()
                        plt.savefig(root_dir / "mu_per_cluster.png")
                        plt.close()
        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] mu_per_cluster error: {_e}")

        try:
            valid_cw = [(ep, m) for ep, m in zip(epochs, cluster_weights_list) if m is not None]
            if valid_cw:
                cw_epochs, cw_vals = zip(*valid_cw)
                cw_arr = np.array(cw_vals)
                slot_maps = [np.asarray(x) if x is not None else None for x in cluster_slot_to_raw_list]

                if cw_arr.ndim == 2:
                    n_buf = cw_arr.shape[1]
                    cluster_slots_to_plot = [cl for cl in range(n_buf) if cw_arr[:, cl].any()]
                    n_cl = len(cluster_slots_to_plot)
                    if n_cl > 0:
                        cols = min(4, n_cl)
                        rows = (n_cl + cols - 1) // cols
                        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True, sharex=True)
                        if rows == 1 and cols == 1:
                            axes = np.array([[axes]])
                        elif rows == 1:
                            axes = axes[None, :]
                        axes_flat = axes.flatten()

                        last_slot_map = slot_maps[-1] if len(slot_maps) > 0 else None

                        for plot_idx, cl in enumerate(cluster_slots_to_plot):
                            ax = axes_flat[plot_idx]
                            ax.plot(list(cw_epochs), cw_arr[:, cl], color="steelblue")
                            if best_epoch is not None:
                                ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)

                            if last_slot_map is not None and cl < len(last_slot_map) and last_slot_map[cl] >= 0:
                                ax.set_title(f"slot {cl} / raw {int(last_slot_map[cl])}")
                            else:
                                ax.set_title(f"slot {cl}")

                            ax.set_xlabel("Epoch")
                            ax.set_ylabel("w_z")
                            ax.grid(True, alpha=0.3)

                        for j in range(n_cl, len(axes_flat)):
                            axes_flat[j].set_visible(False)

                        fig.suptitle(f"{self.__class__.__name__} — Cluster EMA weights (softmax)", fontsize=11)
                        plt.tight_layout()
                        plt.savefig(root_dir / "cluster_weights_evolution.png")
                        plt.close()
        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] cluster_weights_evolution error: {_e}")

        try:
            valid_ma = [(ep, m) for ep, m in zip(epochs, mass_active_list) if m is not None]
            if valid_ma:
                ma_epochs, ma_vals = zip(*valid_ma)
                ma_arr = np.array(ma_vals)
                slot_maps = [np.asarray(x) if x is not None else None for x in cluster_slot_to_raw_list]

                if ma_arr.ndim == 3:
                    n_buf = ma_arr.shape[1]
                    n_classes = ma_arr.shape[2]
                    cluster_slots_to_plot = [cl for cl in range(n_buf) if ma_arr[:, cl, :].any()]
                    n_cl = len(cluster_slots_to_plot)
                    if n_cl > 0:
                        cols = min(4, n_cl)
                        rows = (n_cl + cols - 1) // cols
                        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharey=False, sharex=True)
                        if rows == 1 and cols == 1:
                            axes = np.array([[axes]])
                        elif rows == 1:
                            axes = axes[None, :]
                        axes_flat = axes.flatten()
                        cmap = plt.cm.plasma

                        last_slot_map = slot_maps[-1] if len(slot_maps) > 0 else None

                        for plot_idx, cl in enumerate(cluster_slots_to_plot):
                            ax = axes_flat[plot_idx]
                            for c in range(n_classes):
                                ax.plot(list(ma_epochs), ma_arr[:, cl, c], color=cmap(c / max(n_classes - 1, 1)), alpha=0.7, label=f"class {c}")
                            total_mass = ma_arr[:, cl, :].sum(axis=1)
                            ax.plot(list(ma_epochs), total_mass, color="black", linewidth=1.5, linestyle="--", label="total")
                            if best_epoch is not None:
                                ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)

                            if last_slot_map is not None and cl < len(last_slot_map) and last_slot_map[cl] >= 0:
                                ax.set_title(f"slot {cl} / raw {int(last_slot_map[cl])}")
                            else:
                                ax.set_title(f"slot {cl}")

                            ax.set_xlabel("Epoch")
                            ax.set_ylabel("mass")
                            ax.grid(True, alpha=0.3)

                        for j in range(n_cl, len(axes_flat)):
                            axes_flat[j].set_visible(False)

                        axes_flat[n_cl - 1].legend(fontsize=7, loc="best")
                        fig.suptitle(f"{self.__class__.__name__} — Mass per cluster slot (per class)", fontsize=11)
                        plt.tight_layout()
                        plt.savefig(root_dir / "mass_active_per_cluster.png")
                        plt.close()
        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] mass_active_per_cluster error: {_e}")

        try:
            tr_list, f_list, ep_list = [], [], []
            for ep, entry in iterator:
                p = entry["ordinal_params"].d if hasattr(entry["ordinal_params"], "d") else entry["ordinal_params"]
                if "transition" in p and "focal" in p:
                    ep_list.append(ep)
                    tr_list.append(np.mean(p["transition"]))
                    f_list.append(np.mean(p["focal"]))
            if ep_list:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ep_list, tr_list, label="transition")
                ax.plot(ep_list, f_list, label="focal", linestyle="--")
                ax.set_title(f"{self.__class__.__name__} – Loss components")
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                ax.legend()
                plt.tight_layout()
                plt.savefig(root_dir / "loss_components.png")
                plt.close()
        except Exception:
            plt.close("all")
            
class ClusterDepartmentRankNetLoss(nn.Module):
    """
    RankNet pairwise loss + thresholds learned per department or per cluster.

    Idée
    ----
    - Le terme principal est un RankNet pairwise :
          y_i > y_j  =>  s_i > s_j
      avec BCEWithLogits sur sigma * (s_i - s_j).

    - Les seuils ne servent PAS au ranking lui-même.
      Ils servent à :
        1) convertir un score latent en classe ordinale
        2) éventuellement régulariser la géométrie des seuils via Lmid

    - On peut apprendre :
        * un seul jeu de seuils globaux
        * un jeu de seuils par cluster
        * un jeu de seuils par department

    Paramètres clés
    ---------------
    num_classes : int
        Nombre de classes ordinales finales.
    sigma : float
        Pente du RankNet logit : logits = sigma * (s_i - s_j).
    num_pairs_per_group : Optional[int]
        Nombre de paires échantillonnées par groupe de ranking.
        Si None, utilise toutes les paires du groupe.
    tie_epsilon : float
        Ignore les paires où |y_i - y_j| <= tie_epsilon.
    use_soft_targets : bool
        Si True, cible pairwise douce :
            target_ij = sigmoid((y_i - y_j)/T)
        au lieu de {0,1}.
    soft_target_temperature : float
        Température pour les cibles douces.
    weight_by_delta : bool
        Pondère les paires par |y_i - y_j|^delta_power.
    delta_power : float
        Exposant de la pondération pairwise.
    wrank : float
        Poids du terme RankNet.
    wmid : float
        Poids du terme Lmid de calibration des seuils.
    alphatype : str
        "global", "cluster", ou "department"
        -> où l'on apprend les seuils.
    pair_scope : str
        "global", "cluster", ou "department"
        -> dans quel groupe on forme les paires RankNet.
    nclusters : int
        Nombre maximal de clusters attendus.
    ndepartements : int
        Nombre maximal de départements attendus.
    id : int
        Identifiant optionnel, comme dans ta loss actuelle.
    """

    def __init__(
        self,
        num_classes: int,
        sigma: float = 1.0,
        num_pairs_per_group: Optional[int] = 2048,
        tie_epsilon: float = 0.0,
        use_soft_targets: bool = False,
        soft_target_temperature: float = 1.0,
        weight_by_delta: bool = True,
        delta_power: float = 1.0,
        wrank: float = 1.0,
        wmid: float = 0.0,
        alphatype: str = "department",     # where thresholds are learned
        pair_scope: str = "department",    # where ranking comparisons are formed
        nclusters: int = 1,
        ndepartements: int = 1,
        id: int = 0,
    ):
        super().__init__()

        self.C = int(num_classes)
        self.id = int(id)

        self.sigma = float(sigma)
        self.num_pairs_per_group = num_pairs_per_group
        self.tie_epsilon = float(tie_epsilon)
        self.use_soft_targets = bool(use_soft_targets)
        self.soft_target_temperature = float(soft_target_temperature)
        self.weight_by_delta = bool(weight_by_delta)
        self.delta_power = float(delta_power)

        self.wrank = float(wrank)
        self.wmid = float(wmid)

        self.alphatype = str(alphatype).lower()
        self.pair_scope = str(pair_scope).lower()

        self.nclusters = int(nclusters)
        self.ndepartements = int(ndepartements)

        if self.C < 2:
            raise ValueError("num_classes must be >= 2")
        if self.alphatype not in {"global", "cluster", "department"}:
            raise ValueError("alphatype must be one of: global, cluster, department")
        if self.pair_scope not in {"global", "cluster", "department"}:
            raise ValueError("pair_scope must be one of: global, cluster, department")

        # -----------------------------
        # Raw-id -> local slot mapping
        # -----------------------------
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

        # -----------------------------
        # Threshold parameters
        # alpha -> thresholds monotones via cumulative softplus
        # -----------------------------
        if self.alphatype == "global":
            self.alpha = nn.Parameter(torch.zeros(self.C - 1))
        elif self.alphatype == "cluster":
            self.alpha = nn.Parameter(torch.zeros(self.nclusters, self.C - 1))
        else:  # department
            self.alpha = nn.Parameter(torch.zeros(self.ndepartements, self.C - 1))

        # buffer de confort, mis à jour par update_params()
        init_thr = torch.linspace(-1.0, 1.0, self.C - 1)
        self.register_buffer("thresholds", init_thr.clone())

        self.epoch_stats: Dict[str, list] = {
            "rank": [],
            "mid": [],
            "n_pairs": [],
        }

    # =========================================================
    # Utilities
    # =========================================================
    @staticmethod
    def _validate_1d(name: str, x: torch.Tensor):
        if x.dim() != 1:
            raise ValueError(f"{name} must be 1D")

    def _remap_ids(self, raw_ids: torch.Tensor, buf_size: int, kind: str):
        """
        Remap raw ids -> contiguous local slots in [0 .. buf_size-1]
        """
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
        valid_mask = torch.ones_like(local_ids, dtype=torch.bool, device=device)
        return slot_to_raw.clone(), local_ids, valid_mask

    def _compute_thresholds(self):
        """
        Enforce strictly increasing thresholds by cumulative softplus increments.
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

        theta0 = alpha[:, 0:1]
        if alpha.size(1) > 1:
            incr = F.softplus(alpha[:, 1:])
            theta = torch.cat([theta0, incr], dim=1).cumsum(dim=1)
        else:
            theta = theta0
        return theta

    def _threshold_rows_for_samples(
        self,
        s: torch.Tensor,
        cluster_slot_ids: Optional[torch.Tensor],
        dept_slot_ids: Optional[torch.Tensor],
    ):
        """
        Retourne les thresholds par échantillon selon alphatype.
        """
        theta = self._compute_thresholds().to(device=s.device, dtype=s.dtype)

        if theta.dim() == 1:
            return theta[None, :].expand(s.numel(), -1)

        if self.alphatype == "cluster":
            if cluster_slot_ids is None:
                raise ValueError("cluster_slot_ids is required when alphatype='cluster'")
            return theta.index_select(0, cluster_slot_ids.long())

        if self.alphatype == "department":
            if dept_slot_ids is None:
                raise ValueError("dept_slot_ids is required when alphatype='department'")
            return theta.index_select(0, dept_slot_ids.long())

        raise ValueError(f"Unknown alphatype: {self.alphatype}")

    def _group_ids_for_pairwise(
        self,
        s: torch.Tensor,
        cluster_slot_ids: Optional[torch.Tensor],
        dept_slot_ids: Optional[torch.Tensor],
    ):
        """
        Détermine dans quel scope on forme les paires RankNet.
        """
        if self.pair_scope == "global":
            return torch.zeros(s.numel(), device=s.device, dtype=torch.long)

        if self.pair_scope == "cluster":
            if cluster_slot_ids is None:
                raise ValueError("cluster ids required when pair_scope='cluster'")
            _, local_group_ids = torch.unique(cluster_slot_ids, return_inverse=True)
            return local_group_ids

        if self.pair_scope == "department":
            if dept_slot_ids is None:
                raise ValueError("department ids required when pair_scope='department'")
            _, local_group_ids = torch.unique(dept_slot_ids, return_inverse=True)
            return local_group_ids

        raise ValueError(f"Unknown pair_scope: {self.pair_scope}")

    # =========================================================
    # RankNet loss
    # =========================================================
    def _build_pairs_for_group(
        self,
        idx: torch.Tensor,
        scores: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ):
        """
        Build pairwise logits / targets / weights inside one group.
        """
        yg = y[idx]
        sg = scores[idx]
        wg = sample_weight[idx] if sample_weight is not None else None

        n = idx.numel()
        if n <= 1:
            return None

        # -----------------------------
        # all pairs or sampled pairs
        # -----------------------------
        if self.num_pairs_per_group is None:
            ii, jj = torch.triu_indices(n, n, offset=1, device=idx.device)
        else:
            ii = torch.randint(0, n, (self.num_pairs_per_group,), device=idx.device)
            jj = torch.randint(0, n, (self.num_pairs_per_group,), device=idx.device)
            mask = ii != jj
            ii, jj = ii[mask], jj[mask]
            if ii.numel() == 0:
                return None
            
        yi, yj = yg[ii], yg[jj]
        si, sj = sg[ii], sg[jj]

        dy = yi - yj
        abs_dy = dy.abs()

        # ignore ties / quasi-ties
        valid = abs_dy > self.tie_epsilon
        if not valid.any():
            return None

        yi, yj = yi[valid], yj[valid]
        si, sj = si[valid], sj[valid]
        dy = yi - yj
        abs_dy = abs_dy[valid]

        # target pairwise
        if self.use_soft_targets:
            target = torch.sigmoid(
                dy / max(self.soft_target_temperature, 1e-8)
            )
        else:
            target = (dy > 0).to(dtype=si.dtype)

        logits = self.sigma * (si - sj)
        
        # weights
        if self.weight_by_delta:
            pair_weight = abs_dy.clamp_min(1e-12).pow(self.delta_power)
        else:
            pair_weight = torch.ones_like(abs_dy, dtype=si.dtype)

        if wg is not None:
            wi, wj = wg[ii[valid]], wg[jj[valid]]
            pair_weight = pair_weight * 0.5 * (wi + wj)

        return logits, target.to(logits.dtype), pair_weight.to(logits.dtype)

    def _ranknet_loss(
        self,
        scores: torch.Tensor,
        y: torch.Tensor,
        group_ids: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        RankNet pairwise loss aggregated over groups.
        """
        unique_groups = torch.unique(group_ids)
        all_losses = []
        all_weights = []
        total_pairs = 0

        for g in unique_groups:
            idx = torch.where(group_ids == g)[0]
            out = self._build_pairs_for_group(idx, scores, y, sample_weight)
            if out is None:
                continue

            logits, target, pair_weight = out

            # RankNet = BCE sur logits = sigma*(s_i - s_j)
            loss_ij = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )

            all_losses.append(loss_ij)
            all_weights.append(pair_weight)
            total_pairs += int(loss_ij.numel())

        if len(all_losses) == 0:
            return scores.new_tensor(0.0), 0

        losses = torch.cat(all_losses, dim=0)
        weights = torch.cat(all_weights, dim=0)

        rank_loss = (losses * weights).sum() / weights.sum().clamp_min(1e-12)
        return rank_loss, total_pairs

    # =========================================================
    # Midpoint calibration for thresholds
    # =========================================================
    def _loss_mid_score_from_bins(
        self,
        s: torch.Tensor,
        theta_rows: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ):
        """
        Geometry-only threshold calibration in SCORE space.

        For each group g and class k:
            center_s[g, k] = mean score of samples currently assigned to hard bin k

        Then enforce:
            theta[g, k] ~ 0.5 * (center_s[g, k] + center_s[g, k+1])
        """
        device = s.device
        dtype = s.dtype
        s_det = s.detach()

        if theta_rows.dim() == 1:
            theta_rows = theta_rows.unsqueeze(0)  # (1, C-1)

        if group_ids is None:
            group_ids = torch.zeros_like(s_det, dtype=torch.long, device=device)
            G = 1
        else:
            group_ids = group_ids.to(device=device, dtype=torch.long).view(-1)
            G = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else theta_rows.shape[0]

        if theta_rows.shape[0] == 1 and G > 1:
            theta_rows = theta_rows.expand(G, -1)

        if theta_rows.shape[0] != G:
            raise ValueError(
                f"theta_rows and group_ids mismatch: "
                f"theta_rows.shape={theta_rows.shape}, G={G}"
            )

        thr_s = theta_rows.index_select(0, group_ids)  # (N, C-1)
        hard_bins = (s_det.unsqueeze(1) > thr_s.detach()).sum(dim=1)  # (N,)

        centers_s = torch.full((G, self.C), float("nan"), device=device, dtype=dtype)

        for k in range(self.C):
            mask = (hard_bins == k)
            if not mask.any():
                continue

            count_k = torch.zeros(G, device=device, dtype=dtype)
            sum_k = torch.zeros(G, device=device, dtype=dtype)

            ones_k = torch.ones(mask.sum(), device=device, dtype=dtype)
            count_k.scatter_add_(0, group_ids[mask], ones_k)
            sum_k.scatter_add_(0, group_ids[mask], s_det[mask])

            centers_s[:, k] = sum_k / count_k.clamp_min(1.0)

        target_mid = 0.5 * (centers_s[:, :-1] + centers_s[:, 1:])
        valid = torch.isfinite(target_mid) & torch.isfinite(theta_rows)

        if not valid.any():
            return s.new_tensor(0.0), centers_s, hard_bins

        Lmid = F.smooth_l1_loss(theta_rows[valid], target_mid[valid], reduction="mean")
        return Lmid, centers_s, hard_bins

    # =========================================================
    # Forward
    # =========================================================
    def forward(
        self,
        score: torch.Tensor,
        y_cont: torch.Tensor,
        clusters_ids: Optional[torch.Tensor],
        departement_ids: Optional[torch.Tensor],
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        score          : (N,) raw model score
        y_cont         : (N,) continuous/discrete relevance target for ranking
        clusters_ids   : (N,) raw cluster ids
        departement_ids: (N,) raw department ids
        """
        s = score.view(-1)
        y = y_cont.view(-1).to(device=s.device, dtype=s.dtype)

        self._validate_1d("score", s)
        self._validate_1d("y_cont", y)

        if clusters_ids is not None:
            clusters_ids = clusters_ids.view(-1).long().to(device=s.device)
            _, cluster_slot_ids, _ = self._remap_ids(
                clusters_ids, self.nclusters, kind="cluster"
            )
        else:
            cluster_slot_ids = None

        if departement_ids is not None:
            departement_ids = departement_ids.view(-1).long().to(device=s.device)
            _, dept_slot_ids, _ = self._remap_ids(
                departement_ids, self.ndepartements, kind="department"
            )
        else:
            dept_slot_ids = None

        if sample_weight is not None:
            sw = sample_weight.view(-1).to(device=s.device, dtype=s.dtype)
            self._validate_1d("sample_weight", sw)
        else:
            sw = None

        if not (s.numel() == y.numel()):
            raise ValueError("score and y_cont must have the same length")

        # -----------------------------------
        # 1) RankNet term
        # -----------------------------------
        pair_group_ids = self._group_ids_for_pairwise(
            s=s,
            cluster_slot_ids=cluster_slot_ids,
            dept_slot_ids=dept_slot_ids,
        )
        rank_loss, n_pairs = self._ranknet_loss(
            scores=s,
            y=y,
            group_ids=pair_group_ids,
            sample_weight=sw,
        )

        # -----------------------------------
        # 2) Optional threshold calibration
        # -----------------------------------
        theta_all = self._compute_thresholds().to(device=s.device, dtype=s.dtype)

        if self.wmid > 0.0:
            if self.alphatype == "global":
                Lmid, _, _ = self._loss_mid_score_from_bins(
                    s=s,
                    theta_rows=theta_all,
                    group_ids=None,
                )

            elif self.alphatype == "cluster":
                if cluster_slot_ids is None:
                    raise ValueError("clusters_ids required when alphatype='cluster'")
                active_cluster_slots, local_cluster_ids = torch.unique(
                    cluster_slot_ids, return_inverse=True
                )
                theta_mid = theta_all.index_select(0, active_cluster_slots)
                Lmid, _, _ = self._loss_mid_score_from_bins(
                    s=s,
                    theta_rows=theta_mid,
                    group_ids=local_cluster_ids,
                )

            elif self.alphatype == "department":
                if dept_slot_ids is None:
                    raise ValueError("departement_ids required when alphatype='department'")
                active_dept_slots, local_dept_ids = torch.unique(
                    dept_slot_ids, return_inverse=True
                )
                theta_mid = theta_all.index_select(0, active_dept_slots)
                Lmid, _, _ = self._loss_mid_score_from_bins(
                    s=s,
                    theta_rows=theta_mid,
                    group_ids=local_dept_ids,
                )
            else:
                raise ValueError(f"Unknown alphatype: {self.alphatype}")
        else:
            Lmid = s.new_tensor(0.0)

        total_loss = self.wrank * rank_loss + self.wmid * Lmid

        # logging
        self.epoch_stats.setdefault("rank", []).append(float(rank_loss.detach().cpu()))
        self.epoch_stats.setdefault("mid", []).append(float(Lmid.detach().cpu()))
        self.epoch_stats.setdefault("n_pairs", []).append(int(n_pairs))

        return total_loss

    # =========================================================
    # Inference helpers
    # =========================================================
    @torch.no_grad()
    def score_to_class(
        self,
        scores: torch.Tensor,
        clusters_ids: Optional[torch.Tensor] = None,
        departement_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert score -> class using the learned thresholds.
        """
        s = scores.detach().flatten()
        device = s.device
        dtype = self.alpha.dtype

        thr = self._compute_thresholds().detach().to(device=device, dtype=dtype)

        if thr.dim() == 1:
            return torch.bucketize(s, thr, right=True)

        if self.alphatype == "cluster":
            if clusters_ids is None:
                raise ValueError("clusters_ids required when alphatype='cluster'")
            chosen_ids = clusters_ids.view(-1).long().to(device=device)
            _, idx, _ = self._remap_ids(chosen_ids, self.nclusters, kind="cluster")

        elif self.alphatype == "department":
            if departement_ids is None:
                raise ValueError("departement_ids required when alphatype='department'")
            chosen_ids = departement_ids.view(-1).long().to(device=device)
            _, idx, _ = self._remap_ids(chosen_ids, self.ndepartements, kind="department")

        else:
            raise ValueError(f"Unknown alphatype: {self.alphatype}")

        thr_s = thr.index_select(0, idx)  # (N, C-1)
        return (s.unsqueeze(1) > thr_s).sum(dim=1)

    def get_learnable_parameters(self):
        return {"alpha": self.alpha}

    def get_attribute(self):
        payload: Dict[str, Any] = {
            "alpha": self.alpha.detach().cpu().numpy(),
            "thresholds": self._compute_thresholds().detach().cpu().numpy(),
            "cluster_slot_to_raw": self.cluster_slot_to_raw.detach().cpu().numpy(),
            "departement_slot_to_raw": self.departement_slot_to_raw.detach().cpu().numpy(),
        }

        if self.epoch_stats.get("rank"):
            payload["rank"] = [float(np.mean(self.epoch_stats["rank"]))]
        if self.epoch_stats.get("mid"):
            payload["mid"] = [float(np.mean(self.epoch_stats["mid"]))]
        if self.epoch_stats.get("n_pairs"):
            payload["n_pairs"] = [int(np.mean(self.epoch_stats["n_pairs"]))]

        return [("ranknet_params", DictWrapper(payload))]

    def update_params(self, new_dict, epoch=None):
        payload = new_dict

        # Cas: {"epoch": ..., "ordinal_params": DictWrapper(...)}
        if isinstance(payload, dict) and "ordinal_params" in payload:
            if epoch is None and "epoch" in payload:
                epoch = payload["epoch"]
            payload = payload["ordinal_params"]

        # Cas: DictWrapper(...)
        if hasattr(payload, "numpy") and not isinstance(payload, dict):
            payload = payload.numpy()

        if not isinstance(payload, dict):
            raise TypeError(
                f"update_params expected a dict-like payload after unwrapping, got {type(payload)}"
            )
            
        print('Old alpha', self.alpha)

        if "alpha" not in payload or payload["alpha"] is None:
            return

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
        
        print('New alpha', self.alpha)