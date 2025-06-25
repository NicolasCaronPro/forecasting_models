import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

from forecasting_models.pytorch.tools_2 import *
from torch.functional import F
from typing import Optional
from dlordinal.losses import *

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

            total_loss += group_loss
            total_samples += 1  # Or optionally accumulate weights instead

        return total_loss  # or total_loss / total_samples if you want average

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

    def dice_loss(self, y_true, y_pred, smooth=1e-6):
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
    def __init__(self, weight=None):
        super(DiceLoss2, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size(), device=predict.device)  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)

        dice_loss = 1 - dice_coef.sum() / predict.size(0)

        return dice_loss

class OrdinalDiceLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(OrdinalDiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):

        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size(), device=predict.device)  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, target.float() + 1)  # (N, C, *)
        
        classes = torch.arange(1, 6).to(predict)
        predict *= classes[None, :, None]

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)

        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        
        dice_loss = 1 - dice_coef.sum() / predict.size(0)

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
        use_logits=False,
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
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
        self.mce = MCELoss(self.num_classes, weight=weight, use_logits=self.use_logits)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
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

        wk_result = self.wk(y_true, y_pred)
        mce_result = self.mce(y_true, y_pred)

        return self.C * wk_result + (1 - self.C) * mce_result