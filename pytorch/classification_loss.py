import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

import torch
import torch.nn.functional as F
from typing import Optional
from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.loss_utils import *

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

    def forward(
        self,
        y_pred,
        y_true,
        update_matrix: bool = False,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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
        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss.device)
            weighted_loss = loss * sample_weight
            return torch.sum(weighted_loss) / torch.sum(sample_weight)

        return loss.mean()

    def get_learnable_parameters(self):
        """Expose the learnable parameters to the external optimizer."""
        return {'adjustment_rate': self.adjustment_rate}

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(
        self,
        y_pred,
        y_true,
        update_matrix: bool = False,
        sample_weight: Optional[torch.Tensor] = None,
    ):
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
        
        if sample_weight is not None:
            sample_weight = sample_weight.view(-1).to(loss.device)
            weighted_loss = loss * sample_weight
            return torch.sum(weighted_loss) / torch.sum(sample_weight)

        return loss.mean()
        
def has_method(obj, method_name):
    """
    Vérifie si un objet ou une classe possède une méthode spécifique.

    Args:
        obj: L'objet ou la classe à tester.
        method_name (str): Le nom de la méthode à vérifier.

    Returns:
        bool: True si la méthode existe, False sinon.
    """
    return callable(getattr(obj, method_name, None))