import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

from forecasting_models.pytorch.classification_loss import *
from forecasting_models.pytorch.regression_loss import *
from forecasting_models.pytorch.ordinal_loss import *
from forecasting_models.pytorch.distribution_loss import *

import torch.nn.functional as F
from typing import Optional

################################ Classic #############################

class LossPerId(torch.nn.Module):
    def __init__(self, criterion, id_index, num_classes=5):
        super(LossPerId, self).__init__()
        self.num_classes = num_classes
        self.id_index = id_index  # index used to group by department

        self.criterion = criterion

    def forward(self, y_pred, y_true, cluster_ids, update_matrix=False, sample_weights=None):
        cluster_ids = cluster_ids.reshape(y_true.shape[0])
        y_true = y_true.long()
        group_ids = cluster_ids.unique()
        total_loss = 0.0
        total_samples = 0

        res = []

        for group in group_ids:
            group_mask = (cluster_ids == group)
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