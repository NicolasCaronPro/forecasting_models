import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

from forecasting_models.pytorch.classification_loss import *
from forecasting_models.pytorch.regression_loss import *
from forecasting_models.pytorch.ordinal_loss import *
from forecasting_models.pytorch.distribution_loss import *

import torch.nn.functional as F
from typing import Optional
