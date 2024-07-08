from tools import *
from torch.nn.modules.loss import _Loss
from torch.functional import F

class WeightedMSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='none'):
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, weights):
        return ((F.mse_loss(input, target, reduction='none')) * weights).sum() / weights.sum()

class WeightedPoissonLoss(_Loss):
    def __init__(self, log_input=True, full=False, size_average=None,
                 eps=1e-8, reduce=None, reduction='none'):
        super(WeightedPoissonLoss, self).__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input, target, weights):
        return (F.poisson_nll_loss(log_input, target, log_input=self.log_input,
                full=self.full, eps=self.eps, reduction='none')) * weights / weights.sum()

