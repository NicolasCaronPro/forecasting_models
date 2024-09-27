from tools import *
from torch.functional import F

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
            return torch.mean(weighted_error)
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
            return torch.mean(weighted_error)
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
            return torch.mean(weighted_error)
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
            return torch.mean(weighted_error)
        else:
            return torch.mean(tukey_loss)

class ExponentialLoss(torch.nn.Module):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        exp_loss = torch.exp(torch.abs(y_pred - y_true))
        if sample_weights is not None:
            weighted_error = exp_loss * sample_weights
            return torch.mean(weighted_error)
        else:
            return torch.mean(exp_loss)

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        # Calculer la cross-entropy standard (non pondérée)
        log_prob = F.log_softmax(y_pred, dim=-1)
        loss = F.nll_loss(log_prob, y_true, reduction='none')  # Pas de réduction pour pouvoir appliquer les sample weights
        
        # Appliquer les sample weights si fournis
        if sample_weights is not None:
            weighted_loss = loss * sample_weights
            return torch.mean(weighted_loss)
        else:
            return torch.mean(loss)