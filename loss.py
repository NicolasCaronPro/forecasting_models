from tools import *
from torch.nn.modules.loss import _Loss
from torch.functional import F

class PoissonLoss(nn.Module):
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

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, y_pred, y_true, sample_weights=None):
        # On ajoute 1 aux prédictions et aux vraies valeurs pour éviter les log(0)
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

class RMSELoss(nn.Module):
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

# 1. MSE Loss (Mean Squared Error Loss)
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

# 2. Huber Loss
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        quadratic = torch.where(abs_error <= self.delta, 0.5 * error ** 2, self.delta * (abs_error - 0.5 * self.delta))
        return torch.mean(quadratic)

# 3. Log-Cosh Loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(error + 1e-12)))  # Adding epsilon to avoid log(0)

# 4. Tukey's Biweight Loss
class TukeyBiweightLoss(nn.Module):
    def __init__(self, c=4.685):
        super(TukeyBiweightLoss, self).__init__()
        self.c = c

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        mask = (abs_error <= self.c).float()
        loss = (1 - (1 - (error / self.c) ** 2) ** 3) * mask
        return torch.mean((self.c ** 2 / 6) * loss)

# 5. Exponential Loss
class ExponentialLoss(nn.Module):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.exp(torch.abs(y_pred - y_true)))