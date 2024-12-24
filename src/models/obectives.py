import numpy as np
from .loss import *

def exp_weighted_rmse_obj(y_pred: np.ndarray, dtrain):
    global threshold, max_weight
    try:
        y_true = dtrain.get_label()  # Obtenir les vraies étiquettes
    except:
        y_true = dtrain
    weights = exponential_weights(y_true, threshold, max_weight)
    errors = y_pred - y_true
    grad = weights * errors
    hess = weights
    return grad, hess
    
def weighted_rmse_obj(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Custom objective function for RMSE weighted by deviation from mean.

    Parameters:
    y_true (array): True values.
    y_pred (array): Predicted values.

    Returns:
    grad (array): Gradient of the loss.
    hess (array): Hessian (second derivative) of the loss.
    """

    mean = np.mean(y_true)
    std = np.std(y_true)

    # Calcul des poids en fonction de l'écart par rapport à la moyenne
    weights = 1 + (np.abs(y_true - mean) / std)
    # print(y_pred)
    # print(y_true)
    errors = y_pred - y_true
    grad = weights * errors
    hess = weights

    return grad, hess


def percentiles_weighted_rmse_obj(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Custom percentiles weighted RMSE objective function.

    Parameters:
    y_true (array): True values.
    y_pred (array): Predicted values.

    Returns:
    grad (array): Gradient of the loss.
    hess (array): Hessian (second derivative) of the loss.
    """
    percentile_95 = np.percentile(y_true, 95)
    percentile_90 = np.percentile(y_true, 90)
    percentile_85 = np.percentile(y_true, 85)
    percentile_80 = np.percentile(y_true, 80)
    percentile_75 = np.percentile(y_true, 75)
    percentile_70 = np.percentile(y_true, 70)
    percentile_65 = np.percentile(y_true, 65)
    percentile_60 = np.percentile(y_true, 60)
    percentile_55 = np.percentile(y_true, 55)
    percentile_50 = np.percentile(y_true, 50)

    weights = np.ones_like(y_true)
    weights[y_true > percentile_55] = 1.5
    weights[y_true > percentile_60] = 2
    weights[y_true > percentile_65] = 2.5
    weights[y_true > percentile_70] = 3
    weights[y_true > percentile_75] = 3.5
    weights[y_true > percentile_80] = 4
    weights[y_true > percentile_85] = 4.5
    weights[y_true > percentile_90] = 5
    weights[y_true > percentile_95] = 5.5
    # weights[y_true < percentile_5] = 5   # Poids plus élevé pour les creux

    errors = y_pred - y_true
    grad = weights * errors
    hess = weights

    return grad, hess


# Not used
def mape_objective(y_true, y_pred):
    grad = (y_true - y_pred) / y_true
    hess = 1 / y_true
    return grad, hess

# def mape_objective(y_true, y_pred):
#     epsilon = 1e-9
#     # Calcul du gradient
#     grad = (y_pred - y_true) / (np.maximum(y_true * np.abs(y_true - y_pred), epsilon))
#     # Calcul du hessien
#     hess = 1.0 / (np.maximum(y_true * (y_true + y_pred), epsilon))
#     # Inspecter les gradients et hessiens
#     print("Gradient:", grad[:10])  # Afficher les 10 premiers gradients
#     print("Hessien:", hess[:10])  # Afficher les 10 premiers hessiens
#     return grad, hess

# Not used


def rmsle_objective(y_true, y_pred):
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = (1 - np.log1p(y_pred) + np.log1p(y_true)) / (y_pred + 1)**2
    return grad, hess


def mean_quartic_error_obj(y_true, y_pred):
    # Calculate the difference
    diff = y_true - y_pred
    # Gradient: -4 * (y_true - y_pred)^3
    grad = -4 * diff**3
    # Hessian: 12 * (y_true - y_pred)^2
    hess = 12 * diff**2
    return grad, hess


regression_objective_metrics = {
    'reg:squarederror': ['rmse'],
    'reg:absoluteerror': ['mae'],
    'reg:squaredlogerror': ['rmsle'],
    'reg:logistic': ['r2_score'],
    'reg:tweedie': ['tweedie-nloglik@1.5'],
    'reg:gamma': ['gamma-deviance'],
    'count:poisson': ['poisson-nloglik'],
    # 'reg:pseudohubererror': ['mphe'],
    'reg:quantileerror': ['pinball'],
    weighted_rmse_obj.__name__: ['w_rmse'],
    percentiles_weighted_rmse_obj.__name__: ['pw_rmse'],
    mean_quartic_error_obj.__name__: ['mqe'],
    exp_weighted_rmse_obj.__name__: ['ew_rmse']
}

for obj in list(regression_objective_metrics.keys()):
    regression_objective_metrics[obj].extend([x for x in regression_metrics if x not in regression_objective_metrics[obj]])


classification_objective_metrics = {
    'binary:logistic': ['logloss']
}

for obj in list(classification_objective_metrics.keys()):
    classification_objective_metrics[obj].extend([x for x in classification_metrics if x not in classification_objective_metrics[obj]])
