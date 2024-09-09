from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, mean_squared_error, root_mean_squared_log_error, mean_squared_log_error


# mae, mse, rmse, msle, mape, huber, logcosh
import numpy as np
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

def weighted_rmse(y_true, y_pred):
    """
    Custom evaluation metric for RMSE weighted by deviation from mean.
    
    Parameters:
    y_true (array): True values.
    y_pred (array): Predicted values.
    
    Returns:
    weighted_rmse_value (float): Value of the weighted RMSE.
    """

    
    mean = np.mean(y_true)
    std = np.std(y_true)
    
    # Calcul des poids en fonction de l'écart par rapport à la moyenne
    weights = 1 + (np.abs(y_true - mean) / std)

    # print(weights)

    errors = y_pred - y_true
    weighted_squared_errors = weights * errors ** 2
    weighted_rmse_value = np.sqrt(np.mean(weighted_squared_errors))
    
    return weighted_rmse_value

def percentiles_weighted_rmse(y_true, y_pred):
    """
    Custom evaluation metric for percentiles weighted RMSE.
    
    Parameters:
    y_true (array): True values.
    y_pred (array): Predicted values.
    
    Returns:
    weighted_rmse_value (float): Value of the weighted RMSE.
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

    # print(weights)

    errors = y_pred - y_true
    weighted_squared_errors = weights * errors ** 2
    weighted_rmse_value = np.sqrt(np.mean(weighted_squared_errors))
    
    return weighted_rmse_value

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

def mape_objective(y_true, y_pred):
    grad = (y_true - y_pred) / y_true
    hess = 1 / y_true
    return grad, hess

# # Fonction de calcul du MAPE pour l'évaluation
# def mean_absolute_percentage_error(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmsle_objective(y_true, y_pred):
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = (1 - np.log1p(y_pred) + np.log1p(y_true)) / (y_pred + 1)**2
    return grad, hess

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ''' Root mean squared log error metric.

    :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
    '''
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


metrics = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error,
    'w_rmse': weighted_rmse,
    'pw_rmse': percentiles_weighted_rmse,
    'msle': mean_squared_log_error,
    'rmsle': root_mean_squared_log_error
}