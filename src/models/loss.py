from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, mean_squared_error, root_mean_squared_log_error, mean_squared_log_error
import pandas as pd

# mae, mse, rmse, msle, mape, huber, logcosh
import numpy as np

def weighted_rmse(y_true, y_pred, sample_weight=None):
    """
    Custom evaluation metric for RMSE weighted by deviation from mean.
    
    Parameters:
    y_true (array): True values.
    y_pred (array): Predicted values.
    
    Returns:
    weighted_rmse_value (float): Value of the weighted RMSE.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    # Ensure y_true and y_pred are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    
    mean = np.mean(y_true)
    std = np.std(y_true)
    
    # Calcul des poids en fonction de l'écart par rapport à la moyenne
    weights = 1 + (np.abs(y_true - mean) / std)

    # print(weights)


    errors = y_pred - y_true
    weighted_squared_errors = weights * errors ** 2
    weighted_rmse_value = np.sqrt(np.mean(weighted_squared_errors))
    
    return weighted_rmse_value

def percentiles_weighted_rmse(y_true, y_pred, sample_weight=None):
    """
    Custom evaluation metric for percentiles weighted RMSE.
    
    Parameters:
    y_true (array): True values.
    y_pred (array): Predicted values.
    
    Returns:
    weighted_rmse_value (float): Value of the weighted RMSE.
    """
    
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    # Ensure y_true and y_pred are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

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


# # Fonction de calcul du MAPE pour l'évaluation
# def mean_absolute_percentage_error(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ''' Root mean squared log error metric.

    :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
    '''
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def weighted_mse_loss(y_true, y_pred, sample_weight=None):
    squared_error = (y_pred - y_true) ** 2
    if sample_weight is not None:
        return np.sum(squared_error * sample_weight) / np.sum(sample_weight)
    else:
        return np.mean(squared_error)

def poisson_loss(y_true, y_pred, sample_weight=None):
    y_pred = np.clip(y_pred, 1e-8, None)  # Éviter log(0)
    loss = y_pred - y_true * np.log(y_pred)
    if sample_weight is not None:
        return np.sum(loss * sample_weight) / np.sum(sample_weight)
    else:
        return np.mean(loss)

def rmsle_loss(y_true, y_pred, sample_weight=None):
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)
    squared_log_error = (log_pred - log_true) ** 2
    if sample_weight is not None:
        return np.sqrt(np.sum(squared_log_error * sample_weight) / np.sum(sample_weight))
    else:
        return np.sqrt(np.mean(squared_log_error))

def rmse_loss(y_true, y_pred, sample_weight=None):
    squared_error = (y_pred - y_true) ** 2
    if sample_weight is not None:
        return np.sqrt(np.sum(squared_error * sample_weight) / np.sum(sample_weight))
    else:
        return np.sqrt(np.mean(squared_error))

def huber_loss(y_true, y_pred, delta=1.0, sample_weight=None):
    error = y_pred - y_true
    abs_error = np.abs(error)
    quadratic = np.where(abs_error <= delta, 0.5 * error ** 2, delta * (abs_error - 0.5 * delta))
    
    if sample_weight is not None:
        return np.average(quadratic, weights=sample_weight)
    else:
        return np.mean(quadratic)

def log_cosh_loss(y_true, y_pred, sample_weight=None):
    error = y_pred - y_true
    log_cosh = np.log(np.cosh(error))
    
    if sample_weight is not None:
        return np.average(log_cosh, weights=sample_weight)
    else:
        return np.mean(log_cosh)

def tukey_biweight_loss(y_true, y_pred, c=4.685, sample_weight=None):
    error = y_pred - y_true
    abs_error = np.abs(error)
    mask = (abs_error <= c)
    loss = (1 - (1 - (error / c) ** 2) ** 3) * mask
    tukey_loss = (c ** 2 / 6) * loss
    
    if sample_weight is not None:
        return np.average(tukey_loss, weights=sample_weight)
    else:
        return np.mean(tukey_loss)

def exponential_loss(y_true, y_pred, sample_weight=None):
    exp_loss = np.exp(np.abs(y_pred - y_true))

    if sample_weight is not None:
        return np.average(exp_loss, weights=sample_weight)
    else:
        return np.mean(exp_loss)
    

metrics = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error,
    'w_rmse': weighted_rmse,
    'pw_rmse': percentiles_weighted_rmse,
    'msle': mean_squared_log_error,
    'rmsle': root_mean_squared_log_error
}