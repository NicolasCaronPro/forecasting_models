from typing import Tuple
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, root_mean_squared_log_error, mean_squared_log_error, r2_score, mean_pinball_loss, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, max_error, explained_variance_score, log_loss
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

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

# def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     ''' Root mean squared log error metric.

#     :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
#     '''
#     return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# def weighted_mse_loss(y_true, y_pred, sample_weight=None):
#     squared_error = (y_pred - y_true) ** 2
#     if sample_weight is not None:
#         return np.sum(squared_error * sample_weight) / np.sum(sample_weight)
#     else:
#         return np.mean(squared_error)

# def poisson_loss(y_true, y_pred, sample_weight=None):
#     # Ensure y_true and y_pred are 1D arrays
#     y_true = np.ravel(y_true)
#     y_pred = np.ravel(y_pred)

#     y_pred = np.clip(y_pred, 1e-8, None)  # Éviter log(0)
#     loss = y_pred - y_true * np.log(y_pred)
#     if sample_weight is not None:
#         return np.sum(loss * sample_weight) / np.sum(sample_weight)
#     else:
#         return np.mean(loss)

# def rmsle_loss(y_true, y_pred, sample_weight=None):
#     log_pred = np.log1p(y_pred)
#     log_true = np.log1p(y_true)
#     squared_log_error = (log_pred - log_true) ** 2
#     if sample_weight is not None:
#         return np.sqrt(np.sum(squared_log_error * sample_weight) / np.sum(sample_weight))
#     else:
#         return np.sqrt(np.mean(squared_log_error))

# def rmse_loss(y_true, y_pred, sample_weight=None):
#     squared_error = (y_pred - y_true) ** 2
#     if sample_weight is not None:
#         return np.sqrt(np.sum(squared_error * sample_weight) / np.sum(sample_weight))
#     else:
#         return np.sqrt(np.mean(squared_error))

# def huber_loss(y_true, y_pred, delta=1.0, sample_weight=None):
#     # Ensure y_true and y_pred are 1D arrays
#     y_true = np.ravel(y_true)
#     y_pred = np.ravel(y_pred)

#     error = y_pred - y_true
#     abs_error = np.abs(error)
#     quadratic = np.where(abs_error <= delta, 0.5 * error ** 2, delta * (abs_error - 0.5 * delta))
    
#     if sample_weight is not None:
#         return np.average(quadratic, weights=sample_weight)
#     else:
#         return np.mean(quadratic)

# def log_cosh_loss(y_true, y_pred, sample_weight=None):
#     # Ensure y_true and y_pred are 1D arrays
#     y_true = np.ravel(y_true)
#     y_pred = np.ravel(y_pred)

#     error = y_pred - y_true
#     log_cosh = np.log(np.cosh(error))
    
#     if sample_weight is not None:
#         return np.average(log_cosh, weights=sample_weight)
#     else:
#         return np.mean(log_cosh)

# def tukey_biweight_loss(y_true, y_pred, c=4.685, sample_weight=None):
#     # Ensure y_true and y_pred are 1D arrays
#     y_true = np.ravel(y_true)
#     y_pred = np.ravel(y_pred)

#     error = y_pred - y_true
#     abs_error = np.abs(error)
#     mask = (abs_error <= c)
#     loss = (1 - (1 - (error / c) ** 2) ** 3) * mask
#     tukey_loss = (c ** 2 / 6) * loss
    
#     if sample_weight is not None:
#         return np.average(tukey_loss, weights=sample_weight)
#     else:
#         return np.mean(tukey_loss)

# def exponential_loss(y_true, y_pred, sample_weight=None):
#     # Ensure y_true and y_pred are 1D arrays
#     y_true = np.ravel(y_true)
#     y_pred = np.ravel(y_pred)
    
#     exp_loss = np.exp(np.abs(y_pred - y_true))

#     if sample_weight is not None:
#         return np.average(exp_loss, weights=sample_weight)
#     else:
#         return np.mean(exp_loss)
    
def mean_quartic_error_loss(y_true, y_pred, sample_weight=None):
    # Ensure y_true and y_pred are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    # Compute the quartic error
    error = y_pred - y_true
    quartic = (np.abs(error)) ** 4

    # Compute the average of the quartic error
    if sample_weight is not None:
        return np.average(quartic, weights=sample_weight)
    else:
        return np.mean(quartic)

def mean_seasonal_squared_error(y_true, y_pred, alpha=0.5, freq=7):
    """
    Calculate the Mean Seasonal Squared Error (MSSE) by decomposing both
    the true values and predicted values into trend and residual components.
    
    Parameters:
        y_true (array-like): Actual values of the time series.
        y_pred (array-like): Predicted values of the time series.
        alpha (float): Coefficient for weighting the trend and residual errors (0 <= alpha <= 1).
        freq (int): Seasonal period for decomposition.
    
    Returns:
        float: The calculated MSSE value.
    """
    # Decompose the true values into trend and residual components
    decomposition_true = seasonal_decompose(y_true, period=freq, model='additive', extrapolate_trend='freq')
    y_trend_true = decomposition_true.trend
    y_residuals_true = decomposition_true.resid

    # Decompose the predicted values into trend and residual components
    decomposition_pred = seasonal_decompose(y_pred, period=freq, model='additive', extrapolate_trend='freq')
    y_trend_pred = decomposition_pred.trend
    y_residuals_pred = decomposition_pred.resid

    # # Handle missing values (e.g., NaNs) in trend and residuals
    # mask = ~np.isnan(y_trend_true) & ~np.isnan(y_trend_pred)
    # y_trend_true = y_trend_true[mask]
    # y_trend_pred = y_trend_pred[mask]
    # y_residuals_true = y_residuals_true[mask]
    # y_residuals_pred = y_residuals_pred[mask]

    # Compute MSE for trend and residuals
    mse_trend = np.mean((y_trend_true - y_trend_pred) ** 2)
    mse_residuals = np.mean((y_residuals_true - y_residuals_pred) ** 2)

    # Compute the Mean Squared Seasonality Error (MSSE)
    msse = alpha * mse_trend + (1 - alpha) * mse_residuals
    return msse

metrics = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error,
    'w_rmse': weighted_rmse,
    'pw_rmse': percentiles_weighted_rmse,
    'msle': mean_squared_log_error,
    'rmsle': root_mean_squared_log_error,
    'r2': r2_score,
    'mqe': mean_quartic_error_loss,
    'msse': mean_seasonal_squared_error,
    'pinball': mean_pinball_loss,
    'gamma-deviance': mean_gamma_deviance,
    'tweedie-nloglik@1.7': mean_tweedie_deviance,
    'poisson-nloglik': mean_poisson_deviance,
    'max_error': max_error,
    'explained_variance': explained_variance_score,
    'logloss': log_loss
}

metric_params = {
    'mae': {},
    'mse': {},
    'rmse': {},
    'w_rmse': {},
    'pw_rmse': {},
    'msle': {},
    'rmsle': {},
    'r2': {},
    'mqe': {},
    'msse': {'alpha': 0.5, 'freq': 7},
    'pinball': {'alpha': 0.5},
    'gamma-deviance': {},
    'tweedie-nloglik@1.7': {'power': 1.7},
    'poisson-nloglik': {},
    'max_error': {},
    'explained_variance': {}
}