from random import sample
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from xgboost import DMatrix
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
import seaborn as sns
#from forecasting_models.sklearn.dico_departements import *

def calculate_ks(data, score_col, event_col, thresholds, dir_output):
    """
    Calcule le KS-Statistic pour plusieurs seuils définis et renvoie les seuils optimaux pour chaque KS.
    
    :param data: pd.DataFrame contenant les scores et les événements.
    :param score_col: Nom de la colonne des scores prédits.
    :param event_col: Nom de la colonne des événements observés.
    :param thresholds: Liste des seuils pour diviser les groupes en faible/haut risque.
    :param dir_output: Dossier de sortie pour enregistrer les graphiques.
    :return: dict contenant les seuils optimums et leurs KS pour chaque groupe.
    """
    dir_output = Path(dir_output)  # Assurez-vous que dir_output est un Path
    dir_output.mkdir(parents=True, exist_ok=True)  # Crée le dossier si nécessaire

    ks_results = []
    optimal_thresholds = {}

    for threshold in thresholds:
        # Définir les groupes basés sur les seuils modifiés
        data['group'] = np.where(
            data[event_col] == threshold - 1, 'low_risk',  # Low risk pour les valeurs <= threshold - 1
            np.where(data[event_col] >= threshold, 'high_risk', 'other')  # High risk pour les valeurs == threshold
        )
        
        # Filtrer uniquement les données pertinentes (low_risk et high_risk)
        data_filtered = data[data['group'].isin(['low_risk', 'high_risk'])].copy()

        # Trier par score prédictif
        data_sorted = data_filtered.sort_values(by=score_col).reset_index(drop=True)
        
        # Calcul des distributions cumulées pour chaque groupe
        low_risk_cdf = np.cumsum(data_sorted['group'] == 'low_risk') / (data_filtered['group'] == 'low_risk').sum()
        high_risk_cdf = np.cumsum(data_sorted['group'] == 'high_risk') / (data_filtered['group'] == 'high_risk').sum()
        
        # Calcul du KS : distance maximale entre les deux CDF
        ks_stat = np.max(np.abs(low_risk_cdf - high_risk_cdf))
        try:
            optimal_score = data_sorted[score_col][np.argmax(np.abs(low_risk_cdf - high_risk_cdf))]
        except:
            optimal_score =-1
            ks_stat = 0.0

        # Stocker les résultats pour le seuil courant
        ks_results.append({'threshold': threshold, 'ks_stat': ks_stat, 'optimal_score': optimal_score})
        optimal_thresholds[threshold] = {'ks_stat': ks_stat, 'optimal_score': optimal_score}
        
        # Visualiser les CDF
        plt.figure()
        plt.plot(data_sorted[score_col], low_risk_cdf, label='Low Risk CDF', color='blue')
        plt.plot(data_sorted[score_col], high_risk_cdf, label='High Risk CDF', color='red')
        plt.axvline(optimal_score, color='green', linestyle='--', label=f'Optimal Score: {optimal_score:.3f}')
        plt.title(f"KS Plot for Threshold {threshold} (KS={ks_stat:.3f})")
        plt.xlabel('Score')
        plt.ylabel('Cumulative Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_output / f'Cumulative_Distribution_{score_col}_{event_col}_{threshold}.png')
        plt.close('all')

    # Convertir les résultats en DataFrame pour inspection (facultatif)
    ks_results_df = pd.DataFrame(ks_results)

    return ks_results_df, optimal_thresholds

def iou_score(y_true, y_pred):
    """
    Calcule les scores (aire commune, union, sous-prédiction, sur-prédiction) entre deux signaux.

    Args:
        t (np.array): Tableau de temps ou indices (axe x).
        y_pred (np.array): Signal prédiction (rouge).
        y_true (np.array): Signal vérité terrain (bleu).

    Returns:
        dict: Dictionnaire contenant les scores calculés.
    """

    if isinstance(y_pred, DMatrix):
        y_pred = np.copy(y_pred.get_data().toarray())

    if isinstance(y_true, DMatrix):
        y_true = np.copy(y_true.get_label())

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    return intersection / union if union > 0 else 0

def under_prediction_score(y_true, y_pred):
    """
    Calcule le score de sous-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est inférieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sous-prédiction.
    """

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sous-prédiction
    under_prediction_area = np.trapz(np.maximum(y_true - y_pred, 0))  # Valeurs positives où y_true > y_pred
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return under_prediction_area / union_area if union_area > 0 else 0

def over_prediction_score(y_true, y_pred):
    """
    Calcule le score de sur-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est supérieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sur-prédiction.
    """
    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sur-prédiction
    over_prediction_area = np.trapz(np.maximum(y_pred - y_true, 0))  # Valeurs positives où y_pred > y_true
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return over_prediction_area / union_area if union_area > 0 else 0

def calculate_signal_scores_for_training(y_pred, y_true, y_fire):
    """
    Calcule les scores (aire commune, union, sous-prédiction, sur-prédiction) entre deux signaux.

    Args:
        t (np.array): Tableau de temps ou indices (axe x).
        y_pred (np.array): Signal prédiction (rouge).
        y_true (np.array): Signal vérité terrain (bleu).

    Returns:
        dict: Dictionnaire contenant les scores calculés.
    """

    if y_fire is not None:
        y_true_fire = np.copy(y_true)
        y_true_fire[y_fire == 0] = 0

    ###################################### I. For the all signal ####################################
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    over_prediction_zeros = np.trapz(np.maximum(0, y_pred[y_true == 0]))
    under_prediction_zeros = np.trapz(np.maximum(0, y_true[y_pred == 0]))

    if 'y_true_fire' in locals():
        mask_fire = (y_pred > 0) | (y_true_fire > 0)
        intersection_fire = np.trapz(np.minimum(y_pred[mask_fire], y_true_fire[mask_fire]))
        union_fire = np.trapz(np.maximum(y_pred[mask_fire], y_true_fire[mask_fire]))
        iou_wildfire_or_pred = round(intersection_fire / union_fire, 2) if union_fire > 0 else 0

        mask_fire = (y_pred > 0) & (y_true_fire > 0)
        intersection_fire = np.trapz(np.minimum(y_pred[mask_fire], y_true_fire[mask_fire]))
        union_fire = np.trapz(np.maximum(y_pred[mask_fire], y_true_fire[mask_fire]))
        iou_wildfire_and_pred = round(intersection_fire / union_fire, 2) if union_fire > 0 else 0

        # Limitation des signaux à un maximum de 1
        y_pred_clipped = np.clip(y_pred, 0, 1)  # Limiter y_pred à 1
        y_true_fire_clipped = np.clip(y_true_fire, 0, 1)  # Limiter y_true_fire à 1

        # Masque pour les valeurs pertinentes
        mask_fire_detected = (y_true_fire_clipped > 0)

        # Calcul de l'intersection et de l'union
        intersection_fire_detected = np.trapz(np.minimum(y_pred_clipped[mask_fire_detected], y_true_fire_clipped[mask_fire_detected]))
        union_fire_detected = np.trapz(np.maximum(y_pred_clipped[mask_fire_detected], y_true_fire_clipped[mask_fire_detected]))

        # Calcul de la métrique IOU
        iou_wildfire_detected = round(intersection_fire_detected / union_fire_detected, 2) if union_fire_detected > 0 else 0

    else:
        iou_wildfire_or_pred = np.nan
        iou_wildfire_and_pred = np.nan
        iou_wildfire_detected = np.nan

    y_pred_clipped_ytrue = np.copy(y_pred)
    y_pred_clipped_ytrue[(y_pred > 0) & (y_true > 0)] = np.minimum(y_true[(y_pred > 0) & (y_true > 0)], y_pred[(y_pred > 0) & (y_true > 0)])
    intersection_clipped = np.trapz(np.minimum(y_pred_clipped_ytrue, y_true))  # Aire commune
    union_clipped = np.trapz(np.maximum(y_pred_clipped_ytrue, y_true))         # Aire d'union
    iou_no_overestimation = round(intersection_clipped / union_clipped, 2) if union_clipped > 0 else 0

    # Calcul du Dice coefficient
    dice_coefficient = round(2 * intersection / (union + intersection), 2) if union + intersection > 0 else 0

    # Enregistrement dans un dictionnaire
    scores = {
        "iou": [round(intersection / union, 2) if union > 0 else 0],  # To avoid division by zero
        "iou_wildfire_or_pred": [iou_wildfire_or_pred],
        "iou_wildfire_and_pred": [iou_wildfire_and_pred],
        "iou_wildfire_detected": [iou_wildfire_detected],
        "iou_no_overestimation" : [iou_no_overestimation],
        
        "over_bad_prediction" : [round(over_prediction_zeros / union, 2) if union > 0 else 0],
        "under_bad_prediction" : [round(under_prediction_zeros / union, 2) if union > 0 else 0],
        "bad_prediction" : [round((over_prediction_zeros + under_prediction_zeros) / union, 2) if union > 0 else 0],
        
        # Ajout du Dice coefficient
        "dice_coefficient": [dice_coefficient]
    }

    return pd.DataFrame.from_dict(scores)

# 1. Weighted MSE Loss
def weighted_mse_loss(y_pred, y, sample_weight=None):
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y
    error = (y_pred - y_true)**2
    if sample_weight is not None:
        error *= sample_weight
    grad = 2 * error  # Gradient de la MSE
    hess = 2 * np.ones_like(y_true)  # Hessienne de la MSE
    return grad, hess

# 2. Poisson Loss
def poisson_loss(y_pred, y, sample_weight=None):
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y
    y_pred = np.clip(y_pred, 1e-8, None)  # Éviter log(0)
    grad = 1 - y_true / y_pred  # Gradient pour Poisson
    hess = y_true / y_pred**2  # Hessienne pour Poisson
    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight
    return grad, hess

# 3. RMSLE Loss
def rmsle_loss(y_pred, y, sample_weight=None):
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y

    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = ((-np.log1p(y_pred) + np.log1p(y_true) + 1) /
            np.power(y_pred + 1, 2))

    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight

    return grad, hess

# 4. RMSE Loss
def rmse_loss(y_pred, y, sample_weight=None):
    """
    Calcule le gradient et la hessienne de la perte RMSE.

    Args:
    - y_pred (np.array): Prédictions.
    - y (np.array ou DMatrix): Valeurs réelles ou DMatrix.
    - sample_weight (np.array, optionnel): Poids des échantillons.

    Returns:
    - grad (np.array): Gradient de la RMSE.
    - hess (np.array): Hessienne de la RMSE.
    """
    # Vérification du type de y pour récupérer les labels
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y

    # Calcul de l'erreur
    error = y_pred - y_true
    n = len(y_true)

    # Calcul de la RMSE
    rmse_value = np.sqrt(np.mean(error ** 2))

    # Éviter la division par zéro
    if rmse_value == 0:
        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(y_pred)
    else:
        # Calcul du gradient et de la hessienne
        grad = error / (n * rmse_value)
        hess = np.ones_like(y_true) / (n * rmse_value)

    # Appliquer les poids si fournis
    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight

    return grad, hess

# 5. Huber Loss
def huber_loss(y_pred, y, delta=1.0, sample_weight=None):
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y
    error = y_pred - y_true
    abs_error = np.abs(error)
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))
    hess = np.where(abs_error <= delta, np.ones_like(error), np.zeros_like(error))
    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight
    return grad, hess

# 6. Log-Cosh Loss
def log_cosh_loss(y_pred, y, sample_weight=None):
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y
    error = y_pred - y_true
    grad = np.tanh(error)  # Gradient de Log-Cosh
    hess = 1 - grad**2  # Hessienne de Log-Cosh
    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight
    return grad, hess

# 7. Tukey Biweight Loss
def tukey_biweight_loss(y_pred, y, c=4.685, sample_weight=None):
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y
    error = y_pred - y_true
    abs_error = np.abs(error)
    mask = abs_error <= c
    grad = np.where(mask, error * (1 - (1 - (error / c) ** 2) ** 2), 0)
    hess = np.where(mask, (1 - (error / c) ** 2) * (1 - 3 * (error / c) ** 2), 0)
    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight
    return grad, hess

# 8. Exponential Absolute Error Loss
def exponential_absolute_error_loss(alpha=1.0):
    def grad_hess(y_pred, y, sample_weight=None):
        if isinstance(y, DMatrix):
            y_true = y.get_label()
        else:
            y_true = y
        error = y_pred - y_true
        abs_error = np.abs(error)
        grad = alpha * np.sign(error) * np.exp(alpha * abs_error)
        hess = alpha**2 * np.exp(alpha * abs_error)
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return grad_hess

# 9. Logistic Loss
def logistic_loss(y_pred, y, sample_weight=None):
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y
    sigmoide = 1 / (1 + np.exp(-y_pred))
    grad = sigmoide - y_true  # Gradient de la Logistic Loss
    hess = sigmoide * (1 - sigmoide)  # Hessienne de la Logistic Loss
    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight
    return grad, hess

def exponential_absolute_error(y_pred, y_true, alpha=1.0, sample_weight=None):
    errors = np.abs(y_true - y_pred)
    exp_errors = np.mean(np.exp(alpha * errors))  # Calcul de l'erreur exponentielle

    return exp_errors

def smooth_area_under_prediction_loss(y_pred, y, beta=1.0, sample_weight=None, loss=False):
    """
    Compute a smooth version of the AreaUnderPredictionLoss using the softplus function.

    Args:
    - y_true: Ground truth values (array).
    - y_pred: Predicted values (array).
    - beta: Smoothing factor. Higher beta means closer to the original loss.
    - sample_weight: Optional array of sample weights.

    Returns:
    - grad: Gradient of the smooth loss with respect to y_pred.
    - hess: Hessian of the smooth loss with respect to y_pred.
    """
    import scipy.special

    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = y

    # Calculate under-prediction and over-prediction errors
    under_prediction = y_true - y_pred
    over_prediction = y_pred - y_true

    # Smooth approximation using softplus
    #smooth_under = scipy.special.log1p(np.exp(-beta * under_prediction)) / beta
    #smooth_over = scipy.special.log1p(np.exp(-beta * over_prediction)) / beta
    smooth_under = under_prediction
    smooth_over = over_prediction

    # Sum of the union areas (approximation of total area)
    union_area = np.trapz(np.maximum(y_pred, y_true))

    #smooth_over = smooth_over / union_area
    #smooth_under = smooth_under / union_area

    if loss:
        return smooth_under + smooth_over

    # Gradient (smooth derivative)
    grad_under = 1 / (1 + np.exp(beta * smooth_under))
    grad_over = 1 / (1 + np.exp(beta * smooth_over))

    grad = grad_under + grad_over

    # Hessian (second derivative)
    hess_under = beta * grad_under * (1 - grad_under)
    hess_over = beta * grad_over * (1 - grad_over)

    hess = hess_under + hess_over

    # Apply sample weights if provided
    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight

    return grad, hess

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sig_loss(y_pred, y_true):
    diff = y_pred - y_true
    diff2 = diff**2
    mask = sigmoid(diff)
    losses = diff2 * (1 - mask) + diff2 * mask * 2
    return losses

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def sigmoid_hess(x):
        """Hessian of sigmoid."""
        sig = sigmoid(x)
        return sig * (1 - sig) * (1 - 2 * sig)

def sigmoid_adjusted_loss(y_pred, y, sample_weight=None):
    
    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = np.copy(y)

    diff = y_pred - y_true
    diff2 = diff ** 2
    
    # Gradient computation
    grad = 2 * diff * (1 - sigmoid(diff)) + diff2 * -sigmoid_derivative(diff) + 4 * diff * sigmoid(diff) + 2 * diff2 * sigmoid_derivative(diff) 
    
    # Hessian computation
    sig = sigmoid(diff)
    sig_derivative = sigmoid_derivative(diff)
    
    hess = 2 * (1 - sig) + 2 * diff * -sig_derivative + 2 * diff * -sigmoid_derivative(diff) + diff2 * -sigmoid_hess(diff) + 4 * sig + 4 * diff * sig_derivative + 4 * diff * sig_derivative + 2 * diff2 * sigmoid_hess(diff)

    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight
    
    return grad, hess

def sig_loss_adapated(y_pred, y):

    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = np.copy(y)

    y_true_zero = y_true == 0
    y_true_event = y_true > 0
    y_pred_zeros = y_pred == 0
    y_pred_event = y_pred > 0

    over_estimation = y_pred > y_true
    under_estimation = y_pred < y_true
    good_estimation = y_pred == y_true

    losses = np.zeros(y_true.shape)

    def sig_loss(x, y, bias, factor):
        diff = x - y
        diff2 = (diff + bias)**factor
        mask = sigmoid(diff)
        losses = diff2 * (1 - mask) + diff2 * mask * 2
        return losses
    
    y_true_zero = y_true == 0
    y_true_event = y_true > 0
    y_pred_zeros = y_pred == 0
    y_pred_event = y_pred > 0

    over_estimation = y_pred > y_true
    under_estimation = (y_pred < y_true)
    good_estimation = y_pred == y_true

    mask = (y_true_zero) & (over_estimation)
    losses[mask] = sig_loss(y_pred[mask], y_true[mask], 1, 2)

    mask = (y_true_event) & (y_pred_zeros)
    losses[mask] = sig_loss(y_true[mask], y_pred[mask], 1, 2)

    mask = (y_true_event) & (over_estimation)
    losses[mask] = sig_loss(y_pred[mask], y_true[mask], 0, 2)
    
    mask = (y_true_event) & (under_estimation) & (y_pred_event)
    losses[mask] = sig_loss(y_true[mask], y_pred[mask], 0, 2)

    mask = good_estimation
    losses[mask] = 0

    return losses

def sigmoid_adjusted_loss_adapted(y_pred, y, sample_weight=None):
    """Compute gradient and Hessian of the loss function with respect to y_pred."""

    if isinstance(y, DMatrix):
        y_true = y.get_label()
    else:
        y_true = np.copy(y)

    def sigmoid_grad(x):
        """Gradient of sigmoid."""
        sig = sigmoid(x)
        return sig * (1 - sig)

    def sigmoid_hess(x):
        """Hessian of sigmoid."""
        sig = sigmoid(x)
        return sig * (1 - sig) * (1 - 2 * sig)

    def sig_loss_grad(x, y, bias, factor):

        """Gradient of the loss function."""
        diff = x - y

        diff_bias = diff + bias

        sig = sigmoid(diff_bias)
        
        diff_factor = diff_bias**factor

        diff_factor_1 = factor * (diff_bias ** (factor - 1))

        return diff_factor_1 * (1 - sig) + diff_factor * -sigmoid_grad(diff_bias) + 2 * diff_factor_1 * sig + 2 * sigmoid_grad(diff_bias) * diff_factor

    def sig_loss_hess(x, y, bias, factor):
        """Hessian of the loss function."""

        diff = x - y

        diff_bias = diff + bias

        sig = sigmoid(diff_bias)

        diff_factor = diff_bias**factor
        diff_factor_1 = factor * (diff_bias ** (factor - 1))
        diff_factor_2 = factor * (factor - 1) * (diff_bias ** (factor - 2))

        hessian =  diff_factor_2 * (1 - sig) + diff_factor_1 * -sigmoid_grad(diff_bias) + diff_factor_1 * -sigmoid_grad(diff_bias) + diff_factor * -sigmoid_hess(diff_bias) + 2 * diff_factor_2 * sig + 2 * sigmoid_grad(diff_bias) * diff_factor_1 + 2 * sigmoid_hess(diff_bias) * diff_factor + 2 * diff_factor_1 * sigmoid_grad(diff_bias)

        return hessian
    
    y_true_zero = y_true == 0
    y_true_event = y_true > 0
    y_pred_zeros = y_pred < 0.9
    y_pred_event = y_pred >= 0.9

    over_estimation = y_pred > y_true
    under_estimation = (y_pred < y_true)
    good_estimation = y_pred == y_true

    grad = np.zeros(y_true.shape)
    hess = np.zeros(y_true.shape)
    
    mask = (y_true_zero) & (over_estimation)
    grad[mask] = sig_loss_grad(y_pred[mask], y_true[mask], 0, 4)
    hess[mask] = sig_loss_hess(y_pred[mask], y_true[mask], 0, 4)

    mask = (y_true_event) & (y_pred_zeros)
    grad[mask] = sig_loss_grad(y_pred[mask], y_true[mask], 0, 4)
    hess[mask] = sig_loss_hess(y_pred[mask], y_true[mask], 0, 4)

    mask = (y_true_event) & (over_estimation)
    grad[mask] = sig_loss_grad(y_pred[mask], y_true[mask], 0, 2)
    hess[mask] = sig_loss_hess(y_pred[mask], y_true[mask], 0, 2)
    
    mask = (y_true_event) & (under_estimation) & (y_pred_event)
    grad[mask] = sig_loss_grad(y_pred[mask], y_true[mask], 0, 2)
    hess[mask] = sig_loss_hess(y_pred[mask], y_true[mask], 0, 2)

    mask = good_estimation
    grad[mask] = 0
    hess[mask] = 0
    
    #grad = sig_loss_grad(y_pred, y_true, 0, 2)
    #hess = sig_loss_hess(y_pred, y_true, 0, 2)

    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight

    return grad, hess

def my_r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return round(r2_score(y_true, y_pred), 3)

def dice_coefficient(y_true, y_pred, smooth=1.0, weight=None):
    """
    Calcul de la Dice Loss en NumPy.

    Args:
        y_pred (np.ndarray): Prédictions de forme (N, C, ...).
        y_true (np.ndarray): Cibles vraies de forme (N, ...).
        smooth (float): Paramètre pour éviter la division par zéro.
        weight (np.ndarray, optional): Poids optionnels pour chaque classe.

    Returns:
        float: La valeur de la Dice Loss.
    """
    N, C = y_pred.shape

    # Mise en forme des prédictions et des cibles
    prob = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
    y_pred = np.argmax(prob, axis=1)

    y_pred = y_pred.reshape(N, 1)  # (N, 1, *)
    y_true = y_true.reshape(N, 1)  # (N, 1, *)
    
    # Calcul de l'intersection et de l'union
    intersection = np.sum(y_pred * y_true, axis=1)  # (N, C)
    union = np.sum(y_pred**2, axis=1) + np.sum(y_true**2, axis=1)  # (N, C)
    
    union = union.astype(np.float32)
    intersection = intersection.astype(np.float32)

    # Calcul du Dice Coefficient
    dice_coef = ((2 * intersection) + smooth) / (union + smooth)  # (N, C)

    return np.array(dice_coef)

def dice_loss(y, y_pred):
    """
    Calcul de la Dice Loss pour chaque classe.
    """

    num_classes = 5
    dice = dice_coefficient(y, y_pred, num_classes)
    return 1 - np.mean(dice)  # La Dice Loss est 1 - Dice coefficient

def weighted_class_loss_objective(y, pred):
    """
    Fonction objective personnalisée pour XGBoost basée sur la perte pondérée des classes.
    
    :param y: Vérités terrain (labels réels)
    :param pred: Prédictions du modèle (logits ou probabilités avant softmax)
    :return: gradient et hessian
    """
    num_classes = 5  # Nombre de classes
    epsilon = 1e-6  # Pour éviter les erreurs numériques
    
    # Transformation softmax pour obtenir les probabilités
    prob = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    pred_class = np.argmax(prob, axis=1).reshape(-1)
    
    # Initialisation des gradients et hessiens
    gradient = np.zeros_like(pred, dtype=np.float32)
    hessian = np.zeros_like(pred, dtype=np.float32)
    
    # Calcul de la matrice de pondération par rapport entre les classes
    dist_matrix = np.abs(np.arange(num_classes).reshape(1, -1) - np.arange(num_classes).reshape(-1, 1))
    ratio_matrix = 1 / (dist_matrix + 1)  # Inversement proportionnel à la distance entre les classes

    ratio_matrix =  [[0.1,  0.25,    0.5,    0.75,   1.],
                    [0.25, 0.25,    0.25,    0.5,   0.75],
                    [0.5,  0.25,    0.5 ,   0.25,   0.5],
                    [0.75, 0.5,     0.25,   0.75,   0.25],
                    [1.,   0.75,    0.5,    0.25,    1]]

    # Calcul du gradient et du hessien pour chaque échantillon
    for i in range(y.shape[0]):
        true_class = y[i].astype(int)
        
        ratio_mat_class = ratio_matrix[true_class]

        g = prob[i]
        g[true_class] -= 1.0
        gradient[i] = g * ratio_mat_class
        hess = 2.0 * prob[i] * (1.0 - prob[i]) * (ratio_mat_class)
        hess[hess < epsilon] = epsilon
        hessian[i] = hess

    return gradient.flatten(), hessian.flatten()

def softmax(x):
    '''Softmax function with x as input vector.'''
    e = np.exp(x)
    return e / np.sum(e)

"""def softprob_obj(y, pred):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.
    '''
    grad = np.zeros((y.shape[0], 5), dtype=float)
    hess = np.zeros((y.shape[0], 5), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.

    p = np.exp(pred, axis=1)
    p = p / np.sum(p)
    grad = np.copy(p)
    grad[:, y] = p[:, y] - 1.0
    hess = np.maximum(2 * p * (1.0 - p), eps)

    #for r in range(pred.shape[0]):
    #    target = y[r]
    #    p = softmax(pred[r, :])
    #    for c in range(pred.shape[1]):
    #        g = p[c] - 1.0 if c == target else p[c]
    #        g = g
    #        h = max((2.0 * p[c] * (1.0 - p[c])), eps)
    #        grad[r, c] = g
    #        hess[r, c] = h

    # After 2.1.0, pas
    return grad.flatten(), hess.flatten()"""

def softprob_obj(y, pred):
    '''Loss function: Computes gradient and hessian for multi-class softmax in XGBoost.'''
    
    eps = 1e-6  # Small value to prevent numerical instability

    # Compute softmax probabilities
    exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))  # Stability trick
    p = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)  # Softmax normalization

    # Initialize gradients and Hessians
    grad = np.copy(p)
    grad[np.arange(y.shape[0]), y.astype(int)] -= 1.0  # Applying gradient correction for true class labels

    hess = 2 * p * (1.0 - p)  # Hessian computation
    hess = np.maximum(hess, eps)  # Prevent zero values

    return grad.flatten(), hess.flatten()

def softprob_obj_matrix(y, pred, matrix):
    '''Loss function: Computes gradient and hessian for multi-class softmax in XGBoost.'''
    
    eps = 1e-6  # Small value to prevent numerical instability

    # Compute softmax probabilities
    exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))  # Stability trick
    p = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)  # Softmax normalization

    # Initialize gradients and Hessians
    grad = np.copy(p)
    grad[np.arange(y.shape[0]), y.astype(int)] -= 1.0  # Applying gradient correction for true class labels

    hess = 2 * p * (1.0 - p)  # Hessian computation
    hess = np.maximum(hess, eps)  # Prevent zero values

    pred_class = np.argmax(p, axis=1)

    error_weight_matrix = matrix[y.astype(int), pred_class]

    grad *= error_weight_matrix[:, None]
    hess *= error_weight_matrix[:, None]

    return grad.flatten(), hess.flatten()

######################################### SOFTPROB DUAL ############################################

class softprob_obj_dual(object):
    def __init__(self, y):
        self.y_temp = y
        self.index = 0

    def __call__(self, y_true, y_pred):
        g1, h1 = softprob_obj(y_true, y_pred)
        g2, h2 = softprob_obj(self.y_temp, y_pred)
        return g2 + g1, h2 + h1

    def softmax(self, x):
        """
        Compute softmax probabilities.
        :param x: Input array.
        :return: Softmax-transformed probabilities.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def calc_ders_multi(self, approx, target, weight):
        approx = [approx[i] - max(approx) for i in range(len(approx))]
        exp_approx = [np.exp(approx[i]) for i in range(len(approx))]
        exp_sum = np.sum(exp_approx)
        
        grad = []
        hess = []
        for j in range(len(approx)):
            der1 = -exp_approx[j] / exp_sum
            der2 = -exp_approx[j] / exp_sum
            if j == target:
                der1 += 1
            hess_row = []
            for j2 in range(len(approx)):
                der2 = exp_approx[j] * exp_approx[j2] / (exp_sum**2)
                if j2 == j:
                    der2 -= exp_approx[j] / exp_sum
                hess_row.append(der2 * weight)
                
            grad.append(der1 * weight)
            hess.append(hess_row)
        
        return (grad, hess)

######################################### SOFTPROB RISK ############################################

class softprob_obj_risk(object):
    def __init__(self, kappa_coef=None):
        self.matrix = np.asarray([[1, 2,    3,    4,   5],
                                [2, 2,    1,    1,   1],
                                [3, 1,    3,    1,   1],
                                [4, 1,    1,    4,   1],
                                [5, 1,    1,    1,   5]])
        
        self.matrix = self.matrix.astype(float)
        if kappa_coef is not None:
            for i in range(5):
                for j in range(5):
                    self.matrix[i, j] = abs((i - j) ** kappa_coef) / (4**kappa_coef)

        print(self.matrix)

    def __call__(self, y_true, y_pred):
        g1, h1 = softprob_obj_matrix(y_true, y_pred, self.matrix)
        return g1, h1

    def softmax(self, x):
        """
        Compute softmax probabilities.
        :param x: Input array.
        :return: Softmax-transformed probabilities.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def calc_ders_multi(self, approx, target, weight):
        approx = [approx[i] - max(approx) for i in range(len(approx))]
        exp_approx = [np.exp(approx[i]) for i in range(len(approx))]
        exp_sum = np.sum(exp_approx)
        
        grad = []
        hess = []
        for j in range(len(approx)):
            der1 = -exp_approx[j] / exp_sum
            der2 = -exp_approx[j] / exp_sum
            if j == target:
                der1 += 1
            hess_row = []
            for j2 in range(len(approx)):
                der2 = exp_approx[j] * exp_approx[j2] / (exp_sum**2)
                if j2 == j:
                    der2 -= exp_approx[j] / exp_sum
                hess_row.append(der2 * weight)
                
            grad.append(der1 * weight)
            hess.append(hess_row)
        
        return (grad, hess)

######################################### SOFTPROB RISK DUAL ############################################

class softprob_obj_risk_dual(object):
    def __init__(self, y, kappa_coef=None):
        self.y_ref = y
        self.matrix = np.asarray([[1, 2,    3,    4,   5],
                                [2, 2,    1,    1,   1],
                                [3, 1,    3,    1,   1],
                                [4, 1,    1,    4,   1],
                                [5, 1,    1,    1,   5]])
        
        self.matrix = self.matrix.astype(float)
        if kappa_coef is not None:
            for i in range(5):
                for j in range(5):
                    self.matrix[i, j] = ((i - j) ** kappa_coef) / (4**kappa_coef)

    def __call__(self, y_true, y_pred):
        g1, h1 = softprob_obj_matrix(y_true, y_pred, self.matrix)
        g2, h2 = softprob_obj_matrix(self.y_ref, y_pred, self.matrix)
        return g1 + g2, h1 + h2

    def softmax(self, x):
        """
        Compute softmax probabilities.
        :param x: Input array.
        :return: Softmax-transformed probabilities.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def calc_ders_multi(self, approx, target, weight):
        approx = [approx[i] - max(approx) for i in range(len(approx))]
        exp_approx = [np.exp(approx[i]) for i in range(len(approx))]
        exp_sum = np.sum(exp_approx)
        
        grad = []
        hess = []
        for j in range(len(approx)):
            der1 = -exp_approx[j] / exp_sum
            der2 = -exp_approx[j] / exp_sum
            if j == target:
                der1 += 1
            hess_row = []
            for j2 in range(len(approx)):
                der2 = exp_approx[j] * exp_approx[j2] / (exp_sum**2)
                if j2 == j:
                    der2 -= exp_approx[j] / exp_sum
                hess_row.append(der2 * weight)
                
            grad.append(der1 * weight)
            hess.append(hess_row)
        
        return (grad, hess)

######################################### DICE LOSS ############################################

class dice_loss_class(object):
    def __init__(self, num_classes):
        self.index = 0
        self.num_classes = num_classes

    def __call__(self, y_true, y_pred):
        y_pred = y_pred.reshape(-1, self.num_classes)  # Reshaper les prédictions en (n_samples, num_classes)

        # Transformation softmax pour obtenir les probabilités
        prob = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        pred_classes = np.argmax(prob, axis=1)  # Classe prédite (argmax sur les probabilités)

        # Initialisation des gradients et hessiens
        gradient = np.zeros_like(y_pred, dtype=np.float32)
        hessian = np.zeros_like(y_pred, dtype=np.float32)

        x = sp.Symbol('x')  # y_pred_c
        y = sp.Symbol('y', real=True)  # y_true_c est un scalaire symbolique
        epsilon = sp.Symbol('epsilon', real=True, positive=True)

        intersection = y * x
        #union = y**2 + x**2
        union = y + x
        f = 1 - (2 * intersection + epsilon) / (union + epsilon)

        # Gradient (dérivée de F par rapport à x)
        grad = sp.diff(f, x)

        # Hessienne (dérivée seconde de F par rapport à x)
        hess = sp.diff(grad, x)

        # Calcul de la Dice Loss par classe
        for i in range(y_pred.shape[0]):
            
            target = y_true[i]

            for c in range(self.num_classes):

                #y1 = int(target == c)
                y1 = target
                
                #x1 = y_pred[i, c]
                x1 = c

                grad_val = grad.subs({y: y1, x: x1, epsilon: 1}).evalf()
                hess_val = hess.subs({y: y1, x: x1, epsilon: 1}).evalf()

                gradient[i, c] = float(grad_val)  # Stocker le gradient pour la classe c
                hessian[i, c] = float(hess_val)
        
        gradient = gradient.flatten()
        hessian = hessian.flatten()
        #print(gradient, hessian)
        return gradient, hessian

    def softmax(self, x):
        """
        Compute softmax probabilities.
        :param x: Input array.
        :return: Softmax-transformed probabilities.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def calc_ders_multi(self, approx, target, weight):
        approx = [approx[i] - max(approx) for i in range(len(approx))]
        exp_approx = [np.exp(approx[i]) for i in range(len(approx))]
        exp_sum = np.sum(exp_approx)
        
        grad = []
        hess = []
        for j in range(len(approx)):
            der1 = -exp_approx[j] / exp_sum
            der2 = -exp_approx[j] / exp_sum
            if j == target:
                der1 += 1
            hess_row = []
            for j2 in range(len(approx)):
                der2 = exp_approx[j] * exp_approx[j2] / (exp_sum**2)
                if j2 == j:
                    der2 -= exp_approx[j] / exp_sum
                hess_row.append(der2 * weight)
                
            grad.append(der1 * weight)
            hess.append(hess_row)
        
        return (grad, hess)
    
class LogLossDual:
    def __init__(self, y):
        self.y_temp = y  # Store reference labels
        self.__name__ = "logloss_dual"

    def __call__(self, y_true, y_pred):

        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)

        loss_1 = log_loss(self.y_temp, y_pred)  
        loss_2 = log_loss(y_true, y_pred)

        loss_value = loss_1 + loss_2
        return self.__name__, float(loss_value)
 
"""
def softprob_obj_dual(y1, pred=None):
    def custom(y, pred):
        g1, h1 = softprob_obj(y1, pred)
        g2, h2 = softprob_obj(y, pred)
        return g1 + g2, h1 + h2
    return custom"""

"""def logloss_dual(y1):
    def custom(y, pred):
        return log_loss(y1, pred) + log_loss(y, pred)
    return custom"""

"""def softprob_obj_dual(y, pred):
    g1, h1 = softprob_obj(y[:, 0], pred)
    g2, h2 = softprob_obj(y[:, 0], pred)
    return g1 + g2, h1 + h2

def logloss_dual(y, pred):
    return log_loss(y[:, 0], pred) + log_loss(y[:, 0], pred)"""

def dgpd_loss(y_true, y_pred, alpha=15.0):
    """
    Discrete Generalized Pareto Distribution (dGPD) Loss function.
    
    Parameters:
    y_true : Tensor of true wildfire counts.
    y_pred : Tensor of predicted values (log-scale of scale parameter xi).
    alpha  : Tail shape parameter (1/xi). Must be > 0.
    
    Returns:
    Loss value computed using dGPD.
    """
    xi = np.exp(y_pred)  # Ensuring positivity
    term1 = (1 + xi * y_true) ** (-alpha)
    term2 = (1 + xi * (y_true + 1)) ** (-alpha)
    loss = term1 - term2
    return np.mean(loss)

def dgpd_gradient_hessian(y_true, y_pred, alpha=1.0):
    """
    Computes the gradient and Hessian of the dGPD loss function.
    
    Parameters:
    y_true : Tensor of true wildfire counts.
    y_pred : Tensor of predicted values (log-scale of scale parameter xi).
    alpha  : Tail shape parameter (1/xi).
    
    Returns:
    Gradient and Hessian tensors.
    """
    xi = np.exp(y_pred)
    term1 = (1 + xi * y_true) ** (-alpha)
    term2 = (1 + xi * (y_true + 1)) ** (-alpha)
    
    grad = alpha * (term1 * y_true - term2 * (y_true + 1)) * xi
    hessian = alpha * xi ** 2 * (term1 * y_true ** 2 - term2 * (y_true + 1) ** 2)
    
    return grad, hessian

def compute_kappa(y_true, threshold_class):
    """
    Compute kappa as the percentage of samples beyond a certain class threshold.
    
    Parameters:
    y_true : Tensor of true class labels.
    threshold_class : Class index beyond which extreme events are considered.
    
    Returns:
    kappa value computed as the proportion of extreme events.
    """
    total_samples = y_true.shape[0]
    extreme_samples = np.sum(y_true >= threshold_class)
    kappa = 1.0 - (extreme_samples / total_samples)  # Higher kappa means fewer extreme cases
    return max(min(kappa, 0.99), 0.01)  # Keep kappa within a reasonable range

def gpd_gradient_hessian(y_true, y_pred, threshold_class=4, xi=0.5):
    """
    Computes the gradient and Hessian of the GPD loss function for classification.
    
    Parameters:
    y_true : Tensor of true class labels.
    y_pred : Tensor of predicted values (logits for class probabilities).
    kappa  : High quantile threshold.
    xi     : Tail index parameter.
    
    Returns:
    Gradient and Hessian tensors.
    """
    kappa = compute_kappa(y_true, threshold_class)
    sigma = np.exp(y_pred)
    threshold_term = (1 - kappa) ** (-xi) - 1
    scaled_excess = (y_true * threshold_term) / sigma
    
    grad = ((xi + 1) / (xi * (1 + scaled_excess))) * (threshold_term / sigma) - 1/sigma
    hessian = -((xi + 1) / (xi * (1 + scaled_excess) ** 2)) * (threshold_term / sigma ** 2)
    return grad, hessian

def gpd_multiclass_loss(y_true, y_pred, kappa=0.99, xi=0.5):
    """
    Generalized Pareto Distribution (GPD) Loss function for multi-class classification.
    
    Parameters:
    y_true : Tensor of true class labels (one-hot encoded or categorical indices).
    y_pred : Tensor of predicted logits for each class.
    kappa  : High quantile threshold.
    xi     : Tail index parameter.
    
    Returns:
    Loss value computed using GPD for multi-class classification.
    """
    sigma = np.exp(y_pred)
    threshold_term = (1 - kappa) ** (-xi) - 1
    scaled_excess = (y_true * threshold_term) / sigma
    loss = np.mean(((xi + 1) / xi) * np.log(1 + scaled_excess) + np.log(sigma))
    return loss


def egpd_loss(y_true, y_pred, alpha=1.0, xi=0.5):
    """
    Extended Generalized Pareto Distribution (eGPD) Loss function.
    
    Parameters:
    y_true : Tensor of true wildfire counts or extreme event values.
    y_pred : Tensor of predicted logits for severity scores.
    alpha  : Tail shape parameter (1/xi). Must be > 0.
    xi     : Tail index parameter.
    
    Returns:
    Loss value computed using eGPD.
    """
    sigma = np.exp(y_pred)  # Ensure positivity
    term1 = (1 + xi * y_true / sigma) ** (-alpha)
    term2 = (1 + xi * (y_true + 1) / sigma) ** (-alpha)
    loss = term1 - term2
    return np.mean(loss)

def egpd_gradient_hessian(y_true, y_pred, alpha=1.0, xi=0.5):
    """
    Computes the gradient and Hessian of the eGPD loss function.
    
    Parameters:
    y_true : Tensor of true wildfire counts or extreme event values.
    y_pred : Tensor of predicted logits for severity scores.
    alpha  : Tail shape parameter (1/xi).
    xi     : Tail index parameter.
    
    Returns:
    Gradient and Hessian tensors.
    """
    sigma = np.exp(y_pred)
    term1 = (1 + xi * y_true / sigma) ** (-alpha)
    term2 = (1 + xi * (y_true + 1) / sigma) ** (-alpha)
    
    grad = alpha * xi * (term1 * y_true - term2 * (y_true + 1)) / sigma
    hessian = -alpha * xi ** 2 * (term1 * y_true ** 2 - term2 * (y_true + 1) ** 2) / sigma ** 2
    
    return grad, hessian

def calculate_signal_scores(y_pred, y_true, y_fire, graph_id, saison):
    """
    Calcule les scores (aire commune, union, sous-prédiction, sur-prédiction) entre deux signaux.

    Args:
        t (np.array): Tableau de temps ou indices (axe x).
        y_pred (np.array): Signal prédiction (rouge).
        y_true (np.array): Signal vérité terrain (bleu).

    Returns:
        dict: Dictionnaire contenant les scores calculés.
    """

    y_true_fire = np.copy(y_true)

    ###################################### I. For the all signal ####################################
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    over_prediction_zeros = np.trapz(np.maximum(0, y_pred[y_true == 0]))
    under_prediction_zeros = np.trapz(np.maximum(0, y_true[y_pred == 0]))

    mask_fire = (y_pred > 0) | (y_true_fire > 0)
    intersection_fire = np.trapz(np.minimum(y_pred[mask_fire], y_true_fire[mask_fire]))
    union_fire = np.trapz(np.maximum(y_pred[mask_fire], y_true_fire[mask_fire]))
    iou_wildfire_or_pred = intersection_fire / union_fire if union_fire > 0 else np.nan

    mask_fire = (y_pred > 0) & (y_true_fire > 0)
    intersection_fire = np.trapz(np.minimum(y_pred[mask_fire], y_true_fire[mask_fire]))
    union_fire = np.trapz(np.maximum(y_pred[mask_fire], y_true_fire[mask_fire]))
    iou_wildfire_and_pred = intersection_fire / union_fire if union_fire > 0 else np.nan

    # Limitation des signaux à un maximum de 1
    y_pred_clipped = np.clip(y_pred, 0, 1)  # Limiter y_pred à 1
    y_true_fire_clipped = np.clip(y_true_fire, 0, 1)  # Limiter y_true_fire à 1
    print(np.unique(y_true_fire_clipped))
    print(np.unique(y_pred_clipped))
    iou_wildfire_detected = recall_score(y_true_fire_clipped, y_pred_clipped)

    y_pred_clipped_ytrue = np.copy(y_pred)
    y_pred_clipped_ytrue[(y_pred > 0) & (y_true > 0)] = np.minimum(y_true[(y_pred > 0) & (y_true > 0)], y_pred[(y_pred > 0) & (y_true > 0)])
    intersection_clipped = np.trapz(np.minimum(y_pred_clipped_ytrue, y_true))  # Aire commune
    union_clipped = np.trapz(np.maximum(y_pred_clipped_ytrue, y_true))         # Aire d'union
    iou_no_overestimation = intersection_clipped / union_clipped if union_clipped > 0 else np.nan

    dice_coefficient = 2 * intersection / (union + intersection) if union + intersection > 0 else np.nan

    # Enregistrement dans un dictionnaire
    scores = {
        "iou": intersection / union if union > 0 else np.nan,  # To avoid division by zero
        "iou_wildfire_or_pred": iou_wildfire_or_pred,
        "iou_wildfire_and_pred": iou_wildfire_and_pred,
        "iou_wildfire_detected": iou_wildfire_detected,
        "iou_no_overestimation" : iou_no_overestimation,
        
        "over_bad_prediction" : over_prediction_zeros / union if union > 0 else np.nan,
        "under_bad_prediction" : under_prediction_zeros / union if union > 0 else np.nan,
        "bad_prediction" : (over_prediction_zeros + under_prediction_zeros) / union if union > 0 else np.nan,
        
        # Ajout du Dice coefficient
        "dice_coefficient": dice_coefficient
    }

    ###################################### I. For each graph_id ####################################
    unique_graph_ids = np.unique(graph_id)

    # Trier les graph_id en fonction de la somme de event_col
    graph_sums = {g_id: y_true[graph_id == g_id].sum() for g_id in unique_graph_ids}
    sorted_graph_ids = sorted(graph_sums, key=graph_sums.get, reverse=True)

    # Parcourir les graph_id triés
    for i, g_id in enumerate(sorted_graph_ids):
        mask = graph_id == g_id

        y_pred_graph = y_pred[mask]
        y_true_graph = y_true[mask]
        y_true_fire_graph = y_true_fire[mask]

        mask_fire_graph = (y_pred_graph > 0) | (y_true_fire_graph > 0)
        intersection_fire_graph = np.trapz(np.minimum(y_pred_graph[mask_fire_graph], y_true_fire_graph[mask_fire_graph]))
        union_fire_graph = np.trapz(np.maximum(y_pred_graph[mask_fire_graph], y_true_fire_graph[mask_fire_graph]))
        iou_wildfire_or_pred_graph = intersection_fire_graph / union_fire_graph if union_fire_graph > 0 else np.nan

        mask_fire_graph = (y_pred_graph > 0) & (y_true_fire_graph > 0)
        intersection_fire_graph = np.trapz(np.minimum(y_pred_graph[mask_fire_graph], y_true_fire_graph[mask_fire_graph]))
        union_fire_graph = np.trapz(np.maximum(y_pred_graph[mask_fire_graph], y_true_fire_graph[mask_fire_graph]))
        iou_wildfire_and_pred_graph = intersection_fire_graph / union_fire_graph if union_fire_graph > 0 else np.nan

        # Limitation des signaux à un maximum de 1
        y_pred_clipped = np.clip(y_pred[mask], 0, 1)  # Limiter y_pred à 1
        y_true_fire_clipped = np.clip(y_true_fire[mask], 0, 1)  # Limiter y_true_fire à 1

        iou_wildfire_detected = recall_score(y_true_fire_clipped, y_pred_clipped, zero_division=np.nan)

        y_pred_clipped_ytrue = np.copy(y_pred_graph)
        y_pred_clipped_ytrue[(y_pred_graph > 0) & (y_true_graph > 0)] = np.minimum(y_true_graph[(y_pred_graph > 0) & (y_true_graph > 0)], y_pred_graph[(y_pred_graph > 0) & (y_true_graph > 0)])
        intersection_clipped = np.trapz(np.minimum(y_pred_clipped_ytrue, y_true_graph))  # Aire commune
        union_clipped = np.trapz(np.maximum(y_pred_clipped_ytrue, y_true_graph))         # Aire d'union
        iou_no_overestimation = intersection_clipped / union_clipped if union_clipped > 0 else np.nan

        intersection_graph = np.trapz(np.minimum(y_pred_graph, y_true_graph))  # Aire commune
        union_graph = np.trapz(np.maximum(y_pred_graph, y_true_graph))         # Aire d'union

        under_prediction_graph = np.trapz(np.maximum(0, y_true_graph - y_pred_graph))
        over_prediction_graph = np.trapz(np.maximum(0, y_pred_graph - y_true_graph))

        over_prediction_zeros_graph = np.trapz(np.maximum(0, y_pred_graph[y_true_graph == 0]))
        under_prediction_zeros_graph = np.trapz(np.maximum(0, y_true_graph[y_pred_graph == 0]))

        under_prediction_fire_graph = np.trapz(np.maximum(0, y_true_graph[y_true_graph > 0] - y_pred_graph[y_true_graph > 0]))
        over_prediction_fire_graph = np.trapz(np.maximum(0, y_pred_graph[y_true_graph > 0] - y_true_graph[y_true_graph > 0]))

        only_true_value = np.trapz(y_true_graph)  # Aire sous la courbe des valeurs réelles
        only_pred_value = np.trapz(y_pred_graph)  # Aire sous la courbe des prédictions

        # Stocker les scores avec des clés utilisant uniquement l'indice
        graph_scores = {
            f"iou_wildfire_or_pred_{i}": iou_wildfire_or_pred_graph,
            f"iou_wildfire_and_pred_{i}": iou_wildfire_and_pred_graph,
            f"iou_no_overestimation_{i}": iou_no_overestimation,
            f"iou_{i}": intersection_graph / union_graph if union_graph > 0 else np.nan,  # Pour éviter la division par zéro
            f"iou_wildfire_detected_{i}": iou_wildfire_detected,

            # Ajout du Dice coefficient pour chaque itération
            f"dice_coefficient_{i}": 2 * intersection_graph / (union_graph + intersection_graph) if (union_graph + intersection_graph) > 0 else np.nan,

            f"over_bad_prediction_local_{i}": over_prediction_zeros_graph / union_graph if union_graph > 0 else np.nan,
            f"under_bad_prediction_local_{i}": under_prediction_zeros_graph / union_graph if union_graph > 0 else np.nan,
            f"bad_prediction_local_{i}": (over_prediction_zeros_graph + under_prediction_zeros_graph) / union_graph if union_graph > 0 else np.nan,

            f"over_bad_prediction_global_{i}": over_prediction_zeros_graph / union if union_graph > 0 else np.nan,
            f"under_bad_prediction_global_{i}": under_prediction_zeros_graph / union if union_graph > 0 else np.nan,
            f"bad_prediction_global_{i}": (over_prediction_zeros_graph + under_prediction_zeros_graph) / union if union_graph > 0 else np.nan,
        }

        scores.update(graph_scores)

    unique_seasons = np.unique(saison)

    # Trier les saisons en fonction de la somme de y_true
    season_sums = {s: y_true[saison == s].sum() for s in unique_seasons}
    sorted_seasons = sorted(season_sums, key=season_sums.get, reverse=True)

    # Parcourir les saisons triées
    for i, s in enumerate(sorted_seasons):
        mask = saison == s

        y_pred_season = y_pred[mask]
        y_true_season = y_true[mask]
        y_true_fire_season = y_true_fire[mask]

        mask_fire_season = (y_pred_season > 0) | (y_true_fire_season > 0)
        intersection_fire_season = np.trapz(np.minimum(y_pred_season[mask_fire_season], y_true_fire_season[mask_fire_season]))
        union_fire_season = np.trapz(np.maximum(y_pred_season[mask_fire_season], y_true_fire_season[mask_fire_season]))
        iou_wildfire_or_pred_season = intersection_fire_season / union_fire_season if union_fire_season > 0 else np.nan

        mask_fire_season = (y_pred_season > 0) & (y_true_fire_season > 0)
        intersection_fire_season = np.trapz(np.minimum(y_pred_season[mask_fire_season], y_true_fire_season[mask_fire_season]))
        union_fire_season = np.trapz(np.maximum(y_pred_season[mask_fire_season], y_true_fire_season[mask_fire_season]))
        iou_wildfire_and_pred_season = intersection_fire_season / union_fire_season if union_fire_season > 0 else np.nan

        # Limitation des signaux à un maximum de 1
        y_pred_clipped = np.clip(y_pred[mask], 0, 1)  # Limiter y_pred à 1
        y_true_fire_clipped = np.clip(y_true_fire[mask], 0, 1)  # Limiter y_true_fire à 1

        iou_wildfire_detected = recall_score(y_true_fire_clipped, y_pred_clipped, zero_division=np.nan)

        y_pred_clipped_ytrue = np.copy(y_pred_season)
        y_pred_clipped_ytrue[(y_pred_season > 0) & (y_true_season > 0)] = np.minimum(y_true_season[(y_pred_season > 0) & (y_true_season > 0)], y_pred_season[(y_pred_season > 0) & (y_true_season > 0)])
        intersection_clipped = np.trapz(np.minimum(y_pred_clipped_ytrue, y_true_season))  # Aire commune
        union_clipped = np.trapz(np.maximum(y_pred_clipped_ytrue, y_true_season))         # Aire d'union
        iou_no_overestimation = intersection_clipped / union_clipped if union_clipped > 0 else np.nan

        intersection_season = np.trapz(np.minimum(y_pred_season, y_true_season))  # Aire commune
        union_season = np.trapz(np.maximum(y_pred_season, y_true_season))         # Aire d'union

        over_prediction_zeros_season = np.trapz(np.maximum(0, y_pred_season[y_true_season == 0]))
        under_prediction_zeros_season = np.trapz(np.maximum(0, y_true_season[y_pred_season == 0]))

        # Stocker les scores avec des clés utilisant uniquement l'indice
        season_scores = {
            f"iou_wildfire_or_pred_{s}": iou_wildfire_or_pred_season,
            f"iou_wildfire_and_pred_{s}": iou_wildfire_and_pred_season,
            f"iou_no_overestimation_{s}": iou_no_overestimation,
            f"iou_{s}": intersection_season / union_season if union_season > 0 else np.nan,  # Pour éviter la division par zéro
            f"iou_wildfire_detected_{s}": iou_wildfire_detected,

            f"over_bad_prediction_local_{s}": over_prediction_zeros_season / union_season if union_season > 0 else np.nan,
            f"under_bad_prediction_local_{s}": under_prediction_zeros_season / union_season if union_season > 0 else np.nan,
            f"bad_prediction_local_{s}": (over_prediction_zeros_season + under_prediction_zeros_season) / union_season if union_season > 0 else np.nan,
            
            f"dice_coefficient_{s}": 2 * intersection_season / (union_season + intersection_season) if (union_season + intersection_season) > 0 else np.nan,

            f"over_bad_prediction_global_{s}": over_prediction_zeros_season / union if union_season > 0 else np.nan,
            f"under_bad_prediction_global_{s}": under_prediction_zeros_season / union if union_season > 0 else np.nan,
            f"bad_prediction_global_{s}": (over_prediction_zeros_season + under_prediction_zeros_season) / union if union_season > 0 else np.nan,
        }
        scores.update(season_scores)

    ###################################### For each graph_id in each season ####################################
    # Get unique seasons
    unique_seasons = np.unique(saison)

    # Iterate over seasons
    for season in unique_seasons:
        # Mask for the current season
        season_mask = saison == season

        # Iterate over graphs in this season
        for i, g_id in enumerate(sorted_graph_ids):
            # Mask for the current graph in the current season
            mask = (graph_id == g_id) & season_mask

            y_pred_graph_season = y_pred[mask]
            y_true_graph_season = y_true[mask]
            y_true_fire_graph_season = y_true_fire[mask]

            mask_fire_graph_season = (y_pred_graph_season > 0) | (y_true_fire_graph_season > 0)
            intersection_fire_graph_season = np.trapz(np.minimum(y_pred_graph_season[mask_fire_graph_season], y_true_fire_graph_season[mask_fire_graph_season]))
            union_fire_graph_season = np.trapz(np.maximum(y_pred_graph_season[mask_fire_graph_season], y_true_fire_graph_season[mask_fire_graph_season]))
            iou_wildfire_or_pred_graph_season = intersection_fire_graph_season / union_fire_graph_season if union_fire_graph_season > 0 else np.nan

            mask_fire_graph_season = (y_pred_graph_season > 0) & (y_true_fire_graph_season > 0)
            intersection_fire_graph_season = np.trapz(np.minimum(y_pred_graph_season[mask_fire_graph_season], y_true_fire_graph_season[mask_fire_graph_season]))
            union_fire_graph_season = np.trapz(np.maximum(y_pred_graph_season[mask_fire_graph_season], y_true_fire_graph_season[mask_fire_graph_season]))
            iou_wildfire_and_pred_graph_season = intersection_fire_graph_season / union_fire_graph_season if union_fire_graph_season > 0 else np.nan

            # Limitation des signaux à un maximum de 1
            y_pred_clipped = np.clip(y_pred[mask], 0, 1)  # Limiter y_pred à 1
            y_true_fire_clipped = np.clip(y_true_fire[mask], 0, 1)  # Limiter y_true_fire à 1

            iou_wildfire_detected = recall_score(y_true_fire_clipped, y_pred_clipped, zero_division=np.nan)

            y_pred_clipped_ytrue = np.copy(y_pred_graph_season)
            y_pred_clipped_ytrue[(y_pred_graph_season > 0) & (y_true_graph_season > 0)] = np.minimum(y_true_graph_season[(y_pred_graph_season > 0) & (y_true_graph_season > 0)], y_pred_graph_season[(y_pred_graph_season > 0) & (y_true_graph_season > 0)])
            intersection_clipped = np.trapz(np.minimum(y_pred_clipped_ytrue, y_true_graph_season))  # Aire commune
            union_clipped = np.trapz(np.maximum(y_pred_clipped_ytrue, y_true_graph_season))         # Aire d'union
            iou_no_overestimation = intersection_clipped / union_clipped if union_clipped > 0 else np.nan

            intersection_graph_season = np.trapz(np.minimum(y_pred_graph_season, y_true_graph_season))  # Common area
            union_graph_season = np.trapz(np.maximum(y_pred_graph_season, y_true_graph_season))         # Union area

            over_prediction_zeros_graph_season = np.trapz(np.maximum(0, y_pred_graph_season[y_true_graph_season == 0]))
            under_prediction_zeros_graph_season = np.trapz(np.maximum(0, y_true_graph_season[y_pred_graph_season == 0]))

            # Compute scores for the graph in this season
            graph_season_scores = {
                f"iou_wildfire_or_pred_graph_{i}_season_{season}": iou_wildfire_or_pred_graph_season,
                f"iou_wildfire_and_pred_graph_{i}_season_{season}": iou_wildfire_and_pred_graph_season,
                f"iou_no_overestimation_graph_{i}_season_{season}": iou_no_overestimation,
                f"iou_graph_{i}_season_{season}": intersection_graph_season / union_graph_season if union_graph_season > 0 else np.nan,
                f"iou_wildfire_detected_graph_{i}_season_{season}": iou_wildfire_detected,
                
                f"dice_coefficient_graph_{i}_season_{season}": 2 * intersection_graph_season / (union_graph_season + intersection_graph_season) if (union_graph_season + intersection_graph_season) > 0 else np.nan,

                f"over_bad_prediction_local_graph_{i}_season_{season}": over_prediction_zeros_graph_season / union_graph_season if union_graph_season > 0 else np.nan,
                f"under_bad_prediction_local_graph_{i}_season_{season}": under_prediction_zeros_graph_season / union_graph_season if union_graph_season > 0 else np.nan,
                f"bad_prediction_local_graph_{i}_season_{season}": (over_prediction_zeros_graph_season + under_prediction_zeros_graph_season) / union_graph_season if union_graph_season > 0 else np.nan,
            }

            # Update global scores dictionary
            scores.update(graph_season_scores)

    # Parcourir les valeurs uniques de y_true
    for unique_value in np.unique(y_true[y_true > 0]):
        # Créer un masque pour sélectionner les éléments correspondant à la valeur unique
        mask = (y_true == unique_value) | (y_pred == unique_value)

        y_pred_sample = y_pred[mask]
        y_true_sample = y_true[mask]
        y_true_fire_sample = y_true_fire[mask]

        if y_pred_sample.shape[0] == 0:
            continue

        if y_pred_sample.shape[0] == 1:
            y_pred_sample = np.concatenate((y_pred_sample, y_pred_sample))
            y_true_sample = np.concatenate((y_true_sample, y_true_sample))
            y_true_fire_sample = np.concatenate((y_true_fire_sample, y_true_fire_sample))

        mask_fire_sample = (y_pred_sample > 0) | (y_true_fire_sample > 0)
        intersection_fire_sample = np.trapz(np.minimum(y_pred_sample[mask_fire_sample], y_true_fire_sample[mask_fire_sample]))
        union_fire_sample = np.trapz(np.maximum(y_pred_sample[mask_fire_sample], y_true_fire_sample[mask_fire_sample]))
        iou_wildfire_or_pred_sample = intersection_fire_sample / union_fire_sample if union_fire_sample > 0 else np.nan

        mask_fire_sample = (y_pred_sample > 0) & (y_true_fire_sample > 0)
        intersection_fire_sample = np.trapz(np.minimum(y_pred_sample[mask_fire_sample], y_true_fire_sample[mask_fire_sample]))
        union_fire_sample = np.trapz(np.maximum(y_pred_sample[mask_fire_sample], y_true_fire_sample[mask_fire_sample]))
        iou_wildfire_and_pred_sample = intersection_fire_sample / union_fire_sample if union_fire_sample > 0 else np.nan

        # Limitation des signaux à un maximum de 1
        y_pred_clipped = np.clip(y_pred_sample, 0, 1)  # Limiter y_pred à 1
        y_true_fire_clipped = np.clip(y_true_fire_sample, 0, 1)  # Limiter y_true_fire à 1

        # Calcul de la métrique IOU
        #iou_wildfire_detected = intersection_fire_detected / union_fire_detected if union_fire_detected > 0 else np.nan
        iou_wildfire_detected = recall_score(y_true_fire_clipped, y_pred_clipped, zero_division=np.nan)

        y_pred_clipped_ytrue = np.copy(y_pred_sample)
        y_pred_clipped_ytrue[(y_pred_sample > 0) & (y_true_sample > 0)] = np.minimum(y_true_sample[(y_pred_sample > 0) & (y_true_sample > 0)], y_pred_sample[(y_pred_sample > 0) & (y_true_sample > 0)])
        intersection_clipped = np.trapz(np.minimum(y_pred_clipped_ytrue, y_true_sample))  # Aire commune
        union_clipped = np.trapz(np.maximum(y_pred_clipped_ytrue, y_true_sample))         # Aire d'union
        iou_no_overestimation = intersection_clipped / union_clipped if union_clipped > 0 else np.nan

        # Calculer les aires
        intersection = np.trapz(np.minimum(y_pred_sample, y_true_sample))  # Aire commune
        union = np.trapz(np.maximum(y_pred_sample, y_true_sample))        # Aire d'union

        under_prediction = np.trapz(np.maximum(0, y_true_sample - y_pred_sample))
        over_prediction = np.trapz(np.maximum(0, y_pred_sample - y_true_sample))

        over_prediction_zeros = np.trapz(np.maximum(0, y_pred_sample[y_true_sample == 0]))
        under_prediction_zeros = np.trapz(np.maximum(0, y_true_sample[y_pred_sample == 0]))

        under_prediction_fire = np.trapz(
            np.maximum(0, y_true_sample[y_true_sample > 0] - y_pred_sample[y_true_sample > 0])
        )
        over_prediction_fire = np.trapz(
            np.maximum(0, y_pred_sample[y_true_sample > 0] - y_true_sample[y_true_sample > 0])
        )

        # Enregistrement dans un dictionnaire
        scores_elt = {            
            f"iou_elt_{unique_value}": intersection / union if union > 0 else np.nan,  # Éviter la division par zéro
            f"iou_wildfire_detected_elt_{unique_value}": iou_wildfire_detected,
            f"iou_wildfire_or_pred_elt_{unique_value}": iou_wildfire_or_pred_sample,
            f"iou_wildfire_and_pred_elt_{unique_value}": iou_wildfire_and_pred_sample,
            f"iou_no_overestimation_elt_{unique_value}": iou_no_overestimation,

            f"dice_coefficient_elt_{unique_value}": 2 * intersection / (union + intersection) if (union + intersection) > 0 else np.nan,

            f"over_bad_prediction_elt_{unique_value}": over_prediction_zeros / union if union > 0 else np.nan,
            f"under_bad_prediction_elt_{unique_value}": under_prediction_zeros / union if union > 0 else np.nan,
            f"bad_prediction_elt_{unique_value}": (over_prediction_zeros + under_prediction_zeros) / union if union > 0 else np.nan,
        }

        # Ajouter les scores pour cette valeur unique à la collection globale
        scores.update(scores_elt)

    return scores

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import f1_score

def plot_result(dff, metric, dataset, x_col, y_true_col='target', y_pred_col='y_pred', top='all', show_area=False, show_metric=False):
    """
    Affiche un graphique par Scale, avec une colonne spécifiée pour l'axe des X triée selon 'nbsinister'.
    Chaque courbe représente un modèle et affiche l'IoU entre l'aire sous la courbe et l'aire maximale dans la légende.

    :param dff: DataFrame contenant les colonnes ['Department', 'Scale', 'Model', 'nbsinister', metric, 'target', 'y_pred']
    :param metric: Nom de la colonne contenant la métrique à afficher
    :param dataset: Nom du dataset à filtrer
    :param x_col: Colonne à utiliser pour les axes des X
    :param y_true_col: Colonne représentant les cibles réelles
    :param y_pred_col: Colonne représentant les prédictions
    :param top: Nombre de valeurs à afficher (ou 'all' pour tout afficher)
    :param show_area: Booléen pour afficher l'aire sous la courbe
    :param show_metric: Booléen pour afficher la métrique de chaque modèle
    """
    # Filtrer le DataFrame par dataset
    df = dff[dff['Dataset'] == dataset].copy(deep=True)
    
    # Trier les valeurs par 'nbsinister' décroissant
    df_sorted = df.sort_values(by='nbsinister', ascending=False)
    
    # Sélectionner les "top" valeurs si nécessaire
    if top != 'all':
        top = int(top)
        top_values = df_sorted[x_col].unique()[:top]
        df_sorted = df_sorted[df_sorted[x_col].isin(top_values)]
    
    # Assurer que tous les modèles ont les mêmes valeurs en X
    all_values = df_sorted[x_col].unique()
    num_x_values = len(all_values)
    
    # Déterminer l'aire maximale en fonction de 'nbsinister'
    max_area = np.trapz((df_sorted.groupby(x_col)['nbsinister'].sum() > 0).astype(int), dx=1)
    
    # Récupérer les échelles uniques
    scales = df_sorted['Scale'].unique()
    num_scales = len(scales)
    
    # Définir la disposition de la grille
    if num_scales > 3:
        cols = 3
        rows = math.ceil(num_scales / cols)
    else:
        cols = 1
        rows = num_scales
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i, scale in enumerate(scales):
        ax = axes[i]
        df_scale = df_sorted[df_sorted['Scale'] == scale].copy()
        df_scale.drop_duplicates(subset=[x_col, 'Model'], inplace=True)
        
        # Créer un pivot pour assurer l'alignement des valeurs sur X
        pivot_df = df_scale.pivot(index=x_col, columns='Model', values=metric)
        pivot_df = pivot_df.reindex(all_values)  # Respecter l'ordre des valeurs
        
        # Calcul de l'IoU et du F1-score
        IoU_values = {}
        F1_values = {}
        
        for model in pivot_df.columns:
            y_values = pivot_df[model].dropna()
            if len(y_values) == 0:
                continue
            x_values = np.arange(len(y_values))
            
            # Calcul de l'IoU en utilisant la fonction iou_score
            y_true = df_scale[df_scale['Model'] == model][y_true_col]
            y_pred = df_scale[df_scale['Model'] == model][y_pred_col]
            IoU = iou_score(y_true, y_pred)
            IoU_values[model] = IoU
            
            # Calcul du F1-score
            f1 = f1_score(y_true, y_pred)
            F1_values[model] = f1
            
            # Tracer les courbes par modèle
            model_area = np.trapz(y_values, x_values)
            IoU = model_area / max_area if max_area > 0 else 0

            ax.plot(x_values, y_values, label=f'{model} (IoU={IoU:.3f})')
        
        # Affichage de la métrique F1
        if show_metric:
            for model in F1_values:
                ax.text(0.95, 0.95, f'{model}: F1={F1_values[model]:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=10)

        # Affichage de l'aire sous la courbe
        if show_area:
            ax.fill_between(x_values, y_values, alpha=0.3)
        
        ax.set_title(f'Scale: {scale}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()            
        
def evaluate_pipeline(dir_train, df_test, pred, y, target_name, name,
                      pred_min=None, pred_max=None):
    
    metrics = {}

    dir_predictor = dir_train / 'influenceClustering'

    print(f'WARNING : WE CONSIDER PRED[0] = PRED[1]')

    ############################################## Get daily metrics #######################################################
    if name.find('classification') != -1:
        col_class = target_name
        col_class_1 = 'nbsinister-kmeans-5-Class-Dept'
        col_class_2 = 'nbsinister-kmeans-5-Class-Dept-cubic-Specialized'
        col_nbsinister = 'nbsinister'

    elif name.find('binary') != -1:
        col_class = target_name
        col_class_1 = 'nbsinister-binary'
        col_class_2 = 'nbsinister-kmeans-5-Class-Dept-cubic-Specialized'
        col_nbsinister = 'nbsinister'
        
    elif name.find('regression') != -1:
        col_nbsinister = target_name
        col_class = 'nbsinister-MinMax-5-Class-Dept'
    
    metrics = {}

    print(f'###################### Analysis {target_name} #########################')
 
    #plot_and_save_roc_curve(df_test['nbsinister'].values > 0, pred[:, 0], dir_output / name, target_name, departement_scale)
    #plot_and_save_pr_curve(df_test['nbsinister'].values > 0, pred[:, 0], dir_output / name, target_name, departement_scale)
    
    #calibrated_curve(pred[:, 0], y, dir_output / name, 'calibration')
    #calibrated_curve(pred[:, 1], y, dir_output / name, 'class_calibration')

    #shapiro_wilk(pred[:,0], y[:,-1], dir_output / name, f'shapiro_wilk_{scale}')
    
    df_test['saison'] = df_test['date'].apply(get_saison)

    res_temp = pd.DataFrame(index=np.arange(0, pred.shape[0]))
    res_temp['graph_id'] = y[:, graph_id_index]
    res_temp['date'] = y[:, date_index]
    res_temp[f'prediction_{target_name}'] = pred[:, 0]

    df_test = df_test.set_index(['graph_id', 'date']).join(res_temp.set_index(['graph_id', 'date'])[f'prediction_{target_name}'], on=['graph_id', 'date']).reset_index()
    
    res = df_test.copy(deep=True)

    ####################################### Sinister Evaluation ########################################

    print(f'###################### Analysis {col_nbsinister} #########################')

    y_pred = pred[:, 0]
    if pred_max is not None:
        y_pred_max = pred_max[:, 0]
        y_pred_min = pred_min[:, 0]

    y_true = df_test[col_nbsinister].values
    y_pred = df_test[f'prediction_{target_name}'].values
    
    #apr = round(average_precision_score(y_true > 0, y_pred > 0), 2)

    """ks, _ = calculate_ks_continous(df_test, 'prediction', col_nbsinister, dir_output / name)

    r2 = r2_score(y_true, y_pred)

    metrics[f'nbsinister'] = df_test['nbsinister'].sum()
    print(f'Number of sinister = {df_test["nbsinister"].sum()}')

    metrics[f'nb'] = df_test[col_nbsinister].sum()
    print(f'Number of {col_nbsinister} = {df_test[col_nbsinister].sum()}')

    metrics[f'r2'] = r2
    print(f'r2 = {r2}')

    metrics[f'KS'] = ks
    print(f'KS = {ks}')"""

    #metrics[f'apr'] = apr
    #print(f'apr = {apr}')

    f1 = round(f1_score(y_true > 0, y_pred > 0), 3)
    metrics[f'f1'] = f1
    print(f'f1 = {f1}')

    prec = round(precision_score(y_true > 0, y_pred > 0), 3)
    metrics[f'prec'] = prec
    print(f'prec = {prec}')

    rec = round(recall_score(y_true > 0, y_pred > 0), 3)
    metrics[f'rec'] = rec
    print(f'rec = {rec}')

    bca = round(balanced_accuracy_score(y_true, y_pred), 3)
    metrics[f'bca'] = bca
    print(f'bca = {bca}')

    # Calcul des scores pour les signaux
    #iou_dict = calculate_signal_scores(y_pred, y_true, df_test['nbsinister'].values, df_test['graph_id'].values, df_test['saison'].values)

    """# Sauvegarder toutes les métriques calculées dans le dictionnaire metrics
    for key, value in iou_dict.items():
        metric_key = f'{key}_sinister'  # Ajouter un suffixe basé sur col_for_dict
        metrics[metric_key] = value  # Ajouter au dictionnaire des métriques
        print(f'{metric_key} = {value}')  # Afficher la métrique enregistrée"""

    y_true_temp = np.ones((y_pred.shape[0], y.shape[1]))
    y_true_temp[:, graph_id_index] = df_test['graph_id']
    y_true_temp[:, id_index] = df_test['graph_id']
    y_true_temp[:, departement_index] = df_test['departement']
    y_true_temp[:, date_index] = df_test['date']
    y_true_temp[:, -1] = y_true
    y_true_temp[:, -2] = df_test['nbsinister']
    y_true_temp[:, -3] = df_test[col_nbsinister]

    #realVspredict(y_pred, y_true_temp, -1,
    #    dir_output / name, col_nbsinister,
    #    pred_min, pred_max)
    
    print(f'###################### Analysis {col_class} #########################')

    #_, iv = calculate_woe_iv(res, f'prediction_{target_name}', 'nbsinister')
    #metrics['IV'] = round(iv, 2)  # Ajouter au dictionnaire des métriques
    #print(f'IV = {iv}')

    """y_pred = pred[:, 1]
    silhouette_score = round(silhouette_score_with_plot(y_pred.reshape(-1,1), df_test['nbsinister'].values.reshape(-1,1), 'all', dir_output / name), 2)
    metrics['SS'] = silhouette_score
    print(f'SS = {silhouette_score}')

    mask_fire_pred = (y_pred > 0) | (df_test['nbsinister'].values > 0)

    silhouette_score_no_zeros = round(silhouette_score_with_plot(y_pred[mask_fire_pred].reshape(-1,1), df_test['nbsinister'][mask_fire_pred].values.reshape(-1,1), 'fire_or_pred', dir_output / name), 2)
    metrics['SS_no_zeros'] = silhouette_score_no_zeros
    print(f'SS_no_zeros = {silhouette_score_no_zeros}')

    silhouette_score = round(silhouette_score_with_plot(df_test[col_class].values.reshape(-1,1), df_test['nbsinister'].values.reshape(-1,1), 'all_gt', dir_output / name), 2)
    metrics['SS_gt'] = silhouette_score
    print(f'SS_gt = {silhouette_score}')

    mask_fire_pred = (df_test[col_class].values > 0) | (df_test['nbsinister'].values > 0)

    silhouette_score_no_zeros = round(silhouette_score_with_plot(df_test[col_class][mask_fire_pred].values.reshape(-1,1), df_test['nbsinister'][mask_fire_pred].values.reshape(-1,1), 'fire_or_pred_gt', dir_output / name), 2)
    metrics['SS_no_zeros_gt'] = silhouette_score_no_zeros
    print(f'SS_no_zero_gts = {silhouette_score_no_zeros}')"""

    y_true = df_test[target_name].values

    if pred_max is not None:
        y_pred_max = pred_max[:, 1]
        y_pred_min = pred_min[:, 1]
    else:
        y_pred_max = None
        y_pred_min = None

    all_class = np.unique(np.concatenate((df_test[col_class].values, y_pred)))
    all_class = all_class[~np.isnan(all_class)]
    all_class_label = [int(c) for c in all_class]

    #plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize='true', filename=f'{scale}_confusion_matrix')
    #plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize='all', filename=f'{scale}_confusion_matrix')
    #plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize='pred', filename=f'{scale}_confusion_matrix')
    #plot_custom_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, dir_output=dir_output / name, figsize=(15,8), normalize=None, filename=f'{scale}_confusion_matrix')
    #if name.find('classification') != -1:

        #plot_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, name, dir_output / name, normalize='true')
        #plot_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, name, dir_output / name, normalize='all')
        #plot_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, name, dir_output / name, normalize='pred')
        #plot_confusion_matrix(df_test[col_class].values, y_pred, all_class_label, name, dir_output / name, normalize=None)

        #accuracy = round(accuracy_score(y_true, y_pred), 2)
        #metrics[f'accuracy'] = accuracy
        #print(f'accuracy = {accuracy}')
    
    y_pred = np.round(y_pred).astype(int)
    
    mask_unknowed_sample = (df_test[col_class_1] == 0) & (df_test[col_class_2] > 0)
    metrics['unknow_sample_proportion'] = y_true[mask_unknowed_sample].shape[0] / y_true.shape[0]

    iou_dict = calculate_signal_scores(y_pred, y_true, df_test['nbsinister'].values, df_test['graph_id'].values, df_test['saison'].values)

    # Sauvegarder toutes les métriques calculées dans le dictionnaire metrics
    for key, value in iou_dict.items():
        metric_key = f'{key}_class_hard' # Ajouter un suffixe basé sur col_for_dict
        metrics[metric_key] = round(value, 3)  # Ajouter au dictionnaire des métriques
        if key == 'iou' or key == 'bad_prediction' or key == 'iou_wildfire_detected':
            print(f'{metric_key} = {round(value, 3)}')  # Afficher la métrique enregistrée

    iou_dict = calculate_signal_scores(y_pred, df_test[col_class_2].values, df_test['nbsinister'].values, df_test['graph_id'].values, df_test['saison'].values)
    for key, value in iou_dict.items():
        metric_key = f'{key}_risk'  # Ajouter un suffixe basé sur col_for_dict
        metrics[metric_key] = round(value, 3)  # Ajouter au dictionnaire des métriques        
        if key == 'iou' or key == 'bad_prediction' or key == 'iou_wildfire_detected':
            print(f'{metric_key} = {round(value, 3)}')  # Afficher la métrique enregistrée

    y_pred_ez = np.copy(y_pred)
    y_pred_ez = np.round(y_pred_ez).astype(int)
    y_pred_ez[mask_unknowed_sample] = 0
    iou_dict = calculate_signal_scores(y_pred_ez, y_true, df_test['nbsinister'].values, df_test['graph_id'].values, df_test['saison'].values)

    # Sauvegarder toutes les métriques calculées dans le dictionnaire metrics
    for key, value in iou_dict.items():
        metric_key = f'{key}_class_ez'  # Ajouter un suffixe basé sur col_for_dict
        metrics[metric_key] = round(value, 3)  # Ajouter au dictionnaire des métriques        
        if key == 'iou' or key == 'bad_prediction' or key == 'iou_wildfire_detected':
            print(f'{metric_key} = {round(value, 3)}')  # Afficher la métrique enregistrée

    for cl in np.unique(y_true):
        print(f'{cl} -> {y_true[y_true == cl].shape[0]}, {y_pred[y_pred == cl].shape[0]}')
        metrics[f'{cl}_true'] = y_true[y_true == y_true].shape[0]
        metrics[f'{cl}_pred'] = y_pred[y_pred == y_pred].shape[0]

    """y_true_temp = np.ones((y_pred.shape[0], y.shape[1]))
    y_true_temp[:, graph_id_index] = df_test['graph_id']
    y_true_temp[:, id_index] = df_test['graph_id']
    y_true_temp[:, departement_index] = df_test['departement']
    y_true_temp[:, date_index] = df_test['date']
    y_true_temp[:, -1] = y_true
    y_true_temp[:, -2] = df_test['nbsinister']
    y_true_temp[:, -3] = df_test[col_class]

    realVspredict(y_pred, y_true_temp, -1,
        dir_output / name, f'{col_class}_{scale}_hard',
        y_pred_min, y_pred_max)

    res[f'prediction_{col_class}'] = y_pred

    iou_vis(y_pred, y_true_temp, -1, dir_output / name, f'{col_class}_{scale}_hard')

    y_true_temp = np.ones((y_pred.shape[0], y.shape[1]))
    y_true_temp[:, graph_id_index] = df_test['graph_id']
    y_true_temp[:, id_index] = df_test['graph_id']
    y_true_temp[:, departement_index] = df_test['departement']
    y_true_temp[:, date_index] = df_test['date']
    y_true_temp[:, -1] = y_true
    y_true_temp[:, -2] = df_test['nbsinister']
    y_true_temp[:, -3] = df_test[col_class]

    iou_vis(y_pred_ez, y_true_temp, -1, dir_output / name, f'{col_class}_{scale}_ez')

    y_true_temp = np.ones((y_pred.shape[0], y.shape[1]))
    y_true_temp[:, graph_id_index] = df_test['graph_id']
    y_true_temp[:, id_index] = df_test['graph_id']
    y_true_temp[:, departement_index] = df_test['departement']
    y_true_temp[:, date_index] = df_test['date']
    y_true_temp[:, -1] = df_test[col_class_2]
    y_true_temp[:, -2] = df_test['nbsinister']
    y_true_temp[:, -3] = df_test[col_class]

    iou_vis(y_pred, y_true_temp, -1, dir_output / name, f'{col_class}_{scale}_risk')

    metrics['nbsinister'] = res['nbsinister'].sum()"""

    return metrics, res

def plot_confusion_matrix(y_test, y_pred, labels, model_name, dir_output, figsize=(10, 8), title='Confusion Matrix', filename='confusion_matrix', normalize=None):
    """
    Plots a confusion matrix with annotations and proper formatting.

    Parameters:
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (list): List of label names for the confusion matrix.
    model_name (str): Name of the model for the plot title.

    Returns:
    None
    """
    # Compute confusion matrix with normalization
    conf_matrix = confusion_matrix(y_test, y_pred, normalize=normalize)

    # Convert confusion matrix to DataFrame for better visualization
    cm_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    # Plotting the heatmap
    fig = plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt=".3f", xticklabels=labels, yticklabels=labels)
    
    # Adding titles and labels
    plt.title("Confusion Matrix for: " + model_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Ensure the output directory exists
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    # Save the image to the specified directory
    output_path = Path(dir_output) / f'{filename}_{normalize}.png'
    plt.savefig(output_path)
    
    # If mlflow is enabled, log the figure
    #if MLFLOW:
    #    mlflow.log_figure(fig, str(output_path))
        
    plt.close(fig)

def calculate_ic95(data):
    """
    Function to calculate the 95% confidence interval (IC95) for a given dataset.
    
    Parameters:
    data (array-like): Array of data points (e.g., model performance scores).

    Returns:
    tuple: lower bound and upper bound of the 95% confidence interval.
    """
    # Convert data to numpy array for convenience
    data = np.array(data)
    
    # Calculate the mean and standard error
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    
    # Calculate the 95% confidence interval using 1.96 for a 95% confidence level
    ci_lower = mean - 1.96 * std_err
    ci_upper = mean + 1.96 * std_err
    
    return ci_lower, ci_upper

def calculate_area_under_curve(y_values):
    """
    Calcule l'aire sous la courbe pour une série de valeurs données (méthode de trapèze).

    :param y_values: Valeurs sur l'axe des ordonnées pour calculer l'aire sous la courbe.
    :return: Aire sous la courbe.
    """
    return np.trapz(y_values, dx=1)

def evaluate_metrics(df, y_true_col='target', y_pred=None):
    """
    Calcule l'IoU et le F1-score sur chaque département, puis calcule l'aire sous la courbe normalisée (aire / aire maximale).
    
    :param dff: DataFrame contenant les colonnes ['Department', 'Scale', 'nbsinister', 'target']
    :param dataset: Nom du dataset à filtrer
    :param y_true_col: Colonne représentant les cibles réelles
    :param y_pred: Liste ou tableau des prédictions
    :param metric: Choix de la métrique ('IoU' ou 'F1')
    :param top: Nombre de départements à afficher (ou 'all' pour tout afficher)
    :return: Dictionnaire contenant l'aire normalisée pour chaque modèle.
    """
    
    # Trier les valeurs par 'nbsinister' décroissant
    #df_sorted = df.sort_values(by='nbsinister', ascending=False)
    df_sorted = df

    y_true = df[y_true_col]
    
    iou = iou_score(y_true, y_pred)
    f1 = f1_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
    prec = precision_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
    rec = recall_score((y_true > 0).astype(int), (y_pred > 0).astype(int))

    under = under_prediction_score(y_true, y_pred)
    over = over_prediction_score(y_true, y_pred)

    # Initialiser un dictionnaire pour les résultats
    results = {'iou' : iou, 'f1' : f1, 'under' : under, 'over' : over, 'prec' : prec, 'recall' : rec}

    # Calculer l'IoU et F1 pour chaque département
    IoU_scores = []
    F1_scores = []
    
    for i, department in enumerate(df_sorted['departement'].unique()):
        # Extraire les valeurs pour chaque département
        y_true = df_sorted[df_sorted['departement'] == department][y_true_col].values
        if np.all(y_true == 0):
            continue
        y_pred_department = y_pred[df_sorted['departement'] == department]  # Récupérer les prédictions associées au département
        
        # Calcul des scores IoU et F1
        IoU = iou_score(y_true, y_pred_department)
        F1 = f1_score(y_true > 0, y_pred_department > 0)
        
        IoU_scores.append(IoU)
        F1_scores.append(F1)
        
    df_sorted_test_area = df_sorted[df_sorted[y_true_col] > 0]
    # Calcul de l'aire maximale possible (cas parfait où toutes les prédictions sont correctes)
    max_area = np.trapz(np.ones(len(df_sorted_test_area['departement'].unique())), dx=1)
    
    # Calcul de l'aire sous la courbe pour l'IoU et le F1
    IoU_area = calculate_area_under_curve(IoU_scores)
    F1_area = calculate_area_under_curve(F1_scores)
    
    # Normalisation par l'aire maximale
    normalized_IoU = IoU_area / max_area if max_area > 0 else 0
    normalized_F1 = F1_area / max_area if max_area > 0 else 0
    
    # Stocker les résultats dans le dictionnaire
    results['normalized_iou'] = normalized_IoU
    results['normalized_f1'] = normalized_F1
    
    return results