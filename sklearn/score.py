from random import sample
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from xgboost import DMatrix
from sklearn.metrics import log_loss

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