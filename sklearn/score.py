from random import sample
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import DMatrix

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

def calculate_signal_scores(y_true, y_pred):
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

    y_pred = np.copy(y_pred)

    #thresholds_ks = np.sort(np.unique(test_window['nbsinister'])[1:])
    #df_ks, optimal_thresholds = calculate_ks(test_window, 'prediction', col, thresholds_ks, dir_output / name)

    #new_pred = np.full(y_pred.shape[0], fill_value=np.nan)

    #new_pred[y_pred < optimal_thresholds[thresholds_ks[0]]['optimal_score']] = 0.0
    #max_value = 0
    #for threshold in thresholds_ks:
    #    if optimal_thresholds[threshold]['ks_stat'] > max_value:
    #        max_value = optimal_thresholds[threshold]['ks_stat']
    #        new_pred[(y_pred >= optimal_thresholds[threshold]['optimal_score'])] = threshold

    #y_pred = new_pred

    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    under_prediction = np.trapz(np.maximum(0, y_true - y_pred))
    over_prediction = np.trapz(np.maximum(0, y_pred - y_true))

    over_prediction_zeros = np.trapz(np.maximum(0, y_pred[y_true == 0]))
    under_prediction_zeros = np.trapz(np.maximum(0, y_true[y_pred == 0]))

    under_prediction_fire = np.trapz(np.maximum(0, y_true[y_true > 0] - y_pred[y_true > 0]))
    over_prediction_fire = np.trapz(np.maximum(0, y_pred[y_true > 0] - y_true[y_true > 0]))

    only_true_value = np.trapz(y_true) # Air sous la courbe des prédictions
    only_pred_value = np.trapz(y_pred) # Air sous la courbe des prédictions

    # Enregistrement dans un dictionnaire
    scores = {
        "common_area": intersection,
        "union_area": union,
        "under_predicted_area": under_prediction,
        "over_predicted_area": over_prediction,

        "iou": round(intersection / union, 2) if union > 0 else 0,  # To avoid division by zero
        "iou_under_prediction": round(under_prediction / union, 2) if union > 0 else 0,
        "iou_over_prediction": round(over_prediction / union, 2) if (union) > 0 else 0,

        "wildfire_predicted" : round(intersection / only_true_value, 2) if only_true_value > 0 else 0,
        "wildfire_supposed" : round(intersection / only_pred_value, 2) if only_true_value > 0 else 0,
        
        "wildfire_over_predicted": round(over_prediction_fire / union, 2) if only_true_value > 0 else 0,
        "wildfire_under_predicted": round(under_prediction_fire / union, 2) if only_true_value > 0 else 0,
        
        "over_bad_prediction" : round(over_prediction_zeros / union, 2) if union > 0 else 0,
        "under_bad_prediction" : round(under_prediction_zeros / union, 2) if union > 0 else 0,
        "bad_prediction" : round((over_prediction_zeros + under_prediction_zeros) / union, 2) if union > 0 else 0,
    }

    # Binarisation des prédictions et vérités terrain
    # Binarisation des prédictions et vérités terrain
    y_pred_day = (y_pred > 0).astype(int)  # Prédictions binarisées
    y_true_day = (y_true > 0).astype(int)  # Vérités binarisées

    # Calcul des différentes aires pour les données binaires
    intersection_day = np.trapz(np.minimum(y_pred_day, y_true_day))  # Aire commune binaire
    union_day = np.trapz(np.maximum(y_pred_day, y_true_day))         # Aire d'union binaire

    under_prediction_day = np.trapz(np.maximum(0, y_true_day - y_pred_day))
    over_prediction_day = np.trapz(np.maximum(0, y_pred_day - y_true_day))

    over_prediction_zeros_day = np.trapz(np.maximum(0, y_pred_day[y_true_day == 0]))
    under_prediction_zeros_day = np.trapz(np.maximum(0, y_true_day[y_pred_day == 0]))

    under_prediction_fire_day = np.trapz(np.maximum(0, y_true_day[y_true_day > 0] - y_pred_day[y_true_day > 0]))
    over_prediction_fire_day = np.trapz(np.maximum(0, y_pred_day[y_true_day > 0] - y_true_day[y_true_day > 0]))

    only_true_value_day = np.trapz(y_true_day)  # Aire sous la courbe des vérités terrain binaires
    only_pred_value_day = np.trapz(y_pred_day)  # Aire sous la courbe des prédictions binaires

    # Enregistrement des scores pour les données binaires
    scores.update({
        "common_area_day": intersection_day,
        "union_area_day": union_day,
        "under_predicted_area_day": under_prediction_day,
        "over_predicted_area_day": over_prediction_day,

        "iou_day": round(intersection_day / union_day, 2) if union_day > 0 else 0,  # IoU binaire
        "iou_under_prediction_day": round(under_prediction_day / union_day, 2) if union_day > 0 else 0,
        "iou_over_prediction_day": round(over_prediction_day / union_day, 2) if union_day > 0 else 0,

        "wildfire_predicted_day": round(intersection_day / only_true_value_day, 2) if only_true_value_day > 0 else 0,
        "wildfire_supposed_day": round(intersection_day / only_true_value_day, 2) if only_true_value_day > 0 else 0,

        "wildfire_over_predicted_day": round(over_prediction_fire_day / only_true_value_day, 2) if only_true_value_day > 0 else 0,
        "wildfire_under_predicted_day": round(under_prediction_fire_day / only_true_value_day, 2) if only_true_value_day > 0 else 0,

        "over_bad_prediction_day": round(over_prediction_zeros_day / union_day, 2) if union_day > 0 else 0,
        "under_bad_prediction_day": round(under_prediction_zeros_day / union_day, 2) if union_day > 0 else 0,
        "bad_prediction_day": round((over_prediction_zeros_day + under_prediction_zeros_day) / union_day, 2) if union_day > 0 else 0,
    })


    return scores['iou']

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