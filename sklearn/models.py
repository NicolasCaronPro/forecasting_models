import xgboost as xgb
import catboost
from catboost import CatBoostClassifier, CatBoostRegressor

from forecasting_models.pytorch.tools_2 import *

from lightgbm import LGBMClassifier, LGBMRegressor, plot_tree as lgb_plot_tree

from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Normal, ClassificationDistn, k_categorical
from ngboost.scores import LogScore
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from xgboost import XGBClassifier, XGBRegressor, plot_tree as xgb_plot_tree

from statsmodels.miscmodels.ordinal_model import OrderedModel
from forecasting_models.sklearn.score import *

##########################################################################################

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
import shap
import os
import matplotlib.pyplot as plt

# Définir ou importer les fonctions de perte personnalisées si nécessaire
def weighted_mse_loss(y_true, y_pred):
    # Placeholder pour la perte MSE pondérée
    pass

def poisson_loss(y_true, y_pred):
    # Placeholder pour la perte de Poisson
    pass

def smooth_area_under_prediction_loss(y_true, y_pred):
    # Placeholder pour la perte d'aire sous la courbe
    pass

def exponential_absolute_error_loss(alpha):
    # Placeholder pour la perte exponentielle personnalisée
    pass

def sigmoid_adjusted_loss(y_true, y_pred):
    # Placeholder pour la perte ajustée sigmoid
    pass

def sigmoid_adjusted_loss_adapted(y_true, y_pred):
    # Placeholder pour la perte ajustée sigmoid adaptée
    pass


class MyXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, **kwargs):
        """
        Classe personnalisée pour l'entraînement de modèles XGBoost avec des objectifs standards ou personnalisés.
        :param alpha: Hyperparamètre (non utilisé directement, optionnel).
        :param kwargs: Paramètres supplémentaires pour XGBRegressor.
        """
        self.alpha = alpha
        self.kwargs = kwargs
        
        # Paramètres pour l'entraînement
        params = self.kwargs.copy()
        loss = params.get('objective', None)
        self.loss = loss
        
        self.is_gpu = False
        if self.is_gpu:
            params['predictor'] = 'gpu_predictor'

        # Détection du type de perte (objective)
        if loss == 'mse':
            params['objective'] = 'reg:squarederror'
        
        elif loss == 'rmse':
            params['objective'] = 'reg:squarederror'

        elif loss == 'poisson':
            params['objective'] = 'count:poisson'

        elif loss == 'log':
            params['objective'] = 'reg:logistic'

        elif loss == 'exponential':
            params['objective'] = exponential_absolute_error_loss(self.alpha)

        elif loss == 'area':
            params['objective'] = smooth_area_under_prediction_loss

        elif loss == 'sig':
            params['objective'] = sigmoid_adjusted_loss

        elif loss == 'sig2':
            params['objective'] = sigmoid_adjusted_loss_adapted

        elif loss == 'gpd':
            params['objective'] = gpd_gradient_hessian
            params['eval_metric'] = gpd_multiclass_loss

        elif 'risk' in loss:
            params['objective'] = 'reg:gamma'

        elif loss == 'dgpd':
            params['objective'] = dgpd_gradient_hessian
            params['eval_metric'] = dgpd_loss
        
        elif loss == 'egpd':
            params['objective'] = egpd_gradient_hessian
            params['eval_metric'] = egpd_loss

        else:
            raise ValueError(f'Unknow {loss} function')
        
        # Initialisation du modèle XGBRegressor
        self.model_ = xgb.XGBRegressor(**params)
        if self.is_gpu:
            self.model_.set_params(predictor='cpu_predictor')

    def update_params(self, kwargs):
        """
        Met à jour les paramètres de l'entraînement avec de nouveaux arguments.
        :param kwargs: Les nouveaux paramètres d'entraînement.
        """
        self.kwargs = kwargs
        params = self.kwargs.copy()
        loss = params.get('objective', None)
        self.loss = loss
        
        self.is_gpu = False
        if self.is_gpu:
            params['predictor'] = 'gpu_predictor'

        # Détection du type de perte (objective)
        if loss == 'mse':
            params['objective'] = 'reg:squarederror'
        
        elif loss == 'rmse':
            params['objective'] = 'reg:squarederror'

        elif loss == 'poisson':
            params['objective'] = 'count:poisson'

        elif loss == 'log':
            params['objective'] = 'reg:logistic'

        elif loss == 'mcewk':
            params['objective'] = mcewk_obj

        elif loss == 'exponential':
            params['objective'] = exponential_absolute_error_loss(self.alpha)

        elif loss == 'area':
            params['objective'] = smooth_area_under_prediction_loss

        elif loss == 'sig':
            params['objective'] = sigmoid_adjusted_loss

        elif loss == 'sig2':
            params['objective'] = sigmoid_adjusted_loss_adapted

        elif loss == 'gpd':
            params['objective'] = gpd_gradient_hessian
            params['eval_metric'] = gpd_multiclass_loss

        elif loss == 'egpd':
            params['objective'] = egpd_gradient_hessian
            params['eval_metric'] = egpd_loss

        elif 'risk' in loss:
            params['objective'] = 'reg:gamma'

        elif loss == 'dgpd':
            params['objective'] = dgpd_gradient_hessian
            params['eval_metric'] = dgpd_loss

        else:
            raise ValueError(f'Unknow {loss} function')
        
        # Initialisation du modèle XGBRegressor
        self.model_ = xgb.XGBRegressor(**params)
        if self.is_gpu:
            self.model_.set_params(predictor='cpu_predictor')

    def fit(self, X, y, **fit_params):
        """
        Entraîne le modèle XGBoost sur les données d'entrée.
        :param X: Features d'entrée.
        :param y: Labels de sortie.
        :param fit_params: Paramètres supplémentaires pour l'entraînement (ex: eval_set, early_stopping_rounds).
        """
        # Entraînement du modèle XGBRegressor
        #self.model_.set_params(early_stopping_rounds=fit_params.get('early_stopping_rounds', None))
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        """
        Prédit les valeurs de sortie pour les données d'entrée.
        :param X: Features d'entrée.
        :return: Valeurs prédites.
        """
        return self.model_.predict(X)

    def get_booster(self):
        """
        Retourne l'objet booster du modèle entraîné.
        :return: Booster XGBoost.
        """
        return self.model_.get_booster()

    def shapley_additive_explanation(self, df_set, outname, dir_output, mode='bar', figsize=(50, 25), samples=None, samples_name=None):
        """
        Visualisation des valeurs SHAP pour expliquer les prédictions.
        :param df_set: DataFrame des caractéristiques d'entrée.
        :param outname: Nom de sortie pour le fichier d'image.
        :param dir_output: Répertoire où enregistrer les résultats.
        :param mode: Mode de visualisation ('bar' ou 'beeswarm').
        :param figsize: Taille de la figure.
        :param samples: Échantillons spécifiques à analyser.
        :param samples_name: Noms des échantillons à afficher.
        """
        # Calcul des valeurs SHAP
        explainer = shap.Explainer(self.model_)
        shap_values = explainer(df_set)

        # Créer le répertoire de sortie
        os.makedirs(dir_output, exist_ok=True)

        df = pd.DataFrame({
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0), 
            "stdev_abs_shap": np.std(np.abs(shap_values), axis=0), 
            "name": df.columns
        })
        df.sort_values("mean_abs_shap", ascending=False)
        save_object(df, dir_output)

        # Sauvegarder la visualisation globale
        plt.figure(figsize=figsize)
        if mode == 'bar':
            shap.summary_plot(shap_values, df_set, plot_type='bar', show=False)
        elif mode == 'beeswarm':
            shap.summary_plot(shap_values, df_set, show=False)
        plt.savefig(os.path.join(dir_output, f"{outname}_shapley_additive_explanation.png"))
        plt.close()

        # Sauvegarder les visualisations spécifiques aux échantillons
        if samples is not None and samples_name is not None:
            sample_dir = os.path.join(dir_output, 'sample')
            os.makedirs(sample_dir, exist_ok=True)
            for i, sample in enumerate(samples):
                plt.figure(figsize=figsize)
                shap.force_plot(
                    shap_values[sample], df_set.iloc[sample],
                    matplotlib=True,
                ).savefig(
                    os.path.join(sample_dir, f"{outname}_{samples_name[i]}_shapley_additive_explanation.png")
                )
                plt.close()

class MyXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, **kwargs):
        """
        Classe personnalisée pour l'entraînement de modèles XGBoost avec des objectifs standards ou personnalisés.
        :param alpha: Hyperparamètre (non utilisé directement, optionnel).
        :param kwargs: Paramètres supplémentaires pour XGBClassifier.
        """
        self.alpha = alpha
        self.kwargs = kwargs
        # Paramètres pour l'entraînement
        params = self.kwargs.copy()
        loss = params.get('objective', None)
        self.loss = loss
        
        self.is_gpu = False
        if self.is_gpu:
            params['predictor']='gpu_predictor'
        
        # Détection du type de perte (objective)
        if loss == 'logloss':
            params['objective'] = 'binary:logistic'

        elif loss in ['softmax', 'softprob']:

            params['objective'] = f"multi:{loss}"
            params['eval_metric'] = 'mlogloss'
            num_classes = 5
            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                 f"but found {num_classes} unique classes in y.")
        
        elif loss == 'softprob-dual' or loss == 'softmax-dual':
            params['objective'] = softprob_obj_dual(params['y_train_origin'])
            params['eval_metric'] = 'mlogloss'
            params['num_class'] = 5
            del params['y_train_origin']
            del params['y_val_origin']

        elif loss == 'mcewk':
            params['objective'] = mcewk_obj
            params['eval_metric'] = 'mlogloss'
        
        elif loss == 'softprob-risk-dual' or loss == 'softmax-risk-dual':
            params['objective'] = softprob_obj_risk_dual(params['y_train_origin'])
            params['eval_metric'] = 'mlogloss'
            params['num_class'] = 5
            del params['y_train_origin']
            del params['y_val_origin']

        elif 'risk-dual' in loss:
            if 'kappa-1' in loss:
                params['objective'] = softprob_obj_risk_dual(params['y_train_origin'], 1)
            elif 'kappa-2' in loss:
                params['objective'] = softprob_obj_risk_dual(params['y_train_origin'], 2)
            else:
                params['objective'] = softprob_obj_risk_dual(params['y_train_origin'])

            params['eval_metric'] = 'mlogloss'
            params['num_class'] = 5
            del params['y_train_origin']
            del params['y_val_origin']
        
        elif loss == 'gpd':
            params['objective'] = gpd_gradient_hessian
            params['eval_metric'] = gpd_multiclass_loss

        elif 'risk' in loss:
            if 'kappa-1' in loss:
                params['objective'] = softprob_obj_risk(1)
            elif 'kappa-2' in loss:
                params['objective'] = softprob_obj_risk(2)
            else:
                params['objective'] = softprob_obj_risk()
            params['eval_metric'] = 'mlogloss'

        elif loss == 'dice':
            num_classes = 5
            params['objective'] = dice_loss_class(num_classes)
            params['eval_metric'] = 'mlogloss'

            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                 f"but found {num_classes} unique classes in y.")
        
        elif loss == 'weighted':
            params['objective'] = weighted_class_loss_objective
            params['eval_metric'] = 'mlogloss'
            num_classes = 5

            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                 f"but found {num_classes} unique classes in y.")

        else:
            raise ValueError(f'Unknow {loss} function')

        # Initialisation du modèle XGBClassifier
        self.model_ = XGBClassifier(**params)
        if self.is_gpu:
            self.model_.set_params(predictor='cpu_predictor')

    def update_params(self, kwargs):
        self.kwargs = kwargs
        # Paramètres pour l'entraînement
        params = self.kwargs.copy()
        loss = params.get('objective', None)
        self.loss = loss
        
        self.is_gpu = False
        if self.is_gpu:
            params['predictor']='gpu_predictor'
        
        # Détection du type de perte (objective)
        if loss == 'logloss':
            params['objective'] = 'binary:logistic'
        
        elif loss == 'mcewk':
            params['objective'] = 'mcewk_class'
            params['eval_metric'] = 'mlogloss'

        elif loss in ['softmax', 'softprob']:

            params['objective'] = f"multi:{loss}"
            params['eval_metric'] = 'mlogloss'
            num_classes = 5
            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                 f"but found {num_classes} unique classes in y.")
        
        elif loss == 'softprob-dual' or loss == 'softmax-dual':
            params['objective'] = softprob_obj_dual(params['y_train_origin'])
            params['eval_metric'] = 'mlogloss'
            #params['eval_metric'] = LogLossDual(params['y_val_origin'])
            params['num_class'] = 5
            del params['y_train_origin']
            del params['y_val_origin']

        elif 'risk-dual' in loss:
            if 'kappa-1' in loss:
                params['objective'] = softprob_obj_risk_dual(params['y_train_origin'], 1)
            elif 'kappa-2' in loss:
                params['objective'] = softprob_obj_risk_dual(params['y_train_origin'], 2)
            else:
                params['objective'] = softprob_obj_risk_dual(params['y_train_origin'])
            params['eval_metric'] = 'mlogloss'
            params['num_class'] = 5
            del params['y_train_origin']
            del params['y_val_origin']

        elif 'risk' in loss:
            if 'kappa-1' in loss:
                params['objective'] = softprob_obj_risk(1)
            elif 'kappa-2' in loss:
                params['objective'] = softprob_obj_risk(2)
            else:
                params['objective'] = softprob_obj_risk()
            params['eval_metric'] = 'mlogloss'

        elif loss == 'dice':

            num_classes = 5
            params['objective'] = dice_loss_class(num_classes) 
            params['eval_metric'] = 'mlogloss'

            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                 f"but found {num_classes} unique classes in y.")
        
        elif loss == 'weighted':
            params['objective'] = weighted_class_loss_objective
            params['eval_metric'] = 'mlogloss'
            num_classes = 5

            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                 f"but found {num_classes} unique classes in y.")

        elif loss == 'gpd':
            params['objective'] = gpd_gradient_hessian
            params['eval_metric'] = gpd_multiclass_loss
        
        elif loss == 'dgpd':
            params['objective'] = gpd_gradient_hessian
            params['eval_metric'] = gpd_multiclass_loss

        else:
            raise ValueError(f'Unknow {loss} function')

        # Initialisation du modèle XGBClassifier
        self.model_ = XGBClassifier(**params)
        if self.is_gpu:
            self.model_.set_params(predictor='cpu_predictor')

    def fit(self, X, y, **fit_params):
        """
        Entraîne le modèle XGBoost sur les données d'entrée.
        :param X: Features d'entrée.
        :param y: Labels de sortie.
        :param fit_params: Paramètres supplémentaires pour l'entraînement (ex: sample_weight, eval_set).
        """
        # Convertir les labels en entiers
        y = y.astype(int)
        #self.model_.set_params(early_stopping_rounds=fit_params.get('early_stopping_rounds'))
        #del fit_params['early_stopping_rounds']

        if self.is_gpu:
            X = cp.array(X)
            y = cp.array(y)

        """if self.loss in ['softprob-dual', 'softmax-dual']:
            #loss = softprob_obj_dual(y[:, 1])
            #self.model_.set_params(obective=loss)
            #self.model_.set_params(eval_metric=LogLossDual(fit_params['eval_set'][0][1]))

            y = y[:, 0]"""

        self.model_.fit(
                X, y, 
                **fit_params
            )
        return self

    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée.
        :param X: Features d'entrée.
        :return: Labels prédits.
        """
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque classe.
        :param X: Features d'entrée.
        :return: Matrice de probabilités (lignes: échantillons, colonnes: classes).
        """
        return self.model_.predict_proba(X)

    def get_booster(self):
        """
        Retourne l'objet booster du modèle entraîné.
        :return: Booster XGBoost.
        """
        if self.model_ is not None:
            return self.model_.get_booster()
        raise ValueError("Model not trained yet.")

    def shapley_additive_explanation(self, df_set, outname, dir_output, mode='bar', figsize=(50, 25), samples=None, samples_name=None):
        if not hasattr(self.model_, 'get_booster'):
            raise ValueError("Le modèle doit être un XGBClassifier entraîné.")

        # Calcul des valeurs SHAP
        explainer = shap.Explainer(self.model_)
        shap_values = explainer(df_set)

        # Créer le répertoire de sortie
        os.makedirs(dir_output, exist_ok=True)

        # Sauvegarder la visualisation globale
        plt.figure(figsize=figsize)
        if mode == 'bar':
            shap.summary_plot(shap_values, df_set, plot_type='bar', show=False)
        elif mode == 'beeswarm':
            shap.summary_plot(shap_values, df_set, show=False)
        plt.savefig(os.path.join(dir_output, f"{outname}_shapley_additive_explanation.png"))
        plt.close()

        # Sauvegarder les visualisations spécifiques aux échantillons
        if samples is not None and samples_name is not None:
            sample_dir = os.path.join(dir_output, 'sample')
            os.makedirs(sample_dir, exist_ok=True)
            for i, sample in enumerate(samples):
                plt.figure(figsize=figsize)
                shap.force_plot(
                    shap_values[sample], df_set.iloc[sample],
                    matplotlib=True,
                ).savefig(
                    os.path.join(sample_dir, f"{outname}_{samples_name[i]}_shapley_additive_explanation.png")
                )
                plt.close()

"""import tkinter as tk

def safe_gui_call():
    root = tk.Tk()
    label = tk.Label(root, text="Training Started")
    label.pack()
    root.mainloop()
    root.after(5, root.destroy)

import threading
threading.Thread(target=safe_gui_call).start()"""

class MyCatBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, **kwargs):
        """
        Custom CatBoost Classifier that supports standard and custom loss functions.
        :param alpha: Unused hyperparameter (for consistency with MyXGBClassifier).
        :param kwargs: Additional parameters for CatBoostClassifier.
        """
        self.alpha = alpha
        self.kwargs = kwargs.copy()
        self.loss = self.kwargs.get("loss_function", None)
        params = self.kwargs.copy()

        # Detect and adjust loss functions
        if self.loss == "logloss":
            params["loss_function"] = "Logloss"
        
        elif self.loss == 'mcewk':
            params['loss_function'] = mcewk_class

        elif self.loss in ["softmax", "softprob"]:
            params["loss_function"] = "MultiClass"

        elif self.loss == "softprob-dual" or self.loss == "softmax-dual":
            params["loss_function"] = softprob_obj_dual(params['y_train_origin'])  # Custom function
            del params['y_train_origin']
            del params['y_val_origin']
            #raise NotImplementedError("Custom dual loss functions need manual implementation.")

        elif self.loss == "dice":
            params["loss_function"] = dice_loss_class  # Custom function

        elif self.loss == "weighted":
            params["loss_function"] = weighted_class_loss_objective  # Custom function

        # Initialize CatBoostClassifier
        self.model_ = CatBoostClassifier(**params)

    def update_params(self, kwargs):
        num_class = 5
        self.kwargs = kwargs.copy()
        params = kwargs.copy()

        # Detect and adjust loss functions
        if self.loss == "logloss":
            params["loss_function"] = "Logloss"
        
        elif self.loss == 'mcewk':
            params['loss_function'] = mcewk_class

        elif self.loss in ["softmax", "softprob"]:
            params["loss_function"] = "MultiClass"

        elif self.loss == "softprob-dual" or self.loss == "softmax-dual":
            params["loss_function"] = softprob_obj_dual(params['y_train_origin'])  # Custom function
            del params['y_train_origin']
            del params['y_val_origin']
            #raise NotImplementedError("Custom dual loss functions need manual implementation.")

        elif self.loss == "dice":
            params["loss_function"] = dice_loss_class(num_classes=num_class)  # Custom function

        elif self.loss == "weighted":
            params["loss_function"] = weighted_class_loss_objective  # Custom function

        self.model_ = CatBoostClassifier(**params)

    def fit(self, X, y, **fit_params):
        """
        Train the CatBoost model on input data.
        :param X: Features.
        :param y: Labels.
        :param fit_params: Additional parameters (ex: sample_weight, eval_set).
        """
        y = y.astype(int)

        #if self.loss in ["softprob-dual", "softmax-dual"]:
        #    y = y[:, 0]  # Adjust labels for dual loss

        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        """
        Predict class labels.
        :param X: Input features.
        :return: Predicted class labels.
        """
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Return probability estimates for each class.
        :param X: Input features.
        :return: Probability matrix (rows = samples, columns = class probabilities).
        """
        return self.model_.predict_proba(X)

    def get_model(self):
        """
        Returns the trained CatBoost model.
        :return: CatBoostClassifier model.
        """
        if self.model_ is not None:
            return self.model_
        raise ValueError("Model not trained yet.")

    def shapley_additive_explanation(self, df_set, outname, dir_output, mode="bar", figsize=(50, 25), samples=None, samples_name=None):
        """
        Generate SHAP explanations for model predictions.
        :param df_set: Dataset to analyze.
        :param outname: Output file prefix.
        :param dir_output: Directory to save explanations.
        :param mode: SHAP visualization type ("bar" or "beeswarm").
        :param figsize: Figure size.
        :param samples: Specific samples to analyze.
        :param samples_name: Names of the samples.
        """
        if not hasattr(self.model_, "get_feature_importance"):
            raise ValueError("The model must be a trained CatBoostClassifier.")

        # Compute SHAP values
        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer(df_set)

        # Create output directory
        os.makedirs(dir_output, exist_ok=True)

        # Save global SHAP visualization
        plt.figure(figsize=figsize)
        if mode == "bar":
            shap.summary_plot(shap_values, df_set, plot_type="bar", show=False)
        elif mode == "beeswarm":
            shap.summary_plot(shap_values, df_set, show=False)
        plt.savefig(os.path.join(dir_output, f"{outname}_shapley_additive_explanation.png"))
        plt.close()

        # Save per-sample visualizations
        if samples is not None and samples_name is not None:
            sample_dir = os.path.join(dir_output, "sample")
            os.makedirs(sample_dir, exist_ok=True)
            for i, sample in enumerate(samples):
                plt.figure(figsize=figsize)
                shap.force_plot(
                    shap_values[sample], df_set.iloc[sample],
                    matplotlib=True,
                ).savefig(
                    os.path.join(sample_dir, f"{outname}_{samples_name[i]}_shapley_additive_explanation.png")
                )
                plt.close()

class MyOrderedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, kwargs):
        """
        Custom wrapper for statsmodels' OrderedModel.
        :param kwargs: Additional parameters for OrderedModel.
        """
        self.kwargs = kwargs.copy()
        self.is_fit = False

    def fit(self, X, y, **fit_params):
        """
        Train the OrderedModel.
        :param X: Features.
        :param y: Labels.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.is_fit = True

        self.model_ = OrderedModel(y, X, **self.kwargs)
        self.model_ = self.model_.fit(**fit_params)
        return self

    def predict(self, X):
        """
        Predict class labels.
        :param X: Input features.
        :return: Predicted class labels.
        """
        if not self.is_fit:
            raise ValueError("Model not trained yet.")
        
        res = self.model_.predict(X)
        return np.argmax(res, axis=1)

    def predict_proba(self, X):
        """
        Return probability estimates for each class.
        :param X: Input features.
        :return: Probability matrix (rows = samples, columns = class probabilities).
        """
        if not self.is_fit:
            raise ValueError("Model not trained yet.")
        return self.fitted_model_.model.predict(self.fitted_model_.params, exog=X, transform=True)

    def get_model(self):
        """
        Returns the trained OrderedModel.
        :return: OrderedModel instance.
        """
        if not self.is_fit:
            return self.fitted_model_
        raise ValueError("Model not trained yet.")