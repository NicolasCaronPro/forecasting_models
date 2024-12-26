import copy
import logging
import math
from pyclbr import Class
from re import A, sub
import sys
from pathlib import Path
from typing import Union, final

import matplotlib.pyplot as plt
import numpy as np
import shap
from sympy import false
from torch import poisson_nll_loss
import xgboost as xgb
import catboost
from catboost import CatBoostClassifier, CatBoostRegressor

from forecasting_models.pytorch.tools_2 import *

from lightgbm import LGBMClassifier, LGBMRegressor, plot_tree as lgb_plot_tree

from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

from pygam import GAM

import random

from scipy import stats
from forecasting_models.sklearn.score import *
from skopt import BayesSearchCV, Optimizer
from skopt.space import Integer, Real

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hinge_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    r2_score,
    recall_score,
)

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from sklearn.svm import SVC, SVR

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree as sklearn_plot_tree,
)

from sklearn.utils.validation import check_is_fitted

from xgboost import DMatrix, XGBClassifier, XGBRegressor, plot_tree as xgb_plot_tree

#from scripts.probability_distribution import weight

import numpy as np

##########################################################################################

class MyXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs

    def fit(self, X, y, **fit_params):
        # Convertir les données d'entraînement en DMatrix

        sample_weight = fit_params['sample_weight']
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        
        # Paramètres de l'entraînement, y compris ceux passés via kwargs
        params = self.kwargs.copy()

        eval_set = fit_params.get('eval_set', None)  # Utilisation de eval_set si passé
        early_stopping_rounds = fit_params.get('early_stopping_rounds', None)
        num_boost_round = params.get('n_estimators', 10)
        
        loss = params.get('objective', 'mse')
        #if loss == 'rmse':
        #    loss = rmse_loss
        if loss.find('eae') != -1:
            alpha = int(loss.split('-')[-1])
            loss = exponential_absolute_error_loss(alpha)
        elif loss == 'poisson':
            loss = poisson_loss
        elif loss == 'area':
            loss = smooth_area_under_prediction_loss
        elif loss == 'rmsle':
            loss = 'reg:squaredlogerror'
        elif loss == 'mse' or loss == 'rmse':
            loss = weighted_mse_loss
            loss = 'reg:squarederror'
        elif loss == 'sig':
            loss = sigmoid_adjusted_loss
        elif loss == 'sig2':
            loss = sigmoid_adjusted_loss_adapted
        elif loss == 'quantile':
            loss = 'reg:quantileerror'
            alphas = np.asarray([0.05,0.25,0.50,0.75,0.95])

        custom_metric = lambda x, y : ('error_r2', -my_r2_score(x, y))

        params['quantile_alpha'] = alphas if 'alphas' in locals() else None

        if isinstance(loss, str):
            params['objective'] = loss
            self.model_ = xgb.train(
                params=params,
                num_boost_round=num_boost_round,
                dtrain=dtrain,
                evals=eval_set,
                #custom_metric=custom_metric,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )
        else:
            params['objective'] = 'reg:squarederror'
            self.model_ = xgb.train(
                obj=loss,
                params=params,
                num_boost_round=num_boost_round,
                dtrain=dtrain,
                evals=eval_set,
                #custom_metric=custom_metric,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )
        return self

    def predict(self, X):
        # Convertir les données de test en DMatrix
        dtest = xgb.DMatrix(X)
        return self.model_.predict(dtest)
    
    def get_params(self, deep: bool = True) -> dict:
        return self.kwargs
    
    def get_booster(self):
        """
        Retourne l'objet booster (modèle XGBoost complet) après l'entraînement.
        """
        return self.model_

    def shapley_additive_explanation(self, df_set, outname, dir_output, mode='bar', figsize=(50, 25), samples=None, samples_name=None):
        dtrain = xgb.DMatrix(df_set)
        shap_values = self.model_.predict(dtrain, pred_contribs=True)

        # Save global visualization
        os.makedirs(dir_output, exist_ok=True)
        plt.figure(figsize=figsize)
        if mode == 'bar':
            shap.summary_plot(shap_values, df_set, plot_type='bar', show=False)
        elif mode == 'beeswarm':
            shap.summary_plot(shap_values, df_set, show=False)
        plt.savefig(os.path.join(dir_output, f"{outname}_shapley_additive_explanation.png"))
        plt.close()

        # Save specific samples if provided
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
        self.alpha = alpha
        self.kwargs = kwargs

    def fit(self, X, y, **fit_params):
        # Convertir les données d'entraînement en DMatrix
        sample_weight = fit_params.get('sample_weight', None)
        
        y = y.astype(int)

        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        
        # Paramètres de l'entraînement, y compris ceux passés via kwargs
        params = self.kwargs.copy()

        # Appeler la méthode d'entraînement avec fit_params
        eval_set = fit_params.get('eval_set', None)  # Utilisation de eval_set si passé
        early_stopping_rounds = fit_params.get('early_stopping_rounds', None)
        num_boost_round = params.get('n_estimators', 10)
        
        # Adjust the loss/objective parameter
        loss = params.get('objective', None)
        if loss == 'logloss':
            loss = 'binary:logistic'
        elif loss == 'softmax':
            loss = 'multi:softmax'

            # Ensure num_class is set for multi-class objectives
            num_classes = np.nanmax(y) + 1
            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                f"but found {num_classes} unique classes in y.")
        elif loss == 'softprob':
            loss = 'multi:softprob'

            # Ensure num_class is set for multi-class objectives
            num_classes = np.nanmax(y) + 1
            if 'num_class' not in params:
                params['num_class'] = num_classes
            elif params['num_class'] != num_classes:
                raise ValueError(f"Mismatch: num_class in parameters is {params['num_class']} "
                                f"but found {num_classes} unique classes in y.")

        params['objective'] = loss

        if isinstance(loss, str):
            self.model_ = xgb.train(
                params=params,
                num_boost_round=num_boost_round,
                dtrain=dtrain,
                evals=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )
        else:
            params['objective'] = 'binary:logistic'
            self.model_ = xgb.train(
                obj=loss,
                params=params,
                num_boost_round=num_boost_round,
                dtrain=dtrain,
                evals=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )
        return self

    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée en mode multi-classe.
        """
        # Convertir les données de test en DMatrix
        dtest = xgb.DMatrix(X)
        
        # Obtenir les probabilités prédictives
        prob = self.model_.predict(dtest)
        
        # Pour multi-classes, `prob` est une matrice : une ligne par échantillon, une colonne par classe.
        # La classe prédite est celle avec la probabilité maximale.

        return prob

    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque classe en mode multi-classe.
        """
        # Convertir les données de test en DMatrix
        dtest = xgb.DMatrix(X)
        
        # Obtenir les probabilités prédictives
        prob = self.model_.predict(dtest, output_margin=True)
        
        # Pas de modification nécessaire pour multi-classes, `prob` est déjà une matrice de probabilités.
        return prob
    
    def get_booster(self):
        """
        Retourne l'objet booster (modèle XGBoost complet) après l'entraînement.
        """
        return self.model_

    def shapley_additive_explanation(self, df_set, outname, dir_output, mode='bar', figsize=(50, 25), samples=None, samples_name=None):
        dtrain = xgb.DMatrix(df_set)
        shap_values = self.model_.predict(dtrain, pred_contribs=True)

        # Save global visualization
        os.makedirs(dir_output, exist_ok=True)
        plt.figure(figsize=figsize)
        if mode == 'bar':
            shap.summary_plot(shap_values, df_set, plot_type='bar', show=False)
        elif mode == 'beeswarm':
            shap.summary_plot(shap_values, df_set, show=False)
        plt.savefig(os.path.join(dir_output, f"{outname}_shapley_additive_explanation.png"))
        plt.close()

        # Save specific samples if provided
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

##########################################################################################
#                                                                                        #
#                                   Base class                                           #
#                                                                                        #
##########################################################################################

class Model(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, model_type, loss='logloss', name='Model', dir_log = Path('../'), non_fire_number='full', target_name='nbsinister', task_type='regrssion', post_process=None):
        """
        Initialize the CustomModel class.

        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
        """
        self.best_estimator_ = model
        self.model_type = model_type
        self.name = self.best_estimator_.__class__.__name__ if name == 'Model' else name
        self.loss = loss
        self.X_train = None
        self.y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.dir_log = dir_log
        self.final_score = None
        self.features_selected = None
        self.non_fire_number = non_fire_number
        self.target_name = target_name
        self.task_type = task_type
        self.post_process = post_process

    def fit(self, X, y, X_val, y_val, X_test=None, y_test=None, features_search=False, optimization='skip', grid_params=None, fit_params={}, cv_folds=10):
        """
        Train the model.

        Parameters:
        - X: Training data.
        - y: Labels for the training data.
        - grid_params: Parameters to optimize
        - optimization: Optimization method to use ('grid' or 'bayes').
        - fit_params: Additional parameters for the fit function.
        """

        if self.non_fire_number != 'full':
            if 'binary' in self.non_fire_number:
                vec = self.non_fire_number.split('-')
                try:
                    nb = int(vec[-1]) * len(y[y > 0])
                except ValueError:
                    print(f'{self.non_fire_number} with undefined factor, set to 1 -> {len(y[y > 0])}')
                    nb = len(y[y > 0])

                # Separate the positive and zero classes based on y
                positive_mask = y > 0
                non_fire_mask = y == 0

                X_positive = X[positive_mask]
                y_positive = y[positive_mask]

                X_non_fire = X[non_fire_mask]
                y_non_fire = y[non_fire_mask]

                # Sample non-fire data
                nb = min(len(X_non_fire), nb)
                sampled_indices = np.random.RandomState(42).choice(len(X_non_fire), nb, replace=False)

                X_non_fire_sampled = X_non_fire.iloc[sampled_indices] if isinstance(X, pd.DataFrame) else X_non_fire[sampled_indices]
                y_non_fire_sampled = y_non_fire.iloc[sampled_indices] if isinstance(y, pd.Series) else y_non_fire[sampled_indices]

                # Combine positive and sampled non-fire data
                X_combined = pd.concat([X_positive, X_non_fire_sampled]) if isinstance(X, pd.DataFrame) else np.concatenate([X_positive, X_non_fire_sampled])
                y_combined = pd.concat([y_positive, y_non_fire_sampled]) if isinstance(y, pd.Series) else np.concatenate([y_positive, y_non_fire_sampled])

                # Update X and y for training
                X_combined.reset_index(drop=True, inplace=True)
                y_combined.reset_index(drop=True, inplace=True)
                X = X_combined
                y = y_combined
                print(f'Train mask X shape: {X.shape}, y shape: {y.shape}')

        features = list(X.columns)
        if 'weight' in features:
            features.remove('weight')

        self.X_train = X[features]
        self.y_train = y

        if self.model_type == 'xgboost':
            dval = xgb.DMatrix(X_val[features], label=y_val, weight=X_val['weight'])
            dtrain = xgb.DMatrix(X[features], label=y, weight=X['weight'])
            fit_params = {
                'eval_set': [(dtrain, 'train'), (dval, 'validation')],
                'sample_weight': X['weight'],
                'verbose': False,
                'early_stopping_rounds' : 15
            }

        elif self.model_type == 'catboost':
            cat_features = [col for col in X.columns if str(X[col].dtype) == 'category']  # Identify categorical features
            fit_params = {
                'eval_set': [(X_val[features], y_val)],
                'sample_weight': X['weight'],
                'cat_features': cat_features,
                'verbose': False,
                'early_stopping_rounds': 15,
            }

        elif self.model_type == 'ngboost':
            fit_params = {
                'X_val': X_val[features],
                'Y_val': y_val,
                'sample_weight': X['weight'],
                'early_stopping_rounds': 15,
            }

        elif self.model_type == 'rf':
            fit_params = {
                'sample_weight': X['weight']
            }

        elif self.model_type == 'dt':
            fit_params = {
                'sample_weight': X['weight']
            }

        elif self.model_type == 'lightgbm':
            fit_params = {
                'eval_set': [(X_val[features], y_val)],
                'eval_sample_weight': [X_val['weight']],
                'early_stopping_rounds': 15,
                'verbose': False
            }

        elif self.model_type == 'svm':
            fit_params = {
                'sample_weight': X['weight']
            }

        elif self.model_type == 'poisson':
            fit_params = {
                'sample_weight': X['weight']
            }

        elif self.model_type == 'gam':
            fit_params = {
            'weights': X['weight']
            } 

        elif self.model_type ==  'linear':
            fit_params = {}

        else:
            raise ValueError(f"Unsupported model model_type: {self.model_type}")
        
        if features_search:
            features_selected, selected_features_index, final_score = self.fit_by_features(self.X_train, self.y_train, X_test, y_test, fit_params)
            self.features_selected, self.final_score = features_selected, final_score
            X = X[features_selected]
            fit_params = self.update_fit_params(self.model_type, fit_params, features_selected, selected_features_index)
            
            df_features = pd.DataFrame(index=np.arange(0, len(features_selected)))
            df_features['features'] = features_selected
            df_features['r2_score'] = final_score
            df_features.to_csv(self.dir_log / f'{self.name}_features.csv')
        else:
            self.features_selected = features
            df_features = pd.DataFrame(index=np.arange(0, len(self.features_selected)))
            df_features['features'] = self.features_selected
            df_features['r2_score'] = np.nan
            df_features.to_csv(self.dir_log / f'{self.name}_features.csv')

        # Train the final model with all selected features
        if optimization == 'grid':
            assert grid_params is not None
            grid_search = GridSearchCV(self.best_estimator_, grid_params, scoring=self.get_scorer(), cv=cv_folds, refit=False)
            grid_search.fit(self.X_train, self.y_train, **fit_params)
            best_params = grid_search.best_params_
            self.cv_results_ = grid_search.cv_results_
        elif optimization == 'bayes':
            assert grid_params is not None
            param_list = []
            for param_name, param_values in grid_params.items():
                if isinstance(param_values, list):
                    param_list.append((param_name, param_values))
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    param_list.append((param_name, param_values))
                else:
                    raise ValueError(
                        "Unsupported parameter type in grid_params. Expected list or tuple of size 2.")

            # Configure the parameter space for BayesSearchCV
            param_space = {}
            for param_name, param_range in param_list:
                if isinstance(param_range[0], int):
                    param_space[param_name] = Integer(
                        param_range[0], param_range[-1])
                elif isinstance(param_range[0], float):
                    param_space[param_name] = Real(param_range[0], param_range[-1], prior='log-uniform')

            opt = Optimizer(param_space, base_estimator='GP', acq_func='gp_hedge')
            bayes_search = BayesSearchCV(self.best_estimator_, opt, scoring=self.get_scorer(), cv=cv_folds, Refit=False)
            bayes_search.fit(self.X_train, self.y_train, **fit_params)
            best_params = bayes_search.best_estimator_.get_params()
            self.cv_results_ = bayes_search.cv_results_
        elif optimization == 'skip':
            best_params = self.best_estimator_.get_params()
            self.best_estimator_.fit(self.X_train, self.y_train, **fit_params)
        else:
            raise ValueError("Unsupported optimization method")
        
        self.set_params(**best_params)

        # Perform cross-validation with the specified number of folds
        #cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(self.best_estimator_, X, y, scoring=self.get_scorer(), cv=cv, params=fit_params)
        
        # Fit the model on the entire dataset
        if optimization != 'skip':
            self.best_estimator_.fit(self.X_train, self.y_train, **fit_params)
        
        # Save the best estimator as the one that had the highest cross-validation score
        """self.cv_results_['mean_cv_score'] = np.mean(cv_scores)
        self.cv_results_['std_cv_score'] = np.std(cv_scores)
        self.cv_results_['cv_scores'] = cv_scores"""

        #data_dmatrix = xgboost.DMatrix(data=X, label=y, weight=fit_params['sample_weight'])
        #self.best_estimator_ = xgboost.train(best_params, data_dmatrix)

    def fit_by_features(self, X, y, X_test, y_test, fit_params_ori):
        final_selected_features = []
        final_selected_features_index = []
        final_score = []
        features = list(X.columns)
        num_iteration = 1
        iter_score = -math.inf

        for num_iter in range(num_iteration):
            print(f'###############################################################')
            print(f'#                                                             #')
            print(f'#                 Iteration {num_iter + 1}                              #')
            print(f'#                                                             #')
            print(f'###############################################################')
            features_importance = []
            selected_features_ = []
            selected_features_index = []
            score_ = []
            all_score = []
            all_features = []
            base_score = -math.inf
            count_max = 50
            c = 0
            model = copy.deepcopy(self.best_estimator_)

            if num_iter != 0:
                random.shuffle(features)

            for i, fet in enumerate(features):
                selected_features_.append(fet)
                selected_features_index.append(features.index(fet))

                X_train_single = X[selected_features_]
                
                fit_params = self.update_fit_params(self.model_type, fit_params_ori, selected_features_, selected_features_index)

                model.fit(X=X_train_single, y=y, **fit_params)
                self.best_estimator_ = copy.deepcopy(model)
                self.features_selected = selected_features_
            
                # Calculer le score avec cette seule caractéristique
                single_feature_score = self.score(X_test[selected_features_], y_test)
                all_score.append(single_feature_score)
                all_features.append(fet)

                # Si le score ne s'améliore pas, on retire la variable de la liste
                if single_feature_score <= base_score:
                    selected_features_.pop(-1)
                    selected_features_index.pop(-1)
                    c += 1
                else:
                    print(f'With {fet} number {i}: {base_score} -> {single_feature_score}')
                    base_score = single_feature_score
                    score_.append(single_feature_score)
                    c = 0

                if c > count_max:
                    print(f'Score didn t improove for {count_max} features, we break')
                    break

                features_importance.append(single_feature_score)

            plt.figure(figsize=(15,10))
            x_score = np.arange(len(all_features))
            plt.plot(x_score, all_score)
            plt.xticks(x_score, all_features, rotation=90)
            plt.savefig(self.dir_log / f'{num_iter}.png')

            if base_score > iter_score:
                iter_score = base_score
                final_selected_features = selected_features_
                final_selected_features_index = selected_features_index
                final_score = score_

        plt.figure(figsize=(15,10))
        x_score = np.arange(len(final_selected_features))
        plt.plot(x_score, final_score)
        plt.xticks(x_score, final_selected_features, rotation=90)
        plt.savefig(self.dir_log / f'best_iter.png')

        return final_selected_features, final_selected_features_index, final_score
    
    def update_fit_params(self, model_type, fit_params, features, features_index):
        if model_type.find('xgboost') != -1:

            dval = fit_params.get('eval_set')[1][0]  # Mise à jour de l'ensemble de validation
            data = dval.get_data().toarray()
            label = dval.get_label()
            weight = dval.get_weight()
            data_df = pd.DataFrame(index=np.arange(0, data.shape[0]))
            data_df[features] = data[:, features_index]
            dval = xgb.DMatrix(data_df, label=label, weight=weight)

            dtrain = fit_params.get('eval_set')[0][0]# Mise à jour de l'ensemble d'entraînement
            data = dtrain.get_data().toarray()
            label = dtrain.get_label()
            weight = dtrain.get_weight()
            data_df = pd.DataFrame(index=np.arange(0, data.shape[0]))
            data_df[features] = data[:, features_index]
            dtrain = xgb.DMatrix(data_df, label=label, weight=weight)
            sample_weight = fit_params.get('sample_weight')  # Mise à jour des poids
            fit_params = {
                'eval_set': [(dtrain, 'train'), (dval, 'validation')],
                'sample_weight': sample_weight,
                'verbose': fit_params.get('verbose', False),
                'early_stopping_rounds': fit_params.get('early_stopping_rounds', 15)
            }
            
        elif model_type == 'ngboost':
            dval = fit_params.get('X_val')[features]
            dtrain = fit_params.get('sample_weight')  # Mise à jour de l'ensemble d'entraînement
            fit_params = {
                'X_val': dval,
                'Y_val': fit_params.get('Y_val'),
                'sample_weight': fit_params.get('sample_weight'),
                'early_stopping_rounds': 15,
            }

        elif model_type == 'rf':
            pass

        elif model_type == 'dt':
            pass

        elif model_type == 'lightgbm':
            df_val = fit_params.get('eval_set')[0]
            fit_params = {
                'eval_set': [(df_val[0], df_val[1])],
                'eval_sample_weight': [fit_params.get('eval_sample_weight')[0]],
                'sample_weight': fit_params.get('sample_weight'),
                'early_stopping_rounds': 15,
                'verbose': False
            }

        elif model_type == 'svm':
            pass

        elif model_type == 'poisson':
            pass

        elif model_type == 'gam':
            pass

        elif model_type == 'linear':
            fit_params = {}

        else:
            raise ValueError(f"Unsupported model model_type: {model_type}")
        
        return fit_params

    def get_model(self):
        return self.best_estimator_

    def recursive_fit(self, X_train, y_train, eval_set, features, n_steps, early_stopping_rounds, **fit_params):
        """
        Custom training method that replaces model.fit in the Model class.
        Implements recursive prediction and early stopping.

        Parameters:
        - model: The base estimator (must follow the sklearn API).
        - X_train: Training features.
        - y_train: Training targets.
        - eval_set: Tuple of (X_val, y_val) for validation.
        - n_steps: Number of recursive prediction steps.
        - early_stopping_rounds: Number of rounds without improvement to trigger early stopping.
        - fit_params: Additional parameters for the fit function.

        """
        # Clone the model to avoid modifying the original
        model = clone(self.best_estimator_)

        # Initialize variables for early stopping
        best_score = math.inf
        stopping_rounds = 0
        num_boost_round = fit_params.get('n_estimators', 1000)  # Maximum number of iterations

        X_val, y_val = eval_set if eval_set else (None, None)

        # Check if validation set is provided
        if X_val is None or y_val is None:
            raise ValueError("eval_set must be provided for early stopping.")

        # Initialize lists to store score
        val_scores = []

        # Custom training loop
        for i in range(1, num_boost_round + 1):

            # Fit the model incrementally
            model.set_params(n_estimators=i)
            model.fit(X_train, y_train, **fit_params)

            # Perform recursive predictions on validation set
            val_score = self.evaluate_recursive_predictions(
                model, X_val, y_val, features, n_steps)

            val_scores.append(val_score)

            # Early stopping check
            if val_score < best_score:
                best_score = val_score
                stopping_rounds = 0
                best_model = copy.deepcopy(model)
            else:
                stopping_rounds += 1
                if stopping_rounds >= early_stopping_rounds:
                    print(f"Early stopping at iteration {i}")
                    break

            # Optional: Print progress
            print(f"Iteration {i}, Validation Score: {val_score:.4f}")

        # Return the best model
        self.best_estimator_ = best_model

    def evaluate_recursive_predictions(self, model, X_val, y_val, features, n_steps):
        """
        Evaluate the model using recursive predictions on the validation set,
        handling multiple time series identified by 'id'.

        Parameters:
        - model: The trained model.
        - X_val: Validation features, including 'id' and 'date' columns.
        - y_val: Validation targets.
        - n_steps: Number of recursive prediction steps.
        - loss: Loss function to use ('rmse', 'mae', etc.).

        Returns:
        - The evaluation score.

        """

        forecasts = []
        y_vals = []
        y_weight = []

        # Reset index to ensure proper alignment
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        val_data = X_val[features].copy(deep=True)

        # Get the unique ids
        unique_ids = y_val['id'].unique()

        for uid in unique_ids:
            # Filter data for the current id
            id_data = val_data[y_val['id'] == uid].reset_index(drop=True)
            id_target = y_val[y_val['id'] == uid].reset_index(drop=True)

            # Determine how many predictions we can make
            len_id = len(id_data) - n_steps
            if len_id <= 0:
                continue  # Not enough data for this id

            # Loop over the data for this id
            for index in range(len_id):
                # Initialize current input with known features
                current_input = id_data.iloc[index].copy()

                preds = []
                true_vals = []
                weight = []

                for step in range(n_steps):
                    # Predict
                    pred = model.predict(current_input.values.reshape(1, -1))[0]
                    preds.append(pred)

                    # Update current_input for next prediction
                    # Shift lag features
                    lag_features = [col for col in current_input.index if 'y_lag' in col]
                    print(lag_features)
                    max_lag = max([int(col.split('_')[-1]) for col in lag_features])

                    # Shift lags
                    for lag in range(max_lag, 1, -1):
                        lag_col = f'y_lag_{lag}'
                        prev_lag_col = f'y_lag_{lag-1}'
                        if lag_col in current_input and prev_lag_col in current_input:
                            current_input[lag_col] = current_input[prev_lag_col]

                    # Update the first lag with the latest prediction
                    if 'y_lag_1' in current_input:
                        current_input['y_lag_1'] = pred

                    # Append the true value
                    if index + step + 1 < len(id_data):
                        true_val = id_target['y_true'].iloc[index + step + 1]
                        true_vals.append(true_val)
                        weight.append(id_target['weight'].iloc[index + step + 1])
                    else:
                        break  # No more true values

                forecasts.extend(preds)
                y_vals.extend(true_vals)
                y_weight.extend(weight)

        # Calculate the evaluation metric
        score = self.score_with_prediction(forecasts, y_vals, sample_weight=y_weight)

        return score
    
    def simulate_fit(self, X_train, y_train, eval_set, features, n_steps, early_stopping_rounds, **fit_params):
        pass

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X: Data to predict labels for.

        Returns:
        - Predicted labels.
        """
        #dtest = xgboost.DMatrix(data=X)
        #return self.best_estimator_.predict(dtest)
        return self.best_estimator_.predict(X[self.features_selected])
    
    def predict_nbsinister(self, X, ids=None):
        
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            assert self.post_process is not None
            predict = self.predict(X)
            return self.post_process.predict_nbsinister(predict, ids)
    
    def predict_risk(self, X, ids=None):

        if self.task_type == 'classification':
            return self.predict(X)
        else:
            assert self.post_process is not None
            predict = self.predict(X)
            return self.post_process.predict_risk(predict, ids)
        
    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters:
        - X: Data to predict probabilities for.

        Returns:
        - Predicted probabilities.
        """
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X[self.features_selected])
        elif self.name.find('gam') != -1:
            res = np.zeros((X.shape[0], 2))
            res[:, 1] = self.best_estimator_.predict(X[self.features_selected])
            return res
        else:
            raise AttributeError(
                "The chosen model does not support predict_proba.")

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        #return calculate_signal_scores(y, y_pred)
        if self.loss == 'quantile':
            return my_r2_score(y, y_pred[:, 2])
        return my_r2_score(y, y_pred)
        if self.loss == 'area':
            return calculate_signal_scores(y, y_pred)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def get_params(self, deep=True):
        """
        Get the model's parameters.

        Parameters:
        - deep: If True, return the parameters for this model and nested models.

        Returns:
        - Dictionary of parameters.
        """
        params = {'model': self.best_estimator_,
                  'loss': self.loss, 'name': self.name}
        if deep and hasattr(self.best_estimator_, 'get_params'):
            deep_params = self.best_estimator_.get_params(deep=True)
            params.update(deep_params)
        return params

    def set_params(self, **params):
        """
        Set the model's parameters.

        Parameters:
        - params: Dictionary of parameters to set.

        Returns:
        - Self.
        """
        best_estimator_params = {}
        for key, value in params.items():
            if key in ['model', 'loss', 'name']:
                setattr(self, key, value)
            else:
                best_estimator_params[key] = value

        if best_estimator_params != {}:
            self.best_estimator_.set_params(**best_estimator_params)

        return self

    def get_scorer(self):
        """
        Return the scoring function as a string based on the chosen loss function.
        """
        return r2_score
        if self.loss == 'logloss':
            return 'neg_logloss'
        elif self.loss == 'hinge_loss':
            return 'hinge'
        elif self.loss == 'accuracy':
            return 'accuracy'
        elif self.loss == 'mse':
            return 'neg_mean_squared_error'
        elif self.loss == 'rmse':
            return 'neg_root_mean_squared_error'
        elif self.loss == 'rmsle':
            return 'neg_root_mean_squared_log_error'
        elif self.loss == 'poisson':
            return 'neg_mean_poisson_deviance'
        elif self.loss == 'huber_loss':
            return 'neg_mean_squared_error'
        elif self.loss == 'log_cosh_loss':
            return 'neg_mean_squared_error'
        elif self.loss == 'tukey_biweight_loss':
            return 'neg_mean_squared_error'
        elif self.loss == 'exponential_loss':
            return 'neg_mean_squared_error'
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        
    def plot_features_importance(self, X_set, y_set, outname, dir_output, mode='bar', figsize=(50, 25), limit=10):
        """
        Display the importance of features using feature permutation.

        Parameters:
        - X_set: Data to evaluate feature importance.
        - y_set: Corresponding labels.
        - names: Names of the features.
        - outname : Name of the test set
        - dir_output: Directory to save the plot.
        - mode : mustache (boxplot) or bar.
        """
        names = X_set.columns
        result = permutation_importance(self.best_estimator_, X_set, y_set,
                                        n_repeats=10, random_state=42, n_jobs=-1, scoring=self.get_scorer())
        importances = result.importances_mean
        indices = importances.argsort()[-limit:]
        if mode == 'bar':
            plt.figure(figsize=figsize)
            plt.title(f"Permutation importances {self.name}")
            plt.bar(range(len(importances[indices])),
                    importances[indices], align="center")
            plt.xticks(range(len(importances[indices])), [
                       names[i] for i in indices], rotation=90)
            plt.xlim([-1, len(importances[indices])])
            plt.ylabel(f"Decrease in {self.get_scorer()} score")
            plt.tight_layout()
            plt.savefig(Path(dir_output) /
                        f"{outname}_permutation_importances_{mode}.png")
            plt.close('all')
        elif mode == 'mustache' or mode == 'boxplot':
            plt.figure(figsize=figsize)
            plt.boxplot(importances[indices].T, vert=False, whis=1.5)
            plt.title(f"Permutation Importances {self.name}")
            plt.axvline(x=0, color="k", linestyle="--")
            plt.xlabel(f"Decrease in {self.get_scorer()} score")
            plt.tight_layout()
            plt.savefig(Path(dir_output) /
                        f"{outname}_permutation_importances_{mode}.png")
            plt.close('all')
        else:
            raise ValueError(f'Unknown {mode} for ploting features importance but feel free to add new one')
        
        save_object(result, f"{outname}_permutation_importances.pkl", dir_output)

    def shapley_additive_explanation(self, df_set, outname, dir_output, mode = 'bar', figsize=(50,25), samples=None, samples_name=None):
        """
        Perform shapley additive explanation features on df_set using best_estimator
        
        Parameters:
        - df_set_list : a list for len(self.best_estiamtor) size, with ieme element being the dataframe for ieme estimator 
        - outname : outname of the figure
        - mode : mode of ploting
        - figsize : figure size
        - samples : use for additional plot where the shapley additive explanation is done on each sample
        - samples_name : name of each sample 

        Returns:
        - None
        """
        dir_output = Path(dir_output)
        check_and_create_path(dir_output / 'sample')
        try:
            if isinstance(self.best_estimator_, MyXGBClassifier) or isinstance(self.best_estimator_, MyXGBRegressor):
                self.best_estimator_.shapley_additive_explanation(df_set, outname, dir_output, mode, figsize, samples, samples_name)
                return
            else:
                explainer = shap.Explainer(self.best_estimator_)
            shap_values = explainer(df_set, check_additivity=False)
            plt.figure(figsize=figsize)
            if mode == 'bar':
                shap.plots.bar(shap_values, show=False, max_display=20)
            elif mode == 'beeswarm':
                shap.plots.beeswarm(shap_values, show=False, max_display=20)
            else:
                raise ValueError(f'Unknow {mode} mode')
            
            shap_values_abs = np.abs(shap_values.values).mean(axis=0)  # Importance moyenne absolue des SHAP values
            top_features_indices = np.argsort(shap_values_abs)[-10:]  # Indices des 10 plus importantes
            self.top_features_ = df_set.columns[top_features_indices].tolist()  # Noms des 10 features
            
            plt.tight_layout()
            plt.savefig(dir_output / f'{outname}_shapley_additive_explanation.png')
            plt.close('all')
            if samples is not None and samples_name is not None:
                for i, sample in enumerate(samples):
                    plt.figure(figsize=(30,15))
                    shap.plots.force(shap_values[sample], show=False, matplotlib=True, text_rotation=45, figsize=(30,15))
                    plt.tight_layout()
                    plt.savefig(dir_output / 'sample' / f'{outname}_{samples_name[i]}_shapley_additive_explanation.png')
                    plt.close('all')

        except Exception as e:
            print(f'Error {e} with shapley_additive_explanation')
            return

    def plot_param_influence(self, param, dir_output, figsize=(25,25)):
        """
        Display the influence of parameters on model performance.

        Parameters:
        - param: The parameter to visualize.
        - dir_output: Directory to save the plot.
        """
        if self.cv_results_ is None:
            raise AttributeError(
                "Grid search or bayes search results not available. Please run GridSearchCV or BayesSearchCV first.")

        if param not in self.cv_results_['params'][0]:
            raise ValueError(
                f"The parameter {param} is not in the grid or bayes search results.")

        param_values = [result[param] for result in self.cv_results_['params']]
        means = self.cv_results_['mean_test_score']
        stds = self.cv_results_['std_test_score']

        plt.figure(figsize=figsize)
        plt.title(f"Influence of {param} on performance for {self.name}")
        plt.xlabel(param)
        plt.ylabel("Mean score")
        plt.errorbar(param_values, means, yerr=stds, fmt='-o')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(dir_output) / f"{self.name}_{param}_influence.png")
        plt.close('all')

    def log(self, dir_output):
        assert self.final_score is not None
        check_and_create_path(dir_output)
        plt.figure(figsize=(15,5))
        plt.plot(self.final_score)
        x_score = np.arange(len(self.features_selected))
        plt.xticks(x_score, self.features_selected, rotation=45)
        plt.savefig(self.dir_log / f'{self.name}.png')

class MeanFeaturesModel(Model):
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        # Retourner la moyenne des features
        return np.mean(X, axis=1)

    def predict_proba(self, X):
        A = np.ones(X.shape[0]) - X[:, 0]
        B = np.ones(X.shape[0]) - X[:, 1]

        return 1 - (A * B)

##########################################################################################
#                                                                                        #
#                                   Tree                                                 #
#                                                                                        #
##########################################################################################

class ModelTree(Model):
    def __init__(self, model, model_type, loss='logloss', name='ModelTree', non_fire_number='full', target_name='nbsinister'):
        """
        Initialize the ModelTree class.

        Parameters:
        - model: The base model to use (must follow the sklearn API and support tree plotting).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
        """
        super().__init__(model=model, model_type=model_type, loss=loss, name=name, non_fire_number=non_fire_number, target_name=target_name)

    def plot_tree(self, features_name=None, class_names=None, filled=True, outname="tree_plot", dir_output=".", figsize=(20, 20)):
        """
        Plot a tree for tree-based models.

        Parameters:
        - feature_names: Names of the features.
        - class_names: Names of the classes (for classification tasks).
        - filled: Whether to color the nodes to reflect the majority class or value.
        - outname: Name of the output file.
        - dir_output: Directory to save the plot.
        """
        if isinstance(self.best_estimator_, DecisionTreeClassifier) or isinstance(self.best_estimator_, DecisionTreeRegressor):
            # Plot for DecisionTree
            plt.figure(figsize=figsize)
            sklearn_plot_tree(self.best_estimator_, feature_names=features_name,
                              class_names=class_names, filled=filled)
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, RandomForestClassifier) or isinstance(self.best_estimator_, RandomForestRegressor):
            # Plot for RandomForest - only the first tree
            plt.figure(figsize=figsize)
            sklearn_plot_tree(self.best_estimator_.estimators_[
                              0], feature_names=features_name, class_names=class_names, filled=filled)
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, XGBClassifier) or isinstance(self.best_estimator_, XGBRegressor):
            # Plot for XGBoost
            plt.figure(figsize=figsize)
            xgb_plot_tree(self.best_estimator_, num_trees=0)
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, LGBMClassifier) or isinstance(self.best_estimator_, LGBMRegressor):
            # Plot for LightGBM
            plt.figure(figsize=figsize)
            lgb_plot_tree(self.best_estimator_, tree_index=0, figsize=figsize, show_info=[
                          'split_gain', 'internal_value', 'internal_count', 'leaf_count'])
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, NGBClassifier) or isinstance(self.best_estimator_, NGBRegressor):
            # Plot for NGBoost - not directly supported, but you can plot the base learner
            if hasattr(self.best_estimator_, 'learners_'):
                learner = self.best_estimator_.learners_[0][0]
                if hasattr(learner, 'tree_'):
                    plt.figure(figsize=figsize)
                    sklearn_plot_tree(
                        learner, feature_names=features_name, class_names=class_names, filled=filled)
                    plt.savefig(Path(dir_output) / f"{outname}.png")
                    plt.close('all')
                else:
                    raise AttributeError(
                        "The base learner of NGBoost does not support tree plotting.")
            else:
                raise AttributeError(
                    "The chosen NGBoost model does not support tree plotting.")
        else:
            raise AttributeError(
                "The chosen model does not support tree plotting.")
        
##########################################################################################
#                                                                                        #
#                                   Voting                                              #
#                                                                                        #
##########################################################################################
        
class DualModel(RegressorMixin, ClassifierMixin):
    def __init__(self, models, features, loss='mse', name='DualModel', dir_log=Path('../'), target_name='nbsinister', post_process=None):
        """
        Initialize the DualModel class.

        Parameters:
        - models: A list of two models. The first is trained on all samples, the second on samples with target > 0.
        - features: List of features to use for training.
        - loss: Loss function to use ('mse', 'rmse', etc.).
        - name: The name of the model.
        - dir_log: Directory to save logs and outputs.
        - target_name: Name of the target column.
        """
        if len(models) != 2:
            raise ValueError("DualModel requires exactly two models.")
        self.model_all = models[0]  # Model trained on all samples
        self.model_positive = models[1]  # Model trained only on samples with target > 0
        self.features = features
        self.name = name
        self.loss = loss
        self.dir_log = dir_log
        self.target_name = target_name
        self.is_fitted_ = [False, False]  # Track the fitting status of both models
        self.post_process = post_process

    def fit(self, X, y, X_val=None, y_val=None, X_test=None, y_test=None, features_search=False, optimization='skip', grid_params=[None, None], fit_params=[{}, {}], cv_folds=10):
        """
        Train the DualModel.

        Parameters:
        - X: Training features (DataFrame).
        - y: Training target (Series).
        - X_val: Validation features (optional).
        - y_val: Validation target (optional).
        - optimization: Optimization method to use ('grid' or 'bayes').
        - grid_params_list: List of parameters for grid search (optional).
        - fit_params_list: Additional parameters for fitting (optional).
        - cv_folds: Number of cross-validation folds (default: 10).
        """
        self.is_fitted_ = [True, True]  # Mark both models as fitted after training

        y_binary = (y > 0).astype(int)
        if y_val is not None:
            y_val_binary = (y_val > 0).astype(int)
        else:
            y_val_binary = None

        if y_test is not None:
            y_test_binary = (y_test > 0).astype(int)
        else:
            y_test_binary = y_test

        # Model 1: Train on all data
        fit_params_1 = fit_params[0]
        self.model_positive.fit(X, y_binary, X_val=X_val, y_val=y_val_binary, X_test=X_test, y_test=y_test_binary,
                                features_search=features_search, optimization=optimization,
                                grid_params=grid_params[0],
                                cv_folds=cv_folds, fit_params=fit_params_1)

        # Model 2: Train only on samples where target > 0
        X_train_positive = X[y > 0]
        y_train_positive = y[y > 0]
        X_val_positve = X_val[y_val > 0]
        y_val_positve = y_val[y_val > 0]
        fit_params_2 = fit_params[1]
        self.model_all.fit(X_train_positive, y_train_positive, X_val=X_val_positve, y_val=y_val_positve, X_test=X_test, y_test=y_test,
                           features_search=features_search, optimization=optimization,
                            grid_params=grid_params[1],
                            cv_folds=cv_folds, fit_params=fit_params_2)

    def predict(self, X):
        """
        Predict using the DualModel.

        Parameters:
        - X: Input data for prediction.

        Returns:
        - Predictions from the DualModel.
        """
        if not all(self.is_fitted_):
            raise ValueError("Both models must be fitted before calling predict.")

        # Predict using the first model
        X_test = X[self.features]
        predictions = self.model_positive.predict(X_test)

        # Adjust predictions for positive samples
        mask_positive = predictions > 0
        if mask_positive.any():
            X_positive = X_test[mask_positive]
            predictions[mask_positive] = self.model_all.predict(X_positive)

        return predictions

    def predict_proba(self, X):
        """
        Predict probabilities using the DualModel (classification tasks).

        Parameters:
        - X: Input data for prediction.

        Returns:
        - Predicted probabilities.
        """
        if not hasattr(self.model_all, "predict_proba") or not hasattr(self.model_positive, "predict_proba"):
            raise AttributeError("Both models must support predict_proba for this method.")

        # Predict probabilities using the first model
        X_test = X[self.features]
        proba_all = self.model_positive.predict_proba(X_test)

        # Adjust probabilities for positive samples
        mask_positive = proba_all[:, 1] > 0.5
        if mask_positive.any():
            X_positive = X_test[mask_positive]
            proba_positive = self.model_all.predict_proba(X_positive)
            proba_all[mask_positive] = proba_positive

        return proba_all
    
    def predict_nbsinister(self, X, ids=None):
        """
        Predict the number of sinister cases using the model.

        Parameters:
        - X: Input data for prediction.
        - ids: Optional identifiers for the data.

        Returns:
        - Predicted number of sinister cases.
        """
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            assert self.post_process is not None, "Post-process module is required for non-nbsinister predictions."
            predictions = self.predict(X)
            return self.post_process.predict_nbsinister(predictions, ids)

    def predict_risk(self, X, ids=None):
        """
        Predict the risk using the model.

        Parameters:
        - X: Input data for prediction.
        - ids: Optional identifiers for the data.

        Returns:
        - Predicted risk.
        """
        if self.loss == 'classification':
            return self.predict(X)
        else:
            assert self.post_process is not None, "Post-process module is required for non-classification predictions."
            predictions = self.predict(X)
            return self.post_process.predict_risk(predictions, ids)

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)

    def score_with_prediction(self, y_pred, y, sample_weight=None):
        #return calculate_signal_scores(y, y_pred)
        if self.loss == 'quantile':
            return my_r2_score(y, y_pred[:, 2])
        return my_r2_score(y, y_pred)
        if self.loss == 'area':
            return calculate_signal_scores(y, y_pred)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def save(self, filepath):
        """
        Save the DualModel to a file.

        Parameters:
        - filepath: Path to save the model.
        """
        import joblib
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load a DualModel from a file.

        Parameters:
        - filepath: Path to load the model from.

        Returns:
        - Loaded DualModel.
        """
        import joblib
        return joblib.load(filepath)
    
    def get_params(self, deep=True):
        """
        Get the parameters of both models.

        Parameters:
        - deep: If True, include nested model parameters.

        Returns:
        - A dictionary of parameters.
        """
        params = {
            'model_all': self.model_all,
            'model_positive': self.model_positive,
            'features': self.features,
            'loss': self.loss,
            'name': self.name
        }
        if deep:
            params.update({'model_all_params': self.model_all.get_params(deep=True)})
            params.update({'model_positive_params': self.model_positive.get_params(deep=True)})
        return params

    def set_params(self, **params):
        """
        Set the parameters of both models.

        Parameters:
        - params: Dictionary of parameters to set.

        Returns:
        - Self.
        """
        if 'model_all' in params:
            self.model_all = params['model_all']
        if 'model_positive' in params:
            self.model_positive = params['model_positive']
        if 'features' in params:
            self.features = params['features']
        if 'loss' in params:
            self.loss = params['loss']
        if 'name' in params:
            self.name = params['name']
        return self

    def log(self, dir_output):
        """
        Save logs or visualizations related to the models.

        Parameters:
        - dir_output: Directory to save logs.

        Returns:
        - None
        """
        print(f"Logging model information to {dir_output}")
        # Add any specific logging logic if needed

        
##########################################################################################
#                                                                                        #
#                                   Voting                                              #
#                                                                                        #
##########################################################################################

class ModelVoting(RegressorMixin, ClassifierMixin):
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), non_fire_number='full', target_name='nbsinister'):
        """
        Initialize the ModelVoting class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__()
        self.best_estimator_ = models  # Now a list of models
        self.features = features
        self.name = name
        self.loss = loss
        self.X_train = None
        self.y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.features_per_model = []
        self.dir_log = dir_log
        self.non_fire_number = non_fire_number
        self.target_name = target_name

    def fit(self, X, y, X_val, y_val, X_test, y_test, optimization='skip', grid_params_list=None, fit_params_list=None, cv_folds=10):
        """
        Train each model on the corresponding data.

        Parameters:
        - X_list: List of training data for each model.
        - y_list: List of labels for the training data for each model.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - grid_params_list: List of parameters to optimize for each model.
        - fit_params_list: List of additional parameters for the fit function for each model.
        - cv_folds: Number of cross-validation folds.
        """

        self.cv_results_ = []
        self.is_fitted_ = [True] * len(self.best_estimator_)

        self.features_per_model, self.best_estimator_, self.final_scores = \
                    self.fit_by_features(X, y, X_val, y_val, X_test, y_test, self.best_estimator_, optimization, grid_params_list, fit_params_list, cv_folds)

    def fit_by_features(self, X, y, X_val, y_val, X_test, y_test, models, optimization, grid_params_list, fit_params_list, cv_folds):
        final_selected_features = []
        final_models = []
        final_scores = []
        final_fit_params = []
        num_iteration = 1
        
        features = copy.deepcopy(self.features)  # Liste des features initiales

        for model_index, model in enumerate(models):
            best_selected_features = []
            best_model = None
            best_score = -math.inf
            
            for num_iter in range(num_iteration):
                print(f'###############################################################')
                print(f'#                                                             #')
                print(f'#                 Iteration {num_iter + 1}                    #')
                print(f'#                                                             #')
                print(f'###############################################################')

                # Mélanger les features à chaque itération
                shuffled_features = copy.deepcopy(features)
                random.shuffle(shuffled_features)

                selected_features_ = []
                selected_features_index = []
                all_features = []
                base_score = -math.inf
                c = 0
                count_max = 50

                # Créer une copie indépendante du modèle pour chaque itération
                current_model = copy.deepcopy(model)
                current_model.dir_log = self.dir_log

                all_score = []
                all_features = []

                score_improvment = []
                features_improvment = []
                
                for i, fet in enumerate(shuffled_features):
                    selected_features_.append(fet)
                    selected_features_index.append(features.index(fet))
                    X_train_selected = X[selected_features_]
                    
                    # Mise à jour des paramètres de fit avec les features sélectionnées
                    fit_params = self.update_fit_params(model.model_type, fit_params_list[model_index].copy(), selected_features_, selected_features_index)

                    # Entraîner le modèle avec les features sélectionnées
                    current_model.fit(X=X_train_selected, y=y, X_val=X_val, y_val=y_val, optimization='skip', fit_params=fit_params)

                    # Calculer le score avec la combinaison actuelle de features
                    current_score = current_model.score(X_test[selected_features_], y_test)
                    all_score.append(current_score)
                    all_features.append(features)

                    if current_score > base_score:
                        print(f'Feature added: {fet} (index: {i}) -> Score improved: {base_score} -> {current_score}')
                        base_score = current_score
                        c = 0  # Réinitialiser le compteur d'arrêt
                        best_iteration_model = copy.deepcopy(current_model)  # Sauvegarder le meilleur modèle pour cette itération
                        score_improvment.append(current_score)
                        features_improvment.append(features)
                    else:
                        print(f'Feature removed: {fet} (index: {i}) -> Score did not improve.')
                        selected_features_.pop(-1)  # Retirer la feature
                        selected_features_index.pop(-1)
                        c += 1

                    # Si le score n'a pas été amélioré pour un nombre consécutif de features, arrêter
                    if c > count_max:
                        print(f'No improvement for {count_max} features, stopping feature selection for this iteration.')
                        break

                plt.figure(figsize=(15,5))
                x_score = np.arange(len(all_features))
                plt.plot(x_score, all_score)
                plt.xticks(x_score, all_features, rotation=45)
                plt.savefig(self.dir_log / f'{model.name}_{model_index}_{num_iter}.png')

                # Comparer les résultats de cette itération avec le meilleur résultat global
                if base_score > best_score:
                    best_score = base_score
                    best_selected_features = selected_features_
                    best_model = best_iteration_model
                    best_scores_array = score_improvment

            print(f"Best score for model {model_index + 1}: {best_score}")
            print(f"Best selected features for model {model_index + 1}: {best_selected_features}")

            # Entraîner le meilleur modèle trouvé avec la meilleure combinaison de features et l'optimisation finale
            if optimization != 'skip':
                best_model.fit(X=X[best_selected_features], y=y, optimization=optimization, grid_params=grid_params_list[model_index], cv_folds=cv_folds, fit_params=fit_params)

            # Stocker les résultats finaux pour ce modèle
            final_selected_features.append(best_selected_features)
            final_models.append(best_model)
            final_scores.append(best_scores_array)
            final_fit_params.append(fit_params)

        return final_selected_features, final_models, final_scores

    def update_fit_params(self, model_type, fit_params, features, features_index):
        if model_type.find('xgboost') != -1:

            dval = fit_params.get('eval_set')[1][0]  # Mise à jour de l'ensemble de validation
            data = dval.get_data().toarray()
            label = dval.get_label()
            weight = dval.get_weight()
            data_df = pd.DataFrame(index=np.arange(0, data.shape[0]))
            data_df[features] = data[:, features_index]
            dval = xgb.DMatrix(data_df, label=label, weight=weight)

            dtrain = fit_params.get('eval_set')[0][0]# Mise à jour de l'ensemble d'entraînement
            data = dtrain.get_data().toarray()
            label = dtrain.get_label()
            weight = dtrain.get_weight()
            data_df = pd.DataFrame(index=np.arange(0, data.shape[0]))
            data_df[features] = data[:, features_index]
            dtrain = xgb.DMatrix(data_df, label=label, weight=weight)
            
            sample_weight = fit_params.get('sample_weight')  # Mise à jour des poids
            fit_params = {
                'eval_set': [(dtrain, 'train'), (dval, 'validation')],
                'sample_weight': sample_weight,
                'verbose': fit_params.get('verbose', False),
                'early_stopping_rounds': 15
            }

        elif model_type == 'ngboost':
            dval = fit_params.get('X_val')[features]
            dtrain = fit_params.get('sample_weight')  # Mise à jour de l'ensemble d'entraînement
            fit_params = {
                'X_val': dval,
                'Y_val': fit_params.get('Y_val'),
                'sample_weight': fit_params.get('sample_weight'),
                'early_stopping_rounds': 15,
            }

        elif model_type == 'rf':
            pass

        elif model_type == 'dt':
            pass

        elif model_type == 'lightgbm':
            df_val = fit_params.get('eval_set')[0]
            fit_params = {
                'eval_set': [(df_val[0], df_val[1])],
                'eval_sample_weight': [fit_params.get('eval_sample_weight')[0]],
                'sample_weight': fit_params.get('sample_weight'),
                'early_stopping_rounds': 15,
                'verbose': False
            }

        elif model_type == 'svm':
            pass

        elif model_type == 'poisson':
            pass

        elif model_type == 'gam':
            pass

        elif model_type == 'linear':
            fit_params = {}

        else:
            raise ValueError(f"Unsupported model model_type: {model_type}")
        
        return fit_params

    def predict(self, X):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """
        predictions = []
        for i, estimator in enumerate(self.best_estimator_):
            X_ = X[self.features_per_model[i]]

            pred = estimator.predict(X_)
            predictions.append(pred)

        # Aggregate predictions
        aggregated_pred = self.aggregate_predictions(predictions)
        return aggregated_pred

    def predict_proba(self, X):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.

        Returns:
        - Aggregated predicted probabilities.
        """
        probas = []
        for i, estimator in enumerate(self.best_estimator_):
            X_ = X[self.features_per_model[i]]
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_)
                probas.append(proba)
            else:
                raise AttributeError(f"The model at index {i} does not support predict_proba.")

        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas)
        return aggregated_proba

    def aggregate_predictions(self, predictions_list):
        """
        Aggregate predictions from multiple models.

        Parameters:
        - predictions_list: List of predictions from each model.

        Returns:
        - Aggregated predictions.
        """
        # Determine if it's a classification or regression task
        if self.loss in ['logloss', 'hinge_loss', 'accuracy']:
            # Classification: Use majority vote
            predictions_array = np.array(predictions_list)
            aggregated_pred = stats.mode(predictions_array, axis=0)[0].flatten()
        else:
            # Regression: Average the predictions
            predictions_array = np.array(predictions_list)
            aggregated_pred = np.mean(predictions_array, axis=0)
            #aggregated_pred = np.max(predictions_array, axis=0)
        return aggregated_pred

    def aggregate_probabilities(self, probas_list):
        """
        Aggregate probabilities from multiple models.

        Parameters:
        - probas_list: List of probability predictions from each model.

        Returns:
        - Aggregated probabilities.
        """
        # Average probabilities
        probas_array = np.array(probas_list)
        aggregated_proba = np.mean(probas_array, axis=0)
        return aggregated_proba

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):

        if self.loss == 'quantile':
            return my_r2_score(y, y_pred[:, 2])
        return my_r2_score(y, y_pred)
    
        return calculate_signal_scores(y_pred, y)
        if self.loss == 'area':
            return -smooth_area_under_prediction_loss(y, y_pred, loss=True)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def get_params(self, deep=True):
        """
        Get the ensemble model's parameters.

        Parameters:
        - deep: If True, return the parameters for this model and nested models.

        Returns:
        - Dictionary of parameters.
        """
        params = {'models': self.best_estimator_,
                  'loss': self.loss, 'name': self.name}
        if deep:
            for i, estimator in enumerate(self.best_estimator_):
                params.update({f'model_{i}': estimator})
                if hasattr(estimator, 'get_params'):
                    estimator_params = estimator.get_params(deep=True)
                    params.update({f'model_{i}__{key}': value for key, value in estimator_params.items()})
        return params
    
    def shapley_additive_explanation(self, X, outname, dir_output, mode = 'bar', figsize=(50,25), samples=None, samples_name=None):
        """
        Perform shapley additive explanation features on each estimator
        
        Parameters:
        - df_set_list : a list for len(self.best_estiamtor) size, with ieme element being the dataframe for ieme estimator 
        - outname : outname of the figure
        - mode : mode of ploting
        - figsize : figure size
        - samples : use for additional plot where the shapley additive explanation is done on each sample
        - samples_name : name of each sample 

        Returns:
        - None
        """

        for i, estimator in enumerate(self.best_estimator_):
            self.best_estimator_[i].shapley_additive_explanation(X[self.features_per_model[i]], f'{outname}_{i}', dir_output, mode, figsize, samples, samples_name)

    def set_params(self, **params):
        """
        Set the ensemble model's parameters.

        Parameters:
        - params: Dictionary of parameters to set.

        Returns:
        - Self.
        """
        models_params = {}
        for key, value in params.items():
            if key in ['models', 'loss', 'name']:
                setattr(self, key, value)
            elif key.startswith('model_'):
                idx_and_param = key.split('__')
                if len(idx_and_param) == 1:
                    idx = int(idx_and_param[0].split('_')[1])
                    self.best_estimator_[idx] = value
                else:
                    idx = int(idx_and_param[0].split('_')[1])
                    param_name = idx_and_param[1]
                    if hasattr(self.best_estimator_[idx], 'set_params'):
                        self.best_estimator_[idx].set_params(**{param_name: value})
            else:
                # General parameter, set to all models
                for estimator in self.best_estimator_:
                    if hasattr(estimator, 'set_params'):
                        estimator.set_params(**{key: value})
        return self
    
    def log(self, dir_output):
        check_and_create_path(dir_output)
        print(self.features_per_model)
        for model_index in range(len(self.features_per_model)):
            plt.figure(figsize=(15,5))
            x_score = np.arange(len(self.features_per_model[model_index]))
            plt.plot(x_score, self.final_scores[model_index])
            plt.xticks(x_score, self.features_per_model[model_index], rotation=45)
            plt.savefig(self.dir_log / f'{self.best_estimator_[model_index].name}_{model_index}.png')
    
#################################################################################
#                                                                               #
#                                Stacking                                       #
#                                                                               #
#################################################################################

class ModelStacking(RegressorMixin, ClassifierMixin):
    def __init__(self, models, final_estimator=None, loss='mse', name='ModelStacking'):
        """
        Initialize the ModelStacking class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).œ
        - final_estimator: The final estimator to use in stacking.
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__(model=None, loss=loss, name=name)
        self.base_estimators = models
        self.final_estimator = final_estimator
        self.name = name
        self.loss = loss
        self.is_classifier = self.loss in ['logloss', 'hinge_loss', 'accuracy']
        self.best_estimator_ = None  # Replaced self.stack_model with self.best_estimator_
        self.X_train = None
        self.y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.is_fitted_ = False  # Keep track of whether the stacking model is fitted

    def fit(self, X_list, y_list, optimization='skip', grid_params=None, fit_params=None, cv_folds=10):
        """
        Train the stacking model on the data.

        Parameters:
        - X_list: List of training data for each base model.
        - y_list: List of labels for the training data for each base model.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - grid_params: Parameters to optimize for the stacking model.
        - fit_params: Additional parameters for the fit function.
        - cv_folds: Number of cross-validation folds.
        """
        if not isinstance(X_list, list) or not isinstance(y_list, list):
            raise ValueError("X_list and y_list must be lists of datasets.")
        if len(self.base_estimators) != len(X_list) or len(X_list) != len(y_list):
            raise ValueError("The length of models, X_list, and y_list must be the same.")

        if fit_params is None:
            fit_params = {}

        self.X_train = X_list
        self.y_train = y_list
        estimators = []

        # Prepare estimators as a list of (name, estimator) tuples
        for idx, estimator in enumerate(self.base_estimators):
            estimators.append((f'estimator_{idx}', estimator))

        # Choose stacking model based on whether it's a classifier or regressor
        if self.is_classifier:
            self.best_estimator_ = StackingClassifier(
                estimators=estimators,
                final_estimator=self.final_estimator,
                cv=cv_folds,
                n_jobs=-1
            )
        else:
            self.best_estimator_ = StackingRegressor(
                estimators=estimators,
                final_estimator=self.final_estimator,
                cv=cv_folds,
                n_jobs=-1
            )

        # Combine X_list and y_list into single X and y
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # Train the stacking model
        if optimization == 'grid':
            assert grid_params is not None
            grid_search = GridSearchCV(
                self.best_estimator_,
                grid_params,
                scoring=self.get_scorer(),
                cv=cv_folds,
                refit=True,
                n_jobs=-1
            )
            grid_search.fit(X, y, **fit_params)
            self.best_estimator_ = grid_search.best_estimator_
            self.cv_results_ = grid_search.cv_results_
        elif optimization == 'skip':
            self.best_estimator_.fit(X, y, **fit_params)
        else:
            raise ValueError("Unsupported optimization method")

        self.is_fitted_ = True

    def predict(self, X_list):
        """
        Predict labels for input data using the stacking model.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Predicted labels.
        """
        assert self.best_estimator_ is not None
        check_is_fitted(self.best_estimator_)
        if not isinstance(X_list, list):
            raise ValueError("X_list must be a list of datasets.")
        if len(self.base_estimators) != len(X_list):
            raise ValueError("The length of models and X_list must be the same.")

        # Combine X_list into single XModelVoting
        X = np.concatenate(X_list, axis=0)

        # Predict using the stacking model
        return self.best_estimator_.predict(X)

    def predict_proba(self, X_list):
        """
        Predict probabilities for input data using the stacking model.

        Parameters:
        - X_list: List of data to predict probabilities for.

        Returns:
        - Predicted probabilities.
        """
        assert self.best_estimator_ is not None
        assert self.is_classifier is True
        check_is_fitted(self.best_estimator_)
        if not self.is_classifier:
            raise AttributeError("predict_proba is not available for regression problems.")
        if not isinstance(X_list, list):
            raise ValueError("X_list must be a list of datasets.")
        if len(self.base_estimators) != len(X_list):
            raise ValueError("The length of models and X_list must be the same.")

        # Combine X_list into single X
        X = np.concatenate(X_list, axis=0)

        # Predict probabilities using the stacking model
        return self.best_estimator_.predict_proba(X)

    def score(self, X_list, y_list, sample_weight_list=None):
        """
        Evaluate the stacking model's performance.

        Parameters:
        - X_list: List of input data.
        - y_list: List of true labels.
        - sample_weight_list: List of sample weights for each dataset.

        Returns:
        - The model's score on the provided data.
        """
        y_pred = self.predict(X_list)
        y_true = np.concatenate(y_list)

        if sample_weight_list is not None:
            sample_weight = np.concatenate(sample_weight_list)
        else:
            sample_weight = None

        return self.score_with_prediction(y_pred, y_true, sample_weight=sample_weight)

    def get_params(self, deep=True):
        """
        Get the stacking model's parameters.

        Parameters:
        - deep: If True, return the parameters for this model and nested models.

        Returns:
        - Dictionary of parameters.
        """
        params = {
            'models': self.base_estimators,
            'final_estimator': self.final_estimator,
            'loss': self.loss,
            'name': self.name
        }
        if deep and self.best_estimator_ is not None:
            params.update(self.best_estimator_.get_params(deep=True))
        return params

    def set_params(self, **params):
        """
        Set the stacking model's parameters.

        Parameters:
        - params: Dictionary of parameters to set.

        Returns:
        - Self.
        """
        if 'models' in params:
            self.base_estimators = params.pop('models')
        if 'final_estimator' in params:
            self.final_estimator = params.pop('final_estimator')
        if 'loss' in params:
            self.loss = params.pop('loss')
            self.is_classifier = self.loss in ['logloss', 'hinge_loss', 'accuracy']
        if 'name' in params:
            self.name = params.pop('name')

        if self.best_estimator_ is not None:
            self.best_estimator_.set_params(**params)
        return self
    
    def score(self, X, y, id):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X, id)
        return self.score_with_prediction(predictions, y)

    def score_with_prediction(self, y_pred, y, sample_weight=None):

        if self.loss == 'quantile':
            return my_r2_score(y, y_pred[:, 2])
        return my_r2_score(y, y_pred)
    
        return calculate_signal_scores(y_pred, y)
        if self.loss == 'area':
            return -smooth_area_under_prediction_loss(y, y_pred, loss=True)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

######################################### Model OneById ###################################################

from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

class OneByID(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, model_type, loss='mse', name='OneByID', col_id_name=None, id_train=None, id_val=None, id_test=None, dir_log = Path('./'), non_fire_number='full', target_name='nbsinister'):
        """
        Initialize the OneByID model.

        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        - name: The name of the model.
        - id_train: List of unique IDs corresponding to training data.
        - id_val: List of unique IDs corresponding to validation data.
        """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.loss = loss
        self.name = name
        self.id_train = np.array(id_train) if id_train is not None else None
        self.id_val = np.array(id_val) if id_val is not None else None
        self.id_test = np.array(id_test) if id_val is not None else None
        self.models_by_id = {}  # Dictionary to store models for each ID
        self.is_fitted_ = False
        self.col_id_name = col_id_name
        self.dir_log = dir_log
        self.non_fire_number = non_fire_number
        self.target_name = target_name

    def fit(self, X, y, X_test=None, y_test=None, features_search=False, optimization='skip', grid_params=None, fit_params=None):
        """
        Train a separate model for each unique ID in id_train.

        Parameters:
        - X_train: Training data.
        - y_train: Training labels.
        - optimization: Optimization method to use ('grid' or 'skip').
        - grid_params: Parameters to optimize for each model (if optimization='grid').
        - fit_params: Additional parameters for the fit function.
        """
        if fit_params is None:
            fit_params = {}

        if self.id_train is None:
            raise ValueError("id_train must be provided to train the model.")

        unique_ids = np.unique(self.id_train)

        # Train a model for each unique ID
        for uid in unique_ids:
            print(f"Training model for ID: {uid}")
            mask_train = self.id_train == uid  # Mask for training data corresponding to the current ID

            X_id_train = X[mask_train]
            y_id_train = y[mask_train]

            if X_test is not None and self.id_test is not None:
                mask_test = self.id_test == uid  # Mask for training data corresponding to the current ID
                X_id_test = X_test[mask_test]
                y_id_test = y_test[mask_test]
            else:
                X_id_test = X_test
                y_id_test = y_test

            if X_id_test.shape[0] == 0:
                doFeatures_search = False
            else:
                doFeatures_search = features_search 

            # Create a clone of the base model to avoid interference
            model = copy.deepcopy(self.model)
            fit_params_model = self.update_fit_params(model.model_type, fit_params.copy(), np.argwhere(self.id_train == uid)[:, 0], np.argwhere(self.id_val == uid)[:, 0])
            model.fit(X_id_train, y_id_train, X_test=X_id_test, y_test=y_id_test, features_search=doFeatures_search, optimization=optimization, grid_params=grid_params, fit_params=fit_params_model, cv_folds=10)

            self.models_by_id[uid] = model

        self.is_fitted_ = True

    def predict(self, X, id):
        """
        Predict labels for input data using the models trained by ID.

        Parameters:
        - X_val: Validation data.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Predicted labels.
        """
        check_is_fitted(self, 'is_fitted_')

        if len(X) != len(id):
            raise ValueError("X and id must have the same length.")

        predictions = np.zeros(len(X))

        unique_ids = np.unique(id)
        for uid in unique_ids:
            print(f"Predicting for ID: {uid}")
            mask = id == uid  # Mask for validation data corresponding to the current ID

            X_id = X[mask]

            if uid not in self.models_by_id:
                raise ValueError(f"No model found for ID: {uid}")

            # Predict using the model corresponding to the current ID
            if uid not in self.models_by_id.keys():
                continue 
            predictions[mask] = self.models_by_id[uid].predict(X_id)

        return predictions

    def predict_proba(self, X, id):
        """
        Predict probabilities for input data using the models trained by ID.

        Parameters:
        - X_val: Validation data.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Predicted probabilities.
        """
        check_is_fitted(self, 'is_fitted_')

        if len(X) != len(id):
            raise ValueError("X_val and id_val must have the same length.")

        probabilities = np.zeros(len(X))

        unique_ids = np.unique(id)
        for uid in unique_ids:
            print(f"Predicting probabilities for ID: {uid}")
            mask_val = id == uid  # Mask for validation data corresponding to the current ID

            X_val = X[mask_val]

            if uid not in self.models_by_id:
                raise ValueError(f"No model found for ID: {uid}")

            # Predict probabilities using the model corresponding to the current ID
            probabilities[mask_val] = self.models_by_id[uid].predict_proba(X_val)[:, 1]

        return probabilities

    def shapley_additive_explanation(self, X, outname, dir_output, mode = 'bar', figsize=(50,25), samples=None, samples_name=None):
        """
        Perform shapley additive explanation features on each estimator
        
        Parameters:
        - df_set_list : a list for len(self.best_estiamtor) size, with ieme element being the dataframe for ieme estimator 
        - outname : outname of the figure
        - mode : mode of ploting
        - figsize : figure size
        - samples : use for additional plot where the shapley additive explanation is done on each sample
        - samples_name : name of each sample 

        Returns:
        - None
        """

        unique_ids = np.unique(self.id_train)

        for i, estimator in enumerate(unique_ids):
            self.models_by_id[uid].shapley_additive_explanation(X, f'{outname}_{i}', dir_output, mode, figsize, samples, samples_name)

    def score(self, X, y, id, sample_weight):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X, id)
        return self.score_with_prediction(predictions, y, sample_weight)

    def score_with_prediction(self, y_pred, y, sample_weight=None):

        if self.loss == 'quantile':
            return my_r2_score(y, y_pred[:, 2])
        return my_r2_score(y, y_pred)
    
        return calculate_signal_scores(y_pred, y)
        if self.loss == 'area':
            return -smooth_area_under_prediction_loss(y, y_pred, loss=True)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def update_fit_params(self, model_type, fit_params, id_train, id_val):
        if model_type.find('xgboost') != -1:
            dval = fit_params.get('eval_set')[1][0].slice(id_val)  # Mise à jour de l'ensemble de validation
            dtrain = fit_params.get('eval_set')[0][0].slice(id_train) # Mise à jour de l'ensemble d'entraînement
            sample_weight = fit_params.get('sample_weight').values[id_train]  # Mise à jour des poids
            fit_params = {
                'eval_set': [(dtrain, 'train'), (dval, 'validation')],
                'sample_weight': sample_weight,
                'verbose': fit_params.get('verbose', False),
                'early_stopping_rounds': fit_params.get('early_stopping_rounds', 15)
            }

        elif model_type == 'ngboost':
            dval = fit_params.get('X_val')[id_val]
            dtrain = fit_params.get('sample_weight')[id_train]  # Mise à jour de l'ensemble d'entraînement
            fit_params = {
                'X_val': dval,
                'Y_val': fit_params.get('Y_val')[id_val],
                'sample_weight': fit_params.get('sample_weight')[id_train],
                'early_stopping_rounds': 15,
            }

        elif model_type == 'rf':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'dt':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'lightgbm':
            df_val = fit_params.get('eval_set')[0][id_val]
            fit_params = {
                'eval_set': [(df_val[0], df_val[1])],
                'eval_sample_weight': [fit_params.get('eval_sample_weight')[0][id_val]],
                'sample_weight': fit_params.get('sample_weight')[id_train],
                'early_stopping_rounds': 15,
                'verbose': False
            }

        elif model_type == 'svm':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'poisson':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'gam':
            sample_weight = fit_params.get('weights')[id_train]
            fit_params = {
                'weights': sample_weight
            }

        elif model_type == 'linear':
            fit_params = {}

        else:
            raise ValueError(f"Unsupported model model_type: {model_type}")
        
        return fit_params

############################################## Federated learning ##################################################################

class FederatedByID(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, model_type, n_udpate=1, loss='mse', name='FederatedByID', col_id_name=None, id_train=None, id_val=None, dir_output=Path('./')):
        """
        Initialize the FederatedByID model.

        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - model_type: Type of the model (e.g., 'xgboost', 'lightgbm', etc.).
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        - name: The name of the model.
        - id_train: List of unique IDs corresponding to training data.
        - id_val: List of unique IDs corresponding to validation data.
        """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.loss = loss
        self.name = name
        self.id_train = np.array(id_train) if id_train is not None else None
        self.id_val = np.array(id_val) if id_val is not None else None
        self.models_by_id = {}  # Dictionary to store models for each ID
        self.global_model = None
        self.is_fitted_ = False
        self.n_update = n_udpate
        self.col_id_name = col_id_name
        self.dir_output = dir_output

    def fit(self, X, y, optimization='skip', grid_params=None, fit_params=None):
        """
        Train a separate model for each unique ID and aggregate them into a global model.

        Parameters:
        - X: Training data.
        - y: Training labels.
        - optimization: Optimization method to use ('grid' or 'skip').
        - grid_params: Parameters to optimize for each model (if optimization='grid').
        - fit_params: Additional parameters for the fit function.
        """
        if fit_params is None:
            fit_params = {}

        if self.id_train is None:
            raise ValueError("id_train must be provided to train the model.")

        unique_ids = np.unique(self.id_train)
        local_models = []  # Store local models for aggregation

        # Train a model for each unique ID
        for update in range(self.n_update):
            for i, uid in enumerate(unique_ids):
                print(f"Training model for ID: {uid}")
                mask_train = self.id_train == uid  # Mask for training data corresponding to the current ID

                X_id_train = X[mask_train]
                y_id_train = y[mask_train]

                # Create a clone of the base model to avoid interference
                if len(local_models) <= i:
                    model = copy.deepcopy(self.model)
                else:
                    model = local_models[i-1]

                # Update fit parameters for the specific client (ID)
                fit_params_model = self.update_fit_params(
                    self.model_type,
                    fit_params.copy(), 
                    np.argwhere(self.id_train == uid)[:, 0],
                    np.argwhere(self.id_val == uid)[:, 0]
                )

                # Train the model
                model.fit(X_id_train, y_id_train, optimization=optimization, grid_params=grid_params, fit_params=fit_params_model)

                # Store the trained model
                self.models_by_id[uid] = model
                if len(local_models) <= i:
                    local_models.append(model)
                else:
                    local_models[i-1] = copy.deepcopy(model)

            # Aggregate all local models into a global model
            self.global_model = self.aggregate_models(local_models)
            self.update_local_models_from_global()

        self.is_fitted_ = True

    def aggregate_models(self, local_models):
        """
        Aggregate local models into a global model.

        Parameters:
        - local_models: List of models trained on each client's data.

        Returns:
        - global_model: Aggregated model.
        """
        # For simplicity, we will average the parameters (e.g., weights of the trees, etc.)
        global_model = copy.deepcopy(self.model)

        if hasattr(global_model.best_estimator_, "get_booster"):
            # If model is an XGBoost-like model, average boosters
            boosters = [m.best_estimator_.get_booster() for m in local_models]
            avg_booster = self.average_boosters(boosters)
            global_model.best_estimator__Booster = avg_booster
        else:
            # For simpler models (e.g., linear models), average coefficients
            coef_sum = np.sum([model.best_estimator_.coef_ for model in local_models], axis=0)
            global_model.best_estimator_.coef_ = coef_sum / len(local_models)

        return global_model

    def average_boosters(self, boosters):
        """
        Average the weights of multiple XGBoost boosters.

        Parameters:
        - boosters: List of XGBoost boosters.

        Returns:
        - A new XGBoost booster with averaged weights.
        """
        avg_booster = copy.deepcopy(boosters[0])
        trees = [b.get_dump() for b in boosters]

        for i in range(len(trees[0])):
            avg_tree = sum(float(tree[i]) for tree in trees) / len(boosters)
            avg_booster._Booster[i] = avg_tree

        return avg_booster
    
    def update_local_models_from_global(self):
        """
        Update all local models with the parameters from the global model.
        """
        if hasattr(self.global_model, "get_booster"):
            global_booster = self.global_model.best_estimator_.get_booster()
            for uid, model in self.models_by_id.items():
                model._Booster = copy.deepcopy(global_booster)
        else:
            global_weights = copy.deepcopy(self.global_model.best_estimator_.coef_)
            for uid, model in self.models_by_id.items():
                model.coef_ = global_weights

    def predict(self, X):
        """
        Predict labels for input data using the global model.

        Parameters:
        - X: Validation data.
        - id: List of IDs corresponding to validation data.

        Returns:
        - Predicted labels.
        """
        check_is_fitted(self, 'is_fitted_')
        return self.global_model.predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the global model's performance.

        Parameters:
        - X: Validation data.
        - y: True labels.
        - id: List of IDs corresponding to validation data.

        Returns:
        - Model score.
        """
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)

    def update_fit_params(self, model_type, fit_params, id_train, id_val):
        if model_type.find('xgboost') != -1:
            dval = fit_params.get('eval_set')[1][0].slice(id_val)  # Mise à jour de l'ensemble de validation
            dtrain = fit_params.get('eval_set')[0][0].slice(id_train) # Mise à jour de l'ensemble d'entraînement
            sample_weight = fit_params.get('sample_weight').values[id_train]  # Mise à jour des poids
            fit_params = {
                'eval_set': [(dtrain, 'train'), (dval, 'validation')],
                'sample_weight': sample_weight,
                'verbose': fit_params.get('verbose', False),
                'early_stopping_rounds': fit_params.get('early_stopping_rounds', 15)
            }

        elif model_type == 'ngboost':
            dval = fit_params.get('X_val')[id_val]
            dtrain = fit_params.get('sample_weight')[id_train]  # Mise à jour de l'ensemble d'entraînement
            fit_params = {
                'X_val': dval,
                'Y_val': fit_params.get('Y_val')[id_val],
                'sample_weight': fit_params.get('sample_weight')[id_train],
                'early_stopping_rounds': 15,
            }

        elif model_type == 'rf':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'dt':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'lightgbm':
            df_val = fit_params.get('eval_set')[0][id_val]
            fit_params = {
                'eval_set': [(df_val[0], df_val[1])],
                'eval_sample_weight': [fit_params.get('eval_sample_weight')[0][id_val]],
                'sample_weight': fit_params.get('sample_weight')[id_train],
                'early_stopping_rounds': 15,
                'verbose': False
            }

        elif model_type == 'svm':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'poisson':
            sample_weight = fit_params.get('sample_weight')[id_train]
            fit_params = {
                'sample_weight': sample_weight
            }

        elif model_type == 'gam':
            sample_weight = fit_params.get('weights')[id_train]
            fit_params = {
                'weights': sample_weight
            }

        elif model_type == 'linear':
            fit_params = {}

        else:
            raise ValueError(f"Unsupported model model_type: {model_type}")
        
        return fit_params
    
    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions = self.predict(X)
        return self.score_with_prediction(predictions, y)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):

        if self.loss == 'quantile':
            return my_r2_score(y, y_pred[:, 2])
        return my_r2_score(y, y_pred)
        return calculate_signal_scores(y_pred, y)
        if self.loss == 'area':
            return -smooth_area_under_prediction_loss(y, y_pred, loss=True)
        if self.loss == 'logloss':
            return -log_loss(y, y_pred)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            pass
        elif self.loss == 'poisson':
            pass
        elif self.loss == 'huber_loss':
            pass
        elif self.loss == 'log_cosh_loss':
            pass
        elif self.loss == 'tukey_biweight_loss':
            pass
        elif self.loss == 'exponential_loss':
            pass
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")