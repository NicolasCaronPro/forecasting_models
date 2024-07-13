import pickle
from pathlib import Path
import os
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, hinge_loss, accuracy_score, f1_score, precision_score, recall_score, mean_squared_log_error, mean_squared_error
import math
from skopt import Optimizer, BayesSearchCV
from skopt.space import Integer, Real
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from ngboost import NGBClassifier, NGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

def config_xgboost(device, classifier, objective):
    params = {
        'verbosity':0,
        'early_stopping_rounds':None,
        'learning_rate' :0.01,
        'min_child_weight' : 1.0,
        'max_depth' : 6,
        'max_delta_step' : 1.0,
        'subsample' : 0.5,
        'colsample_bytree' : 0.7,
        'colsample_bylevel': 0.6,
        'reg_lambda' : 1.7,
        'reg_alpha' : 0.7,
        'n_estimators' : 10000,
        'random_state': 42,
        'tree_method':'hist',
        }
    
    param_grid = {
        'learning_rate': [0.01],
        'max_depth': [10, 15, 20],
        'subsample': [0.3, 0.5, 0.7],
        'colsample_bytree': [0.8],
        'colsample_bylevel' : [0.8],
        'min_child_weight' : [5, 7, 10],
        'reg_lambda' : [5.5, 7.5, 10.5],
        'reg_alpha' : [0.9],
        'random_state' : [42]
    }

    if device == 'cuda':
        params['device']='cuda'

    if not classifier:
        return XGBRegressor(**params,
                            objective = objective
                            ), param_grid
    else:
        return XGBClassifier(**params,
                            objective = objective
                            ), param_grid

def config_lightGBM(device, classifier, objective):
    params = {'verbosity':-1,
        #'num_leaves':64,
        'learning_rate':0.01,
        'early_stopping_rounds': 15,
        'bagging_fraction':0.7,
        'colsample_bytree':0.6,
        'max_depth' : 4,
        'num_leaves' : 2**4,
        'reg_lambda' : 1,
        'reg_alpha' : 0.27,
        #'gamma' : 2.5,
        'num_iterations' :10000,
        'random_state':42
        }

    param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'early_stopping_rounds': [15],
    'bagging_fraction': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'max_depth': [3, 4, 5],
    'num_leaves': [16, 32, 64],
    'reg_lambda': [0.5, 1.0, 1.5],
    'reg_alpha': [0.1, 0.2, 0.3],
    'num_iterations': [10000],
    'random_state': [42]
    }

    if device == 'cuda':
        params['device'] = "gpu"

    if not classifier:
        params['objective'] = objective
        return LGBMRegressor(**params), param_grid
    else:
        params['objective'] = objective
        return LGBMClassifier(**params), param_grid

def config_ngboost(classifier):
    params  = {
        'natural_gradient':True,
        'n_estimators':1000,
        'learning_rate':0.01,
        'minibatch_frac':0.7,
        'col_sample':0.6,
        'verbose':False,
        'verbose_eval':100,
        'tol':1e-4,
    }

    param_grid = {
    'natural_gradient': [True, False],
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'minibatch_frac': [0.6, 0.7, 0.8],
    'col_sample': [0.5, 0.6, 0.7],
    'verbose': [False],
    'verbose_eval': [50, 100, 200],
    'tol': [1e-3, 1e-4, 1e-5],
    'random_state': [42]
    }

    if not classifier:
        return NGBRegressor(**params), param_grid
    else:
        return NGBClassifier(**params), param_grid

def save_object(obj, filename: str, path: Path):
    """
    Sauvegarde un objet en utilisant pickle dans un fichier donné.
    
    Parameters:
    - obj: L'objet à sauvegarder.
    - filename: Le nom du fichier.
    - path: Le chemin du répertoire où sauvegarder le fichier.
    """
    # Vérifie et crée le chemin s'il n'existe pas
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Sauvegarde l'objet en utilisant pickle
    with open(path / filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def find_relevant_features(model, features : np.array, X : np.array, y : np.array, w : np.array,
                          X_val=None, y_val=None, w_val=None,
                          X_test=None, y_test=None, w_test=None):
    
    features_importance = []
    selected_features_ = []
    base_score = -math.inf
    for i, fet in enumerate(features):
        
        selected_features_.append(fet)

        X_train_single = X[:, selected_features_]

        fitparams={
                'eval_set':[(X_train_single, y), (X_val[:, selected_features_], y_val)],
                'sample_weight' : w,
                'verbose' : False
                }

        model.fit(X=X_train_single, y=y, fit_params=fitparams)

        # Calculer le score avec cette seule caractéristique
        single_feature_score = model.score(X_test[:, selected_features_], y_test, sample_weight=w_test)

        # Si le score ne s'améliore pas, on retire la variable de la liste
        if single_feature_score < base_score:
            selected_features_.pop(-1)
        else:
            print(f'With {fet} : {base_score} -> {single_feature_score}')
            base_score = single_feature_score

        features_importance.append(single_feature_score)

    return selected_features_

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, model, loss='log_loss', name='Model'):
        """
        Initialise la classe CustomModel.
        
        Parameters:
        - model: Le modèle de base à utiliser (doit suivre l'API sklearn).
        - name : Le nom du modèle
        - loss: Fonction de perte à utiliser ('log_loss', 'hinge_loss', etc.).
        """
        self.best_estimator_ = model
        self.name = name
        self.loss = loss
        self.X_test = None
        self.y_test = None
        self.selected_features_ = []
        self.dir_output = None  # Ajout de l'attribut dir_output

    def fit(self, X, y, optimization='skip', param_grid=None, fit_params=None):
        """
        Entraîne le modèle sur les données en utilisant GridSearchCV ou BayesSearchCV.
        
        Parameters:
        - X: Les données d'entraînement.
        - y: Les étiquettes des données d'entraînement.
        - param_grid : Paramètre à optimiser
        - optimization: Méthode d'optimisation à utiliser ('grid' ou 'bayes').
        - fit_params: Paramètres supplémentaires pour la fonction de fit.
        """

        self.X_train = X
        self.y_train = y

        # Entraîner le modèle final avec toutes les caractéristiques sélectionnées
        if optimization == 'grid':
            assert param_grid is not None
            grid_search = GridSearchCV(self.best_estimator_, param_grid, scoring=self._get_scorer(), cv=5)
            grid_search.fit(X, y, **fit_params)
            self.best_estimator_ = grid_search.best_estimator_
        elif optimization == 'bayes':
            assert param_grid is not None
            param_list = []
            for param_name, param_values in self.param_grid.items():
                if isinstance(param_values, list):
                    param_list.append(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    param_list.append(param_values)
                else:
                    raise ValueError("Unsupported parameter type in param_grid. Expected list or tuple of size 2.")
                
            # Configure the parameter space for BayesSearchCV
            param_space = {}
            for param_values in param_list:
                param_name = param_values[0]
                param_range = param_values[1]
                if isinstance(param_range[0], int):
                    param_space[param_name] = Integer(param_range[0], param_range[-1])
                elif isinstance(param_range[0], float):
                    param_space[param_name] = Real(param_range[0], param_range[-1], prior='log-uniform')
                
            opt = Optimizer(param_space, base_estimator='GP', acq_func='gp_hedge')
            bayes_search = BayesSearchCV(self.model, opt, scoring=self._get_scorer(), cv=5)
            bayes_search.fit(X, y, **fit_params)
            self.best_estimator_= bayes_search.best_estimator_
        elif optimization == 'skip':
            self.best_estimator_.fit(X, y, **fit_params)
        else:
             raise ValueError("Unsupported optimization method")
        
    def predict(self, X):
        """
        Prédit les étiquettes pour les données d'entrée.
        
        Parameters:
        - X: Les données pour lesquelles prédire les étiquettes.
        
        Returns:
        - Les étiquettes prédites.
        """
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Prédit les probabilités pour les données d'entrée.
        
        Parameters:
        - X: Les données pour lesquelles prédire les probabilités.
        
        Returns:
        - Les probabilités prédites.
        """
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError("Le modèle choisi ne supporte pas predict_proba.")

    def score(self, X, y, sample_weight):
        """
        Évalue les performances du modèle.
        
        Parameters:
        - X: Les données d'entrée.
        - y: Les étiquettes réelles.
        
        Returns:
        - Le score du modèle sur les données fournies.
        """
        y_pred = self.predict(X)
        if self.loss == 'log_loss':
            proba = self.predict_proba(X)
            return log_loss(y, proba)
        elif self.loss == 'hinge_loss':
            return hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            return -math.sqrt(mean_squared_log_error(y, y_pred, sample_weight=sample_weight))
        else:
            raise ValueError(f"Fonction de perte inconnue: {self.loss}")

    def get_params(self, deep=True):
        """
        Obtient les paramètres du modèle.
        
        Parameters:
        - deep: Si True, retourne les paramètres pour ce modèle et les modèles imbriqués.
        
        Returns:
        - Dictionnaire des paramètres.
        """
        return {'model': self.best_estimator_, 'loss': self.loss, 'name' : self.name}

    def set_params(self, **params):
        """
        Définit les paramètres du modèle.
        
        Parameters:
        - params: Dictionnaire des paramètres à définir.
        
        Returns:
        - Self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def _get_scorer(self):
        """
        Retourne la fonction de score basée sur la fonction de perte choisie.
        """
        if self.loss == 'log_loss':
            return 'neg_log_loss'
        elif self.loss == 'hinge_loss':
            return 'hinge'
        elif self.loss == 'accuracy':
            return 'accuracy'
        elif self.loss == 'rmse':
            return 'neg_root_mean_squared_error'
        elif self.loss == 'rmsle':
            return 'rmsle'
        else:
            raise ValueError(f"Fonction de perte inconnue: {self.loss}")
        
    def _show_features_importance(self, names : str) -> None:
        # To do
        pass