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

    def fit(self, X, y, X_test=None, y_test=None, w_test=None, evaluate_individual_features=True, optimization='grid', param_grid=None, fit_params=None):
        """
        Entraîne le modèle sur les données en utilisant GridSearchCV ou BayesSearchCV.
        
        Parameters:
        - X: Les données d'entraînement.
        - y: Les étiquettes des données d'entraînement.
        - X_test: Les données de test pour évaluer les performances (optionnel).
        - y_test: Les étiquettes des données de test pour évaluer les performances (optionnel).
        - w_test: Poids des échantillons de la base de test
        - evaluate_individual_features: Booléen pour activer l'évaluation des caractéristiques individuellement.
        - param_grid : Paramètre à optimiser
        - optimization: Méthode d'optimisation à utiliser ('grid' ou 'bayes').
        - fit_params: Paramètres supplémentaires pour la fonction de fit.
        """
        self.X_test = X_test
        self.y_test = y_test

        self.X_train = X
        self.y_train = y

        if evaluate_individual_features and X_test is not None and y_test is not None:
            self.features_importance = []
            self.selected_features_ = []
            n_features = X.shape[1]
            early_stopping_rounds = self.best_estimator_.get_params()['early_stopping_rounds']
            self.best_estimator_.set_params(early_stopping_rounds=0) # On enleve l'early stopping parce que je peux pas accéder à Xval dans fitparams à cause du modèle ngboost
            base_score = -math.inf
            for i in range(n_features):
                
                self.selected_features_.append(i)

                X_train_single = X[:, self.selected_features_]

                self.best_estimator_.fit(X_train_single, y)

                # Calculer le score avec cette seule caractéristique
                single_feature_score = self.score(X_test, y_test, sample_weight=w_test)

                # Si le score ne s'améliore pas, on retire la variable de la liste
                if single_feature_score < base_score:
                    self.selected_features_.pop(-1)
                else:
                    base_score = single_feature_score

                self.features_importance.append(single_feature_score)

            self.best_estimator_.set_params({'early_stopping_rounds': early_stopping_rounds})
        else:
            self.selected_features_ = np.arange(0, X.shape[1])

        # Entraîner le modèle final avec toutes les caractéristiques sélectionnées
        if optimization == 'grid':
            assert param_grid is not None
            grid_search = GridSearchCV(self.best_estimator_, param_grid, scoring=self._get_scorer(), cv=5)
            grid_search.fit(X[:, self.selected_features_], y, **fit_params)
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
            bayes_search.fit(X[:, self.selected_features_], y, **fit_params)
            self.best_estimator_= bayes_search.best_estimator_
        elif optimization == 'skip':
            self.best_estimator_.fit(X[:, self.selected_features_], y, **fit_params)
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
        return self.best_estimator_.predict(X[:, self.selected_features_])

    def predict_proba(self, X):
        """
        Prédit les probabilités pour les données d'entrée.
        
        Parameters:
        - X: Les données pour lesquelles prédire les probabilités.
        
        Returns:
        - Les probabilités prédites.
        """
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X[:, self.selected_features_])
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
            return mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            return math.sqrt(mean_squared_log_error(y, y_pred, sample_weight=sample_weight))
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
        elif self.loss == 'neg_root_mean_squared_error':
            return 'rmse'
        elif self.loss == 'rmsle':
            return 'rmsle'
        else:
            raise ValueError(f"Fonction de perte inconnue: {self.loss}")
        
    def _show_features_importance(self, names : str) -> None:
        # To do
        pass