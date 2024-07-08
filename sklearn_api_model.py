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
        - param_grid: Dictionnaire des paramètres à optimiser pour GridSearchCV.
        - is_binary: Booléen indiquant si le modèle est binaire ou non.
        - loss: Fonction de perte à utiliser ('log_loss', 'hinge_loss', etc.).
        """
        self.model = model
        self.name = name
        self.loss = loss
        self.best_estimator_ = None
        self.X_test = None
        self.y_test = None
        self.selected_features_ = []
        self.dir_output = None  # Ajout de l'attribut dir_output

    def fit(self, X, y, X_test=None, y_test=None, evaluate_individual_features=True, optimization='grid', param_grid=None, fit_params=None):
        """
        Entraîne le modèle sur les données en utilisant GridSearchCV ou BayesSearchCV.
        
        Parameters:
        - X: Les données d'entraînement.
        - y: Les étiquettes des données d'entraînement.
        - X_test: Les données de test pour évaluer les performances (optionnel).
        - y_test: Les étiquettes des données de test pour évaluer les performances (optionnel).
        - sample_weight: Poids des échantillons.
        - evaluate_individual_features: Booléen pour activer l'évaluation des caractéristiques individuellement.
        - dir_output: Chemin du répertoire de sortie pour sauvegarder les caractéristiques sélectionnées.
        - optimization: Méthode d'optimisation à utiliser ('grid' ou 'bayes').
        - fit_params: Paramètres supplémentaires pour la fonction de fit.
        """
        self.X_test = X_test
        self.y_test = y_test

        if evaluate_individual_features and X_test is not None and y_test is not None:
            n_features = X.shape[1]
            base_score = self.score(X_test, y_test)
            self.selected_features_ = [0]
            for i in range(1, n_features):
                X_train_single = X[:, self.selected_features_]
                X_test_single = X_test[:, self.selected_features_]

                # Cloner le modèle pour chaque itération
                temp_model = clone(self.model)

                if optimization == 'grid':
                    assert param_grid is not None
                    grid_search = GridSearchCV(temp_model, self.param_grid, scoring=self._get_scorer(), cv=5)
                    grid_search.fit(X_train_single, y, **fit_params)
                    best_model = grid_search.best_estimator_
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
                    bayes_search = BayesSearchCV(temp_model, opt, scoring=self._get_scorer(), cv=5)
                    bayes_search.fit(X_train_single, y, **fit_params)
                    best_model = bayes_search.best_estimator_
                else:
                    temp_model.fit(X_train_single, y, **fit_params)
                    best_model = temp_model

                # Calculer le score avec cette seule caractéristique
                single_feature_score = self.score(X_test_single, y_test)

                # Si le score s'améliore, ajouter la caractéristique à la liste
                if single_feature_score > base_score:
                    self.selected_features_.append(i)
        else:
            self.selected_features_ = np.arange(0, X.shape[1])
        # Entraîner le modèle final avec toutes les caractéristiques sélectionnées
        if optimization == 'grid':
            assert param_grid is not None
            grid_search = GridSearchCV(self.model, self.param_grid, scoring=self._get_scorer(), cv=5)
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
            self.model.fit(X[:, self.selected_features_], y, **fit_params)
            self.best_estimator_ = self.model
        else:
             raise ValueError("Unsupported optimization parameter method")
        
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
            return math.sqrt(mean_squared_error(y, y_pred), sample_weight=sample_weight)
        elif self.loss == 'msle':
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
        return {'model': self.model, 'param_grid': self.param_grid, 'is_binary': self.is_binary, 'loss': self.loss}

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