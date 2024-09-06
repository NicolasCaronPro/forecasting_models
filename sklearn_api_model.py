from forecasting_models.tools import *
from xgboost import XGBClassifier, XGBRegressor
from ngboost import NGBClassifier, NGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import plot_tree as xgb_plot_tree
from lightgbm import plot_tree as lgb_plot_tree
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import plot_tree as sklearn_plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import log_loss, hinge_loss, accuracy_score, f1_score, precision_score, recall_score, mean_squared_log_error, mean_squared_error
from skopt import Optimizer, BayesSearchCV
from skopt.space import Integer, Real
import logging
import sys
from typing import Union
import shap
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt import Optimizer
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

def weighted_mse_loss(y_true, y_pred, sample_weights=None):
    squared_error = (y_pred - y_true) ** 2
    if sample_weights is not None:
        return np.sum(squared_error * sample_weights) / np.sum(sample_weights)
    else:
        return np.mean(squared_error)

def poisson_loss(y_true, y_pred, sample_weights=None):
    y_pred = np.clip(y_pred, 1e-8, None)  # Éviter log(0)
    loss = y_pred - y_true * np.log(y_pred)
    if sample_weights is not None:
        return np.sum(loss * sample_weights) / np.sum(sample_weights)
    else:
        return np.mean(loss)

def rmsle_loss(y_true, y_pred, sample_weights=None):
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)
    squared_log_error = (log_pred - log_true) ** 2
    if sample_weights is not None:
        return np.sqrt(np.sum(squared_log_error * sample_weights) / np.sum(sample_weights))
    else:
        return np.sqrt(np.mean(squared_log_error))

def rmse_loss(y_true, y_pred, sample_weights=None):
    squared_error = (y_pred - y_true) ** 2
    if sample_weights is not None:
        return np.sqrt(np.sum(squared_error * sample_weights) / np.sum(sample_weights))
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

class Model(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, loss='log_loss', name='Model'):
        """
        Initialize the CustomModel class.

        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('log_loss', 'hinge_loss', etc.).
        """
        self.best_estimator_ = model
        self.name = self.best_estimator_.__class__.__name__ if name == 'Model' else name
        self.loss = loss
        self.X_train = None
        self.y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute

    def fit(self, X, y, optimization='skip', grid_params=None, fit_params={}, cv_folds=10):
        """
        Train the model on the data using GridSearchCV or BayesSearchCV.

        Parameters:
        - X: Training data.
        - y: Labels for the training data.
        - grid_params: Parameters to optimize.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - fit_params: Additional parameters for the fit function.
        """
        self.X_train = X
        self.y_train = y

        # Train the final model with all selected features
        if optimization == 'grid':
            assert grid_params is not None
            grid_search = GridSearchCV(self.best_estimator_, grid_params, scoring=self._get_scorer(), cv=cv_folds, refit=False)
            grid_search.fit(X, y, **fit_params)
            best_params = grid_search.best_estimator_.get_params()
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
            bayes_search.fit(X, y, **fit_params)
            best_params = bayes_search.best_estimator_.get_params()
            self.cv_results_ = bayes_search.cv_results_
        elif optimization == 'skip':
            best_params = self.best_estimator_.get_params()
            self.best_estimator_.fit(X, y, **fit_params)
        else:
            raise ValueError("Unsupported optimization method")
        print(best_params)
        self.set_params(**best_params)
        # Perform cross-validation with the specified number of folds
        #cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(self.best_estimator_, X, y, scoring=self.get_scorer(), cv=cv, params=fit_params)
        
        # Fit the model on the entire dataset
        self.best_estimator_.fit(X, y, **fit_params)
        
        # Save the best estimator as the one that had the highest cross-validation score
        """self.cv_results_['mean_cv_score'] = np.mean(cv_scores)
        self.cv_results_['std_cv_score'] = np.std(cv_scores)
        self.cv_results_['cv_scores'] = cv_scores"""

        print(self.cv_results_)
        
    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X: Data to predict labels for.

        Returns:
        - Predicted labels.
        """
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters:
        - X: Data to predict probabilities for.

        Returns:
        - Predicted probabilities.
        """
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError(
                "The chosen model does not support predict_proba.")

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance.

        Parameters:
        - X: Input data.
        - y: True labels.
        - sample_weight: Sample weights.

        Returns:
        - The model's score on the provided data.
        """
        y_pred = self.predict(X)
        if self.loss == 'log_loss':
            proba = self.predict_proba(X)
            return -log_loss(y, proba)
        elif self.loss == 'hinge_loss':
            return -hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            return -rmsle_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'poisson_loss':
            return -poisson_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'huber_loss':
            return -huber_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'log_cosh_loss':
            return -log_cosh_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'tukey_biweight_loss':
            return -tukey_biweight_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'exponential_loss':
            return -exponential_loss(y, y_pred, sample_weight=sample_weight)
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
        if self.loss == 'log_loss':
            return 'neg_log_loss'
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
        elif self.loss == 'poisson_loss':
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
        dir_output = Path(dir_output)
        check_and_create_path(dir_output / 'sample')
        try:
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


class ModelTree(Model):
    def __init__(self, model, loss='log_loss', name='ModelTree'):
        """
        Initialize the ModelTree class.

        Parameters:
        - model: The base model to use (must follow the sklearn API and support tree plotting).
        - name: The name of the model.
        - loss: Loss function to use ('log_loss', 'hinge_loss', etc.).
        """
        super().__init__(model, loss, name)

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


class ModelFusion(Model):
    def __init__(self, model_list, model, loss='log_loss', name='ModelFusion'):
        """
        Initialize the ModelFusion class.

        Parameters:
        ----------
        - model_list : list
            A list of models to use as base learners.
        - model : object
            The model to use for the fusion.
        - loss : str, optional (default='log_loss')
            Loss function to use ('log_loss', 'hinge_loss', etc.).
        - name : str, optional (default='ModelFusion')
            The name of the model.
        """
        super().__init__(model, loss, name)
        self.model_list = model_list

    def fit(self, X_list, y_list, y, optimization_list='skip', grid_params_list=None, fit_params_list=None, fit_params=None, grid_params=None, deep=True):
        """
        Train the fusion model on the data using the predictions from the base models.

        Parameters:
        ----------
        - X_list : list of np.array
            List of training data for each base model.
        - y_list : list of np.array
            List of labels for the training data for each base model.
        - y : np.array
            Labels for the training data of the fusion model.
        - optimization_list : str or list of str, optional (default='skip')
            List of optimization methods to use ('grid', 'bayes', or 'skip') for each base model.
        - grid_params_list : list of dict, optional
            List of parameters to optimize for each base model.
        - fit_params_list : list of dict, optional
            List of additional parameters for the fit function for each base model.
        - fit_params : dict, optional
            Additional parameters for the fit function of the fusion model.
        - grid_params : dict, optional
            Parameters to optimize for the fusion model.
        - deep : bool, optional (default=True)
            Whether to perform deep training on the base models.
        """
        base_predictions = []

        for i, model in enumerate(self.model_list):
            X = X_list[i]
            y = y_list[i]

            if deep:
                optimization = optimization_list[i] if isinstance(
                    optimization_list, list) else optimization_list
                grid_params = grid_params_list[i] if grid_params_list else None
                fit_params = fit_params_list[i] if fit_params_list else {}

                model.fit(X, y, optimization=optimization,
                          grid_params=grid_params, fit_params=fit_params)
                # Use the last X_list for consistency
                base_predictions.append(model.predict(X))

        # Stack predictions to form new feature set
        stacked_predictions = np.column_stack(base_predictions)

        # Fit the fusion model
        super().fit(stacked_predictions, y, optimization='skip',
                    grid_params=grid_params, fit_params=fit_params)

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        ----------
        - X : list of np.array
            Data to predict labels for.

        Returns:
        -------
        - np.array
            Predicted labels.
        """
        base_predictions = [model.predict(X[i])
                            for i, model in enumerate(self.model_list)]
        stacked_predictions = np.column_stack(base_predictions)
        return self.best_estimator_.predict(stacked_predictions)

    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters:
        ----------
        - X : list of np.array
            Data to predict probabilities for.

        Returns:
        -------
        - np.array
            Predicted probabilities.
        """
        base_predictions = [model.predict(X[i])
                            for i, model in enumerate(self.model_list)]
        stacked_predictions = np.column_stack(base_predictions)
        return self.best_estimator_.predict_proba(stacked_predictions)

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance.

        Parameters:
        ----------
        - X : list of np.array
            Input data.
        - y : np.array
            True labels.
        - sample_weight : np.array, optional
            Sample weights.

        Returns:
        -------
        - float
            The model's score on the provided data.
        """
        base_predictions = [model.predict(X[i])
                            for i, model in enumerate(self.model_list)]
        stacked_predictions = np.column_stack(base_predictions)
        return super().score(stacked_predictions, y, sample_weight)

    def plot_features_importance(self, X, y_list, y, names, outname, dir_output, mode='bar', figsize=(50, 25), deep=True):
        """
        Display the importance of features using feature permutation.

        Parameters:
        ----------
        - X : list of np.array
            Data to evaluate feature importance.
        - y_list : list of np.array
            Corresponding labels for each base model.
        - y : np.array
            Labels for the fusion model.
        - names : list
            Names of the features.
        - outname : str
            Name of the test set.
        - dir_output : Path
            Directory to save the plot.
        - mode : str, optional (default='bar')
            Plot mode ('bar' or 'mustache').
        - figsize : tuple, optional (default=(50, 25))
            Figure size for the plot.
        - deep : bool, optional (default=True)
            Whether to perform deep plotting for base models.
        """
        if deep:
            for i, model in enumerate(self.model_list):
                model.plot_features_importance(
                    X[i], y_list[i], names[i], f'{model.name}_{outname}', dir_output, mode=mode, figsize=figsize)
        base_predictions = [model.predict(X[i])
                            for i, model in enumerate(self.model_list)]
        stacked_predictions = np.column_stack(base_predictions)
        names = [f'{model.name}_prediction' for model in self.model_list]
        super().plot_features_importance(stacked_predictions, y, outname, dir_output, mode=mode, figsize=figsize)

    def plot_tree(self, deep, features_name_list=None, class_names_list=None, filled=True, outname="tree_plot", dir_output=".", figsize=(20, 20)):
        """
        Plot each model from the model list.

        Parameters:
        ----------
        - deep : bool, optional (default=True)
            Whether to perform deep plotting for base models.
        - features_name_list : list of list, optional
            List of feature names for each base model.
        - class_names_list : list of list, optional
            List of class names for each base model.
        - filled : bool, optional (default=True)
            Whether to color the nodes to reflect the majority class or value.
        - outname : str, optional (default='tree_plot')
            Name of the output file.
        - dir_output : Path, optional (default='.')
            Directory to save the plots.
        - figsize : tuple, optional (default=(20, 20))
            Figure size for the plot.
        """
        if deep:
            for i, model in enumerate(self.model_list):
                if isinstance(model, ModelTree):
                    model.plot_tree(outname=f'{model.name}_tree_plot', dir_output=dir_output,
                                    figsize=figsize, filled=filled, features_name=features_name_list[i], class_names=class_names_list[i])

        features_name = [
            f'{model.name}_prediction' for model in self.model_list]

        if isinstance(self.best_estimator_, DecisionTreeClassifier) or isinstance(self.best_estimator_, DecisionTreeRegressor):
            # Plot for DecisionTree
            plt.figure(figsize=figsize)
            sklearn_plot_tree(
                self.best_estimator_, feature_names=features_name, class_names=None, filled=filled)
            plt.savefig(Path(dir_output) / f"{outname}.png")
            plt.close('all')
        elif isinstance(self.best_estimator_, RandomForestClassifier) or isinstance(self.best_estimator_, RandomForestRegressor):
            # Plot for RandomForest - only the first tree
            plt.figure(figsize=figsize)
            sklearn_plot_tree(self.best_estimator_.estimators_[
                              0], feature_names=features_name, class_names=None, filled=filled)
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
                        learner, feature_names=features_name, class_names=None, filled=filled)
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
