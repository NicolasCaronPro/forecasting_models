import copy
import logging
import math
from re import sub
import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost

from forecasting_models.pytorch.tools_2 import *

from lightgbm import LGBMClassifier, LGBMRegressor, plot_tree as lgb_plot_tree

from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

from scipy import stats

from skopt import BayesSearchCV, Optimizer
from skopt.space import Integer, Real

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)

from sklearn.linear_model import LinearRegression, LogisticRegression

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

from xgboost import XGBClassifier, XGBRegressor, plot_tree as xgb_plot_tree


#from scripts.probability_distribution import weight

def weighted_mse_loss(y_true, y_pred, sample_weight=None):
    squared_error = (y_pred - y_true) ** 2
    if sample_weight is not None:
        return np.sum(squared_error * sample_weight) / np.sum(sample_weight)
    else:
        return np.mean(squared_error)

def poisson_loss(y_true, y_pred, sample_weight=None):
    y_pred = np.clip(y_pred, 1e-8, None)  # Éviter log(0)
    loss = y_pred - y_true * np.log(y_pred)
    if sample_weight is not None:
        return np.sum(loss * sample_weight) / np.sum(sample_weight)
    else:
        return np.mean(loss)

def rmsle_loss(y_true, y_pred, sample_weight=None):
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)
    squared_log_error = (log_pred - log_true) ** 2
    if sample_weight is not None:
        return np.sqrt(np.sum(squared_log_error * sample_weight) / np.sum(sample_weight))
    else:
        return np.sqrt(np.mean(squared_log_error))

def rmse_loss(y_true, y_pred, sample_weight=None):
    squared_error = (y_pred - y_true) ** 2
    if sample_weight is not None:
        return np.sqrt(np.sum(squared_error * sample_weight) / np.sum(sample_weight))
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
    
##########################################################################################
#                                                                                        #
#                                   Base class                                           #
#                                                                                        #
##########################################################################################

class Model(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, loss='logloss', name='Model'):
        """
        Initialize the CustomModel class.

        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
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
            grid_search = GridSearchCV(self.best_estimator_, grid_params, scoring=self.get_scorer(), cv=cv_folds, refit=False)
            grid_search.fit(X, y, **fit_params)
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
            bayes_search.fit(X, y, **fit_params)
            best_params = bayes_search.best_estimator_.get_params()
            self.cv_results_ = bayes_search.cv_results_
        elif optimization == 'skip':
            best_params = self.best_estimator_.get_params()
            self.best_estimator_.fit(X, y, **fit_params)
        else:
            raise ValueError("Unsupported optimization method")
        
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

        #data_dmatrix = xgboost.DMatrix(data=X, label=y, weight=fit_params['sample_weight'])
        #self.best_estimator_ = xgboost.train(best_params, data_dmatrix)

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

        # Initialize lists to store scoreS
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
        if self.loss == 'logloss':
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
        
    def score_with_prediction(self, y_pred, y, sample_weight=None):
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


##########################################################################################
#                                                                                        #
#                                   Tree                                                 #
#                                                                                        #
##########################################################################################

class ModelTree(Model):
    def __init__(self, model, loss='logloss', name='ModelTree'):
        """
        Initialize the ModelTree class.

        Parameters:
        - model: The base model to use (must follow the sklearn API and support tree plotting).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
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

##########################################################################################
#                                                                                        #
#                                   Fusion                                               #
#                                   (Not used)                                           #
##########################################################################################

class ModelFusion(Model):
    def __init__(self, model_list, model, loss='logloss', name='ModelFusion'):
        """
        Initialize the ModelFusion class.

        Parameters:
        ----------
        - model_list : list
            A list of models to use as base learners.
        - model : object
            The model to use for the fusion.
        - loss : str, optional (default='logloss')
            Loss function to use ('logloss', 'hinge_loss', etc.).
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
        
##########################################################################################
#                                                                                        #
#                                   Votting                                              #
#                                                                                        #
##########################################################################################

class ModelVoting(Model):
    def __init__(self, models, loss='mse', name='ModelVoting'):
        """
        Initialize the ModelVoting class.

        Parameters:
        - models: A list of base models to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', 'mse', 'rmse', etc.).
        """
        super().__init__(model=None, loss=loss, name=name)
        self.best_estimator_ = models  # Now a list of models
        self.name = name
        self.loss = loss
        self.X_train = None
        self.y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models

    def fit(self, X_list, y_list, optimization='skip', grid_params_list=None, fit_params_list=None, cv_folds=10):
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
        if not isinstance(X_list, list) or not isinstance(y_list, list):
            raise ValueError("X_list and y_list must be lists of datasets.")
        if len(self.best_estimator_) != len(X_list) or len(X_list) != len(y_list):
            raise ValueError("The length of models, X_list, and y_list must be the same.")

        if grid_params_list is None:
            grid_params_list = [None] * len(self.best_estimator_)
        if fit_params_list is None:
            fit_params_list = [{}] * len(self.best_estimator_)

        self.X_train = X_list
        self.y_train = y_list
        self.cv_results_ = []
        self.is_fitted_ = [False] * len(self.best_estimator_)

        for i, estimator in enumerate(self.best_estimator_):
            
            if hasattr(estimator, 'feature_importances_'):
                self.is_fitted_[i] = True
                
            if self.is_fitted_[i]:
                print(f"Model {i} is already fitted. Skipping retraining.")
                continue

            X = X_list[i]
            y = y_list[i]
            grid_params = grid_params_list[i]
            fit_params = fit_params_list[i]

            # Train the final model with all selected features
            if optimization == 'grid':
                assert grid_params is not None
                grid_search = GridSearchCV(estimator, grid_params, scoring=self.get_scorer(), cv=cv_folds, refit=False)
                grid_search.fit(X, y, **fit_params)
                best_params = grid_search.best_params_
                self.cv_results_.append(grid_search.cv_results_)
                estimator.set_params(**best_params)
            elif optimization == 'bayes':
                # For simplicity, bayesian optimization is not implemented here
                raise NotImplementedError("Bayesian optimization is not implemented in ModelVoting.")
            elif optimization == 'skip':
                estimator.fit(X, y, **fit_params)
            else:
                raise ValueError("Unsupported optimization method")

            estimator.fit(X, y, **fit_params)
            self.is_fitted_[i] = True

    def predict(self, X_list):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """
        if not isinstance(X_list, list):
            raise ValueError("X_list must be a list of datasets.")
        if len(self.best_estimator_) != len(X_list):
            raise ValueError("The length of models and X_list must be the same.")

        predictions = []
        for i, estimator in enumerate(self.best_estimator_):
            X = X_list[i]

            pred = estimator.predict(X)
            predictions.append(pred)

        # Aggregate predictions
        aggregated_pred = self.aggregate_predictions(predictions)
        return aggregated_pred

    def predict_proba(self, X_list):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.

        Returns:
        - Aggregated predicted probabilities.
        """
        if not isinstance(X_list, list):
            raise ValueError("X_list must be a list of datasets.")
        if len(self.best_estimator_) != len(X_list):
            raise ValueError("The length of models and X_list must be the same.")

        probas = []
        for i, estimator in enumerate(self.best_estimator_):
            X = X_list[i]
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
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
            print(predictions_array.shape)
            #aggregated_pred = np.mean(predictions_array, axis=0)
            aggregated_pred = np.max(predictions_array, axis=0)
            print(aggregated_pred.shape)
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

    def score(self, X_list, y_true, sample_weight):
        """
        Evaluate the ensemble model's performance.

        Parameters:
        - X_list: List of input data.
        - y_list: List of true labels.
        - sample_weight_list: List of sample weights for each dataset.

        Returns:
        - The model's score on the provided data.
        """
        y_pred = self.predict(X_list)

        return self.score_with_prediction(y_pred, y_true, sample_weight=sample_weight)

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
    
    def shapley_additive_explanation(self, df_set_list, outname, dir_output, mode = 'bar', figsize=(50,25), samples=None, samples_name=None):
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
            sub_model = Model(estimator, self.loss, name=f'estimator_{i}')
            sub_model.shapley_additive_explanation(df_set_list[i], f'{outname}_{i}', dir_output, mode, figsize, samples, samples_name)

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
    
#################################################################################
#                                                                               #
#                                Stacking                                       #
#                                                                               #
#################################################################################

class ModelStacking(Model):
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

        # Combine X_list into single X
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