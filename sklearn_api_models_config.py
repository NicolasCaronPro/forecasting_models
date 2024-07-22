import pickle
from pathlib import Path
import os
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import log_loss, hinge_loss, accuracy_score, f1_score, precision_score, recall_score, mean_squared_log_error, mean_squared_error
import math
from skopt import Optimizer, BayesSearchCV
from skopt.space import Integer, Real
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from ngboost import NGBClassifier, NGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from forecasting_models.mean_model import *

def config_xgboost(device, classifier, objective):
    params = {
        'verbosity':0,
        'early_stopping_rounds':15,
        'learning_rate' :0.01,
        'min_child_weight' : 5.0,
        'max_depth' : 6,
        'max_delta_step' : 1.0,
        'subsample' : 0.3,
        'colsample_bytree' : 0.8,
        'colsample_bylevel': 0.8,
        'reg_lambda' : 10.5,
        'reg_alpha' : 0.9,
        'n_estimators' : 10000,
        'random_state': 42,
        'tree_method':'hist',
    }
    
    param_grid = {
        'learning_rate': [0.01],
        'max_depth': [15],
        'subsample': [0.3],
        'colsample_bytree': [0.8],
        'colsample_bylevel' : [0.8],
        'min_child_weight' : [10, 15, 20],
        'reg_lambda' : [10.5, 15.2, 20.3],
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

def config_svm(device, task_type):

    params = {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale'
    }
    
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.3],
        'gamma': ['scale', 'auto']
    }

    if task_type == 'regression':
        return SVR(**params), param_grid
    else:
        return SVC(**params, probability=True), param_grid


def config_random_forest(device, task_type):

    params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'bootstrap': True,
        'random_state': 42
    }
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    if task_type == 'regression':
        return RandomForestRegressor(**params), param_grid
    else:
        return RandomForestClassifier(**params), param_grid

def config_decision_tree(device, task_type):

    params = {
        'criterion': 'mse' if task_type == 'regression' else 'gini',
        'splitter': 'best',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': None,
        'random_state': 42
    }
    
    param_grid = {
        'criterion': ['mse', 'friedman_mse'] if task_type == 'regression' else ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }

    if task_type == 'regression':
        return DecisionTreeRegressor(**params), param_grid
    else:
        return DecisionTreeClassifier(**params), param_grid
