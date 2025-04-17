from sklearn.utils import check_random_state

import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

import random

from scipy import stats
from forecasting_models.sklearn.score import *
from forecasting_models.sklearn.ordinal_classifier import *
from forecasting_models.sklearn.models import *
from skopt import BayesSearchCV, Optimizer
from skopt.space import Integer, Real

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.model_selection import cross_validate

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

from sklearn.utils.validation import check_is_fitted

from imblearn.over_sampling import SMOTE

try:
    if torch.cuda.is_available():
        import cupy as cp
except:
    pass

np.random.seed(42)
random.seed(42)
random_state = check_random_state(42)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        logger.info(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def calculate_and_plot_feature_importance_shapley(X, y, feature_names, dir_output, target, figsize=(25, 10), task_type='classification'):
    """
    Calculate and plot feature importance using SHAP values from multiple tree-based models.

    Parameters:
    - X: DataFrame or 2D numpy array of features.
    - y: Series or 1D numpy array of target values.
    - feature_names: List of feature names.
    - dir_output: Directory to save plots.
    - target: Name of the target variable.
    - mode: 'bar' for feature importance, 'beeswarm' for feature impact.
    - figsize: Tuple for figure size.

    Returns:
    - DataFrame with feature importance averaged over all models.
    """
    print('Calculate and plot feature importance using SHAP values from multiple tree-based models.')
    # Initialisation des modèles

    #if (dir_output / f'importance_SHAP_{target}.pkl').is_file():
    #    importance_df = read_object(f'importance_SHAP_{target}.pkl', dir_output)
    #    return importance_df
    
    if task_type == 'classification':
        models = [
            ("Decision Tree", DecisionTreeClassifier(random_state=42)),
            ("Random Forest", RandomForestClassifier(random_state=42)),
            ("ExtraTrees", ExtraTreesClassifier(random_state=42)),
            ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False)),
            #("CatBoost", CatBoostClassifier(silent=True, random_state=42)),
            ("LightGBM", LGBMClassifier(random_state=42))
        ]
    else:
        models = [
            ("Decision Tree", DecisionTreeRegressor(random_state=42)),
            ("Random Forest", RandomForestRegressor(random_state=42)),
            ("ExtraTrees", ExtraTreesRegressor(random_state=42)),
            ("XGBoost", XGBRegressor(random_state=42, use_label_encoder=False)),
            #("CatBoost", CatBoostClassifier(silent=True, random_state=42)),
            ("LightGBM", LGBMRegressor(random_state=42))
        ]

    # Dictionnaire pour stocker les SHAP values moyennées
    shap_importance = np.zeros(X.shape[1])

    # Création du dossier de sortie
    dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Calcul des SHAP values pour chaque modèle
    for model_name, model in models:
        print(f'{model_name}')
        try:
            model.fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_values = explainer.shap_values(X, y, check_additivity=False)  # SHAP values calculées

            # Importance moyenne des SHAP values (valeur absolue moyenne)
            mean_shap_values = np.abs(shap_values).mean(axis=(0, 2))  # Moyenne sur samples et classes

            shap_importance += mean_shap_values

        except Exception as e:
            print(f"Erreur avec {model_name}: {e}")

    # Calcul de l'importance moyenne des caractéristiques
    average_importance = shap_importance / len(models)

    # Création d'un DataFrame pour la visualisation
    importance_df = pd.DataFrame({'Feature': feature_names, 'Average Importance': average_importance})
    importance_df = importance_df.sort_values(by='Average Importance', ascending=False)
    
    # Génération du graphique d'importance
    fig = plt.figure(figsize=figsize)
    plt.bar(importance_df['Feature'], importance_df['Average Importance'], color='skyblue')
    plt.ylabel('Average Importance (SHAP)', fontsize=18)
    plt.xlabel('Feature Name', fontsize=18)
    plt.title('Average Feature Importance (SHAP values)', fontsize=18)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Sauvegarde du graphique
    output_path = dir_output / f'Feature_Importance_SHAP_{target}.png'
    plt.savefig(output_path)
    plt.close(fig)

    save_object(importance_df, f'importance_SHAP_{target}.pkl', dir_output)

    """fig = plt.figure(figsize=figsize)
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    beeswarm_path = dir_output / f'Beeswarm_SHAP_{target}.png'
    plt.savefig(beeswarm_path)
    plt.close(fig)"""

    return importance_df

def plot_ecdf_with_threshold(df, dir_output, target_name, importance_col='Average Importance', feature_col='Feature', threshold=0.95):
    """
    Plots the ECDF of cumulative feature importances and adds a threshold line.

    Parameters:
    - df: pandas DataFrame containing the features and their importance.
    - importance_col: str, the name of the column containing the feature importance values.
    - feature_col: str, the name of the column containing the feature names.
    - threshold: float, the threshold value for the ECDF (default is 0.95).

    Returns:
    - None, but displays a plot.
    """

    # Step 1: Sort the DataFrame by importance in descending order
    df_sorted = df.sort_values(by=importance_col, ascending=False).reset_index(drop=True)

    # Step 2: Calculate the cumulative sum of importances
    df_sorted['cumulative_importance'] = df_sorted[importance_col].cumsum()

    # Normalize the cumulative importance to get the ECDF
#     df_sorted['ecdf'] = df_sorted['cumulative_importance'] / df_sorted['cumulative_importance'].iloc[-1]

    # Step 3: Determine the feature corresponding to the threshold
    threshold_index = np.argmax(df_sorted['cumulative_importance'] >= threshold)
    
    print('threshold_index:',threshold_index)
    if threshold_index < len(df_sorted):
        threshold_feature = df_sorted.iloc[threshold_index][feature_col]
#         threshold_importance = df_sorted.iloc[threshold_index][importance_col]
    else:
        threshold_feature = None
        threshold_importance = None

    # Step 4: Plot the ECDF
    fig = plt.figure(figsize=(20, 10))
    plt.plot(df_sorted[feature_col], df_sorted['cumulative_importance'], color='blue', label='ECDF',marker='s')
    plt.xticks(rotation=90,fontsize=14)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=14)
    plt.xlabel('Feature',fontsize=14)
    plt.ylabel('ECDF',fontsize=14)
    plt.title('ECDF of Cumulative Feature Importances')
    plt.grid(True)

    # Step 5: Add the threshold line
    if threshold_feature is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold {threshold}')
        plt.axvline(x=threshold_feature, color='red', linestyle='--')
        plt.text(threshold_index + 0.02, threshold + 0.02, f'{threshold_index+1}',
                 color='black', ha='center', va='bottom',fontsize=14)

    plt.legend()
    plt.tight_layout()

    # Ensure the output directory exists
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    # Save the image to the specified directory
    output_path = Path(dir_output) / f'Feature_Importance_with_thresholds_{target_name}.png'
    plt.savefig(output_path)
    
    plt.close(fig)

    return df_sorted.head(threshold_index + 1)[feature_col].tolist(), df_sorted[feature_col].tolist()

def calculate_and_plot_feature_importance(X, y, feature_names, dir_output, target, task_type='classification'):
    """
    Calculate and plot average feature importance from multiple tree-based models.

    Parameters:
    - X: Features as a DataFrame or 2D array.
    - y: Labels as a 1D array or Series.
    - feature_names: List of feature names.

    Returns:
    - A DataFrame with feature names and average importance.
    - A bar plot showing the average feature importance.
    """
    #if (dir_output / f'importance_model_{target}.pkl').is_file():
    #    importance_df = read_object(f'importance_model_{target}.pkl', dir_output)
    #    return importance_df
    
    # Initialize models

    print(f'Features importance in save {dir_output} for {target}')
    if task_type == 'regression':
        models = [
                    ("Decision Tree", DecisionTreeRegressor(random_state=42)),
                    ("Random Forest", RandomForestRegressor(random_state=42)),
                    ("ExtraTrees", ExtraTreesRegressor(random_state=42)),
                    ("XGBoost", XGBRegressor(random_state=42, use_label_encoder=False)),
                    #("CatBoost", CatBoostClassifier(silent=True, random_state=42)),
                    ("LightGBM", LGBMRegressor(random_state=42))
                ]
    else:
        models = [
            ("Decision Tree", DecisionTreeClassifier(random_state=42)),
            ("Random Forest", RandomForestClassifier(random_state=42)),
            ("ExtraTrees", ExtraTreesClassifier(random_state=42)),
            ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False)),
            ("CatBoost", CatBoostClassifier(silent=True, random_state=42)),
            ("LightGBM", LGBMClassifier(random_state=42))
        ]

    # Dictionary to store feature importances
    feature_importance = {}

    # Fit models and collect feature importances
    for model_name, model in models:
        try:
            model.fit(X, y)
            importances = model.feature_importances_ / np.sum(model.feature_importances_)
            feature_importance[model_name] = importances
        except AttributeError:
            print(f"Model {model_name} does not support feature importances.")

    # Calculate average feature importance
    importance_matrix = np.array(list(feature_importance.values()))
    average_importance = np.mean(importance_matrix, axis=0)

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Average Importance': average_importance})
    importance_df = importance_df.sort_values(by='Average Importance', ascending=False)

    # Plot feature importance
    fig = plt.figure(figsize=(25, 10))
    plt.bar(importance_df['Feature'], importance_df['Average Importance'], color='skyblue')
    plt.ylabel('Average Importance', fontsize=18)
    plt.xlabel('Feature Name', fontsize=18)
    plt.title('Average Feature Importance from Tree-Based Models', fontsize=18)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Ensure the output directory exists
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    # Save the image to the specified directory
    output_path = Path(dir_output) / f'Feature_Importance_{target}.png'
    plt.savefig(output_path)
    
    plt.close(fig)
    save_object(importance_df, f'importance_model_{target}.pkl', dir_output)

    return importance_df

                
##########################################################################################
#                                                                                        #
#                                   Base class                                           #
#                                                                                        #
##########################################################################################

class Model(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, nbfeatures,
                 model_type, loss='logloss', name='Model',
                 dir_log = Path('../'), under_sampling='full',
                 over_sampling='full', target_name='nbsinister',
                 task_type='regression', post_process=None):
        """
        Initialize the CustomModel class.

        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
        """
        if task_type == 'ordinal-classification':
            self.best_estimator_ = OrdinalClassifier(model)
        else:
            self.best_estimator_ = model
        self.model_type = model_type
        self.name = self.best_estimator_.__class__.__name__ if name == 'Model' else name
        self.loss = loss
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.dir_log = dir_log
        self.final_score = None
        self.features_selected = None
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type
        self.post_process = post_process
        self.nbfeatures = nbfeatures
        self.over_sampling = over_sampling

    def split_dataset(self, X, y, y_train_score, nb, is_unknowed_risk):        
        # Separate the positive and zero classes based on y

        if not is_unknowed_risk:
            positive_mask = y > 0
            non_fire_mask = y == 0
        else:
            non_fire_mask = (X['potential_risk'] > 0) & (y == 0)
            positive_mask = ~non_fire_mask

        X_positive = X[positive_mask]
        y_positive = y[positive_mask]
        y_train_score_positive = y_train_score[positive_mask]

        X_non_fire = X[non_fire_mask]
        y_non_fire = y[non_fire_mask]
        y_train_score_non_fire = y_train_score[non_fire_mask]

        # Sample non-fire data
        print(nb, len(X_non_fire))
        nb = min(len(X_non_fire), nb)
        sampled_indices = np.random.RandomState(42).choice(len(X_non_fire), nb, replace=False)
        X_non_fire_sampled = X_non_fire.iloc[sampled_indices] if isinstance(X, pd.DataFrame) else X_non_fire[sampled_indices]

        if not is_unknowed_risk:
            y_non_fire_sampled = y_non_fire.iloc[sampled_indices] if isinstance(y, pd.Series) else y_non_fire[sampled_indices]
            y_train_score_non_fire_sampled = y_train_score_non_fire.iloc[sampled_indices] if isinstance(y_train_score, pd.Series) else y_train_score_non_fire[sampled_indices]
        else:
            print(np.unique(X_non_fire.iloc[sampled_indices]['potential_risk']))
            y_non_fire_sampled = X_non_fire.iloc[sampled_indices]['potential_risk']
            #y_train_score_non_fire_sampled = X_non_fire.iloc[sampled_indices]['potential_risk']

        X_combined = pd.concat([X_positive, X_non_fire_sampled]) if isinstance(X, pd.DataFrame) else np.concatenate([X_positive, X_non_fire_sampled])
        y_combined = pd.concat([y_positive, y_non_fire_sampled]) if isinstance(y, pd.Series) else np.concatenate([y_positive, y_non_fire_sampled])
        y_train_score_combined = pd.concat([y_train_score_positive, y_train_score_non_fire_sampled]) if isinstance(y, pd.Series) else np.concatenate([y_train_score_positive, y_train_score_non_fire_sampled])

        # Update X and y for training
        X_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)
        y_train_score_combined.reset_index(drop=True, inplace=True)

        return X_combined, y_combined, y_train_score_combined
    
    def add_ordinal_class(self, X, y, limit):        
        # Separate the positive and zero classes based on y

        non_fire_mask = (X['potential_risk'] == limit) & (y == 0)
        positive_mask = ~non_fire_mask

        X_positive = X[positive_mask]
        y_positive = y[positive_mask]

        X_non_fire = X[non_fire_mask]
        y_non_fire = y[non_fire_mask]

        # Sample non-fire data
        nb = min(len(X_non_fire), nb)
        print(nb, len(X_non_fire))
        sampled_indices = np.random.RandomState(42).choice(len(X_non_fire), nb, replace=False)
        X_non_fire_sampled = X_non_fire.iloc[sampled_indices] if isinstance(X, pd.DataFrame) else X_non_fire[sampled_indices]

        y_non_fire_sampled = X_non_fire.iloc[sampled_indices]['potential_risk']

        X_combined = pd.concat([X_positive, X_non_fire_sampled]) if isinstance(X, pd.DataFrame) else np.concatenate([X_positive, X_non_fire_sampled])
        y_combined = pd.concat([y_positive, y_non_fire_sampled]) if isinstance(y, pd.Series) else np.concatenate([y_positive, y_non_fire_sampled])

        # Update X and y for training
        X_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)

        return X_combined, y_combined
    
    def search_samples_proportion(self, X, y, X_val, y_val, X_test, y_test, y_train_score=None, y_val_score=None, y_test_score=None, is_unknowed_risk=False):

        if y_test_score is None:
            y_test_score = np.copy(y_test)

        if not is_unknowed_risk:
            test_percentage = np.arange(0.1, 1.05, 0.05)
        else:
            test_percentage = np.arange(0.0, 1.05, 0.05)

        under_prediction_score_scores = []
        over_prediction_score_scores = []
        iou_scores = []

        """if is_unknowed_risk:
            if (self.dir_log / 'unknowned_scores_per_percentage.pkl').is_file():
                data_log = read_object('unknowned_scores_per_percentage.pkl', self.dir_log)
        else:
            if (self.dir_log / 'test_percentage_scores.pkl').is_file():
                data_log = read_object('test_percentage_scores.pkl', self.dir_log)"""

        ############ FIX ############
        if 'data_log' in locals():
            test_percentage, under_prediction_score_scores, over_prediction_score_scores, iou_scores = data_log[0], data_log[1], data_log[2], data_log[3]
        else:
            for tp in test_percentage:
                
                if not is_unknowed_risk:
                    nb = int(tp * len(y[y == 0]))
                else:
                    nb = int(tp * len(X[(X['potential_risk'] > 0) & (y == 0)]))

                print(f'Trained with {tp} -> {nb} sample of class 0')

                X_combined, y_combined, y_train_score_combined = self.split_dataset(X, y, y_train_score, nb, is_unknowed_risk)
                print(y_combined.shape, y_train_score_combined.shape)
                print(f'Train mask X shape: {X_combined.shape}, y shape: {y_combined.shape}')

                copy_model = copy.deepcopy(self)
                if 'dual' in self.loss:
                    params_model = copy_model.best_estimator_.kwargs
                    params_model['y_train_origin'] = y_train_score_combined
                    copy_model.best_estimator_.update_params(params_model)

                copy_model.under_sampling = 'full'

                copy_model.fit(X_combined, y_combined, X_val, y_val, X_test=X_test, y_test=y_test, y_test_score=y_test_score, \
                               y_train_score=y_train_score, y_val_score=y_val_score, training_mode='normal', \
                                optimization='skip', grid_params=None, fit_params={}, cv_folds=10)

                prediction = copy_model.predict(X_val)

                #under_prediction_score_value = under_prediction_score(y_test_score, prediction)
                #over_prediction_score_value = over_prediction_score(y_test_score, prediction)

                under_prediction_score_value = under_prediction_score(y_val, prediction)
                over_prediction_score_value = over_prediction_score(y_val, prediction)
                iou = iou_score(y_val, prediction)
                
                under_prediction_score_scores.append(under_prediction_score_value)
                over_prediction_score_scores.append(over_prediction_score_value)
                iou_scores.append(iou)
                
                print(f'Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, iou : {iou}')

                if under_prediction_score_value > over_prediction_score_value:
                    break
                
        # Find the index where the two scores cross (i.e., where the difference changes sign)
        score_differences = np.array(under_prediction_score_scores) - np.array(over_prediction_score_scores)

        #index_max = np.argmin(np.abs(score_differences))
        index_max = np.argmax(iou_scores)
        best_tp = test_percentage[index_max]
        
        if is_unknowed_risk:
            plt.figure(figsize=(15, 7))
            plt.plot(test_percentage[:len(under_prediction_score_scores)], under_prediction_score_scores, label='under_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], over_prediction_score_scores, label='over_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], iou_scores, label='IoU')
            plt.xticks(test_percentage)
            plt.xlabel('Percentage of Unknowed sample')
            plt.ylabel('IOU Score')

            # Ajouter une ligne verticale pour le meilleur pourcentage (best_tp)
            plt.axvline(x=best_tp, color='r', linestyle='--', label=f'Best TP: {best_tp:.2f}')
            
            # Ajouter une légende
            plt.legend()

            # Sauvegarder et fermer la figure
            plt.savefig(self.dir_log / f'{self.name}_unknowned_scores_per_percentage.png')
            plt.close()
            save_object([test_percentage, under_prediction_score_scores, over_prediction_score_scores, iou_scores], 'unknowned_scores_per_percentage.pkl', self.dir_log)
        else:
            plt.figure(figsize=(15, 7))
            plt.plot(test_percentage[:len(under_prediction_score_scores)], under_prediction_score_scores, label='under_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], over_prediction_score_scores, label='over_prediction')
            plt.plot(test_percentage[:len(under_prediction_score_scores)], iou_scores, label='IoU')
            plt.xticks(test_percentage)
            plt.xlabel('Percentage of Binary sample')
            plt.ylabel('IOU Score')

            # Ajouter une ligne verticale pour le meilleur pourcentage (best_tp)
            plt.axvline(x=best_tp, color='r', linestyle='--', label=f'Best TP: {best_tp:.2f}')
            
            # Ajouter une légende
            plt.legend()

            # Sauvegarder et fermer la figure
            plt.savefig(self.dir_log / f'{self.name}_scores_per_percentage.png')
            plt.close()
            save_object([test_percentage, under_prediction_score_scores, over_prediction_score_scores, iou_scores], 'test_percentage_scores.pkl', self.dir_log)

        return best_tp

    def search_samples_limit(self, X, y, X_val, y_val, X_test, y_test):
        classes = [4,3,2,1]
        under_prediction_score_scores = []
        over_prediction_score_scores = []

        copy_model = copy.deepcopy(self)
        copy_model.under_sampling = 'full'

        copy_model.fit(X_combined, y_combined, X_val, y_val, X_test=X_test, y_test=y_test, training_mode='normal', optimization='skip', grid_params=None, fit_params={}, cv_folds=10)
        
        prediction = copy_model.predict(X_test)
        
        under_prediction_score_value = under_prediction_score(y_test, prediction)
        over_prediction_score_value = over_prediction_score(y_test, prediction)
        
        under_prediction_score_scores.append(under_prediction_score_value)
        over_prediction_score_scores.append(over_prediction_score_value)
        for c in classes:
            
            print(f'Trained with {c}')

            X_combined, y_combined = self.split_dataset(X, y, c)

            print(f'Train mask X shape: {X_combined.shape}, y shape: {y_combined.shape}')

            copy_model = copy.deepcopy(self)
            copy_model.under_sampling = 'full'

            copy_model.fit(X_combined, y_combined, X_val, y_val, X_test=X_test, y_test=y_test, y_test_score=y_test_score, training_mode='normal', optimization='skip', grid_params=None, fit_params={}, cv_folds=10)
            
            prediction = copy_model.predict(X_test)
            
            under_prediction_score_value = under_prediction_score(y_test, prediction)
            over_prediction_score_value = over_prediction_score(y_test, prediction)
            
            under_prediction_score_scores.append(under_prediction_score_value)
            over_prediction_score_scores.append(over_prediction_score_value)

            print(f'Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}')
        
        # Find the index where the two scores cross (i.e., where the difference changes sign)
        score_differences = np.array(under_prediction_score_scores) - np.array(over_prediction_score_scores)

        if score_differences.shape[0] >  0:
            index_max = np.argmin(np.abs(score_differences))
            best_tp = classes[index_max]         
        
            plt.figure(figsize=(15, 7))
            plt.plot(classes, under_prediction_score_scores, label='under_prediction')
            plt.plot(classes, over_prediction_score_scores, label='over_prediction')
            plt.xticks(classes)
            plt.xlabel('Percentage of Binary sample')
            plt.ylabel('IOU Score')

            # Ajouter une ligne verticale pour le meilleur pourcentage (best_tp)
            plt.axvline(x=best_tp, color='r', linestyle='--', label=f'Best TP: {best_tp:.2f}')
            
            # Ajouter une légende
            plt.legend()

            # Sauvegarder et fermer la figure
            plt.savefig(self.dir_log / f'{self.name}_scores_per_classes.png')
            plt.close()

            save_object([classes, under_prediction_score_scores, over_prediction_score_scores], 'test_classes_scores.pkl', self.dir_log)
            return best_tp
        else:
            return 1.0

    def fit(self, X, y, X_val, y_val, X_test=None, y_test=None, y_val_score=None, y_train_score=None, y_test_score=None, training_mode='normal', optimization='skip', grid_params=None, fit_params={}, cv_folds=10):
        """
        Train the model.

        Parameters:
        - X: Training data.
        - y: Labels for the training data.
        - grid_params: Parameters to optimize
        - optimization: Optimization method to use ('grid' or 'bayes').
        - fit_params: Additional parameters for the fit function.
        """

        features = list(X.columns)
        if 'weight' in features:
            features.remove('weight')
        if 'potential_risk' in features:
            features.remove('potential_risk')

        if self.task_type == 'binary':
            y_test = (y_test > 0).astype(int) if y_test is not None else None
            y_test_score = (y_test_score > 0).astype(int) if y_test_score is not None else None
            y_val = (y_val > 0).astype(int)

        #importance_df = calculate_and_plot_feature_importance(X[features], y, features, self.dir_log / '../importance', self.target_name, task_type=self.task_type)
        #importance_df = calculate_and_plot_feature_importance_shapley(X[features], y, features, self.dir_log / '../importance', self.target_name)
        #features95, featuresAll = plot_ecdf_with_threshold(importance_df, dir_output=self.dir_log / '../importance', target_name=self.target_name)
        
        if self.nbfeatures != 'all':
            features = featuresAll[:int(self.nbfeatures)]
        
        #################################################### Handle over sampling ##############################################

        if self.under_sampling != 'full':
            old_shape = X.shape
            if 'binary' in self.under_sampling:
                vec = self.under_sampling.split('-')
                try:
                    nb = int(vec[-1]) * len(y[y > 0])
                except ValueError:
                    print(f'{self.under_sampling} with undefined factor, set to 1 -> {len(y[y > 0])}')
                    nb = len(y[y > 0])

                    X, y, y_score = self.split_dataset(X, y, y_train_score, nb, False)

                print(f'Original shape {old_shape}, Train mask X shape: {X.shape}, y shape: {y.shape}')

            elif self.under_sampling.find('search') != -1 or 'percentage' in self.under_sampling:

                if self.under_sampling.find('search') != -1:
                    ################# No risk sample ##################
                    best_tp_0 = self.search_samples_proportion(X, y, X_val, y_val, X_test, y_test, y_train_score, y_val_score, y_test_score, False)
                else:
                    vec = self.under_sampling.split('-')
                    best_tp_0 = float(vec[1])

                nb = int(best_tp_0 * len(y[y == 0]))
                X, y, y_score = self.split_dataset(X, y, y_train_score, nb, False)
                if 'dual' in self.loss:
                    params_model = self.best_estimator_.kwargs
                    params_model['y_train_origin'] = y_score
                    self.best_estimator_.update_params(params_model)

                print(f'Original shape {old_shape}, Train mask X shape: {X.shape}, y shape: {y.shape}')

            else:
                raise ValueError(f'Unknow value of under_sampling -> {self.under_sampling}')
        
        ######################################### Handle under sampling #####################################

        if self.over_sampling == 'full':
            pass

        elif 'smote' in self.over_sampling:
            smote_coef = int(self.over_sampling.split('-')[1])
            y_negative = y[y == 0].shape[0]
            
            y_one = max(y[y == 1].shape[0], min(y[y == 1].shape[0] * smote_coef, y_negative))
            y_two = max(y[y == 2].shape[0], min(y[y == 2].shape[0] *smote_coef, y_negative))
            y_three = max(y[y == 3].shape[0], min(y[y == 3].shape[0] * smote_coef, y_negative))
            y_four = max(y[y == 4].shape[0], min(y[y == 4].shape[0] * smote_coef, y_negative))

            if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
                """y_negative = y[y == 0].shape[0]
                y_one = y_negative * 0.01
                y_two = y_negative * 0.01
                y_three = y_negative * 0.01
                y_four = y_negative * 0.01
                smote = SMOTE(random_state=42, sampling_strategy={0 : y_negative, 1 : y_one, 2 : y_two, 3 : y_three, 4 : y_four})"""
                smote = SMOTE(random_state=42, sampling_strategy={0 : y_negative, 1 : y_one, 2 : y_two, 3 : y_three, 4 : y_four})
            elif self.task_type == 'binary':
                smote = SMOTE(random_state=42, sampling_strategy='auto')
            X, y = smote.fit_resample(X, y)

        else:
            raise ValueError(f'Unknow value of under_sampling -> {self.over_sampling}')

        #print(f'Positive data after treatment : {positive_shape}, {y[y  > 0].shape}')
        for uy in np.unique(y):
            print(f'Number of {uy} class : {y[y == uy].shape}') 
        
        X_train = X[features]
        y_train = y
        sample_weight = X['weight']

        X_val = X_val[features]

        ##################################### Search for features #############################################
    
        if training_mode == 'features_search':
            features_selected, final_score = self.fit_by_features(X_train, y_train, X_val, y_val, X_test, y_test, featuresAll, sample_weight, False)
            self.features_selected, self.final_score = features_selected, final_score
            X = X[features_selected]
            
            df_features = pd.DataFrame(index=np.arange(0, len(features_selected)))
            df_features['features'] = features_selected
            df_features['iou_score'] = final_score
            save_object(df_features, f'{self.name}_features.csv', self.dir_log)
        else:
            self.features_selected = features
            df_features = pd.DataFrame(index=np.arange(0, len(self.features_selected)))
            df_features['features'] = self.features_selected
            df_features['iou_score'] = np.nan
            save_object(df_features, f'{self.name}_features.csv', self.dir_log)

        X_train = X[self.features_selected]

        ###############################################" Fit pararams depending of the model type #####################################
        
        fit_params = self.update_fit_params(X_val, y_val, sample_weight, self.features_selected)

        """if self.loss in ['softprob-dual', 'softmax-dual']:
            new_y_train = np.zeros((y_train.shape[0], 2))
            new_y_train[:, 0] = y_train
            new_y_train[:, 1] = y_train_score
            y_train = np.copy(new_y_train)
            del new_y_train"""

        """new_y_val = np.zeros((y_val.shape[0], 2))
            new_y_val[:, 0] = y_val
            new_y_val[:, 1] = y_val_score
            y_val = np.copy(new_y_val)
            del new_y_val

            new_y_test = np.zeros((y_test.shape[0], 2))
            new_y_test[:, 0] = y_test
            new_y_test[:, 1] = y_test_score
            y_test = np.copy(new_y_test)
            del new_y_test"""

        ######################################################### Training #########################################
        if optimization == 'grid':
            assert grid_params is not None
            grid_search = GridSearchCV(self.best_estimator_, grid_params, scoring=self.get_scorer(), cv=cv_folds, refit=False)
            grid_search.fit(X_train, y_train, **fit_params)
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
            bayes_search.fit(X_train, y_train, **fit_params)
            best_params = bayes_search.best_estimator_.get_params()
            self.cv_results_ = bayes_search.cv_results_
        elif optimization == 'skip':
            best_params = self.best_estimator_.get_params()
            self.best_estimator_.fit(X_train, y_train, **fit_params)
        elif optimization == 'cv':
            self.cv_Model = cross_validate(self.best_estimator_, X_train, y_train, cv=cv_folds, scoring=iou_score, return_estimator=True)
            self.best_estimator_ = self.cv_Model['']
        else:
            raise ValueError("Unsupported optimization method")
        
        ########################### Fit final model on the entire dataset ###########################
        if optimization != 'skip' and optimization != 'cv':
            self.set_params(**best_params)
            self.best_estimator_.fit(X_train, y_train, **fit_params)

        ############################# Post process fit ##############################
        if self.post_process is not None:
            pred_val = self.best_estimator_.predict(X_val[self.features_selected])
            self.post_process.fit(pred_val, y_val, **{
                'disp':True,
                'method': 'bfgs'
            })
        
    def get_model(self):
        return self.best_estimator_

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X: Data to predict labels for.

        Returns:
        - Predicted labels.
        """
        res = self.best_estimator_.predict(X[self.features_selected])
        if self.post_process is not None:
            res = self.post_process.predict(res)
        return res
    
    def predict_nbsinister(self, X, ids=None, preprocessor_ids=None):
        
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            predict = self.predict(X)
            if self.post_process is not None:
                return self.post_process.predict_nbsinister(predict, ids)
            return predict
    
    def predict_risk(self, X, ids=None, preprocessor_ids=None):

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            return self.predict(X)
        
        elif self.task_type == 'binary':
                predict = self.predict_proba(X)[:, 1]
                if self.post_process is not None:
                    if isinstance(ids, pd.Series):
                        ids = ids.values
                    if isinstance(preprocessor_ids, pd.Series):
                        preprocessor_ids = preprocessor_ids.values

                    return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)
                return predict
        else:
            predict = self.predict(X)
            if self.post_process is not None:

                if isinstance(ids, pd.Series):
                    ids = ids.values
                if isinstance(preprocessor_ids, pd.Series):
                    preprocessor_ids = preprocessor_ids.values

                return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)
            return predict
        
    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters:
        - X: Data to predict probabilities for.

        Returns:
        - Predicted probabilities.
        """

        if hasattr(self.best_estimator_, "predict_proba"):
            if isinstance(X, np.ndarray):
                return self.best_estimator_.predict_proba(X.reshape(-1, len(self.features_selected)))
            elif isinstance(X, torch.Tensor):
                return self.best_estimator_.predict_proba(X.detach().cpu().numpy().reshape(-1, len(self.features_selected)))
            return self.best_estimator_.predict_proba(X[self.features_selected])
        elif self.name.find('gam') != -1:
            res = np.zeros((X.shape[0], 2))
            res[:, 1] = self.best_estimator_.predict(X[self.features_selected])
            return res
        else:
            raise AttributeError(
                "The chosen model does not support predict_proba.")
        
    def get_all_scores(self, X, y_true, y_fire=None):
        prediction = self.predict(X)
        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            scores = calculate_signal_scores_for_training(prediction, y_true, y_fire)
        elif self.task_type == 'regression':
            raise ValueError(f'get_all_scores Not implemented for regression yet')
        return scores

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
        
        return iou_score(y, y_pred)
    
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
        return iou_score
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

    def fit_by_features(self, X, y, X_val, y_val, X_test, y_test, features, sample_weight, select):
        final_selected_features = []
        final_score = []
        num_iteration = 1
        iter_score = -math.inf

        for num_iter in range(num_iteration):
            print(f'###############################################################')
            print(f'#                                                             #')
            print(f'#                 Iteration {num_iter + 1}                              #')
            print(f'#                                                             #')
            print(f'###############################################################')
            selected_features_ = []
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

                X_train_single = X[selected_features_]
                
                fit_params = self.update_fit_params(X_val, y_val, sample_weight, selected_features_)

                model.fit(X=X_train_single, y=y, **fit_params)
                self.best_estimator_ = copy.deepcopy(model)
                self.features_selected = selected_features_

                score = self.score(X_test[selected_features_], y_test)
                single_feature_score = self.get_all_scores(X_test[selected_features_], y_test)
                print(f'All score achived wiht {fet} : {single_feature_score.to_dict()}')

                single_feature_score['Feature'] = fet
                all_score.append(single_feature_score)
                all_features.append(fet)
                if select:
                    if score <= base_score:
                        selected_features_.pop(-1)
                        c += 1
                    else:
                        print(f'With {fet} number {i}: {base_score} -> {single_feature_score}')
                        base_score = single_feature_score
                        score_.append(single_feature_score)
                        c = 0

                    if c > count_max:
                        print(f'Score didn t improove for {count_max} features, we break')
                        break

            if select:
                if base_score > iter_score:
                    final_score = pd.concat(score_).reset_index(drop=True)
                    iter_score = base_score
                    final_selected_features = list(final_score.Feature.values)
            else:
                iter_score = base_score
                final_selected_features = list(final_score['Feature'], values)
                final_score = pd.concat(score_).reset_index(drop=True)

        plt.figure(figsize=(15,10))
        x_score = np.arange(len(final_selected_features))
        plt.plot(x_score, final_score['iou'].values, label='iou')
        plt.plot(x_score, final_score['iou_wildfire_or_pred'], label='iou_wildfire_or_pred')
        plt.plot(x_score, final_score['iou_wildfire_detected'], label='wildfire_detected')
        plt.plot(x_score, final_score['iou_wildfire_and_pred'], label='iou_wildfire_and_pred')
        plt.plot(x_score, final_score['bad_prediction'], label='bad_prediction')
        plt.legend()
        plt.xticks(x_score, final_selected_features, rotation=90)
        plt.savefig(self.dir_log / f'scores_by_features.png')
        plt.close('all')

        save_object(final_score, f'scores_by_features.pkl', self.dir_log)

        return final_selected_features, final_score

    def update_fit_params(self, X_val, y_val, sample_weight, features_selected):
        if self.model_type == 'xgboost':
            fit_params = {
                'eval_set': [(X_val[features_selected], y_val)],
                'sample_weight': sample_weight,
                'verbose': False,
                #'early_stopping_rounds' : 15
            }

        elif self.model_type == 'catboost':
            fit_params = {
                'eval_set': [(X_val[features_selected], y_val)],
                'sample_weight': sample_weight,
                'verbose': False,
                'early_stopping_rounds': 15,
            }

        elif self.model_type == 'ngboost':
            fit_params = {
                'X_val': X_val[features_selected],
                'Y_val': y_val,
                'sample_weight': sample_weight,
                'early_stopping_rounds': 15,
            }

        elif self.model_type == 'rf':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'dt':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'lightgbm':
            fit_params = {
                'eval_set': [(X_val[features_selected], y_val)],
                'eval_sample_weight': [X_val['weight']],
                'early_stopping_rounds': 15,
                'verbose': False
            }

        elif self.model_type == 'svm':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'poisson':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type == 'gam':
            fit_params = {
            'weights': sample_weight
            } 

        elif self.model_type ==  'linear':
            fit_params = {}
        
        elif self.model_type ==  'lg':
            fit_params = {
                'sample_weight': sample_weight
            }

        elif self.model_type ==  'ordered':
            fit_params = {
                'disp':False,
                'method': 'bfgs'
            }

        else:
            raise ValueError(f"Unsupported model model_type: {self.model_type}")
        
        return fit_params

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

##########################################################################################
#                                                                                        #
#                                   Tree                                                 #
#                                                                                        #
##########################################################################################

class ModelTree(Model):
    def __init__(self, model, model_type, loss='logloss', name='ModelTree', under_sampling='full', target_name='nbsinister'):
        """
        Initialize the ModelTree class.

        Parameters:
        - model: The base model to use (must follow the sklearn API and support tree plotting).
        - name: The name of the model.
        - loss: Loss function to use ('logloss', 'hinge_loss', etc.).
        """
        super().__init__(model=model, model_type=model_type, loss=loss, name=name, under_sampling=under_sampling, target_name=target_name)

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
    def __init__(self, models, features, loss='mse', name='DualModel', dir_log=Path('../'), under_sampling='full', target_name='nbsinister', task_type='regression', post_process=None):
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
        self.model_positive = models[0] # Model for detecting positif sample
        self.model_all = models[1] # Model for predicting the risk class
        self.features = features
        self.name = name
        self.loss = loss
        self.dir_log = dir_log
        self.is_fitted_ = [False, False]  # Track the fitting status of both models
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type

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
            res = self.model_all.predict(X_positive)
            predictions[mask_positive] = res.reshape(-1)

        return predictions.reshape(-1)

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
    
    def predict_nbsinister(self, X, ids=None, preprocessor_ids=None):
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

    def predict_risk(self, X, ids=None, preprocessor_ids=None):
        """
        Predict the risk using the model.

        Parameters:
        - X: Input data for prediction.
        - ids: Optional identifiers for the data.

        Returns:
        - Predicted risk.
        """
        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            return self.predict(X)
        else:
            assert self.post_process is not None, "Post-process module is required for non-classification predictions."
            predictions = self.predict(X)
            return self.post_process.predict_risk(predictions, ids, preprocessor_ids)

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
        return iou_score(y, y_pred)
    
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
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), under_sampling='full', over_sampling='full', target_name='nbsinister', post_process=None, task_type='classification'):
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
        X_train = None
        y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.features_per_model = []
        self.dir_log = dir_log
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.target_name = target_name
        self.task_type = task_type

    def fit(self, X, y, X_val, y_val, X_test, y_test, training_mode='normal', optimization='skip', grid_params_list=None, fit_params_list=None, cv_folds=10, id_col=[]):
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
        self.weights_for_model = []
        self.weights_for_model_self = []
        self.weights_id_model = {}
        self.weights_id_model = {}
        if len(id_col) > 0:
            do_id_weight = True
            for id_tuple in id_col:
                id = id_tuple[0]
                vals = id_tuple[1]
                uvalues = np.unique(vals)
                self.weights_id_model[id] = {}
                for val in uvalues:
                    self.weights_id_model[id][val] = []
        else:
            do_id_weight = False

        targets = y.columns
        for i, model in enumerate(self.best_estimator_):
            model.dir_log = self.dir_log / '..' / model.name
            print(f'Fitting model -> {model.name}')
            model.fit(X, y[targets[i]], X_val, y_val[targets[i]], X_test, y_test[targets[i]], y_train_score=y[self.target_name], y_val_score=y_val[self.target_name], y_test_score=y_test[self.target_name], training_mode=training_mode, optimization=optimization, grid_params=grid_params_list[i], fit_params=fit_params_list[i], cv_folds=cv_folds)
            
            score_model = model.score(X_val, y_val[self.target_name])

            self.weights_for_model.append(score_model)
            print(f'Weight achieved : {score_model}')

            score_model = model.score(X_val, y_val[targets[i]])

            self.weights_for_model_self.append(score_model)
            print(f'Weight {targets[i]} achieved : {score_model}')

            if do_id_weight:
                for id_tuple in id_col:
                    id = id_tuple[0]
                    vals = id_tuple[1]
                    uvalues = np.unique(vals)
                    for val in uvalues:
                        mask = vals == val
                        score_model = model.score(X_val[mask], y_val[self.target_name][mask])
                        self.weights_id_model[id][val].append(score_model)
                    
                        #print(f'Weight {targets[i]} achieved on {id} {val}: {score_model}')
                    
        self.weights_for_model = np.asarray(self.weights_for_model)
        # Affichage des poids et des modèles
        print("\n--- Final Model Weights ---")
        for model, weight in zip(self.best_estimator_, self.weights_for_model):
            print(f"Model: {model.name}, Weight: {weight:.4f}")

        # Plot des poids des modèles
        model_names = [model.name for model in self.best_estimator_]
        plt.figure(figsize=(10, 10))
        plt.bar(model_names, self.weights_for_model, color='skyblue', edgecolor='black')
        plt.title('Model Weights', fontsize=16)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Weights', fontsize=14)
        plt.xticks(rotation=90, fontsize=12)  # Rotation de 90° ici
        plt.tight_layout()
        plt.savefig(self.dir_log / 'weights_of_models.png')
        plt.close('all')

        save_object([model_names, self.weights_for_model, self.weights_for_model_self], f'weights.pkl', self.dir_log)

    def predict_nbsinister(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True, top_model='all'):
        
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            assert self.post_process is not None
            predict = self.predict(X, hard_or_soft=hard_or_soft, weights_average=weights_average, top_model=top_model)
            return self.post_process.predict_nbsinister(predict, ids)
    
    def predict_risk(self, X, ids=None, preprocessor_ids=None, hard_or_soft='soft', weights_average=True, top_model='all'):

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            return self.predict(X, hard_or_soft=hard_or_soft, weights_average=weights_average, top_model=top_model)
        
        elif self.task_type == 'binary':
                assert self.post_process is not None
                predict = self.predict_proba(X, weights_average=weights_average)[:, 1]

                if isinstance(ids, pd.Series):
                    ids = ids.values
                if isinstance(preprocessor_ids, pd.Series):
                    preprocessor_ids = preprocessor_ids.values
    
                return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)
        else:
            assert self.post_process is not None
            predict = self.predict(X)

            if isinstance(ids, pd.Series):
                ids = ids.values
            if isinstance(preprocessor_ids, pd.Series):
                preprocessor_ids = preprocessor_ids.values

            return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)

    def predict_with_weight(self, X, hard_or_soft='soft', weights_average='weight', weights2use=[], top_model='all'):
        
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)
        
        if hard_or_soft == 'hard':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))

            models_to_mean = []
            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.name not in models_list:
                    continue
                else:
                    pred = estimator.predict(X)
                    predictions.append(pred)

                models_to_mean.append(key[i])

            try:
                weights2use = weights2use[models_to_mean]
            except:
                pass
            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights2use)
            return aggregated_pred
        else:
            aggregated_pred = self.predict_proba_with_weights(X, weights_average=weights_average, top_model=top_model, weights2use=weights2use)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions

    def predict_proba_with_weights(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', weights2use=[], id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """
        models_list = np.asarray([estimator.name for estimator in self.best_estimator_])
        weights2use = np.asarray(weights2use)

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(weights2use)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
                #weights2use = weights2use[np.asarray(key)]
                #weights2use = weights2use[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        probas = []
        models_to_mean = []
        for i, estimator in enumerate(self.best_estimator_):
            if estimator.name not in models_list:
                continue
            X_ = X
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_)
                if proba.shape[1] != 5:
                    continue
                #print(estimator.name, np.asarray(probas).shape)
                models_to_mean.append(key[i])
                probas.append(proba)
            else:
                raise AttributeError(f"The model {estimator.name} does not support predict_proba.")

        try:
            weights2use = weights2use[models_to_mean]
        except:
            pass
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights2use)
        return aggregated_proba

    def predict(self, X, hard_or_soft='soft', weights_average='weight', top_model='all', id_col=(None, None)):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask] = self.predict_with_weight(X[mask], hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model)
            return prediction

        else:
            return self.predict_with_weight(X, hard_or_soft=hard_or_soft, weights_average='weight', weights2use=self.weights_for_model, top_model=top_model)

        """print(f'Predict with {hard_or_soft} and weighs at {weights_average}')
        if hard_or_soft == 'hard':
            if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(self.weights_for_model)
                models_list = models_list[key]
                models_list = models_list[-top_model:]
            else:
                key = np.arange(0, len(self.best_estimator_))

            predictions = []
            for i, estimator in enumerate(self.best_estimator_):
                if estimator.name not in models_list:
                    continue
                else:
                    pred = estimator.predict(X)
                    predictions.append(pred)

                models_to_mean.append(key[i])

            # Aggregate predictions
            aggregated_pred = self.aggregate_predictions(predictions, models_to_mean, weights_average)
            return aggregated_pred
        else:
            aggregated_pred = self.predict_proba(X, weights_average, top_model)
            predictions = np.argmax(aggregated_pred, axis=1)
            return predictions"""

    def predict_proba(self, X, weights_average='weight', top_model='all', id_col=(None, None)):
        """
        Predict probabilities for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict probabilities for.
        
        Returns:
        - Aggregated predicted probabilities.
        """

        if weights_average not in ['None', 'weight']:
            assert id_col[0] is not None and id_col[1] is not None
            vals = id_col[1]
            unique_ids = np.unique(vals)
            prediction = np.empty(X.shape[0], dtype=int)
            for id in unique_ids:
                print(f'Prediction for {id_col[0]} {id}')
                mask = (id_col[1] == id)
                prediction[mask] = self.predict_proba_with_weights(X[mask], hard_or_soft='soft', weights_average='weight', weights2use=self.weights_id_model[id_col[0]][id], top_model=top_model)
            return prediction

        else:
            return self.predict_proba_with_weights(X, hard_or_soft='soft', weights_average='weight', weights2use=self.weights_for_model, top_model=top_model)

        """models_list = np.asarray([estimator.name for estimator in self.best_estimator_])

        if top_model != 'all':
                top_model = int(top_model)
                key = np.argsort(self.weights_for_model)
                models_list = models_list[np.asarray(key)]
                models_list = models_list[-top_model:]
        else:
            key = np.arange(0, len(self.best_estimator_))
        
        print(models_list)
        probas = []
        models_to_mean = []
        for i, estimator in enumerate(self.best_estimator_):
            if estimator.name not in models_list:
                continue
            X_ = X
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_)
                if proba.shape[1] != 5:
                    continue
                #print(estimator.name, np.asarray(probas).shape)
                models_to_mean.append(key[i])
                probas.append(proba)
            else:
                raise AttributeError(f"The model at index {i} does not support predict_proba.")
            
        # Aggregate probabilities
        aggregated_proba = self.aggregate_probabilities(probas, models_to_mean, weights_average)
        return aggregated_proba"""

    def aggregate_predictions(self, predictions_list, models_to_mean, weight2use=[], id_col=(None, None)):
        """
        Aggregate predictions from multiple models with weights.

        Parameters:
        - predictions_list: List of predictions from each model.

        Returns:
        - Aggregated predictions.
        """
        predictions_array = np.array(predictions_list)
        if len(weight2use) == 0 or weight2use is None:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            # Weighted vote for classification
            unique_classes = np.arange(0, 5)
            weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

            for i, cls in enumerate(unique_classes):
                mask = (predictions_array == cls)
                weighted_votes[i] = np.sum(mask * weight2use.reshape(mask.shape[0], 1), axis=0)

            aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        else:
            # Weighted average for regression
            weighted_sum = np.sum(predictions_array * weight2use[:, None], axis=0)
            aggregated_pred = weighted_sum / np.sum(weight2use)
            #aggregated_pred = np.max(predictions_array * weight2use[:, None], axis=0)
        
        return aggregated_pred

    """def aggregate_predictions_id(self, predictions_array, models_to_mean, id_col=(None, None)):
        assert id_col[0] is not None and id_col[1] is not None
        id = id_col[0]
        vals = id_col[1]
        uvals = np.unique(vals)

        weight2use = np.zeros((len(models_to_mean), predictions_array.shape[1]))
        for val in uvals:
            mask = (vals == val)
            weight2use[:, mask] = self.weights_id_model[id][val][models_to_mean]
            if np.all(weight2use[:, mask] == 0):
                weight2use[:, mask] = self.weights_for_model[models_to_mean]

        unique_classes = np.arange(0, 5)
        weighted_votes = np.zeros((len(unique_classes), predictions_array.shape[1]))

        for i, cls in enumerate(unique_classes):
            mask = (predictions_array == cls)
            weighted_votes[i] = np.sum(mask * weight2use, axis=0)

        aggregated_pred = unique_classes[np.argmax(weighted_votes, axis=0)]
        return aggregated_pred"""   

    def aggregate_probabilities(self, probas_list, models_to_mean, weight2use=[], id_col=(None, None)):
        """
        Aggregate probabilities from multiple models with weights.

        Parameters:
        - probas_list: List of probability predictions from each model.

        Returns:
        - Aggregated probabilities.
        """
        probas_array = np.array(probas_list)
        if weight2use is None or len(weight2use) == 0:
            weight2use = np.ones_like(self.weights_for_model)[models_to_mean]
        
        # Weighted average for probabilities
        weighted_sum = np.sum(probas_array * weight2use[:, None, None], axis=0)
        aggregated_proba = weighted_sum / np.sum(weight2use)
        #aggregated_proba = np.max(probas_array * weight2use[:, None, None], axis=0)
        return aggregated_proba
    
    """def aggregate_probabilities_id(self, probas_array, models_to_mean, id_col=(None, None)):
        assert id_col[0] is not None and id_col[1] is not None
        id = id_col[0]
        vals = id_col[1]
        uvals = np.unique(vals)
        weight2use = np.zeros((len(models_to_mean), probas_array.shape[1]))
        for val in uvals:
            mask = (vals == val)
            weight2use[:, mask] = self.weights_id_model[id][val][models_to_mean]
            if np.all(weight2use[:, mask] == 0):
                weight2use[:, mask] = self.weights_for_model[models_to_mean]

        #aggregated_proba = np.max(probas_array * weight2use[:, :, None], axis=0)
        weighted_sum = np.sum(probas_array * weight2use[:, None, None], axis=0)
        aggregated_proba = weighted_sum / np.sum(weight2use)
        return aggregated_proba"""

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
        
        return iou_score(y, y_pred)
    
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
    def __init__(self, models, features, loss='mse', name='ModelVoting', dir_log=Path('../'), under_sampling='full', target_name='nbsinister', post_process=None, task_type='classification'):
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
        X_train = None
        y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute
        self.is_fitted_ = [False] * len(models)  # Keep track of fitted models
        self.dir_log = dir_log
        self.post_process = post_process
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type

    def fit(self, X, y, X_val, y_val, X_test, y_test, training_mode='normal', optimization='skip', grid_params_list=None, fit_params_list=None, cv_folds=10):
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
        self.targets = y.columns
        orifeatures = list(X.columns)
        self.features_per_models = [orifeatures]

        for i, model in enumerate(self.best_estimator_):

            model.dir_log = self.dir_log / '..' / model.name
            print(f'Fitting model -> {model.name} -> {model.dir_log}')
            model.fit(X[self.features_per_models[i]], y[self.targets[i]], X_val, y_val[self.targets[i]], X_test, y_test[self.targets[i]], training_mode=training_mode, optimization=optimization, grid_params=grid_params_list[i], fit_params=fit_params_list[i], cv_folds=cv_folds)
            
            X[self.targets[i]] = y[self.targets[i]].values
            X_val[self.targets[i]] = y_val[self.targets[i]].values
            X_test[self.targets[i]] = y_test[self.targets[i]].values

            features = copy.deepcopy(orifeatures)
            features.append(self.targets[i])
            self.features_per_models.append(features)

        for i in range(len(self.features_per_models)):

            features = self.features_per_models[i]

            if 'weight' in features:
                features.remove('weight')
            if 'potential_risk' in features:
                features.remove('potential_risk')
            
            self.features_per_models[i] = features

    def predict_nbsinister(self, X, ids=None, preprocessor_ids=None):
        if self.target_name == 'nbsinister':
            return self.predict(X)
        else:
            assert self.post_process is not None
            predict = self.predict(X)
            return self.post_process.predict_nbsinister(predict, ids)
    
    def predict_risk(self, X, ids=None, preprocessor_ids=None):

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            return self.predict(X)
        
        elif self.task_type == 'binary':
                assert self.post_process is not None
                predict = self.predict_proba(X)[:, 1]

                if isinstance(ids, pd.Series):
                    ids = ids.values
                if isinstance(preprocessor_ids, pd.Series):
                    preprocessor_ids = preprocessor_ids.values
    
                return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)
        else:
            assert self.post_process is not None
            predict = self.predict(X)

            if isinstance(ids, pd.Series):
                ids = ids.values
            if isinstance(preprocessor_ids, pd.Series):
                preprocessor_ids = preprocessor_ids.values

            return self.post_process.predict_risk(predict, None, ids, preprocessor_ids)

    def predict(self, X):
        """
        Predict labels for input data using each model and aggregate the results.

        Parameters:
        - X_list: List of data to predict labels for.

        Returns:
        - Aggregated predicted labels.
        """
        for i, estimator in enumerate(self.best_estimator_):
            X_ = X[self.features_per_models[i]]
            pred = estimator.predict(X_)
            
            if i < len(self.best_estimator_) - 1:
                X[self.targets[i]] = pred

            if i == len(self.best_estimator_) -1:
                predictions = pred

        return predictions
    
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
        y = y[self.target_name]
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):

        return iou_score(y, y_pred)
    
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

######################################### Model OneById ###################################################

class OneByID(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, model_type, loss='mse', name='OneByID', col_id_name=None, id_train=None, id_val=None, id_test=None, dir_log = Path('./'), under_sampling='full', target_name='nbsinister', task_type='regression', post_process=None):
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
        self.under_sampling = under_sampling
        self.target_name = target_name
        self.task_type = task_type
        self.post_process = post_process

    def fit(self, X, y, X_val, y_val, X_test=None, y_test=None, features_search=False, optimization='skip', grid_params=None, fit_params=None):
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

            if self.id_val is not None:
                X_id_val = X_val[self.id_val == uid]
                y_id_val = y_val[self.id_val == uid]
            else:
                X_id_val = X_val
                y_id_val = y_val

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
            model.fit(X_id_train, y_id_train, X_val=X_id_val, y_val=y_id_val, X_test=X_id_test, y_test=y_id_test, features_search=doFeatures_search, optimization=optimization, grid_params=grid_params, fit_params=fit_params_model, cv_folds=10)

            self.models_by_id[uid] = copy.deepcopy(model)

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
    
    def predict_nbsinister(self, X, id_unique, ids=None, preprocessor_ids=None):
        
        if self.target_name == 'nbsinister':
            return self.predict(X, id_unique)
        else:
            assert self.post_process is not None
            predict = self.predict(X, id_unique)
            return self.post_process.predict_nbsinister(predict, ids)
    
    def predict_risk(self, X, id_unique, ids=None, preprocessor_ids=None):

        if self.task_type == 'classification' or self.task_type == 'ordinal-classification':
            return self.predict(X, id_unique)
        else:
            assert self.post_process is not None
            predict = self.predict(X, id_unique)
            return self.post_process.predict_risk(predict, ids, preprocessor_ids)

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
        return iou_score(y, y_pred)
    
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