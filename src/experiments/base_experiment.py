"""
Eperiment class to run a single experiment with a given model (or ensemble of model) to train on a dataset, with a given config.
"""

import logging
import os
import sys
import datetime as dt
from typing import List, Union, Optional
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.datasets.base_tabular_dataset import BaseTabularDataset
from src.encoding.tools import create_encoding_pipeline
from src.experiments.features_selection import get_features, explore_features
from src.models.sklearn_api_model import Model, ModelTree
import src.features as ft
import mlflow.sklearn
import mlflow
import mlflow.data.pandas_dataset
from mlflow.models import infer_signature
import os
import matplotlib.pyplot as plt
import pandas as pd
try:
    import cudf as cd
    USE_CUDA = True
except ImportError as e:
    USE_CUDA = False

import numpy as np
import re


class BaseExperiment:
    def __init__(self, logger, name=None, dataset: Optional[BaseTabularDataset] = None, model: Optional[Union[ModelTree, List[ModelTree]]] = None) -> None:
        self.experiment_name = '_'.join(
            dataset.targets_names) if name is None else name
        self.dataset = dataset
        self.model = model
        self.logger: logging.Logger = logger
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        # experiments = mlflow.search_experiments()
        # print(experiments)
        # exit()
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        self.dir_runs = pathlib.Path(os.path.abspath(os.path.join(
            os.path.dirname(__file__), f'../../mlartifacts/{self.experiment_id}')))
        os.makedirs(self.dir_runs, exist_ok=True)
        # self.dir_artifacts = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../mlruns/{self.experiment_id}/artifacts')))
        runs = mlflow.search_runs(experiment_names=[self.experiment_name])
        self.run_nb = len(runs)

    def run(self, dataset_config: dict, model_config: dict, find_best_features: bool = False, int_pred: bool = False, balance_target=False) -> None:
        """
        Run the experiment.

        Parameters:
        - None
        """

        with mlflow.start_run(run_name='run_' + str(self.run_nb), log_system_metrics=True) as run:

            run_dir = self.dir_runs / f'{run.info.run_id}/artifacts/'
            run_dir = pathlib.Path(run_dir)

            self.logger.info(f"Running the experiment on {'GPU' if USE_CUDA else 'CPU'}...")

            # Get a dataset object corresponding to the dataset_config
            # dataset: BaseTabularDataset = self.dataset.get_dataset(**dataset_config)
            # dataset = self.dataset

            # TODO: Certain fit_params doivent être initialisés après la création des datasets : eval_set
            model_config['fit_params'].update({'eval_set': [(
                self.dataset.enc_X_val, self.dataset.y_val[target]) for target in self.dataset.targets_names]})

            # print(dataset.data[['target_Total_CHU Dijon%%mean_7J', 'Total_CHU Dijon%%mean_7J']])
            mlflow.log_table(data=self.dataset.data,
                             artifact_file='datasets/full_dataset.json')
            if find_best_features:
                # selected_features = self.get_important_features(dataset=self.dataset, model=self.model, model_config=model_config)
                selected_features = ['nb_emmergencies%%J-7', 'nb_emmergencies%%J-1', 'nb_emmergencies%%J-2', 'nb_emmergencies%%J-3', 'nb_emmergencies', 'NO2_FR26094%%mean_7J', 'nb_emmergencies%%mean_365J', 'eveBankHolidays', 'meteo_wdir%%J-7', 'confinement1', 'trend_grippe%%mean_7J', 'trend_hopital%%J-3', 'trend_vaccin%%J-2', 'inc_diarrhee%%J-7', 'PM25_FR26094%%J-7',
                                     'trend_crampes abdominales%%J-7', 'trend_médecin', 'trend_crampes abdominales%%mean_7J', 'confinement2', 'NO2_FR26010', 'trend_hopital%%J-2', 'trend_mal de tête%%mean_7J', 'trend_paralysie%%J-7', 'trend_accident de voiture%%mean_7J', 'trend_paralysie%%mean_7J', 'meteo_tavg%%mean_7J', 'trend_insuffisance cardiaque', 'trend_fièvre%%J-7', 'trend_infection respiratoire%%mean_7J']
                # selected_features.extend(['PM10_FR26005%%mean_31J', 'foot%%std_14J', 'inc_ira%%mean_31J',
                #                      'meteo_tmin%%mean_31J', 'trend_vaccin%%mean_31J', 'confinement2',
                #                      'meteo_tmax%%mean_31J', 'after_HNFC_moving', 'trend_vaccin%%mean_14J',
                #                      'trend_hopital%%mean_31J', 'trend_hopital%%mean_14J', 'date##week_cos',
                #                      'O3_FR26010%%mean_31J', 'O3_FR26005%%mean_31J', 'meteo_tavg%%mean_31J',
                #                      'inc_grippe%%mean_31J', 'inc_grippe%%mean_14J', 'date##week_sin',
                #                      'date##dayofYear_sin', 'confinement1'])

                if dataset_config['axis'] == 'columns':
                    selected_features = [
                        feat + '_CHU Dijon' for feat in selected_features]
                else:
                    selected_features.append('location')
                # selected_features = dataset.enc_X_train.columns.to_list()
                dataset_config['features_names'] = selected_features
                self.logger.info(
                    'Features selected: {}'.format(selected_features))
                # print(dataset.y_train)
                self.dataset.get_dataset(**dataset_config)
                mlflow.log_table(data=self.dataset.data,
                                 artifact_file='datasets/full_dataset_feature_selection.csv')
                model_config['fit_params']['eval_set'] = [
                    (self.dataset.enc_X_val, self.dataset.y_val[target]) for target in self.dataset.targets_names]

            mlflow.log_table(data=self.dataset.train_set,
                             artifact_file='datasets/train_set.json')
            mlflow.log_table(data=self.dataset.val_set,
                             artifact_file='datasets/val_set.json')
            mlflow.log_table(data=self.dataset.test_set,
                             artifact_file='datasets/test_set.json')

            train_dataset = mlflow.data.pandas_dataset.from_pandas(
                self.dataset.train_set)
            val_dataset = mlflow.data.pandas_dataset.from_pandas(
                self.dataset.val_set)
            test_dataset = mlflow.data.pandas_dataset.from_pandas(
                self.dataset.test_set)

            mlflow.log_input(dataset=train_dataset, context='training')
            mlflow.log_input(dataset=val_dataset, context='validation')
            mlflow.log_input(dataset=test_dataset, context='testing')

            dataset_config_log = dataset_config.copy()
            dataset_config_log['locations'] = [
                loc.name for loc in dataset_config_log.pop('locations')]
            dataset_config_log['targets_locations'] = [
                loc.name for loc in dataset_config_log.pop('targets_locations')]
            mlflow.log_params(dataset_config_log)

            mlflow.log_params({f'grid_{key}': value for key,
                              value in model_config['grid_params'].items()})
            # mlflow.log_params(model_config['params'])
            mlflow.log_params(model_config['fit_params'])
            mlflow.log_param('optimization', model_config['optimization'])

            # balance training set
            if balance_target:
                # Combine x_train and y_train
                combined = pd.concat(
                    [self.dataset.enc_X_train, self.dataset.y_train], axis=1)

                # find majority and minority classes
                # Count the occurrences of each category
                category_counts = self.dataset.y_train[self.dataset.targets_names[0]].value_counts(
                )

                # Identify majority and minority categories
                # Category with the most occurrences
                majority_category = category_counts.idxmax()
                # Category with the least occurrences
                minority_category = category_counts.idxmin()

                # Separate majority and minority classes
                majority = combined[combined[self.dataset.targets_names[0]]
                                    == majority_category]
                minority = combined[combined[self.dataset.targets_names[0]]
                                    == minority_category]

                # Oversample minority class
                minority_oversampled = resample(minority,
                                                replace=True,    # Sample with replacement
                                                # Match number of majority
                                                n_samples=len(majority),
                                                random_state=42)  # Reproducibility

                # Combine back the oversampled minority class with the majority class
                print('Before:', self.dataset.enc_X_train.shape)
                balanced = pd.concat([majority, minority_oversampled])
                print('After:', balanced.shape)

                # Split back to self.dataset.enc_X_train and self.dataset.y_train
                self.dataset.enc_X_train = balanced.drop(
                    columns=[self.dataset.targets_names[0]])
                self.dataset.y_train = balanced[self.dataset.targets_names[0]]

            self.model.fit(pd.DataFrame(self.dataset.enc_X_train),
                           self.dataset.y_train, **model_config)
            self.logger.info("Model fitted.")

            # self.model.plot_tree(dir_output=run_dir)

            # self.model.plot_param_influence(param='max_depth', dir_output=run_dir)

            # self.model.plot_features_importance(X_set=dataset.enc_X_test, y_set=dataset.y_test, outname='features_importance.png', dir_output=run_dir)

            # self.model.shapley_additive_explanation(df_set=cd.DataFrame(dataset.enc_X_test.join(dataset.y_test)), outname='shap.png', dir_output=run_dir)

            # print(self.model.get_params(deep=True))
            params = self.model.get_params(deep=True)
            if params['objective'] is not None:
                # Check if objective is a function
                if callable(params['objective']):
                    params['objective'] = params['objective'].__name__

            if params['eval_metric'] is not None:
                if callable(params['eval_metric']):
                    params['eval_metric'] = params['eval_metric'].__name__
                else:
                    params['eval_metric'] = params['eval_metric']

            y_pred = self.predict(self.dataset)
            mlflow.log_table(data=y_pred, artifact_file='datasets/pred.json')

            signature = infer_signature(self.dataset.enc_X_test, y_pred)

            mlflow.log_params(params=params)
            mlflow.sklearn.log_model(self.model, "model", signature=signature)

            scores = self.score(self.dataset)
            print(scores)

            mlflow.log_metrics(scores)
            # mlflow.log_metric(self.model.get_scorer(), scores)

            y_pred = self.predict(self.dataset)
            if int_pred:
                y_pred[f'y_pred_{self.dataset.targets_names[0]}'] = y_pred[f'y_pred_{self.dataset.targets_names[0]}'].round(
                )
            # y_pred = self.predict_at_horizon(dataset, horizon=7)
            figure = self.plot(self.dataset, y_pred, scores)
            mlflow.log_figure(figure, 'predictions.png')

            error_fig = self.model.get_prediction_error_display(
                y=self.dataset.y_test, y_pred=y_pred)
            mlflow.log_figure(error_fig, 'errors.png')

            self.run_nb += 1

    def get_bjml(self) -> pd.DataFrame:
        df = pd.read_csv(
            '/home/maxime/Documents/WORKSPACES/forecasting_models/bjml.csv', sep=';')
        # Define a function to get the start (Monday) of each ISO week

        def get_start_of_week(year, week):
            # Use pd.Timestamp and isocalendar to get the first date of each ISO week in 2022
            first_day_of_year = pd.Timestamp(f'{year}-01-01')
            # Find the Monday of that week
            start_of_week = first_day_of_year + \
                pd.offsets.Week(weekday=0) * (week - 1)
            return start_of_week

        # Add a column 'Start Date' to df by calculating the start (Monday) of each week in 2022
        df['Start Date'] = df['Semaine'].apply(
            lambda x: get_start_of_week(2022, x))

        # Create a DataFrame with daily data by expanding each week into daily rows
        daily_df = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        })

        # Merge weekly data with daily data by matching each date to its corresponding ISO week
        daily_df['Semaine'] = daily_df['date'].apply(
            lambda x: x.isocalendar()[1])

        # Merge daily DataFrame with weekly values
        daily_df = pd.merge(daily_df, df[[
                            'Semaine', 'Min.', 'BJML', 'Méd.', 'P75', 'Max.']], on='Semaine', how='left')

        # Now daily_df contains a row for each day of 2022 with the corresponding weekly value
        daily_df.drop('Semaine', axis=1, inplace=True)
        daily_df.set_index('date', inplace=True)
        return daily_df

    def plot(self, dataset: BaseTabularDataset, y_pred: pd.DataFrame, scores: dict) -> plt.Figure:
        """
        Plot the results.

        Parameters:
        - None
        """
        self.logger.info("Plotting the results...")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('True vs Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.text(0.5, 0.5, str(scores))
        ax.text(3, 5, 'Ceci est un texte', fontsize=15, color='red')

        dataset.y_test.plot(ax=ax, label='True', use_index=True)
        y_pred.plot(ax=ax, label='Predicted', use_index=True)

        errors = pd.DataFrame(
            dataset.y_test.iloc[:, 0] - y_pred.iloc[:, 0], columns=['Error'])
        errors.plot(ax=ax, label='Error (target - y_pred)', use_index=True)

        if 'target_nb_vers_hospit' in self.dataset.targets_names and dataset.y_test.index[0].year == 2022:
            bjml = self.get_bjml()
            bjml.plot(ax=ax, label='BJML', use_index=True)

        return fig, ax

    def get_important_features(self, dataset: BaseTabularDataset = None, model: Model = None, preselection: List[str] = [], model_config: dict = None) -> List[str]:
        selected_features = []

        if dataset is None:
            dataset = self.dataset

        if model is None:
            model = self.model

        variables = dataset.enc_data.columns.to_list()
        targets = dataset.targets_names

        encoded_data = dataset.enc_data

        data = pd.concat([encoded_data, dataset.data[targets]], axis=1)

        # afficher le nombre de colonnes portant le même nom si il est supérieur à 1
        # print(data.columns.value_counts().loc[lambda x: x > 1])

        # TODO: ne marchera pas pour du multi-target car le modèle passé à explore_features attendra plusieurs targets
        for target in targets:
            # print(data, variables, target)
            important_features = get_features(
                data, variables, target, logger=self.logger, num_feats=int(10 + 0.1*len(variables)))

            # On transforme le dictionaire de tupples en dictionaire de listes de features importantes
            features_to_test = [f[0] for f in important_features]
            # print(features_to_test)

            selected_features.extend([item for item in explore_features(model=model, model_config=model_config, features=features_to_test,
                                                                        df_train=pd.concat(
                                                                            [dataset.enc_X_train, dataset.y_train], axis=1),
                                                                        df_val=pd.concat(
                                                                            [dataset.enc_X_val, dataset.y_val], axis=1),
                                                                        df_test=pd.concat(
                                                                            [dataset.enc_X_test, dataset.y_test], axis=1),
                                                                        target=target, preselection=preselection, logger=self.logger) if item not in selected_features])

        return selected_features

    def predict(self, dataset: BaseTabularDataset) -> pd.DataFrame:
        """
        Test the model.

        Parameters:
        - None
        """
        self.logger.info("Testing the model...")

        y_pred = pd.DataFrame(self.model.predict(cd.DataFrame(dataset.enc_X_test) if USE_CUDA else dataset.enc_X_test), index=dataset.y_test.index, columns=[
                              f'y_pred_{target}' for target in dataset.targets_names])

        return y_pred

    def predict_at_horizon(self, dataset: BaseTabularDataset, horizon: int = 1):
        """
        Fonction qui prédit les valeurs de la colonne target pour chaque groupe de n jours.
        Le premier jour d'un groupe est prédit avec les vraies données,
        les jours suivants sont prédit avec des dépendances sur les prédictions précédentes.

        Parameters:
        df (DataFrame): Dataset contenant les colonnes features et target.
        horizon (int): Taille du groupe de jours.

        Returns:
        DataFrame: Un nouveau dataframe avec les colonnes 'target_pred' et les prédictions.
        """
        df = pd.DataFrame(dataset.enc_X_test)
        predictions = pd.DataFrame(index=dataset.y_test.index)

        # Extraire les colonnes qui correspondent à des moyennes/écarts-types sur fenêtres mobiles
        rolling_window_cols = [
            col for col in df.columns if re.search(r'%%(mean|std)_(\d+)J', col)]
        rolling_window_sizes = {
            col: int(re.search(r'_(\d+)J', col).group(1)) for col in rolling_window_cols}

        # Parcourir le dataset en groupes de n jours
        for i in range(0, len(df), horizon):
            # Définir les limites du groupe de n jours
            # Copier le groupe pour éviter de modifier df
            groupe: pd.DataFrame = df.iloc[i:i+horizon].copy()

            for j in range(horizon):
                # Si c'est le premier jour, utiliser les vraies données
                # Si c'est un jour suivant, utiliser les prédictions précédentes

                if j > 0:
                    # Pour les jours suivants, remplacer les colonnes target%%J-1, target%%J-2 par les prédictions
                    for k in range(1, j+1):
                        for target in dataset.targets_names:
                            # target = target.replace('target_', '')
                            colonne_a_remplacer = f'{target}%%J-{k}'
                            # Simuler l'existence des colonnes target%%J-k en utilisant les features (ajuster selon ton dataset réel)
                            if colonne_a_remplacer in groupe.columns:
                                self.logger.info(
                                    f"Remplacement de la colonne {colonne_a_remplacer} par la prédiction {f'y_pred_{target}'}")
                                groupe[colonne_a_remplacer].iloc[
                                    j] = groupe[f'y_pred_{target}'].iloc[j-k]

                    # TODO: aussi recalculer les colonnes features%%mean_nJ, features%%std_nJ, etc. si elles existent

                    # Recalculer dynamiquement les colonnes basées sur des rolling windows (moyenne, écart-type, etc.)
                    for feature in dataset.features:
                        for col in rolling_window_cols:
                            if feature in col:
                                window_size = rolling_window_sizes[col]

                                # Vérifier si la taille de la fenêtre est suffisante pour calculer la moyenne/l'écart-type
                                if j >= window_size:
                                    # Recalcul des moyennes/écarts-types
                                    if 'mean' in col:
                                        self.logger.info(
                                            f"Recalcul de {col} (mean sur {window_size} jours) pour le jour {j}")
                                        groupe[col].iloc[j] = groupe[feature].iloc[j -
                                                                                   window_size:j].mean()

                                    if 'std' in col:
                                        self.logger.info(
                                            f"Recalcul de {col} (std sur {window_size} jours) pour le jour {j}")
                                        groupe[col].iloc[j] = groupe[feature].iloc[j -
                                                                                   window_size:j].std()

                features = groupe.iloc[j]
                print(features)
                pred = pd.DataFrame(self.model.predict(features), index=dataset.y_test.index, columns=[
                                    f'y_pred_{target}' for target in dataset.targets])

                predictions = pd.concat([predictions, pred], axis=0)

            # # Mettre à jour les prédictions dans le dataset d'origine
            # df['target_pred'].iloc[i:i+horizon] = groupe['target_pred']

        return predictions

    def score(self, dataset: BaseTabularDataset) -> None:
        """
        Score the model.

        Parameters:
        - None
        """
        self.logger.info("Scoring the model...")

        scores = self.model.score(
            dataset.enc_X_test, dataset.y_test, single_score=False)
        # scorer = self.model.get_scorer()

        return scores
