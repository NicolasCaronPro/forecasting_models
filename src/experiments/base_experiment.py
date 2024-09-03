"""
Eperiment class to run a single experiment with a given model (or ensemble of model) to train on a dataset, with a given config.
"""

import logging
import os
import sys
import datetime as dt
from typing import List, Union, Optional
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.models.sklearn_models import Model
from src.datasets.base_tabular_dataset import BaseTabularDataset
import src.features as ft
import mlflow.sklearn


class BaseExperiment:
    def __init__(self, name=None, dataset: Optional[BaseTabularDataset] = None, model: Optional[Union[Model, List[Model]]] = None, config: Optional[ft.Config] = None) -> None:
        self.experiment_name = 'exp_' + dataset.__class__.__name__ + '_' + '_'.join(dataset.targets) + '_' + model.name if name is None else name
        self.dataset = dataset
        self.model = model
        self.dir = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/experiments/{self.experiment_name}')))
        os.makedirs(self.dir, exist_ok=True)
        self.logger = config.get('logger')
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        experiment = mlflow.set_experiment(self.experiment_name)
        # mlflow.create_experiment(self.experiment_name, artifact_location=self.dir)
        runs = mlflow.search_runs(experiment_names=[self.experiment_name])
        self.runs = len(runs)


    def run(self, dataset_config, model_config, encoding_pipeline: Union[Pipeline, dict, None] = None) -> None:
        """
        Run the experiment.

        Parameters:
        - None
        """
        
        with mlflow.start_run(run_name='run_' + str(self.runs), ):
            self.logger.info("Running the experiment...")


            dataset = self.dataset.get_dataset(**dataset_config)
            dataset.split(test_size=0.2, val_size=0.2)
            dataset.create_X_y()
            dataset.encode(pipeline=encoding_pipeline)



            # TODO: Certain fit_params doivent être initialisés après la création des datasets
            model_config['fit_params'].update({'eval_set': [(dataset.enc_X_val, dataset.y_val)]})
            
            mlflow.log_params(dataset_config)
            # for key, value in dataset_config.items():
            #     mlflow.log_param(key, value)

            # mlflow.log_params(model_config)
            # for key, value in model_config.items():
            #     mlflow.log_param(key, value)
            # mlflow.doctor()

            self.fit(model_config, dataset)
            mlflow.log_params(self.model.get_params(deep=True))

            self.score(dataset)
            self.runs += 1

    def fit(self, model_config, dataset: BaseTabularDataset) -> None:
        """
        Train the model.

        Parameters:
        - None
        """

        # TODO: utiliser des kwargs pour les paramètres de model.fit
        self.logger.info("Training the model...")
        if 'optimization' in model_config:
            optimization = model_config['optimization']
        else:
            optimization = None
        
        if 'grid_params' in model_config:
            grid_params = model_config['grid_params']
        else:
            grid_params = None

        if 'fit_params' in model_config:
            fit_params = model_config['fit_params']
        else:
            fit_params = {}

        if 'params' in model_config:
            params = model_config['params']
        else:
            params = {}

        self.model.fit(dataset.enc_X_train, dataset.y_train, optimization=optimization, grid_params=grid_params, fit_params=fit_params, params=params)

    def predict(self, dataset: BaseTabularDataset) -> None:
        """
        Test the model.

        Parameters:
        - None
        """
        self.logger.info("Testing the model...")
        y_pred = self.model.predict(dataset.enc_X_test)
        return y_pred
    
    def score(self, dataset: BaseTabularDataset) -> None:
        """
        Score the model.

        Parameters:
        - None
        """
        self.logger.info("Scoring the model...")
        score = self.model.score(dataset.enc_X_test, dataset.y_test)
        scorer = self.model.get_scorer()
        mlflow.log_metric(scorer, score)
        return score