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
        self.experiment_name = name
        self.dataset = dataset
        self.model = model
        self.dir = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/experiments/{self.experiment_name}')))
        os.makedirs(self.dir, exist_ok=True)
        self.logger = config.get('logger')
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        experiment = mlflow.set_experiment(self.experiment_name)
        # mlflow.create_experiment(self.experiment_name, artifact_location=self.dir)
        runs = mlflow.search_runs(experiment_names=[self.experiment_name])
        print(runs)
        self.runs = len(runs)


    def run(self, dataset_config, model_config, encoding_pipeline: Union[Pipeline, dict, None] = None) -> None:
        """
        Run the experiment.

        Parameters:
        - None
        """
        
        with mlflow.start_run(run_name='run_' + str(self.runs), ):
            self.logger.info("Running the experiment...")

            self.dataset.fetch_data()
            self.dataset.split(test_size=0.2, val_size=0.2)
            self.dataset.create_X_y()
            self.dataset.encode(pipeline=encoding_pipeline)
            mlflow.log_params(dataset_config)
            # mlflow.log_dict(self.dataset.config, "dataset_config")
            # mlflow.log_dict(model_config, "model_config")
            mlflow.log_params(model_config)
            # mlflow.doctor()
            self.fit(model_config)
            self.score()
            self.runs += 1

    def fit(self, model_config) -> None:
        """
        Train the model.

        Parameters:
        - None
        """
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

        self.model.fit(self.dataset.enc_X_train, self.dataset.y_train, optimization=optimization, grid_params=grid_params, fit_params=fit_params)
        mlflow.sklearn.log_model(self.model, "model_" + str(self.runs))
        print(self.model.get_params(deep=True))
        mlflow.log_params(self.model.get_params(deep=True))

    def predict(self) -> None:
        """
        Test the model.

        Parameters:
        - None
        """
        self.logger.info("Testing the model...")
        y_pred = self.model.predict(self.dataset.enc_X_test)
        return y_pred
    
    def score(self) -> None:
        """
        Score the model.

        Parameters:
        - None
        """
        self.logger.info("Scoring the model...")
        score = self.model.score(self.dataset.enc_X_test, self.dataset.y_test)
        scorer = self.model.get_scorer()
        mlflow.log_metric(scorer, score)
        return score