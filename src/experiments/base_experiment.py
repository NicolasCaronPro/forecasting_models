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
from src.datasets.base_tabular_dataset import BaseTabularDataset
from src.encoding.tools import create_encoding_pipeline
from src.experiments.features_selection import get_features, explore_features
from src.models.sklearn_models import Model, ModelTree
import src.features as ft
import mlflow.sklearn
import mlflow
import mlflow.data.pandas_dataset
from mlflow.models import infer_signature
import os
import matplotlib.pyplot as plt
import cudf as cd
import numpy as np


class BaseExperiment:
    def __init__(self, name=None, dataset: Optional[BaseTabularDataset] = None, model: Optional[Union[ModelTree, List[ModelTree]]] = None, config: Optional[ft.Config] = None) -> None:
        self.experiment_name = 'exp_' + dataset.__class__.__name__ + '_' + '_'.join(dataset.targets) + '_' + model.name if name is None else name
        self.dataset = dataset
        self.model = model
        self.logger: logging.Logger = config.get('logger')
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        # experiments = mlflow.search_experiments()
        # print(experiments)
        # exit()
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        self.dir_runs = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../mlartifacts/{self.experiment_id}')))
        os.makedirs(self.dir_runs, exist_ok=True)
        # self.dir_artifacts = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../mlruns/{self.experiment_id}/artifacts')))
        runs = mlflow.search_runs(experiment_names=[self.experiment_name])
        self.run_nb = len(runs)


    def run(self, dataset_config: dict, model_config: dict, encoding_pipeline: Union[Pipeline, dict, None] = None, find_best_features: bool = False) -> None:
        """
        Run the experiment.

        Parameters:
        - None
        """
        
        with mlflow.start_run(run_name='run_' + str(self.run_nb), log_system_metrics=True) as run:

            run_dir = self.dir_runs / f'{run.info.run_id}/artifacts/'
            run_dir = pathlib.Path(run_dir)


            self.logger.info("Running the experiment...")

            # Get a dataset object corresponding to the dataset_config
            dataset: BaseTabularDataset = self.dataset.get_dataset(**dataset_config)
            # dataset.plot(freq='1D', max_subplots=16)

            # Encode the dataset if needed
            if encoding_pipeline is not None:
                if isinstance(encoding_pipeline, dict):
                    encoding_pipeline = create_encoding_pipeline(encoding_pipeline)
            
                dataset.encode(pipeline=encoding_pipeline)

            
            # TODO: Remove this line, it's just for testing
            # return dataset

            if find_best_features:
                selected_features = self.get_important_features(dataset=dataset, model=self.model)
                dataset: BaseTabularDataset = dataset.get_dataset(features_names=selected_features)

            mlflow.log_table(data=dataset.train_set, artifact_file='datasets/train_set.json')
            mlflow.log_table(data=dataset.val_set, artifact_file='datasets/val_set.json')
            mlflow.log_table(data=dataset.test_set, artifact_file='datasets/test_set.json')

            train_dataset = mlflow.data.pandas_dataset.from_pandas(dataset.train_set)
            val_dataset = mlflow.data.pandas_dataset.from_pandas(dataset.val_set)
            test_dataset = mlflow.data.pandas_dataset.from_pandas(dataset.test_set)

            mlflow.log_input(dataset=train_dataset, context='training')
            mlflow.log_input(dataset=val_dataset, context='validation')
            mlflow.log_input(dataset=test_dataset, context='testing')
            
            mlflow.log_params(dataset_config)


            # TODO: Certain fit_params doivent être initialisés après la création des datasets : eval_set
            model_config['fit_params'].update({'eval_set': [(dataset.enc_X_val, dataset.y_val[target]) for target in dataset.targets]})

            mlflow.log_params({f'grid_{key}': value for key, value in model_config['grid_params'].items()})
            # mlflow.log_params(model_config['params'])
            mlflow.log_params(model_config['fit_params'])
            mlflow.log_param('optimization', model_config['optimization'])

            self.model.fit(cd.DataFrame(dataset.enc_X_train), dataset.y_train, **model_config)

            # self.model.plot_tree(dir_output=run_dir)

            # self.model.plot_param_influence(param='max_depth', dir_output=run_dir)

            # self.model.plot_features_importance(X_set=dataset.enc_X_test, y_set=dataset.y_test, outname='features_importance.png', dir_output=run_dir)

            # self.model.shapley_additive_explanation(df_set=cd.DataFrame(dataset.enc_X_test.join(dataset.y_test)), outname='shap.png', dir_output=run_dir)

            # print(self.model.get_params(deep=True))
            params = self.model.get_params(deep=True)

            signature = infer_signature(dataset.enc_X_test, self.model.predict(dataset.enc_X_test))
            
            mlflow.log_params(params=params)
            mlflow.sklearn.log_model(self.model, "model", signature=signature)
            
            scores = self.score(dataset)

            # mlflow.log_metrics(scores)
            mlflow.log_metric(self.model.get_scorer(), scores)

            y_pred = self.predict(dataset)
            figure = self.plot(dataset, y_pred)

            mlflow.log_figure(figure, 'predictions.png')

            self.run_nb += 1
    
    def plot(self, dataset: BaseTabularDataset, y_pred: pd.DataFrame) -> plt.Figure:
        """
        Plot the results.

        Parameters:
        - None
        """
        self.logger.info("Plotting the results...")

        fig, ax = plt.subplots()
        ax.set_title('True vs Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')

        dataset.y_test.plot(ax=ax, label='True', use_index=True)
        y_pred.plot(ax=ax, label='Predicted', use_index=True)
        ax.legend()

        return fig
    
    def get_important_features(self, dataset: BaseTabularDataset = None, model:Model = None, preselection: List[str] = []) -> List[str]:
        selected_features = []

        if dataset is None:
            dataset = self.dataset

        if model is None:
            model = self.model

        variables = dataset.get_features_names()
        targets = dataset.targets

        encoded_data = dataset.enc_data

        data = pd.concat([encoded_data, dataset.data[targets]], axis=1)

        for target in targets:
            # print(data, variables, target)
            important_features = get_features(data, variables, target, logger=self.logger, num_feats=100)

            # On transforme le dictionaire de tupples en dictionaire de listes de features importantes
            features_to_test = [f[0] for f in important_features]
            print(features_to_test)

            selected_features.extend([item for item in explore_features(model=model, features=features_to_test,
                             df_train=pd.concat([dataset.enc_X_train, dataset.y_train], axis=1),
                             df_val=pd.concat([dataset.enc_X_val, dataset.y_val], axis=1),
                             df_test=pd.concat([dataset.enc_X_test, dataset.y_test], axis=1),
                             target=target,preselection=preselection, logger=self.logger) if item not in selected_features])

        return selected_features

    def predict(self, dataset: BaseTabularDataset) -> pd.DataFrame:
        """
        Test the model.

        Parameters:
        - None
        """
        self.logger.info("Testing the model...")
        y_pred = pd.DataFrame(self.model.predict(cd.DataFrame(dataset.enc_X_test)), index=dataset.y_test.index, columns=[f'y_pred_{target}' for target in dataset.targets])
        
        return y_pred
    
    def score(self, dataset: BaseTabularDataset) -> None:
        """
        Score the model.

        Parameters:
        - None
        """
        self.logger.info("Scoring the model...")

        scores = self.model.score(dataset.enc_X_test, dataset.y_test)
        # scorer = self.model.get_scorer()

        return scores