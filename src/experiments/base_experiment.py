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
from src.models.sklearn_models import Model, ModelTree
from src.datasets.base_tabular_dataset import BaseTabularDataset
from src.encoding.tools import create_encoding_pipeline
import src.features as ft
import mlflow.sklearn
import mlflow
import mlflow.data.pandas_dataset
import os
import matplotlib.pyplot as plt
import cudf as cd


class BaseExperiment:
    def __init__(self, name=None, dataset: Optional[BaseTabularDataset] = None, model: Optional[Union[ModelTree, List[ModelTree]]] = None, config: Optional[ft.Config] = None) -> None:
        self.experiment_name = 'exp_' + dataset.__class__.__name__ + '_' + '_'.join(dataset.targets) + '_' + model.name if name is None else name
        self.dataset = dataset
        self.model = model
        self.logger = config.get('logger')
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        # experiments = mlflow.search_experiments()
        # print(experiments)
        # exit()
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        self.dir_runs = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../mlruns/{self.experiment_id}')))
        os.makedirs(self.dir_runs, exist_ok=True)
        # self.dir_artifacts = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../mlruns/{self.experiment_id}/artifacts')))
        runs = mlflow.search_runs(experiment_names=[self.experiment_name])
        self.run_nb = len(runs)


    def run(self, dataset_config, model_config, encoding_pipeline: Union[Pipeline, dict, None] = None) -> None:
        """
        Run the experiment.

        Parameters:
        - None
        """
        
        with mlflow.start_run(run_name='run_' + str(self.run_nb), log_system_metrics=True) as run:

            run_dir = self.dir_runs / f'{run.info.run_id}'
            run_dir = pathlib.Path(run_dir)


            self.logger.info("Running the experiment...")


            dataset = self.dataset.get_dataset(**dataset_config)
            print(dataset.data.columns)
            dataset.split(test_size=0.2, val_size=0.2, shuffle=False)
            dataset.create_X_y()

            if encoding_pipeline is not None:
                if isinstance(encoding_pipeline, dict):
                    encoding_pipeline = create_encoding_pipeline(encoding_pipeline)
            
            dataset.encode(pipeline=encoding_pipeline)

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

            mlflow.log_params({f'grid_{key}': value for key, value in model_config['grid_params'].items()})
            # mlflow.log_params(model_config['params'])
            mlflow.log_params(model_config['fit_params'])
            mlflow.log_param('optimization', model_config['optimization'])

            # TODO: Certain fit_params doivent être initialisés après la création des datasets
            model_config['fit_params'].update({'eval_set': [(dataset.enc_X_val, dataset.y_val)]})
            self.fit(model_config, dataset)

            # self.model.plot_tree(dir_output=run_dir)

            # self.model.plot_param_influence(param='max_depth', dir_output=run_dir)

            # self.model.plot_features_importance(X_set=dataset.enc_X_test, y_set=dataset.y_test, outname='features_importance.png', dir_output=run_dir)

            # self.model.shapley_additive_explanation(df_set=cd.DataFrame(dataset.enc_X_test.join(dataset.y_test)), outname='shap.png', dir_output=run_dir)

            # print(self.model.get_params(deep=True))
            mlflow.log_params(self.model.get_params(deep=True))
            mlflow.sklearn.log_model(self.model, "model")
            
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

        y_pred.plot(ax=ax, label='Predicted', use_index=True)
        dataset.y_test.plot(ax=ax, label='True', use_index=True)
        ax.legend()

        return fig
        
    def fit(self, model_config, dataset: BaseTabularDataset) -> Model:
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

        # if 'params' in model_config:
        #     params = model_config['params']
        # else:
        #     params = {}

        self.model.fit(cd.DataFrame(dataset.enc_X_train), dataset.y_train, optimization=optimization, grid_params=grid_params, fit_params=fit_params)

        return self.model

    def predict(self, dataset: BaseTabularDataset) -> pd.DataFrame:
        """
        Test the model.

        Parameters:
        - None
        """
        self.logger.info("Testing the model...")
        y_pred = pd.DataFrame(self.model.predict(cd.DataFrame(dataset.enc_X_test)), index=dataset.y_test.index, columns=['y_pred'])
        
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