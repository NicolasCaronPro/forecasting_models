import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# +
# # %pip install -r ../requirements.txt
# -

from src.encoding.encoders import *
from src.encoding.tools import create_encoding_pipeline
from src.models.sklearn_models import save_object, Model
from src.models.sklearn_models_config import get_model
from src.datasets.base_tabular_dataset import BaseTabularDataset
from src.experiments.base_experiment import BaseExperiment
import src.features as ft
import logging
import pandas as pd
import pathlib

# Define a logger used by all modules
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    format="%(name)s %(asctime)s: %(levelname)s: %(message)s", handlers=[logging.StreamHandler()])

# Define the root directory of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
root_dir = pathlib.Path(root_dir)

# Define the configuration for the fetching of the data
fetch_config = {
    "data_start": '01-01-2017',
    "data_stop": '31-12-2023',
    'data_dir': root_dir / 'data',
    "etablissement": "CHU Dijon",
    "departement": "21",
    'region': 'BOURGOGNE'
    }

# Select the features to be used in the dataset
ars_features_class = [
    ft.HopitalFeatures,
    ft.AirQualityFeatures,
    ft.EpidemiologicalFeatures,
    # ft.FireFightersFeatures(include_calls=False),
    ft.GoogleTrendFeatures,
    ft.MeteorologicalFeatures,
    ft.SociologicalFeatures,
    ft.SportsCompetitionFeatures,
    ft.TrafficFeatures
    ]

# Select the target columns to be predicted
# target_colomns = ['nb_vers_hospit']
target_colomns = ['Total_CHU Dijon']

# Define an encoding scheme to create the encoding pipeline
encoders_dict = {
    'number': {
        'as_number': {
            'imputers': [imputers.SimpleImputer(strategy='mean')],
            'encoders': [
                ne.StandardScaler(),
            ]
        }
    },
    'category': {
        'as_category': {
            'imputers': [imputers.SimpleImputer(strategy='most_frequent')],
            'encoders': [
                ne.MultiTargetEncoder(drop_invariant=True, return_df=True),
            ]
        }
    },
    'datetime': {
        'as_number': {
            'imputers': [de.DateFeatureExtractor()],
            'encoders': [
                ne.CyclicalFeatures(drop_original=True)
            ]
        },
        'as_category': {
            'imputers': [de.DateFeatureExtractor(dtype='category')],
            'encoders': [
                ne.MultiTargetEncoder(drop_invariant=True, return_df=True),
            ]
        }
    }
}

# Create the encoding pipeline
pipeline = create_encoding_pipeline(encoders_dict)

# Define the splitting scheme to create the sets
split_config = {'test_size': 0.2, 'val_size': 0.2, 'shuffle': False}

# Define the configuration of the dataset
dataset_config = {
    'from_date': '15-01-2019',
    'to_date': '31-12-2023',
    'shift': range(1, 32, 1),
    'rolling_window': [7, 14, 31],
    'freq': '1D',
    'split_config': split_config,
    'create_X_y': True,
    'encoding_pipeline': pipeline,
    'targets_names': target_colomns,
    'targets_shift': -3,
    'targets_rolling_window': 3,
    'targets_history_shifts': range(1, 14, 1),
    'targets_history_rolling_windows': [7, 14],
    'drop_constant_thr':1.0,
    'data_dir': root_dir / 'data'
    }

# Create the dataset and fetch the data from the source then call get_dataset() method to fill the different attributes (X and y) of the different sets, and their encodings
arsTabularDataset = BaseTabularDataset(features_class=ars_features_class, logger=logger, fetch_config=fetch_config, getter_config=dataset_config)
# print(arsTabularDataset.data.columns.to_list())
# Define the model parameters
model_params = {
    'early_stopping_rounds': 10,
    # 'eval_set': [(arsTabularDataset.enc_X_val, arsTabularDataset.y_val)], # TODO: to be set in the experiment's run method
    'verbosity': 0,
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    # 'multi_strategy': 'one_output_per_tree',
    # 'multi_strategy': 'multi_output_tree' 
}

# Create the model
model = get_model(model_type='xgboost', name='XGBRegressor', device='cuda', task_type='regression', test_metrics='w_rmse', with_metric='w_rmse', params=model_params)

# Create the experiment
ars_experiment = BaseExperiment(logger=logger, dataset=arsTabularDataset, model=model)

# Set the model fitting config
grid_params = {
    'max_depth': [3, 5, 7, 9, 11],
}

fit_params = {
    'verbose': 0,
}

model_config={"optimization": "grid", "grid_params": grid_params, "fit_params": fit_params}

# Run the experiment
ars_experiment.run(dataset_config=dataset_config, model_config=model_config, find_best_features=True)