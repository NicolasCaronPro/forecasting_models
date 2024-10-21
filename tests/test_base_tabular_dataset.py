import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.encoding.encoders import *
from src.encoding.tools import create_encoding_pipeline
import pandas as pd
import src.features as ft
from src.datasets.base_tabular_dataset import BaseTabularDataset
import datetime as dt
import logging
import pathlib
pd.set_option('display.max_columns', None)

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
    'shift': range(1, 14, 1),
    'rolling_window': [7, 14],
    'freq': '1D',
    'split_config': split_config,
    'create_X_y': True,
    'encoding_pipeline': pipeline,
    'targets_names': target_colomns,
    'targets_shift': -7,
    'targets_rolling_window': 7,
    'targets_history_shifts': range(1, 14, 1),
    'targets_history_rolling_windows': [7, 14],
    'drop_constant_thr':1.0,
    'data_dir': root_dir / 'data'
    }

baseTabularDataset = BaseTabularDataset(features_class=ars_features_class, logger=logger, fetch_config=fetch_config, getter_config=dataset_config)
# print(baseTabularDataset.data.columns.to_list())
# liste = ['O3_FR26005%%mean_7J', 'target_Total_CHU Dijon']
# baseTabularDataset = baseTabularDataset.get_dataset(**dataset_config, features_names=liste)
baseTabularDataset.train_set.to_csv('train.csv', index=False)
baseTabularDataset.test_set.to_csv('test.csv', index=False)
baseTabularDataset.val_set.to_csv('val.csv', index=False)

baseTabularDataset.enc_X_train.to_csv('enc_X_train.csv', index=False)
baseTabularDataset.enc_X_test.to_csv('enc_X_test.csv', index=False)
baseTabularDataset.enc_X_val.to_csv('enc_X_val.csv', index=False)
# baseTabularDataset.fetch_data(save=False)
# dataset = baseTabularDataset.get_dataset(from_date=dt.datetime.strptime('15-01-2019', '%d-%m-%Y'), to_date=dt.datetime.strptime('30-12-2023', '%d-%m-%Y'), shift=[1, 2, 3, 4, 5, 6, 7], rolling_window=[7, 14], create_X_y=True, split_config = {'test_size': 0.2, 'val_size': 0.2, 'shuffle': False}, encoding_pipeline=pipeline, freq='1D', targets_names=target_colomns)

# print(dataset.enc_X_train.columns.to_list())
# print(dataset.y_train.columns.to_list())
# dataset.save_dataset()
# dataset2 = dataset.get_dataset(features_names=['O3_FR26005%%J-1', 'confinement1%%J-5##Total_CHU_Dijon'])
# print(dataset2.data.columns.to_list())
# print(dataset2.train_set.columns.to_list())
# print(dataset2.X_train.columns.to_list())
# print(dataset2.y_train.columns.to_list())
# print(dataset2.enc_X_train.columns.to_list())

# dataset.plot(max_subplots=16)
# print(dataset.data)
# dataset.save_data(root_dir, "data")

# baseTabularDataset.plot(from_date='01-01-2022', to_date='15-01-2022', freq='1D', max_subplots=4)
# dataset.encode(pipeline=piepline)

# dataset.save_data(root_dir, "data")


# print(baseTabularDataset.data.info())
# baseTabularDataset.encode(encoders=encoders)

# print(baseTabularDataset.encoded_data)
# print(baseTabularDataset.features)
