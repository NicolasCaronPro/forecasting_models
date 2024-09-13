import datetime as dt
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.datasets.base_tabular_dataset import BaseTabularDataset
import src.features as ft
import pandas as pd
from src.encoding.tools import create_encoding_pipeline
from src.encoding.encoders import *
pd.set_option('display.max_columns', None)


# ars_features_class = ["HopitalFeatures"]

# encoders = {'datetime64': [CyclicalFeatures, ce.TargetEncoder], 'category': [ce.TargetEncoder], 'number': [StandardScaler]}

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    format="%(name)s %(asctime)s: %(levelname)s: %(message)s", handlers=[logging.StreamHandler()])
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config = ft.Config({'max_nan': 0, "departement": "21", "root_dir": root_dir, "start": '01-01-2019',
                    "stop": '31-12-2023', "logger": logger, "step_unit": 'days', "step_value": 1,
                    "shift": 0, "rolling_window": 0, "etablissement": "CHU Dijon", 'region':'BOURGOGNE'})

ars_features_class = [ft.AirQualityFeatures, ft.HopitalFeatures(config=config, include_emmergency_arrivals=True, include_nb_hospit=True), ft.EpidemiologicalFeatures, ft.FireFightersFeatures(config=config, include_calls=False),
                      ft.GoogleTrendFeatures, ft.MeteorologicalFeatures, ft.SociologicalFeatures,
                      ft.SportsCompetitionFeatures, ft.TrafficFeatures]
baseTabularDataset = BaseTabularDataset(target_colomns='Total_CHU Dijon',
                                    config=config, features_class=ars_features_class)

baseTabularDataset.fetch_data()
# print(baseTabularDataset.data)

dataset = baseTabularDataset.get_dataset(from_date=dt.datetime.strptime('01-01-2019', '%d-%m-%Y'), to_date=dt.datetime.strptime('31-12-2023', '%d-%m-%Y'), shift=7, rolling_window=[7, 14], freq='1D', create_X_y=True, split_config = {'test_size': 0.2, 'val_size': 0.2, 'shuffle': False})
# print(dataset.data)

# dataset.plot(freq='1D', max_subplots=16)
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
                # ne.TargetEncoder(target_type='continuous-multioutput'),
                # ne.TargetEncoder(target_type='continuous'),
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
                # ne.TargetEncoder(target_type='continuous'),
                ne.MultiTargetEncoder(drop_invariant=True, return_df=True),


            ]
        }
    }
}

piepline = create_encoding_pipeline(encoders_dict)
dataset.encode(pipeline=piepline)

# dataset.save_data(root_dir, "data")





# print(baseTabularDataset.data.info())
# baseTabularDataset.encode(encoders=encoders)

# print(baseTabularDataset.encoded_data)
# print(baseTabularDataset.features)
