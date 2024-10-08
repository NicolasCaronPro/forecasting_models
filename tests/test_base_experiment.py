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

# mlflow.set_tracking_uri("http://localhost:5000")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s: %(levelname)s: %(message)s",
                    datefmt='%Y/%m/%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config = ft.Config({'max_nan': 15, "departement": "21", "root_dir": root_dir, "start": '01-01-2019',
                    "stop": '31-12-2023', "logger": logger, "step_unit": 'days', "step_value": 1,
                    "shift": 0, "rolling_window": 0, "etablissement": "CHU Dijon", 'region':'BOURGOGNE'})

ars_features_class = [ft.AirQualityFeatures(config=config, drop_const_cols=True), ft.HopitalFeatures(config=config, include_emmergency_arrivals=True, include_nb_hospit=True), ft.EpidemiologicalFeatures, ft.FireFightersFeatures(config=config, include_calls=False),
                      ft.GoogleTrendFeatures, ft.MeteorologicalFeatures, ft.SociologicalFeatures,
                      ft.SportsCompetitionFeatures, ft.TrafficFeatures]
# target_colomns = ['nb_vers_hospit']
target_colomns = ['Total_CHU Dijon']
arsTabularDataset = BaseTabularDataset(features_class=ars_features_class, logger=logger, data_dir=os.path.join(root_dir,'data'))
arsTabularDataset.fetch_datataset(save=True) # Fetch data from the features, do this only once, if you need smaller datasets, use the get_dataset method

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
model = get_model(model_type='xgboost', name='XGBRegressor', device='cuda', task_type='regression', test_metrics='w_rmse', with_metric='w_rmse', params=model_params)

ars_experiment = BaseExperiment(dataset=arsTabularDataset, model=model, config=config)

grid_params = {
    'max_depth': [3, 5, 7, 9, 11],
}

fit_params = {
    'verbose': 0,
}

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

# tes = ne.MultiTargetEncoder(drop_invariant=True, return_df=True)


# +
# tes.set_output(transform='pandas')
# -

split_config = {'test_size': 0.2, 'val_size': 0.2, 'shuffle': False}

model_config={"optimization": "grid", "grid_params": grid_params, "fit_params": fit_params}
encoding_pipeline = create_encoding_pipeline(encoders_dict=encoders_dict)
dataset_config={'from_date': '15-01-2019', 'to_date': '31-12-2023', 'shift':[1, 2, 3, 4, 5, 6, 7], 'rolling_window':[7, 14], 'freq':'1D', 'split_config': split_config, 'encoding_pipeline': encoding_pipeline}

ars_experiment.run(dataset_config=dataset_config, model_config=model_config, find_best_features=True)

# dataset = ars_experiment.dataset
# data = dataset.data
# enc_data = dataset.enc_data

# enc_data

# dataset = dataset.get_dataset(**dataset_config)

# enc_data = dataset.enc_data
# enc_data