import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.encoding.encoders import *
from src.encoding.tools import create_encoding_pipeline
from src.models.sklearn_models import save_object, Model
from src.models.sklearn_models_config import get_model
from src.datasets.base_tabular_dataset import BaseTabularDataset
from src.experiments.base_experiment import BaseExperiment
import src.features as ft
import logging

# mlflow.set_tracking_uri("http://localhost:5000")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    format="%(name)s %(asctime)s: %(levelname)s: %(message)s", handlers=[logging.StreamHandler()])
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config = ft.Config({'max_nan': 0, "departement": "21", "root_dir": root_dir, "start": '01-01-2019',
                    "stop": '31-12-2023', "logger": logger, "step_unit": 'days', "step_value": 1,
                    "shift": 0, "rolling_window": 0, "etablissement": "CHU Dijon"})

ars_features_class = [ft.AirQualityFeatures, ft.HopitalFeatures(config=config, include_emmergency_arrivals=True, include_nb_hospit=False), ft.EpidemiologicalFeatures, ft.FireFightersFeatures,
                      ft.GoogleTrendFeatures, ft.MeteorologicalFeatures, ft.SociologicalFeatures,
                      ft.SportsCompetitionFeatures, ft.TrafficFeatures]

arsTabularDataset = BaseTabularDataset(target_colomns='HopitalFeatures Total_CHU Dijon',  # nb_vers_hospit, 
                                       config=config, features_class=ars_features_class)
arsTabularDataset.fetch_data() # Fetch data from the features, do this only once, if you need smaller datasets, use the get_dataset method

model_params = {
    'early_stopping_rounds': 10,
    # 'eval_set': [(arsTabularDataset.enc_X_val, arsTabularDataset.y_val)], # TODO: to be set in the experiment's run method
    'verbosity': 0
}
model = get_model(model_type='xgboost', name='XGBRegressor', device='cuda', task_type='regression', test_metrics='rmse', with_metric='w_rmse', params=model_params)

ars_experiment = BaseExperiment(dataset=arsTabularDataset, model=model, config=config)

grid_params = {
    'n_estimators': [10000],
    'max_depth': [3,5,7,9,11],
    'learning_rate': [0.1],
    'min_child_weight': [5],
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
                ce.TargetEncoder(),
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
                ce.TargetEncoder(),
            ]
        }
    }
}

encoding_pipeline = create_encoding_pipeline(encoders_dict=encoders_dict)

ars_experiment.run(dataset_config={'from_date': '01-01-2019', 'to_date': '31-12-2023'}, model_config={"optimization": "grid",
                   "grid_params": grid_params, "fit_params": fit_params}, encoding_pipeline=encoding_pipeline)
