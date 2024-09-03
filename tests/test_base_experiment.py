import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import mlflow
import xgboost as xgb
from src.encoding.encoders import *
from src.encoding.tools import create_encoding_pipeline
from src.models.sklearn_models import save_object, Model
from src.datasets.base_tabular_dataset import BaseTabularDataset
from src.experiments.base_experiment import BaseExperiment, ft
import datetime as dt
import logging

# mlflow.set_tracking_uri("http://localhost:5000")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    format="%(name)s %(asctime)s: %(levelname)s: %(message)s", handlers=[logging.StreamHandler()])
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config = ft.Config({'max_nan': 0, "departement": "21", "root_dir": root_dir, "start": dt.datetime.strptime('01-01-2019', '%d-%m-%Y'),
                    "stop": dt.datetime.strptime('31-12-2023', '%d-%m-%Y'), "logger": logger, "step_unit": 'days', "step_value": 1,
                    "shift": 0, "rolling_window": 0, "etablissement": "CHU Dijon"})

ars_features_class = [ft.AirQualityFeatures, ft.HopitalFeatures, ft.EpidemiologicalFeatures, ft.FireFightersFeatures,
                      ft.GoogleTrendFeatures, ft.MeteorologicalFeatures, ft.SociologicalFeatures,
                      ft.SportsCompetitionFeatures, ft.TrafficFeatures]

arsTabularDataset = BaseTabularDataset(target_colomns='nb_vers_hospit',  # , HopitalFeatures Total_CHU Dijon
                                       config=config, features_class=ars_features_class)
arsTabularDataset.fetch_data()

reg = xgb.XGBRegressor()
reg.set_params(device='cuda')
model = Model(model=reg, loss='rmse', name='Model')

ars_experiment = BaseExperiment(
    name='ars_exp', dataset=arsTabularDataset, model=model, config=config)

grid_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001]
}


params = {
    'verbose': 1,
    # 'objective': 'reg:squarederror'
    'early_stopping_rounds': 10,
    #     'eval_set': [(arsTabularDataset.enc_X_val, arsTabularDataset.y_val)]
    'eval_metric': 'rmse'
}

fit_params = {
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
                   "grid_params": grid_params, "fit_params": fit_params, "params": params}, encoding_pipeline=encoding_pipeline)
