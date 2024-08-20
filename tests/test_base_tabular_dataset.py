import datetime as dt
import logging
import os
import sys

from feature_engine.creation import CyclicalFeatures
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, RobustScaler


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.datasets.base_tabular_dataset import BaseTabularDataset, ft


ars_features_class = [ft.AirQualityFeatures, ft.CalendarFeatures, ft.EpidemiologicalFeatures, ft.FireFightersFeatures,
                      ft.GoogleTrendFeatures, ft.HopitalFeatures, ft.MeteorologicalFeatures, ft.SociologicalFeatures,
                      ft.SportsCompetitionFeatures, ft.TrafficFeatures]

ars_features_class = ["HopitalFeatures"]

encoders = {'datetime64': [CyclicalFeatures, ce.TargetEncoder], 'category': [ce.TargetEncoder], 'number': [StandardScaler]}

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    format="%(name)s %(asctime)s: %(levelname)s: %(message)s", handlers=[logging.StreamHandler()])
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config = ft.Config({'max_nan': 0, "departement": "21", "root_dir": root_dir, "start": dt.datetime.strptime('01-01-2018', '%d-%m-%Y'),
                    "stop": dt.datetime.strptime('31-12-2023', '%d-%m-%Y'), "logger": logger, "step_unit": 'days', "step_value": 1,
                    "shift": 0, "rolling_window": 0, "etablissement": "CHU Dijon"})

baseTabularDataset = BaseTabularDataset(target_colomns='HopitalFeatures Total_CHU Dijon',
    config=config, features_class=ars_features_class)

baseTabularDataset.fetch_data()

# baseTabularDataset.plot()

baseTabularDataset.data['date'] = baseTabularDataset.data.index
# print(baseTabularDataset.data.info())
baseTabularDataset.encode(encoders=encoders)

print(baseTabularDataset.encoded_data)
# print(baseTabularDataset.features)
