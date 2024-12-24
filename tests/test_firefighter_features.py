import datetime as dt
import logging
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.features.firefigters_features import FireFightersFeatures, Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    format="%(name)s %(asctime)s: %(levelname)s: %(message)s", handlers=[logging.StreamHandler()])
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config = Config({'max_nan': 0, "departement": "21", "root_dir": root_dir, "start": dt.datetime.strptime('01-01-2018', '%d-%m-%Y'),
                 "stop": dt.datetime.strptime('31-12-2023', '%d-%m-%Y'), "logger": logger, "step_unit": 'days', "step_value": 1,
                 "shift": 0, "rolling_window": 0})

fireFightersFeatures = FireFightersFeatures(config=config)

fireFightersFeatures.fetch_data()

data = fireFightersFeatures.get_data(from_date=dt.datetime.strptime('01-01-2018', '%d-%m-%Y'), to_date=dt.datetime.strptime('31-12-2023', '%d-%m-%Y'), shift=7, rolling_window=[7, 14], freq='D')
print(data)

# fireFightersFeatures.plot(freq='1D')
