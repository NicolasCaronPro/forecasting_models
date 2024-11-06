from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from prophet.make_holidays import get_holiday_names, make_holidays_df
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
from prophet import Prophet
import pathlib
import pandas as pd
import logging
import src.features as ft
from src.experiments.base_experiment import BaseExperiment
from src.datasets.base_tabular_dataset import BaseTabularDataset
from src.models.sklearn_models_config import get_model
from src.models.sklearn_models import save_object, Model
from src.encoding.tools import create_encoding_pipeline
from src.encoding.encoders import *
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# +
# # %pip install -r ../requirements.txt
# -


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
    ft.HospitalFeatures,
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
# target_colomns = ['Total_CHU Dijon']
# target_colomns = ['nb_vers_hospit']
target_colomns = ['nb_hospit_np_adults%%J+1%%mean_7J']

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

features_config_get = {}

# Define the configuration of the dataset
dataset_config = {
    'from_date': '22-01-2019',
    'to_date': '30-12-2023',
    'shift': range(1, 14, 1),
    'rolling_window': [7, 14],
    'freq': '1D',
    'split_config': split_config,
    'create_X_y': True,
    'encoding_pipeline': pipeline,
    'targets_names': target_colomns,
    'targets_shift': 0,
    'targets_rolling_window': 0,
    'targets_history_shifts': range(7, 14, 1),
    'targets_history_rolling_windows': [7, 14],
    'drop_constant_thr': 0.65,
    'data_dir': root_dir / 'data'
}

# Create the dataset and fetch the data from the source then call get_dataset() method to fill the different attributes (X and y) of the different sets, and their encodings
arsTabularDataset = BaseTabularDataset(
    features_classes=ars_features_class, logger=logger, fetch_config=fetch_config, getter_config=dataset_config)

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

# first one is used for evaluation and everywhere a sinlge metric is used, the rest are used for testing
metrics = ['w_rmse', 'pw_rmse', 'rmse', 'mae', 'mse']

# Create the model
model = get_model(model_type='xgboost', name='XGBoost', device='cuda',
                  task_type='regression', test_metrics=metrics, params=model_params)

# Create the experiment
ars_experiment = BaseExperiment(
    logger=logger, dataset=arsTabularDataset, model=model)

# Set the model fitting config
grid_params = {
    'max_depth': [3, 5, 7, 9, 11],
}

fit_params = {
    'verbose': 0,
}

model_config = {"optimization": "grid",
                "grid_params": grid_params, "fit_params": fit_params}

# Run the experiment
ars_experiment.run(dataset_config=dataset_config,
                   model_config=model_config, find_best_features=True)

# +
# train = arsTabularDataset.X_train['Total_CHU Dijon']
# val = arsTabularDataset.X_val['Total_CHU Dijon']
# df = pd.concat([train, val])
# df = df.reset_index()
# df.rename({"Total_CHU Dijon": "y", 'date': 'ds'}, axis=1, inplace=True)
# df
# -

test = arsTabularDataset.X_test['Total_CHU Dijon']
future = test.reset_index()
future.rename({"Total_CHU Dijon": "y", 'date':  'ds'}, axis=1, inplace=True)
future.drop(columns=["y"], inplace=True)
future

m = Prophet()
m.add_country_holidays(country_name='FR')
m.fit(df)
m.train_holiday_names

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig, ax = plt.subplots(figsize=(10, 6))

test.to_frame().plot(ax=ax, color='orange')
m.plot(forecast, ax=ax, uncertainty=False, xlabel='Date', ylabel='Value')


fig2 = m.plot_components(forecast)

# %pip install --upgrade plotly

# +

fig = plot_plotly(m, forecast)
fig.write_html("prophet.html")

# +

fr_holidays = make_holidays_df(
    year_list=[2019 + i for i in range(10)], country='FR'
)

fr_holidays_names = get_holiday_names('FR')
fr_holidays_names

# -

d = arsTabularDataset.data

d

# +


# Génération d'une série temporelle synthétique
data = arsTabularDataset.data['Total_CHU Dijon']
data_exog = arsTabularDataset.enc_data[[col for col in arsTabularDataset.enc_data.columns.to_list(
) if not col.startswith('target') and not col.startswith('Total_CHU Dijon')]]
dates = arsTabularDataset.data.index
series = pd.Series(data, index=dates)

# Train-test split
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]
train_exog, test_exog = data_exog[:train_size], data_exog[train_size:]

# Fonction d'évaluation des modèles


def evaluate_forecast(true, predicted):
    mse = mean_squared_error(true, predicted)
    print(f"Mean Squared Error (MSE): {mse}")
    return mse

# 1. Modèle ARIMA


def fit_arima(train, test):
    model = ARIMA(train, order=(5, 1, 0))  # Paramètres (p,d,q)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    evaluate_forecast(test, forecast)
    return model_fit, forecast

# 2. Modèle SARIMA (avec saisonnalité)


def fit_sarima(train, test):
    model = pm.auto_arima(train, seasonal=True, m=12,
                          stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=len(test))
    evaluate_forecast(test, forecast)
    return model, forecast

# 3. Modèle SARIMAX (avec exogènes)


def fit_sarimax(train, test):
    # exog = np.random.randn(len(train))  # Exemple de variable exogène
    # exog_test = np.random.randn(len(test))
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(
        1, 1, 1, 12), exog=train_exog)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test), exog=test_exog)
    evaluate_forecast(test, forecast)
    return model_fit, forecast

# 4. Modèle auto ARIMA (avec auto-ajustement des paramètres)


def fit_auto_arima(train, test):
    model = pm.auto_arima(train, start_p=1, start_q=1, max_p=5, max_q=5,
                          seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=len(test))
    evaluate_forecast(test, forecast)
    return model, forecast


# 5. Modèle Holt-Winters (pour la décomposition additive ou multiplicative)


def fit_holt_winters(train, test):
    model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    evaluate_forecast(test, forecast)
    return model_fit, forecast


# Appel des fonctions
print("ARIMA:")
fit_arima(train, test)

print("\nSARIMA:")
fit_sarima(train, test)

print("\nSARIMAX:")
fit_sarimax(train, test)

print("\nAuto ARIMA:")
fit_auto_arima(train, test)

print("\nHolt-Winters:")
fit_holt_winters(train, test)
