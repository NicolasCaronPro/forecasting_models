train = arsTabularDataset.X_train['Total_CHU Dijon']
val = arsTabularDataset.X_val['Total_CHU Dijon']
df = pd.concat([train, val])
df = df.reset_index()
df.rename({"Total_CHU Dijon": "y", 'date': 'ds'}, axis=1, inplace=True)
df
test = arsTabularDataset.X_test['Total_CHU Dijon']
future = test.reset_index()
future.rename({"Total_CHU Dijon": "y", 'date':  'ds'}, axis=1, inplace=True)
future.drop(columns=["y"], inplace=True)
future
from prophet import Prophet
m = Prophet()
m.add_country_holidays(country_name='FR')
m.fit(df)
m.train_holiday_names
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
test.to_frame().plot(ax=ax, color='orange')
m.plot(forecast, ax=ax, uncertainty=False, xlabel='Date', ylabel='Value')

fig2 = m.plot_components(forecast)
%pip install --upgrade plotly
from prophet.plot import plot_plotly, plot_components_plotly

fig = plot_plotly(m, forecast)
fig.write_html("prophet.html")
from prophet.make_holidays import get_holiday_names, make_holidays_df

fr_holidays = make_holidays_df(
    year_list=[2019 + i for i in range(10)], country='FR'
)

fr_holidays_names = get_holiday_names('FR')
fr_holidays_names

d = arsTabularDataset.data
d

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_squared_error

# Génération d'une série temporelle synthétique
data = arsTabularDataset.data['Total_CHU Dijon']
data_exog = arsTabularDataset.enc_data[[col for col in arsTabularDataset.enc_data.columns.to_list() if not col.startswith('target') and not col.startswith('Total_CHU Dijon')]]
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
    model = pm.auto_arima(train, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=len(test))
    evaluate_forecast(test, forecast)
    return model, forecast

# 3. Modèle SARIMAX (avec exogènes)
def fit_sarimax(train, test):
    # exog = np.random.randn(len(train))  # Exemple de variable exogène
    # exog_test = np.random.randn(len(test))
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=train_exog)
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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