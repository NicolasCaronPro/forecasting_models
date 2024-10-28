from pathlib import Path
from sklearn_api_model import save_object, Model
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import mlflow

mlflow.sklearn.autolog()

reg = xgb.XGBRegressor()
reg.set_params(device='cuda')
model = Model(model=reg, loss='rmse', name='Model')
# with open('data/models/model.pkl', 'rb') as f:
#     model = pickle.load(f)

features = pd.read_csv(
    "/home/maxime/Documents/WORKSPACES/ARS_Project/data/dataframes/features_encoded/features_CHU Dijon.csv")
features.set_index('date_entree', inplace=True)

features = features.dropna()

train_set, test_set = train_test_split(
    features, test_size=0.2, random_state=42)

X_train = train_set.drop("Total", axis=1)
y_train = train_set["Total"]

X_test = test_set.drop("Total", axis=1)
y_test = test_set["Total"]

param_grid = {
    'max_depth': [3, 4, 5, 6]
}

fit_params = {
}
model.fit(X_train, y_train, optimization='grid',
          grid_params=param_grid, fit_params=fit_params)
save_object(model, 'model.pkl', Path('data/models/'))

print(model.predict(X_test))
print(model.score(X_test, y_test))
