# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.encoding.tools import create_encoding_pipeline
from src.encoding.encoders import *
import pandas as pd

from sklearn import set_config

# +
data = pd.DataFrame({
    'num_feature1': [1.0, 2.0, 3.0, 4.0],
    'num_feature2': [10.0, 20.0, 30.0, 40.0],
    'cat_feature1': ['A', 'B', 'A', 'B'],
    'cat_feature2': ['X', 'Y', 'X', 'Y'],
    'date_feature': pd.to_datetime(['2021-01-01', '2021-01-27', '2021-01-03', '2021-03-26'])
})

y = pd.Series([1, 0, 1, 0])

data['num_feature1'] = data['num_feature1'].astype('float')
data['num_feature2'] = data['num_feature2'].astype('float')
data['cat_feature1'] = data['cat_feature1'].astype('category')
data['cat_feature2'] = data['cat_feature2'].astype('category')
data['date_feature'] = pd.to_datetime(data['date_feature'])

# +
test_data = pd.DataFrame({
    'num_feature1': [5.0],
    'num_feature2': [50.0],
    'cat_feature1': ['A'],
    'cat_feature2': ['X'],
    'date_feature': pd.to_datetime(['2021-03-27'])
})

test_data['num_feature1'] = test_data['num_feature1'].astype('float')
test_data['num_feature2'] = test_data['num_feature2'].astype('float')
test_data['cat_feature1'] = test_data['cat_feature1'].astype('category')
test_data['cat_feature2'] = test_data['cat_feature2'].astype('category')
test_data['date_feature'] = pd.to_datetime(test_data['date_feature'])
# -

encoders_dict = {
    'number': {
        'as_number': {
            'imputers': [imputers.SimpleImputer(strategy='mean')],
            'encoders': [
                ne.StandardScaler(),
                ne.MinMaxScaler(),
                # RobustScaler()
            ]
        }
    },
    'category': {
        'as_category': {
            'imputers': [imputers.SimpleImputer(strategy='most_frequent')],
            'encoders': [
                ce.OneHotEncoder(),
                ce.TargetEncoder(),
                # CatBoostEncoder()
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
                #  CatBoostEncoder()
            ]
        }
    }
}

processor = create_encoding_pipeline(encoders_dict)

print(processor)

processor.fit(data, y)
processor

encoded = processor.transform(data)
print(encoded)

encoded = processor.transform(test_data)
print(encoded)
