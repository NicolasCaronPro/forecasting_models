from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)
# Fonction pour extraire les caractéristiques temporelles


def extract_date_features(X, date_columns=None, date_features_names=None, dtype='number'):
    """
    Decompose the input date into several features.

    Parameters:
    - X (pd.DataFrame): The input data.

    Returns:
    - pd.DataFrame: The decomposed data.
    """
    date_features = pd.DataFrame(index=X.index)

    if date_columns is None:
        date_columns = X.select_dtypes(include=['datetime']).columns.tolist()

    if isinstance(date_columns, str):
        date_columns = [date_columns]

    for col in date_columns:
        if not pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = pd.to_datetime(X[col])

        # date_features[f'{col}_year'] = X[col].dt.year.astype('category')
        date_features[f'{col}_month'] = X[col].dt.month
        date_features[f'{col}_day'] = X[col].dt.day
        date_features[f'{col}_hour'] = X[col].dt.hour
        date_features[f'{col}_minute'] = X[col].dt.minute
        date_features[f'{col}_second'] = X[col].dt.second
        date_features[f'{col}_dayofweek'] = X[col].dt.dayofweek
        date_features[f'{col}_quarter'] = X[col].dt.quarter
        date_features[f'{col}_week'] = X[col].dt.isocalendar().week
        date_features[f'{col}_is_leap'] = X[col].dt.is_leap_year
        date_features[f'{col}_is_leap'] = (
            ~date_features[f'{col}_is_leap']).to_numpy(dtype=int)
        date_features[f'{col}_after_feb'] = X[col].dt.month > 2
        date_features[f'{col}_after_feb'] = date_features[f'{col}_after_feb'].to_numpy(
            dtype=int)
        date_features[f'{col}_dayofYear'] = X[col].dt.dayofyear + date_features[f'{col}_after_feb'].to_numpy() * \
            date_features[f'{col}_is_leap'].to_numpy()
        date_features.drop(
            [f'{col}_is_leap', f'{col}_after_feb'], axis=1, inplace=True)

        # Select only columns with more than one unique value
        if date_features_names is None:
            date_features = date_features.loc[:, date_features.nunique() > 1]
        else:
            date_features = date_features.loc[:, date_features.columns.isin(
                date_features_names)]

        if dtype == 'category':
            for column in date_features.columns:
                date_features[f"{column}_cat"] = date_features[column].astype(
                    'category').fillna(np.nan)
    return date_features


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from date.

    Attributes:
        date_column (str): The name of the date column to encode.
    """

    def __init__(self, dtype='number') -> None:
        self.dtype = dtype
        self.extracted_date_features = None
        self.columns_ = None
        super().__init__()

    def fit(self, X, y=None):
        extracted_date_features = extract_date_features(X, dtype=self.dtype)
        self.columns_ = extracted_date_features.columns
        return self

    def transform(self, X):
        self.extracted_date_features = extract_date_features(
            X, date_features_names=self.columns_, dtype=self.dtype)
        # self.columns_ = self.extracted_date_features.columns
        return self.extracted_date_features

    def get_feature_names_out(self, *args, **params):
        pass


def create_encoding_pipeline(encoders_dict):
    print("Creating encoding pipeline")
    transformers = []
    for col_type, encoders in encoders_dict.items():
        if col_type == 'number':
            imputer = SimpleImputer(strategy='mean')
            encoded_features = make_union(
                *encoders).set_output(transform='pandas')
            pipeline = make_pipeline(
                imputer, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='number')
            transformer = make_column_transformer(
                (pipeline, selector), remainder='passthrough').set_output(transform='pandas')
            transformers.append(transformer)
            # print(transformer)
            # transformers[-1].fit(data)
            # print(transformers[-1].transform(data))
        elif col_type == 'category':
            imputer = SimpleImputer(strategy='constant', fill_value='missing')
            encoded_features = make_union(
                *encoders).set_output(transform='pandas')
            pipeline = make_pipeline(
                imputer, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='category')
            transformer = make_column_transformer(
                (pipeline, selector), remainder='passthrough').set_output(transform='pandas')
            transformers.append(transformer)
            # print(transformer)
            # transformers[-1].fit(data)
            # print(transformers[-1].transform(data))
        elif col_type == 'datetime':
            date_transformers = []

            if 'as_category' in encoders:
                date_extractor = DateFeatureExtractor(dtype='category')
            else:
                date_extractor = DateFeatureExtractor()

            date_transformers.append(date_extractor)
            
            for c_types, enc in encoders.items():

                if c_types == 'as_number':
                    number_selector = make_column_selector(dtype_include='number')
                    number_encoders = enc
                    if len(number_encoders) == 0:
                        continue
                    encoded_features_as_number = make_union(
                        *number_encoders).set_output(transform='pandas')
                    features_encoded_as_number = make_column_transformer(
                        (encoded_features_as_number, number_selector), remainder='passthrough').set_output(transform='pandas')
                    # print(features_encoded_as_number)
                    date_transformers.append(features_encoded_as_number)

                if c_types == 'as_category':
                    category_selector = make_column_selector(
                        dtype_include='category')
                    category_encoders = enc
                    if len(category_encoders) == 0:
                        continue
                    encoded_features_as_category = make_union(
                        *category_encoders).set_output(transform='pandas')
                    features_encoded_as_category = make_column_transformer(
                        (encoded_features_as_category, category_selector), remainder='passthrough').set_output(transform='pandas')
                    # print(features_encoded_as_category)
                    date_transformers.append(features_encoded_as_category)
            
            # print(date_transformers)

            pipeline = make_pipeline(
                *date_transformers).set_output(transform='pandas')
            # pipeline = make_pipeline(date_extractor, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='datetime')
            date_transformer = make_column_transformer(
                (pipeline, selector), remainder='passthrough').set_output(transform='pandas')
            transformers.append(date_transformer)
            # print(date_transformer)
            # transformers[-1].fit(data)
            # print(transformers[-1].transform(data))

    processor = make_pipeline(*transformers, verbose=True).set_output(transform='pandas')
    # processor.set_params(params={'verbose', True})

    return processor
