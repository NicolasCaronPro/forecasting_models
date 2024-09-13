from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

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
            pd.set_option('future.no_silent_downcasting', True)
            for column in date_features.columns:
                date_features[f"{column}_cat"] = date_features[column].astype('category').infer_objects(copy=False)
            
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