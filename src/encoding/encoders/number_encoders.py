from typing import Literal
import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_engine.creation import CyclicalFeatures
from category_encoders import TargetEncoder
from category_encoders.wrapper import PolynomialWrapper
from sklearn.base import BaseEstimator, TransformerMixin


class MultiTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value',
                 handle_unknown='value', min_samples_leaf=20, smoothing=10, hierarchy=None):
        super().__init__()
        self.targets = None
        self.encoders = None
        self.verbose = verbose
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.hierarchy = hierarchy

    def fit(self, X, y):
        self.targets = y.columns
        self.encoders = {target: TargetEncoder(cols=self.cols, drop_invariant=self.drop_invariant,
                                               return_df=self.return_df, handle_missing=self.handle_missing,
                                               handle_unknown=self.handle_unknown, min_samples_leaf=self.min_samples_leaf,
                                               smoothing=self.smoothing, verbose=self.verbose) for target in self.targets}
        for target in self.targets:
            self.encoders[target].fit(X, y[target])
        return self

    def transform(self, X, y=None):
        encoded_X = []
        for target in self.targets:
            df = self.encoders[target].transform(X)
            df.columns = [f"{col}##{target}" for col in df.columns]
            encoded_X.append(df)

        encoded_X = pd.concat(encoded_X, axis=1)
        return encoded_X
    
    def set_output(self, transform):
        # This method can be used to handle `set_output` introduced in sklearn 1.2
        return self