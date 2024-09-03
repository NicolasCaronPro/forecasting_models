from typing import Optional, Union, List
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
import src.features as ft
from src.encoding.tools import create_encoding_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
import datetime as dt



class BaseTabularDataset(ft.BaseFeature):
    def __init__(self, target_colomns: Union[List[str], str], features_class: List[Union[ft.BaseFeature, str]], config: Optional['ft.Config'] = None, parent: Optional['ft.BaseFeature'] = None) -> None:
        super().__init__(config=config, parent=parent)
        # Initialize each features object and fetch their data and then get them for the specified range

        # Get target
        self.targets = target_colomns if isinstance(
            target_colomns, list) else [target_colomns]
        self.features: List[Union[ft.BaseFeature, str]] = []
        self.encoded_data = None

        # Initialize each feature
        self.initialize_features(features_class)

        self.enc_X_train = None
        self.enc_X_val = None
        self.enc_X_test = None

        self.X_train = None
        self.y_train = None
        
        self.X_val = None
        self.y_val = None

        self.X_test = None
        self.y_test = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def initialize_features(self, features_class) -> None:
        """
        Initialize each feature.

        Parameters:
        - None
        """

        for feature_class in features_class:
            if isinstance(feature_class, str):
                feature_class = getattr(ft, feature_class)
            feature = feature_class(config=self.config, parent=self)
            self.features.append(feature)

    def fetch_data(self) -> None:
        """
        Fetch les données.

        Parameters:
        - None
        """

        # Get data from each feature
        for feature in self.features:
            self.logger.info(f"Fetching data from {feature}")
            feature.fetch_data()
            self.data = self.data.join(feature.data)

        # self.targets = self.data[self.targets]

        super().fetch_data()

    def encode(self, pipeline: Pipeline = None) -> None:
        """
        Encode les données.

        Parameters:
        - pipeline: Pipeline
        """
        
        if pipeline is None:
            raise ValueError("Pipeline is required")
        
        if self.train_set is not None:
            self.enc_X_train = pipeline.fit_transform(X=self.X_train, y=self.y_train)

            if self.val_set is not None:
                self.enc_X_val = pipeline.transform(self.X_val)

            if self.test_set is not None:
                self.enc_X_test = pipeline.transform(self.X_test)

    
    def create_X_y(self) -> None:
        """
        Create X and y.

        Parameters:
        - None
        """
        self.X_train = self.train_set.drop(columns=self.targets)
        self.y_train = self.train_set[self.targets]

        self.X_val = self.val_set.drop(columns=self.targets)
        self.y_val = self.val_set[self.targets]

        self.X_test = self.test_set.drop(columns=self.targets)
        self.y_test = self.test_set[self.targets]


    def split(self, test_size=None, train_size=None, val_size=None, random_state=None, shuffle=True, stratify=None):
        """Split arrays or matrices into random train and test subsets.

        Quick utility that wraps input validation,
        ``next(ShuffleSplit().split(X, y))``, and application to input data
        into a single call for splitting (and optionally subsampling) data into a
        one-liner.

        Read more in the :ref:`User Guide <cross_validation>`.

        Parameters
        ----------
        *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse
            matrices or pandas dataframes.

        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25.

        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.

        random_state : int, RandomState instance or None, default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        stratify : array-like, default=None
            If not None, data is split in a stratified fashion, using this as
            the class labels.
            Read more in the :ref:`User Guide <stratification>`.

        Returns
        -------
        splitting : list, length=2 * len(arrays)
            List containing train-test split of inputs.

            .. versionadded:: 0.16
                If the input is sparse, the output will be a
                ``scipy.sparse.csr_matrix``. Else, output type is the same as the
                input type.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = np.arange(10).reshape((5, 2)), range(5)
        >>> X
        array([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
        >>> list(y)
        [0, 1, 2, 3, 4]

        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42)
        ...
        >>> X_train
        array([[4, 5],
            [0, 1],
            [6, 7]])
        >>> y_train
        [2, 0, 3]
        >>> X_test
        array([[2, 3],
            [8, 9]])
        >>> y_test
        [1, 4]

        >>> train_test_split(y, shuffle=False)
        [[0, 1, 2], [3, 4]]
        """

        train_val_set, test_set = train_test_split(self.data, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        self.train_set = train_val_set
        self.test_set = test_set
        if val_size is None:
            return train_val_set, test_set
        
        train_set, val_set = train_test_split(train_val_set, test_size=val_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        self.train_set = train_set
        self.val_set = val_set
        return train_set, val_set, test_set
    
    def get_dataset(self, from_date: Optional[Union[str, dt.datetime]] = None, to_date: Optional[Union[str, dt.datetime]] = None, features_names: Optional[List[str]] = None) -> 'BaseTabularDataset':
        """
        Get the data.

        Parameters:
        - None
        """
        filtered_data = self.get_data(from_date=from_date, to_date=to_date, features_names=features_names)

        new_dataset = deepcopy(self)
        new_dataset.data = filtered_data
        return new_dataset