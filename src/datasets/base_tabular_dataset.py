from typing import Optional, Union, List
from sklearn.pipeline import Pipeline
import src.features as ft
from src.encoding.tools import create_encoding_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
import datetime as dt
import numpy as np


class BaseTabularDataset(ft.BaseFeature):
    def __init__(self, target_colomns: Union[List[str], str], features_class: List[Union[ft.BaseFeature, str]], config: Optional['ft.Config'] = None, parent: Optional['ft.BaseFeature'] = None, chained_targets: bool = False) -> None:
        # Initialize each features object and fetch their data and then get them for the specified range

        # Get target
        self.targets = target_colomns if isinstance(
            target_colomns, list) else [target_colomns]
        self.chained_targets = chained_targets
        self.features: List[Union[ft.BaseFeature, str]] = []

        # Initialize each feature

        self.enc_X_train: pd.DataFrame|None = None
        self.enc_X_val: pd.DataFrame|None = None
        self.enc_X_test: pd.DataFrame|None = None

        self.enc_data: pd.DataFrame|None = None

        self.X_train: pd.DataFrame|None = None
        self.y_train: pd.DataFrame|None = None

        self.X_val: pd.DataFrame|None = None
        self.y_val: pd.DataFrame|None = None

        self.X_test: pd.DataFrame|None = None
        self.y_test: pd.DataFrame|None = None

        self.train_set: pd.DataFrame|None = None
        self.val_set: pd.DataFrame|None = None
        self.test_set: pd.DataFrame|None = None
        super().__init__(config=config, parent=parent)

        self.initialize_features(features_class)
        
    def initialize_features(self, features_class) -> None:
        """
        Initialize each feature.

        Parameters:
        - None
        """

        # Initialize each feature
        for feature_class in features_class:
            # If the feature is a string, get the class from the features module
            if isinstance(feature_class, str):
                feature_class = getattr(ft, feature_class)

            # If the feature is a class, instantiate it, otherwise if it's an instance, use it
            if isinstance(feature_class, ft.BaseFeature):
                feature = feature_class
                feature.config = self.config
                feature.parent = self
            else:
                feature = feature_class(config=self.config, parent=self)

            self.features.append(feature)

    def fetch_data_function(self, args, **kwargs) -> None:
        """
        Fetch les données.

        Parameters:
        - None
        """

        # Get data from each feature
        for feature in self.features:
            self.logger.info(f"Fetching data from {feature}")
            if not feature.is_fetched:
                feature.fetch_data()
            self.data = self.data.join(feature.data)

        # self.targets = self.data[self.targets]

    def encode(self, pipeline: Pipeline = None) -> None:
        """
        Encode les données.

        Parameters:
        - pipeline: Pipeline - The pipeline to use for encoding the data
        """

        if pipeline is None:
            raise ValueError("Pipeline is required")

        encoded_data = []
        
        for target in self.targets:
            pass

        if not self.X_train.empty:
            self.enc_X_train = pipeline.fit_transform(
                X=self.X_train.reset_index().set_index(keys='date', drop=False), y=self.y_train)
            self.enc_X_train.columns = [
                col.split('__')[-1] for col in self.enc_X_train.columns]
            encoded_data.append(self.enc_X_train)
            if not self.X_val.empty:
                self.enc_X_val = pipeline.transform(self.X_val.reset_index().set_index(keys='date', drop=False))
                self.enc_X_val.columns = [
                    col.split('__')[-1] for col in self.enc_X_val.columns]
                encoded_data.append(self.enc_X_val)

            if not self.X_test.empty:
                self.enc_X_test = pipeline.transform(self.X_test.reset_index().set_index(keys='date', drop=False))
                self.enc_X_test.columns = [
                    col.split('__')[-1] for col in self.enc_X_test.columns]
                encoded_data.append(self.enc_X_test)

        self.enc_data = pd.concat(encoded_data, axis=0)

    def create_X_y(self, val=True, test=True) -> None:
        """
        Create X and y.

        Parameters:
        - None
        """

        if not self.train_set.empty:
            self.X_train = self.train_set.drop(columns=self.targets)
            self.y_train = self.train_set[self.targets]
        else:
            self.X_train = self.data.drop(columns=self.targets)
            self.y_train = self.data[self.targets]

        if val:
            if not self.val_set.empty:
                self.X_val = self.val_set.drop(columns=self.targets)
                self.y_val = self.val_set[self.targets]
            else:
                raise ValueError(
                    "Validation set is not available, you need to split the data first")

        if test:
            if not self.test_set.empty:
                self.X_test = self.test_set.drop(columns=self.targets)
                self.y_test = self.test_set[self.targets]
            else:
                raise ValueError(
                    "Test set is not available, you need to split the data first")

    def split(self, train_size: Optional[Union[float, int]] = None,
              test_size: Optional[Union[float, int]] = None,
              val_size: Optional[Union[float, int]] = None,
              random_state: Optional[int] = None,
              shuffle: bool = True,
              stratify: Optional[Union[pd.Series, np.ndarray]] = None) -> tuple:
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
        train_val_set, test_set = train_test_split(
            self.data, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        self.train_set = train_val_set
        self.test_set = test_set
        self.val_set = None

        if val_size:
            train_set, val_set = train_test_split(
                train_val_set, test_size=val_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
            self.train_set = train_set
            self.val_set = val_set
        return self.train_set, self.val_set, self.test_set

    def get_dataset(self, from_date: Optional[Union[str, dt.datetime]] = None,
                    to_date: Optional[Union[str, dt.datetime]] = None,
                    features_names: Optional[List[str]] = None,
                    freq: Optional[str] = '1D',
                    shift: Optional[int] = 0,
                    rolling_window: Optional[Union[int, List[int]]] = 0,
                    split_config: Optional[dict] = {},
                    create_X_y: bool = True) -> 'BaseTabularDataset':
        """
        Get the data.

        Parameters:
        - None
        """
        filtered_data = self.get_data(from_date=from_date, to_date=to_date,
                                      features_names=features_names, freq=freq, shift=shift, rolling_window=rolling_window)

        new_dataset: BaseTabularDataset = deepcopy(self)
        new_dataset.data = filtered_data
        new_dataset.start = from_date
        new_dataset.stop = to_date

        # Update the target columns in the new dataset
        new_dataset.targets = new_dataset.data.loc[:, [col for col in new_dataset.data.columns if any(col.split('___')[0] == target for target in self.targets) or any(col == target for target in self.targets)]].columns.to_list()
        self.chained_targets = True

        # We should also update other attributes like enc_X_train, enc_X_val, enc_X_test, X_train, y_train, X_val, y_val, X_test, y_test, train_set, val_set, test_set if they are not None and if the features_names parameter is the only parameter passed
        if features_names:
            #  and from_date is None and to_date is None and shift == 0 and rolling_window == 0 and split_config == {} and freq == pd.infer_freq(self.data.index) or freq == '1'+pd.infer_freq(self.data.index)
            if not self.enc_X_train.empty:
                new_dataset.enc_X_train = self.enc_X_train[features_names]
                if not self.enc_X_val.empty:
                    new_dataset.enc_X_val = self.enc_X_val[features_names]
                if not self.enc_X_test.empty:
                    new_dataset.enc_X_test = self.enc_X_test[features_names]

            if not self.X_train.empty:
                new_dataset.X_train = self.X_train[features_names]
                if not self.X_val.empty:
                    new_dataset.X_val = self.X_val[features_names]
                if not self.X_test.empty:
                    new_dataset.X_test = self.X_test[features_names]

            if not self.train_set.empty:
                new_dataset.train_set = self.train_set[features_names+self.targets]
                if not self.val_set.empty:
                    new_dataset.val_set = self.val_set[features_names+self.targets]
                if not self.test_set.empty:
                    new_dataset.test_set = self.test_set[features_names+self.targets]

        else:
            if split_config:
                new_dataset.split(**split_config)
            if create_X_y:
                new_dataset.create_X_y()
        return new_dataset

    def get_features_names(self) -> List[str]:
        columns = super().get_features_names()

        # Retirer les targets
        columns = [col for col in columns if col not in self.targets]

        return columns

    def get_targets_names(self) -> List[str]:
        return self.targets
