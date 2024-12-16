from pathlib import Path
import os
import re
from typing import Optional, Union, List
from sklearn.pipeline import Pipeline
import src.features as ft
from src.encoding.tools import create_encoding_pipeline
from src.tools.utils import list_constant_columns
from src.location.location import Location
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
import datetime as dt
import numpy as np
import logging
import sys
import pathlib
from src.location import Location
from sklearn.utils import resample


def balance_target(dataset):
    # Combine x_train and y_train
    combined = pd.concat(
        [dataset.dataset.enc_X_train, dataset.dataset.y_train], axis=1)

    # find majority and minority classes
    # Count the occurrences of each category
    category_counts = dataset.dataset.y_train[dataset.dataset.targets_names[0]].value_counts(
    )

    # Identify majority and minority categories
    # Category with the most occurrences
    majority_category = category_counts.idxmax()
    # Category with the least occurrences
    minority_category = category_counts.idxmin()

    # Separate majority and minority classes
    majority = combined[combined[dataset.dataset.targets_names[0]]
                        == majority_category]
    minority = combined[combined[dataset.dataset.targets_names[0]]
                        == minority_category]

    # Oversample minority class
    minority_oversampled = resample(minority,
                                    replace=True,    # Sample with replacement
                                    # Match number of majority
                                    n_samples=len(majority),
                                    random_state=42)  # Reproducibility

    # Combine back the oversampled minority class with the majority class
    print('Before:', dataset.dataset.enc_X_train.shape)
    balanced = pd.concat([majority, minority_oversampled])
    print('After:', balanced.shape)

    # Split back to dataset.dataset.enc_X_train and dataset.dataset.y_train
    dataset.dataset.enc_X_train = balanced.drop(
        columns=[dataset.dataset.targets_names[0]])
    dataset.dataset.y_train = balanced[dataset.dataset.targets_names[0]]

    return dataset


def categorize(df, column, bins=[0, 0.1, 0.3, 0.7, 0.9, 0.97, 1.0], labels=None, drop=False):
    label_offset = 0
    if labels == None and type(bins) == int:
        labels = [f'{i+label_offset}' for i in range(bins)]
    elif labels == None and type(bins) == list:
        labels = [f'{i+label_offset}' for i in range(len(bins) - 1)]
    col_category = (column if drop else f'{column}_category')
    df[col_category] = pd.qcut(df[column], q=bins, labels=labels)
    df[col_category] = df[col_category].astype(np.float64)
    return df


def extract_column_base(col_name: str, separator='##') -> str:
    # Extraction du radical avant toute transformation (par ex: encodage, shift, aggregation)
    if separator in col_name:
        return col_name.split(separator)[0]
    return col_name


def find_matching_columns(df: pd.DataFrame, base_columns: List[str], to_print=False) -> List[str]:
    """ Trouve les colonnes dans un DataFrame qui correspondent aux bases des noms de colonnes spécifiées. """
    matching_columns = []
    # if to_print:
    # print(base_columns)
    for col in df.columns:
        # On extrait la base du nom de la colonne actuelle
        col_base = extract_column_base(col)
        # if to_print:
        # print(col, col_base)
        if col_base in base_columns:
            matching_columns.append(col)
            #
    return matching_columns


class BaseTabularDataset():
    def __init__(self, features_classes: List[Union[ft.BaseFeature, str]], name: str = None, logger=None, fetch_config: dict = None, getter_config: dict = None) -> None:

        # On initialise name
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        assert isinstance(self.name, str), "name should be of type str"

        # self.features_classes = features_classes
        self.fetch_config = fetch_config
        self.getter_config = getter_config

        # # On initialise feature_dir
        # if data_dir:
        #     if isinstance(data_dir, str):
        #         data_dir = Path(data_dir)
        # else:
        #     data_dir = Path(os.getcwd())

        # assert isinstance(data_dir, Path), "data_dir should be of type Path"
        # self.data_dir = data_dir / self.name
        # if not os.path.isdir(self.data_dir):
        #     os.makedirs(self.data_dir, exist_ok=True)

        # On initialise logger
        if logger is None:
            logger = logging.getLogger(self.name)
        self.logger: logging.Logger = logger
        assert isinstance(self.logger, type(sys.modules['logging'].getLogger(
        ))), f"logger must be of type logging.Logger, not {type(self.logger)}"

        self.logger.info("Initialisation de la classe %s", self.name)
        # Initialize each features object and fetch their data and then get them for the specified range
        self.features: List[Union[ft.BaseFeature, str]] = []
        self.targets_names = []

        self.data: pd.DataFrame = None
        self.enc_X_train: pd.DataFrame = pd.DataFrame()
        self.enc_X_val: pd.DataFrame = pd.DataFrame()
        self.enc_X_test: pd.DataFrame = pd.DataFrame()

        self.enc_data: pd.DataFrame = pd.DataFrame()

        self.X_train: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.DataFrame = pd.DataFrame()

        self.X_val: pd.DataFrame = pd.DataFrame()
        self.y_val: pd.DataFrame = pd.DataFrame()

        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_test: pd.DataFrame = pd.DataFrame()

        self.train_set: pd.DataFrame = pd.DataFrame()
        self.val_set: pd.DataFrame = pd.DataFrame()
        self.test_set: pd.DataFrame = pd.DataFrame()

        # super().__init__(config=config, parent=parent)

        self.initialize_features(features_classes)

        if fetch_config is not None:
            self.fetch_dataset(**fetch_config)

        if getter_config is not None:
            self.get_dataset(**getter_config, inplace=True)

    def initialize_features(self, features_classes) -> None:
        """
        Initialize each feature.

        Parameters:
        - None
        """
        self.logger.info("Initialisation des features")
        # Initialize each feature
        for feature_class in features_classes:
            # If the feature is a string, get the class from the features module
            if isinstance(feature_class, str):
                feature_class = getattr(ft, feature_class)

            # If the feature is a class, instantiate it, otherwise if it's an instance, use it
            if isinstance(feature_class, ft.BaseFeature):
                feature = feature_class

            else:
                feature = feature_class(logger=self.logger)

            self.features.append(feature)

    def fetch_dataset(self, data_start, data_stop, data_dir, locations: str | List[str] | Location | List[Location]) -> None:
        """
        Fetch les données.

        Parameters:
        - None
        """
        if data_dir is not None:
            if isinstance(data_dir, str):
                data_dir = Path(data_dir)
        else:
            data_dir = Path(os.getcwd()) / 'data'

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if isinstance(locations, str):
            locations = [locations]

        if isinstance(locations, Location):
            locations = [locations]

        assert isinstance(
            locations, list), "locations must be a list of strings or Location objects"
        assert all(isinstance(location, (str, Location))
                   for location in locations), "locations must be a list of strings or Location objects"
        self.logger.info("Fetching dataset")
        # Get data from each feature
        for feature in self.features:
            for location in locations:
                self.logger.info(
                    f"Fetching data for {feature.name} at {location}")
                feature.fetch_data(
                    data_start, data_stop, location=location, features_dir=data_dir / 'features')
                self.logger.setLevel(logging.INFO)
                # print(feature.date_max)

    def encode(self, pipeline: Pipeline = None) -> None:
        """
        Encode les données.

        Parameters:
        - pipeline: Pipeline - The pipeline to use for encoding the data
        """

        if pipeline is None:
            raise ValueError("Pipeline is required")

        encoded_data = []

        if not self.X_train.empty:
            not_to_encode = []
            # Do not encode data that are the same unit as the target
            for target in self.targets_names:
                not_to_encode.append(target.split(
                    '%%')[0].split('target_')[1].split('_')[0])
            not_to_encode.append('target_')
            # print(not_to_encode)

            to_encode = [col for col in self.X_train.columns if all(
                not col.startswith(prefix) for prefix in not_to_encode)]
            # print(to_encode)
            not_to_encode = [
                col for col in self.X_train.columns if col not in to_encode]
            # print(not_to_encode)
            print(
                f"X shape: {self.X_train[to_encode].shape}, y shape: {self.y_train.shape}")

            # print(self.y_train)
            self.enc_X_train = pipeline.fit_transform(
                X=self.X_train[to_encode], y=self.y_train)
            # print(self.enc_X_train)
            # print(self.enc_X_train.iloc[0])
            self.enc_X_train.columns = [
                col.split('__')[-1] for col in self.enc_X_train.columns]
            not_encoded_df = self.X_train[not_to_encode]
            self.logger.info(
                f'{len(not_encoded_df.columns.to_list())} features not encoded (same unit as target)')
            self.enc_X_train = pd.concat(
                [self.enc_X_train, not_encoded_df], axis=1)
            # print(self.enc_X_train.columns.to_list())

            # print(self.enc_X_train.columns.value_counts().loc[lambda x: x > 1])

            encoded_data.append(self.enc_X_train)
            if not self.X_val.empty:
                self.enc_X_val = pipeline.transform(self.X_val[to_encode])
                self.enc_X_val.columns = [
                    col.split('__')[-1] for col in self.enc_X_val.columns]
                self.enc_X_val = pd.concat(
                    [self.enc_X_val, self.X_val[not_to_encode]], axis=1)

                encoded_data.append(self.enc_X_val)

            if not self.X_test.empty:
                self.enc_X_test = pipeline.transform(self.X_test[to_encode])
                self.enc_X_test.columns = [
                    col.split('__')[-1] for col in self.enc_X_test.columns]
                self.enc_X_test = pd.concat(
                    [self.enc_X_test, self.X_test[not_to_encode]], axis=1)
                encoded_data.append(self.enc_X_test)

        self.enc_data = pd.concat(encoded_data, axis=0)

    def create_X_y(self, val=True, test=True) -> None:
        """
        Create X and y.

        Parameters:
        - None
        """

        if not self.train_set.empty:
            # print(self.train_set.columns.to_list())
            self.X_train = self.train_set.drop(columns=self.targets_names)
            self.y_train = self.train_set[self.targets_names]
        else:
            self.X_train = self.data.drop(columns=self.targets_names)
            self.y_train = self.data[self.targets_names]

        if val:
            if not self.val_set.empty:
                self.X_val = self.val_set.drop(columns=self.targets_names)
                self.y_val = self.val_set[self.targets_names]
            else:
                raise ValueError(
                    "Validation set is not available, you need to split the data first")

        if test:
            if not self.test_set.empty:
                self.X_test = self.test_set.drop(columns=self.targets_names)
                self.y_test = self.test_set[self.targets_names]
            else:
                raise ValueError(
                    "Test set is not available, you need to split the data first")

    def remove_constant_columns_from_splits(self):
        """
        Remove columns that are constant in either the train or test set.
        Ensures that the same columns are dropped from both sets.
        """
        # Identify constant columns in the train set
        train_constant_cols = list_constant_columns(
            self.train_set, threshold=1.0, exclude_booleans=True, exclude_categories=True)
        # test_constant_cols = []
        # val_constant_cols = []

        # if not self.test_set.empty:
        #     # Identify constant columns in the test set
        #     test_constant_cols = [col for col in self.test_set.columns if self.test_set[col].dtype != 'category' and self.test_set[col].std() == 0]

        # if not self.val_set.empty:
        #     # Identify constant columns in the val set
        #     val_constant_cols = [col for col in self.val_set.columns if self.val_set[col].dtype != 'category' and self.val_set[col].std() == 0]

        # Combine constant columns from both sets
        # constant_columns = set(train_constant_cols).union(set(test_constant_cols)).union(set(val_constant_cols))
        constant_columns = train_constant_cols
        # Drop constant columns from both sets
        self.train_set.drop(columns=constant_columns, inplace=True)
        if not self.test_set.empty:
            self.test_set.drop(columns=constant_columns, inplace=True)

        if not self.val_set.empty:
            self.val_set.drop(columns=constant_columns, inplace=True)

        self.data.drop(columns=constant_columns, inplace=True)

        dropped = [col.split('%%')[0] for col in constant_columns]
        dropped = set(dropped)

        print(
            f"Dropped {len(constant_columns)} constant columns from both sets: {dropped}")

    def split(self, train_size: Optional[Union[float, int]] = None,
              test_size: Optional[Union[float, int]] = None,
              val_size: Optional[Union[float, int]] = None,
              random_state: Optional[int] = None,
              shuffle: bool = True,
              stratify: Optional[Union[pd.Series, np.ndarray]] = None) -> tuple:
        """Split arrays or matrices into random train, test and val subsets.

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
        dataset = self.data
        self.val_set = pd.DataFrame()

        # On ne shuffle pas pour que test soit toujours situé à la fin de notre dataset
        train_val_set, test_set = train_test_split(
            dataset, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=False, stratify=stratify)
        self.train_set = train_val_set
        self.test_set = test_set

        if val_size:
            # ici on peut shuffle
            train_set, val_set = train_test_split(
                train_val_set, test_size=val_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
            self.train_set = train_set
            self.val_set = val_set

        self.remove_constant_columns_from_splits()

        return self.train_set, self.val_set, self.test_set

    def get_targets_names(self) -> List[str]:
        return self.targets_names

    def create_target(self,
                      targets: Union[str, List[str]],
                      targets_locations: List[Location],
                      bins: Optional[int] = None,
                      replace_target: Optional[bool] = True,
                      targets_shift: Optional[int] = None,
                      targets_rolling_window: Optional[int] = None,
                      targets_history_shifts: Optional[int] = [],
                      targets_history_rolling_windows: Optional[Union[int, List[int]]] = [
                      ],
                      axis=None) -> str:

        self.targets_names = []

        assert isinstance(targets, (list, str)
                          ), "targets must be a list of strings or a string"
        # TODO: check other parameters type

        if isinstance(targets, str):
            targets = [targets]

        for location in targets_locations:
            assert isinstance(
                location, Location), "targets_locations must be a list of Location"
            data = self.data.copy(deep=True)
            fully_named_targets = []

            #  Check if targets are in data, and suffix them with location name if not already specified
            # NOTE: no matter if the dataset is concatenated on rows or columns, we always specify targets locations
            for i, target in enumerate(targets):
                assert isinstance(
                    target, str), "targets must be a list of strings or a string"

                if axis is not None:
                    if not target.endswith(f'{location.name}'):
                        target += f'_{location.name}'

                    if axis == 'rows':
                        data = data[data['location'] == location.name]
                        data.rename(columns={targets[i]: target}, inplace=True)

                assert target in data.columns.to_list(
                ), f"Target {target} not in data"
                fully_named_targets.append(target)

            # Create a shift/rolling_mean of the colums and fill the self.targets_names attribute
            for target in fully_named_targets:
                if not target.startswith("target_"):
                    target_name = f"target_{target}"
                else:
                    target_name = target

                if (targets_shift is None or targets_shift >= 0):
                    if "%%" in target_name:
                        raise ValueError(
                            "Passed target name already point to a shifted/rolling column, please provide the base name")
                    else:
                        self.logger.warning("Not shifting the target is not allowed as some features might not be available for today's date,\
                                            \nas well as applying a positive shift as it would results in predicting the past\
                                            \nWill use a default shift of -1 if targets_shift == 0 or of -targets_shift if targets_shift < 0")
                        targets_shift = -1 if targets_shift == 0 or targets_shift is None else -targets_shift

                # df = pd.DataFrame()
                # df['brut'] = self.data[target]
                if targets_rolling_window is None or targets_rolling_window == 0:
                    if targets_shift != 0:
                        self.logger.info(
                            f"Creating the target columns as {target} shifted by {targets_shift}")
                        target_name = f"{target_name}%J{'+' if targets_shift < 0 else '-'}{str(abs(targets_shift))}"
                        data[target_name] = data[target].shift(targets_shift)
                        self.targets_names.append(target_name)
                        # df['shift_final'] = self.data[target_name]
                    targets_rolling_window = 0

                else:
                    self.logger.info(
                        f"Creating the target columns as a rolling mean of {target} on {targets_rolling_window} rows shifted by {targets_shift}")
                    # if targets_shift >= 0 or targets_shift is None or abs(targets_shift) < targets_rolling_window:
                    # Using a rolling mean of the target without shifting (or using a positive shift) will compute means on the past
                    # and might produce a target leak if history of the variable is used,
                    # Will use a initial shift of the size of the rolling window
                    target_name = f"{target_name}{'%J+' if targets_shift < 0 else '%J-' if targets_shift > 0 else ''}{str(abs(targets_shift))}%mean_{str(targets_rolling_window)}J"

                    if targets_shift != 0:
                        data[target_name] = data[target].shift(
                            targets_shift-targets_rolling_window)
                    else:
                        data[target_name] = data[target]
                    # df['shift'] = self.data[target_name]
                    # self.logger.info("Aggregating the target columns...")
                    data[target_name] = data[target_name].rolling(
                        targets_rolling_window).mean()
                    # df['rolling'] = self.data[target_name]
                    self.targets_names.append(target_name)

                self.logger.info("Creating target history columns...")
                min_shift = abs(targets_shift) + targets_rolling_window
                # min_shift = 0  # Use this if you want to use unavailable history
                for shift in targets_history_shifts:
                    if shift and shift < 0:
                        shift = -shift
                    if shift < min_shift:
                        self.logger.warning(f"{shift} as target history shift is not high enough considering that the target is shifted and/or is a rolling mean,\
                                            \ntarget_history_shift value will be set to {shift} + abs(targets_shift) + targets_rolling_window")
                        shift = shift + min_shift
                    # shift = shift -1
                    data[f"{target_name}%%J-{shift}"] = data[target_name].shift(
                        shift)
                    data[f"{target_name}%%J-{shift}"] = data[f"{target_name}%%J-{shift}"].bfill(
                        limit_area='outside')

                if targets_history_rolling_windows is not []:
                    targets_history_rolling_windows_str = [
                        str(r) for r in targets_history_rolling_windows]
                    s = ', '.join(targets_history_rolling_windows_str)
                    self.logger.info(
                        f"Rolling windows will start from {min_shift} samples before to {s} days before this starting sample")

                for rw in targets_history_rolling_windows:
                    data[f'{target_name}%%mean_{rw}J%%J-{min_shift+1}'] = data[target_name].shift(
                        min_shift).rolling(rw, closed="left").mean()
                    data[f'{target_name}%%mean_{rw}J%%J-{min_shift+1}'] = data[f'{target_name}%%mean_{rw}J%%J-{min_shift+1}'].bfill(
                        limit_area='outside')
                    data[f'{target_name}%%std{rw}J%%J-{min_shift+1}'] = data[target_name].shift(
                        min_shift).rolling(rw, closed="left").std()
                    data[f'{target_name}%%std{rw}J%%J-{min_shift+1}'] = data[f'{target_name}%%std{rw}J%%J-{min_shift+1}'].bfill(
                        limit_area='outside')

                if bins:
                    # print(data[target_name])
                    data = data.dropna(subset=[target_name])
                    self.logger.info(
                        "Categorizing the target columns...")
                    data = categorize(data, target_name,
                                      bins=bins, drop=replace_target)

            data = data.loc[:, data.columns.str.startswith(target_name)]
            self.data = self.data.merge(
                data, how="left", left_index=True, right_index=True)
        self.data = self.data.dropna(subset=self.targets_names)

    def get_data(self,
                 from_date: Optional[Union[str, dt.datetime]] = None,
                 to_date: Optional[Union[str, dt.datetime]] = None,
                 location: Optional[Location] = None,
                 freq: Optional[str] = None,
                 shift: Optional[int] = [],
                 rolling_window: Optional[Union[int, List[int]]] = [],
                 drop_constant_thr=1.0,
                 features_dir: Optional[Union[str, pathlib.Path]] = None,
                 filename: Optional[str] = None) -> pd.DataFrame:
        # if from_date is None:
        #     from_date = self.start
        # assert isinstance(from_date, dt.datetime) or isinstance(
        #     from_date, str), f"from_date must be of type datetime.datetime or a string, not {type(from_date)}"
        # if isinstance(from_date, str):
        #     try:
        #         from_date = dt.datetime.strptime(from_date, '%d-%m-%Y')
        #     except ValueError:
        #         raise ValueError(
        #             f"from_date must be of format %d-%m-%Y, not {from_date}")

        # if to_date is None:
        #     to_date = self.stop
        # assert isinstance(to_date, dt.datetime) or isinstance(
        #     to_date, str), f"to_date must be of type datetime.datetime or a string, not {type(to_date)}"
        # if isinstance(to_date, str):
        #     try:
        #         to_date = dt.datetime.strptime(to_date, '%d-%m-%Y')
        #     except ValueError:
        #         raise ValueError(
        #             f"to_date must be of format %d-%m-%Y, not {to_date}")

        # if freq is None:
        #     freq = self.freq

        # print(from_date, to_date, freq)
        date = pd.date_range(start=from_date, end=to_date, freq=freq)
        data = pd.DataFrame(index=date)
        data.index.name = 'date'

        # print(filename)

        features_data = []
        for feature in self.features:
            # print(feature.date_max)
            feature_data = feature.get_data(from_date=from_date, to_date=to_date, location=location, freq=freq, shift=shift,
                                            rolling_window=rolling_window, drop_constant_thr=drop_constant_thr, path=features_dir / feature.name, filename=filename)
            # print(feature_data)
            # data = data.join(feature_data, on='date', how='left')
            features_data.append(feature_data)
        # print(features_data)
        data = pd.concat(features_data, axis='columns')

        # print(data)

        return data

    # TODO : renommer shift en lags et rolling window en trend_windows
    def get_dataset(self, from_date: Optional[Union[str, dt.datetime]] = None,
                    to_date: Optional[Union[str, dt.datetime]] = None,
                    locations: Optional[Union[List[str],
                                              str, List[Location], Location]] = None,
                    axis: Optional[str] = None,
                    features_names: Optional[List[str]] = None,
                    freq: Optional[str] = None,
                    shift: Optional[int] = [],
                    target_bins: Optional[int] = None,
                    replace_target: Optional[bool] = True,
                    rolling_window: Optional[Union[int, List[int]]] = [],
                    drop_constant_thr=1.0,
                    data_dir: Optional[Union[str, pathlib.Path]] = None,
                    filename: Optional[str] = None,
                    split_config: Optional[dict] = {},
                    create_X_y: bool = True,
                    encoding_pipeline: Pipeline = None,
                    targets_names: Optional[List[str]] = None,
                    targets_shift: Optional[int] = None,
                    targets_rolling_window: Optional[Union[int,
                                                           List[int]]] = None,
                    targets_history_shifts: Optional[int] = [],
                    targets_history_rolling_windows: Optional[Union[int, List[int]]] = [
    ],
            targets_locations: Optional[Union[List[str],
                                              str, List[Location], Location]] = None,
            inplace: bool = False) -> None:
        """
        Get the dataset

        Parameters

        - from_date (Optional[Union[str, dt.datetime]]): Start date of the dataset
        - to_date (Optional[Union[str, dt.datetime]]): End date of the dataset
        - features_names (Optional[List[str]]): List of features names to keep in the dataset
        - freq (Optional[str]): Frequency of the dataset
        - shift (Optional[int]): Add lagged features to each features
        - rolling_window (Optional[Union[int, List[int]]]): Add mean and std of each features in a rolling window
        - split_config (Optional[dict]): Configuration of the splitting process
        - create_X_y (bool): Create X and y
        - encoding_pipeline (Optional[Pipeline]): Pipeline to encode categorical features
        - targets_shift (Optional[int]): Shift targets by this number of periods
        - targets_rolling_window (Optional[Union[int, List[int]]]): Replace targets by a rolling window
        - inplace (bool): Modify the dataset in place

        """
        # TODO: set default dates and freq in function of the features availaibility
        assert isinstance(locations, (list, str, Location)
                          ), "locations must be a list of strings, Location objects or a string or a Location object"

        if not isinstance(locations, list):
            locations = [locations]
        for i in range(len(locations)):
            assert isinstance(
                locations[i], (str, Location)), "locations must be a list of strings or Location objects"
            if isinstance(locations[i], str):
                locations[i] = Location(locations[i])

        self.logger.info(
            f"Getting the dataset from {from_date} to {to_date} for {', '.join([location.name for location in locations])}")

        assert isinstance(targets_locations, (list, str, Location)
                          ), "targets_locations must be a list of strings, Location objects or a string or a Location object"

        if not isinstance(targets_locations, list):
            targets_locations = [targets_locations]

        for i in range(len(targets_locations)):
            assert isinstance(targets_locations[i], (
                str, Location)), "targets_locations must be a list of strings or of Location objects"
            if isinstance(targets_locations[i], str):
                targets_locations[i] = Location(targets_locations[i])

        # Make sure that targets_names is a list of strings
        targets_names = targets_names if isinstance(
            targets_names, list) else [targets_names]

        data = []
        if axis is None and len(locations) > 1:
            raise ValueError(
                "axis can't be None if you're getting the dataset for multiple locations")
        # Filtrage des données en fonction des dates et des colonnes demandées
        for location in locations:
            df = self.get_data(from_date=from_date,
                               to_date=to_date,
                               location=location,
                               freq=freq,
                               shift=shift,
                               rolling_window=rolling_window,
                               drop_constant_thr=drop_constant_thr,
                               features_dir=data_dir / 'features',
                               filename=filename)

            if axis == 'columns':
                df = df.add_suffix(f"_{location.name}")
            elif axis == 'rows':
                df['location'] = location.name

            data.append(df)
        if axis is not None:
            if axis == 'columns':
                self.data = pd.concat(data, axis='columns')
            elif axis == 'rows':
                self.data = pd.concat(data, axis='rows')
                self.data = self.data.sort_index()
                self.data['location'] = self.data['location'].astype(
                    'category')
            else:
                raise ValueError("axis must be 'columns' or 'rows'")
        else:
            self.data = data[0]

        # extraire l'index de la nouvelle data si il n'est pas déjà présent
        if self.data.index.name not in self.data.columns:
            new_column = {str(self.data.index.name): self.data.index}
            self.data = self.data.assign(**new_column)
            # self.data.index.rename('index', inplace=True)
        # Créer les targets
        self.create_target(targets=targets_names,
                           targets_locations=targets_locations,
                           targets_shift=targets_shift,
                           targets_rolling_window=targets_rolling_window,
                           targets_history_shifts=targets_history_shifts,
                           targets_history_rolling_windows=targets_history_rolling_windows, bins=target_bins, replace_target=replace_target,
                           axis=axis)

        # Si features_names est fourni, on retire les suffixes spécifiant l'encodage si présent dans le nom des colonnes
        if features_names is not None:
            features_names = [extract_column_base(ft) for ft in features_names]
            # features_names_base = find_matching_columns(self.data, features_names_base, to_print=True)
            # Si les targets ne sont pas dans les features, on les ajoute

            for target in self.targets_names:
                if target not in features_names:
                    self.logger.warning(
                        "Target %s not in features_names, adding it to continue", target)
                    features_names.append(target)

        else:
            features_names = self.data.columns.to_list()

        features_names = list(set(features_names))

        # print(self.data.columns.to_list())
        # print(features_names)
        self.data = self.data[features_names]
        self.data.sort_index(axis=1, inplace=True)

        # print(self.data.columns.to_list())
        # # Update des noms des targets si il y a eu aggregation
        # TODO: à faire plus haut
        # new_dataset.targets_names = []
        # print(self.targets_names)
        # print(new_dataset.data.columns.to_list())
        # for col in new_dataset.data.columns:
        #     if col in self.targets_names or any(col.split('~~')[0] == target for target in self.targets_names):
        #         new_dataset.targets_names.append(col)
        #         print(new_dataset.targets_names)

        # def needs_recalculation(self, new_features, new_dates, new_freq, shift, rolling_window, targets_names):
        #     # Vérification des colonnes en utilisant les noms transformés
        #     current_features = [extract_column_base(
        #         col) for col in self.data.columns]
        #     new_features_base = [extract_column_base(f) for f in new_features]

        #     # Si les features demandées ne sont pas présentes dans les colonnes existantes ou si les paramètres changent
        #     if set(new_features_base) - set(current_features):
        #         # print(set(new_features_base) - set(current_features))
        #         return True

        #     # Si les dates ou la fréquence changent, il faut recalculer
        #     if new_dates != (self.start, self.stop) or new_freq != self.freq:
        #         print(new_dates, (self.start, self.stop), new_freq, self.freq)
        #         return True

        #     # Vérification du shift et du rolling_window
        #     if shift != self.shift or rolling_window != self.rolling_window:
        #         print(shift, self.shift, rolling_window, self.rolling_window)
        #         return True

        #     # Si les targets ont changés, il faut recalculer
        #     if set(targets_names) != set(self.targets_names):
        #         return True

        #     return False

        # # TODO: Attention, il faut recalculer si les targets ont changé, la pipeline d'encodage, les dates, la fréquence, le shift, le rolling_window, les features, les splits, les encodings, etc.
        # # Vérifier si un recalcul est nécessaire
        # if needs_recalculation(self, features_names_base, (from_date, to_date), freq, shift, rolling_window):
        self.logger.info(
            "Calculating train/val/test sets and encodings...")

        self.name += f"_{'_'.join(self.targets_names)}"

        # Splitting the dataset
        if split_config:
            self.split(**split_config)

        # Création des X et y
        if create_X_y:
            self.create_X_y()

        # print(new_dataset.X_train['date'])

        # Encodage si nécessaire
        if encoding_pipeline:
            if isinstance(encoding_pipeline, dict):
                encoding_pipeline = create_encoding_pipeline(
                    encoding_pipeline)

            self.encode(pipeline=encoding_pipeline)
        # else:
        #     # Réutilisation des attributs existants si aucune modification majeure n'est nécessaire
        #     self.logger.info("Re-using existing splits and encodings...")
        #     # Re-utiliser les splits en vérifiant les colonnes correspondantes
        #     if not self.X_train.empty:
        #         matching_columns_train = find_matching_columns(
        #             self.X_train, features_names_base, to_print=False)
        #         new_dataset.X_train = self.X_train[matching_columns_train]
        #         new_dataset.y_train = self.y_train

        #         if not self.X_val.empty:
        #             matching_columns_val = find_matching_columns(
        #                 self.X_val, features_names_base, to_print=False)
        #             new_dataset.X_val = self.X_val[matching_columns_val]
        #             new_dataset.y_val = self.y_val

        #         if not self.X_test.empty:
        #             matching_columns_test = find_matching_columns(
        #                 self.X_test, features_names_base, to_print=False)
        #             new_dataset.X_test = self.X_test[matching_columns_test]
        #             new_dataset.y_test = self.y_test

        #     # Re-utiliser les sets en vérifiant les colonnes correspondantes
        #     if not self.train_set.empty:
        #         matching_columns_train_set = find_matching_columns(
        #             self.train_set, features_names_base, to_print=False)
        #         new_dataset.train_set = self.train_set[matching_columns_train_set]

        #         if not self.val_set.empty:
        #             matching_columns_val_set = find_matching_columns(
        #                 self.val_set, features_names_base, to_print=False)
        #             new_dataset.val_set = self.val_set[matching_columns_val_set]

        #         if not self.test_set.empty:
        #             matching_columns_test_set = find_matching_columns(
        #                 self.test_set, features_names_base, to_print=False)
        #             new_dataset.test_set = self.test_set[matching_columns_test_set]

        #     # Re-utiliser les encodings en vérifiant les colonnes correspondantes
        #     if not self.enc_X_train.empty:
        #         matching_columns_enc_train = find_matching_columns(
        #             self.enc_X_train, features_names_base, to_print=False)
        #         new_dataset.enc_X_train = self.enc_X_train[matching_columns_enc_train]
        #         new_dataset.enc_data = self.enc_data[matching_columns_enc_train]

        #         if not self.enc_X_val.empty:
        #             matching_columns_enc_val = find_matching_columns(
        #                 self.enc_X_val, features_names_base, to_print=False)
        #             new_dataset.enc_X_val = self.enc_X_val[matching_columns_enc_val]

        #         if not self.enc_X_test.empty:
        #             matching_columns_enc_test = find_matching_columns(
        #                 self.enc_X_test, features_names_base, to_print=False)
        #             new_dataset.enc_X_test = self.enc_X_test[matching_columns_enc_test]

        # if not inplace:
        #     return new_dataset

    def save_dataset(self, dataset_dir: str = None) -> None:
        """
        Save the dataset.

        Parameters:
        - root_dir: str - The root directory where to save the dataset
        - name: str - The name of the dataset
        """

        if dataset_dir is None:
            dataset_dir = self.data_dir

        dataset_dir = Path(dataset_dir)

        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True)

        self.logger.info("Saving the raw dataset...")
        # self.save_dataframe(path=dataset_dir, data=self.data, filename="data.feather")

        if not self.train_set.empty:
            self.logger.info("Saving splits...")
            if not dataset_dir.joinpath("train").exists():
                (dataset_dir / "train").mkdir(parents=True)
            self.save_dataframe(path=dataset_dir / "train",
                                filename="train_set.feather", data=self.train_set)

            if not self.val_set.empty:
                if not dataset_dir.joinpath("validation").exists():
                    (dataset_dir / "validation").mkdir(parents=True)
                self.save_dataframe(
                    path=dataset_dir / "validation", filename="val_set.feather", data=self.val_set)
            if not self.test_set.empty:
                if not dataset_dir.joinpath("test").exists():
                    (dataset_dir / "test").mkdir(parents=True)
                self.save_dataframe(
                    path=dataset_dir / "test", filename="test_set.feather", data=self.test_set)

        if not self.X_train.empty:
            self.logger.info("Saving X and y...")
            self.save_dataframe(path=dataset_dir / "train",
                                filename="X_train.feather", data=self.X_train)
            self.save_dataframe(path=dataset_dir / "train",
                                filename="y_train.feather", data=self.y_train)

            if not self.X_val.empty:
                self.save_dataframe(
                    path=dataset_dir / "validation", filename="X_val.feather", data=self.X_val)
                self.save_dataframe(
                    path=dataset_dir / "validation", filename="y_val.feather", data=self.y_val)
            if not self.X_test.empty:
                self.save_dataframe(path=dataset_dir / "test",
                                    filename="X_test.feather", data=self.X_test)
                self.save_dataframe(path=dataset_dir / "test",
                                    filename="y_test.feather", data=self.y_test)

        if not self.enc_X_train.empty:
            self.logger.info("Saving encodings...")
            self.save_dataframe(path=dataset_dir / "train",
                                filename="enc_X_train.feather", data=self.enc_X_train)

            if not self.enc_X_val.empty:
                self.save_dataframe(path=dataset_dir / "validation",
                                    filename="enc_X_val.feather", data=self.enc_X_val)
            if not self.enc_X_test.empty:
                self.save_dataframe(
                    path=dataset_dir / "test", filename="enc_X_test.feather", data=self.enc_X_test)

            self.save_dataframe(
                path=dataset_dir, filename="enc_data.feather", data=self.enc_data)
