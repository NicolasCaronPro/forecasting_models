from pathlib import Path
from typing import Optional, Union, List
from sklearn.pipeline import Pipeline
import src.features as ft
from src.encoding.tools import create_encoding_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
import datetime as dt
import numpy as np
import logging


class BaseTabularDataset(ft.BaseFeature):
    def __init__(self, target_colomns: Union[List[str], str], features_class: List[Union[ft.BaseFeature, str]], config: Optional['ft.Config'] = None, parent: Optional['ft.BaseFeature'] = None) -> None:
        # Initialize each features object and fetch their data and then get them for the specified range

        # Get target
        self.targets = target_colomns if isinstance(
            target_colomns, list) else [target_colomns]
        # self.chained_targets = chained_targets
        self.features: List[Union[ft.BaseFeature, str]] = []

        # Initialize each feature

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
        self.name = self.__class__.__name__
        self.name += f"_{'_'.join(self.targets)}"
        super().__init__(config=config, parent=parent, name=self.name)


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
            if not feature.is_fetched:
                self.logger.info(f"Fetching data for {feature.name}")
                self.logger.setLevel(logging.WARNING)
                feature.fetch_data(save=True)
                self.logger.setLevel(logging.INFO)
            self.data = self.data.join(feature.data)

        self.data = self.data.dropna(subset=self.targets)


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
            # Do not encode data that are the same unit as the target
            to_encode = [col for col in self.X_train.columns if not col.startswith('Total_CHU Dijon') and not col.startswith('nb_vers_hospit')]
            not_to_encode = [col for col in self.X_train.columns if col.startswith('Total_CHU Dijon') or col.startswith('nb_vers_hospit')]

            self.enc_X_train = pipeline.fit_transform(
                X=self.X_train[to_encode], y=self.y_train)
            # print(self.enc_X_train.iloc[0])
            self.enc_X_train.columns = [
                col.split('__')[-1] for col in self.enc_X_train.columns]
            
            self.enc_X_train = pd.concat([self.enc_X_train, self.X_train[not_to_encode]], axis=1)
            # print(self.enc_X_train.columns.to_list())

            # print(self.enc_X_train.columns.value_counts().loc[lambda x: x > 1])

            encoded_data.append(self.enc_X_train)
            if not self.X_val.empty:
                self.enc_X_val = pipeline.transform(self.X_val[to_encode])
                self.enc_X_val.columns = [
                    col.split('__')[-1] for col in self.enc_X_val.columns]
                self.enc_X_val = pd.concat([self.enc_X_val, self.X_val[not_to_encode]], axis=1)

                encoded_data.append(self.enc_X_val)

            if not self.X_test.empty:
                self.enc_X_test = pipeline.transform(self.X_test[to_encode])
                self.enc_X_test.columns = [
                    col.split('__')[-1] for col in self.enc_X_test.columns]
                self.enc_X_test = pd.concat([self.enc_X_test, self.X_test[not_to_encode]], axis=1)
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
        dataset = self.data
        self.val_set = pd.DataFrame()

        train_val_set, test_set = train_test_split(
            dataset, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        self.train_set = train_val_set
        self.test_set = test_set

        if val_size:
            train_set, val_set = train_test_split(
                train_val_set, test_size=val_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
            self.train_set = train_set
            self.val_set = val_set


        return self.train_set, self.val_set, self.test_set

    # def get_dataset(self, from_date: Optional[Union[str, dt.datetime]] = None,
    #                 to_date: Optional[Union[str, dt.datetime]] = None,
    #                 features_names: Optional[List[str]] = None,
    #                 freq: Optional[str] = '1D',
    #                 shift: Optional[int] = 0,
    #                 rolling_window: Optional[Union[int, List[int]]] = 0,
    #                 split_config: Optional[dict] = {},
    #                 create_X_y: bool = True,
    #                 encoding_pipeline: Pipeline = None) -> 'BaseTabularDataset':
    #     """
    #     Get the data.

    #     Parameters:
    #     - None
    #     """
    #     # print(self.enc_X_train.columns.value_counts().loc[lambda x: x > 1])

    #     self.logger.info("Getting the dataset...")
    #     # TODO: Gérer le cas ou self est un dataset déjà splité/encodé, et que l'on appel cette méthode avec des paramètres différents (notamment les dates et les fréquences)
    #     # if features_names is not None:
    #     #     features_names = list({ft.split('##')[0] if '##' in ft else ft for ft in features_names})

    #         # # si les targets ne sont pas dans les features, on les ajoute
    #         # if not any([target in features_names for target in self.targets]):
    #         #     features_names.extend(self.targets)

    #     filtered_data = self.get_data(from_date=from_date, to_date=to_date,
    #                                   features_names=features_names, freq=freq, shift=shift, rolling_window=rolling_window)

    #     print(filtered_data.columns.to_list())

    #     new_dataset: BaseTabularDataset = deepcopy(self)

    #     # Add index to the filtered data
    #     kwargs = {str(filtered_data.index.name): filtered_data.index}
    #     new_dataset.data = filtered_data.assign(**kwargs)
    #     new_dataset.start = from_date
    #     new_dataset.stop = to_date

    #     # Update the target columns in the new dataset
    #     new_dataset.targets = []
    #     # print(self.targets)
    #     # print(new_dataset.data.columns)
    #     for col in new_dataset.data.columns:
    #         if '~~' in col:
    #             # print(col.split('~~')[0])
    #             if any(col.split('~~')[0] in self.targets) or col in self.targets:
    #                 new_dataset.targets.append(col)
    #                 # new_dataset.data.pop(col)
    #         else:
    #             # print(col)
    #             if col in self.targets:
    #                 new_dataset.targets.append(col)
    #                 # new_dataset.data.pop(col)
    #     # print(new_dataset.targets)
    #     # self.chained_targets = True

    #     # if not features_names:
    #     #     features_names = new_dataset.get_features_names()
    #     #     # features_names = [col for col in features_names if col not in new_dataset.targets]
    #     #     # print(features_names)

    #     # We should also update other attributes like enc_X_train, enc_X_val, enc_X_test, X_train, y_train, X_val, y_val, X_test, y_test, train_set, val_set, test_set if they are not None and if the features_names parameter is the only parameter passed
    #     # On copie ou on calcule les attributs de la nouvelle instance
    #     # if not self.train_set.empty:
    #     #     self.logger.info('Re-using the train, val and test sets...')
    #     #     new_dataset.train_set = self.train_set[features_names+self.targets]
    #     #     if not self.val_set.empty:
    #     #         new_dataset.val_set = self.val_set[features_names+self.targets]
    #     #     if not self.test_set.empty:
    #     #         new_dataset.test_set = self.test_set[features_names+self.targets]
    #     # else:
    #     if split_config:
    #         new_dataset.split(**split_config)

    #     # if not self.X_train.empty:
    #     #     self.logger.info('Re-using the X_y splits...')
    #     #     print(features_names)
    #     #     new_dataset.X_train = self.X_train[features_names]

    #     #     if not self.X_val.empty:
    #     #         new_dataset.X_val = self.X_val[features_names]
    #     #     if not self.X_test.empty:
    #     #         new_dataset.X_test = self.X_test[features_names]
    #     # else:
    #     if create_X_y:
    #         new_dataset.create_X_y()

    #     # if not self.enc_X_train.empty:
    #     #     self.logger.info('Re-using the encodings...')
    #     #     features_names = [col for col in self.enc_X_train.columns if col.split('##')[0] in features_names or col in features_names]
    #     #     new_dataset.enc_X_train = self.enc_X_train[features_names]
    #     #     if not self.enc_X_val.empty:
    #     #         new_dataset.enc_X_val = self.enc_X_val[features_names]
    #     #     if not self.enc_X_test.empty:
    #     #         new_dataset.enc_X_test = self.enc_X_test[features_names]
    #     #     self.enc_data = self.enc_data[features_names]
    #     # else:
    #     if encoding_pipeline:
    #         if isinstance(encoding_pipeline, dict):
    #             encoding_pipeline = create_encoding_pipeline(encoding_pipeline)
    #         new_dataset.encode(pipeline=encoding_pipeline)

    #         #         afficher le nombre de colonnes portant le même nom si il est supérieur à 1
    #         # print(new_dataset.enc_data.columns.value_counts().loc[lambda x: x > 1])

    #     return new_dataset

    def get_features_names(self) -> List[str]:
        columns = super().get_features_names()

        # Retirer les targets
        columns = [col for col in columns if col not in self.targets]

        return columns

    def get_targets_names(self) -> List[str]:
        return self.targets
    
    def get_data(self, from_date: Optional[Union[str, dt.datetime]] = None, to_date: Optional[Union[str, dt.datetime]] = None, features_names: Optional[List[str]] = None, freq: Optional[str] = None, shift: Optional[int] = None, rolling_window: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        date = pd.date_range(start=from_date, end=to_date, freq=freq)
        data = pd.DataFrame(index=date)
        data.index.name = 'date'
        # print(data)
        for feature in self.features:
            feature_data = feature.get_data(from_date=from_date, to_date=to_date, freq=freq, shift=shift, rolling_window=rolling_window)
            # print(feature_data)
            data = data.join(feature_data, on='date', how='left')

        # extraire l'index de la nouvelle data si il n'est pas déjà présent
        if data.index.name not in data.columns:
            kwargs = {str(data.index.name): data.index}
            data = data.assign(**kwargs)
        # print(data)
        
        return data

    def get_dataset(self, from_date: Optional[Union[str, dt.datetime]] = None,
                    to_date: Optional[Union[str, dt.datetime]] = None,
                    features_names: Optional[List[str]] = None,
                    freq: Optional[str] = None,
                    shift: Optional[int] = None,
                    rolling_window: Optional[Union[int, List[int]]] = None,
                    split_config: Optional[dict] = {},
                    create_X_y: bool = True,
                    encoding_pipeline: Pipeline = None) -> 'BaseTabularDataset':
        """
        Get the data.
        """
        self.logger.info(f"Getting the dataset from {from_date} to {to_date}...")

        if from_date is None:
            from_date = self.start
        assert isinstance(from_date, dt.datetime) or isinstance(
            from_date, str), f"from_date must be of type datetime.datetime or a string, not {type(from_date)}"
        if isinstance(from_date, str):
            try:
                from_date = dt.datetime.strptime(from_date, '%d-%m-%Y')
            except ValueError:
                raise ValueError(
                    f"from_date must be of format %d-%m-%Y, not {from_date}")

        if to_date is None:
            to_date = self.stop
        assert isinstance(to_date, dt.datetime) or isinstance(
            to_date, str), f"to_date must be of type datetime.datetime or a string, not {type(to_date)}"
        if isinstance(to_date, str):
            try:
                to_date = dt.datetime.strptime(to_date, '%d-%m-%Y')
            except ValueError:
                raise ValueError(
                    f"to_date must be of format %d-%m-%Y, not {to_date}")

        if freq is None:
            freq = self.freq

        if shift is None:
            shift = self.shift

        if rolling_window is None:
            rolling_window = self.rolling_window

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

        # print(self.data.columns.to_list())
        # Si features_names est fourni, on retire les transformations des colonnes
        if features_names is not None:
            features_names_base = [
                extract_column_base(ft) for ft in features_names]
            # print(features_names_base)
            # features_names_base = find_matching_columns(self.data, features_names_base, to_print=True)
            # print(features_names_base)

            # Si les targets ne sont pas dans les features, on les ajoute
            for target in self.targets:
                if target not in features_names_base:
                    features_names_base.append(target)
        else:
            features_names_base = None

        # Filtrage des données en fonction des dates et des colonnes demandées
        new_data = self.get_data(from_date=from_date, to_date=to_date,
                                 features_names=features_names_base, freq=freq, shift=shift, rolling_window=rolling_window)
        
        # print(new_data)
        
        if features_names_base is None:
            features_names_base = new_data.columns.to_list()

        # print(features_names_base)
        # print([col for col in self.data.columns.to_list()])
        new_dataset: BaseTabularDataset = deepcopy(self)

        # Assignation des données filtrées
        new_dataset.data = new_data
        new_dataset.start = from_date
        new_dataset.stop = to_date
        new_dataset.freq = freq
        new_dataset.shift = shift
        new_dataset.rolling_window = rolling_window

        # Initialisation des targets
        new_dataset.targets = []
        for col in new_dataset.data.columns:
            if col in self.targets or any(col.split('~~')[0] == target for target in self.targets):
                new_dataset.targets.append(col)


        def needs_recalculation(self, new_features, new_dates, new_freq, shift, rolling_window):
            # Vérification des colonnes en utilisant les noms transformés
            current_features = [extract_column_base(
                col) for col in self.data.columns]
            new_features_base = [extract_column_base(f) for f in new_features]

            # Si les features demandées ne sont pas présentes dans les colonnes existantes ou si les paramètres changent
            if set(new_features_base) - set(current_features):
                # print(set(new_features_base) - set(current_features))
                return True

            # Si les dates ou la fréquence changent, il faut recalculer
            if new_dates != (self.start, self.stop) or new_freq != self.freq:
                print(new_dates, (self.start, self.stop), new_freq, self.freq)
                return True

            # Vérification du shift et du rolling_window
            if shift != self.shift or rolling_window != self.rolling_window:
                print(shift, self.shift, rolling_window, self.rolling_window)
                return True

            return False

        # Vérifier si un recalcul est nécessaire
        if needs_recalculation(self, features_names_base, (from_date, to_date), freq, shift, rolling_window):
            self.logger.info(
                "Re-calculating train/val/test sets and encodings...")

            # Splitting the dataset
            if split_config:
                new_dataset.split(**split_config)

            # Création des X et y
            if create_X_y:
                new_dataset.create_X_y()

            # print(new_dataset.X_train['date'])

            # Encodage si nécessaire
            if encoding_pipeline:
                if isinstance(encoding_pipeline, dict):
                    encoding_pipeline = create_encoding_pipeline(
                        encoding_pipeline)

                # print(encoding_pipeline)
                new_dataset.encode(pipeline=encoding_pipeline)
        else:
            # Réutilisation des attributs existants si aucune modification majeure n'est nécessaire
            self.logger.info("Re-using existing splits and encodings...")

            # Re-utiliser les splits en vérifiant les colonnes correspondantes
            if not self.X_train.empty:
                matching_columns_train = find_matching_columns(
                    self.X_train, features_names_base, to_print=False)
                new_dataset.X_train = self.X_train[matching_columns_train]
                new_dataset.y_train = self.y_train

                if not self.X_val.empty:
                    matching_columns_val = find_matching_columns(
                        self.X_val, features_names_base, to_print=False)
                    new_dataset.X_val = self.X_val[matching_columns_val]

                if not self.X_test.empty:
                    matching_columns_test = find_matching_columns(
                        self.X_test, features_names_base, to_print=False)
                    new_dataset.X_test = self.X_test[matching_columns_test]

            # Re-utiliser les sets en vérifiant les colonnes correspondantes
            if not self.train_set.empty:
                matching_columns_train_set = find_matching_columns(
                    self.train_set, features_names_base, to_print=False)
                new_dataset.train_set = self.train_set[matching_columns_train_set]

                if not self.val_set.empty:
                    matching_columns_val_set = find_matching_columns(
                        self.val_set, features_names_base, to_print=False)
                    new_dataset.val_set = self.val_set[matching_columns_val_set]

                if not self.test_set.empty:
                    matching_columns_test_set = find_matching_columns(
                        self.test_set, features_names_base, to_print=False)
                    new_dataset.test_set = self.test_set[matching_columns_test_set]

            # Re-utiliser les encodings en vérifiant les colonnes correspondantes
            if not self.enc_X_train.empty:
                matching_columns_enc_train = find_matching_columns(
                    self.enc_X_train, features_names_base, to_print=False)
                new_dataset.enc_X_train = self.enc_X_train[matching_columns_enc_train]
                new_dataset.enc_data = self.enc_data[matching_columns_enc_train]

                if not self.enc_X_val.empty:
                    matching_columns_enc_val = find_matching_columns(
                        self.enc_X_val, features_names_base, to_print=False)
                    new_dataset.enc_X_val = self.enc_X_val[matching_columns_enc_val]

                if not self.enc_X_test.empty:
                    matching_columns_enc_test = find_matching_columns(
                        self.enc_X_test, features_names_base, to_print=False)
                    new_dataset.enc_X_test = self.enc_X_test[matching_columns_enc_test]

        return new_dataset

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
        self.save_dataframe()

        if not self.train_set.empty:
            self.logger.info("Saving splits...")
            if not dataset_dir.joinpath("train").exists():
                (dataset_dir / "train").mkdir(parents=True)
            self.save_dataframe(path=dataset_dir / "train", filename="train_set.feather", data=self.train_set)

            if not self.val_set.empty:
                if not dataset_dir.joinpath("validation").exists():
                    (dataset_dir / "validation").mkdir(parents=True)
                self.save_dataframe(path=dataset_dir / "validation", filename="val_set.feather", data=self.val_set)
            if not self.test_set.empty:
                if not dataset_dir.joinpath("test").exists():
                    (dataset_dir / "test").mkdir(parents=True)
                self.save_dataframe(path=dataset_dir / "test", filename="test_set.feather", data=self.test_set)
        
        if not self.X_train.empty:
            self.logger.info("Saving X and y...")
            self.save_dataframe(path=dataset_dir / "train", filename="X_train.feather", data=self.X_train)
            self.save_dataframe(path=dataset_dir / "train", filename="y_train.feather", data=self.y_train)

            if not self.X_val.empty:
                self.save_dataframe(path=dataset_dir / "validation", filename="X_val.feather", data=self.X_val)
                self.save_dataframe(path=dataset_dir / "validation", filename="y_val.feather", data=self.y_val)
            if not self.X_test.empty:
                self.save_dataframe(path=dataset_dir / "test", filename="X_test.feather", data=self.X_test)
                self.save_dataframe(path=dataset_dir / "test", filename="y_test.feather", data=self.y_test)

        if not self.enc_X_train.empty:
            self.logger.info("Saving encodings...")
            self.save_dataframe(path=dataset_dir / "train", filename="enc_X_train.feather", data=self.enc_X_train)

            if not self.enc_X_val.empty:
                self.save_dataframe(path=dataset_dir / "validation", filename="enc_X_val.feather", data=self.enc_X_val)
            if not self.enc_X_test.empty:
                self.save_dataframe(path=dataset_dir / "test", filename="enc_X_test.feather", data=self.enc_X_test)

            self.save_dataframe(path=dataset_dir, filename="enc_data.feather", data=self.enc_data)