import math
import pathlib
import pandas as pd
import datetime as dt
import os
import sys
from typing import List, Optional, Union
from copy import deepcopy
import matplotlib.pyplot as plt
import abc
import numpy as np
import logging
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def relative_frequency(series: pd.Series) -> pd.Series:
    # Compter les occurrences de chaque catégorie
    counts = (series.value_counts(normalize=True) * 100)

    # Retourner la proportion (fréquence relative)
    return counts


def majority_rule(x):
    mode_val = x.mode()
    if len(mode_val) > 1:  # Si plusieurs valeurs sont à égalité, on choisit la première alphabétiquement
        return sorted(mode_val)[0]
    return mode_val[0]


class BaseFeature(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, name:str = None, logger=None) -> None:

        # On initialise name
        if name is None:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name
        assert isinstance(self.name, str), "name should be of type str"

        # On initialise logger
        if logger is None:
            logger = logging.getLogger(self.name)
        self.logger: logging.Logger = logger
        # assert isinstance(self.logger, type(sys.modules['logging'].getLogger(
        # ))), f"logger must be of type logging.Logger, not {type(self.logger)}"

        self.logger.info("Initialisation de la classe %s",self.name)

        # assert 'root_dir' in self.config, "root_dir must be provided in config"
        # self.data_dir = self.config.get("root_dir")
        # assert isinstance(self.data_dir, str) or isinstance(self.data_dir, type(
        #     pathlib.Path())), f"root_dir must be a string or a pathlib Path, not {type(self.data_dir)}"

        # if isinstance(self.data_dir, str):
        #     self.data_dir = pathlib.Path(self.data_dir)

        # assert self.data_dir.is_dir(
        # ), f"root_dir must be a directory, not {self.data_dir}"

        # self.data_dir = pathlib.Path(self.config.get("root_dir")) / 'data'

        # if 'features' in self.name.lower():
        #     self.data_dir = self.data_dir / 'features'
        # self.data_dir = self.data_dir / self.name.lower()
        # self.data_dir.mkdir(exist_ok=True, parents=True)

        # TODO: REMOVE THE FOLLOWING LINES (those parameters shouldn't be passed, but determined in function of the data fetched)
        # assert 'start' in self.config, "start must be provided in config"
        # self.start = self.config.get('start')
        # assert isinstance(self.start, dt.datetime) or isinstance(
        #     self.start, str), f"start must be of type datetime.datetime or a string, not {type(self.start)}"

        # if isinstance(self.start, str):
        #     try:
        #         self.start = dt.datetime.strptime(self.start, '%d-%m-%Y')
        #     except ValueError:
        #         raise ValueError(
        #             f"start must be of format %d-%m-%Y, not {self.start}")

        # if 'shift' in self.config:
        #     self.shift = self.config.get('shift')
        #     assert isinstance(
        #         self.shift, int), f"shift must be of type int, not {type(self.shift)}"

        # if 'rolling_window' in self.config:
        #     self.rolling_window = self.config.get('rolling_window')
        #     assert isinstance(
        #         self.rolling_window, int), f"rolling_window must be of type int, not {type(self.rolling_window)}"

        # assert 'step_unit' in self.config, "step_unit must be provided in config"
        # self.step_unit = self.config.get('step_unit')
        # assert self.step_unit in ['days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours',
        #                           'weeks'], f"step_unit must be one of ['days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours', 'weeks'], not {self.step_unit}"

        # assert 'step_value' in self.config, "step_value must be provided in config"
        # self.step_value = self.config.get('step_value')
        # assert isinstance(
        #     self.step_value, int), f"step_value must be an integer, not {type(self.step_value)}"

        # self.start -= dt.timedelta(**{self.step_unit: self.step_value *
        #                            max(self.shift, self.rolling_window, 0)})

        # assert 'stop' in self.config, "stop must be provided in config"
        # self.stop = self.config.get('stop')
        # assert isinstance(self.stop, dt.datetime) or isinstance(
        #     self.stop, str), f"stop must be of type datetime.datetime or a string, not {type(self.stop)}"

        # if isinstance(self.stop, str):
        #     try:
        #         self.stop = dt.datetime.strptime(self.stop, '%d-%m-%Y')
        #     except ValueError:
        #         raise ValueError(
        #             f"stop must be of format %d-%m-%Y, not {self.stop}")

        # self.step = {self.step_unit: self.step_value}
        # self.index: pd.Index = pd.Index(data=more_itertools.numeric_range(
        #     self.start, self.stop + dt.timedelta(**self.step), dt.timedelta(**self.step)), name='date')
        # self.data: pd.DataFrame = pd.DataFrame(index=self.index)

        ##########################################################################################################################################

        # if 'max_nan' in self.config:
        #     self.max_nan = self.config.get('max_nan')
        #     assert isinstance(
        #         self.max_nan, int), f"max_nan must be an integer, not {type(self.max_nan)}"

        # if 'drop_const_columns' in self.config:
        #     self.drop_const_columns = self.config.get('drop_const_columns')
        #     assert isinstance(
        #         self.drop_const_columns, bool), f"drop_const_columns must be a boolean, not {type(self.drop_const_columns)}"
        # else:
        #     self.drop_const_columns = False

        self.categorical_columns = []
        self.numeric_columns = []

        # self.columns_types = {}

        self.date_min = None
        self.date_max = None
        self.freq = None

        # self.is_fetched = False
        # self.is_saved = False

        # Check if the data is saved at the location "data_dir/data.feather"

        # if os.path.isfile(self.data_dir / f'data.feather') and load:
        #     self.data = pd.read_feather(
        #         self.data_dir / f'data.feather')
        #     self.is_saved = True
        #     self.is_fetched = True

        # Décorateur pour fetch_data
        # used to fetch data only if it's not already saved (then load it)
        # then, save it if needed, format the data and set the is_fetched flag to True
        self.fetch_data = self.fetch_decorator(self.fetch_data_function)

    def features_augmentation(self, data: pd.DataFrame, features_names: Optional[List[str]] = None, shift: Union[int, List[int]] = [], rolling_window: Union[int, List[int]] = 0) -> pd.DataFrame:

        self.logger.info("Augmentation des features...")

        if features_names is None:
            features_names = data.columns.tolist()

        assert isinstance(
            features_names, List), f"features_names must be of type list, not {type(features_names)}"

        if isinstance(rolling_window, int):
            rolling_window = [rolling_window]

        if isinstance(shift, int):
            shift = [shift]

        for feature_name in features_names:

            # self.logger.info(f"Augmentation de la feature {feature_name}...")
            # On ajoute les valeurs passées
            for dec in shift:
                # self.logger.info(f"  - Ajout de la feature {feature_name}_J-{dec}")
                data[f"{feature_name}%%J{-1*dec}"] = data[f"{feature_name}"].shift(dec)

            # On ajoute la moyenne glissante, standard deviation sur FENETRE_GLISSANTE jours
            for rw in rolling_window:
                # self.logger.info(f"  - Ajout de la moyenne et de l'écart-type de la feature {feature_name} sur {rw} jours")
                if rw > 0 and feature_name in self.numeric_columns:
                    data[f"{feature_name}%%mean_{rw}J"] = data[f"{feature_name}"].rolling(
                        window=rw, closed="left").mean()
                    data[f"{feature_name}%%std_{rw}J"] = data[f"{feature_name}"].rolling(
                        window=rw, closed="left").std()

                    # TODO: Ajouter le mode, la valeur majoritaire, la fréquence relatives. pour les données catégorielles

        return data

    def fetch_decorator(self, func):
        def wrapper(start_date, stop_date=dt.datetime.today(), *args, **kwargs):
            
            if isinstance(start_date, str):
                start_date = dt.datetime.strptime(start_date, "%d-%m-%Y")
            if isinstance(stop_date, str):
                stop_date = dt.datetime.strptime(stop_date, "%d-%m-%Y")
            assert start_date <= stop_date, "start date must be before stop date"
            
            filename = kwargs.pop('filename', 'data.feather')

            features_dir = kwargs.pop('features_dir', f'{os.getcwd()}/data/features')
            if isinstance(features_dir, str):
                features_dir = pathlib.Path(features_dir)
            feature_dir = features_dir / f"{self.name}"
            kwargs['feature_dir'] = feature_dir

            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)

            
            if os.path.isfile(feature_dir / filename):
                self.logger.info(f"Data already fetched for {self.name}")
                # TODO: check for updates
                data = pd.read_feather(feature_dir / filename)
                if data is None or len(data)==0:
                    self.logger.warning("No data found")

                # Check if update is needed
                if data.index.max() < stop_date:
                    self.logger.warning("Updating data")
                    data = pd.concat(data, func(self, data.index.max(), stop_date, *args, **kwargs))
            else:
                self.logger.info(f"Fetching data for {self.name}...")

                # Fetch the data
                data:pd.DataFrame = func(self, start_date=start_date, stop_date=stop_date, *args, **kwargs)
                self.save_dataframe(data, feature_dir, filename=filename)

            # # Identifier les colonnes catégorielles
            self.categorical_columns = data.select_dtypes(
                include=['object', 'category']).columns
            self.numeric_columns = data.select_dtypes(
                include=['number']).columns

            # Définir les dates min et max (Période ou toutes les données sont disponibles)
            self.set_availability(data)

            # self.is_fetched = True

        return wrapper

    @abc.abstractmethod
    def fetch_data_function(self, *args, **kwargs) -> None:
        pass

    def set_availability(self, data:pd.DataFrame) -> None:
            self.date_min = data.dropna().index.min()
            self.date_max = data.dropna().index.max()
            self.freq = pd.infer_freq(data.index)
    
    def get_availability(self):
        return self.date_min, self.date_max, self.freq
    
    def get_data(self,
        from_date: Optional[Union[str, dt.datetime]] = None,
        to_date: Optional[Union[str, dt.datetime]] = None,
        features_names: Optional[List[str]] = None,
        freq: Optional[str] = None,
        shift: Optional[Union[int, List[int]]] = None,
        rolling_window: Optional[Union[int, List[int]]] = None,
        drop_constant_thr = 1.0,
        path: Optional[Union[str, pathlib.Path]] = None,
        filename: Optional[str] = None) -> pd.DataFrame:

        date_min, date_max, initial_freq = self.get_availability()
        if date_min is None or date_max is None or initial_freq is None:
            self.logger.error('It looks like you didn\'t call self.fetch_data() before, you need to call self.fetch_data() even if the data has been fetched during a previous session.')
            return None

        # if not self.is_fetched:
        #     raise Exception("Data not fetched, please call fetch_data() before.")

        if from_date is None:
            from_date = date_min
        assert isinstance(from_date, dt.datetime) or isinstance(
            from_date, str), f"from_date must be of type datetime.datetime or a string, not {type(from_date)}"
        if isinstance(from_date, str):
            try:
                from_date = dt.datetime.strptime(from_date, '%d-%m-%Y')
            except ValueError:
                raise ValueError(
                    f"from_date must be of format %d-%m-%Y, not {from_date}")

        if to_date is None:
            to_date = date_max
        assert isinstance(to_date, dt.datetime) or isinstance(
            to_date, str), f"to_date must be of type datetime.datetime or a string, not {type(to_date)}"
        if isinstance(to_date, str):
            try:
                to_date = dt.datetime.strptime(to_date, '%d-%m-%Y')
            except ValueError:
                raise ValueError(
                    f"to_date must be of format %d-%m-%Y, not {to_date}")

        if freq is None:
            freq = initial_freq

        self.logger.info(f"Getting data for {self.name} from {from_date} to {to_date}, at a {freq} frequency")
        if path is None:
            path = './data/features/' + self.name

        if filename is None:
            filename = 'data.feather'

        try:
            data = self.load(path=path, filename=filename)
        except FileNotFoundError:
            self.logger.error(
                f"Data not found for {self.name}, data needs to be fetched first")
            return None
            # self.fetch_data(start_date=from_date, stop_date=to_date)
            # data = self.load(path=path, filename=filename)

        # Check if the minimum date is after from_date and the maximum date is before to_date
        if date_min > from_date or date_max < to_date:
            raise ValueError(
                f"{self.name} data only available between {date_min} and {date_max} while you requested from {from_date} to {to_date}")

        data = self.drop_constant_columns(data=data, threshold=drop_constant_thr, exclude_categories=True)

        if shift or rolling_window:
            if date_min > from_date - dt.timedelta(days=(max(*shift, *rolling_window, 0))):
                raise ValueError(f"{self.name} data only available from {date_min}, while shift and rolling_window require data from {from_date - dt.timedelta(days=(max(*shift, *rolling_window, 0)))}")
            
            # Si %% est présent dans un des noms des colonnes, alors on préviens l'utilisateur que l'augmentation des features a déjà été effectuée sur ces données
            if [col for col in data.columns if '%%' in col] != []:
                self.logger.warning("Features already augmented, skipping features augmentation")
            else:
                data = self.features_augmentation(data, features_names, shift=shift, rolling_window=rolling_window)
        # data.dropna(inplace=True)

        # On agrège le dataframe à la fréquence souhaitée
        # Pour gagner du temps, on n'effectue pas d'agrégation si la fréquence est la même que celle des données
        # avec 1M == 1ME == M == ME
        # if freq and freq not in {initial_freq, f'1{initial_freq}', f'1{initial_freq}E', f'{initial_freq}E'}:
        #     data = self.aggregate_data_by_dtype(agg_dict={'number': ['mean', 'std', 'min', 'max'], 'category': [relative_frequency]}, freq=freq, data=data)

        if features_names is None:
            features_names = data.columns.tolist()
        assert isinstance(
            features_names, list), f"features_names must be of type list, not {type(features_names)}"
        
        # data = data.iloc[max(0, *rolling_window, *shift):]

        # # Supprimer les lignes contenant des NaN
        # data = data.dropna()
        # print(from_date, data.index.min())
        # print(to_date, data.index.max())
        # print(data)
        data = data.loc[from_date:to_date]
        data = data[features_names]
        # print(data)
        nan_rows = data[data.isna().any(axis=1)].index
        if len(nan_rows):
            self.logger.warning("Plage des indices des lignes contenant des NaN: {}".format(nan_rows))
            self.logger.warning("Colonnes contenant des NaN pour ces indices: {}".format(data.columns[data.isna().any()].tolist()))

        return data

    def save_dataframe(self, data:pd.DataFrame, path: Union[str, pathlib.Path], filename: Optional[str] = 'data.feather') -> None:

        self.logger.info(f"Saving data for {self.name}...")

        assert isinstance(path, str) or isinstance(
            path, pathlib.Path), f"path must be of type str or pathlib.Path, not {type(path)}"
        if isinstance(path, str):
            path = pathlib.Path(path)

        assert isinstance(
            filename, str), f"filename must be of type str, not {type(filename)}"
        if not filename.endswith('.feather'):
            filename += '.feather'

        data.to_feather(path / filename)


    # def get_features_names(self) -> List[str]:
    #     """
    #     Retourne les noms des features.

    #     Parameters:
    #     - None

    #     Returns:
    #     - Les noms des features.
    #     """

    #     return self.data.columns.tolist()

    # def columns_by_type(self) -> dict:
    #     """
    #     Retourne un dictionnaire des colonnes du DataFrame classées par type de données.

    #     Parameters:
    #     - df: Le DataFrame à analyser.

    #     Returns:
    #     - dict: Un dictionnaire où les clés sont les types de données et les valeurs sont les listes de colonnes correspondantes.
    #     """

    #     # Parcourir tous les types de données dans le DataFrame
    #     for dtype in self.data.dtypes.unique():
    #         # Trouver les colonnes qui correspondent à ce type de données
    #         columns = self.data.columns[self.data.dtypes == dtype].tolist()

    #         # Ajouter au dictionnaire, la clé est le nom du type, la valeur est la liste des colonnes
    #         self.columns_types[str(dtype)] = columns

    #     return self.columns_types

    def aggregate_data_by_dtype(self, agg_dict, data: pd.DataFrame, freq='1D', flatten=True) -> pd.DataFrame:

        self.logger.info("Aggregating data by data type...")

        # Vérifier que le DataFrame a un index de type DateTime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "Le DataFrame doit avoir un index de type DateTime.")

        # Créer un dictionnaire pour stocker les agrégations par colonne
        column_agg_dict = {}

        # Pour chaque type dans agg_dict, on applique les fonctions d'agrégation aux colonnes correspondantes
        for dtype, funcs in agg_dict.items():
            matching_columns = data.select_dtypes(include=[dtype]).columns
            for col in matching_columns:
                column_agg_dict[col] = funcs

        # Fonction d'application unifiée pour gérer les différentes fonctions d'agrégation
        def unified_apply_func(series, funcs):
            """
            Applique une liste de fonctions à une série et retourne une Series ou un DataFrame.
            """
            results = {}
            for func in funcs:
                # self.logger.info(f"Applying {func} to {series.name}...")
                if callable(func):
                    # print("callable: ", func)
                    result = func(series)
                    # print(result)
                    if isinstance(result, pd.Series):
                        # print("isinstance: series")
                        # Renommer les résultats de la série pour les inclure dans le MultiIndex
                        results.update(result.rename(
                            lambda x: f"{func.__name__}~{x}"))
                    else:
                        results[func.__name__] = result
                else:
                    # print("not callable: ", func)
                    result = getattr(series, func)()
                    # print(result)
                    results[func] = result
            return pd.Series(results)

        # Appliquer les agrégations à chaque groupe
        def apply_aggregations(group):
            aggregated = {}
            for col, funcs in column_agg_dict.items():
                # print(group[col])
                # self.logger.info(f"Aggregating column {col}...")
                aggregated[col] = unified_apply_func(group[col], funcs)

                concat = pd.concat(aggregated, axis=0)
            return concat

        grouper = pd.Grouper(freq=freq)
        grouped_data = data.groupby(grouper)

        # Appliquer les agrégations
        aggregated_df = grouped_data.apply(apply_aggregations)

        if not isinstance(aggregated_df, pd.DataFrame):
            aggregated_df = aggregated_df.to_frame().unstack(
                level=[1, 2]).droplevel(0, axis=1)

        if flatten:
            aggregated_df.columns = [
                '~~'.join(col).strip() for col in aggregated_df.columns.values]

        return aggregated_df

    def plot(self, from_date: Optional[Union[str, dt.datetime]] = None,
             to_date: Optional[Union[str, dt.datetime]] = None,
             features_names: Optional[List[str]] = None,
             freq: str = '1D',
             max_subplots: int = 4,
             data: pd.DataFrame = None) -> pd.DataFrame:

        if data is None:
            data = self.get_data(
                from_date=from_date, to_date=to_date, features_names=features_names, freq=freq)

        # # Identifier les colonnes catégorielles
        # categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        # categorical_data = data[categorical_columns]
        # data = data.drop(columns=categorical_columns)
        # numeric_columns = data.select_dtypes(include=['number']).columns

        # elif not categorical_columns.empty:
        #     data = data.join(self.aggregate_data_by_dtype(agg_dict={'category': [relative_frequency]}, freq=freq, data=categorical_data))
        # print(data)
        # data = self.groupby(from_date=from_date, to_date=to_date, features_names=features_names, freq=freq)
        # # data = self.get_data(from_date=from_date, to_date=to_date, features_names=features_names)

        # for i in range(0, len(data.columns), min(4, len(data.columns))):
        #     data.iloc[:, i:i+5].plot()
        #     plt.show()

        num_rows = np.ceil(np.sqrt(max_subplots)).astype(int)
        num_cols = np.ceil(max_subplots / num_rows).astype(int)

        # Calcul du nombre total de figures nécessaires
        num_vars = len(data.columns)

        num_figures = math.ceil(num_vars / max_subplots)

        # if (freq != pd.infer_freq(data.index)) and (freq is not None) and (freq != '1' + pd.infer_freq(data.index)) and (freq != '1' + pd.infer_freq(data.index)+'E') and (freq != pd.infer_freq(data.index)+'E'):
        #     agg_data = self.aggregate_data_by_dtype(agg_dict={'number': ['mean', 'std', 'min', 'max'], 'category': [relative_frequency]}, data=data, freq=freq)
        # else:
        #     agg_data = data

        for fig_num in range(num_figures):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()

            start_idx = fig_num * max_subplots
            end_idx = min(start_idx + max_subplots, num_vars)

            for i, variable_name in enumerate(data.columns[start_idx:end_idx]):

                for column in data.columns.to_list():

                    if '~~' in column:
                        column_name, agg_func = column.split('~~', 1)
                    else:
                        column_name = column
                        agg_func = column

                    if variable_name == column_name:
                        col = data[column]
                        # col = data.loc[:, [col for col in data.columns if col.startswith(f'{column}__')]]
                        axes[i].plot(col.index, col, label=agg_func)
                        # col.plot(kind='line', ax=axes[i])
                        axes[i].set_title(
                            f'{variable_name} over time ({freq})')
                        axes[i].set_xlabel('Date')
                        axes[i].set_ylabel('Value')
                        axes[i].legend()

            # Supprimer les axes inutilisés
            for j in range(end_idx - start_idx, len(axes)):
                fig.delaxes(axes[j])

            # Ajuster l'affichage
            plt.tight_layout()
            plt.show()

    def drop_constant_columns(self, data:pd.DataFrame, threshold: float = 1.0, exclude_categories: bool = True) -> pd.DataFrame:

        if exclude_categories:
            # Sélectionner les colonnes qui ne sont pas du type 'category'
            selected_columns = data.select_dtypes(exclude=['category']).columns

        else:
            selected_columns = data.columns

        # Liste pour stocker les colonnes à supprimer
        cols_to_drop = []

        for col in selected_columns:
            # Calculer la proportion de la valeur la plus fréquente dans la colonne
            value_frequencies = data[col].value_counts(normalize=True)

            most_frequent_value = value_frequencies.idxmax()
            most_frequent_value_ratio = value_frequencies.max()

            # Vérifier si cette proportion dépasse le seuil (threshold)
            if most_frequent_value_ratio >= threshold:
                cols_to_drop.append(col)
                self.logger.info(
                    f"Column '{col}' is constant at {most_frequent_value} for {most_frequent_value_ratio:.2%} of the rows.")

        # Supprimer les colonnes constantes
        df_cleaned = data.drop(columns=cols_to_drop)

        # print(df_cleaned.columns)

        return df_cleaned

    def load(self, path: Optional[Union[str, pathlib.Path]], filename: Optional[str]) -> pd.DataFrame:

        assert isinstance(path, str) or isinstance(
            path, pathlib.Path), f"path must be of type str or pathlib.Path, not {type(path)}"
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"'{path}' is not a directory")

        assert isinstance(
            filename, str), f"filename must be of type str, not {type(filename)}"
        if not filename.endswith('.feather'):
            filename += '.feather'

        if not os.path.isfile(path / filename):
            raise FileNotFoundError(
                f"File '{filename}' not found in \'{path}\'")

        return pd.read_feather(path / filename)