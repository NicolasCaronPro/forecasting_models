import math
import pathlib
import pandas as pd
import more_itertools
import datetime as dt
import os
import sys
from src.configs.config import Config
from typing import List, Optional, Union
from copy import deepcopy
import matplotlib.pyplot as plt
import abc
from sklearn.compose import make_column_transformer, make_column_selector

class BaseFeature(object):
    """
    Classe abstraite représentant une feature.

    Attributes:
    - config: Configuration.
    - parent: Parent object.
    - name: Nom de la feature.
    - logger: Logger.
    - data_dir: Répertoire des données.
    - start: Date de début.
    - shift: Décalage (optionel).
    - rolling_window: Fenêtre glissante (optionel).
    - step_unit: Unité de pas.
    - step_value: Valeur de pas.
    - stop: Date de fin.
    - step: Pas. {self.step_unit: self.step_value}
    - index: Index.
    - data: Données.
    - max_nan: Nombre maximum de valeurs manquantes.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        """
        Initialisation de la classe.

        Parameters:
        - config: Configuration.
        - parent: Parent object.
        """

        # TODO: Est ce bien utile de passer le parent en paramètre ?

        self.name = self.__class__.__name__
        print(f"Initialisation de la classe {self.name}")

        assert config is None or isinstance(
            config, Config), f"config must be of type Config, not {type(config)}"
        assert parent is None or isinstance(
            parent, BaseFeature), f"parent must be of type BaseFeature, not {type(parent)}"
        assert config is not None or parent is not None, "Either config or parent must be provided"

        self.config = Config()
        if parent is not None:
            self.config.update(parent.config.to_dict())
        if config is not None:
            self.config.update(config.to_dict())

        self.parent = parent

        assert 'logger' in self.config, "logger must be provided in config"
        self.logger = self.config.get('logger')
        assert isinstance(self.logger, type(sys.modules['logging'].getLogger(
        ))), f"logger must be of type logging.Logger, not {type(self.logger)}"

        assert 'root_dir' in self.config, "root_dir must be provided in config"
        self.data_dir = self.config.get("root_dir")
        assert isinstance(self.data_dir, str) or isinstance(self.data_dir, type(
            pathlib.Path())), f"root_dir must be a string or a pathlib Path, not {type(self.data_dir)}"

        if isinstance(self.data_dir, str):
            self.data_dir = pathlib.Path(self.data_dir)

        assert self.data_dir.is_dir(
        ), f"root_dir must be a directory, not {self.data_dir}"

        self.data_dir = pathlib.Path(self.config.get("root_dir")) / 'data'

        if 'features' in self.__class__.__name__.lower():
            self.data_dir = self.data_dir / 'features'
        self.data_dir = self.data_dir / self.__class__.__name__.lower()
        self.data_dir.mkdir(exist_ok=True, parents=True)

        assert 'start' in self.config, "start must be provided in config"
        self.start = self.config.get('start')
        assert isinstance(self.start, dt.datetime) or isinstance(
            self.start, str), f"start must be of type datetime.datetime or a string, not {type(self.start)}"

        if isinstance(self.start, str):
            try:
                self.start = dt.datetime.strptime(self.start, '%d-%m-%Y')
            except ValueError:
                raise ValueError(
                    f"start must be of format %d-%m-%Y, not {self.start}")

        if 'shift' in self.config:
            self.shift = self.config.get('shift')
            assert isinstance(
                self.shift, int), f"shift must be of type int, not {type(self.shift)}"

        if 'rolling_window' in self.config:
            self.rolling_window = self.config.get('rolling_window')
            assert isinstance(
                self.rolling_window, int), f"rolling_window must be of type int, not {type(self.rolling_window)}"

        assert 'step_unit' in self.config, "step_unit must be provided in config"
        self.step_unit = self.config.get('step_unit')
        assert self.step_unit in ['days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours',
                                  'weeks'], f"step_unit must be one of ['days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours', 'weeks'], not {self.step_unit}"

        assert 'step_value' in self.config, "step_value must be provided in config"
        self.step_value = self.config.get('step_value')
        assert isinstance(
            self.step_value, int), f"step_value must be an integer, not {type(self.step_value)}"

        self.start -= dt.timedelta(**{self.step_unit: self.step_value *
                                   max(max(self.shift, self.rolling_window), 0)})

        assert 'stop' in self.config, "stop must be provided in config"
        self.stop = self.config.get('stop')
        assert isinstance(self.stop, dt.datetime) or isinstance(
            self.stop, str), f"stop must be of type datetime.datetime or a string, not {type(self.stop)}"

        if isinstance(self.stop, str):
            try:
                self.stop = dt.datetime.strptime(self.stop, '%d-%m-%Y')
            except ValueError:
                raise ValueError(
                    f"stop must be of format %d-%m-%Y, not {self.stop}")

        self.step = {self.step_unit: self.step_value}
        self.index: pd.Index = pd.Index(data=more_itertools.numeric_range(
            self.start, self.stop + dt.timedelta(**self.step), dt.timedelta(**self.step)), name='date')
        self.data: pd.DataFrame = pd.DataFrame(index=self.index)

        if 'max_nan' in self.config:
            self.max_nan = self.config.get('max_nan')
            assert isinstance(
                self.max_nan, int), f"max_nan must be an integer, not {type(self.max_nan)}"
            
        self.categorical_columns = []
        self.numeric_columns = []

        self.date_min = None
        self.date_max = None


    def features_augmentation(self, features_names: Optional[List[str]] = None) -> None:

        if features_names is None:
            features_names = self.get_features_names()

        assert isinstance(
            features_names, List), f"features_names must be of type list, not {type(features_names)}"

        for feature_name in features_names:
            # On ajoute les valeurs passées
            for dec in range(1, self.shift+1):
                self.data[f"{feature_name}_J-{dec}"] = self.data[f"{feature_name}"].shift(
                    dec)

            # On ajoute la moyenne glissante, standard deviation sur FENETRE_GLISSANTE jours
            if self.rolling_window > 0 and feature_name in self.numeric_columns:
                self.data[f"{feature_name}_mean"] = self.data[f"{feature_name}"].rolling(
                    window=self.rolling_window, closed="left").mean()
                self.data[f"{feature_name}_std"] = self.data[f"{feature_name}"].rolling(
                    window=self.rolling_window, closed="left").std()
                
        # Identifier les colonnes catégorielles
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        self.numeric_columns = self.data.select_dtypes(include=['number']).columns

    @abc.abstractmethod
    def fetch_data(self) -> None:
        """
        Récupère les données.

        Parameters:
        - None
        """

        # Identifier les colonnes catégorielles
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        self.numeric_columns = self.data.select_dtypes(include=['number']).columns

        # Définir les dates min et max (Période ou toutes les données sont disponibles)
        self.date_min = self.data.dropna().index.min()
        self.date_max = self.data.dropna().index.max()

    @abc.abstractmethod
    def get_data(self, from_date: Optional[Union[str, dt.datetime]] = None, to_date: Optional[Union[str, dt.datetime]] = None, features_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retourne les données.

        Parameters:
        - None

        Returns:
        - data: Les données.
        """

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

        if features_names is None:
            features_names = self.get_features_names()
        assert isinstance(
            features_names, list), f"features_names must be of type list, not {type(features_names)}"
        
        # print(self.data.index)

        return self.data.loc[(self.data.index >= from_date) & (self.data.index <= to_date), features_names]

    @abc.abstractmethod
    def save_data(self, path: Optional[Union[str, pathlib.Path]] = None, filename: Optional[str] = None) -> None:
        """
        Sauvegarde les données.

        Parameters:
        - None
        """

        if path is None:
            path = self.data_dir
        assert isinstance(path, str) or isinstance(
            path, pathlib.Path), f"path must be of type str or pathlib.Path, not {type(path)}"
        if isinstance(path, str):
            path = pathlib.Path(path)

        if filename is None:
            filename = f'data_{self.name}.feather'
        assert isinstance(
            filename, str), f"filename must be of type str, not {type(filename)}"
        if not filename.endswith('.feather'):
            filename += '.feather'

        self.data.to_feather(path / filename)

    @abc.abstractmethod
    def get_features_names(self) -> List[str]:
        """
        Retourne les noms des features.

        Parameters:
        - None

        Returns:
        - Les noms des features.
        """

        return self.data.columns.tolist()

    @abc.abstractmethod
    def groupby(self, from_date: Optional[Union[str, dt.datetime]] = None, to_date: Optional[Union[str, dt.datetime]] = None, features_names: Optional[List[str]] = None, freq: str = '1D') -> pd.DataFrame:
        """
        Groupe les données.

        Parameters:
        - by: L'échelle à laquelle regrouper les données. (ex: 'day', 'week', 'month', 'year')
        - features_names: Les features à grouper.

        Returns:
        - data: Les données groupées.
        """

        data = self.get_data(from_date=from_date,
                             to_date=to_date, features_names=features_names)
        
        # print(data.columns)

        # Grouper les données par la fréquence spécifiée
        grouper = pd.Grouper(freq=freq, level='date')

        numeric_grouped_data = data.groupby(grouper)[self.numeric_columns].mean()

        categorical_grouped_data = (data.groupby(grouper)[self.categorical_columns]
                                       .apply(lambda x: x.apply(lambda y: y.value_counts(normalize=True) * 100))
                                       .fillna(0))

        # Réorganiser les données catégorielles
        categorical_grouped_data = categorical_grouped_data.unstack(fill_value=0)
        categorical_grouped_data.columns = [f'{col[0]}__{col[1]}' for col in categorical_grouped_data.columns]

        # categorical_grouped_data = (data.groupby(grouper)[categorical_columns].value_counts(normalize=True)*100)
        # print(categorical_grouped_data)
        # categorical_grouped_data = categorical_grouped_data.unstack(fill_value=0).rename(columns=lambda x: f'{x}_percentage')
        # print(categorical_grouped_data)

        aggregated_data = pd.concat([numeric_grouped_data, categorical_grouped_data], axis=1)
        
        # print(aggregated_data)
    

        return aggregated_data

    # def plot_numeric_over_time(self, df, column, resample_freq='1D'):
    #     df[column].resample(resample_freq).mean().plot(kind='line', figsize=(10, 6))
    #     plt.title(f'{column} over time ({resample_freq})')
    #     plt.xlabel('Date')
    #     plt.ylabel(column)
    #     plt.show()

    # # Exemple de graphique pour une colonne catégorique
    # def plot_categorical_over_time(self, df, column, resample_freq='1D'):
    #     category_counts = df[column].resample(resample_freq).value_counts().unstack().fillna(0)
    #     category_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    #     plt.title(f'{column} over time ({resample_freq})')
    #     plt.xlabel('Date')
    #     plt.ylabel('Count')
    #     plt.legend(title=column)
    #     plt.show()

    @abc.abstractmethod
    def plot(self, from_date: Optional[Union[str, dt.datetime]] = None, to_date: Optional[Union[str, dt.datetime]] = None, features_names: Optional[List[str]] = None, freq: str = '1D') -> pd.DataFrame:
        """
        Affiche les données.

        Parameters:
        - features: Les features à afficher.
        """

        grouped_data = self.groupby(from_date, to_date, features_names, freq=freq)
        # # data = self.get_data(from_date=from_date, to_date=to_date, features_names=features_names)

        # for i in range(0, len(data.columns), min(4, len(data.columns))):
        #     data.iloc[:, i:i+5].plot()
        #     plt.show()

        max_subplots = 4
        num_cols = 2
        num_rows = 2

        # Calcul du nombre total de figures nécessaires
        num_vars = len(self.data.columns)

        # print(self.data.columns)
        num_figures = math.ceil(num_vars / max_subplots)

        # print(num_figures)

        # # Identifier les colonnes catégorielles
        # categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        # numeric_columns = data.select_dtypes(include=['number']).columns

        # print(data)

        for fig_num in range(num_figures):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            axes = axes.flatten()

            start_idx = fig_num * max_subplots
            end_idx = min(start_idx + max_subplots, num_vars)

            # print(start_idx, end_idx)

            for i, column in enumerate(self.data.columns[start_idx:end_idx]):
                # print(column)
                if column in self.categorical_columns:
                    col = grouped_data.loc[:, [col for col in grouped_data.columns if col.startswith(f'{column}__')]]
                    print(col)
                    col.plot(kind='bar', stacked=True, ax=axes[i])
                    axes[i].set_title(f'{column} over time ({freq})')
                    axes[i].set_xlabel('Date')
                    axes[i].set_ylabel('Count')
                    axes[i].legend(title=column)

                elif column in self.numeric_columns:
                    col = grouped_data[column]
                    # print(col)
                    col.plot(kind='line', ax=axes[i])
                    axes[i].set_title(f'{column} over time ({freq})')
                    axes[i].set_xlabel('Date')
                    axes[i].set_ylabel('Count')
                    axes[i].legend(title=column)

            # Supprimer les axes inutilisés
            for j in range(end_idx - start_idx, len(axes)):
                fig.delaxes(axes[j])

            # Ajuster l'affichage
            plt.tight_layout()
            plt.show()

    @abc.abstractmethod
    def dtypes(self) -> pd.Series:
        """
        Retourne les types des données.

        Parameters:
        - None

        Returns:
        - Les types des données.
        """

        return self.data.dtypes

    @abc.abstractmethod
    def get_dtypes(self) -> List[str]:
        """
        Retourne les types des données.

        Parameters:
        - None

        Returns:
        - Les types des données.
        """

        return set(self.data.dtypes.tolist())

    @abc.abstractmethod
    def describe(self) -> pd.DataFrame:
        """
        Retourne les statistiques descriptives des données.

        Parameters:
        - None

        Returns:
        - Les statistiques descriptives des données.
        """

        return self.data.describe()

    @abc.abstractmethod
    def info(self) -> pd.DataFrame:
        """
        Retourne les informations des données.

        Parameters:
        - None

        Returns:
        - Les informations des données.
        """

        return self.data.info()

    @abc.abstractmethod
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Retourne les premières lignes des données.

        Parameters:
        - n: Le nombre de lignes à retourner.

        Returns:
        - Les premières lignes des données.
        """

        return self.data.head(n)

    @abc.abstractmethod
    def tail(self, n: int = 5) -> pd.DataFrame:
        """
        Retourne les dernières lignes des données.

        Parameters:
        - n: Le nombre de lignes à retourner.

        Returns:
        - Les dernières lignes des données.
        """

        return self.data.tail(n)

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Retourne une représentation de la classe.

        Parameters:
        - None

        Returns:
        - La représentation de la classe.
        """

        return f"{self.name}({self.config})"

    @abc.abstractmethod
    def __repr__(self) -> str:
        """
        Retourne une représentation de la classe.

        Parameters:
        - None

        Returns:
        - La représentation de la classe.
        """

        return self.__str__()

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Retourne la taille des données.

        Parameters:
        - None

        Returns:
        - La taille des données.
        """

        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, key: str) -> pd.Series:
        """
        Retourne une colonne des données.

        Parameters:
        - key: La colonne à retourner.

        Returns:
        - La colonne des données.
        """

        return self.data[key]

    # @abc.abstractmethod
    # def fit(self) -> None:
    #     """
    #     Encode les données.

    #     Parameters:
    #     - None
    #     """
        
    #     self.ct.fit(self.data)

    # @abc.abstractmethod
    # def transform(self) -> None:
    #     """
    #     Encode les données.

    #     Parameters:
    #     - None
    #     """
        
    #     self.ct.transform(self.data)

    # @abc.abstractmethod
    # def fit_transform(self) -> None:
    #     """
    #     Encode les données.

    #     Parameters:
    #     - None
    #     """
        
    #     self.ct.fit_transform(self.data)