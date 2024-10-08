"""
This module contains the `HopitalFeatures` class, which represents features related to a hospital.

The `HopitalFeatures` class inherits from the `BaseFeature` class and provides methods to add a target to the features and fetch the data.

Attributes:
    config (Config | None): The configuration object for the features. Defaults to None.
    parent (BaseFeature | None): The parent feature object. Defaults to None.

Methods:
    __init__(self, config: Optional[Config] = None, parent: Optional[BaseFeature] = None) -> None:
        Initializes a new instance of the `HopitalFeatures` class.
    
    add_target(self):
        Adds the target to the features.
    
    fetch_data(self) -> None:
        Fetches the data by adding the target and calling the parent's fetch_data method.
"""

from typing import Optional
import datetime as dt
import numpy as np
import pandas as pd
from src.features.base_features import BaseFeature, Config


class HopitalFeatures(BaseFeature):
    """
    Represents features related to a hospital.

    Attributes:
        etablissement (str): The name of the hospital.
    """

    def __init__(self, name:str = None, logger=None, include_emmergency_arrivals=True, include_nb_hospit=True, include_hnfc_moving=True) -> None:
        super().__init__(name, logger)
        # self.include_emmergency_arrivals = include_emmergency_arrivals
        # self.include_nb_hospit = include_nb_hospit
        # self.include_hnfc_moving = include_hnfc_moving

    def include_nb_emmergencies(self, from_date, to_date, etablissement, feature_dir, initial_shift: int = 0) -> None:
        """
        Adds the target to the features.
        """
        self.logger.info("Intégration de la target")

        file_name = f"Export complet {etablissement}.xlsx"
        target_file_path = feature_dir / \
            f"{etablissement}_volumes.feather"

        if target_file_path.exists():
            self.logger.info(
                f"  - Chargement des données de {etablissement} depuis le fichier")
            data = pd.read_feather(target_file_path)
        else:
            self.logger.info(
                f"  - Chargement des données de {etablissement} depuis le fichier Excel")
            data = pd.read_excel(feature_dir / file_name, sheet_name=1)

            if "annee" in data:
                data.drop(axis=1, columns="annee", inplace=True)

            data.rename(columns={
                "Total": f"Total_{etablissement}", "date_entree": "date"}, inplace=True)

            if data["date"].dtype != "datetime64[ns]":
                data["date"] = pd.to_datetime(data["date"])

            data.sort_values(by="date", inplace=True)
            data.set_index('date', inplace=True)
        # print(data)
        # data[f"Total_{etablissement}"] = data[f"Total_{etablissement}"].shift(initial_shift)
        data.dropna(subset=[f"Total_{etablissement}"], inplace=True)
        # print(data)
        # self.data = self.data.join(data)
        # print(self.data)
        return data

    def include_HNFC_moving(self, date_range:pd.DatetimeIndex):
        self.logger.info("Intégration du déménagement de l'HNFC")
        start = dt.datetime(2017, 2, 28)
        end = dt.datetime(2018, 1, 1)

        df = pd.DataFrame(index=date_range)
        # df.set_index('date', inplace=True)

        df["HNFC_moving"] = np.where(df.index < start, 'before', np.where(
            df.index >= end, 'after', 'during'))
        df["HNFC_moving"] = df["HNFC_moving"].astype("category")
        # print(df)
        return df

    def include_nb_hospitalized(self, from_date, to_date, feature_dir, initial_shift: int = 0):
        hospitalized = pd.read_excel(
            feature_dir / "nb_hospit/RPU_vers_hospit.xlsx")
        hospitalized['date_entree'] = pd.to_datetime(hospitalized['date_entree'], unit='D', origin='1899-12-30')

        hospitalized.rename(columns={"date_entree": "date"}, inplace=True)
        hospitalized.set_index("date", inplace=True)
        hospitalized.rename(columns={"Total": "nb_vers_hospit"}, inplace=True)
        # print(hospitalized)
        # hospitalized['nb_vers_hospit'] = hospitalized['nb_vers_hospit'].shift(initial_shift)
        # self.data = self.data.join(hospitalized)
        # print(self.data)
        # self.data["nb_vers_hospit"] = self.data["nb_vers_hospit"].shift(initial_shift)
        # self.data.dropna(subset=["nb_vers_hospit"], inplace=True)
        return hospitalized

    def fetch_data_function(self, *args, **kwargs) -> None:
        """
        Fetches the data by adding the target and calling the parent's fetch_data method.
        """

        assert 'etablissement' in kwargs, "etablissement must be provided in config"
        etablissement = kwargs.get('etablissement')
        assert isinstance(etablissement, str), "etablissement must be a string"

        feature_dir = kwargs.get("feature_dir")

        # Set starting date, default is 01/01/1970
        start_date = kwargs.get("start_date")
        start_date = dt.datetime.strptime(start_date, "%d-%M-%Y") if isinstance(start_date, str) else start_date

        # Set ending date, default is today's date
        stop_date = kwargs.get("stop_date")
        stop_date = dt.datetime.strptime(stop_date, "%d-%M-%Y") if isinstance(stop_date, str) else stop_date
        # print(f"Fetching data from {start_date} to {stop_date}")
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date")

        data = pd.DataFrame(index=date_range)
        # data.set_index("date", inplace=True)

        # if self.include_emmergency_arrivals:
        data = data.join(self.include_nb_emmergencies(start_date, stop_date, etablissement=etablissement, feature_dir=feature_dir, initial_shift=-1))
        # if self.include_hnfc_moving:
        data = data.join(self.include_HNFC_moving(date_range))
        # if self.include_nb_hospit:
        data = data.join(self.include_nb_hospitalized(start_date, stop_date, feature_dir, initial_shift=-1))
        return data
