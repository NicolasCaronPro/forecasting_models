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

    def __init__(self, config: Optional[Config] = None, parent: Optional[BaseFeature] = None, include_emmergency_arrivals=True, include_nb_hospit=True, include_hnfc_moving=True, load=False) -> None:
        super().__init__(config, parent, load)
        self.include_emmergency_arrivals = include_emmergency_arrivals
        self.include_nb_hospit = include_nb_hospit
        self.include_hnfc_moving = include_hnfc_moving
        assert 'etablissement' in self.config, "etablissement must be provided in config"
        self.etablissement = self.config.get('etablissement')
        assert isinstance(self.etablissement,
                          str), "etablissement must be a string"

    def include_nb_emmergencies(self, initial_shift: int = 0) -> None:
        """
        Adds the target to the features.
        """
        self.logger.info("Intégration de la target")

        file_name = f"Export complet {self.etablissement}.xlsx"
        target_file_path = self.data_dir / \
            f"{self.etablissement}_volumes.feather"

        if target_file_path.exists():
            self.logger.info(
                f"  - Chargement des données de {self.etablissement} depuis le fichier")
            data = pd.read_feather(target_file_path)
        else:
            self.logger.info(
                f"  - Chargement des données de {self.etablissement} depuis le fichier Excel")
            data = pd.read_excel(self.data_dir / file_name, sheet_name=1)

            if "annee" in data:
                data.drop(axis=1, columns="annee", inplace=True)

            data.rename(columns={
                "Total": f"Total_{self.etablissement}", "date_entree": "date"}, inplace=True)

            if data["date"].dtype != "datetime64[ns]":
                data["date"] = pd.to_datetime(data["date"])

            data.sort_values(by="date", inplace=True)
            data.set_index('date', inplace=True)
        # print(data)
        data[f"Total_{self.etablissement}"] = data[f"Total_{self.etablissement}"].shift(initial_shift)
        data.dropna(subset=[f"Total_{self.etablissement}"], inplace=True)
        # print(data)
        self.data = self.data.join(data)
        # print(self.data)

    def include_HNFC_moving(self):
        self.logger.info("Intégration du déménagement de l'HNFC")
        start = dt.datetime(2017, 2, 28)
        end = dt.datetime(2018, 1, 1)

        self.data["HNFC_moving"] = np.where(self.data.index < start, 'before', np.where(
            self.data.index >= end, 'after', 'during'))
        self.data["HNFC_moving"] = self.data["HNFC_moving"].astype("category")

    def include_nb_hospitalized(self, initial_shift: int = 0):
        hospitalized = pd.read_excel(
            self.data_dir / "nb_hospit/RPU_vers_hospit.xlsx")
        hospitalized['date_entree'] = pd.to_datetime(hospitalized['date_entree'], unit='D', origin='1899-12-30')

        hospitalized.rename(columns={"date_entree": "date"}, inplace=True)
        hospitalized.set_index("date", inplace=True)
        hospitalized.rename(columns={"Total": "nb_vers_hospit"}, inplace=True)
        # print(hospitalized)
        hospitalized['nb_vers_hospit'] = hospitalized['nb_vers_hospit'].shift(initial_shift)
        self.data = self.data.join(hospitalized)
        # print(self.data)
        # self.data["nb_vers_hospit"] = self.data["nb_vers_hospit"].shift(initial_shift)
        self.data.dropna(subset=["nb_vers_hospit"], inplace=True)

    def fetch_data_function(self, *args, **kwargs) -> None:
        """
        Fetches the data by adding the target and calling the parent's fetch_data method.
        """

        if self.include_emmergency_arrivals:
            self.include_nb_emmergencies(initial_shift=-1)

        if self.include_hnfc_moving:
            self.include_HNFC_moving()

        if self.include_nb_hospit:
            self.include_nb_hospitalized(initial_shift=-1)

        # print(self.data)
