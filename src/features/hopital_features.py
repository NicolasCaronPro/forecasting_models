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

    def __init__(self, config: Optional[Config] = None, parent: Optional[BaseFeature] = None, include_emmergency_arrivals = True, include_nb_hospit = True, include_hnfc_moving = True) -> None:
        super().__init__(config, parent)
        self.include_emmergency_arrivals = include_emmergency_arrivals
        self.include_nb_hospit = include_nb_hospit
        self.include_hnfc_moving = include_hnfc_moving
        assert 'etablissement' in self.config, "etablissement must be provided in config"
        self.etablissement = self.config.get('etablissement')
        assert isinstance(self.etablissement,
                          str), "etablissement must be a string"

    def add_target(self):
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
            self.data = pd.read_feather(target_file_path)
        else:
            self.logger.info(
                f"  - Chargement des données de {self.etablissement} depuis le fichier Excel")
            self.data = pd.read_excel(self.data_dir / file_name, sheet_name=1)

            if "annee" in self.data:
                self.data.drop(axis=1, columns="annee", inplace=True)

            self.data.rename(columns={
                             "Total": f"{self.name} Total_{self.etablissement}", "date_entree": "date"}, inplace=True)

            if self.data["date"].dtype != "datetime64[ns]":
                self.data["date"] = pd.to_datetime(self.data["date"])

            self.data.sort_values(by="date", inplace=True)
            self.data.set_index('date', inplace=True)

    def include_HNFC_moving(self):
        self.logger.info("Intégration du déménagement de l'HNFC")
        start = dt.datetime(2017, 2, 28)
        end = dt.datetime(2018, 1, 1)

        self.data["HNFC_moving"] = np.where(self.data.index < start, 'before', np.where(self.data.index >= end, 'after', 'during'))
        self.data["HNFC_moving"] = self.data["HNFC_moving"].astype("category")
        # self.data["2_HNFC_moving"] = self.data['HNFC_moving'].copy().shift(1).astype("category")

    def include_nb_hospitalized(self):
        hospitalized = pd.read_excel(self.data_dir / "nb_hospit/RPU_vers_hospit.xlsx")
        self.data = self.data.join(hospitalized.set_index("date_entree")["nb_vers_hospit"])

    def fetch_data_function(self) -> None:
        """
        Fetches the data by adding the target and calling the parent's fetch_data method.
        """

        if self.include_emmergency_arrivals:
            self.add_target()
        
        if self.include_hnfc_moving:
            self.include_HNFC_moving()

        if self.include_nb_hospit:
            self.include_nb_hospitalized()

