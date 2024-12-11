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
import pandas as pd
from src.features.base_features import BaseFeature
from src.location.location import Location
from pathlib import Path


class HospitalFeatures(BaseFeature):
    """
    Represents features related to a hospital.

    Attributes:
        etablissement (str): The name of the hospital.
    """

    def __init__(self, name: str = None, logger=None,
                 include_emmergency_arrivals=True,
                 include_hnfc_moving=True,
                 include_nb_hospit_np_from_ED_children=True,
                 include_nb_hospit_np_from_ED_adults=True) -> None:
        super().__init__(name, logger,
                         date_max_fetchable=dt.datetime.strptime('31-12-2023', '%d-%m-%Y'))
        self.include_emmergency_arrivals = include_emmergency_arrivals
        self.include_hnfc_moving = include_hnfc_moving
        self.include_nb_hospit_np_from_ED_children = include_nb_hospit_np_from_ED_children
        self.include_nb_hospit_np_from_ED_adults = include_nb_hospit_np_from_ED_adults

    def include_nb_emmergencies(self, from_date, to_date, etablissement, feature_dir) -> None:
        """
        Adds the target to the features.
        """
        self.logger.info("Intégration de la target")

        # file_name = f"Export complet {etablissement}.xlsx"
        file_name = f"export_{etablissement}.csv"

        self.logger.info(
            f"  - Chargement des données de {etablissement} depuis le fichier Excel")
        # data = pd.read_excel(
        #     feature_dir / f"urgences/{file_name}", sheet_name=1)

        data = pd.read_csv(
            feature_dir / f"urgences/exports/{file_name}", sep=";")

        if "annee" in data:
            data.drop(axis=1, columns="annee", inplace=True)

        data.rename(columns={
            "Total": f"nb_emmergencies", "date_entree": "date"}, inplace=True)

        if data["date"].dtype != "datetime64[ns]":
            data["date"] = pd.to_datetime(data["date"])

        data.sort_values(by="date", inplace=True)
        data.set_index('date', inplace=True)

        data.dropna(subset=[f"nb_emmergencies"], inplace=True)

        return data

    def include_nb_hospitalized_np_from_emmergencies_adults(self, from_date, to_date, etablissement, feature_dir):
        # hospitalized = pd.read_excel(
        #     feature_dir / "hospitalisations/non-programé/urgences/RPU_vers_hospit_adultes.xlsx")
        # hospitalized['date_entree'] = pd.to_datetime(
        #     hospitalized['date_entree'], unit='D', origin='1899-12-30')

        file_name = f'rpu_hospit_adultes_{etablissement}.csv'
        data = pd.read_csv(
            feature_dir / f"hospitalisations/non-programé/urgences/exports/{file_name}", sep=";")

        data.rename(columns={
                    "Total": "nb_hospit_np_from_ED_adults", "date_entree": "date"}, inplace=True)
        if data["date"].dtype != "datetime64[ns]":
            data["date"] = pd.to_datetime(data["date"])
        data.sort_values(by="date", inplace=True)
        data.set_index("date", inplace=True)

        return data

    def include_nb_hospitalized_np_from_emmergencies_children(self, from_date, to_date, etablissement, feature_dir):
        file_name_all = f'rpu_hospit_total_{etablissement}.csv'
        hospitalized_all = pd.read_csv(
            feature_dir / f"hospitalisations/non-programé/urgences/exports/{file_name_all}", sep=";")
        hospitalized_all.rename(columns={"date_entree": "date"}, inplace=True)
        if hospitalized_all["date"].dtype != "datetime64[ns]":
            hospitalized_all["date"] = pd.to_datetime(hospitalized_all["date"])
        hospitalized_all.set_index('date', inplace=True)

        file_name_adults = f'rpu_hospit_adultes_{etablissement}.csv'
        hospitalized_adultes = pd.read_csv(
            feature_dir / f"hospitalisations/non-programé/urgences/exports/{file_name_adults}", sep=";")
        hospitalized_adultes.rename(
            columns={"date_entree": "date"}, inplace=True)
        if hospitalized_adultes["date"].dtype != "datetime64[ns]":
            hospitalized_adultes["date"] = pd.to_datetime(
                hospitalized_adultes["date"])
        hospitalized_adultes.set_index('date', inplace=True)

        hospitalized_children = pd.DataFrame(hospitalized_all['Total'] - hospitalized_adultes['Total'], columns=[
                                             'Total'], index=hospitalized_adultes.index)
        hospitalized_children.rename(
            columns={"Total": "nb_hospit_np_from_ED_children"}, inplace=True)

        return hospitalized_children

    def include_nb_hospitalized_np_adults(self, from_date, to_date, feature_dir):
        data = pd.read_csv(
            feature_dir / "hospitalisations/non-programé/export_np_adulte.csv", sep=";")
        data['date_entree'] = pd.to_datetime(
            data['date_entree'], format="%d/%m/%Y")
        data.rename(columns={"date_entree": "date"}, inplace=True)
        data.set_index("date", inplace=True)
        data.rename(
            columns={"valeur": "nb_hospit_np_adults%%J+1%%mean_7J"}, inplace=True)
        data['nb_hospit_np_adults%%J+1%%mean_7J'] = data['nb_hospit_np_adults%%J+1%%mean_7J'] / 7

        return data

    def include_nb_hospitalized_np_children(self, from_date, to_date, feature_dir):
        data = pd.read_csv(
            feature_dir / "hospitalisations/non-programé/export_np_pediatrie.csv", sep=";")
        data['date_entree'] = pd.to_datetime(
            data['date_entree'], format="%d/%m/%Y")
        data.rename(columns={"date_entree": "date"}, inplace=True)
        data.set_index("date", inplace=True)
        data.rename(
            columns={"valeur": "nb_hospit_np_children%%J+1%%mean_7J"}, inplace=True)
        data["nb_hospit_np_children%%J+1%%mean_7J"] = data['nb_hospit_np_children%%J+1%%mean_7J'] / 7

        return data

    def include_nb_hospitalized_adults(self, from_date, to_date, feature_dir):
        data = pd.read_csv(
            feature_dir / "hospitalisations/non-programé/export_tot_adultes.csv", sep=";")
        data['date_entree'] = pd.to_datetime(
            data['date_entree'], format="%d/%m/%Y")
        data.rename(columns={"date_entree": "date"}, inplace=True)
        data.set_index("date", inplace=True)
        data.rename(
            columns={"valeur": "nb_hospit_adults%%J+1%%mean_7J"}, inplace=True)
        data["nb_hospit_adults%%J+1%%mean_7J"] = data['nb_hospit_adults%%J+1%%mean_7J'] / 7

        return data

    def include_nb_hospitalized_children(self, from_date, to_date, feature_dir):
        data = pd.read_csv(
            feature_dir / "hospitalisations/non-programé/export_tot_enfants.csv", sep=";")
        data['date_entree'] = pd.to_datetime(
            data['date_entree'], format="%d/%m/%Y")
        data.rename(columns={"date_entree": "date"}, inplace=True)
        data.set_index("date", inplace=True)
        data.rename(
            columns={"valeur": "nb_hospit_children%%J+1%%mean_7J"}, inplace=True)
        data["nb_hospit_children%%J+1%%mean_7J"] = data['nb_hospit_children%%J+1%%mean_7J'] / 7

        return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        """
        Fetches the data by adding the target and calling the parent's fetch_data method.
        """

        assert 'location' in kwargs, "location must be provided in config"
        location: Location = kwargs.get('location')
        etablissement = location.name
        assert isinstance(etablissement, str), "etablissement must be a string"

        feature_dir = kwargs.get("feature_dir")
        include_hospit_np_from_emmergencies_adults = kwargs.get(
            "include_hospit_np_from_emmergencies_adults", False)

        # Set starting date, default is 01/01/1970
        start_date = kwargs.get("start_date")
        start_date = dt.datetime.strptime(
            start_date, "%Y-%M-%d") if isinstance(start_date, str) else start_date

        # Set ending date, default is today's date
        stop_date = kwargs.get("stop_date")
        stop_date = dt.datetime.strptime(
            stop_date, "%Y-%M-%d") if isinstance(stop_date, str) else stop_date
        # print(f"Fetching data from {start_date} to {stop_date}")
        date_range = pd.date_range(
            start=start_date, end=stop_date, freq='1D', name="date", )

        data = pd.DataFrame(index=date_range)

        include_nb_hospit_np_from_ED_children = kwargs.get(
            'include_nb_hospit_np_from_ED_children', self.include_nb_hospit_np_from_ED_children)
        include_nb_hospit_np_from_ED_adults = kwargs.get(
            'include_nb_hospit_np_from_ED_adults', self.include_nb_hospit_np_from_ED_adults)
        include_emmergency_arrivals = kwargs.get(
            'include_emmergency_arrivals', self.include_emmergency_arrivals)

        if include_emmergency_arrivals:
            data = data.join(self.include_nb_emmergencies(
                start_date, stop_date, etablissement=etablissement, feature_dir=feature_dir))
        self.logger.info(f"{data}")

        if include_nb_hospit_np_from_ED_children:
            data = data.join(self.include_nb_hospitalized_np_from_emmergencies_children(
                start_date, stop_date, etablissement, feature_dir))
        self.logger.info(f"{data}")

        if include_nb_hospit_np_from_ED_adults:
            data = data.join(self.include_nb_hospitalized_np_from_emmergencies_adults(
                start_date, stop_date, etablissement, feature_dir))
        self.logger.info(f"{data}")

        # data = data.join(self.include_nb_hospitalized_np_adults(start_date, stop_date, feature_dir=feature_dir))

        return data
