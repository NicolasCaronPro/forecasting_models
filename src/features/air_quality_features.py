"""
This module contains the AirQualityFeatures class, which represents features related to air quality.

The AirQualityFeatures class inherits from the BaseFeature class and provides methods to include air quality data and fetch the data.
"""
import time
from typing import Optional, Dict
import pandas as pd
from src.features.base_features import BaseFeature
from pathlib import Path
import os
from src.location.location import Location
from shapely.geometry import Polygon, Point
import datetime as dt


class AirQualityFeatures(BaseFeature):
    """
    Represents features related to air quality.

    Attributes:
        departement (str): The department code.
        archived_data_dir (Path): The path to the archived data directory.
    """

    def __init__(self, name: str = None, logger=None) -> None:
        super().__init__(name, logger, date_max_fetchable=dt.datetime.strptime(
            '31-12-2023', '%d-%m-%Y'))

    def __include_air_quality(self, location: Location, feature_dir: str) -> pd.DataFrame:
        # Récupérer les archives sur :
        # https://www.geodair.fr/donnees/export-advanced

        t = time.time()
        self.logger.info("On regarde la qualité de l'air")

        # On récupère les codes des stations de mesure de la qualité de l'air pour le département
        df = pd.read_csv(feature_dir / 'stations_geodair.csv',
                         sep=';', dtype={'departement': str})

        CODES = []
        for i in range(len(df)):
            if location.is_in_shape(Point(df.iloc[i]['longitude'], df.iloc[i]['latitude'])) or df.iloc[i]['departement'] == location.code_departement:
                CODES.append(df.iloc[i]['station'])

        # if len(CODES) == 0:
        # CODES.append(list(df.loc[df['departement'] ==
        #             location.code_departement].station.values))
        # print(CODES)

        self.logger.info(f"On s'intéresse aux codes : {', '.join(CODES)}")

        archived_data_dir = Path(feature_dir / 'archived')
        archived_data_dir.mkdir(exist_ok=True, parents=True)
        if not (archived_data_dir / 'pollution_historique.feather').is_file():
            self.logger.info("On calcule le dataframe d'archive de l'air")

            dico: Dict[str, pd.DataFrame] = {}
            for fic in archived_data_dir.iterdir():
                if fic.suffix == '.csv':
                    df = pd.read_csv(fic, sep=";")

                    # On vérifie qu'il n'y a qu'un seul polluant dans le fichier et on le récupère
                    assert len(df['Polluant'].unique()) == 1
                    polluant = df['Polluant'].unique()[0]
                    polluant = polluant.replace('.', '')

                    if polluant not in dico:
                        dico[polluant] = df.loc[df['code site'].isin(CODES)]
                    else:
                        dico[polluant] = pd.concat([dico[polluant], pd.read_csv(
                            fic, sep=";").loc[df['code site'].isin(CODES)]])

            for polluant in dico:
                dico[polluant].to_feather(
                    archived_data_dir / f"{polluant}.feather")

            del dico
            dg = None
            for fic in archived_data_dir.iterdir():
                if fic.suffix == '.feather' and fic.stem != 'pollution_historique':
                    df = pd.read_feather(fic)
                    polluant = fic.stem
                    df.rename({'Date de début': 'date'},
                              axis=1, inplace=True)
                    groups = df.groupby('code site')
                    for name, group in groups:
                        if dg is None:
                            dg = group[['date', 'valeur']]
                            dg = dg.rename(
                                {'valeur': f'{polluant}_{name}'}, axis=1)
                            dg.set_index('date', inplace=True)
                        else:
                            dh = group[['date', 'valeur']]
                            dh = dh.rename(
                                {'valeur': f'{polluant}_{name}'}, axis=1)
                            dh.set_index('date', inplace=True)
                            dg = pd.merge(dg, dh, left_index=True,
                                          right_index=True, how='outer')

            dg.index = pd.to_datetime(dg.index)

            data = dg.copy(deep=True)

            del df
            del dg
            del dh

            data.interpolate(method='linear', inplace=True)
            data.ffill(inplace=True)
            data.bfill(inplace=True)

            # for k in sorted(data.columns):
            #     if data[k].isna().sum() > self.max_nan:
            #         self.logger.error(
            #             f"{k} possède trop de NaN ({data[k].isna().sum()})")

            # data.rename({k: f"{self.name}_{k}"}, axis=1, inplace=True)

            data.to_feather(archived_data_dir /
                            'pollution_historique.feather')
        else:
            self.logger.info("On relit le dataframe d'archive de l'air")
            data = pd.read_feather(
                archived_data_dir / 'pollution_historique.feather')

        self.logger.info(
            f"Fin de la gestion de la qualité de l'air en {time.time()-t:.2f} s.")
        return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'location' in kwargs, "location must be provided in config"
        location = kwargs.get('location')
        assert type(location) == Location, "location must be a Location object"

        feature_dir = kwargs.get("feature_dir")

        return self.__include_air_quality(location, feature_dir)
