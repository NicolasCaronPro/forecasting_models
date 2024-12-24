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

    # Old function, not used
    def __include_air_quality_old(self, location: Location, feature_dir: str) -> pd.DataFrame:
        # Récupérer les archives sur :
        # https://www.geodair.fr/donnees/export-advanced

        t = time.time()
        self.logger.info("On regarde la qualité de l'air")

        # On récupère les codes des stations de mesure de la qualité de l'air pour le département
        df = pd.read_csv(feature_dir / 'stations_geodair.csv',
                         sep=';', dtype={'departement': str})

        CODES = []
        for i in range(len(df)):
            # location.is_in_shape(Point(df.iloc[i]['longitude'], df.iloc[i]['latitude'])) or
            if df.iloc[i]['departement'] == location.code_departement:
                CODES.append(df.iloc[i]['station'])

        # if len(CODES) == 0:
        # CODES.append(list(df.loc[df['departement'] ==
        #             location.code_departement].station.values))

        self.logger.info(f"On s'intéresse aux codes : {', '.join(CODES)}")

        archived_data_dir = Path(feature_dir / 'archived')
        archived_data_dir.mkdir(exist_ok=True, parents=True)
        if not (archived_data_dir / f'pollution_historique.feather').is_file():
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

    def __include_air_quality(self, location: Location, feature_dir: str, polluants: set = {'O3', 'NO2', 'PM25', 'PM10'}) -> pd.DataFrame:
        def air_avg(df: pd.DataFrame, polluants):
            for name in polluants:
                # Filter columns that start with name +"_"
                name_columns = [col for col in df.columns if col.startswith(name + "_")]
                # Check if there is only one name + "_" column
                if len(name_columns) == 1:
                    # If only one column, copy its values to the name + '_avg' column
                    df[name + '_avg'] = df[name_columns[0]]
                elif len(name_columns) > 1:
                    # If more than one column, calculate the row-wise mean
                    df[name + '_avg'] = df[name_columns].mean(axis=1)
                # Drop the name + "_" columns
                df = df.drop(columns=name_columns)
            return df

        # Read stations data
        df_stations = pd.read_csv(feature_dir / 'stations_geodair.csv',
                        sep=';', dtype={'departement': str})
        
        # Get relevant station codes
        CODES = df_stations[df_stations['departement'] == location.code_departement]['station'].tolist()
        if location.name == 'HÔPITAL NORD FRANCHE COMTE':
            CODES = ['FR82020', 'FR82021', 'FR82010', 'FR82012']
        self.logger.info(f"On s'intéresse aux codes : {', '.join(CODES)}")

        # Process CSV files directly
        archived_data_dir = Path(feature_dir / 'archived')
        dg = None
        missing_polluants = set(polluants)



        # Process each CSV file
        for fic in archived_data_dir.iterdir():
            if fic.suffix == '.csv':
                df = pd.read_csv(fic, sep=";")
                display(df)
                df = df[df['code site'].isin(CODES)]
                display(df)
                assert len(df['Polluant'].unique()) == 1, f"Le fichier contient plusieurs polluants : {df['Polluant'].unique()}"
                polluant = df['Polluant'].unique()[0].replace('.', '')
                if polluant not in polluants:
                    self.logger.info(
                        f"Le polluant {polluant} n'est pas dans la liste des polluants à traiter")
                    continue
                else:
                    try:
                        missing_polluants.remove(polluant)
                        self.logger.info(f"On traite le polluant {polluant}")
                    except KeyError:
                        self.logger.error(
                            f"Le polluant {polluant} est déjà présent dans le dataframe")
                        continue
                # polluants.append(polluant)
                # Filter data for relevant stations
                df.rename({'Date de début': 'date'}, axis=1, inplace=True)
                
                # Process each station's data
                for name, group in df.groupby('code site'):
                    temp_df = group[['date', 'valeur']]
                    temp_df = temp_df.rename({'valeur': f'{polluant}_{name}'}, axis=1)
                    temp_df.set_index('date', inplace=True)
                    
                    if dg is None:
                        dg = temp_df
                    else:
                        dg = pd.merge(dg, temp_df, left_index=True,
                                    right_index=True, how='outer')

        # Final processing
        dg.index = pd.to_datetime(dg.index)

        data = air_avg(dg.copy(deep=True), polluants)

        if missing_polluants:
            self.logger.warning(
                f"Les polluants {', '.join(missing_polluants)} ne sont pas présents dans les données")

        if len(data.columns) == 0:
            self.logger.error("Le dataframe est vide")
            return None

        
        # Clean up data
        data.interpolate(method='linear', inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        return data


    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'location' in kwargs, "location must be provided in config"
        location = kwargs.get('location')
        assert type(location) == Location, "location must be a Location object"

        feature_dir = kwargs.get("feature_dir")

        data = self.__include_air_quality(location, feature_dir)

        # Prefix all columns with air_
        data.columns = [f"air_{col}" for col in data.columns]

        return data