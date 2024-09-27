"""
This module contains the AirQualityFeatures class, which represents features related to air quality.

The AirQualityFeatures class inherits from the BaseFeature class and provides methods to include air quality data and fetch the data.
"""
import time
from typing import Optional, Dict
import pandas as pd
from src.features.base_features import BaseFeature, Config


class AirQualityFeatures(BaseFeature):
    """
    Represents features related to air quality.

    Attributes:
        departement (str): The department code.
        archived_data_dir (Path): The path to the archived data directory.
    """

    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None, drop_const_cols=True) -> None:
        super().__init__(config, parent)
        self.archived_data_dir = self.data_dir / 'archived'
        self.archived_data_dir.mkdir(exist_ok=True, parents=True)
        self.drop_const_cols = drop_const_cols

        assert 'departement' in self.config, "departement must be provided in config"
        self.departement = self.config.get('departement')
        assert type(self.departement) == str, "departement must be a string"

    def __include_air_quality(self):
        # Récupérer les archives sur :
        # https://www.geodair.fr/donnees/export-advanced

        t = time.time()
        self.logger.info("On regarde la qualité de l'air")

        # On récupère les codes des stations de mesure de la qualité de l'air pour le département
        df = pd.read_csv(self.data_dir / 'stations_geodair.csv',
                         sep=';', dtype={'departement': str})
        CODES = list(df.loc[df['departement'] ==
                     self.departement].station.values)
        self.logger.info(f"On s'intéresse aux codes : {', '.join(CODES)}")

        if not (self.archived_data_dir / 'pollution_historique.feather').is_file():
            self.logger.info("On calcule le dataframe d'archive de l'air")

            dico: Dict[str, pd.DataFrame] = {}
            for fic in self.archived_data_dir.iterdir():
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
                    self.archived_data_dir / f"{polluant}.feather")

            del dico
            dg = None
            for fic in self.archived_data_dir.iterdir():
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

            self.data = self.data.join(dg)

            del df
            del dg
            del dh

            self.data.interpolate(method='linear', inplace=True)
            self.data.ffill(inplace=True)
            self.data.bfill(inplace=True)

            for k in sorted(self.data.columns):
                if self.data[k].isna().sum() > self.max_nan:
                    self.logger.error(
                        f"{k} possède trop de NaN ({self.data[k].isna().sum()})")

                # self.data.rename({k: f"{self.name}_{k}"}, axis=1, inplace=True)

            self.data.to_feather(self.archived_data_dir /
                                 'pollution_historique.feather')
        else:
            self.logger.info("On relit le dataframe d'archive de l'air")
            self.data = pd.read_feather(
                self.archived_data_dir / 'pollution_historique.feather')

        self.logger.info(
            f"Fin de la gestion de la qualité de l'air en {time.time()-t:.2f} s.")

    def fetch_data_function(self, *args, **kwargs) -> None:
        self.__include_air_quality()
