from src.features.base_features import BaseFeature, Config
from typing import Optional, Dict
import pandas as pd


class TrafficFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)

    def include_trafic(self):
        # Historique des accidents de la route
        self.logger.info("Intégration des données de trafic")
        accidents = pd.read_csv(
            self.data_dir / 'nombre_accidents_par_date_par_departement.csv', sep=',')
        accidents = accidents.loc[accidents['dep']
                                  == self.config.get('departement')]
        accidents['date'] = pd.to_datetime(accidents['date_entree'])
        accidents.drop(columns=['date_entree'], inplace=True)
        accidents.set_index('date', inplace=True)

        # On complète l'année 2023 avec les données de 2022
        # TODO: On impute plus tard
        # # Filtrer les données pour l'année 2022
        # df_2022 = accidents[(accidents.index >= '2022-01-01') & (accidents.index <= '2022-12-31')].copy()

        # # Créer une nouvelle colonne 'date_entree' pour 2023
        # df_2023 = df_2022.copy()
        # df_2023.index = df_2023.index + pd.DateOffset(years=1)

        # # Fusionner les DataFrames de 2019-2022 avec celui de 2023
        # accidents = pd.concat([accidents, df_2023])

        # Intégration des données au dataframe
        self.data = self.data.merge(
            accidents['nb_accidents'], left_index=True, right_index=True, how='left')

        # On remplit les jours sans accidents par 0
        self.data['nb_accidents'].fillna(0, inplace=True)

        # # Ajout des features décalés pour les DECALAGE_TRAFIC jours précédents
        # for dec in range(1,DECALAGE_TRAFIC+1):
        #     features['nb_accidents-'+str(dec)] = features['nb_accidents'].shift(dec)

        # # Ajout de la moyenne glissante sur FENETRE_GLISSANTE jours
        # features['nb_accidents_rolling'] = features['nb_accidents'].rolling(window=FENETRE_GLISSANTE, closed="left").mean()

        # return features

    def fetch_data_function(self, *args, **kwargs) -> None:
        self.include_trafic()
