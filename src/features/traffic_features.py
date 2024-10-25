from src.features.base_features import BaseFeature
from typing import Optional, Dict
import pandas as pd
from pathlib import Path
from src.location.location import Location

class TrafficFeatures(BaseFeature):
    def __init__(self, name:str = None, logger=None) -> None:
        super().__init__(name, logger)

    def include_trafic(self, feature_dir, date_range, departement):
        # Historique des accidents de la route
        feature_dir = Path(feature_dir)
        data = pd.DataFrame(index=date_range)
        self.logger.info("Intégration des données de trafic")
        accidents = pd.read_csv(
            feature_dir / 'nombre_accidents_par_date_par_departement.csv', sep=',')
        accidents = accidents.loc[accidents['dep'] == departement]
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
        data = data.merge(
            accidents['nb_accidents'], left_index=True, right_index=True, how='left')

        # On remplit les jours sans accidents par 0
        data['nb_accidents'] = data['nb_accidents'].fillna(0)

        return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'feature_dir' in kwargs, f"Le paramètre'feature_dir' est obligatoire pour fetch la feature {self.name}"
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        assert 'location' in kwargs, "location must be provided in config"
        location = kwargs.get('location')
        departement = location.code_departement
        feature_dir = kwargs.get("feature_dir")
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date") # TODO: do not hardcode freq
        data = pd.DataFrame(index=date_range)

        data = data.join(self.include_trafic(feature_dir, date_range, departement))
        return data
