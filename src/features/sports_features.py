from src.features.base_features import BaseFeature, Config
from typing import Optional, Dict
import pandas as pd


class SportsCompetitionFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)

    def include_foot(self):
        self.logger.info("Intégration des données de football")
        # On récupère les données de football depuis chaque fichier csv
        file_list = list(self.data_dir.glob('*.csv'))
        for file in file_list:
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date_entree'])
            df.drop(columns=['date_entree'], inplace=True)
            df.set_index('date', inplace=True)
            # print(df)
            # print(self.data)
            self.data = pd.merge(self.data, df, how='left', on='date')

        self.logger.info("Données de football intégrées")
        return self.data

    def fetch_data_function(self, *args, **kwargs) -> None:
        self.include_foot()
