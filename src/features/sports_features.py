from src.features.base_features import BaseFeature
from typing import Optional, Dict
import pandas as pd
from pathlib import Path
import datetime as dt


class SportsCompetitionFeatures(BaseFeature):
    def __init__(self, name: str = None, logger=None) -> None:
        super().__init__(name, logger, date_max_fetchable=dt.datetime.strptime(
            '31-12-2023', '%d-%m-%Y'))

    def include_foot(self, feature_dir, date_range):
        self.logger.info("Intégration des données de football")
        data = pd.DataFrame(index=date_range)
        # On récupère les données de football depuis chaque fichier csv
        feature_dir = Path(feature_dir)
        file_list = list(feature_dir.glob('*.csv'))
        for file in file_list:
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date_entree'])
            df.drop(columns=['date_entree'], inplace=True)
            df.set_index('date', inplace=True)
            # print(df)
            # print(self.data)
            data = pd.merge(data, df, how='left', on='date')

        # Sum all the columns in to one columns
        data = pd.DataFrame(data.sum(axis=1), columns=[
                            'foot'], index=date_range)

        self.logger.info("Données de football intégrées")
        return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'feature_dir' in kwargs, f"Le paramètre'feature_dir' est obligatoire pour fetch la feature {self.name}"
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"

        feature_dir = kwargs.get("feature_dir")
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        # TODO: do not hardcode freq
        date_range = pd.date_range(
            start=start_date, end=stop_date, freq='1D', name="date")
        data = pd.DataFrame(index=date_range)

        data = data.join(self.include_foot(feature_dir, date_range))

        return data
