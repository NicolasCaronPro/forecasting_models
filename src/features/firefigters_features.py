import pickle
from src.features.base_features import BaseFeature
from typing import Optional, Dict
import pandas as pd
import datetime as dt


class FireFightersFeatures(BaseFeature):
    def __init__(self, name:str = None, logger=None, include_calls=True, include_fire_data=True) -> None:
        super().__init__(name, logger)
        self.include_calls = include_calls
        self.include_fire_data = include_fire_data

    def include_pompier(self, date_range:pd.DatetimeIndex, feature_dir):
        self.logger.info("Intégration des données des pompiers")
        data = pd.DataFrame(index=date_range)
        appel = pd.read_feather(feature_dir / 'appels_25_full.feather')
        tous = pd.read_feather(feature_dir / 'Tous_25_full.feather')
        # On groupe les appel par jour
        appel = appel.groupby(pd.Grouper(key='creneau', freq='D')).sum()
        # Définir la fonction à appliquer à chaque groupe

        def calculate_target(group):
            target = group['interventions0'].sum(
            ) - group['en_cours0'].shift().fillna(0).sum()
            return pd.Series([target], index=['target'])
        # On groupe les interventions par jour
        tous_daily = tous.groupby([pd.Grouper(key='creneau', freq='D')]).apply(
            calculate_target).reset_index()
        # Intégration des données au dataframe
        data['appels_25'] = appel.loc[appel.index.isin(
            data.index), 'target'].values
        data['interventions_25'] = tous_daily.loc[tous_daily['creneau'].isin(
            data.index), 'target'].values

        return data
    
    def include_fire(self, date_range:pd.DatetimeIndex, feature_dir):
        arra = pickle.load(open(feature_dir / "Y_full_10.pkl", "rb"))
        df = pd.DataFrame(arra)

        def find_dates_between(start, end):
            start_date = dt.datetime.strptime(start, '%Y-%m-%d')
            end_date = dt.datetime.strptime(end, '%Y-%m-%d')
            delta = dt.timedelta(days=1)
            date = start_date
            res = []
            while date < end_date:
                res.append(date)
                date += delta
            return res

        df.rename(columns={1: 'longitude', 2: 'latitude', 3: 'departement',
                  4: 'date_index', 6: 'nb_incendie', 7: 'risque'}, inplace=True)

        def convert_to_date(date_index):
            dates = find_dates_between('2017-06-12', '2023-09-11')
            return dates[int(date_index)]

        df['date'] = df['date_index'].apply(convert_to_date)
        grouped_df = df.loc[df['departement'] == 25].groupby('date').agg(
            {'nb_incendie': 'sum', 'risque': 'mean', 'longitude': 'mean', 'latitude': 'mean', 'departement': 'mean'}).reset_index()
        grouped_df.set_index('date', inplace=True)
        grouped_df.to_csv(feature_dir / "incendies_25.csv")
        data = grouped_df['nb_incendie']
        return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'feature_dir' in kwargs, f"Le paramètre'feature_dir' est obligatoire pour fetch la feature {self.name}"
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        
        feature_dir = kwargs.get("feature_dir")
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date") # TODO: do not hardcode freq
        data = pd.DataFrame(index=date_range)

        if self.include_calls:
            data = data.join(self.include_pompier(date_range, feature_dir))

        if self.include_fire:
            data = data.join(self.include_fire(date_range, feature_dir))
        
        return data
