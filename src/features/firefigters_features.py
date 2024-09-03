import pickle
from src.features.base_features import BaseFeature, Config
from typing import Optional, Dict
import pandas as pd
import datetime as dt

class FireFightersFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)

    def include_pompier(self):
        self.logger.info("Intégration des données des pompiers")
        appel = pd.read_feather(self.data_dir / 'appels_25_full.feather')
        tous = pd.read_feather(self.data_dir / 'Tous_25_full.feather')
        # On groupe les appel par jour
        appel = appel.groupby(pd.Grouper(key='creneau', freq='D')).sum()
        # Définir la fonction à appliquer à chaque groupe
        def calculate_target(group):
            target = group['interventions0'].sum() - group['en_cours0'].shift().fillna(0).sum()
            return pd.Series([target], index=['target'])
        # On groupe les interventions par jour
        tous_daily = tous.groupby([pd.Grouper(key='creneau', freq='D')]).apply(calculate_target).reset_index()
        # Intégration des données au dataframe
        self.data['appels_25'] = appel.loc[appel.index.isin(self.data.index), 'target'].values
        self.data['interventions_25'] = tous_daily.loc[tous_daily['creneau'].isin(self.data.index), 'target'].values

        # # Ajout des features décalés pour les DECALAGE_POMPIER jours précédents
        # for dec in range(1,DECALAGE_POMPIER+1):
        #     self.data['appels_25-'+str(dec)] = self.data['appels_25'].shift(dec)
        #     self.data['interventions_25-'+str(dec)] = self.data['interventions_25'].shift(dec)
        
        # # Ajout de la moyenne glissante sur FENETRE_GLISSANTE jours
        # self.data['appels_25_rolling'] = self.data['appels_25'].rolling(window=FENETRE_GLISSANTE, closed="left").mean()
        # self.data['interventions_25_rolling'] = self.data['interventions_25'].rolling(window=FENETRE_GLISSANTE, closed="left").mean()
    
    def include_fire(self):
        arra = pickle.load(open(self.data_dir / "Y_full_10.pkl", "rb"))
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

        dates = find_dates_between('2017-06-12', '2023-09-11')
        df.rename(columns={1: 'longitude', 2:'latitude', 3:'departement', 4:'date_index', 6:'nb_incendie', 7:'risque'}, inplace=True)
        def convert_to_date(date_index):
            return dates[int(date_index)]

        df['date'] = df['date_index'].apply(convert_to_date)
        grouped_df = df.loc[df['departement'] == 25].groupby('date').agg({'nb_incendie': 'sum', 'risque': 'mean', 'longitude': 'mean', 'latitude': 'mean', 'departement': 'mean'}).reset_index()
        grouped_df.set_index('date', inplace=True)
        grouped_df.to_csv(self.data_dir / "incendies_25.csv")
        self.data = pd.concat([self.data, grouped_df['nb_incendie']], axis=1)
    
    def fetch_data(self) -> None:
        self.include_pompier()
        self.include_fire()
        super().fetch_data()