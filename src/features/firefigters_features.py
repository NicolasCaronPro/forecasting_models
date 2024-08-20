from src.features.base_features import BaseFeature, Config
from typing import Optional, Dict

class FireFightersFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)

    def include_pompier(features):
        logger.info("Intégration des données des pompiers")
        appel = pd.read_feather(dir_features / 'pompier/appels_25_full.feather')
        tous = pd.read_feather(dir_features / 'pompier/Tous_25_full.feather')
        # On groupe les appel par jour
        appel = appel.groupby(pd.Grouper(key='creneau', freq='D')).sum()
        # Définir la fonction à appliquer à chaque groupe
        def calculate_target(group):
            target = group['interventions0'].sum() - group['en_cours0'].shift().fillna(0).sum()
            return pd.Series([target], index=['target'])
        # On groupe les interventions par jour
        tous_daily = tous.groupby([pd.Grouper(key='creneau', freq='D')]).apply(calculate_target).reset_index()
        # Intégration des données au dataframe
        features['appels_25'] = appel.loc[appel.index.isin(features['date_entree']), 'target'].values
        features['interventions_25'] = tous_daily.loc[tous_daily['creneau'].isin(features['date_entree']), 'target'].values

        # Ajout des features décalés pour les DECALAGE_POMPIER jours précédents
        for dec in range(1,DECALAGE_POMPIER+1):
            features['appels_25-'+str(dec)] = features['appels_25'].shift(dec)
            features['interventions_25-'+str(dec)] = features['interventions_25'].shift(dec)
        
        # Ajout de la moyenne glissante sur FENETRE_GLISSANTE jours
        features['appels_25_rolling'] = features['appels_25'].rolling(window=FENETRE_GLISSANTE, closed="left").mean()
        features['interventions_25_rolling'] = features['interventions_25'].rolling(window=FENETRE_GLISSANTE, closed="left").mean()
    
    def include_fire():
        arra = pickle.load(open("../../data/features/pompier/Y_full_10.pkl", "rb"))
        df = pd.DataFrame(arra)

        def find_dates_between(start, end):
            start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
            end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()
            delta = dt.timedelta(days=1)
            date = start_date
            res = []
            while date < end_date:
                    res.append(date.strftime("%Y-%m-%d"))
                    date += delta
            return res

        dates = find_dates_between('2017-06-12', '2023-09-11')
        df.rename(columns={1: 'longitude', 2:'latitude', 3:'departement', 4:'date_index', 6:'nb_incendie', 7:'risque'}, inplace=True)
        def convert_to_date(date_index):
            return dates[int(date_index)]

        df['date'] = df['date_index'].apply(convert_to_date)
        grouped_df = df.loc[df['departement'] == 25].groupby('date').agg({'nb_incendie': 'sum', 'risque': 'mean', 'longitude': 'mean', 'latitude': 'mean', 'departement': 'mean'}).reset_index()
        grouped_df.set_index('date', inplace=True)
        grouped_df.to_csv("../../data/features/pompier/incendies_25.csv")
    
    def fetch_data(self) -> None:
        self.include_pompier()
        self.include_fire()
        super().fetch_data()