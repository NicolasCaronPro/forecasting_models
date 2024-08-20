from src.features.base_features import BaseFeature, Config
from typing import Optional, Dict

class TrafficFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)
    
    def include_trafic(features):
        # Historique des accidents de la route
        logger.info("Intégration des données de trafic")
        accidents = pd.read_csv(dir_trafic / 'nombre_accidents_par_date_par_departement.csv', sep=',')
        accidents = accidents.loc[accidents['dep'] == DEPARTEMENT]
        accidents['date_entree'] = pd.to_datetime(accidents['date_entree'])
        # Filtrer les données pour l'année 2022
        df_2022 = accidents[(accidents['date_entree'] >= '2022-01-01') & (accidents['date_entree'] <= '2022-12-31')].copy()

        # Créer une nouvelle colonne 'date_entree' pour 2023
        df_2023 = df_2022.copy()
        df_2023['date_entree'] = df_2023['date_entree'] + pd.DateOffset(years=1)

        # Fusionner les DataFrames de 2019-2022 avec celui de 2023
        accidents = pd.concat([accidents, df_2023])

        # Intégration des données au dataframe
        features = pd.merge(features, accidents[['date_entree', 'nb_accidents']], how='left', on='date_entree')

        # On remplit les jours sans accidents par 0
        features['nb_accidents'].fillna(0, inplace=True)

        # Ajout des features décalés pour les DECALAGE_TRAFIC jours précédents
        for dec in range(1,DECALAGE_TRAFIC+1):
            features['nb_accidents-'+str(dec)] = features['nb_accidents'].shift(dec)
        
        # Ajout de la moyenne glissante sur FENETRE_GLISSANTE jours
        features['nb_accidents_rolling'] = features['nb_accidents'].rolling(window=FENETRE_GLISSANTE, closed="left").mean()

        return features
    
    def fetch_data(self) -> None:
        self.include_trafic()
        super().fetch_data()