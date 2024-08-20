from src.features.base_features import BaseFeature, Config
from typing import Optional, Dict

class SportsCompetitionFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)

    def include_foot(features):
        logger.info("Intégration des données de football")
        # On récupère les données de football depuis chaque fichier csv
        file_list = list(dir_foot.glob('*.csv'))
        for file in file_list:
            df = pd.read_csv(file)
            df['date_entree'] = pd.to_datetime(df['date_entree'])
            features = pd.merge(features, df, how='left', on='date_entree')
        
        logger.info("Données de football intégrées")
        return features
    
    def fetch_data(self) -> None:
        self.include_foot()
        super().fetch_data()