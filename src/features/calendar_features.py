from typing import Optional
from src.features.base_features import BaseFeature, Config

# TODO: remove and implement as a transformer

class CalendarFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)

    def include_calendar(self):
        self.logger.info("Intégration des données calendaires")

        self.data['dayofweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.logger.info("  - Ajout du jour dans l'année")
        self.data['is_leap'] = self.data.index.is_leap_year
        self.data['is_leap'] = (~self.data['is_leap']).to_numpy(dtype=int)
        self.data['after_feb'] = self.data.index.month > 2
        self.data['after_feb'] = self.data['after_feb'].to_numpy(dtype=int)
        self.data['dayofYear'] = self.data.index.dayofyear + self.data['after_feb'].to_numpy() * \
                                self.data['is_leap'].to_numpy()
        self.data.drop(['is_leap', 'after_feb'], axis=1, inplace=True)

        self.data['year'] = self.data.index.year
        self.data['quarter'] = self.data.index.quarter
        self.data['week'] = self.data.index.week
        
        self.data['hour'] = self.data.index.hour
        self.data['minute'] = self.data.index.minute
        self.data['second'] = self.data.index.second

    def fetch_data(self) -> None:
        self.include_calendar()
        super().fetch_data()