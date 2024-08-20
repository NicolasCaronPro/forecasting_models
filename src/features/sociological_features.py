from src.features.base_features import BaseFeature, Config
import jours_feries_france
import vacances_scolaires_france
import convertdate
from typing import Optional, Dict


class SociologicalFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)

    def include_holidays(self):
        self.logger.info("On s'occupe des variables de vacances")
        self.logger.info("On récupère la liste des jours fériés")
        jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(self.features.date_entree.min().year,self.features.date_entree.max().year+1)],[])
        self.logger.info("On l'intègre au dataframe")
        # print(type(jours_feries[0]))
        # print(self.features['date_entree'].dtype)
        self.features['bankHolidays'] = self.features['date_entree'].apply(lambda x: 1 if x.date() in jours_feries else 0)
        # print(self.features.loc[self.features['bankHolidays'] == 1])
        veille_jours_feries = sum([[l-dt.timedelta(days=1) for l in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(self.features.date_entree.min().year,self.features.date_entree.max().year+1)],[])
        self.features['eveBankHolidays'] = self.features['date_entree'].apply(lambda x: 1 if x.date() in veille_jours_feries else 0)
        
        self.logger.info("On s'occupe des vacances en tant que tel")
        def get_academic_zone(name, date):
            dict_zones = {
                'Aix-Marseille': ('B', 'B'),
                'Amiens': ('B', 'B'),
                'Besançon': ('B', 'A'),
                'Bordeaux': ('C', 'A'),
                'Caen': ('A', 'B'),
                'Clermont-Ferrand': ('A', 'A'),
                'Créteil': ('C', 'C'),
                'Dijon': ('B', 'A'),
                'Grenoble': ('A', 'A'),
                'Lille': ('B', 'B'),
                'Limoges': ('B', 'A'),
                'Lyon': ('A', 'A'),
                'Montpellier': ('A', 'C'),
                'Nancy-Metz': ('A', 'B'),
                'Nantes': ('A', 'B'),
                'Nice': ('B', 'B'),
                'Orléans-Tours': ('B', 'B'),
                'Paris': ('C', 'C'),
                'Poitiers': ('B', 'A'),
                'Reims': ('B', 'B'),
                'Rennes': ('A', 'B'),
                'Rouen ': ('B', 'B'),
                'Strasbourg': ('B', 'B'),
                'Toulouse': ('A', 'C'),
                'Versailles': ('C', 'C')
            }
            if date < dt.datetime(2016, 1, 1):
                return dict_zones[name][0]
            return dict_zones[name][1]
        d = vacances_scolaires_france.SchoolHolidayDates()
        self.features['holidays'] = self.features['date_entree'].apply(lambda x: 1 if d.is_holiday_for_zone(x.date(), get_academic_zone(ACADEMIES[DEPARTEMENT], x)) else 0)
        self.features['holidays-1'] = self.features['holidays'].shift(-1)
        self.features['borderHolidays'] = self.features.apply(lambda x: int(x['holidays'] != x['holidays-1']), axis=1)
        self.features.drop('holidays-1', axis=1, inplace=True)
        self.logger.info("Variables de vacances intégrées")

    def include_lockdown(self):
        self.logger.info("On s'occupe des variables de confinement")
        def pendant_couvrefeux(date):
            # Fonction testant is une date tombe dans une période de confinement
            if ((dt.datetime(2020, 12, 15) <= date <= dt.datetime(2021, 1, 2)) 
                and (date.hour >= 20 or date.hour <= 6)):
                return 1
            elif ((dt.datetime(2021, 1, 2) <= date <= dt.datetime(2021, 3, 20))
                and (date.hour >= 18 or date.hour <= 6)):
                    return 1
            elif ((dt.datetime(2021, 3, 20) <= date <= dt.datetime(2021, 5, 19))
                and (date.hour >= 19 or date.hour <= 6)):
                    return 1
            elif ((dt.datetime(2021, 5, 19) <= date <= dt.datetime(2021, 6, 9))
                and (date.hour >= 21 or date.hour <= 6)):
                    return 1
            elif ((dt.datetime(2021, 6, 9) <= date <= dt.datetime(2021, 6, 30))
                and (date.hour >= 23 or date.hour <= 6)):
                    return 1
            return 0
        self.features['confinement1'] = self.features['date_entree'].apply(lambda x: 1 if dt.datetime(2020, 3, 17, 12) <= x <= dt.datetime(2020, 5, 11) else 0)
        self.features['confinement2'] = self.features['date_entree'].apply(lambda x: 1 if dt.datetime(2020, 10, 30) <= x <= dt.datetime(2020, 12, 15) else 0)
        self.features['couvrefeux'] = self.features['date_entree'].apply(pendant_couvrefeux)
        self.logger.info("Variables de confinement intégrées")

    def include_ramadan(self):
        self.logger.info("On s'occupe des variables de Ramadan")
        self.features['ramadan'] = self.features['date_entree'].apply(lambda x: 1 if convertdate.islamic.from_gregorian(x.year, x.month, x.day)[1] == 9 else 0)

        
    def fetch_data(self) -> None:
        """
        Récupère les données.
        
        Parameters:
        - None
        """
        self.include_holidays()
        self.include_lockdown()
        self.include_ramadan()
        self.include_HNFC_moving()

        super().fetch_data()