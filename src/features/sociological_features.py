from src.features.base_features import BaseFeature, Config
import jours_feries_france
import vacances_scolaires_france
import convertdate
from typing import Optional, Dict
import datetime as dt


class SociologicalFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)
        self.academies = {
        'Aix-Marseille': ['04', '05', '13', '84'],
        'Amiens': ['02', '60', '80'],
        'Besançon': ['25', '39', '70', '90'],
        'Bordeaux': ['24', '33', '40', '47', '64'],
        'Caen': ['14', '50', '61'],
        'Clermont-Ferrand': ['03', '15', '43', '63'],
        'Corse': ['2A', '2B'],
        'Créteil': ['77', '93', '94'],
        'Dijon': ['21', '58', '71', '89'],
        'Grenoble': ['07', '26', '38', '73', '74'],
        'Lille': ['59', '62'],
        'Limoges': ['19', '23', '87'],
        'Lyon': ['01', '42', '69'],
        'Montpellier': ['11', '30', '34', '48', '66'],
        'Nancy-Metz': ['54', '55', '57', '88'],
        'Nantes': ['44', '49', '53', '72', '85'],
        'Nice': ['06', '83'],
        'Orléans-Tours': ['18', '28', '36', '37', '41', '45'],
        'Paris': ['75'],
        'Poitiers': ['16', '17', '79', '86'],
        'Reims': ['08', '10', '51', '52'],
        'Rennes': ['22', '29', '35', '56'],
        'Rouen ': ['27', '76'],
        'Strasbourg': ['67', '68'],
        'Toulouse': ['09', '12', '31', '32', '46', '65', '81', '82'],
        'Versailles': ['78', '91', '92', '95']
    }

    def include_holidays(self):
        self.logger.info("On s'occupe des variables de vacances")
        self.logger.info("On récupère la liste des jours fériés")
        jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(self.data.index.min().year,self.data.index.max().year+1)],[])
        self.logger.info("On l'intègre au dataframe")
        # print(type(jours_feries[0]))
        # print(self.data['date'].dtype)
        self.data['bankHolidays'] = self.data.index.map(lambda x: 1 if x.date() in jours_feries else 0).astype('category')
        # print(self.data.loc[self.data['bankHolidays'] == 1])
        veille_jours_feries = sum([[l-dt.timedelta(days=1) for l in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(self.data.index.min().year,self.data.index.max().year+1)],[])
        self.data['eveBankHolidays'] = self.data.index.map(lambda x: 1 if x.date() in veille_jours_feries else 0).astype('category')
        
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
        academie = [k for k in self.academies if self.config.get('departement') in self.academies[k]][0]
        print(academie)
        # print(academie[int(self.config.get('departement'))])
        self.data['holidays'] = self.data.index.map(lambda x: 1 if d.is_holiday_for_zone(x.date(), get_academic_zone(academie, x)) else 0).astype('category')
        self.data['holidays-1'] = self.data['holidays'].shift(-1)
        self.data['borderHolidays'] = self.data.apply(lambda x: int(x['holidays'] != x['holidays-1']), axis=1).astype('category')
        self.data.drop('holidays-1', axis=1, inplace=True)
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
        self.data['confinement1'] = self.data.index.map(lambda x: 1 if dt.datetime(2020, 3, 17, 12) <= x <= dt.datetime(2020, 5, 11) else 0).astype('category')
        self.data['confinement2'] = self.data.index.map(lambda x: 1 if dt.datetime(2020, 10, 30) <= x <= dt.datetime(2020, 12, 15) else 0).astype('category')
        self.data['couvrefeux'] = self.data.index.map(pendant_couvrefeux).astype('category')
        self.logger.info("Variables de confinement intégrées")

    def include_ramadan(self):
        self.logger.info("On s'occupe des variables de Ramadan")
        self.data['ramadan'] = self.data.index.map(lambda x: 1 if convertdate.islamic.from_gregorian(x.year, x.month, x.day)[1] == 9 else 0).astype('category')

        
    def fetch_data_function(self) -> None:
        """
        Récupère les données.
        
        Parameters:
        - None
        """
        self.include_holidays()
        self.include_lockdown()
        self.include_ramadan()
