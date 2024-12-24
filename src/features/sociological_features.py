from src.features.base_features import BaseFeature
import jours_feries_france
import vacances_scolaires_france
import convertdate
from typing import Optional, Dict
import datetime as dt
import pandas as pd
import numpy as np


class SociologicalFeatures(BaseFeature):
    def __init__(self, name:str = None, logger=None) -> None:
        super().__init__(name, logger)
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

    def include_holidays(self, date_range:pd.DatetimeIndex, departement):
        self.logger.info("On s'occupe des variables de vacances")
        self.logger.info("On récupère la liste des jours fériés")
        data = pd.DataFrame(index=date_range)
        jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values(
        )) for k in range(date_range.min().year, date_range.max().year+1)], [])
        self.logger.info("On l'intègre au dataframe")
        # print(type(jours_feries[0]))
        # print(self.data['date'].dtype)
        data['bankHolidays'] = date_range.map(
            lambda x: 1 if x.date() in jours_feries else 0).astype('boolean')
        # print(self.data.loc[self.data['bankHolidays'] == 1])
        veille_jours_feries = sum([[l-dt.timedelta(days=1) for l in jours_feries_france.JoursFeries.for_year(
            k).values()] for k in range(date_range.min().year, date_range.max().year+1)], [])
        data['eveBankHolidays'] = date_range.map(
            lambda x: 1 if x.date() in veille_jours_feries else 0).astype('boolean')

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
        academie = [k for k in self.academies if departement in self.academies[k]][0]
        # print(academie)
        # print(academie[int(self.config.get('departement'))])
        data['holidays'] = date_range.map(lambda x: 1 if d.is_holiday_for_zone(
            x.date(), get_academic_zone(academie, x)) else 0).astype('boolean')
        data['holidays-1'] = data['holidays'].shift(-1)
        data['borderHolidays'] = data.apply(lambda x: x['holidays'] != x['holidays-1'], axis=1).astype('boolean')
        data['borderHolidays'] = data['borderHolidays'].ffill()
        data.drop('holidays-1', axis=1, inplace=True)
        self.logger.info("Variables de vacances intégrées")
        return data

    def include_lockdown(self, date_range:pd.DatetimeIndex):
        self.logger.info("On s'occupe des variables de confinement")
        data = pd.DataFrame(index=date_range)
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
        data['confinement1'] = date_range.map(lambda x: 1 if dt.datetime(
            2020, 3, 17, 12) <= x <= dt.datetime(2020, 5, 11) else 0).astype('boolean')
        data['confinement2'] = date_range.map(lambda x: 1 if dt.datetime(
            2020, 10, 30) <= x <= dt.datetime(2020, 12, 15) else 0).astype('boolean')
        data['couvrefeux'] = date_range.map(
            pendant_couvrefeux).astype('boolean')
        self.logger.info("Variables de confinement intégrées")
        return data

    def include_ramadan(self, date_range:pd.DatetimeIndex):
        self.logger.info("On s'occupe des variables de Ramadan")
        data = pd.DataFrame(index=date_range)
        data['ramadan'] = date_range.map(lambda x: 1 if convertdate.islamic.from_gregorian(
            x.year, x.month, x.day)[1] == 9 else 0).astype('boolean')
        return data
    
    def include_HNFC_moving(self, date_range:pd.DatetimeIndex):
        self.logger.info("Intégration du déménagement de l'HNFC")
        start = dt.datetime(2017, 2, 28)
        end = dt.datetime(2018, 1, 1)

        df = pd.DataFrame(index=date_range)
        # df.set_index('date', inplace=True)

        df["before_HNFC_moving"] = np.where(df.index < start, 1, 0)
        df['during_HNFC_moving'] = np.where((df.index >= start) & (df.index < end), 1, 0)
        df['after_HNFC_moving'] = np.where(df.index >= end, 1, 0)
        df["before_HNFC_moving"] = df["before_HNFC_moving"].astype("boolean")
        df['during_HNFC_moving'] = df["during_HNFC_moving"].astype("boolean")
        df['after_HNFC_moving'] = df['after_HNFC_moving'].astype("boolean")
        # print(df)
        return df
    
    def include_COVID(self, date_range:pd.DatetimeIndex):
        self.logger.info("Intégration du COVID")
        start = dt.datetime(2020, 2, 1)
        end = dt.datetime(2021, 12, 31)

        df = pd.DataFrame(index=date_range)
        # df.set_index('date', inplace=True)

        df["before_COVID"] = np.where(df.index < start, 1, 0)
        df['during_COVID'] = np.where((df.index >= start) & (df.index < end), 1, 0)
        df['after_COVID'] = np.where(df.index >= end, 1, 0)
        df["before_COVID"] = df["before_COVID"].astype("boolean")
        df['during_COVID'] = df["during_COVID"].astype("boolean")
        df['after_COVID'] = df['after_COVID'].astype("boolean")
        # print(df)
        return df


    def fetch_data_function(self, *args, **kwargs) -> None:
        """
        Récupère les données.

        Parameters:
        - None
        """

        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        assert 'location' in kwargs, "location must be provided in config"
        location = kwargs.get('location')
        departement = location.code_departement
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date") # TODO: do not hardcode freq
        data = pd.DataFrame(index=date_range)
        
        data = data.join(self.include_holidays(date_range, departement))
        data = data.join(self.include_lockdown(date_range))
        data = data.join(self.include_ramadan(date_range))
        data = data.join(self.include_HNFC_moving(date_range))
        data = data.join(self.include_COVID(date_range))

        # Prefix all columns with sociological_
        data.columns = [f"sociological_{col}" for col in data.columns]
        return data
