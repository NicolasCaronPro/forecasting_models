import datetime as dt
import numpy as np
import pandas as pd
from src.features.base_features import BaseFeature
from src.location.location import Location
import json
import requests


class PopulationFeatures(BaseFeature):
    def __init__(self, name: str = None, logger=None) -> None:
        super().__init__(name, logger)

    def include_population(self, from_date, to_date, location) -> pd.DataFrame:
        """
        Adds the target to the features.
        """
        self.logger.info("Intégration de la target")

        population = pd.DataFrame()

        # JSON data
        dpt = f'DEP-{location.code_departement}'
        years = ['2019', '2020', '2021', '2022', '2023', '2024']

        requete_dataset = f"https://api.insee.fr/melodi/data/DS_ESTIMATION_POPULATION?GEO={dpt}"

        for year in years:
            requete_dataset += f"&TIME_PERIOD={year}"

        get_data = requests.get(requete_dataset, verify=False)
        data_from_net = get_data.content
        data = json.loads(data_from_net)

        # save data in file
        # with open('data.json', 'w') as f:
        #     json.dump(data, f)

        # Extraire observations
        observations = data['observations']

        # Liste pour les observations
        extracted_data = []

        # Boucle de lecture des observations dans le json
        for obs in observations:
            dimensions = obs['dimensions']
            attributes = obs['attributes']
            measures = obs['measures']['OBS_VALUE_NIVEAU']['value']

            # Combiner dimensions, attributes, et measures dans un dict
            combined_data = {**dimensions, **attributes,
                             'OBS_VALUE_NIVEAU': measures}

            # Ajouter à la liste
            extracted_data.append(combined_data)

        # Création d'un dataframe python
        df = pd.DataFrame(extracted_data)

        # drop rows with EP_MEASURE != 'POP'
        df = df[df['EP_MEASURE'] == 'POP']
        df = df[df['SEX'] == '_T']

        df.drop(columns=['EP_MEASURE', 'SEX',
                'OBS_STATUS_FR', 'GEO'], inplace=True)

        df_pivot = df.pivot(index='TIME_PERIOD',
                            columns='AGE', values='OBS_VALUE_NIVEAU')

        # Step 1: Convert TIME_PERIOD to datetime format, assuming TIME_PERIOD is the year
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y')

        # Step 2: Create a full date range for each day of the year
        # For example, for the year 2021:
        date_range = pd.date_range(
            start='2019-01-01', end='2024-12-31', freq='D')

        # Step 3: Reindex the DataFrame to include each day, assuming df_pivot is your pivoted DataFrame
        df_pivot = df.pivot(index='TIME_PERIOD',
                            columns='AGE', values='OBS_VALUE_NIVEAU')

        # Reindexing the DataFrame to include the complete date range (using df_pivot from earlier)
        df_pivot = df_pivot.reindex(date_range)
        df_pivot = df_pivot.rename_axis('date')

        df_pivot = df_pivot.interpolate(method='linear')

        # Step 4: Fill missing values if necessary (e.g., with forward-fill or NaNs)
        # You can also use 'bfill' or leave as NaN
        # df_pivot.fillna(method='ffill', inplace=True)
        df_pivot.ffill(inplace=True)

        df_pivot = df_pivot.rename_axis(None, axis=1)
        df_pivot = df_pivot.astype(int)
        df_pivot = df_pivot.loc[from_date:to_date]

        population = df_pivot.copy(deep=True)

        return population

    def fetch_data_function(self, *args, **kwargs) -> pd.DataFrame:
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        assert 'location' in kwargs, f"Le paramètre'location' est obligatoire pour fetch la feature {self.name}"

        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        location = kwargs.get("location")

        return self.include_population(start_date, stop_date, location)
