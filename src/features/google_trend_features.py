from src.features.base_features import BaseFeature
from typing import Optional, Dict, List
import pandas as pd
import datetime as dt
from serpapi import GoogleSearch
from pathlib import Path
from src.location.location import Location

class GoogleTrendFeatures(BaseFeature):
    def __init__(self, name:str = None, logger=None) -> None:
        super().__init__(name, logger)

    def include_google_trends(self, location: Location, date_range: pd.DatetimeIndex, feature_dir: Path) -> pd.DataFrame:
        self.logger.info("On récupère les données de Google Trends")
        INITIAL_BATCH_SIZE = 270
        MIN_BATCH_SIZE = 33

        keys = [
            "cf7a3d5a2dd56565ef56451a80d9ea4a5540690bc11c12fc84417be5b4e17f27",
            "9d3432113ee747417e68225f4708971e5b4a2d22b5f9c2edd98f41b44ae1851f",
            "2db567698593dfdbc27232e2244cd1346cac3c105e2b93f11334db10e8536bde",
            "17c26a6b6e4f020d1c607063cdb3ac30cf329060339e5ff34c71778e98574742"
        ]

        bow = ["diarrhée", "vomissements", "toux", "éruption cutanée", "infection urinaire", "hopital", "médecin", 
            "pharmacie", "médicament", "vaccin", "maladie", "fièvre", "grippe", "rhume", "angine", "otite", 
            "allergie", "asthme", "stress", "dépression", "mal de tête", "douleur thoracique", "palpitations", 
            "essoufflement", "vertiges", "crampes abdominales", "saignements", "douleur abdominale", 
            "hypothermie", "hyperthermie", "appendicite", "méningite", "pneumonie", "AVC", "infection respiratoire", 
            "gastro-entérite", "infection cutanée", "insuffisance cardiaque", "épilepsie", "migraines", 
            "accident de voiture", "fracture", "entorse", "brûlure", "empoisonnement", "chute", "noyade", 
            "asphyxie", "crise de panique", "schizophrénie", "trouble bipolaire", "démence", "tentative de suicide", 
            "urgences", "urgence médicale", "SAMU", "douleur", "SOS médecin", "étourdissements", "paralysie"]

        # Génère des sous-intervalles de dates en fonction du batch size
        def generate_date_batches(start_date, end_date, batch_size):
            current_date = start_date
            while current_date <= end_date:
                yield current_date, min(current_date + dt.timedelta(days=batch_size - 1), end_date)
                current_date += dt.timedelta(days=batch_size)

        # Récupère les données de Google Trends de manière récursive
        def fetch_google_trends_batch(location, keywords: List[str], start, end, api_keys, batch_size):
            all_dates_df = pd.DataFrame()
            for date_start, date_end in generate_date_batches(start, end, batch_size):
                group = ', '.join(keywords)
                api_key_index = 0  # Démarrer avec la première clé API
                success = False  # Indicateur de succès de l'appel API

                while api_key_index < len(api_keys) and batch_size >= MIN_BATCH_SIZE:
                    api_key = api_keys[api_key_index]
                    self.logger.info(f"Récupération de {group} pour {date_start} - {date_end} avec un batch de {batch_size} jours")

                    params = {
                        "engine": "google_trends",
                        "q": group,
                        "geo": location.region_trends,
                        "date": f"{date_start.strftime('%Y-%m-%d')} {date_end.strftime('%Y-%m-%d')}",
                        "tz": "0",
                        "data_type": "TIMESERIES",
                        "api_key": api_key
                    }

                    try:
                        results = GoogleSearch(params).get_dict()
                        interest_over_time = results["interest_over_time"]["timeline_data"]
                        df = pd.DataFrame(interest_over_time)
                        for query in df.iloc[0]["values"]:
                            query = query["query"]
                            df[f"trend_{query}"] = df["values"].apply(
                                lambda x: int(x[0]["value"].replace('<', '').replace('>', '')) if "value" in x[0] else 0
                            )
                        df.drop(columns=["values", "timestamp"], inplace=True)
                        df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")
                        df.set_index("date", inplace=True)
                        filepath = feature_dir / f"temp/{group}.csv"
                        df.to_csv(filepath, mode='a')
                        all_dates_df = pd.concat([all_dates_df, df], axis=0)
                        success = True  # Indique que l'appel a réussi
                        break

                    except Exception as e:
                        self.logger.error("Erreur de récupération pour %s : %s", group, str(e))
                        api_key_index += 1  # Essayer la clé API suivante
                        if api_key_index == len(api_keys):
                            batch_size //= 2  # Réduire la taille du batch en cas d'échec avec toutes les clés API
                            api_key_index = 0  # Réinitialiser l'index pour réessayer avec la première clé API

                # Si l'appel a échoué avec toutes les clés API et que batch_size est inférieur à MIN_BATCH_SIZE, remplir avec des 0
                if not success and batch_size < MIN_BATCH_SIZE:
                    self.logger.warning(f"Remplissage des données de {group} avec des zéros pour la période {date_start} - {date_end}")
                    date_index = pd.date_range(date_start, date_end)
                    zero_df = pd.DataFrame(0, index=date_index, columns=[f"trend_{keyword}" for keyword in keywords])
                    all_dates_df = pd.concat([all_dates_df, zero_df], axis=0)

            return all_dates_df

        # Code principal pour le chargement et la récupération des données
        feature_file = feature_dir / f'trends_{location.region_old}.csv'
        data_has_changed = False

        if feature_file.is_file():
            self.logger.info("Chargement des données de Google Trends depuis le fichier existant")
            data = pd.read_csv(feature_file, parse_dates=["date"], index_col="date")
            missing_start = date_range.min() < data.index.min()
            missing_end = date_range.max() > data.index.max()

            if missing_start or missing_end:
                self.logger.info("Mise à jour des données manquantes")
                all_words_start_dfs = []
                all_words_end_dfs = []
                for i in range(0, len(bow), 5):
                    keywords = bow[i:i + 5]
                    if missing_start:
                        batch = fetch_google_trends_batch(location, keywords, date_range.min(), data.index.min() - dt.timedelta(days=1), api_keys=keys, batch_size=INITIAL_BATCH_SIZE)
                        all_words_start_dfs.append(batch)
                    if missing_end:
                        batch = fetch_google_trends_batch(location, keywords, data.index.max() + dt.timedelta(days=1), date_range.max(), api_keys=keys, batch_size=INITIAL_BATCH_SIZE)
                        all_words_end_dfs.append(batch)

                start_df = pd.concat(all_words_start_dfs, axis=1)
                end_df = pd.concat(all_words_end_dfs, axis=1)
                data = pd.concat([start_df, data, end_df], axis=0)
                data_has_changed = True
        else:
            self.logger.info("Création des données de Google Trends depuis l'API")
            data = pd.DataFrame(index=date_range)
            all_words_dfs = []
            for i in range(0, len(bow), 5):
                keywords = bow[i:i + 5]
                batch = fetch_google_trends_batch(location, keywords, date_range.min(), date_range.max(), api_keys=keys, batch_size=INITIAL_BATCH_SIZE)
                all_words_dfs.append(batch)
                data_has_changed = True
            data = pd.concat([data] + all_words_dfs, axis=1)

        if data_has_changed:
            data.to_csv(feature_file)

        return data

    # def include_google_trends(self, location: Location, date_range:pd.DatetimeIndex, feature_dir) -> pd.DataFrame:
    #     self.logger.info("On récupère les données de Google Trends")
    #     BATCH_SIZE = 270

    #     feature_dir = Path(feature_dir)

    #     bow = [
    #         "diarrhée", "vomissements", "toux", "éruption cutanée", "infection urinaire",
    #         "hopital", "médecin", "pharmacie", "médicament", "vaccin", "maladie", "fièvre",
    #         "grippe", "rhume", "angine", "otite", "allergie", "asthme",
    #         "stress", "dépression", "mal de tête",
    #         "douleur thoracique", "palpitations", "essoufflement", "vertiges", "crampes abdominales",
    #         "saignements", "douleur abdominale", "hypothermie", "hyperthermie", "appendicite",
    #         "méningite", "pneumonie", "AVC", "infection respiratoire",
    #         "gastro-entérite", "infection cutanée", "insuffisance cardiaque",
    #         "épilepsie", "migraines", "accident de voiture",
    #         "fracture", "entorse", "brûlure", "empoisonnement", "chute", "noyade", "asphyxie",
    #         "crise de panique", "schizophrénie", "trouble bipolaire", "démence", "tentative de suicide",
    #         "urgences", "urgence médicale", "SAMU", "douleur", "SOS médecin", "étourdissements", "paralysie"
    #     ]

    #     def range_date(start_date, stop_date, batch_size):
    #         current_date = start_date
    #         while current_date <= stop_date:
    #             yield current_date
    #             current_date += dt.timedelta(days=batch_size)

    #     def get_all_dates_df(location, bow, i, start, end, batch_size, api_keys=["cf7a3d5a2dd56565ef56451a80d9ea4a5540690bc11c12fc84417be5b4e17f27", "9d3432113ee747417e68225f4708971e5b4a2d22b5f9c2edd98f41b44ae1851f", "2db567698593dfdbc27232e2244cd1346cac3c105e2b93f11334db10e8536bde", "17c26a6b6e4f020d1c607063cdb3ac30cf329060339e5ff34c71778e98574742"]):
    #         api_key = api_keys[1]
    #         all_dates_df = pd.DataFrame()
    #         for date in range_date(start, end, batch_size):
    #             end_date = min(date + dt.timedelta(days=batch_size-1), end)
    #             group = ', '.join(bow[i:i+5])
    #             self.logger.info(
    #                 f"Récupération de {group} pour la période {date} - {end_date}")
    #             params = {
    #                 "engine": "google_trends",
    #                 "q": f"{group}",
    #                 "geo": location.region_trends,  # FR-D
    #                 "date": f"{date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}",
    #                 "tz": "0",
    #                 "data_type": "TIMESERIES",
    #                 "api_key": api_key
    #             }
    #             search = GoogleSearch(params)
    #             results = search.get_dict()
    #             print(results)
    #             # try:
    #             interest_over_time = results["interest_over_time"]["timeline_data"]
    #             df = pd.DataFrame(interest_over_time)
    #             #print(df)
    #             for query in df.iloc[0]["values"]:
    #                 query = query["query"]
    #                 df["trend_" + query] = df["values"].apply(lambda x: next(
    #                     x[i]["value"] for i in range(len(x)) if x[i]["query"] == query))
    #                 df["trend_" + query] = df["trend_" +
    #                                             query].apply(lambda x: x.replace('<', '').replace('>', ''))
    #                 df["trend_" + query] = df["trend_" + query].astype(int)
    #             df.drop(columns=["values", "timestamp"], inplace=True)
    #             df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")
    #             #df.drop(columns=["date"], inplace=True)
    #             df.set_index("date", inplace=True) #TODO: check if this is necessary
    #             all_dates_df = pd.concat([all_dates_df, df], axis=0)
    #             # except:
    #                 # Ce code permet de réessayer avec un batch plus petit mais cela ne résoud pas le problème, on décide donc de remplir de 0 plus tard
    #                 # On perd 270 jours à chaque fois mais cette erreur ne se produit généralement que quand il n'y a pas (peu?) de données
    #                 # if batch_size//2 > 1:
    #                 #     self.logger.info(f"On réessaye avec un batch plus petit de {batch_size//2} jours")
    #                 #     try:
    #                 # self.logger.info(f"On réessaye avec une autre clé API")
    #                 # api_keys.pop(0)
    #                 # all_dates_df = pd.concat([all_dates_df, get_all_dates_df(bow, i, date, end_date, batch_size,api_keys=api_keys)], axis=0)
    #                 #     except:
    #                 #         self.logger.error(f"Problème avec {group} pour la période {date} - {end_date}")
    #                 #         continue
    #                 # else:
    #                 #     self.logger.error(f"Problème avec {group} pour la période {date} - {end_date}")
    #                 # continue

    #         return all_dates_df
    #     # On récupère les tendance par lot de 5 mots max et 270 jours max
    #     # additional_dates = pd.date_range(end=self.data.index.min() - dt.timedelta(**self.step), periods=max(
    #     #     self.config.get('shift'), self.config.get('rolling_window')), freq=dt.timedelta(**self.step))
    #     # additional_dates.set_names('date', inplace=True)
    #     # additional_df = pd.DataFrame(index=additional_dates)
    #     # self.data = pd.concat([self.data, additional_df])
    #     # self.data = pd.concat([pd.DataFrame({'date_entree': additional_dates}), self.data]).reset_index(drop=True)
    #     file = 'trends_' + location.region_old + '.csv'
    #     if not (feature_dir / file).is_file():
    #         self.logger.info(
    #             "On récupère les données de Google Trends depuis l'API")
    #         # idx = self.data.index
    #         all_words_df = pd.DataFrame(index=date_range)
    #         for i in range(0, len(bow), 5):

    #             all_dates_df = get_all_dates_df(location, bow, i, date_range.min(), date_range.max(), BATCH_SIZE)  # START - dt.timedelta(days=max(DECALAGE_TREND, FENETRE_GLISSANTE)), STOP_DATA
    #             # Fill missing dates
    #             # self.logger.info(all_dates_df.index)
    #             all_dates_df = all_dates_df.reset_index()
    #             all_dates_df.fillna(0, inplace=True)
    #             # all_dates_df.ffill(inplace=True)
    #             # all_dates_df.bfill(inplace=True)
    #             all_dates_df.set_index('date', inplace=True)
    #             all_words_df = pd.concat([all_words_df, all_dates_df], axis=1)
    #         all_words_df.reset_index(inplace=True)
            
    #         # On sauvegarde le dataframe
    #         all_words_df.to_csv(feature_dir / file, index=False)
    #     else:
    #         self.logger.info(
    #             "On charge les données de Google Trends depuis le fichier")
    #         all_words_df = pd.read_csv(feature_dir / file)
    #         all_words_df['date'] = pd.to_datetime(all_words_df['date'])
    #         # all_words_df.drop(columns=["trend_tabac", "trend_alcool", "trend_drogue", "trend_diabète", "trend_obésité",
    #         #                            "trend_cancer", "trend_maladie rénale", "trend_maladie cardiaque", "trend_maladie de Crohn",
    #         #                            "trend_colite ulcéreuse"], inplace=True)
    #         if all_words_df["date"].min() > date_range.min():
    #             self.logger.info("On update les données de Google Trends depuis l'API")
    #             # idx = self.data.index
    #             for i in range(0, len(bow), 5):

    #                 all_dates_df = get_all_dates_df(location, bow, i, date_range.min(), all_words_df.min(), BATCH_SIZE)  # START - dt.timedelta(days=max(DECALAGE_TREND, FENETRE_GLISSANTE)), STOP_DATA
    #                 # Fill missing dates
    #                 # self.logger.info(all_dates_df.index)
    #                 all_dates_df = all_dates_df.reset_index()
    #                 all_dates_df.fillna(0, inplace=True)
    #                 # all_dates_df.ffill(inplace=True)
    #                 # all_dates_df.bfill(inplace=True)
    #                 all_dates_df.set_index('date', inplace=True)
    #                 all_words_df = pd.concat([all_words_df, all_dates_df], axis=0)
    #         if all_words_df.max() < date_range.max():
    #             self.logger.info("On update les données de Google Trends depuis l'API")
    #             # idx = self.data.index
    #             for i in range(0, len(bow), 5):

    #                 all_dates_df = get_all_dates_df(location, bow, i, all_words_df.max(), date_range.max(), BATCH_SIZE)  # START - dt.timedelta(days=max(DECALAGE_TREND, FENETRE_GLISSANTE)), STOP_DATA
    #                 # Fill missing dates
    #                 # self.logger.info(all_dates_df.index)
    #                 all_dates_df = all_dates_df.reset_index()
    #                 all_dates_df.fillna(0, inplace=True)
    #                 # all_dates_df.ffill(inplace=True)
    #                 # all_dates_df.bfill(inplace=True)
    #                 all_dates_df.set_index('date', inplace=True)
    #                 all_words_df = pd.concat([all_words_df, all_dates_df], axis=0)
            
    #         all_words_df.reset_index(inplace=True)
            
    #         # On sauvegarde le dataframe
    #         all_words_df.to_csv(feature_dir / file, index=False)
            



    #     data = all_words_df.set_index('date')
        
    #     del all_words_df

    #     # On ajoute la moyenne glissante sur FENETRE_GLISSANTE joursn et les valeurs shiftées jusqu'à DECALAGE_TREND jours
    #     # for word in bow:
    #     #     self.data = features_augmentation(self.data, feature_name=word, feature_category="trend", shift=DECALAGE_TREND, isShifInDays=True)

    #     # self.data[f"trend_{word}_mean"] = self.data[f"trend_{word}"].rolling(window=FENETRE_GLISSANTE, closed="left").mean()

    #     # # On ajoute les valeurs shiftées jusqu'à DECALAGE_TREND jours
    #     # for dec in range(1, DECALAGE_TREND+1):
    #     #     self.data[f"trend_{word}-{dec}"] = self.data[f"trend_{word}"].shift(dec)

    #     self.logger.info("Données de Google Trends intégrées")

    #     return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'feature_dir' in kwargs, f"Le paramètre'feature_dir' est obligatoire pour fetch la feature {self.name}"
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        assert 'location' in kwargs, f"Le paramètre'location' est obligatoire pour fetch la feature {self.name}"

        location = kwargs.get("location")
        feature_dir = kwargs.get("feature_dir")
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date") # TODO: do not hardcode freq
        #data = pd.DataFrame(index=date_range)
        #data.join(self.include_google_trends(date_range, feature_dir))
        return self.include_google_trends(location, date_range, feature_dir)
