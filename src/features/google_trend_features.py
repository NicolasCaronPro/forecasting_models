from src.features.base_features import BaseFeature, Config
from typing import Optional, Dict
import pandas as pd
import datetime as dt
from serpapi import GoogleSearch



class GoogleTrendFeatures(BaseFeature):
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)
    
    def include_google_trends(self) -> pd.DataFrame:
        self.logger.info("On récupère les données de Google Trends")
        BATCH_SIZE = 270

        bow = [
            "diarrhée", "vomissements", "toux", "éruption cutanée", "infection urinaire",
            "hopital", "médecin", "pharmacie", "médicament", "vaccin", "maladie", "fièvre",
            "grippe", "rhume", "angine", "otite", "allergie", "asthme",
            "stress", "dépression", "mal de tête",
            "douleur thoracique", "palpitations", "essoufflement", "vertiges", "crampes abdominales",
            "saignements", "douleur abdominale", "hypothermie", "hyperthermie", "appendicite",
            "méningite", "pneumonie", "AVC", "infection respiratoire",
            "gastro-entérite", "infection cutanée", "insuffisance cardiaque",
            "épilepsie", "migraines", "accident de voiture",
            "fracture", "entorse", "brûlure", "empoisonnement", "chute", "noyade", "asphyxie",
            "crise de panique", "schizophrénie", "trouble bipolaire", "démence", "tentative de suicide",
            "urgences", "urgence médicale", "SAMU", "douleur", "SOS médecin", "étourdissements", "paralysie"
        ]
        def date_range(start_date, stop_date, batch_size):
            current_date = start_date
            while current_date <= stop_date:
                yield current_date
                current_date += dt.timedelta(days=batch_size)

        def get_all_dates_df(bow, i, start, end, batch_size, api_keys=["cf7a3d5a2dd56565ef56451a80d9ea4a5540690bc11c12fc84417be5b4e17f27", "9d3432113ee747417e68225f4708971e5b4a2d22b5f9c2edd98f41b44ae1851f", "2db567698593dfdbc27232e2244cd1346cac3c105e2b93f11334db10e8536bde", "17c26a6b6e4f020d1c607063cdb3ac30cf329060339e5ff34c71778e98574742"]):
            print(api_keys)
            api_key = api_keys[0]
            all_dates_df = pd.DataFrame()
            for date in date_range(start, end, batch_size):
                end_date = min(date + dt.timedelta(days=batch_size-1), end)
                group = ', '.join(bow[i:i+5])
                self.logger.info(f"Récupération de {group} pour la période {date} - {end_date}")
                params = {
                    "engine": "google_trends",
                    "q": f"{group}",
                    "geo": "FR-D", # FR-D
                    "date": f"{date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}",
                    "tz": "0",
                    "data_type": "TIMESERIES",
                    "api_key": api_key
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                try:
                    interest_over_time = results["interest_over_time"]["timeline_data"]
                    df = pd.DataFrame(interest_over_time)
                    for query in df.iloc[0]["values"]:
                        query = query["query"]
                        df["trend_" + query] = df["values"].apply(lambda x: next(x[i]["value"] for i in range(len(x)) if x[i]["query"] == query))
                        df["trend_" + query] = df["trend_" + query].apply(lambda x: x.replace('<', '').replace('>', ''))
                        df["trend_" + query] = df["trend_" + query].astype(int)
                    df.drop(columns=["values", "timestamp"], inplace=True)
                    df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")
                    df.drop(columns=["date"], inplace=True)
                    df.set_index("date", inplace=True)
                    all_dates_df = pd.concat([all_dates_df, df], axis=0)
                except:
                    # Ce code permet de réessayer avec un batch plus petit mais cela ne résoud pas le problème, on décide donc de remplir de 0 plus tard
                    # On perd 270 jours à chaque fois mais cette erreur ne se produit généralement que quand il n'y a pas (peu?) de données
                    # if batch_size//2 > 1:
                    #     self.logger.info(f"On réessaye avec un batch plus petit de {batch_size//2} jours")
                    #     try:
                    # self.logger.info(f"On réessaye avec une autre clé API")
                    # api_keys.pop(0)
                    # all_dates_df = pd.concat([all_dates_df, get_all_dates_df(bow, i, date, end_date, batch_size,api_keys=api_keys)], axis=0)
                    #     except:
                    #         self.logger.error(f"Problème avec {group} pour la période {date} - {end_date}")
                    #         continue
                    # else:
                    #     self.logger.error(f"Problème avec {group} pour la période {date} - {end_date}")
                    continue
                    
            return all_dates_df
        # On récupère les tendance par lot de 5 mots max et 270 jours max
        additional_dates = pd.date_range(end=self.data.index.min() - dt.timedelta(**self.step), periods=max(self.config.get('shift'), self.config.get('rolling_window')), freq=dt.timedelta(**self.step))
        additional_dates.set_names('date', inplace=True)
        additional_df = pd.DataFrame(index=additional_dates)
        self.data = pd.concat([self.data, additional_df])
        # self.data = pd.concat([pd.DataFrame({'date_entree': additional_dates}), self.data]).reset_index(drop=True)

        if not (self.data_dir / 'trends.csv').is_file():
            self.logger.info("On récupère les données de Google Trends depuis l'API")
            idx = self.data.index
            all_words_df = pd.DataFrame(index=idx)
            for i in range(0, len(bow), 5):
                
                all_dates_df = get_all_dates_df(bow, i, dt.datetime(2018, 1, 1), dt.datetime(2023, 12, 31) + dt.timedelta(days=1), BATCH_SIZE) # START - dt.timedelta(days=max(DECALAGE_TREND, FENETRE_GLISSANTE)), STOP_DATA
                # Fill missing dates
                self.logger.info(all_dates_df.index)
                all_dates_df = all_dates_df.reset_index()
                all_dates_df.fillna(0, inplace=True)
                # all_dates_df.ffill(inplace=True)
                # all_dates_df.bfill(inplace=True)
                all_dates_df.set_index('date', inplace=True)
                all_words_df = pd.concat([all_words_df, all_dates_df], axis=1)
            all_words_df.reset_index(inplace=True)
            # all_words_df.rename({"index": "date"}, axis=1, inplace=True)
            # # Tracer les graphiques
            # for i in range(0, len(bow), 5):
            #     fig, ax = plt.subplots(figsize=(20, 10))
            #     all_words_df.plot(x="date_entree", y=all_words_df.loc[:, all_words_df.columns != "date_entree"].columns.tolist()[i:i + 5], title="Google Trends in France", ax=ax)
            
            # On sauvegarde le dataframe
            all_words_df.to_csv(self.data_dir / 'trends.csv', index=False)
        else:
            self.logger.info("On charge les données de Google Trends depuis le fichier")
            all_words_df = pd.read_csv(self.data_dir / 'trends.csv', parse_dates=['date'])
            # all_words_df.drop(columns=["trend_tabac", "trend_alcool", "trend_drogue", "trend_diabète", "trend_obésité",
            #                            "trend_cancer", "trend_maladie rénale", "trend_maladie cardiaque", "trend_maladie de Crohn",
            #                            "trend_colite ulcéreuse"], inplace=True)
        
        all_words_df.set_index('date', inplace=True)
        self.data = self.data.merge(all_words_df, left_index=True, right_index=True, how='left')
        del all_words_df

        
        # On ajoute la moyenne glissante sur FENETRE_GLISSANTE joursn et les valeurs shiftées jusqu'à DECALAGE_TREND jours
        # for word in bow:
        #     self.data = features_augmentation(self.data, feature_name=word, feature_category="trend", shift=DECALAGE_TREND, isShifInDays=True)

            # self.data[f"trend_{word}_mean"] = self.data[f"trend_{word}"].rolling(window=FENETRE_GLISSANTE, closed="left").mean()

            # # On ajoute les valeurs shiftées jusqu'à DECALAGE_TREND jours
            # for dec in range(1, DECALAGE_TREND+1):
            #     self.data[f"trend_{word}-{dec}"] = self.data[f"trend_{word}"].shift(dec)

        self.logger.info("Données de Google Trends intégrées")

        return self.data
    
    def fetch_data_function(self) -> None:
        self.include_google_trends()