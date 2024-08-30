import os
import pathlib
import pickle
import datetime as dt

import pandas as pd
import requests
from src.features.base_features import BaseFeature, Config
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from typing import List, Optional, Dict

class EpidemiologicalFeatures(BaseFeature):
    
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)
        # self.predictors_dir = pathlib.Path(self.config.get("predictors_dir"))
        self.region = self.config.get("region")

    def include_sentinelles(self):
        self.logger.info("On s'occupe de l'incidence des maladies d'après Sentinelles")
        # Régions pour Sentinelles
        # TODO: Récupérer tous les autre départements, on choisira au moment de get

        models = {}


        for nom, url in [('grippe', 3), ('diarrhee', 6), ('varicelle', 7), ('ira', 25)]:
            self.logger.info(f"  - on regarde l'incidence de {nom} pour la région {self.region}")
            # On récupère les données de Sentinelles si le fichier n'existe pas
            if not os.path.isfile(self.data_dir / f"{nom}_incidence_{self.region}.pkl"):
                r = requests.get(url=f"https://www.sentiweb.fr/datasets/all/inc-{url}-REG.csv")
                dico = {k.split(',')[0]: int(k.split(',')[2].replace('-', '0')) for k in r.text.split('\n')[2:-1]} #  if self.region in k
                # Save dico
                with open(self.data_dir / f"{nom}_incidence_{self.region}.pkl", 'wb') as f:
                    pickle.dump(dico, f)
            else:
                with open(self.data_dir / f"{nom}_incidence_{self.region}.pkl", 'rb') as f:
                    dico = pickle.load(f)
            
            # Ajouter des semaines supplémentaires au DataFrame features pour prendre en compte le décalage
            # TODO: On veut fetch toute les données, la sélection des dates se fera au moment de get (il faudra alors faire attention à prendre l'historique si on le veut)
            additional_dates = pd.date_range(end=self.data.index.min() - dt.timedelta(**self.step), periods=max(self.config.get('shift'), self.config.get('rolling_window'))*7, freq=dt.timedelta(**self.step))
            additional_dates.set_names('date', inplace=True)
            additional_df = pd.DataFrame(index=additional_dates)
            self.data = pd.concat([self.data, additional_df])

            ##################### TODO: Faire une méthode dans BaseFeature pour ralonger les données, ou faire un Imputer #####################
            self.logger.info(f"Chargement du modèle de prédiction pour {nom}")
            models[nom] = SARIMAXResults.load(self.data_dir / f"predictors/model_{nom}.pkl")
            self.logger.info(f"Modèle {nom} chargé")

            # On prolonge la dernière incidence connue jusqu'à maintenant
            # TODO: Faire une méthode meilleure (XGBoost ?) que Sarimax
            last_week, last_date = max([(k, dt.datetime.strptime(k+'1', "%G%V%u")) for k in dico], key=lambda x: x[1])
            self.logger.info(f"    Pour la dernière date connue ({last_date:'%d/%m/%Y'}, semaine {last_week}), l'incidence était de {dico[last_week]}")
            while last_date < self.data.index.max():
                last_date += dt.timedelta(days=7)
                year, week, _ = last_date.isocalendar()
                sunday = dt.datetime.strptime(f'{year}{week:02d}7', '%G%V%u')
                pred = models[nom].predict(sunday)
                pred = max(0, pred[0])
                dico[f"{year}{week:02d}"] = int(pred)
                self.logger.info(f"    On complète la semaine {year}{week:02d} ({last_date:'%d/%m/%Y'}) par {str(int(pred))}")
            # On fait ensuite dans l'autre sens : on complète vers le passé, par des incidences nulles
            first_week, first_date = min([(k, dt.datetime.strptime(k+'1', "%Y%W%w")) for k in dico], key=lambda x: x[1])
            self.logger.info(f"    La première date connue était {first_date:'%d/%m/%Y'}, semaine {first_week}")
            if first_date > self.data.index.min():
                self.logger.info(f"    Cette première date étant postérieure, à notre historique (commençant le {self.data.index.min():'%d/%m/%Y'}), on complète les incidences par des 0")
                while first_date > self.data.index.min():
                    first_date -= dt.timedelta(days=7)
                    year, week, _ = first_date.isocalendar()
                    sunday = dt.datetime.strptime(f'{year}{week:02d}7', '%G%V%u')
                    # pred = models[nom].predict(sunday)
                    # pred = max(0, pred[0])
                    pred = 0
                    self.logger.info(f"    On complète la semaine {year}{week:02d} ({first_date:'%d/%m/%Y'}) par {str(int(pred))}")
                    dico[f"{year}{week:02d}"] = int(pred)
            ###################################################################################################################################

            self.data[f"inc_{nom}"] = self.data.index.map(lambda x: dico[f"{x.isocalendar().year}{x.isocalendar().week:02d}"])

        self.logger.info("Données sentinelles intégralement récupérées")

    def fetch_data(self) -> None:
        self.include_sentinelles()
        super().fetch_data()

    def get_data(self, from_date: str | dt.datetime | None = None, to_date: str | dt.datetime | None = None, features_names: List[str] | None = None, region: str | None = None) -> pd.DataFrame:
        data = super().get_data(from_date, to_date, features_names)
        # if region is None:
        #     region = self.region
        # data = data.loc[data['region'] == region]
        return data
    