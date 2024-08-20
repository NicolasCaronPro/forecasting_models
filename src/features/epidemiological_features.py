import os
import pathlib
import pickle
import datetime as dt

import pandas as pd
import requests
from src.features.base_features import BaseFeature, Config
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from typing import Optional, Dict

class EpidemiologicalFeatures(BaseFeature):
    
    def __init__(self, config: Optional['Config'] = None, parent: Optional['BaseFeature'] = None) -> None:
        super().__init__(config, parent)
        self.predictors_dir = pathlib.Path(self.config.get("predictors_dir"))
        self.regions = self.config.get("regions")

    def include_sentinelles(self):
        self.logger.info("On s'occupe de l'incidence des maladies d'après Sentinelles")
        # Régions pour Sentinelles
        # TODO: Ajouter les autres départements?

        models = {}


                # Ajouter des semaines supplémentaires au DataFrame features pour prendre en compte le décalage
        for nom, url in [('grippe', 3), ('diarrhee', 6), ('varicelle', 7), ('ira', 25)]:
            self.logger.info(f"  - on regarde l'incidence de {nom} pour la région {region}")
            # On récupère les données de Sentinelles si le fichier n'existe pas
            if not os.path.isfile(self.data_dir / f"{nom}_incidence_{region}.pkl"):
                r = requests.get(url=f"https://www.sentiweb.fr/datasets/all/inc-{url}-REG.csv")
                dico = {k.split(',')[0]: int(k.split(',')[2].replace('-', '0')) for k in r.text.split('\n')[2:-1] if region in k}
                # Save dico
                with open(self.data_dir / f"{nom}_incidence_{region}.pkl", 'wb') as f:
                    pickle.dump(dico, f)
            else:
                with open(self.data_dir / f"{nom}_incidence_{region}.pkl", 'rb') as f:
                    dico = pickle.load(f)
            
            additional_dates = pd.date_range(end=self.data_dir['date_entree'].min() - dt.timedelta(days=1), periods=max(DECALAGE_SENTINELLES, FENETRE_GLISSANTE)*7, freq='D')
            self.data_dir = pd.concat([pd.DataFrame({'date_entree': additional_dates}), self.data_dir]).reset_index(drop=True)

            self.logger.info(f"Chargement du modèle de prédiction pour {nom}")
            models[nom] = SARIMAXResults.load(self.predictors_dir / f"model_{nom}.pkl")
            self.logger.info(f"Modèle {nom} chargé")

            # On prolonge la dernière incidence connue jusqu'à maintenant
            # TODO: Faire une méthode meilleure (XGBoost ?) que Sarimax
            last_week, last_date = max([(k, dt.datetime.strptime(k+'1', "%G%V%u")) for k in dico], key=lambda x: x[1])
            self.logger.info(f"    Pour la dernière date connue ({last_date:'%d/%m/%Y'}, semaine {last_week}), l'incidence était de {dico[last_week]}")
            while last_date < self.data_dir.date_entree.max():
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
            if first_date > self.data_dir.date_entree.min():
                self.logger.info(f"    Cette première date étant postérieure, à notre historique (commençant le {self.data_dir.date_entree.min():'%d/%m/%Y'}), on complète les incidences par des 0")
                while first_date > self.data_dir.date_entree.min():
                    first_date -= dt.timedelta(days=7)
                    year, week, _ = first_date.isocalendar()
                    sunday = dt.datetime.strptime(f'{year}{week:02d}7', '%G%V%u')
                    pred = models[nom].predict(sunday)
                    pred = max(0, pred[0])
                    # pred = 0
                    self.logger.info(f"    On complète la semaine {year}{week:02d} ({first_date:'%d/%m/%Y'}) par {str(int(pred))}")
                    dico[f"{year}{week:02d}"] = int(pred)

            self.data_dir[f"inc_{nom}"] = self.data_dir['date_entree'].apply(lambda x: dico[f"{x.isocalendar().year}{x.isocalendar().week:02d}"])

        self.logger.info("Données sentinelles intégralement récupérées")

    def fetch_data(self) -> None:
        self.include_sentinelles()
        super().fetch_data()