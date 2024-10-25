import os
from pathlib import Path
import pickle
import datetime as dt

import pandas as pd
import requests
from src.features.base_features import BaseFeature
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from typing import List, Optional, Dict
from src.location.location import Location



class EpidemiologicalFeatures(BaseFeature):

    def __init__(self, name:str = None, logger=None) -> None:
        super().__init__(name, logger)
        # self.predictors_dir = pathlib.Path(self.config.get("predictors_dir"))


    def include_sentinelles(self, region, feature_dir, date_range):
        self.logger.info(
            "On s'occupe de l'incidence des maladies d'après Sentinelles")
        # Régions pour Sentinelles
        # TODO: Récupérer tous les autre départements, on choisira au moment de get

        models = {}

        data = pd.DataFrame(index=date_range)

        feature_dir = Path(feature_dir)

        for nom, url in [('grippe', 3), ('diarrhee', 6), ('varicelle', 7), ('ira', 25)]:
            self.logger.info(
                f"  - on regarde l'incidence de {nom} pour la région {region}")
            # On récupère les données de Sentinelles si le fichier n'existe pas
            if not os.path.isfile(feature_dir / f"{nom}_incidence_{region}.pkl"):
                r = requests.get(
                    url=f"https://www.sentiweb.fr/datasets/all/inc-{url}-REG.csv")
                dico = {k.split(',')[0]: int(k.split(',')[2].replace(
                    '-', '0')) for k in r.text.split('\n')[2:-1] if region in k}
                # Save dico
                with open(feature_dir / f"{nom}_incidence_{region}.pkl", 'wb') as f:
                    pickle.dump(dico, f)
            else:
                with open(feature_dir / f"{nom}_incidence_{region}.pkl", 'rb') as f:
                    dico = pickle.load(f)

            # Ajouter des semaines supplémentaires au DataFrame features pour prendre en compte le décalage
            # TODO: On veut fetch toute les données, la sélection des dates se fera au moment de get (il faudra alors faire attention à prendre l'historique si on le veut)
            # additional_dates = pd.date_range(end=self.data.index.min() - dt.timedelta(**self.step), periods=max(
            #     self.config.get('shift'), self.config.get('rolling_window'))*7, freq=dt.timedelta(**self.step))
            # additional_dates.set_names('date', inplace=True)
            # additional_df = pd.DataFrame(index=additional_dates)
            # self.data = pd.concat([self.data, additional_df])


            ##################### TODO: Faire une méthode dans BaseFeature pour ralonger les données, ou faire un Imputer #####################
            # self.logger.info(f"Chargement du modèle de prédiction pour {nom}")
            # models[nom] = SARIMAXResults.load(
            #     feature_dir / f"predictors/model_{nom}.pkl")
            # self.logger.info(f"Modèle {nom} chargé")

            # On prolonge la dernière incidence connue jusqu'à maintenant
            # TODO: Faire une méthode meilleure (XGBoost ?) que Sarimax
            last_week, last_date = max(
                [(k, dt.datetime.strptime(k+'1', "%G%V%u")) for k in dico], key=lambda x: x[1])
            self.logger.info(
                f"    Pour la dernière date connue ({last_date:'%d/%m/%Y'}, semaine {last_week}), l'incidence était de {dico[last_week]}")
            while last_date < data.index.max():
                last_date += dt.timedelta(days=7)
                year, week, _ = last_date.isocalendar()
                sunday = dt.datetime.strptime(f'{year}{week:02d}7', '%G%V%u')
                # pred = models[nom].predict(sunday)
                pred = 0 #max(0, pred[0])
                dico[f"{year}{week:02d}"] = int(pred)
                self.logger.info(
                    f"    On complète la semaine {year}{week:02d} ({last_date:'%d/%m/%Y'}) par {str(int(pred))}")
            # On fait ensuite dans l'autre sens : on complète vers le passé, par des incidences nulles
            first_week, first_date = min(
                [(k, dt.datetime.strptime(k+'1', "%Y%W%w")) for k in dico], key=lambda x: x[1])
            self.logger.info(
                f"    La première date connue était {first_date:'%d/%m/%Y'}, semaine {first_week}")
            if first_date > data.index.min():
                self.logger.info(
                    f"    Cette première date étant postérieure, à notre historique (commençant le {data.index.min():'%d/%m/%Y'}), on complète les incidences par des 0")
                while first_date > data.index.min():
                    first_date -= dt.timedelta(days=7)
                    year, week, _ = first_date.isocalendar()
                    sunday = dt.datetime.strptime(
                        f'{year}{week:02d}7', '%G%V%u')

                    # pred = models[nom].predict(start=sunday, end=sunday)
                    # pred = max(0, pred.iloc[0])
                    pred = 0
                    self.logger.info(
                        f"    On complète la semaine {year}{week:02d} ({first_date:'%d/%m/%Y'}) par {str(int(pred))}")
                    dico[f"{year}{week:02d}"] = int(pred)
            ###################################################################################################################################

            data[f"inc_{nom}"] = data.index.map(
                lambda x: dico[f"{x.isocalendar().year}{x.isocalendar().week:02d}"])
            data[f'inc_{nom}'].to_csv(feature_dir / f"{nom}_incidence_{region}.csv")

        self.logger.info("Données sentinelles intégralement récupérées")
        return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'location' in kwargs, f"Le paramètre'location' est obligatoire pour fetch la feature {self.name}"
        assert 'feature_dir' in kwargs, f"Le paramètre'feature_dir' est obligatoire pour fetch la feature {self.name}"
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        
        location = kwargs.get('location')
        region = location.region_old
        feature_dir = kwargs.get("feature_dir")
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date") # TODO: do not hardcode freq
        return self.include_sentinelles(region, feature_dir, date_range)