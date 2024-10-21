from src.features.base_features import BaseFeature
from typing import Optional, Dict
import pandas as pd
import datetime as dt
import meteostat
from shapely import wkt
import pathlib
import os


class MeteorologicalFeatures(BaseFeature):
    def __init__(self, name:str = None, logger=None) -> None:
        super().__init__(name, logger)

    def include_weather(self, etablissement, date_range:pd.DatetimeIndex, feature_dir:pathlib.Path):
        self.logger.info("On récupère les archives Meteostat")

        # TODO: à changer !!

        GEO_ETAB_DICT = {'CH Chaumont': 'CH CHAUMONT',
                         'CH privé Dijon': 'CH privé DIJON',
                         'CHU Besançon': 'CHU BESANCON',
                         'CH Semur': 'CH SEMUR',
                         'CH Chatillon Montbard': 'CH CHATILLON MONTBARD',
                         'CHU Dijon': 'CHU DIJON',
                         'CH Beaune': 'CH BEAUNE',
                         'HNFC': 'HNFC',
                         'CH Langres': 'CH LANGRES'}
        self.GEO_ETAB_LIST = list(GEO_ETAB_DICT[etab] for etab in [etablissement])
        
        print(os.getcwd())
        dir_geo = feature_dir.parent.parent.parent / 'src/geolocalisation'

        print(dir_geo)

        # On charge les données de géolocalisation des établissements
        etablissement_df = pd.read_csv(
            dir_geo / "etab_coord.csv", index_col='etablissement')
        for c in etablissement_df.columns:
            if c == 'selected_centroid':
                etablissement_df[c] = etablissement_df[c].apply(
                    lambda x: [wkt.loads(y[2:-2]) for y in x[2:-2].split(", ")])
            else:
                etablissement_df[c] = etablissement_df[c].apply(wkt.loads)

        # On ne garde que les établissements qui nous intéressent
        etablissement_df = etablissement_df[etablissement_df.index.isin(
            self.GEO_ETAB_LIST)]

        # Pour ne garder que 1 centroid sélectionné
        etablissement_df["selected_centroid"] = etablissement_df["selected_centroid"].apply(
            lambda x: [x[0]])

        data_df = pd.DataFrame(index=date_range)
        # colors = ['green', 'purple', 'blue', 'orange', 'yellow', 'pink', 'brown', 'grey']
        for i, etab in enumerate(etablissement_df.index):
            data, data24h, liste = {}, {}, []
            last_point = None
            meteostat.Point.radius = 60000
            # meteostat.Point.method = "weighted"
            meteostat.Point.alt_range = 1000
            meteostat.Point.max_count = 3
            # # Affichage des coordonné de l'établissement
            # fig, ax = plt.subplots()
            # gdf2 = gpd.GeoDataFrame(geometry=gpd.points_from_xy([etablissement_df.loc[etab, 'position'].x], [etablissement_df.loc[etab, 'position'].y]))
            # gdf2.crs = "EPSG:4326"
            # gdf2 = gdf2.to_crs("EPSG:3857")
            # gdf2.plot(ax=ax, zorder=2, marker='o', color='black', markersize=10)
            for index, pt in enumerate(etablissement_df.loc[etab, 'selected_centroid']):
                point = (pt.y, pt.x)
                location = meteostat.Point(point[0], point[1])
                # # Affichage des centroids sélectionnés
                # gdf2 = gpd.GeoDataFrame(geometry=gpd.points_from_xy([point[1]], [point[0]]))
                # gdf2.crs = "EPSG:4326"
                # gdf2 = gdf2.to_crs("EPSG:3857")
                # gdf2.plot(ax=ax, zorder=2, marker='o', color='red', markersize=7)
                # # Affichage des stations météo les plus proches
                # stations = meteostat.Stations()
                # stations = stations.nearby(point[0], point[1])
                # stations = stations.fetch(5)
                # gdf = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.longitude, stations.latitude))
                # gdf.crs = "EPSG:4326"
                # gdf = gdf.to_crs("EPSG:3857")
                # gdf.plot(ax=ax, markersize=5, marker='o', zorder=1, color=colors[i])
                # if len(stations) == 0:
                #     self.logger.error(f"Aucune station trouvée pour le point {point}")
                #     break
                # else:
                #     self.logger.info(f"Pour le point {point}, on a trouvé {len(stations)} stations")

                # try:
                # print(START - dt.timedelta(days=DECALAGE_METEO))
                data[point] = meteostat.Daily(
                    location, date_range.min(), date_range.max())
                # on comble les heures manquantes (index) dans les données collectées
                data[point] = data[point].normalize()
                # On complète les Nan, quand il n'y en a pas plus de 3 consécutifs
                data[point] = data[point].interpolate()
                data[point] = data[point].fetch()
                assert len(data[point]) > 0
                data[point]['snow'].fillna(0, inplace=True)
                data[point].drop(['tsun', 'coco', 'wpgt'],
                                 axis=1, inplace=True, errors='ignore')
                data[point].ffill(inplace=True)
                data[point].bfill(inplace=True)
                # for k in sorted(data[point].columns):
                #     if data[point][k].isna().sum() > self.max_nan:
                #         self.logger.error(
                #             f"{k} du point {point} possède trop de NaN ({data[point][k].isna().sum()}")
                #         assert False
                data[point].fillna(0, inplace=True)
                data[point].reset_index(inplace=True)
                data[point].rename({'time': 'date'},
                                   axis=1, inplace=True)
                data[point].set_index('date', inplace=True)
                data_df = pd.merge(data_df, data[point], left_index=True, how='left', right_index=True)
                data_df.rename({u: "meteo_" + str(etab) + "_" + str(index) + "_" +
                                  u for u in data[point]}, axis=1, inplace=True)
                # Ajout des features décalés pour les 3 jours précédents et les 3 jours suiva + "_" + str(index)nts
                # for col in data[point]:
                #     if col != 'date_entree':
                #         features = features_augmentation(features, feature_name=f"{etab}_{index}_{col}", feature_category="meteo", shift=DECALAGE_METEO, isShifInDays=True)

                # for dec in range(1,DECALAGE_METEO+1):
                #     features[f"{etab}_{col}_{index}-{dec}"] = data[point][col].shift(dec).shift(-DECALAGE_METEO)
                #     # features[f"{etab}_{col}_{index}+{dec}"] = data[point][col].shift(-dec).shift(-DECALAGE_METEO)
                # # Ajouter une colonne moyenne glissante sur FENETRE_GLISSANTE jours pour chaque variable météo ne contenant pas de "-" dans le nom
                # if '-' not in col:
                #     features[f"{etab}_{col}_{index}_rolling"] = features[f"{etab}_{col}_{index}"].rolling(window=FENETRE_GLISSANTE, closed="left").mean()
                # features.rename({u: u + "_" + str(index) for u in data[point] if u != 'date_entree'}, axis=1, inplace=True)
                last_point = data[point]

                # except:
                #     if last_point is None:
                #         self.logger.info(f"Pas de données pour {point}, et pas de voisin")
                #         del data[point]
                #         continue
                #     else:
                #         self.logger.info(f"Pas de données pour {point}, on prend ceux des voisins")
                #         data[point] = copy.deepcopy(last_point)
        return data_df

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'feature_dir' in kwargs, f"Le paramètre'feature_dir' est obligatoire pour fetch la feature {self.name}"
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        assert 'etablissement' in kwargs, f"Le paramètre'etablissement' est obligatoire pour fetch la feature {self.name}"
        
        feature_dir = kwargs.get("feature_dir")
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        etablissement = kwargs.get("etablissement")
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date") # TODO: do not hardcode freq
        data = pd.DataFrame(index=date_range)

        data = data.join(self.include_weather(etablissement=etablissement, date_range=date_range, feature_dir=feature_dir))

        return data
