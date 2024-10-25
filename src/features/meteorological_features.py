from src.features.base_features import BaseFeature
from src.location.location import Location
from typing import Optional, Dict
import pandas as pd
import datetime as dt
import meteostat
from shapely import wkt
import pathlib
import os
from typing import List


class MeteorologicalFeatures(BaseFeature):
    def __init__(self, name:str = None, logger=None) -> None:
        super().__init__(name, logger)
        
    def include_weather(self, location: Location, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        meteostat.Point.radius = 60000
        # meteostat.Point.method = "weighted"
        meteostat.Point.alt_range = 1000
        meteostat.Point.max_count = 3
            
        
        point = location.coordinates
        loc = meteostat.Point(point[1], point[0])
        
        #data = meteostat.Daily(loc, self.start - dt.timedelta(**self.step), self.stop + dt.timedelta(**self.step))
        data = meteostat.Daily(loc, date_range.min(), date_range.max())
        # on comble les heures manquantes (index) dans les données collectées
        data = data.normalize()
        # On complète les Nan, quand il n'y en a pas plus de 3 consécutifs
        data = data.interpolate()
        data = data.fetch()
        #assert len(data) > 0
        data['snow'] = data['snow'].fillna(0)
        data.drop(['tsun', 'coco', 'wpgt'], axis=1, inplace=True, errors='ignore')
        #data.drop(['tsun', 'wpgt'], axis=1, inplace=True, errors='ignore')
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0, inplace=True)
        data.reset_index(inplace=True)
        data.rename({'time': 'date'}, axis=1, inplace=True)
        data.set_index('date', inplace=True)
        #self.data = pd.merge(self.data, data, left_index=True, how='left', right_index=True)
        #self.data.rename({u: "meteo_" +  u for u in data + str(self.location.name)}, axis=1, inplace=True)

        data.columns = ["meteo_" +  col + "_" + str(location.name) for col in data.columns]
        return data

    def fetch_data_function(self, *args, **kwargs) -> None:
        assert 'feature_dir' in kwargs, f"Le paramètre'feature_dir' est obligatoire pour fetch la feature {self.name}"
        assert 'start_date' in kwargs, f"Le paramètre'start_date' est obligatoire pour fetch la feature {self.name}"
        assert 'stop_date' in kwargs, f"Le paramètre'stop_date' est obligatoire pour fetch la feature {self.name}"
        assert 'location' in kwargs, f"Le paramètre'location' est obligatoire pour fetch la feature {self.name}"
        
        feature_dir = kwargs.get("feature_dir")
        start_date = kwargs.get("start_date")
        stop_date = kwargs.get("stop_date")
        location = kwargs.get("location")
        
        date_range = pd.date_range(start=start_date, end=stop_date, freq='1D', name="date") # TODO: do not hardcode freq
        data = pd.DataFrame(index=date_range)

        data = self.include_weather(location=location, date_range=date_range) #, feature_dir=feature_dir))

        return data
