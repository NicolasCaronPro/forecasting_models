from typing import *
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
from enum import Enum
from geopy.geocoders import Nominatim
import requests


ETAB_FILE = '../data/geolocalisation/etab_coord.csv'

GEO_DIR = '../data/geolocalisation/'

ETAB_NAMES = [
    ['CH BEAUNE', 'CH Beaune', 'CH BEAUNE'],
    ['CH SEMUR EN AUXOIS', 'CH Semur', 'CH SEMUR EN AUXOIS'],
    ['CH MONTBARD', 'CH Chatillon Montbard', 'CH MONTBARD'],
    ['HNFC', 'HNFC', 'HNFC'],
    ['CHU BESANCON', 'CHU Besançon', 'CHU BESANCON'],
    ['CH PRIVE DIJON', 'CH privé Dijon', 'CH privé DIJON'],
    ['CHU DIJON', 'CHU Dijon', 'CHU DIJON'],
    ['CH LANGRES', 'CH Langres', 'CH LANGRES'],
    ['CH CHAUMONT', 'CH Chaumont', 'CH CHAUMONT']
]

REGION_OLD = {
    'ALSACE': ['67', '68'],
    'AQUITAINE': ['24', '33', '40', '47', '64'],
    'AUVERGNE': ['03', '15', '43', '63'],
    'BASSE-NORMANDIE': ['14', '50', '61'],
    'BOURGOGNE': ['21', '58', '71', '89'],
    'BRETAGNE': ['22', '29', '35', '56'],
    'CENTRE': ['18', '28', '36', '37', '41', '45'],
    'CHAMPAGNE-ARDENNE': ['08', '10', '51', '52'],
    'CORSE': ['2A', '2B'],
    'FRANCHE-COMTE': ['25', '39', '70', '90'],
    'HAUTE-NORMANDIE': ['27', '76'],
    'ILE-DE-FRANCE': ['75', '77', '78', '91', '92', '93', '94', '95'],
    'LANGUEDOC-ROUSSILLON': ['11', '30', '34', '48', '66'],
    'LIMOUSIN': ['19', '23', '87'],
    'LORRAINE': ['54', '55', '57', '88'],
    'MIDI-PYRENEES': ['09', '12', '31', '32', '46', '65', '81', '82'],
    'NORD-PAS-DE-CALAIS': ['59', '62'],
    'PAYS-DE-LA-LOIRE': ['44', '49', '53', '72', '85'],
    'PROVENCE-ALPES-COTE-D-AZUR': ['04', '05', '06', '13', '83', '84'],
    'PICARDIE': ['02', '60', '80'],
    'POITOU-CHARENTES': ['16', '17', '79', '86'],
    'RHONE-ALPES': ['01', '07', '26', '38', '42', '69', '73', '74'],
}

REGION_TRENDS = {
    'ALSACE': 'FR-A',
    'AQUITAINE': 'FR-B',
    'AUVERGNE': 'FR-C',
    'BASSE-NORMANDIE': 'FR-P',
    'BOURGOGNE': 'FR-D',
    'BRETAGNE': 'FR-E',
    'CENTRE': 'FR-F',
    'CHAMPAGNE-ARDENNE': 'FR-G',
    'CORSE': 'FR-H',
    'FRANCHE-COMTE': 'FR-I',
    'HAUTE-NORMANDIE': 'FR-Q',
    'ILE-DE-FRANCE': 'FR-J',
    'LANGUEDOC-ROUSSILLON': 'FR-K',
    'LIMOUSIN': 'FR-L',
    'LORRAINE': 'FR-M',
    'MIDI-PYRENEES': 'FR-N',
    'NORD-PAS-DE-CALAIS': 'FR-O',
    'PAYS-DE-LA-LOIRE': 'FR-R',
    'PROVENCE-ALPES-COTE-D-AZUR': 'FR-U',
    'PICARDIE': 'FR-S',
    'POITOU-CHARENTES': 'FR-T',
    'RHONE-ALPES': 'FR-V',
}


class Scale(Enum):
    REGION = 1
    DEPARTEMENT = 2
    COMMUNE = 3
    COORDS = 4
    ETAB = 5


def get_etabs(path: str = ETAB_FILE) -> List[str]:
    etab = pd.read_csv(path, sep=',')
    return etab['etablissement'].values


def get_coordinates(city_name) -> Tuple[float, float]:
    geolocator = Nominatim(user_agent="oui")
    location = geolocator.geocode(city_name + ", France")
    if location:
        return (location.latitude, location.longitude)
    return None


def coord_info(coords) -> Dict[str, str]:
    url = f"https://api-adresse.data.gouv.fr/reverse/?lon={coords[0]}&lat={coords[1]}"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        city_name = response.json()['features'][0]['properties']['city']
        city_code = response.json()['features'][0]['properties']['citycode']
        url = f"https://geo.api.gouv.fr/communes?nom={city_name}&fields=departement,region&format=json&geometry=centre"
        response = requests.get(url)
        if response.status_code == 200 and response.json():
            for city in response.json():
                if city['nom'].lower() == city_name.lower() and city['code'] == city_code:
                    data = city
                    break
        return {
            'city': data['nom'],
            'code': data['code'],
            'departement': data['departement']['nom'],
            'code_departement': data['departement']['code'],
            'region': data['region']['nom'],
            'code_region': data['region']['code']
        }
    else:
        return None


def find_coordinates_etab(name, path: str = ETAB_FILE) -> gpd.GeoDataFrame:
    etab = pd.read_csv(path, sep=',')
    # coords = (None, None)
    x_int = None
    y_int = None
    coords = gpd.GeoDataFrame()
    if name in etab['etablissement'].values:
        coords = etab[etab['etablissement'] == name][['position']].values[0][0]
        # Remove "POINT" and parentheses, then split the numbers
        coords = coords.replace("POINT", "").replace(
            "(", "").replace(")", "").strip().split()
        # Convert the values to integers
        x_int = float(coords[0])
        y_int = float(coords[1])
    coords = (x_int, y_int)
    return coords

# def get_shape(name, path: str = ETAB_FILE) -> Polygon:
#     etab = pd.read_csv(path, sep=',')
#     if name in etab['etablissement'].values:
#         shape_str = etab[etab['etablissement'] == name][['isomap']].values[0][0]
#         #shape_str = shape_str[1:-1]
#         # Remove "POLYGON" and parentheses, then split the coordinates
#         coords = shape_str.replace("POLYGON", "").replace("((", "").replace("))", "").strip().split(", ")

#         # Convert the coordinates to tuples of floats
#         coords_tuple = [tuple(map(float, coord.split())) for coord in coords]

#         # Create the Polygon object
#         polygon = Polygon(coords_tuple)
#         return polygon
#     return None


class Location():
    def __init__(self, name: str, coordinates: Tuple[float, float] = (None, None)) -> None:
        self.name = name
        if coordinates == (None, None):
            self.coordinates = find_coordinates_etab(self.get_name(mode=1))

            if self.coordinates != (None, None):
                self.scale = Scale.ETAB
        else:
            self.coordinates = coordinates
            self.scale = Scale.COORDS

        city_info = coord_info(self.coordinates)
        self.city = city_info['city']
        self.departement = city_info['departement']
        self.code_departement = city_info['code_departement']
        self.region = city_info['region']
        self.code_region = city_info['code_region']
        self.code = city_info['code']
        for reg, dep in REGION_OLD.items():
            if self.code_departement in dep:
                self.region_old = reg
                self.region_trends = REGION_TRENDS[reg]
        self.__shape = None
        self.__centroid = None
        # self.scale = Scale.COORDS

    def get_name(self, mode: int = 0) -> str:
        name = ""
        match mode:
            case 0:  # Original name
                name = self.name
            case 1:  # Capital letters
                for etab in ETAB_NAMES:
                    if self.name in etab:
                        name = etab[0]
                        break
            case 2:  # Normal letters
                for etab in ETAB_NAMES:
                    if self.name in etab:
                        name = etab[1]
                        break
            case 3:  # Capital letters with accents
                for etab in ETAB_NAMES:
                    if self.name in etab:
                        name = etab[2]
                        break
        return name

    def get_shape(self) -> Polygon:
        if self.__shape is None:
            self.__shape, self.__centroid = self.__influence_shape()
        return self.__shape

    def get_centroid(self) -> Point:
        if self.__centroid is None:
            self.__shape, self.__centroid = self.__influence_shape()
        return self.__centroid

    def is_in_shape(self, point: Point) -> bool:
        return self.get_shape().contains(point)

    def __influence_shape(self) -> Tuple[Polygon, Point]:
        df_code = pd.read_excel(
            GEO_DIR + "codes postaux_PMSI_pop_ATIH_2023.xlsx")
        df_code = df_code.drop(columns=[col for col in df_code.columns if col not in [
                               'Code postal 2023', 'Libellé poste', 'Code géographique PMSI 2023']])
        df_code = df_code.rename(columns={'Code postal 2023': 'code_postal',
                                 'Libellé poste': 'libelle', 'Code géographique PMSI 2023': 'code_geo'})
        df_code['code_geo'] = df_code['code_geo'].astype(str)

        geom = gpd.read_file(
            GEO_DIR + 'SECTEURS PMSI_BFC et limitrophes_2023/SECTEURS_PMSI_BFC_et_limitrophes_2023.shp')
        geom.rename(columns={'PMSI_2023': 'code_geo'}, inplace=True)
        geom['code_geo'] = geom['code_geo'].astype(str)

        df_tx = pd.read_excel(
            GEO_DIR + "tx de recours RPU et motifs.xlsx", sheet_name=self.get_name(mode=3))

        df_tx = df_tx.drop(columns="rs2")
        df_tx = df_tx.rename(columns={'codegeo': 'code_geo'})
        df_tx['code_geo'] = df_tx['code_geo'].astype(str)

        df_tx = df_tx.merge(df_code, on='code_geo')
        df_tx = gpd.GeoDataFrame(
            pd.merge(df_tx, geom, on='code_geo', how='left'))
        df_tx.sort_values(by='tx_recours', ascending=False, inplace=True)
        df_tx.drop_duplicates(subset='geometry', keep="last", inplace=True)

        df_tx["centroid"] = df_tx.geometry.centroid.to_crs(epsg=4326)

        df_tx.geometry = df_tx.geometry.to_crs(epsg=4326)

        polygons = []
        for pol in df_tx.geometry:
            polygons.append(pol)

        # Create a GeoSeries or GeoDataFrame from the list of polygons
        gdf = gpd.GeoSeries(polygons)

        # Perform the union of all polygons
        unioned_polygon = gdf.unary_union

        final_polygon = Polygon(unioned_polygon.exterior)

        return final_polygon, df_tx['centroid'].values[0]

    def filter_points(self, list_points: List[Point], n_points: int = 5, buffer_range: int = 10000, buffer_incr: int = 250, verbose=False) -> gpd.GeoDataFrame:
        points_df = gpd.GeoDataFrame({'points': list_points})
        points_df = points_df.set_geometry('points', crs='epsg:4326')

        points_df = points_df.to_crs(epsg=3857)
        # Créer un tampon autour de chaque point
        points_selectionnes = points_df
        while len(points_selectionnes) > n_points:
            tampons = points_df.buffer(buffer_range)

            # Trouver les points qui ne se chevauchent pas
            points_selectionnes = []
            tampons_selectionnes = []
            for index, (point, tampon) in enumerate(zip(points_df.geometry, tampons)):
                intersect = False
                for tamp in tampons_selectionnes:
                    if tampon.intersects(tamp):
                        intersect = True
                        break
                if not intersect:
                    points_selectionnes.append(point)
                    tampons_selectionnes.append(tampon)

            buffer_range += buffer_incr

        if verbose:
            print(
                f"{len(points_selectionnes)} points selected out of {len(points_df)}")

        assert len(
            points_selectionnes) == n_points, "Points number different than {n_points}"

        return gpd.GeoDataFrame(geometry=points_selectionnes, crs='EPSG:3857').to_crs(epsg=4326)

    def __str__(self) -> str:
        # return f"{self.name} is located at {self.coordinates}"
        return f"{self.name} is located at {self.city}, {self.code}, {self.departement}, {self.region}"

    def __print__(self) -> None:
        print(self.__str__())


def get_shape_geo(loc: Location, path: str = "SHAPE_FILE") -> gpd.GeoDataFrame:
    pass
