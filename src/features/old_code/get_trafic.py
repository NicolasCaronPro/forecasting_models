from bs4 import BeautifulSoup
from datetime import date, datetime
import numpy as np
import os
import pandas as pd
import requests
from pathlib import Path

today = date.today()
d = today.strftime('%Y-%m-%d')
print("Today's date:", d)

bison_urls = ["https://tipi.bison-fute.gouv.fr/bison-fute-ouvert/publicationsDIR/Evenementiel-DIR/cnir/RecapBouchonsFranceEntiere.html",
              "https://tipi.bison-fute.gouv.fr/bison-fute-ouvert/publicationsDIR/Evenementiel-DIR/cnir/RecapTraficFranceEntiere.html"]

data_folder = Path(__file__).resolve().parents[0] / 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(data_folder / 'raw'):
    os.makedirs(data_folder / 'raw')

keys = ['nature', 'horodate', 'axe', 'sens_cardinal', 'point_repere', 'longueur_pr', 'commune']
qnames = ['nature', 'ligne_horodate_fin_exception_ve', 'axe', 'sens_cardinal', 'pr', 'longueur', 'commune']


def process_data(next_element, keys, qnames, importance, department_name, df):
    data = {}

    for key, qname in zip(keys, qnames):
        try:
            element = next_element.find('span', {'qname': qname})
            element = next_element.find('span', class_=qname) if not element else element # if qname is not found, try with class
            data[key] = element.get_text(strip=True) if element else None
        except AttributeError:
            data[key] = None

    data['date'] = d
    data['department'] = department_name
    data['importance'] = len(importance)

    new_row_df = pd.DataFrame(data, index=[0])
    updated_df = pd.concat([df, new_row_df], ignore_index=True)

    return updated_df


for bison_url in bison_urls:
    html = requests.get(bison_url)

    bsobj = BeautifulSoup(html.content, 'html.parser')

    df = pd.DataFrame()

    departments = bsobj.find_all('span', class_='rupture')
    initial_rows = df.shape[0]

    for department in departments:
        department_name = department.find('a').get_text(strip=True)
        next_element = department.find_next_sibling()
        while next_element and next_element.name != 'span' and 'rupture' not in next_element.get('class', []):
            if next_element.name == 'div' and 'interligne' in next_element['class'][0]:
                elements = next_element.find_all('span', {'qname': 'element'})
                for element in elements:
                    importance = next_element.find('span', {'qname': 'importance_vr_reduit'}).get_text(strip=True)
                    df = process_data(element, keys, qnames, importance, department_name, df)

            next_element = next_element.find_next_sibling()

    df.drop_duplicates(inplace=True) # I don't want to add elements I already have in my df
    print(f"Added {df.shape[0] - initial_rows} new rows")

    df.to_csv(data_folder / f'raw/bison_fute_{"trafic" if "Trafic" in bison_url else "bouchons"}_raw_{datetime.now()}.csv', index=False)

    df.fillna(value=np.nan, inplace=True)

    df['sens_cardinal'] = df['sens_cardinal'].str.replace(r'^\(|sens(?=\s)', '', regex=True).str.strip()

    df['longueur_pr'] = df['longueur_pr'].str.replace(' environ', '').str.replace('(sur ', '').str.replace(' km)', '').str.replace('de longueur indéterminée', 'nan')
    df['longueur_pr'] = df['longueur_pr'].str.replace(',', '.')
    df['longueur_pr'] = df['longueur_pr'].apply(lambda x: float(x) if x != 'nan' else np.nan)
    df = df.rename(columns={'longueur_pr': 'longueur_pr (km)'})


    df['department_number'] = df['department'].str.extract(r'Département (\d+)')
    df['department_name'] = df['department'].str.extract(r'\(([^()]+)\)')
    df.drop(columns=['department'], inplace=True)

    new_order = ['date', 'department_number', 'department_name'] + [col for col in df.columns if col not in ['date', 'department_number', 'department_name']]
    df = df[new_order]

    df.to_csv(data_folder / f'bison_fute_{"trafic" if "Trafic" in bison_url else "bouchons"}_{datetime.now()}.csv', index=False)