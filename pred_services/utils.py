from xgboost import XGBRegressor
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.encoding.encoders import *
from src.encoding.tools import create_encoding_pipeline

from sklearn.model_selection import train_test_split

from typing import List


TARGETS = ['nb_(vide)', 'nb_CHIR', 'nb_MED', 'nb_PSA', 'nb_SI', 'nb_UHCD', 'nb_autres']
SERVICES_EXCEL_PATH = "pred_services/hebdo_CHU_Dijon.xlsx"
DATA_FEATHER_PATH = "data/basetabulardataset/data_BaseTabularDataset_nb_vers_hospit.feather"


def read_xls_service(path:str, prefix:str = "nb_") -> pd.DataFrame:
    """
    Reads an Excel file and processes the data into a pivoted DataFrame.

    Args:
        path (str): The file path to the Excel file.
        prefix (str, optional): A prefix to add to the column names. Defaults to "nb_".

    Returns:
        pd.DataFrame: A DataFrame with the processed data, indexed by date.

    The function performs the following steps:
    1. Reads the specified Excel file and selects the "orientation" sheet with columns A to D.
    2. Pivots the data on 'Annee' and 'Semaine' with 'orientation' as columns and 'Valeur' as values.
    3. Resets the index to create a clean DataFrame.
    4. Creates a new 'date' column from 'Annee' and 'Semaine'.
    5. Drops the 'Annee' and 'Semaine' columns.
    6. Sets the 'date' column as the index.
    7. Adds the specified prefix to the column names.
    8. Fills any NaN values with 10.

    Example:
        df = read_xls_service('/path/to/file.xlsx', prefix='service_')
    """

    xls = pd.read_excel(path, sheet_name="orientation", usecols="A:D")

    df_pivot = xls.pivot(index=['Annee', 'Semaine'], columns='orientation', values='Valeur')

    # Resetting the index to have a clean DataFrame
    df_pivot = df_pivot.reset_index()
    df_pivot.columns.name = None

    # Create a new column 'Date' from 'Annee' and 'Semaine'
    df_pivot['date'] = pd.to_datetime(df_pivot['Annee'].astype(str) + df_pivot['Semaine'].astype(str).str.zfill(2) + '7', format='%G%V%u')

    # Drop the 'Annee' and 'Semaine' columns
    df = df_pivot.drop(columns=['Annee', 'Semaine'])

    # Set the 'Date' column as the index
    df = df.set_index('date')

    df.columns = [prefix + str(col) for col in df.columns]

    df = df.fillna(10)

    return df




def load_feather_data_to_service(df_incomplet: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and process feather data, resample it weekly, and join with an existing DataFrame.

    Args:
        path (str): The file path to the feather data.
        df (pd.DataFrame): The DataFrame to join with the processed feather data.

    Returns:
        pd.DataFrame: The final processed DataFrame after resampling and joining.
    """

    #df_incomplet.drop(columns='HNFC_moving', inplace=True)
    cols: List[str] = df_incomplet.select_dtypes(include='category').columns.to_list()

    df_incomplet[cols] = df_incomplet[cols].astype('int64')

    print(df_incomplet.index)
    df_resample: pd.DataFrame = df_incomplet.resample('W').sum()

    df_final: pd.DataFrame = df_resample.join(df, on='date')

    #df_final.drop('nb_vers_hospit', axis=1, inplace=True)
    #df_final.drop(df.filter(regex='^nb_vers_hospit').columns, axis=1, inplace=True)

    return df_final



def choose_target(df: pd.DataFrame, targets: List[str], target: int) -> pd.DataFrame:
    """
    Drop columns from a DataFrame that are in the targets list except the target.

    Args:
        df (pd.DataFrame): The DataFrame to drop columns from.
        targets (List[str]): A list of column names to drop.
        target (str): The target column name to keep.

    Returns:
        pd.DataFrame: The DataFrame after dropping the specified columns.
    """

    return df.drop(columns=[col for col in df.columns if col in targets and col != targets[target]])



def fetch_data_services(target:int = -1, excel_path:str = SERVICES_EXCEL_PATH, data_path = DATA_FEATHER_PATH, encode:bool = True) -> pd.DataFrame:
    """
    Fetch the data for the services and optionally encode it.

    Args:
        target (int, optional): The target column index. Defaults to -1, which means all targets column are selected.
        excel_path (str): The file path to the Excel file.
        data_path (str): The file path to the feather data.
        encode (bool, optional): Whether to encode the data. Defaults to True.

    Returns:
        pd.DataFrame: The processed DataFrame after loading the data.
    """

    df = read_xls_service(excel_path, prefix='nb_')
    df_incomplet: pd.DataFrame = pd.read_feather(data_path)
    df_final = load_feather_data_to_service(df_incomplet, df)

    if target == -1:
        data = df_final.copy(deep=True)
        selected_targets = TARGETS
    else:
        data = choose_target(df_final, TARGETS, target)
        selected_targets = [TARGETS[target]]

    if data.index.name not in data.columns:
        kwargs = {str(data.index.name): data.index}
        data = data.assign(**kwargs)

    features = data.drop(selected_targets, axis=1)
    targets = data[selected_targets]

    if encode:
        encoders_dict = {
            'number': {
                'as_number': {
                    'imputers': [imputers.SimpleImputer(strategy='mean')],
                    'encoders': [
                        ne.StandardScaler(),

                    ]
                }
            },
            'category': {
                'as_category': {
                    'imputers': [imputers.SimpleImputer(strategy='most_frequent')],
                    'encoders': [
                        # ne.TargetEncoder(target_type='continuous-multioutput'),
                        # ne.TargetEncoder(target_type='continuous'),
                        ne.MultiTargetEncoder(drop_invariant=True, return_df=True),
                    ]
                }
            },
            'datetime': {
                'as_number': {
                    'imputers': [de.DateFeatureExtractor()],
                    'encoders': [
                        ne.CyclicalFeatures(drop_original=True)
                    ]
                },
                'as_category': {
                    'imputers': [de.DateFeatureExtractor(dtype='category')],
                    'encoders': [
                        # ne.TargetEncoder(target_type='continuous'),
                        ne.MultiTargetEncoder(drop_invariant=True, return_df=True),
                    ]
                }
            }
        }
        processor = create_encoding_pipeline(encoders_dict)
        processor.fit(features, targets)
        encoded_data = processor.transform(data)
        encoded_data.columns = [col.split('__')[-1] for col in encoded_data.columns]
        data = encoded_data.copy(deep=True)
    return data


#print(fetch_data_services())




# # read data
# data = df_final.copy(deep=True)

# # split data into train and test sets
# train_set, test_set = train_test_split(data, test_size=0.2, shuffle=False)
# train_set, val_set = train_test_split(train_set, test_size=0.2, shuffle=False)

# x_train = train_set.drop(targets, axis=1)
# y_train = train_set[targets[target]]

# x_test = test_set.drop(targets, axis=1)
# y_test = test_set[targets[target]]

# x_val = val_set.drop(targets, axis=1)
# y_val = val_set[targets[target]]


# encoded_x_train = processor.transform(x_train)
# encoded_x_train.columns = [col.split('__')[-1] for col in encoded_x_train.columns]

# encoded_x_val = processor.transform(x_val)
# encoded_x_val.columns = [col.split('__')[-1] for col in encoded_x_val.columns]