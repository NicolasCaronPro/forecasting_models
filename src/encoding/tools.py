from typing import Dict, List, Union
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.impute import SimpleImputer
from src.encoding.encoders import *

def create_encoding_pipeline(encoders_dict: Dict[str, Dict[str, Dict[str, List[Union[SimpleImputer, Pipeline]]]]]) -> Pipeline:
    """
    This function creates a sklearn pipeline for encoding and imputing data based on a dictionary of encoders.

    Parameters:
    encoders_dict (Dict[str, Dict[str, Dict[str, List[Union[SimpleImputer, Pipeline]]]]]): A dictionary containing the encoders to use for each data type.
        The dictionary should have the following structure:
        {
            'dtype': {
                'as_[dtype]': {
                    'imputers': [List of imputers],
                    'encoders': [List of encoders]
                }
            }
        }
        - with dype being one of 'number', 'category', 'datetime', specifying the data type to encode.
        - with as_[dtype] specifying how to consider the data type. For example you might want to encode datetime as numbers or as categories.

    Returns:
    Pipeline: A sklearn pipeline that can be used to transform data.
    """
    print("Creating encoding pipeline")
    transformers = []

    for data_type, encoding_types in encoders_dict.items():
        for encoding_type, encoders in encoding_types.items():
            imputers = encoders.get('imputers', [])
            encoders = encoders.get('encoders', [])

            if len(imputers) > 0:
                imputer = make_union(*imputers).set_output(transform='pandas')
            else:
                imputer = None

            if len(encoders) > 0:
                encoded_features = make_union(*encoders).set_output(transform='pandas')
            else:
                encoded_features = None

            if imputer and encoded_features:
                pipeline = make_pipeline(imputer, encoded_features).set_output(transform='pandas')
            elif imputer:
                pipeline = imputer
            elif encoded_features:
                pipeline = encoded_features
            else:
                continue
            
            try:
                selector = make_column_selector(dtype_include=data_type)
            except ValueError:
                print(f"Invalid data type: {data_type}")
                exit(1)

            # transformer = make_column_transformer((pipeline, selector), remainder='drop').set_output(transform='pandas')
            transformers.append((pipeline, selector))

    processor = make_column_transformer(*transformers, remainder='passthrough', verbose=True).set_output(transform='pandas')

    return processor
