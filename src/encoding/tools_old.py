from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.impute import SimpleImputer
from src.encoding.encoders import *


# TODO: rewrite this to make it easier to add support for new data types

def create_encoding_pipeline(encoders_dict) -> 'Pipeline':
    print("Creating encoding pipeline")
    transformers = []
    for data_type, encoders in encoders_dict.items():

        # If the column type is number then we will use the mean strategy to impute the missing values
        if data_type == 'number':

            # Create a simple imputer with the mean strategy
            imputer = SimpleImputer(strategy='mean')

            # Create a union of all the encoders given to encode the number columns
            encoded_features = make_union(
                *encoders).set_output(transform='pandas')

            # Create a pipeline with the imputer and the encoded features
            pipeline = make_pipeline(
                imputer, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='number')
            transformer = make_column_transformer(
                (pipeline, selector), remainder='drop').set_output(transform='pandas')
            transformers.append(transformer)

        elif data_type == 'category':
            imputer = SimpleImputer(strategy='constant', fill_value='missing')
            encoded_features = make_union(
                *encoders).set_output(transform='pandas')
            pipeline = make_pipeline(
                imputer, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='category')
            transformer = make_column_transformer(
                (pipeline, selector), remainder='drop').set_output(transform='pandas')
            transformers.append(transformer)

        elif data_type == 'datetime':
            date_transformers = []

            if 'as_category' in encoders:
                date_extractor = de.DateFeatureExtractor(dtype='category')
            else:
                date_extractor = de.DateFeatureExtractor()

            # date_transformers.append(date_extractor)

            for c_types, enc in encoders.items():

                if c_types == 'as_number':
                    number_selector = make_column_selector(
                        dtype_include='number')
                    number_encoders = enc
                    if len(number_encoders) == 0:
                        continue
                    encoded_features_as_number = make_union(
                        *number_encoders).set_output(transform='pandas')
                    features_encoded_as_number = make_column_transformer(
                        (encoded_features_as_number, number_selector), remainder='drop').set_output(transform='pandas')
                    date_transformers.append(features_encoded_as_number)

                if c_types == 'as_category':
                    category_selector = make_column_selector(
                        dtype_include='category')
                    category_encoders = enc
                    if len(category_encoders) == 0:
                        continue
                    encoded_features_as_category = make_union(
                        *category_encoders).set_output(transform='pandas')
                    features_encoded_as_category = make_column_transformer(
                        (encoded_features_as_category, category_selector), remainder='drop').set_output(transform='pandas')
                    date_transformers.append(features_encoded_as_category)

            union = make_union(
                *date_transformers).set_output(transform='pandas')
            pipeline = make_pipeline(*[date_extractor, union])
            selector = make_column_selector(dtype_include='datetime')
            date_transformer = make_column_transformer(
                (pipeline, selector), remainder='drop').set_output(transform='pandas')
            transformers.append(date_transformer)

        else:
            raise ValueError(
                f"Data type {data_type} not recognized. Please use one of the following: ['number', 'category', 'datetime'], or add a new data type to the function")

    processor = make_union(
        *transformers, verbose=True).set_output(transform='pandas')

    return processor



