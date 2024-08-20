from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from src.encoders.datetime_encoders import DateFeatureExtractor


def create_encoding_pipeline(encoders_dict):
    print("Creating encoding pipeline")
    transformers = []
    for col_type, encoders in encoders_dict.items():
        if col_type == 'number':
            imputer = SimpleImputer(strategy='mean')
            encoded_features = make_union(
                *encoders).set_output(transform='pandas')
            pipeline = make_pipeline(
                imputer, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='number')
            transformer = make_column_transformer(
                (pipeline, selector), remainder='passthrough').set_output(transform='pandas')
            transformers.append(transformer)
            # print(transformer)
            # transformers[-1].fit(data)
            # print(transformers[-1].transform(data))
        elif col_type == 'category':
            imputer = SimpleImputer(strategy='constant', fill_value='missing')
            encoded_features = make_union(
                *encoders).set_output(transform='pandas')
            pipeline = make_pipeline(
                imputer, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='category')
            transformer = make_column_transformer(
                (pipeline, selector), remainder='passthrough').set_output(transform='pandas')
            transformers.append(transformer)
            # print(transformer)
            # transformers[-1].fit(data)
            # print(transformers[-1].transform(data))
        elif col_type == 'datetime':
            date_transformers = []

            if 'as_category' in encoders:
                date_extractor = DateFeatureExtractor(dtype='category')
            else:
                date_extractor = DateFeatureExtractor()

            date_transformers.append(date_extractor)
            
            for c_types, enc in encoders.items():

                if c_types == 'as_number':
                    number_selector = make_column_selector(dtype_include='number')
                    number_encoders = enc
                    if len(number_encoders) == 0:
                        continue
                    encoded_features_as_number = make_union(
                        *number_encoders).set_output(transform='pandas')
                    features_encoded_as_number = make_column_transformer(
                        (encoded_features_as_number, number_selector), remainder='passthrough').set_output(transform='pandas')
                    # print(features_encoded_as_number)
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
                        (encoded_features_as_category, category_selector), remainder='passthrough').set_output(transform='pandas')
                    # print(features_encoded_as_category)
                    date_transformers.append(features_encoded_as_category)
            
            # print(date_transformers)

            pipeline = make_pipeline(
                *date_transformers).set_output(transform='pandas')
            # pipeline = make_pipeline(date_extractor, encoded_features).set_output(transform='pandas')
            selector = make_column_selector(dtype_include='datetime')
            date_transformer = make_column_transformer(
                (pipeline, selector), remainder='passthrough').set_output(transform='pandas')
            transformers.append(date_transformer)
            # print(date_transformer)
            # transformers[-1].fit(data)
            # print(transformers[-1].transform(data))

    processor = make_pipeline(*transformers, verbose=True).set_output(transform='pandas')
    # processor.set_params(params={'verbose', True})

    return processor