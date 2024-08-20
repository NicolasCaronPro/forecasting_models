from typing import Optional, Union, List
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
import src.features as ft
import src.encoders as enc
import pandas as pd


class BaseTabularDataset(ft.BaseFeature):
    def __init__(self, target_colomns: Union[List[str], str], features_class: List[Union[ft.BaseFeature, str]], config: Optional['ft.Config'] = None, parent: Optional['ft.BaseFeature'] = None) -> None:
        super().__init__(config=config, parent=parent)
        # Initialize each features object and fetch their data and then get them for the specified range

        # Get target

        self.targets = target_colomns if isinstance(
            target_colomns, list) else [target_colomns]
        self.features = []
        self.encoded_data = None

        # Initialize each feature
        self.initialize_features(features_class)

    def initialize_features(self, features_class) -> None:
        """
        Initialize each feature.

        Parameters:
        - None
        """

        for feature_class in features_class:
            if isinstance(feature_class, str):
                feature_class = getattr(ft, feature_class)
            feature = feature_class(config=self.config, parent=self)
            self.features.append(feature)

    def fetch_data(self) -> None:
        """
        Fetch les données.

        Parameters:
        - None
        """

        # Get data from each feature
        for feature in self.features:
            feature.fetch_data()

            self.data = self.data.join(feature.data)

        self.targets = self.data[self.targets]

    # def encode(self, encoders=None) -> None:
    #     """
    #     Encode les données.

    #     Parameters:
    #     - None
    #     """

        # for feature in self.features:
        #     for dtype in feature.get_dtypes():
        #         data = feature.data.select_dtypes(include=[dtype])
        #         for encoder in feature.encoders[dtype]:
        #             encoder.fit(data)

        # date_column = self.data.select_dtypes(include=['datetime64']).columns.tolist()
        # date_transformer = enc.de.DateTransformer(date_column=date_column)

        # # Create the column transformer based on the features generated by DateSplitter
        # def create_column_transformer(features):
        #     transformers = [(feature, enc.ce.OneHotEncoder(), [feature])
        #                     for feature in features]
        #     return ColumnTransformer(transformers=transformers, remainder='passthrough')

        # pipeline = Pipeline([
        #     ('date_splitter', date_transformer),
        #     # Placeholder, will be updated dynamically
        #     ('column_transformer', 'passthrough')
        # ])

        # self.encoders = {'dateteime64': enc.de.DateTransformer,
        #                  'category': enc.ce.TargetEncoder, 'number': enc.ne.StandardScaler}
        # if 'encoders' in self.config:
        #     self.encoders = self.config.get('encoders')
        #     assert isinstance(
        #         self.encoders, dict), f"encoders must be a dictionary, not {type(self.encoders)}"

        # for dtype in self.encoders:
        #     assert dtype in ['datetime64', 'category',
        #                      'number'], f"{dtype} must be one of ['datetime64', 'category', 'number']"
        #     assert isinstance(
        #         self.encoders[dtype], list), f"{self.encoders[dtype]} must be a list"
        #     for encoder in self.encoders[dtype]:
        #         assert isinstance(encoder, type), f"{encoder} must be a class"

        # date_pipeline = Pipeline([
        #     ('date_splitter', enc.de.DateTransformer()),
        #     ('column_transformer', create_column_transformer(
        #         date_transformer.decompose(self.data)))

        # self.encoders = [
        #     (enc.ne.StandardScaler(), make_column_selector(dtype_include='number')),
        #     (enc.ce.TargetEncoder(), make_column_selector(dtype_include='category')),
        #     (enc.de.DateTransformer(), make_column_selector(dtype_include='datetime64'))]

        # self.ct = make_column_transformer(*self.encoders)

        # Encodage des dates

        # Pipeline pour l'extraction des features de dates
        # self.date_extraction = make_pipeline((enc.de.DateTransformer(), make_column_selector(dtype_include='datetime64')))

        # # Pipeline pour l'encodage des features de dates
        # for date_encoder in encoders['datetime64']:
        #     self.date_pipeline = make_pipeline(self.date_extraction,
        #                                    make_column_transformer(
        #                                        (enc.de.CyclicalFeatures(), make_column_selector(
        #                                            dtype_include='number'))
        #                                    )
        #                                    )

        # self.date_pipeline = self.date_pipeline.fit(
        #     X=self.data, y=self.targets)

        # self.encoded_data = self.date_pipeline.transform(self.data)

        # print(self.encoded_data)

        # self.preprocessor = make_column_transformer(
        #     (enc.ne.StandardScaler(), make_column_selector(dtype_include='number')),
        #     (enc.ce.TargetEncoder(), make_column_selector(dtype_include='category')),
        #     (self.date_pipeline, make_column_selector(dtype_include='datetime64'))
        # )

        # self.clf = Pipeline([
        #     ('preprocessor', self.preprocessor)
        # ])

        # self.encoded_data = self.clf.fit(self.data, self.targets)

        # print(self.clf)
        # self.preprocessor = ColumnTransformer([
        #     ('date_preprocessor', self.date_preprocessor, make_column_selector(dtype_include='datetime64')),
        #     ('category_encoder', enc.ce.TargetEncoder(), make_column_selector(dtype_include='category')),
        #     ('number_encoder', enc.ne.StandardScaler(), make_column_selector(dtype_include='number'))
        # ])




    def encode(self, encoders: dict) -> pd.DataFrame:
        """
        Encode the data using the specified encoders.

        Parameters:
        - data (pd.DataFrame): The input data.
        - encoders (dict): A dictionary specifying the encoders for different data types.

        Returns:
        - pd.DataFrame: The encoded data.
        """
        
        def create_feature_union(encoders, dtype):
            return FeatureUnion([
                (f'{dtype}_encoder_{i}', Pipeline([
                    ('encoder', enc)
                ])) for i, enc in enumerate(encoders)
            ])
        transformers = []

        # Process numerical features
        if 'number' in encoders:
            num_transformers = ('num', create_feature_union(encoders['number'], 'number'), make_column_selector(dtype_include='number'))
            transformers.append(num_transformers)

        # Process categorical features
        if 'category' in encoders:
            cat_transformers = ('cat', create_feature_union(encoders['category'], 'object'), make_column_selector(dtype_include='object'))
            transformers.append(cat_transformers)

        # Process datetime features
        if 'datetime' in encoders:
            date_encoders = encoders['datetime']
            date_transformers = ('date', Pipeline([
                ('date_extraction', enc.de.DateTransformer()),
                ('date_encoding', create_feature_union(date_encoders, 'number'))
            ]), make_column_selector(dtype_include='datetime'))
            transformers.append(date_transformers)

        # Combine all transformers using ColumnTransformer
        preprocessor = ColumnTransformer(transformers)

        print(preprocessor)

        # Fit and transform the data
        preprocessor.fit(self.data)
        encoded_data = preprocessor.transform(self.data)

        # Convert the result to a DataFrame
        encoded_df = pd.DataFrame(encoded_data, columns=preprocessor.get_feature_names_out(), index=self.data.index)

        return encoded_df