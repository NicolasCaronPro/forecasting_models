"""
Eperiment class to run a single experiment with a given model (or ensemble of model) to train on a dataset, with a given config.
"""

import logging
import os
import sys
import datetime as dt
from typing import List, Union, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.models.base_sklearn_model import BaseSklearnModel
from src.datasets.base_tabular_dataset import BaseTabularDataset
import src.features as ft


class BaseExperiment:
    def __init__(self, model: Union[BaseSklearnModel, List[BaseSklearnModel]], dataset: BaseTabularDataset, config: 'ft.Config') -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = config.logger

    def run(self) -> None:
        """
        Run the experiment.

        Parameters:
        - None
        """
        self.logger.info("Running the experiment...")
        self.dataset.fetch_data()
        self.dataset.encode()
        self.dataset.split_data()
        self.train()
        self.test()

    def train(self) -> None:
        """
        Train the model.

        Parameters:
        - None
        """
        self.logger.info("Training the model...")
        self.model.fit(self.dataset.X_train, self.dataset.y_train)

    def test(self) -> None:
        """
        Test the model.

        Parameters:
        - None
        """
        self.logger.info("Testing the model...")
        y_pred = self.model.predict(self.dataset.X_test)
        mse = mean_squared_error(self.dataset.y_test, y_pred)
        self.logger.info(f"Mean Squared Error: {mse}")

    def save(self) -> None:
        """
        Save the model.

        Parameters:
        - None
        """
        self.logger.info("Saving the model...")
        self.model.save()

    def load(self) -> None:
        """
        Load the model.

        Parameters:
        - None
        """
        self.logger.info("Loading the model...")
        self.model.load()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the target.

        Parameters:
        - X (pd.DataFrame): The input data.

        Returns:
        - pd.DataFrame: The predicted target.
        """
        self.logger.info("Predicting the target...")
        return self.model.predict(X)