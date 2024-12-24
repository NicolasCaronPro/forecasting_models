import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import log_loss, mean_squared_error
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

################################# Never used ##########################################

class ModelTorch():
    def __init__(self, model, loss='mse', name='ModelTorch', device='cpu'):
        self.model = model
        self.name = name
        self.loss = loss
        self.device = device
        self.model.to(self.device)
        self.optimizer = None

    def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.001, optimization='skip', grid_params=None, fit_params={}):
        self.model.train()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self._compute_loss(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()

    def _compute_loss(self, outputs, targets):
        if self.loss == 'mse':
            return F.mse_loss(outputs, targets)
        elif self.loss == 'log_loss':
            return F.binary_cross_entropy_with_logits(outputs, targets)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def score(self, X, y, sample_weight=None):
        predictions = self.predict(X)
        if self.loss == 'mse':
            return -mean_squared_error(y, predictions)
        elif self.loss == 'log_loss':
            return -log_loss(y, predictions)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        
class ModelCNN(ModelTorch):
    def __init__(self, model, loss='mse', name='ModelCNN', device='cpu'):
        super().__init__(model, loss=loss, name=name, device=device)

class ModelGNN(ModelTorch):
    def __init__(self, model, loss='mse', name='ModelGNN', device='cpu'):
        super().__init__(model, loss=loss, name=name, device=device)

class ModelHybrid(ModelTorch):
    def __init__(self, model, loss='mse', name='ModelGNN', device='cpu'):
        super().__init__(model, loss=loss, name=name, device=device)