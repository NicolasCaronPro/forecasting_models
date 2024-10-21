from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class MeanFeaturesModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.mean_ = None
    
    def fit(self, X, y=None):
        # Calculer la moyenne des features
        #self.mean_ = np.mean(X, axis=0)
        #return self
        pass
    
    def predict(self, X):
        # Retourner la moyenne des features
        return np.mean(X, axis=1)