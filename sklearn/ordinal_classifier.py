from sklearn.base import clone, BaseEstimator, ClassifierMixin
import numpy as np

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    #https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
    """
    A classifier that can be trained on a range of classes.
    @param classifier: A scikit-learn classifier.
    """
    def __init__(self,clf):
        self.clf = clf
        self.clfs = {}
        self.uniques_class = None

    def fit(self,X,y, **fit_params):
        self.uniques_class = np.sort(np.unique(y))
        assert self.uniques_class.shape[0] >= 3, f'OrdinalClassifier needs at least 3 classes, only {self.uniques_class.shape[0]} found'

        for i in range(self.uniques_class.shape[0]-1):
            binary_y = (y > self.uniques_class[i]).astype(np.uint8)
            
            clf = clone(self.clf)
            clf.fit(X, binary_y, **fit_params)
            self.clfs[i] = clf

    def predict(self,X):
        return np.argmax( self.predict_proba(X), axis=1 )

    def predict_proba(self,X):
        predicted = [self.clfs[k].predict_proba(X)[:,1].reshape(-1,1) for k in self.clfs]

        p_x_first = 1-predicted[0]
        p_x_last  = predicted[-1]
        p_x_middle= [predicted[i] - predicted[i+1] for i in range(len(predicted) - 1)]
        
        probs = np.hstack([p_x_first, *p_x_middle, p_x_last])

        return probs

    def set_params(self, **params):
        self.clf.set_params(**params)
        for _,clf in self.clfs.items():
            clf.set_params(**params)

class OrderedModelPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, kwargs):
        """
        Custom pipeline that fits an OrderedModel using the predictions of a base estimator.
        :param estimator: The base estimator to generate predictions.
        :param kwargs: Additional parameters for OrderedModel.
        """
        self.estimator = estimator
        self.kwargs = kwargs.copy()
        self.is_fit = False

    def fit(self, X, y, **fit_params):
        """
        Train the OrderedModel using predictions from the base estimator.
        :param X: Features.
        :param y: Labels.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        estimator_predictions = self.estimator.predict(X)
        
        # Step 3: Train the OrderedModel using these predictions
        self.model_ = OrderedModel(y, estimator_predictions, **self.kwargs)
        self.model_ = self.model_.fit(**fit_params)
        
        self.is_fit = True
        return self

    def predict(self, X):
        """
        Predict class labels using the trained OrderedModel.
        :param X: Input features.
        :return: Predicted class labels.
        """
        if not self.is_fit:
            raise ValueError("Model not trained yet.")
        
        # Generate predictions from the base estimator
        estimator_predictions = self.estimator.predict(X)
        
        # Use the OrderedModel to predict using these predictions
        res = self.model_.predict(estimator_predictions)
        return np.argmax(res, axis=1)

    def predict_proba(self, X):
        """
        Return probability estimates for each class.
        :param X: Input features.
        :return: Probability matrix (rows = samples, columns = class probabilities).
        """
        if not self.is_fit:
            raise ValueError("Model not trained yet.")
        
        # Generate predictions from the base estimator
        estimator_predictions = self.estimator.predict(X)
        
        # Use the OrderedModel to predict probabilities using these predictions
        return self.model_.predict(estimator_predictions)

    def get_model(self):
        """
        Returns the trained OrderedModel.
        :return: OrderedModel instance.
        """
        if self.is_fit:
            return self.model_
        else:
            raise ValueError("Model not trained yet.")