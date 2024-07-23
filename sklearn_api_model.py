from forecasting_models.sklearn_api_models_config import *
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

def save_object(obj, filename: str, path: Path):
    """
    Sauvegarde un objet en utilisant pickle dans un fichier donné.
    
    Parameters:
    - obj: L'objet à sauvegarder.
    - filename: Le nom du fichier.
    - path: Le chemin du répertoire où sauvegarder le fichier.
    """
    # Vérifie et crée le chemin s'il n'existe pas
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Sauvegarde l'objet en utilisant pickle
    with open(path / filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def explore_parameters(model,
                       features : np.array,
                       X : np.array, y : np.array, w : np.array,
                       parameter_name : str,
                    values : list,
                    fitparams  : dict = None,
                    X_val=None, y_val=None, w_val=None,
                    X_test=None, y_test=None, w_test=None):
    
    parameter_loss = []
    base_score = -math.inf
    for i, val in enumerate(values):

        params = model.get_params()
        params[parameter_name] = val
        model.set_params(**params)

        model.fit(X=X[:, features], y=y, fit_params=fitparams)

        single_feature_score = model.score(X_test[:, features], y_test, sample_weight=w_test)

        print(f'Parame {parameter_name} at {val} : {base_score} -> {single_feature_score}')
        base_score = single_feature_score

        parameter_loss.append(single_feature_score)

    res = np.row_stack(values, parameter_loss)

    return res

def explore_features(model, features : np.array, X : np.array, y : np.array, w : np.array,
                          X_val=None, y_val=None, w_val=None,
                          X_test=None, y_test=None, w_test=None):
    
    features_importance = []
    selected_features_ = []
    base_score = -math.inf
    for i, fet in enumerate(features):
        
        selected_features_.append(fet)

        X_train_single = X[:, selected_features_]

        fitparams={
                'eval_set':[(X_train_single, y), (X_val[:, selected_features_], y_val)],
                'sample_weight' : w,
                'verbose' : False
                }

        model.fit(X=X_train_single, y=y, fit_params=fitparams)

        # Calculer le score avec cette seule caractéristique
        single_feature_score = model.score(X_test[:, selected_features_], y_test, sample_weight=w_test)

        # Si le score ne s'améliore pas, on retire la variable de la liste
        if single_feature_score <= base_score:
            selected_features_.pop(-1)
        else:
            print(f'With {fet} : {base_score} -> {single_feature_score}')
            base_score = single_feature_score

        features_importance.append(single_feature_score)

    return selected_features_

class Model(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model, loss='log_loss', name='Model'):
        """
        Initialize the CustomModel class.
        
        Parameters:
        - model: The base model to use (must follow the sklearn API).
        - name: The name of the model.
        - loss: Loss function to use ('log_loss', 'hinge_loss', etc.).
        """
        self.best_estimator_ = model
        self.name = name
        self.loss = loss
        self.X_train = None
        self.y_train = None
        self.cv_results_ = None  # Adding the cv_results_ attribute

    def fit(self, X, y, optimization='skip', param_grid=None, fit_params=None):
        """
        Train the model on the data using GridSearchCV or BayesSearchCV.
        
        Parameters:
        - X: Training data.
        - y: Labels for the training data.
        - param_grid: Parameters to optimize.
        - optimization: Optimization method to use ('grid' or 'bayes').
        - fit_params: Additional parameters for the fit function.
        """
        self.X_train = X
        self.y_train = y

        # Train the final model with all selected features
        if optimization == 'grid':
            assert param_grid is not None
            grid_search = GridSearchCV(self.best_estimator_, param_grid, scoring=self._get_scorer(), cv=5)
            grid_search.fit(X, y, **fit_params)
            self.best_estimator_ = grid_search.best_estimator_
            self.cv_results_ = grid_search.cv_results_
        elif optimization == 'bayes':
            assert param_grid is not None
            param_list = []
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    param_list.append((param_name, param_values))
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    param_list.append((param_name, param_values))
                else:
                    raise ValueError("Unsupported parameter type in param_grid. Expected list or tuple of size 2.")
                
            # Configure the parameter space for BayesSearchCV
            param_space = {}
            for param_name, param_range in param_list:
                if isinstance(param_range[0], int):
                    param_space[param_name] = Integer(param_range[0], param_range[-1])
                elif isinstance(param_range[0], float):
                    param_space[param_name] = Real(param_range[0], param_range[-1], prior='log-uniform')
                
            opt = Optimizer(param_space, base_estimator='GP', acq_func='gp_hedge')
            bayes_search = BayesSearchCV(self.best_estimator_, opt, scoring=self._get_scorer(), cv=5)
            bayes_search.fit(X, y, **fit_params)
            self.best_estimator_ = bayes_search.best_estimator_
            self.cv_results_ = bayes_search.cv_results_
        elif optimization == 'skip':
            self.best_estimator_.fit(X, y, **fit_params)
        else:
            raise ValueError("Unsupported optimization method")
        
    def predict(self, X):
        """
        Predict labels for input data.
        
        Parameters:
        - X: Data to predict labels for.
        
        Returns:
        - Predicted labels.
        """
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities for input data.
        
        Parameters:
        - X: Data to predict probabilities for.
        
        Returns:
        - Predicted probabilities.
        """
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError("The chosen model does not support predict_proba.")

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance.
        
        Parameters:
        - X: Input data.
        - y: True labels.
        - sample_weight: Sample weights.
        
        Returns:
        - The model's score on the provided data.
        """
        y_pred = self.predict(X)
        if self.loss == 'log_loss':
            proba = self.predict_proba(X)
            return -log_loss(y, proba)
        elif self.loss == 'hinge_loss':
            return hinge_loss(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'accuracy':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'mse':
            return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
        elif self.loss == 'rmse':
            return -math.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight))
        elif self.loss == 'rmsle':
            return -math.sqrt(mean_squared_log_error(y, y_pred, sample_weight=sample_weight))
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def get_params(self, deep=True):
        """
        Get the model's parameters.
        
        Parameters:
        - deep: If True, return the parameters for this model and nested models.
        
        Returns:
        - Dictionary of parameters.
        """
        params = {'model': self.best_estimator_, 'loss': self.loss, 'name': self.name}
        if deep and hasattr(self.best_estimator_, 'get_params'):
            deep_params = self.best_estimator_.get_params(deep=True)
            params.update(deep_params)
        return params

    def set_params(self, **params):
        """
        Set the model's parameters.
        
        Parameters:
        - params: Dictionary of parameters to set.
        
        Returns:
        - Self.
        """
        best_estimator_params = {}
        for key, value in params.items():
            if key in ['model', 'loss', 'name']:
                setattr(self, key, value)
            else:
                best_estimator_params[key] = value
        
        if best_estimator_params != {}:
            self.best_estimator_.set_params(**best_estimator_params)
        
        return self
    
    def _get_scorer(self):
        """
        Return the scoring function based on the chosen loss function.
        """
        if self.loss == 'log_loss':
            return 'neg_log_loss'
        elif self.loss == 'hinge_loss':
            return 'hinge'
        elif self.loss == 'accuracy':
            return 'accuracy'
        elif self.loss == 'rmse':
            return 'neg_root_mean_squared_error'
        elif self.loss == 'rmsle':
            return 'neg_root_mean_squared_log_error'
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        
    def _plot_features_importance(self, X_set, y_set, names, outname, dir_output, mode = 'bar'):
        """
        Display the importance of features using feature permutation.
        
        Parameters:
        - X_set: Data to evaluate feature importance.
        - y_set: Corresponding labels.
        - names: Names of the features.
        - outname : Name of the test set
        - dir_output: Directory to save the plot.
        - mode : moustache or bar.
        """
        result = permutation_importance(self.best_estimator_, X_set, y_set, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1]
        
        if mode == 'bar':
            plt.figure(figsize=(50,25))
            plt.title(f"Permutation importances {self.name}")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [names[i] for i in indices], rotation=90)
            plt.xlim([-1, len(importances)])
            plt.ylabel(f"Decrease in {self._get_scorer()} score")
            plt.tight_layout()
            plt.savefig(Path(dir_output) / f"{outname}_permutation_importances_{mode}.png")
            plt.close('all')
        elif mode == 'moustache':
            plt.figure(figsize=(50,25))
            plt.boxplot(importances[indices].T, vert=False, whis=10)
            plt.title(f"Permutation Importances {self.name}")
            plt.axvline(x=0, color="k", linestyle="--")
            plt.xlabel(f"Decrease in {self._get_scorer()} score")
            plt.tight_layout()
            plt.savefig(Path(dir_output) / f"{outname}_permutation_importances_{mode}.png")
            plt.close('all')
        else:
            raise ValueError(f'Unknown {mode} for ploting features importance but feel free to add new one')

    def _plot_param_influence(self, param, dir_output):
        """
        Display the influence of parameters on model performance.
        
        Parameters:
        - param: The parameter to visualize.
        - dir_output: Directory to save the plot.
        """
        if self.cv_results_ is None:
            raise AttributeError("Grid search or bayes search results not available. Please run GridSearchCV or BayesSearchCV first.")
        
        if param not in self.cv_results_['params'][0]:
            raise ValueError(f"The parameter {param} is not in the grid or bayes search results.")
        
        param_values = [result[param] for result in self.cv_results_['params']]
        means = self.cv_results_['mean_test_score']
        stds = self.cv_results_['std_test_score']

        plt.figure(figsize=(25,25))
        plt.title(f"Influence of {param} on performance for {self.name}")
        plt.xlabel(param)
        plt.ylabel("Mean score")
        plt.errorbar(param_values, means, yerr=stds, fmt='-o')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(dir_output) / f"{self.name}_{param}_influence.png")
        plt.close('all')