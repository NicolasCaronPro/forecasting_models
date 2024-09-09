if __name__ == '__main__':
    import sys
    import os
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import logging
    import argparse
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

    logger.info('Handler pour afficher les logs dans le terminal')
    streamHandler = logging.StreamHandler(stream=sys.stdout)
    streamHandler.setFormatter(logFormatter)
    logger.addHandler(streamHandler)
    script_path = f'{os.path.dirname(os.path.abspath(__file__))}/../'
    logger.info(f'Ajouter le script pour retrouver les modules : {script_path}')
    sys.path.insert(0, script_path)

from src.models.sklearn_models import *
from src.models.loss import *

def get_model(model_type, name, device, task_type, test_metrics='log_loss', with_metric=None, params:dict=None) -> Union[Model, ModelTree]:
    """
    Returns the model and hyperparameter search grid based on the model name, task type, and device.

    :param type: Model type (xgboost, lightgbm, ngboost, svm, random_forest, decision_tree)
    :param name: Model name
    :param device: Device to use ('cpu' or 'cuda')
    :param task_type: Task type ('regression' or 'classification')
    :param params: Dictionary of model parameters
    :param loss: Specified loss Default is log_loss
    :return: Configured model and hyperparameter search grid
    """
    model = None

    if with_metric is not None:
        eval_metric = metrics[with_metric]

    if model_type == 'xgboost':
        params.update({'eval_metric': eval_metric})
        model = config_xgboost(device, task_type, params)
    elif model_type == 'lightgbm':
        model = config_lightGBM(device, task_type, params)
    elif model_type == 'ngboost':
        model = config_ngboost(device, task_type, params)
    elif model_type == 'svm':
        model = config_svm(device, task_type, params)
    elif model_type == 'rf':
        model = config_random_forest(device, task_type, params)
    elif model_type == 'dt':
        model = config_decision_tree(device, task_type, params)
    else:
        raise ValueError(f"Unrecognized model: {model_type}")
    
    # Check if the model is a tree-based model
    tree_based_models = (DecisionTreeClassifier, DecisionTreeRegressor, 
                         RandomForestClassifier, RandomForestRegressor, 
                         XGBClassifier, XGBRegressor, 
                         LGBMClassifier, LGBMRegressor, 
                         NGBClassifier, NGBRegressor)

    if isinstance(model, tree_based_models):
        return ModelTree(model, loss=test_metrics, name=name)
    else:
        return Model(model, loss=test_metrics, name=name)

def config_xgboost(device, task_type, params=None) -> Union[XGBRegressor, XGBClassifier]:
    """
    Returns a xgboost model define by params

    :param device : cpu or cuda
    :param task_type : regression or classification
    :params : config parameters
    """
    loss_metrics = {
        weighted_rmse: weighted_rmse_obj,
        percentiles_weighted_rmse: percentiles_weighted_rmse_obj,
        root_mean_squared_error: 'reg:squarederror',
        mean_squared_error: 'reg:squarederror',
        mean_absolute_error: 'reg:absoluteerror',
        root_mean_squared_log_error: 'reg:squaredlogerror',
        mean_squared_log_error: 'reg:squaredlogerror'
    }
    if params is None:
        # Random parametring
        params = {
            'verbosity':0,
            'early_stopping_rounds':15,
            'learning_rate' :0.01,
            'min_child_weight' : 5.0,
            'max_depth' : 6,
            'max_delta_step' : 1.0,
            'subsample' : 0.3,
            'colsample_bytree' : 0.8,
            'colsample_bylevel': 0.8,
            'reg_lambda' : 10.5,
            'reg_alpha' : 0.9,
            'n_estimators' : 10000,
            'random_state': 42,
            'tree_method':'hist',
        }

    # Check if eval_metric and objective are compatible
    if 'objective' in params:
        objective = params['objective']
        if 'eval_metric' not in params:
            for metric, obj in loss_metrics.items():
                if obj == objective:
                    params['eval_metric'] = metric
                    break
        else:
            if params['eval_metric'] not in loss_metrics or loss_metrics[params['eval_metric']] != objective:
                raise ValueError(f"The eval_metric '{params['eval_metric']}' is not compatible with the custom objective function.")
    elif 'eval_metric' in params:
        eval_metric = params['eval_metric']
        if eval_metric not in loss_metrics:
            raise ValueError(f"The eval_metric '{eval_metric}' is not compatible with the custom objective function.")
        params['objective'] = loss_metrics[eval_metric]
            
    if device == 'cuda':
        params['device']='cuda'

    if task_type == 'regression':
        return XGBRegressor(**params,
                            )
    else:
        return XGBClassifier(**params,
                            )

def config_lightGBM(device, task_type, params=None) -> Union[LGBMClassifier, LGBMRegressor]:
    """
    Returns a lightGBM model define by params

    :param device : cpu or cuda
    :param task_type : regression or classification
    :params 
    """
    if params is None:
        params = {
            'verbosity': -1,
            'learning_rate': 0.01,
            'early_stopping_rounds': 15,
            'bagging_fraction': 0.7,
            'colsample_bytree': 0.6,
            'max_depth': 4,
            'num_leaves': 16,
            'reg_lambda': 1,
            'reg_alpha': 0.27,
            'num_iterations': 10000,
            'random_state': 42
        }

    if device == 'cuda':
        params['device'] = 'gpu'

    if task_type == 'regression':
        return LGBMRegressor(**params)
    else:
        return LGBMClassifier(**params)

def config_ngboost(device, task_type, params=None) -> Union[NGBClassifier, NGBRegressor]:
    """
    Returns a lightGBM model define by params

    :param device : cpu or cuda
    :param task_type : regression or classification
    :params 
    """
    if params is None:
        params = {
            'natural_gradient': True,
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'minibatch_frac': 0.7,
            'col_sample': 0.6,
            'verbose': False,
            'verbose_eval': 100,
            'tol': 1e-4,
            'random_state': 42
        }

    if task_type == 'regression':
        return NGBRegressor(**params)
    else:
        return NGBClassifier(**params)

def config_svm(device, task_type, params=None) -> Union[SVC, SVR]:
    """
    Returns a svm model define by params

    :param device : cpu or cuda
    :param task_type : regression or classification
    :params 
    """
    if params is None:
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }

    if task_type == 'regression':
        return SVR(**params)
    else:
        return SVC(**params, probability=True)

def config_random_forest(device, task_type, params=None) -> Union[RandomForestClassifier, RandomForestRegressor]:
    """
    Returns a random_fores model define by params

    :param device : cpu or cuda
    :param task_type : regression or classification
    :params 
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'bootstrap': True,
            'random_state': 42
        }

    if task_type == 'regression':
        return RandomForestRegressor(**params)
    else:
        return RandomForestClassifier(**params)

def config_decision_tree(device, task_type, params=None) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
    """
    Returns a decision_tree model define by params

    :param device : cpu or cuda
    :param task_type : regression or classification
    :params 
    """
    if params is None:
        params = {
            'criterion': 'mse' if task_type == 'regression' else 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'random_state': 42
        }

    if task_type == 'regression':
        return DecisionTreeRegressor(**params)
    else:
        return DecisionTreeClassifier(**params)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    prog='Test code',
    description='',
    )
    parser.add_argument('-test_id', '--test_id', type=str, help='Test id')
    args = parser.parse_args()
    test_id = args.test_id

    check_and_create_path(Path('./Test'))

    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f'Test : {test_id}')
    if test_id == '1':
        ############################## XGBOOST TEST ###########################################
        params = {
                'objective':'reg:squarederror',
                'verbosity':0,
                'early_stopping_rounds':None,
                'learning_rate' :0.01,
                'min_child_weight' : 5.0,
                'max_depth' : 6,
                'max_delta_step' : 1.0,
                'subsample' : 0.3,
                'colsample_bytree' : 0.8,
                'colsample_bylevel': 0.8,
                'reg_lambda' : 10.5,
                'reg_alpha' : 0.9,
                'n_estimators' : 10000,
                'random_state': 42,
                'tree_method':'hist',
            }
        
        grid_params = {
                'max_depth': [1,2,5,10,15],
            }
        
        name = 'xgboost_test'
        task_type = 'regression'
        device='cpu'
        loss = 'rmse'
        check_and_create_path(Path('./Test') / name)

        logger.info(f'Configuration du modèle XGBoost pour la {task_type} sur : {device} avec une loss {loss}')
        model = get_model(model_type='xgboost', name=name, device=device, task_type=task_type, params=params, loss='rmse')

        logger.info('Fit du modèle avec les données d entraînement')
        model.fit(X_train, y_train, optimization='grid', grid_params=grid_params, fit_params={})

        logger.info('Prédiction avec les données de test')
        predictions = model.predict(X_test)

        logger.info('Si le modèle est de type arbre, tracer l arbre')
        if isinstance(model, ModelTree):
            features_name = [f"feature_{i}" for i in range(X_train.shape[1])]
            model.plot_tree(features_name=features_name, outname="xgboost", dir_output=Path("./Test") / name, figsize=(50,25))

        logger.info('Afficher les résultats')
        logger.info(f"Sample predictions: {predictions[:10]}")

        logger.info('Afficher l importance des caractéristiques')
        model.plot_features_importance(X_train, y_train, features_name, "xgboost_importance", Path("./Test") / name)

        param_test = 'max_depth'
        logger.info(f'Afficher l influence d un paramètre {param_test}')
        if hasattr(model, 'cv_results_') and model.cv_results_ is not None:
            model.plot_param_influence(param_test, Path("./Test") / name)

    elif test_id == '2':
    ################################ FUSION ########################################
        name = 'fusion_mode'
        check_and_create_path(Path('./Test') / name)
        # Usage example:
        logger.info('Initialize base models')
        xgb_model = Model(XGBRegressor(), loss='rmse', name='xgb')
        lgb_model = Model(LGBMRegressor(), loss='rmse', name='lgb')
        ngb_model = Model(NGBRegressor(Dist=Normal, Score=LogScore), loss='rmse', name='ngb')

        logger.info('Initialize fusion model')
        fusion_model = ModelFusion([xgb_model, lgb_model, ngb_model], LinearRegression(), loss='rmse', name='fusion_rf')

        logger.info('Fit the fusion model')
        fusion_model.fit([X_train, X_train, X_train], [y_train, y_train, y_train], y_train,
                        optimization_list='skip', 
                        grid_params_list=None)
        
        predictions = fusion_model.predict([X_test, X_test, X_test])
        plt.figure(figsize=(15,15))
        plt.plot(y, label='y')
        plt.plot(predictions, label='predictions')
        plt.legend()
        plt.savefig(Path('./Test') / name / 'predictions.png')
        
        features_name = [f"feature_{i}" for i in range(X_train.shape[1])]

        logger.info('Afficher l importance des caractéristiques')
        fusion_model.plot_features_importance(X_set=[X_train, X_train, X_train], y_set=y_train, names=features_name,
                                              outname="train_importance", dir_output=Path("./Test") / name, mode='bar', figsize=(50,25))

        logger.info('Afficher l importance des caractéristiques de chaque modèle')
        fusion_model.plot_features_importance_list(X_list=[X_train, X_train, X_train], y_list=[y_train, y_train, y_train],
                                                   names_list=[features_name, features_name, features_name], outname="train", dir_output=Path("./Test") / name,
                                                   mode='bar', figsize=(50,25))
    else:
        raise ValueError(f'{test_id} Unknowed')
