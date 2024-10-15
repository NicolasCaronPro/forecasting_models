gpu = False

import sys
from typing import List
if gpu:
    import cudf as pd
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import Lasso
    from cuml.linear_model import LinearRegression
    from cuml.metrics import mean_squared_error, mean_absolute_error
    from cuml.metrics.regression import r2_score
    from cuml.model_selection import GridSearchCV
    from cuml.model_selection import train_test_split
    from cuml.preprocessing import MinMaxScaler
else:
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import MinMaxScaler


from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
import logging
import math
from src.models.sklearn_models import Model


def get_features(df: pd.DataFrame, variables: List[str], target:str='appels', num_feats:int=100, learn_mode = 'slow', logger=None):
    """
    Selects relevant features from a DataFrame for a given target variable using various statistical and machine learning methods.
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the features and target variable.
    variables : list
        List of variable names to consider for feature selection.
    target : str, optional
        The name of the target variable. Default is 'appels'.
    num_feats : int, optional
        The maximum number of features to select. Default is 100.
    learn_mode : str, optional
        The learning mode, which can affect the feature selection process. Default is 'slow'.
    logger : logging.Logger, optional
        Logger for logging information during the feature selection process. Default is None.
    Returns:
    --------
    list
        A list of tuples containing the selected features and their counts, sorted by relevance.
    """

    logger.info(f"=> Recherche des {num_feats} features les + pertinentes")
    logger.info("   - Grande variance")
    variances = df.var()
    #high_variance = list(variances[variances > 0.5].index)
    high_variance = list(variances[variances > 0.5].index.values)
    if target in high_variance:
        high_variance.remove(target)
        
    logger.info("   - Coefficient de corrélation de Pearson")
    correlations = sorted([(k, abs(df[k].corr(df[target]))) for k in df.columns if k not in [target]], key=lambda x:x[1], reverse=True)
    #correlations = sorted([(k, abs(np.corrcoef(df[k], df[target])[0,1])) for k in df.columns if k not in [target]], key=lambda x:x[1], reverse=True)
    Liste_Pearson = [u[0] for u in correlations if u[1] >= 0.4]
    if not gpu:
        logger.info("   - Coefficient de corrélation de Kendall")
        method = 'kendall'
        correlations = sorted([(k, abs(df[[k, target]].corr(method=method).values[0,1])) for k in df.columns 
                                   if k not in [target, 'date']], key=lambda x:x[1], reverse=True)
        Liste_Kendall = [u[0] for u in correlations if u[1] >= 0.4]
    else:
        Liste_Kendall = []
    if True:
        logger.info("   - Coefficient de corrélation de Spearman")
        method = 'spearman'
        correlations = sorted([(k, abs(df[[k, target]].corr(method=method).values[0,1])) for k in df.columns 
                                   if k not in [target, 'date']], key=lambda x:x[1], reverse=True)
        Liste_Spearman = [u[0] for u in correlations if u[1] >= 0.4]
    else:
        Liste_Spearman = []
    X = df.drop(target, axis=1)
    y = df[target]
    if not gpu:
        logger.info("   - Random forests")
        dico = dict(n_estimators=100, max_depth=7)
        dico.update(n_jobs=-1)
        embeded_rf_selector = SelectFromModel(
                        RandomForestRegressor(**dico),
                        max_features=num_feats)
        embeded_rf_selector.fit(X, y)
        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
    else:
        embeded_rf_feature = []
    logger.info("   - Régression linéaire")
    embeded_lr_selector = SelectFromModel(LinearRegression(),
                                          max_features=num_feats)
    embeded_lr_selector.fit(X, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    # Convertissez la liste de booléens en Series cuDF
    #embeded_lr_support_series = pd.Series(embeded_lr_support.to_arrow().to_pylist())
    embeded_lr_support_series = pd.Series(embeded_lr_support).reset_index(drop=True)
    # Obtenez les indices où embeded_lr_support_series est True
    true_indices = embeded_lr_support_series[embeded_lr_support_series.values.nonzero()[0]].index.values
    #true_indices = embeded_lr_support_series.to_pandas().values.nonzero()[0].tolist()
    # Obtenez les noms de colonne correspondant à ces indices
    logger.info(true_indices)
    embeded_lr_feature = [X.columns[i] for i in true_indices]
    #embeded_lr_feature = []
    #if not gpu:
    """print("   - Chi-2 selector")
    X_norm = MinMaxScaler().fit_transform(X[variables])
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    #chi_feature = X.loc[:, chi_support].columns.tolist()
    chi_feature = [item for index, item in enumerate(X.columns) if chi_support[index]]
    #else:"""
    #    chi_feature = []
    if not gpu:
        logger.info("   - Extra trees")
        model = ExtraTreesRegressor(n_jobs=-1, max_depth=7)
        model.fit(X,y)
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        xt_features = list(feat_importances.nlargest(num_feats).index)
    else:
        xt_features = []
    #if learn_mode != 'fastest':
    if True:
        dico_xgb = {'verbosity':0,
        'learning_rate':0.1,
        'objectif':'reg:squarederror',
        'early_stopping_rounds':15,
        'n_estimators' : 100000,
        'random_state': 42,
        }
        logger.info("   - XGBoost")
        train_set, val_set = train_test_split(df, test_size=0.2)
        X_train = train_set.drop(target, axis=1)
        X_val = val_set.drop(target, axis=1)
        y_train = train_set[target]
        y_val = val_set[target]
        reg = XGBRegressor(**dico_xgb)
        reg.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)
        a, b = X_train.columns, reg.feature_importances_
        res_xgb = sorted([(a[k], b[k]) for k in range(len(X.columns))], key=lambda x:x[1], reverse=True)
        features_xgb = [k[0] for k in res_xgb[:num_feats]]
    else:
        features_xgb = []
    if True:
        logger.info("   - LightGBM")
        #lightgbm.basic.LightGBMError: GPU Tree Learner was not enabled in this build.
        #Please recompile with CMake option -DUSE_GPU=1
        '''dico = dict(n_estimators=500, max_depth=7, num_leaves=2**7, verbose=-1)
        if gpu:
            dico.update(device='gpu',
                        gpu_platform_id = 0,
                        gpu_device_id = 0)
        else:
            dico.update(n_jobs=-1)
        '''
        dico = {
            'learning_rate': 0.1,
            'objective': 'regression',
            'metric': 'rmse',
            'num_iterations': 100000,
            'max_depth': 7,
            'num_leaves': 2**7,
            'verbose': -1,
            'early_stopping_round':10
        }
        
        lgbc = LGBMRegressor(**dico)
        embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
        
        train_set, val_set = train_test_split(df, test_size=0.2)
        X_train = train_set.drop(target, axis=1)
        X_val = val_set.drop(target, axis=1)
        y_train = train_set[target]
        y_val = val_set[target]
        embeded_lgb_selector.fit(X_train[variables], y_train, 
                eval_set=(X_val[variables], y_val))
        #embeded_lgb_selector.fit(X.to_pandas(), y.to_pandas())
        embeded_lgb_support = embeded_lgb_selector.get_support()
        embeded_lgb_feature = [item for index, item in enumerate(X_train.columns) if embeded_lgb_support[index]]
    else:
        embeded_lgb_feature = []
    #loc[:, embeded_lgb_support].columns.tolist()
    if not gpu:
        logger.info("   - Lasso")
        rfe_selector = RFE(estimator=Lasso(alpha=0.1),
                       n_features_to_select=num_feats,
                       step=100, verbose=5 if logger.level == logging.INFO else 0)
        rfe_selector.fit(X, y)
        rfe_support = rfe_selector.get_support()
        rfe_feature = X.loc[:,rfe_support].columns.tolist()
    else:
        rfe_feature = []
    total = high_variance+Liste_Pearson+Liste_Kendall+Liste_Spearman+embeded_lr_feature+embeded_rf_feature+xt_features+features_xgb+embeded_lgb_feature+rfe_feature
    total = sorted(set([(k, total.count(k)) for k in total if total.count(k)>1]), key=lambda x:x[1], reverse=True)[:num_feats]
    return total
    final = [k for k in total if k[0]=='en_cours']
    final.extend([k for k in total if 'en_cours_' in k[0]])
    final.extend([k for k in total if k[0] in [f'interv-{k}' for k in range(1,6)]])
    final.extend([k for k in total if k[0] in [f'appels-{k}' for k in range(1,6)]])
    final.extend([k for k in total if k[0] in [f'appels_somme-{k}' for k in range(1,6)]])
    final.extend([k for k in total if k[0] in ['hour', 'dayofweek', 'dayofyear']])
    final.extend([k for k in total if 'interv-' not in k[0] 
        and 'appels-' not in k[0] and k[0] not in [u[0] for u in final]])
    final.extend([k for k in total if k[0] not in [u[0] for u in final]])
    return final


def explore_features(model: Model, model_config:dict, features: List[str], df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, target: str, weight_col: str = None, logger:logging.Logger = None, preselection: List[str] = []) -> List[str]:
    """
    Explore and select the most important features for a given model.
    This function iteratively adds features to the model and evaluates their importance
    based on the model's performance. If adding a feature does not improve the model's
    score, it is removed from the list of selected features. The process stops if the 
    score does not improve for a specified number of consecutive features.
    Parameters:
    -----------
    model : object
        The machine learning model to be used for feature selection. The model should 
        have a `fit` method and a `score` method.
    features : list
        A list of feature names to be evaluated.
    df_train : pandas.DataFrame
        The training dataset containing the features and target variable.
    df_val : pandas.DataFrame
        The validation dataset used to evaluate the model during training.
    df_test : pandas.DataFrame
        The test dataset used to evaluate the model's performance.
    weight_col : str
        The name of the column in the datasets that contains sample weights.
    target : str
        The name of the target variable column in the datasets.
    Returns:
    --------
    selected_features_ : list
        A list of selected features that improve the model's performance.
    """
    
    logger.info("=> Exploration des features")
    features_importance = []
    selected_features_ = preselection
    base_score = math.inf
    count_max = math.inf
    c = 0


    for i, fet in enumerate(features):
        
        selected_features_.append(fet)

        X_train_single = df_train[selected_features_]

        fitparams={
                'eval_set':[(X_train_single, df_train[target]), (df_val[selected_features_], df_val[target])],
                # 'sample_weight' : df_train[weight_col],
                'verbose' : False
                }
        
        model_config['fit_params'].update({'eval_set': [(X_train_single, df_train[target]), (df_val[selected_features_], df_val[target])]})

        model.fit(X=X_train_single, y=df_train[target], fit_params=fitparams)

        # Calculer le score avec cette seule caractéristique
        single_feature_score = model.score(df_test[selected_features_], df_test[target]) # , sample_weight=df_test[weight_col]

        # Si le score ne s'améliore pas, on retire la variable de la liste
        if single_feature_score <= base_score:
            multiple_features_score = model.score(df_test[selected_features_], df_test[target], single_score=False)
            logger.info(f'{model.loss} With {fet} number {i}: {base_score} -> {single_feature_score}\n' +
                        "\t".join(f"{metric, value}" for metric, value in multiple_features_score.items()))
            base_score = single_feature_score
            c = 0
        else:
            selected_features_.pop(-1)
            c += 1


        if c > count_max:
            logger.warning(f'Score didn t improve for {count_max} features, we break')
            break
        features_importance.append(single_feature_score)

    return selected_features_
