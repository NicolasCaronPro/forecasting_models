from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
import os
import math
import numpy as np

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

    return np.asarray(selected_features_)

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()