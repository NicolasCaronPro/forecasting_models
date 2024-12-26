from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
import os
import math
import numpy as np
#import shap
from forecasting_models.pytorch.models import *
from forecasting_models.pytorch.models_2D import *
from forecasting_models.pytorch.kan import *
import copy

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

"""def explore_features(model,
                    features,
                    df_train,
                    df_val,
                    df_test,
                    weight_col,
                    target):
    
    features_importance = []
    selected_features_ = []
    base_score = -math.inf
    count_max = 70
    c = 0
    for i, fet in enumerate(features):
        
        selected_features_.append(fet)

        X_train_single = df_train[selected_features_]

        fitparams={
                'eval_set':[(X_train_single, df_train[target]), (df_val[selected_features_], df_val[target])],
                'sample_weight' : df_train[weight_col],
                'verbose' : False
                }

        model.fit(X=X_train_single, y=df_train[target], fit_params=fitparams)

        # Calculer le score avec cette seule caractéristique
        single_feature_score = model.score(df_test[selected_features_], df_test[target], sample_weight=df_test[weight_col])

        # Si le score ne s'améliore pas, on retire la variable de la liste
        if single_feature_score <= base_score:
            selected_features_.pop(-1)
            c += 1
        else:
            print(f'With {fet} number {i}: {base_score} -> {single_feature_score}')
            base_score = single_feature_score
            c = 0

        if c > count_max:
            print(f'Score didn t improove for {count_max} features, we break')
            break
        features_importance.append(single_feature_score)

    return selected_features_"""

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def shapley_additive_explanation_neural_network(model, df_set, outname, dir_output, mode = 'bar', figsize=(50,25), samples=None, samples_name=None):
    try:
        explainer = shap.DeepExplainer(model)
        shap_values = explainer(df_set)
        plt.figure(figsize=figsize)
        if mode == 'bar':
            shap.plots.bar(shap_values, show=False, max_display=20)
        elif mode == 'beeswarm':
            shap.plots.beeswarm(shap_values, show=False, max_display=20)
        else:
            raise ValueError(f'Unknow {mode} mode')
        
        shap_values_abs = np.abs(shap_values.values).mean(axis=0)  # Importance moyenne absolue des SHAP values
        top_features_indices = np.argsort(shap_values_abs)[-10:]  # Indices des 10 plus importantes
        top_features_ = df_set.columns[top_features_indices].tolist()  # Noms des 10 features
            
        
        plt.tight_layout()
        plt.savefig(dir_output / f'{outname}_shapley_additive_explanation.png')
        plt.close('all')
        if samples is not None and samples_name is not None:
            for i, sample in enumerate(samples):
                plt.figure(figsize=(30,15))
                shap.plots.force(shap_values[sample], show=False, matplotlib=True, text_rotation=45, figsize=(30,15))
                plt.tight_layout()
                plt.savefig(dir_output / 'sample' / f'{outname}_{samples_name[i]}_shapley_additive_explanation.png')
                plt.close('all')

    except Exception as e:
        print(f'Error {e} with shapley_additive_explanation')
        if 'top_features_' in locals():
            return top_features_
        else:
            return None
    return top_features_


def make_model(model_name, in_dim, in_dim_2D, graph, dropout, act_func, k_days, task_type, device, num_lstm_layers, out_channels, custom_model_params=None):
    """
    Renvoie un tuple contenant le modèle spécifié et les paramètres utilisés pour sa création.
    
    Parameters:
    - model_name: Nom du modèle à retourner.
    - in_dim: Dimensions d'entrée.
    - in_dim_2D: Dimensions d'entrée pour les données 2D (si applicable).
    - scale: Échelle (peut être utilisé pour la normalisation).
    - dropout: Taux de dropout pour le modèle.
    - act_func: Fonction d'activation
    - k_days: Nombre de séquences/jours pour la prédiction.
    - task_type: Indicateur pour un problème de classification binaire.
    - device: Appareil sur lequel exécuter le modèle (CPU ou GPU).
    - num_lstm_layers: Nombre de couches LSTM (si applicable).
    - out_channels: Nombre de canaux de sortie.
    - custom_model_params: Dictionnaire des paramètres spécifiques du modèle (facultatif).
    
    Returns:
    - Tuple (modèle, paramètres) où 'modèle' est le modèle spécifié et 'paramètres' est un dictionnaire des paramètres utilisés.
    """
    scale = graph.scale
    graph_or_node = graph.graph_method

    model_params = {
        'model_name': model_name,
        'in_dim': in_dim,
        'in_dim_2D': in_dim_2D,
        'scale': scale,
        'graph_or_node': graph_or_node,
        'dropout': dropout,
        'act_func': act_func,
        'k_days': k_days,
        'task_type': task_type,
        'device': device,
        'num_lstm_layers': num_lstm_layers,
        'out_channels': out_channels,  # Added the out_channels parameter
    }

    if model_name == 'GAT':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [128, 256, 512],
            'out_channels' : out_channels,
            'end_channels': 64,
            'heads': [6, 4, 2],
            'dropout': dropout,
            'bias': True,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = GAT(
            in_dim=default_params['in_dim'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            heads=default_params['heads'],
            dropout=default_params['dropout'],
            bias=default_params['bias'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            n_sequences=default_params['n_sequences'],
            task_type=default_params['task_type'],
            out_channels=default_params['out_channels']
        )
        model_params.update(default_params)

    elif model_name == 'GCN':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [128, 256, 512],
            'out_channels' : out_channels,
            'end_channels': 64,
            'dropout': dropout,
            'bias': True,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type,
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = GCN(
            in_dim=default_params['in_dim'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            dropout=default_params['dropout'],
            bias=default_params['bias'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            n_sequences=default_params['n_sequences'],
            task_type=default_params['task_type'],
            graph_or_node=graph_or_node,
            out_channels=default_params['out_channels']
        )
        model_params.update(default_params)

    elif model_name == 'DST-GCN':
        default_params = {
            'in_channels': in_dim,
            'dilation_channels': [64, 128, 256, 512],
            'dilations': [1, 1, 3],
           'out_channels' : out_channels,
            'end_channels': 64,
            'dropout': dropout,
            'act_func': act_func,
            'device': device,
            'task_type': task_type,
            'n_sequences': k_days + 1
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = DSTGCN(
            n_sequences=default_params['n_sequences'],
            in_channels=default_params['in_channels'],
            end_channels=default_params['end_channels'],
            dilation_channels=default_params['dilation_channels'],
            dilations=default_params['dilations'],
            dropout=default_params['dropout'],
            act_func=default_params['act_func'],
            device=default_params['device'],
            task_type=default_params['task_type'],
            graph_or_node=graph_or_node,
            out_channels=default_params['out_channels']
        )
        model_params.update(default_params)

    elif model_name == 'LSTM':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': [in_dim, in_dim, in_dim],
           'out_channels' : out_channels,
            'end_channels': 64,
            'n_sequences': k_days + 1,
            'num_layers': 3,
            'device': device,
            'act_func': act_func,
            'task_type': task_type,
            'dropout': dropout
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = LSTM(
            in_channels=default_params['in_channels'],
            hidden_channels_list=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            n_sequences=default_params['n_sequences'],
            num_layers=default_params['num_layers'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            task_type=default_params['task_type'],
            dropout=default_params['dropout'],
            out_channels=default_params['out_channels']
        )
        model_params.update(default_params)

    elif model_name == 'DST-GAT':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [128, 256, 512],
            'out_channels' : out_channels,
            'end_channels': 64,
            'heads': [6, 4, 2],
            'dropout': dropout,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = DSTGAT(
            n_sequences=default_params['n_sequences'],
            in_channels=default_params['in_channels'],
            end_channels=default_params['end_channels'],
            dilation_channels=default_params['dilation_channels'],
            dilations=default_params['dilations'],
            dropout=default_params['dropout'],
            act_func=default_params['act_func'],
            device=default_params['device'],
            task_type=default_params['task_type'],
            heads=default_params['heads'],
            graph_or_node = graph_or_node,
            out_channels=default_params['out_channels']
        )
        model_params.update(default_params)

    elif model_name == 'ST-GAT':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [128, 256, 512],
            'out_channels' : out_channels,
            'end_channels': 64,
            'heads': [6, 4, 2],
            'dropout': dropout,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = STGAT(
            in_channels=default_params['in_dim'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            heads=default_params['heads'],
            dropout=default_params['dropout'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            n_sequences=default_params['n_sequences'],
            task_type=default_params['task_type'],
            out_channels=default_params['out_channels']
        )
        model_params.update(default_params)

    elif model_name == 'Zhang':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': [64, 128],
            'end_channels': out_channels,  # Use out_channels for end_channels
            'num_layers': 2,
            'device': device,
            'act_func': act_func,
            'task_type': task_type,
            'dropout': dropout
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = Zhang(
            in_channels=default_params['in_channels'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            num_layers=default_params['num_layers'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            task_type=default_params['task_type'],
            dropout=default_params['dropout']
        )
        model_params.update(default_params)

    elif model_name == 'UNet':
        default_params = {
            'in_channels': in_dim,
            'out_channels': out_channels,  # Use out_channels for the final layer
            'dropout': dropout,
            'device': device,
            'act_func': act_func,
            'task_type': task_type
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = UNet(
            in_channels=default_params['in_channels'],
            out_channels=default_params['out_channels'],
            dropout=default_params['dropout'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            task_type=default_params['task_type']
        )
        model_params.update(default_params)

    else:
        raise ValueError(f"Modèle '{model_name}' non reconnu.")
    
    return model, model_params

