from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
import os
import math
import numpy as np
import shap
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

def make_model(model_name, in_dim, in_dim_2D, scale, dropout, act_func, k_days, binary, device, num_lstm_layers, custom_model_params=None):
    """
    Renvoie un tuple contenant le modèle spécifié et les paramètres utilisés pour sa création.
    
    Parameters:
    - model_name: Nom du modèle à retourner.
    - in_dim: Dimensions d'entrée.
    - in_dim_2D: Dimensions d'entrée pour les données 2D (si applicable).
    - scale: Échelle (peut être utilisé pour la normalisation).
    - dropout: Taux de dropout pour le modèle.
    - act_func: Fonction d'activation.
    - k_days: Nombre de séquences/jours pour la prédiction.
    - binary: Indicateur pour un problème de classification binaire.
    - device: Appareil sur lequel exécuter le modèle (CPU ou GPU).
    - num_lstm_layers: Nombre de couches LSTM (si applicable).
    - custom_model_params: Dictionnaire des paramètres spécifiques du modèle (facultatif).
    
    Returns:
    - Tuple (modèle, paramètres) où 'modèle' est le modèle spécifié et 'paramètres' est un dictionnaire des paramètres utilisés.
    """

    shape2D = {
        10: [24, 24],
        30: [30, 30],
        3: [15, 15],
        8: [30, 30],
        'departement': [64, 64],
    }
    
    zhang_layer_conversion = {  # Fire project, if you try this model you need to adapt it scale -> fc dim
        8: 15,
        10: 12,
        9: 15,
    }

    model_params = {
        'model_name': model_name,
        'in_dim': in_dim,
        'in_dim_2D': in_dim_2D,
        'scale': scale,
        'dropout': dropout,
        'act_func': act_func,
        'k_days': k_days,
        'binary': binary,
        'device': device,
        'num_lstm_layers': num_lstm_layers
    }

    if model_name == 'GAT':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [64, 128, 128, 128],
            'end_channels': 64,
            'heads': [1, 4, 4, 4],
            'dropout': dropout,
            'bias': True,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'binary': binary
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
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'GCN':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [64, 128, 128],
            'end_channels': 64,
            'dropout': dropout,
            'bias': True,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'binary': binary
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
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'DST-GCN':
        default_params = {
            'in_channels': in_dim,
            'dilation_channels': [128, 256, 512, 256, 128],
            'dilations': [1, 1, 2, 1, 1],
            'end_channels': 64,
            'dropout': dropout,
            'act_func': act_func,
            'device': device,
            'binary': binary,
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
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'ST-GAT':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': [256, 64],
            'end_channels': 64,
            'heads': 6,
            'dropout': dropout,
            'act_func': act_func,
            'device': device,
            'binary': binary,
            'n_sequences': k_days + 1
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = STGAT(
            n_sequences=default_params['n_sequences'],
            in_channels=default_params['in_channels'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            heads=default_params['heads'],
            dropout=default_params['dropout'],
            act_func=default_params['act_func'],
            device=default_params['device'],
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'ST-GCN':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': [64, 128, 256, 512, 256, 128],
            'end_channels': in_dim // 2,
            'dropout': dropout,
            'act_func': act_func,
            'device': device,
            'binary': binary,
            'n_sequences': k_days + 1
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = STGCN(
            n_sequences=default_params['n_sequences'],
            in_channels=default_params['in_channels'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            dropout=default_params['dropout'],
            act_func=default_params['act_func'],
            device=default_params['device'],
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'SDT-GCN':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels_temporal': [256, 64],
            'hidden_channels_spatial': [256, 64],
            'end_channels': 64,
            'dilations': [1],
            'dropout': dropout,
            'act_func': act_func,
            'device': device,
            'binary': binary,
            'n_sequences': k_days + 1
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = SDSTGCN(
            n_sequences=default_params['n_sequences'],
            in_channels=default_params['in_channels'],
            hidden_channels_temporal=default_params['hidden_channels_temporal'],
            dilations=default_params['dilations'],
            hidden_channels_spatial=default_params['hidden_channels_spatial'],
            end_channels=default_params['end_channels'],
            dropout=default_params['dropout'],
            act_func=default_params['act_func'],
            device=default_params['device'],
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'ATGN':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': 64,
            'out_channels': 64,
            'n_sequences': k_days + 1,
            'device': device,
            'act_func': act_func,
            'dropout': dropout,
            'binary': binary
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = TemporalGNN(
            in_channels=default_params['in_channels'],
            hidden_channels=default_params['hidden_channels'],
            out_channels=default_params['out_channels'],
            n_sequences=default_params['n_sequences'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            dropout=default_params['dropout'],
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'ST-GATLSTM':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': 64,
            'residual_channels': 64,
            'end_channels': 32,
            'n_sequences': k_days + 1,
            'num_layers': num_lstm_layers,
            'heads': 6,
            'dropout': dropout,
            'concat': False,
            'device': device,
            'act_func': act_func,
            'binary': binary
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = ST_GATLSTM(
            in_channels=default_params['in_channels'],
            hidden_channels=default_params['hidden_channels'],
            residual_channels=default_params['residual_channels'],
            end_channels=default_params['end_channels'],
            n_sequences=default_params['n_sequences'],
            num_layers=default_params['num_layers'],
            heads=default_params['heads'],
            dropout=default_params['dropout'],
            concat=default_params['concat'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'LSTM':
        default_params = {
            'in_channels': in_dim,
            'residual_channels': in_dim // 2,
            'hidden_channels': 64,
            'end_channels': 32,
            'n_sequences': k_days + 1,
            'num_layers': num_lstm_layers,
            'device': device,
            'act_func': act_func,
            'binary': binary,
            'dropout': dropout
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = LSTM(
            in_channels=default_params['in_channels'],
            residual_channels=default_params['residual_channels'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            n_sequences=default_params['n_sequences'],
            num_layers=default_params['num_layers'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            binary=default_params['binary'],
            dropout=default_params['dropout']
        )
        model_params.update(default_params)

    elif model_name == 'LSTMGCN':
        default_params = {
            'in_channels': in_dim,
            'residual_channels': in_dim // 2,
            'hidden_channels': 64,
            'end_channels': 32,
            'n_sequences': k_days + 1,
            'num_layers': num_lstm_layers,
            'device': device,
            'act_func': act_func,
            'binary': binary,
            'dropout': dropout
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = LSTMGCN(
            in_channels=default_params['in_channels'],
            residual_channels=default_params['residual_channels'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            n_sequences=default_params['n_sequences'],
            num_layers=default_params['num_layers'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            binary=default_params['binary'],
            dropout=default_params['dropout']
        )
        model_params.update(default_params)

    elif model_name == 'LSTMGAT':
        default_params = {
            'in_channels': in_dim,
            'residual_channels': in_dim // 2,
            'hidden_channels': 64,
            'end_channels': 32,
            'n_sequences': k_days + 1,
            'num_layers': num_lstm_layers,
            'device': device,
            'act_func': act_func,
            'binary': binary,
            'dropout': dropout
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = LSTMGAT(
            in_channels=default_params['in_channels'],
            residual_channels=default_params['residual_channels'],
            hidden_channels=default_params['hidden_channels'],
            end_channels=default_params['end_channels'],
            n_sequences=default_params['n_sequences'],
            num_layers=default_params['num_layers'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            binary=default_params['binary'],
            dropout=default_params['dropout']
        )
        model_params.update(default_params)

    elif model_name == 'Zhang':
        default_params = {
            'in_channels': in_dim_2D,
            'conv_channels': [64, 128, 256],
            'fc_channels': [256 * zhang_layer_conversion[scale] * zhang_layer_conversion[scale], 128, 64, 32],
            'dropout': dropout,
            'binary': binary,
            'device': device,
            'n_sequences': k_days
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = Zhang(
            in_channels=default_params['in_channels'],
            conv_channels=default_params['conv_channels'],
            fc_channels=default_params['fc_channels'],
            dropout=default_params['dropout'],
            binary=default_params['binary'],
            device=default_params['device'],
            n_sequences=default_params['n_sequences']
        )
        model_params.update(default_params)

    elif model_name == 'ResGCN':
        default_params = {
            'in_channels': in_dim_2D,
            'conv_channels': [64, 128, 256],
            'fc_channels': [256 * zhang_layer_conversion[scale] * zhang_layer_conversion[scale], 128, 64, 32],
            'dropout': dropout,
            'binary': binary,
            'device': device,
            'n_sequences': k_days
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = ResGCN(
            in_channels=default_params['in_channels'],
            conv_channels=default_params['conv_channels'],
            fc_channels=default_params['fc_channels'],
            dropout=default_params['dropout'],
            binary=default_params['binary'],
            device=default_params['device'],
            n_sequences=default_params['n_sequences']
        )
        model_params.update(default_params)

    elif model_name == 'ConvLSTM':
        default_params = {
            'in_channels': in_dim_2D,
            'hidden_dim': [64],
            'end_channels': 32,
            'size': shape2D[scale],
            'n_sequences': k_days + 1,
            'device': device,
            'act_func': act_func,
            'dropout': dropout,
            'binary': binary
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = CONVLSTM(
            in_channels=default_params['in_channels'],
            hidden_dim=default_params['hidden_dim'],
            end_channels=default_params['end_channels'],
            n_sequences=default_params['n_sequences'],
            size=default_params['size'],
            device=default_params['device'],
            act_func=default_params['act_func'],
            dropout=default_params['dropout'],
            binary=default_params['binary']
        )
        model_params.update(default_params)

    elif model_name == 'Unet':
        default_params = {
            'n_channels': in_dim_2D,
            'n_classes': 1,
            'features': [64, 128, 256, 512, 1024],
            'bilinear': False,
            'device': device
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = UNet(
            n_channels=default_params['n_channels'],
            n_classes=default_params['n_classes'],
            features=default_params['features'],
            bilinear=default_params['bilinear']
        ).to(default_params['device'])
        model_params.update(default_params)

    elif model_name == 'ULSTM':
        default_params = {
            'n_channels': in_dim_2D,
            'n_classes': 1,
            'n_sequences': k_days + 1,
            'num_lstm_layers': num_lstm_layers,
            'features': [64, 128, 256, 512, 1024],
            'bilinear': False,
            'device': device
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = ULSTM(
            n_channels=default_params['n_channels'],
            n_classes=default_params['n_classes'],
            n_sequences=default_params['n_sequences'],
            num_lstm_layers=default_params['num_lstm_layers'],
            features=default_params['features'],
            bilinear=default_params['bilinear']
        ).to(default_params['device'])
        model_params.update(default_params)

    elif model_name == 'ConvGraphNet':
        default_params = {
            'conv_channels': [64, 128, 256],
            'fc_channels': [256 * zhang_layer_conversion[scale] * zhang_layer_conversion[scale], 128, 64, 32],
            'output_layer_in_channels': 64,
            'output_layer_end_channels': 32,
            'hidden_channels': [256, 64],
            'n_sequence': k_days + 1,
            'binary': binary,
            'device': device,
            'act_func': act_func
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = ConvGraphNet(
            Zhang(
                in_channels=in_dim_2D,
                conv_channels=default_params['conv_channels'],
                fc_channels=default_params['fc_channels'],
                dropout=dropout,
                binary=binary,
                device=device,
                n_sequences=k_days,
                return_hidden=True
            ),
            STGCN(
                n_sequences=k_days + 1,
                in_channels=in_dim,
                hidden_channels=default_params['hidden_channels'],
                end_channels=default_params['output_layer_in_channels'],
                dropout=dropout,
                act_func=act_func,
                device=device,
                binary=binary,
                return_hidden=True
            ),
            output_layer_in_channels=default_params['output_layer_in_channels'],
            output_layer_end_channels=default_params['output_layer_end_channels'],
            n_sequence=default_params['n_sequence'],
            binary=default_params['binary'],
            device=default_params['device'],
            act_func=default_params['act_func']
        )
        model_params.update(default_params)

    elif model_name == 'HybridConvGraphNet':
        default_params = {
            'conv_channels': [64, 128, 256],
            'fc_channels': [256 * zhang_layer_conversion[scale] * zhang_layer_conversion[scale], 128, 64, 32],
            'in_dim_graph': [32, 64, 64],
            'heads': [4, 4],
            'output_layer_in_channels': 64,
            'output_layer_end_channels': 32,
            'n_sequence': k_days + 1,
            'binary': binary,
            'device': device,
            'act_func': act_func
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = HybridConvGraphNet(
            Zhang(
                in_channels=in_dim_2D,
                conv_channels=default_params['conv_channels'],
                fc_channels=default_params['fc_channels'],
                dropout=dropout,
                binary=binary,
                device=device,
                n_sequences=k_days,
                return_hidden=True
            ),
            GAT(
                in_dim=default_params['in_dim_graph'],
                heads=default_params['heads'],
                dropout=dropout,
                bias=True,
                device=device,
                act_func=act_func,
                n_sequences=k_days + 1,
                binary=binary,
                return_hidden=True
            ),
            output_layer_in_channels=default_params['output_layer_in_channels'],
            output_layer_end_channels=default_params['output_layer_end_channels'],
            n_sequence=default_params['n_sequence'],
            binary=default_params['binary'],
            device=default_params['device'],
            act_func=default_params['act_func']
        )
        model_params.update(default_params)

    elif model_name == 'KAN':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': [64, 32],
            'end_channels': 16,
            'k_days': k_days,
            'device': device,
            'binary': binary,
            'act_func': act_func,
            'args': {
                'width': [64, 32],
                'grid': 3,
                'k': 3,
                'mult_arity': 2,
                'noise_scale': 0.3,
                'scale_base_mu': 0.0,
                'scale_base_sigma': 1.0,
                'base_fun': 'silu',
                'symbolic_enabled': True,
                'affine_trainable': False,
                'grid_eps': 0.02,
                'grid_range': [-1, 1],
                'sp_trainable': True,
                'sb_trainable': True,
                'seed': 1,
                'save_act': True,
                'sparse_init': False,
                'auto_save': True,
                'first_init': True,
                'ckpt_path': './model',
                'state_id': 0,
                'round': 0,
                'device': 'cpu'
            }
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = KANnetwork(
            default_params['in_channels'],
            default_params['hidden_channels'],
            default_params['end_channels'],
            default_params['k_days'],
            default_params['device'],
            default_params['binary'],
            default_params['act_func'],
            default_params['args']
        )
        model_params.update(default_params)

    elif model_name == 'TKAN':
        default_params = {
            'input_size': in_dim,
            'hidden_size': [64, 64],
            'end_channels': 32,
            'act_func': act_func,
            'dropout': dropout,
            'binary': binary,
            'k_days': k_days + 1,
            'return_hidden': False,
            'device': device,
            'kan_config': {
                'layers_hidden': [128, 64],
                'grid_size': 5,
                'spline_order': 3,
                'scale_noise': 0.1,
                'scale_base': 1,
                'scale_spline': 1,
                'grid_eps': 0.02,
                'grid_range': [-1, 1],
            }
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = TKAN(
            input_size=default_params['input_size'],
            hidden_size=default_params['hidden_size'],
            end_channels=default_params['end_channels'],
            act_func=default_params['act_func'],
            dropout=default_params['dropout'],
            binary=default_params['binary'],
            k_days=default_params['k_days'],
            return_hidden=default_params['return_hidden'],
            device=default_params['device'],
            kan_config=default_params['kan_config']
        )
        model_params.update(default_params)

    else:
        raise ValueError(f"Modèle '{model_name}' non reconnu.")
    
    return model, model_params