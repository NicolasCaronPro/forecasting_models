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

def make_model(model_name, in_dim, in_dim_2D, scale, dropout, act_func, k_days, binary, device, num_lstm_layers):
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
    
    Returns:
    - Tuple (modèle, paramètres) où 'modèle' est le modèle spécifié et 'paramètres' est un dictionnaire des paramètres utilisés.
    """
    
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
        model = GAT(
            in_dim=[in_dim, 64, 64, 64],
            heads=[4, 4, 2],
            dropout=dropout,
            bias=True,
            device=device,
            act_func=act_func,
            n_sequences=k_days + 1,
            binary=binary
        )
        model_params.update({
            'in_dim_layers': [in_dim, 64, 64, 64],
            'heads': [4, 4, 2],
            'bias': True,
        })
    
    elif model_name == 'GCN':
        model = GCN(
            n_sequences=k_days + 1,
            in_dim=[in_dim, 64, 64, 64],
            dropout=dropout,
            bias=True,
            device=device,
            act_func=act_func,
            binary=binary
        )
        model_params.update({
            'in_dim_layers': [in_dim, 64, 64, 64],
            'bias': True,
        })
    
    elif model_name == 'DST-GCN':
        model = DSTGCN(
            n_sequences=k_days + 1,
            in_channels=in_dim,
            end_channels=64,
            dilation_channels=[128, 256, 512, 256, 128],
            dilations=[1, 1, 2, 1, 1],
            dropout=dropout,
            act_func=act_func,
            device=device,
            binary=binary
        )
        model_params.update({
            'end_channels': 64,
            'dilation_channels': [128, 256, 512, 256, 128],
            'dilations': [1, 1, 2, 1, 1],
        })
    
    elif model_name == 'ST-GAT':
        model = STGAT(
            n_sequences=k_days + 1,
            in_channels=in_dim,
            hidden_channels=[256, 64],
            end_channels=64,
            dropout=dropout,
            heads=6,
            act_func=act_func,
            device=device,
            binary=binary
        )
        model_params.update({
            'hidden_channels': [256, 64],
            'end_channels': 64,
            'heads': 6,
        })
    
    elif model_name == 'ST-GCN':
        model = STGCN(
            n_sequences=k_days + 1,
            in_channels=in_dim,
            hidden_channels=[256, 64],
            end_channels=64,
            dropout=dropout,
            act_func=act_func,
            device=device,
            binary=binary
        )
        model_params.update({
            'hidden_channels': [256, 64],
            'end_channels': 64,
        })
    
    elif model_name == 'SDT-GCN':
        model = SDSTGCN(
            n_sequences=k_days + 1,
            in_channels=in_dim,
            hidden_channels_temporal=[256, 64],
            dilations=[1],
            hidden_channels_spatial=[256, 64],
            end_channels=64,
            dropout=dropout,
            act_func=act_func,
            device=device,
            binary=binary
        )
        model_params.update({
            'hidden_channels_temporal': [256, 64],
            'hidden_channels_spatial': [256, 64],
            'end_channels': 64,
            'dilations': [1],
        })
    
    elif model_name == 'ATGN':
        model = TemporalGNN(
            in_channels=in_dim,
            hidden_channels=64,
            out_channels=64,
            n_sequences=k_days + 1,
            device=device,
            act_func=act_func,
            dropout=dropout,
            binary=binary
        )
        model_params.update({
            'hidden_channels': 64,
            'out_channels': 64,
        })
    
    elif model_name == 'ST-GATLSTM':
        model = ST_GATLSTM(
            in_channels=in_dim,
            hidden_channels=64,
            residual_channels=64,
            end_channels=32,
            n_sequences=k_days + 1,
            num_layers=num_lstm_layers,
            device=device,
            act_func=act_func,
            heads=6,
            dropout=dropout,
            concat=False,
            binary=binary
        )
        model_params.update({
            'hidden_channels': 64,
            'residual_channels': 64,
            'end_channels': 32,
            'num_layers': num_lstm_layers,
            'heads': 6,
            'concat': False,
        })
    
    elif model_name == 'LSTM':
        model = LSTM(
            in_channels=in_dim,
            residual_channels=64,
            hidden_channels=64,
            end_channels=32,
            n_sequences=k_days + 1,
            device=device,
            act_func=act_func,
            binary=binary,
            dropout=dropout,
            num_layers=num_lstm_layers
        )
        model_params.update({
            'residual_channels': 64,
            'hidden_channels': 64,
            'end_channels': 32,
            'num_layers': num_lstm_layers,
        })
    
    elif model_name == 'Zhang':
        model = Zhang(
            in_channels=in_dim_2D,
            conv_channels=[64, 128, 256],
            fc_channels=[256 * 15 * 15, 128, 64, 32],
            dropout=dropout,
            binary=binary,
            device=device,
            n_sequences=k_days
        )
        model_params.update({
            'conv_channels': [64, 128, 256],
            'fc_channels': [256 * 15 * 15, 128, 64, 32],
        })
    
    elif model_name == 'ConvLSTM':
        model = CONVLSTM(
            in_channels=in_dim_2D,
            hidden_dim=[64, 128, 256],
            end_channels=32,
            n_sequences=k_days + 1,
            device=device,
            act_func=act_func,
            dropout=dropout,
            binary=binary
        )
        model_params.update({
            'hidden_dim': [64, 128, 256],
            'end_channels': 32,
        })
    
    elif model_name == 'Unet':
        model = UNet(
            n_channels=in_dim_2D,
            n_classes=1,
            bilinear=False
        ).to(device)
        model_params.update({
            'n_classes': 1,
            'bilinear': False,
        })
    
    elif model_name == 'ConvGraphNet':
        model = ConvGraphNet(
            Zhang(
                in_channels=in_dim_2D,
                conv_channels=[64, 128, 256],
                fc_channels=[256 * 7 * 7, 128, 64, 32],
                dropout=dropout,
                binary=binary,
                device=device,
                n_sequences=k_days,
                return_hidden=True
            ),
            STGCN(
                n_sequences=k_days + 1,
                in_channels=in_dim,
                hidden_channels=[256, 64],
                end_channels=64,
                dropout=dropout,
                act_func=act_func,
                device=device,
                binary=binary,
                return_hidden=True
            ),
            output_layer_in_channels=64,
            output_layer_end_channels=32,
            n_sequence=k_days + 1,
            binary=binary,
            device=device,
            act_func=act_func
        )
        model_params.update({
            'conv_graph_net_params': {
                'output_layer_in_channels': 64,
                'output_layer_end_channels': 32,
                'n_sequence': k_days + 1,
            }
        })
    
    elif model_name == 'HybridConvGraphNet':
        model = HybridConvGraphNet(
            Zhang(
                in_channels=in_dim_2D,
                conv_channels=[64, 128, 256],
                fc_channels=[256 * 15 * 15, 128, 64, 32],
                dropout=dropout,
                binary=binary,
                device=device,
                n_sequences=k_days,
                return_hidden=True
            ),
            GAT(
                in_dim=[32, 64, 64],
                heads=[4, 4],
                dropout=dropout,
                bias=True,
                device=device,
                act_func=act_func,
                n_sequences=k_days + 1,
                binary=binary,
                return_hidden=True
            ),
            output_layer_in_channels=64,
            output_layer_end_channels=32,
            n_sequence=k_days + 1,
            binary=binary,
            device=device,
            act_func=act_func
        )
        model_params.update({
            'hybrid_conv_graph_net_params': {
                'output_layer_in_channels': 64,
                'output_layer_end_channels': 32,
                'n_sequence': k_days + 1,
            }
        })
    
    elif model_name == 'KAN':
        output_layer = 1 if not binary else 2
        model = KAN(
            in_channels=in_dim,
            end_channels=64,
            device=device,
            binary=binary,
            k_days=0,
            layers_hidden=[in_dim, 256, 512, 256, 64],
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1,
            scale_spline=1,
            act_func='relu',
            grid_eps=0.02,
            grid_range=[-1, 1]
        )
        model_params.update({
            'layers_hidden': [in_dim, 128, 64],
            'grid_size': 5,
            'spline_order': 3,
            'scale_noise': 0.1,
            'scale_base': 1,
            'scale_spline': 1,
            'base_activation': act_func,
            'grid_eps': 0.02,
            'grid_range': [-1, 1],
        })
        
    elif model_name == 'TKAN':
        # Configuration pour le modèle TKAN
        kan_config = {
            'layers_hidden': [128, 64],  # Couches cachées pour le KAN
            'grid_size': 5,
            'spline_order': 3,
            'scale_noise': 0.1,
            'scale_base': 1,
            'scale_spline': 1,
            'grid_eps': 0.02,
            'grid_range': [-1, 1],
        }
        model = TKAN(
            input_size=in_dim,
            hidden_size=[64, 64],  # Tailles des couches cachées du TKAN
            end_channels=32,
            act_func=act_func,
            dropout=dropout,
            binary=binary,
            k_days=k_days + 1,
            return_hidden=False,
            device=device,
            kan_config=kan_config
        )
        model_params.update({
            'input_size': in_dim,
            'hidden_size': [64, 64],
            'end_channels': 32,
            'return_hidden': False,
            'kan_config': kan_config,
        })
    
    else:
        raise ValueError(f"Modèle '{model_name}' non reconnu.")
    
    return model, model_params