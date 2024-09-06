from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
import os
import math
import numpy as np
import shap
from forecasting_models.models import *
from forecasting_models.models_2D import *

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
    Renvoie un modèle en fonction du nom spécifié.
    
    Parameters:
    - model_name: Nom du modèle à retourner.
    - in_dim: Dimensions d'entrée.
    - scale: Non utilisé dans l'exemple, mais pourrait être utilisé pour la normalisation.
    - dropout: Taux de dropout pour le modèle.
    - act_func: Fonction d'activation.
    - k_days: Nombre de séquences/jours pour la prédiction.
    - binary: Indicateur pour un problème de classification binaire.

    Returns:
    - Modèle spécifié.
    """

    if model_name == 'GAT':
        return GAT(in_dim=[in_dim, 64, 64, 64],
                   heads=[4, 4, 2],
                   dropout=dropout,
                   bias=True,
                   device=device,
                   act_func=act_func,
                   n_sequences=k_days + 1,
                   binary=binary)

    elif model_name == 'DST-GCN':
        return DSTGCN(n_sequences=k_days+1,
                      in_channels=in_dim,
                      end_channels=64,
                      dilation_channels=[256, 64],
                      dilations=[1],
                      dropout=dropout,
                      act_func=act_func,
                      device=device,
                      binary=binary)
    
    elif model_name == 'ST-GAT':
        return STGAT(n_sequences=k_days + 1,
                     in_channels=in_dim,
                     hidden_channels=[256, 64],
                     end_channels=64,
                     dropout=dropout, heads=6,
                     act_func=act_func, device=device,
                     binary=binary)

    elif model_name == 'ST-GCN':
        return STGCN(n_sequences=k_days + 1,
                     in_channels=in_dim,
                     hidden_channels=[256, 64],
                     end_channels=64,
                     dropout=dropout,
                     act_func=act_func,
                     device=device,
                     binary=binary)

    elif model_name == 'SDT-GCN':
        return SDSTGCN(n_sequences=k_days + 1,
                       in_channels=in_dim,
                       hidden_channels_temporal=[256, 64],
                       dilations=[1],
                       hidden_channels_spatial=[256, 64],
                       end_channels=64,
                       dropout=dropout,
                       act_func=act_func,
                       device=device,
                       binary=binary)

    elif model_name == 'ATGN':
        return TemporalGNN(in_channels=in_dim,
                           hidden_channels=64,
                           out_channels=64,
                           n_sequences=k_days + 1,
                           device=device,
                           act_func=act_func,
                           dropout=dropout,
                           binary=binary)

    elif model_name == 'ST-GATLSTM':
        return ST_GATLSTM(in_channels=in_dim,
                          hidden_channels=64,
                          residual_channels=64,
                          end_channels=32,
                          n_sequences=k_days + 1,
                          num_layers=num_lstm_layers,
                          device=device, act_func=act_func, heads=6, dropout=dropout,
                          concat=False,
                          binary=binary)

    elif model_name == 'LSTM':
        return LSTM(in_channels=in_dim, residual_channels=64,
                    hidden_channels=64,
                    end_channels=32, n_sequences=k_days + 1,
                    device=device, act_func=act_func, binary=binary,
                    dropout=dropout, num_layers=num_lstm_layers)

    elif model_name == 'Zhang':
        return Zhang(in_channels=in_dim_2D, conv_channels=[64, 128, 256], fc_channels=[256 * 15 * 15, 128, 64, 32],
                     dropout=dropout, binary=binary, device=device, n_sequences=k_days)
    
    elif model_name == 'ConvLSTM':
        return CONVLSTM(in_channels=in_dim_2D, hidden_dim=[64, 128, 256], end_channels=32,
        n_sequences=k_days+1, device=device, act_func=act_func, dropout=dropout, binary=binary)

    elif model_name == 'UNet':
        return UNet(n_channels=in_dim_2D, n_classes=1, bilinear=False)

    elif model_name == 'ConvGraphNet':
        return ConvGraphNet(Zhang(in_channels=in_dim_2D, conv_channels=[64, 128, 256], fc_channels=[256 * 7 * 7, 128, 64, 32],
                                  dropout=dropout, binary=binary, device=device, n_sequences=k_days, return_hidden=True),
                            STGCN(n_sequences=k_days + 1,
                                  in_channels=in_dim,
                                  hidden_channels=[256, 64],
                                  end_channels=64,
                                  dropout=dropout,
                                  act_func=act_func,
                                  device=device,
                                  binary=binary,
                                  return_hidden=True),
                            output_layer_in_channels=64,
                            output_layer_end_channels=32,
                            n_sequence=k_days+1,
                            binary=binary,
                            device=device,
                            act_func=act_func)
    
    elif model_name == 'HybridConvGraphNet':
        return HybridConvGraphNet(Zhang(in_channels=in_dim_2D, conv_channels=[64, 128, 256], fc_channels=[256 * 15 * 15, 128, 64, 32],
                                  dropout=dropout, binary=binary, device=device, n_sequences=k_days, return_hidden=True),
                                    GAT(in_dim=[32, 64, 64],
                                    heads=[4, 4],
                                    dropout=dropout,
                                    bias=True,
                                    device=device,
                                    act_func=act_func,
                                    n_sequences=k_days + 1,
                                    binary=binary, 
                                    return_hidden=True),
                                    output_layer_in_channels=64,
                                    output_layer_end_channels=32,
                                    n_sequence=k_days+1,
                                    binary=binary,
                                    device=device,
                                    act_func=act_func,
                                    )

    else:
        raise ValueError(f"Modèle '{model_name}' non reconnu.")