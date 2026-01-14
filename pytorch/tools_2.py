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
from forecasting_models.pytorch.student_distillation import StudentMLP, StudentMLPConcat

torch.manual_seed(42)
np.random.seed(42)

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


def make_model(model_name, in_dim, in_dim_2D, graph, dropout, act_func, k_days, task_type, device, num_lstm_layers, out_channels, custom_model_params=None, horizon=0):
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
    if graph is not None:
        scale = graph.scale
        graph_or_node = graph.graph_method
    else:
        scale = 'departement'
        graph_or_node = 'node'

    shape2D = {10: (24, 24),
           30 : (30, 30),
          2  : (16,16, 2**0),
          3  : (16,16, 2**0),
          4  : (16,16, 2**0),
          5  : (32,32, 2**0),
          6  : (32,32, 2**0),
          7  : (64,64, 2**0),
          8 : (30,30),
            'departement' : (32,32, 2**0)}

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
        'out_channels': out_channels,  # Added the out_channels parameter,
        'horizon': horizon
    }

    if model_name == 'GAT':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [128, 256, 512],
            'out_channels': out_channels,
            'end_channels': 64,
            'heads': [6, 4, 2],
            'dropout': dropout,
            'bias': True,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type,
            'graph_or_node': graph_or_node,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        model = GAT(**default_params)
        model_params.update(default_params)

    elif model_name == 'GCN':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [128, 256, 512],
            'out_channels': out_channels,
            'end_channels': 64,
            'dropout': dropout,
            'bias': True,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type,
            'graph_or_node': graph_or_node,
            'return_hidden': False,
            'horizon': 0
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = GCN(**default_params)
        model_params.update(default_params)

    elif model_name == 'DSTGCN':
        default_params = {
            'in_channels': in_dim,
            'dilation_channels': [in_dim * 2, in_dim * 4],
            'dilations': [1, 2, 3],
            'out_channels': out_channels,
            'end_channels': in_dim,
            'dropout': dropout,
            'act_func': act_func,
            'device': device,
            'task_type': task_type,
            'graph_or_node': graph_or_node,
            'return_hidden': False,
            'n_sequences': k_days + 1,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = DSTGCN(**default_params)
        model_params.update(default_params)

    elif model_name == 'LSTM':
        print(f'num_lstm_layers -> {num_lstm_layers}')
        default_params = {
            'in_channels': in_dim,
            'lstm_size': 128,
            'hidden_channels': 256,
            'out_channels' : out_channels,
            'end_channels': 64,
            'n_sequences': k_days + 1,
            'num_layers': num_lstm_layers,
            'device': device,
            'act_func': act_func,
            'task_type': task_type,
            'return_hidden': False,
            'dropout': dropout,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = LSTM(**default_params)
        model_params.update(default_params)

    elif model_name == 'GRU':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': 256,
            'gru_size': 128,
            'out_channels' : out_channels,
            'end_channels': 64,
            'n_sequences': k_days + 1,
            'num_layers': 2,
            'device': device,
            'act_func': act_func,
            'task_type': task_type,
            'return_hidden': False,
            'dropout': dropout,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = GRU(**default_params)
        model_params.update(default_params)
        
    elif model_name == 'ESN':
        default_params = {
            'in_channels': in_dim,
            'reservoir_size': 256,
            'hidden_channels': 128,
            'out_channels' : out_channels,
            'end_channels': 64,
            'n_sequences': k_days + 1,
            'device': device,
            'act_func': act_func,
            'task_type': task_type,
            'return_hidden': False,
            'dropout': dropout,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = ESN(**default_params)
        model_params.update(default_params)

    elif model_name == 'ClassicESN':
        default_params = {
            'in_channels': in_dim,
            'reservoir_size': 300,
            'hidden_channels': 256,
            'out_channels' : out_channels,
            'dropout': dropout,
            'device': device,
            'task_type': task_type,
            'horizon': horizon,
            'end_channels': 64,
            'n_sequences': k_days + 1,
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = ClassicESN(**default_params)
        model_params.update(default_params)

    elif model_name == 'TransformerNet':
        default_params = {
            'seq_len' : k_days + 1,
            'input_dim' : in_dim,
            'd_model' : 256,
            'nhead' : 8,
            'dim_feedforward' : 512,
            'num_layers' : 4,
            'dropout' : 0.1,
            'channel_attention' : True,
            'graph_or_node' : graph_or_node,
            'task_type' : task_type,
            'out_channels' : out_channels,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        print(custom_model_params)
        model = TransformerNet(**default_params)
        model_params.update(default_params)

    elif model_name == 'TFN':   # Temporal Fusion Network (paper-aligned TFT)
        default_params = {
        'in_channels' : in_dim,
        'tft_size' : 256,                  # d_model TFT
        'hidden_channels' : 256,
        'end_channels' : 128,
        'n_sequences' : k_days + 1,        # longueur historique
        'device' : device,
        'act_func' : 'ReLU',
        'task_type' : task_type,
        'dropout' : 0.1,
        'num_layers' : 1,                  # LSTM encoder/decoder
        'return_hidden' : False,
        'out_channels' : out_channels,
        'use_layernorm' : True,
        'horizon' : horizon,
        'n_heads' : 8,
        'd_grn' : 512,                     # largeur GRN (paper ≈ 2×d_model)
        'd_static' : 0,                    # 0 si tu n'utilises pas de static covariates
        'use_positional_encoding' : False
    }

        if custom_model_params is not None:
            default_params.update(custom_model_params)

        model = TemporalFusionTransformerClassifier(**default_params)
        model_params.update(default_params)

    elif model_name == 'TransformerNetCutClient':
        default_params = {
            'seq_len' : k_days + 1,
            'input_dim' : in_dim,
            'd_model' : 256,
            'nhead' : 8,
            'dim_feedforward' : 512,
            'num_layers' : 4,
            'dropout' : 0.1,
            'graph_or_node' : graph_or_node,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        print(custom_model_params)
        model = TransformerNetCutClient(**default_params)
        model_params.update(default_params)
        
    elif model_name == 'TransformerNetCutServer':
        default_params = {
            'seq_len' : k_days + 1,
            'd_model' : 256,
            'nhead' : 8,
            'dim_feedforward' : 512,
            'num_layers' : 4,
            'dropout' : 0.1,
            'channel_attention' : True,
            'task_type' : task_type,
            'out_channels' : out_channels,
            'graph_or_node' : graph_or_node,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        print(custom_model_params)
        model = TransformerNetCutServer(**default_params)
        model_params.update(default_params)

    elif model_name == 'DilatedCNN':
        default_params = {
            'channels': [in_dim, in_dim, 128],
            'dilations': [1, 3, 4],
            'end_channels': 64,
            'lin_channels': 256,
            'n_sequences': k_days + 1,
            'device': device,
            'act_func': act_func,
            'dropout': dropout,
            'out_channels': out_channels,
            'task_type': task_type,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = DilatedCNN(**default_params)
        model_params.update(default_params)

    elif model_name == 'DSTGAT':
        default_params = {
            'in_dim': in_dim,
            'dilation_channels': [in_dim, in_dim, in_dim],
            'dilations': [1, 2, 3],
            'out_channels' : out_channels,
            'end_channels': in_dim,
            'heads': [6, 4, 2],
            'dropout': dropout,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['graph_or_node'] = graph_or_node
        # Rename key for input dimension
        default_params['in_channels'] = default_params.pop('in_dim')
        model = DSTGAT(**default_params)
        model_params.update(default_params)

    elif model_name == 'SepLSTMGNN':
        input_temporal = len(custom_model_params['temporal_idx'])
        input_spatial = len(custom_model_params['static_idx'])
        default_params = {
            'lstm_hidden': 64,
            'gnn_hidden_list': [64],
            'out_channels': out_channels,
            'end_channels': 64,
            'lin_channels': 256,
            'act_func': act_func,
            'dropout': dropout,
            'device': device,
            'task_type': task_type,
            'n_sequences': k_days + 1,
            'static_idx': custom_model_params['static_idx'],
            'temporal_idx': custom_model_params['temporal_idx'],
            'num_lstm_layers': 2,
            'return_hidden': False,
            'horizon': horizon
            #'num_layers': num_lstm_layers,
        }
         
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        model = Sep_LSTM_GNN(
            lstm_hidden=default_params['lstm_hidden'],
            lin_channels=default_params['lin_channels'],
            gnn_hidden_list=default_params['gnn_hidden_list'],
            out_channels=default_params['out_channels'],
            act_func=default_params['act_func'],
            task_type=default_params['task_type'],
            n_sequences=default_params['n_sequences'],
            end_channels=default_params['end_channels'],
            static_idx=default_params['static_idx'],
            temporal_idx=default_params['temporal_idx'],
            num_lstm_layers=default_params['num_lstm_layers']
        )

        model_params.update(default_params)
        
    elif model_name == 'SepGRUGNN':
        input_temporal = len(custom_model_params['temporal_idx'])
        input_spatial = len(custom_model_params['static_idx'])
        default_params = {
            'gru_hidden': input_temporal,
            'gnn_hidden_list': [input_spatial],
            'out_channels': out_channels,
            'end_channels': 64,
            'lin_channels': 256,
            'act_func': act_func,
            'dropout': dropout,
            'device': device,
            'task_type': task_type,
            'n_sequences': k_days + 1,
            'static_idx': custom_model_params['static_idx'],
            'temporal_idx': custom_model_params['temporal_idx'],
            'num_lstm_layers': num_lstm_layers,
            'return_hidden': False,
            'horizon': horizon
            #'num_layers': num_lstm_layers,
        }
         
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        model = Sep_GRU_GNN(
            gru_hidden=default_params['gru_hidden'],
            lin_channels=default_params['lin_channels'],
            gnn_hidden_list=default_params['gnn_hidden_list'],
            out_channels=default_params['out_channels'],
            act_func=default_params['act_func'],
            task_type=default_params['task_type'],
            n_sequences=default_params['n_sequences'],
            end_channels=default_params['end_channels'],
            static_idx=default_params['static_idx'],
            temporal_idx=default_params['temporal_idx'],
            num_lstm_layers=default_params['num_lstm_layers']
        )

        model_params.update(default_params)
    
    elif model_name == 'LSTMGNNFeedback':
        default_params = {
            'lstm_hidden': in_dim,
            'gnn_hidden': in_dim,
            'out_channels': out_channels,
            'end_channels': 64,
            'act_func': act_func,
            'dropout': dropout,
            'device': device,
            'task_type': task_type,
            'n_sequences': k_days + 1,
            'static_idx': custom_model_params['static_idx'],
            'temporal_idx': custom_model_params['temporal_idx'],
            'num_lstm_layers': num_lstm_layers,
            'return_hidden': False,
            'horizon': horizon
            #'num_layers': num_lstm_layers,
        }
        
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        model = LSTM_GNN_Feedback(**default_params)

        model_params.update(default_params)
        
    elif model_name == 'STGATLSTM':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels_list': [in_dim, in_dim, in_dim],
            'out_channels' : out_channels,
            'end_channels': in_dim,
            'heads': 6,
            'dropout': dropout,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type,
            'num_layers' : num_lstm_layers,
            'concat':False,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['graph_or_node'] = graph_or_node
        default_params['in_channels'] = default_params.pop('in_dim')
        model = ST_GATLSTM(**default_params)
        model_params.update(default_params)

    elif model_name == 'STGAT':
        default_params = {
            'in_dim': in_dim,
            'hidden_channels': [in_dim, in_dim, in_dim],
            'out_channels' : out_channels,
            'end_channels': in_dim,
            'heads': [6, 4, 2],
            'dropout': dropout,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['in_channels'] = default_params.pop('in_dim')
        model = STGAT(**default_params)
        model_params.update(default_params)

    elif model_name == 'STGCN':
        default_params = {
            'in_channels': in_dim,
            'hidden_channels': [in_dim, in_dim, in_dim],
            'out_channels' : out_channels,
            'end_channels': in_dim,
            'dropout': dropout,
            'device': device,
            'act_func': act_func,
            'n_sequences': k_days + 1,
            'task_type': task_type,
            'graph_or_node': graph_or_node,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = STGCN(**default_params)
        model_params.update(default_params)

    elif model_name == 'Zhang':
        default_params = {
                'in_channels': in_dim,
                'conv_channels': [in_dim * 2, in_dim * 4, in_dim * 4],
                'fc_channels': [in_dim * shape2D[scale][2] * shape2D[scale][0] * shape2D[scale][1], 128],
                'out_channels': out_channels,
                'dropout': 0.3,
                'device': device,
                'n_sequences': k_days + 1,
                'return_hidden': False,
                'task_type': task_type,
                'horizon': horizon
        }

        # Mettre à jour avec des paramètres personnalisés si nécessaire
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        # Instanciation du modèle
        model = Zhang(**default_params)
        model_params.update(default_params)

    elif model_name == 'ResNet':
        default_params = {
                'in_channels': in_dim,
                'conv_channels': [in_dim * 2, in_dim * 4, in_dim * 4],
                'fc_channels': [in_dim * 4 * 1 * 1, 128],
                'out_channels': out_channels,
                'dropout': 0.3,
                'device': device,
                'n_sequences': k_days + 1,
                'return_hidden': False,
                'avgpooling': 1,
                'task_type': task_type,
            'horizon': horizon
        }

        # Mettre à jour avec des paramètres personnalisés si nécessaire
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        # Instanciation du modèle
        model = ResNet(**default_params)
        model_params.update(default_params)

    elif model_name == 'ConvLSTM':
        default_params = {
            'in_channels': in_dim,                 # Nombre de canaux en entrée (par ex., 3 pour des images RGB)
            'hidden_dim': [in_dim, in_dim * 2],     # Dimensions des états cachés pour chaque couche ConvLSTM
            'end_channels': 128,              # Dimension finale des couches entièrement connectées
            'size': (shape2D[scale][0], shape2D[scale][1]),                 # Taille spatiale des images d'entrée (Hauteur, Largeur)
            'n_sequences': k_days + 1,                # Nombre de séquences temporelles
            'device': device, # Utilisation de GPU ou CPU
            'act_func': 'relu',      # Fonction d'activation utilisée dans les couches entièrement connectées
            'dropout': 0.3,                   # Taux de dropout (ex. : 30%)
            'out_channels': out_channels,                # Nombre de classes en sortie
            'task_type': task_type,     # Type de tâche ('classification' ou 'régression')
            'return_hidden': False,
            'horizon': horizon
        }

        # Mettre à jour avec des paramètres personnalisés si nécessaire
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        # Instanciation du modèle
        model = CONVLSTM(**default_params)
        model_params.update(default_params)

    elif model_name == 'UNet':
        default_params = {
            'n_channels': in_dim,
            'out_channels': out_channels,
            'conv_channels': [in_dim * 2, in_dim * 4],
            'return_hidden': False
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = UNet(**default_params)
        model_params.update(default_params)

    elif model_name == 'graphCast':
        default_params = {
            'input_dim_grid_nodes': in_dim,
            'input_dim_mesh_nodes': 3,
            'input_dim_edges':4,
            'output_dim_grid_nodes' : 256,
            'processor_layers' : 6,
            'hidden_layers' : 1,
            'hidden_dim' : in_dim,
            'lin_channels' : 256,
            'end_channels' : 64,
            'aggregation' : 'sum',
            'norm_type' : 'LayerNorm',
            'do_concat_trick' : False,
            'out_channels' : out_channels,
            'act_func' : act_func,
            'task_type' : task_type,
            'has_time_dim' : True,
            'graph_or_node':graph_or_node,
            'n_sequences' : k_days + 1,
            'return_hidden': False,
            'horizon': horizon

        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['is_graph_or_node'] = default_params.pop('graph_or_node')
        model = GraphCast(**default_params)
        model_params.update(default_params)
    
    elif model_name == 'graphCastGRU':
        default_params = {
            'num_gru_layers' : 2,
            'in_channels' : in_dim,
            'input_dim_grid_nodes': 128,
            'input_dim_mesh_nodes': 3,
            'input_dim_edges':4,
            'output_dim_grid_nodes' : 128,
            'processor_layers' : 6,
            'hidden_layers' : 1,
            'hidden_dim' : 128,
            'lin_channels' : 256,
            'end_channels' : 64,
            'aggregation' : 'sum',
            'norm_type' : 'LayerNorm',
            'do_concat_trick' : False,
            'out_channels' : out_channels,
            'act_func' : act_func,
            'task_type' : task_type,
            'has_time_dim' : True,
            'graph_or_node':graph_or_node,
            'n_sequences' : k_days + 1,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['is_graph_or_node'] = default_params.pop('graph_or_node')
        model = GraphCastGRU(**default_params)
        model_params.update(default_params)

    elif model_name == 'graphCastGRUWithAttentionTrans':
        default_params = {
            'num_gru_layers' : 2,
            'in_channels' : in_dim,
            'input_dim_grid_nodes': 128,
            'input_dim_mesh_nodes': 3,
            'input_dim_edges':4,
            'output_dim_grid_nodes' : 128,
            'processor_layers' : 6,
            'hidden_layers' : 1,
            'hidden_dim' : 128,
            'lin_channels' : 256,
            'end_channels' : 64,
            'aggregation' : 'sum',
            'norm_type' : 'LayerNorm',
            'do_concat_trick' : False,
            'out_channels' : out_channels,
            'act_func' : act_func,
            'task_type' : task_type,
            'has_time_dim' : True,
            'graph_or_node':graph_or_node,
            'n_sequences' : k_days + 1,
            'return_hidden': False,
            'attention' : "Transformer",
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['is_graph_or_node'] = default_params.pop('graph_or_node')
        model = GraphCastGRUWithAttention(**default_params)
        model_params.update(default_params)
    
    elif model_name == 'graphCastGRUWithAttentionGAT':
        default_params = {
            'num_gru_layers' : 2,
            'in_channels' : in_dim,
            'input_dim_grid_nodes': 128,
            'input_dim_mesh_nodes': 3,
            'input_dim_edges':4,
            'output_dim_grid_nodes' : 128,
            'processor_layers' : 6,
            'hidden_layers' : 1,
            'hidden_dim' : 128,
            'lin_channels' : 256,
            'end_channels' : 64,
            'aggregation' : 'sum',
            'norm_type' : 'LayerNorm',
            'do_concat_trick' : False,
            'out_channels' : out_channels,
            'act_func' : act_func,
            'task_type' : task_type,
            'has_time_dim' : True,
            'graph_or_node':graph_or_node,
            'n_sequences' : k_days + 1,
            'return_hidden': False,
            'attention' : "GAT",
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['is_graph_or_node'] = default_params.pop('graph_or_node')
        model = GraphCastGRUWithAttention(**default_params)
        model_params.update(default_params)

    elif model_name == 'StudentMLPConcat':
        default_params = {
            'input_dim': in_dim,
            'out_channels': out_channels,
            'mlp_hidden_dim': 128, # Default value
            'embedding_dim': 128,  # Default value
            'end_channels' : 64,
            'n_group': 3,          # Default value
            'device': device,
            'task_type': task_type,
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
            
        model = StudentMLPConcat(**default_params)
        model_params.update(default_params)

    elif model_name == 'studenMLP':
        default_params = {
            'input_dim': in_dim,
            'out_channels': out_channels,
            'mlp_hidden_dim': 128, # Default value
            'embedding_dim': 128,  # Default value
            'end_channels' : 64,
            'n_group': 3,          # Default value
            'device': device,
            'task_type': task_type,
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
            
        model = StudentMLP(**default_params)
        model_params.update(default_params)
        
    elif model_name == 'MultiScaleGraph':
        default_params = {
            'input_channels' : in_dim,
            'features_per_scale' : [in_dim * 2, in_dim * 4, in_dim * 6, in_dim * 8, in_dim * 10],
            'num_output_scale' : custom_model_params['num_output_scale'],
            'num_sequence' : k_days + 1, 
            'out_channels' : 5, 
            'graph_or_node' : graph_or_node,
            'task_type':task_type,
            'device':device,
            'return_hidden': False,
            'horizon': horizon
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = MultiScaleGraph(**default_params)
        model_params.update(default_params)

    elif model_name == 'MultiScaleAttentionGraph':
        default_params = {
            'input_channels' : in_dim,
            'features_per_scale' : [in_dim * 2, in_dim * 4, in_dim * 6, in_dim * 8, in_dim * 10],
            'num_output_scale' : custom_model_params['num_output_scale'],
            'num_sequence' : k_days + 1, 
            'out_channels' : 5, 
            'graph_or_node' : graph_or_node,
            'task_type':task_type,
            'device':device,
            'return_hidden': False,
            'horizon': horizon


        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)

        model = MultiScaleAttentionGraph(**default_params)
        model_params.update(default_params)

    elif model_name == 'NetGCN':
        default_params = {
          'in_dim':in_dim,
          'hidden_dim' : in_dim * 2,
          'hidden_dim_2' : in_dim * 4,
          'end_channels' : 64,
          'output_channels' : out_channels,
            'graph_or_node': graph_or_node,
            'n_sequences' : k_days + 1,
            'device' : device,
            'return_hidden': False,
            'horizon': horizon


          
        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['task_type'] = task_type
        model = NetGCN(**default_params)
        model_params.update(default_params)

    elif model_name == 'NetMLP':
        default_params = {
          'in_dim':in_dim,
          'hidden_dim' : [128, 256],
          'end_channels' : 64,
          'output_channels' : out_channels,          
            'n_sequences' : k_days + 1,
            'device' : device,
            'return_hidden': False,
            'horizon': horizon

        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        default_params['task_type'] = task_type
        model = NetMLP(**default_params)
        model_params.update(default_params)
    elif model_name == 'BayesianMLP':
        default_params = {
            'in_dim': in_dim,
            'hidden_dim': in_dim,
            'out_channels': out_channels,
            'task_type': task_type,
            'device': device,
            'graph_or_node': graph_or_node,
            'return_hidden': False,
            'horizon': horizon

        }
        if custom_model_params is not None:
            default_params.update(custom_model_params)
        model = BayesianMLP(**default_params)
        model_params.update(default_params)
    else:
        raise ValueError(f"Modèle '{model_name}' non reconnu.")
    
    return model, model_params