from numpy import dtype
import numpy as np
import gc
import random
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)

from PIL import Image
import torchvision.transforms.functional as TF

from copy import deepcopy
import itertools
from matplotlib import pyplot as plt
from GNN.discretization import *
from GNN.tools import (
    calculate_area_under_curve,
    under_prediction_score,
    over_prediction_score,
    iou_score,
    evaluate_metrics,
    calculate_ic95,
)
from GNN.config import graph_id_index, departement_index
from sklearn.metrics import f1_score, jaccard_score

from GNN.graph_builder import *
from GNN.tools import check_and_create_path, save_object, read_object
from forecasting_models.pytorch.distillation_utils import RelationMLP, RelationAttention, Adapter, FitNet, multi_teacher_kd_loss, multi_teacher_kd_loss_global_weights, lht_loss, angle_triplet_loss, confidence_distillation_loss
from forecasting_models.pytorch.student_distillation import StudentMLP

from tqdm import tqdm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

g = torch.Generator()
g.manual_seed(42)

def plot_score_per_epochs(score_per_epoch, dir_output, name):
    plt.figure(figsize=(15,5))
    scores = score_per_epoch['score']
    epochs = score_per_epoch['epoch']
    plt.plot(epochs, scores)
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.savefig(dir_output / f'{name}.png')
    plt.close('all')

class ReadGraphDataset_2D(Dataset):
    def __init__(self, X : list,
                 Y : list,
                 edges : list,
                 leni : int,
                 device : torch.device,
                 path : Path) -> None:
        
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni
        self.path = path

    def __getitem__(self, index) -> tuple:
        x = read_object(self.X[index], self.path)
        #y = read_object(self.Y[index], self.path)
        y = self.Y[index]

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []

        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

class ReadGraphDataset_2D_from_xarray(Dataset):
    def __init__(self, X : list,
                 Y : list,
                 edges : list,
                 leni : int,
                 device : torch.device,
                 path : Path,
                 features,
                 features_1D,
                 kdays,
                 scale,
                 graph_method,
                 base,
                 target_path
                 ):
                
        self.X = X
        self.Y = np.asarray(Y)
        self.device = device
        self.edges = edges
        self.leni = leni
        self.path = path
        self.features = features
        self.features_1D = features_1D
        self.kdays = kdays
        self.scale = scale
        self.graph_method = graph_method
        self.base = base
        self.target_path = target_path

        self.datacubes = {}
        self.areas = {}

        depts = np.unique(self.Y[:, departement_index, -1])
        
        for dept in depts:
            logger.info(f'Loading features of {dept}')
            mask = self.Y[:, departement_index, -1] == dept
            dates = np.unique(self.Y[mask, date_index, :])
            
            datacube = read_object(
                f'datacube.pkl',
                self.path / int2name[dept] / 'raster' / '2x2'
            )

            datacube_mask = read_object(f'datacube_target_{int2name[dept]}_{self.scale}_{self.base}_{self.graph_method}.pkl', self.target_path)

            dates_str = [allDates[int(date)] for date in dates]

            datacube = datacube.sel(date=dates_str)

            for var in ['precipitationIndexN5', 'precipitationIndexN3', 'precipitationIndexN9']:
                n = int(var[-1])
                array = calculate_precipitation_index_image_full(datacube['prec24h'].values, A=0.1657, n=n)
                datacube[var] = (('latitude', 'longitude', 'date'), array)

            # Ne conserver que les variables 2D demandées et existantes
            existing_vars = set(datacube.data_vars)
            wanted_vars = [v for v in self.features if v in existing_vars]
            #missing = [v for v in self.features if v not in existing_vars]
            #if missing:
            #    logger.warning(f"[{int2name[dept]}] Variables 2D absentes ignorées: {missing}")

            # Sous-échantillonnage du Dataset xarray aux seules features 2D
            if wanted_vars:
                datacube = datacube[wanted_vars]
            else:
                logger.warning(f"[{int2name[dept]}] Aucune feature_2D trouvée; datacube vide en variables.")

            self.datacubes[dept] = datacube
            self.areas[dept] = datacube_mask['area']

    def get_area_coords(self, area_dataarray, area_id):
        mask = (area_dataarray == area_id)
        lat_coords = area_dataarray.latitude.values
        lon_coords = area_dataarray.longitude.values
        positions = mask.values.nonzero()
        
        if len(positions[0]) == 0:
            raise ValueError(f"Aucune position trouvée pour area_id={area_id}")

        lat_indices = positions[1]
        lon_indices = positions[2]

        #print(area_dataarray.values.shape)
        #print(positions)

        lat_start_idx = lat_indices.min()
        lat_end_idx = lat_indices.max() + 1
        
        lon_start_idx = lon_indices.min()
        lon_end_idx = lon_indices.max() + 1

        lat_start = lat_coords[lat_start_idx]
        lat_end = lat_coords[lat_end_idx - 1]
        lon_start = lon_coords[lon_start_idx]
        lon_end = lon_coords[lon_end_idx - 1]

        return lat_start, lat_end, lon_start, lon_end

    def __getitem__(self, index) -> tuple:
        y = self.Y[index]

        dept = y[departement_index][-1]
        area_id = y[graph_id_index][-1]
        dates = y[date_index][-1]

        dates = dates.astype(int)

        datacube = self.datacubes[dept]

        lat_start, lat_end, lon_start, lon_end = self.get_area_coords(
            self.areas[dept], area_id
        )

        feature_cubes = []

        for feat_idx, feat in enumerate(self.features):
            # Si la feature est une variable externe
            if feat in calendar_variables or \
                feat == "id_encoder" or \
                feat == "cluster_encoder" or \
                feat == 'Past_burnedarea' or \
                feat == 'Past_risk' or \
                'calendar' in feat:
                # On récupère depuis self.X
                feat_idx = self.features_1D.index(feat)
                values = self.X[index][feat_idx]  # shape attendue : (kdays,)
                values = np.array(values).reshape(1, 1, self.kdays + 1)
                values = np.tile(values, (shape2D[self.scale][0], shape2D[self.scale][1], 1))  # (16, 16, kdays)
            else:
                if feat == 'foret_encoder':
                    feat = 'forest_landcover'
                elif feat == 'corine_encoder':
                    feat = 'corine_landcover'
                elif feat == 'bdroute_encoder':
                    feat = 'route_landcover'

                da = datacube[feat]
                if "date" in da.dims:
                    if self.kdays > 0:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=slice(allDates[dates - self.kdays], allDates[dates])
                        ).values
                    else:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=allDates[dates]
                        ).values
                        selected = selected[:, :, None]

                    #selected = selected.transpose(1, 2, 0)
                else:
                    selected = da.sel(
                        latitude=slice(lat_start, lat_end),
                        longitude=slice(lon_start, lon_end)
                    ).values  # shape: (h, w)
                    selected = selected[:, :, None].repeat(self.kdays + 1, axis=2)

                nan_mask = np.isnan(selected)
                if np.any(nan_mask):
                    mean_val = np.nanmean(selected)
                    selected[nan_mask] = mean_val

                H, W, T = selected.shape
                resized = np.zeros((shape2D[self.scale][0], shape2D[self.scale][1], T), dtype=selected.dtype)

                for t in range(T):
                    resized[:, :, t] = cv2.resize(selected[:, :, t], (shape2D[self.scale][0], shape2D[self.scale][1]), interpolation=cv2.INTER_LINEAR)

                selected = resized
                values = selected

            feature_cubes.append(values)

        feature_tensor = torch.tensor(
            np.stack(feature_cubes, axis=2), dtype=torch.float32, device=self.device
        )
        #print('features_tensort', feature_tensor.shape)
        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []

        return feature_tensor, \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

class ReadGraphDataset2DOptim(Dataset):
    def __init__(self, X, Y, edges, leni, device, path, features, features_1D,
                 kdays, scale, graph_method, base, target_path):

        self.X = X
        self.Y = np.asarray(Y)
        self.device = device
        self.edges = edges
        self.leni = leni
        self.path = path
        self.features = features
        self.features_1D = features_1D
        self.kdays = kdays
        self.scale = scale
        self.graph_method = graph_method
        self.base = base
        self.target_path = target_path

        self.datacubes = {}
        self.areas = {}
        self.area_coords = {}   # cache des coordonnées

        # Map pour éviter .index() dans __getitem__
        self.features_1D_map = {feat: idx for idx, feat in enumerate(self.features_1D)}

        # Préparer uniquement le cache de zones (faible mémoire)
        depts = np.unique(self.Y[:, departement_index, -1])
        for dept in depts:
            logger.info(f'Loading features from {dept}')
            datacube = read_object(
                f'datacube.pkl',
                self.path / int2name[dept] / 'raster' / '2x2'
            )

            datacube_mask = read_object(
                f'datacube_target_{int2name[dept]}_{self.scale}_{self.base}_{self.graph_method}.pkl',
                self.target_path
            )

            self.datacubes[dept] = datacube  # pas converti en numpy
            self.areas[dept] = datacube_mask['area']

            self.area_coords[dept] = {}
            unique_ids = np.unique(self.areas[dept].values)
            for area_id in unique_ids:
                if np.isnan(area_id):
                    continue
                self.area_coords[dept][area_id] = self._compute_coords(self.areas[dept], area_id)

    def _compute_coords(self, area_dataarray, area_id):
        mask = (area_dataarray == area_id).values
        pos = np.nonzero(mask)
        lat_coords = area_dataarray.latitude.values
        lon_coords = area_dataarray.longitude.values

        lat_start_idx, lat_end_idx = pos[1].min(), pos[1].max() + 1
        lon_start_idx, lon_end_idx = pos[2].min(), pos[2].max() + 1

        return (lat_coords[lat_start_idx], lat_coords[lat_end_idx - 1],
                lon_coords[lon_start_idx], lon_coords[lon_end_idx - 1])

    def _resize_3d(self, arr, out_h, out_w):
        """Redimensionner un tableau (H, W, T) sans boucle Python."""
        t = arr.shape[2]
        resized = np.empty((out_h, out_w, t), dtype=arr.dtype)
        for i in range(t):
            resized[:, :, i] = cv2.resize(arr[:, :, i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def __getitem__(self, index):
        y = self.Y[index]

        dept = y[departement_index][-1]
        area_id = y[graph_id_index][-1]
        date_int = int(y[date_index][-1])

        datacube = self.datacubes[dept]

        lat_start, lat_end, lon_start, lon_end = self.area_coords[dept][area_id]

        feature_cubes = []
        for feat in self.features:
            # Variables 1D -> répétition en 2D
            if feat in calendar_variables or feat in {"id_encoder", "cluster_encoder", "Past_burnedarea", "Past_risk"}:
                feat_idx = self.features_1D_map[feat]
                values = np.array(self.X[index][feat_idx]).reshape(1, 1, self.kdays + 1)
                values = np.tile(values, (shape2D[self.scale][0], shape2D[self.scale][1], 1))
            else:
                mapped_feat = {
                    'foret_encoder': 'forest_landcover',
                    'corine_encoder': 'corine_landcover',
                    'bdroute_encoder': 'route_landcover'
                }.get(feat, feat)

                da = datacube[mapped_feat]
                if "date" in da.dims:
                    if self.kdays > 0:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=slice(allDates[date_int - self.kdays], allDates[date_int])
                        ).values
                    else:
                        selected = da.sel(
                            latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end),
                            date=allDates[date_int]
                        ).values
                        selected = selected[:, :, None]
                else:
                    selected = da.sel(
                        latitude=slice(lat_start, lat_end),
                        longitude=slice(lon_start, lon_end)
                    ).values[:, :, None].repeat(self.kdays + 1, axis=2)

                if np.any(np.isnan(selected)):
                    selected = np.nan_to_num(selected, nan=np.nanmean(selected))

                selected = self._resize_3d(selected, shape2D[self.scale][0], shape2D[self.scale][1])
                values = selected

            feature_cubes.append(values)

        feature_tensor = torch.tensor(np.stack(feature_cubes, axis=2), dtype=torch.float32, device=self.device)
        edges_tensor = torch.tensor(self.edges[index], dtype=torch.long, device=self.device) if len(self.edges) > 0 else []

        return feature_tensor, torch.tensor(y, dtype=torch.float32, device=self.device), edges_tensor

    def __len__(self):
        return self.leni

class InplaceGraphDataset(Dataset):
    def __init__(self, X : list, Y : list, edges : list, leni : int, device : torch.device) -> None:
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []
        
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

class InplaceMeshGraphDataset(Dataset):
    def __init__(self, icospheres_graph_path : str, X : list, Y : list, edges : list, leni : int, device : torch.device) -> None:
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni
        self.icospheres_graph_path = icospheres_graph_path

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []
        
        X, Y, E = torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \
            
        return X, Y, E, self.icospheres_graph_path
    
    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

class InplaceMeshGraphDatasetInplace(Dataset):
    def __init__(self, X : list, Y : list, edges : list, leni : int, device : torch.device, graph_mesh, gridh2mesh, mesh2graph) -> None:
        self.X = X
        self.Y = Y
        self.device = device
        self.edges = edges
        self.leni = leni
            
        self.graph_mesh = graph_mesh
        self.grid2mesh = gridh2mesh
        self.mesh2grid = mesh2graph

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []
        
        X, Y, E = torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \

        return X, Y, E, self.graph_mesh, self.grid2mesh, self.mesh2grid

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

class InplaceMulitpleGraphDataset(Dataset):
    def __init__(self, target_name : str, list_graph_file : list, X : list, Y : list, edges : list, leni : int, device : torch.device) -> None:
        self.X = X
        self.Y = Y
        
        self.device = device
        self.edges = edges
        self.leni = leni
        self.list_graph = [read_object(f, p) for (f, p) in list_graph_file]
        self.other_scale = [graph.scale for graph in self.list_graph]
        self.Y_other_graph = []
        
        for i, graph in enumerate(self.list_graph):
            if i == 0:
                continue
            p = list_graph_file[i][1]
            df_train_scale = read_object(f'df_train_full_{graph.scale}_0_{graph.base}_{graph.graph_method}.pkl', p / 'occurence_voting')
            df_val_scale = read_object(f'df_val_full_{graph.scale}_0_{graph.base}_{graph.graph_method}.pkl', p / 'occurence_voting')
            df_test_scale = read_object(f'df_test_full_{graph.scale}_0_{graph.base}_{graph.graph_method}.pkl', p / 'occurence_voting')

            df_scale = pd.concat((df_train_scale, df_val_scale, df_test_scale)).reset_index(drop=True)

            if df_scale not in np.unique(df_scale.columns):
                df_scale['scale'] = graph.scale
                if graph.scale == 'departement':
                    df_scale['scale'] = 10

            Y_scale = df_scale[ids_columns + targets_columns + [target_name]].values
            Y_scale = np.asarray(Y_scale, dtype=np.float32)
            self.Y_other_graph.append(Y_scale)

    def __getitem__(self, index) -> tuple:
        x = self.X[index]
        y = self.Y[index]

        date_Y = np.unique(y[:, date_index, -1])

        Y = [y]

        for i, y1 in enumerate(self.Y_other_graph):
            mask = np.isin(self.Y_other_graph[i][:, date_index], date_Y)
            new_Y = self.Y_other_graph[i][mask]
            new_Y = new_Y[:, :, np.newaxis]
            Y.append(new_Y)

        if len(self.edges) > 0:
            edges = self.edges[index]
        else:
            edges = []
        
        X, Y, E = torch.tensor(x, dtype=torch.float32, device=self.device), \
            Y, \
            torch.tensor(edges, dtype=torch.long, device=self.device),  \
            
        return X, Y, self.list_graph

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass
    
    def get(self):
        pass

# Créez une classe de Dataset qui applique les transformations
class AugmentedInplaceGraphDataset(Dataset):
    def __init__(self, X, y, edges, transform=None, device=torch.device('cpu')):
        self.X = X
        self.y = y
        self.transform = transform
        self.device = device
        self.edges = edges

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # Appliquez la transformation si elle est définie
        if self.transform:
            x, y = self.transform(x, y)

        if len(self.edges) != 0:
            edges = self.edges[idx]
        else:
            edges = []

        return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), \
            torch.tensor(edges, dtype=torch.long, device=self.device)
    
    def len(self):
        pass

    def get(self):
        pass

class RandomFlipRotateAndCrop:
    def __init__(self, proba_flip, size_crop, max_angle):

        self.proba_flip = proba_flip
        self.size_crop = size_crop
        self.max_angle = max_angle

    def __call__(self, image, mask):
        # Random rotation angle
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32)

        angle = random.uniform(-self.max_angle, self.max_angle)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Random vertical flip
        if random.random() < self.proba_flip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random horizontal flip
        if random.random() < self.proba_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        return image, mask

def graph_collate_fn(batch):
    #node_indice_list = []
    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []
    num_nodes_seen = 0
    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        # Collecter les caractéristiques et les étiquettes des nœuds
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])
        
        edge_index = features_labels_edge_index_tuple[2]  # Tous les composants sont dans la plage [0, N]
        
        # Ajuster l'index des arêtes en fonction du nombre de nœuds vus
        if edge_index.shape[0] > 2:
            edge_index[0] += num_nodes_seen
            edge_index[1] += num_nodes_seen
            edge_index_list.append(edge_index)
        else:
            edge_index_list.append(edge_index + num_nodes_seen)

        # Ajouter l'ID du graphe pour chaque nœud
        num_nodes = features_labels_edge_index_tuple[1].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs
        
        num_nodes_seen += num_nodes  # Mettre à jour le nombre de nœuds vus

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)

    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
    #node_indices = torch.cat(node_indice_list, 0)
    graph = dgl.graph((edge_index[0], edge_index[1]))

    return node_features, node_labels, graph, graph_labels_list

def match_indices(pos, sample, atol=1e-6):
    # Utilise un dtype stable
    pos = pos.to(dtype=torch.float64)
    sample = sample.to(dtype=torch.float64)

    D = torch.cdist(sample, pos)            # [M, N]
    close = D <= atol                       # [M, N] True si sample m est proche de pos n
    valid_pos = close.any(dim=0)            # masque côté POS (N)
    pos_idx_valid = torch.where(valid_pos)[0]
    #j = 7
    #print('shape D =', D.shape)                 # doit être [M, N]
    #print('col7 min =', D[:, j].min().item())
    #print('any(col7 <= atol) =', bool((D[:, j] <= atol).any()))

    valid_pos = (D <= atol).any(dim=0)          # masque côté POS (N)
    #print('valid_pos[7] =', bool(valid_pos[j]))

    # indices pos valides:
    idx_pos = torch.where(valid_pos)[0]
    #print('index valid (pos) =', idx_pos)

    return pos_idx_valid, valid_pos

def mesh_subgraph_from_graphbuilder(mesh_graph, mesh_from_g2m, mesh_from_m2g,
                                    how="intersection"):
    """
    Crée un sous-graphe de mesh_graph en utilisant les nœuds mesh impliqués
    dans g2m_graph et m2g_graph.
    
    - how: 'intersection' (par défaut) ou 'union' entre les deux bipartites.
    """
    mesh_from_g2m = torch.unique(mesh_from_g2m)
    mesh_from_m2g = torch.unique(mesh_from_m2g)

    if how == "union":
        mesh_ids = torch.unique(torch.cat([mesh_from_g2m, mesh_from_m2g], dim=0))
    elif how == "intersection":
        mesh_ids = torch.tensor(
            np.intersect1d(mesh_from_g2m.cpu().numpy(), mesh_from_m2g.cpu().numpy()),
            device=mesh_from_g2m.device, dtype=mesh_from_g2m.dtype
        )
    else:
        raise ValueError("how must be 'union' or 'intersection'")

    # Sous-graphe induit par ces nœuds mesh
    mesh_subg = dgl.node_subgraph(mesh_graph, mesh_ids, relabel_nodes=True)
    return mesh_subg

def select_sample_graph(cartesian_grid, graph, graph_type, atol=1e-6, shrink_mesh=True):
    """
    Sous-graphe hétéro gardant uniquement les nœuds 'grid' correspondant à cartesian_grid
    (match par position) et, optionnellement, les nœuds 'mesh' connectés à ces 'grid'.

    Returns
    -------
    subg : dgl.DGLHeteroGraph
        Le sous-graphe sélectionné.
    valid_sample_idx : torch.Tensor (1D, dtype=long, device=cartesian_grid.device)
        Indices (dans cartesian_grid) qui ont été validés (présents dans le sous-graphe),
        dans l'ordre d'apparition de cartesian_grid (sans doublons).
    """
    if graph_type not in {"g2m", "m2g"}:
        raise ValueError("graph_type must be 'g2m' or 'm2g'")

    idtype = graph.idtype  # ex: torch.int32
    device = graph.device  # même device que le graphe

    # 1) positions des nœuds GRID selon le type de graphe
    if graph_type == "g2m":
        pos_grid = graph.srcdata["pos"]            # grid positions
        etype = ("grid", "g2m", "mesh")
    else:  # "m2g"
        pos_grid = graph.dstdata["pos"]            # grid positions
        etype = ("mesh", "m2g", "grid")
    
    sample = cartesian_grid.to(dtype=pos_grid.dtype, device=pos_grid.device)
    
    #print('Original graph', pos_grid)

    # 2) IDs locaux des nœuds GRID qui matchent 'sample'
    #    match_indices doit renvoyer (idx_pos_grid, idx_sample) alignés
    grid_ids, sample_ids = match_indices(pos_grid, sample, atol=atol)

    #print('valid graph grid', pos_grid[grid_ids])

    if grid_ids.numel() == 0:
        empty = torch.tensor([], dtype=idtype, device=device)
        subg = dgl.node_subgraph(graph, {'grid': empty, 'mesh': empty})
        return subg

    grid_ids = grid_ids.to(dtype=idtype, device=device)

    # 3) IDs des nœuds MESH à garder
    u, v = graph.edges(etype=etype)
    if graph_type == "g2m":
        # u: grid, v: mesh
        mask_e = torch.isin(u, grid_ids)
        mesh_ids = torch.unique(v[mask_e])
    else:
        # u: mesh, v: grid
        mask_e = torch.isin(v, grid_ids)
        mesh_ids = torch.unique(u[mask_e])

    if not shrink_mesh:
        mesh_ids = torch.arange(graph.num_nodes('mesh'), device=device, dtype=idtype)
    else:
        mesh_ids = mesh_ids.to(dtype=idtype, device=device)

    # 4) Sous-graphe hétéro
    subg = dgl.node_subgraph(graph, {
        'grid': grid_ids,
        'mesh': mesh_ids,
    },
    relabel_nodes=True)

    #print('subgraph', subg.srcdata["pos"])

    return subg, mesh_ids

def graph_collate_fn_mesh(batch):
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []

    graph_list = []
    graph_mesh_list = []
    grid2mesh_list = []
    mesh2grid_list = []

    num_nodes_seen = 0

    for graph_id, features_labels_graph_index_tuple in enumerate(batch):
        # Collecter les caractéristiques et les étiquettes des nœuds

        graph_mesh_ = features_labels_graph_index_tuple[3]
        gridh2mesh_ = features_labels_graph_index_tuple[4]
        mesh2graph_ = features_labels_graph_index_tuple[5]

        latitude_batch = features_labels_graph_index_tuple[1][:, latitude_index, -1].reshape(-1,1)
        longitude_batch = features_labels_graph_index_tuple[1][:, longitude_index, -1].reshape(-1,1)
        departement_batch = features_labels_graph_index_tuple[1][:, departement_index, -1].reshape(-1,)

        g_lat_lon_grid = torch.concat((latitude_batch, longitude_batch), dim=1).to('cpu')
        #print('Departement', torch.unique(departement_batch))
        cartesian_grid = latlon_points_to_xyz(g_lat_lon_grid.view(-1,2))
        #print('Original cartesian grid', cartesian_grid)
        ## Select nodes in grid2mesh, graph_mesh, and mesh2graph
        gridh2mesh, mesh_ids_g2m = select_sample_graph(cartesian_grid, gridh2mesh_, 'g2m')
        mesh2graph, mesh_ids_mg2 = select_sample_graph(cartesian_grid, mesh2graph_, 'm2g')

        #mesh2graph, gridh2mesh = restrict_mesh2graph_to_gridh2mesh(mesh2graph, gridh2mesh)

        def _check_non_empty_edges(g, etype, name):
                e = g.num_edges(etype)
                if e == 0:
                    print(f"[WARN] {name}: 0 edges pour etype {etype}.")
                return e

        _check_non_empty_edges(gridh2mesh, ("grid","g2m","mesh"), "gridh2mesh")
        _check_non_empty_edges(mesh2graph, ("mesh","m2g","grid"), "mesh2graph")

        graph_mesh = mesh_subgraph_from_graphbuilder(graph_mesh_, mesh_ids_g2m, mesh_ids_mg2, how="union")
        
        num_nodes = features_labels_graph_index_tuple[1].size(0)
        num_nodes_seen += num_nodes  # Mettre à jour le nombre de nœuds vus
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs

        #print("#############################################################################")

        #print(gridh2mesh_)
        #print(mesh2graph_)
        #print(graph_mesh)
        #exit(1)

        #print(torch.unique(features_labels_graph_index_tuple[1][:, departement_index, -1]))

        node_features_list.append(features_labels_graph_index_tuple[0])
        node_labels_list.append(features_labels_graph_index_tuple[1])

        #print(gridh2mesh)
        #print(mesh2graph)

        #print(torch.unique(features_labels_graph_index_tuple[1][:, departement_index, -1]))

        graph_mesh_list.append(graph_mesh)
        grid2mesh_list.append(gridh2mesh)
        mesh2grid_list.append(mesh2graph)
        
    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)

    for g in graph_mesh_list:  # graphs est une liste de dgl.graph
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.num_edges(), dtype=torch.int32)
        if '_ID' not in g.ndata:
            g.ndata['_ID'] = torch.arange(g.num_nodes(), dtype=torch.int32)

    graph_list.append(dgl.batch(graph_mesh_list))
    graph_list.append(dgl.batch(grid2mesh_list))
    graph_list.append(dgl.batch(mesh2grid_list))

    graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
    return node_features, node_labels, graph_list, graph_labels_list

"""def graph_collate_fn_mesh(batch):
    #node_indice_list = []
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []

    graph_list = []
    graph_mesh_list = []
    grid2mesh_list = []
    mesh2grid_list = []

    num_nodes_seen = 0

    last_graph_1 = None
    last_graph_2 = None
    last_graph_3 = None

    for graph_id, features_labels_graph_index_tuple in enumerate(batch):
        # Collecter les caractéristiques et les étiquettes des nœuds
        node_features_list.append(features_labels_graph_index_tuple[0])
        node_labels_list.append(features_labels_graph_index_tuple[1])

        icospheres_graph_path = features_labels_graph_index_tuple[3]
        
        #graph_mesh = features_labels_graph_index_tuple[4]
        #gridh2mesh = features_labels_graph_index_tuple[5]
        #mesh2graph = features_labels_graph_index_tuple[6]

        num_nodes = features_labels_graph_index_tuple[1].size(0)
        num_nodes_seen += num_nodes  # Mettre à jour le nombre de nœuds vus
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))  # Création d'un tensor d'IDs

        latitudes = features_labels_graph_index_tuple[1][:, latitude_index, -1].reshape(-1,1)
        longitudes = features_labels_graph_index_tuple[1][:, longitude_index, -1].reshape(-1,1)

        g_lat_lon_grid = torch.concat((latitudes, longitudes), dim=1).to('cpu')
        g_lat_lon_grid = torch.unique(g_lat_lon_grid, dim=0)
        graph_builder = GraphBuilder(icospheres_graph_path, g_lat_lon_grid, doPrint=False)

        graph_mesh = graph_builder.create_mesh_graph(last_graph_1)
        gridh2mesh, graph_mesh = graph_builder.create_g2m_graph(last_graph_2, graph_mesh)
        mesh2graph = graph_builder.create_m2g_graph(last_graph_3)

        #gridh2mesh.ndata['_ID'] = torch.arange(gridh2mesh.num_nodes(), dtype=torch.int32)
        #mesh2graph.ndata['_ID'] = torch.arange(mesh2graph.num_nodes(), dtype=torch.int32)

        graph_mesh_list.append(graph_mesh)
        grid2mesh_list.append(gridh2mesh)
        mesh2grid_list.append(mesh2graph)

        #last_graph_1 = deepcopy(graph_mesh)
        #last_graph_2 = deepcopy(gridh2mesh)
        #last_graph_3 = deepcopy(mesh2graph)

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)

    for g in graph_mesh_list:  # graphs est une liste de dgl.graph
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.num_edges(), dtype=torch.int32)
        if '_ID' not in g.ndata:
            g.ndata['_ID'] = torch.arange(g.num_nodes(), dtype=torch.int32)

    graph_list.append(dgl.batch(graph_mesh_list))
    graph_list.append(dgl.batch(grid2mesh_list))
    graph_list.append(dgl.batch(mesh2grid_list))
    
    graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
    return node_features, node_labels, graph_list, graph_labels_list"""

def graph_collate_fn_multiple_graph(batch):
    #node_indice_list = []
    node_features_list = []
    node_labels_list = []
    graph_labels_list = []  

    graph_list = []
    graph_scale_list = []
    decrease_scale = []
    increase_scale = []

    for graph_id, features_labels_graph_index_tuple in enumerate(batch):

        g_lat_lon_grid_scales = []
        for labels in features_labels_graph_index_tuple[1]:
            labels = np.asarray(labels, dtype=np.float32)
            if labels.ndim == 3:
                latitudes = torch.Tensor(labels[:, latitude_index, -1])
                longitudes = torch.Tensor(labels[:, longitude_index, -1])
            else:
                latitudes = torch.Tensor(labels[:, latitude_index])
                longitudes = torch.Tensor(labels[:, longitude_index])

            if latitudes.ndim == 1:
                latitudes = latitudes[:, None]
                longitudes = longitudes[:, None]

            g_lat_lon_grid = torch.concat((latitudes, longitudes), dim=1).to('cpu')
            #g_lat_lon_grid = torch.unique(g_lat_lon_grid, dim=0)
            g_lat_lon_grid_scales.append(g_lat_lon_grid)

        graph_builder = GraphBuilder2(g_lat_lon_grid_scales, features_labels_graph_index_tuple[1], features_labels_graph_index_tuple[2], date_index, id_index)

        current_graph_scale_list = graph_builder.graph_scale()
        increase_graph_scale_list, current_graph_scale_list = graph_builder.increase_scale(current_graph_scale_list)
        if increase_graph_scale_list is None:
            continue
        decrease_graph_scale_list = graph_builder.decrease_scale()
        
        for i, labels in enumerate(features_labels_graph_index_tuple[1]):
            node_graph = current_graph_scale_list[i]
            original_node_indices = node_graph.ndata[dgl.NID]

            if i == 0:
                node_features = features_labels_graph_index_tuple[0][original_node_indices]
                node_features_list.append(node_features)      
                labels = labels[original_node_indices]    

            if 'mask_zeros' in locals():
                if i > 0:
                    increase_graph = increase_graph_scale_list[i - 1]
                    decrease_graph = decrease_graph_scale_list[i - 1]
                    #print('###########################')
                    #print(increase_graph)

                    #print({ntype: increase_graph.nodes(ntype) for ntype in increase_graph.ntypes})

                    #increase_graph = dgl.node_subgraph(increase_graph, nodes={ntype: increase_graph.nodes(ntype) for ntype in increase_graph.ntypes})
                    #decrease_graph = dgl.node_subgraph(decrease_graph, nodes={ntype: decrease_graph.nodes(ntype) for ntype in decrease_graph.ntypes})

                    increase_graph_scale_list[i - 1] = increase_graph
                    decrease_graph_scale_list[i - 1] = decrease_graph

                    num_nodes = current_graph_scale_list[i -1].num_nodes()
                    labels = labels[original_node_indices]

                    if labels.ndim == 2:
                        labels = labels[None, :, :]

                    edge_type = list(increase_graph.canonical_etypes)[0]
                    src_nodes, dst_nodes = increase_graph.edges(etype=edge_type)

                    # Masque destination : destination avec -1,-1 == 0

                    dst_mask = (labels[dst_nodes, -1, -1] == 0)
                    #print(increase_graph)
                    #print(np.unique(src_nodes))
                    valid_src_mask = mask_zeros[src_nodes]

                    # Final: on garde les dst liés à des src masqués et eux-mêmes à 0
                    final_mask = valid_src_mask & dst_mask

                    labels[dst_nodes[final_mask], weight_index] = 0
            
            if labels.shape[2] > 1:
                labels = labels[:, :, -1]
            mask_zeros = (labels[:, weight_index, -1] == 0)

            labels = torch.Tensor(labels)
            node_labels_list.append(labels)

        #graph_scale_list = dgl.batch(graph_scale_list)
        #increase_graph_scale_list = dgl.batch(increase_graph_scale_list)
        #decrease_graph_scale_list = dgl.batch(decrease_graph_scale_list)

        graph_scale_list.append(current_graph_scale_list)
        increase_scale.append(increase_graph_scale_list)
        decrease_scale.append(decrease_graph_scale_list)

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0).to(node_features.device)

    """for g in graph_mesh_list:  # graphs est une liste de dgl.graph
        if '_ID' not in g.edata:
            g.edata['_ID'] = torch.arange(g.num_edges(), dtype=torch.int32)
        if '_ID' not in g.ndata:
            g.ndata['_ID'] = torch.arange(g.num_nodes(), dtype=torch.int32)"""

    # Suppose graph_list, increase_scale, and decrease_scale are lists of lists of DGLGraphs
    
    batched_graph_list = [dgl.batch(graphs_at_i) for graphs_at_i in zip(*graph_scale_list)]
    batched_increase = [dgl.batch(graphs_at_i) for graphs_at_i in zip(*increase_scale)]
    batched_decrease = [dgl.batch(graphs_at_i) for graphs_at_i in zip(*decrease_scale)]

    graph_list.append(batched_graph_list)
    graph_list.append(batched_increase)
    graph_list.append(batched_decrease)
    
    #graph_labels_list = torch.cat(graph_labels_list, 0).to(device)
    return node_features, node_labels, graph_list, graph_labels_list

def graph_collate_fn_hybrid(batch):
    edge_index_list = []
    node_features_list = []
    node_features_list_2D = []
    node_labels_list = []
    graph_labels_list = []
    num_nodes_seen = 0

    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_features_list_2D.append(features_labels_edge_index_tuple[1])
        node_labels_list.append(features_labels_edge_index_tuple[2])
        edge_index = features_labels_edge_index_tuple[3]  # all of the components are in the [0, N] range
        
        # Adjust edge indices
        if edge_index.shape[0] > 2:
            edge_index[0] += num_nodes_seen
            edge_index[1] += num_nodes_seen
            edge_index_list.append(edge_index)
        else:
            edge_index_list.append(edge_index + num_nodes_seen)
        
        # Add graph ID for each node
        num_nodes = features_labels_edge_index_tuple[1].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        num_nodes_seen += num_nodes  # Update the number of nodes seen

    # Merge all components into single tensors
    node_features = torch.cat(node_features_list, 0)
    node_features_2D = torch.cat(node_features_list_2D, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    graph_labels = torch.cat(graph_labels_list, 0).to(device)

    return node_features, node_features_2D, node_labels, edge_index, graph_labels

def graph_collate_fn_no_label(batch):
    edge_index_list = []
    node_features_list = []
    graph_labels_list = []
    num_nodes_seen = 0

    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        node_features_list.append(features_labels_edge_index_tuple[0])
        edge_index = features_labels_edge_index_tuple[1]
        
        # Adjust edge indices
        edge_index_list.append(edge_index + num_nodes_seen)
        
        # Add graph ID for each node
        num_nodes = features_labels_edge_index_tuple[0].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        num_nodes_seen += len(features_labels_edge_index_tuple[0])  # Update the number of nodes seen

    # Merge all components into single tensors
    node_features = torch.cat(node_features_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    graph_labels = torch.cat(graph_labels_list, 0).to(device)

    return node_features, edge_index, graph_labels

def graph_collate_fn_adj_mat(batch):
    node_features_list = []
    node_labels_list = []
    edge_index_list = []
    graph_labels_list = []
    num_nodes_seen = 0

    for graph_id, features_labels_edge_index_tuple in enumerate(batch):
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])
        edge_index = features_labels_edge_index_tuple[2]
        
        # Adjust edge indices
        edge_index_list.append(edge_index + num_nodes_seen)
        
        # Add graph ID for each node
        num_nodes = features_labels_edge_index_tuple[1].size(0)
        graph_labels_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        num_nodes_seen += num_nodes  # Update the number of nodes seen

    # Merge all components into single tensors
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    # Create adjacency matrix
    adjacency_matrix = torch.zeros(num_nodes_seen, num_nodes_seen)
    for edge in edge_index.t():  # Transpose to iterate through edge pairs
        node1, node2 = edge[0].item(), edge[1].item()
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # Since it's an undirected graph

    graph_labels = torch.cat(graph_labels_list, 0)

    return node_features, node_labels, adjacency_matrix, graph_labels.to(device)

def construct_dataset(date_ids, x_data, y_data, graph, ids_columns, ks, horizon, use_temporal_as_edges, isNotmesh=False, proportion_0_sample_with_positive_weight=1.0):
    Xs, Ys, Es = [], [], []
    
    """if graph.graph_method == 'graph':
        # Traiter par identifiant de graph
        graphId = np.unique(x_data[:, graph_id_index])
        for id in graphId:
            x_data_graph = x_data[x_data[:, graph_id_index] == id]
            y_data_graph = y_data[y_data[:, graph_id_index] == id]
            for date_id in date_ids:
                if date_id not in np.unique(x_data_graph[:, date_index]):
                    continue
                if use_temporal_as_edges is None:
                    x, y = construct_time_series(date_id, x_data_graph, y_data_graph, ks, len(ids_columns))
                    if x is not None:
                        for i in range(x.shape[0]):
                            Xs.append(x[i])
                            Ys.append(y[i])
                    continue
                elif use_temporal_as_edges:
                    x, y, e = construct_graph_set(graph, date_id, x_data_graph, y_data_graph, ks, len(ids_columns), mesh=mesh)
                else:
                    x, y, e = construct_graph_with_time_series(graph, date_id, x_data_graph, y_data_graph, ks, len(ids_columns), mesh=mesh)

                if x is None:
                    continue

                if x.shape[0] == 0:
                    continue

                Xs.append(x)
                Ys.append(y)
                Es.append(e)
    
    else:"""
    # Traiter par date
    print(ks, horizon)
    for id in date_ids:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(id, x_data, y_data, ks, horizon, len(ids_columns), proportion_0_with_positive_weight=proportion_0_sample_with_positive_weight)
            if x is not None and isNotmesh:
                for i in range(x.shape[0]):
                    Xs.append(x[i])
                    Ys.append(y[i])
            elif x is not None:
                Xs.append(x)
                Ys.append(y)
            continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, x_data, y_data, ks, horizon, len(ids_columns), proportion_0_with_positive_weight=proportion_0_sample_with_positive_weight)
        else:
            x, y, e = construct_graph_with_time_series(graph, id, x_data, y_data, ks, horizon, len(ids_columns), proportion_0_with_positive_weight=proportion_0_sample_with_positive_weight)

        if x is None:
            continue

        if x.shape[0] == 0:
            continue
        
        Xs.append(x)
        Ys.append(y)
        Es.append(e)
    
    return Xs, Ys, Es

def create_dataset(graph,
                    df_train,
                    df_val,
                    df_test,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    horizon: int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None,
                    proportion_0_with_positive_weight=1.0
                    ):
    
    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))
    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}, {dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_with_positive_weight)

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_with_positive_weight)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_with_positive_weight)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    assert len(XsTe) > 0, "Le jeu de données de test est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    elif graph_mesh is not None:
        train_dataset = InplaceMeshGraphDatasetInplace(Xst, Yst, Est, len(Xst), device, graph_mesh, gridh2mesh, mesh2graph)
        val_dataset = InplaceMeshGraphDatasetInplace(XsV, YsV, EsV, len(XsV), device, graph_mesh, gridh2mesh, mesh2graph)
        test_dataset = InplaceMeshGraphDatasetInplace(XsTe, YsTe, EsTe, len(XsTe), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    train_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, Xst, Yst, Est, len(Xst), device)
    #    val_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsV, YsV, EsV, len(XsV), device)
    #    test_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsTe, YsTe, EsTe, len(XsTe), device)

    return train_dataset, val_dataset, test_dataset

def create_train_dataset(graph,
                    df_train,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    horizon:int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None,
                    proportion_0_sample_with_positive_weight=1.0
                    ):

    x_train, y_train = df_train[ids_columns + features_name].values, df_train[ids_columns + targets_columns + [target_name]].values
    
    #print('weight', df_train['weight'].unique())

    dateTrain = np.sort(np.unique(y_train[y_train[:, weight_index] > 0, date_index]))

    logger.info(f'{dateTrain.shape}')

    logger.info(f'Constructing train Dataset')
    Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None, proportion_0_sample_with_positive_weight)

    # Assurez-vous que les ensembles ne sont pas vides
    assert len(Xst) > 0, "Le jeu de données d'entraînement est vide"

    if graph_mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
    elif graph_mesh is not None:
        train_dataset = InplaceMeshGraphDatasetInplace(Xst, Yst, Est, len(Xst), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    train_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, Xst, Yst, Est, len(Xst), device)

    return train_dataset

def create_test_val_dataset(graph,
                    df_val,
                    df_test,
                    features_name,
                    target_name,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    horizon: int,
                    graph_mesh=None,
                    gridh2mesh=None,
                    mesh2graph=None,
                    proportion_0_sample_with_positive_weight=1.0
                    ):
        
    x_val, y_val = df_val[ids_columns + features_name].values, df_val[ids_columns + targets_columns + [target_name]].values

    x_test, y_test = df_test[ids_columns + features_name].values, df_test[ids_columns + targets_columns + [target_name]].values

    dateVal = np.sort(np.unique(y_val[y_val[:, weight_index] > 0, date_index]))
    dateTest = np.sort(np.unique(y_test[y_test[:, weight_index] > 0, date_index]))

    logger.info(f'{dateVal.shape}, {dateTest.shape}')

    logger.info(f'Constructing val Dataset')
    XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)

    logger.info(f'Constructing test Dataset')
    XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, horizon, use_temporal_as_edges, graph_mesh is None)
    
    # Assurez-vous que les ensembles ne sont pas vides
    assert len(XsV) > 0, "Le jeu de données de validation est vide"
    if len(XsTe) == 0:
        XsTe, YsTe, EsTe = XsV, YsV, EsV

    if gridh2mesh is None:
        # Création des datasets finaux
        print('uzbdkazdkjzan')
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_dataset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    elif gridh2mesh is not None:
        val_dataset = InplaceMeshGraphDatasetInplace(XsV, YsV, EsV, len(XsV), device, graph_mesh, gridh2mesh, mesh2graph)
        test_dataset = InplaceMeshGraphDatasetInplace(XsTe, YsTe, EsTe, len(XsTe), device, graph_mesh, gridh2mesh, mesh2graph)
    #elif mesh == 'mygraph':
    #    val_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsV, YsV, EsV, len(XsV), device)
    #    test_dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, XsTe, YsTe, EsTe, len(XsTe), device)

    return val_dataset, test_dataset

def get_numpy_data(graph, df,
                       features_name,
                       use_temporal_as_edges : bool,
                       ks :int,
                       horizon:int,
                       
                       ):

    Xset = df[ids_columns + features_name].values

    X = []
    E = []
    Yset = None

    graphId = np.unique(Xset[:, date_index])
    for date in graphId:
        if use_temporal_as_edges is None:
            x, _ = construct_time_series(date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
            if x is not None:
                for i in range(x.shape[0]):
                    X.append(x[i])
            continue
        elif use_temporal_as_edges:
            x, _, e = construct_graph_set(graph, date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
        else:
            x, _, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)

        if x is None:
            continue

        if x.shape[0] == 0:
            continue

        X.append(x)
        if 'e' in locals():
            E.append(e)

    return np.asarray(X), np.asarray(E)

def create_test_loader(graph, df,
                       features_name,
                       device : torch.device,
                       use_temporal_as_edges : bool,
                       target_name,
                       ks :int,
                       horizon:int,
                       graph_mesh=None,
                        gridh2mesh=None,
                        mesh2graph=None,
                        proportion_0_sample_witg_positive_weight=1.0):

    if 'DFE' not in df.columns:
        df['DFE'] = 0
    Xset, Yset = df[ids_columns + features_name].values, df[ids_columns + targets_columns + [target_name]].values

    X = []
    Y = []
    E = []

    """if graph.graph_method == 'graph':
        graphId = np.unique(Xset[:, graph_id_index])
        for id in graphId:
            Xset_graph = Xset[Xset[:, graph_id_index] == id]
            Yset_graph = Yset[Yset[:, graph_id_index] == id]
            udates = np.unique(Xset_graph[:, date_index])
            for date in udates:
                if use_temporal_as_edges is None:
                    x, y = construct_time_series(date, Xset_graph, Yset_graph, ks, len(ids_columns))
                    if x is not None:
                        for i in range(x.shape[0]):
                            X.append(x[i])
                            Y.append(y[i])
                    continue
                elif use_temporal_as_edges:
                    x, y, e = construct_graph_set(graph, date, Xset_graph, Yset_graph, ks, len(ids_columns))
                else:
                    x, y, e = construct_graph_with_time_series(graph, date, Xset_graph, Yset_graph, ks, len(ids_columns))

                if x is None:
                    continue

                if x.shape[0] == 0:
                    continue

                X.append(x)
                Y.append(y)
                E.append(e)
    else:"""
    graphId = np.unique(Xset[:, date_index])
    for date in graphId:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
            if x is not None:
                for i in range(x.shape[0]):
                    X.append(x[i])
                    Y.append(y[i])
            continue
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, date, Xset, Yset, ks, horizon, len(ids_columns), 1.0)
        else:
            x, y, e = construct_graph_with_time_series(graph, date, Xset, Yset, ks, horizon,len(ids_columns), 1.0)

        if x is None:
            continue

        if x.shape[0] == 0:
            continue

        X.append(x)
        Y.append(y)
        E.append(e)

    if gridh2mesh is None:
        dataset = InplaceGraphDataset(X, Y, E, len(X), device)
        collate = graph_collate_fn
    elif gridh2mesh is not None:
        dataset = InplaceMeshGraphDatasetInplace(X, Y, E, len(X), device, graph_mesh, gridh2mesh, mesh2graph)
        collate = graph_collate_fn_mesh
    #elif mesh == 'mygraph':
    #    dataset = InplaceMulitpleGraphDataset(target_name, mesh_file, X, Y, E, len(X), device)
    #    collate = graph_collate_fn_multiple_graph
    else:
        raise ValueError(f'{mesh} is not a known mesh value')

    if use_temporal_as_edges is None:
        loader = DataLoader(dataset, dataset.__len__(), False, worker_init_fn=seed_worker,
            generator=g)
    else:
        loader = DataLoader(dataset, dataset.__len__(), False, collate_fn=collate,
                            worker_init_fn=seed_worker,
        generator=g)

    return loader

def load_x_from_pickle(date : int,
                       path : Path,
                       features_name_2D : list,
                       features : list,
                       features_1D,
                       raster : np.ndarray,
                       x_1d,
                       y_1d,
                       name_exp : str,
                       ) -> np.array:
    
    features_name_2D_full, _ = get_features_name_lists_2D(6, features)

    dir_encoder = path / '../../'

    encoder_osmnx = read_object(f'encoder_osmnx.pkl_{name_exp}', dir_encoder)
    encoder_foret = read_object(f'encoder_foret_{name_exp}.pkl', dir_encoder)
    encoder_argile = read_object(f'encoder_argile_{name_exp}.pkl', dir_encoder)
    encoder_cosia = read_object(f'encoder_cosia_{name_exp}.pkl', dir_encoder)

    leni = len(features_name_2D)
    if date < 0:
        return None
    x_2D = read_object(f'X_{date}.pkl', path)

    if x_2D is None:
        return None
    new_x_2D = np.empty((leni, x_2D.shape[1], x_2D.shape[2]))
    
    for i, fet_2D in enumerate(features_name_2D):
        
        #print(fet_2D, np.nanmax(x_2D[features_name_2D_full.index(fet_2D)]))
        """plt.imshow(x_2D[features_name_2D_full.index(fet_2D)])
        plt.colorbar()
        plt.savefig(f'{fet_2D}.png')
        plt.close('all')       """ 
        if fet_2D == 'Past_risk' or fet_2D == 'Past_burnedarea' or fet_2D == 'cluster_encoder' or fet_2D == 'id_encoder':
            unode = np.unique(raster) 
            for node in unode:
                mask = (raster == node)
                m1 = (y_1d[:, id_index] == node)
                if True not in m1:
                    new_x_2D[i, mask] = 0
                else:
                    new_x_2D[i, mask] = x_1d[m1, features_1D.index(fet_2D)]

        elif fet_2D == 'foret_encoder' in features:
            #logger.info('Foret landcover')
            assert encoder_foret is not None
            new_x_2D[i, :, :] = encoder_foret.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        elif fet_2D == 'highway_encoder' in features:
            #logger.info('OSMNX landcover')
            new_x_2D[i, :, :] = encoder_osmnx.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        elif fet_2D == 'argile_encoder' in features:
            #logger.info('OSMNX landcover')
            assert encoder_argile is not None
            new_x_2D[i, :, :] = encoder_argile.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        elif fet_2D == 'cosia_encoder' in features:
            #logger.info('OSMNX landcover')
            assert encoder_cosia is not None
            new_x_2D[i, :, :] = encoder_cosia.transform(x_2D[features_name_2D_full.index(fet_2D), :, :].reshape(-1,1)).values.reshape((x_2D.shape[1], x_2D.shape[2]))

        else:
            new_x_2D[i, :, :] = x_2D[features_name_2D_full.index(fet_2D), :, :]
        if False not in np.isnan(new_x_2D[i, :, :]):
            return None
        else:
            nan_mask = np.isnan(new_x_2D[i, :, :])
            new_x_2D[i, nan_mask] = np.nanmean(new_x_2D[i, :, :])
            #new_x_2D[i, nan_mask] = 0.0
        #   print(np.unique(np.isnan(new_x_2D)))
    #exit(1)
    #
    #  Remplacer les NaN dans une matrice entière
    #nan_mean = np.nanmean(new_x_2D, axis=(1, 2), keepdims=True)  # Moyenne par plan
    #new_x_2D = np.where(np.isnan(new_x_2D), nan_mean, new_x_2D)  # Remplacement conditionnel

    return new_x_2D

def generate_image_y(y, y_raster):
    res = np.empty((y.shape[1], *y_raster.shape, y.shape[-1]))
    for i in range(y.shape[0]):
        graph = y[i][graph_id_index][0]
        node = y[i][id_index][0]
        longitude = y[i][longitude_index][0]
        latitude = y[i][latitude_index][0]
        departement = y[i][departement_index][0]
        mask = y_raster == node
        if mask.shape[0] == 0:
            logger.info(f'{node} not in {np.unique(y_raster)}')
        res[graph_id_index, mask, :] = graph
        res[id_index, mask, :] = node
        res[latitude_index, mask, :] = latitude
        res[longitude_index, mask, :] = longitude
        res[departement_index, mask, :] = departement
        for j in range(weight_index, y.shape[1]):
            for k in range(y.shape[2]):
                res[j, mask, k] = y[i, j, k]

    return res

def process_dept_raster(dept, graph, path, y, features_name_2D, ks, image_per_node, shape2D):
    """Process a department's raster and return processed X and Y data"""
    
    if image_per_node:
        X = np.zeros((y.shape[0], len(features_name_2D), *shape2D[graph.scale], ks + 1))
        Y = y
    else:
        X = np.zeros((len(features_name_2D), 64, 64, ks + 1))
        Y = np.zeros((64, 64, y.shape[1], ks + 1))

    raster_dept = read_object(f'{int2name[dept]}rasterScale{graph.scale}_{graph.base}_{graph.graph_method}_node.pkl', path / 'raster')
    assert raster_dept is not None

    raster_dept_graph = read_object(f'{int2name[dept]}rasterScale{graph.scale}_{graph.base}_{graph.graph_method}.pkl', path / 'raster')

    unodes = np.unique(raster_dept)
    unodes = unodes[~np.isnan(unodes)]

    return X, Y, raster_dept, raster_dept_graph, unodes

def process_time_step(X, Y, dept, graph, ks, id, path, features_name_2D, features, features_1D, x_1d, y_1d, raster_dept, raster_dept_graph, unodes, image_per_node, shape2D, name_exp):
    """Process each time step and update X and Y arrays"""
    for k in range(ks + 1):
        date = int(id) - (ks - k)
        x_date = load_x_from_pickle(date, path / f'2D_database' / int2name[dept], features_name_2D, features, features_1D, raster_dept, x_1d[:, :, k], y_1d[:, :, k], name_exp)

        if x_date is None:
            return None, None

        if x_date is None:
            x_date = np.zeros((len(features_name_2D), raster_dept.shape[0], raster_dept.shape[1]))
        
        if image_per_node:
            X = process_node_images(X, unodes, x_date, raster_dept, y_1d, graph.scale, k, date, shape2D)
        else:
            X = process_dept_images(X, x_date, raster_dept, ks, k)

    if not image_per_node:
        y_dept = generate_image_y(y_1d, raster_dept_graph)
        Y = process_dept_y(Y, y_dept, raster_dept_graph, ks)
    
    return X, Y

def process_node_images(X, unodes, x_date, raster_dept, y_1d, scale, k, date, shape2D):
    """Process images for each node."""
    node2remove = []
    for node in unodes:
        if node not in np.unique(y_1d[:, graph_id_index]):
            node2remove.append(node)
            continue
        mask = np.argwhere(raster_dept == node)
        x_node = extract_node_data(x_date, mask, node, raster_dept)
        X = update_node_images(X, x_node, mask, scale, k, shape2D, y_1d, date, node)
    
    return X

def extract_node_data(x_date, mask, node, raster_dept):
    """Extract and mask data for a specific node."""
    x_date_cp = np.copy(x_date)
    x_date_cp[:, raster_dept != node] = 0.0
    minx, miny, maxx, maxy = np.min(mask[:, 0]), np.min(mask[:, 1]), np.max(mask[:, 0]), np.max(mask[:, 1])
    res = x_date_cp[:, minx:maxx+1, miny:maxy+1]
    res[np.isnan(res)] = np.nanmean(res)
    return res

def update_node_images(X, x_node, mask, scale, k, shape2D, y_1d, date, node):
    """Resize and update node images in X array."""
    for band in range(x_node.shape[0]):
        x_band = x_node[band]
        if False not in np.isnan(x_band):
            x_band[np.isnan(x_band)] = -1
        else:
            x_band[np.isnan(x_band)] = np.nanmean(x_band)
        #index = np.argwhere((y[:, graph_id_index, 0] == node) & (y[:, date_index, :] == date))[:, 0]
        index = np.unique(np.argwhere((y_1d[:, graph_id_index, 0] == node))[:, 0])
        #print(np.unique(x_band))
        X[index, band, :, :, k] = resize_no_dim(x_band, *shape2D[scale])
        mask_nan = np.isnan(X[index, band, :, :, k])
        #X[index, band, mask_nan, k] = 0
    X[np.isnan(X)] = -1
    return X

def process_dept_images(X, x_date, raster_dept, ks, k):
    """Process department images and update X."""
    for band in range(x_date.shape[0]):
        x_band = x_date[band]
        x_band[np.isnan(raster_dept)] = np.nanmean(x_band)
        X[band, :, :, k] = resize_no_dim(x_band, 64, 64)
    
    X[np.isnan(X)] = 0.0
    return X

def process_dept_y(Y, y_date, raster_dept, ks):
    """Process department labels and update Y."""
    for band in range(y_date.shape[0]):
        for k in range(ks + 1):
            y_band = y_date[band, :, :, k]
            y_band[np.isnan(raster_dept)] = 0
            Y[:, :, band, k] = resize_no_dim(y_band, 64, 64)
    
    Y[np.isnan(Y)] = 0.0
    return Y

def create_dataset_2D_2(graph, X_np, Y_np, ks, dates,
                        features_name_2D, use_temporal_as_edges):
    """Main function to create the dataset."""
    Xst, Yst, Est = [], [], []
    leni = len(features_name_2D)
    for id in dates:
        if use_temporal_as_edges is None:
            x, y = construct_time_series(id, X_np, Y_np, ks, len(ids_columns))
        elif use_temporal_as_edges:
            x, y, e = construct_graph_set(graph, id, X_np, Y_np, ks, len(ids_columns))
        else:
            x, y, e = construct_graph_with_time_series(graph, id, X_np, Y_np, ks, len(ids_columns))
        
        if x is None:
            continue
        
        """depts = np.unique(y[:, departement_index].astype(int))
        for dept in depts:
            y_dept = y[y[:, departement_index, 0] == dept]
            x_dept = x[y[:, departement_index, 0] == dept]
            new_x = []
            new_y = []

            sub_dir = f'image_per_node_{leni}' if image_per_node else f'image_per_departement_{leni}'

            for i in range(x_dept.shape[0]):
                cluster_id = y_dept[i, graph_id_index, -1]
                if use_temporal_as_edges is None and image_per_node:
                    is_file = (path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / context / f'X_{int(id)}_{dept}_{cluster_id}.pkl').is_file()
                else:
                    is_file = (path / f'2D_database_{graph.scale}_{graph.base}_{graph.graph_method}' / sub_dir / context / f'X_{int(id)}_{dept}.pkl').is_file()
                if not is_file:
                        new_x.append(x_dept[i])
                        new_y.append(y_dept[i])
                else:
                    Xst.append(f'X_{int(id)}_{dept}_{cluster_id}.pkl')
                    Yst.append(y_dept[i])
            
            #if len(new_y) == 0:
            #    continue
            
            #new_x = np.asarray(new_x)
            #new_y = np.asarray(new_y)

            new_y = np.copy(y_dept)
            new_x = np.copy(x_dept)
            
            X, Y, raster_dept, raster_dept_graph, unodes = process_dept_raster(dept, graph, path, new_y, features_name_2D, ks, image_per_node, shape2D)
            X, Y = process_time_step(X, Y, dept, graph, ks, id, path, features_name_2D, features, features_1D, new_x, new_y, raster_dept, raster_dept_graph, unodes, image_per_node, shape2D, name_exp)
            #print(X.shape)
            
            if X is None:
                continue

            if use_temporal_as_edges is None and image_per_node:
                for i in range(X.shape[0]):
                    #print(np.unique(np.isnan(X[i])))
                    if True:
                        cluster_id = Y[i][graph_id_index, -1]
                        save_object(X[i], f'X_{int(id)}_{dept}_{cluster_id}.pkl', path / f'2D_database' / sub_dir / context)
                        Xst.append(f'X_{int(id)}_{dept}_{cluster_id}.pkl')
                        Yst.append(Y[i])
                    else:
                        Xst.append(X[i])
                        Yst.append(y[i])
            else:
                if True:
                    save_object(X, f'X_{int(id)}_{dept}.pkl', path / f'2D_database' / sub_dir / context)
                    Xst.append(f'X_{int(id)}_{dept}.pkl')
                else:
                    Xst.append(X)
                Y[np.isnan(Y)] = 0
                Yst.append(Y)"""

        if 'e' in locals():
            Est.append(e)

    return Xst, Yst, Est

def create_dataset_2D(graph,
                    df_train,
                    df_val,
                    df_test,
                    path,
                    features_name_2D,
                    features,
                    features_1D,
                    target_name,
                    image_per_node,
                    use_temporal_as_edges : bool,
                    device,
                    ks : int,
                    name_exp):

    if df_train is not None:
        x_train, y_train = df_train[ids_columns + features_1D].values, df_train[ids_columns + [target_name]].values
        dateTrain = np.sort(np.unique(y_train[np.argwhere(y_train[:, weight_index] > 0), date_index]))

    if df_val is not None:
        x_val, y_val = df_val[ids_columns + features_1D].values, df_val[ids_columns + [target_name]].values
        dateVal = np.sort(np.unique(y_val[np.argwhere(y_val[:, weight_index] > 0), date_index]))

    if df_test is not None:
        x_test, y_test = df_test[ids_columns + features_1D].values, df_test[ids_columns + [target_name]].values
        dateTest = np.sort(np.unique(y_test[np.argwhere(y_test[:, weight_index] > 0), date_index]))

    XsTe = []
    YsTe = []
    EsTe = []
    
    logger.info(f'Model configuration : image_per_node {image_per_node}, use_temporal_as_edges {use_temporal_as_edges}')

    if df_train is not None:
        logger.info('Creating train dataset')
        Xst, Yst, Est = construct_dataset(dateTrain, x_train, y_train, graph, ids_columns, ks, use_temporal_as_edges, True) 
        assert len(Xst) > 0
    
    if df_val is not None:
        logger.info('Creating val dataset')
        XsV, YsV, EsV = construct_dataset(dateVal, x_val, y_val, graph, ids_columns, ks, use_temporal_as_edges, True)
        assert len(XsV) > 0

    if df_test is not None:
        logger.info('Creating Test dataset')
        XsTe, YsTe, EsTe = construct_dataset(dateTest, x_test, y_test, graph, ids_columns, ks, use_temporal_as_edges, True)

        assert len(XsTe) > 0

    #logger.info(f'{len(Xst)}, {len(XsV)}, {len(XsTe)}')
    train_dataset = None
    val_dataset = None
    test_dataset = None
    if True:
        if df_train is not None:
            train_dataset = ReadGraphDataset_2D_from_xarray(Xst, Yst, Est, len(Xst), device,
                                                            rootDisk / 'csv', features_name_2D, features_1D, ks,
                                                            graph.scale,
                                                            graph.graph_method,
                                                            graph.base,
                                                            path / 'datacube')
        
        if df_val is not None:
            val_dataset = ReadGraphDataset_2D_from_xarray(XsV, YsV, EsV, len(XsV), device,
                                                          rootDisk / 'csv', features_name_2D, features_1D, ks,
                                                        graph.scale,
                                                            graph.graph_method,
                                                            graph.base,
                                                            path / 'datacube')
        if df_test is not None:
            test_dataset = ReadGraphDataset_2D_from_xarray(XsTe, YsTe, EsTe, len(XsTe), device,
                                                           rootDisk / 'csv', features_name_2D, features_1D, ks,
                                                            graph.scale,
                                                            graph.graph_method,
                                                            graph.base,
                                                            path / 'datacube')
    else:
        train_dataset = InplaceGraphDataset(Xst, Yst, Est, len(Xst), device)
        val_dataset = InplaceGraphDataset(XsV, YsV, EsV, len(XsV), device)
        test_datset = InplaceGraphDataset(XsTe, YsTe, EsTe, len(XsTe), device)
    return train_dataset, val_dataset, test_dataset

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def build_dataframe(
    inputs_horizon: torch.Tensor,   # shape: (X, F, T)
    labels: torch.Tensor,           # shape: (X, L, T)
    features_name,                  # list[str], len F
    ids_columns,                    # list[str]
    targets_columns,                # list[str]
    target_name                     # str
) -> pd.DataFrame:
    # --- vérifs de shapes ---
    X, F, T = inputs_horizon.shape
    X2, L, T2 = labels.shape
    assert X == X2,  f"X mismatch: {X} vs {X2}"
    assert T == T2,  f"T mismatch: {T} vs {T2}"
    expected_L = len(ids_columns) + len(targets_columns) + 1
    assert L == expected_L, f"L mismatch: got {L}, expected {expected_L}"

    # --- (X, F, T) -> (X, T, F) -> (X*T, F)
    inputs_np = to_numpy(inputs_horizon).transpose(0, 2, 1).reshape(-1, F)

    # --- (X, L, T) -> (X, T, L) -> (X*T, L)
    labels_np = to_numpy(labels).transpose(0, 2, 1).reshape(-1, L)

    # --- colonnes ---
    label_cols = list(ids_columns) + list(targets_columns) + [target_name]
    all_cols = list(map(str, features_name)) + list(map(str, label_cols))

    # --- DataFrame final (large) ---
    data = np.concatenate([inputs_np, labels_np], axis=1)
    df = pd.DataFrame(data, columns=all_cols)

    # Toujours des noms de colonnes str
    df.columns = df.columns.astype(str)
    return df

class WrapperModel(torch.nn.Module):
    def __init__(self, original_model, F, T, edges, horizon=0):
        super().__init__()
        self.model = original_model
        self.F = F
        self.T = T
        self.edges = edges

        self.horizon = horizon

    def forward(self, x_flat):
        # reshape x_flat (B, F*T) vers (B, F, T)
        x_orig = x_flat.reshape(-1, self.F, self.T)
        return self.model(x_orig, self.edges)

class Training():
    def __init__(self, model_name, nbfeatures, batch_size, lr, delta_lr, patience_cnt_lr, target_name, task_type,
                 features_name, ks, out_channels, dir_log,
                 loss='mse', name='Training', device='cpu',
                 under_sampling='full', over_sampling='full', n_run=1,
                 horizon=0, post_process=None):
        
        self.model_name = model_name
        self.name = name
        self.loss = loss
        self.device = device
        self.model = None
        self.optimizer = None
        self.batch_size = batch_size
        self.target_name = target_name
        self.features_name = [str(fet) for fet in features_name]
        self.ks = int(ks)
        self.lr = lr
        self.delta_lr = delta_lr
        self.patience_cnt_lr = patience_cnt_lr
        self.out_channels = out_channels
        self.dir_log = dir_log
        self.task_type = task_type
        self.model_params = None
        self.under_sampling = under_sampling
        self.over_sampling = over_sampling
        self.find_log = False
        self.nbfeatures = nbfeatures
        self.student_train = False
        self.use_temporal_as_edges = None
        self.n_run = n_run
        self.metrics = {}
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.constrastive = False
        self.prox_term = False
        self.prox_value = 0.05
        self.use_prototypes = False
        self.prototype_weight = 1.0
        self.prototypes = None
        self.ALATraining = False
        self.area_parameters = None
        # Distillation tracking (best/worst losses per epoch)
        self.distill_best_log = []   # list of dicts: {epoch, graph_id, loss}
        self.distill_worst_log = []  # list of dicts: {epoch, graph_id, loss}
        self.criterion_params = []
        self._current_epoch = None
        self.seed = None
        self.horizon = horizon
        self.seed = None
        self.apply_discretization = post_process is not None
        self.post_process = post_process

        if 'Past_risk' in self.features_name:
            self.id_past_risk = features_name.index('Past_risk')
        else:
            self.id_past_risk = None

        if 'Past_burnedarea' in self.features_name:
            self.id_past_ba = features_name.index('Past_burnedarea')
        else:
            self.id_past_ba = None

        self.prev_idx = []

        if self.task_type == "classification":
            # Pour classification : les colonnes one-hot sont du type f"{colunm}_prev_<classe>"
            new_features = [f"{self.target_name}_prev_{i}" for i in range(self.out_channels)]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]
            print(self.prev_idx)

        elif self.task_type == "binary":
            # Pour binaire : on a colunm_prev_bin, colunm_prev_bin_0 et colunm_prev_bin_1
            new_features = [f"{self.target_name}_prev_bin"] + [f"{self.target_name}_prev_bin_{i}" for i in range(self.out_channels)]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]

        elif self.task_type == "regression" and self.target_name in ["nbsinister", "burnedarea"]:
            # Pour régression : une seule feature ajoutée
            new_features = [f"{self.target_name}_prev"]
            self.prev_idx = [self.features_name.index(f) for f in new_features if f in self.features_name]
        
        if len(self.prev_idx) == 0:
            self.prev_idx = None
        
        # History for loss components decomposition
        self.loss_components_history = {
            'loss_total': [],
            'loss_trans': [],     # Task loss (raw)
            'entropy_pi': [],     # Entropy (raw)
            'mu0_term': [],       # Mu0 term (raw)
            'dirichlet_reg': [],  # Dirichlet reg (raw)
            'ce_loss': [],        # CE loss (raw)
            'epoch': [],
            
            # Detailed scaling stats
            'scale_min': [],
            'scale_mean': [],
            'scale_max': [],
            'diff_raw_mean': [],
            'diff_scaled_mean': [],
            'margin_mean': []
        }

    def remove_graph(self):
        del self.graph
        
    def clean(self):
        try:
            del self.df_train
        except:
            pass
        try:
            del self.df_val
        except:
            pass
        try:
            del self.df_test
        except:
            pass
        try:
            del self.train_loader
        except:
            pass
        try:
            del self.val_loader
        except:
            pass
        try:
            del self.test_loader
        except:
            pass
    
    def compute_weights_and_target(self, labels, band, ids_columns, is_grap_or_node, graphs, H):
        weight_idx = ids_columns.index('weight')
        target_is_binary = self.task_type == 'binary'

        if len(labels.shape) == 3:
            weights = labels[:, weight_idx, H]
            target = (labels[:, band, H] > 0).long() if target_is_binary else labels[:, band, H]

        elif len(labels.shape) == 5:
            weights = labels[:, :, :, weight_idx, H]
            target = (labels[:, :, :, band, H] > 0).long() if target_is_binary else labels[:, :, :, band, H]

        elif len(labels.shape) == 4:
            weights = labels[:, :, :, weight_idx,]
            target = (labels[:, :, :, band] > 0).long() if target_is_binary else labels[:, :, :, band]

        else:
            weights = labels[:, weight_idx]
            target = (labels[:, band] > 0).long() if target_is_binary else labels[:, band]
        
        if is_grap_or_node:
            unique_elements = torch.unique(graphs, return_inverse=False, return_counts=False, sorted=True)
            first_indices = torch.tensor([torch.nonzero(graphs == u, as_tuple=True)[0][0] for u in unique_elements])
            weights = weights[first_indices]
            target = target[first_indices]

        return target, weights
    
    def compute_inputs(self, inputs, H, time_steps):
        if H + 1 == 0:
            if len(inputs.shape) == 3:
                inputs_horizon = inputs[:, :, -(self.ks + 1):]

            elif len(inputs.shape) == 5:
                inputs_horizon = inputs[:, :, :, :, -(self.ks + 1):]

            elif len(inputs.shape) == 4:
                inputs_horizon = inputs
            else:
                inputs_horizon = inputs
        
        else:
            if len(inputs.shape) == 3:
                inputs_horizon = inputs[:, :, H - self.ks:H + 1]

            elif len(inputs.shape) == 5:
                inputs_horizon = inputs[:, :, :, :, H - self.ks:H + 1]

            elif len(inputs.shape) == 4:
                inputs_horizon = inputs
            else:
                inputs_horizon = inputs

        if inputs_horizon.ndim % 2 == 0:
                inputs_horizon = inputs_horizon[:, :, None]
            
        return inputs_horizon
    
    def compute_single_loss(self, out, tar, wei, clusters_ids=None, tolong=False, areas=None, criterion=None):
        if self.task_type == 'regression':
            tar = tar.view(out.shape[0])
            wei = wei.view(out.shape[0])

            tar = torch.masked_select(tar, wei.gt(0))
            out = out[wei.gt(0)]
            wei = torch.masked_select(wei, wei.gt(0))
        else:
            wei = wei.long()

            tar = tar[wei.gt(0)]
            out = out[wei.gt(0)]

            if clusters_ids is not None:
                clusters_ids = clusters_ids[wei.gt(0)]

            wei = torch.masked_select(wei, wei.gt(0))

            if tolong:
                tar = tar.long()
                
        additionnal_params = {}

        if clusters_ids is not None:
            additionnal_params['clusters_ids'] = clusters_ids
        
        if areas is not None:
            additionnal_params['areas'] = areas
        
        try:
            additionnal_params['sample_weight'] = wei
        
            return criterion(out, tar, **additionnal_params)
        except Exception as e:
            print(e)
            return criterion(out, tar)

    def calculate_loss(self, criterion, output, target, weights, label, tolong=True):

        if 'clusters_ids' in required_params(criterion.forward):
            if hasattr(criterion, 'id') and criterion.id is not None:
                if criterion.id == -1 :
                    clusters_ids = torch.ones(target.shape[0], device=target.device)
                else:
                    clusters_ids = label[:, criterion.id, -1]
                    self.cluster_id_index = criterion.id
            else:
                # If id is not defined, use the whole batch as a single cluster
                clusters_ids = torch.zeros(target.shape[0], device=target.device)
        else:
            clusters_ids = None

        if 'areas' in required_params(criterion.forward):
            areas = label[:, area_index, -1]
        
        else:
            areas = None
            
        base_loss = self.compute_single_loss(output, target, weights, clusters_ids, tolong, areas, criterion)
        
        if 'area' in self.loss and False: # Calculate area loss (specify loss-area)
            area_mask = label[:, graph_id_index, -1]
            unique_ids = torch.unique(area_mask)
            values = []
            active_idx = []
            for aid in unique_ids:
                m = area_mask == aid
                if m.sum() == 0:
                    continue
                active_idx.append(int(aid))
                l = self.compute_single_loss(output[m], target[m], weights[m], None, tolong, criterion)
                #print(f'{aid}, {l}')
                values.append(l)
            if len(values) > 0:
                active_idx = torch.as_tensor(active_idx, dtype=torch.long)
                mask = torch.zeros_like(self.area_parameters, dtype=self.area_parameters.dtype)
                mask.index_fill_(0, active_idx, 1.0)
                vals_active = torch.as_tensor(values, device=self.area_parameters.device,
                              dtype=self.area_parameters.dtype)
                vals_full = torch.zeros_like(self.area_parameters)
                vals_full.index_copy_(0, active_idx, vals_active)
                ap_active = self.area_parameters * mask
                eps = 1e-8
                ap_active = torch.log(torch.nn.functional.softplus(ap_active))
                area_loss = (vals_full * ap_active).sum() / ap_active.sum().clamp_min(eps)
            else:
                area_loss = torch.as_tensor(0.0, device=output.device)

            if 'area-global' in self.loss and False:  # Calculate area * global (classic) loss  (specify loss-area-global)
                loss = area_loss + base_loss
                logger.info(f'area_loss : {area_loss}, {base_loss}, {loss}')
            else:
                loss = area_loss
        else:
            loss = base_loss

        return loss
    
    def calculate_contrastive_moon_loss(self, z, zprev, zglob, temperature=0.5):
        """
        Computes the MOON contrastive loss.

        Args:
            z       : Tensor of shape [batch_size, dim] from current local model.
            zprev   : Tensor of shape [batch_size, dim] from previous local model.
            zglob   : Tensor of shape [batch_size, dim] from global model.
            temperature (float): Temperature parameter τ for scaling similarities.

        Returns:
            loss (Tensor): Scalar contrastive loss for the batch.
        """
        # Normalize representations to compute cosine similarity
        z = F.normalize(z, dim=1)
        zprev = F.normalize(zprev, dim=1)
        zglob = F.normalize(zglob, dim=1)

        # Cosine similarities
        sim_pos = torch.sum(z * zglob, dim=1) / temperature  # similarity with global (positive)
        sim_neg = torch.sum(z * zprev, dim=1) / temperature   # similarity with previous (negative)

        # Contrastive loss per sample
        logits = torch.stack([sim_pos, sim_neg], dim=1)  # shape: [batch_size, 2]
        labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)  # positive is at index 0

        # Use cross-entropy to compute: -log( exp(sim_pos) / (exp(sim_pos) + exp(sim_neg)) )
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def pick_params_by_name(self, model, names):
        names, params = [], []
        for n, p in model.named_parameters():
            for pick_parm in names:
                if pick_parm in n:
                    names.append(n); params.append(p)
        return names, params
    
    def calculate_prox_term(self, names):
        prox = 0.0

        model_params = self.pick_params_by_name(self.model, names)
        global_model_params = self.pick_params_by_name(self.global_model, names)

        for (name, p), (_, pg) in zip(model_params, global_model_params):
            if not p.requires_grad:
                continue

            if not name in names:
                continue

            prox = prox + (p - pg.detach()).pow(2).sum()

        return 0.5 * self.prox_value * prox

    def calculate_prototype_loss(self, hidden, target, prototypes):
        """Compute prototype alignment loss."""
        loss = 0.0
        classes = torch.unique(target)
        for cls in classes:
            cls_idx = int(cls.item())
            if prototypes is None or cls_idx not in prototypes:
                continue
            proto = prototypes[cls_idx].to(hidden.device)
            mask = target == cls
            if mask.sum() == 0:
                continue
            diff = hidden[mask] - proto
            loss += torch.mean(torch.norm(diff, dim=1))
        return loss

    def calculate_prototype_alignment_loss(self, hidden, target, prototypes):
        """
        Compute prototype alignment loss:
        Sum over classes of L2 distance squared between local and global prototypes.

        Arguments:
            hidden (Tensor): Embeddings of shape [batch_size, embedding_dim].
            target (Tensor): Class labels of shape [batch_size].
            prototypes (dict): {class_id: global_prototype_tensor}

        Returns:
            loss (Tensor): Scalar tensor representing the total alignment loss.
        """
        loss = 0.0
        classes = torch.unique(target)
        for cls in classes:
            cls_idx = int(cls.item())
            if prototypes is None or cls_idx not in prototypes:
                continue
            # Global prototype
            proto_global = prototypes[cls_idx].to(hidden.device)
            
            # Local prototype for class cls
            mask = target == cls
            if mask.sum() == 0:
                continue
            proto_local = hidden[mask].mean(dim=0)
            
            # L2 distance squared between local and global prototype
            diff = proto_local - proto_global
            loss += torch.sum(diff ** 2)
            
        return loss
    
    def model_distillation_loss(self, loss, inputs_horizon, labels, target, logits, hiddens):
        """
        Compute knowledge distillation loss combining task loss and KL divergence.
        
        Formula: loss = alpha * task_loss + (1 - alpha) * T² * KL(student || teacher)
        
        Args:
            loss: Original task loss (cross-entropy)
            inputs_horizon: Input features
            labels: Ground truth labels
            logits: Student model logits
            hiddens: Student model hidden states
            
        Returns:
            Combined loss following Hinton et al. convention
        """
        criterion_teacher = self.get_loss('kldivloss', {})
        #df_test = build_dataframe(inputs_horizon, labels, self.features_name, ids_columns, targets_columns, self.target_name)
        # 

        teacher_logits = []
        teacher_feats = []
        weights = []

        # Teachers
        models_to_mean, _, _ = self.teacher.get_weights(self.top_model, return_self_model_idx=True)
        leni = len(models_to_mean)
        with torch.no_grad():
            for idx in models_to_mean:
                t_wrapper = self.teacher.best_estimator_[idx]
                assert t_wrapper.features_name == self.features_name
                    
                if t_wrapper.target_name == self.target_name:
                    continue

                t_model = t_wrapper.model
                t_model.eval()
                try:
                    _, t_log, t_feat = t_model(inputs_horizon, self.graph)
                except:
                    _, t_log, t_feat = t_model(inputs_horizon)
                teacher_logits.append(t_log)
                teacher_feats.append(t_feat)
                weights.append(self.teacher.weights_for_model[idx])

        T = self.temperature_value

        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
        weights = torch.as_tensor(weights, device=logits.device, dtype=torch.float32)
        
        device = logits.device
            
        if self.distillation_training_mode == 'normal':
            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  teacher_logits, weights, T=self.temperature_value)
            #print(f'kl_div_loss {kl_div_loss}, loss {loss}')
            loss =  (1 - self.alpha_value) * loss + (self.alpha_value) * kl_div_loss
            return loss
        
        elif self.distillation_training_mode == 'AdaptativeMLP':
            # AdaptativeMLP specific loss logic

            # Access intermediates stored in model
            # Handle DataParallel if necessary
            student_model = self.model
            
            # Adapter
            student_rep = hiddens[-1]

            weights = self.adapter(student_rep)
            
            # LKD
            T = self.temperature_value
            loss_kd, fused_soft, t_soft_all = multi_teacher_kd_loss(
                logits,
                teacher_logits,
                weights,
                T=T
            )

            # LHT
            n_group = student_model.n_group
            chunk_size = (len(teacher_feats) + n_group - 1) // n_group
            student_feats_expanded = []
            for i in range(len(teacher_feats)):
                group_idx = i // chunk_size
                if group_idx > self.model.n_group:
                    continue
                if group_idx >= n_group: group_idx = n_group - 1
                student_feats_expanded.append(hiddens[group_idx])
                
            loss_lht = lht_loss(teacher_feats, student_feats_expanded, self.fitnets)
            
            # LAngle
            student_soft = F.softmax(logits / T, dim=-1)
            loss_angle = angle_triplet_loss(fused_soft, student_soft)
            
            loss = loss + self.alpha_value * loss_kd + self.beta_value * loss_lht + self.gamma_value * loss_angle
            return loss

        elif self.distillation_training_mode == 'MATTKD':
            # Fusion teacher with attention layer
            t_l = torch.stack(teacher_feats, dim=1)
            super_teacher_logits = self.relation_att(t_l)
            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  [super_teacher_logits], [1], T=self.temperature_value)

            # LHT
            n_group = self.model.n_group
            chunk_size = (len(teacher_feats) + n_group - 1) // n_group
            student_feats_expanded = []
            for i in range(len(teacher_feats)):
                group_idx = i // chunk_size
                if group_idx > self.model.n_group:
                    continue
                if group_idx >= n_group: group_idx = n_group - 1
                student_feats_expanded.append(hiddens[group_idx])
                
            loss_lht = lht_loss(teacher_feats, student_feats_expanded, self.fitnets)

            loss = loss + self.alpha_value * kl_div_loss + self.beta_value * loss_lht
            return loss

        elif self.distillation_training_mode == 'RelationMLP':
            # RelationMLP mode: Combined embedding and logits distillation
            # L = L_CE(y, student(x)) + α × ||f_S(x) - f_E(x)||² + β × KL(p_S(x) || p̄_T(x))

            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  teacher_logits, weights, T=self.temperature_value)
            kl_div_loss
            # 3) Embedding distillation: L2 distance between student logits and ensemble embedding
            # Get ensemble embedding from RelationMLP (concatenates all teacher logits)
            t_l = torch.stack(teacher_feats, dim=1)
            ensemble_embedding = self.relation_mlp(t_l)  # [B, num_classes]
            
            # Student embedding: use logits directly (before softmax)
            if isinstance(hiddens, list):
                student_embedding = hiddens[-1]
            else:
                student_embedding = hiddens  # [B, num_classes]
            
            # Compute L2 loss (MSE)
            embedding_loss = F.mse_loss(student_embedding, ensemble_embedding)
            
            # 4) Combine all losses: L_CE + α × embedding_loss + β × kl_div_loss
            #print(f'kl_div_loss {kl_div_loss}, loss {loss}, embedding_loss {embedding_loss}')
            loss = loss + self.beta_value * embedding_loss + self.alpha_value * kl_div_loss
            return loss
        
        elif self.distillation_training_mode == 'RelationATT':
            # RelationATT mode: Combined embedding and logits distillation using Attention
            # L = L_CE(y, student(x)) + α × ||f_S(x) - f_E(x)||² + β × KL(p_S(x) || p̄_T(x))
            
            kl_div_loss, _, _ = multi_teacher_kd_loss_global_weights(logits,  teacher_logits, weights, T=self.temperature_value)
            
            # 2) Embedding distillation: L2 distance between student logits and ensemble embedding
            # Get ensemble embedding from RelationAttention
            t_l = torch.stack(teacher_feats, dim=1)
            ensemble_embedding = self.relation_att(t_l)  # [B, num_classes]

            # Student embedding: use logits directly (before softmax)
            if isinstance(hiddens, list):
                student_embedding = hiddens[-1]
            else:
                student_embedding = hiddens  # [B, num_classes]
            
            # Compute L2 loss (MSE)
            embedding_loss = F.mse_loss(student_embedding, ensemble_embedding)
            
            # 3) Combine all losses: L_CE + α × embedding_loss + β × kl_div_loss
            loss = loss + self.beta_value * embedding_loss + self.alpha_value * kl_div_loss
            return loss
        
        elif self.distillation_training_mode == 'Confidence':
            # Confidence distillation
            
            # Collect teacher classifiers
            teacher_classifiers = []
            for idx in models_to_mean:
                t_wrapper = self.teacher.best_estimator_[idx]
                if t_wrapper.target_name == self.target_name: continue
                
                # Assume teacher model has output_layer (Linear)
                # If not available, we might fail. 
                # For StudentMLP teachers it is available.
                if hasattr(t_wrapper.model, 'output_layer'):
                    teacher_classifiers.append(t_wrapper.model.output_layer)
                elif hasattr(t_wrapper.model, 'fc'): # ResNet style
                     teacher_classifiers.append(t_wrapper.model.fc)
                elif hasattr(t_wrapper.model, 'classifier'): 
                     teacher_classifiers.append(t_wrapper.model.classifier)
                else:
                    # Fallback or error? 
                    # For now let's assume it exists or use a dummy identity if we can't find it
                    # But we need it for w_inter calculation.
                    raise ValueError(f"Teacher model {t_wrapper.name} does not have a known classifier layer (output_layer, fc, classifier)")

            # Student feature (last hidden)
            if isinstance(hiddens, list):
                student_feat = hiddens[-1]
            else:
                student_feat = hiddens
                
            # Call confidence loss
            loss = loss + confidence_distillation_loss(
                student_logits=logits,
                student_feat=student_feat,
                teacher_logits_list=teacher_logits,
                teacher_feat_list=teacher_feats,
                labels=target,
                fitnets=self.fitnets,
                teacher_classifiers=teacher_classifiers,
                alpha=self.alpha_value,
                beta=self.beta_value,
                T=self.temperature_value
            )
            
            return loss
        
    def launch_batch(self, data, criterion, batch_type, do_update):
        inputs, labels, _ = data
        graphs = None
        
        if inputs.shape[0] == 1:
            return 0, 0

        band = -1
        
        hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
        output_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)

        for H in range(self.horizon + 1):

            if hasattr(self.model, 'is_graph_or_node'):
                is_graph_or_node = self.model.is_graph_or_node
            else:
                is_graph_or_node = False

            target, weights = self.compute_weights_and_target(labels, band, ids_columns, is_graph_or_node, graphs,  -1 - (self.horizon - H))

            if self.loss not in ['kldivloss']: # works on probability
                target = target.long()

            inputs_horizon = self.compute_inputs(inputs,  -1 - (self.horizon - H), "current" if H == 0 else "futur")

            if H == 0:
                z_prev = None
            else:
                if self.ks > 0:
                    # on prend les ks derniers états cachés déjà vus
                    history = hidden_past[-(self.ks + 1):]
                    # empilement (B, D, L) avec L = len(history)
                    z_prev = torch.stack(history, dim=2)  # (B, D, L)

                    # padding à gauche si L < ks
                    L = z_prev.size(2)
                    if L < (self.ks + 1):
                        B, D = z_prev.size(0), z_prev.size(1)
                        pad = torch.zeros(
                            (B, D, self.ks + 1 - L),
                            device=z_prev.device,
                            dtype=z_prev .dtype
                        )
                        z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                else:
                    z_prev = hidden_past[-1]
            if H == 0:
                output, logits, hidden = self.model(inputs_horizon, z_prev=None)
                if batch_type == 'train' and do_update:
                    if has_method(criterion, 'update_after_batch'):
                        criterion.update_after_batch(logits, target)
            else:
                if self.id_past_risk is not None:
                    inputs_horizon[:, self.id_past_risk, -H:] = 0
                if self.id_past_ba is not None:
                    inputs_horizon[:, self.id_past_ba, -H:] = 0
                if self.prev_idx is not None:
                    inputs_horizon[:, self.prev_idx, -H:] = torch.stack(output_past, dim=2)

                output, logits, hidden = self.model(inputs_horizon, z_prev=z_prev)
            
            hidden_past.append(hidden)
            output_past.append(output)
            
            loss_res = self.calculate_loss(criterion, logits, target, weights, labels)

            if isinstance(loss_res, dict):
                loss = loss_res['total_loss']
            else:
                loss = loss_res
            
            if self.student_train: # distallation traning
                loss = self.model_distillation_loss(loss, inputs_horizon, labels, target, logits, hidden)
                
            if self.constrastive: # MOON federated training
                _, _, zprev = self.prev_model(inputs)
                _, _, zglob = self.global_model(inputs)
                loss_constrastive = self.calculate_contrastive_moon_loss(hidden, zprev, zglob, self.moon_temperature_value)
                loss = loss + self.smooth_value * loss_constrastive
            
            if self.prox_term:
                prox_term = self.calculate_prox_term(self.fed_prox_names)
                loss = loss + prox_term

            if self.use_prototypes and self.prototypes is not None:
                if not self.model.return_hidden:
                    raise ValueError('Model must return hidden states for prototype training')
                
                proto_loss = self.calculate_prototype_alignment_loss(hidden, target, self.prototypes)
                loss = loss + self.prototype_weight * proto_loss

            if 'distillation' in self.loss:
                distill_loss, region_losses, best, worst = self.loss_distill(
                    output, target, weights, labels, hidden, 0.10, 0.10, graph_id_index,
                    lambda_kd=1, use_cosine=True, tolong=False, cluster_ids=None, criterion=criterion
                )
                loss = loss + distill_loss

                # Update per-epoch best/worst trackers using region_losses
                try:
                    # region_losses is a dict {rid: tensor_loss}
                    if isinstance(region_losses, dict) and len(region_losses) > 0:
                        # Best: smallest loss among reported best IDs
                        if len(best) > 0:
                            best_pair = min(((rid, region_losses[rid].item()) for rid in best if rid in region_losses),
                                            key=lambda kv: kv[1], default=None)
                            if best_pair is not None:
                                rid_b, loss_b = best_pair
                                if hasattr(self, '_epoch_distill_best'):
                                    if loss_b < self._epoch_distill_best['loss']:
                                        self._epoch_distill_best['loss'] = float(loss_b)
                                        self._epoch_distill_best['graph_id'] = int(rid_b)
                        # Worst: largest loss among reported worst IDs
                        if len(worst) > 0:
                            worst_pair = max(((rid, region_losses[rid].item()) for rid in worst if rid in region_losses),
                                            key=lambda kv: kv[1], default=None)
                            if worst_pair is not None:
                                rid_w, loss_w = worst_pair
                                if hasattr(self, '_epoch_distill_worst'):
                                    if loss_w > self._epoch_distill_worst['loss']:
                                        self._epoch_distill_worst['loss'] = float(loss_w)
                                        self._epoch_distill_worst['graph_id'] = int(rid_w)
                except Exception as _e:
                    # Never break training because of logging
                    pass

            if self.model_name in ['BayesianMLP', 'BayesianCNN', 'BayesianRNN']:
                loss += self.model.kl_loss()
            
            if 'total_loss' not in locals():
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_res

    def launch_train_loader(self, loader, criterion, optimizer, do_update):

        self.model.train()
        
        if has_method(criterion, 'get_learnable_parameters'):
            criterion.train()

        # Initialize per-epoch aggregation for distillation best/worst
        if 'distillation' in self.loss:
            self._epoch_distill_best = {'loss': float('inf'), 'graph_id': None}
            self._epoch_distill_worst = {'loss': float('-inf'), 'graph_id': None}
        
        res_loss = 0.0
        res_loss_dict = {'l': 0.0}

        for i, data in enumerate(loader, 0):
            
            loss, loss_res = self.launch_batch(data, criterion, 'train', do_update)

            if isinstance(loss, int) or isinstance(loss, float):
                print(f'loss is does not required grad {loss}')
                continue
            
            if optimizer is not None:
                optimizer.zero_grad()
                try:
                    loss.backward()
                except:
                    continue
            
            if 'res_loss' in locals():
                res_loss += loss.item()
                if isinstance(loss_res, dict):
                    for key in loss_res:
                        if key in res_loss_dict:
                            res_loss_dict[key] += loss_res[key]
                        else:
                            res_loss_dict[key] = loss_res[key]
                else:
                    res_loss_dict['l'] += loss_res
            else:
                res_loss = loss.item()
                if isinstance(loss_res, dict):
                    res_loss_dict = {k: v for k, v in loss_res.items()}
                else:
                    res_loss_dict = {'l': loss_res}

            if self.ALATraining:
                # Mises à jour SANS autograd
                with torch.no_grad():
                    # 1) update des weights
                    for p_t, p_prev, p_g, w in zip(
                            self.params_p, self.params_tp, self.params_gp, self.weights):
                        upd = w - self.eta * ((p_g - p_prev) * p_t.grad)
                        w.copy_(torch.clamp(upd, 0.0, 1.0))

                    #if not self.ala_weight_only:
                    #    # 2) calcul des params interpolés
                    #    for p_t, p_prev, p_g, w in zip(
                    #            self.params_p, self.params_tp, self.params_gp, self.weights):
                    #        p_t.copy(p_t - self.eta * (p_g - p_prev) * (p_t.grad))

            if optimizer is not None and not getattr(self, 'ala_weight_only', False):
                optimizer.step()

                #self.update_weight()
        # After finishing the epoch, persist the best/worst entries for this epoch
        if 'distillation' in self.loss:
            if getattr(self, '_epoch_distill_best', None) is not None and self._epoch_distill_best['graph_id'] is not None:
                self.distill_best_log.append({
                    'epoch': self._current_epoch if self._current_epoch is not None else -1,
                    'graph_id': int(self._epoch_distill_best['graph_id']),
                    'loss': float(self._epoch_distill_best['loss'])
                })
            if getattr(self, '_epoch_distill_worst', None) is not None and self._epoch_distill_worst['graph_id'] is not None:
                self.distill_worst_log.append({
                    'epoch': self._current_epoch if self._current_epoch is not None else -1,
                    'graph_id': int(self._epoch_distill_worst['graph_id']),
                    'loss': float(self._epoch_distill_worst['loss'])
                })

        if has_method(criterion, 'get_attribute'):
            params = criterion.get_attribute()
            dict_params = {'epoch': self._current_epoch}
            for par in params:
                name = par[0]
                value = par[1]
                dict_params[name] = deepcopy(value.detach().cpu().numpy())
            
            self.criterion_params.append(dict_params)

        if len(loader) > 0:
            res_loss /= len(loader)
            for k in res_loss_dict:
                if isinstance(res_loss_dict[k], (int, float, torch.Tensor)):
                    res_loss_dict[k] /= len(loader)

        return res_loss, res_loss_dict

    def launch_val_test_loader(self, loader, criterion, teacher=None):

        self.model.eval()

        if has_method(criterion, 'get_learnable_parameters'):
            criterion.eval()

        total_loss = 0.0

        total_loss_dict = {}

        with torch.no_grad():

            for i, data in enumerate(loader, 0):
                
                loss, loss_res = self.launch_batch(data, criterion, 'val', do_update=False)

                if loss is not None:
                    if torch.is_tensor(loss):
                        total_loss += loss.item()
                    else:
                        total_loss += loss
                else:
                    # Should not happen ideally, but safety first
                    pass

                if isinstance(loss_res, dict):
                    for key in loss_res:
                        if key not in total_loss_dict:
                            total_loss_dict[key] = 0
                        total_loss_dict[key] += loss_res[key].item()
                else:
                    if 'l' not in total_loss_dict:
                        total_loss_dict['l'] = loss_res
                    else:
                        total_loss_dict['l'] += loss_res
            
        if 'learnable-area' in self.loss:
            if hasattr(self, 'area_parameters_log'):
                self.area_parameters_log.append(self.area_parameters)
            else:
                self.area_parameters_log = []
                self.area_parameters_log.append(self.area_parameters)

        if len(loader) > 0:
            total_loss /= len(loader)
            for k in total_loss_dict:
                total_loss_dict[k] /= len(loader)

        return total_loss, total_loss_dict
    
    def make_model(self, graph, custom_model_params):
        model, params = make_model(self.model_name, len(self.features_name), len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params, horizon=self.horizon)

        if self.model_params is None:
            self.model_params = params

        return model, params

    def func_epoch(self, train_loader, val_loader, optimizer, criterion, do_update):

        train_loss, train_loss_dict = self.launch_train_loader(train_loader, criterion, optimizer, do_update)

        if val_loader is not None:
            val_loss, val_loss_dict = self.launch_val_test_loader(val_loader, criterion)
        else:
            val_loss = train_loss.item()
            val_loss_dict = train_loss_dict

        return val_loss, train_loss, val_loss_dict, train_loss_dict

    def get_class_freq(self, df_train):
        uclass = np.sort(df_train[self.target_name].unique())
        res = np.zeros_like(uclass)
        for i, cl in enumerate(uclass):
            res[i] = len(df_train[(df_train[self.target_name] == cl) & (df_train['weight'] > 0)])
        
        return self.compute_global_alpha(res)
    
    def compute_global_alpha(self, global_hist):
        freq = global_hist / global_hist.sum()
        alpha = 1 / np.sqrt(freq)
        alpha = alpha / alpha.sum()
        return alpha

    def train(self, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None, new_model=True, min_epochs=1):
        """
        Train neural network model
        """
        
        self.score_per_epochs = {}
        if MLFLOW:
            existing_run = get_existing_run(f'{self.model_name}_')
            if existing_run:
                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
            else:
                mlflow.start_run(run_name=f'{self.model_name}_', nested=True)

        assert self.train_loader is not None and self.val_loader is not None

        check_and_create_path(self.dir_log)

        loss_params = {}
        if 'fl' in self.loss: # Use focal loss
            if hasattr(self, "class_freq"):
                loss_params = {'alpha' : self.class_freq}
            else:
                loss_params = {'alpha' : self.get_class_freq(self.df_train)}

        criterion = self.get_loss(self.loss, loss_params)

        if has_method(criterion, '_preprocess'):
            if 'id{departement}' in self.loss:
                criterion._preprocess(self.df_train[self.target_name].values, self.df_train['departement'].values, self.df_train['cluster-encoder'].values)
            elif 'id{node}' in self.loss:
                criterion._preprocess(self.df_train[self.target_name].values, self.df_train['graph_id'].values, self.df_train['cluster-encoder'].values)

        static_idx, temporal_idx = get_static_temporal_idx(self.features_name)
        
        new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx}

        if self.model_name == 'TFN':
            new_params = {'static_idx': static_idx, 'temporal_idx' : temporal_idx, 'd_static' : len(static_idx)}
                
        if custom_model_params is None:
            custom_model_params = new_params
        else:
            custom_model_params.update(new_params)

        if new_model or self.model is None:
            self.model, _ = self.make_model(graph, custom_model_params)
        else:
            assert self.model is not None
        
        optimizer = self.get_optimizer(criterion)

        BEST_VAL_LOSS = math.inf
        BEST_MODEL_PARAMS = None
        best_epoch = 0
        patience_cnt = 0
        current_patience_lr = 0

        val_loss_list = []
        train_loss_list = []
        epochs_list = []
        val_loss_dict_list = []
        train_loss_dict_list = []
        
        if (self.dir_log / 'best.pt').is_file():
            print(f"WARNING: Checkpoint found at {self.dir_log / 'best.pt'} but SKIPPING load due to hardcoded False.")
        
        #if (self.dir_log / 'best.pt').is_file():
        if False:
            self._load_model_from_path(self.dir_log / 'best.pt', self.model)
        else:
            for epoch in tqdm(range(epochs), disable=not verbose):
                # Expose current epoch to subroutines for logging
                self._current_epoch = epoch
                val_loss, train_loss, val_loss_dict, train_loss_dict = self.func_epoch(train_loader=self.train_loader, val_loader=self.val_loader,
                                                    optimizer=optimizer, criterion=criterion, do_update=epoch < min_epochs)

                val_loss_list.append(round(val_loss, 3))
                train_loss_list.append(round(train_loss, 3))
                val_loss_dict_list.append(val_loss_dict)
                train_loss_dict_list.append(train_loss_dict)
                epochs_list.append(epoch)
                if val_loss < BEST_VAL_LOSS and epoch > min_epochs:
                    BEST_VAL_LOSS = val_loss
                    BEST_MODEL_PARAMS = self.model.state_dict()
                    patience_cnt = 0
                    current_patience_lr = 0 # Val loss improved, reset retries
                    best_epoch = epoch
                else:
                    patience_cnt += 1
                    
                    # Logic for LR Decay: If PATIENCE_CNT is reached
                    if patience_cnt >= PATIENCE_CNT and epoch >= min_epochs:
                        # Check if we can reduce LR (have we used all retries?)
                        # PATIENCE_CNT_LR is the number of allowed reductions/retries
                        if current_patience_lr >= self.patience_cnt_lr:
                            logger.info(f'Loss has not increased for {patience_cnt} epochs AND max LR reductions ({self.patience_cnt_lr}) reached.')
                            logger.info(f'Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                            save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                            save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                            plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                            if MLFLOW:
                                mlflow.end_run()
                            break
                        else:
                            # Reduce LR and reset patience_cnt
                            if self.delta_lr > 0:
                                current_patience_lr += 1
                                logger.info(f"Patience {PATIENCE_CNT} reached (Retry {current_patience_lr}/{self.patience_cnt_lr}). Decay LR by factor {self.delta_lr}.")
                                
                                current_lr = optimizer.param_groups[0]['lr']
                                new_lr = current_lr * (1 - self.delta_lr)
                                if new_lr <= 1e-9:
                                    new_lr = 1e-9
                                    logger.warn("Learning rate reached floor (1e-9).")
                                
                                logger.info(f"Reducing LR from {current_lr:.6f} to {new_lr:.6f}")
                                
                                # Define new optimizer with new LR (resets state/momentum as requested)
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = new_lr
                                
                                # Reset patience_cnt to give model time to improve with new LR
                                patience_cnt = 0
                            else:
                                # No delta_lr defined, stop normal
                                logger.info(f'Loss has not increased for {patience_cnt} epochs. No delta_lr defined.')
                                save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
                                save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
                                plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
                                if MLFLOW:
                                    mlflow.end_run()
                                break
                if MLFLOW:
                    mlflow.log_metric('loss', val_loss, step=epoch)
                if epoch % CHECKPOINT == 0 and verbose:
                    curr_lr = optimizer.param_groups[0]['lr']
                    logger.info(f'Epoch {epoch}: Val loss {val_loss:.4f}, Train loss {train_loss:.4f}, Best val loss {BEST_VAL_LOSS:.4f}')
                    logger.info(f'    LR: {curr_lr:.6f} | Patience: {patience_cnt}/{PATIENCE_CNT} | Retry: {current_patience_lr}/{self.patience_cnt_lr}')
                    save_object_torch(self.model.state_dict(), str(epoch)+'.pt', self.dir_log)

            logger.info(f'Last val loss {val_loss}')
            save_object_torch(self.model.state_dict(), 'last.pt', self.dir_log)
            save_object_torch(BEST_MODEL_PARAMS, 'best.pt', self.dir_log)
            plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, self.dir_log)
            try:
                plot_train_val_loss(epochs_list, train_loss_dict_list, val_loss_dict_list, self.dir_log)
            except:
                pass

        self.train_loss_dict_list = train_loss_dict_list
        self.val_loss_dict_list = val_loss_dict_list
        self.train_loss_list = train_loss_list
        self.val_loss_list = val_loss_list
        self.epochs_list = epochs_list

        self.best_epoch = best_epoch
        if best_epoch == 0:
            print('WARNING: Best epoch is 0')
            print('Val loss', val_loss)
            print('Train loss', train_loss)
        logger.info(f'Best epoch {best_epoch}, Best val loss {BEST_VAL_LOSS}')

        ##################################### VAL #################################################
        test_output_, y_ = self._predict_test_loader(self.val_loader, output_pdf='test', calibrate=True)
        test_output_ = test_output_.detach().cpu().numpy()
        y_ = y_.detach().cpu().numpy()

        for H in range(self.horizon + 1):
            check_and_create_path(self.dir_log / f"H{H}")
            y = y_[:, :, -1 - (self.horizon - H)]
            test_output = test_output_[:, -1 - (self.horizon - H)]

            if np.any(y[:, -1] > 0) or np.any(test_output > 0):

                under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
                over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
                
                iou = iou_score(y[:, -1], test_output)
                f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
                iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

                print(f'Horizon {H} -> Val -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou}, f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

                plt.figure(figsize=(15,5))
                plt.plot(y[y[:, departement_index] == 13, -1])
                plt.plot(test_output[y[:, departement_index] == 13])
                plt.savefig(self.dir_log / f"H{H}" / 'test_13.png')

                plt.figure(figsize=(15,5))
                plt.plot(y[y[:, departement_index] == 6, -1])
                plt.plot(test_output[y[:, departement_index] == 6])
                plt.savefig(self.dir_log / f"H{H}" / 'test_6.png')
                plt.close('all')

        ##################################### Test #################################################
        test_output_, y_ = self._predict_test_loader(self.test_loader, output_pdf='test')
        test_output_ = test_output_.detach().cpu().numpy()
        y_ = y_.detach().cpu().numpy()
        
        for H in range(self.horizon + 1):

            y = y_[:, :, -1 - (self.horizon - H)]
            test_output = test_output_[:, -1 - (self.horizon - H)]
            
            under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
            over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
            
            iou = iou_score(y[:, -1], test_output)
            f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
            iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

            print(f'Horizon {H} -> Test {y.shape} -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou} f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

        if BEST_MODEL_PARAMS is not None:
            self.update_weight(BEST_MODEL_PARAMS)
        
        if 'learnable-area' in self.loss:
            ids = y[:, 0]                       # première colonne
            values = y[:, 1:]

            # Somme groupée par id
            unique_ids, inverse = np.unique(ids, return_inverse=True)
            sums = np.zeros((len(unique_ids), values.shape[1]), dtype=values.dtype)
            np.add.at(sums, inverse, values)
            
            self.plot_area_parameter(epochs_list, y[:, 0], sums[:, -1])
            save_object(self.area_parameters_log, 'area_parameters_log.pkl' ,self.dir_log)

        if has_method(criterion, 'plot_params'):
            if has_method(criterion, 'update_params'):
                criterion.update_params(self.criterion_params[best_epoch])
            criterion.plot_params(self.criterion_params, self.dir_log, best_epoch=best_epoch)

        # --- LOG LOSS COMPONENTS ---
        # "Je veux les valeurs brutes, sans les multiplications par les lambda"
        if hasattr(criterion, 'epoch_stats'):
            est_g = criterion.epoch_stats.get('global', {})
            if criterion.epoch_stats:
                # We take the mean of the values collected during the epoch for the global component
                # Note: epoch_stats accumulates values at each batch.
                # Ideally we want the average over the epoch.
                
                # Helper to safely get mean
                def safe_mean(key):
                    vals = est_g.get(key, [])
                    if vals: 
                        return np.mean(vals)
                    
                    # Fallback: aggregate from cluster stats if global is missing the key
                    all_vals = []
                    for k, v in criterion.epoch_stats.items():
                        if k == 'global': continue
                        if isinstance(v, dict) and key in v and v[key]:
                            all_vals.extend(v[key])
                    
                    if all_vals:
                        return np.mean(all_vals)
                        
                    return 0.0
                
                # Raw values
                # Raw values
                l_trans = safe_mean('loss_trans')
                l_ent = safe_mean('entropy_pi')
                l_mu0 = safe_mean('mu0_term')
                l_dir = safe_mean('dirichlet_reg')
                l_ce  = safe_mean('ce_loss')
                # Total loss is trickier because it's weighted sum, but let's take loss_total if available
                l_total = safe_mean('loss_total')
                
                # Scaling stats
                s_min = safe_mean('scale_min')
                s_mean = safe_mean('scale_mean')
                s_max = safe_mean('scale_max')
                d_raw = safe_mean('diff_raw_mean')
                d_scaled = safe_mean('diff_scaled_mean')
                m_mean = safe_mean('margin_mean')
                
                # Weighting for loss components plot
                l_ent_w = l_ent
                if hasattr(criterion, 'lambdaentropy'):
                    l_ent_w = - criterion.lambdaentropy * l_ent
                    
                l_dir_w = l_dir
                if hasattr(criterion, 'lambdadir'):
                    l_dir_w = criterion.lambdadir * l_dir

                self.loss_components_history['loss_total'].append(l_total)
                self.loss_components_history['loss_trans'].append(l_trans)
                self.loss_components_history['entropy_pi'].append(l_ent)
                self.loss_components_history['mu0_term'].append(l_mu0)
                self.loss_components_history['dirichlet_reg'].append(l_dir)
                self.loss_components_history['ce_loss'].append(l_ce)
                self.loss_components_history['epoch'].append(epochs_list[-1]) # Current epoch
                
                self.loss_components_history['scale_min'].append(s_min)
                self.loss_components_history['scale_mean'].append(s_mean)
                self.loss_components_history['scale_max'].append(s_max)
                self.loss_components_history['diff_raw_mean'].append(d_raw)
                self.loss_components_history['diff_scaled_mean'].append(d_scaled)
                self.loss_components_history['margin_mean'].append(m_mean)
                
                # Plot
                self.plot_loss_decomposition()
                self.plot_scaling_decomposition()
                
        # Save distillation best/worst logs and 3D plot at the end of training

        if 'distillation' in self.loss:
            try:
                self._save_distill_logs_and_plot()
            except Exception as _e:
                # Keep training flow robust even if plotting fails
                logger.info(f"Distillation log/plot skipped: {_e}")

        self.params = BEST_MODEL_PARAMS

    def _save_distill_logs_and_plot(self):
        """Persist best/worst per-epoch logs and save a 3D scatter plot.
        Axes: X=epoch, Y=loss, Z=graph_id. Two series: best (green) and worst (red).
        """
        # Persist raw logs
        logs = {
            'best': self.distill_best_log,
            'worst': self.distill_worst_log,

        }
        save_object(logs, 'distill_best_worst.pkl', self.dir_log)

        if len(self.distill_best_log) == 0 and len(self.distill_worst_log) == 0:
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

        # Prepare arrays for plotting
        bx = [d['epoch'] for d in self.distill_best_log]
        by = [d['loss'] for d in self.distill_best_log]
        bz = [d['graph_id'] for d in self.distill_best_log]

        wx = [d['epoch'] for d in self.distill_worst_log]
        wy = [d['loss'] for d in self.distill_worst_log]
        wz = [d['graph_id'] for d in self.distill_worst_log]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        if len(bx) > 0:
            ax.scatter(bx, by, bz, c='green', marker='o', label='best')
        if len(wx) > 0:
            ax.scatter(wx, wy, wz, c='red', marker='^', label='worst')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_zlabel('Graph ID')
        ax.set_title('Distillation Best/Worst per Epoch')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.dir_log / 'distill_best_worst_3d.png')
        plt.close('all')
        plt.close('all')

    def plot_loss_decomposition(self):
        """
        Plots the raw values of different loss components over epochs.
        Saves to 'loss_decomposition.png'.
        """
        if not self.loss_components_history['epoch']:
            return

        epochs = self.loss_components_history['epoch']
        
        # Prepare figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # We can plot everything on same axis or use twin axis if scales are very different.
        # Given "ordre de grandeur", maybe log scale or twin axis is better.
        # Let's try plotting raw values on linear scale first, but with different colors.
        
        ax1.plot(epochs, self.loss_components_history['loss_trans'], label='Transitional Loss (task)', color='blue')
        ax1.plot(epochs, self.loss_components_history['entropy_pi'], label='Entropy Term (-H * lambda)', color='green', linestyle='--')
        ax1.plot(epochs, self.loss_components_history['mu0_term'], label='Mu0 Term', color='orange', linestyle=':')
        ax1.plot(epochs, self.loss_components_history['dirichlet_reg'], label='Dirichlet Reg (R * lambda)', color='red', linestyle='-.')
        if any(v != 0 for v in self.loss_components_history['ce_loss']):
             ax1.plot(epochs, self.loss_components_history['ce_loss'], label='CE Loss', color='purple', linestyle='-')
        ax1.plot(epochs, self.loss_components_history['loss_total'], label='Total Loss', color='black', linewidth=2, alpha=0.5)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Raw Value')
        ax1.set_title('Loss Components Decomposition (Raw Values)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dir_log / 'loss_decomposition.png')
        plt.close(fig)

    def plot_scaling_decomposition(self):
        """
        Plots the scaling and margin metrics over epochs.
        Saves to 'loss_scaling_decomposition.png'.
        """
        if not self.loss_components_history['epoch']:
            return

        epochs = self.loss_components_history['epoch']
        
        # Prepare figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Scale stats
        ax1 = axes[0]
        ax1.plot(epochs, self.loss_components_history['scale_mean'], label='Scale Mean', color='blue')
        ax1.fill_between(epochs, self.loss_components_history['scale_min'], self.loss_components_history['scale_max'], color='blue', alpha=0.2, label='Min-Max Range')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Scale Value')
        ax1.set_title('Scale Statistics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Margins and Diffs
        ax2 = axes[1]
        ax2.plot(epochs, self.loss_components_history['diff_raw_mean'], label='Raw Diff Mean', color='orange')
        ax2.plot(epochs, self.loss_components_history['diff_scaled_mean'], label='Scaled Diff Mean (raw/scale)', color='purple', linestyle='--')
        ax2.plot(epochs, self.loss_components_history['margin_mean'], label='Margin Mean (gains)', color='green', linestyle='-.')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.set_title('Margins & Diffs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dir_log / 'loss_scaling_decomposition.png')
        plt.close(fig)

    def plot_area_parameter(self, epochs_list, ids, sinisters):
        """
        Plot self.area_parameter (2D array: epochs x parameters) in 3D.
        X = epochs_list
        Y = parameter index
        Z = value of area_parameter
        """
    
        from mpl_toolkits.mplot3d import Axes3D

        sort_ids = np.argsort(sinisters)

        # Vérifier dimensions
        area_param = np.asarray([params.detach().numpy()[sort_ids] for params in self.area_parameters_log])  # doit être (epochs, n_params)
        
        logger.info(f'Last aera params -> {area_param[-1]}')

        # Créer grilles X et Y
        X = epochs_list
        Y = sinisters[sort_ids]
        X, Y = np.meshgrid(X, Y)
        
        # Z = valeurs de self.area_parameter transposées pour correspondre à la grille
        Z = area_param.T  
        
        # Plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Parameter Index')
        ax.set_zlabel('Area Parameter Value')
        ax.set_title('Evolution of Area Parameter during Training')
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(self.dir_log / 'area_parameters.png')
        plt.close()

    def split_dataset(self, dataset, nb, reset=True):
        # Separate the positive and zero classes based on y
        positive_mask = dataset[self.target_name] > 0
        non_fire_mask = dataset[self.target_name] == 0

        # Filtrer les données positives et non feu
        df_positive = dataset[positive_mask]
        df_non_fire = dataset[non_fire_mask]

        # Échantillonner les données non feu
        nb = min(len(df_non_fire), nb)

        if self.n_run == 1:
            seed = self.seed if self.seed is not None else 42
            sampled_indices = np.random.RandomState(seed).choice(len(df_non_fire), nb, replace=False)
        else:
            sampled_indices = np.random.RandomState().choice(len(df_non_fire), nb, replace=False)
            
        df_non_fire_sampled = df_non_fire.iloc[sampled_indices]

        # Combiner les données positives et non feu échantillonnées
        df_combined = pd.concat([df_positive, df_non_fire_sampled])
        # Réinitialiser les index du DataFrame combiné
        if reset:
            df_combined.reset_index(drop=True, inplace=True)
        return df_combined
    
    def add_ordinal_class(self, X, y, limit):        
        pass

    def calculcate_score(self, pred, y, id_mask=None):
        if id_mask is None:
            under_prediction_score_value = under_prediction_score(y, pred)
            over_prediction_score_value = over_prediction_score(y, pred)

            iou = iou_score(y, pred)
            return under_prediction_score_value, over_prediction_score_value, iou
        else:
            uids = np.unique(id_mask)
            under_prediction_score_value = []
            over_prediction_score_value = []
            iou = []
            
            for id in uids:
                mask = (id_mask == id)
                pred_mask = pred[mask]
                y_mask = y[mask]
                
                if np.any(y_mask > 0):

                    under_score = under_prediction_score(y_mask, pred_mask)
                    over_score = over_prediction_score(y_mask, pred_mask)
                    iou_val = iou_score(y_mask, pred_mask)

                    # Stocker les valeurs
                    under_prediction_score_value.append(under_score)
                    over_prediction_score_value.append(over_score)
                    iou.append(iou_val)

                    # Log propre
                    logger.info(
                        f'Id {id} -> under_prediction_score: {under_score}, over_prediction_score: {over_score}, iou: {iou_val}'
                    )
                
                else:
                     logger.info(
                        f'Id {id} -> No fire'
                    )

            under_prediction_score_value = np.trapz(under_prediction_score_value)
            over_prediction_score_value = np.trapz(over_prediction_score_value)
            iou = np.trapz(iou)

            return under_prediction_score_value, over_prediction_score_value, iou

    def compute_area_score(self, pred, y_true, graph_ids):
        unique_graphs = np.unique(graph_ids)
        graph_sums = {gid: y_true[graph_ids == gid].sum() for gid in unique_graphs}
        sorted_graphs = sorted(graph_sums, key=graph_sums.get, reverse=True)

        iou_scores = []
        f1_scores = []

        for gid in sorted_graphs:
            mask = graph_ids == gid
            y_g = y_true[mask]
            pred_g = pred[mask]
            if np.any(y_g > 0):
                iou = jaccard_score((y_g > 0).astype(int), (pred_g > 0).astype(int), zero_division=0)
                f1 = f1_score((y_g > 0).astype(int), (pred_g > 0).astype(int), zero_division=0)
                iou_scores.append(iou)
                f1_scores.append(f1)

        if len(iou_scores) == 0:
            return 0.0, 0.0

        max_area = np.trapz(np.ones(np.unique(graph_ids[y_true > 0]).shape[0]))
        IoU_area = calculate_area_under_curve(iou_scores)
        F1_area = calculate_area_under_curve(f1_scores)
        if max_area == 0:
            return 0, 0
        return IoU_area / max_area, F1_area / max_area

    def search_samples_proportion(self, graph, df_train, df_val, df_test, is_unknowed_risk, epochs, PATIENCE_CNT, CHECKPOINT, reset=True, custom_model_params=None, use_log=True):
        
        check_and_create_path(self.dir_log)

        if not is_unknowed_risk:
                test_percentage = np.round(np.arange(0.1, 1.05, 0.1), 2)
        else:
            test_percentage = np.arange(0.0, 1.05, 0.05)

        if 'MultiScale' in self.model_name:
            test_percentage = np.arange(0.5, 1.05, 0.05)

        under_prediction_score_scores = []
        over_prediction_score_scores = []
        iou_scores = []
        data_log = None
        find_log = False

        self.metrics['test_percentage'] = []
        self.metrics['under_prediction_scores'] = []
        self.metrics['over_predictio_scores'] = []
        self.metrics['iou_scores'] = []
        
        if use_log:
        #if False:
            if False:
                if (self.dir_log / 'unknowned_scores_per_percentage.pkl').is_file():
                    data_log = read_object('unknowned_scores_per_percentage.pkl', self.dir_log)
            else:
                print(self.dir_log / 'metrics.pkl')
                if (self.dir_log / 'metrics.pkl').is_file():
                    print(f'Load metrics')
                    find_log = True
                    data_log = read_object('metrics.pkl', self.dir_log)
                else:
                    xs = [0, 10]
                    for x in xs:
                        other_model = f'{self.model_name}_search_full_{x}_all_one_{self.target_name}_{self.task_type}_{self.loss}'
                        print(f'{self.dir_log / ".."/ other_model / "metrics.pkl"}')
                        if (self.dir_log / '..'/ other_model / 'metrics.pkl').is_file():
                            data_log = read_object('metrics.pkl', self.dir_log / '..'/ other_model)
                        if data_log is not None:
                            break
                        
                    if data_log is None:
                        xs = [25]
                        for x in xs:
                            other_model = f'{self.model_name}_search_full_{self.ks}_{x}_one_{self.target_name}_{self.task_type}_{self.loss}'
                            print(f'{self.dir_log / ".."/ other_model / "metrics.pkl"}')
                            if (self.dir_log / '..'/ other_model / 'metrics.pkl').is_file():
                                data_log = read_object('metrics.pkl', self.dir_log / '..'/ other_model)
                            if data_log is not None:
                                break

        print(f'data_log : {data_log}')
        if data_log is not None:
            try:
                self.metrics = data_log
                #test_percentage = self.metrics['test_percentage']
                #under_prediction_score_scores = self.metrics['under_prediction_scores']
                #over_prediction_score_scores = self.metrics['over_prediction_scores']
            except Exception as e:
                print(e)
                self.metrics = {}
                data_log = None
                pass

        doSearch = True
        if data_log is not None: #and self.n_run == data_log['n_run']:
            test_percentage = np.asarray(self.metrics['test_percentage'])
            start_test = -1
            for i in range(0, len(test_percentage) - 1):
                start_test = i

                if test_percentage[i] in self.metrics.keys():
                    last_keys = test_percentage[i]
                    val_1 = np.mean(data_log[test_percentage[i]]['iou_val'])
                    val_2 = np.mean(data_log[test_percentage[i + 1]]['iou_val'])

                    std_1 = np.std(data_log[test_percentage[i]]['iou_val'])
                    std_2 = np.std(data_log[test_percentage[i + 1]]['iou_val'])

                    print('#########################"')
                    print(f'{test_percentage[i]} -> {val_1} -> {std_1}')
                    print(f'{test_percentage[i + 1]} -> {val_2} -> {std_2}')

                    try:
                        if  val_1 >  val_2 or ((val_1 == val_2) and (std_1 < std_2)):
                            print(f"Last score {val_1} current score {val_2}")
                            print(f"Last std {std_1} current std {std_2}")
                            doSearch = False
                            tp = test_percentage[i]
                            break
                    except Exception as e:
                        print(e)
                        doSearch = True
                        break

            start_test += 1

        else:
            start_test = 0

        tolerance = 0.03
        
        if doSearch:
            last_score = -math.inf if start_test == 0 else np.mean(self.metrics[last_keys]['iou_val'])
            y_ori = df_train[self.target_name].values
            for i in range(start_test, test_percentage.shape[0]):
                tp = round(test_percentage[i], 2)

                if tp in self.metrics.keys():
                    continue
                
                df_train_copy = df_train.copy(deep=True)
                
                if not is_unknowed_risk:
                    nb = int(tp * y_ori[y_ori == 0].shape[0])
                else:
                    nb = int(tp * len(X[(X['potential_risk'] > 0) & (y_ori == 0)]))

                logger.info(f'Trained with {tp} -> {nb} sample of class 0')

                for run in range(self.n_run):

                    df_combined = self.split_dataset(df_train_copy, nb, reset=False)

                    # Mettre à jour df_train pour l'entraînement
                    #df_train_copy["weight"] = df_combined["weight"].reindex(df_train_copy.index, fill_value=0)
                    
                    df_train_copy['weight'] = 0
                    weight = egpd_trunc_discrete_weights(df_combined[self.target_name].values, df_combined['graph_id'].values)
                    df_train_copy.loc[df_combined.index, 'weight'] = 1
                    #df_train_copy.loc[df_combined.index, 'weight'] = weight
                    
                    # PREVENT DEEP COPY OF HEAVY OBJECTS
                    # We strip all data-related attributes from 'self' before deepcopy
                    # and restore them immediately after.
                    
                    # 1. Save references
                    ref_graph = getattr(self, 'graph', None)
                    ref_df_train = getattr(self, 'df_train', None)
                    ref_df_val = getattr(self, 'df_val', None)
                    ref_df_test = getattr(self, 'df_test', None)
                    ref_val_loader = getattr(self, 'val_loader', None)
                    ref_test_loader = getattr(self, 'test_loader', None)
                    ref_train_loader = getattr(self, 'train_loader', None)
                    ref_optimizer = getattr(self, 'optimizer', None)
                    ref_metrics = getattr(self, 'metrics', {})
                    
                    # 2. Unset attributes
                    self.graph = None
                    self.df_train = None
                    self.df_val = None
                    self.df_test = None
                    self.val_loader = None
                    self.test_loader = None
                    self.train_loader = None
                    self.optimizer = None
                    self.metrics = {}  # Empty dict to avoid copying full history

                    # 3. Deepcopy
                    copy_model = deepcopy(self)
                    
                    # 4. Restore attributes
                    self.graph = ref_graph
                    self.df_train = ref_df_train
                    self.df_val = ref_df_val
                    self.df_test = ref_df_test
                    self.val_loader = ref_val_loader
                    self.test_loader = ref_test_loader
                    self.train_loader = ref_train_loader
                    self.optimizer = ref_optimizer
                    self.metrics = ref_metrics

                    copy_model.under_sampling = 'full'
                    copy_model.horizon = 0
                    copy_model.create_train_val_test_loader(graph, df_train_copy, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT, features_importance=False, custom_model_params=custom_model_params)
                    copy_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=False, custom_model_params=custom_model_params)
                    
                    ############################# On set val ##############################
                    test_output, y = copy_model._predict_test_loader(copy_model.val_loader, output_pdf='Val', prediction_type='Class')
                    
                    test_output = test_output[:, 0]
                    y = y[:, :, 0]
                    
                    prediction = test_output.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                    
                    if 'MultiScale' in self.model_name:
                        id_mask = y[:, scale_index]
                    else:
                        id_mask = y[:, departement_index]
                        id_mask = None
                        
                    dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                    dff['departement'] = y[:, departement_index]
                    dff['date'] = y[:, date_index]
                    dff['graph_id'] = y[:, graph_id_index]
                    dff[self.target_name] = y[:, -1]
                    y = y[:, -1] > 0 if self.task_type == 'binary' else y[:, -1]

                    metrics_run = evaluate_metrics(dff[self.target_name], prediction, zones=dff['graph_id'], dates=dff['date'])
                    metrics_run = round_floats(metrics_run)
                    under_prediction_score_value = under_prediction_score(y, prediction)
                    over_prediction_score_value = over_prediction_score(y, prediction)
                    update_metrics_as_arrays(self, tp, metrics_run, 'val')

                    ############################# On set test ##############################
                    test_output, y = copy_model._predict_test_loader(copy_model.test_loader, output_pdf='Test', prediction_type='Class')
                    
                    test_output = test_output[:, 0]
                    y = y[:, :, 0]
                    
                    prediction = test_output.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                    
                    if 'MultiScale' in self.model_name:
                        id_mask = y[:, scale_index]
                    else:
                        id_mask = y[:, departement_index]
                        id_mask = None
                
                    dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
                    dff['departement'] = y[:, departement_index]
                    dff['date'] = y[:, date_index]
                    dff['graph_id'] = y[:, graph_id_index]
                    dff[self.target_name] = y[:, -1]
                    y = y[:, -1] > 0 if self.task_type == 'binary' else y[:, -1]

                    metrics_run = evaluate_metrics(dff[self.target_name], prediction, zones=dff['graph_id'], dates=dff['date'])
                    metrics_run = round_floats(metrics_run)
                    under_prediction_score_value = under_prediction_score(y, prediction)
                    over_prediction_score_value = over_prediction_score(y, prediction)
                    update_metrics_as_arrays(self, tp, metrics_run, 'test')
                    
                    # Manually cleanup deepcopied model to free memory
                    del copy_model
                    gc.collect()
                
                self.metrics[tp] = add_ic95_to_dict(self.metrics[tp], None, "_ic95")

                # REPLACED: Use 'score' (score_high + score_low) instead of 'iou'
                # metrics has key 'score_val' because update_metrics_as_arrays appends '_val' suffix
                
                iou = np.mean(self.metrics[tp]['score_val']) 
                std_iou = np.std(self.metrics[tp]['score_val'])

                save_object(self.metrics, 'metrics.pkl', self.dir_log)
                
                #print(f'Metrics achieved : {self.metrics[tp]}')

                # REPLACED: Tolerance 0.0 because score scale is arbitrary/large and we want strict maximization
                # CHANGED: We now want to scan ALL candidates for Rank-Based Selection. 
                # So we update last_score for logging but DO NOT BREAK early.
                
                tolerance = 0.0
                if iou >= last_score - tolerance:
                    last_score = iou
                else:
                    print(f'Last score {last_score} current score {iou} (Continuing search for Rank Selection)')
                    # break  <-- COMMENTED OUT TO TEST ALL CANDIDATES
        
        # --- RANK-BASED SELECTION (Borda Count) ---
        # Select tp that minimizes sum of ranks across k=1, 2, 3, 4
        
        tp_candidates = []
        scores_per_k = {1: {}, 2: {}, 3: {}, 4: {}}
        
        # 1. Collect scores for all tp
        for tp in self.metrics.keys():
            if isinstance(tp, float):
                # Check if all k-scores are present
                has_all_k = True
                for k in [1, 2, 3, 4]:
                    key = f"score_k{k}_val"
                    # self.metrics might store single values or arrays
                    if key in self.metrics[tp]:
                         vals = np.atleast_1d(self.metrics[tp][key])
                         # Use mean of values (e.g. cross-validation folds)
                         mean_k = np.nanmean(vals)
                         if np.isnan(mean_k):
                             # print(f"DEBUG: NaN value for {key} for tp={tp}")
                             mean_k = 0.0 # Default to 0.0 if NaN
                         
                         scores_per_k[k][tp] = mean_k
                    else:
                        # print(f"DEBUG: Missing key {key} for tp={tp}")
                        scores_per_k[k][tp] = 0.0 # Default to 0.0 if missing
                
                # We always consider the TP as a candidate now, unless other fundamental issues exist
                tp_candidates.append(tp)
        
        if not tp_candidates:
             # Fallback if no valid k-scores found (e.g. older metrics format or nan)
             logger.warning("No valid k-scores (k1..k4) found for Rank Selection. Falling back to simple score maximization.")
             try:
                 # Fallback logic: maximization of score_val
                 best_score = -np.inf
                 best_tp = None
                 for tp in self.metrics.keys():
                     if isinstance(tp, float) and "score_val" in self.metrics[tp]:
                         s = np.nanmean(self.metrics[tp]["score_val"])
                         if s > best_score:
                             best_score = s
                             best_tp = tp
                 if best_tp is None:
                      raise ValueError("No valid scores found for fallback selection.")
             except Exception as e:
                 raise ValueError(f"Rank Selection failed and Fallback failed: {e}")
                 
        else:
            # 2. Compute Ranks
            # Rank 1 = Best (Highest Score)
            # Ranks are 1-based indices in sorted list
            rank_sums = {tp: 0 for tp in tp_candidates}
            
            for k in [1, 2, 3, 4]:
                # Sort descending by score_k
                sorted_tps = sorted(tp_candidates, key=lambda x: scores_per_k[k][x], reverse=True)
                for rank, tp in enumerate(sorted_tps, 1):
                    rank_sums[tp] += rank
                    
            # 3. Find Best TP (Lowest Rank Sum)
            # Tie-break: Maximize score_k4 (High severity), then Total Score
            # To maximize k4 with min(), we negate it.
            def tie_breaker(tp):
                total_s = sum(scores_per_k[k][tp] for k in [1,2,3,4])
                return (rank_sums[tp], -scores_per_k[4][tp], -total_s)
                
            best_tp = min(tp_candidates, key=tie_breaker)
            
            # Logging details
            logger.info("--- Rank-Based Selection Results ---")
            for tp in tp_candidates:
                ranks = []
                # Re-calculate specific ranks for logging
                for k in [1,2,3,4]:
                    sorted_tps = sorted(tp_candidates, key=lambda x: scores_per_k[k][x], reverse=True)
                    r = sorted_tps.index(tp) + 1
                    ranks.append(r)
                
                logger.info(f"tp={tp}: RankSum={rank_sums[tp]} (Ranks k1..k4: {ranks})")

        logger.info(f'Best tp {best_tp} (Rank-Based)')
        self.metrics['iou_score'] = iou_scores # Keep legacy key name or update? Let's keep data but variable name is misleading. It's actually score history list but variable iou_scores was empty anyway here
        self.metrics['test_percentage'] = test_percentage
        self.metrics['under_prediction_scores'] = under_prediction_score_scores
        self.metrics['over_prediction_scores'] = over_prediction_score_scores
        self.metrics['best_tp'] = best_tp
        self.metrics['run'] = self.n_run

        #logger.info(f'{self.metrics[best_tp]}')

        save_object(self.metrics, 'metrics.pkl', self.dir_log)

        return best_tp, find_log

    def search_samples_limit(self, X, y, X_val, y_val, X_test, y_test):
        pass

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the model's performance for each ID.

        Parameters:
        - X_val: Validation data.
        - y_val: True labels.
        - id_val: List of IDs corresponding to validation data.

        Returns:
        - Mean score across all IDs.
        """
        predictions, y = self.predict(X, return_y=True)
        predictions = predictions[:, 0]
        y = y[:, -1, 0]
        return self.score_with_prediction(predictions, y, sample_weight)
    
    def score_with_prediction(self, y_pred, y, sample_weight=None):
        
        return iou_score(y, y_pred)

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf="test", calibrate=False) -> torch.tensor:
            assert self.model is not None
            self.model.eval()
            criterion = self.get_loss(self.loss, {})
            if len(self.criterion_params) > 0:
                if has_method(criterion, 'update_params'):
                    criterion.update_params(self.criterion_params[self.best_epoch])
                    criterion.eval()

            with torch.no_grad():
                pred = []
                y = []

                for _, data in enumerate(X, 0):
                    
                    pred_horizon, labels_horizon = self._predict_tensor(data, prediction_type=prediction_type, output_pdf=output_pdf, calibrate=calibrate)
                        
                    pred.append(pred_horizon)
                    y.append(labels_horizon)

                y = torch.cat(y, 0)
                pred = torch.cat(pred, 0)

                if self.task_type == 'regression' and prediction_type == 'Class' and self.apply_discretization:
                    for H in range(self.horizon + 1):
                        pred_h = pred[:, -1 - (self.horizon - H)].detach().cpu().numpy()
                        y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                        pred_h = self.post_process.predict(pred_h, pred_h, y_cluster)
                        pred[:, -1 - (self.horizon - H)] = torch.as_tensor(pred_h)
                        
                        y_h = y[:, -1, -1 - (self.horizon - H)].detach().cpu().numpy()
                        y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                        y_h = self.post_process.predict(y_h, y_h, y_cluster)
                        y[:, -1, -1 - (self.horizon - H)] = torch.as_tensor(y_h)
                        
                        print(np.unique(pred_h), np.unique(y_h))
                        
                elif prediction_type == 'Class' and pred.dtype != torch.long:
                #if pred.dtype != torch.long:
                    pred = torch.round(pred, decimals=1)
                    
            return pred, y
            
    def _predict_tensor(self, X, prediction_type='Class', output_pdf="test", calibrate=False) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        criterion = self.get_loss(self.loss, {})
        if len(self.criterion_params) > 0:
            if has_method(criterion, 'update_params'):
                criterion.update_params(self.criterion_params[self.best_epoch])
                criterion.eval()

        with torch.no_grad():
                
            inputs, orilabels_, _ = X

            orilabels_ = orilabels_.to(device)
            pred_horizon = []
            labels_horizon = []

            hidden_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
            output_past: List[torch.Tensor] = []  # contiendra des tenseurs (B, D)
            for H in range(self.horizon + 1):

                orilabels = orilabels_[:, :, -1 - (self.horizon - H)]
                orilabels[:, -1] = orilabels[:,  -1 ] > 0 if self.task_type == 'binary' else orilabels[:,  -1 ]
                inputs_horizon = self.compute_inputs(inputs,  -1 - (self.horizon - H), "current" if H == 0 else "futur")
                
                if H == 0:
                    z_prev = None
                else:
                    if self.ks > 0:
                        # on prend les ks derniers états cachés déjà vus
                        history = hidden_past[-(self.ks + 1):]
                        # empilement (B, D, L) avec L = len(history)
                        z_prev = torch.stack(history, dim=2)  # (B, D, L)

                        # padding à gauche si L < ks
                        L = z_prev.size(2)
                        if L < (self.ks + 1):
                            B, D = z_prev.size(0), z_prev.size(1)
                            pad = torch.zeros(
                                (B, D, self.ks + 1 - L),
                                device=z_prev.device,
                                dtype=z_prev.dtype
                            )
                            z_prev = torch.cat([pad, z_prev], dim=2)  # (B, D, ks)
                    else:
                        z_prev = hidden_past[-1]
                if H == 0:
                    output, logits, hidden = self.model(inputs_horizon, z_prev=None)
                else:
                    if self.id_past_risk is not None:
                        inputs_horizon[:, self.id_past_risk, -H:] = 0
                    if self.id_past_ba is not None:
                        inputs_horizon[:, self.id_past_ba, -H:] = 0
                    if self.prev_idx is not None:
                        inputs_horizon[:, self.prev_idx, -H:] = torch.stack(output_past, dim=2)
                    
                    output, logits, hidden = self.model(inputs_horizon, z_prev=z_prev)
                
                hidden_past.append(hidden)
                output_past.append(output)
                
                if prediction_type != 'RawFormulaVal':
                    if 'criterion' in locals() and hasattr(criterion, 'calibrate') and calibrate:
                        if 'clusters_ids' in required_params(criterion.transform):
                            clusters_ids = orilabels[:, criterion.id].long()
                            calibration = criterion.calibrate(inputs=logits, y_true=orilabels[:, -1], score_fn=iou_score, clusters_ids=clusters_ids, dir_output=self.dir_log)
                        else:
                            calibration = criterion.calibrate(inputs=logits, y_true=orilabels[:, -1], score_fn=iou_score, dir_output=self.dir_log)
                        
                        self.calibration = calibration
                    
                    elif 'criterion' in locals() and hasattr(criterion, 'calibrate'):
                        assert hasattr(self, 'calibration')
                            
                    if 'criterion' in locals() and hasattr(criterion, 'transform'):
                        params = {'inputs' : logits}
                        if 'clusters_ids' in required_params(criterion.transform):
                            clusters_ids = orilabels[:, criterion.id].long()
                            params['clusters_ids'] = clusters_ids
                            
                        if 'output_pdf' in required_params(criterion.transform):
                            assert output_pdf is not None and self.dir_log is not None
                            params['output_pdf'] = output_pdf
                        
                        if 'dir_output' in required_params(criterion.transform):    
                            params['dir_output'] = self.dir_log
                            
                        if 'areas' in required_params(criterion.transform):
                            params['areas'] = orilabels[:, area_index]
                            
                        if 'p_thresh' in required_params(criterion.transform):
                            params['p_thresh'] = self.calibration

                        params['prediction_type'] = prediction_type
                        output = criterion.transform(**params)
                        
                if prediction_type == 'Class':
                    
                    if self.task_type == 'classification' or self.task_type == 'binary':
                        output = torch.argmax(output, dim=1)

                    elif self.task_type == 'regression' and output.ndim > 1 and output.shape[1] > 1:
                        output = torch.argmax(output, dim=1)

                elif prediction_type == 'RawFormulaVal':
                    output = logits
                    
                pred_horizon.append(output[:, None])
                labels_horizon.append(orilabels[:, :, None])
                
        pred = torch.cat(pred_horizon, dim=1)
        y = torch.cat(labels_horizon, dim=2)

        if self.task_type == 'regression' and prediction_type == 'Class' and self.apply_discretization:
            for H in range(self.horizon + 1):
                pred_h = pred[:, -1 - (self.horizon - H)].detach().cpu().numpy()
                y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                pred_h = self.post_process.predict(pred_h, pred_h, y_cluster)
                pred[:, -1 - (self.horizon - H)] = torch.as_tensor(pred_h)
                
                y_h = y[:, -1, -1 - (self.horizon - H)].detach().cpu().numpy()
                y_cluster = y[:, departement_index, -1 - (self.horizon - H)]
                y_h = self.post_process.predict(y_h, y_h, y_cluster)
                y[:, -1, -1 - (self.horizon - H)] = torch.as_tensor(y_h)
                
                print(np.unique(pred_h), np.unique(y_h))
                
        elif prediction_type == 'Class' and pred.dtype != torch.long:
        #if pred.dtype != torch.long:
            pred = torch.round(pred, decimals=1)
            
        return pred, y
    
    def fit(self, graph, X, y, X_val, y_val, X_test, y_test, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=None, use_log=True):
        
        X = X.set_index(ids_columns[:-1]).join(y.set_index(ids_columns[:-1])[targets_columns + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        X_val = X_val.set_index(ids_columns[:-1]).join(y_val.set_index(ids_columns[:-1])[targets_columns + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        X_test = X_test.set_index(ids_columns[:-1]).join(y_test.set_index(ids_columns[:-1])[targets_columns  + [self.target_name]], on=ids_columns[:-1], how='left').reset_index()
        #if (self.dir_log / 'last.pt').is_file():
        #    self.graph = graph
        #    self._load_model_from_path(self.dir_log / 'best.pt', self.model)
        #else:
        self.create_train_val_test_loader(graph, X, X_val, X_test, epochs, PATIENCE_CNT, CHECKPOINT, custom_model_params=custom_model_params, use_log=use_log)
        self.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=custom_model_params)
            
    def filtering_pred(self, df, predTensor, y, graph, return_y = False):
        
        y = y.detach().cpu().numpy()
        predTensor = predTensor.detach().cpu().numpy()

        # Extraire les paires de test_dataset_dept
        test_pairs = set(zip(df['date'], df['graph_id'], df['scale']))

        # Normaliser les valeurs dans YTensor
        date_values = [item for item in y[:, date_index, 0]]
        graph_id_values = [item for item in y[:, graph_id_index, 0]]
        scale_values = [item for item in y[:, scale_index, 0]]

        # Filtrer les lignes de YTensor correspondant aux paires présentes dans test_dataset_dept
        filtered_indices = [
            i for i, (date, graph_id, scale) in enumerate(zip(date_values, graph_id_values, scale_values)) 
            if (date, graph_id, scale) in test_pairs
            ]

        # Créer YTensor filtré
        y = y[filtered_indices]

        # Créer des paires et les convertir en set
        ytensor_pairs = set(zip(date_values, graph_id_values, scale_values))

        # Filtrer les lignes en vérifiant si chaque couple (date, graph_id) appartient à ytensor_pairs
        df = df[
            df.apply(lambda row: (row['date'], row['graph_id'], row['scale']) in ytensor_pairs, axis=1)
        ].reset_index(drop=True)
        
        if graph.graph_method == 'graph':
            def keep_one_per_pair(dataset):
                # Supprime les doublons en gardant uniquement la première occurrence par paire (graph_id, date)
                return dataset.drop_duplicates(subset=['graph_id', 'date'], keep='first')
            
            def get_unique_pair_indices(array, graph_id_index, date_index):
                """
                Retourne les indices des lignes uniques basées sur les paires (graph_id, date).
                :param array: Liste de listes (tableau Python)
                :param graph_id_index: Index de la colonne `graph_id`
                :param date_index: Index de la colonne `date`
                :return: Liste des indices correspondant aux lignes uniques
                """
                seen_pairs = set()
                unique_indices = []
                for i, row in enumerate(array):
                    pair = (row[graph_id_index], row[date_index])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        unique_indices.append(i)
                return unique_indices

            unique_indices = get_unique_pair_indices(y, graph_id_index=graph_id_index, date_index=date_index)
            #predTensor = predTensor[unique_indices]
            y = y[unique_indices]
            df = keep_one_per_pair(df)

        df.sort_values(['graph_id', 'date', 'scale'], inplace=True)
        ind = np.lexsort((y[:, graph_id_index, 0], y[:, date_index, 0], y[:, scale_index, 0]))
        y = y[ind]
        predTensor = predTensor[ind]
        
        if self.target_name == 'binary' or self.target_name == 'nbsinister':
            band = -2
        else:
            band = -1

        pred = np.full((predTensor.shape[0], 1), fill_value=np.nan)
        if name in ['Unet', 'ULSTM']:
            pred = np.full((y.shape[0], 2), fill_value=np.nan)
            pred_2D = predTensor
            Y_2D = y
            udates = np.unique(df['date'].values)
            ugraph = np.unique(df['graph_id'].values)
            for graph in ugraph:
                for date in udates:
                    mask_2D = np.argwhere((Y_2D[:, graph_id_index] == graph) & (Y_2D[:, date_index] == date))
                    mask = np.argwhere((y[:, graph_id_index] == graph) & (y[:, date_index] == date))
                    if mask.shape[0] == 0:
                        continue
                    pred[mask[:, 0]] = pred_2D[mask_2D[:, 0], band, mask_2D[:, 1], mask_2D[:, 2]]
        else:
            pred = predTensor

        if return_y:
            return pred, y
        return pred

    def predict(self, df, graph=None, return_y=False, prediction_type='Class'):
        if graph is None:
            graph = self.graph

        if isinstance(df, pd.DataFrame):

            if self.target_name not in list(df.columns):
                df[self.target_name] = 0
            
            loader = create_test_loader(graph, df,
                        self.features_name,
                        self.device,
                        None,
                        self.target_name,
                        self.ks,
                        self.horizon)
            
            predTensor, YTensor = self._predict_test_loader(loader, prediction_type=prediction_type)
        
        else:
            predTensor, YTensor = self._predict_tensor(df, prediction_type=prediction_type)
            if return_y:
                return predTensor, YTensor
            else:
                return predTensor
            
        if return_y:
            return predTensor, YTensor
        
        return predTensor
    
    def predict_proba(self, df, graph=None, return_y=False, prediction_type="Proba"):
        if graph is None:
            graph = self.graph
            
        if prediction_type == 'Class':
            prediction_type = 'Proba'

        if isinstance(df, pd.DataFrame):
            
            if self.target_name not in list(df.columns):
                df[self.target_name] = 0

            loader = create_test_loader(graph, df,
                        self.features_name,
                        self.device,
                        None,
                        self.target_name,
                        self.ks,
                        self.horizon)
            
            predTensor, YTensor = self._predict_test_loader(loader, prediction_type=prediction_type)
        
        else:
            predTensor, YTensor = self._predict_tensor(df, prediction_type=prediction_type)
            if return_y:
                return predTensor, YTensor
            else:
                return predTensor
            
        if return_y:
            pred, y = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
            return pred, y
        pred = self.filtering_pred(df, predTensor, YTensor, graph, return_y=return_y)
        return pred
    
    def plot_train_val_loss(self, epochs, train_loss_list, val_loss_list, dir_log):
        # Création de la figure et des axes
        plt.figure(figsize=(10, 6))

        # Tracé de la courbe de val_loss
        plt.plot(epochs, val_loss_list, label='Validation Loss', color='blue')

        # Ajout de la légende
        plt.legend()

        # Ajout des labels des axes
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Ajout d'un titre
        plt.title('Validation Loss over Epochs')
        plt.savefig(dir_log / 'Validation.png')
        plt.close('all')

        # Tracé de la courbe de train_loss
        plt.plot(epochs, train_loss_list, label='Training Loss', color='red')

        # Ajout de la légende
        plt.legend()

        # Ajout des labels des axes
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Ajout d'un titre
        plt.title('Training Loss over Epochs')
        plt.savefig(dir_log / 'Training.png')
        plt.close('all')

    def _load_model_from_path(self, path : Path, model) -> None:
        model, _ = self.make_model(self.graph, None)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)
        self.model = model
        
    def update_weight(self, weight):
        """
        Update the model's weights with the given state dictionary.

        Parameters:
        - weight (dict): State dictionary containing the new weights.
        """

        assert self.model is not None

        if not isinstance(weight, dict):
            raise ValueError("The provided weight must be a dictionary containing model parameters.")

        model_state_dict = self.model.state_dict()

        # Vérification que toutes les clés existent dans le modèle
        missing_keys = [key for key in weight.keys() if key not in model_state_dict]
        if missing_keys:
            raise KeyError(f"Some keys in the provided weights do not match the model's parameters: {missing_keys}")

        # Charger les poids dans le modèle
        self.model.load_state_dict(weight)
    
    def update_model(self, model):
        self.model = deepcopy(model)

    def get_loss(self, loss_name, loss_params):
        loss_params.update({'num_classes' : 5})
        return get_loss_function(loss_name, **loss_params)

    def get_learnable_parameters(self, criterion):
        """
        Retourne les paramètres apprenables (list/param groups) pour l'optimizer.
        Tous les objets doivent être nn.Parameter avec requires_grad=True.
        """
        params = list(self.model.parameters())

        # Ajouter paramètres spécifiques à la loss (s'ils existent)
        if has_method(criterion, 'get_learnable_parameters'):
            logger.info(f'Adding {self.loss} parameter(s)')
            # On s'assure que ce sont bien des nn.Parameter
            loss_params = []
            for p in criterion.get_learnable_parameters().values():
                if isinstance(p, torch.nn.Parameter):
                    loss_params.append(p)
                else:
                    # Convertir un tensor en Parameter si besoin
                    loss_params.append(torch.nn.Parameter(p, requires_grad=True))
            params.extend(loss_params)

        # Ajouter les paramètres de distillation s'ils existent
        if hasattr(self, 'fitnets') and self.fitnets is not None:
            logger.info("Adding FitNets parameters")
            params.extend(list(self.fitnets.parameters()))
            
        if hasattr(self, 'relation_mlp') and self.relation_mlp is not None:
            logger.info("Adding RelationMLP parameters")
            params.extend(list(self.relation_mlp.parameters()))
            
        if hasattr(self, 'relation_att') and self.relation_att is not None:
            logger.info("Adding RelationAttention parameters")
            params.extend(list(self.relation_att.parameters()))
            
        if hasattr(self, 'adapter') and self.adapter is not None:
            logger.info("Adding Adapter parameters")
            params.extend(list(self.adapter.parameters()))

        # Ajouter les area_parameters si la loss l'exige
        if 'learnable-area' in self.loss:
            logger.info("Adding learnable area parameters")
            # self.area_parameters est déjà nn.Parameter
            params.append(self.area_parameters)

        if self.student_train and self.temperature == 'seach':
            params.append(self.temperature_value)
            
        if self.student_train and self.alpha == 'seach':
            params.append(self.alpha_value)
        
        # Add RelationMLP parameters for RelationMLP distillation mode
        if self.student_train and self.distillation_training_mode == 'RelationMLP':
            logger.info("Adding RelationMLP parameters to optimizer")
            params.extend(self.relation_mlp.parameters())
            
        return params
    
    def get_optimizer(self, criterion,):
        parameters = self.get_learnable_parameters(criterion)
        optimizer = optim.Adam(parameters, lr=self.lr)
        #optimizer = optim.SGD(parameters, lr=self.lr, momentum=0.9)
        return optimizer
    
    
    def shapley_additive_explanation(self, df, outname, dir_output, mode='beeswarm', figsize=(15, 25), samples=None, samples_name=None, horizon_shap=0, plot=True):
        """
        Visualisation des valeurs SHAP pour expliquer les prédictions.
        :param df_set: DataFrame des caractéristiques d'entrée.
        :param outname: Nom de sortie pour le fichier d'image.
        :param dir_output: Répertoire où enregistrer les résultats.
        :param mode: Mode de visualisation ('bar' ou 'beeswarm').
        :param figsize: Taille de la figure.
        :param samples: Échantillons spécifiques à analyser.
        :param samples_name: Noms des échantillons à afficher.
        """
        # Utiliser un backend non-interactif pour éviter les erreurs Qt
        import matplotlib
        matplotlib.use('Agg')
        
        if hasattr(self, 'use_temporal_as_edges'):
            use_temporal_as_edges = self.use_temporal_as_edges
        else:
            use_temporal_as_edges = None

        Xst, e = get_numpy_data(self.graph, df, self.features_name, use_temporal_as_edges, self.ks, self.horizon)
        Xst = torch.Tensor(Xst).to(self.device)
        B, F, T = Xst.shape
        
        Xst_horizon = self.compute_inputs(Xst,  -1 - (self.horizon - horizon_shap), "current" if horizon_shap == 0 else "futur")
        Xst_flat = Xst.reshape((B, F*T))
        Xst_horizon_flat = Xst_horizon[:, :, -1]
        
        df_features = []

        if self.under_sampling == 'search':
            y = self.df_train[self.target_name].values
            nb = int(self.metrics['best_tp'] * len(y[y == 0]))
            df_combined = self.split_dataset(self.df_train, nb, reset=False)
            self.df_train['weight'] = 0

            # Mettre à jour df_train pour l'entraînement
            self.df_train.loc[df_combined.index, 'weight'] = 1
            
        background_data_train = self.df_train
                        
        background_data_train, e = get_numpy_data(self.graph, background_data_train, self.features_name, use_temporal_as_edges, self.ks, self.horizon)
        background_data_train = torch.Tensor(background_data_train).to(self.device)
        B_train, F_train, T_train = background_data_train.shape
        
        background_data_train = background_data_train.reshape((B_train, F_train*T_train))
        
        # SHAP DeepExplainer avec wrapper du modèle
        self.model.eval()
        self.explainer = shap.DeepExplainer(WrapperModel(self, F, T, e, horizon_shap, return_logits=True).to(self.device), background_data_train)
        self.model.eval()
        shap_values = self.explainer.shap_values(Xst_flat, check_additivity=False)
        
        n_classes = self.out_channels
        
        # Vérifier la forme des valeurs SHAP pour débogage
        print(f"SHAP values type: {type(shap_values)}")
        print(f"SHAP values shape (before processing): {np.asarray(shap_values).shape if isinstance(shap_values, (list, np.ndarray)) else 'N/A'}")

        # Vérifier si la sortie SHAP est multi-classes
        if n_classes == 1:
            shap_values = shap_values[:, :, np.newaxis]
                
        shap_values = np.moveaxis(shap_values, 0, 2)
        
        shap_values = np.asarray(shap_values)
        
        # Vérification de dimensions pour éviter les erreurs de reshape
        expected_shape = (B, F, T, n_classes)
        try:
            shap_values = np.reshape(shap_values, expected_shape)
        except ValueError as e:
            print(f"Erreur de reshape: forme actuelle {shap_values.shape}, forme attendue {expected_shape}")
            raise e
        
        shap_values = shap_values[:, :,  -1 - (self.horizon - horizon_shap), :]
        
        # Pour chaque classe, calculer et sauvegarder les résultats SHAP
        for class_idx in range(n_classes):
            # Calcul des valeurs SHAP moyennes et écarts-types
            shap_mean_abs = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
            shap_std_abs = np.std(np.abs(shap_values[:, :, class_idx]), axis=0)
        
            df_shap = pd.DataFrame({
                "mean_abs_shap": shap_mean_abs,
                "stdev_abs_shap": shap_std_abs,
                "name": self.features_name
            }).sort_values("mean_abs_shap", ascending=False)
            
            print(df_shap.sort_values("mean_abs_shap").head())

            df_shap['class'] = class_idx
            df_features.append(df_shap)

            if plot:
                check_and_create_path(dir_output)
                # Visualisation globale (summary_plot) pour chaque classe
                plt.figure(figsize=figsize)
                if mode == 'bar':
                    shap.summary_plot(
                        shap_values[:, :, class_idx],
                        features=Xst_horizon_flat,
                        feature_names=self.features_name,
                        plot_type='bar',
                        show=False
                    )
                elif mode == 'beeswarm':
                    # Générer le graphique SHAP pour une classe spécifique (class_idx)
                    shap.summary_plot(
                        shap_values[:, :, class_idx],
                        features=Xst_horizon_flat,
                        feature_names=self.features_name,
                        show=False,
                        plot_type="dot"
                    )

            print(f"Sauvegarde: {dir_output / f'{outname}_class_{class_idx}_shapley.png'}")
            if plot:
                plt.savefig(dir_output / f"{outname}_class_{class_idx}_shapley.png", bbox_inches='tight', dpi=100)
                plt.close('all')

            # Visualisations spécifiques aux échantillons (force_plot)
            if samples is not None and samples_name is not None and plot:

                for i, sample in enumerate(samples):
                    plt.figure(figsize=figsize)
                    shap.force_plot(
                        self.explainer.expected_value[class_idx],
                        shap_values[sample, :, class_idx],
                        features=df.iloc[sample].values,
                        feature_names=self.features_name,
                        matplotlib=True,
                        show=False
                    )

                    plt.savefig(
                        dir_output / f"{outname}_class_{class_idx}_{samples_name[i]}_shapley.png",
                        bbox_inches='tight'
                    )
                    plt.close('all')
                    
        df_features = pd.concat(df_features)
        save_object(df_features, 'features_importance.pkl', dir_output)
        
        # Sauvegarder les valeurs SHAP ET l'explainer pour réutilisation ultérieure
        shap_data = {
            'shap_values': shap_values,  # Shape: (B, F, n_classes)
            'expected_values': self.explainer.expected_value,
            'feature_names': self.features_name,
            'n_classes': n_classes,
            'B': B,
            'F': F,
            'T': T,
            'Xst_flat': Xst_flat.cpu().numpy() if torch.is_tensor(Xst_flat) else Xst_flat,
            'horizon_shap': horizon_shap,
            'e': e  # Edges pour reconstruire le WrapperModel si nécessaire
        }
        save_object(shap_data, f'{outname}_shap_values.pkl', dir_output)
        
        # Sauvegarder l'explainer séparément (peut être volumineux)
        explainer_data = {
            'explainer': self.explainer,
            'wrapper_model': WrapperModel(self, F, T, e, horizon_shap),
            'F': F,
            'T': T,
            'e': e,
            'horizon_shap': horizon_shap
        }
        #save_object(explainer_data, f'{outname}_shap_explainer.pkl', dir_output)
        save_object(self.explainer, f'{outname}_shap_explainer.pkl', dir_output)
        print(f"SHAP values sauvegardées dans: {dir_output / f'{outname}_shap_values.pkl'}")
        print(f"SHAP explainer sauvegardé dans: {dir_output / f'{outname}_shap_explainer.pkl'}")

    def shapley_additive_explanation_sample(self, df_sample, explainer, outname, dir_output, 
                                           shap_data_file=None, sample_name=None, 
                                           figsize=(15, 10), generate_force_plot=True, plot=True,
                                           horizon=0):
        """
        Calcule et visualise les valeurs SHAP pour un échantillon spécifique.
        
        :param df_sample: DataFrame contenant un seul échantillon (1 ligne) ou index de l'échantillon dans df_test
        :param outname: Nom de sortie pour les fichiers
        :param dir_output: Répertoire de sortie
        :param shap_data_file: Chemin vers le fichier de SHAP values sauvegardé (optionnel)
        :param sample_name: Nom de l'échantillon pour les fichiers de sortie
        :param figsize: Taille des figures
        :param generate_force_plot: Si True, génère les force plots
        :param plot: Si True, génère les visualisations
        :return: Dictionary contenant les SHAP values pour cet échantillon
        """
        from pathlib import Path
        import pandas as pd
        
        # Priorité 1: Vérifier si self.explainer existe (explainer en mémoire)
        if hasattr(self, 'explainer') and self.explainer is not None:
            print(f"Utilisation de self.explainer (en mémoire)")
            
            # Préparer les données pour cet échantillon
            if hasattr(self, 'use_temporal_as_edges'):
                use_temporal_as_edges = self.use_temporal_as_edges
            else:
                use_temporal_as_edges = None
            
            Xst_sample, e = get_numpy_data(self.graph, df_sample, self.features_name, use_temporal_as_edges, self.ks, self.horizon)
            Xst_sample = torch.Tensor(Xst_sample).to(self.device)
            B, F, T = Xst_sample.shape
            Xst_sample_flat = Xst_sample.reshape((B, F*T))
            
            # Calculer les SHAP values pour cet échantillon
            
            # Activer le mode logits si le modèle est un WrapperModel
            sample_shap_values_raw = explainer.shap_values(Xst_sample_flat, check_additivity=False)
            
            n_classes = self.out_channels
            
            # Reformater les SHAP values
            if n_classes == 1:
                sample_shap_values_raw = sample_shap_values_raw[:, :, np.newaxis]
            
            sample_shap_values_raw = np.asarray(sample_shap_values_raw)
            
            # Reshape selon le format attendu
            expected_shape = (n_classes, B, F, T)
            sample_shap_values_raw = np.reshape(sample_shap_values_raw, expected_shape)
            
            # Extraire le dernier pas de temps (utiliser horizon 0 par défaut si non spécifié)
            sample_shap_values_raw = sample_shap_values_raw[:, :, :, -1 - (self.horizon - horizon)]
            sample_shap_values_raw = np.moveaxis(sample_shap_values_raw, 0, 2)
            
            # Extraire pour cet échantillon
            sample_shap_values = sample_shap_values_raw[0, :, :]  # Shape: (F, n_classes)
            sample_features = Xst_sample[:, :,  -1 - (self.horizon - horizon)].cpu().numpy()
            expected_values = self.explainer.expected_value
            feature_names = self.features_name
        else:
            print(f"Aucune SHAP value ni explainer pré-calculé trouvé.")
            raise FileNotFoundError(
                f"Impossible de trouver les fichiers SHAP nécessaires:\n"
                f"Veuillez d'abord exécuter shapley_additive_explanation() pour calculer et sauvegarder les valeurs SHAP."
            )
        
        # Générer les visualisations pour chaque classe
        results = {
            'shap_values': sample_shap_values,
            'features': sample_features,
            'feature_names': feature_names,
            'plots_generated': []
        }
        
        if plot:
            for class_idx in range(n_classes):
                # 1. Bar plot des valeurs SHAP pour cet échantillon
                plt.figure(figsize=figsize)
                
                # Créer un DataFrame pour faciliter la visualisation
                # Flatten sample_features to 1D if needed (it may have shape (1, F) or (F,))
                sample_features_flat = sample_features.flatten() if sample_features.ndim > 1 else sample_features
                
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': sample_shap_values[:, class_idx],
                    'feature_value': sample_features_flat[:len(feature_names)]
                })
                shap_df = shap_df.reindex(shap_df['shap_value'].abs().sort_values(ascending=False).index)
                
                # Limiter aux 10 features les plus importantes pour le bar plot
                shap_df_top10 = shap_df.head(10)
                
                # Bar plot
                colors = ['red' if x < 0 else 'blue' for x in shap_df_top10['shap_value']]
                plt.barh(range(len(shap_df_top10)), shap_df_top10['shap_value'], color=colors)
                plt.yticks(range(len(shap_df_top10)), shap_df_top10['feature'])
                plt.xlabel('SHAP value')
                plt.title(f'SHAP Values (Top 10) - {sample_name} - Class {class_idx}')
                plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.tight_layout()
                
                bar_plot_path = dir_output / f"{outname}_{sample_name}_class_{class_idx}_shap_bar.png"
                plt.savefig(bar_plot_path, bbox_inches='tight', dpi=100)
                plt.close('all')
                results['plots_generated'].append(str(bar_plot_path))
                print(f"Sauvegardé: {bar_plot_path}")
                # 2. Waterfall plot (si SHAP le supporte)
                try:
                    plt.figure(figsize=figsize)
                    shap.plots._waterfall.waterfall_legacy(
                        expected_values[class_idx] if isinstance(expected_values, (list, np.ndarray)) else expected_values,
                        sample_shap_values[:, class_idx],
                        feature_names=feature_names,
                        max_display=20,
                        show=False
                    )
                    waterfall_path = dir_output / f"{outname}_{sample_name}_class_{class_idx}_shap_waterfall.png"
                    plt.savefig(waterfall_path, bbox_inches='tight', dpi=100)
                    plt.close('all')
                    results['plots_generated'].append(str(waterfall_path))
                    print(f"Sauvegardé: {waterfall_path}")
                except Exception as e:
                    print(f"Impossible de générer le waterfall plot: {e}")
                
                # 3. Force plot (optionnel)
                if generate_force_plot:
                    try:
                        plt.figure(figsize=figsize)
                        # Arrondir les valeurs SHAP à 3 décimales pour meilleure visibilité
                        sample_shap_values_rounded = np.round(sample_shap_values[:, class_idx], 5)
                        expected_values = np.round(expected_values, 3)
                        shap.force_plot(
                            expected_values[class_idx] if isinstance(expected_values, (list, np.ndarray)) else expected_values,
                            sample_shap_values_rounded,
                            features=sample_features_flat[:len(feature_names)],
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False
                        )
                        force_plot_path = dir_output / f"{outname}_{sample_name}_class_{class_idx}_shap_force.png"
                        plt.savefig(force_plot_path, bbox_inches='tight', dpi=100)
                        plt.close('all')
                        results['plots_generated'].append(str(force_plot_path))
                        print(f"Sauvegardé: {force_plot_path}")
                    except Exception as e:
                        print(f"Impossible de générer le force plot: {e}")
            
            # Sauvegarder les résultats pour cet échantillon
            save_object(results, f'{outname}_{sample_name}_shap_results.pkl', dir_output)
            print(f"\nRésultats sauvegardés: {dir_output / f'{outname}_{sample_name}_shap_results.pkl'}")
            print(f"Nombre de visualisations générées: {len(results['plots_generated'])}")
        
        return results

############################################ Split training ##############################################################

class SplitTraining(Training):
    def __init__(self, federated_cluster, cut_layer_name, input_server_model, model_name,
                 nbfeatures, batch_size, lr, delta_lr, patience_cnt_lr, target_name, task_type, out_channels,
                 dir_log, features_name, ks, loss, name, device, under_sampling, over_sampling, n_run,
                 horizon=0, post_process=None):

        super().__init__(model_name, nbfeatures, batch_size, lr, delta_lr, patience_cnt_lr, target_name, task_type, features_name, ks,
                         out_channels, dir_log, loss=loss, name=name, device=device, under_sampling=under_sampling,
                         over_sampling=over_sampling, n_run=n_run, horizon=horizon, post_process=post_process)

        self.federated_cluster = federated_cluster
        self.cut_layer_name = cut_layer_name
        self.input_server_model = input_server_model

        self.horizon = horizon

    def create_client_model_upto_cut_layer(self, model):
        """
        Crée un sous-modèle client contenant les couches jusqu'à (non inclus) la couche de découpe.
        """
        from torch import nn

        layers = []
        for name, layer in model.named_children():
            if name == self.cut_layer_name:
                break
            layers.append(layer)

        client_model = nn.Sequential(*layers)
        return client_model

    def create_server_model_from_cut_layer(self, model):
        """
        Crée un sous-modèle serveur contenant les couches à partir de la couche de découpe (incluse).
        """
        from torch import nn

        start_adding = False
        layers = []

        for name, layer in model.named_children():
            if name == self.cut_layer_name:
                start_adding = True
            if start_adding:
                layers.append(layer)
        
        server_model = nn.Sequential(*layers)
        return server_model

    def initialize_clients(self, base_model, clusters, learning_rate=1e-3):
        client_models = {}
        client_optimizers = {}

        for cluster in clusters:
            model = deepcopy(base_model)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            client_models[cluster] = model
            client_optimizers[cluster] = optimizer
        
        return client_models, client_optimizers
    
    def initialize_server(self, model, learning_rate=1e-3):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer

    def prepare_batch_data(self, df_train, graph, clusters, batch_size):
        batch_data = []

        for cluster in clusters:
            df_c = df_train[df_train[self.federated_cluster] == cluster]

            train_dataset = create_train_dataset(graph, df_c,
                                                 self.features_name,
                                                 self.target_name,
                                                 None, self.device, 
                                                 self.ks, False, '')

            loader = DataLoader(train_dataset, train_dataset.__len__(),
                                False, worker_init_fn=seed_worker,
                                generator=g)

            batch_data.append((cluster, loader))
            
        return batch_data
    
    def clients_forward(self, batch_data, client_models):
        activations = []
        inputs_for_backward = {}
        labels_list = []

        for cluster, batch in batch_data:
            model = client_models[cluster]
            model.train()
            for data in batch:
                inputs, labels, edges = data
                graphs = None

                if inputs.shape[0] == 1:
                    return 0

                band = -1
                
                try:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, model.is_graph_or_node, graphs)
                except Exception as e:
                    target, weights = self.compute_weights_and_target(labels, band, ids_columns, False, graphs)
                
                if self.loss not in ['kldivloss']: # works on probability
                    target = target.long()
                
                output = model(inputs)

                output.retain_grad()

                activations.append(output)
                inputs_for_backward[cluster] = (inputs, output)
                labels_list.append(labels)

        return activations, inputs_for_backward, labels
    
    def server_forward_backward(self, server_model, server_optimizer, activations, labels, criterion):
        server_model.train()
        server_optimizer.zero_grad()

        concat = torch.cat(activations, dim=1)
        print(f'concat : {concat.shape}')
        output = server_model(concat)
        loss = criterion(output, labels)
        loss.backward()

        server_optimizer.step()
        return loss.item(), concat.grad

    def clients_backward_update(self, inputs_for_backward, grad_concat, client_models, client_optimizers):
        split_sizes = [out.shape[1] for _, out in inputs_for_backward.values()]
        grads = torch.split(grad_concat, split_sizes, dim=1)

        for (cluster, (X_batch, out)), grad in zip(inputs_for_backward.items(), grads):
            optimizer = client_optimizers[cluster]
            model = client_models[cluster]
            optimizer.zero_grad()
            out.backward(grad)
            optimizer.step()

    def train_split(self, df_train, df_val, df_test, graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=None):
        
        import torch
        from torch import nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_loader = create_test_loader(graph, df_test,
                            self.features_name,
                            self.device,
                            False,
                            self.target_name,
                            self.ks,
                            self.horizon,
                            False,
                            '')

        clusters = df_train[self.federated_cluster].unique()
        batch_size = self.batch_size
        
        lr = self.lr
        patience = PATIENCE_CNT

        model, _ = make_model(f'{self.model_name}CutClient', len(self.features_name), len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)

        client_models, client_optimizers = self.initialize_clients(model, clusters, lr)
        model, _ = make_model(f'{self.model_name}CutServer', self.input_server_model, len(self.features_name),
                                graph, dropout, activation,
                                self.ks,
                                out_channels=self.out_channels,
                                task_type=self.task_type,
                                device=device, num_lstm_layers=num_lstm_layers,
                                custom_model_params=custom_model_params)
        
        server_model, server_optimizer = self.initialize_server(model, lr)

        criterion = self.get_loss(self.loss)

        best_loss = float('inf')
        best_server_state = None
        patience_counter = 0
        
        batch_data = self.prepare_batch_data(df_train, graph, clusters, batch_size)
        val_batch_data = self.prepare_batch_data(df_val, graph, clusters, batch_size)
        
        val_num_batches = len(val_batch_data)
        num_batch = len(batch_data)

        for epoch in range(epochs):
            if verbose:
                logger.info(f"\nEpoch {epoch+1}/{epochs}")

            epoch_loss = 0

            activations, inputs_for_backward, labels = self.clients_forward(batch_data, client_models)
            loss_value, grad_concat = self.server_forward_backward(server_model, server_optimizer, activations, labels, criterion)
            self.clients_backward_update(inputs_for_backward, grad_concat, client_models, client_optimizers)
            epoch_loss += loss_value

            avg_loss = epoch_loss / num_batch

            # Compute validation loss for early stopping
            val_loss_total = 0
            activations, _, labels = self.clients_forward(val_batch_data, client_models)
            server_model.eval()
            with torch.no_grad():
                concat = torch.cat(activations, dim=1)
                output = server_model(concat)
                vloss = criterion(output, labels)
            val_loss_total += vloss.item()
            val_loss = val_loss_total / val_num_batches

            if epoch % CHECKPOINT and verbose:
                logger.info(f"Avg Epoch Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_server_state = deepcopy(server_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and verbose:
                    logger.info("Early stopping triggered.")
                    break

        if best_server_state is not None:
            server_model.load_state_dict(best_server_state)

        self.client_models = client_models
        self.model = server_model
        self.is_fitted_ = True

        test_output, y = self._predict_test_loader(self.test_loader)
        test_output = test_output.detach().cpu().numpy()

        y = y.detach().cpu().numpy()

        under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
        over_prediction_score_value = over_prediction_score(y[:, -1], test_output)

        iou = iou_score(y[:, -1], test_output)
        f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int), zero_division=0)
        iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

        logger.info(f'Test -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou}, f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

        test_output, y = self._predict_test_loader(self.val_loader)
        test_output = test_output.detach().cpu().numpy()
        
        y = y.detach().cpu().numpy()

        under_prediction_score_value = under_prediction_score(y[:, -1], test_output)
        over_prediction_score_value = over_prediction_score(y[:, -1], test_output)
        
        iou = iou_score(y[:, -1], test_output)
        f1 = f1_score((test_output > 0).astype(int), (y[:, -1] > 0).astype(int))
        iou_area, f1_area = self.compute_area_score(test_output, y[:, -1], y[:, graph_id_index])

        logger.info(f'Val -> Under achieved : {under_prediction_score_value}, Over achived {over_prediction_score_value}, IoU {iou} f1 {f1}, IoU_area {iou_area}, f1_area {f1_area}')

        plt.figure(figsize=(15,5))
        plt.plot(y[y[:, departement_index] == 13, -1])
        plt.plot(test_output[y[:, departement_index] == 13])
        plt.savefig(self.dir_log / 'test.png')
        plt.close('all')

        self.update_weight(server_model.state_dict())

    def _predict_test_loader(self, X: DataLoader, prediction_type='Class', output_pdf="test", proba=False, calibrate=False) -> torch.tensor:

        """Generate predictions using the split learning setup."""

        try:
            if self.training_mode == 'normal':
                return super()._predict_test_loader(X, prediction_type=prediction_type, output_pdf=output_pdf, calibrate=calibrate)
        except:
                return super()._predict_test_loader(X, prediction_type=prediction_type, output_pdf=output_pdf, calibrate=calibrate)

        if not hasattr(self, "server_model") or not hasattr(self, "client_models"):
            raise ValueError("Model is not fitted. Please train the model before predicting.")

        self.server_model.eval()
        for model in self.client_models.values():
            model.eval()

        preds = []
        ys = []
        activations = []

        with torch.no_grad():
            for (cluster, data) in X:
                model = client = self.client_models.get(cluster)
                inputs, labels, edges = data
                labels = labels.to(self.device)
                labels_last = labels[:, :, -1]

                activation = client(inputs)
                activations.append(activation)

                ys.append(labels_last)

            output = self.model(activations)
            if output.shape[1] > 1 and not proba:
                output = torch.argmax(output, dim=1)

                preds.append(output.squeeze(0))

        pred_tensor = torch.stack(preds, 0)
        y_tensor = torch.stack(ys, 0)

        if self.target_name in ["binary", "risk", "nbsinister"]:
            pred_tensor = torch.round(pred_tensor, decimals=1)

        return pred_tensor, y_tensor

    def predict(self, df, graph=None, return_y=False, prediction_type="Class"):
        
        try:
            if self.training_mode == 'normal':
                return super().predict(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
        except:
                return super().predict(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
        
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = self.prepare_batch_data(df, graph, df[self.federated_cluster].unique(), 1)

        pred_tensor, y_tensor = self._predict_test_loader(loader, output_pdf="test")

        if return_y:
            return pred_tensor, y_tensor

        return pred_tensor

    def predict_proba(self, df, graph=None, return_y=False, prediction_type='Proba'):
        try:
            if self.training_mode == 'normal':
                return super().predict_proba(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
        except:
                return super().predict_proba(df, graph=graph, return_y=return_y, prediction_type=prediction_type)
        
        if graph is None:
            graph = self.graph

        if self.target_name not in list(df.columns):
            df[self.target_name] = 0

        loader = create_test_loader(
            graph,
            df,
            self.features_name,
            self.device,
            None,
            self.target_name,
            self.ks,
            self.horizon
        )

        pred_tensor, y_tensor = self._predict_test_loader(loader, True, output_pdf="test")

        if return_y:
            pred, y = self.filtering_pred(df, pred_tensor, y_tensor, graph, return_y=True)
            return pred, y

        pred = self.filtering_pred(df, pred_tensor, y_tensor, graph, return_y=False)
        return pred

    def search_samples_proportion_per_cluster(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT):
        """Search optimal zero sample limits for each cluster.

        Parameters
        ----------
        graph : "Any"
            Graph used for training.
        df_train, df_val, df_test : pandas.DataFrame
            Datasets containing a ``self.federated_cluster`` column.
        args : dict, optional
            Additional arguments forwarded to :func:`train_split`.

        Returns
        -------
        dict
            Mapping cluster id to chosen zero count.
        float
            IoU score obtained with the best combination.
        """

        check_and_create_path(self.dir_log)
        self.metrics = read_object('metrics_cluster.pkl', self.dir_log) or {}
        scores = self.metrics.get('scores', {})

        clusters = df_train[self.federated_cluster].unique()
        best_score = self.metrics.get('best_score', -float("inf"))
        best_combination = self.metrics.get('best_combination')
        best_state = None
        sample_limits = np.arange(0.05, 1.0, 0.05)

        patience = 50
        i = 0
        for numb_combo, combo in enumerate(itertools.product(sample_limits, repeat=len(clusters))):
            logger.info(f'################ {numb_combo} ##################')
            metrics_combo = scores.get(combo, {
                'f1': [], 'iou': [], 'iou_val': [], 'prec': [],
                'recall': [], 'normalized_iou': [], 'normalized_f1': [],
                'under_prediction': [], 'over_prediction': []
            })

            if len(metrics_combo['iou']) >= self.n_run:
                iou_mean = float(np.mean(metrics_combo['iou_val']))
                if iou_mean > best_score:
                    best_score = iou_mean
                    best_combination = dict(zip(clusters, combo))
                continue
            
            for run in range(len(metrics_combo['iou']), self.n_run):
                df_parts = []
                for cluster, tp in zip(clusters, combo):
                    #logger.info(f'Config : {cluster}, {tp}')
                    df_cluster = df_train[df_train[self.federated_cluster] == cluster]
                    nb = int(len(df_cluster[df_cluster[self.target_name] == 0]) * tp)
                    sampled = self.split_dataset(df_cluster, nb, reset=False)
                    df_parts.append(sampled)

                df_train_split = pd.concat(df_parts).reset_index(drop=True)

                model_copy = deepcopy(self)
                model_copy.training_mode = 'normal'
                model_copy.under_sampling = 'full'
                model_copy.train_split(df_train_split, df_val, df_test, graph, epochs=epochs, PATIENCE_CNT=PATIENCE_CNT, CHECKPOINT=CHECKPOINT, verbose=False)

                pred_val, y_val = model_copy._predict_test_loader(model_copy.val_loader, output_pdf="val")
                y_val_np = y_val.detach().cpu().numpy()[:, -1]
                pred_val_np = pred_val.detach().cpu().numpy()
                metrics_val = evaluate_metrics(y_val_np, pred_val_np, zones=y_val_np[:, graph_id_index], dates=y_val_np[:, date_index])
                metrics_combo['iou_val'].append(metrics_val['iou'])

                pred_test, y_test = model_copy._predict_test_loader(model_copy.test_loader, output_pdf="test")

                y_test_np = y_test.detach().cpu().numpy()[:, -1]
                pred_test_np = pred_test.detach().cpu().numpy()
                metrics_test = evaluate_metrics(y_test_np, pred_test_np, zones=y_test_np[:, graph_id_index], dates=y_test_np[:, date_index])

                metrics_combo['iou'].append(metrics_test['iou'])
                metrics_combo['f1'].append(metrics_test['f1'])
                metrics_combo['recall'].append(metrics_test['recall'])
                metrics_combo['prec'].append(metrics_test['prec'])
                metrics_combo['normalized_iou'].append(metrics_test['normalized_iou'])
                metrics_combo['normalized_f1'].append(metrics_test['normalized_f1'])
                metrics_combo['under_prediction'].append(under_prediction_score(y_test_np, pred_test_np))
                metrics_combo['over_prediction'].append(over_prediction_score(y_test_np, pred_test_np))

                scores[combo] = metrics_combo
                self.metrics['scores'] = scores
                save_object(self.metrics, 'metrics_cluster.pkl', self.dir_log)

            if self.n_run == 1:
                metrics_combo['var_f1'] = 0
                metrics_combo['IC_f1'] = (0, 0)
                metrics_combo['var_iou'] = 0
                metrics_combo['IC_iou'] = (0, 0)
                metrics_combo['var_normalized_f1'] = 0
                metrics_combo['IC_normalized_f1'] = (0, 0)
                metrics_combo['var_normalized_iou'] = 0
                metrics_combo['IC_normalized_iou'] = (0, 0)
            else:
                metrics_combo['var_f1'] = np.var(metrics_combo['f1'])
                metrics_combo['IC_f1'] = calculate_ic95(metrics_combo['f1'])
                metrics_combo['var_iou'] = np.var(metrics_combo['iou'])
                metrics_combo['IC_iou'] = calculate_ic95(metrics_combo['iou'])
                metrics_combo['var_normalized_f1'] = np.var(metrics_combo['normalized_f1'])
                metrics_combo['IC_normalized_f1'] = calculate_ic95(metrics_combo['normalized_f1'])
                metrics_combo['var_normalized_iou'] = np.var(metrics_combo['normalized_iou'])
                metrics_combo['IC_normalized_iou'] = calculate_ic95(metrics_combo['normalized_iou'])

            iou_mean = float(np.mean(metrics_combo['iou_val']))
            if iou_mean > best_score:
                best_score = iou_mean
                best_combination = dict(zip(clusters, combo))
                best_state = deepcopy(model_copy.server_model.state_dict())
                i = 0
            else:
                i += 1
                if i == patience:
                    break

            scores[combo] = metrics_combo
            self.metrics['scores'] = scores
            self.metrics['best_score'] = best_score
            self.metrics['best_combination'] = best_combination
            save_object(self.metrics, 'metrics_cluster.pkl', self.dir_log)
            
        if best_state is not None:
            self.server_model.load_state_dict(best_state)

        self.metrics['scores'] = scores
        self.metrics['best_score'] = best_score
        self.metrics['best_combination'] = best_combination
        self.metrics['run'] = self.n_run
        save_object(self.metrics, 'metrics_cluster.pkl', self.dir_log)
        
        return best_combination

###################################################################### DUAL TRAINING ####################################################################################

class DualTraining:
    """Manage joint optimisation of two ``Training`` instances.

    The class orchestrates two sub-models: ``occ_model`` operates on the
    binarised dataset while ``num_model`` is restricted to samples with a
    positive label. Losses and parameters of both sub-models are combined so
    that a single optimisation step updates them simultaneously."""

    def __init__(self, target_name, occ_model: Training, num_model: Training, name, task_type: str, n_run : int = 1, horizon=0):
        self.occ_model = occ_model
        self.num_model = num_model
        self.name = name
        self.task_type = task_type
        self.n_run = n_run
        self.target_name = target_name
        self.horizon = 0

    def train(
        self,
        graph,
        PATIENCE_CNT,
        CHECKPOINT,
        epochs,
        verbose: bool = True,
        custom_model_params=None,
        new_model: bool = True,
    ):
        self.num_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)
        #self.occ_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose, custom_model_params, new_model)

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def create_test_loader(self, graph, df):
        """Create test loaders for both sub-models.

        The occurence model consumes the full dataframe while the numeric
        model will later be applied only on samples predicted positive. We
        therefore keep a reference to ``graph`` and ``df`` so that the
        numeric loader can be rebuilt after filtering.
        """

        # store for later use in ``_predict_test_loader``
        self._test_graph = graph
        self._test_df = df.reset_index(drop=True)

        # loader for the occurence model (covers all samples)
        self.test_loader = self.occ_model.create_test_loader(graph, df)
        self.occ_model.test_loader = self.test_loader
        return self.test_loader
    
    def create_train_val_test_loader(self, graph, dfs_train, dfs_val, dfs_test, epochs, PATIENCE_CNT, CHECKPOINT, custom_model_params, features_importance, use_log):
        train_dataset, train_pos = dfs_train
        val_dataset, val_pos = dfs_val
        test_dataset, test_pos = dfs_test
        
        self.num_model.create_train_val_test_loader(
                graph,
                train_pos,
                val_pos,
                test_pos,
                epochs,
                PATIENCE_CNT,
                CHECKPOINT,
                custom_model_params=custom_model_params,
                features_importance=False,
                use_log=use_log,
            )
        
        self.metrics = {}
        tp = 'occ-based'
        self.num_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, True, custom_model_params=custom_model_params, new_model=True)

        for run in range(self.n_run):
            seed = int(random.random())
            self.occ_model.seed = seed
            self.occ_model.n_run = 1
            self.occ_model.create_train_val_test_loader(
                graph,
                train_dataset,
                val_dataset,
                test_dataset,
                epochs,
                PATIENCE_CNT,
                CHECKPOINT,
                custom_model_params=custom_model_params,
                features_importance=False,
                use_log=use_log,
            )
            self.occ_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, False, custom_model_params=custom_model_params, new_model=True)

            ############################# On set test ##############################
            test_output, y = self._predict_test_loader((self.occ_model.test_loader, self.num_model.test_loader))
            prediction = test_output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            prediction = prediction[:, 0]
            y = y[:, :, 0]
        
            dff = pd.DataFrame(index=np.arange(0, y.shape[0]))
            dff['departement'] = y[:, departement_index]
            dff['date'] = y[:, date_index]
            dff['graph_id'] = y[:, graph_id_index]
            dff[self.target_name] = y[:, -1]
            y = y[:, -1]

            metrics_run = evaluate_metrics(dff[self.target_name], prediction, zones=dff['graph_id'], dates=dff['date'])
            metrics_run = round_floats(metrics_run)
            update_metrics_as_arrays(self, tp, metrics_run, 'test')

        self.metrics[tp] = add_ic95_to_dict(self.metrics[tp], None, "_ic95")
        self.metrics['best_tp'] = tp

    def _predict_test_loader(self, loader=None, prediction_type='Class', output_pdf=None, calibrate=False):
        occ_loader, num_loader = loader
        pred_occ, y_occ = self.occ_model._predict_test_loader(occ_loader, prediction_type, output_pdf, calibrate=calibrate)
        pred_num, y_num = self.num_model._predict_test_loader(num_loader, prediction_type, output_pdf, calibrate=calibrate)
        
        print('Size check ->', y_occ.shape, y_num.shape)
        
        pred_occ = torch.as_tensor(pred_occ, dtype=torch.float32)
        pred_num = torch.as_tensor(pred_num, dtype=torch.float32)
        for H in range(self.horizon + 1):
            
            pred_occ_horizon = pred_occ[:, H]
            pred_num_horizon = pred_num[:, H]
            
            occ_mask = pred_occ_horizon.reshape(-1) > 0
            
            y_occ_positve_samples = y_occ[occ_mask, :, H]
            
            selected_idx_num : List[torch.Tensor] = []
            selected_idx_occ : List[torch.Tensor] = []
            for i in range(y_occ_positve_samples.shape[0]):
                samples = y_occ_positve_samples[i]
                date = samples[date_index]
                graph_id = samples[graph_id_index]
                
                idx = torch.argwhere((y_num[:, date_index, H] == date) & (y_num[:, graph_id_index, H] == graph_id))
                if len(idx) > 0:
                    selected_idx_num += idx
                    
                idx = torch.argwhere((y_occ[:, date_index, H] == date) & (y_occ[:, graph_id_index, H] == graph_id))
                if len(idx) > 0:
                    selected_idx_occ += idx
            
            print('Size idx check ->', len(selected_idx_num), len(selected_idx_occ))
            pred_occ[selected_idx_occ] = pred_num_horizon[selected_idx_num][..., None]
            
        return pred_occ, y_num
    
    def create_test_loader(self, graph, df):
        loader_occ = self.occ_model.create_test_loader(graph, df)
        
        loader_num = self.num_model.create_test_loader(graph, df)

        return (loader_occ, loader_num)
    
    def search_samples_proportion(self, *args, **kwargs):
        """Delegate proportion search to the occurence model.

        This wrapper keeps the signature of :func:`Training.search_samples_proportion`
        for compatibility while relying on the occurence model implementation.
        """

        return self.occ_model.search_samples_proportion(*args, **kwargs)
    
class Distribution2Class:
    def __init__(self, target_name, distrib_model: Training, class_model: Training, name, task_type: str, n_run : int = 1, horizon=0):
        self.distrib_model = distrib_model
        self.class_model = class_model
        self.name = name
        self.task_type = task_type
        self.n_run = n_run
        self.target_name = target_name
        self.horizon = horizon
        
        print(distrib_model.dir_log / f'{distrib_model.name}.pkl')
        if (distrib_model.dir_log / f'{distrib_model.name}.pkl').is_file():
            self.distrib_model = read_object(f'{distrib_model.name}.pkl', distrib_model.dir_log)
            self.train_distrib_model = False
        else:
            self.train_distrib_model = True

    def create_train_val_test_loader(self, graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT,
                                     features_importance=True, custom_model_params=None, use_log=True):
        
        if self.train_distrib_model:
            print('################ Train distribution model ####################')
            self.distrib_model.create_train_val_test_loader(graph, df_train, df_val, df_test, epochs, PATIENCE_CNT, CHECKPOINT,
                                     features_importance=features_importance, custom_model_params=custom_model_params, use_log=use_log)
            
            self.distrib_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, verbose=True, custom_model_params=custom_model_params, new_model=True)

        print('################ Train class model ####################')
        ### Update df_train
        
        loader = self.distrib_model.create_test_loader(graph, df_train)
        output_train, y_train = self.distrib_model._predict_test_loader(loader, prediction_type='RawFormulaVal', output_pdf='train')
        output_val, y_val = self.distrib_model._predict_test_loader(self.distrib_model.val_loader, prediction_type='RawFormulaVal', output_pdf='Val')
        output_test, y_test = self.distrib_model._predict_test_loader(self.distrib_model.test_loader, prediction_type='RawFormulaVal', output_pdf='test')
        
        df_class_train = pd.DataFrame(index=np.arange(0, y_train.shape[0]))
        df_class_val = pd.DataFrame(index=np.arange(0, y_val.shape[0]))
        df_class_test = pd.DataFrame(index=np.arange(0, y_test.shape[0]))
        
        y_train = y_train[:, :, 0]
        y_val = y_val[:, :, 0]
        y_test = y_test[:, :, 0]
        
        output_train = output_train[:, :, 0]
        output_val = output_val[:, :, 0]
        output_test = output_test[:, :, 0]

        columns_y = ids_columns + targets_columns + [f'{self.distrib_model.target_name}']
        
        y_train = y_train.reshape(y_train.shape[0], -1)
        y_val = y_val.reshape(y_val.shape[0], -1)
        y_test = y_test.reshape(y_test.shape[0], -1)
        
        output_train = output_train.reshape(output_train.shape[0], -1)
        output_val = output_val.reshape(output_val.shape[0], -1)
        output_test = output_test.reshape(output_test.shape[0], -1)
                
        df_class_train[columns_y] = y_train
        df_class_val[columns_y] = y_val
        df_class_test[columns_y] = y_test
        
        df_class_train['weight'] = 1
        
        columns_x = [f'fet_{i}' for i in range(output_train.shape[-1])]
        
        df_class_train[columns_x] = output_train
        df_class_val[columns_x] = output_val
        df_class_test[columns_x] = output_test
        
        self.class_model.features_name = columns_x
        
        self.class_model.create_train_val_test_loader(graph, df_class_train, df_class_val, df_class_test, epochs, PATIENCE_CNT, CHECKPOINT, 
                        features_importance=features_importance, custom_model_params=custom_model_params, use_log=use_log)
        
        
        self.metrics = self.class_model.model
        
        self.class_model.graph = graph
        
        self.class_model.train(graph, PATIENCE_CNT, CHECKPOINT, epochs, custom_model_params=custom_model_params)
    
    def _predict_test_loader(self, loader=None, prediction_type='Class', output_pdf=None, calibrate=False):
        pred_distrib, y_distrib = self.distrib_model._predict_test_loader(loader, prediction_type='RawFormulaVal', output_pdf=output_pdf, calibrate=calibrate)

        y_distrib = y_distrib[:, :, 0]
        pred_distrib = pred_distrib[:, :, 0]
        
        y_distrib = y_distrib.reshape(y_distrib.shape[0], -1)
        pred_distrib = pred_distrib.reshape(pred_distrib.shape[0], -1)
        
        df_class = pd.DataFrame(index=np.arange(0, y_distrib.shape[0]))
        
        columns_y = ids_columns + [f'{self.distrib_model.target_name}']
        columns_x = [f'fet_{i}' for i in pred_distrib.shape[-1]]
        
        df_class[columns_y] = y_distrib
        df_class[columns_x] = pred_distrib
        
        loader = self.class_model.create_test_loader(self.class_model.graph, df_class)
        
        res, y = self.class_model._predict_test_loader(loader, prediction_type, output_pdf, calibrate)
        
        return res, y
    
    def create_test_loader(self, graph, df):
        return self.distrib_model.create_test_loader(graph, df)
    
    def score(self, X, y, sample_weight=None):
        pred_distrib, y_distrib = self.distrib_model.predict(X, return_y=True)
        
        y_distrib = y_distrib[:, :, 0]
        pred_distrib = pred_distrib[:, :, 0]
        
        y_distrib = y_distrib.reshape(y_distrib.shape[0], -1)
        pred_distrib = pred_distrib.reshape(pred_distrib.shape[0], -1)
        
        df_class = pd.DataFrame(index=np.arange(0, y_distrib.shape[0]))
        
        columns_y = ids_columns + [f'{self.distrib_model.target_name}']
        columns_x = [f'fet_{i}' for i in pred_distrib.shape[-1]]
        
        df_class[columns_y] = y_distrib
        df_class[columns_x] = pred_distrib
        
        predictions, y = self.class_model.predict(df_class, return_y=True)
        predictions = predictions[:, 0]
        y = y[:, -1, 0]
        
        return self.class_model.score_with_prediction(predictions, y, sample_weight)