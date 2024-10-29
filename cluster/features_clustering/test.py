# %%
import pandas as pd
import numpy as np
from dtaidistance import dtw
import os
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from __future__ import absolute_import
from __future__ import print_function
from GPUDTW import cuda_dtw, cpu_dtw, dtw_1D_jit2
import numpy
import time

# %%
os.getcwd()

# %%
data = pd.read_csv('../../tests/enc_X_train.csv')

# %%
# Calcul de la matrice de distance DTW


def calculate_dtw_distance_matrix(df):
    # num_series = df.shape[1]
    # dist_matrix = np.zeros((num_series, num_series))
    data_array = df.values.astype(np.float32)
    distances = cuda_dtw(data_array.T, data_array.T)

    return pd.DataFrame(distances, columns=df.columns, index=df.columns)


# %%
dtw_matrix = calculate_dtw_distance_matrix(data)
dtw_matrix

# %%


def distance_to_similarity_gaussian(distance_matrix, sigma=1.0):
    # Transformation en similarité avec une fonction gaussienne
    similarity_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
    # Similitude maximale sur la diagonale
    np.fill_diagonal(similarity_matrix.values, 1)
    return similarity_matrix


# %%
similarity_matrix_gaussian = distance_to_similarity_gaussian(dtw_matrix)
similarity_matrix_gaussian

# %%


def clustering_with_spectral(similarity_matrix, n_clusters=3):
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed')
    labels = spectral.fit_predict(similarity_matrix)
    return labels


# %%
labels_spectral = clustering_with_spectral(similarity_matrix_gaussian)
labels_spectral

# %%


def tsne_visualization(similarity_matrix):
    # Conversion de la matrice de similarité en matrice de distance
    distance_matrix = 1 - similarity_matrix
    tsne = TSNE(metric='precomputed', n_components=2, perplexity=30)
    tsne_result = tsne.fit_transform(distance_matrix)
    return tsne_result


# %%
# Visualisation t-SNE
tsne_result = tsne_visualization(similarity_matrix_gaussian)
# `labels` peut venir du clustering
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels)
plt.title('t-SNE Visualisation based on Similarity Matrix')
plt.show()
