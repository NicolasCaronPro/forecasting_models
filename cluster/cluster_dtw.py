import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



def format_data(data: List[pd.DataFrame]) -> np.ndarray:
    # Convert each DataFrame to a 2D numpy array (n_series, n_timepoints)
    format_data = []
    for i in range(len(data)):
        data[i].reset_index(inplace=True)
        format_data.append(data[i].drop(columns='date'))

    return np.array(format_data)


def scale_data(time_series_data: np.ndarray) -> np.ndarray:
    # Assume time_series_data is a 3D array (n_samples, n_timepoints, n_features)
    # Reshape to 2D for scaling
    n_samples, n_timepoints, n_features = time_series_data.shape
    reshaped_data = time_series_data.reshape(n_samples * n_timepoints, n_features)

    # Standard scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(reshaped_data)

    # Reshape back to original 3D shape
    scaled_time_series_data = scaled_data.reshape(n_samples, n_timepoints, n_features)
    return scaled_time_series_data


def output_clusters(names: List[str], labels: np.ndarray) -> None:
    # Output the cluster labels
    clusters = {}
    for i in range(len(names)):
        try:
            clusters[labels[i]].append(names[i])
        except KeyError:
            clusters[labels[i]] = [names[i]]
    for cluster in clusters:
        print("Cluster", cluster)
        print(clusters[cluster])
    return clusters


def plot_pca_3D(distance_matrix, n_clusters: int, names: List[str] = None) -> None:
    # Assuming you have your data and corresponding cluster labels

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_matrix)

    # Step 1: Apply PCA to reduce the data to 3 dimensions
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(distance_matrix)

    # Step 2: Plot in 3D using Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points and color by cluster labels
    sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=cluster_labels, cmap='viridis', s=50)

    # Step 3: Annotate the points with hospital names
    for i, name in enumerate(names):
        ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], name, size=10, zorder=1, color='black')

    # Add labels and title
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.title('3D PCA Plot with Hospital Names')

    # Show the color bar based on clusters
    plt.colorbar(sc)

    plt.show()


def plot_pca(distance_matrix, n_clusters: int, names: List[str] = None) -> None:
   # Step 2: Apply K-Means Clustering (or any other clustering algorithm)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_matrix)

    # Step 3: Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    dtw_pca = pca.fit_transform(distance_matrix)

    # Step 4: Plot the PCA results
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(dtw_pca[:, 0], dtw_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.title('PCA of DTW Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # Add a color bar to show cluster colors
    plt.colorbar(scatter, label='Cluster')

    # Step 5: Add text labels for each point
    for i, name in enumerate(names):
        plt.text(dtw_pca[i, 0] + 0.02, dtw_pca[i, 1] + 0.02, name, fontsize=9, ha='right')  # Adjust offset as needed

    plt.show()


def plot_heatmap(distance_matrix, names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap='viridis', annot=False, xticklabels=names, yticklabels=names)
    plt.title('DTW Distance Heatmap')
    plt.xlabel('Time Series Index')
    plt.ylabel('Time Series Index')
    plt.show()


class Printer():
    def __init__(self, verbose):
        self.verbose = verbose

    def print(self, *args):
        if self.verbose:
            print(*args)


def cluster_dtw(data: List[pd.DataFrame], n_clusters: int = 0, scale=True, verbose=False) -> np.ndarray:

    pr = Printer(verbose)

    time_series_data = format_data(data)
    if scale: time_series_data = scale_data(time_series_data)

    # Range of clusters to try
    range_n_clusters = range(2, len(data))

    # Placeholder for silhouette scores
    silhouette_scores = []

    # Placeholder for labels
    labels = []
    cluster = []

    if not n_clusters:
        # Loop over cluster sizes to find the optimal number
        for n_clusters in range_n_clusters:
            # Apply DTW KMeans clustering
            km_dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
            labels.append(km_dtw.fit_predict(time_series_data))
            
            # Calculate the pairwise DTW distance matrix
            distance_matrix = cdist_dtw(time_series_data)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(distance_matrix, labels[-1], metric="precomputed")
            silhouette_scores.append((n_clusters, silhouette_avg))
            
            pr.print(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")

        # Select the number of clusters with the highest silhouette score
        best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        cluster = labels[best_n_clusters-2]
        pr.print(f"\nOptimal number of clusters: {best_n_clusters} -> {cluster}")
        n_clusters = best_n_clusters
    else:
        # Apply DTW KMeans clustering
        km_dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
        cluster = km_dtw.fit_predict(time_series_data)
        distance_matrix = cdist_dtw(time_series_data)


    

    return cluster, distance_matrix, n_clusters
    