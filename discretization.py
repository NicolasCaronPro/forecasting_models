from GNN.statistical_model import *
from scipy.ndimage import gaussian_filter1d


class KMeansRisk:
    """
    Classe utilisant KMeans pour identifier les classes de risque.
    """
    
    def __init__(self, n_clusters):
        """
        :param n_clusters: Nombre de clusters pour KMeans.
        """
        self.n_clusters = n_clusters
        self.name = f'KMeansRisk_{n_clusters}'
        self.label_map = None

    def fit(self, X, y):
        """
        Ajuste le modèle KMeans aux données.

        :param X: Données à ajuster.
        """
        if np.unique(X).shape[0] < self.n_clusters:
            self.zeros = True
            return np.zeros(X.shape)
        
        self.zeros = False
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.model.fit(np.unique(X).reshape(-1,1))
        #self.model.fit(X)
        centroids = self.model.cluster_centers_

        if centroids.shape[1] > 1:
            magnitudes = np.linalg.norm(centroids, axis=1)
            sorted_indices = np.argsort(magnitudes)
        else:
            centroids = centroids.flatten()
            sorted_indices = np.argsort(centroids)

        self.label_map = {label: i for i, label in enumerate(sorted_indices)}

    def predict(self, X):
        """
        Prédit les classes en utilisant le modèle KMeans.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        if self.zeros:
            return np.zeros_like(X)
        kmeans_labels = self.model.predict(X)
        return np.vectorize(self.label_map.get)(kmeans_labels)
    
class KMeansRiskZerosHandle:
    """
    Classe utilisant KMeans pour identifier les classes de risque.
    """
    
    def __init__(self, n_clusters):
        """
        :param n_clusters: Nombre de clusters pour KMeans.
        """
        self.n_clusters = n_clusters - 1
        self.name = f'KMeansRisk_{n_clusters}'
        self.label_map = None

    def fit(self, X, y):
        """
        Ajuste le modèle KMeans aux données.

        :param X: Données à ajuster.
        """
        X_val = X[X > 0].reshape(-1,1)
        if np.unique(X_val).shape[0] == 0:
            self.model = None
            return
        self.model = KMeans(n_clusters=min(self.n_clusters, np.unique(X_val).shape[0]), random_state=42, n_init=10)
        self.model.fit(X_val)
        centroids = self.model.cluster_centers_
        
        if centroids.shape[1] > 1:
            magnitudes = np.linalg.norm(centroids, axis=1)
            sorted_indices = np.argsort(magnitudes)
        else:
            centroids = centroids.flatten()
            sorted_indices = np.argsort(centroids)

        self.label_map = {label: i + 1 for i, label in enumerate(sorted_indices)} 

    def predict(self, X):
        """
        Prédit les classes en utilisant le modèle KMeans.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        res = np.zeros(X.shape)
        if self.model is None:
            return res.reshape(-1,1)
        X_val = X[X > 0].reshape(-1,1)
        
        if X_val.shape[0] == 0:
            return res.reshape(-1)
        kmeans_labels = self.model.predict(X_val)
        res[X > 0] = np.vectorize(self.label_map.get)(kmeans_labels)
        return res.reshape(-1)

class ThresholdRisk:
    """
    Classe utilisant des seuils définis pour créer les classes de risque.
    """
    def __init__(self, thresholds):
        """
        :param thresholds: Liste de seuils pour définir les classes.
        """
        self.n_clusters = len(thresholds)
        self.thresholds = np.array(thresholds)
        self.name = f'ThresholdRisk_{thresholds}'

    def fit(self, X, y):
        """
        Aucun ajustement requis pour des seuils prédéfinis.
        """
        pass

    def predict(self, X):
        """
        Prédit les classes en fonction des seuils.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        return np.digitize(X.flatten(), self.thresholds, right=True)

class QuantileRisk:
    """
    Classe utilisant des quartiles pour créer les classes de risque,
    avec une classe supplémentaire pour la valeur minimale.
    """
    def __init__(self):
        """
        Initialise sans seuils prédéfinis. Les seuils seront calculés lors de l'ajustement.
        """
        self.n_clusters = None
        self.thresholds = None
        self.min_value = None
        self.name = "QuantileRisk"

    def fit(self, X, y):
        """
        Calcule les seuils basés sur les quartiles des données.

        :param X: Données à partir desquelles calculer les seuils.
        """
        self.min_value = np.min(X)  # Déterminer la valeur minimale
        self.thresholds = np.unique(np.quantile(X[X > self.min_value].flatten(), [0.50, 0.65, 0.85]))

        logger.info(f'Threshold of quantile risk : {self.thresholds}')

        self.n_clusters = 5

    def predict(self, X):
        """
        Prédit les classes en fonction des quartiles calculés et une classe spécifique pour la valeur minimale.

        :param X: Données à classer.
        :return: Classes prédites.
        """
        if self.thresholds is None or self.min_value is None:
            raise ValueError("Les seuils (quartiles) et la valeur minimale n'ont pas été calculés. Appelez `fit` d'abord.")
        
        result = np.empty(X.shape).reshape(-1)

        # Initialiser les classes avec les indices basés sur les quartiles
        result[X.flatten() > self.min_value] = np.digitize(X.flatten()[X.flatten() > self.min_value], self.thresholds, right=True)
        
        result += 1
        # Assigner une classe distincte pour la valeur minimale
        result[X.flatten() == self.min_value] = 0

        return result

class KSRisk:
    def __init__(self, thresholds, score_col='ks_stat', thresh_col='optimal_score', dir_output=Path('./')):
        self.thresholds_ks = thresholds
        self.score_col = score_col
        self.thresh_col = thresh_col
        self.ks_results = None
        self.dir_output = dir_output
        self.name = f'KSRisk_{thresholds}'
        self.n_clusters = len(thresholds)

    def fit(self, y_pred, y_true):
        test_window = pd.DataFrame(index=np.arange(0, y_true.shape[0]))
        test_window['y_pred'] = y_pred
        test_window['y_true'] = y_true 
        self.ks_results = calculate_ks(test_window, 'y_pred', 'y_true', self.thresholds_ks, self.dir_output)
        return self

    def predict(self, y_pred):
        assert self.ks_results is not None
        new_pred = np.full(y_pred.shape[0], fill_value=0.0)
        max_value = self.ks_results[self.ks_results['threshold'] == self.thresholds_ks[0]][self.score_col].values
        for threshold in self.thresholds_ks:
            if self.ks_results[self.ks_results['threshold'] == threshold][self.score_col].values[0] >= max_value:
                new_pred[y_pred >= self.ks_results[self.ks_results['threshold'] == threshold][self.thresh_col].values[0]] = threshold
        return new_pred

class PreprocessorConv:
    def __init__(self, graph, kernel='Specialized', conv_type='laplace+mean', id_col=None, persistence=False,):
        """
        Initialize the PreprocessorConv.

        :param graph: An object containing `sequences_month` data structure.
        :param conv_type: Type of convolution to apply ('laplace', 'mean', 'laplace+mean', 'sum', or 'max').
        :param id_col: List of column names or indices representing IDs. Defaults to None.
        """
        assert conv_type in ['laplace', 'laplace+median', 'mean', 'laplace+mean', 'sum', 'max', 'median', 'gradient', 'gaussian', 'cubic', 'quartic', 'circular'], \
            "conv_type must be 'laplace', 'mean', 'laplace+mean', 'sum', 'gradient' or 'max'."
        self.graph = graph
        self.conv_type = conv_type
        self.id_col = id_col if id_col is not None else ['id']
        self.kernel = kernel
        self.persistence = persistence

    def _get_season_name(self, month):
        """
        Determine the season name based on the month.

        :param month: Integer representing the month (1 to 12).
        :return: Season name as a string ('medium', 'high', or 'low').
        """
        group_month = [
            [2, 3, 4, 5],    # Medium season
            [6, 7, 8, 9],    # High season
            [10, 11, 12, 1]  # Low season
        ]
        names = ['medium', 'high', 'low']
        for i, group in enumerate(group_month):
            if month in group:
                return names[i]
        raise ValueError(f"Month {month} does not belong to any season group.")

    def _laplace_convolution(self, X, kernel_size):
        """
        Apply Laplace convolution using a custom kernel from Astropy.

        :param X: Input array for convolution.
        :param kernel_size: Size of the convolution kernel.
        :return: Convoluted array.
        """
        # Create Laplace kernel
        kernel_daily = np.abs(np.arange(-(kernel_size // 2), kernel_size // 2 + 1))
        kernel_daily += 1
        kernel_daily = 1 / kernel_daily
        kernel_daily = kernel_daily.reshape(-1,1)

        if self.persistence:
            if kernel_size > 1:
                kernel_daily[:kernel_size // 2] = 0
            else:
                return np.zeros(X.shape).reshape(-1)

        if np.unique(kernel_daily)[0] == 0 and np.unique(kernel_daily).shape[0] == 1:
            kernel_daily += 1
            
        # Perform convolution using Astropy
        #print(np.unique(X))
        #print(np.unique(kernel_daily))
        res = convolve_fft(X.reshape(-1), kernel_daily.reshape(-1), normalize_kernel=False, fill_value=0.)

        return res

    # Convolution avec un noyau Gaussien
    def _gaussian_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # S'assurer que kernel_size est impair

        sigma = (kernel_size - 1) / 6  # Relation approx. : kernel_size ≈ 6 * sigma
        #x = np.linspace(-1, 1, kernel_size)
        x = np.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        #plt.plot(kernel, label='Gaussian')
        if self.persistence:
            kernel[:kernel_size // 2] = 0

        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    # Convolution avec un noyau cubique
    def _cubic_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # S'assurer que kernel_size est impair

        x = np.linspace(-1, 1, kernel_size)
        kernel = (1 - np.abs(x))**3
        kernel = np.clip(kernel, 0, None)

        if self.persistence:
            kernel[:kernel_size // 2] = 0

        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    # Convolution avec un noyau quartique
    def _quartic_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # S'assurer que kernel_size est impair

        x = np.linspace(-1, 1, kernel_size)
        kernel = (1 - x**2)**2
        kernel = np.clip(kernel, 0, None)
        #plt.plot(kernel, label='Quartic')
        if self.persistence:
            kernel[:kernel_size // 2] = 0
        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    # Convolution avec un noyau circulaire
    def _circular_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # S'assurer que kernel_size est impair

        x = np.linspace(-1, 1, kernel_size)
        kernel = np.sqrt(1 - x**2)
        kernel = np.clip(kernel, 0, None)
        #plt.plot(kernel, label='Circular')
        if self.persistence:
            kernel[:kernel_size // 2] = 0
        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    def _mean_convolution(self, X, kernel_size):
        """
        :param X: Input array for convolution.
        :param kernel_size: Size
        Apply mean convolution using a custom kernel from Astropy.of the convolution kernel.
        :return: Convoluted array.
        """
        # Create mean kernel
        if kernel_size == 1:
            return np.zeros_like(X).reshape(-1)
        
        kernel_season = np.ones(kernel_size, dtype=float)
        kernel_season /= kernel_size  # Normalize the kernel

        if self.persistence:
            kernel_season[:kernel_size // 2] = 0
 
        #kernel_season[kernel_size // 2] = 0
        res = convolve_fft(X.reshape(-1), kernel_season.reshape(-1), normalize_kernel=False, fill_value=0.).reshape(-1)

        return res
    
    def median_convolution(self, X, kernel_size):
        """
        Apply median convolution using a custom kernel, excluding the center pixel.

        Parameters:
            X (array-like): Input array for convolution.
            kernel_size (int): Size of the convolution kernel (must be odd).

        Returns:
            numpy.ndarray: Convoluted array with median filtering applied.
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")
        
        if kernel_size == 1:
            return np.zeros(X.shape).reshape(-1)

        def custom_median_filter(window):
            # Exclude the center element from the kernel (middle pixel)
            center = len(window) // 2
            if self.persistence:
                window[center:] = np.nan  # Exclude earlier values
            window = np.delete(window, center)  # Remove center element
            return np.nanmedian(window)
        
        # Apply the custom median filter
        return generic_filter(X, custom_median_filter, size=kernel_size).reshape(-1)

    def _gradient_convolution(self, X, kernel_size):
        """
        Applique une convolution de gradient sur l'entrée avec une fenêtre de taille `kernel_size`.

        :param X: Tableau d'entrée (2D: samples x features).
        :param kernel_size: Taille de la fenêtre de convolution.
        :return: Tableau du gradient calculé sur la fenêtre glissante.
        """
        from scipy.ndimage import convolve1d

        # Création d'un noyau de gradient centré
        kernel = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        kernel = kernel / np.sum(np.abs(kernel))  # Normalisation

        return convolve1d(X, kernel, mode='nearest', axis=0).reshape(-1)

    def _sum_convolution(self, X, kernel_size):
        """
        Apply sum operation on the input with a sliding window of size `kernel_size`.

        :param X: Input array for summation (2D: samples x features).
        :param kernel_size: Size of the sliding window.
        :return: Array where each element is the sum of elements in the sliding window.
        """
        from scipy.ndimage import uniform_filter1d

        if self.persistence:
            mask = np.zeros_like(X, dtype=float)
            mask[kernel_size // 2:] = 0
            mask[:kernel_size // 2] = 1  # Persistence applies only to later values
            X = X * mask

        return uniform_filter1d(X, size=kernel_size, mode='constant', origin=0, axis=0) * kernel_size

    def _max_convolution(self, X, kernel_size):
        """
        Apply max operation on the input with a sliding window of size `kernel_size`.

        :param X: Input array for max operation (2D: samples x features).
        :param kernel_size: Size of the sliding window.
        :return: Array where each element is the max of elements in the sliding window.
        """
        from scipy.ndimage import maximum_filter1d

        if self.persistence:
            mask = np.zeros_like(X, dtype=float)
            mask[kernel_size // 2:] = 0
            mask[:kernel_size // 2] = 1
            X = X * mask

        res = maximum_filter1d(X, size=kernel_size, mode='constant', origin=0, axis=0)
        return res

    def apply(self, X, ids):
        """
        Apply the convolution based on the given type to the input data.

        :param X: Input array of shape (n_samples, n_features).
        :param ids: Array of shape (n_samples, len(id_col)) if `id_col` is a list, otherwise (n_samples, 1).
        :return: Processed array with convolutions applied.
        """
        X_processed = np.zeros_like(X, dtype=np.float32).reshape(-1)
        unique_ids = np.unique(ids, axis=0)

        for unique_id in unique_ids:

            if len(self.id_col) > 1 and not isinstance(self.id_col, str):
                mask = (ids[:, 0] == unique_id[0]) & (ids[:, 1] == unique_id[1])
            else:
                mask = ids[:, 0] == unique_id[0]

            if not np.any(mask):
                continue

            if self.kernel == 'Specialized':
                # Handle month_non_encoder
                if 'month_non_encoder' in self.id_col:
                    month_idx = self.id_col.index('month_non_encoder')
                    month = unique_id[month_idx]
                    season_name = self._get_season_name(month)
                    kernel_size = int(self.graph.sequences_month[season_name][unique_id[1]]['mean_size'])
                else:
                    raise ValueError(
                        "Error: 'month_non_encoder' is not specified in id_col. Please include it in id_col to proceed."
                    )
            else:
                kernel_size = (int(self.kernel) * 2 + 1) + 2 # Add 2 to make 0 bound

            if kernel_size == 1:
                X_processed[mask] = 0.0

            # Apply Laplace convolution if specified
            elif self.conv_type == 'laplace':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                X_processed[mask] = laplace_result.reshape(-1)

            # Apply mean convolution if specified
            elif self.conv_type == 'mean':
                mean_result = self._mean_convolution(X[mask], kernel_size)
                #X_processed[mask] = (X[mask].reshape(-1) + mean_result).reshape(-1)
                X_processed[mask] = mean_result.reshape(-1)

            # Apply laplace+mean convolutions if specified
            elif self.conv_type == 'laplace+mean':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                mean_result = self._mean_convolution(X[mask], kernel_size)
                X_processed[mask] = (laplace_result + mean_result).reshape(-1)

            # Apply sum operation if specified
            elif self.conv_type == 'sum':
                sum_result = self._sum_convolution(X[mask], kernel_size)
                X_processed[mask] = sum_result.reshape(-1)

            # Apply max operation if specified
            elif self.conv_type == 'max':
                max_result = self._max_convolution(X[mask], kernel_size)
                X_processed[mask] = max_result.reshape(-1)

            elif self.conv_type == 'median':
                med_result = self.median_convolution(X[mask], kernel_size)
                #X_processed[mask] = (X[mask].reshape(-1) + med_result).reshape(-1)
                X_processed[mask] = med_result.reshape(-1)

            elif self.conv_type == 'laplace+median':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                med_result = self.median_convolution(X[mask], kernel_size)
                X_processed[mask] = (laplace_result + med_result).reshape(-1)
            
            elif self.conv_type == 'gradient':
                gradient_result = self._gradient_convolution(X[mask], kernel_size)
                X_processed[mask] = gradient_result

            elif self.conv_type == 'gaussian':
                result = self._gaussian_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)

            elif self.conv_type == 'cubic':
                result = self._cubic_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)
            
            elif self.conv_type == 'quartic':
                result = self._quartic_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)
            
            elif self.conv_type == 'circular':
                result = self._circular_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)

            else:
                raise ValueError(
                    f"Error: conv_type must be in  ['laplace', 'mean', 'laplace+mean', 'sum', 'max'], got {self.conv_type}"
                )
            
        return X_processed
    
class ScalerClassRisk:
    def __init__(self, col_id, dir_output, target, scaler=None, class_risk=None, preprocessor=None):
        """
        Initialize the ScalerClassRisk.

        :param col_id: Column ID used to group data.
        :param dir_output: Directory to save output files (e.g., histograms).
        :param target: Target name used for labeling outputs.
        :param scaler: A scaler object (e.g., StandardScaler) for normalization. If None, no scaling is applied.
        :param class_risk: An object with `fit` and `predict` methods for risk classification.
        :param preprocessor: An object with an `apply` method to preprocess X before fitting.
        """
        self.n_clusters = 5
        self.col_id = col_id

        if scaler is None:
            scaler_name = None
        elif isinstance(scaler, MinMaxScaler):
            scaler_name = 'MinMax'
        elif isinstance(scaler, StandardScaler):
            scaler_name = 'Standard'
        elif isinstance(scaler, RobustScaler):
            scaler_name = 'Robust'
        else:
            raise ValueError(f'{scaler} wrong value')
        
        class_risk_name = class_risk.name if class_risk is not None else ''
        preprocessor_name = f'{preprocessor.conv_type}_{preprocessor.kernel}' if preprocessor is not None else None
        self.name = f'ScalerClassRisk_{preprocessor_name}_{scaler_name}_{class_risk_name}_{target}'
        self.dir_output = dir_output / self.name
        self.target = target
        self.scaler = scaler
        self.class_risk = class_risk
        self.preprocessor = preprocessor
        self.preprocessor_id_col = preprocessor.id_col if preprocessor is not None else None
        check_and_create_path(self.dir_output)
        self.models_by_id = {}
        self.is_fit = False

    def fit(self, X, sinisters, ids, ids_preprocessor=None):
        """
        Fit models for each unique ID and calculate statistics.

        :param X: Array of values to scale and classify.
        :param sinisters: Array of sinisters values corresponding to each value in X.
        :param ids: Array of IDs corresponding to each value in X.
        :param ids_preprocessor: IDs to use for preprocessing (if preprocessor is not None).
        """
        self.is_fit = True
        logger.info(f'########################################## {self.name} ##########################################')

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if sinisters is not None and len(sinisters.shape) == 1:
            sinisters = sinisters.reshape(-1)

        X_ = np.copy(X)

        # Apply preprocessor if provided
        if self.preprocessor is not None:
            if ids_preprocessor is None:
                raise ValueError("ids_preprocessor must be provided when preprocessor is not None.")
            X = self.preprocessor.apply(X, ids_preprocessor)

        lambda_function = lambda x: max(x, 1)
        self.models_by_id = {}

        for unique_id in np.unique(ids):
            mask = ids == unique_id
            X_id = X[mask]

            if sinisters is not None:
                sinisters_id = sinisters[mask]
            else:
                sinisters_id = None

            # Scale data if scaler is provided
            if self.scaler is not None:
                scaler = deepcopy(self.scaler)
                scaler.fit(X_id.reshape(-1, 1))
                X_scaled = scaler.transform(X_id.reshape(-1, 1))
            else:
                scaler = None
                X_scaled = X_id

            X_scaled = X_scaled.astype(np.float32)

            # Fit the class_risk model
            class_risk = deepcopy(self.class_risk)

            if len(X_scaled.shape) == 1:
                X_scaled = np.reshape(X_scaled, (-1,1))

            sinisters_mean = None
            sinisters_min = None
            sinisters_max = None
            if class_risk is not None:
                class_risk.fit(X_scaled, sinisters_id)

                # Predict classes for current ID
                classes = class_risk.predict(X_scaled)

                if sinisters is not None:
                
                    # Apply the lambda function using np.vectorize for the condition `sinisters > 0`
                    if True in np.unique(sinisters_id > 0):
                        classes[sinisters_id > 0] = np.vectorize(lambda_function)(classes[sinisters_id > 0])

                    # Calculate statistics for each class
                    sinisters_mean = []
                    sinisters_min = []
                    sinisters_max = []

                    for cls in np.unique(classes):
                        class_mask = (classes == cls).reshape(-1)
                        sinisters_class = sinisters_id[class_mask]
                        sinisters_mean.append(np.mean(sinisters_class))
                        sinisters_min.append(np.min(sinisters_class))
                        sinisters_max.append(np.max(sinisters_class))

                    sinisters_mean = np.array(sinisters_mean)
                    sinisters_min = np.array(sinisters_min)
                    sinisters_max = np.array(sinisters_max)

            self.models_by_id[unique_id] = {
                'scaler': scaler,
                'class_risk': class_risk,
                'sinistres_mean': sinisters_mean,
                'sinistres_min': sinisters_min,
                'sinistres_max': sinisters_max
            }

        if sinisters is None:
            return

        pred = self.predict(X_, sinisters, ids, ids_preprocessor)

        # Plot histogram of the classes in X[mask]
        plt.figure(figsize=(8, 6))
        plt.hist(pred, bins=np.unique(pred).shape[0], color='blue', alpha=0.7, edgecolor='black')
        plt.title(f'Histogram')
        plt.xlabel('Values')
        plt.ylabel('Frequency')

        # Save the plot
        output_path = os.path.join(self.dir_output, f'histogram_train.png')
        plt.savefig(output_path)
        plt.close()
        
        if self.class_risk is not None:
            for cl in np.unique(pred):
                logger.info(f'{cl} -> {pred[pred == cl].shape[0]}')

    def predict(self, X, sinisters, ids, ids_preprocessor=None):
        """
        Predict class labels using the appropriate model for each ID.

        :param X: Array of values to predict.
        :param ids: Array of IDs corresponding to each value in X.
        :param ids_preprocessor: IDs to use for preprocessing (if preprocessor is not None).
        :return: Array of predicted class labels.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Apply preprocessor if provided
        if self.preprocessor is not None:
            if ids_preprocessor is None:
                raise ValueError("ids_preprocessor must be provided when preprocessor is not None.")
            X = self.preprocessor.apply(X, ids_preprocessor)

        if self.class_risk is not None:
            predictions = np.zeros_like(ids, dtype=int)
        else:    
            predictions = np.zeros_like(ids, dtype=np.float32)

        for unique_id, model in self.models_by_id.items():
            if unique_id not in np.unique(ids):
                continue

            mask = ids == unique_id
            scaler = model['scaler']
            class_risk = model['class_risk']

            # Scale data if scaler exists
            if scaler is not None:
                X_scaled = scaler.transform(X[mask].reshape(-1,1))
            else:
                X_scaled = X[mask]

            if len(X_scaled.shape) == 1:
                X_scaled = np.reshape(X_scaled, (-1,1))

            if class_risk is not None:
                predictions[mask] = class_risk.predict(X_scaled.astype(np.float32)).reshape(-1)
            else:
                predictions[mask] = X_scaled.astype(np.float32).reshape(-1)
        
        if class_risk is not None:
            predictions[predictions >= self.n_clusters] = self.n_clusters - 1

        # Define the lambda function
        lambda_function = lambda x: max(x, 1)

        if sinisters is not None and class_risk is not None:
            sinisters = sinisters.reshape(-1)
            if np.any(sinisters > 0):
                # Apply the lambda function using np.vectorize for the condition `sinisters > 0`
                predictions[sinisters > 0] = np.vectorize(lambda_function)(predictions[sinisters > 0])

        """if ids_preprocessor is not None:
            fig, ax = plt.subplots(2, figsize=(15,5))
            ax[0].plot(predictions[ids_preprocessor[:, 1] == 4])
            ax[1].plot(X[ids_preprocessor[:, 1] == 4])
            plt.show()"""

        return predictions

    def fit_predict(self, X, ids, sinisters, ids_preprocessor=None):
        self.fit(X, ids, sinisters, ids_preprocessor)
        return self.predict(X, ids, ids_preprocessor)

    def predict_stat(self, X, ids, stat_key):
        """
        Generic method to predict statistics (mean, min, max) for sinistres based on the class of each sample.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :param stat_key: Key to fetch the required statistic ('sinistres_mean', 'sinistres_min', 'sinistres_max').
        :return: Array of predicted statistics for each sample based on its class.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        predictions = np.zeros(X.shape[0])

        for unique_id, model in self.models_by_id.items():
            if unique_id not in np.unique(ids):
                continue

            mask = ids == unique_id
            
            stats = model[stat_key]
            predictions[mask] = np.array([stats[int(cls)] for cls in X[mask]])

        return predictions

    def predict_nbsinister(self, X, ids):
        """
        Predict the mean sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted mean sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_mean')

    def predict_nbsinister_min(self, X, ids):
        """
        Predict the minimum sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted minimum sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_min')

    def predict_nbsinister_max(self, X, ids):
        """
        Predict the maximum sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted maximum sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_max')
    
    def predict_risk(self, X, sinisters, ids, ids_preprocessor=None):
        if self.is_fit:
            return self.predict(X, sinisters, ids, ids_preprocessor)
        else:
            return self.fit_predict(X, sinisters, ids, ids_preprocessor)

# Fonction pour calculer la somme dans une fenêtre de rolling, incluant les fenêtres inversées
def calculate_rolling_sum(dataset, column, shifts, group_col, func):
    """
    Calcule la somme rolling sur une fenêtre donnée pour chaque groupe.
    Combine les fenêtres normales et inversées.

    :param dataset: Le DataFrame Pandas.
    :param column: Colonne sur laquelle appliquer le rolling.
    :param shifts: Taille de la fenêtre rolling.
    :param group_col: Colonne pour le groupby.
    :param func: Fonction à appliquer sur les fenêtres.
    :return: Colonne calculée avec la somme rolling bidirectionnelle.
    """
    if shifts == 0:
        return dataset[column].values
    
    dataset.reset_index(drop=True)
    
    # Rolling forward
    forward_rolling = dataset.groupby(group_col)[column].rolling(window=shifts).apply(func).values
    forward_rolling[np.isnan(forward_rolling)] = 0
    
    # Rolling backward (inversé)
    backward_rolling = (
        dataset.iloc[::-1]
        .groupby(group_col)[column]
        .rolling(window=shifts, min_periods=1)
        .apply(func, raw=True)
        .iloc[::-1]  # Remettre dans l'ordre original
    ).values
    backward_rolling[np.isnan(backward_rolling)] = 0

    # Somme des deux fenêtres
    return forward_rolling + backward_rolling - dataset[column].values
    #return forward_rolling

def calculate_rolling_sum_per_col_id(dataset, column, shifts, group_col, func):
    """
    Calcule la somme rolling sur une fenêtre donnée pour chaque col_id sans utiliser des boucles sur les indices internes.
    Combine les fenêtres normales et inversées, tout en utilisant rolling.

    :param dataset: Le DataFrame Pandas.
    :param column: Colonne sur laquelle appliquer le rolling.
    :param shifts: Taille de la fenêtre rolling.
    :param group_col: Colonne identifiant les groupes (col_id).
    :param func: Fonction à appliquer sur les fenêtres.
    :return: Numpy array contenant les sommes rolling bidirectionnelles pour chaque col_id.
    """
    if shifts == 0:
        return dataset[column].values

    # Initialiser un tableau pour stocker les résultats
    result = np.zeros(len(dataset))

    # Obtenir les valeurs uniques de col_id
    unique_col_ids = dataset[group_col].unique()

    # Parcourir chaque groupe col_id
    for col_id in unique_col_ids:
        # Filtrer le groupe correspondant
        group_data = dataset[dataset[group_col] == col_id]
        group_data.sort_values('date', inplace=True)

        # Calculer rolling forward
        forward_rolling = group_data[column].rolling(window=shifts, min_periods=1).apply(func, raw=True).values

        # Calculer rolling backward (fenêtres inversées)
        backward_rolling = (
            group_data[column][::-1]
            .rolling(window=shifts, min_periods=1)
            .apply(func, raw=True)[::-1]
            .values
        )

        # Combine forward et backward
        group_result = forward_rolling + backward_rolling - group_data[column].values

        # Affecter le résultat au tableau final
        result[group_data.index] = group_result

    return result

def class_window_sum(dataset, group_col, column, shifts):
    # Initialize a column to store the rolling aggregation
    column_name = f'nbsinister_sum_{shifts}'
    dataset[column_name] = 0.0

    # Case when window_size is 1
    if shifts == 0:
        dataset[column_name] = dataset[column].values
        return dataset
    else:
        # For each unique graph_id
        for graph_id in dataset[group_col].unique():
            # Filter data for the current graph_id
            df_graph = dataset[dataset[group_col] == graph_id]

            # Iterate through each row in df_graph
            for idx, row in df_graph.iterrows():
                # Define the window bounds
                date_min = row['date'] - shifts
                date_max = row['date'] + shifts
                
                # Filter rows within the date window
                window_df = df_graph[(df_graph['date'] >= date_min) & (df_graph['date'] <= date_max)]
                
                # Apply the aggregation function
                dataset.at[idx, column_name] = window_df[column].sum()
    return dataset

def class_window_max(dataset, group_col, column, shifts):
    # Initialize a column to store the rolling aggregation
    column_name = f'nbsinister_max_{shifts}'
    dataset[column_name] = 0.0

    # Case when window_size is 1
    if shifts == 0:
        dataset[column_name] = dataset[column].values
        return dataset
    else:
        # For each unique graph_id
        for graph_id in dataset[group_col].unique():
            # Filter data for the current graph_id
            df_graph = dataset[dataset[group_col] == graph_id]

            # Iterate through each row in df_graph
            for idx, row in df_graph.iterrows():
                # Define the window bounds
                date_min = row['date'] - shifts
                date_max = row['date'] + shifts
                
                # Filter rows within the date window
                window_df = df_graph[(df_graph['date'] >= date_min) & (df_graph['date'] <= date_max)]
                
                # Apply the aggregation function
                dataset.at[idx, column_name] = window_df[column].max()
    
    return dataset

def create_risk_target(train_dataset, val_dataset, test_dataset, dir_post_process, graph):

    graph_method = graph.graph_method

    new_cols = []

    if graph_method == 'node':
        train_dataset_ = train_dataset.copy(deep=True)
        val_dataset_ = val_dataset.copy(deep=True)
        test_dataset_ = test_dataset.copy(deep=True)
    else:
        def keep_one_per_pair(dataset):
            # Supprime les doublons en gardant uniquement la première occurrence par paire (graph_id, date)
            return dataset.drop_duplicates(subset=['graph_id', 'date'], keep='first')

        train_dataset_ = keep_one_per_pair(train_dataset)
        val_dataset_ = keep_one_per_pair(val_dataset)
        test_dataset_ = keep_one_per_pair(test_dataset)

    res = {}

    ####################################################################################
    
    obj2 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    obj2.fit(train_dataset_['nbsinister'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(train_dataset_['nbsinister'].values,  train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(val_dataset_['nbsinister'].values,  val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(test_dataset_['nbsinister'].values,  test_dataset_['nbsinister'].values, test_dataset_['departement'].values)
    
    res[obj2.name] = obj2

    new_cols.append('nbsinister-kmeans-5-Class-Dept')

    ###############################################################################

    if graph.sequences_month is None:
        graph.compute_sequence_month(pd.concat([train_dataset, test_dataset]), graph.dataset_name)

    conv_types = ['cubic', 'gaussian', 'circular', 'quartic', 'mean', 'median', 'max', 'sum']
    #conv_types = ['cubic']

    kernels = ['Specialized', 1, 3, 5]
    
    n_clusters = 5

    for conv_type in conv_types:
        for kernel in kernels:
            logger.info(f"Testing with convolution type: {conv_type}")

            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'])

            # Définition de l'objet ScalerClassRisk
            class_risk = KMeansRiskZerosHandle(n_clusters=n_clusters)
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='nbsinister',
                scaler=None,
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            val_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"
            test_col = f"nbsinister-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}"

            train_dataset_[train_col] = obj.predict(
                train_dataset_['nbsinister'].values,
                train_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[val_col] = obj.predict(
                val_dataset_['nbsinister'].values,
                val_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[test_col] = obj.predict(
                test_dataset_['nbsinister'].values,
                test_dataset_['nbsinister'].values,  # Ajout de dataset['nbsinister'] comme 2ème argument
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            # Stockage des résultats
            res[obj.name] = deepcopy(obj)
            new_cols.append(train_col)

            train_col = f"nbsinisterDaily-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}-Past"
    
            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'], persistence=True)

            # Définition de l'objet ScalerClassRisk
            class_risk = KMeansRisk(n_clusters=n_clusters)
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='nbsinisterDaily',
                scaler=None,
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_dataset_[train_col] = obj.predict(
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[train_col] = obj.predict(
                val_dataset_['nbsinisterDaily'].values,
                val_dataset_['nbsinisterDaily'].values,
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[train_col] = obj.predict(
                test_dataset_['nbsinisterDaily'].values,
                test_dataset_['nbsinisterDaily'].values,
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_col = f"burnedareaDaily-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}-Past"

            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'], persistence=True)

            res[obj.name] = deepcopy(obj)
            new_cols.append(train_col)

    logger.info(f"Completed processing for convolution type: {conv_type} with kernel {kernel}")

    ################################################

    logger.info(f'Post process Model -> {res}')

    if graph_method == 'node':
        train_dataset = train_dataset_
        val_dataset = val_dataset_
        test_dataset = test_dataset_
    else:
        def join_on_index_with_new_cols(original_dataset, updated_dataset, new_cols):
            """
            Effectue un join sur les index (graph_id, date) pour ajouter de nouvelles colonnes.
            :param original_dataset: DataFrame original
            :param updated_dataset: DataFrame avec les index et colonnes à joindre
            :param new_cols: Liste des colonnes à ajouter
            :return: DataFrame mis à jour avec les nouvelles colonnes
            """
            # Joindre les deux DataFrames sur leurs index
            original_dataset.reset_index(drop=True, inplace=True)
            updated_dataset.reset_index(drop=True, inplace=True)

            joined_dataset = original_dataset.set_index(['graph_id', 'date']).join(
                updated_dataset.set_index(['graph_id', 'date'])[new_cols],
                on=['graph_id', 'date'],
                how='left'
            ).reset_index()
            return joined_dataset

        # Mise à jour des datasets
        train_dataset = join_on_index_with_new_cols(train_dataset, train_dataset_, new_cols)
        val_dataset = join_on_index_with_new_cols(val_dataset, val_dataset_, new_cols)
        test_dataset = join_on_index_with_new_cols(test_dataset, test_dataset_, new_cols)

    return res, train_dataset, val_dataset, test_dataset, new_cols
