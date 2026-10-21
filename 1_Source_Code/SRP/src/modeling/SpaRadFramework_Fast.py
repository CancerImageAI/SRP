import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
import networkx as nx
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
from functools import partial
import os
from DataLoader import load_nifti_data
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from typing import Dict, List, Tuple, Optional
import pickle
import json
import hashlib
from pathlib import Path
import gc
import traceback
from datetime import datetime
import time
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sparad_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress radiomics warnings
radiomics_logger = logging.getLogger("radiomics")
radiomics_logger.setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """
    Manages hierarchical caching of feature extraction results
    """
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate subdirectories for different cache types
        self.feature_maps_dir = self.cache_dir / 'feature_maps'
        self.final_features_dir = self.cache_dir / 'final_features'
        self.metadata_dir = self.cache_dir / 'metadata'
        
        for dir_path in [self.feature_maps_dir, self.final_features_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Cache manager initialized at: {self.cache_dir}")
    
    def _get_params_hash(self, params):
        """Generate hash from parameters"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    def get_cache_key(self, patient_id, params):
        """Generate cache key from patient ID and parameters"""
        params_hash = self._get_params_hash(params)
        return f"{patient_id}_{params_hash}"
    
    def save_feature_maps(self, patient_id, params, feature_maps):
        """Save feature maps to disk with compression"""
        try:
            cache_key = self.get_cache_key(patient_id, params)
            cache_file = self.feature_maps_dir / f"{cache_key}.pkl.gz"
            
            import gzip
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(feature_maps, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Saved feature maps for {patient_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save feature maps for {patient_id}: {str(e)}")
            return False
    
    def load_feature_maps(self, patient_id, params):
        """Load feature maps from disk"""
        try:
            cache_key = self.get_cache_key(patient_id, params)
            cache_file = self.feature_maps_dir / f"{cache_key}.pkl.gz"
            
            if cache_file.exists():
                import gzip
                with gzip.open(cache_file, 'rb') as f:
                    feature_maps = pickle.load(f)
                logger.debug(f"Loaded feature maps for {patient_id}")
                return feature_maps
        except Exception as e:
            logger.warning(f"Failed to load feature maps for {patient_id}: {str(e)}")
        
        return None
    
    def save_final_features(self, patient_id, params, features_df):
        """Save final features to CSV"""
        try:
            cache_key = self.get_cache_key(patient_id, params)
            cache_file = self.final_features_dir / f"{cache_key}.csv"
            
            features_df.to_csv(cache_file, index=False)
            
            # Save metadata
            metadata = {
                'patient_id': patient_id,
                'params': params,
                'timestamp': datetime.now().isoformat(),
                'n_features': len(features_df)
            }
            metadata_file = self.metadata_dir / f"{cache_key}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved final features for {patient_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save final features for {patient_id}: {str(e)}")
            return False
    
    def load_final_features(self, patient_id, params):
        """Load final features from CSV"""
        try:
            cache_key = self.get_cache_key(patient_id, params)
            cache_file = self.final_features_dir / f"{cache_key}.csv"
            
            if cache_file.exists():
                df = pd.read_csv(cache_file)
                logger.debug(f"Loaded final features for {patient_id}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load final features for {patient_id}: {str(e)}")
        
        return None
    
    def exists_final_features(self, patient_id, params):
        """Check if final features exist"""
        cache_key = self.get_cache_key(patient_id, params)
        cache_file = self.final_features_dir / f"{cache_key}.csv"
        return cache_file.exists()
    
    def get_completed_patients(self, params):
        """Get set of patient IDs that have completed processing"""
        params_hash = self._get_params_hash(params)
        completed = set()
        
        for cache_file in self.final_features_dir.glob(f"*_{params_hash}.csv"):
            # Extract patient_id (everything before _params_hash)
            patient_id = cache_file.stem.replace(f"_{params_hash}", "")
            completed.add(patient_id)
        
        return completed
    
    def clear_patient_cache(self, patient_id, params):
        """Clear all cached data for a specific patient"""
        cache_key = self.get_cache_key(patient_id, params)
        
        files_to_remove = [
            self.feature_maps_dir / f"{cache_key}.pkl.gz",
            self.final_features_dir / f"{cache_key}.csv",
            self.metadata_dir / f"{cache_key}.json"
        ]
        
        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed cache file: {file_path}")
    
    def clear_all_cache(self):
        """Clear entire cache directory"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared all cache in: {self.cache_dir}")


# ============================================================================
# OPTIMIZED SPARAD FRAMEWORK WITH CACHING
# ============================================================================

class OptimizedSpaRadFramework:
    """
    Optimized Spatial Radiomics Framework with caching and robustness improvements
    """
    
    def __init__(self, 
                 bin_width=25,
                 window_size=3,
                 max_clusters=10,
                 feature_classes=['glcm', 'glrlm', 'glszm'],
                 graph_features_list=None,
                 n_threads=4,
                 batch_size=100,
                 cache_manager=None):
        
        self.bin_width = bin_width
        self.window_size = window_size
        self.max_clusters = max_clusters
        self.feature_classes = feature_classes
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.cache_manager = cache_manager
        
        # Store parameters for cache key generation
        self.params = {
            'bin_width': bin_width,
            'window_size': window_size,
            'max_clusters': max_clusters,
            'feature_classes': feature_classes
        }
        
        if graph_features_list is None:
            self.graph_features_list = [
                'average_degree', 'degree_std', 'max_degree',
                'clustering_coefficient', 'average_path_length',
                'diameter', 'density', 'assortativity',
                'betweenness_mean', 'betweenness_std',
                'closeness_mean', 'closeness_std',
                'eigenvector_mean', 'number_of_nodes', 'number_of_edges'
            ]
        else:
            self.graph_features_list = graph_features_list
        
        self.feature_selections = self._define_feature_selections()
        self.extractors = self._initialize_extractors()

    def _define_feature_selections(self):
        """Define which features to extract from each class"""
        return {
            'glcm': [
                'JointEnergy', 'JointEntropy', 'Contrast', 'Correlation',
                'Idm', 'SumAverage', 'SumSquares', 'SumEntropy',
                'DifferenceEntropy', 'DifferenceAverage', 'DifferenceVariance',
                'Imc1', 'Imc2'
            ],
            'glrlm': [
                'ShortRunEmphasis', 'LongRunEmphasis', 'GrayLevelNonUniformity',
                'RunLengthNonUniformity', 'RunPercentage', 'LowGrayLevelRunEmphasis',
                'HighGrayLevelRunEmphasis', 'ShortRunLowGrayLevelEmphasis',
                'ShortRunHighGrayLevelEmphasis', 'LongRunLowGrayLevelEmphasis',
                'LongRunHighGrayLevelEmphasis', 'GrayLevelVariance', 'RunEntropy'
            ],
            'glszm': [
                'SmallAreaEmphasis', 'LargeAreaEmphasis', 'GrayLevelNonUniformity',
                'SizeZoneNonUniformity', 'ZonePercentage', 'LowGrayLevelZoneEmphasis',
                'HighGrayLevelZoneEmphasis', 'SmallAreaLowGrayLevelEmphasis',
                'SmallAreaHighGrayLevelEmphasis', 'LargeAreaLowGrayLevelEmphasis',
                'LargeAreaHighGrayLevelEmphasis', 'GrayLevelVariance', 'ZoneEntropy'
            ]
        }

    def _initialize_extractors(self):
        """Initialize PyRadiomics extractors for each feature class"""
        extractors = {}
        settings = {
            'binWidth': self.bin_width,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': (1.0, 1.0, 1.0),
            'correctMask': True,
            'normalize': False,
            'normalizeScale': 1,
            'removeOutliers': None,
        }
        
        for feature_class in self.feature_classes:
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName(feature_class)
            extractors[feature_class] = extractor
        
        return extractors

    def _extract_window_features_batch(self, coords_batch, image_array, mask_array, 
                                      image_sitk, pad_size):
        """
        Extract features for a batch of coordinates with error handling
        """
        results = {fc: {fn: {} for fn in self.feature_selections[fc]} 
                  for fc in self.feature_classes}
        
        for idx, coord in enumerate(coords_batch):
            try:
                z, y, x = coord
                
                # Define window bounds
                z_min = max(0, z - pad_size)
                z_max = min(mask_array.shape[0], z + pad_size + 1)
                y_min = max(0, y - pad_size)
                y_max = min(mask_array.shape[1], y + pad_size + 1)
                x_min = max(0, x - pad_size)
                x_max = min(mask_array.shape[2], x + pad_size + 1)
                
                # Extract window
                local_image = image_array[z_min:z_max, y_min:y_max, x_min:x_max]
                local_mask = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
                
                # Convert to SimpleITK
                local_image_sitk = sitk.GetImageFromArray(local_image.astype(np.float32))
                local_image_sitk.SetSpacing(image_sitk.GetSpacing())
                local_mask_sitk = sitk.GetImageFromArray(local_mask.astype(np.uint8))
                local_mask_sitk.CopyInformation(local_image_sitk)
                
                # Extract features for all classes
                for feature_class in self.feature_classes:
                    min_voxels = 27 if feature_class == 'glszm' else 10
                    
                    if np.sum(local_mask) < min_voxels:
                        for fn in self.feature_selections[feature_class]:
                            results[feature_class][fn][idx] = 0.0
                        continue
                    
                    try:
                        features = self.extractors[feature_class].execute(
                            local_image_sitk, local_mask_sitk
                        )
                        
                        for fn in self.feature_selections[feature_class]:
                            feature_key = f'original_{feature_class}_{fn}'
                            value = float(features.get(feature_key, 0.0))
                            
                            # Handle NaN and inf
                            if np.isnan(value) or np.isinf(value):
                                value = 0.0
                            
                            results[feature_class][fn][idx] = value
                    
                    except Exception as e:
                        for fn in self.feature_selections[feature_class]:
                            results[feature_class][fn][idx] = 0.0
            
            except Exception as e:
                logger.error(f"Critical error processing coord {coord}: {str(e)}")
                for fc in self.feature_classes:
                    for fn in self.feature_selections[fc]:
                        results[fc][fn][idx] = 0.0
        
        return results

    def extract_all_feature_maps(self, image, mask, patient_id=None):
        """
        Extract all feature maps with caching support
        """
        # Check cache first
        if self.cache_manager and patient_id:
            cached_maps = self.cache_manager.load_feature_maps(patient_id, self.params)
            if cached_maps is not None:
                logger.info(f"✓ Loaded feature maps from cache for {patient_id}")
                return cached_maps
        
        logger.info("=" * 80)
        logger.info("EXTRACTING SPATIAL RADIOMICS FEATURE MAPS")
        logger.info("=" * 80)
        
        try:
            # Convert to arrays
            if isinstance(image, np.ndarray):
                image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
                image_array = image
            else:
                image_sitk = image
                image_array = sitk.GetArrayFromImage(image)
            
            if isinstance(mask, np.ndarray):
                mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
                mask_sitk.CopyInformation(image_sitk)
                mask_array = mask
            else:
                mask_sitk = mask
                mask_array = sitk.GetArrayFromImage(mask)
            
            # Initialize feature maps
            all_feature_maps = {}
            for feature_class in self.feature_classes:
                all_feature_maps[feature_class] = {
                    f"{feature_class}_{fn}": np.zeros_like(mask_array, dtype=np.float32)
                    for fn in self.feature_selections[feature_class]
                }
            
            # Get tumor coordinates
            tumor_coords = np.argwhere(mask_array > 0)
            total_voxels = len(tumor_coords)
            pad_size = self.window_size // 2
            
            logger.info(f"Processing {total_voxels} voxels with {self.n_threads} threads...")
            
            # Split coordinates into batches
            batches = [tumor_coords[i:i + self.batch_size] 
                      for i in range(0, total_voxels, self.batch_size)]
            
            # Process batches with thread pool
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                extract_func = partial(
                    self._extract_window_features_batch,
                    image_array=image_array,
                    mask_array=mask_array,
                    image_sitk=image_sitk,
                    pad_size=pad_size
                )
                
                futures = [executor.submit(extract_func, batch) for batch in batches]
                batch_results = []
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc="Extracting features", ncols=100):
                    try:
                        result = future.result(timeout=300)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
                        empty_result = {fc: {fn: {} for fn in self.feature_selections[fc]} 
                                      for fc in self.feature_classes}
                        batch_results.append(empty_result)
            
            # Assemble results back into feature maps
            logger.info("Assembling feature maps...")
            for batch_idx, batch_result in enumerate(batch_results):
                batch_coords = batches[batch_idx]
                
                for feature_class in self.feature_classes:
                    for feature_name in self.feature_selections[feature_class]:
                        map_key = f"{feature_class}_{feature_name}"
                        
                        for coord_idx, value in batch_result[feature_class][feature_name].items():
                            if coord_idx < len(batch_coords):
                                coord = batch_coords[coord_idx]
                                all_feature_maps[feature_class][map_key][tuple(coord)] = value
            
            # Cache the feature maps
            if self.cache_manager and patient_id:
                self.cache_manager.save_feature_maps(patient_id, self.params, all_feature_maps)
                logger.info(f"✓ Saved feature maps to cache for {patient_id}")
            
            # Force garbage collection
            gc.collect()
            
            return all_feature_maps
        
        except Exception as e:
            logger.error(f"Feature map extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cluster_feature_map_fast(self, feature_map, mask):
        """
        Optimized clustering with improved error handling
        """
        try:
            feature_values = feature_map[mask > 0].reshape(-1, 1)
            valid_idx = np.isfinite(feature_values).flatten()
            feature_values = feature_values[valid_idx]
            
            if len(feature_values) < 10:
                return np.zeros_like(mask, dtype=np.int32), 1, None
            
            # Standardize
            mean, std = np.mean(feature_values), np.std(feature_values)
            if std < 1e-10:
                labels_3d = np.zeros_like(mask, dtype=np.int16)
                labels_3d[mask > 0] = 1
                return labels_3d, 1, None
            feature_values_std = (feature_values - mean) / std
        
            # BIC selection with early stopping
            max_possible_clusters = min(self.max_clusters, len(feature_values) // 20, 15)
            if max_possible_clusters < 2:
                max_possible_clusters = 2
            
            cluster_range = range(2, max_possible_clusters + 1)
            bic_scores = []
            best_bic = np.inf
            patience = 3
            no_improvement = 0
            
            for n in cluster_range:
                try:
                    gmm = GaussianMixture(
                        n_components=n,
                        covariance_type='full',
                        n_init=3,
                        max_iter=50,
                        random_state=42,
                        tol=1e-2
                    )
                    gmm.fit(feature_values_std)
                    bic = gmm.bic(feature_values_std)
                    bic_scores.append(bic)
                    
                    if bic < best_bic:
                        best_bic = bic
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            break
                except Exception as e:
                    bic_scores.append(np.inf)
            
            if not bic_scores or all(np.isinf(bic_scores)):
                labels_3d = np.zeros_like(mask, dtype=np.int16)
                labels_3d[mask > 0] = 1
                return labels_3d, 1, None
            
            optimal_n_clusters = cluster_range[np.argmin(bic_scores)]
            
            # Final clustering
            try:
                gmm_final = GaussianMixture(
                    n_components=optimal_n_clusters,
                    covariance_type='full',
                    n_init=3,
                    max_iter=50,
                    random_state=42,
                    tol=1e-2
                )
                cluster_assignments = gmm_final.fit_predict(feature_values_std)
            except Exception as e:
                labels_3d = np.zeros_like(mask, dtype=np.int16)
                labels_3d[mask > 0] = 1
                return labels_3d, 1, None
            
            # Map to 3D
            labels_3d = np.zeros_like(mask, dtype=np.int16)
            all_masked_coords = np.argwhere(mask > 0)
            valid_masked_coords = all_masked_coords[valid_idx]
            
            for idx, coord in enumerate(valid_masked_coords):
                labels_3d[tuple(coord)] = cluster_assignments[idx] + 1
            
            return labels_3d, optimal_n_clusters, gmm_final
        
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            labels_3d = np.zeros_like(mask, dtype=np.int16)
            labels_3d[mask > 0] = 1
            return labels_3d, 1, None

    def build_graph_from_clusters_fast(self, feature_map, labels, mask,
                                    graph_type='adaptive',
                                    use_spatial_distance=True,
                                    alpha=0.9, beta=0.1,
                                    threshold=0.1,
                                    edge_type='weighted'):
        """
        Optimized graph construction with robust error handling
        """
        try:
            if isinstance(mask, sitk.Image):
                mask = sitk.GetArrayFromImage(mask)
            
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]
            n_clusters = len(unique_labels)
            
            G = nx.Graph()
            
            if n_clusters == 0:
                return G, {'n_clusters': 0, 'skipped': True}
            
            if n_clusters == 1:
                G.add_node(int(unique_labels[0]), size=int(np.sum(labels == unique_labels[0])))
                return G, {'n_clusters': 1, 'single_cluster': True}
            
            # Extract cluster properties
            cluster_properties = {}
            
            for cluster_id in unique_labels:
                cluster_mask = (labels == cluster_id) & (mask > 0)
                if not np.any(cluster_mask):
                    continue
                
                coords = np.argwhere(cluster_mask)
                feature_values = feature_map[cluster_mask]
                feature_values = feature_values[np.isfinite(feature_values)]
                
                if len(feature_values) == 0:
                    continue
                
                centroid = coords.mean(axis=0)
                
                cluster_properties[int(cluster_id)] = {
                    'centroid': centroid,
                    'feature_values': feature_values,
                    'size': len(feature_values)
                }
                
                G.add_node(
                    int(cluster_id),
                    centroid=centroid.tolist(),
                    size=len(feature_values),
                    mean_feature=float(np.mean(feature_values))
                )
            
            valid_cluster_ids = list(cluster_properties.keys())
            n_valid = len(valid_cluster_ids)
            
            if n_valid < 2:
                return G, {'n_clusters': n_valid, 'insufficient_clusters': True}
            
            # Compute distances
            emd_matrix = np.zeros((n_valid, n_valid))
            spatial_matrix = np.zeros((n_valid, n_valid))
            
            for i in range(n_valid):
                for j in range(i+1, n_valid):
                    ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                    
                    # EMD
                    try:
                        emd = wasserstein_distance(
                            cluster_properties[ci]['feature_values'],
                            cluster_properties[cj]['feature_values']
                        )
                        if np.isnan(emd) or np.isinf(emd):
                            emd = 0.0
                    except:
                        emd = 0.0
                    
                    emd_matrix[i, j] = emd_matrix[j, i] = emd
                    
                    # Spatial distance
                    if use_spatial_distance:
                        try:
                            spatial_dist = np.linalg.norm(
                                cluster_properties[ci]['centroid'] - 
                                cluster_properties[cj]['centroid']
                            )
                            if np.isnan(spatial_dist) or np.isinf(spatial_dist):
                                spatial_dist = 0.0
                        except:
                            spatial_dist = 0.0
                    else:
                        spatial_dist = 0.0
                    
                    spatial_matrix[i, j] = spatial_matrix[j, i] = spatial_dist
            
            # Normalize distance matrices
            def normalize_matrix(matrix):
                values = matrix[np.triu_indices(n_valid, k=1)]
                if len(values) == 0:
                    return np.zeros_like(matrix)
                
                std = np.std(values)
                mean = np.mean(values)
                
                if std > 1e-10:
                    matrix_norm = (matrix - mean) / std
                    min_val = np.nanmin(matrix_norm)
                    max_val = np.nanmax(matrix_norm)
                    
                    if max_val > min_val:
                        matrix_norm = (matrix_norm - min_val) / (max_val - min_val)
                    else:
                        matrix_norm = np.zeros_like(matrix)
                else:
                    matrix_norm = np.zeros_like(matrix)
                
                return np.nan_to_num(matrix_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
            emd_matrix_norm = normalize_matrix(emd_matrix)
            spatial_matrix_norm = normalize_matrix(spatial_matrix) if use_spatial_distance else np.zeros_like(emd_matrix)
            
            # Normalize alpha and beta
            if alpha + beta > 0:
                total = alpha + beta
                alpha, beta = alpha / total, beta / total
            
            combined_matrix = alpha * emd_matrix_norm + beta * spatial_matrix_norm
            
            # Build edges based on graph type
            edges_added = 0
            
            if graph_type == 'adaptive':
                if n_valid <= 5:
                    graph_type = 'complete'
                elif n_valid <= 15:
                    graph_type = 'threshold'
                else:
                    graph_type = 'knn'
            
            if graph_type == 'complete':
                for i in range(n_valid):
                    for j in range(i+1, n_valid):
                        ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                        similarity = 1.0 - combined_matrix[i, j]
                        weight = float(similarity) if edge_type == 'weighted' else 1.0
                        
                        G.add_edge(ci, cj, weight=weight)
                        edges_added += 1
            
            elif graph_type == 'threshold':
                for i in range(n_valid):
                    for j in range(i+1, n_valid):
                        similarity = 1.0 - combined_matrix[i, j]
                        if similarity > threshold:
                            ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                            weight = float(similarity) if edge_type == 'weighted' else 1.0
                            G.add_edge(ci, cj, weight=weight)
                            edges_added += 1
            
            elif graph_type == 'knn':
                k = min(5, n_valid - 1)
                for i in range(n_valid):
                    distances = [(j, combined_matrix[i, j]) for j in range(n_valid) if j != i]
                    distances.sort(key=lambda x: x[1])
                    
                    ci = valid_cluster_ids[i]
                    for j_idx, dist in distances[:k]:
                        cj = valid_cluster_ids[j_idx]
                        if not G.has_edge(ci, cj):
                            similarity = 1.0 - dist
                            weight = float(similarity) if edge_type == 'weighted' else 1.0
                            G.add_edge(ci, cj, weight=weight)
                            edges_added += 1
            
            # Handle isolated nodes
            # Handle isolated nodes
            isolated = list(nx.isolates(G))
            for node in isolated:
                try:
                    i = valid_cluster_ids.index(node)
                    other_indices = [j for j in range(n_valid) if j != i]
                    if other_indices:
                        min_dist_idx = min(other_indices, key=lambda j: combined_matrix[i, j])
                        nearest = valid_cluster_ids[min_dist_idx]
                        
                        similarity = 1.0 - combined_matrix[i, min_dist_idx]
                        weight = float(similarity) if edge_type == 'weighted' else 1.0
                        G.add_edge(node, nearest, weight=weight)
                        edges_added += 1
                except Exception as e:
                    logger.warning(f"Failed to connect isolated node {node}: {str(e)}")
            
            cluster_info = {
                'n_clusters': n_valid,
                'n_edges': edges_added,
                'graph_type': graph_type
            }
            
            return G, cluster_info
        
        except Exception as e:
            logger.error(f"Graph construction failed: {str(e)}")
            logger.error(traceback.format_exc())
            G = nx.Graph()
            G.add_node(1, size=1)
            return G, {'n_clusters': 1, 'error': True}

    def extract_graph_features_fast(self, G):
        """
        Robust graph feature extraction with comprehensive error handling
        """
        features = {feat: 0.0 for feat in self.graph_features_list}
        
        try:
            if G.number_of_nodes() == 0:
                return features
            
            if G.number_of_nodes() == 1:
                features['number_of_nodes'] = 1
                features['number_of_edges'] = 0
                return features
            
            # Basic features
            features['number_of_nodes'] = G.number_of_nodes()
            features['number_of_edges'] = G.number_of_edges()
            
            # Degree features
            try:
                degrees = dict(G.degree())
                degree_values = np.array(list(degrees.values()))
                features['average_degree'] = float(np.mean(degree_values))
                features['degree_std'] = float(np.std(degree_values))
                features['max_degree'] = int(np.max(degree_values))
            except Exception as e:
                logger.debug(f"Degree calculation failed: {str(e)}")
            
            # Density
            try:
                features['density'] = nx.density(G)
            except Exception as e:
                logger.debug(f"Density calculation failed: {str(e)}")
            
            # Clustering coefficient
            try:
                features['clustering_coefficient'] = nx.average_clustering(G)
            except Exception as e:
                logger.debug(f"Clustering coefficient calculation failed: {str(e)}")
            
            # Path features
            try:
                if nx.is_connected(G):
                    features['average_path_length'] = nx.average_shortest_path_length(G)
                    features['diameter'] = nx.diameter(G)
                else:
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc).copy()
                    if subgraph.number_of_nodes() > 1:
                        features['average_path_length'] = nx.average_shortest_path_length(subgraph)
                        features['diameter'] = nx.diameter(subgraph)
            except Exception as e:
                logger.debug(f"Path features calculation failed: {str(e)}")
            
            # Assortativity
            try:
                if G.number_of_edges() > 0 and features['degree_std'] > 1e-10:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        assortativity = nx.degree_assortativity_coefficient(G)
                        if not (np.isnan(assortativity) or np.isinf(assortativity)):
                            features['assortativity'] = float(assortativity)
            except Exception as e:
                logger.debug(f"Assortativity calculation failed: {str(e)}")
            
            # Centrality measures (only for reasonably sized graphs)
            if G.number_of_nodes() < 100:
                try:
                    betweenness = nx.betweenness_centrality(G)
                    betweenness_values = np.array(list(betweenness.values()))
                    features['betweenness_mean'] = float(np.mean(betweenness_values))
                    features['betweenness_std'] = float(np.std(betweenness_values))
                except Exception as e:
                    logger.debug(f"Betweenness centrality calculation failed: {str(e)}")
                
                try:
                    closeness = nx.closeness_centrality(G)
                    closeness_values = np.array(list(closeness.values()))
                    features['closeness_mean'] = float(np.mean(closeness_values))
                    features['closeness_std'] = float(np.std(closeness_values))
                except Exception as e:
                    logger.debug(f"Closeness centrality calculation failed: {str(e)}")
                
                try:
                    eigenvector = nx.eigenvector_centrality(G, max_iter=100, tol=1e-3)
                    eigenvector_values = np.array(list(eigenvector.values()))
                    features['eigenvector_mean'] = float(np.mean(eigenvector_values))
                except Exception as e:
                    logger.debug(f"Eigenvector centrality calculation failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Graph feature extraction error: {str(e)}")
        
        # Ensure all features are valid
        for feat in self.graph_features_list:
            if feat not in features:
                features[feat] = 0.0
            elif np.isnan(features[feat]) or np.isinf(features[feat]):
                features[feat] = 0.0
        
        return features

    def process_single_feature_map(self, feature_map, mask, feature_name):
        """Process single feature map with optimized pipeline"""
        try:
            # Cluster
            labels, n_clusters, _ = self.cluster_feature_map_fast(feature_map, mask)
            
            # Build graph
            G, cluster_info = self.build_graph_from_clusters_fast(
                feature_map, labels, mask
            )
            
            # Extract graph features
            graph_features = self.extract_graph_features_fast(G)
            
            return graph_features
        
        except Exception as e:
            logger.error(f"Failed to process feature map {feature_name}: {str(e)}")
            return {feat: 0.0 for feat in self.graph_features_list}

    def extract_sparad_descriptor(self, image, mask, patient_id=None, save_intermediate=False):
        """
        Complete SpaRad pipeline with caching support
        """
        # Check if final features are cached
        if self.cache_manager and patient_id:
            cached_features = self.cache_manager.load_final_features(patient_id, self.params)
            if cached_features is not None:
                logger.info(f"✓ Loaded final features from cache for {patient_id}")
                feature_names = cached_features['feature_name'].tolist()
                feature_values = cached_features['feature_value'].values
                feature_dict = {
                    'feature_names': feature_names,
                    'feature_values': feature_values
                }
                return feature_values, feature_dict
        
        logger.info("\n" + "=" * 80)
        logger.info(f"SPARAD EXTRACTION: {patient_id if patient_id else 'Unknown'}")
        logger.info("=" * 80)
        
        try:
            # Extract all feature maps (with caching)
            all_feature_maps = self.extract_all_feature_maps(image, mask, patient_id=patient_id)
            
            # Convert mask
            if isinstance(mask, sitk.Image):
                mask_array = sitk.GetArrayFromImage(mask)
            else:
                mask_array = mask
            
            # Process all feature maps
            all_graph_features = {}
            total_features = sum(len(maps) for maps in all_feature_maps.values())
            
            logger.info(f"\nProcessing {total_features} feature maps...")
            
            for feature_class in self.feature_classes:
                class_graph_features = {}
                feature_maps = all_feature_maps[feature_class]
                
                for feature_name, feature_map in tqdm(feature_maps.items(), 
                                                     desc=f"  {feature_class.upper()}", 
                                                     ncols=100):
                    try:
                        graph_features = self.process_single_feature_map(
                            feature_map, mask_array, feature_name
                        )
                        class_graph_features[feature_name] = graph_features
                    except Exception as e:
                        logger.error(f"Failed to process {feature_name}: {str(e)}")
                        class_graph_features[feature_name] = {
                            feat: 0.0 for feat in self.graph_features_list
                        }
                    
                    # Periodic garbage collection
                    if len(class_graph_features) % 10 == 0:
                        gc.collect()
                
                all_graph_features[feature_class] = class_graph_features
            
            # Concatenate features
            logger.info("\nConcatenating SpaRad descriptor...")
            sparad_features = []
            feature_names_list = []
            
            for feature_class in self.feature_classes:
                for feature_name in sorted(all_graph_features[feature_class].keys()):
                    graph_features = all_graph_features[feature_class][feature_name]
                    
                    for graph_feat_name in self.graph_features_list:
                        if graph_feat_name in graph_features:
                            value = graph_features[graph_feat_name]
                            if np.isnan(value) or np.isinf(value):
                                value = 0.0
                            sparad_features.append(value)
                            feature_names_list.append(f"{feature_name}_{graph_feat_name}")
            
            sparad_features = np.array(sparad_features)
            
            feature_dict = {
                'feature_names': feature_names_list,
                'feature_values': sparad_features,
                'by_class': all_graph_features
            }
            
            # Cache final features
            if self.cache_manager and patient_id:
                df_features = pd.DataFrame({
                    'feature_name': feature_names_list,
                    'feature_value': sparad_features,
                    'subject_id': patient_id
                })
                self.cache_manager.save_final_features(patient_id, self.params, df_features)
                logger.info(f"✓ Saved final features to cache for {patient_id}")
            
            logger.info(f"\n✓ Completed: {len(sparad_features)} features extracted")
            
            gc.collect()
            
            return sparad_features, feature_dict
        
        except Exception as e:
            logger.error(f"SpaRad extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def export_to_dataframe(self, feature_dict, subject_id=None):
        """Export features to pandas DataFrame"""
        try:
            data = {
                'feature_name': feature_dict['feature_names'],
                'feature_value': feature_dict['feature_values']
            }
            
            if subject_id is not None:
                data['subject_id'] = subject_id
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"DataFrame export failed: {str(e)}")
            raise


# ============================================================================
# PATIENT PROCESSING FUNCTIONS
# ============================================================================

def process_single_patient_with_cache(patient_file, image_root, mask_root, 
                                      sparad_params, cache_manager):
    """Process single patient with caching support"""
    patient_id = None
    try:
        patient_id = patient_file.split('.')[0]
        
        # Check if already cached
        if cache_manager.exists_final_features(patient_id, sparad_params):
            logger.info(f"✓ Loading cached: {patient_id}")
            return cache_manager.load_final_features(patient_id, sparad_params)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing patient: {patient_id}")
        logger.info(f"{'='*80}")
        
        image_path = os.path.join(image_root, patient_file)
        mask_path = os.path.join(mask_root, patient_file)
        
        # Verify files exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Load data
        logger.info(f"Loading data from {image_path}")
        image, mask = load_nifti_data(image_path, mask_path)
        
        # Initialize framework with cache manager
        sparad = OptimizedSpaRadFramework(
            bin_width=sparad_params['bin_width'],
            window_size=sparad_params['window_size'],
            max_clusters=sparad_params['max_clusters'],
            n_threads=sparad_params.get('n_threads', 4),
            batch_size=sparad_params.get('batch_size', 100),
            cache_manager=cache_manager
        )
        
        # Extract features (will use cache if available)
        sparad_features, feature_dict = sparad.extract_sparad_descriptor(
            image, mask, patient_id=patient_id, save_intermediate=False
        )
        
        # Convert to dataframe
        df_features = sparad.export_to_dataframe(feature_dict, subject_id=patient_id)
        
        logger.info(f"✓ Completed: {patient_id}")
        
        # Clean up
        del image, mask, sparad, sparad_features, feature_dict
        gc.collect()
        
        return df_features
        
    except Exception as e:
        logger.error(f"✗ Failed: {patient_file} - Error: {str(e)}")
        logger.error(traceback.format_exc())
        
        if patient_id:
            return pd.DataFrame({
                'subject_id': [patient_id],
                'feature_name': ['ERROR'],
                'feature_value': [0.0]
            })
        return None


def process_batch_with_cache(patient_list, image_root, mask_root, 
                             output_path, sparad_params, cache_dir):
    """
    Process patients with caching and resume capability
    """
    try:
        # Initialize cache manager
        cache_manager = CacheManager(cache_dir)
        
        # Get already processed patients
        completed = cache_manager.get_completed_patients(sparad_params)
        patient_ids = [p.split('.')[0] for p in patient_list]
        
        # Filter to unprocessed patients
        remaining = [p for p in patient_list if p.split('.')[0] not in completed]
        
        logger.info(f"\n{'=' * 80}")
        logger.info("PROCESSING STATUS")
        logger.info(f"\n{'=' * 80}")
        logger.info("PROCESSING STATUS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total patients: {len(patient_list)}")
        logger.info(f"Already completed: {len(completed)}")
        logger.info(f"Remaining: {len(remaining)}")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"{'=' * 80}\n")
        
        if len(remaining) == 0:
            logger.info("All patients already processed!")
        else:
            logger.info(f"Processing {len(remaining)} remaining patients...")
            
            # Process sequentially for stability
            for idx, patient_file in enumerate(remaining):
                logger.info(f"\n{'='*80}")
                logger.info(f"Patient {idx+1}/{len(remaining)}: {patient_file.split('.')[0]}")
                logger.info(f"{'='*80}")
                
                try:
                    result = process_single_patient_with_cache(
                        patient_file, image_root, mask_root, 
                        sparad_params, cache_manager
                    )
                    
                    if result is not None and len(result) > 0:
                        logger.info(f"✓ Successfully processed")
                    else:
                        logger.warning(f"✗ No result returned")
                    
                except Exception as e:
                    logger.error(f"✗ Failed to process: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Force garbage collection
                gc.collect()
                
                # Progress update
                if (idx + 1) % 10 == 0:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Progress: {idx+1}/{len(remaining)} patients processed")
                    logger.info(f"{'='*80}\n")
        
        # Combine all results
        logger.info("\n" + "="*80)
        logger.info("COMBINING ALL RESULTS")
        logger.info("="*80)
        
        all_results = []
        successful = 0
        failed = 0
        
        for patient_file in tqdm(patient_list, desc="Loading results", ncols=100):
            patient_id = patient_file.split('.')[0]
            
            try:
                df = cache_manager.load_final_features(patient_id, sparad_params)
                if df is not None and len(df) > 0:
                    # Verify it's not an error result
                    if not (len(df) == 1 and df['feature_name'].iloc[0] == 'ERROR'):
                        all_results.append(df)
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Could not load result for {patient_id}: {str(e)}")
                failed += 1
        
        logger.info(f"\nProcessing summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {len(patient_list)}")
        
        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
            df_all.to_csv(output_path, index=False)
            logger.info(f"\n✓ Combined results saved to {output_path}")
            logger.info(f"  Total rows: {len(df_all)}")
            logger.info(f"  Unique patients: {df_all['subject_id'].nunique()}")
            
            # Verify uniqueness
            verify_feature_uniqueness(df_all)
            
            return df_all
        else:
            logger.error("No results to combine!")
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Fatal error in process_batch_with_cache: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def verify_feature_uniqueness(df_all):
    """Verify that features are unique per patient"""
    logger.info("\n" + "="*80)
    logger.info("FEATURE UNIQUENESS VERIFICATION")
    logger.info("="*80)
    
    try:
        unique_patients = df_all['subject_id'].unique()
        n_patients = len(unique_patients)
        
        if n_patients < 2:
            logger.warning("Less than 2 patients, skipping uniqueness check")
            return
        
        # Sample up to 5 patients for comparison
        sample_size = min(5, n_patients)
        sample_patients = unique_patients[:sample_size]
        
        all_unique = True
        feature_hashes = {}
        
        for patient_id in sample_patients:
            patient_features = df_all[df_all['subject_id'] == patient_id]['feature_value'].values
            
            # Create hash of feature values
            feature_hash = hashlib.md5(patient_features.tobytes()).hexdigest()
            
            if feature_hash in feature_hashes.values():
                logger.error(f"⚠ Patient {patient_id} has IDENTICAL features to another patient!")
                all_unique = False
            else:
                feature_hashes[patient_id] = feature_hash
                logger.info(f"  {patient_id}: unique (hash: {feature_hash[:8]}...)")
        
        if all_unique:
            logger.info("\n✓ All sampled patients have UNIQUE features")
        else:
            logger.error("\n✗ DUPLICATE features detected! Check cache integrity.")
        
        # Show sample feature values
        logger.info(f"\nSample features from first patient ({sample_patients[0]}):")
        first_patient_features = df_all[df_all['subject_id'] == sample_patients[0]]['feature_value'].values[:10]
        logger.info(f"  {first_patient_features}")
        
        if len(sample_patients) > 1:
            logger.info(f"\nSample features from second patient ({sample_patients[1]}):")
            second_patient_features = df_all[df_all['subject_id'] == sample_patients[1]]['feature_value'].values[:10]
            logger.info(f"  {second_patient_features}")
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@contextmanager
def timer(name):
    """Simple timing context manager"""
    start = time.time()
    logger.info(f"Starting: {name}")
    yield
    elapsed = time.time() - start
    logger.info(f"Completed: {name} - Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")


def clear_cache_interactive(cache_dir):
    """Interactively clear cache"""
    cache_manager = CacheManager(cache_dir)
    
    # Count cached items
    feature_maps = len(list(cache_manager.feature_maps_dir.glob("*.pkl.gz")))
    final_features = len(list(cache_manager.final_features_dir.glob("*.csv")))
    
    logger.info(f"\n{'='*80}")
    logger.info("CACHE STATUS")
    logger.info(f"{'='*80}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Feature maps cached: {feature_maps}")
    logger.info(f"Final features cached: {final_features}")
    logger.info(f"{'='*80}\n")
    
    if feature_maps + final_features == 0:
        logger.info("Cache is empty.")
        return
    
    response = input("Do you want to clear the cache? (yes/no): ")
    if response.lower() == 'yes':
        cache_manager.clear_all_cache()
        logger.info("✓ Cache cleared successfully!")
    else:
        logger.info("Cache not cleared.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive error handling"""
    
    # Setup logging
    log_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f'sparad_processing_{timestamp}.log')
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        logger.info("="*80)
        logger.info("SPARAD FEATURE EXTRACTION WITH CACHING")
        logger.info("="*80)
        logger.info(f"Log file: {log_file_path}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Configuration
        # image_root = 'XXXyour path/Dataset/Dataset002_ECResponse/Image/'
        # mask_root = 'XXXyour path/Dataset/Dataset002_ECResponse/Mask/'
        # cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/cache/'
        # output_path = 'XXXyour path/Dataset/Dataset002_ECResponse/sparad_features_all_patients.csv'

        image_root = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Image/'
        mask_root = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Mask/'
        cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/FeatureMaps/'
        output_path = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/sparad_features_all_patients.csv'
        # Verify directories exist
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        if not os.path.exists(mask_root):
            raise FileNotFoundError(f"Mask directory not found: {mask_root}")
        
        # Parameters
        sparad_params = {
            'bin_width': 25,
            'window_size': 3,
            'max_clusters': 10,
            'n_threads': 512,
            'batch_size': 2048,
            'feature_classes': ['glcm', 'glrlm', 'glszm']
        }
        
        # Get patient list
        patient_list = sorted([f for f in os.listdir(image_root) if f.endswith('.nii.gz')])
        
        if len(patient_list) == 0:
            raise ValueError(f"No .nii.gz files found in {image_root}")
        
        logger.info(f"\nConfiguration:")
        logger.info(f"  Image directory: {image_root}")
        logger.info(f"  Mask directory: {mask_root}")
        logger.info(f"  Cache directory: {cache_dir}")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Total patients: {len(patient_list)}")
        logger.info(f"  Parameters: {sparad_params}")
        
        # Display sample patients
        logger.info(f"\nFirst 5 patients:")
        for i, p in enumerate(patient_list[:5]):
            logger.info(f"  {i+1}. {p}")
        
        # Optional: Clear cache
        if '--clear-cache' in sys.argv:
            cache_manager = CacheManager(cache_dir)
            cache_manager.clear_all_cache()
            logger.info("\n✓ Cache cleared as requested\n")
        
        # Process with caching
        with timer("Total processing time"):
            df_all = process_batch_with_cache(
                patient_list=patient_list,
                image_root=image_root,
                mask_root=mask_root,
                output_path=output_path,
                sparad_params=sparad_params,
                cache_dir=cache_dir
            )
        
        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*80)
        
        # Final verification
        if os.path.exists(output_path):
            logger.info(f"\nOutput file verification:")
            logger.info(f"  Path: {output_path}")
            logger.info(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            
            if len(df_all) > 0:
                logger.info(f"  Total rows: {len(df_all)}")
                logger.info(f"  Unique subjects: {df_all['subject_id'].nunique()}")
                logger.info(f"  Features per subject: {len(df_all[df_all['subject_id'] == df_all['subject_id'].iloc[0]])}")
        else:
            logger.warning(f"Output file not created: {output_path}")
        
        logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"FATAL ERROR")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        logger.info(f"\nLog saved to: {log_file_path}")
        # Remove file handler to avoid duplicate logs
        logger.removeHandler(file_handler)
        file_handler.close()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Prevent multiprocessing issues
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
#    # Check for command line arguments
#    if '--help' in sys.argv or '-h' in sys.argv:
#        print("""
#SpaRad Feature Extraction with Caching
#
#Usage:
#    python script.py [OPTIONS]
#
#Options:
#    --help, -h          Show this help message
#    --clear-cache       Clear all cached data before processing
#    --verify-cache      Verify cache integrity without processing
#    --stats             Show cache statistics
#
#Examples:
#    # Normal run (uses cache if available)
#    python script.py
#    
#    # Clear cache and start fresh
#    python script.py --clear-cache
#    
#    # Check cache status
#    python script.py --verify-cache
#        """)
#        sys.exit(0)
#    
#    if '--verify-cache' in sys.argv:
#        cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/cache/'
#        cache_manager = CacheManager(cache_dir)
#        
#        feature_maps = len(list(cache_manager.feature_maps_dir.glob("*.pkl.gz")))
#        final_features = len(list(cache_manager.final_features_dir.glob("*.csv")))
#        
#        print(f"\n{'='*80}")
#        print("CACHE STATISTICS")
#        print(f"{'='*80}")
#        print(f"Cache directory: {cache_dir}")
#        print(f"Feature maps cached: {feature_maps}")
#        print(f"Final features cached: {final_features}")
#        
#        # List some cached patients
#        if final_features > 0:
#            print(f"\nSample cached patients:")
#            for i, cache_file in enumerate(list(cache_manager.final_features_dir.glob("*.csv"))[:5]):
#                print(f"  {i+1}. {cache_file.stem}")
#        
#        print(f"{'='*80}\n")
#        sys.exit(0)
#    
#    if '--stats' in sys.argv:
#        cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/cache/'
#        output_path = 'XXXyour path/Dataset/Dataset002_ECResponse/sparad_features_all_patients.csv'
#        
#        cache_manager = CacheManager(cache_dir)
#        
#        print(f"\n{'='*80}")
#        print("DETAILED CACHE STATISTICS")
#        print(f"{'='*80}")
#        
#        # Cache statistics
#        feature_maps_files = list(cache_manager.feature_maps_dir.glob("*.pkl.gz"))
#        final_features_files = list(cache_manager.final_features_dir.glob("*.csv"))
#        metadata_files = list(cache_manager.metadata_dir.glob("*.json"))
#        
#        print(f"\nCache directory: {cache_dir}")
#        print(f"  Feature maps: {len(feature_maps_files)} files")
#        print(f"  Final features: {len(final_features_files)} files")
#        print(f"  Metadata: {len(metadata_files)} files")
#        
#        # Calculate total size
#        total_size = 0
#        for file_list in [feature_maps_files, final_features_files, metadata_files]:
#            for f in file_list:
#                total_size += f.stat().st_size
#        
#        print(f"  Total cache size: {total_size / (1024*1024):.2f} MB")
#        
#        # Output file statistics
#        if os.path.exists(output_path):
#            print(f"\nOutput file: {output_path}")
#            print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
#            
#            try:
#                df = pd.read_csv(output_path)
#                print(f"  Rows: {len(df)}")
#                print(f"  Unique patients: {df['subject_id'].nunique()}")
#            except:
#                print(f"  (Could not read file)")
#        else:
#            print(f"\nOutput file not found: {output_path}")
#        
#        # Sample cached patient info
#        if len(metadata_files) > 0:
#            print(f"\nSample cached patients (showing up to 5):")
#            for i, metadata_file in enumerate(metadata_files[:5]):
#                try:
#                    with open(metadata_file, 'r') as f:
#                        metadata = json.load(f)
#                    print(f"  {i+1}. {metadata['patient_id']}")
#                    print(f"      Features: {metadata['n_features']}")
#                    print(f"      Cached: {metadata['timestamp']}")
#                except:
#                    print(f"  {i+1}. {metadata_file.stem} (could not read metadata)")
#        
#        print(f"{'='*80}\n")
#        sys.exit(0)
    
    # Run main processing
    main()