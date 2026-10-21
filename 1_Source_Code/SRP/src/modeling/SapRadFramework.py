import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from scipy.ndimage import generic_filter
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
import networkx as nx
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager, set_start_method
from functools import partial
import os
from DataLoader import load_nifti_data
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
from typing import Dict, List, Tuple, Optional
import pickle
import gc
import traceback
from datetime import datetime

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


class OptimizedSpaRadFramework:
    """
    Optimized Spatial Radiomics Framework with robustness improvements
    """
    
    def __init__(self, 
                 bin_width=25,
                 window_size=7,
                 max_clusters=10,
                 feature_classes=['glcm', 'glrlm', 'glszm'],
                 graph_features_list=None,
                 n_threads=4,
                 batch_size=100,
                 use_caching=False,  # Changed default to False to avoid memory issues
                 max_cache_size=10000):  # Limit cache size
        
        self.bin_width = bin_width
        self.window_size = window_size
        self.max_clusters = max_clusters
        self.feature_classes = feature_classes
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.use_caching = use_caching
        self.max_cache_size = max_cache_size
        
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
        
        # Limited cache
        self._cache = {} if use_caching else None
        self._cache_access_count = {}

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

    def _manage_cache(self):
        """Manage cache size to prevent memory overflow"""
        if self._cache is None or len(self._cache) <= self.max_cache_size:
            return
        
        # Remove least recently used items
        sorted_items = sorted(self._cache_access_count.items(), key=lambda x: x[1])
        items_to_remove = len(self._cache) - self.max_cache_size
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self._cache:
                del self._cache[key]
            if key in self._cache_access_count:
                del self._cache_access_count[key]

    def _extract_window_features_batch(self, coords_batch, image_array, mask_array, 
                                      image_sitk, pad_size):
        """
        Extract features for a batch of coordinates with improved error handling
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
                
                # Create cache key if caching enabled
                window_key = None
                if self.use_caching:
                    window_key = (z_min, z_max, y_min, y_max, x_min, x_max)
                    if window_key in self._cache:
                        cached_features = self._cache[window_key]
                        self._cache_access_count[window_key] = self._cache_access_count.get(window_key, 0) + 1
                        for fc in self.feature_classes:
                            for fn in self.feature_selections[fc]:
                                results[fc][fn][idx] = cached_features[fc][fn]
                        continue
                
                # Convert to SimpleITK
                local_image_sitk = sitk.GetImageFromArray(local_image.astype(np.float32))
                local_image_sitk.SetSpacing(image_sitk.GetSpacing())
                local_mask_sitk = sitk.GetImageFromArray(local_mask.astype(np.uint8))
                local_mask_sitk.CopyInformation(local_image_sitk)
                
                # Extract features for all classes
                window_features = {}
                for feature_class in self.feature_classes:
                    min_voxels = 27 if feature_class == 'glszm' else 10
                    
                    if np.sum(local_mask) < min_voxels:
                        for fn in self.feature_selections[feature_class]:
                            results[feature_class][fn][idx] = 0.0
                        window_features[feature_class] = {fn: 0.0 for fn in self.feature_selections[feature_class]}
                        continue
                    
                    try:
                        features = self.extractors[feature_class].execute(
                            local_image_sitk, local_mask_sitk
                        )
                        
                        window_features[feature_class] = {}
                        for fn in self.feature_selections[feature_class]:
                            feature_key = f'original_{feature_class}_{fn}'
                            value = float(features.get(feature_key, 0.0))
                            
                            # Handle NaN and inf
                            if np.isnan(value) or np.isinf(value):
                                value = 0.0
                            
                            results[feature_class][fn][idx] = value
                            window_features[feature_class][fn] = value
                    
                    except Exception as e:
                        logger.warning(f"Feature extraction failed for {feature_class} at coord {coord}: {str(e)}")
                        for fn in self.feature_selections[feature_class]:
                            results[feature_class][fn][idx] = 0.0
                        window_features[feature_class] = {fn: 0.0 for fn in self.feature_selections[feature_class]}
                
                # Cache results
                if self.use_caching and window_key is not None:
                    self._cache[window_key] = window_features
                    self._cache_access_count[window_key] = 1
                    self._manage_cache()
            
            except Exception as e:
                logger.error(f"Critical error processing coord {coord}: {str(e)}")
                # Fill with zeros
                for fc in self.feature_classes:
                    for fn in self.feature_selections[fc]:
                        results[fc][fn][idx] = 0.0
        
        return results

    def extract_all_feature_maps(self, image, mask):
        """
        Optimized feature map extraction with improved memory management
        """
        logger.info("=" * 80)
        logger.info("EXTRACTING SPATIAL RADIOMICS FEATURE MAPS (OPTIMIZED)")
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
                
                # Use as_completed for better error handling
                futures = [executor.submit(extract_func, batch) for batch in batches]
                batch_results = []
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc="Extracting features", ncols=100):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per batch
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
                        # Create empty result
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
                # All values are essentially the same
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
                        n_init=3,  # Reduced further
                        max_iter=50,
                        random_state=42,
                        tol=1e-2
                    )
                    gmm.fit(feature_values_std)
                    bic = gmm.bic(feature_values_std)
                    bic_scores.append(bic)
                    
                    # Early stopping
                    if bic < best_bic:
                        best_bic = bic
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            break
                except Exception as e:
                    logger.warning(f"GMM fitting failed for n={n}: {str(e)}")
                    bic_scores.append(np.inf)
            
            if not bic_scores or all(np.isinf(bic_scores)):
                # Clustering failed, return single cluster
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
                logger.warning(f"Final GMM fitting failed: {str(e)}")
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
                    'centroid_mm': centroid,
                    'coords': coords,
                    'feature_values': feature_values,
                    'size': len(feature_values),
                    'mean': float(np.mean(feature_values)),
                }
                
                G.add_node(
                    int(cluster_id),
                    centroid=centroid,
                    size=len(feature_values),
                    mean_feature=float(np.mean(feature_values))
                )
            
            valid_cluster_ids = list(cluster_properties.keys())
            n_valid = len(valid_cluster_ids)
            
            if n_valid < 2:
                return G, {'n_clusters': n_valid, 'insufficient_clusters': True}
            
            # Compute distances with timeout protection
            emd_matrix = np.zeros((n_valid, n_valid))
            spatial_matrix = np.zeros((n_valid, n_valid))
            
            for i in range(n_valid):
                for j in range(i+1, n_valid):
                    ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                    
                    # EMD with timeout
                    try:
                        emd = wasserstein_distance(
                            cluster_properties[ci]['feature_values'],
                            cluster_properties[cj]['feature_values']
                        )
                        if np.isnan(emd) or np.isinf(emd):
                            emd = 0.0
                    except Exception as e:
                        logger.warning(f"EMD calculation failed for clusters {ci}-{cj}: {str(e)}")
                        emd = 0.0
                    
                    emd_matrix[i, j] = emd_matrix[j, i] = emd
                    
                    # Spatial distance
                    if use_spatial_distance:
                        try:
                            spatial_dist = np.linalg.norm(
                                cluster_properties[ci]['centroid_mm'] - 
                                cluster_properties[cj]['centroid_mm']
                            )
                            if np.isnan(spatial_dist) or np.isinf(spatial_dist):
                                spatial_dist = 0.0
                        except:
                            spatial_dist = 0.0
                    else:
                        spatial_dist = 0.0
                    
                    spatial_matrix[i, j] = spatial_matrix[j, i] = spatial_dist
            
            # Normalize distances
            emd_values = emd_matrix[np.triu_indices(n_valid, k=1)]
            emd_std = np.std(emd_values)
            emd_mean = np.mean(emd_values)
            
            if emd_std > 1e-10:
                emd_matrix_norm = (emd_matrix - emd_mean) / emd_std
                emd_min, emd_max = np.nanmin(emd_matrix_norm), np.nanmax(emd_matrix_norm)
                if emd_max > emd_min:
                    emd_matrix_norm = (emd_matrix_norm - emd_min) / (emd_max - emd_min)
                else:
                    emd_matrix_norm = np.zeros_like(emd_matrix)
            else:
                emd_matrix_norm = np.zeros_like(emd_matrix)
            
            if use_spatial_distance:
                spatial_values = spatial_matrix[np.triu_indices(n_valid, k=1)]
                spatial_std = np.std(spatial_values)
                spatial_mean = np.mean(spatial_values)
                
                if spatial_std > 1e-10:
                    spatial_matrix_norm = (spatial_matrix - spatial_mean) / spatial_std
                    spatial_min, spatial_max = np.nanmin(spatial_matrix_norm), np.nanmax(spatial_matrix_norm)
                    if spatial_max > spatial_min:
                        spatial_matrix_norm = (spatial_matrix_norm - spatial_min) / (spatial_max - spatial_min)
                    else:
                        spatial_matrix_norm = np.zeros_like(spatial_matrix)
                else:
                    spatial_matrix_norm = np.zeros_like(spatial_matrix)
            else:
                spatial_matrix_norm = np.zeros_like(emd_matrix)
            
            # Replace NaNs and infs
            emd_matrix_norm = np.nan_to_num(emd_matrix_norm, nan=0.0, posinf=0.0, neginf=0.0)
            spatial_matrix_norm = np.nan_to_num(spatial_matrix_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
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
                    graph_type = 'mst'
            
            if graph_type == 'complete':
                for i in range(n_valid):
                    for j in range(i+1, n_valid):
                        ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                        similarity = 1.0 - combined_matrix[i, j]
                        weight = float(similarity) if edge_type == 'weighted' else 1.0
                        
                        G.add_edge(ci, cj, weight=weight, similarity=float(similarity))
                        edges_added += 1
            
            elif graph_type == 'threshold':
                for i in range(n_valid):
                    for j in range(i+1, n_valid):
                        similarity = 1.0 - combined_matrix[i, j]
                        if similarity > threshold:
                            ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                            weight = float(similarity) if edge_type == 'weighted' else 1.0
                            G.add_edge(ci, cj, weight=weight, similarity=float(similarity))
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
                            G.add_edge(ci, cj, weight=weight, similarity=float(similarity))
                            edges_added += 1
            
            elif graph_type == 'mst':
                try:
                    # MST construction
                    G_complete = nx.Graph()
                    for i in range(n_valid):
                        for j in range(i+1, n_valid):
                            ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                            G_complete.add_edge(ci, cj, weight=combined_matrix[i, j])
                    
                    mst = nx.minimum_spanning_tree(G_complete, weight='weight')
                    
                    for ci, cj, data in mst.edges(data=True):
                        similarity = 1.0 - data['weight']
                        weight = float(similarity) if edge_type == 'weighted' else 1.0
                        G.add_edge(ci, cj, weight=weight, similarity=float(similarity))
                        edges_added += 1
                except Exception as e:
                    logger.warning(f"MST construction failed: {str(e)}, falling back to complete graph")
                    graph_type = 'complete'
                    for i in range(n_valid):
                        for j in range(i+1, n_valid):
                            ci, cj = valid_cluster_ids[i], valid_cluster_ids[j]
                            similarity = 1.0 - combined_matrix[i, j]
                            weight = float(similarity) if edge_type == 'weighted' else 1.0
                            G.add_edge(ci, cj, weight=weight, similarity=float(similarity))
                            edges_added += 1
            
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
                        G.add_edge(node, nearest, weight=weight, similarity=float(similarity))
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
            # Return minimal graph
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
                logger.warning(f"Degree calculation failed: {str(e)}")
            
            # Density
            try:
                features['density'] = nx.density(G)
            except Exception as e:
                logger.warning(f"Density calculation failed: {str(e)}")
            
            # Clustering coefficient
            try:
                features['clustering_coefficient'] = nx.average_clustering(G)
            except Exception as e:
                logger.warning(f"Clustering coefficient calculation failed: {str(e)}")
            
            # Path features (only for connected graphs or largest component)
            try:
                if nx.is_connected(G):
                    features['average_path_length'] = nx.average_shortest_path_length(G)
                    features['diameter'] = nx.diameter(G)
                else:
                    # Use largest connected component
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc).copy()
                    if subgraph.number_of_nodes() > 1:
                        features['average_path_length'] = nx.average_shortest_path_length(subgraph)
                        features['diameter'] = nx.diameter(subgraph)
            except Exception as e:
                logger.warning(f"Path features calculation failed: {str(e)}")
            
            # Assortativity - with robust handling
            try:
                if G.number_of_edges() > 0 and features['degree_std'] > 1e-10:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        assortativity = nx.degree_assortativity_coefficient(G)
                        if not (np.isnan(assortativity) or np.isinf(assortativity)):
                            features['assortativity'] = float(assortativity)
            except Exception as e:
                logger.warning(f"Assortativity calculation failed: {str(e)}")
            
            # Centrality measures (with size limit and timeout)
            if G.number_of_nodes() < 100:
                # Betweenness centrality
                try:
                    betweenness = nx.betweenness_centrality(G)
                    betweenness_values = np.array(list(betweenness.values()))
                    features['betweenness_mean'] = float(np.mean(betweenness_values))
                    features['betweenness_std'] = float(np.std(betweenness_values))
                except Exception as e:
                    logger.warning(f"Betweenness centrality calculation failed: {str(e)}")
                
                # Closeness centrality
                try:
                    closeness = nx.closeness_centrality(G)
                    closeness_values = np.array(list(closeness.values()))
                    features['closeness_mean'] = float(np.mean(closeness_values))
                    features['closeness_std'] = float(np.std(closeness_values))
                except Exception as e:
                    logger.warning(f"Closeness centrality calculation failed: {str(e)}")
                
                # Eigenvector centrality
                try:
                    eigenvector = nx.eigenvector_centrality(G, max_iter=100, tol=1e-3)
                    eigenvector_values = np.array(list(eigenvector.values()))
                    features['eigenvector_mean'] = float(np.mean(eigenvector_values))
                except Exception as e:
                    logger.warning(f"Eigenvector centrality calculation failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Graph feature extraction error: {str(e)}")
        
        # Ensure all features are present and valid
        for feat in self.graph_features_list:
            if feat not in features:
                features[feat] = 0.0
            elif np.isnan(features[feat]) or np.isinf(features[feat]):
                features[feat] = 0.0
        
        return features

    def process_single_feature_map(self, feature_map, mask, feature_name):
        """Process single feature map with optimized pipeline and error handling"""
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
            # Return zero features
            return {feat: 0.0 for feat in self.graph_features_list}

    def extract_sparad_descriptor(self, image, mask, save_intermediate=False):
        """
        Complete SpaRad pipeline with comprehensive error handling
        """
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZED SPARAD DESCRIPTOR EXTRACTION")
        logger.info("=" * 80)
        
        try:
            # Clear cache for new image
            if self.use_caching and self._cache is not None:
                self._cache.clear()
                self._cache_access_count.clear()
            
            # Extract all feature maps
            all_feature_maps = self.extract_all_feature_maps(image, mask)
            
            # Convert mask
            if isinstance(mask, sitk.Image):
                mask_array = sitk.GetArrayFromImage(mask)
            else:
                mask_array = mask
            
            # Process all feature maps
            all_graph_features = {}
            
            total_features = sum(len(maps) for maps in all_feature_maps.values())
            feature_count = 0
            
            logger.info(f"\nProcessing {total_features} feature maps...")
            
            for feature_class in self.feature_classes:
                class_graph_features = {}
                feature_maps = all_feature_maps[feature_class]
                
                # Process feature maps sequentially to avoid threading issues with NetworkX
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
                    
                    feature_count += 1
                    
                    # Periodic garbage collection
                    if feature_count % 10 == 0:
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
                            # Ensure valid values
                            if np.isnan(value) or np.isinf(value):
                                value = 0.0
                            sparad_features.append(value)
                            feature_names_list.append(f"{feature_name}_{graph_feat_name}")
            
            sparad_features = np.array(sparad_features)
            
            logger.info(f"\nSpaRad descriptor shape: {sparad_features.shape}")
            
            feature_dict = {
                'feature_names': feature_names_list,
                'feature_values': sparad_features,
                'by_class': all_graph_features
            }
            
            # Final garbage collection
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


def process_single_patient_optimized(patient_file, image_root, mask_root, sparad_params):
    """Optimized single patient processing with robust error handling"""
    patient_id = None
    try:
        patient_id = patient_file.split('.')[0]
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
        
        # Initialize optimized framework
        sparad = OptimizedSpaRadFramework(
            bin_width=sparad_params['bin_width'],
            window_size=sparad_params['window_size'],
            max_clusters=sparad_params['max_clusters'],
            n_threads=sparad_params.get('n_threads', 4),
            batch_size=sparad_params.get('batch_size', 100),
            use_caching=sparad_params.get('use_caching', False)
        )
        
        # Extract features
        sparad_features, feature_dict = sparad.extract_sparad_descriptor(
            image, mask, save_intermediate=False
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
        
        # Return empty dataframe with patient ID
        if patient_id:
            return pd.DataFrame({
                'subject_id': [patient_id],
                'feature_name': ['ERROR'],
                'feature_value': [0.0]
            })
        return None


def process_with_checkpoints(patient_list, image_root, mask_root, 
                            output_path, sparad_params, 
                            checkpoint_dir='./checkpoints',
                            n_workers=4):  # Reduced default workers
    """
    Process patients with checkpoint support and improved error handling
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, 'processed.txt')
        error_file = os.path.join(checkpoint_dir, 'errors.txt')
        
        # Load already processed patients
        processed = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                processed = set(line.strip() for line in f)
            logger.info(f"Found {len(processed)} already processed patients")
        
        # Filter to unprocessed patients
        remaining = [p for p in patient_list if p.split('.')[0] not in processed]
        logger.info(f"Processing {len(remaining)} remaining patients out of {len(patient_list)} total...")
        
        if len(remaining) == 0:
            logger.info("All patients already processed!")
            # Load and combine all results
            all_results = []
            for patient_file in patient_list:
                patient_id = patient_file.split('.')[0]
                result_file = os.path.join(checkpoint_dir, f'{patient_id}.pkl')
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'rb') as f:
                            all_results.append(pickle.load(f))
                    except Exception as e:
                        logger.warning(f"Could not load result for {patient_id}: {str(e)}")
            
            if all_results:
                df_all = pd.concat(all_results, ignore_index=True)
                df_all.to_csv(output_path, index=False)
                logger.info(f"Combined results saved to {output_path}")
            return
        
        # Process sequentially to avoid multiprocessing issues
        logger.info("Processing patients sequentially for stability...")
        
        for idx, patient_file in enumerate(remaining):
            logger.info(f"\n{'='*80}")
            logger.info(f"Patient {idx+1}/{len(remaining)}")
            logger.info(f"{'='*80}")
            
            patient_id = patient_file.split('.')[0]
            result_file = os.path.join(checkpoint_dir, f'{patient_id}.pkl')
            
            try:
                # Process patient
                result = process_single_patient_optimized(
                    patient_file, image_root, mask_root, sparad_params
                )
                
                if result is not None and len(result) > 0:
                    # Save individual result
                    with open(result_file, 'wb') as rf:
                        pickle.dump(result, rf)
                    
                    # Update checkpoint
                    with open(checkpoint_file, 'a') as f:
                        f.write(f"{patient_id}\n")
                        f.flush()
                    
                    logger.info(f"✓ Saved result for {patient_id}")
                else:
                    # Log error
                    with open(error_file, 'a') as f:
                        f.write(f"{patient_id}: No result returned\n")
                        f.flush()
                    logger.warning(f"✗ No result for {patient_id}")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {patient_id}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Log error
                with open(error_file, 'a') as f:
                    f.write(f"{patient_id}: {str(e)}\n")
                    f.flush()
            
            # Force garbage collection after each patient
            gc.collect()
            
            # Periodic progress save
            if (idx + 1) % 10 == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"Progress: {idx+1}/{len(remaining)} patients processed")
                logger.info(f"{'='*80}\n")
        
        # Combine all results
        logger.info("\nCombining all results...")
        all_results = []
        successful = 0
        failed = 0
        
        for patient_file in patient_list:
            patient_id = patient_file.split('.')[0]
            result_file = os.path.join(checkpoint_dir, f'{patient_id}.pkl')
            
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'rb') as f:
                        result = pickle.load(f)
                        if result is not None and len(result) > 0:
                            all_results.append(result)
                            successful += 1
                        else:
                            failed += 1
                except Exception as e:
                    logger.warning(f"Could not load result for {patient_id}: {str(e)}")
                    failed += 1
            else:
                failed += 1
        
        logger.info(f"\nProcessing summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {len(patient_list)}")
        
        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
            df_all.to_csv(output_path, index=False)
            logger.info(f"✓ Combined results saved to {output_path}")
            logger.info(f"  Total features extracted: {len(df_all)}")
        else:
            logger.error("No results to combine!")
        
    except Exception as e:
        logger.error(f"Fatal error in process_with_checkpoints: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """Simple timing context manager"""
    start = time.time()
    logger.info(f"Starting: {name}")
    yield
    elapsed = time.time() - start
    logger.info(f"Completed: {name} - Time: {elapsed:.2f}s ({elapsed/60:.2f}m)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive error handling"""
    
    # Setup logging to both file and console
    log_dir = 'XXXyour path/Dataset/Dataset001_PDL1Expression/'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, f'processing_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    try:
        logger.info("="*80)
        logger.info("SPARAD FEATURE EXTRACTION - OPTIMIZED VERSION")
        logger.info("="*80)
        logger.info(f"Log file: {log_file_path}")
        
        # Parameters
        image_root = 'XXXyour path/Dataset/Dataset001_PDL1Expression/imagesTr/'
        mask_root = 'XXXyour path/Dataset/Dataset001_PDL1Expression/labelsTr/'
        output_path = 'XXXyour path/Dataset/Dataset001_PDL1Expression/sparad_features_optimized.csv'
        checkpoint_dir = 'XXXyour path/Dataset/Dataset001_PDL1Expression/checkpoints'
        
        # Verify directories exist
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        if not os.path.exists(mask_root):
            raise FileNotFoundError(f"Mask directory not found: {mask_root}")
        
        sparad_params = {
            'bin_width': 25,
            'window_size': 7,
            'max_clusters': 10,
            'n_threads': 8,  # Reduced for stability
            'batch_size': 50,  # Reduced for memory efficiency
            'use_caching': False  # Disabled to prevent memory issues
        }
        
        # Get patient list
        patient_list = sorted([f for f in os.listdir(image_root) if f.endswith('.nii.gz')])
        
        if len(patient_list) == 0:
            raise ValueError(f"No .nii.gz files found in {image_root}")
        
        logger.info(f"\nConfiguration:")
        logger.info(f"  Image directory: {image_root}")
        logger.info(f"  Mask directory: {mask_root}")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Checkpoint directory: {checkpoint_dir}")
        logger.info(f"  Total patients: {len(patient_list)}")
        logger.info(f"  Parameters: {sparad_params}")
        
        # Display first few patients
        logger.info(f"\nFirst 5 patients:")
        for i, p in enumerate(patient_list[:5]):
            logger.info(f"  {i+1}. {p}")
        
        with timer("Total processing time"):
            process_with_checkpoints(
                patient_list=patient_list,
                image_root=image_root,
                mask_root=mask_root,
                output_path=output_path,
                sparad_params=sparad_params,
                checkpoint_dir=checkpoint_dir,
                n_workers=1  # Sequential processing
            )
        
        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*80)
        
        # Verify output
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            logger.info(f"\nOutput file verification:")
            logger.info(f"  Path: {output_path}")
            logger.info(f"  Rows: {len(df)}")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Unique subjects: {df['subject_id'].nunique() if 'subject_id' in df.columns else 'N/A'}")
        else:
            logger.warning(f"Output file not created: {output_path}")
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"FATAL ERROR")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        logger.info(f"\nLog saved to: {log_file_path}")


if __name__ == "__main__":
    # Prevent multiprocessing issues
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Run main
    main()