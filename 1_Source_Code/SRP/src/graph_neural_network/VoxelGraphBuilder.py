import os
from FMCIB_FeatureExtractor import VoxelWiseFMCIBExtractor
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import wasserstein_distance
import pickle
from sklearn.decomposition import PCA, IncrementalPCA

class DatasetWidePCAFitter:
    """
    Fit PCA on entire training dataset for consistent dimensionality reduction
    """
    
    def __init__(self, 
                 n_components: int = 64,
                 whiten: bool = False,
                 use_incremental: bool = True,
                 incremental_batch_size: int = 10000):
        """
        Args:
            n_components: Number of PCA components
            whiten: Whether to whiten the components
            use_incremental: Use IncrementalPCA for memory efficiency
            incremental_batch_size: Batch size for incremental PCA
        """
        self.n_components = n_components
        self.whiten = whiten
        self.use_incremental = use_incremental
        self.incremental_batch_size = incremental_batch_size
        self.pca_model = None
        
    def fit_on_dataset(self, 
                      embeddings_dir: str,
                      case_list: List[str] = None,
                      max_samples_per_case: int = None) -> PCA:
        """
        Fit PCA on embeddings from entire training dataset
        
        Args:
            embeddings_dir: Directory containing saved embeddings
            case_list: List of case names to include (None = all)
            max_samples_per_case: Maximum samples per case (for memory control)
            
        Returns:
            Fitted PCA model
        """
        print(f"\n{'='*70}")
        print("Fitting PCA on Entire Training Dataset")
        print(f"{'='*70}")
        print(f"PCA components: {self.n_components}")
        print(f"Whiten: {self.whiten}")
        print(f"Incremental: {self.use_incremental}")
        
        embeddings_dir = Path(embeddings_dir)
        
        # Get all embedding files
        if case_list is None:
            embedding_files = sorted(embeddings_dir.glob('*_embeddings.pkl'))
        else:
            embedding_files = [embeddings_dir / f"{case}_embeddings.pkl" for case in case_list]
            embedding_files = [f for f in embedding_files if f.exists()]
        
        print(f"\nFound {len(embedding_files)} embedding files")
        
        if self.use_incremental:
            # Use IncrementalPCA for memory efficiency (recommended for large datasets)
            self.pca_model = self._fit_incremental_pca(
                embedding_files, 
                max_samples_per_case
            )
        else:
            # Load all embeddings into memory (only for small datasets)
            self.pca_model = self._fit_standard_pca(
                embedding_files,
                max_samples_per_case
            )
        
        print(f"\n{'='*70}")
        print("PCA Fitting Complete")
        print(f"{'='*70}")
        print(f"Total variance explained: {self.pca_model.explained_variance_ratio_.sum():.4f}")
        print(f"Top 10 components variance: {self.pca_model.explained_variance_ratio_[:10]}")
        
        # Print variance thresholds
        explained_var_cumsum = np.cumsum(self.pca_model.explained_variance_ratio_)
        for threshold in [0.5, 0.75, 0.90, 0.95, 0.99]:
            n_comp_needed = np.argmax(explained_var_cumsum >= threshold) + 1
            print(f"  {threshold*100:.0f}% variance: {n_comp_needed} components")
        
        return self.pca_model
    
    def _fit_incremental_pca(self, 
                            embedding_files: List[Path],
                            max_samples_per_case: int = None) -> IncrementalPCA:
        """
        Fit IncrementalPCA by loading data in batches
        """
        print("\nFitting Incremental PCA...")
        
        # Initialize IncrementalPCA
        pca = IncrementalPCA(
            n_components=self.n_components,
            whiten=self.whiten,
            batch_size=self.incremental_batch_size
        )
        
        total_samples = 0
        buffer = []
        
        # Process each file
        for file_path in tqdm(embedding_files, desc="Processing files"):
            # Load embeddings
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data['embeddings']
            
            # Skip empty embeddings
            if len(embeddings) == 0:
                print(f"  Skipping {file_path.name} (empty)")
                continue
            
            # Subsample if needed
            if max_samples_per_case is not None and len(embeddings) > max_samples_per_case:
                indices = np.random.choice(len(embeddings), max_samples_per_case, replace=False)
                embeddings = embeddings[indices]
            
            total_samples += len(embeddings)
            buffer.append(embeddings)
            
            # Fit when buffer is large enough
            if sum(len(e) for e in buffer) >= self.incremental_batch_size:
                batch_data = np.vstack(buffer)
                pca.partial_fit(batch_data)
                buffer = []
        
        # Fit remaining data
        if buffer:
            batch_data = np.vstack(buffer)
            pca.partial_fit(batch_data)
        
        print(f"Total samples processed: {total_samples}")
        
        return pca
    
    def _fit_standard_pca(self,
                         embedding_files: List[Path],
                         max_samples_per_case: int = None) -> PCA:
        """
        Fit standard PCA by loading all data into memory
        """
        print("\nFitting Standard PCA...")
        print("Warning: This loads all embeddings into memory!")
        
        all_embeddings = []
        total_samples = 0
        
        for file_path in tqdm(embedding_files, desc="Loading embeddings"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data['embeddings']
            
            # Skip empty
            if len(embeddings) == 0:
                print(f"  Skipping {file_path.name} (empty)")
                continue
            
            # Subsample if needed
            if max_samples_per_case is not None and len(embeddings) > max_samples_per_case:
                indices = np.random.choice(len(embeddings), max_samples_per_case, replace=False)
                embeddings = embeddings[indices]
            
            all_embeddings.append(embeddings)
            total_samples += len(embeddings)
        
        # Stack all embeddings
        print("Stacking embeddings...")
        all_embeddings = np.vstack(all_embeddings)
        print(f"Total samples: {total_samples}")
        print(f"Memory usage: {all_embeddings.nbytes / 1e9:.2f} GB")
        
        # Fit PCA
        pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=42
        )
        pca.fit(all_embeddings)
        
        return pca
    
    def save_pca_model(self, output_path: str):
        """Save fitted PCA model"""
        if self.pca_model is None:
            raise ValueError("PCA model not fitted yet!")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pca_data = {
            'pca_model': self.pca_model,
            'n_components': self.n_components,
            'whiten': self.whiten,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'total_variance_explained': self.pca_model.explained_variance_ratio_.sum(),
            'use_incremental': self.use_incremental
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(pca_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\nSaved PCA model to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")
    
    def load_pca_model(self, input_path: str):
        """Load fitted PCA model"""
        with open(input_path, 'rb') as f:
            pca_data = pickle.load(f)
        
        self.pca_model = pca_data['pca_model']
        self.n_components = pca_data['n_components']
        self.whiten = pca_data['whiten']
        
        print(f"\nLoaded PCA model from: {input_path}")
        print(f"Components: {self.n_components}")
        print(f"Total variance explained: {pca_data['total_variance_explained']:.4f}")
        
        return self.pca_model
    
class VoxelGraphBuilder:
    """
    Build graph from voxel embeddings
    - Nodes: Individual voxels with embeddings
    - Edges: Connectivity based on spatial proximity
    - Edge weights: EMD (Earth Mover's Distance) between embeddings
    """
    
    def __init__(self, 
                 embeddings_dict: Dict,
                 connectivity: str = 'knn',
                 k_neighbors: int = 6,
                 spatial_distance_threshold: float = None,
                 use_spatial_features: bool = True,
                 normalize_embeddings: bool = True,
                 use_pca: bool = False,
                 pca_components: int = 64,
                 pca_whiten: bool = False):
        """
        Args:
            embeddings_dict: Dictionary from VoxelWiseFMCIBExtractor
            connectivity: 'knn', 'radius', or 'grid6/grid18/grid26'
            k_neighbors: Number of nearest neighbors for 'knn' mode
            spatial_distance_threshold: Max spatial distance for 'radius' mode (in mm)
            use_spatial_features: Include spatial coordinates in node features
            normalize_embeddings: Normalize embeddings before computing distances
            use_pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components (default: 64)
            pca_whiten: Whether to whiten PCA components
        """
        self.coordinates = embeddings_dict['coordinates']
        self.ct_shape = embeddings_dict['ct_shape']
        self.spacing = embeddings_dict.get('spacing', (1.0, 1.0, 1.0))
        self.original_embedding_dim = embeddings_dict['embedding_dim']
        
        self.connectivity = connectivity
        self.k_neighbors = k_neighbors
        self.spatial_distance_threshold = spatial_distance_threshold
        self.use_spatial_features = use_spatial_features
        self.normalize_embeddings = normalize_embeddings
        
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten
        self.pca_model = None
        
        # Apply PCA if requested
        if self.use_pca:
            print(f"\n{'='*70}")
            print("Applying PCA Dimensionality Reduction")
            print(f"{'='*70}")
            print(f"Original embedding dimension: {self.original_embedding_dim}")
            print(f"Target PCA components: {pca_components}")
            print(f"Number of samples: {len(embeddings_dict['embeddings'])}")
            print(f"PCA whitening: {pca_whiten}")
            
            self.embeddings, self.pca_model = self._apply_pca(
                embeddings_dict['embeddings'],
                n_components=pca_components,
                whiten=pca_whiten
            )
            self.embedding_dim = self.embeddings.shape[1]
            
            print(f"\nPCA Results:")
            print(f"  Reduced dimension: {self.embedding_dim}")
            print(f"  Explained variance ratio: {self.pca_model.explained_variance_ratio_.sum():.4f}")
            print(f"  Top 10 components variance: {self.pca_model.explained_variance_ratio_[:10]}")
        else:
            self.embeddings = embeddings_dict['embeddings']
            self.embedding_dim = self.original_embedding_dim
        
        # Normalize embeddings if requested (after PCA)
        if self.normalize_embeddings:
            print("\nNormalizing embeddings...")
            self.embeddings = self._normalize(self.embeddings)
        
        print(f"\n{'='*70}")
        print("Voxel Graph Builder Initialized")
        print(f"{'='*70}")
        print(f"Number of nodes (voxels): {len(self.coordinates)}")
        print(f"Final embedding dimension: {self.embedding_dim}")
        print(f"Connectivity type: {connectivity}")
        if connectivity == 'knn':
            print(f"K neighbors: {k_neighbors}")
        elif connectivity == 'radius':
            print(f"Spatial threshold: {spatial_distance_threshold} mm")
    
    def _apply_pca(self, 
                   embeddings: np.ndarray, 
                   n_components: int = 64,
                   whiten: bool = False) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA to reduce embedding dimensionality
        
        Args:
            embeddings: Original embeddings (N, D)
            n_components: Number of PCA components
            whiten: Whether to whiten the components
            
        Returns:
            Reduced embeddings (N, n_components), fitted PCA model
        """
        # Ensure n_components doesn't exceed min(n_samples, n_features)
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        if n_components > max_components:
            print(f"Warning: Requested {n_components} components, but maximum is {max_components}")
            n_components = max_components
        
        # Fit PCA
        print(f"Fitting PCA with {n_components} components...")
        pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
        
        # Print variance explained
        explained_var_cumsum = np.cumsum(pca.explained_variance_ratio_)
        print(f"\nVariance explained by components:")
        for i, threshold in enumerate([0.5, 0.75, 0.90, 0.95, 0.99]):
            n_comp_needed = np.argmax(explained_var_cumsum >= threshold) + 1
            print(f"  {threshold*100:.0f}% variance: {n_comp_needed} components")
        
        print(f"\nTotal variance explained by {n_components} components: "
              f"{pca.explained_variance_ratio_.sum():.4f}")
        
        return embeddings_reduced, pca
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    def _compute_spatial_distances(self) -> np.ndarray:
        """
        Compute pairwise spatial distances between voxels
        
        Returns:
            Distance matrix (N, N)
        """
        print("\nComputing spatial distances...")
        
        # Scale coordinates by spacing for physical distances
        coords_scaled = self.coordinates * np.array(self.spacing)
        
        # Compute pairwise Euclidean distances
        spatial_distances = cdist(coords_scaled, coords_scaled, metric='euclidean')
        
        return spatial_distances
    
    def _compute_emd_weights(self, node_i: int, node_j: int) -> float:
        """
        Compute EMD (Wasserstein distance) between two voxel embeddings
        
        Args:
            node_i: Index of first node
            node_j: Index of second node
            
        Returns:
            EMD distance (lower = more similar)
        """
        emb_i = self.embeddings[node_i]
        emb_j = self.embeddings[node_j]
        
        # Normalize to make them sum to 1 (like probability distributions)
        emb_i_normalized = np.abs(emb_i) / (np.sum(np.abs(emb_i)) + 1e-8)
        emb_j_normalized = np.abs(emb_j) / (np.sum(np.abs(emb_j)) + 1e-8)
        
        # Compute Wasserstein distance (1D EMD)
        emd = wasserstein_distance(
            np.arange(len(emb_i)), 
            np.arange(len(emb_j)),
            emb_i_normalized,
            emb_j_normalized
        )
        
        return emd
    
    def _compute_cosine_similarity(self, node_i: int, node_j: int) -> float:
        """
        Compute cosine similarity between embeddings
        
        Returns:
            Similarity in [0, 1] (higher = more similar)
        """
        emb_i = self.embeddings[node_i]
        emb_j = self.embeddings[node_j]
        
        similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def _compute_euclidean_distance(self, node_i: int, node_j: int) -> float:
        """
        Compute Euclidean distance between embeddings
        
        Returns:
            Distance (lower = more similar)
        """
        emb_i = self.embeddings[node_i]
        emb_j = self.embeddings[node_j]
        
        return np.linalg.norm(emb_i - emb_j)
    

    def build_graph(self, 
               edge_weight_method: str = 'euclidean',
               max_edges: Optional[int] = None,
               edge_weight_threshold: Optional[float] = None,
               use_vectorized: bool = True) -> nx.Graph:
        """
        Build graph with voxels as nodes and weighted edges
        
        Args:
            edge_weight_method: 'emd', 'cosine', 'euclidean'
            max_edges: Maximum number of edges to include
            edge_weight_threshold: Threshold for edge weights
            use_vectorized: Use vectorized computation (faster for euclidean/cosine)
            
        Returns:
            NetworkX graph
        """
        print(f"\n{'='*70}")
        print("Building Voxel Graph")
        print(f"{'='*70}")
        print(f"Edge weight method: {edge_weight_method}")
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with features
        print("\nAdding nodes...")
        for i in tqdm(range(len(self.coordinates)), desc="Nodes"):
            node_features = {
                'embedding': self.embeddings[i],
                'coordinate': self.coordinates[i],
                'position': tuple(self.coordinates[i])
            }
            
            if self.use_spatial_features:
                node_features['spatial_features'] = self.coordinates[i] * np.array(self.spacing)
            
            # Store PCA info if used
            if self.use_pca:
                node_features['pca_reduced'] = True
                node_features['original_dim'] = self.original_embedding_dim
                node_features['reduced_dim'] = self.embedding_dim
            
            G.add_node(i, **node_features)
        
        # Determine edges based on connectivity type
        print(f"\nDetermining edges (connectivity: {self.connectivity})...")
        
        if self.connectivity == 'knn':
            edges = self._get_knn_edges()
        elif self.connectivity == 'radius':
            edges = self._get_radius_edges()
        elif self.connectivity in ['grid6', 'grid18', 'grid26']:
            edges = self._get_grid_edges()
        else:
            raise ValueError(f"Unknown connectivity: {self.connectivity}")
        
        print(f"Number of potential edges: {len(edges)}")
        
        # Compute edge weights
        if use_vectorized and edge_weight_method in ['euclidean', 'cosine']:
            edge_weights = self._compute_edge_weights_vectorized(edges, edge_weight_method)
        else:
            edge_weights = self._compute_edge_weights_loop(edges, edge_weight_method)
        
        # Filter edges by threshold if specified
        if edge_weight_threshold is not None:
            if edge_weight_method == 'cosine':
                edge_weights = [(i, j, w) for i, j, w in edge_weights if w >= edge_weight_threshold]
            else:
                edge_weights = [(i, j, w) for i, j, w in edge_weights if w <= edge_weight_threshold]
            
            print(f"Edges after threshold filtering: {len(edge_weights)}")
        
        # Keep only top edges if max_edges specified
        if max_edges is not None and len(edge_weights) > max_edges:
            print(f"Keeping top {max_edges} edges...")
            
            if edge_weight_method == 'cosine':
                edge_weights = sorted(edge_weights, key=lambda x: x[2], reverse=True)[:max_edges]
            else:
                edge_weights = sorted(edge_weights, key=lambda x: x[2])[:max_edges]
        
        # Add edges to graph
        print("\nAdding edges to graph...")
        for i, j, weight in tqdm(edge_weights, desc="Adding edges"):
            spatial_dist = np.linalg.norm(
                (self.coordinates[i] - self.coordinates[j]) * np.array(self.spacing)
            )
            
            G.add_edge(i, j, 
                    weight=weight,
                    spatial_distance=spatial_dist,
                    embedding_weight=weight)
        
        print(f"\n{'='*70}")
        print("Graph Construction Complete")
        print(f"{'='*70}")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
        print(f"Graph density: {nx.density(G):.6f}")
        
        if G.number_of_edges() > 0:
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            print(f"\nEdge weight statistics:")
            print(f"  Min: {np.min(weights):.4f}")
            print(f"  Max: {np.max(weights):.4f}")
            print(f"  Mean: {np.mean(weights):.4f}")
            print(f"  Std: {np.std(weights):.4f}")
        
        return G


    def _compute_edge_weights_vectorized(self, 
                                        edges: List[Tuple[int, int]], 
                                        method: str) -> List[Tuple[int, int, float]]:
        """
        Vectorized edge weight computation (FAST)
        """
        print(f"\nComputing {len(edges)} edge weights ({method}) - vectorized...")
        
        edge_array = np.array(edges)
        i_indices = edge_array[:, 0]
        j_indices = edge_array[:, 1]
        
        emb_i = self.embeddings[i_indices]
        emb_j = self.embeddings[j_indices]
        
        if method == 'euclidean':
            weights = np.linalg.norm(emb_i - emb_j, axis=1)
        elif method == 'cosine':
            dot_product = np.sum(emb_i * emb_j, axis=1)
            norm_i = np.linalg.norm(emb_i, axis=1)
            norm_j = np.linalg.norm(emb_j, axis=1)
            weights = (dot_product / (norm_i * norm_j + 1e-8) + 1) / 2
        else:
            raise ValueError(f"Method {method} not supported for vectorized computation")
        
        return [(int(edges[idx][0]), int(edges[idx][1]), float(weights[idx])) 
                for idx in range(len(edges))]


    def _compute_edge_weights_loop(self, 
                                edges: List[Tuple[int, int]], 
                                method: str) -> List[Tuple[int, int, float]]:
        """
        Loop-based edge weight computation (for EMD and fallback)
        """
        print(f"\nComputing {len(edges)} edge weights ({method}) - batched loop...")
        
        edge_weights = []
        for i, j in tqdm(edges, desc="Edge weights"):
            if method == 'emd':
                weight = self._compute_emd_weights(i, j)
            elif method == 'cosine':
                weight = self._compute_cosine_similarity(i, j)
            elif method == 'euclidean':
                weight = self._compute_euclidean_distance(i, j)
            else:
                raise ValueError(f"Unknown edge weight method: {method}")
            
            edge_weights.append((i, j, weight))
        
        return edge_weights
        
    def _get_knn_edges(self) -> List[Tuple[int, int]]:
        """
        Get edges based on k-nearest neighbors (MEMORY-EFFICIENT)
        Uses sklearn's NearestNeighbors instead of computing full distance matrix
        """
        from sklearn.neighbors import NearestNeighbors
        
        print(f"Finding {self.k_neighbors} nearest neighbors for each voxel (efficient method)...")
        
        # Scale coordinates by spacing for physical distances
        coords_scaled = self.coordinates * np.array(self.spacing)
        
        n_samples = len(coords_scaled)
        
        # Adjust k_neighbors if necessary
        # We need k_neighbors + 1 because the point itself is included
        effective_k = min(self.k_neighbors, n_samples - 1)
        
        if effective_k < self.k_neighbors:
            print(f"WARNING: Only {n_samples} samples available, reducing k_neighbors from {self.k_neighbors} to {effective_k}")
        
        if effective_k < 1:
            print(f"ERROR: Not enough samples ({n_samples}) to create any edges!")
            return []
        
        # Build KNN index (uses KD-tree or Ball-tree - memory efficient)
        nbrs = NearestNeighbors(
            n_neighbors=effective_k + 1,  # +1 because it includes the point itself
            algorithm='auto',  # Automatically chooses best algorithm
            metric='euclidean',
            n_jobs=-1  # Use all available CPU cores
        ).fit(coords_scaled)
        
        # Find k nearest neighbors for all points at once
        print("Querying neighbors...")
        distances, indices = nbrs.kneighbors(coords_scaled)
        
        # Build edge list (avoid duplicates)
        print("Building edge list...")
        edges = set()
        for i in tqdm(range(len(self.coordinates)), desc="Creating edges"):
            # indices[i][0] is the point itself, skip it
            for neighbor_idx in indices[i][1:]:
                # Create edge with consistent ordering (smaller index first)
                edge = (min(i, neighbor_idx), max(i, neighbor_idx))
                edges.add(edge)
        
        edges_list = list(edges)
        print(f"Created {len(edges_list)} unique edges with k={effective_k}")
        
        return edges_list

    def _get_radius_edges(self) -> List[Tuple[int, int]]:
        """
        Get edges based on spatial radius (MEMORY-EFFICIENT)
        """
        from sklearn.neighbors import NearestNeighbors
        
        print(f"Finding neighbors within radius {self.spatial_distance_threshold} mm...")
        
        coords_scaled = self.coordinates * np.array(self.spacing)
        
        # Use radius_neighbors instead of computing full distance matrix
        nbrs = NearestNeighbors(
            radius=self.spatial_distance_threshold,
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1
        ).fit(coords_scaled)
        
        print("Querying neighbors...")
        distances, indices = nbrs.radius_neighbors(coords_scaled)
        
        # Build edge list
        print("Building edge list...")
        edges = set()
        for i in tqdm(range(len(self.coordinates)), desc="Creating edges"):
            for neighbor_idx in indices[i]:
                if neighbor_idx != i:  # Skip self
                    edge = (min(i, neighbor_idx), max(i, neighbor_idx))
                    edges.add(edge)
        
        edges_list = list(edges)
        print(f"Created {len(edges_list)} unique edges")
        
        return edges_list
    
    def _get_grid_edges(self) -> List[Tuple[int, int]]:
        """Get edges based on grid connectivity"""
        coord_to_idx = {tuple(coord): i for i, coord in enumerate(self.coordinates)}
        
        if self.connectivity == 'grid6':
            offsets = [
                (-1, 0, 0), (1, 0, 0),
                (0, -1, 0), (0, 1, 0),
                (0, 0, -1), (0, 0, 1)
            ]
        elif self.connectivity == 'grid18':
            offsets = []
            for d in [-1, 0, 1]:
                for h in [-1, 0, 1]:
                    for w in [-1, 0, 1]:
                        if abs(d) + abs(h) + abs(w) <= 2 and (d, h, w) != (0, 0, 0):
                            offsets.append((d, h, w))
        else:  # grid26
            offsets = []
            for d in [-1, 0, 1]:
                for h in [-1, 0, 1]:
                    for w in [-1, 0, 1]:
                        if (d, h, w) != (0, 0, 0):
                            offsets.append((d, h, w))
        
        edges = []
        for i, coord in enumerate(self.coordinates):
            for offset in offsets:
                neighbor_coord = tuple(coord + np.array(offset))
                
                if neighbor_coord in coord_to_idx:
                    j = coord_to_idx[neighbor_coord]
                    if i < j:
                        edges.append((i, j))
        
        return edges
    
    def save_graph(self, graph: nx.Graph, output_path: str, save_pca_model: bool = True):
        """
        Save graph to file (with optional PCA model)
        
        Args:
            graph: NetworkX graph
            output_path: Path to save graph
            save_pca_model: Whether to save the PCA model separately
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        suffix = output_path.suffix
        
        if suffix == '.gpickle' or suffix == '.pkl':
            # Use pickle directly (works with all NetworkX versions)
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        elif suffix == '.graphml':
            # GraphML format (remove numpy arrays)
            G_copy = graph.copy()
            for node in G_copy.nodes():
                if 'embedding' in G_copy.nodes[node]:
                    del G_copy.nodes[node]['embedding']
                if 'spatial_features' in G_copy.nodes[node]:
                    del G_copy.nodes[node]['spatial_features']
                if 'coordinate' in G_copy.nodes[node]:
                    # Convert to string
                    G_copy.nodes[node]['coordinate'] = str(G_copy.nodes[node]['coordinate'])
            
            nx.write_graphml(G_copy, output_path)
            
        elif suffix == '.gexf':
            # GEXF format
            G_copy = graph.copy()
            for node in G_copy.nodes():
                if 'embedding' in G_copy.nodes[node]:
                    del G_copy.nodes[node]['embedding']
                if 'spatial_features' in G_copy.nodes[node]:
                    del G_copy.nodes[node]['spatial_features']
                if 'coordinate' in G_copy.nodes[node]:
                    del G_copy.nodes[node]['coordinate']
            nx.write_gexf(G_copy, output_path)
    
        else:
            # Default to pickle for any other extension
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\nSaved graph to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")
        
        # Save PCA model if applicable
        if save_pca_model and self.use_pca and self.pca_model is not None:
            pca_path = output_path.with_name(output_path.stem + '_pca_model.pkl')
            with open(pca_path, 'wb') as f:
                pickle.dump({
                    'pca_model': self.pca_model,
                    'original_dim': self.original_embedding_dim,
                    'reduced_dim': self.embedding_dim,
                    'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
                    'n_components': self.pca_components,
                    'whiten': self.pca_whiten
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved PCA model to: {pca_path}")
            print(f"PCA file size: {pca_path.stat().st_size / 1e6:.2f} MB")

    def load_graph(self, input_path: str) -> nx.Graph:
        """Load graph from file"""
        input_path = Path(input_path)
        suffix = input_path.suffix
        
        if suffix == '.gpickle' or suffix == '.pkl':
            # Use pickle directly
            with open(input_path, 'rb') as f:
                graph = pickle.load(f)
        elif suffix == '.graphml':
            graph = nx.read_graphml(input_path)
        elif suffix == '.gexf':
            graph = nx.read_gexf(input_path)
        else:
            # Default to pickle
            with open(input_path, 'rb') as f:
                graph = pickle.load(f)
        
        print(f"Loaded graph from: {input_path}")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        
        # Load PCA model if it exists
        pca_path = input_path.with_name(input_path.stem + '_pca_model.pkl')
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                pca_data = pickle.load(f)
                self.pca_model = pca_data['pca_model']
                self.use_pca = True
                print(f"\nLoaded PCA model from: {pca_path}")
                print(f"  Original dim: {pca_data['original_dim']}")
                print(f"  Reduced dim: {pca_data['reduced_dim']}")
                print(f"  Explained variance: {pca_data['explained_variance_ratio'].sum():.4f}")
        
        return graph
    
    def get_pca_info(self) -> Dict:
        """
        Get information about the PCA transformation
        
        Returns:
            Dictionary with PCA statistics
        """
        if not self.use_pca or self.pca_model is None:
            return {'pca_applied': False}
        
        info = {
            'pca_applied': True,
            'original_dim': self.original_embedding_dim,
            'reduced_dim': self.embedding_dim,
            'n_components': self.pca_components,
            'whiten': self.pca_whiten,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'total_explained_variance': self.pca_model.explained_variance_ratio_.sum(),
            'singular_values': self.pca_model.singular_values_,
            'components': self.pca_model.components_
        }
        
        return info
    
    def plot_pca_analysis(self, output_path: str):
        """
        Create visualization of PCA analysis
        
        Args:
            output_path: Path to save the plot
        """
        if not self.use_pca or self.pca_model is None:
            print("PCA was not applied, cannot create PCA analysis plot")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Explained variance ratio
        ax = axes[0, 0]
        n_comp = len(self.pca_model.explained_variance_ratio_)
        ax.bar(range(1, n_comp + 1), self.pca_model.explained_variance_ratio_)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Variance Explained by Each Component')
        ax.grid(True, alpha=0.3)
        
        # 2. Cumulative explained variance
        ax = axes[0, 1]
        cumsum = np.cumsum(self.pca_model.explained_variance_ratio_)
        ax.plot(range(1, n_comp + 1), cumsum, 'b-', linewidth=2)
        ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Variance Explained')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Singular values (scree plot)
        ax = axes[1, 0]
        ax.plot(range(1, n_comp + 1), self.pca_model.singular_values_, 'go-')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Singular Value')
        ax.set_title('Scree Plot')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 4. Component loading heatmap (first 20 components)
        ax = axes[1, 1]
        n_show = min(20, self.pca_model.components_.shape[0])
        im = ax.imshow(self.pca_model.components_[:n_show, :50], 
                      aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_xlabel('Original Feature Dimension')
        ax.set_ylabel('Principal Component')
        ax.set_title(f'First {n_show} Component Loadings (first 50 features)')
        plt.colorbar(im, ax=ax, label='Loading')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved PCA analysis plot to: {output_path}")

class VoxelGraphBuilderWithPreFittedPCA(VoxelGraphBuilder):
    """
    Extended VoxelGraphBuilder that uses pre-fitted PCA model
    """
    
    def __init__(self, 
                 embeddings_dict: Dict,
                 pca_model: PCA = None,
                 **kwargs):
        """
        Args:
            embeddings_dict: Dictionary from VoxelWiseFMCIBExtractor
            pca_model: Pre-fitted PCA model (if None, will fit on this data)
            **kwargs: Other arguments for VoxelGraphBuilder
        """
        # Store original embeddings
        self.original_embeddings = embeddings_dict['embeddings']
        self.coordinates = embeddings_dict['coordinates']
        self.ct_shape = embeddings_dict['ct_shape']
        self.spacing = embeddings_dict.get('spacing', (1.0, 1.0, 1.0))
        self.original_embedding_dim = embeddings_dict['embedding_dim']
        
        # Extract PCA parameters from kwargs
        use_pca = kwargs.pop('use_pca', False)
        pca_components = kwargs.pop('pca_components', 64)
        pca_whiten = kwargs.pop('pca_whiten', False)
        
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten
        
        # Apply PCA transformation
        if use_pca:
            if pca_model is None:
                raise ValueError("use_pca=True but no pca_model provided!")
            
            print(f"\n{'='*70}")
            print("Applying Pre-fitted PCA Transformation")
            print(f"{'='*70}")
            print(f"Original embedding dimension: {self.original_embedding_dim}")
            print(f"PCA components: {pca_model.n_components_}")
            
            # Transform embeddings
            self.embeddings = pca_model.transform(self.original_embeddings)
            self.pca_model = pca_model
            self.embedding_dim = self.embeddings.shape[1]
            
            print(f"Transformed embedding dimension: {self.embedding_dim}")
            print(f"Using pre-fitted PCA model")
        else:
            self.embeddings = self.original_embeddings
            self.embedding_dim = self.original_embedding_dim
            self.pca_model = None
        
        # Initialize remaining attributes
        self.connectivity = kwargs.get('connectivity', 'knn')
        self.k_neighbors = kwargs.get('k_neighbors', 6)
        self.spatial_distance_threshold = kwargs.get('spatial_distance_threshold', None)
        self.use_spatial_features = kwargs.get('use_spatial_features', True)
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        
        # Normalize if requested
        if self.normalize_embeddings:
            print("\nNormalizing embeddings...")
            self.embeddings = self._normalize(self.embeddings)
        
        print(f"\n{'='*70}")
        print("Voxel Graph Builder Initialized with Pre-fitted PCA")
        print(f"{'='*70}")
        print(f"Number of nodes: {len(self.coordinates)}")
        print(f"Final embedding dimension: {self.embedding_dim}")
    
def compute_graph_metrics(G: nx.Graph) -> Dict:
    """
    Compute various graph metrics
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of metrics
    """
    print("\nComputing graph metrics...")

    metrics = {}

    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    metrics['avg_degree'] = np.mean(degrees)
    metrics['min_degree'] = np.min(degrees)
    metrics['max_degree'] = np.max(degrees)
    metrics['std_degree'] = np.std(degrees)

    # Connectivity
    metrics['is_connected'] = nx.is_connected(G)
    metrics['num_components'] = nx.number_connected_components(G)

    if nx.is_connected(G):
        metrics['diameter'] = nx.diameter(G)
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)

    # Clustering
    metrics['avg_clustering'] = nx.average_clustering(G)

    # Centrality (on sample if graph is large)
    if G.number_of_nodes() < 1000:
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        metrics['closeness_centrality'] = nx.closeness_centrality(G)

    print("\nGraph Metrics:")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    return metrics


def visualize_graph_3d(G: nx.Graph,
                       output_path: str,
                       node_color_by: str = 'degree',
                       edge_alpha: float = 0.1):
    """
    Create 3D visualization of the graph
    Args:
        G: NetworkX graph
        output_path: Path to save visualization
        node_color_by: 'degree', 'clustering', or 'embedding_norm'
        edge_alpha: Transparency of edges
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("\nCreating 3D visualization...")

    # Get node positions from coordinates
    positions = {}
    for node in G.nodes():
        coord = G.nodes[node]['coordinate']
        positions[node] = coord

    # Determine node colors
    if node_color_by == 'degree':
        node_colors = [G.degree(node) for node in G.nodes()]
        colorbar_label = 'Degree'
    elif node_color_by == 'clustering':
        clustering = nx.clustering(G)
        node_colors = [clustering[node] for node in G.nodes()]
        colorbar_label = 'Clustering Coefficient'
    elif node_color_by == 'embedding_norm':
        node_colors = [np.linalg.norm(G.nodes[node]['embedding']) for node in G.nodes()]
        colorbar_label = 'Embedding Norm'
    else:
        node_colors = 'blue'
        colorbar_label = None

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot edges
    for edge in G.edges():
        node1, node2 = edge
        pos1 = positions[node1]
        pos2 = positions[node2]
        
        ax.plot([pos1[0], pos2[0]], 
                [pos1[1], pos2[1]], 
                [pos1[2], pos2[2]], 
                'gray', alpha=edge_alpha, linewidth=0.5)

    # Plot nodes
    coords = np.array([positions[node] for node in G.nodes()])
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=node_colors, s=20, cmap='viridis', alpha=0.8)

    if colorbar_label:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label(colorbar_label)

    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_zlabel('Z (voxels)')
    ax.set_title(f'Voxel Graph (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved 3D visualization to: {output_path}")
    

def export_graph_for_gnn(G: nx.Graph, output_path: str):
    """
    Export graph in PyTorch Geometric format for GNN training
    Args:
        G: NetworkX graph
        output_path: Path to save
    """
    import torch
    from torch_geometric.data import Data

    print("\nExporting graph for PyTorch Geometric...")

    # Create node feature matrix
    node_features = []
    node_to_idx = {}

    for idx, node in enumerate(G.nodes()):
        node_to_idx[node] = idx
        embedding = G.nodes[node]['embedding']
        
        # Optionally concatenate spatial features
        if 'spatial_features' in G.nodes[node]:
            spatial = G.nodes[node]['spatial_features']
            features = np.concatenate([embedding, spatial])
        else:
            features = embedding
        
        node_features.append(features)

    x = torch.tensor(np.array(node_features), dtype=torch.float)

    # Create edge index and edge attributes
    edge_index = []
    edge_attr = []

    for edge in G.edges():
        i, j = edge
        idx_i = node_to_idx[i]
        idx_j = node_to_idx[j]
        
        # Add both directions (undirected graph)
        edge_index.append([idx_i, idx_j])
        edge_index.append([idx_j, idx_i])
        
        weight = G[i][j]['weight']
        edge_attr.append([weight])
        edge_attr.append([weight])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Save
    torch.save(data, output_path)

    print(f"Saved PyG graph to: {output_path}")
    print(f"  Node features: {data.x.shape}")
    print(f"  Edge index: {data.edge_index.shape}")
    print(f"  Edge attributes: {data.edge_attr.shape}")

    return data


if __name__ == "__main__":
    # Example usage with PCA
    extractor = VoxelWiseFMCIBExtractor(
        patch_size=(50, 50, 50),
        gpu_id=2,  # Use GPU 2
        scale_range=(-1024, 2048),
        target_spacing=(1, 1, 1),
        orient="SPL"
    )
    
    image_root = 'XXXyour path/Dataset/Dataset002_ECResponse/Image/'
    mask_root = 'XXXyour path/Dataset/Dataset002_ECResponse/Mask/'
    embeddings_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/embeddings/'
    pca_model_path = 'XXXyour path/Dataset/Dataset002_ECResponse/pca_model_dataset_wide.pkl'
    graph_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/graphs/'
    pca_analysis_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/pca_analysis/'
   
    # Create directories
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(pca_analysis_dir, exist_ok=True)
    
    # lists = os.listdir(image_root)
    # if os.path.exists(graph_dir):
    #     exists_files = os.listdir(graph_dir)
    #     lists = [name for name in lists if name.replace('.nii.gz', '_graph.gpickle') not in exists_files]
    
        # Get all cases
    all_cases = sorted([f.replace('.nii.gz', '') for f in os.listdir(image_root) if f.endswith('.nii.gz')])
    
    # ========================================================================
    # STEP 1: Extract embeddings for all cases (save to disk)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Extracting Embeddings for All Cases")
    print("="*80)
    for case in all_cases:
        embedding_path = os.path.join(embeddings_dir, f"{case}_embeddings.pkl")
        
        if os.path.exists(embedding_path):
            print(f"Skipping {case} (embeddings already exist)")
            continue
        
        print(f"\nProcessing {case}...")
        
        ct_path = os.path.join(image_root, f"{case}.nii.gz")
        mask_path = os.path.join(mask_root, f"{case}.nii.gz")
        
        # Extract embeddings with sliding window
        embeddings_dict = extractor.extract_masked_voxel_embeddings(
            ct_path=ct_path,
            mask_path=mask_path,
            batch_size=1024,
            stride_factor=0.0,  # Half patch size stride
            min_mask_overlap=0.0,
        )
        
        # Save embeddings to disk
        with open(embedding_path, 'wb') as f:
            pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved embeddings: {embeddings_dict['embeddings'].shape}")
        print(f"File size: {Path(embedding_path).stat().st_size / 1e6:.2f} MB")
    
        # ========================================================================
    # STEP 2: Fit PCA on entire training dataset
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Fitting PCA on Entire Training Dataset")
    print("="*80)
    
    if not os.path.exists(pca_model_path):
        # Initialize PCA fitter
        pca_fitter = DatasetWidePCAFitter(
            n_components=512,  # Start with a large number of components
            whiten=False,
            use_incremental=True,  # Recommended for large datasets
            incremental_batch_size=100000
        )
        
        # Fit PCA on all training cases
        # If you have a train/val/test split, only use training cases here
        train_cases = all_cases  # Replace with actual training cases if you have splits
        
        pca_model = pca_fitter.fit_on_dataset(
            embeddings_dir=embeddings_dir,
            case_list=train_cases,
            max_samples_per_case=None  # Use all samples; set to limit if needed
        )
        
        # Save PCA model
        pca_fitter.save_pca_model(pca_model_path)
        
        # Create PCA analysis visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Explained variance
        ax = axes[0, 0]
        ax.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
               pca_model.explained_variance_ratio_)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Variance Explained by Each Component')
        ax.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax = axes[0, 1]
        cumsum = np.cumsum(pca_model.explained_variance_ratio_)
        ax.plot(range(1, len(cumsum) + 1), cumsum, 'b-', linewidth=2)
        ax.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax.axhline(y=0.90, color='orange', linestyle='--', label='90%')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Variance Explained')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Singular values
        ax = axes[1, 0]
        if hasattr(pca_model, 'singular_values_'):
            ax.plot(range(1, len(pca_model.singular_values_) + 1), 
                   pca_model.singular_values_, 'go-')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Singular Value')
            ax.set_title('Scree Plot')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Component loadings
        ax = axes[1, 1]
        if hasattr(pca_model, 'components_'):
            n_show = min(20, pca_model.components_.shape[0])
            im = ax.imshow(pca_model.components_[:n_show, :50], 
                          aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            ax.set_xlabel('Original Feature Dimension')
            ax.set_ylabel('Principal Component')
            ax.set_title(f'First {n_show} Component Loadings')
            plt.colorbar(im, ax=ax, label='Loading')
        
        plt.tight_layout()
        pca_plot_path = os.path.join(pca_analysis_dir, 'dataset_wide_pca_analysis.png')
        plt.savefig(pca_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved PCA analysis plot to: {pca_plot_path}")
    else:
        print(f"PCA model already exists at: {pca_model_path}")
        pca_fitter = DatasetWidePCAFitter(n_components=512)  # Parameters should match those used during fitting
        pca_model = pca_fitter.load_pca_model(pca_model_path)

    # ========================================================================
    # STEP 3: Build graphs for all cases using pre-fitted PCA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Building Graphs with Pre-fitted PCA")
    print("="*80)
    
    for case in all_cases:
        graph_path = os.path.join(graph_dir, f"{case}_graph.gpickle")
        
        if os.path.exists(graph_path):
            print(f"Skipping {case} (graph already exists)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {case}")
        print(f"{'='*80}")
        
        # Load embeddings
        embedding_path = os.path.join(embeddings_dir, f"{case}_embeddings.pkl")
        with open(embedding_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        print(f"Loaded embeddings: {embeddings_dict['embeddings'].shape}")
        
        # Check if embeddings are empty
        if len(embeddings_dict['embeddings']) == 0:
            print(f"WARNING: Empty embeddings for {case}, skipping graph construction")
            continue
        
        # Build graph with pre-fitted PCA
        builder = VoxelGraphBuilderWithPreFittedPCA(
            embeddings_dict=embeddings_dict,
            pca_model=pca_model,  # Use pre-fitted PCA
            use_pca=True,
            connectivity='knn',
            # spatial_distance_threshold=5.0,  # 5mm radius for edges
            k_neighbors=10,
            use_spatial_features=True,
            normalize_embeddings=True
        )
        
        # Get PCA info
        pca_info = builder.get_pca_info()
        if pca_info['pca_applied']:
            print(f"\nPCA Transformation Applied:")
            print(f"  Original dimension: {pca_info['original_dim']}")
            print(f"  Reduced dimension: {pca_info['reduced_dim']}")
            print(f"  Total variance explained: {pca_info['total_explained_variance']:.4f}")
        
        # Build graph
        graph = builder.build_graph(
            edge_weight_method='emd', #cosine euclidean emd
            use_vectorized=True
        )
        
        # Save graph
        builder.save_graph(graph, graph_path, save_pca_model=False)  # Don't save PCA per case
        
        print(f"Completed: {case}")
        print(f"Graph nodes: {graph.number_of_nodes()}")
        print(f"Graph edges: {graph.number_of_edges()}")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)


