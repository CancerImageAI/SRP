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
import gzip


def load_nifti_data(image_path, mask_path):
    """
    Load NIfTI format medical images
    
    Parameters:
    -----------
    image_path: str, path to image file (.nii or .nii.gz)
    mask_path: str, path to mask file (.nii or .nii.gz)
    
    Returns:
    --------
    image: SimpleITK.Image
    mask: SimpleITK.Image
    """

    
    # Load image
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    
    # Ensure mask is binary
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    
    # Ensure same geometry
    mask.CopyInformation(image)
    
    return image, mask

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
            
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(feature_maps, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get file size
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Saved feature maps for {patient_id} ({file_size_mb:.2f} MB)")
            return True
        except Exception as e:
            logger.error(f"Failed to save feature maps for {patient_id}: {str(e)}")
            return False
    
    def load_feature_maps(self, patient_id, params):
        """Load feature maps from disk"""
        try:
            cache_key = self.get_cache_key(patient_id, params)
            cache_file = self.feature_maps_dir / f"{cache_key}.pkl.gz"
            
            if cache_file.exists():
                with gzip.open(cache_file, 'rb') as f:
                    feature_maps = pickle.load(f)
                logger.info(f"✓ Loaded feature maps for {patient_id}")
                return feature_maps
        except Exception as e:
            logger.warning(f"Failed to load feature maps for {patient_id}: {str(e)}")
        
        return None
    
    def exists_feature_maps(self, patient_id, params):
        """Check if feature maps exist"""
        cache_key = self.get_cache_key(patient_id, params)
        cache_file = self.feature_maps_dir / f"{cache_key}.pkl.gz"
        return cache_file.exists()
    
    def get_completed_patients(self, params):
        """Get set of patient IDs that have completed feature map extraction"""
        params_hash = self._get_params_hash(params)
        completed = set()
        
        for cache_file in self.feature_maps_dir.glob(f"*_{params_hash}.pkl.gz"):
            # Extract patient_id (everything before _params_hash)
            patient_id = cache_file.stem.replace(f"_{params_hash}", "").replace('.pkl', '')
            completed.add(patient_id)
        
        return completed
    
    def get_cache_statistics(self, params):
        """Get statistics about cached feature maps"""
        params_hash = self._get_params_hash(params)
        files = list(self.feature_maps_dir.glob(f"*_{params_hash}.pkl.gz"))
        
        total_size = sum(f.stat().st_size for f in files)
        total_size_gb = total_size / (1024 ** 3)
        
        stats = {
            'total_files': len(files),
            'total_size_gb': total_size_gb,
            'avg_size_mb': (total_size / len(files) / (1024 ** 2)) if files else 0
        }
        
        return stats


# ============================================================================
# OPTIMIZED SPARAD FRAMEWORK
# ============================================================================

class OptimizedSpaRadFramework:
    """
    Optimized Spatial Radiomics Framework with caching
    """
    
    def __init__(self, 
                 bin_width=25,
                 window_size=3,
                 feature_classes=['glcm', 'glrlm', 'glszm'],
                 n_threads=4,
                 batch_size=100,
                 cache_manager=None):
        
        self.bin_width = bin_width
        self.window_size = window_size
        self.feature_classes = feature_classes
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.cache_manager = cache_manager
        
        # Store parameters for cache key generation
        self.params = {
            'bin_width': bin_width,
            'window_size': window_size,
            'feature_classes': feature_classes
        }
        
        self.feature_selections = self._define_feature_selections()
        self.extractors = self._initialize_extractors()
        
        logger.info(f"Framework initialized with:")
        logger.info(f"  - Bin width: {bin_width}")
        logger.info(f"  - Window size: {window_size}")
        logger.info(f"  - Feature classes: {feature_classes}")
        logger.info(f"  - Threads: {n_threads}")
        logger.info(f"  - Batch size: {batch_size}")

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
        logger.info(f"EXTRACTING SPATIAL RADIOMICS FEATURE MAPS - {patient_id}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
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
            
            logger.info(f"Image shape: {image_array.shape}")
            logger.info(f"Tumor voxels: {total_voxels}")
            logger.info(f"Processing with {self.n_threads} threads, batch size {self.batch_size}")
            
            # Split coordinates into batches
            batches = [tumor_coords[i:i + self.batch_size] 
                      for i in range(0, total_voxels, self.batch_size)]
            
            logger.info(f"Total batches: {len(batches)}")
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
                                 desc=f"Extracting features for {patient_id}", ncols=100):
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
            
            elapsed_time = time.time() - start_time
            logger.info(f"✓ Feature extraction completed in {elapsed_time/60:.2f} minutes")
            
            # Cache the feature maps
            if self.cache_manager and patient_id:
                self.cache_manager.save_feature_maps(patient_id, self.params, all_feature_maps)
            
            # Force garbage collection
            gc.collect()
            
            return all_feature_maps
        
        except Exception as e:
            logger.error(f"Feature map extraction failed for {patient_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# ============================================================================
# BATCH PROCESSING FUNCTION
# ============================================================================

def process_single_patient(patient_id, image_root, mask_root, framework):
    """
    Process a single patient and save feature maps
    
    Args:
        patient_id: Patient identifier
        image_root: Root directory for images
        mask_root: Root directory for masks
        framework: OptimizedSpaRadFramework instance
        
    Returns:
        Dict with processing status
    """
    result = {
        'patient_id': patient_id,
        'status': 'failed',
        'error': None,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        # Check if already processed
        if framework.cache_manager and framework.cache_manager.exists_feature_maps(patient_id, framework.params):
            logger.info(f"⊙ Skipping {patient_id} - already processed")
            result['status'] = 'skipped'
            return result
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing patient: {patient_id}")
        logger.info(f"{'='*80}")
        
        # Load image and mask
        image_path = os.path.join(image_root, f"{patient_id}.nii.gz")
        mask_path = os.path.join(mask_root, f"{patient_id}.nii.gz")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        logger.info(f"Loading data from:")
        logger.info(f"  Image: {image_path}")
        logger.info(f"  Mask:  {mask_path}")
        
        # Load using DataLoader
        image, mask = load_nifti_data(image_path, mask_path)
        
        # Verify data
        if image is None or mask is None:
            raise ValueError("Failed to load image or mask")
        
        # Get arrays for statistics
        if isinstance(image, np.ndarray):
            image_array = image
            mask_array = mask
        else:
            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)
        
        tumor_voxels = np.sum(mask_array > 0)
        logger.info(f"Image shape: {image_array.shape}")
        logger.info(f"Tumor voxels: {tumor_voxels}")
        
        if tumor_voxels == 0:
            raise ValueError("Mask contains no tumor voxels")
        
        # Extract feature maps
        feature_maps = framework.extract_all_feature_maps(
            image=image,
            mask=mask,
            patient_id=patient_id
        )
        
        # Verify feature maps were created
        if feature_maps is None or len(feature_maps) == 0:
            raise ValueError("No feature maps were generated")
        
        processing_time = time.time() - start_time
        result['status'] = 'success'
        result['processing_time'] = processing_time
        
        logger.info(f"✓ Successfully processed {patient_id} in {processing_time/60:.2f} minutes")
        
        # Clean up
        del image, mask, image_array, mask_array, feature_maps
        gc.collect()
        
    except Exception as e:
        processing_time = time.time() - start_time
        result['error'] = str(e)
        result['processing_time'] = processing_time
        logger.error(f"✗ Failed to process {patient_id}: {str(e)}")
        logger.error(traceback.format_exc())
    
    return result


def batch_process_all_patients(image_root, mask_root, cache_dir, 
                               bin_width=25, window_size=3, 
                               feature_classes=['glcm', 'glrlm', 'glszm'],
                               n_threads=4, batch_size=100,
                               patient_ids=None):
    """
    Process all patients and generate feature maps
    
    Args:
        image_root: Directory containing image files
        mask_root: Directory containing mask files
        cache_dir: Directory to store feature maps
        bin_width: Bin width for discretization
        window_size: Window size for local feature extraction
        feature_classes: List of feature classes to extract
        n_threads: Number of threads for parallel processing
        batch_size: Batch size for processing coordinates
        patient_ids: List of patient IDs to process (if None, process all)
    """
    
    logger.info("="*80)
    logger.info("BATCH FEATURE MAP GENERATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize cache manager
    cache_manager = CacheManager(cache_dir)
    
    # Initialize framework
    framework = OptimizedSpaRadFramework(
        bin_width=bin_width,
        window_size=window_size,
        feature_classes=feature_classes,
        n_threads=n_threads,
        batch_size=batch_size,
        cache_manager=cache_manager
    )
    
    # Get list of patients to process
    if patient_ids is None:
        # Get all patient IDs from image directory
        all_files = [f for f in os.listdir(image_root) if f.endswith('.nii.gz')]
        patient_ids = [f.replace('.nii.gz', '') for f in all_files]
        logger.info(f"Found {len(patient_ids)} patients in {image_root}")
    else:
        logger.info(f"Processing {len(patient_ids)} specified patients")
    
    # Check which patients are already completed
    completed = cache_manager.get_completed_patients(framework.params)
    remaining = [pid for pid in patient_ids if pid not in completed]
    
    logger.info(f"Already completed: {len(completed)} patients")
    logger.info(f"Remaining: {len(remaining)} patients")
    
    if len(remaining) == 0:
        logger.info("All patients already processed!")
        return
    
    # Process each patient
    results = []
    total_start_time = time.time()
    
    for idx, patient_id in enumerate(remaining, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Progress: {idx}/{len(remaining)} ({idx/len(remaining)*100:.1f}%)")
        logger.info(f"Patient: {patient_id}")
        logger.info(f"{'#'*80}")
        
        result = process_single_patient(
            patient_id=patient_id,
            image_root=image_root,
            mask_root=mask_root,
            framework=framework
        )
        
        results.append(result)
        
        # Show progress summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        
        logger.info(f"\nCurrent statistics:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Skipped: {skipped}")
        
        # Estimate remaining time
        if idx > 0:
            elapsed = time.time() - total_start_time
            avg_time_per_patient = elapsed / idx
            remaining_time = avg_time_per_patient * (len(remaining) - idx)
            logger.info(f"  Estimated remaining time: {remaining_time/3600:.2f} hours")
    
    # Final summary
    total_time = time.time() - total_start_time
    
    logger.info("\n" + "="*80)
    logger.info("BATCH PROCESSING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Total patients processed: {len(results)}")
    logger.info(f"  Successful: {sum(1 for r in results if r['status'] == 'success')}")
    logger.info(f"  Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    logger.info(f"  Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_file = os.path.join(cache_dir, f'processing_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to: {results_file}")
    
    # Show failed cases
    failed_cases = [r for r in results if r['status'] == 'failed']
    if failed_cases:
        logger.warning(f"\nFailed cases ({len(failed_cases)}):")
        for case in failed_cases:
            logger.warning(f"  {case['patient_id']}: {case['error']}")
    
    # Cache statistics
    stats = cache_manager.get_cache_statistics(framework.params)
    logger.info(f"\nCache statistics:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
    logger.info(f"  Average size: {stats['avg_size_mb']:.2f} MB per file")
    
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    try:
        # Configuration
        # image_root = 'XXXyour path/Dataset/Dataset002_ECResponse/Image/'
        # mask_root = 'XXXyour path/Dataset/Dataset002_ECResponse/Mask/'
        # cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/FeatureMaps/'
        # image_root = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Image/'
        # mask_root = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Mask/'
        # cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/FeatureMaps/'
        image_root = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/Image/'
        mask_root = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/Mask/'
        cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/FeatureMaps/'
        # image_root = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Image/'
        # mask_root = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Mask/'
        # cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/FeatureMaps/'
        # image_root = 'XXXyour path/Dataset/Dataset004-NACI/FUSCC/Image'
        # mask_root = 'XXXyour path/Dataset/Dataset004-NACI/FUSCC/Mask'
        # cache_dir = 'XXXyour path/Dataset/Dataset004-NACI/FUSCC/FeatureMaps'
        # image_root = 'XXXyour path/Dataset/Dataset004-NACI/SHPH/Image'
        # mask_root = 'XXXyour path/Dataset/Dataset004-NACI/SHPH/Mask'
        # cache_dir = 'XXXyour path/Dataset/Dataset004-NACI/SHPH/FeatureMaps'
        # Processing parameters
        bin_width = 25
        window_size = 3
        feature_classes = ['glcm', 'glrlm', 'glszm']
        n_threads = 32  # Matches the 32 physical cores on this machine; only one
        batch_size = 2048
        
        # Verify directories exist
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        if not os.path.exists(mask_root):
            raise FileNotFoundError(f"Mask directory not found: {mask_root}")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info("Configuration:")
        logger.info(f"  Image root: {image_root}")
        logger.info(f"  Mask root: {mask_root}")
        logger.info(f"  Cache directory: {cache_dir}")
        logger.info(f"  Bin width: {bin_width}")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Feature classes: {feature_classes}")
        logger.info(f"  Threads: {n_threads}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Optional: Process only specific patients (set to None to process all)
        # patient_ids = ['patient_001', 'patient_002', 'patient_003']
        patient_ids = None
        
        # Run batch processing
        batch_process_all_patients(
            image_root=image_root,
            mask_root=mask_root,
            cache_dir=cache_dir,
            bin_width=bin_width,
            window_size=window_size,
            feature_classes=feature_classes,
            n_threads=n_threads,
            batch_size=batch_size,
            patient_ids=patient_ids
        )
        
        logger.info("\n" + "="*80)
        logger.info("ALL PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nCritical error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()