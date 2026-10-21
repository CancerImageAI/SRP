import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import gc
from fmcib.models import fmcib_model


class VoxelWiseFMCIBExtractor:
    """
    Extract voxel-wise embeddings for masked regions using FMCIB foundation model
    Each voxel gets an embedding vector (final output of the foundation model)
    Properly handles spacing normalization
    """
    
    def __init__(self, 
                 patch_size=(48, 48, 48),
                 gpu_id=None,
                 scale_range=(-1024, 2048),
                 target_spacing=(1.0, 1.0, 1.0),
                 orient="LPS"):
        """
        Args:
            patch_size: Size of sliding window patches
            gpu_id: GPU to use
            scale_range: HU value range for normalization
            target_spacing: Target spacing in mm (z, y, x)
            orient: Target orientation ('LPS', 'RAS', etc.)
        """
        
        # Setup GPU
        self.gpu_id = gpu_id
        self.device = self._setup_device()
        
        # Load model
        print("Loading FMCIB model...")
        self.model = fmcib_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.scale_range = scale_range
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.orient = orient
        
        # Get embedding dimension
        self.embedding_dim = self._get_embedding_dim()
        
        print(f"\nVoxel-wise FMCIB Extractor initialized")
        print(f"Device: {self.device}")
        print(f"Patch size: {self.patch_size}")
        print(f"Target spacing: {self.target_spacing} mm")
        print(f"Target orientation: {self.orient}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _setup_device(self):
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU")
            return torch.device('cpu')
        
        if self.gpu_id is not None:
            if self.gpu_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {self.gpu_id} not available")
            device = torch.device(f'cuda:{self.gpu_id}')
            torch.cuda.set_device(device)
            print(f"Using GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}")
        else:
            device = torch.device('cuda:0')
        
        return device
    
    def _get_embedding_dim(self):
        """Get the embedding dimension by running a test forward pass"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, *self.patch_size).to(self.device)
            dummy_output = self.model(dummy_input)
            return dummy_output.shape[-1]
    
    def _resample_patch(self, patch_sitk: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
        """
        Resample patch to target spacing using SimpleITK
        
        Args:
            patch_sitk: SimpleITK image
            target_spacing: Target spacing (z, y, x)
            
        Returns:
            Resampled SimpleITK image
        """
        original_spacing = patch_sitk.GetSpacing()  # (x, y, z) in SimpleITK
        original_size = patch_sitk.GetSize()        # (x, y, z)
        
        # Convert target_spacing from (z, y, x) to (x, y, z) for SimpleITK
        target_spacing_xyz = (target_spacing[2], target_spacing[1], target_spacing[0])
        
        # Calculate new size
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing_xyz[i]))
            for i in range(3)
        ]
        
        # Setup resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing_xyz)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(patch_sitk.GetDirection())
        resampler.SetOutputOrigin(patch_sitk.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(self.scale_range[0])
        resampler.SetInterpolator(sitk.sitkLinear)
        
        return resampler.Execute(patch_sitk)
    
    def _orient_patch(self, patch_sitk: sitk.Image, target_orientation: str = "LPS") -> sitk.Image:
        """
        Reorient patch to target orientation using SimpleITK
        
        Args:
            patch_sitk: SimpleITK image
            target_orientation: Target orientation code (e.g., 'LPS', 'RAS')
            
        Returns:
            Reoriented SimpleITK image
        """
        # Use SimpleITK's DICOMOrient which accepts orientation strings
        # Common orientations: LPS, RAS, etc.
        return sitk.DICOMOrient(patch_sitk, target_orientation)
    
    def _preprocess_patch(self, patch_array: np.ndarray, 
                         original_spacing: Tuple[float, float, float],
                         origin: Tuple[float, float, float] = (0, 0, 0)) -> torch.Tensor:
        """
        Preprocess patch: resample to target spacing, orient, and normalize
        
        Args:
            patch_array: Numpy array (D, H, W)
            original_spacing: Original spacing (z, y, x) in mm
            origin: Origin coordinates
            
        Returns:
            Preprocessed tensor (1, 1, D, H, W)
        """
        # Convert numpy array to SimpleITK image
        patch_sitk = sitk.GetImageFromArray(patch_array)
        
        # Set spacing (SimpleITK uses x, y, z order)
        patch_sitk.SetSpacing((original_spacing[2], original_spacing[1], original_spacing[0]))
        patch_sitk.SetOrigin(origin)
        
        # Resample to target spacing if different
        if original_spacing != self.target_spacing:
            patch_resampled = self._resample_patch(patch_sitk, self.target_spacing)
        else:
            patch_resampled = patch_sitk
        
        # Orient to target orientation
        patch_oriented = self._orient_patch(patch_resampled, self.orient)
        
        # Convert back to numpy
        patch_processed = sitk.GetArrayFromImage(patch_oriented)
        
        # Resize/pad to exact patch size
        current_size = patch_processed.shape
        if current_size != self.patch_size:
            patch_processed = self._resize_to_patch_size(patch_processed)
        
        # Convert to tensor and normalize
        patch_tensor = torch.from_numpy(patch_processed).float()
        
        # Normalize HU values to [0, 1]
        patch_tensor = torch.clamp(patch_tensor, self.scale_range[0], self.scale_range[1])
        patch_tensor = (patch_tensor - self.scale_range[0]) / (self.scale_range[1] - self.scale_range[0])
        
        # Add batch and channel dimensions
        patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        return patch_tensor
    
    def _resize_to_patch_size(self, array: np.ndarray) -> np.ndarray:
        """
        Resize array to exact patch size by cropping or padding
        
        Args:
            array: Input array
            
        Returns:
            Resized array with shape self.patch_size
        """
        current_size = array.shape
        target_size = self.patch_size
        
        # Initialize result with background value
        result = np.ones(target_size, dtype=array.dtype) * self.scale_range[0]
        
        # Calculate crop/pad indices for each dimension
        slices = []
        result_slices = []
        
        for i in range(3):
            if current_size[i] > target_size[i]:
                # Need to crop
                start = (current_size[i] - target_size[i]) // 2
                slices.append(slice(start, start + target_size[i]))
                result_slices.append(slice(0, target_size[i]))
            else:
                # Need to pad (or exact fit)
                start = (target_size[i] - current_size[i]) // 2
                slices.append(slice(0, current_size[i]))
                result_slices.append(slice(start, start + current_size[i]))
        
        # Copy data
        result[result_slices[0], result_slices[1], result_slices[2]] = \
            array[slices[0], slices[1], slices[2]]
        
        return result
    
    def _extract_patch_embedding(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from a patch using the foundation model
        
        Args:
            patch: Input patch (B, 1, D, H, W)
            
        Returns:
            Embedding vector (B, embedding_dim)
        """
        with torch.no_grad():
            embedding = self.model(patch)
        
        return embedding
    
    def get_embedding_dim(self):
        """Get dimension of embedding vector"""
        return self.embedding_dim
    
    def extract_masked_voxel_embeddings(
        self,
        ct_path: str,
        mask_path: str,
        batch_size: int = 512,
        stride_factor: float = 0.5,
        return_patch_centers: bool = True,
        min_mask_overlap: float = 0.1
    ) -> Dict:
        """
        Extract embeddings for voxels in masked region using sliding window
        with proper spacing normalization
        
        Args:
            ct_path: Path to CT image
            mask_path: Path to mask
            batch_size: Number of patches to process at once
            stride_factor: Stride as fraction of patch_size (0.5 = half overlap)
            return_patch_centers: Whether to return the actual patch center coordinates
            min_mask_overlap: Minimum fraction of patch overlapping with mask
            
        Returns:
            Dictionary containing:
                - embeddings: (N, embedding_dim) array
                - coordinates: (N, 3) array of patch center coordinates
                - ct_shape: Original CT shape
                - spacing: Voxel spacing
                - embedding_dim: Dimension of embeddings
                - stride: Actual stride used
                - patch_size: Patch size used
        """
        print(f"\n{'='*70}")
        print("Extracting Voxel Embeddings with Sliding Window")
        print(f"{'='*70}")
        
        # Load CT and mask using nibabel
        ct_nifti = nib.load(ct_path)
        ct_array = ct_nifti.get_fdata()
        affine = ct_nifti.affine
        
        mask_nifti = nib.load(mask_path)
        mask_array = mask_nifti.get_fdata().astype(bool)
        
        # Get spacing from SimpleITK
        ct_sitk = sitk.ReadImage(ct_path)
        spacing_xyz = ct_sitk.GetSpacing()  # (x, y, z)
        spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])  # Convert to (z, y, x)
        
        print(f"CT shape: {ct_array.shape}")
        print(f"Mask shape: {mask_array.shape}")
        print(f"Original spacing: {spacing} mm (z, y, x)")
        print(f"Target spacing: {self.target_spacing} mm")
        print(f"Patch size: {self.patch_size}")
        
        # Calculate stride
        stride = tuple(int(p * stride_factor) for p in self.patch_size)
        print(f"Stride (stride_factor={stride_factor}): {stride}")
        
        # Get bounding box of mask
        coords_mask = np.argwhere(mask_array > 0)
        if len(coords_mask) == 0:
            raise ValueError("Mask is empty!")
        
        min_coords = coords_mask.min(axis=0)
        max_coords = coords_mask.max(axis=0)
        
        print(f"Mask bounding box: {min_coords} to {max_coords}")
        print(f"Mask volume: {len(coords_mask)} voxels")
        
        # Generate sliding window centers within mask bounding box
        patch_centers = self._generate_sliding_window_centers(
            min_coords=min_coords,
            max_coords=max_coords,
            ct_shape=ct_array.shape,
            patch_size=self.patch_size,
            stride=stride,
            mask_array=mask_array,
            min_mask_overlap=min_mask_overlap
        )
        
        num_patches = len(patch_centers)
        
        # Check if any patches were generated
        if num_patches == 0:
            print("\n" + "!"*70)
            print("WARNING: No valid patches found!")
            print("!"*70)
            print(f"Possible reasons:")
            print(f"  1. Mask is too small for patch size {self.patch_size}")
            print(f"  2. min_mask_overlap ({min_mask_overlap}) is too strict")
            print(f"  3. Stride ({stride}) is too large")
            print(f"\nSuggestions:")
            print(f"  - Reduce min_mask_overlap (try 0.01 or 0.05)")
            print(f"  - Reduce stride_factor (try 0.25)")
            print(f"  - Check if mask volume ({len(coords_mask)} voxels) is sufficient")
            
            # Return empty result
            return {
                'embeddings': np.array([]),
                'coordinates': np.array([]),
                'ct_shape': ct_array.shape,
                'spacing': spacing,
                'embedding_dim': self.embedding_dim,
                'stride': stride,
                'patch_size': self.patch_size,
                'num_original_voxels': len(coords_mask),
                'reduction_factor': 0.0,
                'affine': affine,
                'target_spacing': self.target_spacing,
                'original_spacing': spacing
            }
        
        print(f"Generated {num_patches} patch centers")
        print(f"Reduction factor: {len(coords_mask) / num_patches:.2f}x")
        
        # Initialize embeddings array
        embeddings = np.zeros((num_patches, self.embedding_dim), dtype=np.float32)
        
        # Extract patches and get embeddings
        half_patch = np.array(self.patch_size) // 2
        num_batches = int(np.ceil(num_patches / batch_size))
        
        print(f"\nProcessing {num_batches} batches...")
        
        for batch_idx in tqdm(range(num_batches), desc="Extracting embeddings"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_patches)
            actual_batch_size = end_idx - start_idx
            
            batch_centers = patch_centers[start_idx:end_idx]
            
            # Extract and preprocess patches
            batch_patches = []
            for center in batch_centers:
                # Extract patch from original CT
                d, h, w = center
                
                # Calculate patch boundaries
                d_start = max(0, d - half_patch[0])
                d_end = min(ct_array.shape[0], d + half_patch[0])
                h_start = max(0, h - half_patch[1])
                h_end = min(ct_array.shape[1], h + half_patch[1])
                w_start = max(0, w - half_patch[2])
                w_end = min(ct_array.shape[2], w + half_patch[2])
                
                # Extract raw patch
                patch = ct_array[d_start:d_end, h_start:h_end, w_start:w_end].copy()
                
                # Calculate origin for this patch (for spacing normalization)
                origin = (
                    d_start * spacing[0],
                    h_start * spacing[1],
                    w_start * spacing[2]
                )
                
                # Pad if at boundary
                current_size = patch.shape
                if current_size != self.patch_size:
                    pad_d = max(0, self.patch_size[0] - current_size[0])
                    pad_h = max(0, self.patch_size[1] - current_size[1])
                    pad_w = max(0, self.patch_size[2] - current_size[2])
                    
                    pad_d_before = pad_d // 2
                    pad_d_after = pad_d - pad_d_before
                    pad_h_before = pad_h // 2
                    pad_h_after = pad_h - pad_h_before
                    pad_w_before = pad_w // 2
                    pad_w_after = pad_w - pad_w_before
                    
                    patch = np.pad(
                        patch,
                        ((pad_d_before, pad_d_after),
                        (pad_h_before, pad_h_after),
                        (pad_w_before, pad_w_after)),
                        mode='constant',
                        constant_values=self.scale_range[0]
                    )
                
                # Preprocess with spacing normalization
                patch_preprocessed = self._preprocess_patch(
                    patch_array=patch,
                    original_spacing=spacing,
                    origin=origin
                )
                
                batch_patches.append(patch_preprocessed)
            
            # Stack patches into batch
            batch_tensor = torch.cat(batch_patches, dim=0).to(self.device)
            
            # Extract embeddings using the _extract_patch_embedding method
            with torch.no_grad():
                batch_embeddings = self._extract_patch_embedding(batch_tensor)
            
            # Store embeddings
            embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()
            
            # Free memory
            del batch_tensor, batch_embeddings, batch_patches
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        print(f"\n{'='*70}")
        print("Extraction Complete")
        print(f"{'='*70}")
        print(f"Total patches extracted: {num_patches}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Memory usage: {embeddings.nbytes / 1e6:.2f} MB")
        
        return {
            'embeddings': embeddings,
            'coordinates': np.array(patch_centers),
            'ct_shape': ct_array.shape,
            'spacing': spacing,
            'embedding_dim': self.embedding_dim,
            'stride': stride,
            'patch_size': self.patch_size,
            'num_original_voxels': len(coords_mask),
            'reduction_factor': len(coords_mask) / num_patches if num_patches > 0 else 0.0,
            'affine': affine,
            'target_spacing': self.target_spacing,
            'original_spacing': spacing
        }


    def _generate_sliding_window_centers(
        self,
        min_coords: np.ndarray,
        max_coords: np.ndarray,
        ct_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        mask_array: np.ndarray,
        min_mask_overlap: float = 0.1  # Minimum fraction of patch that must overlap with mask
    ) -> List[np.ndarray]:
        """
        Generate patch centers using sliding window
        **Centers must be within the mask**
        
        Args:
            min_coords: Minimum coordinates of mask bounding box
            max_coords: Maximum coordinates of mask bounding box
            ct_shape: Shape of CT array
            patch_size: Size of patches
            stride: Stride for sliding window
            mask_array: Binary mask array
            min_mask_overlap: Minimum overlap with mask to keep patch
            
        Returns:
            List of patch center coordinates
        """
        patch_centers = []
        if stride == (0, 0, 0):
            patch_centers = np.where(mask_array > 0)
            patch_centers = list(zip(patch_centers[0], patch_centers[1], patch_centers[2]))
        else:
            
            # Calculate valid range for patch centers (accounting for patch size)
            half_patch = np.array(patch_size) // 2
            
            # Start from min_coords and slide with stride
            d_start = max(half_patch[0], min_coords[0])
            d_end = min(ct_shape[0] - half_patch[0], max_coords[0] + 1)
            
            h_start = max(half_patch[1], min_coords[1])
            h_end = min(ct_shape[1] - half_patch[1], max_coords[1] + 1)
            
            w_start = max(half_patch[2], min_coords[2])
            w_end = min(ct_shape[2] - half_patch[2], max_coords[2] + 1)
            
            print(f"Sliding window range:")
            print(f"  D: {d_start} to {d_end} (step {stride[0]})")
            print(f"  H: {h_start} to {h_end} (step {stride[1]})")
            print(f"  W: {w_start} to {w_end} (step {stride[2]})")
            
            # Generate centers
            total_candidates = 0
            centers_in_mask = 0
            centers_with_overlap = 0
            
            for d in range(d_start, d_end, stride[0]):
                for h in range(h_start, h_end, stride[1]):
                    for w in range(w_start, w_end, stride[2]):
                        total_candidates += 1
                        center = np.array([d, h, w])
                        
                        # CRITICAL: Check if center is within mask
                        if not mask_array[d, h, w]:
                            continue
                        
                        centers_in_mask += 1
                        
                        # Check if patch has sufficient overlap with mask
                        if self._check_patch_mask_overlap(
                            center, patch_size, mask_array, min_mask_overlap
                        ):
                            patch_centers.append(center)
                            centers_with_overlap += 1
            
            print(f"\nPatch center selection:")
            print(f"  Total candidate positions: {total_candidates}")
            print(f"  Centers in mask: {centers_in_mask}")
            print(f"  Centers with sufficient overlap (>={min_mask_overlap}): {centers_with_overlap}")
            
            if centers_with_overlap == 0 and centers_in_mask > 0:
                print(f"\n  WARNING: {centers_in_mask} centers are in mask but none meet overlap threshold!")
                print(f"  Consider reducing min_mask_overlap from {min_mask_overlap}")
        
        return patch_centers


    def _check_patch_mask_overlap(
        self,
        center: np.ndarray,
        patch_size: Tuple[int, int, int],
        mask_array: np.ndarray,
        min_overlap: float
    ) -> bool:
        """
        Check if patch has sufficient overlap with mask
        
        Args:
            center: Patch center coordinates
            patch_size: Patch size
            mask_array: Binary mask
            min_overlap: Minimum fraction of patch overlapping with mask
            
        Returns:
            True if patch has sufficient overlap
        """
        half_patch = np.array(patch_size) // 2
        
        # Get patch bounds
        start = center - half_patch
        end = center + half_patch
        
        # Clip to valid range
        start = np.maximum(start, 0)
        end = np.minimum(end, mask_array.shape)
        
        # Extract patch region from mask
        patch_mask = mask_array[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ]
        
        # Calculate overlap fraction
        overlap_fraction = np.sum(patch_mask > 0) / patch_mask.size
        
        return overlap_fraction >= min_overlap
    # def extract_masked_voxel_embeddings(self, 
    #                                    ct_path: str, 
    #                                    mask_path: str,
    #                                    batch_size: int = 32) -> Dict:
    #     """
    #     Extract embeddings for all voxels in the mask
        
    #     Args:
    #         ct_path: Path to CT scan
    #         mask_path: Path to binary mask
    #         batch_size: Number of patches to process at once
            
    #     Returns:
    #         Dictionary containing embeddings and metadata
    #     """
    #     print(f"\n{'='*70}")
    #     print("Extracting Voxel-wise Embeddings for Masked Region")
    #     print(f"{'='*70}")
    #     print(f"CT: {ct_path}")
    #     print(f"Mask: {mask_path}")
        
    #     # Load CT using nibabel (for affine) and SimpleITK (for spacing)
    #     ct_nifti = nib.load(ct_path)
    #     ct_data = ct_nifti.get_fdata()
    #     affine = ct_nifti.affine
        
    #     # Load with SimpleITK to get spacing
    #     ct_sitk = sitk.ReadImage(ct_path)
    #     spacing_xyz = ct_sitk.GetSpacing()  # (x, y, z)
    #     spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])  # Convert to (z, y, x)
        
    #     # Load mask
    #     mask_nifti = nib.load(mask_path)
    #     mask_data = mask_nifti.get_fdata().astype(bool)
        
    #     print(f"\nCT shape: {ct_data.shape}")
    #     print(f"Original spacing: {spacing_zyx} mm (z, y, x)")
    #     print(f"Mask shape: {mask_data.shape}")
        
    #     # Get masked voxel coordinates
    #     masked_coords = np.argwhere(mask_data)  # (N, 3)
    #     num_voxels = len(masked_coords)
        
    #     print(f"Number of masked voxels: {num_voxels}")
        
    #     if num_voxels == 0:
    #         print("WARNING: Mask is empty!")
    #         return {
    #             'embeddings': np.array([]),
    #             'coordinates': np.array([]),
    #             'mask_indices': np.array([]),
    #             'ct_shape': ct_data.shape,
    #             'embedding_dim': self.embedding_dim,
    #             'spacing': spacing_zyx,
    #             'affine': affine
    #         }
        
    #     # Initialize embeddings array
    #     embeddings = np.zeros((num_voxels, self.embedding_dim), dtype=np.float32)
        
    #     # Calculate patch boundaries
    #     half_patch = tuple(p // 2 for p in self.patch_size)
        
    #     print(f"\nPatch size: {self.patch_size}")
    #     print(f"Embedding dimension: {self.embedding_dim}")
    #     print(f"Batch size: {batch_size}")
        
    #     # Process voxels in batches
    #     num_batches = int(np.ceil(num_voxels / batch_size))
        
    #     print(f"\nProcessing {num_batches} batches...")
        
    #     for batch_idx in tqdm(range(num_batches), desc="Extracting embeddings"):
    #         start_idx = batch_idx * batch_size
    #         end_idx = min(start_idx + batch_size, num_voxels)
    #         actual_batch_size = end_idx - start_idx
            
    #         batch_coords = masked_coords[start_idx:end_idx]
    #         batch_patches = []
            
    #         # Extract patches centered at each voxel
    #         for coord in batch_coords:
    #             d, h, w = coord
                
    #             # Calculate patch boundaries in original space
    #             d_start = max(0, d - half_patch[0])
    #             d_end = min(ct_data.shape[0], d + half_patch[0])
    #             h_start = max(0, h - half_patch[1])
    #             h_end = min(ct_data.shape[1], h + half_patch[1])
    #             w_start = max(0, w - half_patch[2])
    #             w_end = min(ct_data.shape[2], w + half_patch[2])
                
    #             # Extract patch
    #             patch = ct_data[d_start:d_end, h_start:h_end, w_start:w_end].copy()
                
    #             # Calculate origin for this patch
    #             origin = (
    #                 d_start * spacing_zyx[0],
    #                 h_start * spacing_zyx[1],
    #                 w_start * spacing_zyx[2]
    #             )
                
    #             # Pad if necessary (at image boundaries)
    #             current_size = patch.shape
    #             if current_size != self.patch_size:
    #                 pad_d = max(0, self.patch_size[0] - current_size[0])
    #                 pad_h = max(0, self.patch_size[1] - current_size[1])
    #                 pad_w = max(0, self.patch_size[2] - current_size[2])
                    
    #                 pad_d_before = pad_d // 2
    #                 pad_d_after = pad_d - pad_d_before
    #                 pad_h_before = pad_h // 2
    #                 pad_h_after = pad_h - pad_h_before
    #                 pad_w_before = pad_w // 2
    #                 pad_w_after = pad_w - pad_w_before
                    
    #                 patch = np.pad(
    #                     patch,
    #                     ((pad_d_before, pad_d_after), 
    #                      (pad_h_before, pad_h_after),
    #                     (pad_w_before, pad_w_after)),
    #                     mode='constant',
    #                     constant_values=self.scale_range[0]
    #                     )
    #             # Preprocess patch with spacing normalization
    #             patch_preprocessed = self._preprocess_patch(
    #                 patch, 
    #                 original_spacing=spacing_zyx,
    #                 origin=origin
    #             )
    #             batch_patches.append(patch_preprocessed)
            
    #         # Stack patches into batch
    #         batch_tensor = torch.cat(batch_patches, dim=0).to(self.device)
            
    #         # Extract embeddings
    #         with torch.no_grad():
    #             batch_embeddings = self._extract_patch_embedding(batch_tensor)
            
    #         # Store embeddings
    #         embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()
            
    #         # Free memory
    #         del batch_tensor, batch_embeddings, batch_patches
    #         if self.device.type == 'cuda':
    #             torch.cuda.empty_cache()
    #         gc.collect()
        
    #     print(f"\n✓ Extracted embeddings for {num_voxels} voxels")
    #     print(f"Embedding shape: {embeddings.shape}")
        
    #     result = {
    #         'embeddings': embeddings,
    #         'coordinates': masked_coords,
    #         'mask_indices': np.ravel_multi_index(
    #             masked_coords.T, 
    #             ct_data.shape
    #         ),
    #         'ct_shape': ct_data.shape,
    #         'embedding_dim': self.embedding_dim,
    #         'spacing': spacing_zyx,
    #         'affine': affine
    #     }
        
    #     return result

    def save_embeddings(self, 
                    embeddings_dict: Dict,
                    output_path: str,
                    save_format: str = 'npz'):
        """
        Save extracted embeddings
        
        Args:
            embeddings_dict: Dictionary returned by extract_masked_voxel_embeddings
            output_path: Output file path
            save_format: 'npz', 'npy', or 'nifti'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == 'npz':
            # Save as compressed numpy archive (recommended)
            np.savez_compressed(
                output_path,
                embeddings=embeddings_dict['embeddings'],
                coordinates=embeddings_dict['coordinates'],
                mask_indices=embeddings_dict['mask_indices'],
                ct_shape=embeddings_dict['ct_shape'],
                embedding_dim=embeddings_dict['embedding_dim'],
                spacing=embeddings_dict['spacing'],
                affine=embeddings_dict['affine']
            )
            print(f"\nSaved embeddings to: {output_path}")
            print(f"  Embeddings: {embeddings_dict['embeddings'].shape}")
            print(f"  Coordinates: {embeddings_dict['coordinates'].shape}")
            print(f"  Spacing: {embeddings_dict['spacing']} mm")
            print(f"  File size: {output_path.stat().st_size / 1e6:.2f} MB")
            
        elif save_format == 'npy':
            # Save only embeddings as numpy array
            np.save(output_path, embeddings_dict['embeddings'])
            
            # Save metadata separately
            coord_path = output_path.parent / f"{output_path.stem}_coordinates.npy"
            np.save(coord_path, embeddings_dict['coordinates'])
            
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.npz"
            np.savez(metadata_path,
                    ct_shape=embeddings_dict['ct_shape'],
                    spacing=embeddings_dict['spacing'],
                    affine=embeddings_dict['affine'])
            
            print(f"Saved embeddings to: {output_path}")
            print(f"Saved coordinates to: {coord_path}")
            print(f"Saved metadata to: {metadata_path}")
            
        elif save_format == 'nifti':
            # Save as 4D NIfTI (D, H, W, embedding_dim)
            ct_shape = embeddings_dict['ct_shape']
            embedding_dim = embeddings_dict['embedding_dim']
            affine = embeddings_dict['affine']
            
            # Create 4D array
            embedding_volume = np.zeros((*ct_shape, embedding_dim), dtype=np.float32)
            
            # Fill in embeddings at masked voxel locations
            coords = embeddings_dict['coordinates']
            embeddings = embeddings_dict['embeddings']
            
            print(f"Creating embedding volume...")
            for i in tqdm(range(len(coords)), desc="Filling voxels"):
                coord = coords[i]
                embedding_volume[coord[0], coord[1], coord[2], :] = embeddings[i]
            
            # Save as NIfTI
            nifti_img = nib.Nifti1Image(embedding_volume, affine)
            nib.save(nifti_img, output_path)
            
            print(f"\nSaved embedding volume to: {output_path}")
            print(f"  Shape: {embedding_volume.shape}")
            print(f"  File size: {output_path.stat().st_size / 1e6:.2f} MB")
        
        else:
            raise ValueError(f"Unknown save format: {save_format}")


if __name__ == "__main__":
    # Example usage
    extractor = VoxelWiseFMCIBExtractor(
        patch_size=(48, 48, 48),
        gpu_id=2,  # Use GPU 2
        scale_range=(-1024, 2048),
        target_spacing=(1, 1, 1),
        orient="SPL"
    )
    # Extract embeddings for masked voxels
    result = extractor.extract_masked_voxel_embeddings(
        ct_path='XXXyour path/Dataset/Dataset002_ECResponse/imagesTr/CASE_ID.nii.gz',
        mask_path='XXXyour path/Dataset/Dataset002_ECResponse/labelsTr/CASE_ID.nii.gz',
        batch_size=1024,  # Process 1024 patches at a time
        stride_factor=0.0625,  # 75% overlap
        min_mask_overlap=0.01
    )
    
    # Save embeddings
    extractor.save_embeddings(
        embeddings_dict=result,
        output_path='./voxel_embeddings/CASE_ID_embeddings_patchlevel.nii.gz',
        save_format='nifti'  # Recommended format
    )

    # Print summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    print(f"Number of voxels: {result['embeddings'].shape[0]}")
    print(f"Embedding dimension: {result['embedding_dim']}")
    print(f"Total memory: {result['embeddings'].nbytes / 1e6:.2f} MB")

