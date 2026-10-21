# ============================================================================
# REAL DATA LOADING UTILITIES
# ============================================================================
import SimpleITK as sitk
import os

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


def load_dicom_series(dicom_dir, mask_path=None):
    """
    Load DICOM series
    
    Parameters:
    -----------
    dicom_dir: str, directory containing DICOM files
    mask_path: str, path to mask file (optional)
    
    Returns:
    --------
    image: SimpleITK.Image
    mask: SimpleITK.Image or None
    """
    
    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Load mask if provided
    if mask_path and os.path.exists(mask_path):
        mask = sitk.ReadImage(mask_path)
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        mask.CopyInformation(image)
    else:
        mask = None
    
    return image, mask
