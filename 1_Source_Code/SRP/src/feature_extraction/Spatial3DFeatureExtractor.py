# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:22:49 2026

@author: pc
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from libpysal.weights import KNN
from esda import Moran, Geary
from esda.moran import Moran_Local
import gzip
import pickle

        
class Spatial3DFeatureExtractor:
    def __init__(self, image_3d, mask=None, spacing=None):
        """
        Parameters:
        -----------
        image_3d : np.ndarray
            3D image array
        mask : np.ndarray, optional
            3D mask array
        spacing : tuple, optional
            Physical spacing (z, y, x) in mm
        """
        self.image = np.asarray(image_3d)
        self.spacing = spacing if spacing is not None else (1., 1., 1.)

        if mask is None:
            self.mask = self.image > 0
        else:
            mask = np.asarray(mask)
            if mask.dtype != bool:
                self.mask = mask.astype(bool)
            else:
                self.mask = mask

        assert self.mask.shape == self.image.shape, \
            f"Shape mismatch: image {self.image.shape} vs mask {self.mask.shape}"

        if not np.any(self.mask):
            raise ValueError("Mask is empty (all False values)")

    def _prepare_data(self):
        coords = np.argwhere(self.mask)
        values = self.image[self.mask]
        
        # 转换为物理坐标
        coords_mm = coords * np.array(self.spacing)
        
        print(f"Extracted {len(coords):,} voxels from ROI")
        return coords_mm, values

    def extract_features(self):
        features = {}

        coords, values = self._prepare_data()

        n_voxels = len(coords)
        k = min(26, n_voxels - 1)

        if n_voxels < 10:
            raise ValueError(f"Too few voxels: {n_voxels}")

        print(f"Computing spatial weights with k={k}...")
        w = KNN.from_array(coords, k=k)

        print("Computing Moran's I...")
        moran = Moran(values, w)
        features['moran_i'] = moran.I
        features['moran_p'] = moran.p_sim

        print("Computing Geary's C...")
        geary = Geary(values, w)
        features['geary_c'] = geary.C
        features['geary_p'] = geary.p_sim

        print("Computing local statistics...")
        lisa = Moran_Local(values, w)
        features['hotspot_fraction'] = (lisa.q == 1).mean()      # HH: 高值聚集区
        features['coldspot_fraction'] = (lisa.q == 2).mean()     # LL: 低值聚集区
        features['hl_outlier_fraction'] = (lisa.q == 3).mean()    # HL: 高值被低值包围 (侵略性)
        features['lh_outlier_fraction'] = (lisa.q == 4).mean()    # LH: 低值被高值包围 (被浸润)
        features['insignificant_fraction'] = (lisa.q == 0).mean() # 空间模式不显著的区域
        # 3.2 局部模式的变异性
        # 衡量局部Moran's I值的离散程度，反映了空间异质性的另一种形式
        features['local_moran_variability'] = np.std(lisa.Is)
        features['n_voxels'] = n_voxels

        return features


class SpatialRadiomicsPipeline:
    """批量处理管道，包含重采样和特征提取"""
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), interpolation='linear'):
        """
        Parameters:
        -----------
        target_spacing : tuple
            目标分辨率 (z, y, x) in mm，例如 (2.0, 1.0, 1.0)
        interpolation : str
            插值方法: 'linear', 'nearest', 'bspline'
        """
        self.target_spacing = target_spacing
        self.interpolation = interpolation
        
        # 设置插值器
        self.interpolator_map = {
            'linear': sitk.sitkLinear,
            'nearest': sitk.sitkNearestNeighbor,
            'bspline': sitk.sitkBSpline
        }
    
    def resample_image(self, image_sitk, is_mask=False, reference_image=None):
        """
        重采样图像到目标分辨率

        Parameters:
        -----------
        image_sitk : SimpleITK.Image
            输入图像
        is_mask : bool
            是否为mask（使用最近邻插值）
        reference_image : SimpleITK.Image, optional
            若提供，直接复用该图像的网格（size/spacing/origin/direction），
            确保mask与image重采样后体素网格完全一致，
            避免因两者原始spacing的浮点误差导致取整后尺寸不一致

        Returns:
        --------
        resampled : SimpleITK.Image
            重采样后的图像
        """
        resampler = sitk.ResampleImageFilter()

        if reference_image is not None:
            resampler.SetReferenceImage(reference_image)
        else:
            original_spacing = image_sitk.GetSpacing()  # (x, y, z)
            original_size = image_sitk.GetSize()

            # 转换spacing顺序：target_spacing是(z,y,x)，SimpleITK需要(x,y,z)
            target_spacing_sitk = (self.target_spacing[2],
                                  self.target_spacing[1],
                                  self.target_spacing[0])

            # 计算新的尺寸
            new_size = [
                int(round(original_size[i] * original_spacing[i] / target_spacing_sitk[i]))
                for i in range(3)
            ]

            print(f"  Original: size={original_size}, spacing={original_spacing}")
            print(f"  Target:   size={new_size}, spacing={target_spacing_sitk}")

            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(target_spacing_sitk)
            resampler.SetOutputOrigin(image_sitk.GetOrigin())
            resampler.SetOutputDirection(image_sitk.GetDirection())

        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)

        # mask使用最近邻插值，图像使用指定插值
        if is_mask:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(self.interpolator_map[self.interpolation])

        resampled = resampler.Execute(image_sitk)

        return resampled
    
    def process_single_case(self, image_path, mask_path, case_id=None):
        """
        处理单个病例
        
        Parameters:
        -----------
        image_path : str or Path
            图像文件路径
        mask_path : str or Path
            mask文件路径
        case_id : str, optional
            病例ID
        
        Returns:
        --------
        features : dict
            提取的特征
        """
        if case_id is None:
            case_id = Path(image_path).stem.split('.nii')[0]
        
        print(f"\n{'='*60}")
        print(f"Processing case: {case_id}")
        print(f"{'='*60}")
        
        try:
            # 1. 读取图像
            print("Loading images...")
            img_sitk = sitk.ReadImage(str(image_path))
            mask_sitk = sitk.ReadImage(str(mask_path))
            
            # 2. 重采样
            print("Resampling image...")
            img_resampled = self.resample_image(img_sitk, is_mask=False)
            
            print("Resampling mask...")
            mask_resampled = self.resample_image(mask_sitk, is_mask=True, reference_image=img_resampled)
            
            # 3. 转换为numpy数组
            image_3d = sitk.GetArrayFromImage(img_resampled)
            mask_3d = sitk.GetArrayFromImage(mask_resampled)
            
            # 验证mask
            unique_values = np.unique(mask_3d)
            print(f"Mask unique values: {unique_values}")
            
            if len(unique_values) == 1 and unique_values[0] == 0:
                print(f"⚠️  Warning: Empty mask for {case_id}")
                return self._get_empty_features(case_id)
            
            # 4. 提取特征
            print("\nExtracting spatial features...")
            extractor = Spatial3DFeatureExtractor(
                image_3d=image_3d,
                mask=mask_3d.astype(bool),
                spacing=self.target_spacing
            )
            
            features = extractor.extract_features()
            features['case_id'] = case_id
            features['success'] = True
            
            # 添加元数据
            features['original_spacing_z'] = img_sitk.GetSpacing()[2]
            features['original_spacing_y'] = img_sitk.GetSpacing()[1]
            features['original_spacing_x'] = img_sitk.GetSpacing()[0]
            features['resampled_spacing_z'] = self.target_spacing[0]
            features['resampled_spacing_y'] = self.target_spacing[1]
            features['resampled_spacing_x'] = self.target_spacing[2]
            
            print(f"✓ Successfully processed {case_id}")
            return features
            
        except Exception as e:
            print(f"✗ Error processing {case_id}: {str(e)}")
            features = self._get_empty_features(case_id)
            features['error_message'] = str(e)
            return features
    
    def _get_empty_features(self, case_id):
        """返回空特征（处理失败时）"""
        return {
            'case_id': case_id,
            'success': False,
            'moran_i': np.nan,
            'moran_p': np.nan,
            'geary_c': np.nan,
            'geary_p': np.nan,
            'hotspot_fraction': np.nan,
            'coldspot_fraction': np.nan,
            'spatial_variability': np.nan,
            'n_voxels': 0,
            'error_message': 'Empty mask or processing failed'
        }
    
    def process_dataset(self, image_dir, mask_dir, output_csv='spatial_features.csv',
                       pattern='*.nii.gz'):
        """
        批量处理整个数据集
        
        Parameters:
        -----------
        image_dir : str or Path
            图像文件夹路径
        mask_dir : str or Path
            mask文件夹路径
        output_csv : str
            输出CSV文件路径
        pattern : str
            文件匹配模式
        
        Returns:
        --------
        df : pd.DataFrame
            所有病例的特征表
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        
        # 获取所有图像文件
        image_files = sorted(image_dir.glob(pattern))
        
        if len(image_files) == 0:
            raise ValueError(f"No files found matching {pattern} in {image_dir}")
        
        print(f"\nFound {len(image_files)} cases to process")
        print(f"Target spacing: {self.target_spacing}")
        print(f"Interpolation: {self.interpolation}\n")
        
        all_features = []
        
        # 使用tqdm显示进度
        for image_path in tqdm(image_files, desc="Processing cases"):
            case_id = image_path.stem
            mask_path = mask_dir / image_path.name
            
            if not mask_path.exists():
                print(f"⚠️  Warning: Mask not found for {case_id}, skipping...")
                continue
            
            features = self.process_single_case(image_path, mask_path, case_id)
            all_features.append(features)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_features)
        
        # 保存结果
        df.to_csv(output_csv, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_csv}")
        print(f"Total cases processed: {len(df)}")
        print(f"Successful: {df['success'].sum()}")
        print(f"Failed: {(~df['success']).sum()}")
        print(f"{'='*60}")
        
        # 显示统计摘要
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df):
        """打印特征统计摘要"""
        print("\n📊 FEATURE STATISTICS (successful cases only):")
        print("="*60)
        
        df_success = df[df['success'] == True]
        
        if len(df_success) == 0:
            print("No successful cases to summarize")
            return
        
        feature_cols = ['moran_i', 'geary_c', 'hotspot_fraction', 
                       'coldspot_fraction', 'spatial_variability', 'n_voxels']
        
        summary = df_success[feature_cols].describe()
        print(summary.to_string())
        
        print("\n📈 SPATIAL AUTOCORRELATION SUMMARY:")
        print(f"  Positive spatial autocorrelation (Moran's I > 0.3): "
              f"{(df_success['moran_i'] > 0.3).sum()} cases")
        print(f"  Negative spatial autocorrelation (Moran's I < -0.3): "
              f"{(df_success['moran_i'] < -0.3).sum()} cases")
        print(f"  Random pattern (-0.3 ≤ Moran's I ≤ 0.3): "
              f"{((df_success['moran_i'] >= -0.3) & (df_success['moran_i'] <= 0.3)).sum()} cases")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    
    # # 方法1: 处理单个病例
    # print("="*60)
    # print("SINGLE CASE PROCESSING")
    # print("="*60)
    
    # pipeline = SpatialRadiomicsPipeline(
    #     target_spacing=(1.0, 1.0, 1.0),  # Z, Y, X spacing in mm
    #     interpolation='linear'
    # )
    
    # img_path = 'XXXyour path/Dataset/Dataset002_ECResponse/Image/CASE_ID.nii.gz'
    # mask_path = 'XXXyour path/Dataset/Dataset002_ECResponse/Mask/CASE_ID.nii.gz'
    
    # features_single = pipeline.process_single_case(img_path, mask_path)
    # print("\nExtracted features:")
    # for key, value in features_single.items():
    #     print(f"  {key}: {value}")
    
    
    # 方法2: 批量处理整个数据集
    print("\n" + "="*60)
    print("BATCH PROCESSING")
    print("="*60)
    
    pipeline_batch = SpatialRadiomicsPipeline(
        target_spacing=(1.0, 1.0, 1.0),
        interpolation='linear'
    )
    
    # image_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/Image'
    # mask_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/Mask'
    # output_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/spatial3d_features.csv'
    # image_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Image'
    # mask_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Mask'
    # output_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial3d_features.csv'
    image_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/Image'
    mask_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/Mask'
    output_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/spatial3d_features.csv'
    # image_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Image'
    # mask_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Mask'
    # output_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/spatial3d_features.csv'
    # image_dir = 'XXXyour path/Dataset/Dataset004-NACI/FUSCC/Image'
    # mask_dir = 'XXXyour path/Dataset/Dataset004-NACI/FUSCC/Mask'
    # output_csv = 'XXXyour path/Dataset/Dataset004-NACI/FUSCC/spatial3d_features.csv'
    # image_dir = 'XXXyour path/Dataset/Dataset004-NACI/SHPH/Image'
    # mask_dir = 'XXXyour path/Dataset/Dataset004-NACI/SHPH/Mask'
    # output_csv = 'XXXyour path/Dataset/Dataset004-NACI/SHPH/spatial3d_features.csv'
    # image_dir = 'XXXyour path/Dataset/Dataset004-NACI/SPCH/Image'
    # mask_dir = 'XXXyour path/Dataset/Dataset004-NACI/SPCH/Mask'
    # output_csv = 'XXXyour path/Dataset/Dataset004-NACI/SPCH/spatial3d_features.csv'
    # 处理所有病例
    df_results = pipeline_batch.process_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_csv=output_csv,
    pattern='*.nii.gz'
    )

# # 查看结果
# print("\n📋 RESULTS PREVIEW:")
# print(df_results.head(10)

# img_path = 'XXXyour path/Dataset/Dataset002_ECResponse/Image/CASE_ID.nii.gz'
# mask_path = 'XXXyour path/Dataset/Dataset002_ECResponse/Mask/CASE_ID.nii.gz'
# Img = sitk.ReadImage(img_path)
# image_3d = sitk.GetArrayFromImage(Img)
# Mask = sitk.ReadImage(mask_path)
# mask = sitk.GetArrayFromImage(Mask)
# extractor = Spatial3DFeatureExtractor(image_3d=image_3d, mask=mask.astype(int))
# features = extractor.extract_features()

# with gzip.open('XXXyour path/Dataset/Dataset002_ECResponse/FeatureMaps/feature_maps/CASE_ID.pkl.gz', 'rb') as f:
#     data = pickle.load(f)


