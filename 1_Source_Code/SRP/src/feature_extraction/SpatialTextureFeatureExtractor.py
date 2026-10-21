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
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
import os


class SpatialTextureFeatureExtractor:
    def __init__(self, featuremap_3d, original_spacing, target_spacing, feature_name, interpolation='linear'):
        """
        Parameters:
        -----------
        featuremap_3d : np.ndarray
            3D特征图数组
        original_spacing : tuple
            原始spacing (x, y, z) - SimpleITK格式
        target_spacing : tuple
            目标spacing (z, y, x) - 您的格式
        feature_name : str
            特征名称
        interpolation : str
            插值方法
        """
        self.interpolation = interpolation
        self.interpolator_map = {
            'linear': sitk.sitkLinear,
            'nearest': sitk.sitkNearestNeighbor,
            'bspline': sitk.sitkBSpline
        }
        self.feature_name = feature_name
        self.original_spacing = original_spacing  # (x, y, z)
        self.target_spacing = target_spacing      # (z, y, x)
        
        # 转换为SimpleITK图像并重采样
        featuremap_itk = sitk.GetImageFromArray(featuremap_3d)
        featuremap_itk.SetSpacing(original_spacing)
        resampled_featuremap_itk = self.resample_image(featuremap_itk)
        
        # 转换回numpy数组
        self.featuremap = sitk.GetArrayFromImage(resampled_featuremap_itk)
        
    def resample_image(self, image_sitk):
        """
        重采样图像到目标分辨率
        
        Parameters:
        -----------
        image_sitk : SimpleITK.Image
            输入图像
        
        Returns:
        --------
        resampled : SimpleITK.Image
            重采样后的图像
        """
        original_spacing = image_sitk.GetSpacing()  # (x, y, z)
        original_size = image_sitk.GetSize()
        
        # 🔑 关键修正：转换spacing顺序
        # self.target_spacing是(z,y,x)，SimpleITK需要(x,y,z)
        target_spacing_sitk = (
            self.target_spacing[2],  # x
            self.target_spacing[1],  # y
            self.target_spacing[0]   # z
        )
        
        # 计算新的尺寸
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing_sitk[i]))
            for i in range(3)
        ]
        
        # 设置重采样参数
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing_sitk)
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(self.interpolator_map[self.interpolation])
        
        resampled = resampler.Execute(image_sitk)
        
        return resampled    
    
    def extract_features(self):
        """提取空间特征"""
        features = {}
        
        try:
            coords, values = self._prepare_data()

            n_voxels = len(coords)
            
            if n_voxels < 10:
                # print(f"  ⚠️  Too few voxels ({n_voxels}) for {self.feature_name}, skipping...")
                return self._get_empty_features()
            
            k = min(26, n_voxels - 1)

            # 构建空间权重矩阵
            # 🔑 关键修正：使用物理坐标而非体素坐标
            coords_mm = coords * np.array(self.target_spacing)
            w = KNN.from_array(coords_mm, k=k)

            # Moran's I
            moran = Moran(values, w)
            features[f'{self.feature_name}_moran_i'] = moran.I
            features[f'{self.feature_name}_moran_p'] = moran.p_sim

            # Geary's C
            geary = Geary(values, w)
            features[f'{self.feature_name}_geary_c'] = geary.C
            features[f'{self.feature_name}_geary_p'] = geary.p_sim

            # Local statistics
            lisa = Moran_Local(values, w)
            features[f'{self.feature_name}_hotspot_fraction'] = (lisa.q == 1).mean()
            features[f'{self.feature_name}_coldspot_fraction'] = (lisa.q == 2).mean()
            features[f'{self.feature_name}_hl_outlier_fraction'] = (lisa.q == 3).mean()
            features[f'{self.feature_name}_lh_outlier_fraction'] = (lisa.q == 4).mean()
            features[f'{self.feature_name}_insignificant_fraction'] = (lisa.q == 0).mean()
            features[f'{self.feature_name}_spatial_variability'] = np.std(lisa.Is)
            features[f'{self.feature_name}_n_voxels'] = n_voxels
            features[f'{self.feature_name}_success'] = True

        except Exception as e:
            print(f"  ✗ Error processing {self.feature_name}: {str(e)}")
            features = self._get_empty_features()
        
        return features
    
    def _prepare_data(self):
        """准备坐标和值数据"""
        coords = np.argwhere(self.featuremap > 0)
        values = self.featuremap[self.featuremap > 0].astype(np.float64)
        return coords, values
    
    def _get_empty_features(self):
        """返回空特征（处理失败时）"""
        return {
            f'{self.feature_name}_moran_i': np.nan,
            f'{self.feature_name}_moran_p': np.nan,
            f'{self.feature_name}_geary_c': np.nan,
            f'{self.feature_name}_geary_p': np.nan,
            f'{self.feature_name}_hotspot_fraction': np.nan,
            f'{self.feature_name}_coldspot_fraction': np.nan,
            f'{self.feature_name}_spatial_variability': np.nan,
            f'{self.feature_name}_n_voxels': 0,
            f'{self.feature_name}_success': False
        }


class BatchSpatialTextureProcessor:
    """批量处理所有病例的纹理特征图"""
    
    def __init__(self, feature_maps_dir, mask_dir, target_spacing, 
                 output_csv='spatial_texture_features.csv', interpolation='linear'):
        """
        Parameters:
        -----------
        feature_maps_dir : str or Path
            包含所有.pkl.gz文件的目录
        mask_dir : str or Path
            mask文件目录（用于获取原始spacing）
        target_spacing : tuple or list
            目标spacing (z, y, x) in mm
        output_csv : str
            输出CSV文件路径
        interpolation : str
            插值方法
        """
        self.feature_maps_dir = Path(feature_maps_dir)
        self.mask_dir = Path(mask_dir)
        self.output_csv = output_csv
        self.target_spacing = tuple(target_spacing)  # 确保是tuple
        self.interpolation = interpolation
        
    def process_single_case(self, pkl_path, case_id=None):
        """
        处理单个病例的所有纹理特征图
        
        Parameters:
        -----------
        pkl_path : str or Path
            .pkl.gz文件路径
        case_id : str, optional
            病例ID
        
        Returns:
        --------
        case_features : dict
            该病例的所有特征
        """
        if case_id is None:
            # 🔑 修正：提取case_id
            pkl_name = Path(pkl_path).stem.replace('.pkl', '')
            case_id = pkl_name.split('_')[0]  # 假设格式为 "caseID_hash.pkl.gz"
        
        print(f"\nProcessing case: {case_id}")
        
        case_features = {'case_id': case_id}
        
        try:
            # 🔑 修正：读取mask获取原始spacing
            mask_path = os.path.join(self.mask_dir, f'{case_id}.nii.gz')
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            
            mask_sitk = sitk.ReadImage(mask_path)
            original_spacing = mask_sitk.GetSpacing()  # (x, y, z)
            
            # 加载特征图
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # 遍历所有filter和texture
            texture_count = 0
            successful_count = 0
            
            for filter_name in data.keys():
                for texture_name in data[filter_name].keys():
                    featuremap = data[filter_name][texture_name]
                    
                    # 🔑 修正：验证特征图不为空
                    if not isinstance(featuremap, np.ndarray) or featuremap.size == 0:
                        print(f"  ⚠️  Invalid feature map for {filter_name}_{texture_name}")
                        continue
                    
                    # 创建完整的特征名称
                    full_feature_name = f"{filter_name}_{texture_name}"
                    
                    # 提取空间特征
                    extractor = SpatialTextureFeatureExtractor(
                        featuremap_3d=featuremap,
                        original_spacing=original_spacing,
                        target_spacing=self.target_spacing,
                        feature_name=full_feature_name,
                        interpolation=self.interpolation
                    )
                    
                    features = extractor.extract_features()
                    case_features.update(features)
                    
                    texture_count += 1
                    if features.get(f'{full_feature_name}_success', False):
                        successful_count += 1
            
            case_features['total_textures_processed'] = texture_count
            case_features['successful_textures'] = successful_count
            case_features['case_success'] = successful_count > 0
            
            print(f"  ✓ Processed {successful_count}/{texture_count} texture features for {case_id}")
            
        except Exception as e:
            print(f"  ✗ Error processing {case_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            case_features['case_success'] = False
            case_features['error_message'] = str(e)
        
        return case_features
    
    def process_all_cases(self, pattern='*.pkl.gz'):
        """
        处理目录中的所有病例（串行）
        
        Parameters:
        -----------
        pattern : str
            文件匹配模式
        
        Returns:
        --------
        df : pd.DataFrame
            所有病例的特征DataFrame
        """
        pkl_files = sorted(self.feature_maps_dir.glob(pattern))
        
        if len(pkl_files) == 0:
            raise ValueError(f"No files found matching {pattern} in {self.feature_maps_dir}")
        
        print(f"\n{'='*60}")
        print(f"Found {len(pkl_files)} cases to process")
        print(f"Target spacing: {self.target_spacing}")
        print(f"{'='*60}")
        
        all_features = []
        
        # 使用tqdm显示进度
        for pkl_path in tqdm(pkl_files, desc="Processing cases"):
            features = self.process_single_case(pkl_path)
            all_features.append(features)
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(all_features)
        df.to_csv(self.output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {self.output_csv}")
        print(f"Total cases: {len(df)}")
        print(f"Successful cases: {df['case_success'].sum()}")
        print(f"Failed cases: {(~df['case_success']).sum()}")
        print(f"{'='*60}")
        
        self._print_summary(df)
        
        return df
    
    def process_all_cases_parallel(self, pattern='*.pkl.gz', n_jobs=None):
        """
        并行处理所有病例（推荐）
        
        Parameters:
        -----------
        pattern : str
            文件匹配模式
        n_jobs : int, optional
            并行进程数
        
        Returns:
        --------
        df : pd.DataFrame
            所有病例的特征DataFrame
        """
        import multiprocessing
        
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        
        pkl_files = sorted(self.feature_maps_dir.glob(pattern))
        
        if len(pkl_files) == 0:
            raise ValueError(f"No files found matching {pattern} in {self.feature_maps_dir}")
        
        print(f"\n{'='*60}")
        print(f"Found {len(pkl_files)} cases to process")
        print(f"Target spacing: {self.target_spacing}")
        print(f"Using {n_jobs} parallel workers")
        print(f"{'='*60}")
        
        all_features = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # 提交所有任务
            future_to_case = {
                executor.submit(self.process_single_case, str(pkl_path)): pkl_path
                for pkl_path in pkl_files
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_case), 
                             total=len(pkl_files), 
                             desc="Processing"):
                try:
                    features = future.result()
                    all_features.append(features)
                except Exception as e:
                    pkl_path = future_to_case[future]
                    case_id = pkl_path.stem.replace('.pkl', '').split('_')[0]
                    print(f"\n✗ Fatal error in parallel processing of {case_id}: {e}")
                    all_features.append({
                        'case_id': case_id,
                        'case_success': False,
                        'error_message': str(e)
                    })
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(all_features)
        df.to_csv(self.output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {self.output_csv}")
        print(f"Total cases: {len(df)}")
        print(f"Successful cases: {df['case_success'].sum()}")
        print(f"Failed cases: {(~df['case_success']).sum()}")
        print(f"{'='*60}")
        
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df):
        """打印特征统计摘要"""
        print("\n📊 FEATURE STATISTICS:")
        print("="*60)
        
        df_success = df[df['case_success'] == True]
        
        if len(df_success) == 0:
            print("No successful cases to summarize")
            return
        
        # 获取所有moran_i特征列
        moran_cols = [col for col in df.columns if col.endswith('_moran_i')]
        geary_cols = [col for col in df.columns if col.endswith('_geary_c')]
        
        print(f"\nTotal texture features per case: {len(moran_cols)}")
        
        if len(moran_cols) > 0:
            print(f"Average Moran's I across all textures: {df_success[moran_cols].mean().mean():.4f}")
            print(f"Average Geary's C across all textures: {df_success[geary_cols].mean().mean():.4f}")
            
            # 找出空间自相关最强的纹理特征
            print("\n🔝 TOP 5 TEXTURES WITH STRONGEST SPATIAL AUTOCORRELATION:")
            mean_morans = df_success[moran_cols].mean().sort_values(ascending=False)
            for i, (feat, val) in enumerate(mean_morans.head(5).items(), 1):
                texture_name = feat.replace('_moran_i', '')
                print(f"  {i}. {texture_name}: {val:.4f}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # 示例 1: 并行处理（推荐）
    # ========================================================================
    print("\n" + "="*60)
    print("PARALLEL PROCESSING")
    print("="*60)
    
    # processor_parallel = BatchSpatialTextureProcessor(
    #     feature_maps_dir='XXXyour path/Dataset/Dataset002_ECResponse/TJH/FeatureMaps/feature_maps/',
    #     mask_dir='XXXyour path/Dataset/Dataset002_ECResponse/TJH/Mask',
    #     target_spacing=[1.0, 1.0, 1.0],  # (z, y, x) in mm
    #     output_csv='XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial_texture_features.csv',
    #     interpolation='linear'
    # )

    processor_parallel = BatchSpatialTextureProcessor(
        feature_maps_dir='XXXyour path/Dataset/Dataset002_ECResponse/SPCH/FeatureMaps/feature_maps/',
        mask_dir='XXXyour path/Dataset/Dataset002_ECResponse/SPCH/Mask',
        target_spacing=[1.0, 1.0, 1.0],  # (z, y, x) in mm
        output_csv='XXXyour path/Dataset/Dataset002_ECResponse/SPCH/spatial_texture_features.csv',
        interpolation='linear'
    )

    # processor_parallel = BatchSpatialTextureProcessor(
    #     feature_maps_dir='XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/FeatureMaps/feature_maps/',
    #     mask_dir='XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Mask',
    #     target_spacing=[1.0, 1.0, 1.0],  # (z, y, x) in mm
    #     output_csv='XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/spatial_texture_features.csv',
    #     interpolation='linear'
    # )

    # processor_parallel = BatchSpatialTextureProcessor(
    #     feature_maps_dir='XXXyour path/Dataset/Dataset004-NACI/FUSCC/FeatureMaps/feature_maps/',
    #     mask_dir='XXXyour path/Dataset/Dataset004-NACI/FUSCC/Mask',
    #     target_spacing=[1.0, 1.0, 1.0],  # (z, y, x) in mm
    #     output_csv='XXXyour path/Dataset/Dataset004-NACI/FUSCC/spatial_texture_features.csv',
    #     interpolation='linear'
    # )

    # processor_parallel = BatchSpatialTextureProcessor(
    #     feature_maps_dir='XXXyour path/Dataset/Dataset004-NACI/SHPH/FeatureMaps/feature_maps/',
    #     mask_dir='XXXyour path/Dataset/Dataset004-NACI/SHPH/Mask',
    #     target_spacing=[1.0, 1.0, 1.0],  # (z, y, x) in mm
    #     output_csv='XXXyour path/Dataset/Dataset004-NACI/SHPH/spatial_texture_features.csv',
    #     interpolation='linear'
    # )
    # processor_parallel = BatchSpatialTextureProcessor(
    #     feature_maps_dir='XXXyour path/Dataset/Dataset004-NACI/SPCH/FeatureMaps/feature_maps/',
    #     mask_dir='XXXyour path/Dataset/Dataset004-NACI/SPCH/Mask',
    #     target_spacing=[1.0, 1.0, 1.0],  # (z, y, x) in mm
    #     output_csv='XXXyour path/Dataset/Dataset004-NACI/SPCH/spatial_texture_features.csv',
    #     interpolation='linear'
    # )
    df_parallel = processor_parallel.process_all_cases_parallel(
        pattern='*.pkl.gz',
        # n_jobs=28 OOM-crashed the whole pool (each worker loads full-image-
        # resolution feature maps into memory, ~8GB+ per worker and likely
        # 25-35GB for the largest patients). Much more conservative here.
        n_jobs=6
    )
    
    
    # ========================================================================
    # 示例 2: 串行处理（调试用）
    # ========================================================================
    # print("\n" + "="*60)
    # print("SERIAL PROCESSING (for debugging)")
    # print("="*60)
    # 
    # processor_serial = BatchSpatialTextureProcessor(
    #     feature_maps_dir=r'E:\Spatial-Radiomics\cache\feature_maps',
    #     mask_dir=r'E:\Spatial-Radiomics\Mask',
    #     target_spacing=[1.0, 1.0, 1.0],
    #     output_csv=r'E:\Spatial-Radiomics\results\spatial_texture_features_serial.csv',
    #     interpolation='linear'
    # )
    # 
    # df_serial = processor_serial.process_all_cases(pattern='*.pkl.gz')
    
    
    # ========================================================================
    # 示例 3: 单个病例测试
    # ========================================================================
    # print("\n" + "="*60)
    # print("SINGLE CASE TEST")
    # print("="*60)
    # 
    # processor_test = BatchSpatialTextureProcessor(
    #     feature_maps_dir=r'E:\Spatial-Radiomics\cache\feature_maps',
    #     mask_dir=r'E:\Spatial-Radiomics\Mask',
    #     target_spacing=[1.0, 1.0, 1.0],
    #     output_csv=r'E:\Spatial-Radiomics\results\test_single_case.csv'
    # )
    # 
    # test_pkl = 'XXXyour path/Dataset/Dataset002_ECResponse/cache/feature_maps/CASE_ID.pkl.gz'
    # features = processor_test.process_single_case(test_pkl, case_id='CASE_ID')
    # 
    # print("\nSample features extracted:")
    # sample_features = {k: v for k, v in list(features.items())[:15]}
    # for key, value in sample_features.items():
    #     if isinstance(value, float):
    #         print(f"  {key}: {value:.4f}")
    #     else:
    #         print(f"  {key}: {value}")