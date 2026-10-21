import os
from radiomics import featureextractor
import pandas as pd
import six
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadiomicsExtractor:
    def __init__(self, image_dir, mask_dir, output_csv, config_file=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_csv = output_csv
        
        # Initialize with config file if provided
        if config_file and os.path.exists(config_file):
            self.extractor = featureextractor.RadiomicsFeatureExtractor(config_file)
            logger.info(f"Loaded configuration from {config_file}")
        else:
            self.extractor = featureextractor.RadiomicsFeatureExtractor()
            logger.info("Using default PyRadiomics settings")
    
    def extract_features_single(self, image_path, mask_path, case_id):
        try:
            logger.info(f"Processing {case_id}")
            features = self.extractor.execute(str(image_path), str(mask_path))
            
            feature_dict = {'CaseID': case_id}
            for key, val in six.iteritems(features):
                feature_dict[key] = val
            
            logger.info(f"Extracted {len(feature_dict)} features for {case_id}")
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error processing {case_id}: {str(e)}")
            return None
    
    def match_image_mask_pairs(self):
        image_files = sorted(self.image_dir.glob('*.nii.gz'))
        pairs = []
        
        for img_file in image_files:
            case_id = img_file.stem.replace('.nii', '')
            mask_file = self.mask_dir / img_file.name
            
            if mask_file.exists():
                pairs.append((img_file, mask_file, case_id))
            else:
                logger.warning(f"No matching mask found for {img_file.name}")
        
        logger.info(f"Found {len(pairs)} matching image-mask pairs")
        return pairs
    
    def extract_all_features(self):
        pairs = self.match_image_mask_pairs()
        
        if not pairs:
            logger.error("No matching image-mask pairs found!")
            return
        
        all_features = []
        for image_path, mask_path, case_id in pairs:
            features = self.extract_features_single(image_path, mask_path, case_id)
            if features:
                all_features.append(features)
        
        if all_features:
            df = pd.DataFrame(all_features)
            df.to_csv(self.output_csv, index=False)
            logger.info(f"Features saved to {self.output_csv}")
            logger.info(f"Total cases processed: {len(all_features)}")


if __name__ == "__main__":
    # extractor = RadiomicsExtractor(
    #     image_dir="XXXyour path/Dataset/Dataset001_PDL1Expression/imagesTr",
    #     mask_dir="XXXyour path/Dataset/Dataset001_PDL1Expression/labelsTr",
    #     output_csv="radiomics_features.csv",
    #     config_file="radiomics_config.yaml"  # Optional
    # )
    
    # extractor = RadiomicsExtractor(
    #     image_dir="XXXyour path/Dataset/Dataset002_ECResponse/TJH/Image",
    #     mask_dir="XXXyour path/Dataset/Dataset002_ECResponse/TJH/Mask",
    #     output_csv="XXXyour path/Dataset/Dataset002_ECResponse/TJH/radiomics_features.csv",
    #     config_file="radiomics_config.yaml"  # Optional
    # )
    # extractor.extract_all_features()
    
    extractor = RadiomicsExtractor(
        image_dir="XXXyour path/Dataset/Dataset002_ECResponse/SPCH/Image",
        mask_dir="XXXyour path/Dataset/Dataset002_ECResponse/SPCH/Mask",
        output_csv="XXXyour path/Dataset/Dataset002_ECResponse/SPCH/radiomics_features.csv",
        config_file="radiomics_config.yaml"  # Optional
    )
    # extractor.extract_all_features()

    # extractor = RadiomicsExtractor(
    #     image_dir="XXXyour path/Dataset/Dataset002_ECResponse/Image",
    #     mask_dir="XXXyour path/Dataset/Dataset002_ECResponse/Mask",
    #     output_csv="XXXyour path/Dataset/Dataset002_ECResponse/radiomics_features.csv",
    #     config_file="radiomics_config.yaml"  # Optional
    # )
    # extractor.extract_all_features()

    # extractor = RadiomicsExtractor(
    #     image_dir="XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Image",
    #     mask_dir="XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Mask",
    #     output_csv="XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/radiomics_features.csv",
    #     config_file="radiomics_config.yaml"  # Optional
    # )
    # extractor = RadiomicsExtractor(
    #     image_dir="XXXyour path/Dataset/Dataset004-NACI/SHPH/Image",
    #     mask_dir="XXXyour path/Dataset/Dataset004-NACI/SHPH/Mask",
    #     output_csv="XXXyour path/Dataset/Dataset004-NACI/SHPH/radiomics_features.csv",
    #     config_file="radiomics_config.yaml"  # Optional
    # )
    # extractor = RadiomicsExtractor(
    #     image_dir="XXXyour path/Dataset/Dataset004-NACI/FUSCC/Image",
    #     mask_dir="XXXyour path/Dataset/Dataset004-NACI/FUSCC/Mask",
    #     output_csv="XXXyour path/Dataset/Dataset004-NACI/FUSCC/radiomics_features.csv",
    #     config_file="radiomics_config.yaml"  # Optional
    # )
    # extractor = RadiomicsExtractor(
    #     image_dir="XXXyour path/Dataset/Dataset004-NACI/SPCH/Image",
    #     mask_dir="XXXyour path/Dataset/Dataset004-NACI/SPCH/Mask",
    #     output_csv="XXXyour path/Dataset/Dataset004-NACI/SPCH/radiomics_features.csv",
    #     config_file="radiomics_config.yaml"  # Optional
    # )
        
    extractor.extract_all_features()