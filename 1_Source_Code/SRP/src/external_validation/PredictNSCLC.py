"""
Apply the current best SRP (Fusion) model -- Spatial model + the new
1000-trial sum-AUC-selected Radiomics model, fused at weight_spatial=0.8
-- to the NSCLC-CT-RNAseq cohort (no ground-truth labels; pure inference).

Masks were preprocessed to keep only the largest connected tumor
component per case (see preprocess_nsclc_masks.py) before feature
extraction, per the original request.

Output: one row per patient with spatial/radiomics/fused probability and
a binary risk group (High-risk = predicted Non-pCR, Low-risk = predicted
pCR), using the standard 0.5 cutoff on the fused probability to match how
"predicted non-pCR"/"predicted pCR" is defined everywhere else in this
pipeline.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA_ROOT = Path('XXXyour path/Dataset/Dataset004-NACI')
NSCLC_DIR = DATA_ROOT / 'NSCLC-CT-RNAseq'
FUSION_WEIGHT_SPATIAL = 0.8  # fusion_new_radiomics_best_weight


def strip_nii(case_id):
    s = str(case_id).strip()
    return s.split('.nii')[0] if '.nii' in s else s


print("=" * 70)
print("PREDICT NSCLC-CT-RNAseq WITH SRP (FUSION) MODEL")
print("=" * 70)

artifacts = joblib.load(DATA_ROOT / 'model_results' / 'best_models.joblib')
spatial_scaler = artifacts['spatial_scaler']
spatial_selector = artifacts['spatial_selector']
spatial_model = artifacts['spatial_model']
spatial_cols = artifacts['spatial_feature_cols']
rad_scaler = artifacts['radiomics_scaler']
rad_selector = artifacts['radiomics_selector_1000trial_sumauc']
rad_model = artifacts['radiomics_model_1000trial_sumauc']
radiomics_cols = artifacts['radiomics_feature_cols']
print(f"  Loaded SRP model: {len(spatial_cols)} spatial candidate cols, "
      f"{len(radiomics_cols)} radiomics candidate cols, fusion weight_spatial={FUSION_WEIGHT_SPATIAL}")

print("\n" + "=" * 70)
print("STEP 1: LOAD & MERGE NSCLC-CT-RNAseq FEATURES")
print("=" * 70)

df_spatial3d = pd.read_csv(NSCLC_DIR / 'spatial3d_features.csv')
df_texture = pd.read_csv(NSCLC_DIR / 'spatial_texture_features.csv')
df_rad = pd.read_csv(NSCLC_DIR / 'radiomics_features.csv')

df_spatial3d['case_id'] = df_spatial3d['case_id'].apply(strip_nii)
df_texture['case_id'] = df_texture['case_id'].apply(strip_nii)
df_rad = df_rad.rename(columns={'CaseID': 'case_id'})
df_rad['case_id'] = df_rad['case_id'].astype(str)
diagnostics_cols = [c for c in df_rad.columns if 'diagnostics' in c.lower()]
df_rad = df_rad.drop(columns=diagnostics_cols)

print(f"  spatial3d: {len(df_spatial3d)}  texture: {len(df_texture)}  radiomics: {len(df_rad)}")

merged = pd.merge(df_spatial3d, df_texture, on='case_id', how='inner', suffixes=('', '_tex'))
merged = pd.merge(merged, df_rad, on='case_id', how='inner', suffixes=('', '_rad'))
merged = merged.drop_duplicates(subset='case_id').reset_index(drop=True)
print(f"  Merged (all 3 feature sources present): {len(merged)} cases")

valid = merged.reindex(columns=spatial_cols).notna().all(axis=1) & \
        merged.reindex(columns=radiomics_cols).notna().all(axis=1)
n_dropped = (~valid).sum()
if n_dropped:
    dropped_ids = merged.loc[~valid, 'case_id'].tolist()
    print(f"  Dropping {n_dropped} case(s) with missing values in the model's "
          f"candidate feature set: {dropped_ids}")
usable = merged.loc[valid].reset_index(drop=True)
print(f"  Final cohort for prediction: {len(usable)} / {len(merged)} cases")

print("\n" + "=" * 70)
print("STEP 2: APPLY SRP MODEL")
print("=" * 70)

X_spatial = usable[spatial_cols]
X_spatial_scaled = spatial_scaler.transform(X_spatial)
X_spatial_selected = spatial_selector.transform(X_spatial_scaled)
proba_spatial = spatial_model.predict_proba(X_spatial_selected)[:, 1]

X_rad = usable[radiomics_cols]
X_rad_scaled = rad_scaler.transform(X_rad)
X_rad_selected = rad_selector.transform(X_rad_scaled)
proba_radiomics = rad_model.predict_proba(X_rad_selected)[:, 1]

proba_fused = FUSION_WEIGHT_SPATIAL * proba_spatial + (1 - FUSION_WEIGHT_SPATIAL) * proba_radiomics

risk_group = np.where(proba_fused > 0.5, 'High-risk (predicted Non-pCR)', 'Low-risk (predicted pCR)')

report = pd.DataFrame({
    'case_id': usable['case_id'].values,
    'spatial_probability': proba_spatial,
    'radiomics_probability': proba_radiomics,
    'fused_probability': proba_fused,
    'risk_group': risk_group,
}).sort_values('fused_probability', ascending=False).reset_index(drop=True)

# Cases dropped for missing features are still listed, flagged as not scored
if n_dropped:
    dropped_report = pd.DataFrame({
        'case_id': merged.loc[~valid, 'case_id'].values,
        'spatial_probability': np.nan,
        'radiomics_probability': np.nan,
        'fused_probability': np.nan,
        'risk_group': 'Not scored (missing features)',
    })
    report = pd.concat([report, dropped_report], ignore_index=True)

out_path = NSCLC_DIR / 'model_results'
out_path.mkdir(parents=True, exist_ok=True)
report_csv = out_path / 'NSCLC_risk_predictions.csv'
report.to_csv(report_csv, index=False)

print(f"\n  Risk group counts:")
print(report['risk_group'].value_counts().to_string())
print(f"\n  Saved report to {report_csv}")
print("\nDONE")
