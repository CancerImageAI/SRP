"""
Provenance script, kept for transparency/audit purposes only -- it
requires the private Dataset/ directory and CANNOT be run by reviewers
or from the public repository/zip. It documents exactly how
ecresponse_cohort_deidentified.csv (which IS shipped alongside this
script) was derived from the real, restricted-access cohort data.

Dataset: the manuscript's Dataset002_ECResponse "Test" cohort (n=110)
-- esophageal squamous cell carcinoma (ESCC), one bounded validation
split out of the full study. Only already-extracted, already-tabular
feature/label/prediction files are touched -- no image data.

Included per patient:
  - real extracted feature values (radiomics, 3D spatial, spatial-texture)
  - real survival outcomes (DFS/OS event + time in months)
  - the REAL, already-trained, locked SRP model's actual historical
    Spatial / Radiomics / Fused probabilities for this patient (loaded
    from model_results/test_predictions_results.csv -- these are
    genuine predictions from the properly-trained production model,
    not re-derived or retrained here, which is why this demo's
    reported AUCs match the manuscript's own cohort-level statistics
    rather than a simplified/undertrained from-scratch reproduction)

What is deliberately NOT included: real case identifiers (replaced
with anonymous sequential IDs, randomly reordered), CT images or
segmentation masks, any date, or any clinical field beyond the four
survival/outcome columns (no age, sex, histology, etc.).
"""
import pandas as pd
from pathlib import Path

ROOT = Path('XXXyour path/Dataset/Dataset002_ECResponse')  # private, restricted-access cohort data
OUT_PATH = Path('XXXyour path/NatureSubmission/2_Demo/data/ecresponse_cohort_deidentified.csv')
SEED = 42

preds = pd.read_csv(ROOT / 'model_results' / 'test_predictions_results.csv')
preds = preds[['case_id', 'true_label', 'spatial_probability', 'radiomics_probability', 'fused_probability']].copy()
preds['case_id'] = preds['case_id'].astype(int)

rad = pd.read_csv(ROOT / 'radiomics_features.csv')
rad = rad.rename(columns={'CaseID': 'case_id'})
rad['case_id'] = rad['case_id'].astype(int)
diag_cols = [c for c in rad.columns if 'diagnostics' in c.lower()]
rad = rad.drop(columns=diag_cols)
rad = rad.add_prefix('radiomics__')
rad = rad.rename(columns={'radiomics__case_id': 'case_id'})

sp3 = pd.read_csv(ROOT / 'spatial3d_features.csv')
sp3['case_id'] = sp3['case_id'].str.replace('.nii', '', regex=False).astype(int)
sp3 = sp3.add_prefix('spatial3d__')
sp3 = sp3.rename(columns={'spatial3d__case_id': 'case_id'})

spt = pd.read_csv(ROOT / 'spatial_texture_features.csv')
spt['case_id'] = spt['case_id'].astype(int)
spt = spt.add_prefix('spatialtex__')
spt = spt.rename(columns={'spatialtex__case_id': 'case_id'})

surv = pd.read_excel(ROOT / 'Test_PatientLabels.xlsx')
surv = surv.rename(columns={'CaseID': 'case_id'})
surv['case_id'] = pd.to_numeric(surv['case_id'], errors='coerce')
surv = surv.dropna(subset=['case_id'])
surv['case_id'] = surv['case_id'].astype(int)
surv = surv[['case_id', 'DFS', 'DFS-Time', 'OS', 'OS-Time']].rename(columns={
    'DFS': 'dfs_event', 'DFS-Time': 'dfs_time_months',
    'OS': 'os_event', 'OS-Time': 'os_time_months',
})

merged = preds.merge(rad, on='case_id', how='inner') \
              .merge(sp3, on='case_id', how='inner') \
              .merge(spt, on='case_id', how='inner') \
              .merge(surv, on='case_id', how='left')
print(f"Merged cohort: {len(merged)} patients, {merged.shape[1]} columns")
print(merged['true_label'].value_counts())

import numpy as np
rng = np.random.default_rng(SEED)
shuffled_idx = rng.permutation(len(merged))
merged = merged.iloc[shuffled_idx].reset_index(drop=True)
merged.insert(0, 'demo_id', [f'ECR-DEMO-{i+1:03d}' for i in range(len(merged))])
merged = merged.drop(columns=['case_id'])

success_cols = [c for c in merged.columns if c.endswith('_success')]
merged = merged.drop(columns=success_cols)
all_nan_cols = [c for c in merged.columns if merged[c].isna().all()]
merged = merged.drop(columns=all_nan_cols)
print(f"Dropped {len(success_cols)} bookkeeping column(s), {len(all_nan_cols)} all-missing feature column(s)")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"Saved de-identified cohort to {OUT_PATH}")
print(f"Columns: {merged.shape[1]}")
