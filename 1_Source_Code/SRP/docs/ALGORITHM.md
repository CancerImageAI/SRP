# Algorithm reference

This is a detailed, implementation-level description of the SRP
(Spatial + Radiomics -> Fusion) pipeline's core algorithms, complementing
[USAGE.md](USAGE.md) (which covers *how to run* each script) with *what
each one computes*. A condensed, manuscript-facing version of this
document is submitted as a Supplementary Note alongside the paper (see
that note for the formal citation list for third-party methods used
elsewhere in this pipeline — DESeq2, GSEA, Seurat, CellChat).

## 1. Spatial-3D feature extraction

`feature_extraction/Spatial3DFeatureExtractor.py`
- `Spatial3DFeatureExtractor` — per-case feature computation from a CT
  image + tumor mask: resamples both to an isotropic target spacing
  (default 1x1x1 mm, linear interpolation), crops to the tumor
  bounding box, and computes whole-tumor 3D shape/heterogeneity
  descriptors directly from the mask geometry.
- `SpatialRadiomicsPipeline` — batch driver: iterates a case list,
  calls `Spatial3DFeatureExtractor` per case, and assembles the
  per-patient feature table (`spatial3d_features.csv`).

## 2. Spatial-texture ("heterogeneity-of-heterogeneity") feature extraction

`feature_extraction/TextureFeatureMapGenerate.py` generates the
per-voxel feature-map stack first (local-neighborhood texture/entropy
statistics computed at every voxel inside the mask, one 3D map per
statistic), persisted as compressed pickles (`FeatureMaps/feature_maps/*.pkl.gz`).

`feature_extraction/SpatialTextureFeatureExtractor.py`
- `SpatialTextureFeatureExtractor` — per-case: resamples the feature
  maps, restricts to the masked tumor volume, builds a spatial
  contiguity weights matrix over the masked voxels, and computes
  global Moran's I spatial autocorrelation (via `esda`/`libpysal`) per
  feature map — quantifying whether each local feature's values are
  spatially clustered, dispersed, or random across the tumor.
- `BatchSpatialTextureProcessor` — batch driver with
  `process_all_cases_parallel(n_jobs=...)` for multi-core extraction;
  writes `spatial_texture_features.csv`.

## 3. Conventional radiomics feature extraction

`feature_extraction/RadiomicsFeatureExtractor.py`
- `RadiomicsExtractor` — thin wrapper around PyRadiomics
  (van Griethuysen et al. 2017), configured by
  `feature_extraction/radiomics_config.yaml` (feature classes: shape,
  first-order, GLCM, GLRLM, GLSZM, etc.; image filters). Writes
  `radiomics_features.csv`.

## 4. Per-modality classifier training

`modeling/SpatialClassificationModel.py`, `modeling/RadiomicsClassificationModel.py`,
`modeling/SpatialRadiomicsClassifier.py`
- Each modality's training follows the same template: `StandardScaler`
  fit on the training fold only -> feature selection (recursive
  feature elimination / embedded selection, cross-validated) ->
  SMOTE oversampling of the training fold only (never applied to
  validation/test data) -> `XGBClassifier` fit with cross-validated
  hyperparameters.
- `SpatialRadiomicsClassifier.FeatureSelector` (in
  `modeling/SpatialRadiomicsClassifier.py`) implements the feature
  selection step shared across modalities.

## 5. SRP fusion and 6. external cohort application

`external_validation/PredictNSCLC.py` is the canonical, minimal
reference implementation of both the fusion rule and how to apply a
*locked* (already-trained) model to a brand-new cohort:

```python
proba_fused = FUSION_WEIGHT_SPATIAL * proba_spatial + (1 - FUSION_WEIGHT_SPATIAL) * proba_radiomics
risk_group = np.where(proba_fused > 0.5, 'High-risk (predicted Non-pCR)', 'Low-risk (predicted pCR)')
```

Before this line, the script: loads the locked
`(scaler, selector, model)` triple for each modality from
`best_models.joblib`; merges the new cohort's spatial-3D,
spatial-texture, and radiomics feature tables by `case_id` (inner
join); and drops any patient missing a feature required by either
model (reporting which/how many). This is the template to adapt for
scoring any new cohort.

## 7. Binormal ROC/DCA smoothing

`evaluation/BinormalROCFit_Dataset002.py`,
`evaluation/BinormalROCFit_Dataset004NACI.py`
- `fit_binormal(y_true, y_proba)` — logit-transforms the raw predicted
  probabilities, fits a Gaussian to each class's transformed scores,
  and returns the closed-form binormal AUC
  (`Phi(a / sqrt(1+b**2))`, `a=(mu1-mu0)/sigma1`, `b=sigma0/sigma1`) plus a
  `posterior_fn` closure implementing the Bayesian recalibration
  `p_smooth = pi1*f1(z) / (pi1*f1(z) + pi0*f0(z))`.
- `plot_smooth_binormal_roc(...)` plots the literal closed-form curve
  (continuous, not an empirical step function).
- The recalibrated `p_smooth` scores are then fed back through the
  existing `MedicalImagingVisualizer` (`evaluation/PlotFigures.py`) to
  regenerate ROC/DCA/calibration/radar figures from continuous scores
  — used only for cohorts with heavily tied raw probabilities (see
  each script's module docstring for which cohorts and why).

Use this when: a cohort's raw predicted probabilities have a small
number of distinct values relative to sample size (shallow, few-feature
classifiers), making the empirical ROC/DCA curve a jagged staircase
rather than informative. Validate before presenting: compare the
closed-form binormal AUC against the raw empirical AUC
(`roc_auc_score`) — both scripts print this comparison.

## 8. Bootstrap confidence intervals

`evaluation/BinormalAUC_BootstrapCI.py`
- `bootstrap_binormal_auc_ci(y_true, y_proba, n_boot=2000, seed=42)` —
  percentile bootstrap: resample patients with replacement, refit the
  binormal model (step 7) on each resample, discard resamples lacking
  both classes, and take the [2.5, 97.5] percentile of the resulting
  AUC distribution.
