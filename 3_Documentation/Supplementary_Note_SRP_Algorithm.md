## Supplementary Note: SRP (Spatial–Radiomics–Fusion) algorithm

This note gives a complete, implementation-independent description of
the computational methods underlying the Spatial, Radiomics, and
Fusion ("SRP") models reported in the manuscript, and of the
statistical post-processing methods used to evaluate them. It is
intended to satisfy the Nature Research Code and Software Submission
Checklist's requirement for "a complete, detailed description of the
code's functionality (i.e. pseudocode)." The corresponding,
version-controlled source code is publicly available at
https://github.com/CancerImageAI/SRP (see the mapping table at the end
of this note for which script implements each algorithm below).

### Algorithm 1 — 3D spatial-heterogeneity feature extraction

**Input:** CT image *I*, tumor segmentation mask *M*
**Output:** a per-voxel feature-map stack *F*, and a whole-tumor
spatial feature vector

1. Resample *I* and *M* to an isotropic target voxel spacing (1 × 1 ×
   1 mm) using linear interpolation.
2. Crop to the tumor bounding box, with a fixed margin.
3. For every voxel *v* inside *M*, compute a local-neighborhood
   texture/heterogeneity statistic (e.g., local entropy, local
   variance) over a fixed-radius window centered at *v*, producing one
   scalar per voxel per statistic — i.e., one 3D feature map per
   statistic.
4. Persist the feature-map stack (used by Algorithm 2) and compute
   whole-tumor shape/heterogeneity descriptors directly from *M*
   (e.g., surface-to-volume ratio, shape irregularity).
5. Return the per-patient spatial feature vector.

### Algorithm 2 — Spatial-texture ("heterogeneity-of-heterogeneity") feature extraction

**Input:** feature-map stack *F* from Algorithm 1, tumor mask *M*
**Output:** patient-level spatial-texture feature vector

1. Resample each feature map in *F* to a common isotropic spacing and
   restrict to voxels inside *M*.
2. Construct a spatial contiguity weights matrix *W* over the masked
   voxels (queen/rook adjacency).
3. For each feature map, compute global Moran's I spatial
   autocorrelation statistic under *W*, quantifying whether that
   feature's values are spatially clustered, dispersed, or random
   across the tumor volume.
4. Concatenate the per-feature-map statistics into one feature vector
   per patient.

### Algorithm 3 — Conventional radiomics feature extraction

**Input:** CT image *I*, tumor mask *M*
**Output:** radiomics feature vector

1. Resample *I* and *M* to a common spacing.
2. Extract shape, first-order, and texture (GLCM, GLRLM, GLSZM, and
   related matrix-based) features using the PyRadiomics
   implementation¹, configured by a fixed feature-class/filter
   specification (`radiomics_config.yaml`).
3. Return the per-patient radiomics feature vector.

### Algorithm 4 — Per-modality classifier training

**Input:** a feature matrix *X* (either the spatial feature set
[Algorithms 1–2 concatenated] or the radiomics feature set [Algorithm
3]) and binary outcome labels *y* (pathologic complete response, pCR,
vs. Non-pCR) for the training cohort
**Output:** a fitted (scaler, feature-selector, classifier) tuple

1. Fit a `StandardScaler` (zero mean, unit variance) on the training
   features; apply the same transform to any later data.
2. Select a reduced, discriminative feature subset via a feature
   selection step (recursive feature elimination / embedded
   selection) evaluated by cross-validation.
3. Address class imbalance in the training fold using SMOTE
   oversampling (training fold only — never applied to validation or
   test data, to avoid leakage).
4. Fit a gradient-boosted tree classifier (XGBoost), with
   hyperparameters selected by a cross-validated search.
5. Return the fitted scaler, selector, and classifier.

### Algorithm 5 — SRP fusion

**Input:** a new patient's spatial feature vector *x*ₛ and radiomics
feature vector *x*ᵣ; the locked Spatial model and Radiomics model
(each a fitted scaler/selector/classifier triple from Algorithm 4);
fusion weight *w* ∈ [0, 1] (spatial weight)
**Output:** per-modality probabilities, fused probability, binary risk
group

1. *p*ₛ = Spatial-model.predict_proba(Spatial-selector.transform(Spatial-scaler.transform(*x*ₛ)))
2. *p*ᵣ = Radiomics-model.predict_proba(Radiomics-selector.transform(Radiomics-scaler.transform(*x*ᵣ)))
3. *p*fused = *w*·*p*ₛ + (1 − *w*)·*p*ᵣ
4. risk_group = "High-risk (predicted Non-pCR)" if *p*fused > 0.5, else "Low-risk (predicted pCR)"
5. Return *p*ₛ, *p*ᵣ, *p*fused, risk_group

### Algorithm 6 — Applying a locked model to a new cohort (external validation)

**Input:** a new cohort's extracted feature tables (spatial-3D,
spatial-texture, radiomics); the locked Spatial and Radiomics models
and fusion weight from Algorithms 4–5
**Output:** a per-patient prediction table

1. Merge the three feature tables by patient case identifier (inner
   join — a patient must have all three feature sources to be scored).
2. Drop any patient missing a value in a feature required by either
   locked model; report how many/which were dropped.
3. Apply Algorithm 5 to every remaining patient.
4. Return the prediction table, sorted by descending fused
   probability.

### Algorithm 7 — Binormal ROC/DCA smoothing (statistical post-processing)

Used only for cohorts whose raw predicted probabilities are heavily
tied (a symptom of shallow, few-feature classifiers evaluated on
small-to-moderate cohorts), where the empirical ROC/decision curve is
a jagged step function dominated by a handful of large ties rather
than a smooth curve.

**Input:** raw predicted probabilities {*p*ᵢ} and true labels {*y*ᵢ}
for a cohort
**Output:** closed-form smoothed ROC curve and continuous,
per-patient recalibrated probabilities suitable for decision curve
analysis (DCA)

1. Logit-transform: *z*ᵢ = logit(*p*ᵢ) = log(*p*ᵢ / (1 − *p*ᵢ)).
2. Fit a Gaussian *N*(μ₀, σ₀²) to {*z*ᵢ : *y*ᵢ = 0} and a Gaussian
   *N*(μ₁, σ₁²) to {*z*ᵢ : *y*ᵢ = 1} (the binormal model).
3. Compute *a* = (μ₁ − μ₀) / σ₁ and *b* = σ₀ / σ₁.
4. The closed-form binormal ROC curve is TPR(FPR) = Φ(*a* + *b*·Φ⁻¹(FPR)),
   with closed-form AUC = Φ(*a* / √(1 + *b*²)), where Φ is the
   standard normal CDF.
5. Recalibrate each patient's probability via the Bayesian posterior
   under the fitted binormal model:
   *p*smooth,ᵢ = π₁·*N*(*z*ᵢ; μ₁, σ₁²) / [π₁·*N*(*z*ᵢ; μ₁, σ₁²) + π₀·*N*(*z*ᵢ; μ₀, σ₀²)],
   where π₁, π₀ are the empirical class prevalences. This produces a
   continuous probability per patient (ties in the raw score map to
   identical, not necessarily distinct, smoothed values, since the
   transform is monotonic) usable as DCA input.
6. Validate the fit by checking that the closed-form binormal AUC
   agrees with the empirical (raw) AUC within a small tolerance before
   using the smoothed curves for presentation.

### Algorithm 8 — Bootstrap confidence intervals

**Input:** true labels *y*, probabilities *p* (raw or binormal-smoothed,
from Algorithm 5 or 7), number of resamples *B* (2,000 used throughout)
**Output:** a 95% confidence interval for AUC (or for the difference
in AUC, ΔAUC, between two models evaluated on the same cohort)

1. For *b* = 1 … *B*: resample patients with replacement; if the
   resample contains only one class, discard and redraw; otherwise
   compute AUC (or ΔAUC) on the resample.
2. Return the [2.5th, 97.5th] percentile of the *B* bootstrap
   statistics as the 95% confidence interval.

---

### Correspondence to the code repository

| Algorithm | Repository script (`src/...` in https://github.com/CancerImageAI/SRP) |
|---|---|
| 1. Spatial-3D feature extraction | `feature_extraction/Spatial3DFeatureExtractor.py` |
| 2. Spatial-texture feature extraction | `feature_extraction/SpatialTextureFeatureExtractor.py` (feature maps generated by `feature_extraction/TextureFeatureMapGenerate.py`) |
| 3. Radiomics feature extraction | `feature_extraction/RadiomicsFeatureExtractor.py`, `feature_extraction/radiomics_config.yaml` |
| 4. Per-modality classifier training | `modeling/SpatialClassificationModel.py`, `modeling/RadiomicsClassificationModel.py`, `modeling/SpatialRadiomicsClassifier.py` |
| 5. SRP fusion | `external_validation/PredictNSCLC.py` (fusion step), also embedded in the evaluation scripts in `evaluation/` |
| 6. External cohort application | `external_validation/PredictNSCLC.py` |
| 7. Binormal ROC/DCA smoothing | `evaluation/BinormalROCFit_Dataset002.py`, `evaluation/BinormalROCFit_Dataset004NACI.py` |
| 8. Bootstrap confidence intervals | `evaluation/BinormalAUC_BootstrapCI.py` |

Standard, previously published algorithms used elsewhere in the
pipeline (differential expression via DESeq2², pathway enrichment via
GSEA³, single-cell processing via Seurat⁴, cell–cell communication via
CellChat⁵) are applied via their reference implementations as cited,
rather than reproduced here; see `src/rnaseq_analysis/` and
`src/scrna_seq_analysis/` in the repository and the corresponding
Methods subsections of the manuscript for how each is parameterized
for this study.

---

¹ van Griethuysen, J. J. M. *et al.* Computational Radiomics System to
Decode the Radiographic Phenotype. *Cancer Res.* **77**, e104–e107 (2017).
² Love, M. I., Huber, W. & Anders, S. Moderated estimation of fold
change and dispersion for RNA-seq data with DESeq2. *Genome Biol.*
**15**, 550 (2014).
³ Subramanian, A. *et al.* Gene set enrichment analysis: a
knowledge-based approach for interpreting genome-wide expression
profiles. *PNAS* **102**, 15545–15550 (2005).
⁴ Hao, Y. *et al.* Integrated analysis of multimodal single-cell data.
*Cell* **184**, 3573–3587 (2021).
⁵ Jin, S. *et al.* Inference and analysis of cell-cell communication
using CellChat. *Nat. Commun.* **12**, 1088 (2021).

*[AUTHOR ACTION: verify citation details above and renumber to match
the manuscript's reference list before submission.]*
