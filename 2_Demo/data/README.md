# Demo data

Real, de-identified data from the manuscript's Dataset002_ECResponse
"Test" cohort — esophageal squamous cell carcinoma (ESCC) patients, one
bounded validation split out of the full study (the TJH and SPCH
validation cohorts, and all other project data, are excluded).

## `ecresponse_cohort_deidentified.csv` (n=110)

Used by the cohort-level evaluation steps (6-8) in `../app.py`.

- `demo_id` — anonymous sequential identifier (not derived from the real ID)
- `true_label` — binary outcome (1 = Non-pCR, 0 = pCR)
- `spatial_probability`, `radiomics_probability`, `fused_probability` —
  the **real, already-trained, locked SRP model's actual historical
  predictions** for this patient (from
  `model_results/test_predictions_results.csv` in the source data).
  **This demo does not retrain a model** — a simplified from-scratch
  retraining on a cohort this size cannot be expected to reproduce the
  properly-tuned production model's real performance (we verified this:
  a quick from-scratch reproduction gave AUCs of 0.32-0.56, clearly
  wrong/degenerate, whereas the real model's actual predictions give
  0.71-0.82, matching the manuscript). Using the real predictions
  directly is both simpler and actually correct — this demo
  demonstrates the downstream workflow (fusion, evaluation, risk
  stratification, survival analysis) applied to genuine model output,
  consistent with Algorithm 6 (applying a locked model) in the
  Supplementary Note, rather than reproducing Algorithm 4 (training)
  from scratch.

  **Why this is the "Test" split specifically, not the full available
  cohort:** the same source data also includes a 255-patient "Train"
  split the model was fit on. We checked combining it with Test for a
  larger demo cohort (n=365) and found the Train split's own
  predictions are massively overfit (e.g. Radiomics AUC = 0.999 on
  Train vs. 0.707 on Test) — including it would inflate this demo's
  apparent performance in a misleading way. The Test split is the only
  one of the two that represents genuine, held-out generalization
  performance, which is why it is used here on its own.
- `dfs_event`, `dfs_time_months`, `os_event`, `os_time_months` — real
  survival outcomes (months; no calendar dates)
- `radiomics__*`, `spatial3d__*`, `spatialtex__*` — real extracted
  feature values (537 columns)

**Risk-stratification threshold:** steps 7-8 in `../app.py` threshold
`fused_probability` at **0.454**, not an arbitrary 0.5 -- this is the
Fusion (SRP) model's precomputed, log-rank-minimizing optimal cutoff
from the real survival analysis
(`survival_analysis/cox_regression_results.csv`,
`survival_analysis/median_survival.csv` in the source data), and
reproduces the real group sizes (81 High-risk / 29 Low-risk),
HR = 3.05 [1.07-8.66], C-index = 0.596, log-rank P = 0.0280 exactly.

## `case_pCR/`, `case_nonpCR/` (2 example patients)

Used by the per-patient walkthrough steps (1-5) in `../app.py`.

- `slices.npz` — every axial slice through the tumor (cropped to the
  tumor region, soft-tissue windowed), as aligned arrays so the GUI can
  show any slice instantly with a slider above the image, with no
  3D/graphics library needed at runtime:
  - `ct` — plain grayscale CT, shape (n_slices, H, W)
  - `seg_overlay` — the same slices with the tumor contour drawn in,
    RGB, shape (n_slices, H, W, 3)
  - `feat_values` — the raw GLCM-Contrast spatial-heterogeneity value
    per voxel (NaN outside the tumor), shape (n_slices, H, W), float32
    — rendered live in the GUI over the grayscale CT with a
    perceptually-uniform "Nature style" colormap (`viridis`) and a
    colorbar shown alongside, rather than baked into a fixed-color RGB
    image, so the color scale stays visible
  - `feat_vmin`, `feat_vmax` — the fixed color-scale bounds (2nd/98th
    percentile of this case's tumor voxel values) shared by every slice
  - `tumor_area` — tumor pixel count per slice (used to default the
    slider to the largest cross-section)
- `mesh.npz` — a 3D surface mesh of the tumor ROI (`verts`, `faces`;
  marching cubes on the segmentation mask, scaled by the real voxel
  spacing), shown next to the slice image as an interactive, rotatable
  Plotly WebGL mesh (`plotly`, client-side rendering — no 3D graphics
  library needed at GUI runtime beyond that one package)
- `features.csv` — real extracted feature values for this patient
- `prediction.json` — the real locked model's actual prediction for
  this patient

## De-identification

Built by `build_deidentified_cohort.py` and `build_gui_case_assets.py`
(included for transparency/audit only — both require the private,
restricted-access `Dataset/` directory and **cannot be run by
reviewers**; they document the process, they are not part of the
runnable demo).

- No real case identifiers, names, dates, or clinical fields beyond
  outcome/survival are included anywhere.
- **Which two patients were picked for the per-patient walkthrough is
  not hardcoded as a literal case ID anywhere** — they are selected
  programmatically, by two different criteria:
  - **Non-pCR example:** among Test-cohort patients with a tumor large
    enough for a clear slice view, restricted to patients where the
    real locked **Fusion (SRP) model classifies correctly** at the
    standard 0.5 probability threshold **while the Radiomics-only
    model classifies incorrectly** at that same threshold -- a
    concrete, reproducible demonstration of the Spatial model's added
    value -- the patient the Fusion model classifies most confidently
    is picked.
  - **pCR example:** a *representative* case rather than an edge case —
    searched across both the Train and Test splits (a visual example
    does not carry the same train/test overfitting concern as the
    cohort-level statistics, see above), restricted to patients with a
    tumor size inside the interquartile range of all eligible pCR
    tumor sizes (a typical size, not a size outlier) and correctly
    classified by the real locked Fusion model, the one classified
    most confidently is picked.

  This package does not itself disclose which two specific real
  patients were used.
- `slices.npz` contains only rasterized pixel arrays (plain numeric
  arrays, the same kind of data as a PNG) cropped to the tumor region —
  no DICOM/NIfTI-style header metadata survives this, regardless of
  what the source file contained. (We additionally confirmed the
  source NIfTI files carry no identifying header fields before using
  them.)
- Real case IDs are used only internally during the one-off export
  runs to locate source files — never written to any shipped output.

This package includes real, de-identified **3D imaging data** (the
full stack of axial slices through each tumor, not just a single
slice or extracted features) — a deliberate, explicitly-confirmed
data-sharing decision for this demo. Verify it still matches your
data-use agreement / IRB protocol before submission.
