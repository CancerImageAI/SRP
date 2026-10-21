# README (Nature Research Code & Software Submission Checklist)

This README follows the four required sections of the *Nature
Research Code and Software Submission Checklist* (June 2017), in
order. It covers both (a) the full manuscript pipeline
(`../1_Source_Code/SRP/`) and (b) the single consolidated demo
(`../2_Demo/`) built specifically so this checklist's requirements can
be satisfied without the restricted-access clinical imaging data.

---

## 1. System requirements

### All software dependencies and operating systems (including version numbers)

**Full pipeline** (`../1_Source_Code/SRP/`):
- Python >= 3.9 for everything except the single-cell RNA-seq module
- R >= 4.2 (Seurat-based) for `src/scrna_seq_analysis/`
- Platform-independent (no OS-specific code); developed and run on
  Ubuntu 22.04 (x86_64)
- Full Python dependency list: `../1_Source_Code/SRP/requirements.txt`
- Full R/Bioconductor dependency list: `../1_Source_Code/SRP/requirements_R.md`
- `src/graph_neural_network/` additionally needs `torch` and
  `torch-geometric` (only if that exploratory branch is used)

**Demo** (`../2_Demo/`, see section 3 below):
- Python >= 3.9, platform-independent
- 8 packages: `streamlit`, `pandas`, `numpy`, `pillow`, `matplotlib`,
  `plotly`, `scikit-learn`, `lifelines` (`../2_Demo/requirements_demo.txt`)
- No R, no GPU, no imaging libraries (SimpleITK/PyRadiomics) needed —
  every CT/segmentation/heterogeneity image is a precomputed 2D array
  displayed with Pillow/matplotlib, and the 3D tumor mesh is a
  precomputed set of vertices/faces rendered client-side by Plotly
  (WebGL in the browser) — no 3D graphics stack is required server-side

### Versions the software has been tested on

- Python 3.9.18
- streamlit 1.50, pandas 2.2, numpy 1.26, pillow, matplotlib 3.9,
  plotly, scikit-learn 1.5, lifelines 0.29
- Ubuntu 22.04 LTS (x86_64)
- (Full pipeline only) R 4.2+; see `requirements_R.md` for the
  Bioconductor/CRAN package set and versions

### Any required non-standard hardware

None. No GPU is required for the demo or for the core
Spatial/Radiomics/Fusion classification pipeline. The optional
`src/graph_neural_network/` branch benefits from (but does not
strictly require) a CUDA-capable GPU for faster training.

---

## 2. Installation guide

### Instructions

**Demo (recommended starting point for reviewers):**
```bash
cd 2_Demo
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_demo.txt
```

**Full pipeline:**
```bash
cd ../1_Source_Code/SRP
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# for src/scrna_seq_analysis/, separately see requirements_R.md
```

### Typical install time on a "normal" desktop computer

- **Demo:** under 2 minutes on a standard broadband connection.
- **Full pipeline:** typically 5-15 minutes, dominated by larger
  packages (SimpleITK, PyRadiomics, and, if used, torch/torch-geometric);
  the R dependencies for the single-cell module (Seurat, Bioconductor
  packages) can take significantly longer (potentially 30+ minutes) on
  a first install, which is typical for this software ecosystem and
  not specific to this codebase.

---

## 3. Demo

A single, consolidated, button-driven web app (`../2_Demo/`) showing
the complete SRP workflow, on real, de-identified data from the ESCC
(esophageal squamous cell carcinoma) cohort — not simulated:

- **Steps 1-5 (per-patient walkthrough):** input image (the 3D CT
  volume, viewed slice-by-slice on the axial plane with a slider above
  the image, next to a rotatable 3D mesh of the tumor ROI) →
  segmentation (tumor contour on every axial slice, with the same 3D
  mesh) → feature extraction (real extracted values, shown as tables) →
  visualization (a spatial-heterogeneity map, GLCM Contrast, overlaid
  on every axial slice with a perceptually-uniform "Nature style"
  `viridis` colormap and its colorbar shown alongside, next to the 3D
  mesh) → response prediction (the real locked model's actual
  probabilities) — demonstrated on 2 real de-identified example
  patients: a representative pCR patient (typical tumor size,
  confidently and correctly classified) and a Non-pCR patient chosen
  because the real locked Fusion (SRP) model classifies it correctly
  while Conventional Radiomics alone does not.
- **Steps 6-8 (cohort-level evaluation):** ROC and a multi-metric radar
  chart (Specificity/Sensitivity/Accuracy/F1/NPV/PPV), risk
  stratification, and Kaplan-Meier survival analysis with a log-rank
  test — computed from the real, already-trained,
  locked SRP model's actual historical predictions across the full
  de-identified cohort (n=110), styled and numerically matching the
  manuscript's own evaluation figures, including the Fusion model's
  precomputed optimal risk-stratification threshold (0.454, not an
  arbitrary 0.5). This demo does not retrain a model (see
  `../2_Demo/data/README.md` for why a from-scratch reproduction at
  this scale would be both slower and, we verified, numerically wrong)
  — it demonstrates the complete downstream workflow applied to
  genuine model output.

### Instructions to run on data

```bash
cd 2_Demo
streamlit run app.py
```

Opens in your default browser at `http://localhost:8501`. All data is
already committed in `2_Demo/data/` — no separate data-generation step
is needed.

### Expected output

A browser tab with two rows of step buttons and a patient selector.
Each step shows a real result for the selected patient or the full
cohort. The cohort-level statistics (steps 6-8) are deterministic —
computed from precomputed real predictions, so you will see exactly:
```
Spatial AUC    = 0.770   (SEN=0.67 SPE=0.72 ACC=0.68 F1=0.77 NPV=0.39 PPV=0.89)
Radiomics AUC  = 0.707   (SEN=0.66 SPE=0.56 ACC=0.64 F1=0.74 NPV=0.33 PPV=0.84)
Fusion AUC     = 0.823   (SEN=0.75 SPE=0.76 ACC=0.75 F1=0.83 NPV=0.47 PPV=0.91)
Risk groups (threshold 0.454): 81 High-risk / 29 Low-risk
Log-rank P (High- vs. Low-risk DFS) = 0.0280
```
matching the manuscript's own ROC/radar/survival-analysis figures and
statistics. Full detail, including all 8 steps' expected content, in
`../2_Demo/README.md`.

### Expected run time for demo on a "normal" desktop computer

The server starts in a few seconds; every step updates in under a
second on click.

---

## 4. Instructions for use

### How to run the software on your data

- **Cohort-level steps (6-8) on your own scored cohort:** replace
  `2_Demo/data/ecresponse_cohort_deidentified.csv` with your own CSV
  (an identifier column, `true_label`, your own model's
  `spatial_probability`/`radiomics_probability`/`fused_probability`,
  optionally `dfs_event`/`dfs_time_months`).
- **Per-patient steps (1-5) on different example patients:** regenerate
  `2_Demo/data/case_pCR/` and `2_Demo/data/case_nonpCR/` with
  `2_Demo/data/build_gui_case_assets.py` (requires the private,
  restricted-access cohort data; not runnable by reviewers) — see
  `2_Demo/README.md`, "Instructions for use".
- **Full pipeline on new imaging cohorts:** see
  `1_Source_Code/SRP/docs/USAGE.md`, which documents, per pipeline
  stage, which script to run and what each hardcoded `XXXyour path`
  placeholder needs to be replaced with. `src/external_validation/PredictNSCLC.py`
  is the template for scoring a new cohort with an already-trained model.

### (OPTIONAL) Reproduction instructions

We encourage reproducing all quantitative results in the manuscript.
To do so:

1. Obtain the clinical imaging/RNA-seq cohorts described in the
   manuscript's data-availability statement (these are not included
   in this submission package, per the data-sharing restrictions
   stated there).
2. Follow `1_Source_Code/SRP/docs/USAGE.md` in order: feature
   extraction -> model training -> evaluation (ROC/DCA/calibration,
   including the binormal-smoothing step used for cohorts with heavily
   tied predicted probabilities) -> survival analysis -> external
   cohort validation -> RNA-seq and single-cell RNA-seq biological
   validation.
3. Figure/statistic-to-script correspondence for the manuscript's main
   results: *[AUTHOR ACTION NEEDED — list which script(s) in
   `1_Source_Code/SRP/src/` produce each main-text and extended-data
   figure/table, e.g. "Fig. 2a-c: src/evaluation/PlotFigures.py;
   Table 1: src/survival_analysis/SurvivalAnalysis.py", etc. This
   mapping is specific to the final manuscript layout and must be
   filled in by the authors.]*

### Complete, detailed description of the code's functionality (pseudocode)

Drafted at `Supplementary_Note_SRP_Algorithm.md` in this folder — 8
algorithm blocks (feature extraction, per-modality model training,
fusion, external cohort validation, binormal ROC/DCA smoothing,
bootstrap confidence intervals), each mapped to its exact script in
the repository. **This must be submitted as part of the manuscript**
(a Supplementary Note is the natural fit here, matching the
"Elsewhere (specify)" option already selected on the checklist) —
simply publishing it in the GitHub repository does not by itself
satisfy this requirement, since the checklist asks for something
attached to the permanent manuscript record. A condensed,
developer-facing mirror of the same content is also published at
`../1_Source_Code/SRP/docs/ALGORITHM.md` as a complementary reference
for anyone reading the code directly.

---

## See also

- `../1_Source_Code/README.md` — source code pointer and GitHub link
- `../2_Demo/README.md` — the demo, full detail
- `../2_Demo/data/README.md` — demo data description and de-identification
- `../4_License_and_Repository/README.md` — license and repository link
- `../CHECKLIST_MAPPING.md` — maps every item on the Nature Research
  Code and Software Submission Checklist to where it is satisfied in
  this package, including open items that need author input
