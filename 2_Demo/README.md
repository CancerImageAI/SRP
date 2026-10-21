# SRP Demo

A single, consolidated, button-driven web demo of the **complete SRP
(Spatial + Radiomics → Fusion) workflow** — from a patient's 3D CT scan
to a response prediction, plus cohort-level statistical validation —
all in one app, on real, de-identified data from the ESCC (esophageal
squamous cell carcinoma) cohort.

## The 8 steps

**Per-patient walkthrough** (2 real example patients — a **representative
pCR patient** (a typical tumor size, confidently and correctly
classified by the real locked model, picked by searching across both
the Train and Test splits since this is a visual example, not a
held-out performance claim) and a **Non-pCR patient chosen because the
real locked Fusion (SRP) model classifies it correctly while
Conventional Radiomics alone does not** — a concrete demonstration of
the Spatial model's added value — viewed slice-by-slice on the axial
plane, the standard radiology-viewer convention: a slider above the
image steps through every slice of the 3D CT volume, alongside a
rotatable 3D mesh of the tumor ROI):
1. **Input Image** — the CT volume, axial slice by axial slice, with
   the tumor's 3D mesh shown to the right
2. **Segmentation** — the tumor contour on every axial slice, with the
   same segmentation as a rotatable 3D mesh to the right
3. **Feature Extraction** — real extracted radiomics/spatial feature
   values, shown as tables
4. **Visualization** — a spatial-heterogeneity feature map (GLCM
   Contrast) overlaid on every axial slice, rendered live with a
   perceptually-uniform "Nature style" colormap (`viridis`) and its
   colorbar shown alongside, with the 3D ROI mesh to the right
5. **Response Prediction** — the real, locked SRP model's actual
   Spatial / Radiomics / Fused probabilities for this patient

**Cohort-level evaluation** (full de-identified cohort, n=110, using
the real locked model's actual historical predictions — not retrained
here, see `data/README.md`):
6. **Cohort Evaluation** — ROC (styled and numerically matching the
   manuscript's own evaluation figures, with a labeled chance line) and
   a multi-metric radar chart (Specificity, Sensitivity, Accuracy, F1,
   NPV, PPV at the standard 0.5 classification threshold) comparing all
   three models
7. **Risk Stratification** — High-/Low-risk assignment at the Fusion
   model's precomputed optimal cutoff (0.454, not an arbitrary 0.5 —
   see `data/README.md`)
8. **Survival Analysis** — Kaplan-Meier (disease-free survival) by risk
   group, with a log-rank test

## Why a web GUI

Built with [Streamlit](https://streamlit.io), a pure-Python web app
framework: no JavaScript, no separate frontend build, and it runs
identically on Windows, Mac, and Linux — `pip install` plus one command
opens it in your default browser. Every CT/segmentation/heterogeneity
image is a precomputed 2D array displayed with Pillow/matplotlib, and
the 3D tumor mesh is a precomputed set of vertices/faces rendered with
Plotly's client-side WebGL renderer — no 3D graphics stack (no VTK, no
OpenGL, no virtual display) is needed server-side, which keeps
installation simple and reliable across operating systems.

## System requirements

- Python >= 3.9, platform-independent (Windows/Mac/Linux)
- 8 packages: `streamlit`, `pandas`, `numpy`, `pillow`, `matplotlib`,
  `plotly`, `scikit-learn`, `lifelines` (`requirements_demo.txt`)
- No GPU, no R, no imaging libraries (SimpleITK/PyRadiomics) needed —
  this app only displays precomputed images/tables/meshes and computes
  lightweight statistics (ROC/KM curves) on an already-scored cohort;
  it does not load or process raw imaging data itself
- Tested on: Python 3.9.18, streamlit 1.50, Ubuntu 22.04 (x86_64)

## Installation guide

```bash
cd 2_Demo
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements_demo.txt
```

**Typical install time:** under 2 minutes on a normal desktop/laptop
with a broadband connection.

## Demo

### Instructions to run

```bash
streamlit run app.py
```

Opens in your default browser at `http://localhost:8501`. If it
doesn't open automatically, open that URL yourself. Press Ctrl+C in the
terminal to stop the server when done. All data is already committed
in `data/` — no separate data-generation step is needed.

### Expected output

A browser tab titled "SRP Demo" with two rows of step buttons (5
per-patient steps, 3 cohort-level steps) and a patient selector.
Clicking each button shows that step's real result:

- Steps 1-5 (for whichever of the 2 example patients is selected): an
  axial-slice CT viewer (with a slider above the image to step through
  the volume, and a rotatable 3D mesh of the tumor ROI to the right),
  the tumor contour on each slice, a table of real extracted feature
  values, a `viridis` heterogeneity-map overlay with a colorbar on each
  slice, and the locked model's real per-patient prediction.
- Steps 6-8 (always the full n=110 cohort): an ROC plot with
  AUC = 0.770 (Spatial) / 0.707 (Radiomics) / 0.823 (Fusion) and a
  6-axis radar chart (Specificity/Sensitivity/Accuracy/F1/NPV/PPV) —
  deterministic, since these are precomputed real model predictions,
  you will see exactly these numbers, matching the manuscript's own
  evaluation figures — a risk-group histogram (threshold 0.454, the Fusion model's
  precomputed optimal cutoff), and a Kaplan-Meier survival plot with
  group sizes 81 High-risk / 29 Low-risk and log-rank P = 0.0280,
  matching the manuscript's real survival analysis exactly.

### Expected run time

The server starts in a few seconds; every step updates in under a
second on click (everything shown is precomputed or a lightweight
statistic computed on an already-scored, 110-row table — no model
training or image processing happens live).

## Instructions for use

### How to run the software on your data

- **Cohort-level steps (6-8) on your own scored cohort:** replace
  `data/ecresponse_cohort_deidentified.csv` with your own CSV having
  the same column convention (an identifier column, `true_label`,
  `spatial_probability`/`radiomics_probability`/`fused_probability`
  from your own trained model, and optionally
  `dfs_event`/`dfs_time_months` for step 8).
- **Per-patient steps (1-5) on different example patients:** the app
  is scoped to its 2 bundled patients (it is a visual explainer, not a
  scoring tool) — regenerate `data/case_pCR/` and `data/case_nonpCR/`
  with `data/build_gui_case_assets.py` (requires the private,
  restricted-access cohort data; not runnable by reviewers) to point it
  at different example cases.
- **Full pipeline on new imaging cohorts (feature extraction from raw
  CT, model training, etc.):** see `../1_Source_Code/SRP/` and its
  `docs/USAGE.md`. `src/external_validation/PredictNSCLC.py` is the
  template for scoring a new cohort with an already-trained model.

### (OPTIONAL) Reproduction instructions

To reproduce the manuscript's actual reported quantitative results
(this demo's cohort-level numbers already match the manuscript's
Dataset002_ECResponse/Test statistics, but on a smaller feature/outcome
slice than the full analysis), the restricted-access clinical imaging
cohorts described in the manuscript's data-availability statement are
required. Given that data, the full pipeline in `../1_Source_Code/SRP/`
reproduces every figure and statistic in the manuscript; its
`docs/USAGE.md` gives the exact, ordered sequence of scripts to run per
pipeline stage.
