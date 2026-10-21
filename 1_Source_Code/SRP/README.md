# SRP: Spatial Radiomics Profiling
Code for a spatial radiomics profiling pipeline.


## Pipeline overview

```
1. Feature extraction        src/feature_extraction/
   CT images + tumor masks -> radiomics features, 3D spatial features,
   spatial-texture features (Moran's I / heterogeneity on feature maps)

2. Model training             src/modeling/
   Spatial / Radiomics / Fusion classifiers (XGBoost-based), feature
   selection, hyperparameter search
   (src/graph_neural_network/ contains an alternative, exploratory
    branch: FMCIB voxel embeddings + graph neural network classifier)

3. Evaluation                 src/evaluation/
   ROC / DCA / calibration / radar-chart figures, binormal smoothing
   for cohorts with heavily-tied predicted probabilities, bootstrap
   AUC confidence intervals

4. Survival analysis          src/survival_analysis/
   Cox regression + Kaplan-Meier for DFS/OS by model-derived risk group

5. External validation        src/external_validation/
   Apply the locked model to a new imaging cohort

6. Biological validation (RNA-seq)
   6a. Bulk RNA-seq            src/rnaseq_analysis/
       Differential expression (DESeq2) + pathway enrichment (GSEA/ORA)
       between model-predicted risk groups, using a public paired
       CT+RNA-seq cohort (NSCLC), to check that the imaging signal
       corresponds to a coherent tumor biology
   6b. Single-cell RNA-seq     src/scrna_seq_analysis/
       Full Seurat pipeline (QC -> integration -> doublet removal ->
       cell-type annotation -> CD8 subtyping -> CNV inference ->
       CellChat cell-cell communication -> pathway enrichment) on a
       paired CT+scRNA-seq esophageal-cancer cohort, comparing tumor
       microenvironment composition/signaling between model-derived
       risk groups
```

See [docs/USAGE.md](docs/USAGE.md) for the full step-by-step walkthrough,
expected input/output file formats, and the directory layout your data
needs to follow. See [docs/ALGORITHM.md](docs/ALGORITHM.md) for a
detailed, implementation-level description of what each core algorithm
computes (a condensed version of this is submitted as the manuscript's
Supplementary Note on code functionality).

## Repository layout

```
SRP/
├── src/
│   ├── feature_extraction/     radiomics + spatial/texture feature extractors
│   ├── graph_neural_network/   alternative FMCIB-embedding + GNN branch
│   ├── modeling/                classifier training / feature fusion
│   ├── evaluation/              figures, ROC/DCA smoothing, bootstrap CIs
│   ├── survival_analysis/       Cox regression, Kaplan-Meier
│   ├── external_validation/     apply a locked model to a new cohort
│   ├── rnaseq_analysis/         bulk RNA-seq DEG + pathway enrichment (Python)
│   └── scrna_seq_analysis/      single-cell RNA-seq pipeline (R / Seurat)
├── docs/
│   ├── USAGE.md
│   └── ALGORITHM.md
├── requirements.txt
├── requirements_R.md
├── LICENSE
└── README.md
```

Data (`Dataset/`) and generated results (`manuscript/`) are not part of
this repository -- see [docs/USAGE.md](docs/USAGE.md) for the expected
data layout.

## Quick start

```bash
git clone <this-repo-url>
cd SRP
pip install -r requirements.txt   # Python pipeline (everything except scrna_seq_analysis/)
# see requirements_R.md for the R/Bioconductor packages needed by src/scrna_seq_analysis/
```

Every script hardcodes its input/output paths as plain variables near
the top or bottom of the file rather than taking CLI arguments — look
for the placeholder **`XXXyour path`** and replace it with your actual
data location before running. See [docs/USAGE.md](docs/USAGE.md) for
the full walkthrough and the directory layout these paths expect.

## Requirements

Python 3.9+ (developed and tested against 3.9) for everything except
`src/scrna_seq_analysis/`, which is R >= 4.2 (Seurat-based) — see
[requirements.txt](requirements.txt) and
[requirements_R.md](requirements_R.md) respectively. The
graph-neural-network branch (`src/graph_neural_network/`) additionally
needs `torch` and `torch-geometric`, which are heavier/optional
Python dependencies -- only required if you use that branch.

## Citation

If you use this code, please cite the associated manuscript
(citation to be added on publication).

## License

Creative Commons CC BY-NC-ND license -- see [LICENSE](LICENSE).
