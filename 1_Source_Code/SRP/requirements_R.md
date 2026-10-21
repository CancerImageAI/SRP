# R dependencies (src/scrna_seq_analysis/)

The single-cell RNA-seq analysis pipeline (`src/scrna_seq_analysis/`) is
R, not Python, and is not covered by `requirements.txt`. Tested against
R >= 4.2. Install via CRAN/Bioconductor:

```r
install.packages(c(
  "Seurat", "harmony", "tidyverse", "dplyr", "tidyr", "ggplot2",
  "ggpubr", "ggrepel", "ggnewscale", "patchwork", "RColorBrewer",
  "circlize", "igraph", "gridBase", "grid", "rstatix", "clustree"
))

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c(
  "SingleCellExperiment", "scDblFinder", "infercnv", "DESeq2",
  "ComplexHeatmap", "clusterProfiler", "DOSE", "enrichplot",
  "msigdbr", "org.Hs.eg.db", "EnsDb.Hsapiens.v86"
))

# CellChat is not on CRAN/Bioconductor -- install from GitHub
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
devtools::install_github("sqjin/CellChat")
```

`infercnv` (used by `05_CNV_Analysis.R` / `05_CNV_UMAP.R`) additionally
requires a working JAGS installation (`sudo apt install jags` on
Debian/Ubuntu, or see https://mcmc-jags.sourceforge.io/) if you use its
Bayesian HMM mode.
