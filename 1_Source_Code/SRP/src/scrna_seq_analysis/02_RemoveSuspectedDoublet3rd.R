# ============================================================
# Remove Suspected Doublet Clusters (Round 2) and Save
# ============================================================
# 
# After initial doublet removal and re-integration, additional
# suspicious clusters were identified on UMAP:
#   - Clusters 18, 19 at resolution 0.5
#
# These will be removed and the object saved for another round
# of integration.
# ============================================================

library(Seurat)
library(tidyverse)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/02_Integration"
output_dir <- "XXXyour path/scRNA_Results_new/02_Integration"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Load Seurat object (re-integrated after first filtering)
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading re-integrated Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_integrated_new_2rd.rds"))

cat(sprintf("  Total cells before filtering: %d\n", ncol(seu)))
cat(sprintf("  Total genes: %d\n", nrow(seu)))

# ----------------------------------------------------------
# Check available clustering resolutions
# ----------------------------------------------------------
cat("\n  Available metadata columns:\n")
cluster_cols <- grep("snn_res|seurat_clusters", colnames(seu@meta.data), value = TRUE)
cat("  ", paste(cluster_cols, collapse = "\n  "), "\n")

# ----------------------------------------------------------
# Identify the resolution 0.5 column
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Identifying cells in doublet clusters\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Find the resolution 0.5 column
res05_col <- grep("0\\.5", cluster_cols, value = TRUE)
if (length(res05_col) == 0) {
  # Try common naming patterns
  possible_names <- c("RNA_snn_res.0.5", "integrated_snn_res.0.5",
                      "SCT_snn_res.0.5", "wsnn_res.0.5")
  res05_col <- possible_names[possible_names %in% colnames(seu@meta.data)]
}

if (length(res05_col) == 0) {
  stop("Cannot find resolution 0.5 clustering column. Available columns:\n",
       paste(cluster_cols, collapse = "\n"))
}

res05_col <- res05_col[1]
cat(sprintf("  Using resolution 0.5 column: %s\n", res05_col))

# Show cluster distribution at res 0.5
cat("\n  Cluster distribution at resolution 0.5:\n")
cluster_table <- sort(table(seu@meta.data[[res05_col]]), decreasing = TRUE)
print(cluster_table)

# ----------------------------------------------------------
# Identify cells to remove: clusters 18 at resolution 0.5
# ----------------------------------------------------------
clusters_to_remove <- c("18")

cat(sprintf("\n  Clusters to remove: %s\n", paste(clusters_to_remove, collapse = ", ")))

# Get cells in these clusters
cells_to_remove <- colnames(seu)[seu@meta.data[[res05_col]] %in% clusters_to_remove]

cat(sprintf("  Total cells to remove: %d\n", length(cells_to_remove)))

# Breakdown per cluster
for (cl in clusters_to_remove) {
  n_cl <- sum(seu@meta.data[[res05_col]] == cl, na.rm = TRUE)
  cat(sprintf("    Cluster %s: %d cells\n", cl, n_cl))
}

# ----------------------------------------------------------
# Check characteristics of cells to remove (QC sanity check)
# ----------------------------------------------------------
cat("\n  QC characteristics of cells to remove:\n")

meta_remove <- seu@meta.data[cells_to_remove, ]

if ("nFeature_RNA" %in% colnames(meta_remove)) {
  cat(sprintf("    nFeature_RNA: median = %.0f (all cells median = %.0f)\n",
              median(meta_remove$nFeature_RNA),
              median(seu$nFeature_RNA)))
}
if ("nCount_RNA" %in% colnames(meta_remove)) {
  cat(sprintf("    nCount_RNA:   median = %.0f (all cells median = %.0f)\n",
              median(meta_remove$nCount_RNA),
              median(seu$nCount_RNA)))
}
if ("percent.mt" %in% colnames(meta_remove)) {
  cat(sprintf("    percent.mt:   median = %.2f%% (all cells median = %.2f%%)\n",
              median(meta_remove$percent.mt),
              median(seu$percent.mt)))
}

# Sample distribution of removed cells
if ("orig.ident" %in% colnames(meta_remove)) {
  cat("\n    Sample distribution of removed cells:\n")
  sample_dist <- sort(table(meta_remove$orig.ident), decreasing = TRUE)
  print(head(sample_dist, 10))
}

# ----------------------------------------------------------
# Remove cells
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Removing doublet clusters\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cells_to_keep <- setdiff(colnames(seu), cells_to_remove)

cat(sprintf("  Cells to keep: %d\n", length(cells_to_keep)))
cat(sprintf("  Cells removed: %d (%.1f%% of total)\n",
            length(cells_to_remove),
            length(cells_to_remove) / ncol(seu) * 100))

# Subset
seu_filtered <- subset(seu, cells = cells_to_keep)

cat(sprintf("\n  Filtered object: %d cells × %d genes\n",
            ncol(seu_filtered), nrow(seu_filtered)))

# ----------------------------------------------------------
# Verify removal
# ----------------------------------------------------------
cat("\n  Verification:\n")
remaining_clusters <- unique(seu_filtered@meta.data[[res05_col]])
removed_still_present <- intersect(clusters_to_remove, as.character(remaining_clusters))

if (length(removed_still_present) == 0) {
  cat("    ✓ Clusters 18 at resolution 0.5 successfully removed\n")
} else {
  cat(sprintf("    ⚠ WARNING: Clusters still present: %s\n",
              paste(removed_still_present, collapse = ", ")))
}

# ----------------------------------------------------------
# Clean: remove old dimensionality reductions and graphs
# (they will be recomputed during re-integration)
# ----------------------------------------------------------
cat("\n  Cleaning old reductions and graphs for fresh re-integration...\n")

# Remove reductions (PCA, UMAP, etc.)
seu_filtered@reductions <- list()
cat("    ✓ Removed old reductions (PCA, UMAP, etc.)\n")

# Remove graphs (SNN, etc.)
seu_filtered@graphs <- list()
cat("    ✓ Removed old SNN graphs\n")

# Remove old clustering columns (will be recomputed)
old_cluster_cols <- grep("snn_res|seurat_clusters", colnames(seu_filtered@meta.data), value = TRUE)
if (length(old_cluster_cols) > 0) {
  seu_filtered@meta.data <- seu_filtered@meta.data[, !colnames(seu_filtered@meta.data) %in% old_cluster_cols, drop = FALSE]
  cat(sprintf("    ✓ Removed %d old clustering columns: %s\n",
              length(old_cluster_cols), paste(old_cluster_cols, collapse = ", ")))
}

# ----------------------------------------------------------
# Print summary
# ----------------------------------------------------------
cat("\n  Remaining cells per sample/group:\n")
if ("group" %in% colnames(seu_filtered@meta.data)) {
  cat("    Per group:\n")
  print(table(seu_filtered$group))
}

if ("orig.ident" %in% colnames(seu_filtered@meta.data)) {
  cat("\n    Per sample (top 10):\n")
  sample_counts <- sort(table(seu_filtered$orig.ident), decreasing = TRUE)
  print(head(sample_counts, 10))
}

# ----------------------------------------------------------
# Save filtered object
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Saving filtered Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

output_file <- file.path(output_dir, "seu_integrated_new_filtered_round3.rds")
saveRDS(seu_filtered, output_file)

cat(sprintf("  Saved to: %s\n", output_file))
cat(sprintf("  File size: %.1f MB\n", file.size(output_file) / 1024^2))

# ----------------------------------------------------------
# Save removal log
# ----------------------------------------------------------
removal_log <- data.frame(
  cell_barcode = cells_to_remove,
  removed_from = "seu_integrated_new.rds",
  reason = "suspected_doublets",
  resolution = "0.5",
  cluster = seu@meta.data[cells_to_remove, res05_col],
  stringsAsFactors = FALSE
)

# Add QC metrics if available
if ("nFeature_RNA" %in% colnames(seu@meta.data)) {
  removal_log$nFeature_RNA <- seu@meta.data[cells_to_remove, "nFeature_RNA"]
}
if ("nCount_RNA" %in% colnames(seu@meta.data)) {
  removal_log$nCount_RNA <- seu@meta.data[cells_to_remove, "nCount_RNA"]
}
if ("percent.mt" %in% colnames(seu@meta.data)) {
  removal_log$percent.mt <- seu@meta.data[cells_to_remove, "percent.mt"]
}

log_file <- file.path(output_dir, "removed_doublet_cells_round3_log.csv")
write.csv(removal_log, log_file, row.names = FALSE)

cat(sprintf("  Removal log saved: %s (%d cells)\n", log_file, nrow(removal_log)))

# ----------------------------------------------------------
# Final report
# ----------------------------------------------------------
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════╗\n")
cat("║     DOUBLET CLUSTER REMOVAL (ROUND 3) — COMPLETE                ║\n")
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                  ║\n")
cat(sprintf("║  Input:  seu_integrated_new.rds (%d cells)                 ║\n", ncol(seu)))
cat(sprintf("║  Output: seu_integrated_new_filtered_round3.rds (%d cells) ║\n", ncol(seu_filtered)))
cat(sprintf("║  Removed: %d cells (%.1f%%)                                 ║\n",
            length(cells_to_remove), length(cells_to_remove)/ncol(seu)*100))
cat("║                                                                  ║\n")
cat("║  Removed clusters:                                               ║\n")
cat("║    Resolution 0.5: clusters 18                               ║\n")
cat("║                                                                  ║\n")
cat("║  Output files:                                                   ║\n")
cat("║    seu_integrated_new_filtered_round3.rds                        ║\n")
cat("║    removed_doublet_cells_round3_log.csv                          ║\n")
cat("║                                                                  ║\n")
cat("║  Next steps:                                                     ║\n")
cat("║    1. Load seu_integrated_new_filtered_round3.rds                ║\n")
cat("║    2. Re-run normalization (SCTransform or NormalizeData)         ║\n")
cat("║    3. Re-run integration (Harmony/CCA/RPCA)                      ║\n")
cat("║    4. Re-run PCA → UMAP → FindNeighbors → FindClusters          ║\n")
cat("║    5. Re-run cell type annotation                                ║\n")
cat("║                                                                  ║\n")
cat("╚══════════════════════════════════════════════════════════════════╝\n")

cat("\n━━━ Done ━━━\n")
