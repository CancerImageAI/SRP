# ============================================================
# Remove Suspected Doublet Clusters and Save for Re-analysis
# ============================================================
# 
# Removing:
#   - Clusters 34, 35, 36, 37, 38 at resolution 1.5
#   - Cluster 25 at resolution 0.8
#
# These clusters were identified as potential doublets based on
# UMAP visualization patterns.
# ============================================================

library(Seurat)
library(tidyverse)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Load Seurat object
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_annotated.rds"))

cat(sprintf("  Total cells before filtering: %d\n", ncol(seu)))
cat(sprintf("  Total genes: %d\n", nrow(seu)))

# ----------------------------------------------------------
# Check available clustering resolutions
# ----------------------------------------------------------
cat("\n  Available metadata columns with clustering info:\n")
cluster_cols <- grep("RNA_snn_res|seurat_clusters", colnames(seu@meta.data), value = TRUE)
cat("  ", paste(cluster_cols, collapse = ", "), "\n")

# ----------------------------------------------------------
# Identify cells to remove
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Identifying cells in doublet clusters\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Resolution 1.5: clusters 34, 35, 36, 37, 38
res15_col <- grep("1\\.5", cluster_cols, value = TRUE)
if (length(res15_col) == 0) {
  # Try alternative column names
  res15_col <- "RNA_snn_res.1.5"
  if (!res15_col %in% colnames(seu@meta.data)) {
    res15_col <- "integrated_snn_res.1.5"
  }
}
cat(sprintf("  Resolution 1.5 column: %s\n", res15_col[1]))

# Resolution 0.8: cluster 25
res08_col <- grep("0\\.8", cluster_cols, value = TRUE)
if (length(res08_col) == 0) {
  res08_col <- "RNA_snn_res.0.8"
  if (!res08_col %in% colnames(seu@meta.data)) {
    res08_col <- "integrated_snn_res.0.8"
  }
}
cat(sprintf("  Resolution 0.8 column: %s\n", res08_col[1]))

# Identify cells in clusters to remove at resolution 1.5
clusters_to_remove_res15 <- c("34", "35", "36", "37", "38")
cells_res15 <- colnames(seu)[seu@meta.data[[res15_col[1]]] %in% clusters_to_remove_res15]

cat(sprintf("\n  Cells in resolution 1.5 clusters %s: %d\n",
            paste(clusters_to_remove_res15, collapse = ", "), length(cells_res15)))

# Breakdown per cluster at res 1.5
for (cl in clusters_to_remove_res15) {
  n_cl <- sum(seu@meta.data[[res15_col[1]]] == cl, na.rm = TRUE)
  cat(sprintf("    Cluster %s: %d cells\n", cl, n_cl))
}

# Identify cells in cluster to remove at resolution 0.8
clusters_to_remove_res08 <- c("25")
cells_res08 <- colnames(seu)[seu@meta.data[[res08_col[1]]] %in% clusters_to_remove_res08]

cat(sprintf("\n  Cells in resolution 0.8 cluster %s: %d\n",
            paste(clusters_to_remove_res08, collapse = ", "), length(cells_res08)))

# Union of all cells to remove
cells_to_remove <- unique(c(cells_res15, cells_res08))
cat(sprintf("\n  Total unique cells to remove (union): %d\n", length(cells_to_remove)))

# Check overlap between the two sets
overlap <- intersect(cells_res15, cells_res08)
cat(sprintf("  Overlap between res1.5 and res0.8 selections: %d cells\n", length(overlap)))

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

# Subset the Seurat object
seu_filtered <- subset(seu, cells = cells_to_keep)

cat(sprintf("\n  Filtered object: %d cells × %d genes\n",
            ncol(seu_filtered), nrow(seu_filtered)))

# ----------------------------------------------------------
# Verify removal
# ----------------------------------------------------------
cat("\n  Verification:\n")

# Check res 1.5
if (res15_col[1] %in% colnames(seu_filtered@meta.data)) {
  remaining_res15 <- table(seu_filtered@meta.data[[res15_col[1]]])
  removed_check <- clusters_to_remove_res15[clusters_to_remove_res15 %in% names(remaining_res15)]
  if (length(removed_check) == 0) {
    cat("    ✓ All res1.5 clusters (34-38) successfully removed\n")
  } else {
    cat(sprintf("    ⚠ Remaining cells in res1.5 clusters: %s\n",
                paste(removed_check, collapse = ", ")))
  }
}

# Check res 0.8
if (res08_col[1] %in% colnames(seu_filtered@meta.data)) {
  remaining_res08 <- table(seu_filtered@meta.data[[res08_col[1]]])
  removed_check08 <- clusters_to_remove_res08[clusters_to_remove_res08 %in% names(remaining_res08)]
  if (length(removed_check08) == 0) {
    cat("    ✓ All res0.8 clusters (25) successfully removed\n")
  } else {
    cat(sprintf("    ⚠ Remaining cells in res0.8 clusters: %s\n",
                paste(removed_check08, collapse = ", ")))
  }
}

# ----------------------------------------------------------
# Clean metadata (remove old clustering that is now invalid)
# ----------------------------------------------------------
cat("\n  Cleaning metadata (old clustering will be invalid after re-integration)...\n")

# Keep essential metadata, remove old cluster assignments
# (they will be recomputed during re-integration)
cols_to_remove <- grep("RNA_snn_res|integrated_snn_res|seurat_clusters",
                       colnames(seu_filtered@meta.data), value = TRUE)

# Also remove old UMAP/PCA reductions (will be recomputed)
cat(sprintf("    Removing %d old clustering columns\n", length(cols_to_remove)))

# Option A: Keep old clusters for reference (commented out removal)
# seu_filtered@meta.data <- seu_filtered@meta.data[, !colnames(seu_filtered@meta.data) %in% cols_to_remove]

# Option B: Remove old reductions (recommended for clean re-run)
# Uncomment below if you want a completely fresh start:
# seu_filtered@reductions <- list()
# seu_filtered@graphs <- list()

# For now, keep everything — the re-integration script will overwrite

# ----------------------------------------------------------
# Print summary of remaining cells per sample/group
# ----------------------------------------------------------
cat("\n  Remaining cells per group:\n")
if ("group" %in% colnames(seu_filtered@meta.data)) {
  print(table(seu_filtered$group))
}

cat("\n  Remaining cells per sample (top 10):\n")
if ("sample" %in% colnames(seu_filtered@meta.data)) {
  sample_counts <- sort(table(seu_filtered$sample), decreasing = TRUE)
  print(head(sample_counts, 10))
} else if ("orig.ident" %in% colnames(seu_filtered@meta.data)) {
  sample_counts <- sort(table(seu_filtered$orig.ident), decreasing = TRUE)
  print(head(sample_counts, 10))
}

# If cell_type annotation exists, show remaining
if ("cell_type" %in% colnames(seu_filtered@meta.data)) {
  cat("\n  Remaining cells per cell type:\n")
  print(sort(table(seu_filtered$cell_type), decreasing = TRUE))
}

# ----------------------------------------------------------
# Save filtered object
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Saving filtered Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

output_file <- file.path(output_dir, "seu_filtered_no_doublets.rds")
saveRDS(seu_filtered, output_file)

cat(sprintf("  Saved to: %s\n", output_file))
cat(sprintf("  File size: %.1f MB\n", file.size(output_file) / 1024^2))

# Also save a record of what was removed
removal_log <- data.frame(
  cell_barcode = cells_to_remove,
  reason = ifelse(cells_to_remove %in% cells_res15 & cells_to_remove %in% cells_res08,
                  "res1.5_cluster34-38 AND res0.8_cluster25",
                  ifelse(cells_to_remove %in% cells_res15,
                         "res1.5_cluster34-38",
                         "res0.8_cluster25")),
  stringsAsFactors = FALSE
)

# Add cluster info
removal_log$cluster_res15 <- seu@meta.data[removal_log$cell_barcode, res15_col[1]]
removal_log$cluster_res08 <- seu@meta.data[removal_log$cell_barcode, res08_col[1]]

write.csv(removal_log, file.path(output_dir, "removed_doublet_cells_log.csv"),
          row.names = FALSE)

cat(sprintf("  Removal log saved: %d cells documented\n", nrow(removal_log)))

# ----------------------------------------------------------
# Final report
# ----------------------------------------------------------
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════╗\n")
cat("║         DOUBLET CLUSTER REMOVAL — COMPLETE                      ║\n")
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                  ║\n")
cat(sprintf("║  Original cells:    %6d                                      ║\n", ncol(seu)))
cat(sprintf("║  Cells removed:     %6d (%.1f%%)                              ║\n",
            length(cells_to_remove), length(cells_to_remove)/ncol(seu)*100))
cat(sprintf("║  Remaining cells:   %6d                                      ║\n", ncol(seu_filtered)))
cat("║                                                                  ║\n")
cat("║  Removed clusters:                                               ║\n")
cat("║    Resolution 1.5: clusters 34, 35, 36, 37, 38                  ║\n")
cat("║    Resolution 0.8: cluster 25                                    ║\n")
cat("║                                                                  ║\n")
cat("║  Output file:                                                    ║\n")
cat(sprintf("║    %s\n", output_file))
cat("║                                                                  ║\n")
cat("║  Next steps:                                                     ║\n")
cat("║    1. Load seu_filtered_no_doublets.rds                          ║\n")
cat("║    2. Re-run integration (SCTransform/Harmony/etc.)              ║\n")
cat("║    3. Re-run clustering and UMAP                                 ║\n")
cat("║    4. Re-run cell type annotation                                ║\n")
cat("║                                                                  ║\n")
cat("╚══════════════════════════════════════════════════════════════════╝\n")
