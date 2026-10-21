# ============================================================
# Cell Type Annotation Pipeline
# Input: seu_integrated_new_3rd.rds (after 3rd round integration)
# ============================================================

library(Seurat)
library(tidyverse)
library(ggplot2)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/02_Integration"
output_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Load integrated Seurat object
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading integrated Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_integrated_new_3rd.rds"))

cat(sprintf("  Cells: %d\n", ncol(seu)))
cat(sprintf("  Genes: %d\n", nrow(seu)))

# ----------------------------------------------------------
# Check available clustering resolutions
# ----------------------------------------------------------
cat("\n  Available clustering columns:\n")
cluster_cols <- grep("snn_res|seurat_clusters", colnames(seu@meta.data), value = TRUE)
cat("  ", paste(cluster_cols, collapse = "\n  "), "\n")

# ----------------------------------------------------------
# Check available reductions
# ----------------------------------------------------------
cat("\n  Available reductions:\n")
cat("  ", paste(names(seu@reductions), collapse = ", "), "\n")

# ----------------------------------------------------------
# Set default assay
# ----------------------------------------------------------
# Determine which assay to use for marker finding
available_assays <- Assays(seu)
cat("\n  Available assays:", paste(available_assays, collapse = ", "), "\n")

if ("RNA" %in% available_assays) {
  DefaultAssay(seu) <- "RNA"
  # Ensure RNA assay is normalized
  if (length(seu@assays$RNA@data@x) == 0 || all(seu@assays$RNA@data@x == 0)) {
    cat("  Normalizing RNA assay...\n")
    seu <- NormalizeData(seu, assay = "RNA")
  }
} else if ("SCT" %in% available_assays) {
  DefaultAssay(seu) <- "SCT"
}

cat(sprintf("  Default assay set to: %s\n", DefaultAssay(seu)))

# ----------------------------------------------------------
# Choose resolution for annotation
# ----------------------------------------------------------
# Let's examine multiple resolutions
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Cluster counts at different resolutions:\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

for (col in cluster_cols) {
  n_clusters <- length(unique(seu@meta.data[[col]]))
  cat(sprintf("    %s: %d clusters\n", col, n_clusters))
}

# Use resolution 0.5 for annotation (adjust if needed)
res_col <- grep("0\\.5", cluster_cols, value = TRUE)[1]
if (is.na(res_col) || is.null(res_col)) {
  res_col <- cluster_cols[1]
}
cat(sprintf("\n  Using for annotation: %s\n", res_col))

# Set as active identity
Idents(seu) <- res_col
cat(sprintf("  Number of clusters: %d\n", length(levels(Idents(seu)))))
cat("  Cluster sizes:\n")
print(sort(table(Idents(seu)), decreasing = TRUE))

# ----------------------------------------------------------
# UMAP visualization of clusters
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Generating UMAP plots\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

p_umap <- DimPlot(seu, reduction = "umap", label = TRUE, label.size = 4,
                  repel = TRUE, pt.size = 0.1) +
  ggtitle(sprintf("Clusters at %s", res_col)) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(file.path(output_dir, "UMAP_clusters_for_annotation.pdf"), p_umap,
       width = 10, height = 8)

# By sample/group if available
if ("group" %in% colnames(seu@meta.data)) {
  p_group <- DimPlot(seu, reduction = "umap", group.by = "group",
                     pt.size = 0.1) +
    ggtitle("UMAP by Risk Group")
  ggsave(file.path(output_dir, "UMAP_by_group.pdf"), p_group,
         width = 10, height = 8)
}

cat("  UMAP plots saved.\n")

# ----------------------------------------------------------
# Find ALL markers for each cluster
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Finding markers for all clusters (this may take a while)...\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Find markers using Wilcoxon rank-sum test
all_markers <- FindAllMarkers(
  seu,
  only.pos = TRUE,
  min.pct = 0.25,
  logfc.threshold = 0.5,
  test.use = "wilcox",
  verbose = TRUE
)

cat(sprintf("\n  Total markers found: %d\n", nrow(all_markers)))

# Save full marker table
write.csv(all_markers, file.path(output_dir, "all_cluster_markers.csv"), row.names = FALSE)

# ----------------------------------------------------------
# Get top markers per cluster
# ----------------------------------------------------------
top_markers <- all_markers %>%
  group_by(cluster) %>%
  slice_max(order_by = avg_log2FC, n = 20) %>%
  ungroup()

write.csv(top_markers, file.path(output_dir, "top20_markers_per_cluster.csv"), row.names = FALSE)

# ----------------------------------------------------------
# Print top 10 markers per cluster (for annotation decisions)
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  TOP 10 MARKERS PER CLUSTER\n")
cat("  (Use these to decide cell type annotations)\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

clusters <- sort(as.numeric(as.character(unique(all_markers$cluster))))

for (cl in clusters) {
  cl_markers <- all_markers %>%
    filter(cluster == cl) %>%
    arrange(desc(avg_log2FC)) %>%
    head(10)
  
  cat(sprintf("╔═══ CLUSTER %s (%d cells) ═══════════════════════════════════════════╗\n",
              cl, sum(Idents(seu) == cl)))
  cat(sprintf("║  %-3s  %-15s  avg_log2FC  pct.1  pct.2  p_val_adj          ║\n",
              "#", "Gene"))
  cat("╠═══════════════════════════════════════════════════════════════════════╣\n")
  
  for (i in 1:nrow(cl_markers)) {
    cat(sprintf("║  %2d.  %-15s  %8.3f  %5.3f  %5.3f  %s\n",
                i,
                cl_markers$gene[i],
                cl_markers$avg_log2FC[i],
                cl_markers$pct.1[i],
                cl_markers$pct.2[i],
                formatC(cl_markers$p_val_adj[i], format = "e", digits = 2)))
  }
  cat("╚═══════════════════════════════════════════════════════════════════════╝\n\n")
}

# ----------------------------------------------------------
# Feature plots of canonical cell type markers
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Generating feature plots for canonical markers\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Define canonical markers for esophageal/head-neck squamous cell carcinoma
canonical_markers <- list(
  # Epithelial
  Epithelial_basal = c("KRT5", "KRT14", "TP63", "COL17A1"),
  Epithelial_suprabasal = c("KRT4", "KRT13", "KRT6A"),
  Epithelial_differentiated = c("IVL", "FLG", "SPRR1A", "SPRR2A"),
  Epithelial_cycling = c("MKI67", "TOP2A", "PCNA", "CDK1"),
  
  # Fibroblasts / CAF
  Fibroblast = c("COL1A1", "COL1A2", "DCN", "LUM", "VIM"),
  CAF = c("FAP", "ACTA2", "POSTN", "FN1", "PDPN"),
  Myofibroblast = c("ACTA2", "TAGLN", "MYH11"),
  
  # Immune - T cells
  T_cell = c("CD3D", "CD3E", "CD3G", "TRAC"),
  CD8_T = c("CD8A", "CD8B", "GZMK", "GZMB", "PRF1"),
  CD4_T = c("CD4", "IL7R", "CCR7"),
  Treg = c("FOXP3", "IL2RA", "CTLA4", "IKZF2"),
  NK = c("NKG7", "GNLY", "KLRD1", "KLRB1"),
  
  # Immune - Myeloid
  Macrophage = c("CD68", "CD163", "CSF1R", "MSR1"),
  Macrophage_SPP1 = c("SPP1", "APOE", "C1QA", "C1QB"),
  Monocyte = c("S100A8", "S100A9", "FCN1", "VCAN"),
  DC = c("CD1A", "CD1C", "CLEC9A", "LAMP3", "CCR7"),
  Mast_cell = c("TPSAB1", "TPSB2", "CPA3", "KIT"),
  Neutrophil = c("FCGR3B", "CXCR2", "CSF3R", "G0S2"),
  
  # Immune - B cells
  B_cell = c("CD79A", "MS4A1", "CD19", "PAX5"),
  Plasma = c("JCHAIN", "MZB1", "IGHG1", "SDC1"),
  
  # Endothelial
  Endothelial = c("PECAM1", "VWF", "CDH5", "CLDN5", "ERG"),
  Lymphatic_endo = c("PROX1", "LYVE1", "FLT4", "PDPN"),
  
  # Pericyte / Smooth muscle
  Pericyte_SMC = c("RGS5", "ACTA2", "PDGFRB", "NOTCH3", "MYH11"),
  
  # Neural
  Neural = c("S100B", "PLP1", "MPZ", "SOX10")
)

# Plot feature plots for each category
for (category in names(canonical_markers)) {
  genes <- canonical_markers[[category]]
  genes_present <- genes[genes %in% rownames(seu)]
  
  if (length(genes_present) > 0) {
    tryCatch({
      p <- FeaturePlot(seu, features = genes_present,
                       reduction = "umap", ncol = 2,
                       pt.size = 0.1, order = TRUE) &
        theme(plot.title = element_text(size = 10))
      
      ggsave(file.path(output_dir, sprintf("FeaturePlot_%s.pdf", category)), p,
             width = 8, height = ceiling(length(genes_present) / 2) * 3.5)
    }, error = function(e) {
      cat(sprintf("    FeaturePlot %s skipped: %s\n", category, e$message))
    })
  }
}

cat("  Feature plots saved.\n")

# ----------------------------------------------------------
# Dot plot of key markers across clusters
# ----------------------------------------------------------
cat("\n  Generating dot plot of key markers...\n")

key_markers <- c(
  # Epithelial
  "KRT5", "KRT14", "TP63", "KRT4", "KRT13", "IVL",
  # Cycling
  "MKI67", "TOP2A",
  # Fibroblast/CAF
  "COL1A1", "DCN", "FAP", "ACTA2", "POSTN",
  # T cell
  "CD3D", "CD8A", "CD4", "FOXP3", "NKG7",
  # Myeloid
  "CD68", "SPP1", "S100A8", "CD1C", "TPSAB1",
  # B/Plasma
  "CD79A", "JCHAIN",
  # Endothelial
  "PECAM1", "VWF",
  # Pericyte
  "RGS5", "PDGFRB"
)

key_markers_present <- key_markers[key_markers %in% rownames(seu)]

p_dot <- DotPlot(seu, features = key_markers_present, dot.scale = 5) +
  RotatedAxis() +
  theme(axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7)) +
  ggtitle("Key Markers Across Clusters")

ggsave(file.path(output_dir, "DotPlot_key_markers.pdf"), p_dot,
       width = 16, height = 8)

cat("  Dot plot saved.\n")

# ----------------------------------------------------------
# Heatmap of top markers
# ----------------------------------------------------------
cat("\n  Generating heatmap of top 5 markers per cluster...\n")

top5 <- all_markers %>%
  group_by(cluster) %>%
  slice_max(order_by = avg_log2FC, n = 5) %>%
  ungroup()

# Downsample for heatmap
tryCatch({
  p_heatmap <- DoHeatmap(seu, features = top5$gene,
                         size = 3, angle = 90) +
    theme(axis.text.y = element_text(size = 4)) +
    NoLegend()
  
  ggsave(file.path(output_dir, "Heatmap_top5_markers.pdf"), p_heatmap,
         width = 16, height = 12)
  cat("  Heatmap saved.\n")
}, error = function(e) {
  cat(sprintf("  Heatmap skipped: %s\n", e$message))
})

# ----------------------------------------------------------
# Save intermediate results
# ----------------------------------------------------------
saveRDS(all_markers, file.path(output_dir, "all_markers.rds"))

# ----------------------------------------------------------
# Summary report
# ----------------------------------------------------------
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════════════╗\n")
cat("║            MARKER FINDING — COMPLETE                                        ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                              ║\n")
cat(sprintf("║  Clustering resolution used: %s\n", res_col))
cat(sprintf("║  Number of clusters: %d\n", length(levels(Idents(seu)))))
cat(sprintf("║  Total markers found: %d\n", nrow(all_markers)))
cat("║                                                                              ║\n")
cat("║  Output files:                                                               ║\n")
cat("║    UMAP_clusters_for_annotation.pdf  — Cluster UMAP                          ║\n")
cat("║    all_cluster_markers.csv           — All markers (full table)              ║\n")
cat("║    top20_markers_per_cluster.csv     — Top 20 markers per cluster            ║\n")
cat("║    DotPlot_key_markers.pdf           — Dot plot canonical markers            ║\n")
cat("║    Heatmap_top5_markers.pdf          — Heatmap top 5 per cluster             ║\n")
cat("║    FeaturePlot_*.pdf                 — Feature plots per cell type           ║\n")
cat("║    all_markers.rds                   — Markers RDS for downstream            ║\n")
cat("║                                                                              ║\n")
cat("║  ══════════════════════════════════════════════════════════════════════════   ║\n")
cat("║                                                                              ║\n")
cat("║  WHAT I NEED FROM YOU TO COMPLETE ANNOTATION:                                ║\n")
cat("║                                                                              ║\n")
cat("║  Please provide the TOP 10 MARKERS for each cluster                          ║\n")
cat("║  (printed above and saved in top20_markers_per_cluster.csv)                  ║\n")
cat("║                                                                              ║\n")
cat("║  Specifically, I need:                                                       ║\n")
cat("║    1. The top marker gene names for each cluster number                      ║\n")
cat("║    2. The number of clusters at the resolution you want to annotate          ║\n")
cat("║    3. (Optional) The DotPlot result showing marker expression patterns       ║\n")
cat("║                                                                              ║\n")
cat("║  With this information, I can:                                               ║\n")
cat("║    - Assign cell type labels to each cluster                                 ║\n")
cat("║    - Generate the annotation code                                            ║\n")
cat("║    - Create final annotated UMAP figures                                     ║\n")
cat("║                                                                              ║\n")
cat("╚══════════════════════════════════════════════════════════════════════════════╝\n")
