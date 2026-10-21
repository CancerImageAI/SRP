# ============================================================
# 02_Integration.R — Normalization, Integration, Clustering
# ============================================================

options(Seurat.object.assay.version = "v3")

library(Seurat)
library(harmony)
library(tidyverse)
library(patchwork)
library(clustree)

# ----------------------------------------------------------
# Output directory
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/01_QC"
output_dir <- "XXXyour path/scRNA_Results_new/02_Integration"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Theme
# ----------------------------------------------------------
theme_paper <- function(base_size = 7) {
  theme_classic(base_size = base_size) %+replace%
    theme(
      text = element_text(size = base_size, color = "black"),
      axis.title = element_text(size = base_size, face = "bold"),
      axis.text = element_text(size = base_size - 1, color = "black"),
      plot.title = element_text(size = base_size + 1, face = "bold", hjust = 0.5),
      legend.title = element_text(size = base_size, face = "bold"),
      legend.text = element_text(size = base_size - 1),
      legend.key.size = unit(0.3, "cm"),
      axis.line = element_line(linewidth = 0.4, color = "black"),
      axis.ticks = element_line(linewidth = 0.3, color = "black"),
      panel.grid = element_blank(),
      plot.margin = margin(4, 4, 4, 4, "pt"),
      strip.text = element_text(size = base_size, face = "bold"),
      strip.background = element_blank()
    )
}

group_colors <- c("High_Risk" = "#D32F2F", "Low_Risk" = "#1976D2")

# ----------------------------------------------------------
# PART 1: Load cleaned data
# ----------------------------------------------------------
cat("━━━ PART 1: Loading data ━━━\n")
# seu <- readRDS(file.path(input_dir, "merged_clean_singlets.rds"))
# seu <- readRDS("XXXyour path/scRNA_Results_new/03_Annotation/seu_filtered_no_doublets.rds")
seu <- readRDS(file.path(output_dir, "seu_integrated_new_filtered_round3.rds"))
cat("Loaded:", ncol(seu), "cells x", nrow(seu), "genes\n")
cat("Samples:", paste(unique(seu$sample), collapse = ", "), "\n")
cat("Groups:", table(seu$group), "\n")

# ----------------------------------------------------------
# PART 2: Normalization and Variable Feature Selection
# ----------------------------------------------------------
cat("\n━━━ PART 2: Normalization ━━━\n")

seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 3000, verbose = FALSE)

# Plot variable features
top20 <- head(VariableFeatures(seu), 20)
p_var <- VariableFeaturePlot(seu)
p_var <- LabelPoints(plot = p_var, points = top20, repel = TRUE, xnudge = 0, ynudge = 0)
ggsave(file.path(output_dir, "Fig1_VariableFeatures.pdf"), p_var, width = 150, height = 100, units = "mm")

# Scale data (regress out mito and UMI count)
cat("Scaling data (regressing percent.mt and nCount_RNA)...\n")
seu <- ScaleData(seu, vars.to.regress = c("percent.mt", "nCount_RNA"), verbose = FALSE)

# ----------------------------------------------------------
# PART 3: PCA
# ----------------------------------------------------------
cat("\n━━━ PART 3: PCA ━━━\n")

seu <- RunPCA(seu, npcs = 50, verbose = FALSE)

# Elbow plot to determine dimensionality
p_elbow <- ElbowPlot(seu, ndims = 50) + 
  theme_paper() +
  geom_vline(xintercept = 30, linetype = "dashed", color = "red", linewidth = 0.3) +
  ggtitle("PCA Elbow Plot")

ggsave(file.path(output_dir, "Fig2_ElbowPlot.pdf"), p_elbow, width = 120, height = 80, units = "mm")

# PCA colored by sample (before integration)
p_pca_pre <- DimPlot(seu, reduction = "pca", group.by = "sample", pt.size = 0.1) +
  theme_paper() + ggtitle("PCA - Before Integration (by Sample)")

p_pca_group <- DimPlot(seu, reduction = "pca", group.by = "group", 
                       cols = group_colors, pt.size = 0.1) +
  theme_paper() + ggtitle("PCA - Before Integration (by Group)")

ggsave(file.path(output_dir, "Fig3_PCA_before_integration.pdf"), 
       p_pca_pre | p_pca_group, width = 250, height = 100, units = "mm")

# ----------------------------------------------------------
# PART 4: Harmony Integration
# ----------------------------------------------------------
cat("\n━━━ PART 4: Harmony Integration ━━━\n")

# Run Harmony to correct batch effects across patients
# We integrate by "sample" to remove patient-specific technical batch effects
# but preserve biological differences between risk groups
seu <- RunHarmony(seu, 
                  group.by.vars = "sample",
                  reduction.use = "pca",
                  dims.use = 1:30,
                  max_iter = 30,
                  plot_convergence = FALSE,
                  verbose = TRUE)

# Verify Harmony embedding exists
cat("Harmony embeddings:", dim(Embeddings(seu, "harmony")), "\n")

# PCA vs Harmony comparison
p_harmony <- DimPlot(seu, reduction = "harmony", group.by = "sample", pt.size = 0.1) +
  theme_paper() + ggtitle("Harmony - by Sample")

p_harmony_group <- DimPlot(seu, reduction = "harmony", group.by = "group", 
                           cols = group_colors, pt.size = 0.1) +
  theme_paper() + ggtitle("Harmony - by Group")

ggsave(file.path(output_dir, "Fig4_Harmony_integration.pdf"), 
       (p_pca_pre | p_harmony) / (p_pca_group | p_harmony_group), 
       width = 250, height = 180, units = "mm")

# ----------------------------------------------------------
# PART 5: UMAP and Clustering
# ----------------------------------------------------------
cat("\n━━━ PART 5: UMAP and Clustering ━━━\n")

# Run UMAP on Harmony embeddings
seu <- RunUMAP(seu, reduction = "harmony", dims = 1:30, verbose = FALSE)

# Find neighbors using Harmony
seu <- FindNeighbors(seu, reduction = "harmony", dims = 1:30, verbose = FALSE)

# Test multiple resolutions
cat("Finding clusters at multiple resolutions...\n")
seu <- FindClusters(seu, resolution = 0.3, verbose = FALSE)
seu <- FindClusters(seu, resolution = 0.5, verbose = FALSE)
seu <- FindClusters(seu, resolution = 0.8, verbose = FALSE)
seu <- FindClusters(seu, resolution = 1.0, verbose = FALSE)
seu <- FindClusters(seu, resolution = 1.2, verbose = FALSE)
seu <- FindClusters(seu, resolution = 1.5, verbose = FALSE)

# Clustree to visualize resolution selection
p_clustree <- clustree(seu, prefix = "RNA_snn_res.") + 
  theme(legend.position = "right")

ggsave(file.path(output_dir, "Fig5_Clustree.pdf"), p_clustree, width = 250, height = 200, units = "mm")

# Set working resolution (0.5 is typically good for TME analysis)
# You can adjust after reviewing clustree
Idents(seu) <- "RNA_snn_res.0.5"
cat("Number of clusters at res 0.5:", length(unique(Idents(seu))), "\n")

# ----------------------------------------------------------
# PART 6: Visualization
# ----------------------------------------------------------
cat("\n━━━ PART 6: Visualization ━━━\n")

# UMAP by clusters
p_umap_cluster <- DimPlot(seu, reduction = "umap", label = TRUE, label.size = 3, 
                          repel = TRUE, pt.size = 0.1) +
  theme_paper() + ggtitle("UMAP - Clusters (res 0.5)") + NoLegend()

# UMAP by sample
p_umap_sample <- DimPlot(seu, reduction = "umap", group.by = "sample", 
                         pt.size = 0.1, shuffle = TRUE) +
  theme_paper() + ggtitle("UMAP - by Sample")

# UMAP by risk group
p_umap_group <- DimPlot(seu, reduction = "umap", group.by = "group", 
                        cols = group_colors, pt.size = 0.1, shuffle = TRUE) +
  theme_paper() + ggtitle("UMAP - by Risk Group")

# Combined
ggsave(file.path(output_dir, "Fig6_UMAP_overview.pdf"),
       p_umap_cluster | p_umap_sample | p_umap_group,
       width = 300, height = 100, units = "mm")

# UMAP split by sample
p_split_sample <- DimPlot(seu, reduction = "umap", split.by = "sample", 
                          group.by = "RNA_snn_res.0.5",
                          ncol = 4, pt.size = 0.05, label = TRUE, label.size = 2) +
  theme_paper() + NoLegend()

ggsave(file.path(output_dir, "Fig7_UMAP_split_by_sample.pdf"),
       p_split_sample, width = 300, height = 225, units = "mm")

# UMAP split by risk group
p_split_group <- DimPlot(seu, reduction = "umap", split.by = "group", 
                         group.by = "RNA_snn_res.0.5",
                         cols = NULL, pt.size = 0.1, label = TRUE, label.size = 3) +
  theme_paper()

ggsave(file.path(output_dir, "Fig8_UMAP_split_by_group.pdf"),
       p_split_group, width = 200, height = 100, units = "mm")

# ----------------------------------------------------------
# PART 7: Integration Quality Metrics
# ----------------------------------------------------------
cat("\n━━━ PART 7: Integration Quality ━━━\n")

# Check mixing of samples within clusters
mixing_table <- seu@meta.data %>%
  group_by(RNA_snn_res.0.5, sample) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(RNA_snn_res.0.5) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

# Heatmap of sample proportions per cluster
mixing_wide <- mixing_table %>%
  select(RNA_snn_res.0.5, sample, prop) %>%
  pivot_wider(names_from = sample, values_from = prop, values_fill = 0)

write.csv(mixing_wide, file.path(output_dir, "Table_cluster_sample_proportions.csv"), row.names = FALSE)

# Cell counts per cluster per group
cluster_group_table <- seu@meta.data %>%
  group_by(RNA_snn_res.0.5, group) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(RNA_snn_res.0.5) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

p_bar <- ggplot(cluster_group_table, aes(x = RNA_snn_res.0.5, y = prop, fill = group)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = group_colors) +
  theme_paper() +
  labs(x = "Cluster", y = "Proportion", fill = "Risk Group",
       title = "Risk Group Composition per Cluster") +
  theme(axis.text.x = element_text(angle = 0))

ggsave(file.path(output_dir, "Fig9_cluster_group_composition.pdf"),
       p_bar, width = 178, height = 80, units = "mm")

# ----------------------------------------------------------
# PART 8: QC features on UMAP
# ----------------------------------------------------------
cat("\n━━━ PART 8: QC on UMAP ━━━\n")

p_qc_umap <- FeaturePlot(seu, 
                         features = c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.ribo"),
                         ncol = 2, pt.size = 0.05) &
  theme_paper() &
  scale_color_viridis_c()

ggsave(file.path(output_dir, "Fig10_QC_on_UMAP.pdf"),
       p_qc_umap, width = 200, height = 180, units = "mm")

# ----------------------------------------------------------
# PART 9: Multiple resolution comparison
# ----------------------------------------------------------
cat("\n━━━ PART 9: Resolution comparison ━━━\n")

p_res03 <- DimPlot(seu, group.by = "RNA_snn_res.0.3", label = TRUE, pt.size = 0.05) + 
  theme_paper() + NoLegend() + ggtitle("Res 0.3")
p_res05 <- DimPlot(seu, group.by = "RNA_snn_res.0.5", label = TRUE, pt.size = 0.05) + 
  theme_paper() + NoLegend() + ggtitle("Res 0.5")
p_res08 <- DimPlot(seu, group.by = "RNA_snn_res.0.8", label = TRUE, pt.size = 0.05) + 
  theme_paper() + NoLegend() + ggtitle("Res 0.8")
p_res10 <- DimPlot(seu, group.by = "RNA_snn_res.1", label = TRUE, pt.size = 0.05) + 
  theme_paper() + NoLegend() + ggtitle("Res 1.0")
p_res12 <- DimPlot(seu, group.by = "RNA_snn_res.1.2", label = TRUE, pt.size = 0.05) + 
  theme_paper() + NoLegend() + ggtitle("Res 1.2")
p_res15 <- DimPlot(seu, group.by = "RNA_snn_res.1.5", label = TRUE, pt.size = 0.05) + 
  theme_paper() + NoLegend() + ggtitle("Res 1.5")

ggsave(file.path(output_dir, "Fig11_resolution_comparison.pdf"),
       (p_res03 | p_res05 | p_res08) / (p_res10 | p_res12 | p_res15),
       width = 300, height = 200, units = "mm")

# ----------------------------------------------------------
# PART 10: Summary statistics
# ----------------------------------------------------------
cat("\n━━━ PART 10: Summary ━━━\n")

# Cluster summary
cluster_summary <- seu@meta.data %>%
  group_by(RNA_snn_res.0.5) %>%
  summarise(
    n_cells = n(),
    pct_total = round(n() / ncol(seu) * 100, 2),
    n_samples = n_distinct(sample),
    median_genes = median(nFeature_RNA),
    median_UMI = median(nCount_RNA),
    .groups = "drop"
  ) %>%
  arrange(desc(n_cells))

write.csv(cluster_summary, file.path(output_dir, "Table_cluster_summary.csv"), row.names = FALSE)
cat("\nCluster summary:\n")
print(as.data.frame(cluster_summary))

# Sample summary
sample_summary <- seu@meta.data %>%
  group_by(sample, group) %>%
  summarise(n_cells = n(), .groups = "drop")

write.csv(sample_summary, file.path(output_dir, "Table_sample_summary.csv"), row.names = FALSE)
cat("\nSample summary:\n")
print(as.data.frame(sample_summary))

# ----------------------------------------------------------
# Save final integrated object
# ----------------------------------------------------------
cat("\n━━━ Saving ━━━\n")
saveRDS(seu, file.path(output_dir, "seu_integrated_new_3rd.rds"))

cat("\n━━━ 02_Integration COMPLETE ━━━\n")
cat("Output directory:", output_dir, "\n")
cat("Final object:", ncol(seu), "cells,", nrow(seu), "genes,", 
    length(unique(seu$RNA_snn_res.0.5)), "clusters (res 0.5)\n")
cat("\nKey outputs:\n")
cat("  - seu_integrated.rds (main object)\n")
cat("  - Fig5_Clustree.pdf (choose resolution)\n")
cat("  - Fig6_UMAP_overview.pdf (clusters, samples, groups)\n")
cat("  - Fig9_cluster_group_composition.pdf (risk group per cluster)\n")
cat("\nNext step: 03_CellType_Annotation.R\n")