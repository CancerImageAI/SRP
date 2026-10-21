# ============================================================
# Cell Type Annotation — Fixed & Complete
# Input: seu_integrated_new_3rd.rds
# ============================================================

library(Seurat)
library(tidyverse)
library(ggplot2)
library(RColorBrewer)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/02_Integration_New"
output_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Load Seurat object
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading integrated Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_integrated_new_3rd.rds"))
cat(sprintf("  Cells: %d | Genes: %d\n", ncol(seu), nrow(seu)))

# ----------------------------------------------------------
# Set clustering identity (resolution 0.5)
# ----------------------------------------------------------
cluster_cols <- grep("snn_res|seurat_clusters", colnames(seu@meta.data), value = TRUE)
cat("  Available cluster columns:", paste(cluster_cols, collapse = ", "), "\n")

res_col <- grep("0\\.5", cluster_cols, value = TRUE)[1]
if (is.na(res_col) || is.null(res_col)) {
  possible <- c("RNA_snn_res.0.5", "integrated_snn_res.0.5",
                "SCT_snn_res.0.5", "wsnn_res.0.5", "seurat_clusters")
  res_col <- possible[possible %in% colnames(seu@meta.data)][1]
}

cat(sprintf("  Using clustering column: %s\n", res_col))
Idents(seu) <- res_col

# Check cluster IDs
cluster_ids <- levels(Idents(seu))
cat(sprintf("  Number of clusters: %d\n", length(cluster_ids)))
cat("  Cluster IDs:", paste(cluster_ids, collapse = ", "), "\n")
cat("  Cluster sizes:\n")
print(sort(table(Idents(seu)), decreasing = TRUE))

# ----------------------------------------------------------
# Define cluster-to-cell type mapping
# Based on top20_markers_per_cluster.csv analysis:
#
# Cluster 0:  CRTAM, GZMK, CD8A, KLRK1, NKG7, PRF1, GZMA, GZMB → CD8_T_effector
# Cluster 1:  MMP11, COL11A1, POSTN, COL1A1, LRRC15, ITGA11 → CAF
# Cluster 2:  FOXP3, TNFRSF4, IL2RA, RTKN2, BATF, CD28 → Treg
# Cluster 3:  CD163, SPP1, C1QC, APOE, C1QA, C1QB, MSR1 → Macrophage_SPP1
# Cluster 4:  COL17A1, CDH3, LAMA3, TENM2, GNGT1 → Epithelial_basal
# Cluster 5:  KRT16, KRT6B, KRT17, KRT14, KRT5, S100A2, SFN → Epithelial_squamous
# Cluster 6:  CXCR2, FCGR3B, CSF3R, G0S2, AQP9, S100A12 → Neutrophil
# Cluster 7:  PI16, CHRDL1, PLA2G2A, C7, CFD, IGF1, ELN → Fibroblast_PI16
# Cluster 8:  RGS5, MYH11, CDH6, TRPC6, PPP1R14A, SYNPO2 → Pericyte_SMC
# Cluster 9:  CLEC14A, CDH5, KDR, PLVAP, VWF, CLDN5, TEK → Endothelial
# Cluster 10: IGLC2, IGKC, IGHG1, IGHG3, JCHAIN, MZB1 → Plasma
# Cluster 11: MS4A1, PAX5, FCRL1, BANK1, BLK, CD79A → B_cell
# Cluster 12: MKI67, RRM2, ASPM, KIF18B, DLGAP5, CDCA8 → Proliferating
# Cluster 13: FLT3, S100B, LAMP3, IDO1, CD86, CSF2RA → DC
# Cluster 14: RMEL3, ROCR, lncRNAs (stem-like signature) → Epithelial_stem_like
# Cluster 15: PSCA, CEACAM5, LCN2, CEACAM6, SPRR3, SCEL → Epithelial_differentiated
# Cluster 16: TPSB2, CPA3, TPSAB1, MS4A2, KIT, GATA2 → Mast_cell
# Cluster 17: POSTN, KCND2, THY1, GPC6, SGCD, ADAMTS12 → Myofibroblast
# ----------------------------------------------------------

cluster_annotations <- c(
  "0"  = "CD8_T_effector",
  "1"  = "CAF",
  "2"  = "Treg",
  "3"  = "Macrophage_SPP1",
  "4"  = "Epithelial_basal",
  "5"  = "Epithelial_squamous",
  "6"  = "Neutrophil",
  "7"  = "Fibroblast_PI16",
  "8"  = "Pericyte_SMC",
  "9"  = "Endothelial",
  "10" = "Plasma_cell",
  "11" = "B_cell",
  "12" = "Proliferating",
  "13" = "DC_mature",
  "14" = "Epithelial_stem_like",
  "15" = "Epithelial_differentiated",
  "16" = "Mast_cell",
  "17" = "Myofibroblast"
)

# ----------------------------------------------------------
# *** FIX: Direct assignment via metadata (avoids RenameIdents error) ***
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Assigning cell type annotations\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Get cluster assignment for each cell as character
cell_clusters <- as.character(seu@meta.data[[res_col]])

# Map cluster → cell type
seu@meta.data$cell_type <- cluster_annotations[as.character(seu@meta.data[[res_col]])]

# Check for unmapped
n_unmapped <- sum(is.na(seu$cell_type))
if (n_unmapped > 0) {
  unmapped_cls <- unique(cell_clusters[is.na(seu$cell_type)])
  cat(sprintf("  ⚠ WARNING: %d cells unmapped. Clusters: %s\n",
              n_unmapped, paste(unmapped_cls, collapse = ", ")))
  seu$cell_type[is.na(seu$cell_type)] <- "Unknown"
} else {
  cat("  ✓ All 95,120 cells annotated successfully.\n")
}

# ----------------------------------------------------------
# Set as ordered factor
# ----------------------------------------------------------
cell_type_order <- c(
  "Epithelial_basal",
  "Epithelial_squamous",
  "Epithelial_differentiated",
  "Epithelial_stem_like",
  "CAF",
  "Myofibroblast",
  "Fibroblast_PI16",
  "Pericyte_SMC",
  "Endothelial",
  "CD8_T_effector",
  "Treg",
  "B_cell",
  "Plasma_cell",
  "Macrophage_SPP1",
  "DC_mature",
  "Neutrophil",
  "Mast_cell",
  "Proliferating"
)

seu$cell_type <- factor(seu$cell_type, levels = cell_type_order)

cat("\n  Cell type distribution:\n")
print(table(seu$cell_type))

# ----------------------------------------------------------
# Add lineage annotation
# ----------------------------------------------------------
lineage_map <- c(
  "Epithelial_basal" = "Epithelial",
  "Epithelial_squamous" = "Epithelial",
  "Epithelial_differentiated" = "Epithelial",
  "Epithelial_stem_like" = "Epithelial",
  "CAF" = "Stromal",
  "Myofibroblast" = "Stromal",
  "Fibroblast_PI16" = "Stromal",
  "Pericyte_SMC" = "Stromal",
  "Endothelial" = "Endothelial",
  "CD8_T_effector" = "Lymphoid",
  "Treg" = "Lymphoid",
  "B_cell" = "Lymphoid",
  "Plasma_cell" = "Lymphoid",
  "Macrophage_SPP1" = "Myeloid",
  "DC_mature" = "Myeloid",
  "Neutrophil" = "Myeloid",
  "Mast_cell" = "Myeloid",
  "Proliferating" = "Proliferating"
)


seu@meta.data$lineage <- lineage_map[as.character(seu@meta.data$cell_type)]
seu@meta.data$lineage <- factor(seu@meta.data$lineage,
                                levels = c("Epithelial", "Stromal", "Endothelial",
                                           "Lymphoid", "Myeloid", "Proliferating"))


cat("\n  Lineage distribution:\n")
print(table(seu$lineage))

# ----------------------------------------------------------
# Color palettes
# ----------------------------------------------------------
cell_type_colors <- c(
  "Epithelial_basal"          = "#8491B4",
  "Epithelial_squamous"       = "#B09C85",
  "Epithelial_differentiated" = "#91D1C2",
  "Epithelial_stem_like"      = "#7E6148",
  "CAF"                       = "#E64B35",
  "Myofibroblast"             = "#F39B7F",
  "Fibroblast_PI16"           = "#AED6F1",
  "Pericyte_SMC"              = "#7DCEA0",
  "Endothelial"               = "#4DBBD5",
  "CD8_T_effector"            = "#00A087",
  "Treg"                      = "#85C1E9",
  "B_cell"                    = "#3C5488",
  "Plasma_cell"               = "#BB8FCE",
  "Macrophage_SPP1"           = "#DC0000",
  "DC_mature"                 = "#F0E685",
  "Neutrophil"                = "#D4AC0D",
  "Mast_cell"                 = "#F1948A",
  "Proliferating"             = "#82E0AA"
)

lineage_colors <- c(
  "Epithelial"    = "#8491B4",
  "Stromal"       = "#E64B35",
  "Endothelial"   = "#4DBBD5",
  "Lymphoid"      = "#00A087",
  "Myeloid"       = "#DC0000",
  "Proliferating" = "#82E0AA"
)

# ----------------------------------------------------------
# UMAP Visualizations
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Generating UMAP figures\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

Idents(seu) <- "cell_type"

# 1. UMAP by cell type (labeled)
p1 <- DimPlot(seu, reduction = "umap", group.by = "cell_type",
              label = TRUE, label.size = 3, repel = TRUE,
              pt.size = 0.1, cols = cell_type_colors) +
  ggtitle("Cell Type Annotation") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.text = element_text(size = 8))

ggsave(file.path(output_dir, "UMAP_cell_type.pdf"), p1,
       width = 12, height = 9)

# 2. UMAP by cell type (no labels)
p1b <- DimPlot(seu, reduction = "umap", group.by = "cell_type",
               label = FALSE, pt.size = 0.1, cols = cell_type_colors) +
  ggtitle("Cell Type Annotation") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave(file.path(output_dir, "UMAP_cell_type_noLabel.pdf"), p1b,
       width = 12, height = 9)

# 3. UMAP by lineage
p2 <- DimPlot(seu, reduction = "umap", group.by = "lineage",
              label = TRUE, label.size = 4, repel = TRUE,
              pt.size = 0.1, cols = lineage_colors) +
  ggtitle("Cell Lineage") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave(file.path(output_dir, "UMAP_lineage.pdf"), p2,
       width = 10, height = 8)

# 4. UMAP with cluster numbers
Idents(seu) <- res_col
p3 <- DimPlot(seu, reduction = "umap", label = TRUE, label.size = 4,
              repel = TRUE, pt.size = 0.1) +
  ggtitle(sprintf("Clusters (%s)", res_col)) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave(file.path(output_dir, "UMAP_clusters_numbered.pdf"), p3,
       width = 10, height = 8)

# 5. Split by group
if ("group" %in% colnames(seu@meta.data)) {
  Idents(seu) <- "cell_type"
  
  p4 <- DimPlot(seu, reduction = "umap", group.by = "cell_type",
                split.by = "group", label = FALSE,
                pt.size = 0.1, cols = cell_type_colors, ncol = 2) +
    ggtitle("Cell Types by Risk Group") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  ggsave(file.path(output_dir, "UMAP_cell_type_splitByGroup.pdf"), p4,
         width = 18, height = 8)
}

cat("  UMAP figures saved.\n")

# ----------------------------------------------------------
# Cell type proportions
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Cell type proportion analysis\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if ("group" %in% colnames(seu@meta.data)) {
  prop_df <- seu@meta.data %>%
    group_by(group, cell_type) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(group) %>%
    mutate(proportion = n / sum(n)) %>%
    ungroup()
  
  # Stacked bar
  p_prop <- ggplot(prop_df, aes(x = group, y = proportion, fill = cell_type)) +
    geom_col(width = 0.7, color = "white", linewidth = 0.2) +
    scale_fill_manual(values = cell_type_colors) +
    theme_classic(base_size = 10) +
    labs(x = "", y = "Proportion", fill = "Cell Type",
         title = "Cell Type Composition by Risk Group") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  ggsave(file.path(output_dir, "Barplot_cell_type_proportion.pdf"), p_prop,
         width = 8, height = 6)
  
  # Grouped bar
  p_prop2 <- ggplot(prop_df, aes(x = cell_type, y = proportion, fill = group)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")) +
    theme_classic(base_size = 9) +
    labs(x = "", y = "Proportion", fill = "Group",
         title = "Cell Type Proportion Comparison") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          plot.title = element_text(hjust = 0.5, face = "bold"))
  
  ggsave(file.path(output_dir, "Barplot_cell_type_proportion_grouped.pdf"), p_prop2,
         width = 12, height = 6)
  
  write.csv(prop_df, file.path(output_dir, "cell_type_proportions.csv"), row.names = FALSE)
  cat("  Proportion plots saved.\n")
}


# ----------------------------------------------------------
# Validation dot plot
# ----------------------------------------------------------
cat("\n  Generating validation dot plot...\n")

DefaultAssay(seu) <- "RNA"
Idents(seu) <- "cell_type"

# Check if RNA data is normalized
test_data <- GetAssayData(seu, assay = "RNA", layer  = "data")
if (max(test_data[1:min(100, nrow(test_data)), 1:min(100, ncol(test_data))]) == 0) {
  cat("  Normalizing RNA assay...\n")
  seu <- NormalizeData(seu, assay = "RNA")
}

validation_markers <- c(
  # CD8 T effector
  "CD8A", "GZMK", "NKG7", "PRF1", "GZMB",
  # Treg
  "FOXP3", "IL2RA", "CTLA4",
  # CAF
  "COL11A1", "MMP11", "POSTN", "LRRC15",
  # Myofibroblast
  "THY1", "GPC6", "SGCD", "KCND2",
  # Fibroblast_PI16
  "PI16", "CHRDL1", "CFD", "PLA2G2A",
  # Macrophage SPP1
  "CD163", "SPP1", "C1QA", "APOE",
  # DC
  "FLT3", "LAMP3", "IDO1", "CD86",
  # Neutrophil
  "FCGR3B", "CXCR2", "CSF3R", "G0S2",
  # B cell
  "MS4A1", "PAX5", "CD79A",
  # Plasma
  "JCHAIN", "MZB1", "IGHG1",
  # Endothelial
  "CDH5", "VWF", "PLVAP", "KDR",
  # Pericyte/SMC
  "RGS5", "MYH11",
  # Epithelial basal
  "COL17A1", "CDH3", "LAMA3",
  # Epithelial squamous
  "KRT16", "KRT6B", "KRT14", "KRT5",
  # Epithelial differentiated
  "CEACAM5", "PSCA", "LCN2", "SPRR3",
  # Mast cell
  "TPSB2", "CPA3", "KIT",
  # Proliferating
  "MKI67", "TOP2A", "RRM2"
)


validation_markers_present <- validation_markers[validation_markers %in% rownames(seu)]
cat(sprintf("  %d / %d markers found in dataset\n",
            length(validation_markers_present), length(validation_markers)))
cat("Original marker count:", length(validation_markers_present), "\n")
cat("Unique marker count:", length(unique(validation_markers_present)), "\n")
# Find duplicates
duplicated_markers <- validation_markers_present[duplicated(validation_markers_present)]
if(length(duplicated_markers) > 0) {
  cat("\n⚠️  Duplicated markers found:\n")
  print(table(validation_markers_present[validation_markers_present %in% duplicated_markers]))
}

# Step 2: Remove duplicates
validation_markers_present <- unique(validation_markers_present)

# Step 3: Verify markers exist in your Seurat object
available_markers <- validation_markers_present[
  validation_markers_present %in% rownames(seu)
]

missing_markers <- setdiff(validation_markers_present, available_markers)
if(length(missing_markers) > 0) {
  cat("\n⚠️  Markers not found in data:\n")
  print(missing_markers)
}

p_dot <- DotPlot(seu, features = validation_markers_present, dot.scale = 4) +
  RotatedAxis() +
  theme(axis.text.x = element_text(size = 6, face = "italic"),
        axis.text.y = element_text(size = 7)) +
  ggtitle("Marker Validation by Cell Type")

ggsave(file.path(output_dir, "DotPlot_validation_markers.pdf"), p_dot,
       width = 18, height = 7)

cat("  Dot plot saved.\n")

# ----------------------------------------------------------
# Save annotated Seurat object
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Saving annotated Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

Idents(seu) <- "cell_type"

output_file <- file.path(output_dir, "seu_annotated_3rd.rds")
saveRDS(seu, output_file)

cat(sprintf("  Saved to: %s\n", output_file))
cat(sprintf("  File size: %.1f MB\n", file.size(output_file) / 1024^2))

# ----------------------------------------------------------
# Summary mapping table
# ----------------------------------------------------------
mapping_table <- data.frame(
  Cluster = names(cluster_annotations),
  Cell_Type = cluster_annotations,
  N_Cells = as.numeric(table(seu@meta.data[[res_col]])[names(cluster_annotations)]),
  Lineage = lineage_map[cluster_annotations],
  Key_Markers = c(
    "CRTAM, GZMK, CD8A, KLRK1, NKG7, PRF1, GZMA, GZMB",
    "MMP11, COL11A1, POSTN, COL1A1, LRRC15, ITGA11, AEBP1",
    "FOXP3, TNFRSF4, IL2RA, RTKN2, BATF, CCR6, CD28",
    "CD163, SPP1, C1QC, APOE, C1QA, C1QB, MSR1, MRC1, CD14",
    "COL17A1, CDH3, LAMA3, GNGT1, TENM2",
    "DAPL1, KRT16, KRT6B, KRT17, KRT14, KRT5, S100A2, SFN",
    "CXCR2, FCGR3B, CSF3R, G0S2, AQP9, S100A12, HCAR3",
    "PI16, CHRDL1, PLA2G2A, C7, ABI3BP, CFD, IGF1, ELN",
    "RGS5, MYH11, CDH6, TRPC6, PPP1R14A, SYNPO2, LMOD1",
    "CLEC14A, CDH5, KDR, PLVAP, VWF, CLDN5, TEK, TIE1",
    "IGLC2, IGKC, IGHG1, IGHG3, JCHAIN, MZB1, FCRL5",
    "MS4A1, PAX5, FCRL1, BANK1, BLK, CD79A, CD22",
    "MKI67, RRM2, ASPM, KIF18B, DLGAP5, CDCA8",
    "FLT3, S100B, LAMP3, IDO1, CD86, CSF2RA",
    "RMEL3, ROCR, LINC01122, LINC01205, COL9A1, ECE2",
    "PSCA, CEACAM5, LCN2, CEACAM6, SPRR3, SCEL, TMPRSS2",
    "TPSB2, CPA3, TPSAB1, MS4A2, KIT, GATA2, HDC, HPGDS",
    "POSTN, KCND2, THY1, GPC6, SGCD, ADAMTS12, COL5A3"
  ),
  stringsAsFactors = FALSE
)

write.csv(mapping_table, file.path(output_dir, "cluster_annotation_mapping.csv"),
          row.names = FALSE)

# ----------------------------------------------------------
# Cell type composition per sample
# ----------------------------------------------------------
if ("orig.ident" %in% colnames(seu@meta.data)) {
  sample_ct <- seu@meta.data %>%
    group_by(orig.ident, cell_type) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(orig.ident) %>%
    mutate(proportion = n / sum(n)) %>%
    ungroup()
  
  write.csv(sample_ct, file.path(output_dir, "cell_type_per_sample.csv"), row.names = FALSE)
}


# ----------------------------------------------------------
# Final report
# ----------------------------------------------------------
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════════════╗\n")
cat("║              CELL TYPE ANNOTATION — COMPLETE                                ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                              ║\n")
cat(sprintf("║  Total cells:       %6d                                                  ║\n", ncol(seu)))
cat(sprintf("║  Cell types:        %6d                                                  ║\n", length(unique(seu$cell_type))))
cat(sprintf("║  Resolution used:   %s                                              ║\n", res_col))
cat("║                                                                              ║\n")
cat("║  Annotation summary:                                                         ║\n")
for (ct in levels(seu$cell_type)) {
  n <- sum(seu$cell_type == ct, na.rm = TRUE)
  pct <- n / ncol(seu) * 100
  cat(sprintf("║    %-28s  %6d  (%5.1f%%)                      ║\n", ct, n, pct))
}
cat("║                                                                              ║\n")
cat("║  Output files:                                                               ║\n")
cat("║    seu_annotated_3rd.rds                  — Annotated Seurat object          ║\n")
cat("║    UMAP_cell_type.pdf                     — UMAP with labels                 ║\n")
cat("║    UMAP_cell_type_noLabel.pdf             — UMAP clean                       ║\n")
cat("║    UMAP_lineage.pdf                       — UMAP by lineage                  ║\n")
cat("║    UMAP_clusters_numbered.pdf             — UMAP with cluster numbers        ║\n")
cat("║    UMAP_cell_type_splitByGroup.pdf        — UMAP split by risk group         ║\n")
cat("║    DotPlot_validation_markers.pdf         — Marker validation                ║\n")
cat("║    Barplot_cell_type_proportion.pdf       — Stacked bar                      ║\n")
cat("║    Barplot_cell_type_proportion_grouped.pdf — Grouped bar                    ║\n")
cat("║    cluster_annotation_mapping.csv         — Annotation table                 ║\n")
cat("║    cell_type_proportions.csv              — Proportions by group             ║\n")
cat("║    cell_type_per_sample.csv               — Per-sample composition           ║\n")
cat("║                                                                              ║\n")
cat("║  Next steps:                                                                 ║\n")
cat("║    1. Review UMAP_cell_type.pdf for annotation quality                       ║\n")
cat("║    2. Review DotPlot_validation_markers.pdf for marker specificity           ║\n")
cat("║    3. If satisfied → proceed to:                                             ║\n")
cat("║       - DEG analysis between High/Low risk per cell type                     ║\n")
cat("║       - CellChat cell-cell communication                                    ║\n")
cat("║       - CAF subtype analysis                                                 ║\n")
cat("║       - Trajectory/pseudotime                                                ║\n")
cat("║                                                                              ║\n")
cat("╚══════════════════════════════════════════════════════════════════════════════╝\n")

cat("\n━━━ Done ━━━\n")
