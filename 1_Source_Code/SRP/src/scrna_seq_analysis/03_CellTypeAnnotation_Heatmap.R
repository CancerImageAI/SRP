# ============================================================
# Cell Type Annotation Heatmap
# Input: seu_integrated_new_3rd.rds
# Output: marker heatmap and validation dot plot
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

input_rds <- file.path(input_dir, "seu_integrated_new_3rd.rds")
if (!file.exists(input_rds)) {
  stop("Input Seurat object not found: ", input_rds)
}

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
get_assay_data_safe <- function(object, assay, layer = "data") {
  tryCatch(
    GetAssayData(object, assay = assay, layer = layer),
    error = function(e) GetAssayData(object, assay = assay, slot = layer)
  )
}

line <- paste(rep("=", 63), collapse = "")

# ----------------------------------------------------------
# Load Seurat object
# ----------------------------------------------------------
cat(line, "\n")
cat("  Loading integrated Seurat object\n")
cat(line, "\n")

seu <- readRDS(input_rds)
cat(sprintf("  Cells: %d | Genes: %d\n", ncol(seu), nrow(seu)))

# ----------------------------------------------------------
# Set clustering identity: resolution 0.5 preferred
# ----------------------------------------------------------
cluster_cols <- grep("snn_res|seurat_clusters", colnames(seu@meta.data), value = TRUE)
cat("  Available cluster columns:", paste(cluster_cols, collapse = ", "), "\n")

preferred_cols <- c(
  "RNA_snn_res.0.5",
  "integrated_snn_res.0.5",
  "SCT_snn_res.0.5",
  "wsnn_res.0.5"
)
res_col <- preferred_cols[preferred_cols %in% colnames(seu@meta.data)][1]

if (is.na(res_col) || length(res_col) == 0) {
  res_col <- grep("(^|\\.)0\\.5$", cluster_cols, value = TRUE)[1]
}
if (is.na(res_col) || length(res_col) == 0) {
  res_col <- "seurat_clusters"
}
if (!res_col %in% colnames(seu@meta.data)) {
  stop("No suitable clustering column found. Available columns: ", paste(cluster_cols, collapse = ", "))
}

cat(sprintf("  Using clustering column: %s\n", res_col))
Idents(seu) <- res_col

cluster_ids <- levels(Idents(seu))
cat(sprintf("  Number of clusters: %d\n", length(cluster_ids)))
cat("  Cluster IDs:", paste(cluster_ids, collapse = ", "), "\n")
cat("  Cluster sizes:\n")
print(sort(table(Idents(seu)), decreasing = TRUE))

# ----------------------------------------------------------
# Manual cluster annotation
# Keep these names synchronized with downstream figure labels.
# ----------------------------------------------------------
base_annotations <- c(
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

missing_annotation <- setdiff(cluster_ids, names(base_annotations))
if (length(missing_annotation) > 0) {
  cat("  Warning: clusters without manual annotation will be excluded:",
      paste(missing_annotation, collapse = ", "), "\n")
}

cluster_vector <- as.character(seu@meta.data[[res_col]])
names(cluster_vector) <- colnames(seu)
cells_keep <- names(cluster_vector)[cluster_vector %in% names(base_annotations)]
if (length(cells_keep) == 0) {
  stop("No cells matched the annotated cluster IDs. Check res_col and base_annotations.")
}
cat(sprintf("  Cells retained for annotated clusters: %d / %d\n", length(cells_keep), ncol(seu)))

seu_18 <- subset(seu, cells = cells_keep)
seu_18$cell_type <- unname(base_annotations[as.character(seu_18@meta.data[[res_col]])])
if (any(is.na(seu_18$cell_type))) {
  stop("Some retained cells could not be annotated. Check res_col and base_annotations.")
}

ct_order <- c(
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

seu_18$cell_type <- factor(seu_18$cell_type, levels = ct_order)
seu_18$cell_type <- droplevels(seu_18$cell_type)
Idents(seu_18) <- "cell_type"
cat("  Annotated cell-type sizes:\n")
print(table(seu_18$cell_type))

# ----------------------------------------------------------
# Normalize and scale marker genes
# ----------------------------------------------------------
if ("RNA" %in% Assays(seu_18)) {
  DefaultAssay(seu_18) <- "RNA"
}
cat(sprintf("  Default assay for marker plots: %s\n", DefaultAssay(seu_18)))

seu_18 <- NormalizeData(seu_18, verbose = TRUE)

markers_for_heatmap <- c(
  # Epithelial_basal
  "COL17A1", "CDH3", "LAMA3", "TENM2", "GNGT1",
  # Epithelial_squamous
  "KRT16", "KRT6B", "KRT17", "KRT14", "KRT5", "S100A2", "SFN", "DAPL1",
  # Epithelial_differentiated
  "PSCA", "CEACAM5", "LCN2", "CEACAM6", "SPRR3", "SCEL", "TMPRSS2",
  # Epithelial_stem_like
  "RMEL3", "ROCR", "COL9A1", "ECE2", "HS3ST4",
  # CAF
  "MMP11", "COL11A1", "POSTN", "COL1A1", "LRRC15", "ITGA11", "AEBP1", "CREB3L1",
  # Myofibroblast
  "KCND2", "THY1", "GPC6", "SGCD", "ADAMTS12", "COL5A3",
  # Fibroblast_normal
  "PI16", "CHRDL1", "PLA2G2A", "C7", "CFD", "IGF1", "ELN", "MGP", "PTGIS",
  # Pericyte_SMC
  "RGS5", "MYH11", "CDH6", "TRPC6", "PPP1R14A", "SYNPO2", "LMOD1",
  # Endothelial
  "CDH5", "KDR", "PLVAP", "VWF", "CLDN5", "TEK", "CLEC14A", "TIE1",
  # CD8_T_effector
  "CD8A", "GZMK", "NKG7", "PRF1", "GZMB", "CRTAM", "KLRK1", "CCL5", "IFNG", "GNLY",
  # Treg
  "FOXP3", "IL2RA", "TNFRSF4", "RTKN2", "BATF", "CCR6", "CD28", "TNFRSF18",
  # B_cell
  "MS4A1", "PAX5", "FCRL1", "BANK1", "BLK", "CD79A", "CD22",
  # Plasma
  "JCHAIN", "MZB1", "IGHG1", "IGHG3", "IGKC", "IGLC2", "FCRL5",
  # Macrophage_SPP1
  "CD163", "SPP1", "C1QC", "C1QA", "C1QB", "APOE", "MSR1", "MRC1", "CD14", "VSIG4",
  # DC
  "FLT3", "LAMP3", "IDO1", "CD86", "CSF2RA", "S100B",
  # Neutrophil
  "CXCR2", "FCGR3B", "CSF3R", "AQP9", "G0S2", "S100A12", "HCAR3", "OSM",
  # Mast_cell
  "TPSB2", "CPA3", "TPSAB1", "MS4A2", "KIT", "GATA2", "HDC", "HPGDS",
  # Proliferating
  "MKI67", "RRM2", "ASPM", "KIF18B", "DLGAP5", "CDCA8", "KIF15"
)

markers_present <- unique(markers_for_heatmap[markers_for_heatmap %in% rownames(seu_18)])
markers_missing <- setdiff(unique(markers_for_heatmap), markers_present)
cat(sprintf("  Markers for heatmap: %d / %d present\n",
            length(markers_present), length(unique(markers_for_heatmap))))
if (length(markers_missing) > 0) {
  cat("  Missing heatmap markers:", paste(markers_missing, collapse = ", "), "\n")
}
if (length(markers_present) == 0) {
  stop("No heatmap marker genes were found in the Seurat object.")
}

seu_18 <- ScaleData(seu_18, features = markers_present, verbose = TRUE)

# ----------------------------------------------------------
# Downsample for clean visualization
# ----------------------------------------------------------
set.seed(42)
cells_per_type <- 300
cells_by_type <- split(colnames(seu_18), seu_18$cell_type, drop = TRUE)
ct_with_cells <- ct_order[ct_order %in% names(cells_by_type)]

cells_down <- unlist(lapply(ct_with_cells, function(ct) {
  x <- cells_by_type[[ct]]
  sample(x, size = min(length(x), cells_per_type), replace = FALSE)
}), use.names = FALSE)

if (length(cells_down) == 0) {
  stop("Downsampling produced zero cells.")
}

seu_down <- subset(seu_18, cells = cells_down)
seu_down$cell_type <- factor(as.character(seu_down$cell_type), levels = ct_with_cells)
seu_down$cell_type <- droplevels(seu_down$cell_type)
Idents(seu_down) <- "cell_type"

cat(sprintf("  Downsampled to %d cells for heatmap across %d cell types\n",
            ncol(seu_down), length(levels(seu_down$cell_type))))
print(table(seu_down$cell_type))

# ----------------------------------------------------------
# Generate marker heatmap
# ----------------------------------------------------------
group_colors <- c(
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

group_colors <- group_colors[levels(seu_down$cell_type)]

p_heatmap <- DoHeatmap(
  seu_down,
  features = markers_present,
  group.by = "cell_type",
  group.colors = group_colors,
  size = 3,
  angle = 40,
  hjust = 1,
  draw.lines = TRUE,
  lines.width = 1
) +
  scale_fill_gradientn(
    colors = c("#2166AC", "#4393C3", "#AED6F1", "#D6604D", "#B2182B"),
    na.value = "grey90",
    name = "Scaled\nExpression",
    # breaks = c(0, 1, 2),
    # limits = c(-0.5, 2.5)
  ) +
  theme(
    axis.text.y = element_text(size = 5, face = "italic"),
    legend.position = "right",
    # plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
    axis.text.x = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    
    # *** KEY FIX: Add top margin to prevent label overlap ***
    plot.margin = margin(t = 30, r = 8, b = 5, l = 5, unit = "mm"),
    
    # Legend (compact, Nature style)
    # legend.position = "right",
    # legend.title = element_text(size = 7, face = "plain", family = "Helvetica"),
    # legend.text = element_text(size = 6, family = "Helvetica"),
    # legend.key.height = unit(0.7, "cm"),
    # legend.key.width = unit(0.25, "cm"),
    # legend.margin = margin(l = 2),
    
    # No title (Nature uses figure captions)
    # plot.title = element_blank(),
    
    # Clean background
    # panel.background = element_blank()
  ) +
  ggtitle("Top Marker Genes by Cell Type")

# p_heatmap <- DoHeatmap(
#   seu_down,
#   features = markers_present,
#   group.by = "cell_type",
#   group.colors = group_colors,
#   size = 2.5,              # Smaller gene label text
#   angle = 40,              # *** KEY FIX: angle the cluster names ***
#   hjust = 0,               # *** KEY FIX: adjust horizontal justification ***
#   draw.lines = TRUE,
#   lines.width = 0.4,
#   group.bar.height = 0.015,  # Thin color bar (Nature style)
#   combine = TRUE,
#   label = TRUE
# ) +
#   # Nature-style color scale (blue-white-red, RdBu-like)
#   scale_fill_gradientn(
#     colors = c("#2166AC", "#4393C3", "#F7F7F7", "#D6604D", "#B2182B"),
#     na.value = "grey90",
#     name = "Scaled\nexpression",
#     breaks = c(0, 1, 2),
#     limits = c(-0.5, 2.5)
#   ) +
#   theme(
#     # Y-axis: gene names
#     axis.text.y = element_text(
#       size = 4, 
#       face = "italic",
#       family = "Helvetica",
#       color = "black",
#       margin = margin(r = 1)
#     ),
#     axis.text.x = element_blank(),
#     axis.ticks = element_blank(),
#     axis.title = element_blank(),
#     
#     # *** KEY FIX: Add top margin to prevent label overlap ***
#     plot.margin = margin(t = 30, r = 8, b = 5, l = 5, unit = "mm"),
#     
#     # Legend (compact, Nature style)
#     legend.position = "right",
#     legend.title = element_text(size = 7, face = "plain", family = "Helvetica"),
#     legend.text = element_text(size = 6, family = "Helvetica"),
#     legend.key.height = unit(0.7, "cm"),
#     legend.key.width = unit(0.25, "cm"),
#     legend.margin = margin(l = 2),
#     
#     # No title (Nature uses figure captions)
#     plot.title = element_blank(),
#     
#     # Clean background
#     panel.background = element_blank()
#   )

ggsave(file.path(output_dir, "Heatmap_markers_DoHeatmap.pdf"), p_heatmap,
       width = 20, height = 14)
ggsave(file.path(output_dir, "Heatmap_markers_DoHeatmap.png"), p_heatmap,
       width = 20, height = 14, dpi = 300)

cat("  Heatmap saved\n")


# ----------------------------------------------------------
# Validation dot plot
# ----------------------------------------------------------
cat("\n  Generating validation dot plot...\n")
DefaultAssay(seu_18) <- DefaultAssay(seu_down)
Idents(seu_18) <- "cell_type"

# Confirm that normalized data are available. NormalizeData above should create this layer/slot.
test_data <- get_assay_data_safe(seu_18, assay = DefaultAssay(seu_18), layer = "data")
if (nrow(test_data) == 0 || ncol(test_data) == 0) {
  stop("Normalized data layer is empty for assay: ", DefaultAssay(seu_18))
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

validation_markers_present <- unique(validation_markers[validation_markers %in% rownames(seu_18)])
validation_markers_missing <- setdiff(unique(validation_markers), validation_markers_present)
cat(sprintf("  %d / %d validation markers found in dataset\n",
            length(validation_markers_present), length(unique(validation_markers))))
if (length(validation_markers_missing) > 0) {
  cat("  Missing validation markers:", paste(validation_markers_missing, collapse = ", "), "\n")
}
if (length(validation_markers_present) == 0) {
  stop("No validation marker genes were found in the Seurat object.")
}

p_dot <- DotPlot(seu_18, features = validation_markers_present, dot.scale = 4) +
  RotatedAxis() +
  theme(
    axis.text.x = element_text(size = 6, face = "italic"),
    axis.text.y = element_text(size = 7),
    plot.title = element_text(face = "bold", hjust = 0.5)
  ) +
  ggtitle("Marker Validation by Cell Type")

ggsave(file.path(output_dir, "DotPlot_validation_markers.pdf"), p_dot,
       width = 18, height = 7)
ggsave(file.path(output_dir, "DotPlot_validation_markers.png"), p_dot,
       width = 18, height = 7, dpi = 300)

cat("  Dot plot saved\n")
cat("\nDone.\n")
