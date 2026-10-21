# ============================================================
# 03_CellType_Annotation.R — REVISED
# Fixes: UMAP labels in legend, boxplot p-symbols, heatmap x-axis
# ============================================================

options(Seurat.object.assay.version = "v3")

library(Seurat)
library(tidyverse)
library(patchwork)
library(ComplexHeatmap)
library(circlize)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/02_Integration"
output_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
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
# PART 1: Load data
# ----------------------------------------------------------
cat("━━━ PART 1: Loading data ━━━\n")
seu <- readRDS(file.path(input_dir, "seu_integrated.rds"))
Idents(seu) <- "RNA_snn_res.0.5"
cat("Clusters:", length(unique(Idents(seu))), "\n")
cat("Cells:", ncol(seu), "\n")

# ----------------------------------------------------------
# PART 2: Find cluster markers
# ----------------------------------------------------------
cat("\n━━━ PART 2: Finding cluster markers ━━━\n")

marker_file <- file.path(output_dir, "Table_all_cluster_markers.csv")
if (file.exists(marker_file)) {
  all_markers <- read.csv(marker_file)
  cat("Loaded existing markers\n")
} else {
  all_markers <- FindAllMarkers(seu, only.pos = TRUE, min.pct = 0.25, 
                                logfc.threshold = 0.5, verbose = TRUE)
  write.csv(all_markers, marker_file, row.names = FALSE)
}

top5 <- all_markers %>%
  group_by(cluster) %>%
  slice_max(n = 5, order_by = avg_log2FC, with_ties = FALSE) %>%
  ungroup()

# ----------------------------------------------------------
# PART 3: Canonical marker visualization
# ----------------------------------------------------------
cat("\n━━━ PART 3: Marker visualization ━━━\n")

key_markers <- c(
  "CD8A", "GZMK", "GZMB", "PRF1", "NKG7", "CRTAM", "CXCL13",
  "FOXP3", "IL2RA", "TNFRSF4", "RTKN2",
  "MKI67", "TOP2A", "RRM2", "ASPM",
  "CD163", "SPP1", "C1QA", "C1QC", "VSIG4", "APOE",
  "CXCR2", "FCGR3B", "CSF3R", "AQP9", "G0S2",
  "FLT3", "LAMP3", "IDO1", "S100B",
  "TPSB2", "CPA3", "MS4A2", "GATA2", "KIT",
  "MS4A1", "PAX5", "BANK1", "BLK", "CD79A",
  "JCHAIN", "MZB1", "IGHG1", "IGKC",
  "MMP11", "COL11A1", "POSTN", "AEBP1",
  "PI16", "CHRDL1", "C7", "CFD",
  "GUCY1A1", "WNT5A", "GPC6", "SGCD", "PRKG1",
  "IGFL2", "CTHRC1", "EMILIN1", "IGFBP5", "FNDC1",
  "RGS5", "MYH11", "CDH6", "TRPC6",
  "CDH5", "PLVAP", "VWF", "CLDN5",
  "COL17A1", "GRHL2", "CDH3", "ITGB4",
  "KRT16", "KRT17", "KRT6B", "S100A2", "KRT5",
  "PSCA", "LCN2", "CEACAM5", "CEACAM6", "TMPRSS2",
  "LGR6", "EDAR", "EPHA7", "COL9A1",
  "HP", "UPK3B", "PRG4", "MSLN", "TFPI2",
  "MIA", "MPZ", "MAGEA4", "SFRP1"
)
key_markers <- unique(key_markers[key_markers %in% rownames(seu)])

p_dot <- DotPlot(seu, features = key_markers, group.by = "RNA_snn_res.0.5") +
  RotatedAxis() +
  theme_paper() +
  theme(axis.text.x = element_text(size = 4.5, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size = 6)) +
  scale_color_viridis_c() +
  labs(title = "Canonical Markers by Cluster")

ggsave(file.path(output_dir, "Fig1_DotPlot_markers.pdf"), p_dot, 
       width = 400, height = 140, units = "mm")

# ----------------------------------------------------------
# PART 4: Feature plots
# ----------------------------------------------------------
cat("\n━━━ PART 4: Feature plots ━━━\n")

lineage_feat <- c("CD8A", "FOXP3", "CD163", "SPP1", "CXCR2", "MS4A1", 
                  "JCHAIN", "COL1A1", "VWF", "RGS5", "KRT17", "MKI67",
                  "MSLN", "MIA", "IGFL2", "TPSB2")
lineage_feat <- lineage_feat[lineage_feat %in% rownames(seu)]

p_feat <- FeaturePlot(seu, features = lineage_feat, ncol = 4, pt.size = 0.01, 
                      order = TRUE, raster = TRUE) &
  theme_paper() & scale_color_viridis_c() & theme(plot.title = element_text(size = 7))

ggsave(file.path(output_dir, "Fig2_FeaturePlot_lineage.pdf"), p_feat, 
       width = 280, height = 280, units = "mm")

# ----------------------------------------------------------
# PART 5: Heatmap (top5 markers)
# ----------------------------------------------------------
cat("\n━━━ PART 5: Marker heatmap ━━━\n")

genes_to_scale <- unique(top5$gene[top5$gene %in% rownames(seu)])
seu <- ScaleData(seu, features = genes_to_scale, verbose = FALSE)

p_heatmap <- DoHeatmap(subset(seu, downsample = 200), features = top5$gene, 
                       group.by = "RNA_snn_res.0.5", size = 2, angle = 0) +
  theme(axis.text.y = element_text(size = 3)) + NoLegend()

ggsave(file.path(output_dir, "Fig4_Heatmap_top5_markers.pdf"), p_heatmap, 
       width = 300, height = 350, units = "mm")

# ----------------------------------------------------------
# PART 6: ANNOTATION
# ----------------------------------------------------------
cat("\n━━━ PART 6: Cell type annotation ━━━\n")

cluster_annotation <- c(
  "0"  = "CD8_T_effector",
  "1"  = "CAF",
  "2"  = "Macrophage_SPP1",
  "3"  = "Treg",
  "4"  = "Epithelial_basal",
  "5"  = "Epithelial_squamous",
  "6"  = "Neutrophil",
  "7"  = "Fibroblast_normal",
  "8"  = "B_cell",
  "9"  = "Proliferating",
  "10" = "Pericyte_SMC",
  "11" = "Plasma",
  "12" = "Endothelial",
  "13" = "Myofibroblast",
  "14" = "DC_mature",
  "15" = "Epithelial_differentiated",
  "16" = "Epithelial_stem_like",
  "17" = "Mast_cell",
  "18" = "Mesothelial",
  "19" = "Tumor_specific",
  "20" = "Activated_fibroblast"
)

clusters <- as.character(seu@meta.data[["RNA_snn_res.0.5"]])
seu@meta.data[["cell_type"]] <- cluster_annotation[clusters]
stopifnot(!any(is.na(seu@meta.data[["cell_type"]])))

Idents(seu) <- "cell_type"

lineage_map <- c(
  "CD8_T_effector" = "T_NK", "Treg" = "T_NK",
  "Proliferating" = "Proliferating",
  "Macrophage_SPP1" = "Myeloid", "Neutrophil" = "Myeloid",
  "DC_mature" = "Myeloid", "Mast_cell" = "Myeloid",
  "B_cell" = "B_Plasma", "Plasma" = "B_Plasma",
  "CAF" = "Stromal", "Fibroblast_normal" = "Stromal",
  "Myofibroblast" = "Stromal", "Activated_fibroblast" = "Stromal",
  "Pericyte_SMC" = "Stromal", "Endothelial" = "Stromal",
  "Mesothelial" = "Stromal",
  "Epithelial_basal" = "Epithelial", "Epithelial_squamous" = "Epithelial",
  "Epithelial_differentiated" = "Epithelial",
  "Epithelial_stem_like" = "Epithelial", "Tumor_specific" = "Epithelial"
)

seu@meta.data[["major_lineage"]] <- lineage_map[seu@meta.data[["cell_type"]]]
stopifnot(!any(is.na(seu@meta.data[["major_lineage"]])))

cat("Cell type distribution:\n")
print(sort(table(seu$cell_type), decreasing = TRUE))

# ----------------------------------------------------------
# PART 7: UMAP — Labels in LEGEND (not on plot)
# ----------------------------------------------------------
cat("\n━━━ PART 7: Annotated UMAP ━━━\n")

celltype_colors <- c(
  "CD8_T_effector" = "#1F77B4", "Treg" = "#AEC7E8",
  "Proliferating" = "#FF7F0E",
  "Macrophage_SPP1" = "#2CA02C", "Neutrophil" = "#98DF8A",
  "DC_mature" = "#D62728", "Mast_cell" = "#FF9896",
  "B_cell" = "#9467BD", "Plasma" = "#C5B0D5",
  "CAF" = "#8C564B", "Fibroblast_normal" = "#C49C94",
  "Myofibroblast" = "#E377C2", "Activated_fibroblast" = "#DE77AE",
  "Pericyte_SMC" = "#F7B6D2", "Endothelial" = "#7F7F7F",
  "Mesothelial" = "#C7C7C7",
  "Epithelial_basal" = "#BCBD22", "Epithelial_squamous" = "#DBDB8D",
  "Epithelial_differentiated" = "#17BECF", "Epithelial_stem_like" = "#9EDAE5",
  "Tumor_specific" = "#393B79"
)

# UMAP cell type — labels in legend
p_ct <- DimPlot(seu, group.by = "cell_type", label = FALSE, pt.size = 0.05,
                cols = celltype_colors) +
  theme_paper(base_size = 8) + ggtitle("Cell Type Annotation") +
  theme(legend.position = "right", legend.text = element_text(size = 6),
        legend.title = element_blank(), legend.key.size = unit(0.3, "cm")) +
  guides(color = guide_legend(override.aes = list(size = 2.5), ncol = 1))

ggsave(file.path(output_dir, "Fig5_UMAP_celltype.pdf"), p_ct, 
       width = 200, height = 130, units = "mm")

# UMAP clusters — labels in legend
p_cluster <- DimPlot(seu, group.by = "RNA_snn_res.0.5", label = FALSE, pt.size = 0.05) +
  theme_paper(base_size = 8) + ggtitle("Clusters (res 0.5)") +
  theme(legend.position = "right", legend.text = element_text(size = 6),
        legend.title = element_blank(), legend.key.size = unit(0.25, "cm")) +
  guides(color = guide_legend(override.aes = list(size = 2.5), ncol = 2))

ggsave(file.path(output_dir, "Fig5b_UMAP_clusters.pdf"), p_cluster, 
       width = 170, height = 110, units = "mm")

# UMAP split by group
p_ct_split <- DimPlot(seu, group.by = "cell_type", split.by = "group",
                      label = FALSE, pt.size = 0.05, cols = celltype_colors) +
  theme_paper(base_size = 7) + ggtitle("Cell Types by Risk Group") +
  theme(legend.position = "right", legend.text = element_text(size = 5),
        legend.title = element_blank(), legend.key.size = unit(0.2, "cm")) +
  guides(color = guide_legend(override.aes = list(size = 2), ncol = 1))

ggsave(file.path(output_dir, "Fig6_UMAP_celltype_by_group.pdf"), p_ct_split, 
       width = 320, height = 120, units = "mm")

# Major lineage UMAP
lineage_colors <- c("Epithelial" = "#E41A1C", "T_NK" = "#377EB8",
                    "Myeloid" = "#FF7F00", "B_Plasma" = "#4DAF4A",
                    "Stromal" = "#984EA3", "Proliferating" = "#A65628")

p_lineage <- DimPlot(seu, group.by = "major_lineage", cols = lineage_colors,
                     pt.size = 0.05, label = FALSE) +
  theme_paper(base_size = 8) + ggtitle("Major Lineages") +
  theme(legend.position = "right", legend.text = element_text(size = 7),
        legend.title = element_blank()) +
  guides(color = guide_legend(override.aes = list(size = 3)))

ggsave(file.path(output_dir, "Fig7_UMAP_lineage.pdf"), p_lineage, 
       width = 170, height = 110, units = "mm")

# Combined panel
p_combined_umap <- (p_cluster | p_ct | p_lineage) + plot_layout(widths = c(1, 1.3, 1))
ggsave(file.path(output_dir, "Fig_UMAP_panel_combined.pdf"), p_combined_umap,
       width = 450, height = 120, units = "mm")

# ----------------------------------------------------------
# PART 8: Proportion with p-value symbols
# ----------------------------------------------------------
cat("\n━━━ PART 8: Proportion analysis ━━━\n")

prop_df <- seu@meta.data %>%
  group_by(sample, group, cell_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(sample) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

# Fill missing with 0
sample_group_map <- seu@meta.data %>% dplyr::select(sample, group) %>% distinct()
all_cell_types <- unique(seu$cell_type)

all_comb <- expand.grid(sample = unique(sample_group_map$sample),
                        cell_type = all_cell_types, stringsAsFactors = FALSE) %>%
  left_join(sample_group_map, by = "sample") %>%
  left_join(prop_df %>% dplyr::select(sample, cell_type, proportion),
            by = c("sample", "cell_type")) %>%
  mutate(proportion = replace_na(proportion, 0))

# Stacked barplot
p_stack <- ggplot(prop_df, aes(x = sample, y = proportion, fill = cell_type)) +
  geom_bar(stat = "identity") +
  facet_wrap(~group, scales = "free_x") +
  scale_fill_manual(values = celltype_colors) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        legend.text = element_text(size = 5), legend.key.size = unit(0.2, "cm")) +
  labs(x = "", y = "Proportion", fill = "Cell Type", title = "Cell Type Composition")

ggsave(file.path(output_dir, "Fig8_stacked_barplot.pdf"), p_stack, 
       width = 200, height = 120, units = "mm")

# Manual Wilcoxon per cell type
prop_stats <- data.frame(cell_type = character(), mean_Low = numeric(),
                         mean_High = numeric(), p_value = numeric(),
                         stringsAsFactors = FALSE)

for (ct in all_cell_types) {
  ct_data <- all_comb %>% filter(cell_type == ct)
  high_vals <- ct_data %>% filter(group == "High_Risk") %>% pull(proportion)
  low_vals <- ct_data %>% filter(group == "Low_Risk") %>% pull(proportion)
  
  if (sum(high_vals > 0) >= 2 & sum(low_vals > 0) >= 2) {
    p_val <- tryCatch(
      wilcox.test(high_vals, low_vals, exact = FALSE)$p.value,
      error = function(e) NA
    )
  } else {
    p_val <- NA
  }
  
  prop_stats <- rbind(prop_stats, data.frame(
    cell_type = ct,
    mean_Low = mean(low_vals),
    mean_High = mean(high_vals),
    p_value = p_val,
    stringsAsFactors = FALSE
  ))
}

prop_stats$p_adj <- p.adjust(prop_stats$p_value, method = "BH")
prop_stats$significance <- case_when(
  is.na(prop_stats$p_value) ~ "nt",
  prop_stats$p_adj < 0.001 ~ "***",
  prop_stats$p_adj < 0.01 ~ "**",
  prop_stats$p_adj < 0.05 ~ "*",
  prop_stats$p_adj < 0.1 ~ ".",
  TRUE ~ "ns"
)
prop_stats <- prop_stats %>% arrange(p_value)

write.csv(prop_stats, file.path(output_dir, "Table_proportion_statistics.csv"), row.names = FALSE)
cat("\nProportion statistics:\n")
print(prop_stats)

# ─── Boxplot with p-value symbols ───
ct_order <- all_comb %>%
  group_by(cell_type) %>%
  summarise(mean_prop = mean(proportion), .groups = "drop") %>%
  arrange(desc(mean_prop)) %>%
  pull(cell_type)

all_comb$cell_type <- factor(all_comb$cell_type, levels = ct_order)

# Annotation for p-value symbols
annotation_df <- all_comb %>%
  group_by(cell_type) %>%
  summarise(y_max = max(proportion, na.rm = TRUE) * 1.1, .groups = "drop") %>%
  left_join(prop_stats %>% dplyr::select(cell_type, significance), by = "cell_type")

annotation_df$cell_type <- factor(annotation_df$cell_type, levels = ct_order)

p_box <- ggplot(all_comb, aes(x = cell_type, y = proportion, fill = group)) +
  geom_boxplot(outlier.size = 0.3, linewidth = 0.3, width = 0.7,
               position = position_dodge(0.8)) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8),
              size = 0.6, alpha = 0.7, shape = 16) +
  geom_text(data = annotation_df,
            aes(x = cell_type, y = y_max, label = significance),
            inherit.aes = FALSE, size = 3, fontface = "bold", vjust = 0) +
  scale_fill_manual(values = group_colors) +
  theme_paper(base_size = 7) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6)) +
  labs(x = "", y = "Proportion", fill = "Risk Group",
       title = "Cell Type Proportions: Low-Risk vs High-Risk") +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.15)))

ggsave(file.path(output_dir, "Fig9_proportion_boxplot.pdf"), p_box,
       width = 300, height = 110, units = "mm")

# ----------------------------------------------------------
# PART 9: Focus clusters — FIXED (no stat_compare_means)
# ----------------------------------------------------------
cat("\n━━━ PART 9: Focus analysis ━━━\n")

focus_types <- c("Myofibroblast", "Mesothelial", "Tumor_specific", "Activated_fibroblast")

# Highlight on UMAP (labels in legend)
seu$focus_highlight <- ifelse(seu$cell_type %in% focus_types, seu$cell_type, "Other")
seu$focus_highlight <- factor(seu$focus_highlight, levels = c(focus_types, "Other"))

highlight_colors <- c(
  "Myofibroblast" = "#E377C2",
  "Mesothelial" = "#17BECF",
  "Tumor_specific" = "#393B79",
  "Activated_fibroblast" = "#DE77AE",
  "Other" = "grey85"
)

p_highlight <- DimPlot(seu, group.by = "focus_highlight",
                       cols = highlight_colors, pt.size = 0.05,
                       label = FALSE, order = focus_types) +
  theme_paper(base_size = 8) + ggtitle("Focus Clusters") +
  theme(legend.position = "right", legend.text = element_text(size = 7),
        legend.title = element_blank()) +
  guides(color = guide_legend(override.aes = list(size = 3)))

p_highlight_split <- DimPlot(seu, group.by = "focus_highlight",
                             split.by = "group",
                             cols = highlight_colors, pt.size = 0.05,
                             label = FALSE, order = focus_types) +
  theme_paper(base_size = 7) + ggtitle("Focus Clusters by Risk Group") +
  theme(legend.position = "right", legend.text = element_text(size = 6),
        legend.title = element_blank())

ggsave(file.path(output_dir, "Fig10_focus_clusters_highlight.pdf"),
       p_highlight / p_highlight_split, width = 240, height = 220, units = "mm")

# ─── Focus boxplots with manual p-value symbols ───
focus_prop_complete <- all_comb %>% filter(cell_type %in% focus_types)
focus_prop_complete$cell_type <- factor(focus_prop_complete$cell_type, levels = focus_types)

# Get p-values from prop_stats
focus_pvals <- prop_stats %>%
  filter(cell_type %in% focus_types) %>%
  dplyr::select(cell_type, significance)

focus_y_pos <- focus_prop_complete %>%
  group_by(cell_type) %>%
  summarise(y_max = max(proportion, na.rm = TRUE) * 1.2, .groups = "drop") %>%
  left_join(focus_pvals, by = "cell_type")

# Handle case where y_max is 0 (all zeros)
focus_y_pos$y_max[focus_y_pos$y_max == 0] <- 0.01
focus_y_pos$significance[is.na(focus_y_pos$significance)] <- "nt"

p_focus_box <- ggplot(focus_prop_complete, aes(x = cell_type, y = proportion, fill = group)) +
  geom_boxplot(outlier.size = 0.5, linewidth = 0.4, width = 0.6,
               position = position_dodge(0.75)) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.12, dodge.width = 0.75),
              size = 1.5, alpha = 0.8, shape = 16) +
  scale_fill_manual(values = group_colors) +
  geom_text(data = focus_y_pos,
            aes(x = cell_type, y = y_max, label = significance),
            inherit.aes = FALSE, size = 4.5, fontface = "bold", vjust = 0) +
  geom_segment(data = focus_y_pos,
               aes(x = as.numeric(cell_type) - 0.2,
                   xend = as.numeric(cell_type) + 0.2,
                   y = y_max * 0.9, yend = y_max * 0.9),
               inherit.aes = FALSE, linewidth = 0.3) +
  theme_paper(base_size = 8) +
  labs(x = "", y = "Proportion of total cells", fill = "Risk Group",
       title = "Focus Clusters: Proportion by Risk Group") +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.18))) +
  theme(axis.text.x = element_text(size = 7, angle = 20, hjust = 1))

ggsave(file.path(output_dir, "Fig11_focus_clusters_proportion.pdf"),
       p_focus_box, width = 150, height = 100, units = "mm")

# ─── Violin plots for focus cluster markers ───
# Cluster 13 - Myofibroblast (verified from uploaded table)
myofib_markers <- c("GUCY1A1", "POSTN", "SGCD", "WNT5A", "GPC6",
                    "COL5A3", "PRKG1", "PDZRN3", "NAV3", "ROBO2")
myofib_markers <- myofib_markers[myofib_markers %in% rownames(seu)]

if (sum(seu$cell_type == "Myofibroblast") > 10) {
  p_vln_myofib <- VlnPlot(subset(seu, cell_type == "Myofibroblast"),
                          features = myofib_markers,
                          group.by = "group", cols = group_colors,
                          pt.size = 0, ncol = 5) &
    theme_paper()
  ggsave(file.path(output_dir, "Fig12_VlnPlot_Myofibroblast_markers.pdf"),
         p_vln_myofib, width = 300, height = 120, units = "mm")
}

# Cluster 18 - Mesothelial (verified from uploaded table)
meso_markers <- c("HP", "UPK3B", "PRG4", "MSLN", "ALOX15",
                  "TFPI2", "BCHE", "HAS1", "VTN", "CFB")
meso_markers <- meso_markers[meso_markers %in% rownames(seu)]

if (sum(seu$cell_type == "Mesothelial") > 10) {
  p_vln_meso <- VlnPlot(subset(seu, cell_type == "Mesothelial"),
                        features = meso_markers,
                        group.by = "group", cols = group_colors,
                        pt.size = 0, ncol = 5) &
    theme_paper()
  ggsave(file.path(output_dir, "Fig13_VlnPlot_Mesothelial_markers.pdf"),
         p_vln_meso, width = 300, height = 120, units = "mm")
}

# Cluster 19 - Tumor_specific (verified from uploaded table)
tumor_markers <- c("MIA", "MPZ", "MAGEA4", "SFRP1", "DAPL1",
                   "SAA1", "RPL35A", "ACTG2", "TSTD1", "BEX2")
tumor_markers <- tumor_markers[tumor_markers %in% rownames(seu)]

if (sum(seu$cell_type == "Tumor_specific") > 10) {
  p_vln_tumor <- VlnPlot(subset(seu, cell_type == "Tumor_specific"),
                         features = tumor_markers,
                         group.by = "group", cols = group_colors,
                         pt.size = 0, ncol = 5) &
    theme_paper()
  ggsave(file.path(output_dir, "Fig14_VlnPlot_Tumor_specific_markers.pdf"),
         p_vln_tumor, width = 300, height = 120, units = "mm")
}

# Cluster 20 - Activated fibroblast (verified from uploaded table)
actfib_markers <- c("IGFL2", "CYP1B1", "IGFBP2", "ACTA2", "CTHRC1",
                    "EMILIN1", "FNDC1", "IGFBP5", "STEAP1", "APOD")
actfib_markers <- actfib_markers[actfib_markers %in% rownames(seu)]

if (sum(seu$cell_type == "Activated_fibroblast") > 10) {
  p_vln_actfib <- VlnPlot(subset(seu, cell_type == "Activated_fibroblast"),
                          features = actfib_markers,
                          group.by = "group", cols = group_colors,
                          pt.size = 0, ncol = 5) &
    theme_paper()
  ggsave(file.path(output_dir, "Fig14b_VlnPlot_ActivatedFibroblast_markers.pdf"),
         p_vln_actfib, width = 300, height = 120, units = "mm")
}

# Sample distribution for focus clusters
focus_sample_dist <- seu@meta.data %>%
  filter(cell_type %in% focus_types) %>%
  group_by(cell_type, sample, group) %>%
  summarise(n = n(), .groups = "drop")

p_focus_sample <- ggplot(focus_sample_dist, aes(x = sample, y = n, fill = group)) +
  geom_bar(stat = "identity") +
  facet_wrap(~cell_type, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = group_colors) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5)) +
  labs(x = "", y = "Cell count", fill = "Risk Group",
       title = "Focus Clusters: Cell Distribution by Sample")

ggsave(file.path(output_dir, "Fig15_focus_clusters_sample_distribution.pdf"),
       p_focus_sample, width = 150, height = 220, units = "mm")

# ----------------------------------------------------------
# PART 10: Heatmap — FIXED x-axis labels
# ----------------------------------------------------------
cat("\n━━━ PART 10: Annotation verification heatmap ━━━\n")

verification_markers <- c(
  "CRTAM", "GZMK", "CD8A", "NKG7",
  "MMP11", "COL11A1", "POSTN", "AEBP1",
  "CD163", "VSIG4", "SPP1", "C1QC",
  "FOXP3", "TNFRSF4", "IL2RA", "RTKN2",
  "COL17A1", "GRHL2", "CDH3", "ITGB4",
  "KRT16", "KRT17", "KRT6B", "S100A2",
  "CXCR2", "FCGR3B", "CSF3R", "AQP9",
  "PI16", "CHRDL1", "C7", "CFD",
  "MS4A1", "PAX5", "BANK1", "BLK",
  "MKI67", "RRM2", "ASPM", "TOP2A",
  "RGS5", "CDH6", "MYH11", "TRPC6",
  "IGKC", "IGHG1", "JCHAIN", "MZB1",
  "CDH5", "PLVAP", "VWF", "CLDN5",
  "GUCY1A1", "SGCD", "WNT5A", "GPC6",
  "FLT3", "S100B", "LAMP3", "IDO1",
  "PSCA", "LCN2", "CEACAM5", "CEACAM6",
  "LGR6", "EDAR", "EPHA7", "COL9A1",
  "TPSB2", "MS4A2", "CPA3", "GATA2",
  "HP", "UPK3B", "PRG4", "MSLN",
  "MIA", "MPZ", "MAGEA4", "SFRP1",
  "IGFL2", "CTHRC1", "EMILIN1", "IGFBP5"
)

verification_markers <- unique(verification_markers[verification_markers %in% rownames(seu)])

Idents(seu) <- "cell_type"
avg_exp <- AverageExpression(seu, features = verification_markers, return.seurat = FALSE)
avg_mat <- as.matrix(avg_exp$RNA)
write.csv(avg_mat, file.path(output_dir, "Table_avg_expression_key_markers.csv"))

# Scale
avg_mat_scaled <- t(scale(t(avg_mat)))
avg_mat_scaled[avg_mat_scaled > 2.5] <- 2.5
avg_mat_scaled[avg_mat_scaled < -2.5] <- -2.5
avg_mat_scaled <- avg_mat_scaled[complete.cases(avg_mat_scaled), ]

# Shorten column names to prevent truncation
col_names_display <- colnames(avg_mat_scaled)
col_names_display <- gsub("Epithelial_differentiated", "Epi_diff", col_names_display)
col_names_display <- gsub("Epithelial_squamous", "Epi_squam", col_names_display)
col_names_display <- gsub("Epithelial_basal", "Epi_basal", col_names_display)
col_names_display <- gsub("Epithelial_stem_like", "Epi_stem", col_names_display)
col_names_display <- gsub("Activated_fibroblast", "Act_fibro", col_names_display)
col_names_display <- gsub("Fibroblast_normal", "Fibro_norm", col_names_display)
col_names_display <- gsub("Macrophage_SPP1", "Mac_SPP1", col_names_display)
col_names_display <- gsub("CD8_T_effector", "CD8_Teff", col_names_display)
col_names_display <- gsub("Tumor_specific", "Tumor_spec", col_names_display)
col_names_display <- gsub("Pericyte_SMC", "Pericyte", col_names_display)
col_names_display <- gsub("Myofibroblast", "Myofibro", col_names_display)
colnames(avg_mat_scaled) <- col_names_display

# Row grouping
n_markers_present <- nrow(avg_mat_scaled)
group_labels <- c(
  "CD8_Teff", "CAF", "Mac_SPP1", "Treg",
  "Epi_basal", "Epi_squam", "Neutrophil", "Fibro_norm",
  "B_cell", "Prolif", "Pericyte", "Plasma",
  "Endothelial", "Myofibro", "DC_mature", "Epi_diff",
  "Epi_stem", "Mast_cell", "Mesothelial", "Tumor_spec", "Act_fibro"
)
marker_groups <- rep(group_labels, each = 4)[1:n_markers_present]

# KEY FIX: Larger PDF + proper padding for x-axis labels
pdf(file.path(output_dir, "Fig16_Heatmap_annotation_verification.pdf"),
    width = 13, height = 16)

ht <- Heatmap(avg_mat_scaled,
              name = "Scaled\nExpr",
              col = colorRamp2(c(-2.5, 0, 2.5), c("#2166AC", "white", "#B2182B")),
              cluster_rows = FALSE,
              cluster_columns = TRUE,
              show_row_names = TRUE,
              show_column_names = TRUE,
              row_names_gp = gpar(fontsize = 6),
              column_names_gp = gpar(fontsize = 9, fontface = "bold"),
              column_names_rot = 45,
              column_names_max_height = unit(8, "cm"),
              column_title = "Cell Type Annotation Verification",
              column_title_gp = gpar(fontsize = 11, fontface = "bold"),
              row_split = factor(marker_groups, levels = unique(marker_groups)),
              row_title_gp = gpar(fontsize = 5.5),
              row_title_rot = 0,
              row_gap = unit(0.8, "mm"),
              heatmap_legend_param = list(
                title_gp = gpar(fontsize = 8),
                labels_gp = gpar(fontsize = 7),
                legend_height = unit(3, "cm")
              ))

# KEY FIX: padding bottom=20mm ensures x-axis labels are not cut off
draw(ht, padding = unit(c(5, 5, 20, 5), "mm"))
dev.off()

cat("  Heatmap saved with fixed x-axis labels.\n")
# ----------------------------------------------------------
# PART 11: Lineage proportion analysis
# ----------------------------------------------------------
cat("\n━━━ PART 11: Lineage proportions ━━━\n")

lineage_prop <- seu@meta.data %>%
  group_by(sample, group, major_lineage) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(sample) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

p_lineage_box <- ggplot(lineage_prop, aes(x = major_lineage, y = proportion, fill = group)) +
  geom_boxplot(outlier.size = 0.5, linewidth = 0.3) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.1), size = 1, alpha = 0.7) +
  scale_fill_manual(values = group_colors) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x = "", y = "Proportion", fill = "Risk Group",
       title = "Major Lineage Proportions: Low-Risk vs High-Risk")

ggsave(file.path(output_dir, "Fig17_lineage_proportion.pdf"), p_lineage_box, 
       width = 160, height = 100, units = "mm")

p_lineage_stack <- ggplot(lineage_prop, aes(x = sample, y = proportion, fill = major_lineage)) +
  geom_bar(stat = "identity") +
  facet_wrap(~group, scales = "free_x") +
  scale_fill_manual(values = lineage_colors) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5)) +
  labs(x = "", y = "Proportion", fill = "Lineage",
       title = "Lineage Composition by Sample")

ggsave(file.path(output_dir, "Fig18_lineage_stacked_barplot.pdf"), p_lineage_stack, 
       width = 180, height = 100, units = "mm")

# ----------------------------------------------------------
# PART 12: Save final objects
# ----------------------------------------------------------
cat("\n━━━ PART 12: Saving ━━━\n")

saveRDS(seu, file.path(output_dir, "seu_annotated.rds"))
write.csv(seu@meta.data, file.path(output_dir, "Table_metadata_annotated.csv"))

top20 <- all_markers %>%
  group_by(cluster) %>%
  slice_max(n = 20, order_by = avg_log2FC, with_ties = FALSE) %>%
  ungroup()
top20$cell_type <- cluster_annotation[as.character(top20$cluster)]
write.csv(top20, file.path(output_dir, "Table_top20_markers_annotated.csv"), row.names = FALSE)

# ----------------------------------------------------------
# Final summary
# ----------------------------------------------------------
cat("\n━━━ FINAL SUMMARY ━━━\n")
cat("Total cells:", ncol(seu), "\n")

cat("\nCell type distribution:\n")
ct_table <- sort(table(seu$cell_type), decreasing = TRUE)
for (ct in names(ct_table)) {
  cat(sprintf("  %-30s %6d (%4.1f%%)\n", ct, ct_table[ct], ct_table[ct]/ncol(seu)*100))
}

cat("\nMajor lineage distribution:\n")
lin_table <- sort(table(seu$major_lineage), decreasing = TRUE)
for (lin in names(lin_table)) {
  cat(sprintf("  %-15s %6d (%4.1f%%)\n", lin, lin_table[lin], lin_table[lin]/ncol(seu)*100))
}

cat("\nCells per group:\n")
print(table(seu$group))

cat("\nFocus clusters summary:\n")
focus_summary <- seu@meta.data %>%
  filter(cell_type %in% focus_types) %>%
  group_by(cell_type, group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = group, values_from = n, values_fill = 0)
print(as.data.frame(focus_summary))

cat("\n━━━ 03_CellType_Annotation COMPLETE ━━━\n")
cat("Output directory:", output_dir, "\n")
cat("\nNext step: 04_Differential_Analysis.R\n")

