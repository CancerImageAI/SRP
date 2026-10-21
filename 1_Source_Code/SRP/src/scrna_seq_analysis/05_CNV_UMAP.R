# ============================================================
# UMAP colored by CNV score — Split by Risk Group (FIXED)
# ============================================================

library(Seurat)
library(ggplot2)
library(patchwork)
library(dplyr)      # <-- ADD THIS to ensure %>% is available
library(tidyverse)  # <-- OR this (loads dplyr + magrittr)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/05_Heatmap_CNV"
cnv_dir <- file.path(input_dir, "CNV_analysis")

# ----------------------------------------------------------
# Load Seurat object with CNV scores
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading Seurat object with CNV scores\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_with_cnv_scores.rds"))
cat(sprintf("  Cells: %d | Genes: %d\n", ncol(seu), nrow(seu)))

if (!"cnv_score" %in% colnames(seu@meta.data)) {
  stop("cnv_score not found in metadata. Run CNV analysis first.")
}

cat(sprintf("  Cells with CNV score: %d\n", sum(!is.na(seu@meta.data$cnv_score))))

# ----------------------------------------------------------
# Get common color scale range
# ----------------------------------------------------------
cnv_range <- range(seu@meta.data$cnv_score, na.rm = TRUE)
cat(sprintf("  CNV score range: %.2f – %.2f\n", cnv_range[1], cnv_range[2]))

# ----------------------------------------------------------
# Method 1: FeaturePlot split.by
# ----------------------------------------------------------
cat("\n  Generating split UMAP (Method 1)...\n")

p_split <- FeaturePlot(seu, features = "cnv_score", reduction = "umap",
                       split.by = "group", pt.size = 0.1, order = TRUE) &
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", na.value = "grey90",
                        limits = cnv_range) &
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 11))

ggsave(file.path(cnv_dir, "CNV_UMAP_split_by_group.pdf"), p_split,
       width = 12, height = 6)
cat("  ✓ Method 1 saved.\n")

# ----------------------------------------------------------
# Method 2: Manual split with patchwork (FIXED)
# ----------------------------------------------------------
cat("  Generating split UMAP (Method 2)...\n")

# Get UMAP coordinates
umap_coords <- as.data.frame(Embeddings(seu, "umap"))
colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
umap_coords$group <- seu@meta.data$group
umap_coords$cnv_score <- seu@meta.data$cnv_score
umap_coords$cell_type <- seu@meta.data$cell_type

# Get axis limits
x_lim <- range(umap_coords$UMAP_1) + c(-0.5, 0.5)
y_lim <- range(umap_coords$UMAP_2) + c(-0.5, 0.5)

# *** FIX: Use subset() instead of %>% dplyr::filter() ***
# Or ensure dplyr is loaded (library(dplyr) at the top)

# High-Risk panel
umap_high <- subset(umap_coords, group == "High_Risk")
umap_low <- subset(umap_coords, group == "Low_Risk")

p_high <- ggplot(umap_high, aes(x = UMAP_1, y = UMAP_2, color = cnv_score)) +
  geom_point(size = 0.1, alpha = 0.6) +
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", na.value = "grey90",
                        limits = cnv_range) +
  xlim(x_lim) + ylim(y_lim) +
  theme_classic(base_size = 10) +
  labs(title = "High-Risk", x = "UMAP 1", y = "UMAP 2") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, color = "#E64B35"),
        legend.position = "right")

# Low-Risk panel
p_low <- ggplot(umap_low, aes(x = UMAP_1, y = UMAP_2, color = cnv_score)) +
  geom_point(size = 0.1, alpha = 0.6) +
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", na.value = "grey90",
                        limits = cnv_range) +
  xlim(x_lim) + ylim(y_lim) +
  theme_classic(base_size = 10) +
  labs(title = "Low-Risk", x = "UMAP 1", y = "UMAP 2") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, color = "#4DBBD5"),
        legend.position = "right")

# Combine
p_manual_split <- p_high + p_low +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "CNV Score on UMAP: Split by Risk Group",
    subtitle = "Higher CNV score indicates greater chromosomal instability",
    theme = theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
                  plot.subtitle = element_text(size = 9, hjust = 0.5))
  )

ggsave(file.path(cnv_dir, "CNV_UMAP_split_manual.pdf"), p_manual_split,
       width = 13, height = 6)
cat("  ✓ Method 2 saved.\n")

# ----------------------------------------------------------
# Method 3: Three-panel
# ----------------------------------------------------------
cat("  Generating three-panel UMAP...\n")

p_all <- ggplot(umap_coords, aes(x = UMAP_1, y = UMAP_2, color = cnv_score)) +
  geom_point(size = 0.1, alpha = 0.5) +
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", na.value = "grey90",
                        limits = cnv_range) +
  xlim(x_lim) + ylim(y_lim) +
  theme_classic(base_size = 10) +
  labs(title = "All Cells", x = "UMAP 1", y = "UMAP 2") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "right")

p_three <- p_all + p_high + p_low +
  plot_layout(guides = "collect", ncol = 3) +
  plot_annotation(
    title = "CNV Score on UMAP",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
  )

ggsave(file.path(cnv_dir, "CNV_UMAP_three_panel.pdf"), p_three,
       width = 18, height = 6)
cat("  ✓ Three-panel saved.\n")

# ----------------------------------------------------------
# Method 4: Epithelial cells only
# ----------------------------------------------------------
cat("  Generating epithelial-only split UMAP...\n")

tumor_types <- c("Epithelial_basal", "Epithelial_squamous",
                 "Epithelial_differentiated", "Epithelial_stem_like")

epi_umap <- subset(umap_coords, cell_type %in% tumor_types & !is.na(cnv_score))

cnv_range_epi <- range(epi_umap$cnv_score, na.rm = TRUE)

epi_high <- subset(epi_umap, group == "High_Risk")
epi_low <- subset(epi_umap, group == "Low_Risk")

p_epi_high <- ggplot(epi_high, aes(x = UMAP_1, y = UMAP_2, color = cnv_score)) +
  geom_point(size = 0.3, alpha = 0.7) +
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", limits = cnv_range_epi) +
  xlim(x_lim) + ylim(y_lim) +
  theme_classic(base_size = 10) +
  labs(title = "High-Risk (Epithelial)", x = "UMAP 1", y = "UMAP 2") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, color = "#E64B35"))

p_epi_low <- ggplot(epi_low, aes(x = UMAP_1, y = UMAP_2, color = cnv_score)) +
  geom_point(size = 0.3, alpha = 0.7) +
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", limits = cnv_range_epi) +
  xlim(x_lim) + ylim(y_lim) +
  theme_classic(base_size = 10) +
  labs(title = "Low-Risk (Epithelial)", x = "UMAP 1", y = "UMAP 2") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, color = "#4DBBD5"))

p_epi_split <- p_epi_high + p_epi_low +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "CNV Score: Epithelial Cells by Risk Group",
    subtitle = sprintf("High-Risk: n=%d | Low-Risk: n=%d",
                       nrow(epi_high), nrow(epi_low)),
    theme = theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
                  plot.subtitle = element_text(size = 9, hjust = 0.5))
  )

ggsave(file.path(cnv_dir, "CNV_UMAP_split_epithelial_only.pdf"), p_epi_split,
       width = 13, height = 6)
cat("  ✓ Epithelial-only split saved.\n")

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
cat("\n━━━ Split UMAP Complete ━━━\n")
cat(sprintf("  Output: %s\n", cnv_dir))
# ----------------------------------------------------------
# Method 5: Faceted by cell type + group
# ----------------------------------------------------------
cat("  Generating faceted UMAP (cell_type × group)...\n")

epi_umap$facet_label <- paste0(epi_umap$cell_type, "\n(", epi_umap$group, ")")

p_faceted <- ggplot(epi_umap, aes(x = UMAP_1, y = UMAP_2, color = cnv_score)) +
  geom_point(size = 0.2, alpha = 0.6) +
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", limits = cnv_range_epi) +
  facet_grid(group ~ cell_type) +
  theme_classic(base_size = 8) +
  labs(title = "CNV Score: Epithelial Subtypes × Risk Group",
       x = "UMAP 1", y = "UMAP 2") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
        strip.text = element_text(face = "bold", size = 7))

ggsave(file.path(cnv_dir, "CNV_UMAP_faceted_celltype_group.pdf"), p_faceted,
       width = 14, height = 7)
cat("  ✓ Faceted UMAP saved.\n")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
cat("\n")
cat("╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║  CNV UMAP Split — Complete                                          ║\n")
cat("╠══════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                      ║\n")
cat("║  Output files:                                                       ║\n")
cat("║    CNV_UMAP_split_by_group.pdf         — FeaturePlot split.by        ║\n")
cat("║    CNV_UMAP_split_manual.pdf           — Manual patchwork (2-panel)  ║\n")
cat("║    CNV_UMAP_three_panel.pdf            — All + High + Low            ║\n")
cat("║    CNV_UMAP_split_epithelial_only.pdf  — Epithelial cells only       ║\n")
cat("║    CNV_UMAP_faceted_celltype_group.pdf — Cell type × group facets    ║\n")
cat("║                                                                      ║\n")
cat("╚══════════════════════════════════════════════════════════════════════╝\n")

cat("\n━━━ Done ━━━\n")
