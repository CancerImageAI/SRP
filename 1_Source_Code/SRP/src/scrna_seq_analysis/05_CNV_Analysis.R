# ============================================================
# CNV Analysis — Fixed filter() Conflict
# ============================================================

library(Seurat)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ComplexHeatmap)
library(circlize)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/05_Heatmap_CNV"
cnv_dir <- file.path(output_dir, "CNV_analysis")
dir.create(cnv_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading annotated Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_annotated_3rd.rds"))
cat(sprintf("  Cells: %d | Genes: %d\n", ncol(seu), nrow(seu)))

DefaultAssay(seu) <- "RNA"

available_layers <- Layers(seu[["RNA"]])
count_layers <- grep("^counts", available_layers, value = TRUE)
if (length(count_layers) > 1) {
  cat("  Joining split RNA layers...\n")
  seu[["RNA"]] <- JoinLayers(seu[["RNA"]])
}
if (!"data" %in% Layers(seu[["RNA"]])) {
  seu <- NormalizeData(seu, assay = "RNA")
}
cat("  ✓ RNA assay ready.\n")

# ----------------------------------------------------------
# Get gene positions — FIX: use dplyr::filter explicitly
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Getting gene chromosomal positions\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

library(EnsDb.Hsapiens.v86)

# Get all gene info
all_genes_info <- genes(EnsDb.Hsapiens.v86)
gene_df <- as.data.frame(all_genes_info)

# *** KEY FIX: Use dplyr::filter() explicitly ***
gene_positions <- gene_df %>%
  dplyr::filter(symbol %in% rownames(seu)) %>%
  dplyr::filter(seqnames %in% c(as.character(1:22), "X", "Y")) %>%
  dplyr::mutate(chr = paste0("chr", seqnames)) %>%
  dplyr::select(symbol, chr, start, end) %>%
  dplyr::distinct(symbol, .keep_all = TRUE) %>%
  dplyr::arrange(chr, start)

cat(sprintf("  ✓ Gene positions mapped: %d / %d genes\n",
            nrow(gene_positions), nrow(seu)))

# Save for future use (avoid reloading EnsDb next time)
write.csv(gene_positions, file.path(cnv_dir, "gene_positions.csv"), row.names = FALSE)

# ----------------------------------------------------------
# Define cell types
# ----------------------------------------------------------
tumor_types <- c("Epithelial_basal", "Epithelial_squamous",
                 "Epithelial_differentiated", "Epithelial_stem_like")

reference_types <- c("CD8_T_effector", "Treg", "B_cell", "Macrophage_SPP1",
                     "Fibroblast_normal", "Endothelial")

tumor_types <- tumor_types[tumor_types %in% unique(seu@meta.data$cell_type)]
reference_types <- reference_types[reference_types %in% unique(seu@meta.data$cell_type)]

cat(sprintf("\n  Tumor types: %s\n", paste(tumor_types, collapse = ", ")))
cat(sprintf("  Reference types: %s\n", paste(reference_types, collapse = ", ")))

epi_cells <- colnames(seu)[seu@meta.data$cell_type %in% tumor_types]
ref_cells_all <- colnames(seu)[seu@meta.data$cell_type %in% reference_types]

cat(sprintf("  Epithelial cells: %d\n", length(epi_cells)))
cat(sprintf("  Reference cells: %d\n", length(ref_cells_all)))

set.seed(42)
max_ref <- 3000
if (length(ref_cells_all) > max_ref) {
  ref_cells_sample <- sample(ref_cells_all, max_ref)
} else {
  ref_cells_sample <- ref_cells_all
}

# ----------------------------------------------------------
# Calculate CNV scores
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Calculating chromosome-level CNV scores\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

all_cells_for_cnv <- c(epi_cells, ref_cells_sample)
expr_mat <- GetAssayData(seu, assay = "RNA", layer = "data")[, all_cells_for_cnv]

# *** FIX: dplyr::filter ***
gene_chr <- gene_positions %>%
  dplyr::select(symbol, chr) %>%
  dplyr::filter(symbol %in% rownames(expr_mat))

chr_order <- paste0("chr", c(1:22, "X", "Y"))
chromosomes <- chr_order[chr_order %in% unique(gene_chr$chr)]

cat(sprintf("  Chromosomes: %d | Genes: %d\n", length(chromosomes), nrow(gene_chr)))

# CNV matrix
cnv_scores <- matrix(0, nrow = length(all_cells_for_cnv), ncol = length(chromosomes))
rownames(cnv_scores) <- all_cells_for_cnv
colnames(cnv_scores) <- chromosomes

for (chr in chromosomes) {
  chr_genes <- gene_chr$symbol[gene_chr$chr == chr]
  chr_genes <- chr_genes[chr_genes %in% rownames(expr_mat)]
  
  if (length(chr_genes) < 10) next
  
  chr_expr <- colMeans(expr_mat[chr_genes, , drop = FALSE])
  ref_mean <- mean(chr_expr[ref_cells_sample])
  ref_sd <- sd(chr_expr[ref_cells_sample])
  
  if (ref_sd > 0) {
    cnv_scores[, chr] <- (chr_expr - ref_mean) / ref_sd
  }
}

cnv_total_score <- rowSums(abs(cnv_scores))

cat("  ✓ CNV scores calculated.\n")
cat(sprintf("    Epithelial mean: %.2f | Reference mean: %.2f\n",
            mean(cnv_total_score[epi_cells]),
            mean(cnv_total_score[ref_cells_sample])))

# Add to metadata
seu@meta.data$cnv_score <- NA_real_
seu@meta.data[names(cnv_total_score), "cnv_score"] <- cnv_total_score

# ----------------------------------------------------------
# Prepare plot data
# ----------------------------------------------------------
cnv_epi <- data.frame(
  cell = epi_cells,
  cnv_score = cnv_total_score[epi_cells],
  group = as.character(seu@meta.data[epi_cells, "group"]),
  cell_type = as.character(seu@meta.data[epi_cells, "cell_type"]),
  orig.ident = as.character(seu@meta.data[epi_cells, "orig.ident"]),
  stringsAsFactors = FALSE
)

group_colors <- c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")

# ----------------------------------------------------------
# Statistical tests
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Statistical comparisons\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

wilcox_overall <- wilcox.test(cnv_score ~ group, data = cnv_epi)
cat(sprintf("  Overall: p = %.4e\n", wilcox_overall$p.value))
cat(sprintf("    High mean=%.2f | Low mean=%.2f\n",
            mean(cnv_epi$cnv_score[cnv_epi$group == "High_Risk"]),
            mean(cnv_epi$cnv_score[cnv_epi$group == "Low_Risk"])))

for (ct in tumor_types) {
  ct_data <- cnv_epi[cnv_epi$cell_type == ct, ]
  n_high <- sum(ct_data$group == "High_Risk")
  n_low <- sum(ct_data$group == "Low_Risk")
  if (n_high >= 10 && n_low >= 10) {
    test <- wilcox.test(cnv_score ~ group, data = ct_data)
    cat(sprintf("  %s: p=%.4e (H=%.2f, L=%.2f)\n", ct, test$p.value,
                mean(ct_data$cnv_score[ct_data$group == "High_Risk"]),
                mean(ct_data$cnv_score[ct_data$group == "Low_Risk"])))
  } else {
    cat(sprintf("  %s: insufficient cells (H=%d, L=%d)\n", ct, n_high, n_low))
  }
}

# ----------------------------------------------------------
# FIGURES
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Generating CNV figures\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Fig 1: Overall boxplot
p1 <- ggplot(cnv_epi, aes(x = group, y = cnv_score, fill = group)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.8, width = 0.5) +
  scale_fill_manual(values = group_colors) +
  stat_compare_means(method = "wilcox.test", label.x.npc = 0.3, size = 4) +
  theme_classic(base_size = 11) +
  labs(x = "", y = "CNV Score",
       title = "CNV Score: High-Risk vs Low-Risk",
       subtitle = "Epithelial cells") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")
ggsave(file.path(cnv_dir, "Fig1_CNV_boxplot_overall.pdf"), p1, width = 5, height = 5)
cat("  ✓ Fig1: Overall boxplot\n")

# Fig 2: By subtype
p2 <- ggplot(cnv_epi, aes(x = cell_type, y = cnv_score, fill = group)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.8, width = 0.6) +
  scale_fill_manual(values = group_colors) +
  stat_compare_means(aes(group = group), method = "wilcox.test",
                     label = "p.signif", size = 3.5) +
  theme_classic(base_size = 10) +
  labs(x = "", y = "CNV Score",
       title = "CNV by Epithelial Subtype", fill = "Group") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "top")
ggsave(file.path(cnv_dir, "Fig2_CNV_subtype.pdf"), p2, width = 8, height = 5)
cat("  ✓ Fig2: Subtype boxplot\n")

# Fig 3: Violin
p3 <- ggplot(cnv_epi, aes(x = group, y = cnv_score, fill = group)) +
  geom_violin(alpha = 0.7, trim = TRUE) +
  geom_boxplot(width = 0.15, outlier.size = 0.3, fill = "white", alpha = 0.8) +
  scale_fill_manual(values = group_colors) +
  stat_compare_means(method = "wilcox.test", label.x.npc = 0.3, size = 4) +
  theme_classic(base_size = 11) +
  labs(x = "", y = "CNV Score", title = "CNV Distribution") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")
ggsave(file.path(cnv_dir, "Fig3_CNV_violin.pdf"), p3, width = 5, height = 5)
cat("  ✓ Fig3: Violin\n")

# Fig 4: UMAP
p4 <- FeaturePlot(seu, features = "cnv_score", reduction = "umap",
                  pt.size = 0.1, order = TRUE) +
  scale_color_gradientn(colors = c("grey90", "#FEE08B", "#F46D43", "#A50026"),
                        name = "CNV Score", na.value = "grey90") +
  ggtitle("CNV Score on UMAP") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))
ggsave(file.path(cnv_dir, "Fig4_CNV_UMAP.pdf"), p4, width = 8, height = 7)
cat("  ✓ Fig4: UMAP\n")

# Fig 5: Density
median_df <- cnv_epi %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(m = median(cnv_score), .groups = "drop")

p5 <- ggplot(cnv_epi, aes(x = cnv_score, fill = group, color = group)) +
  geom_density(alpha = 0.4) +
  scale_fill_manual(values = group_colors) +
  scale_color_manual(values = group_colors) +
  geom_vline(data = median_df, aes(xintercept = m, color = group),
             linetype = "dashed", linewidth = 0.7) +
  theme_classic(base_size = 10) +
  labs(x = "CNV Score", y = "Density",
       title = "CNV Score Distribution",
       fill = "Group", color = "Group") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "top")
ggsave(file.path(cnv_dir, "Fig5_CNV_density.pdf"), p5, width = 7, height = 5)
cat("  ✓ Fig5: Density\n")

# Fig 6: Per-sample
cnv_sample <- cnv_epi %>%
  dplyr::group_by(orig.ident, group) %>%
  dplyr::summarise(mean_cnv = mean(cnv_score), n_cells = n(), .groups = "drop")

p6 <- ggplot(cnv_sample, aes(x = group, y = mean_cnv, fill = group)) +
  geom_boxplot(outlier.shape = NA, width = 0.5, alpha = 0.8) +
  geom_jitter(width = 0.1, size = 3, alpha = 0.7) +
  scale_fill_manual(values = group_colors) +
  stat_compare_means(method = "wilcox.test", size = 4, label.x.npc = 0.3) +
  theme_classic(base_size = 11) +
  labs(x = "", y = "Mean CNV Score per Sample",
       title = "Sample-level CNV (Epithelial)") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")
ggsave(file.path(cnv_dir, "Fig6_CNV_per_sample.pdf"), p6, width = 5, height = 5)
write.csv(cnv_sample, file.path(cnv_dir, "CNV_per_sample.csv"), row.names = FALSE)
cat("  ✓ Fig6: Per-sample\n")

# Fig 7: Chromosome barplot
cnv_epi_chr <- cnv_epi %>%
  dplyr::left_join(
    as.data.frame(cnv_scores[epi_cells, , drop = FALSE]) %>%
      tibble::rownames_to_column("cell"),
    by = "cell"
  )

cnv_chr_means <- cnv_epi_chr %>%
  tidyr::pivot_longer(cols = all_of(chromosomes),
                      names_to = "chromosome", values_to = "cnv_zscore") %>%
  dplyr::group_by(group, chromosome) %>%
  dplyr::summarise(mean_cnv = mean(cnv_zscore, na.rm = TRUE), .groups = "drop")

p7 <- ggplot(cnv_chr_means, aes(x = factor(chromosome, levels = chr_order),
                                y = mean_cnv, fill = group)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = group_colors) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  theme_classic(base_size = 9) +
  labs(x = "Chromosome", y = "Mean CNV Z-score",
       title = "Chromosome-level CNV: High-Risk vs Low-Risk",
       fill = "Group") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        legend.position = "top")
ggsave(file.path(cnv_dir, "Fig7_CNV_chr_barplot.pdf"), p7, width = 12, height = 5)
cat("  ✓ Fig7: Chromosome barplot\n")

# ----------------------------------------------------------
# Fig 8: Per-chromosome statistical tests + waterfall
# ----------------------------------------------------------
cat("  Per-chromosome tests...\n")

chr_tests <- data.frame(
  chromosome = character(0),
  mean_high = numeric(0),
  mean_low = numeric(0),
  diff = numeric(0),
  p_value = numeric(0),
  stringsAsFactors = FALSE
)

for (chr in chromosomes) {
  high_vals <- cnv_scores[cnv_epi$cell[cnv_epi$group == "High_Risk"], chr]
  low_vals <- cnv_scores[cnv_epi$cell[cnv_epi$group == "Low_Risk"], chr]
  
  if (length(high_vals) < 10 || length(low_vals) < 10) next
  
  test <- wilcox.test(high_vals, low_vals)
  
  chr_tests <- rbind(chr_tests, data.frame(
    chromosome = chr,
    mean_high = mean(high_vals),
    mean_low = mean(low_vals),
    diff = mean(high_vals) - mean(low_vals),
    p_value = test$p.value,
    stringsAsFactors = FALSE
  ))
}

chr_tests$p_adj <- p.adjust(chr_tests$p_value, method = "BH")
chr_tests$significance <- dplyr::case_when(
  chr_tests$p_adj < 0.001 ~ "***",
  chr_tests$p_adj < 0.01 ~ "**",
  chr_tests$p_adj < 0.05 ~ "*",
  TRUE ~ "ns"
)
chr_tests <- chr_tests %>% dplyr::arrange(p_value)

write.csv(chr_tests, file.path(cnv_dir, "CNV_chr_tests.csv"), row.names = FALSE)

# Print significant
sig_chr <- chr_tests %>% dplyr::filter(p_adj < 0.05)
if (nrow(sig_chr) > 0) {
  cat("  Significant chromosomes:\n")
  for (i in 1:nrow(sig_chr)) {
    dir_label <- ifelse(sig_chr$diff[i] > 0, "GAIN in High", "LOSS in High")
    cat(sprintf("    %s: diff=%.3f, padj=%.2e (%s) %s\n",
                sig_chr$chromosome[i], sig_chr$diff[i], sig_chr$p_adj[i],
                sig_chr$significance[i], dir_label))
  }
} else {
  cat("  No significant chromosome-level differences (padj < 0.05).\n")
}

# Waterfall plot
chr_tests_ordered <- chr_tests %>%
  dplyr::mutate(chromosome = factor(chromosome, levels = chr_order))

p8 <- ggplot(chr_tests_ordered, aes(x = chromosome, y = diff, fill = diff > 0)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("TRUE" = "#E64B35", "FALSE" = "#4DBBD5"),
                    labels = c("TRUE" = "Gain in High-Risk", "FALSE" = "Loss in High-Risk"),
                    name = "") +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  geom_text(data = chr_tests_ordered %>% dplyr::filter(p_adj < 0.05),
            aes(label = significance, y = diff + sign(diff) * 0.03),
            size = 3, inherit.aes = FALSE,
            aes_string = NULL) +
  theme_classic(base_size = 9) +
  labs(x = "Chromosome", y = "Mean CNV Difference (High - Low)",
       title = "Chromosome-level CNV: High-Risk vs Low-Risk",
       subtitle = "* padj<0.05, ** padj<0.01, *** padj<0.001") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        legend.position = "top")

# Fix the geom_text issue (remove aes_string)
p8 <- ggplot(chr_tests_ordered, aes(x = chromosome, y = diff, fill = diff > 0)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("TRUE" = "#E64B35", "FALSE" = "#4DBBD5"),
                    labels = c("TRUE" = "Gain in High-Risk", "FALSE" = "Loss in High-Risk"),
                    name = "") +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  theme_classic(base_size = 9) +
  labs(x = "Chromosome", y = "Mean CNV Difference (High − Low)",
       title = "Chromosome-level CNV Changes",
       subtitle = "Positive = relative gain in High-Risk; * padj<0.05") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        legend.position = "top")

# Add significance annotations separately
sig_data <- chr_tests_ordered %>% dplyr::filter(p_adj < 0.05)
if (nrow(sig_data) > 0) {
  p8 <- p8 + geom_text(data = sig_data,
                       aes(x = chromosome, y = diff + sign(diff) * 0.05,
                           label = significance),
                       size = 3, inherit.aes = FALSE)
}

ggsave(file.path(cnv_dir, "Fig8_CNV_waterfall.pdf"), p8, width = 10, height = 5)
cat("  ✓ Fig8: Waterfall plot\n")

# ----------------------------------------------------------
# FIGURE 9: CNV heatmap (cells × chromosomes) — ComplexHeatmap
# ----------------------------------------------------------
cat("  Generating CNV heatmap (cells × chromosomes)...\n")

# Downsample epithelial cells for heatmap visualization
set.seed(42)
n_sample_heatmap <- min(2000, length(epi_cells))
epi_sample <- sample(epi_cells, n_sample_heatmap)

# Order cells by group then CNV score
cell_order_df <- data.frame(
  cell = epi_sample,
  group = as.character(seu@meta.data[epi_sample, "group"]),
  cell_type = as.character(seu@meta.data[epi_sample, "cell_type"]),
  cnv_total = cnv_total_score[epi_sample],
  stringsAsFactors = FALSE
) %>%
  dplyr::arrange(group, cell_type, desc(cnv_total))

# CNV matrix for heatmap
valid_chr_cols <- chromosomes[chromosomes %in% colnames(cnv_scores)]
cnv_heatmap_mat <- cnv_scores[cell_order_df$cell, valid_chr_cols]

# Clip values
cnv_heatmap_mat[cnv_heatmap_mat > 3] <- 3
cnv_heatmap_mat[cnv_heatmap_mat < -3] <- -3

# Row annotation
row_ha_cnv <- rowAnnotation(
  Group = cell_order_df$group,
  Cell_Type = cell_order_df$cell_type,
  CNV_Score = anno_barplot(cell_order_df$cnv_total, width = unit(1.5, "cm"),
                           gp = gpar(fill = ifelse(cell_order_df$group == "High_Risk",
                                                   "#E64B35", "#4DBBD5"))),
  col = list(
    Group = c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5"),
    Cell_Type = c("Epithelial_basal" = "#8491B4",
                  "Epithelial_squamous" = "#B09C85",
                  "Epithelial_differentiated" = "#91D1C2",
                  "Epithelial_stem_like" = "#7E6148")
  ),
  show_annotation_name = TRUE,
  annotation_name_gp = gpar(fontsize = 7)
)

# Color function
cnv_col_fun <- colorRamp2(c(-2, -1, 0, 1, 2),
                          c("#3C5488", "#6BAED6", "white", "#FC8D62", "#D73027"))

# Plot heatmap
pdf(file.path(cnv_dir, "Fig9_CNV_heatmap_cells_chr.pdf"), width = 14, height = 10)
ht_cnv <- Heatmap(
  cnv_heatmap_mat,
  name = "CNV\nZ-score",
  col = cnv_col_fun,
  left_annotation = row_ha_cnv,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_row_names = FALSE,
  show_column_names = TRUE,
  column_names_gp = gpar(fontsize = 8),
  column_names_rot = 45,
  row_split = factor(cell_order_df$group, levels = c("High_Risk", "Low_Risk")),
  row_gap = unit(3, "mm"),
  border = TRUE,
  column_title = "Copy Number Variation: Epithelial Cells by Chromosome",
  column_title_gp = gpar(fontsize = 12, fontface = "bold"),
  row_title_gp = gpar(fontsize = 10, fontface = "bold"),
  use_raster = TRUE,
  raster_quality = 3
)
draw(ht_cnv)
dev.off()

cat("  ✓ Fig9: CNV heatmap (cells × chromosomes)\n")

# ----------------------------------------------------------
# FIGURE 10: CNV difference heatmap (cell_type × chromosome)
# ----------------------------------------------------------
cat("  Generating CNV difference heatmap...\n")

# Mean CNV per cell_type × group × chromosome
cnv_diff_data <- cnv_epi_chr %>%
  tidyr::pivot_longer(cols = all_of(chromosomes),
                      names_to = "chromosome", values_to = "cnv_val") %>%
  dplyr::group_by(cell_type, group, chromosome) %>%
  dplyr::summarise(mean_cnv = mean(cnv_val, na.rm = TRUE), .groups = "drop")

# Calculate difference (High - Low)
cnv_diff_wide <- cnv_diff_data %>%
  tidyr::pivot_wider(names_from = group, values_from = mean_cnv, values_fill = 0) %>%
  dplyr::mutate(diff = High_Risk - Low_Risk) %>%
  dplyr::select(cell_type, chromosome, diff) %>%
  tidyr::pivot_wider(names_from = chromosome, values_from = diff, values_fill = 0)

# Convert to matrix
cnv_diff_mat <- as.matrix(cnv_diff_wide[, -1])
rownames(cnv_diff_mat) <- cnv_diff_wide$cell_type

# Order chromosomes
valid_chr_for_diff <- intersect(chr_order, colnames(cnv_diff_mat))
cnv_diff_mat <- cnv_diff_mat[, valid_chr_for_diff, drop = FALSE]

# Clip
cnv_diff_mat[cnv_diff_mat > 1.5] <- 1.5
cnv_diff_mat[cnv_diff_mat < -1.5] <- -1.5

diff_col_fun <- colorRamp2(c(-1.5, 0, 1.5), c("#4DBBD5", "white", "#E64B35"))

pdf(file.path(cnv_dir, "Fig10_CNV_diff_heatmap.pdf"), width = 12, height = 4)
ht_diff <- Heatmap(
  cnv_diff_mat,
  name = "CNV Diff\n(High-Low)",
  col = diff_col_fun,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_gp = gpar(fontsize = 9),
  column_names_gp = gpar(fontsize = 8),
  column_names_rot = 45,
  border = TRUE,
  column_title = "CNV Difference (High-Risk − Low-Risk) per Chromosome",
  column_title_gp = gpar(fontsize = 11, fontface = "bold"),
  cell_fun = function(j, i, x, y, width, height, fill) {
    grid.text(sprintf("%.2f", cnv_diff_mat[i, j]), x, y,
              gp = gpar(fontsize = 5))
  }
)
draw(ht_diff)
dev.off()

cat("  ✓ Fig10: CNV difference heatmap\n")

# ----------------------------------------------------------
# FIGURE 11: Combined multi-panel figure
# ----------------------------------------------------------
cat("  Generating combined figure...\n")

library(patchwork)

p_combined <- (p1 | p3 | p5) / (p2) +
  plot_annotation(
    title = "Copy Number Variation Analysis: High-Risk vs Low-Risk",
    subtitle = "Epithelial cells, chromosome-level expression Z-scores relative to immune/stromal reference",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
                  plot.subtitle = element_text(size = 10, hjust = 0.5))
  ) +
  plot_layout(heights = c(1, 1))

ggsave(file.path(cnv_dir, "Fig11_CNV_combined.pdf"), p_combined,
       width = 14, height = 10)
cat("  ✓ Fig11: Combined figure\n")

# ----------------------------------------------------------
# Save updated Seurat object
# ----------------------------------------------------------
cat("\n  Saving Seurat object with CNV scores...\n")
saveRDS(seu, file.path(output_dir, "seu_with_cnv_scores.rds"))
cat("  ✓ Saved.\n")

# ----------------------------------------------------------
# inferCNV preparation (for full analysis later)
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Preparing inferCNV input files\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

has_infercnv <- requireNamespace("infercnv", quietly = TRUE)

# Save gene order file regardless
gene_order_file <- file.path(cnv_dir, "infercnv_gene_order.txt")
gene_order_for_infercnv <- gene_positions %>%
  dplyr::select(symbol, chr, start, end)
write.table(gene_order_for_infercnv, gene_order_file, sep = "\t",
            quote = FALSE, row.names = FALSE, col.names = FALSE)
cat(sprintf("  Gene order file: %d genes\n", nrow(gene_order_for_infercnv)))

# Save annotation file
set.seed(42)
max_tumor_infercnv <- 3000
max_ref_infercnv <- 2000

tumor_cells_ds <- epi_cells
if (length(tumor_cells_ds) > max_tumor_infercnv) {
  tumor_cells_ds <- sample(tumor_cells_ds, max_tumor_infercnv)
}

ref_cells_ds <- ref_cells_all
if (length(ref_cells_ds) > max_ref_infercnv) {
  ref_cells_ds <- sample(ref_cells_ds, max_ref_infercnv)
}

all_cells_infercnv <- c(tumor_cells_ds, ref_cells_ds)

# Annotation: cell_type_group for tumor, cell_type for reference
annot_df <- data.frame(row.names = all_cells_infercnv)
for (cell in all_cells_infercnv) {
  ct <- as.character(seu@meta.data[cell, "cell_type"])
  grp <- as.character(seu@meta.data[cell, "group"])
  
  if (ct %in% tumor_types) {
    annot_df[cell, 1] <- paste0(ct, "_", grp)
  } else {
    annot_df[cell, 1] <- ct
  }
}

annot_file <- file.path(cnv_dir, "infercnv_annotations.txt")
write.table(annot_df, annot_file, sep = "\t", quote = FALSE, col.names = FALSE)
cat(sprintf("  Annotation file: %d cells\n", nrow(annot_df)))

# Save count matrix (sparse → dense for inferCNV)
cat("  Saving count matrix for inferCNV (this may take a minute)...\n")
counts_infercnv <- GetAssayData(seu, assay = "RNA", layer = "counts")[, all_cells_infercnv]

# Filter to genes with positions
genes_with_pos <- gene_order_for_infercnv$symbol
genes_in_both <- intersect(genes_with_pos, rownames(counts_infercnv))
counts_infercnv <- counts_infercnv[genes_in_both, ]

# Filter low-expression genes
gene_pct <- rowSums(counts_infercnv > 0) / ncol(counts_infercnv)
genes_keep <- names(gene_pct)[gene_pct > 0.05]
counts_infercnv <- counts_infercnv[genes_keep, ]

counts_file <- file.path(cnv_dir, "infercnv_counts.txt")
write.table(as.matrix(counts_infercnv), counts_file, sep = "\t", quote = FALSE)
cat(sprintf("  Count matrix: %d genes × %d cells\n",
            nrow(counts_infercnv), ncol(counts_infercnv)))

# Run inferCNV if available
if (has_infercnv) {
  library(infercnv)
  
  cat("\n  Running inferCNV (this will take 30-60 minutes)...\n")
  
  ref_group_names <- unique(annot_df[ref_cells_ds, 1])
  
  infercnv_output_dir <- file.path(cnv_dir, "infercnv_output")
  dir.create(infercnv_output_dir, showWarnings = FALSE)
  
  tryCatch({
    infercnv_obj <- CreateInfercnvObject(
      raw_counts_matrix = counts_file,
      annotations_file = annot_file,
      gene_order_file = gene_order_file,
      ref_group_names = ref_group_names
    )
    
    infercnv_obj <- infercnv::run(
      infercnv_obj,
      cutoff = 0.1,
      out_dir = infercnv_output_dir,
      cluster_by_groups = TRUE,
      denoise = TRUE,
      HMM = TRUE,
      num_threads = 8,
      analysis_mode = "subclusters",
      output_format = "pdf",
      no_prelim_plot = FALSE,
      plot_steps = FALSE
    )
    
    cat("  ✓ inferCNV complete.\n")
    
  }, error = function(e) {
    cat(sprintf("  ✗ inferCNV failed: %s\n", e$message))
    cat("  The input files are ready. You can run inferCNV manually later.\n")
  })
  
} else {
  cat("\n  ⚠ inferCNV not installed. Input files prepared for manual run.\n")
  cat("    Install: BiocManager::install('infercnv')\n")
  cat("    Then run:\n")
  cat("      library(infercnv)\n")
  cat(sprintf("      infercnv_obj <- CreateInfercnvObject(\n"))
  cat(sprintf("        raw_counts_matrix = '%s',\n", counts_file))
  cat(sprintf("        annotations_file = '%s',\n", annot_file))
  cat(sprintf("        gene_order_file = '%s',\n", gene_order_file))
  cat(sprintf("        ref_group_names = c('%s'))\n",
              paste(unique(annot_df[ref_cells_ds, 1]), collapse = "', '")))
  cat("      infercnv_obj <- infercnv::run(infercnv_obj, cutoff=0.1, ...)\n")
}

# ===========================================================
# FINAL SUMMARY
# ===========================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════════════════╗\n")
cat("║                    CNV ANALYSIS — COMPLETE                                       ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                                  ║\n")
cat(sprintf("║  Epithelial cells analyzed: %d                                            ║\n",
            length(epi_cells)))
cat(sprintf("║  Reference cells used: %d                                                 ║\n",
            length(ref_cells_sample)))
cat(sprintf("║  Chromosomes analyzed: %d                                                 ║\n",
            length(chromosomes)))
cat(sprintf("║  Overall Wilcoxon p-value: %.4e                                          ║\n",
            wilcox_overall$p.value))
cat("║                                                                                  ║\n")
cat("║  Output files:                                                                   ║\n")
cat("║  ─────────────                                                                   ║\n")
cat("║    Fig1_CNV_boxplot_overall.pdf       — Overall CNV comparison                   ║\n")
cat("║    Fig2_CNV_subtype.pdf               — By epithelial subtype                    ║\n")
cat("║    Fig3_CNV_violin.pdf                — Violin plot                              ║\n")
cat("║    Fig4_CNV_UMAP.pdf                  — UMAP colored by CNV                      ║\n")
cat("║    Fig5_CNV_density.pdf               — Density distribution                     ║\n")
cat("║    Fig6_CNV_per_sample.pdf            — Per-sample comparison                    ║\n")
cat("║    Fig7_CNV_chr_barplot.pdf           — Chromosome-level barplot                 ║\n")
cat("║    Fig8_CNV_waterfall.pdf             — Gain/loss waterfall                      ║\n")
cat("║    Fig9_CNV_heatmap_cells_chr.pdf     — Cells × chromosomes heatmap             ║\n")
cat("║    Fig10_CNV_diff_heatmap.pdf         — Difference heatmap                       ║\n")
cat("║    Fig11_CNV_combined.pdf             — Multi-panel combined                     ║\n")
cat("║                                                                                  ║\n")
cat("║    CNV_chr_tests.csv                  — Per-chromosome Wilcoxon tests            ║\n")
cat("║    CNV_per_sample.csv                 — Per-sample CNV scores                    ║\n")
cat("║    gene_positions.csv                 — Gene chromosomal positions               ║\n")
cat("║    seu_with_cnv_scores.rds            — Updated Seurat object                    ║\n")
cat("║                                                                                  ║\n")
cat("║  inferCNV files (for full analysis):                                              ║\n")
cat("║    infercnv_counts.txt                — Count matrix                             ║\n")
cat("║    infercnv_annotations.txt           — Cell annotations                        ║\n")
cat("║    infercnv_gene_order.txt            — Gene positions                           ║\n")
cat("║                                                                                  ║\n")
cat("║  Next steps:                                                                     ║\n")
cat("║    1. Review CNV figures                                                         ║\n")
cat("║    2. Run full inferCNV if package is installed                                  ║\n")
cat("║    3. Correlate CNV with clinical outcomes                                       ║\n")
cat("║    4. Identify specific amplified/deleted regions                                ║\n")
cat("║    5. Proceed to CellChat / cell-cell communication                             ║\n")
cat("║                                                                                  ║\n")
cat("╚══════════════════════════════════════════════════════════════════════════════════╝\n")

cat("\n━━━ CNV Analysis Complete ━━━\n")
cat(sprintf("  Output: %s\n", cnv_dir))

# ===========================================================
# SESSION INFO
# ===========================================================
cat("\n━━━ Session Info ━━━\n")
cat(sprintf("  R version: %s\n", R.version.string))
cat(sprintf("  Seurat: %s\n", packageVersion("Seurat")))
cat(sprintf("  Date: %s\n", Sys.time()))
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
