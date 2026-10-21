# ============================================================
# 04a_Proportion_Boxplot.R
# Boxplot + Wilcoxon test for focus cell type proportions
# ============================================================

options(Seurat.object.assay.version = "v3")
options(future.globals.maxSize = 8 * 1024^3)

library(Seurat)
library(tidyverse)
library(ggpubr)
library(rstatix)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/04_Differential/Proportion"
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

group_colors <- c("Low_Risk" = "#1976D2", "High_Risk" = "#D32F2F")

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
cat("Loading data...\n")
seu <- readRDS(file.path(input_dir, "seu_annotated.rds"))

# ----------------------------------------------------------
# Calculate proportions per sample
# ----------------------------------------------------------
focus_types <- c("CD8_T_effector", "Myofibroblast", "Mesothelial", "Tumor_specific")

prop_data <- seu@meta.data %>%
  group_by(sample, group) %>%
  summarise(
    total = n(),
    CD8_T_effector = sum(cell_type == "CD8_T_effector") / n() * 100,
    Myofibroblast = sum(cell_type == "Myofibroblast") / n() * 100,
    Mesothelial = sum(cell_type == "Mesothelial") / n() * 100,
    Tumor_specific = sum(cell_type == "Tumor_specific") / n() * 100,
    .groups = "drop"
  ) %>%
  pivot_longer(cols = all_of(focus_types), 
               names_to = "cell_type", values_to = "proportion")

prop_data$cell_type <- factor(prop_data$cell_type, levels = focus_types)
prop_data$group <- factor(prop_data$group, levels = c("Low_Risk", "High_Risk"))

cat("Proportion data computed for", length(unique(prop_data$sample)), "samples\n")
cat("\nSummary:\n")
prop_data %>%
  group_by(cell_type, group) %>%
  summarise(mean = mean(proportion), sd = sd(proportion), .groups = "drop") %>%
  print(n = Inf)

# ----------------------------------------------------------
# Wilcoxon test
# ----------------------------------------------------------
stat_results <- prop_data %>%
  group_by(cell_type) %>%
  wilcox_test(proportion ~ group) %>%
  add_significance() %>%
  add_xy_position(x = "group", dodge = 0.8)

cat("\nWilcoxon test results:\n")
print(as.data.frame(stat_results[, c("cell_type", "group1", "group2", "statistic", "p", "p.signif")]))

# ----------------------------------------------------------
# Plot: Individual boxplots for each cell type
# ----------------------------------------------------------

plot_single_boxplot <- function(data, ct, stat_res) {
  ct_data <- data %>% filter(cell_type == ct)
  ct_stat <- stat_res %>% filter(cell_type == ct)
  
  # Get y position for p-value bracket
  y_max <- max(ct_data$proportion) * 1.15
  
  p <- ggplot(ct_data, aes(x = group, y = proportion, fill = group)) +
    geom_boxplot(width = 0.6, outlier.shape = NA, linewidth = 0.4, alpha = 0.7) +
    geom_jitter(width = 0.15, size = 1.8, alpha = 0.9, shape = 16) +
    scale_fill_manual(values = group_colors) +
    stat_pvalue_manual(ct_stat, label = "p.signif", 
                       y.position = y_max, 
                       bracket.size = 0.3, size = 4,
                       tip.length = 0.02) +
    theme_paper(base_size = 8) +
    labs(x = "", y = "Proportion (%)", title = gsub("_", " ", ct)) +
    theme(legend.position = "none",
          plot.title = element_text(size = 9, face = "bold", hjust = 0.5))
  
  return(p)
}

# Generate individual plots
p_cd8 <- plot_single_boxplot(prop_data, "CD8_T_effector", stat_results)
p_myofib <- plot_single_boxplot(prop_data, "Myofibroblast", stat_results)
p_meso <- plot_single_boxplot(prop_data, "Mesothelial", stat_results)
p_tumor <- plot_single_boxplot(prop_data, "Tumor_specific", stat_results)

# Save individual plots
ggsave(file.path(output_dir, "Boxplot_CD8_T_effector.pdf"), p_cd8,
       width = 55, height = 70, units = "mm")
ggsave(file.path(output_dir, "Boxplot_Myofibroblast.pdf"), p_myofib,
       width = 55, height = 70, units = "mm")
ggsave(file.path(output_dir, "Boxplot_Mesothelial.pdf"), p_meso,
       width = 55, height = 70, units = "mm")
ggsave(file.path(output_dir, "Boxplot_Tumor_specific.pdf"), p_tumor,
       width = 55, height = 70, units = "mm")

# Combined panel (4 in a row)
p_combined <- p_cd8 | p_myofib | p_meso | p_tumor
ggsave(file.path(output_dir, "Boxplot_combined_4panels.pdf"), p_combined,
       width = 220, height = 70, units = "mm")

# Save proportion data
write.csv(prop_data, file.path(output_dir, "Table_proportions_per_sample.csv"), row.names = FALSE)
write.csv(as.matrix(stat_results), file.path(output_dir, "Table_Wilcoxon_results.csv"), row.names = FALSE)

cat("\n━━━ 04a_Proportion_Boxplot COMPLETE ━━━\n")
