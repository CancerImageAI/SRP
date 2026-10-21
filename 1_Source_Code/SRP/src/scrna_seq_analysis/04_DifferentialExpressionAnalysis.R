# ============================================================
# DEG Analysis — Revised
# 1. Symmetric volcano plots
# 2. Cell type proportion boxplots
# ============================================================

library(Seurat)
library(tidyverse)
library(ggplot2)
library(ggrepel)
library(ggpubr)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/04_DEG"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Load annotated Seurat object
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading annotated Seurat object\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_annotated_3rd.rds"))
cat(sprintf("  Cells: %d | Genes: %d\n", ncol(seu), nrow(seu)))

# ----------------------------------------------------------
# Setup RNA assay (Seurat v5 compatible)
# ----------------------------------------------------------
DefaultAssay(seu) <- "RNA"

available_layers <- Layers(seu[["RNA"]])
cat(sprintf("  Available RNA layers: %s\n", paste(available_layers, collapse = ", ")))

# Join layers if split
count_layers <- grep("^counts", available_layers, value = TRUE)
data_layers <- grep("^data", available_layers, value = TRUE)

if (length(count_layers) > 1 || length(data_layers) > 1) {
  cat("  Joining split RNA layers...\n")
  seu[["RNA"]] <- JoinLayers(seu[["RNA"]])
}

# Normalize if needed
if (!"data" %in% Layers(seu[["RNA"]])) {
  cat("  Normalizing RNA assay...\n")
  seu <- NormalizeData(seu, assay = "RNA")
}

cat("  ✓ RNA assay ready.\n")

# ----------------------------------------------------------
# DEG Parameters
# ----------------------------------------------------------
min_cells_per_group <- 30
logfc_threshold <- 0.25
min_pct <- 0.1
test_method <- "wilcox"

# ----------------------------------------------------------
# Run DEG for each cell type
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Running DEG: High_Risk vs Low_Risk per cell type\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cell_types <- levels(seu@meta.data$cell_type)
all_deg_results <- list()
deg_summary <- data.frame()

for (ct in cell_types) {
  cat(sprintf("\n  ── %s ──\n", ct))
  
  cells_ct <- colnames(seu)[seu@meta.data$cell_type == ct]
  group_vec <- seu@meta.data$group[seu@meta.data$cell_type == ct]
  group_counts <- table(group_vec)
  cat(sprintf("    Cells: %s\n",
              paste(names(group_counts), group_counts, sep = "=", collapse = ", ")))
  
  if (length(group_counts) < 2 || any(group_counts < min_cells_per_group)) {
    cat("    ⚠ Skipping: insufficient cells.\n")
    next
  }
  
  seu_ct <- subset(seu, cells = cells_ct)
  Idents(seu_ct) <- "group"
  
  tryCatch({
    deg <- FindMarkers(
      seu_ct,
      ident.1 = "High_Risk",
      ident.2 = "Low_Risk",
      test.use = test_method,
      logfc.threshold = logfc_threshold,
      min.pct = min_pct,
      verbose = FALSE
    )
    
    deg$gene <- rownames(deg)
    deg$cell_type <- ct
    rownames(deg) <- NULL
    
    deg$direction <- case_when(
      deg$avg_log2FC > 0 & deg$p_val_adj < 0.05 ~ "Up_in_High",
      deg$avg_log2FC < 0 & deg$p_val_adj < 0.05 ~ "Down_in_High",
      TRUE ~ "NS"
    )
    
    n_up <- sum(deg$direction == "Up_in_High")
    n_down <- sum(deg$direction == "Down_in_High")
    cat(sprintf("    ✓ DEGs: %d total | %d up | %d down\n", nrow(deg), n_up, n_down))
    
    all_deg_results[[ct]] <- deg
    
    deg_summary <- rbind(deg_summary, data.frame(
      cell_type = ct,
      n_cells_high = as.numeric(group_counts["High_Risk"]),
      n_cells_low = as.numeric(group_counts["Low_Risk"]),
      n_deg_total = nrow(deg),
      n_up_in_high = n_up,
      n_down_in_high = n_down,
      stringsAsFactors = FALSE
    ))
    
  }, error = function(e) {
    cat(sprintf("    ✗ Error: %s\n", e$message))
  })
  
  rm(seu_ct)
  gc(verbose = FALSE)
}

# Save DEG results
all_deg_df <- bind_rows(all_deg_results) %>%
  arrange(cell_type, p_val_adj, desc(abs(avg_log2FC)))

write.csv(all_deg_df, file.path(output_dir, "DEG_all_celltypes_HighvsLow.csv"), row.names = FALSE)
write.csv(deg_summary, file.path(output_dir, "DEG_summary.csv"), row.names = FALSE)
saveRDS(all_deg_results, file.path(output_dir, "all_deg_results.rds"))

# Per cell type
deg_dir <- file.path(output_dir, "per_celltype")
dir.create(deg_dir, showWarnings = FALSE)
for (ct in names(all_deg_results)) {
  ct_clean <- gsub("[/ ]", "_", ct)
  write.csv(all_deg_results[[ct]],
            file.path(deg_dir, sprintf("DEG_%s.csv", ct_clean)), row.names = FALSE)
}

cat("\n  ✓ DEG results saved.\n")

# ===========================================================
# VOLCANO PLOTS — SYMMETRIC X-AXIS
# ===========================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Generating SYMMETRIC volcano plots\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

volcano_dir <- file.path(output_dir, "volcano_plots")
dir.create(volcano_dir, showWarnings = FALSE)

for (ct in names(all_deg_results)) {
  deg <- all_deg_results[[ct]]
  ct_clean <- gsub("[/ ]", "_", ct)
  
  if (nrow(deg) < 5) next
  
  deg_plot <- deg %>%
    mutate(
      sig = case_when(
        p_val_adj < 0.05 & avg_log2FC > 0.5 ~ "Up (|FC|>0.5)",
        p_val_adj < 0.05 & avg_log2FC < -0.5 ~ "Down (|FC|>0.5)",
        p_val_adj < 0.05 ~ "Sig (|FC|≤0.5)",
        TRUE ~ "NS"
      ),
      neg_log10p = -log10(pmax(p_val_adj, 1e-300))
    )
  
  # *** KEY FIX: Force symmetric x-axis ***
  x_max <- max(abs(deg_plot$avg_log2FC), na.rm = TRUE) * 1.1
  x_limits <- c(-x_max, x_max)
  
  # Top genes for labels
  top_genes <- deg_plot %>%
    filter(p_val_adj < 0.05) %>%
    arrange(desc(abs(avg_log2FC))) %>%
    head(15) %>%
    pull(gene)
  
  deg_plot$label <- ifelse(deg_plot$gene %in% top_genes, deg_plot$gene, "")
  
  p <- ggplot(deg_plot, aes(x = avg_log2FC, y = neg_log10p, color = sig)) +
    geom_point(size = 0.5, alpha = 0.6) +
    geom_text_repel(aes(label = label), size = 2.2, max.overlaps = 20,
                    segment.size = 0.2, color = "black", fontface = "italic") +
    scale_color_manual(values = c(
      "Up (|FC|>0.5)" = "#E64B35",
      "Down (|FC|>0.5)" = "#4DBBD5",
      "Sig (|FC|≤0.5)" = "#7F8C8D",
      "NS" = "#D5D8DC"
    ), name = "") +
    # *** SYMMETRIC X-AXIS ***
    scale_x_continuous(limits = x_limits) +
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", linewidth = 0.3, color = "grey50") +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.3, color = "grey50") +
    theme_classic(base_size = 9) +
    labs(x = expression(log[2]~"FC (High-Risk / Low-Risk)"),
         y = expression(-log[10]~"adjusted p-value"),
         title = sprintf("%s: High-Risk vs Low-Risk", ct),
         subtitle = sprintf("Up: %d | Down: %d (padj<0.05, |FC|>0.5)",
                            sum(deg_plot$sig == "Up (|FC|>0.5)"),
                            sum(deg_plot$sig == "Down (|FC|>0.5)"))) +
    theme(plot.title = element_text(face = "bold", size = 10),
          legend.position = "bottom")
  
  ggsave(file.path(volcano_dir, sprintf("Volcano_%s.pdf", ct_clean)), p,
         width = 7, height = 6)
}

cat("  ✓ Symmetric volcano plots saved.\n")

# ===========================================================
# BOXPLOTS OF CELL TYPE PROPORTIONS (High vs Low Risk)
# ===========================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Generating cell type proportion boxplots\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

boxplot_dir <- file.path(output_dir, "proportion_boxplots")
dir.create(boxplot_dir, showWarnings = FALSE)

# Calculate proportions per sample
prop_per_sample <- seu@meta.data %>%
  group_by(orig.ident, group, cell_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(orig.ident) %>%
  mutate(
    total_cells = sum(n),
    proportion = n / total_cells
  ) %>%
  ungroup()

# Ensure all cell type × sample combinations exist (fill missing with 0)
all_combinations <- expand.grid(
  orig.ident = unique(prop_per_sample$orig.ident),
  cell_type = levels(seu@meta.data$cell_type),
  stringsAsFactors = FALSE
)

# Add group info
sample_group <- seu@meta.data %>%
  dplyr::select(orig.ident, group) %>%
  distinct()

all_combinations <- all_combinations %>%
  left_join(sample_group, by = "orig.ident")

prop_complete <- all_combinations %>%
  left_join(prop_per_sample %>% dplyr::select(orig.ident, cell_type, proportion),
            by = c("orig.ident", "cell_type")) %>%
  mutate(proportion = replace_na(proportion, 0))

# Save proportion table
write.csv(prop_complete, file.path(boxplot_dir, "cell_type_proportions_per_sample.csv"),
          row.names = FALSE)

# ----------------------------------------------------------
# Color palette
# ----------------------------------------------------------
group_colors <- c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")

# ----------------------------------------------------------
# 1. Combined boxplot: ALL cell types in one figure
# ----------------------------------------------------------
cat("  Generating combined boxplot...\n")

p_combined <- ggplot(prop_complete, aes(x = cell_type, y = proportion, fill = group)) +
  geom_boxplot(outlier.size = 0.5, width = 0.7, alpha = 0.8,
               position = position_dodge(width = 0.75)) +
  geom_point(aes(color = group), size = 0.8, alpha = 0.6,
             position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.75)) +
  scale_fill_manual(values = group_colors) +
  scale_color_manual(values = group_colors) +
  stat_compare_means(aes(group = group), method = "wilcox.test",
                     label = "p.signif", size = 3, hide.ns = TRUE) +
  theme_classic(base_size = 9) +
  labs(x = "", y = "Proportion of cells per sample",
       title = "Cell Type Proportions: High-Risk vs Low-Risk",
       fill = "Group", color = "Group") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "top")

ggsave(file.path(boxplot_dir, "Boxplot_proportions_all_celltypes.pdf"), p_combined,
       width = 14, height = 6)

# ----------------------------------------------------------
# 2. Individual boxplots per cell type (with statistical test)
# ----------------------------------------------------------
cat("  Generating individual boxplots per cell type...\n")

for (ct in levels(seu@meta.data$cell_type)) {
  ct_clean <- gsub("[/ ]", "_", ct)
  
  ct_data <- prop_complete %>%
    filter(cell_type == ct)
  
  # Skip if no variation
  if (all(ct_data$proportion == 0)) next
  
  # Wilcoxon test
  test_result <- tryCatch({
    wilcox.test(proportion ~ group, data = ct_data)
  }, error = function(e) NULL)
  
  p_val_text <- ifelse(!is.null(test_result),
                       sprintf("Wilcoxon p = %.4f", test_result$p.value),
                       "")
  
  p_ind <- ggplot(ct_data, aes(x = group, y = proportion, fill = group)) +
    geom_boxplot(outlier.shape = NA, width = 0.5, alpha = 0.8) +
    geom_jitter(aes(color = group), width = 0.1, size = 1.5, alpha = 0.7) +
    scale_fill_manual(values = group_colors) +
    scale_color_manual(values = group_colors) +
    stat_compare_means(method = "wilcox.test", label.x.npc = 0.5,
                       label.y.npc = 0.95, size = 3.5) +
    theme_classic(base_size = 10) +
    labs(x = "", y = "Proportion",
         title = ct,
         subtitle = p_val_text) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, size = 8),
          legend.position = "none")
  
  ggsave(file.path(boxplot_dir, sprintf("Boxplot_%s.pdf", ct_clean)), p_ind,
         width = 4, height = 4.5)
}

# ----------------------------------------------------------
# 3. Faceted boxplot: All cell types in facets
# ----------------------------------------------------------
cat("  Generating faceted boxplot...\n")

p_faceted <- ggplot(prop_complete, aes(x = group, y = proportion, fill = group)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.8) +
  geom_jitter(width = 0.1, size = 0.5, alpha = 0.5) +
  scale_fill_manual(values = group_colors) +
  stat_compare_means(method = "wilcox.test", label = "p.format",
                     size = 2.5, label.y.npc = 0.9) +
  facet_wrap(~ cell_type, scales = "free_y", ncol = 6) +
  theme_classic(base_size = 8) +
  labs(x = "", y = "Proportion",
       title = "Cell Type Proportions by Risk Group") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "top",
        strip.text = element_text(size = 7, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 6))

ggsave(file.path(boxplot_dir, "Boxplot_proportions_faceted.pdf"), p_faceted,
       width = 14, height = 10)

# ----------------------------------------------------------
# 4. Statistical test summary table
# ----------------------------------------------------------
cat("  Running statistical tests...\n")

stat_results <- prop_complete %>%
  group_by(cell_type) %>%
  summarise(
    mean_high = mean(proportion[group == "High_Risk"]),
    mean_low = mean(proportion[group == "Low_Risk"]),
    median_high = median(proportion[group == "High_Risk"]),
    median_low = median(proportion[group == "Low_Risk"]),
    fc_high_vs_low = mean(proportion[group == "High_Risk"]) / 
      max(mean(proportion[group == "Low_Risk"]), 1e-10),
    log2fc = log2(max(mean(proportion[group == "High_Risk"]), 1e-10) / 
                    max(mean(proportion[group == "Low_Risk"]), 1e-10)),
    p_value = tryCatch(
      wilcox.test(proportion ~ group, data = cur_data())$p.value,
      error = function(e) NA_real_
    ),
    .groups = "drop"
  ) %>%
  mutate(
    p_adj = p.adjust(p_value, method = "BH"),
    significance = case_when(
      p_adj < 0.001 ~ "***",
      p_adj < 0.01 ~ "**",
      p_adj < 0.05 ~ "*",
      p_adj < 0.1 ~ ".",
      TRUE ~ "ns"
    ),
    enriched_in = case_when(
      p_adj < 0.05 & log2fc > 0 ~ "High_Risk",
      p_adj < 0.05 & log2fc < 0 ~ "Low_Risk",
      TRUE ~ "NS"
    )
  ) %>%
  arrange(p_value)

write.csv(stat_results, file.path(boxplot_dir, "proportion_statistical_tests.csv"),
          row.names = FALSE)

# Print results
cat("\n")
cat("╔══════════════════════════════════════════════════════════════════════════════════╗\n")
cat("║          CELL TYPE PROPORTION COMPARISON (Wilcoxon test)                         ║\n")
cat("╠══════════════════════════════╦══════════╦══════════╦══════════╦═══════╦══════════╣\n")
cat(sprintf("║ %-28s ║ %8s ║ %8s ║ %8s ║ %5s ║ %8s ║\n",
            "Cell Type", "Mean_H", "Mean_L", "log2FC", "Sig", "Enriched"))
cat("╠══════════════════════════════╬══════════╬══════════╬══════════╬═══════╬══════════╣\n")
for (i in 1:nrow(stat_results)) {
  cat(sprintf("║ %-28s ║ %7.4f  ║ %7.4f  ║ %6.2f  ║ %5s ║ %-13s ║\n",
              stat_results$cell_type[i],
              stat_results$mean_high[i],
              stat_results$mean_low[i],
              stat_results$log2fc[i],
              stat_results$significance[i],
              stat_results$enriched_in[i]))
}
cat("╚══════════════════════════════╩══════════╩══════════╩═════════╩═══════╩═══════════════╝\n")

# ----------------------------------------------------------
# 5. Lollipop plot: log2FC of proportions with significance
# ----------------------------------------------------------
cat("\n  Generating lollipop plot of proportion changes...\n")

p_lollipop <- ggplot(stat_results, aes(x = reorder(cell_type, log2fc), y = log2fc)) +
  geom_segment(aes(xend = cell_type, y = 0, yend = log2fc),
               color = "grey60", linewidth = 0.5) +
  geom_point(aes(color = enriched_in, size = -log10(p_adj + 1e-10)), alpha = 0.8) +
  scale_color_manual(values = c("High_Risk" = "#E64B35",
                                "Low_Risk" = "#4DBBD5",
                                "NS" = "grey70"),
                     name = "Enriched in") +
  scale_size_continuous(range = c(2, 6), name = expression(-log[10]~padj)) +
  coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3) +
  theme_classic(base_size = 9) +
  labs(x = "", y = expression(log[2]~"FC (High-Risk / Low-Risk proportion)"),
       title = "Cell Type Proportion Changes",
       subtitle = "Positive = enriched in High-Risk") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave(file.path(boxplot_dir, "Lollipop_proportion_log2FC.pdf"), p_lollipop,
       width = 8, height = 7)

# ----------------------------------------------------------
# 6. Lineage-level proportion boxplot — FIXED
# ----------------------------------------------------------
cat("  Generating lineage-level boxplot...\n")

prop_lineage <- seu@meta.data %>%
  group_by(orig.ident, group, lineage) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(orig.ident) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

p_lineage <- ggplot(prop_lineage, aes(x = lineage, y = proportion, fill = group)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.8) +
  # FIX: Use geom_point with position_jitterdodge instead of geom_jitter with width
  geom_point(aes(color = group), size = 1.5, alpha = 0.6,
             position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.6)) +
  scale_fill_manual(values = c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")) +
  scale_color_manual(values = c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")) +
  stat_compare_means(aes(group = group), method = "wilcox.test",
                     label = "p.format", size = 3) +
  theme_classic(base_size = 10) +
  labs(x = "", y = "Proportion",
       title = "Lineage Proportions: High-Risk vs Low-Risk",
       fill = "Group", color = "Group") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "top")

ggsave(file.path(boxplot_dir, "Boxplot_lineage_proportions.pdf"), p_lineage,
       width = 8, height = 5)

cat("  ✓ Lineage boxplot saved.\n")

# ===========================================================
# FINAL OUTPUT SUMMARY
# ===========================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════════════════════╗\n")
cat("║              DEG + PROPORTION ANALYSIS — COMPLETE                                    ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                                      ║\n")
cat(sprintf("║  Input:  seu_annotated_3rd.rds (%d cells, %d types)                           ║\n",
            ncol(seu), length(cell_types)))
cat(sprintf("║  Comparison: High_Risk vs Low_Risk                                               ║\n"))
cat(sprintf("║  Cell types tested: %d / %d                                                      ║\n",
            nrow(deg_summary), length(cell_types)))
cat(sprintf("║  Total DEGs: %d (padj<0.05, |log2FC|>0.25)                                      ║\n",
            nrow(all_deg_df)))
cat("║                                                                                      ║\n")
cat("║  Output directory: 04_DEG/                                                           ║\n")
cat("║                                                                                      ║\n")
cat("║  ┌─────────────────────────────────────────────────────────────────────────────────┐ ║\n")
cat("║  │ DEG Results                                                                     │ ║\n")
cat("║  ├─────────────────────────────────────────────────────────────────────────────────┤ ║\n")
cat("║  │  DEG_all_celltypes_HighvsLow.csv     — Combined DEG table                      │ ║\n")
cat("║  │  DEG_summary.csv                     — Summary per cell type                   │ ║\n")
cat("║  │  all_deg_results.rds                 — R object                                │ ║\n")
cat("║  │  per_celltype/DEG_*.csv              — Per cell type tables                    │ ║\n")
cat("║  │  top10_DEGs_per_celltype.csv         — Top DEGs                                │ ║\n")
cat("║  └─────────────────────────────────────────────────────────────────────────────────┘ ║\n")
cat("║                                                                                      ║\n")
cat("║  ┌─────────────────────────────────────────────────────────────────────────────────┐ ║\n")
cat("║  │ Volcano Plots (symmetric x-axis)                                                │ ║\n")
cat("║  ├─────────────────────────────────────────────────────────────────────────────────┤ ║\n")
cat("║  │  volcano_plots/Volcano_*.pdf         — Per cell type                           │ ║\n")
cat("║  └─────────────────────────────────────────────────────────────────────────────────┘ ║\n")
cat("║                                                                                      ║\n")
cat("║  ┌─────────────────────────────────────────────────────────────────────────────────┐ ║\n")
cat("║  │ Proportion Boxplots                                                              │ ║\n")
cat("║  ├─────────────────────────────────────────────────────────────────────────────────┤ ║\n")
cat("║  │  proportion_boxplots/                                                           │ ║\n")
cat("║  │    Boxplot_proportions_all_celltypes.pdf   — Combined boxplot                  │ ║\n")
cat("║  │    Boxplot_proportions_faceted.pdf         — Faceted (all types)               │ ║\n")
cat("║  │    Boxplot_*.pdf                           — Individual per type               │ ║\n")
cat("║  │    Boxplot_lineage_proportions.pdf         — Lineage level                     │ ║\n")
cat("║  │    Barplot_proportions_with_significance.pdf — Mean + SE + stars               │ ║\n")
cat("║  │    Lollipop_proportion_log2FC.pdf          — FC lollipop                       │ ║\n")
cat("║  │    proportion_statistical_tests.csv        — Wilcoxon results                  │ ║\n")
cat("║  │    cell_type_proportions_per_sample.csv    — Raw proportions                   │ ║\n")
cat("║  └─────────────────────────────────────────────────────────────────────────────────┘ ║\n")
cat("║                                                                                      ║\n")
cat("║  ┌─────────────────────────────────────────────────────────────────────────────────┐ ║\n")
cat("║  │ Pathway Enrichment                                                               │ ║\n")
cat("║  ├─────────────────────────────────────────────────────────────────────────────────┤ ║\n")
cat("║  │  enrichment/GO_BP_*.csv/pdf                — GO Biological Process             │ ║\n")
cat("║  │  enrichment/KEGG_*.csv                     — KEGG pathways                     │ ║\n")
cat("║  │  GSEA/GSEA_*.csv/pdf                       — GSEA Hallmark                    │ ║\n")
cat("║  │  GSEA/GSEA_NES_heatmap.pdf                 — NES heatmap                      │ ║\n")
cat("║  │  gene_lists/*.txt                          — Gene lists for external tools     │ ║\n")
cat("║  │  gene_lists/*.rnk                          — Ranked lists for GSEA            │ ║\n")
cat("║  └─────────────────────────────────────────────────────────────────────────────────┘ ║\n")
cat("║                                                                                      ║\n")
cat("║  ┌─────────────────────────────────────────────────────────────────────────────────┐ ║\n")
cat("║  │ Other                                                                            │ ║\n")
cat("║  ├─────────────────────────────────────────────────────────────────────────────────┤ ║\n")
cat("║  │  Barplot_DEG_counts.pdf              — DEG count per type (up/down)            │ ║\n")
cat("║  │  BubblePlot_top_DEGs.pdf             — Top DEGs bubble                        │ ║\n")
cat("║  │  Volcano_CAF_publication.pdf         — CAF-focused volcano                    │ ║\n")
cat("║  └─────────────────────────────────────────────────────────────────────────────────┘ ║\n")
cat("║                                                                                      ║\n")
cat("║  Next steps:                                                                         ║\n")
cat("║    1. Review volcano plots and proportion boxplots                                   ║\n")
cat("║    2. Focus on cell types with significant proportion changes                        ║\n")
cat("║    3. CellChat analysis (cell-cell communication)                                    ║\n")
cat("║    4. CAF subtype subclustering                                                      ║\n")
cat("║    5. Spatial transcriptomics integration                                            ║\n")
cat("║                                                                                      ║\n")
cat("╚══════════════════════════════════════════════════════════════════════════════════════╝\n")

cat("\n━━━ Analysis Complete ━━━\n")
cat(sprintf("  Output: %s\n", output_dir))