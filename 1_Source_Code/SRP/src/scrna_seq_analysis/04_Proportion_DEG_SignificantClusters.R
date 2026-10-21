# ============================================================
# 04_Proportion_DEG_SignificantClusters.R
# Step 1: Find clusters with significant proportion differences
# Step 2: Run DE analysis on those clusters
# ============================================================

options(Seurat.object.assay.version = "v3")
options(future.globals.maxSize = 8 * 1024^3)

library(Seurat)
library(tidyverse)
library(ggrepel)
library(patchwork)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/04_Differential/SignificantClusters"
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
cat("━━━ Loading data ━━━\n")
seu <- readRDS(file.path(input_dir, "seu_annotated.rds"))
cat("Total cells:", ncol(seu), "\n")
cat("Clusters:", length(unique(seu$cell_type)), "\n")

# Cluster annotation
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

# ===========================================================
# PART 1: Proportion comparison — ALL clusters
# ===========================================================
cat("\n━━━ PART 1: Proportion comparison ━━━\n")

# Sample-group mapping
sample_group <- seu@meta.data %>%
  dplyr::select(sample, group) %>%
  distinct()

cat("Samples per group:\n")
print(table(sample_group$group))

# Calculate proportion per sample per cluster
cluster_prop <- seu@meta.data %>%
  mutate(cluster_id = as.character(RNA_snn_res.0.5)) %>%
  group_by(sample, group, cluster_id) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(sample, group) %>%
  mutate(total = sum(n), proportion = n / total * 100) %>%
  ungroup()

# Fill missing combinations with 0
all_clusters <- as.character(0:20)
all_combinations <- expand.grid(
  sample = unique(sample_group$sample),
  cluster_id = all_clusters,
  stringsAsFactors = FALSE
) %>%
  left_join(sample_group, by = "sample")

cluster_prop_complete <- left_join(
  all_combinations,
  cluster_prop %>% dplyr::select(sample, cluster_id, n, proportion),
  by = c("sample", "cluster_id")
) %>%
  mutate(
    n = replace_na(n, 0),
    proportion = replace_na(proportion, 0)
  )

cluster_prop_complete$group <- factor(cluster_prop_complete$group, 
                                      levels = c("Low_Risk", "High_Risk"))

# Add cell type annotation
cluster_prop_complete$cell_type <- cluster_annotation[cluster_prop_complete$cluster_id]

# ----------------------------------------------------------
# Wilcoxon test for each cluster
# ----------------------------------------------------------
stat_results <- data.frame(
  cluster_id = character(),
  cell_type = character(),
  n_High = integer(),
  n_Low = integer(),
  mean_High = numeric(),
  mean_Low = numeric(),
  median_High = numeric(),
  median_Low = numeric(),
  FC_HighvsLow = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

for (cl in all_clusters) {
  cl_data <- cluster_prop_complete %>% filter(cluster_id == cl)
  high_vals <- cl_data %>% filter(group == "High_Risk") %>% pull(proportion)
  low_vals <- cl_data %>% filter(group == "Low_Risk") %>% pull(proportion)
  
  ct_name <- cluster_annotation[cl]
  
  # Check testability
  n_nonzero_high <- sum(high_vals > 0)
  n_nonzero_low <- sum(low_vals > 0)
  
  if (n_nonzero_high >= 2 & n_nonzero_low >= 2) {
    wt <- tryCatch(
      wilcox.test(high_vals, low_vals, exact = FALSE),
      error = function(e) list(p.value = NA)
    )
    p_val <- wt$p.value
  } else {
    p_val <- NA
  }
  
  mean_h <- mean(high_vals)
  mean_l <- mean(low_vals)
  
  stat_results <- rbind(stat_results, data.frame(
    cluster_id = cl,
    cell_type = ct_name,
    n_High = length(high_vals),
    n_Low = length(low_vals),
    mean_High = mean_h,
    mean_Low = mean_l,
    median_High = median(high_vals),
    median_Low = median(low_vals),
    FC_HighvsLow = ifelse(mean_l > 0, mean_h / mean_l, ifelse(mean_h > 0, Inf, 1)),
    p_value = p_val,
    stringsAsFactors = FALSE
  ))
}

# Adjust p-values
stat_results$p_adj <- p.adjust(stat_results$p_value, method = "BH")
stat_results$significance <- case_when(
  is.na(stat_results$p_value) ~ "nt",
  stat_results$p_adj < 0.001 ~ "***",
  stat_results$p_adj < 0.01 ~ "**",
  stat_results$p_adj < 0.05 ~ "*",
  stat_results$p_value < 0.05 ~ "*",
  stat_results$p_value < 0.1 ~ ".",
  TRUE ~ "ns"
)
stat_results$direction <- case_when(
  stat_results$mean_High > stat_results$mean_Low ~ "↑ High_Risk",
  stat_results$mean_High < stat_results$mean_Low ~ "↑ Low_Risk",
  TRUE ~ "="
)

stat_results <- stat_results %>% arrange(p_value)

cat("\n╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║              PROPORTION COMPARISON RESULTS (ALL CLUSTERS)            ║\n")
cat("╠══════════════════════════════════════════════════════════════════════╣\n")
print(stat_results %>% 
        dplyr::select(cluster_id, cell_type, mean_High, mean_Low, 
                      FC_HighvsLow, p_value, p_adj, significance, direction),
      row.names = FALSE)
cat("╚══════════════════════════════════════════════════════════════════════╝\n")

write.csv(stat_results, file.path(output_dir, "Table_Wilcoxon_all_clusters.csv"), row.names = FALSE)

# ----------------------------------------------------------
# Identify significant clusters
# ----------------------------------------------------------
# Use nominal p < 0.05 (since small sample sizes limit FDR correction power)
sig_clusters_strict <- stat_results %>% filter(p_adj < 0.05) %>% pull(cluster_id)
sig_clusters_nominal <- stat_results %>% filter(p_value < 0.05) %>% pull(cluster_id)
trending_clusters <- stat_results %>% filter(p_value >= 0.05 & p_value < 0.1) %>% pull(cluster_id)

# Also include group-restricted clusters (only in one group)
group_restricted <- stat_results %>% filter(is.na(p_value)) %>% pull(cluster_id)

cat("\n*** Significant clusters (FDR < 0.05):", 
    ifelse(length(sig_clusters_strict) > 0, paste(sig_clusters_strict, collapse = ", "), "NONE"), "\n")
cat("*** Significant clusters (nominal p < 0.05):", 
    ifelse(length(sig_clusters_nominal) > 0, paste(sig_clusters_nominal, collapse = ", "), "NONE"), "\n")
cat("*** Trending clusters (p < 0.1):", 
    ifelse(length(trending_clusters) > 0, paste(trending_clusters, collapse = ", "), "NONE"), "\n")
cat("*** Group-restricted clusters (not testable):", 
    ifelse(length(group_restricted) > 0, paste(group_restricted, collapse = ", "), "NONE"), "\n")

# For DE analysis: use all clusters with nominal p < 0.05 OR group-restricted
de_clusters <- unique(c(sig_clusters_nominal, group_restricted))
cat("\n>>> Clusters selected for DE analysis:", paste(de_clusters, collapse = ", "), "\n")
cat("    (", paste(cluster_annotation[de_clusters], collapse = ", "), ")\n")

# ----------------------------------------------------------
# Boxplots for significant clusters ONLY
# ----------------------------------------------------------
if (length(sig_clusters_nominal) > 0) {
  cat("\n  Generating boxplots for significant clusters...\n")
  
  sig_boxplots <- list()
  for (cl in sig_clusters_nominal) {
    cl_data <- cluster_prop_complete %>% filter(cluster_id == cl)
    cl_stat <- stat_results %>% filter(cluster_id == cl)
    ct_name <- cl_stat$cell_type
    
    y_max <- max(cl_data$proportion, na.rm = TRUE) * 1.25
    if (y_max == 0 | !is.finite(y_max)) y_max <- 1
    
    p_label <- cl_stat$significance
    
    p <- ggplot(cl_data, aes(x = group, y = proportion, fill = group)) +
      geom_boxplot(width = 0.6, outlier.shape = NA, linewidth = 0.4, alpha = 0.7) +
      geom_jitter(width = 0.15, size = 2, alpha = 0.9, shape = 16) +
      scale_fill_manual(values = group_colors) +
      annotate("text", x = 1.5, y = y_max, label = p_label, 
               size = 5, fontface = "bold") +
      annotate("segment", x = 1, xend = 2, y = y_max * 0.9, yend = y_max * 0.9,
               linewidth = 0.4) +
      theme_paper(base_size = 8) +
      labs(x = "", y = "Proportion (%)", 
           title = sprintf("C%s: %s", cl, ct_name)) +
      theme(legend.position = "none",
            plot.title = element_text(size = 8, face = "bold", hjust = 0.5))
    
    sig_boxplots[[cl]] <- p
    ggsave(file.path(output_dir, sprintf("Boxplot_C%s_%s.pdf", cl, ct_name)),
           p, width = 55, height = 70, units = "mm")
  }
  
  # Combined panel
  ncol_sig <- min(length(sig_boxplots), 5)
  nrow_sig <- ceiling(length(sig_boxplots) / ncol_sig)
  
  ggsave(file.path(output_dir, "Fig1_Boxplot_significant_clusters.pdf"),
         wrap_plots(sig_boxplots, ncol = ncol_sig),
         width = 55 * ncol_sig, height = 70 * nrow_sig, units = "mm")
}

# ===========================================================
# PART 2: DE analysis for SIGNIFICANT clusters
# ===========================================================
cat("\n━━━ PART 2: DE analysis for significant clusters ━━━\n")

MAX_CELLS_PER_GROUP <- 5000

# Fixed Volcano plot function (no "a" in legend)
plot_volcano <- function(deg_df, title, top_n = 20,
                         fc_cutoff = 0.5, padj_cutoff = 0.05) {
  
  deg_df <- deg_df %>%
    mutate(
      color = case_when(
        p_val_adj < padj_cutoff & avg_log2FC > fc_cutoff ~ "Up in High_Risk",
        p_val_adj < padj_cutoff & avg_log2FC < -fc_cutoff ~ "Up in Low_Risk",
        TRUE ~ "NS"
      ),
      neg_log10_padj = -log10(p_val_adj + 1e-300)
    )
  
  # Cap y values
  finite_y <- deg_df$neg_log10_padj[is.finite(deg_df$neg_log10_padj)]
  if (length(finite_y) > 0) {
    max_y <- quantile(finite_y, 0.99, na.rm = TRUE)
    deg_df$neg_log10_padj[deg_df$neg_log10_padj > max_y * 1.2] <- max_y * 1.2
  }
  
  # Top genes
  top_up <- deg_df %>% filter(color == "Up in High_Risk") %>%
    slice_max(n = ceiling(top_n/2), order_by = avg_log2FC, with_ties = FALSE) %>% pull(gene)
  top_down <- deg_df %>% filter(color == "Up in Low_Risk") %>%
    slice_min(n = floor(top_n/2), order_by = avg_log2FC, with_ties = FALSE) %>% pull(gene)
  top_genes <- c(top_up, top_down)
  
  deg_df$label <- ifelse(deg_df$gene %in% top_genes, deg_df$gene, "")
  
  n_up <- sum(deg_df$color == "Up in High_Risk")
  n_down <- sum(deg_df$color == "Up in Low_Risk")
  
  ggplot(deg_df, aes(x = avg_log2FC, y = neg_log10_padj, color = color)) +
    geom_point(size = 0.4, alpha = 0.5) +
    geom_text_repel(aes(label = label), size = 2, max.overlaps = 25,
                    segment.size = 0.2, segment.alpha = 0.5,
                    show.legend = FALSE) +
    scale_color_manual(
      values = c("Up in High_Risk" = "#D32F2F", 
                 "Up in Low_Risk" = "#1976D2", 
                 "NS" = "grey75"),
      name = ""
    ) +
    guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
    geom_vline(xintercept = c(-fc_cutoff, fc_cutoff), 
               linetype = "dashed", linewidth = 0.25, color = "grey40") +
    geom_hline(yintercept = -log10(padj_cutoff), 
               linetype = "dashed", linewidth = 0.25, color = "grey40") +
    theme_paper(base_size = 7) +
    labs(x = "log2FC (High_Risk vs Low_Risk)", 
         y = expression(-log[10](adjusted~P)),
         title = title,
         subtitle = sprintf("Up in High: %d | Up in Low: %d", n_up, n_down)) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 6),
          plot.subtitle = element_text(size = 5.5, hjust = 0.5))
}

# Run DE for each significant cluster
deg_all_results <- list()
deg_summary_list <- list()

for (cl in de_clusters) {
  ct_name <- cluster_annotation[cl]
  cat(sprintf("\n  ── Cluster %s (%s) ──\n", cl, ct_name))
  
  # Subset to this cluster
  sub <- subset(seu, RNA_snn_res.0.5 == cl)
  n_high <- sum(sub$group == "High_Risk")
  n_low <- sum(sub$group == "Low_Risk")
  cat(sprintf("    Cells: High_Risk=%d, Low_Risk=%d\n", n_high, n_low))
  
  # Need >= 30 cells in each group for meaningful DE
  if (n_high < 30 | n_low < 30) {
    cat("    SKIPPED: insufficient cells (need >=30 per group)\n")
    
    # Still report which group dominates
    dominant_group <- ifelse(n_high > n_low, "High_Risk", "Low_Risk")
    cat(sprintf("    This cluster is dominated by %s (%d vs %d cells)\n",
                dominant_group, max(n_high, n_low), min(n_high, n_low)))
    
    deg_summary_list[[cl]] <- data.frame(
      cluster_id = cl, cell_type = ct_name,
      n_High = n_high, n_Low = n_low,
      n_up = NA, n_down = NA, total_DEG = NA,
      status = "skipped_low_cells",
      stringsAsFactors = FALSE
    )
    rm(sub); gc(verbose = FALSE)
    next
  }
  
  # Downsample if needed
  if (n_high > MAX_CELLS_PER_GROUP | n_low > MAX_CELLS_PER_GROUP) {
    set.seed(42)
    cells_high <- WhichCells(sub, expression = group == "High_Risk")
    cells_low <- WhichCells(sub, expression = group == "Low_Risk")
    if (length(cells_high) > MAX_CELLS_PER_GROUP) cells_high <- sample(cells_high, MAX_CELLS_PER_GROUP)
    if (length(cells_low) > MAX_CELLS_PER_GROUP) cells_low <- sample(cells_low, MAX_CELLS_PER_GROUP)
    sub <- subset(sub, cells = c(cells_high, cells_low))
    cat(sprintf("    Downsampled to %d + %d\n", length(cells_high), length(cells_low)))
  }
  
  Idents(sub) <- "group"
  
  # FindMarkers: High vs Low
  deg <- tryCatch({
    FindMarkers(sub, ident.1 = "High_Risk", ident.2 = "Low_Risk",
                min.pct = 0.1, logfc.threshold = 0.1, verbose = FALSE)
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    return(NULL)
  })
  
  if (!is.null(deg) && nrow(deg) > 0) {
    deg$gene <- rownames(deg)
    deg$cluster_id <- cl
    deg$cell_type <- ct_name
    
    n_up <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC > 0.5)
    n_down <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC < -0.5)
    cat(sprintf("    DEGs: UP in High=%d, DOWN in High=%d, Total tested=%d\n", 
                n_up, n_down, nrow(deg)))
    
    deg_all_results[[cl]] <- deg
    
    # Save DEG table
    write.csv(deg, file.path(output_dir, sprintf("Table_DEG_C%s_%s.csv", cl, ct_name)), 
              row.names = FALSE)
    
    # Volcano plot
    p_vol <- plot_volcano(deg, sprintf("C%s: %s (High vs Low Risk)", cl, ct_name), top_n = 20)
    ggsave(file.path(output_dir, sprintf("Volcano_C%s_%s.pdf", cl, ct_name)), 
           p_vol, width = 110, height = 110, units = "mm")
    
    # Top DEGs table (for quick reference)
    top_degs <- deg %>%
      filter(p_val_adj < 0.05) %>%
      arrange(desc(abs(avg_log2FC))) %>%
      head(30)
    
    if (nrow(top_degs) > 0) {
      cat("    Top DEGs:\n")
      top_print <- top_degs %>% 
        dplyr::select(gene, avg_log2FC, p_val_adj) %>%
        head(10)
      print(top_print, row.names = FALSE)
    }
    
    deg_summary_list[[cl]] <- data.frame(
      cluster_id = cl, cell_type = ct_name,
      n_High = n_high, n_Low = n_low,
      n_up = n_up, n_down = n_down, total_DEG = n_up + n_down,
      status = "completed",
      stringsAsFactors = FALSE
    )
  } else {
    cat("    No DEGs found.\n")
    deg_summary_list[[cl]] <- data.frame(
      cluster_id = cl, cell_type = ct_name,
      n_High = n_high, n_Low = n_low,
      n_up = 0, n_down = 0, total_DEG = 0,
      status = "no_results",
      stringsAsFactors = FALSE
    )
  }
  
  rm(sub); gc(verbose = FALSE)
}

# ===========================================================
# PART 3: Summary of DE results
# ===========================================================
cat("\n━━━ PART 3: DE Summary ━━━\n")

deg_summary <- bind_rows(deg_summary_list) %>% arrange(desc(total_DEG))

cat("\n╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║                    DEG SUMMARY (SIGNIFICANT CLUSTERS)                ║\n")
cat("╠══════════════════════════════════════════════════════════════════════╣\n")
print(deg_summary, row.names = FALSE)
cat("╚══════════════════════════════════════════════════════════════════════╝\n")

write.csv(deg_summary, file.path(output_dir, "Table_DEG_summary.csv"), row.names = FALSE)

# Save all DEG results
if (length(deg_all_results) > 0) {
  saveRDS(deg_all_results, file.path(output_dir, "deg_results_significant_clusters.rds"))
  
  # Combined DEG table
  all_deg_combined <- bind_rows(deg_all_results)
  write.csv(all_deg_combined, file.path(output_dir, "Table_DEG_combined.csv"), row.names = FALSE)
  
  cat(sprintf("\nTotal genes tested across all clusters: %d\n", nrow(all_deg_combined)))
  cat(sprintf("Significant DEGs (padj<0.05, |log2FC|>0.5): %d\n",
              sum(all_deg_combined$p_val_adj < 0.05 & abs(all_deg_combined$avg_log2FC) > 0.5)))
}

# ===========================================================
# PART 4: DEG count barplot
# ===========================================================
cat("\n━━━ PART 4: Visualization ━━━\n")

deg_count_df <- deg_summary %>%
  filter(status == "completed" & total_DEG > 0) %>%
  pivot_longer(cols = c(n_up, n_down), names_to = "direction", values_to = "count") %>%
  mutate(
    direction = ifelse(direction == "n_up", "Up in High_Risk", "Down in High_Risk"),
    count_plot = ifelse(direction == "Down in High_Risk", -count, count),
    label = sprintf("C%s: %s", cluster_id, cell_type)
  )

if (nrow(deg_count_df) > 0) {
  cluster_order <- deg_summary %>%
    filter(status == "completed" & total_DEG > 0) %>%
    arrange(desc(total_DEG)) %>%
    mutate(label = sprintf("C%s: %s", cluster_id, cell_type)) %>%
    pull(label)
  
  deg_count_df$label <- factor(deg_count_df$label, levels = rev(cluster_order))
  
  p_deg_bar <- ggplot(deg_count_df, aes(x = count_plot, y = label, fill = direction)) +
    geom_col(width = 0.7) +
    geom_vline(xintercept = 0, linewidth = 0.3) +
    scale_fill_manual(values = c("Up in High_Risk" = "#D32F2F", 
                                 "Down in High_Risk" = "#1976D2")) +
    theme_paper(base_size = 7) +
    labs(x = "Number of DEGs (padj<0.05, |log2FC|>0.5)", y = "", fill = "",
         title = "DEGs in Significant Clusters (High vs Low Risk)") +
    theme(axis.text.y = element_text(size = 6),
          legend.position = "bottom",
          legend.text = element_text(size = 6))
  
  ggsave(file.path(output_dir, "Fig2_DEG_count_barplot.pdf"), p_deg_bar,
         width = 140, height = max(60, length(cluster_order) * 8), units = "mm")
}

# ===========================================================
# PART 5: Heatmap of top DEGs across significant clusters
# ===========================================================
if (length(deg_all_results) > 1) {
  cat("  Generating DEG heatmap...\n")
  
  library(ComplexHeatmap)
  library(circlize)
  
  # Get top 5 DEGs per cluster by absolute log2FC
  top_degs_all <- all_deg_combined %>%
    filter(p_val_adj < 0.05 & abs(avg_log2FC) > 0.5) %>%
    group_by(cluster_id) %>%
    slice_max(n = 5, order_by = abs(avg_log2FC), with_ties = FALSE) %>%
    ungroup()
  
  if (nrow(top_degs_all) > 5) {
    # Create log2FC matrix
    deg_matrix <- top_degs_all %>%
      dplyr::select(gene, cell_type, avg_log2FC) %>%
      distinct(gene, cell_type, .keep_all = TRUE) %>%
      pivot_wider(names_from = cell_type, values_from = avg_log2FC, values_fill = 0) %>%
      column_to_rownames("gene")
    
    if (nrow(deg_matrix) > 3 && ncol(deg_matrix) > 1) {
      mat <- as.matrix(deg_matrix)
      mat[mat > 3] <- 3
      mat[mat < -3] <- -3
      
      # Shorten column names
      colnames(mat) <- gsub("Epithelial_", "Epi_", colnames(mat))
      colnames(mat) <- gsub("Macrophage_SPP1", "Mac_SPP1", colnames(mat))
      colnames(mat) <- gsub("CD8_T_effector", "CD8_Teff", colnames(mat))
      colnames(mat) <- gsub("Activated_fibroblast", "Act_fibro", colnames(mat))
      colnames(mat) <- gsub("Fibroblast_normal", "Fibro_norm", colnames(mat))
      colnames(mat) <- gsub("Tumor_specific", "Tumor_spec", colnames(mat))
      colnames(mat) <- gsub("Pericyte_SMC", "Pericyte", colnames(mat))
      colnames(mat) <- gsub("Myofibroblast", "Myofibro", colnames(mat))
      
      pdf(file.path(output_dir, "Fig3_DEG_heatmap.pdf"), width = 10, height = 14)
      ht <- Heatmap(mat,
                    name = "log2FC\n(High/Low)",
                    col = colorRamp2(c(-3, 0, 3), c("#2166AC", "white", "#B2182B")),
                    cluster_rows = TRUE,
                    cluster_columns = TRUE,
                    show_row_names = TRUE,
                    show_column_names = TRUE,
                    row_names_gp = gpar(fontsize = 5),
                    column_names_gp = gpar(fontsize = 8, fontface = "bold"),
                    column_names_rot = 45,
                    column_names_max_height = unit(6, "cm"),
                    column_title = "Top DEGs in Significant Clusters",
                    column_title_gp = gpar(fontsize = 10, fontface = "bold"))
      draw(ht, padding = unit(c(5, 5, 15, 5), "mm"))
      dev.off()
    }
  }
}

# ===========================================================
# PART 6: Proportion overview figure
# ===========================================================
cat("  Generating proportion overview...\n")

# Log2FC vs -log10(p) scatter (proportion differences)
prop_scatter_df <- stat_results %>%
  filter(!is.na(p_value)) %>%
  mutate(
    log2FC_prop = log2(FC_HighvsLow),
    neg_log10_p = -log10(p_value),
    color = case_when(
      p_value < 0.05 & log2FC_prop > 0 ~ "↑ High_Risk (p<0.05)",
      p_value < 0.05 & log2FC_prop < 0 ~ "↑ Low_Risk (p<0.05)",
      TRUE ~ "NS"
    ),
    label = ifelse(p_value < 0.1, sprintf("C%s:%s", cluster_id, cell_type), "")
  )

p_prop_scatter <- ggplot(prop_scatter_df, aes(x = log2FC_prop, y = neg_log10_p, color = color)) +
  geom_point(size = 2.5) +
  geom_text_repel(aes(label = label), size = 2.2, max.overlaps = 20,
                  segment.size = 0.2, show.legend = FALSE) +
  scale_color_manual(values = c("↑ High_Risk (p<0.05)" = "#D32F2F",
                                "↑ Low_Risk (p<0.05)" = "#1976D2",
                                "NS" = "grey60"),
                     name = "") +
  guides(color = guide_legend(override.aes = list(size = 3))) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.3, color = "grey40") +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3, color = "grey40") +
  theme_paper(base_size = 8) +
  labs(x = "log2(FC) Proportion (High/Low Risk)",
       y = expression(-log[10](P~value)),
       title = "Cluster Proportion Differences") +
  theme(legend.position = "bottom", legend.text = element_text(size = 6))

ggsave(file.path(output_dir, "Fig4_Proportion_volcano.pdf"), p_prop_scatter,
       width = 130, height = 120, units = "mm")

# ===========================================================
# FINAL SUMMARY
# ===========================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║     04_Proportion_DEG_SignificantClusters — COMPLETE                 ║\n")
cat("╠══════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                      ║\n")
cat("║  Output directory:", output_dir, "\n")
cat("║                                                                      ║\n")
cat("║  Figures:                                                            ║\n")
cat("║    Fig1_Boxplot_significant_clusters.pdf                             ║\n")
cat("║    Fig2_DEG_count_barplot.pdf                                        ║\n")
cat("║    Fig3_DEG_heatmap.pdf                                              ║\n")
cat("║    Fig4_Proportion_volcano.pdf                                       ║\n")
cat("║    Volcano_C<N>_<celltype>.pdf (per cluster)                         ║\n")
cat("║    Boxplot_C<N>_<celltype>.pdf (per cluster)                         ║\n")
cat("║                                                                      ║\n")
cat("║  Tables:                                                             ║\n")
cat("║    Table_Wilcoxon_all_clusters.csv                                   ║\n")
cat("║    Table_DEG_summary.csv                                             ║\n")
cat("║    Table_DEG_combined.csv                                            ║\n")
cat("║    Table_DEG_C<N>_<celltype>.csv (per cluster)                       ║\n")
cat("║                                                                      ║\n")
cat("╚══════════════════════════════════════════════════════════════════════╝\n")

# Report key findings
cat("\n*** KEY FINDINGS ***\n")
cat("\n1. Clusters with significant proportion differences (p < 0.05):\n")
sig_prop <- stat_results %>% filter(p_value < 0.05)
if (nrow(sig_prop) > 0) {
  for (i in 1:nrow(sig_prop)) {
    cat(sprintf("   C%s (%s): p=%.4f, %s (FC=%.2f)\n",
                sig_prop$cluster_id[i], sig_prop$cell_type[i],
                sig_prop$p_value[i], sig_prop$direction[i], sig_prop$FC_HighvsLow[i]))
  }
} else {
  cat("   NONE at p<0.05 (small sample size limits power)\n")
  cat("   Trending clusters (p<0.1):\n")
  trend_prop <- stat_results %>% filter(p_value < 0.1)
  if (nrow(trend_prop) > 0) {
    for (i in 1:nrow(trend_prop)) {
      cat(sprintf("   C%s (%s): p=%.4f, %s (FC=%.2f)\n",
                  trend_prop$cluster_id[i], trend_prop$cell_type[i],
                  trend_prop$p_value[i], trend_prop$direction[i], trend_prop$FC_HighvsLow[i]))
    }
  }
}

cat("\n2. Clusters with most DEGs:\n")
top_deg_clusters <- deg_summary %>% filter(status == "completed") %>% head(5)
if (nrow(top_deg_clusters) > 0) {
  for (i in 1:nrow(top_deg_clusters)) {
    cat(sprintf("   C%s (%s): %d UP + %d DOWN = %d DEGs\n",
                top_deg_clusters$cluster_id[i], top_deg_clusters$cell_type[i],
                top_deg_clusters$n_up[i], top_deg_clusters$n_down[i],
                top_deg_clusters$total_DEG[i]))
  }
}

cat("\n3. Group-restricted clusters (biologically important):\n")
if (length(group_restricted) > 0) {
  for (cl in group_restricted) {
    ct <- cluster_annotation[cl]
    n_h <- sum(seu$RNA_snn_res.0.5 == cl & seu$group == "High_Risk")
    n_l <- sum(seu$RNA_snn_res.0.5 == cl & seu$group == "Low_Risk")
    dominant <- ifelse(n_h > n_l, "High_Risk", "Low_Risk")
    pct <- max(n_h, n_l) / (n_h + n_l) * 100
    cat(sprintf("   C%s (%s): %.1f%% %s (%d vs %d cells)\n",
                cl, ct, pct, dominant, max(n_h, n_l), min(n_h, n_l)))
  }
}

cat("\nNext steps:\n")
cat("  1. Review Table_Wilcoxon_all_clusters.csv for proportion findings\n")
cat("  2. Review Volcano plots for DE patterns\n")
cat("  3. Run GO/GSEA on significant DEG lists\n")
cat("  4. Focus on clusters with BOTH proportion shift AND many DEGs\n")

cat("\nDone.\n")