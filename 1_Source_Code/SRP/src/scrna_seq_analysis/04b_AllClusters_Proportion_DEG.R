# ============================================================
# 04b_AllClusters_Proportion_DEG_Fixed.R
# UMAP + Proportion (significant only) + DEG Volcano (fixed legend)
# ============================================================

options(Seurat.object.assay.version = "v3")
options(future.globals.maxSize = 8 * 1024^3)

library(Seurat)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(ggrepel)
library(patchwork)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/04_Differential/AllClusters"
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

# Cluster annotation mapping
cluster_annotation <- data.frame(
  cluster = as.character(0:20),
  cell_type = c(
    "CD8_T_effector",           # 0
    "CAF",                       # 1
    "Macrophage_SPP1",           # 2
    "Treg",                      # 3
    "Epithelial_basal",          # 4
    "Epithelial_squamous",       # 5
    "Neutrophil",                # 6
    "Fibroblast_normal",         # 7
    "B_cell",                    # 8
    "Proliferating",             # 9
    "Pericyte_SMC",              # 10
    "Plasma",                    # 11
    "Endothelial",               # 12
    "Myofibroblast",             # 13
    "DC_mature",                 # 14
    "Epithelial_differentiated", # 15
    "Epithelial_stem_like",      # 16
    "Mast_cell",                 # 17
    "Mesothelial",               # 18
    "Tumor_specific",            # 19
    "Activated_fibroblast"       # 20
  ),
  stringsAsFactors = FALSE
)

# ===========================================================
# PART 0: UMAP — Clusters and Annotated Cell Types
# ===========================================================
cat("\n━━━ PART 0: UMAP visualization ━━━\n")

# Ensure cell_type is mapped
if (!"cell_type" %in% colnames(seu@meta.data)) {
  seu$cell_type <- cluster_annotation$cell_type[match(as.character(seu$seurat_clusters), 
                                                      cluster_annotation$cluster)]
}

# --- UMAP by cluster number ---
Idents(seu) <- "seurat_clusters"
p_umap_clusters <- DimPlot(seu, group.by = "seurat_clusters", label = TRUE, 
                           label.size = 3, repel = TRUE, pt.size = 0.05) +
  theme_paper(base_size = 8) +
  ggtitle("Clusters") +
  theme(legend.position = "none")

# --- UMAP by annotated cell type ---
Idents(seu) <- "cell_type"

# Custom color palette for 21 cell types
cell_type_colors <- c(
  "CD8_T_effector" = "#E41A1C",
  "CAF" = "#377EB8",
  "Macrophage_SPP1" = "#4DAF4A",
  "Treg" = "#984EA3",
  "Epithelial_basal" = "#FF7F00",
  "Epithelial_squamous" = "#FFFF33",
  "Neutrophil" = "#A65628",
  "Fibroblast_normal" = "#F781BF",
  "B_cell" = "#999999",
  "Proliferating" = "#66C2A5",
  "Pericyte_SMC" = "#FC8D62",
  "Plasma" = "#8DA0CB",
  "Endothelial" = "#E78AC3",
  "Myofibroblast" = "#A6D854",
  "DC_mature" = "#FFD92F",
  "Epithelial_differentiated" = "#E5C494",
  "Epithelial_stem_like" = "#B3B3B3",
  "Mast_cell" = "#1B9E77",
  "Mesothelial" = "#D95F02",
  "Tumor_specific" = "#7570B3",
  "Activated_fibroblast" = "#E7298A"
)

p_umap_celltype <- DimPlot(seu, group.by = "cell_type", label = TRUE, 
                           label.size = 2.2, repel = TRUE, pt.size = 0.05,
                           cols = cell_type_colors) +
  theme_paper(base_size = 8) +
  ggtitle("Cell Type Annotation") +
  theme(legend.position = "none")

# --- UMAP by risk group ---
p_umap_group <- DimPlot(seu, group.by = "group", pt.size = 0.05,
                        cols = group_colors) +
  theme_paper(base_size = 8) +
  ggtitle("Risk Group") +
  theme(legend.position = "right",
        legend.text = element_text(size = 7))

# --- UMAP split by group ---
p_umap_split <- DimPlot(seu, group.by = "cell_type", split.by = "group",
                        pt.size = 0.02, cols = cell_type_colors) +
  theme_paper(base_size = 7) +
  ggtitle("Cell Types by Risk Group") +
  theme(legend.position = "none")

# Save UMAPs
ggsave(file.path(output_dir, "Fig1a_UMAP_clusters.pdf"), p_umap_clusters,
       width = 110, height = 100, units = "mm")
ggsave(file.path(output_dir, "Fig1b_UMAP_celltype.pdf"), p_umap_celltype,
       width = 130, height = 100, units = "mm")
ggsave(file.path(output_dir, "Fig1c_UMAP_group.pdf"), p_umap_group,
       width = 130, height = 100, units = "mm")
ggsave(file.path(output_dir, "Fig1d_UMAP_split_by_group.pdf"), p_umap_split,
       width = 240, height = 100, units = "mm")

# Combined UMAP panel (clusters + cell type + group)
p_umap_combined <- p_umap_clusters | p_umap_celltype | p_umap_group
ggsave(file.path(output_dir, "Fig1_UMAP_combined.pdf"), p_umap_combined,
       width = 350, height = 100, units = "mm")

# UMAP with legend on the side (for supplementary)
p_umap_legend <- DimPlot(seu, group.by = "cell_type", label = FALSE,
                         pt.size = 0.05, cols = cell_type_colors) +
  theme_paper(base_size = 8) +
  ggtitle("Cell Type Annotation") +
  theme(legend.position = "right",
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.2, "cm")) +
  guides(color = guide_legend(override.aes = list(size = 2), ncol = 1))

ggsave(file.path(output_dir, "Fig1_UMAP_celltype_with_legend.pdf"), p_umap_legend,
       width = 180, height = 120, units = "mm")

cat("  UMAPs saved.\n")

# ===========================================================
# PART 1: Proportion comparison for ALL clusters
# ===========================================================
cat("\n━━━ PART 1: Proportion comparison ━━━\n")

sample_group <- seu@meta.data %>%
  dplyr::select(sample, group) %>%
  distinct()

# Calculate proportion per sample per cluster
cluster_prop <- seu@meta.data %>%
  group_by(sample, group, seurat_clusters) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(sample, group) %>%
  mutate(total = sum(n), proportion = n / total * 100) %>%
  ungroup()

# Fill missing combinations with 0
all_combinations <- expand.grid(
  sample = unique(seu$sample),
  seurat_clusters = as.character(0:20),
  stringsAsFactors = FALSE
)
all_combinations <- left_join(all_combinations, sample_group, by = "sample")

cluster_prop_complete <- left_join(
  all_combinations,
  cluster_prop %>% 
    mutate(seurat_clusters = as.character(seurat_clusters)) %>%
    dplyr::select(sample, seurat_clusters, n, proportion),
  by = c("sample", "seurat_clusters")
) %>%
  mutate(
    n = replace_na(n, 0),
    proportion = replace_na(proportion, 0)
  )

cluster_prop_complete$group <- factor(cluster_prop_complete$group, levels = c("Low_Risk", "High_Risk"))

# Add annotation
cluster_prop_complete <- left_join(cluster_prop_complete, cluster_annotation,
                                   by = c("seurat_clusters" = "cluster"))

# --- Wilcoxon test (manual, avoids rstatix issues) ---
testable_clusters <- cluster_prop_complete %>%
  filter(proportion > 0) %>%
  group_by(seurat_clusters, group) %>%
  summarise(n_samples = n(), .groups = "drop") %>%
  pivot_wider(names_from = group, values_from = n_samples, values_fill = 0) %>%
  filter(Low_Risk >= 2 & High_Risk >= 2) %>%
  pull(seurat_clusters)

stat_results_list <- list()

for (cl in testable_clusters) {
  cl_data <- cluster_prop_complete %>% filter(seurat_clusters == cl)
  high_vals <- cl_data %>% filter(group == "High_Risk") %>% pull(proportion)
  low_vals <- cl_data %>% filter(group == "Low_Risk") %>% pull(proportion)
  
  wt <- tryCatch(
    wilcox.test(high_vals, low_vals, exact = FALSE),
    error = function(e) list(statistic = NA, p.value = NA)
  )
  
  ct_name <- cluster_annotation$cell_type[cluster_annotation$cluster == cl]
  
  stat_results_list[[cl]] <- data.frame(
    cluster = cl,
    cell_type = ct_name,
    n_High_Risk = length(high_vals),
    n_Low_Risk = length(low_vals),
    mean_High = mean(high_vals),
    mean_Low = mean(low_vals),
    FC_HighvsLow = mean(high_vals) / max(mean(low_vals), 0.001),
    statistic = as.numeric(wt$statistic),
    p_value = wt$p.value,
    stringsAsFactors = FALSE
  )
}

stat_df <- bind_rows(stat_results_list)
stat_df$p_adj <- p.adjust(stat_df$p_value, method = "BH")
stat_df$significance <- case_when(
  stat_df$p_value < 0.001 ~ "***",
  stat_df$p_value < 0.01 ~ "**",
  stat_df$p_value < 0.05 ~ "*",
  stat_df$p_value < 0.1 ~ ".",
  TRUE ~ "ns"
)
stat_df <- stat_df %>% arrange(p_value)

cat("\nWilcoxon results:\n")
print(stat_df %>% dplyr::select(cluster, cell_type, mean_High, mean_Low, p_value, significance),
      row.names = FALSE)

write.csv(stat_df, file.path(output_dir, "Table_Wilcoxon_all_clusters.csv"), row.names = FALSE)

# --- Identify significant clusters ---
sig_clusters <- stat_df %>% filter(p_value < 0.05) %>% pull(cluster)

cat("\n*** Significant clusters (p < 0.05):", 
    ifelse(length(sig_clusters) > 0, paste(sig_clusters, collapse = ", "), "NONE"), "\n")

# --- BOXPLOTS: Only for significant clusters ---
if (length(sig_clusters) > 0) {
  cat("\n  Plotting boxplots for significant clusters only...\n")
  
  plot_sig_boxplot <- function(data, cl, stat_info) {
    cl_data <- data %>% filter(seurat_clusters == cl)
    cl_stat <- stat_info %>% filter(cluster == cl)
    ct_name <- cl_stat$cell_type[1]
    
    y_max <- max(cl_data$proportion, na.rm = TRUE) * 1.25
    if (y_max == 0 | !is.finite(y_max)) y_max <- 1
    
    p_label <- cl_stat$significance[1]
    
    p <- ggplot(cl_data, aes(x = group, y = proportion, fill = group)) +
      geom_boxplot(width = 0.6, outlier.shape = NA, linewidth = 0.4, alpha = 0.7) +
      geom_jitter(width = 0.15, size = 1.8, alpha = 0.9, shape = 16) +
      scale_fill_manual(values = group_colors) +
      # P-value symbol only (no numeric p-value)
      annotate("text", x = 1.5, y = y_max, label = p_label, 
               size = 5, fontface = "bold") +
      annotate("segment", x = 1, xend = 2, y = y_max * 0.9, yend = y_max * 0.9,
               linewidth = 0.4) +
      theme_paper(base_size = 8) +
      labs(x = "", y = "Proportion (%)", 
           title = sprintf("C%s: %s", cl, ct_name)) +
      theme(legend.position = "none",
            plot.title = element_text(size = 8, face = "bold", hjust = 0.5))
    
    return(p)
  }
  
  sig_boxplots <- list()
  for (cl in sig_clusters) {
    sig_boxplots[[cl]] <- plot_sig_boxplot(cluster_prop_complete, cl, stat_df)
    ggsave(file.path(output_dir, sprintf("Boxplot_Cluster%s_significant.pdf", cl)), 
           sig_boxplots[[cl]], width = 55, height = 70, units = "mm")
  }
  
  # Combined significant boxplots
  ncol_sig <- min(length(sig_boxplots), 5)
  nrow_sig <- ceiling(length(sig_boxplots) / ncol_sig)
  
  ggsave(file.path(output_dir, "Fig2_Boxplot_significant_clusters.pdf"),
         wrap_plots(sig_boxplots, ncol = ncol_sig),
         width = 55 * ncol_sig, height = 70 * nrow_sig, units = "mm")
  
} else {
  cat("\n  No significant clusters found. Plotting top 5 trending clusters...\n")
  
  # If no p<0.05, show top 5 by p-value
  top5_clusters <- stat_df %>% head(5) %>% pull(cluster)
  
  plot_trending_boxplot <- function(data, cl, stat_info) {
    cl_data <- data %>% filter(seurat_clusters == cl)
    cl_stat <- stat_info %>% filter(cluster == cl)
    ct_name <- cl_stat$cell_type[1]
    
    y_max <- max(cl_data$proportion, na.rm = TRUE) * 1.25
    if (y_max == 0 | !is.finite(y_max)) y_max <- 1
    
    p_label <- cl_stat$significance[1]
    
    p <- ggplot(cl_data, aes(x = group, y = proportion, fill = group)) +
      geom_boxplot(width = 0.6, outlier.shape = NA, linewidth = 0.4, alpha = 0.7) +
      geom_jitter(width = 0.15, size = 1.8, alpha = 0.9, shape = 16) +
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
    
    return(p)
  }
  
  trending_boxplots <- list()
  for (cl in top5_clusters) {
    trending_boxplots[[cl]] <- plot_trending_boxplot(cluster_prop_complete, cl, stat_df)
  }
  
  ggsave(file.path(output_dir, "Fig2_Boxplot_top5_trending_clusters.pdf"),
         wrap_plots(trending_boxplots, ncol = min(5, length(trending_boxplots))),
         width = 55 * min(5, length(trending_boxplots)), height = 70, units = "mm")
}

  # ===========================================================
  # PART 2: DEG + VOLCANO (FIXED — no "a" in legend)
  # ===========================================================
  cat("\n━━━ PART 2: DEG + Volcano (fixed legend) ━━━\n")
  
  MAX_CELLS_PER_GROUP <- 5000
  
  # ═══════════════════════════════════════════════════════════
  # FIXED Volcano plot function
  # The "a" appears because geom_text_repel inherits the color
  # aesthetic and adds a letter "a" in the legend key.
  # FIX: set show.legend = FALSE in geom_text_repel
  # ═══════════════════════════════════════════════════════════
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
    
    # Cap extreme y values
    finite_y <- deg_df$neg_log10_padj[is.finite(deg_df$neg_log10_padj)]
    if (length(finite_y) > 0) {
      max_y <- quantile(finite_y, 0.99, na.rm = TRUE)
      deg_df$neg_log10_padj[deg_df$neg_log10_padj > max_y * 1.2] <- max_y * 1.2
    }
    
    # Top genes to label
    n_top_up <- as.integer(ceiling(top_n / 2))
    n_top_down <- as.integer(floor(top_n / 2))
    
    top_up <- deg_df %>% filter(color == "Up in High_Risk") %>%
      slice_max(n = n_top_up, order_by = avg_log2FC, with_ties = FALSE) %>% pull(gene)
    top_down <- deg_df %>% filter(color == "Up in Low_Risk") %>%
      slice_min(n = n_top_down, order_by = avg_log2FC, with_ties = FALSE) %>% pull(gene)
    top_genes <- c(top_up, top_down)
    
    deg_df$label <- ifelse(deg_df$gene %in% top_genes, deg_df$gene, "")
    
    n_up <- sum(deg_df$color == "Up in High_Risk")
    n_down <- sum(deg_df$color == "Up in Low_Risk")
    
    p <- ggplot(deg_df, aes(x = avg_log2FC, y = neg_log10_padj, color = color)) +
      # Points — these show in legend as round dots
      geom_point(size = 0.4, alpha = 0.5) +
      # Labels — KEY FIX: show.legend = FALSE prevents "a" in legend
      geom_text_repel(
        aes(label = label), 
        size = 2, 
        max.overlaps = 25,
        segment.size = 0.2, 
        segment.alpha = 0.5,
        show.legend = FALSE  # ← THIS FIXES THE "a" IN LEGEND
      ) +
      scale_color_manual(
        values = c("Up in High_Risk" = "#D32F2F", 
                   "Up in Low_Risk" = "#1976D2", 
                   "NS" = "grey75"),
        name = ""
      ) +
      # Ensure legend shows round dots (not "a")
      guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
      geom_vline(xintercept = c(-fc_cutoff, fc_cutoff), 
                 linetype = "dashed", linewidth = 0.25, color = "grey40") +
      geom_hline(yintercept = -log10(padj_cutoff), 
                 linetype = "dashed", linewidth = 0.25, color = "grey40") +
      theme_paper(base_size = 7) +
      labs(x = "log2FC (High_Risk vs Low_Risk)", 
           y = expression(-log[10](adjusted~P~value)),
           title = title,
           subtitle = sprintf("Up in High: %d | Up in Low: %d", n_up, n_down)) +
      theme(
        legend.position = "bottom",
        legend.text = element_text(size = 6),
        plot.subtitle = element_text(size = 5.5, hjust = 0.5)
      )
    
    return(p)
  }
  
  # ----------------------------------------------------------
  # Run DEG for each cluster
  # ----------------------------------------------------------
  deg_all_results <- list()
  deg_summary_list <- list()
  
  for (cl in as.character(0:20)) {
    ct_name <- cluster_annotation$cell_type[cluster_annotation$cluster == cl]
    cat(sprintf("\n  Cluster %s (%s):\n", cl, ct_name))
    
    sub <- subset(seu, seurat_clusters == cl)
    n_high <- sum(sub$group == "High_Risk")
    n_low <- sum(sub$group == "Low_Risk")
    cat(sprintf("    Cells: High=%d, Low=%d\n", n_high, n_low))
    
    # Need >= 30 cells in each group
    if (n_high < 30 | n_low < 30) {
      cat("    SKIPPED: insufficient cells\n")
      deg_summary_list[[cl]] <- data.frame(
        cluster = cl, cell_type = ct_name,
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
    
    deg <- tryCatch({
      FindMarkers(sub, ident.1 = "High_Risk", ident.2 = "Low_Risk",
                  min.pct = 0.1, logfc.threshold = 0.1, verbose = FALSE)
    }, error = function(e) {
      cat(sprintf("    ERROR: %s\n", e$message))
      return(NULL)
    })
    
    if (!is.null(deg) && nrow(deg) > 0) {
      deg$gene <- rownames(deg)
      deg$cluster <- cl
      deg$cell_type <- ct_name
      
      n_up <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC > 0.5)
      n_down <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC < -0.5)
      cat(sprintf("    DEG: UP=%d, DOWN=%d, Total=%d\n", n_up, n_down, nrow(deg)))
      
      deg_all_results[[cl]] <- deg
      
      # Save DEG table
      write.csv(deg, file.path(output_dir, sprintf("Table_DEG_C%s_%s.csv", cl, ct_name)), 
                row.names = FALSE)
      
      # Volcano plot
      if (n_up + n_down > 0 | nrow(deg) > 50) {
        p_vol <- plot_volcano(deg, sprintf("C%s: %s", cl, ct_name), top_n = 20)
        ggsave(file.path(output_dir, sprintf("Volcano_C%s_%s.pdf", cl, ct_name)), 
               p_vol, width = 100, height = 100, units = "mm")
      }
      
      deg_summary_list[[cl]] <- data.frame(
        cluster = cl, cell_type = ct_name,
        n_High = n_high, n_Low = n_low,
        n_up = n_up, n_down = n_down, total_DEG = n_up + n_down,
        status = "completed",
        stringsAsFactors = FALSE
      )
    } else {
      deg_summary_list[[cl]] <- data.frame(
        cluster = cl, cell_type = ct_name,
        n_High = n_high, n_Low = n_low,
        n_up = 0, n_down = 0, total_DEG = 0,
        status = "no_results",
        stringsAsFactors = FALSE
      )
    }
    
    rm(sub); gc(verbose = FALSE)
  }
  
  # DEG summary
  deg_summary <- bind_rows(deg_summary_list) %>% arrange(desc(total_DEG))
  
  cat("\n╔══════════════════════════════════════════════════════════════════╗\n")
  cat("║                    DEG SUMMARY (ALL CLUSTERS)                    ║\n")
  cat("╠══════════════════════════════════════════════════════════════════╣\n")
  print(deg_summary, row.names = FALSE)
  cat("╚══════════════════════════════════════════════════════════════════╝\n")
  
  write.csv(deg_summary, file.path(output_dir, "Table_DEG_summary_all_clusters.csv"), row.names = FALSE)
  saveRDS(deg_all_results, file.path(output_dir, "deg_results_all_clusters.rds"))
  
  # Combined DEG table
  if (length(deg_all_results) > 0) {
    all_deg_combined <- bind_rows(deg_all_results)
    write.csv(all_deg_combined, file.path(output_dir, "Table_DEG_combined_all_clusters.csv"), row.names = FALSE)
  }
  
  # ----------------------------------------------------------
  # DEG count barplot
  # ----------------------------------------------------------
  deg_count_df <- deg_summary %>%
    filter(status == "completed" & total_DEG > 0) %>%
    pivot_longer(cols = c(n_up, n_down), names_to = "direction", values_to = "count") %>%
    mutate(
      direction = ifelse(direction == "n_up", "Up in High_Risk", "Down in High_Risk"),
      count_plot = ifelse(direction == "Down in High_Risk", -count, count),
      label = sprintf("C%s: %s", cluster, cell_type)
    )
  
  if (nrow(deg_count_df) > 0) {
    cluster_order <- deg_summary %>%
      filter(status == "completed" & total_DEG > 0) %>%
      arrange(desc(total_DEG)) %>%
      mutate(label = sprintf("C%s: %s", cluster, cell_type)) %>%
      pull(label)
    
    deg_count_df$label <- factor(deg_count_df$label, levels = rev(cluster_order))
    
    p_deg_bar <- ggplot(deg_count_df, aes(x = count_plot, y = label, fill = direction)) +
      geom_col(width = 0.7) +
      geom_vline(xintercept = 0, linewidth = 0.3) +
      scale_fill_manual(values = c("Up in High_Risk" = "#D32F2F", 
                                   "Down in High_Risk" = "#1976D2")) +
      theme_paper(base_size = 7) +
      labs(x = "Number of DEGs (padj<0.05, |log2FC|>0.5)", y = "", fill = "",
           title = "DEGs per Cluster (High_Risk vs Low_Risk)") +
      theme(axis.text.y = element_text(size = 5.5),
            legend.position = "bottom",
            legend.text = element_text(size = 6))
    
    ggsave(file.path(output_dir, "Fig3_DEG_count_per_cluster.pdf"), p_deg_bar,
           width = 140, height = max(80, length(cluster_order) * 6), units = "mm")
  }
  
  # ===========================================================
  # FINAL SUMMARY
  # ===========================================================
  cat("\n\n")
  cat("╔══════════════════════════════════════════════════════════════════╗\n")
  cat("║     04b_AllClusters_Proportion_DEG_Fixed — COMPLETE              ║\n")
  cat("╠══════════════════════════════════════════════════════════════════╣\n")
  cat("║                                                                  ║\n")
  cat("║  Output directory:", output_dir, "\n")
  cat("║                                                                  ║\n")
  cat("║  Figures:                                                        ║\n")
  cat("║    Fig1a_UMAP_clusters.pdf                                       ║\n")
  cat("║    Fig1b_UMAP_celltype.pdf                                       ║\n")
  cat("║    Fig1c_UMAP_group.pdf                                          ║\n")
  cat("║    Fig1d_UMAP_split_by_group.pdf                                 ║\n")
  cat("║    Fig1_UMAP_combined.pdf                                        ║\n")
  cat("║    Fig1_UMAP_celltype_with_legend.pdf                            ║\n")
  cat("║    Fig2_Boxplot_significant_clusters.pdf (sig only)              ║\n")
  cat("║    Fig3_DEG_count_per_cluster.pdf                                ║\n")
  cat("║    Volcano_C<N>_<celltype>.pdf (per cluster, fixed legend)       ║\n")
  cat("║                                                                  ║\n")
  cat("║  Tables:                                                         ║\n")
  cat("║    Table_Wilcoxon_all_clusters.csv                               ║\n")
  cat("║    Table_DEG_summary_all_clusters.csv                            ║\n")
  cat("║    Table_DEG_C<N>_<celltype>.csv (per cluster)                   ║\n")
  cat("║    Table_DEG_combined_all_clusters.csv                           ║\n")
  cat("║                                                                  ║\n")
  cat("║  Key fixes:                                                      ║\n")
  cat("║    ✓ Volcano legend: round dots (not 'a')                        ║\n")
  cat("║    ✓ Boxplot: only significant clusters plotted                  ║\n")
  cat("║    ✓ UMAP: clusters + cell types + groups                        ║\n")
  cat("║    ✓ Wilcoxon: handles missing proportions (fill with 0)         ║\n")
  cat("║                                                                  ║\n")
  cat("╚══════════════════════════════════════════════════════════════════╝\n")
  
  cat("\nDone.\n")
  