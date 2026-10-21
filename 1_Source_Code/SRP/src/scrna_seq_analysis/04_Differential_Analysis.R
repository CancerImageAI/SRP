# ============================================================
# 04_Differential_Analysis.R — REVISED (memory-optimized)
# ============================================================

options(Seurat.object.assay.version = "v3")
options(future.globals.maxSize = 8 * 1024^3)  # 8 GB

library(Seurat)
library(tidyverse)
library(patchwork)
library(ComplexHeatmap)
library(circlize)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(msigdbr)
library(ggrepel)
library(ggpubr)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/04_Differential"
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
# PART 1: Load annotated data
# ----------------------------------------------------------
cat("━━━ PART 1: Loading data ━━━\n")
seu <- readRDS(file.path(input_dir, "seu_annotated.rds"))
Idents(seu) <- "cell_type"
cat("Loaded:", ncol(seu), "cells,", length(unique(seu$cell_type)), "cell types\n")
cat("Groups:", table(seu$group), "\n")

# ----------------------------------------------------------
# PART 2: DEG analysis (High_Risk vs Low_Risk per cell type)
# ----------------------------------------------------------
cat("\n━━━ PART 2: DEG analysis (High_Risk vs Low_Risk per cell type) ━━━\n")

ct_counts <- seu@meta.data %>%
  group_by(cell_type, group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = group, values_from = n, values_fill = 0) %>%
  mutate(min_cells = pmin(High_Risk, Low_Risk))

valid_cts <- ct_counts %>% filter(min_cells >= 50) %>% pull(cell_type)
cat("Cell types with >=50 cells per group:", length(valid_cts), "\n")

MAX_CELLS_PER_GROUP <- 5000

deg_results <- list()
for (ct in valid_cts) {
  cat("  DEG:", ct, "...")
  sub <- subset(seu, cell_type == ct)
  
  n_high <- sum(sub$group == "High_Risk")
  n_low <- sum(sub$group == "Low_Risk")
  
  if (n_high > MAX_CELLS_PER_GROUP | n_low > MAX_CELLS_PER_GROUP) {
    set.seed(42)
    cells_high <- WhichCells(sub, expression = group == "High_Risk")
    cells_low <- WhichCells(sub, expression = group == "Low_Risk")
    
    if (length(cells_high) > MAX_CELLS_PER_GROUP) {
      cells_high <- sample(cells_high, MAX_CELLS_PER_GROUP)
    }
    if (length(cells_low) > MAX_CELLS_PER_GROUP) {
      cells_low <- sample(cells_low, MAX_CELLS_PER_GROUP)
    }
    
    sub <- subset(sub, cells = c(cells_high, cells_low))
    cat(sprintf(" [downsampled to %d+%d]", length(cells_high), length(cells_low)), "")
  }
  
  Idents(sub) <- "group"
  
  deg <- tryCatch({
    FindMarkers(sub, ident.1 = "High_Risk", ident.2 = "Low_Risk",
                min.pct = 0.1, logfc.threshold = 0.1, verbose = FALSE)
  }, error = function(e) {
    cat(sprintf(" FAILED (%s)\n", e$message))
    return(NULL)
  })
  
  if (!is.null(deg)) {
    deg$gene <- rownames(deg)
    deg$cell_type <- ct
    deg$significance <- case_when(
      deg$p_val_adj < 0.05 & abs(deg$avg_log2FC) > 0.5 ~ "Significant",
      TRUE ~ "NS"
    )
    deg_results[[ct]] <- deg
    n_up <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC > 0.5)
    n_down <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC < -0.5)
    cat(sprintf(" UP:%d DOWN:%d\n", n_up, n_down))
  }
  
  rm(sub); gc(verbose = FALSE)
}

all_deg <- bind_rows(deg_results)
write.csv(all_deg, file.path(output_dir, "Table_DEG_all_celltypes.csv"), row.names = FALSE)

deg_summary <- all_deg %>%
  filter(p_val_adj < 0.05 & abs(avg_log2FC) > 0.5) %>%
  group_by(cell_type) %>%
  summarise(
    n_up_in_HighRisk = sum(avg_log2FC > 0.5),
    n_down_in_HighRisk = sum(avg_log2FC < -0.5),
    total_DEG = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_DEG))

write.csv(deg_summary, file.path(output_dir, "Table_DEG_summary.csv"), row.names = FALSE)
cat("\nDEG summary:\n")
print(as.data.frame(deg_summary))

# ----------------------------------------------------------
# PART 3: Volcano plots
# ----------------------------------------------------------
cat("\n━━━ PART 3: Volcano plots ━━━\n")

plot_volcano <- function(deg_df, title, top_n = 15) {
  deg_df <- deg_df %>%
    mutate(
      color = case_when(
        p_val_adj < 0.05 & avg_log2FC > 0.5 ~ "Up_HighRisk",
        p_val_adj < 0.05 & avg_log2FC < -0.5 ~ "Up_LowRisk",
        TRUE ~ "NS"
      ),
      neg_log10_padj = -log10(p_val_adj + 1e-300)
    )
  
  top_genes <- deg_df %>%
    filter(color != "NS") %>%
    slice_max(n = top_n, order_by = abs(avg_log2FC)) %>%
    pull(gene)
  
  deg_df$label <- ifelse(deg_df$gene %in% top_genes, deg_df$gene, "")
  
  ggplot(deg_df, aes(x = avg_log2FC, y = neg_log10_padj, color = color)) +
    geom_point(size = 0.5, alpha = 0.6) +
    geom_text_repel(aes(label = label), size = 2, max.overlaps = 20,
                    segment.size = 0.2) +
    scale_color_manual(values = c("Up_HighRisk" = "#D32F2F", 
                                  "Up_LowRisk" = "#1976D2", 
                                  "NS" = "grey70")) +
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", linewidth = 0.3) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.3) +
    theme_paper() +
    labs(x = "log2FC (High_Risk / Low_Risk)", y = "-log10(p_adj)",
         title = title, color = "") +
    theme(legend.position = "none")
}

volcano_cts <- intersect(c("CD8_T_effector", "Treg", "Macrophage_SPP1", 
                           "CAF", "Neutrophil", "Fibroblast_normal",
                           "Epithelial_basal", "Epithelial_squamous"), names(deg_results))

volcano_plots <- list()
for (ct in volcano_cts) {
  volcano_plots[[ct]] <- plot_volcano(deg_results[[ct]], ct)
}

if (length(volcano_plots) > 0) {
  ncol_v <- min(4, length(volcano_plots))
  nrow_v <- ceiling(length(volcano_plots) / ncol_v)
  ggsave(file.path(output_dir, "Fig1_Volcano_major_celltypes.pdf"),
         wrap_plots(volcano_plots, ncol = ncol_v),
         width = 80 * ncol_v, height = 80 * nrow_v, units = "mm")
}

# ----------------------------------------------------------
# PART 4: Focus cluster differential analysis
# ----------------------------------------------------------
cat("\n━━━ PART 4: Focus clusters ━━━\n")

# 4A: Myofibroblast vs CAF
cat("  4A: Myofibroblast vs CAF\n")
stromal_sub <- subset(seu, cell_type %in% c("Myofibroblast", "CAF", "Fibroblast_normal"))
Idents(stromal_sub) <- "cell_type"

deg_myofib_vs_caf <- FindMarkers(stromal_sub, 
                                 ident.1 = "Myofibroblast", 
                                 ident.2 = "CAF",
                                 min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_myofib_vs_caf$gene <- rownames(deg_myofib_vs_caf)
deg_myofib_vs_caf$comparison <- "Myofibroblast_vs_CAF"

deg_myofib_vs_fib <- FindMarkers(stromal_sub, 
                                 ident.1 = "Myofibroblast", 
                                 ident.2 = "Fibroblast_normal",
                                 min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_myofib_vs_fib$gene <- rownames(deg_myofib_vs_fib)
deg_myofib_vs_fib$comparison <- "Myofibroblast_vs_Fibroblast"

write.csv(rbind(deg_myofib_vs_caf, deg_myofib_vs_fib),
          file.path(output_dir, "Table_DEG_Myofibroblast_comparisons.csv"), row.names = FALSE)
rm(stromal_sub); gc(verbose = FALSE)

# 4B: Mesothelial markers (use small subset to avoid memory issue)
cat("  4B: Mesothelial markers\n")
meso_sub <- subset(seu, cell_type %in% c("Mesothelial", "Endothelial", "Pericyte_SMC"))
Idents(meso_sub) <- "cell_type"

deg_meso_markers <- FindMarkers(meso_sub, 
                                ident.1 = "Mesothelial", 
                                ident.2 = NULL,
                                min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_meso_markers$gene <- rownames(deg_meso_markers)
write.csv(deg_meso_markers, file.path(output_dir, "Table_DEG_Mesothelial_markers.csv"), row.names = FALSE)
rm(meso_sub); gc(verbose = FALSE)

# 4C: Tumor_specific vs other epithelial
cat("  4C: Tumor_specific vs other epithelial\n")
epi_sub <- subset(seu, cell_type %in% c("Tumor_specific", "Epithelial_basal", 
                                        "Epithelial_squamous", "Epithelial_differentiated"))
Idents(epi_sub) <- "cell_type"

deg_tumor_spec <- FindMarkers(epi_sub, 
                              ident.1 = "Tumor_specific", 
                              ident.2 = NULL,
                              min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_tumor_spec$gene <- rownames(deg_tumor_spec)
write.csv(deg_tumor_spec, file.path(output_dir, "Table_DEG_Tumor_specific_markers.csv"), row.names = FALSE)
rm(epi_sub); gc(verbose = FALSE)

# Volcano plots for focus comparisons
p_v_myofib <- plot_volcano(deg_myofib_vs_caf, "Myofibroblast vs CAF")
p_v_meso <- plot_volcano(deg_meso_markers, "Mesothelial (unique markers)")
p_v_tumor <- plot_volcano(deg_tumor_spec, "Tumor_specific vs Epithelial")

ggsave(file.path(output_dir, "Fig2_Volcano_focus_clusters.pdf"),
       p_v_myofib | p_v_meso | p_v_tumor,
       width = 250, height = 90, units = "mm")

# ----------------------------------------------------------
# PART 5: Pathway analysis (GO and GSEA)
# ----------------------------------------------------------
cat("\n━━━ PART 5: Pathway analysis ━━━\n")

run_GO <- function(genes, title = "GO") {
  gene_ids <- tryCatch(
    bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db),
    error = function(e) data.frame(SYMBOL = character(), ENTREZID = character())
  )
  if (nrow(gene_ids) < 5) {
    cat("    Too few genes for GO:", title, "\n")
    return(NULL)
  }
  ego <- enrichGO(gene = gene_ids$ENTREZID,
                  OrgDb = org.Hs.eg.db,
                  ont = "BP",
                  pAdjustMethod = "BH",
                  pvalueCutoff = 0.05,
                  qvalueCutoff = 0.1,
                  readable = TRUE)
  return(ego)
}

run_GSEA <- function(deg_df, title = "GSEA") {
  deg_ranked <- deg_df %>%
    filter(!is.na(avg_log2FC) & !is.na(p_val)) %>%
    mutate(rank_metric = avg_log2FC * (-log10(p_val + 1e-300))) %>%
    arrange(desc(rank_metric))
  
  gene_list <- deg_ranked$rank_metric
  names(gene_list) <- deg_ranked$gene
  gene_list <- gene_list[!duplicated(names(gene_list))]
  gene_list <- sort(gene_list, decreasing = TRUE)
  
  hallmark <- msigdbr(species = "Homo sapiens", category = "H") %>%
    dplyr::select(gs_name, gene_symbol)
  
  gsea_res <- tryCatch(
    GSEA(gene_list, TERM2GENE = hallmark, pvalueCutoff = 0.25,
         verbose = FALSE, eps = 1e-100),
    error = function(e) NULL
  )
  return(gsea_res)
}

# 5A: GO for focus clusters
cat("  GO enrichment for focus clusters...\n")

myofib_up_genes <- deg_myofib_vs_caf %>%
  filter(p_val_adj < 0.05 & avg_log2FC > 0.5) %>%
  pull(gene)
go_myofib <- run_GO(myofib_up_genes, title = "Myofibroblast_up_vs_CAF")
if (!is.null(go_myofib) && nrow(as.data.frame(go_myofib)) > 0) {
  write.csv(as.data.frame(go_myofib), 
            file.path(output_dir, "Table_GO_Myofibroblast_up.csv"), row.names = FALSE)
  p_go_myofib <- dotplot(go_myofib, showCategory = 15) + 
    theme_paper() + ggtitle("Myofibroblast (up vs CAF) - GO BP")
  ggsave(file.path(output_dir, "Fig3A_GO_Myofibroblast.pdf"), p_go_myofib,
         width = 180, height = 150, units = "mm")
}

meso_up_genes <- deg_meso_markers %>%
  filter(p_val_adj < 0.05 & avg_log2FC > 1) %>%
  pull(gene)
go_meso <- run_GO(meso_up_genes, title = "Mesothelial_markers")
if (!is.null(go_meso) && nrow(as.data.frame(go_meso)) > 0) {
  write.csv(as.data.frame(go_meso), 
            file.path(output_dir, "Table_GO_Mesothelial.csv"), row.names = FALSE)
  p_go_meso <- dotplot(go_meso, showCategory = 15) + 
    theme_paper() + ggtitle("Mesothelial - GO BP")
  ggsave(file.path(output_dir, "Fig3B_GO_Mesothelial.pdf"), p_go_meso,
         width = 180, height = 150, units = "mm")
}

tumor_up_genes <- deg_tumor_spec %>%
  filter(p_val_adj < 0.05 & avg_log2FC > 1) %>%
  pull(gene)
go_tumor <- run_GO(tumor_up_genes, title = "Tumor_specific_markers")
if (!is.null(go_tumor) && nrow(as.data.frame(go_tumor)) > 0) {
  write.csv(as.data.frame(go_tumor), 
            file.path(output_dir, "Table_GO_Tumor_specific.csv"), row.names = FALSE)
  p_go_tumor <- dotplot(go_tumor, showCategory = 15) + 
    theme_paper() + ggtitle("Tumor_specific - GO BP")
  ggsave(file.path(output_dir, "Fig3C_GO_Tumor_specific.pdf"), p_go_tumor,
         width = 180, height = 150, units = "mm")
}

# 5B: GSEA for major cell types
cat("  GSEA Hallmark for major cell types...\n")
gsea_results <- list()
gsea_cts <- intersect(c("CD8_T_effector", "Treg", "Macrophage_SPP1", "CAF", 
                        "Neutrophil", "Epithelial_basal"), names(deg_results))

for (ct in gsea_cts) {
  cat("    GSEA:", ct, "\n")
  gsea_res <- run_GSEA(deg_results[[ct]], title = ct)
  if (!is.null(gsea_res) && nrow(as.data.frame(gsea_res)) > 0) {
    gsea_df <- as.data.frame(gsea_res)
    gsea_df$cell_type <- ct
    gsea_results[[ct]] <- gsea_df
  }
}

if (length(gsea_results) > 0) {
  all_gsea <- bind_rows(gsea_results)
  write.csv(all_gsea, file.path(output_dir, "Table_GSEA_Hallmark_all_celltypes.csv"), row.names = FALSE)
  
  gsea_nes <- all_gsea %>%
    dplyr::select(ID, NES, cell_type) %>%
    pivot_wider(names_from = cell_type, values_from = NES, values_fill = 0) %>%
    column_to_rownames("ID")
  
  sig_pathways <- all_gsea %>% filter(p.adjust < 0.05) %>% pull(ID) %>% unique()
  
  if (length(sig_pathways) > 0) {
    gsea_nes_filt <- as.matrix(gsea_nes[rownames(gsea_nes) %in% sig_pathways, , drop = FALSE])
    
    pdf(file.path(output_dir, "Fig4_GSEA_Hallmark_heatmap.pdf"), width = 10, height = 8)
    ht <- Heatmap(gsea_nes_filt,
                  name = "NES",
                  col = colorRamp2(c(-3, 0, 3), c("#2166AC", "white", "#B2182B")),
                  cluster_rows = TRUE,
                  cluster_columns = TRUE,
                  show_row_names = TRUE,
                  show_column_names = TRUE,
                  row_names_gp = gpar(fontsize = 6),
                  column_names_gp = gpar(fontsize = 7, fontface = "bold"),
                  column_names_rot = 45,
                  column_title = "GSEA Hallmark NES: High_Risk vs Low_Risk",
                  column_title_gp = gpar(fontsize = 10, fontface = "bold"),
                  heatmap_legend_param = list(
                    title_gp = gpar(fontsize = 7),
                    labels_gp = gpar(fontsize = 6)
                  ))
    draw(ht)
    dev.off()
  }
}

# ----------------------------------------------------------
# PART 6: Focus cluster characterization (memory-optimized)
# ----------------------------------------------------------
cat("\n━━━ PART 6: Focus cluster characterization ━━━\n")

# 6A: Myofibroblast signature vs all stromal (use subset, not full object)
cat("  6A: Myofibroblast signature\n")
stromal_all <- subset(seu, major_lineage == "Stromal")
Idents(stromal_all) <- "cell_type"
deg_myofib_all <- FindMarkers(stromal_all, 
                              ident.1 = "Myofibroblast", 
                              ident.2 = NULL,
                              min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_myofib_all$gene <- rownames(deg_myofib_all)
write.csv(deg_myofib_all, file.path(output_dir, "Table_DEG_Myofibroblast_vs_allStromal.csv"), row.names = FALSE)
rm(stromal_all); gc(verbose = FALSE)

myofib_sig_genes <- deg_myofib_all %>%
  filter(p_val_adj < 0.05 & avg_log2FC > 0.5) %>%
  pull(gene)
cat("    Myofibroblast signature genes:", length(myofib_sig_genes), "\n")

go_myofib_sig <- run_GO(myofib_sig_genes, title = "Myofibroblast_signature")
if (!is.null(go_myofib_sig) && nrow(as.data.frame(go_myofib_sig)) > 0) {
  write.csv(as.data.frame(go_myofib_sig), 
            file.path(output_dir, "Table_GO_Myofibroblast_signature.csv"), row.names = FALSE)
  p_go_ms <- dotplot(go_myofib_sig, showCategory = 20) + 
    theme_paper() + ggtitle("Myofibroblast Signature - GO BP")
  ggsave(file.path(output_dir, "Fig5A_GO_Myofibroblast_signature.pdf"), p_go_ms,
         width = 180, height = 160, units = "mm")
}

# 6B: Mesothelial signature (FIX: use small subset to avoid memory error)
cat("  6B: Mesothelial signature\n")
stromal_for_meso <- subset(seu, cell_type %in% c("Mesothelial", "Endothelial", 
                                                 "Pericyte_SMC", "Fibroblast_normal"))
Idents(stromal_for_meso) <- "cell_type"
deg_meso_global <- FindMarkers(stromal_for_meso, 
                               ident.1 = "Mesothelial", 
                               ident.2 = NULL,
                               min.pct = 0.1, logfc.threshold = 0.5, verbose = FALSE)
deg_meso_global$gene <- rownames(deg_meso_global)
write.csv(deg_meso_global, file.path(output_dir, "Table_DEG_Mesothelial_vs_otherStromal.csv"), row.names = FALSE)
rm(stromal_for_meso); gc(verbose = FALSE)

meso_sig_genes <- deg_meso_global %>%
  filter(p_val_adj < 0.05 & avg_log2FC > 1) %>%
  pull(gene)
cat("    Mesothelial signature genes:", length(meso_sig_genes), "\n")

go_meso_sig <- run_GO(meso_sig_genes, title = "Mesothelial_signature")
if (!is.null(go_meso_sig) && nrow(as.data.frame(go_meso_sig)) > 0) {
  write.csv(as.data.frame(go_meso_sig), 
            file.path(output_dir, "Table_GO_Mesothelial_signature.csv"), row.names = FALSE)
  p_go_meso_s <- dotplot(go_meso_sig, showCategory = 20) + 
    theme_paper() + ggtitle("Mesothelial Signature - GO BP")
  ggsave(file.path(output_dir, "Fig5B_GO_Mesothelial_signature.pdf"), p_go_meso_s,
         width = 180, height = 160, units = "mm")
}

# 6C: Tumor_specific signature
cat("  6C: Tumor_specific signature\n")
epi_all <- subset(seu, major_lineage == "Epithelial")
Idents(epi_all) <- "cell_type"
deg_tumor_global <- FindMarkers(epi_all, 
                                ident.1 = "Tumor_specific", 
                                ident.2 = NULL,
                                min.pct = 0.1, logfc.threshold = 0.5, verbose = FALSE)
deg_tumor_global$gene <- rownames(deg_tumor_global)
write.csv(deg_tumor_global, file.path(output_dir, "Table_DEG_Tumor_specific_vs_allEpithelial.csv"), row.names = FALSE)
rm(epi_all); gc(verbose = FALSE)

tumor_sig_genes <- deg_tumor_global %>%
  filter(p_val_adj < 0.05 & avg_log2FC > 1) %>%
  pull(gene)
cat("    Tumor_specific signature genes:", length(tumor_sig_genes), "\n")

go_tumor_sig <- run_GO(tumor_sig_genes, title = "Tumor_specific_signature")
if (!is.null(go_tumor_sig) && nrow(as.data.frame(go_tumor_sig)) > 0) {
  write.csv(as.data.frame(go_tumor_sig), 
            file.path(output_dir, "Table_GO_Tumor_specific_signature.csv"), row.names = FALSE)
  p_go_ts <- dotplot(go_tumor_sig, showCategory = 20) + 
    theme_paper() + ggtitle("Tumor_specific Signature - GO BP")
  ggsave(file.path(output_dir, "Fig5C_GO_Tumor_specific_signature.pdf"), p_go_ts,
         width = 180, height = 160, units = "mm")
}

# ----------------------------------------------------------
# PART 7: Optimized marker selection for focus clusters
# ----------------------------------------------------------
cat("\n━━━ PART 7: Optimal marker selection ━━━\n")

# Based on Table_top20_markers_per_cluster.csv analysis:
# Select markers with: high log2FC, high specificity (low pct.2), biological relevance

# MYOFIBROBLAST (Cluster 13) — optimal markers
myofib_optimal <- data.frame(
  gene = c("POSTN", "GPC6", "KCND2", "SGCD", "PRKG1", "PDZRN3", 
           "NAV3", "ADAMTS12", "ROBO2", "COL5A3", "WNT5A", "GUCY1A1"),
  avg_log2FC = c(2.25, 2.12, 2.27, 2.25, 2.06, 2.05, 
                 2.20, 2.18, 2.16, 2.15, 2.08, 2.31),
  pct1 = c(0.911, 0.844, 0.642, 0.508, 0.823, 0.594,
           0.484, 0.457, 0.375, 0.431, 0.307, 0.468),
  pct2 = c(0.179, 0.200, 0.118, 0.119, 0.196, 0.151,
           0.111, 0.105, 0.083, 0.093, 0.096, 0.102),
  specificity_ratio = c(5.09, 4.22, 5.44, 4.27, 4.20, 3.93,
                        4.36, 4.35, 4.52, 4.63, 3.20, 4.59),
  category = c("ECM", "ECM/Proteoglycan", "Ion_channel", "Cytoskeleton", 
               "cGMP_signaling", "PDZ_signaling",
               "Cell_migration", "ECM_remodeling", "Axon_guidance/Tumor_suppression", 
               "Fibrillar_collagen", "WNT_signaling", "NO_pathway"),
  biological_significance = c(
    "Periostin - matrix remodeling, tumor stroma interaction",
    "Glypican-6 - growth factor binding, WNT co-receptor",
    "Potassium channel - contractile myofibroblast phenotype",
    "Dystrophin-associated glycoprotein - mechanical sensing",
    "cGMP-dependent kinase - vasodilation, smooth muscle relaxation",
    "PDZ domain - protein-protein interaction scaffolding",
    "Neuron navigator - directed cell migration",
    "ADAMTS metalloproteinase - collagen processing/remodeling",
    "Slit-Robo pathway - tumor suppression, migration inhibition",
    "Type V collagen - fibrillar matrix assembly",
    "Non-canonical WNT - cell polarity, migration",
    "Guanylate cyclase - nitric oxide/cGMP signaling"
  ),
  stringsAsFactors = FALSE
)

# MESOTHELIAL (Cluster 18) — optimal markers
meso_optimal <- data.frame(
  gene = c("HP", "UPK3B", "PRG4", "MSLN", "TFPI2", "ALOX15", 
           "HAS1", "CFB", "VTN", "BCHE", "CPA4", "SYT4"),
  avg_log2FC = c(9.02, 8.47, 8.34, 8.28, 7.46, 7.59, 
                 6.55, 7.05, 7.82, 7.06, 6.73, 9.06),
  pct1 = c(0.874, 0.570, 0.832, 0.520, 0.661, 0.433,
           0.724, 0.906, 0.301, 0.498, 0.612, 0.402),
  pct2 = c(0.011, 0.003, 0.015, 0.002, 0.007, 0.003,
           0.014, 0.029, 0.001, 0.005, 0.010, 0.001),
  specificity_ratio = c(79.5, 190.0, 55.5, 260.0, 94.4, 144.3,
                        51.7, 31.2, 301.0, 99.6, 61.2, 402.0),
  category = c("Acute_phase", "Uroplakin", "Lubricant", "Surface_antigen", 
               "Protease_inhibitor", "Lipid_metabolism",
               "HA_synthesis", "Complement", "ECM_adhesion", 
               "Cholinesterase", "Protease", "Vesicle_fusion"),
  biological_significance = c(
    "Haptoglobin - acute phase, iron binding, inflammation",
    "Uroplakin 3B - mesothelial identity marker",
    "Proteoglycan 4 - serosal surface lubrication",
    "Mesothelin - therapeutic target (anti-MSLN antibodies)",
    "TFPI2 - tissue factor pathway regulation, invasion inhibition",
    "15-lipoxygenase - arachidonic acid metabolism, inflammation",
    "Hyaluronan synthase 1 - HA production, immune exclusion",
    "Complement factor B - innate immune activation",
    "Vitronectin - cell adhesion, integrin signaling",
    "Butyrylcholinesterase - detoxification",
    "Carboxypeptidase A4 - protein processing",
    "Synaptotagmin 4 - vesicle exocytosis, secretory function"
  ),
  stringsAsFactors = FALSE
)

# TUMOR_SPECIFIC (Cluster 19) — optimal markers
tumor_optimal <- data.frame(
  gene = c("MIA", "MPZ", "MAGEA4", "SFRP1", "DAPL1", "RPL35A",
           "SAA1", "ACTG2", "SNRPE", "TSTD1", "NDUFA4", "BEX2"),
  avg_log2FC = c(5.48, 5.13, 4.98, 4.95, 4.55, 4.17,
                 4.16, 3.22, 3.50, 3.30, 3.17, 3.12),
  pct1 = c(0.720, 0.476, 0.308, 0.727, 0.776, 1.000,
           0.308, 0.343, 0.797, 0.580, 0.874, 0.287),
  pct2 = c(0.031, 0.025, 0.022, 0.049, 0.078, 0.824,
           0.031, 0.048, 0.199, 0.128, 0.410, 0.062),
  specificity_ratio = c(23.2, 19.0, 14.0, 14.8, 9.9, 1.2,
                        9.9, 7.1, 4.0, 4.5, 2.1, 4.6),
  category = c("Invasion", "Neural_crest", "Cancer_testis_antigen", "WNT_antagonist", 
               "Apoptosis", "Ribosomal",
               "Acute_phase", "Cytoskeleton_EMT", "Splicing", 
               "Detoxification", "Mitochondria", "Neuronal"),
  biological_significance = c(
    "Melanoma inhibitory activity - invasion, metastasis promotion",
    "Myelin protein zero - neural crest/Schwann cell lineage",
    "MAGE-A4 - cancer-testis antigen, immunotherapy target (TCR-T)",
    "Secreted frizzled-related protein 1 - WNT pathway antagonist",
    "Death-associated protein like 1 - apoptosis regulation",
    "Ribosomal protein L35a - hyperactive translation",
    "Serum amyloid A1 - inflammation, metastasis niche",
    "Actin gamma 2 - smooth muscle actin, EMT marker",
    "snRNP E - spliceosome, RNA processing",
    "Thiosulfate:glutathione sulfurtransferase - redox metabolism",
    "NADH dehydrogenase - mitochondrial respiration",
    "Brain-expressed X-linked 2 - neuronal differentiation"
  ),
  stringsAsFactors = FALSE
)

write.csv(myofib_optimal, file.path(output_dir, "Table_optimal_markers_Myofibroblast.csv"), row.names = FALSE)
write.csv(meso_optimal, file.path(output_dir, "Table_optimal_markers_Mesothelial.csv"), row.names = FALSE)
write.csv(tumor_optimal, file.path(output_dir, "Table_optimal_markers_Tumor_specific.csv"), row.names = FALSE)

cat("\nOptimal markers defined:\n")
cat("  Myofibroblast:", nrow(myofib_optimal), "markers\n")
cat("  Mesothelial:", nrow(meso_optimal), "markers\n")
cat("  Tumor_specific:", nrow(tumor_optimal), "markers\n")

# DotPlot of optimal markers across all cell types
optimal_all <- c(myofib_optimal$gene, meso_optimal$gene, tumor_optimal$gene)
optimal_all <- optimal_all[optimal_all %in% rownames(seu)]

p_optimal_dot <- DotPlot(seu, features = optimal_all, group.by = "cell_type") +
  RotatedAxis() +
  theme_paper() +
  theme(axis.text.x = element_text(size = 4.5, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size = 5)) +
  scale_color_viridis_c() +
  ggtitle("Optimal Markers: Specificity Across All Cell Types")

ggsave(file.path(output_dir, "Fig6_DotPlot_optimal_markers.pdf"), p_optimal_dot,
       width = 320, height = 130, units = "mm")

# ----------------------------------------------------------
# PART 8: Gene signature scoring
# ----------------------------------------------------------
cat("\n━━━ PART 8: Gene signature scoring ━━━\n")

myofib_score_genes <- myofib_optimal$gene[myofib_optimal$gene %in% rownames(seu)]
meso_score_genes <- meso_optimal$gene[meso_optimal$gene %in% rownames(seu)]
tumor_score_genes <- tumor_optimal$gene[tumor_optimal$gene %in% rownames(seu)]

ecm_genes <- c("COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A3", "COL11A1",
               "FN1", "POSTN", "TNC", "SPARC", "THBS1", "THBS2",
               "MMP2", "MMP9", "MMP11", "MMP14", "LOX", "LOXL1", "LOXL2")
ecm_genes <- ecm_genes[ecm_genes %in% rownames(seu)]

texhaust <- c("PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "TOX2", 
              "CTLA4", "ENTPD1", "CD244", "CD160")
texhaust <- texhaust[texhaust %in% rownames(seu)]

immune_evasion <- c("CD274", "PDCD1LG2", "IDO1", "IDO2", "TGFB1", "TGFB2",
                    "IL10", "VEGFA", "HIF1A", "ARG1", "NOS2", "PTGS2")
immune_evasion <- immune_evasion[immune_evasion %in% rownames(seu)]

seu <- AddModuleScore(seu, features = list(myofib_score_genes), name = "Myofib_program", verbose = FALSE)
seu <- AddModuleScore(seu, features = list(meso_score_genes), name = "Meso_program", verbose = FALSE)
seu <- AddModuleScore(seu, features = list(tumor_score_genes), name = "TumorSpec_program", verbose = FALSE)
seu <- AddModuleScore(seu, features = list(ecm_genes), name = "ECM_score", verbose = FALSE)
seu <- AddModuleScore(seu, features = list(texhaust), name = "Texhaustion_score", verbose = FALSE)
seu <- AddModuleScore(seu, features = list(immune_evasion), name = "ImmuneEvasion_score", verbose = FALSE)

# UMAP visualization
p_scores <- FeaturePlot(seu, 
                        features = c("Myofib_program1", "Meso_program1", "TumorSpec_program1",
                                     "ECM_score1", "Texhaustion_score1", "ImmuneEvasion_score1"),
                        ncol = 3, pt.size = 0.01, raster = TRUE) &
  theme_paper() &
  scale_color_viridis_c()

ggsave(file.path(output_dir, "Fig7_ModuleScores_UMAP.pdf"), p_scores,
       width = 250, height = 170, units = "mm")

# Violin plots comparing groups
p_vln_myofib <- VlnPlot(seu, features = "Myofib_program1", 
                        group.by = "group", cols = group_colors, pt.size = 0) +
  theme_paper() + ggtitle("Myofibroblast Program") + NoLegend()

p_vln_meso <- VlnPlot(seu, features = "Meso_program1", 
                      group.by = "group", cols = group_colors, pt.size = 0) +
  theme_paper() + ggtitle("Mesothelial Program") + NoLegend()

p_vln_tumor <- VlnPlot(seu, features = "TumorSpec_program1", 
                       group.by = "group", cols = group_colors, pt.size = 0) +
  theme_paper() + ggtitle("Tumor_specific Program") + NoLegend()

ggsave(file.path(output_dir, "Fig8_FocusProgram_scores_by_group.pdf"),
       p_vln_myofib | p_vln_meso | p_vln_tumor,
       width = 220, height = 80, units = "mm")

# ----------------------------------------------------------
# PART 9: Sample-level metrics and correlations
# ----------------------------------------------------------
cat("\n━━━ PART 9: Sample-level analysis ━━━\n")
sample_metrics <- seu@meta.data %>%
  group_by(sample, group) %>%
  summarise(
    total_cells = n(),
    prop_Myofibroblast = sum(cell_type == "Myofibroblast") / n(),
    prop_Mesothelial = sum(cell_type == "Mesothelial") / n(),
    prop_Tumor_specific = sum(cell_type == "Tumor_specific") / n(),
    prop_CD8_T = sum(cell_type == "CD8_T_effector") / n(),
    prop_Treg = sum(cell_type == "Treg") / n(),
    prop_Macrophage = sum(cell_type == "Macrophage_SPP1") / n(),
    prop_Neutrophil = sum(cell_type == "Neutrophil") / n(),
    prop_CAF = sum(cell_type == "CAF") / n(),
    prop_DC = sum(cell_type == "DC_mature") / n(),
    mean_Myofib_program = mean(Myofib_program1),
    mean_Meso_program = mean(Meso_program1),
    mean_TumorSpec_program = mean(TumorSpec_program1),
    mean_ECM = mean(ECM_score1),
    mean_Texhaustion = mean(Texhaustion_score1),
    mean_ImmuneEvasion = mean(ImmuneEvasion_score1),
    CD8_Treg_ratio = sum(cell_type == "CD8_T_effector") / max(sum(cell_type == "Treg"), 1),
    .groups = "drop"
  )

write.csv(sample_metrics, file.path(output_dir, "Table_sample_metrics.csv"), row.names = FALSE)

# Focus cluster sample detail
focus_sample_detail <- seu@meta.data %>%
  filter(cell_type %in% c("Myofibroblast", "Mesothelial", "Tumor_specific")) %>%
  group_by(cell_type, sample, group) %>%
  summarise(n_cells = n(), .groups = "drop") %>%
  arrange(cell_type, desc(n_cells))

write.csv(focus_sample_detail, file.path(output_dir, "Table_focus_clusters_sample_detail.csv"), row.names = FALSE)
cat("\nFocus cluster sample distribution:\n")
print(as.data.frame(focus_sample_detail))

# Correlation heatmap
cor_vars <- c("prop_Myofibroblast", "prop_Mesothelial", "prop_Tumor_specific",
              "prop_CD8_T", "prop_Treg", "prop_Macrophage", "prop_Neutrophil",
              "prop_CAF", "mean_ECM", "mean_Texhaustion", "mean_ImmuneEvasion", 
              "CD8_Treg_ratio")

cor_mat <- cor(sample_metrics[, cor_vars], use = "complete.obs", method = "spearman")

pdf(file.path(output_dir, "Fig9_correlation_heatmap.pdf"), width = 8, height = 7)
col_fun <- colorRamp2(c(-1, 0, 1), c("#2166AC", "white", "#B2182B"))
ht_cor <- Heatmap(cor_mat,
                  name = "Spearman\nrho",
                  col = col_fun,
                  cluster_rows = TRUE,
                  cluster_columns = TRUE,
                  show_row_names = TRUE,
                  show_column_names = TRUE,
                  row_names_gp = gpar(fontsize = 7),
                  column_names_gp = gpar(fontsize = 7),
                  column_names_rot = 45,
                  cell_fun = function(j, i, x, y, width, height, fill) {
                    grid.text(sprintf("%.2f", cor_mat[i, j]), x, y, 
                              gp = gpar(fontsize = 5))
                  },
                  column_title = "Sample-level Correlations (Spearman)",
                  column_title_gp = gpar(fontsize = 9, fontface = "bold"))
draw(ht_cor)
dev.off()

# Key scatter plots
p_s1 <- ggplot(sample_metrics, aes(x = prop_Mesothelial, y = mean_Texhaustion, color = group)) +
  geom_point(size = 2.5) + scale_color_manual(values = group_colors) +
  theme_paper() + labs(x = "Mesothelial prop.", y = "T exhaustion score",
                       title = "Mesothelial vs T exhaustion")

p_s2 <- ggplot(sample_metrics, aes(x = prop_Myofibroblast, y = CD8_Treg_ratio, color = group)) +
  geom_point(size = 2.5) + scale_color_manual(values = group_colors) +
  theme_paper() + labs(x = "Myofibroblast prop.", y = "CD8/Treg ratio",
                       title = "Myofibroblast vs Immune activation")

p_s3 <- ggplot(sample_metrics, aes(x = prop_Mesothelial, y = prop_CD8_T, color = group)) +
  geom_point(size = 2.5) + scale_color_manual(values = group_colors) +
  theme_paper() + labs(x = "Mesothelial prop.", y = "CD8 T prop.",
                       title = "Mesothelial vs CD8 T")

p_s4 <- ggplot(sample_metrics, aes(x = prop_Myofibroblast, y = prop_Mesothelial, color = group)) +
  geom_point(size = 2.5) + scale_color_manual(values = group_colors) +
  theme_paper() + labs(x = "Myofibroblast prop.", y = "Mesothelial prop.",
                       title = "Myofibroblast vs Mesothelial (mutually exclusive?)")

ggsave(file.path(output_dir, "Fig10_scatter_correlations.pdf"),
       (p_s1 | p_s2) / (p_s3 | p_s4),
       width = 220, height = 180, units = "mm")

# ----------------------------------------------------------
# PART 10: Top DEG heatmap across cell types
# ----------------------------------------------------------
cat("\n━━━ PART 10: DEG heatmap ━━━\n")

top_deg_per_ct <- all_deg %>%
  filter(p_val_adj < 0.05 & abs(avg_log2FC) > 0.5) %>%
  group_by(cell_type) %>%
  slice_max(n = 10, order_by = abs(avg_log2FC)) %>%
  ungroup()

if (nrow(top_deg_per_ct) > 0) {
  deg_matrix <- top_deg_per_ct %>%
    dplyr::select(gene, cell_type, avg_log2FC) %>%
    distinct(gene, cell_type, .keep_all = TRUE) %>%
    pivot_wider(names_from = cell_type, values_from = avg_log2FC, values_fill = 0) %>%
    column_to_rownames("gene")
  
  if (nrow(deg_matrix) > 5 && ncol(deg_matrix) > 1) {
    pdf(file.path(output_dir, "Fig11_DEG_heatmap_all_celltypes.pdf"), width = 12, height = 14)
    
    deg_mat <- as.matrix(deg_matrix)
    deg_mat[deg_mat > 3] <- 3
    deg_mat[deg_mat < -3] <- -3
    
    ht_deg <- Heatmap(deg_mat,
                      name = "log2FC\n(High/Low)",
                      col = colorRamp2(c(-3, 0, 3), c("#2166AC", "white", "#B2182B")),
                      cluster_rows = TRUE,
                      cluster_columns = TRUE,
                      show_row_names = TRUE,
                      show_column_names = TRUE,
                      row_names_gp = gpar(fontsize = 5),
                      column_names_gp = gpar(fontsize = 7, fontface = "bold"),
                      column_names_rot = 45,
                      column_title = "Top DEGs (High_Risk vs Low_Risk) per Cell Type",
                      column_title_gp = gpar(fontsize = 10, fontface = "bold"))
    draw(ht_deg)
    dev.off()
  }
}

# ----------------------------------------------------------
# PART 11: CD8 T cell state analysis
# ----------------------------------------------------------
cat("\n━━━ PART 11: CD8 T cell states ━━━\n")

cd8_sub <- subset(seu, cell_type == "CD8_T_effector")
cat("CD8 T cells:", ncol(cd8_sub), "\n")

# Score CD8 T cells for functional states
cytotoxic_genes <- c("GZMB", "GZMA", "GZMK", "PRF1", "GNLY", "NKG7", "IFNG", "TNF", "FASLG")
cytotoxic_genes <- cytotoxic_genes[cytotoxic_genes %in% rownames(cd8_sub)]

exhaust_genes <- c("PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "TOX2", "CTLA4", "ENTPD1")
exhaust_genes <- exhaust_genes[exhaust_genes %in% rownames(cd8_sub)]

naive_genes <- c("TCF7", "LEF1", "CCR7", "SELL", "IL7R")
naive_genes <- naive_genes[naive_genes %in% rownames(cd8_sub)]

trm_genes <- c("ITGAE", "ZNF683", "CXCR6", "CD69", "ITGA1")
trm_genes <- trm_genes[trm_genes %in% rownames(cd8_sub)]

cd8_sub <- AddModuleScore(cd8_sub, features = list(cytotoxic_genes), name = "Cytotoxic", verbose = FALSE)
cd8_sub <- AddModuleScore(cd8_sub, features = list(exhaust_genes), name = "Exhausted", verbose = FALSE)
cd8_sub <- AddModuleScore(cd8_sub, features = list(naive_genes), name = "Naive_memory", verbose = FALSE)
if (length(trm_genes) >= 3) {
  cd8_sub <- AddModuleScore(cd8_sub, features = list(trm_genes), name = "TRM", verbose = FALSE)
}

# Violin plots comparing CD8 states between groups
p_cd8_cyto <- VlnPlot(cd8_sub, features = "Cytotoxic1", group.by = "group",
                      cols = group_colors, pt.size = 0) +
  theme_paper() + ggtitle("CD8 Cytotoxic Score") + NoLegend()

p_cd8_exh <- VlnPlot(cd8_sub, features = "Exhausted1", group.by = "group",
                     cols = group_colors, pt.size = 0) +
  theme_paper() + ggtitle("CD8 Exhaustion Score") + NoLegend()

p_cd8_naive <- VlnPlot(cd8_sub, features = "Naive_memory1", group.by = "group",
                       cols = group_colors, pt.size = 0) +
  theme_paper() + ggtitle("CD8 Naive/Memory Score") + NoLegend()

ggsave(file.path(output_dir, "Fig12_CD8_states_by_group.pdf"),
       p_cd8_cyto | p_cd8_exh | p_cd8_naive,
       width = 200, height = 80, units = "mm")

# Sample-level CD8 scores
cd8_sample_scores <- cd8_sub@meta.data %>%
  group_by(sample, group) %>%
  summarise(
    n_cd8 = n(),
    mean_cytotoxic = mean(Cytotoxic1),
    mean_exhausted = mean(Exhausted1),
    mean_naive = mean(Naive_memory1),
    .groups = "drop"
  )

cd8_merged <- left_join(sample_metrics, cd8_sample_scores, by = c("sample", "group"))
write.csv(cd8_merged, file.path(output_dir, "Table_CD8_states_with_focus_clusters.csv"), row.names = FALSE)

# Scatter: focus clusters vs CD8 states
p_meso_exh <- ggplot(cd8_merged, aes(x = prop_Mesothelial, y = mean_exhausted, color = group)) +
  geom_point(size = 2.5) + scale_color_manual(values = group_colors) +
  theme_paper() + labs(x = "Mesothelial proportion", y = "CD8 Exhaustion (mean)",
                       title = "Mesothelial → CD8 exhaustion?")

p_myofib_cyto <- ggplot(cd8_merged, aes(x = prop_Myofibroblast, y = mean_cytotoxic, color = group)) +
  geom_point(size = 2.5) + scale_color_manual(values = group_colors) +
  theme_paper() + labs(x = "Myofibroblast proportion", y = "CD8 Cytotoxic (mean)",
                       title = "Myofibroblast → CD8 cytotoxicity?")

ggsave(file.path(output_dir, "Fig13_focus_vs_CD8_states.pdf"),
       p_meso_exh | p_myofib_cyto, width = 220, height = 90, units = "mm")

rm(cd8_sub); gc(verbose = FALSE)

# ----------------------------------------------------------
# PART 12: Save final object
# ----------------------------------------------------------
cat("\n━━━ PART 12: Saving ━━━\n")

saveRDS(seu, file.path(output_dir, "seu_with_scores.rds"))

# ----------------------------------------------------------
# PART 13: Final summary
# ----------------------------------------------------------
cat("\n━━━ 04_Differential_Analysis COMPLETE ━━━\n")
cat("Output directory:", output_dir, "\n")

cat("\n╔══════════════════════════════════════════════════════════════════╗\n")
cat("║                    ANALYSIS SUMMARY                              ║\n")
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                  ║\n")
cat("║ DEG Analysis:                                                    ║\n")
cat(sprintf("║   • %d cell types analyzed                                       ║\n", length(deg_results)))
cat(sprintf("║   • %d total significant DEGs (|log2FC|>0.5, padj<0.05)         ║\n", 
            sum(all_deg$p_val_adj < 0.05 & abs(all_deg$avg_log2FC) > 0.5)))
cat("║                                                                  ║\n")
cat("║ Focus Clusters:                                                  ║\n")
cat("║   MYOFIBROBLAST (96.8% Low_Risk):                               ║\n")
cat("║     Top markers: POSTN, GPC6, KCND2, SGCD, PRKG1, WNT5A         ║\n")
cat("║     Programs: ECM remodeling, cGMP/NO signaling, Slit-Robo       ║\n")
cat("║     Hypothesis: Tumor-restraining stroma lost in High_Risk       ║\n")
cat("║                                                                  ║\n")
cat("║   MESOTHELIAL (99.7% High_Risk):                                 ║\n")
cat("║     Top markers: HP, UPK3B, PRG4, MSLN, TFPI2, HAS1             ║\n")
cat("║     Programs: Serosal identity, HA synthesis, complement         ║\n")
cat("║     Hypothesis: Serosal invasion marker, immune exclusion         ║\n")
cat("║     Therapeutic: Anti-mesothelin (amatuximab/anetumab)            ║\n")
cat("║                                                                  ║\n")
cat("║   TUMOR_SPECIFIC (95.1% High_Risk):                             ║\n")
cat("║     Top markers: MIA, MPZ, MAGEA4, SFRP1, DAPL1, RPL35A         ║\n")
cat("║     Programs: Invasion (MIA), CTA (MAGEA4), hyperactive          ║\n")
cat("║               translation, neural crest features                  ║\n")
cat("║     Therapeutic: MAGEA4-targeted TCR-T, anti-MIA                  ║\n")
cat("║                                                                  ║\n")
cat("╚══════════════════════════════════════════════════════════════════╝\n")

cat("\nKey output files:\n")
cat("  Tables:\n")
cat("    - Table_DEG_all_celltypes.csv\n")
cat("    - Table_DEG_summary.csv\n")
cat("    - Table_DEG_Myofibroblast_comparisons.csv\n")
cat("    - Table_DEG_Mesothelial_markers.csv\n")
cat("    - Table_DEG_Tumor_specific_markers.csv\n")
cat("    - Table_DEG_Myofibroblast_vs_allStromal.csv\n")
cat("    - Table_DEG_Mesothelial_vs_otherStromal.csv\n")
cat("    - Table_DEG_Tumor_specific_vs_allEpithelial.csv\n")
cat("    - Table_GO_*.csv (pathway enrichment)\n")
cat("    - Table_GSEA_Hallmark_all_celltypes.csv\n")
cat("    - Table_optimal_markers_*.csv\n")
cat("    - Table_sample_metrics.csv\n")
cat("    - Table_focus_clusters_sample_detail.csv\n")
cat("    - Table_CD8_states_with_focus_clusters.csv\n")
cat("  Figures:\n")
cat("    - Fig1_Volcano_major_celltypes.pdf\n")
cat("    - Fig2_Volcano_focus_clusters.pdf\n")
cat("    - Fig3A-C_GO_enrichment.pdf\n")
cat("    - Fig4_GSEA_Hallmark_heatmap.pdf\n")
cat("    - Fig5A-C_GO_signature.pdf\n")
cat("    - Fig6_DotPlot_optimal_markers.pdf\n")
cat("    - Fig7_ModuleScores_UMAP.pdf\n")
cat("    - Fig8_FocusProgram_scores_by_group.pdf\n")
cat("    - Fig9_correlation_heatmap.pdf\n")
cat("    - Fig10_scatter_correlations.pdf\n")
cat("    - Fig11_DEG_heatmap_all_celltypes.pdf\n")
cat("    - Fig12_CD8_states_by_group.pdf\n")
cat("    - Fig13_focus_vs_CD8_states.pdf\n")

cat("\nNext step: 05_CellCommunication.R\n")