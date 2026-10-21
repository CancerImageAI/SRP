# ============================================================
# 04a_CD8_Subtyping_HighRes.R
# High-resolution CD8 T cell sub-clustering
# ============================================================

options(Seurat.object.assay.version = "v3")
options(future.globals.maxSize = 8 * 1024^3)

library(Seurat)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(ggrepel)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(msigdbr)
library(patchwork)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/04_Differential/CD8_Subtype"
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

# ===========================================================
# PART 1: High-resolution CD8 sub-clustering
# ===========================================================
cat("━━━ PART 1: High-resolution CD8 T cell sub-clustering ━━━\n")

seu <- readRDS(file.path(input_dir, "seu_annotated.rds"))
cd8 <- subset(seu, cell_type == "CD8_T_effector")
rm(seu); gc(verbose = FALSE)

cat("CD8 T cells:", ncol(cd8), "\n")
cat("  High_Risk:", sum(cd8$group == "High_Risk"), "\n")
cat("  Low_Risk:", sum(cd8$group == "Low_Risk"), "\n")

# Re-process with higher resolution
cd8 <- NormalizeData(cd8, verbose = FALSE)
cd8 <- FindVariableFeatures(cd8, nfeatures = 2000, verbose = FALSE)
cd8 <- ScaleData(cd8, verbose = FALSE)
cd8 <- RunPCA(cd8, npcs = 30, verbose = FALSE)

# Use higher resolution for more clusters
cd8 <- RunUMAP(cd8, dims = 1:25, verbose = FALSE)
cd8 <- FindNeighbors(cd8, dims = 1:25, verbose = FALSE)
cd8 <- FindClusters(cd8, resolution = 0.8, verbose = FALSE)  # Higher resolution

cat("Sub-clusters found:", length(unique(cd8$seurat_clusters)), "\n")
cat("Cluster sizes:\n")
print(sort(table(cd8$seurat_clusters), decreasing = TRUE))

# ----------------------------------------------------------
# Find markers for each sub-cluster
# ----------------------------------------------------------
cat("\nFinding markers for each sub-cluster...\n")
cd8_markers <- FindAllMarkers(cd8, only.pos = TRUE, min.pct = 0.25, 
                              logfc.threshold = 0.5, verbose = FALSE)

top5_markers <- cd8_markers %>%
  group_by(cluster) %>%
  slice_max(n = 5, order_by = avg_log2FC)

write.csv(cd8_markers, file.path(output_dir, "Table_CD8_subcluster_allmarkers.csv"), row.names = FALSE)
write.csv(top5_markers, file.path(output_dir, "Table_CD8_subcluster_top5markers.csv"), row.names = FALSE)

cat("\nTop 3 markers per cluster:\n")
top3_print <- cd8_markers %>%
  group_by(cluster) %>%
  slice_max(n = 3, order_by = avg_log2FC) %>%
  dplyr::select(cluster, gene, avg_log2FC, pct.1, pct.2)
print(as.data.frame(top3_print), row.names = FALSE)

# ----------------------------------------------------------
# Annotate using comprehensive marker panel
# ----------------------------------------------------------
cat("\nAnnotating CD8 sub-clusters...\n")

# Define fine-grained marker signatures
subtype_signatures <- list(
  # Effector/Cytotoxic states
  CD8_Cytotoxic_GZMB = c("GZMB", "PRF1", "GNLY", "NKG7", "FGFBP2", "CX3CR1", "FCGR3A"),
  CD8_Cytotoxic_GZMK = c("GZMK", "GZMA", "EOMES", "CCL4", "CCL5", "CRTAM", "KLRK1"),
  
  # Exhaustion spectrum
  CD8_PreExhausted = c("PDCD1", "CXCL13", "TOX", "ENTPD1", "HAVCR2", "TIGIT"),
  CD8_TerminalExhausted = c("LAYN", "HAVCR2", "LAG3", "TOX2", "SNAP91", "ITGAE"),
  CD8_ProgenitorExhausted = c("TCF7", "PDCD1", "SLAMF6", "XCL1", "IL7R", "TOX"),
  
  # Memory states  
  CD8_CentralMemory = c("TCF7", "LEF1", "CCR7", "SELL", "IL7R", "CD27"),
  CD8_EffectorMemory = c("GZMK", "CCL5", "KLRG1", "GZMA", "EOMES"),
  CD8_TRM = c("ZNF683", "ITGAE", "CXCR6", "CD69", "ITGA1", "RUNX3"),
  
  # Functional states
  CD8_Proliferating = c("MKI67", "TOP2A", "STMN1", "HMGB2", "TYMS", "RRM2", "CDK1"),
  CD8_ISG = c("ISG15", "MX1", "IFIT1", "IFI6", "IFI44L", "RSAD2", "OAS1", "MX2"),
  CD8_MAIT = c("SLC4A10", "KLRB1", "ZBTB16", "NCR3", "RORC"),
  CD8_NKlike = c("KLRD1", "KLRC1", "KIR2DL4", "TYROBP", "NCAM1", "FCER1G"),
  
  # Stress/dysfunctional
  CD8_Stressed = c("HSPA1A", "HSPA1B", "DNAJB1", "HSP90AA1", "HSPB1"),
  CD8_Apoptotic = c("FAS", "BAX", "CASP3", "BCL2L11", "PMAIP1")
)

# Score each cell
for (sig_name in names(subtype_signatures)) {
  genes_use <- subtype_signatures[[sig_name]]
  genes_use <- genes_use[genes_use %in% rownames(cd8)]
  if (length(genes_use) >= 3) {
    cd8 <- AddModuleScore(cd8, features = list(genes_use), 
                          name = sig_name, verbose = FALSE)
  }
}

# Get all score columns
score_cols <- paste0(names(subtype_signatures), "1")
score_cols <- score_cols[score_cols %in% colnames(cd8@meta.data)]

# For each cluster, determine best annotation
cluster_scores <- cd8@meta.data %>%
  group_by(seurat_clusters) %>%
  summarise(across(all_of(score_cols), mean, .names = "{.col}"), .groups = "drop")

# Assign annotation based on highest mean score per cluster
score_matrix <- as.matrix(cluster_scores[, -1])
rownames(score_matrix) <- cluster_scores$seurat_clusters
colnames(score_matrix) <- gsub("1$", "", colnames(score_matrix))

# Get best annotation per cluster
best_annotation <- data.frame(
  cluster = rownames(score_matrix),
  annotation = colnames(score_matrix)[apply(score_matrix, 1, which.max)],
  max_score = apply(score_matrix, 1, max),
  stringsAsFactors = FALSE
)

cat("\nCluster annotations:\n")
print(best_annotation)

# Handle duplicates: if multiple clusters map to same type, add numbers
annotation_counts <- table(best_annotation$annotation)
for (ann in names(annotation_counts[annotation_counts > 1])) {
  idx <- which(best_annotation$annotation == ann)
  # Keep the one with highest score as-is, add _2, _3 to others
  score_order <- order(best_annotation$max_score[idx], decreasing = TRUE)
  for (i in 2:length(idx)) {
    best_annotation$annotation[idx[score_order[i]]] <- paste0(ann, "_", i)
  }
}

# Map to cells
cd8$cd8_subtype <- best_annotation$annotation[match(cd8$seurat_clusters, best_annotation$cluster)]

cat("\nFinal CD8 subtype distribution:\n")
print(sort(table(cd8$cd8_subtype), decreasing = TRUE))

cat("\nCD8 subtype by group:\n")
print(table(cd8$cd8_subtype, cd8$group))

# Also check: which clusters show the biggest group imbalance?
cluster_group_prop <- cd8@meta.data %>%
  group_by(seurat_clusters, cd8_subtype, group) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(seurat_clusters, cd8_subtype) %>%
  mutate(total = sum(n), prop = n / total) %>%
  ungroup() %>%
  filter(group == "High_Risk") %>%
  mutate(high_risk_pct = prop * 100) %>%
  arrange(desc(high_risk_pct))

cat("\nCluster group composition (% High_Risk):\n")
print(as.data.frame(cluster_group_prop[, c("seurat_clusters", "cd8_subtype", "total", "high_risk_pct")]))

# Save
saveRDS(cd8, file.path(output_dir, "cd8_highres_subtyped.rds"))

# ----------------------------------------------------------
# UMAP visualization
# ----------------------------------------------------------
p_umap_clusters <- DimPlot(cd8, group.by = "seurat_clusters", label = TRUE, 
                           label.size = 2.5, pt.size = 0.1) +
  theme_paper() + ggtitle("CD8 Sub-clusters") + NoLegend()

p_umap_subtype <- DimPlot(cd8, group.by = "cd8_subtype", label = TRUE, 
                          label.size = 2, repel = TRUE, pt.size = 0.1) +
  theme_paper() + ggtitle("CD8 Subtypes (annotated)") + NoLegend()

p_umap_group <- DimPlot(cd8, group.by = "group", pt.size = 0.1,
                        cols = group_colors) +
  theme_paper() + ggtitle("Risk Group")

ggsave(file.path(output_dir, "Fig1_CD8_UMAP_highres.pdf"),
       p_umap_clusters | p_umap_subtype | p_umap_group,
       width = 270, height = 90, units = "mm")

# Key markers feature plot
key_markers <- c("GZMB", "GZMK", "PRF1", "PDCD1", "TOX", "HAVCR2",
                 "TCF7", "CCR7", "ZNF683", "MKI67", "ISG15", "CXCL13")
key_markers <- key_markers[key_markers %in% rownames(cd8)]

p_feat <- FeaturePlot(cd8, features = key_markers, ncol = 4, 
                      pt.size = 0.01, raster = TRUE) &
  theme_paper() & scale_color_viridis_c()

ggsave(file.path(output_dir, "Fig2_CD8_key_markers_featureplot.pdf"), p_feat,
       width = 240, height = 180, units = "mm")

# DotPlot
dotplot_genes <- c("GZMB", "GZMK", "PRF1", "GNLY", "NKG7", "FGFBP2",
                   "PDCD1", "HAVCR2", "LAG3", "TOX", "CXCL13", "ENTPD1",
                   "TCF7", "LEF1", "CCR7", "SELL", "IL7R",
                   "ZNF683", "ITGAE", "CXCR6", "CD69",
                   "MKI67", "TOP2A", "STMN1",
                   "ISG15", "MX1", "IFIT1",
                   "SLC4A10", "KLRB1",
                   "KLRD1", "TYROBP")
dotplot_genes <- dotplot_genes[dotplot_genes %in% rownames(cd8)]

p_dot <- DotPlot(cd8, features = dotplot_genes, group.by = "cd8_subtype") +
  RotatedAxis() +
  theme_paper() +
  theme(axis.text.x = element_text(size = 5, angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(size = 5)) +
  scale_color_viridis_c() +
  ggtitle("CD8 Subtype Markers")

ggsave(file.path(output_dir, "Fig3_CD8_subtype_dotplot.pdf"), p_dot,
       width = 250, height = 100, units = "mm")

# ===========================================================
# PART 2: Proportion comparison (FIXED)
# ===========================================================
cat("\n━━━ PART 2: Proportion comparison ━━━\n")

# Calculate proportion per sample
cd8_prop <- cd8@meta.data %>%
  group_by(sample, group, cd8_subtype) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(sample, group) %>%
  mutate(total = sum(n), proportion = n / total * 100) %>%
  ungroup()

cd8_prop$group <- factor(cd8_prop$group, levels = c("Low_Risk", "High_Risk"))

# ─── CRITICAL FIX: Fill missing combinations with 0 ───
sample_group <- cd8@meta.data %>%
  dplyr::select(sample, group) %>%
  distinct()

all_combinations <- expand.grid(
  sample = unique(cd8_prop$sample),
  cd8_subtype = unique(cd8_prop$cd8_subtype),
  stringsAsFactors = FALSE
)
all_combinations <- left_join(all_combinations, sample_group, by = "sample")

cd8_prop_complete <- left_join(
  all_combinations,
  cd8_prop %>% dplyr::select(sample, cd8_subtype, proportion),
  by = c("sample", "cd8_subtype")
) %>%
  mutate(proportion = replace_na(proportion, 0))

cd8_prop_complete$group <- factor(cd8_prop_complete$group, levels = c("Low_Risk", "High_Risk"))

# ─── Filter: only test subtypes with >=2 non-zero samples in EACH group ───
testable_subtypes <- cd8_prop_complete %>%
  filter(proportion > 0) %>%
  group_by(cd8_subtype, group) %>%
  summarise(n_samples = n(), .groups = "drop") %>%
  pivot_wider(names_from = group, values_from = n_samples, values_fill = 0) %>%
  filter(Low_Risk >= 2 & High_Risk >= 2) %>%
  pull(cd8_subtype)

cat("Total subtypes:", length(unique(cd8_prop_complete$cd8_subtype)), "\n")
cat("Testable subtypes (>=2 samples/group with >0 cells):", length(testable_subtypes), "\n")

# Report group-restricted subtypes (biologically interesting!)
all_subtypes <- unique(cd8_prop_complete$cd8_subtype)
nontestable <- setdiff(all_subtypes, testable_subtypes)
if (length(nontestable) > 0) {
  cat("\nGroup-restricted subtypes (cannot test, but biologically interesting):\n")
  for (nt in nontestable) {
    nt_data <- cd8_prop_complete %>% filter(cd8_subtype == nt & proportion > 0)
    groups_present <- paste(unique(nt_data$group), collapse = " + ")
    cat(sprintf("  %s → present only in: %s (%d samples)\n", nt, groups_present, nrow(nt_data)))
  }
}

# ─── Run Wilcoxon only on testable subtypes ───
if (length(testable_subtypes) > 0) {
  stat_results <- cd8_prop_complete %>%
    filter(cd8_subtype %in% testable_subtypes) %>%
    group_by(cd8_subtype) %>%
    wilcox_test(proportion ~ group) %>%
    adjust_pvalue(method = "BH") %>%
    add_significance("p")
  
  # FIX: remove list columns before writing CSV
  stat_df <- stat_results %>%
    as.data.frame() %>%
    dplyr::select(any_of(c("cd8_subtype", ".y.", "group1", "group2",
                           "n1", "n2", "statistic", "p", "p.adj", "p.signif")))
  
  cat("\nWilcoxon test results:\n")
  print(stat_df)
  
  write.csv(stat_df, file.path(output_dir, "Table_CD8_subtype_Wilcoxon_highres.csv"), row.names = FALSE)
  
  sig_subtypes <- stat_df %>% filter(p < 0.05) %>% pull(cd8_subtype)
  trending_subtypes <- stat_df %>% filter(p >= 0.05 & p < 0.1) %>% pull(cd8_subtype)
  
  cat("\n*** Significant (p<0.05):", ifelse(length(sig_subtypes) > 0, paste(sig_subtypes, collapse=", "), "NONE"), "\n")
  cat("*** Trending (p<0.1):", ifelse(length(trending_subtypes) > 0, paste(trending_subtypes, collapse=", "), "NONE"), "\n")
} else {
  cat("\nNo testable subtypes! All are group-restricted.\n")
  stat_df <- data.frame()
  sig_subtypes <- character(0)
  trending_subtypes <- character(0)
}

# ===========================================================
# PART 3: DEG + Volcano for each CD8 subtype (High vs Low Risk)
# ===========================================================
cat("\n━━━ PART 3: DEG analysis per CD8 subtype ━━━\n")

MAX_CELLS_PER_GROUP <- 5000

# Volcano function (with integer fix)
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
  max_y <- quantile(deg_df$neg_log10_padj[is.finite(deg_df$neg_log10_padj)], 0.99, na.rm = TRUE)
  deg_df$neg_log10_padj[deg_df$neg_log10_padj > max_y * 1.2] <- max_y * 1.2
  
  # Select top genes to label (FIX: use ceiling/floor for integer)
  n_top_up <- as.integer(ceiling(top_n / 2))
  n_top_down <- as.integer(floor(top_n / 2))
  
  top_up <- deg_df %>% filter(color == "Up in High_Risk") %>%
    slice_max(n = n_top_up, order_by = avg_log2FC) %>% pull(gene)
  top_down <- deg_df %>% filter(color == "Up in Low_Risk") %>%
    slice_min(n = n_top_down, order_by = avg_log2FC) %>% pull(gene)
  top_genes <- c(top_up, top_down)
  
  deg_df$label <- ifelse(deg_df$gene %in% top_genes, deg_df$gene, "")
  
  n_up <- sum(deg_df$color == "Up in High_Risk")
  n_down <- sum(deg_df$color == "Up in Low_Risk")
  
  ggplot(deg_df, aes(x = avg_log2FC, y = neg_log10_padj, color = color)) +
    geom_point(size = 0.4, alpha = 0.5) +
    geom_text_repel(aes(label = label), size = 2, max.overlaps = 25,
                    segment.size = 0.2, segment.alpha = 0.5) +
    scale_color_manual(values = c("Up in High_Risk" = "#D32F2F", 
                                  "Up in Low_Risk" = "#1976D2", 
                                  "NS" = "grey75"), name = "") +
    geom_vline(xintercept = c(-fc_cutoff, fc_cutoff), 
               linetype = "dashed", linewidth = 0.3, color = "grey40") +
    geom_hline(yintercept = -log10(padj_cutoff), 
               linetype = "dashed", linewidth = 0.3, color = "grey40") +
    theme_paper(base_size = 7) +
    labs(x = "log2FC (High_Risk vs Low_Risk)", 
         y = "-log10(adjusted P-value)",
         title = gsub("_", " ", title),
         subtitle = sprintf("Up: %d | Down: %d", n_up, n_down)) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 6),
          plot.subtitle = element_text(size = 6, hjust = 0.5))
}

# Determine which subtypes to analyze
# Use ALL subtypes with >= 30 cells per group (regardless of proportion significance)
cells_per_group <- cd8@meta.data %>%
  group_by(cd8_subtype, group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = group, values_from = n, values_fill = 0) %>%
  mutate(min_n = pmin(High_Risk, Low_Risk))

deg_subtypes <- cells_per_group %>% filter(min_n >= 30) %>% pull(cd8_subtype)
cat("Subtypes with >=30 cells/group for DEG:", paste(deg_subtypes, collapse = ", "), "\n")

# Run DEG for each subtype
deg_results <- list()

for (st in deg_subtypes) {
  cat(sprintf("\n  DEG: %s\n", st))
  
  sub <- subset(cd8, cd8_subtype == st)
  n_high <- sum(sub$group == "High_Risk")
  n_low <- sum(sub$group == "Low_Risk")
  cat(sprintf("    Cells: High=%d, Low=%d\n", n_high, n_low))
  
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
    cat(sprintf("    FAILED: %s\n", e$message))
    return(NULL)
  })
  
  if (!is.null(deg) && nrow(deg) > 0) {
    deg$gene <- rownames(deg)
    deg$cd8_subtype <- st
    n_up <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC > 0.5)
    n_down <- sum(deg$p_val_adj < 0.05 & deg$avg_log2FC < -0.5)
    cat(sprintf("    Results: UP=%d, DOWN=%d, Total=%d\n", n_up, n_down, nrow(deg)))
    
    deg_results[[st]] <- deg
    
    # Save DEG table
    st_clean <- gsub("[/ ]", "_", st)
    write.csv(deg, file.path(output_dir, sprintf("Table_DEG_%s.csv", st_clean)), row.names = FALSE)
    
    # Volcano plot (only if has significant DEGs)
    if (n_up + n_down > 0) {
      p_vol <- plot_volcano(deg, st, top_n = 20)
      ggsave(file.path(output_dir, sprintf("Volcano_%s.pdf", st_clean)), p_vol,
             width = 100, height = 100, units = "mm")
    }
  }
  
  rm(sub); gc(verbose = FALSE)
}

# Save all DEG results
saveRDS(deg_results, file.path(output_dir, "deg_results_cd8_subtypes.rds"))

# DEG summary
deg_summary <- data.frame(
  subtype = names(deg_results),
  n_up = sapply(deg_results, function(x) sum(x$p_val_adj < 0.05 & x$avg_log2FC > 0.5)),
  n_down = sapply(deg_results, function(x) sum(x$p_val_adj < 0.05 & x$avg_log2FC < -0.5)),
  stringsAsFactors = FALSE
) %>%
  mutate(total_DEG = n_up + n_down) %>%
  arrange(desc(total_DEG))

cat("\nDEG summary per CD8 subtype:\n")
print(deg_summary)
write.csv(deg_summary, file.path(output_dir, "Table_DEG_summary_CD8_subtypes.csv"), row.names = FALSE)

# ===========================================================
# PART 4: GO enrichment for each CD8 subtype
# ===========================================================
cat("\n━━━ PART 4: GO enrichment ━━━\n")

run_GO <- function(genes, title = "GO") {
  if (length(genes) < 5) {
    cat(sprintf("    Too few genes (%d) for GO: %s\n", length(genes), title))
    return(NULL)
  }
  
  gene_ids <- tryCatch(
    bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db),
    error = function(e) data.frame(SYMBOL = character(), ENTREZID = character())
  )
  
  if (nrow(gene_ids) < 5) {
    cat(sprintf("    Too few mapped genes (%d): %s\n", nrow(gene_ids), title))
    return(NULL)
  }
  
  ego <- enrichGO(gene = gene_ids$ENTREZID,
                  OrgDb = org.Hs.eg.db,
                  ont = "BP",
                  pAdjustMethod = "BH",
                  pvalueCutoff = 0.05,
                  qvalueCutoff = 0.2,
                  readable = TRUE)
  return(ego)
}

for (st in names(deg_results)) {
  cat(sprintf("\n  GO: %s\n", st))
  deg <- deg_results[[st]]
  st_clean <- gsub("[/ ]", "_", st)
  
  # UP in High_Risk
  up_genes <- deg %>% filter(p_val_adj < 0.05 & avg_log2FC > 0.5) %>% pull(gene)
  cat(sprintf("    Up genes: %d\n", length(up_genes)))
  
  go_up <- run_GO(up_genes, paste0(st, "_UP"))
  if (!is.null(go_up) && nrow(as.data.frame(go_up)) > 0) {
    write.csv(as.data.frame(go_up),
              file.path(output_dir, sprintf("Table_GO_UP_%s.csv", st_clean)), row.names = FALSE)
    
    p_go <- dotplot(go_up, showCategory = 15, font.size = 7) +
      theme_paper(base_size = 7) +
      ggtitle(sprintf("%s: GO BP (Up in High_Risk)", gsub("CD8_", "", st))) +
      theme(axis.text.y = element_text(size = 5))
    
    ggsave(file.path(output_dir, sprintf("GO_UP_dotplot_%s.pdf", st_clean)), p_go,
           width = 140, height = 120, units = "mm")
  }
  
  # DOWN in High_Risk (Up in Low_Risk)
  down_genes <- deg %>% filter(p_val_adj < 0.05 & avg_log2FC < -0.5) %>% pull(gene)
  cat(sprintf("    Down genes: %d\n", length(down_genes)))
  
  go_down <- run_GO(down_genes, paste0(st, "_DOWN"))
  if (!is.null(go_down) && nrow(as.data.frame(go_down)) > 0) {
    write.csv(as.data.frame(go_down),
              file.path(output_dir, sprintf("Table_GO_DOWN_%s.csv", st_clean)), row.names = FALSE)
    
    p_go_d <- dotplot(go_down, showCategory = 15, font.size = 7) +
      theme_paper(base_size = 7) +
      ggtitle(sprintf("%s: GO BP (Down in High_Risk)", gsub("CD8_", "", st))) +
      theme(axis.text.y = element_text(size = 5))
    
    ggsave(file.path(output_dir, sprintf("GO_DOWN_dotplot_%s.pdf", st_clean)), p_go_d,
           width = 140, height = 120, units = "mm")
  }
}

# ===========================================================
# PART 5: GSEA (Hallmark) for each CD8 subtype
# ===========================================================
cat("\n━━━ PART 5: GSEA Hallmark ━━━\n")

run_GSEA <- function(deg_df, title = "GSEA") {
  deg_ranked <- deg_df %>%
    filter(!is.na(avg_log2FC) & !is.na(p_val) & is.finite(avg_log2FC)) %>%
    mutate(rank_metric = avg_log2FC * (-log10(p_val + 1e-300))) %>%
    arrange(desc(rank_metric))
  
  gene_list <- deg_ranked$rank_metric
  names(gene_list) <- deg_ranked$gene
  gene_list <- gene_list[!duplicated(names(gene_list))]
  gene_list <- sort(gene_list, decreasing = TRUE)
  
  if (length(gene_list) < 100) {
    cat(sprintf("    Too few ranked genes (%d): %s\n", length(gene_list), title))
    return(NULL)
  }
  
  hallmark <- msigdbr(species = "Homo sapiens", category = "H") %>%
    dplyr::select(gs_name, gene_symbol)
  
  gsea_res <- tryCatch(
    GSEA(gene_list, TERM2GENE = hallmark, pvalueCutoff = 0.25,
         verbose = FALSE, eps = 1e-100, nPermSimple = 10000),
    error = function(e) {
      cat(sprintf("    GSEA failed: %s\n", e$message))
      return(NULL)
    }
  )
  return(gsea_res)
}

gsea_all_results <- list()

for (st in names(deg_results)) {
  cat(sprintf("\n  GSEA: %s\n", st))
  deg <- deg_results[[st]]
  st_clean <- gsub("[/ ]", "_", st)
  
  gsea_res <- run_GSEA(deg, title = st)
  
  if (!is.null(gsea_res) && nrow(as.data.frame(gsea_res)) > 0) {
    gsea_df <- as.data.frame(gsea_res)
    gsea_df$cd8_subtype <- st
    gsea_all_results[[st]] <- gsea_df
    
    write.csv(gsea_df, file.path(output_dir, sprintf("Table_GSEA_%s.csv", st_clean)), row.names = FALSE)
    
    # NES barplot
    gsea_plot_df <- gsea_df %>%
      arrange(NES) %>%
      mutate(
        pathway = gsub("HALLMARK_", "", ID),
        pathway = factor(pathway, levels = pathway),
        color = ifelse(NES > 0, "Activated in High_Risk", "Activated in Low_Risk"),
        sig = ifelse(p.adjust < 0.05, "***", ifelse(p.adjust < 0.1, "*", ""))
      )
    
    p_nes <- ggplot(gsea_plot_df, aes(x = NES, y = pathway, fill = color)) +
      geom_col(width = 0.7) +
      geom_vline(xintercept = 0, linewidth = 0.3) +
      geom_text(aes(label = sig, x = NES + sign(NES) * 0.1), size = 2.5, hjust = 0.5) +
      scale_fill_manual(values = c("Activated in High_Risk" = "#D32F2F",
                                   "Activated in Low_Risk" = "#1976D2")) +
      theme_paper(base_size = 7) +
      labs(x = "Normalized Enrichment Score (NES)", y = "", fill = "",
           title = sprintf("%s: GSEA Hallmark", gsub("CD8_", "", st))) +
      theme(axis.text.y = element_text(size = 5),
            legend.position = "bottom",
            legend.text = element_text(size = 6))
    
    plot_height <- max(60, nrow(gsea_plot_df) * 5 + 30)
    ggsave(file.path(output_dir, sprintf("GSEA_NES_barplot_%s.pdf", st_clean)), p_nes,
           width = 140, height = plot_height, units = "mm")
    
    # Running score plots for top significant pathways
    sig_pw <- gsea_df %>% filter(p.adjust < 0.05) %>% arrange(p.adjust) %>% head(4) %>% pull(ID)
    if (length(sig_pw) > 0) {
      for (pw in sig_pw) {
        pw_clean <- gsub("HALLMARK_", "", pw)
        p_run <- tryCatch({
          gseaplot2(gsea_res, geneSetID = pw,
                    title = sprintf("%s: %s", gsub("CD8_", "", st), pw_clean),
                    pvalue_table = TRUE, base_size = 7)
        }, error = function(e) NULL)
        
        if (!is.null(p_run)) {
          ggsave(file.path(output_dir, sprintf("GSEA_running_%s_%s.pdf", st_clean, pw_clean)),
                 p_run, width = 120, height = 100, units = "mm")
        }
      }
    }
    
    cat(sprintf("    Significant pathways (padj<0.05): %d\n", sum(gsea_df$p.adjust < 0.05)))
  }
}

# Combined GSEA NES heatmap across CD8 subtypes
if (length(gsea_all_results) > 1) {
  all_gsea_df <- bind_rows(gsea_all_results)
  write.csv(all_gsea_df, file.path(output_dir, "Table_GSEA_all_CD8_subtypes.csv"), row.names = FALSE)
  
  nes_matrix <- all_gsea_df %>%
    dplyr::select(ID, NES, cd8_subtype) %>%
    pivot_wider(names_from = cd8_subtype, values_from = NES, values_fill = 0) %>%
    column_to_rownames("ID")
  
  # Filter to pathways significant in at least one subtype
  sig_pws <- all_gsea_df %>% filter(p.adjust < 0.05) %>% pull(ID) %>% unique()
  
  if (length(sig_pws) > 1) {
    library(ComplexHeatmap)
    library(circlize)
    
    nes_filt <- as.matrix(nes_matrix[rownames(nes_matrix) %in% sig_pws, , drop = FALSE])
    rownames(nes_filt) <- gsub("HALLMARK_", "", rownames(nes_filt))
    colnames(nes_filt) <- gsub("CD8_", "", colnames(nes_filt))
    
    if (nrow(nes_filt) > 1 && ncol(nes_filt) > 1) {
      pdf(file.path(output_dir, "Fig6_GSEA_heatmap_CD8_subtypes.pdf"), width = 10, height = 8)
      ht <- Heatmap(nes_filt,
                    name = "NES",
                    col = colorRamp2(c(-3, 0, 3), c("#2166AC", "white", "#B2182B")),
                    cluster_rows = TRUE,
                    cluster_columns = TRUE,
                    show_row_names = TRUE,
                    show_column_names = TRUE,
                    row_names_gp = gpar(fontsize = 6),
                    column_names_gp = gpar(fontsize = 7, fontface = "bold"),
                    column_names_rot = 45,
                    column_title = "GSEA Hallmark NES: High_Risk vs Low_Risk (CD8 Subtypes)",
                    column_title_gp = gpar(fontsize = 9, fontface = "bold"),
                    heatmap_legend_param = list(
                      title_gp = gpar(fontsize = 7),
                      labels_gp = gpar(fontsize = 6)
                    ))
      draw(ht)
      dev.off()
    }
  }
}

# ===========================================================
# PART 6: Focus clusters (Myofibroblast, Mesothelial, Tumor_specific)
# ===========================================================
cat("\n━━━ PART 6: Focus clusters (Myofibroblast, Mesothelial, Tumor_specific) ━━━\n")

# Reload full Seurat object
seu <- readRDS(file.path(input_dir, "seu_annotated.rds"))

# --- Myofibroblast: High vs Low Risk ---
cat("  Myofibroblast DEG (High vs Low)...\n")
myofib_sub <- subset(seu, cell_type == "Myofibroblast")
n_h_myo <- sum(myofib_sub$group == "High_Risk")
n_l_myo <- sum(myofib_sub$group == "Low_Risk")
cat(sprintf("    Cells: High=%d, Low=%d\n", n_h_myo, n_l_myo))

if (n_h_myo >= 30 & n_l_myo >= 30) {
  Idents(myofib_sub) <- "group"
  deg_myofib <- FindMarkers(myofib_sub, ident.1 = "High_Risk", ident.2 = "Low_Risk",
                            min.pct = 0.1, logfc.threshold = 0.1, verbose = FALSE)
  deg_myofib$gene <- rownames(deg_myofib)
  deg_myofib$cell_type <- "Myofibroblast"
  
  n_up <- sum(deg_myofib$p_val_adj < 0.05 & deg_myofib$avg_log2FC > 0.5)
  n_down <- sum(deg_myofib$p_val_adj < 0.05 & deg_myofib$avg_log2FC < -0.5)
  cat(sprintf("    DEG: UP=%d, DOWN=%d\n", n_up, n_down))
  
  write.csv(deg_myofib, file.path(output_dir, "Table_DEG_Myofibroblast_HighvsLow.csv"), row.names = FALSE)
  
  p_vol_myo <- plot_volcano(deg_myofib, "Myofibroblast (High vs Low Risk)", top_n = 20)
  ggsave(file.path(output_dir, "Volcano_Myofibroblast_HighvsLow.pdf"), p_vol_myo,
         width = 100, height = 100, units = "mm")
  
  # GO UP
  up_myo <- deg_myofib %>% filter(p_val_adj < 0.05 & avg_log2FC > 0.5) %>% pull(gene)
  go_myo_up <- run_GO(up_myo, "Myofibroblast_UP")
  if (!is.null(go_myo_up) && nrow(as.data.frame(go_myo_up)) > 0) {
    write.csv(as.data.frame(go_myo_up), file.path(output_dir, "Table_GO_UP_Myofibroblast.csv"), row.names = FALSE)
    p <- dotplot(go_myo_up, showCategory = 15, font.size = 7) + theme_paper() +
      ggtitle("Myofibroblast: GO BP (Up in High_Risk)") +
      theme(axis.text.y = element_text(size = 5))
    ggsave(file.path(output_dir, "GO_UP_dotplot_Myofibroblast.pdf"), p, width = 140, height = 120, units = "mm")
  }
  
  # GO DOWN
  down_myo <- deg_myofib %>% filter(p_val_adj < 0.05 & avg_log2FC < -0.5) %>% pull(gene)
  go_myo_down <- run_GO(down_myo, "Myofibroblast_DOWN")
  if (!is.null(go_myo_down) && nrow(as.data.frame(go_myo_down)) > 0) {
    write.csv(as.data.frame(go_myo_down), file.path(output_dir, "Table_GO_DOWN_Myofibroblast.csv"), row.names = FALSE)
    p <- dotplot(go_myo_down, showCategory = 15, font.size = 7) + theme_paper() +
      ggtitle("Myofibroblast: GO BP (Down in High_Risk)") +
      theme(axis.text.y = element_text(size = 5))
    ggsave(file.path(output_dir, "GO_DOWN_dotplot_Myofibroblast.pdf"), p, width = 140, height = 120, units = "mm")
  }
  
  # GSEA Myofibroblast
  gsea_myo <- run_GSEA(deg_myofib, "Myofibroblast")
  if (!is.null(gsea_myo) && nrow(as.data.frame(gsea_myo)) > 0) {
    gsea_myo_df <- as.data.frame(gsea_myo)
    write.csv(gsea_myo_df, file.path(output_dir, "Table_GSEA_Myofibroblast.csv"), row.names = FALSE)
    
    gsea_plot_df <- gsea_myo_df %>%
      arrange(NES) %>%
      mutate(pathway = gsub("HALLMARK_", "", ID),
             pathway = factor(pathway, levels = pathway),
             color = ifelse(NES > 0, "Activated in High_Risk", "Activated in Low_Risk"))
    
    p_nes_myo <- ggplot(gsea_plot_df, aes(x = NES, y = pathway, fill = color)) +
      geom_col(width = 0.7) + geom_vline(xintercept = 0, linewidth = 0.3) +
      scale_fill_manual(values = c("Activated in High_Risk" = "#D32F2F",
                                   "Activated in Low_Risk" = "#1976D2")) +
      theme_paper(base_size = 7) +
      labs(x = "NES", y = "", fill = "", title = "Myofibroblast: GSEA Hallmark") +
      theme(axis.text.y = element_text(size = 5), legend.position = "bottom")
    
    ggsave(file.path(output_dir, "GSEA_NES_barplot_Myofibroblast.pdf"), p_nes_myo,
           width = 140, height = max(60, nrow(gsea_plot_df) * 5 + 30), units = "mm")
    
    # Running score top pathways
    sig_pw <- gsea_myo_df %>% filter(p.adjust < 0.05) %>% arrange(p.adjust) %>% head(4) %>% pull(ID)
    for (pw in sig_pw) {
      pw_clean <- gsub("HALLMARK_", "", pw)
      p_run <- tryCatch(
        gseaplot2(gsea_myo, geneSetID = pw,
                  title = sprintf("Myofibroblast: %s", pw_clean),
                  pvalue_table = TRUE, base_size = 7),
        error = function(e) NULL)
      if (!is.null(p_run)) {
        ggsave(file.path(output_dir, sprintf("GSEA_running_Myofibroblast_%s.pdf", pw_clean)),
               p_run, width = 120, height = 100, units = "mm")
      }
    }
  }
} else {
  cat("    SKIPPED: insufficient cells (High=", n_h_myo, ", Low=", n_l_myo, ")\n")
}
rm(myofib_sub); gc(verbose = FALSE)

# --- Mesothelial: compare vs Endothelial + Pericyte (can't do High vs Low) ---
cat("\n  Mesothelial markers (vs other stromal)...\n")
meso_compare <- subset(seu, cell_type %in% c("Mesothelial", "Endothelial", "Pericyte_SMC"))
Idents(meso_compare) <- "cell_type"

deg_meso <- FindMarkers(meso_compare, ident.1 = "Mesothelial", ident.2 = NULL,
                        min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_meso$gene <- rownames(deg_meso)
deg_meso$cell_type <- "Mesothelial"

n_up_m <- sum(deg_meso$p_val_adj < 0.05 & deg_meso$avg_log2FC > 0.5)
n_down_m <- sum(deg_meso$p_val_adj < 0.05 & deg_meso$avg_log2FC < -0.5)
cat(sprintf("    Markers: UP=%d, DOWN=%d\n", n_up_m, n_down_m))

write.csv(deg_meso, file.path(output_dir, "Table_DEG_Mesothelial_vs_otherStromal.csv"), row.names = FALSE)

p_vol_meso <- plot_volcano(deg_meso, "Mesothelial vs Endothelial + Pericyte", top_n = 20)
ggsave(file.path(output_dir, "Volcano_Mesothelial_vs_otherStromal.pdf"), p_vol_meso,
       width = 100, height = 100, units = "mm")

# GO for Mesothelial markers
meso_markers <- deg_meso %>% filter(p_val_adj < 0.05 & avg_log2FC > 1) %>% pull(gene)
cat(sprintf("    Mesothelial-specific markers (log2FC>1): %d\n", length(meso_markers)))

go_meso <- run_GO(meso_markers, "Mesothelial_markers")
if (!is.null(go_meso) && nrow(as.data.frame(go_meso)) > 0) {
  write.csv(as.data.frame(go_meso), file.path(output_dir, "Table_GO_Mesothelial_markers.csv"), row.names = FALSE)
  p <- dotplot(go_meso, showCategory = 20, font.size = 7) + theme_paper() +
    ggtitle("Mesothelial Markers: GO BP") + theme(axis.text.y = element_text(size = 5))
  ggsave(file.path(output_dir, "GO_dotplot_Mesothelial_markers.pdf"), p,
         width = 150, height = 140, units = "mm")
}

# GSEA for Mesothelial
gsea_meso <- run_GSEA(deg_meso, "Mesothelial")
if (!is.null(gsea_meso) && nrow(as.data.frame(gsea_meso)) > 0) {
  gsea_meso_df <- as.data.frame(gsea_meso)
  write.csv(gsea_meso_df, file.path(output_dir, "Table_GSEA_Mesothelial.csv"), row.names = FALSE)
  
  gsea_plot_df <- gsea_meso_df %>%
    arrange(NES) %>%
    mutate(pathway = gsub("HALLMARK_", "", ID),
           pathway = factor(pathway, levels = pathway),
           color = ifelse(NES > 0, "Enriched in Mesothelial", "Depleted"))
  
  p_nes <- ggplot(gsea_plot_df, aes(x = NES, y = pathway, fill = color)) +
    geom_col(width = 0.7) + geom_vline(xintercept = 0, linewidth = 0.3) +
    scale_fill_manual(values = c("Enriched in Mesothelial" = "#D32F2F",
                                 "Depleted" = "#1976D2")) +
    theme_paper(base_size = 7) +
    labs(x = "NES", y = "", fill = "", title = "Mesothelial: GSEA Hallmark") +
    theme(axis.text.y = element_text(size = 5), legend.position = "bottom")
  
  ggsave(file.path(output_dir, "GSEA_NES_barplot_Mesothelial.pdf"), p_nes,
         width = 140, height = max(60, nrow(gsea_plot_df) * 5 + 30), units = "mm")
}
rm(meso_compare); gc(verbose = FALSE)

# --- Tumor_specific: compare vs other epithelial ---
cat("\n  Tumor_specific markers (vs other epithelial)...\n")
tumor_compare <- subset(seu, cell_type %in% c("Tumor_specific", "Epithelial_basal",
                                              "Epithelial_squamous", "Epithelial_differentiated"))
Idents(tumor_compare) <- "cell_type"

deg_tumor <- FindMarkers(tumor_compare, ident.1 = "Tumor_specific", ident.2 = NULL,
                         min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_tumor$gene <- rownames(deg_tumor)
deg_tumor$cell_type <- "Tumor_specific"

n_up_t <- sum(deg_tumor$p_val_adj < 0.05 & deg_tumor$avg_log2FC > 0.5)
n_down_t <- sum(deg_tumor$p_val_adj < 0.05 & deg_tumor$avg_log2FC < -0.5)
cat(sprintf("    Markers: UP=%d, DOWN=%d\n", n_up_t, n_down_t))

write.csv(deg_tumor, file.path(output_dir, "Table_DEG_Tumor_specific_vs_otherEpithelial.csv"), row.names = FALSE)

p_vol_tumor <- plot_volcano(deg_tumor, "Tumor_specific vs Other Epithelial", top_n = 20)
ggsave(file.path(output_dir, "Volcano_Tumor_specific_vs_otherEpithelial.pdf"), p_vol_tumor,
       width = 100, height = 100, units = "mm")

# GO for Tumor_specific markers
tumor_markers <- deg_tumor %>% filter(p_val_adj < 0.05 & avg_log2FC > 1) %>% pull(gene)
cat(sprintf("    Tumor_specific markers (log2FC>1): %d\n", length(tumor_markers)))

go_tumor <- run_GO(tumor_markers, "Tumor_specific_markers")
if (!is.null(go_tumor) && nrow(as.data.frame(go_tumor)) > 0) {
  write.csv(as.data.frame(go_tumor), file.path(output_dir, "Table_GO_Tumor_specific_markers.csv"), row.names = FALSE)
  p <- dotplot(go_tumor, showCategory = 20, font.size = 7) + theme_paper() +
    ggtitle("Tumor_specific Markers: GO BP") + theme(axis.text.y = element_text(size = 5))
  ggsave(file.path(output_dir, "GO_dotplot_Tumor_specific_markers.pdf"), p,
         width = 150, height = 140, units = "mm")
}

# GSEA for Tumor_specific
gsea_tumor <- run_GSEA(deg_tumor, "Tumor_specific")
if (!is.null(gsea_tumor) && nrow(as.data.frame(gsea_tumor)) > 0) {
  gsea_tumor_df <- as.data.frame(gsea_tumor)
  write.csv(gsea_tumor_df, file.path(output_dir, "Table_GSEA_Tumor_specific.csv"), row.names = FALSE)
  
  gsea_plot_df <- gsea_tumor_df %>%
    arrange(NES) %>%
    mutate(pathway = gsub("HALLMARK_", "", ID),
           pathway = factor(pathway, levels = pathway),
           color = ifelse(NES > 0, "Enriched in Tumor_specific", "Depleted"))
  
  p_nes <- ggplot(gsea_plot_df, aes(x = NES, y = pathway, fill = color)) +
    geom_col(width = 0.7) + geom_vline(xintercept = 0, linewidth = 0.3) +
    scale_fill_manual(values = c("Enriched in Tumor_specific" = "#D32F2F",
                                 "Depleted" = "#1976D2")) +
    theme_paper(base_size = 7) +
    labs(x = "NES", y = "", fill = "", title = "Tumor_specific: GSEA Hallmark") +
    theme(axis.text.y = element_text(size = 5), legend.position = "bottom")
  
  ggsave(file.path(output_dir, "GSEA_NES_barplot_Tumor_specific.pdf"), p_nes,
         width = 140, height = max(60, nrow(gsea_plot_df) * 5 + 30), units = "mm")
}
rm(tumor_compare); gc(verbose = FALSE)

# --- Myofibroblast vs CAF (cell identity comparison) ---
cat("\n  Myofibroblast vs CAF...\n")
stromal_sub <- subset(seu, cell_type %in% c("Myofibroblast", "CAF", "Fibroblast_normal"))
Idents(stromal_sub) <- "cell_type"

deg_myo_vs_caf <- FindMarkers(stromal_sub, ident.1 = "Myofibroblast", ident.2 = "CAF",
                              min.pct = 0.1, logfc.threshold = 0.25, verbose = FALSE)
deg_myo_vs_caf$gene <- rownames(deg_myo_vs_caf)
write.csv(deg_myo_vs_caf, file.path(output_dir, "Table_DEG_Myofibroblast_vs_CAF.csv"), row.names = FALSE)

p_vol_caf <- plot_volcano(deg_myo_vs_caf, "Myofibroblast vs CAF", top_n = 20)
ggsave(file.path(output_dir, "Volcano_Myofibroblast_vs_CAF.pdf"), p_vol_caf,
       width = 100, height = 100, units = "mm")

# GO for Myofibroblast vs CAF
myo_up_vs_caf <- deg_myo_vs_caf %>% filter(p_val_adj < 0.05 & avg_log2FC > 0.5) %>% pull(gene)
go_myo_caf <- run_GO(myo_up_vs_caf, "Myofibroblast_vs_CAF_UP")
if (!is.null(go_myo_caf) && nrow(as.data.frame(go_myo_caf)) > 0) {
  write.csv(as.data.frame(go_myo_caf), file.path(output_dir, "Table_GO_Myofibroblast_vs_CAF.csv"), row.names = FALSE)
  p <- dotplot(go_myo_caf, showCategory = 15, font.size = 7) + theme_paper() +
    ggtitle("Myofibroblast vs CAF: GO BP (Up in Myofib)") +
    theme(axis.text.y = element_text(size = 5))
  ggsave(file.path(output_dir, "GO_dotplot_Myofibroblast_vs_CAF.pdf"), p,
         width = 150, height = 130, units = "mm")
}

# GSEA for Myofibroblast vs CAF
gsea_myo_caf <- run_GSEA(deg_myo_vs_caf, "Myofibroblast_vs_CAF")
if (!is.null(gsea_myo_caf) && nrow(as.data.frame(gsea_myo_caf)) > 0) {
  gsea_myo_caf_df <- as.data.frame(gsea_myo_caf)
  write.csv(gsea_myo_caf_df, file.path(output_dir, "Table_GSEA_Myofibroblast_vs_CAF.csv"), row.names = FALSE)
  
  gsea_plot_df <- gsea_myo_caf_df %>%
    arrange(NES) %>%
    mutate(pathway = gsub("HALLMARK_", "", ID),
           pathway = factor(pathway, levels = pathway),
           color = ifelse(NES > 0, "Enriched in Myofibroblast", "Enriched in CAF"))
  
  p_nes <- ggplot(gsea_plot_df, aes(x = NES, y = pathway, fill = color)) +
    geom_col(width = 0.7) + geom_vline(xintercept = 0, linewidth = 0.3) +
    scale_fill_manual(values = c("Enriched in Myofibroblast" = "#D32F2F",
                                 "Enriched in CAF" = "#1976D2")) +
    theme_paper(base_size = 7) +
    labs(x = "NES", y = "", fill = "", title = "Myofibroblast vs CAF: GSEA Hallmark") +
    theme(axis.text.y = element_text(size = 5), legend.position = "bottom")
  
  ggsave(file.path(output_dir, "GSEA_NES_barplot_Myofibroblast_vs_CAF.pdf"), p_nes,
         width = 140, height = max(60, nrow(gsea_plot_df) * 5 + 30), units = "mm")
}
rm(stromal_sub); gc(verbose = FALSE)

# ===========================================================
# PART 7: Final Summary
# ===========================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════╗\n")
cat("║        04a_CD8_Subtyping_HighRes.R — COMPLETE                   ║\n")
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                  ║\n")
cat("║  PART 1: High-resolution CD8 sub-clustering (res=0.8)           ║\n")
cat("║  PART 2: Proportion comparison (Wilcoxon per subtype/cluster)    ║\n")
cat("║  PART 3: DEG analysis (High vs Low Risk per subtype)             ║\n")
cat("║  PART 4: GO enrichment (Up & Down per subtype)                   ║\n")
cat("║  PART 5: GSEA Hallmark (per subtype + combined heatmap)          ║\n")
cat("║  PART 6: Focus clusters (Myofib, Meso, Tumor_specific)          ║\n")
cat("║                                                                  ║\n")
cat("╠══════════════════════════════════════════════════════════════════╣\n")
cat("║  Output directory:", output_dir, "\n")
cat("║                                                                  ║\n")
cat("║  Key figures:                                                    ║\n")
cat("║    Fig1_CD8_UMAP_highres.pdf                                     ║\n")
cat("║    Fig2_CD8_key_markers_featureplot.pdf                          ║\n")
cat("║    Fig3_CD8_subtype_dotplot.pdf                                  ║\n")
cat("║    Fig4_Boxplot_CD8_subtypes_combined.pdf                        ║\n")
cat("║    Fig5_Stacked_barplot_CD8_subtypes.pdf                         ║\n")
cat("║    Fig6_GSEA_heatmap_CD8_subtypes.pdf                            ║\n")
cat("║    Volcano_<subtype>.pdf (per subtype)                           ║\n")
cat("║    GO_UP/DOWN_dotplot_<subtype>.pdf                              ║\n")
cat("║    GSEA_NES_barplot_<subtype>.pdf                                ║\n")
cat("║    GSEA_running_<subtype>_<pathway>.pdf                          ║\n")
cat("║    Volcano_Myofibroblast_*.pdf                                   ║\n")
cat("║    Volcano_Mesothelial_*.pdf                                     ║\n")
cat("║    Volcano_Tumor_specific_*.pdf                                  ║\n")
cat("║                                                                  ║\n")
cat("║  Key tables:                                                     ║\n")
cat("║    Table_CD8_subcluster_allmarkers.csv                           ║\n")
cat("║    Table_CD8_subtype_Wilcoxon_highres.csv                        ║\n")
cat("║    Table_CD8_cluster_Wilcoxon.csv                                ║\n")
cat("║    Table_DEG_<subtype>.csv                                       ║\n")
cat("║    Table_DEG_summary_CD8_subtypes.csv                            ║\n")
cat("║    Table_GO_UP/DOWN_<subtype>.csv                                ║\n")
cat("║    Table_GSEA_<subtype>.csv                                      ║\n")
cat("║    Table_GSEA_all_CD8_subtypes.csv                               ║\n")
cat("║    Table_DEG_Myofibroblast_*.csv                                 ║\n")
cat("║    Table_DEG_Mesothelial_*.csv                                   ║\n")
cat("║    Table_DEG_Tumor_specific_*.csv                                ║\n")
cat("╚══════════════════════════════════════════════════════════════════╝\n")

cat("\n\nNext steps:\n")
cat("  1. Check Table_CD8_subtype_Wilcoxon_highres.csv for significant subtypes\n")
cat("  2. Check Table_CD8_cluster_Wilcoxon.csv for significant raw clusters\n")
cat("  3. If still no p<0.05, the CD8 difference may be in gene expression\n")
cat("     (captured by DEG/GSEA) rather than proportion\n")
cat("  4. Combine individual PDF figures as needed for manuscript\n")
cat("  5. Run 05_CellCommunication.R for CellChat analysis\n")

cat("\nDone.\n")