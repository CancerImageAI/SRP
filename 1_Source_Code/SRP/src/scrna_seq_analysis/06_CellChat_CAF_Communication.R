# ============================================================
# 06_CellChat_CAF_Communication_v2.R
# 
# Cell-cell communication analysis: CAF → other cell types
# Comparing Low_Risk vs High_Risk groups
#
# Key outputs:
# 1. Number of significant L-R pairs from CAF to each cell type
# 2. Heatmap of ligand activity/regulatory potential
# 3. Revised circle plots without overlaps
# ============================================================

start_time <- Sys.time()

options(Seurat.object.assay.version = "v3")
options(future.globals.maxSize = 8 * 1024^3)

library(Seurat)
library(CellChat)
library(patchwork)
library(tidyverse)
library(ComplexHeatmap)
library(circlize)
library(ggrepel)
library(RColorBrewer)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/03_Annotation"
output_dir <- "XXXyour path/scRNA_Results_new/06_CellChat"
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
      axis.line = element_line(linewidth = 0.4, color = "black"),
      axis.ticks = element_line(linewidth = 0.3, color = "black"),
      panel.grid = element_blank(),
      plot.margin = margin(4, 4, 4, 4, "pt")
    )
}

# ============================================================
# PART 1: Load and Prepare Data
# ============================================================
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 1: Loading and preparing data\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

seu <- readRDS(file.path(input_dir, "seu_annotated.rds"))
DefaultAssay(seu) <- "RNA"

cat(sprintf("  Total cells: %d\n", ncol(seu)))
cat(sprintf("  Cell types: %d\n", length(unique(seu$cell_type))))
cat("\n  Cells per group:\n")
print(table(seu$group))

# ============================================================
# PART 2: Filter and Create CellChat Objects
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 2: Creating CellChat objects\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Filter cell types with >= 50 cells in BOTH groups
min_cells <- 50
ct_counts_high <- table(seu$cell_type[seu$group == "High_Risk"])
ct_counts_low <- table(seu$cell_type[seu$group == "Low_Risk"])
ct_high <- names(ct_counts_high[ct_counts_high >= min_cells])
ct_low <- names(ct_counts_low[ct_counts_low >= min_cells])
ct_common <- intersect(ct_high, ct_low)

stopifnot("CAF" %in% ct_common)

cat(sprintf("\n  Cell types with >= %d cells in BOTH groups: %d\n", min_cells, length(ct_common)))
cat("  ", paste(ct_common, collapse = ", "), "\n")

# Subset
seu_high <- subset(seu, subset = group == "High_Risk" & cell_type %in% ct_common)
seu_low <- subset(seu, subset = group == "Low_Risk" & cell_type %in% ct_common)

# Downsample
MAX_CELLS <- 30000
set.seed(42)
if (ncol(seu_high) > MAX_CELLS) {
  seu_high <- subset(seu_high, cells = sample(colnames(seu_high), MAX_CELLS))
  cat(sprintf("  High_Risk downsampled to %d cells\n", MAX_CELLS))
}
if (ncol(seu_low) > MAX_CELLS) {
  seu_low <- subset(seu_low, cells = sample(colnames(seu_low), MAX_CELLS))
  cat(sprintf("  Low_Risk downsampled to %d cells\n", MAX_CELLS))
}

cat(sprintf("\n  Final: High_Risk = %d cells, Low_Risk = %d cells\n",
            ncol(seu_high), ncol(seu_low)))

# ── Helper to safely extract normalized data ──
get_normalized_data <- function(seu_obj) {
  assay_obj <- seu_obj[["RNA"]]
  assay_class <- class(assay_obj)[1]
  
  if (assay_class == "Assay5") {
    if ("data" %in% Layers(assay_obj)) {
      data_mat <- GetAssayData(seu_obj, assay = "RNA", layer = "data")
    } else {
      seu_obj <- NormalizeData(seu_obj, verbose = FALSE)
      data_mat <- GetAssayData(seu_obj, assay = "RNA", layer = "data")
    }
  } else {
    data_mat <- tryCatch(
      seu_obj[["RNA"]]@data,
      error = function(e) {
        GetAssayData(seu_obj, assay = "RNA", layer = "data")
      }
    )
  }
  return(data_mat)
}

# --- Function to create CellChat object ---
create_cellchat <- function(seu_obj, group_name) {
  cat(sprintf("\n  [%s] Creating CellChat object (%d cells)...\n", group_name, ncol(seu_obj)))
  
  data_input <- get_normalized_data(seu_obj)
  
  max_val <- max(data_input[1:min(100, nrow(data_input)), 1:min(100, ncol(data_input))])
  cat(sprintf("  [%s] Data: %d genes × %d cells (max=%.2f)\n",
              group_name, nrow(data_input), ncol(data_input), max_val))
  
  if (max_val > 100) {
    seu_obj <- NormalizeData(seu_obj, verbose = FALSE)
    data_input <- seu_obj[["RNA"]]@data
  }
  
  meta <- seu_obj@meta.data
  meta$labels <- meta$cell_type
  if (!"samples" %in% colnames(meta)) meta$samples <- group_name
  
  cellchat <- createCellChat(object = data_input, meta = meta, group.by = "labels")
  cellchat@DB <- CellChatDB.human
  
  cat(sprintf("  [%s] Processing...\n", group_name))
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  cellchat <- computeCommunProb(cellchat, type = "triMean")
  cellchat <- filterCommunication(cellchat, min.cells = 10)
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)
  
  cat(sprintf("  [%s] Complete.\n", group_name))
  return(cellchat)
}

# Create or load CellChat objects
cellchat_high_file <- file.path(output_dir, "cellchat_High_Risk.rds")
cellchat_low_file <- file.path(output_dir, "cellchat_Low_Risk.rds")

if (file.exists(cellchat_high_file) & file.exists(cellchat_low_file)) {
  cat("\n  Loading existing CellChat objects...\n")
  cellchat_high <- readRDS(cellchat_high_file)
  cellchat_low <- readRDS(cellchat_low_file)
} else {
  cellchat_high <- create_cellchat(seu_high, "High_Risk")
  cellchat_low <- create_cellchat(seu_low, "Low_Risk")
  saveRDS(cellchat_high, cellchat_high_file)
  saveRDS(cellchat_low, cellchat_low_file)
}

# Merge
object.list <- list(Low_Risk = cellchat_low, High_Risk = cellchat_high)
cellchat_merged <- mergeCellChat(object.list, add.names = names(object.list))

cat("\n  CellChat objects ready.\n")

# ============================================================
# PART 3: Number of Significant L-R Pairs from CAF
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 3: Significant L-R pairs from CAF to each cell type\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Extract all significant CAF outgoing interactions
lr_high <- subsetCommunication(cellchat_high, sources.use = "CAF")
lr_low <- subsetCommunication(cellchat_low, sources.use = "CAF")

cat(sprintf("\n  Total significant L-R pairs from CAF:\n"))
cat(sprintf("    High_Risk: %d\n", nrow(lr_high)))
cat(sprintf("    Low_Risk:  %d\n", nrow(lr_low)))

# Count per target cell type
count_high <- lr_high %>%
  group_by(target) %>%
  summarise(n_LR_High = n(), total_prob_High = sum(prob), .groups = "drop")

count_low <- lr_low %>%
  group_by(target) %>%
  summarise(n_LR_Low = n(), total_prob_Low = sum(prob), .groups = "drop")

# Merge counts
lr_counts <- full_join(count_high, count_low, by = "target") %>%
  mutate(
    n_LR_High = replace_na(n_LR_High, 0L),
    n_LR_Low = replace_na(n_LR_Low, 0L),
    total_prob_High = replace_na(total_prob_High, 0),
    total_prob_Low = replace_na(total_prob_Low, 0),
    diff_count = n_LR_High - n_LR_Low,
    diff_prob = total_prob_High - total_prob_Low
  ) %>%
  arrange(desc(diff_count))

write.csv(lr_counts, file.path(output_dir, "Table_CAF_LR_counts_per_target.csv"),
          row.names = FALSE)

cat("\n  Number of significant L-R pairs from CAF to each cell type:\n")
cat("  ═══════════════════════════════════════════════════════════════════\n")
print(lr_counts %>%
        dplyr::select(target, n_LR_High, n_LR_Low, diff_count, total_prob_High, total_prob_Low) %>%
        mutate(across(where(is.numeric), ~round(., 4))),
      row.names = FALSE, n = 30)

# ─── Barplot: Number of L-R pairs per target ───
lr_counts_long <- lr_counts %>%
  dplyr::select(target, n_LR_High, n_LR_Low) %>%
  pivot_longer(cols = c(n_LR_High, n_LR_Low), names_to = "group", values_to = "n_LR") %>%
  mutate(group = ifelse(group == "n_LR_High", "High_Risk", "Low_Risk"))

# Order targets by High_Risk count
target_order <- lr_counts %>% arrange(n_LR_High) %>% pull(target)
lr_counts_long$target <- factor(lr_counts_long$target, levels = target_order)

p_lr_count <- ggplot(lr_counts_long, aes(x = target, y = n_LR, fill = group)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("High_Risk" = "#D32F2F", "Low_Risk" = "#1976D2"),
                    name = "Group") +
  theme_paper(base_size = 8) +
  labs(x = "Target Cell Type",
       y = "Number of Significant L-R Pairs",
       title = "CAF → Target: Number of Significant L-R Pairs\n(CellChat, High_Risk vs Low_Risk)") +
  theme(axis.text.y = element_text(size = 7),
        legend.position = "top",
        legend.text = element_text(size = 7))

ggsave(file.path(output_dir, "Fig_LR_count_per_target.pdf"), p_lr_count,
       width = 160, height = 140, units = "mm")

cat("\n  L-R count barplot saved.\n")

# ============================================================
# PART 4: Heatmap of Ligand Activity & Regulatory Potential
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 4: Heatmap of ligand-receptor regulatory potential\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Priority ligands from DEG analysis (upregulated in High_Risk CAFs)
priority_ligands <- c("PTGIS", "FGF7", "CCL2", "VEGFA", "TNC", "CCN2", "VEGFC",
                      "GAS6", "NTN1", "THBS1", "THBS2", "LTBP1", "LTBP2",
                      "INHBA", "MDK", "SLIT2", "CSF1", "CXCL2", "PDGFC",
                      "A2M", "SERPINE1", "TIMP1")

# Extract L-R pairs involving these ligands
lr_priority_high <- lr_high %>% filter(ligand %in% priority_ligands)
lr_priority_low <- lr_low %>% filter(ligand %in% priority_ligands)

cat(sprintf("  Priority ligand interactions: High=%d, Low=%d\n",
            nrow(lr_priority_high), nrow(lr_priority_low)))

# ── Build heatmap matrix: Ligand × Target (communication probability) ──
# For High_Risk
prob_matrix_high <- lr_priority_high %>%
  group_by(ligand, target) %>%
  summarise(prob = sum(prob), .groups = "drop") %>%
  pivot_wider(names_from = target, values_from = prob, values_fill = 0) %>%
  column_to_rownames("ligand") %>%
  as.matrix()

# For Low_Risk
prob_matrix_low <- lr_priority_low %>%
  group_by(ligand, target) %>%
  summarise(prob = sum(prob), .groups = "drop") %>%
  pivot_wider(names_from = target, values_from = prob, values_fill = 0) %>%
  column_to_rownames("ligand") %>%
  as.matrix()

# Unify dimensions
all_ligands <- union(rownames(prob_matrix_high), rownames(prob_matrix_low))
all_targets_hm <- union(colnames(prob_matrix_high), colnames(prob_matrix_low))

# Create unified matrices
mat_high <- matrix(0, nrow = length(all_ligands), ncol = length(all_targets_hm),
                   dimnames = list(all_ligands, all_targets_hm))
mat_low <- matrix(0, nrow = length(all_ligands), ncol = length(all_targets_hm),
                  dimnames = list(all_ligands, all_targets_hm))

mat_high[rownames(prob_matrix_high), colnames(prob_matrix_high)] <- prob_matrix_high
mat_low[rownames(prob_matrix_low), colnames(prob_matrix_low)] <- prob_matrix_low

# Filter: keep ligands present in at least one group
active_ligands <- all_ligands[rowSums(mat_high) > 0 | rowSums(mat_low) > 0]
active_targets <- all_targets_hm[colSums(mat_high) > 0 | colSums(mat_low) > 0]

mat_high <- mat_high[active_ligands, active_targets, drop = FALSE]
mat_low <- mat_low[active_ligands, active_targets, drop = FALSE]

# Order ligands by total activity difference
ligand_diff <- rowSums(mat_high) - rowSums(mat_low)
ligand_order <- names(sort(ligand_diff, decreasing = TRUE))
mat_high <- mat_high[ligand_order, , drop = FALSE]
mat_low <- mat_low[ligand_order, , drop = FALSE]

cat(sprintf("  Heatmap dimensions: %d ligands × %d targets\n",
            length(ligand_order), length(active_targets)))

# ── Heatmap 1: High_Risk ──
col_fun_high <- colorRamp2(c(0, max(mat_high)/2, max(mat_high)),
                           c("white", "#FFCDD2", "#B71C1C"))

ht_high <- Heatmap(mat_high,
                   name = "Prob\n(High)",
                   col = col_fun_high,
                   cluster_rows = FALSE,
                   cluster_columns = TRUE,
                   row_names_gp = gpar(fontsize = 7, fontface = "italic"),
                   column_names_gp = gpar(fontsize = 7),
                   column_names_rot = 45,
                   column_title = "High_Risk: CAF Ligand → Target",
                   column_title_gp = gpar(fontsize = 9, fontface = "bold"),
                   row_title = "CAF Ligands (DEG-upregulated)",
                   row_title_gp = gpar(fontsize = 8, fontface = "bold"),
                   heatmap_legend_param = list(title_gp = gpar(fontsize = 7),
                                               labels_gp = gpar(fontsize = 6)),
                   width = unit(ncol(mat_high) * 8, "mm"),
                   height = unit(nrow(mat_high) * 5, "mm"))

# ── Heatmap 2: Low_Risk ──
col_fun_low <- colorRamp2(c(0, max(mat_low)/2, max(mat_low)),
                          c("white", "#BBDEFB", "#0D47A1"))

ht_low <- Heatmap(mat_low,
                  name = "Prob\n(Low)",
                  col = col_fun_low,
                  cluster_rows = FALSE,
                  cluster_columns = TRUE,
                  row_names_gp = gpar(fontsize = 7, fontface = "italic"),
                  column_names_gp = gpar(fontsize = 7),
                  column_names_rot = 45,
                  column_title = "Low_Risk: CAF Ligand → Target",
                  column_title_gp = gpar(fontsize = 9, fontface = "bold"),
                  row_title = "CAF Ligands (DEG-upregulated)",
                  row_title_gp = gpar(fontsize = 8, fontface = "bold"),
                  heatmap_legend_param = list(title_gp = gpar(fontsize = 7),
                                              labels_gp = gpar(fontsize = 6)),
                  width = unit(ncol(mat_low) * 8, "mm"),
                  height = unit(nrow(mat_low) * 5, "mm"))

# Save side by side
pdf(file.path(output_dir, "Fig_Heatmap_Ligand_Activity_HighVsLow.pdf"),
    width = 14, height = 8)
draw(ht_high + ht_low, column_title = "CAF Ligand Regulatory Potential: High_Risk vs Low_Risk",
     column_title_gp = gpar(fontsize = 11, fontface = "bold"))
dev.off()

# ── Heatmap 3: Difference (High - Low) ──
mat_diff <- mat_high - mat_low

col_fun_diff <- colorRamp2(c(-max(abs(mat_diff)), 0, max(abs(mat_diff))),
                           c("#1565C0", "white", "#C62828"))

ht_diff <- Heatmap(mat_diff,
                   name = "Δ Prob\n(H-L)",
                   col = col_fun_diff,
                   cluster_rows = FALSE,
                   cluster_columns = TRUE,
                   row_names_gp = gpar(fontsize = 7, fontface = "italic"),
                   column_names_gp = gpar(fontsize = 7),
                   column_names_rot = 45,
                   column_title = "Difference (High_Risk - Low_Risk)",
                   column_title_gp = gpar(fontsize = 9, fontface = "bold"),
                   row_title = "CAF Ligands (DEG-upregulated)",
                   row_title_gp = gpar(fontsize = 8, fontface = "bold"),
                   heatmap_legend_param = list(title_gp = gpar(fontsize = 7),
                                               labels_gp = gpar(fontsize = 6)),
                   cell_fun = function(j, i, x, y, width, height, fill) {
                     # Add asterisk for large differences
                     if (abs(mat_diff[i, j]) > quantile(abs(mat_diff[mat_diff != 0]), 0.75, na.rm = TRUE)) {
                       grid.text("*", x, y, gp = gpar(fontsize = 6))
                     }
                   },
                   width = unit(ncol(mat_diff) * 8, "mm"),
                   height = unit(nrow(mat_diff) * 5, "mm"))

pdf(file.path(output_dir, "Fig_Heatmap_Ligand_Diff_HighMinusLow.pdf"),
    width = 10, height = 8)
draw(ht_diff, column_title = "CAF Ligand Activity Difference\n(High_Risk - Low_Risk)",
     column_title_gp = gpar(fontsize = 11, fontface = "bold"))
dev.off()

cat("  Heatmaps saved.\n")

# ── Heatmap 4: Detailed L-R level heatmap (Ligand-Receptor pairs) ──
cat("\n  Creating detailed L-R pair heatmap...\n")

# Build L-R × Target matrix for both groups
lr_detail_high <- lr_priority_high %>%
  mutate(LR_pair = paste0(ligand, " → ", receptor)) %>%
  group_by(LR_pair, target) %>%
  summarise(prob = sum(prob), .groups = "drop")

lr_detail_low <- lr_priority_low %>%
  mutate(LR_pair = paste0(ligand, " → ", receptor)) %>%
  group_by(LR_pair, target) %>%
  summarise(prob = sum(prob), .groups = "drop")

# Get top L-R pairs by probability in High_Risk
top_lr_pairs <- lr_detail_high %>%
  group_by(LR_pair) %>%
  summarise(max_prob = max(prob), .groups = "drop") %>%
  arrange(desc(max_prob)) %>%
  head(30) %>%
  pull(LR_pair)

# Build matrices for top pairs
lr_mat_high <- lr_detail_high %>%
  filter(LR_pair %in% top_lr_pairs) %>%
  pivot_wider(names_from = target, values_from = prob, values_fill = 0) %>%
  column_to_rownames("LR_pair") %>%
  as.matrix()

lr_mat_low <- lr_detail_low %>%
  filter(LR_pair %in% top_lr_pairs) %>%
  pivot_wider(names_from = target, values_from = prob, values_fill = 0) %>%
  column_to_rownames("LR_pair") %>%
  as.matrix()

# Unify
all_lr <- union(rownames(lr_mat_high), rownames(lr_mat_low))
all_tgt_lr <- union(colnames(lr_mat_high), colnames(lr_mat_low))

lr_unified_high <- matrix(0, nrow = length(all_lr), ncol = length(all_tgt_lr),
                          dimnames = list(all_lr, all_tgt_lr))
lr_unified_low <- matrix(0, nrow = length(all_lr), ncol = length(all_tgt_lr),
                         dimnames = list(all_lr, all_tgt_lr))

lr_unified_high[rownames(lr_mat_high), colnames(lr_mat_high)] <- lr_mat_high
lr_unified_low[rownames(lr_mat_low), colnames(lr_mat_low)] <- lr_mat_low

# Sort by difference
lr_diff_sum <- rowSums(lr_unified_high) - rowSums(lr_unified_low)
lr_order <- names(sort(lr_diff_sum, decreasing = TRUE))
lr_unified_high <- lr_unified_high[lr_order, , drop = FALSE]
lr_unified_low <- lr_unified_low[lr_order, , drop = FALSE]

# Ligand annotation for row colors
ligand_from_lr <- sub(" → .*", "", lr_order)
ligand_colors <- setNames(
  colorRampPalette(brewer.pal(12, "Set3"))(length(unique(ligand_from_lr))),
  unique(ligand_from_lr)
)

row_anno <- rowAnnotation(
  Ligand = ligand_from_lr,
  col = list(Ligand = ligand_colors),
  annotation_legend_param = list(title_gp = gpar(fontsize = 6),
                                 labels_gp = gpar(fontsize = 5)),
  width = unit(4, "mm")
)

# Combined heatmap (High and Low side by side)
max_val_lr <- max(c(lr_unified_high, lr_unified_low))
col_fun_lr <- colorRamp2(c(0, max_val_lr/2, max_val_lr), c("white", "#FFF9C4", "#E65100"))

ht_lr_high <- Heatmap(lr_unified_high,
                      name = "Prob (High)",
                      col = col_fun_lr,
                      cluster_rows = FALSE,
                      cluster_columns = TRUE,
                      show_row_names = TRUE,
                      row_names_gp = gpar(fontsize = 5.5),
                      column_names_gp = gpar(fontsize = 6.5),
                      column_names_rot = 45,
                      column_title = "High_Risk",
                      column_title_gp = gpar(fontsize = 9, fontface = "bold", col = "#D32F2F"),
                      left_annotation = row_anno,
                      width = unit(ncol(lr_unified_high) * 7, "mm"),
                      height = unit(nrow(lr_unified_high) * 4, "mm"))

ht_lr_low <- Heatmap(lr_unified_low,
                     name = "Prob (Low)",
                     col = col_fun_lr,
                     cluster_rows = FALSE,
                     cluster_columns = TRUE,
                     show_row_names = FALSE,
                     column_names_gp = gpar(fontsize = 6.5),
                     column_names_rot = 45,
                     column_title = "Low_Risk",
                     column_title_gp = gpar(fontsize = 9, fontface = "bold", col = "#1976D2"),
                     width = unit(ncol(lr_unified_low) * 7, "mm"),
                     height = unit(nrow(lr_unified_low) * 4, "mm"))

pdf(file.path(output_dir, "Fig_Heatmap_LR_Detail_HighVsLow.pdf"),
    width = 16, height = 10)
draw(ht_lr_high + ht_lr_low,
     column_title = "CAF Ligand-Receptor Regulatory Potential\n(Top 30 L-R pairs by activity)",
     column_title_gp = gpar(fontsize = 11, fontface = "bold"),
     row_title = "L-R Pairs (Ligand → Receptor)",
     row_title_gp = gpar(fontsize = 8))
dev.off()

cat("  Detailed L-R heatmap saved.\n")

# ============================================================
# PART 5: Revised Circle Plots (No Overlap)
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 5: Revised circle plots (no overlap)\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Get CAF indices
caf_idx_high <- which(levels(cellchat_high@idents) == "CAF")
caf_idx_low <- which(levels(cellchat_low@idents) == "CAF")

# Define consistent color palette for cell types
n_ct <- length(ct_common)
ct_colors <- setNames(
  c(brewer.pal(12, "Set3"), brewer.pal(8, "Dark2"))[1:n_ct],
  sort(ct_common)
)
# Make CAF prominent
ct_colors["CAF"] <- "#FF0000"

# ─── Fixed Custom circle plot function (no overlap) ───
plot_circle_CAF <- function(cellchat_obj, caf_idx, title, ct_colors_input) {
  
  library(igraph)
  
  # Get weight matrix
  net_weight <- cellchat_obj@net$weight
  ct_names <- levels(cellchat_obj@idents)
  n_ct <- length(ct_names)
  
  # CAF outgoing weights
  caf_outgoing <- net_weight[caf_idx, ]
  
  # Cell counts for vertex size
  cell_counts <- as.numeric(table(cellchat_obj@idents))
  names(cell_counts) <- ct_names
  
  # Filter non-zero edges
  non_zero <- caf_outgoing > 0
  if (sum(non_zero) == 0) {
    plot.new()
    title(paste0(title, "\n(No CAF outgoing interactions)"))
    return(invisible(NULL))
  }
  
  # Create edge list
  edge_targets <- ct_names[non_zero]
  edge_weights <- caf_outgoing[non_zero]
  
  edge_df <- data.frame(from = rep("CAF", length(edge_targets)),
                        to = edge_targets,
                        weight = edge_weights,
                        stringsAsFactors = FALSE)
  
  # Remove self-loop for cleaner plot (optional, keep if desired)
  # edge_df <- edge_df[edge_df$to != "CAF", ]
  
  # Create graph
  g <- graph_from_data_frame(edge_df, directed = TRUE, vertices = data.frame(name = ct_names))
  
  # Layout: circle with CAF at top
  ct_order <- c("CAF", sort(setdiff(ct_names, "CAF")))
  angles <- seq(0, 2 * pi, length.out = length(ct_order) + 1)[1:length(ct_order)]
  angles <- angles + pi/2  # Start from top
  
  layout_mat <- cbind(cos(angles), sin(angles))
  rownames(layout_mat) <- ct_order
  layout_use <- layout_mat[V(g)$name, , drop = FALSE]
  
  # Vertex properties
  v_colors <- ct_colors_input[V(g)$name]
  v_colors[is.na(v_colors)] <- "grey70"
  v_sizes <- sqrt(cell_counts[V(g)$name] / max(cell_counts)) * 15
  
  # Edge properties — FIX: vectorized adjustcolor
  edge_w_norm <- E(g)$weight / max(E(g)$weight)
  edge_widths <- edge_w_norm * 5 + 0.5
  
  # Fix: apply adjustcolor element-wise using sapply
  edge_colors <- sapply(edge_w_norm, function(w) {
    adjustcolor("#D32F2F", alpha.f = 0.4 + w * 0.5)
  })
  
  # Plot
  plot(g,
       layout = layout_use,
       vertex.size = v_sizes,
       vertex.color = v_colors,
       vertex.frame.color = "grey30",
       vertex.frame.width = 0.5,
       vertex.label = V(g)$name,
       vertex.label.cex = 0.6,
       vertex.label.color = "black",
       vertex.label.dist = 1.5,
       vertex.label.font = 2,
       edge.width = edge_widths,
       edge.color = edge_colors,
       edge.arrow.size = 0.4,
       edge.curved = 0.2,
       main = title)
}


# ─── Alternative: Use CellChat's built-in with better spacing ───
# The key fix is to increase margins and reduce vertex label overlap

# Fig5a: Circle plot — High_Risk
pdf(file.path(output_dir, "Fig_Circle_CAF_outgoing_High.pdf"), width = 8, height = 8)
par(mar = c(1, 1, 3, 1))
plot_circle_CAF(cellchat_high, caf_idx_high,
                "CAF Outgoing Communication\nHigh_Risk Group",
                ct_colors)
dev.off()

# Fig5b: Circle plot — Low_Risk
pdf(file.path(output_dir, "Fig_Circle_CAF_outgoing_Low.pdf"), width = 8, height = 8)
par(mar = c(1, 1, 3, 1))
plot_circle_CAF(cellchat_low, caf_idx_low,
                "CAF Outgoing Communication\nLow_Risk Group",
                ct_colors)
dev.off()

# Fig5c: Side by side comparison
pdf(file.path(output_dir, "Fig_Circle_CAF_outgoing_comparison.pdf"), width = 16, height = 8)
par(mfrow = c(1, 2), mar = c(1, 1, 3, 1))
plot_circle_CAF(cellchat_low, caf_idx_low,
                "CAF Outgoing: Low_Risk", ct_colors)
plot_circle_CAF(cellchat_high, caf_idx_high,
                "CAF Outgoing: High_Risk", ct_colors)
dev.off()

cat("  Custom circle plots saved.\n")

# ─── Alternative using CellChat's netVisual_circle with fixes ───
cat("\n  Creating CellChat native circle plots with improved layout...\n")

# Fix overlap: Use larger plot, adjusted vertex.label.cex, and better vertex.size
pdf(file.path(output_dir, "Fig_Circle_CellChat_native_comparison.pdf"), width = 18, height = 9)
par(mfrow = c(1, 2), mar = c(0.5, 0.5, 2.5, 0.5))

# Compute vertex weights (normalized cell counts)
vw_low <- as.numeric(table(cellchat_low@idents))
vw_high <- as.numeric(table(cellchat_high@idents))

# Max weight for scaling (use same scale for both)
max_weight <- max(c(cellchat_low@net$weight[caf_idx_low, ],
                    cellchat_high@net$weight[caf_idx_high, ]))

# Low_Risk
netVisual_circle(cellchat_low@net$weight,
                 sources.use = caf_idx_low,
                 vertex.weight = vw_low,
                 weight.scale = TRUE,
                 edge.weight.max = max_weight,
                 label.edge = FALSE,
                 vertex.label.cex = 0.7,
                 vertex.size.max = 12,
                 title.name = "CAF Outgoing Communication: Low_Risk",
                 color.use = ct_colors[levels(cellchat_low@idents)])

# High_Risk
netVisual_circle(cellchat_high@net$weight,
                 sources.use = caf_idx_high,
                 vertex.weight = vw_high,
                 weight.scale = TRUE,
                 edge.weight.max = max_weight,
                 label.edge = FALSE,
                 vertex.label.cex = 0.7,
                 vertex.size.max = 12,
                 title.name = "CAF Outgoing Communication: High_Risk",
                 color.use = ct_colors[levels(cellchat_high@idents)])

dev.off()

# All interactions (not just CAF)
pdf(file.path(output_dir, "Fig_Circle_AllInteractions_comparison.pdf"), width = 18, height = 9)
par(mfrow = c(1, 2), mar = c(0.5, 0.5, 2.5, 0.5))

netVisual_circle(cellchat_low@net$weight,
                 vertex.weight = vw_low,
                 weight.scale = TRUE,
                 label.edge = FALSE,
                 vertex.label.cex = 0.65,
                 vertex.size.max = 10,
                 title.name = "All Cell-Cell Communication: Low_Risk",
                 color.use = ct_colors[levels(cellchat_low@idents)])

netVisual_circle(cellchat_high@net$weight,
                 vertex.weight = vw_high,
                 weight.scale = TRUE,
                 label.edge = FALSE,
                 vertex.label.cex = 0.65,
                 vertex.size.max = 10,
                 title.name = "All Cell-Cell Communication: High_Risk",
                 color.use = ct_colors[levels(cellchat_high@idents)])

dev.off()

cat("  CellChat native circle plots saved.\n")

# ============================================================
# PART 6: Overall Comparison Figures
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 6: Overall comparison figures\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Fig: Number and strength comparison
pdf(file.path(output_dir, "Fig_Overall_interaction_comparison.pdf"), width = 10, height = 5)
gg1 <- compareInteractions(cellchat_merged, show.legend = FALSE, group = c(1, 2))
gg2 <- compareInteractions(cellchat_merged, show.legend = FALSE, group = c(1, 2), measure = "weight")
print(gg1 + gg2)
dev.off()

# Fig: Differential heatmap
pdf(file.path(output_dir, "Fig_Diff_heatmap.pdf"), width = 12, height = 6)
gg1 <- netVisual_heatmap(cellchat_merged)
gg2 <- netVisual_heatmap(cellchat_merged, measure = "weight")
print(gg1 + gg2)
dev.off()

# Fig: Information flow
tryCatch({
  pdf(file.path(output_dir, "Fig_Information_flow.pdf"), width = 10, height = 8)
  gg_flow <- rankNet(cellchat_merged, mode = "comparison", stacked = TRUE, do.stat = TRUE)
  print(gg_flow)
  dev.off()
}, error = function(e) {
  cat("  Information flow skipped:", e$message, "\n")
  if (dev.cur() > 1) dev.off()
})

cat("  Overall comparison figures saved.\n")

# ============================================================
# PART 7: Bubble Plots
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 7: Bubble plots\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# CAF outgoing — all targets
tryCatch({
  pdf(file.path(output_dir, "Fig_Bubble_CAF_outgoing.pdf"), width = 16, height = 14)
  netVisual_bubble(cellchat_merged,
                   sources.use = "CAF",
                   comparison = c(1, 2),
                   angle.x = 45,
                   remove.isolate = TRUE,
                   title.name = "CAF → All Targets: Low_Risk vs High_Risk")
  dev.off()
  cat("  Bubble (CAF outgoing) saved.\n")
}, error = function(e) {
  cat("  Bubble (CAF outgoing) skipped:", e$message, "\n")
  if (dev.cur() > 1) dev.off()
})

# CAF incoming — all sources
tryCatch({
  pdf(file.path(output_dir, "Fig_Bubble_CAF_incoming.pdf"), width = 16, height = 14)
  netVisual_bubble(cellchat_merged,
                   targets.use = "CAF",
                   comparison = c(1, 2),
                   angle.x = 45,
                   remove.isolate = TRUE,
                   title.name = "All Sources → CAF: Low_Risk vs High_Risk")
  dev.off()
  cat("  Bubble (CAF incoming) saved.\n")
}, error = function(e) {
  cat("  Bubble (CAF incoming) skipped:", e$message, "\n")
  if (dev.cur() > 1) dev.off()
})

cat("  Bubble plots saved.\n")

# CAF → Immune cells
immune_candidates <- c("CD8_T_effector", "Treg", "Macrophage_SPP1",
                       "Neutrophil", "DC_mature", "B_cell", "Mast_cell")
immune_present <- intersect(immune_candidates, ct_common)

if (length(immune_present) > 0) {
  tryCatch({
    pdf(file.path(output_dir, "Fig_Bubble_CAF_to_immune.pdf"), width = 16, height = 12)
    netVisual_bubble(cellchat_merged,
                     sources.use = "CAF",
                     targets.use = immune_present,
                     comparison = c(1, 2),
                     angle.x = 45,
                     remove.isolate = TRUE,
                     title.name = "CAF → Immune Cells: Low_Risk vs High_Risk")
    dev.off()
    cat("  Bubble (CAF → Immune) saved.\n")
  }, error = function(e) {
    cat("  Bubble (CAF → Immune) skipped:", e$message, "\n")
    if (dev.cur() > 1) dev.off()
  })
}

# CAF → Epithelial/Tumor
epi_candidates <- c("Epithelial_basal", "Epithelial_squamous",
                    "Epithelial_differentiated", "Epithelial_stem_like")
epi_present <- intersect(epi_candidates, ct_common)

if (length(epi_present) > 0) {
  tryCatch({
    pdf(file.path(output_dir, "Fig_Bubble_CAF_to_epithelial.pdf"), width = 16, height = 12)
    netVisual_bubble(cellchat_merged,
                     sources.use = "CAF",
                     targets.use = epi_present,
                     comparison = c(1, 2),
                     angle.x = 45,
                     remove.isolate = TRUE,
                     title.name = "CAF → Epithelial: Low_Risk vs High_Risk")
    dev.off()
    cat("  Bubble (CAF → Epithelial) saved.\n")
  }, error = function(e) {
    cat("  Bubble (CAF → Epithelial) skipped:", e$message, "\n")
    if (dev.cur() > 1) dev.off()
  })
}

# CAF → Stromal
stromal_candidates <- c("Endothelial", "Pericyte_SMC", "Myofibroblast",
                        "Fibroblast_normal", "CAF")
stromal_present <- intersect(stromal_candidates, ct_common)

if (length(stromal_present) > 0) {
  tryCatch({
    pdf(file.path(output_dir, "Fig_Bubble_CAF_to_stromal.pdf"), width = 16, height = 12)
    netVisual_bubble(cellchat_merged,
                     sources.use = "CAF",
                     targets.use = stromal_present,
                     comparison = c(1, 2),
                     angle.x = 45,
                     remove.isolate = TRUE,
                     title.name = "CAF → Stromal: Low_Risk vs High_Risk")
    dev.off()
    cat("  Bubble (CAF → Stromal) saved.\n")
  }, error = function(e) {
    cat("  Bubble (CAF → Stromal) skipped:", e$message, "\n")
    if (dev.cur() > 1) dev.off()
  })
}

# ============================================================
# PART 8: Key Pathway Chord Diagrams
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 8: Key pathway chord diagrams\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Priority pathways
priority_pathways <- c("VEGF", "FGF", "CCL", "CXCL", "TGFb", "PDGF",
                       "COLLAGEN", "FN1", "THBS", "MK", "GAS", "SLIT",
                       "TENASCIN", "CSF", "ACTIVIN", "NOTCH", "WNT",
                       "EGF", "IGF", "SPP1", "LAMININ", "NETRIN")

pathways_in_high <- cellchat_high@netP$pathways
pathways_in_low <- cellchat_low@netP$pathways
focus_pathways <- intersect(priority_pathways, union(pathways_in_high, pathways_in_low))

cat(sprintf("  Focus pathways found: %d\n", length(focus_pathways)))
cat("  ", paste(focus_pathways, collapse = ", "), "\n")

for (pw in focus_pathways) {
  in_high <- pw %in% pathways_in_high
  in_low <- pw %in% pathways_in_low
  
  tryCatch({
    pdf(file.path(output_dir, sprintf("Fig_Chord_%s_comparison.pdf", pw)), width = 16, height = 8)
    par(mfrow = c(1, 2), mar = c(1, 1, 3, 1))
    
    if (in_low) {
      netVisual_aggregate(cellchat_low, signaling = pw, layout = "chord",
                          sources.use = caf_idx_low,
                          title.name = paste0(pw, " Pathway: Low_Risk (CAF→)"))
    } else {
      plot.new()
      title(paste0(pw, " Pathway: NOT DETECTED in Low_Risk"), cex.main = 1.2, col.main = "#1976D2")
    }
    
    if (in_high) {
      netVisual_aggregate(cellchat_high, signaling = pw, layout = "chord",
                          sources.use = caf_idx_high,
                          title.name = paste0(pw, " Pathway: High_Risk (CAF→)"))
    } else {
      plot.new()
      title(paste0(pw, " Pathway: NOT DETECTED in High_Risk"), cex.main = 1.2, col.main = "#D32F2F")
    }
    
    dev.off()
    cat(sprintf("    %s chord saved.\n", pw))
  }, error = function(e) {
    cat(sprintf("    %s chord SKIPPED: %s\n", pw, e$message))
    if (dev.cur() > 1) dev.off()
  })
}

# ============================================================
# PART 9: L-R Pair Full Comparison
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 9: Full L-R pair comparison\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Create unique keys
lr_high$key <- paste(lr_high$interaction_name_2, lr_high$target, sep = " → ")
lr_low$key <- paste(lr_low$interaction_name_2, lr_low$target, sep = " → ")

all_keys <- union(lr_high$key, lr_low$key)
match_h <- match(all_keys, lr_high$key)
match_l <- match(all_keys, lr_low$key)

lr_comparison <- data.frame(
  key = all_keys,
  prob_High = lr_high$prob[match_h],
  prob_Low = lr_low$prob[match_l],
  interaction = coalesce(lr_high$interaction_name_2[match_h], lr_low$interaction_name_2[match_l]),
  target = coalesce(lr_high$target[match_h], lr_low$target[match_l]),
  ligand = coalesce(lr_high$ligand[match_h], lr_low$ligand[match_l]),
  receptor = coalesce(lr_high$receptor[match_h], lr_low$receptor[match_l]),
  pathway_name = coalesce(lr_high$pathway_name[match_h], lr_low$pathway_name[match_l]),
  stringsAsFactors = FALSE
) %>%
  mutate(
    prob_High = replace_na(prob_High, 0),
    prob_Low = replace_na(prob_Low, 0),
    diff = prob_High - prob_Low,
    log2FC = log2((prob_High + 1e-8) / (prob_Low + 1e-8)),
    status = case_when(
      prob_High > 0 & prob_Low == 0 ~ "High_only",
      prob_High == 0 & prob_Low > 0 ~ "Low_only",
      diff > 0 ~ "Up_in_High",
      diff < 0 ~ "Up_in_Low",
      TRUE ~ "Equal"
    )
  ) %>%
  arrange(desc(diff))

write.csv(lr_comparison, file.path(output_dir, "Table_CAF_LR_full_comparison.csv"), row.names = FALSE)

cat(sprintf("  Total unique L-R pairs: %d\n", nrow(lr_comparison)))
cat(sprintf("    High_only: %d\n", sum(lr_comparison$status == "High_only")))
cat(sprintf("    Up_in_High: %d\n", sum(lr_comparison$status == "Up_in_High")))
cat(sprintf("    Up_in_Low: %d\n", sum(lr_comparison$status == "Up_in_Low")))
cat(sprintf("    Low_only: %d\n", sum(lr_comparison$status == "Low_only")))

# Top gained
lr_top_gained <- lr_comparison %>%
  filter(diff > 0) %>%
  head(30) %>%
  mutate(label = paste0(interaction, " → ", target),
         label = factor(label, levels = rev(label)))

p_gained <- ggplot(lr_top_gained, aes(x = label, y = diff, fill = pathway_name)) +
  geom_col(width = 0.7) + coord_flip() +
  theme_paper(base_size = 7) +
  labs(x = "", y = "Δ Probability (High - Low)",
       title = "CAF L-R Pairs: GAINED/INCREASED in High_Risk\n(Top 30)",
       fill = "Pathway") +
  theme(axis.text.y = element_text(size = 5),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.25, "cm")) +
  scale_fill_viridis_d(option = "turbo")

ggsave(file.path(output_dir, "Fig_LR_gained_HighRisk.pdf"), p_gained,
       width = 220, height = 170, units = "mm")

# DEG-driven
deg_ligands <- c("PTGIS", "FGF7", "CCL2", "VEGFA", "TNC", "CCN2", "VEGFC",
                 "GAS6", "NTN1", "THBS1", "THBS2", "LTBP1", "LTBP2",
                 "INHBA", "MDK", "SLIT2", "CSF1", "CXCL2", "PDGFC",
                 "A2M", "SERPINE1", "TIMP1")

lr_deg_driven <- lr_comparison %>% filter(ligand %in% deg_ligands)
write.csv(lr_deg_driven, file.path(output_dir, "Table_CAF_LR_DEG_driven.csv"), row.names = FALSE)

if (nrow(lr_deg_driven) > 0) {
  lr_deg_plot <- lr_deg_driven %>%
    filter(diff != 0) %>%
    arrange(desc(abs(diff))) %>%
    head(30) %>%
    mutate(label = paste0(interaction, " → ", target),
           label = factor(label, levels = label[order(diff)]),
           direction = ifelse(diff > 0, "↑ High_Risk", "↑ Low_Risk"))
  
  p_deg <- ggplot(lr_deg_plot, aes(x = label, y = diff, fill = direction)) +
    geom_col(width = 0.7) + coord_flip() +
    scale_fill_manual(values = c("↑ High_Risk" = "#D32F2F", "↑ Low_Risk" = "#1976D2")) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    theme_paper(base_size = 7) +
    labs(x = "", y = "Δ Communication Probability (High - Low)",
         title = "DEG-Driven CAF Ligand Interactions\n(Ligands upregulated in High_Risk CAFs)",
         fill = "") +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig_LR_DEG_driven.pdf"), p_deg,
         width = 190, height = 150, units = "mm")
}

cat("  L-R comparison figures saved.\n")

# ============================================================
# PART 10: Pathway Comparison
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 10: Pathway comparison\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

pw_high <- subsetCommunication(cellchat_high, sources.use = "CAF", slot.name = "netP")
pw_low <- subsetCommunication(cellchat_low, sources.use = "CAF", slot.name = "netP")

pw_h_sum <- pw_high %>% group_by(pathway_name) %>%
  summarise(prob_High = sum(prob), n_High = n_distinct(target), .groups = "drop")
pw_l_sum <- pw_low %>% group_by(pathway_name) %>%
  summarise(prob_Low = sum(prob), n_Low = n_distinct(target), .groups = "drop")

pw_comparison <- full_join(pw_h_sum, pw_l_sum, by = "pathway_name") %>%
  mutate(
    across(c(prob_High, prob_Low), ~replace_na(., 0)),
    across(c(n_High, n_Low), ~replace_na(., 0L)),
    diff = prob_High - prob_Low,
    status = case_when(
      prob_High > 0 & prob_Low == 0 ~ "High_only",
      prob_High == 0 & prob_Low > 0 ~ "Low_only",
      diff > 0 ~ "Up_in_High",
      diff < 0 ~ "Up_in_Low",
      TRUE ~ "Equal"
    )
  ) %>%
  arrange(desc(diff))

write.csv(pw_comparison, file.path(output_dir, "Table_CAF_pathway_comparison.csv"), row.names = FALSE)

# Pathway barplot
pw_plot_df <- pw_comparison %>%
  filter(prob_High > 0 | prob_Low > 0) %>%
  arrange(diff) %>%
  tail(35) %>%
  mutate(pathway_name = factor(pathway_name, levels = pathway_name))

p_pathway <- ggplot(pw_plot_df, aes(x = pathway_name, y = diff, fill = status)) +
  geom_col(width = 0.7) + coord_flip() +
  scale_fill_manual(values = c("Up_in_High" = "#D32F2F", "Up_in_Low" = "#1976D2",
                               "High_only" = "#FF6B6B", "Low_only" = "#64B5F6")) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  theme_paper(base_size = 7) +
  labs(x = "Signaling Pathway", y = "Δ Probability (High - Low)",
       title = "CAF Outgoing Pathways\n(High_Risk vs Low_Risk)", fill = "Status") +
  theme(axis.text.y = element_text(size = 6),
        legend.position = "bottom")

ggsave(file.path(output_dir, "Fig_Pathway_comparison.pdf"), p_pathway,
       width = 170, height = 200, units = "mm")

cat("  Pathway comparison saved.\n")

# ============================================================
# PART 11: Signaling Role & Centrality
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 11: Signaling role analysis\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cellchat_high <- netAnalysis_computeCentrality(cellchat_high)
cellchat_low <- netAnalysis_computeCentrality(cellchat_low)

# Outgoing patterns
tryCatch({
  pdf(file.path(output_dir, "Fig_SignalingRole_outgoing_High.pdf"), width = 12, height = 8)
  netAnalysis_signalingRole_heatmap(cellchat_high, pattern = "outgoing",
                                    title = "Outgoing Patterns: High_Risk", font.size = 6)
  dev.off()
}, error = function(e) { if (dev.cur() > 1) dev.off() })

tryCatch({
  pdf(file.path(output_dir, "Fig_SignalingRole_outgoing_Low.pdf"), width = 12, height = 8)
  netAnalysis_signalingRole_heatmap(cellchat_low, pattern = "outgoing",
                                    title = "Outgoing Patterns: Low_Risk", font.size = 6)
  dev.off()
}, error = function(e) { if (dev.cur() > 1) dev.off() })

# Incoming patterns
tryCatch({
  pdf(file.path(output_dir, "Fig_SignalingRole_incoming_High.pdf"), width = 12, height = 8)
  netAnalysis_signalingRole_heatmap(cellchat_high, pattern = "incoming",
                                    title = "Incoming Patterns: High_Risk", font.size = 6)
  dev.off()
}, error = function(e) { if (dev.cur() > 1) dev.off() })

tryCatch({
  pdf(file.path(output_dir, "Fig_SignalingRole_incoming_Low.pdf"), width = 12, height = 8)
  netAnalysis_signalingRole_heatmap(cellchat_low, pattern = "incoming",
                                    title = "Incoming Patterns: Low_Risk", font.size = 6)
  dev.off()
}, error = function(e) { if (dev.cur() > 1) dev.off() })

# Scatter: outgoing vs incoming
tryCatch({
  pdf(file.path(output_dir, "Fig_SignalingRole_scatter.pdf"), width = 14, height = 7)
  par(mfrow = c(1, 2))
  netAnalysis_signalingRole_scatter(cellchat_low, title = "Low_Risk", font.size = 10)
  netAnalysis_signalingRole_scatter(cellchat_high, title = "High_Risk", font.size = 10)
  dev.off()
}, error = function(e) { if (dev.cur() > 1) dev.off() })

cat("  Signaling role figures saved.\n")

# ============================================================
# PART 12: Incoming Signals to CAF
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 12: Incoming signals to CAF\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

lr_in_high <- subsetCommunication(cellchat_high, targets.use = "CAF")
lr_in_low <- subsetCommunication(cellchat_low, targets.use = "CAF")

cat(sprintf("  Incoming to CAF: High=%d, Low=%d\n", nrow(lr_in_high), nrow(lr_in_low)))

# Build comparison
lr_in_high$key <- paste(lr_in_high$interaction_name_2, lr_in_high$source, sep = " ← ")
lr_in_low$key <- paste(lr_in_low$interaction_name_2, lr_in_low$source, sep = " ← ")

all_keys_in <- union(lr_in_high$key, lr_in_low$key)
match_in_h <- match(all_keys_in, lr_in_high$key)
match_in_l <- match(all_keys_in, lr_in_low$key)

lr_in_comparison <- data.frame(
  key = all_keys_in,
  prob_High = lr_in_high$prob[match_in_h],
  prob_Low = lr_in_low$prob[match_in_l],
  interaction = coalesce(lr_in_high$interaction_name_2[match_in_h],
                         lr_in_low$interaction_name_2[match_in_l]),
  source = coalesce(lr_in_high$source[match_in_h],
                    lr_in_low$source[match_in_l]),
  pathway_name = coalesce(lr_in_high$pathway_name[match_in_h],
                          lr_in_low$pathway_name[match_in_l]),
  stringsAsFactors = FALSE
) %>%
  mutate(
    prob_High = replace_na(prob_High, 0),
    prob_Low = replace_na(prob_Low, 0),
    diff = prob_High - prob_Low,
    status = case_when(
      prob_High > 0 & prob_Low == 0 ~ "High_only",
      prob_High == 0 & prob_Low > 0 ~ "Low_only",
      diff > 0 ~ "Up_in_High",
      diff < 0 ~ "Up_in_Low",
      TRUE ~ "Equal"
    )
  ) %>%
  arrange(desc(diff))

write.csv(lr_in_comparison, file.path(output_dir, "Table_incoming_to_CAF_comparison.csv"),
          row.names = FALSE)

# Barplot of incoming signals
lr_in_top <- lr_in_comparison %>%
  arrange(desc(abs(diff))) %>%
  head(30) %>%
  mutate(
    label = paste0(source, ": ", interaction),
    label = factor(label, levels = label[order(diff)]),
    direction = ifelse(diff > 0, "↑ High_Risk", "↑ Low_Risk")
  )

p_incoming <- ggplot(lr_in_top, aes(x = label, y = diff, fill = direction)) +
  geom_col(width = 0.7) + coord_flip() +
  scale_fill_manual(values = c("↑ High_Risk" = "#D32F2F", "↑ Low_Risk" = "#1976D2")) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  theme_paper(base_size = 7) +
  labs(x = "", y = "Δ Probability (High - Low)",
       title = "Signals Incoming TO CAF\n(High_Risk vs Low_Risk)", fill = "") +
  theme(axis.text.y = element_text(size = 5))

ggsave(file.path(output_dir, "Fig_Incoming_to_CAF.pdf"), p_incoming,
       width = 190, height = 160, units = "mm")

cat("  Incoming signals figure saved.\n")

# ============================================================
# PART 13: DEG-CellChat Integration
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 13: DEG-CellChat integration\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Key ligands from DEG table
deg_ligand_info <- data.frame(
  ligand = c("PTGIS", "FGF7", "CCL2", "A2M", "TNC", "VEGFA", "CCN2",
             "VEGFC", "LTBP2", "GAS6", "NTN1", "THBS1", "LTBP1",
             "CXCL2", "MDK", "SLIT2", "INHBA", "CSF1", "PDGFC",
             "THBS2", "SERPINE1", "TIMP1"),
  avg_log2FC = c(3.49, 1.45, 1.38, 1.51, 1.28, 1.27, 1.20,
                 1.11, 1.12, 0.92, 0.82, 0.76, 0.71,
                 0.64, 0.60, 0.60, 0.60, 0.57, 0.42,
                 0.45, 0.86, 0.61),
  expected_pathway = c("Prostaglandin", "FGF", "CCL", "—", "TENASCIN", "VEGF", "CCN",
                       "VEGF", "TGFb", "GAS", "NETRIN", "THBS", "TGFb",
                       "CXCL", "MK", "SLIT", "ACTIVIN", "CSF", "PDGF",
                       "THBS", "—", "—"),
  primary_target = c("Immune/Vascular", "Epithelial", "Macrophage", "Broad",
                     "Integrin+", "Endothelial", "Fibroblast/Epithelial",
                     "Lymphatic", "Broad", "Macrophage", "Tumor/Immune",
                     "CD47+", "Broad", "Neutrophil", "Broad", "Endothelial/Tumor",
                     "Epithelial/Immune", "Macrophage", "Pericyte",
                     "CD47+", "—", "—"),
  stringsAsFactors = FALSE
)

# Check CellChat presence
deg_ligand_info$in_CellChat_High <- sapply(deg_ligand_info$ligand, function(lig) {
  any(lr_high$ligand == lig, na.rm = TRUE)
})

deg_ligand_info$in_CellChat_Low <- sapply(deg_ligand_info$ligand, function(lig) {
  any(lr_low$ligand == lig, na.rm = TRUE)
})

deg_ligand_info$n_targets_High <- sapply(deg_ligand_info$ligand, function(lig) {
  n_distinct(lr_high$target[lr_high$ligand == lig])
})

deg_ligand_info$n_targets_Low <- sapply(deg_ligand_info$ligand, function(lig) {
  n_distinct(lr_low$target[lr_low$ligand == lig])
})

deg_ligand_info$total_prob_High <- sapply(deg_ligand_info$ligand, function(lig) {
  sum(lr_high$prob[lr_high$ligand == lig], na.rm = TRUE)
})

deg_ligand_info$total_prob_Low <- sapply(deg_ligand_info$ligand, function(lig) {
  sum(lr_low$prob[lr_low$ligand == lig], na.rm = TRUE)
})

deg_ligand_info <- deg_ligand_info %>%
  mutate(
    prob_diff = total_prob_High - total_prob_Low,
    cellchat_status = case_when(
      in_CellChat_High & !in_CellChat_Low ~ "GAINED in High",
      !in_CellChat_High & in_CellChat_Low ~ "LOST in High",
      in_CellChat_High & in_CellChat_Low & prob_diff > 0 ~ "INCREASED in High",
      in_CellChat_High & in_CellChat_Low & prob_diff < 0 ~ "DECREASED in High",
      in_CellChat_High & in_CellChat_Low & prob_diff == 0 ~ "No change",
      TRUE ~ "Not detected"
    )
  ) %>%
  arrange(desc(prob_diff))

write.csv(deg_ligand_info, file.path(output_dir, "Table_DEG_CellChat_integration.csv"),
          row.names = FALSE)

cat("\n  DEG-CellChat Integration:\n")
print(deg_ligand_info %>%
        dplyr::select(ligand, avg_log2FC, n_targets_High, n_targets_Low,
                      total_prob_High, total_prob_Low, cellchat_status) %>%
        mutate(across(where(is.numeric), ~round(., 4))),
      row.names = FALSE)

# Fig: Integration scatter
p_integ <- ggplot(deg_ligand_info,
                  aes(x = avg_log2FC, y = prob_diff)) +
  geom_point(aes(size = n_targets_High, color = cellchat_status), alpha = 0.8) +
  geom_text_repel(aes(label = ligand), size = 2.5, max.overlaps = 25,
                  fontface = "italic") +
  scale_color_manual(values = c("GAINED in High" = "#FF0000",
                                "INCREASED in High" = "#D32F2F",
                                "DECREASED in High" = "#1976D2",
                                "LOST in High" = "#0000FF",
                                "No change" = "grey50",
                                "Not detected" = "grey80"),
                     name = "CellChat\nStatus") +
  scale_size_continuous(range = c(2, 8), name = "# Targets\n(High_Risk)") +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3) +
  theme_paper(base_size = 7) +
  labs(x = "Gene Expression log2FC (High vs Low Risk)",
       y = "Δ CellChat Probability (High - Low)",
       title = "CAF Ligands: Expression vs Communication Activity")

ggsave(file.path(output_dir, "Fig_DEG_CellChat_integration.pdf"), p_integ,
       width = 180, height = 140, units = "mm")

cat("  DEG-CellChat integration saved.\n")

# ============================================================
# PART 14: Final Summary & Save
# ============================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  PART 14: Final summary\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Overall metrics
total_out_high <- sum(cellchat_high@net$weight[caf_idx_high, ])
total_out_low <- sum(cellchat_low@net$weight[caf_idx_low, ])
total_in_high <- sum(cellchat_high@net$weight[, caf_idx_high])
total_in_low <- sum(cellchat_low@net$weight[, caf_idx_low])
n_out_high <- sum(cellchat_high@net$count[caf_idx_high, ])
n_out_low <- sum(cellchat_low@net$count[caf_idx_low, ])
n_in_high <- sum(cellchat_high@net$count[, caf_idx_high])
n_in_low <- sum(cellchat_low@net$count[, caf_idx_low])

# Save all results
cellchat_all_results <- list(
  cellchat_high = cellchat_high,
  cellchat_low = cellchat_low,
  cellchat_merged = cellchat_merged,
  lr_comparison = lr_comparison,
  lr_in_comparison = lr_in_comparison,
  pw_comparison = pw_comparison,
  lr_counts = lr_counts,
  deg_ligand_info = deg_ligand_info,
  metrics = list(
    total_out_high = total_out_high,
    total_out_low = total_out_low,
    total_in_high = total_in_high,
    total_in_low = total_in_low,
    n_out_high = n_out_high,
    n_out_low = n_out_low,
    n_in_high = n_in_high,
    n_in_low = n_in_low
  )
)

saveRDS(cellchat_all_results, file.path(output_dir, "cellchat_all_results.rds"))

# ─── Final Report ───
end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════════════╗\n")
cat("║          CELLCHAT ANALYSIS — CAF COMMUNICATION — COMPLETE                    ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                              ║\n")
cat("║  OVERALL METRICS:                                                            ║\n")
cat(sprintf("║    CAF Outgoing Strength: High=%.4f  Low=%.4f  (%.1fx)\n",
            total_out_high, total_out_low, total_out_high / max(total_out_low, 1e-8)))
cat(sprintf("║    CAF Outgoing Count:    High=%d  Low=%d\n", n_out_high, n_out_low))
cat(sprintf("║    CAF Incoming Strength: High=%.4f  Low=%.4f  (%.1fx)\n",
            total_in_high, total_in_low, total_in_high / max(total_in_low, 1e-8)))
cat(sprintf("║    CAF Incoming Count:    High=%d  Low=%d\n", n_in_high, n_in_low))
cat("║                                                                              ║\n")
cat("║  L-R PAIRS (CAF outgoing):                                                  ║\n")
cat(sprintf("║    Total: %d unique pairs\n", nrow(lr_comparison)))
cat(sprintf("║    High_only: %d | Up_High: %d | Up_Low: %d | Low_only: %d\n",
            sum(lr_comparison$status == "High_only"),
            sum(lr_comparison$status == "Up_in_High"),
            sum(lr_comparison$status == "Up_in_Low"),
            sum(lr_comparison$status == "Low_only")))
cat("║                                                                              ║\n")
cat("║  SIGNIFICANT L-R PAIRS PER TARGET:                                           ║\n")
for (i in 1:min(nrow(lr_counts), 18)) {
  cat(sprintf("║    %-25s High=%3d  Low=%3d  (Δ=%+d)\n",
              lr_counts$target[i], lr_counts$n_LR_High[i],
              lr_counts$n_LR_Low[i], lr_counts$diff_count[i]))
}
cat("║                                                                              ║\n")
cat("║  PATHWAYS (CAF outgoing):                                                    ║\n")
cat(sprintf("║    Total: %d pathways\n", nrow(pw_comparison)))
cat(sprintf("║    Increased in High: %d\n", sum(pw_comparison$status %in% c("Up_in_High", "High_only"))))
cat(sprintf("║    Decreased in High: %d\n", sum(pw_comparison$status %in% c("Up_in_Low", "Low_only"))))
cat("║                                                                              ║\n")
cat("║  OUTPUT FILES:                                                               ║\n")
cat("║                                                                              ║\n")
cat("║  Figures:                                                                    ║\n")
cat("║    Fig_LR_count_per_target.pdf         — L-R pair counts per target          ║\n")
cat("║    Fig_Heatmap_Ligand_Activity_HighVsLow.pdf — Ligand activity heatmap       ║\n")
cat("║    Fig_Heatmap_Ligand_Diff_HighMinusLow.pdf  — Difference heatmap            ║\n")
cat("║    Fig_Heatmap_LR_Detail_HighVsLow.pdf — Detailed L-R pair heatmap           ║\n")
cat("║    Fig_Circle_CAF_outgoing_High.pdf    — Circle plot (High, no overlap)      ║\n")
cat("║    Fig_Circle_CAF_outgoing_Low.pdf     — Circle plot (Low, no overlap)       ║\n")
cat("║    Fig_Circle_CAF_outgoing_comparison.pdf — Side-by-side circles             ║\n")
cat("║    Fig_Circle_CellChat_native_comparison.pdf — Native CellChat circles       ║\n")
cat("║    Fig_Circle_AllInteractions_comparison.pdf — All interactions               ║\n")
cat("║    Fig_Overall_interaction_comparison.pdf                                    ║\n")
cat("║    Fig_Diff_heatmap.pdf                                                      ║\n")
cat("║    Fig_Information_flow.pdf                                                  ║\n")
cat("║    Fig_Bubble_CAF_outgoing.pdf                                               ║\n")
cat("║    Fig_Bubble_CAF_incoming.pdf                                               ║\n")
cat("║    Fig_Bubble_CAF_to_immune.pdf                                              ║\n")
cat("║    Fig_Bubble_CAF_to_epithelial.pdf                                          ║\n")
cat("║    Fig_Bubble_CAF_to_stromal.pdf                                             ║\n")
cat("║    Fig_Chord_<pathway>_comparison.pdf  — Per-pathway chord diagrams          ║\n")
cat("║    Fig_LR_gained_HighRisk.pdf                                                ║\n")
cat("║    Fig_LR_DEG_driven.pdf                                                     ║\n")
cat("║    Fig_Pathway_comparison.pdf                                                ║\n")
cat("║    Fig_SignalingRole_*.pdf              — Signaling role heatmaps             ║\n")
cat("║    Fig_Incoming_to_CAF.pdf                                                   ║\n")
cat("║    Fig_DEG_CellChat_integration.pdf                                          ║\n")
cat("║                                                                              ║\n")
cat("║  Tables:                                                                     ║\n")
cat("║    Table_CAF_LR_counts_per_target.csv  — L-R counts per target               ║\n")
cat("║    Table_CAF_outgoing_HighRisk.csv                                           ║\n")
cat("║    Table_CAF_outgoing_LowRisk.csv                                            ║\n")
cat("║    Table_CAF_incoming_HighRisk.csv                                           ║\n")
cat("║    Table_CAF_incoming_LowRisk.csv                                            ║\n")
cat("║    Table_CAF_LR_full_comparison.csv                                          ║\n")
cat("║    Table_CAF_LR_DEG_driven.csv                                               ║\n")
cat("║    Table_CAF_pathway_comparison.csv                                          ║\n")
cat("║    Table_incoming_to_CAF_comparison.csv                                      ║\n")
cat("║    Table_DEG_CellChat_integration.csv                                        ║\n")
cat("║                                                                              ║\n")
cat("║  RDS:                                                                        ║\n")
cat("║    cellchat_High_Risk.rds                                                    ║\n")
cat("║    cellchat_Low_Risk.rds                                                     ║\n")
cat("║    cellchat_merged.rds                                                       ║\n")
cat("║    cellchat_all_results.rds                                                  ║\n")
cat("║                                                                              ║\n")
cat(sprintf("║  Runtime: %.1f minutes                                                      ║\n", runtime))
cat("║                                                                              ║\n")
cat("╚══════════════════════════════════════════════════════════════════════════════╝\n")

cat("\n━━━ CellChat Analysis Complete ━━━\n")
cat(sprintf("Output directory: %s\n", output_dir))