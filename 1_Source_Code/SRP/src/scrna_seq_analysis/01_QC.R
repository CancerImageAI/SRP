# ============================================================
# 01_QC.R — COMPLETE WORKING SCRIPT
# ============================================================

# CRITICAL: Set v3 assay mode BEFORE loading Seurat
options(Seurat.object.assay.version = "v3")

library(Seurat)
library(tidyverse)
library(patchwork)

# ----------------------------------------------------------
# Output directory
# ----------------------------------------------------------
output_dir <- "XXXyour path/scRNA_Results_new/01_QC"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# Theme (paper style)
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
# Sample info
# ----------------------------------------------------------
sample_info <- data.frame(
  sample_name = c("sc-01", "sc-02", "sc-03", "sc-04", "sc-05", "sc-06",
                  "sc-07", "sc-08", "sc-09", "sc-10", "sc-11", "sc-12"),
  group = c("Low_Risk", "High_Risk", "High_Risk", "High_Risk", "Low_Risk", "High_Risk",
            "Low_Risk", "Low_Risk", "High_Risk", "Low_Risk", "Low_Risk", "High_Risk"),
  stringsAsFactors = FALSE
)

base_path <- "XXXyour path/cellranger_output/1_cell_ranger_result"

# ==========================================================
# PART 1-4: LOAD, QC METRICS, VISUALIZE, FILTER
# ==========================================================
cat("━━━ PARTS 1-4: Load, QC, Filter ━━━\n")

seurat_list <- list()
for (i in 1:nrow(sample_info)) {
  s <- sample_info$sample_name[i]
  g <- sample_info$group[i]
  data_path <- file.path(base_path, s, "filtered_feature_bc_matrix")
  counts <- Read10X(data.dir = data_path)
  obj <- CreateSeuratObject(counts = counts, project = s,
                            min.cells = 3, min.features = 200)
  obj$sample <- s
  obj$group <- g
  seurat_list[[s]] <- obj
  cat("  ", s, ":", ncol(obj), "cells\n")
}

# Merge
merged <- merge(seurat_list[[1]], y = seurat_list[2:12],
                add.cell.ids = sample_info$sample_name,
                project = "EC_scRNA")
cat("Total:", ncol(merged), "cells\n")

# QC metrics
merged[["percent.mt"]] <- PercentageFeatureSet(merged, pattern = "^MT-")
merged[["percent.ribo"]] <- PercentageFeatureSet(merged, pattern = "^RP[SL]")
merged[["percent.hb"]] <- PercentageFeatureSet(merged, pattern = "^HB[^(P)]")

# Pre-filter QC violin
p1 <- VlnPlot(merged, features = "nFeature_RNA", group.by = "sample", pt.size = 0) +
  geom_hline(yintercept = 200, linetype = "dashed", color = "red", linewidth = 0.3) +
  theme_paper() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + NoLegend()
p2 <- VlnPlot(merged, features = "nCount_RNA", group.by = "sample", pt.size = 0) +
  geom_hline(yintercept = 1000, linetype = "dashed", color = "red", linewidth = 0.3) +
  theme_paper() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + NoLegend()
p3 <- VlnPlot(merged, features = "percent.mt", group.by = "sample", pt.size = 0) +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red", linewidth = 0.3) +
  theme_paper() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + NoLegend()
p4 <- VlnPlot(merged, features = "percent.hb", group.by = "sample", pt.size = 0) +
  geom_hline(yintercept = 5, linetype = "dashed", color = "red", linewidth = 0.3) +
  theme_paper() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + NoLegend()

ggsave(file.path(output_dir, "Fig1_QC_prefilter.pdf"),
       (p1 | p2) / (p3 | p4), width = 178, height = 130, units = "mm")

# Filter (paper thresholds: >200 genes, >1000 UMIs, <10% mt, <5% hb)
pre_n <- table(merged$sample)
merged_filt <- subset(merged, nFeature_RNA > 200 & 
                        nFeature_RNA < 5000 &
                        nCount_RNA > 1000 &
                        nCount_RNA < 15000 &
                        percent.mt < 10 & 
                        percent.hb < 5)
post_n <- table(merged_filt$sample)

filter_df <- data.frame(
  sample = names(pre_n), before = as.numeric(pre_n),
  after = as.numeric(post_n[names(pre_n)])
) %>% mutate(removed = before - after, pct_retained = round(after/before*100, 1))

write.csv(filter_df, file.path(output_dir, "Table_QC_filter.csv"), row.names = FALSE)
cat("After QC:", ncol(merged_filt), "cells\n")

# Save filtered
saveRDS(merged_filt, file.path(output_dir, "merged_filtered.rds"))
rm(merged, seurat_list); gc()


# ==========================================================
# PART 5: DOUBLET DETECTION using scDblFinder
# Compatible with Seurat v5
# ==========================================================
cat("\n━━━ PART 5: scDblFinder ━━━\n")

# Install if needed (run once)
# BiocManager::install("scDblFinder")

library(scDblFinder)
library(SingleCellExperiment)

# Load filtered data
merged_filt <- readRDS(file.path(output_dir, "merged_filtered.rds"))

# Convert to SingleCellExperiment
sce <- as.SingleCellExperiment(merged_filt)

# Run scDblFinder per sample
set.seed(42)
sce <- scDblFinder(sce, samples = "sample")

# Transfer results back to Seurat object
merged_filt$doublet_classification <- ifelse(
  sce$scDblFinder.class == "singlet", "Singlet", "Doublet"
)
merged_filt$doublet_score <- sce$scDblFinder.score

# Summary table
doublet_summary <- merged_filt@meta.data %>%
  group_by(sample, doublet_classification) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = doublet_classification, values_from = n, values_fill = 0) %>%
  mutate(total = Singlet + Doublet,
         doublet_pct = round(Doublet / total * 100, 1))

write.csv(doublet_summary, file.path(output_dir, "Table_scDblFinder_summary.csv"), row.names = FALSE)
cat("\nDoublet summary:\n")
print(as.data.frame(doublet_summary))

# Visualize doublet scores
p_score <- VlnPlot(merged_filt, features = "doublet_score", 
                   group.by = "sample", pt.size = 0) +
  theme_paper() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  NoLegend() + ggtitle("Doublet Scores by Sample")

ggsave(file.path(output_dir, "Fig2_scDblFinder_scores.pdf"), p_score, 
       width = 178, height = 80, units = "mm")

# Remove doublets
merged_clean <- subset(merged_filt, doublet_classification == "Singlet")
cat("\nFinal cell count after doublet removal:", ncol(merged_clean), "\n")

# Final QC summary
final_summary <- merged_clean@meta.data %>%
  group_by(sample, group) %>%
  summarise(n_cells = n(),
            median_genes = median(nFeature_RNA),
            median_UMI = median(nCount_RNA),
            median_mt = round(median(percent.mt), 2),
            .groups = "drop")

write.csv(final_summary, file.path(output_dir, "Table_Final_QC_summary.csv"), row.names = FALSE)

# Save clean object
saveRDS(merged_clean, file.path(output_dir, "merged_clean_singlets.rds"))

cat("\n━━━ 01_QC COMPLETE ━━━\n")
cat("Final dataset:", ncol(merged_clean), "cells x", nrow(merged_clean), "genes\n")
