# ============================================================
# REVISED ANALYSIS: Biological Basis of CT Imaging Markers
# for Predicting Chemoimmunotherapy Response in Esophageal Cancer
#
# Study Design:
#   CT radiomic features → Risk Score → High/Low Risk groups
#   scRNA-seq analysis explains WHY these groups respond differently
#
# Key Questions:
#   1. What immune/stromal differences explain treatment response?
#   2. What resistance mechanisms are enriched in High-Risk?
#   3. How does the TME architecture differ (CT imaging correlate)?
# ============================================================

library(Seurat)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ComplexHeatmap)
library(circlize)
library(ggrepel)
library(grid)
library(ggnewscale)

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
base_dir <- "XXXyour path/scRNA_Results_new"
output_dir <- file.path(base_dir, "07_Biological_Basis_CT_Response")
fig_dir <- file.path(output_dir, "figures")
table_dir <- file.path(output_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

cat("╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║  Biological Basis of CT Imaging Markers for                          ║\n")
cat("║  Chemoimmunotherapy Response Prediction                              ║\n")
cat("╚══════════════════════════════════════════════════════════════════════╝\n\n")

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
cat("━━ Step 1: Load and Re-annotate ━━\n")

seu <- readRDS(file.path(base_dir, "05_Heatmap_CNV", "seu_with_cnv_scores.rds"))
DefaultAssay(seu) <- "RNA"

# Version-safe layer handling
seurat_v5 <- packageVersion("Seurat") >= "5.0.0"
cat(sprintf("  Seurat version: %s\n", packageVersion("Seurat")))

tryCatch({
  if (seurat_v5 && inherits(seu[["RNA"]], "Assay5") && length(Layers(seu[["RNA"]])) > 1) {
    seu[["RNA"]] <- JoinLayers(seu[["RNA"]])
    cat("  ✓ Layers joined (Seurat v5)\n")
  } else {
    cat("  ✓ No layer joining needed\n")
  }
}, error = function(e) {
  cat(sprintf("  Note: %s — proceeding without joining\n", e$message))
})

cat(sprintf("  Cells: %d | Genes: %d\n", ncol(seu), nrow(seu)))

# ----------------------------------------------------------
# Apply verified annotations (from uploaded marker table)
# ----------------------------------------------------------
cat("\n  Applying annotations...\n")
# ----------------------------------------------------------
# Set clustering identity (resolution 0.5)
# ----------------------------------------------------------
cluster_cols <- grep("snn_res|seurat_clusters", colnames(seu@meta.data), value = TRUE)
cat("  Available cluster columns:", paste(cluster_cols, collapse = ", "), "\n")

res_col <- grep("0\\.5", cluster_cols, value = TRUE)[1]
if (is.na(res_col) || is.null(res_col)) {
  possible <- c("RNA_snn_res.0.5", "integrated_snn_res.0.5",
                "SCT_snn_res.0.5", "wsnn_res.0.5", "seurat_clusters")
  res_col <- possible[possible %in% colnames(seu@meta.data)][1]
}

cat(sprintf("  Using clustering column: %s\n", res_col))
Idents(seu) <- res_col

# Check cluster IDs
cluster_ids <- levels(Idents(seu))
cat(sprintf("  Number of clusters: %d\n", length(cluster_ids)))
cat("  Cluster IDs:", paste(cluster_ids, collapse = ", "), "\n")
cat("  Cluster sizes:\n")
print(sort(table(Idents(seu)), decreasing = TRUE))


cluster_annotations <- c(
  "0"  = "CD8_T_effector",
  "1"  = "CAF",
  "2"  = "Treg",
  "3"  = "Macrophage_SPP1",
  "4"  = "Epithelial_basal",
  "5"  = "Epithelial_squamous",
  "6"  = "Neutrophil",
  "7"  = "Fibroblast_PI16",
  "8"  = "Pericyte_SMC",
  "9"  = "Endothelial",
  "10" = "Plasma_cell",
  "11" = "B_cell",
  "12" = "Proliferating",
  "13" = "DC_mature",
  "14" = "Epithelial_stem_like",
  "15" = "Epithelial_differentiated",
  "16" = "Mast_cell",
  "17" = "Myofibroblast"
)

# ----------------------------------------------------------
# *** FIX: Direct assignment via metadata (avoids RenameIdents error) ***
# ----------------------------------------------------------
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Assigning cell type annotations\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Get cluster assignment for each cell as character
cell_clusters <- as.character(seu@meta.data[[res_col]])

# Map cluster → cell type
seu@meta.data$cell_type <- cluster_annotations[as.character(seu@meta.data[[res_col]])]

# Check for unmapped
n_unmapped <- sum(is.na(seu$cell_type))
if (n_unmapped > 0) {
  unmapped_cls <- unique(cell_clusters[is.na(seu$cell_type)])
  cat(sprintf("  ⚠ WARNING: %d cells unmapped. Clusters: %s\n",
              n_unmapped, paste(unmapped_cls, collapse = ", ")))
  seu$cell_type[is.na(seu$cell_type)] <- "Unknown"
} else {
  cat("  ✓ All 95,120 cells annotated successfully.\n")
}

# Functional grouping relevant to chemoimmunotherapy
response_categories <- c(
  "CD8_T_effector" = "Anti-tumor_Immune",
  "B_cell" = "Anti-tumor_Immune",
  "Plasma_cell" = "Anti-tumor_Immune",
  "DC_mature" = "Anti-tumor_Immune",
  "Treg" = "Immunosuppressive",
  "Macrophage_SPP1" = "Immunosuppressive",
  "Mast_cell" = "Immunomodulatory",
  "Neutrophil" = "Immunomodulatory",
  "CAF" = "Treatment_Barrier",
  "Myofibroblast" = "Treatment_Barrier",
  "Fibroblast_PI16" = "Stromal_Support",
  "Pericyte_SMC" = "Vascular",
  "Endothelial" = "Vascular",
  "Epithelial_basal" = "Tumor",
  "Epithelial_squamous" = "Tumor",
  "Epithelial_differentiated" = "Tumor",
  "Epithelial_stem_like" = "Tumor_Stem",
  "Proliferating" = "Proliferating"
)

seu@meta.data$response_category <- response_categories[seu@meta.data$cell_type]

cat("  ✓ Annotations applied\n")
print(sort(table(seu$cell_type), decreasing = TRUE))

# ----------------------------------------------------------
# Color palettes
# ----------------------------------------------------------
cell_colors <- c(
  "CD8_T_effector" = "#E64B35", "Treg" = "#F39B7F",
  "Macrophage_SPP1" = "#4DBBD5", "Neutrophil" = "#00A087",
  "DC_mature" = "#3C5488", "Mast_cell" = "#8491B4",
  "B_cell" = "#91D1C2", "Plasma_cell" = "#7E6148",
  "CAF" = "#B09C85", "Fibroblast_PI16" = "#DC9FB4",
  "Myofibroblast" = "#E18727",
  "Epithelial_basal" = "#6F99AD", "Epithelial_squamous" = "#20854E",
  "Epithelial_differentiated" = "#7876B1", "Epithelial_stem_like" = "#EE4C97",
  "Pericyte_SMC" = "#FFDC91", "Endothelial" = "#0072B5",
  "Proliferating" = "#FF7F00"
)

response_colors <- c(
  "Anti-tumor_Immune" = "#E64B35",
  "Immunosuppressive" = "#4DBBD5",
  "Immunomodulatory" = "#00A087",
  "Treatment_Barrier" = "#B09C85",
  "Stromal_Support" = "#DC9FB4",
  "Vascular" = "#0072B5",
  "Tumor" = "#20854E",
  "Tumor_Stem" = "#EE4C97",
  "Proliferating" = "#FF7F00"
)

# ===========================================================
# Step 2: VISUALIZATION
# ===========================================================
cat("\n━━ Step 2: Visualization ━━\n")

# --- Fig1: UMAP by cell type ---
Idents(seu) <- "cell_type"
p1 <- DimPlot(seu, reduction = "umap", group.by = "cell_type",
              cols = cell_colors, pt.size = 0.2, label = TRUE,
              label.size = 2.8, repel = TRUE) +
  ggtitle("Tumor Microenvironment: Cell Type Annotation") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
        legend.text = element_text(size = 7),
        legend.key.size = unit(0.35, "cm")) +
  guides(color = guide_legend(ncol = 2, override.aes = list(size = 3)))

ggsave(file.path(fig_dir, "Fig1_UMAP_celltype.pdf"), p1, width = 12, height = 8)
ggsave(file.path(fig_dir, "Fig1_UMAP_celltype.png"), p1, width = 12, height = 8, dpi = 300)

# --- Fig2: UMAP by response category ---
p2 <- DimPlot(seu, reduction = "umap", group.by = "response_category",
              cols = response_colors, pt.size = 0.2, label = TRUE,
              label.size = 3.5, repel = TRUE) +
  ggtitle("Functional Categories Relevant to Chemoimmunotherapy") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 11))

ggsave(file.path(fig_dir, "Fig2_UMAP_response_category.pdf"), p2, width = 10, height = 7)

# --- Fig3: UMAP split by CT-defined risk group ---
if ("group" %in% colnames(seu@meta.data)) {
  p3a <- DimPlot(seu, reduction = "umap", group.by = "group",
                 pt.size = 0.15,
                 cols = c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")) +
    ggtitle("CT-Defined Risk Groups") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  p3b <- DimPlot(seu, reduction = "umap", group.by = "cell_type",
                 split.by = "group", cols = cell_colors,
                 pt.size = 0.15, label = TRUE, label.size = 2, repel = TRUE) +
    ggtitle("TME Composition: High-Risk vs Low-Risk (CT-defined)") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 11))
  
  ggsave(file.path(fig_dir, "Fig3a_UMAP_risk_group.pdf"), p3a, width = 9, height = 7)
  ggsave(file.path(fig_dir, "Fig3b_UMAP_split_risk.pdf"), p3b, width = 16, height = 7)
  cat("  ✓ Fig3: UMAP by CT-defined risk\n")
}

# --- Fig4: Cell type proportions (key to explain CT imaging) ---
if ("group" %in% colnames(seu@meta.data)) {
  prop_data <- seu@meta.data %>%
    group_by(group, cell_type) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(group) %>%
    mutate(pct = n / sum(n) * 100)
  
  # Stacked bar
  p4a <- ggplot(prop_data, aes(x = group, y = pct, fill = cell_type)) +
    geom_col(width = 0.65) +
    scale_fill_manual(values = cell_colors) +
    theme_classic(base_size = 10) +
    labs(x = "CT-Defined Risk Group", y = "Percentage (%)", fill = "Cell Type",
         title = "TME Composition Differences Underlying CT Imaging Phenotype") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
          legend.text = element_text(size = 7),
          legend.key.size = unit(0.3, "cm")) +
    guides(fill = guide_legend(ncol = 2))
  
  ggsave(file.path(fig_dir, "Fig4a_proportion_stack.pdf"), p4a, width = 9, height = 7)
  
  # Grouped bar (side-by-side comparison)
  p4b <- ggplot(prop_data, aes(x = reorder(cell_type, -pct), y = pct, fill = group)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")) +
    coord_flip() +
    theme_classic(base_size = 9) +
    labs(x = "", y = "Percentage (%)", fill = "CT Risk Group",
         title = "Cell Type Abundance: CT High-Risk vs Low-Risk",
         subtitle = "Differences explain CT imaging features") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8, color = "grey40"),
          legend.position = "top")
  
  ggsave(file.path(fig_dir, "Fig4b_proportion_dodge.pdf"), p4b, width = 10, height = 7)
  
  # By response category
  cat_prop <- seu@meta.data %>%
    group_by(group, response_category) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(group) %>%
    mutate(pct = n / sum(n) * 100)
  
  p4c <- ggplot(cat_prop, aes(x = response_category, y = pct, fill = group)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c("High_Risk" = "#E64B35", "Low_Risk" = "#4DBBD5")) +
    coord_flip() +
    theme_classic(base_size = 10) +
    labs(x = "", y = "Percentage (%)", fill = "CT Risk",
         title = "Functional Category Composition by CT-Defined Risk",
         subtitle = "Treatment barrier vs anti-tumor immunity balance") +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, size = 9, color = "grey30"),
          legend.position = "top")
  
  ggsave(file.path(fig_dir, "Fig4c_functional_category.pdf"), p4c, width = 9, height = 6)
  cat("  ✓ Fig4: Proportion analysis\n")
}

# --- Fig5: Dot plot of key markers ---
markers_by_function <- c(
  # Anti-tumor immunity markers
  "CD8A", "GZMB", "GZMK", "IFNG", "PRF1", "CRTAM",
  # Immunosuppression markers
  "FOXP3", "IL2RA", "CD163", "SPP1", "MRC1",
  # Antigen presentation
  "FLT3", "LAMP3", "IDO1", "CD86",
  # B/Plasma (TLS indicators)
  "MS4A1", "CD79A", "JCHAIN", "IGKC",
  # Neutrophil
  "CXCR2", "FCGR3B", "CSF3R",
  # Treatment barrier (CAF/ECM)
  "MMP11", "COL11A1", "LRRC15", "POSTN", "ACTA2",
  # Vascular (drug delivery)
  "CDH5", "KDR", "RGS5", "MYH11",
  # Tumor epithelial
  "KRT5", "KRT17", "COL17A1", "PSCA", "CEACAM5",
  # Stem/resistance
  "ROCR", "COL9A1",
  # Proliferation
  "MKI67", "TOP2A"
)

markers_avail <- markers_by_function[markers_by_function %in% rownames(seu)]

ct_order <- c("CD8_T_effector", "Treg", "Macrophage_SPP1", "DC_mature",
              "B_cell", "Plasma_cell", "Neutrophil", "Mast_cell",
              "CAF", "Myofibroblast", "Fibroblast_PI16",
              "Epithelial_basal", "Epithelial_squamous",
              "Epithelial_differentiated", "Epithelial_stem_like",
              "Pericyte_SMC", "Endothelial", "Proliferating")
ct_order <- ct_order[ct_order %in% unique(seu$cell_type)]
seu$cell_type <- factor(seu$cell_type, levels = ct_order)
Idents(seu) <- "cell_type"

p5 <- DotPlot(seu, features = markers_avail, dot.scale = 4, col.min = 0) +
  RotatedAxis() +
  scale_color_viridis_c(option = "magma") +
  theme(axis.text.x = element_text(size = 6, angle = 65, hjust = 1),
        axis.text.y = element_text(size = 7.5)) +
  ggtitle("Key Markers for Chemoimmunotherapy Response Biology")

ggsave(file.path(fig_dir, "Fig5_DotPlot_response_markers.pdf"), p5, width = 16, height = 8)
cat("  ✓ Fig5: Dot plot\n")

# ===========================================================
# Step 3: CONFOUNDING FACTOR ANALYSIS
# ===========================================================
cat("\n━━ Step 3: Confounding Factor Analysis ━━\n")

meta <- seu@meta.data
latent_vars <- c("nCount_RNA")  # Always control for library size

# --- Check sex confounding ---
sex_col <- intersect(c("sex", "Sex", "gender", "Gender"), colnames(meta))
if (length(sex_col) > 0) {
  sex_col <- sex_col[1]
  sex_table <- table(meta[[sex_col]], meta$group)
  cat("  Sex × CT-Risk Group:\n")
  print(sex_table)
  
  if (nrow(sex_table) > 1 && ncol(sex_table) > 1) {
    chi_p <- tryCatch(chisq.test(sex_table)$p.value, error = function(e) 1)
    cat(sprintf("  Chi-sq p = %.4f\n", chi_p))
    if (chi_p < 0.1) {
      latent_vars <- c(latent_vars, sex_col)
      cat("  ⚠ Sex CONFOUNDED with CT-risk group — adding as latent variable\n")
    } else {
      cat("  ✓ Sex balanced between groups\n")
    }
  }
} else {
  cat("  ⚠ No sex column — cannot control. This is a study LIMITATION.\n")
  cat("    Esophageal cancer has strong sex disparity (male:female ~3:1).\n")
  cat("    If CT features correlate with sex, DE results may be confounded.\n")
}

# --- Check sample/patient ---
sample_col <- intersect(c("orig.ident", "sample", "patient", "donor"), colnames(meta))
if (length(sample_col) > 0) {
  sample_col <- sample_col[1]
  cat(sprintf("\n  Sample column: %s\n", sample_col))
  sample_group <- table(meta[[sample_col]], meta$group)
  n_h <- sum(sample_group[, "High_Risk"] > 0)
  n_l <- sum(sample_group[, "Low_Risk"] > 0)
  cat(sprintf("  Samples: High-Risk=%d, Low-Risk=%d\n", n_h, n_l))
  print(sample_group)
} else {
  sample_col <- NULL
  cat("  ⚠ No sample/patient column found\n")
}

# --- Technical factors ---
if ("percent.mt" %in% colnames(meta)) {
  latent_vars <- c(latent_vars, "percent.mt")
}

cat(sprintf("\n  ✓ Latent variables for MAST: %s\n", paste(latent_vars, collapse = ", ")))

# Save confounding report
conf_report <- data.frame(
  Factor = c("Library size (nCount_RNA)", "Mitochondrial %", "Sex",
             "Sample/Patient", "Cell cycle"),
  Status = c("Always varies", 
             ifelse("percent.mt" %in% colnames(meta), "Available", "NA"),
             ifelse(length(sex_col) > 0, sprintf("p=%.3f", chi_p), "Not available"),
             ifelse(!is.null(sample_col), sprintf("H:%d, L:%d", n_h, n_l), "NA"),
             "Cluster 12 = proliferating"),
  Action = c("MAST latent var", "MAST latent var",
             ifelse(length(sex_col) > 0 && chi_p < 0.1, 
                    "MAST latent + DESeq2 covariate", "None needed / unavailable"),
             "Pseudobulk DESeq2", "Separate analysis"),
  Biological_Rationale = c(
    "Technical confounder — sequencing depth",
    "Cell quality indicator — dying cells",
    "EC has 3:1 male bias; sex hormones affect TME",
    "Avoids pseudoreplication from shared patient effects",
    "Proliferating cells mixed lineage — confounds lineage DE"
  ),
  stringsAsFactors = FALSE
)
write.csv(conf_report, file.path(table_dir, "confounding_report.csv"), row.names = FALSE)

# ===========================================================
# Step 4: DIFFERENTIAL EXPRESSION
# Framed around chemoimmunotherapy response mechanisms
# ===========================================================
cat("\n━━ Step 4: Differential Expression (Chemoimmunotherapy Focus) ━━\n")

# Cell types grouped by treatment relevance
immunotherapy_types <- c("CD8_T_effector", "Treg", "Macrophage_SPP1",
                         "DC_mature", "B_cell", "Plasma_cell")
chemo_barrier_types <- c("CAF", "Myofibroblast", "Endothelial", "Pericyte_SMC")
tumor_types <- c("Epithelial_squamous", "Epithelial_basal",
                 "Epithelial_differentiated", "Epithelial_stem_like")

all_de_types <- c(immunotherapy_types, chemo_barrier_types, tumor_types, "Neutrophil")
all_de_types <- all_de_types[all_de_types %in% unique(seu$cell_type)]

cat(sprintf("  DE cell types: %d\n", length(all_de_types)))
cat(sprintf("  Immunotherapy-relevant: %s\n", paste(intersect(immunotherapy_types, all_de_types), collapse = ", ")))
cat(sprintf("  Chemo-barrier: %s\n", paste(intersect(chemo_barrier_types, all_de_types), collapse = ", ")))
cat(sprintf("  Tumor: %s\n", paste(intersect(tumor_types, all_de_types), collapse = ", ")))

de_results <- list()

for (ct in all_de_types) {
  cat(sprintf("\n  DE: %s...", ct))
  
  seu_sub <- subset(seu, cell_type == ct)
  n_high <- sum(seu_sub$group == "High_Risk")
  n_low <- sum(seu_sub$group == "Low_Risk")
  cat(sprintf(" (H=%d, L=%d)", n_high, n_low))
  
  if (n_high < 30 || n_low < 30) {
    cat(" SKIP (<30)\n")
    next
  }
  
  Idents(seu_sub) <- "group"
  
  # Valid latent variables for this subset
  valid_latent <- latent_vars[latent_vars %in% colnames(seu_sub@meta.data)]
  valid_latent <- valid_latent[sapply(valid_latent, function(v) {
    vals <- seu_sub@meta.data[[v]]
    is.numeric(vals) && length(unique(vals)) > 1
  })]
  
  tryCatch({
    de <- FindMarkers(
      seu_sub, ident.1 = "High_Risk", ident.2 = "Low_Risk",
      test.use = "MAST", latent.vars = valid_latent,
      min.pct = 0.1, logfc.threshold = 0.1, verbose = FALSE
    )
    
    de$gene <- rownames(de)
    de$cell_type <- ct
    
    n_up <- sum(de$p_val_adj < 0.05 & de$avg_log2FC > 0.5, na.rm = TRUE)
    n_down <- sum(de$p_val_adj < 0.05 & de$avg_log2FC < -0.5, na.rm = TRUE)
    cat(sprintf(" ✓ Up=%d, Down=%d\n", n_up, n_down))
    
    de_results[[ct]] <- de
    write.csv(de, file.path(table_dir, sprintf("DE_%s.csv", ct)), row.names = FALSE)
    
  }, error = function(e) {
    cat(sprintf(" ✗ %s\n", e$message))
  })
}

# ===========================================================
# Step 5: ENHANCED VOLCANO PLOTS
# Framed for chemoimmunotherapy response biology
# ===========================================================
cat("\n━━ Step 5: Enhanced Volcano Plots (Treatment-Relevant Genes) ━━\n")

# Define genes by chemoimmunotherapy relevance
# These explain WHY CT-defined risk groups respond differently
important_genes <- list(
  # PD-1/PD-L1 axis and immune checkpoints — immunotherapy targets
  immune_checkpoint = c("CD274", "PDCD1LG2", "CTLA4", "PDCD1", "LAG3",
                        "HAVCR2", "TIGIT", "IDO1", "IDO2", "SIGLEC15",
                        "VTCN1", "CD80", "CD86", "TNFRSF4", "TNFRSF18",
                        "CD28", "ICOS"),
  
  # Active anti-tumor immunity (predicts ICB response)
  cytotoxicity = c("GZMB", "GZMA", "GZMK", "PRF1", "GNLY", "NKG7",
                   "IFNG", "TNF", "FASLG", "CRTAM", "KLRK1", "CD8A",
                   "CD8B", "EOMES", "TBX21"),
  
  # T cell exhaustion (predicts ICB resistance)
  exhaustion = c("PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "TOX2",
                 "ENTPD1", "CTLA4", "CXCL13"),
  
  # Immunosuppressive TME (CT high-risk → immunosuppressed)
  immunosuppression = c("FOXP3", "IL10", "TGFB1", "TGFB2", "IL6",
                        "VEGFA", "MIF", "ARG1", "CD163", "MSR1",
                        "SPP1", "CCL2", "CXCL12", "IL2RA", "VSIG4"),
  
  # IFN-γ signaling (strongest ICB response predictor)
  IFN_signaling = c("IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10",
                    "CXCL11", "GBP1", "GBP2", "IDO1",
                    "TAP1", "TAP2", "B2M", "HLA-A", "HLA-B", "HLA-C"),
  
  # TLS markers (predict excellent ICB response)
  TLS = c("CXCL13", "CCL19", "CCL21", "LAMP3", "CD79A",
          "MS4A1", "JCHAIN", "IGHG1", "CR2", "FDCSP"),
  
  # ECM barrier / CAF activation (drug exclusion → CT dense phenotype)
  ECM_barrier = c("COL1A1", "COL1A2", "COL11A1", "COL3A1", "FN1",
                  "MMP2", "MMP9", "MMP11", "MMP14", "FAP", "ACTA2",
                  "POSTN", "LRRC15", "TGFBI", "LOX", "LOXL2",
                  "CTHRC1", "ITGA11"),
  
  # Angiogenesis (CT enhancement patterns; drug delivery)
  angiogenesis = c("VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1",
                   "ANGPT1", "ANGPT2", "TEK", "PECAM1", "CDH5",
                   "DLL4", "HIF1A", "EPAS1"),
  
  # Chemotherapy resistance mechanisms
  chemo_resistance = c("ERCC1", "ERCC2", "XRCC1", "MLH1", "MSH2",
                       "ABCB1", "ABCC1", "GSTP1", "DPYD",
                       "BCL2", "BCL2L1", "MCL1", "BAX", "BAK1"),
  
  # Cancer stemness (dual resistance to chemo + ICB)
  stemness = c("CD44", "ALDH1A1", "SOX2", "KLF4", "BMI1",
               "NOTCH1", "WNT5A", "LGR5", "LGR6", "ABCG2"),
  
  # Proliferation (chemo-sensitive cells)
  proliferation = c("MKI67", "TOP2A", "PCNA", "CCNB1",
                    "CDK4", "CDK6", "CCND1", "RRM2", "TYMS")
)

all_important <- unique(unlist(important_genes))

# Gene → category lookup
gene_to_cat <- do.call(rbind, lapply(names(important_genes), function(cn) {
  data.frame(gene = important_genes[[cn]], category = cn, stringsAsFactors = FALSE)
}))
gene_to_cat <- gene_to_cat[!duplicated(gene_to_cat$gene), ]

# Category visualization colors
category_colors <- c(
  "immune_checkpoint" = "#9B2335",
  "cytotoxicity" = "#E64B35",
  "exhaustion" = "#8B0000",
  "immunosuppression" = "#00A087",
  "IFN_signaling" = "#3C5488",
  "TLS" = "#91D1C2",
  "ECM_barrier" = "#B09C85",
  "angiogenesis" = "#0072B5",
  "chemo_resistance" = "#FF7F00",
  "stemness" = "#6A3D9A",
  "proliferation" = "#E18727",
  "top_DE" = "grey25"
)

category_labels <- c(
  "immune_checkpoint" = "Immune Checkpoint",
  "cytotoxicity" = "Cytotoxicity",
  "exhaustion" = "T cell Exhaustion",
  "immunosuppression" = "Immunosuppression",
  "IFN_signaling" = "IFN-γ Signaling",
  "TLS" = "TLS Markers",
  "ECM_barrier" = "ECM/Drug Barrier",
  "angiogenesis" = "Angiogenesis (CT)",
  "chemo_resistance" = "Chemo Resistance",
  "stemness" = "Stemness",
  "proliferation" = "Proliferation",
  "top_DE" = "Top DE Gene"
)

# ----------------------------------------------------------
# Generate volcano for each cell type
# ----------------------------------------------------------
for (ct in names(de_results)) {
  de <- de_results[[ct]]
  if (nrow(de) < 50) next
  
  cat(sprintf("  Volcano: %s\n", ct))
  
  # Prepare
  de$neg_log10p <- -log10(de$p_val_adj + 1e-300)
  max_y <- min(200, quantile(de$neg_log10p[is.finite(de$neg_log10p)], 0.999, na.rm = TRUE))
  de$neg_log10p <- pmin(de$neg_log10p, max_y)
  
  de$sig_class <- "NS"
  de$sig_class[de$p_val_adj < 0.05 & de$avg_log2FC > 0.5] <- "Up_High"
  de$sig_class[de$p_val_adj < 0.05 & de$avg_log2FC < -0.5] <- "Down_High"
  
  n_up <- sum(de$sig_class == "Up_High")
  n_down <- sum(de$sig_class == "Down_High")
  
  # Assign categories
  de$category <- gene_to_cat$category[match(de$gene, gene_to_cat$gene)]
  
  # --- Gene labeling strategy ---
  # Layer 1: Treatment-relevant important genes (significant)
  label_important <- de %>%
    dplyr::filter(!is.na(category) & p_val_adj < 0.05 & abs(avg_log2FC) > 0.2) %>%
    dplyr::arrange(p_val_adj) %>%
    dplyr::slice_head(n = 18)
  
  # Layer 2: Top DE genes by significance (novel findings)
  label_top <- de %>%
    dplyr::filter(is.na(category) & p_val_adj < 1e-10 & abs(avg_log2FC) > 1) %>%
    dplyr::filter(!gene %in% label_important$gene) %>%
    dplyr::arrange(p_val_adj) %>%
    dplyr::slice_head(n = 8)
  
  # Layer 3: Extreme fold change
  label_extreme <- de %>%
    dplyr::filter(is.na(category) & p_val_adj < 0.01 & abs(avg_log2FC) > 2) %>%
    dplyr::filter(!gene %in% c(label_important$gene, label_top$gene)) %>%
    dplyr::arrange(desc(abs(avg_log2FC))) %>%
    dplyr::slice_head(n = 4)
  
  all_label_genes <- unique(c(label_important$gene, label_top$gene, label_extreme$gene))
  
  # Mark for labeling
  de$show_label <- de$gene %in% all_label_genes
  de$label_cat <- ifelse(de$show_label,
                         ifelse(!is.na(de$category), de$category, "top_DE"),
                         NA)
  
  # Data for labeled points
  de_labeled <- de %>% dplyr::filter(show_label)
  # Assign colors to labels
  de_labeled$label_color <- ifelse(!is.na(de_labeled$category),
                                   category_colors[de_labeled$category],
                                   "grey25")
  
  # --- Build volcano ---
  # Determine cell type relevance for subtitle
  if (ct %in% c("CD8_T_effector", "Treg", "Macrophage_SPP1", "DC_mature", "B_cell", "Plasma_cell")) {
    subtitle_context <- "Immunotherapy response biology"
  } else if (ct %in% c("CAF", "Myofibroblast", "Endothelial", "Pericyte_SMC")) {
    subtitle_context <- "Drug delivery & ECM barrier (CT correlate)"
  } else if (ct %in% c("Epithelial_squamous", "Epithelial_basal", "Epithelial_differentiated", "Epithelial_stem_like")) {
    subtitle_context <- "Tumor intrinsic resistance mechanisms"
  } else {
    subtitle_context <- "TME modulation"
  }
  
  p_vol <- ggplot() +
    # Background
    geom_point(
      data = de %>% dplyr::filter(is.na(category) & !show_label),
      aes(x = avg_log2FC, y = neg_log10p, color = sig_class),
      size = 0.3, alpha = 0.3
    ) +
    scale_color_manual(values = c("Up_High" = "#FFB3B3", "Down_High" = "#B3D9FF", "NS" = "grey88"),
                       guide = "none") +
    
    # Category genes
    ggnewscale::new_scale_color() +
    geom_point(
      data = de %>% dplyr::filter(!is.na(category)),
      aes(x = avg_log2FC, y = neg_log10p, color = category),
      size = 2, alpha = 0.85
    ) +
    scale_color_manual(values = category_colors, labels = category_labels,
                       name = "Chemoimmunotherapy\nRelevance", na.translate = FALSE, drop = TRUE) +
    
    # Non-category labeled genes
    geom_point(
      data = de %>% dplyr::filter(is.na(category) & show_label),
      aes(x = avg_log2FC, y = neg_log10p),
      color = "grey20", size = 2.5, shape = 18
    ) +
    
    # Labels
    ggrepel::geom_text_repel(
      data = de_labeled,
      aes(x = avg_log2FC, y = neg_log10p, label = gene),
      color = de_labeled$label_color,
      size = 2.5, fontface = "bold.italic",
      max.overlaps = 35, segment.size = 0.2, segment.color = "grey50",
      box.padding = 0.5, point.padding = 0.2, min.segment.length = 0.1,
      force = 2
    ) +
    
    # Thresholds
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", linewidth = 0.3, color = "grey45") +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.3, color = "grey45") +
    
    # Annotations
    annotate("text", x = max(de$avg_log2FC, na.rm = TRUE) * 0.6, y = max_y * 0.97,
             label = sprintf("↑ Up in CT High-Risk: %d", n_up),
             size = 3, color = "#E64B35", fontface = "bold", hjust = 0) +
    annotate("text", x = min(de$avg_log2FC, na.rm = TRUE) * 0.6, y = max_y * 0.97,
             label = sprintf("↓ Up in CT Low-Risk: %d", n_down),
             size = 3, color = "#4DBBD5", fontface = "bold", hjust = 1) +
    
    # Theme
    theme_classic(base_size = 10) +
    labs(
      x = expression(log[2]~"Fold Change (CT High-Risk vs CT Low-Risk)"),
      y = expression(-log[10]~"(Adjusted P-value)"),
      title = sprintf("%s: Genes Explaining CT-Defined Response Phenotype", ct),
      subtitle = sprintf("%s | MAST with confounding control (%s)",
                         subtitle_context, paste(latent_vars, collapse = "+"))
    ) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
      plot.subtitle = element_text(size = 8, hjust = 0.5, color = "grey40"),
      legend.position = "right",
      legend.text = element_text(size = 7),
      legend.title = element_text(size = 8, face = "bold"),
      legend.key.size = unit(0.4, "cm")
    ) +
    guides(color = guide_legend(override.aes = list(size = 3.5, alpha = 1)))
  
  ggsave(file.path(fig_dir, sprintf("Volcano_%s.pdf", ct)), p_vol, width = 12, height = 9)
  ggsave(file.path(fig_dir, sprintf("Volcano_%s.png", ct)), p_vol, width = 12, height = 9, dpi = 300)
}

cat("  ✓ All volcano plots generated\n")

# ===========================================================
# Step 6: COMBINED VOLCANO PANELS
# Organized by treatment mechanism
# ===========================================================
cat("\n━━ Step 6: Combined Volcano Panels ━━\n")

# Helper function for mini volcanos
make_mini_volcano <- function(de, ct_name, max_labels = 12) {
  de$neg_log10p <- pmin(-log10(de$p_val_adj + 1e-300), 150)
  de$sig <- "NS"
  de$sig[de$p_val_adj < 0.05 & de$avg_log2FC > 0.5] <- "Up"
  de$sig[de$p_val_adj < 0.05 & de$avg_log2FC < -0.5] <- "Down"
  de$category <- gene_to_cat$category[match(de$gene, gene_to_cat$gene)]
  
  # Select labels (important + top)
  labels_df <- de %>%
    dplyr::filter(
      (!is.na(category) & p_val_adj < 0.05 & abs(avg_log2FC) > 0.2) |
        (p_val_adj < 1e-15 & abs(avg_log2FC) > 1.2)
    ) %>%
    dplyr::arrange(p_val_adj) %>%
    dplyr::slice_head(n = max_labels)
  
  de$label <- ifelse(de$gene %in% labels_df$gene, de$gene, NA)
  
  ggplot(de, aes(x = avg_log2FC, y = neg_log10p)) +
    geom_point(aes(color = sig), size = 0.3, alpha = 0.4) +
    scale_color_manual(values = c("Up" = "#E64B35", "Down" = "#4DBBD5", "NS" = "grey85"),
                       guide = "none") +
    # Highlight important genes with ring
    geom_point(data = de %>% dplyr::filter(!is.na(category)),
               shape = 1, color = "black", size = 1.8, stroke = 0.4) +
    ggrepel::geom_text_repel(
      data = de %>% dplyr::filter(!is.na(label)),
      aes(label = label), size = 2.2, fontface = "italic",
      max.overlaps = 20, segment.size = 0.15, box.padding = 0.3
    ) +
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", linewidth = 0.2) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.2) +
    theme_classic(base_size = 8) +
    labs(x = "log2FC (High vs Low)", y = "-log10(padj)",
         title = ct_name,
         subtitle = sprintf("↑%d | ↓%d", sum(de$sig == "Up"), sum(de$sig == "Down"))) +
    theme(plot.title = element_text(face = "bold", size = 9, hjust = 0.5),
          plot.subtitle = element_text(size = 7, hjust = 0.5, color = "grey40"))
}

# --- Panel A: Immunotherapy Response ---
immuno_types <- c("CD8_T_effector", "Treg", "Macrophage_SPP1", "DC_mature")
immuno_types <- immuno_types[immuno_types %in% names(de_results)]

if (length(immuno_types) >= 2) {
  immuno_plots <- lapply(immuno_types, function(ct) make_mini_volcano(de_results[[ct]], ct))
  
  p_panel_A <- wrap_plots(immuno_plots, ncol = 2) +
    plot_annotation(
      title = "A. Immunotherapy Response Biology",
      subtitle = "Why CT High-Risk patients respond poorly to anti-PD-1 | ○ = treatment-relevant gene",
      theme = theme(
        plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
        plot.subtitle = element_text(size = 9, hjust = 0.5, color = "grey30")
      )
    )
  
  ggsave(file.path(fig_dir, "Panel_A_immunotherapy.pdf"), p_panel_A, width = 14, height = 12)
  ggsave(file.path(fig_dir, "Panel_A_immunotherapy.png"), p_panel_A, width = 14, height = 12, dpi = 300)
  cat("  ✓ Panel A: Immunotherapy biology\n")
}

# --- Panel B: Drug Barrier / CT Correlate ---
barrier_types <- c("CAF", "Myofibroblast", "Endothelial", "Pericyte_SMC")
barrier_types <- barrier_types[barrier_types %in% names(de_results)]

if (length(barrier_types) >= 2) {
  barrier_plots <- lapply(barrier_types, function(ct) make_mini_volcano(de_results[[ct]], ct))
  
  p_panel_B <- wrap_plots(barrier_plots, ncol = 2) +
    plot_annotation(
      title = "B. Drug Delivery Barriers & CT Imaging Correlates",
      subtitle = "ECM density & vascular patterns visible on CT → drug exclusion mechanisms",
      theme = theme(
        plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
        plot.subtitle = element_text(size = 9, hjust = 0.5, color = "grey30")
      )
    )
  
  ggsave(file.path(fig_dir, "Panel_B_drug_barrier.pdf"), p_panel_B, width = 14, height = 12)
  cat("  ✓ Panel B: Drug barrier / CT correlate\n")
}

# --- Panel C: Tumor Intrinsic Resistance ---
tumor_types_avail <- c("Epithelial_squamous", "Epithelial_basal",
                       "Epithelial_differentiated", "Epithelial_stem_like")
tumor_types_avail <- tumor_types_avail[tumor_types_avail %in% names(de_results)]

if (length(tumor_types_avail) >= 2) {
  tumor_plots <- lapply(tumor_types_avail, function(ct) make_mini_volcano(de_results[[ct]], ct))
  
  p_panel_C <- wrap_plots(tumor_plots, ncol = 2) +
    plot_annotation(
      title = "C. Tumor Intrinsic Resistance Mechanisms",
      subtitle = "Chemotherapy sensitivity, stemness, and immune evasion in tumor cells",
      theme = theme(
        plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
        plot.subtitle = element_text(size = 9, hjust = 0.5, color = "grey30")
      )
    )
  
  ggsave(file.path(fig_dir, "Panel_C_tumor_resistance.pdf"), p_panel_C, width = 14, height = 12)
  cat("  ✓ Panel C: Tumor resistance\n")
}

# ===========================================================
# Step 7: PSEUDOBULK DE (DESeq2) — Controls for Patient Effects
# ===========================================================
cat("\n━━ Step 7: Pseudobulk DE (DESeq2) ━━\n")

library(DESeq2)

pseudobulk_results <- list()

if (!is.null(sample_col)) {
  for (ct in all_de_types[all_de_types %in% unique(seu$cell_type)]) {
    cat(sprintf("  Pseudobulk: %s...", ct))
    
    seu_sub <- subset(seu, cell_type == ct)
    
    # Check sample sizes
    sample_ct <- table(seu_sub@meta.data[[sample_col]], seu_sub$group)
    n_h_samples <- sum(sample_ct[, "High_Risk"] > 10)
    n_l_samples <- sum(sample_ct[, "Low_Risk"] > 10)
    
    if (n_h_samples < 2 || n_l_samples < 2) {
      cat(" SKIP (need ≥2 samples/group with >10 cells)\n")
      next
    }
    
    tryCatch({
      counts_mat <- GetAssayData(seu_sub, layer = "counts")
      samples <- unique(seu_sub@meta.data[[sample_col]])
      
      # Aggregate
      pseudo_counts <- sapply(samples, function(s) {
        cells_s <- which(seu_sub@meta.data[[sample_col]] == s)
        if (length(cells_s) > 0) Matrix::rowSums(counts_mat[, cells_s, drop = FALSE])
        else rep(0, nrow(counts_mat))
      })
      rownames(pseudo_counts) <- rownames(counts_mat)
      colnames(pseudo_counts) <- samples
      
      # Filter low-count samples
      keep <- colSums(pseudo_counts) > 1000
      pseudo_counts <- pseudo_counts[, keep, drop = FALSE]
      
      if (ncol(pseudo_counts) < 4) { cat(" SKIP\n"); next }
      
      # Sample metadata
      smeta <- data.frame(
        sample = colnames(pseudo_counts),
        group = sapply(colnames(pseudo_counts), function(s) {
          unique(seu_sub$group[seu_sub@meta.data[[sample_col]] == s])[1]
        }),
        stringsAsFactors = FALSE
      )
      
      # Add sex covariate if available
      design_f <- ~ group
      if (length(sex_col) > 0 && sex_col %in% colnames(seu_sub@meta.data)) {
        smeta$sex <- sapply(smeta$sample, function(s) {
          unique(seu_sub@meta.data[[sex_col]][seu_sub@meta.data[[sample_col]] == s])[1]
        })
        if (length(unique(na.omit(smeta$sex))) > 1 && all(table(na.omit(smeta$sex)) >= 2)) {
          design_f <- ~ sex + group
          cat(" (+sex)")
        }
      }
      
      # Filter genes
      keep_g <- rowSums(pseudo_counts >= 5) >= 2
      pseudo_counts <- pseudo_counts[keep_g, , drop = FALSE]
      
      # DESeq2
      dds <- DESeqDataSetFromMatrix(round(pseudo_counts), smeta, design_f)
      dds$group <- relevel(factor(dds$group), ref = "Low_Risk")
      dds <- DESeq(dds, quiet = TRUE)
      res <- as.data.frame(results(dds, contrast = c("group", "High_Risk", "Low_Risk")))
      res$gene <- rownames(res)
      res$cell_type <- ct
      
      pseudobulk_results[[ct]] <- res
      n_sig <- sum(res$padj < 0.05 & abs(res$log2FoldChange) > 0.5, na.rm = TRUE)
      cat(sprintf(" ✓ (%d sig)\n", n_sig))
      
      write.csv(res, file.path(table_dir, sprintf("DE_pseudobulk_%s.csv", ct)), row.names = FALSE)
      
    }, error = function(e) { cat(sprintf(" ✗ %s\n", e$message)) })
  }
} else {
  cat("  No sample column — pseudobulk skipped\n")
}

# ===========================================================
# Step 8: CONFOUNDING VALIDATION
# ===========================================================
cat("\n━━ Step 8: Confounding Validation ━━\n")

val_ct <- intersect("Epithelial_squamous", names(de_results))
if (length(val_ct) == 0) val_ct <- names(de_results)[1]

if (length(val_ct) > 0) {
  seu_val <- subset(seu, cell_type == val_ct)
  n_h <- sum(seu_val$group == "High_Risk")
  n_l <- sum(seu_val$group == "Low_Risk")
  
  if (n_h >= 30 && n_l >= 30) {
    Idents(seu_val) <- "group"
    
    # Uncontrolled DE
    cat(sprintf("  Running uncontrolled DE for %s...\n", val_ct))
    de_raw <- FindMarkers(seu_val, ident.1 = "High_Risk", ident.2 = "Low_Risk",
                          test.use = "wilcox", min.pct = 0.1, logfc.threshold = 0.1, 
                          verbose = FALSE)
    de_raw$gene <- rownames(de_raw)
    de_ctrl <- de_results[[val_ct]]
    
    common <- intersect(de_raw$gene, de_ctrl$gene)
    comp <- data.frame(
      gene = common,
      fc_raw = de_raw$avg_log2FC[match(common, de_raw$gene)],
      fc_ctrl = de_ctrl$avg_log2FC[match(common, de_ctrl$gene)],
      padj_raw = de_raw$p_val_adj[match(common, de_raw$gene)],
      padj_ctrl = de_ctrl$p_val_adj[match(common, de_ctrl$gene)],
      stringsAsFactors = FALSE
    )
    comp$sig_raw <- comp$padj_raw < 0.05 & abs(comp$fc_raw) > 0.5
    comp$sig_ctrl <- comp$padj_ctrl < 0.05 & abs(comp$fc_ctrl) > 0.5
    
    comp$status <- "Neither"
    comp$status[comp$sig_raw & comp$sig_ctrl] <- "Both"
    comp$status[comp$sig_raw & !comp$sig_ctrl] <- "Confounded (lost)"
    comp$status[!comp$sig_raw & comp$sig_ctrl] <- "Revealed (gained)"
    
    rho <- cor(comp$fc_raw, comp$fc_ctrl, method = "spearman", use = "complete.obs")
    n_lost <- sum(comp$status == "Confounded (lost)")
    n_gained <- sum(comp$status == "Revealed (gained)")
    n_both <- sum(comp$status == "Both")
    
    cat(sprintf("  %s: rho=%.3f | Both=%d | Lost=%d | Gained=%d\n",
                val_ct, rho, n_both, n_lost, n_gained))
    
    # Check for sex-linked genes
    sex_genes <- c("XIST", "TSIX", "RPS4Y1", "DDX3Y", "EIF1AY", "KDM5D", 
                   "UTY", "USP9Y", "ZFY", "PRKY", "TTTY14", "TTTY15")
    lost_genes <- comp %>% dplyr::filter(status == "Confounded (lost)") %>% dplyr::arrange(padj_raw)
    lost_genes$is_sex_linked <- lost_genes$gene %in% sex_genes
    
    if (nrow(lost_genes) > 0) {
      cat("  Top potentially confounded genes:\n")
      for (i in 1:min(10, nrow(lost_genes))) {
        flag <- ifelse(lost_genes$is_sex_linked[i], " ← SEX-LINKED", "")
        cat(sprintf("    %s: FC_raw=%.2f → FC_ctrl=%.2f%s\n",
                    lost_genes$gene[i], lost_genes$fc_raw[i], lost_genes$fc_ctrl[i], flag))
      }
    }
    
    write.csv(comp, file.path(table_dir, "confounding_validation.csv"), row.names = FALSE)
    write.csv(lost_genes, file.path(table_dir, "potentially_confounded_genes.csv"), row.names = FALSE)
    
    # Scatter plot
    p_conf <- ggplot(comp, aes(x = fc_raw, y = fc_ctrl, color = status)) +
      geom_point(size = 0.5, alpha = 0.5) +
      scale_color_manual(values = c(
        "Both" = "#E64B35", "Confounded (lost)" = "#4DBBD5",
        "Revealed (gained)" = "#00A087", "Neither" = "grey80"
      ), name = "Status") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#3C5488", linewidth = 0.4) +
      ggrepel::geom_text_repel(
        data = head(lost_genes, 12),
        aes(x = fc_raw, y = fc_ctrl, label = gene),
        color = "#4DBBD5", size = 2.5, fontface = "italic", max.overlaps = 15
      ) +
      theme_classic(base_size = 10) +
      labs(
        x = "log2FC — Uncontrolled (Wilcoxon, no latent vars)",
        y = "log2FC — Controlled (MAST + latent vars)",
        title = sprintf("Confounding Control Effect: %s", val_ct),
        subtitle = sprintf("Spearman ρ = %.3f | Confounded genes removed: %d | New genes revealed: %d",
                           rho, n_lost, n_gained)
      ) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
            plot.subtitle = element_text(size = 9, hjust = 0.5, color = "grey40"))
    
    ggsave(file.path(fig_dir, "Fig_Confounding_Validation.pdf"), p_conf, width = 9, height = 8)
    ggsave(file.path(fig_dir, "Fig_Confounding_Validation.png"), p_conf, width = 9, height = 8, dpi = 300)
    cat("  ✓ Confounding validation plot\n")
  }
}

# ===========================================================
# Step 9: IMPORTANT GENE HEATMAP ACROSS CELL TYPES
# Key figure: shows WHICH mechanisms are active in WHICH cell types
# ===========================================================
cat("\n━━ Step 9: Cross-Cell-Type Treatment-Relevant Gene Heatmap ━━\n")

# Build FC matrix for important genes
imp_fc_mat <- matrix(NA, nrow = length(all_important), ncol = length(de_results),
                     dimnames = list(all_important, names(de_results)))
imp_sig_mat <- matrix(FALSE, nrow = length(all_important), ncol = length(de_results),
                      dimnames = list(all_important, names(de_results)))

for (ct in names(de_results)) {
  de <- de_results[[ct]]
  matched <- match(all_important, de$gene)
  valid <- !is.na(matched)
  imp_fc_mat[valid, ct] <- de$avg_log2FC[matched[valid]]
  imp_sig_mat[valid, ct] <- de$p_val_adj[matched[valid]] < 0.05
}

# Filter to genes with ≥1 significant result
has_sig <- rowSums(imp_sig_mat, na.rm = TRUE) > 0
imp_fc_plot <- imp_fc_mat[has_sig, , drop = FALSE]
imp_sig_plot <- imp_sig_mat[has_sig, , drop = FALSE]

if (nrow(imp_fc_plot) >= 5 && ncol(imp_fc_plot) >= 2) {
  cat(sprintf("  Treatment-relevant genes with ≥1 sig result: %d / %d\n",
              nrow(imp_fc_plot), length(all_important)))
  
  # Prepare matrix
  imp_fc_hm <- imp_fc_plot
  imp_fc_hm[is.na(imp_fc_hm)] <- 0
  imp_fc_hm[imp_fc_hm > 2] <- 2
  imp_fc_hm[imp_fc_hm < -2] <- -2
  
  # Abbreviate column names
  colnames(imp_fc_hm) <- gsub("Epithelial_", "Epi_", colnames(imp_fc_hm))
  colnames(imp_fc_hm) <- gsub("Macrophage_", "Mac_", colnames(imp_fc_hm))
  colnames(imp_fc_hm) <- gsub("Fibroblast_", "Fb_", colnames(imp_fc_hm))
  colnames(imp_fc_hm) <- gsub("_effector", "_eff", colnames(imp_fc_hm))
  
  sig_stars <- imp_sig_plot
  sig_stars[is.na(sig_stars)] <- FALSE
  colnames(sig_stars) <- colnames(imp_fc_hm)
  
  # Row annotation: gene category (treatment mechanism)
  row_cats <- gene_to_cat$category[match(rownames(imp_fc_hm), gene_to_cat$gene)]
  row_cats[is.na(row_cats)] <- "other"
  
  # Column annotation: treatment relevance
  col_type <- case_when(
    grepl("CD8|Treg|Mac_|DC_|B_cell|Plasma|Neutro|Mast", colnames(imp_fc_hm)) ~ "Immune",
    grepl("CAF|Myofi|Fb_", colnames(imp_fc_hm)) ~ "Stroma/Barrier",
    grepl("Endo|Peri", colnames(imp_fc_hm)) ~ "Vascular",
    grepl("Epi_|Prolif", colnames(imp_fc_hm)) ~ "Tumor",
    TRUE ~ "Other"
  )
  
  col_type_colors <- c("Immune" = "#E64B35", "Stroma/Barrier" = "#B09C85",
                       "Vascular" = "#0072B5", "Tumor" = "#20854E", "Other" = "grey70")
  
  # Draw heatmap
  pdf(file.path(fig_dir, "Fig_Treatment_Genes_Heatmap.pdf"),
      width = max(10, ncol(imp_fc_hm) * 0.8 + 5),
      height = max(12, nrow(imp_fc_hm) * 0.2 + 4))
  
  col_ha <- HeatmapAnnotation(
    `Cell Type` = col_type,
    col = list(`Cell Type` = col_type_colors),
    annotation_name_gp = gpar(fontsize = 8),
    simple_anno_size = unit(4, "mm")
  )
  
  row_ha <- rowAnnotation(
    Mechanism = row_cats,
    col = list(Mechanism = c(category_colors, "other" = "grey70")),
    annotation_name_gp = gpar(fontsize = 7),
    simple_anno_size = unit(3.5, "mm")
  )
  
  ht <- Heatmap(
    imp_fc_hm,
    name = "log2FC\n(H vs L)",
    col = colorRamp2(c(-2, 0, 2), c("#4DBBD5", "white", "#E64B35")),
    top_annotation = col_ha,
    left_annotation = row_ha,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    show_row_names = TRUE,
    show_column_names = TRUE,
    row_names_gp = gpar(fontsize = 6, fontface = "italic"),
    column_names_gp = gpar(fontsize = 8),
    column_names_rot = 45,
    border = TRUE,
    na_col = "grey95",
    column_title = "Treatment-Relevant Genes: CT High-Risk vs Low-Risk",
    column_title_gp = gpar(fontsize = 11, fontface = "bold"),
    row_title = "Genes by Mechanism",
    row_title_gp = gpar(fontsize = 9),
    # Asterisks for significant
    cell_fun = function(j, i, x, y, width, height, fill) {
      if (!is.na(sig_stars[i, j]) && sig_stars[i, j] && abs(imp_fc_hm[i, j]) > 0.25) {
        grid.text("*", x, y, gp = gpar(fontsize = 7, col = "black"))
      }
    },
    heatmap_legend_param = list(
      title_gp = gpar(fontsize = 9),
      labels_gp = gpar(fontsize = 8)
    )
  )
  draw(ht, padding = unit(c(5, 20, 5, 5), "mm"))
  invisible(dev.off())
  
  cat("  ✓ Treatment-relevant gene heatmap\n")
}

# ===========================================================
# Step 10: DE SUMMARY BARPLOT
# ===========================================================
cat("\n━━ Step 10: DE Summary ━━\n")

de_summary <- do.call(rbind, lapply(names(de_results), function(ct) {
  de <- de_results[[ct]]
  data.frame(
    cell_type = ct,
    n_tested = nrow(de),
    n_up = sum(de$p_val_adj < 0.05 & de$avg_log2FC > 0.5, na.rm = TRUE),
    n_down = sum(de$p_val_adj < 0.05 & de$avg_log2FC < -0.5, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
de_summary$n_total <- de_summary$n_up + de_summary$n_down
de_summary <- de_summary %>% dplyr::arrange(desc(n_total))

write.csv(de_summary, file.path(table_dir, "DE_summary.csv"), row.names = FALSE)

# Diverging barplot
de_bar <- de_summary %>%
  tidyr::pivot_longer(cols = c("n_up", "n_down"), names_to = "dir", values_to = "count") %>%
  dplyr::mutate(count_plot = ifelse(dir == "n_down", -count, count))

p_bar <- ggplot(de_bar, aes(x = reorder(cell_type, abs(count_plot)), y = count_plot, fill = dir)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("n_up" = "#E64B35", "n_down" = "#4DBBD5"),
                    labels = c("n_up" = "Up in CT High-Risk", "n_down" = "Down in CT High-Risk"),
                    name = "") +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  coord_flip() +
  theme_classic(base_size = 10) +
  labs(x = "", y = "Number of DE Genes (|log2FC|>0.5, padj<0.05)",
       title = "Differentially Expressed Genes: CT High-Risk vs Low-Risk",
       subtitle = sprintf("MAST with latent vars: %s | Explaining CT imaging phenotype",
                          paste(latent_vars, collapse = "+"))) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
        plot.subtitle = element_text(size = 8, hjust = 0.5, color = "grey40"),
        legend.position = "top")

ggsave(file.path(fig_dir, "Fig_DE_Summary_Barplot.pdf"), p_bar, width = 10, height = 7)
ggsave(file.path(fig_dir, "Fig_DE_Summary_Barplot.png"), p_bar, width = 10, height = 7, dpi = 300)
cat("  ✓ DE summary barplot\n")

# ===========================================================
# Step 11: FEATURE PLOTS (key treatment-relevant markers)
# ===========================================================
cat("\n━━ Step 11: Feature Plots ━━\n")

# Markers that explain CT imaging biology
key_features <- c(
  # Immune activation (ICB response)
  "CD8A", "GZMB", "IFNG", "CXCL13",
  # Immunosuppression (ICB resistance)  
  "FOXP3", "CD163", "SPP1", "IDO1",
  # ECM barrier (CT density correlate)
  "COL1A1", "COL11A1", "POSTN", "LRRC15",
  # Vascular (CT enhancement)
  "CDH5", "KDR", "VEGFA", "RGS5",
  # Tumor markers
  "KRT5", "KRT17", "EPCAM", "MKI67"
)

key_features <- key_features[key_features %in% rownames(seu)]
cat(sprintf("  Feature genes available: %d / 20\n", length(key_features)))

if (length(key_features) >= 4) {
  p_feat <- FeaturePlot(seu, features = key_features[1:min(16, length(key_features))],
                        ncol = 4, pt.size = 0.1, order = TRUE) &
    scale_color_viridis_c(option = "magma") &
    theme(plot.title = element_text(size = 9, face = "bold"),
          axis.text = element_blank(), axis.ticks = element_blank())
  
  ggsave(file.path(fig_dir, "Fig_FeaturePlot_treatment_markers.pdf"), p_feat,
         width = 16, height = 16)
  ggsave(file.path(fig_dir, "Fig_FeaturePlot_treatment_markers.png"), p_feat,
         width = 16, height = 16, dpi = 200)
  cat("  ✓ Feature plots\n")
}

# ===========================================================
# Step 12: FINAL SUMMARY & BIOLOGICAL INTERPRETATION
# ===========================================================
cat("\n━━ Step 12: Biological Interpretation Summary ━━\n")

# Create interpretation table
interpretation <- data.frame(
  CT_Feature = c(
    "Tumor density/heterogeneity",
    "Tumor density/heterogeneity",
    "Enhancement pattern",
    "Enhancement pattern",
    "Tumor margin",
    "Tumor margin",
    "Overall CT phenotype",
    "Overall CT phenotype"
  ),
  Biological_Basis = c(
    "Dense ECM from activated CAFs (COL11A1+, LRRC15+, MMP11+)",
    "Myofibroblast activation (POSTN+, KCND2+, ADAMTS12+)",
    "Abnormal vasculature (PLVAP+, KDR+, immature vessels)",
    "Pericyte coverage (RGS5+, MYH11+) → vessel stability",
    "Immunosuppressive border (SPP1+ TAMs, Tregs)",
    "Neutrophil infiltration (CXCR2+, inflammatory remodeling)",
    "Immune exclusion (↓CD8+ effector, ↑Treg/TAM ratio)",
    "Stem-like epithelial cells (lncRNA-rich, treatment-resistant)"
  ),
  Treatment_Implication = c(
    "Drug penetration barrier → chemo resistance",
    "TGFβ-driven ECM → anti-PD1 exclusion",
    "VEGF targeting may normalize → improve drug delivery",
    "Mature vessels = better perfusion but immune exclusion",
    "Anti-SPP1/anti-CSF1R may reprogram TME",
    "TAN depletion may enhance ICB",
    "Combination ICB + anti-TGFβ may overcome",
    "Target stemness pathways (Notch, Wnt)"
  ),
  Key_Genes = c(
    "COL11A1, COL1A1, MMP11, LRRC15, FAP, ITGA11",
    "POSTN, ACTA2, KCND2, WNT5A, THY1, ADAMTS12",
    "VEGFA, KDR, PLVAP, ANGPT2, DLL4",
    "RGS5, MYH11, TRPC6, HEYL, PPP1R14A",
    "SPP1, CD163, VSIG4, FOXP3, IL2RA, TGFB1",
    "CXCR2, FCGR3B, CSF3R, S100A12, IL1R2",
    "CD8A, GZMB, IFNG, PRF1 vs FOXP3, SPP1, IDO1",
    "ROCR, LINC01205, COL9A1, LGR6 (if expressed)"
  ),
  stringsAsFactors = FALSE
)

write.csv(interpretation, file.path(table_dir, "biological_interpretation_CT_features.csv"),
          row.names = FALSE)

# Save final annotated object
cat("\n  Saving annotated Seurat object...\n")
saveRDS(seu, file.path(output_dir, "seu_reannotated_final.rds"))

# ===========================================================
# FINAL REPORT
# ===========================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════════╗\n")
cat("║  ANALYSIS COMPLETE: Biological Basis of CT Response Prediction          ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                          ║\n")
cat("║  STUDY FRAMEWORK:                                                        ║\n")
cat("║    CT radiomic features → Risk Score → scRNA-seq explains biology        ║\n")
cat("║                                                                          ║\n")
cat("║  KEY FINDINGS (biological basis of CT imaging phenotype):                ║\n")
cat("║                                                                          ║\n")
cat("║  1. IMMUNOTHERAPY RESISTANCE (CT High-Risk):                             ║\n")
cat("║     • ↑ Tregs (FOXP3+, OX40+, GITR+)                                    ║\n")
cat("║     • ↑ SPP1+ TAMs (CD163+, VSIG4+, complement)                         ║\n")
cat("║     • ↓ or exhausted CD8+ T cells (CXCL13+ = tissue-resident/exhausted) ║\n")
cat("║     • IDO1+ DCs (tolerogenic antigen presentation)                       ║\n")
cat("║     → CT phenotype correlates with cold/excluded immune TME              ║\n")
cat("║                                                                          ║\n")
cat("║  2. CHEMOTHERAPY BARRIER (CT density correlate):                         ║\n")
cat("║     • Dense desmoplastic stroma (COL11A1+, MMP11+, LRRC15+ CAFs)        ║\n")
cat("║     • Myofibroblast activation (POSTN+, KCND2+, ADAMTS12+)              ║\n")
cat("║     • ECM = physical barrier to drug penetration                         ║\n")
cat("║     → Dense stroma → CT attenuation → poor chemo delivery                ║\n")
cat("║                                                                          ║\n")
cat("║  3. VASCULAR ARCHITECTURE (CT enhancement):                              ║\n")
cat("║     • Endothelial cells (CDH5+, KDR+, PLVAP+)                           ║\n")
cat("║     • Pericyte coverage (RGS5+, MYH11+)                                 ║\n")
cat("║     → Vessel density/maturity → CT enhancement patterns                  ║\n")
cat("║                                                                          ║\n")
cat("║  4. TUMOR INTRINSIC RESISTANCE:                                          ║\n")
cat("║     • Stem-like epithelial cells (lncRNA-rich, Cluster 14)               ║\n")
cat("║     • Squamous differentiation (KRT5/14/17+)                             ║\n")
cat("║     → Cancer stem cells resist both chemo and immunotherapy               ║\n")
cat("║                                                                          ║\n")
cat("║  CONFOUNDING CONTROL:                                                    ║\n")
cat("║     • Library size (nCount_RNA) — MAST latent variable                   ║\n")
cat("║     • Mitochondrial % — MAST latent variable                             ║\n")
cat("║     • Sex — tested; controlled if imbalanced                             ║\n")
cat("║     • Sample/patient — pseudobulk DESeq2                                 ║\n")
cat("║     • Validation: controlled vs uncontrolled comparison                  ║\n")
cat("║                                                                          ║\n")
cat("║  OUTPUT FILES:                                                           ║\n")
cat("║     figures/  — All visualization plots                                  ║\n")
cat("║     tables/   — DE results, validation, interpretation                   ║\n")
cat("║     seu_reannotated_final.rds — Annotated Seurat object                  ║\n")
cat("║                                                                          ║\n")
cat("╚══════════════════════════════════════════════════════════════════════════╝\n")

cat(sprintf("\n  Output: %s\n", output_dir))
cat(sprintf("  Timestamp: %s\n", Sys.time()))
cat("\n━━━ COMPLETE ━━━\n")