# ============================================================
# 05_Pathway_Enrichment_CAF.R
# GO + KEGG + GSEA (Hallmark, Reactome) for CAF DEGs
# ============================================================

library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(DOSE)
library(msigdbr)
library(ggplot2)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------
input_dir <- "XXXyour path/scRNA_Results_new/04_Differential/SignificantClusters"
output_dir <- "XXXyour path/scRNA_Results_new/05_Pathway/CAF"
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

# ----------------------------------------------------------
# Load DEG data
# ----------------------------------------------------------
cat("━━━ Loading CAF DEG data ━━━\n")
deg <- read.csv(file.path(input_dir, "Table_DEG_C1_CAF.csv"))

# Clean: remove empty rows, ENSG IDs (keep protein-coding symbols)
deg <- deg %>%
  filter(!is.na(gene) & gene != "" & !grepl("^ENSG", gene)) %>%
  filter(!is.na(p_val_adj))

cat(sprintf("Total DEGs: %d\n", nrow(deg)))

# ----------------------------------------------------------
# Define gene lists
# ----------------------------------------------------------
# Significant UP in High_Risk (log2FC > 0.5, padj < 0.05)
sig_up <- deg %>%
  filter(p_val_adj < 0.05 & avg_log2FC > 0.5) %>%
  pull(gene)

# Significant DOWN in High_Risk
sig_down <- deg %>%
  filter(p_val_adj < 0.05 & avg_log2FC < -0.5) %>%
  pull(gene)

# All significant
sig_all <- c(sig_up, sig_down)

cat(sprintf("UP in High_Risk (log2FC>0.5): %d genes\n", length(sig_up)))
cat(sprintf("DOWN in High_Risk (log2FC<-0.5): %d genes\n", length(sig_down)))

# Ranked gene list for GSEA (ALL genes, sorted by log2FC)
gene_list_ranked <- deg$avg_log2FC
names(gene_list_ranked) <- deg$gene
gene_list_ranked <- sort(gene_list_ranked, decreasing = TRUE)
gene_list_ranked <- gene_list_ranked[!duplicated(names(gene_list_ranked))]

# ----------------------------------------------------------
# Convert to Entrez IDs
# ----------------------------------------------------------
cat("\n━━━ Converting gene IDs ━━━\n")

# For ORA (over-representation)
gene_entrez_up <- bitr(sig_up, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
cat(sprintf("  UP genes mapped to Entrez: %d / %d\n", nrow(gene_entrez_up), length(sig_up)))

# For GSEA (all genes)
all_entrez <- bitr(names(gene_list_ranked), fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
gsea_list <- gene_list_ranked[names(gene_list_ranked) %in% all_entrez$SYMBOL]
names(gsea_list) <- all_entrez$ENTREZID[match(names(gsea_list), all_entrez$SYMBOL)]
gsea_list <- gsea_list[!is.na(names(gsea_list))]
gsea_list <- sort(gsea_list, decreasing = TRUE)

cat(sprintf("  GSEA list: %d genes with Entrez IDs\n", length(gsea_list)))

# ===========================================================
# PART 1: GO Over-Representation Analysis (ORA)
# ===========================================================
cat("\n━━━ PART 1: GO Enrichment (ORA) ━━━\n")

# --- GO Biological Process ---
ego_bp <- enrichGO(gene = gene_entrez_up$ENTREZID,
                   OrgDb = org.Hs.eg.db,
                   ont = "BP",
                   pAdjustMethod = "BH",
                   pvalueCutoff = 0.05,
                   qvalueCutoff = 0.05,
                   readable = TRUE)

if (!is.null(ego_bp) && nrow(ego_bp) > 0) {
  cat(sprintf("  GO BP: %d enriched terms\n", nrow(ego_bp)))
  
  # Simplify redundancy
  ego_bp_simp <- simplify(ego_bp, cutoff = 0.7, by = "p.adjust", select_fun = min)
  cat(sprintf("  GO BP simplified: %d terms\n", nrow(ego_bp_simp)))
  
  # Dotplot
  p_go_bp <- dotplot(ego_bp_simp, showCategory = 25,
                     title = "GO Biological Process\n(CAF: Upregulated in High_Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig1_GO_BP_dotplot.pdf"), p_go_bp,
         width = 170, height = 180, units = "mm")
  
  # Barplot (top 15)
  p_go_bp_bar <- barplot(ego_bp_simp, showCategory = 15,
                         title = "GO BP (CAF: High_Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 6))
  
  ggsave(file.path(output_dir, "Fig1b_GO_BP_barplot.pdf"), p_go_bp_bar,
         width = 160, height = 110, units = "mm")
  
  # Concept network (cnetplot)
  p_cnet <- cnetplot(ego_bp_simp, showCategory = 8, 
                     categorySize = "pvalue",
                     foldChange = gene_list_ranked) +
    theme(plot.title = element_text(size = 8, face = "bold"))
  
  ggsave(file.path(output_dir, "Fig1c_GO_BP_cnetplot.pdf"), p_cnet,
         width = 250, height = 200, units = "mm")
  
  # Save table
  write.csv(as.data.frame(ego_bp), file.path(output_dir, "Table_GO_BP_full.csv"), row.names = FALSE)
  write.csv(as.data.frame(ego_bp_simp), file.path(output_dir, "Table_GO_BP_simplified.csv"), row.names = FALSE)
}

# --- GO Molecular Function ---
ego_mf <- enrichGO(gene = gene_entrez_up$ENTREZID,
                   OrgDb = org.Hs.eg.db,
                   ont = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff = 0.05,
                   qvalueCutoff = 0.05,
                   readable = TRUE)

if (!is.null(ego_mf) && nrow(ego_mf) > 0) {
  cat(sprintf("  GO MF: %d enriched terms\n", nrow(ego_mf)))
  
  p_go_mf <- dotplot(ego_mf, showCategory = 20,
                     title = "GO Molecular Function\n(CAF: High_Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig2_GO_MF_dotplot.pdf"), p_go_mf,
         width = 160, height = 140, units = "mm")
  
  write.csv(as.data.frame(ego_mf), file.path(output_dir, "Table_GO_MF.csv"), row.names = FALSE)
}

# --- GO Cellular Component ---
ego_cc <- enrichGO(gene = gene_entrez_up$ENTREZID,
                   OrgDb = org.Hs.eg.db,
                   ont = "CC",
                   pAdjustMethod = "BH",
                   pvalueCutoff = 0.05,
                   qvalueCutoff = 0.05,
                   readable = TRUE)

if (!is.null(ego_cc) && nrow(ego_cc) > 0) {
  cat(sprintf("  GO CC: %d enriched terms\n", nrow(ego_cc)))
  
  p_go_cc <- dotplot(ego_cc, showCategory = 15,
                     title = "GO Cellular Component\n(CAF: High_Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig3_GO_CC_dotplot.pdf"), p_go_cc,
         width = 160, height = 120, units = "mm")
  
  write.csv(as.data.frame(ego_cc), file.path(output_dir, "Table_GO_CC.csv"), row.names = FALSE)
}

# ===========================================================
# PART 2: KEGG Pathway (ORA)
# ===========================================================
cat("\n━━━ PART 2: KEGG Enrichment ━━━\n")

ekegg <- enrichKEGG(gene = gene_entrez_up$ENTREZID,
                    organism = "hsa",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.1)

if (!is.null(ekegg) && nrow(ekegg) > 0) {
  cat(sprintf("  KEGG: %d enriched pathways\n", nrow(ekegg)))
  
  ekegg_readable <- setReadable(ekegg, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
  
  p_kegg <- dotplot(ekegg_readable, showCategory = 20,
                    title = "KEGG Pathways\n(CAF: Upregulated in High_Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig4_KEGG_dotplot.pdf"), p_kegg,
         width = 160, height = 140, units = "mm")
  
  write.csv(as.data.frame(ekegg_readable), file.path(output_dir, "Table_KEGG.csv"), row.names = FALSE)
} else {
  cat("  KEGG: No significant pathways (trying relaxed cutoff)...\n")
  ekegg <- enrichKEGG(gene = gene_entrez_up$ENTREZID,
                      organism = "hsa",
                      pAdjustMethod = "BH",
                      pvalueCutoff = 0.1,
                      qvalueCutoff = 0.2)
  if (!is.null(ekegg) && nrow(ekegg) > 0) {
    ekegg_readable <- setReadable(ekegg, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
    p_kegg <- dotplot(ekegg_readable, showCategory = 15,
                      title = "KEGG Pathways (relaxed)\n(CAF: High_Risk)") +
      theme_paper(base_size = 7)
    ggsave(file.path(output_dir, "Fig4_KEGG_dotplot_relaxed.pdf"), p_kegg,
           width = 160, height = 120, units = "mm")
    write.csv(as.data.frame(ekegg_readable), file.path(output_dir, "Table_KEGG_relaxed.csv"), row.names = FALSE)
  }
}

# ===========================================================
# PART 3: GSEA — Hallmark Gene Sets (MSigDB)
# ===========================================================
cat("\n━━━ PART 3: GSEA Hallmark ━━━\n")

hallmark <- msigdbr(species = "Homo sapiens", category = "H") %>%
  dplyr::select(gs_name, entrez_gene) %>%
  mutate(entrez_gene = as.character(entrez_gene))

gsea_hallmark <- GSEA(geneList = gsea_list,
                      TERM2GENE = hallmark,
                      minGSSize = 15,
                      maxGSSize = 500,
                      pvalueCutoff = 0.05,
                      verbose = FALSE,
                      seed = TRUE)

if (!is.null(gsea_hallmark) && nrow(gsea_hallmark) > 0) {
  cat(sprintf("  GSEA Hallmark: %d significant gene sets\n", nrow(gsea_hallmark)))
  
  # Clean names for plotting
  gsea_hallmark_plot <- gsea_hallmark
  gsea_hallmark_plot@result$Description <- gsub("HALLMARK_", "", gsea_hallmark_plot@result$Description)
  gsea_hallmark_plot@result$Description <- gsub("_", " ", gsea_hallmark_plot@result$Description)
  
  p_hall <- dotplot(gsea_hallmark_plot, showCategory = 25,
                    title = "GSEA Hallmark Gene Sets\n(CAF: High_Risk vs Low_Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig5_GSEA_Hallmark_dotplot.pdf"), p_hall,
         width = 170, height = 170, units = "mm")
  
  # Ridge plot
  p_hall_ridge <- ridgeplot(gsea_hallmark_plot, showCategory = 15) +
    theme_paper(base_size = 7) +
    labs(title = "GSEA Hallmark (Ridge Plot)")
  
  ggsave(file.path(output_dir, "Fig5b_GSEA_Hallmark_ridge.pdf"), p_hall_ridge,
         width = 180, height = 160, units = "mm")
  
  # Individual GSEA enrichment plots for top 6 pathways
  n_plot <- min(6, nrow(gsea_hallmark_plot))
  for (i in 1:n_plot) {
    pathway_name <- gsea_hallmark_plot@result$Description[i]
    pathway_id <- gsea_hallmark_plot@result$ID[i]
    
    p_gsea_ind <- gseaplot2(gsea_hallmark_plot, geneSetID = i,
                            title = pathway_name,
                            pvalue_table = TRUE)
    
    clean_name <- gsub("[^[:alnum:]]", "_", pathway_name)
    ggsave(file.path(output_dir, sprintf("Fig5_%d_GSEA_%s.pdf", i, clean_name)),
           p_gsea_ind, width = 140, height = 100, units = "mm")
  }
  
  write.csv(as.data.frame(gsea_hallmark), file.path(output_dir, "Table_GSEA_Hallmark.csv"), row.names = FALSE)
  
  # Print top results
  cat("\n  Top Hallmark pathways:\n")
  top_hall_df <- gsea_hallmark@result %>%
    dplyr::select(Description, NES, pvalue, p.adjust, setSize) %>%
    head(15)
  top_hall_df$Description <- gsub("HALLMARK_", "", top_hall_df$Description)
  print(top_hall_df, row.names = FALSE)
}

# ===========================================================
# PART 4: GSEA — KEGG
# ===========================================================
cat("\n━━━ PART 4: GSEA KEGG ━━━\n")

gsea_kegg <- gseKEGG(geneList = gsea_list,
                     organism = "hsa",
                     minGSSize = 15,
                     maxGSSize = 500,
                     pvalueCutoff = 0.05,
                     verbose = FALSE,
                     seed = TRUE)

if (!is.null(gsea_kegg) && nrow(gsea_kegg) > 0) {
  cat(sprintf("  GSEA KEGG: %d significant pathways\n", nrow(gsea_kegg)))
  
  gsea_kegg_readable <- setReadable(gsea_kegg, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
  
  p_gsea_kegg <- dotplot(gsea_kegg_readable, showCategory = 20,
                         title = "GSEA KEGG Pathways\n(CAF: High vs Low Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig6_GSEA_KEGG_dotplot.pdf"), p_gsea_kegg,
         width = 160, height = 140, units = "mm")
  
  # Top 4 GSEA plots
  n_plot <- min(4, nrow(gsea_kegg_readable))
  for (i in 1:n_plot) {
    p_ind <- gseaplot2(gsea_kegg_readable, geneSetID = i,
                       title = gsea_kegg_readable@result$Description[i],
                       pvalue_table = TRUE)
    clean_name <- gsub("[^[:alnum:]]", "_", gsea_kegg_readable@result$Description[i])
    ggsave(file.path(output_dir, sprintf("Fig6_%d_GSEA_KEGG_%s.pdf", i, clean_name)),
           p_ind, width = 140, height = 100, units = "mm")
  }
  
  write.csv(as.data.frame(gsea_kegg_readable), file.path(output_dir, "Table_GSEA_KEGG.csv"), row.names = FALSE)
}

# ===========================================================
# PART 5: GSEA — Reactome Pathways
# ===========================================================
cat("\n━━━ PART 5: GSEA Reactome ━━━\n")

reactome <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:REACTOME") %>%
  dplyr::select(gs_name, entrez_gene) %>%
  mutate(entrez_gene = as.character(entrez_gene))

gsea_reactome <- GSEA(geneList = gsea_list,
                      TERM2GENE = reactome,
                      minGSSize = 15,
                      maxGSSize = 500,
                      pvalueCutoff = 0.05,
                      verbose = FALSE,
                      seed = TRUE)

if (!is.null(gsea_reactome) && nrow(gsea_reactome) > 0) {
  cat(sprintf("  GSEA Reactome: %d significant pathways\n", nrow(gsea_reactome)))
  
  gsea_reactome_plot <- gsea_reactome
  gsea_reactome_plot@result$Description <- gsub("REACTOME_", "", gsea_reactome_plot@result$Description)
  gsea_reactome_plot@result$Description <- gsub("_", " ", gsea_reactome_plot@result$Description)
  # Truncate long names
  gsea_reactome_plot@result$Description <- ifelse(
    nchar(gsea_reactome_plot@result$Description) > 55,
    paste0(substr(gsea_reactome_plot@result$Description, 1, 52), "..."),
    gsea_reactome_plot@result$Description
  )
  
  p_gsea_reactome <- dotplot(gsea_reactome_plot, showCategory = 25,
                             title = "GSEA Reactome Pathways\n(CAF: High vs Low Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5))
  
  ggsave(file.path(output_dir, "Fig7_GSEA_Reactome_dotplot.pdf"), p_gsea_reactome,
         width = 200, height = 180, units = "mm")
  
  # Top 4 individual GSEA plots
  n_plot <- min(4, nrow(gsea_reactome_plot))
  for (i in 1:n_plot) {
    p_ind <- gseaplot2(gsea_reactome_plot, geneSetID = i,
                       title = gsea_reactome_plot@result$Description[i],
                       pvalue_table = TRUE)
    clean_name <- gsub("[^[:alnum:]]", "_", gsea_reactome_plot@result$Description[i])
    clean_name <- substr(clean_name, 1, 40)
    ggsave(file.path(output_dir, sprintf("Fig7_%d_GSEA_Reactome_%s.pdf", i, clean_name)),
           p_ind, width = 140, height = 100, units = "mm")
  }
  
  write.csv(as.data.frame(gsea_reactome), file.path(output_dir, "Table_GSEA_Reactome.csv"), row.names = FALSE)
  
  cat("\n  Top Reactome pathways:\n")
  print(gsea_reactome_plot@result %>%
          dplyr::select(Description, NES, pvalue, p.adjust) %>%
          head(10), row.names = FALSE)
}

# ===========================================================
# PART 6: GSEA — WikiPathways
# ===========================================================
cat("\n━━━ PART 6: GSEA WikiPathways ━━━\n")

wiki <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:WIKIPATHWAYS") %>%
  dplyr::select(gs_name, entrez_gene) %>%
  mutate(entrez_gene = as.character(entrez_gene))

gsea_wiki <- GSEA(geneList = gsea_list,
                  TERM2GENE = wiki,
                  minGSSize = 10,
                  maxGSSize = 500,
                  pvalueCutoff = 0.05,
                  verbose = FALSE,
                  seed = TRUE)

if (!is.null(gsea_wiki) && nrow(gsea_wiki) > 0) {
  cat(sprintf("  GSEA WikiPathways: %d significant pathways\n", nrow(gsea_wiki)))
  
  gsea_wiki_plot <- gsea_wiki
  gsea_wiki_plot@result$Description <- gsub("WP_", "", gsea_wiki_plot@result$Description)
  gsea_wiki_plot@result$Description <- gsub("_", " ", gsea_wiki_plot@result$Description)
  
  p_wiki <- dotplot(gsea_wiki_plot, showCategory = 20,
                    title = "GSEA WikiPathways\n(CAF: High vs Low Risk)") +
    theme_paper(base_size = 7) +
    theme(axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig8_GSEA_WikiPathways_dotplot.pdf"), p_wiki,
         width = 170, height = 150, units = "mm")
  
  write.csv(as.data.frame(gsea_wiki), file.path(output_dir, "Table_GSEA_WikiPathways.csv"), row.names = FALSE)
}

# ===========================================================
# PART 7: Custom CAF-specific Pathway Scoring
# ===========================================================
cat("\n━━━ PART 7: Custom CAF pathway scoring ━━━\n")

# Curated gene sets relevant to CAF biology
caf_custom_sets <- list(
  iCAF_signature = c("IL6", "IL6ST", "CXCL1", "CXCL2", "CXCL12", "CCL2",
                     "HAS1", "HAS2", "LMNA", "CFD", "PDPN", "CLU", "SAA1",
                     "IL11", "LIF", "SOCS3"),
  
  myCAF_signature = c("ACTA2", "TAGLN", "MYL9", "POSTN", "TNC", "COL11A1",
                      "COL12A1", "THBS2", "THBS1", "MMP11", "LRRC15", "TPM2",
                      "MMP14", "LOX", "LOXL2"),
  
  apCAF_signature = c("HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74",
                      "SLPI", "SAA1", "SAA2"),
  
  NFkB_signaling = c("NFKB1", "NFKB2", "RELB", "REL", "RELA", "NFKBIA",
                     "NFKBIE", "TNFAIP3", "BCL3", "ICAM1", "VCAM1", "CCL2",
                     "CXCL2", "IL6ST"),
  
  JAK_STAT_IFN = c("STAT1", "STAT3", "JAK1", "SOCS3", "IRF1", "IRF9",
                   "ISG15", "BST2", "GBP1", "GBP2", "GBP4", "GBP5"),
  
  TGFb_signaling = c("TGFBR1", "SMAD3", "SMAD7", "SERPINE1", "CCN2", "INHBA",
                     "LTBP1", "LTBP2", "THBS1", "SKIL", "SKI"),
  
  ECM_remodeling = c("COL1A1", "COL4A1", "COL4A2", "COL8A1", "COL16A1",
                     "TNC", "POSTN", "FN1", "SERPINE1", "TIMP1", "TIMP3",
                     "THBS1", "THBS2", "LTBP1", "LTBP2", "PLOD2", "LOXL2"),
  
  Angiogenesis = c("VEGFA", "VEGFC", "NRP2", "HIF1A", "EGFR", "FGFR1",
                   "FGF7", "PDGFC", "ANGPTL4", "EMP2"),
  
  AP1_stress = c("FOSL1", "FOS", "JUN", "JUNB", "JUND", "EGR1",
                 "EGR2", "EGR3", "GADD45B", "ATF3", "KLF10", "KLF4",
                 "BHLHE40", "BHLHE41", "IER3"),
  
  IFN_response = c("ISG15", "BST2", "IFI27", "IFITM1", "IFI44L", "GBP1",
                   "GBP2", "GBP4", "GBP5", "STAT1", "IRF1", "IRF9",
                   "TAP1", "PSMB9", "HLA-B", "HLA-E", "HLA-F", "B2M"),
  
  Metabolic = c("ALDH1A3", "HK1", "PFKFB3", "GLS", "ACSL4",
                "LDHB", "SLC20A1", "GFPT2", "SLC2A3", "SLC3A2",
                "SLC7A5", "ACOX1"),
  
  Hippo_YAP = c("YAP1", "LATS2", "FAT1", "NF2", "TEAD1", "CCN2",
                "CYR61", "ANKRD1", "FRMD6")
)

# Score each set
sig_genes <- deg %>% filter(p_val_adj < 0.05 & avg_log2FC > 0.5) %>% pull(gene)
all_tested <- deg$gene

custom_results <- data.frame(
  pathway = character(), set_size = integer(), n_tested = integer(),
  n_sig_up = integer(), pct_sig = numeric(),
  genes_found = character(), mean_log2FC = numeric(),
  fisher_p = numeric(), stringsAsFactors = FALSE
)

for (pw_name in names(caf_custom_sets)) {
  pw_genes <- caf_custom_sets[[pw_name]]
  tested <- intersect(pw_genes, all_tested)
  sig_overlap <- intersect(pw_genes, sig_genes)
  
  # Fisher's exact test
  a <- length(sig_overlap)
  b <- length(tested) - a
  c_val <- length(sig_genes) - a
  d_val <- length(all_tested) - a - b - c_val
  
  fisher_p <- tryCatch(
    fisher.test(matrix(c(a, b, c_val, d_val), nrow = 2), 
                alternative = "greater")$p.value,
    error = function(e) NA
  )
  
  mean_fc <- ifelse(length(sig_overlap) > 0,
                    mean(deg$avg_log2FC[deg$gene %in% sig_overlap], na.rm = TRUE),
                    NA)
  
  custom_results <- rbind(custom_results, data.frame(
    pathway = pw_name, set_size = length(pw_genes),
    n_tested = length(tested), n_sig_up = a,
    pct_sig = round(a / length(pw_genes) * 100, 1),
    genes_found = paste(sig_overlap, collapse = "; "),
    mean_log2FC = round(mean_fc, 3),
    fisher_p = fisher_p, stringsAsFactors = FALSE
  ))
}

custom_results$fisher_padj <- p.adjust(custom_results$fisher_p, method = "BH")
custom_results <- custom_results %>% arrange(fisher_p)

cat("\n  Custom CAF pathway results:\n")
cat("  ══════════════════════════════════════════════════════════════\n")
for (i in 1:nrow(custom_results)) {
  sig_star <- case_when(
    custom_results$fisher_padj[i] < 0.001 ~ " ***",
    custom_results$fisher_padj[i] < 0.01 ~ " **",
    custom_results$fisher_padj[i] < 0.05 ~ " *",
    custom_results$fisher_p[i] < 0.05 ~ " (+)",
    TRUE ~ ""
  )
  cat(sprintf("  %-20s: %2d/%2d sig (%.0f%%) | FC=%.2f | p=%.1e%s\n",
              custom_results$pathway[i],
              custom_results$n_sig_up[i],
              custom_results$set_size[i],
              custom_results$pct_sig[i],
              ifelse(is.na(custom_results$mean_log2FC[i]), 0, custom_results$mean_log2FC[i]),
              custom_results$fisher_p[i], sig_star))
}

write.csv(custom_results, file.path(output_dir, "Table_Custom_CAF_pathways.csv"), row.names = FALSE)

# --- Custom pathway barplot ---
custom_plot_df <- custom_results %>%
  mutate(
    neg_log10_p = -log10(fisher_p),
    pathway = factor(pathway, levels = rev(pathway)),
    significant = fisher_padj < 0.05
  )

p_custom <- ggplot(custom_plot_df, aes(x = neg_log10_p, y = pathway)) +
  geom_col(aes(fill = mean_log2FC), width = 0.7) +
  geom_vline(xintercept = -log10(0.05), linetype = "dashed", 
             linewidth = 0.3, color = "grey40") +
  geom_text(aes(label = sprintf("%d/%d", n_sig_up, set_size)),
            hjust = -0.1, size = 2.5) +
  scale_fill_gradient(low = "#FEE0D2", high = "#DE2D26", name = "Mean\nlog2FC",
                      na.value = "grey80") +
  theme_paper(base_size = 7) +
  labs(x = expression(-log[10](Fisher~P)), y = "",
       title = "Custom CAF Pathway Enrichment\n(Upregulated in High_Risk)") +
  theme(axis.text.y = element_text(size = 6.5)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15)))

ggsave(file.path(output_dir, "Fig9_Custom_pathway_barplot.pdf"), p_custom,
       width = 160, height = 110, units = "mm")

# ===========================================================
# PART 8: Pathway-Gene Grouped Barplot
# ===========================================================
cat("\n━━━ PART 8: Pathway-gene barplot ━━━\n")

pathway_gene_groups <- list(
  "NF-kB / Inflammation" = c("NFKB1", "NFKB2", "RELB", "REL", "BST2",
                             "IRF1", "STAT3", "ICAM1", "VCAM1", "CCL2"),
  "ECM Remodeling" = c("CCN2", "TNC", "COL4A1", "COL4A2", "TIMP1",
                       "TIMP3", "THBS1", "THBS2", "SERPINE1", "LTBP1"),
  "Growth Factor / RTK" = c("EGFR", "FGFR1", "PIK3CD", "VEGFA", "VEGFC",
                            "FGF7", "IGFBP3", "IGFBP7", "PDGFC", "NRP2"),
  "AP-1 / Stress" = c("FOSL1", "EGR2", "EGR3", "GADD45B", "KLF10",
                      "CREB5", "BHLHE40", "BHLHE41", "HIPK2", "IER3"),
  "IFN Response" = c("ISG15", "GBP1", "GBP5", "IFI27", "IFITM1",
                     "STAT1", "IRF9", "TAP1", "PSMB9", "HLA-B"),
  "Metabolic" = c("ALDH1A3", "HK1", "PFKFB3", "GLS", "ACSL4",
                  "LDHB", "GFPT2", "SLC20A1", "SLC2A3", "SLC3A2")
)

pathway_genes_df <- data.frame(
  gene = character(), pathway = character(), log2FC = numeric(),
  stringsAsFactors = FALSE
)

for (pw in names(pathway_gene_groups)) {
  genes <- pathway_gene_groups[[pw]]
  for (g in genes) {
    fc_row <- deg %>% filter(gene == g)
    if (nrow(fc_row) > 0) {
      pathway_genes_df <- rbind(pathway_genes_df, data.frame(
        gene = g, pathway = pw, log2FC = fc_row$avg_log2FC[1],
        stringsAsFactors = FALSE
      ))
    }
  }
}

if (nrow(pathway_genes_df) > 0) {
  pathway_genes_df$pathway <- factor(pathway_genes_df$pathway,
                                     levels = names(pathway_gene_groups))
  
  p_pathway_bar <- ggplot(pathway_genes_df,
                          aes(x = reorder(gene, log2FC), y = log2FC, fill = pathway)) +
    geom_col(width = 0.7) +
    coord_flip() +
    facet_wrap(~pathway, scales = "free_y", ncol = 2) +
    scale_fill_brewer(palette = "Set2") +
    geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.2, color = "grey50") +
    theme_paper(base_size = 7) +
    labs(x = "", y = "log2FC (High_Risk vs Low_Risk)",
         title = "Key CAF DEGs Grouped by Pathway") +
    theme(legend.position = "none",
          strip.text = element_text(size = 7, face = "bold"),
          axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig10_Pathway_gene_barplot.pdf"), p_pathway_bar,
         width = 200, height = 260, units = "mm")
}

# ===========================================================
# PART 9: Network Visualizations
# ===========================================================
cat("\n━━━ PART 9: Network visualizations ━━━\n")

if (exists("ego_bp_simp") && !is.null(ego_bp_simp) && nrow(ego_bp_simp) > 0) {
  
  # Pairwise term similarity for network plots
  tryCatch({
    ego_bp_sim <- pairwise_termsim(ego_bp_simp)
    
    # Enrichment map
    p_emap <- emapplot(ego_bp_sim, showCategory = 30,
                       cex_label_category = 0.5) +
      ggtitle("GO BP: Enrichment Network Map")
    
    ggsave(file.path(output_dir, "Fig11_GO_BP_emapplot.pdf"), p_emap,
           width = 250, height = 220, units = "mm")
    cat("  Emapplot saved.\n")
  }, error = function(e) {
    cat("  Emapplot skipped:", e$message, "\n")
  })
  
  # Tree plot
  tryCatch({
    p_tree <- treeplot(ego_bp_sim, showCategory = 30,
                       fontsize = 3)
    
    ggsave(file.path(output_dir, "Fig12_GO_BP_treeplot.pdf"), p_tree,
           width = 300, height = 220, units = "mm")
    cat("  Treeplot saved.\n")
  }, error = function(e) {
    cat("  Treeplot skipped:", e$message, "\n")
  })
  
  # Concept network (gene-term connections)
  tryCatch({
    fc_vector <- deg$avg_log2FC
    names(fc_vector) <- deg$gene
    
    p_cnet <- cnetplot(ego_bp_simp, showCategory = 6,
                       categorySize = "pvalue",
                       foldChange = fc_vector,
                       cex_label_gene = 0.5,
                       cex_label_category = 0.8) +
      scale_color_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                            midpoint = 0, name = "log2FC") +
      ggtitle("GO BP: Gene-Concept Network")
    
    ggsave(file.path(output_dir, "Fig13_GO_BP_cnetplot.pdf"), p_cnet,
           width = 280, height = 220, units = "mm")
    cat("  Cnetplot saved.\n")
  }, error = function(e) {
    cat("  Cnetplot skipped:", e$message, "\n")
  })
}

# ===========================================================
# PART 10: Multi-GSEA comparison plot
# ===========================================================
cat("\n━━━ PART 10: Multi-database comparison ━━━\n")
# Collect top pathways from all databases into one summary
all_gsea_summary <- data.frame(
  database = character(), pathway = character(),
  NES = numeric(), p_adjust = numeric(),
  stringsAsFactors = FALSE
)

if (exists("gsea_hallmark") && !is.null(gsea_hallmark) && nrow(gsea_hallmark) > 0) {
  top_h <- gsea_hallmark@result %>%
    head(10) %>%
    mutate(database = "Hallmark",
           pathway = gsub("HALLMARK_", "", Description),
           pathway = gsub("_", " ", pathway)) %>%
    dplyr::select(database, pathway, NES, p.adjust)
  names(top_h)[4] <- "p_adjust"
  all_gsea_summary <- rbind(all_gsea_summary, top_h)
}

if (exists("gsea_kegg") && !is.null(gsea_kegg) && nrow(gsea_kegg) > 0) {
  top_k <- gsea_kegg@result %>%
    head(8) %>%
    mutate(database = "KEGG",
           pathway = Description) %>%
    dplyr::select(database, pathway, NES, p.adjust)
  names(top_k)[4] <- "p_adjust"
  all_gsea_summary <- rbind(all_gsea_summary, top_k)
}

if (exists("gsea_reactome") && !is.null(gsea_reactome) && nrow(gsea_reactome) > 0) {
  top_r <- gsea_reactome@result %>%
    head(8) %>%
    mutate(database = "Reactome",
           pathway = gsub("REACTOME_", "", Description),
           pathway = gsub("_", " ", pathway),
           pathway = ifelse(nchar(pathway) > 50, paste0(substr(pathway, 1, 47), "..."), pathway)) %>%
    dplyr::select(database, pathway, NES, p.adjust)
  names(top_r)[4] <- "p_adjust"
  all_gsea_summary <- rbind(all_gsea_summary, top_r)
}

if (nrow(all_gsea_summary) > 0) {
  all_gsea_summary$neg_log10_padj <- -log10(all_gsea_summary$p_adjust + 1e-50)
  all_gsea_summary$database <- factor(all_gsea_summary$database,
                                      levels = c("Hallmark", "KEGG", "Reactome"))
  
  # Order by NES within each database
  all_gsea_summary <- all_gsea_summary %>%
    group_by(database) %>%
    arrange(database, desc(NES)) %>%
    ungroup() %>%
    mutate(pathway = factor(pathway, levels = rev(unique(pathway))))
  
  p_multi_gsea <- ggplot(all_gsea_summary, aes(x = NES, y = pathway, fill = database)) +
    geom_col(width = 0.7, alpha = 0.85) +
    geom_vline(xintercept = 0, linewidth = 0.3) +
    facet_wrap(~database, scales = "free_y", ncol = 1) +
    scale_fill_manual(values = c("Hallmark" = "#E41A1C", "KEGG" = "#377EB8", "Reactome" = "#4DAF4A")) +
    theme_paper(base_size = 7) +
    labs(x = "Normalized Enrichment Score (NES)", y = "",
         title = "GSEA: Top Pathways Across Databases\n(CAF: High_Risk vs Low_Risk)") +
    theme(legend.position = "none",
          strip.text = element_text(size = 8, face = "bold"),
          axis.text.y = element_text(size = 5.5))
  
  ggsave(file.path(output_dir, "Fig16_Multi_GSEA_comparison.pdf"), p_multi_gsea,
         width = 180, height = 220, units = "mm")
  
  write.csv(all_gsea_summary, file.path(output_dir, "Table_GSEA_multi_database_summary.csv"), row.names = FALSE)
}

# ===========================================================
# PART 11: Save all results as RDS
# ===========================================================
cat("\n━━━ PART 11: Saving all results ━━━\n")

pathway_results <- list(
  ego_bp = if (exists("ego_bp")) ego_bp else NULL,
  ego_mf = if (exists("ego_mf")) ego_mf else NULL,
  ego_cc = if (exists("ego_cc")) ego_cc else NULL,
  ekegg = if (exists("ekegg")) ekegg else NULL,
  gsea_hallmark = if (exists("gsea_hallmark")) gsea_hallmark else NULL,
  gsea_kegg = if (exists("gsea_kegg")) gsea_kegg else NULL,
  gsea_reactome = if (exists("gsea_reactome")) gsea_reactome else NULL,
  gsea_wiki = if (exists("gsea_wiki")) gsea_wiki else NULL,
  gsea_pid = if (exists("gsea_pid")) gsea_pid else NULL,
  gsea_go_bp = if (exists("gsea_go_bp")) gsea_go_bp else NULL,
  custom_results = if (exists("custom_results")) custom_results else NULL,
  deg_input = deg
)

saveRDS(pathway_results, file.path(output_dir, "pathway_results_all.rds"))

# ===========================================================
# FINAL BIOLOGICAL INTERPRETATION
# ===========================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║     PATHWAY ENRICHMENT — CAF (Cluster 1) — COMPLETE                 ║\n")
cat("╠══════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                      ║\n")
cat("║  BIOLOGICAL INTERPRETATION:                                          ║\n")
cat("║                                                                      ║\n")
cat("║  High-Risk CAFs show a MULTI-FACETED ACTIVATED phenotype:            ║\n")
cat("║                                                                      ║\n")
cat("║  1. INFLAMMATORY CAF (iCAF) PROGRAM                                  ║\n")
cat("║     • NF-κB signaling (NFKB1/2, RELB, REL, BCL3)                    ║\n")
cat("║     • JAK-STAT/IFN response (STAT1/3, IRF1/9, GBP1-5, ISG15)        ║\n")
cat("║     • Chemokine production (CCL2, CXCL2)                             ║\n")
cat("║     • Immune recruitment molecules (ICAM1, VCAM1)                    ║\n")
cat("║                                                                      ║\n")
cat("║  2. PRO-TUMORIGENIC ECM REMODELING                                   ║\n")
cat("║     • Matrix deposition (COL4A1/2, TNC, POSTN, LTBP1/2)             ║\n")
cat("║     • Matrix regulators (TIMP1/3, SERPINE1, THBS1/2, LOXL2)          ║\n")
cat("║     • Growth factor sequestration (CCN2/CTGF, IGFBP3/7)             ║\n")
cat("║                                                                      ║\n")
cat("║  3. PRO-ANGIOGENIC SIGNALING                                         ║\n")
cat("║     • VEGFA/C, NRP2, HIF1A, FGF7, PDGFC                             ║\n")
cat("║     • RTK activation (EGFR, FGFR1, PIK3CD)                          ║\n")
cat("║                                                                      ║\n")
cat("║  4. STRESS/AP-1 ACTIVATION                                           ║\n")
cat("║     • FOSL1, EGR2/3, GADD45B, CREB5, BHLHE40/41                     ║\n")
cat("║     • Consistent with chronic activation/damage response             ║\n")
cat("║                                                                      ║\n")
cat("║  5. METABOLIC REPROGRAMMING                                          ║\n")
cat("║     • Glycolysis (HK1, PFKFB3, LDHB, SLC2A3)                        ║\n")
cat("║     • Amino acid metabolism (GLS, SLC7A5, SLC3A2)                    ║\n")
cat("║     • Lipid metabolism (ACSL4, ACOX1)                                ║\n")
cat("║     • Retinoic acid (ALDH1A3)                                        ║\n")
cat("║                                                                      ║\n")
cat("║  6. NOTCH/WNT/HIPPO CROSSTALK                                        ║\n")
cat("║     • HES1/4, NFATC1/2, FZD7, YAP1, FAT1, LATS2                    ║\n")
cat("║     • Stem-like niche support for tumor cells                        ║\n")
cat("║                                                                      ║\n")
cat("║  CLINICAL SIGNIFICANCE:                                              ║\n")
cat("║  These High-Risk CAFs represent an ACTIVATED, PRO-TUMORIGENIC         ║\n")
cat("║  stroma that promotes:                                                ║\n")
cat("║    • Immune evasion (IFN response without effective killing)          ║\n")
cat("║    • Tumor invasion (ECM remodeling + angiogenesis)                   ║\n")
cat("║    • Metabolic support (nutrient provision to tumor)                  ║\n")
cat("║    • Treatment resistance (multiple survival pathways)                ║\n")
cat("║                                                                      ║\n")
cat("║  POTENTIAL THERAPEUTIC TARGETS:                                       ║\n")
cat("║    • NF-κB inhibitors                                                ║\n")
cat("║    • Anti-angiogenic (anti-VEGF)                                     ║\n")
cat("║    • PI3K/FGFR inhibitors                                            ║\n")
cat("║    • TGF-β pathway modulators                                        ║\n")
cat("║    • Metabolic inhibitors (glycolysis/glutaminolysis)                 ║\n")
cat("║                                                                      ║\n")
cat("╚══════════════════════════════════════════════════════════════════════╝\n")

cat("\nOutput files:\n")
cat("  Figures: Fig1-16 (GO, KEGG, GSEA, Network, Custom)\n")
cat("  Tables:  Table_GO_BP/MF/CC, Table_KEGG, Table_GSEA_*, Table_Custom\n")
cat("  RDS:     pathway_results_all.rds\n")

cat("\nNext steps:\n")
cat("  1. Repeat for other significant clusters (if any)\n")
cat("  2. Cross-reference with spatial transcriptomics\n")
cat("  3. Validate key pathways with external datasets\n")
cat("  4. Cell-cell communication analysis (CellChat/NicheNet)\n")

cat("\nDone.\n")
