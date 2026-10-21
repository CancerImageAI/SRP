# ============================================================
# Nature-Quality Circle Plots: CAF L-R Communication
# Revised: No overlaps, clear text, visible differences
# ============================================================

library(CellChat)
library(tidyverse)
library(igraph)
library(RColorBrewer)
library(circlize)
library(grid)
library(gridBase)

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
output_dir <- "XXXyour path/scRNA_Results_new/06_CellChat"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Font: use Helvetica (universally available)
FONT_FAMILY <- "Helvetica"
pdf.options(family = FONT_FAMILY)

# Nature dimensions
DOUBLE_COL <- 183 / 25.4  # 7.2 inches
SINGLE_COL <- 89 / 25.4   # 3.5 inches

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Loading CellChat objects\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cellchat_high <- readRDS(file.path(output_dir, "cellchat_High_Risk.rds"))
cellchat_low <- readRDS(file.path(output_dir, "cellchat_Low_Risk.rds"))

# ----------------------------------------------------------
# Build L-R comparison
# ----------------------------------------------------------
lr_high <- subsetCommunication(cellchat_high, sources.use = "CAF")
lr_low <- subsetCommunication(cellchat_low, sources.use = "CAF")

lr_high$key <- paste(lr_high$interaction_name_2, lr_high$target, sep = " | ")
lr_low$key <- paste(lr_low$interaction_name_2, lr_low$target, sep = " | ")

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
    status = case_when(
      prob_High > 0 & prob_Low == 0 ~ "High_only",
      prob_High == 0 & prob_Low > 0 ~ "Low_only",
      diff > 0 ~ "Up_in_High",
      diff < 0 ~ "Down_in_High",
      TRUE ~ "Equal"
    )
  ) %>%
  arrange(desc(diff))

# Select top pairs
n_top <- 20

top_up <- lr_comparison %>%
  filter(status %in% c("High_only", "Up_in_High")) %>%
  arrange(desc(diff)) %>%
  head(n_top)

top_down <- lr_comparison %>%
  filter(status %in% c("Low_only", "Down_in_High")) %>%
  arrange(diff) %>%
  head(n_top)

cat(sprintf("  Top %d upregulated, %d downregulated pairs identified.\n",
            nrow(top_up), nrow(top_down)))

# ----------------------------------------------------------
# Color palette (colorblind-friendly, Nature-compatible)
# ----------------------------------------------------------
ct_all <- sort(unique(c(levels(cellchat_high@idents), levels(cellchat_low@idents))))
n_ct <- length(ct_all)

# Curated palette: distinct, muted, publication-ready
curated_colors <- c(
  "B_cell" = "#4DBBD5",
  "CAF" = "#E64B35",
  "CD8_T_effector" = "#00A087",
  "DC_mature" = "#3C5488",
  "Endothelial" = "#F39B7F",
  "Epithelial_basal" = "#8491B4",
  "Epithelial_differentiated" = "#91D1C2",
  "Epithelial_squamous" = "#B09C85",
  "Epithelial_stem_like" = "#7E6148",
  "Fibroblast_normal" = "#E4E1E3",
  "Macrophage_SPP1" = "#DC0000",
  "Mast_cell" = "#F0E685",
  "Myofibroblast" = "#AED6F1",
  "Neutrophil" = "#D4AC0D",
  "Pericyte_SMC" = "#7DCEA0",
  "Plasma" = "#BB8FCE",
  "Proliferating" = "#F1948A",
  "Treg" = "#85C1E9"
)

ct_colors <- curated_colors[ct_all]
missing_ct <- ct_all[is.na(ct_colors)]
if (length(missing_ct) > 0) {
  extra_cols <- colorRampPalette(brewer.pal(8, "Set2"))(length(missing_ct))
  ct_colors[missing_ct] <- extra_cols
}

# Pathway colors
all_pathways <- unique(c(top_up$pathway_name, top_down$pathway_name))
pw_palette <- c("#2C3E50", "#8E44AD", "#2980B9", "#27AE60", "#F39C12",
                "#E74C3C", "#1ABC9C", "#D35400", "#7F8C8D", "#C0392B",
                "#16A085", "#2E86C1", "#6C3483", "#1F618D", "#196F3D",
                "#B7950B", "#A93226", "#117A65", "#2874A6", "#6E2C00")
pathway_colors <- setNames(pw_palette[1:length(all_pathways)], all_pathways)

# ===========================================================
# MAIN CIRCLE PLOT FUNCTION (Revised: No overlaps)
# ===========================================================
# Key improvements:
# 1. Labels placed OUTSIDE the circle with adequate spacing
# 2. Edge labels (pathway) placed along edges without overlap
# 3. Consistent edge scaling between groups
# 4. Clear visual difference between active/inactive targets
# 5. Adequate margins

plot_circle_nature <- function(lr_df, cellchat_obj, title_text, prob_col,
                               ct_colors_input, pathway_colors_input,
                               max_edge_scale = NULL,
                               panel_label = NULL,
                               show_pathway_on_edge = TRUE,
                               label_size = 0.5) {
  
  ct_names <- levels(cellchat_obj@idents)
  n_ct_local <- length(ct_names)
  
  cell_counts <- as.numeric(table(cellchat_obj@idents))
  names(cell_counts) <- ct_names
  
  # Prepare plot data
  plot_data <- lr_df %>%
    rename(prob = !!sym(prob_col)) %>%
    filter(target %in% ct_names)
  
  # Aggregate by target (sum probability, list pathways)
  target_summary <- plot_data %>%
    filter(prob > 0) %>%
    group_by(target) %>%
    summarise(
      total_prob = sum(prob),
      n_pairs = n(),
      pathways = paste(unique(pathway_name), collapse = ", "),
      top_pathway = names(sort(table(pathway_name), decreasing = TRUE))[1],
      .groups = "drop"
    )
  
  # ── Layout: Circle with CAF at top, evenly spaced ──
  ct_order <- c("CAF", sort(setdiff(ct_names, "CAF")))
  
  # Compute angles (evenly distributed)
  angles <- seq(pi/2, pi/2 - 2*pi, length.out = n_ct_local + 1)[1:n_ct_local]
  names(angles) <- ct_order
  
  # Node positions on unit circle (radius = 1)
  node_x <- cos(angles)
  node_y <- sin(angles)
  names(node_x) <- names(node_y) <- ct_order
  
  # ── Plot setup ──
  plot(NULL, xlim = c(-1.8, 1.8), ylim = c(-1.8, 1.8),
       asp = 1, axes = FALSE, xlab = "", ylab = "")
  
  # ── Draw edges (CAF → targets) ──
  if (nrow(target_summary) > 0) {
    # Scale factor for edge width (use consistent max across groups)
    if (is.null(max_edge_scale)) {
      max_w <- max(target_summary$total_prob)
    } else {
      max_w <- max_edge_scale
    }
    
    caf_x <- node_x["CAF"]
    caf_y <- node_y["CAF"]
    
    for (i in 1:nrow(target_summary)) {
      tgt <- target_summary$target[i]
      tgt_x <- node_x[tgt]
      tgt_y <- node_y[tgt]
      
      # Normalized weight
      w_norm <- target_summary$total_prob[i] / max_w
      
      # Edge width (Nature: visible but not overwhelming)
      edge_lwd <- w_norm * 6 + 0.5
      
      # Edge color by dominant pathway
      pw <- target_summary$top_pathway[i]
      edge_col <- pathway_colors_input[pw]
      if (is.na(edge_col)) edge_col <- "#7F8C8D"
      edge_col_alpha <- adjustcolor(edge_col, alpha.f = 0.4 + w_norm * 0.5)
      
      # Draw curved edge (Bezier-like using arrows)
      # Compute control point for slight curve
      mid_x <- (caf_x + tgt_x) / 2
      mid_y <- (caf_y + tgt_y) / 2
      
      # Perpendicular offset for curve
      dx <- tgt_x - caf_x
      dy <- tgt_y - caf_y
      perp_x <- -dy * 0.1
      perp_y <- dx * 0.1
      
      ctrl_x <- mid_x + perp_x
      ctrl_y <- mid_y + perp_y
      
      # Draw with xspline for smooth curve
      xspline(c(caf_x, ctrl_x, tgt_x),
              c(caf_y, ctrl_y, tgt_y),
              shape = c(0, 1, 0),
              open = TRUE,
              border = edge_col_alpha,
              lwd = edge_lwd)
      
      # Arrowhead at target
      arrow_frac <- 0.85
      arr_x <- caf_x + (tgt_x - caf_x) * arrow_frac
      arr_y <- caf_y + (tgt_y - caf_y) * arrow_frac
      arrows(arr_x - (tgt_x - caf_x) * 0.05,
             arr_y - (tgt_y - caf_y) * 0.05,
             arr_x, arr_y,
             length = 0.06, angle = 25, lwd = edge_lwd * 0.6,
             col = edge_col_alpha)
      
      # ── Pathway label on edge ──
      if (show_pathway_on_edge) {
        # Position: 60% along edge, offset perpendicular
        label_frac <- 0.55
        lbl_x <- caf_x + (tgt_x - caf_x) * label_frac + perp_x * 1.5
        lbl_y <- caf_y + (tgt_y - caf_y) * label_frac + perp_y * 1.5
        
        # Rotation angle
        angle_deg <- atan2(tgt_y - caf_y, tgt_x - caf_x) * 180 / pi
        if (angle_deg > 90 || angle_deg < -90) angle_deg <- angle_deg + 180
        
        # Only show if space permits (skip very short edges)
        edge_length <- sqrt((tgt_x - caf_x)^2 + (tgt_y - caf_y)^2)
        if (edge_length > 0.5) {
          text(lbl_x, lbl_y, pw,
               cex = 0.3, col = adjustcolor(edge_col, alpha.f = 0.9),
               font = 3, srt = angle_deg, family = FONT_FAMILY)
        }
      }
    }
  }
  
  # ── Draw nodes ──
  for (ct in ct_order) {
    x <- node_x[ct]
    y <- node_y[ct]
    
    # Node size proportional to cell count
    node_radius <- sqrt(cell_counts[ct] / max(cell_counts)) * 0.08 + 0.03
    
    # Fill color
    fill_col <- ct_colors_input[ct]
    if (is.na(fill_col)) fill_col <- "#E4E1E3"
    
    # Border: thick for CAF and active targets
    is_active <- ct %in% target_summary$target
    if (ct == "CAF") {
      border_col <- "#8B0000"
      border_lwd <- 2.0
    } else if (is_active) {
      border_col <- "#2C3E50"
      border_lwd <- 1.2
    } else {
      border_col <- "#BDC3C7"
      border_lwd <- 0.5
      fill_col <- adjustcolor(fill_col, alpha.f = 0.4)  # Dim inactive
    }
    
    # Draw node
    symbols(x, y, circles = node_radius, add = TRUE, inches = FALSE,
            bg = fill_col, fg = border_col, lwd = border_lwd)
  }
  
  # ── Draw labels OUTSIDE the circle ──
  for (ct in ct_order) {
    x <- node_x[ct]
    y <- node_y[ct]
    
    # Label position: push outward from center
    label_dist <- 1.35  # Distance from center for label
    lbl_x <- cos(angles[ct]) * label_dist
    lbl_y <- sin(angles[ct]) * label_dist
    
    # Text alignment based on position
    angle_ct <- angles[ct]
    if (cos(angle_ct) > 0.1) {
      adj_x <- 0  # Left-align for right side
    } else if (cos(angle_ct) < -0.1) {
      adj_x <- 1  # Right-align for left side
    } else {
      adj_x <- 0.5  # Center for top/bottom
    }
    
    # Font weight: bold for CAF and active targets
    is_active <- ct %in% target_summary$target
    font_type <- ifelse(ct == "CAF" | is_active, 2, 1)
    font_col <- ifelse(ct == "CAF", "#8B0000",
                       ifelse(is_active, "#2C3E50", "#7F8C8D"))
    
    text(lbl_x, lbl_y, ct,
         cex = label_size, adj = c(adj_x, 0.5),
         col = font_col, font = font_type, family = FONT_FAMILY)
    
    # Add n_pairs annotation for active targets
    if (is_active) {
      n_p <- target_summary$n_pairs[target_summary$target == ct]
      prob_val <- target_summary$total_prob[target_summary$target == ct]
      annotation <- sprintf("(%d pairs)", n_p)
      text(lbl_x, lbl_y - 0.09, annotation,
           cex = label_size * 0.65, adj = c(adj_x, 0.5),
           col = "#7F8C8D", font = 1, family = FONT_FAMILY)
    }
  }
  
  # ── Title ──
  text(0, 1.7, title_text, cex = 0.7, font = 2, family = FONT_FAMILY, col = "#2C3E50")
  
  # ── Panel label ──
  if (!is.null(panel_label)) {
    text(-1.7, 1.7, panel_label, cex = 1.0, font = 2, family = FONT_FAMILY, adj = c(0, 1))
  }
  
  # ── Scale indicator ──
  if (nrow(target_summary) > 0) {
    # Mini legend in bottom-left
    leg_x <- -1.6
    leg_y <- -1.4
    
    text(leg_x, leg_y + 0.15, "Edge width = probability",
         cex = 0.3, adj = c(0, 0), col = "#7F8C8D", family = FONT_FAMILY)
    
    # Draw scale lines
    segments(leg_x, leg_y, leg_x + 0.3, leg_y, lwd = 6, col = "#BDC3C7")
    text(leg_x + 0.35, leg_y, sprintf("%.4f", max_w), cex = 0.25,
         adj = c(0, 0.5), col = "#7F8C8D", family = FONT_FAMILY)
    
    segments(leg_x, leg_y - 0.12, leg_x + 0.15, leg_y - 0.12, lwd = 1.5, col = "#BDC3C7")
    text(leg_x + 0.2, leg_y - 0.12,
         sprintf("%.4f", min(target_summary$total_prob)),
         cex = 0.25, adj = c(0, 0.5), col = "#7F8C8D", family = FONT_FAMILY)
  }
  
  # Return summary for reference
  invisible(target_summary)
}

# ===========================================================
# Compute consistent max scale across groups
# ===========================================================
# For fair comparison, both panels should use the same edge scale

# Upregulated pairs
up_max_high <- top_up %>% filter(prob_High > 0) %>%
  group_by(target) %>% summarise(s = sum(prob_High), .groups = "drop") %>%
  pull(s) %>% max(0)
up_max_low <- top_up %>% filter(prob_Low > 0) %>%
  group_by(target) %>% summarise(s = sum(prob_Low), .groups = "drop") %>%
  pull(s) %>% max(0)
up_max_scale <- max(up_max_high, up_max_low, 1e-6)

# Downregulated pairs
down_max_high <- top_down %>% filter(prob_High > 0) %>%
  group_by(target) %>% summarise(s = sum(prob_High), .groups = "drop") %>%
  pull(s) %>% max(0)
down_max_low <- top_down %>% filter(prob_Low > 0) %>%
  group_by(target) %>% summarise(s = sum(prob_Low), .groups = "drop") %>%
  pull(s) %>% max(0)
down_max_scale <- max(down_max_high, down_max_low, 1e-6)

cat(sprintf("  Edge scale — Upregulated: %.5f, Downregulated: %.5f\n",
            up_max_scale, down_max_scale))

# ===========================================================
# FIGURE 1: Upregulated L-R Pairs (High > Low)
# Side-by-side: Low_Risk vs High_Risk
# ===========================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Figure 1: Upregulated L-R pairs comparison\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cairo_pdf(file.path(output_dir, "Fig_Nature_Upregulated_comparison.pdf"),
          width = DOUBLE_COL, height = DOUBLE_COL * 0.52,
          family = FONT_FAMILY)

par(mfrow = c(1, 2), mar = c(0.5, 0.5, 0.5, 0.5), oma = c(0, 0, 1.5, 0))

# Panel a: Low-risk (weak/absent)
plot_circle_nature(top_up, cellchat_low,
                   "Low-risk group",
                   prob_col = "prob_Low",
                   ct_colors, pathway_colors,
                   max_edge_scale = up_max_scale,
                   panel_label = "a",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

# Panel b: High-risk (strong/gained)
plot_circle_nature(top_up, cellchat_high,
                   "High-risk group",
                   prob_col = "prob_High",
                   ct_colors, pathway_colors,
                   max_edge_scale = up_max_scale,
                   panel_label = "b",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

# Overall title
mtext("Top 20 upregulated CAF L-R pairs (high-risk vs low-risk)",
      outer = TRUE, side = 3, line = 0.2, cex = 0.75, font = 2,
      family = FONT_FAMILY)

dev.off()
cat("  Fig_Nature_Upregulated_comparison.pdf saved.\n")

# ===========================================================
# FIGURE 2: Downregulated L-R Pairs (Low > High)
# Side-by-side: Low_Risk vs High_Risk
# ===========================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Figure 2: Downregulated L-R pairs comparison\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cairo_pdf(file.path(output_dir, "Fig_Nature_Downregulated_comparison.pdf"),
          width = DOUBLE_COL, height = DOUBLE_COL * 0.52,
          family = FONT_FAMILY)

par(mfrow = c(1, 2), mar = c(0.5, 0.5, 0.5, 0.5), oma = c(0, 0, 1.5, 0))

# Panel a: Low-risk (active/strong)
plot_circle_nature(top_down, cellchat_low,
                   "Low-risk group",
                   prob_col = "prob_Low",
                   ct_colors, pathway_colors,
                   max_edge_scale = down_max_scale,
                   panel_label = "a",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

# Panel b: High-risk (reduced/lost)
plot_circle_nature(top_down, cellchat_high,
                   "High-risk group",
                   prob_col = "prob_High",
                   ct_colors, pathway_colors,
                   max_edge_scale = down_max_scale,
                   panel_label = "b",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

mtext("Top 20 downregulated CAF L-R pairs (high-risk vs low-risk)",
      outer = TRUE, side = 3, line = 0.2, cex = 0.75, font = 2,
      family = FONT_FAMILY)

dev.off()
cat("  Fig_Nature_Downregulated_comparison.pdf saved.\n")

# ===========================================================
# FIGURE 3: Combined 4-panel figure
# ===========================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Figure 3: Combined 4-panel\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

cairo_pdf(file.path(output_dir, "Fig_Nature_Combined_4panel.pdf"),
          width = DOUBLE_COL, height = DOUBLE_COL,
          family = FONT_FAMILY)

par(mfrow = c(2, 2), mar = c(0.5, 0.5, 1.0, 0.5), oma = c(0, 0, 0, 0))

# Panel a: Upregulated — Low-risk (weak)
plot_circle_nature(top_up, cellchat_low,
                   "Upregulated pairs: Low-risk",
                   prob_col = "prob_Low",
                   ct_colors, pathway_colors,
                   max_edge_scale = up_max_scale,
                   panel_label = "a",
                   show_pathway_on_edge = FALSE,
                   label_size = 0.4)

# Panel b: Upregulated — High-risk (strong)
plot_circle_nature(top_up, cellchat_high,
                   "Upregulated pairs: High-risk",
                   prob_col = "prob_High",
                   ct_colors, pathway_colors,
                   max_edge_scale = up_max_scale,
                   panel_label = "b",
                   show_pathway_on_edge = FALSE,
                   label_size = 0.4)

# Panel c: Downregulated — Low-risk (strong)
plot_circle_nature(top_down, cellchat_low,
                   "Downregulated pairs: Low-risk",
                   prob_col = "prob_Low",
                   ct_colors, pathway_colors,
                   max_edge_scale = down_max_scale,
                   panel_label = "c",
                   show_pathway_on_edge = FALSE,
                   label_size = 0.4)

# Panel d: Downregulated — High-risk (weak)
plot_circle_nature(top_down, cellchat_high,
                   "Downregulated pairs: High-risk",
                   prob_col = "prob_High",
                   ct_colors, pathway_colors,
                   max_edge_scale = down_max_scale,
                   panel_label = "d",
                   show_pathway_on_edge = FALSE,
                   label_size = 0.4)

dev.off()
cat("  Fig_Nature_Combined_4panel.pdf saved.\n")

# ===========================================================
# FIGURE 4: With pathway legend (3-column layout)
# ===========================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Figure 4: With pathway legend\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# -- Upregulated with legend --
cairo_pdf(file.path(output_dir, "Fig_Nature_Upregulated_withLegend.pdf"),
          width = DOUBLE_COL, height = DOUBLE_COL * 0.55,
          family = FONT_FAMILY)

layout(matrix(c(1, 2, 3), nrow = 1), widths = c(5, 5, 2.5))

# Panel a
par(mar = c(0.5, 0.5, 1.5, 0.2))
plot_circle_nature(top_up, cellchat_low,
                   "Low-risk group",
                   prob_col = "prob_Low",
                   ct_colors, pathway_colors,
                   max_edge_scale = up_max_scale,
                   panel_label = "a",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

# Panel b
par(mar = c(0.5, 0.2, 1.5, 0.5))
plot_circle_nature(top_up, cellchat_high,
                   "High-risk group",
                   prob_col = "prob_High",
                   ct_colors, pathway_colors,
                   max_edge_scale = up_max_scale,
                   panel_label = "b",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

# Panel c: Legend
par(mar = c(1, 0.5, 1.5, 0.5))
plot.new()
plot.window(xlim = c(0, 1), ylim = c(0, 1))

# Pathway legend
pw_in_up <- unique(top_up$pathway_name)
n_pw <- length(pw_in_up)

title_y <- 0.95
text(0.05, title_y, "Signaling pathway", adj = c(0, 1),
     cex = 0.5, font = 2, family = FONT_FAMILY)

y_start <- title_y - 0.06
y_step <- min(0.045, 0.85 / max(n_pw, 1))

for (i in 1:n_pw) {
  y_pos <- y_start - (i - 1) * y_step
  pw_col <- pathway_colors[pw_in_up[i]]
  if (is.na(pw_col)) pw_col <- "#7F8C8D"
  
  rect(0.05, y_pos - 0.015, 0.12, y_pos + 0.015,
       col = pw_col, border = NA)
  text(0.15, y_pos, pw_in_up[i], adj = c(0, 0.5),
       cex = 0.38, family = FONT_FAMILY, col = "#2C3E50")
}

# Cell type legend (below pathway)
ct_legend_y <- y_start - n_pw * y_step - 0.08
text(0.05, ct_legend_y, "Cell type", adj = c(0, 1),
     cex = 0.5, font = 2, family = FONT_FAMILY)

active_targets_up <- unique(top_up$target[top_up$prob_High > 0])
legend_cts <- c("CAF", sort(active_targets_up))
n_leg_ct <- length(legend_cts)
ct_y_step <- min(0.04, (ct_legend_y - 0.05) / max(n_leg_ct, 1))

for (i in 1:min(n_leg_ct, 15)) {
  y_pos <- ct_legend_y - 0.05 - (i - 1) * ct_y_step
  ct_col <- ct_colors[legend_cts[i]]
  if (is.na(ct_col)) ct_col <- "#BDC3C7"
  
  symbols(0.085, y_pos, circles = 0.015, add = TRUE, inches = FALSE,
          bg = ct_col, fg = "#2C3E50", lwd = 0.5)
  text(0.15, y_pos, legend_cts[i], adj = c(0, 0.5),
       cex = 0.35, family = FONT_FAMILY, col = "#2C3E50")
}

dev.off()
cat("  Fig_Nature_Upregulated_withLegend.pdf saved.\n")

# -- Downregulated with legend --
cairo_pdf(file.path(output_dir, "Fig_Nature_Downregulated_withLegend.pdf"),
          width = DOUBLE_COL, height = DOUBLE_COL * 0.55,
          family = FONT_FAMILY)

layout(matrix(c(1, 2, 3), nrow = 1), widths = c(5, 5, 2.5))

# Panel a
par(mar = c(0.5, 0.5, 1.5, 0.2))
plot_circle_nature(top_down, cellchat_low,
                   "Low-risk group",
                   prob_col = "prob_Low",
                   ct_colors, pathway_colors,
                   max_edge_scale = down_max_scale,
                   panel_label = "a",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

# Panel b
par(mar = c(0.5, 0.2, 1.5, 0.5))
plot_circle_nature(top_down, cellchat_high,
                   "High-risk group",
                   prob_col = "prob_High",
                   ct_colors, pathway_colors,
                   max_edge_scale = down_max_scale,
                   panel_label = "b",
                   show_pathway_on_edge = TRUE,
                   label_size = 0.45)

# Panel c: Legend
par(mar = c(1, 0.5, 1.5, 0.5))
plot.new()
plot.window(xlim = c(0, 1), ylim = c(0, 1))

pw_in_down <- unique(top_down$pathway_name)
n_pw_d <- length(pw_in_down)

text(0.05, 0.95, "Signaling pathway", adj = c(0, 1),
     cex = 0.5, font = 2, family = FONT_FAMILY)

y_start_d <- 0.89
y_step_d <- min(0.045, 0.85 / max(n_pw_d, 1))

for (i in 1:n_pw_d) {
  y_pos <- y_start_d - (i - 1) * y_step_d
  pw_col <- pathway_colors[pw_in_down[i]]
  if (is.na(pw_col)) pw_col <- "#7F8C8D"
  
  rect(0.05, y_pos - 0.015, 0.12, y_pos + 0.015,
       col = pw_col, border = NA)
  text(0.15, y_pos, pw_in_down[i], adj = c(0, 0.5),
       cex = 0.38, family = FONT_FAMILY, col = "#2C3E50")
}

# Cell type legend
ct_legend_y_d <- y_start_d - n_pw_d * y_step_d - 0.08
text(0.05, ct_legend_y_d, "Cell type", adj = c(0, 1),
     cex = 0.5, font = 2, family = FONT_FAMILY)

active_targets_down <- unique(top_down$target[top_down$prob_Low > 0])
legend_cts_d <- c("CAF", sort(active_targets_down))
n_leg_ct_d <- length(legend_cts_d)
ct_y_step_d <- min(0.04, (ct_legend_y_d - 0.05) / max(n_leg_ct_d, 1))

for (i in 1:min(n_leg_ct_d, 15)) {
  y_pos <- ct_legend_y_d - 0.05 - (i - 1) * ct_y_step_d
  ct_col <- ct_colors[legend_cts_d[i]]
  if (is.na(ct_col)) ct_col <- "#BDC3C7"
  
  symbols(0.085, y_pos, circles = 0.015, add = TRUE, inches = FALSE,
          bg = ct_col, fg = "#2C3E50", lwd = 0.5)
  text(0.15, y_pos, legend_cts_d[i], adj = c(0, 0.5),
       cex = 0.35, family = FONT_FAMILY, col = "#2C3E50")
}

dev.off()
cat("  Fig_Nature_Downregulated_withLegend.pdf saved.\n")

# ===========================================================
# FIGURE 5: Supplementary barplot (differential L-R pairs)
# ===========================================================
cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  Figure 5: Differential barplot\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

theme_nature <- function(base_size = 6) {
  theme_classic(base_size = base_size) %+replace%
    theme(
      text = element_text(family = FONT_FAMILY, size = base_size, color = "black"),
      axis.title = element_text(size = base_size, face = "bold"),
      axis.text = element_text(size = base_size - 0.5, color = "black"),
      axis.text.y = element_text(size = base_size - 1.5),
      plot.title = element_text(size = base_size + 1, face = "bold", hjust = 0),
      axis.line = element_line(linewidth = 0.3, color = "black"),
      axis.ticks = element_line(linewidth = 0.2, color = "black"),
      panel.grid = element_blank(),
      plot.margin = margin(4, 4, 4, 4, "pt"),
      legend.text = element_text(size = base_size - 1),
      legend.title = element_text(size = base_size - 0.5, face = "bold"),
      legend.key.size = unit(0.25, "cm")
    )
}

combined_df <- bind_rows(
  top_up %>% head(15) %>% mutate(direction = "Upregulated (high-risk)"),
  top_down %>% head(15) %>% mutate(direction = "Downregulated (high-risk)")
) %>%
  mutate(
    label = paste0(interaction, " \u2192 ", target),
    label = factor(label, levels = label[order(diff)])
  )

p_bar <- ggplot(combined_df, aes(x = label, y = diff, fill = pathway_name)) +
  geom_col(width = 0.7, color = "grey30", linewidth = 0.15) +
  coord_flip() +
  scale_fill_manual(values = pathway_colors, name = "Pathway") +
  geom_hline(yintercept = 0, linewidth = 0.4, color = "black") +
  theme_nature(base_size = 6) +
  labs(x = NULL,
       y = expression(Delta * " communication probability (high-risk " - " low-risk)"),
       title = "Differentially active CAF ligand-receptor pairs") +
  theme(legend.position = "right")

ggsave(file.path(output_dir, "Fig_Nature_Barplot_differential_LR.pdf"), p_bar,
       width = DOUBLE_COL, height = DOUBLE_COL * 0.75,
       units = "in", device = cairo_pdf, family = FONT_FAMILY)

cat("  Barplot saved.\n")

# ===========================================================
# FIGURE 6: Paired dot plot
# ===========================================================
cat("\n  Figure 6: Paired dot plot\n")

dot_data <- combined_df %>%
  dplyr::select(label, interaction, target, prob_High, prob_Low, pathway_name, direction) %>%
  pivot_longer(cols = c(prob_High, prob_Low),
               names_to = "group", values_to = "prob") %>%
  mutate(group = ifelse(group == "prob_High", "High-risk", "Low-risk"))

p_dot <- ggplot(dot_data, aes(x = prob, y = label)) +
  geom_line(aes(group = label), color = "grey70", linewidth = 0.25) +
  geom_point(aes(color = group, shape = group), size = 1.3) +
  scale_color_manual(values = c("High-risk" = "#E64B35", "Low-risk" = "#4DBBD5"),
                     name = "") +
  scale_shape_manual(values = c("High-risk" = 16, "Low-risk" = 17), name = "") +
  theme_nature(base_size = 6) +
  labs(x = "Communication probability",
       y = NULL,
       title = "CAF L-R pair communication probability") +
  theme(legend.position = "top",
        legend.margin = margin(0, 0, -3, 0))

ggsave(file.path(output_dir, "Fig_Nature_Dot_paired_LR.pdf"), p_dot,
       width = SINGLE_COL * 1.6, height = DOUBLE_COL * 0.85,
       units = "in", device = cairo_pdf, family = FONT_FAMILY)

cat("  Dot plot saved.\n")

# ===========================================================
# Save tables
# ===========================================================
write.csv(top_up, file.path(output_dir, "Table_Top20_Upregulated_LR.csv"), row.names = FALSE)
write.csv(top_down, file.path(output_dir, "Table_Top20_Downregulated_LR.csv"), row.names = FALSE)

# ===========================================================
# DETAILED FIGURE DESCRIPTIONS
# ===========================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════════════════════╗\n")
cat("║                         FIGURE DESCRIPTIONS                                        ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                                    ║\n")
cat("║  FIGURE 1: Fig_Nature_Upregulated_comparison.pdf                                   ║\n")
cat("║  ──────────────────────────────────────────────────────────                        ║\n")
cat("║  Title: Top 20 upregulated CAF L-R pairs (high-risk vs low-risk)                   ║\n")
cat("║                                                                                    ║\n")
cat("║  Description: Circle plots showing CAF outgoing signaling for the top 20           ║\n")
cat("║  ligand-receptor pairs that are significantly upregulated in the high-risk          ║\n")
cat("║  group compared to the low-risk group. Each node represents a cell type,           ║\n")
cat("║  positioned on a circle with CAF at the top. Directed edges from CAF to            ║\n")
cat("║  target cell types indicate active signaling, with edge width proportional          ║\n")
cat("║  to aggregate communication probability and edge color indicating the               ║\n")
cat("║  dominant signaling pathway. (a) Low-risk group showing weak or absent              ║\n")
cat("║  communication for these pairs. (b) High-risk group showing strong or               ║\n")
cat("║  gained communication. Both panels use identical edge scaling to enable             ║\n")
cat("║  direct visual comparison. Inactive cell types (no incoming signal) appear          ║\n")
cat("║  dimmed. Node size reflects relative cell count; bold labels indicate               ║\n")
cat("║  active signal recipients.                                                          ║\n")
cat("║                                                                                    ║\n")
cat("║  FIGURE 2: Fig_Nature_Downregulated_comparison.pdf                                 ║\n")
cat("║  ──────────────────────────────────────────────────────────                        ║\n")
cat("║  Title: Top 20 downregulated CAF L-R pairs (high-risk vs low-risk)                 ║\n")
cat("║                                                                                    ║\n")
cat("║  Description: Circle plots showing CAF outgoing signaling for the top 20           ║\n")
cat("║  ligand-receptor pairs that are significantly downregulated (lost or reduced)       ║\n")
cat("║  in the high-risk group compared to the low-risk group. These represent             ║\n")
cat("║  homeostatic or protective signaling programs that are disrupted in                 ║\n")
cat("║  aggressive tumors. (a) Low-risk group showing active communication for             ║\n")
cat("║  these pairs (thick, colored edges). (b) High-risk group showing reduced            ║\n")
cat("║  or absent communication (thin or missing edges). Edge color indicates              ║\n")
cat("║  signaling pathway; both panels share the same maximum scale for direct             ║\n")
cat("║  comparison. The contrast between panels highlights the loss of specific            ║\n")
cat("║  CAF-mediated signaling programs in high-risk tumors.                               ║\n")
cat("║                                                                                    ║\n")
cat("║  FIGURE 3: Fig_Nature_Combined_4panel.pdf                                          ║\n")
cat("║  ──────────────────────────────────────────────────────────                        ║\n")
cat("║  Title: Differential CAF L-R communication between risk groups                     ║\n")
cat("║                                                                                    ║\n")
cat("║  Description: Four-panel composite figure summarizing CAF cell-cell                 ║\n")
cat("║  communication differences. Top row (a,b): top 20 upregulated L-R pairs            ║\n")
cat("║  in low-risk (a, weak/absent) and high-risk (b, strong/gained) groups.             ║\n")
cat("║  Bottom row (c,d): top 20 downregulated L-R pairs in low-risk (c, active)          ║\n")
cat("║  and high-risk (d, reduced/lost) groups. This figure demonstrates that              ║\n")
cat("║  high-risk CAFs gain pro-tumorigenic signaling (VEGF, FGF, CCL, TGFb)              ║\n")
cat("║  while losing homeostatic signaling programs. Pathway labels omitted for            ║\n")
cat("║  clarity; see supplementary tables for detailed L-R pair information.               ║\n")
cat("║                                                                                    ║\n")
cat("║  FIGURE 4: Fig_Nature_Upregulated_withLegend.pdf /                                 ║\n")
cat("║            Fig_Nature_Downregulated_withLegend.pdf                                  ║\n")
cat("║  ──────────────────────────────────────────────────────────                        ║\n")
cat("║  Title: CAF L-R communication with pathway and cell type legend                    ║\n")
cat("║                                                                                    ║\n")
cat("║  Description: Same as Figures 1-2 but with an integrated legend panel              ║\n")
cat("║  showing: (i) signaling pathway color coding for edges, and (ii) cell type         ║\n")
cat("║  color coding for nodes. Pathway labels are displayed along edges in               ║\n")
cat("║  italic text, positioned perpendicular to the edge direction to avoid              ║\n")
cat("║  overlap with edge lines. These figures provide the complete visual key             ║\n")
cat("║  for interpreting the circle plots.                                                 ║\n")
cat("║                                                                                    ║\n")
cat("║  FIGURE 5: Fig_Nature_Barplot_differential_LR.pdf                                  ║\n")
cat("║  ──────────────────────────────────────────────────────────                        ║\n")
cat("║  Title: Differentially active CAF ligand-receptor pairs                            ║\n")
cat("║                                                                                    ║\n")
cat("║  Description: Horizontal barplot showing the change in communication               ║\n")
cat("║  probability (Δ = high-risk − low-risk) for the top 15 upregulated and             ║\n")
cat("║  top 15 downregulated L-R pairs from CAF. Positive bars (right) indicate           ║\n")
cat("║  gained/increased signaling in high-risk; negative bars (left) indicate             ║\n")
cat("║  lost/decreased signaling. Bar fill color encodes the signaling pathway.            ║\n")
cat("║  Each bar label shows the L-R pair name and target cell type                        ║\n")
cat("║  (format: 'Ligand-Receptor → Target'). The vertical line at zero separates         ║\n")
cat("║  gained from lost interactions.                                                     ║\n")
cat("║                                                                                    ║\n")
cat("║  FIGURE 6: Fig_Nature_Dot_paired_LR.pdf                                           ║\n")
cat("║  ──────────────────────────────────────────────────────────                        ║\n")
cat("║  Title: CAF L-R pair communication probability                                     ║\n")
cat("║                                                                                    ║\n")
cat("║  Description: Paired dot-and-line plot comparing communication probability         ║\n")
cat("║  between high-risk (red circles) and low-risk (blue triangles) groups for           ║\n")
cat("║  the top 30 differentially active L-R pairs. Connecting lines highlight             ║\n")
cat("║  the magnitude of change. Pairs where high-risk > low-risk show the red             ║\n")
cat("║  dot to the right; pairs where low-risk > high-risk show the blue triangle          ║\n")
cat("║  to the right. Zero-probability values indicate the interaction was not             ║\n")
cat("║  detected in that group (complete gain or loss of signaling).                       ║\n")
cat("║                                                                                    ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                                    ║\n")
cat("║  VISUAL DESIGN FEATURES (addressing overlap issues):                               ║\n")
cat("║                                                                                    ║\n")
cat("║  1. Labels placed OUTSIDE the circle (dist = 1.35 from center)                     ║\n")
cat("║     with horizontal alignment based on angular position                            ║\n")
cat("║  2. Pathway labels positioned along edges with perpendicular offset                ║\n")
cat("║     and rotation to follow edge direction                                          ║\n")
cat("║  3. Inactive cell types are DIMMED (alpha = 0.4) to reduce visual clutter          ║\n")
cat("║  4. Consistent edge scaling between panels enables direct comparison               ║\n")
cat("║  5. Curved edges (xspline) prevent edge-node overlap                               ║\n")
cat("║  6. Small arrowheads positioned at 85% along edge (not touching node)              ║\n")
cat("║  7. Node size scaled moderately (max radius 0.08) to prevent overlap               ║\n")
cat("║  8. Adequate plot margins (xlim/ylim = ±1.8) for label space                       ║\n")
cat("║  9. Font sizes carefully calibrated:                                               ║\n")
cat("║     - Node labels: 0.45 cex                                                       ║\n")
cat("║     - Pathway labels: 0.3 cex (italic)                                            ║\n")
cat("║     - Pair count annotations: 0.29 cex                                            ║\n")
cat("║     - Titles: 0.7 cex (bold)                                                      ║\n")
cat("║     - Panel labels: 1.0 cex (bold)                                                ║\n")
cat("║  10. Cairo PDF device for proper font embedding and vector output                  ║\n")
cat("║                                                                                    ║\n")
cat("╠══════════════════════════════════════════════════════════════════════════════════════╣\n")
cat("║                                                                                    ║\n")
cat("║  HOW FIGURES REFLECT DIFFERENCES BETWEEN GROUPS:                                   ║\n")
cat("║                                                                                    ║\n")
cat("║  • Same edge scale: Both panels in each figure use identical max_edge_scale,       ║\n")
cat("║    so a thicker edge in one panel vs the other directly indicates stronger          ║\n")
cat("║    communication.                                                                   ║\n")
cat("║                                                                                    ║\n")
cat("║  • Missing edges: When an L-R pair has zero probability in one group,              ║\n")
cat("║    no edge is drawn — the absence of edges in one panel (vs presence               ║\n")
cat("║    in the other) clearly shows gained/lost signaling.                               ║\n")
cat("║                                                                                    ║\n")
cat("║  • Dimmed nodes: Cell types that don't receive any signal in a given               ║\n")
cat("║    panel appear with reduced opacity, immediately highlighting which               ║\n")
cat("║    targets are affected in each group.                                              ║\n")
cat("║                                                                                    ║\n")
cat("║  • Pair count annotations: Below each active target label, '(N pairs)'             ║\n")
cat("║    indicates how many L-R pairs target that cell type, allowing                    ║\n")
cat("║    quantitative comparison.                                                         ║\n")
cat("║                                                                                    ║\n")
cat("║  • Pathway colors: Edge colors encode pathway identity, showing which              ║\n")
cat("║    signaling programs are gained (new colors in high-risk) or lost                 ║\n")
cat("║    (colors present in low-risk but absent in high-risk).                            ║\n")
cat("║                                                                                    ║\n")
cat("╚══════════════════════════════════════════════════════════════════════════════════════╝\n")

# ===========================================================
# Print L-R pair names for reference
# ===========================================================
cat("\n\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  TOP 20 UPREGULATED L-R PAIRS (shown in Figures 1, 3a-b, 4):\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
for (i in 1:nrow(top_up)) {
  cat(sprintf("  %2d. %-40s \u2192 %-22s [%s] \u0394=%.5f\n",
              i, top_up$interaction[i], top_up$target[i],
              top_up$pathway_name[i], top_up$diff[i]))
}

cat("\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
cat("  TOP 20 DOWNREGULATED L-R PAIRS (shown in Figures 2, 3c-d, 4):\n")
cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
for (i in 1:nrow(top_down)) {
  cat(sprintf("  %2d. %-40s \u2192 %-22s [%s] \u0394=%.5f\n",
              i, top_down$interaction[i], top_down$target[i],
              top_down$pathway_name[i], top_down$diff[i]))
}

cat("\n━━━ Complete ━━━\n")
cat(sprintf("Output: %s\n", output_dir))
cat("All figures are vector PDF, Nature-compatible (183mm double-column).\n")
