"""
Nature-style figures for the RNA-seq High-risk vs Low-risk differential
expression / pathway enrichment analysis (NSCLC-CT-RNAseq / GSE103584).

Reuses the sizing and typography conventions established in
PlotFigures.py (SINGLE_COL/DOUBLE_COL/SUBPANEL_COL, 6-8pt sans-serif
fonts) and the project's risk-group color scheme (High-risk = red,
matching "predicted Non-pCR" elsewhere in the pipeline; Low-risk = blue,
matching "predicted pCR").

Outputs (RNA_analysis/figures/):
  - Fig1_volcano.png/.pdf
  - Fig2_PCA.png/.pdf
  - Fig3_heatmap_topDEGs.png/.pdf
  - Fig4_GSEA_hallmark.png/.pdf
  - Fig5_ORA_barplot.png/.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path('XXXyour path/Dataset/Dataset004-NACI/NSCLC-CT-RNAseq/RNA_analysis')
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_COL = 3.5
DOUBLE_COL = 7.2
SUBPANEL_COL = DOUBLE_COL / 3

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'legend.title_fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'legend.frameon': True,
})

COLOR_HIGH = '#E74C3C'   # High-risk (predicted Non-pCR)
COLOR_LOW = '#3498DB'    # Low-risk (predicted pCR)
COLOR_NS = '#B0B0B0'     # not significant

PADJ_CUT = 0.05
LFC_CUT = 1.0


def savefig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.png')
    fig.savefig(FIG_DIR / f'{name}.pdf')
    plt.close(fig)
    print(f'  Saved {name}.png / .pdf')


# ─────────────────────────────────────────────────────────────────────
# Fig 1: Volcano plot
# ─────────────────────────────────────────────────────────────────────
def plot_volcano():
    df = pd.read_csv(OUT_DIR / 'DESeq2_results_full.csv')
    df = df.dropna(subset=['padj', 'log2FoldChange'])
    df['neglog10padj'] = -np.log10(df['padj'].clip(lower=1e-300))

    up = (df['padj'] < PADJ_CUT) & (df['log2FoldChange'] >= LFC_CUT)
    down = (df['padj'] < PADJ_CUT) & (df['log2FoldChange'] <= -LFC_CUT)
    ns = ~(up | down)

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.55, SINGLE_COL * 1.35))
    ax.scatter(df.loc[ns, 'log2FoldChange'], df.loc[ns, 'neglog10padj'],
               s=4, c=COLOR_NS, alpha=0.35, linewidths=0, rasterized=True)
    ax.scatter(df.loc[up, 'log2FoldChange'], df.loc[up, 'neglog10padj'],
               s=6, c=COLOR_HIGH, alpha=0.75, linewidths=0, label=f'Up in High-risk (n={up.sum()})')
    ax.scatter(df.loc[down, 'log2FoldChange'], df.loc[down, 'neglog10padj'],
               s=6, c=COLOR_LOW, alpha=0.75, linewidths=0, label=f'Up in Low-risk (n={down.sum()})')

    ax.axvline(LFC_CUT, color='k', lw=0.5, ls='--', alpha=0.6)
    ax.axvline(-LFC_CUT, color='k', lw=0.5, ls='--', alpha=0.6)
    ax.axhline(-np.log10(PADJ_CUT), color='k', lw=0.5, ls='--', alpha=0.6)

    top_labels = pd.concat([
        df[up].nsmallest(8, 'padj'),
        df[down].nsmallest(8, 'padj'),
    ])
    texts = []
    for _, row in top_labels.iterrows():
        texts.append(ax.text(row['log2FoldChange'], row['neglog10padj'], row['Symbol'],
                              fontsize=5.5, fontstyle='italic'))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='grey', lw=0.4))

    ax.set_xlabel('log$_2$ fold change (High-risk vs Low-risk)')
    ax.set_ylabel('$-$log$_{10}$(adjusted P)')
    ax.set_title('Differential expression: High-risk vs Low-risk', fontsize=8)
    ax.legend(loc='lower right', markerscale=2, handletextpad=0.3, borderpad=0.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    savefig(fig, 'Fig1_volcano')


# ─────────────────────────────────────────────────────────────────────
# Fig 2: PCA of VST-normalized expression
# ─────────────────────────────────────────────────────────────────────
def plot_pca():
    vst = pd.read_csv(OUT_DIR / 'normalized_counts_vst.csv', index_col=0)
    meta = pd.read_csv(OUT_DIR / 'sample_metadata_with_risk.csv', index_col=0)
    meta = meta.loc[vst.index]

    # Top 2000 most variable genes, as is standard for expression PCA
    top_var_genes = vst.var(axis=0).nlargest(2000).index
    X = vst[top_var_genes].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.35, SINGLE_COL * 1.2))
    for grp, color, marker in [('Low-risk', COLOR_LOW, 'o'), ('High-risk', COLOR_HIGH, '^')]:
        m = meta['risk_group_short'].values == grp
        ax.scatter(pcs[m, 0], pcs[m, 1], s=16, c=color, marker=marker,
                   alpha=0.75, linewidths=0.3, edgecolors='white', label=f'{grp} (n={m.sum()})')
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)')
    ax.set_title('PCA of VST-normalized expression\n(top 2,000 variable genes)', fontsize=7.5)
    ax.legend(loc='best', markerscale=1.3, handletextpad=0.3, borderpad=0.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    savefig(fig, 'Fig2_PCA')


# ─────────────────────────────────────────────────────────────────────
# Fig 3: Heatmap of top significant DEGs
# ─────────────────────────────────────────────────────────────────────
def plot_heatmap(n_top=40):
    vst = pd.read_csv(OUT_DIR / 'normalized_counts_vst.csv', index_col=0)
    meta = pd.read_csv(OUT_DIR / 'sample_metadata_with_risk.csv', index_col=0)
    sig = pd.read_csv(OUT_DIR / 'DESeq2_results_significant.csv')
    meta = meta.loc[vst.index]

    sig_top = sig.reindex(sig['padj'].sort_values().index).head(n_top)
    gene_ids = sig_top['GeneID'].astype(str).tolist()
    gene_ids = [g for g in gene_ids if g in vst.columns]
    symbols = sig_top.set_index(sig_top['GeneID'].astype(str)).loc[gene_ids, 'Symbol']

    mat = vst[gene_ids].T
    mat.index = symbols.values
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)

    # order samples: Low-risk block then High-risk block, each internally
    # ordered by fused_probability so the transition across the risk
    # threshold is visible left-to-right
    order = meta.sort_values(['risk_group_short', 'fused_probability']).index
    z = z[order]
    risk_ordered = meta.loc[order, 'risk_group_short']

    fig = plt.figure(figsize=(DOUBLE_COL, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.06, 1], width_ratios=[1, 0.03],
                           hspace=0.02, wspace=0.02, top=0.86, bottom=0.10)
    ax_annot = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    from matplotlib.colors import to_rgb
    colors = np.array([to_rgb(COLOR_HIGH) if r == 'High-risk' else to_rgb(COLOR_LOW)
                        for r in risk_ordered.values])
    ax_annot.imshow(colors.reshape(1, -1, 3), aspect='auto',
                     extent=[0, len(order), 0, 1])
    ax_annot.set_xticks([])
    ax_annot.set_yticks([])
    for spine in ax_annot.spines.values():
        spine.set_visible(False)

    cmap = LinearSegmentedColormap.from_list('bwr_nature', ['#2166AC', '#F7F7F7', '#B2182B'])
    sns.heatmap(z, cmap=cmap, center=0, vmin=-2.5, vmax=2.5, ax=ax_hm,
                cbar_ax=ax_cbar, xticklabels=False, yticklabels=True,
                cbar_kws={'label': 'Row Z-score'})
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), fontsize=5.2, fontstyle='italic')
    ax_hm.set_xlabel(f'Patients (n={len(order)}), ordered by risk group and fused probability')
    ax_hm.set_ylabel('')

    handles = [mpatches.Patch(color=COLOR_LOW, label='Low-risk (predicted pCR)'),
               mpatches.Patch(color=COLOR_HIGH, label='High-risk (predicted Non-pCR)')]
    ax_annot.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.15),
                     ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0)

    fig.suptitle(f'Top {len(gene_ids)} differentially expressed genes\n'
                 f'(ranked by adjusted P; padj<{PADJ_CUT}, |log$_2$FC|$\\geq${LFC_CUT:.0f})',
                 fontsize=7.5, y=0.995)
    fig.savefig(FIG_DIR / 'Fig3_heatmap_topDEGs.png')
    fig.savefig(FIG_DIR / 'Fig3_heatmap_topDEGs.pdf')
    plt.close(fig)
    print('  Saved Fig3_heatmap_topDEGs.png / .pdf')


# ─────────────────────────────────────────────────────────────────────
# Fig 4: GSEA Hallmark dot plot
# ─────────────────────────────────────────────────────────────────────
def plot_gsea_hallmark(n_top=20):
    df = pd.read_csv(OUT_DIR / 'GSEA_prerank_MSigDB_Hallmark_2020.csv')
    df['NES'] = df['NES'].astype(float)
    df['FDR q-val'] = df['FDR q-val'].astype(float).clip(lower=1e-4)
    df = df.sort_values('FDR q-val').head(n_top)
    df = df.sort_values('NES')

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, SINGLE_COL * 1.4))
    colors = [COLOR_HIGH if n > 0 else COLOR_LOW for n in df['NES']]
    sizes = -np.log10(df['FDR q-val']) * 18 + 8
    ax.scatter(df['NES'], range(len(df)), s=sizes, c=colors, alpha=0.85,
               linewidths=0.4, edgecolors='k')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Term'], fontsize=6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('Normalized enrichment score (NES)\n(High-risk vs Low-risk)')
    ax.set_title(f'Top {n_top} Hallmark gene sets\n(GSEA preranked, by FDR q-value)', fontsize=7.5)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    handles = [
        plt.scatter([], [], s=60, c=COLOR_LOW, edgecolors='k', linewidths=0.4,
                    label='Up in Low-risk (NES<0)'),
        plt.scatter([], [], s=60, c=COLOR_HIGH, edgecolors='k', linewidths=0.4,
                    label='Up in High-risk (NES>0)'),
    ]
    size_ref = [
        plt.scatter([], [], s=-np.log10(0.05) * 18 + 8, c='grey', edgecolors='k', linewidths=0.4,
                    label='FDR q=0.05'),
        plt.scatter([], [], s=-np.log10(0.001) * 18 + 8, c='grey', edgecolors='k', linewidths=0.4,
                    label='FDR q=0.001'),
    ]
    leg1 = ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, -0.32),
                      frameon=False, handletextpad=0.2, fontsize=5.8)
    ax.add_artist(leg1)
    ax.legend(handles=size_ref, loc='lower left', bbox_to_anchor=(0.0, -0.32),
              frameon=False, handletextpad=0.2, fontsize=5.8, title='Dot size',
              title_fontsize=5.8)
    fig.tight_layout()
    savefig(fig, 'Fig4_GSEA_hallmark')


# ─────────────────────────────────────────────────────────────────────
# Fig 5: ORA barplot (Enrichr, significant DEGs), KEGG neurodegenerative
# disease artifact terms excluded (see manuscript text for rationale)
# ─────────────────────────────────────────────────────────────────────
ARTIFACT_TERMS = [
    'Huntington disease', 'Parkinson disease', 'Alzheimer disease',
    'Prion disease', 'Amyotrophic lateral sclerosis', 'Diabetic cardiomyopathy',
    'Coronavirus disease - COVID-19', 'Non-alcoholic fatty liver disease',
    'Pathways of neurodegeneration - multiple diseases',
]


def plot_ora_barplot(n_top=15):
    df = pd.read_csv(OUT_DIR / 'ORA_enrichr_significant_DEGs.csv')
    df = df[df['Adjusted P-value'] < 0.05].copy()
    df = df[~df['Term'].isin(ARTIFACT_TERMS)]
    df['neglog10padj'] = -np.log10(df['Adjusted P-value'].clip(lower=1e-30))
    df['label'] = df['Term'] + '  (' + df['Library'].str.replace('_2021_Human', '')\
        .str.replace('_2023', '').str.replace('MSigDB_', '') + ')'

    up = df[df['Direction'] == 'up_in_high_risk'].nlargest(n_top, 'neglog10padj').sort_values('neglog10padj')
    down = df[df['Direction'] == 'up_in_low_risk'].nlargest(n_top, 'neglog10padj').sort_values('neglog10padj')

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 1.5))
    fig.subplots_adjust(wspace=0.9)
    for ax, sub, color, title, side in [
        (axes[0], down, COLOR_LOW, 'Up in Low-risk', 'left'),
        (axes[1], up, COLOR_HIGH, 'Up in High-risk', 'right'),
    ]:
        if len(sub) == 0:
            ax.axis('off')
            ax.set_title(f'{title}\n(no significant terms)', fontsize=7)
            continue
        ax.barh(range(len(sub)), sub['neglog10padj'], color=color, alpha=0.85, height=0.65)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub['label'], fontsize=5.6)
        if side == 'right':
            ax.yaxis.tick_right()
        ax.set_xlabel('$-$log$_{10}$(adjusted P)')
        ax.set_title(title, fontsize=7.5)
        sns.despine(ax=ax, left=(side == 'right'), right=(side == 'left'))

    fig.suptitle('Over-representation analysis of significant DEGs\n'
                 '(Enrichr; KEGG neurodegenerative-disease artifact terms excluded — see Methods)',
                 fontsize=7, y=1.06)
    fig.tight_layout()
    savefig(fig, 'Fig5_ORA_barplot')


if __name__ == '__main__':
    print('Generating Fig1_volcano...'); plot_volcano()
    print('Generating Fig2_PCA...'); plot_pca()
    print('Generating Fig3_heatmap_topDEGs...'); plot_heatmap()
    print('Generating Fig4_GSEA_hallmark...'); plot_gsea_hallmark()
    print('Generating Fig5_ORA_barplot...'); plot_ora_barplot()
    print('\nAll figures saved to', FIG_DIR)
