"""
Pathway enrichment for the High-risk vs Low-risk DESeq2 comparison.

Primary analysis: GSEA (pre-ranked) on the full DESeq2 Wald-statistic-
ranked gene list against MSigDB Hallmark gene sets -- avoids depending on
an arbitrary significance cutoff and is well suited to a modest-effect-
size, modest-n RNA-seq comparison like this one. Also run against
KEGG_2021_Human and GO_Biological_Process_2023 for broader pathway
context.

Secondary analysis: over-representation analysis (ORA, Enrichr) on the
62 significant DEGs (padj<0.05, |log2FC|>=1) against the same three
libraries, as a threshold-based cross-check of the GSEA result.
"""
import gseapy as gp
import pandas as pd
from pathlib import Path

OUT_DIR = Path('XXXyour path/Dataset/Dataset004-NACI/NSCLC-CT-RNAseq/RNA_analysis')
LIBRARIES = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2023']

print("=" * 70)
print("GSEA PRERANK (full ranked gene list, DESeq2 Wald statistic)")
print("=" * 70)

rnk = pd.read_csv(OUT_DIR / 'DESeq2_ranked_for_gsea.rnk', sep='\t', header=None, names=['gene', 'rank'])
print(f"  {len(rnk)} genes in ranked list")

gsea_results = {}
for lib in LIBRARIES:
    print(f"\n  --- {lib} ---")
    try:
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=lib,
            threads=8,
            min_size=10,
            max_size=500,
            permutation_num=1000,
            outdir=None,
            seed=42,
            verbose=False,
        )
        res = pre_res.res2d.copy()
        res['Library'] = lib
        gsea_results[lib] = res
        n_sig = (res['FDR q-val'].astype(float) < 0.25).sum()  # GSEA's own conventional FDR cutoff
        print(f"    {len(res)} gene sets tested, {n_sig} with FDR q<0.25 (GSEA convention)")
        out_path = OUT_DIR / f'GSEA_prerank_{lib}.csv'
        res.to_csv(out_path, index=False)
        print(f"    Saved to {out_path}")
    except Exception as e:
        print(f"    FAILED: {e}")

if gsea_results:
    all_gsea = pd.concat(gsea_results.values(), ignore_index=True)
    all_gsea.to_csv(OUT_DIR / 'GSEA_prerank_all_libraries.csv', index=False)
    top20 = all_gsea.copy()
    top20['FDR q-val'] = top20['FDR q-val'].astype(float)
    top20['NES'] = top20['NES'].astype(float)
    top20 = top20.sort_values('FDR q-val').head(20)
    print("\n  Top 20 pathways overall (by FDR q-value):")
    pd.set_option('display.width', 160)
    print(top20[['Library', 'Term', 'NES', 'FDR q-val']].to_string(index=False))

print("\n" + "=" * 70)
print("SECONDARY: OVER-REPRESENTATION ANALYSIS (Enrichr, significant DEGs)")
print("=" * 70)

sig = pd.read_csv(OUT_DIR / 'DESeq2_results_significant.csv')
up_genes = sig.loc[sig['log2FoldChange'] > 0, 'Symbol'].dropna().tolist()
down_genes = sig.loc[sig['log2FoldChange'] < 0, 'Symbol'].dropna().tolist()
print(f"  Up in High-risk: {len(up_genes)} genes; Up in Low-risk: {len(down_genes)} genes")

ora_results = {}
for direction, genes in [('up_in_high_risk', up_genes), ('up_in_low_risk', down_genes)]:
    if len(genes) < 3:
        print(f"  Skipping {direction}: too few genes ({len(genes)})")
        continue
    for lib in LIBRARIES:
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib, outdir=None)
            res = enr.results.copy()
            res['Library'] = lib
            res['Direction'] = direction
            ora_results[(direction, lib)] = res
        except Exception as e:
            print(f"  ORA FAILED for {direction} / {lib}: {e}")

if ora_results:
    all_ora = pd.concat(ora_results.values(), ignore_index=True)
    all_ora.to_csv(OUT_DIR / 'ORA_enrichr_significant_DEGs.csv', index=False)
    print(f"\n  Saved combined ORA results ({len(all_ora)} rows) to "
          f"{OUT_DIR / 'ORA_enrichr_significant_DEGs.csv'}")
    sig_ora = all_ora[all_ora['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')
    print(f"  {len(sig_ora)} terms with Adjusted P-value < 0.05:")
    if len(sig_ora):
        print(sig_ora[['Direction', 'Library', 'Term', 'Adjusted P-value', 'Genes']].head(20).to_string(index=False))

print("\nDONE")
