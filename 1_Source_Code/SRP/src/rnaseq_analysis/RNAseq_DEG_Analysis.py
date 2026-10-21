"""
Differential gene expression analysis: SRP-model High-risk (predicted
Non-pCR) vs Low-risk (predicted pCR) groups in the NSCLC-CT-RNAseq cohort
(GSE103584, the public NSCLC-Radiogenomics RNA-seq dataset paired with
the CT images used for feature extraction and SRP risk stratification).

Design: standard DESeq2 negative-binomial GLM (via pyDESeq2) on raw
RNA-seq counts, condition = risk_group (Low-risk = reference level, so
positive log2FC = higher expression in High-risk/predicted-Non-pCR
patients). No covariate adjustment: histology, gender and histopathologic
grade are balanced between risk groups (chi-square P = 0.97, 0.60, 0.71
respectively; see sample_metadata_with_risk.csv), so a simple
~ risk_group design is appropriate and preserves power at n = 122.

Outputs (RNA_analysis/):
  - sample_metadata_with_risk.csv
  - DESeq2_results_full.csv          (all tested genes)
  - DESeq2_results_significant.csv   (padj<0.05 & |log2FC|>=1)
  - DESeq2_ranked_for_gsea.rnk       (gene symbol + rank metric, for GSEA prerank)
  - normalized_counts_vst.csv        (variance-stabilized, for PCA/heatmap)
"""
import sys
import gzip
import numpy as np
import pandas as pd
from pathlib import Path

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

RNA_DIR = Path('XXXyour path/Dataset/Dataset004-NACI/NSCLC-CT-RNAseq/RNA_sequencing')
OUT_DIR = Path('XXXyour path/Dataset/Dataset004-NACI/NSCLC-CT-RNAseq/RNA_analysis')
RISK_CSV = Path('XXXyour path/Dataset/Dataset004-NACI/NSCLC-CT-RNAseq/model_results/NSCLC_risk_predictions.csv')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP 1: BUILD SAMPLE METADATA (R01 case_id <-> GSM, risk group)")
print("=" * 70)

with gzip.open(RNA_DIR / 'GSE103584_series_matrix.txt.gz', 'rt') as f:
    lines = f.readlines()


def parse_row(prefix):
    for l in lines:
        if l.startswith(prefix):
            parts = l.strip().split('\t')
            return [p.strip('"') for p in parts[1:]]
    return None


titles = parse_row('!Sample_title')
gsm = parse_row('!Sample_geo_accession')
chars_rows = [l for l in lines if l.startswith('!Sample_characteristics_ch1')]

meta = pd.DataFrame({'case_id': titles, 'GSM': gsm})
for row in chars_rows:
    parts = [p.strip('"') for p in row.strip().split('\t')[1:]]
    label = parts[0].split(':')[0].strip()
    vals = [p.split(':', 1)[1].strip() if ':' in p else p for p in parts]
    meta[label] = vals

risk = pd.read_csv(RISK_CSV)
risk['risk_group_short'] = risk['risk_group'].map({
    'High-risk (predicted Non-pCR)': 'High-risk',
    'Low-risk (predicted pCR)': 'Low-risk',
})

sample_meta = pd.merge(meta, risk[['case_id', 'risk_group_short', 'fused_probability',
                                    'spatial_probability', 'radiomics_probability']],
                        on='case_id', how='inner')
sample_meta = sample_meta[sample_meta['risk_group_short'].isin(['High-risk', 'Low-risk'])].reset_index(drop=True)
sample_meta = sample_meta.set_index('case_id')
print(f"  {len(sample_meta)} samples with both RNA-seq and a valid SRP risk group")
print(sample_meta['risk_group_short'].value_counts().to_string())
sample_meta.to_csv(OUT_DIR / 'sample_metadata_with_risk.csv')

print("\n" + "=" * 70)
print("STEP 2: LOAD RAW COUNTS, SUBSET TO MATCHED SAMPLES")
print("=" * 70)

counts_raw = pd.read_csv(RNA_DIR / 'GSE103584_raw_counts_GRCh38.p13_NCBI.tsv.gz', sep='\t', index_col=0)
print(f"  Raw counts matrix: {counts_raw.shape[0]} genes x {counts_raw.shape[1]} samples (GSM)")

gsm_to_case = dict(zip(sample_meta['GSM'], sample_meta.index))
counts_sub = counts_raw[[c for c in counts_raw.columns if c in gsm_to_case]].copy()
counts_sub = counts_sub.rename(columns=gsm_to_case)
counts_sub = counts_sub[sample_meta.index]  # order to match metadata
print(f"  Subset to {counts_sub.shape[1]} matched samples, {counts_sub.shape[0]} genes")

# Low-count filter (standard DESeq2 practice): drop genes with negligible total signal
keep = counts_sub.sum(axis=1) >= 10
counts_filt = counts_sub.loc[keep]
print(f"  After low-count filter (sum >= 10 across samples): {counts_filt.shape[0]} genes retained "
      f"({counts_sub.shape[0] - counts_filt.shape[0]} dropped)")

annot = pd.read_csv(RNA_DIR / 'Human.GRCh38.p13.annot.tsv.gz', sep='\t', usecols=['GeneID', 'Symbol', 'Description'])
annot = annot.set_index('GeneID')
gene_symbol = annot['Symbol'].to_dict()
gene_desc = annot['Description'].to_dict()

print("\n" + "=" * 70)
print("STEP 3: RUN DESeq2 (pyDESeq2), risk_group: High-risk vs Low-risk")
print("=" * 70)

counts_for_dds = counts_filt.T  # pydeseq2 wants samples x genes
condition = sample_meta['risk_group_short']
condition = pd.Categorical(condition, categories=['Low-risk', 'High-risk'])  # Low-risk = reference
meta_for_dds = pd.DataFrame({'riskgroup': condition}, index=sample_meta.index)  # no underscore: pydeseq2 mangles factor names containing '_'

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts_for_dds,
    metadata=meta_for_dds,
    design_factors='riskgroup',
    ref_level=['riskgroup', 'Low-risk'],
    refit_cooks=True,
    inference=inference,
)
dds.deseq2()

stat_res = DeseqStats(dds, contrast=['riskgroup', 'High-risk', 'Low-risk'], inference=inference)
stat_res.summary()
res_df = stat_res.results_df.copy()
res_df.index.name = 'GeneID'
res_df = res_df.reset_index()
res_df['GeneID'] = res_df['GeneID'].astype(int)
res_df['Symbol'] = res_df['GeneID'].map(gene_symbol)
res_df['Description'] = res_df['GeneID'].map(gene_desc)
res_df = res_df.sort_values('padj')

full_path = OUT_DIR / 'DESeq2_results_full.csv'
res_df.to_csv(full_path, index=False)
print(f"  Saved full results ({len(res_df)} genes) to {full_path}")

sig = res_df[(res_df['padj'] < 0.05) & (res_df['log2FoldChange'].abs() >= 1)].copy()
sig_path = OUT_DIR / 'DESeq2_results_significant.csv'
sig.to_csv(sig_path, index=False)
n_up = (sig['log2FoldChange'] > 0).sum()
n_down = (sig['log2FoldChange'] < 0).sum()
print(f"  Significant DEGs (padj<0.05 & |log2FC|>=1): {len(sig)}  "
      f"(up in High-risk: {n_up}, down in High-risk / up in Low-risk: {n_down})")
print(f"  Saved to {sig_path}")

# Secondary, more lenient threshold often reported alongside the primary one
sig_lenient = res_df[res_df['padj'] < 0.05]
print(f"  (padj<0.05 alone, any fold change: {len(sig_lenient)} genes)")

print("\n" + "=" * 70)
print("STEP 4: VARIANCE-STABILIZED COUNTS (for PCA / heatmap)")
print("=" * 70)

dds.vst()
vst_counts = pd.DataFrame(dds.layers['vst_counts'], index=sample_meta.index, columns=counts_filt.index)
vst_path = OUT_DIR / 'normalized_counts_vst.csv'
vst_counts.to_csv(vst_path)
print(f"  Saved VST-normalized counts ({vst_counts.shape}) to {vst_path}")

print("\n" + "=" * 70)
print("STEP 5: RANKED GENE LIST FOR GSEA PRERANK")
print("=" * 70)

rnk = res_df.dropna(subset=['stat', 'Symbol']).copy()
rnk = rnk.drop_duplicates(subset='Symbol', keep='first')  # keep most significant per symbol
rnk = rnk[['Symbol', 'stat']].sort_values('stat', ascending=False)
rnk_path = OUT_DIR / 'DESeq2_ranked_for_gsea.rnk'
rnk.to_csv(rnk_path, sep='\t', index=False, header=False)
print(f"  Saved {len(rnk)}-gene ranked list to {rnk_path}")

print("\nDONE")
