"""
Binormal (smoothed) AUC with bootstrap 95% CI, for all three models across
all six labeled cohorts (Dataset002_ECResponse: Test/TJH/SPCH;
Dataset004-NACI: Test/SPCH/FUSCC).

Point estimate: the closed-form binormal AUC = Phi(a / sqrt(1+b^2)), fit
on logit-transformed raw predicted probabilities, split by true label --
same construction used in BinormalROCFit.py / BinormalROCFit_Dataset002.py
for the smoothed ROC/DCA/calibration/radar figures.

95% CI: percentile bootstrap (2,000 resamples, patients resampled with
replacement, seed=42) -- refit the binormal model on each resample and
recompute the closed-form AUC, then take the 2.5th/97.5th percentiles of
the resulting distribution. This is the same bootstrap convention used
elsewhere in this project for AUC-related confidence intervals
(pairwise ΔAUC comparison, manuscript/Results_Pairwise_AUC_Comparison.md).

Resamples that happen to contain only one class are redrawn (needed to
fit per-class Gaussians).

Output: manuscript/Results_Smoothed_AUC_with_CI.csv and a printed/markdown
summary table.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

EPS = 1e-6
N_BOOT = 2000
SEED = 42

MODEL_LABELS = {
    'spatial_probability': 'Spatial Feature-only Model',
    'radiomics_probability': 'Conventional Radiomics Model',
    'fused_probability': 'SRP Model (Fusion)',
}

DATASET002_ROOT = Path('XXXyour path/Dataset/Dataset002_ECResponse')
DATASET004_ROOT = Path('XXXyour path/Dataset/Dataset004-NACI')

COHORTS = [
    ('Dataset002_ECResponse', 'Test', DATASET002_ROOT / 'model_results' / 'test_predictions_results.csv'),
    ('Dataset002_ECResponse', 'TJH', DATASET002_ROOT / 'TJH' / 'model_results' / 'TJH_predictions_results.csv'),
    ('Dataset002_ECResponse', 'SPCH', DATASET002_ROOT / 'SPCH' / 'model_results' / 'SPCH_predictions_results.csv'),
    ('Dataset004-NACI', 'Test', DATASET004_ROOT / 'model_results' / 'test_predictions_results_new_radiomics.csv'),
    ('Dataset004-NACI', 'SPCH', DATASET004_ROOT / 'SPCH' / 'model_results' / 'SPCH_predictions_results_new_radiomics.csv'),
    ('Dataset004-NACI', 'FUSCC', DATASET004_ROOT / 'FUSCC' / 'model_results' / 'FUSCC_predictions_results_new_radiomics.csv'),
]
PROB_COLS = ['spatial_probability', 'radiomics_probability', 'fused_probability']


def logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def binormal_auc(y_true, y_proba):
    """Closed-form binormal AUC = Phi(a / sqrt(1+b^2))."""
    z = logit(y_proba)
    z0 = z[y_true == 0]
    z1 = z[y_true == 1]
    mu0, sigma0 = np.mean(z0), max(np.std(z0, ddof=1), EPS)
    mu1, sigma1 = np.mean(z1), max(np.std(z1, ddof=1), EPS)
    a = (mu1 - mu0) / sigma1
    b = sigma0 / sigma1
    return stats.norm.cdf(a / np.sqrt(1 + b ** 2))


def bootstrap_binormal_auc_ci(y_true, y_proba, n_boot=N_BOOT, seed=SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []
    attempts = 0
    while len(boot_aucs) < n_boot and attempts < n_boot * 20:
        attempts += 1
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], y_proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(binormal_auc(yb, pb))
    boot_aucs = np.array(boot_aucs)
    alpha = 1 - ci
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, len(boot_aucs)


print("=" * 90)
print("BINORMAL (SMOOTHED) AUC WITH BOOTSTRAP 95% CI")
print("=" * 90)

rows = []
for dataset, cohort, path in COHORTS:
    df = pd.read_csv(path)
    y = df['true_label'].values
    n = len(df)
    n_events = int(y.sum())
    print(f"\n--- {dataset} / {cohort} (n={n}, events={n_events}) ---")
    for col in PROB_COLS:
        y_proba = df[col].values
        point = binormal_auc(y, y_proba)
        lo, hi, n_valid = bootstrap_binormal_auc_ci(y, y_proba)
        print(f"  {MODEL_LABELS[col]:32s}: AUC={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  "
              f"(n_boot valid={n_valid})")
        rows.append({
            'Dataset': dataset,
            'Cohort': cohort,
            'n': n,
            'n_events': n_events,
            'Model': MODEL_LABELS[col],
            'AUC_binormal': round(point, 4),
            'CI_lower': round(lo, 4),
            'CI_upper': round(hi, 4),
        })

out_df = pd.DataFrame(rows)
out_path = Path('XXXyour path/manuscript/Results_Smoothed_AUC_with_CI.csv')
out_df.to_csv(out_path, index=False)
print(f"\nSaved table to {out_path}")

print("\n" + "=" * 90)
print("MARKDOWN TABLE")
print("=" * 90)
print("| Dataset | Cohort (n) | Model | AUC (binormal) | 95% CI |")
print("|---|---|---|---|---|")
for _, r in out_df.iterrows():
    print(f"| {r['Dataset']} | {r['Cohort']} ({int(r['n'])}) | {r['Model']} | "
          f"{r['AUC_binormal']:.3f} | [{r['CI_lower']:.3f}, {r['CI_upper']:.3f}] |")

print("\nDONE")
