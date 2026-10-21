"""
Binormal (parametric) ROC fit for the three models (Spatial, Radiomics,
Fusion) on Dataset002_ECResponse's Test / TJH / SPCH cohorts -- the three
that have ground-truth labels (scRNA-seq-CT has no true_label column, so a
binormal fit -- which needs per-class score distributions -- doesn't apply
to it).

This is the dataset that actually motivated the smoothing exercise: SPCH's
spatial model here has only ~40 distinct predicted probabilities across
217 patients (XGBClassifier(n_estimators=83, learning_rate=0.007) on 7
features -- much coarser than the Dataset004-NACI models), so unlike that
pipeline, the raw empirical ROC/DCA curves here are genuinely jagged from
large ties, not just small-n staircasing.

Method: identical to BinormalROCFit.py (Dataset004-NACI version) -- logit-
transform each model's probabilities, fit a Gaussian per true-label class,
use the two fitted Gaussians as a generative model to compute a smooth
per-patient posterior probability via Bayes' rule.

Outputs, per cohort:
  - *_predictions_results_binormal_fitted.csv
  - roc_curves_binormal_smooth.png/pdf -- literal closed-form binormal curve
  - roc_curves.png/pdf, decision_curve_analysis.png/pdf,
    calibration_curves.png/pdf, combined_radar_chart.png/pdf --
    regenerated from the fitted per-patient scores via the existing
    MedicalImagingVisualizer
Saved to a sibling `visualizations_binormal_fitted` folder per cohort so
the original raw-score figures are untouched. Note the calibration curve
here answers a different question than the original -- it shows whether
the fitted binormal posterior probabilities line up with observed pCR
frequency, not whether the underlying classifier's raw output is
calibrated. The combined radar chart (ACC/SEN/SPE/PPV/NPV/F1 at the
Youden-optimal threshold) is expected to look nearly identical to the
raw-score version: the binormal posterior is a monotonic function of the
raw score (via logit), so ties and rank order -- and therefore which
patients fall on each side of the optimal threshold -- are preserved.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PlotFigures import MedicalImagingVisualizer, SUBPANEL_COL

DATA_ROOT = Path('XXXyour path/Dataset/Dataset002_ECResponse')
EPS = 1e-6
MODEL_STYLE = {
    'spatial_probability':   ('Spatial Feature-only Model',   '#E74C3C', '-'),
    'radiomics_probability': ('Conventional Radiomics Model', '#3498DB', '-'),
    'fused_probability':     ('SRP Model',                    '#009E73', '-'),
}

DATASETS = {
    'Test': DATA_ROOT / 'model_results' / 'test_predictions_results.csv',
    'TJH':  DATA_ROOT / 'TJH' / 'model_results' / 'TJH_predictions_results.csv',
    'SPCH': DATA_ROOT / 'SPCH' / 'model_results' / 'SPCH_predictions_results.csv',
}
PROB_COLS = ['spatial_probability', 'radiomics_probability', 'fused_probability']


def logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_binormal(y_true, y_proba):
    z = logit(y_proba)
    z0 = z[y_true == 0]  # pCR
    z1 = z[y_true == 1]  # Non-pCR

    mu0, sigma0 = np.mean(z0), max(np.std(z0, ddof=1), EPS)
    mu1, sigma1 = np.mean(z1), max(np.std(z1, ddof=1), EPS)

    prior1 = np.mean(y_true == 1)
    prior0 = 1 - prior1

    a = (mu1 - mu0) / sigma1
    b = sigma0 / sigma1
    auc_binormal = stats.norm.cdf(a / np.sqrt(1 + b ** 2))

    def posterior(raw_proba):
        zz = logit(raw_proba)
        f1 = stats.norm.pdf(zz, mu1, sigma1)
        f0 = stats.norm.pdf(zz, mu0, sigma0)
        num = prior1 * f1
        den = num + prior0 * f0
        return np.where(den > 0, num / np.maximum(den, EPS), 0.5)

    return {
        'mu0': mu0, 'sigma0': sigma0, 'mu1': mu1, 'sigma1': sigma1,
        'a': a, 'b': b, 'auc_binormal': auc_binormal,
        'posterior_fn': posterior,
    }


def plot_smooth_binormal_roc(fits_by_col, out_dir, dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(SUBPANEL_COL, SUBPANEL_COL), dpi=dpi)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Chance')

    fpr_grid = np.linspace(1e-4, 1 - 1e-4, 400)
    z = stats.norm.ppf(fpr_grid)
    for col, fit in fits_by_col.items():
        label, color, ls = MODEL_STYLE[col]
        tpr_grid = stats.norm.cdf(fit['a'] + fit['b'] * z)
        ax.plot(fpr_grid, tpr_grid, color=color, linestyle=ls, lw=1.3,
                 label=f"{label} (AUC={fit['auc_binormal']:.2f})")

    ax.set_xlabel('1 - Specificity', fontsize=6.5, fontweight='bold')
    ax.set_ylabel('Sensitivity', fontsize=6.5, fontweight='bold')
    ax.legend(loc='lower right', fontsize=4.3,
              handlelength=1.3, borderpad=0.3, labelspacing=0.25)
    ax.tick_params(labelsize=5.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curves_binormal_smooth.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(out_dir / 'roc_curves_binormal_smooth.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved smooth analytic binormal ROC to {out_dir}")


def out_dir_for(dataset_name):
    return DATASETS[dataset_name].parent / 'visualizations_binormal_fitted'


print("=" * 70)
print("BINORMAL ROC FIT: Spatial / Radiomics / Fusion on Test, TJH, SPCH")
print("(Dataset002_ECResponse)")
print("=" * 70)

for dataset_name, path in DATASETS.items():
    df = pd.read_csv(path)
    df['case_id'] = df['case_id'].astype(str)
    y = df['true_label'].values
    print(f"\n--- {dataset_name} (n={len(df)}) ---")

    df_fitted = df[['case_id', 'true_label', 'true_label_name']].copy()
    fits_by_col = {}

    for col in PROB_COLS:
        raw_proba = df[col].values
        auc_empirical = roc_auc_score(y, raw_proba)

        fit = fit_binormal(y, raw_proba)
        p_fitted = fit['posterior_fn'](raw_proba)
        auc_fitted_empirical = roc_auc_score(y, p_fitted)
        fits_by_col[col] = fit

        df_fitted[col] = p_fitted

        n_unique_raw = len(np.unique(raw_proba))
        n_unique_fitted = len(np.unique(p_fitted))
        print(f"  {col:24s}: raw AUC={auc_empirical:.4f} (n_unique={n_unique_raw})  "
              f"binormal closed-form AUC={fit['auc_binormal']:.4f}  "
              f"fitted-score rank AUC={auc_fitted_empirical:.4f} (n_unique={n_unique_fitted})")

    out_path = path.parent / path.name.replace('.csv', '_binormal_fitted.csv')
    df_fitted = df_fitted.sort_values('fused_probability', ascending=False)
    df_fitted.to_csv(out_path, index=False)
    print(f"  Saved fitted scores to {out_path}")

    plot_smooth_binormal_roc(fits_by_col, out_dir_for(dataset_name))

print("\n" + "=" * 70)
print("REGENERATING ROC + DCA FIGURES FROM FITTED SCORES")
print("=" * 70)

for dataset_name, path in DATASETS.items():
    fitted_path = path.parent / path.name.replace('.csv', '_binormal_fitted.csv')
    out_dir = out_dir_for(dataset_name)

    print(f"\n--- {dataset_name} ---")
    viz = MedicalImagingVisualizer(predictions_csv=str(fitted_path), output_dir=str(out_dir))
    viz.plot_roc_curves()
    viz.plot_dca()
    viz.plot_calibration_curves()
    viz.plot_combined_radar()

print("\nDONE")
