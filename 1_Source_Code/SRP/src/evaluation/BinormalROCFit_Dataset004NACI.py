"""
Binormal (parametric) ROC fit for the three models (Spatial, Radiomics,
Fusion) on Test / SPCH / FUSCC.

Why: the raw predicted probabilities are heavily discretized (e.g. on SPCH,
the spatial model has only ~40 distinct values across 200+ patients -- a
symptom of few-feature, shallow XGBoost trees), so the empirical ROC/DCA
curves are jagged staircases dominated by a handful of large ties rather
than a smooth curve.

Method: logit-transform each model's probabilities, fit a Gaussian to the
transformed scores within each true-label class (mu0/sigma0 for pCR,
mu1/sigma1 for Non-pCR), then use those two fitted Gaussians as a
generative model to compute a smooth POSTERIOR probability for every
patient via Bayes' rule (with the empirical class prevalence as prior).
This is the standard binormal ROC construction; the posterior-probability
step is what turns it into continuous per-patient scores usable for DCA,
not just an ROC curve equation.

Outputs, per dataset:
  - *_predictions_results_binormal_fitted.csv -- same schema as the raw
    predictions file, with spatial/radiomics/fused_probability replaced by
    the binormal-fitted smooth probabilities (case_id/true_label preserved)
  - printed comparison of empirical vs. binormal-fit AUC per model, as a
    sanity check that the parametric fit isn't distorting discrimination
Then re-generates ROC, DCA, calibration, and combined-radar figures from
the fitted scores via the existing MedicalImagingVisualizer (unchanged
code, just pointed at the new fitted-score CSVs), saved to a sibling
`visualizations_binormal_fitted` folder so the original raw-score figures
are left untouched. Note the calibration curve here answers a different
question than the original -- it shows whether the fitted binormal
posterior probabilities line up with observed pCR frequency, not whether
the underlying classifier's raw output is calibrated. The combined radar
chart (ACC/SEN/SPE/PPV/NPV/F1 at the Youden-optimal threshold) is expected
to look nearly identical to the raw-score version: the binormal posterior
is a monotonic function of the raw score (via logit), so ties and rank
order -- and therefore which patients fall on each side of the optimal
threshold -- are preserved.
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

DATA_ROOT = Path('XXXyour path/Dataset/Dataset004-NACI')
EPS = 1e-6
MODEL_STYLE = {
    'spatial_probability':   ('Spatial Feature-only Model',   '#E74C3C', '-'),
    'radiomics_probability': ('Conventional Radiomics Model', '#3498DB', '-'),
    'fused_probability':     ('SRP Model',                    '#009E73', '-'),
}

DATASETS = {
    'Test': DATA_ROOT / 'model_results' / 'test_predictions_results_new_radiomics.csv',
    'SPCH': DATA_ROOT / 'SPCH' / 'model_results' / 'SPCH_predictions_results_new_radiomics.csv',
    'FUSCC': DATA_ROOT / 'FUSCC' / 'model_results' / 'FUSCC_predictions_results_new_radiomics.csv',
}
PROB_COLS = ['spatial_probability', 'radiomics_probability', 'fused_probability']


def logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_binormal(y_true, y_proba):
    """
    Fit a binormal model on logit(y_proba), split by class.
    Returns fitted params plus a function mapping raw proba -> smooth
    posterior probability, and the closed-form binormal AUC.
    """
    z = logit(y_proba)
    z0 = z[y_true == 0]  # pCR
    z1 = z[y_true == 1]  # Non-pCR

    mu0, sigma0 = np.mean(z0), max(np.std(z0, ddof=1), EPS)
    mu1, sigma1 = np.mean(z1), max(np.std(z1, ddof=1), EPS)

    prior1 = np.mean(y_true == 1)
    prior0 = 1 - prior1

    # Closed-form binormal AUC: Phi(a / sqrt(1+b^2)), a=(mu1-mu0)/sigma1, b=sigma0/sigma1
    a = (mu1 - mu0) / sigma1
    b = sigma0 / sigma1
    auc_binormal = stats.norm.cdf(a / np.sqrt(1 + b ** 2))

    def posterior(raw_proba):
        zz = logit(raw_proba)
        f1 = stats.norm.pdf(zz, mu1, sigma1)
        f0 = stats.norm.pdf(zz, mu0, sigma0)
        num = prior1 * f1
        den = num + prior0 * f0
        # guard against 0/0 in the extreme tails
        return np.where(den > 0, num / np.maximum(den, EPS), 0.5)

    return {
        'mu0': mu0, 'sigma0': sigma0, 'mu1': mu1, 'sigma1': sigma1,
        'a': a, 'b': b, 'auc_binormal': auc_binormal,
        'posterior_fn': posterior,
    }


def plot_smooth_binormal_roc(fits_by_col, out_dir, dpi=300):
    """
    Plot the literal closed-form binormal ROC curve (a continuous line,
    not an empirical step function) for each model:
        TPR(FPR) = Phi(a + b * Phi^-1(FPR))
    This is what actually looks "smooth" -- re-running the empirical ROC
    on fitted-but-still-per-patient scores (as MedicalImagingVisualizer's
    plot_roc_curves does) still staircases at the sample size; only the
    analytic curve is continuous.
    """
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


print("=" * 70)
print("BINORMAL ROC FIT: Spatial / Radiomics / Fusion on Test, SPCH, FUSCC")
print("=" * 70)

def out_dir_for(dataset_name):
    if dataset_name == 'Test':
        return DATA_ROOT / 'model_results' / 'visualizations_binormal_fitted' / 'Test'
    elif dataset_name == 'SPCH':
        return DATA_ROOT / 'SPCH' / 'model_results' / 'visualizations_binormal_fitted'
    else:
        return DATA_ROOT / 'FUSCC' / 'model_results' / 'visualizations_binormal_fitted'


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
        auc_fitted_empirical = roc_auc_score(y, p_fitted)  # rank-based AUC of the fitted scores
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
