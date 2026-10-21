"""
Generates the standard model-performance figure suite (ROC, calibration,
DCA, radar charts, confusion matrices, probability distributions, metrics
table) for the three models -- Spatial, Radiomics, Fusion -- using the new
1000-trial sum-AUC-selected radiomics model, for the Test (SHPH holdout),
SPCH, and FUSCC datasets. Also generates a SHAP feature-importance
comparison figure (spatial vs. radiomics) fit on the SHPH Train cohort.

Reuses MedicalImagingVisualizer from PlotFigures.py unchanged for the
per-dataset performance suite: the *_predictions_results_new_radiomics.csv
files already carry the exact columns it expects (true_label,
spatial_probability, radiomics_probability, fused_probability).

For the SHAP figure, FeatureVisualizer.plot_feature_importance_comparison()
can't be reused as-is: it hardcodes stale XGBoost hyperparameters (83/0.007
and 100/0.14, leftover from an early pipeline version) and fits on whatever
cohort its input CSVs contain, not specifically Train. FeatureVisualizerTrain
below subclasses it to (a) restrict to the 186-case SHPH Train cohort (the
only split the models were actually fit on) and (b) use the real current
best hyperparameters for both models, pulled from best_hyperparameters.json
/ best_models.joblib rather than duplicated as literals here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PlotFigures import MedicalImagingVisualizer, FeatureVisualizer, DOUBLE_COL

DATA_ROOT = 'XXXyour path/Dataset/Dataset004-NACI'

DATASETS = {
    'Test': dict(
        predictions_csv=f'{DATA_ROOT}/model_results/test_predictions_results_new_radiomics.csv',
        output_dir=f'{DATA_ROOT}/model_results/visualizations_new_radiomics/Test',
    ),
    'SPCH': dict(
        predictions_csv=f'{DATA_ROOT}/SPCH/model_results/SPCH_predictions_results_new_radiomics.csv',
        output_dir=f'{DATA_ROOT}/SPCH/model_results/visualizations_new_radiomics',
    ),
    'FUSCC': dict(
        predictions_csv=f'{DATA_ROOT}/FUSCC/model_results/FUSCC_predictions_results_new_radiomics.csv',
        output_dir=f'{DATA_ROOT}/FUSCC/model_results/visualizations_new_radiomics',
    ),
}


class FeatureVisualizerTrain(FeatureVisualizer):
    """
    FeatureVisualizer restricted to the SHPH Train cohort, with SHAP fit
    using the actual selected hyperparameters (not PlotFigures.py's stale
    hardcoded 83/0.007 and 100/0.14).
    """

    def __init__(self, *args, train_case_ids, spatial_hp, radiomics_hp, **kwargs):
        super().__init__(*args, **kwargs)
        train_case_ids = set(str(c) for c in train_case_ids)
        before_sp, before_rad = len(self.df_spatial), len(self.df_radiomics)
        self.df_spatial = self.df_spatial[self.df_spatial['case_id'].astype(str).isin(train_case_ids)].reset_index(drop=True)
        self.df_radiomics = self.df_radiomics[self.df_radiomics['case_id'].astype(str).isin(train_case_ids)].reset_index(drop=True)
        print(f"  Restricted to Train cohort: spatial {before_sp}->{len(self.df_spatial)}, "
              f"radiomics {before_rad}->{len(self.df_radiomics)}")
        self.spatial_hp = spatial_hp
        self.radiomics_hp = radiomics_hp

    def plot_feature_importance_comparison(self, figsize=(DOUBLE_COL, 3.6), dpi=300):
        """
        Same SHAP beeswarm layout as FeatureVisualizer's version, but each
        model's XGBClassifier is built with its actual best hyperparameters
        (n_estimators, learning_rate, max_depth, subsample, colsample_bytree)
        instead of stale hardcoded values, fit on Train only.
        """
        import shap
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE

        def _clean_spatial(f):
            return f.split('_', 1)[1]

        def _labels(df):
            return (df['response_label'] != 'pCR').astype(int).values

        def _prepare(df, features, hp):
            y = _labels(df)
            X_raw = df[features].values
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_raw)
            smote = SMOTE(random_state=42)
            X_tr, y_tr = smote.fit_resample(X_sc, y)
            mdl = XGBClassifier(
                n_estimators=int(hp['n_estimators']),
                learning_rate=float(hp['learning_rate']),
                max_depth=int(hp.get('max_depth', 4)),
                subsample=float(hp.get('subsample', 1.0)),
                colsample_bytree=float(hp.get('colsample_bytree', 1.0)),
                random_state=42,
                eval_metric='logloss',
            )
            mdl.fit(X_tr, y_tr)
            expl = shap.TreeExplainer(mdl)
            sv = expl.shap_values(X_sc)
            if isinstance(sv, list):
                sv = sv[1]
            return sv, X_raw, np.array(features)

        def _beeswarm(ax, sv, X_raw, feat_names, title, name_fn=None, panel_label=None):
            if name_fn:
                feat_names = np.array([name_fn(f) for f in feat_names])

            n = len(feat_names)
            mean_abs = np.abs(sv).mean(axis=0)
            order = np.argsort(mean_abs)
            sv_ord = sv[:, order]
            X_ord = X_raw[:, order]
            names_ord = feat_names[order]
            mean_ord = mean_abs[order]

            for i in range(n):
                ax.axhspan(i - 0.5, i + 0.5,
                           color='#f4f4f4' if i % 2 == 0 else 'white', lw=0, zorder=0)

            for i in range(n):
                ax.barh(i, mean_ord[i], height=0.5, color='#bbbbbb', alpha=0.55, zorder=1, linewidth=0)
                ax.barh(i, -mean_ord[i], height=0.5, color='#bbbbbb', alpha=0.55, zorder=1, linewidth=0)

            for i in range(n):
                sv_col = sv_ord[:, i]
                feat_col = X_ord[:, i]
                fn = (feat_col - feat_col.min()) / (feat_col.max() - feat_col.min() + 1e-10)
                jitter = np.random.uniform(-0.30, 0.30, size=len(sv_col))
                ax.scatter(sv_col, i + jitter, c=fn, cmap='RdBu_r', s=8, alpha=0.85,
                           linewidths=0, zorder=3, vmin=0, vmax=1)

            ax.axvline(0, color='#333333', lw=0.8, zorder=2)
            ax.set_yticks(range(n))
            ax.set_yticklabels(names_ord, fontsize=6.5, rotation=30, ha='right', va='top')
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xlabel('SHAP value', fontsize=7.5)
            ax.tick_params(axis='x', labelsize=7, length=3, width=0.6)
            ax.tick_params(axis='y', length=0, pad=2)
            ax.set_title(title, fontsize=8.5, fontweight='bold', pad=6)
            ax.grid(True, axis='x', color='#cccccc', linestyle=':', linewidth=0.5, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.6)

            xmin, xmax = ax.get_xlim()
            kw = dict(fontsize=6.5, color='#555555', va='top', transform=ax.get_xaxis_transform())
            ax.text(xmin, -0.06, '← pCR', ha='left', **kw)
            ax.text(xmax, -0.06, 'Non-pCR →', ha='right', **kw)

            if panel_label:
                ax.text(-0.22, 1.08, panel_label, transform=ax.transAxes,
                        fontsize=8, fontweight='bold', va='bottom', ha='left')

        np.random.seed(42)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 2, 0.10], wspace=1.50,
                                left=0.18, right=0.91, top=0.87, bottom=0.22)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])

        sv1, X1, n1 = _prepare(self.df_spatial, self.selected_spatial, self.spatial_hp)
        _beeswarm(ax1, sv1, X1, n1, 'Spatial Features', name_fn=_clean_spatial, panel_label='(a)')

        sv2, X2, n2 = _prepare(self.df_radiomics, self.selected_radiomics, self.radiomics_hp)
        _beeswarm(ax2, sv2, X2, n2, 'Radiomics Features', name_fn=None, panel_label='(b)')

        sm = ScalarMappable(cmap='RdBu_r', norm=Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('Feature value', fontsize=7, labelpad=3)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(['Low', '', 'High'], fontsize=6.5)
        cb.ax.tick_params(length=2, width=0.5)
        cb.outline.set_linewidth(0.5)

        filename = 'feature_importance_comparison_new_radiomics.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ SHAP feature importance comparison (Train, new radiomics) saved to {self.output_dir}")


if __name__ == "__main__":
    print("=" * 80)
    print(" " * 15 + "FIGURES: SPATIAL / RADIOMICS(NEW) / FUSION(NEW) MODELS")
    print("=" * 80)

    for name, cfg in DATASETS.items():
        print("\n" + "#" * 80)
        print(f"DATASET: {name}")
        print("#" * 80)
        visualizer = MedicalImagingVisualizer(
            predictions_csv=cfg['predictions_csv'],
            output_dir=cfg['output_dir'],
        )
        visualizer.generate_all_plots()

    print("\n" + "#" * 80)
    print("SHAP FEATURE IMPORTANCE COMPARISON (SHPH Train cohort)")
    print("#" * 80)

    with open(f'{DATA_ROOT}/model_results/best_hyperparameters.json') as f:
        hp = json.load(f)
    artifacts = joblib.load(f'{DATA_ROOT}/model_results/best_models.joblib')

    selected_spatial_features = hp['selected_spatial_features']
    selected_radiomics_features = artifacts['radiomics_selected_names_1000trial_sumauc']
    spatial_hp = hp['spatial']
    radiomics_hp = artifacts['radiomics_params_1000trial_sumauc']

    df_train = pd.read_csv(f'{DATA_ROOT}/model_results/train_predictions_results.csv')
    train_case_ids = df_train['case_id'].astype(str).tolist()
    print(f"  Train cohort: {len(train_case_ids)} cases")
    print(f"  Spatial hyperparameters: n_estimators={spatial_hp['n_estimators']}, "
          f"learning_rate={spatial_hp['learning_rate']:.4f}, max_depth={spatial_hp['max_depth']}")
    print(f"  Radiomics(new) hyperparameters: n_estimators={radiomics_hp['n_estimators']}, "
          f"learning_rate={radiomics_hp['learning_rate']:.4f}, max_depth={radiomics_hp['max_depth']}")

    feature_viz = FeatureVisualizerTrain(
        spatial_features_csv=f'{DATA_ROOT}/SHPH/spatial3d_features.csv',
        texture_features_csv=f'{DATA_ROOT}/SHPH/spatial_texture_features.csv',
        radiomics_features_csv=f'{DATA_ROOT}/SHPH/radiomics_features.csv',
        labels_excel=f'{DATA_ROOT}/SHPH.xlsx',
        selected_spatial_features=selected_spatial_features,
        selected_radiomics_features=selected_radiomics_features,
        output_dir=f'{DATA_ROOT}/model_results/visualizations_new_radiomics/Train',
        train_case_ids=train_case_ids,
        spatial_hp=spatial_hp,
        radiomics_hp=radiomics_hp,
    )
    feature_viz.plot_feature_importance_comparison()

    print("\n" + "=" * 80)
    print("ALL FIGURES COMPLETE")
    print("=" * 80)
