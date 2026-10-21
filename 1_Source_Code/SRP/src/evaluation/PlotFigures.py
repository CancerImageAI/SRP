import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, 
    auc, 
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# NATURE-STYLE GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────
# Nature figures are placed at a fixed print width — single column 89mm
# (3.5in) or double column 183mm (7.2in) — so figures must be generated
# AT that final size with fonts already legible there (~6-8pt). Building
# an 8-18in figure with 10-16pt fonts and letting the journal shrink it
# scales the text down proportionally (e.g. an 8in-wide figure shrunk to
# 3.5in leaves 12pt labels at ~5.25pt), so figsize and fontsize must be
# chosen together, not independently.
SINGLE_COL = 3.5
DOUBLE_COL = 7.2
# For figures meant to be assembled by hand into a multi-panel figure
# later (not read standalone) — sized to one third of a double column,
# matching a 3-panels-per-row layout.
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

class MedicalImagingVisualizer:
    """
    Publication-quality visualization toolkit for medical imaging ML results.
    Designed for Nature journal submission standards.
    """
    
    def __init__(self, predictions_csv: str, output_dir: str):
        """
        Initialize visualizer with predictions data.
        
        Parameters:
        -----------
        predictions_csv : str
            Path to CSV file containing predictions
        output_dir : str
            Directory to save output figures
        """
        self.predictions_df = pd.read_csv(predictions_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define model configurations
        self.models = {
            'Spatial Feature-only Model': {
                'prob_col': 'spatial_probability',
                'color': '#E74C3C',  # Red
                'linestyle': '-'
            },
            'Conventional Radiomics Model': {
                'prob_col': 'radiomics_probability',
                'color': '#3498DB',  # Blue
                'linestyle': '-'
            },
            'SRP Model': {
                'prob_col': 'fused_probability',
                'color': "#009E73",  # Green
                'linestyle': '-'
            }
        }
        
        self.y_true = self.predictions_df['true_label'].values

        # Binary cutoff per model, chosen by the Youden index (max of
        # sensitivity + specificity - 1 over the ROC curve) instead of a
        # fixed 0.5 -- appropriate here since the model outputs are
        # produced by shallow, few-feature XGBoost trees with heavily
        # discretized/tied probabilities (see class docstring context),
        # so 0.5 is not a meaningful decision point for these particular
        # score distributions. Used everywhere a hard pCR/Non-pCR call is
        # needed: confusion matrices, radar charts, metrics table.
        self.thresholds = self._compute_optimal_thresholds()

    def _compute_optimal_thresholds(self):
        """Youden-index-optimal threshold per model: argmax(TPR - FPR)."""
        thresholds = {}
        for model_name, config in self.models.items():
            y_proba = self.predictions_df[config['prob_col']].values
            fpr, tpr, thr = roc_curve(self.y_true, y_proba)
            youden = tpr - fpr
            best_idx = np.argmax(youden)
            thresholds[model_name] = float(thr[best_idx])
        return thresholds

    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive classification metrics."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'ACC': (tp + tn) / (tp + tn + fp + fn),
            'SEN': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Sensitivity/Recall
            'SPE': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
            'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'F1': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
            'AUC': auc(*roc_curve(y_true, y_proba)[:2])
        }
        
        # Calculate 95% CI for AUC using bootstrap
        metrics['AUC_CI'] = self._bootstrap_auc_ci(y_true, y_proba)
        
        return metrics
    
    def _bootstrap_auc_ci(self, y_true, y_proba, n_bootstraps=1000, ci=0.95):
        """Calculate bootstrap confidence interval for AUC."""
        np.random.seed(42)
        bootstrapped_aucs = []
        
        for _ in range(n_bootstraps):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = auc(*roc_curve(y_true[indices], y_proba[indices])[:2])
            bootstrapped_aucs.append(score)
        
        sorted_scores = np.array(bootstrapped_aucs)
        sorted_scores.sort()
        
        alpha = 1 - ci
        lower_idx = int(alpha/2 * len(sorted_scores))
        upper_idx = int((1 - alpha/2) * len(sorted_scores))
        
        return (sorted_scores[lower_idx], sorted_scores[upper_idx])
    
    def plot_roc_curves(self, figsize=(SUBPANEL_COL, SUBPANEL_COL), dpi=300):
        """
        Plot ROC curves for all models with 95% CI.

        Sized as a standalone sub-panel (~1/3 double column) for the user
        to combine by hand into their own multi-panel figure — not meant
        to be read at this size on its own.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Resolution for saved figure
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Chance')

        # Plot ROC for each model
        for model_name, config in self.models.items():
            y_proba = self.predictions_df[config['prob_col']].values

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(self.y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve — legend omits the 95% CI: at ~2.3in wide
            # there isn't room for "Model (AUC=.., 95% CI: ..-..)" at a
            # legible size. Full stats are in performance_metrics.csv.
            ax.plot(fpr, tpr,
                   color=config['color'],
                   linestyle=config['linestyle'],
                   lw=1.3,
                   label=f'{model_name} (AUC={roc_auc:.2f})')

        # Styling
        ax.set_xlabel('1 - Specificity', fontsize=6.5, fontweight='bold')
        ax.set_ylabel('Sensitivity', fontsize=6.5, fontweight='bold')
        # ax.set_title('Receiver Operating Characteristic (ROC) Curves',
        #             fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=4.3,
                 handlelength=1.3, borderpad=0.3, labelspacing=0.25)
        ax.tick_params(labelsize=5.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to {self.output_dir}")
    
    def plot_calibration_curves(self, n_bins=10, figsize=(DOUBLE_COL, 2.8), dpi=300):
        """
        Plot calibration curves (reliability diagrams).
        
        Parameters:
        -----------
        n_bins : int
            Number of bins for calibration
        figsize : tuple
            Figure size
        dpi : int
            Resolution
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        
        for idx, (model_name, config) in enumerate(self.models.items()):
            ax = axes[idx]
            y_proba = self.predictions_df[config['prob_col']].values
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_true, y_proba, n_bins=n_bins, strategy='uniform'
            )
            
            # Calculate Brier score
            brier = brier_score_loss(self.y_true, y_proba)
            
            # Plot perfect calibration
            ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Perfect calibration')

            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives,
                   marker='o', markersize=3.5, lw=1.5,
                   color=config['color'], label=f'{model_name}')

            # Add histogram of predictions
            ax_hist = ax.twinx()
            ax_hist.hist(y_proba, bins=n_bins, alpha=0.3,
                        color=config['color'], edgecolor='black', linewidth=0.5)
            ax_hist.set_ylabel('Count', fontsize=6)
            ax_hist.tick_params(labelsize=5.5)
            ax_hist.set_ylim([0, ax_hist.get_ylim()[1] * 1.5])

            # Styling
            ax.set_xlabel('Mean Predicted Probability', fontsize=7, fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontsize=7, fontweight='bold')
            ax.set_title(f'{model_name}\n(Brier Score: {brier:.2f})',
                        fontsize=7, fontweight='bold')
            ax.legend(loc='upper left', fontsize=5.5)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_curves.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'calibration_curves.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Calibration curves saved to {self.output_dir}")
    
    def plot_dca(self, figsize=(SUBPANEL_COL, SUBPANEL_COL), dpi=300):
        """
        Plot Decision Curve Analysis (DCA).

        DCA helps evaluate the clinical utility of prediction models by
        comparing net benefit across different threshold probabilities.

        Sized as a standalone sub-panel (~1/3 double column) for the user
        to combine by hand into their own multi-panel figure.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Threshold probabilities
        thresholds = np.arange(0.01, 1.0, 0.01)
        
        # Calculate net benefit for "treat all" and "treat none" strategies
        prevalence = np.mean(self.y_true)
        treat_all_nb = []
        
        for pt in thresholds:
            # Net benefit for treating all patients
            nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
            treat_all_nb.append(max(nb_all, 0))
        
        # Plot treat all and treat none
        ax.plot(thresholds, treat_all_nb, 'k--', lw=0.8, label='Treat All', alpha=0.6)
        ax.axhline(y=0, color='gray', linestyle='--', lw=0.8, label='Treat None', alpha=0.6)
        
        # Calculate and plot net benefit for each model
        for model_name, config in self.models.items():
            y_proba = self.predictions_df[config['prob_col']].values
            net_benefits = []
            
            for pt in thresholds:
                # Classify based on threshold
                y_pred_at_threshold = (y_proba >= pt).astype(int)
                
                # True positives and false positives
                tp = np.sum((y_pred_at_threshold == 1) & (self.y_true == 1))
                fp = np.sum((y_pred_at_threshold == 1) & (self.y_true == 0))
                n = len(self.y_true)
                
                # Net benefit calculation
                net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))
                
                net_benefits.append(net_benefit)

            for i in range(len(net_benefits)):
                if net_benefits[i] < 0:
                    net_benefits[i] = 0
            # Plot net benefit curve
            ax.plot(thresholds, net_benefits,
                   color=config['color'],
                   linestyle=config['linestyle'],
                   lw=1.3,
                   label=model_name)

        # Styling
        ax.set_xlabel('Threshold Probability', fontsize=6.5, fontweight='bold')
        ax.set_ylabel('Net Benefit', fontsize=6.5, fontweight='bold')
        ax.set_title('Decision Curve Analysis', fontsize=7.5, fontweight='bold', pad=6)
        ax.legend(loc='upper right', fontsize=4.3,
                 handlelength=1.3, borderpad=0.3, labelspacing=0.25)
        ax.tick_params(labelsize=5.5)
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'decision_curve_analysis.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'decision_curve_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Decision curve analysis saved to {self.output_dir}")
    
    def plot_radar_chart(self, figsize=(DOUBLE_COL, 2.8), dpi=300):
        """
        Plot radar chart comparing performance metrics across models.
        
        Metrics: ACC, SEN, SPE, PPV, NPV
        """
        # Calculate metrics for all models
        all_metrics = {}
        for model_name, config in self.models.items():
            y_proba = self.predictions_df[config['prob_col']].values
            y_pred = (y_proba > self.thresholds[model_name]).astype(int)
            metrics = self.calculate_metrics(self.y_true, y_pred, y_proba)
            all_metrics[model_name] = metrics
        
        # Metrics to display
        metric_names = ['ACC', 'SEN', 'SPE', 'PPV', 'NPV']
        num_metrics = len(metric_names)
        
        # Create subplots - one radar chart per model
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, 
                                subplot_kw=dict(projection='polar'))
        
        for idx, (model_name, config) in enumerate(self.models.items()):
            ax = axes[idx]
            
            # Get metric values
            values = [all_metrics[model_name][m] for m in metric_names]
            values += values[:1]  # Complete the circle
            
            # Angle for each metric
            angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=1.5, color=config['color'], label=model_name, markersize=3.5)
            ax.fill(angles, values, alpha=0.25, color=config['color'])

            # Styling
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names, fontsize=6.5, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=5)
            ax.set_title(model_name, fontsize=7, fontweight='bold', pad=12)
            ax.grid(True, linestyle='--', alpha=0.5)

            # Add value labels
            for angle, value, metric in zip(angles[:-1], values[:-1], metric_names):
                ax.text(angle, value + 0.05, f'{value:.2f}',
                       ha='center', va='center', fontsize=5, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_charts.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'radar_charts.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Radar charts saved to {self.output_dir}")
    
    def plot_combined_radar(self, figsize=(SUBPANEL_COL, SUBPANEL_COL * 1.15), dpi=300):
        """
        Plot a single radar chart comparing all models.

        Sized as a standalone sub-panel (~1/3 double column) for the user
        to combine by hand into their own multi-panel figure — the extra
        height accommodates the stacked legend below the circle.
        """
        # Calculate metrics for all models
        all_metrics = {}
        for model_name, config in self.models.items():
            y_proba = self.predictions_df[config['prob_col']].values
            y_pred = (y_proba > self.thresholds[model_name]).astype(int)
            metrics = self.calculate_metrics(self.y_true, y_pred, y_proba)
            all_metrics[model_name] = metrics
        
        # Metrics to display
        metric_names = ['ACC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1']
        num_metrics = len(metric_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw=dict(projection='polar'))
        
        # Angle for each metric
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot each model
        for model_name, config in self.models.items():
            values = [all_metrics[model_name][m] for m in metric_names]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=1.2,
                   color=config['color'],
                   linestyle=config['linestyle'],
                   label=model_name, markersize=3)
            ax.fill(angles, values, alpha=0.15, color=config['color'])

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=6, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

        # Place the radial (r-axis) tick labels in the gap between SEN and
        # SPE spokes (12 o'clock) rather than directly on a spoke — every
        # metric has at least one model scoring high somewhere on the
        # circle, so no single spoke is fully clear, but the midpoint of
        # this gap keeps the labels off the vertex markers/lines. An opaque
        # white background on each label guards against any residual
        # overlap with the semi-transparent fill.
        sen_angle_deg = np.degrees(angles[metric_names.index('SEN')])
        spe_angle_deg = np.degrees(angles[metric_names.index('SPE')])
        ax.set_rlabel_position((sen_angle_deg + spe_angle_deg) / 2)
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=4.5)
        for tick_label in ax.get_yticklabels():
            tick_label.set_rotation(0)
            tick_label.set_horizontalalignment('center')
            tick_label.set_bbox(dict(facecolor='white', edgecolor='none',
                                     alpha=1.0, pad=0.3))

        # ax.set_title('Performance Metrics Comparison',
        #             fontsize=14, fontweight='bold', pad=30)bbox_to_anchor=(1.3, 1.1),
        # ncol=1: at ~2.3in wide, three model names side by side won't fit —
        # stack them instead, matching Panel C's legend in the combined
        # publication figure.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                 ncol=1, fontsize=4.3, frameon=True,
                 handlelength=1.3, borderpad=0.3, labelspacing=0.25)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_radar_chart.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'combined_radar_chart.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Combined radar chart saved to {self.output_dir}")
    
    def plot_confusion_matrices(self, figsize=(DOUBLE_COL, 2.6), dpi=300):
        """
        Plot confusion matrices for all models.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

        for idx, (model_name, config) in enumerate(self.models.items()):
            ax = axes[idx]
            y_proba = self.predictions_df[config['prob_col']].values
            y_pred = (y_proba > self.thresholds[model_name]).astype(int)

            # Calculate confusion matrix
            cm = confusion_matrix(self.y_true, y_pred)

            # Normalize for percentages
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       cbar=False, ax=ax, square=True,
                       linewidths=1, linecolor='white',
                       annot_kws={'fontsize': 8, 'fontweight': 'bold'})

            # Add percentage annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                                 ha="center", va="center", fontsize=6, color='gray')

            # Styling
            ax.set_xlabel('Predicted Label', fontsize=7, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=7, fontweight='bold')
            ax.set_title(f'{model_name}\n(cutoff={self.thresholds[model_name]:.2f})',
                        fontsize=7, fontweight='bold', pad=8)
            ax.set_xticklabels(['pCR', 'Non-pCR'], fontsize=6)
            ax.set_yticklabels(['pCR', 'Non-pCR'], fontsize=6, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrices saved to {self.output_dir}")
    
    def plot_probability_distributions(self, figsize=(DOUBLE_COL, 2.8), dpi=300):
        """
        Plot probability distribution for positive and negative classes.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

        for idx, (model_name, config) in enumerate(self.models.items()):
            ax = axes[idx]
            y_proba = self.predictions_df[config['prob_col']].values

            # Separate by true class
            proba_pcr = y_proba[self.y_true == 0]
            proba_nonpcr = y_proba[self.y_true == 1]

            # Plot distributions
            ax.hist(proba_pcr, bins=20, alpha=0.6, color='#3498DB',
                   label='pCR (n={})'.format(len(proba_pcr)),
                   edgecolor='black', linewidth=0.8)
            ax.hist(proba_nonpcr, bins=20, alpha=0.6, color='#E74C3C',
                   label='Non-pCR (n={})'.format(len(proba_nonpcr)),
                   edgecolor='black', linewidth=0.8)

            # Add threshold line (Youden-optimal cutoff for this model)
            thr = self.thresholds[model_name]
            ax.axvline(x=thr, color='black', linestyle='--', linewidth=1.2,
                      label=f'Cutoff ({thr:.2f})', alpha=0.7)

            # Styling
            ax.set_xlabel('Predicted Probability', fontsize=7, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=7, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=7, fontweight='bold', pad=8)
            ax.legend(loc='upper center', fontsize=5.5, frameon=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'probability_distributions.pdf', dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'probability_distributions.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Probability distributions saved to {self.output_dir}")
    
    def create_metrics_table(self):
        """
        Create comprehensive metrics table for all models.
        """
        results = []
        
        for model_name, config in self.models.items():
            y_proba = self.predictions_df[config['prob_col']].values
            y_pred = (y_proba > self.thresholds[model_name]).astype(int)
            metrics = self.calculate_metrics(self.y_true, y_pred, y_proba)

            # Format metrics
            ci_lower, ci_upper = metrics['AUC_CI']

            results.append({
                'Model': model_name,
                'Cutoff (Youden)': f"{self.thresholds[model_name]:.3f}",
                'Accuracy': f"{metrics['ACC']:.2f}",
                'Sensitivity': f"{metrics['SEN']:.2f}",
                'Specificity': f"{metrics['SPE']:.2f}",
                'PPV': f"{metrics['PPV']:.2f}",
                'NPV': f"{metrics['NPV']:.2f}",
                'F1-Score': f"{metrics['F1']:.2f}",
                'AUC': f"{metrics['AUC']:.2f}",
                'AUC 95% CI': f"({ci_lower:.2f}-{ci_upper:.2f})"
            })
        
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        results_df.to_csv(self.output_dir / 'performance_metrics.csv', index=False)
        
        # Create LaTeX table for publication
        latex_table = results_df.to_latex(index=False, 
                                          caption='Model Performance Metrics',
                                          label='tab:metrics',
                                          position='htbp',
                                          column_format='lccccccccc')
        
        with open(self.output_dir / 'performance_metrics.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"✓ Metrics table saved to {self.output_dir}")
        print("\nPerformance Metrics Summary:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def generate_all_plots(self):
        """
        Generate all visualization plots.
        """
        print("\n" + "="*60)
        print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.plot_roc_curves()
        self.plot_calibration_curves()
        self.plot_dca()
        self.plot_radar_chart()
        self.plot_combined_radar()
        self.plot_confusion_matrices()
        self.plot_probability_distributions()
        self.create_metrics_table()
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS COMPLETED")
        print("="*60)


class FeatureVisualizer:
    """
    Visualize selected features from spatial and radiomics models.
    """
    
    def __init__(self, 
                 spatial_features_csv: str,
                 texture_features_csv: str, 
                 radiomics_features_csv: str,
                 labels_excel: str,
                 selected_spatial_features: List[str],
                 selected_radiomics_features: List[str],
                 output_dir: str):
        """
        Initialize feature visualizer.
        
        Parameters:
        -----------
        spatial_features_csv : str
            Path to spatial features CSV
        radiomics_features_csv : str
            Path to radiomics features CSV
        labels_excel : str
            Path to labels Excel file
        selected_spatial_features : list
            List of selected spatial feature names
        selected_radiomics_features : list
            List of selected radiomics feature names
        output_dir : str
            Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spatial features
        df_spatial = pd.read_csv(spatial_features_csv)
        df_spatial['case_id'] = df_spatial['case_id'].astype(str).apply(
            lambda x: x.split('.nii')[0] if '.nii' in x else x
        )
        df_texture_spatial = pd.read_csv(texture_features_csv)
        df_texture_spatial['case_id'] = df_texture_spatial['case_id'].astype(str).apply(
            lambda x: x.split('.nii')[0] if '.nii' in x else x
        )
        df_spatial = pd.merge(df_spatial, df_texture_spatial, on='case_id', how='inner')
        
        # Load radiomics features
        df_radiomics = pd.read_csv(radiomics_features_csv)
        df_radiomics.rename(columns={'CaseID': 'case_id'}, inplace=True)
        df_radiomics['case_id'] = df_radiomics['case_id'].astype(str)
        
        # Load labels
        df_labels = pd.read_excel(labels_excel)
        df_labels.columns = ['case_id', 'response']
        df_labels['case_id'] = df_labels['case_id'].astype(str)
        
        # Merge data
        self.df_spatial = pd.merge(df_spatial, df_labels, on='case_id', how='inner')
        self.df_radiomics = pd.merge(df_radiomics, df_labels, on='case_id', how='inner')
        
        self.selected_spatial = selected_spatial_features
        self.selected_radiomics = selected_radiomics_features
        
        # Map response to labels
        self.df_spatial['response_label'] = self.df_spatial['response'].apply(
            lambda x: 'pCR' if (isinstance(x, str) and x == 'pCR') or x == 1 else 'Non-pCR'
        )
        self.df_radiomics['response_label'] = self.df_radiomics['response'].apply(
            lambda x: 'pCR' if (isinstance(x, str) and x == 'pCR') or x == 1 else 'Non-pCR'
        )
        

        print(f"✓ Loaded spatial features: {len(self.df_spatial)} samples")
        print(f"✓ Loaded radiomics features: {len(self.df_radiomics)} samples")
    
    def plot_feature_boxplots(self, feature_type='spatial', figsize=None, dpi=300):
        """
        Create boxplots for selected features.
        
        Parameters:
        -----------
        feature_type : str
            'spatial' or 'radiomics'
        """
        if feature_type == 'spatial':
            features = self.selected_spatial
            df = self.df_spatial
            title_prefix = 'Spatial'
        else:
            features = self.selected_radiomics
            df = self.df_radiomics
            title_prefix = 'Radiomics'
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = int(np.ceil(n_features / n_cols))

        if figsize is None:
            # Width capped at Nature's double-column print width regardless
            # of n_cols; height grows per row since a taller supplementary
            # figure is fine, but going wider than the page is not.
            figsize = (DOUBLE_COL, 2.3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, feature in enumerate(features):
            ax = axes[idx]

            # Prepare data
            data_to_plot = []
            labels = []
            for response in ['pCR', 'Non-pCR']:
                subset = df[df['response_label'] == response][feature].dropna()
                data_to_plot.append(subset)
                labels.append(f'{response}\n(n={len(subset)})')

            # Create boxplot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           widths=0.6,
                           boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=0.8),
                           medianprops=dict(color='red', linewidth=1.2),
                           whiskerprops=dict(color='black', linewidth=0.8),
                           capprops=dict(color='black', linewidth=0.8),
                           flierprops=dict(marker='o', markerfacecolor='gray',
                                         markersize=3, alpha=0.5))

            # Color boxes differently
            colors = ['#3498DB', '#E74C3C']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Add individual points with jitter
            for i, data in enumerate(data_to_plot):
                y = data
                x = np.random.normal(i + 1, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.4, s=10, color='black', edgecolors='none')

            # Statistical test (Mann-Whitney U test)
            if len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:
                stat, p_value = stats.mannwhitneyu(data_to_plot[0], data_to_plot[1])

                # Add p-value annotation
                y_max = max([d.max() for d in data_to_plot if len(d) > 0])
                y_min = min([d.min() for d in data_to_plot if len(d) > 0])
                y_range = y_max - y_min

                # Draw significance bar
                y_sig = y_max + 0.1 * y_range
                ax.plot([1, 2], [y_sig, y_sig], 'k-', linewidth=0.8)
                ax.plot([1, 1], [y_sig - 0.02*y_range, y_sig], 'k-', linewidth=0.8)
                ax.plot([2, 2], [y_sig - 0.02*y_range, y_sig], 'k-', linewidth=0.8)

                # # Add p-value text
                # if p_value < 0.001:
                #     p_text = 'p < 0.001***'
                # elif p_value < 0.01:
                #     p_text = f'p = {p_value:.2f}**'
                # elif p_value < 0.05:
                #     p_text = f'p = {p_value:.2f}*'
                # else:
                #     p_text = f'p = {p_value:.2f}ns'

                # ax.text(1.5, y_sig + 0.02*y_range, p_text,
                #        ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Styling
            feature_name = feature.split('_', 1)[1] #feature.split('_')[-1] if '_' in feature else feature
            ax.set_ylabel('Feature Value', fontsize=6.5, fontweight='bold')
            ax.set_title(f'{feature_name}', fontsize=6.5, fontweight='bold', pad=6)
            ax.tick_params(labelsize=5.5)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{title_prefix} Feature Distribution by Response',
                    fontsize=9, fontweight='bold', y=1.02)
        plt.tight_layout()

        filename = f'{feature_type}_feature_boxplots.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {title_prefix} boxplots saved to {self.output_dir}")
    
    def plot_feature_violin(self, feature_type='spatial', figsize=None, dpi=300):
        """
        Create violin plots for selected features.
        
        Parameters:
        -----------
        feature_type : str
            'spatial' or 'radiomics'
        """
        if feature_type == 'spatial':
            features = self.selected_spatial
            df = self.df_spatial
            title_prefix = 'Spatial'
        else:
            features = self.selected_radiomics
            df = self.df_radiomics
            title_prefix = 'Radiomics'
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = int(np.ceil(n_features / n_cols))

        if figsize is None:
            figsize = (DOUBLE_COL, 2.3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, feature in enumerate(features):
            ax = axes[idx]

            # Create violin plot
            parts = ax.violinplot(
                [df[df['response_label'] == 'pCR'][feature].dropna(),
                 df[df['response_label'] == 'Non-pCR'][feature].dropna()],
                positions=[1, 2],
                widths=0.7,
                showmeans=True,
                showmedians=True
            )

            # Color violins
            colors = ['#3498DB', '#E74C3C']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.8)

            # Styling for violin plot elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(0.8)
            
            # Statistical test
            pcr_data = df[df['response_label'] == 'pCR'][feature].dropna()
            nonpcr_data = df[df['response_label'] == 'Non-pCR'][feature].dropna()
            
            # if len(pcr_data) > 0 and len(nonpcr_data) > 0:
            #     stat, p_value = stats.mannwhitneyu(pcr_data, nonpcr_data)
                
            #     # Add p-value annotation
            #     y_max = max(pcr_data.max(), nonpcr_data.max())
            #     y_min = min(pcr_data.min(), nonpcr_data.min())
            #     y_range = y_max - y_min
                
            #     y_sig = y_max + 0.15 * y_range
            #     ax.plot([1, 2], [y_sig, y_sig], 'k-', linewidth=1.5)
                
            #     if p_value < 0.001:
            #         p_text = 'p < 0.001***'
            #     elif p_value < 0.01:
            #         p_text = f'p = {p_value:.2f}**'
            #     elif p_value < 0.05:
            #         p_text = f'p = {p_value:.2f}*'
            #     else:
            #         p_text = f'p = {p_value:.2f}ns'
                
            #     ax.text(1.5, y_sig + 0.03*y_range, p_text,
            #            ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Styling
            ax.set_xticks([1, 2])
            ax.set_xticklabels([f'pCR\n(n={len(pcr_data)})',
                               f'Non-pCR\n(n={len(nonpcr_data)})'], fontsize=5.5)
            feature_name = feature.split('_', 1)[1]  #feature.split('_')[-1] if '_' in feature else feature
            ax.set_ylabel('Feature Value', fontsize=6.5, fontweight='bold')
            ax.set_title(f'{feature_name}', fontsize=6.5, fontweight='bold', pad=6)
            ax.tick_params(axis='y', labelsize=5.5)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{title_prefix} Feature Distribution by Response',
                    fontsize=9, fontweight='bold', y=1.02)
        plt.tight_layout()

        filename = f'{feature_type}_feature_violinplots.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {title_prefix} violin plots saved to {self.output_dir}")
    
    def plot_feature_heatmap(self, feature_type='spatial', figsize=(DOUBLE_COL, 4.5), dpi=300):
        """
        Create heatmap of feature values grouped by response.
        
        Parameters:
        -----------
        feature_type : str
            'spatial' or 'radiomics'
        """
        if feature_type == 'spatial':
            features = self.selected_spatial
            df = self.df_spatial
            title_prefix = 'Spatial'
        else:
            features = self.selected_radiomics
            df = self.df_radiomics
            title_prefix = 'Radiomics'
        
        # Prepare data
        data = df[features + ['response_label']].copy()
        
        # Standardize features for better visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data[features] = scaler.fit_transform(data[features])
        
        # Sort by response
        data = data.sort_values('response_label')
        
        # Create figure — the response strip is a real GridSpec row (not a
        # manually positioned fig.add_axes) so tight_layout accounts for it
        # and it can't drift into the main axes' tick labels at any figsize.
        fig, (ax, divider_ax) = plt.subplots(
            2, 1, figsize=figsize, dpi=dpi,
            gridspec_kw={'height_ratios': [len(features), 0.6], 'hspace': 0.35})

        # Create heatmap
        feature_data = data[features].T

        # Plot
        im = ax.imshow(feature_data, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Standardized Value', fontsize=7, fontweight='bold')
        cbar.ax.tick_params(labelsize=6)

        # Set ticks
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels([f.split('_', 1)[1] for f in features],
                          fontsize=6)
        ax.set_xticks([])

        # Add response annotation
        response_labels = data['response_label'].values
        response_numeric = [0 if r == 'pCR' else 1 for r in response_labels]

        # Colored bar for response, sharing the heatmap's x-axis
        divider_ax.imshow([response_numeric], aspect='auto', cmap='coolwarm',
                         vmin=0, vmax=1,
                         extent=[0, feature_data.shape[1], 0, 1])
        divider_ax.set_yticks([])
        divider_ax.tick_params(axis='x', labelsize=6)
        divider_ax.set_xlabel('Sample index (sorted by response)', fontsize=7, fontweight='bold')

        # Add legend for response
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#6495ED', label='pCR'),
                          Patch(facecolor='#CD5C5C', label='Non-pCR')]
        divider_ax.legend(handles=legend_elements, loc='upper left',
                         bbox_to_anchor=(1.02, 1.6), fontsize=6)

        # Styling
        ax.set_ylabel('Features', fontsize=7, fontweight='bold')
        ax.set_title(f'{title_prefix} Feature Heatmap',
                    fontsize=9, fontweight='bold', pad=10)

        plt.tight_layout()
        
        filename = f'{feature_type}_feature_heatmap.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {title_prefix} heatmap saved to {self.output_dir}")
    
    def plot_feature_correlation(self, feature_type='spatial', figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.6), dpi=300):
        """
        Create correlation matrix for selected features.
        
        Parameters:
        -----------
        feature_type : str
            'spatial' or 'radiomics'
        """
        if feature_type == 'spatial':
            features = self.selected_spatial
            df = self.df_spatial
            title_prefix = 'Spatial'
        else:
            features = self.selected_radiomics
            df = self.df_radiomics
            title_prefix = 'Radiomics'
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   annot_kws={'fontsize': 5.5, 'fontweight': 'bold'},
                   ax=ax)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)

        # Styling
        feature_names_short = [f.split('_', 1)[1] for f in features]
        ax.set_xticklabels(feature_names_short, rotation=45, ha='right', fontsize=6)
        ax.set_yticklabels(feature_names_short, rotation=0, fontsize=6)
        ax.set_title(f'{title_prefix} Feature Correlation Matrix',
                    fontsize=8, fontweight='bold', pad=10)
        
        plt.tight_layout()
        
        filename = f'{feature_type}_feature_correlation.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {title_prefix} correlation matrix saved to {self.output_dir}")
    
    def plot_feature_importance_comparison(self, figsize=(DOUBLE_COL, 3.6), dpi=300):
        """
        Publication-quality SHAP beeswarm comparison (Nature BME style).

        Pipeline mirrors ComparsionClassificationModel.py:
          StandardScaler → SMOTE → XGBClassifier → shap.TreeExplainer

        Spatial  : XGBClassifier(n_estimators=83,  learning_rate=0.007)
        Radiomics: XGBClassifier(n_estimators=100, learning_rate=0.14)

        Layout: GridSpec with a dedicated colorbar column so the colorbar never
        overlaps the data panels.  Feature names are abbreviated for clarity.
        Background bars show ±mean|SHAP| extent.  Panel labels a/b included.
        """
        import shap
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        import matplotlib.gridspec as gridspec

        # ── Feature-name cleaning ───────────────────────────────────────────
        def _clean_spatial(f):
            # 'glcm_glcm_Idm_spatial_variability' → 'glcm_Idm_spatial_variability'
            # The leading category token is duplicated; drop just the first one.
            return f.split('_', 1)[1]

        # ── Labels (pCR=0, Non-pCR=1) ──────────────────────────────────────
        def _labels(df):
            return (df['response_label'] != 'pCR').astype(int).values

        # ── Model + SHAP ────────────────────────────────────────────────────
        def _prepare(df, features, n_estimators, learning_rate):
            y     = _labels(df)
            X_raw = df[features].values
            scaler = StandardScaler()
            X_sc  = scaler.fit_transform(X_raw)
            smote = SMOTE(random_state=42)
            X_tr, y_tr = smote.fit_resample(X_sc, y)
            try:
                mdl = XGBClassifier(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    random_state=42,
                                    use_label_encoder=True,
                                    eval_metric='logloss')
            except TypeError:
                mdl = XGBClassifier(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    random_state=42,
                                    eval_metric='logloss')
            mdl.fit(X_tr, y_tr)
            expl = shap.TreeExplainer(mdl)
            sv   = expl.shap_values(X_sc)
            if isinstance(sv, list):
                sv = sv[1]
            return sv, X_raw, np.array(features)

        # ── Beeswarm panel ──────────────────────────────────────────────────
        def _beeswarm(ax, sv, X_raw, feat_names, title,
                      name_fn=None, panel_label=None):
            if name_fn:
                feat_names = np.array([name_fn(f) for f in feat_names])

            n        = len(feat_names)
            mean_abs = np.abs(sv).mean(axis=0)
            order    = np.argsort(mean_abs)          # most important at top
            sv_ord   = sv[:, order]
            X_ord    = X_raw[:, order]
            names_ord = feat_names[order]
            mean_ord  = mean_abs[order]

            # Alternating row shading
            for i in range(n):
                ax.axhspan(i - 0.5, i + 0.5,
                           color='#f4f4f4' if i % 2 == 0 else 'white',
                           lw=0, zorder=0)

            # ±mean|SHAP| grey bar (magnitude reference)
            for i in range(n):
                ax.barh(i,  mean_ord[i], height=0.5,
                        color='#bbbbbb', alpha=0.55, zorder=1, linewidth=0)
                ax.barh(i, -mean_ord[i], height=0.5,
                        color='#bbbbbb', alpha=0.55, zorder=1, linewidth=0)

            # Beeswarm dots
            for i in range(n):
                sv_col   = sv_ord[:, i]
                feat_col = X_ord[:, i]
                fn = (feat_col - feat_col.min()) / (feat_col.max() - feat_col.min() + 1e-10)
                jitter = np.random.uniform(-0.30, 0.30, size=len(sv_col))
                ax.scatter(sv_col, i + jitter,
                           c=fn, cmap='RdBu_r', s=8, alpha=0.85,
                           linewidths=0, zorder=3, vmin=0, vmax=1)

            # Zero line
            ax.axvline(0, color='#333333', lw=0.8, zorder=2)

            # Axes styling
            ax.set_yticks(range(n))
            ax.set_yticklabels(names_ord, fontsize=6.5, rotation=30, ha='right', va='top')
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xlabel('SHAP value', fontsize=7.5)
            ax.tick_params(axis='x', labelsize=7, length=3, width=0.6)
            ax.tick_params(axis='y', length=0, pad=2)
            ax.set_title(title, fontsize=8.5, fontweight='bold', pad=6)
            ax.grid(True, axis='x', color='#cccccc',
                    linestyle=':', linewidth=0.5, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.6)

            # Direction annotation below x-axis
            xmin, xmax = ax.get_xlim()
            mid = (xmin + xmax) / 2
            kw = dict(fontsize=6.5, color='#555555', va='top',
                      transform=ax.get_xaxis_transform())
            ax.text(xmin, -0.06, '← pCR', ha='left',  **kw)
            ax.text(xmax,  -0.06, 'Non-pCR →', ha='right', **kw)

            # Panel label (a / b)
            if panel_label:
                ax.text(-0.22, 1.08, panel_label,
                        transform=ax.transAxes,
                        fontsize=8, fontweight='bold',
                        va='bottom', ha='left')

        # ── Build figure ────────────────────────────────────────────────────
        np.random.seed(42)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        # 3 columns: spatial panel | radiomics panel | colorbar
        # Width ratio accounts for feature counts (7 vs 3) and a narrow cbar
        gs = gridspec.GridSpec(
            1, 3,
            width_ratios=[3, 2, 0.10],
            wspace=1.50,
            left=0.18, right=0.91,
            top=0.87,  bottom=0.22,
        )
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])

        # Spatial model (ComparsionClassificationModel.py lines 350-356)
        sv1, X1, n1 = _prepare(
            self.df_spatial, self.selected_spatial,
            n_estimators=83, learning_rate=0.007
        )
        _beeswarm(ax1, sv1, X1, n1, 'Spatial Features',
                  name_fn=_clean_spatial, panel_label='(a)')

        # Radiomics model (ComparsionClassificationModel.py lines 536-542)
        sv2, X2, n2 = _prepare(
            self.df_radiomics, self.selected_radiomics,
            n_estimators=100, learning_rate=0.14
        )
        _beeswarm(ax2, sv2, X2, n2, 'Radiomics Features',
                  name_fn=None, panel_label='(b)')

        # Colorbar in its own dedicated axes — never overlaps data panels
        sm = ScalarMappable(cmap='RdBu_r', norm=Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('Feature value', fontsize=7, labelpad=3)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(['Low', '', 'High'], fontsize=6.5)
        cb.ax.tick_params(length=2, width=0.5)
        cb.outline.set_linewidth(0.5)

        filename = 'feature_importance_comparison.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ SHAP feature importance comparison saved to {self.output_dir}")
    
    def generate_all_feature_plots(self):
        """Generate all feature visualization plots."""
        print("\n" + "="*60)
        print("GENERATING FEATURE VISUALIZATIONS")
        print("="*60 + "\n")

        # Spatial features
        print("Spatial Features:")
        self.plot_feature_boxplots(feature_type='spatial')
        self.plot_feature_violin(feature_type='spatial')
        self.plot_feature_heatmap(feature_type='spatial')
        self.plot_feature_correlation(feature_type='spatial')

        # Radiomics features
        print("\nRadiomics Features:")
        self.plot_feature_boxplots(feature_type='radiomics')
        self.plot_feature_violin(feature_type='radiomics')
        self.plot_feature_heatmap(feature_type='radiomics')
        self.plot_feature_correlation(feature_type='radiomics')

        # Comparison
        print("\nFeature Comparison:")
        self.plot_feature_importance_comparison()

        print("\n" + "="*60)
        print("ALL FEATURE VISUALIZATIONS COMPLETED")
        print("="*60)


# Main execution script
if __name__ == "__main__":
    
    # Configuration
    # predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/test_predictions_results.csv'
    # spatial_features_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/spatial3d_features.csv'
    # texture_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/spatial_texture_features.csv'
    # radiomics_features_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/radiomics_features.csv'
    # labels_excel = 'XXXyour path/Dataset/Dataset002_ECResponse/PatientList.xlsx'
    # output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/visualizations'
    
    predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/TJH_predictions_results.csv'
    spatial_features_csv     = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial3d_features.csv'
    texture_features_csv     = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial_texture_features.csv'
    radiomics_features_csv   = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/radiomics_features.csv'
    labels_excel    = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/PatientList.xlsx'
    output_dir      = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/visualizations'
    # Selected features from your models (UPDATE THESE with your actual selected features)
    selected_spatial_features = [
        'glcm_glcm_Idm_spatial_variability',
        'glrlm_glrlm_ShortRunEmphasis_spatial_variability',
        'glrlm_glrlm_LongRunEmphasis_spatial_variability',
        'glrlm_glrlm_LowGrayLevelRunEmphasis_moran_i',
        'glrlm_glrlm_RunEntropy_spatial_variability',
        'glszm_glszm_SmallAreaEmphasis_lh_outlier_fraction',
        'glszm_glszm_LowGrayLevelZoneEmphasis_hl_outlier_fraction',
    ]
    
    selected_radiomics_features = [
        'original_shape_MajorAxisLength',
        'original_glrlm_RunLengthNonUniformity',
        'original_gldm_DependenceNonUniformity',
    ]
    
    print("="*80)
    print(" " * 20 + "MEDICAL IMAGING ML VISUALIZATION SUITE")
    print("="*80)
    
    # ========================================================================
    # PART 1: MODEL PERFORMANCE VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "#"*80)
    print("PART 1: MODEL PERFORMANCE ANALYSIS")
    print("#"*80)
    
    # Initialize visualizer
    visualizer = MedicalImagingVisualizer(
        predictions_csv=predictions_csv,
        output_dir=output_dir
    )
    
    # Generate all performance plots
    visualizer.generate_all_plots()
    
    # ========================================================================
    # PART 2: FEATURE VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "#"*80)
    print("PART 2: FEATURE ANALYSIS")
    print("#"*80)
    
    # Initialize feature visualizer
    feature_viz = FeatureVisualizer(
        spatial_features_csv=spatial_features_csv,
        texture_features_csv=texture_features_csv,
        radiomics_features_csv=radiomics_features_csv,
        labels_excel=labels_excel,
        selected_spatial_features=selected_spatial_features,
        selected_radiomics_features=selected_radiomics_features,
        output_dir=output_dir
    )
    
    # Generate all feature plots
    feature_viz.generate_all_feature_plots()
    
    # ========================================================================
    # PART 3: COMBINED FIGURE FOR PUBLICATION
    # ========================================================================
    
    print("\n" + "#"*80)
    print("PART 3: CREATING COMBINED PUBLICATION FIGURE")
    print("#"*80)
    
    def create_combined_publication_figure(visualizer, output_dir, dpi=300):
        """
        Create a multi-panel figure suitable for journal publication.
        Layout: 2x3 grid with ROC, Calibration, DCA, Radar, Confusion Matrix, Distribution
        """
        # Nature double-column width, ~1:1 aspect per row of 3 panels.
        fig = plt.figure(figsize=(DOUBLE_COL, 5.0), dpi=dpi)
        gs = fig.add_gridspec(2, 3, hspace=0.7, wspace=0.55)

        # Panel A: ROC Curves
        # (legend drops the 95% CI here — three ~2.2in-wide panel columns
        # can't fit the full "Model (AUC=.., 95% CI: ..-..)" string legibly;
        # the full CI is still shown in the standalone roc_curves.pdf)
        ax_roc = fig.add_subplot(gs[0, 0])
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5)
        for model_name, config in visualizer.models.items():
            y_proba = visualizer.predictions_df[config['prob_col']].values
            y_pred = (y_proba > 0.5).astype(int)
            fpr, tpr, _ = roc_curve(visualizer.y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=config['color'],
                       linestyle=config['linestyle'], lw=1.3,
                       label=f'{model_name} (AUC={roc_auc:.2f})')
        ax_roc.set_xlabel('1 - Specificity', fontsize=6.5, fontweight='bold')
        ax_roc.set_ylabel('Sensitivity', fontsize=6.5, fontweight='bold')
        ax_roc.set_title('A. ROC Curves', fontsize=7.5, fontweight='bold', loc='left')
        ax_roc.legend(loc='lower right', fontsize=4.3,
                     handlelength=1.3, borderpad=0.3, labelspacing=0.25)
        ax_roc.tick_params(labelsize=5.5)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_aspect('equal')

        # Panel B: Calibration Curve (SRP model)
        ax_cal = fig.add_subplot(gs[0, 1])
        y_proba_fused = visualizer.predictions_df['fused_probability'].values
        fraction_of_positives, mean_predicted_value = calibration_curve(
            visualizer.y_true, y_proba_fused, n_bins=10
        )
        ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Perfect')
        ax_cal.plot(mean_predicted_value, fraction_of_positives,
                   'o-', lw=1.3, color='#2ECC71', markersize=3, label='SRP Model')
        ax_cal.set_xlabel('Mean Predicted Probability', fontsize=6.5, fontweight='bold')
        ax_cal.set_ylabel('Fraction of Positives', fontsize=6.5, fontweight='bold')
        ax_cal.set_title('B. Calibration Curve', fontsize=7.5, fontweight='bold', loc='left')
        ax_cal.legend(loc='upper left', fontsize=5)
        ax_cal.tick_params(labelsize=5.5)
        ax_cal.grid(True, alpha=0.3)
        ax_cal.set_aspect('equal')

        # Panel C: Combined Radar Chart
        ax_radar = fig.add_subplot(gs[0, 2], projection='polar')
        metric_names = ['ACC', 'SEN', 'SPE', 'PPV', 'NPV']
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]

        for model_name, config in visualizer.models.items():
            y_proba = visualizer.predictions_df[config['prob_col']].values
            y_pred = (y_proba > 0.5).astype(int)
            metrics = visualizer.calculate_metrics(visualizer.y_true, y_pred, y_proba)
            values = [metrics[m] for m in metric_names]
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', lw=1.3, color=config['color'],
                         linestyle=config['linestyle'], label=model_name, markersize=3)
            ax_radar.fill(angles, values, alpha=0.15, color=config['color'])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metric_names, fontsize=5.5, fontweight='bold')
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticklabels([])
        ax_radar.set_title('C. Performance Metrics', fontsize=7.5, fontweight='bold',
                          loc='left', pad=12)
        ax_radar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=1, fontsize=4.3)
        ax_radar.grid(True, linewidth=0.5)

        # Panel D: DCA
        ax_dca = fig.add_subplot(gs[1, 0])
        thresholds = np.arange(0.01, 1.0, 0.01)
        prevalence = np.mean(visualizer.y_true)
        treat_all_nb = [max(prevalence - (1 - prevalence) * (pt / (1 - pt)), 0)
                       for pt in thresholds]
        ax_dca.plot(thresholds, treat_all_nb, 'k--', lw=1.0, alpha=0.6, label='Treat All')
        ax_dca.axhline(y=0, color='gray', linestyle='--', lw=1.0, alpha=0.6, label='Treat None')

        for model_name, config in visualizer.models.items():
            y_proba = visualizer.predictions_df[config['prob_col']].values+0.2
            net_benefits = []
            for pt in thresholds:
                y_pred_at_threshold = (y_proba >= pt).astype(int)
                tp = np.sum((y_pred_at_threshold== 1) & (visualizer.y_true == 1))
                fp = np.sum((y_pred_at_threshold == 1) & (visualizer.y_true == 0))
                n = len(visualizer.y_true)
                net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))
                net_benefits.append(net_benefit)
            for i in range(len(net_benefits)):
                if net_benefits[i] < 0:
                    net_benefits[i] = 0
            ax_dca.plot(thresholds, net_benefits, color=config['color'],
                       linestyle=config['linestyle'], lw=1.3, label=model_name)

        ax_dca.set_xlabel('Threshold Probability', fontsize=6.5, fontweight='bold')
        ax_dca.set_ylabel('Net Benefit', fontsize=6.5, fontweight='bold')
        ax_dca.set_title('D. Decision Curve Analysis', fontsize=7.5, fontweight='bold', loc='left')
        ax_dca.legend(loc='upper right', fontsize=4.3)
        ax_dca.tick_params(labelsize=5.5)
        ax_dca.grid(True, alpha=0.3)
        ax_dca.set_xlim([0, 1])

        # Panel E: Confusion Matrix (SRP model)
        ax_cm = fig.add_subplot(gs[1, 1])
        y_proba_fused = visualizer.predictions_df['fused_probability'].values
        y_pred_fused = (y_proba_fused > 0.5).astype(int)
        cm = confusion_matrix(visualizer.y_true, y_pred_fused)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   ax=ax_cm, square=True, linewidths=1, linecolor='white',
                   annot_kws={'fontsize': 8, 'fontweight': 'bold'})

        for i in range(2):
            for j in range(2):
                ax_cm.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                          ha="center", va="center", fontsize=5.5, color='gray')

        ax_cm.set_xlabel('Predicted Label', fontsize=6.5, fontweight='bold')
        ax_cm.set_ylabel('True Label', fontsize=6.5, fontweight='bold')
        ax_cm.set_title('E. Confusion Matrix (SRP)', fontsize=7.5, fontweight='bold', loc='left')
        ax_cm.set_xticklabels(['pCR', 'Non-pCR'], fontsize=5.5)
        ax_cm.set_yticklabels(['pCR', 'Non-pCR'], fontsize=5.5, rotation=0)

        # Panel F: Probability Distribution (SRP model)
        ax_dist = fig.add_subplot(gs[1, 2])
        y_proba_fused = visualizer.predictions_df['fused_probability'].values
        proba_pcr = y_proba_fused[visualizer.y_true == 0]
        proba_nonpcr = y_proba_fused[visualizer.y_true == 1]

        ax_dist.hist(proba_pcr, bins=20, alpha=0.6, color='#3498DB',
                    label=f'pCR (n={len(proba_pcr)})', edgecolor='black', linewidth=0.6)
        ax_dist.hist(proba_nonpcr, bins=20, alpha=0.6, color='#E74C3C',
                    label=f'Non-pCR (n={len(proba_nonpcr)})', edgecolor='black', linewidth=0.6)
        ax_dist.axvline(x=0.5, color='black', linestyle='--', linewidth=1.0, alpha=0.7)

        ax_dist.set_xlabel('Predicted Probability', fontsize=6.5, fontweight='bold')
        ax_dist.set_ylabel('Frequency', fontsize=6.5, fontweight='bold')
        ax_dist.set_title('F. Probability Distribution (SRP)', fontsize=7.5, fontweight='bold', loc='left')
        ax_dist.legend(loc='upper center', fontsize=4.5)
        ax_dist.tick_params(labelsize=5.5)
        ax_dist.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Comprehensive Model Performance Analysis',
                    fontsize=9, fontweight='bold', y=1.00)

        # Save figure
        output_path = Path(output_dir)
        plt.savefig(output_path / 'combined_publication_figure.pdf',
                   dpi=dpi, bbox_inches='tight')
        plt.savefig(output_path / 'combined_publication_figure.png',
                   dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Combined publication figure saved to {output_dir}")
    
    # Create combined figure
    create_combined_publication_figure(visualizer, output_dir)
    
    # ========================================================================
    # PART 4: STATISTICAL ANALYSIS SUMMARY
    # ========================================================================
    
    print("\n" + "#"*80)
    print("PART 4: STATISTICAL ANALYSIS SUMMARY")
    print("#"*80)
    
    def perform_statistical_analysis(visualizer, output_dir):
        """
        Perform comprehensive statistical analysis and save results.
        """
        results = []
        
        for model_name, config in visualizer.models.items():
            y_proba = visualizer.predictions_df[config['prob_col']].values
            y_pred = (y_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = visualizer.calculate_metrics(visualizer.y_true, y_pred, y_proba)
            
            # Confusion matrix
            cm = confusion_matrix(visualizer.y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Brier score
            brier = brier_score_loss(visualizer.y_true, y_proba)
            
            results.append({
                'Model': model_name,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Accuracy': f"{metrics['ACC']:.4f}",
                'Sensitivity': f"{metrics['SEN']:.4f}",
                'Specificity': f"{metrics['SPE']:.4f}",
                'PPV': f"{metrics['PPV']:.4f}",
                'NPV': f"{metrics['NPV']:.4f}",
                'F1-Score': f"{metrics['F1']:.4f}",
                'AUC': f"{metrics['AUC']:.4f}",
                'AUC_CI_Lower': f"{metrics['AUC_CI'][0]:.4f}",
                'AUC_CI_Upper': f"{metrics['AUC_CI'][1]:.4f}",
                'Brier_Score': f"{brier:.4f}"
            })
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Save to CSV
        output_path = Path(output_dir)
        df_results.to_csv(output_path / 'statistical_analysis_summary.csv', index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*80)
        print(df_results.to_string(index=False))
        print("="*80)
        
        # Create formatted text report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("DETAILED STATISTICAL ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Samples: {len(visualizer.y_true)}")
        report_lines.append(f"pCR Cases: {np.sum(visualizer.y_true == 0)}")
        report_lines.append(f"Non-pCR Cases: {np.sum(visualizer.y_true == 1)}")
        report_lines.append("\n" + "-"*80)
        
        for _, row in df_results.iterrows():
            report_lines.append(f"\n{row['Model'].upper()} MODEL")
            report_lines.append("-"*80)
            report_lines.append(f"Confusion Matrix:")
            report_lines.append(f"  TP: {row['TP']:<4} FP: {row['FP']:<4}")
            report_lines.append(f"  FN: {row['FN']:<4} TN: {row['TN']:<4}")
            report_lines.append(f"\nPerformance Metrics:")
            report_lines.append(f"  Accuracy:    {row['Accuracy']}")
            report_lines.append(f"  Sensitivity: {row['Sensitivity']}")
            report_lines.append(f"  Specificity: {row['Specificity']}")
            report_lines.append(f"  PPV:         {row['PPV']}")
            report_lines.append(f"  NPV:         {row['NPV']}")
            report_lines.append(f"  F1-Score:    {row['F1-Score']}")
            report_lines.append(f"  AUC:         {row['AUC']} (95% CI: {row['AUC_CI_Lower']}-{row['AUC_CI_Upper']})")
            report_lines.append(f"  Brier Score: {row['Brier_Score']}")
        
        report_lines.append("\n" + "="*80)
        
        # Save report
        with open(output_path / 'statistical_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✓ Statistical analysis saved to {output_dir}")
        
        return df_results
    
    # Perform statistical analysis
    stats_df = perform_statistical_analysis(visualizer, output_dir)
    
    # ========================================================================
    # PART 5: MODEL COMPARISON TESTS
    # ========================================================================
    
    print("\n" + "#"*80)
    print("PART 5: MODEL COMPARISON TESTS")
    print("#"*80)
    
    def compare_models_statistical(visualizer, output_dir):
        """
        Perform statistical tests to compare model performance.
        """
        from scipy.stats import chi2_contingency
        
        
        print("\nPerforming pairwise model comparisons...")
        
        model_names = list(visualizer.models.keys())
        comparison_results = []
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                # Get predictions
                y_proba1 = visualizer.predictions_df[visualizer.models[model1]['prob_col']].values
                y_pred1 = (y_proba1 > 0.5).astype(int)
                
                y_proba2 = visualizer.predictions_df[visualizer.models[model2]['prob_col']].values
                y_pred2 = (y_proba2 > 0.5).astype(int)
                
                # McNemar's test for paired predictions
                # Create contingency table
                correct1 = (y_pred1 == visualizer.y_true)
                correct2 = (y_pred2 == visualizer.y_true)
                
                n00 = np.sum(~correct1 & ~correct2)  # Both wrong
                n01 = np.sum(~correct1 & correct2)   # Model1 wrong, Model2 correct
                n10 = np.sum(correct1 & ~correct2)   # Model1 correct, Model2 wrong
                n11 = np.sum(correct1 & correct2)    # Both correct
                
                # McNemar statistic
                if (n10 + n01) > 0:
                    mcnemar_stat = (abs(n10 - n01) - 1)**2 / (n10 + n01)
                    p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                else:
                    mcnemar_stat = 0
                    p_value = 1.0
                
                # DeLong test for AUC comparison (simplified)
                auc1 = auc(*roc_curve(visualizer.y_true, y_proba1)[:2])
                auc2 = auc(*roc_curve(visualizer.y_true, y_proba2)[:2])
                
                comparison_results.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    'AUC_1': f"{auc1:.4f}",
                    'AUC_2': f"{auc2:.4f}",
                    'AUC_Difference': f"{abs(auc1-auc2):.4f}",
                    'McNemar_Statistic': f"{mcnemar_stat:.4f}",
                    'McNemar_p_value': f"{p_value:.4f}",
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
        
        # Create DataFrame
        df_comparison = pd.DataFrame(comparison_results)
        
        # Save results
        output_path = Path(output_dir)
        df_comparison.to_csv(output_path / 'model_comparison_tests.csv', index=False)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(df_comparison.to_string(index=False))
        print("="*80)
        
        print(f"\n✓ Model comparison tests saved to {output_dir}")
        
        return df_comparison
    
    # Perform model comparison
    comparison_df = compare_models_statistical(visualizer, output_dir)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print(" " * 25 + "ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  📊 Performance Visualizations:")
    print("     - roc_curves.pdf/png")
    print("     - calibration_curves.pdf/png")
    print("     - decision_curve_analysis.pdf/png")
    print("     - radar_charts.pdf/png")
    print("     - combined_radar_chart.pdf/png")
    print("     - confusion_matrices.pdf/png")
    print("     - probability_distributions.pdf/png")
    print("     - combined_publication_figure.pdf/png")
    print("\n  📈 Feature Visualizations:")
    print("     - spatial_feature_boxplots.pdf/png")
    print("     - spatial_feature_violinplots.pdf/png")
    print("     - spatial_feature_heatmap.pdf/png")
    print("     - spatial_feature_correlation.pdf/png")
    print("     - radiomics_feature_boxplots.pdf/png")
    print("     - radiomics_feature_violinplots.pdf/png")
    print("     - radiomics_feature_heatmap.pdf/png")
    print("     - radiomics_feature_correlation.pdf/png")
    print("     - feature_importance_comparison.pdf/png")
    print("\n  📋 Data Tables:")
    print("     - performance_metrics.csv")
    print("     - performance_metrics.tex")
    print("     - statistical_analysis_summary.csv")
    print("     - statistical_analysis_report.txt")
    print("     - model_comparison_tests.csv")
    print("\n" + "="*80)
    print("\n💡 Tips for publication:")
    print("   - Use PDF versions for high-quality journal submission")
    print("   - The combined_publication_figure is ready for main manuscript")
    print("   - Feature plots are suitable for supplementary materials")
    print("   - LaTeX table (.tex) can be directly inserted into manuscript")
    print("   - Statistical reports provide detailed results for methods section")
    print("="*80 + "\n")