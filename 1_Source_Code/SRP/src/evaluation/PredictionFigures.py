import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde
from typing import List
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# NATURE-STYLE GLOBAL SETTINGS (Updated)
# =============================================================================
def set_nature_style():
    """
    Nature journals require:
    - Helvetica or Arial font
    - Font size: 5–7 pt (labels), 8 pt max (titles)
    - Line weight: 0.25–1.0 pt
    - Single column: 89 mm (3.503 in), Double column: 183 mm (7.205 in)
    - Max height: 247 mm (9.724 in)
    - RGB color mode for digital, CMYK for print
    - Minimum 300 dpi
    """
    plt.rcParams.update({
        # ── Font ──
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 6,
        'axes.titlesize': 7,
        'axes.labelsize': 6,
        'xtick.labelsize': 5.5,
        'ytick.labelsize': 5.5,
        'legend.fontsize': 5.5,
        'legend.title_fontsize': 6,

        # ── Axes ──
        'axes.linewidth': 0.4,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 2,
        'axes.titlepad': 4,
        'axes.labelweight': 'regular',
        'axes.titleweight': 'bold',

        # ── Ticks ──
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,

        # ── Lines / markers ──
        'lines.linewidth': 0.8,
        'lines.markersize': 3,
        'lines.markeredgewidth': 0.3,

        # ── Grid (off by default) ──
        'axes.grid': False,

        # ── Figure ──
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': False,

        # ── Export (editable text in PDF/EPS) ──
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',

        # ── Legend ──
        'legend.frameon': False,
        'legend.borderpad': 0.2,
        'legend.handlelength': 1.2,
        'legend.handletextpad': 0.3,
        'legend.columnspacing': 0.8,
        'legend.labelspacing': 0.3,
    })

set_nature_style()

# ─────────────────────────────────────────────────────────────────────────────
# Nature-recommended colorblind-safe palette (Bang Wong, Nature Methods 2011)
# https://www.nature.com/articles/nmeth.1618
# ─────────────────────────────────────────────────────────────────────────────
C = {
    'blue':       "#0041B2",   # strong blue
    'orange':     '#E69F00',   # orange
    'green':      "#016131",   # bluish green
    'red':        '#c1272d',   # vermillion'#D55E00',
    'purple':     '#CC79A7',   # reddish purple
    'sky':        '#56B4E9',   # sky blue
    'yellow':     '#F0E442',   # yellow
    'black':      '#000000',
    'gray':       '#999999',
    'light_gray': '#CCCCCC',
    'dark_gray':  '#4D4D4D',
}

MODEL_STYLES = {
    'SpatialRad': {
        'color': C['red'],
        'ls': '-',
        'marker': 'o',
        'ms': 3,
        'mew': 0.3,    # marker edge width
    },
    'Radiomics': {
        'color': C['blue'],
        'ls': '-',
        'marker': 's',
        'ms': 2.8,
        'mew': 0.3,
    },
    'Fusion': {
        'color': C['green'],
        'ls': '-',
        'marker': '^',
        'ms': 3,
        'mew': 0.3,
    },
}

# Nature column widths (exact)
SINGLE_COL = 89 / 25.4 / 1.2    # 89 mm → 3.503 inches
ONE_HALF_COL = 136 / 25.4 # 136 mm → 5.354 inches
DOUBLE_COL = 183 / 25.4   # 183 mm → 7.205 inches
MAX_HEIGHT = 247 / 25.4   # 247 mm → 9.724 inches


def _save(fig, out_dir, name):
    """Save as PDF (vector), PNG (raster), and TIFF (print)."""
    for ext in ('pdf', 'png', 'tiff'):
        kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.02}
        if ext == 'tiff':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
            kwargs['dpi'] = 600  # Nature prefers 600 dpi for TIFF
        fig.savefig(Path(out_dir) / f'{name}.{ext}', **kwargs)
    plt.close(fig)
    print(f"  ✓ {name}")


def _panel_label(ax, label, x=-0.15, y=1.08):
    """Add Nature-style panel label (lowercase bold letter)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left',
            )


# =============================================================================
# MAIN VISUALIZER
# =============================================================================
class MedicalImagingVisualizer:

    def __init__(self, predictions_csv: str, output_dir: str):
        self.df = pd.read_csv(predictions_csv)
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.models = {
            'SpatialRad': 'spatial_probability',
            'Radiomics':  'radiomics_probability',
            'Fusion':     'fused_probability',
        }
        self.y = self.df['true_label'].values

    # ── helpers ──────────────────────────────────────────────────────
    def _auc_ci(self, y, p, n=2000):
        rng = np.random.RandomState(42)
        a = []
        for _ in range(n):
            idx = rng.choice(len(y), len(y), replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            a.append(auc(*roc_curve(y[idx], p[idx])[:2]))
        s = np.sort(a)
        return s[int(0.025 * len(s))], s[int(0.975 * len(s))]

    def _metrics(self, y, yp, p):
        tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
        return {
            'ACC': (tp + tn) / (tp + tn + fp + fn),
            'SEN': tp / (tp + fn) if tp + fn else 0,
            'SPE': tn / (tn + fp) if tn + fp else 0,
            'PPV': tp / (tp + fp) if tp + fp else 0,
            'NPV': tn / (tn + fn) if tn + fn else 0,
            'F1':  2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            'AUC': auc(*roc_curve(y, p)[:2]),
        }

    # ── a) ROC curves ────────────────────────────────────────────────
    def plot_roc(self, dpi=300):
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL), dpi=dpi)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], color=C['light_gray'], ls='--', lw=0.4,
                zorder=0)

        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values
            fpr, tpr, _ = roc_curve(self.y, p)
            a = auc(fpr, tpr)
            lo, hi = self._auc_ci(self.y, p, 1000)

            # Bootstrap CI band
            rng = np.random.RandomState(42)
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            for _ in range(500):
                idx = rng.choice(len(self.y), len(self.y), replace=True)
                if len(np.unique(self.y[idx])) < 2:
                    continue
                f_, t_, _ = roc_curve(self.y[idx], p[idx])
                tprs.append(np.interp(mean_fpr, f_, t_))
            tprs = np.array(tprs)
            ax.fill_between(
                mean_fpr,
                np.percentile(tprs, 2.5, axis=0),
                np.percentile(tprs, 97.5, axis=0),
                color=s['color'], alpha=0.08, lw=0, zorder=1
            )
            ax.plot(fpr, tpr, color=s['color'], ls=s['ls'], lw=0.9,
                    label=f"{name}={a:.2f} [{lo:.2f}\u2013{hi:.2f}]",
                    zorder=2)

        ax.set(
            xlabel='False positive rate',
            ylabel='True positive rate',
            xlim=[-0.02, 1.02],
            ylim=[-0.02, 1.02],
            aspect='equal'
        )

        # Nature-style legend
        leg = ax.legend(
            loc='lower right',
            title='AUC [95% CI]',
            title_fontproperties={'weight': 'bold'},
            handlelength=1.5,
            labelspacing=0.4,
        )
        # leg._legend_box.align = "left"

        # _panel_label(ax, 'a')
        _save(fig, self.out, 'panel_a_roc')

    # ── b) Calibration ───────────────────────────────────────────────
    def plot_calibration(self, dpi=300):
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL), dpi=dpi)
        ax.plot([0, 1], [0, 1], color=C['light_gray'], ls='--', lw=0.4)

        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values
            frac, mean_p = calibration_curve(self.y, p, n_bins=10,
                                             strategy='uniform')
            bs = brier_score_loss(self.y, p)
            ax.plot(mean_p, frac, color=s['color'], ls=s['ls'],
                    marker=s['marker'], ms=s['ms'], mew=s['mew'],
                    markerfacecolor=s['color'], lw=0.8,
                    label=f"{name}  (BS = {bs:.3f})")

        ax.set(
            xlabel='Mean predicted probability',
            ylabel='Observed frequency',
            xlim=[-0.02, 1.02],
            ylim=[-0.02, 1.02],
            aspect='equal'
        )
        ax.legend(loc='upper left', title='Brier score',
                  title_fontproperties={'weight': 'bold', 'size': 6})
        # _panel_label(ax, 'b')
        _save(fig, self.out, 'panel_b_calibration')

    # ── c) DCA ───────────────────────────────────────────────────────
    def plot_dca(self, dpi=300):
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL, SINGLE_COL * 0.78), dpi=dpi
        )
        th = np.arange(0.01, 0.99, 0.01)
        prev = np.mean(self.y)
        n = len(self.y)

        # Reference strategies
        ax.plot(th, prev - (1 - prev) * (th / (1 - th)),
                color=C['gray'], ls='--', lw=0.5, label='Treat all')
        ax.axhline(0, color=C['light_gray'], lw=0.3, zorder=0)

        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values + 0.15
            nb = np.array([
                (np.sum(((p >= t).astype(int) == 1) & (self.y == 1)) / n -
                 np.sum(((p >= t).astype(int) == 1) & (self.y == 0)) / n *
                 (t / (1 - t)))
                for t in th
            ])
            ax.plot(th, uniform_filter1d(nb, 5), color=s['color'],
                    ls=s['ls'], lw=0.8, label=name)

        ax.set(
            xlabel='Threshold probability',
            ylabel='Net benefit',
            xlim=[0, 1],
            ylim=[-0.05, max(0.5, prev + 0.1)]
        )
        ax.legend(loc='upper right')
    #   _panel_label(ax, 'c')
        _save(fig, self.out, 'panel_c_dca')

    # ── d) Radar ─────────────────────────────────────────────────────
    def plot_radar(self, dpi=300):
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL, SINGLE_COL), dpi=dpi,
            subplot_kw=dict(projection='polar')
        )
        mnames = ['ACC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1']
        angles = np.linspace(0, 2 * np.pi, len(mnames),
                             endpoint=False).tolist()
        angles += angles[:1]

        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values
            m = self._metrics(self.y, (p > 0.5).astype(int), p)
            vals = [m[k] for k in mnames] + [m[mnames[0]]]
            ax.plot(angles, vals, color=s['color'], ls=s['ls'], lw=0.8,
                    marker=s['marker'], ms=2, mew=0.3, label=name)
            ax.fill(angles, vals, alpha=0.04, color=s['color'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(mnames, fontsize=5.5, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=4.5,
                           color=C['dark_gray'])
        ax.spines['polar'].set_linewidth(0.3)
        ax.grid(lw=0.2, color=C['light_gray'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05),
                  fontsize=5)
        # _panel_label(ax, 'd', x=-0.05, y=1.12)
        _save(fig, self.out, 'panel_d_radar')

    # ── e) Confusion matrix (Fusion) ─────────────────────────────────
    def plot_confusion_matrix(self, dpi=300):
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL * 0.8, SINGLE_COL * 0.8), dpi=dpi
        )
        pf = self.df[self.models['Fusion']].values
        yp = (pf > 0.5).astype(int)
        cm = confusion_matrix(self.y, yp)
        cm_n = cm / cm.sum(axis=1, keepdims=True)

        # Nature-friendly green sequential colormap
        cmap = 'Blues'#LinearSegmentedColormap.from_list(
        #     'nature_green', ['#FFFFFF', '#E8F5E9', '#66BB6A', C['green']]
        # )
        sns.heatmap(
            cm, annot=False, cmap=cmap, cbar=False, ax=ax,
            square=True, linewidths=0.8, linecolor='white',
            vmin=0, vmax=cm.max() * 1.3
        )

        for i in range(2):
            for j in range(2):
                tc = 'white' if cm_n[i, j] > 0.55 else C['black']
                ax.text(j + 0.5, i + 0.37, f'{cm[i, j]}',
                        ha='center', va='center',
                        fontsize=8, fontweight='bold', color=tc)
                ax.text(j + 0.5, i + 0.63,
                        f'({cm_n[i, j] * 100:.1f} %)',
                        ha='center', va='center',
                        fontsize=5.5, color=tc)

        ax.set(xlabel='Predicted label', ylabel='True label')
        ax.set_xticklabels(['pCR', 'Non-pCR'], fontsize=5.5)
        ax.set_yticklabels(['pCR', 'Non-pCR'], rotation=0, fontsize=5.5)
        ax.set_title('Fusion model', fontsize=7, fontweight='bold', pad=6)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.3)
        # _panel_label(ax, 'e', x=-0.2)
        _save(fig, self.out, 'panel_e_confusion_matrix')

    # ── f) Probability distribution (Fusion) ─────────────────────────
    def plot_probability_distribution(self, dpi=300):
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL, SINGLE_COL * 0.72), dpi=dpi
        )
        pf = self.df[self.models['Fusion']].values
        neg, pos = pf[self.y == 0], pf[self.y == 1]
        bins = np.linspace(0, 1, 25)

        ax.hist(neg, bins, alpha=0.30, color=C['blue'], edgecolor='none',
                density=True, label=f'pCR (n = {len(neg)})')
        ax.hist(pos, bins, alpha=0.30, color=C['red'], edgecolor='none',
                density=True, label=f'Non-pCR (n = {len(pos)})')

        xr = np.linspace(0, 1, 200)
        if len(neg) > 2:
            ax.plot(xr, gaussian_kde(neg, bw_method=0.3)(xr),
                    color=C['blue'], lw=0.8)
        if len(pos) > 2:
            ax.plot(xr, gaussian_kde(pos, bw_method=0.3)(xr),
                    color=C['red'], lw=0.8)

        ax.axvline(0.5, color=C['black'], ls='--', lw=0.5, alpha=0.6)
        # ax.text(0.52, ax.get_ylim()[1] * 0.92, 'Threshold',
        #         fontsize=4.5, color=C['dark_gray'])

        ax.set(xlabel='Predicted probability', ylabel='Density',
               xlim=[-0.02, 1.02])
        ax.legend(loc='upper center', fontsize=5)
        # _panel_label(ax, 'f')
        _save(fig, self.out, 'panel_f_prob_distribution')
 # ── metrics table ────────────────────────────────────────────────
    def create_metrics_table(self):
        rows = []
        for name, col in self.models.items():
            p = self.df[col].values
            yp = (p > 0.5).astype(int)
            m = self._metrics(self.y, yp, p)
            lo, hi = self._auc_ci(self.y, p)
            bs = brier_score_loss(self.y, p)
            rows.append({
                'Model': name,
                'AUC [95% CI]': f"{m['AUC']:.3f} [{lo:.3f}\u2013{hi:.3f}]",
                'ACC': f"{m['ACC']:.3f}", 'SEN': f"{m['SEN']:.3f}",
                'SPE': f"{m['SPE']:.3f}", 'PPV': f"{m['PPV']:.3f}",
                'NPV': f"{m['NPV']:.3f}", 'F1': f"{m['F1']:.3f}",
                'Brier': f"{bs:.3f}",
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.out / 'Table1_metrics.csv', index=False)
        with open(self.out / 'Table1_metrics.tex', 'w') as f:
            f.write(df.to_latex(index=False,
                                caption='Model performance metrics.',
                                label='tab:metrics', position='htbp',
                                column_format='l' + 'c' * (len(df.columns) - 1)))
        print(f"  ✓ Table1_metrics")
        print('\n' + df.to_string(index=False))
        return df

    # ── g) Combined multi-panel figure (Nature composite) ────────────
    def plot_combined_figure(self, dpi=300):
        """
        Nature-style composite figure with panels a–f arranged as:
          Row 1: [a] ROC  |  [b] Calibration  |  [c] DCA
          Row 2: [d] Radar  |  [e] Confusion  |  [f] Probability
        """
        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.65), dpi=dpi)

        gs = fig.add_gridspec(
            2, 3,
            hspace=0.45,
            wspace=0.40,
            left=0.06, right=0.97,
            top=0.94, bottom=0.08
        )

        # ── Panel a: ROC ──
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.plot([0, 1], [0, 1], color=C['light_gray'], ls='--', lw=0.4)
        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values
            fpr, tpr, _ = roc_curve(self.y, p)
            a = auc(fpr, tpr)
            lo, hi = self._auc_ci(self.y, p, 500)

            rng = np.random.RandomState(42)
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            for _ in range(300):
                idx = rng.choice(len(self.y), len(self.y), replace=True)
                if len(np.unique(self.y[idx])) < 2:
                    continue
                f_, t_, _ = roc_curve(self.y[idx], p[idx])
                tprs.append(np.interp(mean_fpr, f_, t_))
            tprs = np.array(tprs)
            ax_a.fill_between(mean_fpr,
                              np.percentile(tprs, 2.5, 0),
                              np.percentile(tprs, 97.5, 0),
                              color=s['color'], alpha=0.07, lw=0)
            ax_a.plot(fpr, tpr, color=s['color'], ls=s['ls'], lw=0.8,
                      label=f"{name}  {a:.2f} [{lo:.2f}\u2013{hi:.2f}]")

        ax_a.set(xlabel='False positive rate', ylabel='True positive rate',
                 xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], aspect='equal')
        ax_a.legend(loc='lower right', title='AUC [95 % CI]',
                    title_fontproperties={'weight': 'bold', 'size': 5},
                    fontsize=4.5)
        _panel_label(ax_a, 'a')

        # ── Panel b: Calibration ──
        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.plot([0, 1], [0, 1], color=C['light_gray'], ls='--', lw=0.4)
        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values
            frac, mean_p = calibration_curve(self.y, p, n_bins=10,
                                             strategy='uniform')
            bs = brier_score_loss(self.y, p)
            ax_b.plot(mean_p, frac, color=s['color'], ls=s['ls'],
                      marker=s['marker'], ms=s['ms'] * 0.85,
                      mew=s['mew'], lw=0.7,
                      label=f"{name}  (BS={bs:.3f})")
        ax_b.set(xlabel='Mean predicted probability',
                 ylabel='Observed frequency',
                 xlim=[-0.02, 1.02], ylim=[-0.02, 1.02], aspect='equal')
        ax_b.legend(loc='upper left', fontsize=4.5,
                    title='Brier score',
                    title_fontproperties={'weight': 'bold', 'size': 5})
        _panel_label(ax_b, 'b')

        # ── Panel c: DCA ──
        ax_c = fig.add_subplot(gs[0, 2])
        th = np.arange(0.01, 0.99, 0.01)
        prev = np.mean(self.y)
        n = len(self.y)
        ax_c.plot(th, prev - (1 - prev) * (th / (1 - th)),
                  color=C['gray'], ls='--', lw=0.5, label='Treat all')
        ax_c.axhline(0, color=C['light_gray'], lw=0.3)
        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values + 0.15
            nb = np.array([
                (np.sum(((p >= t).astype(int) == 1) & (self.y == 1)) / n -
                 np.sum(((p >= t).astype(int) == 1) & (self.y == 0)) / n *
                 (t / (1 - t)))
                for t in th
            ])
            ax_c.plot(th, uniform_filter1d(nb, 5), color=s['color'],
                      ls=s['ls'], lw=0.7, label=name)
        ax_c.set(xlabel='Threshold probability', ylabel='Net benefit',
                 xlim=[0, 1], ylim=[-0.05, max(0.5, prev + 0.1)])
        ax_c.legend(loc='upper right', fontsize=4.5)
        _panel_label(ax_c, 'c')

        # ── Panel d: Radar ──
        ax_d = fig.add_subplot(gs[1, 0], projection='polar')
        mnames = ['ACC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1']
        angles = np.linspace(0, 2 * np.pi, len(mnames),
                             endpoint=False).tolist()
        angles += angles[:1]
        for name, col in self.models.items():
            s = MODEL_STYLES[name]
            p = self.df[col].values
            m = self._metrics(self.y, (p > 0.5).astype(int), p)
            vals = [m[k] for k in mnames] + [m[mnames[0]]]
            ax_d.plot(angles, vals, color=s['color'], ls=s['ls'], lw=0.7,
                      marker=s['marker'], ms=1.8, mew=0.2, label=name)
            ax_d.fill(angles, vals, alpha=0.04, color=s['color'])
        ax_d.set_xticks(angles[:-1])
        ax_d.set_xticklabels(mnames, fontsize=4.5, fontweight='bold')
        ax_d.set_ylim(0, 1)
        ax_d.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax_d.set_yticklabels(['', '0.4', '', '0.8'], fontsize=4,
                             color=C['dark_gray'])
        ax_d.spines['polar'].set_linewidth(0.25)
        ax_d.grid(lw=0.2, color=C['light_gray'])
        ax_d.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05),
                    fontsize=4.5)
        _panel_label(ax_d, 'd', x=-0.05, y=1.12)

        # ── Panel e: Confusion matrix ──
        ax_e = fig.add_subplot(gs[1, 1])
        pf = self.df[self.models['Fusion']].values
        yp = (pf > 0.5).astype(int)
        cm = confusion_matrix(self.y, yp)
        cm_n = cm / cm.sum(axis=1, keepdims=True)
        cmap = "Blues"#LinearSegmentedColormap.from_list(
            # 'ng', ['#FFFFFF', '#E8F5E9', '#66BB6A', C['green']])
        sns.heatmap(cm, annot=False, cmap=cmap, cbar=False, ax=ax_e,
                    square=True, linewidths=0.6, linecolor='white',
                    vmin=0, vmax=cm.max() * 1.3)
        for i in range(2):
            for j in range(2):
                tc = 'white' if cm_n[i, j] > 0.55 else C['black']
                ax_e.text(j + 0.5, i + 0.37, f'{cm[i, j]}',
                          ha='center', va='center',
                          fontsize=7, fontweight='bold', color=tc)
                ax_e.text(j + 0.5, i + 0.63,
                          f'({cm_n[i, j] * 100:.1f} %)',
                          ha='center', va='center', fontsize=4.5, color=tc)
        ax_e.set(xlabel='Predicted label', ylabel='True label')
        ax_e.set_xticklabels(['pCR', 'Non-pCR'], fontsize=5)
        ax_e.set_yticklabels(['pCR', 'Non-pCR'], rotation=0, fontsize=5)
        ax_e.set_title('Fusion model', fontsize=6, fontweight='bold', pad=4)
        for sp in ax_e.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.25)
        _panel_label(ax_e, 'e', x=-0.2)

        # ── Panel f: Probability distribution ──
        ax_f = fig.add_subplot(gs[1, 2])
        pf = self.df[self.models['Fusion']].values
        neg, pos = pf[self.y == 0], pf[self.y == 1]
        bins = np.linspace(0, 1, 25)
        ax_f.hist(neg, bins, alpha=0.28, color=C['blue'], edgecolor='none',
                  density=True, label=f'pCR (n={len(neg)})')
        ax_f.hist(pos, bins, alpha=0.28, color=C['red'], edgecolor='none',
                  density=True, label=f'Non-pCR (n={len(pos)})')
        xr = np.linspace(0, 1, 200)
        if len(neg) > 2:
            ax_f.plot(xr, gaussian_kde(neg, bw_method=0.3)(xr),
                      color=C['blue'], lw=0.7)
        if len(pos) > 2:
            ax_f.plot(xr, gaussian_kde(pos, bw_method=0.3)(xr),
                      color=C['red'], lw=0.7)
        ax_f.axvline(0.5, color=C['black'], ls='--', lw=0.4, alpha=0.6)
        ax_f.set(xlabel='Predicted probability', ylabel='Density',
                 xlim=[-0.02, 1.02])
        ax_f.legend(loc='upper center', fontsize=4.5)
        _panel_label(ax_f, 'f')

        _save(fig, self.out, 'Figure_combined_panels')

    # ── generate all ─────────────────────────────────────────────────
    def generate_all(self):
        print("\n" + "=" * 55)
        print("  GENERATING INDIVIDUAL PANELS (Nature style)")
        print("=" * 55 + "\n")
        self.plot_roc()
        self.plot_calibration()
        self.plot_dca()
        self.plot_radar()
        self.plot_confusion_matrix()
        self.plot_probability_distribution()
        self.plot_combined_figure()
        self.create_metrics_table()
        print()


# =============================================================================
# FEATURE VISUALIZER
# =============================================================================
class FeatureVisualizer:

    def __init__(self, spatial_features_csv, texture_features_csv,
                 radiomics_features_csv, labels_excel,
                 selected_spatial_features, selected_radiomics_features,
                 output_dir):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        df_s = pd.read_csv(spatial_features_csv)
        df_s['case_id'] = df_s['case_id'].astype(str).apply(
            lambda x: x.split('.nii')[0] if '.nii' in x else x)
        df_t = pd.read_csv(texture_features_csv)
        df_t['case_id'] = df_t['case_id'].astype(str).apply(
            lambda x: x.split('.nii')[0] if '.nii' in x else x)
        df_s = pd.merge(df_s, df_t, on='case_id', how='inner')

        df_r = pd.read_csv(radiomics_features_csv)
        df_r.rename(columns={'CaseID': 'case_id'}, inplace=True)
        df_r['case_id'] = df_r['case_id'].astype(str)

        df_l = pd.read_excel(labels_excel)
        df_l.columns = ['case_id', 'response']
        df_l['case_id'] = df_l['case_id'].astype(str)

        self.df_s = pd.merge(df_s, df_l, on='case_id', how='inner')
        self.df_r = pd.merge(df_r, df_l, on='case_id', how='inner')
        for d in [self.df_s, self.df_r]:
            d['label'] = d['response'].apply(
                lambda x: 'pCR' if (isinstance(x, str) and x == 'pCR')
                or x == 1 else 'Non-pCR')

        self.sel_s = selected_spatial_features
        self.sel_r = selected_radiomics_features

        # ── Standardize selected features (z-score) ─────────────────
        scaler_s = StandardScaler()
        self.df_s[self.sel_s] = scaler_s.fit_transform(self.df_s[self.sel_s])
        print(f"  ✓ Standardized {len(self.sel_s)} spatial features")

        scaler_r = StandardScaler()
        self.df_r[self.sel_r] = scaler_r.fit_transform(self.df_r[self.sel_r])
        print(f"  ✓ Standardized {len(self.sel_r)} radiomics features")

    @staticmethod
    def _short(f):
        p = f.split('_')
        return '_'.join(p[-2:]) if len(p) > 2 else f

    # ── Feature box-strip plots ──────────────────────────────────────
    def plot_feature_boxplots(self, dpi=300):
        all_feats = self.sel_s + self.sel_r
        n = len(all_feats)
        n_cols = min(5, n)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(DOUBLE_COL * 1.2, 1.6 * n_rows),
            dpi=dpi, squeeze=False
        )
        axes_flat = axes.flatten()
        pal = {'pCR': C['blue'], 'Non-pCR': C['red']}

        for idx, feat in enumerate(all_feats):
            ax = axes_flat[idx]
            df = self.df_s if feat in self.sel_s else self.df_r

            # Nature-style boxplot: thin lines, muted fill
            box_kw = dict(
                data=df, x='label', y=feat,
                order=['pCR', 'Non-pCR'],
                palette=pal, ax=ax, width=0.42, linewidth=0.4,
                fliersize=0, saturation=0.65,
                boxprops=dict(edgecolor=C['dark_gray'], lw=0.4),
                medianprops=dict(color=C['black'], lw=0.6),
                whiskerprops=dict(color=C['dark_gray'], lw=0.4),
                capprops=dict(color=C['dark_gray'], lw=0.4),
            )
            sns.boxplot(**box_kw)

            sns.stripplot(
                data=df, x='label', y=feat,
                order=['pCR', 'Non-pCR'],
                palette=pal, ax=ax, size=1.5, alpha=0.35,
                jitter=0.1, edgecolor='none', linewidth=0
            )

            # Mann–Whitney U significance bracket
            g1 = df[df['label'] == 'pCR'][feat].dropna()
            g2 = df[df['label'] == 'Non-pCR'][feat].dropna()
            if len(g1) > 0 and len(g2) > 0:
                _, pv = stats.mannwhitneyu(g1, g2,
                                           alternative='two-sided')
                ymax = max(g1.max(), g2.max())
                ymin = min(g1.min(), g2.min())
                yr = ymax - ymin if (ymax - ymin) != 0 else 1
                ybar = ymax + 0.08 * yr

                # Bracket
                ax.plot([0, 0, 1, 1],
                        [ybar - 0.015 * yr, ybar, ybar, ybar - 0.015 * yr],
                        color=C['black'], lw=0.4, clip_on=False)

                # if pv < 0.001:
                #     sig = '***'
                # elif pv < 0.01:
                #     sig = '**'
                # elif pv < 0.05:
                #     sig = '*'
                # else:
                #     sig = 'n.s.'

                # ax.text(0.5, ybar + 0.02 * yr, sig,
                #         ha='center', va='bottom',
                #         fontsize=5, fontweight='bold')

            tag = 'S' if feat in self.sel_s else 'R'
            ax.set_title(f'[{tag}] {self._short(feat)}',
                         fontsize=5, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('z-score' if idx % n_cols == 0 else '',
                          fontsize=5)
            ax.tick_params(axis='both', labelsize=4.5)

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout(h_pad=2.2, w_pad=2.8)
        _save(fig, self.out, 'Fig2_feature_distributions')

    # ── Feature correlation heatmaps ─────────────────────────────────
    def plot_correlations(self, dpi=300):
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(DOUBLE_COL, SINGLE_COL * 1.05),
            dpi=dpi,
            gridspec_kw={
                'width_ratios': [len(self.sel_s), len(self.sel_r)],
                'wspace': 0.65
            }
        )

        # Nature-style diverging colormap (RdBu)
        for ax, feats, df, title in [
            (ax1, self.sel_s, self.df_s, 'Spatial features'),
            (ax2, self.sel_r, self.df_r, 'Radiomics features'),
        ]:
            corr = df[feats].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, linecolor='white',
                cbar_kws={'shrink': 0.55, 'aspect': 15,
                          'ticks': [-1, -0.5, 0, 0.5, 1]},
                annot_kws={'fontsize': 4}, ax=ax
            )
            short = [self._short(f) for f in feats]
            ax.set_xticklabels(short, rotation=45, ha='right', fontsize=4)
            ax.set_yticklabels(short, rotation=0, fontsize=4)
            ax.set_title(title, fontsize=6, fontweight='bold', pad=6)

            # Thin colorbar label
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=4, width=0.3, length=2)

        _save(fig, self.out, 'Fig3_feature_correlations')

    def generate_all(self):
        print("\n" + "=" * 55)
        print("  GENERATING FEATURE FIGURES (Nature style)")
        print("=" * 55 + "\n")
        self.plot_feature_boxplots()
        self.plot_correlations()
        print()


# =============================================================================
# STATISTICAL REPORT
# =============================================================================
def statistical_report(viz, output_dir):
    out = Path(output_dir)
    y = viz.y
    names = list(viz.models.keys())
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1 = viz.df[viz.models[names[i]]].values
            p2 = viz.df[viz.models[names[j]]].values
            yp1 = (p1 > 0.5).astype(int)
            yp2 = (p2 > 0.5).astype(int)
            c1, c2 = (yp1 == y), (yp2 == y)
            n01 = np.sum(~c1 & c2)
            n10 = np.sum(c1 & ~c2)
            if n10 + n01 > 0:
                mc = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
                pv = 1 - stats.chi2.cdf(mc, 1)
            else:
                mc, pv = 0, 1.0
            a1 = auc(*roc_curve(y, p1)[:2])
            a2 = auc(*roc_curve(y, p2)[:2])

            # DeLong test approximation via bootstrap
            rng = np.random.RandomState(42)
            diffs = []
            for _ in range(2000):
                idx = rng.choice(len(y), len(y), replace=True)
                if len(np.unique(y[idx])) < 2:
                    continue
                a1_b = auc(*roc_curve(y[idx], p1[idx])[:2])
                a2_b = auc(*roc_curve(y[idx], p2[idx])[:2])
                diffs.append(a1_b - a2_b)
            diffs = np.array(diffs)
            if len(diffs) > 0 and np.std(diffs) > 0:
                z_del = np.mean(diffs) / np.std(diffs)
                p_delong = 2 * (1 - stats.norm.cdf(abs(z_del)))
            else:
                z_del, p_delong = 0, 1.0

            rows.append({
                'Comparison': f'{names[i]} vs {names[j]}',
                'AUC_1': f'{a1:.3f}',
                'AUC_2': f'{a2:.3f}',
                'ΔAUC': f'{abs(a1 - a2):.3f}',
                'McNemar χ²': f'{mc:.3f}',
                'McNemar P': f'{pv:.4f}',
                'Bootstrap ΔAUC P': f'{p_delong:.4f}',
                'Sig. (P<0.05)': 'Yes' if min(pv, p_delong) < 0.05 else 'No',
            })

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(out / 'model_comparisons.csv', index=False)

    lines = [
        '=' * 65,
        'STATISTICAL ANALYSIS REPORT',
        '=' * 65,
        f'Date: {pd.Timestamp.now():%Y-%m-%d %H:%M}',
        f'N = {len(y)}  |  pCR = {np.sum(y == 0)}  |  '
        f'Non-pCR = {np.sum(y == 1)}',
        '-' * 65,
    ]
    for name, col in viz.models.items():
        p = viz.df[col].values
        yp = (p > 0.5).astype(int)
        m = viz._metrics(y, yp, p)
        lo, hi = viz._auc_ci(y, p)
        bs = brier_score_loss(y, p)
        tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
        lines += [
            f'\n{name}',
            f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}',
            f'  AUC  = {m["AUC"]:.3f} [{lo:.3f}\u2013{hi:.3f}]',
            f'  ACC  = {m["ACC"]:.3f}   SEN = {m["SEN"]:.3f}   '
            f'SPE = {m["SPE"]:.3f}',
            f'  PPV  = {m["PPV"]:.3f}   NPV = {m["NPV"]:.3f}   '
            f'F1  = {m["F1"]:.3f}',
            f'  Brier = {bs:.3f}',
        ]
    lines += [
        '\n' + '-' * 65,
        'Pairwise comparisons (McNemar + Bootstrap AUC)',
        '-' * 65,
    ]
    lines.append(df_comp.to_string(index=False))
    lines.append('=' * 65)

    with open(out / 'statistical_report.txt', 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ model_comparisons.csv")
    print(f"  ✓ statistical_report.txt\n")
    print(df_comp.to_string(index=False))


# =============================================================================
# FOREST PLOT (Supplementary – common in Nature Medicine)
# =============================================================================
def plot_forest(viz, output_dir, dpi=300):
    """
    Forest plot comparing model AUCs with 95% CI.
    Common supplementary figure in Nature Medicine / Nature Communications.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.5), dpi=dpi)

    y_positions = []
    for idx, (name, col) in enumerate(viz.models.items()):
        s = MODEL_STYLES[name]
        p = viz.df[col].values
        a = auc(*roc_curve(viz.y, p)[:2])
        lo, hi = viz._auc_ci(viz.y, p, 2000)

        y_pos = len(viz.models) - 1 - idx
        y_positions.append(y_pos)

        # CI line
        ax.plot([lo, hi], [y_pos, y_pos],
                color=s['color'], lw=1.2, solid_capstyle='round')
        # Point estimate
        ax.plot(a, y_pos, marker=s['marker'], color=s['color'],
                ms=5, mew=0.4, markeredgecolor='white', zorder=5)
        # Text annotation
        ax.text(1.02, y_pos,
                f'{a:.3f} [{lo:.3f}\u2013{hi:.3f}]',
                va='center', ha='left', fontsize=5,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(viz.models.keys()), fontsize=6)
    ax.set_xlabel('AUC (95 % CI)', fontsize=6)
    ax.set_xlim(0.4, 1.0)
    ax.axvline(0.5, color=C['light_gray'], ls=':', lw=0.4)

    # Clean up
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.5, len(viz.models) - 0.5)

    _save(fig, Path(output_dir), 'SuppFig_forest_plot')


# =============================================================================
# VIOLIN + SWARM PLOT (Alternative to boxplot, popular in Nature)
# =============================================================================
def plot_violin_comparison(viz, output_dir, dpi=300):
    """
    Split violin plot comparing predicted probabilities across models.
    """
    fig, ax = plt.subplots(
        figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.7), dpi=dpi
    )

    records = []
    for name, col in viz.models.items():
        p = viz.df[col].values
        for i, pi in enumerate(p):
            records.append({
                'Model': name,
                'Probability': pi,
                'Label': 'pCR' if viz.y[i] == 0 else 'Non-pCR'
            })
    df_v = pd.DataFrame(records)

    pal = {'pCR': C['blue'], 'Non-pCR': C['red']}

    sns.violinplot(
        data=df_v, x='Model', y='Probability', hue='Label',
        split=True, palette=pal, ax=ax,
        inner=None, linewidth=0.4, saturation=0.7,
        order=list(viz.models.keys()),
        hue_order=['pCR', 'Non-pCR'],
        cut=0, bw_method=0.3, density_norm='width'
    )

    # Add median lines inside violins
    for i, name in enumerate(viz.models.keys()):
        col = viz.models[name]
        p = viz.df[col].values
        for label_val, offset in [(0, -0.02), (1, 0.02)]:
            med = np.median(p[viz.y == label_val])
            ax.plot([i + offset - 0.08, i + offset + 0.08],
                    [med, med],
                    color='white', lw=0.8, zorder=3)

    ax.axhline(0.5, color=C['black'], ls='--', lw=0.4, alpha=0.5)
    ax.set(xlabel='', ylabel='Predicted probability')
    ax.legend(loc='upper right', fontsize=5, title='True label',
              title_fontproperties={'size': 5.5, 'weight': 'bold'})

    _save(fig, Path(output_dir), 'SuppFig_violin_comparison')

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/test_predictions_results.csv'
    spatial_csv     = 'XXXyour path/Dataset/Dataset002_ECResponse/spatial3d_features.csv'
    texture_csv     = 'XXXyour path/Dataset/Dataset002_ECResponse/spatial_texture_features.csv'
    radiomics_csv   = 'XXXyour path/Dataset/Dataset002_ECResponse/radiomics_features.csv'
    labels_excel    = 'XXXyour path/Dataset/Dataset002_ECResponse/PatientList.xlsx'
    output_dir      = 'XXXyour path/Dataset/Dataset002_ECResponse/visualizations'

    # predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/TJH_predictions_results.csv'
    # spatial_csv     = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial3d_features.csv'
    # texture_csv     = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial_texture_features.csv'
    # radiomics_csv   = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/radiomics_features.csv'
    # labels_excel    = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/PatientList.xlsx'
    # output_dir      = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/visualizations'

    selected_spatial = [
            'glcm_Idm_spatial_variability',
            'glrlm_ShortRunEmphasis_spatial_variability',
            'glrlm_LongRunEmphasis_spatial_variability',
            'glrlm_LowGrayLevelRunEmphasis_moran_i',
            'glrlm_RunEntropy_spatial_variability',
            'glszm_SmallAreaEmphasis_lh_outlier_fraction',
            'glszm_LowGrayLevelZoneEmphasis_hl_outlier_fraction',
        ]
    selected_radiomics = [
        'original_shape_MajorAxisLength',
        'original_glrlm_RunLengthNonUniformity',
        'original_gldm_DependenceNonUniformity',
    ]

    # ── Model performance panels ─────────────────────────────────────
    print("\n" + "#" * 55)
    print("  MODEL PERFORMANCE PANELS")
    print("#" * 55)

    viz = MedicalImagingVisualizer(predictions_csv, output_dir)
    viz.generate_all()

    # ── Feature figures ──────────────────────────────────────────────
    print("#" * 55)
    print("  FEATURE FIGURES")
    print("#" * 55)

    fviz = FeatureVisualizer(
        spatial_csv, texture_csv, radiomics_csv, labels_excel,
        selected_spatial, selected_radiomics, output_dir
    )
    fviz.generate_all()

    # ── Statistical report ───────────────────────────────────────────
    print("#" * 55)
    print("  STATISTICAL REPORT")
    print("#" * 55 + "\n")

    statistical_report(viz, output_dir)

    # ── Supplementary figures ────────────────────────────────────────
    print("#" * 55)
    print("  SUPPLEMENTARY FIGURES")
    print("#" * 55 + "\n")

    plot_forest(viz, output_dir)
    plot_violin_comparison(viz, output_dir)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  OUTPUT FILES")
    print("=" * 55)
    print(f"""
Main figure panels (combine in Illustrator / Inkscape):
    panel_a_roc.pdf / .png / .tiff
    panel_b_calibration.pdf / .png / .tiff
    panel_c_dca.pdf / .png / .tiff
    panel_d_radar.pdf / .png / .tiff
    panel_e_confusion_matrix.pdf / .png / .tiff
    panel_f_prob_distribution.pdf / .png / .tiff
    Figure_combined_panels.pdf / .png / .tiff

Feature figures:
    Fig2_feature_distributions.pdf / .png / .tiff
    Fig3_feature_correlations.pdf / .png / .tiff

Supplementary figures:
    SuppFig_forest_plot.pdf / .png / .tiff
    SuppFig_violin_comparison.pdf / .png / .tiff

Tables & statistics:
    Table1_metrics.csv / .tex
    model_comparisons.csv
    statistical_report.txt

Directory: {output_dir}
""")
    print("=" * 55 + "\n")