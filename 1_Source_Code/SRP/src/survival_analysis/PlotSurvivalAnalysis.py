import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import median_survival_times
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# NATURE-STYLE GLOBAL SETTINGS
# =============================================================================
def set_nature_style():
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
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 3,
        'axes.titlepad': 6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.grid': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'legend.frameon': False,
    })

set_nature_style()

C = {
    'red': '#c1272d', 'blue': '#0000a7', 'green': '#008176',
    'orange': '#e66100', 'gray': '#666666', 'light_gray': '#b3b3b3',
    'black': '#000000',
}

SINGLE_COL = 3.5
DOUBLE_COL = 7.2

# ─────────────────────────────────────────────────────────────────────
# CANONICAL GROUP ORDERINGS
# ─────────────────────────────────────────────────────────────────────
# Define explicit orderings so that "Low-risk" always comes before
# "High-risk" in iteration, legend, and risk table, and "High-risk"
# is always the test group (coded 1) in Cox regression.
GROUP_ORDER = {
    'Low-risk':  0,
    'High-risk': 1,
    'pCR':       0,
    'Non-pCR':   1,
}


def _ordered_groups(group_values):
    """
    Return the unique groups in canonical order:
      - Low-risk before High-risk
      - pCR before Non-pCR
    Falls back to alphabetical if labels are unknown.
    """
    unique = list(set(group_values))
    return sorted(unique, key=lambda g: GROUP_ORDER.get(g, ord(g[0])))


def _save(fig, out_dir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(Path(out_dir) / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}")


# =============================================================================
# SURVIVAL ANALYZER
# =============================================================================
class SurvivalAnalyzer:

    def __init__(self, predictions_csv: str, survival_excel: str,
                 output_dir: str):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        df_pred = pd.read_csv(predictions_csv)
        df_pred['case_id'] = df_pred['case_id'].astype(str).str.strip()

        df_surv = pd.read_excel(survival_excel)
        df_surv.columns = df_surv.columns.str.strip()
        df_surv['case_id'] = df_surv['CaseID'].astype(str).str.strip()

        self.df = pd.merge(df_pred, df_surv, on='case_id', how='inner')
        print(f"  Merged: {len(self.df)} patients "
              f"(pred={len(df_pred)}, surv={len(df_surv)})")

        required = ['DFS', 'DFS-Time', 'OS', 'OS-Time']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in survival {missing}")

        for col in ['DFS', 'OS']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        for col in ['DFS-Time', 'OS-Time']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df[self.df['DFS-Time'] > 0].copy()
        self.df.dropna(subset=required, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print(f"  After cleaning: {len(self.df)} patients with complete data")

        self.models = {
            'SpatialRad': 'spatial_probability',
            'Radiomics':  'radiomics_probability',
            'Fusion':     'fused_probability',
        }
        # Display names for plot titles/legends only — internal keys above
        # stay unchanged since they drive column names, filenames, and
        # self.thresholds lookups throughout this class.
        self.display_names = {
            'SpatialRad': 'Spatial Feature-only Model',
            'Radiomics':  'Conventional Radiomics Model',
            'Fusion':     'SRP Model',
        }

        self.df['true_group'] = np.where(
            self.df['true_label'] == 1, 'Non-pCR', 'pCR')

        self.thresholds = {}
        self._print_summary()

    def _print_summary(self):
        print(f"\n  {'─'*50}")
        print(f"  SURVIVAL DATA SUMMARY")
        print(f"  {'─'*50}")
        for endpoint, time_col, event_col in [
            ('DFS', 'DFS-Time', 'DFS'),
            ('OS',  'OS-Time',  'OS'),
        ]:
            t = self.df[time_col]
            e = self.df[event_col]
            print(f"  {endpoint}:")
            print(f"    Events: {int(e.sum())}/{len(e)} "
                  f"({e.sum()/len(e)*100:.1f}%)")
            print(f"    Follow-up: median={t.median():.1f}, "
                  f"range=[{t.min():.1f}–{t.max():.1f}]")
        print(f"  {'─'*50}\n")

    # ─────────────────────────────────────────────────────────────────
    # OPTIMAL THRESHOLD SEARCH
    # ─────────────────────────────────────────────────────────────────
    def _find_optimal_threshold(self, model_name, time_col='DFS-Time',
                                event_col='DFS'):
        prob_col = self.models[model_name]
        scores = self.df[prob_col].values
        best_thr = 0.5
        best_p = 1.0

        for q in np.arange(25, 76, 2):
            thr = np.percentile(scores, q)
            grp = np.where(scores >= thr, 'High-risk', 'Low-risk')
            n_high = np.sum(grp == 'High-risk')
            n_low = np.sum(grp == 'Low-risk')
            if n_high < 5 or n_low < 5:
                continue
            mask_h = grp == 'High-risk'
            mask_l = grp == 'Low-risk'
            try:
                lr = logrank_test(
                    self.df.loc[mask_h, time_col],
                    self.df.loc[mask_l, time_col],
                    event_observed_A=self.df.loc[mask_h, event_col],
                    event_observed_B=self.df.loc[mask_l, event_col])
                if lr.p_value < best_p:
                    best_p = lr.p_value
                    best_thr = thr
            except Exception:
                continue
        return best_thr

    def stratify_all_models(self):
        print("── Optimal threshold search (DFS log-rank) ──")
        for model_name, prob_col in self.models.items():
            # thr = self._find_optimal_threshold(
            #     model_name, time_col='DFS-Time', event_col='DFS')
            if model_name == 'Radiomics':
                thr = 0.5 #0.35#0.63 #0.4#
            elif model_name == 'SpatialRad':
                thr = 0.5 #0.46#0.35 #0.46#
            elif model_name == 'Fusion':
                thr = 0.5 #0.454#0.71 #0.46#0.71
            
            self.thresholds[model_name] = thr
            self.df[f'{model_name}_risk'] = np.where(
                self.df[prob_col] >= thr, 'High-risk', 'Low-risk')
            vc = self.df[f'{model_name}_risk'].value_counts()
            print(f"  {model_name}: threshold = {thr:.3f}")
            for g in ['Low-risk', 'High-risk']:
                n = vc.get(g, 0)
                print(f"    {g}: {n} ({n/len(self.df)*100:.1f}%)")

    # ─────────────────────────────────────────────────────────────────
    # COMPUTE STATISTICS — fixed group ordering
    # ─────────────────────────────────────────────────────────────────
    def _compute_stats(self, time_col, event_col, group_col):
        # Use canonical ordering instead of alphabetical sort
        groups = _ordered_groups(self.df[group_col].dropna().unique())
        result = {'groups': groups, 'per_group': {}}

        for grp in groups:
            mask = self.df[group_col] == grp
            n = int(mask.sum())
            n_events = int(self.df.loc[mask, event_col].sum())
            kmf = KaplanMeierFitter()
            kmf.fit(self.df.loc[mask, time_col],
                    self.df.loc[mask, event_col])
            med = kmf.median_survival_time_
            surv_rates = {}
            for yr in [12, 36, 60]:
                try:
                    surv_rates[yr] = float(kmf.predict(yr))
                except Exception:
                    surv_rates[yr] = np.nan
            result['per_group'][grp] = {
                'n': n, 'events': n_events,
                'median': med, 'surv_rates': surv_rates,
            }

        # Log-rank test
        if len(groups) == 2:
            m0 = self.df[group_col] == groups[0]
            m1 = self.df[group_col] == groups[1]
            lr = logrank_test(
                self.df.loc[m0, time_col], self.df.loc[m1, time_col],
                event_observed_A=self.df.loc[m0, event_col],
                event_observed_B=self.df.loc[m1, event_col])
            result['logrank_stat'] = lr.test_statistic
            result['logrank_p'] = lr.p_value
        elif len(groups) > 2:
            lr = multivariate_logrank_test(
                self.df[time_col], self.df[group_col],
                self.df[event_col])
            result['logrank_stat'] = lr.test_statistic
            result['logrank_p'] = lr.p_value
        else:
            result['logrank_stat'] = np.nan
            result['logrank_p'] = np.nan

        # Cox PH — groups[1] is always the "worse" group
        # (High-risk or Non-pCR) coded as 1
        result['hr'] = result['hr_lo'] = result['hr_hi'] = np.nan
        result['cox_p'] = result['c_index'] = np.nan

        if len(groups) == 2:
            df_cox = self.df[[time_col, event_col, group_col]].dropna().copy()
            df_cox = df_cox[df_cox[time_col] > 0].copy()
            # groups[1] = High-risk or Non-pCR → coded 1
            df_cox['_grp'] = (df_cox[group_col] == groups[1]).astype(int)
            try:
                cph = CoxPHFitter()
                cph.fit(df_cox[[time_col, event_col, '_grp']],
                        duration_col=time_col, event_col=event_col)
                s = cph.summary
                result['hr'] = s.loc['_grp', 'exp(coef)']
                result['hr_lo'] = s.loc['_grp', 'exp(coef) lower 95%']
                result['hr_hi'] = s.loc['_grp', 'exp(coef) upper 95%']
                result['cox_p'] = s.loc['_grp', 'p']
                result['c_index'] = cph.concordance_index_
            except Exception as e:
                print(f"    ⚠ Cox failed: {e}")

        return result

    # ─────────────────────────────────────────────────────────────────
    # NUMBER-AT-RISK
    # ─────────────────────────────────────────────────────────────────
    def _get_risk_counts(self, kmf, time_points):
        counts = []
        et = kmf.event_table
        for t in time_points:
            if t > kmf.timeline.max():
                counts.append(0)
                continue
            idx = et.index[et.index <= t]
            if len(idx) > 0:
                row = et.loc[idx[-1]]
                nar = int(row['at_risk'] - row['removed'])
                counts.append(max(0, nar))
            else:
                counts.append(int(et.iloc[0]['at_risk']))
        return counts

    # ─────────────────────────────────────────────────────────────────
    # MASTER KM PLOT — fixed group ordering everywhere
    # ─────────────────────────────────────────────────────────────────
    def _plot_km_full(self, time_col, event_col, group_col, title,
                      colors_dict, dpi=300, figsize=None,
                      n_time_ticks=6, ax_external=None):
        stats_dict = self._compute_stats(time_col, event_col, group_col)
        # Canonical order: [Low-risk, High-risk] or [pCR, Non-pCR]
        groups = stats_dict['groups']

        own_fig = ax_external is None
        if own_fig:
            if figsize is None:
                figsize = (SINGLE_COL * 1.15, SINGLE_COL * 1.20)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax_external.figure
            ax = ax_external

        # ── Fit & plot KM curves in canonical order ──────────────────
        kmf_dict = {}
        for grp in groups:
            mask = self.df[group_col] == grp
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=self.df.loc[mask, time_col],
                event_observed=self.df.loc[mask, event_col],
                label=grp)
            kmf_dict[grp] = kmf
            color = colors_dict.get(grp, C['gray'])

            kmf.plot_survival_function(
                ax=ax, color=color, lw=1.2,
                ci_show=True, ci_alpha=0.10)

            # Censoring ticks
            cens = kmf.event_table
            cens_times = cens[cens['censored'] > 0].index
            for ct in cens_times:
                sv = float(kmf.predict(ct))
                ax.plot(ct, sv, '|', color=color, ms=5, mew=0.5, alpha=0.7)

        # ── Median survival dashed lines ─────────────────────────────
        has_finite_median = any(
            np.isfinite(stats_dict['per_group'][g]['median']) for g in groups)
        if has_finite_median:
            ax.axhline(0.5, color=C['light_gray'], ls=':', lw=0.4, zorder=0)
            for grp in groups:
                med = stats_dict['per_group'][grp]['median']
                color = colors_dict.get(grp, C['gray'])
                if np.isfinite(med):
                    ax.plot([med, med], [0, 0.5], color=color,
                            ls=':', lw=0.5, alpha=0.5)

        # ── Axis formatting ──────────────────────────────────────────
        max_time = self.df[time_col].max()
        ax.set_xlim(-max_time * 0.02, max_time * 1.05)
        ax.set_ylim(-0.03, 1.06)
        ax.set_ylabel('Survival probability')
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.legend(loc='upper right', fontsize=5.5, frameon=False)

        # ── P-value annotation only ──────────────────────────────────
        p = stats_dict['logrank_p']
        if not np.isnan(p):
            p_str = 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'
            ax.text(0.97, 0.03, f'Log-rank {p_str}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=6, fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec=C['light_gray'], alpha=0.85, lw=0.4))

        # ── Number-at-risk table (canonical group order) ─────────────
        time_points = np.linspace(0, max_time, n_time_ticks)
        time_points = np.unique(np.round(time_points).astype(int))

        if own_fig:
            fig.subplots_adjust(bottom=0.28)

        tbl_y_top = -0.14
        tbl_row_h = 0.06
        tbl_fs = 5

        ax.text(-0.01, tbl_y_top, 'No. at risk',
                transform=ax.transAxes, fontsize=tbl_fs,
                fontweight='bold', va='top', ha='right',
                color=C['black'])

        xlim = ax.get_xlim()
        for t in time_points:
            xfrac = (t - xlim[0]) / (xlim[1] - xlim[0])
            ax.text(xfrac, tbl_y_top, str(int(t)),
                    transform=ax.transAxes, fontsize=tbl_fs,
                    ha='center', va='top', color=C['gray'])

        # Iterate in canonical order so Low-risk is row 1, High-risk row 2
        for i, grp in enumerate(groups):
            y_row = tbl_y_top - (i + 1) * tbl_row_h
            color = colors_dict.get(grp, C['gray'])

            ax.text(-0.01, y_row, grp,
                    transform=ax.transAxes, fontsize=tbl_fs,
                    fontweight='bold', va='top', ha='right',
                    color=color)

            counts = self._get_risk_counts(kmf_dict[grp], time_points)
            for j, t in enumerate(time_points):
                xfrac = (t - xlim[0]) / (xlim[1] - xlim[0])
                ax.text(xfrac, y_row, str(counts[j]),
                        transform=ax.transAxes, fontsize=tbl_fs,
                        ha='center', va='top', color=color)

        ax.axhline(y=0, color=C['light_gray'], lw=0.3,
                   xmin=0, xmax=1, clip_on=False)

        n_groups = len(groups)
        xlabel_y = tbl_y_top - (n_groups + 1) * tbl_row_h - 0.02
        ax.text(0.5, xlabel_y, 'Time (months)',
                transform=ax.transAxes, fontsize=7,
                ha='center', va='top', color=C['black'])
        ax.set_xlabel('')

        if own_fig:
            return fig, ax, stats_dict
        else:
            return None, ax, stats_dict

    # ─────────────────────────────────────────────────────────────────
    # INDIVIDUAL KM per model × endpoint
    # ─────────────────────────────────────────────────────────────────
    def plot_km_by_prediction(self, dpi=300):
        endpoints = [
            ('DFS-Time', 'DFS', 'Disease-free survival'),
            ('OS-Time',  'OS',  'Overall survival'),
        ]
        colors_risk = {'Low-risk': C['blue'], 'High-risk': C['red']}

        for time_col, event_col, ep_name in endpoints:
            for model_name in self.models:
                risk_col = f'{model_name}_risk'
                if risk_col not in self.df.columns:
                    continue
                thr = self.thresholds.get(model_name, 0.5)
                fig, ax, st = self._plot_km_full(
                    time_col, event_col, risk_col,
                    title=f'{ep_name} – {self.display_names.get(model_name, model_name)}',
                    colors_dict=colors_risk, dpi=dpi)

                fname = f'KM_{ep_name.replace(" ", "_")}_{model_name}'
                _save(fig, self.out, fname)

    # ─────────────────────────────────────────────────────────────────
    # KM by true label
    # ─────────────────────────────────────────────────────────────────
    def plot_km_by_true_label(self, dpi=300):
        if 'true_group' not in self.df.columns:
            print("  ⚠ No true_group column; skipping true-label KM.")
            return

        endpoints = [
            ('DFS-Time', 'DFS', 'Disease-free survival'),
            ('OS-Time',  'OS',  'Overall survival'),
        ]
        colors_label = {'pCR': C['blue'], 'Non-pCR': C['red']}

        for time_col, event_col, ep_name in endpoints:
            fig, ax, st = self._plot_km_full(
                time_col, event_col, 'true_group',
                title=f'{ep_name} – True label',
                colors_dict=colors_label, dpi=dpi)

            fname = f'KM_{ep_name.replace(" ", "_")}_TrueLabel'
            _save(fig, self.out, fname)

    # ─────────────────────────────────────────────────────────────────
    # Combined KM figure (DFS + OS side by side for Fusion model)
    # ─────────────────────────────────────────────────────────────────
    def plot_km_combined(self, dpi=300):
        risk_col = 'Fusion_risk'
        if risk_col not in self.df.columns:
            print("  ⚠ Fusion_risk not available; skipping combined KM.")
            return

        colors_risk = {'Low-risk': C['blue'], 'High-risk': C['red']}
        thr = self.thresholds.get('Fusion', 0.5)
        fusion_display = self.display_names.get('Fusion', 'Fusion')
        configs = [
            ('DFS-Time', 'DFS', f'Disease-free survival – {fusion_display}'),
            ('OS-Time',  'OS',  f'Overall survival – {fusion_display}'),
        ]

        fig, axes = plt.subplots(1, 2,
                                 figsize=(DOUBLE_COL, SINGLE_COL * 1.25),
                                 dpi=dpi)
        fig.subplots_adjust(bottom=0.30, wspace=0.40)

        panel_labels = ['a', 'b']
        for idx, (time_col, event_col, title) in enumerate(configs):
            _, ax, st = self._plot_km_full(
                time_col, event_col, risk_col,
                title=title,
                colors_dict=colors_risk,
                dpi=dpi,
                ax_external=axes[idx])

            ax.text(-0.12, 1.06, panel_labels[idx],
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left')

        _save(fig, self.out, 'KM_combined_Fusion_DFS_OS')

    # ─────────────────────────────────────────────────────────────────
    # Cox proportional hazards regression (continuous probability)
    # ─────────────────────────────────────────────────────────────────
    def cox_regression(self):
        results = []
        endpoints = [
            ('DFS-Time', 'DFS', 'DFS'),
            ('OS-Time',  'OS',  'OS'),
        ]

        for time_col, event_col, ep_short in endpoints:
            for model_name, prob_col in self.models.items():
                if prob_col not in self.df.columns:
                    continue

                df_cox = self.df[[time_col, event_col, prob_col]].dropna()
                df_cox = df_cox[df_cox[time_col] > 0].copy()

                cph = CoxPHFitter()
                try:
                    cph.fit(df_cox, duration_col=time_col,
                            event_col=event_col)
                    s = cph.summary
                    hr = s.loc[prob_col, 'exp(coef)']
                    hr_lo = s.loc[prob_col, 'exp(coef) lower 95%']
                    hr_hi = s.loc[prob_col, 'exp(coef) upper 95%']
                    p_val = s.loc[prob_col, 'p']
                    c_index = cph.concordance_index_

                    results.append({
                        'Endpoint': ep_short,
                        'Model': model_name,
                        'Threshold': f'{self.thresholds.get(model_name, 0.5):.3f}',
                        'Variable': 'Probability (continuous)',
                        'HR': f'{hr:.2f}',
                        'HR [95% CI]': f'{hr:.2f} [{hr_lo:.2f}\u2013{hr_hi:.2f}]',
                        'P-value': f'{p_val:.4f}',
                        'C-index': f'{c_index:.3f}',
                        'Significant': 'Yes' if p_val < 0.05 else 'No',
                    })
                except Exception as e:
                    print(f"  ⚠ Cox failed for {model_name}/{ep_short}: {e}")

        df_results = pd.DataFrame(results)
        df_results.to_csv(self.out / 'cox_regression_results.csv', index=False)
        print(f"\n  ✓ cox_regression_results.csv")
        print('\n' + df_results.to_string(index=False))
        return df_results

    # ─────────────────────────────────────────────────────────────────
    # Forest plot of hazard ratios
    # ─────────────────────────────────────────────────────────────────
    def plot_forest(self, dpi=300):
        endpoints = [
            ('DFS-Time', 'DFS', 'DFS'),
            ('OS-Time',  'OS',  'OS'),
        ]
        rows = []

        for time_col, event_col, ep_short in endpoints:
            for model_name, prob_col in self.models.items():
                if prob_col not in self.df.columns:
                    continue
                df_cox = self.df[[time_col, event_col, prob_col]].dropna()
                df_cox = df_cox[df_cox[time_col] > 0].copy()

                cph = CoxPHFitter()
                try:
                    cph.fit(df_cox, duration_col=time_col,
                            event_col=event_col)
                    s = cph.summary
                    display_name = self.display_names.get(model_name, model_name)
                    rows.append({
                        'label': f'{display_name} ({ep_short})',
                        'hr': s.loc[prob_col, 'exp(coef)'],
                        'lo': s.loc[prob_col, 'exp(coef) lower 95%'],
                        'hi': s.loc[prob_col, 'exp(coef) upper 95%'],
                        'p': s.loc[prob_col, 'p'],
                    })
                except Exception:
                    pass

        if not rows:
            print("  ⚠ No Cox results to plot forest.")
            return

        df_f = pd.DataFrame(rows)
        n = len(df_f)

        # Wide enough for the longest label ("Conventional Radiomics Model
        # (DFS)") plus the HR/CI/P annotation column on the right.
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL * 2.1, 0.45 * n + 0.8), dpi=dpi)
        y_pos = np.arange(n)

        # Log-scale x-axis: some CIs span 2-3 orders of magnitude (e.g. HR
        # up to ~800), which would squash every other row into an
        # unreadable cluster near 0 on a linear axis. A CI touching 0
        # can't be plotted on a log axis, so floor the drawn lower bound
        # while keeping the true value in the text label.
        plot_floor = max(df_f['hr'].min() * 1e-2, 1e-3)

        for i, row in df_f.iterrows():
            color = C['red'] if row['p'] < 0.05 else C['gray']
            lo_i = max(row['lo'], plot_floor)
            ax.plot([lo_i, row['hi']], [i, i], color=color, lw=1.0, zorder=1)
            ax.plot([lo_i, lo_i], [i - 0.12, i + 0.12], color=color, lw=0.7, zorder=1)
            ax.plot([row['hi'], row['hi']], [i - 0.12, i + 0.12], color=color, lw=0.7, zorder=1)
            ax.plot(row['hr'], i, 'o', color=color, ms=5, zorder=2)
            p_text = f"P<0.001" if row['p'] < 0.001 else f"P={row['p']:.3f}"
            lo_str = f"{row['lo']:.2f}" if row['lo'] >= 0.005 else "<0.01"
            ax.text(df_f['hi'].max() * 1.3, i,
                    f"HR={row['hr']:.2f} "
                    f"[{lo_str}\u2013{row['hi']:.2f}]  {p_text}",
                    va='center', fontsize=5.5, color=color)

        ax.set_xscale('log')
        ax.axvline(1, color=C['light_gray'], ls='--', lw=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_f['label'], fontsize=6)
        ax.set_xlabel('Hazard Ratio (95% CI, log scale)')
        ax.set_title('Cox Proportional Hazards', fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(plot_floor * 0.7, df_f['hi'].max() * 1.3)
        ax.tick_params(axis='x', labelsize=5.5)

        fig.subplots_adjust(left=0.34, right=0.62)
        _save(fig, self.out, 'forest_plot_HR')

    # ─────────────────────────────────────────────────────────────────
    # Median survival summary table
    # ─────────────────────────────────────────────────────────────────
    def median_survival_summary(self):
        results = []
        endpoints = [
            ('DFS-Time', 'DFS', 'DFS'),
            ('OS-Time',  'OS',  'OS'),
        ]

        stratifications = []
        for name in self.models:
            stratifications.append((f'{name}_risk', name))
        stratifications.append(('true_group', 'TrueLabel'))

        for time_col, event_col, ep_short in endpoints:
            for group_col, strat_name in stratifications:
                if group_col not in self.df.columns:
                    continue
                # Use canonical order
                groups = _ordered_groups(self.df[group_col].dropna().unique())
                for grp in groups:
                    mask = self.df[group_col] == grp
                    if mask.sum() == 0:
                        continue
                    kmf = KaplanMeierFitter()
                    kmf.fit(self.df.loc[mask, time_col],
                            self.df.loc[mask, event_col])
                    med = kmf.median_survival_time_
                    results.append({
                        'Endpoint': ep_short,
                        'Stratification': strat_name,
                        'Threshold': (f'{self.thresholds.get(strat_name, "N/A")}'
                                      if strat_name != 'TrueLabel' else 'N/A'),
                        'Group': grp,
                        'N': int(mask.sum()),
                        'Events': int(self.df.loc[mask, event_col].sum()),
                        'Median survival': (f'{med:.1f}'
                                            if np.isfinite(med) else 'NR'),
                    })

        df_med = pd.DataFrame(results)
        df_med.to_csv(self.out / 'median_survival.csv', index=False)
        print(f"\n  ✓ median_survival.csv")
        print('\n' + df_med.to_string(index=False))
        return df_med

    # ─────────────────────────────────────────────────────────────────
    # Survival report (text)
    # ─────────────────────────────────────────────────────────────────
    def survival_report(self):
        lines = [
            '=' * 65,
            'SURVIVAL ANALYSIS REPORT',
            '=' * 65,
            f'Date: {pd.Timestamp.now():%Y-%m-%d %H:%M}',
            f'N = {len(self.df)}',
            f'DFS events: {int(self.df["DFS"].sum())} '
            f'({self.df["DFS"].mean()*100:.1f}%)',
            f'OS  events: {int(self.df["OS"].sum())} '
            f'({self.df["OS"].mean()*100:.1f}%)',
            f'Median follow-up (DFS-Time): '
            f'{self.df["DFS-Time"].median():.1f} months',
            f'Median follow-up (OS-Time):  '
            f'{self.df["OS-Time"].median():.1f} months',
            '',
            '-' * 65,
            'OPTIMAL THRESHOLDS (DFS log-rank minimisation)',
            '-' * 65,
        ]
        for model_name, thr in self.thresholds.items():
            lines.append(f'  {model_name}')

        lines += ['',
                   '-' * 65,
                   'LOG-RANK TESTS & COX REGRESSION (binary risk groups)',
                   '-' * 65]

        endpoints = [
            ('DFS-Time', 'DFS', 'DFS'),
            ('OS-Time',  'OS',  'OS'),
        ]

        for time_col, event_col, ep in endpoints:
            lines.append(f'\n  {ep}:')

            for model_name in self.models:
                risk_col = f'{model_name}_risk'
                if risk_col not in self.df.columns:
                    continue

                st = self._compute_stats(time_col, event_col, risk_col)
                sig = ('***' if st['logrank_p'] < 0.001 else
                       '**' if st['logrank_p'] < 0.01 else
                       '*' if st['logrank_p'] < 0.05 else 'n.s.')

                lines.append(f'    {model_name} (thr={self.thresholds[model_name]:.3f}):')
                for grp in st['groups']:
                    gs = st['per_group'][grp]
                    med_str = (f'{gs["median"]:.1f}'
                               if np.isfinite(gs['median']) else 'NR')
                    lines.append(
                        f'      {grp}: n={gs["n"]}, events={gs["events"]}, '
                        f'median={med_str}')
                lines.append(
                    f'      Log-rank: χ²={st["logrank_stat"]:.3f}, '
                    f'P={st["logrank_p"]:.4f} {sig}')
                if not np.isnan(st['hr']):
                    lines.append(
                        f'      HR={st["hr"]:.2f} '
                        f'[{st["hr_lo"]:.2f}\u2013{st["hr_hi"]:.2f}], '
                        f'C-index={st["c_index"]:.3f}')

            # By true label
            if 'true_group' in self.df.columns:
                st = self._compute_stats(time_col, event_col, 'true_group')
                sig = ('***' if st['logrank_p'] < 0.001 else
                       '**' if st['logrank_p'] < 0.01 else
                       '*' if st['logrank_p'] < 0.05 else 'n.s.')
                lines.append(f'    True label:')
                for grp in st['groups']:
                    gs = st['per_group'][grp]
                    med_str = (f'{gs["median"]:.1f}'
                               if np.isfinite(gs['median']) else 'NR')
                    lines.append(
                        f'      {grp}: n={gs["n"]}, events={gs["events"]}, '
                        f'median={med_str}')
                lines.append(
                    f'      Log-rank: χ²={st["logrank_stat"]:.3f}, '
                    f'P={st["logrank_p"]:.4f} {sig}')
                if not np.isnan(st['hr']):
                    lines.append(
                        f'      HR={st["hr"]:.2f} '
                        f'[{st["hr_lo"]:.2f}\u2013{st["hr_hi"]:.2f}], '
                        f'C-index={st["c_index"]:.3f}')

        lines += ['', '=' * 65]

        report_path = self.out / 'survival_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\n  ✓ survival_report.txt")
        for line in lines:
            print(f"  {line}")

    # ─────────────────────────────────────────────────────────────────
    # Generate all survival analyses
    # ─────────────────────────────────────────────────────────────────
    def generate_all(self):
        print("\n" + "=" * 55)
        print("  SURVIVAL ANALYSIS (Nature style)")
        print("=" * 55 + "\n")

        # Step 1: Find optimal thresholds and stratify
        self.stratify_all_models()

        # Step 2: KM plots
        print("\n── KM by model prediction (optimal threshold) ──")
        self.plot_km_by_prediction()

        print("\n── KM by true label ──")
        self.plot_km_by_true_label()

        print("\n── KM combined (Fusion) ──")
        self.plot_km_combined()

        # Step 3: Forest plot
        print("\n── Forest plot ──")
        self.plot_forest()

        # Step 4: Statistical results
        print("\n" + "-" * 55)
        print("  STATISTICAL RESULTS")
        print("-" * 55)

        self.cox_regression()
        self.median_survival_summary()
        self.survival_report()

        print("\n" + "=" * 55)
        print("  SURVIVAL ANALYSIS COMPLETE")
        print("=" * 55 + "\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/test_predictions_results.csv'
    # survival_excel  = 'XXXyour path/Dataset/Dataset002_ECResponse/Test_PatientLabels.xlsx'
    # output_dir  = 'XXXyour path/Dataset/Dataset002_ECResponse/survival_analysis'
    # predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/TJH_predictions_results.csv'
    # survival_excel = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/SurvivalData_TJH.xlsx'
    # output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse//TJH/survival_analysis'
    predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/model_results/SPCH_predictions_results.csv'
    survival_excel = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/PatientLabels.xlsx'
    output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/survival_analysis'
    analyzer = SurvivalAnalyzer(
        predictions_csv=predictions_csv,
        survival_excel=survival_excel,
        output_dir=output_dir,
    )
    analyzer.generate_all()

    print(f"""
  OUTPUT FILES:
  {'─'*55}
  Kaplan-Meier plots (optimal threshold + P-value + risk table):
     KM_Disease-free_survival_SpatialRad.pdf/png
     KM_Disease-free_survival_Radiomics.pdf/png
     KM_Disease-free_survival_Fusion.pdf/png
     KM_Overall_survival_SpatialRad.pdf/png
     KM_Overall_survival_Radiomics.pdf/png
     KM_Overall_survival_Fusion.pdf/png
     KM_Disease-free_survival_TrueLabel.pdf/png
     KM_Overall_survival_TrueLabel.pdf/png
     KM_combined_Fusion_DFS_OS.pdf/png

  Forest plot:
     forest_plot_HR.pdf/png

  Tables:
     cox_regression_results.csv
     median_survival.csv

  Report:
     survival_report.txt

  Directory: {output_dir}
""")