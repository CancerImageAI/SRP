import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import median_survival_times
import warnings
warnings.filterwarnings('ignore')

# Set Nature Medicine publication-quality plotting parameters
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 0.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


class SurvivalAnalyzer:
    """
    Comprehensive survival analysis with Nature Medicine quality figures.
    """
    
    def __init__(self, 
                 predictions_csv: str,
                 labels_excel: str,
                 output_dir: str):
        """Initialize survival analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.df_pred = pd.read_csv(predictions_csv)
        self.df_labels = pd.read_excel(labels_excel)
        
        # Standardize column names
        self.df_labels.columns = self.df_labels.columns.str.strip()
        
        # Handle alternative column names
        if 'DFS_Time' in self.df_labels.columns:
            self.df_labels.rename(columns={'DFS_Time': 'DFS-Time'}, inplace=True)
        if 'OS_Time' in self.df_labels.columns:
            self.df_labels.rename(columns={'OS_Time': 'OS-Time'}, inplace=True)
        
        # Identify case ID column
        case_id_cols = [col for col in self.df_labels.columns 
                       if 'case' in col.lower() or 'id' in col.lower() or 'patient' in col.lower()]
        
        if case_id_cols:
            self.df_labels.rename(columns={case_id_cols[0]: 'case_id'}, inplace=True)
        else:
            self.df_labels.rename(columns={self.df_labels.columns[0]: 'case_id'}, inplace=True)
        
        self.df_labels['case_id'] = self.df_labels['case_id'].astype(str)
        self.df_pred['case_id'] = self.df_pred['case_id'].astype(str)
        
        # Merge data
        self.df = pd.merge(self.df_pred, 
                          self.df_labels[['case_id', 'DFS', 'DFS-Time', 'OS', 'OS-Time']], 
                          on='case_id', how='inner')
        
        # Define risk scores and colors (Nature Medicine palette)
        self.risk_scores = {
            'Spatial': 'spatial_probability',
            'Radiomics': 'radiomics_probability',
            'Fused': 'fused_probability'
        }
        
        # Nature Medicine color scheme
        self.colors = {
            'Spatial': '#D62728',      # Red
            'Radiomics': '#1F77B4',    # Blue
            'Fused': '#2CA02C',        # Green
            'High Risk': '#D62728',    # Red for high risk
            'Low Risk': '#1F77B4',     # Blue for low risk
            'Intermediate Risk': '#FF7F0E'  # Orange for intermediate
        }
        
        print(f"✓ Loaded data for {len(self.df)} patients with survival information")
        print(f"  DFS events: {self.df['DFS'].sum()}/{len(self.df)} ({self.df['DFS'].sum()/len(self.df)*100:.1f}%)")
        print(f"  OS events: {self.df['OS'].sum()}/{len(self.df)} ({self.df['OS'].sum()/len(self.df)*100:.1f}%)")
        print(f"  Median DFS time: {self.df['DFS-Time'].median():.1f} months")
        print(f"  Median OS time: {self.df['OS-Time'].median():.1f} months")
    
    def stratify_by_risk(self, model_name='Fused', threshold=None, method='median'):
        """Stratify patients into risk groups."""
        risk_col = self.risk_scores[model_name]
        risk_scores = self.df[risk_col].values
        
        if threshold is not None:
            threshold_val = threshold
            risk_groups = ['High Risk' if score >= threshold_val else 'Low Risk' 
                          for score in risk_scores]
            thresholds = [threshold_val]
            
        elif method == 'median':
            threshold_val = np.median(risk_scores)
            risk_groups = ['High Risk' if score >= threshold_val else 'Low Risk' 
                          for score in risk_scores]
            thresholds = [threshold_val]
            
        elif method == 'optimal':
            # Find optimal threshold using log-rank test
            thresholds_to_test = np.percentile(risk_scores, np.arange(30, 71, 5))
            best_threshold = None
            best_pvalue = 1.0
            
            for test_thresh in thresholds_to_test:
                groups = ['High' if score >= test_thresh else 'Low' 
                         for score in risk_scores]
                
                df_temp = self.df.copy()
                df_temp['test_group'] = groups
                
                try:
                    results = logrank_test(
                        df_temp[df_temp['test_group'] == 'High']['DFS-Time'],
                        df_temp[df_temp['test_group'] == 'Low']['DFS-Time'],
                        df_temp[df_temp['test_group'] == 'High']['DFS'],
                        df_temp[df_temp['test_group'] == 'Low']['DFS']
                    )
                    
                    if results.p_value < best_pvalue:
                        best_pvalue = results.p_value
                        best_threshold = test_thresh
                except:
                    continue
            
            threshold_val = best_threshold if best_threshold is not None else np.median(risk_scores)
            risk_groups = ['High Risk' if score >= threshold_val else 'Low Risk' 
                          for score in risk_scores]
            thresholds = [threshold_val]
        
        df_stratified = self.df.copy()
        df_stratified['risk_group'] = risk_groups
        df_stratified['risk_score'] = risk_scores
        
        print(f"\n{model_name} Model Risk Stratification ({method}):")
        print(f"  Threshold: {thresholds[0]:.3f}")
        print(f"  Risk group distribution:")
        for group, count in pd.Series(risk_groups).value_counts().items():
            print(f"    {group}: {count} ({count/len(risk_groups)*100:.1f}%)")
        
        return df_stratified, thresholds
    
    def plot_km_curve_nature(self, df_stratified, outcome='DFS', model_name='Fused', 
                        figsize=(3.5, 3.5), dpi=300):
        """
        Plot Kaplan-Meier curve with R-style formatting.
        """
        time_col = f'{outcome}-Time'
        event_col = outcome
        
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Better spacing: more left margin for labels, taller risk table
        gs = GridSpec(2, 1, height_ratios=[3, 1.5], hspace=0.03,
                    left=0.20, right=0.95, top=0.95, bottom=0.10)
        
        ax_km = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
        
        kmf = KaplanMeierFitter()
        
        # Get unique risk groups and sort
        risk_groups = sorted(df_stratified['risk_group'].unique(), reverse=True)
        
        # R/ggplot2 style colors
        if len(risk_groups) == 2:
            group_colors = {'High Risk': '#F8766D', 'Low Risk': '#00BFC4'}
            group_styles = {'High Risk': '-', 'Low Risk': '-'}
        elif len(risk_groups) == 3:
            group_colors = {
                'High Risk': '#F8766D', 
                'Intermediate Risk': '#00BA38',
                'Low Risk': '#619CFF'
            }
            group_styles = {'High Risk': '-', 'Intermediate Risk': '-', 'Low Risk': '-'}
        
        # Plot KM curves with step style
        for group in risk_groups:
            mask = df_stratified['risk_group'] == group
            
            kmf.fit(
                durations=df_stratified.loc[mask, time_col],
                event_observed=df_stratified.loc[mask, event_col],
                label=f'{group}'
            )
            
            ax_km.step(kmf.survival_function_.index,
                    kmf.survival_function_.iloc[:, 0].values,
                    where='post',
                    color=group_colors.get(group, 'gray'),
                    linestyle=group_styles.get(group, '-'),
                    linewidth=2,
                    alpha=1.0,
                    label=f'{group}')
        
        # Log-rank test
        if len(risk_groups) == 2:
            results = logrank_test(
                df_stratified[df_stratified['risk_group'] == risk_groups[0]][time_col],
                df_stratified[df_stratified['risk_group'] == risk_groups[1]][time_col],
                df_stratified[df_stratified['risk_group'] == risk_groups[0]][event_col],
                df_stratified[df_stratified['risk_group'] == risk_groups[1]][event_col]
            )
            p_value = results.p_value
        else:
            results = multivariate_logrank_test(
                df_stratified[time_col],
                df_stratified['risk_group'],
                df_stratified[event_col]
            )
            p_value = results.p_value
        
        # P-value annotation
        if p_value < 0.001:
            p_text = 'p < 0.001'
        elif p_value < 0.01:
            p_text = f'p = {p_value:.3f}'
        else:
            p_text = f'p = {p_value:.2f}'
        
        ax_km.text(0.98, 0.02, p_text,
                transform=ax_km.transAxes,
                fontsize=9, 
                verticalalignment='bottom',
                horizontalalignment='right',
                style='italic')
        
        # Styling
        outcome_name = 'Overall survival' if outcome == 'OS' else 'Disease-free survival'
        ax_km.set_ylabel(f'{outcome_name} probability', fontsize=10, fontweight='normal')
        
        # Legend
        legend = ax_km.legend(loc='best', fontsize=9, frameon=False, 
                            title='Risk group', title_fontsize=9)
        
        # R/ggplot2 style grid and spines
        ax_km.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray', which='major')
        ax_km.set_axisbelow(True)
        
        for spine in ax_km.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
        
        # Set limits
        max_time = df_stratified[time_col].max()
        ax_km.set_ylim([0, 1])
        ax_km.set_xlim([0, max_time * 1.05])
        
        # Define time points
        if max_time > 60:
            time_points = [0, 12, 24, 36, 48, 60]
        elif max_time > 48:
            time_points = [0, 12, 24, 36, 48]
        elif max_time > 36:
            time_points = [0, 12, 24, 36]
        elif max_time > 24:
            time_points = [0, 6, 12, 18, 24]
        else:
            time_points = [0, 6, 12, 18]
        
        time_points = [t for t in time_points if t <= max_time]
        
        # Set x-axis ticks
        ax_km.set_xticks(time_points)
        ax_km.set_xticklabels([str(int(t)) for t in time_points])
        
        # Tick parameters
        ax_km.tick_params(axis='both', which='major', labelsize=9, length=4, width=0.8, 
                        direction='out', top=False, right=False)
        
        # Remove x-axis labels from KM plot
        ax_km.tick_params(axis='x', labelbottom=False)
        
        # Create risk table
        self._create_risk_table_r_style(ax_table, df_stratified, time_col, event_col, 
                            risk_groups, group_colors, time_points, max_time)
        
        # Save
        filename = f'km_curve_{model_name}_{outcome}_rstyle.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight', 
                pad_inches=0.1, transparent=False, facecolor='white')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi, 
                bbox_inches='tight', pad_inches=0.1)
        plt.savefig(self.output_dir / filename.replace('.pdf', '.svg'), 
                bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"✓ R-style KM curve saved: {filename}")
        
        return {
            'p_value': p_value,
            'n_groups': len(risk_groups)
        }


    def _create_risk_table_r_style(self, ax, df_stratified, time_col, event_col, 
                      risk_groups, group_colors, time_points, max_time):
        """
        Create R-style risk table with proper spacing - using transform coordinates.
        """
        # Remove all axes decorations
        ax.axis('off')
        
        # Calculate number at risk at each time point
        risk_table_data = []
        
        for group in reversed(risk_groups):  # Reverse to match plot order
            mask = df_stratified['risk_group'] == group
            n_at_risk = []
            
            for t in time_points:
                at_risk = df_stratified.loc[mask & (df_stratified[time_col] >= t)]
                n_at_risk.append(len(at_risk))
            
            risk_table_data.append({
                'group': group,
                'n_at_risk': n_at_risk,
                'color': group_colors.get(group, 'gray')
            })
        
        # Use data coordinates for x (time values)
        # Use axes fraction for y (0-1 scale)
        ax.set_xlim([0, max_time * 1.05])
        ax.set_ylim([0, 1])
        
        n_groups = len(risk_table_data)
        
        # Y positions in axes coordinates
        y_positions = []
        if n_groups == 2:
            y_positions = [0.75, 0.50]  # Two rows
        elif n_groups == 3:
            y_positions = [0.80, 0.60, 0.40]  # Three rows
        
        # Draw table
        for i, (row_data, y_pos) in enumerate(zip(risk_table_data, y_positions)):
            
            # Group label - use axes coordinates for x position (negative = left of plot)
            ax.text(-0.05, y_pos, row_data['group'], 
                    fontsize=9, fontweight='normal', 
                    va='center', ha='right',  # Right-aligned to avoid overlap
                    color=row_data['color'],
                    transform=ax.transAxes,  # Use axes fraction for positioning
                    clip_on=False)
            
            # Numbers at risk - use data coordinates (actual time values)
            for n, t in zip(row_data['n_at_risk'], time_points):
                ax.text(t, y_pos, str(n), 
                        fontsize=9, fontweight='normal', 
                        va='center', ha='center',
                        color='black',
                        transform=ax.get_xaxis_transform())  # x=data, y=axes fraction
        
        # X-axis elements at the bottom
        
        # Time values (tick labels)
        for t in time_points:
            ax.text(t, 0.20, str(int(t)), 
                    fontsize=9, fontweight='normal',
                    va='center', ha='center',
                    transform=ax.get_xaxis_transform())
        
        # Small tick marks
        for t in time_points:
            ax.plot([t, t], [0.28, 0.32], 
                    'k-', linewidth=0.8,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False)
        
        # "Time (months)" label
        ax.text(0.5, 0.05, 'Time (months)', 
                fontsize=10, fontweight='normal',
                va='center', ha='center',
                transform=ax.transAxes)
                

    def plot_combined_km_curves_nature(self, model_name='Fused', method='median', 
                                figsize=(7.2, 3.5), dpi=300):
        """
        Plot DFS and OS KM curves side by side (R style).
        """
        # Stratify patients
        df_stratified, thresholds = self.stratify_by_risk(
            model_name=model_name, 
            method=method
        )
        
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create grid
        gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25,
                    left=0.10, right=0.98, top=0.92, bottom=0.12)
        
        outcomes = ['DFS', 'OS']
        outcome_names = ['Disease-free survival', 'Overall survival']
        panel_labels = ['A', 'B']
        
        for idx, (outcome, outcome_name, panel_label) in enumerate(zip(outcomes, outcome_names, panel_labels)):
            # Create axes
            ax_km = fig.add_subplot(gs[0, idx])
            ax_table = fig.add_subplot(gs[1, idx])
            
            time_col = f'{outcome}-Time'
            event_col = outcome
            
            kmf = KaplanMeierFitter()
            
            # Get unique risk groups
            risk_groups = sorted(df_stratified['risk_group'].unique(), reverse=True)
            
            # R/ggplot2 colors
            if len(risk_groups) == 2:
                group_colors = {'High Risk': '#F8766D', 'Low Risk': '#00BFC4'}
            else:
                group_colors = {
                    'High Risk': '#F8766D', 
                    'Intermediate Risk': '#00BA38',
                    'Low Risk': '#619CFF'
                }
            
            # Plot KM curves
            for group in risk_groups:
                mask = df_stratified['risk_group'] == group
                
                kmf.fit(
                    durations=df_stratified.loc[mask, time_col],
                    event_observed=df_stratified.loc[mask, event_col],
                    label=f'{group}'
                )
                
                # Step plot (R style)
                ax_km.step(kmf.survival_function_.index,
                        kmf.survival_function_.iloc[:, 0].values,
                        where='post',
                        color=group_colors.get(group, 'gray'),
                        linewidth=2,
                        alpha=1.0,
                        label=f'{group}')
            
            # Log-rank test
            if len(risk_groups) == 2:
                results = logrank_test(
                    df_stratified[df_stratified['risk_group'] == risk_groups[0]][time_col],
                    df_stratified[df_stratified['risk_group'] == risk_groups[1]][time_col],
                    df_stratified[df_stratified['risk_group'] == risk_groups[0]][event_col],
                    df_stratified[df_stratified['risk_group'] == risk_groups[1]][event_col]
                )
                p_value = results.p_value
            else:
                results = multivariate_logrank_test(
                    df_stratified[time_col],
                    df_stratified['risk_group'],
                    df_stratified[event_col]
                )
                p_value = results.p_value
            
            # P-value annotation (R style)
            if p_value < 0.001:
                p_text = 'p < 0.001'
            elif p_value < 0.01:
                p_text = f'p = {p_value:.3f}'
            else:
                p_text = f'p = {p_value:.2f}'
            
            ax_km.text(0.98, 0.02, p_text,
                    transform=ax_km.transAxes,
                    fontsize=9, 
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    style='italic')
            
            # Styling
            ax_km.set_ylabel(f'{outcome_name} probability', fontsize=10, fontweight='normal')
            
            # Panel label (top-left, outside)
            ax_km.text(-0.12, 1.05, panel_label, transform=ax_km.transAxes,
                    fontsize=12, fontweight='bold', va='top', ha='right')
            # Legend (only on first panel)
            if idx == 0:
                legend = ax_km.legend(loc='best', fontsize=9, frameon=False,
                                    title='Risk group', title_fontsize=9)
            
            # R/ggplot2 style grid
            ax_km.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray', which='major')
            ax_km.set_axisbelow(True)
            
            # Keep all spines (R default)
            for spine in ax_km.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color('black')
        
            # Limits
            max_time = df_stratified[time_col].max()
            ax_km.set_ylim([0, 1])
            ax_km.set_xlim([0, max_time * 1.05])
            
            # Define time points
            if max_time > 60:
                time_points = [0, 12, 24, 36, 48, 60]
            elif max_time > 48:
                time_points = [0, 12, 24, 36, 48]
            elif max_time > 36:
                time_points = [0, 12, 24, 36]
            elif max_time > 24:
                time_points = [0, 6, 12, 18, 24]
            else:
                time_points = [0, 6, 12, 18]
            
            time_points = [t for t in time_points if t <= max_time]
            
            # Set x-axis ticks
            ax_km.set_xticks(time_points)
            ax_km.set_xticklabels([str(int(t)) for t in time_points])
            
            # Tick parameters (R style)
            ax_km.tick_params(axis='both', which='major', labelsize=9, length=4, width=0.8,
                            direction='out', top=False, right=False)
            
            # Remove x-axis label and ticks from KM plot (will show in risk table)
            ax_km.tick_params(axis='x', labelbottom=False)
            
            # Create risk table
            self._create_risk_table_r_style(ax_table, df_stratified, time_col, event_col,
                                risk_groups, group_colors, time_points, max_time)
        
        # Save
        filename = f'combined_km_curves_{model_name}_rstyle.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight',
                pad_inches=0.1, transparent=False, facecolor='white')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi,
                bbox_inches='tight', pad_inches=0.1)
        plt.savefig(self.output_dir / filename.replace('.pdf', '.svg'),
                bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"✓ R-style combined KM curves saved: {filename}")
        
        return df_stratified

    
    def plot_cox_forest_nature(self, outcome='DFS', figsize=(3.5, 2.5), dpi=300):
        """
        Create Nature Medicine style forest plot for Cox regression.
        
        Parameters:
        -----------
        outcome : str
            'DFS' or 'OS'
        figsize : tuple
            Figure size
        """
        time_col = f'{outcome}-Time'
        event_col = outcome
        
        results = []
        
        for model_name, risk_col in self.risk_scores.items():
            # Prepare data
            cox_data = self.df[[time_col, event_col, risk_col]].copy()
            cox_data.columns = ['time', 'event', 'risk_score']
            cox_data = cox_data.dropna()
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='time', event_col='event')
            
            # Extract results
            try:
                summary = cph.summary
                hr = np.exp(summary.loc['risk_score', 'coef'])
                ci_lower = np.exp(summary.loc['risk_score', 'coef lower 95%'])
                ci_upper = np.exp(summary.loc['risk_score', 'coef upper 95%'])
                p_value = summary.loc['risk_score', 'p']
            except (KeyError, AttributeError):
                hr = np.exp(cph.params_['risk_score'])
                ci_vals = cph.confidence_intervals_.values[0]
                ci_lower = np.exp(ci_vals[0])
                ci_upper = np.exp(ci_vals[1])
                p_value = cph.summary['p'].values[0]
            
            c_index = cph.concordance_index_
            
            results.append({
                'Model': model_name,
                'HR': hr,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'p_value': p_value,
                'C_index': c_index
            })
        
        df_results = pd.DataFrame(results)
        
        # Create forest plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        y_positions = np.arange(len(df_results))
        
        # Plot hazard ratios and CIs
        for idx, row in df_results.iterrows():
            color = self.colors.get(row['Model'], 'gray')
            
            # Plot CI line
            ax.plot([row['CI_Lower'], row['CI_Upper']], 
                   [y_positions[idx], y_positions[idx]], 
                   color=color, linewidth=1.5, alpha=0.8, solid_capstyle='round')
            
            # Plot caps on CI
            cap_height = 0.15
            ax.plot([row['CI_Lower'], row['CI_Lower']], 
                   [y_positions[idx] - cap_height, y_positions[idx] + cap_height],
                   color=color, linewidth=1.5, alpha=0.8)
            ax.plot([row['CI_Upper'], row['CI_Upper']], 
                   [y_positions[idx] - cap_height, y_positions[idx] + cap_height],
                   color=color, linewidth=1.5, alpha=0.8)
            
            # Plot HR point (square for Nature style)
            ax.scatter(row['HR'], y_positions[idx], 
                      s=80, color=color, zorder=3, 
                      marker='s', edgecolors='black', linewidth=0.5)
        
        # Add vertical line at HR = 1
        ax.axvline(x=1, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df_results['Model'], fontsize=8, fontweight='normal')
        ax.set_xlabel('Hazard ratio (95% CI)', fontsize=8, fontweight='normal')
        
        outcome_name = 'Disease-free survival' if outcome == 'DFS' else 'Overall survival'
        # ax.set_title(f'{outcome_name}', fontsize=9, fontweight='bold', pad=10)
        
        # Grid (subtle)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray', axis='x')
        ax.set_axisbelow(True)
        
        # Set x-axis limits with some padding
        x_min = min(df_results['CI_Lower']) * 0.8
        x_max = max(df_results['CI_Upper']) * 1.2
        ax.set_xlim([x_min, x_max])
        
        # Use log scale if HR range is large
        if x_max / x_min > 10:
            ax.set_xscale('log')
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=7, length=3, width=0.5)
        ax.tick_params(axis='x', which='minor', length=2, width=0.5)
        
        # Add text annotations for HR, CI, and p-value
        for idx, row in df_results.iterrows():
            # Format p-value
            if row['p_value'] < 0.001:
                p_text = 'P < 0.001'
            elif row['p_value'] < 0.01:
                p_text = f"P = {row['p_value']:.3f}"
            else:
                p_text = f"P = {row['p_value']:.2f}"
            
            # HR and CI text
            hr_text = f"{row['HR']:.2f} ({row['CI_Lower']:.2f}–{row['CI_Upper']:.2f})"
            c_text = f"C-index: {row['C_index']:.2f}"
            
            # Position text to the right of the plot
            text_x = ax.get_xlim()[1] * 1.02
            ax.text(text_x, y_positions[idx], f"{hr_text}  {p_text}  {c_text}",
                   va='center', ha='left', fontsize=6.5, 
                   transform=ax.get_yaxis_transform())
        
        # Adjust layout to accommodate text
        plt.tight_layout()
        
        # Save
        filename = f'cox_forest_plot_{outcome}_nature.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight',
                   pad_inches=0.05, transparent=False, facecolor='white')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi,
                   bbox_inches='tight', pad_inches=0.05)
        plt.savefig(self.output_dir / filename.replace('.pdf', '.svg'),
                   bbox_inches='tight', pad_inches=0.05)
        plt.close()
        
        print(f"✓ Nature-style Cox forest plot saved: {filename}")
        
        # Save results table
        df_results.to_csv(self.output_dir / f'cox_regression_results_{outcome}.csv', index=False)
        
        return df_results
    
    def plot_risk_score_distribution_nature(self, df_stratified, outcome='DFS', 
                                           model_name='Fused', figsize=(3.5, 2.5), dpi=300):
        """
        Plot distribution of risk scores by event status (Nature Medicine style).
        
        Parameters:
        -----------
        df_stratified : pd.DataFrame
            DataFrame with risk scores
        outcome : str
            'DFS' or 'OS'
        model_name : str
            Model name
        """
        event_col = outcome
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Separate by event status
        event_yes = df_stratified[df_stratified[event_col] == 1]['risk_score']
        event_no = df_stratified[df_stratified[event_col] == 0]['risk_score']
        
        # Create violin plots (Nature style)
        parts = ax.violinplot(
            [event_no, event_yes],
            positions=[0, 1],
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        # Color violins
        colors = ['#1F77B4', '#D62728']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(0.6)
        
        # Add box plots on top
        bp = ax.boxplot(
            [event_no, event_yes],
            positions=[0, 1],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.5),
            whiskerprops=dict(color='black', linewidth=0.5),
            capprops=dict(color='black', linewidth=0.5)
        )
        
        # Add individual points with jitter
        np.random.seed(42)
        for i, data in enumerate([event_no, event_yes]):
            y = data.values
            x = np.random.normal(i, 0.02, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=8, color=colors[i], edgecolors='none')
        
        # Statistical test
        from scipy import stats
        stat, p_value = stats.mannwhitneyu(event_yes, event_no)
        
        if p_value < 0.001:
            p_text = 'P < 0.001'
        elif p_value < 0.01:
            p_text = f'P = {p_value:.3f}'
        else:
            p_text = f'P = {p_value:.2f}'
        
        # Add significance annotation
        y_max = max(event_no.max(), event_yes.max())
        y_range = y_max - min(event_no.min(), event_yes.min())
        y_sig = y_max + 0.05 * y_range
        
        ax.plot([0, 1], [y_sig, y_sig], 'k-', linewidth=0.5)
        ax.text(0.5, y_sig + 0.01 * y_range, p_text,
               ha='center', va='bottom', fontsize=7)
        
        # Styling
        outcome_name = 'DFS' if outcome == 'DFS' else 'OS'
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'No event\n(n={len(event_no)})', 
                           f'Event\n(n={len(event_yes)})'], fontsize=8)
        ax.set_ylabel('Risk score', fontsize=8, fontweight='normal')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray', axis='y')
        ax.set_axisbelow(True)
        
        # Ticks
        ax.tick_params(axis='both', which='major', labelsize=7, length=3, width=0.5)
        
        plt.tight_layout()
        
        # Save
        filename = f'risk_score_distribution_{model_name}_{outcome}_nature.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight',
                   pad_inches=0.05, transparent=False, facecolor='white')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi,
                   bbox_inches='tight', pad_inches=0.05)
        plt.savefig(self.output_dir / filename.replace('.pdf', '.svg'),
                   bbox_inches='tight', pad_inches=0.05)
        plt.close()
        
        print(f"✓ Nature-style risk score distribution saved: {filename}")
    
    def plot_time_dependent_roc_nature(self, model_name='Fused', outcome='DFS',
                                      time_points=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60], 
                                      figsize=(3.5, 3.5), dpi=300):
        """
        Plot time-dependent ROC curves (Nature Medicine style).
        
        Parameters:
        -----------
        model_name : str
            Model name
        outcome : str
            'DFS' or 'OS'
        time_points : list
            Time points for ROC analysis
        """
        from sklearn.metrics import roc_curve, auc
        
        time_col = f'{outcome}-Time'
        event_col = outcome
        risk_col = self.risk_scores[model_name]
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Nature color palette for time points
        colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
        
        auc_scores = []
        plotted_times = []
        
        for t, color in zip(time_points, colors[:len(time_points)]):
            # Create binary outcome: event before time t
            y_binary = ((self.df[time_col] <= t) & (self.df[event_col] == 1)).astype(int)
            
            # Skip if not enough events
            if y_binary.sum() < 5:
                print(f"⚠️  Insufficient events at {t} months, skipping...")
                continue
            
            # Calculate ROC
            fpr, tpr, _ = roc_curve(y_binary, self.df[risk_col])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            plotted_times.append(t)
            
            # Plot
            ax.plot(fpr, tpr, color=color, linewidth=1.5, alpha=0.9,
                   label=f'{t} months (AUC = {roc_auc:.2f})')
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
        
        # Styling
        outcome_name = 'Disease-free survival' if outcome == 'DFS' else 'Overall survival'
        ax.set_xlabel('1 − Specificity', fontsize=8, fontweight='normal')
        ax.set_ylabel('Sensitivity', fontsize=8, fontweight='normal')
        # ax.set_title(f'{outcome_name}', fontsize=9, fontweight='bold', pad=10)
        
        # Legend
        legend = ax.legend(loc='lower right', fontsize=7, frameon=True,
                          fancybox=False, shadow=False, framealpha=1,
                          edgecolor='black', linewidth=0.5, title='Time point')
        legend.get_frame().set_linewidth(0.5)
        legend.get_title().set_fontsize(7)
        legend.get_title().set_fontweight('normal')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        
        # Limits and aspect
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect('equal')
        
        # Ticks
        ax.tick_params(axis='both', which='major', labelsize=7, length=3, width=0.5)
        
        plt.tight_layout()
        
        # Save
        filename = f'time_dependent_roc_{model_name}_{outcome}_nature.pdf'
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight',
                   pad_inches=0.05, transparent=False, facecolor='white')
        plt.savefig(self.output_dir / filename.replace('.pdf', '.png'), dpi=dpi,
                   bbox_inches='tight', pad_inches=0.05)
        plt.savefig(self.output_dir / filename.replace('.pdf', '.svg'),
                   bbox_inches='tight', pad_inches=0.05)
        plt.close()
        
        print(f"✓ Nature-style time-dependent ROC saved: {filename}")
        
        return auc_scores, plotted_times
    
    def calculate_survival_statistics(self, df_stratified, outcome='DFS'):
        """Calculate detailed survival statistics for each risk group."""
        time_col = f'{outcome}-Time'
        event_col = outcome
        
        stats_list = []
        
        for group in sorted(df_stratified['risk_group'].unique(), reverse=True):
            mask = df_stratified['risk_group'] == group
            
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df_stratified.loc[mask, time_col],
                event_observed=df_stratified.loc[mask, event_col]
            )
            
            # Calculate median survival time
            median_survival = kmf.median_survival_time_
            
            # Calculate survival probabilities at specific time points
            time_points = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
            survival_probs = {}
            for t in time_points:
                try:
                    prob = kmf.survival_function_at_times(t).values[0]
                    survival_probs[f'{t}mo'] = prob
                except:
                    survival_probs[f'{t}mo'] = np.nan
            
            # Calculate confidence intervals for median
            try:
                ci = median_survival_times(kmf.confidence_interval_survival_function_)
                median_ci_lower = ci.iloc[0, 0]
                median_ci_upper = ci.iloc[0, 1]
            except:
                median_ci_lower = np.nan
                median_ci_upper = np.nan
            
            stats = {
                'Risk_Group': group,
                'N_Patients': mask.sum(),
                'N_Events': df_stratified.loc[mask, event_col].sum(),
                'Event_Rate': f"{df_stratified.loc[mask, event_col].sum() / mask.sum() * 100:.1f}%",
                'Median_Survival': median_survival,
                'Median_CI_Lower': median_ci_lower,
                'Median_CI_Upper': median_ci_upper,
            }
            
            # Add survival probabilities
            stats.update({f'Survival_{k}': f'{v:.3f}' if not np.isnan(v) else 'N/A' 
                         for k, v in survival_probs.items()})
            
            stats_list.append(stats)
        
        df_stats = pd.DataFrame(stats_list)
        return df_stats
    
    def create_survival_summary_table(self, df_stratified, model_name='Fused'):
        """Create comprehensive summary table for both DFS and OS."""
        summary_data = []
        
        for outcome in ['DFS', 'OS']:
            stats_df = self.calculate_survival_statistics(df_stratified, outcome)
            stats_df['Outcome'] = outcome
            summary_data.append(stats_df)
        
        df_summary = pd.concat(summary_data, ignore_index=True)
        
        # Reorder columns
        cols = ['Outcome', 'Risk_Group', 'N_Patients', 'N_Events', 'Event_Rate',
                'Median_Survival', 'Median_CI_Lower', 'Median_CI_Upper',
                'Survival_12mo', 'Survival_24mo', 'Survival_36mo', 'Survival_60mo']
        df_summary = df_summary[cols]
        
        # Save to CSV
        filename = f'survival_summary_{model_name}.csv'
        df_summary.to_csv(self.output_dir / filename, index=False)
        
        print(f"\n✓ Survival summary table saved: {filename}")
        print("\nSurvival Summary:")
        print("="*100)
        print(df_summary.to_string(index=False))
        print("="*100)
        
        return df_summary
    
    def perform_cox_regression(self, df_stratified, outcome='DFS', model_name='Fused'):
        """Perform Cox proportional hazards regression."""
        time_col = f'{outcome}-Time'
        event_col = outcome
        
        # Prepare data
        cox_data = df_stratified[[time_col, event_col, 'risk_score']].copy()
        cox_data.columns = ['time', 'event', 'risk_score']
        cox_data = cox_data.dropna()
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='time', event_col='event')
        
        # Print summary
        print(f"\n{model_name} Model - Cox Regression for {outcome}:")
        print("="*60)
        print(cph.summary)
        
        # Calculate C-index
        c_index = cph.concordance_index_
        print(f"\nConcordance Index (C-index): {c_index:.3f}")
        
        # Extract hazard ratio with CI
        try:
            summary = cph.summary
            hr = np.exp(summary.loc['risk_score', 'coef'])
            ci_lower = np.exp(summary.loc['risk_score', 'coef lower 95%'])
            ci_upper = np.exp(summary.loc['risk_score', 'coef upper 95%'])
            p_value = summary.loc['risk_score', 'p']
        except (KeyError, AttributeError):
            hr = np.exp(cph.params_['risk_score'])
            ci_vals = cph.confidence_intervals_.values[0]
            ci_lower = np.exp(ci_vals[0])
            ci_upper = np.exp(ci_vals[1])
            p_value = cph.summary['p'].values[0]
        
        print(f"\nHazard Ratio: {hr:.3f} (95% CI: {ci_lower:.3f}–{ci_upper:.3f})")
        print(f"P-value: {p_value:.4f}")
        
        return cph
    
    def generate_comprehensive_report(self, model_name='Fused', method='median'):
        """Generate comprehensive survival analysis report with Nature Medicine style figures."""
        print("\n" + "="*80)
        print(f"COMPREHENSIVE SURVIVAL ANALYSIS: {model_name} MODEL")
        print("="*80)
        
        # Stratify patients
        df_stratified, thresholds = self.stratify_by_risk(
            model_name=model_name,
            
            method=method
        )
        
        # Save stratified data
        df_stratified.to_csv(
            self.output_dir / f'stratified_patients_{model_name}.csv',
            index=False
        )
        
        results = {}
        
        # 1. Individual KM curves for DFS and OS
        print("\n1. Generating individual Kaplan-Meier curves...")
        for outcome in ['DFS', 'OS']:
            km_stats = self.plot_km_curve_nature(
                df_stratified, 
                outcome=outcome,
                model_name=model_name
            )
            results[f'{outcome}_km'] = km_stats
        
        # # 2. Combined KM curves (side by side)
        # print("\n2. Generating combined KM curves...")
        self.plot_combined_km_curves_nature(model_name=model_name, method=method)
        
        # 3. Cox regression analysis
        print("\n3. Performing Cox regression analysis...")
        for outcome in ['DFS', 'OS']:
            cph = self.perform_cox_regression(
                df_stratified,
                outcome=outcome,
                model_name=model_name
            )
            results[f'{outcome}_cox'] = cph
        
        # 4. Survival statistics
        print("\n4. Calculating survival statistics...")
        summary_table = self.create_survival_summary_table(
            df_stratified,
            model_name=model_name
        )
        
        # 5. Risk score distributions
        print("\n5. Plotting risk score distributions...")
        for outcome in ['DFS', 'OS']:
            self.plot_risk_score_distribution_nature(
                df_stratified,
                outcome=outcome,
                model_name=model_name
            )
        
        # 6. Time-dependent ROC
        print("\n6. Generating time-dependent ROC curves...")
        try:
            for outcome in ['DFS', 'OS']:
                auc_scores, time_points = self.plot_time_dependent_roc_nature(
                    model_name=model_name,
                    outcome=outcome
                )
                results[f'{outcome}_time_auc'] = auc_scores
                results[f'{outcome}_time_points'] = time_points
        except Exception as e:
            print(f"⚠️  Could not generate time-dependent ROC: {str(e)}")
        
        print("\n" + "="*80)
        print("SURVIVAL ANALYSIS COMPLETE")
        print("="*80)
        
        return results


def create_comprehensive_survival_figure(analyzer, method='median', 
                                         figsize=(7, 10), dpi=300):  # Increased height
    """
    Create a comprehensive multi-panel survival figure for publication.
    Clean KM curves without CI shading, better spacing.
    """
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Create grid with better spacing to avoid overlaps
    gs = GridSpec(5, 2, 
                 height_ratios=[4, 1.2, 2.5, 2.5, 3],  # Increased risk table height
                 hspace=0.45, wspace=0.3,  # Increased vertical spacing
                 left=0.08, right=0.95, top=0.96, bottom=0.04)
    
    outcomes = ['DFS', 'OS']
    outcome_names = ['Disease-free survival', 'Overall survival']
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    # Row 1-2: KM curves with risk tables for Fused model
    for col_idx, (outcome, outcome_name) in enumerate(zip(outcomes, outcome_names)):
        # Stratify
        df_stratified, _ = analyzer.stratify_by_risk(
            model_name='Fused',
            
            method=method
        )
        
        # Create axes for KM plot and risk table
        ax_km = fig.add_subplot(gs[0, col_idx])
        ax_table = fig.add_subplot(gs[1, col_idx])
        
        time_col = f'{outcome}-Time'
        event_col = outcome
        
        kmf = KaplanMeierFitter()
        risk_groups = sorted(df_stratified['risk_group'].unique(), reverse=True)
        
        # Colors
        group_colors = {'High Risk': '#D62728', 'Low Risk': '#1F77B4'}
        
        # Plot KM curves WITHOUT CI
        for group in risk_groups:
            mask = df_stratified['risk_group'] == group
            
            kmf.fit(
                durations=df_stratified.loc[mask, time_col],
                event_observed=df_stratified.loc[mask, event_col],
                label=f'{group}'
            )
            
            # Plot survival function without CI
            ax_km.plot(kmf.survival_function_.index,
                      kmf.survival_function_.iloc[:, 0].values,
                      color=group_colors.get(group, 'gray'),
                      linewidth=1.5,
                      alpha=1.0,
                      label=f'{group}')
        
        # Log-rank test
        results = logrank_test(
            df_stratified[df_stratified['risk_group'] == 'High Risk'][time_col],
            df_stratified[df_stratified['risk_group'] == 'Low Risk'][time_col],
            df_stratified[df_stratified['risk_group'] == 'High Risk'][event_col],
            df_stratified[df_stratified['risk_group'] == 'Low Risk'][event_col]
        )
        p_value = results.p_value
        
        # P-value text
        p_text = 'P < 0.001' if p_value < 0.001 else f'P = {p_value:.3f}'
        
        ax_km.text(0.95, 0.05, p_text,
                  transform=ax_km.transAxes,
                  fontsize=7, va='bottom', ha='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                          edgecolor='black', linewidth=0.5, alpha=0.8))
        
        # Styling
        ax_km.set_xlabel('Time (months)', fontsize=8, labelpad=8)
        ax_km.set_ylabel(f'{outcome_name} probability', fontsize=8)
        ax_km.text(-0.15, 1.05, panel_labels[col_idx], transform=ax_km.transAxes,
                  fontsize=10, fontweight='bold', va='top', ha='right')
        
        if col_idx == 0:
            legend = ax_km.legend(loc='lower left', fontsize=7, frameon=True,
                                 edgecolor='black')
            legend.get_frame().set_linewidth(0.5)
        
        ax_km.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax_km.set_ylim([-0.05, 1.05])
        ax_km.set_xlim([0, df_stratified[time_col].max() * 1.05])
        ax_km.tick_params(labelsize=7, length=3, width=0.5, pad=4)
        ax_km.spines['top'].set_visible(False)
        ax_km.spines['right'].set_visible(False)
        
        # Risk table
        analyzer._create_risk_table_r_style(ax_table, df_stratified, time_col, event_col,
                                    risk_groups, group_colors)
    
    # Row 3: Forest plots for DFS and OS
    for col_idx, outcome in enumerate(['DFS', 'OS']):
        ax_forest = fig.add_subplot(gs[2, col_idx])
        
        # Get Cox results for all models
        results = []
        for model_name, risk_col in analyzer.risk_scores.items():
            time_col = f'{outcome}-Time'
            event_col = outcome
            
            cox_data = analyzer.df[[time_col, event_col, risk_col]].copy()
            cox_data.columns = ['time', 'event', 'risk_score']
            cox_data = cox_data.dropna()
            
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='time', event_col='event')
            
            try:
                summary = cph.summary
                hr = np.exp(summary.loc['risk_score', 'coef'])
                ci_lower = np.exp(summary.loc['risk_score', 'coef lower 95%'])
                ci_upper = np.exp(summary.loc['risk_score', 'coef upper 95%'])
                p_value = summary.loc['risk_score', 'p']
            except:
                hr = np.exp(cph.params_['risk_score'])
                ci_vals = cph.confidence_intervals_.values[0]
                ci_lower = np.exp(ci_vals[0])
                ci_upper = np.exp(ci_vals[1])
                p_value = cph.summary['p'].values[0]
            
            results.append({
                'Model': model_name,
                'HR': hr,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'p_value': p_value
            })
        
        df_results = pd.DataFrame(results)
        y_positions = np.arange(len(df_results))
        
        # Plot forest
        for idx, row in df_results.iterrows():
            color = analyzer.colors.get(row['Model'], 'gray')
            
            ax_forest.plot([row['CI_Lower'], row['CI_Upper']], 
                          [y_positions[idx], y_positions[idx]], 
                          color=color, linewidth=1.5, alpha=0.8)
            
            # Add caps
            cap_height = 0.1
            ax_forest.plot([row['CI_Lower'], row['CI_Lower']], 
                          [y_positions[idx] - cap_height, y_positions[idx] + cap_height],
                          color=color, linewidth=1.5, alpha=0.8)
            ax_forest.plot([row['CI_Upper'], row['CI_Upper']], 
                          [y_positions[idx] - cap_height, y_positions[idx] + cap_height],
                          color=color, linewidth=1.5, alpha=0.8)
            
            ax_forest.scatter(row['HR'], y_positions[idx], 
                             s=80, color=color, marker='s', 
                             edgecolors='black', linewidth=0.5, zorder=3)
        
        ax_forest.axvline(x=1, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax_forest.set_yticks(y_positions)
        ax_forest.set_yticklabels(df_results['Model'], fontsize=8)
        ax_forest.set_xlabel('Hazard ratio (95% CI)', fontsize=8)
        ax_forest.text(-0.15, 1.05, panel_labels[2 + col_idx], 
                      transform=ax_forest.transAxes,
                      fontsize=10, fontweight='bold', va='top', ha='right')
        
        ax_forest.grid(True, alpha=0.2, axis='x')
        ax_forest.set_xlim([min(df_results['CI_Lower']) * 0.8, 
                           max(df_results['CI_Upper']) * 1.2])
        ax_forest.tick_params(labelsize=7, length=3, width=0.5)
        ax_forest.spines['top'].set_visible(False)
        ax_forest.spines['right'].set_visible(False)
        ax_forest.set_axisbelow(True)
    
    # Row 4: Risk score distributions for DFS and OS
    for col_idx, outcome in enumerate(['DFS', 'OS']):
        ax_dist = fig.add_subplot(gs[3, col_idx])
        
        # Get stratified data for Fused model
        df_stratified, _ = analyzer.stratify_by_risk(model_name='Fused',  method=method)
        
        event_col = outcome
        event_yes = df_stratified[df_stratified[event_col] == 1]['risk_score']
        event_no = df_stratified[df_stratified[event_col] == 0]['risk_score']
        
        # Violin plots
        parts = ax_dist.violinplot(
            [event_no, event_yes],
            positions=[0, 1],
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        
        colors = ['#1F77B4', '#D62728']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(0.6)
        
        # Box plots
        bp = ax_dist.boxplot(
            [event_no, event_yes],
            positions=[0, 1],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.5),
            whiskerprops=dict(color='black', linewidth=0.5),
            capprops=dict(color='black', linewidth=0.5)
        )
        
        # Add individual points
        np.random.seed(42)
        for i, data in enumerate([event_no, event_yes]):
            y = data.values
            x = np.random.normal(i, 0.02, size=len(y))
            ax_dist.scatter(x, y, alpha=0.3, s=8, color=colors[i], edgecolors='none')
        
        # Statistical test
        from scipy import stats
        stat, p_value = stats.mannwhitneyu(event_yes, event_no)
        p_text = 'P < 0.001' if p_value < 0.001 else f'P = {p_value:.3f}'
        
        y_max = max(event_no.max(), event_yes.max())
        y_min = min(event_no.min(), event_yes.min())
        y_range = y_max - y_min
        y_sig = y_max + 0.05 * y_range
        
        ax_dist.plot([0, 1], [y_sig, y_sig], 'k-', linewidth=0.5)
        ax_dist.text(0.5, y_sig + 0.01 * y_range, p_text,
                    ha='center', va='bottom', fontsize=7)
        
        ax_dist.set_xticks([0, 1])
        ax_dist.set_xticklabels([f'No event\n(n={len(event_no)})', 
                                 f'Event\n(n={len(event_yes)})'], fontsize=8)
        ax_dist.set_ylabel('Risk score', fontsize=8)
        ax_dist.text(-0.15, 1.05, panel_labels[4 + col_idx], 
                    transform=ax_dist.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='right')
        
        ax_dist.grid(True, alpha=0.2, axis='y')
        ax_dist.tick_params(labelsize=7, length=3, width=0.5)
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        ax_dist.set_axisbelow(True)
    
    # Row 5: Time-dependent ROC curves
    for col_idx, outcome in enumerate(['DFS', 'OS']):
        ax_roc = fig.add_subplot(gs[4, col_idx])
        
        from sklearn.metrics import roc_curve, auc
        
        time_col = f'{outcome}-Time'
        event_col = outcome
        risk_col = analyzer.risk_scores['Fused']
        
        time_points = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        colors_roc = ['#1F77B4', '#FF7F0E', '#2CA02C']
        
        for t, color in zip(time_points, colors_roc):
            y_binary = ((analyzer.df[time_col] <= t) & (analyzer.df[event_col] == 1)).astype(int)
            
            if y_binary.sum() >= 5:
                fpr, tpr, _ = roc_curve(y_binary, analyzer.df[risk_col])
                roc_auc = auc(fpr, tpr)
                
                ax_roc.plot(fpr, tpr, color=color, linewidth=1.5,
                           label=f'{t} mo (AUC = {roc_auc:.2f})')
        
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
        
        ax_roc.set_xlabel('1 − Specificity', fontsize=8)
        ax_roc.set_ylabel('Sensitivity', fontsize=8)
        ax_roc.text(-0.15, 1.05, panel_labels[6 + col_idx], 
                   transform=ax_roc.transAxes,
                   fontsize=10, fontweight='bold', va='top', ha='right')
        
        legend = ax_roc.legend(loc='lower right', fontsize=7, frameon=True,
                              edgecolor='black', title='Time point')
        legend.get_frame().set_linewidth(0.5)
        legend.get_title().set_fontsize(7)
        legend.get_title().set_fontweight('normal')
        
        ax_roc.grid(True, alpha=0.2)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])
        ax_roc.set_aspect('equal')
        ax_roc.tick_params(labelsize=7, length=3, width=0.5)
        ax_roc.spines['top'].set_visible(False)
        ax_roc.spines['right'].set_visible(False)
        ax_roc.set_axisbelow(True)
    
    # Save
    output_dir = Path(analyzer.output_dir)
    filename = 'comprehensive_survival_figure_nature.pdf'
    plt.savefig(output_dir / filename, dpi=dpi, bbox_inches='tight',
               pad_inches=0.15, transparent=False, facecolor='white')
    plt.savefig(output_dir / filename.replace('.pdf', '.png'), dpi=dpi,
               bbox_inches='tight', pad_inches=0.15)
    plt.savefig(output_dir / filename.replace('.pdf', '.svg'),
               bbox_inches='tight', pad_inches=0.15)
    plt.close()
    
    print(f"✓ Comprehensive survival figure saved: {filename}")


def compare_all_models_km_curves(analyzer, method='median', 
                                 figsize=(7, 10), dpi=300):  # Increased height
    """
    Create side-by-side comparison of KM curves for all three models.
    Clean curves without CI shading, better spacing.
    """
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Create grid with better spacing
    gs = GridSpec(6, 2, 
                 height_ratios=[4, 1.2, 4, 1.2, 4, 1.2],  # Increased risk table heights
                 hspace=0.25, wspace=0.3,  # Increased vertical spacing
                 left=0.08, right=0.95, top=0.96, bottom=0.04)
    
    models = ['Spatial', 'Radiomics', 'Fused']
    model_names = ['Spatial model', 'Radiomics model', 'Fused model']
    outcomes = ['DFS', 'OS']
    outcome_names = ['Disease-free survival', 'Overall survival']
    panel_labels = [['a', 'b'], ['c', 'd'], ['e', 'f']]
    
    for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
        # Stratify
        df_stratified, _ = analyzer.stratify_by_risk(model_name=model,  method=method)
        
        for outcome_idx, (outcome, outcome_name) in enumerate(zip(outcomes, outcome_names)):
            # Calculate row indices for this model
            row_km = model_idx * 2
            row_table = model_idx * 2 + 1
            
            # Create axes
            ax_km = fig.add_subplot(gs[row_km, outcome_idx])
            ax_table = fig.add_subplot(gs[row_table, outcome_idx])
            
            time_col = f'{outcome}-Time'
            event_col = outcome
            
            kmf = KaplanMeierFitter()
            risk_groups = sorted(df_stratified['risk_group'].unique(), reverse=True)
            
            # Colors
            group_colors = {'High Risk': '#D62728', 'Low Risk': '#1F77B4'}
            
            # Plot KM curves WITHOUT CI
            for group in risk_groups:
                mask = df_stratified['risk_group'] == group
                
                kmf.fit(
                    durations=df_stratified.loc[mask, time_col],
                    event_observed=df_stratified.loc[mask, event_col],
                    label=f'{group}'
                )
                
                # Plot survival function without CI
                ax_km.plot(kmf.survival_function_.index,
                          kmf.survival_function_.iloc[:, 0].values,
                          color=group_colors.get(group, 'gray'),
                          linewidth=1.5,
                          alpha=1.0,
                          label=f'{group}')
            
            # Log-rank test
            results = logrank_test(
                df_stratified[df_stratified['risk_group'] == 'High Risk'][time_col],
                df_stratified[df_stratified['risk_group'] == 'Low Risk'][time_col],
                df_stratified[df_stratified['risk_group'] == 'High Risk'][event_col],
                df_stratified[df_stratified['risk_group'] == 'Low Risk'][event_col]
            )
            p_value = results.p_value
            
            # P-value
            p_text = 'P < 0.001' if p_value < 0.001 else f'P = {p_value:.3f}'
            
            ax_km.text(0.95, 0.05, p_text,
                      transform=ax_km.transAxes,
                      fontsize=7, va='bottom', ha='right',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', linewidth=0.5, alpha=0.8))
            
            # Styling
            ax_km.set_xlabel('Time (months)', fontsize=8, labelpad=8)
            ax_km.set_ylabel(f'{outcome_name} probability', fontsize=8)
            
            # Panel label
            ax_km.text(-0.15, 1.05, panel_labels[model_idx][outcome_idx],
                      transform=ax_km.transAxes,
                      fontsize=10, fontweight='bold', va='top', ha='right')
            
            # Model name as title only on top row
            if model_idx == 0:
                ax_km.set_title(outcome_name.capitalize(), fontsize=9, 
                               fontweight='bold', pad=10)
            
            # Legend only on first panel
            if model_idx == 0 and outcome_idx == 0:
                legend = ax_km.legend(loc='lower left', fontsize=7, frameon=True,
                                     edgecolor='black')
                legend.get_frame().set_linewidth(0.5)
            
            ax_km.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax_km.set_ylim([-0.05, 1.05])
            ax_km.set_xlim([0, df_stratified[time_col].max() * 1.05])
            ax_km.tick_params(labelsize=7, length=3, width=0.5, pad=4)
            ax_km.spines['top'].set_visible(False)
            ax_km.spines['right'].set_visible(False)
            
            # Add model name on left side
            if outcome_idx == 0:
                ax_km.text(-0.30, 0.5, model_name, transform=ax_km.transAxes,
                          fontsize=9, fontweight='bold', va='center', ha='right',
                          rotation=90)
            
            # Risk table
            analyzer._create_risk_table(ax_table, df_stratified, time_col, event_col,
                                        risk_groups, group_colors)
    
    # Save
    output_dir = Path(analyzer.output_dir)
    filename = 'all_models_km_comparison_nature.pdf'
    plt.savefig(output_dir / filename, dpi=dpi, bbox_inches='tight',
               pad_inches=0.15, transparent=False, facecolor='white')
    plt.savefig(output_dir / filename.replace('.pdf', '.png'), dpi=dpi,
               bbox_inches='tight', pad_inches=0.15)
    plt.savefig(output_dir / filename.replace('.pdf', '.svg'),
               bbox_inches='tight', pad_inches=0.15)
    plt.close()
    
    print(f"✓ All models KM comparison saved: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    # predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/test_predictions_results.csv'
    # labels_excel = 'XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx'
    # output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/survival_analysis'
    # predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/model_results/TJH_predictions_results.csv'
    # labels_excel = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/SurvivalData_TJH.xlsx'
    # output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse//TJH/survival_analysis'
    predictions_csv = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/model_results/SPCH_predictions_results.csv'
    labels_excel = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/PatientLabels.xlsx'
    output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/SPCH/survival_analysis'
    print("="*80)
    print(" " * 20 + "SURVIVAL ANALYSIS - NATURE MEDICINE STYLE")
    print("="*80)
    
    # Initialize analyzer
    print("\nInitializing Survival Analyzer...")
    analyzer = SurvivalAnalyzer(
        predictions_csv=predictions_csv,
        labels_excel=labels_excel,
        output_dir=output_dir
    )
    
    # ========================================================================
    # ANALYSIS 1: FUSED MODEL (PRIMARY ANALYSIS)
    # ========================================================================
    
    print("\n" + "#"*80)
    print("PRIMARY ANALYSIS: FUSED MODEL")
    print("#"*80)
    
    fused_results = analyzer.generate_comprehensive_report(
        model_name='Fused',
        method='optimal'
    )
    
    # ========================================================================
    # ANALYSIS 2: ALL MODELS COMPARISON
    # ========================================================================
    
    print("\n" + "#"*80)
    print("COMPARATIVE ANALYSIS: ALL MODELS")
    print("#"*80)
    
    # Generate for each model
    for model_name in ['Spatial', 'Radiomics']:
        print(f"\n{'-'*80}")
        print(f"Analyzing {model_name} Model...")
        print(f"{'-'*80}")
        
        model_results = analyzer.generate_comprehensive_report(
            model_name=model_name,
            method='optimal'
        )
    
    # ========================================================================
    # ANALYSIS 3: FOREST PLOTS FOR ALL MODELS
    # ========================================================================
    
    print("\n" + "#"*80)
    print("GENERATING FOREST PLOTS")
    print("#"*80)
    
    # Cox regression forest plots
    for outcome in ['DFS', 'OS']:
        print(f"\nGenerating forest plot for {outcome}...")
        cox_results = analyzer.plot_cox_forest_nature(outcome=outcome)
    
    # ========================================================================
    # ANALYSIS 4: COMPREHENSIVE PUBLICATION FIGURES
    # ========================================================================
    
    print("\n" + "#"*80)
    print("GENERATING COMPREHENSIVE PUBLICATION FIGURES")
    print("#"*80)
    
    # Single comprehensive figure for main manuscript
    print("\nCreating comprehensive survival figure...")
    create_comprehensive_survival_figure(analyzer, method='optimal')
    
    # All models comparison figure for supplementary
    print("\nCreating all models KM comparison...")
    compare_all_models_km_curves(analyzer, method='optimal')
    
    # ========================================================================
    # FINAL SUMMARY AND REPORT
    # ========================================================================
    
    print("\n" + "="*80)
    print(" " * 25 + "ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated Files (Nature Medicine Quality):")
    print("  📊 Main Figures:")
    print("     - combined_km_curves_Fused_nature.pdf/png/svg")
    print("       (Side-by-side DFS and OS with risk tables)")
    print("     - comprehensive_survival_figure_nature.pdf/png/svg")
    print("       (Multi-panel figure: KM, forest, distributions, ROC)")
    print("     - all_models_km_comparison_nature.pdf/png/svg")
    print("       (All three models comparison)")
    print("\n  📈 Individual KM Curves:")
    print("     - km_curve_[Model]_DFS_nature.pdf/png/svg")
    print("     - km_curve_[Model]_OS_nature.pdf/png/svg")
    print("\n  📉 Forest Plots:")
    print("     - cox_forest_plot_DFS_nature.pdf/png/svg")
    print("     - cox_forest_plot_OS_nature.pdf/png/svg")
    print("\n  📊 Risk Score Distributions:")
    print("     - risk_score_distribution_[Model]_[Outcome]_nature.pdf/png/svg")
    print("\n  📈 Time-Dependent ROC:")
    print("     - time_dependent_roc_[Model]_[Outcome]_nature.pdf/png/svg")
    print("\n  📋 Data Tables:")
    print("     - survival_summary_[Model].csv")
    print("     - cox_regression_results_[Outcome].csv")
    print("     - stratified_patients_[Model].csv")
    print("\n" + "="*80)
    
    # Print key findings
    print("\n" + "="*80)
    print(" " * 25 + "KEY FINDINGS SUMMARY")
    print("="*80)
    
    try:
        # Load Fused model summary
        fused_summary = pd.read_csv(output_dir / 'survival_summary_Fused.csv')
        
        print("\n🔍 FUSED MODEL PERFORMANCE:")
        print("-"*80)
        
        for outcome in ['DFS', 'OS']:
            outcome_data = fused_summary[fused_summary['Outcome'] == outcome]
            
            outcome_full = 'Disease-Free Survival' if outcome == 'DFS' else 'Overall Survival'
            print(f"\n{outcome_full}:")
            
            for _, row in outcome_data.iterrows():
                print(f"\n  {row['Risk_Group']}:")
                print(f"    • Patients: {row['N_Patients']}")
                print(f"    • Events: {row['N_Events']} ({row['Event_Rate']})")
                
                if not pd.isna(row['Median_Survival']):
                    print(f"    • Median survival: {row['Median_Survival']:.1f} months "
                          f"(95% CI: {row['Median_CI_Lower']:.1f}–{row['Median_CI_Upper']:.1f})")
                else:
                    print(f"    • Median survival: Not reached")
                
                print(f"    • 1-year survival: {row['Survival_12mo']}")
                print(f"    • 2-year survival: {row['Survival_24mo']}")
                print(f"    • 3-year survival: {row['Survival_36mo']}")
        
        # Load Cox regression results
        cox_dfs = pd.read_csv(output_dir / 'cox_regression_results_DFS.csv')
        cox_os = pd.read_csv(output_dir / 'cox_regression_results_OS.csv')
        
        print("\n\n📊 COX REGRESSION HAZARD RATIOS:")
        print("-"*80)
        
        print("\nDisease-Free Survival:")
        for _, row in cox_dfs.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"  {row['Model']:12} HR = {row['HR']:.2f} "
                  f"(95% CI: {row['CI_Lower']:.2f}–{row['CI_Upper']:.2f}), "
                  f"P = {row['p_value']:.4f}{sig}, C-index = {row['C_index']:.3f}")
        
        print("\nOverall Survival:")
        for _, row in cox_os.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"  {row['Model']:12} HR = {row['HR']:.2f} "
                  f"(95% CI: {row['CI_Lower']:.2f}–{row['CI_Upper']:.2f}), "
                  f"P = {row['p_value']:.4f}{sig}, C-index = {row['C_index']:.3f}")
        
    except Exception as e:
        print(f"\n⚠️  Could not generate summary: {str(e)}")
    
    print("\n" + "="*80)
    print("\n💡 PUBLICATION TIPS:")
    print("   • PDF files are publication-ready for journals")
    print("   • SVG files are vector graphics for further editing")
    print("   • PNG files are for presentations and previews")
    print("   • All figures follow Nature Medicine formatting guidelines:")
    print("     - Single column width: 89 mm (3.5 inches)")
    print("     - Double column width: 183 mm (7.2 inches)")
    print("     - Font: Arial (sans-serif)")
    print("     - Font size: 5–7 pt for labels, 8–9 pt for main text")
    print("     - Line width: 0.5–1.5 pt")
    print("     - Resolution: 300–600 dpi")
    print("\n📝 MANUSCRIPT INTEGRATION:")
    print("   • Main Figure: Use comprehensive_survival_figure_nature.pdf")
    print("   • Supplementary: Use all_models_km_comparison_nature.pdf")
    print("   • Tables: Use survival_summary_*.csv and cox_regression_results_*.csv")
    print("\n🔬 STATISTICAL REPORTING:")
    print("   • Log-rank test: Use P values from KM curves")
    print("   • Cox regression: Report HR (95% CI) and P values")
    print("   • C-index: Report for model discrimination")
    print("   • Time-dependent AUC: Report for prediction accuracy at specific timepoints")
    print("\n" + "="*80 + "\n")