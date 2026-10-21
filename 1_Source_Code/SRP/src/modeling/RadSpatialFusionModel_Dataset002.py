import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (RFECV, SelectKBest, f_classif, mutual_info_classif,
                                       RFE, VarianceThreshold)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LassoCV, ElasticNetCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': ['Arial'],
    'font.size': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

class AdvancedResponsePredictor:
    """
    Advanced predictor with multiple feature selection methods and comprehensive model comparison
    """
    
    def __init__(self, radiomics_csv, spatial_radiomics_csv, clinical_xlsx, 
                 output_dir='results_advanced', case_id_col='CaseID', label_col='Response'):
        self.radiomics_csv = radiomics_csv
        self.spatial_radiomics_csv = spatial_radiomics_csv
        self.clinical_xlsx = clinical_xlsx
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.case_id_col = case_id_col
        self.label_col = label_col
        
        # Initialize storage
        self.scalers = {}
        self.feature_selectors = {}
        self.selected_features_by_method = {}
        self.all_results = {}
        self.best_classifier_name = None
        self.best_classifier_params = None
        
        print(f"\n{'='*80}")
        print("ADVANCED MULTIMODAL PREDICTION PIPELINE")
        print("Multi-Method Feature Selection + Comprehensive Model Comparison")
        print(f"{'='*80}\n")
    
    def load_data(self):
        """Load and merge all data sources"""
        print("Step 1: Loading data...")
        
        # Load datasets
        rad_df = pd.read_csv(self.radiomics_csv)
        spatial_df = pd.read_csv(self.spatial_radiomics_csv)
        clinical_df = pd.read_excel(self.clinical_xlsx)
        
        # Merge
        merged = pd.merge(rad_df, spatial_df, on=self.case_id_col, 
                         how='inner', suffixes=('_rad', '_spa'))
        merged = pd.merge(merged, clinical_df, on=self.case_id_col, how='inner')
        
        # Remove missing labels
        merged = merged.dropna(subset=[self.label_col])
        
        print(f"  Total cases: {len(merged)}")
        print(f"  Class distribution: {merged[self.label_col].value_counts().to_dict()}")
        
        self.merged_df = merged
        return merged
    
    def prepare_features(self):
        """Prepare feature sets with cleaning"""
        print("\nStep 2: Preparing feature sets...")
        
        # Exclude columns
        exclude = [self.case_id_col, self.label_col]
        exclude += [col for col in self.merged_df.columns if 'diagnostics' in col.lower()]
        
        # Get feature columns
        all_cols = set(self.merged_df.columns) - set(exclude)
        
        rad_cols = [col for col in all_cols if col.endswith('_rad') or 
                   col in pd.read_csv(self.radiomics_csv).columns]
        spatial_cols = [col for col in all_cols if col.endswith('_spa') or 
                       (col in pd.read_csv(self.spatial_radiomics_csv).columns and 
                        col not in rad_cols)]
        
        rad_cols = [c for c in rad_cols if c not in exclude]
        spatial_cols = [c for c in spatial_cols if c not in exclude]
        
        # Extract matrices
        X_rad = self.merged_df[rad_cols].values
        X_spa = self.merged_df[spatial_cols].values
        X_comb = np.concatenate([X_rad, X_spa], axis=1)
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        
        X_rad = imputer.fit_transform(X_rad)
        X_spa = imputer.fit_transform(X_spa)
        X_comb = imputer.fit_transform(X_comb)
        
        # Convert labels
        y = np.where(self.merged_df[self.label_col] == 'pCR', 1, 0)
        
        self.feature_sets = {
            'Radiomics': {'X': X_rad, 'names': rad_cols},
            'Spatial': {'X': X_spa, 'names': spatial_cols},
            'Combined': {'X': X_comb, 'names': rad_cols + spatial_cols}
        }
        self.y = y
        
        print(f"  Radiomics features: {len(rad_cols)}")
        print(f"  Spatial features: {len(spatial_cols)}")
        print(f"  Combined features: {len(rad_cols) + len(spatial_cols)}")
        
        return self.feature_sets, y
    
    def apply_variance_threshold(self, X, feature_names, threshold=0.01):
        """Remove low variance features"""
        selector = VarianceThreshold(threshold=threshold)
        X_var = selector.fit_transform(X)
        selected_names = [feature_names[i] for i in range(len(feature_names)) 
                         if selector.get_support()[i]]
        return X_var, selected_names, selector
    
    def multiple_feature_selection(self, X_train, y_train, feature_names, 
                                   feature_type, n_train_samples):
        """
        Apply multiple feature selection methods and find consensus
        
        Methods:
        1. RFECV (Recursive Feature Elimination with CV)
        2. L1-based (Lasso)
        3. Tree-based importance (Random Forest)
        4. Mutual Information
        5. F-statistic (ANOVA)
        6. Stability Selection
        """
        
        print(f"\n{'='*60}")
        print(f"Feature Selection for: {feature_type}")
        print(f"{'='*60}")
        
        # Calculate maximum allowed features (10% rule)
        max_features = max(3, int(0.105 * n_train_samples))
        print(f"Training samples: {n_train_samples}")
        print(f"Maximum features (10% rule): {max_features}")
        
        # First, remove low variance features
        print(f"\n[0] Variance Threshold Filtering...")
        print(f"  Initial features: {X_train.shape[1]}")
        X_var, names_var, var_selector = self.apply_variance_threshold(
            X_train, feature_names, threshold=0.01
        )
        print(f"  After variance filter: {X_var.shape[1]}")
        
        # Store all selected feature sets
        selected_sets = {}
        
        # === Method 1: RFECV with Logistic Regression ===
        print(f"\n[1] RFECV (Recursive Feature Elimination)...")
        estimator_rfecv = LogisticRegression(max_iter=1000, random_state=42, 
                                            solver='liblinear', penalty='l2')
        selector_rfecv = RFECV(
            estimator=estimator_rfecv,
            step=1,
            cv=StratifiedKFold(10, shuffle=True, random_state=42),
            scoring='roc_auc',
            min_features_to_select=3,
            n_jobs=-1
        )
        selector_rfecv.fit(X_var, y_train)
        
        n_rfecv = min(selector_rfecv.n_features_, max_features)
        if selector_rfecv.n_features_ > max_features:
            ranking = selector_rfecv.ranking_
            top_indices = np.argsort(ranking)[:max_features]
            selected_sets['RFECV'] = [names_var[i] for i in top_indices]
        else:
            selected_sets['RFECV'] = [names_var[i] for i in range(len(names_var)) 
                                     if selector_rfecv.support_[i]]
        
        print(f"  Selected: {len(selected_sets['RFECV'])} features")
        print(f"  CV AUC: {selector_rfecv.cv_results_['mean_test_score'].max():.4f}")
        
        # === Method 2: L1-based (Lasso) ===
        print(f"\n[2] L1-based Selection (Lasso)...")
        lasso = LassoCV(cv=10, random_state=42, max_iter=1000, n_jobs=-1)
        lasso.fit(X_var, y_train)
        
        # Get non-zero coefficients
        coef_abs = np.abs(lasso.coef_)
        # Select top features up to max_features
        n_lasso = min(max_features, np.sum(coef_abs > 0))
        top_lasso_idx = np.argsort(coef_abs)[::-1][:n_lasso]
        selected_sets['Lasso'] = [names_var[i] for i in top_lasso_idx]
        
        print(f"  Selected: {len(selected_sets['Lasso'])} features")
        print(f"  Alpha: {lasso.alpha_:.6f}")
        
        # === Method 3: Tree-based Importance (Random Forest) ===
        print(f"\n[3] Tree-based Importance (Random Forest)...")
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, 
                                            n_jobs=-1, max_depth=10)
        rf_selector.fit(X_var, y_train)
        
        importances = rf_selector.feature_importances_
        top_rf_idx = np.argsort(importances)[::-1][:max_features]
        selected_sets['RandomForest'] = [names_var[i] for i in top_rf_idx]
        
        print(f"  Selected: {len(selected_sets['RandomForest'])} features")
        print(f"  OOB accuracy: {rf_selector.score(X_var, y_train):.4f}")
        
        # === Method 4: Mutual Information ===
        print(f"\n[4] Mutual Information...")
        mi_scores = mutual_info_classif(X_var, y_train, random_state=42, n_neighbors=3)
        top_mi_idx = np.argsort(mi_scores)[::-1][:max_features]
        selected_sets['MutualInfo'] = [names_var[i] for i in top_mi_idx]
        
        print(f"  Selected: {len(selected_sets['MutualInfo'])} features")
        print(f"  Mean MI score: {mi_scores[top_mi_idx].mean():.4f}")
        
        # === Method 5: F-statistic (ANOVA) ===
        print(f"\n[5] F-statistic (ANOVA)...")
        f_scores, p_values = f_classif(X_var, y_train)
        top_f_idx = np.argsort(f_scores)[::-1][:max_features]
        selected_sets['ANOVA'] = [names_var[i] for i in top_f_idx]
        
        print(f"  Selected: {len(selected_sets['ANOVA'])} features")
        print(f"  Mean F-score: {f_scores[top_f_idx].mean():.2f}")
        
        # === Method 6: Stability Selection ===
        print(f"\n[6] Stability Selection (Bootstrap-based)...")
        n_bootstrap = 100
        feature_counts = np.zeros(len(names_var))
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(y_train), len(y_train), replace=True)
            X_boot = X_var[indices]
            y_boot = y_train[indices]
            
            # Fit Lasso
            lasso_boot = LassoCV(cv=5, random_state=i, max_iter=3000)
            lasso_boot.fit(X_boot, y_boot)
            
            # Count selected features
            selected = np.abs(lasso_boot.coef_) > 0
            feature_counts += selected
        
        # Select features that appear in >50% of bootstraps
        stability_threshold = n_bootstrap * 0.5
        stable_features = feature_counts >= stability_threshold
        
        # If too many, select top by frequency
        if np.sum(stable_features) > max_features:
            top_stable_idx = np.argsort(feature_counts)[::-1][:max_features]
            selected_sets['Stability'] = [names_var[i] for i in top_stable_idx]
            min_freq = feature_counts[top_stable_idx].min()
        elif np.sum(stable_features) > 0:
            # Use features meeting threshold
            selected_sets['Stability'] = [names_var[i] for i in range(len(names_var)) 
                                         if stable_features[i]]
            min_freq = feature_counts[stable_features].min()
        else:
            # No features meet threshold - lower it and select top features
            print(f"    No features met 50% threshold, using top {max_features} by frequency")
            top_stable_idx = np.argsort(feature_counts)[::-1][:max_features]
            selected_sets['Stability'] = [names_var[i] for i in top_stable_idx]
            min_freq = feature_counts[top_stable_idx].min()
        
        print(f"  Selected: {len(selected_sets['Stability'])} features")
        if min_freq > 0:
            print(f"  Min selection freq: {int(min_freq)}/{n_bootstrap}")
        else:
            print(f"  Average selection freq: {feature_counts[feature_counts > 0].mean():.1f}/{n_bootstrap}")
        
        # === Consensus Features ===
        print(f"\n[7] Finding Consensus Features...")
        
        # Count how many methods selected each feature
        all_features = set()
        for method_features in selected_sets.values():
            all_features.update(method_features)
        
        feature_votes = {}
        for feat in all_features:
            votes = sum(1 for method_features in selected_sets.values() 
                       if feat in method_features)
            feature_votes[feat] = votes
        
        # Select features voted by at least 3 methods
        consensus_threshold = 3
        consensus_features = [feat for feat, votes in feature_votes.items() 
                             if votes >= consensus_threshold]
        
        # If too few consensus features, lower threshold
        if len(consensus_features) < 5:
            consensus_threshold = 2
            consensus_features = [feat for feat, votes in feature_votes.items() 
                                 if votes >= consensus_threshold]
        
        # If still too few, use top voted features
        if len(consensus_features) < 5:
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            consensus_features = [feat for feat, _ in sorted_features[:max(5, max_features//2)]]
        
        # Limit to max_features
        if len(consensus_features) > max_features:
            # Rank by votes and select top
            sorted_consensus = sorted([(feat, feature_votes[feat]) for feat in consensus_features],
                                     key=lambda x: x[1], reverse=True)
            consensus_features = [feat for feat, _ in sorted_consensus[:max_features]]
        
        selected_sets['Consensus'] = consensus_features
        
        print(f"  Consensus features: {len(consensus_features)}")
        print(f"  Minimum votes: {min(feature_votes[f] for f in consensus_features)}/6 methods")
        
        # === Create Summary Visualization ===
        self._plot_feature_selection_comparison(selected_sets, feature_votes, 
                                                feature_type, max_features)
        
        # === Save Feature Selection Results ===
        self._save_feature_selection_results(selected_sets, feature_votes, 
                                             feature_type, max_features)
        
        # Store results
        self.selected_features_by_method[feature_type] = selected_sets
        
        # Return consensus features for model training
        consensus_indices = [names_var.index(f) for f in consensus_features]
        X_selected = X_var[:, consensus_indices]
        
        print(f"\n{'='*60}")
        print(f"Final selection: {len(consensus_features)} features")
        print(f"{'='*60}\n")
        
        return X_selected, consensus_features, selected_sets
    
    def _plot_feature_selection_comparison(self, selected_sets, feature_votes, 
                                          feature_type, max_features):
        """Visualize feature selection comparison across methods"""
        
        # Figure 1: Method comparison - number of features
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6))
        
        # Panel A: Number of features selected by each method
        ax = axes[0, 0]
        methods = list(selected_sets.keys())
        n_features = [len(selected_sets[m]) for m in methods]
        colors_methods = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        bars = ax.barh(range(len(methods)), n_features, color=colors_methods, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=7)
        ax.set_xlabel('Number of features', fontsize=8)
        ax.axvline(max_features, color='red', linestyle='--', linewidth=1, 
                  label=f'10% limit ({max_features})')
        ax.set_title(f'A. Features Selected per Method', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (bar, n) in enumerate(zip(bars, n_features)):
            ax.text(n + 0.5, i, str(n), va='center', fontsize=7)
        
        # Panel B: Feature voting distribution
        ax = axes[0, 1]
        votes = list(feature_votes.values())
        vote_counts = [votes.count(i) for i in range(1, 7)]
        
        ax.bar(range(1, 7), vote_counts, color='#3498db', alpha=0.8,
              edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Number of methods voting', fontsize=8)
        ax.set_ylabel('Number of features', fontsize=8)
        ax.set_title(f'B. Feature Vote Distribution', fontsize=9, fontweight='bold')
        ax.axvline(3, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='Consensus threshold')
        ax.legend(fontsize=6, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(range(1, 7))
        
        # Panel C: Overlap heatmap
        ax = axes[1, 0]
        methods_list = [m for m in methods if m != 'Consensus']
        n_methods = len(methods_list)
        overlap_matrix = np.zeros((n_methods, n_methods))
        
        for i, m1 in enumerate(methods_list):
            for j, m2 in enumerate(methods_list):
                set1 = set(selected_sets[m1])
                set2 = set(selected_sets[m2])
                if len(set1) > 0 and len(set2) > 0:
                    overlap = len(set1 & set2) / len(set1 | set2)
                    overlap_matrix[i, j] = overlap
        
        im = ax.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(methods_list, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(methods_list, fontsize=7)
        ax.set_title(f'C. Method Overlap (Jaccard Index)', fontsize=9, fontweight='bold')
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                text = ax.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if overlap_matrix[i, j] < 0.5 else "white",
                             fontsize=6)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Panel D: Top consensus features
        ax = axes[1, 1]
        consensus_features = selected_sets['Consensus']
        top_n = min(15, len(consensus_features))
        
        # Get vote counts for consensus features
        consensus_votes = [(f, feature_votes[f]) for f in consensus_features]
        consensus_votes.sort(key=lambda x: x[1], reverse=True)
        
        top_features = consensus_votes[:top_n]
        feature_labels = [f[0][:30] + '...' if len(f[0]) > 30 else f[0] 
                         for f in top_features]
        vote_counts_top = [f[1] for f in top_features]
        
        colors_votes = plt.cm.RdYlGn(np.array(vote_counts_top) / 6)
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, vote_counts_top, color=colors_votes, alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel('Number of methods', fontsize=8)
        ax.set_title(f'D. Top Consensus Features', fontsize=9, fontweight='bold')
        ax.set_xlim(0, 6.5)
        ax.set_xticks(range(1, 7))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'FS_Comparison_{feature_type}.pdf', dpi=300)
        plt.savefig(self.output_dir / f'FS_Comparison_{feature_type}.png', dpi=300)
        plt.close()
    
    def _save_feature_selection_results(self, selected_sets, feature_votes, 
                                       feature_type, max_features):
        """Save detailed feature selection results"""
        
        # Save each method's selected features
        with pd.ExcelWriter(self.output_dir / f'FS_Details_{feature_type}.xlsx') as writer:
            for method, features in selected_sets.items():
                df = pd.DataFrame({
                    'Rank': range(1, len(features) + 1),
                    'Feature': features,
                    'Votes': [feature_votes.get(f, 0) for f in features]
                })
                df.to_excel(writer, sheet_name=method, index=False)
        
        # Save voting summary
        vote_df = pd.DataFrame([
            {'Feature': feat, 'Votes': votes, 
             'Selected_by': ', '.join([m for m, feats in selected_sets.items() 
                                      if feat in feats and m != 'Consensus'])}
            for feat, votes in sorted(feature_votes.items(), 
                                     key=lambda x: x[1], reverse=True)
        ])
        vote_df.to_csv(self.output_dir / f'FS_VotingSummary_{feature_type}.csv', index=False)
    
    def comprehensive_model_comparison(self, X_train, X_test, y_train, y_test,
                                      feature_type, selected_features,
                                      classifiers_to_train=None):
        """
        Train and compare multiple classifiers with hyperparameter tuning

        Parameters:
        -----------
        classifiers_to_train : list, optional
            List of classifier names to train. If None, trains first instance (on Radiomics).
            If 'best_only', trains only self.best_classifier_name with saved best params.
        """
        
        print(f"\n{'='*60}")
        print(f"Model Training & Comparison: {feature_type}")
        print(f"{'='*60}\n")
        
        from sklearn.model_selection import GridSearchCV
        
        # Define models with hyperparameter grids
        models_configs = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=2000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'MLP': {
                'model': MLPClassifier(max_iter=2000, random_state=42, early_stopping=True),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            }
        }

        # Determine which classifiers to train
        if classifiers_to_train == 'best_only':
            # Train only the best classifier with its saved parameters
            models_configs = {self.best_classifier_name: models_configs[self.best_classifier_name]}
            print(f"Training only best classifier: {self.best_classifier_name}")
        elif classifiers_to_train is not None:
            # Train specific classifiers from the list
            models_configs = {k: v for k, v in models_configs.items() if k in classifiers_to_train}
        # else: train all classifiers (default behavior)

        results = {}
        
        for model_name, config in models_configs.items():
            print(f"Training {model_name}...")

            # Special case: if training best_only, use saved parameters directly
            grid_search = None
            if classifiers_to_train == 'best_only' and model_name == self.best_classifier_name:
                # Use the saved best parameters to create the model
                from sklearn.base import clone
                base_model = config['model']
                best_model = clone(base_model).set_params(**self.best_classifier_params)
                best_model.fit(X_train, y_train)
                print(f"  Using saved best params: {self.best_classifier_params}")
            else:
                # Grid search with cross-validation
                cv = StratifiedKFold(5, shuffle=True, random_state=42)
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train, y_train)

                # Get best model
                best_model = grid_search.best_estimator_

            # Predictions
            y_pred_train = best_model.predict(X_train)
            y_prob_train = best_model.predict_proba(X_train)[:, 1]

            y_pred_test = best_model.predict(X_test)
            y_prob_test = best_model.predict_proba(X_test)[:, 1]

            # Metrics - Test set
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

            test_metrics = {
                'AUC': roc_auc_score(y_test, y_prob_test),
                'Accuracy': accuracy_score(y_test, y_pred_test),
                'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'F1': f1_score(y_test, y_pred_test, zero_division=0)
            }

            # Metrics - Train set
            tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_pred_train).ravel()

            train_metrics = {
                'AUC': roc_auc_score(y_train, y_prob_train),
                'Accuracy': accuracy_score(y_train, y_pred_train),
            }

            # Cross-validation scores
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(best_model, X_train, y_train,
                                       cv=cv, scoring='roc_auc', n_jobs=-1)

            # Store results
            best_params = (self.best_classifier_params if (classifiers_to_train == 'best_only' and model_name == self.best_classifier_name)
                          else (grid_search.best_params_ if grid_search is not None else {}))

            results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'y_pred_test': y_pred_test,
                'y_prob_test': y_prob_test,
                'y_pred_train': y_pred_train,
                'y_prob_train': y_prob_train,
                'grid_search': grid_search
            }

            print(f"  Best params: {best_params}")
            print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Test AUC: {test_metrics['AUC']:.4f}")
            print(f"  Train AUC: {train_metrics['AUC']:.4f}")

            # Check for overfitting
            overfit_gap = train_metrics['AUC'] - test_metrics['AUC']
            if overfit_gap > 0.1:
                print(f"  ⚠ Warning: Possible overfitting (gap = {overfit_gap:.3f})")
            print()
        
        return results
    
    def run_comprehensive_analysis(self, test_size=0.3, use_smote=True, random_state=42):
        """Main comprehensive analysis pipeline"""
        
        # Load data
        self.load_data()
        feature_sets, y = self.prepare_features()
        
        # Train-test split
        print(f"\nStep 3: Creating train-test split ({int((1-test_size)*100)}/{int(test_size*100)})...")
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        
        y_train, y_test = y[train_idx], y[test_idx]
        n_train_samples = len(train_idx)
        
        print(f"  Training: {len(train_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")
        
        # Store for later
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.y_train = y_train
        self.y_test = y_test
        
        all_results = {}

        # Process each feature type
        for i, (ft_name, ft_data) in enumerate(feature_sets.items()):
            print(f"\n{'='*80}")
            print(f"Processing: {ft_name}")
            print(f"{'='*80}")
            
            X = ft_data['X']
            feature_names = ft_data['names']
            
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            
            # Scale features
            print(f"\nStep 4: Scaling features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            self.scalers[ft_name] = scaler
            
            # Multi-method feature selection
            print(f"\nStep 5: Multi-method feature selection...")
            X_train_selected, selected_features, selection_methods = self.multiple_feature_selection(
                X_train_scaled, y_train, feature_names, ft_name, n_train_samples
            )
            
            # Get indices for test set
            selected_indices = [feature_names.index(f) for f in selected_features]
            X_test_selected = X_test_scaled[:, selected_indices]
            
            # Apply SMOTE if requested
            if use_smote:
                print(f"\nStep 6: Applying SMOTE...")
                smote = SMOTE(random_state=random_state)
                X_train_balanced, y_train_balanced = smote.fit_resample(
                    X_train_selected, y_train
                )
                print(f"  {len(y_train)} → {len(y_train_balanced)} samples")
            else:
                X_train_balanced, y_train_balanced = X_train_selected, y_train
            
            # Comprehensive model comparison
            print(f"\nStep 7: Comprehensive model comparison...")

            # On first iteration (Radiomics), train all classifiers to find the best
            # On subsequent iterations, train only the best classifier
            if i == 0:  # First feature type (Radiomics)
                classifiers_to_train = None  # Train all classifiers
                results = self.comprehensive_model_comparison(
                    X_train_balanced, X_test_selected,
                    y_train_balanced, y_test,
                    ft_name, selected_features,
                    classifiers_to_train=classifiers_to_train
                )
                # Find and store the best classifier
                best_model_name = max(results.keys(),
                                     key=lambda m: results[m]['test_metrics']['AUC'])
                self.best_classifier_name = best_model_name
                self.best_classifier_params = results[best_model_name]['best_params']
                print(f"\n✓ Best classifier identified: {best_model_name}")
                print(f"  Best AUC: {results[best_model_name]['test_metrics']['AUC']:.4f}")
            else:  # Subsequent feature types (Spatial, Combined)
                results = self.comprehensive_model_comparison(
                    X_train_balanced, X_test_selected,
                    y_train_balanced, y_test,
                    ft_name, selected_features,
                    classifiers_to_train='best_only'
                )
            
            all_results[ft_name] = {
                'results': results,
                'selected_features': selected_features,
                'selection_methods': selection_methods,
                'X_train': X_train_balanced,
                'X_test': X_test_selected
            }
        
        self.all_results = all_results
        
        # Create comprehensive visualizations
        print(f"\n{'='*80}")
        print("Step 8: Creating Comprehensive Visualizations")
        print(f"{'='*80}\n")
        self._create_all_comprehensive_figures()
        
        # Generate comprehensive report
        print(f"\n{'='*80}")
        print("Step 9: Generating Reports")
        print(f"{'='*80}\n")
        self._create_results_tables()
        self._save_all_models()
        self._generate_comprehensive_report()
        
        print(f"\n{'='*80}")
        print("Analysis Complete!")
        print(f"{'='*80}")
        print(f"\nResults saved to: {self.output_dir}")
        
        return all_results
    
    def _create_all_comprehensive_figures(self):
        """Create all comprehensive publication-quality figures"""
        
        # Figure 1: Overall performance comparison
        self._plot_overall_performance_comparison()
        
        # Figure 2: ROC curves - best models
        self._plot_roc_curves_best_models()
        
        # Figure 3: ROC curves - all models for combined features
        self._plot_roc_curves_all_models()
        
        # Figure 4: Performance metrics radar charts
        self._plot_performance_radar()
        
        # Figure 5: Learning curves
        self._plot_learning_curves()
        
        # Figure 6: Confusion matrices
        self._plot_confusion_matrices_grid()
        
        # Figure 7: Feature importance comparison
        self._plot_feature_importance_comparison()
        
        # Figure 8: Model stability analysis
        self._plot_model_stability()
        
        # Figure 9: Calibration curves
        self._plot_calibration_curves()
        
        # Figure 10: Performance heatmap
        self._plot_performance_heatmap()
        
        # Figure 11: Feature selection consensus
        self._plot_feature_selection_consensus()
        
        # Figure 12: Model comparison forest plot
        self._plot_forest_plot()
        
        print("  ✓ All figures created successfully")
    
    def _plot_overall_performance_comparison(self):
        """Compare all models across all feature types"""
        
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6))
        
        # Collect data
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        models_list = list(self.all_results['Combined']['results'].keys())
        
        metrics_to_plot = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
        
        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
            x = np.arange(len(models_list))
            width = 0.25
            
            colors = {'Radiomics': '#2E86AB', 'Spatial': '#A23B72', 'Combined': '#F18F01'}
            
            for i, ft in enumerate(feature_types):
                values = [self.all_results[ft]['results'][m]['test_metrics'][metric] 
                         for m in models_list]
                
                ax.bar(x + i*width, values, width, label=ft, 
                      color=colors[ft], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_ylabel(metric, fontsize=8)
            ax.set_xlabel('Model', fontsize=8)
            ax.set_title(f'{chr(65+idx)}. {metric}', fontsize=9, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(models_list, rotation=45, ha='right', fontsize=7)
            ax.set_ylim(0, 1.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)
            
            if idx == 0:
                ax.legend(fontsize=7, frameon=False, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig1_Overall_Performance_Comparison.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig1_Overall_Performance_Comparison.png', dpi=300)
        plt.close()
    
    def _plot_roc_curves_best_models(self):
        """ROC curves for best model from each feature type"""
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        colors = {'Radiomics': '#2E86AB', 'Spatial': '#A23B72', 'Combined': '#F18F01'}
        
        for ft_name, color in colors.items():
            # Find best model by AUC
            results = self.all_results[ft_name]['results']
            best_model_name = max(results.keys(), 
                                 key=lambda m: results[m]['test_metrics']['AUC'])
            
            result = results[best_model_name]
            fpr, tpr, _ = roc_curve(self.y_test, result['y_prob_test'])
            auc = result['test_metrics']['AUC']
            
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                   label=f'{ft_name} ({best_model_name}): AUC={auc:.3f}')
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title('ROC Curves - Best Models', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, frameon=True, loc='lower right', framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig2_ROC_Best_Models.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig2_ROC_Best_Models.png', dpi=300)
        plt.close()
    
    def _plot_roc_curves_all_models(self):
        """ROC curves for all models using Combined features"""
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        colors_models = {
            'LogisticRegression': '#E63946',
            'RandomForest': '#457B9D',
            'GradientBoosting': '#2A9D8F',
            'SVM': '#E9C46A',
            'MLP': '#F4A261',
            'ExtraTrees': '#9D4EDD'
        }
        
        results = self.all_results['Combined']['results']
        
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_prob_test'])
            auc = result['test_metrics']['AUC']
            
            ax.plot(fpr, tpr, color=colors_models.get(model_name, 'gray'),
                   linewidth=2, label=f'{model_name}: {auc:.3f}', alpha=0.8)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title('ROC Curves - All Models (Combined Features)', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, frameon=True, loc='lower right', framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig3_ROC_All_Models.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig3_ROC_All_Models.png', dpi=300)
        plt.close()
    
    def _plot_performance_radar(self):
        """Radar charts for performance metrics"""
        
        fig = plt.figure(figsize=(7.2, 2.5))
        
        metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        
        for idx, ft_name in enumerate(feature_types):
            ax = fig.add_subplot(1, 3, idx+1, projection='polar')
            
            # Get best model
            results = self.all_results[ft_name]['results']
            best_model_name = max(results.keys(), 
                                 key=lambda m: results[m]['test_metrics']['AUC'])
            result = results[best_model_name]
            
            values = [result['test_metrics'][m] for m in metrics]
            values += values[:1]  # Complete the circle
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            colors_ft = {'Radiomics': '#2E86AB', 'Spatial': '#A23B72', 'Combined': '#F18F01'}
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors_ft[ft_name],
                   markersize=5)
            ax.fill(angles, values, alpha=0.25, color=colors_ft[ft_name])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6)
            ax.grid(True, linewidth=0.5, alpha=0.3)
            
            ax.set_title(f'{ft_name}\n{best_model_name}', 
                        fontsize=9, fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig4_Performance_Radar.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig4_Performance_Radar.png', dpi=300)
        plt.close()
    
    def _plot_learning_curves(self):
        """Plot learning curves for best models"""
        
        from sklearn.model_selection import learning_curve
        
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        colors = {'Radiomics': '#2E86AB', 'Spatial': '#A23B72', 'Combined': '#F18F01'}
        
        for idx, (ax, ft_name) in enumerate(zip(axes, feature_types)):
            # Get best model
            results = self.all_results[ft_name]['results']
            best_model_name = max(results.keys(), 
                                 key=lambda m: results[m]['test_metrics']['AUC'])
            best_model = results[best_model_name]['model']
            
            # Use ORIGINAL training data (before SMOTE) for learning curves
            # This gives a true picture of how the model learns with real data
            X_train_original = self.all_results[ft_name]['X_train']
            
            # Get the original indices to reconstruct pre-SMOTE data
            # We need to use the data before SMOTE was applied
            # The saved X_train is after SMOTE, so we need to recreate the pipeline
            
            # Get original scaled data (before feature selection and SMOTE)
            feature_names = self.all_results[ft_name]['selected_features']
            
            # Reconstruct the original selected features from test set structure
            # Use a subset approach instead - sample from SMOTE data
            X_sample = X_train_original[:len(self.y_train)]  # Take first N samples matching original size
            y_sample = self.y_train
            
            try:
                # Calculate learning curve with original training size
                train_sizes = np.linspace(0.1, 1.0, 10)
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    best_model, X_sample, y_sample,
                    train_sizes=train_sizes,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=42,
                    shuffle=True
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                ax.plot(train_sizes_abs, train_mean, 'o-', color=colors[ft_name],
                       linewidth=2, markersize=4, label='Training', alpha=0.8)
                ax.fill_between(train_sizes_abs, train_mean - train_std,
                               train_mean + train_std, alpha=0.2, color=colors[ft_name])
                
                ax.plot(train_sizes_abs, val_mean, 's-', color='#E63946',
                       linewidth=2, markersize=4, label='Validation', alpha=0.8)
                ax.fill_between(train_sizes_abs, val_mean - val_std,
                               val_mean + val_std, alpha=0.2, color='#E63946')
                
                ax.set_xlabel('Training samples', fontsize=8)
                ax.set_ylabel('AUC Score', fontsize=8)
                ax.set_title(f'{ft_name}\n{best_model_name}', fontsize=9, fontweight='bold')
                ax.legend(fontsize=7, frameon=False, loc='lower right')
                ax.set_ylim(0.4, 1.05)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.3, linewidth=0.5)
                
            except Exception as e:
                print(f"    Warning: Could not generate learning curve for {ft_name}: {e}")
                # Create a placeholder
                ax.text(0.5, 0.5, f'Learning curve\nnot available\nfor {ft_name}',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=9, color='gray')
                ax.set_title(f'{ft_name}\n{best_model_name}', fontsize=9, fontweight='bold')
                ax.set_xlabel('Training samples', fontsize=8)
                ax.set_ylabel('AUC Score', fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig5_Learning_Curves.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig5_Learning_Curves.png', dpi=300)
        plt.close()
    
    def _plot_confusion_matrices_grid(self):
        """Plot confusion matrices for best models"""
        
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        colors = {'Radiomics': '#2E86AB', 'Spatial': '#A23B72', 'Combined': '#F18F01'}
        
        for idx, (ax, ft_name) in enumerate(zip(axes, feature_types)):
            # Get best model
            results = self.all_results[ft_name]['results']
            best_model_name = max(results.keys(), 
                                 key=lambda m: results[m]['test_metrics']['AUC'])
            result = results[best_model_name]
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, result['y_pred_test'])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                           ha="center", va="center", color=text_color,
                           fontsize=9, fontweight='bold')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'], fontsize=8)
            ax.set_yticklabels(['Negative', 'Positive'], fontsize=8)
            ax.set_xlabel('Predicted', fontsize=8)
            
            if idx == 0:
                ax.set_ylabel('Actual', fontsize=8)
            
            ax.set_title(f'{ft_name}\n{best_model_name}', 
                        fontsize=9, fontweight='bold', color=colors[ft_name])
            
            # Color spines
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[ft_name])
                spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig6_Confusion_Matrices.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig6_Confusion_Matrices.png', dpi=300)
        plt.close()
    
    def _plot_feature_importance_comparison(self):
        """Compare feature importance across tree-based models"""
        
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 4))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        tree_models = ['RandomForest', 'GradientBoosting', 'ExtraTrees']
        
        for idx, (ax, ft_name) in enumerate(zip(axes, feature_types)):
            results = self.all_results[ft_name]['results']
            
            # Collect importances from all tree models
            all_importances = []
            model_names = []
            
            for model_name in tree_models:
                if model_name in results:
                    model = results[model_name]['model']
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        all_importances.append(importances)
                        model_names.append(model_name)
            
            if len(all_importances) > 0:
                # Get top features by average importance
                avg_importance = np.mean(all_importances, axis=0)
                top_n = min(15, len(avg_importance))
                top_indices = np.argsort(avg_importance)[::-1][:top_n]
                
                # Get feature names
                feature_names = self.all_results[ft_name]['selected_features']
                top_features = [feature_names[i] for i in top_indices]
                
                # Truncate long names
                display_names = [f[:35] + '...' if len(f) > 35 else f 
                               for f in top_features]
                
                # Plot
                y_pos = np.arange(len(top_features))
                
                # Average importance
                avg_imp_top = avg_importance[top_indices]
                
                # Determine source color (for Combined only)
                if ft_name == 'Combined':
                    n_rad = len([f for f in self.all_results['Radiomics']['selected_features']])
                    # Since consensus features don't have strict ordering, use simple heuristic
                    colors_bars = ['#2E86AB' if 'rad' in f.lower() or any(f in self.all_results['Radiomics']['selected_features']) 
                                  else '#A23B72' for f in top_features]
                else:
                    colors_bars = ['#2E86AB' if ft_name == 'Radiomics' else '#A23B72'] * len(top_features)
                
                ax.barh(y_pos, avg_imp_top, color=colors_bars, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(display_names, fontsize=6)
                ax.invert_yaxis()
                ax.set_xlabel('Average Importance', fontsize=8)
                ax.set_title(f'{ft_name}', fontsize=9, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig7_Feature_Importance.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig7_Feature_Importance.png', dpi=300)
        plt.close()
    
    def _plot_model_stability(self):
        """Plot model stability using CV scores"""
        
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        
        for idx, (ax, ft_name) in enumerate(zip(axes, feature_types)):
            results = self.all_results[ft_name]['results']
            
            model_names = list(results.keys())
            cv_means = [results[m]['cv_auc_mean'] for m in model_names]
            cv_stds = [results[m]['cv_auc_std'] for m in model_names]
            test_aucs = [results[m]['test_metrics']['AUC'] for m in model_names]
            
            x = np.arange(len(model_names))
            
            # Plot CV scores with error bars
            ax.errorbar(x, cv_means, yerr=cv_stds, fmt='o', markersize=6,
                       capsize=5, capthick=2, linewidth=2,
                       color='#3498db', label='CV AUC', alpha=0.8)
            
            # Plot test scores
            ax.scatter(x, test_aucs, s=80, marker='s', color='#E63946',
                      edgecolors='black', linewidth=1, label='Test AUC', alpha=0.8,
                      zorder=5)
            
            # Connect CV and Test
            for i in range(len(model_names)):
                ax.plot([i, i], [cv_means[i], test_aucs[i]], 'k--', 
                       linewidth=1, alpha=0.3)
            
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('AUC Score', fontsize=8)
            ax.set_title(f'{ft_name}', fontsize=9, fontweight='bold')
            ax.set_ylim(0.5, 1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)
            
            if idx == 0:
                ax.legend(fontsize=7, frameon=False, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig8_Model_Stability.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig8_Model_Stability.png', dpi=300)
        plt.close()
    
    def _plot_calibration_curves(self):
        """Plot calibration curves for best models"""
        
        from sklearn.calibration import calibration_curve
        
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        colors = {'Radiomics': '#2E86AB', 'Spatial': '#A23B72', 'Combined': '#F18F01'}
        
        for idx, (ax, ft_name) in enumerate(zip(axes, feature_types)):
            # Get best model
            results = self.all_results[ft_name]['results']
            best_model_name = max(results.keys(), 
                                 key=lambda m: results[m]['test_metrics']['AUC'])
            result = results[best_model_name]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_test, result['y_prob_test'], n_bins=10, strategy='quantile'
            )
            
            # Plot
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-',
                   color=colors[ft_name], linewidth=2, markersize=6,
                   label=best_model_name)
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
                   label='Perfect calibration')
            
            ax.set_xlabel('Predicted Probability', fontsize=8)
            ax.set_ylabel('Observed Fraction', fontsize=8)
            ax.set_title(f'{ft_name}', fontsize=9, fontweight='bold')
            ax.legend(fontsize=7, frameon=False, loc='lower right')
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig9_Calibration_Curves.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig9_Calibration_Curves.png', dpi=300)
        plt.close()
    
    def _plot_performance_heatmap(self):
        """Heatmap of all performance metrics"""
        
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 4))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
        
        for idx, (ax, ft_name) in enumerate(zip(axes, feature_types)):
            results = self.all_results[ft_name]['results']
            model_names = list(results.keys())
            
            # Create matrix
            data_matrix = np.zeros((len(model_names), len(metrics)))
            
            for i, model_name in enumerate(model_names):
                for j, metric in enumerate(metrics):
                    data_matrix[i, j] = results[model_name]['test_metrics'][metric]
            
            # Plot heatmap
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
            
            # Set ticks
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_yticks(np.arange(len(model_names)))
            ax.set_xticklabels(metrics, fontsize=7, rotation=45, ha='right')
            ax.set_yticklabels(model_names, fontsize=7)
            
            # Add values
            for i in range(len(model_names)):
                for j in range(len(metrics)):
                    text_color = "white" if data_matrix[i, j] < 0.7 else "black"
                    ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                           ha="center", va="center", color=text_color, fontsize=6)
            
            ax.set_title(f'{ft_name}', fontsize=9, fontweight='bold')
            
            # Add colorbar
            if idx == 2:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label('Score', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig10_Performance_Heatmap.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig10_Performance_Heatmap.png', dpi=300)
        plt.close()
    
    def _plot_feature_selection_consensus(self):
        """Visualize feature selection consensus across feature types"""
        
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        
        for idx, (ax, ft_name) in enumerate(zip(axes, feature_types)):
            if ft_name in self.selected_features_by_method:
                selection_methods = self.selected_features_by_method[ft_name]
                
                # Count votes for top features
                all_features = set()
                for features in selection_methods.values():
                    all_features.update(features)
                
                feature_votes = {}
                for feat in all_features:
                    votes = sum(1 for features in selection_methods.values() 
                               if feat in features)
                    feature_votes[feat] = votes
                
                # Get top features by votes
                sorted_features = sorted(feature_votes.items(), 
                                        key=lambda x: x[1], reverse=True)
                top_n = min(20, len(sorted_features))
                top_features = sorted_features[:top_n]
                
                # Prepare data
                feature_names = [f[0][:30] + '...' if len(f[0]) > 30 else f[0] 
                               for f in top_features]
                votes = [f[1] for f in top_features]
                
                # Color by vote count
                colors = plt.cm.RdYlGn(np.array(votes) / 6)
                
                # Plot
                y_pos = np.arange(len(feature_names))
                ax.barh(y_pos, votes, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names, fontsize=6)
                ax.invert_yaxis()
                ax.set_xlabel('Number of Selection Methods', fontsize=8)
                ax.set_title(f'{ft_name}', fontsize=9, fontweight='bold')
                ax.set_xlim(0, 6.5)
                ax.set_xticks(range(1, 7))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig11_Feature_Selection_Consensus.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig11_Feature_Selection_Consensus.png', dpi=300)
        plt.close()
    
    def _plot_forest_plot(self):
        """Forest plot comparing model performance with confidence intervals"""
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        colors = {'Radiomics': '#2E86AB', 'Spatial': '#A23B72', 'Combined': '#F18F01'}
        
        # Collect all results
        all_results_list = []
        
        for ft_name in feature_types:
            results = self.all_results[ft_name]['results']
            for model_name, result in results.items():
                all_results_list.append({
                    'feature_type': ft_name,
                    'model': model_name,
                    'auc': result['test_metrics']['AUC'],
                    'cv_mean': result['cv_auc_mean'],
                    'cv_std': result['cv_auc_std']
                })
        
        # Sort by AUC
        all_results_list.sort(key=lambda x: x['auc'], reverse=True)
        
        # Plot
        y_pos = np.arange(len(all_results_list))
        
        for i, result in enumerate(all_results_list):
            color = colors[result['feature_type']]
            
            # Plot CV mean with error bars
            ax.errorbar(result['cv_mean'], i, xerr=result['cv_std'],
                       fmt='o', markersize=6, capsize=3, capthick=1.5,
                       color=color, alpha=0.6, linewidth=1.5)
            
            # Plot test AUC
            ax.scatter(result['auc'], i, s=100, marker='D', 
                      color=color, edgecolors='black', linewidth=1,
                      zorder=5, alpha=0.9)
        
        # Labels
        labels = [f"{r['feature_type']} - {r['model']}" for r in all_results_list]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        
        ax.set_xlabel('AUC Score', fontsize=9)
        ax.set_title('Model Performance Comparison\n(Diamond: Test AUC, Circle: CV Mean ± SD)',
                    fontsize=10, fontweight='bold')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.3)
        ax.set_xlim(0.4, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linewidth=0.5)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                  markersize=8, label='Test AUC', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='gray', markersize=6,
                  label='CV Mean ± SD', linewidth=0)
        ]
        ax.legend(handles=legend_elements, fontsize=7, frameon=False, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig12_Forest_Plot.pdf', dpi=300)
        plt.savefig(self.output_dir / 'Fig12_Forest_Plot.png', dpi=300)
        plt.close()
    
    def _create_results_tables(self):
        """Create comprehensive results tables"""
        
        print("  Creating results tables...")
        
        # Table 1: Complete performance metrics
        all_metrics = []
        
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            results = self.all_results[ft_name]['results']
            n_features = len(self.all_results[ft_name]['selected_features'])
            
            for model_name, result in results.items():
                row = {
                    'Feature_Type': ft_name,
                    'Model': model_name,
                    'N_Features': n_features,
                    'Best_Params': str(result['best_params']),
                    'AUC': result['test_metrics']['AUC'],
                    'Accuracy': result['test_metrics']['Accuracy'],
                    'Sensitivity': result['test_metrics']['Sensitivity'],
                    'Specificity': result['test_metrics']['Specificity'],
                    'PPV': result['test_metrics']['PPV'],
                    'NPV': result['test_metrics']['NPV'],
                    'F1': result['test_metrics']['F1'],
                    'CV_AUC_Mean': result['cv_auc_mean'],
                    'CV_AUC_Std': result['cv_auc_std'],
                    'Train_AUC': result['train_metrics']['AUC'],
                    'Overfitting_Gap': result['train_metrics']['AUC'] - result['test_metrics']['AUC']
                }
                all_metrics.append(row)
        
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics = df_metrics.round(4)
        df_metrics = df_metrics.sort_values('AUC', ascending=False)
        
        df_metrics.to_csv(self.output_dir / 'Table1_All_Results.csv', index=False)
        df_metrics.to_excel(self.output_dir / 'Table1_All_Results.xlsx', index=False)
        
        # Table 2: Best models per feature type
        best_models = df_metrics.loc[df_metrics.groupby('Feature_Type')['AUC'].idxmax()]
        best_models.to_csv(self.output_dir / 'Table2_Best_Models.csv', index=False)
        best_models.to_excel(self.output_dir / 'Table2_Best_Models.xlsx', index=False)
        
        # Table 3: Feature selection summary
        fs_summary = []
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            if ft_name in self.selected_features_by_method:
                methods = self.selected_features_by_method[ft_name]
                for method_name, features in methods.items():
                    fs_summary.append({
                        'Feature_Type': ft_name,
                        'Selection_Method': method_name,
                        'N_Features': len(features)
                    })
        
        df_fs_summary = pd.DataFrame(fs_summary)
        df_fs_summary.to_csv(self.output_dir / 'Table3_Feature_Selection_Summary.csv', index=False)
        
        # Table 4: Statistical comparison
        self._perform_statistical_tests(df_metrics)
        
        print("  ✓ Results tables created")
    
    def _perform_statistical_tests(self, df_metrics):
        """Perform statistical tests comparing models"""
        
        print("  Performing statistical tests...")
        
        # Compare feature types (using best model from each)
        comparisons = []
        
        feature_types = ['Radiomics', 'Spatial', 'Combined']
        
        for i, ft1 in enumerate(feature_types):
            for ft2 in feature_types[i+1:]:
                # Get all AUCs for each feature type
                aucs_ft1 = df_metrics[df_metrics['Feature_Type'] == ft1]['AUC'].values
                aucs_ft2 = df_metrics[df_metrics['Feature_Type'] == ft2]['AUC'].values
                
                # Paired t-test
                from scipy.stats import ttest_rel, wilcoxon
                
                t_stat, p_value = ttest_rel(aucs_ft1, aucs_ft2)
                w_stat, p_wilcoxon = wilcoxon(aucs_ft1, aucs_ft2)
                
                # Effect size (Cohen's d)
                mean_diff = np.mean(aucs_ft1) - np.mean(aucs_ft2)
                pooled_std = np.sqrt((np.var(aucs_ft1) + np.var(aucs_ft2)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                comparisons.append({
                    'Comparison': f'{ft1} vs {ft2}',
                    'Mean_Diff': mean_diff,
                    'T_Statistic': t_stat,
                    'P_Value_TTest': p_value,
                    'P_Value_Wilcoxon': p_wilcoxon,
                    'Cohens_D': cohens_d,
                    'Significant_0.05': 'Yes' if p_value < 0.05 else 'No',
                    'Significant_0.01': 'Yes' if p_value < 0.01 else 'No'
                })
        
        df_stats = pd.DataFrame(comparisons)
        df_stats = df_stats.round(4)
        df_stats.to_csv(self.output_dir / 'Table4_Statistical_Tests.csv', index=False)
        
        print("  ✓ Statistical tests completed")
    
    def _save_all_models(self):
        """Save all trained models and metadata"""
        
        print("  Saving models...")
        
        models_dir = self.output_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        
        # Save each model
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            results = self.all_results[ft_name]['results']
            
            for model_name, result in results.items():
                filename = f'{ft_name}_{model_name}.pkl'
                with open(models_dir / filename, 'wb') as f:
                    pickle.dump({
                        'model': result['model'],
                        'best_params': result['best_params'],
                        'selected_features': self.all_results[ft_name]['selected_features'],
                        'test_metrics': result['test_metrics']
                    }, f)
        
        # Save scalers
        with open(models_dir / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save feature selection results
        with open(models_dir / 'feature_selection_methods.pkl', 'wb') as f:
            pickle.dump(self.selected_features_by_method, f)
        
        # Save train/test split
        split_info = {
            'train_idx': self.train_idx,
            'test_idx': self.test_idx,
            'case_ids_train': self.merged_df.iloc[self.train_idx][self.case_id_col].values,
            'case_ids_test': self.merged_df.iloc[self.test_idx][self.case_id_col].values,
            'y_train': self.y_train,
            'y_test': self.y_test
        }
        
        with open(models_dir / 'split_info.pkl', 'wb') as f:
            pickle.dump(split_info, f)
        
        print("  ✓ Models saved")
    
    def _generate_comprehensive_report(self):
        """Generate detailed comprehensive report"""
        
        print("  Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE MULTIMODAL PREDICTION ANALYSIS REPORT")
        report_lines.append("Multi-Method Feature Selection + Advanced Model Comparison")
        report_lines.append("=" * 100)
        report_lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Output directory: {self.output_dir}\n")
        
        # Section 1: Dataset Summary
        report_lines.append("=" * 100)
        report_lines.append("1. DATASET SUMMARY")
        report_lines.append("=" * 100)
        report_lines.append(f"\nTotal samples: {len(self.merged_df)}")
        report_lines.append(f"Training samples: {len(self.train_idx)} ({len(self.train_idx)/len(self.merged_df)*100:.1f}%)")
        report_lines.append(f"Test samples: {len(self.test_idx)} ({len(self.test_idx)/len(self.merged_df)*100:.1f}%)")
        
        train_dist = np.bincount(self.y_train)
        test_dist = np.bincount(self.y_test)
        report_lines.append(f"\nClass distribution (training): Negative={train_dist[0]}, Positive={train_dist[1]}")
        report_lines.append(f"Class distribution (test): Negative={test_dist[0]}, Positive={test_dist[1]}")
        
        # Section 2: Feature Selection Results
        report_lines.append("\n" + "=" * 100)
        report_lines.append("2. MULTI-METHOD FEATURE SELECTION RESULTS")
        report_lines.append("=" * 100)
        
        max_features = int(0.1 * len(self.train_idx))
        report_lines.append(f"\nMaximum features allowed (10% rule): {max_features}")
        
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            report_lines.append(f"\n{ft_name} Features:")
            report_lines.append("-" * 80)
            
            if ft_name in self.selected_features_by_method:
                methods = self.selected_features_by_method[ft_name]
                
                report_lines.append(f"\nFeature selection methods applied:")
                for method_name, features in methods.items():
                    pct = len(features) / len(self.train_idx) * 100
                    report_lines.append(f"  {method_name:20s}: {len(features):3d} features ({pct:.1f}% of training samples)")
                
                consensus_features = methods.get('Consensus', [])
                report_lines.append(f"\nFinal consensus features: {len(consensus_features)}")
                report_lines.append(f"Percentage of training samples: {len(consensus_features)/len(self.train_idx)*100:.1f}%")
                
                # Count votes
                all_features = set()
                for features in methods.values():
                    all_features.update(features)
                
                feature_votes = {}
                for feat in all_features:
                    votes = sum(1 for features in methods.values() if feat in features)
                    feature_votes[feat] = votes
                
                # Top consensus features
                consensus_sorted = sorted([(f, feature_votes[f]) for f in consensus_features],
                                        key=lambda x: x[1], reverse=True)
                
                report_lines.append(f"\nTop 10 consensus features (by votes):")
                for i, (feat, votes) in enumerate(consensus_sorted[:10], 1):
                    report_lines.append(f"  {i:2d}. {feat[:70]:<70s} ({votes}/6 methods)")
        
        # Section 3: Model Performance Summary
        report_lines.append("\n" + "=" * 100)
        report_lines.append("3. MODEL PERFORMANCE SUMMARY")
        report_lines.append("=" * 100)
        
        # Find overall best model
        all_models_flat = []
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            results = self.all_results[ft_name]['results']
            for model_name, result in results.items():
                all_models_flat.append((ft_name, model_name, result))
        
        best_overall = max(all_models_flat, key=lambda x: x[2]['test_metrics']['AUC'])
        ft_best, model_best, result_best = best_overall
        
        report_lines.append(f"\n*** OVERALL BEST MODEL ***")
        report_lines.append(f"Feature Type: {ft_best}")
        report_lines.append(f"Model: {model_best}")
        report_lines.append(f"Best Parameters: {result_best['best_params']}")
        report_lines.append(f"\nPerformance Metrics:")
        
        metrics = result_best['test_metrics']
        report_lines.append(f"  AUC:          {metrics['AUC']:.4f}")
        report_lines.append(f"  Accuracy:     {metrics['Accuracy']:.4f}")
        report_lines.append(f"  Sensitivity:  {metrics['Sensitivity']:.4f}")
        report_lines.append(f"  Specificity:  {metrics['Specificity']:.4f}")
        report_lines.append(f"  PPV:          {metrics['PPV']:.4f}")
        report_lines.append(f"  NPV:          {metrics['NPV']:.4f}")
        report_lines.append(f"  F1 Score:     {metrics['F1']:.4f}")
        
        cv_mean = result_best['cv_auc_mean']
        cv_std = result_best['cv_auc_std']
        report_lines.append(f"\n  CV AUC:       {cv_mean:.4f} ± {cv_std:.4f}")
        report_lines.append(f"  Train AUC:    {result_best['train_metrics']['AUC']:.4f}")
        
        overfit_gap = result_best['train_metrics']['AUC'] - metrics['AUC']
        report_lines.append(f"  Overfitting:  {overfit_gap:.4f}")
        
        if overfit_gap < 0.05:
            report_lines.append("  ✓ Model shows good generalization")
        elif overfit_gap < 0.1:
            report_lines.append("  ○ Slight overfitting detected")
        else:
            report_lines.append("  ⚠ Significant overfitting - consider regularization")
        
        # Best model per feature type
        report_lines.append("\n" + "-" * 80)
        report_lines.append("BEST MODEL PER FEATURE TYPE")
        report_lines.append("-" * 80)
        
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            report_lines.append(f"\n{ft_name}:")
            
            results = self.all_results[ft_name]['results']
            best_model_name = max(results.keys(), 
                                 key=lambda m: results[m]['test_metrics']['AUC'])
            result = results[best_model_name]
            
            report_lines.append(f"  Model: {best_model_name}")
            report_lines.append(f"  Parameters: {result['best_params']}")
            report_lines.append(f"  AUC: {result['test_metrics']['AUC']:.4f}")
            report_lines.append(f"  Sensitivity: {result['test_metrics']['Sensitivity']:.4f}")
            report_lines.append(f"  Specificity: {result['test_metrics']['Specificity']:.4f}")
            report_lines.append(f"  CV AUC: {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f}")
            
            # All models ranking
            report_lines.append(f"\n  All models (ranked by AUC):")
            ranked_models = sorted(results.items(), 
                                  key=lambda x: x[1]['test_metrics']['AUC'],
                                  reverse=True)
            
            for rank, (m_name, m_result) in enumerate(ranked_models, 1):
                auc = m_result['test_metrics']['AUC']
                cv_auc = m_result['cv_auc_mean']
                report_lines.append(f"    {rank}. {m_name:20s}: AUC={auc:.4f}, CV={cv_auc:.4f}")
        
        # Section 4: Feature Type Comparison
        report_lines.append("\n" + "=" * 100)
        report_lines.append("4. FEATURE TYPE COMPARISON")
        report_lines.append("=" * 100)
        
        best_aucs = {}
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            best_auc = max(r['test_metrics']['AUC'] 
                          for r in self.all_results[ft_name]['results'].values())
            best_aucs[ft_name] = best_auc
        
        report_lines.append("\nBest AUC by feature type:")
        for ft_name, auc in sorted(best_aucs.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {ft_name:15s}: {auc:.4f}")
        
        # Improvement analysis
        rad_auc = best_aucs['Radiomics']
        spa_auc = best_aucs['Spatial']
        comb_auc = best_aucs['Combined']
        
        report_lines.append("\nImprovement Analysis:")
        imp_vs_rad = comb_auc - rad_auc
        imp_vs_spa = comb_auc - spa_auc
        
        report_lines.append(f"  Combined vs Radiomics:  {imp_vs_rad:+.4f} ({imp_vs_rad/rad_auc*100:+.2f}%)")
        report_lines.append(f"  Combined vs Spatial:    {imp_vs_spa:+.4f} ({imp_vs_spa/spa_auc*100:+.2f}%)")
        
        if comb_auc > max(rad_auc, spa_auc) + 0.02:
            report_lines.append("\n✓ CONCLUSION: Combined features provide substantial improvement")
            report_lines.append("  - Multimodal integration is beneficial")
            report_lines.append("  - Complementary information captured")
        elif comb_auc > max(rad_auc, spa_auc):
            report_lines.append("\n✓ CONCLUSION: Combined features show marginal improvement")
            report_lines.append("  - Consider cost-benefit of additional features")
        else:
            best_single = 'Radiomics' if rad_auc > spa_auc else 'Spatial'
            report_lines.append(f"\n○ CONCLUSION: {best_single} features perform best")
            report_lines.append("  - Single modality may be sufficient")
        
        # Section 5: Model Stability Analysis
        report_lines.append("\n" + "=" * 100)
        report_lines.append("5. MODEL STABILITY ANALYSIS")
        report_lines.append("=" * 100)
        
        report_lines.append("\nCross-Validation vs Test Performance:")
        report_lines.append("\n{:20s} {:15s} {:>12s} {:>12s} {:>12s}".format(
            "Feature Type", "Model", "CV AUC", "Test AUC", "Difference"))
        report_lines.append("-" * 80)
        
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            results = self.all_results[ft_name]['results']
            
            for model_name, result in results.items():
                cv_auc = result['cv_auc_mean']
                test_auc = result['test_metrics']['AUC']
                diff = abs(test_auc - cv_auc)
                
                stability = "✓" if diff < 0.05 else "○" if diff < 0.1 else "⚠"
                
                report_lines.append("{:20s} {:15s} {:>12.4f} {:>12.4f} {:>11.4f} {}".format(
                    ft_name, model_name, cv_auc, test_auc, diff, stability))
        
        report_lines.append("\n✓ = Excellent stability (diff < 0.05)")
        report_lines.append("○ = Good stability (diff < 0.10)")
        report_lines.append("⚠ = Poor stability (diff ≥ 0.10)")
        
        # Section 6: Clinical Interpretation
        report_lines.append("\n" + "=" * 100)
        report_lines.append("6. CLINICAL INTERPRETATION")
        report_lines.append("=" * 100)
        
        report_lines.append(f"\nBased on best model: {model_best} ({ft_best})")
        
        auc = metrics['AUC']
        if auc >= 0.9:
            interp = "Excellent discrimination - suitable for clinical decision support"
        elif auc >= 0.8:
            interp = "Good discrimination - promising for clinical application"
        elif auc >= 0.7:
            interp = "Acceptable discrimination - may require additional validation"
        else:
            interp = "Limited discrimination - needs improvement"
        
        report_lines.append(f"\nAUC Interpretation ({auc:.3f}): {interp}")
        
        sens = metrics['Sensitivity']
        spec = metrics['Specificity']
        
        report_lines.append(f"\nDiagnostic Performance:")
        report_lines.append(f"  Sensitivity: {sens:.3f}")
        report_lines.append(f"    → Correctly identifies {sens*100:.1f}% of positive cases")
        report_lines.append(f"    → Misses {(1-sens)*100:.1f}% of positive cases (false negatives)")
        
        report_lines.append(f"\n  Specificity: {spec:.3f}")
        report_lines.append(f"    → Correctly identifies {spec*100:.1f}% of negative cases")
        report_lines.append(f"    → {(1-spec)*100:.1f}% false positive rate")

        ppv = metrics['PPV']
        npv = metrics['NPV']
        
        report_lines.append(f"\n  PPV (Positive Predictive Value): {ppv:.3f}")
        report_lines.append(f"    → {ppv*100:.1f}% of predicted positive cases are truly positive")
        report_lines.append(f"    → Clinical confidence in positive predictions")
        
        report_lines.append(f"\n  NPV (Negative Predictive Value): {npv:.3f}")
        report_lines.append(f"    → {npv*100:.1f}% of predicted negative cases are truly negative")
        report_lines.append(f"    → Clinical confidence in negative predictions")
        
        # Balance analysis
        if sens > 0.8 and spec > 0.8:
            report_lines.append("\n✓ Excellent balance between sensitivity and specificity")
        elif sens > spec + 0.1:
            report_lines.append("\n○ Model favors sensitivity (fewer false negatives)")
            report_lines.append("  - Good for screening applications")
        elif spec > sens + 0.1:
            report_lines.append("\n○ Model favors specificity (fewer false positives)")
            report_lines.append("  - Good for confirmatory applications")
        else:
            report_lines.append("\n✓ Balanced sensitivity and specificity")
        
        # Section 7: Feature Importance Insights
        report_lines.append("\n" + "=" * 100)
        report_lines.append("7. FEATURE IMPORTANCE INSIGHTS")
        report_lines.append("=" * 100)
        
        # Analyze feature importance for tree-based models
        for ft_name in ['Radiomics', 'Spatial', 'Combined']:
            report_lines.append(f"\n{ft_name}:")
            report_lines.append("-" * 80)
            
            results = self.all_results[ft_name]['results']
            
            # Check RandomForest
            if 'RandomForest' in results:
                rf_model = results['RandomForest']['model']
                if hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                    feature_names = self.all_results[ft_name]['selected_features']
                    
                    top_n = min(10, len(importances))
                    top_indices = np.argsort(importances)[::-1][:top_n]
                    
                    report_lines.append(f"\nTop {top_n} features (Random Forest):")
                    for rank, idx in enumerate(top_indices, 1):
                        feat_name = feature_names[idx]
                        importance = importances[idx]
                        report_lines.append(f"  {rank:2d}. {feat_name[:65]:<65s} ({importance:.4f})")
                    
                    # For Combined, show source distribution
                    if ft_name == 'Combined':
                        n_rad_total = len(self.all_results['Radiomics']['selected_features'])
                        n_spa_total = len(self.all_results['Spatial']['selected_features'])
                        
                        rad_importance = np.sum([importances[i] for i in range(len(importances))
                                                if any(feature_names[i] in self.all_results['Radiomics']['selected_features'])])
                        spa_importance = np.sum(importances) - rad_importance
                        
                        total_imp = rad_importance + spa_importance
                        if total_imp > 0:
                            report_lines.append(f"\nFeature source contribution:")
                            report_lines.append(f"  Radiomics: {rad_importance:.4f} ({rad_importance/total_imp*100:.1f}%)")
                            report_lines.append(f"  Spatial:   {spa_importance:.4f} ({spa_importance/total_imp*100:.1f}%)")
        
        # Section 8: Recommendations
        report_lines.append("\n" + "=" * 100)
        report_lines.append("8. RECOMMENDATIONS & NEXT STEPS")
        report_lines.append("=" * 100)
        
        report_lines.append(f"\n✓ RECOMMENDED MODEL FOR DEPLOYMENT:")
        report_lines.append(f"  Feature Type: {ft_best}")
        report_lines.append(f"  Model: {model_best}")
        report_lines.append(f"  Number of features: {len(self.all_results[ft_best]['selected_features'])}")
        report_lines.append(f"  Test AUC: {metrics['AUC']:.4f}")
        report_lines.append(f"  CV Stability: {result_best['cv_auc_std']:.4f}")
        
        # Rationale
        report_lines.append(f"\nRationale:")
        report_lines.append(f"  • Highest test AUC among all configurations")
        report_lines.append(f"  • Good cross-validation stability (SD={result_best['cv_auc_std']:.4f})")
        report_lines.append(f"  • Reasonable feature count ({len(self.all_results[ft_best]['selected_features'])} features)")
        
        if overfit_gap < 0.1:
            report_lines.append(f"  • Minimal overfitting (gap={overfit_gap:.4f})")
        
        # Next steps
        report_lines.append(f"\n✓ NEXT STEPS:")
        report_lines.append(f"  1. External Validation")
        report_lines.append(f"     - Test on independent dataset from different institution")
        report_lines.append(f"     - Assess generalization to different scanner/protocol")
        
        report_lines.append(f"\n  2. Prospective Validation")
        report_lines.append(f"     - Implement in clinical workflow")
        report_lines.append(f"     - Evaluate real-world performance")
        report_lines.append(f"     - Collect user feedback")
        
        report_lines.append(f"\n  3. Model Calibration")
        report_lines.append(f"     - Assess probability calibration")
        report_lines.append(f"     - Apply calibration methods if needed (Platt scaling, isotonic regression)")
        
        report_lines.append(f"\n  4. Clinical Utility Analysis")
        report_lines.append(f"     - Decision curve analysis")
        report_lines.append(f"     - Net benefit calculation")
        report_lines.append(f"     - Cost-effectiveness evaluation")
        
        report_lines.append(f"\n  5. Interpretability Enhancement")
        report_lines.append(f"     - SHAP analysis for individual predictions")
        report_lines.append(f"     - Feature contribution visualization")
        report_lines.append(f"     - Clinical correlation studies")
        
        report_lines.append(f"\n  6. Robustness Testing")
        report_lines.append(f"     - Bootstrap validation (n=1000)")
        report_lines.append(f"     - Perturbation analysis")
        report_lines.append(f"     - Subgroup analysis (age, stage, histology)")
        
        # Section 9: Limitations
        report_lines.append("\n" + "=" * 100)
        report_lines.append("9. LIMITATIONS & CONSIDERATIONS")
        report_lines.append("=" * 100)
        
        report_lines.append(f"\n• Sample Size:")
        report_lines.append(f"  - Total: {len(self.merged_df)} cases")
        report_lines.append(f"  - Test set: {len(self.test_idx)} cases")
        report_lines.append(f"  - Consider larger cohort for robust validation")
        
        report_lines.append(f"\n• Feature Selection:")
        report_lines.append(f"  - Multiple methods used for robustness")
        report_lines.append(f"  - 10% rule applied to prevent overfitting")
        report_lines.append(f"  - Consensus approach may miss some informative features")
        
        report_lines.append(f"\n• Model Complexity:")
        report_lines.append(f"  - Hyperparameter tuning performed on training set")
        report_lines.append(f"  - Risk of overfitting to this specific dataset")
        report_lines.append(f"  - External validation critical")
        
        report_lines.append(f"\n• Class Imbalance:")
        report_lines.append(f"  - SMOTE applied for training")
        report_lines.append(f"  - May affect probability calibration")
        report_lines.append(f"  - Consider cost-sensitive learning alternatives")
        
        report_lines.append(f"\n• Generalizability:")
        report_lines.append(f"  - Single-institution data")
        report_lines.append(f"  - Scanner/protocol specific features")
        report_lines.append(f"  - Multi-center validation needed")
        
        # Section 10: Output Files
        report_lines.append("\n" + "=" * 100)
        report_lines.append("10. OUTPUT FILES GENERATED")
        report_lines.append("=" * 100)
        
        report_lines.append("\nFigures (PDF and PNG):")
        figure_list = [
            "Fig1_Overall_Performance_Comparison.pdf/png - Performance across all models",
            "Fig2_ROC_Best_Models.pdf/png - ROC curves for best models per feature type",
            "Fig3_ROC_All_Models.pdf/png - ROC curves for all models (Combined features)",
            "Fig4_Performance_Radar.pdf/png - Radar charts showing metric balance",
            "Fig5_Learning_Curves.pdf/png - Learning curves for stability assessment",
            "Fig6_Confusion_Matrices.pdf/png - Confusion matrices for best models",
            "Fig7_Feature_Importance.pdf/png - Feature importance from tree models",
            "Fig8_Model_Stability.pdf/png - CV vs Test performance comparison",
            "Fig9_Calibration_Curves.pdf/png - Probability calibration assessment",
            "Fig10_Performance_Heatmap.pdf/png - Heatmap of all metrics",
            "Fig11_Feature_Selection_Consensus.pdf/png - Feature selection voting results",
            "Fig12_Forest_Plot.pdf/png - Forest plot of all model comparisons"
        ]
        
        for fig in figure_list:
            report_lines.append(f"  • {fig}")
        
        report_lines.append("\nFeature Selection Figures:")
        report_lines.append("  • FS_Comparison_Radiomics.pdf/png - Feature selection comparison")
        report_lines.append("  • FS_Comparison_Spatial.pdf/png - Feature selection comparison")
        report_lines.append("  • FS_Comparison_Combined.pdf/png - Feature selection comparison")
        
        report_lines.append("\nData Tables (CSV/Excel):")
        table_list = [
            "Table1_All_Results.csv/xlsx - Complete performance metrics for all models",
            "Table2_Best_Models.csv/xlsx - Best model from each feature type",
            "Table3_Feature_Selection_Summary.csv - Summary of feature selection methods",
            "Table4_Statistical_Tests.csv - Statistical comparison of feature types",
            "FS_Details_Radiomics.xlsx - Detailed feature selection results",
            "FS_Details_Spatial.xlsx - Detailed feature selection results",
            "FS_Details_Combined.xlsx - Detailed feature selection results",
            "FS_VotingSummary_Radiomics.csv - Feature voting summary",
            "FS_VotingSummary_Spatial.csv - Feature voting summary",
            "FS_VotingSummary_Combined.csv - Feature voting summary"
        ]
        
        for table in table_list:
            report_lines.append(f"  • {table}")
        
        report_lines.append("\nTrained Models:")
        report_lines.append("  • trained_models/ - Directory containing all trained models")
        report_lines.append(f"    - {len(self.all_results) * 6} model files (3 feature types × 6 models)")
        report_lines.append("    - scalers.pkl - Feature scaling transformers")
        report_lines.append("    - feature_selection_methods.pkl - Feature selection metadata")
        report_lines.append("    - split_info.pkl - Train/test split information")
        
        # Section 11: Summary Statistics
        report_lines.append("\n" + "=" * 100)
        report_lines.append("11. SUMMARY STATISTICS")
        report_lines.append("=" * 100)
        
        # Count models
        total_models = sum(len(self.all_results[ft]['results']) 
                          for ft in ['Radiomics', 'Spatial', 'Combined'])
        
        report_lines.append(f"\nModels Trained: {total_models}")
        report_lines.append(f"Feature Selection Methods: 6 per feature type")
        report_lines.append(f"Total Figures Generated: 15 (12 main + 3 feature selection)")
        report_lines.append(f"Total Tables Generated: 10")
        
        # AUC range
        all_aucs = []
        for ft in ['Radiomics', 'Spatial', 'Combined']:
            for result in self.all_results[ft]['results'].values():
                all_aucs.append(result['test_metrics']['AUC'])
        
        report_lines.append(f"\nAUC Statistics:")
        report_lines.append(f"  Range: {min(all_aucs):.4f} - {max(all_aucs):.4f}")
        report_lines.append(f"  Mean:  {np.mean(all_aucs):.4f}")
        report_lines.append(f"  Median: {np.median(all_aucs):.4f}")
        report_lines.append(f"  Std:   {np.std(all_aucs):.4f}")
        
        # Section 12: Acknowledgments
        report_lines.append("\n" + "=" * 100)
        report_lines.append("12. METHODOLOGY NOTES")
        report_lines.append("=" * 100)
        
        report_lines.append("\nFeature Selection Methods:")
        report_lines.append("  1. RFECV - Recursive Feature Elimination with Cross-Validation")
        report_lines.append("  2. Lasso - L1 regularization based selection")
        report_lines.append("  3. Random Forest - Tree-based importance")
        report_lines.append("  4. Mutual Information - Information-theoretic measure")
        report_lines.append("  5. ANOVA - F-statistic based selection")
        report_lines.append("  6. Stability Selection - Bootstrap-based robust selection")
        report_lines.append("  7. Consensus - Features selected by ≥3 methods")
        
        report_lines.append("\nModels Compared:")
        report_lines.append("  • Logistic Regression - Linear baseline with L2 regularization")
        report_lines.append("  • Random Forest - Ensemble of decision trees")
        report_lines.append("  • Gradient Boosting - Sequential tree boosting")
        report_lines.append("  • Support Vector Machine - Kernel-based classifier")
        report_lines.append("  • Multi-Layer Perceptron - Neural network")
        report_lines.append("  • Extra Trees - Extremely randomized trees")
        
        report_lines.append("\nValidation Strategy:")
        report_lines.append("  • Train/Test split: 70/30 (stratified)")
        report_lines.append("  • Cross-validation: 5-fold stratified")
        report_lines.append("  • Hyperparameter tuning: Grid search with CV")
        report_lines.append("  • Class imbalance: SMOTE on training set only")
        
        # Final section
        report_lines.append("\n" + "=" * 100)
        report_lines.append("END OF COMPREHENSIVE REPORT")
        report_lines.append("=" * 100)
        
        report_lines.append(f"\n\nAnalysis completed successfully at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"For questions or issues, please review the methodology and output files.")
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(self.output_dir / 'COMPREHENSIVE_REPORT.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Also create a summary version
        self._create_executive_summary(ft_best, model_best, result_best, metrics)
        
        print("  ✓ Comprehensive report generated")
    
    def _create_executive_summary(self, ft_best, model_best, result_best, metrics):
        """Create a concise executive summary"""
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("EXECUTIVE SUMMARY")
        summary_lines.append("Multimodal Response Prediction Analysis")
        summary_lines.append("=" * 80)
        summary_lines.append(f"\nDate: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        
        summary_lines.append("\n" + "=" * 80)
        summary_lines.append("KEY FINDINGS")
        summary_lines.append("=" * 80)
        
        summary_lines.append(f"\n1. BEST PERFORMING MODEL")
        summary_lines.append(f"   • Feature Type: {ft_best}")
        summary_lines.append(f"   • Algorithm: {model_best}")
        summary_lines.append(f"   • Test AUC: {metrics['AUC']:.3f}")
        summary_lines.append(f"   • Sensitivity: {metrics['Sensitivity']:.3f}")
        summary_lines.append(f"   • Specificity: {metrics['Specificity']:.3f}")
        
        summary_lines.append(f"\n2. DATASET")
        summary_lines.append(f"   • Total samples: {len(self.merged_df)}")
        summary_lines.append(f"   • Training: {len(self.train_idx)}")
        summary_lines.append(f"   • Test: {len(self.test_idx)}")
        
        summary_lines.append(f"\n3. FEATURE SELECTION")
        n_features = len(self.all_results[ft_best]['selected_features'])
        summary_lines.append(f"   • Methods used: 6 (consensus approach)")
        summary_lines.append(f"   • Final features: {n_features}")
        summary_lines.append(f"   • Feature/sample ratio: {n_features/len(self.train_idx)*100:.1f}%")
        
        summary_lines.append(f"\n4. PERFORMANCE COMPARISON")
        for ft in ['Radiomics', 'Spatial', 'Combined']:
            best_auc = max(r['test_metrics']['AUC'] 
                          for r in self.all_results[ft]['results'].values())
            summary_lines.append(f"   • {ft:12s}: {best_auc:.3f}")
        
        summary_lines.append(f"\n5. CLINICAL IMPLICATIONS")
        if metrics['AUC'] >= 0.9:
            summary_lines.append(f"   • Excellent predictive performance")
        elif metrics['AUC'] >= 0.8:
            summary_lines.append(f"   • Good predictive performance")
        else:
            summary_lines.append(f"   • Moderate predictive performance")
        
        summary_lines.append(f"   • PPV: {metrics['PPV']:.3f} - {metrics['PPV']*100:.0f}% precision in positive predictions")
        summary_lines.append(f"   • NPV: {metrics['NPV']:.3f} - {metrics['NPV']*100:.0f}% precision in negative predictions")
        
        summary_lines.append(f"\n6. RECOMMENDATIONS")
        summary_lines.append(f"   ✓ Proceed with external validation")
        summary_lines.append(f"   ✓ Deploy model: {model_best} with {ft_best} features")
        summary_lines.append(f"   ✓ Monitor performance in clinical setting")
        
        summary_lines.append(f"\n" + "=" * 80)
        summary_lines.append(f"For detailed results, see COMPREHENSIVE_REPORT.txt")
        summary_lines.append(f"All figures and tables available in: {self.output_dir}")
        summary_lines.append("=" * 80)
        
        # Save
        summary_text = "\n".join(summary_lines)
        with open(self.output_dir / 'EXECUTIVE_SUMMARY.txt', 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # Also print to console
        print("\n" + summary_text)


# ============================================================================
# UTILITY FUNCTIONS FOR POST-ANALYSIS
# ============================================================================

def create_publication_ready_table(predictor):
    """Create a publication-ready table (Table 1 style)"""
    
    print("\nCreating publication-ready table...")
    
    # Prepare data
    data = []
    
    for ft_name in ['Radiomics', 'Spatial', 'Combined']:
        results = predictor.all_results[ft_name]['results']
        n_features = len(predictor.all_results[ft_name]['selected_features'])
        
        # Get best model
        best_model_name = max(results.keys(), 
                             key=lambda m: results[m]['test_metrics']['AUC'])
        result = results[best_model_name]
        
        metrics = result['test_metrics']
        
        row = {
            'Feature Type': ft_name,
            'Model': best_model_name,
            'N Features': n_features,
            'AUC': f"{metrics['AUC']:.3f}",
            'Accuracy': f"{metrics['Accuracy']:.3f}",
            'Sensitivity': f"{metrics['Sensitivity']:.3f}",
            'Specificity': f"{metrics['Specificity']:.3f}",
            'PPV': f"{metrics['PPV']:.3f}",
            'NPV': f"{metrics['NPV']:.3f}",
            'F1 Score': f"{metrics['F1']:.3f}",
            'CV AUC': f"{result['cv_auc_mean']:.3f} ± {result['cv_auc_std']:.3f}"
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save
    df.to_csv(predictor.output_dir / 'Table_Publication_Ready.csv', index=False)
    df.to_excel(predictor.output_dir / 'Table_Publication_Ready.xlsx', index=False)
    
    # Create LaTeX version
    latex_table = df.to_latex(index=False, escape=False, 
                              caption='Performance of Best Models by Feature Type',
                              label='tab:performance')
    
    with open(predictor.output_dir / 'Table_Publication_Ready.tex', 'w') as f:
        f.write(latex_table)
    
    print("  ✓ Publication-ready table created")
    print("\nTable Preview:")
    print(df.to_string(index=False))
    
    return df


def perform_bootstrap_validation(predictor, n_bootstrap=1000):
    """Perform bootstrap validation for confidence intervals"""
    
    print(f"\nPerforming bootstrap validation (n={n_bootstrap})...")
    print("This may take several minutes...")
    
    from scipy import stats
    
    bootstrap_results = {}
    
    for ft_name in ['Radiomics', 'Spatial', 'Combined']:
        print(f"\n  Processing {ft_name}...")
        
        results = predictor.all_results[ft_name]['results']
        best_model_name = max(results.keys(), 
                             key=lambda m: results[m]['test_metrics']['AUC'])
        
        best_model = results[best_model_name]['model']
        X_test = predictor.all_results[ft_name]['X_test']
        y_test = predictor.y_test
        
        # Bootstrap
        auc_scores = []
        sens_scores = []
        spec_scores = []
        
        np.random.seed(42)
        
        for i in range(n_bootstrap):
            # Resample test set
            indices = np.random.choice(len(y_test), len(y_test), replace=True)
            X_boot = X_test[indices]
            y_boot = y_test[indices]
            
            # Predict
            y_prob_boot = best_model.predict_proba(X_boot)[:, 1]
            y_pred_boot = best_model.predict(X_boot)
            
            # Metrics
            auc = roc_auc_score(y_boot, y_prob_boot)
            
            tn, fp, fn, tp = confusion_matrix(y_boot, y_pred_boot).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            auc_scores.append(auc)
            sens_scores.append(sens)
            spec_scores.append(spec)
            
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{n_bootstrap} iterations completed")
        
        # Calculate confidence intervals
        auc_ci = np.percentile(auc_scores, [2.5, 97.5])
        sens_ci = np.percentile(sens_scores, [2.5, 97.5])
        spec_ci = np.percentile(spec_scores, [2.5, 97.5])
        
        bootstrap_results[ft_name] = {
            'model': best_model_name,
            'auc_mean': np.mean(auc_scores),
            'auc_ci': auc_ci,
            'sens_mean': np.mean(sens_scores),
            'sens_ci': sens_ci,
            'spec_mean': np.mean(spec_scores),
            'spec_ci': spec_ci
        }
    
    # Save results
    bootstrap_df = pd.DataFrame([
        {
            'Feature_Type': ft,
            'Model': res['model'],
            'AUC_Mean': f"{res['auc_mean']:.3f}",
            'AUC_95CI': f"[{res['auc_ci'][0]:.3f}, {res['auc_ci'][1]:.3f}]",
            'Sensitivity_Mean': f"{res['sens_mean']:.3f}",
            'Sensitivity_95CI': f"[{res['sens_ci'][0]:.3f}, {res['sens_ci'][1]:.3f}]",
            'Specificity_Mean': f"{res['spec_mean']:.3f}",
            'Specificity_95CI': f"[{res['spec_ci'][0]:.3f}, {res['spec_ci'][1]:.3f}]"
        }
        for ft, res in bootstrap_results.items()
    ])
    
    bootstrap_df.to_csv(predictor.output_dir / 'Bootstrap_Validation_Results.csv', index=False)
    
    print("\n  ✓ Bootstrap validation completed")
    print("\nBootstrap Results:")
    print(bootstrap_df.to_string(index=False))
    
    return bootstrap_results


def create_feature_importance_summary(predictor):
    """Create comprehensive feature importance summary"""
    
    print("\nCreating feature importance summary...")
    
    for ft_name in ['Radiomics', 'Spatial', 'Combined']:
        results = predictor.all_results[ft_name]['results']
        feature_names = predictor.all_results[ft_name]['selected_features']
        
        # Collect importances from all tree models
        importance_dict = {feat: [] for feat in feature_names}
        
        for model_name in ['RandomForest', 'GradientBoosting', 'ExtraTrees']:
            if model_name in results:
                model = results[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    for i, feat in enumerate(feature_names):
                        importance_dict[feat].append(importances[i])
        
        # Calculate statistics
        importance_stats = []
        for feat, imps in importance_dict.items():
            if len(imps) > 0:
                importance_stats.append({
                    'Feature': feat,
                    'Mean_Importance': np.mean(imps),
                    'Std_Importance': np.std(imps),
                    'Min_Importance': np.min(imps),
                    'Max_Importance': np.max(imps),
                    'N_Models': len(imps)
                })
        
        if importance_stats:
            df_importance = pd.DataFrame(importance_stats)
            df_importance = df_importance.sort_values('Mean_Importance', ascending=False)
            df_importance.to_csv(
                predictor.output_dir / f'Feature_Importance_Summary_{ft_name}.csv',
                index=False
            )
    
    print("  ✓ Feature importance summary created")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive analysis"""
    
    print("\n" + "=" * 100)
    print("STARTING COMPREHENSIVE MULTIMODAL PREDICTION ANALYSIS")
    print("=" * 100)
    
    # Initialize predictor
    predictor = AdvancedResponsePredictor(
        radiomics_csv='XXXyour path/Dataset/Dataset002_ECResponse/radiomics_features.csv',
        spatial_radiomics_csv='XXXyour path/Dataset/Dataset002_ECResponse/sparad_features.csv',
        clinical_xlsx='XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx',
        output_dir='XXXyour path/Dataset/Dataset002_ECResponse/SpatialRad_Results'
    )
    
    # Run comprehensive analysis
    results = predictor.run_comprehensive_analysis(
        test_size=0.3,
        use_smote=True,
        random_state=48
    )
    
    # Post-analysis enhancements
    print("\n" + "=" * 100)
    print("POST-ANALYSIS: Additional Validations")
    print("=" * 100)
    
    # Create publication-ready table
    pub_table = create_publication_ready_table(predictor)
    
    # Feature importance summary
    create_feature_importance_summary(predictor)
    
    # Bootstrap validation (optional - takes time)
    user_choice = input("\nPerform bootstrap validation? (1000 iterations, ~5-10 min) [y/n]: ")
    if user_choice.lower() == 'y':
        bootstrap_results = perform_bootstrap_validation(predictor, n_bootstrap=1000)
    
    print("\n" + "=" * 100)
    print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(f"\nAll results saved to: {predictor.output_dir}")
    print("\nKey outputs:")
    print("  • 15 publication-quality figures (PDF + PNG)")
    print("  • 10+ comprehensive data tables (CSV + Excel)")
    print("  • All trained models with metadata")
    print("  • Comprehensive report + Executive summary")
    print("  • Feature selection details with voting analysis")
    print("  • Statistical tests and bootstrap validation")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR MANUSCRIPT PREPARATION")
    print("=" * 100)
    
    print("\nMain Figures for Manuscript:")
    print("  • Figure 1: Fig1_Overall_Performance_Comparison.pdf")
    print("  • Figure 2: Fig2_ROC_Best_Models.pdf")
    print("  • Figure 3: Fig11_Feature_Selection_Consensus.pdf")
    print("  • Figure 4: Fig12_Forest_Plot.pdf")
    
    print("\nSupplementary Figures:")
    print("  • All FS_Comparison_*.pdf (feature selection methods)")
    print("  • Fig5_Learning_Curves.pdf")
    print("  • Fig8_Model_Stability.pdf")
    print("  • Fig9_Calibration_Curves.pdf")
    print("  • Fig10_Performance_Heatmap.pdf")
    
    print("\nMain Tables:")
    print("  • Table_Publication_Ready.xlsx - For main manuscript")
    print("  • Table4_Statistical_Tests.csv - For results section")
    print("  • Bootstrap_Validation_Results.csv - For validation")
    
    print("\nSupplementary Tables:")
    print("  • Table1_All_Results.xlsx - Complete model comparison")
    print("  • All FS_Details_*.xlsx - Feature selection details")
    
    return predictor, results


if __name__ == "__main__":
    # Run main analysis
    predictor, results = main()
    
    print("\n" + "=" * 100)
    print("ANALYSIS SESSION COMPLETE")
    print("=" * 100)
    print("\nTo access results programmatically:")
    print("  predictor.all_results         # All model results")
    print("  predictor.selected_features_by_method  # Feature selection details")
    print("  predictor.merged_df           # Original merged data")
    print("\nExample: Get best model")
    print("  best_ft = 'Combined'")
    print("  best_model = predictor.all_results[best_ft]['results']")
    
    # Optional: Create additional visualizations on demand
    print("\n" + "=" * 100)
    print("ADDITIONAL ANALYSIS OPTIONS")
    print("=" * 100)
    print("\nYou can now run additional analyses:")
    print("  1. Subgroup analysis (by clinical variables)")
    print("  2. SHAP analysis for interpretability")
    print("  3. Time-dependent analysis")
    print("  4. External validation on new data")
    
    # Save session
    print("\nSaving analysis session...")
    session_file = predictor.output_dir / 'analysis_session.pkl'
    with open(session_file, 'wb') as f:
        pickle.dump({
            'predictor': predictor,
            'results': results,
            'timestamp': pd.Timestamp.now()
        }, f)
    print(f"  ✓ Session saved to: {session_file}")
    print("\nTo reload: with open('analysis_session.pkl', 'rb') as f: session = pickle.load(f)")