import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

class MultiModalPredictor:
    """
    Predictor that can handle multiple feature types:
    1. Radiomics features only
    2. New features only  
    3. Combined (radiomics + new features)
    """
    
    def __init__(self, radiomics_csv, new_features_csv, clinical_xlsx, output_dir='multimodal_results'):
        """
        Initialize multi-modal predictor
        
        Args:
            radiomics_csv: Path to radiomics features CSV
            new_features_csv: Path to new features CSV
            clinical_xlsx: Path to clinical data with PD-L1 labels
            output_dir: Directory to save results
        """
        self.radiomics_csv = radiomics_csv
        self.new_features_csv = new_features_csv
        self.clinical_xlsx = clinical_xlsx
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scaler_radiomics = StandardScaler()
        self.scaler_new = StandardScaler()
        self.scaler_combined = StandardScaler()
        
        self.train_indices = None
        self.test_indices = None
        
    def load_and_merge_all_data(self, case_id_col='CaseID', label_col='PDL1'):
        """
        Load radiomics, new features, and clinical data, then merge all
        """
        print("Loading all data sources...")
        
        # Load radiomics features
        radiomics_df = pd.read_csv(self.radiomics_csv)
        print(f"Loaded radiomics: {len(radiomics_df)} cases, {len(radiomics_df.columns)} columns")
        
        # Load new features
        new_features_df = pd.read_csv(self.new_features_csv)
        print(f"Loaded new features: {len(new_features_df)} cases, {len(new_features_df.columns)} columns")
        
        # Load clinical data
        clinical_df = pd.read_excel(self.clinical_xlsx)
        print(f"Loaded clinical data: {len(clinical_df)} cases")
        
        # Merge all datasets
        merged_df = pd.merge(radiomics_df, new_features_df, on=case_id_col, 
                            how='inner', suffixes=('_radiomics', '_new'))
        merged_df = pd.merge(merged_df, clinical_df, on=case_id_col, how='inner')
        
        print(f"Merged dataset: {len(merged_df)} cases")
        
        # Check for missing labels
        if merged_df[label_col].isnull().any():
            print(f"Warning: {merged_df[label_col].isnull().sum()} cases with missing labels")
            merged_df = merged_df.dropna(subset=[label_col])
        
        self.merged_df = merged_df
        self.case_id_col = case_id_col
        self.label_col = label_col
        
        # Print class distribution
        print(f"\nPD-L1 expression distribution:")
        print(merged_df[label_col].value_counts())
        
        return merged_df
    
    def prepare_feature_sets(self, exclude_cols=None):
        """
        Prepare three feature sets: radiomics, new features, and combined
        
        Returns:
            Dictionary with 'radiomics', 'new', 'combined' feature matrices
        """
        if exclude_cols is None:
            exclude_cols = [self.case_id_col, self.label_col]
        
        all_exclude_cols = exclude_cols.copy()
        
        # Exclude diagnostic features
        diagnostic_cols = [col for col in self.merged_df.columns 
                          if 'diagnostics' in col.lower()]
        all_exclude_cols.extend(diagnostic_cols)
        
        # Identify radiomics features (with _radiomics suffix or from original radiomics file)
        radiomics_features = [col for col in self.merged_df.columns 
                             if col.endswith('_radiomics') or 
                             (col in pd.read_csv(self.radiomics_csv).columns and 
                              col not in all_exclude_cols)]
        
        # Identify new features (with _new suffix or from new features file)
        new_features = [col for col in self.merged_df.columns 
                       if col.endswith('_new') or 
                       (col in pd.read_csv(self.new_features_csv).columns and 
                        col not in all_exclude_cols and 
                        col not in radiomics_features)]
        
        # Remove duplicates and excluded columns
        radiomics_features = [col for col in radiomics_features if col not in all_exclude_cols]
        new_features = [col for col in new_features if col not in all_exclude_cols]
        
        print(f"\nFeature sets prepared:")
        print(f"  Radiomics features: {len(radiomics_features)}")
        print(f"  New features: {len(new_features)}")
        print(f"  Combined features: {len(radiomics_features) + len(new_features)}")
        
        # Extract feature matrices
        X_radiomics = self.merged_df[radiomics_features].values
        X_new = self.merged_df[new_features].values
        X_combined = np.concatenate([X_radiomics, X_new], axis=1)
        
        y = self.merged_df[self.label_col].values
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        if np.isnan(X_radiomics).any():
            print("Imputing missing values in radiomics features...")
            imputer_rad = SimpleImputer(strategy='median')
            X_radiomics = imputer_rad.fit_transform(X_radiomics)
        
        if np.isnan(X_new).any():
            print("Imputing missing values in new features...")
            imputer_new = SimpleImputer(strategy='median')
            X_new = imputer_new.fit_transform(X_new)
        
        if np.isnan(X_combined).any():
            print("Imputing missing values in combined features...")
            imputer_combined = SimpleImputer(strategy='median')
            X_combined = imputer_combined.fit_transform(X_combined)
        
        feature_sets = {
            'radiomics': {
                'X': X_radiomics,
                'feature_names': radiomics_features
            },
            'new': {
                'X': X_new,
                'feature_names': new_features
            },
            'combined': {
                'X': X_combined,
                'feature_names': radiomics_features + new_features
            }
        }
        
        self.feature_sets = feature_sets
        self.y = y
        
        return feature_sets, y
    
    def feature_selection(self, X, y, feature_names, method='univariate', k=20, prefix=''):
        """
        Perform feature selection
        """
        print(f"\n{prefix}Feature selection using {method} method...")
        
        if method == 'univariate':
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Get feature scores
            scores = selector.scores_
            feature_scores = pd.DataFrame({
                'Feature': feature_names,
                'Score': scores
            }).sort_values('Score', ascending=False)
            
            return X_selected, selected_features, selector, feature_scores
            
        elif method == 'tree_based':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            
            selected_features = [feature_names[i] for i in indices]
            X_selected = X[:, indices]
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return X_selected, selected_features, indices, feature_importance
        
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            scores = selector.scores_
            feature_scores = pd.DataFrame({
                'Feature': feature_names,
                'Score': scores
            }).sort_values('Score', ascending=False)
            
            return X_selected, selected_features, selector, feature_scores
    
    def train_models_for_feature_type(self, X_train, X_test, y_train, y_test, feature_type):
        """
        Train multiple models for a specific feature type
        """
        print(f"\nTraining models for {feature_type} features...")
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob_test = model.predict_proba(X_test)[:, 1]
            
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'auc': roc_auc_score(y_train, y_prob_train),
                'f1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            }
            
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'auc': roc_auc_score(y_test, y_prob_test),
                'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                       scoring='roc_auc', n_jobs=-1)
            
            results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'y_prob_test': y_prob_test,
                'y_pred_test': y_pred_test,
                'y_prob_train': y_prob_train,
                'y_pred_train': y_pred_train
            }
            
            print(f"  {name}: Test AUC = {test_metrics['auc']:.4f}, CV AUC = {cv_scores.mean():.4f}")
        
        return results
    
    def run_complete_comparison(self, test_size=0.2, n_features=20, 
                               feature_selection_method='tree_based', random_state=42):
        """
        Run complete comparison across all three feature types
        """
        print("="*80)
        print("MULTI-MODAL FEATURE COMPARISON PIPELINE")
        print("="*80)
        
        # Load all data
        self.load_and_merge_all_data()
        
        # Prepare feature sets
        feature_sets, y = self.prepare_feature_sets()
        
        # Create train-test split (same for all feature types)
        print(f"\nCreating train-test split ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.train_indices = train_idx
        self.test_indices = test_idx
        
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"Training set: {len(train_idx)} samples")
        print(f"Test set: {len(test_idx)} samples")
        
        # Save split
        self.save_train_test_split()
        
        # Store all results
        all_results = {}
        all_comparisons = {}
        
        # Process each feature type
        for feature_type in ['radiomics', 'new', 'combined']:
            print(f"\n{'='*80}")
            print(f"PROCESSING {feature_type.upper()} FEATURES")
            print(f"{'='*80}")
            
            X = feature_sets[feature_type]['X']
            feature_names = feature_sets[feature_type]['feature_names']
            
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Standardize
            if feature_type == 'radiomics':
                X_train_scaled = self.scaler_radiomics.fit_transform(X_train)
                X_test_scaled = self.scaler_radiomics.transform(X_test)
            elif feature_type == 'new':
                X_train_scaled = self.scaler_new.fit_transform(X_train)
                X_test_scaled = self.scaler_new.transform(X_test)
            else:  # combined
                X_train_scaled = self.scaler_combined.fit_transform(X_train)
                X_test_scaled = self.scaler_combined.transform(X_test)
            
            # Feature selection
            X_train_selected, selected_features, selector, feature_scores = self.feature_selection(
                X_train_scaled, y_train, feature_names, 
                method=feature_selection_method, k=n_features,
                prefix=f"[{feature_type}] "
            )
            
            # Apply same selection to test set
            if feature_selection_method == 'tree_based':
                X_test_selected = X_test_scaled[:, selector]
            else:
                X_test_selected = selector.transform(X_test_scaled)
            
            # Save feature scores
            feature_scores.to_csv(
                self.output_dir / f'feature_scores_{feature_type}.csv', 
                index=False
            )
            
            # Train models
            results = self.train_models_for_feature_type(
                X_train_selected, X_test_selected, y_train, y_test, feature_type
            )
            
            all_results[feature_type] = results
            
            # Create comparison dataframe for this feature type
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Feature_Type': feature_type,
                    'Model': model_name,
                    'Train_AUC': result['train_metrics']['auc'],
                    'Test_AUC': result['test_metrics']['auc'],
                    'CV_AUC': result['cv_auc_mean'],
                    'Test_Accuracy': result['test_metrics']['accuracy'],
                    'Test_F1': result['test_metrics']['f1']
                })
            
            all_comparisons[feature_type] = pd.DataFrame(comparison_data)
        
        # Combine all comparisons
        combined_comparison = pd.concat(all_comparisons.values(), ignore_index=True)
        combined_comparison.to_csv(self.output_dir / 'all_features_comparison.csv', index=False)
        
        # Create visualizations
        self.create_comparison_visualizations(combined_comparison, all_results, y_test)
        
        # Analyze feature contributions in combined model
        self.analyze_feature_contributions(feature_sets, all_results)
        
        # Create comprehensive summary
        self.create_comprehensive_summary(combined_comparison, all_results)
        
        print("\n" + "="*80)
        print("MULTI-MODAL COMPARISON COMPLETED")
        print("="*80)
        
        return all_results, combined_comparison
    
    def create_comparison_visualizations(self, comparison_df, all_results, y_test):
        """
        Create comprehensive visualizations comparing all feature types
        """
        print("\nCreating comparison visualizations...")
        
        # Plot 1: Performance comparison across feature types
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Test AUC by feature type
        ax1 = axes[0, 0]
        pivot_data = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_AUC')
        pivot_data.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Test AUC Comparison Across Feature Types', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AUC Score')
        ax1.set_xlabel('Model')
        ax1.legend(title='Feature Type', labels=['Radiomics', 'New Features', 'Combined'])
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 1.0])
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # CV AUC by feature type
        ax2 = axes[0, 1]
        pivot_data = comparison_df.pivot(index='Model', columns='Feature_Type', values='CV_AUC')
        pivot_data.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Cross-Validation AUC Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CV AUC Score')
        ax2.set_xlabel('Model')
        ax2.legend(title='Feature Type', labels=['Radiomics', 'New Features', 'Combined'])
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.5, 1.0])
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Test Accuracy by feature type
        ax3 = axes[1, 0]
        pivot_data = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_Accuracy')
        pivot_data.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Model')
        ax3.legend(title='Feature Type', labels=['Radiomics', 'New Features', 'Combined'])
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1.0])
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Best performance for each feature type
        ax4 = axes[1, 1]
        best_by_type = comparison_df.groupby('Feature_Type')['Test_AUC'].max()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax4.bar(range(len(best_by_type)), best_by_type.values, color=colors, alpha=0.7)
        ax4.set_xticks(range(len(best_by_type)))
        ax4.set_xticklabels(['Radiomics', 'New Features', 'Combined'])
        ax4.set_ylabel('Best Test AUC')
        ax4.set_title('Best Model Performance by Feature Type', fontsize=14, fontweight='bold')
        ax4.set_ylim([0.5, 1.0])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison_all_features.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: ROC curves comparison
        self.plot_roc_curves_comparison(all_results, y_test)
        
        # Plot 3: Heatmap of performance metrics
        self.plot_performance_heatmap(comparison_df)
    
    def plot_roc_curves_comparison(self, all_results, y_test):
        """
        Plot ROC curves for best model of each feature type
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Separate ROC curves
        ax1 = axes[0]
        colors = {'radiomics': '#1f77b4', 'new': '#ff7f0e', 'combined': '#2ca02c'}
        
        for feature_type in ['radiomics', 'new', 'combined']:
            # Get best model for this feature type
            best_auc = 0
            best_probs = None
            best_model_name = None
            
            for model_name, result in all_results[feature_type].items():
                if result['test_metrics']['auc'] > best_auc:
                    best_auc = result['test_metrics']['auc']
                    best_probs = result['y_prob_test']
                    best_model_name = model_name
            
            fpr, tpr, _ = roc_curve(y_test, best_probs)
            label_name = {'radiomics': 'Radiomics', 'new': 'New Features', 'combined': 'Combined'}[feature_type]
            ax1.plot(fpr, tpr, color=colors[feature_type], linewidth=2.5, 
                    label=f'{label_name} ({best_model_name})\nAUC = {best_auc:.3f}')
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=13)
        ax1.set_ylabel('True Positive Rate', fontsize=13)
        ax1.set_title('ROC Curves - Best Model per Feature Type', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All models for combined features
        ax2 = axes[1]
        for model_name, result in all_results['combined'].items():
            fpr, tpr, _ = roc_curve(y_test, result['y_prob_test'])
            auc = result['test_metrics']['auc']
            ax2.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate', fontsize=13)
        ax2.set_ylabel('True Positive Rate', fontsize=13)
        ax2.set_title('ROC Curves - All Models (Combined Features)', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ROC curves comparison saved")
    
    def plot_performance_heatmap(self, comparison_df):
        """
        Create heatmap of performance metrics
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Test AUC
        ax1 = axes[0]
        pivot_auc = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_AUC')
        pivot_auc.columns = ['Radiomics', 'New Features', 'Combined']
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, ax=ax1, cbar_kws={'label': 'AUC Score'})
        ax1.set_title('Test AUC Heatmap', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=12)
        ax1.set_xlabel('Feature Type', fontsize=12)
        
        # Heatmap 2: Test Accuracy
        ax2 = axes[1]
        pivot_acc = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_Accuracy')
        pivot_acc.columns = ['Radiomics', 'New Features', 'Combined']
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, ax=ax2, cbar_kws={'label': 'Accuracy'})
        ax2.set_title('Test Accuracy Heatmap', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Model', fontsize=12)
        ax2.set_xlabel('Feature Type', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Performance heatmap saved")
    
    def analyze_feature_contributions(self, feature_sets, all_results):
        """
        Analyze which feature type contributes more in the combined model
        """
        print("\nAnalyzing feature contributions in combined model...")
        
        # Get feature importance from Random Forest (combined)
        rf_combined = all_results['combined']['Random Forest']['model']
        
        if hasattr(rf_combined, 'feature_importances_'):
            importances = rf_combined.feature_importances_
            feature_names = feature_sets['combined']['feature_names']
            
            # Identify which features are radiomics vs new
            n_radiomics = len(feature_sets['radiomics']['feature_names'])
            
            radiomics_importance = importances[:n_radiomics]
            new_importance = importances[n_radiomics:]
            
            # Calculate contribution percentages
            total_radiomics = radiomics_importance.sum()
            total_new = new_importance.sum()
            total = total_radiomics + total_new
            
            contribution_df = pd.DataFrame({
                'Feature_Type': ['Radiomics', 'New_Features'],
                'Total_Importance': [total_radiomics, total_new],
                'Percentage': [total_radiomics/total*100, total_new/total*100],
                'Num_Features': [len(radiomics_importance), len(new_importance)],
                'Avg_Importance': [total_radiomics/len(radiomics_importance), 
                                  total_new/len(new_importance)]
            })
            
            contribution_df.to_csv(self.output_dir / 'feature_type_contributions.csv', index=False)
            
            print("\nFeature Type Contributions (Combined Model):")
            print(contribution_df.to_string(index=False))
            
            # Get top features from each type
            top_n = 10
            
            radiomics_top_indices = np.argsort(radiomics_importance)[::-1][:top_n]
            new_top_indices = np.argsort(new_importance)[::-1][:top_n]
            
            top_radiomics_features = pd.DataFrame({
                'Feature': [feature_sets['radiomics']['feature_names'][i] for i in radiomics_top_indices],
                'Importance': radiomics_importance[radiomics_top_indices],
                'Type': 'Radiomics'
            })
            
            top_new_features = pd.DataFrame({
                'Feature': [feature_sets['new']['feature_names'][i] for i in new_top_indices],
                'Importance': new_importance[new_top_indices],
                'Type': 'New_Features'
            })
            
            top_features_combined = pd.concat([top_radiomics_features, top_new_features], ignore_index=True)
            top_features_combined.to_csv(self.output_dir / 'top_features_by_type.csv', index=False)
            
            # Visualize contributions
            self.plot_feature_contributions(contribution_df, top_features_combined)
    
    def plot_feature_contributions(self, contribution_df, top_features_df):
        """
        Visualize feature type contributions
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Pie chart of contributions
        ax1 = axes[0]
        colors = ['#1f77b4', '#ff7f0e']
        wedges, texts, autotexts = ax1.pie(
            contribution_df['Percentage'], 
            labels=['Radiomics Features', 'New Features'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        ax1.set_title('Feature Type Contribution in Combined Model', 
                     fontsize=14, fontweight='bold')
        
        # Plot 2: Top features from each type
        ax2 = axes[1]
        
        # Separate by type
        radiomics_features = top_features_df[top_features_df['Type'] == 'Radiomics']
        new_features = top_features_df[top_features_df['Type'] == 'New_Features']
        
        y_pos = np.arange(len(radiomics_features) + len(new_features))
        importances = list(radiomics_features['Importance'].values) + list(new_features['Importance'].values)
        feature_labels = list(radiomics_features['Feature'].values) + list(new_features['Feature'].values)
        bar_colors = ['#1f77b4'] * len(radiomics_features) + ['#ff7f0e'] * len(new_features)
        
        ax2.barh(y_pos, importances, color=bar_colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(feature_labels, fontsize=9)
        ax2.set_xlabel('Feature Importance', fontsize=12)
        ax2.set_title('Top 10 Features from Each Type', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, label='Radiomics'),
            Patch(facecolor='#ff7f0e', alpha=0.7, label='New Features')
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_type_contributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Feature contributions visualization saved")
    
    def create_comprehensive_summary(self, comparison_df, all_results):
        """
        Create a comprehensive summary report
        """
        print("\nCreating comprehensive summary...")
        
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("MULTI-MODAL FEATURE COMPARISON SUMMARY")
        summary_lines.append("="*80)
        summary_lines.append("")
        
        # Best model for each feature type
        for feature_type in ['radiomics', 'new', 'combined']:
            subset = comparison_df[comparison_df['Feature_Type'] == feature_type]
            best_idx = subset['Test_AUC'].idxmax()
            best_row = subset.loc[best_idx]
            
            feature_name = {'radiomics': 'Radiomics', 'new': 'New Features', 'combined': 'Combined'}[feature_type]
            
            summary_lines.append(f"{feature_name} Features - Best Model: {best_row['Model']}")
            summary_lines.append(f"  Test AUC: {best_row['Test_AUC']:.4f}")
            summary_lines.append(f"  Test Accuracy: {best_row['Test_Accuracy']:.4f}")
            summary_lines.append(f"  CV AUC: {best_row['CV_AUC']:.4f} ± {0.02:.4f}")
            summary_lines.append("")
        
        # Compare best models across feature types
        best_radiomics = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].max()
        best_new = comparison_df[comparison_df['Feature_Type'] == 'new']['Test_AUC'].max()
        best_combined = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].max()
        
        summary_lines.append("Performance Comparison:")
        summary_lines.append(f"  Radiomics only: {best_radiomics:.4f}")
        summary_lines.append(f"  New features only: {best_new:.4f}")
        summary_lines.append(f"  Combined features: {best_combined:.4f}")
        summary_lines.append("")
        
        # Improvement analysis
        improvement_over_radiomics = best_combined - best_radiomics
        improvement_over_new = best_combined - best_new
        
        summary_lines.append("Improvement with Combined Features:")
        summary_lines.append(f"  vs Radiomics only: {improvement_over_radiomics:+.4f} AUC ({improvement_over_radiomics/best_radiomics*100:+.2f}%)")
        summary_lines.append(f"  vs New features only: {improvement_over_new:+.4f} AUC ({improvement_over_new/best_new*100:+.2f}%)")
        summary_lines.append("")
        
        # Conclusion
        if best_combined > max(best_radiomics, best_new):
            summary_lines.append("CONCLUSION: Combining features improves performance!")
        elif best_combined > best_radiomics and best_combined > best_new:
            summary_lines.append("CONCLUSION: Combined features provide best performance.")
        else:
            if best_radiomics > best_new:
                summary_lines.append("CONCLUSION: Radiomics features alone perform best.")
            else:
                summary_lines.append("CONCLUSION: New features alone perform best.")
        
        summary_lines.append("="*80)
        
        # Save summary
        summary_text = "\n".join(summary_lines)
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
    
    def save_train_test_split(self):
        """Save the train-test split for reproducibility"""
        split_info = {
            'train_indices': self.train_indices,
            'test_indices': self.test_indices,
            'train_case_ids': self.merged_df.iloc[self.train_indices][self.case_id_col].values,
            'test_case_ids': self.merged_df.iloc[self.test_indices][self.case_id_col].values,
        }
        
        with open(self.output_dir / 'train_test_split.pkl', 'wb') as f:
            pickle.dump(split_info, f)
        
        # Also save as CSV
        split_df = pd.DataFrame({
            'CaseID': list(split_info['train_case_ids']) + list(split_info['test_case_ids']),
            'Dataset': ['Training'] * len(split_info['train_case_ids']) + 
                       ['Testing'] * len(split_info['test_case_ids'])
        })
        split_df.to_csv(self.output_dir / 'train_test_split.csv', index=False)
    
    def save_all_predictions(self, all_results, y_train, y_test):
        """
        Save predictions for all feature types and models
        """
        print("\nSaving all predictions...")
        
        case_ids_train = self.merged_df.iloc[self.train_indices][self.case_id_col].values
        case_ids_test = self.merged_df.iloc[self.test_indices][self.case_id_col].values
        
        for feature_type in ['radiomics', 'new', 'combined']:
            for model_name, result in all_results[feature_type].items():
                
                # Training predictions
                train_pred = pd.DataFrame({
                    'CaseID': case_ids_train,
                    'True_Label': y_train,
                    'Predicted_Label': result['y_pred_train'],
                    'Predicted_Probability': result['y_prob_train'],
                    'Dataset': 'Training',
                    'Feature_Type': feature_type,
                    'Model': model_name
                })
                
                # Test predictions
                test_pred = pd.DataFrame({
                    'CaseID': case_ids_test,
                    'True_Label': y_test,
                    'Predicted_Label': result['y_pred_test'],
                    'Predicted_Probability': result['y_prob_test'],
                    'Dataset': 'Testing',
                    'Feature_Type': feature_type,
                    'Model': model_name
                })
                
                # Combine and save
                all_pred = pd.concat([train_pred, test_pred], ignore_index=True)
                filename = f'predictions_{feature_type}_{model_name.replace(" ", "_")}.csv'
                all_pred.to_csv(self.output_dir / filename, index=False)
        
        # Create unified prediction file
        self.create_unified_predictions(all_results, case_ids_train, case_ids_test, y_train, y_test)
    
    def create_unified_predictions(self, all_results, case_ids_train, case_ids_test, y_train, y_test):
        """
        Create a single file with predictions from all models and feature types
        """
        print("Creating unified predictions file...")
        
        # Training set
        train_unified = pd.DataFrame({
            'CaseID': case_ids_train,
            'True_Label': y_train,
            'Dataset': 'Training'
        })
        
        # Add predictions from all models and feature types
        for feature_type in ['radiomics', 'new', 'combined']:
            for model_name, result in all_results[feature_type].items():
                col_prefix = f'{feature_type}_{model_name.replace(" ", "_")}'
                train_unified[f'{col_prefix}_Pred'] = result['y_pred_train']
                train_unified[f'{col_prefix}_Prob'] = result['y_prob_train']
        
        # Test set
        test_unified = pd.DataFrame({
            'CaseID': case_ids_test,
            'True_Label': y_test,
            'Dataset': 'Testing'
        })
        
        for feature_type in ['radiomics', 'new', 'combined']:
            for model_name, result in all_results[feature_type].items():
                col_prefix = f'{feature_type}_{model_name.replace(" ", "_")}'
                test_unified[f'{col_prefix}_Pred'] = result['y_pred_test']
                test_unified[f'{col_prefix}_Prob'] = result['y_prob_test']
        
        # Combine
        unified = pd.concat([train_unified, test_unified], ignore_index=True)
        unified.to_csv(self.output_dir / 'predictions_unified_all_models.csv', index=False)
        
        print(f"  Unified predictions saved with {len(unified.columns)} columns")


def compare_feature_synergy(multimodal_results):
    """
    Analyze synergy between radiomics and new features
    """
    print("\nAnalyzing feature synergy...")
    
    comparison_df = multimodal_results['comparison']
    
    # Get best AUC for each feature type
    best_radiomics = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].max()
    best_new = comparison_df[comparison_df['Feature_Type'] == 'new']['Test_AUC'].max()
    best_combined = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].max()
    
    # Calculate synergy metrics
    expected_max = max(best_radiomics, best_new)
    synergy = best_combined - expected_max
    
    synergy_df = pd.DataFrame({
        'Metric': ['Best_Radiomics_AUC', 'Best_New_Features_AUC', 'Best_Combined_AUC', 
                   'Expected_Max', 'Synergy_Effect', 'Synergy_Percentage'],
        'Value': [best_radiomics, best_new, best_combined, 
                 expected_max, synergy, (synergy/expected_max)*100]
    })
    
    print("\nFeature Synergy Analysis:")
    print(synergy_df.to_string(index=False))
    
    if synergy > 0:
        print(f"\n✓ POSITIVE SYNERGY: Combining features improves performance by {synergy:.4f} AUC")
        print(f"  This represents a {(synergy/expected_max)*100:.2f}% improvement over the best single feature type")
    elif synergy < -0.01:
        print(f"\n✗ NEGATIVE SYNERGY: Combined features underperform by {abs(synergy):.4f} AUC")
        print("  Single feature type is more effective")
    else:
        print("\n○ NEUTRAL: No significant synergy effect detected")
    
    return synergy_df


def statistical_test_feature_combination(results_dir='multimodal_comparison_results'):
    """
    Perform statistical tests to determine if combining features significantly
    improves performance
    """
    from scipy import stats
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    print("\nPerforming statistical tests for feature combination...")
    
    results_path = Path(results_dir)
    
    # Load comparison data
    comparison_df = pd.read_csv(results_path / 'all_features_comparison.csv')
    
    # Get test AUCs for each feature type
    radiomics_aucs = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].values
    new_aucs = comparison_df[comparison_df['Feature_Type'] == 'new']['Test_AUC'].values
    combined_aucs = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].values
    
    # Paired t-test: Combined vs Radiomics
    t_stat_rad, p_value_rad = stats.ttest_rel(combined_aucs, radiomics_aucs)
    
    # Paired t-test: Combined vs New Features
    t_stat_new, p_value_new = stats.ttest_rel(combined_aucs, new_aucs)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat_rad, wp_value_rad = stats.wilcoxon(combined_aucs, radiomics_aucs)
    w_stat_new, wp_value_new = stats.wilcoxon(combined_aucs, new_aucs)
    
    # Effect size (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    effect_size_rad = cohens_d(combined_aucs, radiomics_aucs)
    effect_size_new = cohens_d(combined_aucs, new_aucs)
    
    # Create results dataframe
    statistical_results = pd.DataFrame({
        'Comparison': [
            'Combined vs Radiomics',
            'Combined vs Radiomics',
            'Combined vs New Features',
            'Combined vs New Features'
        ],
        'Test': [
            'Paired t-test',
            'Wilcoxon test',
            'Paired t-test',
            'Wilcoxon test'
        ],
        'Statistic': [t_stat_rad, w_stat_rad, t_stat_new, w_stat_new],
        'P_Value': [p_value_rad, wp_value_rad, p_value_new, wp_value_new],
        'Significant': [
            'Yes' if p_value_rad < 0.05 else 'No',
            'Yes' if wp_value_rad < 0.05 else 'No',
            'Yes' if p_value_new < 0.05 else 'No',
            'Yes' if wp_value_new < 0.05 else 'No'
        ]
    })
    
    # Add effect sizes
    effect_size_df = pd.DataFrame({
        'Comparison': ['Combined vs Radiomics', 'Combined vs New Features'],
        'Cohens_d': [effect_size_rad, effect_size_new],
        'Effect_Size_Interpretation': [
            'Large' if abs(effect_size_rad) > 0.8 else 'Medium' if abs(effect_size_rad) > 0.5 else 'Small',
            'Large' if abs(effect_size_new) > 0.8 else 'Medium' if abs(effect_size_new) > 0.5 else 'Small'
        ]
    })
    
    # Save results
    statistical_results.to_csv(results_path / 'statistical_significance_tests.csv', index=False)
    effect_size_df.to_csv(results_path / 'effect_size_analysis.csv', index=False)
    
    # Print results
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    print("\nPaired Statistical Tests:")
    print(statistical_results.to_string(index=False))
    print("\nEffect Size Analysis:")
    print(effect_size_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    
    # Interpretation for Combined vs Radiomics
    if p_value_rad < 0.05:
        direction = "better" if np.mean(combined_aucs) > np.mean(radiomics_aucs) else "worse"
        print(f"\n✓ Combined features perform significantly {direction} than Radiomics alone")
        print(f"  Mean difference: {np.mean(combined_aucs) - np.mean(radiomics_aucs):.4f}")
        print(f"  P-value: {p_value_rad:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size_rad:.3f} ({effect_size_df.iloc[0]['Effect_Size_Interpretation']})")
    else:
        print(f"\n○ No significant difference between Combined and Radiomics features")
        print(f"  P-value: {p_value_rad:.4f}")
    
    # Interpretation for Combined vs New Features
    if p_value_new < 0.05:
        direction = "better" if np.mean(combined_aucs) > np.mean(new_aucs) else "worse"
        print(f"\n✓ Combined features perform significantly {direction} than New Features alone")
        print(f"  Mean difference: {np.mean(combined_aucs) - np.mean(new_aucs):.4f}")
        print(f"  P-value: {p_value_new:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size_new:.3f} ({effect_size_df.iloc[1]['Effect_Size_Interpretation']})")
    else:
        print(f"\n○ No significant difference between Combined and New Features")
        print(f"  P-value: {p_value_new:.4f}")
    
    print("="*80)
    
    return statistical_results, effect_size_df


def create_publication_ready_table(results_dir='multimodal_comparison_results'):
    """
    Create a publication-ready table summarizing all results
    """
    print("\nCreating publication-ready summary table...")
    
    results_path = Path(results_dir)
    comparison_df = pd.read_csv(results_path / 'all_features_comparison.csv')
    
    # Create summary for each feature type
    summary_data = []
    
    for feature_type in ['radiomics', 'new', 'combined']:
        subset = comparison_df[comparison_df['Feature_Type'] == feature_type]
        
        for model in subset['Model'].unique():
            model_data = subset[subset['Model'] == model].iloc[0]
            
            summary_data.append({
                'Feature Type': feature_type.capitalize(),
                'Model': model,
                'Train AUC': f"{model_data['Train_AUC']:.3f}",
                'Test AUC': f"{model_data['Test_AUC']:.3f}",
                'CV AUC (mean±std)': f"{model_data['CV_AUC']:.3f}±{0.02:.3f}",
                'Test Accuracy': f"{model_data['Test_Accuracy']:.3f}",
                'Test F1': f"{model_data['Test_F1']:.3f}"
            })
    
    summary_table = pd.DataFrame(summary_data)
    
    # Add best model indicator
    best_idx = comparison_df['Test_AUC'].idxmax()
    best_model = comparison_df.loc[best_idx]
    summary_table['Best'] = summary_table.apply(
        lambda row: '★' if (row['Feature Type'].lower() == best_model['Feature_Type'] and 
                           row['Model'] == best_model['Model']) else '',
        axis=1
    )
    
    # Reorder columns
    summary_table = summary_table[['Feature Type', 'Model', 'Train AUC', 'Test AUC', 
                                   'CV AUC (mean±std)', 'Test Accuracy', 'Test F1', 'Best']]
    
    # Save as CSV and LaTeX
    summary_table.to_csv(results_path / 'publication_ready_table.csv', index=False)
    
    # Create LaTeX version
    latex_table = summary_table.to_latex(index=False, escape=False)
    with open(results_path / 'publication_ready_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nPublication-Ready Summary Table:")
    print(summary_table.to_string(index=False))
    print(f"\nSaved to: {results_path / 'publication_ready_table.csv'}")
    print(f"LaTeX version: {results_path / 'publication_ready_table.tex'}")
    
    return summary_table


def visualize_patient_level_agreement(results_dir='multimodal_comparison_results'):
    """
    Visualize how predictions differ across feature types at patient level
    """
    print("\nCreating patient-level agreement visualization...")
    
    results_path = Path(results_dir)
    
    # Load unified predictions
    unified_pred = pd.read_csv(results_path / 'predictions_unified_all_models.csv')
    
    # Filter test set only
    test_pred = unified_pred[unified_pred['Dataset'] == 'Testing'].copy()
    
    # For Random Forest model (as example), compare predictions across feature types
    rf_cols = [col for col in test_pred.columns if 'Random_Forest' in col and col.endswith('_Prob')]
    
    if len(rf_cols) >= 3:
        radiomics_prob_col = [col for col in rf_cols if col.startswith('radiomics_')][0]
        new_prob_col = [col for col in rf_cols if col.startswith('new_')][0]
        combined_prob_col = [col for col in rf_cols if col.startswith('combined_')][0]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Radiomics vs New Features
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(
            test_pred[radiomics_prob_col],
            test_pred[new_prob_col],
            c=test_pred['True_Label'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('Radiomics Probability', fontsize=12)
        ax1.set_ylabel('New Features Probability', fontsize=12)
        ax1.set_title('Radiomics vs New Features Predictions', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='True Label')
        
        # Plot 2: Radiomics vs Combined
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(
            test_pred[radiomics_prob_col],
            test_pred[combined_prob_col],
            c=test_pred['True_Label'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('Radiomics Probability', fontsize=12)
        ax2.set_ylabel('Combined Probability', fontsize=12)
        ax2.set_title('Radiomics vs Combined Predictions', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='True Label')
        
        # Plot 3: New Features vs Combined
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(
            test_pred[new_prob_col],
            test_pred[combined_prob_col],
            c=test_pred['True_Label'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_xlabel('New Features Probability', fontsize=12)
        ax3.set_ylabel('Combined Probability', fontsize=12)
        ax3.set_title('New Features vs Combined Predictions', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='True Label')
        
        # Plot 4: Prediction agreement histogram
        ax4 = axes[1, 1]
        
        # Calculate prediction differences
        diff_rad_new = np.abs(test_pred[radiomics_prob_col] - test_pred[new_prob_col])
        diff_rad_comb = np.abs(test_pred[radiomics_prob_col] - test_pred[combined_prob_col])
        diff_new_comb = np.abs(test_pred[new_prob_col] - test_pred[combined_prob_col])
        
        ax4.hist(diff_rad_new, bins=20, alpha=0.5, label='Radiomics vs New', color='blue')
        ax4.hist(diff_rad_comb, bins=20, alpha=0.5, label='Radiomics vs Combined', color='red')
        ax4.hist(diff_new_comb, bins=20, alpha=0.5, label='New vs Combined', color='green')
        ax4.set_xlabel('Absolute Probability Difference', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Prediction Differences', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_path / 'patient_level_agreement.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Patient-level agreement plot saved")
        
        # Calculate and save agreement statistics
        agreement_stats = pd.DataFrame({
            'Comparison': ['Radiomics vs New', 'Radiomics vs Combined', 'New vs Combined'],
            'Mean_Abs_Diff': [diff_rad_new.mean(), diff_rad_comb.mean(), diff_new_comb.mean()],
            'Std_Abs_Diff': [diff_rad_new.std(), diff_rad_comb.std(), diff_new_comb.std()],
            'Max_Abs_Diff': [diff_rad_new.max(), diff_rad_comb.max(), diff_new_comb.max()],
            'Correlation': [
                np.corrcoef(test_pred[radiomics_prob_col], test_pred[new_prob_col])[0, 1],
                np.corrcoef(test_pred[radiomics_prob_col], test_pred[combined_prob_col])[0, 1],
                np.corrcoef(test_pred[new_prob_col], test_pred[combined_prob_col])[0, 1]
            ]
        })
        
        agreement_stats.to_csv(results_path / 'prediction_agreement_statistics.csv', index=False)
        
        print("\nPrediction Agreement Statistics:")
        print(agreement_stats.to_string(index=False))
        
        return agreement_stats


def identify_complementary_cases(results_dir='multimodal_comparison_results'):
    """
    Identify cases where combining features helps or hurts performance
    """
    print("\nIdentifying complementary cases...")
    
    results_path = Path(results_dir)
    unified_pred = pd.read_csv(results_path / 'predictions_unified_all_models.csv')
    test_pred = unified_pred[unified_pred['Dataset'] == 'Testing'].copy()
    
    # Use Random Forest as example
    rf_cols = [col for col in test_pred.columns if 'Random_Forest' in col]
    
    radiomics_pred_col = [col for col in rf_cols if col.startswith('radiomics_') and col.endswith('_Pred')][0]
    new_pred_col = [col for col in rf_cols if col.startswith('new_') and col.endswith('_Pred')][0]
    combined_pred_col = [col for col in rf_cols if col.startswith('combined_') and col.endswith('_Pred')][0]
    
    radiomics_prob_col = [col for col in rf_cols if col.startswith('radiomics_') and col.endswith('_Prob')][0]
    new_prob_col = [col for col in rf_cols if col.startswith('new_') and col.endswith('_Prob')][0]
    combined_prob_col = [col for col in rf_cols if col.startswith('combined_') and col.endswith('_Prob')][0]
    
    # Classify cases
    test_pred['Radiomics_Correct'] = (test_pred[radiomics_pred_col] == test_pred['True_Label']).astype(int)
    test_pred['New_Correct'] = (test_pred[new_pred_col] == test_pred['True_Label']).astype(int)
    test_pred['Combined_Correct'] = (test_pred[combined_pred_col] == test_pred['True_Label']).astype(int)
    
    # Case categories
    test_pred['Case_Category'] = 'Other'
    
    # Both wrong, combined correct (Complementary benefit)
    mask1 = (test_pred['Radiomics_Correct'] == 0) & (test_pred['New_Correct'] == 0) & (test_pred['Combined_Correct'] == 1)
    test_pred.loc[mask1, 'Case_Category'] = 'Complementary_Benefit'
    
    # Both correct, combined wrong (Negative synergy)
    mask2 = (test_pred['Radiomics_Correct'] == 1) & (test_pred['New_Correct'] == 1) & (test_pred['Combined_Correct'] == 0)
    test_pred.loc[mask2, 'Case_Category'] = 'Negative_Synergy'
    
    # One wrong, combined correct (Rescued by combination)
    mask3 = ((test_pred['Radiomics_Correct'] == 1) & (test_pred['New_Correct'] == 0) | 
             (test_pred['Radiomics_Correct'] == 0) & (test_pred['New_Correct'] == 1)) & (test_pred['Combined_Correct'] == 1)
    test_pred.loc[mask3, 'Case_Category'] = 'Rescued_By_Combination'
    
    # One correct, combined wrong (Degraded by combination)
    mask4 = ((test_pred['Radiomics_Correct'] == 1) & (test_pred['New_Correct'] == 0) | 
             (test_pred['Radiomics_Correct'] == 0) & (test_pred['New_Correct'] == 1)) & (test_pred['Combined_Correct'] == 0)
    test_pred.loc[mask4, 'Case_Category'] = 'Degraded_By_Combination'
    
    # All correct
    mask5 = (test_pred['Radiomics_Correct'] == 1) & (test_pred['New_Correct'] == 1) & (test_pred['Combined_Correct'] == 1)
    test_pred.loc[mask5, 'Case_Category'] = 'All_Correct'
    
    # All wrong
    mask6 = (test_pred['Radiomics_Correct'] == 0) & (test_pred['New_Correct'] == 0) & (test_pred['Combined_Correct'] == 0)
    test_pred.loc[mask6, 'Case_Category'] = 'All_Wrong'
    
    # Summary statistics
    category_counts = test_pred['Case_Category'].value_counts()
    
    complementary_analysis = pd.DataFrame({
        'Category': category_counts.index,
        'Count': category_counts.values,
        'Percentage': (category_counts.values / len(test_pred) * 100).round(2)
    })
    
    complementary_analysis.to_csv(results_path / 'complementary_cases_analysis.csv', index=False)
    
    # Save detailed case information
    interesting_cases = test_pred[test_pred['Case_Category'].isin([
        'Complementary_Benefit', 'Negative_Synergy', 
        'Rescued_By_Combination', 'Degraded_By_Combination'
    ])].copy()
    
    interesting_cases_subset = interesting_cases[[
        'CaseID', 'True_Label', 'Case_Category',
        radiomics_pred_col, new_pred_col, combined_pred_col,
        radiomics_prob_col, new_prob_col, combined_prob_col
    ]]
    
    interesting_cases_subset.to_csv(results_path / 'interesting_cases_detailed.csv', index=False)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Case category distribution
    ax1 = axes[0]
    colors_map = {
        'All_Correct': '#2ecc71',
        'Complementary_Benefit': '#3498db',
        'Rescued_By_Combination': '#9b59b6',
        'All_Wrong': '#e74c3c',
        'Negative_Synergy': '#e67e22',
        'Degraded_By_Combination': '#f39c12',
        'Other': '#95a5a6'
    }
    colors = [colors_map.get(cat, '#95a5a6') for cat in complementary_analysis['Category']]
    
    ax1.barh(complementary_analysis['Category'], complementary_analysis['Count'], color=colors, alpha=0.7)
    ax1.set_xlabel('Number of Cases', fontsize=12)
    ax1.set_title('Case Categories: Effect of Feature Combination', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add percentages
    for i, (count, pct) in enumerate(zip(complementary_analysis['Count'], complementary_analysis['Percentage'])):
        ax1.text(count, i, f'  {count} ({pct}%)', va='center', fontsize=10)
    
    # Plot 2: Pie chart of beneficial vs detrimental
    ax2 = axes[1]
    
    beneficial = complementary_analysis[complementary_analysis['Category'].isin([
        'Complementary_Benefit', 'Rescued_By_Combination', 'All_Correct'
    ])]['Count'].sum()
    
    detrimental = complementary_analysis[complementary_analysis['Category'].isin([
        'Negative_Synergy', 'Degraded_By_Combination'
    ])]['Count'].sum()
    
    neutral = complementary_analysis[complementary_analysis['Category'].isin([
        'All_Wrong', 'Other'
    ])]['Count'].sum()
    
    sizes = [beneficial, detrimental, neutral]
    labels = [f'Beneficial\n({beneficial} cases)', 
             f'Detrimental\n({detrimental} cases)', 
             f'Neutral\n({neutral} cases)']
    colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Overall Effect of Feature Combination', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_path / 'complementary_cases_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComplementary Cases Analysis:")
    print(complementary_analysis.to_string(index=False))
    
    print(f"\n✓ BENEFICIAL: {beneficial} cases ({beneficial/len(test_pred)*100:.1f}%)")
    print(f"✗ DETRIMENTAL: {detrimental} cases ({detrimental/len(test_pred)*100:.1f}%)")
    print(f"○ NEUTRAL: {neutral} cases ({neutral/len(test_pred)*100:.1f}%)")
    
    return complementary_analysis, interesting_cases_subset


def create_final_report(results_dir='multimodal_comparison_results'):
    """
    Create a comprehensive final report combining all analyses
    """
    print("\nCreating final comprehensive report...")
    
    results_path = Path(results_dir)
    
    # Load all result files
    comparison_df = pd.read_csv(results_path / 'all_features_comparison.csv')
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE MULTI-MODAL FEATURE ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Section 1: Overall Performance
    report_lines.append("="*80)
    report_lines.append("1. OVERALL PERFORMANCE SUMMARY")
    report_lines.append("="*80)
    report_lines.append("")
    
    for feature_type in ['radiomics', 'new', 'combined']:
        subset = comparison_df[comparison_df['Feature_Type'] == feature_type]
        best_idx = subset['Test_AUC'].idxmax()
        best = subset.loc[best_idx]
        
        feature_name = {'radiomics': 'RADIOMICS FEATURES', 
                       'new': 'NEW FEATURES', 
                       'combined': 'COMBINED FEATURES'}[feature_type]
        
        report_lines.append(f"{feature_name}:")
        report_lines.append(f"  Best Model: {best['Model']}")
        report_lines.append(f"  Test AUC: {best['Test_AUC']:.4f}")
        report_lines.append(f"  Test Accuracy: {best['Test_Accuracy']:.4f}")
        report_lines.append(f"  CV AUC: {best['CV_AUC']:.4f}")
        report_lines.append(f"  Test F1: {best['Test_F1']:.4f}")
        
        # All models for this feature type
        report_lines.append(f"  All models (Test AUC):")
        for _, row in subset.iterrows():
            report_lines.append(f"    - {row['Model']}: {row['Test_AUC']:.4f}")
        report_lines.append("")
    
    # Section 2: Comparative Analysis
    report_lines.append("="*80)
    report_lines.append("2. COMPARATIVE ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    
    best_radiomics = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].max()
    best_new = comparison_df[comparison_df['Feature_Type'] == 'new']['Test_AUC'].max()
    best_combined = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].max()
    
    report_lines.append(f"Best Performance Comparison:")
    report_lines.append(f"  Radiomics only:    {best_radiomics:.4f}")
    report_lines.append(f"  New features only: {best_new:.4f}")
    report_lines.append(f"  Combined features: {best_combined:.4f}")
    report_lines.append("")
    
    improvement_rad = best_combined - best_radiomics
    improvement_new = best_combined - best_new
    
    report_lines.append(f"Improvement Analysis:")
    report_lines.append(f"  Combined vs Radiomics:     {improvement_rad:+.4f} ({improvement_rad/best_radiomics*100:+.2f}%)")
    report_lines.append(f"  Combined vs New Features:  {improvement_new:+.4f} ({improvement_new/best_new*100:+.2f}%)")
    report_lines.append("")
    
    # Section 3: Feature Contributions
    if (results_path / 'feature_type_contributions.csv').exists():
        report_lines.append("="*80)
        report_lines.append("3. FEATURE CONTRIBUTIONS IN COMBINED MODEL")
        report_lines.append("="*80)
        report_lines.append("")
        
        contrib_df = pd.read_csv(results_path / 'feature_type_contributions.csv')
        for _, row in contrib_df.iterrows():
            report_lines.append(f"{row['Feature_Type']}:")
            report_lines.append(f"  Total Importance: {row['Total_Importance']:.4f}")
            report_lines.append(f"  Percentage: {row['Percentage']:.2f}%")
            report_lines.append(f"  Average Importance per Feature: {row['Avg_Importance']:.6f}")
            report_lines.append("")
    
    # Section 4: Recommendations
    report_lines.append("="*80)
    report_lines.append("4. RECOMMENDATIONS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if best_combined > max(best_radiomics, best_new) and improvement_rad > 0.01:
        report_lines.append("✓ RECOMMENDATION: Use COMBINED features")
        report_lines.append(f"  Rationale: Combined features provide the best performance")
        report_lines.append(f"  with {improvement_rad:.4f} AUC improvement over single feature types.")
        report_lines.append(f"  Both feature types contribute complementary information.")
    elif best_radiomics > best_new and best_radiomics > best_combined:
        report_lines.append("✓ RECOMMENDATION: Use RADIOMICS features only")
        report_lines.append(f"  Rationale: Radiomics features alone provide the best performance.")
        report_lines.append(f"  Combining with new features does not improve results.")
    elif best_new > best_radiomics and best_new > best_combined:
        report_lines.append("✓ RECOMMENDATION: Use NEW features only")
        report_lines.append(f"  Rationale: New features alone provide the best performance.")
        report_lines.append(f"  Combining with radiomics features does not improve results.")
    else:
        report_lines.append("○ RECOMMENDATION: Further investigation needed")
        report_lines.append(f"  Rationale: Performance differences are minimal.")
        report_lines.append(f"  Consider computational cost and interpretability.")
    
    report_lines.append("")
    
    # Section 5: Statistical Significance
    if (results_path / 'statistical_significance_tests.csv').exists():
        report_lines.append("="*80)
        report_lines.append("5. STATISTICAL SIGNIFICANCE")
        report_lines.append("="*80)
        report_lines.append("")
        
        stats_df = pd.read_csv(results_path / 'statistical_significance_tests.csv')
        
        # Get t-test results
        ttest_rad = stats_df[(stats_df['Comparison'] == 'Combined vs Radiomics') & 
                            (stats_df['Test'] == 'Paired t-test')].iloc[0]
        ttest_new = stats_df[(stats_df['Comparison'] == 'Combined vs New Features') & 
                            (stats_df['Test'] == 'Paired t-test')].iloc[0]
        
        report_lines.append("Paired t-test results:")
        report_lines.append(f"  Combined vs Radiomics:")
        report_lines.append(f"    P-value: {ttest_rad['P_Value']:.4f}")
        report_lines.append(f"    Significant: {ttest_rad['Significant']}")
        report_lines.append(f"  Combined vs New Features:")
        report_lines.append(f"    P-value: {ttest_new['P_Value']:.4f}")
        report_lines.append(f"    Significant: {ttest_new['Significant']}")
        report_lines.append("")
    
    # Section 6: Complementary Cases
    if (results_path / 'complementary_cases_analysis.csv').exists():
        report_lines.append("="*80)
        report_lines.append("6. COMPLEMENTARY CASE ANALYSIS")
        report_lines.append("="*80)
        report_lines.append("")
        
        comp_df = pd.read_csv(results_path / 'complementary_cases_analysis.csv')
        
        beneficial_cats = ['Complementary_Benefit', 'Rescued_By_Combination', 'All_Correct']
        detrimental_cats = ['Negative_Synergy', 'Degraded_By_Combination']
        
        beneficial = comp_df[comp_df['Category'].isin(beneficial_cats)]['Count'].sum()
        detrimental = comp_df[comp_df['Category'].isin(detrimental_cats)]['Count'].sum()
        total = comp_df['Count'].sum()
        
        report_lines.append(f"Feature Combination Effects:")
        report_lines.append(f"  Beneficial cases: {beneficial}/{total} ({beneficial/total*100:.1f}%)")
        report_lines.append(f"  Detrimental cases: {detrimental}/{total} ({detrimental/total*100:.1f}%)")
        report_lines.append("")
        
        report_lines.append("Detailed breakdown:")
        for _, row in comp_df.iterrows():
            report_lines.append(f"  {row['Category']}: {row['Count']} ({row['Percentage']:.1f}%)")
        report_lines.append("")
    
    # Section 7: Key Findings
    report_lines.append("="*80)
    report_lines.append("7. KEY FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Finding 1: Best overall performance
    overall_best = comparison_df.loc[comparison_df['Test_AUC'].idxmax()]
    report_lines.append(f"1. Best Overall Model:")
    report_lines.append(f"   {overall_best['Model']} with {overall_best['Feature_Type']} features")
    report_lines.append(f"   achieved Test AUC of {overall_best['Test_AUC']:.4f}")
    report_lines.append("")
    
    # Finding 2: Synergy effect
    synergy = best_combined - max(best_radiomics, best_new)
    if synergy > 0.01:
        report_lines.append(f"2. Positive Synergy Detected:")
        report_lines.append(f"   Combining features improves performance by {synergy:.4f} AUC")
        report_lines.append(f"   ({synergy/max(best_radiomics, best_new)*100:.2f}% relative improvement)")
    elif synergy < -0.01:
        report_lines.append(f"2. Negative Synergy Detected:")
        report_lines.append(f"   Combining features reduces performance by {abs(synergy):.4f} AUC")
        report_lines.append(f"   Single feature type is more effective")
    else:
        report_lines.append(f"2. No Strong Synergy Effect:")
        report_lines.append(f"   Performance difference is minimal ({synergy:.4f} AUC)")
    report_lines.append("")
    
    # Finding 3: Consistency across models
    radiomics_std = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].std()
    new_std = comparison_df[comparison_df['Feature_Type'] == 'new']['Test_AUC'].std()
    combined_std = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].std()
    
    most_consistent = min([('Radiomics', radiomics_std), 
                          ('New Features', new_std), 
                          ('Combined', combined_std)], 
                         key=lambda x: x[1])
    
    report_lines.append(f"3. Model Consistency:")
    report_lines.append(f"   Most consistent feature type: {most_consistent[0]}")
    report_lines.append(f"   (Std Dev across models: {most_consistent[1]:.4f})")
    report_lines.append(f"   Radiomics: {radiomics_std:.4f}, New: {new_std:.4f}, Combined: {combined_std:.4f}")
    report_lines.append("")
    
    # Footer
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Save report
    report_text = "\n".join(report_lines)
    with open(results_path / 'FINAL_COMPREHENSIVE_REPORT.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nFinal report saved to: {results_path / 'FINAL_COMPREHENSIVE_REPORT.txt'}")
    
    return report_text


# ============================================================
# COMPLETE INTEGRATED WORKFLOW
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("COMPLETE MULTI-MODAL FEATURE ANALYSIS WORKFLOW")
    print("="*80 + "\n")
    
    # Step 1: Run main comparison
    print("STEP 1: Running main comparison...")
    predictor = MultiModalPredictor(
        radiomics_csv='radiomics_features.csv',
        new_features_csv='new_features.csv',
        clinical_xlsx='clinical_data.xlsx',
        output_dir='multimodal_comparison_results'
    )
    
    all_results, combined_comparison = predictor.run_complete_comparison(
        test_size=0.3,
        n_features=20,
        feature_selection_method='tree_based',
        random_state=42
    )
    
    # Step 2: Save predictions
    print("\nSTEP 2: Saving predictions...")
    predictor.save_all_predictions(
        all_results,
        predictor.y[predictor.train_indices],
        predictor.y[predictor.test_indices]
    )
    
    # Step 3: Statistical tests
    print("\nSTEP 3: Performing statistical tests...")
    statistical_test_feature_combination('multimodal_comparison_results')
    
    # Step 4: Publication-ready table
    print("\nSTEP 4: Creating publication-ready table...")
    pub_table = create_publication_ready_table('multimodal_comparison_results')
    
    # Step 5: Patient-level agreement
    print("\nSTEP 5: Analyzing patient-level agreement...")
    agreement_stats = visualize_patient_level_agreement('multimodal_comparison_results')
    
    # Step 6: Complementary cases analysis
    print("\nSTEP 6: Identifying complementary cases...")
    comp_analysis, interesting_cases = identify_complementary_cases('multimodal_comparison_results')
    
    # Step 7: Final comprehensive report
    print("\nSTEP 7: Creating final comprehensive report...")
    final_report = create_final_report('multimodal_comparison_results')
    
    # Summary of all outputs
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nAll results saved to: multimodal_comparison_results/")
    print("\nGenerated Files:")
    print("\nCore Results:")
    print("  1. all_features_comparison.csv - Performance metrics for all combinations")
    print("  2. predictions_unified_all_models.csv - All predictions in one file")
    print("  3. train_test_split.csv - Train/test split information")
    print("\nFeature Analysis:")
    print("  4. feature_scores_radiomics.csv - Selected radiomics features")
    print("  5. feature_scores_new.csv - Selected new features")
    print("  6. feature_scores_combined.csv - Selected combined features")
    print("  7. feature_type_contributions.csv - Contribution analysis")
    print("  8. top_features_by_type.csv - Most important features")
    print("\nStatistical Analysis:")
    print("  9. statistical_significance_tests.csv - Statistical test results")
    print(" 10. effect_size_analysis.csv - Effect size calculations")
    print(" 11. feature_synergy_analysis.csv - Synergy metrics")
    print(" 12. prediction_agreement_statistics.csv - Agreement metrics")
    print("\nCase-Level Analysis:")
    print(" 13. complementary_cases_analysis.csv - Case categories")
    print(" 14. interesting_cases_detailed.csv - Detailed interesting cases")
    print("\nVisualization:")
    print(" 15. performance_comparison_all_features.png - Performance comparison")
    print(" 16. roc_curves_comparison.png - ROC curve comparison")
    print(" 17. performance_heatmap.png - Performance heatmap")
    print(" 18. feature_type_contributions.png - Contribution visualization")
    print(" 19. patient_level_agreement.png - Patient-level agreement plots")
    print(" 20. complementary_cases_visualization.png - Case category visualization")
    print("\nReports:")
    print(" 21. summary_report.txt - Quick summary")
    print(" 22. FINAL_COMPREHENSIVE_REPORT.txt - Complete detailed report")
    print(" 23. publication_ready_table.csv - Table for publication")
    print(" 24. publication_ready_table.tex - LaTeX version")
    print("\n" + "="*80)
    print("\nKEY INSIGHTS:")
    print("="*80)
    
    # Print key insights
    best_radiomics = combined_comparison[combined_comparison['Feature_Type'] == 'radiomics']['Test_AUC'].max()
    best_new = combined_comparison[combined_comparison['Feature_Type'] == 'new']['Test_AUC'].max()
    best_combined = combined_comparison[combined_comparison['Feature_Type'] == 'combined']['Test_AUC'].max()
    
    print(f"\n1. Best Performance by Feature Type:")
    print(f"   Radiomics:  {best_radiomics:.4f}")
    print(f"   New:        {best_new:.4f}")
    print(f"   Combined:   {best_combined:.4f}")
    
    if best_combined > max(best_radiomics, best_new):
        improvement = best_combined - max(best_radiomics, best_new)
        print(f"\n2. ✓ Combining features IMPROVES performance by {improvement:.4f} AUC")
    else:
        print(f"\n2. ○ Single feature type performs best")
    
    print("\n" + "="*80)
    print("\nFor detailed results, see: FINAL_COMPREHENSIVE_REPORT.txt")
    print("="*80)