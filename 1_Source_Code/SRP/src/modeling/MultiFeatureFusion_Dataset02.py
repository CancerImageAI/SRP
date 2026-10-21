import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
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
    2. Spatial Radiomics features only  
    3. Combined (radiomics + spatial radiomics features)
    
    Includes RFE feature selection and SMOTE for class balancing
    """
    
    def __init__(self, radiomics_csv, spatial_radiomics_csv, clinical_xlsx, output_dir='multimodal_results'):
        """
        Initialize multi-modal predictor
        
        Args:
            radiomics_csv: Path to radiomics features CSV
            spatial_radiomics_csv: Path to spatial radiomics features CSV
            clinical_xlsx: Path to clinical data with Response labels
            output_dir: Directory to save results
        """
        self.radiomics_csv = radiomics_csv
        self.spatial_radiomics_csv = spatial_radiomics_csv
        self.clinical_xlsx = clinical_xlsx
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scaler_radiomics =  StandardScaler()
        self.scaler_spatial = StandardScaler()
        self.scaler_combined = StandardScaler()
        
        self.train_indices = None
        self.test_indices = None
        
    def load_and_merge_all_data(self, case_id_col='CaseID', label_col='Response'):
        """
        Load radiomics, spatial radiomics, and clinical data, then merge all
        """
        print("Loading all data sources...")
        
        # Load radiomics features
        radiomics_df = pd.read_csv(self.radiomics_csv)
        print(f"Loaded radiomics: {len(radiomics_df)} cases, {len(radiomics_df.columns)} columns")
        
        # Load spatial radiomics features
        spatial_radiomics_df = pd.read_csv(self.spatial_radiomics_csv)
        print(f"Loaded spatial radiomics: {len(spatial_radiomics_df)} cases, {len(spatial_radiomics_df.columns)} columns")
        
        # Load clinical data
        clinical_df = pd.read_excel(self.clinical_xlsx)
        print(f"Loaded clinical data: {len(clinical_df)} cases")
        
        # Merge all datasets
        merged_df = pd.merge(radiomics_df, spatial_radiomics_df, on=case_id_col, 
                            how='inner', suffixes=('_radiomics', '_spatial'))
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
        print(f"\nResponse expression distribution:")
        print(merged_df[label_col].value_counts())
        
        return merged_df
    
    def prepare_feature_sets(self, exclude_cols=None):
        """
        Prepare three feature sets: radiomics, spatial radiomics, and combined
        
        Returns:
            Dictionary with 'radiomics', 'spatial_radiomics', 'combined' feature matrices
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
        
        # Identify spatial radiomics features (with _spatial suffix or from spatial radiomics file)
        spatial_radiomics_features = [col for col in self.merged_df.columns 
                       if col.endswith('_spatial') or 
                       (col in pd.read_csv(self.spatial_radiomics_csv).columns and 
                        col not in all_exclude_cols and 
                        col not in radiomics_features)]
        
        # Remove duplicates and excluded columns
        radiomics_features = [col for col in radiomics_features if col not in all_exclude_cols]
        spatial_radiomics_features = [col for col in spatial_radiomics_features if col not in all_exclude_cols]
        
        print(f"\nFeature sets prepared:")
        print(f"  Radiomics features: {len(radiomics_features)}")
        print(f"  Spatial Radiomics features: {len(spatial_radiomics_features)}")
        print(f"  Combined features: {len(radiomics_features) + len(spatial_radiomics_features)}")
        
        # Extract feature matrices
        X_radiomics = self.merged_df[radiomics_features].values
        X_spatial = self.merged_df[spatial_radiomics_features].values
        X_combined = np.concatenate([X_radiomics, X_spatial], axis=1)
        
        y = self.merged_df[self.label_col].values
        y = np.where(y == 'pCR', 1, 0)  # Convert to binary labels
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        if np.isnan(X_radiomics).any():
            print("Imputing missing values in radiomics features...")
            imputer_rad = SimpleImputer(strategy='median')
            X_radiomics = imputer_rad.fit_transform(X_radiomics)
        
        if np.isnan(X_spatial).any():
            print("Imputing missing values in spatial radiomics features...")
            imputer_spatial = SimpleImputer(strategy='median')
            X_spatial = imputer_spatial.fit_transform(X_spatial)
        
        if np.isnan(X_combined).any():
            print("Imputing missing values in combined features...")
            imputer_combined = SimpleImputer(strategy='median')
            X_combined = imputer_combined.fit_transform(X_combined)
        
        feature_sets = {
            'radiomics': {
                'X': X_radiomics,
                'feature_names': radiomics_features
            },
            'spatial_radiomics': {
                'X': X_spatial,
                'feature_names': spatial_radiomics_features
            },
            'combined': {
                'X': X_combined,
                'feature_names': radiomics_features + spatial_radiomics_features
            }
        }
        
        self.feature_sets = feature_sets
        self.y = y
        
        return feature_sets, y
    
    def feature_selection(self, X, y, feature_names, method='univariate', k=10, prefix=''):
        """
        Perform feature selection with multiple methods including RFE
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            method: 'univariate', 'tree_based', 'mutual_info', or 'rfe'
            k: Number of features to select
            prefix: Prefix for print statements
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
        
        elif method == 'rfe':
            # Use Logistic Regression as the base estimator for RFE
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=min(k, X.shape[1]), step=1)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Get feature ranking
            rankings = selector.ranking_
            feature_scores = pd.DataFrame({
                'Feature': feature_names,
                'Ranking': rankings,
                'Selected': selector.support_
            }).sort_values('Ranking')
            
            print(f"  RFE selected {len(selected_features)} features")
            
            return X_selected, selected_features, selector, feature_scores
    
    def apply_smote(self, X_train, y_train, prefix=''):
        """
        Apply SMOTE to balance the training dataset
        
        Args:
            X_train: Training features
            y_train: Training labels
            prefix: Prefix for print statements
        
        Returns:
            X_resampled, y_resampled: Balanced training data
        """
        print(f"\n{prefix}Applying SMOTE for class balancing...")
        print(f"  Original class distribution: {np.bincount(y_train)}")
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"  Resampled class distribution: {np.bincount(y_resampled)}")
        print(f"  Training samples: {len(y_train)} -> {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_models_for_feature_type(self, X_train, X_test, y_train, y_test, feature_type, 
                                   X_train_original=None, y_train_original=None):
        """
        Train multiple models for a specific feature type, including MLP
        
        Args:
            X_train: Training features (after SMOTE if applied)
            X_test: Test features
            y_train: Training labels (after SMOTE if applied)
            y_test: Test labels
            feature_type: Type of features being used
            X_train_original: Original training features (before SMOTE)
            y_train_original: Original training labels (before SMOTE)
        """
        print(f"\nTraining models for {feature_type} features...")
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                random_state=42, early_stopping=True, 
                                validation_fraction=0.1, n_iter_no_change=10)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            # Predictions on SMOTE-augmented training data
            y_pred_train = model.predict(X_train)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            
            # Predictions on ORIGINAL training data (before SMOTE)
            if X_train_original is not None and y_train_original is not None:
                y_pred_train_original = model.predict(X_train_original)
                y_prob_train_original = model.predict_proba(X_train_original)[:, 1]
            else:
                y_pred_train_original = y_pred_train
                y_prob_train_original = y_prob_train
            
            # Predictions on test data
            y_pred_test = model.predict(X_test)
            y_prob_test = model.predict_proba(X_test)[:, 1]
            
            # Metrics on SMOTE-augmented training data
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'AUC': roc_auc_score(y_train, y_prob_train),
                'f1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            }
            
            # Metrics on test data
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'AUC': roc_auc_score(y_test, y_prob_test),
                'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            }
            
            # Cross-validation on SMOTE-augmented data
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
                'y_pred_train': y_pred_train,
                'y_prob_train_original': y_prob_train_original,  # Predictions on original training data
                'y_pred_train_original': y_pred_train_original   # Predictions on original training data
            }
            
            print(f"  {name}: Test AUC = {test_metrics['AUC']:.4f}, CV AUC = {cv_scores.mean():.4f}")
        
        return results
    
    def run_complete_comparison(self, test_size=0.2, n_features=10, 
                           feature_selection_method='rfe', random_state=42,
                           use_smote=True):
        """
        Run complete comparison across all three feature types with RFE and SMOTE
        
        For combined model: concatenate selected features from radiomics and spatial radiomics
        """
        print("="*80)
        print("MULTI-MODAL FEATURE COMPARISON PIPELINE")
        print("="*80)
        print(f"Feature Selection Method: {feature_selection_method}")
        print(f"SMOTE Balancing: {'Enabled' if use_smote else 'Disabled'}")
        print(f"Combined Model Strategy: Concatenate selected features from individual models")
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
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        # Save split
        self.save_train_test_split()
        
        # Store all results
        all_results = {}
        all_comparisons = {}
        
        # Store original y_train for saving predictions later
        y_train_original = y_train.copy()
        
        # Store selected features and data for combined model
        selected_data = {}
        
        # Process radiomics and spatial_radiomics first
        for feature_type in ['radiomics', 'spatial_radiomics']:
            print(f"\n{'='*80}")
            print(f"PROCESSING {feature_type.upper().replace('_', ' ')} FEATURES")
            print(f"{'='*80}")
            
            X = feature_sets[feature_type]['X']
            feature_names = feature_sets[feature_type]['feature_names']
            
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Reset y_train to original for each feature type
            y_train_current = y_train_original.copy()
            
            # Standardize
            if feature_type == 'radiomics':
                X_train_scaled = self.scaler_radiomics.fit_transform(X_train)
                X_test_scaled = self.scaler_radiomics.transform(X_test)
            elif feature_type == 'spatial_radiomics':
                X_train_scaled = self.scaler_spatial.fit_transform(X_train)
                X_test_scaled = self.scaler_spatial.transform(X_test)
            
            # Feature selection
            X_train_selected, selected_features, selector, feature_scores = self.feature_selection(
                X_train_scaled, y_train_current, feature_names, 
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
                self.output_dir / f'feature_scores_{feature_type}_{feature_selection_method}.csv', 
                index=False
            )
            
            # Store selected data for combined model
            selected_data[feature_type] = {
                'X_train_selected': X_train_selected.copy(),
                'X_test_selected': X_test_selected.copy(),
                'selected_features': selected_features,
                'selector': selector,
                'feature_scores': feature_scores
            }
            
            # Store original training data before SMOTE
            X_train_selected_original = X_train_selected.copy()
            y_train_original_copy = y_train_current.copy()
            
            # Apply SMOTE if enabled
            if use_smote:
                X_train_balanced, y_train_balanced = self.apply_smote(
                    X_train_selected, y_train_current, 
                    prefix=f"[{feature_type}] "
                )
            else:
                X_train_balanced = X_train_selected
                y_train_balanced = y_train_current
            
            # Train models - pass both SMOTE and original data
            results = self.train_models_for_feature_type(
                X_train_balanced, X_test_selected, y_train_balanced, y_test, feature_type,
                X_train_original=X_train_selected_original,
                y_train_original=y_train_original_copy
            )
            
            all_results[feature_type] = results
            
            # Create comparison dataframe for this feature type
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Feature_Type': feature_type,
                    'Model': model_name,
                    'Train_AUC': result['train_metrics']['AUC'],
                    'Test_AUC': result['test_metrics']['AUC'],
                    'CV_AUC': result['cv_auc_mean'],
                    'CV_AUC_Std': result['cv_auc_std'],
                    'Test_Accuracy': result['test_metrics']['accuracy'],
                    'Test_F1': result['test_metrics']['f1'],
                    'SMOTE_Applied': use_smote,
                    'Feature_Selection': feature_selection_method,
                    'Num_Features': n_features
                })
            
            all_comparisons[feature_type] = pd.DataFrame(comparison_data)
        
        # Now process COMBINED model by concatenating selected features
        print(f"\n{'='*80}")
        print(f"PROCESSING COMBINED FEATURES")
        print(f"{'='*80}")
        print(f"Strategy: Concatenating {n_features} selected features from each type")
        print(f"Total features in combined model: {n_features * 2}")
        
        # Concatenate selected features from radiomics and spatial radiomics
        X_train_combined = np.concatenate([
            selected_data['radiomics']['X_train_selected'],
            selected_data['spatial_radiomics']['X_train_selected']
        ], axis=1)
        
        X_test_combined = np.concatenate([
            selected_data['radiomics']['X_test_selected'],
            selected_data['spatial_radiomics']['X_test_selected']
        ], axis=1)
        
        # Combine feature names
        combined_feature_names = (
            selected_data['radiomics']['selected_features'] + 
            selected_data['spatial_radiomics']['selected_features']
        )
        
        print(f"Combined features: {len(combined_feature_names)} total")
        print(f"  - {len(selected_data['radiomics']['selected_features'])} from Radiomics")
        print(f"  - {len(selected_data['spatial_radiomics']['selected_features'])} from Spatial Radiomics")
        
        # Scale the combined features
        X_train_combined_scaled = self.scaler_combined.fit_transform(X_train_combined)
        X_test_combined_scaled = self.scaler_combined.transform(X_test_combined)
        
        # Save combined feature information
        combined_feature_info = pd.DataFrame({
            'Feature': combined_feature_names,
            'Source': (
                ['Radiomics'] * len(selected_data['radiomics']['selected_features']) +
                ['Spatial_Radiomics'] * len(selected_data['spatial_radiomics']['selected_features'])
            ),
            'Original_Rank': list(range(1, len(selected_data['radiomics']['selected_features']) + 1)) +
                            list(range(1, len(selected_data['spatial_radiomics']['selected_features']) + 1))
        })
        combined_feature_info.to_csv(
            self.output_dir / f'feature_list_combined_{feature_selection_method}.csv',
            index=False
        )
        
        # Store original training data before SMOTE
        X_train_combined_original = X_train_combined_scaled.copy()
        y_train_original_copy = y_train_original.copy()
        
        # Apply SMOTE if enabled
        if use_smote:
            X_train_combined_balanced, y_train_combined_balanced = self.apply_smote(
                X_train_combined_scaled, y_train_original, 
                prefix=f"[combined] "
            )
        else:
            X_train_combined_balanced = X_train_combined_scaled
            y_train_combined_balanced = y_train_original
        
        # Train models on combined features
        results_combined = self.train_models_for_feature_type(
            X_train_combined_balanced, X_test_combined_scaled, 
            y_train_combined_balanced, y_test, 'combined',
            X_train_original=X_train_combined_original,
            y_train_original=y_train_original_copy
        )
        
        all_results['combined'] = results_combined
        
        # Create comparison dataframe for combined
        comparison_data = []
        for model_name, result in results_combined.items():
            comparison_data.append({
                'Feature_Type': 'combined',
                'Model': model_name,
                'Train_AUC': result['train_metrics']['AUC'],
                'Test_AUC': result['test_metrics']['AUC'],
                'CV_AUC': result['cv_auc_mean'],
                'CV_AUC_Std': result['cv_auc_std'],
                'Test_Accuracy': result['test_metrics']['accuracy'],
                'Test_F1': result['test_metrics']['f1'],
                'SMOTE_Applied': use_smote,
                'Feature_Selection': feature_selection_method,
                'Num_Features': len(combined_feature_names)
            })
        
        all_comparisons['combined'] = pd.DataFrame(comparison_data)
        
        # Store selected data for combined (for feature contribution analysis)
        selected_data['combined'] = {
            'selected_features': combined_feature_names,
            'feature_sources': combined_feature_info
        }
        self.selected_feature_data = selected_data
        
        # Combine all comparisons
        combined_comparison = pd.concat(all_comparisons.values(), ignore_index=True)
        combined_comparison.to_csv(
            self.output_dir / f'all_features_comparison_{feature_selection_method}_smote_{use_smote}.csv', 
            index=False
        )
        
        # Create visualizations
        self.create_comparison_visualizations(combined_comparison, all_results, y_test)
        
        # Store the best classifier name for documentation
        self.best_classifier_for_roc = getattr(self, 'best_classifier_for_roc', None)
        
        # Analyze feature contributions in combined model
        self.analyze_feature_contributions_v2(selected_data, all_results)
        
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
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Test AUC by feature type
        ax1 = axes[0, 0]
        pivot_data = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_AUC')
        pivot_data = pivot_data.rename(columns={
            'radiomics': 'Radiomics',
            'spatial_radiomics': 'Spatial Radiomics',
            'combined': 'Combined'
        })
        pivot_data.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Test AUC Comparison Across Feature Types', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AUC Score')
        ax1.set_xlabel('Model')
        ax1.legend(title='Feature Type')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 1.0])
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # CV AUC by feature type
        ax2 = axes[0, 1]
        pivot_data = comparison_df.pivot(index='Model', columns='Feature_Type', values='CV_AUC')
        pivot_data = pivot_data.rename(columns={
            'radiomics': 'Radiomics',
            'spatial_radiomics': 'Spatial Radiomics',
            'combined': 'Combined'
        })
        pivot_data.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Cross-Validation AUC Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CV AUC Score')
        ax2.set_xlabel('Model')
        ax2.legend(title='Feature Type')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.5, 1.0])
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Test Accuracy by feature type
        ax3 = axes[1, 0]
        pivot_data = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_Accuracy')
        pivot_data = pivot_data.rename(columns={
            'radiomics': 'Radiomics',
            'spatial_radiomics': 'Spatial Radiomics',
            'combined': 'Combined'
        })
        pivot_data.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Model')
        ax3.legend(title='Feature Type')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1.0])
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Best performance for each feature type
        ax4 = axes[1, 1]
        best_by_type = comparison_df.groupby('Feature_Type')['Test_AUC'].max()
        feature_type_names = ['Combined', 'Radiomics', 'Spatial\nRadiomics']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax4.bar(range(len(best_by_type)), best_by_type.values, color=colors, alpha=0.7)
        ax4.set_xticks(range(len(best_by_type)))
        ax4.set_xticklabels(feature_type_names)
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
        
        # Plot 2: ROC curves comparison (using same classifier)
        best_classifier_name = self.plot_roc_curves_comparison(all_results, y_test)
        self.best_classifier_for_roc = best_classifier_name
        
        # Plot 3: Heatmap of performance metrics
        self.plot_performance_heatmap(comparison_df)
    
    def plot_roc_curves_comparison(self, all_results, y_test):
        """
        Plot ROC curves using the SAME classifier for all feature types for fair comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Same classifier across all feature types
        ax1 = axes[0]
        colors = {'radiomics': '#1f77b4', 'spatial_radiomics': '#ff7f0e', 'combined': '#2ca02c'}
        
        # Find the best classifier for combined model
        best_auc_combined = 0
        best_classifier_name = None
        
        for model_name, result in all_results['combined'].items():
            current_auc = result['test_metrics']['AUC']
            if current_auc > best_auc_combined:
                best_auc_combined = current_auc
                best_classifier_name = model_name
        
        print(f"\n  Using {best_classifier_name} for ROC curve comparison (best for combined model)")
        
        # Plot ROC curves for the same classifier across all feature types
        for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
            result = all_results[feature_type][best_classifier_name]
            y_probs = result['y_prob_test']
            auc = result['test_metrics']['AUC']
            
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            label_name = {
                'radiomics': 'Radiomics', 
                'spatial_radiomics': 'Spatial Radiomics', 
                'combined': 'Combined'
            }[feature_type]
            
            ax1.plot(fpr, tpr, color=colors[feature_type], linewidth=2.5, 
                    label=f'{label_name}\nAUC = {auc:.3f}')
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=13)
        ax1.set_ylabel('True Positive Rate', fontsize=13)
        ax1.set_title(f'ROC Curves - {best_classifier_name}\n(Same Classifier, Different Features)', 
                    fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All models for combined features
        ax2 = axes[1]
        for model_name, result in all_results['combined'].items():
            fpr, tpr, _ = roc_curve(y_test, result['y_prob_test'])
            auc = result['test_metrics']['AUC']
            ax2.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate', fontsize=13)
        ax2.set_ylabel('True Positive Rate', fontsize=13)
        ax2.set_title('ROC Curves - All Models\n(Combined Features)', 
                    fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ROC curves comparison saved")
        
        # Return the classifier name used for documentation
        return best_classifier_name
        

    def plot_performance_heatmap(self, comparison_df):
        """
        Create heatmap of performance metrics
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Heatmap 1: Test A
        # Heatmap 1: Test AUC
        ax1 = axes[0]
        pivot_auc = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_AUC')
        pivot_auc.columns = ['Combined', 'Radiomics', 'Spatial Radiomics']
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, ax=ax1, cbar_kws={'label': 'AUC Score'})
        ax1.set_title('Test AUC Heatmap', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=12)
        ax1.set_xlabel('Feature Type', fontsize=12)
        
        # Heatmap 2: Test Accuracy
        ax2 = axes[1]
        pivot_acc = comparison_df.pivot(index='Model', columns='Feature_Type', values='Test_Accuracy')
        pivot_acc.columns = ['Combined', 'Radiomics', 'Spatial Radiomics']
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
            
            # Identify which features are radiomics vs spatial radiomics
            n_radiomics = len(feature_sets['radiomics']['feature_names'])
            
            radiomics_importance = importances[:n_radiomics]
            spatial_importance = importances[n_radiomics:]
            
            # Calculate contribution percentages
            total_radiomics = radiomics_importance.sum()
            total_spatial = spatial_importance.sum()
            total = total_radiomics + total_spatial
            
            contribution_df = pd.DataFrame({
                'Feature_Type': ['Radiomics', 'Spatial_Radiomics'],
                'Total_Importance': [total_radiomics, total_spatial],
                'Percentage': [total_radiomics/total*100, total_spatial/total*100],
                'Num_Features': [len(radiomics_importance), len(spatial_importance)],
                'Avg_Importance': [total_radiomics/len(radiomics_importance) if len(radiomics_importance) > 0 else 0, 
                                  total_spatial/len(spatial_importance) if len(spatial_importance) > 0 else 0]
            })
            
            contribution_df.to_csv(self.output_dir / 'feature_type_contributions.csv', index=False)
            
            print("\nFeature Type Contributions (Combined Model - Random Forest):")
            print(contribution_df.to_string(index=False))
            
            # Get top features from each type
            top_n = 10
            
            if len(radiomics_importance) > 0:
                radiomics_top_indices = np.argsort(radiomics_importance)[::-1][:min(top_n, len(radiomics_importance))]
                top_radiomics_features = pd.DataFrame({
                    'Feature': [feature_sets['radiomics']['feature_names'][i] for i in radiomics_top_indices],
                    'Importance': radiomics_importance[radiomics_top_indices],
                    'Type': 'Radiomics'
                })
            else:
                top_radiomics_features = pd.DataFrame(columns=['Feature', 'Importance', 'Type'])
            
            if len(spatial_importance) > 0:
                spatial_top_indices = np.argsort(spatial_importance)[::-1][:min(top_n, len(spatial_importance))]
                top_spatial_features = pd.DataFrame({
                    'Feature': [feature_sets['spatial_radiomics']['feature_names'][i] for i in spatial_top_indices],
                    'Importance': spatial_importance[spatial_top_indices],
                    'Type': 'Spatial_Radiomics'
                })
            else:
                top_spatial_features = pd.DataFrame(columns=['Feature', 'Importance', 'Type'])
            
            top_features_combined = pd.concat([top_radiomics_features, top_spatial_features], ignore_index=True)
            top_features_combined.to_csv(self.output_dir / 'top_features_by_type.csv', index=False)
            
            # Visualize contributions
            self.plot_feature_contributions(contribution_df, top_features_combined)

    def analyze_feature_contributions_v2(self, selected_data, all_results):
        """
        Analyze which feature type contributes more in the combined model
        Considers that combined features come from pre-selected radiomics and spatial features
        """
        print("\nAnalyzing feature contributions in combined model...")
        
        # Get feature importance from Random Forest (combined)
        rf_combined = all_results['combined']['Random Forest']['model']
        
        if hasattr(rf_combined, 'feature_importances_'):
            importances = rf_combined.feature_importances_
            
            # Get feature source information
            feature_sources = selected_data['combined']['feature_sources']
            
            # Calculate importance by source
            radiomics_mask = feature_sources['Source'] == 'Radiomics'
            spatial_mask = feature_sources['Source'] == 'Spatial_Radiomics'
            
            radiomics_importance = importances[radiomics_mask.values]
            spatial_importance = importances[spatial_mask.values]
            
            # Calculate contribution percentages
            total_radiomics = radiomics_importance.sum()
            total_spatial = spatial_importance.sum()
            total = total_radiomics + total_spatial
            
            contribution_df = pd.DataFrame({
                'Feature_Type': ['Radiomics', 'Spatial_Radiomics'],
                'Total_Importance': [total_radiomics, total_spatial],
                'Percentage': [total_radiomics/total*100, total_spatial/total*100],
                'Num_Features': [radiomics_mask.sum(), spatial_mask.sum()],
                'Avg_Importance': [
                    total_radiomics/radiomics_mask.sum() if radiomics_mask.sum() > 0 else 0,
                    total_spatial/spatial_mask.sum() if spatial_mask.sum() > 0 else 0
                ]
            })
            
            contribution_df.to_csv(self.output_dir / 'feature_type_contributions.csv', index=False)
            
            print("\nFeature Type Contributions (Combined Model - Random Forest):")
            print(contribution_df.to_string(index=False))
            
            # Get top features from each type
            top_n = 10
            
            # Create detailed feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'Feature': feature_sources['Feature'].values,
                'Source': feature_sources['Source'].values,
                'Importance': importances,
                'Original_Rank': feature_sources['Original_Rank'].values
            }).sort_values('Importance', ascending=False)
            
            feature_importance_df.to_csv(
                self.output_dir / 'feature_importance_combined_detailed.csv',
                index=False
            )
            
            # Get top features from each source
            top_radiomics = feature_importance_df[
                feature_importance_df['Source'] == 'Radiomics'
            ].head(top_n)
            
            top_spatial = feature_importance_df[
                feature_importance_df['Source'] == 'Spatial_Radiomics'
            ].head(top_n)
            
            top_features_combined = pd.concat([top_radiomics, top_spatial], ignore_index=True)
            top_features_combined.to_csv(self.output_dir / 'top_features_by_type.csv', index=False)
            
            # Visualize contributions
            self.plot_feature_contributions_v2(contribution_df, feature_importance_df, top_n)


    def plot_feature_contributions_v2(self, contribution_df, feature_importance_df, top_n=10):
        """
        Visualize feature type contributions with enhanced information
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
        # Plot 1: Pie chart of contributions
        ax1 = axes[0, 0]
        colors = ['#1f77b4', '#ff7f0e']
        wedges, texts, autotexts = ax1.pie(
            contribution_df['Percentage'], 
            labels=['Radiomics Features', 'Spatial Radiomics Features'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        ax1.set_title('Feature Type Contribution in Combined Model\n(Random Forest)', 
                    fontsize=14, fontweight='bold')
        
        # Plot 2: Average importance per feature
        ax2 = axes[0, 1]
        feature_types = ['Radiomics', 'Spatial\nRadiomics']
        avg_importances = contribution_df['Avg_Importance'].values
        bars = ax2.bar(feature_types, avg_importances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Average Importance per Feature', fontsize=12, fontweight='bold')
        ax2.set_title('Average Feature Importance by Type', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        # Plot 3: Top features from each type
        ax3 = axes[1, 0]
        
        # Get top features from each source
        top_radiomics = feature_importance_df[
            feature_importance_df['Source'] == 'Radiomics'
        ].head(top_n)
        top_spatial = feature_importance_df[
            feature_importance_df['Source'] == 'Spatial_Radiomics'
        ].head(top_n)
        
        y_pos = np.arange(len(top_radiomics) + len(top_spatial))
        importances = list(top_radiomics['Importance'].values) + list(top_spatial['Importance'].values)
        feature_labels = list(top_radiomics['Feature'].values) + list(top_spatial['Feature'].values)
        bar_colors = ['#1f77b4'] * len(top_radiomics) + ['#ff7f0e'] * len(top_spatial)
        
        ax3.barh(y_pos, importances, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(feature_labels, fontsize=8)
        ax3.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax3.set_title(f'Top {top_n} Features from Each Type', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, label='Radiomics'),
            Patch(facecolor='#ff7f0e', alpha=0.7, label='Spatial Radiomics')
        ]
        ax3.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Plot 4: Distribution of importance by source
        ax4 = axes[1, 1]
        
        radiomics_importances = feature_importance_df[
            feature_importance_df['Source'] == 'Radiomics'
        ]['Importance'].values
        spatial_importances = feature_importance_df[
            feature_importance_df['Source'] == 'Spatial_Radiomics'
        ]['Importance'].values
        
        positions = [1, 2]
        bp = ax4.boxplot([radiomics_importances, spatial_importances],
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        labels=['Radiomics', 'Spatial\nRadiomics'])
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Feature Importance', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        means = [radiomics_importances.mean(), spatial_importances.mean()]
        ax4.plot(positions, means, 'r*', markersize=15, label='Mean', zorder=3)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_type_contributions.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Feature contributions visualization saved")
    
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
            labels=['Radiomics Features', 'Spatial Radiomics Features'],
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
        spatial_features = top_features_df[top_features_df['Type'] == 'Spatial_Radiomics']
        
        y_pos = np.arange(len(radiomics_features) + len(spatial_features))
        importances = list(radiomics_features['Importance'].values) + list(spatial_features['Importance'].values)
        feature_labels = list(radiomics_features['Feature'].values) + list(spatial_features['Feature'].values)
        bar_colors = ['#1f77b4'] * len(radiomics_features) + ['#ff7f0e'] * len(spatial_features)
        
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
            Patch(facecolor='#ff7f0e', alpha=0.7, label='Spatial Radiomics')
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_type_contributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Feature contributions visualization saved")

    
    def create_feature_selection_summary(self):
        """
        Create a summary of selected features from each model
        """
        print("\nCreating feature selection summary...")
        
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("FEATURE SELECTION SUMMARY")
        summary_lines.append("="*80)
        summary_lines.append("")
        
        if hasattr(self, 'selected_feature_data'):
            selected_data = self.selected_feature_data
            
            # Radiomics features
            summary_lines.append("RADIOMICS FEATURES:")
            summary_lines.append(f"  Selected: {len(selected_data['radiomics']['selected_features'])} features")
            summary_lines.append("  Top 10:")
            for i, feat in enumerate(selected_data['radiomics']['selected_features'][:10], 1):
                summary_lines.append(f"    {i:2d}. {feat}")
            summary_lines.append("")
            
            # Spatial Radiomics features
            summary_lines.append("SPATIAL RADIOMICS FEATURES:")
            summary_lines.append(f"  Selected: {len(selected_data['spatial_radiomics']['selected_features'])} features")
            summary_lines.append("  Top 10:")
            for i, feat in enumerate(selected_data['spatial_radiomics']['selected_features'][:10], 1):
                summary_lines.append(f"    {i:2d}. {feat}")
            summary_lines.append("")
            
            # Combined features
            summary_lines.append("COMBINED MODEL:")
            summary_lines.append(f"  Total features: {len(selected_data['combined']['selected_features'])}")
            summary_lines.append(f"  From Radiomics: {len(selected_data['radiomics']['selected_features'])}")
            summary_lines.append(f"  From Spatial Radiomics: {len(selected_data['spatial_radiomics']['selected_features'])}")
            summary_lines.append("")
            summary_lines.append("  Strategy: Concatenation of selected features from individual models")
            summary_lines.append("  Rationale: Preserve best features from each domain")
            summary_lines.append("")
        
        summary_lines.append("="*80)
        
        summary_text = "\n".join(summary_lines)
        with open(self.output_dir / 'feature_selection_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print(summary_text)

    
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
        
        # Add configuration info
        smote_status = comparison_df['SMOTE_Applied'].iloc[0]
        fs_method = comparison_df['Feature_Selection'].iloc[0]
        summary_lines.append(f"Configuration:")
        summary_lines.append(f"  Feature Selection Method: {fs_method.upper()}")
        summary_lines.append(f"  SMOTE Balancing: {'Enabled' if smote_status else 'Disabled'}")
        summary_lines.append(f"  Combined Model Strategy: Concatenate selected features from individual models")
        summary_lines.append("")
        
        # Feature counts
        radiomics_count = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Num_Features'].iloc[0]
        spatial_count = comparison_df[comparison_df['Feature_Type'] == 'spatial_radiomics']['Num_Features'].iloc[0]
        combined_count = comparison_df[comparison_df['Feature_Type'] == 'combined']['Num_Features'].iloc[0]
        
        summary_lines.append(f"Feature Counts:")
        summary_lines.append(f"  Radiomics model:          {radiomics_count} features")
        summary_lines.append(f"  Spatial Radiomics model:  {spatial_count} features")
        summary_lines.append(f"  Combined model:           {combined_count} features ({radiomics_count} + {spatial_count})")
        summary_lines.append("")
        
        # Best model for each feature type
        for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
            subset = comparison_df[comparison_df['Feature_Type'] == feature_type]
            best_idx = subset['Test_AUC'].idxmax()
            best_row = subset.loc[best_idx]
            
            feature_name = {
                'radiomics': 'Radiomics', 
                'spatial_radiomics': 'Spatial Radiomics', 
                'combined': 'Combined'
            }[feature_type]
            
            summary_lines.append(f"{feature_name} Features - Best Model: {best_row['Model']}")
            summary_lines.append(f"  Test AUC: {best_row['Test_AUC']:.4f}")
            summary_lines.append(f"  Test Accuracy: {best_row['Test_Accuracy']:.4f}")
            summary_lines.append(f"  CV AUC: {best_row['CV_AUC']:.4f} ± {best_row['CV_AUC_Std']:.4f}")
            summary_lines.append(f"  Test F1: {best_row['Test_F1']:.4f}")
            summary_lines.append("")
        
        # Compare best models across feature types
        best_radiomics = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].max()
        best_spatial = comparison_df[comparison_df['Feature_Type'] == 'spatial_radiomics']['Test_AUC'].max()
        best_combined = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].max()
        
        summary_lines.append("Performance Comparison:")
        summary_lines.append(f"  Radiomics only:          {best_radiomics:.4f}")
        summary_lines.append(f"  Spatial Radiomics only:  {best_spatial:.4f}")
        summary_lines.append(f"  Combined features:       {best_combined:.4f}")
        summary_lines.append("")
        
        # Improvement analysis
        improvement_over_radiomics = best_combined - best_radiomics
        improvement_over_spatial = best_combined - best_spatial
        
        summary_lines.append("Improvement with Combined Features:")
        summary_lines.append(f"  vs Radiomics only:        {improvement_over_radiomics:+.4f} AUC ({improvement_over_radiomics/best_radiomics*100:+.2f}%)")
        summary_lines.append(f"  vs Spatial Radiomics only: {improvement_over_spatial:+.4f} AUC ({improvement_over_spatial/best_spatial*100:+.2f}%)")
        summary_lines.append("")
        
        # Feature contribution analysis
        if (self.output_dir / 'feature_type_contributions.csv').exists():
            contrib_df = pd.read_csv(self.output_dir / 'feature_type_contributions.csv')
            summary_lines.append("Feature Contributions in Combined Model:")
            for _, row in contrib_df.iterrows():
                summary_lines.append(f"  {row['Feature_Type']:20s}: {row['Percentage']:5.1f}% (avg importance: {row['Avg_Importance']:.4f})")
            summary_lines.append("")
        
        # Conclusion
        if best_combined > max(best_radiomics, best_spatial):
            summary_lines.append("CONCLUSION: Combining selected features improves performance!")
            summary_lines.append("  The concatenation strategy successfully integrates complementary information")
            summary_lines.append("  from both radiomics and spatial radiomics features.")
        elif best_combined > best_radiomics and best_combined > best_spatial:
            summary_lines.append("CONCLUSION: Combined features provide best performance.")
        else:
            if best_radiomics > best_spatial:
                summary_lines.append("CONCLUSION: Radiomics features alone perform best.")
            else:
                summary_lines.append("CONCLUSION: Spatial Radiomics features alone perform best.")
            summary_lines.append("  Combining features did not improve performance in this case.")
        
        summary_lines.append("="*80)
        
        # Save summary
        summary_text = "\n".join(summary_lines)
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        
        # Also create feature selection summary
        self.create_feature_selection_summary()
    
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

    def save_all_predictions(self, all_results, y_train_original, y_test):
        """
        Save predictions for all feature types and models
        
        Args:
            y_train_original: Original training labels (before SMOTE)
            y_test: Test labels
        """
        print("\nSaving all predictions...")
        
        case_ids_train = self.merged_df.iloc[self.train_indices][self.case_id_col].values
        case_ids_test = self.merged_df.iloc[self.test_indices][self.case_id_col].values
        
        for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
            for model_name, result in all_results[feature_type].items():
                
                # Training predictions - use predictions on ORIGINAL training data
                train_pred = pd.DataFrame({
                    'CaseID': case_ids_train,
                    'True_Label': y_train_original,
                    'Predicted_Label': result['y_pred_train_original'],
                    'Predicted_Probability': result['y_prob_train_original'],
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
        self.create_unified_predictions(all_results, case_ids_train, case_ids_test, y_train_original, y_test)


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
        for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
            for model_name, result in all_results[feature_type].items():
                col_prefix = f'{feature_type}_{model_name.replace(" ", "_")}'
                # Use original training predictions (not SMOTE-augmented)
                train_unified[f'{col_prefix}_Pred'] = result['y_pred_train_original']
                train_unified[f'{col_prefix}_Prob'] = result['y_prob_train_original']
        
        # Test set
        test_unified = pd.DataFrame({
            'CaseID': case_ids_test,
            'True_Label': y_test,
            'Dataset': 'Testing'
        })
        
        for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
            for model_name, result in all_results[feature_type].items():
                col_prefix = f'{feature_type}_{model_name.replace(" ", "_")}'
                test_unified[f'{col_prefix}_Pred'] = result['y_pred_test']
                test_unified[f'{col_prefix}_Prob'] = result['y_prob_test']
        
        # Combine
        unified = pd.concat([train_unified, test_unified], ignore_index=True)
        unified.to_csv(self.output_dir / 'predictions_unified_all_models.csv', index=False)
        
        print(f"  Unified predictions saved with {len(unified.columns)} columns")

    def save_models_and_selectors(self, all_results, feature_selection_method):
        """
        Save trained models and feature selectors for future use
        """
        print("\nSaving trained models and feature selectors...")
        
        models_dir = self.output_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        
        for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
            for model_name, result in all_results[feature_type].items():
                # Save model
                model_filename = f'model_{feature_type}_{model_name.replace(" ", "_")}.pkl'
                with open(models_dir / model_filename, 'wb') as f:
                    pickle.dump(result['model'], f)
        
        # Save scalers
        with open(models_dir / 'scaler_radiomics.pkl', 'wb') as f:
            pickle.dump(self.scaler_radiomics, f)
        with open(models_dir / 'scaler_spatial.pkl', 'wb') as f:
            pickle.dump(self.scaler_spatial, f)
        with open(models_dir / 'scaler_combined.pkl', 'wb') as f:
            pickle.dump(self.scaler_combined, f)
        
        print(f"  Models and scalers saved to {models_dir}")

def plot_comprehensive_roc_comparison(all_results, y_test, results_dir='multimodal_results'):
    """
    Create comprehensive ROC curve comparison with multiple panels
    """
    print("\nCreating comprehensive ROC curve comparison...")
    
    results_path = Path(results_dir)
    
    # Find best classifier for combined model
    best_auc_combined = 0
    best_classifier_name = None
    
    for model_name, result in all_results['combined'].items():
        current_auc = result['test_metrics']['AUC']
        if current_auc > best_auc_combined:
            best_auc_combined = current_auc
            best_classifier_name = model_name
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    colors_features = {'radiomics': '#1f77b4', 'spatial_radiomics': '#ff7f0e', 'combined': '#2ca02c'}
    colors_models = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Panel 1: Same classifier (best from combined) across all feature types
    ax1 = axes[0, 0]
    for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
        result = all_results[feature_type][best_classifier_name]
        y_probs = result['y_prob_test']
        auc = result['test_metrics']['AUC']
        sens = result['test_metrics']['Sensitivity']
        spec = result['test_metrics']['Specificity']
        
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        label_name = {
            'radiomics': 'Radiomics', 
            'spatial_radiomics': 'Spatial Radiomics', 
            'combined': 'Combined'
        }[feature_type]
        
        ax1.plot(fpr, tpr, color=colors_features[feature_type], linewidth=3, 
                label=f'{label_name}: AUC={auc:.3f}\nSens={sens:.3f}, Spec={spec:.3f}')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{best_classifier_name}\nAcross All Feature Types', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # Panel 2: All models for Radiomics features
    ax2 = axes[0, 1]
    for i, (model_name, result) in enumerate(all_results['radiomics'].items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_prob_test'])
        auc = result['test_metrics']['AUC']
        ax2.plot(fpr, tpr, linewidth=2, color=colors_models[i], 
                label=f'{model_name}: {auc:.3f}')
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Radiomics Features\nAll Models', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    # Panel 3: All models for Spatial Radiomics features
    ax3 = axes[1, 0]
    for i, (model_name, result) in enumerate(all_results['spatial_radiomics'].items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_prob_test'])
        auc = result['test_metrics']['AUC']
        ax3.plot(fpr, tpr, linewidth=2, color=colors_models[i], 
                label=f'{model_name}: {auc:.3f}')
    
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Spatial Radiomics Features\nAll Models', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-0.02, 1.02])
    ax3.set_ylim([-0.02, 1.02])
    
    # Panel 4: All models for Combined features
    ax4 = axes[1, 1]
    for i, (model_name, result) in enumerate(all_results['combined'].items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_prob_test'])
        auc = result['test_metrics']['AUC']
        linestyle = '-' if model_name == best_classifier_name else '--'
        linewidth = 3 if model_name == best_classifier_name else 2
        ax4.plot(fpr, tpr, linewidth=linewidth, linestyle=linestyle, 
                color=colors_models[i], label=f'{model_name}: {auc:.3f}')
    
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax4.set_title('Combined Features\nAll Models (Best: solid line)', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-0.02, 1.02])
    ax4.set_ylim([-0.02, 1.02])
    
    plt.suptitle(f'Comprehensive ROC Curve Analysis\nBest Combined Classifier: {best_classifier_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(results_path / 'roc_curves_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Comprehensive ROC curves saved (using {best_classifier_name} for comparison)")


def create_classifier_specific_roc_comparison(all_results, y_test, results_dir='multimodal_results'):
    """
    Create separate ROC comparisons for each classifier
    """
    print("\nCreating classifier-specific ROC comparisons...")
    
    results_path = Path(results_dir)
    
    # Get all model names
    model_names = list(all_results['combined'].keys())
    
    # Create figure with subplots for each classifier
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    colors = {'radiomics': '#1f77b4', 'spatial_radiomics': '#ff7f0e', 'combined': '#2ca02c'}
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
            result = all_results[feature_type][model_name]
            fpr, tpr, _ = roc_curve(y_test, result['y_prob_test'])
            auc = result['test_metrics']['AUC']
            
            label_name = {
                'radiomics': 'Radiomics', 
                'spatial_radiomics': 'Spatial Radiomics', 
                'combined': 'Combined'
            }[feature_type]
            
            ax.plot(fpr, tpr, color=colors[feature_type], linewidth=2.5, 
                   label=f'{label_name}: {auc:.3f}')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('ROC Curves by Classifier\n(Comparing Feature Types)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(results_path / 'roc_curves_by_classifier.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Classifier-specific ROC curves saved")
        
def compare_feature_synergy(comparison_df):
    """
    Analyze synergy between radiomics and spatial radiomics features
    """
    print("\nAnalyzing feature synergy...")
    
    # Get best AUC for each feature type
    best_radiomics = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].max()
    best_spatial = comparison_df[comparison_df['Feature_Type'] == 'spatial_radiomics']['Test_AUC'].max()
    best_combined = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].max()
    
    # Calculate synergy metrics
    expected_max = max(best_radiomics, best_spatial)
    synergy = best_combined - expected_max
    
    synergy_df = pd.DataFrame({
        'Metric': ['Best_Radiomics_AUC', 'Best_Spatial_Radiomics_AUC', 'Best_Combined_AUC', 
                   'Expected_Max', 'Synergy_Effect', 'Synergy_Percentage'],
        'Value': [best_radiomics, best_spatial, best_combined, 
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


def statistical_test_feature_combination(results_dir='multimodal_results'):
    """
    Perform statistical tests to determine if combining features significantly
    improves performance
    """
    from scipy import stats
    
    print("\nPerforming statistical tests for feature combination...")
    
    results_path = Path(results_dir)
    
    # Load comparison data
    comparison_files = list(results_path.glob('all_features_comparison_*.csv'))
    if not comparison_files:
        print("Error: No comparison files found")
        return None, None
    
    comparison_df = pd.read_csv(comparison_files[0])
    
    # Get test AUCs for each feature type
    radiomics_aucs = comparison_df[comparison_df['Feature_Type'] == 'radiomics']['Test_AUC'].values
    spatial_aucs = comparison_df[comparison_df['Feature_Type'] == 'spatial_radiomics']['Test_AUC'].values
    combined_aucs = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].values
    
    # Paired t-test: Combined vs Radiomics
    t_stat_rad, p_value_rad = stats.ttest_rel(combined_aucs, radiomics_aucs)
    
    # Paired t-test: Combined vs Spatial Radiomics
    t_stat_spatial, p_value_spatial = stats.ttest_rel(combined_aucs, spatial_aucs)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat_rad, wp_value_rad = stats.wilcoxon(combined_aucs, radiomics_aucs)
    w_stat_spatial, wp_value_spatial = stats.wilcoxon(combined_aucs, spatial_aucs)
    
    # Effect size (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    effect_size_rad = cohens_d(combined_aucs, radiomics_aucs)
    effect_size_spatial = cohens_d(combined_aucs, spatial_aucs)
    
    # Create results dataframe
    statistical_results = pd.DataFrame({
        'Comparison': [
            'Combined vs Radiomics',
            'Combined vs Radiomics',
            'Combined vs Spatial Radiomics',
            'Combined vs Spatial Radiomics'
        ],
        'Test': [
            'Paired t-test',
            'Wilcoxon test',
            'Paired t-test',
            'Wilcoxon test'
        ],
        'Statistic': [t_stat_rad, w_stat_rad, t_stat_spatial, w_stat_spatial],
        'P_Value': [p_value_rad, wp_value_rad, p_value_spatial, wp_value_spatial],
        'Significant': [
            'Yes' if p_value_rad < 0.05 else 'No',
            'Yes' if wp_value_rad < 0.05 else 'No',
            'Yes' if p_value_spatial < 0.05 else 'No',
            'Yes' if wp_value_spatial < 0.05 else 'No'
        ]
    })
    
    # Add effect sizes
    effect_size_df = pd.DataFrame({
        'Comparison': ['Combined vs Radiomics', 'Combined vs Spatial Radiomics'],
        'Cohens_d': [effect_size_rad, effect_size_spatial],
        'Effect_Size_Interpretation': [
            'Large' if abs(effect_size_rad) > 0.8 else 'Medium' if abs(effect_size_rad) > 0.5 else 'Small',
            'Large' if abs(effect_size_spatial) > 0.8 else 'Medium' if abs(effect_size_spatial) > 0.5 else 'Small'
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
    
    # Interpretation for Combined vs Spatial Radiomics
    if p_value_spatial < 0.05:
        direction = "better" if np.mean(combined_aucs) > np.mean(spatial_aucs) else "worse"
        print(f"\n✓ Combined features perform significantly {direction} than Spatial Radiomics alone")
        print(f"  Mean difference: {np.mean(combined_aucs) - np.mean(spatial_aucs):.4f}")
        print(f"  P-value: {p_value_spatial:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size_spatial:.3f} ({effect_size_df.iloc[1]['Effect_Size_Interpretation']})")
    else:
        print(f"\n○ No significant difference between Combined and Spatial Radiomics features")
        print(f"  P-value: {p_value_spatial:.4f}")
    
    print("="*80)
    
    return statistical_results, effect_size_df


def create_publication_ready_table(results_dir='multimodal_results'):
    """
    Create a publication-ready table summarizing all results
    """
    print("\nCreating publication-ready summary table...")
    
    results_path = Path(results_dir)
    comparison_files = list(results_path.glob('all_features_comparison_*.csv'))
    if not comparison_files:
        print("Error: No comparison files found")
        return None
    
    comparison_df = pd.read_csv(comparison_files[0])
    
    # Create summary for each feature type
    summary_data = []
    
    for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
        subset = comparison_df[comparison_df['Feature_Type'] == feature_type]
        
        for model in subset['Model'].unique():
            model_data = subset[subset['Model'] == model].iloc[0]
            
            feature_name = {
                'radiomics': 'Radiomics',
                'spatial_radiomics': 'Spatial Radiomics',
                'combined': 'Combined'
            }[feature_type]
            
            summary_data.append({
                'Feature Type': feature_name,
                'Model': model,
                'Train AUC': f"{model_data['Train_AUC']:.3f}",
                'Test AUC': f"{model_data['Test_AUC']:.3f}",
                'CV AUC (mean±std)': f"{model_data['CV_AUC']:.3f}±{model_data['CV_AUC_Std']:.3f}",
                'Test Accuracy': f"{model_data['Test_Accuracy']:.3f}",
                'Test F1': f"{model_data['Test_F1']:.3f}"
            })
    
    summary_table = pd.DataFrame(summary_data)
    
    # Add best model indicator
    best_idx = comparison_df['Test_AUC'].idxmax()
    best_model = comparison_df.loc[best_idx]
    
    feature_type_map = {
        'radiomics': 'Radiomics',
        'spatial_radiomics': 'Spatial Radiomics',
        'combined': 'Combined'
    }
    
    summary_table['Best'] = summary_table.apply(
        lambda row: '★' if (row['Feature Type'] == feature_type_map.get(best_model['Feature_Type'], '') and 
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


def visualize_patient_level_agreement(results_dir='multimodal_results'):
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
        spatial_prob_col = [col for col in rf_cols if col.startswith('spatial_radiomics_')][0]
        combined_prob_col = [col for col in rf_cols if col.startswith('combined_')][0]
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Radiomics vs Spatial Radiomics
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(
            test_pred[radiomics_prob_col],
            test_pred[spatial_prob_col],
            c=test_pred['True_Label'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('Radiomics Probability', fontsize=12)
        ax1.set_ylabel('Spatial Radiomics Probability', fontsize=12)
        ax1.set_title('Radiomics vs Spatial Radiomics Predictions', fontsize=13, fontweight='bold')
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
        
        # Plot 3: Spatial Radiomics vs Combined
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(
            test_pred[spatial_prob_col],
            test_pred[combined_prob_col],
            c=test_pred['True_Label'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_xlabel('Spatial Radiomics Probability', fontsize=12)
        ax3.set_ylabel('Combined Probability', fontsize=12)
        ax3.set_title('Spatial Radiomics vs Combined Predictions', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='True Label')
        
        # Plot 4: Prediction agreement histogram
        ax4 = axes[1, 1]
        
        # Calculate prediction differences
        diff_rad_spatial = np.abs(test_pred[radiomics_prob_col] - test_pred[spatial_prob_col])
        diff_rad_comb = np.abs(test_pred[radiomics_prob_col] - test_pred[combined_prob_col])
        diff_spatial_comb = np.abs(test_pred[spatial_prob_col] - test_pred[combined_prob_col])
        
        ax4.hist(diff_rad_spatial, bins=20, alpha=0.5, label='Radiomics vs Spatial', color='blue')
        ax4.hist(diff_rad_comb, bins=20, alpha=0.5, label='Radiomics vs Combined', color='red')
        ax4.hist(diff_spatial_comb, bins=20, alpha=0.5, label='Spatial vs Combined', color='green')
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
            'Comparison': ['Radiomics vs Spatial', 'Radiomics vs Combined', 'Spatial vs Combined'],
            'Mean_Abs_Diff': [diff_rad_spatial.mean(), diff_rad_comb.mean(), diff_spatial_comb.mean()],
            'Std_Abs_Diff': [diff_rad_spatial.std(), diff_rad_comb.std(), diff_spatial_comb.std()],
            'Max_Abs_Diff': [diff_rad_spatial.max(), diff_rad_comb.max(), diff_spatial_comb.max()],
            'Correlation': [
                np.corrcoef(test_pred[radiomics_prob_col], test_pred[spatial_prob_col])[0, 1],
                np.corrcoef(test_pred[radiomics_prob_col], test_pred[combined_prob_col])[0, 1],
                np.corrcoef(test_pred[spatial_prob_col], test_pred[combined_prob_col])[0, 1]
            ]
        })
        
        agreement_stats.to_csv(results_path / 'prediction_agreement_statistics.csv', index=False)
        
        print("\nPrediction Agreement Statistics:")
        print(agreement_stats.to_string(index=False))
        
        return agreement_stats


def identify_complementary_cases(results_dir='multimodal_results'):
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
    spatial_pred_col = [col for col in rf_cols if col.startswith('spatial_radiomics_') and col.endswith('_Pred')][0]
    combined_pred_col = [col for col in rf_cols if col.startswith('combined_') and col.endswith('_Pred')][0]
    
    radiomics_prob_col = [col for col in rf_cols if col.startswith('radiomics_') and col.endswith('_Prob')][0]
    spatial_prob_col = [col for col in rf_cols if col.startswith('spatial_radiomics_') and col.endswith('_Prob')][0]
    combined_prob_col = [col for col in rf_cols if col.startswith('combined_') and col.endswith('_Prob')][0]
    
    # Classify cases
    test_pred['Radiomics_Correct'] = (test_pred[radiomics_pred_col] == test_pred['True_Label']).astype(int)
    test_pred['Spatial_Correct'] = (test_pred[spatial_pred_col] == test_pred['True_Label']).astype(int)
    test_pred['Combined_Correct'] = (test_pred[combined_pred_col] == test_pred['True_Label']).astype(int)
    
    # Case categories
    test_pred['Case_Category'] = 'Other'
    
    # Both wrong, combined correct (Complementary benefit)
    mask1 = (test_pred['Radiomics_Correct'] == 0) & (test_pred['Spatial_Correct'] == 0) & (test_pred['Combined_Correct'] == 1)
    test_pred.loc[mask1, 'Case_Category'] = 'Complementary_Benefit'
    
    # Both correct, combined wrong (Negative synergy)
    mask2 = (test_pred['Radiomics_Correct'] == 1) & (test_pred['Spatial_Correct'] == 1) & (test_pred['Combined_Correct'] == 0)
    test_pred.loc[mask2, 'Case_Category'] = 'Negative_Synergy'
    
    # One wrong, combined correct (Rescued by combination)
    mask3 = ((test_pred['Radiomics_Correct'] == 1) & (test_pred['Spatial_Correct'] == 0) | 
             (test_pred['Radiomics_Correct'] == 0) & (test_pred['Spatial_Correct'] == 1)) & (test_pred['Combined_Correct'] == 1)
    test_pred.loc[mask3, 'Case_Category'] = 'Rescued_By_Combination'
    
    # One correct, combined wrong (Degraded by combination)
    mask4 = ((test_pred['Radiomics_Correct'] == 1) & (test_pred['Spatial_Correct'] == 0) | 
             (test_pred['Radiomics_Correct'] == 0) & (test_pred['Spatial_Correct'] == 1)) & (test_pred['Combined_Correct'] == 0)
    test_pred.loc[mask4, 'Case_Category'] = 'Degraded_By_Combination'
    
    # All correct
    mask5 = (test_pred['Radiomics_Correct'] == 1) & (test_pred['Spatial_Correct'] == 1) & (test_pred['Combined_Correct'] == 1)
    test_pred.loc[mask5, 'Case_Category'] = 'All_Correct'
    
    # All wrong
    mask6 = (test_pred['Radiomics_Correct'] == 0) & (test_pred['Spatial_Correct'] == 0) & (test_pred['Combined_Correct'] == 0)
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
        radiomics_pred_col, spatial_pred_col, combined_pred_col,
        radiomics_prob_col, spatial_prob_col, combined_prob_col
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


def create_final_report(results_dir='multimodal_results'):
    """
    Create a comprehensive final report combining all analyses
    """
    print("\nCreating final comprehensive report...")
    
    results_path = Path(results_dir)
    
    # Load all result files
    comparison_files = list(results_path.glob('all_features_comparison_*.csv'))
    if not comparison_files:
        print("Error: No comparison files found")
        return None
    
    comparison_df = pd.read_csv(comparison_files[0])
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE MULTI-MODAL FEATURE ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Configuration
    report_lines.append("="*80)
    report_lines.append("CONFIGURATION")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Feature Selection Method: {comparison_df['Feature_Selection'].iloc[0].upper()}")
    report_lines.append(f"SMOTE Balancing: {'Enabled' if comparison_df['SMOTE_Applied'].iloc[0] else 'Disabled'}")
    report_lines.append("")
    
    # Section 1: Overall Performance
    report_lines.append("="*80)
    report_lines.append("1. OVERALL PERFORMANCE SUMMARY")
    report_lines.append("="*80)
    report_lines.append("")
    
    for feature_type in ['radiomics', 'spatial_radiomics', 'combined']:
        subset = comparison_df[comparison_df['Feature_Type'] == feature_type]
        best_idx = subset['Test_AUC'].idxmax()
        best = subset.loc[best_idx]
        
        feature_name = {
            'radiomics': 'RADIOMICS FEATURES', 
            'spatial_radiomics': 'SPATIAL RADIOMICS FEATURES', 
            'combined': 'COMBINED FEATURES'
        }[feature_type]
        
        report_lines.append(f"{feature_name}:")
        report_lines.append(f"  Best Model: {best['Model']}")
        report_lines.append(f"  Test AUC: {best['Test_AUC']:.4f}")
        report_lines.append(f"  Test Accuracy: {best['Test_Accuracy']:.4f}")
        report_lines.append(f"  CV AUC: {best['CV_AUC']:.4f} ± {best['CV_AUC_Std']:.4f}")
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
    best_spatial = comparison_df[comparison_df['Feature_Type'] == 'spatial_radiomics']['Test_AUC'].max()
    best_combined = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].max()
    
    report_lines.append(f"Best Performance Comparison:")
    report_lines.append(f"  Radiomics only:          {best_radiomics:.4f}")
    report_lines.append(f"  Spatial Radiomics only:  {best_spatial:.4f}")
    report_lines.append(f"  Combined features:       {best_combined:.4f}")
    report_lines.append("")
    
    improvement_rad = best_combined - best_radiomics
    improvement_spatial = best_combined - best_spatial
    
    report_lines.append(f"Improvement Analysis:")
    report_lines.append(f"  Combined vs Radiomics:         {improvement_rad:+.4f} AUC ({improvement_rad/best_radiomics*100:+.2f}%)")
    report_lines.append(f"  Combined vs Spatial Radiomics: {improvement_spatial:+.4f} AUC ({improvement_spatial/best_spatial*100:+.2f}%)")
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
            report_lines.append(f"  Number of Features: {int(row['Num_Features'])}")
            report_lines.append(f"  Average Importance per Feature: {row['Avg_Importance']:.6f}")
            report_lines.append("")
    
    # Section 4: Model Performance Across Feature Types
    report_lines.append("="*80)
    report_lines.append("4. MODEL PERFORMANCE ACROSS FEATURE TYPES")
    report_lines.append("="*80)
    report_lines.append("")
    
    for model in comparison_df['Model'].unique():
        model_subset = comparison_df[comparison_df['Model'] == model]
        report_lines.append(f"{model}:")
        for _, row in model_subset.iterrows():
            feature_name = {
                'radiomics': 'Radiomics',
                'spatial_radiomics': 'Spatial Radiomics',
                'combined': 'Combined'
            }[row['Feature_Type']]
            report_lines.append(f"  {feature_name:20s}: AUC = {row['Test_AUC']:.4f}, Acc = {row['Test_Accuracy']:.4f}")
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
        ttest_spatial = stats_df[(stats_df['Comparison'] == 'Combined vs Spatial Radiomics') & 
                                (stats_df['Test'] == 'Paired t-test')].iloc[0]
        
        report_lines.append("Paired t-test results:")
        report_lines.append(f"  Combined vs Radiomics:")
        report_lines.append(f"    P-value: {ttest_rad['P_Value']:.4f}")
        report_lines.append(f"    Significant: {ttest_rad['Significant']}")
        report_lines.append(f"  Combined vs Spatial Radiomics:")
        report_lines.append(f"    P-value: {ttest_spatial['P_Value']:.4f}")
        report_lines.append(f"    Significant: {ttest_spatial['Significant']}")
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
            report_lines.append(f"  {row['Category']:25s}: {int(row['Count']):3d} ({row['Percentage']:5.1f}%)")
        report_lines.append("")
    
    # Section 7: Key Findings
    report_lines.append("="*80)
    report_lines.append("7. KEY FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Finding 1: Best overall performance
    overall_best = comparison_df.loc[comparison_df['Test_AUC'].idxmax()]
    feature_type_name = {
        'radiomics': 'Radiomics',
        'spatial_radiomics': 'Spatial Radiomics',
        'combined': 'Combined'
    }[overall_best['Feature_Type']]
    
    report_lines.append(f"1. Best Overall Model:")
    report_lines.append(f"   {overall_best['Model']} with {feature_type_name} features")
    report_lines.append(f"   achieved Test AUC of {overall_best['Test_AUC']:.4f}")
    report_lines.append("")
    
    # Finding 2: Synergy effect
    synergy = best_combined - max(best_radiomics, best_spatial)
    if synergy > 0.01:
        report_lines.append(f"2. Positive Synergy Detected:")
        report_lines.append(f"   Combining features improves performance by {synergy:.4f} AUC")
        report_lines.append(f"   ({synergy/max(best_radiomics, best_spatial)*100:.2f}% relative improvement)")
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
    spatial_std = comparison_df[comparison_df['Feature_Type'] == 'spatial_radiomics']['Test_AUC'].std()
    combined_std = comparison_df[comparison_df['Feature_Type'] == 'combined']['Test_AUC'].std()
    
    most_consistent = min([('Radiomics', radiomics_std), 
                          ('Spatial Radiomics', spatial_std), 
                          ('Combined', combined_std)], 
                         key=lambda x: x[1])
    
    report_lines.append(f"3. Model Consistency:")
    report_lines.append(f"   Most consistent feature type: {most_consistent[0]}")
    report_lines.append(f"   (Std Dev across models: {most_consistent[1]:.4f})")
    report_lines.append(f"   Radiomics: {radiomics_std:.4f}, Spatial: {spatial_std:.4f}, Combined: {combined_std:.4f}")
    report_lines.append("")
    
    # Finding 4: MLP performance
    mlp_radiomics = comparison_df[(comparison_df['Feature_Type'] == 'radiomics') & 
                                  (comparison_df['Model'] == 'MLP')]['Test_AUC'].values[0]
    mlp_spatial = comparison_df[(comparison_df['Feature_Type'] == 'spatial_radiomics') & 
                                (comparison_df['Model'] == 'MLP')]['Test_AUC'].values[0]
    mlp_combined = comparison_df[(comparison_df['Feature_Type'] == 'combined') & 
                                 (comparison_df['Model'] == 'MLP')]['Test_AUC'].values[0]
    
    report_lines.append(f"4. Deep Learning (MLP) Performance:")
    report_lines.append(f"   Radiomics:        {mlp_radiomics:.4f}")
    report_lines.append(f"   Spatial Radiomics: {mlp_spatial:.4f}")
    report_lines.append(f"   Combined:         {mlp_combined:.4f}")
    
    best_mlp = max(mlp_radiomics, mlp_spatial, mlp_combined)
    best_overall_auc = comparison_df['Test_AUC'].max()
    if best_mlp >= best_overall_auc - 0.01:
        report_lines.append(f"   → MLP achieves competitive performance with traditional ML")
    else:
        report_lines.append(f"   → Traditional ML outperforms MLP by {best_overall_auc - best_mlp:.4f} AUC")
    report_lines.append("")
    
    # Section 8: Recommendations
    report_lines.append("="*80)
    report_lines.append("8. RECOMMENDATIONS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if best_combined > max(best_radiomics, best_spatial) and improvement_rad > 0.01:
        report_lines.append("✓ RECOMMENDATION: Use COMBINED features")
        report_lines.append(f"  Rationale: Combined features provide the best performance")
        report_lines.append(f"  with {synergy:.4f} AUC improvement over single feature types.")
        report_lines.append(f"  Both feature types contribute complementary information.")
    elif best_radiomics > best_spatial and best_radiomics > best_combined:
        report_lines.append("✓ RECOMMENDATION: Use RADIOMICS features only")
        report_lines.append(f"  Rationale: Radiomics features alone provide the best performance.")
        report_lines.append(f"  Combining with spatial radiomics does not improve results.")
    elif best_spatial > best_radiomics and best_spatial > best_combined:
        report_lines.append("✓ RECOMMENDATION: Use SPATIAL RADIOMICS features only")
        report_lines.append(f"  Rationale: Spatial radiomics features alone provide the best performance.")
        report_lines.append(f"  Combining with traditional radiomics does not improve results.")
    else:
        report_lines.append("○ RECOMMENDATION: Further investigation needed")
        report_lines.append(f"  Rationale: Performance differences are minimal.")
        report_lines.append(f"  Consider computational cost and interpretability.")
    
    report_lines.append("")
    
    # Section 9: Clinical Implications
    report_lines.append("="*80)
    report_lines.append("9. CLINICAL IMPLICATIONS")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("Based on the analysis:")
    report_lines.append("")
    
    # Performance interpretation
    if best_combined >= 0.80:
        report_lines.append("✓ Strong Predictive Performance (AUC ≥ 0.80)")
        report_lines.append("  The model shows excellent discrimination for Response expression prediction.")
    elif best_combined >= 0.70:
        report_lines.append("○ Moderate Predictive Performance (0.70 ≤ AUC < 0.80)")
        report_lines.append("  The model shows acceptable discrimination but may benefit from refinement.")
    else:
        report_lines.append("✗ Limited Predictive Performance (AUC < 0.70)")
        report_lines.append("  Additional features or alternative approaches should be considered.")
    report_lines.append("")
    
    # Feature type implications
    if best_spatial > best_radiomics:
        report_lines.append("• Spatial Radiomics features outperform traditional radiomics:")
        report_lines.append("  Suggests that spatial relationships in tumor microenvironment")
        report_lines.append("  provide additional predictive value for Response expression.")
    else:
        report_lines.append("• Traditional Radiomics features perform strongly:")
        report_lines.append("  Standard texture and intensity features capture relevant")
        report_lines.append("  tumor characteristics for Response prediction.")
    report_lines.append("")
    
    # SMOTE implications
    if comparison_df['SMOTE_Applied'].iloc[0]:
        report_lines.append("• Class Balancing with SMOTE:")
        report_lines.append("  Applied to address class imbalance in the training set.")
        report_lines.append("  Results reflect balanced learning across Response expression levels.")
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


def create_model_comparison_plot(results_dir='multimodal_results'):
    """
    Create a comprehensive model comparison visualization
    """
    print("\nCreating model comparison plot...")
    
    results_path = Path(results_dir)
    comparison_files = list(results_path.glob('all_features_comparison_*.csv'))
    if not comparison_files:
        print("Error: No comparison files found")
        return
    
    comparison_df = pd.read_csv(comparison_files[0])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Test AUC comparison for all models and feature types
    ax1 = fig.add_subplot(gs[0, :2])
    
    feature_types = ['radiomics', 'spatial_radiomics', 'combined']
    feature_labels = ['Radiomics', 'Spatial Radiomics', 'Combined']
    models = comparison_df['Model'].unique()
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, (ft, label) in enumerate(zip(feature_types, feature_labels)):
        aucs = [comparison_df[(comparison_df['Feature_Type'] == ft) & 
                             (comparison_df['Model'] == model)]['Test_AUC'].values[0] 
                for model in models]
        ax1.bar(x + i*width, aucs, width, label=label, alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Test AUC Comparison: All Models and Feature Types', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.5, 1.0])
    
    # Plot 2: CV AUC with error bars
    ax2 = fig.add_subplot(gs[0, 2])
    
    for i, (ft, label) in enumerate(zip(feature_types, feature_labels)):
        cv_means = [comparison_df[(comparison_df['Feature_Type'] == ft) & 
                                  (comparison_df['Model'] == model)]['CV_AUC'].values[0] 
                   for model in models]
        cv_stds = [comparison_df[(comparison_df['Feature_Type'] == ft) & 
                                 (comparison_df['Model'] == model)]['CV_AUC_Std'].values[0] 
                  for model in models]
        ax2.errorbar(x + i*width, cv_means, yerr=cv_stds, 
                    fmt='o', capsize=5, label=label, alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CV AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Cross-Validation AUC\n(with std dev)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.5, 1.0])
    
    # Plot 3: Accuracy comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, (ft, label) in enumerate(zip(feature_types, feature_labels)):
        accs = [comparison_df[(comparison_df['Feature_Type'] == ft) & 
                             (comparison_df['Model'] == model)]['Test_Accuracy'].values[0] 
                for model in models]
        ax3.bar(x + i*width, accs, width, label=label, alpha=0.8)
    
    ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0.5, 1.0])
    
    # Plot 4: F1 Score comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, (ft, label) in enumerate(zip(feature_types, feature_labels)):
        f1s = [comparison_df[(comparison_df['Feature_Type'] == ft) & 
                            (comparison_df['Model'] == model)]['Test_F1'].values[0] 
               for model in models]
        ax4.bar(x + i*width, f1s, width, label=label, alpha=0.8)
    
    ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Test F1 Score', fontsize=11, fontweight='bold')
    ax4.set_title('Test F1 Score Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0.5, 1.0])
    
    # Plot 5: Best performance by feature type
    ax5 = fig.add_subplot(gs[1, 2])
    
    best_aucs = [comparison_df[comparison_df['Feature_Type'] == ft]['Test_AUC'].max() 
                 for ft in feature_types]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax5.bar(feature_labels, best_aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax5.set_ylabel('Best Test AUC', fontsize=11, fontweight='bold')
    ax5.set_title('Best Model Performance\nby Feature Type', fontsize=13, fontweight='bold')
    ax5.set_ylim([0.5, 1.0])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Plot 6: Performance ranking
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create ranking dataframe
    ranking_data = []
    for _, row in comparison_df.iterrows():
        ranking_data.append({
            'Model_Feature': f"{row['Model']}\n({row['Feature_Type'].replace('_', ' ').title()})",
            'Test_AUC': row['Test_AUC'],
            'Feature_Type': row['Feature_Type']
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values('Test_AUC', ascending=True)
    
    # Color by feature type
    color_map = {'radiomics': '#1f77b4', 'spatial_radiomics': '#ff7f0e', 'combined': '#2ca02c'}
    colors = [color_map[ft] for ft in ranking_df['Feature_Type']]
    
    y_pos = np.arange(len(ranking_df))
    ax6.barh(y_pos, ranking_df['Test_AUC'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(ranking_df['Model_Feature'], fontsize=9)
    ax6.set_xlabel('Test AUC', fontsize=12, fontweight='bold')
    ax6.set_title('Performance Ranking: All Model-Feature Combinations', 
                  fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.set_xlim([0.5, 1.0])
    
    # Add value labels
    for i, (idx, row) in enumerate(ranking_df.iterrows()):
        ax6.text(row['Test_AUC'], i, f" {row['Test_AUC']:.3f}", 
                va='center', fontsize=8, fontweight='bold')
    
    # Add legend for feature types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='Radiomics'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='Spatial Radiomics'),
        Patch(facecolor='#2ca02c', alpha=0.7, label='Combined')
    ]
    ax6.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.savefig(results_path / 'comprehensive_model_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Comprehensive model comparison plot saved")

def create_roc_analysis_report(all_results, y_test, results_dir='multimodal_results'):
    """
    Create detailed report on ROC curve analysis
    """
    print("\nCreating ROC curve analysis report...")
    
    results_path = Path(results_dir)
    
    # Find best classifier for combined model
    best_auc_combined = 0
    best_classifier_name = None
    
    for model_name, result in all_results['combined'].items():
        current_auc = result['test_metrics']['AUC']
        if current_auc > best_auc_combined:
            best_auc_combined = current_auc
            best_classifier_name = model_name
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ROC CURVE ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Purpose: Fair comparison using the same classifier across feature types")
    report_lines.append("")
    
    # Section 1: Classifier selection
    report_lines.append("="*80)
    report_lines.append("1. CLASSIFIER SELECTION FOR COMPARISON")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Selected Classifier: {best_classifier_name}")
    report_lines.append(f"Rationale: Best performing on Combined features (AUC = {best_auc_combined:.4f})")
    report_lines.append("")
    
    # Section 2: Performance with selected classifier
    report_lines.append("="*80)
    report_lines.append(f"2. PERFORMANCE OF {best_classifier_name.upper()} ACROSS FEATURE TYPES")
    report_lines.append("="*80)
    report_lines.append("")
    
    for ft in ['radiomics', 'spatial_radiomics', 'combined']:
        result = all_results[ft][best_classifier_name]
        
        ft_name = {
            'radiomics': 'Radiomics',
            'spatial_radiomics': 'Spatial Radiomics',
            'combined': 'Combined'
        }[ft]
        
        # Calculate additional metrics from ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, result['y_prob_test'])
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sens = tpr[optimal_idx]
        optimal_spec = 1 - fpr[optimal_idx]
        
        report_lines.append(f"{ft_name} Features:")
        report_lines.append(f"  AUC: {result['test_metrics']['AUC']:.4f}")
        report_lines.append(f"  Sensitivity: {result['test_metrics']['Sensitivity']:.4f}")
        report_lines.append(f"  Specificity: {result['test_metrics']['Specificity']:.4f}")
        report_lines.append(f"  Optimal Threshold: {optimal_threshold:.4f}")
        report_lines.append(f"    - Sensitivity at optimal: {optimal_sens:.4f}")
        report_lines.append(f"    - Specificity at optimal: {optimal_spec:.4f}")
        report_lines.append(f"    - Youden's J: {j_scores[optimal_idx]:.4f}")
        report_lines.append("")
    
    # Section 3: Comparison analysis
    report_lines.append("="*80)
    report_lines.append("3. COMPARATIVE ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    
    radiomics_result = all_results['radiomics'][best_classifier_name]
    spatial_result = all_results['spatial_radiomics'][best_classifier_name]
    combined_result = all_results['combined'][best_classifier_name]
    
    rad_auc = radiomics_result['test_metrics']['AUC']
    spatial_auc = spatial_result['test_metrics']['AUC']
    combined_auc = combined_result['test_metrics']['AUC']
    
    report_lines.append("AUC Differences:")
    report_lines.append(f"  Combined vs Radiomics: {combined_auc - rad_auc:+.4f}")
    report_lines.append(f"  Combined vs Spatial:   {combined_auc - spatial_auc:+.4f}")
    report_lines.append(f"  Spatial vs Radiomics:  {spatial_auc - rad_auc:+.4f}")
    report_lines.append("")
    
    if combined_auc > max(rad_auc, spatial_auc):
        improvement = combined_auc - max(rad_auc, spatial_auc)
        report_lines.append(f"✓ Combined features show improvement: +{improvement:.4f} AUC")
        report_lines.append(f"  Relative improvement: {improvement/max(rad_auc, spatial_auc)*100:.2f}%")
    else:
        best_single = 'Radiomics' if rad_auc > spatial_auc else 'Spatial Radiomics'
        report_lines.append(f"○ {best_single} features perform best")
        report_lines.append(f"  Combined does not improve over single feature type")
    
    report_lines.append("")
    
    # Section 4: All classifiers comparison
    report_lines.append("="*80)
    report_lines.append("4. ALL CLASSIFIERS PERFORMANCE (COMBINED FEATURES)")
    report_lines.append("="*80)
    report_lines.append("")
    
    classifier_aucs = []
    for model_name, result in all_results['combined'].items():
        auc = result['test_metrics']['AUC']
        classifier_aucs.append((model_name, auc))
    
    # Sort by AUC
    classifier_aucs.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, auc) in enumerate(classifier_aucs, 1):
        marker = "★" if model_name == best_classifier_name else " "
        report_lines.append(f"  {rank}. {marker} {model_name:25s} AUC = {auc:.4f}")
    
    report_lines.append("")
    
    # Section 5: Recommendations
    report_lines.append("="*80)
    report_lines.append("5. RECOMMENDATIONS FOR ROC CURVE REPORTING")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("Main Figure:")
    report_lines.append(f"  • Use roc_curves_comparison.png")
    report_lines.append(f"  • Shows {best_classifier_name} across all feature types")
    report_lines.append(f"  • Fair comparison (same classifier, different features)")
    report_lines.append("")
    
    report_lines.append("Supplementary Figures:")
    report_lines.append("  • roc_curves_comprehensive.png")
    report_lines.append("    - 4-panel view of all combinations")
    report_lines.append("  • roc_curves_by_classifier.png")
    report_lines.append("    - Individual classifier comparisons")
    report_lines.append("")
    
    report_lines.append("Caption Suggestions:")
    report_lines.append(f"  'ROC curves comparing feature types using {best_classifier_name}.")
    report_lines.append(f"   Combined features achieved AUC of {combined_auc:.3f}, compared to")
    report_lines.append(f"   {rad_auc:.3f} for Radiomics and {spatial_auc:.3f} for Spatial Radiomics.'")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Save report
    report_text = "\n".join(report_lines)
    with open(results_path / 'roc_curve_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nROC curve analysis report saved to: {results_path / 'roc_curve_analysis_report.txt'}")
    
    return best_classifier_name
# ============================================================
# COMPLETE INTEGRATED WORKFLOW
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("COMPLETE MULTI-MODAL FEATURE ANALYSIS WORKFLOW")
    print("WITH RFE FEATURE SELECTION, SMOTE, AND MLP CLASSIFIER")
    print("="*80)
    print("\nCOMBINED MODEL STRATEGY:")
    print("  • Select top features from Radiomics model using RFE")
    print("  • Select top features from Spatial Radiomics model using RFE")
    print("  • Concatenate both selected feature sets for Combined model")
    print("  • This preserves the best features from each domain")
    print("="*80 + "\n")
    
    # Step 1: Run main comparison with RFE and SMOTE
    print("STEP 1: Running main comparison...")
    predictor = MultiModalPredictor(
        radiomics_csv='XXXyour path/Dataset/Dataset002_ECResponse/radiomics_features.csv',
        spatial_radiomics_csv='XXXyour path/Dataset/Dataset002_ECResponse/sparad_features.csv',
        clinical_xlsx='XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx',
        output_dir='XXXyour path/Dataset/Dataset002_ECResponse/multimodal_results'
    )
    
    all_results, combined_comparison = predictor.run_complete_comparison(
        test_size=0.3,
        n_features=21,
        feature_selection_method='rfe',  # Using RFE univariate rfe tree_based mutual_info
        random_state=42,
        use_smote=True  # Using SMOTE
    )
    
    # Step 2: Save predictions (with corrected method)
    print("\nSTEP 2: Saving predictions...")
    y_train_original = predictor.y[predictor.train_indices]
    y_test = predictor.y[predictor.test_indices]
    
    predictor.save_all_predictions(
        all_results,
        y_train_original,
        y_test
    )
    
    # Step 2.5: Save trained models and selectors
    print("\nSTEP 2.5: Saving trained models and selectors...")
    predictor.save_models_and_selectors(all_results, 'rfe')
    
    # Step 3: Analyze synergy
    print("\nSTEP 3: Analyzing feature synergy...")
    synergy_df = compare_feature_synergy(combined_comparison)
    synergy_df.to_csv(predictor.output_dir / 'feature_synergy_analysis.csv', index=False)
    
    # Step 4: Statistical tests
    print("\nSTEP 4: Performing statistical tests...")
    try:
        statistical_test_feature_combination('multimodal_results')
    except Exception as e:
        print(f"  Warning: Statistical tests skipped due to error: {e}")
    
    # Step 5: Publication-ready table
    print("\nSTEP 5: Creating publication-ready table...")
    try:
        pub_table = create_publication_ready_table('multimodal_results')
    except Exception as e:
        print(f"  Warning: Publication table skipped due to error: {e}")
    
    # Step 6: Patient-level agreement
    print("\nSTEP 6: Analyzing patient-level agreement...")
    try:
        agreement_stats = visualize_patient_level_agreement('multimodal_results')
    except Exception as e:
        print(f"  Warning: Patient-level agreement skipped due to error: {e}")
    
    # Step 7: Complementary cases analysis
    print("\nSTEP 7: Identifying complementary cases...")
    try:
        comp_analysis, interesting_cases = identify_complementary_cases('multimodal_results')
    except Exception as e:
        print(f"  Warning: Complementary cases analysis skipped due to error: {e}")
    
    # Step 8: Comprehensive visualizations
    print("\nSTEP 8: Creating comprehensive visualizations...")
    try:
        create_model_comparison_plot('multimodal_results')
        
        # Additional ROC curve analyses
        print("\nCreating additional ROC curve analyses...")
        plot_comprehensive_roc_comparison(all_results, y_test, 'multimodal_results')
        create_classifier_specific_roc_comparison(all_results, y_test, 'multimodal_results')
        
    except Exception as e:
        print(f"  Warning: Model comparison plot skipped due to error: {e}")
        import traceback
        traceback.print_exc()

    # Step 8.6: ROC curve analysis report
    print("\nSTEP 8.6: Creating ROC curve analysis report...")
    try:
        best_classifier_for_roc = create_roc_analysis_report(all_results, y_test, 'multimodal_results')
    except Exception as e:
        print(f"  Warning: ROC analysis report encountered error: {e}")

    # Step 9: Final comprehensive report
    print("\nSTEP 9: Creating final comprehensive report...")
    try:
        final_report = create_final_report('multimodal_results')
    except Exception as e:
        print(f"  Warning: Final report creation encountered error: {e}")
    
    # Step 10: Summary of all outputs
    print("\n" + "="*80)
    print("GENERATED FILES - ROC CURVE ANALYSIS:")
    print("="*80)
    
    print("\nROC Curve Visualizations:")
    print("  • roc_curves_comparison.png")
    print("    → Main figure: Same classifier across all feature types")
    print(f"    → Uses: {predictor.best_classifier_for_roc if hasattr(predictor, 'best_classifier_for_roc') else 'Best classifier'}")
    print("    → Fair comparison for feature type evaluation")
    print("")
    print("  • roc_curves_comprehensive.png")
    print("    → 4-panel comprehensive view")
    print("    → Panel 1: Same classifier across features")
    print("    → Panel 2-4: All models for each feature type")
    print("")
    print("  • roc_curves_by_classifier.png")
    print("    → Separate panel for each classifier")
    print("    → Comparing feature types within each classifier")
    print("")
    
    print("\nROC Curve Analysis:")
    print("  • roc_curve_analysis_report.txt")
    print("    → Detailed analysis of ROC curve comparisons")
    print("    → Optimal thresholds for each feature type")
    print("    → Youden's J statistic calculations")
    print("    → Classifier selection rationale")
    
    print("\n" + "="*80)
    print("ROC CURVE INTERPRETATION GUIDE:")
    print("="*80)
    
    print("\nWhy use the same classifier for comparison?")
    print("  • Ensures fair comparison across feature types")
    print("  • Isolates the effect of features (not classifier)")
    print("  • Standard practice in feature comparison studies")
    print("  • Selected: Best performing classifier on combined features")
    
    print("\nHow to interpret the ROC curves:")
    print("  1. Higher curve = Better performance")
    print("  2. AUC closer to 1.0 = Better discrimination")
    print("  3. Curve closer to top-left corner = Optimal")
    print("  4. Diagonal line = Random classifier (AUC = 0.5)")
    
    print("\nOptimal Operating Point:")
    print("  • Youden's J = Sensitivity + Specificity - 1")
    print("  • Maximizes J statistic to find optimal threshold")
    print("  • Balance between sensitivity and specificity")
    print("  • May adjust based on clinical requirements")
    
    print("\n" + "="*80)
    print("COMPLETE SUMMARY OF ALL OUTPUTS:")
    print("="*80)
    
    print("\n📊 PERFORMANCE METRICS FILES:")
    print("  1. all_features_comparison_rfe_smote_True.csv")
    print("     → All models, all metrics (Sens, Spec, Acc, PPV, NPV, F1, AUC)")
    print("  2. detailed_metrics_table.csv / .txt")
    print("     → Formatted comprehensive metrics table")
    print("  3. best_models_metrics_comparison.csv / .tex / .txt")
    print("     → Best model from each feature type")
    print("  4. best_model_metrics_summary.txt")
    print("     → Detailed narrative summary with clinical interpretation")
    
    print("\n📈 VISUALIZATION FILES (Main Figures):")
    print("  5. comprehensive_performance_summary.png")
    print("     → Ultimate summary: 9-panel comprehensive view")
    print("  6. comprehensive_metrics_comparison.png")
    print("     → 7-panel: Sens, Spec, Acc, PPV, NPV, F1 + Radar + Heatmap")
    print("  7. roc_curves_comparison.png ★")
    print("     → Main ROC figure: Same classifier, all feature types")
    print("  8. roc_curves_comprehensive.png")
    print("     → 4-panel ROC analysis")
    print("  9. confusion_matrices_best_models.png")
    print("     → Confusion matrices for best models")
    
    print("\n📊 ADDITIONAL VISUALIZATIONS:")
    print("  10. metrics_by_feature_type.png")
    print("      → Average and best metrics by feature type")
    print("  11. sensitivity_specificity_tradeoff.png")
    print("      → Sens vs Spec scatter plots")
    print("  12. performance_comparison_all_features.png")
    print("      → 4-panel basic comparison")
    print("  13. performance_heatmap.png")
    print("      → AUC and Accuracy heatmaps")
    print("  14. feature_type_contributions.png")
    print("      → 4-panel feature importance analysis")
    print("  15. roc_curves_by_classifier.png")
    print("      → Individual classifier ROC comparisons")
    print("  16. comprehensive_model_comparison.png")
    print("      → Complete model ranking visualization")
    
    print("\n📝 REPORTS AND SUMMARIES:")
    print("  17. FINAL_COMPREHENSIVE_REPORT.txt")
    print("      → Complete analysis narrative")
    print("  18. summary_report.txt")
    print("      → Quick summary of key findings")
    print("  19. feature_selection_summary.txt")
    print("      → Selected features documentation")
    print("  20. roc_curve_analysis_report.txt ★")
    print("      → Detailed ROC analysis and recommendations")
    
    print("\n📋 PUBLICATION-READY EXPORTS:")
    print("  21. publication_metrics_complete.csv / .xlsx / .tex")
    print("      → All metrics formatted for publication")
    print("  22. publication_best_models_only.csv / .tex")
    print("      → Best models table for main manuscript")
    print("  23. publication_ready_table.csv / .tex")
    print("      → Alternative publication format")
    
    print("\n🔬 FEATURE ANALYSIS FILES:")
    print("  24. feature_scores_radiomics_rfe.csv")
    print("      → 20 selected radiomics features with rankings")
    print("  25. feature_scores_spatial_radiomics_rfe.csv")
    print("      → 20 selected spatial radiomics features")
    print("  26. feature_list_combined_rfe.csv")
    print("      → 40 combined features with source attribution")
    print("  27. feature_type_contributions.csv")
    print("      → Importance contribution by feature type")
    print("  28. feature_importance_combined_detailed.csv")
    print("      → All features ranked with sources")
    print("  29. top_features_by_type.csv")
    print("      → Top 10 features from each type")
    
    print("\n🔢 STATISTICAL ANALYSIS:")
    print("  30. statistical_significance_tests.csv")
    print("      → t-test and Wilcoxon test results")
    print("  31. effect_size_analysis.csv")
    print("      → Cohen's d effect sizes")
    print("  32. feature_synergy_analysis.csv")
    print("      → Synergy metrics between feature types")
    print("  33. prediction_agreement_statistics.csv")
    print("      → Agreement metrics across feature types")
    
    print("\n🎯 CASE-LEVEL ANALYSIS:")
    print("  34. complementary_cases_analysis.csv")
    print("      → Case categories (beneficial/detrimental/neutral)")
    print("  35. interesting_cases_detailed.csv")
    print("      → Detailed information on edge cases")
    print("  36. predictions_unified_all_models.csv")
    print("      → All predictions in unified format")
    print("  37. predictions_[feature_type]_[model].csv (15 files)")
    print("      → Individual prediction files for each model")
    
    print("\n💾 REPRODUCIBILITY FILES:")
    print("  38. train_test_split.csv / .pkl")
    print("      → Exact train/test split for reproducibility")
    print("  39. trained_models/ directory")
    print("      → All trained models saved (.pkl files)")
    print("      → Feature scalers (3 files)")
    print("      → Can be loaded for deployment")
    
    print("\n" + "="*80)
    print("RECOMMENDED MANUSCRIPT STRUCTURE:")
    print("="*80)
    
    print("\n📄 MAIN MANUSCRIPT:")
    print("\nFigure 1: Study Design")
    print("  • Create flowchart showing patient selection")
    print("  • Include feature extraction pipeline")
    
    print("\nFigure 2: Feature Selection and Model Strategy")
    print("  • Illustrate RFE process")
    print("  • Show concatenation strategy for combined model")
    
    if hasattr(predictor, 'best_classifier_for_roc'):
        print(f"\nFigure 3: ROC Curve Comparison (roc_curves_comparison.png) ★")
        print(f"  • Shows {predictor.best_classifier_for_roc} across all feature types")
        print(f"  • Fair comparison (same classifier)")
        print(f"  • Caption: 'Receiver operating characteristic curves comparing")
        print(f"    feature types using {predictor.best_classifier_for_roc}.")
    else:
        print("\nFigure 3: ROC Curve Comparison (roc_curves_comparison.png)")
        print("  • Shows best classifier across all feature types")
    
    print("\nFigure 4: Comprehensive Performance Metrics")
    print("  • Use: comprehensive_metrics_comparison.png")
    print("  • Shows: Sens, Spec, Acc, PPV, NPV with radar chart")
    
    print("\nFigure 5: Feature Importance Analysis")
    print("  • Use: feature_type_contributions.png")
    print("  • Shows contribution of each feature type in combined model")
    
    print("\nTable 1: Patient Characteristics")
    print("  • Create from clinical data")
    print("  • Include demographics and imaging parameters")
    
    print("\nTable 2: Model Performance Comparison")
    print("  • Use: best_models_metrics_comparison.csv")
    print("  • All metrics for best model from each feature type")
    print("  • Format suggestion:")
    print("    | Feature Type | Model | Sens | Spec | Acc | PPV | NPV | F1 | AUC |")
    
    print("\n📑 SUPPLEMENTARY MATERIALS:")
    print("\nSupplementary Figure 1: Complete Performance Overview")
    print("  • comprehensive_performance_summary.png")
    
    print("\nSupplementary Figure 2: All Classifiers ROC Curves")
    print("  • roc_curves_comprehensive.png")
    
    print("\nSupplementary Figure 3: Classifier-Specific Comparisons")
    print("  • roc_curves_by_classifier.png")
    
    print("\nSupplementary Figure 4: Confusion Matrices")
    print("  • confusion_matrices_best_models.png")
    
    print("\nSupplementary Figure 5: Sensitivity-Specificity Tradeoff")
    print("  • sensitivity_specificity_tradeoff.png")
    
    print("\nSupplementary Figure 6: Patient-Level Agreement")
    print("  • patient_level_agreement.png")
    
    print("\nSupplementary Table 1: Complete Performance Metrics")
    print("  • publication_metrics_complete.csv")
    print("  • All models, all metrics")
    
    print("\nSupplementary Table 2: Selected Features")
    print("  • Combine all feature_scores_*.csv files")
    print("  • Show RFE rankings and scores")
    
    print("\nSupplementary Table 3: Statistical Tests")
    print("  • statistical_significance_tests.csv")
    print("  • effect_size_analysis.csv")
    
    print("\nSupplementary Table 4: Feature Importance")
    print("  • feature_importance_combined_detailed.csv")
    
    print("\n" + "="*80)
    print("KEY FINDINGS TO REPORT:")
    print("="*80)
    
#    # Calculate key findings
#    best_overall_idx = combined_comparison['Test_AUC'].idxmax()
#    best_overall = combined_comparison.loc[best_overall_idx]
#    
#    print(f"\n1. OVERALL BEST PERFORMANCE:")
#    print(f"   '{best_overall['Model']} using {best_overall['Feature_Type'].replace('_', ' ').title()} features")
#    print(f"    achieved an AUC of {best_overall['Test_AUC']:.3f} (95% CI: [calculate]),")
#    print(f"    with sensitivity of {best_overall['Test_Sensitivity']:.3f} and")
#    print(f"    specificity of {best_overall['Test_Specificity']:.3f}.'")
#    
#    print(f"\n2. FEATURE TYPE COMPARISON:")
#    if best_combined > max(best_radiomics, best_spatial):
#        improvement = best_combined - max(best_radiomics, best_spatial)
#        print(f"   'Combined features outperformed individual feature types,")
#        print(f"    showing {improvement:.3f} AUC improvement ({improvement/max(best_radiomics, best_spatial)*100:.1f}%")
#        print(f"    relative improvement) over the best single feature type.'")
#    else:
#        best_single = 'Radiomics' if best_radiomics > best_spatial else 'Spatial Radiomics'
#        print(f"   '{best_single} features performed best with AUC of")
#        print(f"    {max(best_radiomics, best_spatial):.3f}. Combined features did not")
#        print(f"    significantly improve performance (AUC: {best_combined:.3f}).'")
#    
#    print(f"\n3. CLINICAL UTILITY:")
#    sens = best_overall['Test_Sensitivity']
#    spec = best_overall['Test_Specificity']
#    ppv = best_overall['Test_PPV']
#    npv = best_overall['Test_NPV']
#    
#    print(f"   'The model correctly identified {int(sens*100)}% of Response positive cases")
#    print(f"    and {int(spec*100)}% of negative cases. When predicting positive Response")
#    print(f"    expression, the model was correct {int(ppv*100)}% of the time (PPV),")
#    print(f"    while negative predictions had {int(npv*100)}% reliability (NPV).'")
#    
#    print(f"\n4. FEATURE CONTRIBUTION:")
#    if (predictor.output_dir / 'feature_type_contributions.csv').exists():
#        contrib_df = pd.read_csv(predictor.output_dir / 'feature_type_contributions.csv')
#        rad_pct = contrib_df[contrib_df['Feature_Type'] == 'Radiomics']['Percentage'].values[0]
#        spatial_pct = contrib_df[contrib_df['Feature_Type'] == 'Spatial_Radiomics']['Percentage'].values[0]
#        
#        print(f"   'In the combined model, radiomics features contributed {rad_pct:.1f}%")
#        print(f"    and spatial radiomics features contributed {spatial_pct:.1f}% to the")
#        print(f"    overall feature importance (Random Forest analysis).'")
#    
#    print(f"\n5. SMOTE IMPACT:")
#    print(f"   'SMOTE was applied to address class imbalance in the training set")
#    print(f"    (original distribution: {np.bincount(y_train_original)}), improving")
#    print(f"    model performance on the minority class while maintaining")
#    print(f"    generalization on the original test distribution.'")
#    
#    print("\n" + "="*80)
#    print("FINAL CHECKLIST FOR MANUSCRIPT SUBMISSION:")
#    print("="*80)
#    
#    print("\n☐ Methods Section:")
#    print("  ☐ Describe patient cohort and selection criteria")
#    print("  ☐ Detail imaging protocols and parameters")
#    print("  ☐ Explain feature extraction (radiomics + spatial radiomics)")
#    print("  ☐ Document RFE feature selection process")
#    print("  ☐ Justify SMOTE usage for class balancing")
#    print("  ☐ Describe combined model concatenation strategy")
#    print("  ☐ List all classifiers with hyperparameters")
#    print("  ☐ Explain train/test split and cross-validation")
#    print("  ☐ Define all performance metrics")
#    
#    print("\n☐ Results Section:")
#    print("  ☐ Report patient characteristics (Table 1)")
#    print("  ☐ Show feature selection results")
#    print("  ☐ Present performance metrics (Table 2)")
#    print("  ☐ Include ROC curves (Figure 3)")
#    print("  ☐ Show comprehensive metrics (Figure 4)")
#    print("  ☐ Report feature importance (Figure 5)")
#    print("  ☐ Present confusion matrices")
#    print("  ☐ Describe statistical significance")
#    print("  ☐ Report optimal thresholds (Youden's J)")
#    
#    print("\n☐ Discussion Section:")
#    print("  ☐ Compare with published Response prediction studies")
#    print("  ☐ Discuss clinical implications of sensitivity/specificity")
#    print("  ☐ Interpret feature importance findings")
#    print("  ☐ Explain spatial radiomics contribution")
#    print("  ☐ Address SMOTE impact on generalizability")
#    print("  ☐ Discuss limitations (sample size, single-center, etc.)")
#    print("  ☐ Propose future validation strategies")
#    
#    print("\n☐ Figures and Tables:")
#    print("  ☐ All figures high resolution (300 DPI minimum)")
#    print("  ☐ Color schemes accessible (colorblind-friendly)")
#    print("  ☐ Font sizes readable when scaled")
#    print("  ☐ Captions complete with statistical details")
#    print("  ☐ Tables formatted per journal guidelines")
#    
#    print("\n☐ Supplementary Materials:")
#    print("  ☐ Complete performance metrics table")
#    print("  ☐ Selected features list with rankings")
#    print("  ☐ Statistical test results")
#    print("  ☐ Additional visualization panels")
#    print("  ☐ Code availability statement")
#    
#    print("\n☐ Data and Code Sharing:")
#    print("  ☐ De-identified data (if permitted)")
#    print("  ☐ Feature extraction code")
#    print("  ☐ Model training pipeline")
#    print("  ☐ Trained models (for reproducibility)")
#    print("  ☐ Requirements.txt or environment.yml")
#    print("  ☐ README with usage instructions")
#    
#    print("\n" + "="*80)
#    print("EXAMPLE RESULTS PARAGRAPH:")
#    print("="*80)
#    
#    print(f"""
#    A total of {n_train + n_test} patients were included in this study, with 
#    {n_train} ({n_train/(n_train+n_test)*100:.1f}%) allocated to the training set and 
#    {n_test} ({n_test/(n_train+n_test)*100:.1f}%) to the test set. The training set had 
#    Response expression distribution of {np.bincount(y_train_original)}, which was balanced 
#    using SMOTE before model training. 
#
#    Using recursive feature elimination (RFE), we selected 20 features from each domain 
#    (radiomics and spatial radiomics), resulting in 40 features for the combined model. 
#    Five machine learning classifiers were evaluated: Logistic Regression, Random Forest, 
#    Gradient Boosting, Support Vector Machine, and Multi-Layer Perceptron.
#
#    The best overall performance was achieved by {best_overall['Model']} using 
#    {best_overall['Feature_Type'].replace('_', ' ').title()} features, with a test AUC of 
#    {best_overall['Test_AUC']:.3f}, sensitivity of {best_overall['Test_Sensitivity']:.3f}, 
#    specificity of {best_overall['Test_Specificity']:.3f}, and accuracy of 
#    {best_overall['Test_Accuracy']:.3f}. The positive predictive value was 
#    {best_overall['Test_PPV']:.3f} and negative predictive value was {best_overall['Test_NPV']:.3f}.
#
#    Comparing feature types using the same classifier ({predictor.best_classifier_for_roc if hasattr(predictor, 'best_classifier_for_roc') else 'best classifier'}), 
#    radiomics features achieved AUC of {best_radiomics:.3f}, spatial radiomics features 
#    achieved {best_spatial:.3f}, and combined features achieved {best_combined:.3f}. 
#    """)
#        
#    if best_combined > max(best_radiomics, best_spatial):
#        improvement = best_combined - max(best_radiomics, best_spatial)
#        print(f"""The combined model showed significant improvement over individual feature types 
#(+{improvement:.3f} AUC, p < 0.05), demonstrating complementary information from 
#both radiomics and spatial radiomics features.""")
#    else:
#        best_single = 'Radiomics' if best_radiomics > best_spatial else 'Spatial Radiomics'
#        print(f"""{best_single} features alone achieved the best performance, suggesting that 
#combining feature types did not provide additional predictive value in this cohort.""")
#    
#    if (predictor.output_dir / 'feature_type_contributions.csv').exists():
#        contrib_df = pd.read_csv(predictor.output_dir / 'feature_type_contributions.csv')
#        rad_pct = contrib_df[contrib_df['Feature_Type'] == 'Radiomics']['Percentage'].values[0]
#        spatial_pct = contrib_df[contrib_df['Feature_Type'] == 'Spatial_Radiomics']['Percentage'].values[0]
#        
#        print(f"""
#    Feature importance analysis revealed that radiomics features contributed {rad_pct:.1f}% 
#    and spatial radiomics features contributed {spatial_pct:.1f}% to the Random Forest 
#    model's predictive power, indicating {"balanced" if abs(rad_pct - 50) < 10 else "unbalanced"} 
#    contribution from both domains.""")
#        
#        print("\n" + "="*80)
#        print("EXAMPLE METHODS PARAGRAPH:")
#        print("="*80)
#        
#        print(f"""
#    FEATURE EXTRACTION AND SELECTION:
#    Radiomics features were extracted from segmented tumor regions using PyRadiomics 
#    (version X.X), including first-order statistics, shape features, and texture features 
#    (GLCM, GLRLM, GLSZM, GLDM, NGTDM). Spatial radiomics features were computed to capture 
#    tumor microenvironment patterns and spatial relationships within the tumor region.
#
#    Feature selection was performed separately for each domain using Recursive Feature 
#    Elimination (RFE) with logistic regression as the base estimator. We selected the top 
#    20 features from radiomics and top 20 from spatial radiomics. For the combined model, 
#    we concatenated both selected feature sets, resulting in 40 features total. This 
#    approach ensures equal representation from both domains while preserving domain-specific 
#    feature selection.
#
#    All features were standardized using Z-score normalization fitted on the training set.
#
#    CLASSIFICATION AND CLASS BALANCING:
#    To address class imbalance in the training set ({np.bincount(y_train_original)[0]} negative 
#    vs {np.bincount(y_train_original)[1]} positive cases), we applied Synthetic Minority 
#    Over-sampling Technique (SMOTE) to create a balanced training set. SMOTE was applied 
#    only to the training data after feature selection and standardization. The test set 
#    remained in its original distribution to evaluate real-world performance.
#
#    We evaluated five classification algorithms: Logistic Regression (L2 regularization, 
#    C=1.0), Random Forest (100 trees, max_depth=None), Gradient Boosting (100 estimators, 
#    learning_rate=0.1), Support Vector Machine (RBF kernel, γ=auto), and Multi-Layer 
#    Perceptron (hidden layers: 100-50 neurons, ReLU activation, Adam optimizer with 
#    early stopping).
#
#    MODEL EVALUATION:
#    Model performance was assessed using 5-fold stratified cross-validation on the training 
#    set and independent evaluation on the held-out test set. We computed multiple metrics: 
#    Area Under the ROC Curve (AUC), sensitivity (recall), specificity, accuracy, positive 
#    predictive value (PPV/precision), negative predictive value (NPV), and F1 score.
#
#    For fair comparison across feature types, we used the same classifier (the best-performing 
#    model on combined features) when comparing ROC curves. Statistical significance was 
#    assessed using paired t-tests and Wilcoxon signed-rank tests, with Cohen's d for effect 
#    size estimation.
#
#    All analyses were performed using Python 3.x with scikit-learn, imbalanced-learn, and 
#    custom scripts. Random seed was fixed (seed=42) for reproducibility.
#    """)
#        
#        print("\n" + "="*80)
#        print("TIPS FOR PEER REVIEW RESPONSE:")
#        print("="*80)
#        
#        print("""
#    Common Reviewer Questions and Where to Find Answers:
#
#    Q1: "Why use SMOTE? How does it affect generalizability?"
#    A: See best_model_metrics_summary.txt and FINAL_COMPREHENSIVE_REPORT.txt
#    → SMOTE applied only to training; test set unchanged
#    → Compare metrics with/without SMOTE if needed
#
#    Q2: "How were features selected? Why RFE?"
#    A: See feature_selection_summary.txt and feature_scores_*.csv
#    → RFE provides interpretable rankings
#    → Separate selection preserves domain expertise
#
#    Q3: "Why concatenate instead of joint selection?"
#    A: See summary_report.txt
#    → Ensures balanced representation (20+20)
#    → Avoids bias toward more numerous features
#    → Preserves interpretability of sources
#
#    Q4: "Are the improvements statistically significant?"
#    A: See statistical_significance_tests.csv
#    → Paired t-tests and Wilcoxon tests provided
#    → Effect sizes (Cohen's d) calculated
#
#    Q5: "Which features are most important?"
#    A: See feature_importance_combined_detailed.csv
#    → Top features from each type identified
#    → Contribution percentages calculated
#
#    Q6: "How does performance vary across models?"
#    A: See all_features_comparison_*.csv
#    → All models evaluated consistently
#    → Prevents cherry-picking
#
#    Q7: "What about overfitting?"
#    A: Compare Train_AUC vs Test_AUC vs CV_AUC in comparison table
#    → Cross-validation results provided
#    → Train vs test differences documented
#
#    Q8: "Can these results be reproduced?"
#    A: YES! All materials provided:
#    → Trained models saved (trained_models/)
#    → Exact train/test split (train_test_split.pkl)
#    → Selected features documented
#    → Code available
#    → Random seeds fixed
#
#    Q9: "Why use this specific classifier for comparison?"
#    A: See roc_curve_analysis_report.txt
#    → Best performer on combined features
#    → Fair comparison (same algorithm, different features)
#    → All classifiers shown in supplementary
#
#    Q10: "What are the clinical implications?"
#    A: See best_model_metrics_summary.txt
#    → Sensitivity/specificity interpreted clinically
#    → PPV/NPV calculated
#    → Decision threshold optimization discussed
#    """)
#        
#        print("\n" + "="*80)
#        print("ADVANCED ANALYSIS SUGGESTIONS:")
#        print("="*80)
#        
#        print("""
#    If reviewers request additional analyses:
#
#    1. SUBGROUP ANALYSIS:
#    → Stratify by tumor stage, size, location
#    → Analyze performance in specific patient subgroups
#    → Use saved predictions (predictions_unified_all_models.csv)
#
#    2. CALIBRATION CURVES:
#    → Assess probability calibration
#    → Plot predicted vs observed probabilities
#    → Use saved probability predictions
#
#    3. DECISION CURVE ANALYSIS:
#    → Clinical utility assessment
#    → Net benefit across thresholds
#    → Compare with "treat all" / "treat none" strategies
#
#    4. FEATURE STABILITY ANALYSIS:
#    → Bootstrap feature importance
#    → Assess feature selection stability
#    → Run pipeline multiple times with different seeds
#
#    5. EXTERNAL VALIDATION:
#    → Use saved models (trained_models/)
#    → Apply to independent cohort
#    → Report performance on external data
#
#    6. THRESHOLD OPTIMIZATION:
#    → Optimize for specific clinical scenario
#    → Maximize sensitivity (screening)
#    → Maximize specificity (confirmation)
#    → Balance based on cost matrix
#
#    7. ENSEMBLE METHODS:
#    → Combine multiple models
#    → Weighted voting based on CV performance
#    → Meta-learner on top of base models
#
#    8. LEARNING CURVES:
#    → Plot performance vs training set size
#    → Assess if more data would help
#    → Estimate sample size for future studies
#
#    9. SHAP VALUES:
#    → Model interpretability
#    → Individual prediction explanations
#    → Feature interaction analysis
#
#    10. COMPARISON WITH CLINICAL VARIABLES:
#        → Add age, stage, histology
#        → Compare imaging vs clinical models
#        → Assess added value of imaging
#    """)
#        
#        print("\n" + "="*80)
#        print("WORKFLOW COMPLETE - ALL FILES GENERATED!")
#        print("="*80)
#        
#        print(f"""
#    ✓ Total Files Generated: ~50+ files
#    ✓ Visualizations: 16 publication-quality figures
#    ✓ Tables: 10+ CSV/Excel/LaTeX tables
#    ✓ Reports: 5 comprehensive text reports
#    ✓ Models: All trained models saved
#    ✓ Reproducibility: Complete pipeline documented
#
#    📍 Main Directory: {predictor.output_dir}
#
#    ⭐ START HERE:
#    1. Open: FINAL_COMPREHENSIVE_REPORT.txt
#    2. Review: comprehensive_performance_summary.png
#    3. Check: best_models_metrics_comparison.csv
#    4. Use: roc_curves_comparison.png for manuscript
#
#    🎯 Next Steps:
#    1. Review all visualizations
#    2. Select figures for manuscript
#    3. Prepare supplementary materials
#    4. External validation (if data available)
#    5. Manuscript writing
#
#    📧 For questions or issues:
#    → Check documentation in reports/
#    → Review feature_selection_summary.txt
#    → Consult roc_curve_analysis_report.txt
#
#    Thank you for using the Multi-Modal Feature Fusion Pipeline!
#    """)
#        
#        print("="*80 + "\n")