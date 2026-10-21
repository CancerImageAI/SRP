import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PDL1Predictor:
    def __init__(self, features_csv, clinical_xlsx, output_dir='results'):
        """
        Initialize PD-L1 expression predictor
        
        Args:
            features_csv: Path to radiomics features CSV
            clinical_xlsx: Path to clinical data with PD-L1 labels
            output_dir: Directory to save results
        """
        self.features_csv = features_csv
        self.clinical_xlsx = clinical_xlsx
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None
        self.selected_features = None
        
        # Store indices for train/test split
        self.train_indices = None
        self.test_indices = None
        
    def load_and_merge_data(self, case_id_col='CaseID', label_col='PD-L1'):
        """
        Load radiomics features and clinical data, then merge
        
        Args:
            case_id_col: Column name for case identifier
            label_col: Column name for PD-L1 expression label
            
        Returns:
            Merged dataframe with features and labels
        """
        # Load radiomics features
        print("Loading radiomics features...")
        radiomics_df = pd.read_csv(self.features_csv)
        print(f"Loaded {len(radiomics_df)} cases with {len(radiomics_df.columns)} features")
        
        # Load clinical data
        print("Loading clinical data...")
        clinical_df = pd.read_excel(self.clinical_xlsx)
        print(f"Loaded clinical data for {len(clinical_df)} cases")
        
        # Merge datasets
        merged_df = pd.merge(radiomics_df, clinical_df, on=case_id_col, how='inner')
        print(f"Merged dataset: {len(merged_df)} cases")
        
        # Check for missing values in label
        if merged_df[label_col].isnull().any():
            print(f"Warning: {merged_df[label_col].isnull().sum()} cases with missing labels")
            merged_df = merged_df.dropna(subset=[label_col])
        
        self.merged_df = merged_df
        self.case_id_col = case_id_col
        self.label_col = label_col
        
        # Print class distribution
        print(f"\nPD-L1 expression distribution:")
        print(merged_df[label_col].value_counts())
        print(f"Class balance: {merged_df[label_col].value_counts(normalize=True)}")
        
        return merged_df
    
    def prepare_features(self, exclude_cols=None):
        """
        Prepare feature matrix and target vector
        
        Args:
            exclude_cols: List of columns to exclude from features
            
        Returns:
            X (features), y (labels), feature_names
        """
        if exclude_cols is None:
            exclude_cols = [self.case_id_col, self.label_col]
        
        # Remove diagnostic features and metadata from PyRadiomics
        all_exclude_cols = exclude_cols.copy()
        
        # Exclude PyRadiomics diagnostic features
        diagnostic_cols = [col for col in self.merged_df.columns 
                          if 'diagnostics' in col.lower()]
        all_exclude_cols.extend(diagnostic_cols)
        
        # Get feature columns
        feature_cols = [col for col in self.merged_df.columns 
                       if col not in all_exclude_cols]
        
        print(f"\nPreparing features...")
        print(f"Total feature columns: {len(feature_cols)}")
        
        X = self.merged_df[feature_cols].values
        y = self.merged_df[self.label_col].values
        y = np.where(y == '<1', 0, 1)  # Convert to binary labels
        
        # Handle any remaining missing values
        if np.isnan(X).any():
            print(f"Warning: Found NaN values, filling with median...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    def feature_selection(self, X, y, method='univariate', k=20):
        """
        Perform feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            method: 'univariate', 'rfe', or 'tree_based'
            k: Number of features to select
            
        Returns:
            X_selected, selected feature names
        """
        print(f"\nPerforming feature selection using {method} method...")
        
        if method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            # Store selector for later use
            self.feature_selector = selector
            
            # Get feature scores
            scores = selector.scores_
            feature_scores = pd.DataFrame({
                'Feature': self.feature_names,
                'Score': scores
            }).sort_values('Score', ascending=False)
            
            feature_scores.to_csv(self.output_dir / 'feature_scores.csv', index=False)
            print(f"Selected {len(selected_features)} features")
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            selector = RFECV(estimator, step=1, cv=5, scoring='roc_auc', n_jobs=-1)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            self.feature_selector = selector
            
            print(f"Optimal number of features: {selector.n_features_}")
            
        elif method == 'tree_based':
            # Tree-based feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            
            selected_features = [self.feature_names[i] for i in indices]
            X_selected = X[:, indices]
            
            # Store indices for later use
            self.selected_feature_indices = indices
            
            # Save feature importances
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            feature_importance.to_csv(self.output_dir / 'feature_importance.csv', index=False)
            
            # Plot top features
            self.plot_feature_importance(feature_importance.head(20))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.selected_features = selected_features
        print(f"Selected top {len(selected_features)} features based on feature importance")
        print(f"Selected features: {selected_features[:10]}..." if len(selected_features) > 10 
              else f"Selected features: {selected_features}")
        
        return X_selected, selected_features
    
    def plot_feature_importance(self, feature_importance_df, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        
        sns.barplot(data=top_features, y='Feature', x='Importance')
        plt.title(f'Top {top_n} Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved")
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple ML models and compare performance
        
        Returns:
            Dictionary of trained models and their performance
        """
        print("\nTraining multiple models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
            'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob_test = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'auc': roc_auc_score(y_train, y_prob_train),
                'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'f1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            }
            
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'auc': roc_auc_score(y_test, y_prob_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            }
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                       scoring='roc_auc', n_jobs=-1)
            
            results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'y_prob_train': y_prob_train,
                'y_prob_test': y_prob_test,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
            
            print(f"Test AUC: {test_metrics['auc']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def save_predictions(self, results, y_train, y_test):
        """
        Save prediction results for each patient in both training and testing sets
        
        Args:
            results: Dictionary containing model results
            y_train: True labels for training set
            y_test: True labels for test set
        """
        print("\nSaving individual patient predictions...")
        
        # Get case IDs for train and test sets
        train_case_ids = self.merged_df.iloc[self.train_indices][self.case_id_col].values
        test_case_ids = self.merged_df.iloc[self.test_indices][self.case_id_col].values
        
        # For each model, save predictions
        for model_name, result in results.items():
            print(f"Saving predictions for {model_name}...")
            
            # Training set predictions
            train_predictions = pd.DataFrame({
                'CaseID': train_case_ids,
                'True_Label': y_train,
                'Predicted_Label': result['y_pred_train'],
                'Predicted_Probability': result['y_prob_train'],
                'Dataset': 'Training',
                'Correct_Prediction': (y_train == result['y_pred_train']).astype(int)
            })
            
            # Test set predictions
            test_predictions = pd.DataFrame({
                'CaseID': test_case_ids,
                'True_Label': y_test,
                'Predicted_Label': result['y_pred_test'],
                'Predicted_Probability': result['y_prob_test'],
                'Dataset': 'Testing',
                'Correct_Prediction': (y_test == result['y_pred_test']).astype(int)
            })
            
            # Combine train and test predictions
            all_predictions = pd.concat([train_predictions, test_predictions], ignore_index=True)
            
            # Add prediction confidence (distance from decision boundary at 0.5)
            all_predictions['Prediction_Confidence'] = np.abs(
                all_predictions['Predicted_Probability'] - 0.5
            )
            
            # Save to CSV
            filename = f"predictions_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
            all_predictions.to_csv(self.output_dir / filename, index=False)
            
            print(f"  Saved to {filename}")
            
            # Print summary statistics
            print(f"  Training set - Correct: {train_predictions['Correct_Prediction'].sum()}/{len(train_predictions)}")
            print(f"  Testing set  - Correct: {test_predictions['Correct_Prediction'].sum()}/{len(test_predictions)}")
        
        # Save combined predictions from all models
        self.save_combined_predictions(results, train_case_ids, test_case_ids, y_train, y_test)
    
    def save_combined_predictions(self, results, train_case_ids, test_case_ids, y_train, y_test):
        """
        Save predictions from all models in a single comprehensive file
        """
        print("\nCreating combined predictions file...")
        
        # Training set combined
        train_combined = pd.DataFrame({
            'CaseID': train_case_ids,
            'True_Label': y_train,
            'Dataset': 'Training'
        })
        
        # Add predictions from each model
        for model_name, result in results.items():
            clean_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
            train_combined[f'{clean_name}_Pred'] = result['y_pred_train']
            train_combined[f'{clean_name}_Prob'] = result['y_prob_train']
        
        # Test set combined
        test_combined = pd.DataFrame({
            'CaseID': test_case_ids,
            'True_Label': y_test,
            'Dataset': 'Testing'
        })
        
        # Add predictions from each model
        for model_name, result in results.items():
            clean_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
            test_combined[f'{clean_name}_Pred'] = result['y_pred_test']
            test_combined[f'{clean_name}_Prob'] = result['y_prob_test']
        
        # Combine both datasets
        all_combined = pd.concat([train_combined, test_combined], ignore_index=True)
        
        # Add ensemble prediction (majority vote)
        pred_cols = [col for col in all_combined.columns if col.endswith('_Pred')]
        all_combined['Ensemble_Pred'] = all_combined[pred_cols].mode(axis=1)[0]
        
        # Add average probability across models
        prob_cols = [col for col in all_combined.columns if col.endswith('_Prob')]
        all_combined['Ensemble_Prob'] = all_combined[prob_cols].mean(axis=1)
        
        # Add agreement score (how many models agree)
        all_combined['Model_Agreement'] = all_combined[pred_cols].sum(axis=1) / len(pred_cols)
        
        # Save combined predictions
        all_combined.to_csv(self.output_dir / 'predictions_all_models_combined.csv', index=False)
        print(f"Combined predictions saved to predictions_all_models_combined.csv")
        
        # Calculate ensemble performance
        train_ensemble_acc = accuracy_score(
            train_combined['True_Label'], 
            all_combined[all_combined['Dataset'] == 'Training']['Ensemble_Pred']
        )
        test_ensemble_acc = accuracy_score(
            test_combined['True_Label'], 
            all_combined[all_combined['Dataset'] == 'Testing']['Ensemble_Pred']
        )
        
        print(f"\nEnsemble Performance:")
        print(f"  Training Accuracy: {train_ensemble_acc:.4f}")
        print(f"  Testing Accuracy: {test_ensemble_acc:.4f}")
    
    def save_best_model_predictions(self, best_model_name, results, y_train, y_test):
        """
        Create a detailed prediction file for the best model with additional statistics
        """
        print(f"\nCreating detailed predictions for best model: {best_model_name}...")
        
        best_result = results[best_model_name]
        
        # Get case IDs
        train_case_ids = self.merged_df.iloc[self.train_indices][self.case_id_col].values
        test_case_ids = self.merged_df.iloc[self.test_indices][self.case_id_col].values
        
        # Create detailed training predictions
        train_detailed = pd.DataFrame({
            'CaseID': train_case_ids,
            'Dataset': 'Training',
            'True_Label': y_train,
            'True_Label_Text': ['Positive' if y == 1 else 'Negative' for y in y_train],
            'Predicted_Label': best_result['y_pred_train'],
            'Predicted_Label_Text': ['Positive' if y == 1 else 'Negative' for y in best_result['y_pred_train']],
            'Probability_Negative': 1 - best_result['y_prob_train'],
            'Probability_Positive': best_result['y_prob_train'],
            'Prediction_Correct': (y_train == best_result['y_pred_train']).astype(int),
            'Prediction_Type': self._get_prediction_type(y_train, best_result['y_pred_train'])
        })
        
        # Create detailed test predictions
        test_detailed = pd.DataFrame({
            'CaseID': test_case_ids,
            'Dataset': 'Testing',
            'True_Label': y_test,
            'True_Label_Text': ['Positive' if y == 1 else 'Negative' for y in y_test],
            'Predicted_Label': best_result['y_pred_test'],
            'Predicted_Label_Text': ['Positive' if y == 1 else 'Negative' for y in best_result['y_pred_test']],
            'Probability_Negative': 1 - best_result['y_prob_test'],
            'Probability_Positive': best_result['y_prob_test'],
            'Prediction_Correct': (y_test == best_result['y_pred_test']).astype(int),
            'Prediction_Type': self._get_prediction_type(y_test, best_result['y_pred_test'])
        })
        
        # Combine
        all_detailed = pd.concat([train_detailed, test_detailed], ignore_index=True)
        
        # Add confidence categories
        all_detailed['Confidence_Category'] = pd.cut(
            all_detailed['Probability_Positive'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low_Confidence_Negative', 'Uncertain', 'High_Confidence_Positive']
        )
        
        # Save
        filename = f"predictions_best_model_{best_model_name.replace(' ', '_').replace('(', '').replace(')', '')}_detailed.csv"
        all_detailed.to_csv(self.output_dir / filename, index=False)
        print(f"Detailed predictions saved to {filename}")
        
        # Create summary statistics
        self._create_prediction_summary(all_detailed, best_model_name)
        
        return all_detailed
    
    def _get_prediction_type(self, y_true, y_pred):
        """
        Classify prediction type (TP, TN, FP, FN)
        """
        prediction_types = []
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1:
                prediction_types.append('True_Positive')
            elif true == 0 and pred == 0:
                prediction_types.append('True_Negative')
            elif true == 0 and pred == 1:
                prediction_types.append('False_Positive')
            else:  # true == 1 and pred == 0
                prediction_types.append('False_Negative')
        return prediction_types
    
    def _create_prediction_summary(self, predictions_df, model_name):
        """
        Create summary statistics for predictions
        """
        summary_data = []
        
        for dataset in ['Training', 'Testing']:
            subset = predictions_df[predictions_df['Dataset'] == dataset]
            
            # Overall statistics
            accuracy = subset['Prediction_Correct'].mean()
            
            # By prediction type
            type_counts = subset['Prediction_Type'].value_counts()
            
            summary_data.append({
                'Model': model_name,
                'Dataset': dataset,
                'Total_Cases': len(subset),
                'Accuracy': accuracy,
                'True_Positives': type_counts.get('True_Positive', 0),
                'True_Negatives': type_counts.get('True_Negative', 0),
                'False_Positives': type_counts.get('False_Positive', 0),
                'False_Negatives': type_counts.get('False_Negative', 0),
                'Avg_Prob_Positive': subset['Probability_Positive'].mean(),
                'Avg_Prob_Negative': subset['Probability_Negative'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'prediction_summary.csv', index=False)
        
        print("\nPrediction Summary:")
        print(summary_df.to_string(index=False))
    
    def create_misclassification_report(self, predictions_df):
        """
        Create a detailed report of misclassified cases
        """
        print("\nCreating misclassification report...")
        
        # Get misclassified cases
        misclassified = predictions_df[predictions_df['Prediction_Correct'] == 0].copy()
        
        if len(misclassified) == 0:
            print("No misclassified cases found!")
            return
        
        # Sort by probability (most confident mistakes first)
        misclassified['Confidence'] = np.abs(misclassified['Probability_Positive'] - 0.5)
        misclassified = misclassified.sort_values('Confidence', ascending=False)
        
        # Save misclassification report
        misclassified.to_csv(self.output_dir / 'misclassified_cases.csv', index=False)
        
        print(f"Found {len(misclassified)} misclassified cases:")
        print(f"  Training: {len(misclassified[misclassified['Dataset'] == 'Training'])}")
        print(f"  Testing: {len(misclassified[misclassified['Dataset'] == 'Testing'])}")
        print(f"\nMost confident mistakes:")
        print(misclassified[['CaseID', 'Dataset', 'True_Label_Text', 'Predicted_Label_Text', 
                            'Probability_Positive', 'Confidence']].head(10).to_string(index=False))
    
    def evaluate_and_visualize(self, results, y_test):
        """
        Evaluate models and create visualizations
        """
        print("\nGenerating evaluation plots...")
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Train AUC': result['train_metrics']['auc'],
                'Test AUC': result['test_metrics']['auc'],
                'CV AUC': result['cv_auc_mean'],
                'Test Accuracy': result['test_metrics']['accuracy'],
                'Test F1': result['test_metrics']['f1']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test AUC', ascending=False)
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_prob_test'])
            auc = result['test_metrics']['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - PD-L1 Expression Prediction')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot model comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        metrics_to_plot = ['Train AUC', 'Test AUC', 'CV AUC']
        comparison_melted = comparison_df.melt(
            id_vars='Model', 
            value_vars=metrics_to_plot,
            var_name='Metric', 
            value_name='Score'
        )
        
        sns.barplot(data=comparison_melted, x='Model', y='Score', hue='Metric', ax=axes[0])
        axes[0].set_title('Model Performance - AUC Comparison')
        axes[0].set_ylim([0.5, 1.0])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(loc='lower right')
        
        # Other metrics comparison
        metrics_to_plot2 = ['Test Accuracy', 'Test F1']
        comparison_melted2 = comparison_df.melt(
            id_vars='Model', 
            value_vars=metrics_to_plot2,
            var_name='Metric', 
            value_name='Score'
        )
        
        sns.barplot(data=comparison_melted2, x='Model', y='Score', hue='Metric', ax=axes[1])
        axes[1].set_title('Model Performance - Accuracy & F1')
        axes[1].set_ylim([0.5, 1.0])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Get best model
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = results[best_model_name]['model']
        print(f"\nBest model: {best_model_name}")
        
        # Plot confusion matrix for best model
        self.plot_confusion_matrix(
            y_test, 
            results[best_model_name]['y_pred_test'],
            best_model_name
        )
        
        return comparison_df, best_model_name
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model, model_name, scaler, selected_features):
        """Save trained model and preprocessing objects"""
        import joblib
        
        model_path = self.output_dir / f'{model_name.replace(" ", "_")}_model.pkl'
        scaler_path = self.output_dir / 'scaler.pkl'
        features_path = self.output_dir / 'selected_features.txt'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(features_path, 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Selected features saved to {features_path}")
    
    def run_complete_pipeline(self, 
                             test_size=0.2, 
                             feature_selection_method='univariate',
                             n_features=20,
                             random_state=42):
        """
        Run the complete ML pipeline
        
        Args:
            test_size: Proportion of test set
            feature_selection_method: Method for feature selection
            n_features: Number of features to select
            random_state: Random seed
        """
        print("="*60)
        print("PD-L1 EXPRESSION PREDICTION PIPELINE")
        print("="*60)
        
        # Step 1: Load and merge data
        self.load_and_merge_data()
        
        # Step 2: Prepare features
        X, y, feature_names = self.prepare_features()
        
        # Step 3: Split data and store indices
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        
        # Create stratified split
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Store indices for later use
        self.train_indices = train_idx
        self.test_indices = test_idx
        
        # Split the data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Training set PD-L1 distribution: {np.bincount(y_train)}")
        print(f"Test set PD-L1 distribution: {np.bincount(y_test)}")
        
        # Step 4: Standardize features
        print("\nStandardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 5: Feature selection
        X_train_selected, selected_features = self.feature_selection(
            X_train_scaled, y_train, 
            method=feature_selection_method, 
            k=n_features
        )
        
        # Apply same feature selection to test set
        if feature_selection_method == 'univariate':
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        elif feature_selection_method == 'rfe':
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        elif feature_selection_method == 'tree_based':
            X_test_selected = X_test_scaled[:, self.selected_feature_indices]
        else:
            X_test_selected = X_test_scaled[:, :len(selected_features)]
        
        # Step 6: Train models
        results = self.train_models(X_train_selected, X_test_selected, y_train, y_test)
        
        # Step 7: Evaluate and visualize
        comparison_df, best_model_name = self.evaluate_and_visualize(results, y_test)
        
        # Step 8: Save predictions for all patients
        print("\n" + "="*60)
        print("SAVING PATIENT-LEVEL PREDICTIONS")
        print("="*60)
        self.save_predictions(results, y_train, y_test)
        
        # Step 9: Save detailed predictions for best model
        best_predictions = self.save_best_model_predictions(best_model_name, results, y_train, y_test)
        
        # Step 10: Create misclassification report
        self.create_misclassification_report(best_predictions)
        
        # Step 11: Save best model
        self.save_model(self.best_model, best_model_name, self.scaler, selected_features)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  1. predictions_[model_name].csv - Individual predictions per model")
        print("  2. predictions_all_models_combined.csv - All models combined with ensemble")
        print("  3. predictions_best_model_[name]_detailed.csv - Detailed best model predictions")
        print("  4. misclassified_cases.csv - Report of misclassified cases")
        print("  5. prediction_summary.csv - Summary statistics")
        print("  6. model_comparison.csv - Model performance comparison")
        print("  7. Various visualization plots (ROC, confusion matrix, etc.)")
        
        return results, comparison_df, best_predictions


def load_and_predict_new_cases(model_path, scaler_path, features_path, new_features_csv, output_csv):
    """
    Load a trained model and make predictions on new cases
    
    Args:
        model_path: Path to saved model (.pkl)
        scaler_path: Path to saved scaler (.pkl)
        features_path: Path to selected features (.txt)
        new_features_csv: Path to new radiomics features
        output_csv: Path to save predictions
    """
    import joblib
    
    print("Loading trained model and preprocessing objects...")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load selected features
    with open(features_path, 'r') as f:
        selected_features = [line.strip() for line in f]
    
    print(f"Loaded model with {len(selected_features)} features")
    
    # Load new data
    new_data = pd.read_csv(new_features_csv)
    case_ids = new_data['CaseID'].values
    
    # Extract selected features
    X_new = new_data[selected_features].values
    
    # Handle missing values
    if np.isnan(X_new).any():
        print("Warning: Found NaN values, filling with median...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_new = imputer.fit_transform(X_new)
    
    # Standardize
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'CaseID': case_ids,
        'Predicted_Label': predictions,
        'Predicted_Label_Text': ['Positive' if p == 1 else 'Negative' for p in predictions],
        'Probability_Negative': 1 - probabilities,
        'Probability_Positive': probabilities,
        'Confidence_Category': pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low_Confidence_Negative', 'Uncertain', 'High_Confidence_Positive']
        )
    })
    
    # Save predictions
    results_df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to {output_csv}")
    print(f"Total cases predicted: {len(results_df)}")
    print(f"\nPrediction distribution:")
    print(results_df['Predicted_Label_Text'].value_counts())
    
    return results_df


# Example usage
if __name__ == "__main__":
    # ============================================================
    # PART 1: Train models and get predictions on training/test sets
    # ============================================================
    
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60 + "\n")
    
    predictor = PDL1Predictor(
        features_csv='XXXyour path/Dataset/Dataset001_PDL1Expression/radiomics_features.csv',
        clinical_xlsx='XXXyour path/Dataset/Dataset001_PDL1Expression/PatientLabels.xlsx',
        output_dir='XXXyour path/Dataset/Dataset001_PDL1Expression/pdl1_prediction_results'
    )
    
    results, comparison, best_predictions = predictor.run_complete_pipeline(
        test_size=0.3,
        feature_selection_method='rfe',  # or 'univariate' or 'rfe' or 'tree_based'
        n_features=20,
        random_state=42
    )
    
    # ============================================================
    # PART 2: Use trained model to predict new cases (optional)
    # ============================================================
    
    # Uncomment to predict on new cases:
    # print("\n" + "="*60)
    # print("PREDICTION ON NEW CASES")
    # print("="*60 + "\n")
    # 
    # new_predictions = load_and_predict_new_cases(
    #     model_path='pdl1_prediction_results/Random_Forest_model.pkl',
    #     scaler_path='pdl1_prediction_results/scaler.pkl',
    #     features_path='pdl1_prediction_results/selected_features.txt',
    #     new_features_csv='new_radiomics_features.csv',
    #     output_csv='new_case_predictions.csv'
    # )