import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    RFE,
    SelectKBest, 
    f_classif, 
    RFECV,
    mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    average_precision_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LassoCV, Lasso
import joblib
import json

def plot_roc_curves(results: Dict, y_test: np.ndarray, output_dir: Path, n_features: int):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_proba_test'])
        auc = result['auc']
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Top {n_features} Features', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'roc_curves_top{n_features}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(results: Dict, y_test: np.ndarray, output_dir: Path, n_features: int):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred_test'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    ax=axes[idx], cbar=True,
                    xticklabels=['pCR', 'Non-pCR'],
                    yticklabels=['pCR', 'Non-pCR'])
        
        axes[idx].set_title(f'{model_name}\nAcc={result["test_accuracy"]:.3f}', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    # Hide extra subplot
    if n_models < len(axes):
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
    
    plt.suptitle(f'Confusion Matrices - Top {n_features} Features', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrices_top{n_features}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_net_benefit(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Calculate net benefit for decision curve analysis"""
    net_benefits = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # True positives and false positives
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        
        # Net benefit calculation
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)
    
    return np.array(net_benefits)

def plot_decision_curve_analysis(results: Dict, y_test: np.ndarray, 
                                 output_dir: Path, n_features: int):
    """Plot Decision Curve Analysis (DCA)"""
    plt.figure(figsize=(10, 8))
    
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # Calculate net benefit for each model
    for model_name, result in results.items():
        net_benefits = calculate_net_benefit(y_test, result['y_proba_test'], thresholds)
        plt.plot(thresholds, net_benefits, linewidth=2, label=model_name)
    
    # Treat all (assume all positive)
    prevalence = np.mean(y_test)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, treat_all, 'k--', linewidth=1.5, label='Treat All')
    
    # Treat none
    plt.plot(thresholds, np.zeros_like(thresholds), 'k:', linewidth=1.5, label='Treat None')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.1, np.max(treat_all) * 1.1])
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title(f'Decision Curve Analysis - Top {n_features} Features', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'decision_curve_analysis_top{n_features}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart(results: Dict, output_dir: Path, n_features: int):
    """Plot radar chart comparing model performance"""
    metrics = ['Test Accuracy', 'AUC', 'Avg Precision']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for idx, (model_name, result) in enumerate(results.items()):
        values = [
            result['test_accuracy'],
            result['auc'],
            result['average_precision'] if result['average_precision'] is not None else 0
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.title(f'Model Performance Comparison - Top {n_features} Features', 
              fontsize=14, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'radar_chart_top{n_features}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(results: Dict, y_test: np.ndarray, output_dir: Path, n_features: int):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['y_proba_test'])
        ap = result['average_precision']
        plt.plot(recall, precision, linewidth=2, 
                label=f'{model_name} (AP={ap:.3f})' if ap else model_name)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - Top {n_features} Features', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'pr_curves_top{n_features}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_best_model(all_results: Dict, output_dir: Path):
    """Save the best performing model"""
    best_auc = 0
    best_model_info = None
    
    for n_features, results_dict in all_results.items():
        for model_name, result in results_dict['results'].items():
            if result['auc'] > best_auc:
                best_auc = result['auc']
                best_model_info = {
                    'n_features': n_features,
                    'model_name': model_name,
                    'model': result['model'],
                    'scaler': results_dict['scaler'],
                    'selector': results_dict['selector'],
                    'feature_names': results_dict['selected_features'],
                    'metrics': {
                        'train_accuracy': result['train_accuracy'],
                        'test_accuracy': result['test_accuracy'],
                        'auc': result['auc'],
                        'average_precision': result['average_precision']
                    }
                }
    
    if best_model_info:
        # Save model
        model_path = output_dir / 'best_model.pkl'
        joblib.dump({
            'model': best_model_info['model'],
            'scaler': best_model_info['scaler'],
            'selector': best_model_info['selector'],
            'feature_names': best_model_info['feature_names']
        }, model_path)
        
        # Save metadata
        metadata = {
            'n_features': int(best_model_info['n_features']),
            'model_name': best_model_info['model_name'],
            'metrics': {k: float(v) if v is not None else None 
                       for k, v in best_model_info['metrics'].items()},
            'feature_names': best_model_info['feature_names']
        }
        
        with open(output_dir / 'best_model_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print("BEST MODEL SAVED")
        print(f"{'='*60}")
        print(f"Model: {best_model_info['model_name']}")
        print(f"Features: {best_model_info['n_features']}")
        print(f"AUC: {best_auc:.4f}")
        print(f"Saved to: {model_path}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    texture_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/spatial_texture_features.csv'
    features_csv='XXXyour path/Dataset/Dataset002_ECResponse/spatial3d_features.csv'
    labels_excel='XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx'
    output_dir='XXXyour path/Dataset/Dataset002_ECResponse/model_results'

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STEP 1: LOADING AND MERGING DATA")
    print("="*60)
    
    # 加载特征
    df_features = pd.read_csv(features_csv)
    print(f"\n✓ Loaded features: {df_features.shape}")
    print(f"  Columns: {len(df_features.columns)}")
    
    df_texture_features = pd.read_csv(texture_features_csv)
    print(f"\n✓ Loaded texture features: {df_texture_features.shape}")
    print(f"  Columns: {len(df_texture_features.columns)}")
    
    # 加载标签
    df_labels = pd.read_excel(labels_excel)
    print(f"\n✓ Loaded labels: {df_labels.shape}")
    print(f"  Columns: {list(df_labels.columns)}")
    
    # 标准化列名
    df_labels.columns = ['case_id', 'response']
    
    # 确保case_id类型一致
    df_features['case_id'] = df_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
    df_texture_features['case_id'] = df_texture_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
    df_labels['case_id'] = df_labels['case_id'].astype(str)
    
    # 合并数据
    df_merged = pd.merge(
        df_features,
        df_labels,
        on='case_id',
        how='inner'
    )
    
    df_merged = pd.merge(
        df_texture_features,
        df_merged,
        on='case_id',
        how='inner'
    )

    print(f"\n✓ Merged dataset: {df_merged.shape}")
    print(f"  Matched cases: {len(df_merged)}")
    print(f"  Features lost: {len(df_features) - len(df_merged)}")
    print(f"  Labels without features: {len(df_labels) - len(df_merged)}")
    
    # 检查标签分布
    print(f"\n📊 Response distribution:")
    response_counts = df_merged['response'].value_counts()
    for label, count in response_counts.items():
        percentage = count / len(df_merged) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    print("\n" + "="*60)
    print("STEP 2: FEATURE PREPARATION")
    print("="*60)
    
    exclude_cols = [
        'case_id', 
        'response', 
        'success', 
        'error_message',
        'total_textures_processed',
        'successful_textures'
    ]
    
    # 选择特征列
    feature_cols = [col for col in df_merged.columns 
                    if col not in exclude_cols]
    
    # 只保留数值型特征
    numeric_cols = df_merged[feature_cols].select_dtypes(
        include=[np.number]
    ).columns.tolist()
    
    print(f"\n✓ Total numeric features: {len(numeric_cols)}")
    
    # 处理缺失值
    X_raw = df_merged[numeric_cols].copy()
    
    missing_summary = X_raw.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    handle_missing = 'drop'

    if len(missing_features) > 0:
        print(f"\n⚠️  Features with missing values: {len(missing_features)}")
        print(missing_features.head(10))
        
        if handle_missing == 'drop':
            # 删除缺失值过多的特征（>50%）
            drop_features = missing_summary[missing_summary > len(X_raw)*0.5].index
            X_raw = X_raw.drop(columns=drop_features)
            print(f"\n  Dropped {len(drop_features)} features with >50% missing")
            
            # 删除有任何缺失值的样本
            before = len(X_raw)
            X_raw = X_raw.dropna()
            after = len(X_raw)
            print(f"  Dropped {before-after} samples with missing values")
            
        elif handle_missing == 'mean':
            X_raw = X_raw.fillna(X_raw.mean())
        elif handle_missing == 'median':
            X_raw = X_raw.fillna(X_raw.median())
    
    # 删除零方差特征
    variances = X_raw.var()
    zero_var_features = variances[variances == 0].index
    if len(zero_var_features) > 0:
        X_raw = X_raw.drop(columns=zero_var_features)
        print(f"\n  Dropped {len(zero_var_features)} zero-variance features")
    
    # 删除高度相关的特征（>0.95）
    corr_matrix = X_raw.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_triangle.columns 
                if any(upper_triangle[col] > 0.95)]
    
    if len(to_drop) > 0:
        X_raw = X_raw.drop(columns=to_drop)
        print(f"  Dropped {len(to_drop)} highly correlated features (r>0.95)")
    
    X = X_raw
    feature_names = X_raw.columns.tolist()
    
    # 获取对应的标签
    y = df_merged.loc[X_raw.index, 'response'].values
    y = [0 if (isinstance(x, str) and x == 'pCR') or x == 0 else 1 for x in y]

    print(f"\n✓ Final feature matrix: {X.shape}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")

    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    random_state = 42 
    test_size = 0.3
    
    # Store all results for comparison
    all_results = {}
    performance_summary = []
    
    for n_features in range(3, 25):
        print(f"\n{'#'*60}")
        print(f"Evaluating with top {n_features} features")
        print(f"{'#'*60}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n✓ Data split:")
        print(f"  Training (before SMOTE): {len(X_train_scaled)} samples")
        print(f"  Testing: {len(X_test_scaled)} samples")

        # Feature selection using RFE with Lasso
        estimator = Lasso(alpha=0.00005, random_state=42, max_iter=10000)
        selector = RFE(estimator, n_features_to_select=n_features, step=3)
        
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_feature_names = [feature_names[i] for i, selected in enumerate(selector.support_) if selected]
        
        print(f"\n✓ Selected features: {len(selected_feature_names)}")
        print(f"  Top 5: {selected_feature_names[:5]}")
        
        # Apply SMOTE to balance the training dataset
        smote = SMOTE(random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train_selected, y_train)

        print(f"  Training (after SMOTE): {len(X_train_final)} samples")
        print(f"  Class distribution (train): {np.bincount(y_train_final)}")

        # 定义模型
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state,
                class_weight='balanced'
            ),
            'MLP': MLPClassifier(
                max_iter=1000,
                random_state=random_state,
                early_stopping=True
            ),
            'XGBoost': XGBClassifier(
                n_estimators=10,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        # 训练和评估每个模型
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name}...")
            print(f"{'='*60}")
            
            # 训练模型
            model.fit(X_train_final, y_train_final)
            
            # 预测
            y_pred_train = model.predict(X_train_final)
            y_pred_test = model.predict(X_test_selected)
            y_proba_test = model.predict_proba(X_test_selected)[:, 1]
            
            # 评估指标
            train_acc = accuracy_score(y_train_final, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            auc = roc_auc_score(y_test, y_proba_test)
            ap = average_precision_score(y_test, y_proba_test)
            
            results[model_name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'auc': auc,
                'average_precision': ap,
                'y_pred_test': y_pred_test,
                'y_proba_test': y_proba_test
            }
            
            # Store for summary
            performance_summary.append({
                'n_features': n_features,
                'model': model_name,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'auc': auc,
                'avg_precision': ap
            })
            
            print(f"\n  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  AUC:           {auc:.4f}")
            print(f"  Avg Precision: {ap:.4f}")
        
        # Store results for this feature count
        all_results[n_features] = {
            'results': results,
            'scaler': scaler,
            'selector': selector,
            'selected_features': selected_feature_names,
            'y_test': y_test
        }
        
        # Generate plots for this feature configuration
        print(f"\n{'='*60}")
        print(f"Generating plots for {n_features} features...")
        print(f"{'='*60}")
        
        plot_roc_curves(results, y_test, output_dir, n_features)
        print(f"  ✓ ROC curves saved")
        
        plot_confusion_matrices(results, y_test, output_dir, n_features)
        print(f"  ✓ Confusion matrices saved")
        
        plot_decision_curve_analysis(results, y_test, output_dir, n_features)
        print(f"  ✓ Decision curve analysis saved")
        
        plot_radar_chart(results, output_dir, n_features)
        print(f"  ✓ Radar chart saved")
        
        plot_pr_curves(results, y_test, output_dir, n_features)
        print(f"  ✓ Precision-Recall curves saved")

    print("\n" + "="*60)
    print("STEP 4: PERFORMANCE SUMMARY AND MODEL SELECTION")
    print("="*60)
    
    # Create performance summary DataFrame
    df_summary = pd.DataFrame(performance_summary)
    df_summary.to_csv(output_dir / 'performance_summary.csv', index=False)
    print(f"\n✓ Performance summary saved to: {output_dir / 'performance_summary.csv'}")
    
    # Plot performance vs number of features
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['test_acc', 'auc', 'avg_precision', 'train_acc']
    titles = ['Test Accuracy', 'AUC', 'Average Precision', 'Train Accuracy']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for model_name in df_summary['model'].unique():
            model_data = df_summary[df_summary['model'] == model_name]
            ax.plot(model_data['n_features'], model_data[metric], 
                   marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Number of Features', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Model Performance vs Number of Features', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance comparison plot saved")
    
    # Find and save best model
    save_best_model(all_results, output_dir)
    
    # Print top 10 configurations
    print(f"\n{'='*60}")
    print("TOP 10 MODEL CONFIGURATIONS BY AUC")
    print(f"{'='*60}")
    
    top_configs = df_summary.nlargest(10, 'auc')
    print("\n" + top_configs.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"Total configurations evaluated: {len(performance_summary)}")
    print(f"Feature ranges tested: 3-24")
    print(f"Models tested: {len(models)}")