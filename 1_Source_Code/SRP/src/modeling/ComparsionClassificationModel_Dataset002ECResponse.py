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
from sklearn.linear_model import LogisticRegression, Ridge
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


if __name__ == "__main__":
    ## Spatial Feature Extraction - Comparison Classification Model
    print("Spatial Model")
    texture_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/spatial_texture_features.csv'
    features_csv='XXXyour path/Dataset/Dataset002_ECResponse/spatial3d_features.csv'
    labels_excel='XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx'
    output_dir='XXXyour path/Dataset/Dataset002_ECResponse/model_results'

    TJH_texture_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial_texture_features.csv'
    TJH_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/TJH/spatial3d_features.csv'
    TJH_labels_excel='XXXyour path/Dataset/Dataset002_ECResponse/TJH/PatientLabels.xlsx'
    TJH_output_dir='XXXyour path/Dataset/Dataset002_ECResponse/TJH/model_results'

    scRNA_texture_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/spatial_texture_features.csv'
    scRNA_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/spatial3d_features.csv'
    scRNA_output_dir='XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/model_results'

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    TJH_output_dir = Path(TJH_output_dir)
    TJH_output_dir.mkdir(parents=True, exist_ok=True)

    scRNA_output_dir = Path(scRNA_output_dir)
    scRNA_output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STEP 1: LOADING AND MERGING DATA")
    print("="*60)
    
    # 加载特征
    df_features = pd.read_csv(features_csv)
    print(f"\n✓ Loaded features: {df_features.shape}")
    print(f"  Columns: {len(df_features.columns)}")
    
    TJH_df_features = pd.read_csv(TJH_features_csv)
    print(f"\n✓ Loaded features: {TJH_df_features.shape}")
    print(f"  Columns: {len(TJH_df_features.columns)}")

    scRNA_df_features = pd.read_csv(scRNA_features_csv)
    print(f"\n✓ Loaded features: {scRNA_df_features.shape}")
    print(f"  Columns: {len(scRNA_df_features.columns)}")

    df_texture_features = pd.read_csv(texture_features_csv)
    print(f"\n✓ Loaded texture features: {df_texture_features.shape}")
    print(f"  Columns: {len(df_texture_features.columns)}")
    
    TJH_df_texture_features = pd.read_csv(TJH_texture_features_csv)
    print(f"\n✓ Loaded texture features: {TJH_df_texture_features.shape}")
    print(f"  Columns: {len(TJH_df_texture_features.columns)}")

    scRNA_df_texture_features = pd.read_csv(scRNA_texture_features_csv)
    print(f"\n✓ Loaded texture features: {scRNA_df_texture_features.shape}")
    print(f"  Columns: {len(scRNA_df_texture_features.columns)}")

    # 加载标签
    df_labels = pd.read_excel(labels_excel)
    print(f"\n✓ Loaded labels: {df_labels.shape}")
    print(f"  Columns: {list(df_labels.columns)}")
    
    TJH_df_labels = pd.read_excel(TJH_labels_excel)
    print(f"\n✓ Loaded labels: {TJH_df_labels.shape}")
    print(f"  Columns: {list(TJH_df_labels.columns)}")
    # 标准化列名
    df_labels = df_labels.drop(columns=['DFS', 'DFS-Time', 'OS', 'OS-Time'])
    df_labels.columns = ['case_id', 'response']

    TJH_df_labels = TJH_df_labels.drop(columns=['DFS', 'DFS-Time', 'OS', 'OS-Time'])
    TJH_df_labels.columns = ['case_id', 'response']
    # df_labels.rename(columns={'CaseID': 'case_id', 'Response': 'response'}, inplace=True)
    print(f"  Columns: {list(df_labels.columns)}")
    # 确保case_id类型一致
    df_features['case_id'] = df_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
    df_texture_features['case_id'] = df_texture_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
    df_labels['case_id'] = df_labels['case_id'].astype(str)

    TJH_df_features['case_id'] = TJH_df_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
    TJH_df_texture_features['case_id'] = TJH_df_texture_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
    TJH_df_labels['case_id'] = TJH_df_labels['case_id'].astype(str)

    scRNA_df_features['case_id'] = scRNA_df_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
    scRNA_df_texture_features['case_id'] = scRNA_df_texture_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
    )
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

    TJH_df_merged = pd.merge(
        TJH_df_features,
        TJH_df_labels,
        on='case_id',
        how='inner'
    )
    
    TJH_df_merged = pd.merge(
        TJH_df_texture_features,
        TJH_df_merged,
        on='case_id',
        how='inner'
    )

    print(f"\n✓ Merged TJH dataset: {TJH_df_merged.shape}")
    print(f"  Matched cases: {len(TJH_df_merged)}")
    print(f"  Features lost: {len(TJH_df_features) - len(TJH_df_merged)}")
    print(f"  Labels without features: {len(TJH_df_labels) - len(TJH_df_merged)}")


    scRNA_df_merged = pd.merge(
        scRNA_df_texture_features,
        scRNA_df_features,
        on='case_id',
        how='inner'
    )
    # 检查标签分布
    print(f"\n📊 Response distribution:")
    response_counts = TJH_df_merged['response'].value_counts()
    for label, count in response_counts.items():
        percentage = count / len(TJH_df_merged) * 100
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
    TJH_X_raw = TJH_df_merged[numeric_cols].copy()
    scRNA_X_raw = scRNA_df_merged[numeric_cols].copy()

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
    X_TJH = TJH_X_raw.drop(columns=[col for col in TJH_X_raw.columns if col not in X.columns], errors='ignore')
    X_scRNA = scRNA_X_raw.drop(columns=[col for col in scRNA_X_raw.columns if col not in X.columns], errors='ignore')
    feature_names = X_raw.columns.tolist()
    
    # 获取对应的标签
    y = df_merged.loc[X_raw.index, 'response'].values
    y = [0 if (isinstance(x, str) and x == 'pCR') or x == 0 else 1 for x in y]
    case_ids_all = df_merged.loc[X_raw.index, 'case_id'].values

    y_TJH = TJH_df_merged.loc[X_TJH.index, 'response'].values
    y_TJH = [1 if int(x) == 0 else 0 for x in y_TJH]
    case_ids_TJH = TJH_df_merged.loc[X_TJH.index, 'case_id'].values

    case_ids_scRNA = scRNA_df_merged.loc[X_scRNA.index, 'case_id'].values

    print(f"\n✓ Final feature matrix: {X.shape}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")

    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    random_state = 42 
    test_size = 0.3
    n_features = 7
    # for n_features in range(3,25):
    # Store all results for comparison
    all_results = {}
    performance_summary = []
    

    print(f"\n{'#'*60}")
    print(f"Evaluating with top {n_features} features")
    print(f"{'#'*60}")
    
    # 划分训练集和测试集（保留case_ids）
    X_train, X_test, y_train, y_test, case_ids_train, case_ids_test = train_test_split(
        X, y, case_ids_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_TJH_scaled = scaler.fit_transform(X_TJH)
    X_scRNA_scaled = scaler.fit_transform(X_scRNA)

    print(f"\n✓ Data split:")
    print(f"  Training (before SMOTE): {len(X_train_scaled)} samples")
    print(f"  Testing: {len(X_test_scaled)} samples")

    # Feature selection using RFE with Lasso
    # estimator = Lasso(alpha=0.00005, random_state=42)
    estimator = Ridge(alpha=0.1, random_state=42, max_iter=1000)
    # estimator = SVC(kernel='linear', random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=11)
    
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    X_TJH_selected = selector.transform(X_TJH_scaled)
    X_scRNA_selected = selector.transform(X_scRNA_scaled)
    # Get selected feature names
    selected_feature_names = [feature_names[i] for i, selected in enumerate(selector.support_) if selected]
    
    print(f"\n✓ Selected features: {len(selected_feature_names)}")
    print(f"Spatial Feature names:")
    for i, feat in enumerate(selected_feature_names, 1):
        print(f"{feat}")
    
    # Apply SMOTE to balance the training dataset
    smote = SMOTE(random_state=random_state)
    X_train_final, y_train_final = smote.fit_resample(X_train_selected, y_train)

    print(f"  Training (after SMOTE): {len(X_train_final)} samples")
    print(f"  Class distribution (train): {np.bincount(y_train_final)}")
    model = XGBClassifier(
            n_estimators=83,
            random_state=random_state,
            learning_rate=0.007,
            use_label_encoder=True,
            # eval_metric='auc'
        )
    # model = MLPClassifier(random_state=42, max_iter=500)
    # 训练模型
    model.fit(X_train_final, y_train_final)
    
    # 预测
    y_pred_sp_train = model.predict(X_train_selected)
    y_proba_sp_train = model.predict_proba(X_train_selected)[:, 1]
    y_pred_sp_test = model.predict(X_test_selected)
    y_proba_sp_test = model.predict_proba(X_test_selected)[:, 1]
    y_pred_sp_TJH = model.predict(X_TJH_selected)
    y_proba_sp_TJH = model.predict_proba(X_TJH_selected)[:, 1]
    y_proba_sp_scRNA = model.predict_proba(X_scRNA_selected)[:, 1]

    # 评估指标
    train_acc = accuracy_score(y_train, y_pred_sp_train)
    train_auc = roc_auc_score(y_train, y_proba_sp_train)
    test_acc = accuracy_score(y_test, y_pred_sp_test)
    auc = roc_auc_score(y_test, y_proba_sp_test)
    TJH_acc = accuracy_score(y_TJH, y_pred_sp_TJH)
    auc_TJH = roc_auc_score(y_TJH, y_proba_sp_TJH)
    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Train AUC:      {train_auc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Test AUC:       {auc:.4f}")
    print(f"  TJH Accuracy:  {TJH_acc:.4f}")
    print(f"  TJH AUC:       {auc_TJH:.4f}")

    # ##########################################################################
    # ## Graph Spatial Model
    # print("="*60)
    # print('Graph Spatial Model')
    # print("="*60)

    # graphSpatial_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/sparad_features.csv'
    # # Load datasets
    # grasp_df = pd.read_csv(graphSpatial_features_csv)
    # # 加载标签
    # df_labels = pd.read_excel(labels_excel)
    # df_labels = df_labels.drop(columns=['DFS', 'DFS-Time', 'OS', 'OS-Time'])
    # grasp_merged = pd.merge(grasp_df, df_labels, on="CaseID", how='inner')
    # # Remove missing labels
    # grasp_merged = grasp_merged.dropna(subset=['Response'])
    # # Exclude columns
    # grasp_merged['CaseID'] = grasp_merged['CaseID'].astype(str)
    # grasp_train = grasp_merged[grasp_merged['CaseID'].isin(case_ids_train)]
    # grasp_test = grasp_merged[grasp_merged['CaseID'].isin(case_ids_test)]
    # X_train_grasp = grasp_train.drop(columns=['CaseID', 'Response'])
    # X_test_grasp = grasp_test.drop(columns=['CaseID', 'Response'])

    # # 数据标准化
    # scaler = StandardScaler()
    # X_train_grasp_scaled = scaler.fit_transform(X_train_grasp)
    # X_test_grasp_scaled = scaler.transform(X_test_grasp)
    # grasp_feature_names = X_train_grasp.columns.tolist()
    # print(f"\n✓ Data split:")
    # print(f"  Training (before SMOTE): {len(X_train_grasp_scaled)} samples")
    # print(f"  Testing: {len(X_test_grasp_scaled)} samples")
    # for n_features in range(3,25):
    #     # Feature selection using RFE with Lasso
    #     estimator = Lasso(alpha=0.0001, random_state=42, max_iter=10000)
    #     grasp_selector = RFE(estimator, n_features_to_select=n_features, step=10)
        
    #     X_train_grasp_selected = grasp_selector.fit_transform(X_train_grasp_scaled, y_train)
    #     X_test_grasp_selected = grasp_selector.transform(X_test_grasp_scaled)
        
    #     # Get selected feature names
    #     selected_graspfeature_names = [grasp_feature_names[i] for i, selected in enumerate(grasp_selector.support_) if selected]
        
    #     print(f"\n✓ Selected features: {len(selected_graspfeature_names)}")
    #     # print(f"Graph Spatial Feature names:")
    #     # for i, feat in enumerate(selected_graspfeature_names, 1):
    #     #     print(f"{feat}")
        
    #     # Apply SMOTE to balance the training dataset
    #     smote = SMOTE(random_state=random_state)
    #     X_train_final, y_train_final = smote.fit_resample(X_train_grasp_selected, y_train)

    #     print(f"  Training (after SMOTE): {len(X_train_final)} samples")
    #     print(f"  Class distribution (train): {np.bincount(y_train_final)}")
    #     model = XGBClassifier(
    #             n_estimators=10,
    #             random_state=random_state,
    #             use_label_encoder=False,
    #             eval_metric='logloss'
    #         )
        
    #     # 训练模型
    #     model.fit(X_train_final, y_train_final)
        
    #     # 预测
    #     y_pred_grasp_train = model.predict(X_train_final)
    #     y_pred_grasp_test = model.predict(X_test_grasp_selected)
    #     y_proba_grasp_test = model.predict_proba(X_test_grasp_selected)[:, 1]
        
    #     # 评估指标
    #     train_acc = accuracy_score(y_train_final, y_pred_grasp_train)
    #     test_acc = accuracy_score(y_test, y_pred_grasp_test)
    #     auc = roc_auc_score(y_test, y_proba_grasp_test)
    #     print(f"\n  Train Accuracy: {train_acc:.4f}")
    #     print(f"  Test Accuracy:  {test_acc:.4f}")
    #     print(f"  AUC:           {auc:.4f}")
    #     print("="*60)

    ##########################################################################
    ## Radiomics Model
    print("="*60)
    print('Radiomics Model')
    print("="*60)
    radiomics_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/radiomics_features.csv'
    TJH_radiomics_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/TJH/radiomics_features.csv'
    scRNA_radiomics_features_csv='XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/radiomics_features.csv'
    # Load datasets
    rad_df = pd.read_csv(radiomics_features_csv)
    TJH_rad_df = pd.read_csv(TJH_radiomics_features_csv)
    scRNA_rad_df = pd.read_csv(scRNA_radiomics_features_csv)
    # 加载标签
    df_labels = pd.read_excel(labels_excel)
    df_labels = df_labels.drop(columns=['DFS', 'DFS-Time', 'OS', 'OS-Time'])
    rad_merged = pd.merge(rad_df, df_labels, on="CaseID", how='inner')

    TJH_df_labels = pd.read_excel(TJH_labels_excel)
    TJH_df_labels = TJH_df_labels.drop(columns=['DFS', 'DFS-Time', 'OS', 'OS-Time'])
    TJH_rad_merged = pd.merge(TJH_rad_df, TJH_df_labels, on="CaseID", how='inner')

    # Remove missing labels
    rad_merged = rad_merged.dropna(subset=['Response'])
    TJH_rad_merged = TJH_rad_merged.dropna(subset=['Response'])
    # Exclude columns
    exclude = [col for col in rad_merged.columns if 'diagnostics' in col.lower()]
    rad_merged = rad_merged.drop(columns=exclude)
    rad_merged['CaseID'] = rad_merged['CaseID'].astype(str)
    rad_train = rad_merged[rad_merged['CaseID'].isin(case_ids_train)]
    rad_test = rad_merged[rad_merged['CaseID'].isin(case_ids_test)]
    X_train_rad = rad_train.drop(columns=['CaseID', 'Response'])
    X_test_rad = rad_test.drop(columns=['CaseID', 'Response'])

    TJH_rad_merged = TJH_rad_merged.drop(columns=exclude)
    TJH_rad_merged['CaseID'] = TJH_rad_merged['CaseID'].astype(str)
    TJH_rad = TJH_rad_merged[TJH_rad_merged['CaseID'].isin(case_ids_TJH)]
    X_TJH_rad = TJH_rad.drop(columns=['CaseID', 'Response'])

    scRNA_rad_df = scRNA_rad_df.drop(columns=exclude)
    scRNA_rad_df['CaseID'] = scRNA_rad_df['CaseID'].astype(str)
    scRNA_rad = scRNA_rad_df[scRNA_rad_df['CaseID'].isin(case_ids_scRNA)]
    X_scRNA_rad = scRNA_rad.drop(columns=['CaseID'])
    # 数据标准化
    scaler = StandardScaler()
    X_train_rad_scaled = scaler.fit_transform(X_train_rad)
    X_test_rad_scaled = scaler.transform(X_test_rad)
    X_TJH_rad_scaled = scaler.transform(X_TJH_rad)
    X_scRNA_rad_scaled = scaler.transform(X_scRNA_rad)
    rad_feature_names = X_train_rad.columns.tolist()
    print(f"\n✓ Data split:")
    print(f"  Training (before SMOTE): {len(X_train_rad_scaled)} samples")
    print(f"  Testing: {len(X_test_rad_scaled)} samples")
    # for n_features in range(3,25):
    # Feature selection using RFE with Lasso
    # estimator = Lasso(alpha=0.00005, random_state=42)
    estimator = Ridge(alpha=1, random_state=42)
    rad_selector = RFE(estimator, n_features_to_select=3, step=19)
    
    X_train_rad_selected = rad_selector.fit_transform(X_train_rad_scaled, y_train)
    X_test_rad_selected = rad_selector.transform(X_test_rad_scaled)
    X_TJH_rad_selected = rad_selector.transform(X_TJH_rad_scaled)
    X_scRNA_rad_selected = rad_selector.transform(X_scRNA_rad_scaled)
    # Get selected feature names
    selected_radfeature_names = [rad_feature_names[i] for i, selected in enumerate(rad_selector.support_) if selected]
    
    print(f"\n✓ Selected features: {len(selected_radfeature_names)}")
    print(f"Radiomics Feature names:")
    for i, feat in enumerate(selected_radfeature_names, 1):
        print(f"{feat}")
    
    # Apply SMOTE to balance the training dataset
    smote = SMOTE(random_state=random_state)
    X_train_final, y_train_final = smote.fit_resample(X_train_rad_selected, y_train)

    print(f"  Training (after SMOTE): {len(X_train_final)} samples")
    print(f"  Class distribution (train): {np.bincount(y_train_final)}")
    model = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.14,
            use_label_encoder=True,
            # eval_metric='logloss'
        )
    
    # 训练模型
    model.fit(X_train_final, y_train_final)
    
    # 预测
    y_pred_rad_train = model.predict(X_train_rad_selected)
    y_proba_rad_train = model.predict_proba(X_train_rad_selected)[:, 1]

    y_pred_rad_test = model.predict(X_test_rad_selected)
    y_proba_rad_test = model.predict_proba(X_test_rad_selected)[:, 1]

    y_pred_TJH_rad = model.predict(X_TJH_rad_selected)
    y_proba_TJH_rad = model.predict_proba(X_TJH_rad_selected)[:, 1]

    y_proba_scRNA_rad = model.predict_proba(X_scRNA_rad_selected)[:, 1]
    # 评估指标
    train_acc = accuracy_score(y_train, y_pred_rad_train)
    test_acc = accuracy_score(y_test, y_pred_rad_test)
    auc = roc_auc_score(y_test, y_proba_rad_test)
    TJH_acc_rad = accuracy_score(y_TJH, y_pred_TJH_rad)
    auc_TJH_rad = roc_auc_score(y_TJH, y_proba_TJH_rad)
    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  AUC:           {auc:.4f}")
    print(f"  TJH Accuracy:  {TJH_acc_rad:.4f}")
    print(f"  TJH AUC:       {auc_TJH_rad:.4f}")
    print("="*60)

    #########################################################################
    ## Score Fusion Model
    print("\n" + "="*60)
    print('Score Fusion Model')
    print("="*60)
    scale = 0.7
    y_proba_fusion_train = scale * y_proba_sp_train + (1-scale) * y_proba_rad_train
    y_proba_fusion_test = scale * y_proba_sp_test + (1-scale) * y_proba_rad_test
    y_proba_fusion_TJH = scale * y_proba_sp_TJH + (1-scale) * y_proba_TJH_rad
    y_proba_fusion_scRNA = scale * y_proba_sp_scRNA + (1-scale) * y_proba_scRNA_rad
    # 评估指标
    test_acc = accuracy_score(y_test, y_proba_fusion_test>0.5)
    auc = roc_auc_score(y_test, y_proba_fusion_test)
    TJH_acc_fusion = accuracy_score(y_TJH, y_proba_fusion_TJH>0.5)
    auc_TJH_fusion = roc_auc_score(y_TJH, y_proba_fusion_TJH)
    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  AUC:           {auc:.4f}")
    print(f"  TJH Accuracy:  {TJH_acc_fusion:.4f}")
    print(f"  TJH AUC:       {auc_TJH_fusion:.4f}")
    print("="*60)


            
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'case_id': case_ids_train,
        'true_label': y_train,
        'true_label_name': ['pCR' if y == 0 else 'Non-pCR' for y in y_train],
        'spatial_probability': y_proba_sp_train,   # Probability of class 1 (Non-pCR)
        'radiomics_probability': y_proba_rad_train,   # Probability of class 1 (Non-pCR)
        'fused_probability': y_proba_fusion_train   # Probability of class 1 (Non-pCR)
    })
    
    # Sort by probability for better readability
    predictions_df = predictions_df.sort_values('fused_probability', ascending=False)
    # Save to CSV
    csv_path = output_dir / 'train_predictions_results.csv'
    predictions_df.to_csv(csv_path, index=False)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'case_id': case_ids_test,
        'true_label': y_test,
        'true_label_name': ['pCR' if y == 0 else 'Non-pCR' for y in y_test],
        'spatial_probability': y_proba_sp_test,   # Probability of class 1 (Non-pCR)
        'radiomics_probability': y_proba_rad_test,   # Probability of class 1 (Non-pCR)
        'fused_probability': y_proba_fusion_test   # Probability of class 1 (Non-pCR)
    })
    
    # Sort by probability for better readability
    predictions_df = predictions_df.sort_values('fused_probability', ascending=False)
    
    # Save to CSV
    csv_path = output_dir / 'test_predictions_results.csv'
    predictions_df.to_csv(csv_path, index=False)

    # Create predictions DataFrame
    TJH_predictions_df = pd.DataFrame({
        'case_id': case_ids_TJH,
        'true_label': y_TJH,
        'true_label_name': ['pCR' if y == 0 else 'Non-pCR' for y in y_TJH],
        'spatial_probability': y_proba_sp_TJH,   # Probability of class 1 (Non-pCR)
        'radiomics_probability': y_proba_TJH_rad,   # Probability of class 1 (Non-pCR)
        'fused_probability': y_proba_fusion_TJH   # Probability of class 1 (Non-pCR)
    })
    
    # Sort by probability for better readability
    TJH_predictions_df = TJH_predictions_df.sort_values('fused_probability', ascending=False)
    
    # Save to CSV
    csv_path = TJH_output_dir / 'TJH_predictions_results.csv'
    TJH_predictions_df.to_csv(csv_path, index=False)
    # ## Feature Fusion Model


    # Create predictions DataFrame
    scRNA_predictions_df = pd.DataFrame({
        'case_id': case_ids_scRNA,
        'spatial_probability': y_proba_sp_scRNA,   # Probability of class 1 (Non-pCR)
        'radiomics_probability': y_proba_scRNA_rad,   # Probability of class 1 (Non-pCR)
        'fused_probability': y_proba_fusion_scRNA   # Probability of class 1 (Non-pCR)
    })
    
    # Sort by probability for better readability
    scRNA_predictions_df = scRNA_predictions_df.sort_values('fused_probability', ascending=False)
    
    # Save to CSV
    csv_path = scRNA_output_dir / 'scRNA_predictions_results.csv'
    scRNA_predictions_df.to_csv(csv_path, index=False)
    # print("\n" + "="*60)
    # print('Feature Fusion Model')
    # print("="*60)
    # # Combine selected features from both models
    # X_train_fusion = np.hstack((X_train_selected, X_train_rad_selected))
    # X_test_fusion = np.hstack((X_test_selected, X_test_rad_selected))
    # fusion_feature_names = selected_feature_names + selected_radfeature_names
    # # for n_features in range(3,11):
    # #     # Feature selection using RFE with Lasso
    # #     estimator = Lasso(alpha=0.00005, random_state=42, max_iter=10000)
    # #     fusion_selector = RFE(estimator, n_features_to_select=n_features, step=12)
        
    # #     X_train_fusion_selected = fusion_selector.fit_transform(X_train_fusion, y_train)
    # #     X_test_fusion_selected = fusion_selector.transform(X_test_fusion)
        
    # #     # Get selected feature names
    # #     selected_fusionfeature_names = [fusion_feature_names[i] for i, selected in enumerate(fusion_selector.support_) if selected]
        
    # #     print(f"\n✓ Selected features: {len(selected_fusionfeature_names)}")
    # # print(f"  Feature names:")
    # # for i, feat in enumerate(selected_fusionfeature_names, 1):
    # #     print(f"    {i}. {feat}")
    
    # # Apply SMOTE to balance the training dataset
    # smote = SMOTE(random_state=random_state)
    # X_train_final, y_train_final = smote.fit_resample(X_train_fusion, y_train)

    # print(f"  Training (after SMOTE): {len(X_train_final)} samples")
    # print(f"  Class distribution (train): {np.bincount(y_train_final)}")
    # model = XGBClassifier(
    #         n_estimators=10,
    #         random_state=random_state,
    #         use_label_encoder=False,
    #         eval_metric='logloss'
    #     )
    
    # # 训练模型
    # model.fit(X_train_final, y_train_final)
    
    # # 预测
    # y_pred_fusion_train = model.predict(X_train_final)
    # y_pred_fusion_test = model.predict(X_test_fusion)
    # y_proba_fusion_test = model.predict_proba(X_test_fusion)[:, 1]
    
    # # 评估指标
    # train_acc = accuracy_score(y_train_final, y_pred_fusion_train)
    # test_acc = accuracy_score(y_test, y_pred_fusion_test)
    # auc = roc_auc_score(y_test, y_proba_fusion_test)
    # print(f"\n  Train Accuracy: {train_acc:.4f}")
    # print(f"  Test Accuracy:  {test_acc:.4f}")
    # print(f"  AUC:           {auc:.4f}")
    # print("="*60)

    ## Score Fusion Model
    # print("\n" + "="*60)
    # print('Score Fusion Model')
    # print("="*60)
    # y_proba_fusion_test = np.zeros_like(y_proba_sp_test)
    # y_proba_fusion_TJH = np.zeros_like(y_proba_sp_TJH) 
    # for weight in np.arange(0.1,1.0,0.1):
    #     print(f"\n  Weight: {weight:.1f}")
    #     y_proba_fusion_test = weight * y_proba_sp_test + (1-weight) * y_proba_rad_test 
    #     y_proba_fusion_TJH = weight * y_proba_sp_TJH + (1-weight) * y_proba_TJH_rad
    #     # 评估指标
    #     test_acc = accuracy_score(y_test, y_proba_fusion_test>0.5)
    #     auc = roc_auc_score(y_test, y_proba_fusion_test)
    #     TJH_acc = accuracy_score(y_TJH, y_proba_fusion_TJH>0.5)
    #     TJH_auc = roc_auc_score(y_TJH, y_proba_fusion_TJH)
    #     print(f"\n  Train Accuracy: {train_acc:.4f}")
    #     print(f"  Test Accuracy:  {test_acc:.4f}")
    #     print(f"  Test AUC:       {auc:.4f}")
    #     print(f"  TJH Accuracy:  {TJH_acc:.4f}")
    #     print(f"  TJH AUC:       {TJH_auc:.4f}")
    #     print("="*60)

    # y_proba_fusion_test = np.zeros((len(y_proba_sp_test), 2))
    # y_proba_fusion_test[:, 0] = y_proba_sp_test
    # y_proba_fusion_test[:, 1] = y_proba_rad_test
    
    # y_proba_fusion_TJH = np.zeros((len(y_proba_sp_TJH), 2))
    # y_proba_fusion_TJH[:, 0] = y_proba_sp_TJH
    # y_proba_fusion_TJH[:, 1] = y_proba_TJH_rad
    # print(f"\n Minimum score fusion:")
    # min_score = np.min(y_proba_fusion_test, axis=1)
    # TJH_min_score = np.min(y_proba_fusion_TJH, axis=1)
    # # 评估指标
    # test_acc = accuracy_score(y_test, min_score>0.5)
    # auc = roc_auc_score(y_test, min_score)

    # TJH_test_acc = accuracy_score(y_TJH, TJH_min_score>0.5)
    # TJH_auc = roc_auc_score(y_TJH, TJH_min_score)
    # print(f"\n  Train Accuracy: {train_acc:.4f}")
    # print(f"  Test Accuracy:  {test_acc:.4f}")
    # print(f"  Test AUC:       {auc:.4f}")
    # print(f"  TJH Accuracy:  {TJH_test_acc:.4f}")
    # print(f"  TJH AUC:       {TJH_auc:.4f}")
    # print("="*60)
    # print(f"\n Maximum score fusion:")
    # max_score = np.max(y_proba_fusion_test, axis=1)
    # TJH_max_score = np.max(y_proba_fusion_TJH, axis=1)
    # # 评估指标
    # test_acc = accuracy_score(y_test, max_score>0.5)
    # auc = roc_auc_score(y_test, max_score)
    # TJH_test_acc = accuracy_score(y_TJH, TJH_max_score>0.5)
    # TJH_auc = roc_auc_score(y_TJH, TJH_max_score)
    # print(f"\n  Train Accuracy: {train_acc:.4f}")
    # print(f"  Test Accuracy:  {test_acc:.4f}")
    # print(f"  Test AUC:       {auc:.4f}")
    # print(f"  TJH Accuracy:  {TJH_test_acc:.4f}")
    # print(f"  TJH AUC:       {TJH_auc:.4f}")
    # print("="*60)
    # print(f"\n Average score fusion:")   
    # y_proba_fusion_test = (min_score + max_score) / 2
    # y_proba_fusion_TJH = (TJH_min_score + TJH_max_score) / 2
    # # 评估指标
    # test_acc = accuracy_score(y_test, y_proba_fusion_test>0.5)
    # auc = roc_auc_score(y_test, y_proba_fusion_test)
    # TJH_test_acc = accuracy_score(y_TJH, y_proba_fusion_TJH>0.5)
    # TJH_auc = roc_auc_score(y_TJH, y_proba_fusion_TJH)
    # print(f"\n  Train Accuracy: {train_acc:.4f}")
    # print(f"  Test Accuracy:  {test_acc:.4f}")
    # print(f"  Test AUC:       {auc:.4f}")
    # print(f"  TJH Accuracy:  {TJH_test_acc:.4f}")
    # print(f"  TJH AUC:       {TJH_auc:.4f}")
    # print("="*60)