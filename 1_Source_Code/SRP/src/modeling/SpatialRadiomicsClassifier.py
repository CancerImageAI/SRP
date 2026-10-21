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

class FeatureSelector:
    """
    Feature selector that properly handles train/test split
    """
    
    def __init__(self, method: str = 'mutual_info', n_features: int = 30):
        """
        Args:
            method: Feature selection method
                   'mutual_info', 'anova', 'rfe', 'lasso', 'random_forest'
            n_features: Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.scaler = None
        self.selected_indices_ = None  # Indices of selected features
        self.selected_features_ = None
        self.feature_scores_ = None
        self.is_fitted_ = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'RobustFeatureSelector':
        """
        Fit feature selector on training data
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels
            feature_names: Optional feature names
            
        Returns:
            self
        """
        print("\n" + "="*80)
        print(f"FITTING FEATURE SELECTOR: {self.method.upper()}")
        print("="*80)
        print(f"Training samples: {X.shape[0]}")
        print(f"Total features: {X.shape[1]}")
        print(f"Target features: {self.n_features}")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.feature_names_in_ = np.array(feature_names)
        self.n_features_in_ = X.shape[1]
        
        # Apply feature selection method
        if self.method == 'mutual_info':
            self._fit_mutual_info(X, y)
            
        elif self.method == 'anova':
            self._fit_anova(X, y)
            
        elif self.method == 'rfe':
            self._fit_rfe(X, y)
            
        elif self.method == 'lasso':
            self._fit_lasso(X, y)
            
        elif self.method == 'random_forest':
            self._fit_random_forest(X, y)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted_ = True
        
        print(f"\n✓ Selected {len(self.selected_features_)} features")
        print(f"\nTop 10 features:")
        for i, (feat, score) in enumerate(
            list(zip(self.selected_features_[:10], 
                    self.feature_scores_[:10])), 1
        ):
            print(f"  {i:2d}. {feat:40s} (score: {score:.4f})")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted selector
        
        Args:
            X: Features to transform (n_samples, n_features)
            
        Returns:
            Transformed features (n_samples, n_selected_features)
        """
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted before transform")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but selector was fitted with "
                f"{self.n_features_in_} features"
            )
        
        # Select features by indices
        X_selected = X[:, self.selected_indices_]
        
        # Apply scaling if fitted (for Lasso)
        if self.scaler is not None and self.method == 'lasso':
            X_selected = self.scaler.transform(X_selected)
        
        return X_selected
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                     feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step (only for training data!)
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def _fit_mutual_info(self, X: np.ndarray, y: np.ndarray):
        """Fit using mutual information"""
        from sklearn.feature_selection import mutual_info_classif
        
        n_features = min(self.n_features, X.shape[1])
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Select top k features
        self.selected_indices_ = np.argsort(mi_scores)[-n_features:][::-1]
        self.selected_features_ = self.feature_names_in_[self.selected_indices_]
        self.feature_scores_ = mi_scores[self.selected_indices_]
    
    def _fit_anova(self, X: np.ndarray, y: np.ndarray):
        """Fit using ANOVA F-test"""
        from sklearn.feature_selection import f_classif
        
        n_features = min(self.n_features, X.shape[1])
        
        # Calculate F-scores
        f_scores, _ = f_classif(X, y)
        
        # Handle any NaN values
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        
        # Select top k features
        self.selected_indices_ = np.argsort(f_scores)[-n_features:][::-1]
        self.selected_features_ = self.feature_names_in_[self.selected_indices_]
        self.feature_scores_ = f_scores[self.selected_indices_]
    
    def _fit_rfe(self, X: np.ndarray, y: np.ndarray):
        """Fit using Recursive Feature Elimination"""
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Use RFECV to find optimal number of features
        selector = RFECV(
            estimator,
            step=1,
            cv=5,
            scoring='roc_auc',
            min_features_to_select=min(10, self.n_features),
            n_jobs=-1
        )
        selector.fit(X, y)
        
        # Get selected feature indices
        self.selected_indices_ = np.where(selector.support_)[0]
        
        # If RFECV selected more features than requested, keep top n
        if len(self.selected_indices_) > self.n_features:
            # Use feature importance from fitted estimator
            importances = selector.estimator_.feature_importances_
            selected_importances = importances[self.selected_indices_]
            top_n_idx = np.argsort(selected_importances)[-self.n_features:][::-1]
            self.selected_indices_ = self.selected_indices_[top_n_idx]
        
        self.selected_features_ = self.feature_names_in_[self.selected_indices_]
        
        # Get ranking as scores (lower is better, so invert)
        rankings = selector.ranking_[self.selected_indices_]
        self.feature_scores_ = 1.0 / rankings
        
        # Sort by score
        sorted_idx = np.argsort(self.feature_scores_)[::-1]
        self.selected_indices_ = self.selected_indices_[sorted_idx]
        self.selected_features_ = self.selected_features_[sorted_idx]
        self.feature_scores_ = self.feature_scores_[sorted_idx]
        
        print(f"  RFE optimal features: {selector.n_features_}")
    
    def _fit_lasso(self, X: np.ndarray, y: np.ndarray):
        """Fit using Lasso regularization - CORRECTED VERSION"""
        
        # Step 1: Fit LassoCV to find optimal alpha
        lasso_cv = LassoCV(
            cv=5,
            random_state=42,
            max_iter=10000,
            n_jobs=-1
        )
        lasso_cv.fit(X, y)
        
        print(f"  Lasso optimal alpha: {lasso_cv.alpha_:.4f}")
        
        # Step 2: Get coefficients
        coef_abs = np.abs(lasso_cv.coef_)
        
        # Step 3: Select features with non-zero coefficients
        non_zero_mask = coef_abs > 1e-10
        
        if non_zero_mask.sum() == 0:
            print("  WARNING: No features with non-zero coefficients")
            print("  Selecting top features by coefficient magnitude")
            n_select = min(self.n_features, X.shape[1])
            self.selected_indices_ = np.argsort(coef_abs)[-n_select:][::-1]
        elif non_zero_mask.sum() < self.n_features:
            print(f"  Only {non_zero_mask.sum()} non-zero coefficients")
            print(f"  Selecting all {non_zero_mask.sum()} features")
            self.selected_indices_ = np.where(non_zero_mask)[0]
            # Sort by coefficient magnitude
            sorted_idx = np.argsort(coef_abs[self.selected_indices_])[::-1]
            self.selected_indices_ = self.selected_indices_[sorted_idx]
        else:
            # More non-zero features than requested, select top n
            n_select = min(self.n_features, non_zero_mask.sum())
            self.selected_indices_ = np.argsort(coef_abs)[-n_select:][::-1]
        
        self.selected_features_ = self.feature_names_in_[self.selected_indices_]
        self.feature_scores_ = coef_abs[self.selected_indices_]
        
        # Step 4: Fit scaler on selected features only
        # This will be used in transform()
        self.scaler = StandardScaler()
        self.scaler.fit(X[:, self.selected_indices_])
    
    def _fit_random_forest(self, X: np.ndarray, y: np.ndarray):
        """Fit using Random Forest feature importance"""
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top n_features
        n_features = min(self.n_features, X.shape[1])
        self.selected_indices_ = np.argsort(importances)[-n_features:][::-1]
        
        self.selected_features_ = self.feature_names_in_[self.selected_indices_]
        self.feature_scores_ = importances[self.selected_indices_]
    
    def get_support(self, indices: bool = False):
        """
        Get mask or indices of selected features
        
        Args:
            indices: If True, return indices; if False, return boolean mask
            
        Returns:
            Boolean mask or integer indices
        """
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        
        if indices:
            return self.selected_indices_.copy()
        
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask
    
    def save(self, output_path: str):
        """Save fitted selector"""
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted selector")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, output_path)
        print(f"✓ Selector saved to: {output_path}")
    
    @staticmethod
    def load(input_path: str) -> 'RobustFeatureSelector':
        """Load fitted selector"""
        selector = joblib.load(input_path)
        print(f"✓ Selector loaded from: {input_path}")
        print(f"  Method: {selector.method}")
        print(f"  Selected features: {len(selector.selected_features_)}")
        return selector

class SpatialRadiomicsClassifier:
    """空间放射组学分类模型"""
    
    def __init__(self, features_csv, labels_excel, output_dir='model_results'):
        """
        Parameters:
        -----------
        features_csv : str
            特征表CSV路径
        labels_excel : str
            标签表Excel路径
        output_dir : str
            结果输出目录
        """
        self.features_csv = features_csv
        self.labels_excel = labels_excel
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载数据
        self.df_features = None
        self.df_labels = None
        self.df_merged = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = None
        self.best_model = None
        
    def load_and_merge_data(self):
        """加载并合并特征和标签"""
        print("="*60)
        print("STEP 1: LOADING AND MERGING DATA")
        print("="*60)
        
        # 加载特征
        self.df_features = pd.read_csv(self.features_csv)
        print(f"\n✓ Loaded features: {self.df_features.shape}")
        print(f"  Columns: {len(self.df_features.columns)}")
        
        # 加载标签
        self.df_labels = pd.read_excel(self.labels_excel)
        print(f"\n✓ Loaded labels: {self.df_labels.shape}")
        print(f"  Columns: {list(self.df_labels.columns)}")
        
        # 标准化列名
        self.df_labels.columns = ['case_id', 'response']
        
        # 确保case_id类型一致
        self.df_features['case_id'] = self.df_features['case_id'].astype(str).apply(
        lambda x: x.split('.nii')[0] if '.nii' in x else x
        )
        self.df_labels['case_id'] = self.df_labels['case_id'].astype(str)
        
        # 合并数据
        self.df_merged = pd.merge(
            self.df_features,
            self.df_labels,
            on='case_id',
            how='inner'
        )
        
        print(f"\n✓ Merged dataset: {self.df_merged.shape}")
        print(f"  Matched cases: {len(self.df_merged)}")
        print(f"  Features lost: {len(self.df_features) - len(self.df_merged)}")
        print(f"  Labels without features: {len(self.df_labels) - len(self.df_merged)}")
        
        # 检查标签分布
        print(f"\n📊 Response distribution:")
        response_counts = self.df_merged['response'].value_counts()
        for label, count in response_counts.items():
            percentage = count / len(self.df_merged) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        return self.df_merged
    
    def prepare_features(self, exclude_cols=None, handle_missing='drop'):
        """
        准备特征矩阵
        
        Parameters:
        -----------
        exclude_cols : list, optional
            要排除的列名列表
        handle_missing : str
            处理缺失值的方法: 'drop', 'mean', 'median'
        """
        print("\n" + "="*60)
        print("STEP 2: FEATURE PREPARATION")
        print("="*60)
        
        if exclude_cols is None:
            exclude_cols = [
                'case_id', 
                'response', 
                'success', 
                'error_message',
                'total_textures_processed',
                'successful_textures'
            ]
        
        # 选择特征列
        feature_cols = [col for col in self.df_merged.columns 
                       if col not in exclude_cols]
        
        # 只保留数值型特征
        numeric_cols = self.df_merged[feature_cols].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        print(f"\n✓ Total numeric features: {len(numeric_cols)}")
        
        # 处理缺失值
        X_raw = self.df_merged[numeric_cols].copy()
        
        missing_summary = X_raw.isnull().sum()
        missing_features = missing_summary[missing_summary > 0]
        
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
        
        self.X = X_raw
        self.feature_names = X_raw.columns.tolist()
        
        # 获取对应的标签
        self.y = self.df_merged.loc[X_raw.index, 'response'].values
        self.y = [0 if (isinstance(x, str) and x == 'pCR') or x == 0 else 1 for x in self.y]

        print(f"\n✓ Final feature matrix: {self.X.shape}")
        print(f"  Samples: {len(self.X)}")
        print(f"  Features: {len(self.feature_names)}")
        
        return self.X, self.y
    
    
    # def feature_selection(self, method='mutual_info', n_features=30):
    #     """
    #     特征选择
        
    #     Parameters:
    #     -----------
    #     method : str
    #         特征选择方法: 'mutual_info', 'anova', 'rfe', 'lasso'
    #     n_features : int
    #         要选择的特征数量
    #     """
    #     print("\n" + "="*60)
    #     print(f"STEP 3: FEATURE SELECTION ({method.upper()})")
    #     print("="*60)
        
    #     if method == 'mutual_info':
    #         # 互信息
    #         selector = SelectKBest(
    #             score_func=mutual_info_classif, 
    #             k=min(n_features, X.shape[1])
    #         )
    #         selector.fit(X, y)
            
    #         scores = pd.DataFrame({
    #             'feature': self.feature_names,
    #             'score': selector.scores_
    #         }).sort_values('score', ascending=False)
            
    #     elif method == 'anova':
    #         # ANOVA F-test
    #         selector = SelectKBest(
    #             score_func=f_classif, 
    #             k=min(n_features, X.shape[1])
    #         )
    #         selector.fit(X, y)
            
    #         scores = pd.DataFrame({
    #             'feature': self.feature_names,
    #             'score': selector.scores_
    #         }).sort_values('score', ascending=False)
            
    #     elif method == 'rfe':
    #         # 递归特征消除
    #         estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    #         selector = RFECV(
    #             estimator, 
    #             step=1, 
    #             cv=5,
    #             scoring='roc_auc',
    #             min_features_to_select=min(10, n_features)
    #         )
    #         selector.fit(X, y)
            
    #         scores = pd.DataFrame({
    #             'feature': self.feature_names,
    #             'ranking': selector.ranking_,
    #             'selected': selector.support_
    #         }).sort_values('ranking')
            
    #         n_features = selector.n_features_
    #         print(f"\n✓ Optimal number of features: {n_features}")
            
    #     elif method == 'lasso':
    #         # Lasso正则化
    #         from sklearn.linear_model import LassoCV
            
    #         # 标准化
    #         X_scaled = StandardScaler().fit_transform(X)
            
    #         lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    #         lasso.fit(X_scaled, y)
            
    #         scores = pd.DataFrame({
    #             'feature': self.feature_names,
    #             'coefficient': np.abs(lasso.coef_)
    #         }).sort_values('coefficient', ascending=False)
            
    #         scores = scores[scores['coefficient'] > 0]
    #         n_features = min(n_features, len(scores))
        
    #     # 选择top特征
    #     if method == 'rfe':
    #         selected_features = scores[scores['selected']]['feature'].tolist()
    #     else:
    #         selected_features = scores.head(n_features)['feature'].tolist()
        
    #     print(f"\n✓ Selected {len(selected_features)} features")
    #     print(f"\nTop 10 features:")
    #     for i, feat in enumerate(selected_features[:10], 1):
    #         print(f"  {i}. {feat}")
        
    #     # 可视化特征重要性
    #     self._plot_feature_importance(scores, method, n_top=20)
        
    #     # 更新特征矩阵
    #     X = X[selected_features]
    #     self.feature_names = selected_features
        
    #     return selected_features
    
    def _plot_feature_importance(self, scores, method, n_top=20):
        """绘制特征重要性图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if method == 'rfe':
            plot_data = scores[scores['selected']].head(n_top)
            x_col = 'ranking'
            title = 'Feature Ranking (RFE)'
        else:
            plot_data = scores.head(n_top)
            x_col = 'score' if 'score' in scores.columns else 'coefficient'
            title = f'Top {n_top} Features ({method.upper()})'
        
        y_pos = np.arange(len(plot_data))
        ax.barh(y_pos, plot_data[x_col].values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data['feature'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'feature_importance_{method}.png', 
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def train_models(self, test_size=0.2, random_state=42, n_features=3):
        """
        训练多个分类模型
        
        Parameters:
        -----------
        test_size : float
            测试集比例
        random_state : int
            随机种子
        """
        print("\n" + "="*60)
        print("STEP 4: MODEL TRAINING AND EVALUATION")
        print("="*60)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
                
        # 数据标准化
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"\n✓ Data split:")
        print(f"  Training (before SMOTE): {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")

        # Lasso正则化
        # selector = FeatureSelector(method='lasso', n_features=n_features)
        # estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        # estimator = SVC(kernel='linear', random_state=42)
        estimator = Lasso(alpha=0.001, random_state=42, max_iter=10000)
        selector = RFE(estimator, n_features_to_select=n_features, step=9)
        # selector = RFECV(
        #     estimator, 
        #     step=1, 
        #     cv=5,
        #     scoring='roc_auc',
        #     min_features_to_select=n_features
        # )
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test) 
        n_features = selector.n_features_
        print(f"\n✓ Optimal number of features: {n_features}")
        # Apply SMOTE to balance the training dataset
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        print(f"  Training (after SMOTE): {len(X_train)} samples")
        print(f"  Class distribution (train): {np.bincount(y_train)}")

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
                max_iter=100,
                random_state=random_state,
                early_stopping=True
            ),
            'XGBoost': XGBClassifier(
                n_estimators=5,
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
            model.fit(X_train, y_train)
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            
            # 评估指标
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            # 处理多分类和二分类
            if len(np.unique(self.y)) == 2:
                auc = roc_auc_score(y_test, y_proba_test)
                ap = average_precision_score(y_test, y_proba_test)
            else:
                # 多分类：使用one-vs-rest
                y_proba_test_multi = model.predict_proba(X_test)
                auc = roc_auc_score(
                    y_test, y_proba_test_multi, 
                    multi_class='ovr', average='weighted'
                )
                ap = None
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='accuracy'
            )
            
            results[model_name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'auc': auc,
                'average_precision': ap,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_test': y_pred_test,
                'y_proba_test': y_proba_test
            }
            
            print(f"\n  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  AUC:           {auc:.4f}")
            if ap is not None:
                print(f"  Avg Precision: {ap:.4f}")
            print(f"  CV Accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 保存结果
        self.results = results
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # 选择最佳模型（基于测试集AUC）
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n{'='*60}")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"   Test AUC: {results[best_model_name]['auc']:.4f}")
        print(f"{'='*60}")
        
        return results
    
    def hyperparameter_tuning(self, model_type='RandomForest', cv=5):
        """
        超参数优化
        
        Parameters:
        -----------
        model_type : str
            模型类型
        cv : int
            交叉验证折数
        """
        print("\n" + "="*60)
        print(f"STEP 5: HYPERPARAMETER TUNING ({model_type})")
        print("="*60)
        
        X_scaled = self.scaler.transform(self.X)
        
        if model_type == 'RandomForest':
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_type == 'SVM':
            model = SVC(probability=True, random_state=42, class_weight='balanced')
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        elif model_type == 'MLP':
            model = MLPClassifier(max_iter=2000, random_state=42)
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        elif model_type == 'XGBoost':
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 网格搜索
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, self.y)
        
        print(f"\n✓ Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\n✓ Best CV AUC: {grid_search.best_score_:.4f}")
        
        # 保存最佳模型
        self.best_model = grid_search.best_estimator_
        self.best_model_name = f"{model_type} (Tuned)"
        
        # 保存参数搜索结果
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(
            self.output_dir / f'hyperparameter_tuning_{model_type}.csv',
            index=False
        )
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_model(self, model=None, model_name=None):
        """
        详细评估模型
        
        Parameters:
        -----------
        model : sklearn model, optional
            要评估的模型（默认使用最佳模型）
        model_name : str, optional
            模型名称
        """
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        
        print("\n" + "="*60)
        print(f"DETAILED EVALUATION: {model_name}")
        print("="*60)
        
        # 预测
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        
        # 分类报告
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        
        # 可视化结果
        self._plot_evaluation_results(
            self.y_test, y_pred, y_proba, cm, model_name
        )
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'case_id': self.df_merged.loc[self.X_test.index, 'case_id'].values,
            'true_label': self.y_test,
            'predicted_label': y_pred,
            'probability': y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(axis=1)
        })
        
        results_df.to_csv(
            self.output_dir / f'predictions_{model_name.replace(" ", "_")}.csv',
            index=False
        )
        
        print(f"\n✓ Results saved to {self.output_dir}")
        
        return results_df
    
    def _plot_evaluation_results(self, y_true, y_pred, y_proba, cm, model_name):
        """绘制评估结果图"""
        
        # 创建2x2子图
        fig = plt.figure(figsize=(15, 12))
        
        # 1. 混淆矩阵
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC曲线
        ax2 = plt.subplot(2, 2, 2)
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc = roc_auc_score(y_true, y_proba[:, 1])
            
            ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
            ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend()
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Multi-class ROC\n(see individual plots)', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Precision-Recall曲线
        ax3 = plt.subplot(2, 2, 3)
        if len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            ap = average_precision_score(y_true, y_proba[:, 1])
            
            ax3.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.set_title('Precision-Recall Curve')
            ax3.legend()
            ax3.grid(alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Multi-class PR\n(see individual plots)', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 预测概率分布
        ax4 = plt.subplot(2, 2, 4)
        if len(np.unique(y_true)) == 2:
            for label in np.unique(y_true):
                mask = y_true == label
                ax4.hist(y_proba[mask, 1], bins=20, alpha=0.6, 
                        label=f'True Label {label}')
            ax4.set_xlabel('Predicted Probability')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Prediction Probability Distribution')
            ax4.legend()
            ax4.grid(alpha=0.3)
        else:
            # 多分类：显示每个类别的概率分布
            for i in range(y_proba.shape[1]):
                ax4.hist(y_proba[:, i], bins=20, alpha=0.5, label=f'Class {i}')
            ax4.set_xlabel('Predicted Probability')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Prediction Probability Distribution (All Classes)')
            ax4.legend()
            ax4.grid(alpha=0.3)
        
        plt.suptitle(f'Model Evaluation: {model_name}', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'evaluation_{model_name.replace(" ", "_")}.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def plot_feature_importance_for_model(self, model=None, model_name=None, top_n=20):
        """
        绘制模型的特征重要性
        
        Parameters:
        -----------
        model : sklearn model, optional
            要分析的模型
        model_name : str, optional
            模型名称
        top_n : int
            显示前N个重要特征
        """
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        
        print(f"\n{'='*60}")
        print(f"Feature Importance Analysis: {model_name}")
        print(f"{'='*60}")
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            # 树模型
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 线性模型
            importances = np.abs(model.coef_[0])
        else:
            print("  Model does not support feature importance extraction")
            return
        
        # 创建DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 保存
        feature_importance.to_csv(
            self.output_dir / f'model_feature_importance_{model_name.replace(" ", "_")}.csv',
            index=False
        )
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = feature_importance.head(top_n)
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_features['importance'].values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'model_feature_importance_{model_name.replace(" ", "_")}.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print(f"\n📊 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    def cross_validation_analysis(self, model=None, cv=5):
        """
        详细的交叉验证分析
        
        Parameters:
        -----------
        model : sklearn model, optional
            要评估的模型
        cv : int
            交叉验证折数
        """
        if model is None:
            model = self.best_model
        
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ANALYSIS (CV={cv})")
        print(f"{'='*60}")
        
        X_scaled = self.scaler.transform(self.X)
        
        # 设置交叉验证
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_results = {
            'fold': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_auc': [],
            'val_auc': []
        }
        
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, self.y), 1):
            X_train_cv = X_scaled[train_idx]
            X_val_cv = X_scaled[val_idx]
            y_train_cv = self.y[train_idx]
            y_val_cv = self.y[val_idx]
            
            # 训练模型
            from sklearn.base import clone
            model_cv = clone(model)
            model_cv.fit(X_train_cv, y_train_cv)
            
            # 预测
            y_train_pred = model_cv.predict(X_train_cv)
            y_val_pred = model_cv.predict(X_val_cv)
            y_val_proba = model_cv.predict_proba(X_val_cv)
            
            # 计算指标
            train_acc = accuracy_score(y_train_cv, y_train_pred)
            val_acc = accuracy_score(y_val_cv, y_val_pred)
            
            if len(np.unique(self.y)) == 2:
                train_auc = roc_auc_score(y_train_cv, model_cv.predict_proba(X_train_cv)[:, 1])
                val_auc = roc_auc_score(y_val_cv, y_val_proba[:, 1])
            else:
                train_auc = roc_auc_score(
                    y_train_cv, model_cv.predict_proba(X_train_cv),
                    multi_class='ovr', average='weighted'
                )
                val_auc = roc_auc_score(
                    y_val_cv, y_val_proba,
                    multi_class='ovr', average='weighted'
                )
            
            cv_results['fold'].append(fold)
            cv_results['train_accuracy'].append(train_acc)
            cv_results['val_accuracy'].append(val_acc)
            cv_results['train_auc'].append(train_auc)
            cv_results['val_auc'].append(val_auc)
            
            all_y_true.extend(y_val_cv)
            all_y_pred.extend(y_val_pred)
            all_y_proba.extend(y_val_proba[:, 1] if len(np.unique(self.y)) == 2 else y_val_proba.max(axis=1))
            
            print(f"\nFold {fold}:")
            print(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        
        # 汇总结果
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(self.output_dir / 'cross_validation_results.csv', index=False)
        
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY:")
        print(f"{'='*60}")
        print(f"Mean Val Accuracy: {cv_df['val_accuracy'].mean():.4f} ± {cv_df['val_accuracy'].std():.4f}")
        print(f"Mean Val AUC:      {cv_df['val_auc'].mean():.4f} ± {cv_df['val_auc'].std():.4f}")
        
        # 可视化CV结果
        self._plot_cv_results(cv_df)
        
        return cv_df
    
    def _plot_cv_results(self, cv_df):
        """可视化交叉验证结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准确率
        ax1 = axes[0]
        x = cv_df['fold']
        ax1.plot(x, cv_df['train_accuracy'], 'o-', label='Train', linewidth=2, markersize=8)
        ax1.plot(x, cv_df['val_accuracy'], 's-', label='Validation', linewidth=2, markersize=8)
        ax1.axhline(cv_df['val_accuracy'].mean(), color='red', linestyle='--', 
                   label=f'Mean Val: {cv_df["val_accuracy"].mean():.3f}', alpha=0.7)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Cross-Validation Accuracy')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # AUC
        ax2 = axes[1]
        ax2.plot(x, cv_df['train_auc'], 'o-', label='Train', linewidth=2, markersize=8)
        ax2.plot(x, cv_df['val_auc'], 's-', label='Validation', linewidth=2, markersize=8)
        ax2.axhline(cv_df['val_auc'].mean(), color='red', linestyle='--',
                   label=f'Mean Val: {cv_df["val_auc"].mean():.3f}', alpha=0.7)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('AUC')
        ax2.set_title('Cross-Validation AUC')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_validation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """生成完整的分析报告"""
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*60}")
        
        report_path = self.output_dir / 'classification_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SPATIAL RADIOMICS CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # 数据概览
            f.write("1. DATA OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Total samples: {len(self.df_merged)}\n")
            f.write(f"Total features: {len(self.feature_names)}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Testing samples: {len(self.X_test)}\n\n")
            
            f.write("Response distribution:\n")
            for label, count in pd.Series(self.y).value_counts().items():
                f.write(f"  {label}: {count} ({count/len(self.y)*100:.1f}%)\n")
            f.write("\n")
            
            # 特征信息
            f.write("2. FEATURE INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Selected features: {len(self.feature_names)}\n")
            f.write("\nTop 10 features:\n")
            for i, feat in enumerate(self.feature_names[:10], 1):
                f.write(f"  {i}. {feat}\n")
            f.write("\n")
            
            # 模型性能
            f.write("3. MODEL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            if hasattr(self, 'results'):
                for model_name, result in self.results.items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n")
                    f.write(f"  AUC: {result['auc']:.4f}\n")
                    if result['average_precision'] is not None:
                        f.write(f"  Average Precision: {result['average_precision']:.4f}\n")
                    f.write(f"  CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n")
            
            f.write(f"\n\nBest Model: {self.best_model_name}\n")
            f.write("\n")
            
            # 详细评估
            f.write("4. DETAILED EVALUATION (Best Model)\n")
            f.write("-"*80 + "\n")
            y_pred = self.best_model.predict(self.X_test)
            f.write("\nClassification Report:\n")
            f.write(classification_report(self.y_test, y_pred))
            
            f.write("\n\nConfusion Matrix:\n")
            cm = confusion_matrix(self.y_test, y_pred)
            f.write(str(cm))
            f.write("\n")
        
        print(f"\n✓ Report saved to: {report_path}")
        
        return report_path


# ============================================================================
# 完整工作流示例
# ============================================================================

def main():
    """完整的分类模型开发流程"""
    
    # 初始化分类器
    classifier = SpatialRadiomicsClassifier(
        features_csv='XXXyour path/Dataset/Dataset002_ECResponse/spatial_texture_features.csv',
        labels_excel='XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx',
        output_dir='XXXyour path/Dataset/Dataset002_ECResponse/model_results'
    )
    
    # 步骤1: 加载和合并数据
    df_merged = classifier.load_and_merge_data()
    
    # 步骤2: 准备特征
    X, y = classifier.prepare_features(handle_missing='drop')
    
    # 步骤3: 特征选择（尝试多种方法）
    print("\n" + "="*80)
    print("TRYING MULTIPLE FEATURE SELECTION METHODS")
    print("="*80)
    
    # # # 方法1: 互信息
    # selected_features_mi = classifier.feature_selection(method='mutual_info', n_features=6)
    
    # # 方法2: ANOVA (重新准备特征以重置)
    # classifier.prepare_features(handle_missing='drop')
    # selected_features_anova = classifier.feature_selection(method='anova', n_features=30)
    
    # 方法3: RFE (递归特征消除)
    # classifier.prepare_features(handle_missing='drop')
    # selected_features_rfe = classifier.feature_selection(method='rfe', n_features=3)
    
    # # # 使用互信息选择的特征继续
    # # classifier.prepare_features(handle_missing='drop')
    # # classifier.feature_selection(method='mutual_info', n_features=30)
    
    # # 步骤4: 训练多个模型
    results = classifier.train_models(test_size=0.3, random_state=42, n_features=18)
    
    # # 步骤5: 超参数优化（针对最佳模型）
    # print("\n" + "="*80)
    # print("HYPERPARAMETER TUNING")
    # print("="*80)
    
    # # 假设Random Forest表现最好
    # best_model_tuned, best_params = classifier.hyperparameter_tuning(
    #     model_type='RandomForest', 
    #     cv=5
    # )
    
    # # 步骤6: 详细评估
    # # 重新训练测试集划分
    # classifier.train_models(test_size=0.3, random_state=42)
    # predictions_df = classifier.evaluate_model()
    
    # # 步骤7: 特征重要性分析
    # feature_importance = classifier.plot_feature_importance_for_model(top_n=20)
    
    # # 步骤8: 交叉验证分析
    # cv_results = classifier.cross_validation_analysis(cv=5)
    
    # # 步骤9: 生成报告
    # report_path = classifier.generate_report()
    
    # print("\n" + "="*80)
    # print("🎉 CLASSIFICATION PIPELINE COMPLETED!")
    # print("="*80)
    # print(f"\nAll results saved to: {classifier.output_dir}")
    # print("\nGenerated files:")
    # for file in sorted(classifier.output_dir.glob('*')):
    #     print(f"  - {file.name}")


if __name__ == "__main__":
    main()