"""
Evaluate the Dataset002_ECResponse classification model (Spatial + Radiomics
+ Score Fusion) on the new SPCH validation cohort, and generate model
performance + survival analysis figures.

IMPORTANT: code/SpatialFeatureExtraction/ComparsionClassificationModel.py
(the script that "developed" this model) never persists a trained model --
no joblib.dump/json.dump despite importing both. So this script first
reproduces its exact training procedure (same random_state=42, test_size=0.3,
RFE(Ridge) feature selection, XGBClassifier hyperparameters, scale=0.7 score
fusion) on the original Dataset002_ECResponse cohort, then scores SPCH with
the resulting fitted scaler/selector/model -- mirroring exactly how that
script already handles its other external cohort (TJH).

One correction vs. the original script: it calls
`scaler.fit_transform(X_TJH)` for the spatial model's external cohorts
(TJH/scRNA), i.e. it refits the scaler on the external data instead of
reusing the train-fitted one -- a leakage bug (the radiomics model in the
same file correctly uses `.transform()` only). This script uses
`.transform()` for SPCH throughout, for both models.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / 'survival_analysis'))
from PlotFigures import MedicalImagingVisualizer
from PlotSurvivalAnalysis import SurvivalAnalyzer

RANDOM_STATE = 42
TEST_SIZE = 0.3
N_SPATIAL_FEATURES = 7
SPATIAL_RFE_STEP = 11
N_RADIOMICS_FEATURES = 3
RADIOMICS_RFE_STEP = 19
FUSION_SCALE = 0.7  # weight on spatial score, matching the original script

DATA_ROOT = Path('XXXyour path/Dataset/Dataset002_ECResponse')
SPCH_DIR = DATA_ROOT / 'SPCH'
SPCH_MODEL_RESULTS = SPCH_DIR / 'model_results'
SPCH_MODEL_RESULTS.mkdir(parents=True, exist_ok=True)


def strip_nii(case_id):
    s = str(case_id).strip()
    return s.split('.nii')[0] if '.nii' in s else s


def response_to_binary(values):
    """pCR -> 0, everything else -> 1 (matches ComparsionClassificationModel.py)."""
    return [0 if (isinstance(v, str) and v == 'pCR') or v == 0 else 1 for v in values]


print("=" * 70)
print("STEP 1: REPRODUCE THE Dataset002_ECResponse MODEL (Spatial + Radiomics)")
print("=" * 70)

# ── Spatial model: spatial3d + texture features ────────────────────────────
df_features = pd.read_csv(DATA_ROOT / 'spatial3d_features.csv')
df_texture_features = pd.read_csv(DATA_ROOT / 'spatial_texture_features.csv')
df_labels = pd.read_excel(DATA_ROOT / 'PatientLabels.xlsx')
df_labels = df_labels.drop(columns=['DFS', 'DFS-Time', 'OS', 'OS-Time'])
df_labels.columns = ['case_id', 'response']

df_features['case_id'] = df_features['case_id'].apply(strip_nii)
df_texture_features['case_id'] = df_texture_features['case_id'].apply(strip_nii)
df_labels['case_id'] = df_labels['case_id'].astype(str)

df_merged = pd.merge(df_features, df_labels, on='case_id', how='inner')
df_merged = pd.merge(df_texture_features, df_merged, on='case_id', how='inner')
print(f"  Merged Dataset002 spatial cohort: {len(df_merged)} cases")

exclude_cols = ['case_id', 'response', 'success', 'error_message',
                'total_textures_processed', 'successful_textures']
feature_cols = [c for c in df_merged.columns if c not in exclude_cols]
numeric_cols = df_merged[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X_raw = df_merged[numeric_cols].copy()
missing_summary = X_raw.isnull().sum()
drop_features = missing_summary[missing_summary > len(X_raw) * 0.5].index
X_raw = X_raw.drop(columns=drop_features)
before = len(X_raw)
X_raw = X_raw.dropna()
print(f"  Dropped {len(drop_features)} features >50% missing, "
      f"{before - len(X_raw)} rows with remaining missing values")

variances = X_raw.var()
X_raw = X_raw.drop(columns=variances[variances == 0].index)

corr_matrix = X_raw.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
X_raw = X_raw.drop(columns=to_drop)

X = X_raw
feature_names = X.columns.tolist()
y = response_to_binary(df_merged.loc[X.index, 'response'].values)
case_ids_all = df_merged.loc[X.index, 'case_id'].values
print(f"  Final spatial feature matrix: {X.shape}")

X_train, X_test, y_train, y_test, case_ids_train, case_ids_test = train_test_split(
    X, y, case_ids_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(y_train)}  Test: {len(y_test)}")

spatial_scaler = StandardScaler().fit(X_train)
X_train_scaled = spatial_scaler.transform(X_train)
X_test_scaled = spatial_scaler.transform(X_test)

spatial_estimator = Ridge(alpha=0.1, random_state=RANDOM_STATE, max_iter=1000)
spatial_selector = RFE(spatial_estimator, n_features_to_select=N_SPATIAL_FEATURES, step=SPATIAL_RFE_STEP)
X_train_selected = spatial_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = spatial_selector.transform(X_test_scaled)
selected_spatial_names = [feature_names[i] for i, k in enumerate(spatial_selector.support_) if k]
print(f"  Selected {len(selected_spatial_names)} spatial features:")
for f in selected_spatial_names:
    print(f"    - {f}")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_final, y_train_final = smote.fit_resample(X_train_selected, y_train)
spatial_model = XGBClassifier(n_estimators=83, learning_rate=0.007,
                               random_state=RANDOM_STATE, eval_metric='logloss')
spatial_model.fit(X_train_final, y_train_final)

proba_sp_train = spatial_model.predict_proba(X_train_selected)[:, 1]
proba_sp_test = spatial_model.predict_proba(X_test_selected)[:, 1]
print(f"  [Spatial] Train AUC={roc_auc_score(y_train, proba_sp_train):.4f}  "
      f"Test AUC={roc_auc_score(y_test, proba_sp_test):.4f}")

# ── Radiomics model ─────────────────────────────────────────────────────────
rad_df = pd.read_csv(DATA_ROOT / 'radiomics_features.csv')
df_labels2 = pd.read_excel(DATA_ROOT / 'PatientLabels.xlsx')
df_labels2 = df_labels2.drop(columns=['DFS', 'DFS-Time', 'OS', 'OS-Time'])
rad_merged = pd.merge(rad_df, df_labels2, on='CaseID', how='inner')
rad_merged = rad_merged.dropna(subset=['Response'])
exclude_diag = [c for c in rad_merged.columns if 'diagnostics' in c.lower()]
rad_merged = rad_merged.drop(columns=exclude_diag)
rad_merged['CaseID'] = rad_merged['CaseID'].astype(str)

rad_train = rad_merged[rad_merged['CaseID'].isin(case_ids_train)]
rad_test = rad_merged[rad_merged['CaseID'].isin(case_ids_test)]
X_train_rad = rad_train.drop(columns=['CaseID', 'Response'])
X_test_rad = rad_test.drop(columns=['CaseID', 'Response'])
rad_feature_names = X_train_rad.columns.tolist()

rad_scaler = StandardScaler().fit(X_train_rad)
X_train_rad_scaled = rad_scaler.transform(X_train_rad)
X_test_rad_scaled = rad_scaler.transform(X_test_rad)

rad_estimator = Ridge(alpha=1, random_state=RANDOM_STATE)
rad_selector = RFE(rad_estimator, n_features_to_select=N_RADIOMICS_FEATURES, step=RADIOMICS_RFE_STEP)
X_train_rad_selected = rad_selector.fit_transform(X_train_rad_scaled, y_train)
X_test_rad_selected = rad_selector.transform(X_test_rad_scaled)
selected_rad_names = [rad_feature_names[i] for i, k in enumerate(rad_selector.support_) if k]
print(f"\n  Selected {len(selected_rad_names)} radiomics features:")
for f in selected_rad_names:
    print(f"    - {f}")

smote2 = SMOTE(random_state=RANDOM_STATE)
X_train_rad_final, y_train_rad_final = smote2.fit_resample(X_train_rad_selected, y_train)
rad_model = XGBClassifier(n_estimators=100, learning_rate=0.14,
                           random_state=RANDOM_STATE, eval_metric='logloss')
rad_model.fit(X_train_rad_final, y_train_rad_final)

proba_rad_train = rad_model.predict_proba(X_train_rad_selected)[:, 1]
proba_rad_test = rad_model.predict_proba(X_test_rad_selected)[:, 1]
print(f"  [Radiomics] Train AUC={roc_auc_score(y_train, proba_rad_train):.4f}  "
      f"Test AUC={roc_auc_score(y_test, proba_rad_test):.4f}")

proba_fused_test = FUSION_SCALE * proba_sp_test + (1 - FUSION_SCALE) * proba_rad_test
print(f"\n  [Fusion, spatial weight={FUSION_SCALE}] Test AUC="
      f"{roc_auc_score(y_test, proba_fused_test):.4f}  "
      f"ACC={accuracy_score(y_test, proba_fused_test > 0.5):.4f}")

print("\n" + "=" * 70)
print("STEP 2: LOAD & MERGE SPCH (external validation)")
print("=" * 70)

spch_spatial3d = pd.read_csv(SPCH_DIR / 'spatial3d_features.csv')
spch_texture = pd.read_csv(SPCH_DIR / 'spatial_texture_features.csv')
spch_rad = pd.read_csv(SPCH_DIR / 'radiomics_features.csv')
spch_labels_full = pd.read_excel(SPCH_DIR / 'PatientLabels.xlsx')

spch_spatial3d['case_id'] = spch_spatial3d['case_id'].apply(strip_nii)
spch_texture['case_id'] = spch_texture['case_id'].apply(strip_nii)
spch_rad = spch_rad.rename(columns={'CaseID': 'case_id'})
spch_rad['case_id'] = spch_rad['case_id'].astype(str)
spch_rad = spch_rad.drop(columns=[c for c in spch_rad.columns if 'diagnostics' in c.lower()])

spch_labels_cls = spch_labels_full[['CaseID', 'Response']].rename(
    columns={'CaseID': 'case_id', 'Response': 'response'})
spch_labels_cls['case_id'] = spch_labels_cls['case_id'].astype(str)

# Inner-join everything so spatial & radiomics predictions land on the SAME
# case set (otherwise the two probability arrays wouldn't align for fusion).
spch_all = pd.merge(spch_spatial3d, spch_texture, on='case_id', how='inner', suffixes=('', '_tex'))
spch_all = pd.merge(spch_all, spch_rad, on='case_id', how='inner', suffixes=('', '_rad'))
spch_all = pd.merge(spch_all, spch_labels_cls, on='case_id', how='inner')
spch_all = spch_all.drop_duplicates(subset='case_id').reset_index(drop=True)
print(f"  SPCH cases with complete spatial+texture+radiomics+label data: {len(spch_all)}")

missing_mask = spch_all.reindex(columns=feature_names).isnull().any(axis=1) | \
               spch_all.reindex(columns=rad_feature_names).isnull().any(axis=1)
n_missing = missing_mask.sum()
if n_missing:
    missing_cols = sorted(c for c in feature_names + rad_feature_names
                           if spch_all[c].isnull().any())
    print(f"  {n_missing} SPCH case(s) have NaN in candidate feature columns "
          f"{missing_cols} -- imputing with 0 instead of dropping "
          f"(per explicit request), so all cases are kept for evaluation")
    spch_all[feature_names] = spch_all[feature_names].fillna(0)
    spch_all[rad_feature_names] = spch_all[rad_feature_names].fillna(0)
print(f"  Final SPCH cohort for evaluation: {len(spch_all)} cases")

y_spch = response_to_binary(spch_all['response'].values)
case_ids_spch = spch_all['case_id'].values

X_spch = spch_all[feature_names]
X_spch_scaled = spatial_scaler.transform(X_spch)  # .transform only -- see module docstring
X_spch_selected = spatial_selector.transform(X_spch_scaled)
proba_sp_spch = spatial_model.predict_proba(X_spch_selected)[:, 1]

X_spch_rad = spch_all[rad_feature_names]
X_spch_rad_scaled = rad_scaler.transform(X_spch_rad)
X_spch_rad_selected = rad_selector.transform(X_spch_rad_scaled)
proba_rad_spch = rad_model.predict_proba(X_spch_rad_selected)[:, 1]

proba_fused_spch = FUSION_SCALE * proba_sp_spch + (1 - FUSION_SCALE) * proba_rad_spch

print(f"\n  [Spatial]   SPCH: AUC={roc_auc_score(y_spch, proba_sp_spch):.4f}  "
      f"ACC={accuracy_score(y_spch, proba_sp_spch > 0.5):.4f}  n={len(y_spch)}")
print(f"  [Radiomics] SPCH: AUC={roc_auc_score(y_spch, proba_rad_spch):.4f}  "
      f"ACC={accuracy_score(y_spch, proba_rad_spch > 0.5):.4f}  n={len(y_spch)}")
print(f"  [Fusion]    SPCH: AUC={roc_auc_score(y_spch, proba_fused_spch):.4f}  "
      f"ACC={accuracy_score(y_spch, proba_fused_spch > 0.5):.4f}  n={len(y_spch)}")

# ── Save predictions (schema matches MedicalImagingVisualizer / SurvivalAnalyzer) ──
spch_predictions = pd.DataFrame({
    'case_id': case_ids_spch,
    'true_label': y_spch,
    'true_label_name': ['pCR' if v == 0 else 'Non-pCR' for v in y_spch],
    'spatial_probability': proba_sp_spch,
    'radiomics_probability': proba_rad_spch,
    'fused_probability': proba_fused_spch,
}).sort_values('fused_probability', ascending=False)

pred_path = SPCH_MODEL_RESULTS / 'SPCH_predictions_results.csv'
spch_predictions.to_csv(pred_path, index=False)
print(f"\n  Saved predictions to {pred_path}")

print("\n" + "=" * 70)
print("STEP 3: MODEL PERFORMANCE FIGURES")
print("=" * 70)

visualizer = MedicalImagingVisualizer(
    predictions_csv=str(pred_path),
    output_dir=str(SPCH_DIR / 'visualizations'),
)
visualizer.generate_all_plots()

print("\n" + "=" * 70)
print("STEP 4: SURVIVAL ANALYSIS FIGURES")
print("=" * 70)
print("  NOTE: SurvivalAnalyzer.stratify_all_models() uses hardcoded risk")
print("  thresholds (0.35 / 0.46 / 0.454) baked into PlotSurvivalAnalysis.py,")
print("  apparently tuned for the TJH cohort it was originally written for")
print("  (the log-rank optimal-threshold search is present but commented")
print("  out). These are reused as-is for SPCH here, unchanged -- flagging")
print("  this since SPCH's score distribution may differ from TJH's.")

survival_analyzer = SurvivalAnalyzer(
    predictions_csv=str(pred_path),
    survival_excel=str(SPCH_DIR / 'PatientLabels.xlsx'),
    output_dir=str(SPCH_DIR / 'survival_analysis'),
)
survival_analyzer.generate_all()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
