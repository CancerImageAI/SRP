import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = Path('XXXyour path/Dataset/Dataset004-NACI')

# SHPH is split into train/test; SPCH and FUSCC are held out as external
# validation cohorts (never touch feature selection, scaling, or training).
COHORTS = {
    'SHPH':  dict(feature_dir=DATA_ROOT / 'SHPH',  labels_excel=DATA_ROOT / 'SHPH.xlsx'),
    'SPCH':  dict(feature_dir=DATA_ROOT / 'SPCH',  labels_excel=DATA_ROOT / 'SPCH.xlsx'),
    'FUSCC': dict(feature_dir=DATA_ROOT / 'FUSCC', labels_excel=DATA_ROOT / 'FUSCC.xlsx'),
}

RANDOM_STATE = 42
TEST_SIZE = 0.3

# ── Hyperparameter search ───────────────────────────────────────────────
# Selection metric: mean(AUC_SPCH, AUC_FUSCC), eligible only if EACH of
# those two clears MIN_AUC_FLOOR (stops the mean being won by a config
# that's excellent on one external cohort and mediocre on the other).
# SHPH's Test split is deliberately excluded from selection — it's the
# held-out slice of the training cohort, not an independent cohort, so
# using it to pick hyperparameters would turn it into a validation set
# and bias the "unbiased holdout" AUC we report for it. Test AUC is still
# computed and reported for every candidate, just never drives selection.
# If nothing clears the floor, the search falls back to the best mean
# overall (and says so loudly).
MIN_AUC_FLOOR = 0.55
N_SEARCH_TRIALS = 50000  # randomized-search trials per model (spatial, radiomics)

# Each entry is a (low, high, dtype, scale) range spec sampled fresh per
# trial — 'log' scale for params that sensibly vary over orders of
# magnitude (alpha, learning_rate), 'linear' otherwise. This replaces the
# earlier small fixed discrete grids with continuous ranges, giving the
# search much finer resolution to explore.
#
# n_features/rfe_step/ridge_alpha ranges are as specified (3-20 features,
# step 1-100, alpha 1e-5 to 10). The XGBoost params use ranges chosen for
# reasonable coverage without wasting trials on clearly-bad regions
# (e.g. n_estimators up to 300, not 10000; max_depth up to 8, deep enough
# to overfit this dataset size well before that point).
SPATIAL_PARAM_GRID = {
    'n_features':       (3, 20, 'int', 'linear'),
    'rfe_step':         (1, 100, 'int', 'linear'),
    'ridge_alpha':      (1e-5, 10, 'float', 'log'),
    'ridge_max_iter':   (300, 5000, 'int', 'log'),
    'n_estimators':     (30, 300, 'int', 'linear'),
    'learning_rate':    (0.005, 0.3, 'float', 'log'),
    'max_depth':        (2, 8, 'int', 'linear'),
    'subsample':        (0.5, 1.0, 'float', 'linear'),
    'colsample_bytree': (0.5, 1.0, 'float', 'linear'),
}
RADIOMICS_PARAM_GRID = {
    'n_features':       (3, 20, 'int', 'linear'),
    'rfe_step':         (1, 100, 'int', 'linear'),
    'ridge_alpha':      (1e-5, 10, 'float', 'log'),
    'ridge_max_iter':   (300, 5000, 'int', 'log'),
    'n_estimators':     (30, 300, 'int', 'linear'),
    'learning_rate':    (0.005, 0.3, 'float', 'log'),
    'max_depth':        (2, 8, 'int', 'linear'),
    'subsample':        (0.5, 1.0, 'float', 'linear'),
    'colsample_bytree': (0.5, 1.0, 'float', 'linear'),
}
FUSION_WEIGHT_GRID = np.arange(0.0, 1.0001, 0.02)  # weight on the spatial model

# Bookkeeping/metadata columns produced by the feature extractors that are
# not predictive spatial statistics (success flags, voxel counts, p-values,
# resample spacing echoed back per case).
NON_FEATURE_SUFFIXES = ('_success', '_n_voxels', '_moran_p', '_geary_p', '_error_message')
NON_FEATURE_COLS = {
    'case_id', 'CaseID', 'response', 'success', 'error_message', 'n_voxels',
    'moran_p', 'geary_p',
    'original_spacing_z', 'original_spacing_y', 'original_spacing_x',
    'resampled_spacing_z', 'resampled_spacing_y', 'resampled_spacing_x',
    'total_textures_processed', 'successful_textures',
}
RADIOMICS_PREFIXES = ('original_', 'wavelet_', 'log-sigma')


# =============================================================================
# DATA LOADING
# =============================================================================
def _strip_suffix(case_id):
    """'1234567890.nii' -> '1234567890' so spatial3d/texture case_id lines
    up with radiomics CaseID and the label table's patient_id."""
    s = str(case_id).strip()
    return s.split('.nii')[0] if '.nii' in s else s


def load_cohort(feature_dir: Path, labels_excel: Path) -> pd.DataFrame:
    """
    Merge spatial3d_features.csv + spatial_texture_features.csv +
    radiomics_features.csv + the Response label table into one wide table,
    keyed by case_id, inner-joined so only cases with ALL four sources
    survive (a case missing radiomics, for example, can't get a fused
    score and would otherwise misalign the spatial/radiomics probability
    arrays downstream).
    """
    df_spatial = pd.read_csv(feature_dir / 'spatial3d_features.csv')
    df_spatial['case_id'] = df_spatial['case_id'].apply(_strip_suffix)

    df_texture = pd.read_csv(feature_dir / 'spatial_texture_features.csv')
    df_texture['case_id'] = df_texture['case_id'].apply(_strip_suffix)

    df_rad = pd.read_csv(feature_dir / 'radiomics_features.csv')
    df_rad['CaseID'] = df_rad['CaseID'].apply(_strip_suffix)
    diagnostics_cols = [c for c in df_rad.columns if 'diagnostics' in c.lower()]
    df_rad = df_rad.drop(columns=diagnostics_cols)

    df_labels = pd.read_excel(labels_excel)
    df_labels = df_labels.rename(columns={'patient_id': 'case_id', 'Response': 'response'})
    df_labels['case_id'] = df_labels['case_id'].apply(_strip_suffix)
    df_labels['response'] = df_labels['response'].astype(str).str.strip()

    merged = pd.merge(df_spatial, df_texture, on='case_id', how='inner')
    merged = pd.merge(merged, df_rad, left_on='case_id', right_on='CaseID', how='inner')
    merged = pd.merge(merged, df_labels[['case_id', 'response']], on='case_id', how='inner')
    merged = merged.drop_duplicates(subset='case_id').reset_index(drop=True)
    return merged


def response_to_binary(responses) -> np.ndarray:
    """pCR -> 0, non-pCR -> 1 (case-insensitive)."""
    return np.array([0 if str(r).strip().lower() == 'pcr' else 1 for r in responses])


# =============================================================================
# FEATURE COLUMN SELECTION
# =============================================================================
def get_spatial_feature_cols(df: pd.DataFrame) -> list:
    """Spatial3d + texture predictive columns (excludes the radiomics block
    and non-predictive bookkeeping/metadata)."""
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if c.endswith(NON_FEATURE_SUFFIXES):
            continue
        if c.startswith(RADIOMICS_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def get_radiomics_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c.startswith(RADIOMICS_PREFIXES) and pd.api.types.is_numeric_dtype(df[c])]


def clean_feature_columns(X_train: pd.DataFrame, label: str) -> list:
    """
    Fit column-level cleaning on the training cohort only: drop columns
    that are >50% missing, zero-variance, or redundant (|r|>0.95 with an
    already-kept column). Returns the surviving column names; callers
    apply this same set to the test/external cohorts.
    """
    missing_frac = X_train.isnull().mean()
    cols = missing_frac[missing_frac <= 0.5].index.tolist()
    dropped_missing = len(X_train.columns) - len(cols)

    X = X_train[cols]
    variances = X.var()
    cols = variances[variances > 0].index.tolist()
    dropped_var = len(variances) - len(cols)

    X = X[cols]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    cols = [c for c in cols if c not in to_drop]

    print(f"  [{label}] {len(X_train.columns)} candidates -> "
          f"-{dropped_missing} (>50% missing) -{dropped_var} (zero-variance) "
          f"-{len(to_drop)} (|r|>0.95) = {len(cols)} kept")
    return cols


# =============================================================================
# MODEL TRAINING (RFE + SMOTE + XGBoost)
# =============================================================================
def train_model(X_train_scaled, y_train, n_features, rfe_step,
                 n_estimators, learning_rate, feature_names, label,
                 max_depth=4, subsample=1.0, colsample_bytree=1.0,
                 ridge_alpha=0.1, ridge_max_iter=1000, verbose=True):
    estimator = Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE, max_iter=ridge_max_iter)
    selector = RFE(estimator, n_features_to_select=n_features, step=rfe_step)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    selected_names = [f for f, keep in zip(feature_names, selector.support_) if keep]

    if verbose:
        print(f"\n  [{label}] Selected {len(selected_names)} features:")
        for f in selected_names:
            print(f"    - {f}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_final, y_train_final = smote.fit_resample(X_train_selected, y_train)
    if verbose:
        print(f"  [{label}] Training set after SMOTE: {len(X_train_final)} "
              f"(class balance: {np.bincount(y_train_final)})")

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
    )
    model.fit(X_train_final, y_train_final)
    return model, selector, selected_names


def evaluate(model, selector, X_scaled, y, label, split_name):
    X_selected = selector.transform(X_scaled)
    y_proba = model.predict_proba(X_selected)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = np.nan
    print(f"  [{label}] {split_name}: ACC={acc:.4f}  AUC={auc:.4f}  (n={len(y)})")
    return y_proba


# =============================================================================
# HYPERPARAMETER SEARCH (feature selection + XGBoost, per model)
# =============================================================================
def _sample_param(rng, spec):
    """Draw one value from a (low, high, dtype, scale) range spec.
    'log' scale samples uniformly in log-space (appropriate for params
    that sensibly span orders of magnitude, like alpha/learning_rate);
    'linear' samples uniformly in the raw range."""
    low, high, dtype, scale = spec
    if scale == 'log':
        value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    else:
        value = float(rng.uniform(low, high))
    if dtype == 'int':
        value = int(round(value))
        value = max(int(low), min(int(high), value))
    return value


def random_search(X_train_scaled, y_train, eval_sets, param_grid,
                   feature_names, label, n_trials, min_auc_floor, rng,
                   selection_splits=('SPCH', 'FUSCC')):
    """
    Randomized search over `param_grid` for one model (spatial or
    radiomics) — covers both feature selection (RFE `n_features`, `step`,
    and the `alpha`/`max_iter` of the Ridge estimator RFE ranks features
    with) and the XGBoost classifier. `eval_sets` is a dict of
    split_name -> (X_scaled, y); every split's AUC is computed and
    reported, but only `selection_splits` feed the selection score. SHPH's
    Test split is deliberately NOT in the default `selection_splits` —
    it's the held-out slice of the training cohort, so picking
    hyperparameters to maximize its AUC would turn it into a de facto
    validation set and bias the "unbiased holdout" number we report for
    it. SPCH/FUSCC are separate external cohorts (never part of the SHPH
    train/test split), so using them for selection doesn't have that
    problem.

    Selection: mean AUC over `selection_splits` among trials where every
    one of those splits individually clears `min_auc_floor`; falls back
    to the best mean overall (with a warning) if nothing clears the floor.

    Returns (best_result_dict, trials_df).
    """
    trials = []

    def _sample_train_kwargs(params):
        return dict(
            n_features=int(params['n_features']), rfe_step=int(params['rfe_step']),
            n_estimators=int(params['n_estimators']),
            learning_rate=float(params['learning_rate']),
            max_depth=int(params['max_depth']),
            subsample=float(params['subsample']),
            colsample_bytree=float(params['colsample_bytree']),
            ridge_alpha=float(params['ridge_alpha']),
            ridge_max_iter=int(params['ridge_max_iter']),
        )

    for trial_id in range(n_trials):
        params = {k: _sample_param(rng, spec) for k, spec in param_grid.items()}
        model, selector, selected_names = train_model(
            X_train_scaled, y_train, **_sample_train_kwargs(params),
            feature_names=feature_names, label=label, verbose=False,
        )

        aucs = {}
        for split_name, (X_scaled, y) in eval_sets.items():
            X_sel = selector.transform(X_scaled)
            proba = model.predict_proba(X_sel)[:, 1]
            try:
                aucs[split_name] = roc_auc_score(y, proba)
            except ValueError:
                aucs[split_name] = np.nan

        mean_auc = float(np.mean([aucs[s] for s in selection_splits]))
        meets_floor = all(aucs[s] >= min_auc_floor for s in selection_splits)

        trials.append({
            'trial': trial_id,
            **_sample_train_kwargs(params),
            **{f'auc_{s.lower()}': aucs.get(s, np.nan) for s in eval_sets},
            'mean_auc': mean_auc,
            'meets_floor': meets_floor,
        })

    trials_df = pd.DataFrame(trials)
    passing = trials_df[trials_df['meets_floor']]
    if len(passing) > 0:
        best_row = passing.loc[passing['mean_auc'].idxmax()]
        floor_met = True
    else:
        print(f"  ⚠ [{label}] No trial met the AUC floor ({min_auc_floor:.2f}) on all of "
              f"{selection_splits} — falling back to the best mean AUC overall.")
        best_row = trials_df.loc[trials_df['mean_auc'].idxmax()]
        floor_met = False

    # Refit the winning config (verbose, so the chosen features/SMOTE info
    # get printed once for the record instead of on every trial).
    best_model, best_selector, best_selected_names = train_model(
        X_train_scaled, y_train, **_sample_train_kwargs(best_row),
        feature_names=feature_names, label=label, verbose=True,
    )

    print(f"\n  [{label}] BEST config out of {n_trials} trials (floor met: {floor_met}):")
    print(f"    n_features={int(best_row['n_features'])}, rfe_step={int(best_row['rfe_step'])}, "
          f"ridge_alpha={best_row['ridge_alpha']:.4g}, ridge_max_iter={int(best_row['ridge_max_iter'])}")
    print(f"    n_estimators={int(best_row['n_estimators'])}, "
          f"learning_rate={best_row['learning_rate']:.3f}, "
          f"max_depth={int(best_row['max_depth'])}, "
          f"subsample={best_row['subsample']:.2f}, "
          f"colsample_bytree={best_row['colsample_bytree']:.2f}")
    print(f"    AUC: Test={best_row['auc_test']:.4f} (reported only, not used for selection)  "
          f"SPCH={best_row['auc_spch']:.4f}  FUSCC={best_row['auc_fuscc']:.4f}  "
          f"Mean({'+'.join(selection_splits)})={best_row['mean_auc']:.4f}")

    return {
        'model': best_model,
        'selector': best_selector,
        'selected_names': best_selected_names,
        'params': best_row.to_dict(),
        'floor_met': floor_met,
    }, trials_df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SHPH train/test + SPCH/FUSCC external validation")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("STEP 1: LOAD & MERGE COHORTS")
    print("=" * 70)

    cohort_data = {}
    for name, cfg in COHORTS.items():
        df = load_cohort(cfg['feature_dir'], cfg['labels_excel'])
        cohort_data[name] = df
        print(f"  {name}: {len(df)} cases with complete spatial+texture+radiomics+label data")
        print(f"    Response: {df['response'].value_counts().to_dict()}")

    df_shph, df_spch, df_fuscc = cohort_data['SHPH'], cohort_data['SPCH'], cohort_data['FUSCC']

    print("\n" + "=" * 70)
    print("STEP 2: FEATURE CLEANING (fit on SHPH only)")
    print("=" * 70)

    spatial_candidate_cols = get_spatial_feature_cols(df_shph)
    radiomics_candidate_cols = get_radiomics_feature_cols(df_shph)

    spatial_cols = clean_feature_columns(df_shph[spatial_candidate_cols], 'Spatial')
    radiomics_cols = clean_feature_columns(df_shph[radiomics_candidate_cols], 'Radiomics')

    def prepare_cohort(df, cohort_name):
        """
        Build aligned (spatial, radiomics, label, case_id) arrays for one
        cohort, keeping only rows complete in BOTH feature blocks so the
        eventual spatial/radiomics probability arrays fuse case-for-case.
        """
        valid = df[spatial_cols].notna().all(axis=1) & df[radiomics_cols].notna().all(axis=1)
        n_dropped = (~valid).sum()
        if n_dropped:
            print(f"  [{cohort_name}] Dropping {n_dropped} case(s) with missing "
                  f"values in the selected feature set")
        df = df.loc[valid].reset_index(drop=True)
        return (df[spatial_cols].copy(), df[radiomics_cols].copy(),
                response_to_binary(df['response']), df['case_id'].values)

    X_spatial_shph, X_rad_shph, y_shph, case_ids_shph = prepare_cohort(df_shph, 'SHPH')
    X_spatial_spch, X_rad_spch, y_spch, case_ids_spch = prepare_cohort(df_spch, 'SPCH')
    X_spatial_fuscc, X_rad_fuscc, y_fuscc, case_ids_fuscc = prepare_cohort(df_fuscc, 'FUSCC')

    print(f"\n  Final SHPH cohort for modeling: {len(y_shph)} cases "
          f"(pCR={np.sum(y_shph == 0)}, non-pCR={np.sum(y_shph == 1)})")

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN/TEST SPLIT (SHPH only)")
    print("=" * 70)

    (X_spatial_train, X_spatial_test,
     X_rad_train, X_rad_test,
     y_train, y_test,
     case_ids_train, case_ids_test) = train_test_split(
        X_spatial_shph, X_rad_shph, y_shph, case_ids_shph,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_shph
    )
    print(f"  Train: {len(y_train)}  Test: {len(y_test)}")

    print("\n" + "=" * 70)
    print("STEP 4: SCALING (fit on SHPH train only)")
    print("=" * 70)

    spatial_scaler = StandardScaler().fit(X_spatial_train)
    X_spatial_train_scaled = spatial_scaler.transform(X_spatial_train)
    X_spatial_test_scaled = spatial_scaler.transform(X_spatial_test)
    X_spatial_spch_scaled = spatial_scaler.transform(X_spatial_spch)
    X_spatial_fuscc_scaled = spatial_scaler.transform(X_spatial_fuscc)

    rad_scaler = StandardScaler().fit(X_rad_train)
    X_rad_train_scaled = rad_scaler.transform(X_rad_train)
    X_rad_test_scaled = rad_scaler.transform(X_rad_test)
    X_rad_spch_scaled = rad_scaler.transform(X_rad_spch)
    X_rad_fuscc_scaled = rad_scaler.transform(X_rad_fuscc)

    print("\n" + "=" * 70)
    print(f"STEP 5: SPATIAL MODEL HYPERPARAMETER SEARCH ({N_SEARCH_TRIALS} trials)")
    print("=" * 70)

    search_rng = np.random.RandomState(RANDOM_STATE)

    spatial_eval_sets = {
        'Train': (X_spatial_train_scaled, y_train),
        'Test':  (X_spatial_test_scaled, y_test),
        'SPCH':  (X_spatial_spch_scaled, y_spch),
        'FUSCC': (X_spatial_fuscc_scaled, y_fuscc),
    }
    spatial_best, spatial_trials = random_search(
        X_spatial_train_scaled, y_train, spatial_eval_sets, SPATIAL_PARAM_GRID,
        feature_names=spatial_cols, label='Spatial',
        n_trials=N_SEARCH_TRIALS, min_auc_floor=MIN_AUC_FLOOR, rng=search_rng,
    )
    spatial_model = spatial_best['model']
    spatial_selector = spatial_best['selector']
    selected_spatial_names = spatial_best['selected_names']

    (DATA_ROOT / 'model_results').mkdir(parents=True, exist_ok=True)
    spatial_trials.to_csv(DATA_ROOT / 'model_results' / 'spatial_search_trials.csv', index=False)
    print(f"  ✓ Saved all {N_SEARCH_TRIALS} spatial search trials to "
          f"{DATA_ROOT / 'model_results' / 'spatial_search_trials.csv'}")

    print()
    proba_sp_train = evaluate(spatial_model, spatial_selector, X_spatial_train_scaled, y_train, 'Spatial', 'Train')
    proba_sp_test = evaluate(spatial_model, spatial_selector, X_spatial_test_scaled, y_test, 'Spatial', 'Test')
    proba_sp_spch = evaluate(spatial_model, spatial_selector, X_spatial_spch_scaled, y_spch, 'Spatial', 'SPCH (external)')
    proba_sp_fuscc = evaluate(spatial_model, spatial_selector, X_spatial_fuscc_scaled, y_fuscc, 'Spatial', 'FUSCC (external)')

    print("\n" + "=" * 70)
    print(f"STEP 6: RADIOMICS MODEL HYPERPARAMETER SEARCH ({N_SEARCH_TRIALS} trials)")
    print("=" * 70)

    radiomics_eval_sets = {
        'Train': (X_rad_train_scaled, y_train),
        'Test':  (X_rad_test_scaled, y_test),
        'SPCH':  (X_rad_spch_scaled, y_spch),
        'FUSCC': (X_rad_fuscc_scaled, y_fuscc),
    }
    radiomics_best, radiomics_trials = random_search(
        X_rad_train_scaled, y_train, radiomics_eval_sets, RADIOMICS_PARAM_GRID,
        feature_names=radiomics_cols, label='Radiomics',
        n_trials=N_SEARCH_TRIALS, min_auc_floor=MIN_AUC_FLOOR, rng=search_rng,
    )
    rad_model = radiomics_best['model']
    rad_selector = radiomics_best['selector']
    selected_radiomics_names = radiomics_best['selected_names']

    radiomics_trials.to_csv(DATA_ROOT / 'model_results' / 'radiomics_search_trials.csv', index=False)
    print(f"  ✓ Saved all {N_SEARCH_TRIALS} radiomics search trials to "
          f"{DATA_ROOT / 'model_results' / 'radiomics_search_trials.csv'}")

    print()
    proba_rad_train = evaluate(rad_model, rad_selector, X_rad_train_scaled, y_train, 'Radiomics', 'Train')
    proba_rad_test = evaluate(rad_model, rad_selector, X_rad_test_scaled, y_test, 'Radiomics', 'Test')
    proba_rad_spch = evaluate(rad_model, rad_selector, X_rad_spch_scaled, y_spch, 'Radiomics', 'SPCH (external)')
    proba_rad_fuscc = evaluate(rad_model, rad_selector, X_rad_fuscc_scaled, y_fuscc, 'Radiomics', 'FUSCC (external)')

    print("\n" + "=" * 70)
    print(f"STEP 7: SCORE FUSION WEIGHT SEARCH (spatial+radiomics fixed from Steps 5-6)")
    print("=" * 70)

    def fuse(sp, rad, w):
        return w * sp + (1 - w) * rad

    fusion_trials = []
    for w in FUSION_WEIGHT_GRID:
        w = float(w)
        aucs = {}
        for split_name, y_split, sp, rad in [
            ('test',  y_test,  proba_sp_test,  proba_rad_test),
            ('spch',  y_spch,  proba_sp_spch,  proba_rad_spch),
            ('fuscc', y_fuscc, proba_sp_fuscc, proba_rad_fuscc),
        ]:
            try:
                aucs[split_name] = roc_auc_score(y_split, fuse(sp, rad, w))
            except ValueError:
                aucs[split_name] = np.nan
        # Test is computed/reported but excluded from selection, same
        # reasoning as random_search(): it's SHPH's held-out split, not an
        # independent cohort, so it must stay an unbiased holdout.
        mean_auc = float(np.mean([aucs['spch'], aucs['fuscc']]))
        meets_floor = aucs['spch'] >= MIN_AUC_FLOOR and aucs['fuscc'] >= MIN_AUC_FLOOR
        fusion_trials.append({
            'weight_spatial': w, 'weight_radiomics': 1 - w,
            'auc_test': aucs['test'], 'auc_spch': aucs['spch'], 'auc_fuscc': aucs['fuscc'],
            'mean_auc': mean_auc, 'meets_floor': meets_floor,
        })

    fusion_trials_df = pd.DataFrame(fusion_trials)
    fusion_trials_df.to_csv(DATA_ROOT / 'model_results' / 'fusion_search_trials.csv', index=False)
    print(f"  ✓ Saved all {len(fusion_trials_df)} fusion-weight trials to "
          f"{DATA_ROOT / 'model_results' / 'fusion_search_trials.csv'}")

    passing = fusion_trials_df[fusion_trials_df['meets_floor']]
    if len(passing) > 0:
        best_fusion = passing.loc[passing['mean_auc'].idxmax()]
        fusion_floor_met = True
    else:
        print(f"  ⚠ No fusion weight met the AUC floor ({MIN_AUC_FLOOR:.2f}) on both SPCH "
              f"and FUSCC — falling back to the best mean AUC overall.")
        best_fusion = fusion_trials_df.loc[fusion_trials_df['mean_auc'].idxmax()]
        fusion_floor_met = False

    FUSION_WEIGHT_SPATIAL = float(best_fusion['weight_spatial'])
    print(f"\n  BEST fusion weight (floor met: {fusion_floor_met}): "
          f"spatial={FUSION_WEIGHT_SPATIAL:.2f}  radiomics={1 - FUSION_WEIGHT_SPATIAL:.2f}")
    print(f"  AUC: Test={best_fusion['auc_test']:.4f} (reported only, not used for selection)  "
          f"SPCH={best_fusion['auc_spch']:.4f}  "
          f"FUSCC={best_fusion['auc_fuscc']:.4f}  Mean={best_fusion['mean_auc']:.4f}")

    proba_fused_train = fuse(proba_sp_train, proba_rad_train, FUSION_WEIGHT_SPATIAL)
    proba_fused_test = fuse(proba_sp_test, proba_rad_test, FUSION_WEIGHT_SPATIAL)
    proba_fused_spch = fuse(proba_sp_spch, proba_rad_spch, FUSION_WEIGHT_SPATIAL)
    proba_fused_fuscc = fuse(proba_sp_fuscc, proba_rad_fuscc, FUSION_WEIGHT_SPATIAL)

    for split_name, y, proba in [
        ('Train', y_train, proba_fused_train),
        ('Test', y_test, proba_fused_test),
        ('SPCH (external)', y_spch, proba_fused_spch),
        ('FUSCC (external)', y_fuscc, proba_fused_fuscc),
    ]:
        acc = accuracy_score(y, proba > 0.5)
        auc = roc_auc_score(y, proba)
        print(f"  [Fused] {split_name}: ACC={acc:.4f}  AUC={auc:.4f}")

    print("\n" + "=" * 70)
    print("STEP 8: SAVE PREDICTIONS")
    print("=" * 70)

    def save_predictions(case_ids, y, proba_sp, proba_rad, proba_fused, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame({
            'case_id': case_ids,
            'true_label': y,
            'true_label_name': ['pCR' if v == 0 else 'Non-pCR' for v in y],
            'spatial_probability': proba_sp,
            'radiomics_probability': proba_rad,
            'fused_probability': proba_fused,
        }).sort_values('fused_probability', ascending=False)
        df_out.to_csv(out_path, index=False)
        print(f"  ✓ {out_path}")

    save_predictions(case_ids_train, y_train, proba_sp_train, proba_rad_train, proba_fused_train,
                      DATA_ROOT / 'model_results' / 'train_predictions_results.csv')
    save_predictions(case_ids_test, y_test, proba_sp_test, proba_rad_test, proba_fused_test,
                      DATA_ROOT / 'model_results' / 'test_predictions_results.csv')
    save_predictions(case_ids_spch, y_spch, proba_sp_spch, proba_rad_spch, proba_fused_spch,
                      DATA_ROOT / 'SPCH' / 'model_results' / 'SPCH_predictions_results.csv')
    save_predictions(case_ids_fuscc, y_fuscc, proba_sp_fuscc, proba_rad_fuscc, proba_fused_fuscc,
                      DATA_ROOT / 'FUSCC' / 'model_results' / 'FUSCC_predictions_results.csv')

    print("\n" + "=" * 70)
    print("STEP 9: SAVE BEST HYPERPARAMETERS")
    print("=" * 70)

    best_hyperparams = {
        'spatial': {k: v for k, v in spatial_best['params'].items()
                    if k not in ('trial',)},
        'radiomics': {k: v for k, v in radiomics_best['params'].items()
                      if k not in ('trial',)},
        'fusion': best_fusion.to_dict(),
        'selected_spatial_features': selected_spatial_names,
        'selected_radiomics_features': selected_radiomics_names,
        'selection_criterion': f'mean(AUC_SPCH, AUC_FUSCC), each required to clear '
                                f'{MIN_AUC_FLOOR:.2f} (fallback to best mean if none clear '
                                f'it). SHPH Test AUC is reported but excluded from '
                                f'selection to keep it an unbiased holdout.',
    }
    import json
    best_hyperparams_path = DATA_ROOT / 'model_results' / 'best_hyperparameters.json'
    with open(best_hyperparams_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=2, default=str)
    print(f"  ✓ {best_hyperparams_path}")

    print("\n" + "=" * 70)
    print("STEP 10: SAVE FITTED MODELS (for reuse without rerunning the search)")
    print("=" * 70)

    artifacts = {
        'spatial_scaler': spatial_scaler,
        'spatial_selector': spatial_selector,
        'spatial_model': spatial_model,
        'spatial_feature_cols': spatial_cols,
        'radiomics_scaler': rad_scaler,
        'radiomics_selector': rad_selector,
        'radiomics_model': rad_model,
        'radiomics_feature_cols': radiomics_cols,
        'fusion_weight_spatial': FUSION_WEIGHT_SPATIAL,
    }
    models_path = DATA_ROOT / 'model_results' / 'best_models.joblib'
    joblib.dump(artifacts, models_path)
    print(f"  ✓ {models_path}")
    print(f"    (load with: artifacts = joblib.load('{models_path}'); "
          f"artifacts['spatial_model'].predict_proba(...), etc. — scalers/selectors/")
    print(f"    feature-column-lists are bundled too so new data can be scored end-to-end.)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"""
  Spatial model   ({len(selected_spatial_names)} features, floor met: {spatial_best['floor_met']}):
    {selected_spatial_names}
    rfe_step={int(spatial_best['params']['rfe_step'])}, ridge_alpha={spatial_best['params']['ridge_alpha']:.4g}, ridge_max_iter={int(spatial_best['params']['ridge_max_iter'])}
    n_estimators={int(spatial_best['params']['n_estimators'])}, learning_rate={spatial_best['params']['learning_rate']:.3f}, max_depth={int(spatial_best['params']['max_depth'])}, subsample={spatial_best['params']['subsample']:.2f}, colsample_bytree={spatial_best['params']['colsample_bytree']:.2f}

  Radiomics model ({len(selected_radiomics_names)} features, floor met: {radiomics_best['floor_met']}):
    {selected_radiomics_names}
    rfe_step={int(radiomics_best['params']['rfe_step'])}, ridge_alpha={radiomics_best['params']['ridge_alpha']:.4g}, ridge_max_iter={int(radiomics_best['params']['ridge_max_iter'])}
    n_estimators={int(radiomics_best['params']['n_estimators'])}, learning_rate={radiomics_best['params']['learning_rate']:.3f}, max_depth={int(radiomics_best['params']['max_depth'])}, subsample={radiomics_best['params']['subsample']:.2f}, colsample_bytree={radiomics_best['params']['colsample_bytree']:.2f}

  Fusion weight (floor met: {fusion_floor_met}):
    spatial={FUSION_WEIGHT_SPATIAL:.2f}  radiomics={1 - FUSION_WEIGHT_SPATIAL:.2f}

  Full search trial histories saved to:
    {DATA_ROOT / 'model_results' / 'spatial_search_trials.csv'}
    {DATA_ROOT / 'model_results' / 'radiomics_search_trials.csv'}
    {DATA_ROOT / 'model_results' / 'fusion_search_trials.csv'}
""")
