"""
Provenance script (requires the private Dataset/ directory -- NOT
runnable by reviewers) that exports de-identified per-patient assets
for the two example patients used by NatureSubmission/2_Demo/.

For each case, exports:
  - slices.npz containing:
      - ct          (n_slices, H, W)    uint8 grayscale, soft-tissue window
      - seg_overlay (n_slices, H, W, 3) uint8 RGB, tumor contour baked in
      - feat_values (n_slices, H, W)    float32, raw GLCM-Contrast value
                                          per voxel (NaN outside the tumor)
                                          -- rendered live in the GUI with
                                          matplotlib (Nature-style 'viridis'
                                          colormap + a colorbar), not baked
                                          into RGB, so the colorbar's scale
                                          is visible
      - feat_vmin, feat_vmax            float scalars, the fixed color
                                          scale (2nd/98th percentile of
                                          tumor voxel values) so every
                                          slice of this case shares one
                                          colorbar
      - tumor_area  (n_slices,)          int, tumor pixel count per
                                          slice (lets the GUI default the
                                          slider to the largest
                                          cross-section)
  - mesh.npz containing:
      - verts (V, 3), faces (F, 3)       a 3D surface mesh of the tumor
                                          ROI (marching cubes on the
                                          segmentation mask, scaled by
                                          the real voxel spacing), shown
                                          in the GUI with Plotly (an
                                          interactive, rotatable WebGL
                                          mesh -- no 3D graphics library
                                          needed at GUI runtime)
  - features.csv  -- real extracted feature values for this patient
                      (same 116 radiomics + 3D-spatial features as the
                      cohort table)
  - prediction.json -- the real locked model's actual prediction for
                      this patient

Example-patient selection:

- **Non-pCR example:** among Test-cohort patients with a tumor large
  enough for a clear slice view (n_voxels >= MIN_VOXELS), restricted to
  patients where the real locked SRP (fused) model classifies correctly
  at the standard 0.5 probability threshold *and* the Radiomics-only
  model classifies incorrectly at that same threshold -- i.e., a case
  that concretely demonstrates the Fusion model succeeding where
  Conventional Radiomics alone fails -- the patient the Fusion model
  classifies most confidently (highest fused probability) is picked.
- **pCR example:** a *representative* true-pCR patient rather than an
  edge case -- searched across both the Train and Test splits (this is
  a visual walkthrough example, not a held-out performance claim, so
  Train/Test overfitting concerns don't apply here the way they do for
  the cohort-level statistics in `ecresponse_cohort_deidentified.csv`),
  restricted to patients with a tumor size inside the interquartile
  range of all eligible pCR tumor sizes (i.e. a typical size, not a
  size outlier) and correctly classified by the real locked Fusion
  model; among those, the one the Fusion model classifies most
  confidently (lowest fused probability) is picked.

This is fully determined by the above criteria; no literal case ID is
hardcoded anywhere in this script.

De-identification: as before -- which two patients are used is not
disclosed; real case IDs are used only internally to locate source
files and never written to any output; only the tumor's cropped
bounding-box region is exported (not the full image), and only as
rasterized PNG-equivalent arrays (no header metadata of any kind
survives).
"""
import gzip
import json
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from skimage.measure import marching_cubes

ROOT = Path('XXXyour path/Dataset/Dataset002_ECResponse')  # private, restricted-access cohort data
OUT_ROOT = Path('XXXyour path/NatureSubmission/2_Demo/data')
MIN_VOXELS = 20000
CLASSIFICATION_THRESHOLD = 0.5
WINDOW_LEVEL, WINDOW_WIDTH = 50, 350
MARGIN_FRAC = 0.6
MIN_SIZE = 140


def select_example_cases():
    sp3 = pd.read_csv(ROOT / 'spatial3d_features.csv')
    sp3['case_id'] = sp3['case_id'].str.replace('.nii', '', regex=False).astype(int)

    # Non-pCR example: Test cohort only (this is the split whose
    # predictions are reported as this demo's headline statistics).
    test_preds = pd.read_csv(ROOT / 'model_results' / 'test_predictions_results.csv')
    test_preds['case_id'] = test_preds['case_id'].astype(int)
    test_preds = test_preds.merge(sp3[['case_id', 'n_voxels']], on='case_id', how='left')
    t = CLASSIFICATION_THRESHOLD
    test_eligible = test_preds[test_preds['n_voxels'] >= MIN_VOXELS].copy()
    fused_correct = (test_eligible['fused_probability'] > t).astype(int) == test_eligible['true_label']
    radiomics_correct = (test_eligible['radiomics_probability'] > t).astype(int) == test_eligible['true_label']
    srp_wins = test_eligible[fused_correct & ~radiomics_correct]
    nonpcr_row = srp_wins[srp_wins['true_label'] == 1].sort_values('fused_probability', ascending=False).iloc[0]

    # pCR example: searched across Train + Test for a representative
    # (typical tumor size), confidently-and-correctly-classified case.
    train_preds = pd.read_csv(ROOT / 'model_results' / 'train_predictions_results.csv')
    all_preds = pd.concat([train_preds, test_preds[train_preds.columns]], ignore_index=True)
    all_preds['case_id'] = all_preds['case_id'].astype(int)
    all_preds = all_preds.merge(sp3[['case_id', 'n_voxels']], on='case_id', how='left')
    pcr_eligible = all_preds[(all_preds['true_label'] == 0) & (all_preds['n_voxels'] >= MIN_VOXELS)]
    size_q1, size_q3 = pcr_eligible['n_voxels'].quantile([0.25, 0.75])
    typical_size = pcr_eligible['n_voxels'].between(size_q1, size_q3)
    fused_correct_pcr = pcr_eligible['fused_probability'] <= t
    pcr_row = pcr_eligible[typical_size & fused_correct_pcr].sort_values('fused_probability').iloc[0]

    return {
        'case_pCR': {'case_id': int(pcr_row['case_id']), 'true_label_name': 'pCR'},
        'case_nonpCR': {'case_id': int(nonpcr_row['case_id']), 'true_label_name': 'Non-pCR'},
    }


CASES = select_example_cases()


def window_to_uint8(arr, level=WINDOW_LEVEL, width=WINDOW_WIDTH):
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def crop_box_3d(mask):
    zs, ys, xs = np.where(mask)
    pad_z = 2
    z0, z1 = max(0, zs.min() - pad_z), zs.max() + pad_z + 1
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    size = int(max(y1 - y0, x1 - x0, MIN_SIZE) * (1 + MARGIN_FRAC))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = size // 2
    H, W = mask.shape[1], mask.shape[2]
    return z0, z1, max(0, cy - half), min(H, cy + half), max(0, cx - half), min(W, cx + half)


def overlay_contour(gray_u8, mask2d, color=(255, 60, 60)):
    from scipy import ndimage
    rgb = np.stack([gray_u8] * 3, axis=-1)
    if not mask2d.any():
        return rgb
    eroded = ndimage.binary_erosion(mask2d)
    boundary = mask2d & ~eroded
    rgb[boundary] = color
    return rgb


def build_roi_mesh(mask_c, spacing_xyz):
    """Marching-cubes surface mesh of the cropped tumor mask, scaled by
    the real voxel spacing so the 3D shape is not distorted."""
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    verts, faces, _, _ = marching_cubes(mask_c.astype(np.uint8), level=0.5, spacing=spacing_zyx)
    return verts.astype(np.float32), faces.astype(np.int32)


def export_prediction_and_features(cid, true_label_name, out_dir, preds, rad, sp3):
    row = preds[preds['case_id'] == cid].iloc[0]
    fused = float(row['fused_probability'])
    risk_group = ('High-risk (predicted Non-pCR)' if fused > CLASSIFICATION_THRESHOLD
                  else 'Low-risk (predicted pCR)')
    with open(out_dir / 'prediction.json', 'w') as f:
        json.dump({
            'true_label': true_label_name,
            'spatial_probability': round(float(row['spatial_probability']), 4),
            'radiomics_probability': round(float(row['radiomics_probability']), 4),
            'fused_probability': round(fused, 4),
            'predicted_risk_group': risk_group,
        }, f, indent=2)

    rad_row = rad[rad['case_id'] == cid].iloc[0]
    sp3_row = sp3[sp3['case_id'] == cid].iloc[0]
    feat = pd.concat([
        rad_row.drop('case_id').add_prefix('radiomics__'),
        sp3_row.drop('case_id').add_prefix('spatial3d__'),
    ])
    feat.rename('value').rename_axis('feature').reset_index().to_csv(out_dir / 'features.csv', index=False)


preds_full = pd.concat([
    pd.read_csv(ROOT / 'model_results' / 'train_predictions_results.csv'),
    pd.read_csv(ROOT / 'model_results' / 'test_predictions_results.csv'),
], ignore_index=True)
preds_full['case_id'] = preds_full['case_id'].astype(int)

rad_full = pd.read_csv(ROOT / 'radiomics_features.csv').rename(columns={'CaseID': 'case_id'})
rad_full['case_id'] = rad_full['case_id'].astype(int)
rad_full = rad_full.drop(columns=[c for c in rad_full.columns if 'diagnostics' in c.lower()])

sp3_full = pd.read_csv(ROOT / 'spatial3d_features.csv')
sp3_full['case_id'] = sp3_full['case_id'].str.replace('.nii', '', regex=False).astype(int)

for role, info in CASES.items():
    cid = info['case_id']
    out_dir = OUT_ROOT / role
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- {role} (source case de-identified on export) ---")

    image_sitk = sitk.ReadImage(str(ROOT / 'Image' / f'{cid}.nii.gz'))
    spacing_xyz = image_sitk.GetSpacing()
    img = sitk.GetArrayFromImage(image_sitk)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / 'Mask' / f'{cid}.nii.gz'))) > 0

    with gzip.open(ROOT / 'cache' / 'feature_maps' / f'{cid}_adc655be.pkl.gz', 'rb') as f:
        fmap_data = pickle.load(f)
    feat_vol = fmap_data['glcm']['glcm_Contrast']

    z0, z1, y0, y1, x0, x1 = crop_box_3d(mask)
    img_c = img[z0:z1, y0:y1, x0:x1]
    mask_c = mask[z0:z1, y0:y1, x0:x1]
    feat_c = feat_vol[z0:z1, y0:y1, x0:x1]

    tumor_vals = feat_c[mask_c]
    vmin, vmax = np.percentile(tumor_vals, [2, 98])

    n = img_c.shape[0]
    ct_stack = np.zeros((n,) + img_c.shape[1:], dtype=np.uint8)
    seg_stack = np.zeros((n,) + img_c.shape[1:] + (3,), dtype=np.uint8)
    feat_values = np.full((n,) + img_c.shape[1:], np.nan, dtype=np.float32)
    tumor_area = np.zeros(n, dtype=np.int32)

    for i in range(n):
        gray = window_to_uint8(img_c[i])
        ct_stack[i] = gray
        seg_stack[i] = overlay_contour(gray, mask_c[i])
        feat_values[i][mask_c[i]] = feat_c[i][mask_c[i]].astype(np.float32)
        tumor_area[i] = int(mask_c[i].sum())

    np.savez_compressed(
        out_dir / 'slices.npz',
        ct=ct_stack, seg_overlay=seg_stack, feat_values=feat_values,
        feat_vmin=np.float32(vmin), feat_vmax=np.float32(vmax), tumor_area=tumor_area,
    )
    print(f"  {n} axial slices exported, shape {img_c.shape[1:]}, "
          f"tumor present in {int((tumor_area > 0).sum())} slices")

    verts, faces = build_roi_mesh(mask_c, spacing_xyz)
    np.savez_compressed(out_dir / 'mesh.npz', verts=verts, faces=faces)
    print(f"  3D ROI mesh exported: {len(verts)} vertices, {len(faces)} faces")

    export_prediction_and_features(cid, info['true_label_name'], out_dir, preds_full, rad_full, sp3_full)
    print(f"  prediction.json + features.csv exported to {out_dir}")

print("\nDONE")
