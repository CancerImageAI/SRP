"""
SRP Demo -- a single, consolidated, button-driven walkthrough of the
complete SRP (Spatial + Radiomics -> Fusion) workflow, combining:

  (a) a per-patient visual walkthrough (input image -> segmentation ->
      feature extraction -> visualization -> response prediction) on
      two real, de-identified 3D example patients from the ESCC
      (esophageal squamous cell carcinoma) cohort -- a representative
      pCR patient (typical tumor size, confidently and correctly
      classified by the real locked model) and a Non-pCR patient chosen
      because the real locked SRP (fused) model classifies it correctly
      while the Conventional Radiomics-only model does not, a concrete
      demonstration of the Spatial model's added value -- viewed
      slice-by-slice on the axial plane (a slider above the image steps
      through every slice of the 3D volume, the standard radiology-
      viewer convention) alongside a rotatable 3D mesh of the tumor
      ROI; and
  (b) cohort-level statistical evaluation (ROC/decision-curve/
      calibration, risk stratification, survival analysis) computed
      from the real, already-trained, locked SRP model's actual
      historical predictions for the full 110-patient de-identified
      cohort -- not retrained here (see data/README.md for why: a
      simplified from-scratch retraining on this modest cohort size
      cannot be expected to reproduce the properly-tuned production
      model's real performance, so this demo uses that model's real,
      already-computed output directly, which is both faster and
      actually correct).

Run with:
    streamlit run app.py

Opens in your default web browser (http://localhost:8501) and works
identically on Windows, Mac, and Linux -- Streamlit is pure Python.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

DATA_DIR = Path(__file__).resolve().parent / 'data'

st.set_page_config(page_title='SRP Demo', page_icon='🫁', layout='wide')

CASES = {
    'Case A -- pCR (responder)': 'case_pCR',
    'Case B -- Non-pCR (non-responder)': 'case_nonpCR',
}
CASE_LABELS = list(CASES.keys())

PATIENT_STEPS = [
    '1. Input Image (axial)',
    '2. Segmentation (axial)',
    '3. Feature Extraction',
    '4. Visualization (axial)',
    '5. Response Prediction',
]
COHORT_STEPS = [
    '6. Cohort Evaluation',
    '7. Risk Stratification',
    '8. Survival Analysis',
]
ALL_STEPS = PATIENT_STEPS + COHORT_STEPS

CURATED_FEATURES = [
    'spatial3d__moran_i', 'spatial3d__geary_c', 'spatial3d__hotspot_fraction',
    'radiomics__original_shape_Sphericity', 'radiomics__original_firstorder_Mean',
    'radiomics__original_glcm_Contrast',
]

MODEL_COLORS = {'Spatial': '#E74C3C', 'Radiomics': '#3498DB', 'Fusion (SRP)': '#009E73'}

# The Fusion (SRP) model's precomputed log-rank-optimal risk-stratification
# threshold from the real survival analysis (Dataset002_ECResponse/Test,
# survival_analysis/cox_regression_results.csv and median_survival.csv) --
# not an arbitrary 0.5. Reproduces the real group sizes (81 High-risk / 29
# Low-risk), HR=3.05 [1.07-8.66], C-index=0.596, log-rank P=0.0280.
RISK_THRESHOLD = 0.454

# Standard 0.5 probability threshold used to judge per-model classification
# correctness (distinct from RISK_THRESHOLD, which is specific to the
# Fusion model's survival risk-stratification) -- used to confirm, for the
# two example patients, that the real Fusion model classifies correctly
# where the real Radiomics-only model does not (see data/README.md).
CLASSIFICATION_THRESHOLD = 0.5


@st.cache_data
def load_features(case_dir):
    df = pd.read_csv(DATA_DIR / case_dir / 'features.csv')
    return df.set_index('feature')['value']


@st.cache_data
def load_prediction(case_dir):
    with open(DATA_DIR / case_dir / 'prediction.json') as f:
        return json.load(f)


@st.cache_data
def load_slices(case_dir):
    npz = np.load(DATA_DIR / case_dir / 'slices.npz')
    return npz['ct'], npz['seg_overlay'], npz['tumor_area']


@st.cache_data
def load_feature_slices(case_dir):
    npz = np.load(DATA_DIR / case_dir / 'slices.npz')
    return npz['ct'], npz['feat_values'], float(npz['feat_vmin']), float(npz['feat_vmax']), npz['tumor_area']


@st.cache_data
def load_mesh(case_dir):
    npz = np.load(DATA_DIR / case_dir / 'mesh.npz')
    return npz['verts'], npz['faces']


@st.cache_data
def load_cohort():
    return pd.read_csv(DATA_DIR / 'ecresponse_cohort_deidentified.csv')


def roi_mesh_figure(verts, faces, title):
    """An interactive, rotatable 3D surface mesh of the tumor ROI
    (Plotly WebGL Mesh3d, precomputed via marching cubes) -- no 3D
    graphics library is needed at GUI runtime, only Plotly's
    client-side renderer."""
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#E74C3C', opacity=0.85, flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
    )])
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=30, b=0), height=420,
        scene=dict(xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)', aspectmode='data'),
    )
    return fig


def axial_slice_viewer(stack, tumor_area, case_label, step_key, caption_fn, mesh):
    """A slider (above the image) over every axial slice of the 3D
    volume (stack is either grayscale (n,H,W) or RGB (n,H,W,3)),
    defaulting to the slice with the largest tumor cross-section, with
    a rotatable 3D mesh of the tumor ROI shown to the right."""
    n = stack.shape[0]
    default_idx = int(np.argmax(tumor_area))
    img_col, mesh_col = st.columns([3, 2])
    with img_col:
        idx = st.slider('Slice', 0, n - 1, default_idx, key=f'{step_key}_{case_label}',
                         help='Scroll through every slice of the 3D CT volume.')
        frame = stack[idx]
        mode = 'L' if frame.ndim == 2 else 'RGB'
        img = Image.fromarray(frame, mode=mode).resize((420, 420), Image.LANCZOS)
        st.image(img, caption=caption_fn(idx, n, int(tumor_area[idx])))
    with mesh_col:
        verts, faces = mesh
        st.plotly_chart(roi_mesh_figure(verts, faces, '3D tumor ROI (drag to rotate)'),
                         use_container_width=True)


def axial_feature_viewer(ct, feat_values, vmin, vmax, tumor_area, case_label, step_key, mesh):
    """Like axial_slice_viewer, but renders the spatial-heterogeneity
    feature map live with matplotlib (Nature-style 'viridis' colormap,
    masked to the tumor) so a colorbar can be shown alongside it. The
    slider sits above the image; the 3D ROI mesh sits to the right."""
    n = ct.shape[0]
    default_idx = int(np.argmax(tumor_area))
    img_col, mesh_col = st.columns([3, 2])
    with img_col:
        idx = st.slider('Slice', 0, n - 1, default_idx, key=f'{step_key}_{case_label}',
                         help='Scroll through every slice of the 3D CT volume.')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(ct[idx], cmap='gray')
        masked = np.ma.masked_invalid(feat_values[idx])
        im = ax.imshow(masked, cmap='viridis', vmin=vmin, vmax=vmax, alpha=0.80)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('GLCM Contrast (tumor-masked)')
        ax.set_title(f'{case_label} -- axial slice {idx + 1}/{n}', fontsize=9)
        ax.axis('off')
        fig.tight_layout()
        st.pyplot(fig)
    with mesh_col:
        verts, faces = mesh
        st.plotly_chart(roi_mesh_figure(verts, faces, '3D tumor ROI (drag to rotate)'),
                         use_container_width=True)


def main():
    st.title('🫁 SRP Demo')
    st.caption(
        'The complete Spatial-Radiomics-Fusion (SRP) workflow, from a patient\'s CT scan '
        'to a response prediction and cohort-level validation -- real, de-identified data '
        'throughout (ESCC cohort: esophageal squamous cell carcinoma). No real case identifiers, '
        'names, or dates are included anywhere -- see `data/README.md` for exactly what was '
        'de-identified and how.'
    )

    if 'step' not in st.session_state:
        st.session_state.step = ALL_STEPS[0]
    if 'case_label' not in st.session_state:
        st.session_state.case_label = CASE_LABELS[0]

    st.markdown('**Per-patient walkthrough** (2 real example patients)')
    cols = st.columns(len(PATIENT_STEPS))
    for col, step in zip(cols, PATIENT_STEPS):
        if col.button(step, width='stretch', key=f'btn_{step}',
                      type='primary' if st.session_state.step == step else 'secondary'):
            st.session_state.step = step
            st.rerun()

    st.markdown('**Cohort-level evaluation** (full de-identified cohort, n=110)')
    cols2 = st.columns(len(COHORT_STEPS))
    for col, step in zip(cols2, COHORT_STEPS):
        if col.button(step, width='stretch', key=f'btn_{step}',
                      type='primary' if st.session_state.step == step else 'secondary'):
            st.session_state.step = step
            st.rerun()

    step = st.session_state.step

    if step in PATIENT_STEPS:
        case_label = st.radio('Choose an example patient:', CASE_LABELS, horizontal=True,
                               index=CASE_LABELS.index(st.session_state.case_label),
                               key='case_selector')
        st.session_state.case_label = case_label
        case_dir = CASES[case_label]
        prediction = load_prediction(case_dir)
        mesh = load_mesh(case_dir)

    st.divider()
    st.subheader(step)

    # ---------------- per-patient steps ----------------
    if step == '1. Input Image (axial)':
        st.markdown(
            'The raw 3D CT volume, as the pipeline receives it, viewed '
            '**slice by slice on the axial plane** -- the standard radiology-'
            'viewer convention -- at a standard soft-tissue window '
            '(level 50 / width 350 HU), cropped around the tumor region. '
            'Drag the slider above the image to step through every slice of '
            'the volume; the rotatable 3D mesh on the right shows the same '
            'tumor\'s full 3D shape.'
        )
        ct, _, tumor_area = load_slices(case_dir)
        axial_slice_viewer(
            ct, tumor_area, case_label, 'step1',
            lambda idx, n, area: f'{case_label} -- axial slice {idx + 1}/{n}',
            mesh,
        )

    elif step == '2. Segmentation (axial)':
        st.markdown(
            'The tumor segmentation (red contour) on every axial slice, with '
            'the same segmentation shown as a rotatable 3D surface mesh on '
            'the right (marching cubes on the segmentation mask, scaled by '
            'the real voxel spacing). Every downstream feature in this demo '
            'is computed only from voxels inside this boundary, across all '
            'slices, not just the one shown by default.'
        )
        _, seg_overlay, tumor_area = load_slices(case_dir)
        axial_slice_viewer(
            seg_overlay, tumor_area, case_label, 'step2',
            lambda idx, n, area: f'{case_label} -- axial slice {idx + 1}/{n} ({area} tumor px)',
            mesh,
        )

    elif step == '3. Feature Extraction':
        st.markdown(
            'Real extracted feature values for this patient -- radiomics '
            '(PyRadiomics shape/first-order/texture) and 3D spatial-'
            'heterogeneity features, computed from the full 3D volume shown '
            'in steps 1-2.'
        )
        feats = load_features(case_dir)
        curated = feats.reindex(CURATED_FEATURES).rename(
            index=lambda f: f.split('__', 1)[-1].replace('original_', ''))
        st.markdown('**Representative features**')
        st.dataframe(curated.rename('value'), width='stretch')
        st.markdown('**Full extracted feature table (116 features)**')
        st.dataframe(feats.rename('value'), width='stretch')

    elif step == '4. Visualization (axial)':
        st.markdown(
            'A spatial-heterogeneity feature map (GLCM Contrast, computed '
            'voxel-by-voxel across the tumor) overlaid on every axial slice, '
            'masked to the tumor region, using a perceptually-uniform '
            '"Nature style" colormap (`viridis`) with its colorbar shown '
            'alongside, next to the tumor\'s rotatable 3D ROI mesh. This is '
            'what the Spatial model\'s spatial-texture features actually '
            'summarize: *where* in the tumor heterogeneity is concentrated, '
            'not just a single whole-tumor average (Algorithm 2 in the '
            'Supplementary Note).'
        )
        ct, feat_values, vmin, vmax, tumor_area = load_feature_slices(case_dir)
        axial_feature_viewer(ct, feat_values, vmin, vmax, tumor_area, case_label, 'step4', mesh)

    elif step == '5. Response Prediction':
        st.markdown(
            'The locked SRP model\'s actual prediction for this patient: '
            'Spatial-only and Radiomics-only probabilities, and their '
            'fused (SRP) combination.'
        )
        model_order = ['Spatial', 'Radiomics', 'Fusion (SRP)']
        prob_df = pd.DataFrame({
            'Model': pd.Categorical(model_order, categories=model_order, ordered=True),
            'Probability of Non-pCR': [
                prediction['spatial_probability'],
                prediction['radiomics_probability'],
                prediction['fused_probability'],
            ],
        }).set_index('Model')
        st.bar_chart(prob_df)

        col1, col2, col3 = st.columns(3)
        col1.metric('Spatial', f"{prediction['spatial_probability']:.2f}")
        col2.metric('Radiomics', f"{prediction['radiomics_probability']:.2f}")
        col3.metric('Fusion (SRP)', f"{prediction['fused_probability']:.2f}")

        st.markdown(f"**Predicted risk group:** {prediction['predicted_risk_group']}")
        st.markdown(f"**True outcome:** {prediction['true_label']}")
        true_nonpcr = prediction['true_label'] == 'Non-pCR'
        fused_correct = (prediction['fused_probability'] > CLASSIFICATION_THRESHOLD) == true_nonpcr
        if fused_correct:
            st.success('✓ The fused (SRP) prediction agrees with this patient\'s actual outcome.')
        else:
            st.warning('✗ The fused (SRP) prediction disagrees with this patient\'s actual outcome.')

    # ---------------- cohort-level steps ----------------
    elif step == '6. Cohort Evaluation':
        st.markdown(
            'ROC and a multi-metric radar comparison for all three models, '
            'computed from the real locked SRP model\'s actual historical '
            'predictions across the full de-identified cohort (n=110).'
        )
        df = load_cohort()
        y = df['true_label'].values
        results = {
            'Spatial': df['spatial_probability'].values,
            'Radiomics': df['radiomics_probability'].values,
            'Fusion (SRP)': df['fused_probability'].values,
        }
        aucs = {name: roc_auc_score(y, p) for name, p in results.items()}

        c1, c2, c3 = st.columns(3)
        c1.metric('Spatial AUC', f"{aucs['Spatial']:.3f}")
        c2.metric('Radiomics AUC', f"{aucs['Radiomics']:.3f}")
        c3.metric('Fusion (SRP) AUC', f"{aucs['Fusion (SRP)']:.3f}")

        st.markdown('**ROC curves**')
        fig1, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, label='Chance')
        for name, p in results.items():
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2, label=f'{name} (AUC={aucs[name]:.2f})')
        ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
        ax.legend(fontsize=9, loc='lower right')
        fig1.tight_layout()
        st.pyplot(fig1)

        st.markdown('**Multi-metric radar comparison**')
        radar_metrics = ['SPE', 'SEN', 'ACC', 'F1', 'NPV', 'PPV']
        fig2 = plt.figure(figsize=(6, 6))
        ax = fig2.add_subplot(111, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        for name, p in results.items():
            pred = p > CLASSIFICATION_THRESHOLD
            tp = np.sum(pred & (y == 1)); tn = np.sum(~pred & (y == 0))
            fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
            values = [
                tn / (tn + fp), tp / (tp + fn), (tp + tn) / len(y),
                2 * tp / (2 * tp + fp + fn), tn / (tn + fn), tp / (tp + fp),
            ]
            values += values[:1]
            ax.plot(angles, values, color=MODEL_COLORS[name], lw=2, marker='o', label=name)
            ax.fill(angles, values, color=MODEL_COLORS[name], alpha=0.1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)

    elif step == '7. Risk Stratification':
        st.markdown(
            f'Patients are assigned a risk group by thresholding the fused '
            f'(SRP) probability at **{RISK_THRESHOLD}** -- the model\'s precomputed '
            f'optimal risk-stratification cutoff (the log-rank-minimizing '
            f'threshold found during the real survival analysis), not an '
            f'arbitrary 0.5.'
        )
        df = load_cohort()
        y = df['true_label'].values
        fused = df['fused_probability'].values
        risk_group = np.where(fused > RISK_THRESHOLD, 'High-risk', 'Low-risk')
        n_high, n_low = (risk_group == 'High-risk').sum(), (risk_group == 'Low-risk').sum()
        c1, c2 = st.columns(2)
        c1.metric('High-risk', int(n_high))
        c2.metric('Low-risk', int(n_low))

        fig, ax = plt.subplots(figsize=(6, 4))
        for label_val, label_name, color in [(0, 'pCR (true)', '#3498DB'), (1, 'Non-pCR (true)', '#E74C3C')]:
            mask = y == label_val
            ax.hist(fused[mask], bins=12, range=(0, 1), alpha=0.6, color=color, label=label_name)
        ax.axvline(RISK_THRESHOLD, color='black', linestyle='--', lw=1, label='Risk-group cutoff')
        ax.set_xlabel('Fused (SRP) probability'); ax.set_ylabel('Number of patients')
        ax.legend(fontsize=8)
        st.pyplot(fig)

        with st.expander('Show full cohort predictions table'):
            st.dataframe(
                df[['demo_id', 'true_label', 'spatial_probability', 'radiomics_probability',
                    'fused_probability']].assign(predicted_risk_group=risk_group),
                width='stretch',
            )

    elif step == '8. Survival Analysis':
        st.markdown(
            'Kaplan-Meier disease-free survival (DFS) curves by predicted risk '
            'group, with a log-rank test, computed on the real survival '
            'outcomes for the full de-identified cohort.'
        )
        df = load_cohort()
        fused = df['fused_probability'].values
        risk_group = np.where(fused > RISK_THRESHOLD, 'High-risk', 'Low-risk')
        dfs_time = df['dfs_time_months'].values
        dfs_event = df['dfs_event'].values
        high_mask = risk_group == 'High-risk'
        low_mask = ~high_mask

        fig, ax = plt.subplots(figsize=(6, 5))
        kmf = KaplanMeierFitter()
        for mask, label, color in [(high_mask, 'High-risk', '#E74C3C'), (low_mask, 'Low-risk', '#3498DB')]:
            kmf.fit(dfs_time[mask], event_observed=dfs_event[mask], label=f'{label} (n={mask.sum()})')
            kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
        ax.set_xlabel('Disease-free survival, months'); ax.set_ylabel('Survival probability')

        result = logrank_test(dfs_time[high_mask], dfs_time[low_mask],
                               event_observed_A=dfs_event[high_mask], event_observed_B=dfs_event[low_mask])
        ax.text(0.02, 0.05, f'Log-rank P = {result.p_value:.4f}', transform=ax.transAxes, fontsize=9)
        st.pyplot(fig)
        st.markdown(f"**Log-rank test (High-risk vs. Low-risk):** P = {result.p_value:.4f}")

    st.divider()
    st.caption(
        'Source: ESCC (esophageal squamous cell carcinoma) cohort, de-identified. '
        'Full manuscript pipeline: `../1_Source_Code/SRP/` '
        '(also public at https://github.com/CancerImageAI/SRP).'
    )


if __name__ == '__main__':
    main()
