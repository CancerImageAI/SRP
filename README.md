# Nature Research Code & Software Submission Package — SRP

This package contains everything required by the *Nature Research Code
and Software Submission Checklist* (see `nr-software-policy.pdf` in
the parent directory) for the manuscript's SRP (Spatial + Radiomics ->
Fusion) pipeline.

**Start here:**
- Reviewer wanting to run the demo: `2_Demo/README.md`
- Full required-content README (the checklist's 4 required sections): `3_Documentation/README.md`
- Full manuscript pipeline source: `1_Source_Code/SRP/`
- Exactly which checklist item is satisfied where, and what's still
  open for the authors to fill in: `CHECKLIST_MAPPING.md`

## Folder guide

| Folder | Contents |
|---|---|
| `1_Source_Code/` | Full pipeline source code (mirrors https://github.com/CancerImageAI/SRP) |
| `2_Demo/` | The single consolidated demo: a button-driven web app covering the complete SRP workflow (3D CT input image and segmentation viewed slice-by-slice → feature extraction → heterogeneity visualization → prediction, plus cohort-level ROC/risk-stratification/survival analysis), on real de-identified ESCC-cohort data |
| `3_Documentation/` | The required README, organized exactly per the checklist's 4 sections, plus the Supplementary Note (pseudocode) |
| `4_License_and_Repository/` | LICENSE (Creative Commons CC BY-NC-ND license) and open-source repository link |

## Quickest path to verify this all works

```bash
cd 2_Demo
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_demo.txt
streamlit run app.py
```
Takes under 2 minutes total; opens in your browser. See
`2_Demo/README.md` for what each of the 8 steps shows.
