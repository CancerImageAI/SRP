"""
Visualize the original CT image and local texture feature maps for a single case.

Feature maps are produced by TextureFeatureMapGenerate.py (OptimizedSpaRadFramework)
and cached as gzip-pickled dicts keyed by feature class -> "{class}_{FeatureName}" ->
3D array (z, y, x), same shape/orientation as the case's image and mask volumes.

Esophageal tumors sit in the mediastinum, so the CT is displayed with the
mediastinal / soft-tissue window (default WL=40 HU, WW=350 HU) rather than a
lung or bone window, to give good soft-tissue contrast around the tumor.
"""

import gzip
import pickle
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class CaseFeatureMapVisualizer:
    """
    Load one case's CT image, tumor mask and cached texture feature maps,
    and plot the windowed CT alongside selected feature maps on the same slice.
    """

    def __init__(self, image_dir, mask_dir, feature_maps_dir, output_dir,
                 window_center=40, window_width=350):
        """
        Parameters
        ----------
        image_dir, mask_dir : str
            Directories containing "{case_id}.nii.gz" image/mask volumes.
        feature_maps_dir : str
            Directory containing cached "{case_id}_{params_hash}.pkl.gz" feature maps
            (the `feature_maps` subfolder produced by TextureFeatureMapGenerate.py).
        output_dir : str
            Directory to save the figure.
        window_center, window_width : float
            CT display window in HU. Default (40, 350) is the mediastinal /
            soft-tissue window standard for evaluating esophageal tumors.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.feature_maps_dir = Path(feature_maps_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.window_center = window_center
        self.window_width = window_width

    # Which array axis (of the (z, y, x) volume) each view slices along.
    VIEW_AXIS = {'axial': 0, 'coronal': 1, 'sagittal': 2}

    def _load_image_mask(self, case_id):
        image_path = self.image_dir / f"{case_id}.nii.gz"
        mask_path = self.mask_dir / f"{case_id}.nii.gz"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image_sitk = sitk.ReadImage(str(image_path))
        mask_sitk = sitk.ReadImage(str(mask_path))

        image_arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)  # (z, y, x), HU
        mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)

        spacing = image_sitk.GetSpacing()  # (sx, sy, sz)
        # Direction cosine of the array's z-axis in physical (ITK/LPS) space: positive
        # means increasing z-index runs toward Superior. Use it to decide whether
        # coronal/sagittal panels need origin='lower' so Superior displays at the top.
        z_points_up = image_sitk.GetDirection()[8] > 0
        sag_cor_origin = 'lower' if z_points_up else 'upper'

        return image_arr, mask_arr, spacing, sag_cor_origin

    def _load_feature_maps(self, case_id):
        matches = sorted(self.feature_maps_dir.glob(f"{case_id}_*.pkl.gz"))
        if not matches:
            raise FileNotFoundError(
                f"No cached feature maps found for case {case_id} in {self.feature_maps_dir}"
            )
        if len(matches) > 1:
            print(f"⚠ Multiple cached feature map files found for {case_id}, "
                  f"using the most recently modified: {matches[-1].name}")
            matches = [max(matches, key=lambda p: p.stat().st_mtime)]

        with gzip.open(matches[0], 'rb') as f:
            feature_maps = pickle.load(f)

        # Flatten from {feature_class: {full_key: array}} to {full_key: array}
        flat = {}
        for class_maps in feature_maps.values():
            flat.update(class_maps)
        return flat

    @classmethod
    def _select_slice_for_view(cls, mask_arr, view, idx=None):
        """Pick the slice with the largest tumor cross-section along `view`, unless given."""
        if idx is not None:
            return idx
        axis = cls.VIEW_AXIS[view]
        other_axes = tuple(a for a in range(3) if a != axis)
        areas = mask_arr.sum(axis=other_axes)
        return int(np.argmax(areas))

    @classmethod
    def _get_plane(cls, volume, view, idx):
        """Extract a 2D plane from a (z, y, x) volume for the given view."""
        if view == 'axial':
            return volume[idx, :, :]
        if view == 'coronal':
            return volume[:, idx, :]
        if view == 'sagittal':
            return volume[:, :, idx]
        raise ValueError(f"Unknown view '{view}', expected one of {list(cls.VIEW_AXIS)}")

    @staticmethod
    def _aspect_for_view(view, spacing):
        """Pixel aspect ratio (row spacing / column spacing) so anatomy isn't distorted."""
        sx, sy, sz = spacing
        if view == 'axial':
            return sy / sx
        if view == 'coronal':
            return sz / sx
        if view == 'sagittal':
            return sz / sy
        raise ValueError(f"Unknown view '{view}'")

    @staticmethod
    def _crop_bbox(mask_slice, margin):
        ys, xs = np.where(mask_slice > 0)
        y0, y1 = max(0, ys.min() - margin), min(mask_slice.shape[0], ys.max() + margin + 1)
        x0, x1 = max(0, xs.min() - margin), min(mask_slice.shape[1], xs.max() + margin + 1)
        return y0, y1, x0, x1

    def _apply_window(self, image_hu):
        low = self.window_center - self.window_width / 2
        high = self.window_center + self.window_width / 2
        windowed = np.clip(image_hu, low, high)
        return (windowed - low) / (high - low)

    @staticmethod
    def _pretty_name(full_key):
        feature_class, feature_name = full_key.split('_', 1)
        return f"{feature_class.upper()}: {feature_name}"

    def plot_case(self, case_id, feature_keys=None, views=('axial',), slice_idx=None,
                  margin=30, overlay_alpha=0.65, cmap='jet', panel_size=4.2, dpi=300):
        """
        Plot the windowed CT (with tumor contour) plus selected texture feature
        maps for one case, cropped around the tumor.

        With a single view (default) this produces one row: CT + feature maps.
        With multiple views (e.g. ('axial', 'coronal', 'sagittal')) it produces
        a grid with one row per view and the same CT/feature columns in each,
        giving an orthogonal (multi-planar) look at the tumor.

        Parameters
        ----------
        case_id : str
            Case identifier (matches "{case_id}.nii.gz" filenames).
        feature_keys : list of str, optional
            Feature map keys to display, e.g. 'glcm_Contrast', 'glrlm_RunEntropy'.
            Defaults to two representative features per texture class.
        views : sequence of {'axial', 'coronal', 'sagittal'}
            Which anatomical planes to show, one per row.
        slice_idx : int or dict, optional
            Slice index to display. Either a single int (applied to all requested
            views) or a dict mapping view name -> index. Defaults to the slice
            with the largest tumor cross-section in each view.
        margin : int
            Padding (voxels) around the tumor bounding box used for cropping.
        """
        if feature_keys is None:
            feature_keys = [
                'glcm_Contrast', 'glcm_JointEntropy',
                'glrlm_ShortRunEmphasis', 'glrlm_RunEntropy',
                'glszm_ZoneEntropy', 'glszm_GrayLevelNonUniformity',
            ]
        if isinstance(slice_idx, dict):
            slice_idx_per_view = slice_idx
        else:
            slice_idx_per_view = {view: slice_idx for view in views}

        image_arr, mask_arr, spacing, sag_cor_origin = self._load_image_mask(case_id)
        feature_maps = self._load_feature_maps(case_id)

        missing = [k for k in feature_keys if k not in feature_maps]
        if missing:
            available = sorted(feature_maps.keys())
            raise KeyError(f"Feature key(s) not found: {missing}. Available: {available}")

        view_origin = {'axial': 'upper', 'coronal': sag_cor_origin, 'sagittal': sag_cor_origin}

        # Precompute per-view slice index, crop, windowed CT and aspect ratio.
        view_data = {}
        for view in views:
            idx = self._select_slice_for_view(mask_arr, view, slice_idx_per_view.get(view))
            mask_plane = self._get_plane(mask_arr, view, idx)
            y0, y1, x0, x1 = self._crop_bbox(mask_plane, margin)
            ct_plane = self._apply_window(self._get_plane(image_arr, view, idx))[y0:y1, x0:x1]
            mask_crop = mask_plane[y0:y1, x0:x1]
            view_data[view] = {
                'idx': idx, 'bbox': (y0, y1, x0, x1),
                'ct': ct_plane, 'mask': mask_crop,
                'aspect': self._aspect_for_view(view, spacing),
                'origin': view_origin[view],
            }

        # Shared color scale per feature across all requested views, so the same
        # feature is directly comparable panel-to-panel.
        feature_ranges = {}
        for key in feature_keys:
            vals = []
            for view in views:
                d = view_data[view]
                idx = d['idx']
                y0, y1, x0, x1 = d['bbox']
                plane = self._get_plane(feature_maps[key], view, idx)[y0:y1, x0:x1]
                vals.append(plane[d['mask'] > 0])
            vals = np.concatenate(vals) if vals else np.array([])
            vmin, vmax = (np.percentile(vals, 2), np.percentile(vals, 98)) if len(vals) else (0, 1)
            if vmin == vmax:
                vmax = vmin + 1e-6
            feature_ranges[key] = (vmin, vmax)

        n_cols = 2 + len(feature_keys)
        n_rows = len(views)
        figsize = (panel_size * n_cols, panel_size * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)

        for row, view in enumerate(views):
            d = view_data[view]
            ct_plane, mask_crop, aspect, origin = d['ct'], d['mask'], d['aspect'], d['origin']
            window_label = f"WL {self.window_center} / WW {self.window_width}"

            # First column: windowed CT, no boundary marked
            ax = axes[row][0]
            ax.imshow(ct_plane, cmap='gray', vmin=0, vmax=1, aspect=aspect, origin=origin)
            title = f"{view.capitalize()} CT (slice {d['idx']})" if len(views) > 1 \
                else f"CT ({window_label})"
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

            # Second column: same windowed CT with tumor contour overlaid
            ax = axes[row][1]
            ax.imshow(ct_plane, cmap='gray', vmin=0, vmax=1, aspect=aspect, origin=origin)
            for contour in measure.find_contours(mask_crop.astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color='#E74C3C', linewidth=1.8)
            title = f"{view.capitalize()} CT + Tumor" if len(views) > 1 \
                else f"CT + Tumor ({window_label})"
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

            # Remaining columns: feature maps overlaid on grayscale CT, tumor region only
            for col, key in enumerate(feature_keys, start=2):
                ax = axes[row][col]
                idx = d['idx']
                y0, y1, x0, x1 = d['bbox']
                feat_slice = self._get_plane(feature_maps[key], view, idx)[y0:y1, x0:x1].astype(np.float32)
                feat_masked = np.where(mask_crop > 0, feat_slice, np.nan)
                vmin, vmax = feature_ranges[key]

                ax.imshow(ct_plane, cmap='gray', vmin=0, vmax=1, aspect=aspect, origin=origin)
                im = ax.imshow(feat_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                                alpha=overlay_alpha, aspect=aspect, origin=origin)
                for contour in measure.find_contours(mask_crop.astype(float), 0.5):
                    ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=1.0)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                title = self._pretty_name(key)
                if len(views) > 1:
                    title = f"{view.capitalize()} — {title}"
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.axis('off')

        view_tag = '_'.join(views)
        fig.suptitle(f"Case {case_id} ({', '.join(views)})", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_stem = self.output_dir / f"{case_id}_{view_tag}_feature_maps"
        plt.savefig(f"{out_stem}.pdf", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{out_stem}.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        print(f"✓ Saved case {case_id} ({', '.join(views)}) visualization to {out_stem}.pdf/.png")

        del feature_maps
        return out_stem


if __name__ == "__main__":

    # --- Configuration -------------------------------------------------
    # # scRNA-seq-CT cohort (default)
    # image_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Image/'
    # mask_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/Mask/'
    # feature_maps_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/FeatureMaps/feature_maps/'
    # output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/scRNA-seq-CT/visualizations/'
    # case_id = '<patient_id>'

    # Main Dataset002_ECResponse cohort
    image_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/Image/'
    mask_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/Mask/'
    feature_maps_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/cache/feature_maps/'
    output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/visualizations/'
    case_id = '<patient_id>'

    # TJH cohort
    # image_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Image/'
    # mask_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/Mask/'
    # feature_maps_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/FeatureMaps/feature_maps/'
    # output_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/TJH/visualizations/'
    # case_id = '<patient_id>'

    visualizer = CaseFeatureMapVisualizer(
        image_dir=image_dir,
        mask_dir=mask_dir,
        feature_maps_dir=feature_maps_dir,
        output_dir=output_dir,
        window_center=40,   # mediastinal / soft-tissue window, standard for esophageal CT
        window_width=350,
    )

    visualizer.plot_case(
        case_id=case_id,
        feature_keys=[
            'glcm_Contrast', 'glcm_JointEntropy',
            'glrlm_ShortRunEmphasis', 'glrlm_RunEntropy',
            'glszm_ZoneEntropy', 'glszm_GrayLevelNonUniformity',
        ],
        views=('axial', 'coronal', 'sagittal'),
    )
