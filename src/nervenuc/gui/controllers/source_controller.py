"""Controller for file inspection, axis resolution, and channel loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from nervenuc.io import load_tiff

from nervenuc.preparation import (
    DirectImageSelectionConfig,
    PreparedImageSet,
    PreparedVolumeSet,
    StackInspectionResult,
    compute_direct_selection_fingerprint,
    compute_preparation_fingerprint,
    inspect_stack,
    prepare_direct_image,
    prepare_stack,
    prepare_volume,
    resolve_axes,
)

from nervenuc.workflow import ImageAnalysisSession, ImageSourceProvenance, VolumeAnalysisSession
from nervenuc.workflow import (
    compute_analysis_config_fingerprint,
    compute_pipeline_fingerprint,
    compute_volume_analysis_config_fingerprint,
    compute_volume_pipeline_fingerprint,
)


from ..errors import InvalidDraftConfigError, NerveNucGuiError
from ..state import GuiSessionState, DisplayTransformSettings


class SourceController:
    """Coordinates source file opening, axis metadata resolution, and spatial preparation."""

    def __init__(self, state: GuiSessionState) -> None:
        self.state = state

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _verify_source_unchanged(self, path: Path) -> None:
        expected_id = self.state.source_id
        if expected_id is None or not expected_id.startswith("file_sha256:"):
            raise NerveNucGuiError("Inspected source is missing its authoritative SHA-256 identity.")
        expected_hash = expected_id.removeprefix("file_sha256:")
        observed_hash = self._hash_file(path)
        if observed_hash != expected_hash:
            raise NerveNucGuiError(
                "Source file changed after inspection; inspect it again before analysis."
            )

    def inspect_file(self, path: Path | str, declared_axes: str | None = None) -> StackInspectionResult:
        """Inspect file metadata and dimensions."""
        p = Path(path)
        if not p.is_file():
            raise NerveNucGuiError(f"File not found: {p}")

        if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
            from PySide6.QtGui import QImage
            image = QImage(str(p))
            if image.isNull():
                raise NerveNucGuiError('This image could not be read. Try TIFF, PNG, JPEG, or BMP.')
            count = 1 if image.isGrayscale() else 3
            insp = StackInspectionResult(shape=(image.height(), image.width()) if count == 1 else (count, image.height(), image.width()), axes='YX' if count == 1 else 'CYX', dtype='uint8', ndim=2 if count == 1 else 3, is_ome=False, size_x=image.width(), size_y=image.height(), size_c=count)
        else:
            insp = inspect_stack(p, declared_axes=declared_axes)
        self.state.loaded_path = p
        self.state.inspection = insp
        self.state.loaded_raw_array = None
        self.state.quick_result = None
        self.state.source_metadata = None
        self.state.preview_axes = None
        self.state.display = DisplayTransformSettings()
        self.state.draft.marker_channels.clear()
        self.state.draft.threshold_drafts.clear()
        self.state.draft.metric_drafts.clear()
        self.state.draft.nuclear_channel = 0
        self.state.draft.z_start_1based = 1
        self.state.draft.z_stop_1based = insp.size_z or 1
        self.state.current_z_index = 0
        self.state.session_2d = None
        self.state.session_3d = None
        self.state.draft.declared_axes = declared_axes if declared_axes is not None else insp.axes

        # Compute source execution identifier and require an explicit mapping confirmation.
        self.state.source_id = f"file_sha256:{self._hash_file(p)}"
        self.state.draft.channel_mapping_confirmed = False

        # Initialize default channel mapping hints
        if not self.state.draft.marker_channels:
            if insp.size_c and insp.size_c > 1:
                self.state.draft.nuclear_channel = 0
                for c_idx in range(1, insp.size_c):
                    m_name = f"Marker_{c_idx}"
                    if insp.channel_names and c_idx < len(insp.channel_names):
                        m_name = insp.channel_names[c_idx]
                    self.state.draft.marker_channels[m_name] = c_idx


        self.state.mark_draft_modified()
        return insp

    def load_source_preview(self) -> None:
        """Cache source pixels for display without preparing a scientific session."""
        path = self.state.loaded_path
        if path is None:
            raise NerveNucGuiError("Open an image before previewing it.")
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
            from PySide6.QtGui import QImage
            image = QImage(str(path)).convertToFormat(QImage.Format.Format_RGB888)
            pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()*3].reshape(image.height(), image.width(), 3).copy()
            self.state.loaded_raw_array = pixels[:, :, 0] if self.state.inspection.size_c == 1 else np.moveaxis(pixels, -1, 0)
            self.state.preview_axes = self.state.inspection.axes
            self.state.display.channel_color = {c: ('Gray' if self.state.inspection.size_c == 1 else ('Red', 'Green', 'Blue')[c]) for c in range(self.state.inspection.size_c)}
            self._verify_source_unchanged(path)
            return
        loaded = load_tiff(path)
        self.state.source_metadata = loaded.metadata
        layout = resolve_axes(
            loaded.data.shape,
            declared_axes=self.state.draft.declared_axes,
            metadata_axes=loaded.metadata.axes,
            is_ome=loaded.metadata.is_ome,
        )
        self._verify_source_unchanged(path)
        self.state.preview_axes = layout.axes
        self.state.loaded_raw_array = loaded.data
        palette = ("Blue", "Green", "Red", "Magenta", "Cyan", "Yellow")
        count = self.state.inspection.size_c or 1
        for channel in range(count):
            self.state.display.channel_color[channel] = palette[channel % len(palette)] if count > 1 else "Gray"
            self.state.display.channel_visible[channel] = True

    def source_preview_planes(self) -> dict[int, np.ndarray]:
        """Return source-channel XY planes; preview time defaults explicitly to frame 1."""
        raw = self.state.loaded_raw_array
        axes = self.state.preview_axes
        if raw is None or axes is None:
            return {}
        indices = tuple(
            min(self.state.current_z_index, raw.shape[i] - 1) if axis == "Z"
            else min(self.state.draft.time_index or 0, raw.shape[i]-1) if axis == "T" else slice(None)
            for i, axis in enumerate(axes)
        )
        selected = raw[indices]
        remaining = "".join(axis for axis in axes if axis not in "ZT")
        target = "CYX" if "C" in remaining else "YX"
        selected = np.transpose(selected, tuple(remaining.index(axis) for axis in target))
        return dict(enumerate(selected)) if "C" in remaining else {0: selected}

    def prepare_and_load_session(self) -> None:
        """Prepare raw image/stack data and initialize the active workflow session according to mode."""
        if self.state.loaded_path is None or self.state.inspection is None:
            raise NerveNucGuiError("No source image has been inspected.")

        mode = self.state.draft.mode
        p = self.state.loaded_path
        src_id = self.state.source_id or str(p)
        if self.state.inspection.size_c and self.state.inspection.size_c > 1:
            if not self.state.draft.channel_mapping_confirmed:
                raise NerveNucGuiError(
                    "Confirm the nuclear and marker channel mapping before loading this multichannel source."
                )
        self._verify_source_unchanged(p)

        if mode == "direct_2d":
            selection = DirectImageSelectionConfig(
                axes=self.state.draft.declared_axes,
                nuclear_channel=self.state.draft.nuclear_channel,
                marker_channels=dict(self.state.draft.marker_channels),
            )
            prepared = prepare_direct_image(p, selection)
            self._verify_source_unchanged(p)
            cfg_2d = self.state.draft.build_core_2d_config()
            sess = ImageAnalysisSession(cfg_2d)
            source_hash = src_id.removeprefix("file_sha256:") if src_id.startswith("file_sha256:") else None
            config_fp = compute_analysis_config_fingerprint(cfg_2d)
            prep_fp = compute_direct_selection_fingerprint(selection)
            prov_2d = ImageSourceProvenance(
                source_kind="direct_2d",
                source_id=src_id,
                source_hash=source_hash,
                metadata_axes=prepared.provenance.metadata_axes,
                declared_axes=prepared.provenance.declared_axes,
                effective_axes=prepared.provenance.effective_axes,
                nuclear_channel=selection.nuclear_channel,
                marker_channels=dict(selection.marker_channels),
                preparation_fingerprint=prep_fp,
                pipeline_fingerprint=compute_pipeline_fingerprint(config_fp, prep_fp),
            )
            sess.set_image(
                prepared.nuclear_image,
                dict(prepared.marker_images),
                source_provenance=prov_2d,
            )
            self.state.session_2d = sess
            self.state.session_3d = None

        elif mode == "projected_2d":
            prep_cfg = self.state.draft.build_core_stack_preparation_config(self.state.inspection.size_z)
            prepared: PreparedImageSet = prepare_stack(p, prep_cfg)
            self._verify_source_unchanged(p)
            cfg_2d = self.state.draft.build_core_2d_config()
            sess = ImageAnalysisSession(cfg_2d)
            prep_fp = compute_preparation_fingerprint(prep_cfg)
            config_fp = compute_analysis_config_fingerprint(cfg_2d)
            nuc_prov = prepared.provenance["nuclear"]
            source_hash = src_id.removeprefix("file_sha256:") if src_id.startswith("file_sha256:") else None
            prov_proj = ImageSourceProvenance(
                source_kind="projected_stack",
                source_id=src_id,
                source_hash=source_hash,
                metadata_axes=nuc_prov.metadata_axes,
                declared_axes=nuc_prov.declared_axes,
                effective_axes=nuc_prov.effective_axes,
                time_index=nuc_prov.time_index,
                nuclear_channel=prep_cfg.nuclear_channel,
                marker_channels=dict(prep_cfg.marker_channels),
                z_start=nuc_prov.z_start,
                z_stop=nuc_prov.z_stop,
                projection_method=nuc_prov.method,
                physical_size_z_um=nuc_prov.physical_size_z_um,
                preparation_fingerprint=prep_fp,
                pipeline_fingerprint=compute_pipeline_fingerprint(config_fp, prep_fp),
            )
            sess.set_image(prepared.nuclear_image, dict(prepared.marker_images), source_provenance=prov_proj)
            self.state.session_2d = sess
            self.state.session_3d = None



        elif mode == "volume_3d":
            sel_cfg = self.state.draft.build_core_volume_selection_config(self.state.inspection.size_z)
            prepared_vol: PreparedVolumeSet = prepare_volume(p, sel_cfg)
            self._verify_source_unchanged(p)
            vol_cfg = self.state.draft.build_core_volume_config(self.state.inspection.size_z)
            config_fp = compute_volume_analysis_config_fingerprint(vol_cfg)
            sess_3d = VolumeAnalysisSession()
            sess_3d.set_volume(
                prepared_vol,
                source_id=src_id,
                source_hash=src_id.removeprefix("file_sha256:") if src_id.startswith("file_sha256:") else None,
                config_fingerprint=config_fp,
                pipeline_fingerprint=compute_volume_pipeline_fingerprint(config_fp),
                analysis_config=vol_cfg,
            )
            self.state.session_3d = sess_3d
            self.state.session_2d = None


        self.state.results_stale = False
        self.state.configuration_differs_from_run = False
        self.state.notify()
