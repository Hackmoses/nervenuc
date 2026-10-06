"""Fiji and ImageJ interoperability module for NerveNuc."""

from __future__ import annotations

from .bridge import FijiBridge
from .conversion import (
    SUPPORTED_CONVERSION_AXES,
    SUPPORTED_DTYPES,
    build_imagej_calibration,
    from_imagej,
    resolve_axis_conversion,
    restore_voxel_spacing,
    to_imagej,
    validate_array_for_conversion,
)
from .errors import (
    FijiConversionError,
    FijiExecutionError,
    FijiUnavailableError,
    InvalidRoiGeometryError,
    JavaUnavailableError,
    NerveNucFijiError,
    UnsupportedRoiError,
)
from .exchange import FijiExchangeSession
from .launcher import FijiLauncher, discover_fiji
from .models import (
    AxisConversion,
    FijiConfig,
    FijiExecutableInfo,
    FijiProvenance,
    ImageJCalibration,
)
from .qc import (
    FijiImportedRoiRecord,
    FijiRoiImportResult,
    FijiRoiSplitResult,
    export_review_rois,
    import_rois_as_manual,
    replace_with_fiji_rois,
)
from .roi import (
    SUPPORTED_ROI_TYPES,
    mask_to_roi,
    rasterize_roi,
    read_roi,
    read_roi_set,
    validate_roi_type,
    write_roi,
    write_roi_set,
)

__all__ = [
    "AxisConversion",
    "FijiBridge",
    "FijiConfig",
    "FijiConversionError",
    "FijiExchangeSession",
    "FijiExecutableInfo",
    "FijiExecutionError",
    "FijiImportedRoiRecord",
    "FijiLauncher",
    "FijiProvenance",
    "FijiRoiImportResult",
    "FijiRoiSplitResult",
    "FijiUnavailableError",
    "ImageJCalibration",
    "InvalidRoiGeometryError",
    "JavaUnavailableError",
    "NerveNucFijiError",
    "SUPPORTED_CONVERSION_AXES",
    "SUPPORTED_DTYPES",
    "SUPPORTED_ROI_TYPES",
    "UnsupportedRoiError",
    "build_imagej_calibration",
    "discover_fiji",
    "export_review_rois",
    "from_imagej",
    "import_rois_as_manual",
    "mask_to_roi",
    "rasterize_roi",
    "read_roi",
    "read_roi_set",
    "replace_with_fiji_rois",
    "resolve_axis_conversion",
    "restore_voxel_spacing",
    "to_imagej",
    "validate_array_for_conversion",
    "validate_roi_type",
    "write_roi",
    "write_roi_set",
]
