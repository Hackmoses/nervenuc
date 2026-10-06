"""Public API for NerveNuc image preparation and Z-stack projection."""

from .axes import (
    extract_subvolume,
    resolve_axes,
)
from .errors import (
    AmbiguousImageAxesError,
    InvalidAxisLayoutError,
    InvalidChannelSelectionError,
    InvalidProjectionError,
    InvalidZSelectionError,
    NerveNucPreparationError,
)
from .direct import (
    DirectImagePreparationProvenance,
    DirectImageSelectionConfig,
    PreparedDirectImageSet,
    compute_direct_selection_fingerprint,
    prepare_direct_image,
)
from .fingerprint import compute_preparation_fingerprint
from .inspection import inspect_stack
from .models import (
    AxisLayout,
    PreparedImageSet,
    PreparedVolumeSet,
    ProjectionResult,
    StackInspectionResult,
    StackPreparationConfig,
    VolumePreparationProvenance,
    VolumeSelectionConfig,
    VoxelSpacing,
    ZSelection,
)
from .projection import (
    prepare_stack,
    project_stack,
)
from .volume import prepare_volume

__all__ = [
    # Models
    "AxisLayout",
    "DirectImageSelectionConfig",
    "DirectImagePreparationProvenance",
    "PreparedDirectImageSet",
    "ZSelection",
    "StackPreparationConfig",
    "ProjectionResult",
    "PreparedImageSet",
    "StackInspectionResult",
    "VoxelSpacing",
    "VolumeSelectionConfig",
    "VolumePreparationProvenance",
    "PreparedVolumeSet",
    # Functions
    "resolve_axes",
    "extract_subvolume",
    "project_stack",
    "prepare_stack",
    "prepare_volume",
    "inspect_stack",
    "compute_preparation_fingerprint",
    "compute_direct_selection_fingerprint",
    "prepare_direct_image",
    # Exceptions
    "NerveNucPreparationError",
    "InvalidAxisLayoutError",
    "AmbiguousImageAxesError",
    "InvalidChannelSelectionError",
    "InvalidZSelectionError",
    "InvalidProjectionError",
]
