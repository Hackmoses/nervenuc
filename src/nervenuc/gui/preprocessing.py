"""Non-destructive, explicit preprocessing for the quick detection workflow."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import rolling_ball, ellipsoid_kernel

@dataclass(frozen=True)
class PreprocessingConfig:
    subtract_background: bool = False
    background_radius_px: int = 50
    denoise: bool = True
    gaussian_sigma_px: float = 1.0
    correct_illumination: bool = False
    illumination_sigma_px: float = 50.0
    normalize_intensity: bool = False

    def __post_init__(self):
        for name in ('subtract_background', 'denoise', 'correct_illumination', 'normalize_intensity'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be enabled or disabled.')
        if isinstance(self.background_radius_px, bool) or not isinstance(self.background_radius_px, int) or not 1 <= self.background_radius_px <= 200:
            raise ValueError('Background radius must be an integer from 1 to 200 pixels.')
        for name, lower, upper in (('gaussian_sigma_px', 0, 5), ('illumination_sigma_px', 1, 500)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name} must be finite and between {lower} and {upper}.')

    def to_dict(self):
        return dict(asdict(self), order=['background_subtraction', 'gaussian_denoising', 'estimated_illumination_correction', 'intensity_scaling'], stage='after_spatial_preparation_before_detection', stack_processing='independent_XY_planes; intensity_scaling_shared_across_selected_stack', background_kernel='ellipsoid; intensity height equals each plane dynamic range', primary_measurements='raw', detection_units='0_to_1' if self.normalize_intensity else 'processed_floating_point_intensity')


def preprocess_image(image, config):
    """Process YX or ZYX in floating point without modifying input arrays.

    Estimated illumination comes from the sample itself, not a calibration frame.
    Percentile scaling is for detection; raw signal measurements remain separate.
    """
    if image.ndim not in (2, 3) or image.size == 0 or not np.isfinite(image).all():
        raise ValueError('Preprocessing needs a nonempty finite 2D image or 3D stack.')
    output = np.array(image, dtype=np.float64, copy=True)
    planes = output[None] if output.ndim == 2 else output
    for plane in planes:
        if config.subtract_background:
            height = float(np.ptp(plane))
            if height == 0:
                plane.fill(0)
            else:
                radius = config.background_radius_px
                kernel = ellipsoid_kernel((2*radius+1, 2*radius+1), height)
                background = rolling_ball(plane, kernel=kernel)
                plane[:] = np.maximum(plane-background, 0)
        if config.denoise and config.gaussian_sigma_px > 0:
            plane[:] = gaussian_filter(plane, config.gaussian_sigma_px, mode='reflect')
        if config.correct_illumination:
            field = gaussian_filter(plane, config.illumination_sigma_px, mode='reflect')
            scale = float(np.mean(field))
            if scale > 0:
                floor = max(scale * .01, np.finfo(float).eps)
                plane[:] = plane * scale / np.maximum(field, floor)
    if config.normalize_intensity:
        low, high = np.percentile(output, [1, 99])
        output = np.clip((output-low)/(high-low), 0, 1) if high > low else np.zeros_like(output)
    return output
