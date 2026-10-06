"""Optional preprocessing controls kept within the Analyze step."""
from dataclasses import asdict
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QCheckBox,QSpinBox,QDoubleSpinBox,QPushButton,QLabel
from ..preprocessing import PreprocessingConfig

class PreprocessingWidget(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state=state
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.toggle=QPushButton('Preprocessing options')
        self.toggle.setCheckable(True)
        layout.addWidget(self.toggle)
        self.options=QWidget()
        form=QFormLayout(self.options)
        self.background=QCheckBox('Subtract fluorescence background')
        self.radius=QSpinBox();self.radius.setRange(1,200);self.radius.setValue(50)
        self.radius.setToolTip('Choose a radius larger than the structures to preserve. Larger kernels take longer to process.')
        self.denoise=QCheckBox('Mild Gaussian denoising');self.denoise.setChecked(True)
        self.sigma=QDoubleSpinBox();self.sigma.setRange(0,5);self.sigma.setSingleStep(.1);self.sigma.setValue(1)
        self.illumination=QCheckBox('Estimate uneven illumination')
        self.field_sigma=QDoubleSpinBox();self.field_sigma.setRange(1,500);self.field_sigma.setValue(50)
        self.field_sigma.setToolTip('Use a scale much larger than the objects. This estimates shading from the sample, without a calibration reference.')
        self.normalize=QCheckBox('Scale detection intensity to 0–1')
        form.addRow(self.background);form.addRow('Rolling-ball radius (pixels):',self.radius)
        form.addRow(self.denoise);form.addRow('Gaussian sigma (pixels):',self.sigma)
        form.addRow(self.illumination);form.addRow('Illumination scale (pixels):',self.field_sigma)
        form.addRow(self.normalize)
        hint=QLabel('Detection uses processed copies. Primary signal measurements stay raw. Estimated illumination is not a calibrated flat-field correction; intensity scaling does not make different acquisitions quantitatively comparable.')
        hint.setWordWrap(True);form.addRow(hint)
        layout.addWidget(self.options)
        self.options.hide()
        self.toggle.toggled.connect(self.options.setVisible)
        for control in (self.background,self.denoise,self.illumination,self.normalize):
            control.toggled.connect(self.sync)
        for control in (self.radius,self.sigma,self.field_sigma): control.valueChanged.connect(self.sync)
        self._form=form
        self._refresh_enabled()

    def values(self):
        return asdict(PreprocessingConfig(self.background.isChecked(),self.radius.value(),self.denoise.isChecked(),self.sigma.value(),self.illumination.isChecked(),self.field_sigma.value(),self.normalize.isChecked()))

    def _refresh_enabled(self):
        self.radius.setEnabled(self.background.isChecked())
        self.sigma.setEnabled(self.denoise.isChecked())
        self.field_sigma.setEnabled(self.illumination.isChecked())

    def sync(self,*_args):
        self.state.draft.preprocessing_values=self.values()
        self._refresh_enabled()
        self.state.mark_draft_modified()

    def load_from_draft(self):
        config=PreprocessingConfig(**self.state.draft.preprocessing_values) if self.state.draft.preprocessing_values else PreprocessingConfig()
        values=[(self.background,config.subtract_background),(self.radius,config.background_radius_px),(self.denoise,config.denoise),(self.sigma,config.gaussian_sigma_px),(self.illumination,config.correct_illumination),(self.field_sigma,config.illumination_sigma_px),(self.normalize,config.normalize_intensity)]
        for control,value in values:
            control.blockSignals(True)
            control.setChecked(value) if isinstance(control,QCheckBox) else control.setValue(value)
            control.blockSignals(False)
        self._refresh_enabled()
