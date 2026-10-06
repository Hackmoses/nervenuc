"""Hardware-aware QGraphicsView canvas for responsive microscopy image display."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class MicroscopyCanvas(QGraphicsView):
    """Interactive multi-layer 2D viewport supporting pan, zoom, and coordinate tracking."""

    coordinates_hovered = Signal(int, int, float)  # (x, y, pixel_value)

    def __init__(self) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.black)

        self._base_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self.scene.addItem(self._base_item)
        self.scene.addItem(self._overlay_item)
        self._overlay_item.setZValue(10.0)

        self._zoom_factor = 1.15

    def set_base_image(self, qimg: QImage) -> None:
        """Update base microscopy image layer."""
        if qimg.isNull():
            self._base_item.setPixmap(QPixmap())
            return
        pix = QPixmap.fromImage(qimg)
        self._base_item.setPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))

    def set_overlay_image(self, qimg: QImage) -> None:
        """Update segmentation/QC overlay layer."""
        if qimg.isNull():
            self._overlay_item.setPixmap(QPixmap())
            return
        pix = QPixmap.fromImage(qimg)
        self._overlay_item.setPixmap(pix)

    def fit_image_to_view(self) -> None:
        """Fit image inside viewport."""
        rect = self.scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom on mouse wheel."""
        if event.angleDelta().y() > 0:
            self.scale(self._zoom_factor, self._zoom_factor)
        else:
            self.scale(1.0 / self._zoom_factor, 1.0 / self._zoom_factor)
