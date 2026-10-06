"""Accessible Quality Control review table and action panel."""

from __future__ import annotations

import pandas as pd
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..state import GuiSessionState


class QualityControlTable(QWidget):
    """Accessible interactive table for nucleus candidate review and status tracking."""

    object_selected = Signal(int, int, int)  # (z, y, x) centroid
    review_updated = Signal()

    def __init__(self, state: GuiSessionState) -> None:
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Status Summary Header
        self.lbl_summary = QLabel("QC Status: No active review session.")
        self.lbl_summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_summary)

        # Review Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Status", "Size", "Centroid (Z, Y, X)", "Note"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.table)

        # Action Buttons
        btn_layout = QGridLayout()
        self.btn_accept = QPushButton("Accept (A)")
        self.btn_reject = QPushButton("Reject (R)")
        self.btn_restore = QPushButton("Restore (U)")
        self.btn_auto = QPushButton("Auto-Accept All")
        self.btn_finalize = QPushButton("Finalize Review")
        self.btn_finalize.setStyleSheet("font-weight: bold;")

        btn_layout.addWidget(self.btn_accept, 0, 0)
        btn_layout.addWidget(self.btn_reject, 0, 1)
        btn_layout.addWidget(self.btn_restore, 0, 2)
        btn_layout.addWidget(self.btn_auto, 1, 0, 1, 2)
        btn_layout.addWidget(self.btn_finalize, 1, 2)

        layout.addLayout(btn_layout)

        self._shortcuts = []
        for key, button in (("A", self.btn_accept), ("R", self.btn_reject), ("U", self.btn_restore)):
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(button.click)
            self._shortcuts.append(shortcut)
        # Connect signals
        self.table.itemSelectionChanged.connect(self._on_row_selected)

    def refresh_from_session(self) -> None:
        """Populate table rows based on active 2D or 3D review session."""
        selected = self.table.selectedItems()
        selected_id = self.table.item(selected[0].row(), 0).text() if selected and self.table.item(selected[0].row(), 0) else None
        mode = self.state.draft.mode
        objects_df: pd.DataFrame | None = None
        status_dict: dict | None = None

        if mode in ("direct_2d", "projected_2d"):
            sess = self.state.session_2d
            if sess and sess.review_session:
                rs = sess.review_session
                objects_df = rs.result().objects
                total_cnt = len(rs._active_objects) + len(rs._retired_objects)
                unrev = sum(1 for rec in rs._active_objects.values() if rec.review_status == "unreviewed")
                acc = sum(1 for rec in rs._active_objects.values() if rec.review_status in ("accepted", "edited", "merged", "manual"))
                rej = sum(1 for rec in rs._retired_objects.values() if rec.retirement_reason == "rejected")
                status_dict = {
                    "total_count": total_cnt,
                    "unreviewed_count": unrev,
                    "accepted_count": acc,
                    "rejected_count": rej,
                    "is_complete": unrev == 0,
                }
        elif mode == "volume_3d":
            sess_3d = self.state.session_3d
            if sess_3d and sess_3d.review_session:
                objects_df = sess_3d.review_session.objects
                status_dict = sess_3d.review_session.status()

        quick = self.state.quick_result
        if quick is not None:
            if quick.mode == 'volume_3d':
                objects_df = quick.review.objects
                status_dict = quick.review.status()
            else:
                records = list(quick.review._active_objects.values()) + list(quick.review._retired_objects.values())
                objects_df = pd.DataFrame([dict(review_id=rec.review_id, review_status='rejected' if getattr(rec, 'retirement_reason', None) == 'rejected' else getattr(rec, "review_status", "deleted"), area_px=int(rec.mask.sum()), centroid_y=float(np.argwhere(rec.mask)[:, 0].mean()), centroid_x=float(np.argwhere(rec.mask)[:, 1].mean())) for rec in records])
                status_dict = dict(total_count=len(records), unreviewed_count=0, accepted_count=len(quick.objects), rejected_count=len(records)-len(quick.objects), is_complete=True)
        if objects_df is None or status_dict is None:
            self.table.setRowCount(0)
            self.lbl_summary.setText("QC Status: No active review session.")
            self.btn_finalize.setEnabled(False)
            return

        # Update Summary text
        tot = status_dict.get("total_count", 0)
        unrev = status_dict.get("unreviewed_count", 0)
        acc = status_dict.get("accepted_count", 0)
        rej = status_dict.get("rejected_count", 0)
        is_comp = status_dict.get("is_complete", False)
        self.lbl_summary.setText(
            f"Total: {tot}  |  ✓ Accepted: {acc}  |  ✕ Rejected: {rej}  |  ○ Unreviewed: {unrev}"
        )
        self.btn_finalize.setEnabled(is_comp)

        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setCurrentItem(None)
        self.table.setRowCount(len(objects_df))
        selected_row = None
        for r, (_, row) in enumerate(objects_df.iterrows()):
            # 3D review operations use the original numeric label; 2D uses stable review IDs.
            is_3d = (quick.mode if quick is not None else mode) == 'volume_3d'
            oid = str(row.get('current_label', r + 1) if is_3d else row.get('review_id', row.get('current_label', r + 1)))
            if oid == selected_id:
                selected_row = r
            st = str(row.get("review_status", "unreviewed"))
            size_val = str(row.get("voxel_count", row.get("area_px", "")))
            cz = int(round(float(row["centroid_z"]))) if "centroid_z" in row and pd.notna(row["centroid_z"]) else 0
            cy = int(round(float(row.get("centroid_y", 0))))
            cx = int(round(float(row.get("centroid_x", 0))))
            note = str(row.get("review_note", ""))

            # Format status with text icon
            st_text = "○ Unreviewed"
            if st == "accepted":
                st_text = "✓ Accepted"
            elif st == "rejected":
                st_text = "✕ Rejected"
            elif st == "edited":
                st_text = "✎ Edited"

            item_id = QTableWidgetItem(oid)
            item_st = QTableWidgetItem(st_text)
            item_sz = QTableWidgetItem(size_val)
            item_coord = QTableWidgetItem(f"({cz}, {cy}, {cx})")
            item_note = QTableWidgetItem(note)

            for item in (item_id, item_st, item_sz, item_coord, item_note):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(r, 0, item_id)
            self.table.setItem(r, 1, item_st)
            self.table.setItem(r, 2, item_sz)
            self.table.setItem(r, 3, item_coord)
            self.table.setItem(r, 4, item_note)

        if selected_row is not None:
            self.table.selectRow(selected_row)
            self.table.scrollToItem(self.table.item(selected_row, 0))
        self.table.blockSignals(False)
        if selected_row is not None:
            self._on_row_selected()
        self.review_updated.emit()

    def _on_row_selected(self) -> None:
        sel = self.table.selectedItems()
        if not sel:
            return
        r = sel[0].row()
        coord_item = self.table.item(r, 3)
        if coord_item:
            try:
                coords = coord_item.text().strip("()").split(",")
                z, y, x = int(coords[0]), int(coords[1]), int(coords[2])
                self.object_selected.emit(z, y, x)
            except Exception:
                pass
