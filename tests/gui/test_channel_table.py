"""Adversarial tests for explicit channel-role confirmation."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLineEdit

from nervenuc.gui.state import GuiSessionState
from nervenuc.gui.widgets import ChannelAssignmentTable
from nervenuc.preparation import StackInspectionResult


def test_channel_confirmation_requires_one_nuclear_and_unique_named_markers(qtbot: object) -> None:
    state = GuiSessionState()
    state.inspection = StackInspectionResult(
        shape=(3, 16, 16),
        axes="CYX",
        dtype="uint16",
        ndim=3,
        is_ome=False,
        size_x=16,
        size_y=16,
        size_c=3,
        channel_names=("DAPI", "SOX10", "BrdU"),
    )
    widget = ChannelAssignmentTable(state)
    qtbot.addWidget(widget)  # type: ignore[attr-defined]
    widget.populate_from_inspection()
    assert widget.confirm_mapping.isEnabled()

    second_role = widget.table.cellWidget(1, 2)
    assert isinstance(second_role, QComboBox)
    second_role.setCurrentText("Objects to count")
    assert not widget.confirm_mapping.isEnabled()

    second_role.setCurrentText("Signal to measure")
    second_name = widget.table.cellWidget(1, 3)
    third_name = widget.table.cellWidget(2, 3)
    assert isinstance(second_name, QLineEdit)
    assert isinstance(third_name, QLineEdit)
    third_name.setText(second_name.text())
    assert not widget.confirm_mapping.isEnabled()

    third_name.setText("BrdU_unique")
    assert widget.confirm_mapping.isEnabled()
    widget.confirm_mapping.setChecked(True)
    assert state.draft.channel_mapping_confirmed is True
