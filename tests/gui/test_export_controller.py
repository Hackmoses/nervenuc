"""Tests for ExportController group assignment and file exports."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from nervenuc.aggregation import SampleIdentity
from nervenuc.gui.controllers import ExportController
from nervenuc.gui.state import GuiSessionState
from nervenuc.summary import BiologicalSummaryResult


from unittest.mock import MagicMock
from nervenuc.summary import BiologicalMetricDefinition, BiologicalSummaryResult, PhenotypeFilter

def _create_mock_summary_result() -> BiologicalSummaryResult:
    df = pd.DataFrame([
        {
            "metric_name": "SOX10_pos_pct",
            "numerator_count": 10,
            "denominator_count": 50,
            "percentage": 20.0,
            "denominator_filter": "ALL",
            "numerator_filter": "SOX10+",
        }
    ])
    mock_class = MagicMock()
    mock_class.source_id = "mock_sample"
    metric_def = BiologicalMetricDefinition(
        name="SOX10_pos_pct",
        denominator=PhenotypeFilter(),
        numerator=PhenotypeFilter({"SOX10": True}),
    )
    return BiologicalSummaryResult(
        metrics=df,
        definitions=(metric_def,),
        classification_result=mock_class,
    )




def test_export_controller_csv_and_excel(tmp_path: Path) -> None:
    """Test group assignment and exporting to CSV and Excel."""
    state = GuiSessionState()
    res1 = _create_mock_summary_result()
    res2 = _create_mock_summary_result()

    state.study_session.add_sample(SampleIdentity("s1", "A1"), res1)
    state.study_session.add_sample(SampleIdentity("s2", "A2"), res2)

    ctrl = ExportController(state)
    ctrl.set_animal_group("A1", "Control")
    ctrl.set_animal_group("A2", "Treated")

    csv_path = tmp_path / "results.csv"
    ctrl.export_csv(csv_path, overwrite=True)
    assert csv_path.is_file()

    xlsx_path = tmp_path / "results.xlsx"
    ctrl.export_excel(xlsx_path, overwrite=True)
    assert xlsx_path.is_file()


def test_reproducible_export_rejects_stale_gui_draft(tmp_path: Path) -> None:
    state = GuiSessionState()
    state.results_stale = True
    ctrl = ExportController(state)
    with pytest.raises(Exception, match="settings differ from the executed run"):
        ctrl.export_reproducible_bundle(tmp_path / "bundle")
