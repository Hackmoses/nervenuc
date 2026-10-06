"""Controller for animal grouping, study summaries, and tabular exports."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from nervenuc.export import (
    AnimalGroupAssignment,
    GroupTableResult,
    prepare_group_tables,
    write_dataframe_csv,
    write_group_csv,
    write_group_excel,
)
from nervenuc.workflow import StudyAnalysisSession

from ..errors import NerveNucGuiError
from ..state import GuiSessionState


class ExportController:
    """Coordinates study aggregation, experimental group assignments, and file exports."""

    def __init__(self, state: GuiSessionState) -> None:
        self.state = state
        self.group_assignments: dict[str, str] = {}  # animal_id -> group_name

    def set_animal_group(self, animal_id: str, group_name: str) -> None:
        """Assign an animal replicate to an experimental group."""
        if not group_name.strip():
            self.group_assignments.pop(animal_id, None)
        else:
            self.group_assignments[animal_id] = group_name.strip()
        self.state.notify()

    def clear_group_assignments(self) -> None:
        """Clear assignments when a new authoritative study replaces prior results."""
        self.group_assignments.clear()
        self.state.notify()

    def _study_animal_ids(self) -> set[str]:
        if not self.state.study_session.samples:
            return set()
        aggregation = self.state.study_session.aggregate()
        return {str(value) for value in aggregation.animal_metrics["animal_id"].unique()}

    def groups_complete(self) -> bool:
        animal_ids = self._study_animal_ids()
        return bool(animal_ids) and set(self.group_assignments) == animal_ids

    def can_export_reproducible(self) -> bool:
        if self.state.results_stale or self.state.configuration_differs_from_run:
            return False
        if self.state.batch_result is not None:
            statuses = list(self.state.batch_result.sample_status["status"])
            return bool(statuses) and all(status == "completed" for status in statuses)
        return bool(self.state.study_session.samples) or self.state.image_session is not None

    def get_animal_metrics(self) -> pd.DataFrame:
        """Aggregate study session and return pooled animal-level metrics."""
        agg = self.state.study_session.aggregate()
        return agg.animal_metrics

    def export_csv(
        self,
        output_path: Path | str,
        table_kind: str = "wide",
        metric_name: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Export study results to CSV using validated export engine."""
        p = Path(output_path)
        agg = self.state.study_session.aggregate()

        if self.group_assignments:
            assigns = tuple(
                AnimalGroupAssignment(animal_id=a, group=g)
                for a, g in self.group_assignments.items()
            )
            group_res = prepare_group_tables(agg, assigns)
            if p.suffix.lower() == ".csv":
                write_dataframe_csv(group_res.long_table, p, overwrite=overwrite)
            else:
                p.mkdir(parents=True, exist_ok=True)
                write_group_csv(group_res, p, overwrite=overwrite)
        else:
            # Direct export of animal metrics table
            write_dataframe_csv(agg.animal_metrics, p, overwrite=overwrite)

        return p


    def export_excel(
        self,
        output_path: Path | str,
        overwrite: bool = False,
    ) -> Path:
        """Export study results to multi-sheet Excel workbook using validated export engine."""
        p = Path(output_path)
        agg = self.state.study_session.aggregate()

        animal_ids = {str(value) for value in agg.animal_metrics["animal_id"].unique()}
        if set(self.group_assignments) != animal_ids:
            missing = sorted(animal_ids - set(self.group_assignments))
            raise NerveNucGuiError(
                "Excel group export requires an explicit group for every animal. "
                f"Unassigned animals: {missing}."
            )
        assigns = tuple(
            AnimalGroupAssignment(animal_id=a, group=g)
            for a, g in self.group_assignments.items()
        )


        group_res = prepare_group_tables(agg, assigns)
        write_group_excel(group_res, p, overwrite=overwrite)
        return p

    def export_reproducible_bundle(
        self,
        output_dir: Path | str,
        include_objects: bool = False,
        write_csv: bool = True,
        write_excel: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Export full reproducible analysis bundle from study or active image session."""
        from nervenuc.export import prepare_reproducible_bundle, write_reproducible_bundle

        if self.state.results_stale or self.state.configuration_differs_from_run:
            raise NerveNucGuiError(
                "Cannot export reproducible results while analysis settings differ from the executed run. "
                "Re-run the affected analysis stages first."
            )
        if not self.can_export_reproducible():
            raise NerveNucGuiError(
                "No scientifically complete batch/study is available for reproducible export."
            )

        assigns = tuple(
            AnimalGroupAssignment(animal_id=a, group=g)
            for a, g in self.group_assignments.items()
        ) if self.group_assignments else None

        # Prefer study session if samples exist, otherwise active image session
        if len(self.state.study_session.samples) > 0:
            target_session = self.state.study_session
        elif self.state.image_session is not None:
            target_session = self.state.image_session
        else:
            raise NerveNucGuiError("No completed analysis session available for export.")

        try:
            bundle = prepare_reproducible_bundle(
                target_session,
                group_assignments=assigns,
                include_objects=include_objects,
            )
            return write_reproducible_bundle(
                bundle,
                output_dir,
                write_csv=write_csv,
                write_excel=write_excel,
                overwrite=overwrite,
            )
        except Exception as e:
            raise NerveNucGuiError(f"Reproducible export failed: {e}") from e
