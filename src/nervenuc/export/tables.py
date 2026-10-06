"""Extractors for clean per-image, per-animal, and object-level tabular results."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nervenuc.aggregation import (
    AnimalAggregationResult,
    SampleIdentity,
    SampleSummary,
    aggregate_by_animal,
)
from nervenuc.export.engine import prepare_group_tables
from nervenuc.export.errors import (
    IncompleteWorkflowExportError,
    InvalidGroupAssignmentError,
    StaleExportError,
)
from nervenuc.export.models import AnimalGroupAssignment, GroupTableResult


def extract_per_image_table(
    session_data: Sequence[Any],
    group_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Assemble a clean tabular DataFrame with one row per technical observation.

    Args:
        session_data: Sequence of (sample_id, animal_id, session_or_summary) or (SampleIdentity, session_or_summary) pairs.
        group_map: Optional mapping of animal_id -> group name.

    Returns:
        pd.DataFrame containing per-image scientific results.
    """
    rows: list[dict[str, Any]] = []

    for item in session_data:
        if len(item) == 3:
            sample_id, animal_id, sess = item
            animal_id = animal_id or ""
        elif len(item) == 2:
            first, sess = item
            if isinstance(first, SampleIdentity):
                sample_id = first.sample_id
                animal_id = first.animal_id or ""
            else:
                sample_id = str(first)
                animal_id = ""
        else:
            continue

        group_name = group_map.get(animal_id, "") if group_map and animal_id else ""

        # Check session completion and staleness
        prov = getattr(sess, "source_provenance", None)
        source_id = getattr(prov, "source_id", "") or sample_id
        pipeline_fp = getattr(prov, "pipeline_fingerprint", "") or ""

        # Determine analysis mode
        mode = "direct_2d"
        if prov is not None:
            sk = getattr(prov, "source_kind", None)
            rep = getattr(prov, "representation", None)
            if sk == "projected_stack":
                mode = "projected_2d"
            elif rep == "volume_3d":
                mode = "volume_3d"

        # Extract summary result and classifications
        summary_res = getattr(sess, "summary_result", None)
        class_res = getattr(sess, "classification_result", None)
        reviewed_res = getattr(sess, "reviewed_result", None)

        if summary_res is None:
            # Maybe sess is already a BiologicalSummaryResult
            from nervenuc.summary import BiologicalSummary3DResult, BiologicalSummaryResult
            if isinstance(sess, (BiologicalSummaryResult, BiologicalSummary3DResult)):
                summary_res = sess
            else:
                raise IncompleteWorkflowExportError(
                    f"Sample '{sample_id}' has not completed summary calculation."
                )
        if class_res is None:
            class_res = getattr(summary_res, "classification_result", None)

        valid_count = 0
        if class_res is not None and hasattr(class_res, "phenotypes"):
            valid_count = len(class_res.phenotypes)
        elif hasattr(summary_res, "classification_result") and summary_res.classification_result is not None and hasattr(summary_res.classification_result, "phenotypes"):
            valid_count = len(summary_res.classification_result.phenotypes)
        elif reviewed_res is not None and hasattr(reviewed_res, "objects"):
            valid_count = len(reviewed_res.objects)
        elif hasattr(summary_res, "metrics") and len(summary_res.metrics) > 0:
            valid_count = int(summary_res.metrics.iloc[0]["denominator_count"])

        # Base row attributes
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "animal_id": animal_id if animal_id else None,
            "group": group_name if group_name else None,
            "source_id": source_id,
            "analysis_mode": mode,
            "pipeline_fingerprint": pipeline_fp,
            "valid_nucleus_count": valid_count,
        }

        # Marker positive counts from classifications
        if class_res is not None:
            df_class = getattr(class_res, "classifications", None)
            if isinstance(df_class, pd.DataFrame) and "marker_name" in df_class.columns and "is_positive" in df_class.columns:
                for m_name in df_class["marker_name"].unique():
                    m_sub = df_class[df_class["marker_name"] == m_name]
                    pos_count = int(m_sub["is_positive"].sum())
                    row[f"{m_name}_positive_count"] = pos_count

            # Phenotype counts
            df_phenos = getattr(class_res, "phenotypes", None)
            if isinstance(df_phenos, pd.DataFrame) and "phenotype" in df_phenos.columns:
                for p_name, p_group in df_phenos.groupby("phenotype"):
                    row[f"phenotype_{p_name}_count"] = len(p_group)

        # Biological metrics from summary_res
        for metric_row in summary_res.metrics.itertuples():
            m_name = metric_row.metric_name
            num = int(metric_row.numerator_count)
            den = int(metric_row.denominator_count)
            pct = float(metric_row.percentage) if getattr(metric_row, "percentage_defined", True) and den > 0 else None
            row[f"metric_{m_name}_numerator"] = num
            row[f"metric_{m_name}_denominator"] = den
            row[f"metric_{m_name}_percentage"] = pct
            row[f"metric_{m_name}_percentage_defined"] = (den > 0)

        # QC Status
        qc_status = "completed"
        if reviewed_res is not None:
            hist = getattr(reviewed_res, "history", ())
            if any("auto-accept" in getattr(a, "note", "") for a in hist):
                qc_status = "auto_accepted"
        row["qc_status"] = qc_status

        rows.append(row)

    return pd.DataFrame(rows)


def extract_per_animal_table(
    samples: Sequence[SampleSummary],
    definitions: Sequence[Any],
    group_assignments: Sequence[AnimalGroupAssignment] | None = None,
) -> tuple[pd.DataFrame, GroupTableResult | None]:
    """Aggregate sample observations by animal biological replicate and build group tables.

    Args:
        samples: Sequence of SampleSummary objects.
        definitions: Sequence of BiologicalMetricDefinition objects.
        group_assignments: Optional sequence of AnimalGroupAssignment objects.

    Returns:
        Tuple of (per_animal_dataframe, optional_GroupTableResult).
    """
    agg_res = aggregate_by_animal(samples)
    df_animal = agg_res.animal_metrics.copy()

    # Ensure zero-denominator percentages are represented as None rather than NaN
    # and add percentage_defined boolean column
    percentages_clean: list[float | None] = []
    percentages_def: list[bool] = []
    for row in df_animal.itertuples():
        den = row.denominator_count
        if den > 0:
            percentages_clean.append(float(row.percentage))
            percentages_def.append(True)
        else:
            percentages_clean.append(None)
            percentages_def.append(False)

    df_animal["percentage"] = percentages_clean
    df_animal["percentage_defined"] = percentages_def

    group_res: GroupTableResult | None = None
    if group_assignments:
        group_res = prepare_group_tables(agg_res, group_assignments)
        # Add group column to per-animal table
        group_dict = {a.animal_id: a.group for a in group_assignments}
        df_animal["group"] = [group_dict.get(aid, "") for aid in df_animal["animal_id"]]
        # Reorder columns so group is near animal_id
        cols = list(df_animal.columns)
        cols.remove("group")
        cols.insert(1, "group")
        df_animal = df_animal[cols]

    return df_animal, group_res


def extract_object_level_table(
    session_data: Sequence[Any],
) -> pd.DataFrame:
    """Extract object-level single-nucleus measurements, classifications, and phenotypes.

    Args:
        session_data: Sequence of (sample_id, animal_id, session) or (SampleIdentity, session) pairs.

    Returns:
        pd.DataFrame containing one row per active reviewed nucleus.
    """
    rows: list[dict[str, Any]] = []

    for item in session_data:
        if len(item) == 3:
            sample_id, animal_id, sess = item
            animal_id = animal_id or None
        elif len(item) == 2:
            first, sess = item
            if isinstance(first, SampleIdentity):
                sample_id = first.sample_id
                animal_id = first.animal_id or None
            else:
                sample_id = str(first)
                animal_id = None
        else:
            continue

        reviewed = getattr(sess, "reviewed_result", None)
        meas_res = getattr(sess, "measurement_result", None)
        class_res = getattr(sess, "classification_result", None)

        if reviewed is None or meas_res is None or class_res is None:
            continue

        df_objects = reviewed.objects
        df_meas = meas_res.measurements
        df_class = class_res.classifications
        df_phenos = class_res.phenotypes

        # Map current_label -> phenotype
        pheno_map = dict(zip(df_phenos["current_label"], df_phenos["phenotype"]))

        for obj_row in df_objects.itertuples():
            label_id = getattr(obj_row, "current_label", getattr(obj_row, "object_id", None))
            rev_id = getattr(obj_row, "review_id", f"REV-{label_id:06d}")
            qc_stat = getattr(obj_row, "qc_status", "accepted")

            # Centroids & size
            cx = getattr(obj_row, "centroid_x", None)
            cy = getattr(obj_row, "centroid_y", None)
            cz = getattr(obj_row, "centroid_z", None)
            area_px = getattr(obj_row, "area_px", None)
            voxel_count = getattr(obj_row, "voxel_count", None)

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "animal_id": animal_id,
                "object_id": f"AUTO-{label_id:06d}" if isinstance(label_id, int) else str(label_id),
                "review_id": rev_id,
                "qc_status": qc_stat,
                "centroid_x_px": cx,
                "centroid_y_px": cy,
            }
            if cz is not None:
                row["centroid_z_vox"] = cz
            if area_px is not None:
                row["area_px2"] = area_px
            if voxel_count is not None:
                row["volume_voxels"] = voxel_count
                row["volume_um3"] = getattr(obj_row, "volume_um3", None)

            # Measurements for this object
            obj_meas = df_meas[df_meas["current_label"] == label_id]
            for m_row in obj_meas.itertuples():
                m_name = m_row.marker_name
                row[f"{m_name}_mean_intensity"] = getattr(m_row, "mean_intensity", None)
                row[f"{m_name}_median_intensity"] = getattr(m_row, "median_intensity", None)
                row[f"{m_name}_max_intensity"] = getattr(m_row, "max_intensity", None)
                row[f"{m_name}_integrated_intensity"] = getattr(m_row, "integrated_intensity", None)

            # Classifications for this object
            obj_class = df_class[df_class["current_label"] == label_id]
            for c_row in obj_class.itertuples():
                m_name = c_row.marker_name
                row[f"{m_name}_positive"] = bool(c_row.is_positive)

            row["phenotype"] = pheno_map.get(label_id, "Unclassified")
            rows.append(row)

    return pd.DataFrame(rows)
