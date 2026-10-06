"""High-level export bundle orchestration and filesystem persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from types import SimpleNamespace

import pandas as pd

from nervenuc.aggregation import (
    AnimalAggregationResult,
    SampleIdentity,
    SampleSummary,
    aggregate_by_animal,
)
from .errors import (
    ExportFileExistsError,
    ExportWriteError,
    IncompleteWorkflowExportError,
    StaleExportError,
)
from .methods import capture_runtime_environment, generate_methods_summary
from .models import (
    AnimalGroupAssignment,
    GroupTableResult,
    QCSummaryRecord,
    ReproducibleExportBundle,
)
from .tables import (
    extract_object_level_table,
    extract_per_animal_table,
    extract_per_image_table,
)
from .writers import sanitize_dataframe_for_spreadsheet, write_reproducible_excel_workbook


def _serialize_provenance(prov_obj: Any) -> Any:
    """Convert an ImageSourceProvenance or VolumeSourceProvenance to a clean dict."""
    if prov_obj is None:
        return {}
    if isinstance(prov_obj, Mapping):
        return {str(key): _serialize_provenance(value) for key, value in prov_obj.items()}
    if isinstance(prov_obj, (tuple, list)):
        return [_serialize_provenance(value) for value in prov_obj]
    if hasattr(prov_obj, "__dict__"):
        d = {}
        for k, v in prov_obj.__dict__.items():
            if not k.startswith("_"):
                d[k] = _serialize_provenance(v)
        return d
    if isinstance(prov_obj, (str, int, float, bool)):
        return prov_obj
    return str(prov_obj)


def prepare_reproducible_bundle(
    session: Any,
    group_assignments: Sequence[AnimalGroupAssignment] | None = None,
    include_objects: bool = False,
) -> ReproducibleExportBundle:
    """Assemble a complete, validated reproducibility bundle from an analysis session.

    Args:
        session: Completed ImageAnalysisSession, VolumeAnalysisSession, or StudyAnalysisSession.
        group_assignments: Optional sequence of AnimalGroupAssignment objects.
        include_objects: If True, include detailed single-nucleus object measurements table.
            Defaults to False to prevent large files unless requested.

    Returns:
        Immutable/validated ReproducibleExportBundle.

    Raises:
        IncompleteWorkflowExportError: If required analysis stages are incomplete.
        StaleExportError: If downstream results were invalidated.
    """
    from nervenuc.workflow.image import ImageAnalysisSession
    from nervenuc.workflow.study import StudyAnalysisSession
    from nervenuc.workflow.volume import VolumeAnalysisSession

    now_utc = datetime.now(timezone.utc).isoformat()
    env = capture_runtime_environment()

    # Case 1: StudyAnalysisSession
    if isinstance(session, StudyAnalysisSession):
        if not session.samples:
            raise IncompleteWorkflowExportError("Cannot export an empty study session.")

        # Validate that all samples have completed summary results
        for sid, s_summary in session.samples.items():
            if s_summary.summary_result is None:
                raise IncompleteWorkflowExportError(
                    f"Sample '{sid}' in study session has not completed summary calculation."
                )

        sample_list = list(session.samples.values())
        sample_ids = [s.identity.sample_id for s in sample_list]
        missing_prov = [sid for sid in sample_ids if sid not in session.sample_provenance]
        missing_cfg = [sid for sid in sample_ids if sid not in session.sample_configs]
        missing_qc = [sid for sid in sample_ids if sid not in session.sample_qc_records]
        if missing_prov or missing_cfg or missing_qc:
            details = []
            if missing_prov:
                details.append(f"source provenance missing for {missing_prov}")
            if missing_cfg:
                details.append(f"analysis configuration missing for {missing_cfg}")
            if missing_qc:
                details.append(f"finalized QC accounting missing for {missing_qc}")
            raise IncompleteWorkflowExportError(
                "Cannot create a reproducible study bundle: " + "; ".join(details) + "."
            )
        for sample in sample_list:
            sid = sample.identity.sample_id
            cfg = session.sample_configs[sid]
            if tuple(cfg.metrics) != tuple(sample.summary_result.definitions):
                raise StaleExportError(
                    f"Sample '{sid}' configuration metrics do not match its authoritative summary definitions."
                )
            actual_thresholds = getattr(sample.summary_result.classification_result, "thresholds", None)
            if actual_thresholds is not None and tuple(cfg.thresholds) != tuple(actual_thresholds):
                raise StaleExportError(
                    f"Sample '{sid}' configuration thresholds do not match its authoritative classification result."
                )
        provenance_list = [session.sample_provenance.get(s.identity.sample_id) for s in sample_list]
        effective_prov = [p for p in provenance_list if p is not None]

        # Check modality from provenance
        modality = "direct_2d"
        if effective_prov:
            first_p = effective_prov[0]
            if getattr(first_p, "source_kind", None) == "projected_stack":
                modality = "projected_2d"
            elif getattr(first_p, "representation", None) == "volume_3d":
                modality = "volume_3d"

        # Group assignments mapping
        effective_groups = group_assignments if group_assignments is not None else (
            tuple(session.group_assignments) if session.group_assignments else None
        )
        group_map = {a.animal_id: a.group for a in effective_groups} if effective_groups else {}

        # 1. Per-Image Table
        image_pairs = [
            (
                s.identity,
                SimpleNamespace(
                    summary_result=s.summary_result,
                    classification_result=s.summary_result.classification_result,
                    source_provenance=session.sample_provenance[s.identity.sample_id],
                    reviewed_result=None,
                ),
            )
            for s in sample_list
        ]
        per_image_df = extract_per_image_table(image_pairs, group_map=group_map)

        # 2. Per-Animal Table
        first_summary = sample_list[0].summary_result
        defs = getattr(first_summary, "definitions", None)
        if defs is None:
            # Reconstruct definitions from metric names
            from nervenuc.summary.models import BiologicalMetricDefinition, PhenotypeFilter
            defs = tuple(
                BiologicalMetricDefinition(name=r.metric_name, denominator=PhenotypeFilter())
                for r in first_summary.metrics.itertuples()
            )

        # Check if animal IDs are present
        has_animal_ids = any(bool(s.identity.animal_id) for s in sample_list)
        if has_animal_ids:
            per_animal_df, group_res = extract_per_animal_table(
                sample_list,
                defs,
                group_assignments=effective_groups,
            )
            animal_count = len(per_animal_df["animal_id"].unique())
        else:
            per_animal_df = None
            group_res = None
            animal_count = None

        # 3. Object-level table (omitted by default)
        object_df = None
        # (StudyAnalysisSession stores sample summaries; object table is populated when detailed sessions are provided)

        # 4. Config JSON
        config_data = {
            "study_sample_count": len(sample_list),
            "per_sample": {
                sid: session.sample_configs[sid].to_dict()
                for sid in sample_ids
            },
        }
        config_json = json.dumps(config_data, indent=2)

        # 5. Provenance JSON
        prov_dict = {
            s.identity.sample_id: _serialize_provenance(session.sample_provenance.get(s.identity.sample_id))
            for s in sample_list
        }
        provenance_json = json.dumps(prov_dict, indent=2)

        # 6. QC Summary JSON
        qc_records = [session.sample_qc_records[sid] for sid in sample_ids]
        qc_json = json.dumps([r.__dict__ for r in qc_records], indent=2)

        # 7. Methods Summary Text
        configs = [session.sample_configs[sid] for sid in sample_ids]
        config_signatures = {json.dumps(c.to_dict(), sort_keys=True) for c in configs}
        methods_config = configs[0] if len(config_signatures) == 1 else None
        methods_text = generate_methods_summary(
            config=methods_config,
            provenance_list=effective_prov,
            qc_records=qc_records,
            modality=modality,
            animal_count=animal_count,
            sample_count=len(sample_list),
        )
        if len(config_signatures) > 1:
            methods_text += (
                "\nConfiguration note: Scientific parameter settings varied across samples; "
                "the authoritative per-sample configurations are recorded in analysis_config.json.\n"
            )

        manifest = {
            "export_version": "1.0",
            "timestamp_utc": now_utc,
            "nervenuc_version": env["nervenuc_version"],
            "session_type": "StudyAnalysisSession",
            "sample_count": len(sample_list),
            "animal_count": animal_count or 0,
            "has_per_animal_table": per_animal_df is not None,
            "has_object_table": object_df is not None,
            "has_group_tables": group_res is not None,
            "modality": modality,
        }

        return ReproducibleExportBundle(
            per_image_table=per_image_df,
            per_animal_table=per_animal_df,
            group_tables=group_res,
            object_table=object_df,
            analysis_config_json=config_json,
            provenance_json=provenance_json,
            qc_summary_json=qc_json,
            reproducibility_env_json=json.dumps(env, indent=2),
            methods_summary_text=methods_text,
            manifest=manifest,
        )

    # Case 2: ImageAnalysisSession
    elif isinstance(session, ImageAnalysisSession):
        status = session.status()
        if not status["image_set"]:
            raise IncompleteWorkflowExportError("Cannot export session: image has not been loaded.")
        if not status["segmented"]:
            raise IncompleteWorkflowExportError("Cannot export session: segmentation has not been run.")
        if not status["review_complete"]:
            raise IncompleteWorkflowExportError("Cannot export session: quality review has not been finalized.")
        if not status["measured"]:
            raise IncompleteWorkflowExportError("Cannot export session: marker measurements have not been run.")
        if not status["classified"]:
            raise IncompleteWorkflowExportError("Cannot export session: marker classification has not been run.")
        if not status["summarized"] or session.summary_result is None:
            raise IncompleteWorkflowExportError("Cannot export session: biological metric summary is missing.")

        prov = session.source_provenance
        sample_id = getattr(prov, "source_id", "IMAGE-001") or "IMAGE-001"
        # Check if caller supplied animal_id
        animal_id = getattr(prov, "animal_id", None)

        modality = "direct_2d"
        if prov is not None and getattr(prov, "source_kind", None) == "projected_stack":
            modality = "projected_2d"

        # Group mapping
        group_map = {}
        if group_assignments and animal_id:
            group_map = {a.animal_id: a.group for a in group_assignments}

        # 1. Per-Image Table
        per_image_df = extract_per_image_table([(sample_id, animal_id, session)], group_map=group_map)

        # 2. Per-Animal Table (only when animal_id is present)
        if animal_id:
            identity = SampleIdentity(sample_id=sample_id, animal_id=animal_id)
            sample_summary = SampleSummary(identity=identity, summary_result=session.summary_result)
            per_animal_df, group_res = extract_per_animal_table(
                [sample_summary],
                session.config.metrics,
                group_assignments=group_assignments,
            )
            animal_count = 1
        else:
            per_animal_df = None
            group_res = None
            animal_count = None

        # 3. Object-level table (optional)
        object_df = extract_object_level_table([(sample_id, animal_id, session)]) if include_objects else None

        # 4. Config JSON
        config_json = json.dumps(session.config.to_dict(), indent=2)

        # 5. Provenance JSON
        provenance_json = json.dumps(_serialize_provenance(prov), indent=2)

        # 6. QC Summary JSON
        reviewed = session.reviewed_result
        source_seg = getattr(reviewed, "source_result", None)
        if source_seg is not None and hasattr(source_seg, "objects"):
            init_count = len(source_seg.objects)
        elif hasattr(reviewed, "objects"):
            init_count = len(reviewed.objects)
        else:
            init_count = 0

        final_count = len(reviewed.objects) if hasattr(reviewed, "objects") else 0
        hist = getattr(reviewed, "history", ())
        accepted_c = sum(1 for a in hist if getattr(a, "action_type", "") == "accept")
        rejected_c = sum(1 for a in hist if getattr(a, "action_type", "") in ("reject", "delete"))
        added_c = sum(1 for a in hist if getattr(a, "action_type", "") == "add")
        edited_c = sum(1 for a in hist if getattr(a, "action_type", "") in ("split", "merge", "set_note"))
        rev_state = "auto_accepted" if any("auto-accept" in getattr(a, "note", "") for a in hist) else "completed"

        final_objects = reviewed.objects
        if "review_origin" in final_objects.columns:
            accepted_c = int((final_objects["review_origin"] == "automated").sum())
            added_c = int((final_objects["review_origin"] == "manual_add").sum())
            edited_c = int(final_count - accepted_c - added_c)
        qc_rec = QCSummaryRecord(
            sample_id=sample_id,
            initial_automated_count=init_count,
            accepted_count=accepted_c,
            rejected_count=rejected_c,
            added_count=added_c,
            edited_count=edited_c,
            final_count=final_count,
            review_state=rev_state,
        )
        qc_json = json.dumps([qc_rec.__dict__], indent=2)

        # 7. Methods Summary Text
        methods_text = generate_methods_summary(
            config=session.config,
            provenance_list=[prov] if prov else None,
            qc_records=[qc_rec],
            modality=modality,
            animal_count=animal_count,
            sample_count=1,
        )

        manifest = {
            "export_version": "1.0",
            "timestamp_utc": now_utc,
            "nervenuc_version": env["nervenuc_version"],
            "session_type": "ImageAnalysisSession",
            "sample_count": 1,
            "animal_count": animal_count or 0,
            "has_per_animal_table": per_animal_df is not None,
            "has_object_table": object_df is not None,
            "has_group_tables": group_res is not None,
            "modality": modality,
        }

        return ReproducibleExportBundle(
            per_image_table=per_image_df,
            per_animal_table=per_animal_df,
            group_tables=group_res,
            object_table=object_df,
            analysis_config_json=config_json,
            provenance_json=provenance_json,
            qc_summary_json=qc_json,
            reproducibility_env_json=json.dumps(env, indent=2),
            methods_summary_text=methods_text,
            manifest=manifest,
        )

    # Case 3: VolumeAnalysisSession
    elif isinstance(session, VolumeAnalysisSession):
        if session.summary_result is None:
            raise IncompleteWorkflowExportError("Cannot export 3D volume session: summary result is missing.")
        if session.classification_result is None or session.reviewed_result is None:
            raise IncompleteWorkflowExportError("Cannot export 3D volume session: analysis stages incomplete.")
        if session.analysis_config is None:
            raise IncompleteWorkflowExportError(
                "Cannot export 3D volume session: authoritative executed analysis configuration is missing."
            )

        prov = session.source_provenance
        sample_id = getattr(prov, "source_id", "VOLUME-001") or "VOLUME-001"
        animal_id = getattr(prov, "animal_id", None)

        group_map = {}
        if group_assignments and animal_id:
            group_map = {a.animal_id: a.group for a in group_assignments}

        per_image_df = extract_per_image_table([(sample_id, animal_id, session)], group_map=group_map)

        if animal_id:
            identity = SampleIdentity(sample_id=sample_id, animal_id=animal_id)
            sample_summary = SampleSummary(identity=identity, summary_result=session.summary_result)
            per_animal_df, group_res = extract_per_animal_table(
                [sample_summary],
                session.summary_result.definitions if hasattr(session.summary_result, "definitions") else (),
                group_assignments=group_assignments,
            )
            animal_count = 1
        else:
            per_animal_df = None
            group_res = None
            animal_count = None

        object_df = extract_object_level_table([(sample_id, animal_id, session)]) if include_objects else None

        config_json = json.dumps(session.analysis_config.to_dict(), indent=2)
        provenance_json = json.dumps(_serialize_provenance(prov), indent=2)

        reviewed = session.reviewed_result
        source_seg = getattr(reviewed, "source_result", None)
        if source_seg is not None and hasattr(source_seg, "objects"):
            init_count = len(source_seg.objects)
        elif hasattr(reviewed, "objects"):
            init_count = len(reviewed.objects)
        else:
            init_count = 0
        final_count = len(reviewed.objects) if hasattr(reviewed, "objects") else 0
        hist = getattr(reviewed, "history", ())
        rejected_count = sum(1 for a in hist if getattr(a, "action_type", "") == "reject")
        qc_rec = QCSummaryRecord(
            sample_id=sample_id,
            initial_automated_count=init_count,
            accepted_count=final_count,
            rejected_count=rejected_count,
            added_count=0,
            edited_count=0,
            final_count=final_count,
            review_state="completed",
        )
        qc_json = json.dumps([qc_rec.__dict__], indent=2)

        methods_text = generate_methods_summary(
            config=session.analysis_config,
            provenance_list=[prov] if prov else None,
            qc_records=[qc_rec],
            modality="volume_3d",
            animal_count=animal_count,
            sample_count=1,
        )

        manifest = {
            "export_version": "1.0",
            "timestamp_utc": now_utc,
            "nervenuc_version": env["nervenuc_version"],
            "session_type": "VolumeAnalysisSession",
            "sample_count": 1,
            "animal_count": animal_count or 0,
            "has_per_animal_table": per_animal_df is not None,
            "has_object_table": object_df is not None,
            "has_group_tables": group_res is not None,
            "modality": "volume_3d",
        }

        return ReproducibleExportBundle(
            per_image_table=per_image_df,
            per_animal_table=per_animal_df,
            group_tables=group_res,
            object_table=object_df,
            analysis_config_json=config_json,
            provenance_json=provenance_json,
            qc_summary_json=qc_json,
            reproducibility_env_json=json.dumps(env, indent=2),
            methods_summary_text=methods_text,
            manifest=manifest,
        )

    else:
        raise IncompleteWorkflowExportError(f"Unsupported session type for export: {type(session).__name__}.")


def write_reproducible_bundle(
    bundle: ReproducibleExportBundle,
    output_dir: str | Path,
    write_csv: bool = True,
    write_excel: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write all artifacts of a ReproducibleExportBundle to the filesystem atomically.

    Args:
        bundle: ReproducibleExportBundle instance.
        output_dir: Destination directory path.
        write_csv: Whether to write canonical CSV and JSON files.
        write_excel: Whether to write the companion results.xlsx workbook.
        overwrite: If True, overwrite existing files. If False, raise ExportFileExistsError.

    Returns:
        Dictionary mapping artifact keys to their written Path destinations.

    Raises:
        ExportFileExistsError: If target directory or any target file exists and overwrite is False.
        ExportWriteError: If writing fails.
    """
    if not isinstance(bundle, ReproducibleExportBundle):
        raise ExportWriteError(f"bundle must be a ReproducibleExportBundle, got {type(bundle).__name__}.")

    out_dir = Path(output_dir)

    target_map: dict[str, Path] = {}
    if write_csv:
        target_map["per_image_csv"] = out_dir / "per_image_results.csv"
        if bundle.per_animal_table is not None:
            target_map["per_animal_csv"] = out_dir / "per_animal_results.csv"
        if bundle.group_tables is not None:
            wide_dir = out_dir / "per_animal_wide"
            for row in bundle.group_tables.manifest.itertuples():
                safe_key = f"wide_metric_{row.metric_index:03d}"
                target_map[safe_key] = wide_dir / row.csv_filename
        if bundle.object_table is not None:
            target_map["object_csv"] = out_dir / "object_results.csv"
        target_map["config_json"] = out_dir / "analysis_config.json"
        target_map["provenance_json"] = out_dir / "provenance.json"
        target_map["qc_summary_json"] = out_dir / "qc_summary.json"
        target_map["reproducibility_env_json"] = out_dir / "reproducibility_env.json"
        target_map["methods_summary_txt"] = out_dir / "methods_summary.txt"
        target_map["manifest_json"] = out_dir / "bundle_manifest.json"

    if write_excel:
        target_map["results_excel"] = out_dir / "results.xlsx"

    # Preflight check
    if not overwrite:
        for key, target_path in target_map.items():
            if target_path.exists():
                raise ExportFileExistsError(
                    f"Export target file already exists and overwrite is False: '{target_path}'."
                )

    try:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(exist_ok=True)
    except OSError as err:
        raise ExportWriteError(f"Failed to create output directory '{out_dir}': {err}") from err

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))
    backup_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.backup-", dir=out_dir.parent))
    staged_map: dict[str, Path] = {}
    for key, target in target_map.items():
        rel = target.relative_to(out_dir)
        staged_map[key] = staging_dir / rel

    committed: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []

    try:
        if write_csv:
            staged_map["per_image_csv"].parent.mkdir(parents=True, exist_ok=True)
            sanitize_dataframe_for_spreadsheet(bundle.per_image_table).to_csv(
                staged_map["per_image_csv"], index=False, na_rep=""
            )

            if bundle.per_animal_table is not None:
                staged_map["per_animal_csv"].parent.mkdir(parents=True, exist_ok=True)
                sanitize_dataframe_for_spreadsheet(bundle.per_animal_table).to_csv(
                    staged_map["per_animal_csv"], index=False, na_rep=""
                )

            if bundle.group_tables is not None:
                for row in bundle.group_tables.manifest.itertuples():
                    safe_key = f"wide_metric_{row.metric_index:03d}"
                    wide_path = staged_map[safe_key]
                    wide_path.parent.mkdir(parents=True, exist_ok=True)
                    wide_df = bundle.group_tables.wide_tables[row.metric_name]
                    sanitize_dataframe_for_spreadsheet(wide_df).to_csv(
                        wide_path, index=False, na_rep=""
                    )

            if bundle.object_table is not None:
                staged_map["object_csv"].parent.mkdir(parents=True, exist_ok=True)
                sanitize_dataframe_for_spreadsheet(bundle.object_table).to_csv(
                    staged_map["object_csv"], index=False, na_rep=""
                )

            staged_map["config_json"].write_text(bundle.analysis_config_json, encoding="utf-8")
            staged_map["provenance_json"].write_text(bundle.provenance_json, encoding="utf-8")
            staged_map["qc_summary_json"].write_text(bundle.qc_summary_json, encoding="utf-8")
            staged_map["reproducibility_env_json"].write_text(bundle.reproducibility_env_json, encoding="utf-8")
            staged_map["methods_summary_txt"].write_text(bundle.methods_summary_text, encoding="utf-8")
            staged_map["manifest_json"].write_text(json.dumps(bundle.manifest, indent=2), encoding="utf-8")

        if write_excel:
            write_reproducible_excel_workbook(bundle, staged_map["results_excel"], overwrite=True)

        # Move files from staging to target atomically
        for key, target in target_map.items():
            if target.exists():
                backup = backup_dir / target.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                backed_up.append((target, backup))

        for key, target in target_map.items():
            staged_file = staged_map[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_file, target)
            committed.append(target)

    except Exception as err:
        for target in committed:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        for target, backup in reversed(backed_up):
            try:
                if backup.exists():
                    os.replace(backup, target)
            except OSError:
                pass
        raise ExportWriteError(f"Failed to write reproducible export bundle to '{out_dir}': {err}") from err
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)

    return target_map
