"""Filesystem serialization routines for CSV and multi-sheet Excel exports."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .errors import ExportFileExistsError, ExportWriteError
from .models import GroupTableResult


_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def _escape_spreadsheet_value(value: object) -> object:
    """Render untrusted text inert in CSV/Excel spreadsheet consumers."""
    if not isinstance(value, str) or not value:
        return value
    significant = value.lstrip(" ")
    if significant.startswith(_SPREADSHEET_FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_dataframe_for_spreadsheet(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with formula-capable strings escaped, including headers."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ExportWriteError(
            f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}."
        )
    safe = dataframe.copy(deep=True)
    safe.columns = [_escape_spreadsheet_value(str(column)) for column in safe.columns]
    for column in safe.columns:
        safe[column] = safe[column].map(_escape_spreadsheet_value)
    return safe


def write_dataframe_csv(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write one formula-safe CSV table."""
    out_file = Path(output_path)
    if out_file.exists() and not overwrite:
        raise ExportFileExistsError(
            f"Export target CSV file already exists and overwrite is False: '{out_file}'."
        )
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ExportWriteError(f"Failed to create directory for '{out_file}': {err}") from err

    handle = tempfile.NamedTemporaryFile(
        prefix=f".{out_file.stem}.",
        suffix=".tmp.csv",
        dir=out_file.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        sanitize_dataframe_for_spreadsheet(dataframe).to_csv(temp_path, index=False)
        os.replace(temp_path, out_file)
    except Exception as err:
        raise ExportWriteError(f"Failed to write CSV file '{out_file}': {err}") from err
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return out_file


def _validate_for_export(result: object) -> GroupTableResult:
    """Reject mutable or forged M8 tables before writing scientific output."""
    if not isinstance(result, GroupTableResult):
        raise ExportWriteError(f"result must be a GroupTableResult, got {type(result).__name__}.")
    try:
        from .engine import validate_group_table_result

        validate_group_table_result(result)
    except Exception as err:
        raise ExportWriteError(f"Group table integrity validation failed: {err}") from err
    return result


def write_group_csv(
    result: GroupTableResult,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write group analysis tables and manifest to separate CSV files in the specified directory.

    Args:
        result: GroupTableResult containing long table, wide tables, animal ID table, and manifest.
        output_dir: Target directory path for exported CSV files.
        overwrite: If True, overwrite existing target CSV files. If False, raise ExportFileExistsError.

    Returns:
        Dictionary mapping deterministic keys ('long_table', 'animal_ids', 'manifest', 'metric_001', ...)
        to the written Path objects.

    Raises:
        ExportFileExistsError: If any target CSV file already exists and overwrite is False.
        ExportWriteError: If filesystem directory creation or CSV writing fails.
    """
    result = _validate_for_export(result)

    out_dir = Path(output_dir)

    target_map: dict[str, Path] = {
        "long_table": out_dir / "animal_metrics_long.csv",
        "animal_ids": out_dir / "animal_ids.csv",
        "manifest": out_dir / "manifest.csv",
    }

    for row in result.manifest.itertuples():
        safe_key = f"metric_{row.metric_index:03d}"
        target_map[safe_key] = out_dir / row.csv_filename

    # Preflight target file existence before writing anything
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
    staged_map = {key: staging_dir / target.name for key, target in target_map.items()}
    committed: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        sanitize_dataframe_for_spreadsheet(result.long_table).to_csv(staged_map["long_table"], index=False)
        sanitize_dataframe_for_spreadsheet(result.animal_id_table).to_csv(staged_map["animal_ids"], index=False)
        sanitize_dataframe_for_spreadsheet(result.manifest).to_csv(staged_map["manifest"], index=False)

        for row in result.manifest.itertuples():
            safe_key = f"metric_{row.metric_index:03d}"
            wide_df = result.wide_tables[row.metric_name]
            sanitize_dataframe_for_spreadsheet(wide_df).to_csv(staged_map[safe_key], index=False)

        # Commit only after every coordinated file has been written. Existing
        # targets are moved aside first so a failed commit can restore them.
        for target in target_map.values():
            if target.exists():
                backup = backup_dir / target.name
                os.replace(target, backup)
                backed_up.append((target, backup))
        for key, target in target_map.items():
            os.replace(staged_map[key], target)
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
        raise ExportWriteError(f"Failed to write CSV files to '{out_dir}': {err}") from err
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)

    return target_map


def write_group_excel(
    result: GroupTableResult,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write group analysis tables and manifest to a multi-sheet Excel workbook (.xlsx).

    Args:
        result: GroupTableResult containing long table, wide tables, animal ID table, and manifest.
        output_path: Target .xlsx workbook file path.
        overwrite: If True, overwrite an existing workbook. If False, raise ExportFileExistsError.

    Returns:
        Path to the written Excel workbook.

    Raises:
        ExportFileExistsError: If target workbook already exists and overwrite is False.
        ExportWriteError: If directory creation or workbook writing fails.
    """
    result = _validate_for_export(result)

    out_file = Path(output_path)

    if out_file.exists() and not overwrite:
        raise ExportFileExistsError(
            f"Export target Excel file already exists and overwrite is False: '{out_file}'."
        )

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ExportWriteError(f"Failed to create directory for '{out_file}': {err}") from err

    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{out_file.stem}.",
        suffix=".tmp.xlsx",
        dir=out_file.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    try:
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            sanitize_dataframe_for_spreadsheet(result.long_table).to_excel(writer, sheet_name="Animal Metrics", index=False)
            sanitize_dataframe_for_spreadsheet(result.animal_id_table).to_excel(writer, sheet_name="Animal IDs", index=False)
            for row in result.manifest.itertuples():
                wide_df = result.wide_tables[row.metric_name]
                sanitize_dataframe_for_spreadsheet(wide_df).to_excel(writer, sheet_name=row.excel_sheet_name, index=False)
            sanitize_dataframe_for_spreadsheet(result.manifest).to_excel(writer, sheet_name="Manifest", index=False)
        if not temp_path.is_file():
            raise OSError("temporary workbook was not created")
        os.replace(temp_path, out_file)
    except Exception as err:
        raise ExportWriteError(f"Failed to write Excel workbook '{out_file}': {err}") from err
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return out_file


def write_reproducible_excel_workbook(
    bundle: Any,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write all tabular and metadata artifacts of a ReproducibleExportBundle to a multi-sheet Excel workbook.

    Args:
        bundle: ReproducibleExportBundle instance.
        output_path: Target .xlsx workbook path.
        overwrite: If True, overwrite existing file. If False, raise ExportFileExistsError.

    Returns:
        Path to the written Excel workbook.
    """
    import json

    out_file = Path(output_path)

    if out_file.exists() and not overwrite:
        raise ExportFileExistsError(
            f"Export target Excel file already exists and overwrite is False: '{out_file}'."
        )

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ExportWriteError(f"Failed to create directory for '{out_file}': {err}") from err

    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{out_file.stem}.",
        suffix=".tmp.xlsx",
        dir=out_file.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    try:
        try:
            qc_data = json.loads(bundle.qc_summary_json)
            prov_data = json.loads(bundle.provenance_json)
            config_data = json.loads(bundle.analysis_config_json)
            env_data = json.loads(bundle.reproducibility_env_json)
        except (TypeError, json.JSONDecodeError) as err:
            raise ExportWriteError(
                f"Reproducible bundle contains invalid metadata JSON: {err}"
            ) from err

        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            # 1. Per Image
            sanitize_dataframe_for_spreadsheet(bundle.per_image_table).to_excel(writer, sheet_name="Per Image", index=False, na_rep="")

            # 2. Per Animal (if available)
            if bundle.per_animal_table is not None:
                sanitize_dataframe_for_spreadsheet(bundle.per_animal_table).to_excel(writer, sheet_name="Per Animal", index=False, na_rep="")

            # 3. Objects (if available)
            if bundle.object_table is not None:
                sanitize_dataframe_for_spreadsheet(bundle.object_table).to_excel(writer, sheet_name="Objects", index=False, na_rep="")

            # 4. Group / Prism Wide Tables (if available)
            if bundle.group_tables is not None:
                for row in bundle.group_tables.manifest.itertuples():
                    wide_df = bundle.group_tables.wide_tables[row.metric_name]
                    sheet_name = str(row.excel_sheet_name)[:31]
                    sanitize_dataframe_for_spreadsheet(wide_df).to_excel(writer, sheet_name=sheet_name, index=False, na_rep="")

            # 5. QC Summary
            qc_df = pd.DataFrame(qc_data) if isinstance(qc_data, list) else pd.DataFrame([qc_data])
            sanitize_dataframe_for_spreadsheet(qc_df).to_excel(writer, sheet_name="QC Summary", index=False, na_rep="")

            # 6. Provenance
            prov_df = pd.DataFrame(prov_data) if isinstance(prov_data, list) else pd.DataFrame([prov_data])
            sanitize_dataframe_for_spreadsheet(prov_df).to_excel(writer, sheet_name="Provenance", index=False, na_rep="")

            # 7. Configuration Snapshot
            config_rows = [{"parameter": k, "value": str(v)} for k, v in config_data.items()]
            config_df = pd.DataFrame(config_rows)
            sanitize_dataframe_for_spreadsheet(config_df).to_excel(writer, sheet_name="Configuration", index=False, na_rep="")

            env_rows = [{"component": k, "version": str(v)} for k, v in env_data.items()]
            sanitize_dataframe_for_spreadsheet(pd.DataFrame(env_rows)).to_excel(writer, sheet_name="Environment", index=False, na_rep="")

            # 8. Methods Summary Narrative
            methods_lines = bundle.methods_summary_text.splitlines()
            methods_df = pd.DataFrame({"Methods Summary": methods_lines})
            sanitize_dataframe_for_spreadsheet(methods_df).to_excel(writer, sheet_name="Methods Summary", index=False)

        if not temp_path.is_file():
            raise OSError("temporary workbook was not created")
        os.replace(temp_path, out_file)
    except Exception as err:
        raise ExportWriteError(f"Failed to write reproducible Excel workbook '{out_file}': {err}") from err
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return out_file
