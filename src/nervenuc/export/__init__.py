"""Result export support and GraphPad/CSV/Excel group tables."""

from .bundle import (
    prepare_reproducible_bundle,
    write_reproducible_bundle,
)
from .engine import (
    prepare_group_tables,
    validate_group_assignments,
    validate_group_table_result,
)
from .errors import (
    ExportFileExistsError,
    ExportWriteError,
    GroupTableConsistencyError,
    IncompleteWorkflowExportError,
    InvalidGroupAssignmentError,
    NerveNucExportError,
    StaleExportError,
)
from .methods import generate_methods_summary
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
from .writers import (
    sanitize_dataframe_for_spreadsheet,
    write_dataframe_csv,
    write_group_csv,
    write_group_excel,
    write_reproducible_excel_workbook,
)

__all__ = [
    "AnimalGroupAssignment",
    "ExportFileExistsError",
    "ExportWriteError",
    "GroupTableConsistencyError",
    "GroupTableResult",
    "IncompleteWorkflowExportError",
    "InvalidGroupAssignmentError",
    "NerveNucExportError",
    "QCSummaryRecord",
    "ReproducibleExportBundle",
    "StaleExportError",
    "extract_object_level_table",
    "extract_per_animal_table",
    "extract_per_image_table",
    "generate_methods_summary",
    "prepare_group_tables",
    "prepare_reproducible_bundle",
    "validate_group_assignments",
    "validate_group_table_result",
    "write_group_csv",
    "write_group_excel",
    "write_reproducible_bundle",
    "write_reproducible_excel_workbook",
    "sanitize_dataframe_for_spreadsheet",
    "write_dataframe_csv",
]
