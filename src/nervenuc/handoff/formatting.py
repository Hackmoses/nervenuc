"""Researcher-readable formatting for analysis handoff reports."""

from __future__ import annotations

from .models import AnalysisHandoffResult


def format_analysis_handoff(result: AnalysisHandoffResult) -> str:
    """Format the AnalysisHandoffResult into a deterministic, researcher-readable text report.

    Args:
        result: AnalysisHandoffResult object.

    Returns:
        Formatted multi-line text report.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("NerveNuc Analysis Readiness Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Biological replicate: Animal")

    mr_df = result.metric_readiness
    gr_df = result.group_readiness

    metric_names = list(mr_df["metric_name"])
    groups = list(dict.fromkeys(gr_df["group"]))

    lines.append(f"Metric count: {len(metric_names)}")
    lines.append(f"Group count: {len(groups)} ({', '.join(groups)})")
    lines.append("")

    for _, m_row in mr_df.iterrows():
        m_name = m_row["metric_name"]
        status = m_row["readiness_status"]

        lines.append("-" * 80)
        lines.append(f"Metric: {m_name}")
        lines.append("-" * 80)

        if status == "ready":
            status_str = "Ready"
        elif status == "ready_with_missing_values":
            status_str = "Ready (with missing values)"
        else:
            status_str = "Not Ready"

        lines.append(f"NerveNuc analysis handoff status: {status_str}")
        lines.append(f"Effective numerator: {m_row['numerator_filter']}")
        lines.append(f"Denominator: {m_row['denominator_filter']}")
        lines.append("")

        lines.append("Groups:")
        g_subset = gr_df[gr_df["metric_name"] == m_name]
        max_g_len = max(len(str(g)) for g in g_subset["group"]) if len(g_subset) > 0 else 0
        for _, g_row in g_subset.iterrows():
            g_str = str(g_row["group"]).ljust(max(max_g_len, 4))
            nt = g_row["n_total"]
            nv = g_row["n_valid"]
            nm = g_row["n_missing"]
            lines.append(f"  {g_str}  n_total={nt}  n_valid={nv}  n_missing={nm}")
        lines.append("")

        total_eq = "Yes" if m_row["total_group_sizes_equal"] else "No"
        valid_eq = "Yes" if m_row["valid_group_sizes_equal"] else "No"
        has_missing = "Yes" if m_row["has_missing_values"] else "No"

        lines.append(f"Total group sizes equal: {total_eq}")
        lines.append(f"Valid group sizes equal: {valid_eq}")
        lines.append(f"Missing observations: {has_missing}")
        lines.append("")

        if status == "not_ready":
            lines.append("One or more groups have no valid animal-level observation for this metric.")
            lines.append("")

        lines.append("Available data:")
        lines.append("  - Animal-level biological replicate observations")
        lines.append("  - Raw numerator/denominator count provenance")
        lines.append("  - Group descriptive statistics")
        lines.append("  - GraphPad Prism-ready wide tables (prepared in memory)")
        lines.append("  - Companion animal-ID alignment table (prepared in memory)")
        lines.append("  - CSV export: Supported")
        lines.append("  - Multi-sheet Excel export: Supported")
        lines.append("")

        lines.append("Inferential test: Not selected by NerveNuc.")
        lines.append("The researcher should select the statistical method based on the experimental design and analysis plan.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
