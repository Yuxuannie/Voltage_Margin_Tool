"""Testable Phase 1 backend helpers for the Tkinter workbench."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from voltage_margin.core.config_loader import (
    DEFAULT_COLUMN_MAPPING,
    DEFAULT_POLICY,
    load_policy,
)
from voltage_margin.core.data_loader import load_all_data
from voltage_margin.core.margin_engine import build_margin_outputs, build_sensitivity_rows
from voltage_margin.core.normalizer import load_normalized_data
from voltage_margin.core.pass_rate_engine import (
    filter_results_by_waivers,
    results_to_dataframe,
    run_full_analysis,
)
from run_analysis import _arc_results_to_dataframe, _write_phase1_outputs


@dataclass
class Phase1RunConfig:
    data_dir: Path
    output_dir: Path
    column_map: Path = DEFAULT_COLUMN_MAPPING
    policy: Path = DEFAULT_POLICY
    corners: list[str] | None = None
    types: list[str] | None = None
    waiver1_enabled: bool = True
    optimistic_enabled: bool = True


@dataclass
class Phase1RunResult:
    output_dir: Path
    parameter_groups: int
    total_arcs: int
    pass_rate: pd.DataFrame


@dataclass
class OutputTables:
    pass_rate: pd.DataFrame
    per_arc: pd.DataFrame
    normalized: pd.DataFrame
    normalization_warnings: pd.DataFrame
    sensitivity: pd.DataFrame
    sensitivity_warnings: pd.DataFrame
    all_margins: pd.DataFrame
    optimistic_margins: pd.DataFrame
    source_rows: pd.DataFrame
    sensitivity_source_refs: pd.DataFrame
    margin_trace: pd.DataFrame


@dataclass
class MarginSummary:
    total_margins: int = 0
    ok_margins: int = 0
    needs_review: int = 0
    trace_rows: int = 0


def run_phase1_pipeline(config: Phase1RunConfig) -> Phase1RunResult:
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    bundles = load_all_data(data_dir, config.corners, config.types)
    if not bundles:
        raise ValueError("No supported .rpt files found in the selected input directory.")

    results = run_full_analysis(bundles)
    if not results:
        raise ValueError("No pass-rate results were produced from the selected input directory.")

    full_df = results_to_dataframe(results)
    pass_rate_df = filter_results_by_waivers(
        full_df,
        waiver1_enabled=config.waiver1_enabled,
        optimistic_enabled=config.optimistic_enabled,
    )
    policy = load_policy(config.policy)
    normalized_df, manifest_df, normalization_warnings_df = load_normalized_data(
        data_dir,
        corners=config.corners,
        types=config.types,
        column_mapping=config.column_map,
    )
    sensitivity_df, sensitivity_warnings_df = build_sensitivity_rows(normalized_df, policy)
    margin_outputs = build_margin_outputs(normalized_df, sensitivity_df, policy)
    pass_rate_path = output_dir / "pass_rate" / "pass_rate_results.csv"
    pass_rate_path.parent.mkdir(parents=True, exist_ok=True)
    pass_rate_df.to_csv(pass_rate_path, index=False)
    _write_phase1_outputs(
        output_dir=output_dir,
        manifest_df=manifest_df,
        normalized_df=normalized_df,
        normalization_warnings_df=normalization_warnings_df,
        pass_rate_df=pass_rate_df,
        pass_rate_output_path=pass_rate_path,
        arc_results_df=_arc_results_to_dataframe(results, normalized_df, policy),
        sensitivity_df=sensitivity_df,
        sensitivity_warnings_df=sensitivity_warnings_df,
        margin_outputs=margin_outputs,
    )
    return Phase1RunResult(
        output_dir=output_dir,
        parameter_groups=len(results),
        total_arcs=sum(r.total_arcs for r in results),
        pass_rate=pass_rate_df,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_output_package(output_dir: Path) -> OutputTables:
    output_dir = Path(output_dir)
    return OutputTables(
        pass_rate=_read_csv(output_dir / "pass_rate" / "pass_rate_results.csv"),
        per_arc=_read_csv(output_dir / "pass_rate" / "per_arc_pass_fail.csv"),
        normalized=_read_csv(output_dir / "normalized" / "normalized_rows.csv"),
        normalization_warnings=_read_csv(
            output_dir / "normalized" / "normalization_warnings.csv"),
        sensitivity=_read_csv(output_dir / "sensitivity" / "sensitivity.csv"),
        sensitivity_warnings=_read_csv(
            output_dir / "sensitivity" / "sensitivity_warnings.csv"),
        all_margins=_read_csv(output_dir / "all_errors" / "per_object_margin.csv"),
        optimistic_margins=_read_csv(
            output_dir / "optimistic_only" / "per_object_margin.csv"),
        source_rows=_read_csv(output_dir / "trace" / "source_rows.csv"),
        sensitivity_source_refs=_read_csv(
            output_dir / "trace" / "sensitivity_source_refs.csv"),
        margin_trace=_read_csv(output_dir / "trace" / "margin_trace.csv"),
    )


def unique_values(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    return sorted(str(value) for value in df[column].dropna().unique())


def filter_margins(
    margins: pd.DataFrame,
    source: str = "All",
    analysis_type: str = "All",
    metric: str = "All",
    corner: str = "All",
    status: str = "All",
) -> pd.DataFrame:
    filtered = margins.copy()
    filters = {
        "compare_source": source,
        "analysis_type": analysis_type,
        "metric": metric,
        "corner": corner,
        "margin_status": status,
    }
    for column, value in filters.items():
        if value and value != "All" and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str) == str(value)]
    return filtered


def summarize_margins(margins: pd.DataFrame, margin_trace: pd.DataFrame) -> MarginSummary:
    if margins.empty:
        return MarginSummary(trace_rows=len(margin_trace))
    status = margins["margin_status"].astype(str) if "margin_status" in margins.columns else ""
    ok = int((status == "ok").sum()) if "margin_status" in margins.columns else 0
    return MarginSummary(
        total_margins=len(margins),
        ok_margins=ok,
        needs_review=len(margins) - ok,
        trace_rows=len(margin_trace),
    )


def format_margin_trace_detail(row: pd.Series) -> dict[str, str]:
    source_file = str(row.get("source_file", "") or "")
    line = row.get("source_line_number", "")
    path_line = f"{source_file}:{line}" if source_file and line != "" else ""
    return {
        "title": " / ".join(
            str(row.get(column, ""))
            for column in ["corner", "analysis_type", "metric"]
            if str(row.get(column, ""))
        ),
        "arc": str(row.get("arc", "") or ""),
        "path_line": path_line,
        "source_file": source_file,
        "source_line_number": str(line),
        "required_formula": str(row.get("required_margin_formula", "") or ""),
        "signed_formula": str(row.get("signed_margin_formula", "") or ""),
        "sensitivity_trace_id": str(row.get("sensitivity_trace_id", "") or ""),
        "low_sources": str(row.get("low_source_refs_summary", "") or ""),
        "high_sources": str(row.get("high_source_refs_summary", "") or ""),
    }


def read_source_line(path: Path | str, line_number: int | str) -> str:
    path = Path(path)
    line_number = int(float(line_number))
    if line_number < 1:
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for current, line in enumerate(handle, start=1):
            if current == line_number:
                return line.rstrip("\n")
    return ""


def table_columns(df: pd.DataFrame, preferred: Iterable[str]) -> list[str]:
    if df.empty:
        return list(preferred)
    preferred_existing = [column for column in preferred if column in df.columns]
    remaining = [column for column in df.columns if column not in preferred_existing]
    return preferred_existing + remaining
