"""Testable Phase 1 backend helpers for the Tkinter workbench."""

from dataclasses import dataclass
import math
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


def enrich_margin_rows(
    margins: pd.DataFrame,
    sensitivity: pd.DataFrame,
    per_arc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if margins.empty or sensitivity.empty or "sensitivity_trace_id" not in margins.columns:
        enriched = margins.copy()
    else:
        enrich_columns = [
            "sensitivity_trace_id",
            "low_source_refs_summary",
            "high_source_refs_summary",
            "sensitivity_formula",
        ]
        available = [column for column in enrich_columns if column in sensitivity.columns]
        if available == ["sensitivity_trace_id"]:
            enriched = margins.copy()
        else:
            lookup = sensitivity[available].drop_duplicates("sensitivity_trace_id")
            enriched = margins.merge(lookup, on="sensitivity_trace_id", how="left")
    if per_arc is None or per_arc.empty:
        return enriched
    join_cols = ["corner", "analysis_type", "metric", "arc"]
    if not set(join_cols).issubset(enriched.columns) or not set(join_cols).issubset(per_arc.columns):
        return enriched
    per_arc_columns = join_cols + [
        "rel_threshold",
        "abs_threshold_ps",
        "base_pass",
        "ci_bounds_pass",
        "mc_ci_lb_ps",
        "mc_ci_ub_ps",
    ]
    available = [column for column in per_arc_columns if column in per_arc.columns]
    lookup = per_arc[available].drop_duplicates(join_cols)
    return enriched.merge(lookup, on=join_cols, how="left", suffixes=("", "_pass_rate"))


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if pd.isna(value):
        return False
    return bool(value)


def _finite(value):
    numeric = pd.to_numeric(value, errors="coerce")
    try:
        return float(numeric) if math.isfinite(float(numeric)) else None
    except (TypeError, ValueError):
        return None


def _base_pass(row) -> bool:
    if "base_pass" in row and not pd.isna(row.get("base_pass")):
        return _truthy(row.get("base_pass"))
    return _passes_with_dif(row, _finite(row.get("dif_ps")))


def _passes_with_dif(row, adjusted_dif) -> bool:
    if adjusted_dif is None:
        return False
    mc_value = _finite(row.get("mc_value_ps"))
    rel_threshold = _finite(row.get("rel_threshold"))
    abs_threshold = _finite(row.get("abs_threshold_ps"))
    rel_pass = False
    abs_pass = False
    if mc_value not in (None, 0.0) and rel_threshold is not None:
        rel_pass = abs(adjusted_dif / abs(mc_value)) <= rel_threshold
    if abs_threshold is not None:
        abs_pass = abs(adjusted_dif) <= abs_threshold
    ci_pass = _ci_pass(row, adjusted_dif)
    return rel_pass or abs_pass or ci_pass


def _ci_pass(row, adjusted_dif) -> bool:
    mc_value = _finite(row.get("mc_value_ps"))
    ci_lb = _finite(row.get("mc_ci_lb_ps"))
    ci_ub = _finite(row.get("mc_ci_ub_ps"))
    if mc_value is None or ci_lb is None or ci_ub is None:
        return False
    adjusted_lib = mc_value + adjusted_dif
    lower = min(ci_lb, ci_ub)
    upper = max(ci_lb, ci_ub)
    return lower <= adjusted_lib <= upper


def build_vm_sweep(
    margins: pd.DataFrame,
    max_margin_mv: int = 10,
    step_mv: int = 1,
    group_columns: Iterable[str] = ("compare_source", "corner", "analysis_type", "metric"),
) -> pd.DataFrame:
    group_columns = list(group_columns)
    output_columns = [
        "scope",
        *group_columns,
        "margin_mv",
        "pass_count",
        "total_count",
        "base_pass_count",
        "fixed_count",
        "new_fail_count",
        "pass_rate_pct",
    ]
    required = set(group_columns + ["dif_ps", "sensitivity_ps_per_mv"])
    if margins.empty or not required.issubset(margins.columns):
        return pd.DataFrame(columns=output_columns)

    margin_points = list(range(0, int(max_margin_mv) + 1, int(step_mv)))
    rows = []
    for group_key, group in margins.groupby(group_columns, dropna=False):
        group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
        base_passes = group.apply(_base_pass, axis=1)
        base_pass_count = int(base_passes.sum())
        total = len(group)
        for margin_mv in margin_points:
            simulated = []
            for idx, row in group.iterrows():
                dif = _finite(row.get("dif_ps"))
                sensitivity = _finite(row.get("sensitivity_ps_per_mv"))
                adjusted_dif = dif
                if dif is not None and sensitivity is not None:
                    adjusted_dif = dif + margin_mv * sensitivity
                simulated.append(_passes_with_dif(row, adjusted_dif))
            simulated = pd.Series(simulated, index=group.index)
            for scope in ["all_rows", "outliers_only"]:
                if scope == "outliers_only":
                    scoped_passes = base_passes | simulated
                else:
                    scoped_passes = simulated
                pass_count = int(scoped_passes.sum())
                fixed_count = int((~base_passes & scoped_passes).sum())
                new_fail_count = int((base_passes & ~scoped_passes).sum())
                rows.append(
                    {
                        "scope": scope,
                        **group_values,
                        "margin_mv": margin_mv,
                        "pass_count": pass_count,
                        "total_count": total,
                        "base_pass_count": base_pass_count,
                        "fixed_count": fixed_count,
                        "new_fail_count": new_fail_count,
                        "pass_rate_pct": round(pass_count / total * 100.0, 6) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows, columns=output_columns)


def build_vm_target_summary(
    sweep: pd.DataFrame,
    target_pass_rate_pct: float = 95.0,
) -> pd.DataFrame:
    group_columns = ["scope", "compare_source", "corner", "analysis_type", "metric"]
    output_columns = group_columns + [
        "target_pass_rate_pct",
        "base_pr_pct",
        "required_vm_mv",
        "pass_rate_at_required_vm_pct",
        "fixed_count_at_required_vm",
        "new_fail_count_at_required_vm",
        "total_count",
        "status",
    ]
    if sweep.empty or not set(group_columns).issubset(sweep.columns):
        return pd.DataFrame(columns=output_columns)

    rows = []
    for group_key, group in sweep.groupby(group_columns, dropna=False):
        ordered = group.sort_values("margin_mv", kind="mergesort")
        base_rows = ordered[ordered["margin_mv"] == ordered["margin_mv"].min()]
        base_pr = float(base_rows.iloc[0]["pass_rate_pct"]) if not base_rows.empty else 0.0
        reaching = ordered[ordered["pass_rate_pct"] >= target_pass_rate_pct]
        if reaching.empty:
            selected = ordered.iloc[-1]
            status = "not_reached"
            required_vm = None
        else:
            selected = reaching.iloc[0]
            status = "ok"
            required_vm = selected["margin_mv"]
        rows.append(
            {
                **dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,))),
                "target_pass_rate_pct": target_pass_rate_pct,
                "base_pr_pct": base_pr,
                "required_vm_mv": required_vm,
                "pass_rate_at_required_vm_pct": float(selected["pass_rate_pct"]),
                "fixed_count_at_required_vm": int(selected["fixed_count"]),
                "new_fail_count_at_required_vm": int(selected["new_fail_count"]),
                "total_count": int(selected["total_count"]),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=output_columns).sort_values(
        ["status", "required_vm_mv", "base_pr_pct"],
        ascending=[True, True, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def build_target_margin_plan(
    margins: pd.DataFrame,
    target_coverage: float = 0.95,
    group_columns: Iterable[str] = ("compare_source", "corner", "analysis_type"),
) -> pd.DataFrame:
    group_columns = list(group_columns)
    output_columns = group_columns + [
        "target_coverage_pct",
        "required_margin_mv",
        "covered_rows",
        "valid_rows",
        "total_rows",
        "skipped_rows",
        "coverage_pct",
        "worst_margin_mv",
        "worst_metric",
        "worst_arc",
        "worst_source_file_relative",
        "worst_source_line_number",
        "worst_margin_trace_id",
    ]
    required = set(group_columns + ["required_margin_mv", "margin_status"])
    if margins.empty or not required.issubset(margins.columns):
        return pd.DataFrame(columns=output_columns)

    rows = []
    for group_key, group in margins.groupby(group_columns, dropna=False):
        valid = group[
            (group["margin_status"].astype(str) == "ok")
            & pd.to_numeric(group["required_margin_mv"], errors="coerce").apply(math.isfinite)
        ].copy()
        valid["required_margin_mv"] = valid["required_margin_mv"].astype(float)
        if valid.empty:
            continue
        ordered = valid.sort_values("required_margin_mv", kind="mergesort")
        target_index = max(math.ceil(target_coverage * len(ordered)) - 1, 0)
        target_row = ordered.iloc[target_index]
        worst_row = ordered.iloc[-1]
        covered = int((ordered["required_margin_mv"] <= target_row["required_margin_mv"]).sum())
        rows.append(
            {
                **dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,))),
                "target_coverage_pct": target_coverage * 100.0,
                "required_margin_mv": float(target_row["required_margin_mv"]),
                "covered_rows": covered,
                "valid_rows": len(valid),
                "total_rows": len(group),
                "skipped_rows": len(group) - len(valid),
                "coverage_pct": covered / len(valid) * 100.0,
                "worst_margin_mv": float(worst_row["required_margin_mv"]),
                "worst_metric": str(worst_row.get("metric", "") or ""),
                "worst_arc": str(worst_row.get("arc", "") or ""),
                "worst_source_file_relative": str(
                    worst_row.get("source_file_relative", "") or ""),
                "worst_source_line_number": worst_row.get("source_line_number", ""),
                "worst_margin_trace_id": str(worst_row.get("margin_trace_id", "") or ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows, columns=output_columns).sort_values(
        ["required_margin_mv"] + group_columns,
        ascending=[False] + [True] * len(group_columns),
        kind="mergesort",
    ).reset_index(drop=True)


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


def read_source_context(path: Path | str, line_number: int | str, radius: int = 2) -> str:
    path = Path(path)
    line_number = int(float(line_number))
    start = max(line_number - radius, 1)
    end = line_number + radius
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for current, line in enumerate(handle, start=1):
            if current < start:
                continue
            if current > end:
                break
            marker = ">>" if current == line_number else "  "
            lines.append(f"{marker} {current} | {line.rstrip()}")
    return "\n".join(lines)


def table_columns(df: pd.DataFrame, preferred: Iterable[str]) -> list[str]:
    if df.empty:
        return list(preferred)
    preferred_existing = [column for column in preferred if column in df.columns]
    remaining = [column for column in df.columns if column not in preferred_existing]
    return preferred_existing + remaining
