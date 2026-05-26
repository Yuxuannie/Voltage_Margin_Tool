"""Phase 1 sensitivity and voltage-margin outputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config_loader import load_policy


@dataclass
class MarginOutputs:
    all_errors_per_object: pd.DataFrame
    optimistic_only_per_object: pd.DataFrame
    all_errors_summary: pd.DataFrame
    optimistic_only_summary: pd.DataFrame
    all_errors_curve: pd.DataFrame
    optimistic_only_curve: pd.DataFrame
    all_errors_high_margin: pd.DataFrame
    optimistic_only_high_margin: pd.DataFrame
    margin_trace: pd.DataFrame


SENSITIVITY_COLUMNS = [
    "process",
    "process_version",
    "compare_source",
    "analysis_type",
    "metric",
    "arc",
    "corner_family",
    "temperature_c",
    "corner",
    "voltage_v",
    "sensitivity_trace_id",
    "pair_v_low",
    "pair_v_high",
    "pair_role",
    "low_voltage_v",
    "high_voltage_v",
    "low_lib_value_ps",
    "high_lib_value_ps",
    "low_source_refs_summary",
    "high_source_refs_summary",
    "sensitivity_formula_id",
    "sensitivity_formula",
    "voltage_values_v",
    "lib_values_ps",
    "valid_points",
    "slope_ps_per_v",
    "sensitivity_ps_per_mv",
    "intercept_ps",
    "fit_r2",
    "fit_status",
    "warning",
]


SENSITIVITY_WARNING_COLUMNS = [
    "process",
    "process_version",
    "compare_source",
    "analysis_type",
    "metric",
    "arc",
    "corner_family",
    "temperature_c",
    "corner",
    "voltage_v",
    "valid_points",
    "warning_code",
    "warning_message",
]


SENSITIVITY_SOURCE_REF_COLUMNS = [
    "sensitivity_trace_id",
    "pair_side",
    "source_trace_id",
    "input_root",
    "source_file",
    "source_file_relative",
    "source_line_number",
    "source_row_index",
    "voltage_v",
    "lib_value_ps",
]


MARGIN_TRACE_COLUMNS = [
    "margin_trace_id",
    "normalized_trace_id",
    "sensitivity_trace_id",
    "input_root",
    "source_file",
    "source_file_relative",
    "source_line_number",
    "source_row_index",
    "dif_ps",
    "sensitivity_ps_per_mv",
    "required_margin_mv",
    "signed_margin_mv",
    "margin_formula_id",
    "required_margin_formula",
    "signed_margin_formula",
]


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return value


def _ensure_compare_source(df):
    df = df.copy()
    if "compare_source" not in df.columns:
        df["compare_source"] = "unknown"
    return df


def _linear_fit(voltages, values):
    v = np.asarray(voltages, dtype=float)
    y = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(v, y, 1)
    predicted = slope * v + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    return slope, intercept, r_squared


def _source_ref_summary(source_rows):
    refs = []
    for _, source in source_rows.iterrows():
        file_name = source.get("source_file_relative") or source.get("source_file")
        line = source.get("source_line_number")
        refs.append(f"{file_name}:{line}")
    return ";".join(refs)


def _source_ref_rows(sensitivity_trace_id, pair_side, source_rows):
    rows = []
    for _, source in source_rows.iterrows():
        rows.append(
            {
                "sensitivity_trace_id": sensitivity_trace_id,
                "pair_side": pair_side,
                "source_trace_id": source.get("trace_id"),
                "input_root": source.get("input_root"),
                "source_file": source.get("source_file"),
                "source_file_relative": source.get("source_file_relative"),
                "source_line_number": source.get("source_line_number"),
                "source_row_index": source.get("source_row_index"),
                "voltage_v": source.get("voltage_v"),
                "lib_value_ps": source.get("lib_value_ps"),
            }
        )
    return rows


def _sensitivity_formula(low_value, high_value, low_voltage, high_voltage):
    return (
        "sensitivity_ps_per_mv = "
        f"abs({high_value:g} - {low_value:g}) / "
        f"abs(({high_voltage:g} - {low_voltage:g}) * 1000)"
    )


def _required_margin_formula(dif_ps, sensitivity):
    return f"required_margin_mv = abs({dif_ps:g}) / {sensitivity:g}"


def _signed_margin_formula(dif_ps, sensitivity):
    return f"signed_margin_mv = {dif_ps:g} / {sensitivity:g}"


def build_sensitivity_rows(normalized_rows, policy=None):
    """Build per-target-corner sensitivities from adjacent voltage pairs."""
    if policy is None:
        policy = load_policy()

    rows = []
    source_ref_rows = []
    warnings = []
    if normalized_rows.empty:
        sensitivity_df = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
        sensitivity_df.attrs["source_refs"] = pd.DataFrame(
            source_ref_rows, columns=SENSITIVITY_SOURCE_REF_COLUMNS)
        return sensitivity_df, pd.DataFrame(warnings, columns=SENSITIVITY_WARNING_COLUMNS)

    normalized_rows = _ensure_compare_source(normalized_rows)
    group_cols = [
        "process",
        "process_version",
        "compare_source",
        "analysis_type",
        "metric",
        "arc",
        "corner_family",
        "temperature_c",
    ]
    for group_key, group in normalized_rows.groupby(group_cols, dropna=False):
        group_data = group.dropna(subset=["voltage_v", "lib_value_ps"]).copy()
        metadata = dict(zip(group_cols, group_key))
        voltage_values = (
            group_data[["voltage_v", "lib_value_ps"]]
            .astype({"voltage_v": float, "lib_value_ps": float})
            .groupby("voltage_v", as_index=False)["lib_value_ps"]
            .mean()
            .sort_values("voltage_v")
        )
        available = voltage_values["voltage_v"].tolist()
        lib_values = voltage_values["lib_value_ps"].tolist()
        voltage_sources = {
            float(voltage): source_group.copy()
            for voltage, source_group in group_data.groupby("voltage_v")
        }

        if len(available) < policy["sensitivity"].get("min_voltage_points", 2):
            warnings.append(
                {
                    **metadata,
                    "valid_points": len(available),
                    "warning_code": "insufficient_voltage_points",
                    "warning_message": "At least 2 voltage points are required for sensitivity fitting",
                }
            )
            continue

        target_rows = (
            group_data[["corner", "voltage_v"]]
            .drop_duplicates()
            .sort_values(["voltage_v", "corner"])
        )
        for _, target in target_rows.iterrows():
            target_voltage = float(target["voltage_v"])
            target_index = available.index(target_voltage)
            for low_index, high_index, role in _adjacent_pairs_for_index(
                target_index, len(available)
            ):
                pair_voltages = [available[low_index], available[high_index]]
                pair_values = [lib_values[low_index], lib_values[high_index]]
                slope, intercept, r_squared = _linear_fit(pair_voltages, pair_values)
                sensitivity_trace_id = f"sens-{len(rows) + 1:08d}"
                low_sources = voltage_sources[pair_voltages[0]]
                high_sources = voltage_sources[pair_voltages[1]]
                source_ref_rows.extend(
                    _source_ref_rows(sensitivity_trace_id, "low", low_sources))
                source_ref_rows.extend(
                    _source_ref_rows(sensitivity_trace_id, "high", high_sources))
                rows.append(
                    {
                        **metadata,
                        "corner": target["corner"],
                        "voltage_v": target_voltage,
                        "sensitivity_trace_id": sensitivity_trace_id,
                        "pair_v_low": pair_voltages[0],
                        "pair_v_high": pair_voltages[1],
                        "pair_role": role,
                        "low_voltage_v": pair_voltages[0],
                        "high_voltage_v": pair_voltages[1],
                        "low_lib_value_ps": pair_values[0],
                        "high_lib_value_ps": pair_values[1],
                        "low_source_refs_summary": _source_ref_summary(low_sources),
                        "high_source_refs_summary": _source_ref_summary(high_sources),
                        "sensitivity_formula_id": "adjacent_pair_lib_slope_ps_per_mv",
                        "sensitivity_formula": _sensitivity_formula(
                            pair_values[0], pair_values[1], pair_voltages[0], pair_voltages[1]),
                        "voltage_values_v": ";".join(f"{v:g}" for v in pair_voltages),
                        "lib_values_ps": ";".join(f"{v:g}" for v in pair_values),
                        "valid_points": len(pair_voltages),
                        "slope_ps_per_v": slope,
                        "sensitivity_ps_per_mv": abs(slope) / 1000.0,
                        "intercept_ps": intercept,
                        "fit_r2": r_squared,
                        "fit_status": "ok",
                        "warning": "",
                    }
                )

    sensitivity_df = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
    sensitivity_df.attrs["source_refs"] = pd.DataFrame(
        source_ref_rows, columns=SENSITIVITY_SOURCE_REF_COLUMNS)
    return sensitivity_df, pd.DataFrame(warnings, columns=SENSITIVITY_WARNING_COLUMNS)


def _adjacent_pairs_for_index(index, length):
    if length == 2:
        return [(0, 1, "only_pair")]
    if index == 0:
        return [(0, 1, "only_pair")]
    if index == length - 1:
        return [(length - 2, length - 1, "only_pair")]
    return [
        (index - 1, index, "lower_pair"),
        (index, index + 1, "upper_pair"),
    ]


def build_margin_outputs(normalized_rows, sensitivity_rows, policy=None):
    """Join normalized errors to sensitivity rows and compute margin CSV tables."""
    if policy is None:
        policy = load_policy()

    join_cols = [
        "process",
        "process_version",
        "compare_source",
        "corner",
        "voltage_v",
        "analysis_type",
        "metric",
        "arc",
        "corner_family",
        "temperature_c",
    ]
    if normalized_rows.empty:
        empty = pd.DataFrame()
        return MarginOutputs(empty, empty, empty, empty, empty, empty, empty, empty, empty)

    normalized_rows = _ensure_compare_source(normalized_rows)
    sensitivity_rows = _ensure_compare_source(sensitivity_rows)
    merged = normalized_rows.merge(
        sensitivity_rows,
        on=join_cols,
        how="left",
        suffixes=("", "_sensitivity"),
    )

    rows = []
    margin_trace_rows = []
    for _, row in merged.iterrows():
        dif_ps = _finite_float(row.get("dif_ps"))
        sensitivity = _finite_float(row.get("sensitivity_ps_per_mv"))
        if dif_ps is None:
            status = "skipped_missing_value"
            signed_margin = np.nan
            required_margin = np.nan
        elif sensitivity is None:
            status = "skipped_no_sensitivity"
            signed_margin = np.nan
            required_margin = np.nan
        elif sensitivity == 0:
            status = "skipped_zero_sensitivity"
            signed_margin = np.nan
            required_margin = np.inf
        else:
            status = "ok"
            signed_margin = dif_ps / sensitivity
            required_margin = abs(dif_ps) / sensitivity

        margin_trace_id = f"margin-{len(rows) + 1:08d}"
        normalized_trace_id = row.get("trace_id")
        sensitivity_trace_id = row.get("sensitivity_trace_id")
        margin_formula_id = "margin_from_dif_and_sensitivity"
        required_formula = (
            _required_margin_formula(dif_ps, sensitivity)
            if dif_ps is not None and sensitivity is not None
            else ""
        )
        signed_formula = (
            _signed_margin_formula(dif_ps, sensitivity)
            if dif_ps is not None and sensitivity is not None
            else ""
        )
        rows.append(
            {
                "margin_trace_id": margin_trace_id,
                "normalized_trace_id": normalized_trace_id,
                "sensitivity_trace_id": sensitivity_trace_id,
                "process": row.get("process"),
                "process_version": row.get("process_version"),
                "compare_source": row.get("compare_source"),
                "corner": row.get("corner"),
                "corner_family": row.get("corner_family"),
                "temperature_c": row.get("temperature_c"),
                "voltage_v": row.get("voltage_v"),
                "analysis_type": row.get("analysis_type"),
                "table_type": row.get("table_type"),
                "metric": row.get("metric"),
                "arc": row.get("arc"),
                "cell_name": row.get("cell_name"),
                "source_file": row.get("source_file"),
                "source_file_relative": row.get("source_file_relative"),
                "source_line_number": row.get("source_line_number"),
                "source_row_index": row.get("source_row_index"),
                "mc_value_ps": row.get("mc_value_ps"),
                "lib_value_ps": row.get("lib_value_ps"),
                "dif_ps": dif_ps,
                "signed_margin_mv": signed_margin,
                "required_margin_mv": required_margin,
                "is_optimistic_risk": bool(row.get("is_optimistic_risk")),
                "voltage_values_v": row.get("voltage_values_v"),
                "lib_values_ps": row.get("lib_values_ps"),
                "pair_v_low": row.get("pair_v_low"),
                "pair_v_high": row.get("pair_v_high"),
                "pair_role": row.get("pair_role"),
                "sensitivity_ps_per_mv": sensitivity,
                "slope_ps_per_v": row.get("slope_ps_per_v"),
                "intercept_ps": row.get("intercept_ps"),
                "fit_r2": row.get("fit_r2"),
                "valid_voltage_points": row.get("valid_points"),
                "margin_status": status,
                "margin_formula_id": margin_formula_id,
                "required_margin_formula": required_formula,
                "signed_margin_formula": signed_formula,
                "warning": "" if status == "ok" else status,
            }
        )
        margin_trace_rows.append(
            {
                "margin_trace_id": margin_trace_id,
                "normalized_trace_id": normalized_trace_id,
                "sensitivity_trace_id": sensitivity_trace_id,
                "input_root": row.get("input_root"),
                "source_file": row.get("source_file"),
                "source_file_relative": row.get("source_file_relative"),
                "source_line_number": row.get("source_line_number"),
                "source_row_index": row.get("source_row_index"),
                "dif_ps": dif_ps,
                "sensitivity_ps_per_mv": sensitivity,
                "required_margin_mv": required_margin,
                "signed_margin_mv": signed_margin,
                "margin_formula_id": margin_formula_id,
                "required_margin_formula": required_formula,
                "signed_margin_formula": signed_formula,
            }
        )

    all_errors = pd.DataFrame(rows)
    if not all_errors.empty:
        all_errors["is_optimistic_risk"] = all_errors["is_optimistic_risk"].astype(object)
    optimistic = all_errors[all_errors["is_optimistic_risk"] == True].copy()  # noqa: E712

    return MarginOutputs(
        all_errors_per_object=all_errors,
        optimistic_only_per_object=optimistic,
        all_errors_summary=_summary(all_errors),
        optimistic_only_summary=_summary(optimistic),
        all_errors_curve=_coverage_curve(all_errors, policy),
        optimistic_only_curve=_coverage_curve(optimistic, policy),
        all_errors_high_margin=_high_margin(all_errors),
        optimistic_only_high_margin=_high_margin(optimistic),
        margin_trace=pd.DataFrame(margin_trace_rows, columns=MARGIN_TRACE_COLUMNS),
    )


def _valid_margin_rows(df):
    if df.empty or "margin_status" not in df.columns:
        return df.iloc[0:0].copy()
    return df[(df["margin_status"] == "ok") & np.isfinite(df["required_margin_mv"])].copy()


def _summary(df):
    valid = _valid_margin_rows(df)
    group_cols = ["process", "process_version", "compare_source", "corner", "analysis_type", "metric"]
    rows = []
    if valid.empty:
        return pd.DataFrame(rows)
    skipped = df[df["margin_status"] != "ok"].groupby(group_cols).size().to_dict()
    for group_key, group in valid.groupby(group_cols):
        margins = group["required_margin_mv"].astype(float)
        rows.append(
            {
                **dict(zip(group_cols, group_key)),
                "aggregation": "per_object",
                "count": len(group),
                "mean_margin_mv": float(np.mean(margins)),
                "median_margin_mv": float(np.median(margins)),
                "p90_margin_mv": float(np.percentile(margins, 90)),
                "p95_margin_mv": float(np.percentile(margins, 95)),
                "p99_margin_mv": float(np.percentile(margins, 99)),
                "max_margin_mv": float(np.max(margins)),
                "skipped_count": int(skipped.get(group_key, 0)),
            }
        )
    return pd.DataFrame(rows)


def _coverage_curve(df, policy):
    valid = _valid_margin_rows(df)
    group_cols = ["process", "process_version", "compare_source", "corner", "analysis_type", "metric"]
    margin_points = policy.get("margin_outputs", {}).get("margin_points_mv", list(range(0, 55, 5)))
    rows = []
    if valid.empty:
        return pd.DataFrame(rows)
    for group_key, group in valid.groupby(group_cols):
        margins = group["required_margin_mv"].astype(float)
        total = len(margins)
        for margin_point in margin_points:
            covered = int((margins <= margin_point).sum())
            rows.append(
                {
                    **dict(zip(group_cols, group_key)),
                    "margin_mv": margin_point,
                    "covered_count": covered,
                    "total_count": total,
                    "coverage_fraction": covered / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _high_margin(df):
    valid = _valid_margin_rows(df)
    if valid.empty:
        return valid
    group_cols = ["process", "process_version", "compare_source", "corner", "analysis_type", "metric"]
    valid = valid.sort_values(group_cols + ["required_margin_mv"], ascending=[True] * len(group_cols) + [False])
    valid["rank_within_group"] = valid.groupby(group_cols)["required_margin_mv"].rank(
        method="first", ascending=False
    ).astype(int)
    return valid
