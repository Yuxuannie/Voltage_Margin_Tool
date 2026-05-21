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


def build_sensitivity_rows(normalized_rows, policy=None):
    """Build per-target-corner sensitivities from a multi-voltage linear fit."""
    if policy is None:
        policy = load_policy()

    rows = []
    warnings = []
    if normalized_rows.empty:
        return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS), pd.DataFrame(
            warnings, columns=SENSITIVITY_WARNING_COLUMNS)

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

        slope, intercept, r_squared = _linear_fit(available, lib_values)
        target_rows = (
            group_data[["corner", "voltage_v"]]
            .drop_duplicates()
            .sort_values(["voltage_v", "corner"])
        )
        for _, target in target_rows.iterrows():
            rows.append(
                {
                    **metadata,
                    "corner": target["corner"],
                    "voltage_v": float(target["voltage_v"]),
                    "voltage_values_v": ";".join(f"{v:g}" for v in available),
                    "lib_values_ps": ";".join(f"{v:g}" for v in lib_values),
                    "valid_points": len(available),
                    "slope_ps_per_v": slope,
                    "sensitivity_ps_per_mv": abs(slope) / 1000.0,
                    "intercept_ps": intercept,
                    "fit_r2": r_squared,
                    "fit_status": "ok",
                    "warning": "",
                }
            )

    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS), pd.DataFrame(
        warnings, columns=SENSITIVITY_WARNING_COLUMNS)


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
        return MarginOutputs(empty, empty, empty, empty, empty, empty, empty, empty)

    normalized_rows = _ensure_compare_source(normalized_rows)
    sensitivity_rows = _ensure_compare_source(sensitivity_rows)
    merged = normalized_rows.merge(
        sensitivity_rows,
        on=join_cols,
        how="left",
        suffixes=("", "_sensitivity"),
    )

    rows = []
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

        rows.append(
            {
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
                "mc_value_ps": row.get("mc_value_ps"),
                "lib_value_ps": row.get("lib_value_ps"),
                "dif_ps": dif_ps,
                "signed_margin_mv": signed_margin,
                "required_margin_mv": required_margin,
                "is_optimistic_risk": bool(row.get("is_optimistic_risk")),
                "voltage_values_v": row.get("voltage_values_v"),
                "lib_values_ps": row.get("lib_values_ps"),
                "sensitivity_ps_per_mv": sensitivity,
                "slope_ps_per_v": row.get("slope_ps_per_v"),
                "intercept_ps": row.get("intercept_ps"),
                "fit_r2": row.get("fit_r2"),
                "valid_voltage_points": row.get("valid_points"),
                "margin_status": status,
                "warning": "" if status == "ok" else status,
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
