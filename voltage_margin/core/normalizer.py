"""Normalize Phase 1 rpt files into one row per canonical arc and metric."""

import logging
import re
from pathlib import Path

import pandas as pd

from .arc_utils import (
    corner_family,
    extract_temperature_from_corner,
    extract_voltage_from_corner,
    normalize_arc,
    parse_cell_name,
)
from .config_loader import load_column_mapping

logger = logging.getLogger(__name__)


def parse_phase1_filename(path, config):
    """Parse a Phase 1 rpt filename into file-level metadata."""
    name = Path(path).name
    match = re.match(config["filename"]["pattern"], name)
    if not match:
        return None
    data = match.groupdict()
    data["compare_source"] = config["filename"]["prefix_to_compare_source"][data["prefix"]]
    data["file_name"] = name
    data["file_path"] = str(path)
    data["voltage_v"] = extract_voltage_from_corner(data["corner"])
    data["temperature_c"] = extract_temperature_from_corner(data["corner"])
    data["corner_family"] = corner_family(data["corner"])
    return data


def read_phase1_rpt(path, config):
    """Read CSV rpt with header on line 1 and explicit blank line 2 skip."""
    loading = config["file_loading"]
    return pd.read_csv(
        path,
        header=loading.get("header", 0),
        skiprows=loading.get("skiprows", [1]),
        skip_blank_lines=True,
    )


def _as_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _column_value(row, column):
    if not column:
        return None
    return row.get(column)


def _metric_columns_present(df, metric_map):
    return bool(metric_map.get("mc_value") in df.columns and metric_map.get("lib_value") in df.columns)


def _normalize_file(path, config):
    metadata = parse_phase1_filename(path, config)
    if metadata is None:
        return [], None, []

    df = read_phase1_rpt(path, config)
    source_cfg = config["compare_sources"][metadata["compare_source"]]
    warnings = []
    rows = []

    missing_required = [col for col in source_cfg["required_columns"] if col not in df.columns]
    if missing_required:
        warnings.append(
            {
                "file_path": str(path),
                "row_index": None,
                "arc_original": None,
                "arc": None,
                "analysis_type": metadata["analysis_type"],
                "metric": None,
                "warning_code": "missing_required_column",
                "warning_message": f"Missing required columns: {missing_required}",
            }
        )

    allowed_metrics = config["analysis_metrics"][metadata["analysis_type"]]
    table_type_map = config["table_type_to_analysis_type"]

    for row_index, row in df.iterrows():
        table_type = row.get(config["canonical_columns"]["table_type"])
        table_analysis = table_type_map.get(table_type)
        if table_analysis and table_analysis != metadata["analysis_type"]:
            warnings.append(
                _warning(path, row_index, row.get("Arc"), None, metadata["analysis_type"], None,
                         "table_type_analysis_mismatch",
                         f"{table_type} maps to {table_analysis}, filename says {metadata['analysis_type']}")
            )
            continue

        arc_original = row.get(config["canonical_columns"]["arc"])
        arc = normalize_arc(arc_original, metadata["compare_source"], config)
        cell_name = parse_cell_name(
            arc,
            metadata["analysis_type"],
            metadata["compare_source"],
            config,
            row=row,
        )
        rel_pin_slew_ps = _as_float(row.get(config["canonical_columns"]["rel_pin_slew"]))

        for metric in allowed_metrics:
            metric_map = source_cfg["metrics"].get(metric)
            if not metric_map or not _metric_columns_present(df, metric_map):
                continue

            mc_value = _as_float(_column_value(row, metric_map["mc_value"]))
            lib_value = _as_float(_column_value(row, metric_map["lib_value"]))
            if mc_value is None or lib_value is None:
                warnings.append(_warning(path, row_index, arc_original, arc, metadata["analysis_type"],
                                         metric, "invalid_numeric", "Missing or invalid MC/Lib value"))
                continue

            computed_dif = lib_value - mc_value
            mapped_dif = _as_float(_column_value(row, metric_map.get("dif")))
            dif_ps = mapped_dif if mapped_dif is not None else computed_dif

            dif_cfg = config["dif_resolution"]
            abs_diff = abs(mapped_dif - computed_dif) if mapped_dif is not None else 0.0
            denom = max(abs(mc_value), abs(lib_value), 1e-12)
            rel_diff = abs_diff / denom
            tolerance = dif_cfg.get("consistency_tolerance", {})
            if isinstance(tolerance, dict):
                abs_tol = tolerance.get("absolute_ps", dif_cfg.get("consistency_tolerance_ps", 0.01))
                rel_tol = tolerance.get("relative", 1.0e-6)
            else:
                abs_tol = dif_cfg.get("consistency_tolerance_ps", float(tolerance or 0.01))
                rel_tol = 1.0e-6
            if (
                mapped_dif is not None
                and dif_cfg.get("validate_consistency", False)
                and abs_diff > abs_tol
                and rel_diff > rel_tol
            ):
                warnings.append(_warning(path, row_index, arc_original, arc, metadata["analysis_type"],
                                         metric, "dif_mismatch",
                                         f"mapped dif {mapped_dif} != computed dif {computed_dif}"))

            rel_err = _as_float(_column_value(row, metric_map.get("rel")))
            if rel_err is None:
                rel_err = computed_dif / abs(mc_value) if mc_value != 0 else None

            ci_lb = _as_float(_column_value(row, metric_map.get("ci_lb")))
            ci_ub = _as_float(_column_value(row, metric_map.get("ci_ub")))
            has_real_ci = ci_lb is not None and ci_ub is not None

            rows.append(
                {
                    "compare_source": metadata["compare_source"],
                    "process": metadata["process"],
                    "process_version": metadata["process_version"],
                    "corner": metadata["corner"],
                    "corner_family": metadata["corner_family"],
                    "voltage_v": metadata["voltage_v"],
                    "temperature_c": metadata["temperature_c"],
                    "analysis_type": metadata["analysis_type"],
                    "table_type": table_type,
                    "metric": metric,
                    "arc_original": arc_original,
                    "arc": arc,
                    "cell_name": cell_name,
                    "rel_pin_slew_ps": rel_pin_slew_ps,
                    "mc_value_ps": mc_value,
                    "lib_value_ps": lib_value,
                    "dif_ps": dif_ps,
                    "rel_err": rel_err,
                    "mc_ci_lb_ps": ci_lb,
                    "mc_ci_ub_ps": ci_ub,
                    "has_real_ci": bool(has_real_ci),
                    "ci_source": "real" if has_real_ci else "none",
                    "is_optimistic_risk": bool(dif_ps < 0),
                    "row_status": "ok",
                    "warnings": "",
                }
            )

    manifest = {
        "file_path": str(path),
        "file_name": Path(path).name,
        "compare_source": metadata["compare_source"],
        "process": metadata["process"],
        "process_version": metadata["process_version"],
        "corner": metadata["corner"],
        "corner_family": metadata["corner_family"],
        "voltage_v": metadata["voltage_v"],
        "temperature_c": metadata["temperature_c"],
        "analysis_type": metadata["analysis_type"],
        "compare_kind": metadata["compare_kind"],
        "row_count_raw": len(df),
        "row_count_normalized": len(rows),
        "warnings": ";".join(w["warning_code"] for w in warnings),
    }
    return rows, manifest, warnings


def _warning(path, row_index, arc_original, arc, analysis_type, metric, code, message):
    return {
        "file_path": str(path),
        "row_index": row_index,
        "arc_original": arc_original,
        "arc": arc,
        "analysis_type": analysis_type,
        "metric": metric,
        "warning_code": code,
        "warning_message": message,
    }


def load_normalized_data(data_dir, corners=None, types=None, column_mapping=None):
    """Load all Phase 1 rpt files and return normalized, manifest, warnings DataFrames."""
    config = load_column_mapping(column_mapping)
    all_rows = []
    manifest_rows = []
    warning_rows = []

    for path in sorted(Path(data_dir).glob("*.rpt")):
        metadata = parse_phase1_filename(path, config)
        if metadata is None:
            continue
        if corners is not None and metadata["corner"] not in corners:
            continue
        if types is not None and metadata["analysis_type"] not in types:
            continue
        rows, manifest, warnings = _normalize_file(path, config)
        all_rows.extend(rows)
        if manifest:
            manifest_rows.append(manifest)
        warning_rows.extend(warnings)

    normalized = pd.DataFrame(all_rows)
    if not normalized.empty:
        normalized["is_optimistic_risk"] = normalized["is_optimistic_risk"].astype(object)
        normalized["has_real_ci"] = normalized["has_real_ci"].astype(object)
    return normalized, pd.DataFrame(manifest_rows), pd.DataFrame(warning_rows)
