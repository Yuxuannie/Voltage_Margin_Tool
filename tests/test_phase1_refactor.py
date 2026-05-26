import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from voltage_margin.core.config_loader import load_column_mapping, load_policy
from voltage_margin.core.normalizer import load_normalized_data
from voltage_margin.core.margin_engine import (
    build_margin_outputs,
    build_sensitivity_rows,
)
from voltage_margin.core.thresholds import get_abs_threshold, get_all_params_for_type


def write_rpt(path: Path, columns, rows):
    lines = [",".join(columns), ""]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in columns))
    path.write_text("\n".join(lines) + "\n")


def test_normalized_loader_handles_hold_blank_line_negative_nominal_and_arc_suffix(tmp_path):
    columns = [
        "Arc",
        "Cell_Name",
        "rel_pin_slew",
        "Table_Type",
        "MC_Late_Sigma",
        "CDNS_Lib_Late_Sigma",
        "CDNS_Lib_Late_Sigma_Dif",
        "CDNS_Lib_Late_Sigma_Rel",
        "MC_Late_Sigma_LB",
        "MC_Late_Sigma_UB",
        "MC_Nominal",
        "CDNS_Lib_Nominal",
        "CDNS_Lib_Nominal_Dif",
        "CDNS_Lib_Nominal_Rel",
    ]
    arc = (
        "hold_SDFRPQTXGOPTBCELVAMZD4BWP130HPNPN3P48CPD_D_rise_"
        "CP_rise_notCD_notSE_SI_4_2"
    )
    write_rpt(
        tmp_path / "fmc_result_n2p_v1p0_ssgnp_0p450v_m40c_hold_fmc_cdns_lib_comp.rpt",
        columns,
        [
            {
                "Arc": arc,
                "Cell_Name": "SDFRPQTXGOPTBCELVAMZD4BWP130HPNPN3P48CPD",
                "rel_pin_slew": 40.0,
                "Table_Type": "rise_constraint",
                "MC_Late_Sigma": 20.0,
                "CDNS_Lib_Late_Sigma": 18.0,
                "CDNS_Lib_Late_Sigma_Dif": -2.0,
                "CDNS_Lib_Late_Sigma_Rel": -0.10,
                "MC_Late_Sigma_LB": 19.0,
                "MC_Late_Sigma_UB": 21.0,
                "MC_Nominal": -1240.43,
                "CDNS_Lib_Nominal": -1250.43,
                "CDNS_Lib_Nominal_Dif": -10.0,
                "CDNS_Lib_Nominal_Rel": -0.008,
            }
        ],
    )

    normalized, manifest, warnings = load_normalized_data(tmp_path)

    assert warnings.empty
    assert manifest.loc[0, "row_count_raw"] == 1
    assert manifest.loc[0, "temperature_c"] == -40
    assert set(normalized["metric"]) == {"Late_Sigma", "Nominal"}
    assert "Early_Sigma" not in set(normalized["metric"])
    assert normalized["arc"].unique().tolist() == [
        arc.replace("_4_2", "_4-2")
    ]
    assert {
        "trace_id",
        "input_root",
        "source_file",
        "source_file_relative",
        "source_line_number",
        "source_row_index",
        "source_mc_column",
        "source_lib_column",
        "source_dif_column",
        "source_rel_column",
    }.issubset(normalized.columns)
    assert normalized["input_root"].unique().tolist() == [str(tmp_path.resolve())]
    assert normalized["source_line_number"].unique().tolist() == [3]
    assert normalized["source_row_index"].unique().tolist() == [0]
    assert normalized["source_file_relative"].str.endswith(".rpt").all()

    nominal = normalized[normalized["metric"] == "Nominal"].iloc[0]
    assert nominal["mc_value_ps"] == -1240.43
    assert nominal["dif_ps"] == -10.0
    assert nominal["is_optimistic_risk"] is True
    assert nominal["source_mc_column"] == "MC_Nominal"
    assert nominal["source_lib_column"] == "CDNS_Lib_Nominal"
    assert nominal["source_dif_column"] == "CDNS_Lib_Nominal_Dif"
    assert nominal["source_rel_column"] == "CDNS_Lib_Nominal_Rel"


def test_normalized_loader_parses_mc_cell_name_and_canonical_metric_case(tmp_path):
    columns = [
        "Arc",
        "rel_pin_slew",
        "Table_Type",
        "MC_late_sigma",
        "Lib_late_sigma",
        "late_sigma_abs_err",
        "late_sigma_rel_err",
        "MC_late_sigma_LB",
        "MC_late_sigma_UB",
    ]
    arc = "combinational_ND2D1BWP130HPNPN3P48CPD_ZN_fall_A1_rise_NO_CONDITION_4-4"
    write_rpt(
        tmp_path / "MC_result_n2p_v1p0_ssgnp_0p450v_m40c_delay_moments_comp.rpt",
        columns,
        [
            {
                "Arc": arc,
                "rel_pin_slew": 20.0,
                "Table_Type": "cell_fall",
                "MC_late_sigma": 194.0751666,
                "Lib_late_sigma": 178.3260040283203,
                "late_sigma_abs_err": -15.74916263834632,
                "late_sigma_rel_err": -0.0811456,
                "MC_late_sigma_LB": 190.0,
                "MC_late_sigma_UB": 198.0,
            }
        ],
    )

    normalized, _, warnings = load_normalized_data(tmp_path)

    assert warnings.empty
    row = normalized.iloc[0]
    assert row["metric"] == "Late_Sigma"
    assert row["cell_name"] == "ND2D1BWP130HPNPN3P48CPD"
    assert row["arc"] == arc
    assert math.isclose(row["dif_ps"], -15.74916263834632)
    assert row["is_optimistic_risk"] is True


def test_dif_consistency_warning_requires_absolute_and_relative_mismatch(tmp_path):
    columns = [
        "Arc",
        "Cell_Name",
        "rel_pin_slew",
        "Table_Type",
        "MC_Late_Sigma",
        "CDNS_Lib_Late_Sigma",
        "CDNS_Lib_Late_Sigma_Dif",
        "CDNS_Lib_Late_Sigma_Rel",
        "MC_Late_Sigma_LB",
        "MC_Late_Sigma_UB",
    ]
    arc_a = "combinational_BIG_Z_rise_A_rise_NO_CONDITION_4_2"
    arc_b = "combinational_SMALL_Z_rise_A_rise_NO_CONDITION_4_3"
    write_rpt(
        tmp_path / "fmc_result_n2p_v1p0_ssgnp_0p450v_m40c_delay_fmc_cdns_lib_comp.rpt",
        columns,
        [
            {
                "Arc": arc_a,
                "Cell_Name": "BIG",
                "rel_pin_slew": 20.0,
                "Table_Type": "cell_rise",
                "MC_Late_Sigma": 1_000_000.0,
                "CDNS_Lib_Late_Sigma": 1_000_010.0,
                "CDNS_Lib_Late_Sigma_Dif": 10.02,
                "CDNS_Lib_Late_Sigma_Rel": 0.0,
                "MC_Late_Sigma_LB": 999_999.0,
                "MC_Late_Sigma_UB": 1_000_001.0,
            },
            {
                "Arc": arc_b,
                "Cell_Name": "SMALL",
                "rel_pin_slew": 20.0,
                "Table_Type": "cell_rise",
                "MC_Late_Sigma": 100.0,
                "CDNS_Lib_Late_Sigma": 110.0,
                "CDNS_Lib_Late_Sigma_Dif": 10.02,
                "CDNS_Lib_Late_Sigma_Rel": 0.0,
                "MC_Late_Sigma_LB": 99.0,
                "MC_Late_Sigma_UB": 101.0,
            },
        ],
    )

    _, _, warnings = load_normalized_data(tmp_path)

    mismatches = warnings[warnings["warning_code"] == "dif_mismatch"]
    assert len(mismatches) == 1
    assert mismatches.iloc[0]["arc"] == arc_b.replace("_4_3", "_4-3")


def test_policy_thresholds_use_ps_floors_and_hold_metric_set():
    policy = load_policy()

    assert get_abs_threshold("hold", "Nominal", 100.0, policy=policy) == 10.0
    assert get_abs_threshold("delay", "Late_Sigma", 100.0, policy=policy) == 1.0
    assert get_all_params_for_type("hold") == ["Late_Sigma", "Nominal"]


def test_sensitivity_uses_adjacent_pair_local_slope_for_target_corner():
    arc = "combinational_CELL_Z_rise_A_rise_NO_CONDITION_4-4"
    normalized = pd.DataFrame(
        [
            {
                "process": "n2p",
                "process_version": "v1p0",
                "corner": "ssgnp_0p450v_m40c",
                "corner_family": "ssgnp_<V>_m40c",
                "temperature_c": -40,
                "compare_source": "fmc_compare",
                "voltage_v": 0.450,
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "arc": arc,
                "cell_name": "CELL",
                "table_type": "cell_rise",
                "mc_value_ps": 100.0,
                "lib_value_ps": 100.0,
                "dif_ps": 0.0,
                "is_optimistic_risk": False,
            },
            {
                "process": "n2p",
                "process_version": "v1p0",
                "corner": "ssgnp_0p465v_m40c",
                "corner_family": "ssgnp_<V>_m40c",
                "temperature_c": -40,
                "compare_source": "fmc_compare",
                "voltage_v": 0.465,
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "arc": arc,
                "cell_name": "CELL",
                "table_type": "cell_rise",
                "mc_value_ps": 130.0,
                "lib_value_ps": 130.0,
                "dif_ps": -6.0,
                "is_optimistic_risk": True,
            },
            {
                "process": "n2p",
                "process_version": "v1p0",
                "corner": "ssgnp_0p480v_m40c",
                "corner_family": "ssgnp_<V>_m40c",
                "temperature_c": -40,
                "compare_source": "fmc_compare",
                "voltage_v": 0.480,
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "arc": arc,
                "cell_name": "CELL",
                "table_type": "cell_rise",
                "mc_value_ps": 190.0,
                "lib_value_ps": 190.0,
                "dif_ps": -8.0,
                "is_optimistic_risk": True,
            },
        ]
    )
    normalized["trace_id"] = [f"norm-{idx + 1:08d}" for idx in range(len(normalized))]
    normalized["input_root"] = "/input"
    normalized["source_file"] = [f"/input/source_{idx}.rpt" for idx in range(len(normalized))]
    normalized["source_file_relative"] = [f"source_{idx}.rpt" for idx in range(len(normalized))]
    normalized["source_line_number"] = [idx + 3 for idx in range(len(normalized))]
    normalized["source_row_index"] = list(range(len(normalized)))

    sensitivity, warnings = build_sensitivity_rows(normalized, load_policy())

    assert warnings.empty
    assert len(sensitivity) == 4
    assert {"pair_v_low", "pair_v_high", "pair_role"}.issubset(sensitivity.columns)
    assert {
        "sensitivity_trace_id",
        "low_voltage_v",
        "high_voltage_v",
        "low_lib_value_ps",
        "high_lib_value_ps",
        "low_source_refs_summary",
        "high_source_refs_summary",
        "sensitivity_formula_id",
        "sensitivity_formula",
    }.issubset(sensitivity.columns)
    assert (sensitivity["valid_points"] == 2).all()
    assert sensitivity["sensitivity_trace_id"].notna().all()
    assert sensitivity["sensitivity_formula_id"].eq("adjacent_pair_lib_slope_ps_per_mv").all()

    first = sensitivity[sensitivity["corner"] == "ssgnp_0p450v_m40c"].iloc[0]
    assert first["pair_role"] == "only_pair"
    assert math.isclose(first["pair_v_low"], 0.450)
    assert math.isclose(first["pair_v_high"], 0.465)
    assert math.isclose(first["sensitivity_ps_per_mv"], 2.0)

    middle = sensitivity[sensitivity["corner"] == "ssgnp_0p465v_m40c"]
    assert middle["pair_role"].tolist() == ["lower_pair", "upper_pair"]
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(middle["sensitivity_ps_per_mv"], [2.0, 4.0])
    )

    margins = build_margin_outputs(normalized, sensitivity, load_policy())
    optimistic = margins.optimistic_only_per_object
    assert {
        "margin_trace_id",
        "normalized_trace_id",
        "sensitivity_trace_id",
        "source_file",
        "source_file_relative",
        "source_line_number",
        "margin_formula_id",
        "required_margin_formula",
        "signed_margin_formula",
    }.issubset(optimistic.columns)
    assert optimistic["margin_formula_id"].eq("margin_from_dif_and_sensitivity").all()
    assert hasattr(margins, "margin_trace")
    assert not margins.margin_trace.empty
    assert set(optimistic["corner"]) == {"ssgnp_0p465v_m40c", "ssgnp_0p480v_m40c"}
    mid_margins = optimistic[optimistic["corner"] == "ssgnp_0p465v_m40c"]
    assert len(mid_margins) == 2
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(mid_margins["required_margin_mv"], [3.0, 1.5])
    )

    mid_summary = margins.optimistic_only_summary[
        margins.optimistic_only_summary["corner"] == "ssgnp_0p465v_m40c"
    ].iloc[0]
    assert mid_summary["aggregation"] == "per_object"
    assert math.isclose(mid_summary["max_margin_mv"], 3.0)


def test_sensitivity_trace_records_all_duplicate_voltage_source_rows():
    arc = "combinational_CELL_Z_rise_A_rise_NO_CONDITION_4-4"
    normalized = pd.DataFrame(
        [
            {
                "trace_id": "norm-low-a",
                "input_root": "/input",
                "source_file": "/input/low_a.rpt",
                "source_file_relative": "low_a.rpt",
                "source_line_number": 3,
                "source_row_index": 0,
                "process": "n2p",
                "process_version": "v1p0",
                "corner": "ssgnp_0p450v_m40c",
                "corner_family": "ssgnp_<V>_m40c",
                "temperature_c": -40,
                "compare_source": "fmc_compare",
                "voltage_v": 0.450,
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "arc": arc,
                "lib_value_ps": 100.0,
            },
            {
                "trace_id": "norm-low-b",
                "input_root": "/input",
                "source_file": "/input/low_b.rpt",
                "source_file_relative": "low_b.rpt",
                "source_line_number": 4,
                "source_row_index": 1,
                "process": "n2p",
                "process_version": "v1p0",
                "corner": "ssgnp_0p450v_m40c",
                "corner_family": "ssgnp_<V>_m40c",
                "temperature_c": -40,
                "compare_source": "fmc_compare",
                "voltage_v": 0.450,
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "arc": arc,
                "lib_value_ps": 120.0,
            },
            {
                "trace_id": "norm-high",
                "input_root": "/input",
                "source_file": "/input/high.rpt",
                "source_file_relative": "high.rpt",
                "source_line_number": 3,
                "source_row_index": 0,
                "process": "n2p",
                "process_version": "v1p0",
                "corner": "ssgnp_0p465v_m40c",
                "corner_family": "ssgnp_<V>_m40c",
                "temperature_c": -40,
                "compare_source": "fmc_compare",
                "voltage_v": 0.465,
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "arc": arc,
                "lib_value_ps": 140.0,
            },
        ]
    )

    sensitivity, warnings = build_sensitivity_rows(normalized, load_policy())

    assert warnings.empty
    assert len(sensitivity) == 2
    assert sensitivity["low_lib_value_ps"].unique().tolist() == [110.0]
    refs = sensitivity.attrs["source_refs"]
    low_refs = refs[refs["pair_side"] == "low"]
    assert set(low_refs["source_trace_id"]) == {"norm-low-a", "norm-low-b"}
    assert set(low_refs["source_file_relative"]) == {"low_a.rpt", "low_b.rpt"}


def test_margin_output_skips_rows_when_sensitivity_has_insufficient_voltage_points():
    normalized = pd.DataFrame(
        [
            {
                "process": "n2p",
                "process_version": "v1p0",
                "corner": "ssgnp_0p450v_m40c",
                "corner_family": "ssgnp_<V>_m40c",
                "temperature_c": -40,
                "compare_source": "fmc_compare",
                "voltage_v": 0.450,
                "analysis_type": "hold",
                "metric": "Nominal",
                "arc": "hold_CELL_D_rise_CP_rise_NO_CONDITION_4-2",
                "cell_name": "CELL",
                "table_type": "rise_constraint",
                "mc_value_ps": -1240.0,
                "lib_value_ps": -1250.0,
                "dif_ps": -10.0,
                "is_optimistic_risk": True,
            }
        ]
    )

    sensitivity, warnings = build_sensitivity_rows(normalized, load_policy())
    outputs = build_margin_outputs(normalized, sensitivity, load_policy())

    assert sensitivity.empty
    assert warnings.iloc[0]["warning_code"] == "insufficient_voltage_points"
    row = outputs.all_errors_per_object.iloc[0]
    assert row["margin_status"] == "skipped_no_sensitivity"
    assert math.isnan(row["required_margin_mv"])


def test_default_config_files_load():
    column_mapping = load_column_mapping()
    policy = load_policy()

    assert column_mapping["file_loading"]["skiprows"] == [1]
    assert column_mapping["analysis_metrics"]["hold"] == ["Late_Sigma", "Nominal"]
    assert policy["direction_rule"]["scope"] == "global"
    assert policy["thresholds"]["hold"]["Nominal"]["abs_threshold"]["floor_ps"] == 10.0


def test_cli_writes_phase1_output_package(tmp_path):
    columns = [
        "Arc",
        "Cell_Name",
        "rel_pin_slew",
        "Table_Type",
        "MC_Late_Sigma",
        "CDNS_Lib_Late_Sigma",
        "CDNS_Lib_Late_Sigma_Dif",
        "CDNS_Lib_Late_Sigma_Rel",
        "MC_Late_Sigma_LB",
        "MC_Late_Sigma_UB",
        "MC_Nominal",
        "CDNS_Lib_Nominal",
        "CDNS_Lib_Nominal_Dif",
        "CDNS_Lib_Nominal_Rel",
    ]
    arc = "hold_CELL_D_rise_CP_rise_NO_CONDITION_4_2"
    for voltage_token, late_lib, nominal_lib, dif in [
        ("0p450v", 18.0, -1250.0, -10.0),
        ("0p465v", 21.0, -1244.0, -4.0),
    ]:
        write_rpt(
            tmp_path / f"fmc_result_n2p_v1p0_ssgnp_{voltage_token}_m40c_hold_fmc_cdns_lib_comp.rpt",
            columns,
            [
                {
                    "Arc": arc,
                    "Cell_Name": "CELL",
                    "rel_pin_slew": 40.0,
                    "Table_Type": "rise_constraint",
                    "MC_Late_Sigma": 20.0,
                    "CDNS_Lib_Late_Sigma": late_lib,
                    "CDNS_Lib_Late_Sigma_Dif": dif,
                    "CDNS_Lib_Late_Sigma_Rel": -0.10,
                    "MC_Late_Sigma_LB": 19.0,
                    "MC_Late_Sigma_UB": 21.0,
                    "MC_Nominal": -1240.0,
                    "CDNS_Lib_Nominal": nominal_lib,
                    "CDNS_Lib_Nominal_Dif": dif,
                    "CDNS_Lib_Nominal_Rel": -0.008,
                }
            ],
        )

    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "run_analysis.py",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--types",
            "hold",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (output_dir / "manifest.csv").exists()
    assert (output_dir / "normalized" / "normalized_rows.csv").exists()
    assert (output_dir / "sensitivity" / "sensitivity.csv").exists()
    assert (output_dir / "all_errors" / "per_object_margin.csv").exists()
    assert (output_dir / "optimistic_only" / "per_object_margin.csv").exists()
    assert (output_dir / "pass_rate" / "pass_rate_results.csv").exists()
    assert (output_dir / "trace" / "source_rows.csv").exists()
    assert (output_dir / "trace" / "sensitivity_source_refs.csv").exists()
    assert (output_dir / "trace" / "margin_trace.csv").exists()

    per_arc = pd.read_csv(output_dir / "pass_rate" / "per_arc_pass_fail.csv")
    expected_per_arc_columns = {
        "process",
        "process_version",
        "corner",
        "voltage_v",
        "analysis_type",
        "metric",
        "arc",
        "mc_value_ps",
        "lib_value_ps",
        "dif_ps",
        "rel_err",
        "rel_pin_slew_ps",
        "mc_ci_lb_ps",
        "mc_ci_ub_ps",
        "rel_threshold",
        "abs_threshold_ps",
    }
    assert expected_per_arc_columns.issubset(per_arc.columns)
    assert "Corner" not in per_arc.columns
    assert "Parameter" not in per_arc.columns
    assert per_arc["process"].unique().tolist() == ["n2p"]
    assert per_arc["process_version"].unique().tolist() == ["v1p0"]
    assert set(per_arc["metric"]) == {"Late_Sigma", "Nominal"}

    optimistic = pd.read_csv(output_dir / "optimistic_only" / "per_object_margin.csv")
    assert not optimistic.empty
    assert set(optimistic["is_optimistic_risk"]) == {True}
    assert {"margin_trace_id", "normalized_trace_id", "sensitivity_trace_id"}.issubset(
        optimistic.columns)

    source_trace = pd.read_csv(output_dir / "trace" / "source_rows.csv")
    assert {"trace_id", "source_file", "source_line_number"}.issubset(source_trace.columns)
    sensitivity_trace = pd.read_csv(output_dir / "trace" / "sensitivity_source_refs.csv")
    assert {"sensitivity_trace_id", "pair_side", "source_trace_id"}.issubset(
        sensitivity_trace.columns)
    margin_trace = pd.read_csv(output_dir / "trace" / "margin_trace.csv")
    assert {"margin_trace_id", "normalized_trace_id", "sensitivity_trace_id"}.issubset(
        margin_trace.columns)
