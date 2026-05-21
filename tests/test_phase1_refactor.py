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

    nominal = normalized[normalized["metric"] == "Nominal"].iloc[0]
    assert nominal["mc_value_ps"] == -1240.43
    assert nominal["dif_ps"] == -10.0
    assert nominal["is_optimistic_risk"] is True


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


def test_sensitivity_uses_single_multivoltage_fit_for_target_corner():
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

    sensitivity, warnings = build_sensitivity_rows(normalized, load_policy())

    assert warnings.empty
    assert len(sensitivity) == 3

    middle = sensitivity[sensitivity["corner"] == "ssgnp_0p465v_m40c"]
    assert len(middle) == 1
    assert middle.iloc[0]["valid_points"] == 3
    assert math.isclose(middle.iloc[0]["sensitivity_ps_per_mv"], 3.0)
    assert 0.96 < middle.iloc[0]["fit_r2"] < 0.97

    margins = build_margin_outputs(normalized, sensitivity, load_policy())
    optimistic = margins.optimistic_only_per_object
    assert set(optimistic["corner"]) == {"ssgnp_0p465v_m40c", "ssgnp_0p480v_m40c"}
    mid_margins = optimistic[optimistic["corner"] == "ssgnp_0p465v_m40c"]
    assert len(mid_margins) == 1
    assert math.isclose(mid_margins.iloc[0]["required_margin_mv"], 2.0)

    mid_summary = margins.optimistic_only_summary[
        margins.optimistic_only_summary["corner"] == "ssgnp_0p465v_m40c"
    ].iloc[0]
    assert mid_summary["aggregation"] == "per_object"
    assert math.isclose(mid_summary["max_margin_mv"], 2.0)


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
