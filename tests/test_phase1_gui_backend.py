from pathlib import Path

import pandas as pd

from voltage_margin.gui.phase1_backend import (
    Phase1RunConfig,
    filter_margins,
    format_margin_trace_detail,
    load_output_package,
    read_source_line,
    run_phase1_pipeline,
    summarize_margins,
)


def write_rpt(path: Path, columns, rows):
    lines = [",".join(columns), ""]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in columns))
    path.write_text("\n".join(lines) + "\n")


def write_hold_fixture(data_dir: Path):
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
            data_dir / f"fmc_result_n2p_v1p0_ssgnp_{voltage_token}_m40c_hold_fmc_cdns_lib_comp.rpt",
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


def test_phase1_backend_runs_and_loads_traceable_output(tmp_path):
    write_hold_fixture(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_phase1_pipeline(
        Phase1RunConfig(data_dir=tmp_path, output_dir=output_dir, types=["hold"])
    )
    tables = load_output_package(result.output_dir)

    assert result.parameter_groups == 4
    assert result.total_arcs == 4
    assert not tables.pass_rate.empty
    assert not tables.all_margins.empty
    assert not tables.sensitivity.empty
    assert not tables.source_rows.empty
    assert not tables.sensitivity_source_refs.empty
    assert not tables.margin_trace.empty


def test_filter_margins_filters_by_user_visible_fields():
    margins = pd.DataFrame(
        [
            {
                "compare_source": "fmc_compare",
                "analysis_type": "hold",
                "metric": "Nominal",
                "corner": "ssgnp_0p450v_m40c",
                "margin_status": "ok",
            },
            {
                "compare_source": "mc_compare",
                "analysis_type": "delay",
                "metric": "Skew",
                "corner": "ssgnp_0p465v_m40c",
                "margin_status": "skipped_no_sensitivity",
            },
        ]
    )

    filtered = filter_margins(
        margins,
        source="fmc_compare",
        analysis_type="hold",
        metric="Nominal",
        corner="ssgnp_0p450v_m40c",
        status="ok",
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["compare_source"] == "fmc_compare"


def test_summarize_margins_counts_status_and_trace_rows():
    margins = pd.DataFrame(
        [
            {"margin_status": "ok"},
            {"margin_status": "ok"},
            {"margin_status": "skipped_no_sensitivity"},
        ]
    )
    trace = pd.DataFrame([{"margin_trace_id": "m-1"}, {"margin_trace_id": "m-2"}])

    summary = summarize_margins(margins, trace)

    assert summary.total_margins == 3
    assert summary.ok_margins == 2
    assert summary.needs_review == 1
    assert summary.trace_rows == 2


def test_load_output_package_treats_empty_csv_files_as_empty_tables(tmp_path):
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized").mkdir(parents=True)
    (output_dir / "normalized" / "normalization_warnings.csv").write_text("")

    tables = load_output_package(output_dir)

    assert tables.normalization_warnings.empty


def test_margin_trace_detail_and_source_line_lookup(tmp_path):
    source = tmp_path / "source.rpt"
    source.write_text("header\n\nrow-one\nrow-two\n")
    row = pd.Series(
        {
            "corner": "ssgnp_0p450v_m40c",
            "analysis_type": "hold",
            "metric": "Nominal",
            "arc": "hold_CELL_D_rise_CP_rise_NO_CONDITION_4-2",
            "source_file": str(source),
            "source_line_number": 4,
            "required_margin_formula": "required_margin_mv = abs(-10) / 0.4",
            "signed_margin_formula": "signed_margin_mv = -10 / 0.4",
            "sensitivity_trace_id": "sens-00000001",
            "low_source_refs_summary": "low.rpt:3",
            "high_source_refs_summary": "high.rpt:3",
        }
    )

    detail = format_margin_trace_detail(row)

    assert detail["path_line"] == f"{source}:4"
    assert "required_margin_mv" in detail["required_formula"]
    assert detail["low_sources"] == "low.rpt:3"
    assert read_source_line(source, 4) == "row-two"
