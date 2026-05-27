from pathlib import Path

import pandas as pd

from voltage_margin.gui.phase1_backend import (
    Phase1RunConfig,
    build_target_margin_plan,
    build_margin_audit_rows,
    build_vm_sweep,
    build_vm_target_summary,
    find_vm_observations,
    enrich_margin_rows,
    filter_margins,
    format_margin_trace_detail,
    load_output_package,
    read_source_context,
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


def test_build_target_margin_plan_answers_95_percent_by_corner_and_type():
    margins = pd.DataFrame(
        [
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "arc": "a1",
                "required_margin_mv": 1.0,
                "margin_status": "ok",
                "margin_trace_id": "m1",
            },
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Skew",
                "arc": "a2",
                "required_margin_mv": 3.0,
                "margin_status": "ok",
                "margin_trace_id": "m2",
            },
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Nominal",
                "arc": "a3",
                "required_margin_mv": 9.0,
                "margin_status": "ok",
                "margin_trace_id": "m3",
            },
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "slew",
                "metric": "Late_Sigma",
                "arc": "s1",
                "required_margin_mv": 2.0,
                "margin_status": "ok",
                "margin_trace_id": "s1",
            },
        ]
    )

    plan = build_target_margin_plan(margins, target_coverage=0.95)

    delay = plan[plan["analysis_type"] == "delay"].iloc[0]
    assert delay["required_margin_mv"] == 9.0
    assert delay["covered_rows"] == 3
    assert delay["valid_rows"] == 3
    assert delay["coverage_pct"] == 100.0
    assert delay["worst_metric"] == "Nominal"
    assert delay["worst_margin_trace_id"] == "m3"


def test_build_vm_sweep_models_all_rows_and_outliers_only_scopes():
    margins = pd.DataFrame(
        [
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Meanshift",
                "arc": "optimistic_big",
                "mc_value_ps": 100.0,
                "dif_ps": -3.0,
                "sensitivity_ps_per_mv": 1.0,
                "abs_threshold_ps": 1.0,
                "rel_threshold": 0.01,
                "base_pass": False,
            },
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Meanshift",
                "arc": "optimistic_small",
                "mc_value_ps": 100.0,
                "dif_ps": -2.0,
                "sensitivity_ps_per_mv": 1.0,
                "abs_threshold_ps": 1.0,
                "rel_threshold": 0.01,
                "base_pass": False,
            },
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Meanshift",
                "arc": "pessimistic_pass",
                "mc_value_ps": 100.0,
                "dif_ps": 0.5,
                "sensitivity_ps_per_mv": 1.0,
                "abs_threshold_ps": 1.0,
                "rel_threshold": 0.01,
                "base_pass": True,
            },
        ]
    )

    sweep = build_vm_sweep(margins, max_margin_mv=2, step_mv=1)

    outlier_2mv = sweep[
        (sweep["scope"] == "outliers_only") & (sweep["margin_mv"] == 2)
    ].iloc[0]
    all_2mv = sweep[(sweep["scope"] == "all_rows") & (sweep["margin_mv"] == 2)].iloc[0]

    assert outlier_2mv["pass_rate_pct"] == 100.0
    assert outlier_2mv["fixed_count"] == 2
    assert outlier_2mv["new_fail_count"] == 0
    assert all_2mv["pass_rate_pct"] == 66.666667
    assert all_2mv["fixed_count"] == 2
    assert all_2mv["new_fail_count"] == 1


def test_outliers_only_uses_all_rows_denominator_but_only_updates_base_fails():
    margins = pd.DataFrame(
        [
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Skew",
                "arc": "base_pass_that_would_fail",
                "mc_value_ps": 100.0,
                "dif_ps": 0.5,
                "sensitivity_ps_per_mv": 1.0,
                "abs_threshold_ps": 1.0,
                "rel_threshold": 0.01,
                "base_pass": True,
            },
            {
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Skew",
                "arc": "base_fail_fixed",
                "mc_value_ps": 100.0,
                "dif_ps": -2.0,
                "sensitivity_ps_per_mv": 1.0,
                "abs_threshold_ps": 1.0,
                "rel_threshold": 0.01,
                "base_pass": False,
            },
        ]
    )

    sweep = build_vm_sweep(margins, max_margin_mv=2, step_mv=1)
    outlier_2mv = sweep[
        (sweep["scope"] == "outliers_only") & (sweep["margin_mv"] == 2)
    ].iloc[0]
    all_2mv = sweep[(sweep["scope"] == "all_rows") & (sweep["margin_mv"] == 2)].iloc[0]

    assert outlier_2mv["total_count"] == 2
    assert outlier_2mv["pass_count"] == 2
    assert outlier_2mv["new_fail_count"] == 0
    assert all_2mv["pass_count"] == 1
    assert all_2mv["new_fail_count"] == 1


def test_build_vm_target_summary_finds_first_margin_reaching_target():
    sweep = pd.DataFrame(
        [
            {
                "scope": "outliers_only",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Meanshift",
                "margin_mv": 0,
                "pass_rate_pct": 82.2,
                "pass_count": 822,
                "total_count": 1000,
                "fixed_count": 0,
                "new_fail_count": 0,
            },
            {
                "scope": "outliers_only",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Meanshift",
                "margin_mv": 1,
                "pass_rate_pct": 87.6,
                "pass_count": 876,
                "total_count": 1000,
                "fixed_count": 54,
                "new_fail_count": 0,
            },
            {
                "scope": "outliers_only",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Meanshift",
                "margin_mv": 2,
                "pass_rate_pct": 95.4,
                "pass_count": 954,
                "total_count": 1000,
                "fixed_count": 132,
                "new_fail_count": 0,
            },
        ]
    )

    summary = build_vm_target_summary(sweep, target_pass_rate_pct=95.0)

    row = summary.iloc[0]
    assert row["required_vm_mv"] == 2
    assert row["target_pass_rate_pct"] == 95.0
    assert row["pass_rate_at_required_vm_pct"] == 95.4
    assert row["base_pr_pct"] == 82.2


def test_build_vm_target_summary_reports_best_when_target_not_reached():
    sweep = pd.DataFrame(
        [
            {
                "scope": "all_rows",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "margin_mv": 0,
                "pass_rate_pct": 74.0,
                "pass_count": 74,
                "total_count": 100,
                "fixed_count": 0,
                "new_fail_count": 0,
            },
            {
                "scope": "all_rows",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "margin_mv": 1,
                "pass_rate_pct": 52.0,
                "pass_count": 52,
                "total_count": 100,
                "fixed_count": 1,
                "new_fail_count": 23,
            },
        ]
    )

    summary = build_vm_target_summary(sweep, target_pass_rate_pct=95.0)

    row = summary.iloc[0]
    assert pd.isna(row["required_vm_mv"])
    assert row["status"] == "not_reached"
    assert row["best_vm_mv"] == 0
    assert row["best_pr_pct"] == 74.0
    assert row["trend"] == "declines"


def test_find_vm_observations_flags_all_rows_decline_and_outlier_best_point():
    sweep = pd.DataFrame(
        [
            {
                "scope": "all_rows",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "margin_mv": 0,
                "pass_rate_pct": 80.0,
                "pass_count": 80,
                "total_count": 100,
                "fixed_count": 0,
                "new_fail_count": 0,
            },
            {
                "scope": "all_rows",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "margin_mv": 1,
                "pass_rate_pct": 70.0,
                "pass_count": 70,
                "total_count": 100,
                "fixed_count": 2,
                "new_fail_count": 12,
            },
            {
                "scope": "outliers_only",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "margin_mv": 0,
                "pass_rate_pct": 80.0,
                "pass_count": 80,
                "total_count": 100,
                "fixed_count": 0,
                "new_fail_count": 0,
            },
            {
                "scope": "outliers_only",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "margin_mv": 1,
                "pass_rate_pct": 90.0,
                "pass_count": 90,
                "total_count": 100,
                "fixed_count": 10,
                "new_fail_count": 0,
            },
            {
                "scope": "outliers_only",
                "compare_source": "fmc_compare",
                "corner": "c1",
                "analysis_type": "delay",
                "metric": "Late_Sigma",
                "margin_mv": 2,
                "pass_rate_pct": 85.0,
                "pass_count": 85,
                "total_count": 100,
                "fixed_count": 5,
                "new_fail_count": 0,
            },
        ]
    )

    observations = find_vm_observations(sweep)

    assert "all_rows_degrades" in observations["observation_code"].tolist()
    assert "outliers_only_has_peak" in observations["observation_code"].tolist()


def test_build_margin_audit_rows_limits_to_most_relevant_rows():
    margins = pd.DataFrame(
        [
            {"arc": "small", "required_margin_mv": 1.0, "base_pass": True},
            {"arc": "large", "required_margin_mv": 9.0, "base_pass": False},
            {"arc": "mid", "required_margin_mv": 5.0, "base_pass": False},
        ]
    )

    audit = build_margin_audit_rows(margins, limit=2)

    assert audit["arc"].tolist() == ["large", "mid"]


def test_enrich_margin_rows_adds_sensitivity_sources_for_audit():
    margins = pd.DataFrame(
        [{"sensitivity_trace_id": "sens-1", "required_margin_mv": 2.0}]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "sensitivity_trace_id": "sens-1",
                "low_source_refs_summary": "low.rpt:3",
                "high_source_refs_summary": "high.rpt:3",
                "sensitivity_formula": "sensitivity_ps_per_mv = abs(3 - 1) / 15",
            }
        ]
    )

    enriched = enrich_margin_rows(margins, sensitivity)

    assert enriched.iloc[0]["low_source_refs_summary"] == "low.rpt:3"
    assert "sensitivity_ps_per_mv" in enriched.iloc[0]["sensitivity_formula"]


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


def test_read_source_context_marks_requested_line(tmp_path):
    source = tmp_path / "source.rpt"
    source.write_text("header\ncolumns\nrow-one\nrow-two\nrow-three\n")

    context = read_source_context(source, 4, radius=1)

    assert "   3 | row-one" in context
    assert ">> 4 | row-two" in context
    assert "   5 | row-three" in context
