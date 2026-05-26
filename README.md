# Voltage Margin Tool

Phase 1 analyzes CDNS library-characterization compare reports against Monte
Carlo golden targets. It preserves the existing pass-rate summary workflow and
adds normalized data, local voltage sensitivity, and per-object voltage margin
CSV outputs.

Phase 1 scope is intentionally narrow:

- Golden-target mode only.
- CDNS report schemas only.
- CLI and core modules only.
- No setup/recovery/removal analysis.
- No lib-to-lib mode.
- No rpt generation or FMC/LDBX orchestration.
- GUI is legacy and not updated for Phase 1.

## Install

```bash
pip install -r requirements.txt
```

The Phase 1 CLI uses `pandas`, `numpy`, `scipy`, and `PyYAML`.

## Input Reports

Put compare `.rpt` files in one directory. Reports are CSV files with:

- row 1: header
- row 2: blank line
- row 3 onward: data

Supported filename format:

```text
(fmc|MC)_result_<process>_<version>_<corner>_<analysis_type>_<compare_kind>.rpt
```

Examples:

```text
fmc_result_n2p_v1p0_ssgnp_0p450v_m40c_delay_fmc_cdns_lib_comp.rpt
MC_result_n2p_v1p0_ssgnp_0p450v_m40c_delay_moments_comp.rpt
```

The tool parses:

- `process`: `n2p`
- `process_version`: `v1p0`
- `corner`: `ssgnp_0p450v_m40c`
- `voltage_v`: `0.450`
- `temperature_c`: `-40`
- `analysis_type`: `delay`, `slew`, or `hold`

## Supported Metrics

Delay and slew support:

```text
Late_Sigma, Early_Sigma, Std, Skew, Meanshift, Nominal
```

Hold supports only:

```text
Late_Sigma, Nominal
```

Hold reports do not contain `Early_Sigma`, `Std`, `Skew`, or `Meanshift`, and
Phase 1 does not try to synthesize them.

## CLI

```bash
python run_analysis.py DATA_DIR \
  --output-dir voltage_margin_outputs \
  --column-map voltage_margin/config/column_mapping_phase1.yaml \
  --policy voltage_margin/config/policy_phase1.yaml \
  --corners ssgnp_0p450v_m40c ssgnp_0p465v_m40c \
  --types delay slew hold
```

Arguments:

- `DATA_DIR`: directory containing supported `.rpt` files.
- `--output-dir`: output package directory. Default: `voltage_margin_outputs`.
- `--output`, `-o`: pass-rate summary CSV path. Default:
  `<output-dir>/pass_rate/pass_rate_results.csv`.
- `--column-map`: column mapping YAML. Default:
  `voltage_margin/config/column_mapping_phase1.yaml`.
- `--policy`: threshold and waiver policy YAML. Default:
  `voltage_margin/config/policy_phase1.yaml`.
- `--corners`: optional corner filter.
- `--types`: optional analysis type filter: `delay`, `slew`, `hold`.
- `--no-waiver1`: hide CI-enlargement waiver columns in the summary view.
- `--no-optimistic`: hide optimistic-direction waiver columns in the summary view.
- `--verbose`, `-v`: debug logging.

The pass-rate summary CSV keeps its pre-Phase-1 column names for backward
compatibility:

```text
Corner, Type, Parameter, Total_Arcs, Base_PR, PR_with_Waiver1,
PR_Optimistic_Only, PR_with_Both_Waivers
```

## Output Package

```text
voltage_margin_outputs/
  manifest.csv
  normalized/
    normalized_rows.csv
    normalization_warnings.csv
  pass_rate/
    pass_rate_results.csv
    per_arc_pass_fail.csv
  sensitivity/
    sensitivity.csv
    sensitivity_warnings.csv
  all_errors/
    per_object_margin.csv
    margin_summary.csv
    margin_efficiency_curve.csv
    high_margin_objects.csv
  optimistic_only/
    per_object_margin.csv
    margin_summary.csv
    margin_efficiency_curve.csv
    high_margin_objects.csv
```

`all_errors/` includes optimistic and pessimistic rows. `optimistic_only/`
includes only rows where:

```text
dif_ps < 0
```

The global direction rule is `Dif = Lib - MC`; `Dif < 0` means `Lib < MC`,
which is optimistic risk.

## Normalization

`normalized/normalized_rows.csv` emits one row per report row and canonical
metric. The normalized `metric` value is always canonical PascalCase, regardless
of source column case, for example `Late_Sigma`, not `late_sigma`.

Matching uses canonical `arc` only. FMC arcs ending in `_N_M` are normalized to
`_N-M` so they can match MC-style arcs.

`Cell_Name` is optional internally:

- `fmc_result_*.rpt` reads it from the `Cell_Name` column.
- `MC_result_*.rpt` parses it from `Arc`.

## Sensitivity

Phase 1 sensitivity uses source-library values from the compare report's Lib
side. It does not use MC values for sensitivity.

The method is an adjacent-pair local slope per
`(compare_source, process, process_version, analysis_type, metric, arc,
corner_family, temperature_c)` group:

```text
slope_ps_per_v = (lib_high_ps - lib_low_ps) / (v_high - v_low)
sensitivity_ps_per_mv = abs(slope_ps_per_v) / 1000
```

Boundary target corners emit one sensitivity row using the only adjacent pair.
Intermediate target corners emit two rows, `lower_pair` and `upper_pair`.
Rows with fewer than two voltage points are skipped with a warning.

`sensitivity/sensitivity.csv` columns:

```text
process, process_version, compare_source, analysis_type, metric, arc,
corner_family, temperature_c, corner, voltage_v,
pair_v_low, pair_v_high, pair_role,
voltage_values_v, lib_values_ps,
slope_ps_per_v, sensitivity_ps_per_mv,
intercept_ps, fit_r2, valid_points, fit_status, warning
```

`valid_points` is the number of unique voltage points used in the fit.

## Voltage Margin

`per_object_margin.csv` joins each normalized error row to its fitted
sensitivity row:

```text
required_margin_mv = abs(dif_ps) / sensitivity_ps_per_mv
signed_margin_mv = dif_ps / sensitivity_ps_per_mv
```

`margin_summary.csv` and `margin_efficiency_curve.csv` aggregate directly over
per-object margin rows. They include:

```text
aggregation = per_object
```

## Policy And Mapping

Thresholds, waivers, direction rule, and margin points live in:

```text
voltage_margin/config/policy_phase1.yaml
```

Column names, filename parsing, metric availability, arc normalization, and diff
consistency tolerance live in:

```text
voltage_margin/config/column_mapping_phase1.yaml
```

Do not edit Python constants for threshold tuning. Edit the policy YAML.

## GUI

The Tkinter GUI is still present for the old workflow:

```bash
python run_gui.py
```

Phase 1 does not update the GUI. It does not expose normalized rows, per-pair
sensitivity, per-object margin outputs, or the YAML policy/mapping controls.
Those are CLI/core features for this phase.
