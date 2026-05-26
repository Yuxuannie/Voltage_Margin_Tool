# Traceability Core Design

## Purpose

Voltage Margin Tool Phase 1 already computes normalized rows, adjacent-pair
sensitivity, pass-rate outputs, and per-object voltage margins. The next
requirement is auditability: every displayed voltage margin must be traceable
back to the exact source `.rpt` file rows and the formula inputs used to compute
it.

This design adds traceability metadata to the backend output package before the
GUI is upgraded. The GUI will rely on these fields for clickable `path:line`
navigation, formula panels, raw source row lookup, and trace exports.

## Scope

In scope:

- Add source path and line provenance for every normalized row.
- Add source-column provenance for normalized metric values.
- Preserve both absolute and input-root-relative source paths.
- Preserve both physical source line number and pandas row index.
- Add formula identifiers and human-readable formula strings.
- Trace sensitivity low/high voltage points back to every normalized row that
  contributed to the voltage mean.
- Add margin trace records that connect normalized rows, sensitivity rows, and
  margin outputs.
- Keep existing algorithms and acceptance behavior unchanged.

Out of scope:

- Building the GUI itself.
- Copying raw `.rpt` row text into the main result CSVs.
- Changing threshold, waiver, sensitivity, or margin formulas.
- Adding setup/recovery/removal, lib-to-lib mode, rpt generation, or
  orchestration.

## Output Package Changes

The existing output package remains backward-compatible. Existing CSVs keep
their current columns and gain additive trace columns.

New trace directory:

```text
voltage_margin_outputs/
  trace/
    source_rows.csv
    sensitivity_source_refs.csv
    margin_trace.csv
```

## Normalized Row Trace

`normalized/normalized_rows.csv` gains essential source trace columns:

```text
trace_id
input_root
source_file
source_file_relative
source_line_number
source_row_index
source_mc_column
source_lib_column
source_dif_column
source_rel_column
```

Definitions:

- `trace_id`: stable row identifier for joining trace outputs within a run.
- `input_root`: absolute input directory used for the run.
- `source_file`: absolute source `.rpt` path.
- `source_file_relative`: source path relative to `input_root`.
- `source_line_number`: physical line number in the `.rpt` file. Header is
  line 1, blank line is line 2, first data row is line 3.
- `source_row_index`: pandas dataframe row index. First data row is index 0.
- `source_*_column`: source column names used for the normalized metric row.

`trace/source_rows.csv` mirrors one row per normalized row and includes the same
trace identifiers. It is intended for GUI joins and future trace exports. It
does not copy full raw source row text; the GUI will lazy-load raw text from
`source_file:source_line_number`.

## Sensitivity Trace

`sensitivity/sensitivity.csv` gains:

```text
sensitivity_trace_id
low_voltage_v
high_voltage_v
low_lib_value_ps
high_lib_value_ps
low_source_refs_summary
high_source_refs_summary
sensitivity_formula_id
sensitivity_formula
```

The sensitivity algorithm remains adjacent-pair local slope:

```text
sensitivity_ps_per_mv = abs(high_lib_value_ps - low_lib_value_ps) / abs((high_voltage_v - low_voltage_v) * 1000)
```

When multiple normalized rows exist for the same voltage point, the current mean
behavior is preserved. The trace records every contributing source row rather
than choosing one representative row. `low_lib_value_ps` and
`high_lib_value_ps` are the mean Lib values used in the slope calculation.

`trace/sensitivity_source_refs.csv` stores detailed low/high source
contributions:

```text
sensitivity_trace_id
pair_side
source_trace_id
input_root
source_file
source_file_relative
source_line_number
source_row_index
voltage_v
lib_value_ps
```

`pair_side` is `low` or `high`.

## Margin Trace

`all_errors/per_object_margin.csv` and
`optimistic_only/per_object_margin.csv` gain:

```text
margin_trace_id
normalized_trace_id
sensitivity_trace_id
source_file
source_file_relative
source_line_number
margin_formula_id
required_margin_formula
signed_margin_formula
```

Margin formulas remain:

```text
required_margin_mv = abs(dif_ps) / sensitivity_ps_per_mv
signed_margin_mv = dif_ps / sensitivity_ps_per_mv
```

`trace/margin_trace.csv` provides one audit record per margin row:

```text
margin_trace_id
normalized_trace_id
sensitivity_trace_id
input_root
source_file
source_file_relative
source_line_number
source_row_index
dif_ps
sensitivity_ps_per_mv
required_margin_mv
signed_margin_mv
margin_formula_id
required_margin_formula
signed_margin_formula
```

The margin trace links a margin row to the current normalized source row and the
sensitivity trace row. Detailed low/high voltage source contributions are
resolved through `sensitivity_trace_id`.

## GUI Usage Model

The future Phase 1 GUI will use these outputs as follows:

1. User selects a margin row.
2. GUI displays `required_margin_formula` and `signed_margin_formula`.
3. GUI opens or copies `source_file:source_line_number` for the current error
   row.
4. GUI uses `sensitivity_trace_id` to show all low/high voltage source rows that
   contributed to the sensitivity.
5. GUI lazy-loads raw `.rpt` row text only when the user asks to view it.

## Compatibility

This work is additive. Existing acceptance checks should continue to pass:

- Existing output files remain in place.
- Existing columns are not renamed or removed.
- Existing calculations are not changed.
- New trace files are additional outputs.

## Testing Plan

Unit tests should cover:

- Normalized rows include correct absolute path, relative path, physical line
  number, dataframe row index, and source column names.
- Sensitivity rows retain adjacent-pair behavior and include stable
  `sensitivity_trace_id` values.
- Duplicate voltage rows keep mean behavior while
  `trace/sensitivity_source_refs.csv` records every contributing source row.
- Margin outputs include formula ids, formula strings, normalized trace ids, and
  sensitivity trace ids.
- Trace CSVs are written by the CLI output package.

Acceptance tests should still pass after trace columns are added.

## Open Decisions

No open product decisions remain for this design. Implementation may choose
exact `trace_id` formatting, but ids must be stable within one output package
and safe for CSV joins.
