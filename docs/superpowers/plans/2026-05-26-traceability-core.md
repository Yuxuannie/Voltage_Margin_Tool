# Traceability Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend traceability metadata so every Phase 1 voltage margin can be traced to source `.rpt` file rows and formula inputs.

**Architecture:** Keep the existing CLI and output package structure, adding trace columns and additive `trace/*.csv` files. Normalization owns source file/line/column provenance, sensitivity owns low/high pair provenance, and margin output joins both traces with formula strings.

**Tech Stack:** Python, pandas, pytest, existing `voltage_margin.core` modules.

---

### Task 1: Normalized Source Provenance

**Files:**
- Modify: `voltage_margin/core/normalizer.py`
- Test: `tests/test_phase1_refactor.py`

- [ ] **Step 1: Write the failing test**

Add assertions to `test_normalized_loader_handles_hold_blank_line_negative_nominal_and_arc_suffix`:

```python
assert {"trace_id", "input_root", "source_file", "source_file_relative",
        "source_line_number", "source_row_index", "source_mc_column",
        "source_lib_column", "source_dif_column", "source_rel_column"}.issubset(normalized.columns)
assert normalized["source_line_number"].unique().tolist() == [3]
assert normalized["source_row_index"].unique().tolist() == [0]
assert normalized["source_file_relative"].str.endswith(".rpt").all()
nominal = normalized[normalized["metric"] == "Nominal"].iloc[0]
assert nominal["source_mc_column"] == "MC_Nominal"
assert nominal["source_lib_column"] == "CDNS_Lib_Nominal"
assert nominal["source_dif_column"] == "CDNS_Lib_Nominal_Dif"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_phase1_refactor.py::test_normalized_loader_handles_hold_blank_line_negative_nominal_and_arc_suffix -q`

Expected: FAIL because trace columns are missing.

- [ ] **Step 3: Implement normalized provenance**

Add `input_root = str(Path(data_dir).resolve())` in `load_normalized_data`, pass it into `_normalize_file`, and add trace fields per emitted metric row. Use physical line number `int(row_index) + 3`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_phase1_refactor.py::test_normalized_loader_handles_hold_blank_line_negative_nominal_and_arc_suffix -q`

Expected: PASS.

### Task 2: Sensitivity Pair Provenance

**Files:**
- Modify: `voltage_margin/core/margin_engine.py`
- Test: `tests/test_phase1_refactor.py`

- [ ] **Step 1: Write the failing test**

Extend `test_sensitivity_uses_adjacent_pair_local_slope_for_target_corner` to assert:

```python
assert {"sensitivity_trace_id", "low_voltage_v", "high_voltage_v",
        "low_lib_value_ps", "high_lib_value_ps", "low_source_refs_summary",
        "high_source_refs_summary", "sensitivity_formula_id",
        "sensitivity_formula"}.issubset(sensitivity.columns)
assert sensitivity["sensitivity_trace_id"].notna().all()
assert sensitivity["sensitivity_formula_id"].eq("adjacent_pair_lib_slope_ps_per_mv").all()
```

Add a duplicate-voltage test that creates two normalized rows at the same voltage with different `trace_id` and `lib_value_ps`, then asserts `sensitivity.attrs["source_refs"]` contains both low-side source trace ids.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_phase1_refactor.py::test_sensitivity_uses_adjacent_pair_local_slope_for_target_corner tests/test_phase1_refactor.py::test_sensitivity_trace_records_all_duplicate_voltage_source_rows -q`

Expected: FAIL because columns and attrs are missing.

- [ ] **Step 3: Implement sensitivity provenance**

In `build_sensitivity_rows`, preserve mean behavior but keep every contributing normalized row for low/high voltage points. Add sensitivity trace columns and store detailed refs in `sensitivity_df.attrs["source_refs"]`.

- [ ] **Step 4: Run tests to verify pass**

Run the same two pytest targets.

Expected: PASS.

### Task 3: Margin Trace and Formula Metadata

**Files:**
- Modify: `voltage_margin/core/margin_engine.py`
- Test: `tests/test_phase1_refactor.py`

- [ ] **Step 1: Write the failing test**

Extend the sensitivity/margin test to assert optimistic margin rows include:

```python
assert {"margin_trace_id", "normalized_trace_id", "sensitivity_trace_id",
        "source_file", "source_file_relative", "source_line_number",
        "margin_formula_id", "required_margin_formula",
        "signed_margin_formula"}.issubset(optimistic.columns)
assert optimistic["margin_formula_id"].eq("margin_from_dif_and_sensitivity").all()
assert hasattr(margins, "margin_trace")
assert not margins.margin_trace.empty
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_phase1_refactor.py::test_sensitivity_uses_adjacent_pair_local_slope_for_target_corner -q`

Expected: FAIL because margin trace fields are missing.

- [ ] **Step 3: Implement margin trace**

Add `margin_trace` to `MarginOutputs`, populate trace ids and formula strings in `build_margin_outputs`, and retain source path/line fields from normalized rows.

- [ ] **Step 4: Run test to verify pass**

Run the same pytest target.

Expected: PASS.

### Task 4: CLI Trace CSV Outputs

**Files:**
- Modify: `run_analysis.py`
- Test: `tests/test_phase1_refactor.py`

- [ ] **Step 1: Write the failing test**

Extend `test_cli_writes_phase1_output_package` to assert:

```python
assert (output_dir / "trace" / "source_rows.csv").exists()
assert (output_dir / "trace" / "sensitivity_source_refs.csv").exists()
assert (output_dir / "trace" / "margin_trace.csv").exists()
source_trace = pd.read_csv(output_dir / "trace" / "source_rows.csv")
assert {"trace_id", "source_file", "source_line_number"}.issubset(source_trace.columns)
margin_trace = pd.read_csv(output_dir / "trace" / "margin_trace.csv")
assert {"margin_trace_id", "normalized_trace_id", "sensitivity_trace_id"}.issubset(margin_trace.columns)
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_phase1_refactor.py::test_cli_writes_phase1_output_package -q`

Expected: FAIL because trace CSVs are missing.

- [ ] **Step 3: Implement trace CSV writing**

Write `trace/source_rows.csv` from normalized trace columns, `trace/sensitivity_source_refs.csv` from `sensitivity_df.attrs["source_refs"]`, and `trace/margin_trace.csv` from `margin_outputs.margin_trace`.

- [ ] **Step 4: Run test to verify pass**

Run the same pytest target.

Expected: PASS.

### Task 5: Full Verification

**Files:**
- No new code unless verification exposes a failure.

- [ ] **Step 1: Run full pytest**

Run: `.venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run a CLI smoke test**

Run the existing CLI package test through pytest, which creates `.rpt` fixtures and validates trace files.

Expected: output package includes existing files plus `trace/*.csv`.
