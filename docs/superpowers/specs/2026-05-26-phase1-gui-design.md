# Phase 1 GUI Design

## Purpose

Build a Tkinter Phase 1 workbench for Voltage Margin Tool so users can run the
accepted CLI pipeline, inspect output CSVs, and trace any voltage margin back to
source `.rpt` path/line and formula inputs.

## Audience And Tone

The first audience is engineers reviewing library characterization compare
results. The GUI should feel like a quiet engineering workbench: dense but
organized, readable, and trace-focused. It should avoid decorative UI and
prioritize fast scanning, filtering, and source verification.

## Scope

In scope:

- Upgrade `python run_gui.py` to run Phase 1.
- Let users select input directory, output directory, policy YAML, column map
  YAML, and analysis types.
- Load generated output package CSVs after a run.
- Show tabs for pass rate, margins, sensitivity, warnings, and trace tables.
- Provide source/type/metric/corner/status filters for margin review.
- Show formula and source traceback for the selected margin row.
- Provide buttons to copy `source_file:source_line_number`, open source file,
  show raw source row, open output folder, export current table, and save plots
  where available.

Out of scope:

- Web app.
- Advanced policy editor.
- Saved project database.
- Manager dashboard/report polish.
- Changing the Phase 1 math or trace output schema.

## Architecture

Create a small GUI backend module that exposes testable functions for running
the Phase 1 pipeline, loading output package tables, filtering tables, and
formatting selected margin trace details. The Tkinter app consumes this backend
and focuses on layout and interactions.

The existing `voltage_margin/gui/app.py` remains the entry point behind
`run_gui.py`, but its UI is upgraded to a Phase 1 workbench.

## Layout

Top control band:

- Input directory selector.
- Output directory selector.
- Policy YAML selector.
- Column map YAML selector.
- Type checkboxes: delay, slew, hold.
- Run, Open Output, Export Table, Save Plot.

Main body:

- Left filter panel: source, type, metric, corner, margin status.
- Center/right notebook:
  - Pass Rate
  - Margins
  - Sensitivity
  - Warnings
  - Trace
  - Plots

Bottom/right trace panel:

- Selected margin formula fields.
- Current source `path:line`.
- Sensitivity id and low/high source summaries.
- Buttons for copy/open/show raw row.

## Visual Direction

Use a restrained workbench style:

- Soft off-white app background.
- Dark ink text.
- Muted blue-gray section headers.
- Small accent color for Run and selected/important states.
- Compact row heights and sortable tables.
- Clear panel titles and stable spacing.

## Data Flow

1. User selects inputs and clicks Run.
2. GUI calls backend `run_phase1_pipeline`.
3. Backend writes the same output package as CLI.
4. GUI loads output tables into memory.
5. GUI populates tabs and filter options.
6. User selects a margin row.
7. GUI displays formula/source details and can lazy-load raw source line.

## Testing

Backend unit tests should verify:

- Phase 1 backend writes and loads output tables.
- Margin filtering works by source, type, metric, corner, and status.
- Selected margin rows format trace details with formulas and `path:line`.
- Raw source row lookup reads the expected physical line.

GUI smoke verification should verify:

- `voltage_margin.gui.app` imports without side effects.
- Existing pytest suite passes.
