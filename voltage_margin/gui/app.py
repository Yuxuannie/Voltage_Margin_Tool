"""Tkinter Phase 1 workbench for Voltage Margin Tool."""

import logging
import os
import shlex
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ..core.config_loader import DEFAULT_COLUMN_MAPPING, DEFAULT_POLICY
from .phase1_backend import (
    OutputTables,
    Phase1RunConfig,
    build_margin_audit_rows,
    build_vm_sweep,
    build_vm_target_summary,
    enrich_margin_rows,
    find_vm_observations,
    filter_margins,
    format_margin_trace_detail,
    load_output_package,
    read_source_context,
    read_source_line,
    run_phase1_pipeline,
    summarize_margins,
    table_columns,
    unique_values,
)

logger = logging.getLogger(__name__)


TARGET_COLUMNS = [
    "scope",
    "compare_source",
    "corner",
    "analysis_type",
    "metric",
    "base_pr_pct",
    "required_vm_mv",
    "pass_rate_at_required_vm_pct",
    "fixed_count_at_required_vm",
    "new_fail_count_at_required_vm",
    "best_vm_mv",
    "best_pr_pct",
    "trend",
    "total_count",
    "status",
]

VM_SWEEP_COLUMNS = [
    "scope",
    "compare_source",
    "corner",
    "analysis_type",
    "metric",
    "margin_mv",
    "pass_rate_pct",
    "pass_count",
    "total_count",
    "fixed_count",
    "new_fail_count",
]

OBSERVATION_COLUMNS = [
    "observation_code",
    "scope",
    "compare_source",
    "corner",
    "analysis_type",
    "metric",
    "message",
    "base_pr_pct",
    "best_vm_mv",
    "best_pr_pct",
    "new_fail_count_at_best",
]

PASS_RATE_COLUMNS = [
    "Corner",
    "Type",
    "Parameter",
    "Total_Arcs",
    "Base_PR",
    "PR_with_Waiver1",
    "PR_Optimistic_Only",
    "PR_with_Both_Waivers",
]

MARGIN_COLUMNS = [
    "compare_source",
    "corner",
    "analysis_type",
    "metric",
    "arc",
    "dif_ps",
    "sensitivity_ps_per_mv",
    "required_margin_mv",
    "margin_status",
    "source_file_relative",
    "source_line_number",
    "margin_trace_id",
]

SENSITIVITY_COLUMNS = [
    "compare_source",
    "corner",
    "analysis_type",
    "metric",
    "arc",
    "pair_role",
    "low_voltage_v",
    "high_voltage_v",
    "low_lib_value_ps",
    "high_lib_value_ps",
    "sensitivity_ps_per_mv",
    "sensitivity_trace_id",
]

WARNING_COLUMNS = [
    "warning_source",
    "file_path",
    "corner",
    "analysis_type",
    "metric",
    "arc",
    "warning_code",
    "warning_message",
]

TRACE_COLUMNS = [
    "margin_trace_id",
    "normalized_trace_id",
    "sensitivity_trace_id",
    "source_file_relative",
    "source_line_number",
    "dif_ps",
    "sensitivity_ps_per_mv",
    "required_margin_mv",
    "margin_formula_id",
]


class VoltageMarginApp:
    """Main Phase 1 workbench application."""

    def __init__(self, root):
        self.root = root
        self.root.title("Voltage Margin Tool")
        self.root.geometry("1540x960")
        self.root.minsize(1180, 760)

        self.data_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.policy_path = tk.StringVar(value=str(DEFAULT_POLICY))
        self.column_map_path = tk.StringVar(value=str(DEFAULT_COLUMN_MAPPING))
        self.type_vars = {
            "delay": tk.BooleanVar(value=True),
            "slew": tk.BooleanVar(value=True),
            "hold": tk.BooleanVar(value=True),
        }
        self.filter_vars = {
            "source": tk.StringVar(value="All"),
            "type": tk.StringVar(value="All"),
            "metric": tk.StringVar(value="All"),
            "corner": tk.StringVar(value="All"),
            "status": tk.StringVar(value="All"),
        }
        self.plot_type_var = tk.StringVar(value="Selected PR Curve")
        self.status_var = tk.StringVar(value="Ready. Select input and output folders.")
        self.summary_var = tk.StringVar(value="No run loaded.")
        self.trace_title_var = tk.StringVar(value="Select a margin row to inspect trace.")
        self.path_line_var = tk.StringVar(value="")
        self.current_tab_name = "95% Target"
        self.kpi_vars = {
            "margins": tk.StringVar(value="0"),
            "ok": tk.StringVar(value="0"),
            "review": tk.StringVar(value="0"),
            "trace": tk.StringVar(value="0"),
        }
        self.selected_margin_detail = {}
        self.tables: OutputTables | None = None
        self.margin_rows = pd.DataFrame()
        self.vm_sweep = pd.DataFrame()
        self.target_plan = pd.DataFrame()
        self.observations = pd.DataFrame()
        self.display_frames = {}
        self.display_trees = {}
        self.display_dfs = {}

        self._build_style()
        self._build_ui()

    def _build_style(self):
        self.root.configure(bg="#edf1f4")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Helvetica", 10), background="#edf1f4")
        style.configure("TFrame", background="#edf1f4")
        style.configure("Panel.TFrame", background="#f8fafb")
        style.configure("TLabel", background="#edf1f4", foreground="#17212b")
        style.configure("Muted.TLabel", background="#edf1f4", foreground="#5b6773")
        style.configure("Title.TLabel", background="#edf1f4", foreground="#0f1720",
                        font=("Helvetica", 18, "bold"))
        style.configure("Hero.TLabel", background="#edf1f4", foreground="#16202a",
                        font=("Helvetica", 11))
        style.configure("Kpi.TFrame", background="#f8fafb", relief="solid", borderwidth=1)
        style.configure("KpiValue.TLabel", background="#f8fafb", foreground="#0f1720",
                        font=("Helvetica", 19, "bold"))
        style.configure("KpiLabel.TLabel", background="#f8fafb", foreground="#5b6773",
                        font=("Helvetica", 9))
        style.configure("Link.TLabel", background="#f8fafb", foreground="#0b63ce")
        style.configure("Card.TLabelframe", background="#f8fafb", bordercolor="#cfd7df")
        style.configure("Card.TLabelframe.Label", background="#f8fafb",
                        foreground="#223142", font=("Helvetica", 10, "bold"))
        style.configure("Accent.TButton", background="#1f6feb", foreground="#ffffff",
                        font=("Helvetica", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#185abc")])
        style.configure("Treeview", rowheight=24, background="#ffffff",
                        fieldbackground="#ffffff", foreground="#16202a")
        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"),
                        background="#dfe7ef", foreground="#17212b")
        style.configure("TNotebook", background="#edf1f4", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 7), font=("Helvetica", 10))

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(14, 12, 14, 6))
        header.pack(fill="x")
        ttk.Label(header, text="Voltage Margin Tool", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Traceable Phase 1 workbench for voltage margin, sensitivity, and source-line review.",
            style="Hero.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.LabelFrame(self.root, text="Run Configuration", style="Card.TLabelframe",
                                  padding=10)
        controls.pack(fill="x", padx=12, pady=(4, 8))
        self._build_controls(controls)

        self._build_kpis()

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        filters = ttk.LabelFrame(body, text="Filters", style="Card.TLabelframe", padding=10)
        body.add(filters, weight=0)
        self._build_filters(filters)

        right = ttk.PanedWindow(body, orient="vertical")
        body.add(right, weight=1)

        work = ttk.Frame(right)
        right.add(work, weight=4)
        self._build_notebook(work)

        trace = ttk.LabelFrame(right, text="Selected Margin Trace", style="Card.TLabelframe",
                               padding=10)
        right.add(trace, weight=1)
        self._build_trace_panel(trace)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                               anchor="w", padding=(8, 3))
        status_bar.pack(fill="x", padx=12, pady=(0, 10))

    def _build_kpis(self):
        kpi_bar = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        kpi_bar.pack(fill="x")
        specs = [
            ("Margin Rows", "margins"),
            ("OK", "ok"),
            ("Review", "review"),
            ("Trace Links", "trace"),
        ]
        for idx, (label, key) in enumerate(specs):
            card = ttk.Frame(kpi_bar, style="Kpi.TFrame", padding=(14, 8))
            card.grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 8, 0))
            kpi_bar.columnconfigure(idx, weight=1)
            ttk.Label(card, textvariable=self.kpi_vars[key], style="KpiValue.TLabel").pack(
                anchor="w")
            ttk.Label(card, text=label, style="KpiLabel.TLabel").pack(anchor="w")

    def _build_controls(self, parent):
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(4, weight=1)
        self._path_row(parent, 0, "Input", self.data_dir, self._browse_data_dir)
        self._path_row(parent, 1, "Output", self.output_dir, self._browse_output_dir)
        self._path_row(parent, 2, "Policy", self.policy_path,
                       lambda: self._browse_file(self.policy_path, "Select policy YAML"))
        self._path_row(parent, 3, "Column Map", self.column_map_path,
                       lambda: self._browse_file(self.column_map_path, "Select column map YAML"))

        type_frame = ttk.Frame(parent)
        type_frame.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(8, 0))
        ttk.Label(type_frame, text="Types:").pack(side="left")
        for type_name, var in self.type_vars.items():
            ttk.Checkbutton(type_frame, text=type_name, variable=var).pack(
                side="left", padx=(8, 0))

        ttk.Button(type_frame, text="Run Phase 1", style="Accent.TButton",
                   command=self._run_analysis).pack(side="left", padx=(20, 0))
        ttk.Button(type_frame, text="Open Output", command=self._open_output_folder).pack(
            side="left", padx=(8, 0))
        ttk.Button(type_frame, text="Export Current Table", command=self._export_current_table).pack(
            side="left", padx=(8, 0))
        ttk.Button(type_frame, text="Save Plot", command=self._save_plot).pack(
            side="left", padx=(8, 0))
        ttk.Label(type_frame, text="Plot:").pack(side="left", padx=(18, 4))
        plot_box = ttk.Combobox(
            type_frame,
            textvariable=self.plot_type_var,
            values=[
                "Selected PR Curve",
                "Target VM Ranking",
                "Scope Difference",
                "Observation Summary",
            ],
            state="readonly",
            width=22,
        )
        plot_box.pack(side="left")
        plot_box.bind("<<ComboboxSelected>>", lambda _event: self._update_plot())
        ttk.Label(type_frame, textvariable=self.summary_var, style="Muted.TLabel").pack(
            side="left", padx=(18, 0))

    def _path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, columnspan=3, sticky="ew", padx=6, pady=2)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=4, sticky="e")

    def _build_filters(self, parent):
        filter_specs = [
            ("Source", "source"),
            ("Type", "type"),
            ("Metric", "metric"),
            ("Corner", "corner"),
            ("Status", "status"),
        ]
        self.filter_boxes = {}
        for label, key in filter_specs:
            ttk.Label(parent, text=label).pack(anchor="w", pady=(0, 2))
            box = ttk.Combobox(parent, textvariable=self.filter_vars[key],
                               values=["All"], state="readonly", width=24)
            box.pack(fill="x", pady=(0, 10))
            box.bind("<<ComboboxSelected>>", lambda _event: self._apply_margin_filters())
            self.filter_boxes[key] = box
        ttk.Button(parent, text="Reset Filters", command=self._reset_filters).pack(fill="x")

    def _build_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        for name in [
            "95% Target",
            "Observations",
            "VM Sweep",
            "Margins",
            "Pass Rate",
            "Sensitivity",
            "Warnings",
            "Trace",
        ]:
            frame = ttk.Frame(self.notebook, style="Panel.TFrame")
            self.notebook.add(frame, text=name)
            self._build_table_tab(name, frame)

        plot_frame = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.notebook.add(plot_frame, text="Plots")
        self._build_plot_tab(plot_frame)
        self.notebook.select(0)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_table_tab(self, name, frame):
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.tag_configure("ok", background="#ffffff")
        tree.tag_configure("review", background="#fff7e6")
        tree.tag_configure("missing", background="#f3f6f9")
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree.bind("<<TreeviewSelect>>", lambda _event, tab=name: self._on_table_select(tab))
        self.display_trees[name] = tree
        self.display_frames[name] = frame
        self.display_dfs[name] = pd.DataFrame()

    def _build_plot_tab(self, parent):
        self.current_fig, self.current_ax = plt.subplots(figsize=(8, 5))
        self.current_ax.text(0.5, 0.5, "Run analysis to see pass-rate plot",
                             ha="center", va="center", fontsize=13, color="#66717d")
        self.canvas = FigureCanvasTkAgg(self.current_fig, master=parent)
        self.canvas.draw()
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_trace_panel(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.trace_title_var,
                  font=("Helvetica", 11, "bold")).pack(side="left", anchor="w")
        ttk.Button(top, text="Copy path:line", command=self._copy_path_line).pack(
            side="right", padx=(6, 0))
        ttk.Button(top, text="Copy less command", command=self._copy_less_command).pack(
            side="right", padx=(6, 0))
        ttk.Button(top, text="Open Source", command=self._open_source_file).pack(
            side="right", padx=(6, 0))
        ttk.Button(top, text="Show Source Context", command=self._show_raw_row).pack(
            side="right")

        self.path_line_value = ttk.Label(
            parent,
            textvariable=self.path_line_var,
            style="Link.TLabel",
            cursor="hand2",
        )
        self.path_line_value.pack(fill="x", pady=(4, 0))
        self.path_line_value.bind("<Button-1>", lambda _event: self._open_source_file())
        self.path_line_hint = ttk.Label(
            parent,
            text="Click the path or use Open Source to verify the row in the original report.",
            style="Muted.TLabel",
        )
        self.path_line_hint.pack(fill="x", pady=(0, 6))
        self.trace_text = tk.Text(parent, height=7, wrap="word", bd=0,
                                  bg="#ffffff", fg="#17212b", relief="flat")
        self.trace_text.pack(fill="both", expand=True)
        self.trace_text.configure(state="disabled")

    def _browse_data_dir(self):
        path = filedialog.askdirectory(title="Select directory with .rpt files")
        if path:
            self.data_dir.set(path)
            if not self.output_dir.get().strip():
                self.output_dir.set(str(Path(path) / "voltage_margin_outputs"))

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    def _browse_file(self, variable, title):
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            variable.set(path)

    def _selected_types(self):
        return [name for name, var in self.type_vars.items() if var.get()]

    def _run_analysis(self):
        data_dir = self.data_dir.get().strip()
        output_dir = self.output_dir.get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("Input required", "Please select a valid input directory.")
            return
        if not output_dir:
            output_dir = str(Path(data_dir) / "voltage_margin_outputs")
            self.output_dir.set(output_dir)
        types = self._selected_types()
        if not types:
            messagebox.showerror("Type required", "Select at least one analysis type.")
            return

        self.status_var.set("Running Phase 1 analysis...")
        self.root.update_idletasks()
        try:
            result = run_phase1_pipeline(
                Phase1RunConfig(
                    data_dir=Path(data_dir),
                    output_dir=Path(output_dir),
                    column_map=Path(self.column_map_path.get()),
                    policy=Path(self.policy_path.get()),
                    types=types,
                )
            )
            self.tables = load_output_package(result.output_dir)
            self.margin_rows = enrich_margin_rows(
                self.tables.all_margins, self.tables.sensitivity, self.tables.per_arc)
            self.vm_sweep = build_vm_sweep(self.margin_rows, max_margin_mv=50, step_mv=1)
            self.target_plan = build_vm_target_summary(self.vm_sweep)
            self.observations = find_vm_observations(self.vm_sweep)
            self.summary_var.set(
                f"{result.parameter_groups} groups, {result.total_arcs} arcs")
            self._refresh_all_tables()
            self._refresh_filter_options()
            self._refresh_kpis()
            self._show_margin_workspace()
            self._update_plot()
            self.status_var.set(f"Analysis complete. Output: {result.output_dir}")
        except Exception as exc:
            logger.exception("Analysis failed")
            messagebox.showerror("Analysis failed", str(exc))
            self.status_var.set("Analysis failed.")

    def _refresh_all_tables(self):
        if self.tables is None:
            return
        self._populate_table("95% Target", self.target_plan, TARGET_COLUMNS)
        self._populate_table("Observations", self.observations, OBSERVATION_COLUMNS)
        self._populate_table("VM Sweep", self.vm_sweep, VM_SWEEP_COLUMNS)
        self._populate_table("Pass Rate", self.tables.pass_rate, PASS_RATE_COLUMNS)
        self._apply_margin_filters()
        self._populate_table("Sensitivity", self.tables.sensitivity, SENSITIVITY_COLUMNS)
        self._populate_table("Warnings", self._combined_warnings(), WARNING_COLUMNS)
        self._populate_table("Trace", self.tables.margin_trace, TRACE_COLUMNS)

    def _refresh_kpis(self):
        if self.tables is None:
            return
        summary = summarize_margins(self.margin_rows, self.tables.margin_trace)
        self.kpi_vars["margins"].set(str(summary.total_margins))
        self.kpi_vars["ok"].set(str(summary.ok_margins))
        self.kpi_vars["review"].set(str(summary.needs_review))
        self.kpi_vars["trace"].set(str(summary.trace_rows))

    def _combined_warnings(self):
        if self.tables is None:
            return pd.DataFrame()
        warnings = []
        if not self.tables.normalization_warnings.empty:
            df = self.tables.normalization_warnings.copy()
            df["warning_source"] = "normalization"
            warnings.append(df)
        if not self.tables.sensitivity_warnings.empty:
            df = self.tables.sensitivity_warnings.copy()
            df["warning_source"] = "sensitivity"
            warnings.append(df)
        return pd.concat(warnings, ignore_index=True, sort=False) if warnings else pd.DataFrame()

    def _refresh_filter_options(self):
        if self.tables is None:
            return
        margins = self.margin_rows
        mapping = {
            "source": "compare_source",
            "type": "analysis_type",
            "metric": "metric",
            "corner": "corner",
            "status": "margin_status",
        }
        for key, column in mapping.items():
            values = ["All"] + unique_values(margins, column)
            self.filter_boxes[key]["values"] = values
            if self.filter_vars[key].get() not in values:
                self.filter_vars[key].set("All")

    def _reset_filters(self):
        for var in self.filter_vars.values():
            var.set("All")
        self._apply_margin_filters()

    def _apply_margin_filters(self):
        if self.tables is None:
            return
        filtered = filter_margins(
            self.margin_rows,
            source=self.filter_vars["source"].get(),
            analysis_type=self.filter_vars["type"].get(),
            metric=self.filter_vars["metric"].get(),
            corner=self.filter_vars["corner"].get(),
            status=self.filter_vars["status"].get(),
        )
        self._populate_table("Margins", build_margin_audit_rows(filtered), MARGIN_COLUMNS)
        filtered_sweep = self._filtered_sweep_from_filters()
        self._populate_table(
            "95% Target",
            build_vm_target_summary(filtered_sweep),
            TARGET_COLUMNS,
        )
        self._populate_table("VM Sweep", filtered_sweep, VM_SWEEP_COLUMNS)
        self._populate_table(
            "Observations",
            self._filter_group_table(self.observations),
            OBSERVATION_COLUMNS,
        )
        if self.current_tab_name == "Margins":
            self._select_first_margin()
        if self.current_tab_name in {"95% Target", "Observations", "VM Sweep", "Plots"}:
            self._update_plot()

    def _filtered_sweep_from_filters(self):
        return self._filter_group_table(self.vm_sweep)

    def _filter_group_table(self, df):
        if df.empty:
            return df
        filtered = df.copy()
        mapping = {
            "source": "compare_source",
            "type": "analysis_type",
            "metric": "metric",
            "corner": "corner",
        }
        for key, column in mapping.items():
            value = self.filter_vars[key].get()
            if value and value != "All" and column in filtered.columns:
                filtered = filtered[filtered[column].astype(str) == str(value)]
        return filtered

    def _populate_table(self, name, df, preferred_columns):
        tree = self.display_trees[name]
        tree.delete(*tree.get_children())
        columns = table_columns(df, preferred_columns)
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column, command=lambda c=column, n=name: self._sort_table(n, c))
            width = 280 if column in {"arc", "worst_arc"} else 180 if "source" in column else 120
            tree.column(column, width=width, minwidth=70, anchor="w")
        if df.empty:
            self.display_dfs[name] = df
            return
        display_df = df.reset_index(drop=True)
        self.display_dfs[name] = display_df
        for idx, row in display_df.iterrows():
            values = [self._format_cell(row.get(column, "")) for column in columns]
            tags = self._row_tags(name, row)
            tree.insert("", "end", iid=str(idx), values=values, tags=tags)

    def _row_tags(self, name, row):
        if name not in {"Margins", "95% Target"}:
            return ()
        status = str(row.get("margin_status", "") or row.get("status", "") or "ok")
        if status == "ok":
            return ("ok",)
        if status:
            return ("review",)
        return ("missing",)

    def _format_cell(self, value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _sort_table(self, name, column):
        df = self.display_dfs.get(name, pd.DataFrame())
        if df.empty or column not in df.columns:
            return
        sorted_df = df.sort_values(column, kind="mergesort")
        preferred = list(self.display_trees[name]["columns"])
        self._populate_table(name, sorted_df, preferred)

    def _on_tab_changed(self, _event):
        selected = self.notebook.select()
        self.current_tab_name = self.notebook.tab(selected, "text")

    def _on_table_select(self, tab_name):
        tree = self.display_trees[tab_name]
        selected = tree.selection()
        if not selected:
            return
        row = self.display_dfs[tab_name].iloc[int(selected[0])]
        if tab_name == "Margins":
            self._show_margin_detail(row)
        elif tab_name == "95% Target":
            self._show_target_detail(row)
        elif tab_name == "Observations":
            self._show_observation_detail(row)

    def _show_margin_workspace(self):
        self.notebook.select(0)
        self.current_tab_name = "95% Target"
        self._select_first_target()

    def _select_first_target(self):
        tree = self.display_trees.get("95% Target")
        if tree is None:
            return
        children = tree.get_children()
        if not children:
            self.trace_title_var.set("No 95% target margin rows available.")
            self.path_line_var.set("")
            self.selected_margin_detail = {}
            self._write_trace_text("No target margin rows match the current filters.")
            return
        first = children[0]
        tree.selection_set(first)
        tree.focus(first)
        tree.see(first)
        row = self.display_dfs["95% Target"].iloc[int(first)]
        self._show_target_detail(row)

    def _select_first_margin(self):
        tree = self.display_trees.get("Margins")
        if tree is None:
            return
        children = tree.get_children()
        if not children:
            self.trace_title_var.set("No margin rows available for current filters.")
            self.path_line_var.set("")
            self.selected_margin_detail = {}
            self._write_trace_text("No voltage margin rows match the current filters.")
            return
        first = children[0]
        tree.selection_set(first)
        tree.focus(first)
        tree.see(first)
        row = self.display_dfs["Margins"].iloc[int(first)]
        self._show_margin_detail(row)

    def _show_margin_detail(self, row):
        detail = format_margin_trace_detail(row)
        self.selected_margin_detail = detail
        self.trace_title_var.set(detail["title"] or "Selected margin")
        self.path_line_var.set(detail["path_line"])
        context = ""
        if detail["source_file"] and detail["source_line_number"]:
            try:
                context = read_source_context(
                    detail["source_file"], detail["source_line_number"], radius=2)
            except Exception:
                context = ""
        text = "\n".join(
            [
                f"Arc: {detail['arc']}",
                f"Required: {detail['required_formula']}",
                f"Signed: {detail['signed_formula']}",
                f"Sensitivity trace: {detail['sensitivity_trace_id']}",
                f"Low sources: {detail['low_sources']}",
                f"High sources: {detail['high_sources']}",
                "",
                "Source context:",
                context,
            ]
        )
        self._write_trace_text(text)

    def _show_target_detail(self, row):
        self.trace_title_var.set(
            f"VM target: {row.get('corner')} / {row.get('analysis_type')} / {row.get('metric')}")
        self.path_line_var.set("")
        self.selected_margin_detail = {}
        required_vm = row.get("required_vm_mv")
        required_text = "not reached" if pd.isna(required_vm) else f"{required_vm:.0f} mV"
        self._write_trace_text(
            "\n".join(
                [
                    f"Scope: {row.get('scope')}",
                    f"Base PR: {row.get('base_pr_pct'):.3g}%",
                    f"Required VM to reach 95% PR: {required_text}",
                    f"PR at required VM: {row.get('pass_rate_at_required_vm_pct'):.3g}%",
                    f"Best VM in sweep: {row.get('best_vm_mv'):.0f} mV",
                    f"Best PR in sweep: {row.get('best_pr_pct'):.3g}%",
                    f"Trend: {row.get('trend')}",
                    f"Fixed outliers at required VM: {row.get('fixed_count_at_required_vm')}",
                    f"New fails at required VM: {row.get('new_fail_count_at_required_vm')}",
                    "",
                    "Scope meaning:",
                    "all_rows applies VM to every row, so pessimistic rows can become new fails.",
                    "outliers_only applies VM only to base failing rows and holds base passing rows fixed.",
                ]
            )
        )

    def _show_observation_detail(self, row):
        self.trace_title_var.set(
            f"Observation: {row.get('corner')} / {row.get('analysis_type')} / {row.get('metric')}"
        )
        self.path_line_var.set("")
        self.selected_margin_detail = {}
        self._write_trace_text(
            "\n".join(
                [
                    f"{row.get('observation_code')}: {row.get('message')}",
                    f"Scope: {row.get('scope')}",
                    f"Base PR: {row.get('base_pr_pct'):.3g}%",
                    f"Best VM: {row.get('best_vm_mv'):.0f} mV",
                    f"Best PR: {row.get('best_pr_pct'):.3g}%",
                    f"New fails at best VM: {row.get('new_fail_count_at_best')}",
                ]
            )
        )

    def _write_trace_text(self, text):
        self.trace_text.configure(state="normal")
        self.trace_text.delete("1.0", "end")
        self.trace_text.insert("1.0", text)
        self.trace_text.configure(state="disabled")

    def _copy_path_line(self):
        path_line = self.selected_margin_detail.get("path_line", "")
        if path_line:
            self.root.clipboard_clear()
            self.root.clipboard_append(path_line)
            self.status_var.set(f"Copied {path_line}")

    def _copy_less_command(self):
        source_file = self.selected_margin_detail.get("source_file", "")
        line_number = self.selected_margin_detail.get("source_line_number", "")
        if source_file and line_number:
            command = f"less +{int(float(line_number))} {shlex.quote(source_file)}"
            self.root.clipboard_clear()
            self.root.clipboard_append(command)
            self.status_var.set(f"Copied {command}")

    def _open_source_file(self):
        source_file = self.selected_margin_detail.get("source_file", "")
        if source_file and Path(source_file).exists():
            webbrowser.open(Path(source_file).resolve().as_uri())
        elif source_file:
            messagebox.showwarning("File not found", source_file)

    def _show_raw_row(self):
        source_file = self.selected_margin_detail.get("source_file", "")
        line_number = self.selected_margin_detail.get("source_line_number", "")
        if not source_file or not line_number:
            return
        try:
            raw = read_source_context(source_file, line_number, radius=3)
        except Exception as exc:
            messagebox.showerror("Could not read source row", str(exc))
            return
        messagebox.showinfo("Source Context", raw or "Line not found.")

    def _open_output_folder(self):
        output_dir = self.output_dir.get().strip()
        if output_dir and Path(output_dir).exists():
            webbrowser.open(Path(output_dir).resolve().as_uri())

    def _export_current_table(self):
        df = self.display_dfs.get(self.current_tab_name, pd.DataFrame())
        if df.empty:
            messagebox.showinfo("No data", "Current tab has no rows to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Current Table",
        )
        if path:
            df.to_csv(path, index=False)
            self.status_var.set(f"Exported {self.current_tab_name} to {path}")

    def _update_plot(self):
        if self.tables is None:
            return
        plt.close(self.current_fig)
        target_df = self.display_dfs.get("95% Target", self.target_plan)
        sweep_df = self.display_dfs.get("VM Sweep", self.vm_sweep)
        observations_df = self.display_dfs.get("Observations", self.observations)
        self.current_fig = self._build_selected_plot(target_df, sweep_df, observations_df)
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()
        parent = self.notebook.nametowidget(self.notebook.tabs()[-1])
        self.canvas = FigureCanvasTkAgg(self.current_fig, master=parent)
        self.canvas.draw()
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_selected_plot(self, plan, sweep, observations):
        plot_type = self.plot_type_var.get()
        if plot_type == "Target VM Ranking":
            return self._plot_vm_target(plan)
        if plot_type == "Scope Difference":
            return self._plot_scope_difference(plan)
        if plot_type == "Observation Summary":
            return self._plot_observation_summary(observations)
        return self._plot_selected_pr_curve(plan, sweep)

    def _plot_vm_target(self, plan):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#f8fafb")
        ax.set_facecolor("#ffffff")
        if plan is None or plan.empty:
            ax.text(0.5, 0.5, "Run analysis to see VM sweep pass-rate simulation",
                    ha="center", va="center", fontsize=13, color="#66717d")
            ax.axis("off")
            return fig
        plot_df = plan[plan["scope"].astype(str) == "all_rows"].copy()
        if plot_df.empty:
            plot_df = plan.copy()
        plot_df = plot_df.sort_values(
            ["required_vm_mv", "base_pr_pct"],
            ascending=[False, True],
            na_position="first",
        ).head(24)
        plot_df = plot_df.sort_values("required_vm_mv", ascending=True, na_position="first")
        labels = plot_df.apply(
            lambda r: f"{r['corner']} / {r['analysis_type']} / {r['metric']}",
            axis=1,
        )
        values = pd.to_numeric(plot_df["required_vm_mv"], errors="coerce").fillna(50)
        colors = ["#c2410c" if status == "not_reached" else "#0b63ce"
                  for status in plot_df["status"]]
        bars = ax.barh(labels, values, color=colors, alpha=0.88)
        max_value = max(float(values.max()), 1.0)
        for bar, value, status in zip(bars, values, plot_df["status"]):
            label = ">50 mV" if status == "not_reached" else f"{value:.0f} mV"
            ax.text(
                bar.get_width() + max(max_value * 0.01, 0.2),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=8,
                color="#17212b",
            )
        ax.set_xlabel("VM required for pass rate >=95% (mV)")
        ax.set_title("All Rows VM Sweep: Minimum VM to Reach 95% Pass Rate")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        return fig

    def _selected_target_key(self, plan):
        tree = self.display_trees.get("95% Target")
        if tree is not None and tree.selection():
            row = self.display_dfs["95% Target"].iloc[int(tree.selection()[0])]
        elif plan is not None and not plan.empty:
            row = plan.iloc[0]
        else:
            return None
        return {
            "compare_source": row.get("compare_source"),
            "corner": row.get("corner"),
            "analysis_type": row.get("analysis_type"),
            "metric": row.get("metric"),
        }

    def _plot_selected_pr_curve(self, plan, sweep):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#f8fafb")
        ax.set_facecolor("#ffffff")
        key = self._selected_target_key(plan)
        if key is None or sweep is None or sweep.empty:
            ax.text(0.5, 0.5, "Select a 95% Target row to plot VM sweep",
                    ha="center", va="center", fontsize=13, color="#66717d")
            ax.axis("off")
            return fig
        df = sweep.copy()
        for column, value in key.items():
            df = df[df[column].astype(str) == str(value)]
        if df.empty:
            ax.text(0.5, 0.5, "No sweep rows for selected target",
                    ha="center", va="center", fontsize=13, color="#66717d")
            ax.axis("off")
            return fig
        colors = {"all_rows": "#c2410c", "outliers_only": "#0b63ce"}
        for scope, group in df.groupby("scope"):
            group = group.sort_values("margin_mv")
            ax.plot(
                group["margin_mv"],
                group["pass_rate_pct"],
                marker="o",
                linewidth=2,
                markersize=3,
                label=scope,
                color=colors.get(scope, None),
            )
        ax.axhline(95.0, color="#166534", linestyle="--", linewidth=1.2, label="95% target")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Applied voltage margin (mV)")
        ax.set_ylabel("Pass rate (%)")
        ax.set_title(
            f"PR vs VM: {key['corner']} / {key['analysis_type']} / {key['metric']}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_scope_difference(self, plan):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#f8fafb")
        ax.set_facecolor("#ffffff")
        if plan is None or plan.empty:
            ax.text(0.5, 0.5, "No target data", ha="center", va="center")
            ax.axis("off")
            return fig
        pivot = plan.pivot_table(
            index=["compare_source", "corner", "analysis_type", "metric"],
            columns="scope",
            values="best_pr_pct",
            aggfunc="first",
        ).reset_index()
        if not {"all_rows", "outliers_only"}.issubset(pivot.columns):
            ax.text(0.5, 0.5, "Need both scopes for difference plot", ha="center", va="center")
            ax.axis("off")
            return fig
        pivot["delta_pct"] = pivot["outliers_only"] - pivot["all_rows"]
        plot_df = pivot.sort_values("delta_pct", ascending=True).tail(24)
        labels = plot_df.apply(
            lambda r: f"{r['corner']} / {r['analysis_type']} / {r['metric']}",
            axis=1,
        )
        bars = ax.barh(labels, plot_df["delta_pct"], color="#7c3aed", alpha=0.85)
        for bar, value in zip(bars, plot_df["delta_pct"]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}pp", va="center", fontsize=8)
        ax.set_xlabel("Best PR difference: outliers_only - all_rows (percentage points)")
        ax.set_title("Where outliers-only is much more optimistic than all-rows")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        return fig

    def _plot_observation_summary(self, observations):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f8fafb")
        ax.set_facecolor("#ffffff")
        if observations is None or observations.empty:
            ax.text(0.5, 0.5, "No observations under current filters",
                    ha="center", va="center", fontsize=13, color="#66717d")
            ax.axis("off")
            return fig
        counts = observations["observation_code"].value_counts()
        bars = ax.bar(counts.index, counts.values, color=["#c2410c", "#0b63ce", "#7c3aed"])
        for bar, value in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(value), ha="center", va="bottom")
        ax.set_ylabel("Groups")
        ax.set_title("Automatic VM Sweep Observations")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        return fig

    def _save_plot(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Save Plot",
        )
        if path:
            self.current_fig.savefig(path, dpi=150, bbox_inches="tight")
            self.status_var.set(f"Plot saved to {path}")


def launch_gui():
    """Entry point to launch the GUI."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    root = tk.Tk()
    VoltageMarginApp(root)
    root.mainloop()
