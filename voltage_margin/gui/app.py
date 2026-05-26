"""Tkinter Phase 1 workbench for Voltage Margin Tool."""

import logging
import os
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ..core.config_loader import DEFAULT_COLUMN_MAPPING, DEFAULT_POLICY
from ..visualization.plots import plot_pass_rate_bars
from .phase1_backend import (
    OutputTables,
    Phase1RunConfig,
    filter_margins,
    format_margin_trace_detail,
    load_output_package,
    read_source_line,
    run_phase1_pipeline,
    table_columns,
    unique_values,
)

logger = logging.getLogger(__name__)


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
        self.status_var = tk.StringVar(value="Ready. Select input and output folders.")
        self.summary_var = tk.StringVar(value="No run loaded.")
        self.trace_title_var = tk.StringVar(value="Select a margin row to inspect trace.")
        self.path_line_var = tk.StringVar(value="")
        self.current_tab_name = "Pass Rate"
        self.selected_margin_detail = {}
        self.tables: OutputTables | None = None
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
            text="Phase 1 workbench: run analysis, inspect margins, and trace every value to source.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.LabelFrame(self.root, text="Run Configuration", style="Card.TLabelframe",
                                  padding=10)
        controls.pack(fill="x", padx=12, pady=(4, 8))
        self._build_controls(controls)

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
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        for name in ["Pass Rate", "Margins", "Sensitivity", "Warnings", "Trace"]:
            frame = ttk.Frame(self.notebook, style="Panel.TFrame")
            self.notebook.add(frame, text=name)
            self._build_table_tab(name, frame)

        plot_frame = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.notebook.add(plot_frame, text="Plots")
        self._build_plot_tab(plot_frame)

    def _build_table_tab(self, name, frame):
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
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
        ttk.Button(top, text="Open Source", command=self._open_source_file).pack(
            side="right", padx=(6, 0))
        ttk.Button(top, text="Show Raw Row", command=self._show_raw_row).pack(side="right")

        ttk.Label(parent, textvariable=self.path_line_var, style="Muted.TLabel").pack(
            fill="x", pady=(4, 6))
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
            self.summary_var.set(
                f"{result.parameter_groups} groups, {result.total_arcs} arcs")
            self._refresh_all_tables()
            self._refresh_filter_options()
            self._update_plot()
            self.status_var.set(f"Analysis complete. Output: {result.output_dir}")
        except Exception as exc:
            logger.exception("Analysis failed")
            messagebox.showerror("Analysis failed", str(exc))
            self.status_var.set("Analysis failed.")

    def _refresh_all_tables(self):
        if self.tables is None:
            return
        self._populate_table("Pass Rate", self.tables.pass_rate, PASS_RATE_COLUMNS)
        self._apply_margin_filters()
        self._populate_table("Sensitivity", self.tables.sensitivity, SENSITIVITY_COLUMNS)
        self._populate_table("Warnings", self._combined_warnings(), WARNING_COLUMNS)
        self._populate_table("Trace", self.tables.margin_trace, TRACE_COLUMNS)

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
        margins = self.tables.all_margins
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
            self.tables.all_margins,
            source=self.filter_vars["source"].get(),
            analysis_type=self.filter_vars["type"].get(),
            metric=self.filter_vars["metric"].get(),
            corner=self.filter_vars["corner"].get(),
            status=self.filter_vars["status"].get(),
        )
        self._populate_table("Margins", filtered, MARGIN_COLUMNS)

    def _populate_table(self, name, df, preferred_columns):
        tree = self.display_trees[name]
        tree.delete(*tree.get_children())
        columns = table_columns(df, preferred_columns)
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column, command=lambda c=column, n=name: self._sort_table(n, c))
            width = 230 if column in {"arc", "source_file", "source_file_relative"} else 120
            tree.column(column, width=width, minwidth=70, anchor="w")
        if df.empty:
            self.display_dfs[name] = df
            return
        display_df = df.reset_index(drop=True)
        self.display_dfs[name] = display_df
        for idx, row in display_df.iterrows():
            values = [self._format_cell(row.get(column, "")) for column in columns]
            tree.insert("", "end", iid=str(idx), values=values)

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
        if tab_name != "Margins":
            return
        tree = self.display_trees[tab_name]
        selected = tree.selection()
        if not selected:
            return
        row = self.display_dfs[tab_name].iloc[int(selected[0])]
        self._show_margin_detail(row)

    def _show_margin_detail(self, row):
        detail = format_margin_trace_detail(row)
        self.selected_margin_detail = detail
        self.trace_title_var.set(detail["title"] or "Selected margin")
        self.path_line_var.set(detail["path_line"])
        text = "\n".join(
            [
                f"Arc: {detail['arc']}",
                f"Required: {detail['required_formula']}",
                f"Signed: {detail['signed_formula']}",
                f"Sensitivity trace: {detail['sensitivity_trace_id']}",
                f"Low sources: {detail['low_sources']}",
                f"High sources: {detail['high_sources']}",
            ]
        )
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
            raw = read_source_line(source_file, line_number)
        except Exception as exc:
            messagebox.showerror("Could not read source row", str(exc))
            return
        messagebox.showinfo("Raw Source Row", raw or "Line not found.")

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
        if self.tables is None or self.tables.pass_rate.empty:
            return
        plt.close(self.current_fig)
        self.current_fig = plot_pass_rate_bars(self.tables.pass_rate, title="Pass Rate Summary")
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
