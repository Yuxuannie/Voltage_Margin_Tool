"""
Main GUI application for Voltage Margin Analysis Tool.

Uses Tkinter with embedded matplotlib for interactive analysis.
"""

import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

from ..core.data_loader import load_all_data, auto_discover_files
from ..core.pass_rate_engine import (
    run_full_analysis, results_to_dataframe, filter_results_by_waivers,
)
from ..visualization.plots import (
    plot_pass_rate_bars, plot_pass_rate_heatmap,
    plot_error_distribution, plot_margin_efficiency,
)

logger = logging.getLogger(__name__)


class VoltageMarginApp:
    """Main GUI application."""

    def __init__(self, root):
        self.root = root
        self.root.title('Voltage Margin Analysis Tool')
        self.root.geometry('1400x900')
        self.root.minsize(1000, 700)

        # State
        self.data_dir = tk.StringVar()
        self.bundles = {}
        self.analysis_results = []
        self.results_df = pd.DataFrame()
        self.full_results_df = pd.DataFrame()  # unfiltered

        # Waiver toggles
        self.waiver1_var = tk.BooleanVar(value=True)
        self.optimistic_var = tk.BooleanVar(value=True)

        # Plot selector
        self.plot_type_var = tk.StringVar(value='bar_chart')

        # Heatmap PR column selector
        self.heatmap_pr_var = tk.StringVar(value='Base_PR')

        self._build_ui()

    def _build_ui(self):
        """Build the complete UI layout."""
        # --- Top: Control panel ---
        control_frame = ttk.LabelFrame(self.root, text='Controls', padding=8)
        control_frame.pack(fill='x', padx=8, pady=(8, 4))
        self._build_control_panel(control_frame)

        # --- Middle: PanedWindow with table + plot ---
        paned = ttk.PanedWindow(self.root, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=4)

        # Left: results table
        table_frame = ttk.LabelFrame(paned, text='Pass Rate Results', padding=4)
        paned.add(table_frame, weight=1)
        self._build_table(table_frame)

        # Right: plot viewer
        plot_frame = ttk.LabelFrame(paned, text='Plots', padding=4)
        paned.add(plot_frame, weight=2)
        self._build_plot_viewer(plot_frame)

        # --- Bottom: status bar ---
        self.status_var = tk.StringVar(value='Ready. Select a data directory to begin.')
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief='sunken', anchor='w')
        status_bar.pack(fill='x', padx=8, pady=(0, 8))

    def _build_control_panel(self, parent):
        """Input selection, waiver toggles, action buttons."""
        # Row 1: directory selection
        row1 = ttk.Frame(parent)
        row1.pack(fill='x', pady=2)

        ttk.Label(row1, text='Data Directory:').pack(side='left')
        ttk.Entry(row1, textvariable=self.data_dir, width=60).pack(
            side='left', padx=4, fill='x', expand=True)
        ttk.Button(row1, text='Browse...', command=self._browse_dir).pack(side='left')
        ttk.Button(row1, text='Run Analysis', command=self._run_analysis,
                   style='Accent.TButton').pack(side='left', padx=(12, 0))

        # Row 2: waiver toggles + plot selector + export
        row2 = ttk.Frame(parent)
        row2.pack(fill='x', pady=2)

        ttk.Label(row2, text='Waivers:').pack(side='left')
        ttk.Checkbutton(row2, text='CI Enlargement (6%)',
                        variable=self.waiver1_var,
                        command=self._on_waiver_toggle).pack(side='left', padx=4)
        ttk.Checkbutton(row2, text='Optimistic Direction',
                        variable=self.optimistic_var,
                        command=self._on_waiver_toggle).pack(side='left', padx=4)

        ttk.Separator(row2, orient='vertical').pack(side='left', fill='y', padx=8)

        ttk.Label(row2, text='Plot:').pack(side='left')
        plot_combo = ttk.Combobox(row2, textvariable=self.plot_type_var,
                                  values=['bar_chart', 'heatmap',
                                          'error_distribution'],
                                  state='readonly', width=18)
        plot_combo.pack(side='left', padx=4)
        plot_combo.bind('<<ComboboxSelected>>', lambda e: self._update_plot())

        # Heatmap PR column selector
        ttk.Label(row2, text='Heatmap PR:').pack(side='left', padx=(8, 0))
        heatmap_combo = ttk.Combobox(
            row2, textvariable=self.heatmap_pr_var,
            values=['Base_PR', 'PR_with_Waiver1',
                    'PR_Optimistic_Only', 'PR_with_Both_Waivers'],
            state='readonly', width=20)
        heatmap_combo.pack(side='left', padx=4)
        heatmap_combo.bind('<<ComboboxSelected>>', lambda e: self._update_plot())

        ttk.Separator(row2, orient='vertical').pack(side='left', fill='y', padx=8)

        ttk.Button(row2, text='Export CSV', command=self._export_csv).pack(side='left')
        ttk.Button(row2, text='Save Plot', command=self._save_plot).pack(
            side='left', padx=4)

    def _build_table(self, parent):
        """Sortable results table using Treeview."""
        columns = ('Corner', 'Type', 'Parameter', 'Total_Arcs',
                   'Base_PR', 'PR_with_Waiver1',
                   'PR_Optimistic_Only', 'PR_with_Both_Waivers')

        self.tree = ttk.Treeview(parent, columns=columns, show='headings',
                                 selectmode='browse')

        col_widths = {
            'Corner': 130, 'Type': 60, 'Parameter': 100, 'Total_Arcs': 70,
            'Base_PR': 75, 'PR_with_Waiver1': 95,
            'PR_Optimistic_Only': 95, 'PR_with_Both_Waivers': 95,
        }
        for col in columns:
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_table(c))
            self.tree.column(col, width=col_widths.get(col, 80), anchor='center')

        # Scrollbars
        vsb = ttk.Scrollbar(parent, orient='vertical', command=self.tree.yview)
        hsb = ttk.Scrollbar(parent, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Bind row selection to update error distribution plot
        self.tree.bind('<<TreeviewSelect>>', self._on_row_select)

    def _build_plot_viewer(self, parent):
        """Embedded matplotlib canvas."""
        self.plot_frame = parent

        # Create a default empty figure
        self.current_fig, self.current_ax = plt.subplots(figsize=(8, 5))
        self.current_ax.text(0.5, 0.5, 'Run analysis to see plots',
                             ha='center', va='center', fontsize=14, color='gray')

        self.canvas = FigureCanvasTkAgg(self.current_fig, master=parent)
        self.canvas.draw()

        # Toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # ---- Actions ----

    def _browse_dir(self):
        path = filedialog.askdirectory(title='Select directory with .rpt files')
        if path:
            self.data_dir.set(path)

    def _run_analysis(self):
        """Load data and run pass rate analysis."""
        data_dir = self.data_dir.get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror('Error', 'Please select a valid data directory.')
            return

        self.status_var.set('Loading data...')
        self.root.update_idletasks()

        try:
            self.bundles = load_all_data(data_dir)
            if not self.bundles:
                messagebox.showwarning('Warning', 'No .rpt files found in directory.')
                self.status_var.set('No data found.')
                return

            self.status_var.set(
                f'Loaded {len(self.bundles)} data bundles. Running analysis...')
            self.root.update_idletasks()

            self.analysis_results = run_full_analysis(self.bundles)
            self.full_results_df = results_to_dataframe(self.analysis_results)

            self._apply_waiver_filter()
            self._populate_table()
            self._update_plot()

            n = len(self.analysis_results)
            total_arcs = sum(r.total_arcs for r in self.analysis_results)
            self.status_var.set(
                f'Analysis complete: {n} parameter groups, {total_arcs} total arcs.')

        except Exception as e:
            logger.exception('Analysis failed')
            messagebox.showerror('Error', f'Analysis failed:\n{e}')
            self.status_var.set('Analysis failed.')

    def _apply_waiver_filter(self):
        """Apply waiver toggle filter to results."""
        if self.full_results_df.empty:
            self.results_df = self.full_results_df
            return
        self.results_df = filter_results_by_waivers(
            self.full_results_df,
            waiver1_enabled=self.waiver1_var.get(),
            optimistic_enabled=self.optimistic_var.get(),
        )

    def _on_waiver_toggle(self):
        """Handle waiver checkbox change."""
        self._apply_waiver_filter()
        self._populate_table()
        self._update_plot()

    def _populate_table(self):
        """Fill the treeview with current results."""
        self.tree.delete(*self.tree.get_children())

        if self.results_df.empty:
            return

        for _, row in self.results_df.iterrows():
            values = []
            for col in self.tree['columns']:
                val = row.get(col, '')
                if isinstance(val, float):
                    val = f'{val:.1f}'
                values.append(val)
            self.tree.insert('', 'end', values=values)

    def _sort_table(self, col):
        """Sort treeview by column."""
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children()]
        try:
            items.sort(key=lambda t: float(t[0]))
        except ValueError:
            items.sort(key=lambda t: t[0])

        for idx, (_, k) in enumerate(items):
            self.tree.move(k, '', idx)

    def _on_row_select(self, event):
        """When a table row is selected, show error distribution if applicable."""
        if self.plot_type_var.get() != 'error_distribution':
            return
        self._update_plot()

    def _update_plot(self):
        """Redraw the plot based on current selection."""
        plot_type = self.plot_type_var.get()

        # Clear old canvas
        plt.close(self.current_fig)

        if plot_type == 'bar_chart':
            self.current_fig = plot_pass_rate_bars(
                self.full_results_df,
                title='Pass Rate Summary',
                waiver1=self.waiver1_var.get(),
                optimistic=self.optimistic_var.get(),
            )
        elif plot_type == 'heatmap':
            pr_col = self.heatmap_pr_var.get()
            self.current_fig = plot_pass_rate_heatmap(
                self.full_results_df, pr_column=pr_col,
                title=f'Heatmap: {pr_col}',
            )
        elif plot_type == 'error_distribution':
            self.current_fig = self._get_error_dist_plot()
        else:
            self.current_fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Select a plot type', ha='center', va='center')

        # Replace canvas
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()

        self.canvas = FigureCanvasTkAgg(self.current_fig, master=self.plot_frame)
        self.canvas.draw()

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _get_error_dist_plot(self):
        """Get error distribution plot for selected row."""
        selected = self.tree.selection()
        if not selected:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Select a row in the table\nto see error distribution',
                    ha='center', va='center', fontsize=12, color='gray')
            return fig

        item = self.tree.item(selected[0])
        vals = item['values']
        corner, type_name, param = str(vals[0]), str(vals[1]), str(vals[2])

        # Find matching analysis result
        for r in self.analysis_results:
            if (r.corner == corner and r.type_name == type_name
                    and r.param_name == param):
                return plot_error_distribution(
                    r.arc_results, param,
                    title=f'Error Distribution: {corner} / {type_name} / {param}')

        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No arc data found', ha='center', va='center')
        return fig

    def _export_csv(self):
        """Export current results table to CSV."""
        if self.results_df.empty:
            messagebox.showinfo('Info', 'No results to export.')
            return

        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Export Results',
        )
        if path:
            self.results_df.to_csv(path, index=False)
            self.status_var.set(f'Exported to {path}')

    def _save_plot(self):
        """Save current plot to file."""
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')],
            title='Save Plot',
        )
        if path:
            self.current_fig.savefig(path, dpi=150, bbox_inches='tight')
            self.status_var.set(f'Plot saved to {path}')


def launch_gui():
    """Entry point to launch the GUI."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    root = tk.Tk()

    # Try to use a modern theme
    style = ttk.Style(root)
    available_themes = style.theme_names()
    for theme in ['clam', 'alt', 'default']:
        if theme in available_themes:
            style.theme_use(theme)
            break

    app = VoltageMarginApp(root)
    root.mainloop()
