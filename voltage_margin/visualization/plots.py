"""
Visualization module for voltage margin analysis.

Provides:
  - Pass rate bar charts (4-bar grouped by waiver mode)
  - Heatmaps (corner x parameter)
  - Margin efficiency curves
  - Error distribution histograms
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; GUI will embed via FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional
from pathlib import Path


# Consistent color scheme
COLORS = {
    'Base_PR': '#2196F3',
    'PR_with_Waiver1': '#FF9800',
    'PR_Optimistic_Only': '#4CAF50',
    'PR_with_Both_Waivers': '#9C27B0',
}

PR_LABELS = {
    'Base_PR': 'Base PR',
    'PR_with_Waiver1': 'PR + CI Waiver',
    'PR_Optimistic_Only': 'PR Optimistic',
    'PR_with_Both_Waivers': 'PR + Both',
}


def plot_pass_rate_bars(results_df, title='Pass Rate Summary',
                        waiver1=True, optimistic=True,
                        figsize=(14, 6)):
    """
    Grouped bar chart of pass rates.

    Args:
        results_df: DataFrame from results_to_dataframe()
            Columns: Corner, Type, Parameter, Total_Arcs, Base_PR, ...
        title: Plot title
        waiver1: Show waiver1 bars
        optimistic: Show optimistic bars

    Returns:
        matplotlib Figure
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        return fig

    # Build list of PR columns to show
    pr_cols = ['Base_PR']
    if waiver1:
        pr_cols.append('PR_with_Waiver1')
    if optimistic:
        pr_cols.append('PR_Optimistic_Only')
    if waiver1 and optimistic:
        pr_cols.append('PR_with_Both_Waivers')
    pr_cols = [c for c in pr_cols if c in results_df.columns]

    # Create x-axis labels
    labels = results_df.apply(
        lambda r: f"{r['Corner']}\n{r['Type']}/{r['Parameter']}", axis=1)

    x = np.arange(len(labels))
    n_bars = len(pr_cols)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=figsize)

    for i, col in enumerate(pr_cols):
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, results_df[col], width,
                      label=PR_LABELS.get(col, col),
                      color=COLORS.get(col, f'C{i}'),
                      edgecolor='white', linewidth=0.5)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Pass Rate (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    return fig


def plot_pass_rate_heatmap(results_df, pr_column='Base_PR',
                            title='Pass Rate Heatmap', figsize=(12, 6)):
    """
    Heatmap of pass rates: rows = Corner, columns = Type/Parameter.

    Returns:
        matplotlib Figure
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        return fig

    if pr_column not in results_df.columns:
        pr_column = 'Base_PR'

    # Pivot: rows=Corner, cols=Type+Param
    df = results_df.copy()
    df['TypeParam'] = df['Type'] + ' / ' + df['Parameter']
    pivot = df.pivot_table(values=pr_column, index='Corner',
                           columns='TypeParam', aggfunc='first')

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label='Pass Rate (%)', shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_margin_efficiency(margin_curve, title='Margin Efficiency',
                           figsize=(8, 5)):
    """
    Line plot of coverage fraction vs voltage margin.

    Args:
        margin_curve: list of (margin_mV, fraction) tuples

    Returns:
        matplotlib Figure
    """
    if not margin_curve:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    margins = [m for m, _ in margin_curve]
    fracs = [f * 100 for _, f in margin_curve]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(margins, fracs, 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax.fill_between(margins, fracs, alpha=0.15, color='#2196F3')

    ax.set_xlabel('Voltage Margin (mV)')
    ax.set_ylabel('Arc Coverage (%)')
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.axhline(y=95, color='red', linestyle='--', linewidth=1, alpha=0.6,
               label='95% target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_distribution(arc_results, param_name,
                            title=None, figsize=(8, 5)):
    """
    Histogram of absolute errors, colored by optimistic vs pessimistic.

    Args:
        arc_results: list of ArcResult
        param_name: parameter name for labeling

    Returns:
        matplotlib Figure
    """
    if not arc_results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    opt_errors = [ar.abs_err for ar in arc_results
                  if ar.error_direction == 'optimistic']
    pes_errors = [ar.abs_err for ar in arc_results
                  if ar.error_direction == 'pessimistic']

    fig, ax = plt.subplots(figsize=figsize)

    bins = 40
    if opt_errors:
        ax.hist(opt_errors, bins=bins, alpha=0.6, color='#4CAF50',
                label=f'Optimistic ({len(opt_errors)})', edgecolor='white')
    if pes_errors:
        ax.hist(pes_errors, bins=bins, alpha=0.6, color='#F44336',
                label=f'Pessimistic ({len(pes_errors)})', edgecolor='white')

    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Count')
    ax.set_title(title or f'Error Distribution: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path):
    """Save figure to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
