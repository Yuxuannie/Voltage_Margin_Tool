"""
Voltage margin analyzer.

Given sensitivity data and pass rate arc results, computes how much
voltage margin (mV) is needed to cover each arc's error, and produces
margin efficiency curves.
"""

import numpy as np
import logging
from typing import List, Dict
from dataclasses import dataclass

from .sensitivity_calculator import calculate_voltage_margin

logger = logging.getLogger(__name__)


@dataclass
class MarginStats:
    """Statistics for margin distribution of one parameter group."""
    corner: str
    type_name: str
    param_name: str
    count: int
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    max_val: float


def compute_arc_margins(arc_results, sensitivity_ps_mv):
    """
    For each arc result, compute the voltage margin required.

    Args:
        arc_results: list of ArcResult from pass_rate_engine
        sensitivity_ps_mv: sensitivity in ps/mV

    Returns:
        list of (arc_name, margin_mV) tuples
    """
    margins = []
    for ar in arc_results:
        margin = calculate_voltage_margin(sensitivity_ps_mv, ar.abs_err)
        margins.append((ar.arc, margin))
    return margins


def margin_efficiency_curve(margins_mv, margin_points=None):
    """
    Calculate fraction of arcs covered at each margin level.

    Args:
        margins_mv: list of margin values in mV (one per arc)
        margin_points: list of margin levels to evaluate (default 0..50 step 5)

    Returns:
        List of (margin_mV, fraction_covered) tuples
    """
    if margin_points is None:
        margin_points = list(range(0, 55, 5))

    arr = np.array([m for m in margins_mv if np.isfinite(m)])
    if len(arr) == 0:
        return [(mp, 0.0) for mp in margin_points]

    total = len(arr)
    curve = []
    for mp in margin_points:
        covered = np.sum(arr <= mp)
        curve.append((mp, covered / total))
    return curve


def suggest_margin_for_target(margins_mv, target_coverage=0.95):
    """
    Suggest the voltage margin needed to achieve a target coverage.

    Returns:
        margin in mV, or inf if not achievable.
    """
    arr = np.array([m for m in margins_mv if np.isfinite(m)])
    if len(arr) == 0:
        return float('inf')
    return float(np.percentile(arr, target_coverage * 100))


def compute_margin_stats(arc_results, sensitivity_ps_mv,
                         corner, type_name, param_name):
    """Compute MarginStats for a parameter group."""
    margins = [calculate_voltage_margin(sensitivity_ps_mv, ar.abs_err)
               for ar in arc_results]
    finite = np.array([m for m in margins if np.isfinite(m)])

    if len(finite) == 0:
        return None

    return MarginStats(
        corner=corner, type_name=type_name, param_name=param_name,
        count=len(finite),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        p90=float(np.percentile(finite, 90)),
        p95=float(np.percentile(finite, 95)),
        p99=float(np.percentile(finite, 99)),
        max_val=float(np.max(finite)),
    )
