"""
Voltage sensitivity calculator.

Calculates dLib/dV (ps/mV) using linear regression across voltage corners.
Placeholder for full multi-corner implementation -- currently supports
two-point sensitivity and basic linear regression when multiple corners
are available.
"""

import re
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of a voltage sensitivity calculation."""
    slope_per_v: float        # ps/V slope
    sensitivity_ps_mv: float  # |slope| in ps/mV
    intercept: float
    r_squared: float
    valid_points: int

    @staticmethod
    def failed():
        return None


def calculate_sensitivity(voltages, values):
    """
    Calculate voltage sensitivity via linear regression.

    Args:
        voltages: array-like of voltage values (V)
        values: array-like of corresponding timing/lib values

    Returns:
        SensitivityResult or None
    """
    v = np.array(voltages, dtype=float)
    y = np.array(values, dtype=float)

    # Remove NaNs
    mask = ~(np.isnan(v) | np.isnan(y))
    v, y = v[mask], y[mask]

    if len(v) < 2:
        return None

    try:
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(v, y)
        return SensitivityResult(
            slope_per_v=slope,
            sensitivity_ps_mv=abs(slope) / 1000.0,  # V -> mV
            intercept=intercept,
            r_squared=r_value ** 2,
            valid_points=len(v),
        )
    except Exception as e:
        logger.error(f"Linear regression failed: {e}")
        return None


def two_point_sensitivity(v1, val1, v2, val2):
    """Simple two-point sensitivity: dval/dV."""
    dv = v2 - v1
    if abs(dv) < 1e-12:
        return None
    slope = (val2 - val1) / dv
    return SensitivityResult(
        slope_per_v=slope,
        sensitivity_ps_mv=abs(slope) / 1000.0,
        intercept=val1 - slope * v1,
        r_squared=1.0,
        valid_points=2,
    )


def calculate_voltage_margin(sensitivity_ps_mv, error_ps):
    """
    Calculate voltage margin required to compensate for a timing error.

    Voltage_Margin (mV) = |error| / sensitivity (ps/mV)
    """
    if (sensitivity_ps_mv == 0 or np.isnan(sensitivity_ps_mv)
            or np.isinf(sensitivity_ps_mv)):
        return float('inf')
    return abs(error_ps) / abs(sensitivity_ps_mv)


def extract_voltage_from_corner(corner_name):
    """Extract voltage from corner name, e.g. 'ssgnp_0p450v_m40c' -> 0.450."""
    if not corner_name:
        return None
    match = re.search(r'(\d+)p(\d+)v', corner_name)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None
