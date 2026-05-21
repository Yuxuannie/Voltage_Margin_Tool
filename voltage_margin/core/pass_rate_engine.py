"""
Unified pass rate engine supporting both Sigma and Moments analysis.

Implements all four pass rate modes (A/B/C/D):
  - Base pass rate (Check1: error-based, Check2: CI bounds)
  - With Waiver 1 (CI enlargement +/- 6%)
  - Optimistic only (lib < mc direction filter)
  - With both waivers

Waivers are independently toggleable.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .thresholds import (
    get_rel_threshold, get_abs_threshold,
    CI_ENLARGEMENT_FACTOR, MOMENTS_CI_ESTIMATION_FRACTION,
    SIGMA_PARAMS, MOMENTS_PARAMS, get_sigma_params_for_type,
)
from .data_loader import DataBundle, resolve_columns

logger = logging.getLogger(__name__)


@dataclass
class ArcResult:
    """Result of pass/fail check for a single arc + parameter."""
    arc: str
    param_name: str
    mc_value: float
    lib_value: float
    abs_err: float
    rel_err: float
    mc_ci_lb: Optional[float]
    mc_ci_ub: Optional[float]
    rel_pin_slew: float

    # Check results
    rel_pass: bool = False
    abs_pass: bool = False
    ci_bounds_pass: bool = False
    base_pass: bool = False
    pass_reason: str = 'fail'

    # Waiver results
    waiver1_ci_enlarged: bool = False
    error_direction: str = 'unknown'  # 'optimistic' or 'pessimistic'

    @property
    def pass_with_waiver1(self):
        return self.base_pass or self.waiver1_ci_enlarged

    @property
    def is_optimistic(self):
        return self.error_direction == 'optimistic'

    @property
    def pass_optimistic_only(self):
        """Pass if base_pass, or if the error is in the optimistic direction."""
        return self.base_pass or self.is_optimistic

    @property
    def pass_with_both_waivers(self):
        """Pass if optimistic, or if CI-enlarged covers the error."""
        return self.base_pass or self.is_optimistic or self.waiver1_ci_enlarged


@dataclass
class PassRateSummary:
    """Aggregated pass rates for one (corner, type, param) group."""
    corner: str
    type_name: str
    param_name: str
    total_arcs: int = 0
    base_pass_count: int = 0
    waiver1_pass_count: int = 0
    optimistic_pass_count: int = 0
    both_waivers_pass_count: int = 0
    arc_results: List[ArcResult] = field(default_factory=list)

    @property
    def base_pr(self):
        return self.base_pass_count / self.total_arcs if self.total_arcs else 0

    @property
    def pr_with_waiver1(self):
        return self.waiver1_pass_count / self.total_arcs if self.total_arcs else 0

    @property
    def pr_optimistic(self):
        return self.optimistic_pass_count / self.total_arcs if self.total_arcs else 0

    @property
    def pr_with_both(self):
        return self.both_waivers_pass_count / self.total_arcs if self.total_arcs else 0

    def as_dict(self):
        return {
            'Corner': self.corner,
            'Type': self.type_name,
            'Parameter': self.param_name,
            'Total_Arcs': self.total_arcs,
            'Base_PR': self.base_pr,
            'PR_with_Waiver1': self.pr_with_waiver1,
            'PR_Optimistic_Only': self.pr_optimistic,
            'PR_with_Both_Waivers': self.pr_with_both,
        }


def check_single_arc(row, type_name, param_name, mc_prefix, lib_prefix,
                     col_map):
    """
    Run all pass/fail checks for one arc and one parameter.

    Args:
        row: pandas Series (one row of the DataFrame)
        type_name: 'delay', 'slew', or 'hold'
        param_name: e.g. 'Early_Sigma', 'Std'
        mc_prefix: 'MC'
        lib_prefix: vendor prefix (e.g. 'CDNS_Lib')
        col_map: dict from resolve_columns()

    Returns:
        ArcResult or None if data is missing.
    """
    try:
        arc = row.get('Arc', '')
        rel_pin_slew = row.get('rel_pin_slew', 0.0)

        mc_val = row[col_map['mc_val']] if col_map['mc_val'] else None
        lib_val = row[col_map['lib_val']] if col_map['lib_val'] else None

        if mc_val is None or lib_val is None:
            return None
        if pd.isna(mc_val) or pd.isna(lib_val):
            return None

        # Errors: use pre-calculated if available, else compute
        if col_map['abs_err'] and not pd.isna(row.get(col_map['abs_err'])):
            abs_err = row[col_map['abs_err']]
        else:
            abs_err = lib_val - mc_val

        if col_map['rel_err'] and not pd.isna(row.get(col_map['rel_err'])):
            rel_err = row[col_map['rel_err']]
        else:
            rel_err = ((lib_val - mc_val) / abs(mc_val)) if mc_val != 0 else 0.0

        # CI bounds: real if available, estimated for moments
        if col_map['mc_ci_lb'] and col_map['mc_ci_ub']:
            mc_ci_lb = row[col_map['mc_ci_lb']]
            mc_ci_ub = row[col_map['mc_ci_ub']]
        else:
            # Estimate CI for moments data
            mc_abs = abs(mc_val)
            frac = MOMENTS_CI_ESTIMATION_FRACTION
            mc_ci_lb = mc_val - mc_abs * frac
            mc_ci_ub = mc_val + mc_abs * frac

        # --- Check 1: Error-based pass ---
        rel_threshold = get_rel_threshold(type_name, param_name)
        abs_threshold = get_abs_threshold(type_name, param_name, rel_pin_slew)

        rel_pass = abs(rel_err) <= rel_threshold if rel_threshold is not None else False
        abs_pass = abs(abs_err) <= abs_threshold if abs_threshold is not None else False
        error_based_pass = rel_pass or abs_pass

        # --- Check 2: CI bounds pass ---
        ci_lb = min(mc_ci_lb, mc_ci_ub)
        ci_ub = max(mc_ci_lb, mc_ci_ub)
        ci_bounds_pass = (ci_lb <= lib_val <= ci_ub)

        # --- Base pass ---
        base_pass = error_based_pass or ci_bounds_pass

        if base_pass:
            if rel_pass and abs_pass:
                pass_reason = 'both'
            elif rel_pass:
                pass_reason = 'rel_pass'
            elif abs_pass:
                pass_reason = 'abs_pass'
            elif ci_bounds_pass:
                pass_reason = 'ci_bounds'
            else:
                pass_reason = 'unknown'
        else:
            pass_reason = 'fail'

        # --- Waiver 1: CI enlargement ---
        ci_width = abs(ci_ub - ci_lb)
        enlarge = ci_width * CI_ENLARGEMENT_FACTOR
        waiver1 = (ci_lb - enlarge) <= lib_val <= (ci_ub + enlarge)

        # --- Waiver 2: Error direction ---
        # Phase 1 global rule: Dif = Lib - MC; Dif < 0 is optimistic risk.
        direction = 'optimistic' if abs_err < 0 else 'pessimistic'

        return ArcResult(
            arc=arc, param_name=param_name,
            mc_value=mc_val, lib_value=lib_val,
            abs_err=abs_err, rel_err=rel_err,
            mc_ci_lb=mc_ci_lb, mc_ci_ub=mc_ci_ub,
            rel_pin_slew=rel_pin_slew,
            rel_pass=rel_pass, abs_pass=abs_pass,
            ci_bounds_pass=ci_bounds_pass, base_pass=base_pass,
            pass_reason=pass_reason,
            waiver1_ci_enlarged=waiver1,
            error_direction=direction,
        )

    except Exception as e:
        logger.debug(f"Error checking arc for {param_name}: {e}")
        return None


def analyze_parameter(df, type_name, param_name, corner,
                      mc_prefix='MC', lib_prefix='CDNS_Lib'):
    """
    Analyze pass rates for one parameter across all arcs in a DataFrame.

    Returns:
        PassRateSummary
    """
    col_map = resolve_columns(df, mc_prefix, lib_prefix, param_name)

    if not col_map['mc_val'] or not col_map['lib_val']:
        logger.warning(f"Missing columns for {param_name} in {corner}/{type_name}")
        return None

    summary = PassRateSummary(corner=corner, type_name=type_name,
                              param_name=param_name)

    for _, row in df.iterrows():
        result = check_single_arc(row, type_name, param_name,
                                  mc_prefix, lib_prefix, col_map)
        if result is None:
            continue

        summary.total_arcs += 1
        summary.arc_results.append(result)

        if result.base_pass:
            summary.base_pass_count += 1
        if result.pass_with_waiver1:
            summary.waiver1_pass_count += 1
        if result.pass_optimistic_only:
            summary.optimistic_pass_count += 1
        if result.pass_with_both_waivers:
            summary.both_waivers_pass_count += 1

    return summary


def analyze_bundle(bundle):
    """
    Run full pass rate analysis on a DataBundle.

    Returns:
        List[PassRateSummary] for all available parameters.
    """
    results = []
    mc_prefix = 'MC'
    lib_prefix = bundle.vendor_prefix or 'CDNS_Lib'

    # Sigma parameters (from sigma_df)
    if bundle.has_sigma():
        sigma_params = get_sigma_params_for_type(bundle.type_name)
        for param in sigma_params:
            summary = analyze_parameter(
                bundle.sigma_df, bundle.type_name, param,
                bundle.corner, mc_prefix, lib_prefix)
            if summary and summary.total_arcs > 0:
                results.append(summary)

    # Moments parameters (from moments_df)
    if bundle.has_moments():
        lib_prefix_moments = 'Lib'  # Moments files use 'Lib' prefix
        for param in MOMENTS_PARAMS:
            summary = analyze_parameter(
                bundle.moments_df, bundle.type_name, param,
                bundle.corner, mc_prefix, lib_prefix_moments)
            if summary and summary.total_arcs > 0:
                results.append(summary)

    return results


def run_full_analysis(bundles):
    """
    Run pass rate analysis across all loaded data bundles.

    Args:
        bundles: Dict[(corner, type)] -> DataBundle

    Returns:
        List[PassRateSummary]
    """
    all_results = []
    for key, bundle in bundles.items():
        logger.info(f"Analyzing {key[0]} / {key[1]}...")
        summaries = analyze_bundle(bundle)
        all_results.extend(summaries)
    return all_results


def results_to_dataframe(results):
    """Convert list of PassRateSummary to a summary DataFrame."""
    rows = [r.as_dict() for r in results]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Format pass rates as percentages with 1 decimal
    for col in ['Base_PR', 'PR_with_Waiver1', 'PR_Optimistic_Only', 'PR_with_Both_Waivers']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(x * 100, 1))
    return df


def filter_results_by_waivers(results_df, waiver1_enabled=True,
                               optimistic_enabled=True):
    """
    Return a view of the results DataFrame based on waiver toggles.

    Always includes Base_PR. Adds waiver columns based on toggles.
    """
    cols = ['Corner', 'Type', 'Parameter', 'Total_Arcs', 'Base_PR']
    if waiver1_enabled:
        cols.append('PR_with_Waiver1')
    if optimistic_enabled:
        cols.append('PR_Optimistic_Only')
    if waiver1_enabled and optimistic_enabled:
        cols.append('PR_with_Both_Waivers')
    return results_df[[c for c in cols if c in results_df.columns]]
