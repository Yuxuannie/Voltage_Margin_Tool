"""
Data loader for .rpt files (sigma and moments).

Handles:
  - Auto-detection of CDNS vs SNPS vendor columns
  - Loading sigma .rpt files (fmc*corner*type*.rpt)
  - Loading moments .rpt files (MC_*corner*type*.rpt)
  - Corner name extraction from filenames
"""

import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .thresholds import SIGMA_PARAMS, MOMENTS_PARAMS

logger = logging.getLogger(__name__)


def detect_vendor_prefix(df):
    """Auto-detect CDNS or SNPS vendor from DataFrame column names."""
    for col in df.columns:
        col_lower = col.lower()
        if 'cdns' in col_lower:
            return 'CDNS_Lib'
        if 'snps' in col_lower:
            return 'SNPS_Lib'
    return 'CDNS_Lib'  # default


def extract_corner_from_filename(filename):
    """
    Extract corner name from filename.

    Examples:
        'fmc_ssgnp_0p450v_m40c_delay.rpt' -> 'ssgnp_0p450v_m40c'
        'MC_ssgnp_0p450v_m40c_delay.rpt'  -> 'ssgnp_0p450v_m40c'
    """
    base = Path(filename).stem
    match = re.search(r'(ssg[ng][pg]_\d+p\d+v_[mn]\d+c)', base)
    if match:
        return match.group(1)

    # Broader fallback: look for process_voltage_temp pattern
    match = re.search(r'([a-z]+_\d+p\d+v_[mn]?\d+c)', base)
    if match:
        return match.group(1)

    return None


def extract_type_from_filename(filename):
    """Extract analysis type (delay/slew/hold) from filename."""
    base = Path(filename).stem.lower()
    for t in ['delay', 'slew', 'hold']:
        if t in base:
            return t
    return None


def extract_voltage_from_corner(corner_name):
    """
    Extract voltage value from corner name.

    'ssgnp_0p450v_m40c' -> 0.450
    """
    if not corner_name:
        return None
    match = re.search(r'(\d+)p(\d+)v', corner_name)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def find_rpt_files(root_path, corners, types, file_prefix='fmc'):
    """
    Find .rpt files matching pattern: {prefix}*{corner}*{type}*.rpt

    Args:
        root_path: Directory containing .rpt files
        corners: List of corner names to look for
        types: List of analysis types
        file_prefix: 'fmc' for sigma files, 'MC' for moments files

    Returns:
        Dict mapping (corner, type) -> file_path
    """
    found = {}
    try:
        all_files = os.listdir(root_path)
        rpt_files = [f for f in all_files if f.endswith('.rpt')]

        if file_prefix:
            rpt_files = [f for f in rpt_files
                         if f.lower().startswith(file_prefix.lower())]

        for corner in corners:
            for type_name in types:
                matches = [f for f in rpt_files
                           if corner in f and type_name in f]
                if matches:
                    found[(corner, type_name)] = os.path.join(root_path, matches[0])
                    if len(matches) > 1:
                        logger.warning(
                            f"Multiple matches for {corner}*{type_name}: {matches}")
                else:
                    logger.warning(f"No {file_prefix} file for {corner}*{type_name}")
    except Exception as e:
        logger.error(f"Error scanning {root_path}: {e}")

    return found


def auto_discover_files(root_path):
    """
    Auto-discover all .rpt files and extract corners/types.

    Returns:
        Tuple of:
          - sigma_files: Dict[(corner, type)] -> path
          - moments_files: Dict[(corner, type)] -> path
          - corners: sorted list of discovered corners
          - types: sorted list of discovered types
    """
    sigma_files = {}
    moments_files = {}
    corners_set = set()
    types_set = set()

    try:
        for f in os.listdir(root_path):
            if not f.endswith('.rpt'):
                continue

            corner = extract_corner_from_filename(f)
            type_name = extract_type_from_filename(f)
            if not corner or not type_name:
                continue

            corners_set.add(corner)
            types_set.add(type_name)

            path = os.path.join(root_path, f)
            f_lower = f.lower()
            if f_lower.startswith('fmc'):
                sigma_files[(corner, type_name)] = path
            elif f_lower.startswith('mc'):
                moments_files[(corner, type_name)] = path
    except Exception as e:
        logger.error(f"Error auto-discovering files in {root_path}: {e}")

    return (sigma_files, moments_files,
            sorted(corners_set), sorted(types_set))


def load_rpt_file(file_path):
    """
    Load a single .rpt file as a DataFrame.

    Handles both CSV-formatted .rpt and whitespace-delimited .rpt.
    """
    try:
        df = pd.read_csv(file_path, header=0, skiprows=[1], skip_blank_lines=True)
        if len(df.columns) <= 1:
            # Retry with whitespace delimiter
            df = pd.read_csv(file_path, sep=r'\s+', header=0, skiprows=[1],
                             skip_blank_lines=True)
        logger.info(f"Loaded {file_path}: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def resolve_columns(df, mc_prefix, lib_prefix, param_name):
    """
    Resolve actual column names for a parameter.

    Returns dict with keys: mc_val, lib_val, mc_ci_lb, mc_ci_ub, abs_err, rel_err
    Each value is the column name (str) or None if not found.
    """
    cols = {}

    # MC and Lib value columns
    cols['mc_val'] = _find_col(df, f"{mc_prefix}_{param_name}")
    cols['lib_val'] = _find_col(df, f"{lib_prefix}_{param_name}")

    # CI bounds (sigma files have these; moments may not)
    cols['mc_ci_lb'] = _find_col(df, f"{mc_prefix}_{param_name}_LB")
    cols['mc_ci_ub'] = _find_col(df, f"{mc_prefix}_{param_name}_UB")

    # Pre-calculated errors (may have different naming conventions)
    cols['abs_err'] = (_find_col(df, f"{lib_prefix}_{param_name}_Dif")
                       or _find_col(df, f"{param_name}_abs_err"))
    cols['rel_err'] = (_find_col(df, f"{lib_prefix}_{param_name}_Rel")
                       or _find_col(df, f"{param_name}_rel_err"))

    return cols


def _find_col(df, name):
    """Find a column by exact name, return name if found, else None."""
    if name in df.columns:
        return name
    return None


class DataBundle:
    """
    Container for loaded data for one (corner, type) combination.

    Attributes:
        corner: Corner name
        type_name: Analysis type (delay/slew/hold)
        sigma_df: DataFrame from sigma .rpt (or None)
        moments_df: DataFrame from moments .rpt (or None)
        vendor_prefix: Detected vendor prefix (CDNS_Lib or SNPS_Lib)
    """

    def __init__(self, corner, type_name, sigma_df=None, moments_df=None):
        self.corner = corner
        self.type_name = type_name
        self.sigma_df = sigma_df
        self.moments_df = moments_df
        self.vendor_prefix = None

        if sigma_df is not None:
            self.vendor_prefix = detect_vendor_prefix(sigma_df)
        elif moments_df is not None:
            self.vendor_prefix = detect_vendor_prefix(moments_df)

    @property
    def voltage(self):
        return extract_voltage_from_corner(self.corner)

    def has_sigma(self):
        return self.sigma_df is not None and not self.sigma_df.empty

    def has_moments(self):
        return self.moments_df is not None and not self.moments_df.empty

    def available_params(self):
        """Return list of parameter names available in loaded data."""
        params = []
        if self.has_sigma():
            mc_prefix = 'MC'
            for p in SIGMA_PARAMS:
                if f"{mc_prefix}_{p}" in self.sigma_df.columns:
                    params.append(p)
        if self.has_moments():
            mc_prefix = 'MC'
            for p in MOMENTS_PARAMS:
                if f"{mc_prefix}_{p}" in self.moments_df.columns:
                    params.append(p)
        return params


def load_all_data(root_path, corners=None, types=None):
    """
    Load all data from a directory.

    Args:
        root_path: Directory with .rpt files
        corners: List of corners (auto-discover if None)
        types: List of types (auto-discover if None)

    Returns:
        Dict[(corner, type_name)] -> DataBundle
    """
    sigma_files, moments_files, disc_corners, disc_types = auto_discover_files(root_path)

    if corners is None:
        corners = disc_corners
    if types is None:
        types = disc_types

    logger.info(f"Corners: {corners}")
    logger.info(f"Types: {types}")
    logger.info(f"Sigma files found: {len(sigma_files)}")
    logger.info(f"Moments files found: {len(moments_files)}")

    bundles = {}
    for corner in corners:
        for type_name in types:
            key = (corner, type_name)
            sigma_df = None
            moments_df = None

            if key in sigma_files:
                sigma_df = load_rpt_file(sigma_files[key])
            if key in moments_files:
                moments_df = load_rpt_file(moments_files[key])

            if sigma_df is not None or moments_df is not None:
                bundles[key] = DataBundle(corner, type_name, sigma_df, moments_df)

    logger.info(f"Loaded {len(bundles)} data bundles")
    return bundles
