#!/usr/bin/env python3
"""
CLI entry point for Voltage Margin Analysis Tool.

Usage:
    python run_analysis.py /path/to/rpt_files [--output results.csv]
    python run_analysis.py /path/to/rpt_files --no-waiver1 --no-optimistic
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from voltage_margin.core.data_loader import load_all_data
from voltage_margin.core.config_loader import (
    DEFAULT_COLUMN_MAPPING, DEFAULT_POLICY, load_policy,
)
from voltage_margin.core.margin_engine import (
    build_margin_outputs, build_sensitivity_rows,
)
from voltage_margin.core.normalizer import load_normalized_data
from voltage_margin.core.pass_rate_engine import (
    run_full_analysis, results_to_dataframe, filter_results_by_waivers,
)
from voltage_margin.core.thresholds import get_abs_threshold, get_rel_threshold


def main():
    parser = argparse.ArgumentParser(
        description='Voltage Margin Analysis Tool - Pass Rate Calculation')
    parser.add_argument('data_dir', help='Directory containing .rpt files')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file (default: <output-dir>/pass_rate/pass_rate_results.csv)')
    parser.add_argument('--output-dir', default='voltage_margin_outputs',
                        help='Phase 1 output package directory (default: voltage_margin_outputs)')
    parser.add_argument('--column-map', default=str(DEFAULT_COLUMN_MAPPING),
                        help='Column mapping YAML file')
    parser.add_argument('--policy', default=str(DEFAULT_POLICY),
                        help='Policy YAML file')
    parser.add_argument('--corners', nargs='+', default=None,
                        help='Corners to analyze (default: auto-detect)')
    parser.add_argument('--types', nargs='+', default=None,
                        help='Types to analyze (default: auto-detect)')
    parser.add_argument('--no-waiver1', action='store_true',
                        help='Disable CI enlargement waiver')
    parser.add_argument('--no-optimistic', action='store_true',
                        help='Disable optimistic direction waiver')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Load data
    logging.info(f'Loading data from: {args.data_dir}')
    bundles = load_all_data(args.data_dir, args.corners, args.types)
    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else output_dir / 'pass_rate' / 'pass_rate_results.csv'

    if not bundles:
        logging.error('No data found. Check your directory and file patterns.')
        sys.exit(1)

    logging.info(f'Loaded {len(bundles)} data bundles')

    # Run analysis
    results = run_full_analysis(bundles)
    if not results:
        logging.error('No results produced.')
        sys.exit(1)

    # Convert to DataFrame
    full_df = results_to_dataframe(results)

    # Apply waiver filter
    filtered_df = filter_results_by_waivers(
        full_df,
        waiver1_enabled=not args.no_waiver1,
        optimistic_enabled=not args.no_optimistic,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    logging.info(f'Results saved to: {output_path}')

    # Phase 1 normalized, sensitivity, and margin output package.
    policy = load_policy(args.policy)
    normalized_df, manifest_df, normalization_warnings_df = load_normalized_data(
        args.data_dir,
        corners=args.corners,
        types=args.types,
        column_mapping=args.column_map,
    )
    sensitivity_df, sensitivity_warnings_df = build_sensitivity_rows(
        normalized_df, policy)
    margin_outputs = build_margin_outputs(normalized_df, sensitivity_df, policy)
    _write_phase1_outputs(
        output_dir=output_dir,
        manifest_df=manifest_df,
        normalized_df=normalized_df,
        normalization_warnings_df=normalization_warnings_df,
        pass_rate_df=filtered_df,
        pass_rate_output_path=output_path,
        arc_results_df=_arc_results_to_dataframe(results, normalized_df, policy),
        sensitivity_df=sensitivity_df,
        sensitivity_warnings_df=sensitivity_warnings_df,
        margin_outputs=margin_outputs,
    )

    # Print summary
    print('\n' + '=' * 70)
    print('PASS RATE SUMMARY')
    print('=' * 70)
    print(filtered_df.to_string(index=False))
    print('=' * 70)

    total_arcs = sum(r.total_arcs for r in results)
    print(f'\nTotal: {len(results)} parameter groups, {total_arcs} arcs')

    waivers = []
    if not args.no_waiver1:
        waivers.append('CI Enlargement')
    if not args.no_optimistic:
        waivers.append('Optimistic Direction')
    print(f'Active waivers: {", ".join(waivers) if waivers else "None"}')


def _write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_phase1_outputs(
    output_dir,
    manifest_df,
    normalized_df,
    normalization_warnings_df,
    pass_rate_df,
    pass_rate_output_path,
    arc_results_df,
    sensitivity_df,
    sensitivity_warnings_df,
    margin_outputs,
):
    _write_csv(manifest_df, output_dir / 'manifest.csv')
    _write_csv(normalized_df, output_dir / 'normalized' / 'normalized_rows.csv')
    _write_csv(normalization_warnings_df, output_dir / 'normalized' / 'normalization_warnings.csv')

    package_pass_rate = output_dir / 'pass_rate' / 'pass_rate_results.csv'
    if pass_rate_output_path.resolve() != package_pass_rate.resolve():
        _write_csv(pass_rate_df, package_pass_rate)
    _write_csv(arc_results_df, output_dir / 'pass_rate' / 'per_arc_pass_fail.csv')

    _write_csv(sensitivity_df, output_dir / 'sensitivity' / 'sensitivity.csv')
    _write_csv(sensitivity_warnings_df, output_dir / 'sensitivity' / 'sensitivity_warnings.csv')

    _write_margin_set(output_dir / 'all_errors', margin_outputs.all_errors_per_object,
                      margin_outputs.all_errors_summary, margin_outputs.all_errors_curve,
                      margin_outputs.all_errors_high_margin)
    _write_margin_set(output_dir / 'optimistic_only', margin_outputs.optimistic_only_per_object,
                      margin_outputs.optimistic_only_summary, margin_outputs.optimistic_only_curve,
                      margin_outputs.optimistic_only_high_margin)


def _write_margin_set(directory, per_object, summary, curve, high_margin):
    _write_csv(per_object, directory / 'per_object_margin.csv')
    _write_csv(summary, directory / 'margin_summary.csv')
    _write_csv(curve, directory / 'margin_efficiency_curve.csv')
    _write_csv(high_margin, directory / 'high_margin_objects.csv')


def _arc_results_to_dataframe(results, normalized_df, policy):
    metadata = _normalized_metadata_lookup(normalized_df)
    rows = []
    for summary in results:
        for arc in summary.arc_results:
            key = (summary.corner, summary.type_name, summary.param_name, arc.arc)
            meta = metadata.get(key, {})
            rel_threshold = get_rel_threshold(summary.type_name, summary.param_name, policy=policy)
            abs_threshold = get_abs_threshold(
                summary.type_name, summary.param_name, arc.rel_pin_slew, policy=policy)
            rows.append(
                {
                    'process': meta.get('process'),
                    'process_version': meta.get('process_version'),
                    'corner': summary.corner,
                    'voltage_v': meta.get('voltage_v'),
                    'analysis_type': summary.type_name,
                    'metric': summary.param_name,
                    'arc': meta.get('arc', arc.arc),
                    'mc_value_ps': arc.mc_value,
                    'lib_value_ps': arc.lib_value,
                    'dif_ps': arc.abs_err,
                    'rel_err': arc.rel_err,
                    'rel_pin_slew_ps': arc.rel_pin_slew,
                    'mc_ci_lb_ps': arc.mc_ci_lb,
                    'mc_ci_ub_ps': arc.mc_ci_ub,
                    'rel_threshold': rel_threshold,
                    'abs_threshold_ps': abs_threshold,
                    'rel_pass': arc.rel_pass,
                    'abs_pass': arc.abs_pass,
                    'ci_bounds_pass': arc.ci_bounds_pass,
                    'base_pass': arc.base_pass,
                    'pass_reason': arc.pass_reason,
                    'waiver1_ci_enlarged': arc.waiver1_ci_enlarged,
                    'is_optimistic_risk': arc.is_optimistic,
                    'pass_with_waiver1': arc.pass_with_waiver1,
                    'pass_optimistic_only': arc.pass_optimistic_only,
                    'pass_with_both_waivers': arc.pass_with_both_waivers,
                }
            )
    return pd.DataFrame(rows)


def _normalized_metadata_lookup(normalized_df):
    if normalized_df.empty:
        return {}
    lookup = {}
    columns = [
        'process',
        'process_version',
        'corner',
        'voltage_v',
        'analysis_type',
        'metric',
        'arc_original',
        'arc',
    ]
    for _, row in normalized_df[columns].drop_duplicates().iterrows():
        key = (row['corner'], row['analysis_type'], row['metric'], row['arc_original'])
        lookup[key] = row.to_dict()
    return lookup


if __name__ == '__main__':
    main()
