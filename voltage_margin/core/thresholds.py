"""Per-type, per-parameter threshold definitions for pass rate analysis."""

from .config_loader import load_policy


# --- Relative error thresholds (fraction, not percent) ---
REL_THRESHOLDS = {
    'delay': {
        'Early_Sigma': 0.03,
        'Late_Sigma':  0.03,
        'Meanshift':   0.01,
        'Nominal':     0.01,
        'Std':         0.02,
        'Skew':        0.05,
    },
    'slew': {
        'Early_Sigma': 0.06,
        'Late_Sigma':  0.06,
        'Meanshift':   0.02,
        'Nominal':     0.02,
        'Std':         0.04,
        'Skew':        0.10,
    },
    'hold': {
        'Late_Sigma':  0.03,
        'Nominal':     0.01,
    },
}

# --- Absolute error thresholds ---
# abs_threshold = max(slew_multiplier * rel_pin_slew, ps_value)
# ps_value is in raw units (same as data); slew_multiplier is unitless ratio
ABS_THRESHOLD_PARAMS = {
    'delay': {
        'Early_Sigma': {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Late_Sigma':  {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Meanshift':   {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Nominal':     {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Std':         {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Skew':        {'slew_multiplier': 0.005, 'ps_value': 1.0},
    },
    'slew': {
        'Early_Sigma': {'slew_multiplier': 0.01,  'ps_value': 2.0},
        'Late_Sigma':  {'slew_multiplier': 0.01,  'ps_value': 2.0},
        'Meanshift':   {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Nominal':     {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Std':         {'slew_multiplier': 0.005, 'ps_value': 1.0},
        'Skew':        {'slew_multiplier': 0.005, 'ps_value': 1.0},
    },
    'hold': {
        'Late_Sigma':  {'slew_multiplier': 0.005, 'ps_value': 10.0},
        'Nominal':     {'slew_multiplier': 0.005, 'ps_value': 10.0},
    },
}

# CI enlargement factor (6% of CI width)
CI_ENLARGEMENT_FACTOR = 0.06

# Moments CI estimation: when real CI bounds are not available,
# estimate as MC_value +/- (estimation_fraction * |MC_value|)
MOMENTS_CI_ESTIMATION_FRACTION = 0.10

# Sigma parameters (have real CI bounds from MC simulation)
SIGMA_PARAMS = ['Early_Sigma', 'Late_Sigma']

# Moments parameters (may need estimated CI bounds)
MOMENTS_PARAMS = ['Std', 'Skew', 'Meanshift', 'Nominal']

# All parameters
ALL_PARAMS = SIGMA_PARAMS + MOMENTS_PARAMS

# Analysis types
ANALYSIS_TYPES = ['delay', 'slew', 'hold']

# Hold type only uses Late_Sigma for sigma
HOLD_SIGMA_PARAMS = ['Late_Sigma']


def _policy_metric(policy, type_name, param_name):
    if policy is None:
        return None
    return policy.get('thresholds', {}).get(type_name, {}).get(param_name)


def get_rel_threshold(type_name, param_name, policy=None):
    """Get relative error threshold for a given type and parameter."""
    metric_policy = _policy_metric(policy, type_name, param_name)
    if metric_policy is not None:
        return metric_policy.get('rel_threshold')
    return REL_THRESHOLDS.get(type_name, {}).get(param_name)


def get_abs_threshold(type_name, param_name, rel_pin_slew, policy=None):
    """
    Calculate absolute error threshold.

    Returns:
        float: max(slew_multiplier * rel_pin_slew, ps_value)
    """
    metric_policy = _policy_metric(policy, type_name, param_name)
    if metric_policy is not None:
        params = metric_policy.get('abs_threshold')
        return max(params['slew_multiplier'] * rel_pin_slew, params['floor_ps'])

    params = ABS_THRESHOLD_PARAMS.get(type_name, {}).get(param_name)
    if params is None:
        return None
    return max(params['slew_multiplier'] * rel_pin_slew, params['ps_value'])


def get_sigma_params_for_type(type_name):
    """Return which sigma parameters apply for a given type."""
    if type_name == 'hold':
        return ['Late_Sigma', 'Nominal']
    return SIGMA_PARAMS


def get_all_params_for_type(type_name):
    """Return all parameters to check for a given type."""
    if type_name == 'hold':
        return ['Late_Sigma', 'Nominal']
    sigma = get_sigma_params_for_type(type_name)
    return sigma + MOMENTS_PARAMS


def load_default_policy():
    """Compatibility helper for callers that want the Phase 1 policy."""
    return load_policy()
