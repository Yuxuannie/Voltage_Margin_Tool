"""Load Phase 1 mapping and policy configs."""

from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_COLUMN_MAPPING = CONFIG_DIR / "column_mapping_phase1.yaml"
DEFAULT_POLICY = CONFIG_DIR / "policy_phase1.yaml"


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_column_mapping(path=None):
    """Load and lightly validate the Phase 1 column mapping config."""
    config = _load_yaml(path or DEFAULT_COLUMN_MAPPING)
    required = [
        "file_loading",
        "filename",
        "table_type_to_analysis_type",
        "analysis_metrics",
        "arc_normalization",
        "cell_name",
        "dif_resolution",
        "compare_sources",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Column mapping missing required keys: {missing}")
    if config["analysis_metrics"].get("hold") != ["Late_Sigma", "Nominal"]:
        raise ValueError("Hold metrics must be exactly Late_Sigma and Nominal")
    return config


def load_policy(path=None):
    """Load and lightly validate the Phase 1 policy config."""
    config = _load_yaml(path or DEFAULT_POLICY)
    required = ["units", "direction_rule", "sensitivity", "waivers", "thresholds"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Policy missing required keys: {missing}")
    if config["direction_rule"].get("scope") != "global":
        raise ValueError("Direction rule must be global in Phase 1")
    return config
