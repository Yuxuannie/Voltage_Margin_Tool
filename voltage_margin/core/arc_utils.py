"""Arc and corner helpers for Phase 1 normalization."""

import re


def normalize_arc(arc, compare_source, config):
    """Return canonical Arc with table-position suffix normalized to _N-M."""
    if not isinstance(arc, str):
        return arc
    source_cfg = config["arc_normalization"]["compare_sources"][compare_source]
    pattern = re.compile(source_cfg["normalize_suffix_pattern"])
    match = pattern.search(arc)
    if not match:
        return arc
    replacement = source_cfg["replacement"].format(**match.groupdict())
    return pattern.sub(replacement, arc)


def parse_cell_name(arc, analysis_type, compare_source, config, row=None):
    """Resolve Cell_Name from a column when present, otherwise from Arc."""
    cell_cfg = config["cell_name"][compare_source]
    if cell_cfg["source"] == "column":
        value = row.get(cell_cfg["column"]) if row is not None else None
        return None if value is None else str(value)

    parse_key = "hold" if analysis_type == "hold" else "delay_slew"
    pattern = cell_cfg["parse"][parse_key]["pattern"]
    match = re.search(pattern, arc or "")
    return match.group("cell") if match else None


def extract_voltage_from_corner(corner):
    if not corner:
        return None
    match = re.search(r"(\d+)p(\d+)v", corner)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def extract_temperature_from_corner(corner):
    if not corner:
        return None
    match = re.search(r"(?P<sign>m?)(?P<temp>\d+)c", corner)
    if not match:
        return None
    value = int(match.group("temp"))
    return -value if match.group("sign") == "m" else value


def corner_family(corner):
    """Replace only the voltage token so sensitivities do not cross temperature."""
    if not corner:
        return None
    return re.sub(r"\d+p\d+v", "<V>", corner, count=1)
