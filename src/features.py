"""Catalog feature engineering for e-commerce and tabular competitions.

Extracts structured physical quantities, pack configurations, model codes,
text surface statistics, domain interactions, and fold-safe Bayesian target
encodings from messy product catalog fields.

Key capabilities:
- Context-aware regex extraction with guards for common '5G', '2 in 1', and dimension/pack collisions
- Hierarchical field resolution (Title > Pack Size > Description) with ingredient filtering
- Canonical unit conversion (to grams, milliliters, centimeters, watts, GB)
- Domain interactions (density proxy, aspect ratio, tech spec indicators, multipack consistency)
- Optional missingness imputation for scikit-learn linear models (Ridge/Logistic)
- Strict fold-safe out-of-fold Bayesian target encoding with prior smoothing
- Optional multicore parallel execution with joblib (n_jobs)
- Standalone CLI with parquet caching and metadata generation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


FEATURES_VERSION = "2.1"
MISSING_CATEGORY = "__MISSING__"


def _clean_text(value: Any) -> str:
    """Return a safe scalar string; pandas missing scalars become empty text."""
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    if not np.isscalar(value):
        raise TypeError(f"Expected a scalar text value, got {type(value).__name__}.")
    return str(value).strip()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# ---------------------------------------------------------------------------
# Pre-compiled Regex Patterns & Conversion Lookup Tables
# ---------------------------------------------------------------------------

# Multipliers & pack patterns
RE_MULTIPACK_DIM = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(?:x|\*)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+|\"|inch(?:es)?)\b"
)
RE_PACK_OF = re.compile(
    r"(?i)\b(?:pack\s+of|set\s+of|combo\s+of|box\s+of|case\s+of|bundle\s+of|pair\s+of)\s*(\d+)\b"
)
RE_COUNT_PACK = re.compile(
    r"(?i)\b(\d+)\s*[- ]*(?:pack|pk|pcs|pieces|bottles|cans|count|ct|units)\b"
)
RE_NESTED_PACK = re.compile(
    r"(?i)\b(\d+)\s*(?:boxes?|packs?|cases?|cartons?|bundles?)\s*"
    r"(?:of|x)\s*(\d+)\s*(?:pcs?|pieces?|units?|items?|pens?|bottles?|cans?|"
    r"sachets?|bags?|bars?|tablets?|capsules?)?\b"
)
RE_SINGLE_PAIR = re.compile(r"(?i)\bpair\b(?!\s+of\b)")

# Valid units specifically allowed to define a multipack quantity (excludes distance units like cm/mm/inch)
MULTIPACK_VALID_UNITS: set[str] = {
    "ml", "mls", "milliliter", "milliliters", "millilitre", "millilitres",
    "l", "lt", "ltr", "ltrs", "liter", "liters", "litre", "litres",
    "g", "gm", "gms", "gram", "grams",
    "kg", "kgs", "kilo", "kilos", "kilogram", "kilograms",
    "oz", "ounce", "ounces", "fl oz", "floz",
    "lb", "lbs", "pound", "pounds",
    "pcs", "piece", "pieces", "pack", "packs", "bottle", "bottles",
    "can", "cans", "sachet", "sachets", "tab", "tabs", "tablet", "tablets",
    "cap", "caps", "capsule", "capsules", "bags", "wipes", "sheets",
}

# Fraction pattern: '1/2 liter', '3/4 kg', '1/2 inch'
RE_FRACTION = re.compile(r"(?i)\b(\d+)\s*/\s*(\d+)\s*([a-zA-Z]+|\"|inch(?:es)?)\b")

# Weight patterns: '1.5 kg', '500g', '250 gm', '16 oz', '2.2 lbs'
RE_WEIGHT = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|kilos?|gm?s?|grams?|mg|mgs|milligrams?|oz|ounces?|lbs?|pounds?|g)\b"
)
WEIGHT_TO_GRAMS: dict[str, float] = {
    "mg": 0.001,
    "mgs": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
    "g": 1.0,
    "gm": 1.0,
    "gms": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kgs": 1000.0,
    "kilo": 1000.0,
    "kilos": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.349523,
    "ounce": 28.349523,
    "ounces": 28.349523,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
}

# Volume patterns: '250 ml', '1.5 L', '500 ltr', '16 fl oz', '1 gallon'
RE_VOLUME = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(ml|mls|millilit(?:er|re)s?|ltr?s?|lit(?:er|re)s?|cl|fl\.?\s*oz|fluid\s*ounces?|gallons?|gals?|pints?|quarts?|l)\b"
)
VOLUME_TO_ML: dict[str, float] = {
    "ml": 1.0,
    "mls": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "millilitre": 1.0,
    "millilitres": 1.0,
    "cl": 10.0,
    "cls": 10.0,
    "l": 1000.0,
    "lt": 1000.0,
    "ltr": 1000.0,
    "ltrs": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "litre": 1000.0,
    "litres": 1000.0,
    "fl oz": 29.5735,
    "fl. oz": 29.5735,
    "fl.oz": 29.5735,
    "floz": 29.5735,
    "fluid ounce": 29.5735,
    "fluid ounces": 29.5735,
    "gal": 3785.41,
    "gals": 3785.41,
    "gallon": 3785.41,
    "gallons": 3785.41,
    "pint": 473.176,
    "pints": 473.176,
    "quart": 946.353,
    "quarts": 946.353,
}

# Linear & 3D dimensions: '12-inch', '10 x 20 cm', '10x20x30 cm', '15.6"'
RE_DIM_3D = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(?:x|\*)\s*(\d+(?:\.\d+)?)\s*(?:x|\*)\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|inch(?:es)?|\"|in|ft|feet)(?!\w)"
)
RE_DIM_2D = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(?:x|\*)\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|inch(?:es)?|\"|in|ft|feet)(?!\w)"
)
RE_DIM_1D = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*[- ]*(inch(?:es)?|\"|in|cm|mm|m|ft|feet)(?!\w)"
)
LENGTH_TO_CM: dict[str, float] = {
    "mm": 0.1,
    "cm": 1.0,
    "m": 100.0,
    "inch": 2.54,
    "inches": 2.54,
    '"': 2.54,
    "in": 2.54,
    "ft": 30.48,
    "feet": 30.48,
}

# Power, voltage, battery capacity: '60 W', '1200 watt', '12 v', '5000 mah'
RE_POWER = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*[- ]*(w|watts?|kw|kilowatts?)\b")
RE_VOLTAGE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*[- ]*(v|volts?|kv)\b")
RE_BATTERY = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*[- ]*(mah|ah)\b")

# Storage / Memory: '16 GB', '512 GB SSD', '1 TB', '32 mb'
RE_STORAGE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*[- ]*(tb|gb|mb)\b")

# Percentage / Concentration: '60%', '10% off', 'SPF 50'
RE_PERCENTAGE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*%(?!\w)")
RE_SPF = re.compile(r"(?i)\bspf\s*[- ]*(\d+)\+?\b")

# Model identifiers: uppercase/digit codes with hyphens, slashes, or alphanumeric mix
RE_MODEL_CODE = re.compile(
    r"\b(?:[A-Z0-9]{1,8}[-_/][A-Z0-9]{1,8}|[A-Z]{1,4}\d{2,6}[A-Z0-9]*|[A-Z0-9]{2,}\d{2,}[A-Z0-9]*)\b"
)
COMMON_WORDS_FILTER = {
    "PACK", "SET", "COMBO", "SIZE", "INCH", "INCHES", "GRAMS", "WATT", "VOLT", "PRICE", "INDIA",
    "TABLET", "TABLETS", "CAPSULES", "BOTTLE", "PIECE", "PIECES"
}

# ---------------------------------------------------------------------------
# Robust Disambiguated Parsers
# ---------------------------------------------------------------------------

def parse_pack_count(text: str | None) -> tuple[float, int]:
    """Extract pack multiplier and multipack indicator from text.

    Disambiguates multipacks from 2D dimensions by validating the unit.
    """
    s = _clean_text(text)
    if not s:
        return 1.0, 0

    # Explicit two-level structures must be handled before their inner/outer
    # fragments. Example: "2 boxes of 6 pens" means 12 saleable units.
    nested = RE_NESTED_PACK.search(s)
    if nested:
        count = float(nested.group(1)) * float(nested.group(2))
        if 1.0 < count <= 200.0:
            return count, 1

    # Independent multiplier families may coexist, as in
    # "Pack of 2 (3 x 100 ml)". Use at most one match per family so repeated
    # marketing text cannot exponentiate the count accidentally.
    factors: list[float] = []

    for match in RE_MULTIPACK_DIM.finditer(s):
        unit = match.group(3).lower()
        if unit in MULTIPACK_VALID_UNITS:
            count = float(match.group(1))
            if 1.0 < count <= 200.0:
                factors.append(count)
                break

    match = RE_PACK_OF.search(s)
    if match:
        count = float(match.group(1))
        if 1.0 < count <= 200.0:
            factors.append(count)

    match = RE_COUNT_PACK.search(s)
    if match:
        count = float(match.group(1))
        if 1.0 < count <= 200.0:
            factors.append(count)

    if RE_SINGLE_PAIR.search(s):
        factors.append(2.0)

    if not factors:
        return 1.0, 0
    product = float(math.prod(factors))
    # Implausibly large compound parses are more dangerous than a conservative
    # single-factor estimate. Keep the strongest observed valid multiplier.
    return (product if product <= 200.0 else max(factors)), 1


def parse_weight_grams(text: str | None) -> float | None:
    """Extract canonical weight in grams, immune to cellular '5G' and fractions."""
    s = _clean_text(text)
    if not s:
        return None

    # Fraction first: '1/2 kg'
    m_frac = RE_FRACTION.search(s)
    if m_frac:
        num, denom, unit_raw = m_frac.group(1), m_frac.group(2), m_frac.group(3).lower()
        multiplier = WEIGHT_TO_GRAMS.get(unit_raw)
        if multiplier and float(denom) != 0:
            return (float(num) / float(denom)) * multiplier

    # Regular weight matches
    candidates: list[float] = []
    for match in RE_WEIGHT.finditer(s):
        val_str, unit_raw = match.group(1), match.group(2)
        full_token = match.group(0).strip()

        # Semantic check: filter out cellular 2G/3G/4G/5G/6G false positives
        if unit_raw.lower() == "g":
            if full_token.upper() in {"2G", "3G", "4G", "5G", "6G"}:
                continue
            tail = s[match.end():match.end() + 25].lower()
            if any(term in tail for term in ["network", "cellular", "lte", "sim", "band", "phone", "speed", "wireless", "mobile"]):
                continue

        multiplier = WEIGHT_TO_GRAMS.get(unit_raw.lower())
        if multiplier is not None:
            try:
                val = float(val_str) * multiplier
                if 0.0001 <= val <= 200_000.0:  # up to 200 kg
                    candidates.append(val)
            except ValueError:
                continue

    if not candidates:
        return None
    # If multiple weights found, prefer the largest plausible net package weight
    return max(candidates)


def parse_volume_ml(text: str | None) -> float | None:
    """Extract canonical volume in milliliters from text, handling fractions."""
    s = _clean_text(text)
    if not s:
        return None

    m_frac = RE_FRACTION.search(s)
    if m_frac:
        num, denom, unit_raw = m_frac.group(1), m_frac.group(2), m_frac.group(3).lower()
        multiplier = VOLUME_TO_ML.get(unit_raw)
        if multiplier and float(denom) != 0:
            return (float(num) / float(denom)) * multiplier

    candidates: list[float] = []
    for match in RE_VOLUME.finditer(s):
        val_str, unit_raw = match.group(1), match.group(2).lower()
        multiplier = VOLUME_TO_ML.get(unit_raw)
        if multiplier is not None:
            try:
                val = float(val_str) * multiplier
                if 0.01 <= val <= 50_000.0:
                    candidates.append(val)
            except ValueError:
                continue

    if not candidates:
        return None
    return max(candidates)


def parse_dimensions_cm(text: str | None) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract 1D, 2D, or 3D dimensions in cm, immune to '2 in 1' preposition trap.

    Returns:
        tuple[dim_length_cm, dim_width_cm, dim_height_cm, dim_volume_cm3]
    """
    s = _clean_text(text)
    if not s:
        return None, None, None, None

    # 1. 3D: '10 x 20 x 30 cm'
    m3 = RE_DIM_3D.search(s)
    if m3:
        d1, d2, d3, unit = float(m3.group(1)), float(m3.group(2)), float(m3.group(3)), m3.group(4).lower()
        mult = LENGTH_TO_CM.get(unit, 1.0)
        c1, c2, c3 = d1 * mult, d2 * mult, d3 * mult
        return max(c1, c2, c3), sorted([c1, c2, c3])[1], min(c1, c2, c3), c1 * c2 * c3

    # 2. 2D: '10 x 20 cm'
    m2 = RE_DIM_2D.search(s)
    if m2:
        d1, d2, unit = float(m2.group(1)), float(m2.group(2)), m2.group(3).lower()
        mult = LENGTH_TO_CM.get(unit, 1.0)
        c1, c2 = d1 * mult, d2 * mult
        return max(c1, c2), min(c1, c2), None, c1 * c2

    # 3. 1D: '12-inch', '15.6"', '250 mm'
    m1 = RE_DIM_1D.search(s)
    if m1:
        d1, unit = float(m1.group(1)), m1.group(2).lower()
        # Semantic check: filter out English preposition 'in' as in '2 in 1' or '100g in pouch'
        if unit == "in":
            tail = s[m1.end():m1.end() + 15].strip().lower()
            if any(tail.startswith(w) for w in ["1", "one", "pouch", "pack", "box", "bag", "a", "the", "stock", "color"]):
                return None, None, None, None
            head = s[max(0, m1.start() - 10):m1.start()].strip().lower()
            if any(head.endswith(w) for w in ["all", "built", "plugged", "made", "packed"]):
                return None, None, None, None

        mult = LENGTH_TO_CM.get(unit, 1.0)
        return d1 * mult, None, None, None

    return None, None, None, None


def parse_power_watts(text: str | None) -> float | None:
    """Extract canonical power in Watts from text."""
    s = _clean_text(text)
    if not s:
        return None
    m = RE_POWER.search(s)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        mult = 1000.0 if "k" in unit else 1.0
        return val * mult
    return None


def parse_voltage_volts(text: str | None) -> float | None:
    """Extract voltage in Volts from text."""
    s = _clean_text(text)
    if not s:
        return None
    m = RE_VOLTAGE.search(s)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        mult = 1000.0 if "k" in unit else 1.0
        return val * mult
    return None


def parse_battery_mah(text: str | None) -> float | None:
    """Extract battery capacity in mAh from text."""
    s = _clean_text(text)
    if not s:
        return None
    m = RE_BATTERY.search(s)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        mult = 1000.0 if unit == "ah" else 1.0
        return val * mult
    return None


def parse_storage_gb(text: str | None) -> float | None:
    """Extract digital storage in GB from text."""
    s = _clean_text(text)
    if not s:
        return None
    m = RE_STORAGE.search(s)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        if unit == "tb":
            return val * 1024.0
        elif unit == "gb":
            return val
        elif unit == "mb":
            return val / 1024.0
    return None


def parse_percentage(text: str | None) -> float | None:
    """Extract percentage or concentration value from text."""
    s = _clean_text(text)
    if not s:
        return None
    m = RE_PERCENTAGE.search(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def extract_model_identifiers(text: str | None) -> tuple[int, int, int]:
    """Identify model codes and hardware identifiers in text."""
    s = _clean_text(text)
    if not s:
        return 0, 0, 0
    matches = RE_MODEL_CODE.findall(s)
    valid_codes = [
        c for c in matches
        if c.upper() not in COMMON_WORDS_FILTER
        and any(ch.isdigit() for ch in c)
        and any(ch.isalpha() for ch in c)
    ]
    if not valid_codes:
        return 0, 0, 0
    return 1, len(valid_codes), max(len(c) for c in valid_codes)


# ---------------------------------------------------------------------------
# Hierarchical Multi-Field Extractor (Title > Pack > Description)
# ---------------------------------------------------------------------------

RE_TOTAL_QUANTITY_PREFIX = re.compile(
    r"(?i)(?:\btotal(?:\s+(?:net\s+)?(?:weight|wt|volume|vol))?|"
    r"\bnet\s+(?:wt|weight|vol|volume)|\bgross\s+weight)\.?\s*[:=-]?\s*$"
)
RE_PER_ITEM_SUFFIX = re.compile(
    r"(?i)^\s*(?:each\b|per\s+(?:item|unit|piece|pack|bottle|bar|can)\b|"
    r"/\s*(?:ea|each|pc|item|unit|bar|bottle|can)\b)"
)


def _classify_quantity_scope(
    text: str,
    selected_value: float | None,
    pattern: re.Pattern[str],
    conversions: dict[str, float],
) -> str:
    """Classify the selected measurement as total, per-item, or unknown.

    Scope is evaluated around the exact measurement that won parsing, rather
    than anywhere in the field. This prevents a remote word such as ``total``
    from changing the meaning of ``50 g each`` later in the description.
    """
    if not text or selected_value is None:
        return "unknown"

    selected_matches: list[re.Match[str]] = []
    for match in pattern.finditer(text):
        multiplier = conversions.get(match.group(2).lower().replace(" ", ""))
        if multiplier is None:
            continue
        candidate = float(match.group(1)) * multiplier
        tolerance = max(1e-6, abs(float(selected_value)) * 1e-6)
        if abs(candidate - float(selected_value)) <= tolerance:
            selected_matches.append(match)

    # An explicit suffix is the strongest signal and overrides a preceding
    # "net wt" label: "Net Wt 50 g each" is still a per-item quantity.
    for match in selected_matches:
        if RE_PER_ITEM_SUFFIX.search(text[match.end():match.end() + 40]):
            return "per_item"
    for match in selected_matches:
        if RE_TOTAL_QUANTITY_PREFIX.search(text[max(0, match.start() - 60):match.start()]):
            return "total"
    return "unknown"

def extract_row_physical_specs(
    title: str | None,
    pack: str | None,
    desc: str | None,
) -> dict[str, Any]:
    """Extract physical quantities using strict field hierarchy to avoid ingredient collisions."""
    t_clean = _clean_text(title)
    p_clean = _clean_text(pack)
    d_clean = _clean_text(desc)

    # 1. Pack count: reconcile PACK_SIZE and TITLE because either can contain
    # only the outer multiplier ("Pack of 2") while the other carries a richer
    # compound expression ("Pack of 2 (3 x 100 ml)"). Description is used only
    # when neither high-confidence field contains a valid multiplier.
    primary_pack_candidates = [
        parse_pack_count(value) for value in (p_clean, t_clean) if value
    ]
    detected = [count for count, multi in primary_pack_candidates if multi]
    if detected:
        pack_cnt, is_multi = max(detected), 1
    elif d_clean:
        pack_cnt, is_multi = parse_pack_count(d_clean)
    else:
        pack_cnt, is_multi = 1.0, 0

    # 2. Weight: TITLE > PACK_SIZE > DESCRIPTION. Track provenance because a
    # quantity explicitly described as a total must not be multiplied by pack count.
    weight_source = 0
    weight_source_text = ""
    weight = parse_weight_grams(t_clean)
    if weight is not None:
        weight_source, weight_source_text = 1, t_clean
    if weight is None:
        weight = parse_weight_grams(p_clean)
        if weight is not None:
            weight_source, weight_source_text = 2, p_clean
    if weight is None:
        weight = parse_weight_grams(d_clean)
        if weight is not None:
            weight_source, weight_source_text = 3, d_clean

    # 3. Volume: TITLE > PACK_SIZE > DESCRIPTION
    volume_source = 0
    volume_source_text = ""
    volume = parse_volume_ml(t_clean)
    if volume is not None:
        volume_source, volume_source_text = 1, t_clean
    if volume is None:
        volume = parse_volume_ml(p_clean)
        if volume is not None:
            volume_source, volume_source_text = 2, p_clean
    if volume is None:
        volume = parse_volume_ml(d_clean)
        if volume is not None:
            volume_source, volume_source_text = 3, d_clean

    # 4. Dimensions: TITLE > PACK_SIZE > DESCRIPTION
    l, w, h, v = parse_dimensions_cm(t_clean)
    if l is None:
        l, w, h, v = parse_dimensions_cm(p_clean)
    if l is None:
        l, w, h, v = parse_dimensions_cm(d_clean)

    # 5. Electrical & Storage: Search across combined title + desc
    tech_str = t_clean + " " + d_clean
    power = parse_power_watts(tech_str)
    voltage = parse_voltage_volts(tech_str)
    battery = parse_battery_mah(tech_str)
    storage = parse_storage_gb(tech_str)
    pct = parse_percentage(t_clean or d_clean)

    hm, mc, ml = extract_model_identifiers(t_clean or d_clean)

    # Derived totals. Classify the exact selected measurement, not the entire
    # source string, so "Total 2 kg" and "Net Wt 50 g each" behave correctly.
    weight_scope = _classify_quantity_scope(
        weight_source_text, weight, RE_WEIGHT, WEIGHT_TO_GRAMS
    )
    volume_scope = _classify_quantity_scope(
        volume_source_text, volume, RE_VOLUME, VOLUME_TO_ML
    )
    weight_is_explicit_total = weight_scope == "total"
    volume_is_explicit_total = volume_scope == "total"
    weight_is_per_item = weight_scope == "per_item"
    volume_is_per_item = volume_scope == "per_item"

    total_w = (
        weight if weight_is_explicit_total else weight * pack_cnt
    ) if weight is not None else np.nan
    total_v = (
        volume if volume_is_explicit_total else volume * pack_cnt
    ) if volume is not None else np.nan

    # Domain Interactions: Density proxy and aspect ratio
    density = np.nan
    if total_w is not None and not np.isnan(total_w) and v is not None and not np.isnan(v):
        density = float(total_w / (v + 1.0))

    aspect = np.nan
    if l is not None and not np.isnan(l) and w is not None and not np.isnan(w):
        aspect = float(l / (w + 1e-3))

    is_tech = 1 if (hm == 1 or (storage is not None and not np.isnan(storage)) or (power is not None and not np.isnan(power))) else 0

    return {
        "pack_count": np.float32(pack_cnt),
        "is_multipack": np.int8(is_multi),
        "unit_weight_g": np.float32(weight) if weight is not None else np.nan,
        "total_weight_g": np.float32(total_w),
        "log_total_weight_g": np.float32(np.log1p(total_w)) if not np.isnan(total_w) else np.nan,
        "has_weight": np.int8(1 if weight is not None else 0),
        "weight_source_rank": np.int8(weight_source),
        "weight_is_explicit_total": np.int8(weight_is_explicit_total),
        "weight_is_per_item": np.int8(weight_is_per_item),
        "unit_volume_ml": np.float32(volume) if volume is not None else np.nan,
        "total_volume_ml": np.float32(total_v),
        "log_total_volume_ml": np.float32(np.log1p(total_v)) if not np.isnan(total_v) else np.nan,
        "has_volume": np.int8(1 if volume is not None else 0),
        "volume_source_rank": np.int8(volume_source),
        "volume_is_explicit_total": np.int8(volume_is_explicit_total),
        "volume_is_per_item": np.int8(volume_is_per_item),
        "dim_length_cm": np.float32(l) if l is not None else np.nan,
        "dim_width_cm": np.float32(w) if w is not None else np.nan,
        "dim_height_cm": np.float32(h) if h is not None else np.nan,
        "dim_volume_cm3": np.float32(v) if v is not None else np.nan,
        "has_dimensions": np.int8(1 if l is not None else 0),
        "density_proxy": np.float32(density),
        "aspect_ratio": np.float32(aspect),
        "power_watts": np.float32(power) if power is not None else np.nan,
        "has_power": np.int8(1 if power is not None else 0),
        "voltage_volts": np.float32(voltage) if voltage is not None else np.nan,
        "battery_mah": np.float32(battery) if battery is not None else np.nan,
        "storage_gb": np.float32(storage) if storage is not None else np.nan,
        "has_storage": np.int8(1 if storage is not None else 0),
        "percentage_spec": np.float32(pct) if pct is not None else np.nan,
        "has_model_code": np.int8(hm),
        "model_code_count": np.int32(mc),
        "max_model_code_len": np.int32(ml),
        "is_tech_spec": np.int8(is_tech),
    }


# ---------------------------------------------------------------------------
# Text Surface Statistics
# ---------------------------------------------------------------------------

def compute_text_surface_stats(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Vectorized calculation of text lengths, token counts, and densities."""
    str_series = series.fillna("").astype(str)
    char_len = str_series.str.len()
    word_count = str_series.str.split().str.len()

    digits_count = str_series.str.count(r"\d")
    upper_count = str_series.str.count(r"[A-Z]")
    delimiters_count = str_series.str.count(r"[|/+\-*]")

    digit_ratio = (digits_count / np.maximum(char_len, 1.0)).astype(np.float32)
    upper_ratio = (upper_count / np.maximum(char_len, 1.0)).astype(np.float32)

    return pd.DataFrame({
        f"{prefix}_char_len": char_len.astype(np.int32),
        f"{prefix}_word_count": word_count.astype(np.int32),
        f"{prefix}_digit_ratio": digit_ratio,
        f"{prefix}_upper_ratio": upper_ratio,
        f"{prefix}_delimiters_count": delimiters_count.astype(np.int32),
        f"{prefix}_is_empty": (char_len == 0).astype(np.int8),
    }, index=series.index)


# ---------------------------------------------------------------------------
# Fold-Safe Out-Of-Fold Bayesian Target Encoder
# ---------------------------------------------------------------------------

class FoldSafeTargetEncoder(BaseEstimator, TransformerMixin):
    """Bayesian target encoder whose OOF path fails closed.

    ``fit`` learns full-training maps for inference. ``fit_transform`` produces
    out-of-fold values by default and accepts ``y`` separately, as sklearn does.
    ``transform`` is inference-only unless ``is_train_oof=True`` is explicit.
    """

    def __init__(
        self,
        columns: Sequence[str],
        target_column: str,
        fold_column: str = "fold",
        m: float = 10.0,
        transform_target: str = "log1p",
    ):
        self.columns = tuple(columns)
        self.target_column = target_column
        self.fold_column = fold_column
        self.m = float(m)
        self.transform_target = transform_target
        self.global_mean_: float | None = None
        self.encoding_maps_: dict[str, dict[Any, float]] = {}
        self.fitted_columns_: tuple[str, ...] = ()

    def _validate_params(self) -> None:
        if not self.columns:
            raise ValueError("At least one target-encoding column is required.")
        if not math.isfinite(self.m) or self.m < 0:
            raise ValueError("m must be a finite non-negative smoothing value.")
        if self.transform_target not in {"log1p", "none"}:
            raise ValueError("transform_target must be 'log1p' or 'none'.")

    @staticmethod
    def _coerce_target(X: pd.DataFrame, y: Any) -> pd.Series:
        if y is None:
            raise ValueError("Target values are required for target encoding.")
        if len(y) != len(X):
            raise ValueError(f"Target length {len(y)} does not match X length {len(X)}.")
        values = pd.to_numeric(pd.Series(np.asarray(y), index=X.index), errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError("Target values must be numeric, finite, and non-missing.")
        return values.astype(np.float64)

    @staticmethod
    def _categories(X: pd.DataFrame, column: str) -> pd.Series:
        if column not in X.columns:
            return pd.Series(MISSING_CATEGORY, index=X.index, dtype="object")
        return X[column].astype("object").where(X[column].notna(), MISSING_CATEGORY)

    def _transform_y(self, y: pd.Series) -> pd.Series:
        if self.transform_target == "log1p":
            if (y < 0).any():
                raise ValueError("log1p target encoding requires non-negative targets.")
            return np.log1p(y)
        return y.astype(np.float64)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FoldSafeTargetEncoder:
        self._validate_params()
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X must be a non-empty pandas DataFrame.")
        source_y = X[self.target_column] if y is None and self.target_column in X.columns else y
        y_target = self._coerce_target(X, source_y)

        y_trans = self._transform_y(y_target)
        self.global_mean_ = float(y_trans.mean())

        self.encoding_maps_ = {}
        self.fitted_columns_ = tuple(col for col in self.columns if col in X.columns)
        if not self.fitted_columns_:
            raise ValueError(f"None of the configured columns are present: {list(self.columns)}")
        for col in self.fitted_columns_:
            cats = self._categories(X, col)
            grouped = pd.DataFrame({"cat": cats.to_numpy(), "target": y_trans.to_numpy()}).groupby(
                "cat", dropna=False
            )
            counts = grouped["target"].count()
            means = grouped["target"].mean()
            smoothed = (counts * means + self.m * self.global_mean_) / (counts + self.m)
            self.encoding_maps_[col] = smoothed.to_dict()

        return self

    def _require_fitted(self) -> None:
        if self.global_mean_ is None or not self.fitted_columns_:
            raise NotFittedError("FoldSafeTargetEncoder is not fitted.")

    def transform_oof(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Create strict OOF encodings; missing or invalid folds raise an error."""
        self._require_fitted()
        if self.fold_column not in X.columns:
            raise ValueError(
                f"OOF target encoding requires fold column '{self.fold_column}'."
            )
        source_y = X[self.target_column] if y is None and self.target_column in X.columns else y
        y_target = self._coerce_target(X, source_y)
        y_trans = self._transform_y(y_target)
        folds = X[self.fold_column]
        if folds.isna().any():
            raise ValueError("Fold assignments contain missing values.")
        unique_folds = list(pd.unique(folds))
        if len(unique_folds) < 2:
            raise ValueError("OOF target encoding requires at least two folds.")
        out = pd.DataFrame(index=X.index)
        fold_values = folds.to_numpy()
        for col in self.fitted_columns_:
            encoded = np.full(len(X), float(self.global_mean_), dtype=np.float32)
            cats = self._categories(X, col).to_numpy()
            for fold in unique_folds:
                train_mask = fold_values != fold
                val_mask = fold_values == fold
                if not train_mask.any() or not val_mask.any():
                    raise ValueError(f"Fold {fold!r} has an empty train or validation partition.")
                train_targets = y_trans.to_numpy()[train_mask]
                f_global = float(train_targets.mean())
                sub_grp = pd.DataFrame(
                    {"cat": cats[train_mask], "target": train_targets}
                ).groupby("cat", dropna=False)["target"]
                counts = sub_grp.count()
                means = sub_grp.mean()
                mapping = ((counts * means + self.m * f_global) / (counts + self.m)).to_dict()
                encoded[val_mask] = [mapping.get(cat, f_global) for cat in cats[val_mask]]
            out[f"te_{col}"] = encoded
        return out

    def transform(self, X: pd.DataFrame, is_train_oof: bool | None = False) -> pd.DataFrame:
        """Transform inference rows, or explicitly request strict OOF values."""
        self._require_fitted()
        if is_train_oof:
            return self.transform_oof(X)
        out = pd.DataFrame(index=X.index)
        for col in self.fitted_columns_:
            mapping = self.encoding_maps_.get(col, {})
            categories = self._categories(X, col)
            encoded_vals = categories.map(mapping).fillna(float(self.global_mean_)).astype(np.float32)
            out[f"te_{col}"] = encoded_vals
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
        is_train_oof: bool = True,
        **fit_params: Any,
    ) -> pd.DataFrame:
        self.fit(X, y=y)
        if is_train_oof:
            source_y = X[self.target_column] if y is None and self.target_column in X.columns else y
            return self.transform_oof(X, y=source_y)
        return self.transform(X, is_train_oof=False)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        self._require_fitted()
        return np.asarray([f"te_{col}" for col in self.fitted_columns_], dtype=object)


# ---------------------------------------------------------------------------
# High-Level Catalog Feature Extractor (Scikit-Learn Transformer)
# ---------------------------------------------------------------------------

def _extract_chunk(
    df_chunk: pd.DataFrame,
    title_col: str,
    pack_col: str,
    desc_col: str,
) -> pd.DataFrame:
    """Extract one multiprocessing chunk into a typed, indexed frame."""
    rows: list[dict[str, Any]] = []
    titles = df_chunk[title_col].values if title_col in df_chunk.columns else [None] * len(df_chunk)
    packs = df_chunk[pack_col].values if pack_col in df_chunk.columns else [None] * len(df_chunk)
    descs = df_chunk[desc_col].values if desc_col in df_chunk.columns else [None] * len(df_chunk)

    for t, p, d in zip(titles, packs, descs):
        rows.append(extract_row_physical_specs(t, p, d))
    return pd.DataFrame(rows, index=df_chunk.index)


def extract_deterministic_domain_features(
    df: pd.DataFrame,
    title_column: str = "TITLE",
    pack_column: str = "PACK_SIZE",
    desc_column: str = "DESCRIPTION",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Vectorized/chunked extraction of deterministic domain rules for any DataFrame.

    Extracts explicit pack counts, unit/total weights, unit/total volumes, linear dimensions,
    power/voltage/battery specs, model code flags, and physical density proxies.
    Returns a clean numeric DataFrame aligned with df.index ready for GBDT consumption.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)
    n_rows = len(df)
    if n_jobs != 1 and n_rows > 5000:
        workers = max(1, joblib.effective_n_jobs(n_jobs))
        chunk_count = min(n_rows, workers * 4)
        chunk_size = math.ceil(n_rows / chunk_count)
        chunks = [df.iloc[i : i + chunk_size] for i in range(0, n_rows, chunk_size)]
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_extract_chunk)(chunk, title_column, pack_column, desc_column)
            for chunk in chunks
        )
        return pd.concat(results, axis=0).reindex(df.index)
    return _extract_chunk(df, title_column, pack_column, desc_column)


@dataclass
class CatalogFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts dense, leak-safe catalog features from raw text and entity fields."""

    title_column: str = "TITLE"
    desc_column: str = "DESCRIPTION"
    pack_column: str = "PACK_SIZE"
    brand_column: str = "BRAND"
    category_column: str = "CATEGORY"
    target_column: str | None = None
    fold_column: str = "fold"
    enable_target_encoding: bool = True
    te_smoothing: float = 10.0
    impute_missing: bool = False
    n_jobs: int = 1
    feature_names_: list[str] = field(default_factory=list, init=False)
    target_encoder_: FoldSafeTargetEncoder | None = field(default=None, init=False)
    imputer_medians_: dict[str, float] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CatalogFeatureExtractor:
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X must be a non-empty pandas DataFrame.")
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError("n_jobs must be a non-zero integer.")
        if not math.isfinite(self.te_smoothing) or self.te_smoothing < 0:
            raise ValueError("te_smoothing must be finite and non-negative.")
        self.target_encoder_ = None
        if self.enable_target_encoding and (self.target_column or y is not None):
            cols_to_encode = [c for c in [self.brand_column, self.category_column] if c in X.columns]
            if cols_to_encode:
                self.target_encoder_ = FoldSafeTargetEncoder(
                    columns=cols_to_encode,
                    target_column=self.target_column or "PRICE",
                    fold_column=self.fold_column,
                    m=self.te_smoothing,
                )
                self.target_encoder_.fit(X, y=y)

        fitted_features = self._base_features(X)
        if self.target_encoder_ is not None:
            fitted_features = pd.concat(
                [fitted_features, self.target_encoder_.transform(X)], axis=1
            )
        self.feature_names_ = fitted_features.columns.tolist()
        self.imputer_medians_ = {}
        if self.impute_missing:
            numeric = fitted_features.select_dtypes(include=[np.number])
            self.imputer_medians_ = numeric.median().fillna(0.0).astype(float).to_dict()
        self.is_fitted_ = True
        return self

    def _base_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract target-independent features with a stable base schema."""
        n_rows = len(X)
        title_s = X[self.title_column] if self.title_column in X.columns else pd.Series([None] * n_rows, index=X.index)
        desc_s = X[self.desc_column] if self.desc_column in X.columns else pd.Series([None] * n_rows, index=X.index)
        pack_s = X[self.pack_column] if self.pack_column in X.columns else pd.Series([None] * n_rows, index=X.index)
        brand_s = X[self.brand_column] if self.brand_column in X.columns else pd.Series([None] * n_rows, index=X.index)

        # Batch extraction (single-thread or joblib parallel)
        if self.n_jobs != 1 and n_rows > 5000:
            workers = max(1, joblib.effective_n_jobs(self.n_jobs))
            # A few chunks per worker balances skew without creating thousands
            # of tiny Python objects. Workers return DataFrames, not list[dict].
            chunk_count = min(n_rows, workers * 4)
            chunk_size = math.ceil(n_rows / chunk_count)
            chunks = [X.iloc[i:i + chunk_size] for i in range(0, n_rows, chunk_size)]
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(_extract_chunk)(chunk, self.title_column, self.pack_column, self.desc_column)
                for chunk in chunks
            )
            specs_df = pd.concat(results, axis=0).reindex(X.index)
        else:
            specs_df = _extract_chunk(X, self.title_column, self.pack_column, self.desc_column)

        # Text surface stats
        title_stats = compute_text_surface_stats(title_s, prefix="title")
        desc_stats = compute_text_surface_stats(desc_s, prefix="desc")
        features = pd.concat([specs_df, title_stats, desc_stats], axis=1)

        # Always emit these columns, even when the raw source column is absent.
        features["is_missing_desc"] = desc_s.isna().astype(np.int8)
        features["is_missing_pack_size"] = pack_s.isna().astype(np.int8)
        features["is_missing_brand"] = brand_s.isna().astype(np.int8)
        return features

    def _apply_imputation(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.impute_missing:
            for col in self.feature_names_:
                if col in features.columns and features[col].isna().any():
                    features[col] = features[col].fillna(self.imputer_medians_.get(col, 0.0))
        return features

    def transform(self, X: pd.DataFrame, is_train_oof: bool | None = False) -> pd.DataFrame:
        """Transform rows to the exact ordered schema established during fit."""
        if not self.is_fitted_:
            raise NotFittedError("CatalogFeatureExtractor is not fitted.")
        features = self._base_features(X)

        if self.target_encoder_ is not None:
            te_df = self.target_encoder_.transform(X, is_train_oof=bool(is_train_oof))
            features = pd.concat([features, te_df], axis=1)
        features = features.reindex(columns=self.feature_names_)
        return self._apply_imputation(features)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, is_train_oof: bool = True) -> pd.DataFrame:
        self.fit(X, y=y)
        if not is_train_oof or self.target_encoder_ is None:
            return self.transform(X, is_train_oof=False)
        features = self._base_features(X)
        source_y = X[self.target_column] if y is None and self.target_column in X.columns else y
        te_df = self.target_encoder_.transform_oof(X, y=source_y)
        features = pd.concat([features, te_df], axis=1).reindex(columns=self.feature_names_)
        return self._apply_imputation(features)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not self.is_fitted_:
            raise NotFittedError("CatalogFeatureExtractor is not fitted.")
        return np.asarray(self.feature_names_, dtype=object)


DeterministicDomainExtractor = CatalogFeatureExtractor


# ---------------------------------------------------------------------------
# CLI Command Runner
# ---------------------------------------------------------------------------

def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Catalog input must be CSV or Parquet.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract catalog features with disambiguated unit/pack parsing and fold-safe target encoding.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-test", required=True, type=Path)
    parser.add_argument("--target-column", default="PRICE")
    parser.add_argument("--id-column", default="PRODUCT_ID")
    parser.add_argument("--title-column", default="TITLE")
    parser.add_argument("--desc-column", default="DESCRIPTION")
    parser.add_argument("--pack-column", default="PACK_SIZE")
    parser.add_argument("--brand-column", default="BRAND")
    parser.add_argument("--category-column", default="CATEGORY")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--te-smoothing", type=float, default=10.0)
    parser.add_argument("--disable-target-encoding", action="store_true")
    parser.add_argument("--impute-missing", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    start_time = time.time()

    print(f"Loading train from: {args.train}")
    train_df = _load_table(args.train)
    print(f"Loading test from: {args.test}")
    test_df = _load_table(args.test)

    for name, frame in (("train", train_df), ("test", test_df)):
        if frame.empty:
            raise ValueError(f"{name} data is empty.")
        if args.id_column not in frame.columns:
            raise ValueError(f"{name} is missing ID column '{args.id_column}'.")
        if frame[args.id_column].isna().any() or not frame[args.id_column].is_unique:
            raise ValueError(f"{name} IDs must be non-missing and unique.")
    if args.target_column not in train_df.columns:
        raise ValueError(f"train is missing target column '{args.target_column}'.")
    if args.target_column in test_df.columns:
        raise ValueError(f"test must not contain target column '{args.target_column}'.")

    if not args.disable_target_encoding and (not args.folds or not args.folds.exists()):
        raise ValueError("Leak-safe target encoding requires an existing --folds file.")
    if args.folds and args.folds.exists():
        print(f"Attaching folds from: {args.folds}")
        folds_df = pd.read_parquet(args.folds) if args.folds.suffix.lower() == ".parquet" else pd.read_csv(args.folds)
        required = {args.id_column, args.fold_column}
        missing = required - set(folds_df.columns)
        if missing:
            raise ValueError(f"Folds file is missing required columns: {sorted(missing)}")
        if folds_df[args.id_column].isna().any() or not folds_df[args.id_column].is_unique:
            raise ValueError("Fold IDs must be non-missing and unique.")
        if args.fold_column in train_df.columns:
            comparison = train_df[[args.id_column, args.fold_column]].merge(
                folds_df[[args.id_column, args.fold_column]],
                on=args.id_column,
                how="left",
                validate="one_to_one",
                suffixes=("_train", "_file"),
                sort=False,
            )
            left = comparison[f"{args.fold_column}_train"]
            right = comparison[f"{args.fold_column}_file"]
            if right.isna().any() or not np.array_equal(left.to_numpy(), right.to_numpy()):
                raise ValueError("Existing training folds disagree with the supplied folds file.")
            train_df = train_df.drop(columns=[args.fold_column])
        original_order = train_df[args.id_column].copy()
        train_df = train_df.merge(
            folds_df[[args.id_column, args.fold_column]],
            on=args.id_column,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if not train_df[args.id_column].reset_index(drop=True).equals(original_order.reset_index(drop=True)):
            raise RuntimeError("Fold merge changed training row order.")
        if train_df[args.fold_column].isna().any():
            raise ValueError("Some training IDs have no fold assignment.")

    extractor = CatalogFeatureExtractor(
        title_column=args.title_column,
        desc_column=args.desc_column,
        pack_column=args.pack_column,
        brand_column=args.brand_column,
        category_column=args.category_column,
        target_column=args.target_column,
        fold_column=args.fold_column,
        enable_target_encoding=not args.disable_target_encoding,
        te_smoothing=args.te_smoothing,
        impute_missing=args.impute_missing,
        n_jobs=args.n_jobs,
    )

    print("Extracting features on train (strictly out-of-fold target encoding)...")
    train_feats = extractor.fit_transform(train_df, is_train_oof=True)

    print("Extracting features on test...")
    test_feats = extractor.transform(test_df, is_train_oof=False)

    if list(train_feats.columns) != list(test_feats.columns):
        raise RuntimeError("Train/test feature schemas differ after extraction.")

    if args.id_column in train_df.columns:
        train_feats.insert(0, args.id_column, train_df[args.id_column].values)
        test_feats.insert(0, args.id_column, test_df[args.id_column].values)
    if args.fold_column in train_df.columns:
        train_feats.insert(1, args.fold_column, train_df[args.fold_column].values)

    print(f"Saving train features to: {args.output_train}")
    _atomic_parquet(train_feats, args.output_train)

    print(f"Saving test features to: {args.output_test}")
    _atomic_parquet(test_feats, args.output_test)

    elapsed = time.time() - start_time
    manifest = {
        "features_version": FEATURES_VERSION,
        "train_source_sha256": _sha256(args.train),
        "test_source_sha256": _sha256(args.test),
        "folds_source_sha256": _sha256(args.folds) if args.folds else None,
        "train_rows": len(train_feats),
        "test_rows": len(test_feats),
        "feature_count": train_feats.shape[1],
        "feature_names": train_feats.columns.tolist(),
        "target_encoding_enabled": not args.disable_target_encoding,
        "target_encoding_smoothing": args.te_smoothing,
        "impute_missing": args.impute_missing,
        "elapsed_seconds": round(elapsed, 2),
        "coverage": {
            "weight_coverage_train_pct": float((train_feats["has_weight"] == 1).mean() * 100),
            "volume_coverage_train_pct": float((train_feats["has_volume"] == 1).mean() * 100),
            "multipack_pct_train": float((train_feats["is_multipack"] == 1).mean() * 100),
            "model_code_pct_train": float((train_feats["has_model_code"] == 1).mean() * 100),
            "tech_spec_pct_train": float((train_feats["is_tech_spec"] == 1).mean() * 100),
        }
    }

    manifest_path = args.output_train.parent / "features_manifest.json"
    _atomic_json(manifest, manifest_path)

    print("\n--- Refined Feature Engineering Complete ---")
    print(f"Train features: {train_feats.shape} | Test features: {test_feats.shape}")
    print(f"Weight recovery: {manifest['coverage']['weight_coverage_train_pct']:.1f}%")
    print(f"Volume recovery: {manifest['coverage']['volume_coverage_train_pct']:.1f}%")
    print(f"Multipack recovery: {manifest['coverage']['multipack_pct_train']:.1f}%")
    print(f"Tech specs recovery: {manifest['coverage']['tech_spec_pct_train']:.1f}%")
    print(f"Saved manifest: {manifest_path} (Elapsed: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
