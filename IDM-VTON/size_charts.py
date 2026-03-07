"""
FitFusion — Size Chart Database
================================
Standard garment measurements per size for computing resize ratios.

Two tiers:
  1. Generic charts (UPPER_BODY_CHART, LOWER_BODY_CHART, DRESS_CHART) —
     ISO/ASTM-based fallback values used when no brand is specified.
  2. Brand-specific charts (SNAG_*, US_*) — real measured garment dimensions
     sourced from data/measurements/garments.json. Used when brand= is passed
     to get_garment_dimensions() or compute_size_ratio().

Usage:
    from size_charts import get_garment_dimensions, compute_size_ratio

    # Generic
    dims = get_garment_dimensions("jacket", "XL")

    # Brand-specific (uses real Snag Tights Borg Jacket measurements)
    dims = get_garment_dimensions("jacket", "E1", brand="snag_tights")
    ratio = compute_size_ratio("jacket", "E1", "C1", brand="snag_tights")
    # → {"width_ratio": 1.20, "length_ratio": 1.06, ...}
"""


# ════════════════════════════════════════════════════════════════════
#  Generic Garment Measurements (in cm)
#  Source: International sizing standards (ISO, ASTM)
# ════════════════════════════════════════════════════════════════════

# Width at chest/bust level for upper-body garments
UPPER_BODY_CHART = {
    #           chest_width  shoulder_width  sleeve_length  body_length
    "4XS":  {"chest": 80,  "shoulder": 36, "sleeve": 58, "length": 58},
    "3XS":  {"chest": 84,  "shoulder": 37, "sleeve": 59, "length": 59},
    "2XS":  {"chest": 88,  "shoulder": 38, "sleeve": 60, "length": 60},
    "XS":   {"chest": 92,  "shoulder": 39, "sleeve": 61, "length": 61},
    "S":    {"chest": 96,  "shoulder": 40, "sleeve": 62, "length": 62},
    "M":    {"chest": 100, "shoulder": 42, "sleeve": 63, "length": 63},
    "L":    {"chest": 106, "shoulder": 44, "sleeve": 64, "length": 64},
    "XL":   {"chest": 112, "shoulder": 46, "sleeve": 65, "length": 65},
    "2XL":  {"chest": 118, "shoulder": 48, "sleeve": 66, "length": 66},
    "3XL":  {"chest": 124, "shoulder": 50, "sleeve": 67, "length": 67},
    "4XL":  {"chest": 130, "shoulder": 52, "sleeve": 68, "length": 68},
}


# ════════════════════════════════════════════════════════════════════
#  Brand-Specific Garment Measurements (in cm)
#  Source: data/measurements/garments.json (real brand measurements)
# ════════════════════════════════════════════════════════════════════

# Snag Tights — Cropped Borg Aviator Jacket
# D1(L), E1(XL), G2(3XL), H2(4XL) are real brand measurements.
# S, M, 2XL are interpolated (jacket not produced in those sizes).
SNAG_UPPER_BODY_CHART = {
    "XS":  {"chest": 88,  "shoulder": 38, "sleeve": 57, "length": 48},
    "S":   {"chest": 94,  "shoulder": 40, "sleeve": 58, "length": 50},
    "M":   {"chest": 100, "shoulder": 42, "sleeve": 59, "length": 51},
    "L":   {"chest": 108, "shoulder": 44, "sleeve": 60, "length": 52},  # D1
    "XL":  {"chest": 120, "shoulder": 46, "sleeve": 62, "length": 54},  # E1
    "2XL": {"chest": 128, "shoulder": 48, "sleeve": 63, "length": 55},
    "3XL": {"chest": 134, "shoulder": 50, "sleeve": 64, "length": 56},  # G2
    "4XL": {"chest": 144, "shoulder": 52, "sleeve": 66, "length": 58},  # H2
}

# Snag Tights — Classic Tights (A–H sizes, all real brand measurements)
SNAG_LOWER_BODY_CHART = {
    "XS":  {"waist": 60,  "hip": 84,  "inseam": 68, "outseam": 96},   # A
    "S":   {"waist": 68,  "hip": 92,  "inseam": 69, "outseam": 98},   # B
    "M":   {"waist": 76,  "hip": 100, "inseam": 70, "outseam": 100},  # C
    "L":   {"waist": 84,  "hip": 108, "inseam": 70, "outseam": 101},  # D
    "XL":  {"waist": 94,  "hip": 118, "inseam": 71, "outseam": 103},  # E
    "2XL": {"waist": 104, "hip": 128, "inseam": 72, "outseam": 105},  # F
    "3XL": {"waist": 114, "hip": 138, "inseam": 72, "outseam": 106},  # G
    "4XL": {"waist": 124, "hip": 148, "inseam": 73, "outseam": 108},  # H
}

# Universal Standard — Geneva Classic Top (all real brand measurements)
US_UPPER_BODY_CHART = {
    "XS":  {"chest": 86,  "shoulder": 38, "sleeve": 22, "length": 68},
    "S":   {"chest": 92,  "shoulder": 40, "sleeve": 22, "length": 69},
    "M":   {"chest": 98,  "shoulder": 42, "sleeve": 23, "length": 70},
    "L":   {"chest": 106, "shoulder": 44, "sleeve": 23, "length": 72},
    "XL":  {"chest": 114, "shoulder": 46, "sleeve": 24, "length": 74},
    "2XL": {"chest": 122, "shoulder": 48, "sleeve": 24, "length": 76},
    "3XL": {"chest": 130, "shoulder": 50, "sleeve": 25, "length": 78},
    "4XL": {"chest": 140, "shoulder": 52, "sleeve": 25, "length": 80},
}

# brand name (lowercased, underscored) → garment_type → chart
BRAND_CHARTS = {
    "snag_tights": {
        "jacket": SNAG_UPPER_BODY_CHART,
        "top":    SNAG_UPPER_BODY_CHART,
        "shirt":  SNAG_UPPER_BODY_CHART,
        "tights":   SNAG_LOWER_BODY_CHART,
        "leggings": SNAG_LOWER_BODY_CHART,
    },
    "universal_standard": {
        "top":    US_UPPER_BODY_CHART,
        "shirt":  US_UPPER_BODY_CHART,
        "blouse": US_UPPER_BODY_CHART,
    },
}

# Width at hip/waist for lower-body garments
LOWER_BODY_CHART = {
    #           waist    hip     inseam   outseam
    "4XS":  {"waist": 60, "hip": 84,  "inseam": 76, "outseam": 100},
    "3XS":  {"waist": 64, "hip": 88,  "inseam": 76, "outseam": 100},
    "2XS":  {"waist": 68, "hip": 92,  "inseam": 77, "outseam": 101},
    "XS":   {"waist": 72, "hip": 96,  "inseam": 77, "outseam": 101},
    "S":    {"waist": 76, "hip": 100, "inseam": 78, "outseam": 102},
    "M":    {"waist": 80, "hip": 104, "inseam": 78, "outseam": 102},
    "L":    {"waist": 86, "hip": 110, "inseam": 79, "outseam": 103},
    "XL":   {"waist": 92, "hip": 116, "inseam": 79, "outseam": 103},
    "2XL":  {"waist": 100,"hip": 124, "inseam": 80, "outseam": 104},
    "3XL":  {"waist": 108,"hip": 132, "inseam": 80, "outseam": 104},
    "4XL":  {"waist": 116,"hip": 140, "inseam": 81, "outseam": 105},
}

# Full-body garments (dresses, bodysuits)
DRESS_CHART = {
    #           bust    waist   hip     length
    "4XS":  {"bust": 80,  "waist": 64, "hip": 86,  "length": 90},
    "3XS":  {"bust": 84,  "waist": 68, "hip": 90,  "length": 91},
    "2XS":  {"bust": 88,  "waist": 72, "hip": 94,  "length": 92},
    "XS":   {"bust": 92,  "waist": 74, "hip": 98,  "length": 93},
    "S":    {"bust": 96,  "waist": 78, "hip": 102, "length": 94},
    "M":    {"bust": 100, "waist": 82, "hip": 106, "length": 95},
    "L":    {"bust": 106, "waist": 88, "hip": 112, "length": 96},
    "XL":   {"bust": 112, "waist": 94, "hip": 118, "length": 97},
    "2XL":  {"bust": 118, "waist":102, "hip": 126, "length": 98},
    "3XL":  {"bust": 124, "waist":110, "hip": 134, "length": 99},
    "4XL":  {"bust": 130, "waist":118, "hip": 142, "length":100},
}


# Garment type → chart lookup
GARMENT_CHARTS = {
    "top": UPPER_BODY_CHART,
    "shirt": UPPER_BODY_CHART,
    "t-shirt": UPPER_BODY_CHART,
    "tee": UPPER_BODY_CHART,
    "hoodie": UPPER_BODY_CHART,
    "sweater": UPPER_BODY_CHART,
    "jacket": UPPER_BODY_CHART,
    "coat": UPPER_BODY_CHART,
    "blouse": UPPER_BODY_CHART,
    "pants": LOWER_BODY_CHART,
    "jeans": LOWER_BODY_CHART,
    "trousers": LOWER_BODY_CHART,
    "shorts": LOWER_BODY_CHART,
    "skirt": LOWER_BODY_CHART,
    "tights": LOWER_BODY_CHART,
    "leggings": LOWER_BODY_CHART,
    "dress": DRESS_CHART,
    "bodysuit": DRESS_CHART,
}


def get_garment_dimensions(garment_type: str, size_label: str, brand: str = None) -> dict:
    """
    Look up garment measurements for a given type and size.

    Args:
        garment_type: e.g. "top", "jacket", "tights"
        size_label:   standard (e.g. "XL") or brand-specific (e.g. "E1", "UK14")
        brand:        optional brand key, e.g. "snag_tights", "universal_standard"
                      When supplied, brand-specific real measurements are used.

    Returns:
        dict of measurements in cm
    """
    normalized = normalize_size_label(size_label)
    gtype = garment_type.lower()

    # Brand-specific chart takes priority
    if brand:
        brand_key = brand.lower().replace(" ", "_")
        brand_chart = BRAND_CHARTS.get(brand_key, {}).get(gtype)
        if brand_chart:
            return brand_chart.get(normalized, brand_chart.get("M", {}))

    # Fallback: generic ISO/ASTM chart
    chart = GARMENT_CHARTS.get(gtype, UPPER_BODY_CHART)
    return chart.get(normalized, chart.get("M", {}))


def compute_size_ratio(
    garment_type: str,
    garment_size: str,
    person_size: str,
    brand: str = None,
) -> dict:
    """
    Compute the ratio between garment dimensions and the person's expected
    body dimensions at their size.

    Args:
        garment_type: e.g. "jacket", "tights"
        garment_size: size of the garment (e.g. "E1", "XL")
        person_size:  size of the person (e.g. "C1", "M")
        brand:        optional brand key for real measurements
                      (e.g. "snag_tights", "universal_standard")

    Returns:
        dict with "width_ratio" and "length_ratio"
        - > 1.0 means garment is LARGER than person (loose fit)
        - < 1.0 means garment is SMALLER than person (tight fit)
        - = 1.0 means exact fit
    """
    garment_dims = get_garment_dimensions(garment_type, normalize_size_label(garment_size), brand=brand)
    person_dims = get_garment_dimensions(garment_type, normalize_size_label(person_size), brand=brand)
    
    if not garment_dims or not person_dims:
        return {"width_ratio": 1.0, "length_ratio": 1.0}
    
    # Pick the primary width and length keys based on garment type
    gtype = garment_type.lower()
    
    if gtype in ("pants", "jeans", "trousers", "shorts", "skirt", "tights", "leggings"):
        width_key = "hip"
        length_key = "outseam"
    elif gtype in ("dress", "bodysuit"):
        width_key = "bust"
        length_key = "length"
    else:
        # Upper body default
        width_key = "chest"
        length_key = "length"
    
    garment_width = garment_dims.get(width_key, 100)
    person_width = person_dims.get(width_key, 100)
    garment_length = garment_dims.get(length_key, 65)
    person_length = person_dims.get(length_key, 65)
    
    width_ratio = garment_width / max(person_width, 1)
    length_ratio = garment_length / max(person_length, 1)
    
    return {
        "width_ratio": round(width_ratio, 4),
        "length_ratio": round(length_ratio, 4),
        "garment_width_cm": garment_width,
        "person_width_cm": person_width,
        "garment_length_cm": garment_length,
        "person_length_cm": person_length,
        "size_gap": _size_to_idx(garment_size) - _size_to_idx(person_size),
    }


def normalize_size_label(size_label: str) -> str:
    """
    Normalize brand-specific size labels to standard XS-4XL.
    
    Handles:
        - Standard: XS, S, M, L, XL, 2XL, 3XL, 4XL (pass-through)
        - Snag Tights: A1→XS, B1→S, C1→M, D1→L, E1→XL, F2→2XL, G2→3XL, H2→4XL
        - Numeric: 0→4XS, 2→2XS, 4→S, 6→L, 8→2XL, 10→4XL
        - UK: 6→XS, 8→S, 10→M, 12→L, 14→XL, 16→2XL, 18→3XL, 20→4XL
        - EU: 34→XS, 36→S, 38→M, 40→L, 42→XL, 44→2XL
    """
    s = size_label.strip().upper()
    
    # Already standard
    standard = {"4XS","3XS","2XS","XS","S","M","L","XL","2XL","3XL","4XL"}
    if s in standard:
        return s
    
    # Snag Tights letter-number system
    snag_map = {
        "A": "XS", "A1": "XS", "A2": "XS",
        "B": "S",  "B1": "S",  "B2": "S",
        "C": "M",  "C1": "M",  "C2": "M",
        "D": "L",  "D1": "L",  "D2": "L",
        "E": "XL", "E1": "XL", "E2": "XL",
        "F": "2XL","F1": "2XL","F2": "2XL",
        "G": "3XL","G1": "3XL","G2": "3XL",
        "H": "4XL","H1": "4XL","H2": "4XL",
    }
    if s in snag_map:
        return snag_map[s]
    
    # Numeric sizes (US/generic)
    numeric_map = {
        "0": "4XS", "2": "3XS", "4": "2XS", "6": "XS", "8": "S",
        "10": "M", "12": "L", "14": "XL", "16": "2XL", "18": "3XL", "20": "4XL",
    }
    if s in numeric_map:
        return numeric_map[s]
    
    # UK sizes
    uk_map = {
        "UK6": "XS", "UK8": "S", "UK10": "M", "UK12": "L",
        "UK14": "XL", "UK16": "2XL", "UK18": "3XL", "UK20": "4XL",
    }
    if s in uk_map:
        return uk_map[s]
    
    # EU sizes
    eu_map = {
        "EU34": "XS", "EU36": "S", "EU38": "M", "EU40": "L",
        "EU42": "XL", "EU44": "2XL", "EU46": "3XL", "EU48": "4XL",
    }
    if s in eu_map:
        return eu_map[s]
    
    # Fallback: return M if completely unknown
    return "M"


def _size_to_idx(size_label: str) -> int:
    """Convert size label to numeric index, normalizing first."""
    normalized = normalize_size_label(size_label)
    mapping = {
        "4XS": 0, "3XS": 1, "2XS": 2, "XS": 3, "S": 4,
        "M": 5, "L": 6, "XL": 7, "2XL": 8, "3XL": 9, "4XL": 10,
    }
    return mapping.get(normalized, 5)


if __name__ == "__main__":
    print("Generic size ratio examples (ISO/ASTM chart):")
    for gs in ["S", "M", "L", "XL", "2XL"]:
        ratio = compute_size_ratio("top", gs, "M")
        print(f"  {gs} shirt on M person: width={ratio['width_ratio']:.2f}x, "
              f"length={ratio['length_ratio']:.2f}x, gap={ratio['size_gap']:+d}")

    print("\nSnag Tights Borg Jacket — real brand measurements:")
    for gs, ps in [("D1","C1"), ("E1","C1"), ("G2","C1"), ("H2","C1")]:
        ratio = compute_size_ratio("jacket", gs, ps, brand="snag_tights")
        print(f"  Jacket {gs} on model {ps}: width={ratio['width_ratio']:.2f}x "
              f"(garment {ratio['garment_width_cm']}cm / person {ratio['person_width_cm']}cm), "
              f"gap={ratio['size_gap']:+d}")

    print("\nUniversal Standard Geneva Top — real brand measurements:")
    for gs in ["M", "XL", "2XL", "4XL"]:
        ratio = compute_size_ratio("top", gs, "M", brand="universal_standard")
        print(f"  Geneva Top {gs} on M model: width={ratio['width_ratio']:.2f}x, "
              f"length={ratio['length_ratio']:.2f}x")

    print("\nBrand size normalization:")
    for brand_size in ["A1", "C1", "D1", "E1", "G2", "H2", "10", "UK14", "EU42"]:
        print(f"  {brand_size:>5} → {normalize_size_label(brand_size)}")
