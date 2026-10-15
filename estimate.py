"""
Estimate industrial coating work from an uploaded image.

This version avoids generic image-caption labeling and instead uses:
1. Manual profiles for known reference images
2. Previous-work matches from the bearing catalog
3. Visual condition heuristics (rust / already-coated / complexity)
4. Explicit labor + material equations with a minimum job charge
"""
import math
import os

import numpy as np
from PIL import Image

from catalog_metadata import REFERENCE_IMAGE_METADATA, lookup_reference_profile, normalize_project_label
from clip_onnx import to_rgb


COATING_TYPES = {
    "epoxy_industrial": {
        "name": "Industrial Epoxy System",
        "coverage_sqft_per_gallon": 225,
        "primer_coverage_sqft_per_gallon": 240,
        "topcoat_gallon_cost": 145.0,
        "primer_gallon_cost": 110.0,
        "cure_time_hours": 24,
        "durability_years": 12,
    },
    "urethane_heavy_duty": {
        "name": "Heavy-Duty Urethane System",
        "coverage_sqft_per_gallon": 210,
        "primer_coverage_sqft_per_gallon": 235,
        "topcoat_gallon_cost": 185.0,
        "primer_gallon_cost": 125.0,
        "cure_time_hours": 36,
        "durability_years": 18,
    },
    "zinc_primer": {
        "name": "Zinc-Rich Primer + Urethane Topcoat",
        "coverage_sqft_per_gallon": 190,
        "primer_coverage_sqft_per_gallon": 205,
        "topcoat_gallon_cost": 210.0,
        "primer_gallon_cost": 160.0,
        "cure_time_hours": 48,
        "durability_years": 22,
    },
}


COMPONENT_PROFILES = {
    "pillow_block": {
        "label": "Pillow Block Bearing",
        "default_width": 10.5,
        "default_height": 7.0,
        "default_depth": 5.5,
        "width_range": (6.0, 18.0),
        "height_range": (4.0, 12.0),
        "depth_range": (4.0, 10.0),
        "coating_key": "urethane_heavy_duty",
        "prep_hours_per_sqft": 0.42,
        "prime_hours_per_sqft": 0.18,
        "finish_hours_per_sqft": 0.28,
        "setup_hours": 1.1,
        "finish_coats": 2,
        "labor_rate": 95.0,
        "geometry": "pillow_block",
        "min_job_charge": 425.0,
        "consumables_per_sqft": 4.0,
    },
    "flange": {
        "label": "Flange Assembly",
        "default_width": 14.0,
        "default_height": 10.0,
        "default_depth": 4.0,
        "width_range": (8.0, 24.0),
        "height_range": (6.0, 18.0),
        "depth_range": (2.0, 8.0),
        "coating_key": "zinc_primer",
        "prep_hours_per_sqft": 0.34,
        "prime_hours_per_sqft": 0.14,
        "finish_hours_per_sqft": 0.22,
        "setup_hours": 1.0,
        "finish_coats": 2,
        "labor_rate": 95.0,
        "geometry": "flange",
        "min_job_charge": 400.0,
        "consumables_per_sqft": 3.0,
    },
    "gearbox": {
        "label": "Gearbox / Drive Housing",
        "default_width": 18.0,
        "default_height": 15.0,
        "default_depth": 12.0,
        "width_range": (10.0, 32.0),
        "height_range": (8.0, 24.0),
        "depth_range": (8.0, 20.0),
        "coating_key": "urethane_heavy_duty",
        "prep_hours_per_sqft": 0.46,
        "prime_hours_per_sqft": 0.18,
        "finish_hours_per_sqft": 0.30,
        "setup_hours": 1.4,
        "finish_coats": 2,
        "labor_rate": 98.0,
        "geometry": "gearbox",
        "min_job_charge": 575.0,
        "consumables_per_sqft": 4.8,
    },
    "bearing_housing": {
        "label": "Bearing Housing / Support Base",
        "default_width": 16.0,
        "default_height": 14.0,
        "default_depth": 8.0,
        "width_range": (10.0, 28.0),
        "height_range": (8.0, 24.0),
        "depth_range": (5.0, 14.0),
        "coating_key": "urethane_heavy_duty",
        "prep_hours_per_sqft": 0.38,
        "prime_hours_per_sqft": 0.16,
        "finish_hours_per_sqft": 0.24,
        "setup_hours": 1.2,
        "finish_coats": 2,
        "labor_rate": 95.0,
        "geometry": "housing",
        "min_job_charge": 450.0,
        "consumables_per_sqft": 3.5,
    },
    "support_pedestal": {
        "label": "Support Pedestal / Mount Base",
        "default_width": 20.0,
        "default_height": 18.0,
        "default_depth": 12.0,
        "width_range": (12.0, 36.0),
        "height_range": (10.0, 28.0),
        "depth_range": (8.0, 20.0),
        "coating_key": "zinc_primer",
        "prep_hours_per_sqft": 0.36,
        "prime_hours_per_sqft": 0.16,
        "finish_hours_per_sqft": 0.24,
        "setup_hours": 1.35,
        "finish_coats": 2,
        "labor_rate": 95.0,
        "geometry": "housing",
        "min_job_charge": 525.0,
        "consumables_per_sqft": 3.8,
    },
}


MANUAL_REFERENCE_PROFILES = REFERENCE_IMAGE_METADATA


def _clamp(value, low, high):
    return max(low, min(high, value))


def normalize_component_type(value):
    return normalize_project_label(value)


def _safe_caption(image_path):
    try:
        from caption import get_caption

        return get_caption(image_path, max_len=120) or "Industrial component"
    except Exception:
        return "Industrial component"


def analyze_visual_features(image_path, rows_per_strip=256):
    """Measure visual cues that help condition and scale estimation.

    The photo is processed in horizontal strips so a 12-megapixel upload stays within a small
    server's memory. Each strip carries one extra row above and below, so the vertical gradient
    matches a whole-image np.gradient exactly.
    """
    img = to_rgb(Image.open(image_path))
    width, height = img.size
    edge_count = rust_count = painted_count = object_count = 0

    for top in range(0, height, rows_per_strip):
        bottom = min(top + rows_per_strip, height)
        lo, hi = max(top - 1, 0), min(bottom + 1, height)
        arr = np.asarray(img.crop((0, lo, width, hi)), dtype=np.float32) / 255.0
        gy, gx = np.gradient(arr.mean(axis=2))
        keep = slice(top - lo, top - lo + bottom - top)
        arr, gx, gy = arr[keep], gx[keep], gy[keep]
        gray = arr.mean(axis=2)
        edge_count += int(np.count_nonzero(np.sqrt(gx ** 2 + gy ** 2) > 0.10))

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        sat = arr.max(axis=2) - arr.min(axis=2)

        rust_mask = (r > 0.38) & (r > g * 1.08) & (g > b * 1.03) & (sat > 0.10)
        blue_mask = (b > 0.35) & (b > r * 1.08) & (b > g * 1.05)
        yellow_mask = (r > 0.55) & (g > 0.45) & (b < 0.30)
        white_mask = (gray > 0.72) & (sat < 0.10)
        rust_count += int(np.count_nonzero(rust_mask))
        painted_count += int(np.count_nonzero(blue_mask | yellow_mask | white_mask))
        object_count += int(np.count_nonzero((gray > 0.10) & (gray < 0.95)))

    total = float(width * height)
    edge_ratio = edge_count / total
    rust_ratio = rust_count / total
    coated_ratio = painted_count / total
    object_fill = object_count / total

    if edge_ratio > 0.12:
        complexity = "high"
    elif edge_ratio > 0.06:
        complexity = "medium"
    else:
        complexity = "low"

    return {
        "pixel_width": img.size[0],
        "pixel_height": img.size[1],
        "aspect_ratio": round(img.size[0] / max(img.size[1], 1), 3),
        "edge_ratio": round(edge_ratio, 3),
        "rust_ratio": round(rust_ratio, 3),
        "coated_ratio": round(coated_ratio, 3),
        "object_fill": round(object_fill, 3),
        "complexity": complexity,
    }


def _manual_profile_from_name(name):
    return lookup_reference_profile(name)


def infer_component_profile(image_path, previous_work=None):
    """
    Pick the component type using manual reference labels first.
    If a previous-work match exists, trust that labeled catalog image more than a caption.
    """
    direct_profile = _manual_profile_from_name(image_path)
    if direct_profile:
        return direct_profile

    previous_work = previous_work or []
    if previous_work:
        votes = {}
        descriptions = {}
        for match in previous_work:
            matched_name = match.get("filename") or match.get("image_url") or ""
            score = float(match.get("similarity_score") or 0.0)
            manual = _manual_profile_from_name(matched_name)
            if manual:
                comp_type = manual["component_type"]
                votes[comp_type] = votes.get(comp_type, 0.0) + max(score, 0.01)
                descriptions.setdefault(comp_type, manual["description"])
                continue

            top_type = normalize_component_type(match.get("bearing_type"))
            if top_type:
                votes[top_type] = votes.get(top_type, 0.0) + max(score, 0.01)
                descriptions.setdefault(top_type, COMPONENT_PROFILES[top_type]["label"])

        if votes:
            component_type = max(votes.items(), key=lambda item: item[1])[0]
            return {
                "component_type": component_type,
                "description": descriptions.get(component_type, "Industrial component"),
            }

    caption = _safe_caption(image_path).lower()
    if "gear" in caption or "gearbox" in caption or "drive" in caption:
        return {"component_type": "gearbox", "description": "Gearbox or drive housing"}
    if "pedestal" in caption or "column" in caption:
        return {"component_type": "support_pedestal", "description": "Support pedestal or mount base"}
    if "mount" in caption or "base" in caption or "housing" in caption:
        return {"component_type": "bearing_housing", "description": "Bearing housing or support base"}
    if "flange" in caption or "pipe" in caption or "elbow" in caption:
        return {"component_type": "flange", "description": "Pipe flange or flange-connected assembly"}
    return {"component_type": "pillow_block", "description": "Pillow block bearing assembly"}


def _size_scale_from_label(size_info):
    size_text = (size_info or "").lower()
    if "large" in size_text:
        return 1.18
    if "small" in size_text:
        return 0.84
    return 1.0


def analyze_image_dimensions(image_path, component_type, previous_work=None):
    """
    Estimate physical dimensions from a known component profile plus visual fill.
    This is still heuristic, but much more stable than raw pixel-to-inch conversion.
    """
    visual = analyze_visual_features(image_path)
    profile = COMPONENT_PROFILES[component_type]
    previous_work = previous_work or []

    width = profile["default_width"]
    height = profile["default_height"]
    depth = profile["default_depth"]

    if previous_work:
        size_scales = []
        for match in previous_work[:3]:
            size_scales.append(_size_scale_from_label(match.get("size_info")))
        if size_scales:
            anchor_scale = sum(size_scales) / len(size_scales)
            width *= anchor_scale
            height *= anchor_scale
            depth *= anchor_scale

    fill_scale = _clamp(0.85 + (visual["object_fill"] - 0.45), 0.75, 1.20)
    complexity_scale = 1.10 if visual["complexity"] == "high" else 1.0 if visual["complexity"] == "medium" else 0.92
    aspect_scale = _clamp(visual["aspect_ratio"] / (width / height), 0.82, 1.18)

    width *= fill_scale * complexity_scale * math.sqrt(aspect_scale)
    height *= fill_scale * complexity_scale / math.sqrt(aspect_scale)
    depth *= _clamp(fill_scale, 0.85, 1.15)

    width = _clamp(width, *profile["width_range"])
    height = _clamp(height, *profile["height_range"])
    depth = _clamp(depth, *profile["depth_range"])

    condition = "good"
    if visual["rust_ratio"] > 0.10:
        condition = "poor"
    elif visual["rust_ratio"] > 0.04:
        condition = "fair"
    elif visual["coated_ratio"] > 0.18:
        condition = "good_existing_coating"

    return {
        "width_inches": round(width, 1),
        "height_inches": round(height, 1),
        "depth_inches": round(depth, 1),
        "edge_ratio": visual["edge_ratio"],
        "rust_ratio": visual["rust_ratio"],
        "coated_ratio": visual["coated_ratio"],
        "complexity": visual["complexity"],
        "condition": condition,
    }


def calculate_surface_area(dims, component_type):
    """Use component-specific geometry instead of a single generic box."""
    w = dims["width_inches"]
    h = dims["height_inches"]
    d = dims["depth_inches"]

    if component_type == "pillow_block":
        housing = 2 * (w * d + h * d) * 0.72
        base = (w * d * 1.10) + (w * h * 0.55)
        area = housing + base
    elif component_type == "flange":
        radius = w / 2.0
        face = math.pi * radius * radius * 1.35
        edge = math.pi * w * max(d, 2.0)
        pipe_stub = math.pi * (max(w * 0.45, 2.0)) * max(h * 0.45, 2.0)
        area = face + edge + pipe_stub
    elif component_type == "gearbox":
        area = 2 * (w * h + w * d + h * d) * 0.88
    elif component_type == "support_pedestal":
        area = 2 * (w * h + w * d + h * d) * 0.78
    else:
        area = 2 * (w * h + w * d + h * d) * 0.74

    return round(area / 144.0, 2)


def select_coating_type(component_type, dims):
    if component_type == "flange" or dims["condition"] == "poor":
        return "zinc_primer"
    if component_type == "support_pedestal":
        return "zinc_primer"
    if component_type in {"pillow_block", "gearbox", "bearing_housing"}:
        return "urethane_heavy_duty"
    return "epoxy_industrial"


def calculate_coating_cost(surface_area_sqft, coating_key, component_type, dims):
    """
    Estimate cost from labor hours + material gallons + consumables.
    This yields a more believable quote than a single flat $/sqft multiplier.
    """
    coating = COATING_TYPES[coating_key]
    profile = COMPONENT_PROFILES[component_type]

    condition_multiplier = {
        "good": 1.00,
        "good_existing_coating": 0.92,
        "fair": 1.15,
        "poor": 1.35,
    }[dims["condition"]]

    complexity_multiplier = {"low": 0.92, "medium": 1.00, "high": 1.15}[dims["complexity"]]
    labor_rate = profile["labor_rate"]
    waste_factor = 1.18

    prep_hours = max(0.8, surface_area_sqft * profile["prep_hours_per_sqft"] * condition_multiplier)
    primer_hours = max(0.4, surface_area_sqft * profile["prime_hours_per_sqft"] * complexity_multiplier)
    finish_hours = max(
        0.7,
        surface_area_sqft * profile["finish_hours_per_sqft"] * profile["finish_coats"] * complexity_multiplier,
    )
    support_hours = profile["setup_hours"] + (0.35 if surface_area_sqft > 6 else 0.0)

    prep_cost = (prep_hours * labor_rate) + (surface_area_sqft * profile["consumables_per_sqft"])
    primer_cost = primer_hours * labor_rate
    coating_cost = finish_hours * labor_rate
    labor_cost = support_hours * labor_rate

    primer_gallons = max(
        0.2,
        (surface_area_sqft / coating["primer_coverage_sqft_per_gallon"]) * waste_factor,
    )
    topcoat_gallons = max(
        0.3,
        (surface_area_sqft * profile["finish_coats"] / coating["coverage_sqft_per_gallon"]) * waste_factor,
    )
    gallons_needed = primer_gallons + topcoat_gallons
    material_cost = (
        primer_gallons * coating["primer_gallon_cost"]
        + topcoat_gallons * coating["topcoat_gallon_cost"]
    )

    subtotal = prep_cost + primer_cost + coating_cost + labor_cost + material_cost
    if subtotal < profile["min_job_charge"]:
        labor_cost += profile["min_job_charge"] - subtotal
        subtotal = profile["min_job_charge"]

    return {
        "prep_cost": round(prep_cost, 2),
        "primer_cost": round(primer_cost, 2),
        "coating_cost": round(coating_cost, 2),
        "labor_cost": round(labor_cost, 2),
        "material_cost": round(material_cost, 2),
        "subtotal": round(subtotal, 2),
        "total": round(subtotal, 2),
        "gallons_needed": round(gallons_needed, 2),
        "cost_per_sqft": round(subtotal / max(surface_area_sqft, 0.1), 2),
        "primer_gallons": round(primer_gallons, 2),
        "topcoat_gallons": round(topcoat_gallons, 2),
    }


def get_coating_estimate(image_path, previous_work=None):
    """Return a manual-profile-driven coating estimate for the uploaded image."""
    profile_info = infer_component_profile(image_path, previous_work=previous_work)
    component_type = normalize_component_type(profile_info.get("component_type")) or "bearing_housing"
    dims = analyze_image_dimensions(image_path, component_type, previous_work=previous_work)
    surface_area = calculate_surface_area(dims, component_type)
    coating_key = select_coating_type(component_type, dims)
    coating_info = COATING_TYPES[coating_key]
    costs = calculate_coating_cost(surface_area, coating_key, component_type, dims)

    description = profile_info.get("description") or _safe_caption(image_path)
    component_label = COMPONENT_PROFILES[component_type]["label"]

    estimate_lines = [
        "═══════════════════════════════════════════════",
        "         INDUSTRIAL COATING ESTIMATE",
        "═══════════════════════════════════════════════",
        "",
        "COMPONENT ANALYSIS:",
        f"  Type: {component_label}",
        f"  Description: {description}",
        f"  Estimated Size: {dims['width_inches']}\" W x {dims['height_inches']}\" H x {dims['depth_inches']}\" D",
        f"  Condition: {dims['condition'].replace('_', ' ').title()}",
        f"  Complexity: {dims['complexity'].upper()}",
        f"  Estimated Surface Area: {surface_area} sq ft",
        "",
        "RECOMMENDED SYSTEM:",
        f"  {coating_info['name']}",
        f"  Primer Coverage: {coating_info['primer_coverage_sqft_per_gallon']} sq ft/gal",
        f"  Topcoat Coverage: {coating_info['coverage_sqft_per_gallon']} sq ft/gal",
        f"  Cure Time: {coating_info['cure_time_hours']} hours",
        f"  Expected Service Life: {coating_info['durability_years']} years",
        "",
        "COST BREAKDOWN:",
        f"  Surface Preparation:  ${costs['prep_cost']:>8.2f}",
        f"  Primer Application:   ${costs['primer_cost']:>8.2f}",
        f"  Topcoat Application:  ${costs['coating_cost']:>8.2f}",
        f"  Setup / Handling:     ${costs['labor_cost']:>8.2f}",
        f"  Materials ({costs['gallons_needed']} gal): ${costs['material_cost']:>8.2f}",
        "  ─────────────────────────────────",
        f"  TOTAL ESTIMATE:       ${costs['total']:>8.2f}",
        f"  Cost per sq ft:       ${costs['cost_per_sqft']:>8.2f}",
        "",
        "PROCESS NOTES:",
        "  • Degrease and abrade exposed metal before coating",
        "  • Spot blast or mechanically prep rusted areas as required",
        f"  • Apply primer, then {COMPONENT_PROFILES[component_type]['finish_coats']} finish coats",
        f"  • Estimated primer use: {costs['primer_gallons']} gal",
        f"  • Estimated topcoat use: {costs['topcoat_gallons']} gal",
        f"  • Allow full cure before service: {coating_info['cure_time_hours']}h",
        "",
        "This estimate uses manual component labeling plus visual condition heuristics.",
        "Use as a prototype quote; confirm final pricing after field review.",
    ]

    return {
        "description": description,
        "estimate_text": "\n".join(estimate_lines),
        "component_type": component_type,
        "dimensions": dims,
        "surface_area_sqft": surface_area,
        "coating_type": coating_info["name"],
        "costs": costs,
        "coating_details": coating_info,
    }
