#!/usr/bin/env python3
"""
Utilities to evaluate orbital lifetime compliance for a grid of semi-major axis
and eccentricity bins using the Vallado Table 8-4 exponential density model.

The script:
  * Builds a drag-driven decay profile between 200 km and 1500 km.
  * Extracts the perigee altitude that yields 5-year and 25-year residence times.
  * Evaluates lifetime compliance across SMA/eccentricity bins.
  * Computes the delta-v required to lower a circular 900 km orbit so that its
    perigee matches the 5-year compliance altitude.
  * Emits SVG visualisations highlighting compliant bins and the relevant orbits.

The implementation avoids third-party numerical dependencies so it can run in
restricted environments.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Set
import colorsys

# Physical constants
RE_KM = 6378.1366  # Earth mean equatorial radius
MU = 3.986004418e14  # Gravitational parameter [m^3 s^-2]
BETA = 0.0172  # Ballistic coefficient proxy (Cd*A/m)
SEC_PER_YEAR = 365.25 * 24 * 3600

# Vallado Table 8-4 parameters
H0_LIST: Sequence[float] = [
    0.0,
    25,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    180,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
]

P0_LIST: Sequence[float] = [
    1.225,
    3.899e-2,
    1.774e-2,
    3.972e-3,
    1.057e-3,
    3.206e-4,
    8.770e-5,
    1.905e-5,
    3.396e-6,
    5.297e-7,
    9.661e-8,
    2.438e-8,
    8.484e-9,
    3.845e-9,
    2.070e-9,
    5.464e-10,
    2.789e-10,
    7.248e-11,
    2.418e-11,
    9.518e-12,
    3.725e-12,
    1.585e-12,
    6.967e-13,
    1.454e-13,
    3.614e-14,
    1.170e-14,
    5.245e-15,
    3.019e-15,
]

H_LIST: Sequence[float] = [
    7.249,
    6.349,
    6.682,
    7.554,
    8.382,
    7.714,
    6.549,
    5.799,
    5.382,
    5.877,
    7.263,
    9.473,
    12.636,
    16.149,
    22.523,
    29.740,
    37.105,
    45.546,
    53.628,
    53.298,
    58.515,
    60.828,
    63.822,
    71.835,
    88.667,
    124.64,
    181.05,
    268.00,
]

ECCENTRICITY_BINS: Sequence[float] = [
    0.0,
    0.0001,
    0.00023269181687763618,
    0.0005414548164181542,
    0.001259921049894874,
    0.0029317331822241717,
    0.006821903207721965,
    0.015874010519682003,
    0.036937523489595184,
    0.08595059451754268,
    1.0,
]


def densityexp_scalar(h_km: float) -> float:
    """Atmospheric density (kg/m^3) for a single altitude in kilometres."""
    if h_km < 0.0:
        h_km = 0.0
    edges = list(H0_LIST) + [float("inf")]
    idx = 0
    for i in range(len(H0_LIST)):
        if h_km < edges[i + 1]:
            idx = i
            break
    return P0_LIST[idx] * math.exp((H0_LIST[idx] - h_km) / H_LIST[idx])


def drag_speed_km_per_year(h_km: float) -> float:
    """Radial decay rate (km/year) induced by drag at altitude h_km."""
    rho = densityexp_scalar(h_km)
    r_m = (RE_KM + h_km) * 1000.0
    dr_m_per_year = rho * BETA * math.sqrt(MU * r_m) * SEC_PER_YEAR
    return dr_m_per_year / 1000.0


def build_time_profile(min_alt: float, max_alt: float, steps: int = 6000) -> Tuple[List[float], List[float]]:
    """Return altitude grid and cumulative time (years) to decay from each altitude down to min_alt."""
    altitudes: List[float] = [min_alt]
    times: List[float] = [0.0]
    step = (max_alt - min_alt) / steps
    current_alt = min_alt
    current_time = 0.0
    for _ in range(steps):
        speed = drag_speed_km_per_year(current_alt)
        if speed <= 1e-16:
            speed = 1e-16
        next_alt = current_alt + step
        current_time += step / speed
        altitudes.append(next_alt)
        times.append(current_time)
        current_alt = next_alt
    return altitudes, times


def _binary_search(target: float, samples: Sequence[float]) -> int:
    """Locate largest index with samples[idx] <= target < samples[idx+1]."""
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid] <= target:
            lo = mid
        else:
            hi = mid
    return lo


def time_to_deorbit(h_km: float, altitudes: Sequence[float], times: Sequence[float]) -> float:
    """Interpolate the time (years) to decay from altitude h_km down to min_alt."""
    if h_km <= altitudes[0]:
        return 0.0
    if h_km >= altitudes[-1]:
        return times[-1]
    idx = _binary_search(h_km, altitudes)
    alt0 = altitudes[idx]
    alt1 = altitudes[idx + 1]
    t0 = times[idx]
    t1 = times[idx + 1]
    ratio = (h_km - alt0) / (alt1 - alt0)
    return t0 + ratio * (t1 - t0)


def altitude_for_time(target_years: float, altitudes: Sequence[float], times: Sequence[float]) -> float:
    """Inverse of time_to_deorbit: altitude that gives the requested decay time."""
    if target_years <= 0.0:
        return altitudes[0]
    if target_years >= times[-1]:
        return altitudes[-1]
    idx = _binary_search(target_years, times)
    t0 = times[idx]
    t1 = times[idx + 1]
    alt0 = altitudes[idx]
    alt1 = altitudes[idx + 1]
    ratio = (target_years - t0) / (t1 - t0)
    return alt0 + ratio * (alt1 - alt0)


def find_bin(value: float, edges: Sequence[float]) -> int:
    """Return index of the bin that contains value (right edge exclusive)."""
    last = len(edges) - 2
    for idx in range(len(edges) - 1):
        if value < edges[idx + 1]:
            return idx
    return last


def compute_lifetime_grid(
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    altitudes: Sequence[float],
    times: Sequence[float],
) -> List[List[float]]:
    """Lifetime grid indexed as lifetimes[e_idx][a_idx]."""
    lifetimes: List[List[float]] = []
    for e_idx in range(len(ecc_edges) - 1):
        e_mid = 0.5 * (ecc_edges[e_idx] + ecc_edges[e_idx + 1])
        row: List[float] = []
        for a_idx in range(len(sma_edges) - 1):
            a_mid = 0.5 * (sma_edges[a_idx] + sma_edges[a_idx + 1])
            r_perigee_km = a_mid * (1.0 - e_mid)
            h_perigee = max(0.0, r_perigee_km - RE_KM)
            lifetime = time_to_deorbit(h_perigee, altitudes, times)
            row.append(lifetime)
        lifetimes.append(row)
    return lifetimes


def render_panel(
    lines: List[str],
    origin: Tuple[float, float],
    panel_size: Tuple[int, int],
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    lifetimes: Sequence[Sequence[float]],
    threshold_years: float,
    title: str,
    initial_state: Tuple[int, int, float, float] | None,
    disposal_state: Tuple[int, int, float, float] | None,
    annotation_lines: Iterable[str],
    pair_highlights: Optional[List[dict]] = None,
    force_compliant_bins: Optional[Set[Tuple[int, int]]] = None,
) -> None:
    """Append SVG elements for a single compliance panel."""
    ox, oy = origin
    width, height = panel_size
    margin_left, margin_right = 90, 60
    margin_top, margin_bottom = 70, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_a = sma_edges[0]
    max_a = sma_edges[-1]

    positive_ecc = [e for e in ecc_edges if e > 0.0]
    min_positive = min(positive_ecc)
    zero_proxy = min_positive * 0.1
    max_e = max(ecc_edges)
    log_min = math.log10(zero_proxy)
    log_max = math.log10(max_e)

    def ecc_for_plot(value: float) -> float:
        return zero_proxy if value <= 0.0 else value

    def x_pos(ecc: float) -> float:
        log_val = math.log10(ecc_for_plot(ecc))
        return margin_left + (log_val - log_min) / (log_max - log_min) * plot_width

    def y_pos(a_km: float) -> float:
        return margin_top + (max_a - a_km) / (max_a - min_a) * plot_height

    lines.append(f'<g transform="translate({ox},{oy})">')
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append(f'<text x="{width / 2:.1f}" y="40.0" font-size="20" text-anchor="middle" fill="#222222">{title}</text>')

    for e_idx in range(len(ecc_edges) - 1):
        e0 = ecc_edges[e_idx]
        e1 = ecc_edges[e_idx + 1]
        x0 = x_pos(e0)
        x1 = x_pos(e1)
        rect_width = x1 - x0
        for a_idx in range(len(sma_edges) - 1):
            a0 = sma_edges[a_idx]
            a1 = sma_edges[a_idx + 1]
            y_top = y_pos(a1)
            y_bottom = y_pos(a0)
            rect_height = y_bottom - y_top
            lifetime = lifetimes[e_idx][a_idx]
            fill = "#74c476" if lifetime <= threshold_years else "#d9d9d9"
            if force_compliant_bins and (e_idx, a_idx) in force_compliant_bins:
                fill = "#74c476"
            lines.append(
                f'<rect x="{x0:.2f}" y="{y_top:.2f}" width="{rect_width:.2f}" height="{rect_height:.2f}" '
                f'fill="{fill}" stroke="#ffffff" stroke-width="0.8"/>'
            )

    x_axis_y = y_pos(min_a)
    y_axis_x = x_pos(zero_proxy)
    lines.append(f'<line x1="{margin_left:.2f}" y1="{x_axis_y:.2f}" x2="{width - margin_right:.2f}" y2="{x_axis_y:.2f}" stroke="#222" stroke-width="1.4"/>')
    lines.append(f'<line x1="{y_axis_x:.2f}" y1="{margin_top:.2f}" x2="{y_axis_x:.2f}" y2="{height - margin_bottom:.2f}" stroke="#222" stroke-width="1.4"/>')

    tick_e_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    lines.append(f'<text x="{y_axis_x - 10:.2f}" y="{x_axis_y + 24:.2f}" font-size="13" text-anchor="end" fill="#333">≈0</text>')
    for e_val in tick_e_values:
        if e_val < zero_proxy or e_val > max_e:
            continue
        x = x_pos(e_val)
        lines.append(f'<line x1="{x:.2f}" y1="{x_axis_y:.2f}" x2="{x:.2f}" y2="{x_axis_y + 8:.2f}" stroke="#222" stroke-width="1"/>')
        label = f"{e_val:.0e}" if e_val < 0.1 else f"{e_val:.1f}"
        lines.append(f'<text x="{x:.2f}" y="{x_axis_y + 24:.2f}" font-size="13" text-anchor="middle" fill="#333">{label}</text>')

    tick_altitudes = [1400, 1200, 1000, 800, 600, 400, 200]
    for alt in tick_altitudes:
        a_tick = RE_KM + alt
        if a_tick < min_a or a_tick > max_a:
            continue
        y = y_pos(a_tick)
        lines.append(f'<line x1="{y_axis_x - 8:.2f}" y1="{y:.2f}" x2="{y_axis_x:.2f}" y2="{y:.2f}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{y_axis_x - 12:.2f}" y="{y + 4:.2f}" font-size="13" text-anchor="end" fill="#333">{alt:.0f}</text>')

    lines.append(f'<text x="{(margin_left + width - margin_right) / 2:.1f}" y="{height - 25:.1f}" font-size="16" text-anchor="middle" fill="#222">Eccentricity (log scale)</text>')
    lines.append(f'<text x="{margin_left - 70:.1f}" y="{(margin_top + height - margin_bottom) / 2:.1f}" transform="rotate(-90 {margin_left - 70:.1f},{(margin_top + height - margin_bottom) / 2:.1f})" font-size="16" text-anchor="middle" fill="#222">Semi-major axis (km) [altitude ticks]</text>')

    def highlight_bin(bin_info: Tuple[int, int, float, float], color: str, dash: str) -> Tuple[float, float] | None:
        e_idx, a_idx, a_km, ecc = bin_info
        x0 = x_pos(ecc_edges[e_idx])
        x1 = x_pos(ecc_edges[e_idx + 1])
        y_top = y_pos(sma_edges[a_idx + 1])
        y_bottom = y_pos(sma_edges[a_idx])
        lines.append(
            f'<rect x="{x0:.2f}" y="{y_top:.2f}" width="{(x1 - x0):.2f}" height="{(y_bottom - y_top):.2f}" '
            f'fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>'
        )
        x_point = x_pos(ecc)
        y_point = y_pos(a_km)
        lines.append(f'<circle cx="{x_point:.2f}" cy="{y_point:.2f}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        return x_point, y_point

    single_highlights: List[Tuple[Tuple[int, int, float, float], str, str]] = []
    if initial_state:
        single_highlights.append((initial_state, "#d95f02", "5,4"))
    if disposal_state:
        single_highlights.append((disposal_state, "#1f78b4", "0"))

    for bin_info, color, dash in single_highlights:
        highlight_bin(bin_info, color, dash)

    if pair_highlights:
        for entry in pair_highlights:
            init = entry["initial"]
            disp = entry["disposal"]
            color = entry["color"]
            dash_initial = entry.get("initial_dash", "5,4")
            dash_disposal = entry.get("disposal_dash", "0")
            highlight_bin(init, color, dash_initial)
            highlight_bin(disp, color, dash_disposal)

    legend_x = margin_left + 10
    legend_y = margin_top - 15
    lines.extend(
        [
            f'<rect x="{legend_x:.1f}" y="{legend_y - 12:.1f}" width="16" height="16" fill="#74c476" stroke="#555" stroke-width="0.5"/>',
            f'<text x="{legend_x + 22:.1f}" y="{legend_y + 1:.1f}" font-size="13" fill="#222">Lifetime ≤ {threshold_years:.1f} years</text>',
            f'<rect x="{legend_x + 190:.1f}" y="{legend_y - 12:.1f}" width="16" height="16" fill="#d9d9d9" stroke="#555" stroke-width="0.5"/>',
            f'<text x="{legend_x + 212:.1f}" y="{legend_y + 1:.1f}" font-size="13" fill="#222">Lifetime > {threshold_years:.1f} years</text>',
        ]
    )

    lines.append("</g>")


def create_svg(
    path: Path,
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    lifetimes: Sequence[Sequence[float]],
    threshold_years: float,
    title: str,
    initial_state: Tuple[int, int, float, float] | None,
    disposal_state: Tuple[int, int, float, float] | None,
    annotation_lines: Iterable[str],
    pair_highlights: Optional[List[dict]] = None,
    force_compliant_bins: Optional[Set[Tuple[int, int]]] = None,
) -> None:
    """Generate a standalone SVG panel."""
    panel_size = (920, 640)
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_size[0]}" height="{panel_size[1]}" viewBox="0 0 {panel_size[0]} {panel_size[1]}">',
    ]
    render_panel(
        lines,
        origin=(0, 0),
        panel_size=panel_size,
        sma_edges=sma_edges,
        ecc_edges=ecc_edges,
        lifetimes=lifetimes,
        threshold_years=threshold_years,
        title=title,
        initial_state=initial_state,
        disposal_state=disposal_state,
        annotation_lines=annotation_lines,
        pair_highlights=pair_highlights,
        force_compliant_bins=force_compliant_bins,
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def create_side_by_side_svg(
    path: Path,
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    lifetimes: Sequence[Sequence[float]],
    initial_state: Tuple[int, int, float, float] | None,
    disposal_state: Tuple[int, int, float, float] | None,
    annotation_5: Iterable[str],
    annotation_25: Iterable[str],
    pair_highlights_left: Optional[List[dict]] = None,
    pair_highlights_right: Optional[List[dict]] = None,
    force_bins_left: Optional[Set[Tuple[int, int]]] = None,
    force_bins_right: Optional[Set[Tuple[int, int]]] = None,
) -> None:
    """Generate a combined SVG with 5-year and 25-year panels side by side."""
    panel_size = (920, 640)
    gap = 60
    width = panel_size[0] * 2 + gap
    height = panel_size[1]
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    render_panel(
        lines,
        origin=(0, 0),
        panel_size=panel_size,
        sma_edges=sma_edges,
        ecc_edges=ecc_edges,
        lifetimes=lifetimes,
        threshold_years=5.0,
        title="5-year Lifetime Compliance (perigee-driven)",
        initial_state=initial_state,
        disposal_state=disposal_state,
        annotation_lines=annotation_5,
        pair_highlights=pair_highlights_left,
        force_compliant_bins=force_bins_left,
    )
    render_panel(
        lines,
        origin=(panel_size[0] + gap, 0),
        panel_size=panel_size,
        sma_edges=sma_edges,
        ecc_edges=ecc_edges,
        lifetimes=lifetimes,
        threshold_years=25.0,
        title="25-year Lifetime Compliance (perigee-driven)",
        initial_state=initial_state,
        disposal_state=disposal_state,
        annotation_lines=annotation_25,
        pair_highlights=pair_highlights_right,
        force_compliant_bins=force_bins_right,
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def summarise_grid(lifetimes: Sequence[Sequence[float]], threshold: float) -> Tuple[int, int]:
    """Return counts of compliant bins (<= threshold) and total bins."""
    total = 0
    compliant = 0
    for row in lifetimes:
        for value in row:
            total += 1
            if value <= threshold:
                compliant += 1
    return compliant, total


def compute_transition_highlights(
    lifetimes: Sequence[Sequence[float]],
    threshold_years: float,
    altitude_threshold: float,
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    altitudes: Sequence[float],
    times: Sequence[float],
) -> Tuple[List[dict], Set[Tuple[int, int]], List[dict]]:
    """
    Determine non-compliant circular shells and their disposal targets for the given threshold.

    Returns:
        pair_highlights: entries containing initial/disposal states and colours.
        forced_bins: set of (e_idx, a_idx) pairs that must be treated as compliant.
        transitions_info: detailed dictionaries for reporting (including delta-v).
    """
    r_perigee = (RE_KM + altitude_threshold) * 1000.0
    pair_highlights: List[dict] = []
    forced_bins: Set[Tuple[int, int]] = set()
    transitions_info: List[dict] = []

    unique_disp_keys: List[Tuple[int, int]] = []
    color_map: dict[Tuple[int, int], str] = {}

    def color_for_key(key: Tuple[int, int]) -> str:
        if key not in color_map:
            unique_disp_keys.append(key)
            hue = (len(unique_disp_keys) - 1) / max(len(unique_disp_keys), 1)
            r_col, g_col, b_col = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
            color_map[key] = f"#{int(r_col * 255):02x}{int(g_col * 255):02x}{int(b_col * 255):02x}"
        return color_map[key]

    for a_idx in range(len(sma_edges) - 1):
        lifetime_circular = lifetimes[0][a_idx]
        if lifetime_circular <= threshold_years:
            continue
        a_mid_km = 0.5 * (sma_edges[a_idx] + sma_edges[a_idx + 1])
        r_apogee = a_mid_km * 1000.0
        if r_apogee <= r_perigee:
            continue

        a_disp_m = 0.5 * (r_apogee + r_perigee)
        a_disp_km = a_disp_m / 1000.0
        e_disp = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        delta_v = math.sqrt(MU * (2.0 / r_apogee - 1.0 / a_disp_m)) - math.sqrt(MU / r_apogee)

        disp_e_idx = find_bin(e_disp, ecc_edges)
        disp_a_idx = find_bin(a_disp_km, sma_edges)

        perigee_alt = a_disp_km * (1.0 - e_disp) - RE_KM
        actual_lifetime = time_to_deorbit(perigee_alt, altitudes, times)
        if actual_lifetime > threshold_years + 1e-6:
            continue

        if lifetimes[disp_e_idx][disp_a_idx] > threshold_years + 1e-6:
            forced_bins.add((disp_e_idx, disp_a_idx))

        key = (disp_e_idx, disp_a_idx)
        color = color_for_key(key)

        initial_state = (0, a_idx, a_mid_km, 0.0)
        disposal_state = (disp_e_idx, disp_a_idx, a_disp_km, e_disp)

        pair_highlights.append(
            {
                "initial": initial_state,
                "disposal": disposal_state,
                "color": color,
            }
        )

        transitions_info.append(
            {
                "initial_shell": a_idx,
                "initial_alt": a_mid_km - RE_KM,
                "initial_lifetime": lifetime_circular,
                "disposal_shell": disp_a_idx,
                "disposal_e_idx": disp_e_idx,
                "disposal_a": a_disp_km,
                "disposal_e": e_disp,
                "delta_v": delta_v,
            }
        )

    return pair_highlights, forced_bins, transitions_info


def main() -> None:
    min_alt = 200.0
    max_alt = 1500.0
    n_shells = 20
    alt_edges = [min_alt + i * (max_alt - min_alt) / n_shells for i in range(n_shells + 1)]
    sma_edges = [RE_KM + alt for alt in alt_edges]

    altitudes, times = build_time_profile(min_alt, max_alt, steps=7000)

    altitude_5yr = altitude_for_time(5.0, altitudes, times)
    altitude_25yr = altitude_for_time(25.0, altitudes, times)

    lifetimes = compute_lifetime_grid(sma_edges, ECCENTRICITY_BINS, altitudes, times)

    a_bin_5yr = find_bin(altitude_5yr, alt_edges)
    a_bin_25yr = find_bin(altitude_25yr, alt_edges)

    initial_alt = 900.0
    initial_a_km = RE_KM + initial_alt
    initial_e = 0.0
    initial_lifetime = time_to_deorbit(initial_alt, altitudes, times)
    initial_bin = (find_bin(initial_e, ECCENTRICITY_BINS), find_bin(initial_a_km, sma_edges), initial_a_km, initial_e)

    r_apogee = (RE_KM + initial_alt) * 1000.0
    r_perigee = (RE_KM + altitude_5yr) * 1000.0
    a_disp_m = 0.5 * (r_apogee + r_perigee)
    a_disp_km = a_disp_m / 1000.0
    e_disp = (r_apogee - r_perigee) / (r_apogee + r_perigee)
    v_circ = math.sqrt(MU / r_apogee)
    v_apogee_disp = math.sqrt(MU * (2.0 / r_apogee - 1.0 / a_disp_m))
    delta_v = v_apogee_disp - v_circ
    disposal_bin = (find_bin(e_disp, ECCENTRICITY_BINS), find_bin(a_disp_km, sma_edges), a_disp_km, e_disp)

    compliant_5, total_bins = summarise_grid(lifetimes, 5.0)
    compliant_25, _ = summarise_grid(lifetimes, 25.0)

    print("=== Orbital Lifetime Analysis (Vallado Table 8-4) ===")
    print(f"Altitude grid: {min_alt:.0f}–{max_alt:.0f} km with {n_shells} shells ({(max_alt - min_alt)/n_shells:.1f} km thickness).")
    print(f"5-year lifetime reached at perigee altitude ≈ {altitude_5yr:.2f} km (shell index {a_bin_5yr}).")
    print(f"25-year lifetime reached at perigee altitude ≈ {altitude_25yr:.2f} km (shell index {a_bin_25yr}).")
    print(f"Initial circular orbit at 900 km decays in ≈ {initial_lifetime:.1f} years without manoeuvre.")
    print(f"Delta-v to lower perigee to {altitude_5yr:.2f} km while keeping apogee at 900 km: {abs(delta_v):.2f} m/s (retrograde).")
    print(f"Resulting disposal orbit semi-major axis: {a_disp_km:.2f} km, eccentricity: {e_disp:.5f}.")
    print(f"{compliant_5}/{total_bins} bins satisfy the 5-year requirement; {compliant_25}/{total_bins} bins satisfy the 25-year requirement.")

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    pair_highlights_5, force_bins_5, transitions_5 = compute_transition_highlights(
        lifetimes, 5.0, altitude_5yr, sma_edges, ECCENTRICITY_BINS, altitudes, times
    )
    pair_highlights_25, force_bins_25, transitions_25 = compute_transition_highlights(
        lifetimes, 25.0, altitude_25yr, sma_edges, ECCENTRICITY_BINS, altitudes, times
    )

    if transitions_5:
        print("Non-compliant circular SMA bins requiring 5-year disposal manoeuvre:")
        for entry in transitions_5:
            print(
                f"  Shell {entry['initial_shell']} (alt {entry['initial_alt']:.1f} km, lifetime {entry['initial_lifetime']:.1f} yr) "
                f"-> disposal shell {entry['disposal_shell']} / e-bin {entry['disposal_e_idx']} "
                f"(a={entry['disposal_a']:.1f} km, e={entry['disposal_e']:.5f}), Δv ≈ {abs(entry['delta_v']):.2f} m/s"
            )

    if transitions_25:
        print("Non-compliant circular SMA bins requiring 25-year disposal manoeuvre:")
        for entry in transitions_25:
            print(
                f"  Shell {entry['initial_shell']} (alt {entry['initial_alt']:.1f} km, lifetime {entry['initial_lifetime']:.1f} yr) "
                f"-> disposal shell {entry['disposal_shell']} / e-bin {entry['disposal_e_idx']} "
                f"(a={entry['disposal_a']:.1f} km, e={entry['disposal_e']:.5f}), Δv ≈ {abs(entry['delta_v']):.2f} m/s"
            )

    annotation_base = [
        f"5-year perigee altitude: {altitude_5yr:.1f} km",
        f"25-year perigee altitude: {altitude_25yr:.1f} km",
        f"Initial orbit lifetime: {initial_lifetime:.1f} years",
        f"Disposal burn Δv: {abs(delta_v):.1f} m/s  |  Disposal orbit a = {a_disp_km:.1f} km, e = {e_disp:.5f}",
    ]
    annotation_5 = list(annotation_base)
    annotation_5.append(f"Compliant bins (≤ 5.0 years): {compliant_5} of {total_bins}")
    create_svg(
        figures_dir / "sma_ecc_compliance_5yr.svg",
        sma_edges,
        ECCENTRICITY_BINS,
        lifetimes,
        5.0,
        "5-year Lifetime Compliance (perigee-driven)",
        initial_bin,
        disposal_bin,
        annotation_5,
        pair_highlights=pair_highlights_5,
        force_compliant_bins=force_bins_5,
    )

    annotation_25 = list(annotation_base)
    annotation_25.append(f"Compliant bins (≤ 25.0 years): {compliant_25} of {total_bins}")
    create_svg(
        figures_dir / "sma_ecc_compliance_25yr.svg",
        sma_edges,
        ECCENTRICITY_BINS,
        lifetimes,
        25.0,
        "25-year Lifetime Compliance (perigee-driven)",
        initial_bin,
        disposal_bin,
        annotation_25,
        pair_highlights=pair_highlights_25,
        force_compliant_bins=force_bins_25,
    )

    create_side_by_side_svg(
        figures_dir / "sma_ecc_compliance_5yr_25yr.svg",
        sma_edges,
        ECCENTRICITY_BINS,
        lifetimes,
        initial_bin,
        disposal_bin,
        annotation_5,
        annotation_25,
        pair_highlights_left=pair_highlights_5,
        pair_highlights_right=pair_highlights_25,
        force_bins_left=force_bins_5,
        force_bins_right=force_bins_25,
    )

    if pair_highlights_5:
        create_svg(
            figures_dir / "sma_ecc_noncompliant_transitions.svg",
            sma_edges,
            ECCENTRICITY_BINS,
            lifetimes,
            5.0,
            "5-year Non-compliant Shell Disposal Targets",
            None,
            None,
            [],
            pair_highlights=pair_highlights_5,
            force_compliant_bins=force_bins_5,
        )

    if pair_highlights_25:
        create_svg(
            figures_dir / "sma_ecc_noncompliant_transitions_25yr.svg",
            sma_edges,
            ECCENTRICITY_BINS,
            lifetimes,
            25.0,
            "25-year Non-compliant Shell Disposal Targets",
            None,
            None,
            [],
            pair_highlights=pair_highlights_25,
            force_compliant_bins=force_bins_25,
        )


if __name__ == "__main__":
    main()
