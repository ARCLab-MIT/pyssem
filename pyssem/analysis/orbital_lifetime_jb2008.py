#!/usr/bin/env python3
"""
Orbital lifetime compliance analysis using the JB2008 density model for two
representative solar activity epochs (solar maximum in 2024 and solar minimum
in 2030). The study mirrors the static-density analysis but relies on the logic
of the JB2008 density interpolator and the time-dependent drag-velocity term
derived from `drag_func_exp_time_dep` (replicated here to avoid external
dependencies that are unavailable in this execution environment).

Outputs:
    * figures/jb2008_solar_cycle_compliance.svg      # 2 x 2 combined panels
    * figures/jb2008_solar_max_transitions.svg       # 5-year disposal map
    * figures/jb2008_solar_min_transitions.svg       # 5-year disposal map

The grid spans 20 altitude shells from 200 km to 1500 km with the eccentricity
bins supplied in the elliptical configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import colorsys

# Physical constants and global parameters
RE_KM = 6378.1366
MU = 3.986004418e14          # m^3 / s^2
BETA = 0.0172                # Ballistic coefficient proxy
SEC_PER_YEAR = 365.25 * 24 * 3600

ECCRANGE: Sequence[float] = [
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


# ---------------------------------------------------------------------------
# Helpers mirroring JB2008_dens_func without requiring NumPy
# ---------------------------------------------------------------------------

def load_density_data(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r") as handle:
        raw = json.load(handle)
    # Convert inner dictionaries to float values once for faster access later.
    return {date: {alt: float(val) for alt, val in entries.items()} for date, entries in raw.items()}


def sorted_altitudes(data: Dict[str, float]) -> List[float]:
    values = sorted(float(key) for key in data.keys())
    return values


def interpolate_density_for_date(
    density_data: Dict[str, Dict[str, float]],
    date_keys: Sequence[str],
    target_date: str,
    altitude_km: float,
) -> float:
    """
    JB2008 densities are tabulated in 50 km increments. Perform a log-linear
    interpolation across the altitude grid for the supplied month.
    """
    if target_date not in density_data:
        raise ValueError(f"Requested date {target_date} not available in JB2008 dataset.")

    table = density_data[target_date]
    alts = sorted_altitudes(table)

    if altitude_km <= alts[0]:
        return table[str(int(alts[0]))]
    if altitude_km >= alts[-1]:
        return table[str(int(alts[-1]))]

    # Locate bounding altitudes
    lower_idx = 0
    upper_idx = 1
    for idx in range(len(alts) - 1):
        if alts[idx] <= altitude_km <= alts[idx + 1]:
            lower_idx = idx
            upper_idx = idx + 1
            break

    alt_lower = alts[lower_idx]
    alt_upper = alts[upper_idx]
    dens_lower = table[str(int(alt_lower))]
    dens_upper = table[str(int(alt_upper))]

    # Interpolate in log space for smoother variation
    if dens_lower <= 0.0 or dens_upper <= 0.0:
        weight = (altitude_km - alt_lower) / (alt_upper - alt_lower)
        return dens_lower * (1.0 - weight) + dens_upper * weight

    log_lower = math.log(dens_lower)
    log_upper = math.log(dens_upper)
    weight = (altitude_km - alt_lower) / (alt_upper - alt_lower)
    return math.exp(log_lower * (1.0 - weight) + log_upper * weight)


# ---------------------------------------------------------------------------
# Drag dynamics helper (mirrors drag_func_exp_time_dep logic)
# ---------------------------------------------------------------------------

def relative_velocity_factor(radius_m: float) -> float:
    """
    Reproduce the radial flux term from drag_func_exp_time_dep without species
    bookkeeping. The returned value is |sqrt(mu * r)| * seconds_per_year.
    """
    return math.sqrt(MU * radius_m) * SEC_PER_YEAR


def drag_speed_km_per_year(density: float, altitude_km: float) -> float:
    radius_m = (RE_KM + altitude_km) * 1000.0
    speed_m_per_year = density * BETA * relative_velocity_factor(radius_m)
    return max(speed_m_per_year / 1000.0, 1e-16)


# ---------------------------------------------------------------------------
# Lifetime integration utilities (adapted from the static analysis)
# ---------------------------------------------------------------------------

def build_time_profile(
    density_func,
    min_alt: float,
    max_alt: float,
    steps: int = 7000,
) -> Tuple[List[float], List[float]]:
    """
    Integrate dr/dt over altitude using JB2008 densities to obtain cumulative
    residence times. The density_func returns kg/m^3 for the supplied altitude.
    """
    altitudes: List[float] = [min_alt]
    times: List[float] = [0.0]
    step_km = (max_alt - min_alt) / steps
    current_alt = min_alt
    current_time = 0.0

    for _ in range(steps):
        speed_km_per_year = drag_speed_km_per_year(density_func(current_alt), current_alt)
        dt_years = step_km / speed_km_per_year
        current_time += dt_years
        current_alt += step_km
        altitudes.append(current_alt)
        times.append(current_time)

    return altitudes, times


def _binary_search(target: float, samples: Sequence[float]) -> int:
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid] <= target:
            lo = mid
        else:
            hi = mid
    return lo


def time_to_deorbit(altitude_km: float, altitudes: Sequence[float], times: Sequence[float]) -> float:
    if altitude_km <= altitudes[0]:
        return 0.0
    if altitude_km >= altitudes[-1]:
        return times[-1]
    idx = _binary_search(altitude_km, altitudes)
    alt0 = altitudes[idx]
    alt1 = altitudes[idx + 1]
    t0 = times[idx]
    t1 = times[idx + 1]
    weight = (altitude_km - alt0) / (alt1 - alt0)
    return t0 + weight * (t1 - t0)


def altitude_for_time(target_years: float, altitudes: Sequence[float], times: Sequence[float]) -> float:
    if target_years <= 0.0:
        return altitudes[0]
    if target_years >= times[-1]:
        return altitudes[-1]
    idx = _binary_search(target_years, times)
    t0 = times[idx]
    t1 = times[idx + 1]
    alt0 = altitudes[idx]
    alt1 = altitudes[idx + 1]
    weight = (target_years - t0) / (t1 - t0)
    return alt0 + weight * (alt1 - alt0)


def find_bin(value: float, edges: Sequence[float]) -> int:
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
    grid: List[List[float]] = []
    for e_idx in range(len(ecc_edges) - 1):
        e_mid = 0.5 * (ecc_edges[e_idx] + ecc_edges[e_idx + 1])
        row: List[float] = []
        for a_idx in range(len(sma_edges) - 1):
            a_mid = 0.5 * (sma_edges[a_idx] + sma_edges[a_idx + 1])
            perigee = a_mid * (1.0 - e_mid) - RE_KM
            lifetime = time_to_deorbit(max(0.0, perigee), altitudes, times)
            row.append(lifetime)
        grid.append(row)
    return grid


# ---------------------------------------------------------------------------
# Visualization helpers (ported from the static analysis script)
# ---------------------------------------------------------------------------

def render_panel(
    lines: List[str],
    origin: Tuple[float, float],
    panel_size: Tuple[int, int],
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    lifetimes: Sequence[Sequence[float]],
    threshold_years: float,
    title: str,
    initial_state: Optional[Tuple[int, int, float, float]],
    disposal_state: Optional[Tuple[int, int, float, float]],
    pair_highlights: Optional[List[dict]],
    force_compliant_bins: Optional[Set[Tuple[int, int]]],
) -> None:
    ox, oy = origin
    width, height = panel_size
    margin_left, margin_right = 90, 60
    margin_top, margin_bottom = 70, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_a = sma_edges[0]
    max_a = sma_edges[-1]

    positive_ecc = [value for value in ecc_edges if value > 0.0]
    min_positive = min(positive_ecc)
    zero_proxy = min_positive * 0.1
    max_e = ecc_edges[-1]
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
        x_tick = x_pos(e_val)
        label = f"{e_val:.0e}" if e_val < 0.1 else f"{e_val:.1f}"
        lines.append(f'<line x1="{x_tick:.2f}" y1="{x_axis_y:.2f}" x2="{x_tick:.2f}" y2="{x_axis_y + 8:.2f}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{x_tick:.2f}" y="{x_axis_y + 24:.2f}" font-size="13" text-anchor="middle" fill="#333">{label}</text>')

    tick_altitudes = [1400, 1200, 1000, 800, 600, 400, 200]
    for alt in tick_altitudes:
        a_tick = RE_KM + alt
        if a_tick < min_a or a_tick > max_a:
            continue
        y_tick = y_pos(a_tick)
        lines.append(f'<line x1="{y_axis_x - 8:.2f}" y1="{y_tick:.2f}" x2="{y_axis_x:.2f}" y2="{y_tick:.2f}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{y_axis_x - 12:.2f}" y="{y_tick + 4:.2f}" font-size="13" text-anchor="end" fill="#333">{alt:.0f}</text>')

    lines.append(f'<text x="{(margin_left + width - margin_right) / 2:.1f}" y="{height - 25:.1f}" font-size="16" text-anchor="middle" fill="#222">Eccentricity (log scale)</text>')
    lines.append(f'<text x="{margin_left - 70:.1f}" y="{(margin_top + height - margin_bottom) / 2:.1f}" transform="rotate(-90 {margin_left - 70:.1f},{(margin_top + height - margin_bottom) / 2:.1f})" font-size="16" text-anchor="middle" fill="#222">Semi-major axis (km) [altitude ticks]</text>')

    def highlight_bin(state: Tuple[int, int, float, float], color: str, dash: str) -> None:
        e_idx, a_idx, a_km, ecc = state
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

    if initial_state:
        highlight_bin(initial_state, "#d95f02", "5,4")
    if disposal_state:
        highlight_bin(disposal_state, "#1f78b4", "0")

    if pair_highlights:
        for entry in pair_highlights:
            highlight_bin(entry["initial"], entry["color"], entry.get("initial_dash", "5,4"))
            highlight_bin(entry["disposal"], entry["color"], entry.get("disposal_dash", "0"))

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


def create_2x2_svg(
    path: Path,
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    scenarios: Sequence[dict],
) -> None:
    panel_size = (920, 640)
    gap_x = 60
    gap_y = 80
    width = panel_size[0] * 2 + gap_x
    height = panel_size[1] * 2 + gap_y

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    # Expected scenario order: [(max, 5), (max, 25), (min, 5), (min, 25)]
    for idx, info in enumerate(scenarios):
        row = idx // 2
        col = idx % 2
        origin = (col * (panel_size[0] + gap_x), row * (panel_size[1] + gap_y))
        render_panel(
            lines=lines,
            origin=origin,
            panel_size=panel_size,
            sma_edges=sma_edges,
            ecc_edges=ECCRANGE,
            lifetimes=info["lifetimes"],
            threshold_years=info["threshold"],
            title=info["title"],
            initial_state=info.get("initial_state"),
            disposal_state=info.get("disposal_state"),
            pair_highlights=info.get("pair_highlights"),
            force_compliant_bins=info.get("force_bins"),
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def create_single_panel_svg(
    path: Path,
    sma_edges: Sequence[float],
    ecc_edges: Sequence[float],
    panel_spec: dict,
) -> None:
    panel_size = (920, 640)
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_size[0]}" height="{panel_size[1]}" viewBox="0 0 {panel_size[0]} {panel_size[1]}">',
    ]
    render_panel(
        lines=lines,
        origin=(0, 0),
        panel_size=panel_size,
        sma_edges=sma_edges,
        ecc_edges=ecc_edges,
        lifetimes=panel_spec["lifetimes"],
        threshold_years=panel_spec["threshold"],
        title=panel_spec["title"],
        initial_state=panel_spec.get("initial_state"),
        disposal_state=panel_spec.get("disposal_state"),
        pair_highlights=panel_spec.get("pair_highlights"),
        force_compliant_bins=panel_spec.get("force_bins"),
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Transition analysis and presentation
# ---------------------------------------------------------------------------

def summarise_grid(lifetimes: Sequence[Sequence[float]], threshold: float) -> Tuple[int, int]:
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
    r_perigee = (RE_KM + altitude_threshold) * 1000.0
    pair_highlights: List[dict] = []
    forced_bins: Set[Tuple[int, int]] = set()
    transitions: List[dict] = []

    color_map: Dict[Tuple[int, int], str] = {}

    def color_for_key(key: Tuple[int, int]) -> str:
        if key not in color_map:
            hue = len(color_map) / max(len(color_map) + 1, 1)
            r_val, g_val, b_val = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
            color_map[key] = f"#{int(r_val * 255):02x}{int(g_val * 255):02x}{int(b_val * 255):02x}"
        return color_map[key]

    for a_idx in range(len(sma_edges) - 1):
        lifetime = lifetimes[0][a_idx]
        if lifetime <= threshold_years:
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
        actual_lifetime = time_to_deorbit(max(0.0, perigee_alt), altitudes, times)
        if actual_lifetime > threshold_years + 1e-6:
            continue

        if lifetimes[disp_e_idx][disp_a_idx] > threshold_years + 1e-6:
            forced_bins.add((disp_e_idx, disp_a_idx))

        color = color_for_key((disp_e_idx, disp_a_idx))
        pair_highlights.append(
            {
                "initial": (0, a_idx, a_mid_km, 0.0),
                "disposal": (disp_e_idx, disp_a_idx, a_disp_km, e_disp),
                "color": color,
            }
        )
        transitions.append(
            {
                "initial_shell": a_idx,
                "initial_alt": a_mid_km - RE_KM,
                "initial_lifetime": lifetime,
                "disposal_shell": disp_a_idx,
                "disposal_e_idx": disp_e_idx,
                "disposal_a": a_disp_km,
                "disposal_e": e_disp,
                "delta_v": delta_v,
            }
        )

    return pair_highlights, forced_bins, transitions


def report_transitions(label: str, transitions: List[dict]) -> None:
    if not transitions:
        print(f"No non-compliant circular shells for {label}.")
        return
    print(f"Non-compliant circular shells for {label}:")
    for entry in transitions:
        print(
            f"  Shell {entry['initial_shell']} (alt {entry['initial_alt']:.1f} km, lifetime {entry['initial_lifetime']:.1f} yr) "
            f"-> disposal shell {entry['disposal_shell']} / e-bin {entry['disposal_e_idx']} "
            f"(a={entry['disposal_a']:.1f} km, e={entry['disposal_e']:.5f}), Δv ≈ {abs(entry['delta_v']):.2f} m/s"
        )


# ---------------------------------------------------------------------------
# Scenario orchestration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    label: str
    date: str
    altitudes: List[float]
    times: List[float]
    lifetimes: List[List[float]]
    altitude_5yr: float
    altitude_25yr: float
    compliant_5: int
    compliant_25: int
    total_bins: int
    initial_lifetime: float
    initial_altitude: float
    disposal_orbit_5: Tuple[float, float, float]
    disposal_orbit_25: Tuple[float, float, float]
    pair_5: List[dict]
    pair_25: List[dict]
    force_5: Set[Tuple[int, int]]
    force_25: Set[Tuple[int, int]]
    transitions_5: List[dict]
    transitions_25: List[dict]


def analyse_scenario(
    label: str,
    date_str: str,
    density_data: Dict[str, Dict[str, float]],
    sma_edges: Sequence[float],
) -> ScenarioResult:
    def density_func(alt_km: float) -> float:
        return interpolate_density_for_date(density_data, (), date_str, alt_km)

    altitudes, times = build_time_profile(density_func, 200.0, 1500.0)
    lifetimes = compute_lifetime_grid(sma_edges, ECCRANGE, altitudes, times)

    altitude_5yr = altitude_for_time(5.0, altitudes, times)
    altitude_25yr = altitude_for_time(25.0, altitudes, times)

    compliant_5, total_bins = summarise_grid(lifetimes, 5.0)
    compliant_25, _ = summarise_grid(lifetimes, 25.0)

    initial_altitude = 900.0
    initial_a_km = RE_KM + initial_altitude
    initial_lifetime = time_to_deorbit(initial_altitude, altitudes, times)

    def disposal_orbit(target_altitude: float) -> Tuple[float, float, float]:
        r_apogee = (RE_KM + initial_altitude) * 1000.0
        r_perigee = (RE_KM + target_altitude) * 1000.0
        a_disp_m = 0.5 * (r_apogee + r_perigee)
        a_disp_km = a_disp_m / 1000.0
        e_disp = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        delta_v = math.sqrt(MU * (2.0 / r_apogee - 1.0 / a_disp_m)) - math.sqrt(MU / r_apogee)
        return a_disp_km, e_disp, delta_v

    disposal_5 = disposal_orbit(altitude_5yr)
    disposal_25 = disposal_orbit(altitude_25yr)

    pair_5, force_5, transitions_5 = compute_transition_highlights(
        lifetimes, 5.0, altitude_5yr, sma_edges, ECCRANGE, altitudes, times
    )
    pair_25, force_25, transitions_25 = compute_transition_highlights(
        lifetimes, 25.0, altitude_25yr, sma_edges, ECCRANGE, altitudes, times
    )

    report_transitions(f"{label} (5-year target)", transitions_5)
    report_transitions(f"{label} (25-year target)", transitions_25)

    return ScenarioResult(
        label=label,
        date=date_str,
        altitudes=altitudes,
        times=times,
        lifetimes=lifetimes,
        altitude_5yr=altitude_5yr,
        altitude_25yr=altitude_25yr,
        compliant_5=compliant_5,
        compliant_25=compliant_25,
        total_bins=total_bins,
        initial_lifetime=initial_lifetime,
        initial_altitude=initial_altitude,
        disposal_orbit_5=disposal_5,
        disposal_orbit_25=disposal_25,
        pair_5=pair_5,
        pair_25=pair_25,
        force_5=force_5,
        force_25=force_25,
        transitions_5=transitions_5,
        transitions_25=transitions_25,
    )


def main() -> None:
    density_path = Path("pyssem/utils/drag/dens_highvar_2000_dens_highvar_2000_lookup.json")
    density_data = load_density_data(density_path)

    min_alt = 200.0
    max_alt = 1500.0
    n_shells = 20
    alt_edges = [min_alt + i * (max_alt - min_alt) / n_shells for i in range(n_shells + 1)]
    sma_edges = [RE_KM + alt for alt in alt_edges]

    scenarios = [
        ("Solar Maximum 2024", "2024-07"),
        ("Solar Minimum 2030", "2030-01"),
    ]

    results: List[ScenarioResult] = []
    for label, date_str in scenarios:
        print(f"\n=== Analysing {label} ({date_str}) ===")
        results.append(analyse_scenario(label, date_str, density_data, sma_edges))

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined_specs = []
    for result in results:
        initial_bin = (0, find_bin(RE_KM + result.initial_altitude, sma_edges), RE_KM + result.initial_altitude, 0.0)
        disposal_bin_5 = (
            find_bin(result.disposal_orbit_5[1], ECCRANGE),
            find_bin(result.disposal_orbit_5[0], sma_edges),
            result.disposal_orbit_5[0],
            result.disposal_orbit_5[1],
        )
        disposal_bin_25 = (
            find_bin(result.disposal_orbit_25[1], ECCRANGE),
            find_bin(result.disposal_orbit_25[0], sma_edges),
            result.disposal_orbit_25[0],
            result.disposal_orbit_25[1],
        )

        combined_specs.append(
            {
                "title": f"{result.label} – 5-year target",
                "lifetimes": result.lifetimes,
                "threshold": 5.0,
                "initial_state": initial_bin,
                "disposal_state": disposal_bin_5,
                "pair_highlights": result.pair_5,
                "force_bins": result.force_5,
            }
        )
        combined_specs.append(
            {
                "title": f"{result.label} – 25-year target",
                "lifetimes": result.lifetimes,
                "threshold": 25.0,
                "initial_state": initial_bin,
                "disposal_state": disposal_bin_25,
                "pair_highlights": result.pair_25,
                "force_bins": result.force_25,
            }
        )

    create_2x2_svg(
        figures_dir / "jb2008_solar_cycle_compliance.svg",
        sma_edges,
        ECCRANGE,
        combined_specs,
    )

    if results[0].pair_5:
        create_single_panel_svg(
            figures_dir / "sma_ecc_solar_max_5yr.svg",
            sma_edges,
            ECCRANGE,
            {
                "title": "Solar Maximum 2024 – 5-year transitions",
                "lifetimes": results[0].lifetimes,
                "threshold": 5.0,
                "pair_highlights": results[0].pair_5,
                "force_bins": results[0].force_5,
            },
        )

    if results[1].pair_5:
        create_single_panel_svg(
            figures_dir / "sma_ecc_solar_min_5yr.svg",
            sma_edges,
            ECCRANGE,
            {
                "title": "Solar Minimum 2030 – 5-year transitions",
                "lifetimes": results[1].lifetimes,
                "threshold": 5.0,
                "pair_highlights": results[1].pair_5,
                "force_bins": results[1].force_5,
            },
        )

    print("\nSummary:")
    for result in results:
        a_disp_5, e_disp_5, delta_v_5 = result.disposal_orbit_5
        a_disp_25, e_disp_25, delta_v_25 = result.disposal_orbit_25
        print(f"- {result.label}:")
        print(f"    5-year perigee altitude ≈ {result.altitude_5yr:.1f} km; 25-year perigee altitude ≈ {result.altitude_25yr:.1f} km")
        print(f"    900 km circular lifetime ≈ {result.initial_lifetime:.1f} years")
        print(f"    5-year disposal orbit: a = {a_disp_5:.1f} km, e = {e_disp_5:.5f}, Δv ≈ {abs(delta_v_5):.1f} m/s")
        print(f"    25-year disposal orbit: a = {a_disp_25:.1f} km, e = {e_disp_25:.5f}, Δv ≈ {abs(delta_v_25):.1f} m/s")
        print(f"    Compliance: {result.compliant_5}/{result.total_bins} bins (≤5 yr), {result.compliant_25}/{result.total_bins} bins (≤25 yr)")


if __name__ == "__main__":
    main()
