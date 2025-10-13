#!/usr/bin/env python3
"""
Generate year-by-year JB2008 compliance plots using a custom lightweight
renderer that preserves axes, tick labels, and annotations, then assemble
them into an animated PNG with a slower frame rate.
"""

from __future__ import annotations

import math
import runpy
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Import helper functions from the existing JB2008 analysis script
# ---------------------------------------------------------------------------

MODULE_PATH = Path(__file__).with_name("orbital_lifetime_jb2008.py")
MODULE_NS = runpy.run_path(str(MODULE_PATH), run_name="__jb2008_helpers__")

analyse_scenario = MODULE_NS["analyse_scenario"]
load_density_data = MODULE_NS["load_density_data"]
ECCRANGE = MODULE_NS["ECCRANGE"]
RE_KM = MODULE_NS["RE_KM"]
find_bin = MODULE_NS["find_bin"]

# Silence verbose reporting inside analyse_scenario
def _quiet_transitions(label, transitions):
    return None

MODULE_NS["report_transitions"] = _quiet_transitions
analyse_scenario.__globals__["report_transitions"] = _quiet_transitions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YEARS = list(range(2020, 2036))
MONTH = "07"
SERIES_DIR = Path("figures/jb2008_yearly_series")

COLOR_CYCLE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]

BASE_COLOURS = [
    "#ffffff",  # background
    "#000000",  # text/grid
    "#74c476",  # compliant cell
    "#d9d9d9",  # non-compliant cell
    "#d95f02",  # initial state
    "#1f78b4",  # disposal state
]

CANVAS_WIDTH = 920
PANEL_HEIGHT = 640
GAP_BETWEEN_PANELS = 80
CANVAS_HEIGHT = PANEL_HEIGHT * 2 + GAP_BETWEEN_PANELS + 80

# ---------------------------------------------------------------------------
# Palette and canvas utilities
# ---------------------------------------------------------------------------


class Palette:
    def __init__(self) -> None:
        self.colours: List[Tuple[int, int, int]] = []
        self.index: Dict[str, int] = {}

    def ensure(self, colour_hex: str) -> int:
        if colour_hex not in self.index:
            rgb = tuple(int(colour_hex[i : i + 2], 16) for i in (1, 3, 5))
            self.index[colour_hex] = len(self.colours)
            self.colours.append(rgb)
        return self.index[colour_hex]


class Canvas:
    def __init__(self, width: int, height: int, palette: Palette, background: str = "#ffffff") -> None:
        self.width = width
        self.height = height
        self.palette = palette
        bg_idx = palette.ensure(background)
        self.pixels: List[bytearray] = [bytearray([bg_idx] * width) for _ in range(height)]

    def fill_rect(self, x0: float, y0: float, x1: float, y1: float, colour: str) -> None:
        idx = self.palette.ensure(colour)
        xi0 = max(0, int(math.floor(x0)))
        yi0 = max(0, int(math.floor(y0)))
        xi1 = min(self.width, int(math.ceil(x1)))
        yi1 = min(self.height, int(math.ceil(y1)))
        if xi0 >= xi1 or yi0 >= yi1:
            return
        row_bytes = bytes([idx] * (xi1 - xi0))
        for y in range(yi0, yi1):
            self.pixels[y][xi0:xi1] = row_bytes

    def draw_rect_outline(self, x0: float, y0: float, x1: float, y1: float, colour: str, thickness: int = 3) -> None:
        self.fill_rect(x0, y0, x1, y0 + thickness, colour)
        self.fill_rect(x0, y1 - thickness, x1, y1, colour)
        self.fill_rect(x0, y0, x0 + thickness, y1, colour)
        self.fill_rect(x1 - thickness, y0, x1, y1, colour)

    def draw_circle(self, cx: float, cy: float, radius: float, colour: str) -> None:
        idx = self.palette.ensure(colour)
        x0 = max(0, int(math.floor(cx - radius)))
        y0 = max(0, int(math.floor(cy - radius)))
        x1 = min(self.width, int(math.ceil(cx + radius)))
        y1 = min(self.height, int(math.ceil(cy + radius)))
        r2 = radius * radius
        for y in range(y0, y1):
            row = self.pixels[y]
            dy = (y + 0.5) - cy
            dy2 = dy * dy
            for x in range(x0, x1):
                dx = (x + 0.5) - cx
                if dx * dx + dy2 <= r2:
                    row[x] = idx


# ---------------------------------------------------------------------------
# Bitmap font (5x7) for uppercase letters, digits, and a few symbols
# ---------------------------------------------------------------------------


def expand(pattern: List[str]) -> List[str]:
    return [row.ljust(5, "0") for row in pattern]


FONT_RAW: Dict[str, List[str]] = {
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "B": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10001",
        "10001",
        "11110",
    ],
    "C": [
        "01111",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "01111",
    ],
    "D": [
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "F": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "G": [
        "01111",
        "10000",
        "10000",
        "10111",
        "10001",
        "10001",
        "01111",
    ],
    "I": [
        "01110",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "K": [
        "10001",
        "10010",
        "10100",
        "11000",
        "10100",
        "10010",
        "10001",
    ],
    "L": [
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10001",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "O": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "R": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10100",
        "10010",
        "10001",
    ],
    "S": [
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "Y": [
        "10001",
        "10001",
        "10001",
        "01110",
        "00100",
        "00100",
        "00100",
    ],
    "Z": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "11111",
    ],
    " ": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ],
    "(": [
        "00110",
        "01100",
        "01000",
        "01000",
        "01000",
        "01100",
        "00110",
    ],
    ")": [
        "01100",
        "00110",
        "00010",
        "00010",
        "00010",
        "00110",
        "01100",
    ],
    "-": [
        "00000",
        "00000",
        "00000",
        "11111",
        "00000",
        "00000",
        "00000",
    ],
    ".": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "01100",
        "01100",
    ],
    "≤": [
        "00000",
        "00011",
        "00110",
        "01100",
        "11000",
        "01100",
        "00011",
    ],
}

# Digits
digits = {
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    "3": [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "5": [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
    "6": [
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    "7": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    "9": [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100",
    ],
}

FONT_RAW.update(digits)

FONT_DATA = {char: expand(pattern) for char, pattern in FONT_RAW.items()}


def draw_text(canvas: Canvas, x: int, y: int, text: str, colour: str = "#000000") -> None:
    idx = canvas.palette.ensure(colour)
    cursor_x = x
    upper = text.upper()
    for char in upper:
        pattern = FONT_DATA.get(char, FONT_DATA[" "])
        for row_offset, row_bits in enumerate(pattern):
            ty = y + row_offset
            if 0 <= ty < canvas.height:
                row = canvas.pixels[ty]
                for col_offset, bit in enumerate(row_bits):
                    if bit == "1":
                        tx = cursor_x + col_offset
                        if 0 <= tx < canvas.width:
                            row[tx] = idx
        cursor_x += len(pattern[0]) + 1


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def assign_series_colours(pairs: List[dict], colour_map: Dict[Tuple[int, int], str], counter: List[int]) -> None:
    for entry in pairs:
        key = (entry["disposal"][0], entry["disposal"][1])
        if key not in colour_map:
            colour_map[key] = COLOR_CYCLE[counter[0] % len(COLOR_CYCLE)]
            counter[0] += 1
        entry["color"] = colour_map[key]


def x_position(ecc: float, plot_width: float, margin_left: float, zero_proxy: float, log_min: float, log_max: float) -> float:
    value = zero_proxy if ecc <= 0 else ecc
    return margin_left + (math.log10(value) - log_min) / (log_max - log_min) * plot_width


def y_position(a_km: float, plot_height: float, margin_top: float, min_a: float, max_a: float) -> float:
    return margin_top + (max_a - a_km) / (max_a - min_a) * plot_height


def render_panel(
    canvas: Canvas,
    origin_y: int,
    lifetimes: Sequence[Sequence[float]],
    threshold: float,
    pair_highlights: Sequence[dict],
    force_bins: Iterable[Tuple[int, int]],
    initial_state: Tuple[int, int, float, float],
    disposal_state: Tuple[int, int, float, float],
    sma_edges: Sequence[float],
) -> None:
    margin_left, margin_right = 90, 60
    margin_top, margin_bottom = 70, 60
    plot_width = CANVAS_WIDTH - margin_left - margin_right
    plot_height = PANEL_HEIGHT - margin_top - margin_bottom

    positive_ecc = [value for value in ECCRANGE if value > 0.0]
    min_positive = min(positive_ecc)
    zero_proxy = min_positive * 0.1
    log_min = math.log10(zero_proxy)
    log_max = math.log10(ECCRANGE[-1])

    min_a = sma_edges[0]
    max_a = sma_edges[-1]

    green = "#74c476"
    grey = "#d9d9d9"

    force_set = set(force_bins)
    for e_idx in range(len(ECCRANGE) - 1):
        x0 = x_position(ECCRANGE[e_idx], plot_width, margin_left, zero_proxy, log_min, log_max)
        x1 = x_position(ECCRANGE[e_idx + 1], plot_width, margin_left, zero_proxy, log_min, log_max)
        for a_idx in range(len(sma_edges) - 1):
            y0 = y_position(sma_edges[a_idx], plot_height, origin_y + margin_top, min_a, max_a)
            y1 = y_position(sma_edges[a_idx + 1], plot_height, origin_y + margin_top, min_a, max_a)
            colour = green if lifetimes[e_idx][a_idx] <= threshold or (e_idx, a_idx) in force_set else grey
            canvas.fill_rect(x0, y1, x1, y0, colour)

    def highlight(state: Tuple[int, int, float, float], colour: str) -> None:
        e_idx, a_idx, a_km, ecc = state
        x0 = x_position(ECCRANGE[e_idx], plot_width, margin_left, zero_proxy, log_min, log_max)
        x1 = x_position(ECCRANGE[e_idx + 1], plot_width, margin_left, zero_proxy, log_min, log_max)
        y0 = y_position(sma_edges[a_idx], plot_height, origin_y + margin_top, min_a, max_a)
        y1 = y_position(sma_edges[a_idx + 1], plot_height, origin_y + margin_top, min_a, max_a)
        canvas.draw_rect_outline(x0, y1, x1, y0, colour, thickness=4)
        canvas.draw_circle(
            x_position(ecc, plot_width, margin_left, zero_proxy, log_min, log_max),
            y_position(a_km, plot_height, origin_y + margin_top, min_a, max_a),
            radius=8,
            colour=colour,
        )

    highlight(initial_state, "#d95f02")
    highlight(disposal_state, "#1f78b4")
    for entry in pair_highlights:
        highlight(entry["initial"], entry["color"])
        highlight(entry["disposal"], entry["color"])

    # Axes
    x_axis_y = y_position(min_a, plot_height, origin_y + margin_top, min_a, max_a)
    canvas.fill_rect(margin_left, x_axis_y, CANVAS_WIDTH - margin_right, x_axis_y + 2, "#000000")
    canvas.fill_rect(margin_left, origin_y + margin_top, margin_left + 2, origin_y + margin_top + plot_height, "#000000")

    for ecc in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        if ecc < zero_proxy or ecc > ECCRANGE[-1]:
            continue
        xtick = x_position(ecc, plot_width, margin_left, zero_proxy, log_min, log_max)
        canvas.fill_rect(xtick, x_axis_y, xtick + 1, x_axis_y + 12, "#000000")
        label = f"{ecc:.0e}" if ecc < 0.1 else f"{ecc:.1f}"
        draw_text(canvas, int(xtick - 10), int(x_axis_y + 20), label, "#000000")

    for alt in [200, 400, 600, 800, 1000, 1200, 1400]:
        ytick = y_position(RE_KM + alt, plot_height, origin_y + margin_top, min_a, max_a)
        canvas.fill_rect(margin_left - 10, ytick, margin_left, ytick + 1, "#000000")
        draw_text(canvas, margin_left - 60, int(ytick - 4), f"{alt}", "#000000")

    draw_text(canvas, margin_left + 180, int(x_axis_y + 36), "ECCENTRICITY", "#000000")
    draw_text(canvas, margin_left - 85, origin_y + margin_top + plot_height // 2, "SMA (KM)", "#000000")
    draw_text(canvas, margin_left, origin_y + margin_top - 20, f"LIFETIME ≤ {threshold:.0f} YEARS", "#000000")


# ---------------------------------------------------------------------------
# PNG / APNG writers
# ---------------------------------------------------------------------------


def convert_to_png(canvas: Canvas, palette: Palette, path: Path) -> None:
    import zlib
    import struct
    import binascii

    colours = palette.colours[:]
    if len(colours) > 256:
        raise ValueError("Palette too large for indexed PNG.")
    while len(colours) < 256:
        colours.append((0, 0, 0))

    raw = bytearray()
    for row in canvas.pixels:
        raw.append(0)
        raw.extend(row)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = len(data).to_bytes(4, "big")
        crc = binascii.crc32(chunk_type)
        crc = binascii.crc32(data, crc) & 0xFFFFFFFF
        return length + chunk_type + data + crc.to_bytes(4, "big")

    ihdr = struct.pack(
        ">IIBBBBB",
        canvas.width,
        canvas.height,
        8,  # bit depth
        3,  # indexed colour
        0,
        0,
        0,
    )

    plte = bytearray()
    for r, g, b in colours:
        plte.extend([r, g, b])

    idat = zlib.compress(bytes(raw), level=9)

    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"PLTE", bytes(plte)))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def write_apng(canvases: Sequence[Canvas], palette: Palette, path: Path, delay_num: int = 3, delay_den: int = 4) -> None:
    import zlib
    import struct
    import binascii

    width = canvases[0].width
    height = canvases[0].height

    colours = palette.colours[:]
    if len(colours) > 256:
        raise ValueError("Palette too large for indexed PNG.")
    while len(colours) < 256:
        colours.append((0, 0, 0))

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = len(data).to_bytes(4, "big")
        crc = binascii.crc32(chunk_type)
        crc = binascii.crc32(data, crc) & 0xFFFFFFFF
        return length + chunk_type + data + crc.to_bytes(4, "big")

    def frame_bytes(canvas: Canvas) -> bytes:
        raw = bytearray()
        for row in canvas.pixels:
            raw.append(0)
            raw.extend(row)
        return zlib.compress(bytes(raw), level=9)

    compressed_frames = [frame_bytes(canvas) for canvas in canvases]

    plte = bytearray()
    for r, g, b in colours:
        plte.extend([r, g, b])

    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"acTL", struct.pack(">II", len(compressed_frames), 0)))
        f.write(chunk(b"PLTE", bytes(plte)))

        sequence = 0
        for idx, data in enumerate(compressed_frames):
            fcTL = struct.pack(
                ">IIIIIHHBB",
                sequence,
                width,
                height,
                0,
                0,
                delay_num,
                delay_den,
                0,
                0,
            )
            f.write(chunk(b"fcTL", fcTL))
            sequence += 1

            if idx == 0:
                f.write(chunk(b"IDAT", data))
            else:
                fdAT = sequence.to_bytes(4, "big") + data
                f.write(chunk(b"fdAT", fdAT))
                sequence += 1

        f.write(chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main() -> None:
    SERIES_DIR.mkdir(parents=True, exist_ok=True)

    density_path = Path("pyssem/utils/drag/dens_highvar_2000_dens_highvar_2000_lookup.json")
    density_data = load_density_data(density_path)

    min_alt, max_alt = 200.0, 1500.0
    n_shells = 20
    alt_edges = [min_alt + i * (max_alt - min_alt) / n_shells for i in range(n_shells + 1)]
    sma_edges = [RE_KM + alt for alt in alt_edges]

    palette = Palette()
    for colour in BASE_COLOURS:
        palette.ensure(colour)

    colour_map_5: Dict[Tuple[int, int], str] = {}
    colour_map_25: Dict[Tuple[int, int], str] = {}
    counter_5 = [0]
    counter_25 = [0]

    canvases: List[Canvas] = []

    for year in YEARS:
        result = analyse_scenario(f"Year {year}", f"{year}-{MONTH}", density_data, sma_edges)

        canvas = Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, palette)
        draw_text(canvas, 20, 30, f"YEAR {year}", "#000000")

        initial_state = (
            0,
            find_bin(RE_KM + result.initial_altitude, sma_edges),
            RE_KM + result.initial_altitude,
            0.0,
        )

        disp5_a, disp5_e, _ = result.disposal_orbit_5
        disp25_a, disp25_e, _ = result.disposal_orbit_25

        disposal_state_5 = (
            find_bin(disp5_e, ECCRANGE),
            find_bin(disp5_a, sma_edges),
            disp5_a,
            disp5_e,
        )
        disposal_state_25 = (
            find_bin(disp25_e, ECCRANGE),
            find_bin(disp25_a, sma_edges),
            disp25_a,
            disp25_e,
        )

        pairs_5 = deepcopy(result.pair_5)
        pairs_25 = deepcopy(result.pair_25)
        assign_series_colours(pairs_5, colour_map_5, counter_5)
        assign_series_colours(pairs_25, colour_map_25, counter_25)

        for entry in pairs_5 + pairs_25:
            palette.ensure(entry["color"])

        draw_text(canvas, 20, 60, "SOLAR DENSITY", "#000000")

        render_panel(
            canvas,
            origin_y=80,
            lifetimes=result.lifetimes,
            threshold=5.0,
            pair_highlights=pairs_5,
            force_bins=result.force_5,
            initial_state=initial_state,
            disposal_state=disposal_state_5,
            sma_edges=sma_edges,
        )

        render_panel(
            canvas,
            origin_y=80 + PANEL_HEIGHT,
            lifetimes=result.lifetimes,
            threshold=25.0,
            pair_highlights=pairs_25,
            force_bins=result.force_25,
            initial_state=initial_state,
            disposal_state=disposal_state_25,
            sma_edges=sma_edges,
        )

        canvases.append(canvas)
        png_path = SERIES_DIR / f"{year}_compliance.png"
        convert_to_png(canvas, palette, png_path)
        print(f"Saved plot for {year}")

    apng_path = SERIES_DIR / "jb2008_disposal_evolution.png"
    write_apng(canvases, palette, apng_path, delay_num=3, delay_den=2)  # 1.5 seconds per frame
    print(f"Animated PNG saved to {apng_path}")


if __name__ == "__main__":
    main()
