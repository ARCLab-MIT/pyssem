#!/usr/bin/env python3
"""
Benchmark RK45 versus explicit Euler propagation for the elliptical test scenario.

Usage
-----
    python propagation_tests/run_propagation_benchmarks.py --dt 0.01 0.05 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyssem.model import Model  # noqa: E402


DEFAULT_DT_VALUES: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5)
CONFIG_PATH = ROOT / "pyssem" / "simulation_configurations" / "elliptical-test.json"


def load_configuration(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def create_model(config: Dict, integrator: str, parallel_processing: bool) -> Model:
    scenario_props = deepcopy(config["scenario_properties"])
    scenario_props["integrator"] = integrator
    scenario_props["parallel_processing"] = parallel_processing
    scenario_props["elliptical"] = True

    model = Model(
        start_date=scenario_props["start_date"].split("T")[0],
        simulation_duration=scenario_props["simulation_duration"],
        steps=scenario_props["steps"],
        min_altitude=scenario_props["min_altitude"],
        max_altitude=scenario_props["max_altitude"],
        n_shells=scenario_props["n_shells"],
        launch_function=scenario_props["launch_function"],
        integrator=scenario_props["integrator"],
        density_model=scenario_props["density_model"],
        LC=scenario_props["LC"],
        v_imp=scenario_props.get("v_imp"),
        fragment_spreading=scenario_props.get("fragment_spreading", False),
        parallel_processing=scenario_props.get("parallel_processing", False),
        baseline=scenario_props.get("baseline", False),
        indicator_variables=scenario_props.get("indicator_variables"),
        launch_scenario=scenario_props["launch_scenario"],
        SEP_mapping=config.get("SEP_mapping"),
        elliptical=scenario_props.get("elliptical", True),
        eccentricity_bins=scenario_props.get("eccentricity_bins"),
    )

    model.configure_species(config["species"])
    model.build_model(elliptical=True)
    return model


def run_propagation(model: Model, years: float, step_size: float | None) -> Dict:
    scenario = model.scenario_properties
    initial_state = np.asarray(scenario.x0, dtype=float)
    times = np.array([0.0, float(years)], dtype=float)

    start = time.perf_counter()
    ecc_matrix, alt_matrix = model.propagate(
        times=times,
        population=initial_state,
        launch=None,
        elliptical=scenario.elliptical,
        step_size=step_size,
    )
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "eccentricity_matrix": ecc_matrix,
        "altitude_matrix": alt_matrix,
    }


def shell_totals(altitude_matrix: np.ndarray) -> np.ndarray:
    return np.sum(altitude_matrix, axis=1)


def compute_metrics(reference: np.ndarray, comparison: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(comparison, dtype=float) - np.asarray(reference, dtype=float)
    rmse = np.sqrt(np.mean(diff ** 2))
    baseline_norm = np.sqrt(np.mean(reference ** 2)) if np.any(reference) else 1.0
    rel_rmse = rmse / baseline_norm

    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(rmse),
        "relative_rmse": float(rel_rmse),
        "total_abs": float(np.sum(np.abs(diff))),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"max|Δ|={metrics['max_abs']:.3f}, "
        f"RMSE={metrics['rmse']:.3f}, "
        f"relRMSE={metrics['relative_rmse']:.3%}, "
        f"Σ|Δ|={metrics['total_abs']:.3f}"
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the elliptical test configuration JSON file.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        nargs="+",
        default=DEFAULT_DT_VALUES,
        help="Fixed Euler timesteps (years) to benchmark.",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=1.0,
        help="Propagation duration in years (default: 1).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel lambdification (defaults to disabled for faster experimentation).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_configuration(args.config)

    baseline_model = create_model(config, integrator=config["scenario_properties"]["integrator"], parallel_processing=args.parallel)
    baseline_run = run_propagation(baseline_model, years=args.years, step_size=None)
    baseline_alt = np.asarray(baseline_run["altitude_matrix"], dtype=float)
    baseline_totals = shell_totals(baseline_alt)

    print("Baseline (RK45):")
    print(f"  runtime = {baseline_run['duration']:.3f}s")
    print(f"  total objects = {baseline_totals.sum():.3f}")
    print()

    for dt in args.dt:
        euler_model = create_model(config, integrator="EULER", parallel_processing=args.parallel)
        # Ensure the integrator flag is set on the scenario properties
        euler_model.scenario_properties.integrator = "EULER"

        euler_run = run_propagation(euler_model, years=args.years, step_size=dt)
        euler_alt = np.asarray(euler_run["altitude_matrix"], dtype=float)
        euler_totals = shell_totals(euler_alt)

        metrics = compute_metrics(baseline_alt, euler_alt)
        speedup = baseline_run["duration"] / euler_run["duration"] if euler_run["duration"] > 0 else float("inf")

        print(f"Euler dt={dt:.3f}:")
        print(f"  runtime = {euler_run['duration']:.3f}s (speedup {speedup:.2f}x)")
        print(f"  total objects = {euler_totals.sum():.3f}")
        print(f"  {format_metrics(metrics)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
