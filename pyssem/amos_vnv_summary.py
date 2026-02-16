#!/usr/bin/env python3
import os
import csv
import math
import glob
import re
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

EPS = 1e-12

# -----------------------------
# Utility: grids & volumes
# -----------------------------
def edges_from_mids(mids: np.ndarray) -> np.ndarray:
    mids = np.asarray(mids, float)
    if len(mids) < 2:
        # construct a tiny edge if only one altitude appears (degenerate case)
        w = 1.0
        return np.array([[mids[0]-w/2, mids[0]+w/2]], float)
    left  = np.r_[mids[0] - (mids[1]-mids[0])/2,  (mids[1:]+mids[:-1])/2]
    right = np.r_[(mids[1:]+mids[:-1])/2,         mids[-1] + (mids[-1]-mids[-2])/2]
    return np.c_[left, right]

def mass_conserve_rebin_1d(src_edges, src_mass, tgt_edges):
    src_edges = np.asarray(src_edges, float)
    tgt_edges = np.asarray(tgt_edges, float)
    src_mass = np.asarray(src_mass, float)
    out = np.zeros(len(tgt_edges), float)
    if len(src_edges)==0 or len(tgt_edges)==0:
        return out
    i = j = 0
    sL, sR = src_edges[i,0], src_edges[i,1]
    tL, tR = tgt_edges[j,0], tgt_edges[j,1]
    while True:
        overlap = max(0.0, min(sR, tR) - max(sL, tL))
        if overlap > 0:
            out[j] += src_mass[i] * (overlap / max(sR - sL, EPS))
        if sR <= tR:
            i += 1
            if i == len(src_edges): break
            sL, sR = src_edges[i]
        else:
            j += 1
            if j == len(tgt_edges): break
            tL, tR = tgt_edges[j]
    return out

def sphere_shell_volumes(edges, R_earth_km=6378.136):
    edges = np.asarray(edges, float)
    r_in  = R_earth_km + edges[:,0]
    r_out = R_earth_km + edges[:,1]
    return (4.0/3.0) * np.pi * (np.maximum(r_out,0)**3 - np.maximum(r_in,0)**3)

def gauss_kernel_km(step_km, sigma_km, half_width_sigmas=4):
    half_w = max(1, int(np.ceil(half_width_sigmas * sigma_km / max(step_km, EPS))))
    x = np.arange(-half_w, half_w+1) * step_km
    k = np.exp(-0.5*(x/sigma_km)**2)
    k /= max(k.sum(), EPS)
    return k

def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p,float); q = np.asarray(q,float)
    ps = p.sum(); qs = q.sum()
    if ps <= eps and qs <= eps:
        return 0.0
    p = p / max(ps, eps); q = q / max(qs, eps)
    m = 0.5*(p+q)
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0); m = np.clip(m, eps, 1.0)
    return float(0.5*np.sum(p*np.log(p/m)) + 0.5*np.sum(q*np.log(q/m)))

# -----------------------------
# Group list
# -----------------------------
GROUPS = ["S", "N", "D", "B"]

# -----------------------------
# Time-series metrics
# -----------------------------
def pearson_r(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2: return 0.0
    aa = a - a.mean(); bb = b - b.mean()
    denom = math.sqrt((aa**2).sum() * (bb**2).sum())
    return float((aa*bb).sum() / max(denom, EPS))

def r2_score(y, yhat):
    y = np.asarray(y,float); yhat = np.asarray(yhat,float)
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    return float(1 - ss_res/max(ss_tot, EPS))

def rmse(a, b): 
    a = np.asarray(a,float); b = np.asarray(b,float)
    return float(np.sqrt(((a-b)**2).mean()))

def dtw_abs(a, b):
    # Simple O(n^2) DTW with |Δ| cost
    a = np.asarray(a,float); b = np.asarray(b,float)
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf, float)
    D[0,0] = 0.0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = abs(a[i-1]-b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m])

def wape_all(y_true, y_pred):
    y_true = np.asarray(y_true,float); y_pred = np.asarray(y_pred,float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true))
    return float(num / max(den, EPS))

def smape_global(y_true, y_pred):
    y_true = np.asarray(y_true,float); y_pred = np.asarray(y_pred,float)
    num = 2*np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true) + np.abs(y_pred))
    return float(num / max(den, EPS))

def group_scale_median_dy(mc_series):
    dy = np.diff(np.asarray(mc_series,float))
    if dy.size == 0: return 1.0
    scale = np.median(np.abs(dy))
    if not np.isfinite(scale) or scale < EPS:
        # fallback: median level
        scale = max(np.median(np.abs(mc_series)), 1.0)
    return float(scale)

# -----------------------------
# Volume-normalized spatial metrics (Option 3)
# -----------------------------
def volume_normalized_spatial_metrics(
    alt_mids_mc, counts_mc, alt_mids_ssem, counts_ssem,
    ref_step_km=10.0, sigma_km=50.0, alt_min=None, alt_max=None, R_earth_km=6378.136
):
    alt_mids_mc   = np.asarray(alt_mids_mc, float)
    alt_mids_ssem = np.asarray(alt_mids_ssem, float)
    counts_mc     = np.asarray(counts_mc, float)
    counts_ssem   = np.asarray(counts_ssem, float)

    if alt_mids_mc.size == 0 and alt_mids_ssem.size == 0:
        return dict(JSdiv_alt_smoothed_ref_vol=np.nan,
                    AltCentroidDiff_km_ref_vol=np.nan,
                    TailFracDelta_gt1000km_vol=np.nan,
                    VWAPE_alt=np.nan)

    if alt_min is None:
        alt_min = float(min(alt_mids_mc.min() if alt_mids_mc.size else np.inf,
                            alt_mids_ssem.min() if alt_mids_ssem.size else np.inf))
    if alt_max is None:
        alt_max = float(max(alt_mids_mc.max() if alt_mids_mc.size else -np.inf,
                            alt_mids_ssem.max() if alt_mids_ssem.size else -np.inf))

    tg_mids  = np.arange(alt_min, alt_max + ref_step_km, ref_step_km)
    tg_edges = edges_from_mids(tg_mids)

    A = mass_conserve_rebin_1d(edges_from_mids(alt_mids_mc), counts_mc, tg_edges)
    B = mass_conserve_rebin_1d(edges_from_mids(alt_mids_ssem), counts_ssem, tg_edges)

    V  = sphere_shell_volumes(tg_edges, R_earth_km=R_earth_km)    # km^3
    rhoA = A / np.maximum(V, 1e-12)   # number per km^3
    rhoB = B / np.maximum(V, 1e-12)

    k = gauss_kernel_km(ref_step_km, sigma_km)
    rhoA_s = np.convolve(rhoA, k, mode="same")
    rhoB_s = np.convolve(rhoB, k, mode="same")

    pA = rhoA_s / max(rhoA_s.sum(), 1e-12)
    pB = rhoB_s / max(rhoB_s.sum(), 1e-12)

    JS_vol = js_divergence(pA, pB)

    centroid_A = float((tg_mids @ pA))
    centroid_B = float((tg_mids @ pB))
    centroid_diff_km = centroid_A - centroid_B

    tail_mask = tg_mids >= 1000.0
    tailA = float(pA[tail_mask].sum())
    tailB = float(pB[tail_mask].sum())
    tail_frac_delta = tailA - tailB  # MC - SSEM

    num = float(np.sum(np.abs(A - B) * V))
    den = float(np.sum((np.abs(A) + np.abs(B)) * V / 2.0))
    vwape = (num / den) if den > 0 else 0.0

    return {
        "JSdiv_alt_smoothed_ref_vol": JS_vol,
        "AltCentroidDiff_km_ref_vol": centroid_diff_km,
        "TailFracDelta_gt1000km_vol": tail_frac_delta,
        "VWAPE_alt": vwape,
    }

# -----------------------------
# Composite score weights
# -----------------------------
# 30/30/20/20 across (WAPE_all / Spatial / Final / Shape)
W_WAPE  = 0.40
W_SPAT  = 0.40
W_FINAL = 0.10
W_SHAPE = 0.10

# Spatial internal weights (Option 3)
SPAT_CENT_W = 0.40
SPAT_TAIL_W = 0.20
SPAT_JS_W   = 0.40

# Fallback spatial mix (if smoothed metrics missing)
SPAT_HHI_W   = 0.50
SPAT_sRMSE_W = 0.50

# Shape mix
SHAPE_R_W   = 0.60
SHAPE_DTW_W = 0.40

# Normalization scales
CENTROID_SCALE_KM = 50.0     # “moderate” centroid bias
TAIL_SCALE        = 0.10     # 10% mass difference in upper tail
JS_SCALE          = 0.06     # “moderate” smoothed JS
DTW_SCALE_DEFAULT = 7.0      # fallback if we can't compute a group-specific DTW scale

# -----------------------------
# Core computations
# -----------------------------
def compute_time_metrics_for_group(df_mc, df_ssem, group):
    # df_* columns: Species, Year, Population
    mc = df_mc[df_mc["Species"]==group]
    ss = df_ssem[df_ssem["Species"]==group]
    if mc.empty or ss.empty:
        return None
    years_common = np.intersect1d(mc["Year"].unique(), ss["Year"].unique()).astype(int)
    if years_common.size == 0:
        return None
    mc_series = mc.set_index("Year")["Population"].reindex(years_common).fillna(0.0).to_numpy(float)
    ss_series = ss.set_index("Year")["Population"].reindex(years_common).fillna(0.0).to_numpy(float)

    # Metrics
    WAPE_all = wape_all(mc_series, ss_series)
    sMAPE_g  = smape_global(mc_series, ss_series)
    RMSE     = rmse(mc_series, ss_series)
    NRMSE    = RMSE / max(np.mean(np.abs(mc_series)), EPS)
    Bias     = float(np.mean(ss_series - mc_series))
    Final_rel_err = float((ss_series[-1]-mc_series[-1]) / max(abs(mc_series[-1]), EPS))
    Total_ratio   = float(np.sum(ss_series) / max(np.sum(mc_series), EPS))
    abs_err = np.abs(ss_series - mc_series)
    i_max   = int(abs_err.argmax())
    Max_abs_err     = float(abs_err[i_max])
    Year_of_max_err = int(years_common[i_max])
    r  = pearson_r(mc_series, ss_series)
    R2 = r2_score(mc_series, ss_series)
    DTW = dtw_abs(mc_series, ss_series)

    # group-specific DTW scale from MC
    dtw_scale = group_scale_median_dy(mc_series)
    DTW_norm = DTW / max(dtw_scale, EPS)

    return dict(
        WAPE=WAPE_all, sMAPE_global=sMAPE_g, RMSE=RMSE, NRMSE=NRMSE, Bias=Bias,
        Final_rel_err=Final_rel_err, Total_ratio=Total_ratio,
        Max_abs_err=Max_abs_err, Year_of_max_err=Year_of_max_err,
        r=r, R2=R2, DTW=DTW, DTW_norm=DTW_norm, WAPE_all=WAPE_all,
        r_clamped=float(np.clip(r, -1, 1)),
        years_common=list(map(int, years_common))
    )

def compute_spatial_metrics_for_group(df_mc_alt, df_ssem_alt, group,
                                      ref_step_km=10.0, sigma_km=50.0):
    sub_mc  = df_mc_alt[df_mc_alt["Species"]==group]
    sub_ss  = df_ssem_alt[df_ssem_alt["Species"]==group]
    if sub_mc.empty or sub_ss.empty:
        return dict(JSdiv_alt_smoothed_ref_vol=np.nan,
                    AltCentroidDiff_km_ref_vol=np.nan,
                    TailFracDelta_gt1000km_vol=np.nan,
                    VWAPE_alt=np.nan)

    years_common = np.intersect1d(sub_mc["Year"].unique(), sub_ss["Year"].unique()).astype(int)
    if years_common.size == 0:
        return dict(JSdiv_alt_smoothed_ref_vol=np.nan,
                    AltCentroidDiff_km_ref_vol=np.nan,
                    TailFracDelta_gt1000km_vol=np.nan,
                    VWAPE_alt=np.nan)

    # Compute per-year, then take median across years for robustness
    js_list, cent_list, tail_list, vwape_list = [], [], [], []
    for y in years_common:
        mc_y   = sub_mc[sub_mc["Year"]==y][["Altitude","Population"]]
        ssem_y = sub_ss[sub_ss["Year"]==y][["Altitude","Population"]]
        # make sure both are altitude-sorted and aggregated
        g_mc = mc_y.groupby("Altitude", as_index=False)["Population"].sum().sort_values("Altitude")
        g_ss = ssem_y.groupby("Altitude", as_index=False)["Population"].sum().sort_values("Altitude")

        m = volume_normalized_spatial_metrics(
            g_mc["Altitude"].to_numpy(float), g_mc["Population"].to_numpy(float),
            g_ss["Altitude"].to_numpy(float), g_ss["Population"].to_numpy(float),
            ref_step_km=ref_step_km, sigma_km=sigma_km
        )
        js_list.append(m["JSdiv_alt_smoothed_ref_vol"])
        cent_list.append(m["AltCentroidDiff_km_ref_vol"])
        tail_list.append(m["TailFracDelta_gt1000km_vol"])
        vwape_list.append(m["VWAPE_alt"])

    def med(x): 
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else np.nan

    return dict(
        JSdiv_alt_smoothed_ref_vol = med(js_list),
        AltCentroidDiff_km_ref_vol = med(cent_list),
        TailFracDelta_gt1000km_vol = med(tail_list),
        VWAPE_alt                  = med(vwape_list)
    )

def composite_score(row):
    # Magnitude
    wape = float(row.get("WAPE_all", np.nan))
    # Final state
    final = abs(float(row.get("Final_rel_err", np.nan)))
    # Shape
    r = float(row.get("r_clamped", 0.0))
    dtw_n = float(row.get("DTW_norm", np.nan))
    shape = SHAPE_R_W*(1.0 - r) + SHAPE_DTW_W*min(dtw_n/DTW_SCALE_DEFAULT, 2.0)

    # Spatial (Option 3 + volume-normalised)
    cent = float(row.get("AltCentroidDiff_km_ref_vol", np.nan))
    tail = float(row.get("TailFracDelta_gt1000km_vol", np.nan))
    js   = float(row.get("JSdiv_alt_smoothed_ref_vol", np.nan))
    parts, wsum = [], 0.0
    if np.isfinite(cent):
        parts.append(SPAT_CENT_W * min(abs(cent)/CENTROID_SCALE_KM, 2.0)); wsum += SPAT_CENT_W
    if np.isfinite(tail):
        parts.append(SPAT_TAIL_W * min(abs(tail)/TAIL_SCALE, 2.0)); wsum += SPAT_TAIL_W
    if np.isfinite(js):
        parts.append(SPAT_JS_W   * min(js/JS_SCALE, 2.0));           wsum += SPAT_JS_W

    if wsum > 0:
        spatial = sum(parts)/wsum
    else:
        # fallback: if no spatial metrics, use volumetric WAPE across altitude if available
        vw = row.get("VWAPE_alt", np.nan)
        spatial = float(vw) if isinstance(vw, (int,float)) and np.isfinite(vw) else 0.0

    score = W_WAPE*wape + W_SPAT*spatial + W_FINAL*final + W_SHAPE*shape
    return float(score)

# -----------------------------
# IO helpers
# -----------------------------
def read_pop_time_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize columns
    assert {"Species","Year","Population"}.issubset(df.columns), f"Missing cols in {path}"
    df["Species"] = df["Species"].astype(str)
    df["Year"] = df["Year"].astype(int)
    df["Population"] = df["Population"].astype(float)
    # If more than our groups exist, aggregate to our GROUPS if they are already group labels
    return df

def read_pop_time_alt_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"Species","Year","Altitude","Population"}.issubset(df.columns), f"Missing cols in {path}"
    df["Species"] = df["Species"].astype(str)
    df["Year"] = df["Year"].astype(int)
    df["Altitude"] = df["Altitude"].astype(float)
    df["Population"] = df["Population"].astype(float)
    return df

# Helper: Read CPU seconds from cpu_time.csv in a scenario folder
def read_cpu_seconds(sim_dir: Path) -> float | None:
    """
    Read CPU seconds from grid_search/<scenario>/cpu_time.csv.
    Returns None if the file is missing or malformed.
    """
    path = sim_dir / "cpu_time.csv"
    if not path.exists():
        return None
    try:
        with path.open("r", newline="") as f:
            import csv as _csv
            reader = _csv.DictReader(f)
            for row in reader:
                val = row.get("cpu_seconds")
                if val is not None and val != "":
                    try:
                        return float(val)
                    except Exception:
                        pass
                break
        # Fallback: naive 2-column CSV
        with path.open("r", newline="") as f:
            rr = _csv.reader(f)
            _ = next(rr, None)  # header
            for r in rr:
                if len(r) >= 2:
                    try:
                        return float(r[1])
                    except Exception:
                        continue
    except Exception:
        return None
    return None

# -----------------------------
# Main per-scenario worker
# -----------------------------
def process_scenario(sim_dir: Path, mc_dir: Path,
                     ref_step_km=10.0, sigma_km=50.0) -> tuple[pd.DataFrame, float]:
    sim_name = sim_dir.name

    ssem_tot_path = sim_dir / "pop_time.csv"
    ssem_alt_path = sim_dir / "pop_time_alt.csv"
    mc_tot_path   = mc_dir / "pop_time_mc.csv"
    mc_alt_path   = mc_dir / "pop_time_alt_mc.csv"

    if not ssem_tot_path.exists() or not ssem_alt_path.exists():
        raise FileNotFoundError(f"{sim_name}: missing pop_time.csv or pop_time_alt.csv")
    if not mc_tot_path.exists() or not mc_alt_path.exists():
        raise FileNotFoundError("MC directory must contain pop_time_mc.csv and pop_time_alt_mc.csv")

    df_ssem_tot = read_pop_time_csv(ssem_tot_path)
    df_ssem_alt = read_pop_time_alt_csv(ssem_alt_path)
    df_mc_tot   = read_pop_time_csv(mc_tot_path)
    df_mc_alt   = read_pop_time_alt_csv(mc_alt_path)

    rows = []
    for g in GROUPS:
        tm = compute_time_metrics_for_group(df_mc_tot, df_ssem_tot, g)
        if tm is None:
            continue
        sm = compute_spatial_metrics_for_group(df_mc_alt, df_ssem_alt, g,
                                               ref_step_km=ref_step_km, sigma_km=sigma_km)
        row = {"simulation_name": sim_name, "Group": g, **tm, **sm}
        row["Score_group"] = composite_score(row)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"{sim_name}: no groups with overlapping years")

    df = pd.DataFrame(rows)

    # Scenario-level score: weight groups by MC total mass share over all years
    mc_totals = (
        df_mc_tot.groupby("Species")["Population"].sum()
                 .reindex(GROUPS).fillna(0.0)
    )
    if mc_totals.sum() > 0:
        w = (mc_totals / mc_totals.sum()).to_dict()
    else:
        w = {g: 1.0/len(GROUPS) for g in GROUPS}

    score_scenario = float(sum(w.get(g,0.0) * s for g, s in zip(df["Group"], df["Score_group"])))

    # Write per-scenario metrics
    out_path = sim_dir / f"metrics_{sim_name}.csv"
    df_out = df.copy()
    if "simulation_name" not in df_out.columns:
        df_out.insert(0, "simulation_name", sim_name)
    else:
        # ensure simulation_name is the first column
        cols = ["simulation_name"] + [c for c in df_out.columns if c != "simulation_name"]
        df_out = df_out[cols]
    df_out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path}  (lower score is better). Scenario score = {score_scenario:.4f}")

    # Also write a tiny scenario-level CSV for convenience
    with (sim_dir / f"metrics_scenario_{sim_name}.csv").open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["simulation_name", "Score_scenario"])
        wri.writerow([sim_name, f"{score_scenario:.6f}"])

    return df_out, score_scenario

# -----------------------------
# Cross-scenario plotting
# -----------------------------
def plot_scenario_scores(scores: list[tuple[str,float]], out_dir: Path):
    if not scores:
        return
    scores_sorted = sorted(scores, key=lambda x: x[1])
    labels = [s[0] for s in scores_sorted]
    vals   = [s[1] for s in scores_sorted]

    plt.figure(figsize=(10, 6))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Composite scenario score (lower is better)")
    plt.title("Scenario scores across runs")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out_png = out_dir / "scenario_scores.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"📈 Wrote {out_png}")

def plot_group_heatmap(all_group_df: pd.DataFrame, out_dir: Path):
    if all_group_df.empty:
        return
    # pivot: rows=scenario, cols=group, values=Score_group
    piv = all_group_df.pivot(index="simulation_name", columns="Group", values="Score_group").fillna(np.nan)
    # sort scenarios by mean score
    piv = piv.loc[piv.mean(axis=1).sort_values().index]

    plt.figure(figsize=(8, max(4, 0.35*len(piv))))
    im = plt.imshow(piv.to_numpy(float), aspect="auto", origin="upper")
    plt.xticks(range(len(piv.columns)), piv.columns)
    plt.yticks(range(len(piv.index)), piv.index, fontsize=8)
    plt.colorbar(im, label="Per-group score (lower is better)")
    plt.title("Per-group scores across scenarios")
    plt.tight_layout()
    out_png = out_dir / "group_scores_heatmap.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"📊 Wrote {out_png}")

# -----------------------------
# Knee-Pareto plot (Score vs CPU time)
# -----------------------------
def _pareto_front_indices(points: np.ndarray) -> np.ndarray:
    """
    Compute indices of the Pareto-efficient set for minimisation of both axes.
    points: array of shape (N,2) with [cpu_seconds, score].
    Returns indices (sorted along increasing cpu).
    """
    if points.size == 0:
        return np.array([], dtype=int)
    # Sort by cpu ascending, then by score ascending
    order = np.lexsort((points[:,1], points[:,0]))
    pts = points[order]
    keep = []
    best_score = np.inf
    for i, (_, sc) in enumerate(pts):
        if sc <= best_score + 1e-12:
            keep.append(i)
            best_score = min(best_score, sc)
    return order[np.array(keep, dtype=int)]

def _knee_index_on_front(front_xy: np.ndarray) -> int | None:
    """
    Given Pareto-front points sorted by increasing cpu (shape (M,2)),
    find the knee as the point with maximum perpendicular distance
    from the line joining the first and last front points, in normalised space.
    Returns the index (0..M-1) within front_xy, or None if undefined.
    """
    M = len(front_xy)
    if M < 3:
        return None
    # Normalise to [0,1] on each axis to avoid scale bias
    x = front_xy[:,0].astype(float)
    y = front_xy[:,1].astype(float)
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    xn = (x - x0) / (x1 - x0)
    yn = (y - y0) / (y1 - y0)
    p0 = np.array([xn[0], yn[0]])
    p1 = np.array([xn[-1], yn[-1]])
    v = p1 - p0
    vnorm = np.linalg.norm(v)
    if vnorm <= 1e-12:
        return None
    # Perpendicular distance from each point to the chord
    dists = []
    for i in range(M):
        p = np.array([xn[i], yn[i]])
        # area of parallelogram / base length
        dist = np.abs(np.cross(v, p - p0)) / vnorm
        dists.append(dist)
    return int(np.argmax(dists))

def plot_score_cpu_pareto(scores_df: pd.DataFrame, out_dir: Path):
    """
    Create a scatter plot of Score_scenario vs CPU seconds, highlight the Pareto front,
    and mark the 'knee' point.
    Expects columns: simulation_name, Score_scenario, cpu_seconds
    """
    df = scores_df.copy()
    # Keep finite
    df = df[np.isfinite(df["Score_scenario"]) & np.isfinite(df["cpu_seconds"])]
    if df.empty:
        print("No finite (score, cpu) pairs to plot.")
        return

    pts = df[["cpu_seconds","Score_scenario"]].to_numpy(float)
    idx_front = _pareto_front_indices(pts)
    front = pts[idx_front]
    # Sort front by cpu ascending for a clean line
    order_front = np.argsort(front[:,0])
    idx_front = idx_front[order_front]
    front = front[order_front]

    knee_rel = _knee_index_on_front(front)
    knee_xy = None
    if knee_rel is not None:
        knee_xy = front[knee_rel]

    # ---- Plot ----
    plt.figure(figsize=(9, 6))
    # All points
    plt.scatter(df["cpu_seconds"], df["Score_scenario"], alpha=0.6, label="All runs")
    # Pareto front
    plt.plot(front[:,0], front[:,1], marker="o", linewidth=1.5, label="Pareto front")
    # Knee
    if knee_xy is not None:
        plt.scatter([knee_xy[0]], [knee_xy[1]], marker="*", s=180, label="Knee", zorder=5)
        # annotate knee with scenario name
        # find the scenario whose point matches knee_xy
        k_mask = (np.isclose(df["cpu_seconds"], knee_xy[0]) & np.isclose(df["Score_scenario"], knee_xy[1]))
        if k_mask.any():
            k_name = df.loc[k_mask, "simulation_name"].iloc[0]
            plt.annotate(k_name, (knee_xy[0], knee_xy[1]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    # Label Pareto points lightly
    front_names = df.iloc[idx_front]["simulation_name"].to_list()
    for (x, y), name in zip(front, front_names):
        plt.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, alpha=0.8)

    plt.xlabel("CPU time (seconds)")
    plt.ylabel("Composite scenario score (lower is better)")
    plt.title("Score vs CPU time — Pareto front with knee")
    plt.ylim(0, 1)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_png = out_dir / "score_vs_cpu_pareto.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    # Also save Pareto set and knee metadata
    pareto_csv = out_dir / "pareto_points.csv"
    knee_csv   = out_dir / "pareto_knee.csv"
    df_pareto = df.iloc[idx_front][["simulation_name","Score_scenario","cpu_seconds"]].copy()
    df_pareto.to_csv(pareto_csv, index=False)
    if knee_xy is not None:
        df_knee = df_pareto[(np.isclose(df_pareto["cpu_seconds"], knee_xy[0]) & np.isclose(df_pareto["Score_scenario"], knee_xy[1]))]
        if not df_knee.empty:
            df_knee.to_csv(knee_csv, index=False)
    print(f"📉 Wrote {out_png}; saved Pareto set to {pareto_csv}")

# -----------------------------
# Pareto component breakdown helpers
# -----------------------------
def mc_species_weights(mc_dir: Path) -> dict:
    """
    Compute species weights (MC total mass share across all years) from MC pop_time_mc.csv.
    """
    mc_tot_path = mc_dir / "pop_time_mc.csv"
    df_mc_tot = read_pop_time_csv(mc_tot_path)
    mc_totals = df_mc_tot.groupby("Species")["Population"].sum().reindex(GROUPS).fillna(0.0)
    if mc_totals.sum() > 0:
        w = (mc_totals / mc_totals.sum()).to_dict()
    else:
        w = {g: 1.0/len(GROUPS) for g in GROUPS}
    return w

def _decompose_components_from_row(row: pd.Series) -> dict:
    """
    Given a per-group metrics row, return component contributions BEFORE species weighting.
    Keys: comp_WAPE, comp_SPAT, comp_FINAL, comp_SHAPE
    """
    # Magnitude
    wape = float(row.get("WAPE_all", np.nan))
    comp_WAPE = W_WAPE * (wape if np.isfinite(wape) else 0.0)

    # Final state (absolute relative error)
    final = float(row.get("Final_rel_err", np.nan))
    comp_FINAL = W_FINAL * (abs(final) if np.isfinite(final) else 0.0)

    # Shape: correlation + DTW
    r = float(row.get("r_clamped", 0.0))
    dtw_n = float(row.get("DTW_norm", np.nan))
    shape_base = SHAPE_R_W*(1.0 - r) + SHAPE_DTW_W*min(dtw_n/DTW_SCALE_DEFAULT, 2.0 if np.isfinite(dtw_n) else 0.0)
    comp_SHAPE = W_SHAPE * shape_base

    # Spatial (volume-normalised)
    cent = float(row.get("AltCentroidDiff_km_ref_vol", np.nan))
    tail = float(row.get("TailFracDelta_gt1000km_vol", np.nan))
    js   = float(row.get("JSdiv_alt_smoothed_ref_vol", np.nan))
    parts, wsum = [], 0.0
    if np.isfinite(cent):
        parts.append(SPAT_CENT_W * min(abs(cent)/CENTROID_SCALE_KM, 2.0)); wsum += SPAT_CENT_W
    if np.isfinite(tail):
        parts.append(SPAT_TAIL_W * min(abs(tail)/TAIL_SCALE, 2.0));       wsum += SPAT_TAIL_W
    if np.isfinite(js):
        parts.append(SPAT_JS_W   * min(js/JS_SCALE, 2.0));                 wsum += SPAT_JS_W
    if wsum > 0:
        spatial_norm = sum(parts)/wsum
    else:
        vw = row.get("VWAPE_alt", np.nan)
        spatial_norm = float(vw) if isinstance(vw, (int,float)) and np.isfinite(vw) else 0.0
    comp_SPAT = W_SPAT * spatial_norm

    return dict(comp_WAPE=comp_WAPE, comp_SPAT=comp_SPAT, comp_FINAL=comp_FINAL, comp_SHAPE=comp_SHAPE)

def build_scenario_component_table(all_group_df: pd.DataFrame, species_weights: dict) -> pd.DataFrame:
    """
    From per-group metrics (all scenarios), build scenario-level component contributions,
    weighting each group's components by MC species share.
    Returns a DataFrame with columns:
      simulation_name, comp_WAPE, comp_SPAT, comp_FINAL, comp_SHAPE, rebuilt_score
    """
    if all_group_df.empty:
        return pd.DataFrame(columns=["simulation_name","comp_WAPE","comp_SPAT","comp_FINAL","comp_SHAPE","rebuilt_score"])
    tmp = all_group_df.copy()
    # Decompose per-row
    decomp = tmp.apply(_decompose_components_from_row, axis=1, result_type="expand")
    for c in decomp.columns:
        tmp[c] = decomp[c]
    # Species weights
    tmp["w_species"] = tmp["Group"].map(species_weights).fillna(0.0)
    for c in ["comp_WAPE","comp_SPAT","comp_FINAL","comp_SHAPE"]:
        tmp[c] = tmp[c] * tmp["w_species"]
    agg = (tmp.groupby("simulation_name")[["comp_WAPE","comp_SPAT","comp_FINAL","comp_SHAPE"]]
              .sum()
              .reset_index())
    agg["rebuilt_score"] = agg[["comp_WAPE","comp_SPAT","comp_FINAL","comp_SHAPE"]].sum(axis=1)
    return agg

def plot_pareto_component_stacks(scores_df: pd.DataFrame, all_group_df: pd.DataFrame, species_weights: dict, out_dir: Path):
    """
    Print the Pareto-front simulation names and create stacked bar charts showing
    component contributions for those scenarios.
    Outputs:
      - pareto_component_stacks.png/pdf (all Pareto scenarios side-by-side)
      - pareto_component_breakdown.csv (table of contributions)
      - pareto_component_stack_<scenario>.png/pdf (individual bars for each)
    """
    df = scores_df.copy()
    df = df[np.isfinite(df["Score_scenario"]) & np.isfinite(df["cpu_seconds"])]
    if df.empty or all_group_df.empty:
        print("[INFO] No data available to plot Pareto component stacks.")
        return

    pts = df[["cpu_seconds","Score_scenario"]].to_numpy(float)
    idx_front = _pareto_front_indices(pts)
    if idx_front.size == 0:
        print("[INFO] No Pareto front found.")
        return

    # Order front by CPU for readability
    front = pts[idx_front]
    order_front = np.argsort(front[:,0])
    idx_front = idx_front[order_front]
    pareto_names = df.iloc[idx_front]["simulation_name"].tolist()

    # Print the list
    print("\n=== Pareto-front simulations (by increasing CPU) ===")
    for i, name in enumerate(pareto_names, 1):
        sc = float(df.loc[df["simulation_name"]==name, "Score_scenario"].iloc[0])
        cp = float(df.loc[df["simulation_name"]==name, "cpu_seconds"].iloc[0])
        print(f"{i:2d}. {name:>30s}   CPU={cp:.2f}s   Score={sc:.4f}")
    print("====================================================\n")

    # Print the 5 worst scenarios (highest score; ties -> higher CPU)
    worst5 = (
        df[["simulation_name", "Score_scenario", "cpu_seconds"]]
        .sort_values(["Score_scenario", "cpu_seconds"], ascending=[False, False])
        .head(5)
    )

    print("=== 5 worst scenarios (by highest Score_scenario) ===")
    for i, r in enumerate(worst5.itertuples(index=False), 1):
        print(f"{i:2d}. {r.simulation_name:>30s}   CPU={r.cpu_seconds:.2f}s   Score={r.Score_scenario:.4f}")
    print("======================================================\n")

    # Build contributions and filter to Pareto set
    scen_tbl = build_scenario_component_table(all_group_df, species_weights)
    # attach order by score for plotting
    scen_tbl = scen_tbl.merge(df[["simulation_name","Score_scenario"]], on="simulation_name", how="left")
    scen_tbl = scen_tbl[scen_tbl["simulation_name"].isin(pareto_names)]
    # keep in the same order as pareto_names (cpu ascending)
    scen_tbl["__order__"] = scen_tbl["simulation_name"].apply(lambda s: pareto_names.index(s))
    scen_tbl = scen_tbl.sort_values("__order__")
    scen_tbl.drop(columns="__order__", inplace=True)

    # Save CSV
    scen_tbl[["simulation_name","comp_WAPE","comp_SPAT","comp_FINAL","comp_SHAPE","rebuilt_score","Score_scenario"]].to_csv(
        out_dir / "pareto_component_breakdown.csv", index=False
    )

    # Combined stacked bar across Pareto scenarios
    labels = scen_tbl["simulation_name"].tolist()
    comps = ["comp_WAPE","comp_SPAT","comp_FINAL","comp_SHAPE"]
    X = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    plt.figure(figsize=(max(10, 0.6*len(labels)+4), 6))
    for c in comps:
        vals = scen_tbl[c].to_numpy(float)
        plt.bar(X, vals, bottom=bottoms, label=c.replace("comp_","").title())
        bottoms += vals
    plt.xticks(X, labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Contribution to scenario score (lower total is better)")
    plt.title("Pareto scenarios — component breakdown")
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "pareto_component_stacks.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"🧱 Wrote {out_png}")

    # Individual per-scenario bars
    for _, r in scen_tbl.iterrows():
        labs = ["WAPE","Spatial","Final","Shape"]
        vals = [r["comp_WAPE"], r["comp_SPAT"], r["comp_FINAL"], r["comp_SHAPE"]]
        plt.figure(figsize=(5.2, 3.6))
        b = 0.0
        for lab, v in zip(labs, vals):
            plt.bar([0], [v], bottom=[b], label=lab)
            b += v
        plt.xticks([0], [r["simulation_name"]], rotation=30, ha="right", fontsize=8)
        plt.ylabel("Contribution")
        plt.title(f"Component breakdown — {r['simulation_name']}")
        plt.legend()
        plt.tight_layout()
        fn = f"pareto_component_stack_{r['simulation_name']}".replace("/", "_")
        plt.savefig(out_dir / f"{fn}.png", dpi=180)
        plt.close()

# -----------------------------
# Factor importance & diagnostic plots
# -----------------------------
def _parse_sim_name(name: str):
    """
    Extract model_type, n_shells, n_species from simulation_name.
    Supports names like:
      - 'circular_nshell_20'
      - 'frag_spread_nshell_40_sp_9'
      - 'elliptical_nshell_25'
    """
    model_type = name.split("_nshell_")[0]
    nshells = None
    nspecies = None
    m = re.search(r"_nshell_(\d+)", name)
    if m:
        try: nshells = int(m.group(1))
        except Exception: nshells = None
    m2 = re.search(r"_sp_(\d+)", name)
    if m2:
        try: nspecies = int(m2.group(1))
        except Exception: nspecies = None
    return model_type, nshells, nspecies

def add_factor_columns(scores_df: pd.DataFrame) -> pd.DataFrame:
    df = scores_df.copy()
    parsed = df["simulation_name"].astype(str).apply(_parse_sim_name)
    df["model_type"] = parsed.apply(lambda t: t[0])
    df["n_shells"]   = parsed.apply(lambda t: t[1])
    df["n_species"]  = parsed.apply(lambda t: t[2])
    # Fill missing species with mode (if any) to keep models happy
    if df["n_species"].isna().all():
        df["n_species"] = 0
    else:
        try:
            mode_val = df["n_species"].dropna().mode().iloc[0]
            df["n_species"] = df["n_species"].fillna(mode_val)
        except Exception:
            df["n_species"] = df["n_species"].fillna(0)
    return df

def plot_score_cpu_colored(df: pd.DataFrame, out_dir: Path):
    """
    Scatter of Score vs CPU, color by model_type, marker size by n_shells.
    """
    if df.empty: return
    if not {"Score_scenario","cpu_seconds","model_type","n_shells"}.issubset(df.columns):
        return
    d = df.dropna(subset=["Score_scenario","cpu_seconds"])
    if d.empty: return

    # Map model types to specific colors
    mtypes = sorted(d["model_type"].astype(str).unique())
    # Use distinct colors for each model type
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color_map = {m: color_palette[i % len(color_palette)] for i, m in enumerate(mtypes)}
    
    # Assign colors directly to each point
    point_colors = [color_map[m] for m in d["model_type"].astype(str)]
    sizes = 40 + 4 * d["n_shells"].astype(float).fillna(d["n_shells"].median() if "n_shells" in d else 20)

    plt.figure(figsize=(9,6))
    sc = plt.scatter(d["cpu_seconds"], d["Score_scenario"], c=point_colors, s=sizes, alpha=0.75)
    # Build a simple legend for model types with matching colors
    for m in mtypes:
        mask = d["model_type"].astype(str) == m
        if mask.any():
            plt.scatter([], [], label=m, s=80, alpha=0.9, c=color_map[m])
    plt.xlabel("CPU time (seconds)")
    plt.ylabel("Composite scenario score (lower is better)")
    plt.title("Score vs CPU — colored by model type, size ∝ n_shells")
    plt.ylim(0, 1)
    plt.legend(title="model_type", fontsize=9)
    plt.tight_layout()
    png = out_dir / "score_vs_cpu_colored.png"
    plt.savefig(png, dpi=180)
    plt.close()
    print(f"🎯 Wrote {png}")

def run_factor_importance(df_in: pd.DataFrame, out_dir: Path, all_group_df: pd.DataFrame | None = None):
    """
    Compute interpretable factor importance:
      - Type-II ANOVA with partial eta^2 for Score and log(CPU)
      - Permutation importance via RandomForest for both
      - Pareto membership odds (logistic regression)
    Writes CSVs and bar plots into out_dir.
    Falls back gracefully if optional deps are missing.
    """
    df = df_in.copy()
    # Clean & prepare
    df = df.dropna(subset=["Score_scenario","cpu_seconds","model_type","n_shells"])
    if df.empty:
        print("[WARN] Importance: nothing to analyze.")
        return

    # Fill species
    if df["n_species"].isna().all():
        df["n_species"] = 0
    else:
        try:
            mode_val = df["n_species"].dropna().mode().iloc[0]
            df["n_species"] = df["n_species"].fillna(mode_val)
        except Exception:
            df["n_species"] = df["n_species"].fillna(0)

    # Save factor table
    df.to_csv(out_dir / "factor_table.csv", index=False)

    # ---------- Type-II ANOVA (statsmodels) ----------
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        # Score model
        model_s = ols("Score_scenario ~ C(model_type) + n_shells + n_species + C(model_type):n_shells + C(model_type):n_species + n_shells:n_species", data=df).fit()
        anova_s = anova_lm(model_s, typ=2)
        ss_res  = anova_s.loc["Residual","sum_sq"]
        anova_s["partial_eta_sq"] = anova_s["sum_sq"] / (anova_s["sum_sq"] + ss_res)
        anova_s.to_csv(out_dir / "importance_score_anova.csv")
        # plot
        eff = anova_s.drop(index="Residual", errors="ignore")["partial_eta_sq"].sort_values(ascending=True)
        plt.figure(figsize=(8,5))
        plt.barh(eff.index.astype(str), eff.values)
        plt.xlabel("Partial $\eta^2$ (Score)")
        plt.tight_layout()
        plt.savefig(out_dir / "importance_score_bar.png", dpi=180); plt.close()

        # CPU model
        df["log_cpu"] = np.log(np.clip(df["cpu_seconds"].astype(float), 1e-9, None))
        model_c = ols("log_cpu ~ C(model_type) + n_shells + n_species + C(model_type):n_shells + C(model_type):n_species + n_shells:n_species", data=df).fit()
        anova_c = anova_lm(model_c, typ=2)
        ss_resc = anova_c.loc["Residual","sum_sq"]
        anova_c["partial_eta_sq"] = anova_c["sum_sq"] / (anova_c["sum_sq"] + ss_resc)
        anova_c.to_csv(out_dir / "importance_cpu_anova.csv")
        effc = anova_c.drop(index="Residual", errors="ignore")["partial_eta_sq"].sort_values(ascending=True)
        plt.figure(figsize=(8,5))
        plt.barh(effc.index.astype(str), effc.values)
        plt.xlabel("Partial $\eta^2$ (log CPU)")
        plt.tight_layout()
        plt.savefig(out_dir / "importance_cpu_bar.png", dpi=180); plt.close()
    except Exception as e:
        with open(out_dir / "importance_anova_warning.txt", "w") as f:
            f.write(f"ANOVA skipped: {type(e).__name__}: {e}\n")
        print(f"[WARN] ANOVA skipped: {e}")

    # ---------- Permutation importance (sklearn) ----------
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        import pandas as _pd
        X = pd.get_dummies(df[["model_type","n_shells","n_species"]], drop_first=True)
        # Score
        rf_s = RandomForestRegressor(n_estimators=400, random_state=42)
        rf_s.fit(X, df["Score_scenario"])
        pi_s = permutation_importance(rf_s, X, df["Score_scenario"], n_repeats=25, random_state=42, scoring="neg_mean_squared_error")
        imp_s = _pd.DataFrame({"feature": X.columns, "importance": pi_s.importances_mean}).sort_values("importance", ascending=True)
        imp_s.to_csv(out_dir / "importance_score_permutation.csv", index=False)
        plt.figure(figsize=(8,5)); plt.barh(imp_s["feature"], imp_s["importance"]); plt.xlabel("Permutation importance (Score)"); plt.tight_layout()
        plt.savefig(out_dir / "importance_score_permutation.png", dpi=180); plt.close()

        # CPU (raw, not log)
        y_cpu = df["cpu_seconds"].astype(float)
        rf_c = RandomForestRegressor(n_estimators=400, random_state=42)
        rf_c.fit(X, y_cpu)
        pi_c = permutation_importance(rf_c, X, y_cpu, n_repeats=25, random_state=42, scoring="neg_mean_squared_error")
        imp_c = _pd.DataFrame({"feature": X.columns, "importance": pi_c.importances_mean}).sort_values("importance", ascending=True)
        imp_c.to_csv(out_dir / "importance_cpu_permutation.csv", index=False)
        plt.figure(figsize=(8,5)); plt.barh(imp_c["feature"], imp_c["importance"]); plt.xlabel("Permutation importance (CPU)"); plt.tight_layout()
        plt.savefig(out_dir / "importance_cpu_permutation.png", dpi=180); plt.close()

        # ---- Compact grid heatmap of grouped permutation importance (model_type, n_shells, n_species) ----
        # Group one-hot model_type columns into a single factor, keep numeric n_shells / n_species as-is.
        def _group_perm(imp_df: _pd.DataFrame) -> dict:
            mt_mask = imp_df["feature"].str.startswith("model_type_")
            mt_val = float(imp_df.loc[mt_mask, "importance"].sum())
            ns_val = float(imp_df.loc[imp_df["feature"]=="n_shells", "importance"].sum())
            sp_val = float(imp_df.loc[imp_df["feature"]=="n_species", "importance"].sum())
            return {"model_type": mt_val, "n_shells": ns_val, "n_species": sp_val}

        g_score = _group_perm(imp_s)
        g_cpu   = _group_perm(imp_c)

        grid = _pd.DataFrame({
            "Score": [g_score["model_type"], g_score["n_shells"], g_score["n_species"]],
            "CPU":   [g_cpu["model_type"],   g_cpu["n_shells"],   g_cpu["n_species"]],
        }, index=["model_type","n_shells","n_species"])

        grid.to_csv(out_dir / "importance_compact_grouped.csv")

        # Column-normalize per target (column-wise)
        grid_norm = grid.copy()
        for c in grid_norm.columns:
            s = float(grid_norm[c].sum())
            if s > 0:
                grid_norm[c] = grid_norm[c] / s

        # Small heatmap without title/colorbar, with in-cell labels
        fig, ax = plt.subplots(figsize=(4.2, 2.9))
        im = ax.imshow(grid_norm.values, aspect="auto")
        ax.set_xticks(range(len(grid_norm.columns)))
        ax.set_xticklabels(grid_norm.columns, fontsize=10)
        ax.set_yticks(range(len(grid_norm.index)))
        ax.set_yticklabels(["model type", "# shells", "# species"], fontsize=10)
        for i in range(grid_norm.shape[0]):
            for j in range(grid_norm.shape[1]):
                val = grid_norm.iat[i, j]
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=9, color="white" if val > 0.5 else "black")
        plt.tight_layout()
        fig.savefig(out_dir / "importance_compact_heatmap.png", dpi=200)
        plt.close(fig)

        # ---- Species-focused compact heatmap (N, D, B) for per-species Score_group ----
        if all_group_df is not None and not all_group_df.empty:
            focus_species = ["N", "D", "B"]
            factors = df[["simulation_name","model_type","n_shells","n_species"]].drop_duplicates()
            col_map = {}
            for sp in focus_species:
                sub = all_group_df[all_group_df["Group"] == sp][["simulation_name","Score_group"]].dropna()
                if sub.empty:
                    continue
                merged = sub.merge(factors, on="simulation_name", how="inner").dropna(subset=["model_type","n_shells","n_species"])
                if merged.empty:
                    continue
                Xsp = _pd.get_dummies(merged[["model_type","n_shells","n_species"]], drop_first=True)
                ysp = merged["Score_group"].astype(float)
                if Xsp.empty or ysp.empty:
                    continue
                rf_sp = RandomForestRegressor(n_estimators=400, random_state=42)
                rf_sp.fit(Xsp, ysp)
                pis = permutation_importance(rf_sp, Xsp, ysp, n_repeats=25, random_state=42, scoring="neg_mean_squared_error")
                imp_sp = _pd.DataFrame({"feature": Xsp.columns, "importance": pis.importances_mean})
                g_sp = _group_perm(imp_sp)
                col_map[sp] = [g_sp["model_type"], g_sp["n_shells"], g_sp["n_species"]]

            if col_map:
                grid_sp = _pd.DataFrame(col_map, index=["model_type","n_shells","n_species"])
                # normalize per column
                grid_sp_norm = grid_sp.copy()
                for c in grid_sp_norm.columns:
                    s = float(grid_sp_norm[c].sum())
                    if s > 0:
                        grid_sp_norm[c] = grid_sp_norm[c] / s
                fig2, ax2 = plt.subplots(figsize=(5.0, 2.9))
                im2 = ax2.imshow(grid_sp_norm.values, aspect="auto")
                ax2.set_xticks(range(len(grid_sp_norm.columns)))
                ax2.set_xticklabels(grid_sp_norm.columns, fontsize=10)
                ax2.set_yticks(range(len(grid_sp_norm.index)))
                ax2.set_yticklabels(["model type", "# shells", "# species"], fontsize=10)
                for i in range(grid_sp_norm.shape[0]):
                    for j in range(grid_sp_norm.shape[1]):
                        val = grid_sp_norm.iat[i, j]
                        ax2.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                                 fontsize=9, color="white" if val > 0.5 else "black")
                plt.tight_layout()
                fig2.savefig(out_dir / "importance_compact_species_heatmap.png", dpi=200)
                plt.close(fig2)
    except Exception as e:
        with open(out_dir / "importance_permutation_warning.txt", "w") as f:
            f.write(f"Permutation importance skipped: {type(e).__name__}: {e}\n")
        print(f"[WARN] Permutation importance skipped: {e}")

    # ---------- Pareto membership logistic regression (optional) ----------
    try:
        from sklearn.linear_model import LogisticRegression
        # Build Pareto membership
        pts = df[["cpu_seconds","Score_scenario"]].to_numpy(float)
        idx_pf = set(_pareto_front_indices(pts).tolist())
        df["on_pareto"] = [1 if i in idx_pf else 0 for i in range(len(df))]
        Xp = pd.get_dummies(df[["model_type","n_shells","n_species"]], drop_first=True)
        lr = LogisticRegression(max_iter=2000)
        lr.fit(Xp, df["on_pareto"])
        odds = pd.DataFrame({"feature": Xp.columns, "odds_ratio": np.exp(lr.coef_[0])}).sort_values("odds_ratio", ascending=True)
        odds.to_csv(out_dir / "pareto_membership_odds.csv", index=False)
        plt.figure(figsize=(8,5)); plt.barh(odds["feature"], odds["odds_ratio"]); plt.xlabel("Odds ratio (on Pareto front)"); plt.tight_layout()
        plt.savefig(out_dir / "pareto_membership_odds.png", dpi=180); plt.close()
    except Exception as e:
        with open(out_dir / "pareto_membership_warning.txt", "w") as f:
            f.write(f"Pareto membership skipped: {type(e).__name__}: {e}\n")
        print(f"[WARN] Pareto membership skipped: {e}")

###############################################################################
# Compact ANOVA share heatmap (helper functions)
###############################################################################

def _anova_share(y, X):
    """Return dict of partial-eta-squared for the 3 factors, normalized to sum=1.
    y: pandas Series with name (e.g., Score_scenario); X: DataFrame with columns
       model_type, n_shells, n_species (model_type treated categorically).
    """
    import pandas as _pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df = _pd.concat([y, X], axis=1).dropna()
    if df.empty:
        return {"model_type": np.nan, "n_shells": np.nan, "n_species": np.nan}

    model = ols("y ~ C(model_type) + n_shells + n_species",
                data=df.rename(columns={y.name: "y"})).fit()
    a = anova_lm(model, typ=2)
    if "sum_sq" not in a.columns or "Residual" not in a.index:
        return {"model_type": np.nan, "n_shells": np.nan, "n_species": np.nan}

    ss_res = float(a.loc["Residual", "sum_sq"])
    a = a.drop(index="Residual", errors="ignore")
    a["partial_eta_sq"] = a["sum_sq"] / (a["sum_sq"] + ss_res)

    out = {
        "model_type": float(a.loc["C(model_type)", "partial_eta_sq"]) if "C(model_type)" in a.index else np.nan,
        "n_shells":   float(a.loc["n_shells",       "partial_eta_sq"]) if "n_shells"       in a.index else np.nan,
        "n_species":  float(a.loc["n_species",      "partial_eta_sq"]) if "n_species"      in a.index else np.nan,
    }
    vals = np.array([out["model_type"], out["n_shells"], out["n_species"]], float)
    s = np.nansum(vals)
    if s > 0:
        vals = vals / s
    out["model_type"], out["n_shells"], out["n_species"] = vals
    return out


def plot_anova_share_heatmap(scores_df: pd.DataFrame, all_group_df: pd.DataFrame, out_dir: Path):
    """
    Compact heatmap of ANOVA 'shares' (partial η² normalized across the three factors)
    for Overall Score, CPU time, and per-species (N, D, B).
    Saves PNG/PDF in out_dir as importance_compact_anova_shares.*
    """
    import pandas as _pd
    import matplotlib.pyplot as plt

    # Factor table
    factors = scores_df[["simulation_name","model_type","n_shells","n_species","Score_scenario","cpu_seconds"]].copy()
    factors = factors.dropna(subset=["model_type","n_shells","n_species"]).drop_duplicates()

    grid_cols = []

    # Overall Score
    s_sh = _anova_share(
        y=factors.set_index("simulation_name")["Score_scenario"],
        X=factors.set_index("simulation_name")[["model_type","n_shells","n_species"]],
    )
    grid_cols.append(("Overall Score", s_sh))

    # CPU (raw, not log)
    c_sh = _anova_share(
        y=factors.set_index("simulation_name")["cpu_seconds"],
        X=factors.set_index("simulation_name")[["model_type","n_shells","n_species"]],
    )
    grid_cols.append(("CPU time", c_sh))

    # Per-species (N, D, B)
    for sp in ["N", "D", "B"]:
        sub = all_group_df[all_group_df["Group"]==sp][["simulation_name","Score_group"]].dropna()
        if sub.empty:
            continue
        merged = sub.merge(factors[["simulation_name","model_type","n_shells","n_species"]],
                           on="simulation_name", how="inner")
        if merged.empty:
            continue
        sh = _anova_share(
            y=merged.set_index("simulation_name")["Score_group"].rename("y"),
            X=merged.set_index("simulation_name")[ ["model_type","n_shells","n_species"] ],
        )
        grid_cols.append((sp, sh))

    if not grid_cols:
        return

    mat = {}
    for col_name, d in grid_cols:
        mat[col_name] = [d["model_type"], d["n_shells"], d["n_species"]]
    grid = _pd.DataFrame(mat, index=["model type", "# shells", "# species"])    

    # Preferred column order if present
    order = [c for c in ["N","D","B","Overall Score","CPU time"] if c in grid.columns]
    grid = grid[order]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    im = ax.imshow(grid.values.astype(float), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels(grid.columns, fontsize=11)
    ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(grid.index, fontsize=11)

    def _txt_color(val):
        r,g,b,_ = im.cmap(im.norm(val))
        lum = 0.2126*r + 0.7152*g + 0.0722*b
        return "black" if lum >= 0.58 else "white"

    V = grid.values.astype(float)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i,j]
            if np.isfinite(v):
                ax.text(j, i, f"{100*v:.1f}%", ha="center", va="center", fontsize=10, color=_txt_color(v))
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=10, color="black")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of variance explained (partial $\\eta^2$)")
    cb.set_ticks([0,0.25,0.5,0.75,1.0]); cb.set_ticklabels(["0","25%","50%","75%","100%"])

    plt.tight_layout()
    fig.savefig(out_dir / "importance_compact_anova_shares.png", dpi=220)
    plt.close(fig)

# -----------------------------
# Best scenarios comparison plots
# -----------------------------
def plot_best_scenarios_comparison(scores_df: pd.DataFrame, root_dir: Path, mc_dir: Path, out_dir: Path):
    """
    Create grouped_population_comparison.png plots for the best-scoring scenario
    of each model type (elliptical, frag_spread, circular).
    """
    if scores_df.empty or "model_type" not in scores_df.columns:
        print("[WARN] Cannot create best scenarios comparison: missing model_type column")
        return
    
    # Find best scenario (lowest score) for each model type
    best_scenarios = {}
    for model_type in ["elliptical", "frag_spread", "circular"]:
        model_df = scores_df[scores_df["model_type"] == model_type].copy()
        if model_df.empty:
            print(f"[WARN] No scenarios found for model_type={model_type}")
            continue
        
        # Get best scoring scenario (lowest Score_scenario)
        best = model_df.loc[model_df["Score_scenario"].idxmin()]
        best_scenarios[model_type] = best["simulation_name"]
        print(f"Best {model_type}: {best['simulation_name']} (score={best['Score_scenario']:.4f})")
    
    if not best_scenarios:
        print("[WARN] No best scenarios found to plot")
        return
    
    # Load MC data
    mc_tot_path = mc_dir / "pop_time_mc.csv"
    if not mc_tot_path.exists():
        print(f"[WARN] MC data not found at {mc_tot_path}")
        return
    
    mc_df = pd.read_csv(mc_tot_path)
    mc_df["Species"] = mc_df["Species"].astype(str)
    mc_df["Year"] = mc_df["Year"].astype(int)
    mc_df["Population"] = mc_df["Population"].astype(float)
    
    # Load SSEM data for all best scenarios
    ssem_data = {}
    for model_type, sim_name in best_scenarios.items():
        sim_dir = root_dir / sim_name
        ssem_pop_path = sim_dir / "pop_time.csv"
        
        if not ssem_pop_path.exists():
            print(f"[WARN] pop_time.csv not found for {sim_name}")
            continue
        
        # Load SSEM data
        ssem_df = read_pop_time_csv(ssem_pop_path)
        
        # Create pivot table
        pivot_ssem = ssem_df.pivot_table(
            index="Year", 
            columns="Species", 
            values="Population", 
            aggfunc="sum"
        ).fillna(0.0)
        
        ssem_data[model_type] = {
            'pivot': pivot_ssem,
            'years': np.array(sorted(pivot_ssem.index.values), dtype=int),
            'sim_name': sim_name
        }
    
    if not ssem_data:
        print("[WARN] No SSEM data loaded for any best scenarios")
        return
    
    # Create one plot with separate subplots for each species
    species_groups = ['S', 'Su', 'Sns', 'N', 'D', 'B']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    
    # Color palette for model types
    model_colors = {
        'elliptical': '#1f77b4',
        'frag_spread': '#2ca02c',
        'circular': '#d62728'
    }
    
    for ax, group in zip(axes, species_groups):
        # Plot MC line (shared across all model types)
        sub_mc = mc_df[mc_df["Species"] == group]
        if not sub_mc.empty:
            ax.plot(sub_mc["Year"], sub_mc["Population"], '--', 
                   label='MC', linewidth=2, color='black', alpha=0.7)
        
        # Plot SSEM lines for each model type
        for model_type, data in ssem_data.items():
            years = data['years']
            pivot_ssem = data['pivot']
            
            if group in pivot_ssem.columns:
                ax.plot(years, pivot_ssem[group].reindex(years, fill_value=0.0), 
                       label=f'SSEM {model_type}', linewidth=2, 
                       color=model_colors.get(model_type, '#9467bd'))
        
        ax.set_title(group)
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    axes[-2].set_xlabel("Year")
    axes[-1].set_xlabel("Year")
    
    # Create title with all best scenario names
    best_names = [f"{mt}: {data['sim_name']}" for mt, data in ssem_data.items()]
    title = "SSEM vs MOCAT-MC: Grouped Population Over Time\nBest scenarios: " + ", ".join(best_names)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_png = out_dir / "best_all_models_grouped_population_comparison.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"📊 Wrote {out_png}")
    
    # Create a second plot with only S, N, D, B (excluding Su and Sns)
    species_groups_filtered = ['S', 'N', 'D', 'B']
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes2 = axes2.flatten()
    
    for ax, group in zip(axes2, species_groups_filtered):
        # Plot MC line (shared across all model types)
        sub_mc = mc_df[mc_df["Species"] == group]
        if not sub_mc.empty:
            ax.plot(sub_mc["Year"], sub_mc["Population"], '--', 
                   label='MC', linewidth=2, color='black', alpha=0.7)
        
        # Plot SSEM lines for each model type
        for model_type, data in ssem_data.items():
            years = data['years']
            pivot_ssem = data['pivot']
            
            if group in pivot_ssem.columns:
                ax.plot(years, pivot_ssem[group].reindex(years, fill_value=0.0), 
                       label=f'SSEM {model_type}', linewidth=2, 
                       color=model_colors.get(model_type, '#9467bd'))
        
        ax.set_title(group)
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    axes2[-2].set_xlabel("Year")
    axes2[-1].set_xlabel("Year")
    
    # Create title with all best scenario names
    best_names = [f"{mt}: {data['sim_name']}" for mt, data in ssem_data.items()]
    title = "SSEM vs MOCAT-MC: Grouped Population Over Time (S, N, D, B)\nBest scenarios: " + ", ".join(best_names)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_png2 = out_dir / "best_all_models_grouped_population_comparison_SNDB.png"
    plt.savefig(out_png2, dpi=300)
    plt.close()
    print(f"📊 Wrote {out_png2}")

# -----------------------------
# Standalone run configuration
# -----------------------------
def main():
    """
    Standalone entry point.
    Configure default paths here or via environment variables:
      - SSEM_ROOT: root folder containing scenario subfolders
      - SSEM_MC:   folder containing MC truth pop_time_mc.csv and pop_time_alt_mc.csv
      - SSEM_OUT:  output folder for cross-scenario plots/CSVs (defaults to SSEM_ROOT)
      - SSEM_REF_STEP_KM: reference grid step (km) for altitude metrics (default 10)
      - SSEM_SIGMA_KM:    Gaussian sigma (km) for spatial smoothing (default 50)
    """
    # ---- User-configurable defaults ----
    ROOT_DIR_DEFAULT = "./grid_search"              # change if needed
    MC_DIR_DEFAULT   = "/Users/indigobrownhall/Code/pyssem/pyssem/VnV"                 # change if needed
    OUT_DIR_DEFAULT  = None                          # None -> defaults to ROOT_DIR
    REF_STEP_DEFAULT = 10.0
    SIGMA_DEFAULT    = 50.0

    # ---- Resolve configuration (env overrides allowed) ----
    ROOT_DIR = Path(os.environ.get("SSEM_ROOT", ROOT_DIR_DEFAULT)).expanduser().resolve()
    MC_DIR   = Path(os.environ.get("SSEM_MC", MC_DIR_DEFAULT)).expanduser().resolve()
    OUT_DIR  = Path(os.environ.get("SSEM_OUT", str(ROOT_DIR if OUT_DIR_DEFAULT is None else OUT_DIR_DEFAULT))).expanduser().resolve()
    REF_STEP_KM = float(os.environ.get("SSEM_REF_STEP_KM", REF_STEP_DEFAULT))
    SIGMA_KM    = float(os.environ.get("SSEM_SIGMA_KM", SIGMA_DEFAULT))

    print("\n=== VnV summary configuration ===")
    print(f"ROOT_DIR   : {ROOT_DIR}")
    print(f"MC_DIR     : {MC_DIR}")
    print(f"OUT_DIR    : {OUT_DIR}") 
    print(f"ref_step_km: {REF_STEP_KM}")
    print(f"sigma_km   : {SIGMA_KM}")
    print("=================================\n")

    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Root folder not found: {ROOT_DIR}")
    if not MC_DIR.exists():
        raise FileNotFoundError(f"MC folder not found: {MC_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Precompute MC species weights (used for component breakdown plots)
    try:
        species_w = mc_species_weights(MC_DIR)
    except Exception as e:
        print(f"[WARN] Could not compute MC species weights: {e}")
        species_w = {g: 1.0/len(GROUPS) for g in GROUPS}

    # Discover scenarios: each subfolder with both CSVs present
    scenario_dirs = [p for p in ROOT_DIR.iterdir() if p.is_dir()]

    all_group_rows = []
    scenario_scores = []
    for sim_dir in sorted(scenario_dirs):
        if not (sim_dir / "pop_time.csv").exists() or not (sim_dir / "pop_time_alt.csv").exists():
            print(f"[SKIP] {sim_dir.name}: missing pop_time*.csv")
            continue
        try:
            df_metrics, sc = process_scenario(sim_dir, MC_DIR,
                                              ref_step_km=REF_STEP_KM,
                                              sigma_km=SIGMA_KM)
            all_group_rows.append(df_metrics)
            scenario_scores.append((sim_dir.name, sc))
        except Exception as e:
            print(f"[ERROR] {sim_dir.name}: {e}")

    if not scenario_scores:
        print("No scenarios processed.")
        return

    # Write cross-scenario CSVs
    scores_df = pd.DataFrame(scenario_scores, columns=["simulation_name","Score_scenario"])
    scores_df.to_csv(OUT_DIR / "all_scenario_scores.csv", index=False)
    print(f"📝 Wrote {OUT_DIR / 'all_scenario_scores.csv'}")

    # Attach CPU times
    cpu_map = {}
    for sim_dir in scenario_dirs:
        cpu = read_cpu_seconds(sim_dir)
        if cpu is not None and np.isfinite(cpu):
            cpu_map[sim_dir.name] = float(cpu)
    scores_df["cpu_seconds"] = scores_df["simulation_name"].map(cpu_map)

    # Persist combined table and make knee-Pareto plot
    scores_df.to_csv(OUT_DIR / "all_scenario_scores_with_cpu.csv", index=False)
    print(f"📝 Wrote {OUT_DIR / 'all_scenario_scores_with_cpu.csv'}")
    try:
        plot_score_cpu_pareto(scores_df.dropna(subset=["cpu_seconds"]), OUT_DIR)
    except Exception as e:
        print(f"[WARN] Pareto plot failed: {e}")

    # Add factors parsed from simulation_name
    scores_df = add_factor_columns(scores_df)
    # Diagnostic scatter with color/size encoding
    try:
        plot_score_cpu_colored(scores_df, OUT_DIR)
    except Exception as e:
        print(f"[WARN] Colored scatter failed: {e}")

    if all_group_rows:
        all_groups = pd.concat(all_group_rows, ignore_index=True)
        all_groups.to_csv(OUT_DIR / "all_group_metrics.csv", index=False)
        print(f"📝 Wrote {OUT_DIR / 'all_group_metrics.csv'}")
        plot_group_heatmap(all_groups, OUT_DIR)
        # Factor importance analyses (now that per-group table exists)
        try:
            run_factor_importance(scores_df, OUT_DIR, all_groups)
        except Exception as e:
            print(f"[WARN] Factor importance failed: {e}")
        # Compact ANOVA share heatmap (Overall Score, CPU, and per-species N/D/B)
        try:
            plot_anova_share_heatmap(scores_df, all_groups, OUT_DIR)
        except Exception as e:
            print(f"[WARN] ANOVA share heatmap failed: {e}")
        # Print Pareto-front simulations and plot component stacks for them
        try:
            plot_pareto_component_stacks(scores_df, all_groups, species_w, OUT_DIR)
        except Exception as e:
            print(f"[WARN] Pareto component stack plotting failed: {e}")

    plot_scenario_scores(scenario_scores, OUT_DIR)
    
    # Create grouped population comparison plots for best scenarios per model type
    try:
        plot_best_scenarios_comparison(scores_df, ROOT_DIR, MC_DIR, OUT_DIR)
    except Exception as e:
        print(f"[WARN] Best scenarios comparison plot failed: {e}")
    
    
    print("Done. (Lower scores indicate better agreement with MC.)")

if __name__ == "__main__":
    main()