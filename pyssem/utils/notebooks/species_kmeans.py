# --- setup ---
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# ---------- user inputs ----------
PATH = '/Users/indigobrownhall/Code/pyssem/pyssem/utils/launch/data/ref_scen_SEP2.csv'
OUT_DIR = Path("./species_kmeans_results").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# species rules (your originals)
species_configuration = [
    "T.loc[(T['obj_type'] == 2) & (T['phase'] == 2) & (T['maneuverable'] == 1), 'species_class'] = 'Su'",
    "T.loc[(T['obj_type'] == 2) & (T['mass'] <= 20) & (T['phase'] == 2), 'species_class'] = 'Sns'",
    "T.loc[(T['obj_type'] == 2) & (T['phase'] == 2) & (T['maneuverable'] == 1) & T['const_name'].notna(), 'species_class'] = 'S'",
    "T.loc[(T['obj_type'] >= 3) & (T['diam_char'] >= 0.05) & (T['diam_char']/2 < 10), 'species_class'] = 'N'",
    "T.loc[(T['obj_type'] == 1), 'species_class'] = 'B'"
]

# which species/K to fit (ignore D as requested)
species_k_map = {
    "S":  [1, 3, 5],
    "Su": [1, 3, 5],
    "N":  [5],
    "B":  [1, 2, 3],
}

# outlier filtering strength (IQR multiplier). 2.5–3.5 are typical.
IQR_K = 3.0

# reproducibility
RANDOM_STATE = 0

# ---------- helpers ----------
def assign_species_to_population(T: pd.DataFrame, mapping_rules: list[str]) -> pd.DataFrame:
    T = T.copy()
    T['species_class'] = "Unknown"
    for rule in mapping_rules:
        try:
            # applies rules that reference the local name "T"
            exec(rule, {}, {"T": T})
        except Exception as e:
            print(f"Error applying rule: {rule}\n  {e}")

    print("\nSpecies class distribution (incl. Unknown):")
    print(T['species_class'].value_counts(dropna=False))

    # drop Unknown
    before = len(T)
    T = T[T['species_class'] != "Unknown"].copy()
    print(f"\nRemoved {before - len(T)} rows with species_class == 'Unknown'")
    return T

def remove_outliers_iqr(df: pd.DataFrame, cols=("radius", "mass"), k=3.0) -> pd.DataFrame:
    """
    Keep rows within [Q1 - k*IQR, Q3 + k*IQR] for *each* column; drop otherwise.
    """
    G = df.copy()
    mask = np.ones(len(G), dtype=bool)
    for c in cols:
        x = G[c].astype(float)
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        mask &= (x >= lo) & (x <= hi)
    return G[mask].copy()

def bstar_from_area_mass(area_m2, mass_kg):
    # per your formula: bstar = 2.2 * (area * 1e-6 / mass)
    return 2.2 * (area_m2 * 1e-6 / np.maximum(mass_kg, 1e-12))

# ---------- Elbow (knee) detection for k selection ----------
def select_elbow_k(k_list, inertia_list):
    """
    Pick k at the 'elbow' using the max distance to the line between first and last points.
    Falls back to argmin(inertia) if fewer than 3 points.
    """
    ks = np.asarray(k_list, dtype=float)
    sses = np.asarray(inertia_list, dtype=float)
    if ks.size == 0:
        raise ValueError("select_elbow_k received an empty k_list.")
    if ks.size < 3:
        return int(ks[np.argmin(sses)])

    # Normalize both axes to [0,1] to make distance scale-invariant
    x = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    y = (sses - sses.min()) / (sses.max() - sses.min() + 1e-12)

    # Compute perpendicular distance from each point to the line through endpoints
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denom = np.hypot(x2 - x1, y2 - y1) + 1e-12
    # Line in Ax + By + C = 0 form
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    dists = np.abs(A * x + B * y + C) / denom

    idx = int(np.argmax(dists))
    return int(k_list[idx])
# ---------- load & prepare ----------
T = pd.read_csv(PATH)
T['apogee']  = T['sma'] * (1 + T['ecc'])
T['perigee'] = T['sma'] * (1 - T['ecc'])
T['alt']     = (T['apogee'] + T['perigee']) / 2 - 6378.1

# LEO slice: 200–2000 km
T = T[(T['alt'] <= 2000) & (T['alt'] >= 200)].copy()

# species assignment
assigned_df = assign_species_to_population(T, species_configuration)

# compute radius (m) from characteristic diameter (m)
assigned_df = assigned_df.copy()
assigned_df['radius'] = assigned_df['diam_char'] / 2.0
assigned_df = assigned_df.dropna(subset=['radius','mass'])

# safety: remove non-physical values
assigned_df = assigned_df[(assigned_df['radius'] > 0) & (assigned_df['mass'] > 0)].copy()

# ---------- clustering & plotting ----------
all_centers_rows = []

for sp, ks in species_k_map.items():
    df_sp = assigned_df[assigned_df['species_class'] == sp].copy()
    if df_sp.empty:
        print(f"[WARN] No rows for species {sp}; skipping.")
        continue

    # outlier removal per species (mass & radius)
    n0 = len(df_sp)
    df_sp = remove_outliers_iqr(df_sp, cols=("radius", "mass"), k=IQR_K)
    n1 = len(df_sp)
    print(f"{sp}: removed {n0-n1} outliers via IQR (k={IQR_K}). Remaining: {n1}")

    # features for clustering
    X_raw = df_sp[['radius','mass']].to_numpy(float)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    # also compute area for plotting on x-axis
    df_sp['area'] = np.pi * (df_sp['radius']**2)

    # folder per species
    sp_dir = OUT_DIR / f"species_{sp}"
    sp_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        if len(df_sp) < k:
            print(f"[WARN] {sp} has only {len(df_sp)} rows; cannot fit k={k}. Skipping.")
            continue

        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(X)
        labels = km.labels_
        centers_unscaled = scaler.inverse_transform(km.cluster_centers_)
        inertia = float(km.inertia_)

        # plot: Area (m^2) vs Mass (kg)
        plt.figure(figsize=(7,5))
        sc = plt.scatter(df_sp['area'], df_sp['mass'], c=labels, cmap='tab10', s=16, alpha=0.35, label='Data')
        # overlay centres
        centers_radius = centers_unscaled[:,0]
        centers_mass   = centers_unscaled[:,1]
        centers_area   = np.pi * (centers_radius**2)
        plt.scatter(centers_area, centers_mass, marker='X', s=150, edgecolor='k', linewidths=0.5, c='red', label='K-Means centres')

        plt.xlabel("Area (m²)")
        plt.ylabel("Mass (kg)")
        plt.title(f"{sp}: Area vs Mass with K-Means centres (k={k})")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        out_png = sp_dir / f"{sp}_area_mass_k{k}.png"
        plt.savefig(out_png, dpi=180)
        plt.close()
        print(f"  ⬇️ saved plot → {out_png}")

        # print & store centres
        print(f"\n{sp} — K={k} cluster centres (radius[m], mass[kg], area[m²], bstar):")
        counts = np.bincount(labels, minlength=k)
        for j in range(k):
            r_m   = float(centers_radius[j])
            m_kg  = float(centers_mass[j])
            area  = float(np.pi * r_m**2)
            bstar = float(bstar_from_area_mass(area, m_kg))
            cnt   = int(counts[j])
            print(f"  cluster {j+1}: radius={r_m:.4f} m, mass={m_kg:.4f} kg, area={area:.4f} m², bstar={bstar:.6e}  (n={cnt})")
            all_centers_rows.append({
                "species": sp,
                "k": k,
                "cluster": j+1,
                "radius": r_m,
                "mass": m_kg,
                "area": area,
                "bstar": bstar,
                "count": cnt,
                "inertia": inertia,
            })
        print("")

# write all centres to CSV
centers_df = pd.DataFrame(all_centers_rows)
centers_csv = OUT_DIR / "cluster_centres_by_species.csv"
centers_df.to_csv(centers_csv, index=False)
print(f"📝 wrote {centers_csv}")

# quick per-species summary (best inertia per k)
if not centers_df.empty:
    summary = (
        centers_df.groupby(["species","k"], as_index=False)
                  .agg(best_inertia=("inertia","min"),
                       total_points=("count","sum"))
                  .sort_values(["species","k"])
    )
    summary_csv = OUT_DIR / "cluster_fit_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"🧾 wrote {summary_csv}")

# ---------- Debris (N) elbow plot (left) and best-k radius–mass clusters (right) ----------
try:
    df_N = assigned_df[assigned_df['species_class'] == 'N'].copy()
    if df_N.empty:
        print("[N elbow] No debris rows found; skipping debris elbow plot.")
    else:
        # Remove outliers in the same way as above
        n0 = len(df_N)
        df_N = remove_outliers_iqr(df_N, cols=("radius", "mass"), k=IQR_K)
        n1 = len(df_N)
        print(f"[N elbow] removed {n0-n1} outliers via IQR (k={IQR_K}). Remaining: {n1}")

        # Prepare features
        X_raw = df_N[['radius', 'mass']].to_numpy(float)
        scaler_N = StandardScaler().fit(X_raw)
        Xn = scaler_N.transform(X_raw)

        # Range of k for elbow search
        k_max = int(min(10, len(df_N)))  # cap at 10 and not more than number of samples
        if k_max < 1:
            print("[N elbow] Not enough samples for KMeans; skipping.")
        else:
            k_values = []
            inertias = []
            for k in range(1, k_max + 1):
                try:
                    km_tmp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(Xn)
                    k_values.append(k)
                    inertias.append(float(km_tmp.inertia_))
                except Exception as _e:
                    # In rare cases, KMeans may fail for certain k; skip that k
                    print(f"[N elbow] KMeans failed for k={k}: {_e}")
                    continue

            if len(k_values) == 0:
                print("[N elbow] No successful KMeans fits; skipping plot.")
            else:
                # Choose best k using elbow method
                best_k = select_elbow_k(k_values, inertias)
                print(f"[N elbow] Selected k={best_k} via elbow method.")

                # Fit final model with best_k
                km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto").fit(Xn)
                labels_best = km_best.labels_
                centers_best_unscaled = scaler_N.inverse_transform(km_best.cluster_centers_)

                # Make side-by-side figure (no titles as requested)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Left: elbow curve
                axes[0].plot(k_values, inertias, marker="o")
                axes[0].set_xlabel("Number of clusters (k)")
                axes[0].set_ylabel("Inertia (within-cluster SSE)")
                axes[0].grid(alpha=0.3)
                # mark chosen k
                try:
                    axes[0].axvline(best_k, linestyle="--", alpha=0.6)
                except Exception:
                    pass

                # Right: radius vs mass coloured by best-k labels
                axes[1].scatter(df_N['radius'], df_N['mass'], c=labels_best, cmap='tab10', s=16, alpha=0.35)
                axes[1].scatter(centers_best_unscaled[:, 0], centers_best_unscaled[:, 1],
                                marker='X', s=150, edgecolor='k', linewidths=0.5, c='red')
                axes[1].set_xlabel("Radius (m)")
                axes[1].set_ylabel("Mass (kg)")
                axes[1].grid(alpha=0.3)

                plt.tight_layout()
                debris_dir = OUT_DIR / "species_N"
                debris_dir.mkdir(parents=True, exist_ok=True)
                out_png = debris_dir / "N_elbow_and_radius_mass.png"
                plt.savefig(out_png, dpi=180)
                plt.close()
                print(f"  ⬇️ saved debris elbow+clusters → {out_png}")
except Exception as e:
    print(f"[N elbow] WARNING: {type(e).__name__}: {e}")
# ---------- eccentricity bin search (N, B) ----------
# Goal: for species N and B, find the number of eccentricity bins (1..10) and
# binning scheme (linear vs log) that best represent the distribution of ecc.
# We score each (scheme, k) using an approximate 1D EMD (L1 distance between
# the empirical CDF and the histogram-induced CDF) on a dense grid. Lower is better.

try:
    # ensure required columns exist
    if "ecc" not in assigned_df.columns:
        raise KeyError("'ecc' column not found in assigned_df")

    eps = 1e-6
    ecc_species = {sp: assigned_df.loc[assigned_df["species_class"] == sp, "ecc"].to_numpy(float)
                   for sp in ("N", "B")}

    results_rows = []

    def _cdf_from_hist(edges, counts, grid):
        counts = counts.astype(float)
        total = counts.sum()
        if total <= 0:
            return np.zeros_like(grid)
        p = counts / total
        # CDF at grid: for each x, sum probabilities of all bins fully below x
        # plus partial for current bin using uniform-within-bin assumption.
        cdf = np.zeros_like(grid, dtype=float)
        bin_left = edges[:-1]
        bin_right = edges[1:]
        bin_width = np.maximum(bin_right - bin_left, eps)
        # For vectorization, iterate bins and accumulate
        cum = np.zeros_like(grid, dtype=float)
        for i in range(len(p)):
            left, right, prob = bin_left[i], bin_right[i], p[i]
            mask_full = grid >= right
            mask_part = (grid >= left) & (grid < right)
            cdf[mask_full] += prob
            # linear ramp within the bin assuming uniform density
            cdf[mask_part] += prob * (grid[mask_part] - left) / bin_width[i]
        return cdf

    def _emd_l1_ecdf_vs_hist(x, edges, counts, ngrid=2000):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.nan
        lo = 0.0
        hi = max(float(np.max(x)), eps)
        grid = np.linspace(lo, hi, ngrid)
        # empirical CDF on grid
        xs = np.sort(x)
        ecdf = np.searchsorted(xs, grid, side="right") / xs.size
        # histogram CDF on grid
        cdf_h = _cdf_from_hist(edges, counts, grid)
        # approximate EMD as integral of absolute CDF difference
        emd = np.trapz(np.abs(ecdf - cdf_h), grid)
        # normalize by range so values are in [0,1]
        rng = max(hi - lo, eps)
        return emd / rng

    for sp, x in ecc_species.items():
        # clean vector: finite, within [0,1)
        x = x[np.isfinite(x)]
        x = x[(x >= 0.0) & (x < 1.0)]
        if x.size == 0:
            print(f"[WARN] No valid eccentricities for species {sp}; skipping.")
            continue

        e_max = float(x.max())
        if e_max <= 0:
            print(f"[WARN] All eccentricities are zero for {sp}; only k=1 meaningful.")

        ks = range(1, 11)  # 1..10 bins
        schemes = ("linear", "log")
        sp_rows = []

        for scheme in schemes:
            for k in ks:
                # construct edges
                if scheme == "linear":
                    if e_max <= 0 and k > 1:
                        continue
                    edges = np.linspace(0.0, max(e_max, eps), k+1)
                else:  # log
                    pos = x[x > 0]
                    if pos.size == 0:
                        # fallback to linear if no positive values
                        if e_max <= 0 and k > 1:
                            continue
                        edges = np.linspace(0.0, max(e_max, eps), k+1)
                    else:
                        e_min_pos = float(pos.min())
                        edges = np.concatenate(([0.0], np.geomspace(max(eps, e_min_pos), e_max + eps, k)))
                        # ensure strictly increasing
                        edges = np.unique(edges)
                        if edges.size < k+1:
                            # not enough unique edges; skip this k
                            continue
                # histogram counts
                counts, edges2 = np.histogram(x, bins=edges)
                if np.all(counts == 0):
                    continue
                emd = _emd_l1_ecdf_vs_hist(x, edges2, counts)
                sp_rows.append({"species": sp, "scheme": scheme, "k": k, "emd_l1": float(emd),
                                "e_max": e_max})

        # save per-species score vs k plot
        if sp_rows:
            df_sp = pd.DataFrame(sp_rows)
            for scheme in schemes:
                df_s = df_sp[df_sp["scheme"] == scheme]
                if not df_s.empty:
                    plt.plot(df_s["k"], df_s["emd_l1"], marker="o", label=scheme.capitalize())
            plt.xlabel("Number of bins k")
            plt.ylabel("EMD (L1) between ECDF and histogram CDF")
            plt.title(f"Eccentricity bin score vs k — {sp}")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_png = OUT_DIR / f"ecc_score_vs_k_{sp}.png"
            plt.savefig(out_png, dpi=180)
            plt.close()
            print(f"  ⬇️ saved ecc score plot → {out_png}")

            # choose best (lowest emd)
            best = df_sp.loc[df_sp["emd_l1"].idxmin()].to_dict()
            results_rows.extend(df_sp.to_dict("records"))

            # make a comparison histogram for the best setting
            best_k = int(best["k"]) ; best_scheme = best["scheme"]
            if best_scheme == "linear":
                edges_best = np.linspace(0.0, max(e_max, eps), best_k+1)
            else:
                pos = x[x > 0]
                if pos.size == 0:
                    edges_best = np.linspace(0.0, max(e_max, eps), best_k+1)
                else:
                    e_min_pos = float(pos.min())
                    edges_best = np.concatenate(([0.0], np.geomspace(max(eps, e_min_pos), e_max + eps, best_k)))
            counts_best, edges_best = np.histogram(x, bins=edges_best)
            # normalized for plotting
            p_best = counts_best / max(counts_best.sum(), 1)

            plt.figure(figsize=(7,5))
            plt.hist(x, bins=50, density=True, alpha=0.35, label="Empirical (50 bins)")
            # plot best histogram as step
            centers = 0.5*(edges_best[:-1] + edges_best[1:])
            width = np.diff(edges_best)
            # convert probabilities to density for step plotting: p / width
            dens = np.where(width > 0, p_best / width, 0.0)
            plt.step(centers, dens, where="mid", label=f"Best {best_scheme} k={best_k}")
            plt.xlabel("Eccentricity e")
            plt.ylabel("Density")
            plt.title(f"Best ecc bins — {sp} (scheme={best_scheme}, k={best_k}, EMD={best['emd_l1']:.4f})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out_png2 = OUT_DIR / f"ecc_hist_best_{sp}.png"
            plt.savefig(out_png2, dpi=180)
            plt.close()
            print(f"  ⬇️ saved best ecc hist → {out_png2}")

            print(f"Best for {sp}: scheme={best_scheme}, k={best_k}, EMD={best['emd_l1']:.4f}")
        else:
            print(f"[WARN] No (scheme,k) candidates evaluated for {sp}.")

    # write CSV summary
    if results_rows:
        ecc_df = pd.DataFrame(results_rows)
        ecc_csv = OUT_DIR / "ecc_bins_summary.csv"
        ecc_df.to_csv(ecc_csv, index=False)
        print(f"📝 wrote {ecc_csv}")

        # --- Print best-fit bins and all options (k=1..10) in config-ready JSON ("eccentricity_bins": [...]) ---
        def _edges_unit_linear(k: int):
            return [float(x) for x in np.linspace(0.0, 0.2, k + 1)]

        def _edges_unit_log(k: int, e_min: float = 1e-4):
            # avoid repeated zeros; start at small positive and include 0 explicitly
            return [0.0] + [float(x) for x in np.geomspace(max(e_min, 1e-8), 0.2, k)]

        for sp in ("N", "B"):
            sub = ecc_df[ecc_df["species"] == sp]
            if sub.empty:
                continue

            print(f"\n===== Eccentricity bin suggestions for {sp} =====")
            # Best per scheme
            for scheme in ("linear", "log"):
                sub_s = sub[sub["scheme"] == scheme]
                if sub_s.empty:
                    continue
                k_best = int(sub_s.loc[sub_s["emd_l1"].idxmin(), "k"])
                edges_unit = _edges_unit_linear(k_best) if scheme == "linear" else _edges_unit_log(k_best)
                print(f"Best {scheme} (k={k_best}):")
                print(f'  "eccentricity_bins": {json.dumps(edges_unit)}')

            # All options k=1..10
            print("All linear k options:")
            for k in range(1, 20):
                print(f'  k={k}: "eccentricity_bins": {json.dumps(_edges_unit_linear(k))}')
            print("All log k options:")
            for k in range(1, 20):
                print(f'  k={k}: "eccentricity_bins": {json.dumps(_edges_unit_log(k))}')

            # Save to a per-species text file as well
            out_txt = OUT_DIR / f"ecc_bins_options_{sp}.txt"
            with open(out_txt, "w") as fh:
                fh.write(f"Eccentricity bin suggestions for {sp}\n\n")
                for scheme in ("linear", "log"):
                    sub_s = sub[sub["scheme"] == scheme]
                    if sub_s.empty:
                        continue
                    k_best = int(sub_s.loc[sub_s["emd_l1"].idxmin(), "k"])
                    edges_unit = _edges_unit_linear(k_best) if scheme == "linear" else _edges_unit_log(k_best)
                    fh.write(f"Best {scheme} (k={k_best}):\n  eccentricity_bins = {json.dumps(edges_unit)}\n\n")
                fh.write("All linear k options:\n")
                for k in range(1, 11):
                    fh.write(f"  k={k}: {json.dumps(_edges_unit_linear(k))}\n")
                fh.write("All log k options:\n")
                for k in range(1, 11):
                    fh.write(f"  k={k}: {json.dumps(_edges_unit_log(k))}\n")
            print(f"📝 wrote {out_txt}")

except Exception as e:
    print(f"[ecc binning] WARNING: {type(e).__name__}: {e}")