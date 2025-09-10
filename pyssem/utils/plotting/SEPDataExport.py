import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.simulation.scen_properties import ScenarioProperties


class SEPDataExport:
    """
    Export population and collision data from SEP simulations.

    Creates:
      - pop_time.csv            (total population over time)
      - pop_time_alt.csv        (population by altitude over time)
      - heatmaps/               (heatmap PNGs per species)
      - pairwise_collisions_time_alt.csv (if indicator variables available)
    """

    def __init__(self,
                 scenario_properties: ScenarioProperties,
                 simulation_name: str,
                 elliptical: bool,
                 MOCAT_MC_Path: str = None,
                 output_dir: str = None):
        self.scenario_properties = scenario_properties
        self.simulation_name = simulation_name
        self.MOCAT_MC_Path = MOCAT_MC_Path
        # Base directory for CSVs and figures
        self.base_path = output_dir
        os.makedirs(self.base_path, exist_ok=True)
        # Directory for heatmaps
        self.heatmap_dir = os.path.join(self.base_path, "heatmaps")
        os.makedirs(self.heatmap_dir, exist_ok=True)

        # Load simulation outputs
        sp = self.scenario_properties
        self.Hmid = sp.HMid                # [n_shells]
        self.start_year = pd.to_datetime(sp.start_date).year
        self.species = sp.species_names    # list[str]
        self.n_shells = sp.n_shells
        self.times = sp.scen_times         # list[float]
        # Optional: large fragments list for grouping
        self.large_fragments = self.scenario_properties.pmd_debris_names
        
        if elliptical:
            # self.pop_time_df, self.pop_time_alt_df, self.pop_time_df_grouped, self.pop_time_alt_df_grouped = self.elliptical_to_effective_altitude_bins()
            # self.generate_heatmaps(self.pop_time_alt_df_grouped)
            self.scenario_properties.output.y = sp.output.y_alt  # use the altitude-resolved data directly
            self.y = sp.output.y_alt  # shape [n_alt * n_species, n_time]
        # else: 
        self.y = sp.output.y  # shape [n_shells * n_species, n_time]
        self.ssem_pop_time = self.pop_time()
        self.ssem_pop_time_alt = self.pop_time_alt()
        self.generate_heatmaps()

        snapshot_years = [2025, 2050, 2075, 2100, 2125]

        self.export_snapshots(snapshot_years=snapshot_years)

        if scenario_properties.indicator_results is not None:
            self.plot_cumulative_collisions_by_prefix()
            self.plot_cumulative_indicator()
            self.plot_cumulative_pairwise_by_species()
            # self.export_pairwise_collisions_time_alt()
        
        self.compute_metrics()
        self.plot_altitude_heatmap_comparison()
    
        if MOCAT_MC_Path:
            if elliptical:
                self.grouped_population_mc_comparison(self.pop_time_df_grouped)
            else:
                self.grouped_population_mc_comparison(self.pop_time_df_grouped)

    def _group_label(self, sp: str) -> str:
        """
        Determine group label from species code.
        """
        if sp.startswith('Sns'):
            return 'Sns'
        if sp.startswith('Su'):
            return 'Su'
        if sp.startswith('S'):
            return 'S'
        if sp.startswith('B'):
            return 'B'
        if sp in self.large_fragments:
            return 'D'
        return 'N'

    def pop_time(self) -> pd.DataFrame:
        """
        Export total population per species over time to pop_time.csv.
        """
        rows = []
        for i, sp in enumerate(self.species):
            label = self._group_label(sp)
            start_idx = i * self.n_shells
            end_idx = (i + 1) * self.n_shells
            if end_idx > self.y.shape[0]:
                continue
            shell_data = self.y[start_idx:end_idx, :]
            # sum over shells for each time step
            for t_idx, offset in enumerate(self.times):
                year = int(self.start_year + offset)
                pop = shell_data[:, t_idx].sum()
                rows.append({
                    "Species": label,
                    "Year": year,
                    "Population": pop
                })

        df = pd.DataFrame(rows)
        df_grouped = (
            df
            .groupby(["Species", "Year"], as_index=False)
            .sum()
            .sort_values(["Species", "Year"])  
            .reset_index(drop=True)
        )
        path = os.path.join(self.base_path, "pop_time.csv")
        df_grouped.to_csv(path, index=False)
        print(f"Saved total population over time to {path}")
        self.pop_time_df_grouped = df_grouped
        return df_grouped

    def pop_time_alt(self) -> pd.DataFrame:
        """
        Export shell-resolved population per species over time and altitude to pop_time_alt.csv.
        """
        rows = []
        for i, sp in enumerate(self.species):
            label = self._group_label(sp)
            start_idx = i * self.n_shells
            end_idx = (i + 1) * self.n_shells
            if end_idx > self.y.shape[0]:
                continue
            shell_block = self.y[start_idx:end_idx, :]
            for s in range(self.n_shells):
                alt = self.Hmid[s]
                for t_idx, offset in enumerate(self.times):
                    year = int(self.start_year + offset)
                    pop = shell_block[s, t_idx]
                    rows.append({
                        "Species": label,
                        "Year": year,
                        "Altitude": alt,
                        "Population": pop
                    })

        df = pd.DataFrame(rows)
        df_grouped = (
            df
            .groupby(["Species", "Year", "Altitude"], as_index=False)
            .sum()
            .sort_values(["Species", "Year", "Altitude"] )
            .reset_index(drop=True)
        )
        path = os.path.join(self.base_path, "pop_time_alt.csv")
        df_grouped.to_csv(path, index=False)
        # print(f"Saved altitude-resolved population data to {path}")
        return df_grouped

    def generate_heatmaps(self, df_alt: pd.DataFrame = None):
        """
        Generate and save heatmaps for each species using the altitude-resolved data.
        """
        if df_alt is None:
            df_alt = self.pop_time_alt()

        for sp in df_alt["Species"].unique():
            df_sp = df_alt[df_alt["Species"] == sp]
            pivot = df_sp.pivot_table(
                index="Altitude", columns="Year", values="Population",
                aggfunc="sum", fill_value=0
            )
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': 'Population'})
            plt.gca().invert_yaxis()
            plt.title(f"Population Heatmap for Species: {sp}")
            plt.xlabel("Year")
            plt.ylabel("Altitude (km)")
            plt.tight_layout()

            fname = os.path.join(self.heatmap_dir, f"{sp}_heatmap.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            # print(f"Saved heatmap for {sp} to {fname}")

    def export_snapshots(self,
                         snapshot_years: list[int] = None
                         ) -> pd.DataFrame:
        """
        For each year in `snapshot_years`, save:
          1. A bar chart of total population per species.
          2. Line plots of population vs. altitude per species.
          3. A CSV of the filtered snapshot data.
        Returns the DataFrame of snapshot rows.
        """
        import matplotlib.pyplot as plt

        # default snapshots if none provided
        if snapshot_years is None:
            snapshot_years = [2025, 2050, 2075, 2100, 2125]

        # make sure we have the altitude-resolved table
        df_alt = self.pop_time_alt()

        # filter
        df_snap = df_alt[df_alt["Year"].isin(snapshot_years)]

        # 1. bar chart
        bar_data = (
            df_snap
            .groupby(["Year", "Species"])["Population"]
            .sum()
            .unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_data.plot(kind="bar", ax=ax)
        ax.set_ylabel("Total Population")
        ax.set_title("Snapshot: Total Population per Species")
        ax.set_xticklabels(bar_data.index, rotation=0)
        plt.tight_layout()
        bar_path = os.path.join(self.base_path, "pop_snapshot_bar.png")
        fig.savefig(bar_path, dpi=300)
        plt.close(fig)
        # print(f"✅ Saved bar chart snapshot to {bar_path}")

        # 2. fixed line plots by altitude
        line_dir = os.path.join(self.base_path, "snapshots_altitude_lines")
        os.makedirs(line_dir, exist_ok=True)
        for year in snapshot_years:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_y = df_snap[df_snap["Year"] == year]
            alts = sorted(df_y["Altitude"].unique())
            for sp in sorted(df_y["Species"].unique()):
                df_sp = (
                    df_y[df_y["Species"] == sp]
                    .groupby("Altitude", as_index=False)["Population"]
                    .sum()
                    .set_index("Altitude")
                    .reindex(alts, fill_value=0)
                    .reset_index()
                )
                ax.plot(df_sp["Altitude"], df_sp["Population"],
                        marker="o", linestyle="-", label=sp)
            ax.set_title(f"Population by Altitude – {year}")
            ax.set_xlabel("Altitude (km)")
            ax.set_ylabel("Population")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            path = os.path.join(line_dir, f"pop_by_altitude_{year}.png")
            fig.savefig(path, dpi=300)
            plt.close(fig)
            # print(f"✅ Saved line plot for {year} to {path}")

        # 3. save snapshot CSV
        csv_path = os.path.join(self.base_path, "pop_snapshots.csv")
        df_snap.to_csv(csv_path, index=False)
        # print(f"✅ Saved snapshot data to {csv_path}")

        return df_snap
    
    def plot_cumulative_indicator(self, indicator_name: str = None) -> np.ndarray:
        """
        Generate & save a cumulative‐indicator vs time line plot.
        If no name given, uses the last indicator in scenario_properties.indicator_results.
        Returns the cumulative array.
        """
        import os, numpy as np, matplotlib.pyplot as plt

        inds = self.scenario_properties.indicator_results['indicators']
        if indicator_name is None:
            indicator_name = list(inds.keys())[-1]
        data = inds[indicator_name]

        times = np.array(list(data.keys()))
        mat   = np.array([np.squeeze(v) for v in data.values()])  # [t, shells]
        total = mat.sum(axis=1)
        cum   = np.cumsum(total)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times, cum, marker='o')
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Indicator Value")
        ax.set_title(f"Cumulative Indicator: {indicator_name}")
        ax.grid(True)
        plt.tight_layout()

        out = os.path.join(self.base_path, f"cumulative_indicator_{indicator_name}.png")
        fig.savefig(out, dpi=300)
        plt.close(fig)
        # print(f"✅ Saved cumulative indicator plot to {out}")

        return cum

    def plot_cumulative_collisions_by_prefix(self) -> tuple[np.ndarray, dict]:
        inds = {
            n: d for n, d in self.scenario_properties.indicator_results.get('indicators', {}).items()
            if n.endswith('aggregate_collisions') and n != 'active_aggregate_collisions'
        }
        sum_by = {'N': None, 'S': None, 'B': None}
        times = None
        for name, data_dict in inds.items():
            prefix = name.split('_')[0]
            if prefix not in sum_by:
                continue
            arr = np.vstack([np.squeeze(v) for v in data_dict.values()])
            tsum = arr.sum(axis=1)
            if times is None:
                times = np.array(list(data_dict.keys()))
            if sum_by[prefix] is None:
                sum_by[prefix] = tsum.copy()
            else:
                sum_by[prefix] = sum_by[prefix] + tsum
        if times is None:
            raise ValueError("No aggregate_collisions indicators found.")
        total = None
        for arr in sum_by.values():
            if arr is None:
                continue
            total = arr.copy() if total is None else total + arr
        if total is None:
            raise ValueError("No collision data to plot.")
        total_cum = np.cumsum(total)
        cum_by = {p: np.cumsum(v) for p, v in sum_by.items() if v is not None}

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, total_cum, label='Total Collisions', linewidth=2)
        for p, c in cum_by.items():
            ax.plot(times, c, label=p)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Collisions")
        ax.set_title("Cumulative Aggregate Collisions by Class")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        out = os.path.join(self.base_path, "cumulative_collisions_by_prefix.png")
        fig.savefig(out, dpi=300)
        plt.close(fig)
        # print(f"Saved cumulative collisions by prefix to {out}")
        return total_cum, cum_by

    def plot_cumulative_pairwise_by_species(self) -> dict:
        """
        Sum all 'pair_collisions' by starting‐species prefix (N, S, Su, Sns, B)
        and plot their cumulative collisions over time.
        Returns the dict of cumulative arrays.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        inds = {
            n: d for n, d in self.scenario_properties.indicator_results.get('indicators', {}).items()
            if 'pair_collisions' in n
        }
        prefixes = ['N', 'S', 'Su', 'B']
        times = None
        cum_by = {}

        for pref in prefixes:
            agg = None
            for name, data_dict in inds.items():
                base = name.replace('_pair_collisions', '').split('__')[0]
                if not base.startswith(pref):
                    continue
                arr = np.vstack([np.squeeze(v) for v in data_dict.values()])
                tsum = arr.sum(axis=1)
                # only set times once, avoid ambiguous truth test on array
                if times is None:
                    times = np.array(list(data_dict.keys()))
                agg = tsum.copy() if agg is None else agg + tsum
            if agg is not None:
                cum_by[pref] = np.cumsum(agg)

        if not cum_by:
            raise ValueError("No pairwise collision indicators found.")

        fig, ax = plt.subplots(figsize=(12, 6))
        for p, c in cum_by.items():
            ax.plot(times, c, label=p)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Pairwise Collisions")
        ax.set_title("Cumulative Pairwise Collisions by Starting Species")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        out = os.path.join(self.base_path, "cumulative_pairwise_by_species.png")
        fig.savefig(out, dpi=300)
        plt.close(fig)
        # print(f"Saved cumulative pairwise by species to {out}")
        return cum_by

    def export_pairwise_collisions_time_alt(self) -> pd.DataFrame:
        """
        Build & save `pairwise_collisions_time_alt.csv`, then
        generate a heatmap per species-pair (Year vs Altitude).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import defaultdict

        # --- build the flat table ---
        Hmid = self.Hmid
        sy   = self.start_year
        lf   = self.large_fragments

        def grp(n):
            if n in lf:        return 'D'
            if n.startswith('Sns'): return 'Sns'
            if n.startswith('Su'):  return 'Su'
            if n.startswith('S'):   return 'S'
            if n == 'B':       return 'B'
            return 'N'

        inds = {
            name: data
            for name, data in self.scenario_properties.indicator_results
                                .get('indicators', {}).items()
            if name.endswith('pair_collisions')
        }
        if not inds:
            raise ValueError("No pairwise collision indicators to export.")

        rows = []
        for fullname, data_dict in inds.items():
            raw1, raw2 = fullname.replace('_pair_collisions','').split('__')
            s1, s2 = grp(raw1), grp(raw2)
            for offset, arr in data_dict.items():
                year = int(sy + offset)
                vals = np.squeeze(arr)
                # handle 0-D vs 1-D
                for idx, v in enumerate(vals if vals.ndim else [vals]):
                    rows.append({
                        "Species 1": s1,
                        "Species 2": s2,
                        "Year": year,
                        "Altitude": Hmid[idx],
                        "Collisions": v
                    })

        df = pd.DataFrame(rows)\
            .sort_values(["Species 1", "Species 2", "Year", "Altitude"])\
            .reset_index(drop=True)

        # --- write CSV ---
        csv_path = os.path.join(self.base_path, "pairwise_collisions_time_alt.csv")
        df.to_csv(csv_path, index=False)
        # print(f"✅ Saved pairwise collisions data to {csv_path}")

        # --- now generate heatmaps ---
        heat_dir = os.path.join(self.base_path, "pairwise_heatmaps")
        os.makedirs(heat_dir, exist_ok=True)

        # for each pair, pivot & plot
        pairs = df[["Species 1","Species 2"]].drop_duplicates().itertuples(index=False)
        for sp1, sp2 in pairs:
            sub = df[(df["Species 1"]==sp1)&(df["Species 2"]==sp2)]
            pivot = sub.pivot_table(
                index="Altitude",
                columns="Year",
                values="Collisions",
                aggfunc="sum",
                fill_value=0
            )
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(pivot, cmap="viridis", cbar_kws={'label':'Collisions'}, ax=ax)
            ax.invert_yaxis()
            ax.set_title(f"Collisions: {sp1}–{sp2}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Altitude (km)")
            plt.tight_layout()

            fname = os.path.join(heat_dir, f"{sp1}_{sp2}_heatmap.png")
            fig.savefig(fname, dpi=300)
            plt.close(fig)
            # print(f"✅ Saved heatmap for {sp1}–{sp2} to {fname}")
    
    # def elliptical_to_effective_altitude_bins(self):
    #     """
    #     Convert the raw SMA–species–eccentricity population array into
    #     effective populations per altitude shell over time, and store
    #     the result back into scenario_properties.output for reuse.

    #     On completion, sets:
    #     scenario_properties.output.y_alt       (shape [n_alt, n_species, n_time])
    #     scenario_properties.output.altitudes   (1-D array of HMid)

    #     Returns:
    #         df_pop_time:       total population per species over time
    #         df_pop_time_alt:   altitude-resolved population per species over time
    #         df_group_time:     grouped species population per time
    #         df_group_time_alt: grouped species & altitude population per time
    #     """
    #     import numpy as np
    #     import pandas as pd
    #     import os

    #     sp = self.scenario_properties

    #     # --- Dimensions / metadata ---
    #     n_sma_bins    = sp.n_shells
    #     n_alt_shells  = sp.n_shells
    #     n_species     = sp.species_length
    #     n_ecc_bins    = len(sp.eccentricity_bins) - 1
    #     times         = np.asarray(sp.output.t)             # time offsets (years)
    #     n_time        = int(len(times))
    #     years_abs     = np.array([int(self.start_year + off) for off in times], dtype=int)
    #     species_names = list(sp.species_names)
    #     altitudes     = np.asarray(sp.HMid, dtype=float)

    #     # --- Midpoints for cutoff (preferred) with safe fallbacks ---
    #     # Try direct on sp; if the user has nested, also try sp.scenario_properties
    #     sp_nested = getattr(sp, "scenario_properties", sp)

    #     ecc_centers = getattr(sp, "binE_ecc_mid_point", None)
    #     if ecc_centers is None:
    #         ecc_centers = getattr(sp_nested, "binE_ecc_mid_point", None)
    #     if ecc_centers is None:
    #         # fallback: midpoints from bin edges
    #         edges = np.asarray(sp.eccentricity_bins, float)
    #         ecc_centers = 0.5 * (edges[:-1] + edges[1:])
    #     ecc_centers = np.asarray(ecc_centers, float)
    #     if ecc_centers.shape[0] != n_ecc_bins:
    #         raise ValueError(f"ecc midpoint length {len(ecc_centers)} != n_ecc_bins {n_ecc_bins}")

    #     sma_centers_km = getattr(sp, "sma_HMid_km", None)
    #     if sma_centers_km is None:
    #         sma_centers_km = getattr(sp_nested, "sma_HMid_km", None)
    #     if sma_centers_km is None:
    #         # fallback: infer from apogee/perigee mids if available, else raise
    #         raise ValueError("sma_HMid_km not found on scenario_properties; please provide SMA midpoints in km.")
    #     sma_centers_km = np.asarray(sma_centers_km, float)
    #     if sma_centers_km.shape[0] != n_sma_bins:
    #         raise ValueError(f"sma midpoint length {len(sma_centers_km)} != n_sma_bins {n_sma_bins}")

    #     # Earth radius and cutoff
    #     R_earth_km = float(getattr(sp, "R_earth_km", 6378.136))  # fallback to WGS-84 equatorial
    #     H_DECAY_KM = 150.0
    #     Rcut = R_earth_km + H_DECAY_KM

    #     # --- Unpack & reshape population (SMA × Species × ECC × Time) ---
    #     y = np.asarray(sp.output.y)  # shape (n_sma*n_species*n_ecc, n_time)
    #     x_full = y.reshape(n_sma_bins, n_species, n_ecc_bins, n_time)  # (SMA, Spp, ECC, T)

    #     # --- Hard perigee cutoff AFTER the first year only (no fractional removal) ---
    #     # Build midpoint keep mask: keep if a_mid*(1 - e_mid) > Rcut    (2-D: [n_sma, n_ecc])
    #     A_mid, E_mid = np.meshgrid(sma_centers_km, ecc_centers, indexing="ij")  # (n_sma, n_ecc)
    #     mask_keep_mid = (A_mid * (1.0 - E_mid)) > Rcut  # bool, shape (n_sma, n_ecc)

    #     # Indices at/after 1 year (be robust to tiny FP noise)
    #     t_mask = times >= (1.0 - 1e-12)

    #     if np.any(t_mask):
    #         # Promote to 4-D keep mask so it broadcasts across species & time:
    #         # (n_sma, 1, n_ecc, 1)  → broadcasts to (n_sma, n_species, n_ecc, n_tmask)
    #         keep4 = mask_keep_mid[:, None, :, None]

    #         # Ensure float (or leave as-is if y is already float)
    #         x_full = x_full.astype(float, copy=False)

    #         # Multiply in-place: False→0 kills that whole SMA–ECC cell for all species at t>=1
    #         x_full[:, :, :, t_mask] *= keep4

    #     # --- Project (SMA × ECC) -> altitude shells for each time, all species at once ---
    #     # y_alt has shape (n_alt, n_species, n_time)
    #     y_alt = np.zeros((n_alt_shells, n_species, n_time), dtype=float)
    #     for t_idx in range(n_time):
    #         cube_t = x_full[:, :, :, t_idx]  # (n_sma, n_species, n_ecc)
    #         # Expect: returns (n_alt, n_species)
    #         alt_proj_all = sp.sma_ecc_mat_to_altitude_mat(cube_t)
    #         if alt_proj_all.shape != (n_alt_shells, n_species):
    #             raise ValueError(
    #                 f"sma_ecc_mat_to_altitude_mat returned {alt_proj_all.shape}, "
    #                 f"expected ({n_alt_shells}, {n_species})"
    #             )
    #         y_alt[:, :, t_idx] = alt_proj_all

    #     # Stash into output as promised in docstring
    #     sp.output.y_alt = y_alt
    #     sp.output.altitudes = altitudes

    #     # ─────────────────────────────────────────────────────────────
    #     # Build tidy DataFrames (vectorized) and write CSVs
    #     # ─────────────────────────────────────────────────────────────

    #     # (1) Altitude-resolved table: Species × Year × Altitude
    #     pop_TSA = np.transpose(y_alt, (2, 1, 0))  # (T, S, A)
    #     pop_flat = pop_TSA.reshape(-1)            # length T*S*A
    #     T, S, A = n_time, n_species, n_alt_shells
    #     years_rep   = np.repeat(years_abs, S * A)
    #     species_idx = np.tile(np.repeat(np.arange(S), A), T)
    #     alt_idx     = np.tile(np.arange(A), T * S)

    #     df_pop_time_alt = pd.DataFrame({
    #         "Species":    np.array(species_names, dtype=object)[species_idx],
    #         "Year":       years_rep,
    #         "Altitude":   altitudes[alt_idx],
    #         "Population": pop_flat.astype(float),
    #     })
    #     df_pop_time_alt = (
    #         df_pop_time_alt
    #         .groupby(["Species", "Year", "Altitude"], as_index=False).sum()
    #         .sort_values(["Species", "Year", "Altitude"])
    #         .reset_index(drop=True)
    #     )
    #     df_pop_time_alt.to_csv(os.path.join(self.base_path, "pop_time_alt.csv"), index=False)

    #     # (2) Total per species × year (sum over altitude)
    #     pop_ST = y_alt.sum(axis=0)  # (S, T)
    #     df_pop_time = pd.DataFrame({
    #         "Species":    np.repeat(species_names, T),
    #         "Year":       np.tile(years_abs, S),
    #         "Population": pop_ST.reshape(-1).astype(float),
    #     })
    #     df_pop_time = (
    #         df_pop_time
    #         .groupby(["Species", "Year"], as_index=False).sum()
    #         .sort_values(["Species", "Year"])
    #         .reset_index(drop=True)
    #     )
    #     df_pop_time.to_csv(os.path.join(self.base_path, "pop_time.csv"), index=False)

    #     # ─────────────────────────────────────────────────────────────
    #     # Group rollups (S, Su, Sns, N, D, B)
    #     # ─────────────────────────────────────────────────────────────

    #     def map_group(spn: str):
    #         spn = spn.strip() if isinstance(spn, str) else spn
    #         if spn == 'S': return 'S'
    #         if spn == 'Su': return 'Su'
    #         if spn == 'Sns': return 'Sns'
    #         if spn == 'B': return 'B'
    #         if isinstance(spn, str) and spn in self.large_fragments: return 'D'
    #         if isinstance(spn, str) and spn.startswith('N') and spn not in self.large_fragments: return 'N'
    #         return None

    #     # Grouped totals per year
    #     df_tmp = df_pop_time.copy()
    #     df_tmp["Group"] = df_tmp["Species"].map(map_group)
    #     df_tmp = df_tmp.dropna(subset=["Group"])
    #     df_group_time = (
    #         df_tmp.groupby(["Group", "Year"], as_index=False)["Population"].sum()
    #             .rename(columns={"Group": "Species"})
    #             .sort_values(["Species", "Year"])
    #             .reset_index(drop=True)
    #     )
    #     df_group_time.to_csv(os.path.join(self.base_path, "pop_time_grouped.csv"), index=False)

    #     # Grouped altitude-resolved per year
    #     df_alt_tmp = df_pop_time_alt.copy()
    #     df_alt_tmp["Group"] = df_alt_tmp["Species"].map(map_group)
    #     df_alt_tmp = df_alt_tmp.dropna(subset=["Group"])
    #     df_group_time_alt = (
    #         df_alt_tmp.groupby(["Group", "Year", "Altitude"], as_index=False)["Population"].sum()
    #                 .rename(columns={"Group": "Species"})
    #                 .sort_values(["Species", "Year", "Altitude"])
    #                 .reset_index(drop=True)
    #     )
    #     df_group_time_alt.to_csv(os.path.join(self.base_path, "pop_time_alt_grouped.csv"), index=False)

    #     return df_pop_time, df_pop_time_alt, df_group_time, df_group_time_alt
    
    def grouped_population_mc_comparison(self, df_any=None):
        """
        Plot grouped SSEM vs MOCAT-MC *total population over time* for ['S','Su','N','D','B'].

        Accepts either:
        - totals per (Species, Year)  (columns: Species, Year, Population), or
        - altitude-resolved per (Species, Year, Altitude) — it will aggregate by year.

        This function does NOT depend on the number of shells.
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # If nothing provided, use what we exported earlier
        if df_any is None:
            # Prefer grouped totals if present on disk
            p_tot = os.path.join(self.base_path, "pop_time_grouped.csv")
            p_alt = os.path.join(self.base_path, "pop_time_alt_grouped.csv")
            if os.path.exists(p_tot):
                df_any = pd.read_csv(p_tot)
            elif os.path.exists(p_alt):
                df_any = pd.read_csv(p_alt)
            else:
                # Fall back to in-memory grouped totals produced by pop_time()
                if hasattr(self, "pop_time_df_grouped"):
                    df_any = self.pop_time_df_grouped.copy()
                else:
                    raise ValueError("No grouped population data found to plot.")

        df_any = df_any.copy()

        # If altitude-resolved, aggregate to totals by (Species, Year)
        if "Altitude" in df_any.columns:
            df_plot = (df_any.groupby(["Species","Year"], as_index=False)["Population"].sum())
        else:
            df_plot = df_any

        # SSEM pivot
        pivot_ssem = df_plot.pivot_table(index="Year", columns="Species", values="Population", aggfunc="sum").fillna(0.0)
        years = np.array(sorted(pivot_ssem.index.values), dtype=int)

        # Load MC totals
        if not self.MOCAT_MC_Path or not os.path.exists(self.MOCAT_MC_Path):
            raise ValueError("MOCAT-MC totals path not set or not found (self.MOCAT_MC_Path).")
        mc_df = pd.read_csv(self.MOCAT_MC_Path)

        species_groups = ['S','Su','N','D','B']
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        axes = axes.flatten()

        for ax, group in zip(axes, species_groups):
            # SSEM
            if group in pivot_ssem.columns:
                ax.plot(years, pivot_ssem[group].reindex(years, fill_value=0.0), label='SSEM', linewidth=2)

            # MC
            sub_mc = mc_df[mc_df["Species"] == group]
            if not sub_mc.empty:
                ax.plot(sub_mc["Year"], sub_mc["Population"], '--', label='MC', linewidth=1.5)

            ax.set_title(group)
            ax.set_ylabel("Population")
            ax.grid(True)
            ax.legend()

        axes[-2].set_xlabel("Year")
        axes[-1].set_xlabel("Year")

        plt.suptitle("SSEM vs MOCAT-MC: Grouped Population Over Time", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out = os.path.join(self.base_path, "grouped_population_comparison.png")
        plt.savefig(out, dpi=300)
        plt.close(fig)
        # print(f"✅ Saved comparison figure to {out}")
        return out
    
    def plot_altitude_heatmap_comparison(self,
                                     mc_pop_time_alt_path: str = None,
                                     include_diff: bool = True,
                                     resample_kind: str = "linear",
                                     fname: str = None):
        """
        Compare MC vs SSEM as altitude×time heatmaps per group using the tidy tables:
        - SSEM: self.ssem_pop_time_alt   (Species, Year, Altitude, Population)
        - MC:   pop_time_alt.csv         (Species, Year, Altitude, Population)

        Assumes SSEM 'Species' already uses group labels (S, Su, N, D, B).
        Robust to different altitude grids between MC and SSEM (MC is resampled).
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # ---------- Inputs ----------
        if not hasattr(self, "ssem_pop_time_alt") or self.ssem_pop_time_alt is None:
            # fall back to regenerating if needed
            self.ssem_pop_time_alt = self.pop_time_alt()

        ssem_df = self.ssem_pop_time_alt.copy()

        # Resolve MC altitude CSV path
        if mc_pop_time_alt_path is None and isinstance(self.MOCAT_MC_Path, str):
            if self.MOCAT_MC_Path.endswith("pop_time.csv"):
                guess = self.MOCAT_MC_Path.replace("pop_time.csv", "pop_time_alt.csv")
                if os.path.exists(guess):
                    mc_pop_time_alt_path = guess
        if not mc_pop_time_alt_path or not os.path.exists(mc_pop_time_alt_path):
            raise ValueError("MC altitude CSV not found. Provide mc_pop_time_alt_path, "
                            "or set MOCAT_MC_Path to .../pop_time.csv in the same folder.")
        mc_df = pd.read_csv(mc_pop_time_alt_path)

        # ---------- Axes/grids ----------
        altitudes   = np.asarray(self.Hmid, float)  # SSEM altitude grid for plotting
        years_ssem  = np.array(sorted(ssem_df["Year"].unique()), dtype=int)
        years_mc    = np.array(sorted(mc_df["Year"].unique()), dtype=int)
        years_common = np.intersect1d(years_ssem, years_mc)
        if years_common.size == 0:
            raise ValueError("No overlapping years between SSEM and MC.")

        # Only keep common years
        ssem_df = ssem_df[ssem_df["Year"].isin(years_common)]
        mc_df   = mc_df[mc_df["Year"].isin(years_common)]

        # Species groups to plot (keep those present in SSEM)
        wanted_groups = ['S', 'Su', 'N', 'D', 'B']
        groups = [g for g in wanted_groups if g in set(ssem_df["Species"].unique())]

        # ---------- Helper: resample MC to SSEM alt grid ----------
        def _resample_alt(mat, src_alts, tgt_alts, kind="nearest"):
            src_alts = np.asarray(src_alts, float)
            tgt_alts = np.asarray(tgt_alts, float)
            mat = np.asarray(mat, float)
            if kind == "nearest":
                idx = np.searchsorted(src_alts, tgt_alts)
                idx = np.clip(idx, 0, len(src_alts) - 1)
                idxm = np.maximum(idx - 1, 0)
                use_prev = (idx > 0) & (np.abs(src_alts[idx] - tgt_alts) > np.abs(src_alts[idxm] - tgt_alts))
                idx[use_prev] = idxm[use_prev]
                return mat[idx, :]
            elif kind == "linear":
                out = np.empty((len(tgt_alts), mat.shape[1]), float)
                for j in range(mat.shape[1]):
                    out[:, j] = np.interp(tgt_alts, src_alts, mat[:, j])
                return out
            else:
                raise ValueError("kind must be 'nearest' or 'linear'")

        # ---------- Figure layout ----------
        if include_diff:
            ncols, wr = 5, [1, 1, 1, 0.03, 0.03]
        else:
            ncols, wr = 4, [1, 1, 0.03, 0.03]

        fig = plt.figure(figsize=(20, 5 * len(groups)))
        gs  = fig.add_gridspec(len(groups), ncols, width_ratios=wr, wspace=0.4, hspace=0.4)

        for i, grp in enumerate(groups):
            # SSEM pivot (alt × time) on SSEM altitude grid & common years
            sub_ssem = ssem_df[ssem_df["Species"] == grp]
            pivot_ssem = (
                sub_ssem.pivot(index="Altitude", columns="Year", values="Population")
                        .reindex(index=np.sort(sub_ssem["Altitude"].unique()), fill_value=0.0)
                        .reindex(columns=years_common, fill_value=0.0)
            )
            # Reindex to full SSEM altitude grid for consistent y-axis
            pivot_ssem = pivot_ssem.reindex(index=altitudes, fill_value=0.0)
            ssem_hm = pivot_ssem.to_numpy(float)  # (n_alt, n_years)

            # MC pivot (alt × time) → resample to SSEM altitude grid
            sub_mc = mc_df[mc_df["Species"] == grp]
            if sub_mc.empty:
                # still draw SSEM + colorbar; skip MC for this group
                mc_hm = np.zeros_like(ssem_hm)
            else:
                pivot_mc = (
                    sub_mc.pivot(index="Altitude", columns="Year", values="Population")
                        .fillna(0.0)
                        .reindex(columns=years_common, fill_value=0.0)
                        .sort_index(axis=0)
                )
                mc_alts = pivot_mc.index.to_numpy(float)
                mc_mat  = pivot_mc.to_numpy(float)
                mc_hm   = _resample_alt(mc_mat, mc_alts, altitudes, kind=resample_kind)

            # --- MC heatmap
            ax0 = fig.add_subplot(gs[i, 0])
            im0 = ax0.imshow(
                mc_hm, origin='lower', aspect='auto',
                extent=[years_common.min(), years_common.max(), altitudes.min(), altitudes.max()],
                cmap='viridis'
            )
            ax0.set_title(f"{grp} (MC)")
            if i == len(groups) - 1:
                ax0.set_xlabel("Year")
            ax0.set_ylabel("Altitude (km)")

            # --- SSEM heatmap
            ax1 = fig.add_subplot(gs[i, 1], sharey=ax0)
            im1 = ax1.imshow(
                ssem_hm, origin='lower', aspect='auto',
                extent=[years_common.min(), years_common.max(), altitudes.min(), altitudes.max()],
                cmap='viridis'
            )
            ax1.set_title(f"{grp} (pySSEM)")
            if i == len(groups) - 1:
                ax1.set_xlabel("Year")

            # --- Difference heatmap (optional)
            col_offset = 2
            if include_diff:
                diff_hm = mc_hm - ssem_hm
                ax2 = fig.add_subplot(gs[i, 2], sharey=ax0)
                im2 = ax2.imshow(
                    diff_hm, origin='lower', aspect='auto',
                    extent=[years_common.min(), years_common.max(), altitudes.min(), altitudes.max()],
                    cmap='seismic'
                )
                ax2.set_title(f"{grp} (MC − pySSEM)")
                if i == len(groups) - 1:
                    ax2.set_xlabel("Year")
                col_offset = 3

            # --- Colorbars
            cax_pop = fig.add_subplot(gs[i, col_offset])
            plt.colorbar(im0, cax=cax_pop).set_label("Population")
            cax_pop.yaxis.set_ticks_position('left')

            if include_diff:
                cax_diff = fig.add_subplot(gs[i, col_offset + 1])
                plt.colorbar(im2, cax=cax_diff).set_label("Δ Population")
                cax_diff.yaxis.set_ticks_position('left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if fname is None:
            fname = f"mc_vs_ssem_alt_heatmaps_{self.simulation_name}.png"
        out_path = os.path.join(self.base_path, fname)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        # print(f"✅ Saved altitude×time heatmap comparison to {out_path}")
        return out_path

    def compute_metrics(self,
                    mc_pop_time_path: str = None,
                    mc_pop_time_alt_path: str = None,
                    save_parquet: bool = False):
        """
        Compute SSEM-vs-MC metrics for grouped totals and (optionally) altitude concentration,
        build composite scores (WAPE-all-years), apply wider-band grading, and save results.

        Outputs (in self.base_path):
        - metrics_<simulation_name>.csv              (per-group metrics + extras + grades)
        - metrics_scenario_<simulation_name>.csv     (scenario-level weighted score)
        - (optional) .parquet versions if save_parquet=True

        Notes:
        * Uses self.pop_time_df_grouped (created by pop_time()) for SSEM grouped totals.
        * Expects MC totals at mc_pop_time_path (defaults to self.MOCAT_MC_Path).
        * If mc_pop_time_alt_path is provided (or derived from mc_pop_time_path), HHI is computed.
        * “Scenario” is renamed to “simulation_name” in all outputs.
        """
        import os
        import numpy as np
        import pandas as pd

        EPS = 1e-12

        # -------- helpers --------
        def _mc_series(mc_df_g, years_ref):
            s = mc_df_g.groupby("Year")["Population"].sum()
            s = s.reindex(years_ref, fill_value=0.0)
            return s.to_numpy(float)

        def _align_series(mc_df_g, years_ref, ssem_vals):
            years_ref = np.asarray(years_ref, int)
            y_true = _mc_series(mc_df_g, years_ref)
            y_pred = np.asarray(ssem_vals, float)
            n = min(len(y_true), len(y_pred))
            return years_ref[:n], y_true[:n], y_pred[:n]

        def _dtw_distance(y1, y2):
            a = np.asarray(y1, float); b = np.asarray(y2, float)
            n, m = a.size, b.size
            D = np.full((n+1, m+1), np.inf); D[0,0] = 0.0
            for i in range(1, n+1):
                ai = a[i-1]
                for j in range(1, m+1):
                    cost = abs(ai - b[j-1])
                    D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
            return float(D[n,m] / (n + m))  # average cost along optimal path

        def _stats_core(years, y_true, y_pred):
            diff = y_pred - y_true
            abs_diff = np.abs(diff)
            sum_true = y_true.sum()
            sum_pred = y_pred.sum()
            both_all_zero = (sum_true <= EPS) and (sum_pred <= EPS)

            # WAPE (global, all years)
            if sum_true <= EPS:
                WAPE = 0.0 if both_all_zero else np.inf
            else:
                WAPE = abs_diff.sum() / sum_true

            # sMAPE global (magnitude-weighted)
            denom = (np.abs(y_true) + np.abs(y_pred)).sum()
            sMAPE_global = 0.0 if both_all_zero else (2.0 * abs_diff.sum()) / max(denom, EPS)

            # RMSE / NRMSE
            RMSE = float(np.sqrt(np.mean(diff**2)))
            mean_true = y_true.mean()
            NRMSE = RMSE / (mean_true + EPS)

            # Bias
            Bias = float(np.mean(diff))

            # Final-year rel error
            fy_true, fy_pred = y_true[-1], y_pred[-1]
            if fy_true == 0 and fy_pred == 0:
                Final_rel_err = 0.0
            else:
                Final_rel_err = (fy_pred - fy_true) / (fy_true + EPS)

            # Ratios & extremes
            Total_ratio = (sum_pred / max(sum_true, EPS)) if not both_all_zero else 1.0
            Max_abs_err = float(abs_diff.max())
            Year_of_max_err = int(years[int(np.argmax(abs_diff))])

            # Correlation
            if np.std(y_true) < EPS or np.std(y_pred) < EPS:
                r, R2 = np.nan, np.nan
            else:
                r = float(np.corrcoef(y_true, y_pred)[0,1])
                R2 = r*r

            # DTW
            DTW = _dtw_distance(y_true, y_pred)

            return {
                "WAPE": WAPE, "sMAPE_global": sMAPE_global,
                "RMSE": RMSE, "NRMSE": NRMSE, "Bias": Bias,
                "Final_rel_err": Final_rel_err, "Total_ratio": Total_ratio,
                "Max_abs_err": Max_abs_err, "Year_of_max_err": Year_of_max_err,
                "r": r, "R2": R2, "DTW": DTW
            }

        def _window_wape(y_true, y_pred, k=10):
            n = len(y_true); k = min(k, n//2 if n>=2 else n)
            if k == 0: return np.nan, np.nan
            w_early = np.sum(np.abs(y_pred[:k] - y_true[:k])) / max(np.sum(y_true[:k]), EPS)
            w_late  = np.sum(np.abs(y_pred[-k:] - y_true[-k:])) / max(np.sum(y_true[-k:]), EPS)
            return float(w_early), float(w_late)

        def linear_calibration(y_true, y_pred):
            y = np.asarray(y_true, float); x = np.asarray(y_pred, float)
            X = np.c_[x, np.ones_like(x)]
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = float(sol[0]), float(sol[1])
            y_cal = alpha * x + beta
            sRMSE = float(np.sqrt(np.mean((y_cal - y)**2)))
            return alpha, beta, sRMSE

        def year_of_peak(years, y):
            i = int(np.argmax(y))
            return int(years[i])

        def time_to_half_from_peak(years, y):
            i = int(np.argmax(y)); y_peak = y[i]
            if y_peak <= 0: return np.nan
            half = 0.5 * y_peak
            idx = np.where(y[i:] <= half)[0]
            if idx.size == 0: return np.nan
            return float(years[i + idx[0]] - years[i])

        def nse(y_true, y_pred):
            y = np.asarray(y_true, float); f = np.asarray(y_pred, float)
            denom = np.sum((y - y.mean())**2)
            if denom <= EPS: return np.nan
            return float(1.0 - np.sum((f - y)**2) / denom)

        def theils_u2(y_true, y_pred):
            y = np.asarray(y_true, float); f = np.asarray(y_pred, float)
            num = np.sqrt(np.mean((f - y)**2))
            den = np.sqrt(np.mean((y[1:] - y[:-1])**2)) if len(y) > 1 else np.nan
            if not np.isfinite(den) or den <= EPS: return np.nan
            return float(num / den)

        def hhi_concentration(v):
            v = np.asarray(v, float); s = v.sum()
            if s <= EPS: return 0.0
            p = v / s
            return float(np.sum(p**2))

        def median_hhi_from_alt_df(df_alt, group, years_subset=None):
            try:
                sub = df_alt[df_alt["Species"] == group]
                if sub.empty: return np.nan
                piv = sub.pivot(index="Altitude", columns="Year", values="Population").fillna(0.0)
                cols = sorted(piv.columns)
                if years_subset is not None:
                    cols = [c for c in cols if c in set(years_subset)]
                if not cols: return np.nan
                vals = []
                for yr in cols:
                    vals.append(hhi_concentration(piv[yr].to_numpy()))
                return float(np.median(vals)) if vals else np.nan
            except Exception:
                return np.nan

        # -------- data prep --------
        # SSEM grouped totals (already grouped by your pop_time()).
        if not hasattr(self, "pop_time_df_grouped"):
            self.pop_time()  # creates self.pop_time_df_grouped
        ssem_group_df = self.pop_time_df_grouped.copy()

        # MC totals
        if mc_pop_time_path is None:
            mc_pop_time_path = self.MOCAT_MC_Path
        if not mc_pop_time_path or not os.path.exists(mc_pop_time_path):
            raise ValueError("MC totals path not provided or not found. "
                            "Provide mc_pop_time_path or set self.MOCAT_MC_Path to a valid CSV.")

        mc_df = pd.read_csv(mc_pop_time_path)

        # MC altitude path (optional) for HHI
        if mc_pop_time_alt_path is None and isinstance(mc_pop_time_path, str):
            if mc_pop_time_path.endswith("pop_time.csv"):
                guess = mc_pop_time_path.replace("pop_time.csv", "pop_time_alt.csv")
                mc_pop_time_alt_path = guess if os.path.exists(guess) else None

        mc_alt_df = None
        if mc_pop_time_alt_path and os.path.exists(mc_pop_time_alt_path):
            try:
                mc_alt_df = pd.read_csv(mc_pop_time_alt_path)
            except Exception:
                mc_alt_df = None

        # SSEM years and group series
        years_ref = np.array(sorted(ssem_group_df["Year"].unique()), dtype=int)
        ssem_pivot = ssem_group_df.pivot(index="Year", columns="Species", values="Population").fillna(0.0)
        ssem_pivot = ssem_pivot.reindex(index=years_ref, fill_value=0.0)

        # canonical groups
        species_groups = ["S", "Su", "N", "D", "B"]
        groups_present = [g for g in species_groups if g in ssem_pivot.columns and g in mc_df["Species"].unique()]

        # per-group DTW normalizer from MC (median |Δy|)
        def _dtw_scale_for_group(g):
            mc_g = mc_df[mc_df["Species"] == g]
            y_true = _mc_series(mc_g, years_ref)
            dy = np.diff(y_true)
            scale = np.median(np.abs(dy)) if dy.size else 0.0
            if not np.isfinite(scale) or scale < EPS:
                scale = max(np.median(np.abs(y_true)), 1.0)
            return float(scale)
        dtw_scales = {g: _dtw_scale_for_group(g) for g in groups_present}

        # -------- compute per-group metrics --------
        metrics_rows = []
        extras_rows  = []

        for g in groups_present:
            mc_g = mc_df[mc_df["Species"] == g]
            ssem_vals = ssem_pivot[g].to_numpy()
            years, y_true, y_pred = _align_series(mc_g, years_ref, ssem_vals)

            core = _stats_core(years, y_true, y_pred)
            w_early, w_late = _window_wape(y_true, y_pred, k=10)

            # extras
            alpha, beta, srmse = linear_calibration(y_true, y_pred)
            peak_mc = year_of_peak(years, y_true)
            peak_ss = year_of_peak(years, y_pred)
            t_half_mc = time_to_half_from_peak(years, y_true)
            t_half_ss = time_to_half_from_peak(years, y_pred)
            nse_val = nse(y_true, y_pred)
            u2_val  = theils_u2(y_true, y_pred)

            # HHI (altitude concentration), SSEM from our export, MC if available
            try:
                ssem_alt_df = pd.read_csv(os.path.join(self.base_path, "pop_time_alt.csv"))
            except Exception:
                ssem_alt_df = None
            hhi_mc_med   = median_hhi_from_alt_df(mc_alt_df,   g, years_subset=years) if mc_alt_df is not None else np.nan
            hhi_ssem_med = median_hhi_from_alt_df(ssem_alt_df, g, years_subset=years) if ssem_alt_df is not None else np.nan

            # core + some convenience fields
            metrics_rows.append({
                "simulation_name": self.simulation_name,
                "Group": g,
                "WAPE": core["WAPE"],
                "sMAPE_global": core["sMAPE_global"],
                "RMSE": core["RMSE"],
                "NRMSE": core["NRMSE"],
                "Bias": core["Bias"],
                "Final_rel_err": core["Final_rel_err"],
                "Total_ratio": core["Total_ratio"],
                "Max_abs_err": core["Max_abs_err"],
                "Year_of_max_err": core["Year_of_max_err"],
                "r": core["r"],
                "R2": core["R2"],
                "DTW": core["DTW"],
                "WAPE_early10": w_early,
                "WAPE_late10":  w_late
            })

            extras_rows.append({
                "simulation_name": self.simulation_name,
                "Group": g,
                "alpha_scale": alpha, "beta_offset": beta, "sRMSE": srmse,
                "PeakYear_MC": peak_mc, "PeakYear_SSEM": peak_ss,
                "DeltaPeakYear": int(peak_ss - peak_mc),
                "T_half_MC": t_half_mc, "T_half_SSEM": t_half_ss,
                "NSE": nse_val, "Theils_U2": u2_val,
                "HHI_MC_med": hhi_mc_med, "HHI_SSEM_med": hhi_ssem_med,
                "HHI_delta": (hhi_ssem_med - hhi_mc_med) if (np.isfinite(hhi_mc_med) and np.isfinite(hhi_ssem_med)) else np.nan
            })

        metrics_df = pd.DataFrame(metrics_rows)
        extras_df  = pd.DataFrame(extras_rows)

        if metrics_df.empty:
            raise ValueError("No overlapping groups between SSEM and MC to score.")

        # -------- composite score (WAPE over all years) + grading --------
        years_ref = np.array(sorted(ssem_group_df["Year"].unique()), dtype=int)  # re-ensure
        def _dtw_norm(row):
            return float(row.DTW) / max(dtw_scales.get(row.Group, 1.0), EPS)

        df = metrics_df.copy()
        df["DTW_norm"] = df.apply(_dtw_norm, axis=1)
        df["WAPE_all"] = df["WAPE"].astype(float)
        df["r_clamped"] = df["r"].fillna(0.0).clip(-1, 1)

        df["Score_group"] = (
            0.40 * df["WAPE_all"] +
            0.30 * np.abs(df["Final_rel_err"]) +
            0.20 * (1.0 - df["r_clamped"]) +
            0.10 * df["DTW_norm"]
        )

        # scenario-level mass-share weights from MC
        mc_totals = mc_df.groupby("Species")["Population"].sum().reindex(groups_present).fillna(0.0)
        if mc_totals.sum() > 0:
            w = (mc_totals / mc_totals.sum()).to_dict()
        else:
            w = {g: 1.0 / len(groups_present) for g in groups_present}

        scenario_score = float(np.sum([w.get(g, 0.0) * s for g, s in zip(df["Group"], df["Score_group"])]))
        df_scenario = pd.DataFrame([{"simulation_name": self.simulation_name,
                                    "Score_scenario": scenario_score}])

        # grading bands (wider)
        def _bucket(x, cuts):
            for upper, label in cuts:
                if x < upper:
                    return label
            return cuts[-1][1]

        score_cuts_wide = [(0.60,"Good"), (1.40,"Moderate"), (2.60,"Poor"), (float("inf"),"Mischaracterized")]
        wape_cuts_wide  = [(0.15,"Good"), (0.35,"Moderate"), (0.65,"Poor"), (float("inf"),"Mischaracterized")]
        final_cuts_wide = wape_cuts_wide
        dtw_cuts_wide   = [(3,"Good"), (7,"Moderate"), (12,"Poor"), (float("inf"),"Mischaracterized")]

        def _grade_r(r):
            if r >= 0.90: return "Good"
            if r >= 0.80: return "Moderate"
            if r >= 0.65: return "Poor"
            return "Mischaracterized"

        df["Grade"]       = df["Score_group"].apply(lambda v: _bucket(float(v), score_cuts_wide))
        df["Grade_WAPE"]  = df["WAPE_all"].apply(lambda v: _bucket(float(v), wape_cuts_wide))
        df["Grade_Final"] = df["Final_rel_err"].abs().apply(lambda v: _bucket(float(v), final_cuts_wide))
        df["Grade_r"]     = df["r"].fillna(0.0).apply(_grade_r)
        df["Grade_DTWn"]  = df["DTW_norm"].apply(lambda v: _bucket(float(v), dtw_cuts_wide))

        def _why(row):
            issues = []
            if row["Grade_WAPE"]  in ("Poor","Mischaracterized"): issues.append("magnitude (all years)")
            if row["Grade_Final"] in ("Poor","Mischaracterized"): issues.append("endpoint")
            if row["Grade_r"]     in ("Poor","Mischaracterized"): issues.append("shape")
            if row["Grade_DTWn"]  in ("Poor","Mischaracterized"): issues.append("timing")
            return ", ".join(issues) if issues else "well-matched"

        df["Why"] = df.apply(_why, axis=1)

        # merge extras for final per-group table
        df_full = df.merge(extras_df, on=["simulation_name","Group"], how="left")

        # -------- save outputs --------
        out_groups_csv   = os.path.join(self.base_path, f"metrics_{self.simulation_name}.csv")
        out_scenario_csv = os.path.join(self.base_path, f"metrics_scenario_{self.simulation_name}.csv")

        df_full.sort_values(["Group"]).to_csv(out_groups_csv, index=False)
        df_scenario.to_csv(out_scenario_csv, index=False)

        if save_parquet:
            df_full.to_parquet(out_groups_csv.replace(".csv", ".parquet"), index=False)
            df_scenario.to_parquet(out_scenario_csv.replace(".csv", ".parquet"), index=False)

        # print(f"✅ Saved per-group metrics to {out_groups_csv}")
        # print(f"✅ Saved scenario-level score to {out_scenario_csv}")

        # return DataFrames for immediate use if needed
        return df_full, df_scenario