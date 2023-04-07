#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-15 14:45:35 (ywatanabe)"

"""
./tmp/figs/time_dependent_dist
./tmp/figs/hist/traj_dist
./tmp/figs/hist/traj_pos
"""

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from elephant.gpfa import GPFA
import mngs
import neo
import numpy as np
import pandas as pd
import ffmpeg
from matplotlib import animation
import os
from mpl_toolkits.mplot3d import Axes3D
from bisect import bisect_right
from numpy.linalg import norm
import seaborn as sns
import random
from scipy.stats import brunnermunzel, ttest_ind
from scipy.linalg import norm
import sys
import quantities as pq

sys.path.append(".")
import utils


# Functions
# def collect_dist():
#     dists_all = []
#     for subject, roi in ROIs.items():
#         subject = f"{subject:02d}"
#         for session in ["01", "02"]:
#             traj_session = mngs.io.load(
#                 f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
#             )
#             dist_session = norm(traj_session, axis=1)
#             df = pd.DataFrame(dist_session)
#             df["subject"] = subject
#             df["session"] = session
#             df["trial_number"] = np.arange(len(traj_session)) + 1
#             dists_all.append(df)
#     return pd.concat(dists_all)


# def collect_peri_swr_dist(dists, events_df):
#     dists_all = []
#     for i_event, (_, event) in enumerate(events_df.iterrows()):
#         i_bin = int(event.center_time / (50 / 1000))
#         _dist = dists[
#             (dists.subject == event.subject)
#             * (dists.session == event.session)
#             * (dists.trial_number == event.trial_number)
#         ]
#         _dists = []
#         for ii in range(-11, 12):
#             try:
#                 _dists.append(_dist[i_bin - ii].iloc[0])
#             except:
#                 _dists.append(np.nan)
#         dists_all.append(_dists)
#     return np.vstack(dists_all)


# def compair_dist_of_rips_and_cons(rips_df, cons_df):
#     width_ms = 500
#     width_bins = width_ms / BIN_SIZE.magnitude
#     start = -int(width_bins / 2)
#     end = int(width_bins / 2)

#     dists_rips, dists_cons = [], []
#     for i_bin in range(start, end):
#         dist_rips = np.vstack(rips_df[f"{i_bin}"])
#         dist_cons = np.vstack(cons_df[f"{i_bin}"])

#         dist_rips = [mngs.linalg.nannorm(dist_rips[ii]) for ii in range(len(dist_rips))]
#         dist_cons = [mngs.linalg.nannorm(dist_cons[ii]) for ii in range(len(dist_cons))]

#         dists_rips.append(dist_rips)
#         dists_cons.append(dist_cons)

#     dists_rips = np.vstack(dists_rips)
#     dists_cons = np.vstack(dists_cons)

#     df = {}
#     for phase in PHASES:
#         dists_rips_phase = np.hstack(dists_rips[:, rips_df.phase == phase])
#         dists_cons_phase = np.hstack(dists_cons[:, cons_df.phase == phase])

#         dists_rips_phase = dists_rips_phase[~np.isnan(dists_rips_phase)]
#         dists_cons_phase = dists_cons_phase[~np.isnan(dists_cons_phase)]

#         df[f"SWR_{phase}"] = dists_rips_phase
#         df[f"Control_{phase}"] = dists_cons_phase

#         stats, pval = brunnermunzel(
#             dists_rips_phase,
#             dists_cons_phase,
#         )
#         mark = mngs.stats.to_asterisks(pval)
#         print(phase, round(pval, 3), mark)
#     df = mngs.general.force_dataframe(df)
#     return df


# def print_stats(df):
#     for phase in PHASES:
#         print(phase)
#         d1 = df[f"Control_{phase}"]
#         d2 = df[f"SWR_{phase}"]
#         d1 = d1.replace({"": np.nan})
#         d2 = d2.replace({"": np.nan})
#         print(f"n_SWR-: {(~np.isnan(d1)).sum()}")
#         mm, ss = mngs.gen.describe(d1, "median")
#         print(f"median: {round(mm, 3)}; IQR: {round(ss, 3)}")
#         print()
#         print(f"n_SWR+: {(~np.isnan(d2)).sum()}")
#         mm, ss = mngs.gen.describe(d2, "median")
#         print(f"median: {round(mm, 3)}; IQR: {round(ss, 3)}")
#         print()


# def calc_dist_from_O_of_SWR(event_df):
#     for ii in range(-5, 6):
#         coords_ii = event_df[f"{ii}"]
#         dists_ii = [mngs.linalg.nannorm(coords_ii[jj]) for jj in range(len(coords_ii))]
#         event_df[f"dist_{ii}"] = dists_ii
#     event_df["dist"] = np.nanmean(
#         np.array([event_df[f"dist_{ii}"] for ii in range(-5, 6)]), axis=0
#     )
#     return event_df


# def sort_peri_SWR_dist_from_O(cons_df, rips_df):
#     out_df = {}
#     for event_str, event_df in zip(["SWR-", "SWR+"], [cons_df, rips_df]):
#         for phase in PHASES:
#             for set_size in [4, 6, 8]:
#                 data = event_df[
#                     (event_df.phase == phase) * (event_df.set_size == set_size)
#                 ].dist
#                 out_df[f"{event_str}_{phase}_{set_size}"] = data
#     return mngs.gen.force_dataframe(out_df)


# def plot(cons_df_m, rips_df_m):
#     # Set size dependancy
#     fig, axes = plt.subplots(
#         ncols=2, sharex=True, sharey=True, figsize=(6.4 * 2, 4.8 * 2)
#     )
#     for ax, event_df in zip(axes, [cons_df_m, rips_df_m]):
#         sns.boxplot(
#             data=event_df,
#             x="phase",
#             hue="set_size",
#             y="dist",
#             order=PHASES,
#             ax=ax,
#             showfliers=False,
#         )
#     axes[0].set_title("SWR-")
#     axes[1].set_title("SWR+")
#     axes[0].set_ylabel("Peri-SWR distance from O")
#     axes[1].set_ylabel("Peri-SWR distance from O")
#     return fig

# def corr_test(cons_df_m, rips_df_m):
#     # corr test
#     count = 0
#     for phase in PHASES:
#         print(phase)

#         # Control
#         indi = cons_df_m.phase == phase
#         pval_cons, corr_obs_cons, corrs_shuffled_cons = mngs.stats.corr_test(
#             np.log10(cons_df_m["dist"][indi]), cons_df_m["set_size"][indi]
#         )
#         count += 1
#         spath = f"./tmp/figs/corr/peri_SWR_dist_from_O/match_{match}/{count}_by_phase_and_set_size_SWR-.pkl"
#         spath = spath.replace(".pkl", f"corr_{round(corr_obs_cons,2)}_pval_{round(pval_cons,3)}.pkl")
#         mngs.io.save(
#             {"observed": corr_obs_cons, "surrogate": corrs_shuffled_cons},
#             spath,
#         )

#         # SWR
#         indi = rips_df_m.phase == phase
#         pval_rips, corr_obs_rips, corrs_shuffled_rips = mngs.stats.corr_test(
#             np.log10(rips_df_m["dist"][indi]), rips_df_m["set_size"][indi]
#         )
#         count += 1
#         spath = f"./tmp/figs/corr/peri_SWR_dist_from_O/match_{match}/{count}_by_phase_and_set_size_SWR+.pkl"
#         spath = spath.replace(".pkl", f"corr_{round(corr_obs_rips,2)}_pval_{round(pval_rips,3)}.pkl")
#         mngs.io.save(
#             {"observed": corr_obs_rips, "surrogate": corrs_shuffled_rips},
#             spath,
#         )
#         print()

# def sort_peri_SWR_dist_from_SWR_center(rips_df, cons_df):
#     # for boxplot in sigmaplot, peri-SWR distance from SWR center
#     dfs = []
#     for phase in PHASES:
#         dists = collect_dist()
#         dists_rips = collect_peri_swr_dist(
#             dists, rips_df[rips_df.phase == phase]
#         )
#         mm_rips, sd_rips = np.nanmean(dists_rips, axis=0), np.nanstd(
#             dists_rips, axis=0
#         )
#         nn_rips = (~np.isnan(dists_rips)).sum(axis=0)
#         ci_rips = 1.96 * sd_rips / nn_rips

#         dists_cons = collect_peri_swr_dist(
#             dists, cons_df[cons_df.phase == phase]
#         )
#         mm_cons, sd_cons = np.nanmean(dists_cons, axis=0), np.nanstd(
#             dists_cons, axis=0
#         )
#         nn_cons = (~np.isnan(dists_cons)).sum(axis=0)
#         ci_cons = 1.96 * sd_cons / nn_cons

#         df = pd.DataFrame(
#             {
#                 f"{phase}_under_SWR+": mm_rips - ci_rips,
#                 f"{phase}_mean_SWR+": mm_rips,
#                 f"{phase}_upper_SWR+": mm_rips + ci_rips,
#                 f"{phase}_under_SWR-": mm_cons - ci_cons,
#                 f"{phase}_mean_SWR-": mm_cons,
#                 f"{phase}_upper_SWR-": mm_cons + ci_cons,
#             }
#         )
#         dfs.append(df)

#     df = pd.concat(dfs, axis=1)
#     return df

if __name__ == "__main__":
    import mngs
    import numpy as np
    from tslearn.metrics import dtw

    # Fixes seeds
    mngs.gen.fix_seeds(42, np=np)

    # Parameters
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )

    # koko
    fig, axes = plt.subplots(ncols=12, sharey=True, figsize=(6.4*2, 4.8))    

    for i_match, match in enumerate([1, 2]):
        match_str = ["IN", "OUT"][i_match]
        for i_event, events_df in enumerate([cons_df, rips_df]):
            for i_set_size, set_size in enumerate([4,6,8]):                
                event_str = ["SWR-", "SWR+"][i_event]
                
                # ax = axes[i_set_size * 4 + i_event * 2 + i_match]
                i_global = i_match * 6 + i_event * 3 + i_set_size
                ax = axes[i_global]                
                
                dists = mngs.gen.listed_dict()
                dtw_vals = []
                for subject in events_df.subject.unique():
                    for session in events_df.session.unique():
                        events_df_session = events_df[
                            (events_df.subject == subject)
                            * (events_df.session == session)
                            * (events_df.match == match)
                            * (events_df.set_size == set_size)
                            * (events_df.correct == True)
                        ]
                        rips_E_s = events_df_session[events_df_session.phase == "Encoding"]
                        rips_R_s = events_df_session[events_df_session.phase == "Retrieval"]

                        for _, rip_E in rips_E_s.iterrows():
                            for _, rip_R in rips_R_s.iterrows():
                                traj_rip_E = np.array(
                                    [rip_E[f"{ii}"] for ii in range(-5, 6)]
                                )
                                traj_rip_R = np.array(
                                    [rip_R[f"{ii}"] for ii in range(-5, 6)]
                                )
                                isnan = np.isnan(traj_rip_E).any(axis=-1) + np.isnan(
                                    traj_rip_R
                                ).any(axis=-1)
                                dtw_val = dtw(traj_rip_E[~isnan], traj_rip_R[~isnan])
                                dtw_vals.append(dtw_val)
                ax.boxplot(dtw_vals, positions=[i_global+1])
                ax.set_title(f"{match_str} {event_str} {set_size}")            
                        # for tgt_bin in range(-10, 11):
                        #     coords_E = events_df_session[
                        #         events_df_session.phase == "Encoding"
                        #     ][f"{tgt_bin}"]
                        #     coords_R = events_df_session[
                        #         events_df_session.phase == "Retrieval"
                        #     ][f"{tgt_bin}"]
                        #     for cE in coords_E:
                        #         for cR in coords_R:
                        #             dists[f"{tgt_bin}"].append(mngs.linalg.nannorm(cE - cR))
                # df = mngs.gen.force_dataframe(dists)
                # mm = df.apply(mngs.gen.describe).loc[0]
                # ss = df.apply(mngs.gen.describe).loc[1]
                # ax = mngs.plt.ax_fill_between(ax, np.arange(-11, 10) * 50, mm, ss, "None")
                # ax.set_title(f"{match_str} {event_str}")
    plt.show()
