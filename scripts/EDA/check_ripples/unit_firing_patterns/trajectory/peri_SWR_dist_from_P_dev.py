#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-19 16:57:33 (ywatanabe)"

"""
./tmp/figs/time_dependent_dist
./tmp/figs/hist/traj_dist
./tmp/figs/hist/traj_pos
"""

import matplotlib

matplotlib.use("Agg")
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
from itertools import product, combinations

# Functions
def collect_dist_from_P(phase_base):
    dists_all = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            traj_session = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            if phase_base != "None":
                gP = np.nanmedian(
                    traj_session.transpose(1, 0, 2)[
                        :, :, GS_BINS_DICT[phase_base][0] : GS_BINS_DICT[phase_base][1]
                    ].reshape(3, -1),
                    axis=-1,
                    keepdims=True,
                )[np.newaxis]
            else:
                gP = np.zeros_like(traj_session)
            dist_session = norm(traj_session - gP, axis=1)
            df = pd.DataFrame(dist_session)
            df["subject"] = subject
            df["session"] = session
            df["trial_number"] = np.arange(len(traj_session)) + 1
            dists_all.append(df)
    return pd.concat(dists_all)


# def add_peri_swr_dist_from_P(dists_from_P, base_phase, events_df):
#     start_bin = -20
#     end_bin = 21

#     dists_from_P_all = []
#     for i_event, (_, event) in enumerate(events_df.iterrows()):
#         i_bin = int(event.center_time / (50 / 1000))

#         _dist_rip = dists_from_P[
#             (dists_from_P.subject == event.subject)
#             * (dists_from_P.session == event.session)
#             * (dists_from_P.trial_number == event.trial_number)
#         ]

#         _dists_from_P = []
#         for ii in range(start_bin, end_bin):
#             try:
#                 _dists_from_P.append(_dist_rip[i_bin + ii].iloc[0])
#             except:
#                 _dists_from_P.append(np.nan)
#         dists_from_P_all.append(_dists_from_P)

#     columns = [f"dist_from_{base_phase[0]}_{ii}" for ii in range(start_bin, end_bin)]
#     dists_from_P_all = pd.DataFrame(
#         data=np.vstack(dists_from_P_all), columns=columns
#     )  # (928, 41)

#     for ii in range(start_bin, end_bin):
#         col = f"dist_from_{base_phase[0]}_{ii}"
#         events_df[col] = None
#         events_df[col] = dists_from_P_all[col]

#     return events_df


# def plot(phase_base, phase_tgt, rips_df_from_P, cons_df_from_P):
#     start_bin = mngs.io.load("./config/global.yaml")["SWR_BINS"]["pre"][0]
#     end_bin = mngs.io.load("./config/global.yaml")["SWR_BINS"]["post"][1]

#     fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)

#     for i_event, events_df in enumerate([cons_df_from_P, rips_df_from_P]):
#         event_str = ["SWR-", "SWR+"][i_event]

#         for i_match, match in enumerate([1, 2]):
#             match_str = ["IN", "OUT"][i_match]

#             ax = axes[2 * i_match + i_event]

#             # for set_size in [4, 6, 8]:
#             mm = []
#             # ss = []
#             ci = []
#             xx = []

#             for ii in range(start_bin, end_bin):
#                 xx.append(ii * 50)
#                 data = events_df[f"dist_from_{phase_base[0]}_{ii}"][
#                     (events_df.phase == phase_tgt)
#                     * (events_df.match == match)
#                     # * (events_df.set_size == set_size)
#                     # * (events_df.correct == True)
#                 ]
#                 data = data[~data.isna()]

#                 # _mm, _ss = mngs.gen.describe(data, method="median")
#                 _mm, _ss = mngs.gen.describe(data, method="mean")
#                 _nn = len(data)
#                 _ci = 1.96 *_ss/_nn
#                 mm.append(_mm)
#                 # ss.append(_ss)
#                 ci.append(_ci)

#             ax = mngs.plt.ax_fill_between(
#                 ax,
#                 np.arange(start_bin, end_bin) * 50,
#                 np.array(mm),
#                 np.array(ci),
#                 None,
#                 alpha=0.5,
#             )
#             ax.set_title(match_str + " " + event_str)

#             df = pd.DataFrame(
#                 {
#                     "Time from SWR": xx,
#                     "under": np.array(mm) - ci,
#                     "mean": np.array(mm),
#                     "upper": np.array(mm) + ci,
#                 }
#             )
#             mngs.io.save(
#                 df,
#                 f"./tmp/figs/line/peri-SWR_distance_from_P_new/raw/match_{match}/"
#                 f"{phase_tgt[0].lower()}SWR/{phase_tgt[0].lower()}{event_str}_from_g{phase_base[0]}.csv",
#             )

#     fig.supylabel(f"Peri-{phase_tgt[0].lower()}SWR distance from g{phase_base[0]}")
#     fig.supxlabel("Time from SWR center [ms]")
#     return fig


# def collect_dists_for_pre_mid_or_post_SWR(events_df, period, base_phase):
#     import ipdb; ipdb.set_trace()
#     dists = []
#     for tgt_bin in range(SWR_BINS[period][0], SWR_BINS[period][1] + 1):
#         dists.append(events_df[f"dist_from_{base_phase[0]}_{tgt_bin}"])
#     return np.hstack(dists)


# def to_pre_mid_post_xSWR_from_P_df(rips_df, cons_df):
#     out = {}
#     ER_PHASES = ["Encoding", "Retrieval"]
#     for match in [1, 2]:
#         for phase_tgt, phase_base in product(ER_PHASES, ER_PHASES+["None"]):

#             rips_df_mp = rips_df[
#                 (rips_df.match == match) * (rips_df.phase == phase_tgt)
#             ]
#             cons_df_mp = cons_df[
#                 (cons_df.match == match) * (cons_df.phase == phase_tgt)
#             ]

#             dists_from_P = collect_dist_from_P(phase_base)

#             rips_df_mpP = add_peri_swr_dist_from_P(dists_from_P, phase_base, rips_df_mp)
#             cons_df_mpP = add_peri_swr_dist_from_P(dists_from_P, phase_base, cons_df_mp)

#             for period in ["pre", "mid", "post"]:
#                 out[
#                     f"match_{match}_{period}_{phase_tgt[0].lower()}SWR_from_{phase_base[0]}"
#                 ] = collect_dists_for_pre_mid_or_post_SWR(
#                     rips_df_mpP, period, phase_base
#                 )
#     return mngs.gen.force_dataframe(out, filler=np.nan)


def calc_dist_from_P(events_df, base_phase, period):
    pre_bins = mngs.io.load("./config/global.yaml")["SWR_BINS"][period]

    dists_from_P = []
    for tgt_bin in range(pre_bins[0], pre_bins[1]):
        coords_bin = events_df[f"{tgt_bin}"]
        if base_phase != "None":
            base_bin = events_df[base_phase]
            dists_bin = (coords_bin - base_bin).apply(mngs.linalg.nannorm)
        else:
            dists_bin = coords_bin.apply(mngs.linalg.nannorm)
        dists_from_P.append(dists_bin)
    return np.hstack(dists_from_P)

def print_stats(df, match, phase):
    """
    phase = "Encoding"
    is_control = True
    """
    phase_str = phase[0].lower()

    print(f"match: {match}")

    for event_str, is_control in zip(["SWR-", "SWR+"], [True, False]):
        print(event_str)
        SWR_str = "+" if not is_control else "-"
        print(f"{phase_str}SWR{SWR_str}")                    
        # within SWR+ or SWR-
        for pp in ["pre", "mid", "post"]:
            dd = df[f'match_{match}_{pp}-{phase_str}SWR{SWR_str}_dist_from_N']
            dd = dd[~dd.isna()]
            print(f"n = {len(dd)}", pp, mngs.gen.describe(dd, method="median"))
        print()
        for p1, p2 in combinations(["pre", "mid", "post"], 2):
            d1 = df[f'match_{match}_{p1}-{phase_str}SWR{SWR_str}_dist_from_N']        
            d1 = d1[~d1.isna()]
            d2 = df[f'match_{match}_{p2}-{phase_str}SWR{SWR_str}_dist_from_N']                
            d2 = d2[~d2.isna()]        
            w, p, dof, effsize = mngs.stats.brunner_munzel_test(d1, d2)
            print(p1, p2, round(p, 3), round(effsize, 3))
        print()
        
    d_con = df[f'match_{match}_mid-{phase_str}SWR-_dist_from_N']
    d_con = d_con[~d_con.isna()]
    d_rip = df[f'match_{match}_mid-{phase_str}SWR+_dist_from_N']
    d_rip = d_rip[~d_rip.isna()]    
    w, p, dof, effsize = mngs.stats.brunner_munzel_test(d_con, d_rip)
    print(f"mid-{phase_str}SWR- vs. mid-{phase_str}SWR+", round(p, 3), round(effsize, 3))
    print()

if __name__ == "__main__":
    import mngs
    import numpy as np

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
    SWR_BINS = mngs.io.load("./config/global.yaml")["SWR_BINS"]
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )
    # rips_df[rips_df.phase == "Encoding"]
    # rips_df[rips_df.phase == "Retrieval"]
    # cons_df_w = utils.rips.add_coordinates(
    #     utils.rips.load_rips_df_with_traj(BIN_SIZE, is_within_control=True)
    # )

    # tgt_phase = "Encoding"
    # base_phase = "Retrieval"
    # period = "pre"
    dfs = {}
    for match in [1, 2]:
        for tgt_phase in ["Encoding", "Retrieval"]:        
            # for base_phase in ["Encoding", "Retrieval", "None"]:
            for base_phase in ["None"]:
                for event_str, events_df in zip(["SWR-", "SWR+"], [cons_df, rips_df]):                    
                    for period in ["pre", "mid", "post"]:                                    
                        events_df_mp = events_df[
                            (events_df.match == match) * (events_df.phase == tgt_phase)
                        ]
                        dfs[
                            f"match_{match}_{period}-{tgt_phase[0].lower()}{event_str}_dist_from_{base_phase[0]}"
                        ] = calc_dist_from_P(events_df_mp, base_phase, period)
    df = mngs.gen.force_dataframe(dfs)

    # rips_df

    # df = to_pre_mid_post_xSWR_from_P_df(rips_df, cons_df)
    mngs.io.save(df, "./tmp/figs/box/peri_SWR_dist_from_P_new/data.csv")

    ER_PHASES = ["Encoding", "Retrieval"]
    for phase_base, phase_tgt in product(ER_PHASES + ["None"], ER_PHASES):
        print(phase_base, phase_tgt)
        dists_from_P = collect_dist_from_P(phase_base)

        rips_df_from_P = add_peri_swr_dist_from_P(dists_from_P, phase_base, rips_df)
        cons_df_from_P = add_peri_swr_dist_from_P(dists_from_P, phase_base, cons_df)

        fig = plot(
            phase_base, phase_tgt, rips_df_from_P, cons_df_from_P
        )  # , cons_df_w_m)
        mngs.io.save(
            fig,
            f"./tmp/figs/line/peri-SWR_distance_from_P_new/"
            f"{phase_tgt[0].lower()}SWR/{phase_tgt[0].lower()}SWR_from_g{phase_base[0]}.png",
        )

    df = mngs.io.load("./tmp/figs/box/peri_SWR_dist_from_P_new/data.csv")

    # Results 3.5
    match = 2
    phase = "Retrieval"
    print_stats(df, match, phase)
