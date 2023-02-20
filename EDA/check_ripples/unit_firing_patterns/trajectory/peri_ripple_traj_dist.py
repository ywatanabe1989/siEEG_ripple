#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-20 12:37:20 (ywatanabe)"

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
def compair_dist_of_rips_and_cons(rips_df, cons_df):
    width_ms = 500
    width_bins = width_ms / BIN_SIZE.magnitude
    start = -int(width_bins / 2)
    end = int(width_bins / 2)

    dists_rips, dists_cons = [], []
    for i_bin in range(start, end):
        dist_rips = np.vstack(rips_df[f"{i_bin}"])
        dist_cons = np.vstack(cons_df[f"{i_bin}"])

        dist_rips = [mngs.linalg.nannorm(dist_rips[ii]) for ii in range(len(dist_rips))]
        dist_cons = [mngs.linalg.nannorm(dist_cons[ii]) for ii in range(len(dist_cons))]

        dists_rips.append(
            dist_rips
        )
        dists_cons.append(
            dist_cons
        )
    dists_rips = np.vstack(dists_rips)
    dists_cons = np.vstack(dists_cons)

    df = {}
    for phase in PHASES:
        dists_rips_phase = np.hstack(dists_rips[:,rips_df.phase == phase])
        dists_cons_phase = np.hstack(dists_cons[:,cons_df.phase == phase])
        nan_indi = np.isnan(dists_rips_phase) + np.isnan(dists_cons_phase)

        dists_rips_phase = dists_rips_phase[~nan_indi]
        dists_cons_phase = dists_cons_phase[~nan_indi]
        
        df[f"SWR_{phase}"] = dists_rips_phase
        df[f"Control_{phase}"] = dists_cons_phase
        
        starts, pval = brunnermunzel(
            dists_rips_phase,
            dists_cons_phase,
        )
        print(phase, pval)

    mngs.io.save(mngs.general.force_dataframe(df), "./tmp/figs/box/dist_comparison_rips_and_cons.csv")
    return df

def plot_dist(rips_df, is_control=False, set_size=None, match=None):

    events_str = "cons" if is_control else "rips"

    if set_size is not None:
        rips_df = rips_df[rips_df.set_size == set_size]
    set_size_str = f"_set_size_{set_size}" if set_size is not None else ""

    if match is not None:
        rips_df = rips_df[rips_df.match == match]
    match_str = f"_match_{match}" if match is not None else ""

    fig, ax = plt.subplots(figsize=(6.4 * 3, 4.8 * 3))
    xlim = (-500, 500)
    out_df = pd.DataFrame()

    samp_m = mngs.general.listed_dict(PHASES)
    samp_s = mngs.general.listed_dict(PHASES)

    for i_phase, phase in enumerate(PHASES):

        rips_df_phase = rips_df[rips_df.phase == phase]

        centers_ms = []        
        for i_bin in range(-11, 12):
            centers_ms.append(int((i_bin) * BIN_SIZE.magnitude))
            dists_i_bin = rips_df_phase[f"{i_bin}"].apply(mngs.linalg.nannorm)

            mm, iqr = mngs.gen.describe(np.array(dists_i_bin))

            samp_m[phase].append(mm)
            samp_s[phase].append(iqr)            


        ax.axhline(y=0, xmin=xlim[0], xmax=xlim[1], linestyle="--", color="gray")

        ax.errorbar(
            x=np.array(centers_ms) + i_phase * 3,
            y=samp_m[phase],
            yerr=samp_s[phase],
            label=phase,
        )
        ax.legend(loc="upper right")
        ax.set_xlim(xlim)

        ylim = (0, 3)
        title = "Dist (norm)"        

        ax.set_title(title)

    samp_m = pd.DataFrame(samp_m)
    samp_m.columns = ["mm" for col in samp_m.columns]
    samp_s = pd.DataFrame(samp_s)
    samp_s.columns = ["ss" for col in samp_s.columns]
    out_df = pd.concat([out_df, pd.concat([samp_m, samp_s], axis=1)], axis=1)

    fig.suptitle(f"Set size: {set_size}\nMatch: {match}")
    fig.supylabel("Dist from O [a.u.]")
    fig.supxlabel("Time from SWR [ms]")
    # plt.show()
    mngs.io.save(
        fig,
        f"./tmp/figs/line/traj_dist/all_{events_str}_set_size_{set_size_str}_match_{match}.png",
    )
    mngs.io.save(
        out_df,
        f"./tmp/figs/line/traj_dist/all_{events_str}_set_size_{set_size_str}_match_{match}.csv",
    )
    return fig

if __name__ == "__main__":
    import mngs
    import numpy as np

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

    df = compair_dist_of_rips_and_cons(rips_df, cons_df)
    for phase in PHASES:
        w, p, dof, effsize = mngs.stats.brunner_munzel_test(df[f"Control_{phase}"], df[f"SWR_{phase}"])
        print(p)

    

    # indi = []
    # for subject, roi in ROIs.items():
    #     indi.append((rips_df.subject == f"{int(subject):02d}") * (rips_df.session.astype(int) <= 2))
    # np.array(indi).sum(axis=0)

    # rips_df[["subject", "ROI"]].drop_duplicates()

    ###
    # rips_df = add_vec_4_8(rips_df)
    # cons_df = add_vec_4_8(cons_df)

    # plot_pre_post_positions_2(rips_df, cons_df)

    # rips_df = load_rips_df_with_traj(BIN_SIZE, is_control=False)
    # cons_df = load_rips_df_with_traj(BIN_SIZE, is_control=True)
    for is_control in [False, True]:
        events_df = cons_df if is_control else rips_df
        for set_size in [None, 4, 6, 8]:
            for match in [None, 1, 2]:
                """
                set_size = None
                match = None                
                """                
                fig = plot_dist(
                    events_df,
                    is_control=is_control,
                    set_size=set_size,
                    match=match,
                )  # fig 4
                plt.close()
