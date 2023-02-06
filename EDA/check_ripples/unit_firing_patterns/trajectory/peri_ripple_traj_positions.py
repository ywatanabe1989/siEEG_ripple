#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-04 18:27:17 (ywatanabe)"


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

sys.path.append(".")
import utils

# Functions
def plot_positions(rips_df, is_control=False, set_size=None, match=None):
    events_str = "cons" if is_control else "rips"

    if set_size is not None:
        rips_df = rips_df[rips_df.set_size == set_size]
    set_size_str = f"_set_size_{set_size}" if set_size is not None else ""

    if match is not None:
        rips_df = rips_df[rips_df.match == match]
    match_str = f"_match_{match}" if match is not None else ""

    fig, axes = plt.subplots(
        nrows=len(PHASES), sharex=True, sharey=True, figsize=(6.4 * 2, 4.8 * 2)
    )
    xlim = (-1000, 1000)
    # ylim = (-0.3, 0.3)
    # ylim = (-1.5, 1.5)
    ylim = (-0.5, 4.5)

    out_df = pd.DataFrame()
    for i_ax, (ax, g_phase) in enumerate(zip(axes, PHASES)):

        samp_m = mngs.general.listed_dict(PHASES)
        samp_s = mngs.general.listed_dict(PHASES)

        for i_phase, rip_phase in enumerate(PHASES):
            rips_df_phase = rips_df[rips_df.phase == rip_phase]

            centers_ms = []

            for delta_bin in range(-39, 39):
                col1 = f"{delta_bin}"
                col1_str = f"{int((delta_bin)*BIN_SIZE.magnitude)}"

                centers_ms.append(int(delta_bin * BIN_SIZE.magnitude))

                # gets trajectory
                vv = np.vstack(rips_df_phase[col1])
                gg = np.vstack(rips_df_phase[g_phase])
                dist_gg = [mngs.linalg.nannorm(vv[ii] - gg[ii]) for ii in range(len(vv))]

                mm, ss = mngs.gen.describe(dist_gg, method="mean", factor=1)
                samp_m[rip_phase].append(mm)
                samp_s[rip_phase].append(ss)

            ax.errorbar(
                x=np.array(centers_ms) + i_phase * 3,
                y=samp_m[rip_phase],
                yerr=samp_s[rip_phase],
                label=rip_phase,
            )

        samp_m = pd.DataFrame(samp_m)
        samp_m.columns = [f"{col}_distance_from_{g_phase}" for col in samp_m.columns]
        samp_s = pd.DataFrame(samp_s)
        samp_s.columns = [f"{col}_distance_from_{g_phase}" for col in samp_s.columns]

        out_df = pd.concat([out_df, pd.concat([samp_m, samp_s], axis=1)], axis=1)

        ax.legend(loc="upper right")
        # ax.axhline(y=0, xmin=xlim[0], xmax=xlim[1], linestyle="--", color="gray")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Distance from {g_phase}")

    fig.suptitle(f"Set size: {set_size}\nMatch: {match}")
    mngs.io.save(
        fig,
        f"./tmp/figs/hist/traj_pos/all_{events_str}{set_size_str}{match_str}.png",
    )
    mngs.io.save(
        out_df,
        f"./tmp/figs/hist/traj_pos/all_{events_str}{set_size_str}{match_str}.csv",
    )
    return fig




if __name__ == "__main__":
    import mngs
    import numpy as np

    matplotlib.use("Agg")

    # Parameters
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    # Loads data
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )

    # Plots
    for is_control in [False, True]:
        events_df = cons_df if is_control else rips_df
        for set_size in [None, 4, 6, 8]:
            for match in [None, 1, 2]:
                """
                set_size = None
                match = None
                """
                fig = plot_positions(
                    events_df,
                    is_control=is_control,
                    set_size=set_size,
                    match=match,
                )  # fig 5
                plt.close()
