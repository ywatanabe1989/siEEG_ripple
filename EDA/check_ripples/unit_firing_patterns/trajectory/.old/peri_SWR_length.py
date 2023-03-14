#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-27 10:19:06 (ywatanabe)"

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
def collect_speed():
    speeds_all = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            traj_session = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            speed_session = norm(traj_session[..., 1:] - traj_session[..., :-1], axis=1)
            df = pd.DataFrame(speed_session)
            df["subject"] = subject
            df["session"] = session
            df["trial_number"] = np.arange(len(traj_session)) + 1
            speeds_all.append(df)
    return pd.concat(speeds_all)


def collect_peri_swr_speed(speeds, events_df):
    speeds_all = []
    for i_event, (_, event) in enumerate(events_df.iterrows()):
        i_bin = int(event.center_time / (50 / 1000))
        _speed = speeds[
            (speeds.subject == event.subject)
            * (speeds.session == event.session)
            * (speeds.trial_number == event.trial_number)
        ]
        _speeds = []
        for ii in range(-12, 12):
            try:
                _speeds.append(_speed[i_bin - ii].iloc[0])  # fixme
            except:
                _speeds.append(np.nan)
        speeds_all.append(_speeds)
    return np.vstack(speeds_all)


def compair_length_in_memory_load_direction_of_rips_and_cons(rips_df, cons_df):
    width_ms = 500
    width_bins = width_ms / BIN_SIZE.magnitude
    start = -int(width_bins / 2)
    end = int(width_bins / 2)
    rips_df["length"] = np.nan
    cons_df["length"] = np.nan    

    rebased_speeds_rips, rebased_speeds_cons = [], []
    for i_bin in range(start, end):
        v_rips = np.vstack(rips_df[f"{i_bin}"] - rips_df[f"{i_bin-1}"])
        v_cons = np.vstack(cons_df[f"{i_bin}"] - cons_df[f"{i_bin-1}"])
        
        v_base_rips = [rip[f"v_{rip['phase']}_4_8"]  for _, rip in rips_df.iterrows()]
        v_base_cons = [rip[f"v_{rip['phase']}_4_8"]  for _, rip in cons_df.iterrows()]

        v_rebased_rips = [mngs.linalg.rebase_a_vec(v_rips[ii], v_base_rips[ii])
                          for ii in range(len(v_rips))]
        v_rebased_cons = [mngs.linalg.rebase_a_vec(v_cons[ii], v_base_cons[ii])
                          for ii in range(len(v_cons))]

        speed_rebased_rips = np.abs(np.array(v_rebased_rips))
        speed_rebased_cons = np.abs(np.array(v_rebased_cons))        

        rebased_speeds_rips.append(speed_rebased_rips)
        rebased_speeds_cons.append(speed_rebased_cons)

    rebased_speeds_rips = np.vstack(rebased_speeds_rips)
    rebased_speeds_cons = np.vstack(rebased_speeds_cons)

    lengths_rips = rebased_speeds_rips.sum(axis=0)
    lengths_cons = rebased_speeds_cons.sum(axis=0)

    rips_df["length"] = lengths_rips
    cons_df["length"] = lengths_cons    

    df = {}
    for phase in PHASES:
        lengths_rips_phase = np.hstack(lengths_rips[rips_df.phase == phase])
        lengths_cons_phase = np.hstack(lengths_cons[cons_df.phase == phase])
        nan_indi = np.isnan(lengths_rips_phase) + np.isnan(lengths_cons_phase)
        lengths_rips_phase = lengths_rips_phase[~nan_indi]
        lengths_cons_phase = lengths_cons_phase[~nan_indi]
        

        df[f"SWR_{phase}"] = lengths_rips_phase
        df[f"Control_{phase}"] = lengths_cons_phase

        w, pval, dof, eff = mngs.stats.brunner_munzel_test(
            lengths_cons_phase,
            lengths_rips_phase,
        )

        print(phase, round(pval, 3), eff)

    mngs.io.save(
        mngs.general.force_dataframe(df),
        "./tmp/figs/box/length_comparison_rips_and_cons.csv",
    )
    return df, rips_df, cons_df


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

    df, rips_df, cons_df = compair_length_in_memory_load_direction_of_rips_and_cons(rips_df, cons_df)

    rips_df["length"][(rips_df.phase == "Encoding") * (rips_df.set_size == 4)]
    import seaborn as sns
    fig, axes = plt.subplots(ncols=2, sharey=True)
    for i_ax, ax in enumerate(axes):
        sns.boxplot(
            data=[cons_df, rips_df][i_ax],
            y="length",
            x="phase",
            order=PHASES,
            hue="set_size",
            ax=ax,
            showfliers=False,
        )
        ax.set_yscale("log")
    plt.show()

    for phase in PHASES:
        print(phase)
        d1 = df[f"Control_{phase}"]
        d2 = df[f"SWR_{phase}"]
        print((~np.isnan(d1)).sum())
        print(mngs.gen.describe(d1, "median"))
        print((~np.isnan(d2)).sum())
        print(mngs.gen.describe(d2, "median"))

    # fig, ax = plt.subplots()
    # for i_key, (key, vals) in enumerate(df.items()):
    #     ax.boxplot(
    #         vals,
    #         positions=[i_key],
    #         showfliers=False,
    #     )
    # plt.show()

    dfs = []

    for phase in PHASES:
        speeds = collect_speed()
        speeds_rips = collect_peri_swr_speed(speeds, rips_df[rips_df.phase == phase])
        mm_rips, sd_rips = np.nanmean(speeds_rips, axis=0), np.nanstd(
            speeds_rips, axis=0
        )
        nn_rips = (~np.isnan(speeds_rips)).sum(axis=0)
        ci_rips = 1.96 * sd_rips / nn_rips

        speeds_cons = collect_peri_swr_speed(speeds, cons_df[cons_df.phase == phase])
        mm_cons, sd_cons = np.nanmean(speeds_cons, axis=0), np.nanstd(
            speeds_cons, axis=0
        )
        nn_cons = (~np.isnan(speeds_cons)).sum(axis=0)
        ci_cons = 1.96 * sd_cons / nn_cons

        xx_s = (np.arange(-12, 12) * 50) + 25
        df = pd.DataFrame(
            {
                "xx_s": xx_s,
                f"{phase}_under_SWR+": mm_rips - ci_rips,
                f"{phase}_mean_SWR+": mm_rips,
                f"{phase}_upper_SWR+": mm_rips + ci_rips,
                f"{phase}_under_SWR-": mm_cons - ci_cons,
                f"{phase}_mean_SWR-": mm_cons,
                f"{phase}_upper_SWR-": mm_cons + ci_cons,
            }
        )
        dfs.append(df)
    df = pd.concat(dfs, axis=1)

    mngs.io.save(df, "./tmp/figs/line/peri_SWR_speed.csv")
