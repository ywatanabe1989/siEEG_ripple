#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-27 08:33:08 (ywatanabe)"

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
        i_bin = int(event.center_time / (50/1000))
        _speed = speeds[(speeds.subject == event.subject)*
              (speeds.session == event.session)*
              (speeds.trial_number == event.trial_number)
              ]
        _speeds = []
        for ii in range(-12, 12):
            try:
                _speeds.append(_speed[i_bin-ii].iloc[0]) # fixme
            except:
                _speeds.append(np.nan)
        speeds_all.append(_speeds)
    return np.vstack(speeds_all)

def compair_speed_of_rips_and_cons(rips_df, cons_df):
    width_ms = 500
    width_bins = width_ms / BIN_SIZE.magnitude
    start = -int(width_bins / 2)
    end = int(width_bins / 2)

    speeds_rips, speeds_cons = [], []
    for i_bin in range(start, end):
        coords_rips = np.vstack(rips_df[f"{i_bin}"] - rips_df[f"{i_bin-1}"])
        coords_cons = np.vstack(cons_df[f"{i_bin}"] - cons_df[f"{i_bin-1}"])

        speed_rips = [mngs.linalg.nannorm(coords_rips[ii]) for ii in range(len(coords_rips))]
        speed_cons = [mngs.linalg.nannorm(coords_cons[ii]) for ii in range(len(coords_cons))]

        speeds_rips.append(speed_rips)
        speeds_cons.append(speed_cons)
    speeds_rips = np.vstack(speeds_rips)
    speeds_cons = np.vstack(speeds_cons)

    df = {}
    for phase in PHASES:
        speeds_rips_phase = np.hstack(speeds_rips[:, rips_df.phase == phase])
        speeds_cons_phase = np.hstack(speeds_cons[:, cons_df.phase == phase])
        nan_indi = np.isnan(speeds_rips_phase) + np.isnan(speeds_cons_phase)

        speeds_rips_phase = speeds_rips_phase[~nan_indi]
        speeds_cons_phase = speeds_cons_phase[~nan_indi]

        df[f"SWR_{phase}"] = speeds_rips_phase
        df[f"Control_{phase}"] = speeds_cons_phase

        w, pval, dof, eff = mngs.stats.brunner_munzel_test(
            speeds_cons_phase,
            speeds_rips_phase,            
            )
            
        # stats, pval = brunnermunzel(
        #     speeds_rips_phase,
        #     speeds_cons_phase,
        # )
        print(phase, round(pval, 3), eff)

    mngs.io.save(
        mngs.general.force_dataframe(df),
        "./tmp/figs/box/speed_comparison_rips_and_cons.csv",
    )
    return df


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

    df = compair_speed_of_rips_and_cons(rips_df, cons_df)

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
        mm_rips, sd_rips = np.nanmean(speeds_rips, axis=0), np.nanstd(speeds_rips, axis=0)
        nn_rips = (~np.isnan(speeds_rips)).sum(axis=0)
        ci_rips = 1.96 * sd_rips / nn_rips

        speeds_cons = collect_peri_swr_speed(speeds, cons_df[cons_df.phase == phase])            
        mm_cons, sd_cons = np.nanmean(speeds_cons, axis=0), np.nanstd(speeds_cons, axis=0)
        nn_cons = (~np.isnan(speeds_cons)).sum(axis=0)        
        ci_cons = 1.96 * sd_cons / nn_cons

        xx_s = (np.arange(-12, 12)*50) + 25
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


