#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-09 11:12:59 (ywatanabe)"

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
def collect_dist():
    dists_all = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            traj_session = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            dist_session = norm(traj_session, axis=1)
            df = pd.DataFrame(dist_session)
            df["subject"] = subject
            df["session"] = session
            df["trial_number"] = np.arange(len(traj_session)) + 1
            dists_all.append(df)
    return pd.concat(dists_all)

def collect_peri_swr_dist(dists, events_df):
    dists_all = []
    for i_event, (_, event) in enumerate(events_df.iterrows()):
        i_bin = int(event.center_time / (50/1000))
        _dist = dists[(dists.subject == event.subject)*
              (dists.session == event.session)*
              (dists.trial_number == event.trial_number)
              ]
        _dists = []
        for ii in range(-11, 12):
            try:
                _dists.append(_dist[i_bin-ii].iloc[0])
            except:
                _dists.append(np.nan)
        dists_all.append(_dists)
    return np.vstack(dists_all)

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

        dists_rips.append(dist_rips)
        dists_cons.append(dist_cons)
        
    dists_rips = np.vstack(dists_rips)
    dists_cons = np.vstack(dists_cons)

    df = {}
    for phase in PHASES:
        dists_rips_phase = np.hstack(dists_rips[:, rips_df.phase == phase])
        dists_cons_phase = np.hstack(dists_cons[:, cons_df.phase == phase])
        
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

    mngs.io.save(
        mngs.general.force_dataframe(df),
        "./tmp/figs/box/dist_comparison_rips_and_cons.csv",
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

    df = compair_dist_of_rips_and_cons(rips_df, cons_df)

    for phase in PHASES:
        print(phase)
        d1 = df[f"Control_{phase}"]
        d2 = df[f"SWR_{phase}"]
        print((~np.isnan(d1)).sum())
        print(mngs.gen.describe(d1, "median"))
        print((~np.isnan(d2)).sum())                
        print(mngs.gen.describe(d2, "median"))
    

    dfs = []
    for phase in PHASES:
        dists = collect_dist()
        dists_rips = collect_peri_swr_dist(dists, rips_df[rips_df.phase == phase])
        mm_rips, sd_rips = np.nanmean(dists_rips, axis=0), np.nanstd(dists_rips, axis=0)
        nn_rips = (~np.isnan(dists_rips)).sum(axis=0)
        ci_rips = 1.96 * sd_rips / nn_rips

        dists_cons = collect_peri_swr_dist(dists, cons_df[cons_df.phase == phase])            
        mm_cons, sd_cons = np.nanmean(dists_cons, axis=0), np.nanstd(dists_cons, axis=0)
        nn_cons = (~np.isnan(dists_cons)).sum(axis=0)        
        ci_cons = 1.96 * sd_cons / nn_cons

        df = pd.DataFrame(
            {
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
        
    mngs.io.save(df, "./tmp/figs/line/peri_SWR_dist_from_O.csv")


