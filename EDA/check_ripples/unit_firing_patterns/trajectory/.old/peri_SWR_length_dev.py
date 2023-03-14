#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-28 12:02:25 (ywatanabe)"

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


def compair_length_rips_and_cons(rips_df, cons_df, direction=None):
    width_ms = 500
    width_bins = width_ms / BIN_SIZE.magnitude
    start = -int(width_bins / 2)
    end = int(width_bins / 2)
    rips_df["length"] = np.nan
    cons_df["length"] = np.nan

    speeds_rips, speeds_cons = [], []
    set_size_rips, set_size_cons = [], []
    for i_bin in range(start, end):
        set_size_rips.append(rips_df.set_size)
        set_size_cons.append(cons_df.set_size)
        
        v_rips = np.vstack(rips_df[f"{i_bin}"] - rips_df[f"{i_bin-1}"])
        v_cons = np.vstack(cons_df[f"{i_bin}"] - cons_df[f"{i_bin-1}"])

        if direction is None:
            speed_rips = [mngs.linalg.nannorm(v_rips[ii]) for ii in range(len(v_rips))]
            speed_cons = [mngs.linalg.nannorm(v_cons[ii]) for ii in range(len(v_cons))]
            speeds_rips.append(speed_rips)
            speeds_cons.append(speed_cons)
            
        else:
            
            if direction == "memory_load_direction":
                v_base_rips = [
                    rip[f"v_{rip['phase']}_4_8"] for _, rip in rips_df.iterrows()
                ]
                v_base_cons = [
                    rip[f"v_{rip['phase']}_4_8"] for _, rip in cons_df.iterrows()
                ]
                
            if direction == "random_memory_load_direction":
                v_base_rips = [
                    np.array([random.random() - 0.5 for _ in range(3)])
                    for _ in range(len(rips_df))
                ]
                v_base_cons = [
                    np.array([random.random() - 0.5 for _ in range(3)])
                    for _ in range(len(rips_df))
                ]

            v_rebased_rips = [
                mngs.linalg.rebase_a_vec(v_rips[ii], v_base_rips[ii])
                for ii in range(len(v_rips))
            ]
            v_rebased_cons = [
                mngs.linalg.rebase_a_vec(v_cons[ii], v_base_cons[ii])
                for ii in range(len(v_cons))
            ]

            rebased_speed_rips = np.abs(np.array(v_rebased_rips))
            rebased_speed_cons = np.abs(np.array(v_rebased_cons))

            speeds_rips.append(rebased_speed_rips)
            speeds_cons.append(rebased_speed_cons)

    set_size_rips = np.vstack(set_size_rips)
    set_size_cons = np.vstack(set_size_cons)
            
    speeds_rips = np.vstack(speeds_rips)
    speeds_cons = np.vstack(speeds_cons)

    # length_rips = speeds_rips.sum(axis=0)
    # length_cons = speeds_cons.sum(axis=0)    

    lengths_rips = speeds_rips.sum(axis=0)
    lengths_cons = speeds_cons.sum(axis=0)

    rips_df["length"] = lengths_rips
    cons_df["length"] = lengths_cons

    df = {}
    df_ss = {}
    for phase in PHASES:
        set_size_rips_phase = np.hstack(set_size_rips[0, rips_df.phase == phase])
        set_size_cons_phase = np.hstack(set_size_cons[0, cons_df.phase == phase])

        lengths_rips_phase = lengths_rips[rips_df.phase == phase]
        lengths_cons_phase = lengths_cons[cons_df.phase == phase]
        
        # speeds_rips_phase = np.hstack(speeds_rips[:, rips_df.phase == phase])
        # speeds_cons_phase = np.hstack(speeds_cons[:, cons_df.phase == phase])
        nan_indi = np.isnan(lengths_rips_phase) + np.isnan(lengths_cons_phase)
        
        set_size_rips_phase = np.hstack(set_size_rips_phase)[~nan_indi]
        set_size_cons_phase = np.hstack(set_size_cons_phase)[~nan_indi]
        
        lengths_rips_phase = lengths_rips_phase[~nan_indi]
        lengths_cons_phase = lengths_cons_phase[~nan_indi]

        df[f"SWR_{phase}"] = lengths_rips_phase
        df_ss[f"SWR_{phase}_set_size"] = set_size_rips_phase        
        df[f"Control_{phase}"] = lengths_cons_phase
        df_ss[f"Control_{phase}_set_size"] = set_size_cons_phase                

        w, pval, dof, eff = mngs.stats.brunner_munzel_test(
            lengths_cons_phase,
            lengths_rips_phase,
        )

        print(phase, round(pval, 3), eff)

    mngs.io.save(
        mngs.general.force_dataframe(df),
        "./tmp/figs/box/length_comparison_rips_and_cons.csv",
    )
    return df, df_ss


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

    df, df_ss = compair_length_rips_and_cons(rips_df, cons_df, direction=None)
    df_ml, df_ml_ss = compair_length_rips_and_cons(rips_df, cons_df, direction="memory_load_direction")
    df_rd, df_rd_ss = compair_length_rips_and_cons(rips_df, cons_df, direction="random_memory_load_direction")

    mngs.gen.force_dataframe(df)

    out_df = {}
    for (col, length), (_, set_size) in zip(df.items(), df_ss.items()):
        for ss in np.unique(set_size):
            out_df[f"{col}_{ss}"] = length[set_size == ss]
    mngs.io.save(mngs.gen.force_dataframe(out_df), "./tmp/figs/box/set_size_dependent_peri_SWR_length.csv")


    def save_corr(df, df_ss, key):
        mngs.gen.fix_seeds(42, np=np)                    
        data = np.log10(df[key])
        set_sizes = df_ss[f"{key}_set_size"]
        corr_obs = np.corrcoef(data, set_sizes)[0,1]
        shuffled_corrs = [np.corrcoef(data, np.random.permutation(set_sizes))[0,1] for _ in range(1000)]
        out = {"observed": corr_obs, "surrogate": shuffled_corrs}
        rank = bisect_right(shuffled_corrs, corr_obs)
        print(round(corr_obs, 3))
        print(rank)
        # mngs.io.save(out, f"./tmp/figs/corr/{count}_{key}_peri-SWR_length.pkl")

    count = 17
    _df, _df_ss = df, df_ss
    for key in _df.keys():
        if "Control" in key:
            print(key)
            save_corr(_df, _df_ss, key)
            count += 1
    for key in _df.keys():
        if "SWR" in key:
            print(key)            
            save_corr(_df, _df_ss, key)
            count += 1


    df_melted = mngs.gen.force_dataframe(df).melt()
    df_ss_melted = mngs.gen.force_dataframe(df_ss).melt()
    df_ml_melted = mngs.gen.force_dataframe(df_ml).melt()
    df_ml_ss_melted = mngs.gen.force_dataframe(df_ml_ss).melt()    
    df_rd_melted = mngs.gen.force_dataframe(df_rd).melt()
    df_rd_ss_melted = mngs.gen.force_dataframe(df_rd_ss).melt()

    df_melted["set_size"] = df_ss_melted["value"]
    df_ml_melted["set_size"] = df_ml_ss_melted["value"]
    df_rd_melted["set_size"] = df_rd_ss_melted["value"]    

    df_melted["value"] = df_melted["value"].replace({"": np.nan}).astype(float)
    df_ml_melted["value"] = df_ml_melted["value"].replace({"": np.nan}).astype(float)
    df_rd_melted["value"] = df_rd_melted["value"].replace({"": np.nan}).astype(float)    

    import seaborn as sns

    fig, axes = plt.subplots(ncols=3, sharey=True)
    for i_ax, ax in enumerate(axes):
        sns.boxplot(
            data=[df_melted, df_ml_melted, df_rd_melted][i_ax],
            y="value",
            x="variable",
            order=[f"Control_{phase}" for phase in PHASES]
            + [f"SWR_{phase}" for phase in PHASES],
            # order=PHASES,
            hue="set_size",
            hue_order=[4, 6, 8],
            ax=ax,
            showfliers=False,
        )
        ax.set_yscale("log")
    plt.show()

    from bisect import bisect_right
    mngs.gen.fix_seeds(42, np=np)
    for phase in PHASES:
        melted = df_ml_melted
        tmp = melted[melted.variable == f"Control_{phase}"]
        tmp = tmp[~tmp.isna().any(axis=1)]
        corr_obs = np.corrcoef(np.log10(tmp["value"].astype(float)), tmp["set_size"].astype(float))[0,1]
        corrs_surrogate = [np.corrcoef(np.log10(tmp["value"].astype(float)),
                                       np.random.permutation(tmp["set_size"].astype(float)))[0,1]
                           for _ in range(1000)]
        print(corr_obs)
        rank = bisect_right(corrs_surrogate, corr_obs)
        print(rank)

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
        lengths = collect_length()
        lengths_rips = collect_peri_swr_length(lengths, rips_df[rips_df.phase == phase])
        mm_rips, sd_rips = np.nanmean(lengths_rips, axis=0), np.nanstd(
            lengths_rips, axis=0
        )
        nn_rips = (~np.isnan(lengths_rips)).sum(axis=0)
        ci_rips = 1.96 * sd_rips / nn_rips

        lengths_cons = collect_peri_swr_length(lengths, cons_df[cons_df.phase == phase])
        mm_cons, sd_cons = np.nanmean(lengths_cons, axis=0), np.nanstd(
            lengths_cons, axis=0
        )
        nn_cons = (~np.isnan(lengths_cons)).sum(axis=0)
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

    mngs.io.save(df, "./tmp/figs/line/peri_SWR_length.csv")
