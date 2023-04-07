#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-14 14:03:47 (ywatanabe)"

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


def compair_speed_rips_and_cons(rips_df, cons_df, direction=None):
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

    # lengths_rips = rebased_speeds_rips.sum(axis=0)
    # lengths_cons = rebased_speeds_cons.sum(axis=0)

    # rips_df["length"] = lengths_rips
    # cons_df["length"] = lengths_cons

    df = {}
    df_ss = {}
    for phase in PHASES:
        set_size_rips_phase = np.hstack(set_size_rips[:, rips_df.phase == phase])
        set_size_cons_phase = np.hstack(set_size_cons[:, cons_df.phase == phase])

        speeds_rips_phase = np.hstack(speeds_rips[:, rips_df.phase == phase])
        speeds_cons_phase = np.hstack(speeds_cons[:, cons_df.phase == phase])

        nan_indi_rips = np.isnan(speeds_rips_phase)
        nan_indi_cons = np.isnan(speeds_cons_phase)
        # nan_indi = np.isnan(speeds_rips_phase) + np.isnan(speeds_cons_phase)

        set_size_rips_phase = np.hstack(set_size_rips_phase)[~nan_indi_rips]
        set_size_cons_phase = np.hstack(set_size_cons_phase)[~nan_indi_cons]

        speeds_rips_phase = speeds_rips_phase[~nan_indi_rips]
        speeds_cons_phase = speeds_cons_phase[~nan_indi_cons]

        df[f"SWR_{phase}"] = speeds_rips_phase
        df_ss[f"SWR_{phase}_set_size"] = set_size_rips_phase
        df[f"Control_{phase}"] = speeds_cons_phase
        df_ss[f"Control_{phase}_set_size"] = set_size_cons_phase

        w, pval, dof, eff = mngs.stats.brunner_munzel_test(
            speeds_cons_phase,
            speeds_rips_phase,
        )
        mark = mngs.stats.to_asterisks(pval)
        print(phase, round(pval, 3), mark)

        # print(phase, round(pval, 3), eff)

    df = mngs.general.force_dataframe(df)
    return df, df_ss


def print_stats(df):
    for phase in PHASES:
        print(phase)
        d1 = df[f"Control_{phase}"]
        d2 = df[f"SWR_{phase}"]
        d1 = d1.replace({"": np.nan})
        d2 = d2.replace({"": np.nan})
        print(f"n_SWR-: {(~np.isnan(d1)).sum()}")
        mm, ss = mngs.gen.describe(d1, "median")
        print(f"median: {round(mm, 3)}; IQR: {round(ss, 3)}")
        print()
        print(f"n_SWR+: {(~np.isnan(d2)).sum()}")
        mm, ss = mngs.gen.describe(d2, "median")
        print(f"median: {round(mm, 3)}; IQR: {round(ss, 3)}")
        print()


def calc_speed_of_SWR(event_df):
    for ii in range(-5, 6):
        coords_ii = event_df[f"{ii}"] - event_df[f"{ii-1}"]
        speeds_ii = [mngs.linalg.nannorm(coords_ii[jj]) for jj in range(len(coords_ii))]
        event_df[f"speed_{ii}"] = speeds_ii
    event_df["speed"] = np.nanmean(
        np.array([event_df[f"speed_{ii}"] for ii in range(-5, 6)]), axis=0
    )
    return event_df


def sort_peri_SWR_speed(cons_df, rips_df):
    out_df = {}
    for event_str, event_df in zip(["SWR-", "SWR+"], [cons_df, rips_df]):
        for phase in PHASES:
            for set_size in [4, 6, 8]:
                data = event_df[
                    (event_df.phase == phase) * (event_df.set_size == set_size)
                ].speed
                out_df[f"{event_str}_{phase}_{set_size}"] = data
    return mngs.gen.force_dataframe(out_df)


def plot(cons_df_m, rips_df_m):
    # Set size dependancy
    fig, axes = plt.subplots(
        ncols=2, sharex=True, sharey=True, figsize=(6.4 * 2, 4.8 * 2)
    )
    for ax, event_df in zip(axes, [cons_df_m, rips_df_m]):
        sns.boxplot(
            data=event_df,
            x="phase",
            hue="set_size",
            y="speed",
            order=PHASES,
            ax=ax,
            showfliers=False,
        )
    axes[0].set_title("SWR-")
    axes[1].set_title("SWR+")
    axes[0].set_ylabel("Peri-SWR speed from O")
    axes[1].set_ylabel("Peri-SWR speed from O")
    return fig


def corr_test(cons_df_m, rips_df_m):
    count = 0

    for phase in PHASES:
        print(phase)
        # Control
        indi = cons_df_m.phase == phase
        pval_cons, corr_obs_cons, corrs_shuffled_cons = mngs.stats.corr_test(
            np.log10(cons_df_m["speed"][indi]), cons_df_m["set_size"][indi]
        )
        out = {"observed": corr_obs_cons, "surrogate": corrs_shuffled_cons}
        count += 1
        spath = f"./tmp/figs/corr/peri_SWR_speed/match_{match}/{count}_by_phase_and_set_size_SWR-.pkl"
        spath = spath.replace(
            ".pkl", f"corr_{round(corr_obs_cons,2)}_pval_{round(pval_cons,3)}.pkl"
        )
        mngs.io.save(
            out,
            spath,
        )
        # SWR
        indi = rips_df_m.phase == phase
        pval_rips, corr_obs_rips, corrs_shuffled_rips = mngs.stats.corr_test(
            np.log10(rips_df_m["speed"][indi]), rips_df_m["set_size"][indi]
        )
        out = {"observed": corr_obs_rips, "surrogate": corrs_shuffled_rips}
        count += 1
        spath = f"./tmp/figs/corr/peri_SWR_speed/match_{match}/{count}_by_phase_and_set_size_SWR+.pkl"
        spath = spath.replace(
            ".pkl", f"corr_{round(corr_obs_rips,2)}_pval_{round(pval_rips,3)}.pkl"
        )
        mngs.io.save(
            out,
            spath,
        )
        print()


def plot(cons_df, rips_df):
    # Set size dependancy
    fig, axes = plt.subplots(
        ncols=2, sharex=True, sharey=True, figsize=(6.4 * 2, 4.8 * 2)
    )
    for ax, event_df in zip(axes, [cons_df, rips_df]):
        sns.boxplot(
            data=event_df,
            x="phase",
            hue="set_size",
            y="speed",
            order=PHASES,
            ax=ax,
            showfliers=False,
        )
    axes[0].set_title("SWR-")
    axes[1].set_title("SWR+")
    axes[0].set_ylabel("Peri-SWR speed")
    axes[1].set_ylabel("Peri-SWR speed")
    return fig


def sort_peri_SWR_speed_from_SWR_center(rips_df_m, cons_df_m):
    dfs = []

    for phase in PHASES:
        speeds = collect_speed()
        speeds_rips = collect_peri_swr_speed(
            speeds, rips_df_m[rips_df_m.phase == phase]
        )
        mm_rips, sd_rips = np.nanmean(speeds_rips, axis=0), np.nanstd(
            speeds_rips, axis=0
        )
        nn_rips = (~np.isnan(speeds_rips)).sum(axis=0)
        ci_rips = 1.96 * sd_rips / nn_rips

        speeds_cons = collect_peri_swr_speed(
            speeds, cons_df_m[cons_df_m.phase == phase]
        )
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
    return df


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
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )
    rips_df = calc_speed_of_SWR(rips_df)
    cons_df = calc_speed_of_SWR(cons_df)

    for match in [None, 1, 2]:
        if match is not None:
            rips_df_m = rips_df[rips_df.match == match]
            cons_df_m = cons_df[cons_df.match == match]
        else:
            rips_df_m = rips_df
            cons_df_m = cons_df

        # for boxplot in sigmaplot, peri-SWR distance from O by phase and set size
        df_m = sort_peri_SWR_speed(cons_df_m, rips_df_m)
        mngs.io.save(
            df_m,
            f"./tmp/figs/box/peri_SWR_speed/match_{match}/by_phase_and_set_size.csv",
        )

        mngs.gen.fix_seeds(42, np=np)
        corr_test(cons_df_m, rips_df_m)

        fig = plot(cons_df_m, rips_df_m)
        # plt.show()
        mngs.io.save(
            fig,
            f"./tmp/figs/box/peri_SWR_speed/match_{match}/set_size_dependent_increase_of_peri_SWR_speed.png",
        )

        df, df_ss = compair_speed_rips_and_cons(rips_df_m, cons_df_m, direction=None)
        mngs.io.save(
            df,
            f"./tmp/figs/box/peri_SWR_speed/match_{match}/speed_comparison_rips_and_cons.csv",
        )

        df_ml, df_ml_ss = compair_speed_rips_and_cons(
            rips_df_m, cons_df_m, direction="memory_load_direction"
        )
        df_rd, df_rd_ss = compair_speed_rips_and_cons(
            rips_df_m, cons_df_m, direction="random_memory_load_direction"
        )

        mngs.gen.force_dataframe(df)

        out_df = {}
        for (col, speed), (_, set_size) in zip(df.items(), df_ss.items()):
            for ss in np.unique(set_size):
                out_df[f"{col}_{ss}"] = speed[set_size == ss]
        mngs.io.save(
            mngs.gen.force_dataframe(out_df),
            "./tmp/figs/box/set_size_dependent_peri_SWR_speed.csv",
        )

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
        df_ml_melted["value"] = (
            df_ml_melted["value"].replace({"": np.nan}).astype(float)
        )
        df_rd_melted["value"] = (
            df_rd_melted["value"].replace({"": np.nan}).astype(float)
        )

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
            corr_obs = np.corrcoef(
                np.log10(tmp["value"].astype(float)), tmp["set_size"].astype(float)
            )[0, 1]
            corrs_surrogate = [
                np.corrcoef(
                    np.log10(tmp["value"].astype(float)),
                    np.random.permutation(tmp["set_size"].astype(float)),
                )[0, 1]
                for _ in range(1000)
            ]
            print(corr_obs)
            rank = bisect_right(corrs_surrogate, corr_obs)
            print(rank)

        print_stats(df)

        # for phase in PHASES:
        #     print(phase)
        #     d1 = df[f"Control_{phase}"]
        #     d2 = df[f"SWR_{phase}"]
        #     print((~np.isnan(d1)).sum())
        #     print(mngs.gen.describe(d1, "median"))
        #     print((~np.isnan(d2)).sum())
        #     print(mngs.gen.describe(d2, "median"))

        # fig, ax = plt.subplots()
        # for i_key, (key, vals) in enumerate(df.items()):
        #     ax.boxplot(
        #         vals,
        #         positions=[i_key],
        #         showfliers=False,
        #     )
        # plt.show()

        df = sort_peri_SWR_speed_from_SWR_center(rips_df_m, cons_df_m)
        mngs.io.save(df, f"./tmp/figs/line/peri_SWR_speed/match_{match}.csv")
