#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-28 12:23:13 (ywatanabe)"


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
from time import sleep

from itertools import combinations
from bisect import bisect_right
import pingouin as pg

# Functions
def calc_corr(data, event, return_raw=False):
    mngs.general.fix_seeds(42, np=np)

    is_nan = data.length.isna()
    corr_observed = np.corrcoef(
        np.array(data.length.astype(float)[~is_nan]),
        np.array(data.set_size.astype(float)[~is_nan]),
    )[0, 1]

    corrs_shuffle = []
    for _ in range(1000):
        corr_shuffle = np.corrcoef(
            data.length.astype(float)[~is_nan],
            np.random.permutation(data.set_size.astype(float)[~is_nan]),
        )[0, 1]
        corrs_shuffle.append(corr_shuffle)
    corrs_shuffle = pd.DataFrame(corrs_shuffle)

    if return_raw:
        return np.array(corr_observed), np.array(corrs_shuffle).squeeze()

    corr_0005 = corrs_shuffle.quantile(0.005).iloc[0]
    corr_0025 = corrs_shuffle.quantile(0.025).iloc[0]
    corr_0975 = corrs_shuffle.quantile(0.975).iloc[0]
    corr_0995 = corrs_shuffle.quantile(0.9995).iloc[0]

    rank = bisect_right(sorted(np.array(corrs_shuffle)), corr_observed)
    rank_rate = rank / len(corrs_shuffle)

    # p_val = 1 - rank_rate # one-directed; larger
    if 0.5 <= rank_rate:
        p_val = (1 - rank_rate) * 2
    if rank_rate < 0.5:
        p_val = rank_rate * 2
    return corr_observed, corrs_shuffle, p_val


def mk_v_base_str(base_phase_1, base_phase_2):
    if base_phase_1 is None:
        return f"g{base_phase_2[0]}_4 to g{base_phase_2[0]}_8 direction"
    elif base_phase_2 is None:
        return f"g{base_phase_1[0]}_4 to g{base_phase_1[0]}_8 direction"
    else:
        return f"g{base_phase_1[0]} to g{base_phase_2[0]} direction"




def get_length_df(rips_df, cons_df):
    def get_directed_speed(rips_df, tgt_bin, base_phase_1, base_phase_2):
        v = np.vstack((rips_df[f"{tgt_bin}"] - rips_df[f"{tgt_bin-1}"]))

        if base_phase_1 is None:
            v_base = np.vstack(rips_df[f"v_{base_phase_2}_4_8"])
        elif base_phase_2 is None:
            v_base = np.vstack(rips_df[f"v_{base_phase_1}_4_8"])
        else:
            v_base = np.vstack(rips_df[f"{base_phase_1}"]) - np.vstack(
                rips_df[f"{base_phase_2}"]
            )

        v_rebased = np.array(
            ([mngs.linalg.rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))])
        )
        v_rebased_norm = np.abs(v_rebased)
        return v_rebased_norm
    
    def get_length(
        events_df_ss, base_phase_1, base_phase_2, tgt_bin_start=-5
    ):
        lengthes_all = []
        phases_all = []
        for tgt_bin in range(tgt_bin_start, np.abs(tgt_bin_start) + 1):
            lengthes = get_directed_speed(events_df_ss, tgt_bin, base_phase_1, base_phase_2)
            lengthes_all.append(lengthes)
            phases_all.append(events_df_ss.phase)
        # return np.hstack(lengthes_all), np.hstack(
        #     phases_all
        # )
        return np.log10(np.hstack(lengthes_all)), np.hstack(
            phases_all
        )
    
    dfs = []
    for is_control in [False, True]:
        events_df = cons_df if is_control else rips_df
        for set_size in [4, 6, 8]:
            events_df_ss = events_df.copy()

            if set_size is not None:
                events_df_ss = events_df_ss[(events_df_ss.set_size == set_size)]

            for base_phase_1, base_phase_2 in combinations([None] + PHASES, 2):
                lengthes, event_phases = get_length(
                    events_df_ss, base_phase_1, base_phase_2
                )

                df = pd.DataFrame(
                    {
                        "length": lengthes,
                        "event_phase": event_phases,
                    }
                )
                df["is_control"] = is_control
                df["set_size"] = set_size
                df["base_phase_1"] = base_phase_1
                df["base_phase_2"] = base_phase_2
                dfs.append(df.copy())
    df = pd.concat(dfs)

    df["Event"] = df["is_control"].replace({False: "SWR", True: "Control"})
    # df["set_size"]
    df["set_size_Event"] = mngs.ml.utils.merge_labels(df["set_size"], df["Event"])
    return df



def get_stats(df):
    df_stats = pd.DataFrame()
    for i_event, event in enumerate(["Control", "SWR"]):
        for iep, event_phase in enumerate(PHASES):
            # for ibp, (base_phase_1, base_phase_2) in enumerate(
            #     combinations([None] + PHASES, 2)
            # ):
            ibp = None
            base_phase_1 = None
            base_phase_2 = event_phase

            indi_base_phase_1 = (
                df.base_phase_1.isna()
                if base_phase_1 is None
                else df.base_phase_1 == base_phase_1
            )
            indi_base_phase_2 = (
                df.base_phase_2.isna()
                if base_phase_2 is None
                else df.base_phase_2 == base_phase_2
            )

            print("\n" + "-" * 40 + "\n")
            data = df[
                (df.event_phase == event_phase)
                * indi_base_phase_1
                * indi_base_phase_2
                * (df.Event == event)
            ]

            corr, corrs_shuffled, p_val_corr = calc_corr(data, event)
            print(iep, event_phase)
            print(ibp, base_phase_1, base_phase_2)
            if (base_phase_1 == None) and (event_phase == base_phase_2):
                print(event, round(p_val_corr, 3))
                import ipdb; ipdb.set_trace()
                out = {"observed": corr, "surrogate": corrs_shuffled}
                mngs.io.save(
                    out,
                    f"./tmp/figs/corr/{iep*2 + i_event + 9}_{event_phase}_{event}.pkl",
                )

            for ss1, ss2 in [(4, 6), (6, 8), (4, 8)]:
                d1 = data[data.set_size == ss1]["length"]
                d2 = data[data.set_size == ss2]["length"]
                d1 = d1[~d1.isna()]
                d2 = d2[~d2.isna()]
                nan_indi = d1.isna() + d2.isna()
                w, pval, dof, pp = mngs.stats.brunner_munzel_test(d1, d2)

                df_stats_new = pd.DataFrame(
                    pd.Series(
                        {
                            "Event": event,
                            "Event phase": event_phase,
                            "Base direction": mk_v_base_str(
                                base_phase_1, base_phase_2
                            ),
                            "Set size 1": ss1,
                            "Set size 2": ss2,
                            "Corr.": corr.round(6),
                            "Unc. p-value_corr": p_val_corr,
                            "Unc. p-value": pval.round(6),
                            "effect size": pp.round(6),
                            "n1": len(d1),
                            "n2": len(d2),
                        }
                    )
                )
                df_stats = pd.concat([df_stats, df_stats_new], axis=1)
            print("\n" + "-" * 40 + "\n")
    return df_stats



def print_unc_pvals_corr(df_length_stats):
    for phase in PHASES:
        tmp = df_length_stats.T[
            (df_length_stats.T["Event phase"] == phase)
            * df_length_stats.T["Base direction"]
            == f"g{phase[0]}_4 to g{phase[0]}_8 direction"
        ]
        print(phase)
        print("SWR unc. p-val", tmp["Unc. p-value_corr"].iloc[0])
        print("Control unc. p-val", tmp["Unc. p-value_corr"].iloc[4])
        print()


def print_corr(df_length_stats):
    for phase in PHASES:
        tmp = df_length_stats.T[
            (df_length_stats.T["Event phase"] == phase)
            * df_length_stats.T["Base direction"]
            == f"g{phase[0]}_4 to g{phase[0]}_8 direction"
        ]
        print(phase)
        print("SWR corr. coef", tmp["Corr."].iloc[0])
        print("Control corr. coef", tmp["Corr."].iloc[4])
        print()


def get_correlations(df_length):
    """
    df = df_length
    """
    corr_df = pd.DataFrame()
    for phase in PHASES:
        df_length_phase_directed = df_length[
            (df_length.event_phase == phase)
            * df_length.base_phase_1.isna()
            * (df_length.base_phase_2 == phase)
        ]

        df_length_phase_directed = df_length_phase_directed[
            ["length", "Event", "set_size"]
        ]
        df_length_phase_directed = df_length_phase_directed.sort_values(
            ["Event", "set_size"]
        )

        for event in ["SWR", "Control"]:
            data = df_length_phase_directed[(df_length_phase_directed.Event == event)]
            corr_obs, corrs_shuffled = calc_corr(data, event, return_raw=True)
            _corr_df = pd.DataFrame(
                {
                    "phase": phase,
                    "correlation": np.hstack([corr_obs, corrs_shuffled]),
                    "is_shuffle": [False] + [True for _ in range(len(corrs_shuffled))],
                }
            )
            _corr_df["event"] = event
            corr_df = pd.concat([corr_df, _corr_df])
    return corr_df


def save_df_length_phases(df):

    for phase in PHASES:
        df_length_phase = df[
            (df.event_phase == phase)
            * df.base_phase_1.isna()
            * (df.base_phase_2 == phase)
        ]
        df_length_phase = df_length_phase[["length", "Event", "set_size"]]
        df_length_phase = df_length_phase.sort_values(["Event", "set_size"])

        df_length_phase_out = {}
        for event in ["SWR", "Control"]:
            for set_size in [4, 6, 8]:
                df_length_phase_out[f"{event}_set_size_{set_size}"] = np.array(
                    df_length_phase[
                        (df_length_phase.Event == event)
                        * (df_length_phase.set_size == set_size)
                    ]["length"]
                )
        df_length_phase_out = mngs.gen.force_dataframe(df_length_phase_out)
        mngs.io.save(
            df_length_phase_out,
            f"./tmp/figs/box/peri_ripple_length/peri_{phase}_"
            f"ripple_length_based_on_g{phase[0]}4_to_g{phase[0]}8_raw.csv",
        )

def plot_corr_df(corr_df):
    for phase in PHASES:
        fig, ax = plt.subplots()
        data = corr_df[(corr_df.is_shuffle != False) * (corr_df.phase == phase)]
        data = mngs.plt.add_hue(data)
        sns.violinplot(
            data=data,
            y="correlation",
            x="event",
            order=["Control", "SWR"],
            hue="hue",
            split=True,
            ax=ax,
            color=mngs.plt.colors.to_RGBA("gray", alpha=1.0),
        )

        for event in ["Control", "SWR"]:
            data = corr_df[
                (corr_df.is_shuffle == False)
                * (corr_df.event == event)
                * (corr_df.phase == phase)
            ]["correlation"]
            ax.scatter(
                event,
                data,
                color=mngs.plt.colors.to_RGBA("red", alpha=1.0),
                s=100,
            )

        ax.set_ylim(-0.17, 0.17)
        ax.set_xlim(-1, 2)
        ax.set_title(phase)
        mngs.io.save(
            fig,
            f"./tmp/figs/violin/peri_ripple_length/"
            f"peri_{phase}_ripple_length_correlation_based_on_g{phase[0]}4_to_g{phase[0]}8_fig.tif",
        )

    plt.show()


if __name__ == "__main__":
    import mngs
    import numpy as np

    mngs.gen.fix_seeds(42, np=np)
    # matplotlib.use("Agg")

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

    df_length = get_length_df(rips_df, cons_df)
    df_length_stats = get_stats(df_length)
    mngs.io.save(df_length_stats.T, "./tmp/figs/box/peri_ripple_traj_length_stats.csv")

    print_unc_pvals_corr(df_length_stats)
    print_corr(df_length_stats)

    corr_df = get_correlations(df_length)

    save_df_length_phases(df_length)

    plot_corr_df(corr_df)
