#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-05 12:44:24 (ywatanabe)"


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

# def plot_speed(rips_df, is_control=False, set_size=None, match=None):

#     events_str = "cons" if is_control else "rips"

#     if set_size is not None:
#         rips_df = rips_df[rips_df.set_size == set_size]
#     set_size_str = f"_set_size_{set_size}" if set_size is not None else ""

#     if match is not None:
#         rips_df = rips_df[rips_df.match == match]
#     match_str = f"_match_{match}" if match is not None else ""

#     cols_base_starts_ends = [
#         (None, None),
#         ("Fixation", "Encoding"),
#         ("Fixation", "Maintenance"),
#         ("Fixation", "Retrieval"),
#         ("Encoding", "Maintenance"),
#         ("Encoding", "Retrieval"),
#         ("Maintenance", "Retrieval"),
#         # ("vec_4_8_Fixation", "vec_4_8_Fixation"),
#         # ("vec_4_8_Encoding", "vec_4_8_Encoding"),
#         # ("vec_4_8_Maintenance","vec_4_8_Maintenance"),
#         # ("vec_4_8_Retrieval", "vec_4_8_Retrieval"),
#     ]

#     n_bases = len(cols_base_starts_ends)

#     fig, axes = plt.subplots(
#         ncols=n_bases, sharex=True, sharey=True, figsize=(6.4 * 3, 4.8 * 3)
#     )  # sharey=True,
#     xlim = (-500, 500)
#     out_df = pd.DataFrame()
#     for i_base, (cbs, cbe) in enumerate(cols_base_starts_ends):
#         dir_txt = f"{cbs}-{cbe}_based"

#         ax = axes[i_base]
#         i_ax = i_base

#         samp_m = mngs.general.listed_dict(PHASES)
#         samp_s = mngs.general.listed_dict(PHASES)

#         for i_phase, phase in enumerate(PHASES):

#             rips_df_phase = rips_df[rips_df.phase == phase]

#             centers_ms = []
#             for delta_bin in range(-39, 39):
#                 col1 = f"{delta_bin-1}"
#                 col1_str = f"{int((delta_bin-1)*BIN_SIZE.magnitude)}"
#                 col2 = f"{delta_bin}"
#                 col2_str = f"{int((delta_bin)*BIN_SIZE.magnitude)}"

#                 centers_ms.append(int((delta_bin - 0.5) * BIN_SIZE.magnitude))

#                 # gets vectors
#                 v = np.vstack(rips_df_phase[col2]) - np.vstack(rips_df_phase[col1])
#                 if cbs in PHASES:  # basis transformation
#                     v_base = np.vstack(rips_df_phase[cbe]) - np.vstack(
#                         rips_df_phase[cbs]
#                     )
#                     v_rebased = [
#                         mngs.linalg.rebase_a_vec(v[ii], v_base[ii])
#                         for ii in range(len(v))
#                     ]
#                     # v_rebased = np.log10(np.abs(v_rebased)) # fixme
#                     v_rebased = np.abs(v_rebased)  # fixme

#                 if cbs is None:  # just the norm
#                     v_rebased = [mngs.linalg.nannorm(v[ii]) for ii in range(len(v))]
#                     v_rebased = np.abs(v_rebased)  # fixme

#                 elif "vec_4_8" in cbs:
#                     v_base = rips_df_phase[cbs]
#                     v_rebased = [
#                         mngs.linalg.rebase_a_vec(v[ii], v_base[ii])
#                         for ii in range(len(v))
#                     ]
#                     v_rebased = np.abs(v_rebased)  # fixme

#                 mm, ss = mngs.gen.describe(v_rebased, method="mean")

#                 samp_m[phase].append(mm)
#                 samp_s[phase].append(ss)

#                 # nan_indi = np.isnan(v_rebased)
#                 # n_samp = (~nan_indi).sum()
#                 # v_rebased = v_revased[nan_indi]

#                 # samp_m[phase].append(np.nanmean(v_rebased))
#                 # samp_s[phase].append(np.nanstd(v_rebased) / 3)

#             ax.axhline(y=0, xmin=xlim[0], xmax=xlim[1], linestyle="--", color="gray")

#             ax.errorbar(
#                 x=np.array(centers_ms) + i_phase * 3,
#                 y=samp_m[phase],
#                 yerr=samp_s[phase],
#                 label=phase,
#             )
#             ax.legend(loc="upper right")

#         ax.set_xlim(xlim)

#         # ylim = (-1.25, 0.25) # log10
#         ylim = (0, 3)
#         if i_ax != 0:
#             title = dir_txt.replace("_based", "").replace("-", " -> ")
#             ax.set_ylim(ylim)
#             # ax.set_ylim(-0.3, 1)
#         else:
#             title = "Speed (norm)"
#             ax.set_ylim(ylim)
#             # ax.set_ylim(-0.3, 2)

#         ax.set_title(title)

#         samp_m = pd.DataFrame(samp_m)
#         samp_m.columns = [f"{col}_{cbs}-{cbe}_med" for col in samp_m.columns]
#         samp_s = pd.DataFrame(samp_s)
#         samp_s.columns = [f"{col}_{cbs}-{cbe}_s" for col in samp_s.columns]

#         # out_dict[f"{cbs}-{cbe}_med"] = samp_m
#         # out_dict[f"{cbs}-{cbe}_se"] = samp_s

#         out_df = pd.concat([out_df, pd.concat([samp_m, samp_s], axis=1)], axis=1)

#     fig.suptitle(f"Set size: {set_size}\nMatch: {match}")
#     fig.supylabel("Speed")
#     fig.supxlabel("Time from SWR [ms]")
#     # plt.show()
#     mngs.io.save(
#         fig,
#         f"./tmp/figs/hist/traj_speed/{match_str}/all_{events_str}{set_size_str}.png",
#     )
#     mngs.io.save(
#         out_df,
#         f"./tmp/figs/hist/traj_speed/{match_str}/all_{events_str}{set_size_str}.csv",
#     )
#     return fig


# def get_speed_vec_of_memory_load_direction(rips_df, tgt_bin, base_phase):
#     v = np.vstack((rips_df[f"{tgt_bin}"] - rips_df[f"{tgt_bin-1}"]))
#     v_base = np.vstack(rips_df[f"v_{base_phase}_4_8"])

#     v_rebased = np.array(
#         ([mngs.linalg.rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))])
#     )
#     # return v_rebased

#     v_rebased_norm = np.abs(v_rebased)
#     return v_rebased_norm


def get_length(
    rips_df, base_phase_1, base_phase_2, tgt_bin_start=-2
):  # , tgt_bin_end=6
    lengthes_all = []
    phases_all = []
    for tgt_bin in range(tgt_bin_start, np.abs(tgt_bin_start) + 2):
        # lengthes = get_speed_vec_of_memory_load_direction(rips_df, tgt_bin, base_phase)
        lengthes = get_speed_vec(rips_df, tgt_bin, base_phase_1, base_phase_2)
        lengthes_all.append(lengthes)
        phases_all.append(rips_df.phase)
    # return np.hstack(lengthes_all)  # mngs.gen.describe(lengthes_all)
    return np.log10(np.hstack(lengthes_all)), np.hstack(
        phases_all
    )  # mngs.gen.describe(lengthes_all)
    # return np.hstack(lengthes_all)  # mngs.gen.describe(lengthes_all)


def calc_corr(data, event, return_raw=False):
    mngs.general.fix_seeds(42, np=np)
    # data = df[
    #     (df.Event == event)
    #     * (~df.set_size.isna())
    #     * (~df.match.isna())
    # ]

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
    return corr_observed, p_val

    # print(event)
    # if corr_observed < corr_000025:
    #     print("p < 0.0005")
    # elif corr_observed < corr_00025:
    #     print("p < 0.005")
    # elif corr_observed < corr_0025:
    #     print("p < 0.05")
    # elif corr_0975 < corr_observed:
    #     print("p < 0.05")
    # elif corr_09975 < corr_observed:
    #     print("p < 0.005")
    # elif corr_099975 < corr_observed:
    #     print("p < 0.0005")
    # else:
    #     pass

    # if (corr_observed < corr_0025) or (corr_0975 < corr_observed):
    #     print(event)
    #     print(corr_0005.round(3))
    #     print(corr_0025.round(3))
    #     print(corr_0975.round(3))
    #     print(corr_0995.round(3))
    #     print()
    #     print(corr_observed.round(3))


def mk_v_base_str(base_phase_1, base_phase_2):
    if base_phase_1 is None:
        return f"g{base_phase_2[0]}_4 to g{base_phase_2[0]}_8 direction"
    elif base_phase_2 is None:
        return f"g{base_phase_1[0]}_4 to g{base_phase_1[0]}_8 direction"
    else:
        return f"g{base_phase_1[0]} to g{base_phase_2[0]} direction"


def get_speed_vec(rips_df, tgt_bin, base_phase_1, base_phase_2):
    v = np.vstack((rips_df[f"{tgt_bin}"] - rips_df[f"{tgt_bin-1}"]))

    if base_phase_1 is None:
        v_base = np.vstack(rips_df[f"v_{base_phase_2}_4_8"])
    elif base_phase_2 is None:
        v_base = np.vstack(rips_df[f"v_{base_phase_1}_4_8"])
    else:
        v_base = np.vstack(rips_df[f"{base_phase_1}"]) - np.vstack(
            rips_df[f"{base_phase_2}"]
        )
    # v_base = np.vstack(rips_df[f"v_{base_phase}_4_8"])

    v_rebased = np.array(
        ([mngs.linalg.rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))])
    )
    # return v_rebased

    v_rebased_norm = np.abs(v_rebased)
    return v_rebased_norm


if __name__ == "__main__":
    import mngs
    import numpy as np

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

    # koko
    dfs = []
    for is_control in [False, True]:
        events_df = cons_df if is_control else rips_df
        for set_size in [4, 6, 8]:
            for match in [1, 2]:
                """
                set_size = None
                match = None
                """
                _events_df = events_df.copy()

                if set_size is not None:
                    _events_df = _events_df[(_events_df.set_size == set_size)]

                if match is not None:
                    _events_df = _events_df[(_events_df.match == match)]

                for base_phase_1, base_phase_2 in combinations([None] + PHASES, 2):
                    lengthes, event_phases = get_length(
                        _events_df, base_phase_1, base_phase_2
                    )

                    df = pd.DataFrame(
                        {
                            "length": lengthes,
                            "event_phase": event_phases,
                        }
                    )
                    df["is_control"] = is_control
                    df["set_size"] = set_size
                    df["match"] = match
                    df["base_phase_1"] = base_phase_1
                    df["base_phase_2"] = base_phase_2

                    # df["correct"] = _events_df.correct # fixme

                    dfs.append(df.copy())
    df = pd.concat(dfs)
    # df[(~df.set_size.isna())*(~df.match.isna())]

    df["Event"] = df["is_control"].replace({False: "SWR", True: "Control"})
    df["set_size"]
    df["set_size_Event"] = mngs.ml.utils.merge_labels(df["set_size"], df["Event"])

    # for iep, event_phase in enumerate(PHASES):
    #     for ibp, (base_phase_1, base_phase_2) in enumerate(
    #         combinations([None] + PHASES, 2)
    #     ):
    #         fig, ax = plt.subplots(sharex=True, sharey=True)
    #         indi_base_phase_1 = (
    #             df.base_phase_1.isna()
    #             if base_phase_1 is None
    #             else df.base_phase_1 == base_phase_1
    #         )
    #         indi_base_phase_2 = (
    #             df.base_phase_2.isna()
    #             if base_phase_2 is None
    #             else df.base_phase_2 == base_phase_2
    #         )
    #         data = df[
    #             (df.event_phase == event_phase) * indi_base_phase_1 * indi_base_phase_2
    #         ]

    #         sns.ecdfplot(
    #             data=data.reset_index(),
    #             x="length",
    #             hue="set_size_Event",
    #             ax=ax,
    #         )
    #         fig.suptitle(
    #             f"Event phase: {event_phase}\nBase phase: {base_phase_1}-{base_phase_2}"
    #         )
    # plt.show()

    df_stats = pd.DataFrame()
    for event in ["SWR", "Control"]:
        for iep, event_phase in enumerate(PHASES):
            for ibp, (base_phase_1, base_phase_2) in enumerate(
                combinations([None] + PHASES, 2)
            ):
                """
                event_phase = "Encoding"
                base_phase_1 = None
                base_phase_2 = "Encoding"

                base_phase_1 = "Encoding"
                base_phase_2 = "Retrieval"

                """

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
                    # * (df.match == match)
                    * indi_base_phase_1
                    * indi_base_phase_2
                    * (df.Event == event)
                ]
                print(event_phase, base_phase_1, base_phase_2)

                corr, p_val_corr = calc_corr(data, event)

                for ss1, ss2 in [(4, 6), (6, 8), (4, 8)]:
                    d1 = data[data.set_size == ss1]["length"]
                    d2 = data[data.set_size == ss2]["length"]
                    d1 = d1[~d1.isna()]
                    d2 = d2[~d2.isna()]
                    nan_indi = d1.isna() + d2.isna()
                    w, pval, dof, pp = mngs.stats.brunner_munzel_test(d1, d2)

                    # print(pval)
                    # print(pp)
                    # print()

                    df_stats_new = pd.DataFrame(
                        pd.Series(
                            {
                                "Event": event,
                                # "match": "Match IN" if match == 1 else "Mismatch OUT",
                                "Event phase": event_phase,
                                "Base direction": mk_v_base_str(
                                    base_phase_1, base_phase_2
                                ),
                                # "base_phase_1": base_phase_1,
                                # "base_phase_2": base_phase_2,
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

                # pg.homoscedasticity(data=data, dv="length", group="set_size")
                # print(pg.kruskal(data=data, dv="length", between="set_size")["p-unc"].iloc[0].round(3))
                print("\n" + "-" * 40 + "\n")

    mngs.io.save(df_stats.T, "./tmp/figs/box/peri_ripple_traj_length_stats.csv")

    df_E = df[
        (df.event_phase == "Encoding")
        * df.base_phase_1.isna()
        * (df.base_phase_2 == "Encoding")
    ]
    df_E = df_E[["length", "Event", "set_size"]]
    df_E = df_E.sort_values(["Event", "set_size"])

    corr_df = pd.DataFrame()
    df_E_out = {}
    for event in ["SWR", "Control"]:
        data_corr = df_E[(df_E.Event == event)]
        corr_obs, corrs_shuffled = calc_corr(data_corr, event, return_raw=True)
        _corr_df=pd.DataFrame({
            "correlation": np.hstack([corr_obs, corrs_shuffled]),
            "is_shuffle": [False] + [True for _ in range(len(corrs_shuffled))],            
        })
        _corr_df["event"] = event

        corr_df = pd.concat([corr_df, _corr_df])
        for set_size in [4, 6, 8]:
            df_E_out[f"{event}_set_size_{set_size}"] = np.array(
                df_E[(df_E.Event == event) * (df_E.set_size == set_size)]["length"]
            )
    df_E_out = mngs.gen.force_dataframe(df_E_out)

    mngs.io.save(
        df_E_out,
        "./tmp/figs/box/peri_Encoding_ripple_length_based_on_gE4_to_gE8_raw.csv",
    )

    corr_df["hue"] = 0
    dummy_row = pd.DataFrame(
        columns=corr_df.columns,
        data=np.array([np.nan for _ in corr_df.columns]).reshape(1, -1),
    )
    dummy_row["hue"] = 1
    _corr_df = pd.concat([corr_df, dummy_row], axis=0)

    # koko
    fig, ax = plt.subplots()
    sns.violinplot(
        data=_corr_df[_corr_df.is_shuffle != False],
        y="correlation",
        x="event",
        order=["Control", "SWR"],
        hue="hue",
        split=True,
        ax=ax,
        color=mngs.plt.colors.to_RGBA("gray", alpha=1.0),
        )
    
    for event in ["Control", "SWR"]:
        ax.scatter(
            event,
            _corr_df[(_corr_df.is_shuffle == False)*(_corr_df.event == event)]["correlation"],
            # corr_out[f"{event}_corr_observed"].iloc[0],
            color=mngs.plt.colors.to_RGBA("red", alpha=1.0),
            s=100,
        )

    ax.set_ylim(-.15, .15)
    ax.set_xlim(-1, 2)    
    mngs.io.save(
        fig,
        "./tmp/figs/violin/peri_Encoding_ripple_length_correlation_based_on_gE4_to_gE8_fig.tif",
    )

    plt.show()
