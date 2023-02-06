#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-22 13:55:20 (ywatanabe)"

"""
./tmp/figs/time_dependent_dist
./tmp/figs/hist/traj_speed
./tmp/figs/hist/traj_pos
"""

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from elephant.gpfa import GPFA
import quantities as pq
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
def define_phase_time():
    global PHASES, bin_size, width_bins
    # Parameters
    bin_size = 50 * pq.ms
    width_ms = 500
    width_bins = width_ms / bin_size

    # Preparation
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    global starts, ends, colors
    starts, ends = [], []
    PHASE_START_END_DICT = {}
    for i_phase, phase in enumerate(PHASES):
        start_s = int(
            DURS_OF_PHASES[:i_phase].sum() / (bin_size.rescale("s").magnitude)
        )
        end_s = int(
            DURS_OF_PHASES[: (i_phase + 1)].sum() / (bin_size.rescale("s").magnitude)
        )
        PHASE_START_END_DICT[phase] = (start_s, end_s)
        center_s = int((start_s + end_s) / 2)
        start_s = center_s - int(width_bins / 2)
        end_s = center_s + int(width_bins / 2)
        starts.append(start_s)
        ends.append(end_s)

    colors = ["black", "blue", "green", "red"]

    return starts, ends, PHASE_START_END_DICT, colors


def describe(df):
    df = pd.DataFrame(df)
    df = df[~df[0].isna()]
    mm = np.nanmean(df)
    ss = np.nanstd(df) / 3
    return mm, ss
    # med = df.describe().T["50%"].iloc[0]
    # IQR = df.describe().T["75%"].iloc[0] - df.describe().T["25%"].iloc[0]
    # return med, IQR / 3


def get_i_bin(times_sec, bin_size, n_bins):
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    # bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in times_sec]
    return bins


def cosine(v1, v2):
    if np.isnan(v1).any():
        return np.nan
    if np.isnan(v2).any():
        return np.nan
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def nannorm(v):
    if np.isnan(v).any():
        return np.nan
    else:
        return norm(v)


def rebase_a_vec(v, v_base):
    def production_vector(v1, v0):
        """
        production_vector(np.array([3,4]), np.array([10,0])) # np.array([3, 0])
        """
        return norm(v1) * cosine(v1, v0) * v0 / norm(v0)

    if np.isnan(v).any():
        return np.nan
    if np.isnan(v_base).any():
        return np.nan
    v_prod = production_vector(v, v_base)
    sign = np.sign(cosine(v, v_base))
    return sign * norm(v_prod)


def discard_initial_ripples(rips_df, width_s=0.3):
    indi = []
    starts = [0, 1, 3, 6]
    ends = [1, 3, 6, 8]
    for ss, ee in zip(starts, ends):
        indi.append((ss + width_s < rips_df.center_time) * (rips_df.center_time < ee))
    indi = np.vstack(indi).sum(axis=0).astype(bool)
    return rips_df[indi]


def discard_last_ripples(rips_df, width_s=0.3):
    return rips_df[rips_df["center_time"] < 8 - width_s]


def load_rips_df_with_traj(bin_size, is_control=False):
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    n_bins = int((8 / bin_size.rescale("s")).magnitude)

    if not is_control:
        rips_df = utils.load_rips(from_pkl=False, only_correct=False)
    if is_control:
        # cons_df = utils.load_cons(from_pkl=False, only_correct=False)
        cons_df = utils.load_cons_across_trials(from_pkl=False, only_correct=False)
        cons_df["center_time"] = (
            (cons_df["end_time"] + cons_df["start_time"]) / 2
        ).astype(float)
        rips_df = cons_df

    rips_df = discard_initial_ripples(rips_df)
    rips_df = discard_last_ripples(rips_df)

    all_rips = []
    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        for session in ["01", "02"]:
            # Loads
            lpath = (
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            trajs = mngs.io.load(lpath)

            rips_df_session = rips_df[
                (rips_df.subject == subject) * (rips_df.session == session)
            ]

            rips_df_session["traj"] = None
            for i_trial in range(len(trajs)):
                traj = trajs[i_trial, :, :]
                # indi = rips_df_session.index == i_trial + 1
                indi = rips_df_session.trial_number == i_trial + 1
                rip_bins = get_i_bin(
                    rips_df_session[indi].center_time,
                    bin_size,
                    n_bins,
                )

                # rips_df_session.loc[indi, "traj"] = [[traj] for _ in range(indi.sum())]

                rips_df_session.loc[indi, "traj"] = [[traj] for _ in range(indi.sum())]

            rips_df_session = rips_df_session.reset_index()
            all_rips.append(rips_df_session)
    all_rips = pd.concat(all_rips).reset_index()
    return all_rips


def get_vec_from_rips_df(rips_df, col1, col2, col_base_start=None, col_base_end=None):
    v = np.vstack(rips_df[col2]) - np.vstack(rips_df[col1])

    if col_base_start is not None:
        v_base = np.vstack(rips_df[col_base_end]) - np.vstack(rips_df[col_base_start])
        return np.array([rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))])
    else:
        return v


def add_coordinates(rips_df):
    def extract_coordinates_before_or_after_ripple(rips_df, delta_bin=0):
        out = []
        for i_rip, (_, rip) in enumerate(rips_df.iterrows()):
            rip_traj = np.array(rip.traj).squeeze()
            bin_tgt = rip.i_bin + delta_bin
            bin_phase_start, bin_phase_end = PHASE_START_END_DICT[
                rips_df.iloc[i_rip].phase
            ]
            if (bin_phase_start <= bin_tgt) * (bin_tgt < bin_phase_end):
                rip_coord = rip_traj[:, rip.i_bin + delta_bin]
            else:
                rip_coord = np.array([np.nan, np.nan, np.nan])

            out.append(rip_coord)
        return out

    rips_df["Fixation"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[0] : ends[0]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]
    rips_df["Encoding"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[1] : ends[1]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]
    rips_df["Maintenance"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[2] : ends[2]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]
    rips_df["Retrieval"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[3] : ends[3]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]

    n_bins = int((8 / bin_size.rescale("s")).magnitude)
    rips_df["i_bin"] = get_i_bin(rips_df.center_time, bin_size, n_bins)

    # nn = 20
    nn = 80
    for ii in range(nn):
        delta_bin = ii - nn // 2
        rips_df[f"{delta_bin}"] = extract_coordinates_before_or_after_ripple(
            rips_df, delta_bin=delta_bin
        )

    return rips_df


def plot_speed(rips_df, is_control=False, set_size=None, match=None):

    rips_or_cons_str = "cons" if is_control else "rips"

    if set_size is not None:
        rips_df = rips_df[rips_df.set_size == set_size]
    set_size_str = f"_set_size_{set_size}" if set_size is not None else ""

    if match is not None:
        rips_df = rips_df[rips_df.match == match]
    match_str = f"_match_{match}" if match is not None else ""

    n_bases = len(
        [
            None,
            "Fixation",
            "Fixation",
            "Fixation",
            "Encoding",
            "Encoding",
            "Maintenance",
        ]
    )
    # n_bases = len([None, "Fixation", "Encoding", "Maintenance", "Encoding"])
    fig, axes = plt.subplots(
        nrows=n_bases, sharex=True, figsize=(6.4 * 3, 4.8 * 3)
    )  # sharey=True,
    xlim = (-1000, 1000)
    out_df = pd.DataFrame()
    for i_base, (cbs, cbe) in enumerate(
        zip(
            # [None, "Fixation", "Encoding", "Maintenance", "Encoding"],
            # [None, "Encoding", "Maintenance", "Retrieval", "Retrieval"],
            [
                None,
                "Fixation",
                "Fixation",
                "Fixation",
                "Encoding",
                "Encoding",
                "Maintenance",
            ],
            [
                None,
                "Encoding",
                "Maintenance",
                "Retrieval",
                "Maintenance",
                "Retrieval",
                "Retrieval",
            ],
        )
    ):

        dir_txt = f"{cbs}-{cbe}_based"
        ax = axes[i_base]
        i_ax = i_base

        samp_m = mngs.general.listed_dict(PHASES)
        samp_s = mngs.general.listed_dict(PHASES)

        for i_phase, phase in enumerate(PHASES):

            rips_df_phase = rips_df[rips_df.phase == phase]

            centers_ms = []
            for delta_bin in range(-39, 39):
                col1 = f"{delta_bin-1}"
                col1_str = f"{int((delta_bin-1)*bin_size.magnitude)}"
                col2 = f"{delta_bin}"
                col2_str = f"{int((delta_bin)*bin_size.magnitude)}"

                centers_ms.append(int((delta_bin - 0.5) * bin_size.magnitude))

                # gets vectors
                v = np.vstack(rips_df_phase[col2]) - np.vstack(rips_df_phase[col1])
                if cbs is not None:  # basis transformation
                    v_base = np.vstack(rips_df_phase[cbe]) - np.vstack(
                        rips_df_phase[cbs]
                    )
                    v_rebased = [
                        rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))
                    ]
                else:  # just the norm
                    v_rebased = [nannorm(v[ii]) for ii in range(len(v))]

                mm, iqr = describe(v_rebased)

                samp_m[phase].append(mm)
                samp_s[phase].append(iqr)

                # nan_indi = np.isnan(v_rebased)
                # n_samp = (~nan_indi).sum()
                # v_rebased = v_revased[nan_indi]

                # samp_m[phase].append(np.nanmean(v_rebased))
                # samp_s[phase].append(np.nanstd(v_rebased) / 3)

            ax.axhline(y=0, xmin=xlim[0], xmax=xlim[1], linestyle="--", color="gray")

            ax.errorbar(
                x=np.array(centers_ms) + i_phase * 3,
                y=samp_m[phase],
                yerr=samp_s[phase],
                label=phase,
            )
            ax.legend(loc="upper right")

        ax.set_xlim(xlim)

        if i_ax != 0:
            title = dir_txt.replace("_based", " speed").replace("-", " -> ")
            ax.set_ylim(-0.5, 0.5)
        else:
            title = "Speed (norm)"
            ax.set_ylim(-0.3, 5)

        ax.set_title(title)

        samp_m = pd.DataFrame(samp_m)
        samp_m.columns = [f"{col}_{cbs}-{cbe}_med" for col in samp_m.columns]
        samp_s = pd.DataFrame(samp_s)
        samp_s.columns = [f"{col}_{cbs}-{cbe}_s" for col in samp_s.columns]

        # out_dict[f"{cbs}-{cbe}_med"] = samp_m
        # out_dict[f"{cbs}-{cbe}_se"] = samp_s

        out_df = pd.concat([out_df, pd.concat([samp_m, samp_s], axis=1)], axis=1)

    fig.suptitle(f"Set size: {set_size}\nMatch: {match}")
    fig.supylabel("Speed")
    fig.supxlabel("Time from SWR [ms]")
    # plt.show()
    mngs.io.save(
        fig,
        f"./tmp/figs/hist/traj_speed/all_{rips_or_cons_str}{set_size_str}{match_str}.png",
    )
    mngs.io.save(
        out_df,
        f"./tmp/figs/hist/traj_speed/all_{rips_or_cons_str}{set_size_str}{match_str}.csv",
    )
    return fig


def plot_positions(rips_df, is_control=False, set_size=None, match=None):
    rips_or_cons_str = "cons" if is_control else "rips"

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
                col1_str = f"{int((delta_bin)*bin_size.magnitude)}"

                centers_ms.append(int(delta_bin * bin_size.magnitude))

                # gets trajectory
                vv = np.vstack(rips_df_phase[col1])
                gg = np.vstack(rips_df_phase[g_phase])
                dist_gg = [nannorm(vv[ii] - gg[ii]) for ii in range(len(vv))]

                mm, iqr = describe(dist_gg)
                samp_m[rip_phase].append(mm)
                samp_s[rip_phase].append(iqr)

                # samp_m[rip_phase].append(np.nanmean(dist_gg))
                # samp_s[rip_phase].append(np.nanstd(dist_gg) / 3)

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
        f"./tmp/figs/hist/traj_pos/all_{rips_or_cons_str}{set_size_str}{match_str}.png",
    )
    mngs.io.save(
        out_df,
        f"./tmp/figs/hist/traj_pos/all_{rips_or_cons_str}{set_size_str}{match_str}.csv",
    )
    return fig
    # plt.show()


# def plot_pre_post_positions():


#     rips_or_cons_str = "cons" if is_control else "rips"

#     pre_start_ms, pre_end_ms = -1000, -100
#     post_start_ms, post_end_ms = 100, 1000

#     pre_start_bin, pre_end_bin = int(pre_start_ms / bin_size), int(pre_end_ms / bin_size)
#     post_start_bin, post_end_bin = int(post_start_ms / bin_size), int(post_end_ms / bin_size)

#     for rip_phase in PHASES:
#         rips_df_phase = rips_df[rips_df.phase == rip_phase]
#         set_sizes = [4, 6, 8]
#         fig, axes = plt.subplots(ncols=len(set_sizes), sharey=True)
#         for ax, set_size in zip(axes, set_sizes):
#             rips_df_ss = rips_df_phase[rips_df_phase.set_size == set_size]
#             pre_dict = mngs.general.listed_dict()
#             post_dict = mngs.general.listed_dict()

#             for phase in PHASES:
#                 for col in range(pre_start_bin+1, pre_end_bin+1):
#                     vv = np.vstack(rips_df_ss[str(col)])
#                     gg = np.vstack(rips_df_ss[phase])
#                     dist_gg = [nannorm(vv[ii] - gg[ii]) for ii in range(len(vv))]
#                     pre_dict[phase].append(dist_gg)
#                 for col in range(post_start_bin+1, post_end_bin+1):
#                     vv = np.vstack(rips_df_ss[str(col)])
#                     gg = np.vstack(rips_df_ss[phase])
#                     dist_gg = [nannorm(vv[ii] - gg[ii]) for ii in range(len(vv))]
#                     post_dict[phase].append(dist_gg)

#             dfs = []
#             for phase in PHASES:
#                 pre_all = np.hstack(mngs.general.force_dataframe(pre_dict)[phase])
#                 post_all = np.hstack(mngs.general.force_dataframe(post_dict)[phase])
#                 df = pd.DataFrame(
#                     {
#                         f"pre_{phase}": pre_all,
#                         f"post_{phase}": post_all,
#                         })
#                 dfs.append(df)
#             dfs = pd.concat(dfs)

#             sns.boxplot(
#                 data=dfs,
#                 ax=ax,
#                 )
#             fig.suptitle(f"{rip_phase}_{rips_or_cons_str}")
#         mngs.io.save(fig, f"./tmp/figs/ripple_pre_post_dist/{rip_phase}_{rips_or_cons_str}.png")


def plot_pre_post_positions_2(rips_df, cons_df):
    def calc_diff(rips_df, rip_phase, g_phase, match=None):

        rips_df_phase = rips_df[rips_df.phase == rip_phase]

        if match is not None:
            rips_df_phase = rips_df_phase[rips_df_phase.match == match]

        # pre
        pre_dict = mngs.general.listed_dict()
        mid_dict = mngs.general.listed_dict()
        post_dict = mngs.general.listed_dict()
        ss_dict = mngs.general.listed_dict()

        for col_pre, col_mid, col_post in zip(
            range(pre_start_bin + 1, pre_end_bin + 1),
            range(mid_start_bin + 1, mid_end_bin + 1),
            range(post_start_bin + 1, post_end_bin + 1),
        ):
            vv_pre = np.vstack(rips_df_phase[str(col_pre)])
            vv_mid = np.vstack(rips_df_phase[str(col_mid)])
            vv_post = np.vstack(rips_df_phase[str(col_post)])

            gg = np.vstack(rips_df_phase[g_phase])

            dist_gg_pre = [nannorm(vv_pre[ii] - gg[ii]) for ii in range(len(vv_pre))]
            dist_gg_mid = [nannorm(vv_mid[ii] - gg[ii]) for ii in range(len(vv_mid))]
            dist_gg_post = [nannorm(vv_post[ii] - gg[ii]) for ii in range(len(vv_post))]
            ss = np.array(rips_df_phase.set_size)

            pre_dict[g_phase].append(dist_gg_pre)
            mid_dict[g_phase].append(dist_gg_mid)
            post_dict[g_phase].append(dist_gg_post)
            ss_dict[g_phase].append(ss)

        pres = np.hstack(pre_dict[g_phase])
        mids = np.hstack(mid_dict[g_phase])
        posts = np.hstack(post_dict[g_phase])
        ss = np.hstack(ss_dict[g_phase])

        return pres, mids, posts, ss

    pre_start_ms, pre_end_ms = -750, -250
    mid_start_ms, mid_end_ms = -250, 250
    post_start_ms, post_end_ms = 250, 750

    pre_start_bin, pre_end_bin = int(pre_start_ms / bin_size), int(
        pre_end_ms / bin_size
    )
    mid_start_bin, mid_end_bin = int(mid_start_ms / bin_size), int(
        mid_end_ms / bin_size
    )
    post_start_bin, post_end_bin = int(post_start_ms / bin_size), int(
        post_end_ms / bin_size
    )

    """
    rip_phase = "Encoding"
    g_phase = "Retrieval"
    
    rip_phase = "Retrieval"
    g_phase = "Encoding"
    
    rip_phase = "Retrieval"
    g_phase = "Retrieval"    
    """
    for rip_phase in PHASES:
        for g_phase in PHASES:

            for match in [1, 2]:
                pres_rips, mids_rips, posts_rips, set_sizes_rips = calc_diff(
                    rips_df, rip_phase, g_phase, match=match
                )
                rips_str = np.array(
                    ["Pre_Ripple" for _ in range(len(pres_rips))]
                    + ["Mid_Ripple" for _ in range(len(mids_rips))]
                    + ["Post_Ripple" for _ in range(len(posts_rips))]
                )
                pres_cons, mids_cons, posts_cons, set_sizes_cons = calc_diff(
                    cons_df, rip_phase, g_phase, match=match
                )
                cons_str = np.array(
                    ["Pre_Control" for _ in range(len(pres_cons))]
                    + ["Mid_Control" for _ in range(len(mids_cons))]
                    + ["Post_Control" for _ in range(len(posts_cons))]
                )

                data = pd.DataFrame(
                    {
                        "distance": np.concatenate(
                            [
                                pres_rips,
                                mids_rips,
                                posts_rips,
                                pres_cons,
                                mids_cons,
                                posts_cons,
                            ]
                        ),
                        "set_size": np.concatenate(
                            [
                                set_sizes_rips,
                                set_sizes_rips,
                                set_sizes_rips,
                                set_sizes_cons,
                                set_sizes_cons,
                                set_sizes_cons,
                            ]
                        ),
                        "event": np.concatenate([rips_str, cons_str]),
                    }
                )

                fig, ax = plt.subplots()
                sns.boxplot(
                    data=data,
                    y="distance",
                    x="set_size",
                    hue="event",
                    # hue_order=["Pre_Control", "Post_Control", "Pre_Ripple", "Post_Ripple"],
                    hue_order=[
                        "Pre_Control",
                        "Mid_Control",
                        "Post_Control",
                        "Pre_Ripple",
                        "Mid_Ripple",
                        "Post_Ripple",
                    ],
                    # hue_order=["Pre_Ripple", "Post_Ripple"],
                    showfliers=False,
                    ax=ax,
                )
                ax.set_ylim(-0.5, 6.5)
                fig.suptitle(
                    f"Ripple phase: {rip_phase}\ng phase: {g_phase}\nmatch: {match}"
                )

                mngs.io.save(
                    fig,
                    f"./tmp/figs/time_dependent_dist/rip_{rip_phase}_g_{g_phase}_match_{match}.png",
                )

            # plt.show()

        def tmp(width_ms):
            # width_ms = 200
            width_bins = int(width_ms / bin_size.magnitude)
            rips_df["-0"] = rips_df["0"]
            cons_df["-0"] = cons_df["0"]

            from itertools import combinations

            BASE_PHASES = zip(["Encoding", "Encoding", "Encoding"],
                              ["Fixation", "Maintenance", "Retrieval"]
                              )
            for rip_phase in PHASES:
                rips_df_phase = rips_df[rips_df.phase == rip_phase]
                cons_df_phase = cons_df[cons_df.phase == rip_phase]                
                for p1, p2 in BASE_PHASES: #combinations(PHASES, 2):
                    start_bin = int(width_bins/2)
                    vs_based_rip, vs_based_con = [], []
                    for tgt_bin in range(-start_bin, start_bin):

                        v_based_rip = [
                            rebase_a_vec(
                                rips_df_phase.iloc[i_rip][f"{tgt_bin}"]
                                - rips_df_phase.iloc[i_rip][f"{tgt_bin-1}"],
                                rips_df_phase.iloc[i_rip][p1] - rips_df_phase.iloc[i_rip][p2],
                            )
                            for i_rip in range(len(rips_df_phase))
                        ]
                        vs_based_rip.append(np.nansum(v_based_rip))

                        v_based_con = [
                            rebase_a_vec(
                                cons_df_phase.iloc[i_con][f"{tgt_bin}"]
                                - cons_df_phase.iloc[i_con][f"{tgt_bin-1}"],
                                cons_df_phase.iloc[i_con][p1] - cons_df_phase.iloc[i_con][p2],
                            )
                            for i_con in range(len(cons_df_phase))
                        ]
                        vs_based_con.append(np.nansum(v_based_con))

                    # print(describe(vs_based_rip))
                    # print(describe(vs_based_con))

                    non_nan = ~(np.isnan(vs_based_rip) + np.isnan(vs_based_con))
                    stats, pval = brunnermunzel(
                        np.array(vs_based_rip)[non_nan],
                        np.array(vs_based_con)[non_nan],
                    )

                    if pval < 0.05:
                        print(rip_phase, p1, p2, pval)
                

        # tmp(200)
        # tmp(400)
        # tmp(600)
        # tmp(800)
        # tmp(1000)                        


if __name__ == "__main__":
    import mngs
    import numpy as np

    # phase
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    bin_size = 50 * pq.ms
    starts, ends, PHASE_START_END_DICT, colors = define_phase_time()
    matplotlib.use("Agg")

    rips_df = add_coordinates(load_rips_df_with_traj(bin_size, is_control=False))
    cons_df = add_coordinates(load_rips_df_with_traj(bin_size, is_control=True))

    plot_pre_post_positions_2(rips_df, cons_df)

    # rips_df = load_rips_df_with_traj(bin_size, is_control=False)
    # cons_df = load_rips_df_with_traj(bin_size, is_control=True)
    for is_control in [False, True]:
        rips_or_cons_df = cons_df if is_control else rips_df
        # rips_df = add_coordinates(
        #     load_rips_df_with_traj(bin_size, is_control=is_control)
        # )
        # plot_pre_post_positions(rips_df, is_control=is_control)
        # for set_size in [None, 4, 6, 8]:
        for set_size in [None, 4, 6, 8]:
            for match in [None, 1, 2]:
                fig = plot_speed(
                    rips_or_cons_df,
                    is_control=is_control,
                    set_size=set_size,
                    match=match,
                )  # fig 4
                fig = plot_positions(
                    rips_or_cons_df,
                    is_control=is_control,
                    set_size=set_size,
                    match=match,
                )  # fig 5
                plt.close()
                plt.close()

    # plt.show()

    # plt.show()

    # # trajs_all = calc_trajs_all(bin_size)
    # # mngs.io.save(trajs_all, "./tmp/trajs_all.npy")
    # trajs_all = mngs.io.load("./tmp/trajs_all.npy")  # 493
    # trajs_rip = np.vstack(
    #     rips_df.loc[
    #         rips_df[["subject", "session", "trial_number"]].drop_duplicates().index
    #     ]["traj"]
    # )
