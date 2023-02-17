#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-11 15:59:45 (ywatanabe)"

"""
./tmp/figs/time_dependent_dist
./tmp/figs/hist/traj_speed
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
def plot_speed(rips_df, is_control=False, set_size=None, match=None):

    events_str = "cons" if is_control else "rips"

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
            # "vec_4_8_Fixation",
            # "vec_4_8_Encoding",
            # "vec_4_8_Maintenance",
            # "vec_4_8_Retrieval",
        ]
    )
    # n_bases = len([None, "Fixation", "Encoding", "Maintenance", "Encoding"])
    fig, axes = plt.subplots(
        ncols=n_bases, sharex=True, sharey=True, figsize=(6.4 * 3, 4.8 * 3)
    )  # sharey=True,
    xlim = (-500, 500)
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
                # "vec_4_8_Fixation",
                # "vec_4_8_Encoding",
                # "vec_4_8_Maintenance",
                # "vec_4_8_Retrieval",
            ],
            [
                None,
                "Encoding",
                "Maintenance",
                "Retrieval",
                "Maintenance",
                "Retrieval",
                "Retrieval",
                # "vec_4_8_Fixation",
                # "vec_4_8_Encoding",
                # "vec_4_8_Maintenance",
                # "vec_4_8_Retrieval",
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
                if cbs in PHASES:  # basis transformation
                    v_base = np.vstack(rips_df_phase[cbe]) - np.vstack(
                        rips_df_phase[cbs]
                    )
                    v_rebased = [
                        mngs.linalg.rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))
                    ]
                    # v_rebased = np.log10(np.abs(v_rebased)) # fixme
                    v_rebased = np.abs(v_rebased) # fixme                    

                if cbs is None:  # just the norm
                    v_rebased = [mngs.linalg.nannorm(v[ii]) for ii in range(len(v))]
                    v_rebased = np.abs(v_rebased) # fixme                    

                elif "vec_4_8" in cbs:
                    v_base = rips_df_phase[cbs]
                    v_rebased = [
                        mngs.linalg.rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))
                    ]
                    v_rebased = np.abs(v_rebased) # fixme

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

        # ylim = (-1.25, 0.25) # log10
        ylim = (0, 3)
        if i_ax != 0:
            title = dir_txt.replace("_based", "").replace("-", " -> ")
            ax.set_ylim(ylim)
            # ax.set_ylim(-0.3, 1)
        else:
            title = "Speed (norm)"
            ax.set_ylim(ylim)            
            # ax.set_ylim(-0.3, 2)

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
        f"./tmp/figs/hist/traj_speed/{match_str}/all_{events_str}{set_size_str}.png",
    )
    mngs.io.save(
        out_df,
        f"./tmp/figs/hist/traj_speed/{match_str}/all_{events_str}{set_size_str}.csv",
    )
    return fig


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
                col1_str = f"{int((delta_bin)*bin_size.magnitude)}"

                centers_ms.append(int(delta_bin * bin_size.magnitude))

                # gets trajectory
                vv = np.vstack(rips_df_phase[col1])
                gg = np.vstack(rips_df_phase[g_phase])
                dist_gg = [mngs.linalg.nannorm(vv[ii] - gg[ii]) for ii in range(len(vv))]

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
        f"./tmp/figs/hist/traj_pos/all_{events_str}{set_size_str}{match_str}.png",
    )
    mngs.io.save(
        out_df,
        f"./tmp/figs/hist/traj_pos/all_{events_str}{set_size_str}{match_str}.csv",
    )
    return fig
    # plt.show()


# def add_vec_4_8(rips_df):
#     def _in_add_vec_4_8(rips_df, subject, session, roi):
#         indi = (
#             (rips_df.subject == subject)
#             * (rips_df.session == session)
#             * (rips_df.ROI == roi)
#         )
#         print(indi.sum())

#         trajs = mngs.io.load(f"./data/Sub_{subject}/Session_{session}/traj_{roi}.npy")
#         trials_info = mngs.io.load(
#             f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
#         )

#         trajs_4 = trajs[trials_info.set_size == 4]
#         trajs_8 = trajs[trials_info.set_size == 8]

#         eight_four = np.median(trajs_8, axis=0) - np.median(trajs_4, axis=0)

#         for i_phase, phase in enumerate(PHASES):
#             data = pd.Series(
#                 [
#                     np.nanmedian(
#                         eight_four[:, starts[i_phase] : ends[i_phase]], axis=-1
#                     )
#                     for _ in range(indi.sum())
#                 ]
#             ).copy()
#             import ipdb

#             ipdb.set_trace()
#             rips_df.loc[indi, f"vec_4_8_{phase}"] = data  # NaN

#         return rips_df

#     for phase in PHASES:
#         rips_df[f"vec_4_8_{phase}"] = None

#     ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
#     for subject, roi in ROIs.items():
#         subject = f"{int(subject):02d}"
#         for session in ["01", "02"]:
#             rips_df = _in_add_vec_4_8(rips_df, subject, session, roi).copy()

#     return rips_df


def compair_speed_of_rips_and_cons(rips_df, cons_df):
    width_ms = 500
    width_bins = width_ms / bin_size.magnitude
    start = -int(width_bins / 2)
    end = int(width_bins / 2)

    speeds_rips, speeds_cons = [], []
    for i_bin in range(start, end):
        v_rips = np.vstack(rips_df[f"{i_bin}"]) - np.vstack(rips_df[f"{i_bin-1}"])
        v_cons = np.vstack(cons_df[f"{i_bin}"]) - np.vstack(cons_df[f"{i_bin-1}"])

        speed_rips = [mngs.linalg.nannorm(v_rips[ii]) for ii in range(len(v_rips))]
        speed_cons = [mngs.linalg.nannorm(v_cons[ii]) for ii in range(len(v_cons))]

        speeds_rips.append(
            speed_rips
        )
        speeds_cons.append(
            speed_cons
        )
    speeds_rips = np.vstack(speeds_rips)
    speeds_cons = np.vstack(speeds_cons)

    df = {}
    for phase in PHASES:
        speeds_rips_phase = np.hstack(speeds_rips[:,rips_df.phase == phase])
        speeds_cons_phase = np.hstack(speeds_cons[:,cons_df.phase == phase])
        nan_indi = np.isnan(speeds_rips_phase) + np.isnan(speeds_cons_phase)
        df[f"SWR_{phase}"] = speeds_rips_phase
        df[f"Control_{phase}"] = speeds_cons_phase        
        speeds_rips_phase = speeds_rips_phase[~nan_indi]
        speeds_cons_phase = speeds_cons_phase[~nan_indi]
        speeds_rips_phase.sum()
        speeds_cons_phase.sum()
        starts, pval = brunnermunzel(
            speeds_rips_phase,
            speeds_cons_phase,
        )
        print(phase, pval)

    mngs.io.save(mngs.general.force_dataframe(df), "./tmp/figs/box/speed_comparison_rips_and_cons.csv")
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
    
    # # phase
    # PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    # DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    # bin_size = 50 * pq.ms
    # starts, ends, PHASE_START_END_DICT, colors = define_phase_time()
    # matplotlib.use("Agg")

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )
    
    # rips_df = add_coordinates(load_rips_df_with_traj(bin_size, is_control=False))
    # cons_df = add_coordinates(load_rips_df_with_traj(bin_size, is_control=True))

    df = compair_speed_of_rips_and_cons(rips_df, cons_df)
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

    # rips_df = load_rips_df_with_traj(bin_size, is_control=False)
    # cons_df = load_rips_df_with_traj(bin_size, is_control=True)
    for is_control in [False, True]:
        events_df = cons_df if is_control else rips_df
        # rips_df = add_coordinates(
        #     load_rips_df_with_traj(bin_size, is_control=is_control)
        # )
        # plot_pre_post_positions(rips_df, is_control=is_control)
        # for set_size in [None, 4, 6, 8]:
        for set_size in [None, 4, 6, 8]:
            for match in [None, 1, 2]:
                """
                set_size = None
                match = None                
                """                
                fig = plot_speed(
                    events_df,
                    is_control=is_control,
                    set_size=set_size,
                    match=match,
                )  # fig 4
                fig = plot_positions(
                    events_df,
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
