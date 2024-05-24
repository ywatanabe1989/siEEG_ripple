#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-09 15:36:40 (ywatanabe)"

import matplotlib
import mngs
import neo
import numpy as np
import pandas as pd
import quantities as pq
from elephant.gpfa import GPFA
from numpy.linalg import norm

matplotlib.use("TkAgg")
import random
import sys
import warnings
from itertools import combinations, product

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import brunnermunzel

sys.path.append("./siEEG_ripple/")
import re
from glob import glob

import utils
from natsort import natsorted

# Functions
# def describe(df):
#     df = pd.DataFrame(df)
#     med = df.describe().T["50%"].iloc[0]
#     IQR = df.describe().T["75%"].iloc[0] - df.describe().T["25%"].iloc[0]
#     # print(med, IQR)
#     return med, IQR


# def to_spiketrains(spike_times_all_trials):
#     spike_trains_all_trials = []
#     for st_trial in spike_times_all_trials:

#         spike_trains_trial = []
#         for col, col_df in st_trial.iteritems():

#             spike_times = col_df[col_df != ""]
#             train = neo.SpikeTrain(list(spike_times) * pq.s, t_start=-6.0, t_stop=2.0)
#             spike_trains_trial.append(train)

#         spike_trains_all_trials.append(spike_trains_trial)

#     return spike_trains_all_trials


# def define_phase_time():
#     global PHASES, bin_size, width_bins
#     # Parameters
#     bin_size = 50 * pq.ms
#     width_ms = 500
#     width_bins = width_ms / bin_size

#     # Preparation
#     PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
#     DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
#     global starts, ends, colors
#     starts, ends = [], []
#     for i_phase in range(len(PHASES)):
#         start_s = int(
#             DURS_OF_PHASES[:i_phase].sum() / (bin_size.rescale("s").magnitude)
#         )
#         end_s = int(
#             DURS_OF_PHASES[: (i_phase + 1)].sum() / (bin_size.rescale("s").magnitude)
#         )

#         center_s = int((start_s + end_s) / 2)
#         start_s = center_s - int(width_bins / 2)
#         end_s = center_s + int(width_bins / 2)
#         starts.append(start_s)
#         ends.append(end_s)
#     colors = ["black", "blue", "green", "red"]

#     return PHASES, starts, ends, colors


def get_subject_session_roi(lpath):
    subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
    session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
    roi = (
        re.findall("spike_times_[\w]{2,3}.pkl", lpath)[0]
        .split("spike_times_")[1]
        .split(".pkl")[0]
    )
    return subject, session, roi


def calc_g_dist(match, set_size):

    # Loads
    LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times_*.pkl"))

    dfs = []
    info_df = pd.DataFrame()

    for lpath in LPATHs:
        subject, session, roi = get_subject_session_roi(lpath)

        try:
            if int(session) <= 2:
                trajs = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
                )
                trials_df = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
                )

                # drops unnecessary rows
                # by match
                if match is not None:
                    trajs = trajs[trials_df.match == match]
                    trials_df = trials_df[trials_df.match == match]

                # by correct
                if False:
                    trajs = trajs[trials_df.correct == 1]
                    trials_df = trials_df[trials_df.correct == 1]

                # by set_size
                if set_size is not None:
                    trajs = trajs[trials_df.set_size == set_size]
                    trials_df = trials_df[trials_df.set_size == set_size]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    g_phases = {}
                    for phase in PHASES:
                        g_phases[phase] = np.nanmedian(
                            trajs[
                                :, :, GS_BINS_DICT[phase][0] : GS_BINS_DICT[phase][1]
                            ],
                            axis=-1,
                        )

                dfs.append(g_phases)

                info_df_new = pd.DataFrame(
                    pd.Series(
                        {
                            "subject": subject,
                            "session": session,
                            "ROI": roi,
                            "set_size": set_size,
                        }
                    )
                )
                info_df_new = pd.concat(
                    [info_df_new for _ in range(len(g_phases["Fixation"]))], axis=1
                )
                info_df = pd.concat(
                    [
                        info_df,
                        info_df_new,
                    ],
                    axis=1,
                )
        except Exception as e:
            # print(e)
            pass

    return dfs, info_df


def get_indi_MTL(info_df, MTL_region):

    if MTL_region == "Hipp.":
        return mngs.general.search(
            ["AHL", "AHR", "PHL", "PHR"], info_df.T.ROI, as_bool=True
        )[0]

    if MTL_region == "EC":
        return mngs.general.search(["ECL", "ECR"], info_df.T.ROI, as_bool=True)[0]

    if MTL_region == "Amy.":
        return mngs.general.search(["AL", "AR"], info_df.T.ROI, as_bool=True)[0]


def extract_g_phases(dfs, info_df, MTL_region):
    g_phases = pd.concat(
        [mngs.general.force_dataframe(dfs[ii]) for ii in range(len(dfs))]
    )[get_indi_MTL(info_df, MTL_region)]
    nan_indi = (
        np.stack([np.isnan(np.vstack(g_phases[phase])) for phase in PHASES])
        .any(axis=0)
        .any(axis=-1)
    )
    g_phases = g_phases[~nan_indi]
    return g_phases


def calc_dist_between_gs(g_phases):
    # raw data for hist plot
    data = {}
    for p1, p2 in combinations(PHASES, 2):
        data[f"{p1[0]}{p2[0]}"] = norm(
            np.stack(g_phases[p1] - g_phases[p2]).astype(float),
            axis=-1,
        )
    out_df = mngs.general.force_dataframe(data)
    
    return out_df


def test_bm(g_phases):
    # stats
    count = 0
    pval = 1
    is_significant = False
    for ii, (p1, p2) in enumerate(combinations(PHASES, 2)):
        comp_1_str = f"{p1[0]}-{p2[0]}"
        for jj, (p3, p4) in enumerate(combinations(PHASES, 2)):
            comp_2_str = f"{p3[0]}-{p4[0]}"
            if ii < jj:
                count += 1
                stats, pval = brunnermunzel(
                    norm(
                        np.stack(g_phases[p1] - g_phases[p2]).astype(float),
                        axis=-1,
                    ),
                    norm(
                        np.stack(g_phases[p3] - g_phases[p4]).astype(float),
                        axis=-1,
                    ),
                )
                pval *= 15
                if pval < 0.05:
                    # if True:
                    print(
                        f"{p1[0]}-{p2[0]}, {p3[0]}-{p4[0]}",
                        pval.round(3),
                    )
                    is_significant = True
    if is_significant:
        # import ipdb; ipdb.set_trace()
        pass

def plot_box(dists_between_gs, MTL_region, set_size, match):
    fig, ax = plt.subplots()
    for i_col, col in enumerate(dists_between_gs.columns):
        ax.boxplot(dists_between_gs[col],
                positions=[i_col],
                   showfliers=False,
                )
    title = f"{MTL_region}; set_size: {set_size}; match: {match}"
    ax.set_title(title)
    ax.set_xticklabels(dists_between_gs.columns)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 2)
    return fig
    


if __name__ == "__main__":

    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    for match in [None, 1, 2]:
        for set_size in [None, 4, 6, 8]:
            """
            match = 1
            set_size = None
            """

            dfs, info_df = calc_g_dist(match, set_size)

            for MTL_region in ["Hipp.", "EC", "Amy."]:
                """
                MTL_region = "Hipp."
                """
                g_phases = extract_g_phases(dfs, info_df, MTL_region)
                dists_between_gs = calc_dist_between_gs(g_phases)

                fig = plot_box(dists_between_gs, MTL_region, set_size, match)
                spath = f"./tmp/figs/box/dists_between_gs/{MTL_region}/{set_size}_match_{match}.png"
                mngs.io.save(fig, spath)
                # plt.show()

                spath = f"./tmp/figs/box/dists_between_gs/{MTL_region}/{set_size}_match_{match}.csv"
                mngs.io.save(
                    dists_between_gs,
                    spath,
                )

                test_bm(g_phases)


# """
# Saved to: ./tmp/g_dist_z_by_trial/Hipp._None_match_None.csv

# F-M, F-R 0.002
# F-R, E-M 0.0
# E-M, E-R 0.001

# Saved to: ./tmp/g_dist_z_by_trial/Amy._None_match_None.csv

# F-R, E-M 0.001

# Saved to: ./tmp/g_dist_z_by_trial/Hipp._4_match_None.csv

# E-M, E-R 0.001

# Saved to: ./tmp/g_dist_z_by_trial/Hipp._None_match_2.csv

# F-M, F-R 0.003
# F-R, E-M 0.0
# """
# z_by = "by_session"
# MTL_region = "Hipp."
# match = 1
# set_size = 4
# for set_size in [4, 6, 8]:
#     data = mngs.io.load(
#         f"./tmp/g_dist_z_{z_by}/{MTL_region}_{set_size}_match_{match}_raw_dists.csv"
#     )
#     print(describe(np.log10(data["ER"]), "median"))
