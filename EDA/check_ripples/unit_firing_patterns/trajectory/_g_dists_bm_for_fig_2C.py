#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-25 15:55:53 (ywatanabe)"

import mngs
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from itertools import combinations
from scipy.linalg import norm
import pandas as pd
from pprint import pprint
import random
from natsort import natsorted
from glob import glob
import re
import warnings


def get_subject_session_roi(lpath):
    subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
    session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
    roi = (
        re.findall("traj_z_by_session_[\w]{2,3}.npy", lpath)[0]
        .split("traj_z_by_session_")[1]
        .split(".npy")[0]
    )
    return subject, session, roi


def define_phase_time():
    import quantities as pq

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
    for i_phase in range(len(PHASES)):
        start_s = int(
            DURS_OF_PHASES[:i_phase].sum() / (bin_size.rescale("s").magnitude)
        )
        end_s = int(
            DURS_OF_PHASES[: (i_phase + 1)].sum() / (bin_size.rescale("s").magnitude)
        )

        center_s = int((start_s + end_s) / 2)
        start_s = center_s - int(width_bins / 2)
        end_s = center_s + int(width_bins / 2)
        starts.append(start_s)
        ends.append(end_s)
    colors = ["black", "blue", "green", "red"]

    return PHASES, starts, ends, colors


def calc_g_phases():
    pass
    # PHASES, starts, ends, colors = define_phase_time()

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", RuntimeWarning)
    #     g_phases = {
    #         "Fixation": np.nanmedian(trajs[:, :, starts[0] : ends[0]], axis=-1),
    #         "Encoding": np.nanmedian(trajs[:, :, starts[1] : ends[1]], axis=-1),
    #         "Maintenance": np.nanmedian(
    #             trajs[:, :, starts[2] : ends[2]], axis=-1
    #         ),
    #         "Retrieval": np.nanmedian(trajs[:, :, starts[3] : ends[3]], axis=-1),
    #     }

    # dfs_g_phases.append(g_phases)

    # g_phases = pd.concat(
    #     [
    #         mngs.general.force_dataframe(dfs[ii])
    #         for ii in range(len(dfs))
    #     ]
    # )
    # nan_indi = (
    #     np.stack(
    #         [np.isnan(np.vstack(g_phases[phase])) for phase in PHASES]
    #     )
    #     .any(axis=0)
    #     .any(axis=-1)
    # )
    # g_phases = g_phases[~nan_indi]
    # info_df = info_df.T[~nan_indi]
    # trajs_all = np.vstack(dfs_trajs)[~nan_indi]


def load_trajs(hipp_ec_or_amy, match, set_size):

    # Loads
    LPATHs = natsorted(glob("./data/Sub_*/Session_*/traj_z_by_session_*.npy"))

    dfs_trajs = []
    dfs_g_phases = []
    info_df = pd.DataFrame()

    for lpath in LPATHs:
        subject, session, roi = get_subject_session_roi(lpath)

        try:
            if int(session) <= 2:
                trajs = mngs.io.load(lpath)
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

                dfs_trajs.append(trajs)

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
                    [info_df_new for _ in range(len(trajs))], axis=1
                )
                info_df = pd.concat(
                    [
                        info_df,
                        info_df_new,
                    ],
                    axis=1,
                )
        except Exception as e:
            print(e)

    trajs_all = np.vstack(dfs_trajs)
    info_df = info_df.T

    roi_keys = {
        "Hipp.": ["AHL", "AHR", "PHL", "PHR"],
        "EC": ["ECL", "ECR"],
        "Amy.": ["AL", "AR"],
    }[hipp_ec_or_amy]

    indi_roi = mngs.general.search(
        roi_keys, info_df.ROI, as_bool=True
    )[0]

    trajs_all = trajs_all[indi_roi]
    info_df = info_df[indi_roi]

    return trajs_all, info_df


trajs_hipp, info_df_hipp = load_trajs("Hipp.", None, None)
trajs_ec, info_df_ec = load_trajs("EC", None, None)
trajs_amy, info_df_amy = load_trajs("Amy.", None, None)

med_traj_hipp = np.nanmedian(trajs_hipp, axis=0)
med_traj_ec = np.nanmedian(trajs_ec, axis=0)
med_traj_amy = np.nanmedian(trajs_amy, axis=0)

# import scipy
# sigma = 0.001
# med_traj_hipp = np.array([scipy.ndimage.gaussian_filter1d(med_traj_hipp[ii], sigma) for ii in range(len(med_traj_hipp))])
# med_traj_ec = np.array([scipy.ndimage.gaussian_filter1d(med_traj_ec[ii], sigma) for ii in range(len(med_traj_ec))])
# med_traj_amy = np.array([scipy.ndimage.gaussian_filter1d(med_traj_amy[ii], sigma) for ii in range(len(med_traj_amy))])

n_dims = med_traj_hipp.shape[0]
fig, axes = plt.subplots(nrows=n_dims, sharex=True, sharey=True)
starts = [0, 20, 60, 120]
ends = [20, 60, 120, 160]
colors = ["gray", "blue", "green", "red"]
linestyles = ["solid", "dotted", "dashed"]

for i_ax, (ax, traj_str, traj) in enumerate(
    zip(axes, ["Hipp.", "EC", "Amy."], [med_traj_hipp, med_traj_ec, med_traj_amy])
):
    for i_dim in range(n_dims):
        for ss, ee, cc in zip(starts, ends, colors):
            ax.plot(
                np.arange(ss, ee),
                traj[i_dim][ss:ee],
                label=f"Factor {i_dim+1}",
                # color=colors[i_dim],
                color=cc,
                linestyle=linestyles[i_dim],
            )
    ax.set_ylabel(traj_str)
    # ax.legend(loc="upper right")
plt.show()

def get_g_dists(trajs_roi):
    PHASES, starts, ends, colors = define_phase_time()
    gs_f = np.nanmedian(trajs_roi[:,:,starts[0]:ends[0]], axis=-1) 
    gs_e = np.nanmedian(trajs_roi[:,:,starts[1]:ends[1]], axis=-1)
    gs_m = np.nanmedian(trajs_roi[:,:,starts[2]:ends[2]], axis=-1)
    gs_r = np.nanmedian(trajs_roi[:,:,starts[3]:ends[3]], axis=-1)
    
    nan_indi = np.isnan(gs_f).any(axis=-1) \
        + np.isnan(gs_e).any(axis=-1) \
        + np.isnan(gs_m).any(axis=-1) \
        + np.isnan(gs_r).any(axis=-1)

    dists = []
    from itertools import product, combinations
    for gs_1, gs_2 in combinations([gs_f, gs_e, gs_m, gs_r], 2):
        dists.append(norm(gs_1[~nan_indi] - gs_2[~nan_indi], axis=-1))

    return np.hstack(dists)

dists_hipp = get_g_dists(trajs_hipp)
dists_ec = get_g_dists(trajs_ec)
dists_amy = get_g_dists(trajs_amy)

from scipy.stats import brunnermunzel
print(brunnermunzel(dists_hipp, dists_ec))
print(brunnermunzel(dists_ec, dists_amy))
print(brunnermunzel(dists_amy, dists_hipp))

dists_df = mngs.general.force_dataframe({"Hipp.":dists_hipp,
                                       "EC": dists_ec,
                                       "Amy.": dists_amy})
mngs.io.save(dists_df, "./tmp/figs/heatmap/g_dists.csv")
fig = mngs.plt.mk_colorbar()
mngs.io.save(fig, "./tmp/figs/heatmap/colorbar.tif")




# gs_hipp = {pp:np.median(med_traj_hipp[:,ss:ee], axis=-1) for pp, ss, ee in zip(PHASES, starts, ends)}
# gs_ec = {pp:np.median(med_traj_ec[:,ss:ee], axis=-1) for pp, ss, ee in zip(PHASES, starts, ends)}
# gs_amy = {pp:np.median(med_traj_amy[:,ss:ee], axis=-1) for pp, ss, ee in zip(PHASES, starts, ends)}

# from itertools import product
# from scipy.linalg import norm
# for gs in [gs_hipp, gs_ec, gs_amy]:
#     for items1, items2 in product(gs.items(), gs.items()):
#         p1, g1 = items1
#         p2, g2 = items2
#         print(norm(g1-g2))
#         print()
#     import ipdb; ipdb.set_trace()
# ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
# subject = "06"  # f"{int(list(ROIs.keys())[2]):02d}"  # 04
# session = "02"  # 01
# # subject = "06"
# # session = "02"
# # i_trial = 5 + 1
# i_trial = 5 + 1  # random.randint(0,49)

# traj_AHL = mngs.io.load(
#     f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_AHL.npy"
# )
# traj_ECL = mngs.io.load(
#     f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_ECL.npy"
# )
# traj_AL = mngs.io.load(
#     f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_AL.npy"
# )

# # plt.plot(traj_AHL[5].T)
# # plt.show()

# trajs = dict(
#     AHL=traj_AHL,
#     ECL=traj_ECL,
#     AL=traj_AL,
# )

# # for traj_str, traj in trajs.items():
# #     trajs[traj_str] = trajs[traj_str][i_trial]
# #     # trajs[traj_str] = np.array(
# #     #     [
# #     #         np.nanmedian(traj, axis=0)[i_dim]
# #     #         for i_dim in range(3)]
# #     # )

# # scipy.ndimage.gaussian_filter1d(    , sigma=3)
# PHASES = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
# starts = [0, 20, 60, 120]
# ends = [20, 60, 120, 160]
# centers = ((np.array(starts) + np.array(ends)) / 2).astype(int)
# width = 10
# colors = ["gray", "blue", "green", "red"]
# linestyles = ["solid", "dotted", "dashed"]

# fig, axes = plt.subplots(nrows=len(trajs), sharex=True, sharey=True)
# for i_ax, (ax, (traj_str, traj)) in enumerate(zip(axes, trajs.items())):
#     for i_dim in range(traj.shape[1]):
#         for ss, ee, cc in zip(starts, ends, colors):
#             ax.plot(
#                 np.arange(ss, ee),
#                 np.median(traj, axis=0)[i_dim][ss:ee],
#                 label=f"Factor {i_dim+1}",
#                 # color=colors[i_dim],
#                 color=cc,
#                 linestyle=linestyles[i_dim],
#             )
#     ax.set_ylabel(traj_str)
#     # ax.legend(loc="upper right")
# plt.show()


# gs = mngs.general.listed_dict()
# for traj_str, traj in trajs.items():
#     for cc in centers:
#         g = np.median(traj[:, int(cc - width / 2) : int(cc + width / 2)], axis=-1)
#         gs[traj_str].append(g)


# dfs = {}
# for roi in ["AHL", "ECL", "AL"]:
#     df = pd.DataFrame(columns=PHASES, index=PHASES)
#     for ii, jj in combinations(np.arange(4), 2):
#         df.iloc[ii, jj] = norm(gs[roi][ii] - gs[roi][jj])
#     dfs[roi] = df


# pprint(dfs)
