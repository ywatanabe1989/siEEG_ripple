#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-28 09:32:02 (ywatanabe)"

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

def to_spiketrains(spike_times_all_trials):
    spike_trains_all_trials = []
    for st_trial in spike_times_all_trials:

        spike_trains_trial = []
        for col, col_df in st_trial.iteritems():

            spike_times = col_df[col_df != ""]
            train = neo.SpikeTrain(list(spike_times) * pq.s, t_start=-6.0, t_stop=2.0)
            spike_trains_trial.append(train)

        spike_trains_all_trials.append(spike_trains_trial)

    return spike_trains_all_trials


def get_i_bin(times_sec, bin_size, n_bins):
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    # bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in times_sec]
    return bins


def cosine(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def production_vector(v1, v0):
    """
    production_vector(np.array([3,4]), np.array([10,0])) # np.array([3, 0])
    """    
    return norm(v1) * cosine(v1, v0) * v0 / norm(v0)

# def calc_cosine(v1, v2):
#     return np.dot(v1, v2) / (norm(v1) * norm(v2))


if __name__ == "__main__":
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    all_rips = []
    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        for session in ["01", "02"]:
            # # parameters
            # subject = "03"
            # session = "01"
            # roi = "AHR"
            bin_size = 200 * pq.ms
            n_bins = int((8 / bin_size.rescale("s")).magnitude)

            # Loads
            LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
            spike_times = mngs.io.load(LPATH)
            spike_trains = to_spiketrains(spike_times)
            rips_df = mngs.io.load(f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl")
            rips_df = rips_df[
                (rips_df.subject == subject) * (rips_df.session == session)
            ]
            # trials_df = mngs.io.load(f"./data/Sub_{subject}/Session_{session}/trials_info.csv")
            rips_df["cos_ER_and_pre_pre"] = np.nan
            rips_df["cos_ER_and_pre"] = np.nan
            rips_df["cos_ER_and_rip"] = np.nan
            rips_df["cos_ER_and_post"] = np.nan
            rips_df["cos_ER_and_post_post"] = np.nan

            # GPFA
            gpfa = GPFA(bin_size=bin_size, x_dim=3)

            # phase
            PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
            DURS_OF_PHASES = np.array(
                mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"]
            )

            global starts, ends, colors
            starts, ends = [], []
            for i_phase in range(len(PHASES)):
                start_s = DURS_OF_PHASES[:i_phase].sum()
                end_s = DURS_OF_PHASES[: (i_phase + 1)].sum()
                starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
                ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
            colors = ["black", "blue", "green", "red"]

            trajs = gpfa.fit_transform(spike_trains)

            for i_trial in range(len(trajs)):
                traj = trajs[i_trial]
                traj_F_med = np.median(traj[:, starts[0] : ends[0]], axis=-1)
                traj_E_med = np.median(traj[:, starts[1] : ends[1]], axis=-1)
                traj_M_med = np.median(traj[:, starts[2] : ends[2]], axis=-1)
                traj_R_med = np.median(traj[:, starts[3] : ends[3]], axis=-1)

                rip_bins = get_i_bin(
                    rips_df[rips_df.index == i_trial + 1].center_time, bin_size, n_bins
                )
                cos_ER_and_pre_pre = []
                cos_ER_and_pre = []
                cos_ER_and_rip = []
                cos_ER_and_post = []
                cos_ER_and_post_post = []
                for rip_bin in rip_bins:
                    try:
                        control = False
                        if control:
                            for ii in range(len(starts)):
                                if (starts[ii] <= rip_bin) and (rip_bin < ends[ii]):
                                    rip_bin = random.randint(starts[ii], ends[ii])
                        # rip_bin = random.randint(0+7, n_bins-7)
                        # traj.shape # (3,40)
                        traj_pre_pre = traj[:, rip_bin - 7 : rip_bin - 4]
                        traj_pre = traj[:, rip_bin - 4 : rip_bin - 1]
                        traj_rip = traj[:, rip_bin - 1 : rip_bin + 2]
                        traj_post = traj[:, rip_bin + 2 : rip_bin + 5]
                        traj_post_post = traj[:, rip_bin + 5 : rip_bin + 8]

                        _cos_ER_and_pre_pre = calc_cosine(
                            traj_R_med - traj_E_med,
                            traj_pre_pre[:, -1] - traj_pre_pre[:, 0],
                        )

                        _cos_ER_and_pre = calc_cosine(
                            traj_R_med - traj_E_med,
                            traj_pre[:, -1] - traj_pre[:, 0],
                        )

                        _cos_ER_and_rip = calc_cosine(
                            traj_R_med - traj_E_med,
                            traj_rip[:, -1] - traj_rip[:, 0],
                        )

                        _cos_ER_and_post = calc_cosine(
                            traj_R_med - traj_E_med,
                            traj_post[:, -1] - traj_post[:, 0],
                        )

                        _cos_ER_and_post_post = calc_cosine(
                            traj_R_med - traj_E_med,
                            traj_post_post[:, -1] - traj_post_post[:, 0],
                        )

                        cos_ER_and_pre_pre.append(_cos_ER_and_pre_pre)
                        cos_ER_and_pre.append(_cos_ER_and_pre)
                        cos_ER_and_rip.append(_cos_ER_and_rip)
                        cos_ER_and_post.append(_cos_ER_and_post)
                        cos_ER_and_post_post.append(_cos_ER_and_post_post)
                    except Exception as e:
                        print(e)
                        cos_ER_and_pre_pre.append(np.nan)
                        cos_ER_and_pre.append(np.nan)
                        cos_ER_and_rip.append(np.nan)
                        cos_ER_and_post.append(np.nan)
                        cos_ER_and_post_post.append(np.nan)

                rips_df.loc[
                    (rips_df.index == i_trial + 1), "cos_ER_and_pre_pre"
                ] = cos_ER_and_pre_pre
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "cos_ER_and_pre"
                ] = cos_ER_and_pre
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "cos_ER_and_rip"
                ] = cos_ER_and_rip
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "cos_ER_and_post"
                ] = cos_ER_and_post
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "cos_ER_and_post_post"
                ] = cos_ER_and_post_post

            rips_df = rips_df.reset_index()
            all_rips.append(rips_df)
    all_rips = pd.concat(all_rips).reset_index()

    for x in [
        "cos_ER_and_pre_pre",
        "cos_ER_and_pre",
        "cos_ER_and_rip",
        "cos_ER_and_post",
        "cos_ER_and_post_post",
    ]:
        fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
        for (ax, phase) in zip(axes, PHASES):
            sns.histplot(
                # data = rips_df[rips_df.phase == phase],
                data=all_rips[all_rips.phase == phase],
                x=x,
                hue="phase",
                common_norm=False,
                stat="probability",
                # hue_order=PHASES,
                kde=True,
                ax=ax,
            )
            ax.set_ylim(0, 0.3)
            ax.set_title(f"{phase}")
            fig.suptitle(x)
    plt.show()
