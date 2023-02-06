#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-31 12:22:49 (ywatanabe)"

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

def calc_ER_based_vec(v, v_ER):
    v_prod = production_vector(v, v_ER)
    sign = np.sign(cosine(v, v_ER))
    return sign * norm(v_prod)

def collect_ER_based_vec():
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

            # min max norm
            trajs = np.stack(trajs, axis=0)            
            trajs_min = trajs.min(axis=0, keepdims=True).min(axis=-1, keepdims=True)
            trajs -= trajs_min
            trajs_max = trajs.max(axis=0, keepdims=True).max(axis=-1, keepdims=True)
            trajs /= trajs_max

            for i_trial in range(len(trajs)):
                traj = trajs[i_trial]
                traj_F_med = np.median(traj[:, starts[0] : ends[0]], axis=-1)
                traj_E_med = np.median(traj[:, starts[1] : ends[1]], axis=-1)
                traj_M_med = np.median(traj[:, starts[2] : ends[2]], axis=-1)
                traj_R_med = np.median(traj[:, starts[3] : ends[3]], axis=-1)

                rip_bins = get_i_bin(
                    rips_df[rips_df.index == i_trial + 1].center_time, bin_size, n_bins
                )
                
                ER_based_vec_pre_pre = []
                ER_based_vec_pre = []
                ER_based_vec_rip = []
                ER_based_vec_post = []
                ER_based_vec_post_post = []
                for rip_bin in rip_bins:
                    try:
                        control = False
                        if control:
                            for ii in range(len(starts)):
                                if (starts[ii] <= rip_bin) and (rip_bin < ends[ii]):
                                    rip_bin = random.randint(starts[ii], ends[ii])
                        traj_pre_pre = traj[:, rip_bin - 7 : rip_bin - 4]
                        traj_pre = traj[:, rip_bin - 4 : rip_bin - 1]
                        traj_rip = traj[:, rip_bin - 1 : rip_bin + 2]
                        traj_post = traj[:, rip_bin + 2 : rip_bin + 5]
                        traj_post_post = traj[:, rip_bin + 5 : rip_bin + 8]

                        _ER_based_vec_pre_pre = calc_ER_based_vec(
                            traj_pre_pre[:, -1] - traj_pre_pre[:, 0],
                            traj_R_med - traj_E_med,                            
                        )

                        _ER_based_vec_pre = calc_ER_based_vec(
                            traj_pre[:, -1] - traj_pre[:, 0],
                            traj_R_med - traj_E_med,                            
                        )

                        _ER_based_vec_rip = calc_ER_based_vec(
                            traj_rip[:, -1] - traj_rip[:, 0],
                            traj_R_med - traj_E_med,                            
                        )

                        _ER_based_vec_post = calc_ER_based_vec(
                            traj_post[:, -1] - traj_post[:, 0],
                            traj_R_med - traj_E_med,                            
                        )

                        _ER_based_vec_post_post = calc_ER_based_vec(
                            traj_post_post[:, -1] - traj_post_post[:, 0],
                            traj_R_med - traj_E_med,                            
                        )

                        ER_based_vec_pre_pre.append(_ER_based_vec_pre_pre)
                        ER_based_vec_pre.append(_ER_based_vec_pre)
                        ER_based_vec_rip.append(_ER_based_vec_rip)
                        ER_based_vec_post.append(_ER_based_vec_post)
                        ER_based_vec_post_post.append(_ER_based_vec_post_post)
                    except Exception as e:
                        print(e)
                        ER_based_vec_pre_pre.append(np.nan)
                        ER_based_vec_pre.append(np.nan)
                        ER_based_vec_rip.append(np.nan)
                        ER_based_vec_post.append(np.nan)
                        ER_based_vec_post_post.append(np.nan)

                rips_df.loc[
                    (rips_df.index == i_trial + 1), "ER_based_vec_pre_pre"
                ] = ER_based_vec_pre_pre
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "ER_based_vec_pre"
                ] = ER_based_vec_pre
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "ER_based_vec_rip"
                ] = ER_based_vec_rip
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "ER_based_vec_post"
                ] = ER_based_vec_post
                rips_df.loc[
                    (rips_df.index == i_trial + 1), "ER_based_vec_post_post"
                ] = ER_based_vec_post_post

            rips_df = rips_df.reset_index()
            all_rips.append(rips_df)
    all_rips = pd.concat(all_rips).reset_index()
    return all_rips
    
if __name__ == "__main__":
    all_rips = collect_ER_based_vec()
    
    # plots
    fig, axes = plt.subplots(ncols=4, nrows=5, sharex=True, sharey=True, figsize=(6.4*2, 4.8*2))
    for i_x, x in enumerate([
        "ER_based_vec_pre_pre",
        "ER_based_vec_pre",
        "ER_based_vec_rip",
        "ER_based_vec_post",
        "ER_based_vec_post_post",
    ]):
        for i_phase, phase in enumerate(PHASES):
            ax = axes[i_x, i_phase]
            
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
                legend=False,
            )
            
            ax.axvline(x=np.nanmean(all_rips[all_rips.phase == phase][x]), color="black")
            
            ax.set_ylim(0, 0.3)
            ax.set_xlim(-1, 1)
            
            if i_x == 0:
                ax.set_title(f"{phase}")
                
            if i_phase == 0:
                if x[13:] == "pre_pre":
                    ylabel = "-1,400 - -1,000 ms"
                if x[13:] == "pre":
                    ylabel = "-800 - -400 ms"
                if x[13:] == "rip":
                    ylabel = "-200 - +200 ms"
                if x[13:] == "post":
                    ylabel = "+400 - +800 ms"
                if x[13:] == "post_post":
                    ylabel = "+1,000 - +1,400 ms"
                ax.set_ylabel(f"Ripple\n{ylabel}\nProbability")

            ax.set_xlabel("")
            
            fig.supxlabel("ER-based trajectory vector")
            # fig.suptitle("ER-based trajectory around ripples")
            
    mngs.io.save(fig, "./tmp/figs/hist/ER_based_trajectory_arround_ripples.png")
    plt.show()
