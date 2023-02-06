#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-12 15:38:54 (ywatanabe)"

import matplotlib

matplotlib.use("Agg")
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


def get_n_bin(times_sec, bin_size, n_bins):
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    return bins


def plot_3d_traj(
    trajectories,
    subject,
    session,
    roi,
    rips_df,
    bin_size,
    trials_df,
    plot_each_trial=False,
    movie=False,
    match=[1, 2],
):
    title = f"Subject {subject}; Session {session}\nROI {roi}"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("Encoding")
    ax.set_ylabel("Maintenance")
    ax.set_zlabel("Retrieval")
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    
    n_bins = trajectories[0].shape[-1]

    # Extract according to match
    rips_df = pd.concat([rips_df[rips_df.match == mm] for mm in match])
    rips_df = rips_df.sort_values("start_time")
    rips_df = rips_df.sort_index()

    indi_b = np.vstack([trials_df.match == mm for mm in match]).sum(axis=0).astype(bool)
    trajectories = trajectories[indi_b]
    trials_df = trials_df.iloc[indi_b]

    # ss, ee, cc: for adding colors for phases
    for ii, (ss, ee, cc) in enumerate(zip(starts, ends, colors)):
        ss_bias = 0 if ii == 0 else 1  # for connecting lines

        if plot_each_trial:
            for i_trial, traj in enumerate(trajectories):

                # ripple
                rip_times = rips_df.iloc[i_trial].center_time
                rip_bins = get_n_bin(rip_times, bin_size, n_bins)
                for rip_bin in rip_bins:
                    ax.scatter(
                        traj[0][rip_bin - 1],
                        traj[1][rip_bin - 1],
                        traj[2][rip_bin - 1],
                        color="yellow",
                        alpha=0.5,
                        # linewidth=0.5,
                    )

                # response time
                response_time = trials_df["response_time"].iloc[i_trial] + 6
                response_time_bin = get_n_bin([response_time], bin_size, n_bins)[0]
                ax.scatter(
                    traj[0][response_time_bin - 1],
                    traj[1][response_time_bin - 1],
                    traj[2][response_time_bin - 1],
                    color="purple",
                    alpha=0.5,
                    # linewidth=0.5,
                )

                ax.scatter(
                    traj[0][rip_bin - 1],
                    traj[1][rip_bin - 1],
                    traj[2][rip_bin - 1],
                    color="yellow",
                    alpha=0.5,
                    # linewidth=0.5,
                )

                ax.plot(
                    traj[0][ss - ss_bias : ee],
                    traj[1][ss - ss_bias : ee],
                    traj[2][ss - ss_bias : ee],
                    color=cc,
                    alpha=0.3,
                    # linewidth=0.5,
                )

        else:
            med_traj = np.median(np.stack(trajectories, axis=0), axis=0)
            ax.plot(
                med_traj[0][ss - ss_bias : ee],
                med_traj[1][ss - ss_bias : ee],
                med_traj[2][ss - ss_bias : ee],
                color=cc,
                # alpha=,
                # linewidth=2,
            )
            med_response_time = (trials_df["response_time"] + 6).median()
            med_response_time_bin = get_n_bin([med_response_time], bin_size, n_bins)[0]
            ax.scatter(
                med_traj[0][med_response_time_bin - 1],
                med_traj[1][med_response_time_bin - 1],
                med_traj[2][med_response_time_bin - 1],
                color=cc,
                # alpha=,
                # linewidth=2,
            )
    phi, theta = 30, 30
    ax.view_init(phi, theta)

    if not movie:
        plt.show()
    else:

        def init():
            return (fig,)

        def animate(i):
            ax.view_init(elev=10.0, azim=i)
            return (fig,)

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=360,
            interval=20,
            blit=True,
        )
        writermp4 = animation.FFMpegWriter(fps=60, extra_args=["-vcodec", "libx264"])
        trial_str = "all_trials" if plot_each_trial else "median"
        if match == [1, 2]:
            match_str = "match_in_and_mismatch_out"
        if match == [1]:
            match_str = "match_in"
        if match == [2]:
            match_str = "mismatch_out"

        os.makedirs(
            f"./tmp/figs/line/neural_trajectory_transformed/{roi}/{trial_str}/{match_str}",
            exist_ok=True,
        )
        spath = (
            f"./tmp/figs/line/neural_trajectory_transformed/{roi}/{trial_str}/{match_str}/"
            f"Sub_{subject}_Session_{session}.mp4"
        )
        anim.save(spath, writer=writermp4)
        print(f"Saved to: {spath}")
        # plt.show()


def main(
    subject,
    session,
    roi,
    plot_each_trial=False,
    match=[1, 2],
):

    # LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
    LPATH = f"./data/Sub_{subject}/Session_{session}/traj_transformed_{roi}.npy"
    trajs = mngs.io.load(LPATH)

    # spike_times = mngs.io.load(LPATH)
    # spike_trains = to_spiketrains(spike_times)

    rips_df = mngs.io.load(f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl")
    rips_df = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]

    trials_df = mngs.io.load(f"./data/Sub_{subject}/Session_{session}/trials_info.csv")

    bin_size = 200 * pq.ms
    # gpfa = GPFA(bin_size=bin_size, x_dim=3)

    # phase
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])

    global starts, ends, colors
    starts, ends = [], []
    for i_phase in range(len(PHASES)):
        start_s = DURS_OF_PHASES[:i_phase].sum()
        end_s = DURS_OF_PHASES[: (i_phase + 1)].sum()
        starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
        ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
    colors = ["black", "blue", "green", "red"]

    trajectories = trajs
    # trajectories = gpfa.fit_transform(spike_trains)
    # traj_df = [pd.DataFrame(traj).T for traj in trajectories]
    # traj_df = pd.concat(traj_df)
    trajectoriess_df = []
    for traj in trajectories:
        traj_df = pd.DataFrame(traj).T
        traj_df["phase"] = "None"
        traj_df["color"] = "None"
        for ss, ee, pp, cc in zip(starts, ends, PHASES, colors):
            indi = (ss <= traj_df.index) * (traj_df.index < ee)
            traj_df.loc[indi, "phase"] = pp
            traj_df.loc[indi, "color"] = cc
        trajectoriess_df.append(traj_df)

    plot_3d_traj(
        trajectories,
        subject,
        session,
        roi,
        rips_df,
        bin_size,
        trials_df,
        plot_each_trial=plot_each_trial,
        movie=True,
        match=match,
    )


if __name__ == "__main__":
    for plot_each_trial in [False, True]:
        for match in [[1, 2], [1], [2]]:
            for subject in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
                for session in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
                    for roi in ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]:
                        try:
                            main(
                                subject,
                                session,
                                roi,
                                plot_each_trial=plot_each_trial,
                                match=match,
                            )
                        except Exception as e:
                            print(e)

    # main("01", "02", "AHL")
    # main("02", "02", "PHR")
    # main("03", "01", "AHR")
    # main("04", "01", "AHL")
    # main("06", "01", "AHL")
    # main("09", "02", "AHR")
