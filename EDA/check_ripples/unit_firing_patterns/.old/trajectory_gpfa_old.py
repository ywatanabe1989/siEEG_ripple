#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-23 19:31:06 (ywatanabe)"

import matplotlib
matplotlib.use('TkAgg')
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

# from quantities import s
# train = neo.SpikeTrain([3.0, 4.0]*pq.s, t_stop=8.0)
# train.t_start = array(-6.0) * pq.s
# train.t_stop = array(2.0) * pq.s

def to_spiketrains(spike_times_all_trials):
    spike_trains_all_trials = []    
    for st_trial in spike_times_all_trials:

        spike_trains_trial = []
        for col, col_df in st_trial.iteritems():

            spike_times = col_df[col_df != ""]            
            train = neo.SpikeTrain(list(spike_times)*pq.s, t_start=-6.0, t_stop=2.0)
            # train.t_start = array(-6.0) * pq.s
            # train.t_stop = array(2.0) * pq.s
            
            spike_trains_trial.append(train)

        spike_trains_all_trials.append(spike_trains_trial)
            
    return spike_trains_all_trials

def plot_3d_traj(trajectories, subject, session, roi, i_trial=0):
    from mpl_toolkits.mplot3d import Axes3D

    title = f"Subject {subject}; Session {session}\nROI {roi}"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    # ax.plot(trajectories[i_trial][0],
    #         trajectories[i_trial][1],
    #         trajectories[i_trial][2],
    #         color="blue",
    #         # alpha=,
    #         # linewidth=2,
    #         )    

    for ii, (ss, ee, cc) in enumerate(zip(starts, ends, colors)):
        # for traj in trajectories:
        #     ax.plot(traj[0][ss:ee],
        #             traj[1][ss:ee],
        #             traj[2][ss:ee],
        #             color=cc,
        #             alpha=0.3,
        #             # linewidth=0.5,
        #             )
        ss_bias = 0 if ii == 0 else 1
        mean_traj = np.median(np.stack(trajectories, axis=0), axis=0)    
        ax.plot(mean_traj[0][ss-ss_bias:ee],
                mean_traj[1][ss-ss_bias:ee],
                mean_traj[2][ss-ss_bias:ee],
                color=cc,
                # alpha=,
                # linewidth=2,
                )
    ax.view_init(phi, theta)
    def init():
        return (fig,)

    def animate(i):
        ax.view_init(elev=10.0, azim=i)
        return (fig,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=360, interval=20, blit=True,
        )
    writermp4 = animation.FFMpegWriter(fps=60, extra_args=["-vcodec", "libx264"])
    os.makedirs(f"./tmp/figs/line/neural_trajectory/{roi}", exist_ok=True)
    spath = f"./tmp/figs/line/neural_trajectory/{roi}/Sub_{subject}_Session_{session}.mp4"
    anim.save(spath, writer=writermp4)
    print(f"Saved to: {spath}")
    # plt.show()

# def plot_2d_traj(trajectories, i_trial=0):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     # ax.plot(trajectories[i_trial][0],
#     #         trajectories[i_trial][1],
#     #         color="blue",
#     #         # alpha=,
#     #         # linewidth=2,
#     #         )    

#     for traj in trajectories:
#         ax.plot(traj[0],
#                 traj[1],
#                 color="gray",
#                 alpha=0.3,
#                 # linewidth=0.5,
#                 )
#     mean_traj = np.mean(trajectories, axis=0)    
#     ax.plot(mean_traj[0],
#             mean_traj[1],
#             color="red",
#             # alpha=,
#             # linewidth=2,
#             )    
#     plt.show()

def plot_2d_traj(trajectories, i_trial=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for ss, ee, cc in zip(starts, ends, colors):
        for traj in trajectories:
            ax.plot(traj[0][ss:ee],
                    traj[1][ss:ee],
                    color=cc,
                    alpha=0.3,
                    # linewidth=0.5,
                    )
    # mean_traj = np.mean(trajectories, axis=0)    
    # ax.plot(mean_traj[0],
    #         mean_traj[1],
    #         color="red",
    #         # alpha=,
    #         # linewidth=2,
    #         )    
    plt.show()

def main(subject, session, roi):
    
    LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"


    spike_times = mngs.io.load(LPATH)
    spike_trains = to_spiketrains(spike_times)


    # GPFA
    n_dim = 3
    bin_size = 200 * pq.ms
    gpfa = GPFA(bin_size=bin_size, x_dim=n_dim)

    # phase
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])

    global starts, ends, colors
    starts, ends = [], []
    for i_phase in range(len(PHASES)):
        start_s = DURS_OF_PHASES[:i_phase].sum()
        end_s = DURS_OF_PHASES[:(i_phase+1)].sum()
        starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
        ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
    colors = ["black", "blue", "green", "red"]


    

    

    trajectories = gpfa.fit_transform(spike_trains)
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

    if n_dim == 2:
        plot_2d_traj(trajectories)
    if n_dim == 3:        
        plot_3d_traj(trajectories, subject, session, roi)    
    
if __name__ == "__main__":
    phi, theta = 30, 30
    for subject in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
        for session in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            for roi in ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]:
                try:
                    main(subject, session, roi)
                except Exception as e:
                    print(e)
                    
    # main("01", "02", "AHL")                        
    # main("02", "02", "PHR")
    # main("03", "01", "AHR")
    # main("04", "01", "AHL")
    # main("06", "01", "AHL")
    # main("09", "02", "AHR")                
