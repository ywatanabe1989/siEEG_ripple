#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-22 13:33:53 (ywatanabe)"

from glob import glob
import mngs
import numpy as np
import matplotlib

# matplotlib.use("Agg")
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
import quantities as pq
import os

def get_i_bin(times_sec, bin_size, n_bins):
    from bisect import bisect_right
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    # bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in times_sec]
    return bins


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


def main(lpath_traj, movie=True):
    """
    lpath_traj = lpath
    lpath_traj = LPATHs[0]
    """

    ldir = mngs.general.split_fpath(lpath_traj)[0]

    trajs = mngs.io.load(lpath_traj)  # (50, 3, 160)
    trials_df = mngs.io.load(ldir + "trials_info.csv")

    starts, ends, PHASE_START_END_DICT, colors = define_phase_time()
    colors_dict = {pp: cc for pp, cc in zip(PHASES, colors)}
    colors_dict_ss = {4:"gray", 6: "orange", 8:"red"}

    gs = {}
    med_trajs = {}
    for i_phase, phase in enumerate(PHASES):
        for set_size in [4, 6, 8]:
            med_traj = np.median(
                    trajs[:, :, :][
                        trials_df.set_size == set_size
                    ],
                    axis=0,
                )
            med_trajs[f"{phase}_{set_size}"] = med_traj
            gg = np.median(med_traj[:, starts[i_phase] : ends[i_phase]], axis=-1)
            gs[f"{phase}_{set_size}"] = gg

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    # gs
    for phase_set_size, gg in gs.items():
        phase = phase_set_size.split("_")[0]
        set_size = phase_set_size.split("_")[1]
        ax.scatter(gg[0], gg[1], gg[2], label=phase, color=colors_dict[phase])
        ax.text(gg[0], gg[1], gg[2], s=set_size)

    # med trajectories
    for phase_set_size, med_traj in med_trajs.items():
        phase = phase_set_size.split("_")[0]
        set_size = phase_set_size.split("_")[1]
        ax.plot(med_traj[0], med_traj[1], med_traj[2], label=phase, color=colors_dict_ss[int(set_size)])
        # ax.text(gg[0], gg[1], gg[2], s=set_size)

    import ipdb; ipdb.set_trace()


        
    for phase in PHASES:
        gg_p = np.array([gs[f"{phase}_4"], gs[f"{phase}_6"], gs[f"{phase}_8"]])
        ax.plot(xs=gg_p.T[0], ys=gg_p.T[1], zs=gg_p.T[2], color=colors_dict[phase])

    for i_traj in range(len(trajs)):
        traj = trajs[i_traj]
        trial = trials_df.iloc[i_traj]
        color = {4:"gray", 6:"orange", 8:"red"}[trial.set_size]
        ax.plot(xs=traj[0], ys=traj[1], zs=traj[2], color=color)

    for set_size in [4, 6, 8]:
        indi = trials_df.set_size == set_size
        ax.scatter(np.median(trajs[indi,i_dim,:]))

    ax.legend()

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

        os.makedirs(
            f"./tmp/figs/line/set_size_dependent_neural_states/{roi}/",
            exist_ok=True,
        )
        spath = (
            f"./tmp/figs/line/set_size_dependent_neural_states/{roi}/"
            f"Sub_{subject}_Session_{session}.mp4"
        )
        anim.save(spath, writer=writermp4)
        print(f"Saved to: {spath}")
        # plt.show()


if __name__ == "__main__":
    import re
    import sys
    sys.path.append(".")
    import utils
    roi = "AHL"
    LPATHs = glob(f"data/Sub_0?/Session_0?/traj_z_by_session_{roi}.npy")
    lpath_traj = LPATHs[0]
    subject = re.findall("Sub_0[0-1]", lpath_traj)[0][-2:]
    session = re.findall("Session_0[0-1]", lpath_traj)[0][-2:]    

    ldir = mngs.general.split_fpath(lpath_traj)[0]

    trajs = mngs.io.load(lpath_traj)  # (50, 3, 160)
    trials_df = mngs.io.load(ldir + "trials_info.csv")
    rips_df = utils.load_rips()
    rips_df = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]

    starts, ends, PHASE_START_END_DICT, colors = define_phase_time()
    colors_dict = {pp: cc for pp, cc in zip(PHASES, colors)}
    colors_dict_ss = {4:"gray", 6: "orange", 8:"red"}

    # extract g and med trajs
    gs = {}
    med_trajs = {}
    for i_phase, phase in enumerate(PHASES):
        for set_size in [4, 6, 8]:
            med_traj = np.median(
                    trajs[:, :, :][
                        trials_df.set_size == set_size
                    ],
                    axis=0,
                )
            med_trajs[f"{set_size}"] = med_traj
            gg = np.median(med_traj[:, starts[i_phase] : ends[i_phase]], axis=-1)
            gs[f"{phase}_{set_size}"] = gg



    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")


    # i_traj = 0
    # for ii in range(trajs_filted.shape[-1]):
    #     xx = trajs_filted[i_traj,0,ii:ii+2]
    #     yy = trajs_filted[i_traj,1,ii:ii+2]
    #     zz = trajs_filted[i_traj,2,ii:ii+2]        
    #     ax.plot(xs=xx,
    #                ys=yy,
    #                zs=zz,
    #                color=cm.copper(ii/trajs_filted.shape[-1])
    #                )

    # # ripple times
    # bin_size = 50 * pq.ms    
    # i_rip_bins = get_i_bin(rips_df[rips_df.trial_number == i_traj + 1]["center_time"], bin_size, 160)
    # for i_rip_bin in i_rip_bins:
    #     ax.scatter(xs=trajs_filted[i_traj,0,i_rip_bin],
    #                ys=trajs_filted[i_traj,1,i_rip_bin],
    #                zs=trajs_filted[i_traj,2,i_rip_bin],
    #                color="red",
    #                )
        
    
    # plt.show()

    # gs
    for phase_set_size, gg in gs.items():
        phase = phase_set_size.split("_")[0]
        set_size = phase_set_size.split("_")[1]
        ax.scatter(gg[0], gg[1], gg[2], label=phase, color=colors_dict[phase])
        ax.text(gg[0], gg[1], gg[2], s=set_size)

    # med trajectories
    for set_size, med_traj in med_trajs.items():
        ax.plot(med_traj[0], med_traj[1], med_traj[2], label=phase, color=colors_dict_ss[int(set_size)])
        # ax.text(gg[0], gg[1], gg[2], s=set_size)

    # # med of med trajectories
    # for set_size, med_traj in med_trajs.items():
    #     ax.scatter(np.median(med_traj[0], axis=-1),
    #             np.median(med_traj[1], axis=-1),
    #             np.median(med_traj[2], axis=-1),
    #             label=phase, color=colors_dict_ss[int(set_size)])
    #     # ax.text(gg[0], gg[1], gg[2], s=set_size)
        

    plt.show()

    
