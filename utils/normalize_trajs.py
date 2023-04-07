#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-06 07:57:01 (ywatanabe)"

from glob import glob

import mngs
import numpy as np
import quantities as pq
import scipy


def define_phase_time():
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    bin_size = 50 * pq.ms

    starts, ends = [], []
    PHASE_START_END_DICT = {}
    for i_phase, phase in enumerate(PHASES):
        start_s = DURS_OF_PHASES[:i_phase].sum()
        end_s = DURS_OF_PHASES[: (i_phase + 1)].sum()
        PHASE_START_END_DICT[phase] = (start_s, end_s)
        starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
        ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
    colors = ["black", "blue", "green", "red"]
    return starts, ends, PHASE_START_END_DICT, colors


def main(lpath_traj, by):
    if by == "by_session":
        axis = 0

    if by == "by_trial":
        axis = -1

    traj = mngs.io.load(lpath_traj)

    # traj_filted = np.array(
    #     [
    #         scipy.ndimage.gaussian_filter1d(traj[i_traj], sigma=3, axis=-1)
    #         for i_traj in range(len(traj))
    #     ]
    # )
    # traj = traj_filted

    # matplotlib.use("TkAgg")
    # fig, ax = plt.subplots()
    # ax.plot(traj[0, 0, :], label="orig")
    # ax.plot(traj_filted[0, 0, :], label="filted")
    # ax.legend()
    # plt.show()
    # import ipdb

    # ipdb.set_trace()

    traj_mm = np.nanmean(traj, axis=axis, keepdims=True)
    traj_ss = np.nanstd(traj, axis=axis, keepdims=True)
    traj_z = (traj - traj_mm) / traj_ss

    return traj_z


if __name__ == "__main__":
    import re

    LPATHs_traj = glob(f"./data/Sub_0?/Session_0?/traj.npy")

    by = ["by_trial", "by_session"]

    for lpath_traj in LPATHs_traj:
        if re.search("traj_[A-Z]{1,3}\.npy", lpath_traj):
            for _by in by:
                z_traj = main(lpath_traj, _by)
                spath_z_traj = lpath_traj.replace("traj_", f"traj_z_{_by}_")
                mngs.io.save(z_traj, spath_z_traj)
