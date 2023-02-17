#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-12 14:47:19 (ywatanabe)"

from glob import glob
import mngs
import numpy as np


def define_phase_time():
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    bin_size = 200 * pq.ms

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


def main(lpath_traj):
    starts, ends, PHASE_START_END_DICT, colors = define_phase_time()

    traj = mngs.io.load(lpath_traj)
    traj_med = np.median(traj, axis=0)

    g_f = np.median(traj_med[:, starts[0] : ends[0]], axis=-1)
    g_e = np.median(traj_med[:, starts[1] : ends[1]], axis=-1)
    g_m = np.median(traj_med[:, starts[2] : ends[2]], axis=-1)
    g_r = np.median(traj_med[:, starts[3] : ends[3]], axis=-1)

    g_f_prime = g_f - g_f
    g_e_prime = g_e - g_f
    g_m_prime = g_m - g_f
    g_r_prime = g_r - g_f

    v_fe = g_e - g_f
    v_fm = g_m - g_f
    v_fr = g_r - g_f

    P = np.stack([g_e_prime, g_m_prime, g_r_prime], axis=1).reshape(3, 3).T

    traj_prime = traj - g_f[np.newaxis, :, np.newaxis]
    try:
        P_inv = np.linalg.inv(P)
    except Exception as e:
        print(e)
        try:
            P_inv = np.linalg.inv(P + 1e-10)
        except Exception as e:
            return np.nan
            # print(e)
            # import ipdb; ipdb.set_trace()
    # np.dot(g_e_prime, P_inv) # array([1., 0, 0])

    traj_prime_prime = np.dot(traj_prime.transpose(0, 2, 1), P_inv).transpose(0, 2, 1)

    return traj_prime_prime


if __name__ == "__main__":
    LPATHs_traj = glob(f"./data/Sub_0?/Session_0?/traj_*.npy")
    for lpath_traj in LPATHs_traj:
        print()
        transposed_traj = main(lpath_traj)
        spath_transposed_traj = lpath_traj.replace("traj_", "traj_transposed_")
        mngs.io.save(transposed_traj, spath_transposed_traj)
