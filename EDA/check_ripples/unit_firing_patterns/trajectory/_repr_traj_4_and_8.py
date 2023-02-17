#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-15 12:29:45 (ywatanabe)"

import mngs
import numpy as np
import sys

sys.path.append(".")
import utils
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Parameters
(
    PHASES,
    PHASES_BINS_DICT,
    GS_BINS_DICT,
    COLORS_DICT,
    BIN_SIZE,
) = utils.define_phase_time()

# Loads
traj = mngs.io.load("./data/Sub_06/Session_02/traj_z_by_session_AHL.npy")
trials_df = mngs.io.load("./data/Sub_06/Session_02/trials_info.csv")

traj_4 = traj[trials_df.set_size == 4]
traj_8 = traj[trials_df.set_size == 8]


median_traj_4 = np.median(traj_4, axis=0)
median_traj_8 = np.median(traj_8, axis=0)

sigma = 5
median_traj_4_smoothed = np.array(
    [
        gaussian_filter1d(median_traj_4[i_factor], sigma=sigma)
        for i_factor in range(median_traj_4.shape[0])
    ]
)
median_traj_8_smoothed = np.array(
    [
        gaussian_filter1d(median_traj_8[i_factor], sigma=sigma)
        for i_factor in range(median_traj_8.shape[0])
    ]
)

median_traj_4 = median_traj_4_smoothed
median_traj_8 = median_traj_8_smoothed

# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(nrows=2)
# axes[0].plot(median_traj_4_smoothed.T)
# axes[1].plot(median_traj_8_smoothed.T)
# plt.show()

df = pd.DataFrame(
    {
        "median_traj_4_Factor_1": median_traj_4[0],
        "median_traj_4_Factor_2": median_traj_4[1],
        "median_traj_4_Factor_3": median_traj_4[2],
        "median_traj_8_Factor_1": median_traj_8[0],
        "median_traj_8_Factor_2": median_traj_8[1],
        "median_traj_8_Factor_3": median_traj_8[2],
    }
)
mngs.io.save(df, "./tmp/figs/line/repr_traj_4_and_8.csv")

# df = pd.DataFrame({
#     "median_smoothed_traj_4_Factor_1": median_traj_4_smoothed[0],
#     "median_smoothed_traj_4_Factor_2": median_traj_4_smoothed[1],
#     "median_smoothed_traj_4_Factor_3": median_traj_4_smoothed[2],
#     "median_smoothed_traj_8_Factor_1": median_traj_8_smoothed[0],
#     "median_smoothed_traj_8_Factor_2": median_traj_8_smoothed[1],
#     "median_smoothed_traj_8_Factor_3": median_traj_8_smoothed[2],
# })
# mngs.io.save(df, "./tmp/figs/line/repr_smoothed_traj_4_and_8.csv")



gs_df = {}
for phase, (start_bin, end_bin) in GS_BINS_DICT.items():
    gs_df[f"{phase}_4"] = np.median(
        median_traj_4[..., start_bin:end_bin], axis=-1
    )
    gs_df[f"{phase}_8"] = np.median(
        median_traj_8[..., start_bin:end_bin], axis=-1
    )
gs_df = pd.DataFrame(gs_df)
mngs.io.save(gs_df, "./tmp/figs/line/repr_traj_gs_4_and_8.csv")
