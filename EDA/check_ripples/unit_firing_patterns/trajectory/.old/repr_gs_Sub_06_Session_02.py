#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-22 11:24:34 (ywatanabe)"

import mngs
import numpy as np
import sys

sys.path.append(".")
import utils
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def collect_gs(med_traj):
    gs = {}
    for phase, (bin_start, bin_end) in GS_BINS_DICT.items():
        # g = np.nanmedian(_traj[..., bin_start:bin_end].transpose(1,0,2).reshape(3, -1), axis=-1)
        g = np.median(med_traj[..., bin_start:bin_end], axis=-1)
        gs[f"{phase[0]}_x"] = g[0]
        gs[f"{phase[0]}_y"] = g[1]
        gs[f"{phase[0]}_z"] = g[2]            
        print(phase)
    return gs

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


med_traj = np.median(traj, axis=0)
gs = collect_gs(med_traj)

mngs.io.save(pd.DataFrame(med_traj).T, f"./tmp/figs/scatter/repr_med_traj_Subject_06_Session_02.csv")
mngs.io.save(pd.DataFrame(pd.Series(gs)).T, f"./tmp/figs/scatter/repr_med_traj_gs_Subject_06_Session_02.csv")

# for match in [1,2]:
#     traj_match = traj[trials_df.match == match]
#     med_traj_match = np.median(traj_match, axis=0)
#     gs_match = collect_gs(med_traj_match)

#     mngs.io.save(pd.DataFrame(med_traj_match).T, f"./tmp/figs/scatter/repr_med_traj_match_{match}.csv")
#     mngs.io.save(pd.DataFrame(pd.Series(gs_match)).T, f"./tmp/figs/scatter/repr_med_traj_gs_match_{match}.csv")



# traj_4_in = traj[(trials_df.set_size == 4) * (trials_df.match == 1)]
# traj_6_in = traj[(trials_df.set_size == 6) * (trials_df.match == 1)]
# traj_8_in = traj[(trials_df.set_size == 8) * (trials_df.match == 1)]
# traj_4_out = traj[(trials_df.set_size == 4) * (trials_df.match == 2)]
# traj_6_out = traj[(trials_df.set_size == 6) * (trials_df.match == 2)]
# traj_8_out = traj[(trials_df.set_size == 8) * (trials_df.match == 2)]

# gs = {}
# for match in [1,2]:
#     for set_size in [4, 6, 8]:
#         _traj = traj[(trials_df.set_size == set_size)] #  * (trials_df.match == match)
#         for phase, (bin_start, bin_end) in GS_BINS_DICT.items():
#             # g = np.nanmedian(_traj[..., bin_start:bin_end].transpose(1,0,2).reshape(3, -1), axis=-1)
#             g = np.median(np.nanmedian(_traj[..., bin_start:bin_end], axis=0), axis=-1)
#             gs[f"{phase[0]}_{set_size}_{match}_x"] = g[0]
#             gs[f"{phase[0]}_{set_size}_{match}_y"] = g[1]
#             gs[f"{phase[0]}_{set_size}_{match}_z"] = g[2]            
#             print(phase)
# mngs.io.save(pd.DataFrame(pd.Series(gs)).T, "./tmp/figs/scatter/repr_traj_gs.csv")



# # median_traj_4 = np.median(traj_4, axis=0)
# # median_traj_8 = np.median(traj_8, axis=0)

# # sigma = 5
# # median_traj_4_smoothed = np.array(
# #     [
# #         gaussian_filter1d(median_traj_4[i_factor], sigma=sigma)
# #         for i_factor in range(median_traj_4.shape[0])
# #     ]
# # )
# # median_traj_8_smoothed = np.array(
# #     [
# #         gaussian_filter1d(median_traj_8[i_factor], sigma=sigma)
# #         for i_factor in range(median_traj_8.shape[0])
# #     ]
# # )

# # median_traj_4 = median_traj_4_smoothed
# # median_traj_8 = median_traj_8_smoothed

# # # import matplotlib
# # # matplotlib.use("TkAgg")
# # # import matplotlib.pyplot as plt

# # # fig, axes = plt.subplots(nrows=2)
# # # axes[0].plot(median_traj_4_smoothed.T)
# # # axes[1].plot(median_traj_8_smoothed.T)
# # # plt.show()

# # df = pd.DataFrame(
# #     {
# #         "median_traj_4_Factor_1": median_traj_4[0],
# #         "median_traj_4_Factor_2": median_traj_4[1],
# #         "median_traj_4_Factor_3": median_traj_4[2],
# #         "median_traj_8_Factor_1": median_traj_8[0],
# #         "median_traj_8_Factor_2": median_traj_8[1],
# #         "median_traj_8_Factor_3": median_traj_8[2],
# #     }
# # )
# # mngs.io.save(df, "./tmp/figs/line/repr_traj_4_and_8.csv")

# # # df = pd.DataFrame({
# # #     "median_smoothed_traj_4_Factor_1": median_traj_4_smoothed[0],
# # #     "median_smoothed_traj_4_Factor_2": median_traj_4_smoothed[1],
# # #     "median_smoothed_traj_4_Factor_3": median_traj_4_smoothed[2],
# # #     "median_smoothed_traj_8_Factor_1": median_traj_8_smoothed[0],
# # #     "median_smoothed_traj_8_Factor_2": median_traj_8_smoothed[1],
# # #     "median_smoothed_traj_8_Factor_3": median_traj_8_smoothed[2],
# # # })
# # # mngs.io.save(df, "./tmp/figs/line/repr_smoothed_traj_4_and_8.csv")



# # gs_df = {}
# # for phase, (start_bin, end_bin) in GS_BINS_DICT.items():
# #     gs_df[f"{phase}_4"] = np.median(
# #         median_traj_4[..., start_bin:end_bin], axis=-1
# #     )
# #     gs_df[f"{phase}_8"] = np.median(
# #         median_traj_8[..., start_bin:end_bin], axis=-1
# #     )
# # gs_df = pd.DataFrame(gs_df)
# # mngs.io.save(gs_df, "./tmp/figs/line/repr_traj_gs_4_and_8.csv")
