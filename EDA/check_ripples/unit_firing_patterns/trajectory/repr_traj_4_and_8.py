#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-10 15:19:34 (ywatanabe)"

import mngs
import numpy as np
import sys
sys.path.append(".")
import utils
import pandas as pd

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

df = pd.DataFrame()
df = {}
for phase, (start_bin, end_bin) in GS_BINS_DICT.items():
    np.median(traj_4[..., start_bin:end_bin].transpose(1,0,2).reshape(3,-1), axis=-1)
    np.median(traj_8[..., start_bin:end_bin].transpose(1,0,2).reshape(3,-1), axis=-1)
    df[f"{phase}]
    




median_traj_4 = np.median(traj_4, axis=0)
median_traj_8 = np.median(traj_8, axis=0)

df = mngs.pd.DataFrame({
    "median_traj_4_Factor_1": median_traj_4[0], 
    "median_traj_4_Factor_2": median_traj_4[1], 
    "median_traj_4_Factor_3": median_traj_4[2],    
    "median_traj_8_Factor_1": median_traj_8[0], 
    "median_traj_8_Factor_2": median_traj_8[1], 
    "median_traj_8_Factor_3": median_traj_8[2],    
})
mngs.io.save(df, "./tmp/figs/line/repr_traj_4_and_8.csv")


