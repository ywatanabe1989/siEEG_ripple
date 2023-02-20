#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-25 14:15:23 (ywatanabe)"

import mngs
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from itertools import combinations
from scipy.linalg import norm
import pandas as pd
from pprint import pprint
import random

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
subject = "06"  # f"{int(list(ROIs.keys())[2]):02d}"  # 04
session = "02"  # 01
# subject = "06"
# session = "02"
# i_trial = 5 + 1
i_trial = 4#random.randint(0,49)
# for i_trial in range(49):

traj_AHL = mngs.io.load(
    f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_AHL.npy"
)
traj_ECL = mngs.io.load(
    f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_ECL.npy"
)
traj_AL = mngs.io.load(
    f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_AL.npy"
)

# plt.plot(traj_AHL[5].T)
# plt.show()

trajs = dict(
    AHL=traj_AHL,
    ECL=traj_ECL,
    AL=traj_AL,
)

for traj_str, traj in trajs.items():
    # trajs[traj_str] = trajs[traj_str][i_trial]
    trajs[traj_str] = np.array(
        [
            scipy.ndimage.gaussian_filter1d(traj[:,i_dim], sigma=0.01, axis=-1)
            for i_dim in range(3)]
    ).transpose(1,0,2)

# np.nanmedian(traj    
# scipy.ndimage.gaussian_filter1d(    , sigma=3)
PHASES = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
starts = [0, 20, 60, 120]
ends = [20, 60, 120, 160]
centers = ((np.array(starts) + np.array(ends)) / 2).astype(int)
width = 10
colors = ["gray", "blue", "green", "red"]
linestyles = ["solid", "dotted", "dashed"]
xx = np.linspace(0, 8, 160) -6 # (np.arange(160)+1) / 160 / (50*1e-3)

out_dict = {}
fig, axes = plt.subplots(nrows=len(trajs), sharex=True, sharey=True)
for i_ax, (ax, (traj_str, traj)) in enumerate(zip(axes, trajs.items())):
    for i_dim in range(traj.shape[1]):
        for ss, ee, cc, pp in zip(starts, ends, colors, PHASES):
            yy = traj[i_trial][i_dim]
            ax.plot(
                xx[ss:ee],
                yy[ss:ee],
                label=f"Factor {i_dim+1}",
                # color=colors[i_dim],
                color=cc,
                linestyle=linestyles[i_dim],
            )
            out_dict[f"xx_{traj_str}_{pp}_Factor_{i_dim}"] = xx[ss:ee]            
            out_dict[f"yy_{traj_str}_{pp}_Factor_{i_dim}"] = yy[ss:ee]
    ax.set_ylabel(traj_str)
    # ax.legend(loc="upper right")
mngs.io.save(fig, "./tmp/figs/line/representative_trajectory/fig.png")
plt.show()
out_df = mngs.general.force_dataframe(out_dict)
mngs.io.save(out_df, "./tmp/figs/line/representative_trajectory/data.csv")


gs = {}
for traj_str, traj in trajs.items():
    for cc, pp in zip(centers, PHASES):
        g = np.median(traj[i_trial, :, int(cc - width / 2) : int(cc + width / 2)], axis=-1)
        gs[f"{traj_str}_{pp}"] = g
mngs.io.save(pd.DataFrame(gs), "./tmp/figs/line/representative_trajectory/gs.csv")


dfs = {}
for roi in ["AHL", "ECL", "AL"]:
    df = pd.DataFrame(columns=PHASES, index=PHASES)
    for ii, jj in combinations(np.arange(4), 2):
        df.iloc[ii, jj] = norm(gs[roi][ii] - gs[roi][jj])
    dfs[roi] = df


pprint(dfs)
# import ipdb; ipdb.set_trace()

# 2
# 4
# 5
# 8
# 11
# 12

