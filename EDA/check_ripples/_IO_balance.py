#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-26 11:38:24 (ywatanabe)"

import matplotlib
import mngs

matplotlib.use("TkAgg")
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(".")
from itertools import combinations, product
import scipy
from scipy.stats import brunnermunzel
from siEEG_ripple import utils

# Parameters
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
DURS_SEC = [1, 2, 3, 2]
ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

# Loads
rips_df = utils.load_rips()

# Scatter
fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
for ax, phase in zip(axes, PHASES):
    sns.scatterplot(
        data=rips_df[rips_df.phase == phase],
        x="ripple_amplitude_sd",
        y="population_burst_rate",
        hue="set_size",
        ax=ax,
        alpha=.7,
        )
    ax.set_title(phase)
mngs.io.save(fig, "./tmp/figs/scatter/IO_balance.png")
plt.show()

# Box by phase
fig, ax = plt.subplots()
sns.boxplot(
    data=rips_df,
    x="phase",
    y="IO_balance",
    hue="set_size",
    order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
    # showfliers=False,
    ax=ax,
    )
mngs.io.save(fig, "./tmp/figs/box/IO_balance.png")
plt.show()


# BM Test
for phase in PHASES:
    for ss1, ss2 in product([4,6,8], [4,6,8]):
        print(phase, ss1, ss2)
        print(
        brunnermunzel(
            rips_df[(rips_df.phase == phase) * (rips_df.set_size == ss1)]["n_firings"],
            rips_df[(rips_df.phase == phase) * (rips_df.set_size == ss2)]["n_firings"],
            alternative="less",
            ))
    import ipdb; ipdb.set_trace()
