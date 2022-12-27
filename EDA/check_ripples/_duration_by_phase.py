#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-02 10:04:43 (ywatanabe)"

import sys
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mngs
import seaborn as sns

sys.path.append(".")
from siEEG_ripple import utils

# Parameters
SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
LONG_RIPPLE_THRES_MS = mngs.io.load("./config/global.yaml")["LONG_RIPPLE_THRES_MS"]

# Loads
rips_df = utils.load_rips().reset_index()
lrips_df = rips_df[rips_df.duration_ms > LONG_RIPPLE_THRES_MS]

# counts sample size
# rips_df["n"] = 1
# lrips_df["n"] = 1
n_rips_df = (
    rips_df.pivot_table(columns=["phase", "correct"], aggfunc=sum)
    .loc["n"]
    .loc[PHASES]
)
n_lrips_df = (
    lrips_df.pivot_table(columns=["phase", "correct"], aggfunc=sum)
    .loc["n"]
    .loc[PHASES]
)
lrips_perc_df = (100 * n_lrips_df / n_rips_df).round(2)[PHASES]

# plots
fig, ax = plt.subplots()
sns.boxplot(
    data=rips_df,
    y="duration_ms",
    x="phase",
    order=PHASES,
    hue="correct",
)
title_l1 = f"ns = {mngs.general.connect_strs([str(int(n_rip)) for n_rip in n_rips_df])}"
title_l2 = f"Long ripple rate [%] = \
{mngs.general.connect_strs([str(lrip_perc) for lrip_perc in lrips_perc_df])}"    
ax.set_title(f"{title_l1}\n{title_l2}")
# ax.set_yscale("log")

# saves
mngs.io.save(fig, "./tmp/figs/box/duration_and_correct.png")

## EOF
