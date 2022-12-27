#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-01 13:45:22 (ywatanabe)"

import re
from glob import glob
from bisect import bisect_left
import mngs
import numpy as np
from natsort import natsorted
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from itertools import combinations
import warnings

import sys

sys.path.append(".")
from siEEG_ripple import utils
import seaborn as sns
import scipy
import pingouin

def plot_bar_set_size_and_inci(trials_df):
    fig, axes = plt.subplots(ncols=len(PHASES), sharex=True, sharey=True)
    for ax, phase, dur_sec in zip(axes, PHASES, DURS_SEC):
        trials_df[f"inci_rips_hz_{phase}"] = trials_df[f"n_rips_{phase}"] / dur_sec
        sns.barplot(
            data=trials_df,
            x="set_size",
            y=f"inci_rips_hz_{phase}",
            ax=ax,
            )
        ax.set_xlabel("")        
        ax.set_ylabel("")
        ax.set_title(phase)
    fig.supylabel("Ripples incidence [Hz]")
    fig.supxlabel("# of letters in Encoding")

    out_df = trials_df[["set_size", "correct"] + [f"inci_rips_hz_{phase}" for phase in PHASES]]
    return fig, out_df

def test_set_size_and_inci(df):
    # ANOVA
    print("\nANOVA\n")
    for phase in PHASES:
        print(phase)
        print(
            pingouin.anova(
                data=df[df.correct == True],
                dv=f"inci_rips_hz_{phase}",
                between="set_size"
                )
            )

    # post-hoc ttest
    from itertools import combinations

    print("\nEncoding\n")    
    for ss_combi in combinations([4, 6, 8], 2):
        print(f"set size: {ss_combi[0]}, {ss_combi[1]}")
        print(scipy.stats.brunnermunzel(
            df[df.set_size == ss_combi[0]]["inci_rips_hz_Encoding"],
            df[df.set_size == ss_combi[1]]["inci_rips_hz_Encoding"],
            ))

        # fig, ax = plt.subplots()
        # sns.histplot(
        #     data=df[df.correct == True],
        #     x="inci_rips_hz_Encoding",
        #     hue="set_size",
        #     stat="probability",
        #     common_norm=False,
        #     multiple="dodge",
        #     ax=ax,
        #     # cumulative=True,
        #     # kde=True,
        #     )
        # plt.show()
        
        # print(pingouin.ttest(
        #     x=df[df.set_size == ss_combi[0]]["inci_rips_hz_Encoding"],
        #     y=df[df.set_size == ss_combi[1]]["inci_rips_hz_Encoding"],
        #     ))
    # print(pingouin.ttest(
    #     x=df[df.set_size == 4]["inci_rips_hz_Encoding"],
    #     y=df[df.set_size == 6]["inci_rips_hz_Encoding"],
    #     ))
    # print(pingouin.ttest(
    #     x=df[df.set_size == 6]["inci_rips_hz_Encoding"],
    #     y=df[df.set_size == 8]["inci_rips_hz_Encoding"],
    #     ))
    # print(pingouin.ttest(
    #     x=df[df.set_size == 4]["inci_rips_hz_Encoding"],
    #     y=df[df.set_size == 8]["inci_rips_hz_Encoding"],
    #     ))
    for phase in PHASES:
        print(phase)
        print(scipy.stats.spearmanr(df.set_size, df[f"inci_rips_hz_{phase}"]))

if __name__ == "__main__":
    import argparse
    import mngs

    # Parameters
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_SEC = [1, 2, 3, 2]

    # Loads
    rips_df = utils.load_rips()
    trials_df = utils.load_trials(add_n_ripples=True)

    # Plots
    fig, df = plot_bar_set_size_and_inci(trials_df)

    mngs.io.save(fig, "./tmp/figs/bar/set_size_and_incidence.png")
    plt.show()

    test_set_size_and_inci(df)
