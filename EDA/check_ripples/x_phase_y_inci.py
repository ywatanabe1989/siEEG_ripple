#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-02 11:19:19 (ywatanabe)"

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
from eeg_ieeg_ripple_clf import utils
import seaborn as sns
import scipy
import pingouin

def plot_bar_x_phase_y_inci(trials_df):
    fig, axes = plt.subplots(ncols=len(PHASES), sharex=True, sharey=True)
    for phase, ax in zip(PHASES, axes):
        sns.barplot(
            data=trials_df,
            y=f"inci_rips_hz_{phase}",
            ax=ax,
            )
        ax.set_xlabel(phase)        
        ax.set_ylabel("")
        ax.set_title(phase)
    fig.supylabel("Ripples incidence [Hz]")

    return fig

def test_x_phase_y_inci(trials_df):
    # Kruskal-Wallis
    scipy.stats.kruskal(
        *[trials_df[f"inci_rips_hz_{phase}"] for phase in PHASES]
    )

    # post-hoc
    from itertools import combinations
    for phase_a, phase_b in combinations(PHASES, 2):
        print(phase_a, phase_b)
        print(scipy.stats.brunnermunzel(
            trials_df[f"inci_rips_hz_{phase_a}"],
            trials_df[f"inci_rips_hz_{phase_b}"]                      
            ))
        print()
        # Fixation Encoding
        # BrunnerMunzelResult(statistic=4.854320122170011, pvalue=1.3355291286210047e-06)

        # Fixation Maintenance
        # BrunnerMunzelResult(statistic=12.112883145981202, pvalue=0.0)

        # Fixation Retrieval
        # BrunnerMunzelResult(statistic=8.094432595684705, pvalue=1.3322676295501878e-15)

        # Encoding Maintenance
        # BrunnerMunzelResult(statistic=6.358020268779824, pvalue=2.7359070564614285e-10)

        # Encoding Retrieval
        # BrunnerMunzelResult(statistic=3.8794035216767355, pvalue=0.00010911597242380111)

        # Maintenance Retrieval
        # BrunnerMunzelResult(statistic=-1.8129938550680396, pvalue=0.07003899932505346)


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
    fig = plot_bar_x_phase_y_inci(trials_df)
    mngs.io.save(fig, "./tmp/figs/bar/x_phase_y_inci.png")
    plt.show()


    # Test
    test_x_phase_y_inci(trials_df)
