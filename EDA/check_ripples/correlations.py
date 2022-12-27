#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-26 12:22:28 (ywatanabe)"

import re
from glob import glob
from bisect import bisect_left
import mngs
import numpy as np
from natsort import natsorted
import matplotlib

matplotlib.use("TkAgg")
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
from itertools import combinations
from pprint import pprint

def add_previous_trials_parameters(rips_df, trials_df):
    rips_df["previous_set_size"] = np.nan
    rips_df["previous_correct"] = np.nan
    rips_df["previous_responset_time"] = np.nan
    rips_df = rips_df.reset_index()
    for i_rip, (_, rip) in enumerate(rips_df.iterrows()):
        if rip.trial_number == 1:
            continue
        else:
            indi_subject = trials_df.subject == rip.subject
            indi_session = trials_df.session == rip.session
            indi_trial_prev = trials_df.trial_number == rip.trial_number - 1
            indi = indi_subject * indi_session * indi_trial_prev

            rips_df.loc[i_rip, "previous_set_size"] = float(trials_df[indi].set_size)
            rips_df.loc[i_rip, "previous_correct"] = float(trials_df[indi].correct)
            rips_df.loc[i_rip, "previous_response_time"] = float(
                trials_df[indi].response_time
            )
    return rips_df

if __name__ == "__main__":
    import argparse
    import mngs
    import scipy
    import pingouin

    # Parameters
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_SEC = [1, 2, 3, 2]
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    # Loads
    rips_df = utils.load_rips()
    trials_df = utils.load_trials(add_n_ripples=True)
    rips_df = add_previous_trials_parameters(rips_df, trials_df)
        
    # Correlation among variables                
    corr_df = rips_df[
        [
            "previous_set_size",
            "previous_correct",
            "previous_response_time",
            "set_size",
            "correct",
            "response_time",
            "match",
            # "phase",
            "log10(duration_ms)",
            "log10(ripple_peak_amplitude_sd)",
            # "ripple_amplitude_sd",
            "n_firings",
            # "unit_participation_rate",
            # "IO_balance",
        ]
    ]
    corr_matrix = corr_df.corr()        

    # heatmap
    fig, ax = plt.subplots(figsize=(6.4*2, 4.8*2))
    sns.heatmap(corr_matrix, annot=True, ax=ax, cmap="vlag", vmin=-1, vmax=1)
    mngs.io.save(fig, "./tmp/figs/heatmap/correlations.png")
    # plt.show()

    # variables pair
    g = sns.pairplot(corr_matrix, height=2.5)
    mngs.io.save(g, "./tmp/figs/pair/variables.png")
    # plt.show()

    # effect on duration
    fig, ax = plt.subplots()
    sns.boxplot(
        data=rips_df[(rips_df.previous_set_size == 6)],  # + (rips_df.set_size == 8)],
        x="phase",
        y="log10(duration_ms)",
        hue="set_size",
        order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
        ax=ax,
    )
    # ax.set_ylim(0, 200)
    plt.show()
