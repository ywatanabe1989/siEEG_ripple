#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-30 14:50:02 (ywatanabe)"

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
import pingouin
    
def plot_bar_E_or_R_and_correct_rate_by_match(trials_df):
    cr_df = trials_df.pivot_table(columns=["match", "E_or_R"]).T.reset_index()\
        [["correct", "match", "E_or_R"]]
    cr_df["correct_rate"] = cr_df["correct"] * 100

    fig, ax = plt.subplots()
    sns.barplot(
        data=cr_df,
        x="E_or_R",
        order=["None", "E_only", "R_only", "E_and_R"],
        y="correct_rate",
        hue="match",
        ax=ax,
    )
    # ax.axhline(y=100)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Correct rate [%]")
    import ipdb; ipdb.set_trace()
    return fig

def test_E_or_R_and_correct_rate_by_match(trials_df):
    print(pingouin.anova(
        data=trials_df,
        dv="correct",
        between="E_or_R",
        ))
    

def plot_bar_E_or_R_and_response_time_by_match(trials_df):
    fig, axes = plt.subplots(ncols=2)

    for ax, is_match in zip(axes, [1, 2]):
        # correct_str = "Correct" if is_correct else "Incorrect"
        match_str = "Match IN" if is_match  == 1 else "Mismatch OUT"
        sns.boxplot(
            data=trials_df[(trials_df.response_time <= 2.0) * (trials_df.match == is_match)],
            x="E_or_R",
            order=["None", "E_only", "R_only", "E_and_R"],
            y="response_time",
            hue="correct",
            ax=ax,
        )
        sns.move_legend(ax, "upper right")
        
        ax.set_ylabel("Response time [s]")
        ax.set_title(match_str)
    
    return fig

def test_E_or_R_and_response_time_by_match(trials_df):
    print(pingouin.anova(
        data=trials_df,
        dv="response_time",
        between="E_or_R",
        ))
    

def add_E_or_R(trials_df):
    trials_df["n"] = 1
    trials_df["E_or_R"] = np.nan
    trials_df["E_or_R"][
        (trials_df.n_rips_Encoding > 0) * (trials_df.n_rips_Retrieval == 0)
    ] = "E_only"
    trials_df["E_or_R"][
        (trials_df.n_rips_Encoding == 0) * (trials_df.n_rips_Retrieval > 0)
    ] = "R_only"
    trials_df["E_or_R"][
        (trials_df.n_rips_Encoding > 0) * (trials_df.n_rips_Retrieval > 0)
    ] = "E_and_R"
    trials_df["E_or_R"][
        (trials_df.n_rips_Encoding == 0) * (trials_df.n_rips_Retrieval == 0)
    ] = "None"
    return trials_df

if __name__ == "__main__":
    import argparse
    import mngs

    # Parameters
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]

    # Loads
    rips_df = utils.load_rips()
    trials_df = utils.load_trials(add_n_ripples=True)
    trials_df = add_E_or_R(trials_df)
    
    # Plots
    # np.unique(trials_df[trials_df["E_or_R"] == "E_and_R"].correct, return_counts=True)
    fig = plot_bar_E_or_R_and_correct_rate_by_match(trials_df)
    fig = plot_bar_E_or_R_and_response_time_by_match(trials_df)    

    plt.show()
