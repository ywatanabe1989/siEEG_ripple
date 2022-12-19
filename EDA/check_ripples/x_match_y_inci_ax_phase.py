#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-02 10:50:38 (ywatanabe)"

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


def calc_inci_rips(trials_df, rips_df):
    dfs = []
    trials_df["n_trial"] = 1
    dur_s_dict = {
        "Fixation": 1,
        "Encoding": 2,
        "Maintenance": 3,
        "Retrieval": 2,
        }
    for phase in PHASES:
        df = trials_df[["correct", f"n_rips_{phase}", "n_trial"]].copy()
        # print(df[f"n_rips_{phase}"].sum())
        _df_1 = df.pivot_table(columns="correct").loc[f"n_rips_{phase}"]
        _df_1 /= dur_s_dict[phase]
        _df_1.name = f"mean_rips_inci_hz"                
        _df_2 = df.pivot_table(columns="correct", aggfunc="sum").loc["n_trial"]
        _df_2.name = f"n_trials"
        df = pd.concat([_df_1.T, _df_2.T], axis=1).reset_index()
        df["phase"] = phase
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    return dfs


# def plot_bar_phase_and_inci_by_correct_and_match(trials_df):
def plot_bar_x_match_y_inci_ax_phase(trials_df):
    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True) # figsize=(6.4*2, 4.8*2)
    for ax, phase in zip(axes, PHASES):
        sns.barplot(
            data=trials_df[trials_df.correct == True],
            x="match",
            y=f"inci_rips_{phase}",
            # hue="correct",
            ax=ax,
            )
        ax.set_xlabel("")        
        ax.set_ylabel("")
        ax.set_title(phase)
    # fig.supxlabel("Match IN/Mismatch OUT")        
    fig.supylabel("Ripple incidence [Hz]")
    # plt.show()
    return fig

def test_phase_and_inci_by_correct_and_match(trials_df):
    match = 1
    data1 = trials_df.inci_rips_Encoding[(trials_df.match == match) * (trials_df.correct == False)]
    data2 = trials_df.inci_rips_Encoding[(trials_df.match == match) * (trials_df.correct == True)]

    import scipy
    print(scipy.stats.mannwhitneyu(data1, data2))

    
    # fig, axes = plt.subplots(ncols=4, figsize=(6.4*2, 4.8*2), sharex=True, sharey=True)
    # for ax, phase in zip(axes, PHASES):
    #     sns.barplot(
    #         data=trials_df,
    #         x="match",
    #         y=f"inci_rips_{phase}",
    #         hue="correct",
    #         ax=ax,
    #         )
    #     ax.set_ylabel("")
    # fig.supylabel("Ripple incidence [Hz]")
    # # plt.show()
    # return fig

def add_inci(trials_df):
    durs_sec = [1, 2, 3, 2]    
    for phase, dur_sec in zip(PHASES, durs_sec):
        trials_df[f"inci_rips_{phase}"] = trials_df[f"n_rips_{phase}"] / dur_sec
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
    trials_df = add_inci(utils.load_trials(add_n_ripples=True))
    trials_df["match"] = trials_df["match"].replace({1:"Match IN", 2:"Mismatch OUT"})
    trials_df["correct"] = trials_df["correct"].astype(bool)

    # inci_df = calc_inci_rips(trials_df, rips_df)
    fig = plot_bar_x_match_y_inci_ax_phase(trials_df)
    mngs.io.save(fig, "./tmp/figs/bar/x_match_y_inci_ax_phase.png")
    # phase_and_ripple_incidence_by_correct_and_match.png")    
