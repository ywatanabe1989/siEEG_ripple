#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-27 17:06:09 (ywatanabe)"

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


def get_indi(rips_df): # peak_amp_thres=3
    # indi_session = rips_df["session"] <= 2
    # indi_peak_amp = peak_amp_thres < rips_df["ripple_peak_amplitude_sd"]
    # phase

    _indi_fixation = (
        rips_df["phase"] == "Fixation"
    )  # indi_fixation.sum() # 352 / 1 = 352
    _indi_encoding = rips_df["phase"] == "Encoding"  # 737 / 2 = 368.5
    _indi_maintanence = rips_df["phase"] == "Maintenance"  # 1427 / 3 = 475.7
    _indi_retrieval = rips_df["phase"] == "Retrieval"  # 1000 / 2 = 500
    indi_phases = [_indi_fixation, _indi_encoding, _indi_maintanence, _indi_retrieval]
    # correct
    indi_correct = rips_df["correct"] == True  # 3244
    indi_incorrect = rips_df["correct"] == False  # 272
    # set_size
    indi_4 = rips_df["set_size"] == 4.0  # 1338
    indi_6 = rips_df["set_size"] == 6.0  # 1085
    indi_8 = rips_df["set_size"] == 8.0  # 1093
    # # IoU
    # indi_IoU = rips_df["IoU"] == 0
    return dict(
        # session=indi_session,
        # peak_amp=indi_peak_amp,
        phases=indi_phases,
        correct=indi_correct,
        incorrect=indi_incorrect,
        _4=indi_4,
        _6=indi_6,
        _8=indi_8,
        # IoU=indi_IoU,
    )

long_percs = 100 * rips_df.pivot_table(columns=["phase", "correct"], aggfunc="mean")[PHASES].T["is_long"]

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
for i_ax, (ax, is_correct) in enumerate(zip(axes, (False, True))):
    sns.histplot(
            data=rips_df[rips_df.correct == is_correct],
            x="duration_ms",
            hue="phase",
            hue_order=PHASES,
            stat="probability",
            kde=True,
            common_norm=False,
            log_scale=True,
            ax=ax,
            alpha=0.3,
        )
    correct_str = "Correct" if is_correct else "Incorrect"
    ax.set_title(correct_str)
plt.show()


# sample_sizes = [
#     str(int(f))
#     for f in list(
#         rips_df.pivot_table(columns=["phase"], aggfunc="sum")[PHASES].T["n"]
#     )
# ]
# sample_sizes = mngs.general.connect_strs(sample_sizes, filler="-")



def plot_dur_hist(rips_df, indi_d):
    _plot_dur_or_amp_hist(rips_df, indi_d, "duration_ms")

def plot_amp_hist(rips_df, indi_d):
    _plot_dur_or_amp_hist(rips_df, indi_d, "ripple_peak_amplitude_sd")

def _plot_dur_or_amp_hist(rips_df, indi_d, x):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

    correct_str = "Correct"
    # data = rips_df[indi_d["session"] * indi_d["correct"] * indi_d["IoU"] * indi_d["peak_amp"]]
    data = rips_df[rips_df.correct == True]

    import ipdb; ipdb.set_trace()
    if x == "duration_ms":
        # data["is_long"] = data["duration_ms"] > dur_ms_thres
        long_rates = [
            str(round(long_rate, 1))
            for long_rate in list(
                100
                * data.pivot_table(columns=["phase"], aggfunc="mean")[PHASES].T[
                    "is_long"
                ]
            )
        ]
        long_rates = mngs.general.connect_strs(long_rates, filler="-")

    sns.histplot(
        data=data.reset_index(),
        x=x,
        hue="phase",
        hue_order=PHASES,
        stat="probability",
        kde=True,
        common_norm=False,
        log_scale=True,
        ax=ax,
        alpha=0.3,
    )

    # sample size
    data["n"] = 1
    sample_sizes = [
        str(int(f))
        for f in list(
            data.pivot_table(columns=["phase"], aggfunc="sum")[PHASES].T["n"]
        )
    ]
    sample_sizes = mngs.general.connect_strs(sample_sizes, filler="-")

    # title
    if x == "duration_ms":
        ax.set_title(
            f"{correct_str}\nSample size: {sample_sizes}\n{dur_ms_thres}-ms> ripple rate: {long_rates}"
        )
        ax.set_xlim(10, 1000)
    if x == "ripple_peak_amplitude_sd":
        ax.set_title(f"{correct_str}\nSample size: {sample_sizes}")
        ax.set_xlim(1, 20)

    # ax.set_yscale("log")

    fig.legend()

    if x == "duration_ms":    
        return fig, sample_sizes, long_rates
    if x == "ripple_peak_amplitude_sd":    
        return fig, sample_sizes

def plot_inci(rips_df, samp_sizes, n_trials):
    samp_sizes = np.array(samp_sizes.split("-")).astype(int)
    inci_hz = samp_sizes / n_trials / np.array([1, 2, 3, 2])

    fig, ax = plt.subplots()
    ax.bar(PHASES, inci_hz)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Ripple incidence [Hz]")    
    # plt.show()
    return fig

def plot_long_ripple_rates(rips_df, long_rates):
    long_rates = np.array(long_rates.split("-")).astype(float)
    # inci_hz = samp_sizes / n_trials / np.array([1, 2, 3, 2])

    fig, ax = plt.subplots()
    ax.bar(PHASES, long_rates)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Long ripple rate")
    # ax.set_ylim(0, 1)
    # plt.show()
    return fig

if __name__ == "__main__":
    import argparse
    import mngs

    # Parameters
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    # sd = 2.0
    SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]    
    # LONG_RIPPLE_THRES_MS = mngs.io.load("./config/global.yaml")["LONG_RIPPLE_THRES_MS"]    

    # Loads
    rips_df = utils.load_rips()
    # n_rips = rips_df.pivot_table(columns=["correct", "set_size"], aggfunc=sum)\
    #                   .loc["n"].reset_index()
    # n_trials_correct = int(n_trials[n_trials["correct"] == 1].sum()["n"])

    # gets indices
    indi_d = get_indi(rips_df) # peak_amp_thres=3

    # plots
    fig, samp_sizes, long_rates = plot_dur_hist(rips_df, indi_d)
    mngs.io.save(fig, "./tmp/figs/hist/dur.png")
    plt.close()

    fig, samp_sizes = plot_amp_hist(rips_df, indi_d)
    mngs.io.save(fig, "./tmp/figs/hist/amp.png")
    plt.close()    

    fig = plot_inci(rips_df, samp_sizes, n_trials)
    mngs.io.save(fig, "./tmp/figs/bar/inci.png")
    plt.close()    

    fig = plot_long_ripple_rates(rips_df, long_rates)
    mngs.io.save(fig, "./tmp/figs/bar/long_ripple_rates.png")
    plt.close()    
    
