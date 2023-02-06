#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-21 14:44:07 (ywatanabe)"

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

def count_n_trials(subs):
    sessions = ["01", "02"]
    dfs = []
    for sub in subs:
        sub = f"{sub:02d}"
        for session in sessions:
            trials_info = mngs.io.load(
                f"./data/Sub_{sub}/Session_{session}/trials_info.csv"
            )
            trials_info["subject"] = sub
            trials_info["session"] = session
            dfs.append(trials_info)
    dfs = pd.concat(dfs).reset_index()
    dfs = dfs[["subject", "session", "trial_number", "correct", "set_size"]]
    dfs["n"] = 1
    dfs = dfs[np.array(dfs["session"]).astype(int) <= 2]
    dfs = dfs.pivot_table(
        columns=["correct", "set_size"], aggfunc="sum"
    ).T.reset_index()[["correct", "set_size", "n"]]
    return dfs

if __name__ == "__main__":
    import argparse
    import mngs


    # Parameters
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    sd = 2.0
    SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]

    # Loads
    rips_df = []
    subs = []
    for sub, roi in ROIs.items():
        _rips_df = mngs.io.load(f"./tmp/rips_df/common_average_{sd}_SD_{roi}.csv")
        _rips_df = _rips_df[_rips_df["subject"] == sub]
        rips_df.append(_rips_df)
        subs.append(sub)
    rips_df = pd.concat(rips_df)

    n_trials = count_n_trials(subs)

    # indices
    indi_session = rips_df["session"] <= 2
    # peak amp
    peak_amp_thres = 3
    indi_peak_amp = peak_amp_thres < rips_df["ripple_peak_amplitude_sd"] 
    # phase
    phases = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
    indi_fixation = (
        rips_df["phase"] == "Fixation"
    )  # indi_fixation.sum() # 352 / 1 = 352
    indi_encoding = rips_df["phase"] == "Encoding"  # 737 / 2 = 368.5
    indi_maintanence = rips_df["phase"] == "Maintenance"  # 1427 / 3 = 475.7
    indi_retrieval = rips_df["phase"] == "Retrieval"  # 1000 / 2 = 500
    indi_phases = [indi_fixation, indi_encoding, indi_maintanence, indi_retrieval]
    # correct
    indi_correct = rips_df["correct"] == True  # 3244
    indi_incorrect = rips_df["correct"] == False  # 272
    # set_size
    indi_4 = rips_df["set_size"] == 4.0  # 1338
    indi_6 = rips_df["set_size"] == 6.0  # 1085
    indi_8 = rips_df["set_size"] == 8.0  # 1093
    # IoU
    indi_IoU = rips_df["IoU"] == 0

    # fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)    
    for i_correct, _indi_correct in enumerate([indi_incorrect, indi_correct]):
        correct_str = "Incorrect" if bool(i_correct) == 0 else "Correct"
        if correct_str == "Incorrect":
            continue
        
        # ax = axes[i_correct]
        ax = axes
        
        data = rips_df[
            indi_session * _indi_correct * indi_IoU * indi_peak_amp
        ]

        if data.shape == (0,):
            continue

        sns.histplot(
            data=data.reset_index(),
            x="ripple_peak_amplitude_sd",
            hue="phase",
            hue_order=phases,
            stat="probability",
            kde=True,
            common_norm=False,
            log_scale=True,
            ax=ax,
            alpha=0.3,
            )
        
        # sample size
        data["n"] = 1
        sample_sizes = [str(int(f)) for f in
                        list(data.pivot_table(columns=["phase"], aggfunc="sum")[phases].T["n"])]
        sample_sizes = mngs.general.connect_strs(sample_sizes, filler="-")

        # title
        ax.set_title(f"{correct_str}\nSample size: {sample_sizes}")
        ax.set_xlim(1, 20)
        # ax.set_yscale("log")
        

    fig.legend()

    mngs.io.save(fig, "./tmp/figs/hist/peak_amp.png")
    # plt.show()
