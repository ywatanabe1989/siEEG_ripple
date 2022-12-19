#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-10-21 15:55:37 (ywatanabe)"

import sys

sys.path.append("./externals/ripple_detection/")
from ripple_detection.detectors import Kay_ripple_detector
from ripple_detection.core import filter_ripple_band
import numpy as np
import mngs
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def digitalize_ripple_events(rips_df, i_sub_str, i_session_str):
    lpath_EEG = f"./data/Sub_{i_sub_str}/Session_{i_session_str}/EEG.pkl"
    EEG = mngs.io.load(lpath_EEG)
    channels = list(np.array(EEG.coords["channel"]))
    SAMP_RATE_EEG = 200

    digi_rips = 0 * (EEG.copy()[:, 0, :].astype(int))  # digital ripple time
    indi_sub_session = (rips_df["subject"] == int(i_sub_str)) * (
        rips_df["session"] == int(i_session_str)
    )
    rips_sub_session_df = rips_df[indi_sub_session][["trial number", "center_time"]]

    for i_rip, rip in rips_sub_session_df.iterrows():
        trial_number = int(rip["trial number"]) - 1

        try:
            center_pts = int(rip["center_time"] * SAMP_RATE_EEG)
            digi_rips[trial_number, center_pts] = 1
        except Exception as e:
            # print(e)
            digi_rips[trial_number] = np.nan
            center_pts = np.nan

    spath_digi_rips = lpath_EEG.replace("EEG.pkl", "digital_ripples.pkl")
    mngs.io.save(digi_rips, spath_digi_rips)

    return EEG, digi_rips


if __name__ == "__main__":
    from glob import glob
    from natsort import natsorted
    import re

    sub_dirs = natsorted(glob("./data/Sub_*"))
    indi_subs_str = [re.match("./data/Sub_\w{2}", sd).string[-2:] for sd in sub_dirs]

    rips_df = mngs.io.load("./tmp/rips_df.csv")

    EEG_8_COMMON_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2", "A1", "A2"]

    dd = mngs.general.listed_dict()

    for i_sub_str in indi_subs_str:

        """
        i_sub_str = "01"
        """
        session_dirs = natsorted(glob(f"./data/Sub_{i_sub_str}/*"))
        indi_sessions_str = [
            re.match("./data/Sub_\w{2}/Session_\w{2}", sd).string[-2:]
            for sd in session_dirs
        ]

        for i_session_str in indi_sessions_str:
            """
            i_session_str = "01"
            """
            EEG, digi_rips = digitalize_ripple_events(rips_df, i_sub_str, i_session_str)

            EEG = np.array(EEG.loc[:, EEG_8_COMMON_CHANNELS, :]) # slice for 8 channels

            # Ear-referencing
            EEG_odd = EEG[:, mngs.general.search(["1", "3", "5"], EEG_8_COMMON_CHANNELS)[0], :]
            EEG_even = EEG[:, mngs.general.search(["2", "4", "6"], EEG_8_COMMON_CHANNELS)[0], :]
            EEG_odd = EEG_odd[:,:-1,:] - EEG_odd[:,-1,:][:,np.newaxis,:]
            EEG_even = EEG_even[:,:-1,:] - EEG_even[:,-1,:][:,np.newaxis,:]
            EEG = np.concatenate([EEG_odd, EEG_even], axis=1)

            digi_rips = np.array(digi_rips)

            n_trials = len(EEG)
            for i_trial in range(n_trials):
                dd["subject"].append(i_sub_str)
                dd["session"].append(i_session_str)
                dd["trial"].append(f"{i_trial:02d}")
                dd["EEG"].append(EEG[i_trial])
                dd["Ripples"].append(digi_rips[i_trial])

    ds = pd.DataFrame(dd)
    mngs.io.save(ds, "./tmp/dataset.pkl")

    ds_balanced = ds[(ds["session"] == "01") + (ds["session"] == "02")]

    
        
