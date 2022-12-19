#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-10 14:09:18 (ywatanabe)"

import mngs
import pandas as pd
import numpy as np
import warnings

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
LONG_RIPPLE_THRES_MS = mngs.io.load("./config/global.yaml")["LONG_RIPPLE_THRES_MS"]
LARGE_RIPPLE_THRES_SD = mngs.io.load("./config/global.yaml")["LARGE_RIPPLE_THRES_SD"]
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]

def load_rips(sd=2, from_pkl=True, only_correct=True):
    if from_pkl:
        return mngs.io.load("./tmp/rips.pkl")
    
    rips_df = []
    for sub, roi in ROIs.items():
        # rips_df_roi = mngs.io.load(
        #     f"./tmp/rips_df/common_average_2.0_SD_{roi}.csv"
        #     # f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl"            
        # ).rename(columns={"Unnamed: 0": "trial_number"})
        rips_df_roi = mngs.io.load(
            f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl"
            # f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl"            
        ).reset_index()
        rips_df_roi = rips_df_roi.rename(columns={"index": "trial_number"})
        # rips_df_roi = rips_df_roi[rips_df_roi["subject"] == int(sub)]
        rips_df_roi = rips_df_roi[rips_df_roi["subject"] == f"{sub:02d}"]        
        rips_df_roi["ROI"] = roi
        rips_df.append(rips_df_roi)
    rips_df = pd.concat(rips_df)

    # rips_df = rips_df[rips_df["session"] <= 2]
    rips_df.reset_index(inplace=True)
    del rips_df["index"]

    # firing patterns
    rips_df["firing_pattern"] = [_determine_firing_patterns(rips_df.iloc[ii])
                                 for ii in range(len(rips_df))]
    rips_df["n_firings"] = rips_df["firing_pattern"].apply(sum)

    # duration
    rips_df["duration_ms"] = (rips_df.end_time - rips_df.start_time) * 1000
    rips_df["is_long"] = LONG_RIPPLE_THRES_MS < rips_df["duration_ms"]

    # amplitude    
    rips_df["is_large"] = LARGE_RIPPLE_THRES_SD < rips_df["ripple_peak_amplitude_sd"]         
    rips_df = rips_df[sd <= rips_df.ripple_peak_amplitude_sd]

    # IoU
    rips_df = rips_df[rips_df.IoU <= IoU_RIPPLE_THRES]

    # etc
    rips_df["n"] = 1

    # correct
    if only_correct:
        rips_df = rips_df[rips_df.correct == True]

    # session
    rips_df = rips_df[rips_df.session.astype(int) <= SESSION_THRES]
    
    return rips_df

def _determine_firing_patterns(rip):
    try:
        sub = f"{int(rip.subject):02d}"
    except:
        return np.nan

    session = f"{int(rip.session):02d}"

    try:
        roi = ROIs[int(sub)]
    except:
        return np.nan

    i_trial = int(rip.trial_number) - 1

    spike_times = mngs.io.load(
        f"./data/Sub_{sub}/Session_{session}/spike_times_{roi}.pkl"
    )[i_trial].replace({"": np.nan})
    spike_pattern = spike_times[
        ((rip.start_time - 6 < spike_times) * (spike_times < rip.end_time - 6))
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        spike_pattern = (~spike_pattern.isna()).sum()

    return spike_pattern

if __name__ == "__main__":
    rips_df = load_rips(from_pkl=False)
    mngs.io.save(rips_df, "./tmp/rips.pkl")

    rips_df.pivot_table(columns=["subject"], aggfunc=sum).T.n
