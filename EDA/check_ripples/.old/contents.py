#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-22 12:57:17 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

roi = "PHR"
sub = "02"
session = "01"
trials_info = mngs.io.load(f"./data/Sub_{sub}/Session_{session}/trials_info.csv")
rips_df = mngs.io.load(f"./tmp/rips_df/common_average_2.0_SD_{roi}.csv")\
    .rename(columns={"Unnamed: 0": "trial_number"})

for i_trial in range(len(trials_info)):
    cols = np.array(["FE", "FM", "FR", "EM", "ER", "MR", "letter"])
    corr_df = pd.DataFrame(
        columns=cols,
        data=np.nan * np.ones(len(cols))[np.newaxis,:],
        )

    spike_times = mngs.io.load(f"./data/Sub_{sub}/Session_{session}/spike_times_{roi}.pkl")\
        [i_trial].replace({"": np.nan})

    rips = rips_df[
        (rips_df["subject"] == int(sub)) * \
        (rips_df["session"] == int(session)) * \
        (rips_df["trial_number"] == i_trial + 1)
        ]

    for i_rip, rip in rips.iterrows():

        ss = rip.start_time - 6
        ee = rip.end_time - 6

        spike_times[((ss < spike_times) * (spike_times < ee))]
        pattern = .sum(axis=0)

        
        spike_times -= 6

        rip.end_time  
        print(rip)
    
np.unique(df["probe_letter"], return_counts=True)

ls "./data/Sub_02/Session_01/"
