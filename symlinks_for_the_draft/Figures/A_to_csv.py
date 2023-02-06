#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-02 12:35:05 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
from siEEG_ripple import utils
import numpy as np
import torch
from bisect import bisect_left 
import pandas as pd

# Parameters
SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
LOW_HZ, HIGH_HZ = 80, 140
sub = "06"
roi = ROIs[int(sub)]
session = "02"
i_trial = 5 - 1

# Loads
iEEG = utils.load_iEEG(sub, session, roi)[i_trial]
time = np.arange(iEEG.shape[-1]) / SAMP_RATE_iEEG - 6
    
iEEG_ripple_band = np.array(
    mngs.dsp.bandpass(
        torch.tensor(np.array(iEEG)),
        SAMP_RATE_iEEG,
        low_hz=LOW_HZ,
        high_hz=HIGH_HZ,
    )
).squeeze()

spike_times = mngs.io.load(
    f"./data/Sub_{sub}/Session_{session}/spike_times_{roi}.pkl"
)[i_trial].T
"""
mngs.io.load("./data/Sub_01/Session_03/spike_times_AHL.pkl")
"""    

spike_times_digi = np.nan * np.zeros([len(spike_times), len(time)], dtype=int)

for i_unit, (_, unit_spike_times) in enumerate(spike_times.iterrows()):
    unit_spike_times = unit_spike_times[np.where(unit_spike_times != "")[0]]  # + 6
    for tt in unit_spike_times:
        i_ins = bisect_left(time, tt)
        if i_ins == spike_times_digi.shape[-1]:
            i_ins -= i_unit

        spike_times_digi[i_unit, i_ins] = i_unit

sync_z = mngs.io.load(f"./data/Sub_{sub}/Session_{session}/sync_z/{roi}.npy")[
    i_trial
]

rips_df = utils.rips.load_rips()
rips_df = rips_df[
        (rips_df["subject"] == f"{int(sub):02d}")
        * (rips_df["session"] == f"{int(session):02d}")
        * (rips_df["trial_number"] == int(i_trial + 1))
    ]

cons_df = utils.rips.load_cons_across_trials()
cons_df = cons_df[
        (cons_df["subject"] == f"{int(sub):02d}")
        * (cons_df["session"] == f"{int(session):02d}")
        * (cons_df["trial_number"] == int(i_trial + 1))
    ]


    
def get_rips_digi(rips_df):
    # rips_df = mngs.io.load(f"./tmp/rips_df/common_average_2.0_SD_{roi}.csv")
    rips_digi = np.zeros(len(time), dtype=int)
    for i_rip, rip in rips_df.iterrows():
        ss = int((rip.start_time) * SAMP_RATE_iEEG)
        ee = int((rip.end_time) * SAMP_RATE_iEEG)
        rips_digi[ss:ee] = 1
    return rips_digi

rips_digi = get_rips_digi(rips_df)
cons_digi = get_rips_digi(cons_df)

# Saves
mngs.io.save(pd.DataFrame(rips_digi).set_index(time), "./tmp/figs/A/ripple_time.csv")
mngs.io.save(pd.DataFrame(cons_digi).set_index(time), "./tmp/figs/A/control_time.csv")
mngs.io.save(pd.DataFrame(np.array(iEEG)).T.set_index(time), "./tmp/figs/A/raw_iEEG.csv")
mngs.io.save(pd.DataFrame(np.array(iEEG_ripple_band)).T.set_index(time), "./tmp/figs/A/ripple_band_iEEG.csv")
mngs.io.save(pd.DataFrame(np.array(spike_times_digi)).T.set_index(time), "./tmp/figs/A/unit_spike_times.csv")
mngs.io.save(pd.DataFrame(sync_z).set_index(time), "./tmp/figs/A/synchronicity.csv")

