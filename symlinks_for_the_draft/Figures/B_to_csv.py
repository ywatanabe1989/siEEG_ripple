#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-05 09:46:39 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
from eeg_ieeg_ripple_clf import utils
import pandas as pd

# Parameters
ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
# sub = "06"

# Loads
rips_df = utils.load_rips()
# rips_df = rips_df[rips_df.subject == int(sub)]

df = pd.DataFrame()
for phase in PHASES:
    dur_ss_4_phase = \
        rips_df[(rips_df.set_size == 4) * rips_df.phase == phase]["duration_ms"]
    dur_ss_6_phase = \
        rips_df[(rips_df.set_size == 6) * rips_df.phase == phase]["duration_ms"]
    dur_ss_8_phase = \
        rips_df[(rips_df.set_size == 8) * rips_df.phase == phase]["duration_ms"]
    
    dur_phase = mngs.general.force_dataframe(
        {f"{phase} setsize 4": dur_ss_4_phase,
         f"{phase} setsize 6": dur_ss_6_phase,
         f"{phase} setsize 8": dur_ss_8_phase,
         })
    df = pd.concat([df, dur_phase], axis=1)
mngs.io.save(df, "./tmp/figs/B/x_duration_y_ripple_probability.csv")


