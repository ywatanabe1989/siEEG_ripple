#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-07 12:27:26 (ywatanabe)"

import mngs
from glob import glob
import pandas as pd

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

trials_df = pd.DataFrame()
for subject, _ in ROIs.items():
    for session in [1,2]:
        lpath = f"./data/Sub_{subject:02d}/Session_{session:02d}/trials_info.csv"
        _trials_df = mngs.io.load(lpath)
        _trials_df["subject"] = subject
        _trials_df['session'] = session
        trials_df = pd.concat([trials_df, _trials_df])

trials_df[trials_df.subject == 9]
