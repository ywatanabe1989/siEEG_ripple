#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-09 23:35:06 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
from eeg_ieeg_ripple_clf import utils

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
LONG_RIPPLE_THRES_MS = mngs.io.load("./config/global.yaml")["LONG_RIPPLE_THRES_MS"]
LARGE_RIPPLE_THRES_SD = mngs.io.load("./config/global.yaml")["LARGE_RIPPLE_THRES_SD"]
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]

def load_dist(suffix): # from_pkl=True
    assert suffix in ["rips", "cons"]
    # if from_pkl:
    #     return mngs.io.load("./tmp/dist_df.pkl")

    dist_df_all = mngs.io.load(f"./tmp/dist_df_{suffix}.pkl")
    return dist_df_all[dist_df_all.session.astype(int) <= SESSION_THRES]

if __name__ == "__main__":
    dist_df = load_dist("rips")
