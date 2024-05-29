#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-10 15:33:00 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
from siEEG_ripple import utils
from tqdm import tqdm

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
LONG_RIPPLE_THRES_MS = mngs.io.load("./config/global.yaml")["LONG_RIPPLE_THRES_MS"]
LARGE_RIPPLE_THRES_SD = mngs.io.load("./config/global.yaml")["LARGE_RIPPLE_THRES_SD"]
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]


def load_sim(suffix): # from_pkl=True
    # if from_pkl:
    #     return mngs.io.load("./tmp/sim_df.pkl")

    sim_df_all = mngs.io.load(f"./tmp/sim_df_{suffix}.pkl")
    sim_df_all = sim_df_all[sim_df_all.session.astype(int) <= SESSION_THRES]
    return sim_df_all

if __name__ == "__main__":
    sim_df = load_sim("cons")

    out = utils.sim.add_match_correct_response_time(sim_df)

