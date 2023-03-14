#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-01 16:25:30 (ywatanabe)"

import sys

sys.path.append(".")
from siEEG_ripple import utils
import random
import mngs
import pandas as pd
import numpy as np
from utils._load_rips import _determine_firing_patterns, load_rips
from tqdm import tqdm
# from ._load_rips import load_rips


def load_cons_across_trials(
    from_pkl=False, only_correct=False, ROI=None, extracts_firing_patterns=False
):
    mngs.general.fix_seeds(random=random)

    global PHASE2DUR_DICT, PHASE2START_SEC
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"]
    PHASE2DUR_DICT = {phase: dur for phase, dur in zip(PHASES, DURS_OF_PHASES)}
    PHASE2START_SEC = {
        "Fixation": 0,
        "Encoding": 1,
        "Maintenance": 3,
        "Retrieval": 6,
    }

    rips_df = load_rips(
        from_pkl=from_pkl,
        only_correct=only_correct,
        ROI=ROI,
        extracts_firing_patterns=extracts_firing_patterns,
    ).reset_index()
    del rips_df["index"]

    columns = [
            "trial_number",
            "start_time",
            "end_time",
            "set_size",
            "match",
            "correct",
            "response_time",
            "subject",
            "session",
            "ROI",
        ]
    if extracts_firing_patterns:
        columns += ["firing_pattern"]
    rips_df = rips_df[columns]

    cons = rips_df.copy()

    rips_all = cons[["start_time", "end_time"]]

    shuffled_rips_all = rips_all.iloc[
        np.random.permutation(rips_all.index)
    ].reset_index()
    del shuffled_rips_all["index"]

    cons = cons[[col for col in list(cons.columns) if not "_time" in col]]

    cons = pd.concat([cons, shuffled_rips_all], axis=1)

    # trials_uq = rips_df[["subject", "session", "trial_number"]].drop_duplicates()

    if only_correct:
        cons = cons[cons.correct == True]

    cons["center_time"] = ((cons["end_time"] + cons["start_time"]) / 2).astype(float)

    cons["phase"] = None
    for phase, phase_start_s in PHASE2START_SEC.items():
        # cons["phase"][phase_start_s <= cons["center_time"]] = phase
        cons.loc[phase_start_s <= cons["center_time"], "phase"] = phase  # fixme

    return cons


if __name__ == "__main__":
    # cons = load_cons_across_trials(ROI="AHR")  # "AHL"

    # rips_df = load_rips(
    #     from_pkl=False,
    #     only_correct=False,
    #     extracts_firing_patterns=False,
    # ).reset_index()

    cons = load_cons_across_trials(extracts_firing_patterns=True)

    rips_df["center_time"] = (rips_df.start_time + rips_df.end_time) / 2
    sorted(rips_df.center_time.astype(float)) == sorted(cons.center_time) # True

    utils.rips.mk_events_mask(rips_df, "01", "01", "AHL", 0)
