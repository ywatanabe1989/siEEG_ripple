#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-31 11:25:08 (ywatanabe)"

import mngs
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
LONG_RIPPLE_THRES_MS = mngs.io.load("./config/global.yaml")["LONG_RIPPLE_THRES_MS"]
LARGE_RIPPLE_THRES_SD = mngs.io.load("./config/global.yaml")["LARGE_RIPPLE_THRES_SD"]
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]


def load_rips(
    sd=2,
    from_pkl=False,
    only_correct=False,
    ROI=None,
    extracts_firing_patterns=False,
):
    """
    When ROI=None, ROIs written in ./config/ripple_detectable_rois.yaml are loaded.
    """
    if ROI is None:
        out = load_rips_rois_in_config(
            sd=sd, from_pkl=from_pkl, only_correct=only_correct
        )
    else:
        rips_df_roi = mngs.io.load(
            f"./tmp/rips_df/common_average_2.0_SD_{ROI}.pkl"
        ).reset_index()  # Nonetype object is not iterable
        rips_df_roi = rips_df_roi[~rips_df_roi.subject.isna()]
        rips_df_roi = rips_df_roi.rename(columns={"index": "trial_number"})
        rips_df_roi["duration_ms"] = 1000 * (
            rips_df_roi.end_time - rips_df_roi.start_time
        )
        rips_df_roi["ROI"] = ROI
        out = rips_df_roi

    if extracts_firing_patterns:
        print("extracting firing patterns...")
        out["firing_pattern"] = [
            _determine_firing_patterns(out.iloc[ii]) for ii in tqdm(range(len(out)))
        ]

    return out

def calc_unit_participation_rate(row):
    return (row >= 1).mean()


def load_rips_rois_in_config(sd=2, from_pkl=False, only_correct=False):
    if from_pkl:
        return mngs.io.load("./tmp/rips.pkl")

    rips_df = []
    for sub, roi in ROIs.items():
        rips_df_roi = mngs.io.load(
            f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl"
            # f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl"
        ).reset_index()

        rips_df_roi = rips_df_roi[~rips_df_roi["subject"].isna()]
        rips_df_roi = rips_df_roi.rename(columns={"index": "trial_number"})
        rips_df_roi = rips_df_roi[rips_df_roi["subject"] == f"{int(sub):02d}"]
        rips_df_roi["ROI"] = roi
        rips_df.append(rips_df_roi)
    rips_df = pd.concat(rips_df)

    # rips_df = rips_df[rips_df["session"] <= 2]
    rips_df.reset_index(inplace=True)
    del rips_df["index"]

    # firing patterns
    rips_df["firing_pattern"] = [
        _determine_firing_patterns(rips_df.iloc[ii]) for ii in range(len(rips_df))
    ]
    rips_df["n_firings"] = rips_df["firing_pattern"].apply(sum)
    rips_df["population_burst_rate"] = rips_df["firing_pattern"].apply(np.mean)

    rips_df["unit_participation_rate"] = rips_df["firing_pattern"].apply(calc_unit_participation_rate)        

    # duration
    rips_df["duration_ms"] = 1000 * (rips_df.end_time - rips_df.start_time).astype(
        float
    )
    rips_df["log10(duration_ms)"] = np.log10(rips_df["duration_ms"])
    rips_df["is_long"] = LONG_RIPPLE_THRES_MS < rips_df["duration_ms"]

    # amplitude
    rips_df["log10(ripple_peak_amplitude_sd)"] = np.log10(
        rips_df["ripple_peak_amplitude_sd"]
    ).astype(float)
    rips_df["is_large"] = LARGE_RIPPLE_THRES_SD < rips_df["ripple_peak_amplitude_sd"]
    rips_df = rips_df[sd <= rips_df.ripple_peak_amplitude_sd]

    # IoU
    rips_df = rips_df[rips_df.IoU <= IoU_RIPPLE_THRES]

    # etc
    rips_df["n"] = 1

    # edge effect
    rips_df["center_time"] = (rips_df.end_time + rips_df.start_time) / 2
    rips_df = rips_df[(0.1 < rips_df.center_time) * (rips_df.center_time < 7.9)]
    
    # correct
    if only_correct:
        rips_df = rips_df[rips_df.correct == True]

    # session
    rips_df = rips_df[rips_df.session.astype(int) <= SESSION_THRES]

    # IO balance
    # rips_df["IO_balance"] = rips_df["population_burst_rate"] / rips_df["ripple_amplitude_sd"]
    rips_df["IO_balance"] = rips_df["unit_participation_rate"] / rips_df["ripple_amplitude_sd"]        

    return rips_df


def _determine_firing_patterns(rip):
    sub = f"{int(rip.subject):02d}"

    session = f"{int(rip.session):02d}"

    roi = rip.ROI

    i_trial = int(rip.trial_number) - 1

    spike_times = mngs.io.load(
        f"./data/Sub_{sub}/Session_{session}/spike_times_{roi}.pkl"
    )[i_trial].replace({"": np.nan})
    if rip.start_time is not None:
        spike_pattern = spike_times[
            ((rip.start_time - 6 < spike_times) * (spike_times < rip.end_time - 6))
        ]
    else:
        spike_pattern = spike_times

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        spike_pattern = (~spike_pattern.isna()).sum()

    return spike_pattern


if __name__ == "__main__":
    # rips_df = load_rips(from_pkl=False)
    # mngs.io.save(rips_df, "./tmp/rips.pkl")

    # rips_df.pivot_table(columns=["subject"], aggfunc=sum).T.n

    # "AHR", "PHL", "PHR", "ECL", "ECR", "AR"
    # rips = load_rips(ROI="AHR")

    # rips_df = load_rips(from_pkl=False, only_correct=False, ROI="AHR")
    rips_df = load_rips(from_pkl=False)

    rips_df[rips_df.subject == "01"]
