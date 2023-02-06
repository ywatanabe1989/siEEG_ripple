#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-10 12:40:26 (ywatanabe)"

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
from itertools import combinations
import warnings
from tqdm import tqdm

LOW_HZ = 80
HIGH_HZ = 140


def to_bipolar(ECoG):
    bi = pd.DataFrame()
    cols = ECoG.columns
    for comb in combinations(cols, 2):
        col1 = comb[0]
        col2 = comb[1]
        with warnings.catch_warnings():
            warnings.simplefilter(
                action="ignore", category=pd.errors.PerformanceWarning
            )
            bi[f"{col1}-{col2}"] = ECoG[col1] - ECoG[col2]
    return bi


def detect_ripples(i_sub_str, i_session_str, sd, iEEG_positions_str):
    global iEEG, iEEG_ripple_band_passed, SAMP_RATE_iEEG
    SAMP_RATE_iEEG = 2000

    trials_info = mngs.io.load(
        f"./data/Sub_{i_sub_str}/Session_{i_session_str}/trials_info.csv"
    )
    trials_info["set_size"]
    trials_info["correct"] = trials_info["correct"].replace({0: False, 1: True})
    trials_info["match"] = trials_info["match"].replace(
        {1: "match IN", 2: "mismatch OUT"}
    )

    if iEEG_positions_str == "PHL_PHR":
        iEEG = mngs.io.load(f"./data/Sub_{i_sub_str}/Session_{i_session_str}/iEEG.pkl")
    else:
        iEEG = mngs.io.load(f"./data/Sub_{i_sub_str}/Session_{i_session_str}/iEEG_{iEEG_positions_str}.pkl")

    # to bipolar
    iEEG_bi = []
    for i_iEEG in range(len(iEEG)):
        iEEG_2D = iEEG[i_iEEG]
        iEEG_2D_bi = np.array(to_bipolar(pd.DataFrame(iEEG_2D).T).T)
        iEEG_bi.append(iEEG_2D_bi)
    iEEG = np.array(iEEG_bi)

    iEEG_ripple_band_passed = np.array(
        mngs.dsp.bandpass(
            torch.tensor(np.array(iEEG).astype(np.float32)),
            SAMP_RATE_iEEG,
            low_hz=LOW_HZ,
            high_hz=HIGH_HZ,
        )
    )

    # n_chs = iEEG_ripple_band_passed.shape[1]
    time_iEEG = np.arange(iEEG.shape[-1]) / SAMP_RATE_iEEG
    speed = 0 * time_iEEG

    rip_dfs = []
    for i_trial in range(len(iEEG_ripple_band_passed)):
        rip_df = Kay_ripple_detector(
            time_iEEG,
            iEEG_ripple_band_passed[i_trial].T,
            speed,
            SAMP_RATE_iEEG,
            minimum_duration=0.010,
            zscore_threshold=sd,
        )
        rip_df["trial number"] = i_trial + 1

        # append ripple band filtered iEEG traces
        ripple_band_iEEG_traces = []
        for i_rip, rip in rip_df.reset_index().iterrows():
            start_pts = int(rip["start_time"] * SAMP_RATE_iEEG)
            end_pts = int(rip["end_time"] * SAMP_RATE_iEEG)
            ripple_band_iEEG_traces.append(
                iEEG_ripple_band_passed[i_trial][:, start_pts:end_pts]
            )
        rip_df["ripple band iEEG trace"] = ripple_band_iEEG_traces

        # ripple peak amplitude
        ripple_peak_amplitude = [
            np.abs(rbt).max(axis=-1) for rbt in ripple_band_iEEG_traces
        ]
        ripple_band_baseline_sd = iEEG_ripple_band_passed[i_trial].std(axis=-1)
        rip_df["ripple_peak_amplitude_sd"] = [
            (rpa / ripple_band_baseline_sd).mean() for rpa in ripple_peak_amplitude
        ]

        rip_df["ripple_amplitude_sd"] = [
            (np.abs(rbt).mean(axis=-1) / ripple_band_baseline_sd).mean()
            for rbt in rip_df["ripple band iEEG trace"]
        ]

        rip_dfs.append(rip_df)

    rip_dfs = pd.concat(rip_dfs)

    # set trial number
    rip_dfs = rip_dfs.set_index("trial number")
    trials_info["trial number"] = trials_info["trial_number"]
    del trials_info["trial_number"]
    trials_info = trials_info.set_index("trial number")

    # adds info
    transfer_keys = [
        "set_size",
        "match",
        "correct",
        "response_time",
    ]
    for k in transfer_keys:
        rip_dfs[k] = trials_info[k]

    return rip_dfs


def add_phase(rip_df):
    rip_df["center_time"] = (rip_df["start_time"] + rip_df["end_time"]) / 2
    rip_df["phase"] = None
    rip_df.loc[rip_df["center_time"] < 1, "phase"] = "Fixation"
    rip_df.loc[
        (1 < rip_df["center_time"]) & (rip_df["center_time"] < 3), "phase"
    ] = "Encoding"
    rip_df.loc[
        (3 < rip_df["center_time"]) & (rip_df["center_time"] < 6), "phase"
    ] = "Maintenance"
    rip_df.loc[6 < rip_df["center_time"], "phase"] = "Retrieval"
    return rip_df


def plot_traces(SAMP_RATE_iEEG):
    plt.close()
    plot_start_sec = 0
    plot_dur_sec = 8

    plot_dur_pts = plot_dur_sec * SAMP_RATE_iEEG
    plot_start_pts = plot_start_sec * SAMP_RATE_iEEG

    i_ch = np.random.randint(iEEG_ripple_band_passed.shape[1])
    i_trial = 0

    fig, axes = plt.subplots(2, 1, sharex=True)
    lw = 1

    SAMP_RATE_iEEG = 2000
    time_iEEG = (np.arange(iEEG.shape[-1]) / SAMP_RATE_iEEG) - 6

    axes[0].plot(
        time_iEEG[plot_start_pts : plot_start_pts + plot_dur_pts],
        iEEG[0][i_ch, plot_start_pts : plot_start_pts + plot_dur_pts],
        linewidth=lw,
        label="Raw LFP",
    )

    axes[1].plot(
        time_iEEG[plot_start_pts : plot_start_pts + plot_dur_pts],
        iEEG_ripple_band_passed[0][
            i_ch, plot_start_pts : plot_start_pts + plot_dur_pts
        ],
        linewidth=lw,
        label="Ripple-band-passed LFP",
    )
    # fille ripple time
    rip_plot_df = rip_df[
        (rip_df["trial number"] == i_trial + 1)
        & (plot_start_sec < rip_df["start_time"])
        & (rip_df["end_time"] < plot_start_sec + plot_dur_sec)
    ]

    for ax in axes:
        for ripple in rip_plot_df.itertuples():
            ax.axvspan(
                ripple.start_time - 6,
                ripple.end_time - 6,
                alpha=0.1,
                color="red",
                zorder=1000,
            )
            ax.axvline(x=-5, color="gray", linestyle="dotted")
            ax.axvline(x=-3, color="gray", linestyle="dotted")
            ax.axvline(x=0, color="gray", linestyle="dotted")

    axes[-1].set_xlabel("Time from probe [sec]")

    # plt.show()
    mngs.io.save(plt, "./tmp/ripple_repr_traces_sub_01_session_01_trial_01-50.png")


def plot_hist(rip_df):
    plt.close()
    rip_df["dur_time"] = rip_df["end_time"] - rip_df["start_time"]
    rip_df["dur_ms"] = rip_df["dur_time"] * 1000

    plt.hist(rip_df["dur_ms"], bins=100)
    plt.xlabel("Ripple duration [sec]")
    plt.ylabel("Count of ripple events")
    # plt.show()
    mngs.io.save(plt, "./tmp/ripple_count_sub_01_session_01_trial_01-50.png")


def calc_rip_incidence_hz(rip_df):
    rip_df["n"] = 1
    rip_incidence_hz = pd.concat(
        [
            rip_df[rip_df["phase"] == "Fixation"]
            .pivot_table(columns=["trial number"], aggfunc="sum")
            .T["n"]
            / 1,
            rip_df[rip_df["phase"] == "Encoding"]
            .pivot_table(columns=["trial number"], aggfunc="sum")
            .T["n"]
            / 2,
            rip_df[rip_df["phase"] == "Maintenance"]
            .pivot_table(columns=["trial number"], aggfunc="sum")
            .T["n"]
            / 3,
            rip_df[rip_df["phase"] == "Retrieval"]
            .pivot_table(columns=["trial number"], aggfunc="sum")
            .T["n"]
            / 2,
        ],
        axis=1,
    ).fillna(0)
    rip_incidence_hz.columns = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
    return rip_incidence_hz


if __name__ == "__main__":
    from glob import glob
    from natsort import natsorted
    import re

    iEEG_positions_str = "PHL" # "ECL_ECR"  # ["PHL_PHR"]

    for sd in [2.0, 3.0, 4.0, 5.0]:



        sub_dirs = natsorted(glob("./data/Sub_*"))
        indi_subs_str = [
            re.match("./data/Sub_\w{2}", sd).string[-2:] for sd in sub_dirs
        ]


        rips_df = pd.DataFrame()

        for i_sub_str in indi_subs_str:
            try:
                """
                i_sub_str = "01"
                """
                session_dirs = natsorted(glob(f"./data/Sub_{i_sub_str}/*"))
                indi_sessions_str = [
                    re.match("./data/Sub_\w{2}/Session_\w{2}", sd).string[-2:]
                    for sd in session_dirs
                ]

                for i_session_str in tqdm(indi_sessions_str):
                    """
                    i_session_str = "01"
                    """

                    rip_df = add_phase(
                        detect_ripples(
                            i_sub_str=i_sub_str,
                            i_session_str=i_session_str,
                            sd=sd,
                            iEEG_positions_str=iEEG_positions_str,
                        )
                    )
                    rip_df["subject"] = i_sub_str
                    rip_df["session"] = i_session_str

                    rips_df = pd.concat([rips_df, rip_df])
                    
            except Exception as e:
                print(e)

        mngs.io.save(rips_df, f"./tmp/rips_df_bi_{sd}_SD_{iEEG_positions_str}.csv")


# EOF
