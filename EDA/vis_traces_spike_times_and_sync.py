#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-20 17:09:17 (ywatanabe)"

import re
from glob import glob
from bisect import bisect_left
import mngs
import numpy as np
from natsort import natsorted
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from itertools import combinations
import warnings

import sys

sys.path.append(".")
from eeg_ieeg_ripple_clf import utils


def plot_traces_spike_times_and_sync(rips_df, sub, session, i_trial, sd, iEEG_ROI):
    # Loads
    trial_info = mngs.io.load(
        f"./data/Sub_{sub}/Session_{session}/trials_info.csv"
    ).iloc[i_trial]
    iEEG = utils.load_iEEG(sub, session, iEEG_ROI)[i_trial]

    if iEEG.shape == (0, 16000):
        iEEG_ripple_band = np.nan * iEEG
    else:
        iEEG_ripple_band = np.array(
            mngs.dsp.bandpass(
                torch.tensor(np.array(iEEG)),
                SAMP_RATE_iEEG,
                low_hz=LOW_HZ,
                high_hz=HIGH_HZ,
            )
        ).squeeze()
    time = np.arange(iEEG.shape[-1]) / SAMP_RATE_iEEG - 6

    rips_df = rips_df[
        (rips_df["subject"] == int(sub))
        * (rips_df["session"] == int(session))
        * (rips_df["trial_number"] == int(i_trial + 1))
    ]

    spike_times = mngs.io.load(
        f"./data/Sub_{sub}/Session_{session}/spike_times_{iEEG_ROI}.pkl"
    )[i_trial].T
    """
    mngs.io.load("./data/Sub_01/Session_03/spike_times_AHL.pkl")
    """    

    spike_times_digi = np.zeros([len(spike_times), len(time)], dtype=int)

    for i_unit, (_, unit_spike_times) in enumerate(spike_times.iterrows()):
        unit_spike_times = unit_spike_times[np.where(unit_spike_times != "")[0]]  # + 6
        for tt in unit_spike_times:
            i_ins = bisect_left(time, tt)
            if i_ins == spike_times_digi.shape[-1]:
                i_ins -= 1

            spike_times_digi[i_unit, i_ins] = 1

    spike_times_digi_tens = torch.tensor(spike_times_digi.astype(np.float32))

    sync_z = mngs.io.load(f"./data/Sub_{sub}/Session_{session}/sync_z/{iEEG_ROI}.npy")[
        i_trial
    ]

    # Gets information
    set_size = trial_info["set_size"]
    is_match = trial_info["match"]
    is_correct = trial_info["correct"]
    response_time = trial_info["response_time"]
    probe_letter = trial_info["probe_letter"]

    # mngs.plt.configure_mpl(plt, figscale=2.5)
    fig, axes = plt.subplots(nrows=4, sharex=True)

    for i_ieeg in range(len(iEEG)):
        ieeg = iEEG[i_ieeg]
        axes[0].plot(time, ieeg, label="Raw", alpha=0.3, linewidth=0.1)

    for i_ieeg in range(len(iEEG_ripple_band)):
        ieeg_ripple_band = iEEG_ripple_band[i_ieeg]
        axes[1].plot(
            time,
            ieeg_ripple_band,
            label=f"Ripple band ({LOW_HZ}-{HIGH_HZ} Hz)",
            alpha=0.3,
            linewidth=0.1,
        )

    events = np.array(spike_times.replace({"": -100})).round(3)
    if events.shape != (0, 0):
        axes[2].eventplot(
            events, lineoffsets=list(np.arange(len(spike_times))), linewidths=1
        )

    axes[3].plot(time, sync_z, label="# of recruited cells within 200 ms (zscore)")
    # fills ripple area
    for i_ax, ax in enumerate(axes):
        for i_rip, rip in rips_df.iterrows():
            x_rip_start = rip.start_time - 6
            x_rip_end = rip.end_time - 6

            if (rip.IoU != 0) * (i_ax == 0):
                _ylim = ax.get_ylim()
                ax.text((x_rip_start + x_rip_end) / 2, _ylim[0], f"{rip.IoU:.2f}")

            ax.axvspan(
                x_rip_start,
                x_rip_end,
                alpha=0.1,
                color="red",
                zorder=1000,
            )
        ax.axvline(x=-5, color="gray", linestyle="dotted")
        ax.axvline(x=-3, color="gray", linestyle="dotted")
        ax.axvline(x=0, color="gray", linestyle="dotted")
        ax.axvline(x=response_time, color="green")

    axes[-1].set_ylim(-3, 10)
    axes[-1].set_xlim(-5.9, 1.9)

    correct_str = "Correct" if is_correct else "Incorrect"
    title = (
        f"Set size: {int(set_size)}; {correct_str}\nSubject {int(sub):02d}; "
        f"Session {int(session):02d}; Trial {i_trial+1:02d}"
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("Raw [uV]")
    axes[1].set_ylabel("Ripple band [uV]")
    axes[2].set_ylabel("Unit #")
    # axes[3].set_ylabel("# of recruited units\nwithin 200 ms\n(zscore)")
    axes[3].set_ylabel("Synchronicity\nwithin 200 ms\n(zscore)")
    axes[-1].set_xlabel("Time from probe [s]")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ROI", "--iEEG_ROI", type=str, default="AHL")
    args = parser.parse_args()

    # Parameters
    sd = 2.0
    # iEEG_ROI = "PHR"

    # for iEEG_ROI in ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]: # ["PHR", "PHL"]: # "PHR",
    iEEG_ROI = args.iEEG_ROI

    rips_df = mngs.io.load(f"./tmp/rips_df/common_average_{sd}_SD_{iEEG_ROI}.csv")
    SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
    LOW_HZ, HIGH_HZ = 80, 140

    sub_dirs = natsorted(glob("./data/Sub_*"))
    subs = [re.match("./data/Sub_\w{2}", sd).string[-2:] for sd in sub_dirs]

    for sub in subs:

        if int(sub) < 5:
            continue
        
        session_dirs = natsorted(glob(f"./data/Sub_{sub}/*"))
        sessions_str = [
            re.match("./data/Sub_\w{2}/Session_\w{2}", sd).string[-2:]
            for sd in session_dirs
        ]
        for session_str in sessions_str:

            trials_info = mngs.io.load(
                f"./data/Sub_{sub}/Session_{session_str}/trials_info.csv"
            )
            n_trials = len(trials_info)
            for i_trial in range(n_trials):
                # if count < 5:
                fig = plot_traces_spike_times_and_sync(
                    rips_df, sub, session_str, i_trial, sd, iEEG_ROI
                )
                mngs.io.save(
                    fig,
                    f"./tmp/figs/traces_spike_times_and_sync/{sd}_SD/{iEEG_ROI}/"
                    f"Sub_{int(sub):02d}/Session_{int(session_str):02d}_Trial_{i_trial+1:02d}.png",
                )
                plt.close()

                #     count += 1
                # else:
                #     continue
