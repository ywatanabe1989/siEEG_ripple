#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-21 10:59:27 (ywatanabe)"

import re
from glob import glob
from bisect import bisect_left, bisect_right
import mngs
import numpy as np
from natsort import natsorted
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from itertools import combinations
import warnings


# def to_bipolar(ECoG):
#     bi = pd.DataFrame()
#     cols = ECoG.columns
#     for comb in combinations(cols, 2):
#         col1 = comb[0]
#         col2 = comb[1]
#         with warnings.catch_warnings():
#             warnings.simplefilter(
#                 action="ignore", category=pd.errors.PerformanceWarning
#             )
#             bi[f"{col1}-{col2}"] = ECoG[col1] - ECoG[col2]
#     return bi


def calc_synchronicity(sub, session, i_trial, sd, iEEG_positions_str, synchronous_window_ms=200):    
    # Loads
    # trial_info = mngs.io.load(
    #     f"./data/Sub_{sub}/Session_{session}/trials_info.csv"
    # ).iloc[i_trial]
    # sEEG = mngs.io.load(f"./data/Sub_{sub}/Session_{session}/EEG.pkl")[i_trial]    
    # iEEG = mngs.io.load(f"./data/Sub_{sub}/Session_{session}/iEEG_{iEEG_positions_str}.pkl")[i_trial]
    # iEEG = to_bipolar(pd.DataFrame(iEEG).copy().T).T

    # if iEEG.shape == (0,0):
    #     return np.nan * np.zeros(int(8 * SAMP_RATE_iEEG))

    # iEEG_ripple_band = np.array(
    #     mngs.dsp.bandpass(
    #         torch.tensor(np.array(iEEG)), SAMP_RATE_iEEG, low_hz=LOW_HZ, high_hz=HIGH_HZ
    #     )
    # ).squeeze()

    N_SAMP_PTS = 16000
    time = np.arange(N_SAMP_PTS) / SAMP_RATE_iEEG - 6 # iEEG.shape[-1]

    # rips_df = rips_df[
    #     (rips_df["subject"] == int(sub))
    #     * (rips_df["session"] == int(session))
    #     * (rips_df["trial number"] == int(i_trial + 1))
    # ]

    spike_times = mngs.io.load(f"./data/Sub_{sub}/Session_{session}/spike_times_{iEEG_positions_str}.pkl")[
        i_trial
    ].T

    if spike_times.shape == (0,0):
        return np.nan * time

    else:
        # prepares spike times
        spike_times_digi = np.zeros([len(spike_times), len(time)], dtype=int)
        for i_unit, (_, unit_spike_times) in enumerate(spike_times.iterrows()):
            unit_spike_times = unit_spike_times[np.where(unit_spike_times != "")[0]]  # + 6
            for tt in unit_spike_times:
                i_ins = bisect_right(time, tt)
                if i_ins == len(time):
                    i_ins -= 1
                    
                spike_times_digi[i_unit, i_ins] = 1
        spike_times_digi_tens = torch.tensor(spike_times_digi.astype(np.float32))


        # counts unit firings in sliding windows
        synchronous_window_pts = int(synchronous_window_ms / 1000 * SAMP_RATE_iEEG) - 1
        filters = torch.ones(1, synchronous_window_pts)
        synchronicity = (
            torch.nn.functional.conv1d(
                spike_times_digi_tens.unsqueeze(1),
                filters.unsqueeze(0),
                padding=int(filters.shape[-1] / 2),
            )
            .squeeze()
            .clip(0, 1)
            )
        if synchronicity.ndim == 1:
            synchronicity = synchronicity.unsqueeze(0)

        synchronicity = synchronicity.mean(axis=0)

        # normalizes the synchronicity based on the fixation period
        baseline_sec = 1
        baseline_pts = int(baseline_sec * SAMP_RATE_iEEG)
        synchronicity_z = (
            synchronicity - (synchronicity[:baseline_pts].mean())
        ) / synchronicity[:baseline_pts].std()

        # print(synchronicity) # tensor(0.0748)
        return synchronicity_z
    # mngs.io.save(
    #     synchronicity_z,
    #     f"./tmp/synchronicity/{sd}_SD/Sub_{int(sub):02d}_Session_{int(session):02d}_Trial_{i_trial+1:02d}.png",
    # )
    



    # plt.plot(ratio)
    # plt.show()

    # # Gets information
    # set_size = trial_info["set_size"]
    # is_match = trial_info["match"]
    # is_correct = trial_info["correct"]
    # response_time = trial_info["response_time"]
    # probe_letter = trial_info["probe_letter"]

    # # mngs.plt.configure_mpl(plt, figscale=2.5)
    # fig, axes = plt.subplots(nrows=4, sharex=True)
    # # sns.lineplot(time, iEEG, label="Raw", ax=axes[0])

    # for i_ieeg in range(len(iEEG)):
    #     ieeg = iEEG.iloc[i_ieeg]
    #     axes[0].plot(time, ieeg, label="Raw", alpha=0.3, linewidth=0.1)

    # for i_ieeg in range(len(iEEG_ripple_band)):
    #     ieeg_ripple_band = iEEG_ripple_band[i_ieeg]
    #     axes[1].plot(
    #         time,
    #         ieeg_ripple_band,
    #         label=f"Ripple band ({LOW_HZ}-{HIGH_HZ} Hz)",
    #         alpha=0.3,
    #         linewidth=0.1,
    #     )

    # events = np.array(spike_times.replace({"": -100})).round(3)

    # axes[2].eventplot(
    #     events, lineoffsets=list(np.arange(len(spike_times))), linewidths=1
    # )
    # axes[3].plot(
    #     time, recruited_ratio_z, label="# of recruited cells within 200 ms (zscore)"
    # )
    # # fills ripple area
    # for ax in axes:
    #     for i_rip, rip in rips_df.iterrows():
    #         x_rip_start = rip.start_time - 6
    #         x_rip_end = rip.end_time - 6
    #         ax.axvspan(
    #             x_rip_start,
    #             x_rip_end,
    #             alpha=0.1,
    #             color="red",
    #             zorder=1000,
    #         )
    #     ax.axvline(x=-5, color="gray", linestyle="dotted")
    #     ax.axvline(x=-3, color="gray", linestyle="dotted")
    #     ax.axvline(x=0, color="gray", linestyle="dotted")
    #     ax.axvline(x=response_time, color="green")

    # axes[-1].set_ylim(-3, 10)
    # axes[-1].set_xlim(-5.9, 1.9)

    # correct_str = "Correct" if is_correct else "Incorrect"
    # title = f"Set size: {int(set_size)}; {correct_str}\nSubject {int(sub):02d}; Session {int(session):02d}; Trial {i_trial+1:02d}"
    # axes[0].set_title(title)
    # axes[0].set_ylabel("Raw [uV]")
    # axes[1].set_ylabel("Ripple band [uV]")
    # axes[2].set_ylabel("Unit #")
    # axes[3].set_ylabel("# of recruited units\nwithin 200 ms\n(zscore)")
    # axes[-1].set_xlabel("Time from probe [s]")

    # # plt.show()
    # mngs.io.save(
    #     fig,
    #     f"./tmp/recruited_ratio/{sd}_SD/Sub_{int(sub):02d}_Session_{int(session):02d}_Trial_{i_trial+1:02d}.png",
    # )
    # plt.close()


if __name__ == "__main__":
    # Parameters
    sd = 2.0
    iEEG_positions_str = "PHL"
    # rips_df = mngs.io.load(f"./tmp/rips_df_bi_{sd}_SD_{iEEG_positions_str}.csv")
    SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
    LOW_HZ, HIGH_HZ = 80, 140

    for iEEG_POSITIONS in ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]:
        iEEG_POSITIONS = [iEEG_POSITIONS]
        # iEEG_POSITIONS = ["PHL"] # ["ECL", "ECR"] # ["PHL", "PHR"]
        iEEG_POSITIONS_STR = mngs.general.connect_strs(iEEG_POSITIONS)    
    
        sub_dirs = natsorted(glob("./data/Sub_*"))
        subs = [re.match("./data/Sub_\w{2}", sd).string[-2:] for sd in sub_dirs]
        for sub in subs:
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
                sync_z_session = []
                for i_trial in range(n_trials):
                    sync_z = calc_synchronicity(sub, session_str, i_trial, sd, iEEG_POSITIONS_STR)
                    sync_z_session.append(sync_z)
                sync_z_session = np.vstack(sync_z_session)
                import ipdb; ipdb.set_trace()                
                # mngs.io.save(
                #     sync_z_session,
                #     f"./data/Sub_{sub}/Session_{session_str}/sync_z/{iEEG_POSITIONS_STR}.npy"
                # )

