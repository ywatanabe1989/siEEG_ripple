#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-28 11:25:37 (ywatanabe)"
# detect_ripples.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
import torch
from scripts import utils
from scripts.externals.ripple_detection.ripple_detection.detectors import (
    Kay_ripple_detector,
)
from tqdm import tqdm

"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def calc_iou(a, b):
    """
    Calculate Intersection over Union
    a = [0, 10]
    b = [0, 3]
    calc_iou(a, b) # 0.3
    """
    (a_s, a_e) = a
    (b_s, b_e) = b

    a_len = a_e - a_s
    b_len = b_e - b_s

    abx_s = max(a_s, b_s)
    abx_e = min(a_e, b_e)

    abx_len = max(0, abx_e - abx_s)

    return abx_len / (a_len + b_len - abx_len)


def detect_ripples(i_sub_str, i_session_str, sd, iEEG_ROI):
    """
    1) take the common averaged signal of ROI
    2) detect ripples from the signals of ROI and the common averaged signal
    3) drop ROI ripples based on IoU
    """
    global iEEG, iEEG_ripple_band_passed

    trials_info = mngs.io.load(
        f"./data/Sub_{i_sub_str}/Session_{i_session_str}/trials_info.csv"
    )

    # trials_info["set_size"]
    trials_info["correct"] = trials_info["correct"].replace(
        {0: False, 1: True}
    )

    # sync_z_session = mngs.io.load(
    #     f"./data/Sub_{i_sub_str}/Session_{i_session_str}/sync_z/{iEEG_ROI}.npy"
    # )

    # koko
    iEEG, iEEG_common_ave = utils.load.iEEG(
        i_sub_str, i_session_str, iEEG_ROI, return_common_averaged_signal=True
    )
    # iEEG.shape # (50, 8, 16000)

    if iEEG.shape[1] == 0:  # no channels
        return pd.DataFrame(
            columns=["start_time", "end_time"],
            data=np.array([[np.nan, np.nan]]),
        )

    # Bandpass filtering
    RIPPLE_BANDS = np.vstack(
        [[CONFIG["RIPPLE_LOW_HZ"], CONFIG["RIPPLE_HIGH_HZ"]]]
    )
    iEEG_ripple_band_passed = (
        mngs.dsp.filt.bandpass(
            np.array(iEEG).astype(float),
            np.array(CONFIG["FS_iEEG"]).astype(float),
            RIPPLE_BANDS,
        ).astype(np.float64)
    ).squeeze(-2)
    iEEG_ripple_band_passed_common = (
        mngs.dsp.filt.bandpass(
            np.array(iEEG_common_ave).astype(float),
            np.array(CONFIG["FS_iEEG"]).astype(float),
            RIPPLE_BANDS,
        ).astype(np.float64)
    ).squeeze(-2)

    # Main
    time_iEEG = np.arange(iEEG.shape[-1]) / CONFIG["FS_iEEG"]
    speed = 0 * time_iEEG
    rip_dfs = []
    for i_trial in range(len(iEEG_ripple_band_passed)):
        try:
            # Ripple detection
            rip_df = Kay_ripple_detector(
                time_iEEG,
                iEEG_ripple_band_passed[i_trial].T,
                speed,
                float(CONFIG["FS_iEEG"]),
                minimum_duration=0.020,
                zscore_threshold=CONFIG["RIPPLE_SD"],
            )

            rip_df_common = Kay_ripple_detector(
                time_iEEG,
                iEEG_ripple_band_passed_common[i_trial].T,
                speed,
                float(CONFIG["FS_iEEG"]),
                minimum_duration=0.010,
                zscore_threshold=CONFIG["RIPPLE_SD"],
            )

            # IoU between ripples of ROI and those of the common average signal
            IoUs_all = []
            for i_rip, rip in rip_df.iterrows():
                IoUs = []
                for i_rip_c, rip_c in rip_df_common.iterrows():
                    IoUs.append(
                        calc_iou(
                            [rip.start_time, rip.end_time],
                            [rip_c.start_time, rip_c.end_time],
                        )
                    )
                IoUs_all.append(np.max(IoUs))
            IoUs_all = np.array(IoUs_all)
            rip_df["IoU"] = IoUs_all
            # rip_df = rip_df[IoUs_all == 0]

            # # sync_z
            # # UnboundLocalError: local variable 'rip' referenced before assignment
            # sync_z = sync_z_session[i_trial]
            # sync_z_all = []
            # for i_rip, rip in rip_df.iterrows():
            #     sync_z_all.append(
            #         sync_z[
            #             int(rip.start_time * CONFIG["FS_iEEG"]) : int(
            #                 rip.end_time * CONFIG["FS_iEEG"]
            #             )
            #         ].mean()
            #     )
            # sync_z_all = np.array(sync_z_all)
            # rip_df["sync_z"] = sync_z_all

            # exclude the edge effects
            rip_df = rip_df[rip_df["start_time"] != time_iEEG[0]]
            rip_df = rip_df[rip_df["start_time"] != time_iEEG[-1]]

            # trial_number
            rip_df["trial_number"] = i_trial + 1

            # append iEEG traces
            iEEG_traces = []
            for (
                i_rip,
                rip,
            ) in (
                rip_df.reset_index().iterrows()
            ):  # rip_df = rips_df.iloc[i_trial]

                start_pts = int(rip["start_time"] * CONFIG["FS_iEEG"])
                end_pts = int(rip["end_time"] * CONFIG["FS_iEEG"])
                iEEG_traces.append(
                    # iEEG[i_trial][:, start_pts:end_pts]
                    np.array(iEEG)[i_trial][:, start_pts:end_pts]
                )
            rip_df["iEEG trace"] = iEEG_traces

            # append ripple band filtered iEEG traces
            ripple_band_iEEG_traces = []
            for (
                i_rip,
                rip,
            ) in (
                rip_df.reset_index().iterrows()
            ):  # rip_df = rips_df.iloc[i_trial]
                start_pts = int(rip["start_time"] * CONFIG["FS_iEEG"])
                end_pts = int(rip["end_time"] * CONFIG["FS_iEEG"])
                ripple_band_iEEG_traces.append(
                    iEEG_ripple_band_passed[i_trial][:, start_pts:end_pts]
                )
            rip_df["ripple band iEEG trace"] = ripple_band_iEEG_traces

            # ripple peak amplitude
            ripple_peak_amplitude = [
                np.abs(rbt).max(axis=-1) for rbt in ripple_band_iEEG_traces
            ]
            ripple_band_baseline_sd = iEEG_ripple_band_passed[i_trial].std(
                axis=-1
            )
            rip_df["ripple_peak_amplitude_sd"] = [
                (rpa / ripple_band_baseline_sd).mean()
                for rpa in ripple_peak_amplitude
            ]

            rip_df["ripple_amplitude_sd"] = [
                (np.abs(rbt).mean(axis=-1) / ripple_band_baseline_sd).mean()
                for rbt in rip_df["ripple band iEEG trace"]
            ]
        except Exception as e:
            print(e)
            __import__("ipdb").set_trace()

        rip_dfs.append(rip_df)

    rip_dfs = pd.concat(rip_dfs)

    # set trial number
    rip_dfs = rip_dfs.set_index("trial_number")
    # trials_info["trial_number"] = trials_info["trial_number"]
    # del trials_info["trial_number"]
    trials_info = trials_info.set_index("trial_number")
    # del trials_info["Unnamed: 0"]

    # adds info
    transfer_keys = [
        "set_size",
        "match",
        "correct",
        "response_time",
    ]
    for k in transfer_keys:
        rip_dfs[k] = trials_info[k]

    rip_dfs["subject"] = i_sub_str
    rip_dfs["session"] = i_session_str

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


# def plot_traces(FS_iEEG):
#     plt.close()
#     plot_start_sec = 0
#     plot_dur_sec = 8

#     plot_dur_pts = plot_dur_sec * FS_iEEG
#     plot_start_pts = plot_start_sec * FS_iEEG

#     i_ch = np.random.randint(iEEG_ripple_band_passed.shape[1])
#     i_trial = 0

#     fig, axes = plt.subplots(2, 1, sharex=True)
#     lw = 1

#     time_iEEG = (np.arange(iEEG.shape[-1]) / FS_iEEG) - 6

#     axes[0].plot(
#         time_iEEG[plot_start_pts : plot_start_pts + plot_dur_pts],
#         iEEG[0][i_ch, plot_start_pts : plot_start_pts + plot_dur_pts],
#         linewidth=lw,
#         label="Raw LFP",
#     )

#     axes[1].plot(
#         time_iEEG[plot_start_pts : plot_start_pts + plot_dur_pts],
#         iEEG_ripple_band_passed[0][
#             i_ch, plot_start_pts : plot_start_pts + plot_dur_pts
#         ],
#         linewidth=lw,
#         label="Ripple-band-passed LFP",
#     )
#     # fills ripple time
#     rip_plot_df = rip_df[
#         (rip_df["trial_number"] == i_trial + 1)
#         & (plot_start_sec < rip_df["start_time"])
#         & (rip_df["end_time"] < plot_start_sec + plot_dur_sec)
#     ]

#     for ax in axes:
#         for ripple in rip_plot_df.itertuples():
#             ax.axvspan(
#                 ripple.start_time - 6,
#                 ripple.end_time - 6,
#                 alpha=0.1,
#                 color="red",
#                 zorder=1000,
#             )
#             ax.axvline(x=-5, color="gray", linestyle="dotted")
#             ax.axvline(x=-3, color="gray", linestyle="dotted")
#             ax.axvline(x=0, color="gray", linestyle="dotted")

#     axes[-1].set_xlabel("Time from probe [sec]")

#     # plt.show()
#     mngs.io.save(
#         plt, "./tmp/ripple_repr_traces_sub_01_session_01_trial_01-50.png"
#     )


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
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 1,
            rip_df[rip_df["phase"] == "Encoding"]
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 2,
            rip_df[rip_df["phase"] == "Maintenance"]
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 3,
            rip_df[rip_df["phase"] == "Retrieval"]
            .pivot_table(columns=["trial_number"], aggfunc="sum")
            .T["n"]
            / 2,
        ],
        axis=1,
    ).fillna(0)
    rip_incidence_hz.columns = [
        "Fixation",
        "Encoding",
        "Maintenance",
        "Retrieval",
    ]
    return rip_incidence_hz


def main():
    from glob import glob

    from natsort import natsorted

    # Parameters
    # iEEG_ROIs = ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]
    # indi_subs_str = [
    #     re.match("./data/Sub_\w{2}", CONFIG["RIPPLE_SD"]).string[-2:]
    #     for CONFIG["RIPPLE_SD"] in natsorted(glob("./data/Sub_*"))
    # ]

    indi_subs_str = [
        sub_dir[-2:] for sub_dir in natsorted(glob("./data/Sub_*"))
    ]

    for iEEG_ROI in CONFIG["iEEG_ROIS"]:
        # For multiple ROIs
        iEEG_ROI = [iEEG_ROI]
        iEEG_ROI_STR = mngs.general.connect_strs(iEEG_ROI)

        rips_df = pd.DataFrame()
        for i_sub_str in indi_subs_str:

            indi_sessions_str = [
                session_dir[-2:]
                for session_dir in glob(f"./data/Sub_{i_sub_str}/*")
            ][: CONFIG["SESSION_THRES"]]

            for i_session_str in tqdm(indi_sessions_str):

                rip_df = add_phase(
                    detect_ripples(
                        i_sub_str=i_sub_str,
                        i_session_str=i_session_str,
                        sd=CONFIG["RIPPLE_SD"],
                        iEEG_ROI=iEEG_ROI_STR,
                    )
                )
                rips_df = pd.concat([rips_df, rip_df])

            # except Exception as e:
            #     print(e)

        mngs.io.save(
            rips_df,
            f"./rips_df/common_average_{CONFIG['RIPPLE_SD']}_sd_{iEEG_ROI_STR}.pkl",
        )


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
