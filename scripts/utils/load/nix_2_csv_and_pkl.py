#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: 2024-05-27 17:07:27
# load_nix_and_save_as_csv_and_pkl.py

"""
This script loads .h5 files downloaded by the publicly available repository:
https://gin.g-node.org/USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM

Then, saves the following under CONFIG["SESSION_DIR"]:
    meta.csv
    trials_info.csv
    iEEG_{ROI}.pkl
    EEG.pkl
    spike_times_{ROI}.pkl

This source code is based on the provided matlab script:
https://gin.g-node.org/USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/src/master/code_MATLAB/Load_Data_Example_Script.m
"""


"""
Imports
"""
import re
import sys
from glob import glob

import h5py
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray

"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main():
    fpaths = glob("./data/data_nix/*")
    for iEEG_ROI in ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]:
        # iEEG_ROI_STR = mngs.general.connect_strs(iEEG_ROI)
        for fpath in fpaths:

            # session
            f = h5py.File(fpath, "r+")

            meta_df, trials_info, iEEG, EEG, spike_times = load_h5(f, iEEG_ROI)

            # Saves
            subject = re.findall("Subject_[\0-9]{2}", fpath)[0][-2:]
            session = re.findall("Session_[\0-9]{2}", fpath)[0][-2:]
            session_dir = (
                CONFIG["SESSION_DIR"]
                .replace("XX", f"{subject}")
                .replace("YY", f"{session}")
            )

            mngs.io.save(meta_df, session_dir + f"meta.csv", from_git=True)
            mngs.io.save(
                trials_info, session_dir + f"trials_info.csv", from_git=True
            )
            mngs.io.save(
                iEEG, session_dir + f"iEEG/{iEEG_ROI}.pkl", from_git=True
            )
            mngs.io.save(EEG, session_dir + f"EEG.pkl", from_git=True)
            mngs.io.save(
                spike_times,
                session_dir + f"spike_times/{iEEG_ROI}.pkl",
                from_git=True,
            )


def load_h5(f, iEEG_ROI):
    # meta data
    general_info = get_general_info(f)
    task_info = get_task_info(f)
    subject_info = get_subject_info(f)
    session_info = get_session_info(f)
    meta_df = pd.concat(
        [
            general_info,
            task_info,
            subject_info,
            session_info,
        ]
    )

    # trials info
    trials_info = get_trials_info(f)

    # signals
    _signals = get_signals(f, iEEG_ROI=iEEG_ROI)
    iEEG, EEG, spike_times = (
        _signals["iEEG"],
        _signals["EEG"],
        _signals["spike_times"],
    )
    return meta_df, trials_info, iEEG, EEG, spike_times


def to_ascii(binary):
    try:
        return binary.decode("ascii")
    except Exception as e:
        # print(e)
        return binary


def get_general_info(f):
    return pd.Series(
        dict(
            institution=f["metadata"]["General"]["properties"]["Institution"][
                0
            ][0],
            recording_location=f["metadata"]["General"]["properties"][
                "Recording location"
            ][0][0],
            publication_name=f["metadata"]["General"]["sections"][
                "Related publications"
            ]["properties"]["Publication name"][0][0],
            publication_DOI=f["metadata"]["General"]["sections"][
                "Related publications"
            ]["properties"]["Publication DOI"][0][0],
            recording_setup_iEEG=f["metadata"]["General"]["sections"][
                "Recording setup"
            ]["properties"]["Recording setup iEEG"][0][0],
            recording_setup_EEG=f["metadata"]["General"]["sections"][
                "Recording setup"
            ]["properties"]["Recording setup EEG"][0][0],
        )
    ).apply(to_ascii)


def get_task_info(f):
    return pd.Series(
        dict(
            task_name=f["metadata"]["Task"]["properties"]["Task name"][0][0],
            task_description=f["metadata"]["Task"]["properties"][
                "Task description"
            ][0][0],
            task_URL=f["metadata"]["Task"]["properties"]["Task URL"][0][0],
        )
    ).apply(to_ascii)


def get_subject_info(f):
    return pd.Series(
        dict(
            age=f["metadata"]["Subject"]["properties"]["Age"][0][0],
            sex=f["metadata"]["Subject"]["properties"]["Sex"][0][0],
            handedness=f["metadata"]["Subject"]["properties"]["Handedness"][0][
                0
            ],
            pathology=f["metadata"]["Subject"]["properties"]["Pathology"][0][
                0
            ],
            depth_electrodes=f["metadata"]["Subject"]["properties"][
                "Depth electrodes"
            ][0][0],
            electrodes_in_seizure_onset_zone=f["metadata"]["Subject"][
                "properties"
            ]["Electrodes in seizure onset zone (SOZ)"][0][0],
        )
    ).apply(to_ascii)


def get_session_info(f):
    return pd.Series(
        dict(
            number_of_trials=f["metadata"]["Session"]["properties"][
                "Number of trials"
            ][0][0],
            trial_duration=f["metadata"]["Session"]["properties"][
                "Trial duration"
            ][0][0],
        )
    ).apply(to_ascii)


def get_trials_info(f):
    def _get_single_trial_info(single_trial):
        return pd.Series(
            dict(
                trial_number=single_trial["Trial number"][0][0],
                set_size=single_trial["Set size"][0][0],
                match=single_trial["Match"][0][0],
                correct=single_trial["Correct"][0][0],
                response=single_trial["Response"][0][0],
                response_time=single_trial["Response time"][0][0],
                probe_letter=single_trial["Probe letter"][0][0],
                artifact=single_trial["Artifact"][0][0],
            )
        ).apply(to_ascii)

    n_trials = len(
        f["metadata"]["Session"]["sections"]["Trial properties"]["sections"]
    )
    all_trial_info = []
    for i_trial in range(n_trials):
        i_trial += 1
        _single_trial = f["metadata"]["Session"]["sections"][
            "Trial properties"
        ]["sections"][f"Trial_{i_trial:02d}"]["properties"]
        all_trial_info.append(_get_single_trial_info(_single_trial))

    return pd.concat(all_trial_info, axis=1).T.apply(to_ascii)


def get_signals(f, iEEG_ROI=["PHL", "PHR"]):
    def _in_get_signals_trial(f, trial_number=1):

        # Scalp EEG and iEEG data were resampled at 200 Hz and 2kHz, respectively.
        SAMP_RATE_iEEG = 2000
        SAMP_RATE_EEG = 200

        # iEEG
        iEEG = np.array(
            f["data"][list(f["data"].keys())[0]]["data_arrays"][
                f"iEEG_Data_Trial_{trial_number:02d}"
            ]["data"]
        )
        iEEG_labels = np.array(
            f["data"][list(f["data"].keys())[0]]["data_arrays"][
                f"iEEG_Data_Trial_{trial_number:02d}"
            ]["dimensions"]["1"]["labels"]
        )
        iEEG_labels = [l.decode("ascii") for l in iEEG_labels]
        # volt unit

        # EEG
        EEG = np.array(
            f["data"][list(f["data"].keys())[0]]["data_arrays"][
                f"Scalp_EEG_Data_Trial_{trial_number:02d}"
            ]["data"]
        )
        EEG_labels = np.array(
            f["data"][list(f["data"].keys())[0]]["data_arrays"][
                f"Scalp_EEG_Data_Trial_{trial_number:02d}"
            ]["dimensions"]["1"]["labels"]
        )
        EEG_labels = [l.decode("ascii") for l in EEG_labels]
        # volt unit

        # iEEG_sec = iEEG.shape[1] / SAMP_RATE_iEEG
        # EEG_sec = EEG.shape[1] / SAMP_RATE_EEG
        iEEG_sec = iEEG.shape[1] / CONFIG["FS_iEEG"]
        EEG_sec = EEG.shape[1] / CONFIG["FS_EEG"]

        assert iEEG_sec == EEG_sec

        EEG = pd.DataFrame(data=EEG.T, columns=EEG_labels).T
        iEEG = pd.DataFrame(data=iEEG.T, columns=iEEG_labels).T

        iEEG = iEEG[
            mngs.general.search(iEEG_ROI, list(iEEG.index), as_bool=True)[0]
        ]

        # spike times
        keys_list = [
            k
            for k in f["data"][list(f["data"].keys())[0]]["data_arrays"].keys()
        ]

        st_keys = keys_list.copy()
        for keys in ["^Spike_Times_", iEEG_ROI, f"_Trial_{trial_number:02d}$"]:
            st_keys = mngs.general.search(keys, st_keys)[1]

        st_dict = {}
        for st_key in st_keys:
            st_dict[st_key] = np.array(
                f["data"][list(f["data"].keys())[0]]["data_arrays"][st_key][
                    "data"
                ]
            )
        st_df = mngs.general.force_dataframe(st_dict)

        # # spike waveforms
        # sf_keys = mngs.general.search("^Spike_Waveform_",
        #                               f["data"][list(f["data"].keys())[0]]["data_arrays"].keys())[1]
        # sf_keys = mngs.general.search(iEEG_ROI, sf_keys)[1]
        # # sf_keys = mngs.general.search(f"_Trial_{trial_number:02d}$", sf_keys)[1]
        # sf_key = sf_keys[0]
        # sfs = np.array(f["data"][list(f["data"].keys())[0]]["data_arrays"][
        #         sf_key
        #     ]["data"]) # 2 ms, 32 kHz
        # mean and SD

        return dict(
            iEEG=iEEG,
            EEG=EEG,
            spike_times=st_df,
        )

    n_trials = len(
        f["metadata"]["Session"]["sections"]["Trial properties"]["sections"]
    )

    iEEGs, EEGs, spike_times = [], [], []
    for i_trial in range(n_trials):
        iEEG_trial = _in_get_signals_trial(f, trial_number=i_trial + 1)["iEEG"]
        EEG_trial = _in_get_signals_trial(f, trial_number=i_trial + 1)["EEG"]
        st_df_trial = _in_get_signals_trial(f, trial_number=i_trial + 1)[
            "spike_times"
        ]

        iEEGs.append(iEEG_trial)
        EEGs.append(EEG_trial)
        spike_times.append(st_df_trial)

    iEEGs = xarray.DataArray(
        iEEGs,
        dims=("trial", "channel", "time"),
        coords=(
            np.arange(n_trials) + 1,
            np.array(iEEGs[0].index).astype(str),
            iEEGs[0].columns,
        ),
    )

    EEGs = xarray.DataArray(
        EEGs,
        dims=("trial", "channel", "time"),
        coords=(
            np.arange(n_trials) + 1,
            np.array(EEGs[0].index).astype(str),
            EEGs[0].columns,
        ),
    )

    return dict(
        iEEG=iEEGs,
        EEG=EEGs,
        spike_times=spike_times,
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
