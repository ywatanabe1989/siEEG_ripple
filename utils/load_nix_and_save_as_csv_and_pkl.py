#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-29 13:50:23 (ywatanabe)"

"""
https://gin.g-node.org/USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/src/master/code_MATLAB/Load_Data_Example_Script.m
"""

import h5py
import numpy as np
import pandas as pd
import xarray
import mngs


def to_ascii(binary):
    try:
        return binary.decode("ascii")
    except Exception as e:
        # print(e)
        return binary


def get_general_info(f):
    return pd.Series(
        dict(
            institution=f["metadata"]["General"]["properties"]["Institution"][0][0],
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
            recording_setup_EEG=f["metadata"]["General"]["sections"]["Recording setup"][
                "properties"
            ]["Recording setup EEG"][0][0],
        )
    ).apply(to_ascii)


def get_task_info(f):
    return pd.Series(
        dict(
            task_name=f["metadata"]["Task"]["properties"]["Task name"][0][0],
            task_description=f["metadata"]["Task"]["properties"]["Task description"][0][
                0
            ],
            task_URL=f["metadata"]["Task"]["properties"]["Task URL"][0][0],
        )
    ).apply(to_ascii)


def get_subject_info(f):
    return pd.Series(
        dict(
            age=f["metadata"]["Subject"]["properties"]["Age"][0][0],
            sex=f["metadata"]["Subject"]["properties"]["Sex"][0][0],
            handedness=f["metadata"]["Subject"]["properties"]["Handedness"][0][0],
            pathology=f["metadata"]["Subject"]["properties"]["Pathology"][0][0],
            depth_electrodes=f["metadata"]["Subject"]["properties"]["Depth electrodes"][
                0
            ][0],
            electrodes_in_seizure_onset_zone=f["metadata"]["Subject"]["properties"][
                "Electrodes in seizure onset zone (SOZ)"
            ][0][0],
        )
    ).apply(to_ascii)


def get_session_info(f):
    return pd.Series(
        dict(
            number_of_trials=f["metadata"]["Session"]["properties"]["Number of trials"][
                0
            ][0],
            trial_duration=f["metadata"]["Session"]["properties"]["Trial duration"][0][
                0
            ],
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

    n_trials = len(f["metadata"]["Session"]["sections"]["Trial properties"]["sections"])
    all_trial_info = []
    for i_trial in range(n_trials):
        i_trial += 1
        _single_trial = f["metadata"]["Session"]["sections"]["Trial properties"][
            "sections"
        ][f"Trial_{i_trial:02d}"]["properties"]
        all_trial_info.append(_get_single_trial_info(_single_trial))

    return pd.concat(all_trial_info, axis=1).T.apply(to_ascii)


def load_signals(f, iEEG_ROI=["PHL", "PHR"]):
    def _in_load_signals_trial(f, trial_number=1):

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

        iEEG_sec = iEEG.shape[1] / SAMP_RATE_iEEG
        EEG_sec = EEG.shape[1] / SAMP_RATE_EEG

        assert iEEG_sec == EEG_sec

        EEG = pd.DataFrame(data=EEG.T, columns=EEG_labels).T
        iEEG = pd.DataFrame(data=iEEG.T, columns=iEEG_labels).T

        iEEG = iEEG[mngs.general.search(iEEG_ROI, list(iEEG.index), as_bool=True)[0]]

        # spike times
        st_keys = mngs.general.search(
            "^Spike_Times_", f["data"][list(f["data"].keys())[0]]["data_arrays"].keys()
        )[1]
        st_keys = mngs.general.search(iEEG_ROI, st_keys)[1]
        st_keys = mngs.general.search(f"_Trial_{trial_number:02d}$", st_keys)[1]

        st_dict = {}
        for st_key in st_keys:
            st_dict[st_key] = np.array(
                f["data"][list(f["data"].keys())[0]]["data_arrays"][st_key]["data"]
            )
        st_df = mngs.general.force_dataframe(st_dict)
        # if st_df.shape == (0, 0):
        #     print(
        #         f"\nNo spikes: Subject #{subject}, Session #{session}, Trial #{trial_number}, ROI {iEEG_ROI}\n"
        #     )

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

        # import matplotlib.pyplot as plt
        # plt.plot(sfs[0])

        return dict(
            iEEG=iEEG,
            EEG=EEG,
            spike_times=st_df,
        )

    n_trials = len(f["metadata"]["Session"]["sections"]["Trial properties"]["sections"])

    iEEGs, EEGs, spike_times = [], [], []
    for i_trial in range(n_trials):
        iEEG_trial = _in_load_signals_trial(f, trial_number=i_trial + 1)["iEEG"]
        EEG_trial = _in_load_signals_trial(f, trial_number=i_trial + 1)["EEG"]
        st_df_trial = _in_load_signals_trial(f, trial_number=i_trial + 1)["spike_times"]

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


# https://gin.g-node.org/USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/src/master/code_MATLAB/Load_Data_Example_Script.m#L138


if __name__ == "__main__":
    from glob import glob
    import re

    fpaths = glob("./data/data_nix/*")
    # for iEEG_ROI in ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]:
    for iEEG_ROI in ["ECL", "ECR", "AL", "AR"]:
        """
        iEEG_ROI = "AHR"
        """
        iEEG_ROI_STR = mngs.general.connect_strs(iEEG_ROI)
        for fpath in fpaths:

            # session
            f = h5py.File(fpath, "r+")

            # i_Sub
            subject = re.findall("Subject_[\0-9]{2}", fpath)[0][-2:]

            # i_Session
            session = re.findall("Session_[\0-9]{2}", fpath)[0][-2:]

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
            _signals = load_signals(f, iEEG_ROI=iEEG_ROI)
            iEEG, EEG, spike_times = (
                _signals["iEEG"],
                _signals["EEG"],
                _signals["spike_times"],
            )

            # saves
            session_dir = f"./data/Sub_{subject}/Session_{session}/"

            # print(pd.concat(spike_times))
            # if pd.concat(spike_times).shape == (0,0):
            #     print(subject, session, iEEG_ROI)

            # mngs.io.save(meta_df, session_dir + f"meta.csv")
            # mngs.io.save(trials_info, session_dir + f"trials_info.csv")
            # mngs.io.save(iEEG, session_dir + f"iEEG_{iEEG_ROI_STR}.pkl")
            # mngs.io.save(EEG, session_dir + f"EEG.pkl")
            # mngs.io.save(spike_times, session_dir + f"spike_times_{iEEG_ROI_STR}.pkl")

    """
    ls "./data/Sub_01/Session_01/spike_times_AHL.pkl"
    mngs.io.load("./data/Sub_01/Session_01/spike_times_AHL.pkl")
    mngs.io.load("./data/Sub_01/Session_03/spike_times_AHL.pkl")    
    """
