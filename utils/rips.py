#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-16 07:34:03 (ywatanabe)"

import sys
from bisect import bisect_right

import mngs
import numpy as np
import pandas as pd

# sys.path.append(".")
# import utils
from ._define_phase_time import define_phase_time
from . import rips
from ._load_rips import load_rips
from ._load_cons import load_cons
from ._load_cons_across_trials import load_cons_across_trials
from ._add_ripple_peak_amplitude_to_cons import add_ripple_peak_amplitudeto_to_cons
import warnings


ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
(
    PHASES,
    PHASES_BINS_DICT,
    GS_BINS_DICT,
    COLORS_DICT,
    BIN_SIZE,
) = define_phase_time()
N_BINS = int((8 / BIN_SIZE.rescale("s")).magnitude)


def get_i_bins(times_sec):
    bin_centers = (
        (np.arange(N_BINS) * BIN_SIZE) + ((np.arange(N_BINS) + 1) * BIN_SIZE)
    ) / 2
    # bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in times_sec]
    return bins


def discard_initial_ripples(rips_df, width_s=0.3):
    indi = []
    starts = [0, 1, 3, 6]
    ends = [1, 3, 6, 8]
    for ss, ee in zip(starts, ends):
        indi.append((ss + width_s < rips_df.center_time) * (rips_df.center_time < ee))
    indi = np.vstack(indi).sum(axis=0).astype(bool)
    return rips_df[indi]


def discard_last_ripples(rips_df, width_s=0.3):
    return rips_df[rips_df["center_time"] < 8 - width_s]


def get_4_8_directions(trajs, trials_df):
    gs = {}
    for set_size in [4, 8]:  # 6
        med_traj_ss = np.median(
            trajs[trials_df.set_size == set_size],
            axis=0,
        )
        for i_phase, phase in enumerate(PHASES):
            gg = np.median(
                med_traj_ss[:, GS_BINS_DICT[phase][0] : GS_BINS_DICT[phase][1]], axis=-1
            )
            gs[f"{phase}_{set_size}"] = gg

    vF_4_8 = gs["Fixation_8"] - gs["Fixation_4"]
    vE_4_8 = gs["Encoding_8"] - gs["Encoding_4"]
    vM_4_8 = gs["Maintenance_8"] - gs["Maintenance_4"]
    vR_4_8 = gs["Retrieval_8"] - gs["Retrieval_4"]
    return vF_4_8, vE_4_8, vM_4_8, vR_4_8


def load_rips_df_with_traj(bin_size, is_control=False):
    if not is_control:
        rips_df = load_rips(from_pkl=False, only_correct=False)
    if is_control:
        # cons_df = load_cons(from_pkl=False, only_correct=False)
        cons_df = load_cons_across_trials(from_pkl=False, only_correct=False)
        cons_df["center_time"] = (
            (cons_df["end_time"] + cons_df["start_time"]) / 2
        ).astype(float)
        rips_df = cons_df

    rips_df = discard_initial_ripples(rips_df)
    rips_df = discard_last_ripples(rips_df)

    all_rips = []
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        for session in ["01", "02"]:
            # Loads
            trajs = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            trials_df = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
            )

            rips_df_session = rips_df[
                (rips_df.subject == subject) * (rips_df.session == session)
            ]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", pd.core.common.SettingWithCopyWarning
                    )

                    vF_4_8, vE_4_8, vM_4_8, vR_4_8 = get_4_8_directions(trajs, trials_df)

                    rips_df_session["v_Fixation_4_8"] = [vF_4_8 for _ in range(len(rips_df_session))]
                    rips_df_session["v_Encoding_4_8"] = [vE_4_8 for _ in range(len(rips_df_session))]
                    rips_df_session["v_Maintenance_4_8"] = [vM_4_8 for _ in range(len(rips_df_session))]
                    rips_df_session["v_Retrieval_4_8"] = [vR_4_8 for _ in range(len(rips_df_session))]                    
                    

                    # for ii in range(len(rips_df_session)): # fixme
                    #     (
                    #         rips_df_session["v_Fixation_4_8"].iloc[ii],
                    #         rips_df_session["v_Encoding_4_8"].iloc[ii],
                    #         rips_df_session["v_Maintenance_4_8"].iloc[ii],
                    #         rips_df_session["v_Retrieval_4_8"].iloc[ii],
                    #     ) = get_4_8_directions(trajs, trials_df)

            except Exception as e:
                print(e)
                import ipdb

                ipdb.set_trace()

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", pd.core.common.SettingWithCopyWarning
                )  # fixme
                rips_df_session["traj"] = None
                
            for i_trial in range(len(trajs)):
                traj = trajs[i_trial, :, :]
                # indi = rips_df_session.index == i_trial + 1
                indi = rips_df_session.trial_number == i_trial + 1
                rip_bins = get_i_bins(
                    rips_df_session[indi].center_time,
                )

                rips_df_session.loc[indi, "traj"] = [[traj] for _ in range(indi.sum())]

            rips_df_session = rips_df_session.reset_index()
            all_rips.append(rips_df_session)
    all_rips = pd.concat(all_rips).reset_index()
    return all_rips


def get_vec_from_rips_df(rips_df, col1, col2, col_base_start=None, col_base_end=None):
    v = np.vstack(rips_df[col2]) - np.vstack(rips_df[col1])

    if col_base_start is not None:
        v_base = np.vstack(rips_df[col_base_end]) - np.vstack(rips_df[col_base_start])
        return np.array(
            [mngs.linalg.rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))]
        )
    else:
        return v


def add_coordinates(rips_df):
    def extract_coordinates_before_or_after_ripple(rips_df, delta_bin=0):
        out = []
        for i_rip, (_, rip) in enumerate(rips_df.iterrows()):
            rip_traj = np.array(rip.traj).squeeze()
            bin_tgt = rip.i_bin + delta_bin

            bin_phase_start, bin_phase_end = PHASES_BINS_DICT[rips_df.iloc[i_rip].phase]
            if (bin_phase_start <= bin_tgt) * (bin_tgt < bin_phase_end):
                rip_coord = rip_traj[:, rip.i_bin + delta_bin]
            else:
                rip_coord = np.array([np.nan, np.nan, np.nan])

            out.append(rip_coord)
        return out

    for phase in PHASES:
        rips_df[phase] = [
            np.median(
                np.vstack(rips_df.traj)[
                    :, :, GS_BINS_DICT[phase][0] : GS_BINS_DICT[phase][1]
                ],
                axis=-1,
            )[ii]
            for ii in range(len(rips_df))
        ]

    rips_df["i_bin"] = get_i_bins(rips_df.center_time)

    nn = int(3 / BIN_SIZE.rescale("s").magnitude)
    for ii in range(nn):
        delta_bin = ii - nn // 2
        rips_df[f"{delta_bin}"] = extract_coordinates_before_or_after_ripple(
            rips_df, delta_bin=delta_bin
        )

    return rips_df

def to_digi_rips(rips_df, subject, session, roi):
    rips_df_session = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]
    rips_df_session = rips_df_session\
        [["subject", "session", "trial_number", "start_time", "center_time", "end_time", "set_size", "match"]]

    # ripples digi
    n_trials = len(mngs.io.load(f"./data/Sub_{subject}/Session_{session}/trials_info.csv"))
    bin_s = 50 / 1000
    n_bins = int(8 / bin_s)
    rips_digi = np.zeros([n_trials, n_bins], dtype=int)
    for i_trial in range(n_trials):
        rips_df_trial = rips_df_session[rips_df_session.trial_number == i_trial+1]
        for i_rip, (_, rip) in enumerate(rips_df_trial.iterrows()):
            start_bin = int(rip.start_time / bin_s)
            end_bin = int(rip.end_time / bin_s)
            rips_digi[i_trial, start_bin:end_bin] = 1
    return rips_digi
