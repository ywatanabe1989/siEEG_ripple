#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-26 16:01:21 (ywatanabe)"

from glob import glob
import mngs
import numpy as np
import matplotlib

# matplotlib.use("Agg")
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
import quantities as pq
import os
from bisect import bisect_right
import pandas as pd
import numpy as np
from scipy.linalg import norm


def get_i_bin(times_sec, bin_size, n_bins):
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    # bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in times_sec]
    return bins


def define_phase_time():
    global PHASES, bin_size, width_bins
    # Parameters
    bin_size = 50 * pq.ms
    width_ms = 500
    width_bins = width_ms / bin_size

    # Preparation
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    global starts, ends, colors
    starts, ends = [], []
    PHASE_START_END_DICT = {}
    for i_phase, phase in enumerate(PHASES):
        start_s = int(
            DURS_OF_PHASES[:i_phase].sum() / (bin_size.rescale("s").magnitude)
        )
        end_s = int(
            DURS_OF_PHASES[: (i_phase + 1)].sum() / (bin_size.rescale("s").magnitude)
        )
        PHASE_START_END_DICT[phase] = (start_s, end_s)
        center_s = int((start_s + end_s) / 2)
        start_s = center_s - int(width_bins / 2)
        end_s = center_s + int(width_bins / 2)
        starts.append(start_s)
        ends.append(end_s)

    colors = ["black", "blue", "green", "red"]

    return starts, ends, PHASE_START_END_DICT, colors


def cosine(v1, v2):
    if np.isnan(v1).any():
        return np.nan
    if np.isnan(v2).any():
        return np.nan
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def rebase_a_vec(v, v_base):
    def production_vector(v1, v0):
        """
        production_vector(np.array([3,4]), np.array([10,0])) # np.array([3, 0])
        """
        return norm(v1) * cosine(v1, v0) * v0 / norm(v0)

    if np.isnan(v).any():
        return np.nan
    if np.isnan(v_base).any():
        return np.nan
    v_prod = production_vector(v, v_base)
    sign = np.sign(cosine(v, v_base))
    return sign * norm(v_prod)


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
    # get 4 -> 8 directions
    gs = {}
    for set_size in [4, 8]:  # 6
        med_traj_ss = np.median(
            trajs[:, :, :][trials_df.set_size == set_size],
            axis=0,
        )
        for i_phase, phase in enumerate(PHASES):
            gg = np.median(med_traj_ss[:, starts[i_phase] : ends[i_phase]], axis=-1)
            gs[f"{phase}_{set_size}"] = gg

    vF_4_8 = gs["Fixation_8"] - gs["Fixation_4"]
    vE_4_8 = gs["Encoding_8"] - gs["Encoding_4"]
    vM_4_8 = gs["Maintenance_8"] - gs["Maintenance_4"]
    vR_4_8 = gs["Retrieval_8"] - gs["Retrieval_4"]
    return vF_4_8, vE_4_8, vM_4_8, vR_4_8


def add_traj(rips_df, bin_size):

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    n_bins = int((8 / bin_size.rescale("s")).magnitude)

    # if not is_control:
    #     rips_df = utils.load_rips(from_pkl=False, only_correct=False)
    # if is_control:
    #     # cons_df = utils.load_cons(from_pkl=False, only_correct=False)
    #     cons_df = utils.load_cons_across_trials(from_pkl=False, only_correct=False)
    #     cons_df["center_time"] = (
    #         (cons_df["end_time"] + cons_df["start_time"]) / 2
    #     ).astype(float)
    #     rips_df = cons_df

    rips_df = discard_initial_ripples(rips_df)
    rips_df = discard_last_ripples(rips_df)

    all_rips = []

    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        for session in ["01", "02"]:
            rips_df_session = rips_df[
                (rips_df.subject == subject) * (rips_df.session == session)
            ]

            # Loads trajs
            lpath = (
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            trajs = mngs.io.load(lpath)

            # get memory-load directions
            trials_df = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
            )
            vF_4_8, vE_4_8, vM_4_8, vR_4_8 = get_4_8_directions(trajs, trials_df)
            rips_df_session["v_Fixation_4_8"] = [
                vF_4_8 for _ in range(len(rips_df_session))
            ]
            rips_df_session["v_Encoding_4_8"] = [
                vE_4_8 for _ in range(len(rips_df_session))
            ]
            rips_df_session["v_Maintenance_4_8"] = [
                vM_4_8 for _ in range(len(rips_df_session))
            ]
            rips_df_session["v_Retrieval_4_8"] = [
                vR_4_8 for _ in range(len(rips_df_session))
            ]

            rips_df_session["traj"] = None
            for i_trial in range(len(trajs)):
                traj = trajs[i_trial, :, :]
                indi = rips_df_session.trial_number == i_trial + 1
                rip_bins = get_i_bin(
                    rips_df_session[indi].center_time,
                    bin_size,
                    n_bins,
                )
                rips_df_session.loc[indi, "traj"] = [[traj] for _ in range(indi.sum())]

            rips_df_session = rips_df_session.reset_index()
            all_rips.append(rips_df_session)
    all_rips = pd.concat(all_rips).reset_index()
    return all_rips


def add_coordinates(rips_df):
    def extract_coordinates_before_or_after_ripple(rips_df, delta_bin=0):
        out = []
        for i_rip, (_, rip) in enumerate(rips_df.iterrows()):
            rip_traj = np.array(rip.traj).squeeze()
            bin_tgt = rip.i_bin + delta_bin
            bin_phase_start, bin_phase_end = PHASE_START_END_DICT[
                rips_df.iloc[i_rip].phase
            ]
            if (bin_phase_start <= bin_tgt) * (bin_tgt < bin_phase_end):
                rip_coord = rip_traj[:, rip.i_bin + delta_bin]
            else:
                rip_coord = np.array([np.nan, np.nan, np.nan])

            out.append(rip_coord)
        return out

    rips_df["Fixation"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[0] : ends[0]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]
    rips_df["Encoding"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[1] : ends[1]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]
    rips_df["Maintenance"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[2] : ends[2]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]
    rips_df["Retrieval"] = [
        np.median(np.vstack(rips_df.traj)[:, :, starts[3] : ends[3]], axis=-1)[ii]
        for ii in range(len(rips_df))
    ]

    n_bins = int((8 / bin_size.rescale("s")).magnitude)
    rips_df["i_bin"] = get_i_bin(rips_df.center_time, bin_size, n_bins)

    # nn = 20
    nn = 80
    for ii in range(nn):
        delta_bin = ii - nn // 2
        rips_df[f"{delta_bin}"] = extract_coordinates_before_or_after_ripple(
            rips_df, delta_bin=delta_bin
        )

    return rips_df


def get_rebased_speed(event_df, tgt_bin, phase_event, phase_direction):
    event_df_phase = event_df[event_df.phase == phase_event]
    v_speed = np.vstack(event_df_phase[f"{tgt_bin}"]) - np.vstack(
        event_df_phase[f"{tgt_bin-1}"]
    )
    rebased_speed = [
        rebase_a_vec(
            v_speed[i_rip], event_df_phase.iloc[i_rip][f"v_{phase_direction}_4_8"]
        )
        for i_rip in range(len(event_df_phase))
    ]
    return rebased_speed

def save_directed_peri_event_speed(event_df, event_str):
    dfs = []
    for i_pd, phase_direction in enumerate(PHASES):
        for i_pr, phase_event in enumerate(PHASES):
            df = {}
            for tgt_bin in range(-11, 13):
                df[f"{tgt_bin-1} to {tgt_bin}"] = np.abs(get_rebased_speed(
                    event_df, tgt_bin, phase_event, phase_direction
                ))
            # df = mngs.general.force_dataframe(df)

            df = pd.DataFrame(df)
            df2 = pd.DataFrame({
                "bins":list(df.columns),
                f"mean_{i_pr}_{phase_event}_based_on_{i_pd}_{phase_direction}_4_8": list(df.mean()),
                f"std_{i_pr}_{phase_event}_based_on_{i_pd}_{phase_direction}_4_8": list(df.std()),                
            })

            dfs.append(df2)
    dfs = pd.concat(dfs, axis=1)

    mngs.io.save(dfs, f"./tmp/figs/line/directed_peri_ripple_speed/{event_str}.csv")
    

            
if __name__ == "__main__":
    import re
    import sys

    sys.path.append(".")
    import utils

    starts, ends, PHASE_START_END_DICT, colors = define_phase_time()
    # roi = "AHL"
    # LPATHs = glob(f"data/Sub_0?/Session_0?/traj_z_by_session_{roi}.npy")
    # lpath_traj = LPATHs[0]
    # subject = re.findall("Sub_0[0-1]", lpath_traj)[0][-2:]
    # session = re.findall("Session_0[0-1]", lpath_traj)[0][-2:]

    # ldir = mngs.general.split_fpath(lpath_traj)[0]

    # trajs = mngs.io.load(lpath_traj)  # (50, 3, 160)
    # trials_df = mngs.io.load(ldir + "trials_info.csv")
    rips_df = add_coordinates(add_traj(utils.load_rips(), 50 * pq.ms))
    cons_df = add_coordinates(add_traj(utils.load_cons_across_trials(), 50 * pq.ms))    

    save_directed_peri_event_speed(rips_df, "SWR")
    save_directed_peri_event_speed(cons_df, "Control")    



    rips_df = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]

    rips_df.center_time
    bin_size = 50 * pq.ms
    n_bins = 160
    rips_df["i_bin"] = get_i_bin(rips_df.center_time, bin_size, n_bins)
    width_ms = 500
    width_bins = width_ms / bin_size.magnitude

    for delta_bin in range(-5, 5):
        rips_df[f"{delta_bin}"] = np.array(np.nan)

    rips_df = rips_df.reset_index()
    del rips_df["index"]
    for i_rip, rip in rips_df.iterrows():
        traj = trajs[int(rip.trial_number - 1)]

        start_bin = int(rip.i_bin - (width_bins / 2))
        end_bin = int(rip.i_bin + (width_bins / 2))
        for tgt_bin in range(start_bin, end_bin):
            delta_bin = tgt_bin - rip.i_bin
            rips_df.iloc[i_rip][f"{delta_bin}"] = traj[:, tgt_bin]
            print(delta_bin)

    trajs.shape
