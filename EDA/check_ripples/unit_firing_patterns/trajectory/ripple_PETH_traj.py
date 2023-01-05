#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-05 13:19:44 (ywatanabe)"

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from elephant.gpfa import GPFA
import quantities as pq
import mngs
import neo
import numpy as np
import pandas as pd
import ffmpeg
from matplotlib import animation
import os
from mpl_toolkits.mplot3d import Axes3D
from bisect import bisect_right
from numpy.linalg import norm
import seaborn as sns
import random
from scipy.stats import brunnermunzel, ttest_ind
from scipy.linalg import norm


def to_spiketrains(spike_times_all_trials):
    spike_trains_all_trials = []
    for st_trial in spike_times_all_trials:

        spike_trains_trial = []
        for col, col_df in st_trial.iteritems():

            spike_times = col_df[col_df != ""]
            train = neo.SpikeTrain(list(spike_times) * pq.s, t_start=-6.0, t_stop=2.0)
            spike_trains_trial.append(train)

        spike_trains_all_trials.append(spike_trains_trial)

    return spike_trains_all_trials


def get_i_bin(times_sec, bin_size, n_bins):
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    # bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in times_sec]
    return bins


def cosine(v1, v2):
    if np.isnan(v1).any():
        return np.nan
    if np.isnan(v2).any():
        return np.nan
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def nannorm(v):
    if np.isnan(v).any():
        return np.nan
    else:
        return norm(v)


def production_vector(v1, v0):
    """
    production_vector(np.array([3,4]), np.array([10,0])) # np.array([3, 0])
    """
    return norm(v1) * cosine(v1, v0) * v0 / norm(v0)


def rebase_a_vec(v, v_base):
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


def normalize_trajs(trajs):
    # min max norm
    trajs = np.stack(trajs, axis=0)
    trajs_min = trajs.min(axis=0, keepdims=True).min(axis=-1, keepdims=True)
    trajs -= trajs_min
    trajs_max = trajs.max(axis=0, keepdims=True).max(axis=-1, keepdims=True)
    trajs /= trajs_max
    return trajs


def load_rips_df_with_traj(bin_size):
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    n_bins = int((8 / bin_size.rescale("s")).magnitude)

    # global starts, ends, colors

    all_rips = []
    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        rips_df = mngs.io.load(f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl")
        rips_df = discard_initial_ripples(rips_df)
        rips_df = discard_last_ripples(rips_df)
        rips_df["traj"] = None

        for session in ["01", "02"]:
            # Loads
            LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
            spike_trains = to_spiketrains(mngs.io.load(LPATH))

            rips_df_session = rips_df[
                (rips_df.subject == subject) * (rips_df.session == session)
            ]

            # GPFA
            gpfa = GPFA(bin_size=bin_size, x_dim=3)
            trajs = gpfa.fit_transform(spike_trains)
            # trajs = normalize_trajs(trajs)

            for i_trial in range(len(trajs)):
                traj = trajs[i_trial]
                indi = rips_df_session.index == i_trial + 1
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


def get_vec(rips_df, col1, col2, col_base_start=None, col_base_end=None):
    v = np.vstack(rips_df[col2]) - np.vstack(rips_df[col1])

    if col_base_start is not None:
        v_base = np.vstack(rips_df[col_base_end]) - np.vstack(rips_df[col_base_start])
        return np.array([rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))])
    else:
        return v


def extract_coordinates_before_or_after_ripple(rips_df, delta_bin=0):
    out = []
    for i_rip, (_, rip) in enumerate(rips_df.iterrows()):
        try:
            tmp = np.array(rip.traj).squeeze()[:, rip.i_bin + delta_bin]
        except:
            tmp = np.array([np.nan, np.nan, np.nan])
        out.append(tmp)
    return out


def add_coordinates(rips_df):
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

    nn = 20
    for ii in range(nn):
        delta_bin = ii - nn // 2
        rips_df[f"{delta_bin}"] = extract_coordinates_before_or_after_ripple(
            rips_df, delta_bin=delta_bin
        )

    for i_rip in range(len(rips_df)):
        i_bin = rips_df.iloc[i_rip]["i_bin"]
        bin_range_phase = (
            np.array(PHASE_START_END_DICT[rips_df.iloc[i_rip].phase])
            / (bin_size.rescale("s").magnitude)
        ).astype(int)

        for ii in range(nn):
            delta_bin = ii - nn // 2
            if (bin_range_phase[0] <= i_bin + delta_bin) and (
                i_bin + delta_bin <= bin_range_phase[1]
            ):
                pass
            else:
                # print(bin_range_phase, i_bin + delta_bin)
                # rips_df.iloc[i_rip][f"{delta_bin}"] = np.array([np.nan, np.nan, np.nan]).reshape(-1, 1)
                rips_df.loc[i_rip, f"{delta_bin}"] = np.array(
                    [np.nan, np.nan, np.nan]
                ).reshape(-1, 1)
    return rips_df


def plot_cosine(rips_df):
    # Cosine
    nn = 20
    xlim = (-1, 1)
    binwidth = 0.1
    for i_phase, phase in enumerate(PHASES):
        fig, axes = plt.subplots(
            nrows=nn, sharex=True, sharey=True, figsize=(6.4 * 3, 4.8 * 3)
        )
        rips_df_phase = rips_df[rips_df.phase == phase]
        for ii in range(nn - 2):
            ax = axes[ii]
            if (ii != 0) and (ii != nn):
                delta_bin = int(ii - nn / 2)
                col1 = f"{delta_bin-1}"
                col1_str = f"{int((delta_bin-1)*bin_size.magnitude)}"
                col2 = f"{delta_bin}"
                col2_str = f"{int((delta_bin)*bin_size.magnitude)}"
                col3 = f"{delta_bin+1}"
                col3_str = f"{int((delta_bin+1)*bin_size.magnitude)}"
                col4 = f"{delta_bin+2}"
                col4_str = f"{int((delta_bin+2)*bin_size.magnitude)}"
                cos = [
                    cosine(
                        get_vec(rips_df_phase, col1, col2)[ii],
                        get_vec(rips_df_phase, col3, col4)[ii],
                    )
                    for ii in range(len(rips_df_phase))
                ]
                if not np.isnan(cos).all():
                    ax.hist(
                        cos,
                        density=True,
                        bins=np.arange(xlim[0], xlim[1] + binwidth, binwidth),
                    )

                ax.set_ylabel(
                    f"cosine(\n{col1_str}-{col2_str}\n, \n{col3_str}-{col4_str})"
                )
        mngs.io.save(fig, f"./tmp/figs/hist/traj_cosine/{i_phase}_{phase}.png")
        plt.close()

def plot_norm_2(rips_df):
    n_bases = len([None, "Fixation", "Encoding", "Maintenance", "Encoding"])
    fig, axes = plt.subplots(
        nrows=n_bases, sharex=True, figsize=(6.4 * 3, 4.8 * 3)
    )  # sharey=True,
    xlim = (-1000, 1000)
    ylim = (-0.3, 0.3)
    out_df = pd.DataFrame()
    for i_base, (cbs, cbe) in enumerate(
        zip(
            [None, "Fixation", "Encoding", "Maintenance", "Encoding"],
            [None, "Encoding", "Maintenance", "Retrieval", "Retrieval"],
        )
    ):

        dir_txt = f"{cbs}-{cbe}_based"
        ax = axes[i_base]
        i_ax = i_base

        samp_m = mngs.general.listed_dict(PHASES)
        samp_s = mngs.general.listed_dict(PHASES)

        for i_phase, phase in enumerate(PHASES):

            rips_df_phase = rips_df[rips_df.phase == phase]

            centers_ms = []
            for delta_bin in range(-8, 9):
                col1 = f"{delta_bin-1}"
                col1_str = f"{int((delta_bin-1)*bin_size.magnitude)}"
                col2 = f"{delta_bin}"
                col2_str = f"{int((delta_bin)*bin_size.magnitude)}"

                centers_ms.append(int((delta_bin - 0.5) * bin_size.magnitude))

                # gets vectors
                v = np.vstack(rips_df_phase[col2]) - np.vstack(rips_df_phase[col1])
                if cbs is not None:  # basis transformation
                    v_base = np.vstack(rips_df_phase[cbe]) - np.vstack(
                        rips_df_phase[cbs]
                    )
                    v_rebased = [
                        rebase_a_vec(v[ii], v_base[ii]) for ii in range(len(v))
                    ]
                else:  # just the norm
                    v_rebased = [nannorm(v[ii]) for ii in range(len(v))]

                samp_m[phase].append(np.nanmean(v_rebased))
                n_samp = (~np.isnan(v_rebased)).sum()
                # print(n_samp)
                if n_samp != 0:
                    se = np.nanstd(v_rebased) / n_samp
                else:
                    se = np.nan
                samp_s[phase].append(se)

            ax.axhline(y=0, xmin=xlim[0], xmax=xlim[1], linestyle="--", color="gray")

            ax.errorbar(
                x=np.array(centers_ms) + i_phase * 5,
                y=samp_m[phase],
                yerr=samp_s[phase],
                label=phase,
            )
            ax.legend(loc="upper right")

        ax.set_xlim(xlim)
        if i_ax != 0:
            ax.set_ylim(ylim)
            title = dir_txt.replace("_based", " direction").replace("-", " -> ")
        else:
            title = "Norm"

        ax.set_title(title)

        samp_m = pd.DataFrame(samp_m)
        samp_m.columns = [f"{col}_{cbs}-{cbe}_med" for col in samp_m.columns]
        samp_s = pd.DataFrame(samp_s)
        samp_s.columns = [f"{col}_{cbs}-{cbe}_s" for col in samp_s.columns]
        
        # out_dict[f"{cbs}-{cbe}_med"] = samp_m
        # out_dict[f"{cbs}-{cbe}_se"] = samp_s

        out_df = pd.concat([out_df, pd.concat([samp_m, samp_s], axis=1)], axis=1)

    fig.supylabel("Norm")
    fig.supxlabel("Time from SWR [ms]")
    # plt.show()
    mngs.io.save(fig, f"./tmp/figs/hist/traj_norm/all.png")
    mngs.io.save(out_df, f"./tmp/figs/hist/traj_norm/all.csv")    
    plt.close()


def plot_positions(rips_df):
    fig, axes = plt.subplots(nrows=len(PHASES), sharex=True, sharey=True, figsize=(6.4*2, 4.8*2))
    xlim = (-1000, 1000)
    ylim = (-0.3, 0.3)

    out_df = pd.DataFrame()
    for i_ax, (ax, g_phase) in enumerate(zip(axes, PHASES)):

        samp_m = mngs.general.listed_dict(PHASES)
        samp_s = mngs.general.listed_dict(PHASES)
        
        for i_phase, rip_phase in enumerate(PHASES):
            rips_df_phase = rips_df[rips_df.phase == rip_phase]

            centers_ms = []

            for delta_bin in range(-8, 9):
                col1 = f"{delta_bin}"
                col1_str = f"{int((delta_bin)*bin_size.magnitude)}"

                centers_ms.append(int(delta_bin * bin_size.magnitude))

                # gets trajectory
                vv = np.vstack(rips_df_phase[col1])
                gg = np.vstack(rips_df_phase[g_phase])
                dist_gg = [nannorm(vv[ii] - gg[ii]) for ii in range(len(vv))]                

                n_samp = (~np.isnan(dist_gg)).sum()
                if n_samp != 0:
                    samp_m[rip_phase].append(np.nanmean(dist_gg))                    
                    se = np.nanstd(dist_gg) / n_samp
                    samp_s[rip_phase].append(se)
                else:
                    samp_m[rip_phase].append(np.nan)
                    samp_s[rip_phase].append(np.nan)                    

            ax.errorbar(
                x=np.array(centers_ms) + i_phase * 5,
                y=samp_m[rip_phase],
                yerr=samp_s[rip_phase],
                label=rip_phase,
            )

        samp_m = pd.DataFrame(samp_m)
        samp_m.columns = [f"{col}_distance_from_{g_phase}" for col in samp_m.columns]        
        samp_s = pd.DataFrame(samp_s)
        samp_s.columns = [f"{col}_distance_from_{g_phase}" for col in samp_s.columns]

        out_df = pd.concat([out_df, pd.concat([samp_m, samp_s], axis=1)], axis=1)

        ax.legend(loc="upper right")
        # ax.axhline(y=0, xmin=xlim[0], xmax=xlim[1], linestyle="--", color="gray")
        ax.set_xlim(xlim)
        ax.set_title(f"Distance from {g_phase}")

    mngs.io.save(fig, f"./tmp/figs/hist/traj_pos/all.png")
    mngs.io.save(out_df, f"./tmp/figs/hist/traj_pos/all.csv")            
    plt.show()


def calc_trajs_all(bin_size):    
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    n_bins = int((8 / bin_size.rescale("s")).magnitude)

    trajs_all = []
    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        for session in ["01", "02"]:
            # Loads
            LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
            spike_trains = to_spiketrains(mngs.io.load(LPATH))

            # GPFA
            gpfa = GPFA(bin_size=bin_size, x_dim=3)
            trajs = gpfa.fit_transform(spike_trains)
            trajs = normalize_trajs(trajs)                        
            trajs_all.append(np.stack(trajs, axis=0))

    return np.vstack(trajs_all)

def calc_position_matrix(trajs_all):
    ff = np.median(trajs_all[:, :, starts[0]:ends[0]], axis=-1) 
    ee = np.median(trajs_all[:, :, starts[1]:ends[1]], axis=-1) 
    mm = np.median(trajs_all[:, :, starts[2]:ends[2]], axis=-1) 
    rr = np.median(trajs_all[:, :, starts[3]:ends[3]], axis=-1) 

    xxs = {pp:xx for pp, xx in zip(PHASES, [ff, ee, mm, rr])}

    from itertools import product
    dist_df = pd.DataFrame(columns=PHASES, index=PHASES)
    for p1, p2 in product(PHASES, PHASES):
        x1 = xxs[p1]
        x2 = xxs[p2]        
        dist = [nannorm(x1[ii] - x2[ii]) for ii in range(len(x1))]
        dist_df.loc[p1, p2] = np.nanmean(dist)

    from pprint import pprint
    pprint(dist_df)
        



if __name__ == "__main__":
    import mngs
    import numpy as np

    # phase
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    bin_size = 200 * pq.ms
    starts, ends = [], []
    PHASE_START_END_DICT = {}
    for i_phase, phase in enumerate(PHASES):
        start_s = DURS_OF_PHASES[:i_phase].sum()
        end_s = DURS_OF_PHASES[: (i_phase + 1)].sum()
        PHASE_START_END_DICT[phase] = (start_s, end_s)
        starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
        ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
    colors = ["black", "blue", "green", "red"]

    # rips_df = load_rips_df_with_traj(bin_size)
    # rips_df = discard_last_ripples(rips_df)
    # mngs.io.save(rips_df, "./tmp/rips_df_with_traj.pkl")

    rips_df = mngs.io.load("./tmp/rips_df_with_traj.pkl")

    rips_df = add_coordinates(rips_df)

    # plot_cosine(rips_df)
    plot_norm_2(rips_df)
    plot_positions(rips_df)

    # trajs_all = calc_trajs_all(bin_size)
    # mngs.io.save(trajs_all, "./tmp/trajs_all.npy")
    trajs_all = mngs.io.load("./tmp/trajs_all.npy") # 493
    trajs_rip = np.vstack(rips_df.loc[rips_df[["subject", "session", "trial_number"]].drop_duplicates().index]["traj"])
    
