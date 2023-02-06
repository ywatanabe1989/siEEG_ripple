#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-31 16:56:06 (ywatanabe)"

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
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def production_vector(v1, v0):
    """
    production_vector(np.array([3,4]), np.array([10,0])) # np.array([3, 0])
    """
    return norm(v1) * cosine(v1, v0) * v0 / norm(v0)


def rebase_a_vec(v, v_base):
    v_prod = production_vector(v, v_base)
    sign = np.sign(cosine(v, v_base))
    return sign * norm(v_prod)


def discard_initial_ripples(rips_df, width_s=0.3):
    indi = []
    starts = [0, 1, 3, 6]
    ends = [1, 3, 6, 8]
    for ss, ee in zip(starts, ends):
        indi.append((ss + width_s < rips_df.center_time) * (rips_df.center_time < ee))
    indi = np.vstack(indi).sum(axis=0)
    return rips_df[indi]


def collect_ER_based_vec():
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    bin_size = 200 * pq.ms
    n_bins = int((8 / bin_size.rescale("s")).magnitude)

    global starts, ends, colors
    starts, ends = [], []
    for i_phase in range(len(PHASES)):
        start_s = DURS_OF_PHASES[:i_phase].sum()
        end_s = DURS_OF_PHASES[: (i_phase + 1)].sum()
        starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
        ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
    colors = ["black", "blue", "green", "red"]

    all_rips = []
    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        rips_df = mngs.io.load(f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl")
        rips_df = discard_initial_ripples(rips_df)
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

            # # min max norm
            # trajs = np.stack(trajs, axis=0)
            # trajs_min = trajs.min(axis=0, keepdims=True).min(axis=-1, keepdims=True)
            # trajs -= trajs_min
            # trajs_max = trajs.max(axis=0, keepdims=True).max(axis=-1, keepdims=True)
            # trajs /= trajs_max

            for i_trial in range(len(trajs)):
                traj = trajs[i_trial]
                rip_bins = get_i_bin(
                    rips_df_session[rips_df_session.index == i_trial + 1].center_time,
                    bin_size,
                    n_bins,
                )

                # for delta_bin_start in [-5, -3, -1, 1, 3]:
                #     rips_df_session = get_traj_tau(rips_df_session, delta_bin_start, 2)
                # for delta_bin_start in [-7, -4, -1, 2, 5]:
                #     rips_df_session = get_traj_tau(rips_df_session, delta_bin_start, 3)
                # for delta_bin_start in [-5, -3, -1, 1, 3]:
                #     rips_df_session = get_traj_tau(rips_df_session, delta_bin_start, 3)
                for delta_bin_start in [-5, -3, -1, 1, 3]:
                    rips_df_session = get_traj_tau(
                        rips_df_session, rip_bins, traj, i_trial, delta_bin_start, 2
                    )
                # for delta_bin_start in [-4, -1, 2]:
                #     rips_df_session = get_traj_tau(rips_df_session, delta_bin_start, 3)
                # for delta_bin_start in [-12, -7, -2, 3, 8]:
                #     rips_df_session = get_traj_tau(rips_df_session, delta_bin_start, 4)
                # for delta_bin_start in [-12, -7, -2, 3, 8]:
                #     rips_df_session = get_traj_tau(rips_df_session, delta_bin_start, 5)

            rips_df_session = rips_df_session.reset_index()
            all_rips.append(rips_df_session)
    all_rips = pd.concat(all_rips).reset_index()
    return all_rips


def get_traj_tau(rips_df_session, rip_bins, traj, i_trial, delta_bin_start, width_bin):
    traj_F_med = np.median(traj[:, starts[0] : ends[0]], axis=-1)
    traj_E_med = np.median(traj[:, starts[1] : ends[1]], axis=-1)
    traj_M_med = np.median(traj[:, starts[2] : ends[2]], axis=-1)
    traj_R_med = np.median(traj[:, starts[3] : ends[3]], axis=-1)

    delta_bin_end = delta_bin_start + width_bin

    col = f"ER_based_vec_{delta_bin_start}-{delta_bin_end}"
    if col not in rips_df_session.columns:
        rips_df_session[col] = np.nan

    ER_based_vec_tau = []
    ER_based_vec_tau_control = []

    for control in [False, True]:
        for rip_bin in rip_bins:

            if control:
                for ii in range(len(starts)):
                    if (starts[ii] <= rip_bin) and (rip_bin < ends[ii]):
                        rip_bin = random.randint(starts[ii], ends[ii])

            traj_tau = traj[:, rip_bin + delta_bin_start : rip_bin + delta_bin_end]

            try:
                _ER_based_vec_tau = rebase_a_vec(
                    traj_tau[:, -1] - traj_tau[:, 0],
                    # traj_R_med - traj_E_med, # fixme
                    # traj_M_med - traj_E_med,
                    # traj_E_med - traj_F_med, # E
                    traj_R_med - traj_M_med,
                )
            except Exception as e:
                print(e)
                _ER_based_vec_tau = np.nan

            if not control:
                ER_based_vec_tau.append(_ER_based_vec_tau)
            else:
                ER_based_vec_tau_control.append(_ER_based_vec_tau)

    rips_df_session.loc[(rips_df_session.index == i_trial + 1), col] = ER_based_vec_tau
    rips_df_session.loc[
        (rips_df_session.index == i_trial + 1), "control_" + col
    ] = ER_based_vec_tau_control
    return rips_df_session


if __name__ == "__main__":
    import mngs

    # phase
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])

    all_rips = collect_ER_based_vec()
    cols = mngs.general.search("^ER_based_vec_*", all_rips.columns)[1]
    cols_control = mngs.general.search("control_ER_based_vec_*", all_rips.columns)[1]

    # koko
    # plots
    fig, axes = plt.subplots(
        ncols=len(all_rips.phase.unique()),
        nrows=len(cols),
        sharex=True,
        sharey=True,
        figsize=(6.4 * 2, 4.8 * 2),
    )
    for i_col, (col, col_control) in enumerate(zip(cols, cols_control)):
        for i_phase, phase in enumerate(PHASES):
            ax = axes[i_col, i_phase]

            sns.histplot(
                data=all_rips[all_rips.phase == phase],
                x=col,
                common_norm=False,
                stat="probability",
                kde=True,
                ax=ax,
                legend=False,
                alpha=0.3,
                color="blue",
            )

            sns.histplot(
                data=all_rips[all_rips.phase == phase],
                x=col_control,
                common_norm=False,
                stat="probability",
                kde=True,
                ax=ax,
                legend=False,
                alpha=0.3,
                color="red",
            )

            ax.axvline(
                x=np.nanmedian(all_rips[all_rips.phase == phase][col]), color="blue"
            )
            ax.axvline(
                x=np.nanmedian(all_rips[all_rips.phase == phase][col_control]),
                color="red",
            )

            # ax.set_ylim(0, 0.3)
            # ax.set_xlim(-1, 1)

            if i_col == 0:
                ax.set_title(f"{phase}")

            ax.set_ylabel(col)
            # if i_phase == 0:
            #     if x[13:] == "pre_pre":
            #         ylabel = "-1,400 - -1,000 ms"
            #     if x[13:] == "pre":
            #         ylabel = "-800 - -400 ms"
            #     if x[13:] == "rip":
            #         ylabel = "-200 - +200 ms"
            #     if x[13:] == "post":
            #         ylabel = "+400 - +800 ms"
            #     if x[13:] == "post_post":
            #         ylabel = "+1,000 - +1,400 ms"
            #     ax.set_ylabel(f"Ripple\n{ylabel}\nProbability")

            ax.set_xlabel("")

            fig.supxlabel("ER-based trajectory vector")

            samp1 = all_rips[all_rips.phase == phase][col_control][
                ~all_rips[all_rips.phase == phase][col_control].isna()
            ]
            samp2 = all_rips[all_rips.phase == phase][col][
                ~all_rips[all_rips.phase == phase][col].isna()
            ]

            stats, pval = brunnermunzel(samp1, samp2, alternative="less")
            # stats, pval = ttest_ind(samp1, samp2)

            ax.text(x=0.5, y=0.2, s=pval.round(3))

            print(col, col_control, phase)
            print(pval)
            print()
            # fig.suptitle("ER-based trajectory around ripples")

    # mngs.io.save(fig, "./tmp/figs/hist/ER_based_trajectory_arround_ripples.png")
    plt.show()
