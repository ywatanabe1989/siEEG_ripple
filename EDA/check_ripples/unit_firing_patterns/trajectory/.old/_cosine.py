#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-15 09:49:22 (ywatanabe)"

from elephant.gpfa import GPFA
import quantities as pq
import mngs
import neo
import numpy as np
import pandas as pd
from numpy.linalg import norm
import mngs
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import seaborn as sns
import random
from scipy.stats import brunnermunzel
from itertools import product, combinations

# Functions
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


def calc_cosine(v1, v2):
    if np.isnan(v1).any():
        return np.nan
    if np.isnan(v2).any():
        return np.nan
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def describe(df):
    df = pd.DataFrame(df)
    med = df.describe().T["50%"].iloc[0]
    IQR = df.describe().T["75%"].iloc[0] - df.describe().T["25%"].iloc[0]
    print(med, IQR)


# def collect_cosine_EM_MR():
#     LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times_*.pkl"))
#     df = pd.DataFrame()
#     for lpath in LPATHs:

#         subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
#         session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
#         roi = (
#             re.findall("spike_times_[\w]{2,3}.pkl", lpath)[0]
#             .split("spike_times_")[1]
#             .split(".pkl")[0]
#         )
#         _dict = dict(
#             subject=subject,
#             session=session,
#             ROI=roi,
#         )

#         # Loads
#         LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
#         spike_trains = to_spiketrains(mngs.io.load(LPATH))

#         # Parameters
#         bin_size = 200 * pq.ms

#         # phase
#         PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
#         DURS_OF_PHASES = np.array(
#             mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"]
#         )
#         global starts, ends, colors
#         starts, ends = [], []
#         for i_phase in range(len(PHASES)):
#             start_s = DURS_OF_PHASES[:i_phase].sum()
#             end_s = DURS_OF_PHASES[: (i_phase + 1)].sum()
#             starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
#             ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
#         colors = ["black", "blue", "green", "red"]

#         # GPFA
#         gpfa = GPFA(bin_size=bin_size, x_dim=3)
#         trajs = gpfa.fit_transform(spike_trains)
#         traj_med = np.median(np.stack(trajs, axis=0).astype(float), axis=0)

#         g_phases = {
#             "Fixations": np.median(traj_med[:, starts[0] : ends[0]], axis=-1),
#             "Encoding": np.median(traj_med[:, starts[1] : ends[1]], axis=-1),
#             "Maintenance": np.median(traj_med[:, starts[2] : ends[2]], axis=-1),
#             "Retrieval": np.median(traj_med[:, starts[3] : ends[3]], axis=-1),
#         }

#         from itertools import product, combinations
#         combi = list(combinations(PHASES, 2))
#         for i_c, (c1, c2) in enumerate(product(combi, combi)):
#             p1, p2 = c1
#             p3, p4 = c2
#             v1 = g_phases[p2] - g_phases[p1]
#             v2 = g_phases[p4] - g_phases[p3]
#         import ipdb; ipdb.set_trace()

#         med_E = np.median(traj_phases["E"], axis=-1)
#         med_M = np.median(traj_phases["M"], axis=-1)
#         med_R = np.median(traj_phases["R"], axis=-1)

#         EM_MR_dist = calc_cosine(med_M - med_E, med_R - med_M)
#         print(
#             f"\nSubject {subject}, Seesion {session}, ROI {roi}, match {match}\n"
#         )
#         print(f"\nE -> M and M -> R distance:\n{EM_MR_dist}\n")

#         _dict["EM_MR_dist"] = round(EM_MR_dist, 3)
#         df = pd.concat([df, pd.DataFrame(pd.Series(_dict))], axis=1)
#     return df


def main():
    # Parameters
    bin_size = 200 * pq.ms

    # Loads
    LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times_*.pkl"))

    dfs = []
    subject_session_roi_df = pd.DataFrame()
    for lpath in LPATHs:

        subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
        session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
        roi = (
            re.findall("spike_times_[\w]{2,3}.pkl", lpath)[0]
            .split("spike_times_")[1]
            .split(".pkl")[0]
        )

        try:
            if int(session) <= 2:

                spike_trains = to_spiketrains(
                    mngs.io.load(
                        f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
                    )
                )

                gpfa = GPFA(bin_size=bin_size, x_dim=3)
                trajs = gpfa.fit_transform(spike_trains)
                traj_med = np.median(np.stack(trajs, axis=0).astype(float), axis=0)

                # phase
                PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
                DURS_OF_PHASES = np.array(
                    mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"]
                )
                global starts, ends, colors
                starts, ends = [], []
                for i_phase in range(len(PHASES)):
                    start_s = DURS_OF_PHASES[:i_phase].sum()
                    end_s = DURS_OF_PHASES[: (i_phase + 1)].sum()
                    starts.append(int(start_s / (bin_size.rescale("s").magnitude)))
                    ends.append(int(end_s / (bin_size.rescale("s").magnitude)))
                colors = ["black", "blue", "green", "red"]

                g_phases = {
                    "Fixation": np.median(traj_med[:, starts[0] : ends[0]], axis=-1),
                    "Encoding": np.median(traj_med[:, starts[1] : ends[1]], axis=-1),
                    "Maintenance": np.median(traj_med[:, starts[2] : ends[2]], axis=-1),
                    "Retrieval": np.median(traj_med[:, starts[3] : ends[3]], axis=-1),
                }

                combi = list(combinations(PHASES, 2))
                combi_str = [cc[0] + "-" + cc[1] for cc in combi]
                df = pd.DataFrame(index=combi_str, columns=combi_str)
                for i_c, (c1, c2) in enumerate(product(combi, combi)):
                    p1, p2 = c1
                    p3, p4 = c2
                    v1 = g_phases[p2] - g_phases[p1]
                    v1_str = p1 + "-" + p2
                    v2 = g_phases[p4] - g_phases[p3]
                    v2_str = p3 + "-" + p4
                    df.loc[v1_str, v2_str] = calc_cosine(v1, v2)
                dfs.append(df)
                subject_session_roi_df = pd.concat(
                    [
                        subject_session_roi_df,
                        pd.DataFrame(
                            pd.Series(
                                {
                                    "subject": subject,
                                    "session": session,
                                    "ROI": roi,
                                }
                            )
                        ),
                    ],
                    axis=1,
                )
        except Exception as e:
            print(e)
    return dfs, subject_session_roi_df

def calc_med(dfs):
    from itertools import combinations

    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    df_med = np.nanmedian(dfs, axis=0)
    combi = list(combinations(PHASES, 2))
    combi_str = [cc[0] + "-" + cc[1] for cc in combi]
    df_med = pd.DataFrame(index=combi_str, columns=combi_str, data=df_med)
    return df_med

def to_phase_vectors(data):
    from itertools import combinations

    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    # df_med = np.nanmedian(dfs, axis=0)
    combi = list(combinations(PHASES, 2))
    combi_str = [cc[0] + "-" + cc[1] for cc in combi]
    return pd.DataFrame(index=combi_str, columns=combi_str, data=data)

if __name__ == "__main__":
    from glob import glob
    import re
    from natsort import natsorted

    dfs, subject_session_roi_df = main()

    mngs.io.save(dfs, "./tmp/cos/dfs.pkl")
    mngs.io.save(subject_session_roi_df, "./tmp/cos/subject_session_roi.pkl")

    dfs = mngs.io.load("./tmp/cos/dfs.pkl")
    dfs = np.stack(dfs, axis=0).astype(float)
    subject_session_roi_df = subject_session_roi_df.T


    indi_hipp = mngs.general.search(
        ["AHL", "AHR", "PHL", "PHR"], subject_session_roi_df.ROI, as_bool=True
    )[0]
    indi_ec = mngs.general.search(
        ["ECL", "ECR"], subject_session_roi_df.ROI, as_bool=True
    )[0]
    indi_amy = mngs.general.search(
        ["AL", "AR"], subject_session_roi_df.ROI, as_bool=True
    )[0]

    mngs.io.save(calc_med(dfs[indi_hipp]), "./tmp/cos/df_med_hipp.csv")
    mngs.io.save(calc_med(dfs[indi_ec]), "./tmp/cos/df_med_EC.csv")
    mngs.io.save(calc_med(dfs[indi_amy]), "./tmp/cos/df_med_amy.csv")

    fig, ax = plt.subplots(figsize=(6.4 * 3, 4.8 * 3))
    sns.heatmap(calc_med(dfs[indi_hipp]), ax=ax, vmin=-1, vmax=1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.show()

    # cos(EM, MR)
    hipp_data = [
        to_phase_vectors(dfs[indi_hipp][ii]).loc[
            "Encoding-Maintenance", "Maintenance-Retrieval"
        ]
        for ii in range(indi_hipp.sum())
    ]
    ec_data = [
        to_phase_vectors(dfs[indi_ec][ii]).loc[
            "Encoding-Maintenance", "Maintenance-Retrieval"
        ]
        for ii in range(indi_ec.sum())
    ]
    amy_data = [
        to_phase_vectors(dfs[indi_amy][ii]).loc[
            "Encoding-Maintenance", "Maintenance-Retrieval"
        ]
        for ii in range(indi_amy.sum())
    ]
    n_sim = int(np.mean([indi_hipp.sum(), indi_ec.sum(), indi_amy.sum()]))
    rs = np.random.RandomState(42)
    sim_data = 2 * rs.rand(n_sim) - 1

    # import matplotlib
    # matplotlib.use("TkAgg")
    # plt.box(sim_data)

    # from scipy.stats import brunnermunzel


    print(brunnermunzel(hipp_data, sim_data)) #
    # BrunnerMunzelResult(statistic=2.9111842977034446, pvalue=0.005998147164847811)
    print(brunnermunzel(ec_data, sim_data))
    # BrunnerMunzelResult(statistic=2.73282702097557, pvalue=0.010247385125593533)
    print(brunnermunzel(amy_data, sim_data))
    # BrunnerMunzelResult(statistic=1.6096101946350319, pvalue=0.1176303338132123)    

    df = mngs.general.force_dataframe({
        "Hipp.": hipp_data,
        "EC": ec_data,
        "Amy.": amy_data,
        "Sim.": sim_data,
        })
    mngs.io.save(df, "./tmp/cos/df_EM_MR.csv")
