#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-05 12:01:18 (ywatanabe)"

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
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def describe(df):
    df = pd.DataFrame(df)
    med = df.describe().T["50%"].iloc[0]
    IQR = df.describe().T["75%"].iloc[0] - df.describe().T["25%"].iloc[0]
    print(med, IQR)


def collect_cosine_EM_MR():
    LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times_*.pkl"))
    df = pd.DataFrame()
    for lpath in LPATHs:

        subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
        session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
        roi = (
            re.findall("spike_times_[\w]{2,3}.pkl", lpath)[0]
            .split("spike_times_")[1]
            .split(".pkl")[0]
        )
        _dict = dict(
            subject=subject,
            session=session,
            ROI=roi,
        )
        for match in [[1, 2]]:  # [[1], [1,2], [2]]:
            match_str = "all" if match == [1, 2] else ""
            _dict["match"] = match_str

            try:
                # Loads
                LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
                spike_times = mngs.io.load(LPATH)
                spike_trains = to_spiketrains(spike_times)
                trials_df = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
                )

                indi_b = (
                    np.vstack([trials_df.match == mm for mm in match])
                    .sum(axis=0)
                    .astype(bool)
                )
                spike_trains = (np.array(spike_trains)[indi_b]).tolist()
                # trials_df = trials_df.iloc[indi_b]

                # Parameters
                bin_size = 200 * pq.ms

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

                # GPFA
                gpfa = GPFA(bin_size=bin_size, x_dim=3)
                trajs = gpfa.fit_transform(spike_trains)
                traj_med = np.median(np.stack(trajs, axis=0).astype(float), axis=0)

                traj_phases = dict(
                    F=traj_med[:, starts[0] : ends[0]],
                    E=traj_med[:, starts[1] : ends[1]],
                    M=traj_med[:, starts[2] : ends[2]],
                    R=traj_med[:, starts[3] : ends[3]],
                )

                med_E = np.median(traj_phases["E"], axis=-1)
                med_M = np.median(traj_phases["M"], axis=-1)
                med_R = np.median(traj_phases["R"], axis=-1)

                EM_MR_dist = calc_cosine(med_M - med_E, med_R - med_M)
                print(
                    f"\nSubject {subject}, Seesion {session}, ROI {roi}, match {match}\n"
                )
                print(f"\nE -> M and M -> R distance:\n{EM_MR_dist}\n")

                _dict["EM_MR_dist"] = round(EM_MR_dist, 3)
                df = pd.concat([df, pd.DataFrame(pd.Series(_dict))], axis=1)

            except Exception as e:
                print(e)
    return df


if __name__ == "__main__":
    from glob import glob
    import re
    from natsort import natsorted

    df = collect_cosine_EM_MR()
    mngs.io.save(df, "./tmp/EM_MR_cosine.csv")

    # here
    df = mngs.io.load("./tmp/EM_MR_cosine.csv").T
    df.columns = list(df.loc["Unnamed: 0"])
    df = df.iloc[1:, :]
    df["EM_MR_dist"] = df["EM_MR_dist"].astype(float)
    df = df[df.session.astype(int) <= 2]

    ripple_rois = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    indi = []
    for subject, roi in ripple_rois.items():
        subject = f"{int(subject):02d}"
        indi.append((df.subject == subject) * (df.ROI == roi))
    indi = np.vstack(indi).sum(axis=0).astype(bool)

    # plots
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    for ii, (ax, _indi) in enumerate(zip(axes, [indi, ~indi])):
        sns.boxplot(
            data=df[_indi],
            y="EM_MR_dist",
            x="ROI",
            order=["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"],
            ax=ax,
        )
        sns.stripplot(
            data=df[_indi],
            y="EM_MR_dist",
            x="ROI",
            order=["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"],
            color="black",
            alpha=0.5,
            ax=ax,
        )
        ax.set_ylabel("Cosine(EM, MR)")
        ax.set_ylim(-1.1, 1.1)
        title = (
            "Ripple-detectable channels" if ii == 0 else "Ripple-undetectable channels"
        )
        ax.set_title(title)
    mngs.io.save(fig, "./tmp/figs/hist/cosine_EM_MR.png")

    # stats
    EM_MR_ripple = df[indi].EM_MR_dist
    EM_MR_others = df[~indi].EM_MR_dist
    EM_MR_amy = df[(df.ROI == "AL") + (df.ROI == "AR")][~indi].EM_MR_dist
    EM_MR_rand = [2 * random.Random(ii).random() - 1 for ii in range(len(df[indi]))]

    describe(EM_MR_ripple)
    describe(EM_MR_rand)

    print(brunnermunzel(EM_MR_ripple, EM_MR_others))
    print(brunnermunzel(EM_MR_ripple, EM_MR_rand))
    print(brunnermunzel(EM_MR_ripple, EM_MR_amy))


    # to csv
    df[["Hipp", "EC", "Amy."]] = False
    df.loc[mngs.general.search(["AHL", "AHR", "PHL", "PHR"], df.ROI, as_bool=True)[0], "Hipp"] = True
    df.loc[mngs.general.search(["ECL", "ECR"], df.ROI, as_bool=True)[0], "EC"] = True
    df.loc[mngs.general.search(["AL", "AR"], df.ROI, as_bool=True)[0], "Amy."] = True        

    df["is_CA1"] = False
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    for subject, roi in ROIs.items():
        subject = f"{int(subject):02d}"
        df.loc[(df.subject == subject) * (df.ROI == roi), "is_CA1"] = True

    df = df.sort_values(["Hipp", "is_CA1", "EC", "Amy."])
    df = df[["subject", "session", "ROI", "EM_MR_dist", "Hipp", "is_CA1", "EC", "Amy."]]

    mngs.io.save(df, "./tmp/figs/hist/cosine_EM_MR.csv")

    describe(df[df.Hipp == True]["EM_MR_dist"])
    describe(df[df.EC == True]["EM_MR_dist"])
    describe(df[df["Amy."] == True]["EM_MR_dist"])    

    EM_MR_rand = [2 * random.Random(ii).random() - 1 for ii in range(df.Hipp.sum())]
    brunnermunzel(
        df[df.Hipp == True]["EM_MR_dist"],
        EM_MR_rand,
        )

    EM_MR_rand = [2 * random.Random(ii).random() - 1 for ii in range(df.EC.sum())]
    brunnermunzel(
        df[df.EC == True]["EM_MR_dist"],
        EM_MR_rand,
        )

    EM_MR_rand = [2 * random.Random(ii).random() - 1 for ii in range(df["Amy."].sum())]
    brunnermunzel(
        df[df["Amy."]]["EM_MR_dist"],
        EM_MR_rand,
        )
    
    brunnermunzel(
        df[df.Hipp == True]["EM_MR_dist"],
        df[df.EC == True]["EM_MR_dist"],
        )

    brunnermunzel(
        df[df.EC == True]["EM_MR_dist"],
        df[df["Amy."] == True]["EM_MR_dist"],
        )    

    brunnermunzel(
        df[df["Amy."] == True]["EM_MR_dist"],
        df[df.Hipp == True]["EM_MR_dist"],
        )    

    (df.Hipp == True).sum()
    (df.EC == True).sum()
    (df["Amy."] == True).sum()    
