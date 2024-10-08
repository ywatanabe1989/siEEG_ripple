#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-06 07:50:26 (ywatanabe)"

import matplotlib
import mngs
import neo
import numpy as np
import pandas as pd
import quantities as pq
from elephant.gpfa import GPFA

matplotlib.use("TkAgg")
import re
from glob import glob

import matplotlib.pyplot as plt
from natsort import natsorted


# Functions
def to_spiketrains(spike_times_all_trials, without_retrieval_phase=False):
    spike_trains_all_trials = []
    for st_trial in spike_times_all_trials:

        spike_trains_trial = []
        for col, col_df in st_trial.iteritems():

            spike_times = col_df[col_df != ""]
            if without_retrieval_phase:
                spike_times = spike_times[spike_times < 0]
                train = neo.SpikeTrain(list(spike_times) * pq.s, t_start=-6.0, t_stop=0)
            else:
                train = neo.SpikeTrain(
                    list(spike_times) * pq.s, t_start=-6.0, t_stop=2.0
                )
            spike_trains_trial.append(train)

        spike_trains_all_trials.append(spike_trains_trial)

    return spike_trains_all_trials


def main(match, without_retrieval_phase=False):
    # Parameters
    bin_size = 50 * pq.ms

    # Loads
    LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times_*.pkl"))

    for lpath in LPATHs:

        subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
        session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
        roi = (
            re.findall("spike_times_[\w]{2,3}.pkl", lpath)[0]
            .split("spike_times_")[1]
            .split(".pkl")[0]
        )

        try:
            lpath_spike_times = (
                f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
            )
            spike_trains = to_spiketrains(
                mngs.io.load(lpath_spike_times),
                without_retrieval_phase=without_retrieval_phase,
            )

            if match is not None:
                trials_info = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
                )
                indi = trials_info.match == match
                spike_trains = [
                    spike_trains[ii] for ii, bl in enumerate(indi) if bl == True
                ]

            gpfa = GPFA(bin_size=bin_size, x_dim=3)
            trajs = gpfa.fit_transform(spike_trains)
            trajs = np.stack(trajs, axis=0)
            spath_trajs = (
                lpath_spike_times.replace("spike_times", "traj")
                .replace(".pkl", ".npy")
                .replace(".npy", f"_match_{match}.npy")
            )
            if without_retrieval_phase:
                spath_trajs = spath_trajs.replace(".npy", "_wo_R.npy")
            mngs.io.save(trajs, spath_trajs)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    main(match=None, without_retrieval_phase=True)    
    # for match in [1, 2]:
    #     main(match, without_retrieval_phase=False)

    # import mngs
    # mngs.io.load("./data/Sub_01/Session_01/traj_AHL_match_None_wo_R.npy")
