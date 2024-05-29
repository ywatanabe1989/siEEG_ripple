#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-28 16:52:59 (ywatanabe)"
# calc_NT_with_GPFA.py


"""
This script does XYZ.
"""


"""
Imports
"""
import re
import sys
from glob import glob

import matplotlib.pyplot as plt
import mngs
import neo
import numpy as np
import quantities as pq
from elephant.gpfa import GPFA
from natsort import natsorted

"""
Config
"""
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def spiketimes_to_spiketrains(
    spike_times_all_trials, without_retrieval_phase=False
):
    spike_trains_all_trials = []
    for st_trial in spike_times_all_trials:
        spike_trains_trial = []
        for col, col_df in st_trial.items():
            spike_times = col_df[col_df != ""]
            if without_retrieval_phase:
                spike_times = spike_times[spike_times < 0]
                train = neo.SpikeTrain(
                    list(spike_times) * pq.s, t_start=-6.0, t_stop=0
                )
            else:
                train = neo.SpikeTrain(
                    list(spike_times) * pq.s, t_start=-6.0, t_stop=2.0
                )
            spike_trains_trial.append(train)

        spike_trains_all_trials.append(spike_trains_trial)

    return spike_trains_all_trials


def parse_lpath(lpath):
    subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
    session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
    roi = (
        re.findall("spike_times/[\w]{2,3}.pkl", lpath)[0]
        .split("spike_times/")[-1]
        .split(".pkl")[0]
    )
    return subject, session, roi


def switch_regarding_match(spike_trains, subject, session, match):
    if match != "all":
        trials_info = mngs.io.load(
            f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
        )
        indi = trials_info.match == match
        spike_trains = [
            spike_trains[ii] for ii, bl in enumerate(indi) if bl == True
        ]
    return spike_trains


def determine_spath(lpath_spike_times, match, without_retrieval_phase):
    spath_NTs = lpath_spike_times.replace("spike_times", "NT").replace(
        ".pkl", f"_match_{match}.npy"
    )
    if without_retrieval_phase:
        spath_NTs = spath_NTs.replace(".npy", "_wo_R.npy")
    return spath_NTs


def main(match="all", without_retrieval_phase=False):
    # Parameters
    BIN_SIZE = CONFIG["GPFA_BIN_SIZE_MS"] * pq.ms

    # Loads spike timings
    LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times/*.pkl"))

    for lpath in LPATHs:

        subject, session, roi = parse_lpath(lpath)

        # Spike trains of all trials; some of spike trains data are unavailable in the original datset.
        lpath_spike_times = (
            f"./data/Sub_{subject}/Session_{session}/spike_times/{roi}.pkl"
        )
        spike_times_all_trials = mngs.io.load(lpath_spike_times)

        spike_trains = spiketimes_to_spiketrains(
            spike_times_all_trials,
            without_retrieval_phase=without_retrieval_phase,
        )
        spike_trains = switch_regarding_match(
            spike_trains, subject, session, match
        )

        # GPFA calculation
        gpfa = GPFA(bin_size=BIN_SIZE, x_dim=3)
        try:
            NTs = np.stack(gpfa.fit_transform(spike_trains), axis=0)

            # Saving
            spath_NTs = determine_spath(
                lpath_spike_times, match, without_retrieval_phase
            )
            mngs.io.save(NTs, spath_NTs, from_git=True)

        except Exception as e:
            print(
                f"\nError raised during GPFA calculation. Spike_trains might be unavailable. "
                f"Skipping {spath_NTs}.:\n"
            )
            print(e)


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
    main(match="all", without_retrieval_phase=False)
    mngs.gen.close(CONFIG, verbose=False, notify=True)

# EOF
