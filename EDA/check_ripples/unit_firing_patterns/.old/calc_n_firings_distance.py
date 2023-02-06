#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-26 18:03:18 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
import warnings
import re
import random
import scipy
import mngs
from tqdm import tqdm
import matplotlib.pyplot as plt


def determine_firing_patterns(rip):
    try:
        sub = f"{int(rip.subject):02d}"
    except:
        return np.nan

    session = f"{int(rip.session):02d}"

    try:
        roi = ROIs[int(sub)]
    except:
        return np.nan

    i_trial = int(rip.trial_number) - 1
    spike_times = mngs.io.load(
        f"./data/Sub_{sub}/Session_{session}/spike_times_{roi}.pkl"
    )[i_trial].replace({"": np.nan})
    spike_pattern = spike_times[
        ((rip.start_time - 6 < spike_times) * (spike_times < rip.end_time - 6))
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        spike_pattern = (~spike_pattern.isna()).sum()
        # spike_pattern = spike_pattern.apply(np.nanmean)

        # spike_pattern = spike_pattern[~spike_pattern.isna()]

        # if spike_pattern.empty:
        #     return np.nan

    # spike_pattern = np.argsort(np.argsort(spike_pattern))
    return spike_pattern


def calc_dist(pattern_1, pattern_2):
    does_no_zero_patterns_exist = (pattern_1 == 0).all() + (pattern_2 == 0).all()

    if does_no_zero_patterns_exist:
        return np.nan

    else:
        dist = scipy.spatial.distance.cosine(pattern_1, pattern_2)
        return dist


if __name__ == "__main__":
    from itertools import combinations

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    rips_df = []
    for sub, roi in ROIs.items():
        rips_df_roi = mngs.io.load(
            f"./tmp/rips_df/common_average_2.0_SD_{roi}.csv"
        ).rename(columns={"Unnamed: 0": "trial_number"})
        rips_df_roi = rips_df_roi[rips_df_roi["subject"] == int(sub)]
        rips_df_roi["ROI"] = roi
        rips_df.append(rips_df_roi)
    rips_df = pd.concat(rips_df)

    # rips_df = rips_df[rips_df["session"] <= 2]
    rips_df.reset_index(inplace=True)
    del rips_df["index"]

    # np.unique(rips_df.subject, return_counts=True)
    # (array([2., 3., 4., 6., 7.]), array([ 985,  583,  328, 1004,  616]))

    columns = ["i_rip_1", "i_rip_2", "n_common", "distelation", "pval"]
    dist_df = pd.DataFrame(columns=columns)

    for sub in rips_df.subject.unique():
        rips_df_sub = rips_df[rips_df.subject == sub]

        for session in tqdm(rips_df_sub.session.unique()):
            trials_info = mngs.io.load(
                f"./data/Sub_{int(sub):02d}/Session_{int(session):02d}/trials_info.csv"
            )

            rips_df_session = rips_df_sub[rips_df_sub.session == session]

            for i_rip_1, i_rip_2 in combinations(np.arange(len(rips_df_session)), 2):

                columns = ["i_rip1", "i_rip2", "n_common", "distance", "pval"]

                rip_1 = rips_df_session.iloc[i_rip_1]
                rip_2 = rips_df_session.iloc[i_rip_2]

                pattern_1 = determine_firing_patterns(rip_1)
                pattern_2 = determine_firing_patterns(rip_2)

                dist = calc_dist(pattern_1, pattern_2)

                probe_letter_1 = trials_info.iloc[
                    int(rip_1.trial_number - 1)
                ].probe_letter

                probe_letter_2 = trials_info.iloc[
                    int(rip_2.trial_number - 1)
                ].probe_letter

                dist_df_tmp = pd.DataFrame(
                    pd.Series(
                        dict(
                            subject=sub,
                            session=session,
                            i_rip_1=i_rip_1,
                            i_rip_2=i_rip_2,
                            phase_1=rip_1.phase,
                            phase_2=rip_2.phase,
                            probe_letter_1=probe_letter_1,
                            probe_letter_2=probe_letter_2,
                            trial_number_1=rip_1.trial_number,
                            trial_number_2=rip_2.trial_number,
                            distance=dist,
                            firing_pattern_1=pattern_1,
                            firing_pattern_2=pattern_2,
                            rip_1_dur_ms=(rip_1.end_time - rip_1.start_time) * 1000,
                            rip_2_dur_ms=(rip_2.end_time - rip_2.start_time) * 1000,
                        )
                    )
                ).T

                dist_df = pd.concat([dist_df, dist_df_tmp])

            mngs.io.save(dist_df, "./tmp/dist_df.pkl")
