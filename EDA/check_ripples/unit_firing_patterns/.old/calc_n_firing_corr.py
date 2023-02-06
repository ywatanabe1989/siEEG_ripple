#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-24 16:28:05 (ywatanabe)"

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


# def extract_common_units(rip_1, rip_2):
#     def _in_extract_common_member(a, b):
#         a_set = set(a)
#         b_set = set(b)

#         if a_set & b_set:
#             return list(a_set & b_set)
#             # print(a_set & b_set)
#         else:
#             # print("No common elements")
#             return np.nan

#     pattern_1 = determine_firing_patterns(rip_1)
#     pattern_2 = determine_firing_patterns(rip_2)

#     import ipdb; ipdb.set_trace()

#     is_no_nan_patterns = bool(((pattern_1 is not np.nan) * (pattern_2 is not np.nan)))
#     if is_no_nan_patterns:

#         pattern_1_units = [
#             re.sub("_[0-9]_Trial_[0-9]{1,2}", "", unit)
#             for unit in list(pattern_1.keys())
#         ]
#         pattern_2_units = [
#             re.sub("_[0-9]_Trial_[0-9]{1,2}", "", unit)
#             for unit in list(pattern_2.keys())
#         ]
#         common_units = _in_extract_common_member(pattern_1_units, pattern_2_units)

#         if common_units is not np.nan:
#             pattern_1.index = pattern_1_units
#             pattern_2.index = pattern_2_units

#             common_1 = pattern_1[common_units]
#             common_2 = pattern_2[common_units]

#             return np.argsort(np.argsort(common_1)), np.argsort(np.argsort(common_2))

#     return np.nan, np.nan


def calc_dist(pattern_1, pattern_2):
    does_no_zero_patterns_exist = (pattern_1 == 0).all() + (pattern_2 == 0).all()

    if does_no_zero_patterns_exist:
        return np.nan

    else:
        dist = scipy.spatial.distance.cosine(pattern_1, pattern_2)
        return dist    
    # else:
    #     # return np.nan, np.nan
    #     return np.nan 


def surrogate(ns_units, ns_rep, actual_corrs=None, actual_pvals=None):
    def _in_surrogate(n_units, n_rep):
        corrs, pvals = [], []

        for _ in range(n_rep * 100):
            _common_1 = np.random.permutation(np.arange(n_units))
            _common_2 = np.random.permutation(np.arange(n_units))
            corr, pval = calc_rank_order_correlation(_common_1, _common_2)
            corrs.append(corr)
            pvals.append(pval)
        corrs = np.array(corrs)
        pvals = np.array(pvals)
        return corrs, pvals

    corrs, pvals = [], []
    for n_units, n_rep in tqdm(zip(ns_units, ns_rep)):
        _corrs, _pvals = _in_surrogate(n_units, n_rep)
        corrs.append(_corrs)
        pvals.append(_pvals)
    corrs = np.hstack(corrs)
    pvals = np.hstack(pvals)    

    fig, axes = plt.subplots(ncols=2)
    axes[0].hist(corrs, bins=20, alpha=.3, density=True, label="surrogate")
    axes[1].hist(pvals, bins=20, alpha=.3, density=True, label="surrogate")

    if actual_corrs is not None:
        axes[0].hist(actual_corrs, bins=20, alpha=.3, density=True, label="actual")
    if actual_pvals is not None:        
        axes[1].hist(actual_pvals, bins=20, alpha=.3, density=True, label="actual")

    fig.legend()
    return fig    


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

    rips_df = rips_df[rips_df["session"] <= 2].reset_index()
    del rips_df["index"]

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

                # koko
                # corr, pval = calc_correlation(pattern_1, pattern_2)                
                dist = calc_dist(pattern_1, pattern_2) 
                
                # common_1, common_2 = extract_common_units(rip_1, rip_2)
                # corr, pval = calc_rank_order_correlation(common_1, common_2)

                # n_common = np.nan if common_1 is np.nan else len(common_1)

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
                            # n_common=n_common,
                            # correlation=corr,
                            # p_value=pval,
                        )
                    )
                ).T

                dist_df = pd.concat([dist_df, dist_df_tmp])

            mngs.io.save(dist_df, "./tmp/dist_df.csv")
