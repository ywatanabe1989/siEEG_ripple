#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-24 15:08:02 (ywatanabe)"

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


def calc_rank_order_correlation(common_1, common_2):
    if common_1 is not np.nan:
        corr, pval = scipy.stats.spearmanr(common_1, common_2)
        return corr, pval
    else:
        return np.nan, np.nan


def surrogate(ns_units, ns_rep, n_factor=10, actual_corrs=None, actual_pvals=None):
    def _in_surrogate(n_units, n_rep):
        corrs, pvals = [], []

        for _ in range(n_rep * n_factor):
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
    axes[0].hist(corrs, bins=20, alpha=0.3, density=True, label="surrogate")
    axes[1].hist(pvals, bins=20, alpha=0.3, density=True, label="surrogate")

    if actual_corrs is not None:
        axes[0].hist(actual_corrs, bins=20, alpha=0.3, density=True, label="actual")

    if actual_pvals is not None:
        axes[1].hist(actual_pvals, bins=20, alpha=0.3, density=True, label="actual")

    axes[0].set_xlim(-1, 1)
    axes[1].set_xlim(0, 1)

    axes[0].set_xlabel("Rank-order correlation")
    axes[1].set_xlabel("P-value")        
        
    fig.legend()
    return fig


def get_phase_indi(df, phase_a, phase_b):
    if phase_a != "Any":
        indi_1 = df.phase_1 == phase_a
        indi_2 = df.phase_2 == phase_b

        indi_3 = df.phase_1 == phase_b
        indi_4 = df.phase_2 == phase_a

        indi = (indi_1 * indi_2) + (indi_3 * indi_4)

    else:
        indi = np.ones(len(df), dtype=bool)

    return indi


# surrogate(4)

if __name__ == "__main__":
    import mngs
    import matplotlib.pyplot as plt
    from itertools import combinations
    from time import sleep

    
    corr_df = mngs.io.load("./tmp/rank_order_corr_df.csv")
    phases = ["Fixation", "Encoding", "Maintenance", "Retrieval"]

    indi_session = np.ones(len(corr_df), dtype=bool)
    indi_letter = corr_df.probe_letter_1 == corr_df.probe_letter_2
    indi_trial = corr_df.trial_number_1 == corr_df.trial_number_2
    indi_within_groups_dict = dict(
        session=indi_session, letter=indi_letter, trial=indi_trial
    )

    phase_combis = [("Any", "Any")] + list(
        combinations(["Fixation", "Encoding", "Maintenance", "Retrieval"], 2)
    )

    for with_group_str, indi_within in indi_within_groups_dict.items():
        for i_phase_combi, (phase_a, phase_b) in enumerate(phase_combis):
            indi_phase_combi = get_phase_indi(corr_df, phase_a, phase_b)
            df = corr_df[indi_within * indi_phase_combi].copy()
            df = df[df.n_common > 5]

            # print(indi.sum())

            ns_common, counts = np.unique(df.n_common, return_counts=True)
            fig = surrogate(
                ns_common,
                counts,
                n_factor=10,
                actual_corrs=df.correlation,
                actual_pvals=df.p_value,
            )

            n_samp = len(df)
            title = f"{with_group_str}\n{phase_a}-{phase_b}\nn = {n_samp}"
            fig.suptitle(title)
            mngs.io.save(
                fig,
                f"./tmp/figs/unit_firing_patterns/{with_group_str}/{i_phase_combi}_{phase_a}-{phase_b}.png",
            )
            plt.close()
            # plt.show()

