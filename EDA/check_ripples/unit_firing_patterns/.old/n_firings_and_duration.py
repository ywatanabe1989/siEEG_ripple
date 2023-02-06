#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-26 23:37:48 (ywatanabe)"

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
import itertools
import seaborn as sns


def perm(n, seq):
    out = []
    for p in itertools.product(seq, repeat=n):
        out.append(p)
    return out
    # print(p)
    # file.write("".join(p))
    # file.write("\n")


def calc_rank_order_correlation(common_1, common_2):
    if common_1 is not np.nan:
        corr, pval = scipy.stats.spearmanr(common_1, common_2)
        return corr, pval
    else:
        return np.nan, np.nan


def add_match_correct(df):
    matches = []
    corrects = []
    response_times = []
    for i_row, row in df.iterrows():
        trials_info = mngs.io.load(
            f"./data/Sub_{int(row.subject):02d}/"
            f"Session_{int(row.session):02d}/trials_info.csv"
        )
        match = trials_info.iloc[int(row.trial_number_1) - 1].match
        correct = trials_info.iloc[int(row.trial_number_1) - 1].correct
        response_time = trials_info.iloc[int(row.trial_number_1) - 1].response_time
        matches.append(match)
        corrects.append(correct)
        response_times.append(response_time)

    df["match"] = matches
    df["correct"] = corrects
    df["response_time"] = response_times
    return df


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


def mk_indi_within_groups(dist_df):
    indi_session = np.ones(len(dist_df), dtype=bool)
    indi_letter = dist_df.probe_letter_1 == dist_df.probe_letter_2
    indi_trial = dist_df.trial_number_1 == dist_df.trial_number_2
    indi_within_groups_dict = dict(
        session=indi_session, letter=indi_letter, trial=indi_trial
    )
    return indi_within_groups_dict


if __name__ == "__main__":
    import mngs
    import matplotlib.pyplot as plt
    from itertools import combinations
    from time import sleep

    dist_df = mngs.io.load("./tmp/dist_df.pkl")
    phases = ["Fixation", "Encoding", "Maintenance", "Retrieval"]

    indi_within_groups_dict = mk_indi_within_groups(dist_df)

    phase_combis = [("Any", "Any")] + perm(
        2, ["Fixation", "Encoding", "Maintenance", "Retrieval"]
    )

    for with_group_str, indi_within in indi_within_groups_dict.items():
        for i_phase_combi, (phase_a, phase_b) in enumerate(phase_combis):
            print(phase_a)
            print(phase_b)
            """
            indi_within = indi_within_groups_dict["trial"]
            phase_a = "Encoding"
            phase_b = "Retrieval"            
            """

            indi_phase_combi = get_phase_indi(dist_df, phase_a, phase_b)
            df = dist_df[indi_within * indi_phase_combi].copy()

            fig, ax = plt.subplots()
            ax.hist(df.distance, density=True)
            ax.set_xlabel("Cosine distance between unit firings of ripple pairs")
            ax.set_ylabel("Ripple probability")

            median = np.nanmedian(np.array(df.distance).astype(float))
            ax.axvline(x=median, color="gray", linestyle="dotted")

            ax.set_ylim(0, 2.5)

            # df.distance.hist()
            # df = df[df.n_common > 5]

            # print(indi.sum())

            # ns_common, counts = np.unique(df.n_common, return_counts=True)
            # fig = surrogate(
            #     ns_common,
            #     counts,
            #     n_factor=10,
            #     actual_dists=df.distance,
            #     # actual_pvals=df.p_value,
            # )

            n_samp = len(df)
            title = f"{with_group_str}\n{phase_a}-{phase_b}\nn = {n_samp}"
            fig.suptitle(title)
            if (phase_a == "Encoding") * (phase_b == "Retrieval"):

                _df = df[(df.trial_number_1 == df.trial_number_2)][
                    [
                        "subject",
                        "session",
                        "trial_number_1",
                        "trial_number_2",
                        "distance",
                        "rip_1_dur_ms",
                        "rip_2_dur_ms",
                        "firing_pattern_1",
                        "firing_pattern_2",                        
                    ]
                ]
                _df = add_match_correct(_df)
                indi_non_nan = ~_df.distance.isna()
                _df = _df[indi_non_nan].reset_index()
                # _df = _df.astype(float)
                
                _df_in = _df[_df.match == 1]
                _df_out = _df[_df.match == 2]


                # indi_non_nan_in = ~_df_in.distance.isna()
                # indi_non_nan_out = ~_df_out.distance.isna()

                ## 1
                fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
                
                sns.histplot(
                    data=_df[_df.correct==True],
                    x="rip_1_dur_ms",
                    ax=axes[0]
                    )
                sns.histplot(
                    data=_df[_df.correct==False],
                    x="rip_1_dur_ms",
                    ax=axes[1]
                    )
                sns.histplot(
                    data=_df[_df.correct==True],
                    x="rip_2_dur_ms",
                    ax=axes[2]
                    )
                sns.histplot(
                    data=_df[_df.correct==False],
                    x="rip_2_dur_ms",
                    ax=axes[3]
                    )                
                plt.show()

                ## 2
                _df["n_firings_1"] = _df.firing_pattern_1.apply(sum)
                _df["n_firings_2"] = _df.firing_pattern_1.apply(sum)                

                fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

                hue = "correct" # response_time, match
                sns.scatterplot(
                    data = _df,
                    x="rip_1_dur_ms",
                    y="n_firings_1",
                    hue=hue,
                    ax=axes[0]
                    )
                corr = np.corrcoef(_df.rip_1_dur_ms.astype(float),
                                   _df.n_firings_1.astype(float))[0,1]

                axes[0].set_title(f"Corr = {corr:.1f}")

                # 0.41

                sns.scatterplot(
                    data = _df,
                    x="rip_2_dur_ms",
                    y="n_firings_2",
                    hue=hue,
                    ax=axes[1]
                    )

                corr = np.corrcoef(_df.rip_2_dur_ms.astype(float),
                                _df.n_firings_2.astype(float))[0,1]
                
                axes[1].set_title(f"Corr = {corr:.1f}")
                # 0.15

                mngs.io.save(fig, "./tmp/figs/duration_and_n_firings.png")
                plt.show()
                
                ## 2
                fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

                # sns.regplot(x=np.array(_df_in.distance[indi_non_nan_in]).astype(float),
                #             y=_df_in.response_time[indi_non_nan_in])
                sns.boxplot(
                    data=_df_in[indi_non_nan_in].astype(float).reset_index(),
                    x="correct",
                    y="distance",
                    ax=axes[0],
                )
                axes[0].set_title("Match IN")
                # incorrect 1; correct 119


                # sns.regplot(x=np.array(_df_out.distance[indi_non_nan_out]).astype(float),
                #             y=_df_out.response_time[indi_non_nan_out])
                sns.boxplot(
                    data=_df_out[indi_non_nan_out].astype(float).reset_index(),
                    x="correct",
                    y="distance",
                    ax=axes[1],
                )
                # incorrect 6; correct 120
                axes[1].set_title("Mismatch OUT")
                mngs.io.save(
                    fig,
                    "./tmp/figs/Encoding-Retrieval_ripple_similarity_and_correct.png",
                )

                # x=np.array(_df_in.distance[indi_non_nan]).astype(float),
                #         y=_df_in.response_time[indi_non_nan])

                # indi_non_nan = ~_df_out.distance.isna()
                # sns.regplot(x=np.array(_df_out.distance[indi_non_nan]).astype(float),
                #             y=_df_out.response_time[indi_non_nan])

                # np.corrcoef(np.array(_df_out.distance[indi_non_nan]).astype(float),
                #             _df_out.response_time[indi_non_nan])

                # plt.scatter(_df_in.distance, _df_in.response_time)
                # plt.xlabel("Distance")
                # plt.ylabel("Response time")
                # indi_non_nan = ~_df_in.distance.isna()
                # np.corrcoef(
                #     np.array(_df_in.distance[indi_non_nan]).astype(float),
                #     _df_in.response_time[indi_non_nan],
                # )

                # plt.scatter(_df_in.distance, _df_in.response_time)

                """
                fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
                axes[0].hist(_df_in.distance, density=True)
                axes[0].set_title("match: 1.0 (Match IN?)")
                
                axes[1].hist(_df_out.distance, density=True)
                axes[1].set_title("match: 2.0 (Mismatch OUT?)")

                # for ax in axes:
                #     ax.set_xlabel("Cosine distance between unit firings of ripple pairs")
                #     ax.set_ylabel("Ripple probability")

                fig.suptitle("Encoding - Retrieval")                
                fig.supylabel("Ripple probability")
                fig.supxlabel("Cosine distance between unit firings of ripple pairs")
                mngs.io.save(fig, "./tmp/figs/unit_firing_patterns/trial/X_Encoding-Retrieval_match_in_and_out.png")
                
                plt.show()
                
                """

                (~_df.distance.isna()).sum()  # 54
                (_df.distance > 0.9).sum()  # 13
                df_new = _df[(_df.distance > 0.9)]
                #   subject session trial_number_1 trial_number_2 distance match
                # 0     2.0     2.0           42.0           42.0      1.0
                # 0     3.0     1.0            2.0            2.0      1.0
                # 0     3.0     1.0            2.0            2.0      1.0
                # 0     3.0     1.0            4.0            4.0      1.0
                # 0     3.0     1.0            4.0            4.0      1.0
                # 0     3.0     2.0            2.0            2.0      1.0
                # 0     4.0     1.0           17.0           17.0      1.0
                # 0     4.0     1.0           23.0           23.0      1.0
                # 0     4.0     1.0           28.0           28.0      1.0
                # 0     4.0     1.0           36.0           36.0      1.0
                # 0     4.0     1.0           37.0           37.0      1.0
                # 0     4.0     2.0           35.0           35.0      1.0
                # 0     6.0     1.0           41.0           41.0      1.0

                df_new = add_match(df_new)

                import ipdb

                ipdb.set_trace()

                print((df.distance >= 0.9).sum())  # 436 / 4667

                e_ripples = df[(df.distance == 1)][
                    ["subject", "session", "trial_number_1", "trial_number_2"]
                ]
                e_ripples[(e_ripples.trial_number_1 == e_ripples.trial_number_2)]

            # mngs.io.save(
            #     fig,
            #     f"./tmp/figs/unit_firing_patterns/dist/{with_group_str}/"
            #     f"{i_phase_combi}_{phase_a}-{phase_b}.png",
            # )
            # plt.close()
            # plt.show()
