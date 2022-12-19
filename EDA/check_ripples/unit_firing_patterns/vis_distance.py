#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-10 00:15:01 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
import warnings
import re
import random
import scipy
import mngs
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import pingouin
from pprint import pprint
from itertools import combinations
    
def perm(n, seq):
    out = []
    for p in itertools.product(seq, repeat=n):
        out.append(p)
    return out


def calc_spearman_correlation(common_1, common_2):
    if common_1 is not np.nan:
        corr, pval = scipy.stats.spearmanr(common_1, common_2)
        return corr, pval
    else:
        return np.nan, np.nan


def add_match_correct_response_time(df):
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


# def surrogate(ns_units, ns_rep, n_factor=10, actual_corrs=None, actual_pvals=None):
#     def _in_surrogate(n_units, n_rep):
#         corrs, pvals = [], []

#         for _ in range(n_rep * n_factor):
#             _common_1 = np.random.permutation(np.arange(n_units))
#             _common_2 = np.random.permutation(np.arange(n_units))
#             corr, pval = calc_spearman_correlation(_common_1, _common_2)
#             corrs.append(corr)
#             pvals.append(pval)
#         corrs = np.array(corrs)
#         pvals = np.array(pvals)
#         return corrs, pvals

#     corrs, pvals = [], []
#     for n_units, n_rep in tqdm(zip(ns_units, ns_rep)):
#         _corrs, _pvals = _in_surrogate(n_units, n_rep)
#         corrs.append(_corrs)
#         pvals.append(_pvals)
#     corrs = np.hstack(corrs)
#     pvals = np.hstack(pvals)

#     fig, axes = plt.subplots(ncols=2)
#     axes[0].hist(corrs, bins=20, alpha=0.3, density=True, label="surrogate")
#     axes[1].hist(pvals, bins=20, alpha=0.3, density=True, label="surrogate")

#     if actual_corrs is not None:
#         axes[0].hist(actual_corrs, bins=20, alpha=0.3, density=True, label="actual")

#     if actual_pvals is not None:
#         axes[1].hist(actual_pvals, bins=20, alpha=0.3, density=True, label="actual")

#     axes[0].set_xlim(-1, 1)
#     axes[1].set_xlim(0, 1)

#     axes[0].set_xlabel("Rank-order correlation")
#     axes[1].set_xlabel("P-value")

#     fig.legend()
#     return fig


def get_crossed_phase_indi(df, phase_a, phase_b):
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


def plot_hist_distance_and_ripple_count(_dist_df, with_group_str, suptitle):
    fig, ax = plt.subplots()
    ax.hist(_dist_df.distance, density=True)
    ax.set_xlabel("Cosine distance between unit firings of ripple pairs")
    ax.set_ylabel("Ripple probability")

    median = np.nanmedian(np.array(_dist_df.distance).astype(float))
    ax.axvline(x=median, color="gray", linestyle="dotted")

    ax.set_ylim(0, 2.5)

    # title = f"{with_group_str}\n{phase_a}-{phase_b}\nn = {len(_dist_df)}"
    suptitle = f"{suptitle}\nn = {len(_dist_df)}"
    fig.suptitle(suptitle)

    return fig


def plot_hist_duration_by_correct_and_phase(_df, phase_a, phase_b, suptitle):
    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
    i_ax = 0
    for is_correct in [False, True]:
        for rip_1_or_2 in (1, 2):
            ax = axes[i_ax]
            correct_str = "Correct" if is_correct else "Incorrect"
            rip_dur_ms_str = f"rip_{rip_1_or_2}_dur_ms"
            phase_str = phase_a if rip_1_or_2 == 1 else phase_b
            sns.histplot(data=_df[_df.correct == is_correct], x=rip_dur_ms_str, ax=ax)
            ax.set_title(correct_str)
            ax.set_xlabel(f"Ripple duration of {phase_str} [ms]")
            i_ax += 1
    fig.suptitle(suptitle)
    return fig


def plot_scatter_duration_and_n_spikes(_df, suptitle, hue="correct"):
    _df["n_firings_1"] = _df.firing_pattern_1.apply(sum)
    _df["n_firings_2"] = _df.firing_pattern_1.apply(sum)

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    rips_indi = [1, 2]

    for ax, rip_1_or_2 in zip(axes, rips_indi):
        x = f"rip_{rip_1_or_2}_dur_ms"
        y = f"n_firings_{rip_1_or_2}"

        sns.scatterplot(data=_df, x=x, y=y, hue=hue, ax=ax)
        corr = np.corrcoef(_df[x].astype(float), _df[y].astype(float))[0, 1]

        ax.set_title(f"Corr = {corr:.2f}")

    fig.suptitle(suptitle)
    return fig




# def plot_box_correct_and_distance_by_match(_df_in, _df_out, suptitle):
def plot_box_x_match_y_distance(_dist_df, suptitle):
    fig,ax = plt.subplots()
    _dist_df["match"] =     _dist_df["match"].replace({1:"IN", 2: "OUT"})
    sns.boxplot(
        data=_dist_df,
        x="match",
        order=["IN", "OUT"],
        y="distance"
        )
    ax.set_xlabel("Match")
    ax.set_ylabel("Ripple distance")

    ax.set_title(suptitle)
    # ax.set_ylim(0,1)

    return fig
    

    # fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    # _df_in = _df_in[["distance", "correct"]]
    # _df_out = _df_out[["distance", "correct"]]

    # for ax, _df_in_or_out, match_str in zip(
    #     axes, [_df_in, _df_out], ["Match IN", "Mismatch OUT"]
    # ):

    #     n_in_or_out_i = (_df_in_or_out.correct == False).sum()
    #     n_in_or_out_c = (_df_in_or_out.correct == True).sum()

    #     sns.boxplot(
    #         data=_df_in_or_out.astype(float).reset_index(),  # [indi_non_nan_in]
    #         x="correct",
    #         y="distance",
    #         ax=ax,
    #         order=[0, 1],
    #     )
    #     ax.set_xlabel("")
    #     ax.set_ylabel("")

    #     ax.set_title(f"{match_str}\nn_i = {n_in_or_out_i}; n_c = {n_in_or_out_c}")

    # fig.supxlabel("Is correct")
    # fig.supylabel("Cosine distance between unit firings of ripple pairs")
    # fig.suptitle(suptitle)
    # return fig

def test_x_match_y_distance(_df_in, _df_out):
    _df_in = _df_in[["distance", "correct"]]
    _df_out = _df_out[["distance", "correct"]]

    _df_in_c = _df_in[_df_in.correct == True]
    _df_in_i = _df_in[_df_in.correct == False]
    _df_out_c = _df_out[_df_out.correct == True]
    _df_out_i = _df_out[_df_out.correct == False]

    _df_io_ic_d = {
        "Match IN - Correct": _df_in_c,
        "Match IN - Incorrect": _df_in_i,
        "Mismatch OUT - Correct": _df_out_c,
        "Mismatch OUT - Incorrect": _df_out_i,
        }

    for combi in combinations(list(_df_io_ic_d.keys()), 2):
        _df_io_ic_1 = _df_io_ic_d[combi[0]]
        _df_io_ic_2 = _df_io_ic_d[combi[1]]
        print(combi)
        try:
            print(scipy.stats.brunnermunzel(_df_io_ic_1.distance,
                                            _df_io_ic_2.distance,
                                            alternative="less",
                                            ))
            # 0.06

            # # import ipdb; ipdb.set_trace()
            # print(scipy.stats.mannwhitneyu(_df_io_ic_1.distance,
            #                                _df_io_ic_2.distance,
            #                                 alternative="less",
            #                                 ))
            # # 0.06
            
            # print(scipy.stats.ks_2samp(_df_io_ic_1.distance,
            #                                _df_io_ic_2.distance,
            #                                 alternative="two-sided",
            #                                 ))
            # # less: 0.06
            # # 0.07
            

            
        except Exception as e:
            print(e)


# def test_plot_box_correct_and_distance_by_match(_df_in, _df_out):
#     pass

            
    


def plot_scatter_distance_and_response_time_by_match(_df_in, _df_out, suptitle):
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    _df_in = _df_in[["distance", "response_time", "correct"]][
        _df_in.response_time <= 2.0
    ]
    _df_out = _df_out[["distance", "response_time", "correct"]][
        _df_out.response_time <= 2.0
    ]

    for ax, _df_in_or_out, match_str in zip(
        axes, [_df_in, _df_out], ["Match IN", "Mismatch OUT"]
    ):
        corrs = []
        for is_correct in [False, True]:
            correct_str = "Correct" if is_correct else "Incorrect"

            data = _df_in_or_out[_df_in_or_out.correct == is_correct]
            x = data["distance"]
            y = data["response_time"]
            sns.regplot(
                data=data,
                x=x,
                y=y,
                ax=ax,
                label=correct_str,
            )

            ax.legend(loc="upper right")

            corr = np.corrcoef(x, y)[0, 1]
            corrs.append(corr)

        # ax.set_title(f"{match_str}\nr = {corr:.2f}")
        ax.set_title(f"{match_str}\nr_i = {corrs[0]:.2f}; r_c = {corrs[1]:.2f}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supxlabel("Distance")
    fig.supylabel("Response time")
    fig.suptitle(suptitle)
    return fig

def test_distance_and_ripple_count(dist_df, dissimilar_thres=0.75):
    are_dissimilar = []
    for phase_a, phase_b in [
        ("Fixation", "Encoding"),
        ("Fixation", "Maintenance"),
        ("Encoding", "Maintenance"),
    ]:
        indi_phase_combi = get_crossed_phase_indi(dist_df, phase_a, phase_b)
        _dist_df = dist_df[indi_within * indi_phase_combi].copy().reset_index()
        _dist_df = _dist_df[~_dist_df.distance.isna()]
        _dist_df["is_dissimilar"] = _dist_df.distance >= dissimilar_thres
        are_dissimilar.append(_dist_df.is_dissimilar)

    are_dissimilar_df = pd.DataFrame(are_dissimilar).T
    are_dissimilar_df.columns = ["F-E", "F-M", "E-M"]
    melted_df = are_dissimilar_df.melt()
    melted_df["value"] = melted_df["value"].astype(float)

    # chi2 ANOVA
    pingouin.chi2_independence(data=melted_df, x="value", y="variable")

    # pair-wise
    for combi in combinations(["F-E", "F-M", "E-M"], 2):
        indi = (melted_df.variable == combi[0]) + (melted_df.variable == combi[1])
        pprint(
            pingouin.chi2_independence(data=melted_df[indi], x="value", y="variable")
        )
        # F-E F-M
        # test    lambda      chi2  dof      pval    cramer     power
        # 0             pearson  1.000000  4.168673  1.0  0.041178  0.077727  0.532616
        # E-M F-E
        # 0             pearson  1.000000   0.0  1.0   1.0     0.0   0.05
        # E-M F-M
        # 0             pearson  1.000000  7.319583  1.0  0.006821  0.102996  0.772019

def plot_hist_x_distance_y_ripple_paircount_hue_match(_dist_df, suptitle):
    fig, ax = plt.subplots()
    ax.hist(
        _dist_df_in.distance, alpha=0.3, density=True, label="Match IN"
        )
    ax.hist(
        _dist_df_out.distance, alpha=0.3, density=True, label="Mismatch OUT"
        )
    ax.legend(loc="upper right")
    ax.set_xlabel("Ripple distance")
    ax.set_ylabel("Ripple pair probability")
    ax.set_title(suptitle)
    return fig
        
        

# def test_distance_and_ripple_count(dist_df):
#     histograms = []
#     for phase_a, phase_b in [
#         ("Fixation", "Encoding"),
#         ("Fixation", "Maintenance"),
#         ("Encoding", "Maintenance"),
#     ]:
#         import ipdb; ipdb.set_trace()
        
#         indi_phase_combi = get_crossed_phase_indi(dist_df, phase_a, phase_b)
#         _dist_df = dist_df[indi_within * indi_phase_combi].copy().reset_index()
#         _dist_df = _dist_df[~_dist_df.distance.isna()]
#         # _dist_df["is_dissimilar"] = _dist_df.distance >= 0.75  # 0.75
#         histograms.append(_dist_df.distance)  #  >= 0.9

#     histograms_df = pd.DataFrame(histograms).T
#     histograms_df.columns = ["F-E", "F-M", "E-M"]
#     melted_df = histograms_df.melt()
#     melted_df["value"] = melted_df["value"].astype(float)
#     melted_df = melted_df.rename(columns={"variable": "Phase combination",
#                                           "value": "Ripple distance"})    

#     sns.histplot(
#         data=melted_df,
#         x="Ripple distance",
#         hue="Phase combination",
#         cumulative=True,
#         common_norm=False,
#         stat="probability",
#         kde=True,
#         )

#     # # chi2 ANOVA
#     # pingouin.chi2_independence(data=melted_df, x="value", y="variable")

#     # pair-wise
#     combi_str_d = {0: "F-E", 1: "F-M", 2: "E-M"}
#     for combi in combinations([0, 1, 2], 2):
#         data1 = histograms[combi[0]]
#         data2 = histograms[combi[1]]        
#         print(combi_str_d[combi[0]], combi_str_d[combi[1]])
#         pprint(
#             # scipy.stats.ks_2samp(data1, data2, alternative="two-sided")
#             scipy.stats.ks_2samp(data1, data2, alternative="two-sided")            
#         )
#         # F-E F-M
#         # KstestResult(statistic=0.13454545454545455, pvalue=0.448405540345836)
#         # F-E E-M
#         # KstestResult(statistic=0.08727272727272728, pvalue=0.8573836519453342)
#         # F-M E-M
#         # KstestResult(statistic=0.13163636363636363, pvalue=0.09254018764749167)

def test_dissimilar_ripple_perc_by_match(_dist_df_in, _dist_df_out, distance_thres=0.6):
    df_in_far = pd.DataFrame(_dist_df_in.distance > distance_thres)
    df_in_far["match"] = "IN"
    df_out_far = pd.DataFrame(_dist_df_out.distance > distance_thres)
    df_out_far["match"] = "OUT"
    df = pd.concat([df_in_far, df_out_far])
    pprint(
        pingouin.chi2_independence(
            data=df,
            x="distance",
            y="match",
        ))


if __name__ == "__main__":
    import mngs
    import matplotlib.pyplot as plt
    from itertools import combinations
    from time import sleep

    dist_df = mngs.io.load("./tmp/dist_df.pkl")
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]

    indi_within_groups_dict = mk_indi_within_groups(dist_df)

    phase_combis = [("Any", "Any")] + perm(
        2, ["Fixation", "Encoding", "Maintenance", "Retrieval"]
    )

    # for with_group_str, indi_within in indi_within_groups_dict.items():
    with_group_str = "trial"
    indi_within = indi_within_groups_dict["trial"]
    
    test_distance_and_ripple_count(dist_df, dissimilar_thres=0.9)
        
    for i_phase_combi, (phase_a, phase_b) in enumerate(phase_combis):
        # print(f"\n{phase_a}, {phase_b}\n")
        pcs = f"{i_phase_combi:02d}_{phase_a}-{phase_b}"  # phase_combi_str
        print(f"\n{pcs}\n")

        indi_phase_combi = get_crossed_phase_indi(dist_df, phase_a, phase_b)
        _dist_df = dist_df[indi_within * indi_phase_combi].copy().reset_index()
        _dist_df = _dist_df[~_dist_df.distance.isna()]
        _dist_df["distance"] = _dist_df["distance"].astype(float)

        # if indi_within == "trial"
        assert (_dist_df.trial_number_1 == _dist_df.trial_number_2).all()
        _dist_df = add_match_correct_response_time(_dist_df)
        _dist_df_in = _dist_df[_dist_df.match == 1]
        _dist_df_out = _dist_df[_dist_df.match == 2]

        # Plots
        fig = plot_hist_distance_and_ripple_count(_dist_df, with_group_str, pcs)
        mngs.io.save(fig, f"./tmp/figs/hist/distance_and_ripple_count/{pcs}.png")
        plt.close()

        fig = plot_hist_duration_by_correct_and_phase(_dist_df, phase_a, phase_b, pcs)
        mngs.io.save(fig, f"./tmp/figs/hist/duration_by_correct_and_phase/{pcs}.png")
        plt.close()

        fig = plot_scatter_duration_and_n_spikes(_dist_df, pcs, hue="correct")
        mngs.io.save(fig, f"./tmp/figs/scatter/duration_and_n_spikes/{pcs}.png")
        plt.close()

        fig = plot_box_x_match_y_distance(_dist_df, pcs)
        mngs.io.save(fig, f"./tmp/figs/box/x_macth_y_distance/{pcs}.png")
        plt.close()
        
        fig = plot_hist_x_distance_y_ripple_paircount_hue_match(_dist_df, pcs)
        mngs.io.save(fig, f"./tmp/figs/hist/x_distance_y_ripple_pair_count_hue_match/{pcs}.png")
        plt.close()
        
        fig = plot_scatter_distance_and_response_time_by_match(
            _dist_df_in, _dist_df_out, pcs
        )
        mngs.io.save(
            fig, f"./tmp/figs/scatter/distance_and_response_time_by_match/{pcs}.png"
        )
        plt.close()


        if (phase_a == "Encoding") * (phase_b == "Retrieval"):
            import ipdb; ipdb.set_trace()
            test_x_match_y_distance(_dist_df_in, _dist_df_out) # brunnermunzel
            test_dissimilar_ripple_perc_by_match(_dist_df_in, _dist_df_out, distance_thres=0.6) # chi2
