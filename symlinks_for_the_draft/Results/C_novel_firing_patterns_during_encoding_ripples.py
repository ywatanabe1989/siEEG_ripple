#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-05 23:40:45 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from itertools import combinations


def surrogate(_dist_df, N=10):
    fig, ax = plt.subplots()
    # _dist_df["surrogated_distances"] = np.nan
    s_dists_all = []

    for ii in range(len(_dist_df)):
        s_dists = []
        for _ in range(N):
            s_dist = scipy.spatial.distance.cosine(
                np.random.permutation(_dist_df.iloc[ii].firing_pattern_1),
                np.random.permutation(_dist_df.iloc[ii].firing_pattern_2),
            )
            s_dists.append(s_dist)
        # _dist_df.iloc[ii]["surrogated_distances"] = s_dists
        s_dists_all.append(s_dists)

    _dist_df["surrogated_distances"] = s_dists_all

    ax.hist(np.hstack(s_dists_all))


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


def test_bm(dfs, alternative="two_sided"):
    # for  alternative in ["two-sided", "less", "greater"]:
    for combi in combinations(
        ["Fixation - Encoding", "Encoding - Maintenance", "Fixation - Maintenance"], 2
    ):
        print(combi, alternative)
        print(
            scipy.stats.brunnermunzel(
                dfs[combi[0]],
                dfs[combi[1]],
                alternative=alternative,
            )
        )
        print()

def test_mwu(dfs, alternative="two_sided"):
    # for  alternative in ["two-sided", "less", "greater"]:
    for combi in combinations(
        ["Fixation - Encoding", "Encoding - Maintenance", "Fixation - Maintenance"], 2
    ):
        print(combi, alternative)
        print(
            scipy.stats.mannwhitneyu(
                dfs[combi[0]],
                dfs[combi[1]],
                alternative=alternative,
            )
        )
        print()
        
def test_ks(dfs, alternative="two-sided"):
    for combi in combinations(
        ["Fixation - Encoding", "Encoding - Maintenance", "Fixation - Maintenance"], 2
    ):
        print(combi, alternative)
        print(
            scipy.stats.ks_2samp(
                dfs[combi[0]],
                dfs[combi[1]],
                alternative=alternative,
            )
        )
        print()
    


def test_chi2(dfs, thres=0.9):
    for alternative in ["two-sided"]:  # , "less", "greater"]:
        for combi in combinations(
            ["Fixation - Encoding", "Encoding - Maintenance", "Fixation - Maintenance"],
            2,
        ):
            print(combi, alternative)

            data1 = dfs[combi[0]]
            data2 = dfs[combi[1]]

            is_dissimilar_1 = (data1 >= thres).astype(int)
            is_dissimilar_2 = (data2 >= thres).astype(int)

            # print(len(data1), len(data2))
            # print(data1.sum()/len(data1), data2.sum()/len(data2))

            df = pd.DataFrame(
                [
                    [(is_dissimilar_1 == 0).sum(), (is_dissimilar_1 == 1).sum()],
                    [(is_dissimilar_2 == 0).sum(), (is_dissimilar_2 == 1).sum()],
                ],
                columns=["similar", "dissimilar"],
                index=[combi[0], combi[1]],
            )
            print(df)
            print(df["dissimilar"] / (df["similar"] + df["dissimilar"]))

            chi2, p, dof, exp = scipy.stats.chi2_contingency(df)  # , correction=False
            print(p)
            print()

            # obs = np.array([
            #     np.unique(data1, return_counts=True)[1],
            #     np.unique(data2, return_counts=True)[1]
            #     ])

            # out = scipy.stats.chisquare(obs)
            # print(out)
            # # chi2, p, dof, expected = scipy.stats.chi2_contingency(obs)
            # # print(p)
            # print()

def bootstrap(data):
    from scipy.stats import bootstrap
    data = (data,) # _dist_df.distance
    # res = bootstrap(data, np.median, confidence_level=0.95)
    res = bootstrap(data, np.sum, confidence_level=0.95)    
    # sample_med = np.median(_dist_df.distance)
    sample_statistic = np.sum(data[0])

    print(res)
    print(sample_statistic)

    print((res.confidence_interval.low / len(data[0])).round(3))
    print((res.confidence_interval.high / len(data[0])).round(3))    


def calc_rate(data, axis=-1):
    return np.sum((data[axis]>0.9)) / len(data[axis])

def describe(data):
    described = data.describe()
    print(described["50%"].round(3), (described["75%"].round(3) - described["25%"]).round(3))
    print(described["count"])

            

# Loads
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]
dist_df = mngs.io.load("./tmp/dist_df.pkl")

# IoU_RIPPLE_THRES = 0.5
dist_df = dist_df[(dist_df.IoU_1 <= IoU_RIPPLE_THRES)
                  * (dist_df.IoU_2 <= IoU_RIPPLE_THRES)
                  * (dist_df.session <= SESSION_THRES)]

indi_within_groups_dict = mk_indi_within_groups(dist_df)

centers = np.arange(10) / 10 + 0.05
dfs = {}
dfs["centers"] = centers
dists = []
similarities = []
for phase_a, phase_b in (
    ["Fixation", "Encoding"],
    ["Fixation", "Maintenance"],
    ["Encoding", "Maintenance"],        
):

    indi_within = indi_within_groups_dict["trial"]  # 2716
    indi_phase_combi = get_crossed_phase_indi(dist_df, phase_a, phase_b)  # 10378
    _dist_df = (
        dist_df[indi_within * indi_phase_combi].copy().reset_index()
    )  # 127, 490, 207
    """
    test = _dist_df[_dist_df.distance.isna()]
    test.firing_pattern_1
    import sklearn
    sklearn.metrics.pairwise.cosine_similarity(
    X = np.array(test.iloc[0].firing_pattern_1).reshape(-1,1).T,
    Y=np.array(test.iloc[0].firing_pattern_2).reshape(-1,1).T,
    )
    from numpy.linalg import norm
    norm(X)
    norm(Y)
    np.dot(X.squeeze(),Y.squeeze())
    """
    _dist_df = _dist_df[~_dist_df.distance.isna()]
    _dist_df["distance"] = _dist_df["distance"].astype(float)
    print((~_dist_df.distance.isna()).sum())

    dists.append(_dist_df.distance)
    similarities.append(1-_dist_df.distance)    

    # bootstrap(np.array(_dist_df.distance>0.75))

    dfs.update({f"{phase_a} - {phase_b}": _dist_df.distance})

    # surrogate(_dist_df)


# fig, ax = plt.subplots()
# ax.boxplot(dists[0], positions=[1])
# ax.boxplot(dists[1], positions=[2])
# ax.boxplot(dists[2], positions=[3])
# plt.show()






describe(similarities[0])
describe(similarities[1])
describe(similarities[2])

# (dists[0] > 0.9).mean()
# (dists[1] > 0.9).mean()
# (dists[2] > 0.9).mean()

test_bm(dfs, alternative="greater")
# test_mwu(dfs, alternative="two-sided")
test_chi2(dfs, thres=0.75)
# test_ks(dfs, alternative="two-sided")

# bootstrap(_dist_df.distance)

# del dfs["centers"]
# mngs.general.force_dataframe(dfs)
