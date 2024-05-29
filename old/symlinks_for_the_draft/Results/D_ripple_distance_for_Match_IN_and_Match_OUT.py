#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-05 23:49:52 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
import scipy

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

def test_chi2(_df_in, _df_out, thres = 0.8):
    is_dissimilar_1 = (_df_in.distance >= thres).astype(int)
    is_dissimilar_2 = (_df_out.distance >= thres).astype(int)
    df = pd.DataFrame(
        [
            [(is_dissimilar_1 == 0).sum(), (is_dissimilar_1 == 1).sum()],
            [(is_dissimilar_2 == 0).sum(), (is_dissimilar_2 == 1).sum()],
        ],
        columns=["similar", "dissimilar"],
        index=["Encoding", "Retrieval"],
    )
    chi2, p, dof, exp = scipy.stats.chi2_contingency(df)  # , correction=False    
    print(p)
    print()

def describe(data):
    # data = _dist_df_in["similarity"]
    described = data.describe()
    med, IQR = described["50%"], described["75%"] - described["25%"]
    print(len(data))
    print(med.round(3), IQR.round(3))
    print()
    
if __name__ == "__main__":

    ## Loads
    IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
    SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]
    
    dist_df = mngs.io.load("./tmp/dist_df.pkl")
    # IoU_RIPPLE_THRES = 0.5
    dist_df = dist_df[(dist_df.IoU_1 <= IoU_RIPPLE_THRES)
                      * (dist_df.IoU_2 <= IoU_RIPPLE_THRES)
                      * (dist_df.session <= SESSION_THRES)]
    indi_within_groups_dict = mk_indi_within_groups(dist_df)

    # dfs = {}
    # for phase_a, phase_b in (["Fixation", "Encoding"],
    #                          ["Encoding", "Maintenance"],
    #                          ["Fixation", "Maintenance"]
    #                          ):

    phase_a, phase_b = "Encoding", "Retrieval"
    indi_within = indi_within_groups_dict["trial"]
    indi_phase_combi = get_crossed_phase_indi(dist_df, phase_a, phase_b)
    _dist_df = dist_df[indi_within * indi_phase_combi].copy().reset_index()

    _dist_df = _dist_df[~_dist_df.distance.isna()]
    _dist_df["distance"] = _dist_df["distance"].astype(float)
    _dist_df = add_match_correct_response_time(_dist_df)

    _dist_df_in = _dist_df[_dist_df.match == 1]
    _dist_df_out = _dist_df[_dist_df.match == 2]

    _dist_df_in["similarity"] = 1 - _dist_df_in["distance"]
    _dist_df_out["similarity"] = 1 - _dist_df_out["distance"]

    # len(_dist_df_in["similarity"])
    # len(_dist_df_out["similarity"])

    describe(_dist_df_in["similarity"])
    describe(_dist_df_out["similarity"])
    
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    # axes[0].hist(_dist_df_in.distance, density=True)
    # axes[1].hist(_dist_df_out.distance, density=True)
    # plt.show()

    print(
    scipy.stats.brunnermunzel(
        _dist_df_in.distance,
        _dist_df_out.distance,
        alternative="less",
        )
        )

    test_chi2(_dist_df_in, _dist_df_out, thres = 0.75)
