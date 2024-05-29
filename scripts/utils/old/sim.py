#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-10 15:33:33 (ywatanabe)"

import numpy as np
import mngs

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

def add_match_correct_response_time(df):
    df = df[df.trial_number_1 == df.trial_number_2]
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

def mk_indi_within_groups(sim_df):
    indi_session = np.ones(len(sim_df), dtype=bool)
    indi_letter = sim_df.probe_letter_1 == sim_df.probe_letter_2
    indi_trial = sim_df.trial_number_1 == sim_df.trial_number_2
    indi_within_groups_dict = dict(
        session=indi_session, letter=indi_letter, trial=indi_trial
    )
    return indi_within_groups_dict

