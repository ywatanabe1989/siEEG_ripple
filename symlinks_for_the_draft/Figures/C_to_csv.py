#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-05 21:26:37 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


dist_df = mngs.io.load("./tmp/dist_df.pkl")
indi_within_groups_dict = mk_indi_within_groups(dist_df)

centers = np.arange(10) / 10 + 0.05
dfs_box = {}
dfs = {}
dfs["centers"] = centers
for phase_a, phase_b in (["Fixation", "Encoding"],
                         ["Encoding", "Maintenance"],
                         ["Fixation", "Maintenance"]
                         ):

    indi_within = indi_within_groups_dict["trial"]
    indi_phase_combi = get_crossed_phase_indi(dist_df, phase_a, phase_b)
    _dist_df = dist_df[indi_within * indi_phase_combi].copy().reset_index()
    _dist_df = _dist_df[~_dist_df.distance.isna()]
    _dist_df["distance"] = _dist_df["distance"].astype(float)

    n, bins, patches = plt.hist(_dist_df.distance, bins=10)
    n /= n.sum()
    cumsum = np.cumsum(n)
    dfs_box.update({f"{phase_a} - {phase_b}": 1 - _dist_df.distance})
    # dfs.update({f"{phase_a} - {phase_b}": cumsum})
    dfs.update({f"{phase_a} - {phase_b}": n})        
    
    # dfs.update({f"{phase_a} - {phase_b}": _dist_df.distance})    

out = pd.DataFrame(dfs)
mngs.io.save(out, "./tmp/figs/C/x_distance_y_ripple_probability.csv")

out = mngs.general.force_dataframe(dfs_box)
mngs.io.save(out, "./tmp/figs/C/x_phases_y_similarity.csv")
                   
# _dist_df.distance = _dist_df.distance.replace({1:0.999})


# def count_bins(n_bins):
    


# import matplotlib.pyplot as plt
# import seaborn as sns
# data = out.melt()
# data = data[~(data=="").sum(axis=1).astype(bool)]
# sns.histplot(
#     data=data,
#     x="value",    
#     hue="variable",
#     # cumulative=True,
#     stat="probability",
#     cumulative=True,
#     common_norm=False,
#     # kde=True,
# )
# plt.show()
