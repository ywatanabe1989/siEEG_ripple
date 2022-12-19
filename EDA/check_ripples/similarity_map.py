#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-10 02:29:09 (ywatanabe)"

import sys
sys.path.append(".")
from eeg_ieeg_ripple_clf import utils
from utils.dist import get_crossed_phase_indi
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import mngs
import numpy as np
import seaborn as sns

def mk_sim_map(dist_df):
    sim_map_df = pd.DataFrame()
    for phase_a, phase_b in utils.perm(2, PHASES):    
        indi_phase = get_crossed_phase_indi(dist_df, phase_a, phase_b)
        indi_within_trial = dist_df.trial_number_1 == dist_df.trial_number_2

        dist_df_tmp = dist_df[indi_phase * indi_within_trial]

        tmp_df = pd.DataFrame(pd.Series({
            "phase_a": phase_a,
            "phase_b": phase_b,        
            "median": 1-dist_df_tmp.distance.median().round(3),
            }))

        sim_map_df = pd.concat([sim_map_df, tmp_df], axis=1)
    return sim_map_df

def plot_hm(sim_map_df):
    # phases_a = np.array(sim_map_df.T["phase_a"]).reshape(4,4)
    # phases_b = np.array(sim_map_df.T["phase_b"]).reshape(4,4)
    medians = np.array(sim_map_df.T["median"], dtype=float).reshape(4,4).round(3)
    fig, ax = plt.subplots()
    sns.heatmap(medians,
                xticklabels=PHASES,
                yticklabels=PHASES,
                ax=ax,
                vmin=-.3,
                vmax=.3,                
                )
    ax.set_title(suffix)
    # plt.show()
    return fig

def calc_sim_map(suffix, match):
    dist_df = utils.load_dist(suffix)
    dist_df = utils.dist.add_match_correct_response_time(dist_df)

    sim_map_df = mk_sim_map(dist_df)
    sim_map_match_df = mk_sim_map(dist_df[dist_df.match == match])
    return sim_map_match_df

# Loads
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
sm_cons_1 = calc_sim_map("cons", 1)
sm_cons_2 = calc_sim_map("cons", 2)
sm_rips_1 = calc_sim_map("rips", 1)
sm_rips_2 = calc_sim_map("rips", 2)

_sm_1 = sm_rips_1.T["median"] - sm_cons_1.T["median"]
_sm_2 = sm_rips_2.T["median"] - sm_cons_2.T["median"]

sm_1 = sm_cons_1.copy().T
sm_1["median"] = np.array(_sm_1).astype(float)

sm_2 = sm_cons_2.copy().T
sm_2["median"] = np.array(_sm_2).astype(float)

fig_1 = plot_hm(sm_1.T)
fig_2 = plot_hm(sm_2.T)
plt.show()

mngs.io.save(sim_map_df.T, f"./tmp/figs/C/similarity_map_{suffix}.csv")
