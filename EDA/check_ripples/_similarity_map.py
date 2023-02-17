#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-24 12:23:43 (ywatanabe)"

import sys
sys.path.append(".")
from siEEG_ripple import utils
from utils.sim import get_crossed_phase_indi
from itertools import combinations
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import mngs
import numpy as np
import seaborn as sns

def mk_sim_map(sim_df):
    sim_map_df = pd.DataFrame()
    for phase_a, phase_b in utils.perm(2, PHASES):    
        indi_phase = get_crossed_phase_indi(sim_df, phase_a, phase_b)
        indi_within_trial = sim_df.trial_number_1 == sim_df.trial_number_2

        sim_df_tmp = sim_df[indi_phase * indi_within_trial]

        tmp_df = pd.DataFrame(pd.Series({
            "phase_a": phase_a,
            "phase_b": phase_b,        
            "median": sim_df_tmp.similarity.median().round(3),
            }))

        sim_map_df = pd.concat([sim_map_df, tmp_df], axis=1)
    return sim_map_df

def plot_hm(sim_map_df, suffix=None):
    # phases_a = np.array(sim_map_df.T["phase_a"]).reshape(4,4)
    # phases_b = np.array(sim_map_df.T["phase_b"]).reshape(4,4)
    medians = np.array(sim_map_df.T["median"], dtype=float).reshape(4,4).round(3)
    fig, ax = plt.subplots()
    lim_val = 0.5
    sns.heatmap(medians,
                xticklabels=PHASES,
                yticklabels=PHASES,
                ax=ax,
                vmin=-lim_val,
                vmax=lim_val,                
                )
    ax.set_title(suffix)
    # plt.show()
    return fig

def calc_sim_map(suffix, match):
    sim_df = utils.load_sim(suffix)
    sim_df = utils.sim.add_match_correct_response_time(sim_df)

    sim_map_df = mk_sim_map(sim_df)
    sim_map_match_df = mk_sim_map(sim_df[sim_df.match == match])
    return sim_map_match_df

# Loads
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
sm_cons_1 = calc_sim_map("cons", match=1)
sm_cons_2 = calc_sim_map("cons", match=2)
sm_rips_1 = calc_sim_map("rips", match=1)
sm_rips_2 = calc_sim_map("rips", match=2)

_sm_1 = sm_rips_1.T["median"] - sm_cons_1.T["median"]
_sm_2 = sm_rips_2.T["median"] - sm_cons_2.T["median"]

sm_1 = sm_cons_1.copy().T
sm_1["median"] = np.array(_sm_1).astype(float)

sm_2 = sm_cons_2.copy().T
sm_2["median"] = np.array(_sm_2).astype(float)

fig_1 = plot_hm(sm_1.T, suffix="match 1")
fig_2 = plot_hm(sm_2.T, suffix="match 2")
plt.show()

# mngs.io.save(sim_map_df.T, f"./tmp/figs/C/similarity_map_{suffix}.csv")
