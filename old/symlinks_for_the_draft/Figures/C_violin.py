#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-09 16:17:25 (ywatanabe)"

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

def extract_sim_df_of_phase_a_and_b(sim_df, phase_a, phase_b):
    indi_phase = utils.dist.get_crossed_phase_indi(dist_df, phase_a, phase_b)
    indi_within_trial = utils.dist.mk_indi_within_groups(sim_df)["trial"]
    extracted = sim_df[indi_phase * indi_within_trial]
    extracted = extracted[~extracted.similarity.isna()]
    extracted["phase_combination"] = f"{phase_a} - {phase_b}"
    extracted.similarity = extracted.similarity.astype(float)
    return extracted

# Loads
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
dist_df = utils.load_dist()
sim_df = dist_df.copy()
sim_df["similarity"] = 1 - sim_df.distance

# Extracts part of sim_df
FE_sim_df = extract_sim_df_of_phase_a_and_b(sim_df, "Fixation", "Encoding") # 54
FM_sim_df = extract_sim_df_of_phase_a_and_b(sim_df, "Fixation", "Maintenance") # 115
EM_sim_df = extract_sim_df_of_phase_a_and_b(sim_df, "Encoding", "Maintenance") # 243
FE_FM_EM_sim_df = pd.concat([FE_sim_df, FM_sim_df, EM_sim_df])


def plot_violin(FE_FM_EM_sim_df):
    # violin plot
    fig, ax = plt.subplots(figsize=(1.3, 0.91))
    FE_FM_EM_sim_df["hue"] = True
    sns.violinplot(
        data=FE_FM_EM_sim_df,
        x="phase_combination",
        y="similarity",
        order=["Fixation - Encoding", "Fixation - Maintenance", "Encoding - Maintenance"],
        hue="hue",
        hue_order=[False, True],
        split=True,
        ax=ax,
        color=mngs.plt.colors.to_RGBA("blue"),
        width=0.4,
    )
    
    # sns.violinplot(
    #     data=FE_FM_EM_sim_df,
    #     x="match",
    #     y="similarity",
    #     hue="hue",
    #     hue_order=[False, True],
    #     split=True,
    #     ax=ax,
    #     color=mngs.plt.colors.to_RGBA("blue"),
    #     width=0.4,
    #     )
    ax.legend_ = None
    ax = mngs.plt.ax_set_size(ax, 1.3, 0.91)
    # plt.show()
    return fig

fig = plot_violin(FE_FM_EM_sim_df)
mngs.io.save(fig, "./tmp/figs/violin/similarity_between_phases.tif")
# fig, ax = plt.subplots(2)

# sns.violinplot(
#     data=FE_FM_EM_sim_df,
#     x="phase_combination",
#     y="similarity",
#     order=["Fixation - Encoding", "Fixation - Maintenance", "Encoding - Maintenance"],
#     color=mngs.plt.colors.to_RGBA("blue"),
#     )
# plt.show()
