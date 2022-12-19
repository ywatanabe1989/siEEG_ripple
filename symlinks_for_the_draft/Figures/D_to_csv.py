#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-09 16:14:12 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(".")
from eeg_ieeg_ripple_clf import utils

# Loads
def load_sim_df():
    dist_df = utils.load_dist()
    indi_within = utils.dist.mk_indi_within_groups(dist_df)["trial"]
    phase_a, phase_b = "Encoding", "Retrieval"
    indi_phase_combi = utils.dist.get_crossed_phase_indi(dist_df, phase_a, phase_b)
    _dist_df = dist_df[indi_within * indi_phase_combi].copy().reset_index()
    _dist_df = _dist_df[~_dist_df.distance.isna()]
    _dist_df["distance"] = _dist_df["distance"].astype(float)
    _dist_df = utils.dist.add_match_correct_response_time(_dist_df)
    sim_df = _dist_df.copy()
    sim_df["similarity"] = 1 - sim_df["distance"]
    return sim_df



def plot_violin(sim_df):
    # violin plot
    fig, ax = plt.subplots(figsize=(1.3, 0.91))
    sim_df["hue"] = True
    sns.violinplot(
        data=sim_df,
        x="match",
        y="similarity",
        hue="hue",
        hue_order=[False, True],
        split=True,
        ax=ax,
        color=mngs.plt.colors.to_RGBA("blue"),
        width=0.4,
        )
    ax.legend_ = None
    ax = mngs.plt.ax_set_size(ax, 1.3, 0.91)
    # plt.show()
    return fig

def test_bm(sim_df_match_1, sim_df_match_2):
    import scipy
    print(
    scipy.stats.brunnermunzel(
        sim_df_match_1.distance,
        sim_df_match_2.distance,        
        alternative="less",
        )
        )


sim_df = load_sim_df()
sim_df_match_1 = sim_df[sim_df.match == 1]
sim_df_match_2 = sim_df[sim_df.match == 2]

fig = plot_violin(sim_df)
mngs.io.save(fig, "./tmp/figs/violin/Encoding-Retrieval_similarity_by_match.tif")

test_bm(sim_df_match_1, sim_df_match_2)

EM_similarity_out = mngs.general.force_dataframe({
    "match 1": sim_df_match_1.similarity,
    "match 2": sim_df_match_2.similarity,
})
mngs.io.save(EM_similarity_out, "./tmp/figs/D/x_match_y_similarity_Encoding-Retrieval.csv")

# # hist
# centers = np.arange(10) / 10 + 0.05
# dfs = {}
# dfs["centers"] = centers
# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(_dist_df_in.distance, bins=10)
# n /= n.sum()
# dfs["Encoding_Retrieval_Match_IN"] = n
# n, bins, patches = plt.hist(_dist_df_out.distance, bins=10)
# n /= n.sum()
# dfs["Encoding_Retrieval_Mismatch_OUT"] = n
# dfs = pd.DataFrame(dfs)

# mngs.io.save(dfs, "./tmp/figs/D/x_distance_y_ripple_probability_by_match_Encoding-Retrieval.csv")
