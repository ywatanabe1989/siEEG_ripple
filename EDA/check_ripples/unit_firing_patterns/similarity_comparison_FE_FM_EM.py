#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-12 13:17:00 (ywatanabe)"

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt    
import mngs
import scipy
from itertools import combinations
import statsmodels
    
sys.path.append(".")
from eeg_ieeg_ripple_clf import utils

"""
phase_a, phase_b = "Fixation", "Encoding"
"""
def load_sims_by_match(phase_a, phase_b, split_by_match=False):
    sim_rips = utils.load_sim("rips")
    sim_rips = sim_rips[utils.sim.mk_indi_within_groups(sim_rips)["trial"]]
    sim_rips = sim_rips[
        utils.sim.get_crossed_phase_indi(sim_rips, phase_a, phase_b)
    ]
    sim_rips = utils.sim.add_match_correct_response_time(sim_rips)
    sim_rips["phase_combi"] = f"r_{phase_a[0]}{phase_b[0]}"
    # sim_rips["similarity"] = 1 - sim_rips["simance"].astype(float)
    sim_rips = sim_rips[~sim_rips.similarity.isna()]
    sim_rips_in = sim_rips[sim_rips.match == 1]
    sim_rips_out = sim_rips[sim_rips.match == 2]

    sim_cons = utils.load_sim("cons")
    sim_cons = sim_cons[utils.sim.mk_indi_within_groups(sim_cons)["trial"]]
    sim_cons = utils.sim.add_match_correct_response_time(sim_cons)
    sim_cons = sim_cons[
        utils.sim.get_crossed_phase_indi(sim_cons, phase_a, phase_b)
    ]
    sim_cons["phase_combi"] = f"c_{phase_a[0]}{phase_b[0]}"
    # sim_cons["similarity"] = 1 - sim_cons["simance"].astype(float)
    sim_cons = sim_cons[~sim_cons.similarity.isna()]    
    sim_cons_in = sim_cons[sim_cons.match == 1]
    sim_cons_out = sim_cons[sim_cons.match == 2]

    if split_by_match:
        return sim_rips_in, sim_rips_out, sim_cons_in, sim_cons_out
    else:
        sim_rips.match = "IN & OUT"
        sim_cons.match = "IN & OUT"        
        return sim_rips, sim_cons

def bm_effsize(x, y):
    less = []
    for _x in x:
        less.append(_x < y)
    effsize = np.array(less).mean()
    return effsize
    

if __name__ == "__main__":
    FE_rips, FE_cons = load_sims_by_match(
        "Fixation", "Encoding"
    )
    
    FM_rips, FM_cons = load_sims_by_match(
        "Fixation", "Maintenance"
    )
    
    EM_rips, EM_cons = load_sims_by_match(
        "Encoding", "Maintenance"
    )
    
    ER_rips_in, ER_rips_out, ER_cons_in, ER_cons_out = load_sims_by_match(
        "Encoding", "Retrieval", split_by_match=True
    )
    ER_rips, ER_cons = load_sims_by_match(
        "Encoding", "Retrieval"
    )

    sims = [
            FE_rips,
            FE_cons,
            FM_rips,
            FM_cons,
            EM_rips,
            EM_cons,
            ER_rips,
            ER_cons,
            # ER_rips_in,
            # ER_rips_out,
            # ER_cons_in,
            # ER_cons_out,
        ]
    
    df = pd.concat(sims)

    df.similarity = df.similarity.astype(float)
    fig, ax = plt.subplots(figsize=(6.4*2, 4.8*2))
    sns.violinplot(
        data=df,
        x="phase_combi",
        # x="match",
        y="similarity",
        hue="match",
        # hue="phase_combi",
        ax=ax,
        )
    # mngs.io.save(fig, "./tmp/figs/violin/similarity_of_phase_ER.png")    
    # mngs.io.save(fig, "./tmp/figs/violin/similarity_of_phase_combis.png")
    
    plt.show()

    # df["n"] = 1
    # df.pivot_table(columns=["phase_combi", "match"], aggfunc=sum).T.reset_index().set_index("phase_combi")\
    #     .loc[["r_FE", "c_FE", "r_FM", "c_FM", "r_EM", "c_EM", "r_ER", "c_ER"]]


    pvals = []
    comparisons = []
    for sim1, sim2 in combinations(sims, 2):
        _, pval = scipy.stats.brunnermunzel(
            sim1.similarity,
            sim2.similarity,
            alternative="two-sided",
            )
        effsize = bm_effsize(sim1.similarity, sim2.similarity)
        pvals.append(pval)
        comparisons.append((sim1.phase_combi.iloc[0], sim2.phase_combi.iloc[0]))

    out = statsmodels.stats.multitest.multipletests(pvals)
    np.array(comparisons)[out[0]]
