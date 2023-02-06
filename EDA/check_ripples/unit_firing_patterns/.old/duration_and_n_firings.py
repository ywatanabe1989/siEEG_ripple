#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-27 16:01:09 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
import warnings
import re
import random
import scipy
import mngs
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from siEEG_ripple import utils



if __name__ == "__main__":
    from itertools import combinations
    import seaborn as sns
    
    rips_df = utils.load_rips()

    phases = mngs.io.load("./config/global.yaml")["PHASES"]

    corrs = []
    for phase in phases:
        indi = rips_df.phase == phase
        corrs.append(np.corrcoef(rips_df.loc[indi, "duration_ms"],
                                 rips_df.loc[indi, "n_firings"])[0,1].round(2))
        
    fig, ax = plt.subplots(sharex=True, sharey=True)
    
    sns.scatterplot(
        data=rips_df,
        x="duration_ms",
        y="n_firings",
        hue="phase",
        ax=ax,
        )
    for phase in phases:
        sns.regplot(
            data=rips_df[rips_df.phase == phase],
            x="duration_ms",
            y="n_firings",
            ax=ax,
            )
    ax.set_title(f"Correlations: {corrs}")
    
    mngs.io.save(fig, "./tmp/figs/scatter/duration_and_n_firings.png")
    plt.show()



    sns.lmplot(
        data=rips_df,
        x="duration_ms",
        y="n_firings",
        col="phase",
        col_order=phases,
        # hue="phase",
        # ax=ax,
        )
    

    # np.unique(rips_df.subject, return_counts=True)
    # (array([2., 3., 4., 6., 7.]), array([ 985,  583,  328, 1004,  616]))

    
    
    # columns = ["i_rip_1", "i_rip_2", "n_common", "distelation", "pval"]
    # dist_df = pd.DataFrame(columns=columns)

    # for sub in rips_df.subject.unique():
    #     rips_df_sub = rips_df[rips_df.subject == sub]

    #     for session in tqdm(rips_df_sub.session.unique()):
    #         trials_info = mngs.io.load(
    #             f"./data/Sub_{int(sub):02d}/Session_{int(session):02d}/trials_info.csv"
    #         )

    #         rips_df_session = rips_df_sub[rips_df_sub.session == session]

    #         for i_rip_1, i_rip_2 in combinations(np.arange(len(rips_df_session)), 2):

    #             columns = ["i_rip1", "i_rip2", "n_common", "distance", "pval"]

    #             rip_1 = rips_df_session.iloc[i_rip_1]
    #             rip_2 = rips_df_session.iloc[i_rip_2]

    #             pattern_1 = determine_firing_patterns(rip_1)
    #             pattern_2 = determine_firing_patterns(rip_2)

    #             dist = calc_dist(pattern_1, pattern_2)

    #             probe_letter_1 = trials_info.iloc[
    #                 int(rip_1.trial_number - 1)
    #             ].probe_letter

    #             probe_letter_2 = trials_info.iloc[
    #                 int(rip_2.trial_number - 1)
    #             ].probe_letter

    #             dist_df_tmp = pd.DataFrame(
    #                 pd.Series(
    #                     dict(
    #                         subject=sub,
    #                         session=session,
    #                         i_rip_1=i_rip_1,
    #                         i_rip_2=i_rip_2,
    #                         phase_1=rip_1.phase,
    #                         phase_2=rip_2.phase,
    #                         probe_letter_1=probe_letter_1,
    #                         probe_letter_2=probe_letter_2,
    #                         trial_number_1=rip_1.trial_number,
    #                         trial_number_2=rip_2.trial_number,
    #                         distance=dist,
    #                         firing_pattern_1=pattern_1,
    #                         firing_pattern_2=pattern_2,
    #                         rip_1_dur_ms=(rip_1.end_time - rip_1.start_time) * 1000,
    #                         rip_2_dur_ms=(rip_2.end_time - rip_2.start_time) * 1000,
    #                     )
    #                 )
    #             ).T

    #             dist_df = pd.concat([dist_df, dist_df_tmp])

    #         mngs.io.save(dist_df, "./tmp/dist_df.pkl")
