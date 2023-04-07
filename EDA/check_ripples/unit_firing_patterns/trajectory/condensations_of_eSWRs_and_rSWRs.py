#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-15 18:41:24 (ywatanabe)"
import sys

sys.path.append(".")
import utils

import mngs
import numpy as np
from itertools import combinations
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Funcations
def calc_within_dists(coords):
    within_dists = []
    for i_combi, (a, b) in enumerate(combinations(coords, 2)):
        within_dists.append(mngs.linalg.nannorm(a - b))
    within_dists = np.array(within_dists)
    return within_dists[~np.isnan(within_dists)]


def calc_across_dists(coords_1, coords_2):
    across_dists = []
    for a in coords_1:
        for b in coords_2:
            across_dists.append(mngs.linalg.nannorm(a - b))
    across_dists = np.array(across_dists)
    return across_dists[~np.isnan(across_dists)]


# Fixes seeds
mngs.gen.fix_seeds(42, np=np)

# Parameters
(
    PHASES,
    PHASES_BINS_DICT,
    GS_BINS_DICT,
    COLORS_DICT,
    BIN_SIZE,
) = utils.define_phase_time()

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
rips_df = utils.rips.add_coordinates(
    utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
)
cons_df = utils.rips.add_coordinates(
    utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
)

# koko
start_bin, end_bin = 2, 10
fig, axes = plt.subplots(ncols=12, sharex=True, sharey=True)
for i_match, match in enumerate([1, 2]):
    match_str = ["IN", "OUT"][i_match]
    for i_event, events_df in enumerate([cons_df, rips_df]):
        event_str = ["SWR-", "SWR+"][i_event]
        for i_set_size, set_size in enumerate([4,6,8]):
            ax = axes[6*i_match+3*i_event+i_set_size]
            
            wd_E, wd_R, wd_ER, ad_ER = [], [], [], []
            for subject in ROIs.keys():
                subject = f"{subject:02d}"
                for session in ["01", "02"]:
                    events_df_s = events_df[
                        (events_df.subject == subject)
                        * (events_df.session == session)
                        * (events_df.match == match)
                        * (events_df.correct == True)
                        * (events_df.set_size == set_size)                        
                    ]

                    coords_E = pd.concat(
                        [
                            events_df_s[f"{ii}"][events_df_s.phase == "Encoding"]
                            for ii in range(start_bin, end_bin)
                        ]
                    )
                    coords_R = pd.concat(
                        [
                            events_df_s[f"{ii}"][events_df_s.phase == "Retrieval"]
                            for ii in range(start_bin, end_bin)
                        ]
                    )

                    wd_E.append(calc_within_dists(coords_E))
                    wd_R.append(calc_within_dists(coords_R))
                    wd_ER.append(calc_within_dists(pd.concat([coords_E, coords_R])))
                    ad_ER.append(calc_across_dists(coords_E, coords_R))
            wd_E, wd_R, wd_ER, ad_ER = (
                np.hstack(wd_E),
                np.hstack(wd_R),
                np.hstack(wd_ER),
                np.hstack(ad_ER),
            )

            ax.boxplot(
                [wd_E, wd_R, ad_ER],
                positions=[1, 2, 3],
                showfliers=False,
            )
            ax.set_title(f"{match_str} {event_str}")
plt.show()

print(mngs.gen.describe(wd_E, "median"))  # IN 1.72; OUT 1.36
print(mngs.gen.describe(wd_R, "median"))  # IN 2.03; OUT 1.33
print(mngs.gen.describe(wd_ER, "median"))  # IN 1.88; OUT 1.56
