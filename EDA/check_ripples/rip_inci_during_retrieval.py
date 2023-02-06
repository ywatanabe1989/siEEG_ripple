#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-05 19:40:53 (ywatanabe)"

import sys
sys.path.append(".")
import utils
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import mngs
import pandas as pd

def get_a_line(ctimes):
    t_sec = 8
    xx_s = np.linspace(0,t_sec,160)
    n_trials_per_session = 50
    incis = []
    width_s = xx_s[1] - xx_s[0]
    for ii in range(0, len(xx_s)):
        cc_s = xx_s[ii]
        pre = cc_s - width_s / 2
        post = cc_s + width_s / 2    
        # cur = x[ii]
        n = ((pre <= ctimes) * (ctimes < post)).sum()
        inci = n / width_s
        incis.append(inci)
    return gaussian_filter1d(incis, truncate=1, sigma=4, mode="constant") / n_trials_per_session


rips_df = utils.rips.load_rips()
cons_df = utils.rips.load_cons_across_trials()


rips_df = rips_df[(0.2 < rips_df.center_time) * (rips_df.center_time < 7.8)]
cons_df = cons_df[(0.2 < cons_df.center_time) * (cons_df.center_time < 7.8)]

rips_df["subject_session"] = mngs.ml.utils.merge_labels(rips_df["subject"], rips_df["session"])
cons_df["subject_session"] = mngs.ml.utils.merge_labels(cons_df["subject"], cons_df["session"])

ns_rips = np.array([get_a_line(rips_df[rips_df.subject_session == ss].center_time)
                    for ss in rips_df.subject_session.unique()])
ns_cons = np.array([get_a_line(cons_df[cons_df.subject_session == ss].center_time)
                    for ss in cons_df.subject_session.unique()])

xx = np.linspace(0,8,160)
yy_rips, ss_rips = ns_rips.mean(axis=0), ns_rips.std(axis=0)
yy_cons, ss_cons = ns_cons.mean(axis=0), ns_cons.std(axis=0)






import random
bs_samples = np.array([random.sample(sorted(yy_rips.tolist()), 1) for _ in range(1000)]).squeeze()
bs_samples = pd.DataFrame(bs_samples)
# bs_0025 = bs_samples.quantile(0.025)
bs_0950 = bs_samples.quantile(0.950)
# bs_0975 = bs_samples.quantile(0.975)


fig, ax = plt.subplots()
ax.fill_between(xx, yy_rips - ss_rips/2, yy_rips + ss_rips/2, alpha=.1)
ax.plot(xx, yy_rips)
ax.fill_between(xx, yy_cons - ss_cons/2, yy_cons + ss_cons/2, alpha=.1)
ax.plot(xx, yy_cons)
ax.plot(xx, [bs_0950 for _ in range(len(xx))])
# ax.fill_between(xx, bs_0025, bs_0975, alpha=0.1)
plt.show()



df = pd.DataFrame({
    "x": xx-6,
    "y_under_rips": yy_rips - ss_rips / 2,
    "y_mean_rips": yy_rips,
    "y_upper_rips": yy_rips + ss_rips / 2,
    "y_under_cons": yy_cons - ss_cons / 2,
    "y_mean_cons": yy_cons,
    "y_upper_cons": yy_cons + ss_cons / 2,
    "is_significant": (yy_rips > bs_0950.iloc[0]).astype(int),
})
mngs.io.save(df, "./tmp/figs/line/ripple_inci.csv")
