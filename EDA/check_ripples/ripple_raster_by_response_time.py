#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-25 15:14:25 (ywatanabe)"
import mngs
import sys

sys.path.append(".")
import utils
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Loads
rips_df = utils.load_rips()
trials_df = utils.load_trials(add_n_ripples=True)

df = []
for _, trial in trials_df.iterrows():
    _rips_df = rips_df[
        (rips_df.subject == trial.subject)
        * (rips_df.session == trial.session)
        * (rips_df.trial_number == trial.trial_number)
    ]
    trial["ripple_centers"] = (_rips_df["start_time"] + _rips_df["end_time"]) / 2
    df.append(trial)
# df = pd.concat(df, axis=1).T
df = pd.concat(df, axis=1).T
df = df[["subject", "session", "trial_number", "response_time", "ripple_centers"]]

df = df.sort_values(["subject", "session", "response_time"])  # , "trial_number"
# df = df.sort_values(["response_time"]) # , "trial_number"
df = df[df.response_time <= 1.9]

# koko
colors_dict = {
    "01": "blue",
    "02": "orange",
    "03": "green",
    "04": "purple",
    "06": "brown",
    "09": "pink",
}
# plots
fig, ax = plt.subplots(figsize=(6.4*2, 4.8*2))
ax.set_xlabel("Time [s]")
ax.set_ylabel("Trial #")
before_subject, before_session = df.iloc[0].subject, df.iloc[0].session # "00", "00"# 
Es, Ms, Rs, rt_starts, rt_ends = [], [], [], [], []
xx, yy = [], []
for ii, (_, trial) in enumerate(df.iterrows()):
    # Response time
    _x_rt = trial.response_time
    x_rt = _x_rt
    ax.plot(x_rt, np.ones_like(x_rt) * ii, ls="", marker="|", color="red")
    rt_starts.append(np.nanmin(x_rt))
    rt_ends.append(np.nanmax(x_rt))

    # Ripple time
    start = -6
    end = 1.9
    x_rc = trial.ripple_centers.astype(float) - 6
    x_rc = x_rc[(start < x_rc) * (x_rc < end)]
    ax.plot(x_rc, np.ones_like(x_rc) * ii, ls="", marker="|", color="black")
    xx.append(x_rc)
    yy.append(np.ones_like(x_rc) * ii)    

    # ########################################
    # # median
    # # Encoding
    # start = -5
    # end = -3
    # # x_rc = trial.ripple_centers.astype(float) - 6
    # x_E = x_rc[(start < x_rc) * (x_rc < end)].median()
    # Es.append((x_E, np.ones_like(x_E) * ii))
    # ax.plot(x_E, np.ones_like(x_E) * ii, ls="", marker="|", color="blue")

    # # Maintenance
    # start = -3
    # end = 0
    # # x_rc = trial.ripple_centers.astype(float) - 6
    # x_M = x_rc[(start < x_rc) * (x_rc < end)].median()
    # Ms.append((x_M, np.ones_like(x_M) * ii))
    # ax.plot(x_M, np.ones_like(x_M) * ii, ls="", marker="|", color="blue")

    # # Retrieval
    # start = 0  # rt_start # 0.7
    # end = 1.8  # rt_end # 1.7
    # # x_rc = trial.ripple_centers.astype(float) - 6
    # x_R = x_rc[(start <= x_rc) * (x_rc <= end)].median()
    # Rs.append((x_R, np.ones_like(x_R) * ii))
    # ax.plot(x_R, np.ones_like(x_R) * ii, ls="", marker="|", color="blue")
    # ########################################    

    # Vertical line
    retrieval_start = 0
    maintenance_start = -3
    encoding_start = -5
    fixation_start = -6
    ax.plot(retrieval_start, 1 * ii, ls="", marker="|", color="gray", alpha=0.5)
    ax.plot(maintenance_start, 1 * ii, ls="", marker="|", color="gray", alpha=0.5)
    ax.plot(encoding_start, 1 * ii, ls="", marker="|", color="gray", alpha=0.5)

    # Horizontal line
    current_subject, current_session = df.iloc[ii].subject, df.iloc[ii].session
    if ((before_session != current_session) and (ii != 0)) or (ii == len(df) - 1):# or (ii == 0): # :
        ax.axhline(y=ii, linestyle="--", color="gray", alpha=0.5)
        ax.text(
            x=-7.5, y=ii, s=f"Subject #{before_subject}\nSession #{before_session}",
            color=colors_dict[before_subject]
        )
        # try:        
        xx, yy = np.hstack(xx), np.hstack(yy)
        nonnan = ~np.isnan(xx)
        xx,yy = xx[nonnan], yy[nonnan]
        order = np.argsort(xx)
        xx, yy = xx[order], yy[order]                    
        ax.plot(xx, yy, linestyle="--", color=colors_dict[before_subject])  # xx, yy, "yo",
        # ax.set_xlim(0.5, None)
        xx, yy = [], []
        # except Exception as e:
        #     print(e)
        Es, Ms, Rs, rt_starts, rt_ends = [], [], [], [], []
        before_subject = current_subject        
        before_session = current_session

mngs.io.save(fig, "./tmp/figs/raster/ripple.png")
# plt.show()
