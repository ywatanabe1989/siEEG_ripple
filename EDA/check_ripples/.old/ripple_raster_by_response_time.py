#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-25 13:41:13 (ywatanabe)"
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
# plots
fig, ax = plt.subplots()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Trial #")
before_subject, before_session = df.iloc[0].subject, df.iloc[0].session
Es, Ms, Rs, rt_starts, rt_ends = [], [], [], [], []
for ii, (_, trial) in enumerate(df.iterrows()):
    # Response time
    _x_rt = trial.response_time
    x_rt = _x_rt
    ax.plot(x_rt, np.ones_like(x_rt) * ii, ls="", marker="|", color="red")
    rt_starts.append(np.nanmin(x_rt))
    rt_ends.append(np.nanmax(x_rt))

    # # Ripple time
    start = -6
    end = 1.9
    x_rc = trial.ripple_centers.astype(float) - 6
    x_rc = x_rc[(start < x_rc) * (x_rc < end)]
    ax.plot(x_rc, np.ones_like(x_rc) * ii, ls="", marker="|", color="black")

    # Encoding median
    start = -4.5
    end = -3
    # x_rc = trial.ripple_centers.astype(float) - 6
    x_E = x_rc[(start < x_rc) * (x_rc < end)].median()
    Es.append((x_E, np.ones_like(x_E) * ii))
    ax.plot(x_E, np.ones_like(x_E) * ii, ls="", marker="|", color="blue")

    # Maintenance
    start = -2.5
    end = 0
    # x_rc = trial.ripple_centers.astype(float) - 6
    x_M = x_rc[(start < x_rc) * (x_rc < end)].median()
    Ms.append((x_M, np.ones_like(x_M) * ii))
    ax.plot(x_M, np.ones_like(x_M) * ii, ls="", marker="|", color="blue")

    # Retrieval
    start = 0  # rt_start # 0.7
    end = 1.8  # rt_end # 1.7
    # x_rc = trial.ripple_centers.astype(float) - 6
    x_R = x_rc[(start <= x_rc) * (x_rc <= end)]#.median()
    Rs.append((x_R, np.ones_like(x_R) * ii))
    ax.plot(x_R, np.ones_like(x_R) * ii, ls="", marker="|", color="blue")

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
    if (before_session != current_session) or (ii == 0) or (ii == len(df) - 1):
        ax.axhline(y=ii, linestyle="--", color="gray", alpha=0.5)
        ax.text(
            x=-7.5, y=ii, s=f"Subject #{current_subject}\nSession #{current_session}"
        )
        for i_Xs, Xs in enumerate([Es, Ms, Rs]):
            if i_Xs == 2:
                try:
                    xx = np.hstack([x[0] for x in Xs])
                    yy = np.hstack([x[1] for x in Xs])
                    nonnan = ~np.isnan(xx)
                    xx = xx[nonnan]
                    yy = yy[nonnan]

                    within_rt = (np.min(rt_starts) <= xx) * (xx <= np.max(rt_ends))
                    xx = xx[within_rt]
                    yy = yy[within_rt]

                    order = np.argsort(xx)
                    xx = xx[order]
                    yy = yy[order]                    

                    

                    # from scipy.optimize import curve_fit

                    # def sigmoid(k, L, x, x0, b):
                    #     return L / (1 + np.exp(-k + (x - x0))) + b

                    # p0 = [max(yy), np.median(xx), ii, min(yy)]
                    # popt, pcov = curve_fit(sigmoid, xx, yy, p0, method="dogbox")                    
                    # y_fit = sigmoid(xx, *popt)

                    # def func3(param, x, y):
                    #     residual = y - (param[0] * x**2 + param[1] * x + param[2])
                    #     return residual

                    # param3 = [0, 0, 0, 0]
                    # coefs = optimize.leastsq(func3, param3, args=(xx, yy))

                    # y_fit = (
                    #     coefs[0][0] * xx**3
                    #     + coefs[0][1] * xx**2
                    #     + coefs[0][2] * xx**1
                    #     + coefs[0][3] * xx**0
                    # )
                    # ax.plot(
                    #     xx, np.poly1d(np.polyfit(xx, yy, 3))(xx), linestyle="--"
                    # )  # xx, yy, "yo",
                    # coef = np.polyfit(xx, yy, 3)
                    # poly_fn = np.poly3d(coef)

                    ax.plot(xx, yy, linestyle="--")  # xx, yy, "yo",                    
                    # ax.plot(xx, np.poly1d(np.polyfit(xx, yy, 1))(xx), linestyle="--")  # xx, yy, "yo",
                    # ax.plot(xx, poly_fn(xx), linestyle="--")  # xx, yy, "yo",

                    # y_fit = coef[0] * xx**3 + coef[1] * xx**2 + coef[2] * xx**1 + coef[3] * xx**0
                    # ax.plot(xx, y_fit, linestyle="--")  # xx, yy, "yo",

                except Exception as e:
                    print(e)
        Es, Ms, Rs, rt_starts, rt_ends = [], [], [], [], []
        before_session = current_session

plt.show()
