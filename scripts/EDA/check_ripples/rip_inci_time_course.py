#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-15 15:54:03 (ywatanabe)"

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


# def get_a_line(ctimes):
#     t_sec = 8
#     xx_s = np.linspace(0, t_sec, 160)
#     n_trials_per_session = 50
#     incis = []
#     width_s = xx_s[1] - xx_s[0]
#     for ii in range(0, len(xx_s)):
#         cc_s = xx_s[ii]
#         pre = cc_s - width_s / 2
#         post = cc_s + width_s / 2
#         # cur = x[ii]
#         n = ((pre <= ctimes) * (ctimes < post)).sum()
#         inci = n / width_s
#         incis.append(inci)
#     return (
#         gaussian_filter1d(incis, truncate=1, sigma=4, mode="constant")
#         / n_trials_per_session
#     )

def calc_inci_ci(event_df):
    # to digi
    events_digi = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            event_df_session = event_df[(event_df.subject == subject) * (event_df.session == session)]
            # roi = event_df_session.roi.iloc[0]
            events_digi.append(utils.rips.mk_events_mask(event_df_session, subject, session, roi, 0))
    events_digi = np.vstack(events_digi)
    # return events_digi
    bin_s = 50 / 1000
    events_digi_inci = events_digi / bin_s
    events_digi_inci_smoothed = np.array(
        [
            gaussian_filter1d(events_digi_inci[ii], truncate=1, sigma=4, mode="constant")
            for ii in range(len(events_digi_inci))
        ]
    )
    ns = events_digi_inci_smoothed.shape[0] * np.ones(events_digi_inci_smoothed.shape[-1])

    # # filtering
    # ns = gaussian_filter1d(ns, truncate=1, sigma=4, mode="constant")
    # events_digi_inci = np.array([gaussian_filter1d(events_digi_inci[ii], truncate=1, sigma=4, mode="constant")
    #                   for ii in range(len(events_digi_inci))])

    mm = events_digi_inci_smoothed.mean(axis=0)
    sd = events_digi_inci_smoothed.std(axis=0)
    ci = 1.96 * sd / ns

    under = mm - ci
    middle = mm
    upper = mm + ci
    # under = mm - sd
    # middle = mm
    # upper = mm + sd

    # under = gaussian_filter1d(under, truncate=1, sigma=4, mode="constant")
    # middle = gaussian_filter1d(middle, truncate=1, sigma=4, mode="constant")
    # upper = gaussian_filter1d(upper, truncate=1, sigma=4, mode="constant")

    return under, middle, upper

if __name__ == "__main__":
    import random
    
    # Loads
    rips_df = utils.rips.load_rips()
    cons_df = utils.rips.load_cons_across_trials()

    rips_df["center_time"] = (rips_df.start_time + rips_df.end_time).astype(float) / 2
    assert np.all(sorted(rips_df.center_time) == sorted(cons_df.center_time))
    
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    """
    rips_digi = calc_inci_ci(rips_df) # 937
    cons_digi = calc_inci_ci(cons_df) # 986
    """    

    
    # fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    # for i_match, match in enumerate([1,2]):
    #     ax = axes[i_match]
    #     for i_set_size, set_size in enumerate([4,6,8]):
    #         rips_df_s = rips_df[(rips_df.set_size == set_size)*(rips_df.match == match)*(rips_df.correct==True)]
    #         # cons_df_s = cons_df[cons_df.set_size == set_size]

    #         # Calculates SWR incidence
    #         under_rip, middle_rip, upper_rip = calc_inci_ci(rips_df_s)
    #         # under_con, middle_con, upper_con = calc_inci_ci(cons_df_s)

    #         # Plots
    #         t_sec = 8
    #         xx_s = np.linspace(0, t_sec, 160)

    #         c = ["orange", "red", "pink"][i_set_size]
    #         ax.plot(xx_s, under_rip, color=c)
    #         ax.plot(xx_s, middle_rip, label=set_size, color=c)
    #         ax.plot(xx_s, upper_rip, color=c)
    #         ax.legend()
    # plt.show()

    # koko
    fig, ax = plt.subplots()

    # Calculates SWR incidence
    under_rip, middle_rip, upper_rip = calc_inci_ci(rips_df)
    under_con, middle_con, upper_con = calc_inci_ci(cons_df)

    # Plots
    t_sec = 8
    xx_s = np.linspace(0, t_sec, 160)

    ax.plot(xx_s, under_rip)
    ax.plot(xx_s, middle_rip)
    ax.plot(xx_s, upper_rip)
    ax.legend()
    plt.show()
    

    bs_samples = np.array(
        [random.sample(sorted(middle_rip.tolist()), 1) for _ in range(1000)]
    ).squeeze()
    bs_samples = pd.DataFrame(bs_samples)
    # bs_0025 = bs_samples.quantile(0.025)
    bs_0950 = bs_samples.quantile(0.950)
    # bs_0975 = bs_samples.quantile(0.975)


    fig, ax = plt.subplots()
    ax.fill_between(xx_s, under_rip, upper_rip, alpha=0.1)
    ax.plot(xx_s, middle_rip)
    ax.fill_between(xx_s, under_con, upper_con, alpha=0.1)
    ax.plot(xx_s, middle_con)
    ax.plot(xx_s, [bs_0950 for _ in range(len(xx_s))])
    # ax.fill_between(xx, bs_0025, bs_0975, alpha=0.1)
    plt.show()


    df = pd.DataFrame(
        {
            "x": xx_s - 6,
            "y_under_rips": under_rip,
            "y_mean_rips": middle_rip,
            "y_upper_rips": upper_rip,
            "y_under_cons": under_con,
            "y_mean_cons": middle_con,
            "y_upper_cons": upper_con,            
            "is_significant": (middle_rip > bs_0950.iloc[0]).astype(int),
            "bs_0950": bs_0950.iloc[0] * np.ones_like(under_rip)
        }
    )
    mngs.io.save(df, "./tmp/figs/line/ripple_inci.csv")
