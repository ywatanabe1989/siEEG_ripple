#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-30 16:37:16 (ywatanabe)"

import mngs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def prepare_rips_df(roi, x_width_s=.5):
    
    
    rips_df = mngs.io.load(f"./tmp/rips_df/common_average_2.0_SD_{roi}.pkl")

    rips_df = rips_df[["subject", "session", "center_time", "response_time", "phase", "set_size"]]
    rips_df["response_time_centered_time"] = (rips_df["center_time"] - 6) - rips_df[
        "response_time"
    ]

    rips_df = rips_df[(rips_df.phase == "Retrieval") + (rips_df.phase == "Fixation")]

    rips_df = rips_df[1 < rips_df.response_time]
    rips_df = rips_df[
        (-x_width_s < rips_df["response_time_centered_time"])
        * (rips_df["response_time_centered_time"] < 0) # x_width_s
        + (rips_df["phase"] == "Fixation")
    ]

    # rips_df = rips_df[rips_df.session.astype(int) <= 2]
    rips_df["subject-session"] = mngs.ml.utils.merge_labels(
        rips_df.subject, rips_df.session
    )
    rips_df["ROI"] = roi
    return rips_df.reset_index()


def plot_1(rips_df, set_size=None, x_width_s = .5):
    if set_size is not None:
        rips_df = rips_df[rips_df.set_size == set_size]
    # plots 1
    fig, axes = plt.subplots(
        nrows=len(rips_df.subject.unique()), sharex=True, sharey=True
    )
    for i_subject, subject in enumerate(rips_df.subject.unique()[::-1]):
        try:
            ax = axes[i_subject]

            rips_df_F_subj = rips_df[
                (rips_df.subject == subject) * (rips_df.phase == "Fixation")
            ]
            rips_df_R_subj = rips_df[
                (rips_df.subject == subject) * (rips_df.phase == "Retrieval")
            ]
            # n_rips = len(rips_df_F_subj)
            # n_rips = len(rips_df_R_subj)
            n_rips = 1
            n_sessions = int(rips_df_R_subj.session.unique()[-1])
            hist, bin_edges = np.histogram(
                rips_df_R_subj.response_time_centered_time,
                bins=10,
                range=(-x_width_s, 0),
            )
            hist = hist.astype(float) / n_sessions / n_rips
            x = (bin_edges[1:] + bin_edges[:-1]) / 2
            y = hist
            ax.bar(x, y, width=(bin_edges[1] - bin_edges[0]))
            # sns.kdeplot(rips_df_R_subj.response_time_centered_time, ax=ax)
            ax.set_ylabel(f"{subject}\n{hist.sum():.3f}\nn_sessions: {n_sessions}")
        except Exception as e:
            print(e)
        ax.set_xlim(-.5, 0)
    roi = rips_df["ROI"].iloc[0]
    fig.suptitle(roi)
    # plt.show()
    mngs.io.save(
        fig, f"./tmp/figs/hist/corr_between_ripple_time_and_response_time/1/{roi}_set_size_{set_size}.png"
    )
    plt.close()


def plot_2(rips_df, set_size=None):
    if set_size is not None:
        rips_df = rips_df[rips_df.set_size == set_size]

    # plots 2
    fig, ax = plt.subplots()
    sns.histplot(
        data=rips_df,
        x="response_time_centered_time",
        hue="subject",
        # hue="subject-session",
        kde=True,
        stat="probability",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlim(-.5, 0)
    roi = rips_df["ROI"].iloc[0]
    ax.set_title(roi)
    mngs.io.save(
        fig, f"./tmp/figs/hist/corr_between_ripple_time_and_response_time/2/{roi}_set_size_{set_size}.png"
    )
    plt.close()


def main(roi, set_size=None):
    rips_df = prepare_rips_df(roi)
    import ipdb; ipdb.set_trace()
    rips_df["response_time_centered_time"] =     rips_df["response_time_centered_time"].astype(float)
    print(rips_df[rips_df.phase == "Retrieval"].pivot_table(columns=["set_size"]))

    plot_1(rips_df, set_size=set_size)
    # plot_2(rips_df, set_size=set_size)


if __name__ == "__main__":
    for set_size in [4, 6, 8]:
        try:
            main("AHL", set_size=set_size)
            main("AHR", set_size=set_size)
            main("PHL", set_size=set_size)
            main("PHR", set_size=set_size)
        except Exception as e:
            print(e)
        # main("ECL", set_size=set_size)
        # main("ECR", set_size=set_size)
        # main("AL", set_size=set_size)
        # main("AR", set_size=set_size)

    # rips_df = prepare_rips_df("PHR")
    # rips_df["pre_to_post_ratio"] = (rips_df["response_time_centered_time"] < 0).astype(
    #     float
    # ) - (0 < rips_df["response_time_centered_time"]).astype(float)

    # print(rips_df.pivot_table(columns="subject").T)
