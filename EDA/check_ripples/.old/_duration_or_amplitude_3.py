#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-25 17:27:47 (ywatanabe)"

import re
from glob import glob
from bisect import bisect_left
import mngs
import numpy as np
from natsort import natsorted
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from itertools import combinations
import warnings

import sys

sys.path.append(".")
from siEEG_ripple import utils
import seaborn as sns
import scipy
import pingouin
from itertools import combinations
from pprint import pprint


def plot_violin_set_size_and_duration_by_phase(rips_df):
    fig, axes = plt.subplots(
        ncols=4, sharex=True, sharey=True, figsize=(1.3 * 4, 0.91 * 2)
    )
    rips_df["set_size"] = rips_df["set_size"].astype(int)
    for ax, phase in zip(axes, PHASES):
        rips_df["hue"] = True
        sns.violinplot(
            data=rips_df[rips_df.phase == phase],
            x="set_size",
            # y="duration_ms",
            y="log10(duration_ms)",
            hue="hue",
            hue_order=[False, True],
            split=True,
            ax=ax,
            color=mngs.plt.colors.to_RGBA("blue"),
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend_ = None
        # ax.set_yscale("log")
    # fig.supxlabel("Ripple duration [ms]")
    # plt.show()
    return fig


def plot_hist_set_size_and_duration_by_phase(rips_df):
    fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True)
    for ax, phase in zip(axes, PHASES):
        sns.histplot(
            data=rips_df[rips_df.phase == phase],
            hue="set_size",
            x="log10(duration_ms)",
            stat="probability",
            common_norm=False,
            ax=ax,
            kde=True,
            legend=False,
        )
        ax.set_xlabel("")
        # ax.set_xscale("log")
    fig.supxlabel("log10(Ripple duration [ms])")
    # plt.show()
    return fig


def test_set_size_and_duration_by_phase(rips_df):
    pass


def plot_hist_set_size_and_amplitude_by_phase(rips_df):
    fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True)
    for ax, phase in zip(axes, PHASES):
        sns.histplot(
            data=rips_df[rips_df.phase == phase],
            hue="set_size",
            x="log10(ripple_peak_amplitude_sd)",
            stat="probability",
            common_norm=False,
            ax=ax,
            kde=True,
            legend=True,
        )
        ax.set_xlabel("")
        # ax.set_xscale("log")
    fig.supxlabel("log10(Ripple peak amplitude [SD of baseline])")
    # plt.show()
    return fig


def plot_box_x_set_size_y_duration_ax_phase(rips_df, phase):
    return plot_box_x_set_size_y_var_ax_phase(rips_df, phase, "log10(duration_ms)")


def plot_box_x_set_size_y_amplitude_ax_phase(rips_df, phase):
    return plot_box_x_set_size_y_var_ax_phase(
        rips_df, phase, "log10(ripple_peak_amplitude_sd)"
    )


def plot_box_x_set_size_y_var_ax_phase(rips_df, phase, var):
    data = rips_df[rips_df.phase == phase]
    fig, ax = plt.subplots()
    sns.boxplot(
        data=data,
        y=var,
        x="set_size",
        ax=ax,
    )
    ax.set_title(phase)
    ax.set_xlabel(f"# of letters in {phase}")
    ylabel = "Ripple duration [ms]" if var == "duration_ms" else None
    ylabel = (
        "Ripple peak amplitude[SD of baseline]"
        if var == "ripple_peak_amplitude_sd"
        else ylabel
    )
    ax.set_ylabel(ylabel)
    return fig


def test_set_size_and_duration(rips_df, var="duration_ms", alternative="two-sided"):

    assert (var == "duration_ms") or (var == "ripple_peak_amplitude_sd")
    rips_df["log_var"] = np.log(rips_df[var])

    for phase in PHASES:
        # anova
        print(phase)
        pprint(
            pingouin.kruskal(
                data=rips_df[rips_df.phase == phase],
                dv="log_var",
                between="set_size",
            )
        )
        print()

        # pairwise
        for ss_a, ss_b in combinations([4, 6, 8], 2):
            print(ss_a, ss_b)
            pprint(
                scipy.stats.brunnermunzel(
                    rips_df[(rips_df.phase == phase) * (rips_df.set_size == ss_a)][
                        "log_var"
                    ],
                    rips_df[(rips_df.phase == phase) * (rips_df.set_size == ss_b)][
                        "log_var"
                    ],
                    alternative=alternative,
                )
            )
            print()
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    import argparse
    import mngs
    import scipy
    import pingouin

    # Parameters
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_SEC = [1, 2, 3, 2]
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    # Loads
    _rips_df = utils.load_rips()
    trials_df = utils.load_trials(add_n_ripples=True)
    # rips_df["duration_ms"] = (1000 * (rips_df.end_time - rips_df.start_time)).astype(
    #     float
    # )
    # rips_df["log10(duration_ms)"] = np.log10(rips_df["duration_ms"])
    # rips_df["log10(ripple_peak_amplitude_sd)"] = np.log10(
    #     rips_df["ripple_peak_amplitude_sd"]
    # ).astype(float)

    # Normalize log10(duration_ms) and ripple_peak_amplitude_sd by Fixation
    rips_normed = _rips_df.copy()[
        [
            "subject",
            "session",
            "phase",
            "set_size",
            "log10(duration_ms)",
            "log10(ripple_peak_amplitude_sd)",
        ]
    ]
    base_df = rips_normed[rips_normed.phase == "Fixation"]

    def get_indi(df, subject, session, set_size):
        indi_subject = df.subject == row.subject
        indi_session = df.session == row.session
        indi_set_size = df.set_size == row.set_size
        return indi_subject * indi_session * indi_set_size
    
    for _, row in rips_normed[["subject", "session", "set_size"]].drop_duplicates().iterrows():
        print(len(base_df[get_indi(base_df, row.subject, row.session, row.set_size)]))
        rips_normed[get_indi(rips_normed, row.subject, row.session, row.set_size)]
    
    base_df = rips_normed.pivot_table(
        columns=["subject", "session", "phase", "set_size"], aggfunc=np.median
    ).T.reset_index()
    base_df = base_df[base_df["phase"] == "Fixation"]# [
    #     [
    #         "subject",
    #         "session",
    #         "phase",
    #         "set_size",
    #         "log10(duration_ms)",
    #         "log10(ripple_peak_amplitude_sd)",
    #     ]
    # ]
    from scipy.stats import zscore
    for ii, (_, base) in enumerate(base_df.iterrows()):
        indi_subject = rips_normed.subject == base.subject
        indi_session = rips_normed.session == base.session
        indi_set_size = rips_normed.set_size == base.set_size

        
        for k in ["log10(duration_ms)", "log10(ripple_peak_amplitude_sd)"]:
            rips_normed.loc[(indi_subject * indi_session * indi_set_size), k] = \
                zscore(rips_normed.loc[
                    (indi_subject * indi_session * indi_set_size), k
                ])

    # Remove outliers
    # rips_normed = rips_normed[rips_normed["log10(duration_ms)"] < 1.5]
    # rips_normed = rips_normed[rips_normed["log10(ripple_peak_amplitude_sd)"] < 1.5]    

    # Remove subject #03 and #09
    # rips_df = rips_df[rips_df.subject != "03"]
    # trials_df = trials_df[trials_df.subject != "03"]
    # rips_df = rips_df[rips_df.subject != "09"]
    # trials_df = trials_df[trials_df.subject != "09"]

    # Subtract biases for set size

    # Plots
    # duration
    fig = plot_violin_set_size_and_duration_by_phase(rips_normed)
    # plt.show()
    mngs.io.save(fig, "./tmp/figs/violin/set_size_and_duration_by_phase.tif")

    fig = plot_hist_set_size_and_duration_by_phase(rips_normed)
    mngs.io.save(fig, "./tmp/figs/hist/set_size_and_duration_by_phase.png")

    fig = plot_box_x_set_size_y_duration_ax_phase(rips_normed, phase="Fixation")
    fig = plot_box_x_set_size_y_duration_ax_phase(rips_normed, phase="Encoding")
    mngs.io.save(fig, "./tmp/figs/box/set_size_and_duration_by_Encoding.png")

    test_set_size_and_duration(rips_df, var="duration_ms", alternative="less")
    # less, Encoding
    # 4 8
    # BrunnerMunzelResult(statistic=2.112247564360095, pvalue=0.017884862050538297)

    # amplitude
    fig = plot_hist_set_size_and_amplitude_by_phase(rips_normed)
    mngs.io.save(fig, "./tmp/figs/hist/set_size_and_amplitude_by_phase.png")

    fig = plot_box_x_set_size_y_amplitude_ax_phase(rips_df, phase="Encoding")
    fig = plot_box_x_set_size_y_amplitude_ax_phase(rips_df, phase="Retrieval")
    mngs.io.save(fig, "./tmp/figs/box/set_size_and_amplitude_by_Retrieval.png")

    test_set_size_and_duration(
        rips_df, var="ripple_peak_amplitude_sd", alternative="less"
    )
    # less, Retrieval
    # 4 8
    # BrunnerMunzelResult(statistic=1.6780143990059153, pvalue=0.04715061103424223)
    # 6 8
    # BrunnerMunzelResult(statistic=1.7558886715109228, pvalue=0.04002156819038405)
