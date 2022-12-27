#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-26 11:50:14 (ywatanabe)"

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

# def plot_hist_duration_and_ripple_count_by_phase_and_correct(rips_df):
#     fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
#     for i_ax, (ax, is_correct) in enumerate(zip(axes, (False, True))):
#         sns.histplot(
#                 data=rips_df[rips_df.correct == is_correct],
#                 x="duration_ms",
#                 hue="phase",
#                 hue_order=PHASES,
#                 stat="probability",
#                 kde=True,
#                 common_norm=False,
#                 log_scale=True,
#                 ax=ax,
#                 alpha=0.3,
#             )
#         ax.set_xlabel("Ripple duration [ms]")
#         correct_str = "Correct" if is_correct else "Incorrect"
#         ax.set_title(correct_str)
#     # plt.show()
#     return fig

def plot_hist_x_duration_y_ripple_count_hue_phase(rips_df):
    return plot_hist_x_var_y_ripple_count_hue_phase(rips_df, "duration_ms")

def plot_hist_x_amplitude_y_ripple_count_hue_phase(rips_df):
    return plot_hist_x_var_y_ripple_count_hue_phase(rips_df, "ripple_peak_amplitude_sd")

def plot_hist_x_var_y_ripple_count_hue_phase(rips_df, var):
    fig, ax = plt.subplots()
    sns.histplot(
            data=rips_df[rips_df.correct == True],
            x=var,
            hue="phase",
            hue_order=PHASES,
            stat="probability",
            kde=True,
            common_norm=False,
            log_scale=True,
            ax=ax,
            alpha=0.3,
        )
    xlabel = "Ripple duration [ms]" if var == "duration_ms" else None
    xlabel = "Ripple peak amplitude [SD]" if var == "ripple_peak_amplitude_sd" else xlabel
    ax.set_xlabel(xlabel)
    # ax.set_title(correct_str)
    # plt.show()
    return fig

# def plot_hist_amplitude_and_ripple_count_by_phase_and_correct(rips_df):
#     fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
#     for i_ax, (ax, is_correct) in enumerate(zip(axes, (False, True))):
#         sns.histplot(
#                 data=rips_df[rips_df.correct == is_correct],
#                 x="ripple_peak_amplitude_sd",
#                 hue="phase",
#                 hue_order=PHASES,
#                 stat="probability",
#                 kde=True,
#                 common_norm=False,
#                 log_scale=True,
#                 ax=ax,
#                 alpha=0.3,
#             )

#         correct_str = "Correct" if is_correct else "Incorrect"
#         ax.set_title(correct_str)
#         ax.set_xlabel("")
#     fig.supxlabel("Ripple peak amplitude [SD of baseline]")        
#     # plt.show()
#     return fig

# def plot_bar_long_ripple_rates_by_correct(rips_df):
#     long_percs = 100 * rips_df.pivot_table(columns=["phase", "correct"], aggfunc="mean")\
#         [PHASES].T["is_long"]
#     long_percs.name = "long_percs"

#     fig, ax = plt.subplots()
#     sns.barplot(
#         data=long_percs.reset_index(),
#             x="phase",
#             y="long_percs",
#             hue="correct",
#             ax=ax,
#         )
#     ax.set_xlabel("Phase")
#     ax.set_ylabel("Long ripple rate")

#     # plt.show()
#     return fig

def plot_bar_x_phase_y_long_ripple_rate(rips_df):
    return plot_bar_x_phase_y_var(rips_df, var="is_long")

def plot_bar_x_phase_y_large_ripple_rate(rips_df):
    return plot_bar_x_phase_y_var(rips_df, var="is_large")    

def plot_bar_x_phase_y_var(rips_df, var):
    fig, ax = plt.subplots()
    sns.barplot(
        data=rips_df[rips_df.correct == True],
        x="phase",
        order=PHASES,
        y=var,
        # hue="correct",
        ax=ax,
        )
    ax.set_xlabel("Phase")
    ylabel = "Long ripple rate" if var == "is_long" else None
    ylabel = "Large ripple rate" if var == "is_large" else ylabel
    ax.set_ylabel(ylabel)
    # plt.show()
    return fig

# def plot_bar_large_ripple_rates(rips_df):
#     fig, ax = plt.subplots()
#     sns.barplot(
#         data=rips_df,
#         x="phase",
#         order=PHASES,
#         y="is_large",
#         ax=ax,
#         )
#     ax.set_xlabel("Phase")
#     ax.set_ylabel("Large ripple rate")

#     # plt.show()
#     return fig


def test_long_ripple_rate(rips_df):
    test_var_ripple_rate(rips_df, "is_long")

def test_large_ripple_rate(rips_df):
    test_var_ripple_rate(rips_df, "is_large")

def test_var_ripple_rate(rips_df, var):    

    F_rips_df = rips_df[rips_df.phase == "Faintenance"]
    E_rips_df = rips_df[rips_df.phase == "Encoding"]    
    M_rips_df = rips_df[rips_df.phase == "Maintenance"]
    R_rips_df = rips_df[rips_df.phase == "Retrieval"]        

    for combi in combinations([{"Fixation": F_rips_df},
                               {"Encoding": E_rips_df},
                               {"Maintenance": M_rips_df},
                               {"Retrieval": R_rips_df},
                               ],
                              2
                              ):
        phase_1_str = list(combi[0].keys())[0]
        phase_1_rips_df = rips_df[rips_df.phase == phase_1_str]
        phase_2_str = list(combi[1].keys())[0]
        phase_2_rips_df = rips_df[rips_df.phase == phase_2_str]

        print(phase_1_str, phase_2_str)
        
        print(scipy.stats.brunnermunzel(
            phase_1_rips_df[var][phase_1_rips_df.correct == True],                
            phase_2_rips_df[var][phase_2_rips_df.correct == True],
        ))
        print()

# def test_large_ripple_rates_by_correct(rips_df):
#     rips_df["is_large"] = rips_df.ripple_peak_amplitude_sd > 3
    
#     F_rips_df = rips_df[rips_df.phase == "Faintenance"]
#     E_rips_df = rips_df[rips_df.phase == "Encoding"]    
#     M_rips_df = rips_df[rips_df.phase == "Maintenance"]
#     R_rips_df = rips_df[rips_df.phase == "Retrieval"]        

#     for combi in combinations([{"Fixation": F_rips_df},
#                                {"Encoding": E_rips_df},
#                                {"Maintenance": M_rips_df},
#                                {"Retrieval": R_rips_df},
#                                ],
#                               2
#                               ):
#         phase_1_str = list(combi[0].keys())[0]
#         phase_1_rips_df = rips_df[rips_df.phase == phase_1_str]
#         phase_2_str = list(combi[1].keys())[0]
#         phase_2_rips_df = rips_df[rips_df.phase == phase_2_str]

#         print(phase_1_str, phase_2_str)
        
#         print(scipy.stats.brunnermunzel(
#             phase_1_rips_df.is_large[phase_1_rips_df.correct == True],                
#             phase_2_rips_df.is_large[phase_2_rips_df.correct == True],
#         ))


if __name__ == "__main__":
    import argparse
    import mngs

    # Parameters
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    SAMP_RATE_iEEG = mngs.io.load("./config/global.yaml")["SAMP_RATE_iEEG"]
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]    

    # Loads
    rips_df = utils.load_rips(from_pkl=False)

    # Plots
    # fig = plot_hist_duration_and_ripple_count_by_phase_and_correct(rips_df)
    fig = plot_hist_x_duration_y_ripple_count_hue_phase(rips_df)
    mngs.io.save(fig, "./tmp/figs/hist/x_duration_y_ripple_count_hue_phase.png")
    plt.close()

    fig = plot_bar_x_phase_y_long_ripple_rate(rips_df)
    mngs.io.save(fig, "./tmp/figs/bar/x_phase_y_long_ripple_rate.png")
    plt.close()
    # fig = plot_bar_long_ripple_rates_by_correct(rips_df)
    # mngs.io.save(fig, "./tmp/figs/bar/phase_and_long_ripple_rate_by_correct.png")
    # plt.close()
    
    test_long_ripple_rate(rips_df)
    # Maintenance Retrieval
    # BrunnerMunzelResult(statistic=-3.0633755230534754, pvalue=0.0022418003976355785)
    
    # Peak amplitude
    fig = plot_hist_x_amplitude_y_ripple_count_hue_phase(rips_df)
    mngs.io.save(fig, "./tmp/figs/hist/x_amplitude_y_ripple_count_hue_phase.png")    
    plt.close()

    fig = plot_bar_x_phase_y_large_ripple_rate(rips_df)
    mngs.io.save(fig, "./tmp/figs/bar/x_phase_y_large_ripple_rate.png")
    plt.close()
    # fig = plot_bar_large_ripple_rates_by_correct(rips_df)
    # mngs.io.save(fig, "./tmp/figs/bar/phase_and_large_ripple_rate_by_correct.png")
    # plt.close()

    test_large_ripple_rate(rips_df)
    # Fixation Encoding
    # BrunnerMunzelResult(statistic=-2.969165503840378, pvalue=0.0032390942829877496)

    # Fixation Retrieval
    # BrunnerMunzelResult(statistic=2.343242668696929, pvalue=0.019630937639853796)

    # Encoding Maintenance
    # BrunnerMunzelResult(statistic=4.484723881336943, pvalue=8.115720163237583e-06)

    # Encoding Retrieval
    # BrunnerMunzelResult(statistic=7.204224046406757, pvalue=1.2612133559741778e-12)

    # Maintenance Retrieval
    # BrunnerMunzelResult(statistic=3.630775007917092, pvalue=0.00029743536066462184)
