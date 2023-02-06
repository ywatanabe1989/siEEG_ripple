#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-10-12 21:26:06 (ywatanabe)"

import mngs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def sample_n_min_sessions_per_subject(rips_df):
    n_sessions_per_sub = rips_df[["subject", "session"]].drop_duplicates()

    # n_sessions_min
    n_sessions_per_sub = rips_df[["subject", "session"]].drop_duplicates()
    n_sessions_per_sub["n"] = 1
    n_sessions_min = min(
        n_sessions_per_sub.pivot_table(columns=["subject"], aggfunc="sum").T["n"]
    )
    
    # sample n_sessions_min (= 2) x sessions from each subject
    indi_to_pick = []
    for i_subject in n_sessions_per_sub["subject"].unique():

        picked_sessions = np.random.permutation(
            n_sessions_per_sub[n_sessions_per_sub["subject"] == i_subject]["session"]
        )[:n_sessions_min]

        indi_to_pick_tmp = (rips_df["subject"] == i_subject) \
            * mngs.general.search(list(np.array(picked_sessions).astype(str)),
                                  list(np.array(rips_df["session"]).astype(str)), as_bool=True)[0]

        indi_to_pick.append(indi_to_pick_tmp)

    indi_to_pick = pd.concat(indi_to_pick, axis=1).any(axis=1)
    df = rips_df[indi_to_pick]
    # print(df[["subject", "session"]].drop_duplicates())
    df["Time from probe [s]"] = df["center_time"] - 6
    return df

def plot_kde(correct, set_size):
    _df = df[(df["correct"] == correct) * \
             (df["set_size"] == set_size)
             ]

    ax = sns.kdeplot(
        data=_df,
        x="center_time",
    )

    plt.show()

    
def plot_correct_incorrect_kde(df):
    fig, axes = plt.subplots(ncols=2, sharey=True)
    
    for i_correct, is_correct in enumerate([True, False]):
        
        ax = axes[i_correct]
        is_correct_str = "correct" if is_correct else "incorrect"
        
        sns.kdeplot(
            data=df[df["correct"] == is_correct],
            x="Time from probe [s]",
            hue="set_size",
            ax=ax
            )
        
        for x in [-6, -5, -3, -0]:
            ax.axvline(x=x, linestyle="dotted", color="gray")
            
        ax.set_ylim(0, .08)
        ax.set_xlim(-6, 2)
        ax.set_title(is_correct_str)
        ax.set_ylabel("KDE of ripple incidence [Hz]")
        
    mngs.io.save(fig, f"./tmp/ripple_incidence_correct_vs_incorrect_kdeplot.png")

def plot_correct_incorrect_bar(df):
    fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True)

    # count sample sizes
    df_trials = df[["subject", "session", "trial number", "correct", "set_size"]].drop_duplicates().copy()
    df_trials["n_trials"] = 1
    n_trials_df = df_trials.pivot_table(columns=["correct", "set_size"], aggfunc="sum").T.reset_index()\
        [["correct", "set_size", "n_trials"]]
    """
       correct  set_size  n_trials
    0    False       4.0       2.0
    1    False       6.0      23.0
    2    False       8.0      40.0
    3     True       4.0     323.0
    4     True       6.0     256.0
    5     True       8.0     242.0
    """

    df["n_ripples"] = 1 # n_ripples
    _df = df.pivot_table(columns=["phase", "set_size", "correct"], aggfunc="sum").T["n_ripples"].reset_index()

    # Ripple incidence per trial
    _df["Ripple incidence [Hz]"] = None # = n    
    for correct in n_trials_df["correct"].unique():
        for set_size in n_trials_df["set_size"].unique():
            _df_indi = (_df["correct"] == correct) * (_df["set_size"] == set_size)
            _trial_indi = (n_trials_df["correct"] == correct) * (n_trials_df["set_size"] == set_size)            

            n_ripples = _df.loc[_df_indi, "n_ripples"]
            n_trials = int(n_trials_df.loc[_trial_indi, "n_trials"])

            _df.loc[_df_indi, "Ripple incidence [Hz]"] = n_ripples / n_trials

    # devide by time [sec]
    _df.loc[_df["phase"] == "Fixation", "Ripple incidence [Hz]"] /= 1
    _df.loc[_df["phase"] == "Encoding", "Ripple incidence [Hz]"] /= 2
    _df.loc[_df["phase"] == "Maintenance", "Ripple incidence [Hz]"] /= 3
    _df.loc[_df["phase"] == "Retrieval", "Ripple incidence [Hz]"] /= 2

    for i_correct, is_correct in enumerate([True, False]):
        
        ax = axes[i_correct]
        is_correct_str = "correct" if is_correct else "incorrect"
        _df_correct_or_incorrect = _df[_df["correct"] == is_correct]

        for ss in [4, 6, 8]:
            n_trials = \
                n_trials_df[(n_trials_df["correct"] == is_correct) * (n_trials_df["set_size"] == ss)]["n_trials"]

            ii = (_df_correct_or_incorrect["correct"] == is_correct) * \
                 (_df_correct_or_incorrect["set_size"] == ss)
            
            _df_correct_or_incorrect.loc[ii, "set_size", ] = f"{ss} (n={int(n_trials.iloc[0])})"
        
        sns.barplot(
            data=_df_correct_or_incorrect,
            x="phase",
            y="Ripple incidence [Hz]",
            hue="set_size",
            order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
            ax=ax
            )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        ax.set_title(is_correct_str)
        ax.set_ylabel("Ripple incidence [Hz]")
        # ax.set_yscale("log")

        extend_ratio = 0.5
        ax = mngs.plt.ax_extend(ax, x_extend_ratio=extend_ratio, y_extend_ratio=extend_ratio)
        
    mngs.io.save(fig, f"./tmp/ripple_incidence_correct_vs_incorrect_barplot.png")
    
if __name__ == "__main__":
    mngs.general.fix_seeds(np=np)
    rips_df = mngs.io.load("./tmp/rips_df.csv")

    df = sample_n_min_sessions_per_subject(rips_df)

    n_trials = len(df[["trial number", "subject"]].drop_duplicates())

    plot_correct_incorrect_kde(df)


    mngs.plt.configure_mpl(plt, figscale=1.5)
    plot_correct_incorrect_bar(df)    


    # plot_kde(True, 4)



    # plt.show()

    # neuralData = np.random.random([8, 50])
    # plt.eventplot(neuralData, linelengths=0.5)
    # plt.show()
