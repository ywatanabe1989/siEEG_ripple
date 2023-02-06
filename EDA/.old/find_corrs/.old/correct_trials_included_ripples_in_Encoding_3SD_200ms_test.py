#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-21 11:02:47 (ywatanabe)"

import mngs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re
from natsort import natsorted
import pandas as pd

sd = 4
# def count_n_sessions(sub, session):
#     return len(mngs.io.load(f"./data/Sub_{sub}/Session_{session}/trials_info.csv"))


# Counts number of trials
subs = natsorted(
    [re.findall("Sub_\w{2}", sub_dir)[0][4:] for sub_dir in glob("./data/Sub_??/")]
)
sessions = ["01", "02"]
dfs = []
for sub in subs:
    for session in sessions:
        trials_info = mngs.io.load(
            f"./data/Sub_{sub}/Session_{session}/trials_info.csv"
        )
        trials_info["subject"] = sub
        trials_info["session"] = session
        dfs.append(trials_info)
dfs = pd.concat(dfs).reset_index()
dfs = dfs[["subject", "session", "trial_number"]]
dfs["n_ripples_in_Fixation"] = 0
dfs["n_ripples_in_Encoding"] = 0
dfs["n_ripples_in_Maintenance"] = 0
dfs["n_ripples_in_Retrieval"] = 0
dfs["set_size"] = 0
dfs["n"] = 1
dfs["correct"] = np.nan

# n_trials_df = pd.DataFrame(
#     columns=subs,
#     index=sessions,
#     )
# for sub in n_trials_df.columns:
#     for session in sessions:
#         print(sub, session, count_n_sessions(sub, session))
#         n_trials_df.loc[session, sub] = count_n_sessions(sub, session)

# dd = mngs.general.listed_dict(subs)
# for sub in subs:
#     sessions = natsorted([re.findall(f"Sub_{sub}"+"/Session_\w{2}", session_dir)[0][-2:]
#                           for session_dir in glob(f"./data/Sub_{sub}/Session_??/")])
#     dd[sub] = sessions

# sub = "01"
# session = "01"


rips_df = mngs.io.load("./tmp/rips_df_bi_2.0_SD.csv")
rips_df["Ripple duration [ms]"] = (rips_df["end_time"] - rips_df["start_time"]) * 1000
rips_df["Ripple peak amplitude [SD of baseline]"] = rips_df["ripple_peak_amplitude_sd"]

indi = rips_df["session"] <= 2
indi *= rips_df["Ripple duration [ms]"] <= 200
indi *= rips_df["Ripple peak amplitude [SD of baseline]"] >= sd
rips_df = rips_df[indi]

# from rips_df to dfs
rips_df = rips_df[
    ["subject", "session", "trial number", "correct", "set_size", "phase"]
]
for i_rip, rip in rips_df.iterrows():
    indi = dfs["subject"] == f"{rip['subject']:02d}"
    indi *= dfs["session"] == f"{rip['session']:02d}"
    indi *= dfs["trial_number"] == rip["trial number"]
    assert indi.sum() == 1
    phase = rip["phase"]
    if phase is not np.nan:
        dfs.loc[indi, f"n_ripples_in_{phase}"] += 1
        dfs.loc[indi, "set_size"] = rip["set_size"]
        dfs.loc[indi, "correct"] = rip["correct"]

del dfs["trial_number"]
# sns.boxplot(
#     data=dfs,
#     y="n_ripples_in_Retrieval",
#     x="set_size",
#     hue="correct",
#     )
# plt.show()

dfs_pivot = dfs.pivot_table(columns=["subject", "set_size", "correct"], aggfunc="sum")
mngs.io.save(dfs_pivot, "./tmp/n_ripples.csv")


inci_dfs = dfs_pivot.copy().T
inci_dfs["n_ripples_in_Fixation"] = (
    inci_dfs["n_ripples_in_Fixation"] / inci_dfs["n"] / 1
)
inci_dfs["n_ripples_in_Encoding"] = (
    inci_dfs["n_ripples_in_Encoding"] / inci_dfs["n"] / 2
)
inci_dfs["n_ripples_in_Maintenance"] = (
    inci_dfs["n_ripples_in_Maintenance"] / inci_dfs["n"] / 3
)
inci_dfs["n_ripples_in_Retrieval"] = (
    inci_dfs["n_ripples_in_Retrieval"] / inci_dfs["n"] / 2
)

inci_dfs = inci_dfs.reset_index()
inci_dfs = inci_dfs.rename(
    {
        "n_ripples_in_Fixation": "Ripple incidence in Fixation [Hz]",
        "n_ripples_in_Encoding": "Ripple incidence in Encoding [Hz]",
        "n_ripples_in_Maintenance": "Ripple incidence in Maintenance [Hz]",
        "n_ripples_in_Retrieval": "Ripple incidence in Retrieval [Hz]",
    },
    axis=1,
)

fig, axes = plt.subplots(ncols=4, sharey=True)
phases = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
for ax, phase in zip(axes, phases):
    sns.boxplot(
        data=inci_dfs,
        y=f"Ripple incidence in {phase} [Hz]",
        x="set_size",
        hue="correct",
        ax=ax,
        )
    ax.set_title(phase)
    ax.set_ylabel("")
    ax.legend(loc="upper right")
axes[0].set_ylabel("Ripple incidence [Hz]")
plt.show()

# sns.barplot(data=inci_dfs, y="n_ripples_in_Encoding")
# rips_df["n"] = 1
# rips_df.pivot_table(columns=["subject", "phase", "set_size"], aggfunc="sum")

# sns.boxplot(
#     data=rips_df,
#     x="set_size",
#     hue="phase"
# )

# def plot_correct_incorrect_bar(df):
#     fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True)

#     # count sample sizes
#     df_trials = df[["subject", "session", "trial number", "correct", "set_size"]].drop_duplicates().copy()
#     df_trials["n_trials"] = 1
#     n_trials_df = df_trials.pivot_table(columns=["correct", "set_size"], aggfunc="sum").T.reset_index()\
#         [["correct", "set_size", "n_trials"]]
#     """
#        correct  set_size  n_trials
#     0    False       4.0       2.0
#     1    False       6.0      23.0
#     2    False       8.0      40.0
#     3     True       4.0     323.0
#     4     True       6.0     256.0
#     5     True       8.0     242.0
#     """

#     df["n_ripples"] = 1 # n_ripples
#     _df = df.pivot_table(columns=["phase", "set_size", "correct"], aggfunc="sum").T["n_ripples"].reset_index()

#     # Ripple incidence per trial
#     _df["Ripple incidence [Hz]"] = None # = n
#     for correct in n_trials_df["correct"].unique():
#         for set_size in n_trials_df["set_size"].unique():
#             _df_indi = (_df["correct"] == correct) * (_df["set_size"] == set_size)
#             _trial_indi = (n_trials_df["correct"] == correct) * (n_trials_df["set_size"] == set_size)

#             n_ripples = _df.loc[_df_indi, "n_ripples"]
#             n_trials = int(n_trials_df.loc[_trial_indi, "n_trials"])

#             _df.loc[_df_indi, "Ripple incidence [Hz]"] = n_ripples / n_trials

#     # devide by time [sec]
#     _df.loc[_df["phase"] == "Fixation", "Ripple incidence [Hz]"] /= 1
#     _df.loc[_df["phase"] == "Encoding", "Ripple incidence [Hz]"] /= 2
#     _df.loc[_df["phase"] == "Maintenance", "Ripple incidence [Hz]"] /= 3
#     _df.loc[_df["phase"] == "Retrieval", "Ripple incidence [Hz]"] /= 2

#     for i_correct, is_correct in enumerate([True, False]):

#         ax = axes[i_correct]
#         is_correct_str = "correct" if is_correct else "incorrect"
#         _df_correct_or_incorrect = _df[_df["correct"] == is_correct]

#         for ss in [4, 6, 8]:
#             n_trials = \
#                 n_trials_df[(n_trials_df["correct"] == is_correct) * (n_trials_df["set_size"] == ss)]["n_trials"]

#             ii = (_df_correct_or_incorrect["correct"] == is_correct) * \
#                  (_df_correct_or_incorrect["set_size"] == ss)

#             _df_correct_or_incorrect.loc[ii, "set_size", ] = f"{ss} (n={int(n_trials.iloc[0])})"

#         sns.barplot(
#             data=_df_correct_or_incorrect,
#             x="phase",
#             y="Ripple incidence [Hz]",
#             hue="set_size",
#             order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
#             ax=ax
#             )

#         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

#         ax.set_title(is_correct_str)
#         ax.set_ylabel("Ripple incidence [Hz]")
#         # ax.set_yscale("log")

#         extend_ratio = 0.5
#         ax = mngs.plt.ax_extend(ax, x_extend_ratio=extend_ratio, y_extend_ratio=extend_ratio)


# # for phase in ["Fixation", "Encoding", "Maintenance", "Retrieval"]:
# #     indi2 = indi * rips_df["phase"] == phase

# #     fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True)

# #     for i_ax, ax in enumerate(axes):
# #         is_correct = bool(i_ax)
# #         indi3 = indi2 * (rips_df["correct"] == is_correct)

# #         sns.scatterplot(
# #             data=rips_df[indi3],
# #             x="Ripple duration [ms]",
# #             y="Ripple peak amplitude [SD of baseline]",
# #             # y='ripple_amplitude_sd',
# #             hue="set_size",
# #             # hue_order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
# #             alpha=.3,
# #             ax=ax,
# #             )
# #         correct_str = "Correct" if is_correct else "Incorrect"
# #         ax.set_title(correct_str)
# #         fig.suptitle(phase)
# #     mngs.io.save(fig, f"./tmp/correct_trials_included_ripples_in_{phase}.png")
