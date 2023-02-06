#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-25 20:10:31 (ywatanabe)"

import mngs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re
from natsort import natsorted
import pandas as pd

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
dfs = pd.concat(dfs).reset_index()[
    ["subject", "session", "set_size", "match", "correct", "response_time"]
]
# dfs["n"] = 1


################################################################################
## correct rate
def calc_mean_and_std_correct_rate(dfs, match, set_size):
    is_correct = dfs[(dfs["match"] == match) * (dfs["set_size"] == set_size)]["correct"]
    return is_correct.mean(), is_correct.std()


df_correct_rate = (
    dfs[["match", "set_size"]]
    .drop_duplicates()
    .sort_values(["match", "set_size"])
    .reset_index()
)
del df_correct_rate["index"]

df_correct_rate["mean"], df_correct_rate["std"] = np.nan, np.nan
for i_row, row in df_correct_rate.iterrows():
    match = row["match"]
    set_size = row["setx_size"]

    mean, std = calc_mean_and_std_correct_rate(dfs, match, set_size)

    df_correct_rate.loc[
        (df_correct_rate["match"] == match) * (df_correct_rate["set_size"] == set_size),
        "mean",
    ] = mean
    df_correct_rate.loc[
        (df_correct_rate["match"] == match) * (df_correct_rate["set_size"] == set_size), "std"
    ] = std


sns.barplot(
    data=dfs,
    x="set_size",
    y="correct",
    hue="match",
)
mngs.io.save(plt, "./tmp/figs/task_difficulty/correct_rate.png")
mngs.io.save(df_correct_rate, "./tmp/figs/task_difficulty/correct_rate.csv")
plt.close()
# plt.show()
################################################################################
def calc_mean_and_std_response_time(dfs, match, set_size):
    is_correct = dfs[(dfs["match"] == match) * (dfs["set_size"] == set_size)]["response_time"]
    return is_correct.mean(), is_correct.std()

df_response_time = (
    dfs[["match", "set_size"]]
    .drop_duplicates()
    .sort_values(["match", "set_size"])
    .reset_index()
)
del df_response_time["index"]
df_response_time["mean"], df_response_time["std"] = np.nan, np.nan
for i_row, row in df_response_time.iterrows():
    match = row["match"]
    set_size = row["set_size"]

    mean, std = calc_mean_and_std_response_time(dfs, match, set_size)

    df_response_time.loc[
        (df_response_time["match"] == match) * (df_response_time["set_size"] == set_size),
        "mean",
    ] = mean
    df_response_time.loc[
        (df_response_time["match"] == match) * (df_response_time["set_size"] == set_size), "std"
    ] = std



sns.barplot(
    data=dfs,
    x="set_size",
    y="response_time",
    hue="match",
)
    
mngs.io.save(plt, "./tmp/figs/task_difficulty/response_time.png")
mngs.io.save(df_response_time, "./tmp/figs/task_difficulty/response_time.csv")
plt.close()






## EOF
