#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-11 21:21:03 (ywatanabe)"

import mngs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re
from natsort import natsorted
import pandas as pd
import scipy
from itertools import combinations


def main(correct_rate_or_response_time):
    df = mngs.gen.force_dataframe(
        {
            f"set_size_{set_size}_match_{match}": dfs[
                (dfs.set_size == set_size) * (dfs.match == match)
            ][correct_rate_or_response_time]
            for set_size in [4, 6, 8]
            for match in [1, 2]
        }
    ).replace({"": np.nan})

    # Kruskal Wallis
    data = [np.array(df[col]) for col in df.columns]
    data = [dd[~np.isnan(dd)] for dd in data]
    print(scipy.stats.kruskal(*data))

    # Pairwise Brunner-munzel
    n_pairs = len(list(combinations(df.columns, 2)))
    for col1, col2 in combinations(df.columns, 2):
        print(col1, col2)
        w, p, dof, eff = mngs.stats.brunner_munzel_test(df[col1], df[col2])
        print(round(p * n_pairs, 3))
        print()
    return df


# def calc_mean_and_std_correct_rate(dfs, match, set_size):
#     is_correct = dfs[(dfs["match"] == match) * (dfs["set_size"] == set_size)]["correct"]
#     return is_correct.mean(), is_correct.std()


# def calc_mean_and_std_response_time(dfs, match, set_size):
#     is_correct = dfs[(dfs["match"] == match) * (dfs["set_size"] == set_size)][
#         "response_time"
#     ]
#     return is_correct.mean(), is_correct.std()


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


def add_task_difficulty(df):
    count = 0
    df_corr = pd.DataFrame()
    for set_size in [4, 6, 8]:
        for match in [1, 2]:
            count += 1
            data = df[f"set_size_{set_size}_match_{match}"]
            data = data[~data.isna()]
            _df_corr = pd.DataFrame({"data": data})
            _df_corr["task_difficulty"] = count
            df_corr = pd.concat([df_corr, _df_corr])
    return df_corr


def calc_corr(df_with_task_difficulty):
    """
    df_with_task_difficulty = df_correct_with_task_difficulty.copy()
    """
    corr_obs = np.corrcoef(
        df_with_task_difficulty["data"], df_with_task_difficulty["task_difficulty"]
    )[0, 1]
    corrs_shuffle = []
    for _ in range(1000):
        data = df_with_task_difficulty["data"]
        shuffled_task_difficulty = np.random.permutation(
            df_with_task_difficulty["task_difficulty"]
        )
        corr_shuffle = np.corrcoef(data, shuffled_task_difficulty)[0, 1]
        corrs_shuffle.append(corr_shuffle)
    return corr_obs, corrs_shuffle


################################################################################
## correct rate


df_correct = main("correct")
df_response_time = main("response_time")

mngs.io.save(df_correct, "./tmp/figs/task_difficulty/correct_rate.csv")
mngs.io.save(df_response_time, "./tmp/figs/task_difficulty/response_time.csv")

df_correct_with_task_difficulty = add_task_difficulty(df_correct)
df_response_time_with_task_difficulty = add_task_difficulty(df_response_time)

corr_correct, corr_shuffled_correct = calc_corr(df_correct_with_task_difficulty)
corr_response_time, corr_shuffled_response_time = calc_corr(
    df_response_time_with_task_difficulty
)

from bisect import bisect_right

rank_rate_correct = bisect_right(sorted(corr_shuffled_correct), corr_correct) / len(
    corr_shuffled_correct
)
rank_rate_response_time = bisect_right(
    sorted(corr_shuffled_response_time), corr_response_time
) / len(corr_shuffled_response_time)

import seaborn as sns
df_correct_rate = pd.DataFrame({
    "x": ["correct_rate" for _ in range(len(corr_shuffled_correct))],
    "correlation": corr_shuffled_correct,
})
df_response_time = pd.DataFrame({
    "x": ["response_time" for _ in range(len(corr_shuffled_response_time))],
    "correlation": corr_shuffled_response_time,
})
df = pd.concat([df_correct_rate, df_response_time])
df = add_hue(df)

# koko
fig, ax = plt.subplots()
sns.violinplot(data=df,
               x="x",
               y="correlation",
               ax=ax,
               hue="hue",
               split=True,
               color="gray",
               width=0.08,
               )
ax.scatter(
    x="correct_rate",
    y=corr_correct,
    color="red",
    s=100,
    )
ax.scatter(
    x="response_time",
    y=corr_response_time,
    color="red",
    s=100,    
    )
ax.set_ylim(-.25,.25)

mngs.io.save(fig, "./tmp/figs/task_difficulty/task_difficulty.tif")
plt.show()
# df_response_time = mngs.gen.force_dataframe(
#     {
#         f"set_size_{set_size}_match_{match}": dfs[
#             (dfs.set_size == set_size) * (dfs.match == match)
#         ]["response_time"]
#         for set_size in [4, 6, 8]
#         for match in [1, 2]
#     }
# ).replace({"":np.nan})


# df_correct_rate = (
#     dfs[["match", "set_size"]]
#     .drop_duplicates()
#     .sort_values(["match", "set_size"])
#     .reset_index()
# )
# del df_correct_rate["index"]

# df_correct_rate["mean"], df_correct_rate["std"] = np.nan, np.nan
# for i_row, row in df_correct_rate.iterrows():
#     match = row["match"]
#     set_size = row["set_size"]

#     mean, std = calc_mean_and_std_correct_rate(dfs, match, set_size)

#     df_correct_rate.loc[
#         (df_correct_rate["match"] == match) * (df_correct_rate["set_size"] == set_size),
#         "mean",
#     ] = mean
#     df_correct_rate.loc[
#         (df_correct_rate["match"] == match) * (df_correct_rate["set_size"] == set_size),
#         "std",
#     ] = std


# sns.barplot(
#     data=dfs,
#     x="set_size",
#     y="correct",
#     hue="match",
# )
# mngs.io.save(plt, "./tmp/figs/task_difficulty/correct_rate.png")
# mngs.io.save(df_correct_rate, "./tmp/figs/task_difficulty/correct_rate.csv")
# plt.close()
# # plt.show()
# ################################################################################

# df_response_time = (
#     dfs[["match", "set_size"]]
#     .drop_duplicates()
#     .sort_values(["match", "set_size"])
#     .reset_index()
# )
# del df_response_time["index"]
# df_response_time["mean"], df_response_time["std"] = np.nan, np.nan
# for i_row, row in df_response_time.iterrows():
#     match = row["match"]
#     set_size = row["set_size"]

#     mean, std = calc_mean_and_std_response_time(dfs, match, set_size)

#     df_response_time.loc[
#         (df_response_time["match"] == match)
#         * (df_response_time["set_size"] == set_size),
#         "mean",
#     ] = mean
#     df_response_time.loc[
#         (df_response_time["match"] == match)
#         * (df_response_time["set_size"] == set_size),
#         "std",
#     ] = std


# sns.barplot(
#     data=dfs,
#     x="set_size",
#     y="response_time",
#     hue="match",
# )

# mngs.io.save(plt, "./tmp/figs/task_difficulty/response_time.png")
# mngs.io.save(df_response_time, "./tmp/figs/task_difficulty/response_time.csv")
# plt.close()


# ## EOF
