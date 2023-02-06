#/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-04 13:48:47 (ywatanabe)"

import sys

sys.path.append(".")
# from siEEG_ripple import utils
import utils

import mngs

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def plot(yy):
    sns.boxplot(
        data=rips_df_E,
        x="set_size",
        y=yy,
        showfliers=False,
    )
    plt.show()


def bm(rips_df, yy):
    for ss1, ss2 in ([4, 6], [6, 8], [4, 8]):
        w, p, dof, effsize = mngs.stats.brunner_munzel_test(
            rips_df[rips_df.set_size == ss1][yy],
            rips_df[rips_df.set_size == ss2][yy],
        )
        print(ss1, ss2, p.round(3))


# Loads
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
rips_df = utils.rips.load_rips()
rips_df_E = rips_df[rips_df.phase == "Retrieval"]
rips_df_E.pivot_table(columns=["set_size"], aggfunc="sum")
rips_df_E["log10(duration_ms)"]

yy = "log10(duration_ms)"
yy = "log10(ripple_peak_amplitude_sd)"
yy = "start_time"

from time import sleep

for i_comb, (yy1, yy2) in enumerate(
    combinations(
        [
            # "start_time",
            # "center_time",
            # "end_time",
            "ripple_peak_amplitude_sd",
            "ripple_amplitude_sd",
            "n_firings",
            "population_burst_rate",
            "unit_participation_rate",
            "log10(duration_ms)",
            "log10(ripple_peak_amplitude_sd)",
            "IO_balance",
        ],
        2,
    )
):
    yy1 = "ripple_peak_amplitude_sd"
    yy2 = "log10(ripple_peak_amplitude_sd)"
    
    print("----------------------------------------")
    print(i_comb, yy1, yy2)
    rips_df["yy1/yy2"] = rips_df[yy1] / rips_df[yy2]

    for phase in PHASES:
        print()
        print(phase, yy1, yy2)
        rips_df_phase = rips_df[rips_df.phase == phase]
        bm(rips_df_phase, "yy1/yy2")
        print()
    print("----------------------------------------")
    # sleep(10)

    fig, ax = plt.subplots()
    sns.boxplot(
        data=rips_df,
        y="yy1/yy2",
        x="phase",
        hue="set_size",
        order=PHASES,
        showfliers=False,
        )
    ax.set_title(yy)


df = rips_df.copy()
df = df[(df.phase == "Encoding") + (df.phase == "Retrieval")]
df["X"] = df["log10(ripple_peak_amplitude_sd)"] / df["log10(duration_ms)"]
for sub in df.subject.unique():
    df_sub = df[df.subject == sub]
    fig, ax = plt.subplots()
    sns.boxplot(
        data=df_sub,
        x="phase",
        y="X",
        hue="set_size",
        ax=ax,
        )
    ax.set_title(sub)
plt.show()

yy = "log10(duration_ms)"
sns.boxplot(
    data=df, # [df.match == 2],
    y=yy,
    x="set_size",
    hue="phase"
    )
plt.show()

yy = "IO_balance"
data = df[df.phase == "Encoding"]
w,p,d, e=mngs.stats.brunner_munzel_test(
    data[yy][data.set_size == 4],
    data[yy][data.set_size == 8],
    )
print(p)
    

# df = rips_df
import numpy as np
def get_corrs(shuffle=False):
    _rips_df = rips_df.copy()

    if shuffle:
        _rips_df.set_size = np.random.permutation(_rips_df.set_size)
    
    df = _rips_df.pivot_table(columns=["subject", "phase", "set_size"]).T\
        [["log10(duration_ms)", "log10(ripple_peak_amplitude_sd)"]].reset_index()
    # df = rips_df
    df = df[(df.phase == "Encoding") + (df.phase == "Retrieval")]
    df["X"] = df["log10(ripple_peak_amplitude_sd)"] / df["log10(duration_ms)"]

    
    corrs = []
    for sub in df.subject.unique():
        try:
            df_sub = df[df.subject == sub]
            corr = np.corrcoef(df_sub[df_sub.phase == "Encoding"]["X"],
                               df_sub[df_sub.phase == "Retrieval"]["X"])[0,1]
            corrs.append(corr)
        except Exception as e:
            print(e)
            corrs.append(np.nan)

    return np.array(corrs)

corrs_obs = get_corrs(shuffle=False)
corrs_sim = np.array([get_corrs(shuffle=True) for _ in range(1000)])
plt.hist(corrs_sim[:,1])
plt.show()


(corrs_obs).mean()
mngs.gen.describe(corrs_sim.mean(axis=-1))
plt.hist(np.abs(corrs_sim).mean(axis=-1))
plt.show()

mngs.gen.describe(np.abs(corrs))

corrs_sim = np.array([2*np.random.rand(5)-1 for _ in range(1000)])
plt.boxplot(np.abs(corrs_sim).mean(axis=-1))
plt.show()

plt.boxplot(sim)
corrs_sim = np.array([mngs.gen.describe(2*np.random.rand(5)-1) for _ in range(1000)])
np.array



df["sub_phase"] = mngs.ml.utils.merge_labels(df["subject"], df["phase"])
sns.boxplot(
    data=df,
    x="sub_phase",
    y="X",
    hue="set_size",
    )


df["log10(ripple_peak_amplitude_sd)/log10(duration_ms)"]
df = df.reset_index()
columns = ["subject", "phase", "set_size", "log10(ripple_peak_amplitude_sd)/log10(duration_ms)"]
df[(df.phase == "Encoding") + (df.phase == "Retrieval")][columns]

phase = "Encoding"
sns.scatterplot(
    data=df[(df.phase == "Encoding") + (df.phase == "Retrieval")],
    # data=df[(df.phase == phase)],
    hue="log10(ripple_peak_amplitude_sd)/log10(duration_ms)",
    x=,
    hue="subject",
    )
plt.show()

data = df[(df.phase == "Encoding") + (df.phase == "Retrieval")],
plt.scatterplot(
    x=data["phase"] == "Encoding",
    y=data["phase"] == "Retrieval",
    
    
