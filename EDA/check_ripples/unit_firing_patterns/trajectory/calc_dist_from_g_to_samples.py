#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-26 12:49:52 (ywatanabe)"

import matplotlib

matplotlib.use("TkAgg")

import mngs
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.utils import shuffle


def to_X_T(df, phases):
    X = []
    T = []
    for phase in phases:
        df_p = df[[f"{phase}", f"{phase}.1", f"{phase}.2"]]
        df_p = df_p[~df_p.isna().any(axis=1)]
        # mid = len(df_p) // 2
        # df_p = df_p.iloc[mid - 10 : mid + 10]
        X.append(np.array(df_p))
        T.append(np.full(len(df_p), phase))
    return np.vstack(X), np.hstack(T)


def evaluate_model(
    X, T, cv, model=None, shuffle=False, n_repeat=1, under_sample=False
):
    if model is None:
        model = SVC(class_weight="balanced", kernel="linear")
    rus = RandomUnderSampler(random_state=42)

    all_scores = []
    for _ in range(n_repeat):
        if shuffle:
            T = np.random.permutation(T)

        for train_index, test_index in cv.split(X, T):
            X_train, X_test = X[train_index], X[test_index]
            T_train, T_test = T[train_index], T[test_index]

            if under_sample:
                X_train, T_train = rus.fit_resample(X_train, T_train)

            model.fit(X_train, T_train)
            T_pred = model.predict(X_test)

            score = balanced_accuracy_score(T_test, T_pred)
            all_scores.append(score)
    return np.hstack(all_scores)


def load_dfs():
    def _load_dfs():
        dfs = {}
        for set_size in [4, 6, 8]:
            for match in [1, 2]:

                lpath = (
                    f"./res/figs/scatter/repr_traj/session_traj_Subject_06_Session_02_"
                    f"set_size_{set_size}_match_{match}.csv"
                )

                df = mngs.io.load(lpath)
                del df["Unnamed: 0"]

                dfs[lpath] = df
        return dfs

    dfs = _load_dfs()
    dfs["all"] = pd.concat([v for k, v in dfs.items()])
    return dfs


def df_to_gs(df):
    gs = df.iloc[0, :12]

    gs = pd.DataFrame(
        data=np.array(gs).reshape(-1, 3),
        columns=["x", "y", "z"],
        index=PHASES,
    )
    return gs


def df_to_NTs(df):
    NTs = df.iloc[:, 12:]

    out = {}

    for phase in PHASES:
        NTs_P = NTs[mngs.gen.search(phase, NTs.columns)[1]]
        NTs_P = NTs_P[~NTs_P.isna().any(axis=1)]
        out[phase] = NTs_P

    return out


def calc_dist(NTs, gs):
    # for broadcasting
    NTs = np.vstack([v for k, v in NTs.items()])
    NTs = NTs[np.newaxis, :]
    gs = np.array(gs)[:, np.newaxis, :]

    # extract the mid 20 bins for each phase
    NTs_mid = []
    for ss, ee in PHASES_BINS_DICT.values():
        mm = (ss + ee) // 2
        NTs_mid.append(NTs[:, mm - 10 : mm + 10, :])
    NTs_mid = np.vstack(NTs_mid)

    # for broadcasting
    # dim0 -> g phase, dim1 -> NT phase
    NTs = NTs_mid[np.newaxis]
    gs = gs[:, np.newaxis]

    diff = NTs - gs
    squared_diff = diff**2
    squared_dist = np.sum(squared_diff, axis=-1)
    dist = np.sqrt(squared_dist)

    # to df
    out = {}
    for i_g_phase, g_phase in enumerate(PHASES):
        for i_NT_phase, NT_phase in enumerate(PHASES):
            out[f"g_{g_phase[0]}_NT_{NT_phase[0]}"] = dist[
                i_g_phase, i_NT_phase
            ]
    out = pd.DataFrame(out)

    return out


def main(phases, under_sample, n_splits, n_surrogate):

    for lpath, df in dfs.items():

        # if lpath != "all":
        #     continue

        X, T = to_X_T(df, phases)

        # Model
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Evaluation
        real_scores = evaluate_model(
            X,
            T,
            skf,
            shuffle=False,
            n_repeat=1,
            under_sample=under_sample,
        )
        fake_scores = evaluate_model(
            X,
            T,
            skf,
            shuffle=True,
            n_repeat=n_surrogate,
            under_sample=under_sample,
        )

        # Brunner-Munzel Test
        w, p_value, dof, eff = mngs.stats.brunner_munzel_test(
            real_scores, fake_scores
        )

        print()
        print(lpath)
        print(phases)
        print(under_sample)
        print(n_splits)
        print(n_surrogate)
        print(mngs.gen.describe(real_scores))
        print(mngs.gen.describe(fake_scores))
        print(p_value)
        print(eff)
        print()


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    import utils

    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()
    # N_SPLITS = 10
    # # N_SURROGATE = 500
    # N_SURROGATE = 100
    # UNDER_SAMPLE = True
    # phases = PHASES
    # phases = ["Encoding", "Retrieval"]

    dfs = load_dfs()

    # lpath = list(dfs.keys())[0]
    # df = dfs[lpath]
    df = dfs["all"]

    gs = df_to_gs(df)
    NTs = df_to_NTs(df)
    dist = calc_dist(NTs, gs)

    # fig, ax = mngs.plt.subplots()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()

    # dists = mngs.gen.force_dataframe(dists)
    # dists = dists.melt()

    dist = dist.melt()
    dist["value"] = pd.to_numeric(dist["value"])

    sns.boxplot(
        data=dist,
        x="variable",
        y="value",
        ax=ax,
    )

    plt.show()
