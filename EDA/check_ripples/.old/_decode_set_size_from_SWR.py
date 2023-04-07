#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-07 08:25:09 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
import utils
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Functions
def train_and_predict(X, y):
    scores = []
    corrs = []
    ys_pred = []
    ys_gt = []
    for _ in range(100):
        X_tra, X_tes, y_tra, y_tes = train_test_split(X, y, test_size=.2)
        X_tra, X_val, y_tra, y_val = train_test_split(X_tra, y_tra, test_size=.25)        
        # create model instance
        bst = XGBClassifier() # n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'
        # bst = XGBRegressor() # n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'
        # fit model
        eval_set = [(X_val, y_val)]
        bst.fit(X_tra, y_tra, eval_set=eval_set, verbose=False)
        # make predictions
        preds = bst.predict(X_tes)
        corr = np.corrcoef(preds, y_tes)[0,1]
        corrs.append(corr)
        # scores.append(corr)
        scores.append(bst.score(X_tes, y_tes))
        ys_gt.append(y_tes)
        ys_pred.append(preds)

    ys_gt = np.hstack(ys_gt)
    ys_pred = np.hstack(ys_pred)
    
    return scores, corrs, ys_gt, ys_pred

def kurtosis(x):
    return scipy.stats.kurtosis(np.hstack(x), axis=-1)

def skewness(x):
    return scipy.stats.skew(np.hstack(x), axis=-1)

# Loads
# rips_df = utils.rips.load_rips()

def main(phase):
    # Preparations
    # phase = "Encoding" # "Encoding"
    rips_df_phase = rips_df[rips_df.phase == phase]

    ns_ss = np.unique(rips_df_phase.set_size, return_counts=True)[1]
    n_min = np.min(ns_ss)

    indi_4 = np.random.permutation(np.where(rips_df_phase.set_size == 4)[0])[:n_min] 
    indi_6 = np.random.permutation(np.where(rips_df_phase.set_size == 6)[0])[:n_min] 
    indi_8 = np.random.permutation(np.where(rips_df_phase.set_size == 8)[0])[:n_min] 
    indi = sorted(np.hstack([indi_4, indi_6, indi_8]))

    # balanced_rips_df = rips_df.iloc[indi]


    # df4 = np.random.permutation(rips_df["ripple band iEEG trace"][rips_df.set_size == 4])[:n_min] 
    # df6 = np.random.permutation(rips_df["ripple band iEEG trace"][rips_df.set_size == 6])[:n_min] 
    # df8 = np.random.permutation(rips_df["ripple band iEEG trace"][rips_df.set_size == 8])[:n_min] 

    # y_4 = [4 for _ in range(len(df4))]
    # y_6 = [6 for _ in range(len(df6))]
    # y_8 = [8 for _ in range(len(df8))]


    # X = np.stack([df4, df6, df8], axis=-1)
    # y = np.array([y_4, y_6, y_8])


    X = rips_df_phase["ripple band iEEG trace"].iloc[indi]
    y = np.array(rips_df_phase["set_size"].replace({4:0, 6:1, 8:2})).astype(int)[indi]


    # X0 = [np.log10(xx.shape[-1]) for xx in X]
    X1 = X.apply(np.mean)
    X2 = X.apply(np.std)
    X3 = X.apply(kurtosis)
    X4 = X.apply(skewness)
    # X5 = X.apply(np.square).apply(np.mean)
    # X6 = X.apply(np.square).apply(np.std)
    # X7 = X.apply(np.square).apply(kurtosis)
    # X8 = X.apply(np.square).apply(skewness)


    X9 = rips_df_phase["subject"].astype(int).iloc[indi]
    # X = np.stack([X1, X2, X3, X4, X5], axis=-1)
    # X = np.stack([X1, X2, X3, X4, X5, X6, X7, X8, X9], axis=-1)
    X = np.stack([X1, X2, X3, X4, X9], axis=-1)
    # X = np.stack([X1, X2, X9], axis=-1)
    # X = np.stack([X1, X2], axis=-1)            

    # scores, corrs = train_and_predict(X, y.astype(float) / 8)
    # scores_surrogate, corrs_surrogate = train_and_predict(X, np.random.permutation(y.astype(float) / 8))
    scores, corrs, ys_gt, ys_pred = train_and_predict(X, y)
    scores_sur, corrs_sur, ys_gt_sur, ys_pred_sur = train_and_predict(X, np.random.permutation(y))

    print(mngs.gen.describe(scores))
    print(mngs.gen.describe(scores_sur))
    print(mngs.gen.describe(corrs))
    print(mngs.gen.describe(corrs_sur))


    # plt.boxplot(np.sqrt((ys_gt-ys_pred)**2))
    # plt.show()

    # plt.scatter(ys_gt, ys_pred)
    # plt.show()


    cm = confusion_matrix(ys_gt, ys_pred)

    mngs.ml.plt.confusion_matrix(plt, cm, labels=["4", "6", "8"])
    plt.show()

if __name__ == "__main__":
    main("Encoding") # 0.39
    main("Fixation")

    main("Maintenance")
    main("Retrieval")    
