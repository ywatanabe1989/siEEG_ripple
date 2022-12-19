#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-06 15:08:14 (ywatanabe)"

import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(".")

import utils

# Funcations
BANDS_LIM_HZ_DICT = {
    # "delta": [0.5, 4],
    # "theta": [4, 8],
    "0-7 Hz": [0, 7],
    "7-12 Hz": [7, 12],        
    # "lalpha": [8, 10],
    # "halpha": [10, 13],
    # "beta": [13, 32],
    # "gamma": [32, 75],
    "80-140 Hz": [80, 140],
}
def calc_fft_powers(iEEG):
    elong_sig = np.zeros([len(iEEG), 400])
    elonged_iEEG = np.hstack([elong_sig, iEEG, elong_sig])
    iEEG_arr = np.array(elonged_iEEG)
    amps = mngs.dsp.calc_fft_powers(iEEG_arr, samp_rate=1000)
    amps = amps.mean()
    freqs = amps.index.astype(float)
    amps_band = {}
    for band_str, (low, high) in BANDS_LIM_HZ_DICT.items():
        band_mean = np.nanmean(amps[(low <= freqs) * (freqs < high)])
        amps_band[band_str] = band_mean
    return pd.Series(amps_band)


# Parameters
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]

# Loads
rips_df = utils.load_rips()
rips_df = rips_df[(rips_df.IoU <= IoU_RIPPLE_THRES) * rips_df.session.astype(int) <= SESSION_THRES]
rips_df = rips_df[rips_df.phase == "Encoding"]

# FFT
iEEG_FFT_powers = rips_df["iEEG trace"].apply(calc_fft_powers).astype(float) # rips_df["iEEG FFT powers"]

features = ["subject", "n_firings", "ripple_peak_amplitude_sd", "duration_ms"] # "n_firings",
categorical_features_names = ["subject"]
rips_df[categorical_features_names].nunique()
# out_X, out_y = [], []
# for sub in rips_df.subject.unique():
#     rips_df_sub = rips_df[rips_df.subject == sub]
#     rips_df_sub_X = rips_df_sub[features]
#     rips_df_sub_y = rips_df_sub["set_size"]    
#     rips_df_sub_X = (rips_df_sub_X - rips_df_sub_X.mean(axis=0)) / rips_df_sub_X.std(axis=0)
#     out_X.append(rips_df_sub_X)
#     out_y.append(rips_df_sub_y)    
#     # rips_df[features][rips_df.subject == sub] = rips_df_sub

# rips_df_z = pd.concat(out_X)
# rips_df_z["set_size"] = pd.concat(out_y)#["set_size"]
rips_df_z = rips_df

# trace = rips_df["ripple band iEEG trace"].iloc[0]
# elong_sig = np.zeros([len(trace), 400])
# trace = np.hstack([elong_sig, trace, elong_sig])
# mngs.dsp.




# features = ["n_firings", "ripple_peak_amplitude_sd", "duration_ms", "center_time", "ripple_amplitude_sd"] # "n_firings",
# features = ["n_firings", "ripple_peak_amplitude_sd", "duration_ms"]# ["n_firings"] #, "duration_ms", "ripple_peak_amplitude_sd", "ripple_amplitude_sd"] # "n_firings",
X = pd.concat([rips_df_z[features], iEEG_FFT_powers], axis=1)

# rips_df_z["ripple band iEEG trace"]
y = rips_df_z["set_size"]

n_min = np.unique(y, return_counts=True)[1].min()



# indi_4 = y.iloc[np.where(y == 4)[0]]
indi_4 = np.random.permutation(np.where(y == 4)[0])[:n_min] 
indi_6 = np.random.permutation(np.where(y == 6)[0])[:n_min] 
indi_8 = np.random.permutation(np.where(y == 8)[0])[:n_min] 

y_balanced = y.iloc[list(indi_4) + list(indi_6) + list(indi_8)]
# (y_balanced == 4).sum()
# (y_balanced == 6).sum()
# (y_balanced == 8).sum()
X_balanced = X.iloc[list(indi_4) + list(indi_6) + list(indi_8)]
# indi_6 = np.random.permutation((y == 6).index)[:n_min]
# indi_8 = np.random.permutation((y == 8).index)[:n_min]

# mngs.general.search(list(y.index.astype(str)), list(indi_4.astype(str)), as_bool=True)[0]

# np.where(y.index == indi_4)
# y[indi_4]

mngs.general.fix_seeds(np=np)
bACCs = []
feature_importances = mngs.general.listed_dict()
y_tes_all, pred_tes_all = [], []
for _ in range(30):
    # X.isna()
    _X_tra, X_tes, _y_tra, y_tes = train_test_split(X_balanced, y_balanced, test_size=0.2)
    # X_tra, X_val, y_tra, y_val = train_test_split(_X_tra, _y_tra, test_size=0.2)
    X_tra, y_tra = _X_tra, _y_tra

    clf = CatBoostClassifier(
        verbose=False, allow_writing_files=False,
    )  # task_type="GPU" does not work

    cb_pool_tra = Pool(X_tra, label=y_tra, cat_features=categorical_features_names)
    # cb_pool_val = Pool(X_val, label=y_val, cat_features=categorical_features_names)
    cb_pool_tes = Pool(X_tes, label=y_tes, cat_features=categorical_features_names)

    ## Training
    # clf.fit(cb_pool_tra, eval_set=cb_pool_val, plot=False, verbose=False)
    clf.fit(cb_pool_tra, plot=False, verbose=False)    

    ## Prediction
    true_class_tes = np.array(y_tes)
    pred_proba_tes = clf.predict_proba(cb_pool_tes)
    pred_class_tes = np.argmax(pred_proba_tes, axis=1)

    # bACC_tra = balanced_accuracy_score(y_tra.squeeze(), clf.predict(X_tra))
    bACC_tes = balanced_accuracy_score(y_tes.squeeze(), clf.predict(X_tes))
    bACCs.append(bACC_tes)
    print(bACC_tes.round(3))

    y_tes_all.append(y_tes)
    pred_tes_all.append(clf.predict(X_tes))

    z_fis = (clf.feature_importances_ - clf.feature_importances_.mean()) / clf.feature_importances_.std()
    for fn, fi in zip(clf.feature_names_, z_fis):
        feature_importances[fn].append(fi)
        
    # print(clf.feature_importances_) # 30.4, 37.7, 31.9

print(np.mean(bACCs).round(3), np.std(bACCs).round(3))


feature_importances = pd.DataFrame(feature_importances)
fig, ax = plt.subplots(figsize=(6.4, 6.4))
feature_importances.boxplot(ax=ax)
ax.set_ylabel("Feature importance")
plt.xticks(rotation=45)
mngs.io.save(fig, "./tmp/figs/B/feature_importances.png")
# plt.show()

import seaborn as sns


cm = confusion_matrix(np.hstack(y_tes_all), np.vstack(pred_tes_all).squeeze())
# fig = sns.heatmap(cm)
fig = mngs.ml.plt.confusion_matrix(plt, cm, labels=["4", "6", "8"])
mngs.io.save(fig, "./tmp/figs/B/confusion_matrix.png")
plt.show()

# (np.hstack(y_tes_all) == 4).sum()
# (np.hstack(y_tes_all) == 6).sum()
# (np.hstack(y_tes_all) == 8).sum()
