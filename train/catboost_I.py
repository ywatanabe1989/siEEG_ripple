#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-06 17:07:55 (ywatanabe)"

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.svm import SVR
import torch
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(".")
from eeg_human_ripple_clf.utils._DataLoaderFiller import DataLoaderFiller
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns
import mngs

# Fixes random seed
mngs.general.fix_seeds(seed=42, np=np, torch=torch)


# Configures matplotlib
mngs.plt.configure_mpl(plt)


def train_and_predict_cb(dlf):
    """
    Using FFT power as features,
        1) Trains catboost
        2) Predict the classe of each EEG segment
    """

    ## Get EEG, Target (= Diagnosis labels)
    X_tra, _, T_tra = dlf.dl_tra.dataset.tensors
    X_val, _, T_val = dlf.dl_val.dataset.tensors
    X_tes, _, T_tes = dlf.dl_tes.dataset.tensors

    SAMP_RATE_EEG = 200
    seq_len_sec = round(X_tra.shape[-1] / SAMP_RATE_EEG, 3)

    # counts = dlf.sample_counts
    # neg_weight = counts[1] / counts.sum()
    # pos_weight = counts[0] / counts.sum()
    # counts = {i_class:count for i_class, count in enumerate(dlf.sample_counts)}
    # from sklearn.utils.class_weight import compute_class_weight

    # compute_class_weight("balanced", classes=[0, 1], y=counts)

    # np.unique(S_tra, return_counts=True)
    SAMP_RATE_EEG = 200
    X_tra = mngs.dsp.feature_extractions.rfft_bands(
        X_tra, SAMP_RATE_EEG, normalize=True
    )
    X_val = mngs.dsp.feature_extractions.rfft_bands(
        X_val, SAMP_RATE_EEG, normalize=True
    )
    X_tes = mngs.dsp.feature_extractions.rfft_bands(
        X_tes, SAMP_RATE_EEG, normalize=True
    )


    # ## From EEG to FFT powers
    # max_hz = 58
    # X_tra = abs(np.fft.rfft(X_tra, norm="ortho")[:, :, :max_hz])
    # X_val = abs(np.fft.rfft(X_val, norm="ortho")[:, :, :max_hz])
    # X_tes = abs(np.fft.rfft(X_tes, norm="ortho")[:, :, :max_hz])

    # Reshape the inputs
    X_tra = X_tra.reshape(len(X_tra), -1)
    X_val = X_val.reshape(len(X_val), -1)
    X_tes = X_tes.reshape(len(X_tes), -1)


    # z-norm
    X_tra = (X_tra - X_tra.mean(axis=1, keepdims=True)) / X_tra.std(
        axis=1, keepdims=True
    ) / 2
    X_val = (X_val - X_val.mean(axis=1, keepdims=True)) / X_val.std(
        axis=1, keepdims=True
    ) / 2
    X_tes = (X_tes - X_tes.mean(axis=1, keepdims=True)) / X_tes.std(
        axis=1, keepdims=True
    ) / 2

    # T_tra = (T_tra - T_tra.mean(axis=0, keepdims=True)) / T_tra.std(
    #     axis=0, keepdims=True
    # ) / 2
    # T_val = (T_val - T_val.mean(axis=0, keepdims=True)) / T_val.std(
    #     axis=0, keepdims=True
    # ) / 2
    # T_tes = (T_tes - T_tes.mean(axis=0, keepdims=True)) / T_tes.std(
    #     axis=0, keepdims=True
    # ) / 2
    
    # to numpy
    X_tra = np.array(X_tra)
    X_val = np.array(X_val)
    X_tes = np.array(X_tes)
    T_tra = np.array(T_tra)
    T_val = np.array(T_val)
    T_tes = np.array(T_tes)

    # merge training and validation data
    X_tra = np.concatenate([X_tra, X_val])
    T_tra = np.concatenate([T_tra, T_val])

    # z-norm
    X_tra = (X_tra - X_tra.mean(axis=0, keepdims=True)) / X_tra.std(
        axis=0, keepdims=True
    )
    # X_val = (X_val - X_val.mean(axis=0, keepdims=True)) / X_val.std(axis=0, keepdims=True)
    X_tes = (X_tes - X_tes.mean(axis=0, keepdims=True)) / X_tes.std(
        axis=0, keepdims=True
    )

    # training
    model = SVR(kernel="rbf")#, C=2.0, epsilon=0.1)
    model.fit(X_tra, T_tra)

    # prediction
    y_pred_tes = model.predict(X_tes)
    # ################################################################################
    # ## catboost
    # ## Model
    # cbr = CatBoostRegressor(
    #     verbose=False, allow_writing_files=False
    # )  # task_type="GPU" does not work
    # #, class_weights=[neg_weight, pos_weight]

    # cb_pool_tra = Pool(X_tra, label=T_tra)
    # cb_pool_val = Pool(X_val, label=T_val)
    # cb_pool_tes = Pool(X_tes, label=T_tes)

    # ## Training
    # cbr.fit(cb_pool_tra, eval_set=cb_pool_val, plot=True, verbose=True)#, plot=False)#, verbose=False)

    # # learning curve
    # plt.plot(cbr.evals_result_["validation"]["RMSE"])
    # plt.show()

    # y_pred_tes = cbr.predict(cb_pool_tes)
    # ################################################################################

    mngs.plt.configure_mpl(plt, figscale=1.5)
    
    fig, ax = plt.subplots()
    x = T_tes
    y = y_pred_tes
    m, b = np.polyfit(x, y, 1)
    
    ax.plot(x, m * x + b)
    ax.scatter(x, y, alpha=0.3)
    # sns.regplot(x=x, y=y, ax=ax, alpha=0.3) #, ci=None)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect(1)


    r = np.corrcoef(x, y)[0,1]
    ax.set_title(f"y = {m:.3f} * x + {b:.3f}; r = {r:.3f}")
    
    ax.set_xlabel(f"True ripple incidence of {seq_len_sec}-sec segments [Hz]")
    ax.set_ylabel("Prediction")

    mngs.io.save(fig, f"./tmp/ripple_inci_hz_regression_with_SVM_from_{ws_ms/1000:.1f}-s_segments.png")
    # plt.show()
    
    # import ipdb

    # ipdb.set_trace()

    # m, b = np.polyfit(T_tes, y_pred_tes, 1)

    # plt.plot(T_tes, m*T_tes+b)
    # plt.scatter(T_tes, y_pred_tes)
    # plt.xlim(0,5)
    # plt.ylim(0,5)


    # ## Prediction
    # true_class_tes = np.array(T_tes)
    # pred_proba_tes = cbr.predict_proba(cb_pool_tes)
    # pred_class_tes = np.argmax(pred_proba_tes, axis=1)

    # bACC_tra = balanced_accuracy_score(T_tra.squeeze(), cbr.predict(X_tra))
    # bACC_val = balanced_accuracy_score(T_val.squeeze(), cbr.predict(X_val))
    # bACC_tes = balanced_accuracy_score(T_tes.squeeze(), cbr.predict(X_tes))
    # print(bACC_tra.round(3), bACC_val.round(3), bACC_tes.round(3))

    # return true_class_tes, pred_proba_tes, pred_class_tes


def main(ws_ms, tau_ms):
    # sdir = determine_save_dir(disease_types, "CatBoostClassifier", window_size_sec)
    # sys.stdout, sys.stderr = mngs.general.tee(sys, sdir)  # log
    sdir = mngs.general.mk_spath("")
    sdir = (
        sdir + f"Catboost-{mngs.general.gen_timestamp()}/ws-{ws_ms}-ms_tau-{tau_ms}-ms/"
    )
    mngs.general.tee(sys)
    reporter = mngs.ml.ClassificationReporter(sdir)

    SAMP_RATE_EEG = 200
    ws_pts = int(ws_ms * 1e-3 * SAMP_RATE_EEG)
    # ws_pts = 1550

    dlf = DataLoaderFiller(
        n_repeat=5,
        # ws_pts=ws_pts,
        ws_pts=ws_pts,
        # ws_pts=775,
        tau_ms=tau_ms,
        val_ratio=1 / 9,
        tes_ratio=3 / 9,
        apply_sliding_window_data_augmentation=False,
    )

    # num_folds = mngs.io.load("./config/load_params.yaml")["num_folds"]
    n_folds = 5
    for i_fold in range(dlf.n_repeat):
        """
        i_fold = 0
        """
        print(f"\n {'-'*40} fold#{i_fold} starts. {'-'*40} \n")

        ## Initializes Dataloader
        dlf.fill(i_fold, reset_fill_counter=True)

        ## Training and Prediction
        true_class_tes, pred_proba_tes, pred_class_tes = train_and_predict_cb(dlf)

        ## Metrics
        reporter.calc_metrics(
            true_class_tes,
            pred_class_tes,
            pred_proba_tes,
            labels=["NoN-Ripple", "Ripple"],
            i_fold=i_fold,
        )

    reporter.summarize()

    reporter.save(meta_dict={})


if __name__ == "__main__":
    import argparse
    import mngs

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-dts",
        "--disease_types",
        default=["HV", "AD", "DLB", "NPH"],
        nargs="*",
        help=" ",
    )
    ap.add_argument("-ws", "--window_size_sec", default=2, type=int, help=" ")
    args = ap.parse_args()

    ws_ms = 7.75 * 1e3 / 2
    tau_ms = 0
    main(ws_ms, tau_ms)
