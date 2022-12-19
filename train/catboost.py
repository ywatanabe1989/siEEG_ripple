#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-01 18:11:22 (ywatanabe)"

import numpy as np
from catboost import CatBoostClassifier, Pool
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(".")
from eeg_human_ripple_clf.utils._DataLoaderFiller import DataLoaderFiller
from sklearn.metrics import balanced_accuracy_score

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
    X_tra, T_tra = dlf.dl_tra.dataset.tensors
    X_val, T_val = dlf.dl_val.dataset.tensors
    X_tes, T_tes = dlf.dl_tes.dataset.tensors

    counts = dlf.sample_counts
    neg_weight = counts[1] / counts.sum()
    pos_weight = counts[0] / counts.sum()    
    # counts = {i_class:count for i_class, count in enumerate(dlf.sample_counts)}
    # from sklearn.utils.class_weight import compute_class_weight

    # compute_class_weight("balanced", classes=[0, 1], y=counts)
    
    # np.unique(S_tra, return_counts=True)
    SAMP_RATE_EEG = 200
    X_tra = mngs.dsp.feature_extractions.rfft_bands(X_tra, SAMP_RATE_EEG)
    X_val = mngs.dsp.feature_extractions.rfft_bands(X_val, SAMP_RATE_EEG)
    X_tes = mngs.dsp.feature_extractions.rfft_bands(X_tes, SAMP_RATE_EEG)    
    
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
    X_tra = (X_tra - X_tra.mean(axis=1, keepdims=True)) / X_tra.std(axis=1, keepdims=True)
    X_val = (X_val - X_val.mean(axis=1, keepdims=True)) / X_val.std(axis=1, keepdims=True)
    X_tes = (X_tes - X_tes.mean(axis=1, keepdims=True)) / X_tes.std(axis=1, keepdims=True)    

    # to numpy
    X_tra = np.array(X_tra)
    X_val = np.array(X_val)
    X_tes = np.array(X_tes)    
    T_tra = np.array(T_tra)
    T_val = np.array(T_val)
    T_tes = np.array(T_tes)    


    ## Model
    clf = CatBoostClassifier(
        verbose=False, allow_writing_files=False, class_weights=[neg_weight, pos_weight]
    )  # task_type="GPU" does not work

    cb_pool_tra = Pool(X_tra, label=T_tra)
    cb_pool_val = Pool(X_val, label=T_val)
    cb_pool_tes = Pool(X_tes, label=T_tes)

    ## Training
    clf.fit(cb_pool_tra, eval_set=cb_pool_val, plot=False, verbose=False)

    ## Prediction
    true_class_tes = np.array(T_tes)
    pred_proba_tes = clf.predict_proba(cb_pool_tes)
    pred_class_tes = np.argmax(pred_proba_tes, axis=1)


    bACC_tra = balanced_accuracy_score(T_tra.squeeze(), clf.predict(X_tra))
    bACC_val = balanced_accuracy_score(T_val.squeeze(), clf.predict(X_val))
    bACC_tes = balanced_accuracy_score(T_tes.squeeze(), clf.predict(X_tes))
    print(bACC_tra.round(3), bACC_val.round(3), bACC_tes.round(3))
    # import ipdb; ipdb.set_trace()
    
    return true_class_tes, pred_proba_tes, pred_class_tes


def main(ws_ms, tau_ms):
    # sdir = determine_save_dir(disease_types, "CatBoostClassifier", window_size_sec)
    # sys.stdout, sys.stderr = mngs.general.tee(sys, sdir)  # log
    sdir = mngs.general.mk_spath("")
    sdir = sdir + f"Catboost-{mngs.general.gen_timestamp()}/ws-{ws_ms}-ms_tau-{tau_ms}-ms/"
    mngs.general.tee(sys)
    reporter = mngs.ml.ClassificationReporter(sdir)
    
    SAMP_RATE_EEG = 200
    ws_pts = int(ws_ms * 1e-3 * SAMP_RATE_EEG)
    
    dlf = DataLoaderFiller(
        n_repeat=5,
        ws_pts=ws_pts,
        tau_ms=tau_ms,
        # koko
        val_ratio=3/9,
        tes_ratio=3/9,
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

    ws_ms = 500
    tau_ms = 0
    main(ws_ms, tau_ms)

