#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-23 17:17:30 (ywatanabe)"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from elephant.gpfa import GPFA
import quantities as pq
import mngs
import neo
import numpy as np
import pandas as pd

# import ffmpeg
# from matplotlib import animation
# import os
# from mpl_toolkits.mplot3d import Axes3D
# from bisect import bisect_right
from sklearn.model_selection import cross_val_score


def to_spiketrains(spike_times_all_trials):
    spike_trains_all_trials = []
    for st_trial in spike_times_all_trials:

        spike_trains_trial = []
        for col, col_df in st_trial.iteritems():

            spike_times = col_df[col_df != ""]
            train = neo.SpikeTrain(list(spike_times) * pq.s, t_start=-6.0, t_stop=2.0)
            spike_trains_trial.append(train)

        spike_trains_all_trials.append(spike_trains_trial)

    return spike_trains_all_trials


def get_n_bin(times_sec, bin_size, n_bins):
    bin_centers = (
        (np.arange(n_bins) * bin_size) + ((np.arange(n_bins) + 1) * bin_size)
    ) / 2
    bins = [bisect_right(bin_centers.rescale("s"), ts) for ts in np.array([times_sec])]
    return bins


def main(
    subject,
    session,
    roi,
):
    # Loads spike trains
    LPATH = f"./data/Sub_{subject}/Session_{session}/spike_times_{roi}.pkl"
    spike_trains = to_spiketrains(mngs.io.load(LPATH))

    # Parameters
    bin_size = 50 * pq.ms

    # GPFA
    # Estimate optimal dimension
    x_dims = range(1, 7)
    log_likelihoods_mm = []
    log_likelihoods_ss = []
    for x_dim in x_dims:
        try:
            gpfa_cv = GPFA(bin_size=bin_size, x_dim=x_dim)
            cv_log_likelihoods = cross_val_score(
                gpfa_cv, spike_trains, cv=3, n_jobs=-1, verbose=True
            )
            log_likelihoods_mm.append(np.mean(cv_log_likelihoods))
            # sd = 
            # se = sd / len(cv_log_likelihoods)
            # ci = 1.96 * se
            log_likelihoods_ss.append(np.std(cv_log_likelihoods)) # sd; fixme
        except:
            log_likelihoods_mm.append(np.nan)
            log_likelihoods_ss.append(np.nan)            

    # # plots
    # fig, ax = plt.subplots()
    # # ax.plot(x_dims, log_likelihoods_mm)
    # ax.errorbar(x_dims, log_likelihoods_mm, yerr=log_likelihoods_ss)
    # ax.set_xlabel("Dimensionality of latent variables for GPFA")
    # ax.set_ylabel("Log-likelihood")
    # # ax.plot(x_dims[np.argmax(log_likelihoods)], np.max(log_likelihoods), "x", markersize=10, color="r")
    # mngs.io.save(
    #     fig,
    #     f"./tmp/figs/line/GPFA_log_likelihoods/Subject_{subject}_Session_{session}_ROI_{roi}.png",
    # )
    # plt.close()

    # trajectories = gpfa.fit_transform(spike_trains)
    df = pd.DataFrame()
    df["log-likelihoods_mm"] = [np.array(log_likelihoods_mm)]
    df["log-likelihoods_ss"] = [np.array(log_likelihoods_ss)]
    df["subject"] = subject
    df["session"] = session
    df["ROI"] = roi
    df = df[["subject", "session", "ROI", "log-likelihoods_mm", "log-likelihoods_ss"]]

    return df


if __name__ == "__main__":
    from glob import glob
    import re
    from tqdm import tqdm
    LPATHs = glob(f"./data/Sub_??/Session_??/spike_times_*.pkl")

    df = pd.DataFrame()    
    for lpath in tqdm(LPATHs):
        subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
        session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
        roi = re.findall("spike_times_[\w]{2,3}", lpath)[0].split("spike_times_")[-1]
        try:
            _df = main(subject, session, roi)
            df = pd.concat([df, _df])
        except Exception as e:
            print(e)

        mngs.io.save(df, "./tmp/figs/line/GPFA_log_likelihoods/all.pkl")
        

    df = mngs.io.load("./tmp/figs/line/GPFA_log_likelihoods/all.pkl")

    matplotlib.use("TkAgg")

    # np.vstack([np.array(row).astype(float) for i_row, row in enumerate(df["log-likelihoods_mm"])])
    mm = np.nanmean(np.vstack(df["log-likelihoods_mm"]), axis=0)
    ss = np.nanstd(np.vstack(df["log-likelihoods_mm"]), axis=0)
    nn = np.sum(~np.isnan(np.vstack(df["log-likelihoods_mm"])), axis=0)
    ci = 1.96 * ss/nn

    # sd = 2 # fixme
    # se = sd / len(cv_log_likelihoods)
    # ci = 1.96 * se
    
    # is_nan = np.isnan(np.vstack(df["log-likelihoods_mm"])).any(axis=-1)
    # ss = ss / (~is_nan).sum()

    plt.errorbar(x=np.arange(len(mm)), y=mm, yerr=ci)
    
    plt.errorbar(x=np.arange(len(mm)), y=mm, yerr=ss/5)
    plt.show()

    out_df = pd.DataFrame({
        "dim": np.arange(len(mm)),
        "under": mm - ci,
        "mean": mm,
        "upper": mm + ci,
        # "std/5": ss/5,
    })
    mngs.io.save(out_df, "./tmp/figs/line/GPFA_log_likelihoods_all.csv")
    # data = np.vstack(df["log-likelihoods_mm"])
    # plt.boxplot([data[:,ii] for ii in range(data.shape[1])])
    # plt.show()

    
    
    # df = pd.DataFrame()
    # # ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    # for subject, roi in ROIs.items():
    #     subject = f"{int(subject):02d}"
    #     for session in ["01", "02"]:
    #         _df = main(subject, session, roi)
    #         df = pd.concat([df, _df])
    # mngs.io.save(df, "./tmp/figs/line/GPFA_log_likelihoods/all.csv")

    # matplotlib.use("TkAgg")
    # fig, axes = plt.subplots(nrows=len(df), sharex=True, figsize=(6.4*1, 4.8*2))  # , sharey=True
    # for i_ax, ax in enumerate(axes):
    #     label = f"{df.iloc[i_ax].subject}-{df.iloc[i_ax].session}"
    #     # ax.plot(range(1, 5), df.iloc[i_ax]["log-likelihoods"], label=label)
    #     ax.errorbar(
    #         range(1, 7),
    #         df.iloc[i_ax]["log-likelihoods_mm"],
    #         yerr=df.iloc[i_ax]["log-likelihoods_ss"],
    #         label=label,
    #     )
    # fig.supylabel("Log-likelihood")
    # fig.supxlabel("Dimensionality of latent variables for GPFA")
    # # plt.show()
    # mngs.io.save(fig, "./tmp/figs/line/GPFA_log_likelihoods/all.png")
    # mngs.io.save(df, "./tmp/figs/line/GPFA_log_likelihoods/all.pkl")

    # import mngs
    # import numpy as np
    # df = mngs.io.load("./tmp/figs/line/GPFA_log_likelihoods/all.pkl")

    # log_likelihoods_mm = np.vstack(df["log-likelihoods_mm"])
    # log_likelihoods_ss = np.vstack(df["log-likelihoods_ss"])

    # for ii in range(1, log_likelihoods_mm.shape[-1]+1):
    #     df[f"dim_{ii}_mm"] = log_likelihoods_mm[:, ii-1]
    #     df[f"dim_{ii}_ss"] = log_likelihoods_ss[:, ii-1]        

    
    # mngs.io.save(df, "./tmp/figs/line/GPFA_log_likelihoods/all.csv")
