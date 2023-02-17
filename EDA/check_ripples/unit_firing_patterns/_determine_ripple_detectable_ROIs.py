#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-23 17:26:26 (ywatanabe)"

import sys

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("TkAgg")
import re
from glob import glob

import matplotlib.pyplot as plt
import mngs
import numpy as np
import umap
from natsort import natsorted
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from tqdm import tqdm

# from scipy.sparse.csgraph import connected_components

sys.path.append(".")
from siEEG_ripple import utils


def extract_events(rips, cons, subject, session, only_ripples=False):
    rips_session = rips[(rips.subject == subject) * (rips.session == session)]
    rips_session["Ripple_or_Control"] = "Ripple"
    cons_session = cons[(cons.subject == subject) * (cons.session == session)]
    cons_session["Ripple_or_Control"] = "Control"
    cols = [
        "Ripple_or_Control",
        "phase",
        "match",
        "set_size",
        "correct",
        "response_time",
        "firing_pattern",
    ]
    eves_session = pd.concat([rips_session[cols], cons_session[cols]])
    eves_session["match"] = eves_session["match"].replace(
        {1: "Match IN", 2: "Mismatch OUT"}
    )

    if only_ripples:
        eves_session = eves_session[eves_session["Ripple_or_Control"] == "Ripple"]
    return eves_session


def to_X(rr):
    import re

    for rfp in rr.firing_pattern:
        rfp.index = [
            st_str[: re.search("_Trial", st_str).span()[0]] for st_str in rfp.index
        ]
        rfp = pd.DataFrame()
    return pd.concat([pd.DataFrame(st) for st in rr.firing_pattern], axis=1).T


def reduce_dim(X, method, y=None):
    # Dimentionality reduction
    # if method == "PCA":
    #     # PCA
    #     _pca = PCA(n_components=2, random_state=0)
    #     X_reduced_pca = _pca.fit_transform(X)
    #     return X_reduced_pca

    # if method == "t-SNE":
    #     # t-SNE
    #     _tsne = TSNE(n_components=2, random_state=0)
    #     X_reduced_tsne = _tsne.fit_transform(X)
    #     return X_reduced_tsne

    if method == "UMAP":
        # UMAP
        _umap = umap.UMAP(n_components=2, random_state=0, metric="cosine")
        X_reduced_umap = _umap.fit_transform(X, y)
        return X_reduced_umap


if __name__ == "__main__":
    from sklearn.metrics import silhouette_score

    sd = 2.0
    for iEEG_ROI_STR in ["AHR", "PHL", "PHR", "ECL", "ECR", "AR"]: # "AHL", "AL",
        rips = utils.load_rips(
            ROI=iEEG_ROI_STR, only_correct=False, from_pkl=False
        )
        cons = utils.load_cons(
            ROI=iEEG_ROI_STR, only_correct=False, from_pkl=False
        )

        mngs.io.save(
            rips,
            f"./tmp/rips_df/common_average_{sd}_SD_{iEEG_ROI_STR}_with_sp.pkl",
        )
        mngs.io.save(
            cons,
            f"./tmp/cons_df/common_average_{sd}_SD_{iEEG_ROI_STR}_with_sp.pkl",
        )
        # rips = mngs.io.load(f"./tmp/rips_df/common_average_{sd}_SD_{iEEG_ROI_STR}_with_sp.pkl")
        # cons = mngs.io.load(f"./tmp/cons_df/common_average_{sd}_SD_{iEEG_ROI_STR}_with_sp.pkl")
        # rips = mngs.io.load("./tmp/_rips.pkl")
        # cons = mngs.io.load("./tmp/_cons.pkl")
        
        
        for subject in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            sessions = [
                re.match("./data/Sub_\w{2}/Session_\w{2}", sd).string[-2:]
                for sd in natsorted(glob(f"./data/Sub_{subject}/*"))
            ]
            for session in tqdm(sessions):

                try:

                    rr = rips[~rips.firing_pattern.isna()]
                    cc = cons[~cons.firing_pattern.isna()]

                    rr = rr[(rr.subject == subject) * (rr.session == session)]
                    cc = cc[(cc.subject == subject) * (cc.session == session)]

                    rr = to_X(rr)
                    cc = to_X(cc)

                    rr["label"] = 0  # "Ripple"
                    cc["label"] = 1  # "Control"

                    # Supervised UMAP
                    XY = pd.concat([rr, cc])
                    embedded = reduce_dim(XY.iloc[:, :-1], "UMAP", y=XY.iloc[:, -1])
                    XY["UMAP 1"] = embedded[:, 0]
                    XY["UMAP 2"] = embedded[:, 1]

                    sil_score = silhouette_score(embedded, XY["label"])

                    # plots
                    fig, ax = plt.subplots()
                    hue = XY["label"].replace({0:"Ripple", 1:"Control"})
                    sns.scatterplot(
                        data=XY,
                        x="UMAP 1",
                        y="UMAP 2",
                        # hue="label",
                        hue=hue,
                        ax=ax,
                    )
                    ax.set_title(f"Subject: {subject}; Session: {session}\nSilhouette score: {sil_score:.1f}")
                    # plt.show()
                    mngs.io.save(
                        fig,
                        (
                            f"./tmp/figs/scatter/ripple_detectable_ROIs/{iEEG_ROI_STR}/"
                            f"Sub_{subject}_Session_{session}.png"
                        ),
                    )
                    plt.close()

                except Exception as e:
                    print(e)
# at least onearray or dtype is required?
