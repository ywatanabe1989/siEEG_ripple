#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-04 11:42:27 (ywatanabe)"

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mngs
# from scipy.sparse.csgraph import connected_components

sys.path.append(".")
from siEEG_ripple import utils

import numpy as np


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


def to_id(y):
    """
    col = "Ripple_or_Control"
    """
    uqs = y.unique()
    _dict = {cls: i_cls for i_cls, cls in enumerate(uqs)}
    return y.replace(_dict)


def reduce_dim(X, method, y=None):
    # Dimentionality reduction
    if method == "PCA":
        # PCA
        _pca = PCA(n_components=2, random_state=42)
        X_reduced_pca = _pca.fit_transform(X)
        return X_reduced_pca

    if method == "t-SNE":
        # t-SNE
        _tsne = TSNE(n_components=2, random_state=42)
        X_reduced_tsne = _tsne.fit_transform(X)
        return X_reduced_tsne

    if method == "UMAP":
        # UMAP
        _umap = umap.UMAP(n_components=2, random_state=42, metric="cosine")
        X_reduced_umap = _umap.fit_transform(X, y)
        return X_reduced_umap


def plot(subject, session, only_ripples=False):
    eves = extract_events(rips, cons, subject, session, only_ripples=only_ripples)

    X = np.vstack(eves["firing_pattern"])
    # y = to_id(eves["phase"])
    y = to_id(eves["Ripple_or_Control"])

    # X_reduced = reduce_dim(X, "t-SNE", y=y)
    X_reduced = reduce_dim(X, "UMAP", y=y)
    eves["y"] = X_reduced[:, 0]
    eves["x"] = X_reduced[:, 1]
    import ipdb; ipdb.set_trace()
    """
    mngs.io.save(eves[["Ripple_or_Control", "x", "y"]],
    "./tmp/figs/scatter/ripple_detectable_ROIs/Subject_06_Session_02.csv")
    """    

    # plots
    cols = ["Ripple_or_Control", "phase", "match", "set_size", "correct"]
    fig, axes = plt.subplots(
        ncols=len(cols), sharex=True, sharey=True, figsize=(6.4 * 2, 4.8 * 2)
    )
    for ax, col in zip(axes, cols):
        hue_order = ["Fixation", "Encoding", "Maintenance", "Retrieval"] if col == "phase" else None
        sns.scatterplot(data=eves, x="x", y="y", hue=col, hue_order=hue_order, ax=ax)
        ax.set_title(col)
        ax.legend(loc="upper right")
    fig.suptitle(f"Subject {subject}\nSession {session}")
    # plt.show()
    return fig


rips = utils.load_rips(from_pkl=False, only_correct=False)
cons = utils.load_cons(from_pkl=False, only_correct=False)



for i_row, row in rips[["subject", "session"]].drop_duplicates().iterrows():
    """
    subject = "06"
    session = "02"
    
    fig = plot(subject, session, only_ripples=False)
    """
    fig = plot(row.subject, row.session, only_ripples=False)
    mngs.io.save(
        fig,
        f"./tmp/figs/scatter/t-SNE/ripples_and_controls/subject_{row.subject}_session_{row.session}.png",
    )
    plt.close()
    fig = plot(row.subject, row.session, only_ripples=True)
    mngs.io.save(
        fig,
        f"./tmp/figs/scatter/t-SNE/ripples/subject_{row.subject}_session_{row.session}.png",
    )
    plt.close()    
