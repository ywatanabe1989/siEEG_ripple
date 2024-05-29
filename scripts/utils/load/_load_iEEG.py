#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-28 10:28:23 (ywatanabe)"

import warnings
from glob import glob

import mngs
import numpy as np


def load_iEEG(
    i_sub_str, i_session_str, iEEG_ROI, return_common_averaged_signal=False
):
    iEEG = mngs.io.load(
        f"./data/Sub_{i_sub_str}/Session_{i_session_str}/iEEG/{iEEG_ROI}.pkl"
    )

    # apply the common average referencing
    lpaths_iEEG = glob(
        f"./data/Sub_{i_sub_str}/Session_{i_session_str}/iEEG/*.pkl"
    )

    lpaths_iEEG = [
        lpaths_iEEG[ii]
        for ii in np.where(
            ~mngs.general.search(iEEG_ROI, lpaths_iEEG, as_bool=True)[0]
            == True
        )[0]
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        iEEG_all = np.hstack(
            [
                np.nanmean(mngs.io.load(lpath_iEEG), axis=1, keepdims=True)
                for lpath_iEEG in lpaths_iEEG
            ]
        )
        iEEG_common_ave = np.nanmean(iEEG_all, axis=1, keepdims=True)
        iEEG -= iEEG_common_ave
        if return_common_averaged_signal:
            return iEEG, iEEG_common_ave
        else:
            return iEEG
