#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-06 14:01:26 (ywatanabe)"


import random
import sys
from copy import deepcopy
from pprint import pprint
from time import sleep
from functools import partial

import mngs
import numpy as np
import pandas as pd
import sklearn
import torch
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import power_transform
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import skimage

# from meg_asd_clf.utils._load_all import load_all
# from meg_asd_clf.utils._divide_into_datasets import divide_into_datasets

# Crops EEG data
def _crop_a_signal(
    sig_arr_trial,
    rand_init_value=0,
    ws_pts=100,
    start_cut_pts=0,
    end_cut_pts=0,
):
    """
    sig_arr_trial: (n_chs, 1600)
    viewed       : (15, 100, n_chs)

    Example:
        n_chs = 6
        seq_len = 1600
        sig_arr_trial = np.random.rand(n_chs, seq_len)
        _crop_a_signal(sig_arr_trial)
    """

    # to 2D
    if sig_arr_trial.ndim == 1:
        sig_arr_trial = sig_arr_trial[np.newaxis, ...]

    n_chs = len(sig_arr_trial)

    # slides the signal
    rand_init_value = rand_init_value % ws_pts
    sig_arr_trial = sig_arr_trial[
        :, start_cut_pts : sig_arr_trial.shape[-1] - end_cut_pts
    ]
    sig_arr_trial = sig_arr_trial[:, rand_init_value:]

    # crops the signal
    viewed = skimage.util.view_as_windows(
        sig_arr_trial.T,
        window_shape=(ws_pts, n_chs),
        step=ws_pts,
    ).squeeze()

    # to 3D
    if viewed.ndim == 2:
        viewed = viewed[..., np.newaxis]

    return viewed


def _to_hz(cropped_ripples, SAMP_RATE_EEG=200):
    SAMP_RATE_EEG = 200
    ws_sec = cropped_ripples.shape[1] / SAMP_RATE_EEG
    hz = cropped_ripples.sum(axis=1) / ws_sec
    return hz


class DataLoaderFiller(object):
    def __init__(
        self,
        n_repeat=5,
        ws_pts=100,
        batch_size=64,
        num_workers=20,
        do_under_sampling=True,
        dtype="fp32",
        val_ratio=2 / 9,
        tes_ratio=2 / 9,
        random_state=42,
        tau_ms=250,
        SAMP_RATE_EEG=200,
        **kwargs,
    ):

        # Fix random seed
        self.random_state = random_state
        mngs.general.fix_seeds(seed=random_state, random=random, np=np, torch=torch)

        # Attributes
        self.n_repeat = n_repeat
        self.ws_pts = ws_pts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.do_under_sampling = do_under_sampling
        self.dtype_np = np.float32 if dtype == "fp32" else np.float16
        self.dtype_torch = torch.float32 if dtype == "fp32" else torch.float16
        self.val_ratio = val_ratio
        self.tes_ratio = tes_ratio
        self.tau_pts = int(tau_ms * 1e-3 * SAMP_RATE_EEG)
        self.i_fold = 0

        # Loads the dataset
        self.data_all = mngs.io.load("./tmp/dataset.pkl")
        # Applies under-sampling
        self.data_all = self.data_all[
            (self.data_all["session"] == "01") + (self.data_all["session"] == "02")
        ]

        # Data splitting
        self.data_tra = (
            self.data_all.set_index("subject").loc[list(self.subs_tra)].copy()
        )
        self.data_val = (
            self.data_all.set_index("subject").loc[list(self.subs_val)].copy()
        )
        self.data_tes = (
            self.data_all.set_index("subject").loc[list(self.subs_tes)].copy()
        )

        # Fills dataloaders
        self._fill_counter = 0
        self.fill(i_fold=0, reset_fill_counter=True)

    def fill(self, i_fold, reset_fill_counter=True):
        """
        Fills self.dl_tra, self.dl_val, and self.dl_tes.

        self.dl_val and self.dl_tes are always the same because they have no randomness.
        """
        self.i_fold = i_fold

        if reset_fill_counter:
            self._fill_counter = 0

        rand_init_value = random.randint(0, self.ws_pts - 1)
        self.dl_tra = self.fill_a_dataset(
            self.data_tra,
            rand_init_value=rand_init_value,
            ws_pts=self.ws_pts,
            do_shuffle=True,
        )
        # count sample sizes
        self.sample_counts = np.unique(
            self.dl_tra.dataset.tensors[1].numpy(), return_counts=True
        )[1]

        if self._fill_counter == 0:

            if self.val_ratio == 0:
                self._dl_val = None
            else:
                self._dl_val = self.fill_a_dataset(
                    self.data_val,
                    rand_init_value=0,
                    ws_pts=self.ws_pts,
                    do_shuffle=True,
                )

            if self.tes_ratio == 0:
                self._dl_tes = None
            else:
                self._dl_tes = self.fill_a_dataset(
                    self.data_tes,
                    rand_init_value=0,
                    ws_pts=self.ws_pts,
                    do_shuffle=False,
                )

            self.dl_val = deepcopy(self._dl_val)
            self.dl_tes = deepcopy(self._dl_tes)

        if self._fill_counter >= 1:
            self.dl_val = deepcopy(self._dl_val)
            self.dl_tes = deepcopy(self._dl_tes)

        self._fill_counter += 1

    def fill_a_dataset(self, subset, rand_init_value, ws_pts, do_shuffle):
        """
        subset: self.data_tra, self.data_val, or self.data_tes
        """

        if self.tau_pts < 0:
            start_cut_pts_EEG = -self.tau_pts
            end_cut_pts_EEG = 0

            start_cut_pts_Ripple = 0
            end_cut_pts_Ripple = -self.tau_pts

        if 0 <= self.tau_pts:
            start_cut_pts_EEG = 0
            end_cut_pts_EEG = self.tau_pts

            start_cut_pts_Ripple = self.tau_pts
            end_cut_pts_Ripple = 0

        _crop_a_signal_EEG = partial(
            _crop_a_signal,
            rand_init_value=rand_init_value,
            ws_pts=ws_pts,
            start_cut_pts=start_cut_pts_EEG,
            end_cut_pts=end_cut_pts_EEG,
        )

        _crop_a_signal_Ripple = partial(
            _crop_a_signal,
            rand_init_value=rand_init_value,
            ws_pts=ws_pts,
            start_cut_pts=start_cut_pts_Ripple,
            end_cut_pts=end_cut_pts_Ripple,
        )

        EEG = subset["EEG"].apply(_crop_a_signal_EEG)
        ripples = subset["Ripples"].apply(_crop_a_signal_Ripple)
        ripples_hz = ripples.apply(_to_hz)

        E, R = [], []
        # for i_row, (e, r) in self.data_val[["EEG", "Ripples_hz"]].iterrows():
        for i_row, (e, r) in enumerate(zip(EEG, ripples_hz)):
            e = e.transpose(0, 2, 1)
            E.append(e)
            R.append(r)

        E = np.vstack(E).astype(np.float32)
        R = np.vstack(R).astype(np.float32)
        R[R > 0] = 1
        # R[R > 1] = 1        
        R = R.astype(int)

        arrs_list = [E, R]
        dl = DataLoader(
            TensorDataset(*[torch.tensor(d) for d in arrs_list]),
            batch_size=self.batch_size,
            shuffle=do_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return dl

    @property
    def subs_all(
        self,
    ):
        return natsorted(np.unique(self.data_all["subject"]))

    @property
    def subs_tra(
        self,
        random_state=42,
    ):
        random_state += self.i_fold
        return self.split_subs(
            self.subs_all, self.val_ratio, self.tes_ratio, random_state=random_state
        )[0]

    @property
    def subs_val(
        self,
        random_state=42,
    ):
        random_state += self.i_fold
        return self.split_subs(
            self.subs_all, self.val_ratio, self.tes_ratio, random_state=random_state
        )[1]

    @property
    def subs_tes(
        self,
        random_state=42,
    ):
        random_state += self.i_fold
        return self.split_subs(
            self.subs_all, self.val_ratio, self.tes_ratio, random_state=random_state
        )[2]

    @property
    def n_subs_tra(
        self,
    ):
        return len(self.subs_tra)

    @property
    def n_subs_val(
        self,
    ):
        return len(self.subs_val)

    @property
    def n_subs_tes(
        self,
    ):
        return len(self.subs_tes)

    @staticmethod
    def split_subs(subs_all, val_ratio, tes_ratio, random_state=42):
        val_ratio_to_tra_and_val = val_ratio / (1 - tes_ratio)
        _subs_tra_val, subs_tes = train_test_split(subs_all, test_size=tes_ratio)
        subs_tra, subs_val = train_test_split(
            _subs_tra_val, test_size=val_ratio_to_tra_and_val
        )
        return subs_tra, subs_val, subs_tes


def seed_worker(worker_id):
    """
    https://dajiro.com/entry/2021/04/13/233032
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    dlf = DataLoaderFiller()

    n_epochs = 3
    for i_fold in range(dlf.n_repeat):
        dlf.fill(i_fold, reset_fill_counter=True)
        for epoch in range(n_epochs):
            dlf.fill(i_fold, reset_fill_counter=False)

            for i_batch, batch in enumerate(dlf.dl_val):  # 45 batches * 64 samples
                Eb_val, Rb_val = batch

                # Validate your model

            for i_batch, batch in enumerate(dlf.dl_tra):  # 106
                Eb_tra, Rb_tra = batch
                print(Rb_tra)
    #             print()
    #             print(epoch)
    #             print()
    #             print(Sgb_tra)
    #             # Train your model

    #     for i_batch, batch in enumerate(dlf.dl_tes):
    #         Xb_tes, Tb_tes, Sgb_tes, Slb_tes, A_tes, G_tes, M_tes = batch
    #         # Test your model

    # # Summarize Cross Validation Scores
