#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-16 08:24:31 (ywatanabe)"

import quantities as pq
import numpy as np
import mngs

def define_phase_time():
    """
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()
    """
    # Parameters
    bin_size = 50 * pq.ms
    width_ms = 500
    width_bins = width_ms / bin_size

    # Preparation
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = np.array(mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"])
    gs_start_end_dict = {}
    # starts, ends = [], []
    phases_starts_ends_dict = {}
    colors_dict = {}
    for i_phase, phase in enumerate(PHASES):
        start_s = int(
            DURS_OF_PHASES[:i_phase].sum() / (bin_size.rescale("s").magnitude)
        )
        end_s = int(
            DURS_OF_PHASES[: (i_phase + 1)].sum() / (bin_size.rescale("s").magnitude)
        )
        phases_starts_ends_dict[phase] = (start_s, end_s)
        center_s = int((start_s + end_s) / 2)
        start_s = center_s - int(width_bins / 2)
        end_s = center_s + int(width_bins / 2)
        gs_start_end_dict[phase] = (start_s, end_s)
        # starts.append(start_s)
        # ends.append(end_s)

        colors_dict[phase] = mngs.plt.colors.to_RGBA(["gray", "blue", "green", "red"][i_phase], alpha=1.)

    return PHASES, phases_starts_ends_dict, gs_start_end_dict, colors_dict, bin_size


