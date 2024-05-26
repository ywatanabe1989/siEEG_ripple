#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: 2024-05-26 15:57:30 (7)
# /ssh:ywatanabe@g06:/home/ywatanabe/proj/siEEG_ripple/utils/calc_geometric_median.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import mngs

mngs.gen.reload(mngs)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main():
    pass


import numpy as np
from scipy.spatial import distance


def geometric_median(X, eps=1e-5):
    """
    Calculate the geometric median for a set of points (X) using Weiszfeld's algorithm.

    Parameters:
    - X (np.array): An array of points, shape (n_points, n_dimensions)
    - eps (float): Convergence threshold, default is 1e-5

    Returns:
    - np.array: Coordinates of the geometric median
    """
    y = np.mean(X, 0)
    while True:
        D = distance.cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        Dinv = 1 / D[nonzeros]
        Dinv_sum = np.sum(Dinv)
        W = np.einsum("ij,i->ij", X[nonzeros], Dinv)
        T = np.sum(W, 0) / Dinv_sum
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        else:
            R = (y * num_zeros + T * Dinv_sum) / (num_zeros + Dinv_sum)
            y1 = np.where(np.isnan(R), y, R)
        if np.linalg.norm(y - y1) < eps:
            return y1
        y = y1


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
