#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-29 13:45:22 (ywatanabe)"
# calc_wavelet.py


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
CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main():
    lpath = "data/Sub_01/Session_01/iEEG/AHL.pkl"
    iEEG = mngs.io.load(lpath)

    pp, aa, ff = mngs.dsp.wavelet(np.array(iEEG), CONFIG["FS_iEEG"])


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
