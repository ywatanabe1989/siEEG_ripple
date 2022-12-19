#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-07 12:27:46 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
from eeg_ieeg_ripple_clf import utils
import numpy as np

def describe(set_size):
    rips = rips_df.duration_ms[(rips_df.set_size == set_size) * (rips_df.phase == "Encoding")]
    described = rips.astype(float).describe()
    med = described["50%"]
    IQR = described["75%"] - described["25%"]
    print(med.round(1), IQR.round(1), len(rips))

def test(phase):
    from itertools import combinations
    import scipy    
    for ss1, ss2 in combinations([4, 6, 8], 2):
        rips1 = rips_df.duration_ms[(rips_df.set_size == ss1) * (rips_df.phase == phase)]
        rips2 = rips_df.duration_ms[(rips_df.set_size == ss2) * (rips_df.phase == phase)]
        print(ss1, ss2)
        print(scipy.stats.ttest_ind(
            np.log10(rips1.astype(float)),
            np.log10(rips2.astype(float)),
            alternative="less",
            ))
        # print(scipy.stats.brunnermunzel(
        #     rips1,
        #     rips2,
        #     alternative="less",
        #     ))
    
    
# Loads
rips_df = utils.load_rips()
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
# IoU_RIPPLE_THRES = 0.5
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]
rips_df = rips_df[(rips_df.IoU <= IoU_RIPPLE_THRES) * rips_df.session.astype(int) <= SESSION_THRES]

# median [IQR]
describe(4)
describe(6)
describe(8)

# ttest
test("Fixation")
test("Encoding")
test("Maintenance")
test("Retrieval")

