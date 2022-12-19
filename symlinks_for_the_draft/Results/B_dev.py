#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-05 16:38:59 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
from eeg_ieeg_ripple_clf import utils

def describe(set_size):
    rips = rips_df.duration_ms[(rips_df.set_size == set_size) * (rips_df.phase == "Encoding")]
    described = rips.astype(float).describe()
    med = described["50%"]
    IQR = described["75%"] - described["25%"]
    print(med.round(1), IQR.round(1), len(rips))

def test():
    from itertools import combinations
    import scipy    
    for ss1, ss2 in combinations([4, 6, 8], 2):
        rips1 = rips_df.duration_ms[(rips_df.set_size == ss1) * (rips_df.phase == "Encoding")]
        rips2 = rips_df.duration_ms[(rips_df.set_size == ss2) * (rips_df.phase == "Encoding")]
        print(ss1, ss2)
        print(scipy.stats.brunnermunzel(
            rips1,
            rips2,
            alternative="less",
            ))
    
    
# Loads
rips_df = utils.load_rips()
IoU_RIPPLE_THRES = mngs.io.load("./config/global.yaml")["IoU_RIPPLE_THRES"]
# IoU_RIPPLE_THRES = 0.5
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]
rips_df = rips_df[(rips_df.IoU <= IoU_RIPPLE_THRES) * rips_df.session.astype(int) <= SESSION_THRES]

import matplotlib.pyplot as plt
plt.scatter(
    rips_df.duration_ms,
    rips_df.set_size,
    )
rips4 = rips_df.duration_ms[(rips_df.set_size == 4) * (rips_df.phase == "Encoding")]
rips6 = rips_df.duration_ms[(rips_df.set_size == 6) * (rips_df.phase == "Encoding")]
rips8 = rips_df.duration_ms[(rips_df.set_size == 8) * (rips_df.phase == "Encoding")]

rips46 = pd.concat([rips4, rips6])
rips68 = pd.concat([rips6, rips8])

plt.hist(np.log(rips4.astype(float)))
fig, ax = plt.subplots()
ax.hist(np.log10(rips4.astype(float)), density=True)
ax.hist(np.log10(rips8.astype(float)), density=True)
plt.show()

import seaborn as sns
g = sns.boxplot(data=rips_df[rips_df.phase == "Encoding"],
             x="set_size",
             y="duration_ms",

             )
g.set_yscale("log")
plt.show()
         

fig, ax = plt.subplots()
ax.boxplot(np.log10(rips4.astype(float)), positions=[4])
ax.boxplot(np.log10(rips6.astype(float)), positions=[6])
ax.boxplot(np.log10(rips8.astype(float)), positions=[8])
plt.show()

scipy.stats.brunnermunzel(
    np.log10(rips4.astype(float)),
    np.log10(rips8.astype(float)),
    alternative="less",
    )

scipy.stats.brunnermunzel(
    rips46,
    rips8,
    alternative="less"
    )

import scipy
import pandas as pd
scipy.stats.brunnermunzel(rips46, rips8) # , alternative="less"
import numpy as np
scipy.stats.pearsonr(np.log(rips_df.duration_ms[rips_df.phase == "Encoding"].astype(float)),
                     rips_df.set_size[rips_df.phase == "Encoding"])
scipy.stats.spearmanr(rips_df.duration_ms[rips_df.phase == "Encoding"],
                     rips_df.set_size[rips_df.phase == "Encoding"])

rips4.mean()
rips6.mean()
rips8.mean()


# median [IQR]
describe(4)
describe(6)
describe(8)

# Brunner-Munzel test
test()
