#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-03 09:52:53 (ywatanabe)"

import numpy as np
import scipy
import mngs

data = np.array(
    [
        0.60,
        0.21,
        0.40,
        0.10,
        0.63,
        0.10,
        0.13,
        0.17,
        0.83,
        0.10,
        0.35,
        0.85,
        0.18,
        0.90,
        0.37,
        0.28,
        0.15,
        0.1,
        0.20,
        0.10,
        0.27,
        0.10,
    ]
)

mngs.gen.describe(data, method="median")
mngs.gen.describe(data, method="mean")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.plot(sorted(data))
plt.boxplot(sorted(data))

import pandas as pd
df = pd.DataFrame({
    "silhouette score mean": sorted(data),
})
mngs.io.save(df, "./tmp/figs/line/silhouette_score_means.csv")



np.array(sorted(data))[-int(len(data)/4):]
