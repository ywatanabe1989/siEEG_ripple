#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-05 13:03:48 (ywatanabe)"

import scipy
import numpy as np
import pandas as pd
import mngs

sil_scores = np.array(
    [
        0.1000,
        0.1000,
        0.1000,
        0.1000,
        0.1000,
        0.1000,
        0.1300,
        0.1500,
        0.1700,
        0.1800,
        0.2000,
        0.2100,
        0.2700,
        0.2800,
        0.3500,
        0.3700,
        0.4000,
        0.6000,
        0.6300,
        0.8300,
        0.8500,
        0.9000,
    ]
)

x = np.linspace(0, 1, 100)
y = [(sil_scores<xx).sum()/len(sil_scores) for xx in x]
plt.plot(x, y)
plt.show()

df = pd.DataFrame({
    "silhouette_score": x,
    "probability": y,
})

mngs.io.save(df, "./tmp/figs/line/silhouette_score_cum_dist.csv")
