#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-16 10:03:21 (ywatanabe)"

import mngs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


x = "log(ripple_peak_amplitude_sd)"
x = "log(ripple duration [ms])"


for sd in [2.]:    
    rips_df = mngs.io.load(f"./tmp/rips_df_bi_{sd}_SD.csv")
    rips_df["Ripple duration [ms]"] = (rips_df["end_time"] - rips_df["start_time"]) * 1000
    rips_df["log(ripple duration [ms])"] = np.log10((rips_df["end_time"] - rips_df["start_time"]) * 1000)

    rips_df["log(ripple_peak_amplitude_sd)"] = np.log10(rips_df["ripple_peak_amplitude_sd"])

    indi = rips_df["session"] <= 2
    rips_df = rips_df[indi]

    for phase in ["Fixation", "Encoding", "Maintenance", "Retrieval"]:
        fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True)

        for i_ax, ax in enumerate(axes.ravel()):

            is_correct, match_str = \
                list(product([True, False], ["match IN", "mismatch OUT"]))[i_ax]

            if not is_correct:
                continue

            indi = rips_df["correct"] == is_correct
            # indi *= rips_df["set_size"] != 4.0
            indi *= rips_df["phase"] == phase

            # match_str = "match IN" if bool(i_ax % 2) else "mismatch OUT"
            indi *= rips_df["match"] == match_str
            print(indi.sum())

            data = rips_df[indi]

            # data = data[~data[y].isna()]
            sns.histplot(
                data=data,
                x=x,
                hue="set_size",
                kde=True,
                ax=ax,
                stat="probability",
                common_norm=False,
                bins=30,
                )

            ax.set_xlim(1, 3)
            # ax.set_xlim(np.log10(1.5), np.log10(20))                        

            correct_str = "Correct" if is_correct else "Incorrect"
            ax.set_title(f"{correct_str}; {match_str}")
            fig.suptitle(f"{phase}")
            fig.set_tight_layout(True)

        mngs.io.save(fig, f"./tmp/duration_and_set_size_correct_only/{phase}_{sd}_SD.png")
        # mngs.io.save(fig, f"./tmp/peak_amp_and_set_size_correct_only/{phase}_{sd}_SD.png")        
    
# plt.show()


# for _ in itertools.product(["correct", "incorrect"], ["match IN", "mismatch OUT"]):
#     print(_)



# list(itertools.product(["correct", "incorrect"], ["match IN", "mismatch OUT"])    )
