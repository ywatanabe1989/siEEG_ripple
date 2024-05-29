#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-28 17:18:54 (ywatanabe)"

from glob import glob

import mngs


def main():

    LPATHS_NT = glob(f"./data/Sub_0?/Session_0?/NT/*.npy")

    by = ["by_trial", "by_session"]

    for lpath_nt in LPATHS_NT:
        if not "_z_by" in lpath_nt:
            for _by in by:
                NT = mngs.io.load(lpath_nt)
                dim = 0 if by == "by_session" else -1
                NT_z = mngs.gen.to_z(NT, dim=dim)
                mngs.io.save(NT_z, lpath_nt.replace(".npy", f"_z_{_by}.npy"))


if __name__ == "__main__":
    main()
