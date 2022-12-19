#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-09 00:08:02 (ywatanabe)"

import itertools

def perm(n, seq):
    out = []
    for p in itertools.product(seq, repeat=n):
        out.append(p)
    return out
