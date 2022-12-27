#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-23 12:09:28 (ywatanabe)"

import numpy as np

def calc_iou(a, b):
    """
    Calculate Intersection over Union
    a = [0, 10]
    b = [0, 3]
    b = [None, None]
    calc_iou(a, b) # 0.3
    """
    
    try:
        a = float(a[0]), float(a[1])
        b = float(b[0]), float(b[1])
    except Exception as e:
        # print(e)
        return np.nan
    
    (a_s, a_e) = a
    (b_s, b_e) = b

    a_len = a_e - a_s
    b_len = b_e - b_s

    abx_s = max(a_s, b_s)
    abx_e = min(a_e, b_e)

    abx_len = max(0, abx_e - abx_s)

    return abx_len / (a_len + b_len - abx_len)
