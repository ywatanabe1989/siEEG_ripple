#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-09 19:48:38 (ywatanabe)"

def calc_iou(a, b):
    """
    Calculate Intersection over Union
    a = [0, 10]
    b = [0, 3]
    calc_iou(a, b) # 0.3
    """
    (a_s, a_e) = a
    (b_s, b_e) = b

    a_len = a_e - a_s
    b_len = b_e - b_s

    abx_s = max(a_s, b_s)
    abx_e = min(a_e, b_e)

    abx_len = max(0, abx_e - abx_s)

    return abx_len / (a_len + b_len - abx_len)
