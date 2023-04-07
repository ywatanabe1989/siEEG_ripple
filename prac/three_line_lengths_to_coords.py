#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-17 18:55:34 (ywatanabe)"

import sympy
import numpy as np


def three_line_lengths_to_coords(aa, bb, cc):
    # Definition
    a1 = sympy.Symbol("a1")    
    b1 = sympy.Symbol("b1")
    b2 = sympy.Symbol("b2")    
    
    a1 = aa
    b1 = bb

    # Calculates
    cos = (aa**2 + bb**2 - cc**22) / (2 * aa * bb)
    sin = np.sqrt(1 - cos**2)
    S1 = 1 / 2 * aa * bb * sin
    S2 = 1 / 2 * aa * b2

    # Solves
    b2 = sympy.solve(S1 - S2)[0]

    # tan1 = b2 / b1
    # tan2 = sin/cos

    # b1 = sympy.solve(tan1-tan2)[0]
    O = (0, 0, 0)
    A = (a1, 0, 0)
    B = (b1, b2, 0)

    return O, A, B


if __name__ == "__main__":
    O, A, B = three_line_lengths_to_coords(2, np.sqrt(3), 1)
    print(O, A, B)
