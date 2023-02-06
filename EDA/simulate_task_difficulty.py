#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-25 19:51:54 (ywatanabe)"

import string
import numpy as np
from natsort import natsorted
from itertools import combinations

ALPHABETS = list(string.ascii_lowercase)

def do_trial(kk, nn):
    presented = np.random.permutation(ALPHABETS)[:nn]
    probe_letter = np.random.permutation(ALPHABETS)[0]
    is_in_gt = ["OUT", "IN"][probe_letter in presented] # ground truth
    
    remembered = presented[:kk]
    
    # if probe_letter in remembered:
    #     return True

    answer = guess(presented, remembered, probe_letter)
    
    if answer == "100% Correct":
        return True, is_in_gt

    if (answer == "IN") and (is_in_gt == "IN"):
        return True, is_in_gt
    if (answer == "OUT") and (is_in_gt == "IN"):
        return False, is_in_gt
    if (answer == "IN") and (is_in_gt != "IN"):
        return False, is_in_gt
    if (answer == "OUT") and (is_in_gt != "IN"):
        return True, is_in_gt

def guess(presented, remembered, probe_letter):
    """
    presented = ["a", "b", "c", "d", "e", "f"]
    remembered = ["a", "b", "c"]
    probe_letter = "a"
    """
    nn = len(presented)
    kk = len(remembered)
    uu = nn - kk

    if uu <= 0:
        return "100% Correct"

    unremembered_alphabet_pool = natsorted(list(set(ALPHABETS) - set(remembered)))
    
    possible_patterns = np.array(list(combinations(unremembered_alphabet_pool, uu)))

    is_in_patterns = [probe_letter in possible_patterns[ipp] for ipp in range(len(possible_patterns))]

    is_in_pred = np.random.permutation(is_in_patterns)[0]
    
    if is_in_pred:
        return "IN"
    
    else:
        return "OUT"


def do_trials(kk, nn, match=None):
    corrects = []
    match_gts = []
    for _ in range(1000):
        correct, match_gt = do_trial(kk, nn)
        corrects.append(correct)
        match_gts.append(match_gt)

    corrects = np.array(corrects)
    match_gts = np.array(match_gts)

    if match is not None:
        corrects = corrects[[mg == match for mg in match_gts]]
        match_gts = match_gts[[mg == match for mg in match_gts]]        

    print(np.mean(corrects))    



if __name__ == "__main__":
    do_trials(7,8, "OUT")
    do_trials(7,8, "IN")            
