#!/bin/bash

# python ./utils/_load_rips.py
# python ./utils/_load_trials.py
# python ./EDA/check_ripples/unit_firing_patterns/calc_n_firings_distance.py

python ./symlinks_for_the_draft/Results/B_ripple_duration_by_set_size.py
python ./symlinks_for_the_draft/Results/C_novel_firing_patterns_during_encoding_ripples.py
python ./symlinks_for_the_draft/Results/D_ripple_distance_for_Match_IN_and_Match_OUT.py 
# session <= 7
# ('Fixation - Encoding', 'Encoding - Maintenance') greater
# BrunnerMunzelResult(statistic=0.24011131508565853, pvalue=0.5945571202915223)

# ('Fixation - Encoding', 'Fixation - Maintenance') greater
# BrunnerMunzelResult(statistic=-0.9977253935078131, pvalue=0.16047648503379652)

# ('Encoding - Maintenance', 'Fixation - Maintenance') greater
# BrunnerMunzelResult(statistic=-1.9415962451509612, pvalue=0.026644384061901928)

# Session <= 4
# ('Fixation - Encoding', 'Encoding - Maintenance') greater
# BrunnerMunzelResult(statistic=0.28240486032410117, pvalue=0.6108011507416256)

# ('Fixation - Encoding', 'Fixation - Maintenance') greater
# BrunnerMunzelResult(statistic=-0.8344923325391996, pvalue=0.20306990786910606)

# ('Encoding - Maintenance', 'Fixation - Maintenance') greater
# BrunnerMunzelResult(statistic=-1.7681180799387215, pvalue=0.03912070020039918)
