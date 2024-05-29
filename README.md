# siEEGRipple (scalpe and intracranial EEG data and Ripples)
## Hippocampal neural fluctuation between memory encoding and retrieval states during a working memory task in humans

#### [Installation](./docs/installation.md)

#### Converts the downloaded .h5 files into csv and pkl files
```bash
./scripts/utils/load/nix_2_csv_and_pkl.py
```

#### Detects ripples
```bash
./scripts/detect_ripples.py
```

#### Calculates neural trajectory (NT) with GPFA
```bash
find data -name '*NT*' | xargs rm -rf
./scripts/calc_NT_with_GPFA.py
./scripts/znorm_NT.py
```

#### Wavelet transformation

``` bash
./scripts/calc_wavelet.py
```









./EDA/check_ripples/
./EDA/check_ripples/unit_firing_patterns/
./EDA/check_ripples/unit_firing_patterns/trajectory/


#### Distance between phases
```
./EDA/check_ripples/unit_firing_patterns/trajectory/peri_SWR_dist_from_P_dev.py
```

#### Representative Trajectory of Subject 06, Session 02
```
./EDA/check_ripples/unit_firing_patterns/trajectory/repr_traj.py
./res/figs/scatter/repr_traj/session_traj_Subject_06_Session_02.csv
```

#### Representative Trajectory of Subject 06, Session 02 by condition
```
./EDA/check_ripples/unit_firing_patterns/trajectory/repr_traj_by_set_size_and_task_type.py 
./res/figs/scatter/repr_traj/

./EDA/check_ripples/unit_firing_patterns/trajectory/classify_trajectories.py 


```


please tell me useful commands to handle singularity sandbox


