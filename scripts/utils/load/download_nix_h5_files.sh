#!/bin/bash

ORIG_DIR=`pwd`

cd ./externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix/
for f in ./*.h5; do
    COMMAND="gin get-content $f"
    echo $COMMAND
    $COMMAND
done

cd $ORIG_DIR

# ./scripts/utils/load/download_nix_h5_files.sh

# EOF


