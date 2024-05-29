## Installation

``` bash
git clone git@github.com:ywatanabe1989/siEEG_ripple.git && cd siEEG_ripple
python -m venv env && source ./env/bin/activate
pip install -U pip && pip install -r requirements.txt
pip install -e ~/proj/mngs # for development
```

#### Adds current directory in PYTHONPATH
``` bash
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
```

## Downloads the original .h5 files using gin
```bash
# See ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/README.md

# ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/human_hippocampus/
ln -s ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/human_hippocampus ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix
ln -s ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix ./data/data_nix
```

#### Installation of git-annex using cabal
``` bash
sudo yum -y install libxml2-devel gnutls-devel libgsasl-devel ghc cabal-install happy alex libidn-devel file-devel
cabal update
BINDIR=$HOME/.cabal/bin
cabal install --bindir=$BINDIR c2hs
cabal install --bindir=$BINDIR git-annex
export PATH=$BINDIR:$PATH # Your might want to write this to ~/.bashrc as well
git-annex --version
```

#### Installation of GIN client
``` bash
cd ~/Downloads
wget https://gin.g-node.org/G-Node/gin-cli-releases/raw/master/gin-cli-latest-linux.tar.gz
tar xvf gin-cli-latest-linux.tar.gz
sudo mv gin /usr/local/bin
gin --version
gin login # ywatanabe1989
```

#### Download the dataset by Boran et al., 2020

``` bash
mkdir externals
cd externals
gin get USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM
# See ./externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/README.md
./scripts/utils/load/download_nix_h5_files.sh
```


<!-- #### Singulartiy
 !-- ```bash
 !-- cd /home/ywatanabe/proj/build
 !-- sbuildw_edit ~/proj/build/ripple_wm_2024_0526
 !-- python -m pip install -U pip && python -m pip install -Ur requirements.txt
 !-- ```
 !-- 
 !-- #### mngs v1.4.0
 !-- ``` bash
 !-- cd ./scripts/externals
 !-- git clone git@github.com:ywatanabe1989/mngs.git
 !-- cd mngs
 !-- git switch develop
 !-- python -m pip install -Ue mngs
 !-- python -c "import mngs; print(mngs.__version__)" # 1.4.0
 !-- python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
 !-- ``` -->
