## Installation
#### Singulartiy
```bash
cd /home/ywatanabe/proj/build
sbuildw_edit ~/proj/build/ripple_wm_2024_0526
python -m pip install -U pip && python -m pip install -Ur requirements.txt
```
#### mngs v1.4.0
``` bash
cd ./scripts/externals
git clone git@github.com:ywatanabe1989/mngs.git
cd mngs
git switch develop
python -m pip install -Ue mngs
python -c "import mngs; print(mngs.__version__)" # 1.4.0
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### Python path
``` bash
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
```

## Downloads the original .h5 files (using the gin command if available) 
```bash
# https://git-annex.branchable.com/install/fromsource/
# See ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/README.md
cd ~/Downloads
wget https://gin.g-node.org/G-Node/gin-cli-releases/raw/master/gin-cli-latest-linux.tar.gz
tar xvf gin-cli-latest-linux.tar.gz
sudo mv gin /usr/local/bin

# ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/human_hippocampus/
ln -s ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/human_hippocampus ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix
ln -s ./scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix ./data/data_nix
```

#### Using gin and git-annex

``` bash
gin login # ywatanabe1989
git clone git://git-annex.branchable.com/ git-annex
cd git-annex
sudo dnf -y install libtool automake autoconf curl-devel zlib-devel gmp-devel
cd git-annex

# stack
curl -sSL https://get.haskellstack.org/ | sh
stack --version
stack setup
stack build
sudo make install BUILDER=stack PREFIX=/usr/local

gin get USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM
```
