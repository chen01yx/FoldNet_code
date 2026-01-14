# Codebase of FoldNet
This repository is the official implementation of [FoldNet](https://pku-epic.github.io/FoldNet/).

## install
```
git clone https://github.com/chen01yx/FoldNet_code.git --recurse-submodules
cd FoldNet_code
git submodule update --init --recursive
```

## conda and dependency
```
conda create -n garmentds python=3.9
pip install -e . --use-pep517
sh setup.sh
```

## simulation environment
Please refer to src/garmentds/foldenv/simenv_readme.md for more details.

## mesh synthesis
Coming soon.