
# CavityPoser
CavityPoser: Mining For Bound-Like Conformations of RNA Using a Binding Cavity Screening Approach.

## Prerequisite
* Python and Python packages: numpy, pandas and scikit-learn v0.21.2

* [openbabel](http://openbabel.org/wiki/Category:Installation)

* [rdock](http://rdock.sourceforge.net/installation/)

* [PyMOL](https://pymol.org/). For first time PyMOL users: you will need a [PyMOL license file](https://pymol.org/2/buy.html?q=buy), as PyMOL is a commercial software.

## Quick Start

### Install Dependencies
```
git clone git@github.com:karoka/CavityPoser.git
cd cavity_mining/
git clone --depth=1 git@github.com:atfrank/RNAPosers.git
cd RNAPosers/
make clean
make
cd ..
cp RNAPosers/bin/featurize bin/
rm -rf RNAPosers
```

### Using CavityPoser
Main script is `src/cavityPoser.sh`. It requires a pdb file containing RNA 3D structure as input.
Example:
```
cd test/
./../src/cavityPoser.sh receptor.pdb
```
The predicted cavities and their corresponding scores are written in file `receptor/predicted_pockets.txt`.

#### Output
```
PDB, tag, cavityID, x, y, z, pred_MLP, pred_XGB, pred_RF
```

#### Optional Arguments
```
./../src/cavityPoser.sh
```
It takes the following arguments:
*  `path-to-pdb-file`: input pdb file to perform cavity mining on.
* `scalar`: 1 or 0, corresponds to different version of models (1: scalar; 0: vector). Currently the scalar version is faster and has better performance. Default: 1
* `rdock-param-file`: rdock parameter file. 3 parameter files are provided under `params/` folder. Default: `blind_docking_de_novo.prm`.
* `working-dir`: working directory. Default: a sub-directory in current working directory. The sub-directory is named as the input pdb-file name without extension. e.g. If the input filename is `receptor.pdb`, then a folder named `receptor/` will be created if not existed in current working directory and all operations will be performed there.
* `tag`: an arbitrary identifier of the structure. Default:   "decoy".
* `output-file`: output file path and name.

## Publications

* `RNAPosers`(In revision): Chhabra, Sahil, Jingru Xie, and Aaron T. Frank. "RNAPosers: Machine Learning Classifiers For RNA-Ligand Poses." bioRxiv (2019): 702449.
