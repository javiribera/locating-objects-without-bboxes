# Locating Objects Without Bounding Boxes
PyTorch code for https://arxiv.org/pdf/1806.07564.pdf

![Object localization](https://i.postimg.cc/bNcFr9Pf/collage3x3.png)
![Loss convergence](https://i.postimg.cc/xC32tbYF/convergence-cropped.gif)
  

## Citing this work
```
@article{ribera2019,
  title={Locating Objects Without Bounding Boxes},
  author={Javier Ribera and David G\"{u}era and Yuhao Chen and Edward J. Delp},
  journal={Proceedings of the Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2019},
  note={{Long Beach, CA}}
}
```

## Datasets
  The datasets used in the paper can be downloaded from:
  - [Mall dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)
  - [Pupil dataset](http://www.ti.uni-tuebingen.de/Pupil-detection.1827.0.html)
  - [Plant dataset](https://engineering.purdue.edu/~sorghum/dataset-plant-centers-2016)

## Installation
Use conda to recreate the environment provided with the code:
<pre>
conda env create -f environment.yml
</pre>

Activate the environment:
<pre>
conda activate object-locator
</pre>

Install the tool:
<pre>
pip install .
</pre>

## Usage  
Activate the environment:
<pre>
conda activate object-locator
</pre>

Run this to get help (usage instructions):
<pre>
python -m object-locator.locate -h
python -m object-locator.train -h
</pre>

Example:

<pre>
python -m object-locator.locate \
       --dataset DIRECTORY \
       --out DIRECTORY \
       --model CHECKPOINTS \
       --evaluate \
       --no-gpu \
       --radius 5
</pre>

<pre>
python -m object-locator.train \
       --train-dir TRAINING_DIRECTORY \
       --batch-size 32 \
       --env-name sorghum \
       --lr 1e-3 \
       --val-dir TRAINING_DIRECTORY \
       --optim Adam \
       --save saved_model.ckpt
</pre>

## Pre-trained models
Models are trained separately for each of the four datasets, as described in the paper:
1. [Mall dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/mall,lambdaa=1,BS=32,Adam,LR1e-4.ckpt)
2. [Pupil dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/pupil,lambdaa=1,BS=64,SGD,LR1e-3,p=-1,ultrasmallNet.ckpt)
3. [Plant dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/plants_20160613_F54,BS=32,Adam,LR1e-5,p=-1.ckpt)
4. [ShanghaiTechB dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/shanghai,lambdaa=1,p=-1,BS=32,Adam,LR=1e-4.ckpt)

The [COPYRIGHT](COPYRIGHT.txt) of the pre-trained models is the same as in this repository.

## Uninstall
<pre>
conda deactivate object-locator
conda env remove --name object-locator
</pre>


## Code Versioning
The code used in the paper corresponds to the tag `used-for-cvpr2019-submission`.
If you want to reproduce the results, checkout that tag with `git checkout used-for-cvpr2019-submission`.
The master branch is the latest version available, with convenient bug fixes and better documentation.
If you want to develop or retrain your models, we recommend the master branch.
Versions numbers follow [semantic versioning](https://semver.org) and the changelog is in [CHANGELOG.md](CHANGELOG.md).

