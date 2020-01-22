# Locating Objects Without Bounding Boxes
PyTorch code for "Locating Objects Without Bounding Boxes" , CVPR 2019 - Oral, Best Paper Finalist (Top 1 %) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Ribera_Locating_Objects_Without_Bounding_Boxes_CVPR_2019_paper.html) [[Youtube]](https://youtu.be/8qkrPSjONhA?t=2620)

<img src="https://i.postimg.cc/bNcFr9Pf/collage3x3.png" width="500em"/>
<img src="https://i.postimg.cc/xC32tbYF/convergence-cropped.gif" width="500em"/>
  

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
If you are only interested in the code of the Weighted Hausdorff Distance (which is the loss used in the paper and the main contribution), you can just get the [losses.py](object-locator/losses.py) file. If you want to use the entire object location tool:

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
       --visdom-env mytrainingsession \
       --visdom-server localhost \
       --lr 1e-3 \
       --val-dir TRAINING_DIRECTORY \
       --optim Adam \
       --save saved_model.ckpt
</pre>

## <a name="datasetformat">Dataset format</a>
The options `--dataset` and `--train-dir` should point to a directory.
This directory must contain your dataset, meaning:
1. One file per image  to analyze (png, jpg, jpeg, tiff or tif).
2. One ground truth file called `gt.csv` with the following format:
```
filename,count,locations
img1.png,3,"[(28, 52), (58, 53), (135, 50)]"
img2.png,2,"[(92, 47), (33, 82)]"
```
Each row of the CSV must describe the ground truth of an image: the count (number) and location of all objects in that image.
The locations are in (y, x) format, being the origin the most top left pixel, y being the pixel row number, and x being the pixel column number.

Optionally, if you are working on precision agriculture or plant phenotyping you can use an XML file `gt.xml` instead of a CSV.
The required XML specifications can be found in
[https://communityhub.purdue.edu/groups/phenosorg/wiki/APIspecs](https://communityhub.purdue.edu/groups/phenosorg/wiki/APIspecs)
(accessible only to Purdue users) and in [this](https://hammer.figshare.com/articles/Image-based_Plant_Phenotyping_Using_Machine_Learning/7774313) thesis, but this is only useful in agronomy/phenotyping applications.
The XML file is parsed by the file `data_plant_stuff.py`.

## Pre-trained models
Models are trained separately for each of the four datasets, as described in the paper:
1. [Mall dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/mall,lambdaa=1,BS=32,Adam,LR1e-4.ckpt)
2. [Pupil dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/pupil,lambdaa=1,BS=64,SGD,LR1e-3,p=-1,ultrasmallNet.ckpt)
3. [Plant dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/plants_20160613_F54,BS=32,Adam,LR1e-5,p=-1.ckpt)
4. [ShanghaiTechB dataset](https://lorenz.ecn.purdue.edu/~cvpr2019/pretrained_models/shanghai,lambdaa=1,p=-1,BS=32,Adam,LR=1e-4.ckpt)

The [COPYRIGHT](COPYRIGHT.txt) of the pre-trained models is the same as in this repository.

As described in the paper, the pre-trained model for the pupil dataset excludes the five central layers. Thus if you want to use this model you will have to use the option `--ultrasmallnet`.

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


## Creating an issue
If you're experiencing a problem or a bug, creating a GitHub issue is encouraged, but please include the following:
1. The commit version of this repository that you ran (`git show | head -n 1`)
2. The dataset you used (including images and the CSV with groundtruth with the [appropriate format](#datasetformat))
3. CPU and GPU model(s) you are using
4. The full standard output of the training log if you are training, and the testing log if you are evaluating (you can upload it to https://pastebin.com)
5. The operating system you are using
6. The command you run to train and evaluate
