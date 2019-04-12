# Locating Objects Without Bounding Boxes
PyTorch code for https://arxiv.org/pdf/1806.07564.pdf

![Object localization](https://i.postimg.cc/bNcFr9Pf/collage3x3.png)
  

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

and install the tool:

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

## Uninstall
  
<pre>
conda deactivate object-locator
conda env remove --name object-locator
</pre>


