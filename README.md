# Table of Contents
1. [Using Conda (recommended)](#conda)
    1. [Installation](#installation)
    1. [Usage](#usage)
2. [Using Docker](#docker)


## Using Conda (recommended) <a name="conda"></a>
### Installation

You need to have CUDA 8 installed before proceeding with these installation instructions. This will depend greatly on the specific model of your NVIDIA card and your operating system version.

1. Install Conda (if you already have it you can go to step 2)
<pre>
curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -bf
rm -rf /tmp/miniconda.sh
</pre>
2. Make sure you have access to Conda with `which conda`. If not, maybe the conda directory is not in your math. Make sure that in your `~/.bash.rc` file you have a line such as 
<pre>
export PATH="~/miniconda3/bin:$PATH"`
</pre>
3. Create the provided conda environment (and also install all the dependencies) with
<pre>
conda env create -f environment.yml
</pre>

4. Activate it
<pre>
source activate object-location
</pre>
5. Install the object locator software in this environment
<pre>
pip install object_locator-1.1.0-py3-none-any.whl
</pre>

### Usage
1. Activate the environment
<pre>
source activate object-location
</pre>

#### Train (optional)
If you do not want to use one of the provided pretrained models, you can train your own model. Run the following command to get the full help message on how to train, with an explanation of all available training parameters.
<pre>
python -m object-locator.train -h
</pre>
This is a simple example:
<pre>
CUDA_VISIBLE_DEVICES="0,1,3" python -m object-locator.train --train-dir ~/data/20160613_F54_training_256x256 --batch-size 32 --env-name sorghum --lr 1e-3 --val-dir ~/data/plant_counts_random_patches/20160613_F54_validation_256x256 --optim Adam --save unet_model.ckpt 
</pre>

#### Test/evaluate
Use a model/checkpoint to evaluate, i.e, to locate all the objects in a dataset. Run the following command to get the full help message on how to evaluate, with an explanation of all available evbaluation parameters.
<pre>
python -m object-locator.locate -h
</pre>
<pre>
python -m object-locator.locate --dataset ~/data/20160613_F54_testing_256x256 --radius 5 --model ~/checkpoints/unet_model.ckpt --out results
</pre>
## Using Docker <a name="docker"></a>
1. Install docker-ce as described in https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository
2. Install NVIDIA drivers
3. Run the following commands
<pre><code>
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
</code></pre>
You can get more details at https://github.com/NVIDIA/nvidia-docker

