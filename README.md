# Installation instructions

## Using Docker
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
