FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get -qq update && \
    apt-get -qq -y install curl bzip2 qtbase5-dev libgtk2.0-0\
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

COPY . /plant_locator
WORKDIR /plant_locator

RUN [ "conda", "env", "create", "--file", "environment.yml" ]

# Install plant-locator python package inside the conda environment
RUN [ "/bin/bash", "-c", "source activate plant-location-unet && python setup.py install" ]

# Prepare entrypoint, which just calls the plant-locator
RUN echo -e '#!/bin/bash \n source activate plant-location-unet \n python -m plant-locator "$@"' > /plant_locator/entrypoint.bash
RUN chmod u+x "/plant_locator/entrypoint.bash"
ENTRYPOINT ["/plant_locator/entrypoint.bash"]
