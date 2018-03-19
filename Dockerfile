FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Install conda
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

COPY . /object_locator
WORKDIR /object_locator

# Create conda environment with all the dependencies
RUN [ "conda", "env", "create", "--file", "environment.yml" ]

# Install object-locator python package inside the conda environment
RUN [ "/bin/bash", "-c", "source activate object-location && python setup.py install" ]

# Prepare entrypoint, which just calls the object-locator
RUN echo '#!/bin/bash \n source activate object \n python -m object-locator "$@"' > /object_locator/entrypoint.bash
RUN chmod u+x "/object_locator/entrypoint.bash"
ENTRYPOINT ["/object_locator/entrypoint.bash"]
