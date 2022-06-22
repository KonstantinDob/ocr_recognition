# Dockerfile based on: https://github.com/osai-ai/dokai
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV PYTHONPATH $PYTHONPATH:/workdir
WORKDIR /workdir

# Install python and apt-get packages
RUN apt-get update && apt -y upgrade &&\
    apt-get -y install \
    software-properties-common \
    build-essential yasm nasm ninja-build \
    unzip git wget curl nano vim tmux \
    sysstat libtcmalloc-minimal4 pkgconf \
    autoconf libtool flex bison \
    libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    libsndfile1 libmp3lame-dev libssl-dev \
    python3 python3-dev python3-pip \
    liblapack-dev libopenblas-dev gfortran &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    apt-get clean &&\
    apt-get -y autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Install CMake
RUN CMAKE_VERSION=3.21.4 &&\
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz &&\
    tar -zxvf cmake-${CMAKE_VERSION}.tar.gz &&\
    cd cmake-${CMAKE_VERSION} &&\
    ./bootstrap &&\
    make && make install &&\
    cd .. && rm -rf cmake-${CMAKE_VERSION} cmake-${CMAKE_VERSION}.tar.gz

# Install pip and setuptools
RUN pip3 install --upgrade --no-cache-dir \
    pip==21.3.1 \
    setuptools==59.5.0 \
    packaging==21.2

# Install python packages
COPY ./requirements.txt /workdir
RUN pip3 install -r ./requirements.txt
