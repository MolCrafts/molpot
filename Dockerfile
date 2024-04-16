# syntax=docker/dockerfile:1

ARG cuda_version=11.1

FROM docker.io/pytorch/pytorch:2.2.2-cuda${cuda_version}-cudnn8-runtime as base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM base as build
COPY --from=base /opt/conda /opt/conda
WORKDIR /work
RUN git clone https://github.com/MolCrafts/molpy.git
RUN /opt/conda/bin/python -m pip install -e molpy
RUN git clone https://github.com/MolCrafts/molpot.git
RUN /opt/conda/bin/python -m pip install -e molpot