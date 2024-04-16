# syntax=docker/dockerfile:1

ARG cuda_version=11.8

FROM docker.io/pytorch/pytorch:2.2.2-cuda${cuda_version}-cudnn8-runtime as base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

FROM base as build
COPY --from=base /opt/conda /opt/conda
WORKDIR /work
RUN git clone https://github.com/MolCrafts/molpy.git
RUN /opt/conda/bin/python -m pip install -e molpy
RUN git clone https://github.com/MolCrafts/molpot.git
RUN /opt/conda/bin/python -m pip install -e molpot
