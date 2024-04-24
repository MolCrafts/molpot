# syntax=docker/dockerfile:1

ARG cuda_version=11.8

FROM docker.io/pytorch/pytorch:2.2.2-cuda${cuda_version}-cudnn8-runtime as base

WORKDIR /work
RUN /opt/conda/bin/python -m pip install -i https://test.pypi.org/simple/ molpot==0.0.1
RUN /opt/conda/bin/python -m pip install -i https://test.pypi.org/simple/ molcrafts-molpy==0.0.1
