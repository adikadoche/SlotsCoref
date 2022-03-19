#!/bin/bash
ADDITIONAL_PARAMS="${@:1:99999}"


PYTHONUNBUFFERED=1 /home/gamir/adiz/miniconda3/envs/torchGPU/bin/python \
  t5.py \
  ${ADDITIONAL_PARAMS}