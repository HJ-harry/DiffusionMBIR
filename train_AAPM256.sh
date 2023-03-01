#!/bin/bash

python main.py \
  --config=configs/ve/AAPM_256_ncsnpp_continuous.py \
  --eval_folder=eval/AAPM256 \
  --mode='train' \
  --workdir=workdir/AAPM256