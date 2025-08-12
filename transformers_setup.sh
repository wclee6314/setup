#!/bin/bash

# conda 예시
conda create -n transformers python=3.11
conda activate transformers
python -m pip install -U pip "setuptools>=80" wheel

# set root detectory
ROOT_DIR="/home/jovyan"

# make workspace if not exists
if [ ! -d "$ROOT_DIR/workspace" ]; then
    mkdir -p "$ROOT_DIR/workspace"
fi

# git clone transformers
cd "$ROOT_DIR/workspace"
git clone https://github.com/huggingface/transformers.git

# install transformers
cd transformers
pip install -e .[torch]
