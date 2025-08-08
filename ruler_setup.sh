#!/bin/bash

# define directories
HOME_DIR="/home/jovyan"
WORKSPACE_DIR="/home/jovyan/workspace"
MODEL_DIR="/home/jovyan/home-wclee-model"

# check if the directories exist
if [ ! -d "$HOME_DIR" ]; then
    echo "Home directory $HOME_DIR does not exist."
    exit 1
fi

if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Workspace directory $WORKSPACE_DIR does not exist."
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory $MODEL_DIR does not exist."
    exit 1
fi

# clone the RULER repository
cd $WORKSPACE_DIR
git clone https://github.com/NVIDIA/RULER.git
cd RULER

# data download
cd scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh

# huggingface login
pip install huggingface_hub
huggingface-cli login

# download huggingface models
cd $MODEL_DIR
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-1M --cache-dir $MODEL_DIR
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-1M --cache-dir $MODEL_DIR
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --cache-dir $MODEL_DIR