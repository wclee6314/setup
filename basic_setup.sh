#!/bin/bash

# tmux
apt install tmux

# miniconda
ARCH=$(uname -m); \
if [ "$ARCH" = "x86_64" ]; then F=Miniconda3-latest-Linux-x86_64.sh; \
elif [ "$ARCH" = "aarch64" ]; then F=Miniconda3-latest-Linux-aarch64.sh; \
else echo "Unsupported arch: $ARCH"; exit 1; fi; \
wget https://repo.anaconda.com/miniconda/$F && \
bash $F -b -p $HOME/miniconda3 && \
$HOME/miniconda3/bin/conda init "$(basename "$SHELL")" && \
exec $SHELL -l
