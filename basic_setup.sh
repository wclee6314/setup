#!/bin/bash
set -e

# tmux
apt install -y tmux

# miniconda
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    F=Miniconda3-latest-Linux-x86_64.sh
elif [ "$ARCH" = "aarch64" ]; then
    F=Miniconda3-latest-Linux-aarch64.sh
else
    echo "Unsupported arch: $ARCH"; exit 1
fi
apt install -y wget
if [ -d "$HOME/miniconda3" ]; then
    echo "miniconda3 already installed at $HOME/miniconda3, skipping installer"
else
    wget -nc https://repo.anaconda.com/miniconda/$F
    bash $F -b -p $HOME/miniconda3
    rm -f $F
fi
$HOME/miniconda3/bin/conda init "$(basename "$SHELL")"

# use conda directly (current shell hasn't been re-sourced yet)
CONDA=$HOME/miniconda3/bin/conda
PIP=$HOME/miniconda3/bin/pip

#
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

#
$PIP install torch
$PIP install pynvml
$PIP install numpy

echo "Done. Run 'exec \$SHELL -l' or open a new shell to activate conda."