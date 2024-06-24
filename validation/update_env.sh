#!/bin/bash

# Stop the script on any error
set -e

# Inform user the script has started
echo "Starting environment update script..."

# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)

if [ -z "${CONDA_BASE}" ]; then
    echo "Conda is not installed or not in the PATH"
    exit 1
fi

PATH="${CONDA_BASE}/bin/":$PATH
source "${CONDA_BASE}/etc/profile.d/conda.sh"

active_env=$(conda info --envs | grep '*' | awk '{print $1}')

# Check if the active environment is 'base'
if [ "$active_env" == "base" ]; then
  echo "The active environment is 'base'."
else
  conda deactivate
fi

# Delete previous conda environment
conda env remove --name three-gen-validation -y

# Re-create conda environment and activate it
conda env create -f conda_env_validation.yml
conda activate three-gen-validation
conda info --env

CUDA_HOME=${CONDA_PREFIX}
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.0.0

echo "Environment update completed successfully."