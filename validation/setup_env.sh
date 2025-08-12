#!/bin/bash

# Stop the script on any error
set -e

# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)

if [ -z "${CONDA_BASE}" ]; then
    echo "Conda is not installed or not in the PATH"
    exit 1
fi

PATH="${CONDA_BASE}/bin/":$PATH
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create conda environment and activate it
conda env create -f conda_env_validation.yml
conda activate three-gen-validation
conda info --env

CUDA_HOME=${CONDA_PREFIX}
CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include"
LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib"
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the validation.config.js file for PM2 with specified configurations
cat <<EOF > validation.config.js
module.exports = {
  apps : [{
    name: 'validation',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--port 8094'
  }]
};
EOF

echo -e "\n\n[INFO] validation.config.js generated for PM2."