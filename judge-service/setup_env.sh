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

# Create environment and activate it
conda env create -f conda_env_judge.yml
conda activate three-gen-judge
conda info --env

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the miner.config.js file for PM2 with specified configurations
cat <<EOF > judge.config.js
module.exports = {
  apps : [{
    name: 'duels',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}'
  }]
};
EOF

echo -e "\n\n[INFO] judge.config.js generated for PM2."

