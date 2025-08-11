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
conda env create -f conda_env_vllm.yml
conda activate three-gen-vllm
conda info --env

# Store the path of the Conda interpreter
CONDA_VLLM_PATH=$(which vllm)
CONDA_BIN_DIR=$(dirname "$CONDA_VLLM_PATH")

# Generate the duels.config.js file for PM2 with specified configurations
cat <<EOF > duels.config.js
module.exports = {
  apps : [{
    name: 'vllm-glm4v',
    script: '${CONDA_VLLM_PATH}',
    args: 'serve THUDM/GLM-4.1V-9B-Thinking --max-model-len 8096 --tensor-parallel-size 1 --gpu-memory-utilization 0.7  --max_num_seqs 12  --port 8095 --api-key local',
    interpreter: 'none',
    env: {
      PATH: '${CONDA_BIN_DIR}:/usr/local/bin:/usr/bin:/bin'
    }
  }]
};
EOF

echo -e "\n\n[INFO] duels.config.js generated for PM2."
