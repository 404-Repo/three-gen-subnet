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

TEMP_DIR=$(mktemp -d -t flash_install_XXXXXX)
wget -O "$TEMP_DIR/flash_attn-2.7.4.post1-cp311-cp311-linux_x86_64.whl" "https://github.com/404-Repo/compiled_libs/releases/download/flash-attn-cu126-torch270/flash_attn-2.7.4.post1-cp311-cp311-linux_x86_64.whl"
pip install "$TEMP_DIR/flash_attn-2.7.4.post1-cp311-cp311-linux_x86_64.whl"

wget -O "$TEMP_DIR/flashinfer_python-0.2.5-cp39-abi3-linux_x86_64.whl" "https://github.com/404-Repo/compiled_libs/releases/download/flashinfer-cu126-torch270/flashinfer_python-0.2.5-cp39-abi3-linux_x86_64.whl"
pip install "$TEMP_DIR/flashinfer_python-0.2.5-cp39-abi3-linux_x86_64.whl"
rm -rf "$TEMP_DIR"

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
