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
conda env create -f conda_env_mining.yml
conda activate three-gen-mining
conda info --env
CUDA_HOME=${CONDA_PREFIX}

echo -e "\n\n[INFO] Installing diff-gaussian-rasterization package\n"
mkdir -p ./extras/diff_gaussian_rasterization/third_party
git clone --branch 0.9.9.0 https://github.com/g-truc/glm.git ./extras/diff_gaussian_rasterization/third_party/glm
pip install ./extras/diff_gaussian_rasterization

echo -e "\n\n[INFO] Installing simple-knn package\n"
pip install ./extras/simple-knn

echo -e "\n\n[INFO] Installing MVDream package\n"
pip install ./extras/MVDream

echo -e "\n\n[INFO] Installing ImageDream package\n"
pip install ./extras/ImageDream

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the generation.config.js file for PM2 with specified configurations
cat <<EOF > generation.config.js
module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--port 8093'
  }]
};
EOF

echo -e "\n\n[INFO] generation.config.js generated for PM2."