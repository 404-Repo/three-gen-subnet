#!/bin/bash

# Stop the script on any error
set -e

# Check for Conda installation and initialize Conda in script
if [ -z "$(which conda)" ]; then
    echo "Conda is not installed or not in the PATH"
    exit 1
fi

# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment and activate it
conda env create -f conda_env_validation.yml
conda activate three-gen-validation
conda info --env
CUDA_HOME=${CONDA_PREFIX}

echo -e "\n\n[INFO] Installing diff-gaussian-rasterization package\n"
mkdir -p ./extras/diff_gaussian_rasterization/third_party
git clone --branch 0.9.9.0 https://github.com/g-truc/glm.git ./extras/diff_gaussian_rasterization/third_party/glm
pip install ./extras/diff_gaussian_rasterization

echo -e "\n\n[INFO] Installing simple-knn package\n"
pip install ./extras/simple-knn

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
