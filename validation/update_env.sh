#!/bin/bash

# Stop the script on any error
set -e

# Inform user the script has started
echo "Starting environment update script..."

# Check for Conda installation and initialize Conda in script
CONDA_PATH=$(which conda)
if [ -z "$CONDA_PATH" ]; then
    echo "Error: Conda is not installed or not in the PATH."
    exit 1
else
    echo "Found Conda at: $CONDA_PATH"
    eval "$($CONDA_PATH shell.bash hook)"
fi

# Ensure the conda_env_validation.yml is present
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ENV_FILE="$SCRIPT_DIR/conda_env_validation.yml"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file 'conda_env_validation.yml' does not exist in the script directory."
    exit 1
fi

# Update the Conda environment and prune any removed packages
echo "Updating the 'three-gen-validation' environment using '$ENV_FILE'."
conda env update --name three-gen-validation --file "$ENV_FILE" --prune
echo "Environment update completed successfully."

CUDA_HOME=${CONDA_PREFIX}

conda activate three-gen-validation

echo -e "\n\n[INFO] Installing diff-gaussian-rasterization package\n"
git clone --depth 1 https://github.com/ashawkey/diff-gaussian-rasterization/ ./extras/diff-gaussian-rasterization
cd ./extras/diff-gaussian-rasterization
git checkout d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea

cd ../../
git clone --branch 0.9.9.0 https://github.com/g-truc/glm.git ./extras/diff-gaussian-rasterization/third_party/glm
pip install ./extras/diff-gaussian-rasterization
rm -rf ./extras