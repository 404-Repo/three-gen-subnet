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

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ENV_FILE="$SCRIPT_DIR/conda_env_neurons.yml"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file 'conda_env_neurons.yml' does not exist in the script directory."
    exit 1
fi

# Update the Conda environment and prune any removed packages
echo "Updating the 'three-gen-neurons' environment using '$ENV_FILE'."
conda env update --name three-gen-neurons --file "$ENV_FILE" --prune
echo "Environment update completed successfully."