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
conda env create -f conda_env_updater.yml
conda activate three-gen-updater
conda info --env

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the updater.config.js file for PM2 with specified configurations
cat <<EOF > updater.config.js
module.exports = {
    apps : [{
      name: 'updater',
      script: 'updater.py',
      interpreter: '${CONDA_INTERPRETER_PATH}',
      args: '--delay 30 --check 10',
    }]
  };
EOF

echo -e "\n\n[INFO] updater.config.js generated for PM2."
