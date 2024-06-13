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
conda env create -f conda_env_neurons.yml
conda activate three-gen-neurons
conda info --env

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the miner.config.js file for PM2 with specified configurations
cat <<EOF > miner.config.js
module.exports = {
  apps : [{
    name: 'miner',
    script: 'serve_miner.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--wallet.name default --wallet.hotkey default --subtensor.network finney --netuid 17 --axon.port 8091 --generation.endpoint http://127.0.0.1:8093 --logging.debug'
  }]
};
EOF

echo -e "\n\n[INFO] miner.config.js generated for PM2."

cat <<EOF > validator.config.js
module.exports = {
  apps : [{
    name: 'validator',
    script: 'serve_validator.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--wallet.name default --wallet.hotkey default --subtensor.network finney --netuid 17 --axon.port 8092 --validation.endpoint http://127.0.0.1:8094 --logging.debug'
  }]
};
EOF

echo -e "\n\n[INFO] validator.config.js generated for PM2."
