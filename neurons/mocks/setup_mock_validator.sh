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
conda env create -f conda_env_mock_validator.yml
conda activate mock-validator
conda info --env

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

cat <<EOF > mock.validator.config.js
module.exports = {
  apps : [{
    name: 'mock.validator',
    script: 'mock_validator.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--wallet.name validator --wallet.hotkey defalut --gateway_wallet_name default --gateway_wallet_hotkey default --subtensor.network test --netuid 89 --logging.trace'
  }]
};
EOF

echo -e "\n\n[INFO] mock.validator.config.js generated for PM2."
