#!/bin/bash

# Get the name of the active environment
active_env=$(conda info --envs | grep '*' | awk '{print $1}')

# Check if the active environment is 'base'
if [ "$active_env" == "base" ]; then
  echo "The active environment is 'base'."
else
  CONDA_BASE=$(conda info --base)
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda deactivate
fi

conda env remove --name three-gen-validation -y