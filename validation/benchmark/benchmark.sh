#!/bin/bash

# Set variables
SCRIPT_DIR=$(dirname "$(realpath "$0")")
CONDA_VALIDATION_CLIENT_ENV=three-gen-validation
while [[ $# -gt 0 ]]; do
    case "$1" in
        --try-cnt)
            TRY_CNT=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z $TRY_CNT ]]; then
    echo -e "\033[0;31mError: --try-cnt parameter is mandatory\033[0m"
    exit 1
fi

# Setup conda environment
if [[ "$CONDA_DEFAULT_ENV" = "$CONDA_VALIDATION_CLIENT_ENV" ]]; then
    echo "Conda environment '$CONDA_DEFAULT_ENV' is already activated."
    DEACTIVATE_ENV=false
else
    echo "No Conda environment is activated."
    CONDA_BASE=$(conda info --base)
    if [ -z "${CONDA_BASE}" ]; then
        echo -e "\033[0;31mError: Conda is not installed or not in the PATH\033[0m"
        exit 1
    fi
    PATH="${CONDA_BASE}/bin/":$PATH
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate $CONDA_VALIDATION_CLIENT_ENV
    DEACTIVATE_ENV=true
fi

# Run script
PYTHONPATH="$SCRIPT_DIR/.." python3 "$SCRIPT_DIR/benchmark_validation.py" --try-cnt $TRY_CNT
if [[ -z $DEACTIVATE_ENV ]]; then
  conda deactivate
fi
