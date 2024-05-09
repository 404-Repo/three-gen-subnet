#!/bin/bash

# Define the directory and script names
script_dir="$(dirname "${BASH_SOURCE[0]}")"

# Navigate to the script directory
cd "$script_dir" || exit

# Source the update scripts from their paths
source neurons/update_env.sh
source validation/update_env.sh

# Restart pm2 processes
pm2 restart validation
pm2 restart validator