#!/bin/bash

# Define the directory and script names
script_dir="$(dirname "${BASH_SOURCE[0]}")"

# Base directory for scripts
base_dir="$script_dir"

# Navigate to the neurons directory and source update script
cd "$base_dir/neurons" || exit 1
source update_env.sh

# Navigate to the validation directory and source update script
cd "$base_dir/validation" || exit 1
source update_env.sh

# Return to the base directory
cd "$base_dir" || exit 1

# Restart pm2 processes
pm2 restart validation
pm2 restart validator