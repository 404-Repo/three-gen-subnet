#!/bin/bash

# Define the directory and script names
script_dir="$(dirname "${BASH_SOURCE[0]}")"

# Base directory for scripts
base_dir="$script_dir"

# Function to source script and check for errors
source_script() {
    local script_path="$1"
    # Source the script and check if it fails
    # shellcheck disable=SC1090
    if ! source "$script_path"; then
        echo "Error: Failed to source $script_path"
        exit 1
    fi
}

# Navigate to the neurons directory and source update script
cd "$base_dir/neurons" || exit 1
source_script "update_env.sh"

# Navigate to the validation directory and source update script
cd "../validation" || exit 1
source_script "update_env.sh"

# Return to the base directory
cd "$base_dir" || exit 1

# Restart pm2 process
pm2 restart validation
pm2 restart validator