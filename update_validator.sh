#!/bin/bash

# Define the process names as variables
VALIDATION_CONFIG="validation/validation.config.js"
VALIDATOR_CONFIG="neurons/validator.config.js"

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

# Skipping env update for this update

cd "$base_dir/neurons" || exit 1
#source_script "update_env.sh"

# Navigate to the validation directory and source update script

cd "../validation" || exit 1
#source_script "update_env.sh"

# Return to the base directory
cd ".." || exit 1

# Extract the validation process name from the validation config file

#validation_process=$(grep -oP "(?<=name: ').+?(?=')" "$VALIDATION_CONFIG")
#if [ -z "$validation_process" ]; then
#    echo "Error: Could not find the validation process name in $VALIDATION_CONFIG"
#    exit 1
#fi


# Function to check if a pm2 process is running successfully
is_process_running() {
    local process_name="$1"

    # Get the status of the process
    local status
    status=$(pm2 info "$validation_process" | grep -i "status" | awk '{print $4}')

    if [[ "$status" == "online" ]] || [[ "$status" == "launching" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to restart a pm2 process until it succeeds or max attempts reached
restart_with_retry() {
    local process_name="$1"
    local max_attempts="$2"
    local attempt=0

    while (( attempt < max_attempts )); do
        pm2 stop "$process_name"

        sleep 10 # Adding a delay to allow process to stop completely

        pm2 restart "$process_name"

        sleep 10 # Adding a delay to allow process to restart

        if is_process_running "$process_name"; then
            echo "$process_name restarted successfully."
            return 0
        else
            echo "Attempt $(( attempt + 1 )) to restart $process_name failed."
        fi
        (( attempt++ ))
    done

    echo "Error: Failed to restart $process_name after $max_attempts attempts."
    return 1
}

# Restart validation process up to 5 times
#restart_with_retry "$VALIDATION_CONFIG" 5

# Restart validator process
pm2 restart "$VALIDATOR_CONFIG"