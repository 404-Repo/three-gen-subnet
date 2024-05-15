#!/bin/bash

# Define the directory and script names
script_dir="$(dirname "${BASH_SOURCE[0]}")"
update_validator_script="update_validator.sh"
self_script="git_pull.sh"

# Navigate to the script directory
cd "$script_dir" || exit

# Stash changes to update_validator.sh
#if [ -f "$update_validator_script" ]; then
#    git stash push -m "Stash update_validator.sh" "$update_validator_script"
#fi

# Fetch the latest changes and update
git fetch origin main
git checkout main
git reset --hard origin/main

# Ensure self-preservation: script should not modify its own execution process
if [ -f "$self_script" ]; then
    chmod +x "$self_script"
fi

# Apply the previously stashed changes if they exist
#stash_id=$(git stash list | grep "Stash update_validator.sh" | cut -d: -f1 | head -n1)
#if [ -n "$stash_id" ]; then
#    git stash apply "$stash_id"
#fi

# In case 'update_validator.sh' was deleted upstream, check before changing permissions
if [ -f "$update_validator_script" ]; then
    chmod +x "$update_validator_script"
fi