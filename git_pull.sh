#!/bin/bash

# Navigate to the directory containing this script
cd "$(dirname "${BASH_SOURCE[0]}")" || exit

# Stash changes to update_validator.sh
git stash push -m "Stash update_validator.sh" update_validator.sh

# Pull the latest changes from the main branch
git fetch origin main
git checkout main
git reset --hard origin/main

# Ignore changes to git_pull.sh (assume script is executable and no need to restore executable permission)
chmod +x git_pull.sh

# Apply the previously stashed changes (flexible with stash index if multiple stashes are present)
git stash list | grep "Stash update_validator.sh" | cut -d: -f1 | xargs -I {} git stash apply {}

# Reapply the executable permissions to ensure the scripts remain executable
chmod +x update_validator.sh
chmod +x git_pull.sh