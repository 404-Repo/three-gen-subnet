#!/bin/bash

# Navigate to the directory containing this script
cd "$(dirname "${BASH_SOURCE[0]}")" || exit

# Source the update scripts from their paths
source neurons/update_env.sh
source validation/update_env.sh

# Restart pm2 processes
pm2 restart validation
pm2 restart validator