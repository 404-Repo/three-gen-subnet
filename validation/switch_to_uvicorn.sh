#!/bin/bash

# Original config file
input_file="validation.config.js"
backup_file="${input_file}.bak"
output_file="$input_file"

# Check if the script is already set to uvicorn
current_script=$(grep -oP '(?<=script: .)\w+' "$input_file")

if [ "$current_script" == "uvicorn" ]; then
    echo "The script is already set to 'uvicorn'. No changes made."
    exit 0
fi

# Back up the original config file
cp "$input_file" "$backup_file"
if [ $? -ne 0 ]; then
    echo "Error: Could not create backup of $input_file"
    exit 1
fi
echo "Backup of $input_file created as $backup_file"

# Extract the port and interpreter from the original config
port=$(grep -oP '(?<=args: .--port )\d+' "$input_file")
interpreter=$(grep -oP "(?<=interpreter: ').+?(?=')" "$input_file")

# Output the new configuration to the original file
cat <<EOL > "$output_file"
module.exports = {
  apps: [
    {
      name: 'validation',
      script: 'uvicorn',
      args: 'serve:app --host 0.0.0.0 --port $port --backlog 256 --workers 4',
      interpreter: '$interpreter',
    }
  ]
};
EOL

echo "New configuration saved to $output_file"
