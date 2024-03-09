#!/bin/bash

# Set the DISPLAY environment variable
set -x
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
set +x

# Start your Python application
python neurons/validator.py --netuid 24 --subtensor.network finney --wallet.name ... --wallet.hotkey ... --logging.debug --axon.port 8091 --neuron.device cuda:0