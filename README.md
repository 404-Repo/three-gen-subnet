<div align="center">

# **THREE GEN | SUBNET 29**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

Subnet 29 provides a platform to democratize 3D content creation, ultimately allowing anyone to create virtual worlds, games and AR/VR/XR experiences. This subnet leverages the existing fragmented and diverse landscape of Open Source 3D generative models ranging from Gaussian Splatting, Neural Radiance Fields, 3D Diffusion Models and Point-Cloud approaches to facilitate innovation - ideal for decentralized incentive-based networks via Bittensor. This subnet aims to kickstart the next revolution in gaming around AI native games, ultimately leveraging the broader Bittensor ecosystem to facilitate experiences in which assets, voice and sound are all generated at runtime. This would effectively allow a creative individual without any coding or game-dev experience to simply describe the game they want to create and have it manifested before them in real time.

🚧 This project is a work in progress! Please excuse the dust 🚧

---

## Setup

To run the project, first create and activate a Python virtual environment:
```commandline
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:
```commandline
pip install -r requirements.txt
```

## Running the Miner 

To start the miner:
```commandline
python neurons/miner.py --netuid 29 --subtensor.network finney --wallet.name <miner-cold-wallet> --wallet.hotkey <miner-hot-wallet> --logging.debug --neuron.device cuda:0
```

Replace `<miner-cold-wallet>` and `<miner-hot-wallet>` with your actual wallet names.

## Running the Validator

The procedure to start the validator depends on your hardware setup. Follow the instructions corresponding to your system's capabilities:

### CUDA-enabled GPU Systems

For systems equipped with CUDA-enabled GPUs, execute the following command:
```commandline
python neurons/validator.py --netuid 29 --subtensor.network finney --wallet.name <validator-cold-wallet> --wallet.hotkey <validator-hot-wallet> --logging.debug --neuron.device cuda:0
```
This will utilize the GPU to run the validator, which is typically more performant.

### GPU Systems (Using CPU Instead)

If your system has a GPU, but you prefer to use the CPU, use this command:
```commandline
python neurons/validator.py --netuid 29 --subtensor.network finney --wallet.name <validator-cold-wallet> --wallet.hotkey <validator-hot-wallet> --logging.debug --neuron.device cpu
```
This option forces the validator to use the CPU resources instead of the GPU.

### Systems Without a GPU (But with a Display)

For systems without a dedicated GPU, but with display capabilities, run:
```commandline
python neurons/validator.py --netuid 29 --subtensor.network finney --wallet.name <validator-cold-wallet> --wallet.hotkey <validator-hot-wallet> --logging.debug --neuron.device cpu --neuron.opengl_platform pyglet
```
This configuration is suitable for running the validator using CPU rendering with an available display environment.

### Systems With No GPU or Display

For headless systems without a GPU or display (work in progress and may not function correctly):
```commandline
python neurons/validator.py --netuid 29 --subtensor.network finney --wallet.name <validator-cold-wallet> --wallet.hotkey <validator-hot-wallet> --logging.debug --neuron.device cpu --neuron.opengl_platform osmesa
```
Note: This option is currently under development and may encounter issues during execution.

### Important: 
Replace <validator-cold-wallet> and <validator-hot-wallet> with the actual names of your wallets before running the commands.

## Process Management

We recommend using PM2 or similar process manager to run the miner and validator persistently:

```
pm2 start --name miner neurons/miner.py -- [OPTIONS] 
pm2 start --name validator neurons/validator.py -- [OPTIONS]
```