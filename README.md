<div align="center">

# **THREE GEN | SUBNET 24**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

3D generation subnet provides a platform to democratize 3D content creation, ultimately allowing anyone to create virtual worlds, games and AR/VR/XR experiences. This subnet leverages the existing fragmented and diverse landscape of Open Source 3D generative models ranging from Gaussian Splatting, Neural Radiance Fields, 3D Diffusion Models and Point-Cloud approaches to facilitate innovation - ideal for decentralized incentive-based networks via Bittensor. This subnet aims to kickstart the next revolution in gaming around AI native games, ultimately leveraging the broader Bittensor ecosystem to facilitate experiences in which assets, voice and sound are all generated at runtime. This would effectively allow a creative individual without any coding or game-dev experience to simply describe the game they want to create and have it manifested before them in real time.

---

## Hardware Requirements

Pending detailed benchmark results, our recommended setup aligns with Google Cloud's a2-highgpu-1g specs:
- GPU: NVIDIA A100 40GB
- CPU: 12 vCPUs
- RAM: 85GB
- Storage: 200GB SSD

Expectations under continuous operation include about 500GB/month in network traffic and 0.2MB/s throughput.

## OS Requirements

Our code is compatible across various operating systems, yet it has undergone most of its testing on Debian 11, Ubuntu 20 and Arch Linux. The most rigorous testing environment used is the Deep Learning VM Image, which includes pre-installed ML frameworks and tools essential for development.

## Setup Instructions for Miners and Validators

### Environment Preparation

* *Virtual Environment:* For a clean and isolated setup, use a Python virtual environment. To set up one, execute:
```commandline
python3 -m venv .venv
source .venv/bin/activate
```

* *Process Management:* Utilize PM2 for managing application processes. It offers advantages such as automatic restarts, load balancing, and comprehensive monitoring.
* *Dependencies:* Install all necessary Python packages:
```commandline
pip install -r requirements.txt
```

### Miner Configuration

#### Starting the Miner

To initialize the miner, use:
```commandline
python neurons/miner.py --netuid 24 --subtensor.network finney --wallet.name YOUR_MINER_COLD_WALLET --wallet.hotkey YOUR_MINER_HOT_WALLET --logging.debug --neuron.device cuda:0
```
Ensure to replace `YOUR_MINER_COLD_WALLET` and `YOUR_MINER_HOT_WALLET` with your specific wallet identifiers.

### Validator Setup

#### Additional Dependencies

Before running the validator, ensure to install:
```commandline
sudo apt update
sudo apt install -y libglfw3-dev libgles2-mesa-dev libgl1-mesa-dev libgl1-mesa-glx xvfb
```
or use `setup-validator.sh`
```commandline
chmod +x setup-validator.sh
./setup-validator.sh
```

#### Launching the Validator

Edit `start-validator.sh` to set cold and hot wallet names and run it with
```commandline
chmod +x start-validator.sh
./start-validator.sh
```

You can use `ecosystem.config.json` to use run with PM2 if needed:
```commandline
pm2 start ecosystem.config.json
```