<div align="center">

# **THREE GEN | SUBNET 29**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

3D generation subnet provides a platform to democratize 3D content creation, ultimately allowing anyone to create virtual worlds, games and AR/VR/XR experiences. This subnet leverages the existing fragmented and diverse landscape of Open Source 3D generative models ranging from Gaussian Splatting, Neural Radiance Fields, 3D Diffusion Models and Point-Cloud approaches to facilitate innovation - ideal for decentralized incentive-based networks via Bittensor. This subnet aims to kickstart the next revolution in gaming around AI native games, ultimately leveraging the broader Bittensor ecosystem to facilitate experiences in which assets, voice and sound are all generated at runtime. This would effectively allow a creative individual without any coding or game-dev experience to simply describe the game they want to create and have it manifested before them in real time.

---
## Project Structure

The project is divided into three key modules, each designed to perform specific tasks within our 3D content generation and validation framework:

- Mining Module(`mining`): Central to 3D content creation, compatible with miner neurons but can also be used independently for development and testing.

- Neurons Module (`neurons`): This module contains the neuron entrypoints for miners and validators. Miners call the RPC endpoints in the `mining` module to generate images. Validators retrieve and validate generated images. This module handles running the Bittensor subnet protocols and workflows.

- Validation Module (`validation`): Dedicated to ensuring the quality and integrity of 3D content. Like the mining module, it is designed for tandem operation with validator neurons or standalone use for thorough testing and quality checks.

## Hardware Requirements

Pending detailed benchmark results (see TODOs), our recommended setup aligns with Google Cloud's a2-highgpu-1g specs:
- GPU: NVIDIA A100 40GB
- CPU: 12 vCPUs
- RAM: 85GB
- Storage: 200GB SSD
Expectations under continuous operation include about 500GB/month in network traffic and 0.2MB/s throughput.

## OS Requirements

Our code is compatible across various operating systems, yet it has undergone most of its testing on Debian 11 and Ubuntu 20. The most rigorous testing environment used is the Deep Learning VM Image, which includes pre-installed ML frameworks and tools essential for development.  

## Running the miner
### Setting up the miner

To initialize the miner node, leverage [Conda](https://docs.conda.io/en/latest/) for a straightforward setup process:
```commandline
git clone git@github.com:404-Repo/three-gen-subnet.git
cd three-gen-subnet/mining
chmod +x setup_env.sh
./setup_env.sh
```
This script clones the repository, creates a Conda environment named three-gen-mining, installs all necessary dependencies, and prepares a PM2 configuration files.

### Running the miner

For running services, [PM2](https://pm2.io) is recommended. Ensure it's installed and update configuration files as needed for customization.
If not using PM2, activate the Conda environment with:
```commandline
conda activate three_gen_mining
```
**Note:** Avoid activating Conda when running services with PM2.

#### Generation endpoint
Start the generation service:
```commandline
pm2 start generation.config.js
```
Validate functionality by requesting video generation:
```commandline
curl -d "prompt=pink bicycle" -X POST http://127.0.0.1:8093/generate_video/ > video.mp4
```
This endpoint offers a visual output for experimentation. For 3D object generation, use a different endpoint (http://127.0.0.1:8093/generate).

#### Miner neuron
... to be updated ...


# ... to be updated ...

[//]: # (## TODO:)
[//]: # (- tests and benchmarking on different setups)