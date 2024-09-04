<div align="center">

# **THREE GEN | SUBNET 17**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

3D generation subnet provides a platform to democratize 3D content creation, ultimately allowing anyone to create virtual worlds, games and AR/VR/XR experiences. This subnet leverages the existing fragmented and diverse landscape of Open Source 3D generative models ranging from Gaussian Splatting, Neural Radiance Fields, 3D Diffusion Models and Point-Cloud approaches to facilitate innovation - ideal for decentralized incentive-based networks via Bittensor. This subnet aims to kickstart the next revolution in gaming around AI native games, ultimately leveraging the broader Bittensor ecosystem to facilitate experiences in which assets, voice and sound are all generated at runtime. This would effectively allow a creative individual without any coding or game-dev experience to simply describe the game they want to create and have it manifested before them in real time.

---
## Table of Content
1. [Project Structure](#project-structure)
2. [Hardware Requirements](#hardware-requirements)
3. [OS Requirements](#os-requirements)
4. [Setup Guidelines for Miners and Validators](#setup-guidelines-for-miners-and-validators)
   1. [Environment Management With Conda](#environment-management-with-conda)
   2. [Process Supervision With PM2](#process-supervision-with-pm2)
5. [Running the Miner](#running-the-miner)
   1. [Generation Endpoint](#generation-endpoints)
   2. [Miner Neuron](#miner-neuron)
6. [Running the Validator](#running-the-validator)
   1. [Validation Endpoint](#validation-endpoint)
   2. [Validation Neuron](#validator-neuron)
7. [Prompt Generation](#prompt-generation)
   1. [Prompt Generators](#prompt-generators)
   2. [Prompt Collector](#prompt-collector)

---
## Project Structure

The project is divided into three key modules, each designed to perform specific tasks within our 3D content generation and validation framework:

- Generation Module(`generation`): Central to 3D content creation, compatible with miner neurons but can also be used independently for development and testing.

- Neurons Module (`neurons`): This module contains the neuron entrypoints for miners and validators. Miners call the RPC endpoints in the `mining` module to generate images. Validators retrieve and validate generated images. This module handles running the Bittensor subnet protocols and workflows.

- Validation Module (`validation`): Dedicated to ensuring the quality and integrity of 3D content. Like the mining module, it is designed for tandem operation with validator neurons or standalone use for thorough testing and quality checks.

## Hardware Requirements

Our recommended setup aligns with RunPod 1 x RTX 4090 specs:
- GPU: NVIDIA 1 x RTX 4090
- CPU: 16 vCPU
- RAM: 62 GB
- Storage: 50GB SSD

Minimal setup aligns with RunPod 1 x L4 specs:
- GPU: NVIDIA 1 x L4
- CPU: 12 vCPU
- RAM: 62 GB

## OS Requirements

Our code is compatible across various operating systems, yet it has undergone most of its testing on Debian 11, Ubuntu 20 and Arch Linux. 
The most rigorous testing environment used is the Deep Learning VM Image, which includes pre-installed ML frameworks and tools essential for development.

**NOTE**: the linux image should come with **pytorch 2.1+** and **CUDA 12.1.1** otherwise you might have problems with running miner or validator pipelines.

## Setup Guidelines for Miners and Validators

### Environment Management With Conda

For optimal environment setup:
- Prefer [Conda](https://docs.conda.io/en/latest/) for handling dependencies and isolating environments. It’s straightforward and efficient for project setup.
- If Conda isn’t viable, fallback to manual installations guided by `conda_env_*.yml` files for package details, and use `requirements.txt`. Utilizing a virtual environment is highly advised for dependency management.

### Process Supervision With PM2

To manage application processes:
- Adopt [PM2](https://pm2.io) for benefits like auto-restarts, load balancing, and detailed monitoring. Setup scripts provide PM2 configuration templates for initial use. Modify these templates according to your setup needs before starting your processes.
- If PM2 is incompatible with your setup, but you're using [Conda](https://docs.conda.io/en/latest/), remember to activate the Conda environment first or specify the correct Python interpreter before executing any scripts.

## Running the Miner

By running a miner on this subnet you agree that you have obtained all licenses, rights and consents required to use, reproduce, modify, display, distribute and make available your submitted results to this subnet and its end users.

To operate the miner, the miner neuron and generation endpoints must be initiated. While currently supporting a single generation endpoint, future updates are intended to allow a miner to utilize multiple generation endpoints simultaneously.

### Generation Endpoints

Set up the environment by navigating to the directory and running the setup script:
```commandline
cd three-gen-subnet/generation
./setup_env.sh
```
This script creates a Conda environment `three-gen-mining`, installs dependencies, and sets up a PM2 configuration file (`generation.config.js`).

After optional modifications to `generation.config.js`, initiate it using [PM2](https://pm2.io):
```commandline
pm2 start generation.config.js
```

To verify the endpoint's functionality generate a test video:
```commandline
curl -d "prompt=pink bicycle" -X POST http://127.0.0.1:8093/generate_video/ > video.mp4
```

### Miner Neuron

#### Prerequisites

Ensure wallet registration as per the [official bittensor guide](https://docs.bittensor.com/subnets/register-validate-mine).

#### Setup
Prepare the neuron by executing the setup script in the `neurons` directory:
```commandline
cd three-gen-subnet/neurons
./setup_env.sh
```
This script generates a Conda environment `three-gen-neurons`, installs required dependencies, and prepares `miner.config.js` for PM2 configuration.

#### Running
Update `miner.config.js` with wallet information and ports, then execute with [PM2](https://pm2.io):
```commandline
pm2 start miner.config.js
```




## Running the Validator

Key Aspects of Operating a Validator:
1. A validator requires the operation of a validation endpoint. This endpoint functions as an independent local web server, which operates concurrently with the neuron process.
2. The validator must serve an axon, enabling miners to retrieve tasks and submit their results.

### Validation Endpoint

Set up the environment by navigating to the directory and running the setup script:
```commandline
cd three-gen-subnet/validation
./setup_env.sh
```
This script creates a Conda environment `three-gen-validation`, installs dependencies, and sets up a PM2 configuration file (`validation.config.js`).

After optional modifications to `validation.config.js`, initiate it using [PM2](https://pm2.io):
```commandline
pm2 start validation.config.js
```

**Security considerations:** Run validation endpoint behind the firewall (close the validation endpoint port).

### Validator Neuron

Ensure wallet registration as per the [official bittensor guide](https://docs.bittensor.com/subnets/register-validate-mine).

Prepare the neuron by executing the setup script in the `neurons` directory:
```commandline
cd three-gen-subnet/neurons
./setup_env.sh
```
This script generates a Conda environment `three-gen-neurons`, installs required dependencies, and prepares `validator.config.js` for PM2 configuration.

Update `validator.config.js` with wallet information and ports, then execute with [PM2](https://pm2.io):
```commandline
pm2 start validator.config.js
```

#### Important
Validator must serve the axon and the port must be opened. You can check the port using `nc`. 
```commandline
nc -vz [Your Validator IP] [Port]
```
You can also test the validator using the mock script. Navigate to the `mocks` folder and run
```commandline
PYTHONPATH=$PWD/.. python mock_miner.py --subtensor.network finney --netuid 17 --wallet.name default --wallet.hotkey default --logging.trace
```


## Prompt Generation

Our subnet supports prompt generation from two main sources: organic traffic via Public API 
and continuously updated datasets. By default, it regularly fetches new batches of prompts from our service. 
For real-time prompt generation, we currently utilize two different LLM models: 
`llama3-8b-instruct` and `gemma-1.1-7b-instruct`.

To ensure suitability for 3D generation, our system employs a carefully tailored input 
[prompt-instruction](https://github.com/404-Repo/text-prompt-generator/blob/LLM1_online_prompt_generator/launching_config.yml). 
This instruction forces the LLM to select objects for prompt generation from one of the 13 object categories identified 
based on our industry knowledge and research of gaming asset store trends. 
These selections can be updated in the future to better align with more specific datasets or marketplace curation.

To achieve true decentralization, you can switch to running the prompt generation locally and change the 
`--dataset.prompter.endpoint` parameter. 

Our prompter solution consists of two services: the generator and the collector.

### Prompt Generators

Multiple instances of prompt generators continuously produce small batches of prompts and send them to the 
collector service. You can and should launch multiple generator services to maintain a robust and dynamic system.

To set up the prompt generators:
- Generate an API key for the collector service.
- Configure the prompt generators to send batches of prompts to the collector using this API key.

For more details and to get started with prompt generators, visit the following URL:
- [Prompt Generators Repository](https://github.com/404-Repo/text-prompt-generator)

### Prompt Collector

The prompt collector accumulates prompts from multiple generators and serves fresh large batches of prompts to 
validators upon request. Validators fetch these batches every hour by default, but this interval can be customized.

To set up the prompt collector:
- Use the same API key generated for the prompt generators.
- Configure firewall rules to secure the collector service.

For more details and to get started with the prompt collector, visit the following URL:
- [Prompt Collector Repository](https://github.com/404-Repo/get-prompts)

