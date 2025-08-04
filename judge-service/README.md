# Judge Service

A judge service that uses the THUDM/GLM-4.1V-9B-Thinking LLM model to determine winners in miner vs miner duels. The service runs on a vLLM server for efficient inference.

## Overview

The Judge service acts as an impartial arbiter in miner duels, leveraging advanced language model capabilities to evaluate and determine winners. It's designed to work with validator neurons.

## Hardware Requirements

- **VRAM**: 48GB minimum
- **GPU**: NVIDIA RTX 6000 Ada or higher

## Installation

### Environment Setup

Set up your environment using the provided setup script:

```bash
./setup_env.sh
```

This script will:
1. Create a conda environment and install all requirements
2. Generate a `duels.config.js` configuration file for PM2

## Running the Service

You have two options for running the Judge service:

### Option 1: Using PM2 (Recommended)

After running `setup_env.sh`, you can manage the service using PM2 with the generated `duels.config.js` configuration. No need to activate the conda environment manually.

```bash
pm2 start duels.config.js
```

### Option 2: Direct vLLM Server Launch

Run the vLLM server directly with the following command:

```bash
conda activate three-gen-vllm
vllm serve THUDM/GLM-4.1V-9B-Thinking \
    --max-model-len 8096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --port 8095 \
    --api-key local
```

## Configuration

### Port Configuration

The default port is `8095`. If you need to use a different port:

1. Update the vLLM server launch command with your desired port
2. Update the corresponding port in `neurons/validator.config.js`

### Security Considerations

When running the validator neuron on the same machine as the vLLM server:
- You can keep `--api-key local` for simplicity
- **Important**: Ensure the port is not exposed to external networks
- Use firewall rules to restrict access if necessary
