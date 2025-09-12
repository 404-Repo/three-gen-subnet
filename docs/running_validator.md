# Running a Validator

## Overview

A validator node consists of three coordinated components:

| Endpoint | Description | Port Exposure |
|-----------|--------------|----------------|
| **neurons/validator** | Main validator neuron that serves tasks to miners and interacts with the network. | **External** (open `axon.port`) |
| **validation/validation** | Fast local validation service used internally by the neuron. | Internal only |
| **judge-service/vllm-glm4v** | vLLM-based Judge Service that resolves miner duels. | Internal or isolated network only |

You can deploy everything on **a single GPU** or distribute components across multiple GPUs for higher throughput or redundancy.

---

## Hardware and OS

- **GPU (Recommended):** RTX 6000 Ada (48 GB VRAM minimum).  
- **Alternatives:** 2x4090. (A6000 acceptable for testing only).  
- **Supported with modification:** 5090 and other *Blackwell* architecture GPUs — these require slight adjustments to the setup scripts (this will be unified in future releases).  
- **Not recommended:** A100 and H100 GPUs — both display poor inference performance during validation and duel judging.  
- **OS:** Ubuntu 22.04 LTS with NVIDIA drivers and CUDA installed.  
- **Dependencies:** Conda and PM2 (recommended for ease of setup and service management).

> You may choose to run without Conda or PM2, but this requires manual environment and process handling.  
> Contact the core team if you need assistance with a custom deployment.

For fine-tuning and performance validation, see the [Benchmarking](#benchmarking) section below.

---

## Single-GPU Recommended Setup

The easiest and most stable deployment method is to run all components on a single GPU with at least **48 GB VRAM**.

Run the following commands on a prepared Ubuntu host with drivers and CUDA installed:

```bash
# clone repo
git clone https://github.com/404-Repo/three-gen-subnet.git
cd three-gen-subnet

# setup validation
cd validation
./setup_env.sh
pm2 start validation.config.js

# setup judge service
cd ../judge-service
./setup_env.sh

# setup neuron
cd ../neurons
./setup_env.sh
# edit validator.config.js:
#   set wallet.name, wallet.hotkey, subtensor
#   set axon.port and axon.external_port (if different)
#   set duels.judge_endpoint and duels.judge_api_key (if judge runs on another GPU)
vi validator.config.js

pm2 start validator.config.js

# check process status and logs
pm2 list
pm2 logs
```

### What This Script Does
1. **Clones** the subnet repository and enters the workspace.  
2. **Sets up isolated Conda environments** for each component using `setup_env.sh`.  
3. **Starts services with PM2**, ensuring background execution, logging, and restart on failure.  
4. **Configures the neuron** with wallet details, ports, and Judge Service endpoint before launch.  
5. **Displays status and logs** for quick inspection and monitoring.

---

## Multi-GPU Setup

For larger or production-grade deployments, you can split workloads across multiple GPUs:

| Component | Recommended GPU | Notes |
|------------|------------------|-------|
| **Neuron + Validation** | A4500 or higher | Keep both on the same machine for low latency |
| **Judge Service** | RTX 4090 or higher (≥ 24 GB VRAM) | Can run on a separate GPU or host |

When running the Judge Service on a separate GPU or host:
- Use a private or isolated network connection.  
- Configure a unique `duels.judge_api_key` and matching `--api-key` on the Judge Service.  
- Do **not** expose the Judge or Validation ports publicly.

---

## Benchmarking

To benchmark your GPU and setup performance, you can use the provided benchmark script located in the validation module.

```bash
cd three-gen-subnet
cd validation
./setup_env.sh
cd benchmark
./benchmark.sh --try-cnt 20
```

**Expected Results:**  
A good total validation time is approximately **0.35 seconds** per run.  
If your times are significantly higher, check:
- GPU driver and CUDA version compatibility.  
- Power management settings (ensure full performance mode).  
- Other workloads running on the same GPU. 

---

## Auto-Updates

The provided solution has an embedded auto-update option that is **enabled by default**. Check the `three-gen-subnet/update_validator.sh` script for details.

If you don't modify the code or setup, your validator will be updated to a newer version automatically when a new version is released.

**The update script performs the following actions:**
1. Checks the subnet repository for new versions.  
2. If an update is available, resets the repo to the latest `main` branch.  
3. Updates the neuron and validation Conda environments.  
4. Restarts the neuron and validation endpoints.  

This ensures your validator stays current with minimal manual intervention.

---

## Notes

- The above method is the **recommended** way to bring a validator online using Conda and PM2.  
- Running without Conda or PM2 (e.g., via Docker or systemd) is supported but requires manual management.  
- Only the **validator neuron** port (`axon.port`) should be externally accessible — all other endpoints must remain private.  
- Blackwell (5090 and similar) GPUs are supported with minor setup modifications, which will be unified in future updates.  
