<div align="center">

# 404—GEN | SUBNET 17

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

404's Mission is Simple: Democratize 3D Content Creation.

We provide a platform to accelerate the creation of 3D assets, ultimately empowering creators of all skill levels to build immersive virtual worlds, games, and AR/VR/XR experiences.

The Open Source 3D Generative Model lanscape is fragmented and diverse. We believe that by leveraging Bittensor - a decentralised incentive based network - we can facilitate innovation.

We aim to kickstart the next revolution in virtual world building, AI native games and beyond by leveraging the broader Bittnesor Ecosystem - facilitating experiences in which all elements, 3D Assets, Dialogue and Sound are all generated at runtime. The would empower those without any coding or game-dev experience to bring their ideas to life with a prompt and see them manifest in real time.

---

## Table of Content

- [Dashboard](#dashboard)
- [Running a Validator](#running-a-validator)
- [Prompt Generation](#prompt-generation)

---

## Dashboard

Monitor subnet activity, validator performance, and network metrics on our dashboard:

**[404-GEN Dashboard](https://dashboard.404.xyz/d/main/404-gen/)**

---

## Running a Validator

To run a validator on Subnet 17, please refer to our comprehensive documentation:

**[Validator Setup Guide](https://github.com/404-Repo/three-gen-subnet/blob/main/docs/running_validator.md)**

This guide covers all the necessary steps for setting up and running a validator, including requirements, configuration, and best practices.

---

## Prompt Generation

Our subnet supports prompt generation from two sources: 

- Organic Traffic via Public API.
- Prompt Datasets (that are continuously updated).

By default, the prompt querying system regularly fetches new prompt batches from our autonomous prompt generation service. For real-time prompt generation, we currently utilize two prompt generators working on different principals. We will refer to them as [prompt_generator_1](#prompt-generator-1-) and [prompt_generator_2](#prompt-generator-2).

### Prompt Generator 1 
Uses three LLM models ["Qwen/Qwen2.5-7B-Instruct-1M", "microsoft/phi-4", "THUDM/glm-4-9b-chat-1m"] for generating prompts that are automatically swapped after a certain period of time.

To ensure suitability for 3D generation using this version of the prompt generator, our system employs a carefully tailored input 
[prompt-instruction](https://github.com/404-Repo/text-prompt-generator/blob/main/configs/pipeline_config.yml). This instruction forces the LLM that is being used to pick objects (main subjects for the generating prompt) from one of the pre-defined 
[categories](https://github.com/404-Repo/text-prompt-generator/blob/main/configs/pipeline_config.yml) (30 in total).

Those categories were defined using our industry knowledge and research conducted on store trends for gaming assets.

The predefined selection of object categories can be updated/extended in the future for better alignment with current market needs, e.g. datasets for training/evaluating 3D generative AI models, marketplace curation, etc.

### Prompt Generator 2
To better simulate human-like object descriptions (organic traffic), we use a secondary text generation approach. In this method, we randomly sample initial letters for both object categories and specific objects, then prompt an LLM to select a category and an object beginning with those letters. The model is then asked to write a one-sentence description elaborating on the object's qualities. Although this process is slower, it produces more diverse, detailed, and natural-sounding prompts.

### Prompt Collector

The prompt collector accumulates prompts from multiple generators and serves fresh large batches of prompts to 
validators upon request. Validators fetch these batches every hour by default, but this interval can be customized.

To set up the prompt collector:
- Use the same API key generated for the prompt generators.
- Configure firewall rules to secure the collector service.

For more details and to get started with the prompt collector, visit the following URL:
- [Prompt Collector Repository](https://github.com/404-Repo/get-prompts)
