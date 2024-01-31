# THREE GEN | SUBNET 32

This project is a work in progress! Please excuse the dust 🚧

## Setup

To run the project, first create and activate a Python virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```
pip install -r requirements.txt
```

## Running the Miner 

To start the miner:

```
python neurons/miner.py --netuid 32 --subtensor.network finney --wallet.name <miner-cold-wallet> --wallet.hotkey <miner-hot-wallet> --logging.debug --neuron.device cuda:0
```

Replace `<miner-cold-wallet>` and `<miner-hot-wallet>` with your actual wallet names.

## Running the Validator

To start the validator:

``` 
python neurons/validator.py --netuid 32 --subtensor.network finney --wallet.name <validator-cold-wallet> --wallet.hotkey <validator-hot-wallet> --logging.debug --neuron.device cuda:0
```

Replace `<validator-cold-wallet>` and `<validator-hot-wallet>` with your actual wallet names.

## Process Management

We recommend using PM2 or similar process manager to run the miner and validator persistently:

```
pm2 start --name miner neurons/miner.py [OPTIONS] 
pm2 start --name validator neurons/validator.py [OPTIONS]
```

Let me know if any sections need more detail! This is just an initial rough draft.