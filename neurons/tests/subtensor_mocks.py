import base58
import bittensor as bt
from bittensor_wallet.mock import get_mock_wallet

WALLETS = [get_mock_wallet() for _ in range(200)]
NEURONS = [bt.NeuronInfoLite.get_null_neuron() for _ in range(len(WALLETS))]
for uid, (w, n) in enumerate(zip(WALLETS, NEURONS)):
    n.uid = uid
    n.hotkey = w.hotkey.ss58_address
    n.coldkey = w.coldkey.ss58_address
    n.axon_info = bt.AxonInfo.from_dict(
        {
            "version": 0,
            "ip": "0.0.0.0",
            "port": 8092,
            "ip_type": 4,
            "hotkey": n.hotkey,
            "coldkey": n.coldkey,
            "placeholder1": 0,
            "placeholder2": 0,
            "protocol": 4,
        }
    )
METAGRAPH_INFO = bt.MetagraphInfo.from_dict(
    {
        "netuid": 17,
        "name": b"rho",
        "symbol": b"rho",
        "network_registered_at": 0,
        "owner_hotkey": "hotkey",
        "owner_coldkey": "coldkey",
        "block": 1,
        "tempo": 1,
        "last_step": 1,
        "blocks_since_last_step": 1,
        "subnet_emission": 1,
        "alpha_in": 1,
        "alpha_out": 1,
        "tao_in": 1,
        "alpha_out_emission": 1,
        "alpha_in_emission": 1,
        "tao_in_emission": 1,
        "pending_alpha_emission": 1,
        "pending_root_emission": 1,
        "subnet_volume": 1,
        "moving_price": {"bits": 1},
        "rho": 1,
        "kappa": 1,
        "min_allowed_weights": 1,
        "max_allowed_weights": 1,
        "weights_version": 0,
        "weights_rate_limit": 1,
        "activity_cutoff": 1,
        "max_validators": 1,
        "num_uids": len(WALLETS),
        "max_uids": len(WALLETS),
        "burn": 0,
        "difficulty": 0,
        "registration_allowed": True,
        "pow_registration_allowed": True,
        "immunity_period": 0,
        "min_difficulty": 0,
        "max_difficulty": 0,
        "max_weights_limit": 0,
        "min_burn": 0,
        "max_burn": 0,
        "adjustment_alpha": 1,
        "adjustment_interval": 1,
        "target_regs_per_interval": 1,
        "max_regs_per_block": 1,
        "serving_rate_limit": 1,
        "commit_reveal_weights_enabled": True,
        "commit_reveal_period": 4,
        "liquid_alpha_enabled": True,
        "alpha_high": 1,
        "alpha_low": 1,
        "bonds_moving_avg": 1,
        "hotkeys": [tuple(base58.b58decode(w.hotkey.ss58_address[1:])[:-2]) for w in WALLETS],
        "active": [
            True,
        ]
        * len(WALLETS),
        "validator_permit": [
            True,
        ]
        * len(WALLETS),
        "last_update": [
            0,
        ]
        * len(WALLETS),
        "block_at_registration": [
            0,
        ]
        * len(WALLETS),
        "alpha_stake": [
            0,
        ]
        * len(WALLETS),
        "tao_stake": [
            0,
        ]
        * len(WALLETS),
        "total_stake": [
            0,
        ]
        * len(WALLETS),
        "tao_dividends_per_hotkey": [],
        "alpha_dividends_per_hotkey": [],
    }
)
