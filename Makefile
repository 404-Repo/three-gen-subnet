miner:
    ./miner/setup_env.sh

validator:
    ./validator/setup_env.sh

all: miner validator

clean_miner:
    ./miner/cleanup_env.sh

clean_validator:
    ./validator/cleanup_env.sh

clean: clean_miner clean_validator

test:
    python -m pytest --full-trace

