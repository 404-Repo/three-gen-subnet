miner:
	./miner/setup_env.sh

validator:
	./validator/setup_env.sh

clean:
	./miner/cleanup_env.sh
	./validator/cleanup_env.sh

test:
	python -m pytest
