.ONESHELL:

# Define the default target
.DEFAULT_GOAL := help

# Define commands for Linux and macOS
ifeq ($(shell uname -s),Linux)
	PYTHON := python
	SHELL := bash
else ifeq ($(shell uname -s),Darwin)
	PYTHON := python
	SHELL := sh
else
# Define commands for Windows
	PYTHON := python.exe
# No idea how this works on Windows. Maybe Powershell or WSL ?
	SHELL := sh
endif

miner:
	cd mining && \
	$(SHELL) setup_env.sh


validator:
	cd validation && \
	$(SHELL) setup_env.sh

all: miner validator

start_validator:
	cd validation && \
	pm2 start validation.config.js

clean_miner:
	cd mining && \
	$(SHELL) cleanup_env.sh

clean_validator:
	cd validation && \
	$(SHELL) cleanup_env.sh

clean: clean_miner clean_validator

test:
	python -m pytest --full-trace

# Target: help
help:
	@echo "Available targets:"
	@echo "  miner               - Run miner setup script"
	@echo "  validator            - Run validator setup script"
	@echo "  all                   - Run miner and validator setup scripts"
	@echo "  start_validator       - Run validator startup script"	
	@echo "  test                  - Run tests"
	@echo "  clean_miner           - Run miner cleanup script"
	@echo "  clean_validator       - Run validator cleanup script"
	@echo "  clean                 - Run miner and validator cleanup scripts"	
	@echo "  help                  - Display this help message"