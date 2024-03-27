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

# Run generation endpoint setup script
gen_ep:
	cd generation && \
	$(SHELL) setup_env.sh

# Run validation endpoint setup script
val_ep:
	cd validation && \
	$(SHELL) setup_env.sh

# Run generation and validation endpoint setup scripts
all: gen_ep val_ep

# Run validation endpoint startup script
start_gen_ep:
	cd generation && \
	pm2 start generation.config.js

# Run validation endpoint startup script
start_val_ep:
	cd validation && \
	pm2 start validation.config.js

clean_gen_ep:
	cd generation && \
	$(SHELL) cleanup_env.sh

clean_val_ep:
	cd validation && \
	$(SHELL) cleanup_env.sh

clean: clean_gen_ep clean_val_ep

test:
	python -m pytest --full-trace

# Target: help
help:
	@echo "Available targets:"
	@echo "  gen_ep            - Run generation endpoint setup script"
	@echo "  val_ep            - Run validation endpoint setup script"
	@echo "  all               - Run generation and validation endpoint setup scripts"
	@echo "  start_gen_ep      - Run generation endpoint startup script"	
	@echo "  start_val_ep      - Run validation endpoint startup script"	
	@echo "  test              - Run tests"
	@echo "  clean_gen_ep      - Run generation endpoint cleanup script"
	@echo "  clean_val_ep      - Run validation endpoint cleanup script"
	@echo "  clean             - Run generation and validation endpoint cleanup scripts"	
	@echo "  help              - Display this help message"