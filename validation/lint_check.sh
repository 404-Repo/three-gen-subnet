#!/bin/bash

# Script to run poe run-check

# Ensure we're in the project root directory
cd "$(dirname "$0")/.." || exit

# Define color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running linters and formatters...${NC}"

if poetry run poe run-check; then
    echo -e "${GREEN}All checks passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review the output above for details.${NC}"
    exit 1
fi