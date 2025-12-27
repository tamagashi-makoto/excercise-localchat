#!/bin/bash
set -euo pipefail

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "Virtual environment (.venv) already exists."
fi

# Activate virtual environment for the script execution
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install -U pip

# Install the package in editable mode
echo "Installing localchat in editable mode..."
pip install -e .

echo ""
echo "----------------------------------------------------------------"
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Then you can run the application:"
echo "  localchat --model <PATH_OR_REPO_ID>"
echo "----------------------------------------------------------------"
