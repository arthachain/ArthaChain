# AI Engine Models for Blockchain

This directory contains neural network models for the blockchain's brain-computer interface (BCI) implementation.

## Requirements

- Python 3.8+
- PyTorch 1.13.0+
- NumPy 1.22.0+

## Installation

To install the required dependencies, run:

```bash
# Navigate to this directory
cd blockchain_node/src/ai_engine/models

# Make the install script executable
chmod +x install_deps.sh

# Run the installation script
./install_deps.sh
```

This will create a Python virtual environment and install all necessary dependencies.

## Usage from Rust

The Rust code uses PyO3 to interface with these Python models. PyO3 needs to know where to find the Python interpreter and packages. Ensure your Python environment is properly set up before running the Rust application.

### Troubleshooting

If you see import errors for torch:

1. Make sure you've run the installation script
2. If using an IDE like VSCode, configure the Python interpreter to use the virtual environment
3. Set the PYTHONPATH environment variable if needed:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/blockchain_node/src/ai_engine/models
``` 