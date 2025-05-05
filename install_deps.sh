#!/bin/bash
# Install PyTorch dependencies for blockchain AI models

echo "Installing PyTorch dependencies for blockchain AI models..."

# Get the absolute path to the current directory
MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$MODELS_DIR/../../.." && pwd)"

# Create a Python virtual environment if it doesn't exist
if [ ! -d "$MODELS_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$MODELS_DIR/venv"
fi

# Activate the virtual environment
source "$MODELS_DIR/venv/bin/activate"

# Get Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python $PYTHON_VERSION"

# Install dependencies
pip install -r "$MODELS_DIR/requirements.txt"

# Create a .env file for VS Code
cat > "$MODELS_DIR/.env" << EOL
PYTHONPATH=${PROJECT_ROOT}:${MODELS_DIR}:${MODELS_DIR}/venv/lib/python${PYTHON_VERSION}/site-packages
EOL

echo "Dependencies installed successfully!"
echo "Environment variables set in $MODELS_DIR/.env"
echo ""
echo "To activate the environment, run: source $MODELS_DIR/venv/bin/activate"
echo ""
echo "In VS Code, you need to:"
echo "1. Install the Python extension"
echo "2. Select the Python interpreter: Cmd+Shift+P -> Python: Select Interpreter -> $MODELS_DIR/venv/bin/python"
echo "3. Reload VS Code window: Cmd+Shift+P -> Developer: Reload Window" 