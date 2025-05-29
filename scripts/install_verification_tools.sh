#!/bin/bash

set -e  # Exit on error

echo "==== Installing Formal Verification and Contract Safety Tools ===="

# Create directory for verification tools
VERIFICATION_DIR="$HOME/.blockchain-verification"
mkdir -p "$VERIFICATION_DIR"

echo "Installing verification tools to $VERIFICATION_DIR"

# Check operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Check if Z3 is already installed
if command -v z3 &> /dev/null; then
    Z3_PATH=$(which z3)
    echo "Z3 already installed at $Z3_PATH"
else
    echo "Installing Z3 solver..."
    
    # Download and install Z3
    Z3_VERSION="4.12.2"
    Z3_DIR="$VERIFICATION_DIR/z3"
    mkdir -p "$Z3_DIR"
    
    if [ "$OS" == "linux" ]; then
        Z3_URL="https://github.com/Z3Prover/z3/releases/download/z3-$Z3_VERSION/z3-$Z3_VERSION-x64-glibc-2.31.zip"
    else
        Z3_URL="https://github.com/Z3Prover/z3/releases/download/z3-$Z3_VERSION/z3-$Z3_VERSION-x64-osx-11.0.zip"
    fi
    
    echo "Downloading Z3 from $Z3_URL"
    curl -L "$Z3_URL" -o "$VERIFICATION_DIR/z3.zip"
    
    echo "Extracting Z3..."
    unzip -q "$VERIFICATION_DIR/z3.zip" -d "$VERIFICATION_DIR"
    
    # Rename directory to more accessible name
    Z3_EXTRACT_DIR=$(find "$VERIFICATION_DIR" -maxdepth 1 -name "z3-*" -type d | head -n 1)
    mv "$Z3_EXTRACT_DIR" "$Z3_DIR"
    
    # Clean up
    rm "$VERIFICATION_DIR/z3.zip"
    
    # Add to PATH temporarily for this script
    export PATH="$Z3_DIR/bin:$PATH"
    Z3_PATH="$Z3_DIR/bin/z3"
    
    echo "Z3 installed at $Z3_PATH"
fi

# Check if K Framework is already installed
if command -v kompile &> /dev/null; then
    K_PATH=$(which kompile)
    K_PATH=${K_PATH%/*}  # Get directory
    echo "K Framework already installed at $K_PATH"
else
    echo "Installing K Framework..."
    
    # Install dependencies for K Framework
    if [ "$OS" == "linux" ]; then
        echo "Installing K Framework dependencies..."
        sudo apt-get update
        sudo apt-get install -y build-essential m4 openjdk-11-jdk libgmp-dev libmpfr-dev pkg-config z3 libz3-dev
    else
        echo "Installing K Framework dependencies..."
        brew install opam pkg-config gmp mpfr automake libtool z3
    fi
    
    # Download and install K Framework
    K_DIR="$VERIFICATION_DIR/k-framework"
    mkdir -p "$K_DIR"
    
    echo "Cloning K Framework repository..."
    git clone https://github.com/runtimeverification/k.git "$K_DIR"
    
    echo "Building K Framework..."
    cd "$K_DIR"
    mvn package -DskipTests
    
    # Add to PATH temporarily for this script
    export PATH="$K_DIR/k-distribution/target/release/k/bin:$PATH"
    K_PATH="$K_DIR/k-distribution/target/release/k/bin"
    
    echo "K Framework installed at $K_PATH"
fi

# Install WASM contract verification tools
echo "Installing WASM contract verification tools..."

# Clone WASM semantics repository
WASM_SEMANTICS_DIR="$VERIFICATION_DIR/wasm-semantics"

if [ -d "$WASM_SEMANTICS_DIR" ]; then
    echo "WASM semantics already installed, updating..."
    cd "$WASM_SEMANTICS_DIR"
    git pull
else
    echo "Cloning WASM semantics repository..."
    git clone https://github.com/webassembly/wasm-semantics.git "$WASM_SEMANTICS_DIR"
    cd "$WASM_SEMANTICS_DIR"
fi

# Build WASM semantics
echo "Building WASM semantics..."
make build

# Install Python dependencies for property-based testing
echo "Installing Python dependencies for property-based testing..."
pip install pytest hypothesis z3-solver

# Create environment file
cat > "$VERIFICATION_DIR/env.sh" <<EOL
#!/bin/bash

# Environment variables for blockchain verification tools
export Z3_PATH="$Z3_PATH"
export K_FRAMEWORK_PATH="$K_PATH"
export WASM_SEMANTICS_PATH="$WASM_SEMANTICS_DIR"

# Add tools to PATH
export PATH="\$Z3_PATH:\$K_FRAMEWORK_PATH:\$PATH"

echo "Verification environment activated"
EOL

chmod +x "$VERIFICATION_DIR/env.sh"

# Create shell completion for verification CLI
COMPLETION_DIR="$VERIFICATION_DIR/completion"
mkdir -p "$COMPLETION_DIR"

echo "Generating shell completion for verification CLI..."
cargo run --bin contract_verifier -- --completion=bash > "$COMPLETION_DIR/contract_verifier.bash"
cargo run --bin contract_verifier -- --completion=zsh > "$COMPLETION_DIR/contract_verifier.zsh"

# Create README
cat > "$VERIFICATION_DIR/README.md" <<EOL
# Blockchain Verification Tools

This directory contains the formal verification and contract safety tools for the blockchain project.

## Tools Installed

- Z3 Solver: $Z3_PATH
- K Framework: $K_PATH
- WASM Semantics: $WASM_SEMANTICS_DIR

## Usage

1. Source the environment file:

   \`\`\`bash
   source $VERIFICATION_DIR/env.sh
   \`\`\`

2. Run the verification CLI:

   \`\`\`bash
   contract_verifier --help
   \`\`\`

## Shell Completion

Shell completion for the verification CLI is available in the \`completion\` directory.

For bash:
\`\`\`bash
source $COMPLETION_DIR/contract_verifier.bash
\`\`\`

For zsh:
\`\`\`bash
source $COMPLETION_DIR/contract_verifier.zsh
\`\`\`
EOL

echo "==== Installation Complete ===="
echo "To activate the verification environment, run:"
echo "source $VERIFICATION_DIR/env.sh"
echo "See $VERIFICATION_DIR/README.md for more information." 