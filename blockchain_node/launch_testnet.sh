#!/bin/bash

# ArthaChain Testnet Launch Script
set -e

echo "🚀 Launching ArthaChain Testnet..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TESTNET_DIR="./testnet_data"
CONFIG_FILE="./testnet_config.toml"
LOG_FILE="./testnet.log"
PID_FILE="./testnet.pid"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if testnet is already running
check_running() {
    if [ -f "$PID_FILE" ]; then
        local PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            print_warning "Testnet is already running with PID $PID"
            echo "Use './stop_testnet.sh' to stop it first"
            exit 1
        else
            print_warning "Stale PID file found, removing..."
            rm "$PID_FILE"
        fi
    fi
}

# Function to setup testnet directory
setup_directory() {
    print_status "Setting up testnet directory..."
    
    if [ -d "$TESTNET_DIR" ]; then
        print_warning "Testnet directory already exists. Backing up..."
        mv "$TESTNET_DIR" "${TESTNET_DIR}.backup.$(date +%s)"
    fi
    
    mkdir -p "$TESTNET_DIR"
    mkdir -p "$TESTNET_DIR/rocksdb"
    mkdir -p "$TESTNET_DIR/memmap"
    mkdir -p "$TESTNET_DIR/models"
    mkdir -p "$TESTNET_DIR/logs"
    
    print_success "Testnet directory setup complete"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if cargo is installed
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo (Rust) is not installed. Please install Rust first."
        exit 1
    fi
    
    # Check if Python is available for AI models
    if ! command -v python3 &> /dev/null; then
        print_warning "Python3 not found. AI features may not work properly."
    fi
    
    print_success "Dependencies check complete"
}

# Function to build the project
build_project() {
    print_status "Building ArthaChain testnet binary..."
    
    if cargo build --release --bin arthachain; then
        print_success "Build completed successfully"
    else
        print_error "Build failed. Please check the compilation errors."
        exit 1
    fi
}

# Function to initialize genesis block
init_genesis() {
    print_status "Initializing genesis block..."
    
    # Create genesis configuration if it doesn't exist
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Initialize blockchain with genesis block
    if ./target/release/arthachain init --config "$CONFIG_FILE" --data-dir "$TESTNET_DIR"; then
        print_success "Genesis block initialized"
    else
        print_error "Failed to initialize genesis block"
        exit 1
    fi
}

# Function to start the testnet
start_testnet() {
    print_status "Starting ArthaChain testnet..."
    
    # Start the node in background
    nohup ./target/release/arthachain run \
        --config "$CONFIG_FILE" \
        --data-dir "$TESTNET_DIR" \
        --log-level info > "$LOG_FILE" 2>&1 &
    
    local PID=$!
    echo "$PID" > "$PID_FILE"
    
    print_success "Testnet started with PID $PID"
    print_status "Log file: $LOG_FILE"
    print_status "Config file: $CONFIG_FILE"
    print_status "Data directory: $TESTNET_DIR"
}

# Function to wait for node to be ready
wait_for_ready() {
    print_status "Waiting for node to be ready..."
    
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://127.0.0.1:3000/health > /dev/null 2>&1; then
            print_success "Node is ready and responding to API calls"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Node failed to start properly within timeout"
    return 1
}

# Function to display testnet information
show_info() {
    echo ""
    echo "==============================================="
    echo "🎉 ArthaChain Testnet Successfully Launched!"
    echo "==============================================="
    echo ""
    echo "Network Information:"
    echo "  Network ID: arthachain-testnet-1"
    echo "  Chain ID: 1337"
    echo ""
    echo "API Endpoints:"
    echo "  HTTP RPC: http://127.0.0.1:8545"
    echo "  WebSocket: ws://127.0.0.1:8546"
    echo "  REST API: http://127.0.0.1:3000"
    echo "  Metrics: http://127.0.0.1:9090"
    echo ""
    echo "Useful Commands:"
    echo "  Health Check: curl http://127.0.0.1:3000/health"
    echo "  Node Info: curl http://127.0.0.1:3000/node/info"
    echo "  Block Height: curl http://127.0.0.1:3000/blocks/latest"
    echo "  Faucet: curl -X POST http://127.0.0.1:3000/faucet/request"
    echo ""
    echo "Log Files:"
    echo "  Node Logs: tail -f $LOG_FILE"
    echo ""
    echo "To stop the testnet: ./stop_testnet.sh"
    echo "==============================================="
}

# Main execution
main() {
    echo "🔗 ArthaChain Testnet Launcher"
    echo "==============================="
    
    check_running
    check_dependencies
    setup_directory
    build_project
    init_genesis
    start_testnet
    
    if wait_for_ready; then
        show_info
    else
        print_error "Testnet launch failed. Check the logs: $LOG_FILE"
        exit 1
    fi
}

# Handle Ctrl+C
trap 'print_warning "Launch interrupted by user"; exit 1' INT

# Run main function
main "$@"
