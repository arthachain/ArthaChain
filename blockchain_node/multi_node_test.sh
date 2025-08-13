#!/bin/bash

# ArthaChain Multi-Node Testing Script
set -e

echo "🌐 Starting ArthaChain Multi-Node Test Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NUM_NODES=4
BASE_PORT=30300
BASE_RPC_PORT=8540
BASE_API_PORT=3000
BASE_WS_PORT=8540
TEST_DIR="./multi_node_test"
NODE_DATA_DIR="$TEST_DIR/nodes"
LOG_DIR="$TEST_DIR/logs"

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

print_node() {
    echo -e "${CYAN}[NODE $1]${NC} $2"
}

# Function to cleanup previous test environment
cleanup_previous() {
    print_status "Cleaning up previous test environment..."
    
    # Kill any existing nodes
    pkill -f "arthachain.*node-" 2>/dev/null || true
    
    # Remove old test directory
    if [ -d "$TEST_DIR" ]; then
        print_warning "Removing existing test directory..."
        rm -rf "$TEST_DIR"
    fi
    
    sleep 2
    print_success "Cleanup completed"
}

# Function to setup test directories
setup_directories() {
    print_status "Setting up test directories..."
    
    mkdir -p "$NODE_DATA_DIR"
    mkdir -p "$LOG_DIR"
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        mkdir -p "$NODE_DATA_DIR/node-$i"
        mkdir -p "$NODE_DATA_DIR/node-$i/rocksdb"
        mkdir -p "$NODE_DATA_DIR/node-$i/memmap"
        mkdir -p "$NODE_DATA_DIR/node-$i/models"
    done
    
    print_success "Test directories created"
}

# Function to generate node configuration
generate_node_config() {
    local node_id=$1
    local p2p_port=$((BASE_PORT + node_id))
    local rpc_port=$((BASE_RPC_PORT + node_id))
    local api_port=$((BASE_API_PORT + node_id))
    local ws_port=$((BASE_WS_PORT + node_id))
    
    # Generate boot nodes list (all other nodes)
    local boot_nodes=""
    for i in $(seq 0 $((NUM_NODES-1))); do
        if [ $i -ne $node_id ]; then
            local other_port=$((BASE_PORT + i))
            boot_nodes="$boot_nodes\"/ip4/127.0.0.1/tcp/$other_port/p2p/12D3KooWTestNode$i\","
        fi
    done
    boot_nodes=$(echo "$boot_nodes" | sed 's/,$//')  # Remove trailing comma
    
    cat > "$NODE_DATA_DIR/node-$node_id/config.toml" << EOF
# Multi-Node Test Configuration for Node $node_id

[network]
network_id = "arthachain-multinode-test"
chain_id = 1337
name = "ArthaChain Multi-Node Test"

[node]
node_id = "testnet-node-$(printf "%02d" $node_id)"
data_dir = "$NODE_DATA_DIR/node-$node_id"
log_level = "info"

[consensus]
algorithm = "SVBFT"
block_time = 2
max_block_size = 1048576  # 1MB
max_tx_pool_size = 5000
validator_set_size = $NUM_NODES
min_validator_stake = 100000

[network_p2p]
listen_addr = "0.0.0.0:$p2p_port"
max_peers = $((NUM_NODES * 2))
boot_nodes = [
    $boot_nodes
]

[rpc]
http_enabled = true
http_addr = "127.0.0.1"
http_port = $rpc_port
http_cors_origins = ["*"]

ws_enabled = true
ws_addr = "127.0.0.1"
ws_port = $ws_port

[api]
enabled = true
addr = "127.0.0.1"
port = $api_port
rate_limit = 500

[ai_engine]
enabled = true
fraud_detection_model = "./models/fraud_detection.onnx"
identity_model = "./models/identity_verification.onnx"
inference_batch_size = 16
model_update_frequency = 500

[storage]
backend = "hybrid"
rocksdb_path = "$NODE_DATA_DIR/node-$node_id/rocksdb"
rocksdb_max_files = 500
memmap_path = "$NODE_DATA_DIR/node-$node_id/memmap"
memmap_size = 536870912  # 512MB

[metrics]
enabled = true
prometheus_addr = "127.0.0.1"
prometheus_port = $((9090 + node_id))
health_check_interval = 15

[security]
quantum_resistance = true
signature_algorithm = "Dilithium3"
encryption_algorithm = "Kyber768"
hash_algorithm = "BLAKE3"

[faucet]
enabled = $([ $node_id -eq 0 ] && echo "true" || echo "false")
distribution_amount = 500
cooldown_period = 1800
max_daily_requests = 20

[genesis]
timestamp = $(date +%s)
initial_supply = 500000000

[testing]
debug_mode = true
fast_mode = true
skip_verification = false
tx_tracing = true
EOF

    print_node $node_id "Configuration generated (P2P: $p2p_port, RPC: $rpc_port, API: $api_port)"
}

# Function to build the project
build_project() {
    print_status "Building ArthaChain for multi-node testing..."
    
    if cargo build --release --bin arthachain; then
        print_success "Build completed successfully"
    else
        print_error "Build failed. Please check the compilation errors."
        exit 1
    fi
}

# Function to start a node
start_node() {
    local node_id=$1
    local config_file="$NODE_DATA_DIR/node-$node_id/config.toml"
    local log_file="$LOG_DIR/node-$node_id.log"
    local pid_file="$LOG_DIR/node-$node_id.pid"
    
    print_node $node_id "Starting node..."
    
    # Start node in background
    nohup ./target/release/arthachain run \
        --config "$config_file" \
        --data-dir "$NODE_DATA_DIR/node-$node_id" \
        --log-level info > "$log_file" 2>&1 &
    
    local pid=$!
    echo "$pid" > "$pid_file"
    
    print_node $node_id "Started with PID $pid"
}

# Function to wait for nodes to be ready
wait_for_nodes() {
    print_status "Waiting for nodes to be ready..."
    
    local ready_count=0
    local max_attempts=60  # 2 minutes
    local attempt=0
    
    while [ $attempt -lt $max_attempts ] && [ $ready_count -lt $NUM_NODES ]; do
        ready_count=0
        
        for i in $(seq 0 $((NUM_NODES-1))); do
            local api_port=$((BASE_API_PORT + i))
            if curl -s "http://127.0.0.1:$api_port/health" > /dev/null 2>&1; then
                ready_count=$((ready_count + 1))
            fi
        done
        
        echo -ne "\rNodes ready: $ready_count/$NUM_NODES (attempt $((attempt + 1))/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo ""
    
    if [ $ready_count -eq $NUM_NODES ]; then
        print_success "All $NUM_NODES nodes are ready!"
        return 0
    else
        print_error "Only $ready_count/$NUM_NODES nodes became ready within timeout"
        return 1
    fi
}

# Function to run basic connectivity tests
test_connectivity() {
    print_status "Testing node connectivity..."
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        local rpc_port=$((BASE_RPC_PORT + i))
        
        # Test API endpoint
        if curl -s "http://127.0.0.1:$api_port/node/info" > /dev/null; then
            print_node $i "API endpoint responding"
        else
            print_error "Node $i API endpoint not responding"
        fi
        
        # Test RPC endpoint
        if curl -s -X POST \
            -H "Content-Type: application/json" \
            -d '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
            "http://127.0.0.1:$rpc_port" > /dev/null; then
            print_node $i "RPC endpoint responding"
        else
            print_warning "Node $i RPC endpoint not responding (may not be fully implemented)"
        fi
    done
}

# Function to run consensus tests
test_consensus() {
    print_status "Testing consensus mechanism..."
    
    # Send a test transaction to node 0
    local api_port=$BASE_API_PORT
    
    print_status "Sending test transaction..."
    
    local tx_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{
            "from": "0x742d35Cc6634C0532925a3b8D6Dd6782b4c99999",
            "to": "0x742d35Cc6634C0532925a3b8D6Dd6782b4c88888",
            "amount": 100,
            "fee": 10
        }' \
        "http://127.0.0.1:$api_port/transactions/send" 2>/dev/null || echo "")
    
    if [ -n "$tx_response" ]; then
        print_success "Test transaction sent"
    else
        print_warning "Test transaction failed (API may not be fully implemented)"
    fi
    
    # Wait for block propagation
    print_status "Waiting for block propagation..."
    sleep 5
    
    # Check block heights on all nodes
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        local height=$(curl -s "http://127.0.0.1:$api_port/blocks/latest" 2>/dev/null | grep -o '"height":[0-9]*' | cut -d':' -f2 || echo "0")
        print_node $i "Block height: ${height:-0}"
    done
}

# Function to show node status
show_status() {
    echo ""
    echo "==============================================="
    echo "🎉 Multi-Node Test Environment Active!"
    echo "==============================================="
    echo ""
    echo "Node Information:"
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local p2p_port=$((BASE_PORT + i))
        local rpc_port=$((BASE_RPC_PORT + i))
        local api_port=$((BASE_API_PORT + i))
        local ws_port=$((BASE_WS_PORT + i))
        local prometheus_port=$((9090 + i))
        
        echo "  Node $i:"
        echo "    P2P Port: $p2p_port"
        echo "    RPC: http://127.0.0.1:$rpc_port"
        echo "    API: http://127.0.0.1:$api_port"
        echo "    WebSocket: ws://127.0.0.1:$ws_port"
        echo "    Metrics: http://127.0.0.1:$prometheus_port"
        echo "    Logs: tail -f $LOG_DIR/node-$i.log"
        echo ""
    done
    
    echo "Test Commands:"
    echo "  Node Status: curl http://127.0.0.1:3000/node/info"
    echo "  Block Height: curl http://127.0.0.1:3000/blocks/latest"
    echo "  Send Transaction: curl -X POST -H 'Content-Type: application/json' \\"
    echo "    -d '{\"from\":\"0x...\",\"to\":\"0x...\",\"amount\":100}' \\"
    echo "    http://127.0.0.1:3000/transactions/send"
    echo ""
    echo "To stop all nodes: ./stop_multi_node_test.sh"
    echo "==============================================="
}

# Main execution
main() {
    echo "🔗 ArthaChain Multi-Node Test Setup"
    echo "====================================="
    
    cleanup_previous
    setup_directories
    build_project
    
    # Generate configurations for all nodes
    print_status "Generating node configurations..."
    for i in $(seq 0 $((NUM_NODES-1))); do
        generate_node_config $i
    done
    
    # Start all nodes
    print_status "Starting $NUM_NODES nodes..."
    for i in $(seq 0 $((NUM_NODES-1))); do
        start_node $i
        sleep 1  # Stagger startup
    done
    
    if wait_for_nodes; then
        test_connectivity
        test_consensus
        show_status
        
        print_success "Multi-node test environment is ready!"
        print_status "Nodes are running in the background. Check logs in $LOG_DIR/"
    else
        print_error "Failed to start all nodes properly"
        exit 1
    fi
}

# Handle Ctrl+C
trap 'print_warning "Test setup interrupted by user"; exit 1' INT

# Run main function
main "$@"
