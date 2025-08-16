#!/bin/bash
# ArthaChain Quick Node Setup Script
# Usage: ./quick_node_setup.sh <node_number>
# Example: ./quick_node_setup.sh 2

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

NODE_ID=$1
BOOTSTRAP_IP="103.160.27.61"
BOOTSTRAP_PORT="30303"

if [ -z "$NODE_ID" ]; then
    print_error "Node ID is required!"
    echo "Usage: $0 <node_number>"
    echo "Example: $0 2 (for Node 2)"
    echo "Valid range: 2-10"
    exit 1
fi

if [ "$NODE_ID" -lt 2 ] || [ "$NODE_ID" -gt 10 ]; then
    print_error "Node ID must be between 2 and 10"
    exit 1
fi

print_status "🚀 Setting up ArthaChain Testnet Node $NODE_ID"
echo "================================================="

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v cargo &> /dev/null; then
    print_error "Cargo not found. Please install Rust first:"
    echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

if ! command -v git &> /dev/null; then
    print_error "Git not found. Please install Git first."
    exit 1
fi

print_success "Prerequisites check passed!"

# Create node configuration
CONFIG_FILE="node${NODE_ID}_config.toml"
DATA_DIR="testnet_data_node${NODE_ID}"

print_status "Creating configuration for Node $NODE_ID..."

cat > "$CONFIG_FILE" << EOF
# ArthaChain Distributed Testnet Configuration - Node $NODE_ID

[network]
network_id = "arthachain-testnet-1"
chain_id = 201766
name = "ArthaChain Testnet"

[node]
node_id = "testnet-node-$(printf "%02d" $NODE_ID)"
data_dir = "./$DATA_DIR"
log_level = "info"

[consensus]
algorithm = "SVBFT"
block_time = 5
max_block_size = 2097152
max_tx_pool_size = 10000
validator_set_size = 10
min_validator_stake = 1000000

[network_p2p]
listen_addr = "0.0.0.0:30303"
max_peers = 20
boot_nodes = [
    "/ip4/$BOOTSTRAP_IP/tcp/$BOOTSTRAP_PORT/p2p/bootstrap-node-1"
]

[rpc]
http_enabled = true
http_addr = "127.0.0.1"
http_port = 8545
http_cors_origins = ["*"]

ws_enabled = true
ws_addr = "127.0.0.1"
ws_port = 8546

[api]
enabled = true
addr = "127.0.0.1"
port = 8080
rate_limit = 1000

[ai_engine]
enabled = true
fraud_detection_model = "./models/fraud_detection.onnx"
identity_model = "./models/identity_verification.onnx"
inference_batch_size = 32
model_update_frequency = 1000

[storage]
backend = "hybrid"
rocksdb_path = "./$DATA_DIR/rocksdb"
rocksdb_max_files = 1000
memmap_path = "./$DATA_DIR/memmap"
memmap_size = 1073741824

[metrics]
enabled = true
prometheus_addr = "127.0.0.1"
prometheus_port = 9090
health_check_interval = 30

[security]
quantum_resistance = true
signature_algorithm = "Dilithium3"
encryption_algorithm = "Kyber768"
hash_algorithm = "BLAKE3"

[faucet]
enabled = false  # Only Node 1 (bootstrap) has faucet enabled

[genesis]
timestamp = 1704067200
initial_supply = 1000000000

[testing]
debug_mode = true
fast_mode = true
skip_verification = false
tx_tracing = true
EOF

print_success "Configuration created: $CONFIG_FILE"

# Create data directory
print_status "Creating data directory..."
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/rocksdb"
mkdir -p "$DATA_DIR/memmap"
mkdir -p "$DATA_DIR/logs"
print_success "Data directory created: $DATA_DIR"

# Check network connectivity to bootstrap
print_status "Testing connectivity to bootstrap node..."
if timeout 5 bash -c "</dev/tcp/$BOOTSTRAP_IP/22" 2>/dev/null; then
    print_success "Bootstrap node is reachable!"
else
    print_warning "Cannot reach bootstrap node. Check network connectivity."
    print_warning "Bootstrap IP: $BOOTSTRAP_IP"
    print_warning "Make sure port $BOOTSTRAP_PORT is open on both machines."
fi

# Build the project
print_status "Building ArthaChain (this may take a few minutes)..."
if cargo build --release --bin testnet_api_server; then
    print_success "Build completed successfully!"
else
    print_error "Build failed. Check the error messages above."
    exit 1
fi

# Create start script
START_SCRIPT="start_node${NODE_ID}.sh"
cat > "$START_SCRIPT" << EOF
#!/bin/bash
# Start ArthaChain Node $NODE_ID

echo "🚀 Starting ArthaChain Testnet Node $NODE_ID..."
echo "================================================"
echo "Config: $CONFIG_FILE"
echo "Data: $DATA_DIR"
echo "Bootstrap: $BOOTSTRAP_IP:$BOOTSTRAP_PORT"
echo "================================================"

# Start the node
cargo run --bin testnet_api_server --release -- --config $CONFIG_FILE

EOF

chmod +x "$START_SCRIPT"

# Create monitoring script
MONITOR_SCRIPT="monitor_node${NODE_ID}.sh"
cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
# Monitor ArthaChain Node $NODE_ID

echo "📊 ArthaChain Node $NODE_ID Status"
echo "=================================="

echo -e "\n🏥 Health Check:"
curl -s http://localhost:8080/api/health | jq . || echo "❌ API not responding"

echo -e "\n📈 Node Status:"
curl -s http://localhost:8080/api/status | jq . || echo "❌ Status not available"

echo -e "\n👥 Validators:"
curl -s http://localhost:8080/api/validators | jq '.total_count' || echo "❌ Validators not available"

echo -e "\n📦 Latest Block:"
curl -s http://localhost:8080/api/blocks/latest | jq '.height' || echo "❌ Block info not available"

echo -e "\n🔗 Network Info:"
echo "Local API: http://localhost:8080/api/status"
echo "Local RPC: http://localhost:8545"
echo "Bootstrap: $BOOTSTRAP_IP:$BOOTSTRAP_PORT"

EOF

chmod +x "$MONITOR_SCRIPT"

# Final instructions
echo ""
echo "🎉 Node $NODE_ID setup completed successfully!"
echo "=============================================="
echo ""
echo "📂 Files created:"
echo "  • Configuration: $CONFIG_FILE"
echo "  • Data directory: $DATA_DIR"
echo "  • Start script: $START_SCRIPT"
echo "  • Monitor script: $MONITOR_SCRIPT"
echo ""
echo "🚀 To start your node:"
echo "  ./$START_SCRIPT"
echo ""
echo "📊 To monitor your node:"
echo "  ./$MONITOR_SCRIPT"
echo ""
echo "🔗 Manual start command:"
echo "  cargo run --bin testnet_api_server --release -- --config $CONFIG_FILE"
echo ""
echo "⚡ After starting, your node will:"
echo "  • Connect to bootstrap node: $BOOTSTRAP_IP:$BOOTSTRAP_PORT"
echo "  • Join the testnet validator set"
echo "  • Start participating in consensus"
echo "  • Sync with the existing blockchain (current height: 140+)"
echo ""
echo "🌐 Test connectivity:"
echo "  curl http://localhost:8080/api/health"
echo ""
print_success "Ready to join the ArthaChain testnet! 🚀"
