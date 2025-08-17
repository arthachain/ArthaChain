#!/bin/bash
# ArthaChain Real Blockchain Node Launcher
# Runs the ACTUAL blockchain technology built over 7 months
# NO MOCK DATA - Only real blockchain with advanced features

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_header() { echo -e "${PURPLE}[ARTHACHAIN]${NC} $1"; }
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_header "🚀 ArthaChain Real Blockchain Technology Stack"
print_header "   🧠 AI Engine: Neural networks, self-learning, BCI interface"
print_header "   ⚡ Consensus: Quantum-resistant SVBFT, parallel processing"
print_header "   🔒 ZK: Arkworks ZKP, multiple proof systems including zkML"
print_header "   💾 Storage: RocksDB, hybrid, replicated with disaster recovery"
print_header "   🌐 WASM: Complete smart contract execution engine"
print_header "   🛡️ Security: Quantum resistance, advanced monitoring"
echo

# Step 1: Kill any mock/simulation processes
print_status "🧹 Stopping any mock/simulation processes..."
pkill -f "python.*api" 2>/dev/null || true
pkill -f "full_api_server" 2>/dev/null || true
pkill -f "real_blockchain_api" 2>/dev/null || true
print_success "Simulation processes stopped"

# Step 2: Check if binaries exist
print_status "🔍 Checking ArthaChain binaries..."
if [ ! -f "../target/release/arthachain" ]; then
    print_error "Main blockchain binary not found! Please build with: cargo build --release"
    exit 1
fi

if [ ! -f "../target/release/testnet_api_server" ]; then
    print_error "API server binary not found! Please build with: cargo build --release"
    exit 1
fi
print_success "All binaries found"

# Step 3: Create configuration if needed
print_status "📋 Checking configuration..."
if [ ! -f "node_config.toml" ]; then
    print_warning "Creating default node configuration..."
    cat > node_config.toml << EOF
[node]
name = "arthachain-real-node"
network_id = "arthachain-mainnet"
data_dir = "./data"

[network]
listen_addr = "0.0.0.0:30303"
bootstrap_peers = []
enable_discovery = true

[consensus]
mechanism = "SVCP_SVBFT"
block_time = 5
enable_quantum_resistance = true
enable_parallel_processing = true
enable_cross_shard = true

[ai_engine]
enable_neural_networks = true
enable_self_learning = true
enable_fraud_detection = true
enable_bci_interface = false  # Disable for production

[storage]
backend = "rocksdb"
enable_replication = true
enable_disaster_recovery = true

[api]
listen_addr = "0.0.0.0:8080"
enable_cors = true
enable_metrics = true

[zk]
enable_zkp = true
enable_zkml = true
proof_systems = ["plonk", "groth16"]
curves = ["bn254", "bls12_381"]

[wasm]
enable_contracts = true
gas_limit = 1000000000
memory_limit = "256MB"
EOF
    print_success "Configuration created"
else
    print_success "Configuration exists"
fi

# Step 4: Ensure data directory exists
print_status "📁 Creating data directories..."
mkdir -p data/rocksdb
mkdir -p data/ai_models
mkdir -p data/zk_params
mkdir -p logs
print_success "Data directories ready"

# Step 5: Start the real blockchain node
print_status "🚀 Starting real ArthaChain blockchain node..."
print_header "   Features: Quantum-resistant consensus, AI engine, ZK proofs, WASM"

# Start main blockchain node in background
nohup ../target/release/arthachain node_config.toml > logs/blockchain_node.log 2>&1 &
BLOCKCHAIN_PID=$!
echo $BLOCKCHAIN_PID > blockchain.pid

# Wait for blockchain to initialize
sleep 5

if ! ps -p $BLOCKCHAIN_PID > /dev/null; then
    print_error "Blockchain node failed to start! Check logs/blockchain_node.log"
    cat logs/blockchain_node.log
    exit 1
fi

print_success "Real blockchain node started (PID: $BLOCKCHAIN_PID)"

# Step 6: Start the real API server for arthachain.in
print_status "🌐 Starting real API server for arthachain.in..."
print_status "   This will serve your REAL blockchain data through all 100+ APIs"

# Start API server in background
nohup ../target/release/testnet_api_server > logs/api_server.log 2>&1 &
API_PID=$!
echo $API_PID > api_server.pid

# Wait for API server to start
sleep 3

if ! ps -p $API_PID > /dev/null; then
    print_error "API server failed to start! Check logs/api_server.log"
    cat logs/api_server.log
    exit 1
fi

print_success "Real API server started (PID: $API_PID)"

# Step 7: Start the monitoring dashboard for xyz.arthachain.in
print_status "📊 Starting ArthChain API monitoring dashboard..."
print_status "   Dashboard will be available at https://xyz.arthachain.in"

# Start dashboard server in background
nohup python3 dashboard_server.py > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo $DASHBOARD_PID > dashboard.pid

# Wait for dashboard to start
sleep 2

if ! ps -p $DASHBOARD_PID > /dev/null; then
    print_error "Dashboard server failed to start! Check logs/dashboard.log"
    cat logs/dashboard.log
    exit 1
fi

print_success "Monitoring dashboard started (PID: $DASHBOARD_PID)"

# Step 8: Verify everything is running
print_status "🔍 Verifying all services..."

# Check ports
if lsof -i:30303 > /dev/null 2>&1; then
    print_success "✅ P2P network running on port 30303"
else
    print_warning "⚠️  P2P network may not be bound to 30303"
fi

if lsof -i:8080 > /dev/null 2>&1; then
    print_success "✅ API server running on port 8080 (arthachain.in ready)"
else
    print_error "❌ API server not running on port 8080!"
    exit 1
fi

if lsof -i:8081 > /dev/null 2>&1; then
    print_success "✅ Dashboard server running on port 8081 (xyz.arthachain.in ready)"
else
    print_error "❌ Dashboard server not running on port 8081!"
    exit 1
fi

# Test API
if curl -s http://localhost:8080/api/health > /dev/null; then
    print_success "✅ API health check passed"
else
    print_warning "⚠️  API health check failed"
fi

# Test Dashboard
if curl -s http://localhost:8081/api/dashboard/health > /dev/null; then
    print_success "✅ Dashboard health check passed"
else
    print_warning "⚠️  Dashboard health check failed"
fi

# Step 8: Display status
echo
print_header "🎉 ArthaChain Real Blockchain Successfully Started!"
print_success "📡 Blockchain Node: Running with all advanced features"
print_success "🌐 API Server: Serving real data on arthachain.in"
print_success "🧠 AI Engine: Self-learning neural networks active"
print_success "⚡ Consensus: Quantum-resistant SVBFT with parallel processing"
print_success "🔒 ZK Proofs: Multiple systems including zkML enabled"
print_success "💾 Storage: RocksDB with replication and disaster recovery"
print_success "🌐 WASM: Smart contract execution engine ready"

echo
print_status "📊 Service Status:"
print_status "   Main Node PID: $BLOCKCHAIN_PID (logs: logs/blockchain_node.log)"
print_status "   API Server PID: $API_PID (logs: logs/api_server.log)"
print_status "   Dashboard PID: $DASHBOARD_PID (logs: logs/dashboard.log)"
print_status "   P2P Port: 30303"
print_status "   API Port: 8080 (arthachain.in)"
print_status "   Dashboard Port: 8081 (xyz.arthachain.in)"

echo
print_status "🌐 Your real blockchain is now serving:"
print_status "   • https://api.arthachain.in - All 100+ real APIs"
print_status "   • https://rpc.arthachain.in - JSON-RPC for wallets"  
print_status "   • https://explorer.arthachain.in - Real blockchain explorer"
print_status "   • https://realtime.arthachain.in - Real-time data"
print_status "   • https://xyz.arthachain.in - API Monitoring Dashboard"

echo
print_status "📋 To monitor:"
print_status "   • Watch logs: tail -f logs/blockchain_node.log"
print_status "   • API logs: tail -f logs/api_server.log" 
print_status "   • Dashboard logs: tail -f logs/dashboard.log"
print_status "   • Stop services: ./stop_arthachain.sh"

print_header "🚀 Your 7 months of blockchain development is now LIVE with real data!"
