#!/bin/bash

# 🚀 ArthaChain BEAST MODE Deployment Script
# Optimized for 32-core CPU + RTX 4090 GPU + 50GB RAM + 1TB Storage

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🔥 ArthaChain BEAST MODE Deployment 🔥${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "${CYAN}32 Cores | RTX 4090 | 50GB RAM | 1TB Storage${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me)
echo -e "${YELLOW}🌐 Detected public IP: ${PUBLIC_IP}${NC}"

# Update system with parallel processing
echo -e "${YELLOW}📦 Updating system (using all 32 cores)...${NC}"
export MAKEFLAGS="-j32"
apt update && apt upgrade -y

# Install all dependencies in parallel
echo -e "${YELLOW}🛠️ Installing dependencies...${NC}"
apt install -y \
    build-essential \
    clang \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    htop \
    nvtop \
    unzip \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release

# Install CUDA for RTX 4090 support
echo -e "${YELLOW}🎮 Installing CUDA for RTX 4090...${NC}"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-3 nvidia-driver-535

# Install Rust with optimal settings
echo -e "${YELLOW}🦀 Installing Rust (optimized for 32 cores)...${NC}"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Set optimal Rust compilation flags
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort"
export CARGO_BUILD_JOBS=32

# Install Python ML libraries for AI engine
echo -e "${YELLOW}🧠 Installing Python ML libraries...${NC}"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install onnx onnxruntime-gpu numpy scikit-learn pandas

# Setup persistent storage
echo -e "${YELLOW}💾 Setting up persistent storage...${NC}"
mkdir -p /mnt/data/arthachain
mkdir -p /mnt/data/arthachain/blockchain_data
mkdir -p /mnt/data/arthachain/logs
mkdir -p /mnt/data/arthachain/config

# Create symbolic links for optimal storage usage
ln -sf /mnt/data/arthachain/blockchain_data /app/blockchain_data
ln -sf /mnt/data/arthachain/logs /app/logs

# Clone ArthaChain repository
echo -e "${YELLOW}⬇️ Cloning ArthaChain...${NC}"
cd /root
git clone https://github.com/DiigooSai/ArthaChain.git arthachain
cd arthachain

# Create optimized build configuration
echo -e "${YELLOW}⚙️ Creating BEAST MODE configuration...${NC}"
cat > blockchain_node/beast_config.toml << EOF
[network]
network_id = "arthachain-mainnet-1"
chain_id = 1337
name = "ArthaChain Beast Mode"

[node]
node_id = "beast-node-${RANDOM}"
data_dir = "/mnt/data/arthachain/blockchain_data"
log_level = "info"
public_ip = "${PUBLIC_IP}"

[consensus]
algorithm = "SVBFT"
block_time = 1
max_block_size = 8388608  # 8MB blocks for high TPS
validator_set_size = 8
parallel_execution_threads = 24  # Leave 8 cores for system

[network_p2p]
listen_addr = "0.0.0.0:30303"
external_addr = "/ip4/${PUBLIC_IP}/tcp/30303"
max_peers = 200
enable_nat = true
enable_upnp = true

[rpc]
http_enabled = true
http_addr = "0.0.0.0"
http_port = 8545
http_cors_origins = ["*"]
ws_enabled = true
ws_addr = "0.0.0.0"
ws_port = 8546

[api]
enabled = true
addr = "0.0.0.0"
port = 3000
rate_limit = 10000  # Higher limit for beast mode
max_connections_per_ip = 50

[metrics]
enabled = true
prometheus_addr = "0.0.0.0"
prometheus_port = 9090

[storage]
backend = "hybrid"
rocksdb_max_files = 10000
memmap_size = 8589934592  # 8GB memory mapping
cache_size = 4294967296   # 4GB cache
write_buffer_size = 268435456  # 256MB write buffer

[ai_engine]
enabled = true
device = "cuda"
batch_size = 64  # Larger batches for RTX 4090
inference_threads = 4
fraud_detection_model = "./models/fraud_detection.onnx"

[security]
quantum_resistance = true
signature_algorithm = "Dilithium3"
encryption_algorithm = "Kyber768"

[performance]
max_tx_pool_size = 100000  # Massive pool for high TPS
gc_interval = 300
checkpoint_interval = 1000
EOF

# Build ArthaChain with maximum optimization
echo -e "${YELLOW}🔨 Building ArthaChain (BEAST MODE - using all 32 cores)...${NC}"
export CARGO_BUILD_TARGET_DIR="/tmp/arthachain_build"
cargo build --release --config beast_config.toml

# Copy binaries to optimal locations
cp target/release/arthachain /usr/local/bin/
chmod +x /usr/local/bin/arthachain

# Create systemd service
echo -e "${YELLOW}🔧 Creating systemd service...${NC}"
cat > /etc/systemd/system/arthachain-beast.service << EOF
[Unit]
Description=ArthaChain Beast Mode Node
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/arthachain
ExecStart=/usr/local/bin/arthachain run --config blockchain_node/beast_config.toml
Restart=always
RestartSec=10
LimitNOFILE=1000000
Environment="RUST_LOG=info"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="RAYON_NUM_THREADS=24"

[Install]
WantedBy=multi-user.target
EOF

# Create monitoring script
cat > /usr/local/bin/arthachain-monitor.sh << 'EOF'
#!/bin/bash
echo "🔥 ArthaChain Beast Mode Status 🔥"
echo "================================="
echo "📊 System Resources:"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
echo ""
echo "🔗 Blockchain Status:"
curl -s http://localhost:3000/api/stats | jq '.' 2>/dev/null || echo "Node starting..."
echo ""
echo "⚡ Performance Metrics:"
curl -s http://localhost:9090/metrics | grep arthachain | head -10
EOF
chmod +x /usr/local/bin/arthachain-monitor.sh

# Start the beast!
echo -e "${YELLOW}🚀 Starting ArthaChain Beast Mode...${NC}"
systemctl daemon-reload
systemctl enable arthachain-beast
systemctl start arthachain-beast

# Wait for startup
echo -e "${YELLOW}⏳ Waiting for node to initialize...${NC}"
sleep 30

# Health check
echo -e "${YELLOW}🔍 Performing health check...${NC}"
if curl -f -s http://localhost:3000/api/health > /dev/null; then
    echo -e "${GREEN}✅ ArthaChain Beast Mode is ALIVE!${NC}"
else
    echo -e "${RED}❌ Health check failed, checking logs...${NC}"
    journalctl -u arthachain-beast --no-pager -n 20
fi

# Display final status
echo -e "${PURPLE}=================================${NC}"
echo -e "${GREEN}🎉 ARTHACHAIN BEAST MODE DEPLOYED! 🎉${NC}"
echo -e "${PURPLE}=================================${NC}"
echo ""
echo -e "${CYAN}🌐 Access Points:${NC}"
echo -e "${WHITE}REST API:    http://${PUBLIC_IP}:3000${NC}"
echo -e "${WHITE}JSON-RPC:    http://${PUBLIC_IP}:8545${NC}"
echo -e "${WHITE}WebSocket:   ws://${PUBLIC_IP}:8546${NC}"
echo -e "${WHITE}Metrics:     http://${PUBLIC_IP}:9090${NC}"
echo ""
echo -e "${CYAN}🔧 Useful Commands:${NC}"
echo -e "${WHITE}Monitor:     /usr/local/bin/arthachain-monitor.sh${NC}"
echo -e "${WHITE}Logs:        journalctl -u arthachain-beast -f${NC}"
echo -e "${WHITE}Status:      systemctl status arthachain-beast${NC}"
echo -e "${WHITE}GPU Status:  nvidia-smi${NC}"
echo ""
echo -e "${CYAN}📱 MetaMask Config:${NC}"
echo -e "${WHITE}Network:     ArthaChain Beast Mode${NC}"
echo -e "${WHITE}RPC URL:     http://${PUBLIC_IP}:8545${NC}"
echo -e "${WHITE}Chain ID:    1337${NC}"
echo -e "${WHITE}Symbol:      ARTHA${NC}"
echo ""
echo -e "${GREEN}🔥 Your blockchain beast is ready to handle MASSIVE TPS! 🔥${NC}"
