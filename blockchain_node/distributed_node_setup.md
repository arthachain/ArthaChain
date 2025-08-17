# 🌐 ArthaChain 10-Node Distributed Testnet Setup

## 🎯 Overview
Setting up a 10-node testnet across different machines with Node 1 as the bootstrap.

### Current Bootstrap Node (Node 1)
- **IP Address**: 103.160.27.61
- **P2P Port**: 30303
- **API**: https://api.arthachain.in
- **Current Height**: 141+ blocks
- **Status**: ✅ Running

---

## 📋 Prerequisites for Each New Machine

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL
- **Memory**: 4GB+ RAM
- **Storage**: 20GB+ free space
- **Network**: Open port 30303 for P2P communication

### Software Requirements
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Git
# Ubuntu/Debian: sudo apt install git
# CentOS/RHEL: sudo yum install git
# macOS: git (comes with Xcode tools)

# Verify installations
rustc --version
cargo --version
git --version
```

---

## 🚀 Step-by-Step Setup for Each New Node

### Step 1: Clone Repository on New Machine
```bash
# Clone the ArthaChain repository
git clone <YOUR_REPOSITORY_URL>
cd ArthaChain/blockchain_node

# Or copy the entire blockchain_node folder to the new machine
```

### Step 2: Create Node Configuration

For **Node 2** (and adapt for nodes 3-10):
```bash
# Create node2_config.toml
cat > node2_config.toml << 'EOF'
# ArthaChain Distributed Testnet Configuration - Node 2

[network]
network_id = "arthachain-testnet-1"
chain_id = 201766
name = "ArthaChain Testnet"

[node]
node_id = "testnet-node-02"
data_dir = "./testnet_data_node2"
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
    "/ip4/103.160.27.61/tcp/30303/p2p/bootstrap-node-1"
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
rocksdb_path = "./testnet_data_node2/rocksdb"
rocksdb_max_files = 1000
memmap_path = "./testnet_data_node2/memmap"
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
enabled = false  # Only Node 1 has faucet enabled

[genesis]
timestamp = 1704067200
initial_supply = 1000000000

[testing]
debug_mode = true
fast_mode = true
skip_verification = false
tx_tracing = true
EOF
```

### Step 3: Build and Run
```bash
# Build the project
cargo build --release --bin testnet_api_server

# Create data directory
mkdir -p testnet_data_node2

# Start the node
cargo run --bin testnet_api_server --release -- --config node2_config.toml
```

---

## 📝 Configuration Templates for All Nodes

### Node 3 Configuration
```toml
[node]
node_id = "testnet-node-03"
data_dir = "./testnet_data_node3"

[storage]
rocksdb_path = "./testnet_data_node3/rocksdb"
memmap_path = "./testnet_data_node3/memmap"
```

### Node 4 Configuration
```toml
[node]
node_id = "testnet-node-04"
data_dir = "./testnet_data_node4"

[storage]
rocksdb_path = "./testnet_data_node4/rocksdb"
memmap_path = "./testnet_data_node4/memmap"
```

### Continue for Nodes 5-10...
Follow the same pattern, incrementing:
- `node_id`: "testnet-node-05", "testnet-node-06", etc.
- `data_dir`: "./testnet_data_node5", "./testnet_data_node6", etc.
- Storage paths accordingly

---

## 🔧 Automated Setup Script

Create this script on each new machine:

```bash
#!/bin/bash
# setup_node.sh - Quick node setup script

NODE_ID=$1
if [ -z "$NODE_ID" ]; then
    echo "Usage: ./setup_node.sh <node_number>"
    echo "Example: ./setup_node.sh 2"
    exit 1
fi

echo "🚀 Setting up ArthaChain Node $NODE_ID..."

# Create configuration
cat > "node${NODE_ID}_config.toml" << EOF
[network]
network_id = "arthachain-testnet-1"
chain_id = 201766
name = "ArthaChain Testnet"

[node]
node_id = "testnet-node-$(printf "%02d" $NODE_ID)"
data_dir = "./testnet_data_node${NODE_ID}"
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
    "/ip4/103.160.27.61/tcp/30303/p2p/bootstrap-node-1"
]

[rpc]
http_enabled = true
http_addr = "127.0.0.1"
http_port = 8545
http_cors_origins = ["*"]

[api]
enabled = true
addr = "127.0.0.1"
port = 8080

[storage]
backend = "hybrid"
rocksdb_path = "./testnet_data_node${NODE_ID}/rocksdb"
memmap_path = "./testnet_data_node${NODE_ID}/memmap"

[faucet]
enabled = false
EOF

# Create data directory
mkdir -p "testnet_data_node${NODE_ID}"

echo "✅ Node $NODE_ID configuration created!"
echo "📂 Config file: node${NODE_ID}_config.toml"
echo "📁 Data directory: testnet_data_node${NODE_ID}"
echo ""
echo "🚀 To start the node:"
echo "cargo run --bin testnet_api_server --release -- --config node${NODE_ID}_config.toml"
```

---

## 🌐 Network Configuration

### Firewall Settings
Each machine must allow:
```bash
# Ubuntu/Debian
sudo ufw allow 30303/tcp
sudo ufw allow 8080/tcp  # If hosting APIs
sudo ufw allow 8545/tcp  # RPC port

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=30303/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=8545/tcp
sudo firewall-cmd --reload
```

### Cloud Provider Settings
- **AWS**: Configure Security Groups
- **GCP**: Configure Firewall Rules
- **Azure**: Configure Network Security Groups
- **DigitalOcean**: Configure Firewall rules

---

## 🔍 Verification Commands

After starting each node:

```bash
# Check if node is running
curl http://localhost:8080/api/health

# Check peer connections
curl http://localhost:8080/api/status

# Check validator set
curl http://localhost:8080/api/validators
```

---

## 📊 Monitoring All Nodes

From your bootstrap node (Node 1), monitor the network:

```bash
# Check total validators
curl https://api.arthachain.in/api/validators

# Check network stats
curl https://api.arthachain.in/api/stats

# Check consensus status
curl https://api.arthachain.in/api/consensus/status
```

---

## 🎯 Final Network Topology

```
Node 1 (Bootstrap) ←→ Node 2
       ↕                ↕
Node 10 ←→ [Network] ←→ Node 3
       ↕                ↕
Node 9  ←→ ......... ←→ Node 4
       ↕                ↕
Node 8  ←→ Node 6 ←→ Node 5
              ↕
            Node 7
```

All nodes connect to all other nodes in a mesh network for maximum resilience.

---

## 🆘 Troubleshooting

### Common Issues:

1. **"Connection refused"**
   - Check firewall settings
   - Verify bootstrap node IP is correct
   - Ensure port 30303 is open

2. **"No peers connected"**
   - Wait 1-2 minutes for peer discovery
   - Check network connectivity
   - Verify boot_nodes configuration

3. **"Build failed"**
   - Update Rust: `rustup update`
   - Clean build: `cargo clean && cargo build --release`

### Support Commands:
```bash
# View logs
tail -f ~/.cargo/registry/src/*/arthachain*/logs/node.log

# Check port availability
netstat -tulpn | grep 30303

# Test connectivity to bootstrap
telnet 103.160.27.61 30303
```

---

## ✅ Success Criteria

Your 10-node testnet is successful when:
- [ ] All 10 nodes show in validator set
- [ ] Peer count is 9+ on each node
- [ ] Blocks are being produced consistently
- [ ] Transactions propagate across all nodes
- [ ] Consensus works with majority participation

**🎉 You'll have a fully distributed, production-like testnet!**
