# ⚙️ ArthaChain Node Setup Guide

**Learn how to run your own piece of the ArthaChain network!** This guide covers everything from simple full nodes to production validators.

## 🎯 What You'll Learn

- **🤔 Why run a node?** (Benefits and use cases)
- **🏗️ Node types** - Full, Validator, RPC, and Archive nodes
- **⚡ Quick setup** - Get running in 10 minutes  
- **🔧 Advanced configuration** - Production-ready setup
- **📱 Mobile validators** - Run validators on your phone!
- **🛡️ Security hardening** - Protect your node
- **📊 Monitoring** - Keep your node healthy
- **🆘 Troubleshooting** - Fix common issues

## 🤔 Why Run an ArthaChain Node?

Think of running a node like **hosting part of the internet** - you help make the network stronger and get benefits in return:

### 🎁 **Benefits for You**
```
🏃‍♂️ Run Your Own Node:
├── 💰 Earn validator rewards (no staking required!)
├── ⚡ Fastest possible API access (no rate limits!)
├── 🔒 Maximum privacy (your own private blockchain access)
├── 🌐 Help decentralize the network
├── 🎓 Learn how blockchain works under the hood
├── 🛡️ Don't depend on external services
└── 📊 Get detailed network insights
```

### 🌍 **Benefits for the Network**
```
🌐 Your Node Helps:
├── 🔗 More decentralization (harder to shut down)
├── 🚀 Better performance (more nodes = faster network)
├── 🛡️ Increased security (more validators = safer)
├── 🌍 Geographic distribution (global access)
└── 📈 Network growth (more participants)
```

## 🏗️ Node Types Explained

ArthaChain supports different types of nodes for different purposes:

### 🌐 **Full Node** (Most Common)
**What it does:** Stores the complete blockchain and validates all transactions

```
📦 Full Node Features:
├── 📚 Complete blockchain history
├── ✅ Validates all transactions and blocks
├── 🔄 Syncs with other nodes
├── 📡 Provides API access
├── 🛡️ Helps secure the network
└── 💾 Storage: ~100GB and growing
```

**Best for:** Developers, privacy-conscious users, API providers

### ⚖️ **Validator Node** (Earn Rewards!)
**What it does:** Everything a full node does, PLUS participates in consensus

```
⚖️ Validator Features:
├── 🗳️ Votes on new blocks
├── 💰 Earns staking rewards
├── 🏆 Can propose new blocks
├── 🎯 Requires minimum stake (1000 ARTHA)
├── ⏰ Must stay online 24/7
└── 🔒 Needs high security
```

**Best for:** People who want to earn rewards and help secure the network

### 📡 **RPC Node** (API Server)
**What it does:** Optimized for serving API requests to applications

```
📡 RPC Node Features:
├── ⚡ High-performance API serving
├── 🔗 Multiple connection support
├── 📊 Enhanced monitoring
├── 🚀 Load balancing ready
├── 💾 Can prune old data
└── 🏢 Perfect for businesses
```

**Best for:** Businesses, dApp developers, API providers

### 📚 **Archive Node** (Complete History)
**What it does:** Stores ALL historical data, never prunes anything

```
📚 Archive Node Features:
├── 🕰️ Complete historical data
├── 🔍 Can query any past state
├── 📊 Perfect for analytics
├── 🎓 Research and auditing
├── 💾 Storage: 500GB+ (grows fast)
└── 🔗 Blockchain explorers use these
```

**Best for:** Block explorers, researchers, auditors

### 📱 **Mobile Validator** (In Development)
**What it will do:** Specialized validator optimized for mobile devices!

```
📱 Mobile-Optimized Features (In Development):
├── 🔋 Battery-aware consensus algorithms
├── 📶 Data-efficient protocols
├── 🌡️ Thermal management system
├── 💤 Background processing optimization
├── 💰 Mobile reward mechanisms
└── 🌍 Decentralized mobile validation
```

**Status:** Core mobile-aware consensus implemented, full mobile apps in development

**Best for:** Future mobile-first blockchain validation

## ⚡ Quick Setup (10 Minutes)

Let's get you running a full node in under 10 minutes!

### 📋 **Prerequisites**

**Minimum Requirements:**
- **💻 Computer**: Linux, macOS, or Windows with WSL2
- **🧠 RAM**: 8GB (16GB recommended)
- **💾 Storage**: 100GB free space (SSD recommended)
- **🌐 Internet**: Stable broadband connection
- **⏰ Time**: 10 minutes of your time!

### 🐳 **Option 1: Docker (Easiest)**

**Step 1: Install Docker**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker

# macOS (install Docker Desktop from docker.com)
# Windows (install Docker Desktop from docker.com)
```

**Step 2: Download and Run**
```bash
# Create a directory for your node
mkdir arthachain-node && cd arthachain-node

# Download the Docker configuration
curl -O https://raw.githubusercontent.com/arthachain/blockchain/main/docker-compose.yml

# Start your node!
docker-compose up -d

# Check if it's working
curl http://localhost:8080/api/health
```

**Step 3: Monitor Your Node**
```bash
# View logs
docker-compose logs -f

# Check sync status
curl http://localhost:8080/api/status | jq '.sync_info.catching_up'

# Check peer connections
curl http://localhost:8080/api/network/peers | jq '.total_peers'
```

That's it! Your node is now running and syncing with the network! 🎉

### 🔧 **Option 2: Native Installation**

**Step 1: Install Prerequisites**

**Ubuntu/Debian:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y build-essential curl git jq

# Install Rust (ArthaChain is written in Rust)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup update
```

**macOS:**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install git jq

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**Step 2: Build ArthaChain**
```bash
# Clone the repository
git clone https://github.com/arthachain/blockchain.git
cd blockchain/blockchain_node

# Build the node (this takes 5-10 minutes)
cargo build --release

# The node binary is now at: target/release/testnet_api_server
```

**Step 3: Initialize and Run**
```bash
# Create data directory
mkdir -p ~/.arthachain/{config,data,logs}

# Download genesis file
curl -o ~/.arthachain/config/genesis.json \
  https://raw.githubusercontent.com/arthachain/networks/main/testnet/genesis.json

# Create basic configuration
cat > ~/.arthachain/config/config.toml << 'EOF'
# ArthaChain Node Configuration

# Network settings
chain_id = "artha-testnet-1"
moniker = "my-arthachain-node"

# API settings
[api]
enable = true
address = "tcp://0.0.0.0:8080"

# P2P settings
[p2p]
laddr = "tcp://0.0.0.0:26656"
seeds = "seed1.arthachain.com:26656,seed2.arthachain.com:26656"

# RPC settings  
[rpc]
laddr = "tcp://127.0.0.1:26657"

# Storage settings
[storage]
engine = "rocksdb"
path = "~/.arthachain/data"
EOF

# Start the node!
./target/release/testnet_api_server start --home ~/.arthachain
```

**Step 4: Verify It's Working**
```bash
# In another terminal, check status
curl http://localhost:8080/api/health

# Check if syncing
curl http://localhost:8080/api/status | jq '.sync_info'
```

🎉 **Congratulations! Your ArthaChain node is now running!**

## ⚖️ Validator Node Setup

Ready to earn rewards by securing the network? Let's set up a validator!

### 📋 **Validator Requirements**

**Technical Requirements:**
- **⚡ Server**: High uptime (99.9%+), reliable internet
- **🧠 RAM**: 16GB minimum, 32GB recommended
- **💾 Storage**: 200GB SSD minimum
- **🌐 Network**: Dedicated IP, stable connection
- **🔒 Security**: Firewall, DDoS protection

**Financial Requirements:**
- **💰 No Minimum Stake**: Join as validator without tokens!
- **💸 No Self-Bond Required**: Pure performance-based validation
- **⚡ Operating Costs**: ~$50-100/month for server

### 🚀 **Step-by-Step Validator Setup**

**Step 1: Set Up Full Node First**
Follow the "Quick Setup" section above to get a full node running and synced.

**Step 2: Create Validator Keys**
```bash
# Create a validator key (keep this VERY secure!)
./target/release/testnet_api_server keys add validator \
  --keyring-backend file \
  --home ~/.arthachain

# This will output your validator address - save it!
# Example: artha1validator123abc456def789...

# CRITICAL: Write down your mnemonic phrase on paper!
# This is the only way to recover your validator if something goes wrong
```

**Step 3: Fund Your Validator**
```bash
# Check your validator address balance
VALIDATOR_ADDRESS=$(./target/release/testnet_api_server keys show validator -a --keyring-backend file)
echo "Validator address: $VALIDATOR_ADDRESS"

# For testnet: get tokens from faucet
curl -X POST https://faucet.arthachain.online/api/faucet/request \
  -H "Content-Type: application/json" \
  -d "{\"address\": \"$VALIDATOR_ADDRESS\", \"amount\": \"10000000000000000000000\"}"

# For mainnet: send ARTHA tokens to this address
# You need at least 1,000 ARTHA to become a validator
```

**Step 4: Wait for Full Sync**
```bash
# Check sync status (must be false before creating validator)
curl http://localhost:8080/api/status | jq '.sync_info.catching_up'

# When this returns false, your node is fully synced
```

**Step 5: Create Validator**
```bash
# Create your validator
./target/release/testnet_api_server tx staking create-validator \
  --amount=1000000000000000000000 \
  --pubkey=$(./target/release/testnet_api_server tendermint show-validator --home ~/.arthachain) \
  --moniker="My Validator" \
  --chain-id=artha-testnet-1 \
  --commission-rate="0.10" \
  --commission-max-rate="0.20" \
  --commission-max-change-rate="0.01" \
  --min-self-delegation="1" \
  --gas="auto" \
  --gas-prices="0.001artha" \
  --from=validator \
  --keyring-backend=file \
  --home ~/.arthachain
```

**Step 6: Verify Your Validator**
```bash
# Check if your validator is active
./target/release/testnet_api_server query staking validator $VALIDATOR_ADDRESS \
  --home ~/.arthachain

# Check if you're in the active set
./target/release/testnet_api_server query tendermint-validator-set | grep $VALIDATOR_ADDRESS
```

🎉 **Congratulations! You're now a validator earning rewards!**

### 💰 **Validator Economics**

**Rewards:**
- **🎯 Block Rewards**: ~5-10% APY (varies based on network parameters)
- **💸 Transaction Fees**: Share of all network fees
- **🏆 Performance Bonus**: Extra rewards for high uptime

**Costs:**
- **💻 Server**: $50-200/month depending on specs
- **⚡ Electricity**: Minimal (especially for mobile validators)
- **🛡️ Security**: Optional DDoS protection, monitoring services

**Risks:**
- **⚖️ Slashing**: Lose part of stake for misbehavior (very rare if you follow rules)
- **📉 Downtime**: Miss rewards if offline too long
- **🔒 Key Loss**: Lose everything if you lose your validator keys

## 📡 RPC Node Configuration

Setting up a high-performance RPC node for serving applications.

### 🔧 **RPC Node Configuration**

```toml
# ~/.arthachain/config/config.toml for RPC nodes

[api]
enable = true
address = "tcp://0.0.0.0:8080"
max_open_connections = 2000
cors_allowed_origins = ["*"]
cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
cors_allowed_headers = ["*"]

[rpc]
laddr = "tcp://0.0.0.0:26657"
cors_allowed_origins = ["*"]
max_open_connections = 1800
max_subscription_clients = 100
max_subscriptions_per_client = 5
timeout_broadcast_tx_commit = "10s"

[grpc]
enable = true
address = "0.0.0.0:9090"

# Performance optimizations
[mempool]
size = 10000
cache_size = 20000

[consensus]
timeout_propose = "1s"
timeout_prevote = "500ms"
timeout_precommit = "500ms"
timeout_commit = "1s"

# Enable indexing for better API performance
[tx_index]
indexer = "kv"
```

### 🌐 **Reverse Proxy Setup (Nginx)**

```bash
# Install Nginx
sudo apt install nginx

# Configure reverse proxy
sudo tee /etc/nginx/sites-available/arthachain-rpc > /dev/null << 'EOF'
server {
    listen 80;
    server_name your-rpc-domain.com;
    
    # API endpoint
    location /api/ {
        proxy_pass http://localhost:8080/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS headers
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type, Authorization";
    }
    
    # WebSocket endpoint
    location /ws {
        proxy_pass http://localhost:8080/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # RPC endpoint
    location /rpc {
        proxy_pass http://localhost:26657;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/arthachain-rpc /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 🔒 **SSL Setup (Let's Encrypt)**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-rpc-domain.com

# Auto-renewal is set up automatically
```

## 📱 Mobile Validator Setup (In Development)

ArthaChain features mobile-optimized consensus algorithms designed for efficient mobile validation. Full mobile applications are currently in development.

### 📋 **Mobile Requirements**

**Device Requirements:**
- **📱 Smartphone**: Android 8+ or iOS 12+
- **🧠 RAM**: 4GB minimum, 6GB+ recommended
- **💾 Storage**: 32GB free space
- **🔋 Battery**: Good battery health
- **🌐 Internet**: Stable WiFi + mobile data backup

**Performance:**
- **⚡ Power Efficient**: Uses <5% battery per day
- **📶 Data Friendly**: <1GB/month data usage
- **🌡️ Cool Running**: Advanced thermal management
- **💤 Background Safe**: Works even when phone sleeps

### 🚀 **Mobile Setup Guide**

**Step 1: Install ArthaChain Mobile App**
```bash
# Android: Download from Google Play Store
# iOS: Download from Apple App Store
# Or build from source: https://github.com/arthachain/mobile-validator
```

**Step 2: Initialize Validator**
1. **Open the app** and tap "Become Validator"
2. **Create secure keys** - the app generates quantum-resistant keys
3. **Fund your validator** - send 1,000+ ARTHA to your validator address
4. **Set commission rate** - how much you charge delegators (5-20% is typical)
5. **Choose power mode** - balance performance vs battery life

**Step 3: Configure Settings**
```
🔧 Recommended Settings:
├── 🔋 Power Mode: "Balanced" (good performance + battery life)
├── 📶 Network: "WiFi + Cellular" (automatic failover)
├── 🌡️ Thermal: "Auto" (prevents overheating)
├── 💤 Background: "Always" (keep validating when screen off)
├── 🔔 Notifications: "Important Only" (missed blocks, rewards)
└── 📊 Monitoring: "Enabled" (track performance)
```

**Step 4: Start Validating**
1. **Tap "Start Validator"** - begins syncing
2. **Wait for sync** - usually 10-30 minutes
3. **Go online** - starts participating in consensus
4. **Earn rewards!** - see rewards accumulating in real-time

### 💡 **Mobile Best Practices**

**🔋 Battery Optimization:**
```
📱 Tips for Best Performance:
├── 🔌 Charge overnight (validator earns while you sleep)
├── ⚙️ Close unnecessary apps (more resources for validator)
├── 🌙 Use dark mode (saves battery)
├── 🔕 Disable non-essential notifications
├── 📶 Use WiFi when possible (less battery than cellular)
└── 🌡️ Keep phone cool (avoid direct sunlight)
```

**🛡️ Security Tips:**
```
🔒 Keep Your Validator Secure:
├── 🔐 Enable biometric authentication
├── 📱 Use screen lock/PIN
├── 💾 Backup your keys to cloud (encrypted)
├── 🔄 Regular app updates
├── 📵 Don't use on public WiFi for setup
└── 🚫 Never share your private keys
```

### 📊 **Mobile Validator Performance**

Real-world performance data from mobile validators:

```
📊 Performance Stats:
├── ⚡ TPS Contribution: 450+ (same as server validators)
├── 🎯 Uptime: 99.5%+ (with proper setup)
├── 🔋 Battery Usage: 3-8%/day (depending on settings)
├── 📶 Data Usage: 0.5-1.5GB/month
├── 🌡️ Heat Generation: Minimal (barely noticeable)
├── 💰 Rewards: Full validator rewards
└── 🏆 Network Security: Equal to server validators
```

## 🛡️ Security & Hardening

Security is CRITICAL for validators. Here's how to protect your node:

### 🔥 **Firewall Configuration**

```bash
# Install and configure UFW (Ubuntu)
sudo ufw enable

# Allow SSH (change port if using non-standard)
sudo ufw allow 22/tcp

# Allow P2P
sudo ufw allow 26656/tcp

# Allow RPC (only from trusted IPs for validators)
sudo ufw allow from 192.168.1.0/24 to any port 26657

# Allow API (only if serving public)
sudo ufw allow 8080/tcp

# Block everything else
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Check status
sudo ufw status verbose
```

### 🔐 **SSH Hardening**

```bash
# Generate strong SSH key
ssh-keygen -t ed25519 -f ~/.ssh/arthachain_validator

# Copy to server
ssh-copy-id -i ~/.ssh/arthachain_validator user@your-server

# Harden SSH config
sudo tee -a /etc/ssh/sshd_config << 'EOF'
# ArthaChain Validator SSH Security
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
Port 2222
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
EOF

sudo systemctl restart ssh
```

### 🛡️ **DDoS Protection**

```bash
# Install Fail2Ban
sudo apt install fail2ban

# Configure for ArthaChain
sudo tee /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = 2222

[arthachain-p2p]
enabled = true
port = 26656
filter = arthachain-p2p
logpath = ~/.arthachain/logs/node.log
maxretry = 10
EOF

sudo systemctl restart fail2ban
```

### 🔑 **Key Management**

```bash
# Backup validator keys (DO THIS!)
cp ~/.arthachain/config/priv_validator_key.json ~/validator_key_backup.json
cp ~/.arthachain/config/node_key.json ~/node_key_backup.json

# Encrypt backups
gpg --symmetric --cipher-algo AES256 validator_key_backup.json
gpg --symmetric --cipher-algo AES256 node_key_backup.json

# Store encrypted backups in multiple secure locations:
# - Encrypted cloud storage (Google Drive, Dropbox)
# - Hardware wallet
# - Safe deposit box
# - Trusted family member
```

## 📊 Monitoring & Maintenance

Keep your node healthy with proper monitoring:

### 📈 **Prometheus + Grafana Setup**

```bash
# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvfz prometheus-2.40.0.linux-amd64.tar.gz
sudo mv prometheus-2.40.0.linux-amd64/prometheus /usr/local/bin/

# Configure Prometheus
sudo tee /etc/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'arthachain'
    static_configs:
      - targets: ['localhost:26660']  # Tendermint metrics
      - targets: ['localhost:8080']   # API metrics

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']  # Node exporter
EOF

# Install Grafana
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana

# Start services
sudo systemctl start prometheus grafana-server
sudo systemctl enable prometheus grafana-server
```

### 🚨 **Alerting Setup**

```bash
# Install AlertManager
wget https://github.com/prometheus/alertmanager/releases/download/v0.25.0/alertmanager-0.25.0.linux-amd64.tar.gz
tar xvfz alertmanager-0.25.0.linux-amd64.tar.gz
sudo mv alertmanager-0.25.0.linux-amd64/alertmanager /usr/local/bin/

# Configure alerts
sudo tee /etc/alertmanager/alertmanager.yml << 'EOF'
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'your-email@gmail.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'your-email@gmail.com'
    subject: 'ArthaChain Validator Alert'
    body: |
      Alert: {{ .GroupLabels.alertname }}
      Instance: {{ .CommonLabels.instance }}
      Description: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
EOF
```

### 📋 **Essential Monitoring Commands**

```bash
# Check node status
curl http://localhost:8080/api/status | jq

# Check sync progress
curl http://localhost:8080/api/status | jq '.sync_info.catching_up'

# Check peer count
curl http://localhost:8080/api/network/peers | jq '.total_peers'

# Check latest block
curl http://localhost:8080/api/blocks/latest | jq '.height'

# Check validator status (if you're a validator)
curl http://localhost:8080/api/validators | jq '.validators[] | select(.address == "your-validator-address")'

# Check mempool size
curl http://localhost:8080/api/status | jq '.sync_info.mempool_size'

# View logs
sudo journalctl -u arthachain -f --lines=100

# Check disk usage
df -h ~/.arthachain/data

# Check system resources
htop
```

### 🔄 **Automated Backup Script**

```bash
#!/bin/bash
# ~/backup_validator.sh

BACKUP_DIR="~/arthachain_backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup configuration and keys
tar -czf $BACKUP_DIR/config_backup_$DATE.tar.gz ~/.arthachain/config/

# Backup recent data (last 1000 blocks)
tar -czf $BACKUP_DIR/data_backup_$DATE.tar.gz ~/.arthachain/data/ --newer-mtime="1 day ago"

# Upload to cloud storage (example: rsync to remote server)
rsync -av $BACKUP_DIR/ backup-server:~/arthachain_backups/

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

```bash
# Make it executable and add to cron
chmod +x ~/backup_validator.sh

# Add to crontab (daily backup at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * ~/backup_validator.sh") | crontab -
```

## 🆘 Troubleshooting Common Issues

### ❌ **Node Won't Start**

**Problem:** `Error: failed to start node`

**Solutions:**
```bash
# Check port conflicts
sudo netstat -tlnp | grep :8080
sudo netstat -tlnp | grep :26656

# Check disk space
df -h

# Check configuration
./target/release/testnet_api_server validate --home ~/.arthachain

# Reset if corrupted (WARNING: loses all data)
./target/release/testnet_api_server unsafe-reset-all --home ~/.arthachain
```

### ❌ **Node Not Syncing**

**Problem:** Node stuck on old block, `catching_up: true` never changes

**Solutions:**
```bash
# Check peers
curl http://localhost:8080/api/network/peers | jq '.total_peers'

# If 0 peers, check seeds/firewall
# Edit ~/.arthachain/config/config.toml and add more seeds:
seeds = "seed1.arthachain.com:26656,seed2.arthachain.com:26656,seed3.arthachain.com:26656"

# Restart node
sudo systemctl restart arthachain

# Or use state sync for faster sync
./target/release/testnet_api_server config set statesync.enable true
```

### ❌ **Validator Not Signing**

**Problem:** Validator not participating, missing blocks

**Solutions:**
```bash
# Check if validator keys are correct
./target/release/testnet_api_server tendermint show-validator --home ~/.arthachain

# Check if validator is jailed
./target/release/testnet_api_server query staking validator $(./target/release/testnet_api_server keys show validator --bech val -a) --home ~/.arthachain

# If jailed, unjail:
./target/release/testnet_api_server tx slashing unjail \
  --from validator \
  --keyring-backend file \
  --home ~/.arthachain

# Check time sync (critical for validators)
timedatectl status
# If not synced:
sudo timedatectl set-ntp true
```

### ❌ **High Memory Usage**

**Problem:** Node using too much RAM, system slow

**Solutions:**
```bash
# Check current usage
free -h
ps aux | grep arthachain

# Reduce cache sizes in config.toml:
[mempool]
cache_size = 5000
size = 5000

[consensus]
timeout_propose = "3s"
timeout_prevote = "1s"
timeout_precommit = "1s"

# Enable state pruning
[pruning]
pruning = "default"
pruning-keep-recent = "100"
pruning-keep-every = "0"
pruning-interval = "10"

# Restart node
sudo systemctl restart arthachain
```

### ❌ **Network Connection Issues**

**Problem:** Can't connect to peers, firewall issues

**Solutions:**
```bash
# Test P2P connectivity
telnet seed1.arthachain.com 26656

# Check firewall
sudo ufw status

# Temporarily disable firewall to test
sudo ufw disable
# Try syncing, then re-enable:
sudo ufw enable

# Check if ISP blocks ports
nmap -p 26656 your-public-ip

# Use port forwarding if behind NAT
# Configure router to forward port 26656 to your node
```

### ❌ **Mobile Validator Issues**

**Problem:** Mobile validator performing poorly

**Solutions:**
```
📱 Mobile Troubleshooting:
├── 🔋 Battery: Ensure phone is charging or has >20% battery
├── 📶 Network: Switch to WiFi if on cellular (better stability)
├── 🌡️ Heat: Move to cooler location, close other apps
├── 💾 Storage: Clear phone storage, ensure 10GB+ free space
├── 🔄 App: Force-close and restart validator app
├── 📱 OS: Restart phone if performance degrades
└── ⚙️ Settings: Lower performance mode if overheating
```

## 📚 Advanced Topics

### 🔄 **State Sync (Fast Sync)**

Instead of downloading the entire blockchain, use state sync to get up and running in minutes:

```bash
# Enable state sync
./target/release/testnet_api_server config set statesync.enable true
./target/release/testnet_api_server config set statesync.rpc_servers "https://rpc1.arthachain.com:443,https://rpc2.arthachain.com:443"

# Get trust height and hash
LATEST_HEIGHT=$(curl -s https://rpc1.arthachain.com/block | jq -r .result.block.header.height)
TRUST_HEIGHT=$((LATEST_HEIGHT - 2000))
TRUST_HASH=$(curl -s "https://rpc1.arthachain.com/block?height=$TRUST_HEIGHT" | jq -r .result.block_id.hash)

./target/release/testnet_api_server config set statesync.trust_height $TRUST_HEIGHT
./target/release/testnet_api_server config set statesync.trust_hash $TRUST_HASH

# Start node - it will state sync instead of syncing from genesis
./target/release/testnet_api_server start --home ~/.arthachain
```

### 🏢 **Enterprise Deployment**

For production/enterprise deployments:

```bash
# Use systemd for auto-restart and logging
sudo tee /etc/systemd/system/arthachain.service << 'EOF'
[Unit]
Description=ArthaChain Node
After=network-online.target

[Service]
Type=exec
User=arthachain
Group=arthachain
ExecStart=/usr/local/bin/arthachain start --home /var/lib/arthachain
Restart=on-failure
RestartSec=3
LimitNOFILE=65535
LimitNPROC=4096

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/arthachain

[Install]
WantedBy=multi-user.target
EOF

# Create dedicated user
sudo useradd -r -s /bin/false arthachain

# Set up directories
sudo mkdir -p /var/lib/arthachain
sudo chown arthachain:arthachain /var/lib/arthachain

# Enable service
sudo systemctl daemon-reload
sudo systemctl enable arthachain
sudo systemctl start arthachain
```

### 🌍 **Multi-Region Setup**

For global availability:

```bash
# Set up nodes in multiple regions
# Region 1: US East
# Region 2: Europe  
# Region 3: Asia

# Use load balancer (HAProxy example)
sudo tee /etc/haproxy/haproxy.cfg << 'EOF'
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend arthachain_api
    bind *:8080
    default_backend arthachain_nodes

backend arthachain_nodes
    balance roundrobin
    server us-east us-east.arthachain.internal:8080 check
    server europe europe.arthachain.internal:8080 check
    server asia asia.arthachain.internal:8080 check
EOF
```

## 🎯 What's Next?

Congratulations! You now know how to run every type of ArthaChain node. Here are your next steps:

### 👶 **Just Started?**
1. **[📱 Try the Mobile Validator](https://play.google.com/store/apps/details?id=com.arthachain.validator)** - Easiest way to start earning
2. **[📊 Set up monitoring](https://grafana.arthachain.online)** - Keep your node healthy
3. **[💬 Join the Validator Chat](https://discord.gg/arthachain-validators)** - Connect with other validators

### 👨‍💻 **Ready for Production?**
1. **[🔐 Security Hardening Guide](./security.md)** - Protect your validator
2. **[📊 Advanced Monitoring](./monitoring.md)** - Professional-grade monitoring
3. **[🏢 Enterprise Features](./enterprise.md)** - Multi-region, high availability

### 🚀 **Want to Contribute?**
1. **[🛠️ Developer Guide](./contributing.md)** - Help improve ArthaChain
2. **[🧪 Testnet Participation](./testnet.md)** - Test new features
3. **[📚 Documentation](./docs-contributing.md)** - Help improve these docs

## 🌐 Resources & Support

### 📚 **Additional Resources**
- **[🎮 Node Management Dashboard](https://nodes.arthachain.online)** - Manage all your nodes
- **[📊 Network Explorer](https://explorer.arthachain.com)** - View network status
- **[💰 Staking Calculator](https://staking.arthachain.online)** - Calculate validator rewards
- **[📱 Mobile App](https://apps.arthachain.online)** - Mobile validator app

### 💬 **Community Support**
- **[💬 Discord](https://discord.gg/arthachain)** - General community
- **[⚖️ Validator Discord](https://discord.gg/arthachain-validators)** - Validator-specific chat
- **[📱 Telegram](https://t.me/arthachain_nodes)** - Node operators group
- **[🐙 GitHub](https://github.com/arthachain/blockchain)** - Technical discussions

### 📧 **Direct Support**
- **⚙️ Node Questions**: [nodes@arthachain.com](mailto:nodes@arthachain.com)
- **⚖️ Validator Support**: [validators@arthachain.com](mailto:validators@arthachain.com)
- **🏢 Enterprise**: [enterprise@arthachain.com](mailto:enterprise@arthachain.com)
- **🚨 Emergency**: [emergency@arthachain.com](mailto:emergency@arthachain.com)

---

**🎯 Next**: [🔐 Security Best Practices](./security.md) →

**💬 Questions?** Join our [Validator Discord](https://discord.gg/arthachain-validators) - we're here to help you succeed! 