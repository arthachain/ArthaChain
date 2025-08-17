#!/bin/bash
# ArthChain Validator Node - Simple One-Command Setup
# Similar to Shardeum's approach

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_logo() {
    clear
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                                                          ║"
    echo "║            🚀 ArthChain Validator Setup 🚀               ║"
    echo "║                                                          ║"
    echo "║         Complete Environment → Validator Setup          ║"
    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

# Progress animation
show_progress() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Loading animation
loading_animation() {
    local message="$1"
    local duration="$2"
    echo -n "$message"
    for i in $(seq 1 $duration); do
        echo -n "."
        sleep 0.5
    done
    echo " ✅"
}

# Generate wallet
generate_wallet() {
    echo -e "${BLUE}💰 Generating ArthChain Validator Wallet...${NC}"
    
    # Generate random private key
    PRIVATE_KEY=$(openssl rand -hex 32)
    
    # Generate wallet address (simplified)
    WALLET_ADDRESS="0x$(echo -n "$PRIVATE_KEY" | sha256sum | cut -c1-40)"
    
    echo -e "${GREEN}✅ Wallet Generated!${NC}"
    echo -e "${YELLOW}📋 SAVE THESE CREDENTIALS:${NC}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  🔐 Wallet Address: $WALLET_ADDRESS      │"
    echo "│  🗝️  Private Key: $PRIVATE_KEY │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""
    echo -e "${RED}⚠️  IMPORTANT: Save your private key securely!${NC}"
    echo -e "${YELLOW}Press ENTER to continue after saving your wallet info...${NC}"
    read -r
}

ask_permission() {
    echo -e "${YELLOW}Join the ArthChain network as a validator? (y/n):${NC}"
    read -r PERMISSION
    if [[ ! "$PERMISSION" =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
}

# Smart defaults with minimal questions
set_defaults() {
    # Auto-detect external IP
    EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s icanhazip.com 2>/dev/null || echo "127.0.0.1")
    
    # Use standard ports
    P2P_PORT=30303
    API_PORT=8080
    DASHBOARD_PORT=8080
    
    # Default paths
    INSTALL_PATH="$HOME/.arthachain"
    
    # Enable dashboard by default
    ENABLE_DASHBOARD=true
    
    # Generate secure random password
    DASHBOARD_PASSWORD=$(openssl rand -base64 12 2>/dev/null || echo "arthachain$(date +%s)")
    
    echo -e "${BLUE}🔧 Using smart defaults:${NC}"
    echo "   📁 Install Path: $INSTALL_PATH"
    echo "   🌐 External IP: $EXTERNAL_IP"
    echo "   📡 P2P Port: $P2P_PORT"
    echo "   🔗 API Port: $API_PORT"
    echo "   📊 Dashboard: http://localhost:$DASHBOARD_PORT"
    echo ""
}

ask_custom_config() {
    echo -e "${YELLOW}Use custom configuration? (n=use defaults, y=customize):${NC}"
    read -r CUSTOM_CONFIG
    
    if [[ "$CUSTOM_CONFIG" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}P2P port (default 30303):${NC}"
        read -r CUSTOM_P2P_PORT
        P2P_PORT=${CUSTOM_P2P_PORT:-30303}
        
        echo -e "${YELLOW}API port (default 8080):${NC}"
        read -r CUSTOM_API_PORT
        API_PORT=${CUSTOM_API_PORT:-8080}
        
        echo -e "${YELLOW}Dashboard password (leave empty for auto-generated):${NC}"
        read -s CUSTOM_PASSWORD
        if [ ! -z "$CUSTOM_PASSWORD" ]; then
            DASHBOARD_PASSWORD="$CUSTOM_PASSWORD"
        fi
        echo ""
    fi
}

install_dependencies() {
    echo -e "${BLUE}📦 Setting up environment...${NC}"
    loading_animation "Preparing system" 3
    
    # Detect OS and install accordingly
    if [ -f /etc/debian_version ]; then
        echo "🔄 Installing dependencies..."
        (apt update -qq && apt install -y curl wget git build-essential pkg-config libssl-dev python3 clang libclang-dev llvm-dev cmake openssl >/dev/null 2>&1) &
        show_progress $!
    elif [ -f /etc/redhat-release ]; then
        echo "🔄 Installing dependencies..."
        (yum update -y && yum install -y curl wget git gcc gcc-c++ make openssl-devel python3 clang llvm-devel cmake >/dev/null 2>&1) &
        show_progress $!
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🔄 Installing dependencies..."
        if ! command -v brew &> /dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        (brew install curl wget git openssl python3 llvm cmake >/dev/null 2>&1) &
        show_progress $!
    fi
    echo -e "${GREEN}✅ Environment ready!${NC}"
}

install_rust() {
    echo -e "${BLUE}🔧 Installing Rust...${NC}"
    
    if ! command -v rustc &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi
    
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
}

download_arthachain() {
    echo -e "${BLUE}📥 Downloading ArthChain...${NC}"
    
    mkdir -p "$INSTALL_PATH"
    cd "$INSTALL_PATH"
    
    if [ -d "ArthaChain" ]; then
        cd ArthaChain
        git pull origin main
    else
        git clone https://github.com/arthachain/ArthaChain.git
        cd ArthaChain
    fi
}

build_validator() {
    echo -e "${BLUE}🔨 Building ArthChain Validator...${NC}"
    loading_animation "Compiling blockchain code (this takes 5-10 minutes)" 6
    
    cd blockchain_node
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "🔄 Building validator binary..."
    (cargo build --release --bin testnet_api_server >/dev/null 2>&1) &
    BUILD_PID=$!
    
    # Show progress while building
    while ps -p $BUILD_PID > /dev/null; do
        for i in {1..10}; do
            echo -n "█"
            sleep 3
        done
        echo -n " Building..."
        echo ""
    done
    
    # Verify build
    if [ -f "target/release/testnet_api_server" ]; then
        echo -e "${GREEN}✅ Validator built successfully!${NC}"
        BINARY_PATH="$(pwd)/target/release/testnet_api_server"
    elif [ -f "../target/release/testnet_api_server" ]; then
        echo -e "${GREEN}✅ Validator built successfully!${NC}"
        BINARY_PATH="$(pwd)/../target/release/testnet_api_server"
    else
        echo -e "${RED}❌ Build failed!${NC}"
        exit 1
    fi
}

create_config() {
    echo -e "${BLUE}⚙️ Creating validator configuration...${NC}"
    
    cat > validator_config.toml << EOF
[node]
name = "validator-$(date +%s)"
network_id = "arthachain-testnet-1"
data_dir = "./validator_data"

[network]
listen_addr = "0.0.0.0:$P2P_PORT"
bootstrap_peers = [
    "/ip4/103.160.27.49/tcp/30303",
    "https://api.arthachain.in"
]
enable_discovery = true
genesis_sync = true
sync_from_network = true

[api]
listen_addr = "0.0.0.0:$API_PORT"
enable_cors = true
enable_metrics = true

[consensus]
mechanism = "SVCP_SVBFT"
block_time = 5
validator_enabled = true
join_existing_network = true

[storage]
backend = "rocksdb"
path = "./validator_data/rocksdb"

[dashboard]
enabled = $ENABLE_DASHBOARD
port = $DASHBOARD_PORT
password = "$DASHBOARD_PASSWORD"
EOF
}

create_scripts() {
    echo -e "${BLUE}📝 Creating management scripts...${NC}"
    loading_animation "Setting up scripts" 2
    
    # Create scripts in the installation directory
    cd "$INSTALL_PATH/ArthaChain"
    
    # Start script with proper binary path
    cat > start-validator.sh << EOF
#!/bin/bash
echo -e "${GREEN}🚀 Starting ArthChain Validator...${NC}"
echo "💰 Wallet: $WALLET_ADDRESS"
echo "📊 Dashboard: http://localhost:$DASHBOARD_PORT"
echo "🔗 API: http://localhost:$API_PORT"
echo "📡 P2P: $P2P_PORT"
echo ""

cd $INSTALL_PATH/ArthaChain/blockchain_node
source ~/.cargo/env

# Use the correct binary path
$BINARY_PATH --config validator_config.toml &
sleep 5

echo "✅ Validator started!"
echo "📊 Status: curl http://localhost:$API_PORT/api/status"
echo "🌐 Dashboard: http://localhost:$DASHBOARD_PORT"
echo "🔐 Password: $DASHBOARD_PASSWORD"
EOF
    chmod +x start-validator.sh
    
    # Auto-start the validator
    echo -e "${BLUE}🚀 Starting your validator now...${NC}"
    loading_animation "Launching validator" 4
    ./start-validator.sh
    
    # Status script
    cat > check-status.sh << EOF
#!/bin/bash
echo "📊 ArthChain Validator Status:"
echo "=============================="
curl -s http://localhost:$API_PORT/api/status | python3 -m json.tool 2>/dev/null || echo "Validator not responding"
echo ""
echo "🌐 Network Status:"
curl -s https://api.arthachain.in/api/validators | python3 -c "import sys,json; data=json.load(sys.stdin); print(f'Validators: {data[\"active_count\"]} active')" 2>/dev/null
EOF
    chmod +x check-status.sh
    
    # Stop script
    cat > stop-validator.sh << EOF
#!/bin/bash
echo "🛑 Stopping ArthChain Validator..."
pkill -f testnet_api_server
echo "✅ Validator stopped"
EOF
    chmod +x stop-validator.sh
}

print_success() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  🎉 SETUP COMPLETE! 🎉                   ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}📋 Your ArthChain Validator is ready!${NC}"
    echo ""
    echo -e "${BLUE}🚀 Start validator:${NC} ./start-validator.sh"
    echo -e "${BLUE}📊 Check status:${NC} ./check-status.sh"
    echo -e "${BLUE}🛑 Stop validator:${NC} ./stop-validator.sh"
    echo ""
    echo -e "${GREEN}📊 Dashboard:${NC} http://localhost:$DASHBOARD_PORT"
    echo -e "${GREEN}🔗 Password:${NC} $DASHBOARD_PASSWORD"
    echo ""
    echo -e "${BLUE}📁 Path:${NC} $INSTALL_PATH/ArthaChain"
    echo -e "${BLUE}🌐 Network:${NC} ArthChain Testnet (existing blocks)"
    echo -e "${BLUE}📡 Bootstrap:${NC} 103.160.27.49:30303"
    echo ""
    echo -e "${GREEN}🎯 Your validator will join the existing network at current block height!${NC}"
    echo -e "${YELLOW}⚠️  IMPORTANT: Save dashboard password: $DASHBOARD_PASSWORD${NC}"
}

# Main execution
main() {
    print_logo
    ask_permission
    set_defaults
    ask_custom_config
    
    install_dependencies
    install_rust
    generate_wallet
    download_arthachain
    build_validator
    create_config
    create_scripts
    print_success
}

main "$@"
