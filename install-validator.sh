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
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                                                          ║"
    echo "║            🚀 ArthChain Validator Setup 🚀               ║"
    echo "║                                                          ║"
    echo "║              Simple One-Command Installation             ║"
    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

ask_permission() {
    echo -e "${YELLOW}By running this installer, you agree to join the ArthChain network as a validator. (y/n)?:${NC}"
    read -r PERMISSION
    if [[ ! "$PERMISSION" =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
}

ask_dashboard() {
    echo -e "${YELLOW}Do you want to run the web based Dashboard? (y/n):${NC}"
    read -r DASHBOARD
    if [[ "$DASHBOARD" =~ ^[Yy]$ ]]; then
        ENABLE_DASHBOARD=true
    else
        ENABLE_DASHBOARD=false
    fi
}

ask_dashboard_port() {
    if [ "$ENABLE_DASHBOARD" = true ]; then
        echo -e "${YELLOW}Enter the port (1025-65536) to access the web based Dashboard (default 8080):${NC}"
        read -r DASHBOARD_PORT
        DASHBOARD_PORT=${DASHBOARD_PORT:-8080}
    else
        DASHBOARD_PORT=8080
    fi
}

ask_external_ip() {
    echo -e "${YELLOW}If you wish to set an explicit external IP, enter an IPv4 address (default=auto):${NC}"
    read -r EXTERNAL_IP
    if [ -z "$EXTERNAL_IP" ]; then
        EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "auto")
    fi
}

ask_p2p_port() {
    echo -e "${YELLOW}Enter the first port (1025-65536) for p2p communication (default 30303):${NC}"
    read -r P2P_PORT
    P2P_PORT=${P2P_PORT:-30303}
}

ask_api_port() {
    echo -e "${YELLOW}Enter the API port (1025-65536) for blockchain API (default 8080):${NC}"
    read -r API_PORT
    API_PORT=${API_PORT:-8080}
}

ask_install_path() {
    echo -e "${YELLOW}What base directory should the node use (defaults to ~/.arthachain):${NC}"
    read -r INSTALL_PATH
    INSTALL_PATH=${INSTALL_PATH:-~/.arthachain}
}

ask_password() {
    echo -e "${YELLOW}Set the password to access the Dashboard:${NC}"
    read -s DASHBOARD_PASSWORD
    echo ""
}

install_dependencies() {
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    
    # Detect OS and install accordingly
    if [ -f /etc/debian_version ]; then
        apt update -qq
        apt install -y curl wget git build-essential pkg-config libssl-dev python3 clang libclang-dev llvm-dev cmake
    elif [ -f /etc/redhat-release ]; then
        yum update -y
        yum install -y curl wget git gcc gcc-c++ make openssl-devel python3 clang llvm-devel cmake
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command -v brew &> /dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install curl wget git openssl python3 llvm cmake
    fi
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
    echo -e "${BLUE}🔨 Building ArthChain Validator (this may take 10 minutes)...${NC}"
    
    cd blockchain_node
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
    
    cargo build --release --bin testnet_api_server
    cargo build --release --bin arthachain
}

create_config() {
    echo -e "${BLUE}⚙️ Creating validator configuration...${NC}"
    
    cat > validator_config.toml << EOF
[network]
network_id = "arthachain-testnet-1"
chain_id = 201766

[node]
node_id = "validator-$(date +%s)"
data_dir = "./validator_data"

[api]
addr = "0.0.0.0"
port = $API_PORT

[p2p]
listen_addr = "0.0.0.0:$P2P_PORT"
external_addr = "$EXTERNAL_IP:$P2P_PORT"
bootstrap_peers = [
    "/ip4/103.160.27.61/tcp/30303"
]

[dashboard]
enabled = $ENABLE_DASHBOARD
port = $DASHBOARD_PORT
password = "$DASHBOARD_PASSWORD"

[consensus]
mechanism = "SVCP"
validator_enabled = true
EOF
}

create_scripts() {
    echo -e "${BLUE}📝 Creating management scripts...${NC}"
    
    # Start script
    cat > start-validator.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting ArthChain Validator..."
echo "Dashboard: http://localhost:DASHBOARD_PORT"
echo "API: http://localhost:API_PORT"
echo "P2P: PORT P2P_PORT"
echo ""

cd INSTALL_PATH/ArthaChain/blockchain_node
source ~/.cargo/env

./target/release/testnet_api_server --config validator_config.toml
EOF
    
    # Replace placeholders
    sed -i "s/DASHBOARD_PORT/$DASHBOARD_PORT/g" start-validator.sh
    sed -i "s/API_PORT/$API_PORT/g" start-validator.sh
    sed -i "s/P2P_PORT/$P2P_PORT/g" start-validator.sh
    sed -i "s|INSTALL_PATH|$INSTALL_PATH|g" start-validator.sh
    chmod +x start-validator.sh
    
    # Stop script
    cat > stop-validator.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping ArthChain Validator..."
pkill -f testnet_api_server
pkill -f arthachain
echo "✅ Validator stopped"
EOF
    chmod +x stop-validator.sh
    
    # Status script
    cat > check-status.sh << 'EOF'
#!/bin/bash
echo "📊 ArthChain Validator Status:"
echo "=============================="
curl -s http://localhost:API_PORT/api/status | python3 -m json.tool 2>/dev/null || echo "Validator not responding"
EOF
    sed -i "s/API_PORT/$API_PORT/g" check-status.sh
    chmod +x check-status.sh
}

print_success() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  🎉 SETUP COMPLETE! 🎉                   ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}📋 Your ArthChain Validator is ready!${NC}"
    echo ""
    echo -e "${BLUE}🚀 To start your validator:${NC}"
    echo "   ./start-validator.sh"
    echo ""
    echo -e "${BLUE}📊 To check status:${NC}"
    echo "   ./check-status.sh"
    echo ""
    echo -e "${BLUE}🛑 To stop your validator:${NC}"
    echo "   ./stop-validator.sh"
    echo ""
    if [ "$ENABLE_DASHBOARD" = true ]; then
        echo -e "${BLUE}🌐 Dashboard:${NC} http://localhost:$DASHBOARD_PORT"
        echo ""
    fi
    echo -e "${BLUE}📁 Installation Path:${NC} $INSTALL_PATH/ArthaChain"
    echo -e "${BLUE}🌐 Network:${NC} ArthChain Testnet"
    echo -e "${BLUE}🔗 API Port:${NC} $API_PORT"
    echo -e "${BLUE}📡 P2P Port:${NC} $P2P_PORT"
    echo ""
    echo -e "${GREEN}🎯 Your validator will help secure the ArthChain network!${NC}"
}

# Main execution
main() {
    print_logo
    ask_permission
    ask_dashboard
    ask_dashboard_port
    ask_external_ip
    ask_p2p_port
    ask_api_port
    ask_install_path
    if [ "$ENABLE_DASHBOARD" = true ]; then
        ask_password
    fi
    
    install_dependencies
    install_rust
    download_arthachain
    build_validator
    create_config
    create_scripts
    print_success
}

main "$@"
