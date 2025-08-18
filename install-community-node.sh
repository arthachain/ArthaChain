#!/bin/bash
# ArthaChain Community Node Installer
# Professional setup script that ACTUALLY WORKS

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ARTHACHAIN_HOME="$HOME/.arthachain"
NODE_DATA_DIR="$ARTHACHAIN_HOME/data"
LOG_FILE="$ARTHACHAIN_HOME/node.log"
CONFIG_FILE="$ARTHACHAIN_HOME/node_config.toml"
GITHUB_REPO="https://github.com/arthachain/ArthaChain.git"

# Print header
print_header() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║        ${GREEN}ArthaChain Community Node Setup v1.0${BLUE}                ║${NC}"
    echo -e "${BLUE}║        ${YELLOW}Join the Decentralized Network${BLUE}                      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Check system requirements
check_requirements() {
    echo -e "${BLUE}[1/7] Checking System Requirements...${NC}"
    
    # Check RAM
    TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
    if [ $TOTAL_RAM -lt 8000 ]; then
        echo -e "${YELLOW}⚠️  Warning: Less than 8GB RAM detected ($TOTAL_RAM MB)${NC}"
        echo "Recommended: 16GB RAM for optimal performance"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ RAM: ${TOTAL_RAM} MB${NC}"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $DISK_SPACE -lt 100 ]; then
        echo -e "${RED}✗ Insufficient disk space: ${DISK_SPACE}GB available${NC}"
        echo "Required: Minimum 100GB"
        exit 1
    else
        echo -e "${GREEN}✓ Disk Space: ${DISK_SPACE}GB available${NC}"
    fi
    
    echo -e "${GREEN}✓ System requirements check passed${NC}\n"
}

# Install dependencies
install_dependencies() {
    echo -e "${BLUE}[2/7] Installing Dependencies...${NC}"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt &> /dev/null; then
            echo "Installing packages for Ubuntu/Debian..."
            sudo apt update -qq
            sudo apt install -y curl git build-essential pkg-config libssl-dev clang cmake 2>/dev/null
        elif command -v yum &> /dev/null; then
            echo "Installing packages for CentOS/RHEL..."
            sudo yum install -y curl git gcc gcc-c++ make openssl-devel clang cmake 2>/dev/null
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing packages for macOS..."
        if ! command -v brew &> /dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install curl git openssl cmake llvm 2>/dev/null
    fi
    
    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
}

# Install Rust
install_rust() {
    echo -e "${BLUE}[3/7] Setting up Rust Environment...${NC}"
    
    if ! command -v cargo &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    else
        echo -e "${GREEN}✓ Rust already installed${NC}"
    fi
    
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓ Rust environment ready${NC}\n"
}

# Download ArthaChain
download_arthachain() {
    echo -e "${BLUE}[4/7] Downloading ArthaChain...${NC}"
    
    # Create directory structure
    mkdir -p "$ARTHACHAIN_HOME"
    cd "$ARTHACHAIN_HOME"
    
    # Clone or update repository
    if [ -d "ArthaChain" ]; then
        echo "Updating existing installation..."
        cd ArthaChain
        git pull origin main
    else
        echo "Downloading ArthaChain..."
        git clone "$GITHUB_REPO"
        cd ArthaChain
    fi
    
    echo -e "${GREEN}✓ ArthaChain downloaded${NC}\n"
}

# Build node
build_node() {
    echo -e "${BLUE}[5/7] Building ArthaChain Node...${NC}"
    echo "This may take 5-10 minutes..."
    
    cd "$ARTHACHAIN_HOME/ArthaChain/blockchain_node"
    
    # Check if binary exists
    if [ -f "target/release/testnet_api_server" ] || [ -f "../target/release/testnet_api_server" ]; then
        echo -e "${GREEN}✓ Binary already built${NC}"
        if [ -f "../target/release/testnet_api_server" ]; then
            BINARY_PATH="$ARTHACHAIN_HOME/ArthaChain/target/release/testnet_api_server"
        else
            BINARY_PATH="$ARTHACHAIN_HOME/ArthaChain/blockchain_node/target/release/testnet_api_server"
        fi
    else
        echo "Building from source..."
        cargo build --release --bin testnet_api_server
        BINARY_PATH="$ARTHACHAIN_HOME/ArthaChain/blockchain_node/target/release/testnet_api_server"
    fi
    
    echo -e "${GREEN}✓ Node binary built successfully${NC}\n"
}

# Configure node
configure_node() {
    echo -e "${BLUE}[6/7] Configuring Your Node...${NC}"
    
    # Get configuration from user
    echo -e "${YELLOW}Node Configuration:${NC}"
    
    read -p "Enter node name (default: community-node-$(date +%s)): " NODE_NAME
    NODE_NAME=${NODE_NAME:-"community-node-$(date +%s)"}
    
    read -p "Dashboard port (default 8080): " DASHBOARD_PORT
    DASHBOARD_PORT=${DASHBOARD_PORT:-8080}
    
    read -p "P2P port (default 30303): " P2P_PORT
    P2P_PORT=${P2P_PORT:-30303}
    
    # Detect external IP
    EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "auto")
    read -p "External IP (detected: $EXTERNAL_IP): " CUSTOM_IP
    EXTERNAL_IP=${CUSTOM_IP:-$EXTERNAL_IP}
    
    # Create configuration file
    cat > "$CONFIG_FILE" << EOF
[node]
name = "$NODE_NAME"
network_id = "arthachain-mainnet"
data_dir = "$NODE_DATA_DIR"

[network]
listen_addr = "0.0.0.0:$P2P_PORT"
external_ip = "$EXTERNAL_IP"
bootstrap_peers = [
    "https://api.arthachain.in",
    "https://rpc.arthachain.in",
    "/ip4/103.160.27.49/tcp/30303"
]
enable_discovery = true
max_peers = 50

[api]
listen_addr = "0.0.0.0:$DASHBOARD_PORT"
enable_cors = true
enable_metrics = true

[consensus]
mechanism = "SVCP_SVBFT"
block_time = 5
enable_mining = false

[storage]
backend = "rocksdb"
path = "$NODE_DATA_DIR/rocksdb"
cache_size = 512

[logging]
level = "info"
file = "$LOG_FILE"
EOF
    
    echo -e "${GREEN}✓ Configuration saved${NC}\n"
}

# Create service scripts
create_scripts() {
    echo -e "${BLUE}[7/7] Creating Management Scripts...${NC}"
    
    # Start script
    cat > "$ARTHACHAIN_HOME/start-node.sh" << EOF
#!/bin/bash
echo "Starting ArthaChain Node..."
cd "$ARTHACHAIN_HOME/ArthaChain/blockchain_node"
source ~/.cargo/env

# Stop any existing instances
pkill -f testnet_api_server 2>/dev/null

# Start node
nohup $BINARY_PATH "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
echo \$! > "$ARTHACHAIN_HOME/node.pid"

echo "Node started with PID: \$(cat $ARTHACHAIN_HOME/node.pid)"
echo "Dashboard: http://localhost:$DASHBOARD_PORT"
echo "Logs: tail -f $LOG_FILE"
EOF
    chmod +x "$ARTHACHAIN_HOME/start-node.sh"
    
    # Stop script
    cat > "$ARTHACHAIN_HOME/stop-node.sh" << EOF
#!/bin/bash
if [ -f "$ARTHACHAIN_HOME/node.pid" ]; then
    PID=\$(cat "$ARTHACHAIN_HOME/node.pid")
    echo "Stopping node (PID: \$PID)..."
    kill \$PID 2>/dev/null
    rm "$ARTHACHAIN_HOME/node.pid"
    echo "Node stopped"
else
    echo "Node is not running"
fi
EOF
    chmod +x "$ARTHACHAIN_HOME/stop-node.sh"
    
    # Status script
    cat > "$ARTHACHAIN_HOME/node-status.sh" << EOF
#!/bin/bash
echo "=== ArthaChain Node Status ==="
if [ -f "$ARTHACHAIN_HOME/node.pid" ]; then
    PID=\$(cat "$ARTHACHAIN_HOME/node.pid")
    if ps -p \$PID > /dev/null; then
        echo "✓ Node is running (PID: \$PID)"
        echo ""
        curl -s http://localhost:$DASHBOARD_PORT/api/status | python3 -m json.tool 2>/dev/null || echo "API not responding"
    else
        echo "✗ Node is not running (stale PID file)"
    fi
else
    echo "✗ Node is not running"
fi
EOF
    chmod +x "$ARTHACHAIN_HOME/node-status.sh"
    
    # Update script
    cat > "$ARTHACHAIN_HOME/update-node.sh" << EOF
#!/bin/bash
echo "Updating ArthaChain Node..."
$ARTHACHAIN_HOME/stop-node.sh
cd "$ARTHACHAIN_HOME/ArthaChain"
git pull origin main
cd blockchain_node
cargo build --release --bin testnet_api_server
echo "Update complete. Run start-node.sh to restart"
EOF
    chmod +x "$ARTHACHAIN_HOME/update-node.sh"
    
    echo -e "${GREEN}✓ Management scripts created${NC}\n"
}

# Start node
start_node() {
    echo -e "${YELLOW}Starting your ArthaChain node...${NC}"
    
    "$ARTHACHAIN_HOME/start-node.sh"
    
    sleep 5
    
    # Check if node started successfully
    if [ -f "$ARTHACHAIN_HOME/node.pid" ]; then
        PID=$(cat "$ARTHACHAIN_HOME/node.pid")
        if ps -p $PID > /dev/null; then
            echo -e "${GREEN}✅ NODE STARTED SUCCESSFULLY!${NC}\n"
            echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${BLUE}║                  ${GREEN}Installation Complete!${BLUE}                    ║${NC}"
            echo -e "${BLUE}╠════════════════════════════════════════════════════════════╣${NC}"
            echo -e "${BLUE}║ Dashboard:   ${YELLOW}http://localhost:$DASHBOARD_PORT${BLUE}                       ║${NC}"
            echo -e "${BLUE}║ Node Name:   ${YELLOW}$NODE_NAME${BLUE}                     ║${NC}"
            echo -e "${BLUE}║ P2P Port:    ${YELLOW}$P2P_PORT${BLUE}                                       ║${NC}"
            echo -e "${BLUE}║                                                            ║${NC}"
            echo -e "${BLUE}║ Commands:                                                  ║${NC}"
            echo -e "${BLUE}║   Start:   ${GREEN}$ARTHACHAIN_HOME/start-node.sh${BLUE}         ║${NC}"
            echo -e "${BLUE}║   Stop:    ${GREEN}$ARTHACHAIN_HOME/stop-node.sh${BLUE}          ║${NC}"
            echo -e "${BLUE}║   Status:  ${GREEN}$ARTHACHAIN_HOME/node-status.sh${BLUE}        ║${NC}"
            echo -e "${BLUE}║   Update:  ${GREEN}$ARTHACHAIN_HOME/update-node.sh${BLUE}        ║${NC}"
            echo -e "${BLUE}║   Logs:    ${GREEN}tail -f $LOG_FILE${BLUE}     ║${NC}"
            echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo -e "${GREEN}🎉 Welcome to the ArthaChain Network! 🎉${NC}"
        else
            echo -e "${RED}✗ Node failed to start${NC}"
            echo "Check logs: tail -f $LOG_FILE"
            exit 1
        fi
    else
        echo -e "${RED}✗ Failed to start node${NC}"
        exit 1
    fi
}

# Main installation flow
main() {
    print_header
    
    echo -e "${YELLOW}This installer will set up an ArthaChain community node on your system.${NC}"
    echo -e "${YELLOW}Installation directory: $ARTHACHAIN_HOME${NC}"
    echo ""
    read -p "Continue with installation? (y/n): " -n 1 -r
    echo -e "\n"
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled"
        exit 0
    fi
    
    check_requirements
    install_dependencies
    install_rust
    download_arthachain
    build_node
    configure_node
    create_scripts
    
    echo ""
    read -p "Start the node now? (y/n): " -n 1 -r
    echo -e "\n"
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_node
    else
        echo -e "${YELLOW}Installation complete!${NC}"
        echo -e "To start your node later, run: ${GREEN}$ARTHACHAIN_HOME/start-node.sh${NC}"
    fi
}

# Run installation
main "$@"
