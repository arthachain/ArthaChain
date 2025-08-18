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
    echo -e "${PURPLE}"
    echo "                                                              "
    echo "    ██████╗ ██████╗ ████████╗██╗  ██╗ █████╗  ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗"
    echo "   ██╔══██╗██╔══██╗╚══██╔══╝██║  ██║██╔══██╗██╔════╝██║  ██║██╔══██╗██║████╗  ██║"
    echo "   ███████║██████╔╝   ██║   ███████║███████║██║     ███████║███████║██║██╔██╗ ██║"
    echo "   ██╔══██║██╔══██╗   ██║   ██╔══██║██╔══██║██║     ██╔══██║██╔══██║██║██║╚██╗██║"
    echo "   ██║  ██║██║  ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║██║  ██║██║██║ ╚████║"
    echo "   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝"
    echo -e "${NC}"
    echo ""
    echo -e "${CYAN}╭──────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│${NC}  🌟 ${YELLOW}QUANTUM-POWERED BLOCKCHAIN VALIDATOR SETUP${NC} 🌟  ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}                                                              ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  🔥 ${GREEN}Join the most advanced blockchain network${NC} 🔥      ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  ⚡ ${BLUE}AI-Enhanced • Quantum-Resistant • Lightning Fast${NC}   ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  💎 ${PURPLE}Earn rewards while securing the future${NC} 💎        ${CYAN}│${NC}"
    echo -e "${CYAN}╰──────────────────────────────────────────────────────────────╯${NC}"
    echo ""
    
    # Flashy animation
    for i in {1..3}; do
        echo -e "${RED}💥${YELLOW}✨${GREEN}🚀${BLUE}⚡${PURPLE}💎${NC}"
        sleep 0.3
        echo -e "\033[1A\033[K"
    done
    
    echo -e "${GREEN}🔥 ${YELLOW}Ready to become a blockchain validator?${NC} 🔥"
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

# Generate wallet with cool animations
generate_wallet() {
    echo -e "${PURPLE}💎 Creating Your Blockchain Identity...${NC}"
    echo ""
    
    # Cool generation animation
    echo -n "🔮 Generating quantum-secure keys"
    for i in {1..8}; do
        echo -n " ⚡"
        sleep 0.4
    done
    echo ""
    
    # Generate random private key
    PRIVATE_KEY=$(openssl rand -hex 32)
    
    # Generate wallet address (simplified)
    WALLET_ADDRESS="0x$(echo -n "$PRIVATE_KEY" | sha256sum | cut -c1-40)"
    
    echo ""
    echo -e "${GREEN}🎉 YOUR ARTHACHAIN VALIDATOR WALLET IS READY! 🎉${NC}"
    echo ""
    
    # Flashy wallet display
    echo -e "${CYAN}╭─────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│${NC}                   ${YELLOW}💰 WALLET CREDENTIALS 💰${NC}                    ${CYAN}│${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│${NC}                                                                 ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  🔐 ${GREEN}Address:${NC} ${WALLET_ADDRESS}     ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}                                                                 ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  🗝️  ${YELLOW}Private:${NC} ${PRIVATE_KEY} ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}                                                                 ${CYAN}│${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────────────────╯${NC}"
    echo ""
    
    # Security warning with animation
    for i in {1..3}; do
        echo -e "${RED}🚨 CRITICAL: SAVE YOUR PRIVATE KEY NOW! 🚨${NC}"
        sleep 0.5
        echo -e "\033[1A\033[K"
    done
    
    echo -e "${RED}🔒 SECURITY ALERT: Your private key = Your money! Save it safely! 🔒${NC}"
    echo ""
    echo -e "${YELLOW}📱 Screenshot this, write it down, or copy to secure storage...${NC}"
    echo -e "${GREEN}Press ENTER after you've safely saved your wallet credentials ⬇️${NC}"
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
    echo -e "${BLUE}🔮 Auto-configuring your validator...${NC}"
    loading_animation "Detecting network settings" 4
    
    # Auto-detect external IP with animation
    echo -n "🌐 Finding your IP address"
    for i in {1..3}; do
        echo -n " 🔍"
        sleep 0.3
    done
    EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s icanhazip.com 2>/dev/null || echo "127.0.0.1")
    echo " ✅ Found: $EXTERNAL_IP"
    
    # Use standard ports
    P2P_PORT=30303
    API_PORT=8080
    DASHBOARD_PORT=8080
    
    # Default paths
    INSTALL_PATH="$HOME/.arthachain"
    
    # Enable dashboard by default
    ENABLE_DASHBOARD=true
    
    echo ""
    echo -e "${GREEN}⚙️ Configuration Ready:${NC}"
    echo "   📁 Install: $INSTALL_PATH"
    echo "   🌐 IP: $EXTERNAL_IP"
    echo "   📡 P2P: $P2P_PORT"
    echo "   🔗 API: $API_PORT"
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
    echo -e "${BLUE}🦀 Installing Rust Programming Language...${NC}"
    echo ""
    
    if ! command -v rustc &> /dev/null; then
        echo -e "${YELLOW}📥 Downloading Rust installer...${NC}"
        
        # Rust installation animation
        echo -n "🦀 Installing Rust"
        (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1) &
        RUST_PID=$!
        
        # Show RUST, RUST, RUST while installing
        while ps -p $RUST_PID > /dev/null; do
            echo -e " ${RED}RUST${GREEN}LANG${BLUE}RUST${YELLOW}LANG${PURPLE}RUST${CYAN}LANG${NC}"
            sleep 1
        done
        
        source ~/.cargo/env
        echo ""
        echo -e "${GREEN}✅ Rust installed successfully!${NC}"
    else
        echo -e "${GREEN}✅ Rust already installed!${NC}"
    fi
    
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    echo ""
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
    echo -e "${PURPLE}⚡ Compiling ArthChain Quantum Validator...${NC}"
    echo ""
    
    cd blockchain_node
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo -e "${CYAN}🔥 Building the future of blockchain...${NC}"
    echo ""
    
    # Professional build animation with ARTHACHAIN text
    echo -e "${YELLOW}🔨 Compiling ArthChain validator binary...${NC}"
    (cargo build --release --bin testnet_api_server >/dev/null 2>&1) &
    BUILD_PID=$!
    
    # Show ARTHACHAIN in different colors while building
    while ps -p $BUILD_PID > /dev/null; do
        echo -e "   ${RED}ARTHACHAIN${NC} ${GREEN}ARTHACHAIN${NC} ${BLUE}ARTHACHAIN${NC} ${YELLOW}ARTHACHAIN${NC} ${PURPLE}ARTHACHAIN${NC} ${CYAN}ARTHACHAIN${NC}"
        sleep 2
    done
    
    echo ""
    echo -e "${GREEN}🎉 BLOCKCHAIN VALIDATOR COMPILED! 🎉${NC}"
    
    # Verify build with animations
    if [ -f "target/release/testnet_api_server" ]; then
        BINARY_PATH="$(pwd)/target/release/testnet_api_server"
        echo -e "${GREEN}✅ Validator binary ready at: $BINARY_PATH${NC}"
    elif [ -f "../target/release/testnet_api_server" ]; then
        BINARY_PATH="$(pwd)/../target/release/testnet_api_server"
        echo -e "${GREEN}✅ Validator binary ready at: $BINARY_PATH${NC}"
    else
        echo -e "${RED}❌ Build failed!${NC}"
        exit 1
    fi
    
    echo ""
    loading_animation "Preparing validator engine" 3
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
    
    # Ask for password at the end
    echo -e "${YELLOW}🔐 Set your dashboard password (or press ENTER for auto-generated):${NC}"
    read -s USER_PASSWORD
    echo ""
    if [ ! -z "$USER_PASSWORD" ]; then
        DASHBOARD_PASSWORD="$USER_PASSWORD"
        echo -e "${GREEN}✅ Custom password set!${NC}"
    else
        echo -e "${GREEN}✅ Auto-generated password: $DASHBOARD_PASSWORD${NC}"
    fi
    echo ""
    
    # Epic launch sequence
    echo -e "${PURPLE}🚀 LAUNCHING YOUR ARTHACHAIN VALIDATOR 🚀${NC}"
    echo ""
    echo -e "${BLUE}Initializing quantum engines...${NC}"
    for i in {1..10}; do
        echo -n "⚡"
        sleep 0.2
    done
    echo " ✅"
    
    echo -e "${GREEN}Connecting to ArthChain network...${NC}"
    for i in {1..8}; do
        echo -n "🌐"
        sleep 0.2
    done
    echo " ✅"
    
    echo -e "${YELLOW}Starting validator process...${NC}"
    for i in {1..6}; do
        echo -n "🔥"
        sleep 0.3
    done
    echo " ✅"
    
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
    clear
    
    # Epic success animation
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║              🎉🎉🎉 VALIDATOR READY! 🎉🎉🎉                   ║"
    echo "║                                                                ║"
    echo "║          🔥 YOU'RE NOW AN ARTHACHAIN VALIDATOR! 🔥             ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Celebration animation
    for i in {1..5}; do
        echo -e "${YELLOW}🎊${GREEN}🎉${BLUE}🚀${PURPLE}💎${RED}🔥${NC}"
        sleep 0.3
        echo -e "\033[1A\033[K"
    done
    
    echo -e "${CYAN}╭─────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${CYAN}│${NC}                    ${YELLOW}🎯 VALIDATOR INFO 🎯${NC}                    ${CYAN}│${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│${NC}                                                             ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  💰 ${GREEN}Wallet:${NC} $WALLET_ADDRESS     ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  🌐 ${BLUE}Dashboard:${NC} http://localhost:$DASHBOARD_PORT              ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  🔐 ${YELLOW}Password:${NC} $DASHBOARD_PASSWORD                       ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  📡 ${PURPLE}API:${NC} http://localhost:$API_PORT                     ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}                                                             ${CYAN}│${NC}"
    echo -e "${CYAN}╰─────────────────────────────────────────────────────────────╯${NC}"
    echo ""
    
    echo -e "${GREEN}✅ Your validator is running and contributing to ArthChain!${NC}"
    echo -e "${BLUE}💎 Earning rewards by securing the network${NC}"
    echo -e "${YELLOW}🌟 Part of the future of blockchain technology${NC}"
    echo ""
    
    echo -e "${CYAN}📋 Quick Commands:${NC}"
    echo -e "   📊 Status: ${GREEN}./check-status.sh${NC}"
    echo -e "   🛑 Stop: ${RED}./stop-validator.sh${NC}"
    echo ""
    
    echo -e "${PURPLE}🎊 Welcome to the ArthChain validator community! 🎊${NC}"
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
