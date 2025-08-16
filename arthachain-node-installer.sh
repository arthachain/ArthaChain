#!/bin/bash
# ArthaChain Node Complete Setup Script
# One-command setup from dependencies to running validator
# Usage: curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/arthachain-node-installer.sh | bash -s -- [NODE_ID]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Global variables
NODE_ID=${1:-2}
BOOTSTRAP_IP="103.160.27.61"
BOOTSTRAP_PORT="30303"
REPO_URL="https://github.com/arthachain/ArthaChain.git"
WORK_DIR="$HOME/arthachain"

print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║              🚀 ArthaChain Node Installer 🚀              ║"
    echo "║                                                           ║"
    echo "║        Complete setup from dependencies to validator      ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${CYAN}[STEP $1]${NC} $2"; }

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
            OS_VERSION=$VERSION_ID
        fi
        PLATFORM="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
        PLATFORM="darwin"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Install system dependencies
install_system_deps() {
    print_step "1" "Installing system dependencies..."
    
    if [[ "$PLATFORM" == "linux" ]]; then
        # Detect package manager
        if command -v apt-get &> /dev/null; then
            PKG_MANAGER="apt"
            print_status "Using apt package manager"
            sudo apt-get update -y
            sudo apt-get install -y curl wget git build-essential pkg-config libssl-dev python3 python3-pip jq unzip
        elif command -v yum &> /dev/null; then
            PKG_MANAGER="yum"
            print_status "Using yum package manager"
            sudo yum update -y
            sudo yum install -y curl wget git gcc gcc-c++ make openssl-devel python3 python3-pip jq unzip
        elif command -v dnf &> /dev/null; then
            PKG_MANAGER="dnf"
            print_status "Using dnf package manager"
            sudo dnf update -y
            sudo dnf install -y curl wget git gcc gcc-c++ make openssl-devel python3 python3-pip jq unzip
        elif command -v pacman &> /dev/null; then
            PKG_MANAGER="pacman"
            print_status "Using pacman package manager"
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm curl wget git base-devel openssl python python-pip jq unzip
        else
            print_error "No supported package manager found (apt, yum, dnf, pacman)"
            exit 1
        fi
    elif [[ "$PLATFORM" == "darwin" ]]; then
        print_status "Using macOS"
        # Install Homebrew if not present
        if ! command -v brew &> /dev/null; then
            print_status "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew update
        brew install curl wget git openssl python3 jq
    fi
    
    print_success "System dependencies installed"
}

# Install Rust
install_rust() {
    print_step "2" "Installing Rust..."
    
    if command -v rustc &> /dev/null; then
        print_warning "Rust already installed: $(rustc --version)"
        return
    fi
    
    print_status "Downloading and installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Source Rust environment
    source "$HOME/.cargo/env"
    
    # Verify installation
    if command -v rustc &> /dev/null; then
        print_success "Rust installed: $(rustc --version)"
        print_success "Cargo installed: $(cargo --version)"
    else
        print_error "Failed to install Rust"
        exit 1
    fi
    
    # Install additional components
    rustup component add clippy rustfmt
    rustup target add wasm32-unknown-unknown
    
    print_success "Rust installation complete"
}

# Install additional tools
install_tools() {
    print_step "3" "Installing additional tools..."
    
    # Install Node.js (for some tools)
    if ! command -v node &> /dev/null; then
        print_status "Installing Node.js..."
        if [[ "$PLATFORM" == "linux" ]]; then
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            if [[ "$PKG_MANAGER" == "apt" ]]; then
                sudo apt-get install -y nodejs
            elif [[ "$PKG_MANAGER" == "yum" || "$PKG_MANAGER" == "dnf" ]]; then
                sudo $PKG_MANAGER install -y nodejs npm
            fi
        elif [[ "$PLATFORM" == "darwin" ]]; then
            brew install node
        fi
    fi
    
    print_success "Additional tools installed"
}

# Setup workspace
setup_workspace() {
    print_step "4" "Setting up workspace..."
    
    # Create work directory
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    print_status "Working directory: $WORK_DIR"
    print_success "Workspace ready"
}

# Clone repository
clone_repo() {
    print_step "5" "Cloning ArthaChain repository..."
    
    if [ -d "ArthaChain" ]; then
        print_warning "Repository already exists, updating..."
        cd ArthaChain
        git pull origin main
    else
        print_status "Cloning from $REPO_URL"
        git clone "$REPO_URL"
        cd ArthaChain
    fi
    
    print_success "Repository ready"
}

# Build project
build_project() {
    print_step "6" "Building ArthaChain (this may take 10-15 minutes)..."
    
    cd blockchain_node
    
    print_status "Building project..."
    cargo build --release --bin testnet_api_server
    
    if [ $? -eq 0 ]; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Setup node configuration
setup_node() {
    print_step "7" "Setting up Node $NODE_ID configuration..."
    
    # Run the quick setup script
    chmod +x quick_node_setup.sh
    ./quick_node_setup.sh "$NODE_ID"
    
    print_success "Node $NODE_ID configured"
}

# Configure firewall
configure_firewall() {
    print_step "8" "Configuring firewall..."
    
    if command -v ufw &> /dev/null; then
        print_status "Configuring UFW firewall..."
        sudo ufw allow 30303/tcp comment "ArthaChain P2P"
        sudo ufw allow 8080/tcp comment "ArthaChain API"
        sudo ufw allow 8545/tcp comment "ArthaChain RPC"
        print_success "UFW firewall configured"
    elif command -v firewall-cmd &> /dev/null; then
        print_status "Configuring firewalld..."
        sudo firewall-cmd --permanent --add-port=30303/tcp
        sudo firewall-cmd --permanent --add-port=8080/tcp
        sudo firewall-cmd --permanent --add-port=8545/tcp
        sudo firewall-cmd --reload
        print_success "Firewalld configured"
    else
        print_warning "No firewall manager found. Please manually open ports 30303, 8080, 8545"
    fi
}

# Test connectivity
test_connectivity() {
    print_step "9" "Testing connectivity..."
    
    print_status "Testing connection to bootstrap node..."
    if timeout 5 bash -c "</dev/tcp/$BOOTSTRAP_IP/22" 2>/dev/null; then
        print_success "Bootstrap node is reachable"
    else
        print_warning "Cannot reach bootstrap node. Check network connectivity."
    fi
    
    print_status "Testing ArthaChain API..."
    if curl -s https://api.arthachain.in/api/health >/dev/null 2>&1; then
        print_success "ArthaChain API is accessible"
    else
        print_warning "Cannot reach ArthaChain API"
    fi
}

# Create service file (optional)
create_service() {
    print_step "10" "Creating systemd service (optional)..."
    
    if [[ "$PLATFORM" == "linux" ]] && command -v systemctl &> /dev/null; then
        cat > arthachain-node.service << EOF
[Unit]
Description=ArthaChain Node $NODE_ID
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR/ArthaChain/blockchain_node
ExecStart=$WORK_DIR/ArthaChain/blockchain_node/target/release/testnet_api_server --config node${NODE_ID}_config.toml
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
EOF
        
        print_status "Service file created: arthachain-node.service"
        print_status "To install: sudo cp arthachain-node.service /etc/systemd/system/"
        print_status "To enable: sudo systemctl enable arthachain-node"
        print_status "To start: sudo systemctl start arthachain-node"
    fi
}

# Start node
start_node() {
    print_step "11" "Starting ArthaChain Node $NODE_ID..."
    
    print_status "Node configuration: node${NODE_ID}_config.toml"
    print_status "Data directory: testnet_data_node${NODE_ID}"
    print_status "Connecting to bootstrap: $BOOTSTRAP_IP:$BOOTSTRAP_PORT"
    
    # Create start script
    cat > start_node.sh << EOF
#!/bin/bash
echo "🚀 Starting ArthaChain Node $NODE_ID..."
echo "=================================="
echo "Time: \$(date)"
echo "Config: node${NODE_ID}_config.toml"
echo "Bootstrap: $BOOTSTRAP_IP:$BOOTSTRAP_PORT"
echo "=================================="

cargo run --bin testnet_api_server --release -- --config node${NODE_ID}_config.toml
EOF
    chmod +x start_node.sh
    
    print_success "Node setup complete!"
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    🎉 SETUP COMPLETE! 🎉                  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}📋 NEXT STEPS:${NC}"
    echo -e "${YELLOW}1. Start your node:${NC}"
    echo "   ./start_node.sh"
    echo ""
    echo -e "${YELLOW}2. Monitor your node:${NC}"
    echo "   ./monitor_node${NODE_ID}.sh"
    echo ""
    echo -e "${YELLOW}3. Check local status:${NC}"
    echo "   curl http://localhost:8080/api/health"
    echo ""
    echo -e "${YELLOW}4. Check network status:${NC}"
    echo "   curl https://api.arthachain.in/api/validators"
    echo ""
    echo -e "${CYAN}📂 Working Directory:${NC} $WORK_DIR/ArthaChain/blockchain_node"
    echo -e "${CYAN}🌐 Bootstrap Node:${NC} $BOOTSTRAP_IP:$BOOTSTRAP_PORT"
    echo -e "${CYAN}🔗 Node ID:${NC} $NODE_ID"
    echo ""
    echo -e "${GREEN}Your ArthaChain validator node is ready to join the testnet! 🚀${NC}"
}

# Main execution
main() {
    print_banner
    
    print_status "Starting ArthaChain Node Setup..."
    print_status "Node ID: $NODE_ID"
    print_status "Bootstrap: $BOOTSTRAP_IP:$BOOTSTRAP_PORT"
    echo ""
    
    check_root
    detect_os
    install_system_deps
    install_rust
    install_tools
    setup_workspace
    clone_repo
    build_project
    setup_node
    configure_firewall
    test_connectivity
    create_service
    start_node
}

# Handle Ctrl+C
trap 'print_error "Setup interrupted by user"; exit 1' INT

# Run main function
main "$@"
