#!/bin/bash

# ArthaChain Quick Start Deployment Script
# This script will help you deploy ArthaChain quickly for testing or development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 ArthaChain Quick Start Deployment${NC}"
echo "======================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ This script should not be run as root${NC}"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check Docker
if ! command_exists docker; then
    echo -e "${YELLOW}🐳 Installing Docker...${NC}"
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}✅ Docker installed. Please log out and log back in, then run this script again.${NC}"
    exit 0
fi

# Check Docker Compose
if ! command_exists docker-compose; then
    echo -e "${YELLOW}🔧 Installing Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Check if in Docker group
if ! groups $USER | grep -q docker; then
    echo -e "${RED}❌ User not in docker group. Run: sudo usermod -aG docker $USER${NC}"
    echo -e "${YELLOW}Then log out and log back in, and run this script again.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites met!${NC}"

# Get deployment type
echo -e "${YELLOW}🔧 Select deployment type:${NC}"
echo "1) Local Development (localhost only)"
echo "2) Production (with domain and SSL)"
echo "3) Public IP (no domain, but accessible from internet)"

read -p "Enter choice (1-3): " DEPLOY_TYPE

case $DEPLOY_TYPE in
    1)
        echo -e "${BLUE}🏠 Setting up local development environment...${NC}"
        ENV_FILE="local"
        ;;
    2)
        echo -e "${BLUE}🌐 Setting up production environment...${NC}"
        read -p "Enter your domain name: " DOMAIN
        read -p "Enter your email for SSL certificates: " EMAIL
        ENV_FILE="production"
        ;;
    3)
        echo -e "${BLUE}🌍 Setting up public IP environment...${NC}"
        PUBLIC_IP=$(curl -s https://ipinfo.io/ip)
        echo "Detected public IP: $PUBLIC_IP"
        read -p "Is this correct? (y/n): " IP_CONFIRM
        if [[ $IP_CONFIRM != "y" ]]; then
            read -p "Enter your public IP: " PUBLIC_IP
        fi
        ENV_FILE="public"
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

# Create .env file
echo -e "${YELLOW}📝 Creating environment configuration...${NC}"

cat > .env << EOF
# ArthaChain Environment Configuration
ARTHACHAIN_NODE_ID=arthachain-node-$(date +%s)
ARTHACHAIN_NETWORK_ID=arthachain-testnet-1
ARTHACHAIN_CHAIN_ID=1337
ARTHACHAIN_LOG_LEVEL=info

# Generated configuration for $ENV_FILE deployment
DEPLOYMENT_TYPE=$ENV_FILE
EOF

case $ENV_FILE in
    "local")
        cat >> .env << EOF
PUBLIC_IP=127.0.0.1
API_HOST=localhost:3000
RPC_HOST=localhost:8545
WS_HOST=localhost:8546
METRICS_HOST=localhost:9090
CADDY_EMAIL=admin@localhost
EOF
        ;;
    "production")
        cat >> .env << EOF
PUBLIC_IP=$(curl -s https://ipinfo.io/ip)
DOMAIN=$DOMAIN
API_HOST=api.$DOMAIN
RPC_HOST=rpc.$DOMAIN
WS_HOST=ws.$DOMAIN
METRICS_HOST=metrics.$DOMAIN
CADDY_EMAIL=$EMAIL
EOF
        ;;
    "public")
        cat >> .env << EOF
PUBLIC_IP=$PUBLIC_IP
API_HOST=$PUBLIC_IP:3000
RPC_HOST=$PUBLIC_IP:8545
WS_HOST=$PUBLIC_IP:8546
METRICS_HOST=$PUBLIC_IP:9090
CADDY_EMAIL=admin@example.com
EOF
        ;;
esac

# Add common settings
cat >> .env << EOF

# Security
QUANTUM_RESISTANCE=true
API_ADMIN_KEY=$(openssl rand -hex 32)
RATE_LIMIT=1000
MAX_CONNECTIONS_PER_IP=10

# Storage
STORAGE_BACKEND=hybrid
ROCKSDB_MAX_FILES=1000
MEMMAP_SIZE=1073741824

# Consensus
BLOCK_TIME=3
MAX_BLOCK_SIZE=2097152
VALIDATOR_SET_SIZE=4

# AI Engine
AI_ENGINE_ENABLED=true
AI_BATCH_SIZE=32

# Monitoring
DETAILED_METRICS=true
HEALTH_CHECK_INTERVAL=30
GRAFANA_PASSWORD=$(openssl rand -base64 12)

# Faucet (enabled for testnet)
FAUCET_ENABLED=true
FAUCET_AMOUNT=1000
FAUCET_COOLDOWN=3600
FAUCET_MAX_DAILY=10
EOF

echo -e "${GREEN}✅ Environment configuration created!${NC}"

# Check if blockchain source is available
if [[ ! -d "../blockchain_node" ]]; then
    echo -e "${RED}❌ ArthaChain source code not found!${NC}"
    echo "Please ensure you're running this script from the deploy/ directory"
    echo "and that the blockchain_node/ directory exists in the parent directory."
    exit 1
fi

# Build and start services
echo -e "${YELLOW}🔨 Building ArthaChain...${NC}"
docker-compose build

echo -e "${YELLOW}🚀 Starting services...${NC}"
docker-compose up -d

# Wait for services to start
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 30

# Health check
echo -e "${YELLOW}🔍 Performing health checks...${NC}"

# Check if node is responding
if curl -f -s http://localhost:3000/api/health > /dev/null; then
    echo -e "${GREEN}✅ ArthaChain node is healthy!${NC}"
else
    echo -e "${RED}❌ ArthaChain node health check failed${NC}"
    echo "Check logs with: docker-compose logs node"
fi

# Check if Grafana is responding
if curl -f -s http://localhost:3001 > /dev/null; then
    echo -e "${GREEN}✅ Grafana is healthy!${NC}"
else
    echo -e "${YELLOW}⚠️  Grafana may still be starting up${NC}"
fi

# Display access information
echo -e "${BLUE}======================================"
echo -e "🎉 ArthaChain deployment complete!"
echo -e "======================================${NC}"

case $ENV_FILE in
    "local")
        echo -e "${GREEN}📡 Local Access Points:${NC}"
        echo "🌐 REST API: http://localhost:3000"
        echo "📊 JSON-RPC: http://localhost:8545"
        echo "🔌 WebSocket: ws://localhost:8546"
        echo "📈 Metrics: http://localhost:9090"
        echo "📊 Grafana: http://localhost:3001"
        ;;
    "production")
        echo -e "${GREEN}🌐 Production Access Points:${NC}"
        echo "🌐 REST API: https://api.$DOMAIN"
        echo "📊 JSON-RPC: https://rpc.$DOMAIN"
        echo "🔌 WebSocket: wss://ws.$DOMAIN"
        echo "📈 Metrics: https://metrics.$DOMAIN"
        echo "📊 Grafana: https://$DOMAIN:3001"
        ;;
    "public")
        echo -e "${GREEN}🌍 Public Access Points:${NC}"
        echo "🌐 REST API: http://$PUBLIC_IP:3000"
        echo "📊 JSON-RPC: http://$PUBLIC_IP:8545"
        echo "🔌 WebSocket: ws://$PUBLIC_IP:8546"
        echo "📈 Metrics: http://$PUBLIC_IP:9090"
        echo "📊 Grafana: http://$PUBLIC_IP:3001"
        ;;
esac

echo -e "${BLUE}📱 MetaMask Configuration:${NC}"
echo "Network Name: ArthaChain Testnet"
echo "Chain ID: 1337"
case $ENV_FILE in
    "local")
        echo "RPC URL: http://localhost:8545"
        ;;
    "production")
        echo "RPC URL: https://rpc.$DOMAIN"
        ;;
    "public")
        echo "RPC URL: http://$PUBLIC_IP:8545"
        ;;
esac
echo "Currency Symbol: ARTHA"

echo -e "${BLUE}🔑 Grafana Credentials:${NC}"
echo "Username: admin"
echo "Password: $(grep GRAFANA_PASSWORD .env | cut -d'=' -f2)"

echo -e "${BLUE}🔧 Useful Commands:${NC}"
echo "View logs: docker-compose logs -f node"
echo "Restart: docker-compose restart"
echo "Stop: docker-compose down"
echo "Update: docker-compose pull && docker-compose up -d"

echo -e "${BLUE}🧪 Test Your Deployment:${NC}"
case $ENV_FILE in
    "local")
        echo "curl http://localhost:3000/api/health"
        echo "curl http://localhost:3000/api/stats"
        ;;
    "production")
        echo "curl https://api.$DOMAIN/health"
        echo "curl https://api.$DOMAIN/stats"
        ;;
    "public")
        echo "curl http://$PUBLIC_IP:3000/api/health"
        echo "curl http://$PUBLIC_IP:3000/api/stats"
        ;;
esac

echo -e "${GREEN}🎊 Happy blockchain building! 🎊${NC}"
