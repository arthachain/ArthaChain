#!/bin/bash
# 🚀 ArthaChain DigitalOcean One-Click Deployment Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 ArthaChain Global Testnet - DigitalOcean Deployment${NC}"
echo "=================================================="

# Check for doctl
if ! command -v doctl &> /dev/null; then
    echo -e "${YELLOW}Installing DigitalOcean CLI (doctl)...${NC}"
    wget https://github.com/digitalocean/doctl/releases/download/v1.98.1/doctl-1.98.1-linux-amd64.tar.gz
    tar xf doctl-1.98.1-linux-amd64.tar.gz
    sudo mv doctl /usr/local/bin
    rm doctl-1.98.1-linux-amd64.tar.gz
fi

# Get user input
echo -e "${YELLOW}Please provide the following information:${NC}"
read -p "DigitalOcean API Token: " DO_TOKEN
read -p "Droplet Name (default: arthachain-testnet): " DROPLET_NAME
read -p "Region (nyc3/sfo3/lon1/sgp1, default: nyc3): " REGION
read -p "Size (s-2vcpu-4gb/s-4vcpu-8gb, default: s-4vcpu-8gb): " SIZE

# Set defaults
DROPLET_NAME=${DROPLET_NAME:-arthachain-testnet}
REGION=${REGION:-nyc3}
SIZE=${SIZE:-s-4vcpu-8gb}

# Authenticate
echo -e "${BLUE}Authenticating with DigitalOcean...${NC}"
doctl auth init -t $DO_TOKEN

# Create SSH key if needed
if ! doctl compute ssh-key list | grep -q "arthachain-key"; then
    echo -e "${BLUE}Creating SSH key...${NC}"
    ssh-keygen -t ed25519 -f ~/.ssh/arthachain-key -N ""
    doctl compute ssh-key create arthachain-key --public-key-file ~/.ssh/arthachain-key.pub
fi

# Get SSH key ID
SSH_KEY_ID=$(doctl compute ssh-key list --format ID,Name --no-header | grep arthachain-key | awk '{print $1}')

# Create user data script
cat > /tmp/arthachain-userdata.sh << 'EOF'
#!/bin/bash
# ArthaChain Testnet Setup Script

# Update system
apt-get update && apt-get upgrade -y

# Install dependencies
apt-get install -y build-essential pkg-config libssl-dev git curl ufw nginx certbot python3-certbot-nginx

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source /root/.cargo/env

# Clone ArthaChain
cd /root
git clone https://github.com/arthachain/arthachain.git
cd arthachain

# Build the project
cargo build --release

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address)

# Configure the node with public IP
cat > blockchain_node/testnet_config.toml << EOCONFIG
[network]
network_id = "arthachain-global-testnet-1"
chain_id = 1337
name = "ArthaChain Global Testnet"

[node]
node_id = "do-${DROPLET_NAME}"
data_dir = "./testnet_data"
log_level = "info"
public_ip = "${PUBLIC_IP}"

[network_p2p]
listen_addr = "0.0.0.0:30303"
external_addr = "/ip4/${PUBLIC_IP}/tcp/30303"
max_peers = 100
boot_nodes = [
    "/ip4/147.182.246.123/tcp/30303/p2p/12D3KooWGlobalNode1",
    "/ip4/164.92.123.45/tcp/30303/p2p/12D3KooWGlobalNode2",
]
enable_nat = true

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
rate_limit = 100

[metrics]
enabled = true
prometheus_addr = "0.0.0.0"
prometheus_port = 9090
EOCONFIG

# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 30303/tcp
ufw allow 8545/tcp
ufw allow 8546/tcp
ufw allow 3000/tcp
ufw allow 9090/tcp
ufw --force enable

# Create systemd service
cat > /etc/systemd/system/arthachain.service << EOSERVICE
[Unit]
Description=ArthaChain Global Testnet Node
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/arthachain
ExecStart=/root/arthachain/target/release/arthachain run --config blockchain_node/testnet_config.toml
Restart=always
RestartSec=10
Environment="RUST_LOG=info"

[Install]
WantedBy=multi-user.target
EOSERVICE

# Start the service
systemctl daemon-reload
systemctl enable arthachain
systemctl start arthachain

# Configure Nginx
cat > /etc/nginx/sites-available/arthachain << EONGINX
server {
    listen 80;
    server_name ${PUBLIC_IP};

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }

    location /rpc {
        proxy_pass http://localhost:8545;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    location /ws {
        proxy_pass http://localhost:8546;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /metrics {
        proxy_pass http://localhost:9090;
        proxy_set_header Host \$host;
    }
}
EONGINX

ln -s /etc/nginx/sites-available/arthachain /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
systemctl restart nginx

# Create welcome page
mkdir -p /var/www/html
cat > /var/www/html/index.html << EOHTML
<!DOCTYPE html>
<html>
<head>
    <title>ArthaChain Global Testnet Node</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; }
        .container { max-width: 800px; margin: 0 auto; }
        .status { color: green; font-size: 24px; }
        .endpoint { background: #f0f0f0; padding: 10px; margin: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ArthaChain Global Testnet Node</h1>
        <p class="status">✅ Node is running!</p>
        
        <h2>📡 API Endpoints</h2>
        <div class="endpoint">REST API: http://${PUBLIC_IP}:3000</div>
        <div class="endpoint">JSON-RPC: http://${PUBLIC_IP}:8545</div>
        <div class="endpoint">WebSocket: ws://${PUBLIC_IP}:8546</div>
        <div class="endpoint">Metrics: http://${PUBLIC_IP}:9090</div>
        
        <h2>🔗 Quick Links</h2>
        <p><a href="/health">Health Check</a> | <a href="/metrics">Metrics</a></p>
        
        <h2>📱 Add to MetaMask</h2>
        <p>Network Name: ArthaChain Testnet<br>
        RPC URL: http://${PUBLIC_IP}:8545<br>
        Chain ID: 1337<br>
        Symbol: ARTHA</p>
    </div>
</body>
</html>
EOHTML

echo "✅ ArthaChain Testnet deployment complete!"
echo "🌐 Access your node at: http://${PUBLIC_IP}"
EOF

# Create the droplet
echo -e "${BLUE}Creating DigitalOcean droplet...${NC}"
DROPLET_ID=$(doctl compute droplet create $DROPLET_NAME \
    --region $REGION \
    --image ubuntu-22-04-x64 \
    --size $SIZE \
    --ssh-keys $SSH_KEY_ID \
    --user-data-file /tmp/arthachain-userdata.sh \
    --format ID \
    --no-header \
    --wait)

# Get droplet IP
echo -e "${BLUE}Waiting for droplet to be ready...${NC}"
sleep 30
DROPLET_IP=$(doctl compute droplet get $DROPLET_ID --format PublicIPv4 --no-header)

# Clean up
rm /tmp/arthachain-userdata.sh

# Display results
echo -e "${GREEN}=================================================="
echo -e "✅ DEPLOYMENT SUCCESSFUL!"
echo -e "=================================================="
echo -e "Droplet Name: ${DROPLET_NAME}"
echo -e "Droplet ID: ${DROPLET_ID}"
echo -e "Public IP: ${DROPLET_IP}"
echo -e ""
echo -e "🌐 Access Points:"
echo -e "REST API: http://${DROPLET_IP}:3000"
echo -e "JSON-RPC: http://${DROPLET_IP}:8545"
echo -e "WebSocket: ws://${DROPLET_IP}:8546"
echo -e "Metrics: http://${DROPLET_IP}:9090"
echo -e ""
echo -e "SSH Access: ssh root@${DROPLET_IP} -i ~/.ssh/arthachain-key"
echo -e ""
echo -e "⏳ Note: Node initialization may take 5-10 minutes"
echo -e "Check status: curl http://${DROPLET_IP}:3000/health"
echo -e "==================================================${NC}"

# Save connection info
cat > arthachain-testnet-info.txt << EOF
ArthaChain Global Testnet Information
=====================================
Droplet Name: $DROPLET_NAME
Droplet ID: $DROPLET_ID
Public IP: $DROPLET_IP
Region: $REGION
Size: $SIZE

API Endpoints:
- REST API: http://$DROPLET_IP:3000
- JSON-RPC: http://$DROPLET_IP:8545
- WebSocket: ws://$DROPLET_IP:8546
- Metrics: http://$DROPLET_IP:9090

SSH Access: ssh root@$DROPLET_IP -i ~/.ssh/arthachain-key

MetaMask Configuration:
- Network Name: ArthaChain Testnet
- RPC URL: http://$DROPLET_IP:8545
- Chain ID: 1337
- Symbol: ARTHA
EOF

echo -e "${YELLOW}Connection info saved to: arthachain-testnet-info.txt${NC}"
