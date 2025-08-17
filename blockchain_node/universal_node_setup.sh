#!/bin/bash

echo "🚀 ArthaChain Universal Node Setup"
echo "🌐 Join the ArthaChain Network"
echo ""

# Auto-detect available ports
find_free_port() {
    local start_port=$1
    local port=$start_port
    while netstat -ln 2>/dev/null | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo $port
}

# Find free ports automatically
P2P_PORT=$(find_free_port 30301)
API_PORT=$(find_free_port 8081)

echo "🔧 Auto-detected ports:"
echo "   P2P Port: $P2P_PORT"
echo "   API Port: $API_PORT"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
apt update -qq
apt install -y python3 curl git >/dev/null 2>&1

# Download universal node script
echo "📥 Downloading ArthaChain node..."
curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/blockchain_node/simple_p2p_server.py > arthachain_node.py

# Configure for this node
echo "⚙️ Configuring node..."
python3 -c "
import re
with open('arthachain_node.py', 'r') as f:
    content = f.read()
content = re.sub(r'port=30303', 'port=$P2P_PORT', content)
content = re.sub(r'api_port=8080', 'api_port=$API_PORT', content)
with open('arthachain_node.py', 'w') as f:
    f.write(content)
"

# Open firewall
echo "🔥 Configuring firewall..."
ufw allow $P2P_PORT/tcp >/dev/null 2>&1
ufw allow $API_PORT/tcp >/dev/null 2>&1

# Start node
echo "🚀 Starting ArthaChain node..."
echo "📡 P2P Port: $P2P_PORT"
echo "🌐 API Port: $API_PORT"
echo "🔗 Bootstrap: 223.228.101.153:30303"
echo ""
echo "✅ Node starting... Press Ctrl+C to run in background"

python3 arthachain_node.py
