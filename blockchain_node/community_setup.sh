#!/bin/bash

echo "🚀 ArthaChain Community Node Setup"
echo "🌐 Joining the ArthaChain Network..."
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Running as root - creating user 'arthachain'"
    useradd -m -s /bin/bash arthachain 2>/dev/null || true
    echo "🔄 Switching to arthachain user..."
    su - arthachain -c "curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/blockchain_node/community_setup.sh | bash"
    exit 0
fi

# Update system
echo "📦 Updating system..."
sudo apt update -qq

# Install dependencies
echo "🔧 Installing dependencies..."
sudo apt install -y python3 python3-pip curl git >/dev/null 2>&1

# Auto-detect free ports
find_free_port() {
    local start_port=$1
    for port in $(seq $start_port $((start_port + 100))); do
        if ! netstat -ln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return
        fi
    done
    echo $((start_port + 50))
}

P2P_PORT=$(find_free_port 30301)
API_PORT=$(find_free_port 8081)

echo "🔧 Auto-detected ports:"
echo "   P2P Port: $P2P_PORT"
echo "   API Port: $API_PORT"

# Download ArthaChain node
echo "📥 Downloading ArthaChain node..."
curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/blockchain_node/synchronized_p2p_node.py > arthachain_node.py

# Configure ports
echo "⚙️ Configuring node..."
python3 -c "
import re
with open('arthachain_node.py', 'r') as f:
    content = f.read()
content = re.sub(r'self\.p2p_port = self\.find_free_port\(30301\)', 'self.p2p_port = $P2P_PORT', content)
content = re.sub(r'self\.api_port = self\.find_free_port\(8081\)', 'self.api_port = $API_PORT', content)
with open('arthachain_node.py', 'w') as f:
    f.write(content)
"

# Configure firewall
echo "🔥 Configuring firewall..."
sudo ufw allow $P2P_PORT/tcp >/dev/null 2>&1
sudo ufw allow $API_PORT/tcp >/dev/null 2>&1

# Create service directory
mkdir -p ~/arthachain_logs

echo "🚀 Starting ArthaChain node..."
echo "📡 P2P Port: $P2P_PORT"
echo "�� API Port: $API_PORT"
echo "🔗 Network: ArthaChain Mainnet"
echo ""
echo "✅ Node starting in background..."

# Start in background
nohup python3 arthachain_node.py > ~/arthachain_logs/node.log 2>&1 &
NODE_PID=$!

sleep 5

# Verify startup
if ps -p $NODE_PID > /dev/null; then
    echo "✅ ArthaChain node started successfully!"
    echo "🆔 Process ID: $NODE_PID"
    echo "📊 API: http://localhost:$API_PORT/api/stats"
    echo "📝 Logs: tail -f ~/arthachain_logs/node.log"
    echo ""
    echo "🎉 Welcome to the ArthaChain network!"
    echo "🌐 Your node is now contributing to the blockchain!"
else
    echo "❌ Node startup failed. Check logs:"
    echo "   tail -f ~/arthachain_logs/node.log"
fi

echo ""
echo "🔗 Network Status: https://api.arthachain.in/api/stats"
echo "📚 Documentation: https://github.com/arthachain/ArthaChain"
