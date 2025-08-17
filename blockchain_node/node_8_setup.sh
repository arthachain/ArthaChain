#!/bin/bash
echo "🚀 Setting up ArthaChain Node 8"

# Install dependencies
apt update && apt install -y python3 curl git

# Download P2P node script
curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/blockchain_node/simple_p2p_server.py > arthachain_node.py
chmod +x arthachain_node.py

# Modify for this node
sed -i 's/port=30303/port=30308/' arthachain_node.py
sed -i 's/api_port=8080/api_port=8088/' arthachain_node.py

# Start node
echo "🔗 Starting ArthaChain Node 8"
echo "📡 P2P Port: 30308"
echo "🌐 API Port: 8088"
echo "🎯 Bootstrap: 223.228.101.153:30303"

python3 arthachain_node.py
