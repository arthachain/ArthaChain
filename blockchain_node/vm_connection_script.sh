#!/bin/bash
echo "🔗 Connecting VM node to main ArthaChain network..."
echo "🎯 Bootstrap node: 223.228.101.153:30303"

# Stop any existing independent node
pkill -f testnet_api_server
pkill -f arthachain

# Wait a moment
sleep 2

# Start unified P2P node
echo "🚀 Starting unified blockchain node..."
nohup cargo run --release --bin arthachain --config vm_p2p_config.toml > unified_blockchain.log 2>&1 &

echo "✅ VM node connecting to main blockchain..."
echo "📊 Monitor with: tail -f unified_blockchain.log"
echo "🔗 This node will sync with the main blockchain"
echo "🌐 API will be available at: http://VM_IP:8080"
