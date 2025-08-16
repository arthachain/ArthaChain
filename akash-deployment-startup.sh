#!/bin/bash
# ArthaChain Node Startup Script for Akash Deployment
# Run this script after deploying to Akash Network

set -e

echo "🚀 ArthaChain Node - Akash Deployment Startup"
echo "============================================="
echo "Deployment: $(hostname)"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"

# Setup data directory
if [ -d "/mnt/data" ]; then
    export DATA_DIR="/mnt/data/arthachain"
    echo "📁 Using persistent storage: $DATA_DIR"
else
    export DATA_DIR="/home/arthachain"
    echo "📁 Using local storage: $DATA_DIR"
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Install basic dependencies if needed
echo "📦 Installing dependencies..."
apt-get update -y
apt-get install -y curl wget git jq

# Download and run the installer
echo "⬇️ Downloading ArthaChain installer..."
curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/arthachain-node-installer.sh -o installer.sh
chmod +x installer.sh

# Determine node ID based on hostname or generate random
NODE_ID=$(echo $HOSTNAME | grep -o '[0-9]*' | head -1)
if [ -z "$NODE_ID" ] || [ "$NODE_ID" -lt 2 ] || [ "$NODE_ID" -gt 10 ]; then
    NODE_ID=$((2 + RANDOM % 9))  # Random between 2-10
fi

echo "🏷️ Assigned Node ID: $NODE_ID"

# Run the installer
echo "🚀 Starting ArthaChain installation..."
./installer.sh "$NODE_ID"

echo "✅ ArthaChain node ready!"
echo "🌐 Node ID: $NODE_ID"
echo "📂 Data: $DATA_DIR"
echo "🔗 API: http://$(hostname):8080"
