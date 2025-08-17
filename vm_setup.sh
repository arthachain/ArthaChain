#!/bin/bash
# ArthaChain VM Node - Simple Setup

echo "🚀 Setting up ArthaChain on VM (11.6 CPU, 24GB RAM, 1TB)"

# Update system
sudo apt update && sudo apt upgrade -y

# Install essentials
sudo apt install -y curl git build-essential pkg-config libssl-dev clang

# Install Rust
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Clone latest ArthaChain
git clone https://github.com/arthachain/ArthaChain.git
cd ArthaChain/blockchain_node

# Build optimized for VM
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Start node with bootstrap connection
echo "🌐 Starting ArthaChain node..."
echo "📡 Connecting to bootstrap node..."
nohup ./target/release/arthachain &

echo "✅ ArthaChain VM node is starting!"
echo "🔗 Will connect to existing network automatically"
echo "📊 Monitor with: tail -f nohup.out"
