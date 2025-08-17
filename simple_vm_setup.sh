#!/bin/bash
# Simple ArthaChain VM Node Setup

# Install dependencies
sudo apt update && sudo apt install -y curl git build-essential

# Install Rust
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Clone and build
git clone https://github.com/arthachain/ArthaChain.git
cd ArthaChain/blockchain_node
cargo build --release --bin arthachain

# Run node
./target/release/arthachain

