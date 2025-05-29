#!/bin/bash
set -e

# This script builds only the performance optimization components
echo "Building quantum-resistant performance optimization components..."

# Create directory structure if it doesn't exist
mkdir -p blockchain_node/target/release/examples

# Build the individual components
echo "Building adaptive_gossip.rs..."
rustc -C opt-level=3 blockchain_node/src/network/adaptive_gossip.rs --crate-type lib --edition=2021 -o blockchain_node/target/release/examples/adaptive_gossip.rlib

echo "Building mempool.rs..."
rustc -C opt-level=3 blockchain_node/src/transaction/mempool.rs --crate-type lib --edition=2021 -o blockchain_node/target/release/examples/mempool.rlib

echo "Building quantum_merkle.rs..."
rustc -C opt-level=3 blockchain_node/src/utils/quantum_merkle.rs --crate-type lib --edition=2021 -o blockchain_node/target/release/examples/quantum_merkle.rlib

echo "Building quantum_cache.rs..."
rustc -C opt-level=3 blockchain_node/src/state/quantum_cache.rs --crate-type lib --edition=2021 -o blockchain_node/target/release/examples/quantum_cache.rlib

echo "Building documentation..."
cd docs
markdown performance_optimizations.md > performance_optimizations.html
cd ..

echo "Build completed successfully!"
echo "You can find the compiled components in blockchain_node/target/release/examples/"
echo "Documentation is available in docs/performance_optimizations.html" 