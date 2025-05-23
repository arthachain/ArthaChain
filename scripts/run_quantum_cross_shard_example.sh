#!/bin/bash

echo "Building and running the quantum cross-shard transaction example..."

# Use RUST_LOG for debug output
export RUST_LOG=debug

# Build and run the example
cd "$(dirname "$0")/.."
cargo build --example cross_shard_tx_example
cargo run --example cross_shard_tx_example

echo "Example completed." 