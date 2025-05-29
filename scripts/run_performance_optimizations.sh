#!/bin/bash
set -e

echo "Building performance optimizations example..."
cargo build --release --example performance_optimizations

echo "Running performance optimizations example..."
RUST_LOG=info ./target/release/examples/performance_optimizations 