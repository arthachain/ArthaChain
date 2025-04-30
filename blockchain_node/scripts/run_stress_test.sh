#!/bin/bash
set -e

echo "Building stress test..."
cd "$(dirname "$0")/.."
cargo build --bin stress_test --release

echo "Running stress test..."
./target/release/stress_test 