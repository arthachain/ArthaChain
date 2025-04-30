#!/bin/bash
set -e

echo "Running stress test..."
cd "$(dirname "$0")/.."
cargo run --bin stress_test 