#!/bin/bash
set -e

# Run the WASM contract example
echo "Running WebAssembly Smart Contract Example..."

# Build the example
cargo build --example wasm_contract_example

# Run with debug logging enabled
RUST_LOG=debug ./target/debug/examples/wasm_contract_example

echo "Example completed!" 