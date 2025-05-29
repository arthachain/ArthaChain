.PHONY: all build test run docs clean

# Default target
all: build test

# Build performance optimization components
build:
	@echo "Building performance optimization components..."
	@mkdir -p target/release/examples
	cargo build --release --example performance_optimizations
	cargo build --release --example integration_example

# Run tests for performance optimization components
test:
	@echo "Running tests for performance optimization components..."
	@mkdir -p tests/performance
	bash scripts/test_performance_components.sh

# Run integration example
run:
	@echo "Running integration example..."
	RUST_LOG=info cargo run --release --example integration_example

# Run performance optimizations example
run-performance:
	@echo "Running performance optimizations example..."
	RUST_LOG=info cargo run --release --example performance_optimizations

# Build documentation
docs:
	@echo "Building documentation..."
	@mkdir -p docs/html
	markdown docs/performance_optimizations.md > docs/html/performance_optimizations.html

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf tests/performance
	rm -rf docs/html

# Component-specific builds
build-gossip:
	@echo "Building adaptive gossip component..."
	cargo build --release --lib --features="network"

build-mempool:
	@echo "Building enhanced mempool component..."
	cargo build --release --lib --features="transaction"

build-merkle:
	@echo "Building quantum merkle component..."
	cargo build --release --lib --features="utils"

build-cache:
	@echo "Building quantum cache component..."
	cargo build --release --lib --features="state"

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build and test components (default)"
	@echo "  build            - Build all performance optimization components"
	@echo "  test             - Run tests for all components"
	@echo "  run              - Run integration example"
	@echo "  run-performance  - Run performance optimizations example"
	@echo "  docs             - Build documentation"
	@echo "  clean            - Clean build artifacts"
	@echo "Component-specific builds:"
	@echo "  build-gossip     - Build adaptive gossip component"
	@echo "  build-mempool    - Build enhanced mempool component"
	@echo "  build-merkle     - Build quantum merkle component"
	@echo "  build-cache      - Build quantum cache component" 