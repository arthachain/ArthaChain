# Multi-stage build for ArthaChain

FROM rust:1.78-bullseye AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    zlib1g-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    python3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy workspace configuration first
COPY ./Cargo.toml ./Cargo.lock ./rust-toolchain.toml ./

# Copy all source directories
COPY ./blockchain_node ./blockchain_node/
COPY ./src ./src/
COPY ./contracts ./contracts/
COPY ./telegram_faucet_bot ./telegram_faucet_bot/

# Set production build flags
ENV RUSTFLAGS="-C target-cpu=native"
ENV CARGO_BUILD_TARGET_DIR="/src/target"

# Pre-fetch dependencies for faster rebuilds
RUN cargo fetch

# Build the main blockchain node binary
RUN cargo build --release --package arthachain_node --bin arthachain

# Build API servers
RUN cargo build --release --package arthachain_node --bin api_server
RUN cargo build --release --package arthachain_node --bin real_time_api_server
RUN cargo build --release --package arthachain_node --bin testnet_api_server


FROM debian:bullseye-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd --gid 1000 arthachain \
    && useradd --uid 1000 --gid arthachain --shell /bin/bash --create-home arthachain

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /src/target/release/arthachain /app/arthachain
COPY --from=builder /src/target/release/api_server /app/api_server
COPY --from=builder /src/target/release/real_time_api_server /app/real_time_api_server
COPY --from=builder /src/target/release/testnet_api_server /app/testnet_api_server

# Copy configuration and data directories
COPY --from=builder /src/blockchain_node/testnet_config_global.toml /app/testnet_config.toml
COPY --from=builder /src/blockchain_node/models /app/models/

# Create data directory
RUN mkdir -p /app/testnet_data /app/logs && \
    chown -R arthachain:arthachain /app

# Expose ports
EXPOSE 3000 8545 8546 9090 30303

# Switch to non-root user
USER arthachain

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Default to running the main node with fallback to API servers
ENTRYPOINT ["/bin/sh", "-c", "exec /app/arthachain run --config /app/testnet_config.toml"]


