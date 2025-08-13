# Multi-stage build for ArthaChain

FROM rust:1.78-bullseye AS builder
RUN apt-get update && apt-get install -y \
    build-essential clang cmake pkg-config libssl-dev \
    libclang-dev zlib1g-dev libbz2-dev liblz4-dev libzstd-dev \
    python3 python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY .. .

# Pre-fetch dependencies for faster rebuilds
RUN cargo fetch

# Build only a minimal API server first to avoid heavy deps; fall back progressively
ENV RUSTFLAGS="--cfg skip_problematic_modules"
RUN cargo build --release -p blockchain_node --bin real_time_api_server \
 || cargo build --release --bin real_time_api_server \
 || cargo build --release -p blockchain_node --bin api_server \
 || cargo build --release --bin api_server \
 || cargo build --release -p blockchain_node --bin testnet_api_server \
 || cargo build --release --bin testnet_api_server \
 || cargo build --release -p arthachain_node \
 || cargo build --release


FROM debian:bullseye-slim AS runtime
RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Try common locations for the compiled binary(s)
COPY --from=builder /src/target/release/real_time_api_server /app/real_time_api_server || true
COPY --from=builder /src/target/release/api_server /app/api_server || true
COPY --from=builder /src/target/release/testnet_api_server /app/testnet_api_server || true
COPY --from=builder /src/target/release/arthachain /app/arthachain || true
COPY --from=builder /src/blockchain_node /app/blockchain_node

EXPOSE 3000 8545 8546 9090 30303
# Prefer the lightest server we managed to build
ENTRYPOINT ["/bin/sh","-lc","[ -x /app/real_time_api_server ] && exec /app/real_time_api_server || ([ -x /app/api_server ] && exec /app/api_server) || ([ -x /app/testnet_api_server ] && exec /app/testnet_api_server) || exec /app/arthachain run --config blockchain_node/testnet_config.toml"]


