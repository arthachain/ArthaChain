FROM rust:1.70-slim-bullseye as builder

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    git \
    ca-certificates \
    clang \
    cmake \
    libclang-dev && \
    rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy source code
COPY . .

# Build the blockchain node
RUN cargo build --release --bin arthachain

# Create a smaller runtime image
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libssl1.1 \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy the built binary
COPY --from=builder /build/target/release/arthachain /app/arthachain

# Create data directory
RUN mkdir -p /app/data

# Default command
ENTRYPOINT ["/app/arthachain"] 