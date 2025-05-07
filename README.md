# Artha Chain: Hybrid Consensus Blockchain

<p align="center">
  <img src="docs/assets/img/logo.png" alt="ArthaChain Logo" width="200" />
</p>

Artha Chain is a next-generation blockchain platform that implements a hybrid consensus mechanism combining Social Verified Consensus Protocol (SVCP), Social Verified Byzantine Fault Tolerance (SVBFT), and Objective Sharding for scalability.

## Key Features

- **SVCP Mining**: Selects block proposers based on social metrics and contribution (compute, network, storage, engagement, AI trust)
- **SVBFT Finality**: Fast finality based on the HotStuff BFT protocol, optimized for mobile and sharded chains
- **Objective Sharding**: Dynamic sharding that scales based on demand and node distribution
- **AI Security**: Built-in AI modules for security, identity, chunking, and device health
- **Social Engagement**: Rewards for social actions and contributions to the network
- **SVDB Storage**: Decentralized storage validation integrated with the blockchain

## Architecture

The Artha Chain architecture combines several innovative components:

1. **SVCP (Social Verified Consensus Protocol)**
   - Selects block proposers based on contribution metrics, not hash power or stake
   - Integrates compute, network, storage, engagement, and AI trust scores
   - Results in a more energy-efficient and fair consensus process

2. **SVBFT (Social Verified BFT)**
   - Derived from HotStuff, optimized for mobile devices
   - Provides fast finality (1-3 seconds)
   - Validator weights adjusted by social metrics

3. **Objective Sharding**
   - Dynamic TPS that increases with active miners
   - Auto shard resizing as miners join/leave
   - Cross-shard finality through SVBFT
   - Mobile-optimized shard assignment

## High Performance TPS Optimizations

Artha Chain implements cutting-edge optimizations to achieve up to 500,000 transactions per second (TPS):

1. **Massive Sharding Architecture**
   - Scaled from 4 to 128 shards with optimized cross-shard communication
   - Intelligent transaction routing to minimize cross-shard overhead
   - Custom resource monitoring and dynamic load balancing

2. **SIMD-Optimized Execution Engine**
   - Parallel transaction execution using CPU SIMD instructions
   - Work-stealing algorithm for optimal multi-core utilization
   - Batch processing with optimized memory access patterns

3. **Memory-Mapped Storage with Adaptive Compression**
   - Custom memory-mapped database for microsecond storage access
   - Adaptive compression switching between LZ4, Zstd, and Brotli
   - Inline storage for small values with zero-copy access

4. **Batched Zero-Knowledge Proofs System**
   - Parallel ZKP validation for transaction batches
   - Optimized cryptographic primitives for ARM and x86
   - Incremental verification for cross-shard transactions

5. **Custom UDP Network Protocol**
   - Binary serialization with minimal overhead
   - Reliable UDP with congestion control and selective acknowledgment
   - Message fragmentation and reassembly for large payloads

### Benchmark Results

Our benchmarks demonstrate impressive performance:
- Raw parallel processing: ~827,650 TPS
- Sharded transactions: ~420,767 TPS (378,286 intra-shard, 42,481 cross-shard)
- Storage performance: ~285 MB/s write, ~19.5 MB/s read
- End-to-end pipeline: ~193,761 TPS on a single machine

In a distributed environment with proper hardware, the system is projected to exceed 500,000 TPS.

## Getting Started

### Prerequisites

- Rust 1.70 or higher
- RocksDB dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/DiigooSai/ArthaChain.git
cd ArthaChain

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential librocksdb-dev libssl-dev pkg-config

# Build the project
cargo build --release

# Build with specific features
cargo build --release --features "ai_security sharding"
```

### Running a Node

```bash
# Run a node with default settings
./target/release/blockchain_node

# Run with custom configuration
./target/release/blockchain_node --data-dir ./data --p2p-listen-addr 0.0.0.0:7000

# Run as a genesis node
./target/release/blockchain_node --is-genesis --genesis-path ./testnet-genesis.json

# Run with Docker
docker-compose -f docker-compose-simple.yml up
```

### Configuration

Artha Chain can be configured using command-line arguments, environment variables, or Docker environment settings:

#### Core Settings
- `--data-dir` / `DATA_DIR`: Data directory for blockchain storage (default: `./data`)
- `--p2p-listen-addr` / `P2P_LISTEN_ADDR`: Listen address for P2P network (default: `0.0.0.0:7000`)
- `--rpc-listen-addr` / `RPC_LISTEN_ADDR`: Listen address for RPC server (default: `127.0.0.1:8545`)
- `--bootstrap-peers` / `BOOTSTRAP_PEERS`: Bootstrap peers (comma-separated multiaddresses)

#### Consensus Settings
- `--shard-id` / `SHARD_ID`: Shard ID (0 for primary shard)
- `--max-shards` / `MAX_SHARDS`: Maximum number of shards (default: 4)
- `--is-genesis` / `IS_GENESIS`: Run as a genesis node
- `--genesis-path` / `GENESIS_PATH`: Path to genesis configuration file

#### Feature Flags
- `--enable-ai` / `ENABLE_AI`: Enable AI security features
- `--enable-metrics` / `ENABLE_METRICS`: Enable Prometheus metrics
- `--enable-tracing` / `ENABLE_TRACING`: Enable OpenTelemetry tracing

#### Network Settings
- `--network` / `NETWORK`: Network to connect to (`mainnet`, `testnet`, `devnet`)
- `--max-peers` / `MAX_PEERS`: Maximum number of P2P connections (default: 50)
- `--min-peers` / `MIN_PEERS`: Minimum number of P2P connections (default: 3)

#### Docker Configuration
Use environment variables in `docker-compose.yml` or pass them directly:
```bash
docker run -e DATA_DIR=/data -e NETWORK=testnet arthachain/node:latest
```

For detailed configuration examples, see:
- `docker-compose.yml` - Full node setup
- `docker-compose-simple.yml` - Simple single node setup
- `docker-compose-single.yml` - Development setup
- `testnet-genesis.json` - Genesis configuration example

## Development

### Project Structure

```
arthachain/
├── blockchain_node/         # Main blockchain node
│   ├── config.rs            # Config & CLI args
│   ├── node.rs              # Node init & event loop
│   ├── network/             # P2P and RPC
│   ├── ledger/              # Blocks, transactions, state
│   ├── consensus/           # SVCP, SVBFT, sharding
│   ├── ai_engine/           # AI security components
│   └── utils/               # Crypto, logging, metrics
├── evm_compat/              # Lightweight EVM layer
└── sdk/                     # Development SDKs
```

### Building for Development

```bash
# Build with debug symbols
cargo build

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## Website

Visit our website at [https://arthachain.com](https://arthachain.com) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Genesis Setup and Network Launch

### Setting Up Genesis Node

To initialize and run a genesis node:

1. Configure your `genesis.json` file with the following parameters:
   - Chain ID: Either a numeric ID or a string identifier (e.g., "arthachain-mainnet" or "arthachain-testnet")
   - Validator set: List of initial validators with their node IDs and staking power
   - Initial state: Account balances for genesis accounts
   - Timestamp: Set this to a future time for coordinated launch

2. Start your node with the genesis flag:
   ```
   cargo run --bin blockchain_node -- --is-genesis --genesis-path ./path/to/genesis.json
   ```

3. For detailed validator coordination instructions, see [Validator Coordination Guide](docs/VALIDATOR_COORDINATION.md)

### Joining an Existing Network

To join an existing network:

1. Obtain the correct `genesis.json` file from existing network operators
2. Configure bootstrap peers to connect to the network
   ```
   cargo run --bin blockchain_node -- --bootstrap-peers /ip4/192.168.1.1/tcp/7000/p2p/QmHashOfPeer1,/ip4/192.168.1.2/tcp/7000/p2p/QmHashOfPeer2
   ```

## Testnet Features

### Faucet

The testnet includes a faucet service for obtaining test tokens. To enable it:

1. Add the following to your config:
   ```json
   {
     "faucet": {
       "enabled": true,
       "amount": 1000,
       "cooldown": 3600,
       "max_requests_per_ip": 5,
       "max_requests_per_account": 3,
       "address": "faucet"
     }
   }
   ```

2. Request tokens via the API:
   ```bash
   curl -X POST http://localhost:3000/faucet \
     -H "Content-Type: application/json" \
     -d '{"address":"your_address_here"}'
   ```

# Blockchain AI Engine

This directory contains the AI models and neural networks used by the blockchain for various functions including:

- BCI (Brain-Computer Interface) models
- Neural network processing
- Signal processing
- Self-learning systems

## Setup Instructions

### 1. Python Environment Setup

The AI components require PyTorch and NumPy. To set them up:

```bash
# Run the installation script to create a Python virtual environment with all required dependencies
./install_deps.sh
```

This script will:
- Create a Python virtual environment in `./venv`
- Install PyTorch, NumPy and other dependencies
- Configure environment variables
- Create an `.env` file for VS Code to recognize paths

### 2. VS Code Integration

For proper VS Code integration:

1. Install the Python extension in VS Code
2. Select the Python interpreter from the venv:
   - Cmd+Shift+P -> Python: Select Interpreter -> Select the venv python
3. Reload VS Code window after setup:
   - Cmd+Shift+P -> Developer: Reload Window

### 3. Fallback for PyTorch-less Environments

If PyTorch cannot be installed (e.g., in CI environments or for quick testing), the codebase includes a mock implementation:

- `mock_torch.py` provides a minimal implementation that allows code to pass linting and basic functionality checks
- The import system will automatically fall back to the mock version if the real PyTorch is not available

## Code Structure

- `neural_base.rs` - Core neural network infrastructure
- `bci_interface.rs` - Brain-Computer Interface model implementations
- `registry.rs` - Model registry for managing AI models
- `adaptive_network.py` - Python implementation of the adaptive neural network
- `spike_detector.py` - Spike detection for BCI signals
- `decoder.py` - Neural decoder for interpreting BCI signals

## Integration with Rust

This codebase uses PyO3 to integrate Python with Rust:

- Rust code can instantiate and use Python objects
- Neural models defined in Python are called from Rust
- The Rust side manages storage, configuration, and coordination

## Troubleshooting

### "Import 'torch' could not be resolved"

If you're seeing this error in VS Code:

1. Make sure you've run `./install_deps.sh`
2. Verify VS Code is using the correct Python interpreter from venv
3. Check that the `.env` file was created correctly
4. Reload VS Code window

### Cannot Find PyTorch

The code is designed to fall back to a mock implementation if PyTorch is not available. If you're encountering runtime errors:

1. Check if PyTorch is installed in your venv: `source venv/bin/activate && pip list | grep torch`
2. If not, run `pip install torch` in your activated venv
3. Alternatively, you can use the mock implementation for basic functionality

# AI Engine Models

This directory contains neural network models used by the blockchain's AI engine for brain-computer interfaces, signal processing, and self-learning systems.

## Models

The following models are implemented:

1. **Adaptive Network** - A neural network with attention mechanisms for adaptive learning
2. **Spike Detector** - A convolutional neural network for detecting neural spikes in brain signals
3. **Neural Decoder** - A model that decodes neural activity into commands and actions

## Installation

To set up the environment for the AI models:

```bash
# Run the installation script
./install_deps.sh
```

This script will:
- Create a Python virtual environment
- Install PyTorch, NumPy, and other dependencies
- Set up environment variables

## Testing

To test the models, run:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the test script
python test_models.py
```

## Usage in Rust

These models are used within the Rust codebase via PyO3 bindings. The `NeuralBase`, `BCIModel`, and `SelfLearningSystem` structs provide interfaces to the Python models.

## Environment Variables

- `PYTHONPATH` - Set to include the model directory
- `PYTORCH_ENABLE_MPS_FALLBACK` - Enables Metal Performance Shaders fallback for Apple Silicon

## Mock Implementation

For environments where PyTorch cannot be installed, a mock implementation is provided in `mock_torch.py`. This allows the codebase to compile and run basic tests without requiring the full PyTorch installation.
