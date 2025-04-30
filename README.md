# Artha Chain: Hybrid Consensus Blockchain

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
