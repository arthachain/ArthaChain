# Artha Chain Testnet Setup

This guide explains how to set up and run a local testnet for Artha Chain using Docker.

## Prerequisites

- Docker and Docker Compose
- Curl (for API interaction)
- Bash shell

## Testnet Components

The testnet consists of:

- 4 validator nodes running the Artha Chain blockchain with SVCP and SVBFT consensus
- A simple block explorer web interface
- All components packaged in Docker containers for easy deployment

## Quick Start

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Start the testnet:
   ```
   ./testnet.sh start
   ```

3. Check the status:
   ```
   ./testnet.sh status
   ```

4. Generate some test transactions:
   ```
   ./testnet.sh generate-tx
   ```

5. Open the block explorer in your browser:
   ```
   http://localhost:8080
   ```

## Management Commands

The `testnet.sh` script provides several commands for managing the testnet:

- `./testnet.sh start` - Start the testnet
- `./testnet.sh stop` - Stop the testnet
- `./testnet.sh restart` - Restart the testnet
- `./testnet.sh status` - Check the status of the testnet
- `./testnet.sh logs` - Show logs (use -f to follow)
- `./testnet.sh logs --node=validator1` - Show logs for a specific node
- `./testnet.sh clean` - Clean all data (WARNING: Resets blockchain state)
- `./testnet.sh generate-tx` - Generate sample transactions for testing
- `./testnet.sh help` - Show help message

## Network Details

- Genesis configuration: `testnet-genesis.json`
- Data directory: `./data/`
- Blockchain node image: Built from local Dockerfile

### Node Endpoints

| Node | RPC Endpoint | API Endpoint | P2P Address |
|------|--------------|--------------|------------|
| validator1 | http://localhost:8545 | http://localhost:3000 | http://localhost:7000 |
| validator2 | http://localhost:8546 | http://localhost:3001 | http://localhost:7001 |
| validator3 | http://localhost:8547 | http://localhost:3002 | http://localhost:7002 |
| validator4 | http://localhost:8548 | http://localhost:3003 | http://localhost:7003 |

## API Interaction

You can interact with the nodes using the HTTP API:

```bash
# Get network status
curl http://localhost:3000/api/status

# Get latest blocks
curl http://localhost:3000/api/blocks?limit=5

# Get account balance
curl http://localhost:3000/api/accounts/validator1/balance

# Send a transaction
curl -X POST http://localhost:3000/api/transactions \
  -H "Content-Type: application/json" \
  -d '{"sender":"validator1","recipient":"user1","amount":100,"data":"Test transaction"}'
```

## Explorer

A simple block explorer is included and can be accessed at:
```
http://localhost:8080
```

The explorer provides basic information about:
- Network status
- Latest blocks
- Recent transactions
- Validator status

## Troubleshooting

### Common Issues

1. **Containers not starting properly:**
   Check the logs with `./testnet.sh logs`

2. **Nodes not connecting to each other:**
   Verify the bootstrap peers configuration in `docker-compose.yml`

3. **Explorer not showing data:**
   Make sure the API endpoints are accessible and working

4. **Reset the testnet:**
   Use `./testnet.sh clean` to reset all data and start fresh

## Configuration

You can modify the testnet configuration by editing:

- `docker-compose.yml` - Container setup and networking
- `testnet-genesis.json` - Initial blockchain state and validators
- `Dockerfile` - Node build process 