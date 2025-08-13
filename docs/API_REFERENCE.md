# ArthaChain API Reference

Complete API documentation for ArthaChain blockchain platform.

## Table of Contents

1. [Core Blockchain API](#core-blockchain-api)
2. [JSON-RPC API](#json-rpc-api)
3. [WebSocket API](#websocket-api)
4. [Wallet Integration API](#wallet-integration-api)
5. [Fraud Monitoring API](#fraud-monitoring-api)
6. [Recovery API](#recovery-api)
7. [Cross-Shard API](#cross-shard-api)

---

## Core Blockchain API

The core blockchain API provides fundamental blockchain operations.

### Endpoints

#### Get Blockchain Information
```rust
pub async fn get_blockchain_info(&self) -> Result<BlockchainInfoResponse>
```

**Response:**
```json
{
  "latest_block_height": 12345,
  "latest_block_hash": "0x123...",
  "total_transactions": 98765,
  "total_validators": 21
}
```

#### Get Block by Hash
```rust
pub async fn get_block_by_hash(&self, hash: &Hash) -> Result<Option<Block>>
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/block/hash \
  -H "Content-Type: application/json" \
  -d '{"block_hash": "0x123..."}'
```

#### Submit Transaction
```rust
pub async fn submit_transaction(&self, transaction: Transaction) -> Result<String>
```

**Example:**
```json
{
  "from": "0x123...",
  "to": "0x456...",
  "value": 1000000000000000000,
  "gas_price": 20000000000,
  "gas_limit": 21000,
  "nonce": 1,
  "data": "0x",
  "signature": "0x789..."
}
```

---

## JSON-RPC API

ArthaChain implements JSON-RPC 2.0 specification with blockchain-specific methods.

### Supported Methods

| Method | Description |
|--------|-------------|
| `getBlockchainInfo` | Get blockchain status |
| `getBlockByHash` | Get block by hash |
| `getBlockByHeight` | Get block by height |
| `submitTransaction` | Submit transaction |
| `getBalance` | Get account balance |
| `getNonce` | Get account nonce |
| `getPendingTransactions` | Get pending transactions |
| `getRecentBlocks` | Get recent blocks |

### Example Request
```json
{
  "jsonrpc": "2.0",
  "method": "getBalance",
  "params": "0x123...",
  "id": 1
}
```

### Example Response
```json
{
  "jsonrpc": "2.0",
  "result": {
    "balance": 1000000000000000000
  },
  "id": 1
}
```

---

## WebSocket API

Real-time event streaming for blockchain updates.

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
```

### Event Types

#### Subscribe to Events
```json
{
  "action": "subscribe",
  "events": ["new_block", "new_transaction", "consensus_update"]
}
```

#### Event Responses

**New Block:**
```json
{
  "type": "new_block",
  "data": {
    "hash": "0x123...",
    "height": 12345,
    "tx_count": 10,
    "timestamp": 1640995200
  }
}
```

**New Transaction:**
```json
{
  "type": "new_transaction", 
  "data": {
    "hash": "0x456...",
    "sender": "0x123...",
    "recipient": "0x789...",
    "amount": 1000000000000000000
  }
}
```

---

## Wallet Integration API

Comprehensive wallet and IDE support.

### Supported Wallets

#### EVM Wallets (15+)
- MetaMask
- Trust Wallet
- Coinbase Wallet
- WalletConnect
- Rainbow Wallet
- Phantom (EVM mode)
- Brave Wallet
- 1inch Wallet
- Argent
- Zerion

#### WASM Wallets (5+)
- Phantom
- Solflare
- Backpack
- Slope
- Glow

### Chain Configuration
```json
{
  "chainId": "0x1337",
  "chainName": "ArthaChain Mainnet",
  "nativeCurrency": {
    "name": "ARTHA",
    "symbol": "ARTHA",
    "decimals": 18
  },
  "rpcUrls": ["https://rpc.arthachain.online"],
  "blockExplorerUrls": ["https://explorer.arthachain.online"]
}
```

### IDE Support (10+)
- Remix IDE
- Hardhat
- Truffle
- Foundry
- Brownie
- OpenZeppelin Defender
- Solana Playground
- Anchor Framework
- AssemblyScript Studio
- CosmWasm Studio

---

## Fraud Monitoring API

AI-powered fraud detection with real-time analytics.

### Endpoints

#### Dashboard Statistics
```
GET /api/fraud/dashboard
```

**Response:**
```json
{
  "total_transactions": 123456,
  "suspicious_transactions": 123,
  "suspicious_rate": 0.001,
  "last_update": "2024-01-01T00:00:00Z",
  "risk_level_counts": {
    "Low": 100,
    "Medium": 20,
    "High": 3
  },
  "recent_detections": [...],
  "top_risky_addresses": [...],
  "detection_rate_history": [...]
}
```

#### Detection History
```
GET /api/fraud/history?limit=100&risk_level=high
```

#### Transaction Details
```
GET /api/fraud/transaction/{tx_hash}
```

### Risk Levels
- **Low**: Normal transactions
- **Medium**: Requires attention
- **High**: Highly suspicious
- **Critical**: Immediate action required

---

## Recovery API

Enterprise disaster recovery and system management.

### Operations

#### Restart from Checkpoint
```json
{
  "operation": "RestartFromCheckpoint",
  "force": false,
  "parameters": {}
}
```

#### Force Leader Election
```json
{
  "operation": "ForceLeaderElection",
  "force": false,
  "parameters": {}
}
```

#### Restore from Backup
```json
{
  "operation": {
    "RestoreFromBackup": {
      "backup_id": "backup_20240101_000000"
    }
  },
  "force": false,
  "parameters": {}
}
```

#### System Status
```
GET /api/recovery/status
```

**Response:**
```json
{
  "overall_health": "healthy",
  "consensus": {
    "state": "Normal",
    "current_leader": "node_1",
    "round": 12345,
    "validators": ["node_1", "node_2", "node_3"],
    "is_healthy": true
  },
  "storage": {
    "is_healthy": true,
    "last_backup": "backup_20240101_000000",
    "available_backups": [...],
    "corruption_detected": false
  },
  "network": {
    "active_connections": 10,
    "partitions": [],
    "is_healthy": true,
    "peer_count": 15
  },
  "recovery": {
    "in_progress": false,
    "operation": null,
    "progress": 0.0,
    "estimated_completion": null
  },
  "uptime_secs": 86400,
  "last_checkpoint": "checkpoint_20240101_120000"
}
```

---

## Cross-Shard API

Advanced cross-shard transaction support.

### Submit Cross-Shard Transaction
```
POST /api/transaction/cross-shard
```

**Request:**
```json
{
  "from_shard": 0,
  "to_shard": 1,
  "from_address": "0x123...",
  "to_address": "0x456...",
  "amount": 1000000000000000000,
  "gas_limit": 100000
}
```

**Response:**
```json
{
  "transaction_id": "tx_12345",
  "status": "pending",
  "message": "Transaction submitted successfully"
}
```

### Get Transaction Status
```
GET /api/transaction/status/{tx_id}
```

**Response:**
```json
{
  "transaction_id": "tx_12345",
  "phase": "Commit",
  "status": "Success",
  "timestamp": 1640995200
}
```

### Network Statistics
```
GET /api/network/stats
```

**Response:**
```json
{
  "total_shards": 4,
  "active_nodes": 12,
  "pending_transactions": 5,
  "processed_transactions": 98765,
  "network_health": 0.95
}
```

---

## Error Handling

All APIs use consistent error handling:

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

### Error Response Format
```json
{
  "error": {
    "code": -32603,
    "message": "Error description",
    "data": null
  }
}
```

---

## Rate Limiting

- **API Calls**: 1000 requests per minute per IP
- **WebSocket Connections**: 10 concurrent connections per IP
- **Transaction Submission**: 100 transactions per minute per address

---

## Authentication

### API Key (Optional)
```bash
curl -H "X-API-Key: your_api_key" \
     -X GET http://localhost:8080/api/blockchain/info
```

### Wallet Signatures (For sensitive operations)
```json
{
  "address": "0x123...",
  "signature": "0x456...",
  "message": "operation_specific_message"
}
```

---

## Examples

### Complete Transaction Flow
```javascript
// 1. Get nonce
const nonce = await rpc('getNonce', address);

// 2. Create transaction
const tx = {
  from: address,
  to: recipient,
  value: amount,
  gas_price: 20000000000,
  gas_limit: 21000,
  nonce: nonce,
  data: "0x"
};

// 3. Sign transaction
const signature = await wallet.sign(tx);
tx.signature = signature;

// 4. Submit transaction
const hash = await rpc('submitTransaction', tx);

// 5. Monitor via WebSocket
ws.send(JSON.stringify({
  action: "subscribe",
  events: ["new_transaction"]
}));
```

### Fraud Monitoring Integration
```javascript
// Monitor suspicious activity
const response = await fetch('/api/fraud/dashboard');
const stats = await response.json();

if (stats.suspicious_rate > 0.05) {
  console.warn('High fraud rate detected:', stats.suspicious_rate);
  
  // Get details
  const history = await fetch('/api/fraud/history?risk_level=high');
  const incidents = await history.json();
  
  incidents.forEach(incident => {
    console.log('Suspicious transaction:', incident.tx_hash);
  });
}
```

---

## SDK Examples

### JavaScript/TypeScript
```bash
npm install arthachain-sdk
```

```javascript
import { ArthaChain } from 'arthachain-sdk';

const client = new ArthaChain({
  rpcUrl: 'http://localhost:8080/rpc',
  wsUrl: 'ws://localhost:8080/ws'
});

const balance = await client.getBalance(address);
const block = await client.getBlockByHeight(12345);
```

### Python
```bash
pip install arthachain-py
```

```python
from arthachain import Client

client = Client('http://localhost:8080/rpc')

balance = client.get_balance(address)
block = client.get_block_by_height(12345)
```

### Rust
```toml
[dependencies]
arthachain-client = "0.1.0"
```

```rust
use arthachain_client::Client;

let client = Client::new("http://localhost:8080/rpc");
let balance = client.get_balance(address).await?;
let block = client.get_block_by_height(12345).await?;
```

---

This comprehensive API reference covers all major ArthaChain APIs with practical examples and complete documentation.
