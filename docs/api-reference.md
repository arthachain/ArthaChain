# 📱 ArthaChain API Reference

**Complete API documentation for building applications on ArthaChain.** Copy-paste examples that actually work!

## 🎯 What You'll Find Here

- **🌐 Base URLs & Networks** - Where to connect
- **🔐 Authentication** - API keys and rate limits  
- **📦 REST APIs** - HTTP endpoints for everything
- **⚡ WebSocket APIs** - Real-time blockchain updates
- **🔗 JSON-RPC APIs** - Ethereum-compatible interface
- **🛡️ Fraud Detection APIs** - AI-powered security
- **💰 Faucet APIs** - Get test tokens programmatically
- **📊 Examples** - Copy-paste working code

## 🌐 Base URLs & Networks

| Network | Base URL | Chain ID | Purpose |
|---------|----------|----------|---------|
| **🧪 Testnet** | `https://testnet.arthachain.online` | `artha-testnet-1` | Development & Testing |
| **🚀 Mainnet** | `https://api.arthachain.com` | `artha-mainnet-1` | Production Applications |

### 🔌 Available Endpoints

| Service | Testnet | Mainnet |
|---------|---------|---------|
| **REST API** | `https://testnet.arthachain.online/api` | `https://api.arthachain.com/api` |
| **JSON-RPC** | `https://testnet.arthachain.online/rpc` | `https://api.arthachain.com/rpc` |
| **WebSocket** | `wss://testnet.arthachain.online/ws` | `wss://api.arthachain.com/ws` |
| **Explorer** | `https://testnet.arthachain.online` | `https://explorer.arthachain.com` |
| **Faucet** | `https://faucet.arthachain.online` | N/A (Mainnet has no faucet) |

## 🔐 Authentication & Rate Limits

### 📋 Public Access (No API Key Needed)
Most read operations are completely free:

```bash
# Get latest block - no authentication needed
curl https://testnet.arthachain.online/api/blocks/latest
```

### 🔑 API Key Authentication (Optional)
For higher rate limits and premium features:

```bash
# Add API key to headers
curl -H "X-API-Key: your-api-key-here" \
     https://api.arthachain.com/api/blocks/latest
```

**Get an API key**: [developer.arthachain.com/api-keys](https://developer.arthachain.com/api-keys)

### 🚦 Rate Limits

| Tier | Requests/Minute | WebSocket Connections | Features |
|------|-----------------|----------------------|----------|
| **Free** | 100 | 5 | Basic read operations |
| **Developer** | 1,000 | 25 | Submit transactions |
| **Pro** | 10,000 | 100 | Advanced analytics |
| **Enterprise** | Unlimited | Unlimited | Custom support |

**Rate limit headers** are included in every response:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95  
X-RateLimit-Reset: 1640345400
```

## 📦 REST API Endpoints

### 🧱 Block Endpoints

#### Get Latest Block
Get information about the most recent block.

**Endpoint:** `GET /api/blocks/latest`

```bash
curl https://testnet.arthachain.online/api/blocks/latest
```

**Response:**
```json
{
  "height": 142857,
  "hash": "0x1a2b3c4d5e6f7890...",
  "timestamp": "2024-12-16T10:30:00Z",
  "proposer": "artha1validator123...",
  "num_txs": 25,
  "total_size": 15420,
  "gas_used": 2500000,
  "gas_limit": 10000000,
  "previous_hash": "0x9f8e7d6c5b4a3210...",
  "merkle_root": "0x3c2b1a9d8e7f6543...",
  "state_root": "0x7f6e5d4c3b2a1098...",
  "transactions": [
    {
      "hash": "0xabc123def456...",
      "from": "artha1sender123...",
      "to": "artha1recipient456...",
      "amount": "1000000000000000000",
      "gas_used": 21000,
      "status": "success"
    }
  ],
  "consensus_info": {
    "view": 12345,
    "round": 1,
    "validators_signed": 67,
    "total_validators": 100
  }
}
```

#### Get Block by Hash
Retrieve a specific block by its hash.

**Endpoint:** `GET /api/blocks/{hash}`

```bash
curl https://testnet.arthachain.online/api/blocks/0x1a2b3c4d5e6f7890...
```

#### Get Block by Height
Retrieve a specific block by its height.

**Endpoint:** `GET /api/blocks/height/{height}`

```bash
curl https://testnet.arthachain.online/api/blocks/height/142857
```

#### Get Blocks (Paginated)
Retrieve multiple blocks with pagination.

**Endpoint:** `GET /api/blocks`

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 20, max: 100)
- `order` (optional): `asc` or `desc` (default: desc)

```bash
curl "https://testnet.arthachain.online/api/blocks?page=1&limit=10&order=desc"
```

**Response:**
```json
{
  "blocks": [
    {
      "height": 142857,
      "hash": "0x1a2b3c4d...",
      "timestamp": "2024-12-16T10:30:00Z",
      "num_txs": 25,
      "proposer": "artha1validator123..."
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 142857,
    "total_pages": 14286,
    "has_next": true,
    "has_prev": false
  }
}
```

### 💸 Transaction Endpoints

#### Get Transaction by Hash
Retrieve detailed transaction information.

**Endpoint:** `GET /api/transactions/{hash}`

```bash
curl https://testnet.arthachain.online/api/transactions/0xabc123def456...
```

**Response:**
```json
{
  "hash": "0xabc123def456...",
  "block_height": 142857,
  "block_hash": "0x1a2b3c4d...",
  "transaction_index": 5,
  "from": "artha1sender123...",
  "to": "artha1recipient456...",
  "amount": "1000000000000000000",
  "gas_limit": 21000,
  "gas_used": 21000,
  "gas_price": "1000000000",
  "fee": "21000000000000",
  "nonce": 42,
  "status": "success",
  "timestamp": "2024-12-16T10:30:15Z",
  "memo": "Payment for services",
  "signature": "0x1b4f...",
  "logs": [],
  "events": [
    {
      "event": "Transfer",
      "data": {
        "from": "artha1sender123...",
        "to": "artha1recipient456...",
        "amount": "1000000000000000000"
      }
    }
  ]
}
```

#### Submit Transaction
Submit a signed transaction to the network.

**Endpoint:** `POST /api/transactions`

```bash
curl -X POST https://testnet.arthachain.online/api/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": "base64-encoded-signed-transaction",
    "mode": "broadcast_mode_sync"
  }'
```

**Request Body:**
```json
{
  "transaction": "CogBCiYKHy9hc3RyYWNoYWluLm1zZ...",
  "mode": "broadcast_mode_sync"
}
```

**Response:**
```json
{
  "transaction_hash": "0xabc123def456...",
  "code": 0,
  "data": "",
  "log": "[]",
  "info": "",
  "gas_wanted": "21000",
  "gas_used": "21000",
  "events": [
    {
      "type": "transfer",
      "attributes": [
        {"key": "sender", "value": "artha1sender123..."},
        {"key": "recipient", "value": "artha1recipient456..."},
        {"key": "amount", "value": "1000000000000000000"}
      ]
    }
  ]
}
```

#### Get Recent Transactions
Get recently submitted transactions.

**Endpoint:** `GET /api/explorer/transactions/recent`

```bash
curl "https://testnet.arthachain.online/api/explorer/transactions/recent?limit=10"
```

### 👤 Account Endpoints

#### Get Account Information
Retrieve account details including balance and metadata.

**Endpoint:** `GET /api/accounts/{address}`

```bash
curl https://testnet.arthachain.online/api/accounts/artha1xyz123abc456...
```

**Response:**
```json
{
  "address": "artha1xyz123abc456...",
  "balance": {
    "denom": "artha",
    "amount": "1000000000000000000"
  },
  "nonce": 42,
  "public_key": "arthapub1abc123...",
  "account_type": "base_account",
  "created_at": "2024-01-01T00:00:00Z",
  "last_activity": "2024-12-16T10:30:00Z",
  "transaction_count": 157,
  "contract_address": null,
  "is_validator": false,
  "validator_info": null
}
```

#### Get Account Transactions
Retrieve transaction history for a specific account.

**Endpoint:** `GET /api/accounts/{address}/transactions`

**Query Parameters:**
- `page` (optional): Page number
- `limit` (optional): Items per page  
- `type` (optional): Transaction type filter (`send`, `receive`, `contract`)

```bash
curl "https://testnet.arthachain.online/api/accounts/artha1xyz123.../transactions?page=1&limit=20&type=send"
```

**Response:**
```json
{
  "transactions": [
    {
      "hash": "0xabc123...",
      "type": "send",
      "from": "artha1xyz123...",
      "to": "artha1recipient456...",
      "amount": "1000000000000000000",
      "timestamp": "2024-12-16T10:30:00Z",
      "status": "success",
      "block_height": 142857,
      "gas_used": 21000,
      "fee": "21000000000000"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 157,
    "total_pages": 8
  }
}
```

### 🌐 Network & Status Endpoints

#### Get Node Status
Get current node and network status.

**Endpoint:** `GET /api/status`

```bash
curl https://testnet.arthachain.online/api/status
```

**Response:**
```json
{
  "node_info": {
    "protocol_version": "1.0.0",
    "id": "node-abc123...",
    "moniker": "arthachain-testnet-1",
    "network": "artha-testnet-1",
    "version": "1.0.0",
    "channels": "40202122233038606100",
    "other": {
      "tx_index": "on",
      "rpc_address": "tcp://0.0.0.0:26657"
    }
  },
  "sync_info": {
    "latest_block_hash": "0x1a2b3c4d...",
    "latest_block_height": "142857",
    "latest_block_time": "2024-12-16T10:30:00Z",
    "earliest_block_hash": "0x0000000...",
    "earliest_block_height": "1",
    "earliest_block_time": "2024-01-01T00:00:00Z",
    "catching_up": false
  },
  "validator_info": {
    "address": "arthapub1abc123...",
    "pub_key": {
      "type": "tendermint/PubKeyEd25519",
      "value": "ABC123..."
    },
    "voting_power": "100"
  }
}
```

#### Get Network Peers
Get information about connected peers.

**Endpoint:** `GET /api/network/peers`

```bash
curl https://testnet.arthachain.online/api/network/peers
```

**Response:**
```json
{
  "peers": [
    {
      "node_id": "abc123def456...",
      "address": "192.168.1.100:26656",
      "network": "artha-testnet-1",
      "version": "1.0.0",
      "channels": "40202122233038606100",
      "moniker": "validator-node-1",
      "connected_time": "2024-12-16T08:00:00Z",
      "is_outbound": true,
      "connection_status": {
        "duration": "2h30m",
        "send_monitor": {
          "active": true,
          "start": "2024-12-16T08:00:00Z",
          "bytes": 15672890,
          "samples": 125
        },
        "recv_monitor": {
          "active": true, 
          "start": "2024-12-16T08:00:00Z",
          "bytes": 8934562,
          "samples": 98
        }
      }
    }
  ],
  "total_peers": 25,
  "listening": true,
  "listeners": ["tcp://0.0.0.0:26656"]
}
```

#### Health Check
Simple health check endpoint.

**Endpoint:** `GET /api/health`

```bash
curl https://testnet.arthachain.online/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-16T10:30:00Z",
  "version": "1.0.0",
  "block_height": 142857,
  "sync_status": "synced",
  "peer_count": 25,
  "mempool_size": 156,
  "uptime": "99.97%"
}
```

#### Blockchain Statistics
Get comprehensive blockchain statistics.

**Endpoint:** `GET /api/stats`

```bash
curl https://testnet.arthachain.online/api/stats
```

**Response:**
```json
{
  "network": {
    "chain_id": "artha-testnet-1",
    "block_height": 142857,
    "block_time": 2.3,
    "total_transactions": 1247890,
    "total_accounts": 15678,
    "active_validators": 100,
    "bonded_tokens": "50000000000000000000000000",
    "total_supply": "100000000000000000000000000"
  },
  "performance": {
    "tps_current": 450,
    "tps_peak": 1200,
    "tps_average_24h": 387,
    "avg_block_time": 2.3,
    "avg_tx_fee": "0.001",
    "network_uptime": 99.97
  },
  "consensus": {
    "consensus_type": "SVCP + Quantum SVBFT",
    "view": 12345,
    "round": 1,
    "validators_active": 100,
    "validators_total": 150,
    "voting_power_online": "95.5%"
  }
}
```

### 🛡️ Fraud Detection APIs

#### Get Fraud Dashboard Stats
Get AI-powered fraud detection statistics.

**Endpoint:** `GET /api/fraud/dashboard`

```bash
curl https://testnet.arthachain.online/api/fraud/dashboard
```

**Response:**
```json
{
  "summary": {
    "total_transactions_analyzed": 1247890,
    "suspicious_transactions": 23,
    "fraud_detected": 3,
    "fraud_prevented_value": "15000000000000000000",
    "risk_score_average": 0.12,
    "ai_confidence": 0.97
  },
  "real_time": {
    "transactions_per_second": 450,
    "current_risk_level": "low",
    "active_monitoring": true,
    "neural_network_status": "optimal"
  },
  "last_24h": {
    "transactions_processed": 38765,
    "anomalies_detected": 5,
    "patterns_learned": 127,
    "false_positives": 0
  },
  "last_updated": "2024-12-16T10:30:00Z"
}
```

#### Get Detection History
Get fraud detection event history.

**Endpoint:** `GET /api/fraud/history`

```bash
curl "https://testnet.arthachain.online/api/fraud/history?limit=20"
```

#### Get Transaction Risk Analysis
Get detailed AI risk analysis for a specific transaction.

**Endpoint:** `GET /api/fraud/transaction/{tx_hash}`

```bash
curl https://testnet.arthachain.online/api/fraud/transaction/0xabc123...
```

**Response:**
```json
{
  "transaction_hash": "0xabc123def456...",
  "risk_analysis": {
    "fraud_probability": 0.15,
    "anomaly_score": 0.23,
    "risk_level": "low",
    "confidence": 0.94,
    "is_suspicious": false
  },
  "feature_analysis": {
    "amount_analysis": {
      "value": "1000000000000000000",
      "risk_factor": 0.1,
      "reason": "Normal transaction amount"
    },
    "frequency_analysis": {
      "sender_tx_count_24h": 5,
      "risk_factor": 0.05,
      "reason": "Normal transaction frequency"
    },
    "pattern_analysis": {
      "matches_known_patterns": false,
      "similarity_score": 0.12,
      "risk_factor": 0.08
    },
    "network_analysis": {
      "peer_reputation": 0.87,
      "geographic_risk": 0.02,
      "risk_factor": 0.05
    }
  },
  "recommendations": {
    "action": "allow",
    "monitoring_level": "standard",
    "additional_checks": []
  },
  "ai_model_info": {
    "model_version": "2.1.0",
    "features_count": 15,
    "training_data_size": 10000000,
    "last_updated": "2024-12-10T00:00:00Z"
  }
}
```

### 💰 Faucet APIs (Testnet Only)

#### Request Testnet Tokens
Programmatically request tokens from the testnet faucet.

**Endpoint:** `POST /api/faucet/request`

```bash
curl -X POST https://faucet.arthachain.online/api/faucet/request \
  -H "Content-Type: application/json" \
  -d '{
    "address": "artha1xyz123abc456...",
    "amount": "1000000000000000000"
  }'
```

**Request Body:**
```json
{
  "address": "artha1xyz123abc456...",
  "amount": "1000000000000000000",
  "captcha_token": "optional-captcha-token"
}
```

**Response:**
```json
{
  "success": true,
  "transaction_hash": "0xdef456abc123...",
  "amount_sent": "1000000000000000000",
  "recipient": "artha1xyz123abc456...",
  "wait_time": 86400,
  "message": "Tokens sent successfully"
}
```

#### Get Faucet Status
Check faucet availability and limits.

**Endpoint:** `GET /api/faucet/status`

```bash
curl https://faucet.arthachain.online/api/faucet/status
```

**Response:**
```json
{
  "available": true,
  "balance": "50000000000000000000000",
  "daily_limit": "1000000000000000000",
  "requests_today": 156,
  "max_requests_per_day": 1000,
  "cooldown_period": 86400,
  "supported_networks": ["artha-testnet-1"]
}
```

## ⚡ WebSocket API

### 🔌 Real-time Event Streaming

Connect to WebSocket for real-time blockchain updates.

**Endpoint:** `GET /api/ws`

```javascript
// JavaScript WebSocket example
const ws = new WebSocket('wss://testnet.arthachain.online/api/ws');

ws.onopen = () => {
  console.log('Connected to ArthaChain WebSocket');
  
  // Subscribe to new blocks
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'blocks'
  }));
  
  // Subscribe to new transactions
  ws.send(JSON.stringify({
    type: 'subscribe', 
    channel: 'transactions'
  }));
  
  // Subscribe to specific account
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'accounts',
    address: 'artha1xyz123abc456...'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.channel) {
    case 'blocks':
      console.log('New block:', data.data.height);
      break;
    case 'transactions':
      console.log('New transaction:', data.data.hash);
      break;
    case 'accounts':
      console.log('Account update:', data.data);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket connection closed');
};
```

### 📡 Available Channels

#### New Blocks
Receive notifications when new blocks are added.

**Subscribe:**
```json
{
  "type": "subscribe",
  "channel": "blocks"
}
```

**Event:**
```json
{
  "channel": "blocks",
  "type": "new_block",
  "data": {
    "height": 142858,
    "hash": "0x1a2b3c4d...",
    "timestamp": "2024-12-16T10:32:00Z",
    "num_txs": 30,
    "proposer": "artha1validator123..."
  }
}
```

#### New Transactions
Receive notifications for new transactions.

**Subscribe:**
```json
{
  "type": "subscribe",
  "channel": "transactions"
}
```

**Event:**
```json
{
  "channel": "transactions",
  "type": "new_transaction",
  "data": {
    "hash": "0xabc123...",
    "from": "artha1sender123...",
    "to": "artha1recipient456...",
    "amount": "1000000000000000000",
    "status": "pending"
  }
}
```

#### Account Updates
Monitor specific accounts for balance changes.

**Subscribe:**
```json
{
  "type": "subscribe",
  "channel": "accounts",
  "address": "artha1xyz123abc456..."
}
```

**Event:**
```json
{
  "channel": "accounts",
  "type": "balance_change",
  "data": {
    "address": "artha1xyz123abc456...",
    "old_balance": "1000000000000000000",
    "new_balance": "999000000000000000",
    "transaction_hash": "0xdef456..."
  }
}
```

#### Consensus Updates
Monitor consensus state changes.

**Subscribe:**
```json
{
  "type": "subscribe",
  "channel": "consensus"
}
```

**Event:**
```json
{
  "channel": "consensus",
  "type": "view_change",
  "data": {
    "old_view": 12345,
    "new_view": 12346,
    "new_leader": "artha1validator456...",
    "reason": "leader_timeout"
  }
}
```

## 🔗 JSON-RPC API (Ethereum Compatible)

ArthaChain provides full Ethereum JSON-RPC compatibility for easy migration.

**Endpoint:** `POST /rpc`

### 📋 Chain Information

#### eth_chainId
Get the chain ID.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_chainId",
    "params": [],
    "id": 1
  }'
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0x539"
}
```

#### net_version
Get the network version.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "net_version",
    "params": [],
    "id": 1
  }'
```

### 👤 Account Operations

#### eth_accounts
Get list of available accounts.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_accounts",
    "params": [],
    "id": 1
  }'
```

#### eth_getBalance
Get account balance.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_getBalance",
    "params": ["0x742d35Cc66C3D18b1d53C03ec4AC9d52B4D8ae4e", "latest"],
    "id": 1
  }'
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0xde0b6b3a7640000"
}
```

### 🧱 Block Operations

#### eth_blockNumber
Get the latest block number.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_blockNumber",
    "params": [],
    "id": 1
  }'
```

#### eth_getBlockByNumber
Get block by number.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_getBlockByNumber",
    "params": ["latest", true],
    "id": 1
  }'
```

### 💸 Transaction Operations

#### eth_sendTransaction
Send a transaction.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_sendTransaction",
    "params": [{
      "from": "0x742d35Cc66C3D18b1d53C03ec4AC9d52B4D8ae4e",
      "to": "0x8ba1f109551bd432803012645hac136c5e1234",
      "value": "0xde0b6b3a7640000",
      "gas": "0x5208",
      "gasPrice": "0x3b9aca00"
    }],
    "id": 1
  }'
```

#### eth_estimateGas
Estimate gas for a transaction.

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_estimateGas",
    "params": [{
      "from": "0x742d35Cc66C3D18b1d53C03ec4AC9d52B4D8ae4e",
      "to": "0x8ba1f109551bd432803012645hac136c5e1234",
      "value": "0xde0b6b3a7640000"
    }],
    "id": 1
  }'
```

### 🤖 Smart Contract Operations

#### eth_call
Call a contract function (read-only).

```bash
curl -X POST https://testnet.arthachain.online/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_call",
    "params": [{
      "to": "0x742d35Cc66C3D18b1d53C03ec4AC9d52B4D8ae4e",
      "data": "0x70a08231000000000000000000000000742d35Cc66C3D18b1d53C03ec4AC9d52B4D8ae4e"
    }, "latest"],
    "id": 1
  }'
```

## 📚 SDK Examples

### 🟢 JavaScript/TypeScript SDK

```bash
npm install @arthachain/sdk
```

```javascript
import { ArthaChain } from '@arthachain/sdk';

// Initialize client
const client = new ArthaChain({
  rpcUrl: 'https://testnet.arthachain.online/rpc',
  restUrl: 'https://testnet.arthachain.online/api',
  chainId: 'artha-testnet-1'
});

// Get account balance
const balance = await client.getBalance('artha1xyz123abc456...');
console.log('Balance:', balance);

// Send transaction
const tx = await client.sendTokens(
  'artha1sender123...',
  'artha1recipient456...',
  [{ denom: 'artha', amount: '1000000' }],
  {
    gas: '21000',
    gasPrice: '1000000000'
  }
);
console.log('Transaction hash:', tx.transactionHash);

// Call smart contract
const result = await client.queryContract(
  'artha1contract123...',
  { get_value: {} }
);
console.log('Contract result:', result);

// Listen to real-time events
client.subscribeToBlocks((block) => {
  console.log('New block:', block.height);
});
```

### 🐍 Python SDK

```bash
pip install arthachain-py
```

```python
from arthachain import ArthaChainClient

# Initialize client
client = ArthaChainClient(
    rpc_url='https://testnet.arthachain.online/rpc',
    rest_url='https://testnet.arthachain.online/api',
    chain_id='artha-testnet-1'
)

# Get account balance
balance = client.get_balance('artha1xyz123abc456...')
print(f'Balance: {balance}')

# Send transaction
tx = client.send_tokens(
    from_address='artha1sender123...',
    to_address='artha1recipient456...',
    amount='1000000',
    denom='artha'
)
print(f'Transaction hash: {tx.hash}')

# Query smart contract
result = client.query_contract(
    'artha1contract123...',
    {'get_value': {}}
)
print(f'Contract result: {result}')

# WebSocket events
async def on_new_block(block):
    print(f'New block: {block.height}')

client.subscribe_to_blocks(on_new_block)
```

### 🦀 Rust SDK

```bash
cargo add arthachain-rs
```

```rust
use arthachain_rs::{ArthaChainClient, Config};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client
    let config = Config {
        rpc_url: "https://testnet.arthachain.online/rpc".to_string(),
        rest_url: "https://testnet.arthachain.online/api".to_string(),
        chain_id: "artha-testnet-1".to_string(),
    };
    
    let client = ArthaChainClient::new(config).await?;
    
    // Get account balance
    let balance = client.get_balance("artha1xyz123abc456...").await?;
    println!("Balance: {}", balance);
    
    // Send transaction
    let tx = client.send_tokens(
        "artha1sender123...",
        "artha1recipient456...",
        1_000_000u64,
        Some(21000),
        Some(1_000_000_000u64)
    ).await?;
    
    println!("Transaction hash: {}", tx.hash);
    
    Ok(())
}
```

### 🐹 Go SDK

```bash
go get github.com/arthachain/go-sdk
```

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/arthachain/go-sdk"
)

func main() {
    // Initialize client
    client, err := arthachain.NewClient(&arthachain.Config{
        RPCUrl:  "https://testnet.arthachain.online/rpc",
        RestURL: "https://testnet.arthachain.online/api",
        ChainID: "artha-testnet-1",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Get account balance
    balance, err := client.GetBalance("artha1xyz123abc456...")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Balance: %s\n", balance)
    
    // Send transaction
    tx, err := client.SendTokens(
        "artha1sender123...",
        "artha1recipient456...",
        1000000,
        arthachain.TxOptions{
            Gas:      21000,
            GasPrice: 1000000000,
        },
    )
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Transaction hash: %s\n", tx.Hash)
}
```

## ⚠️ Error Handling

### 🚨 HTTP Status Codes
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid API key)
- `404` - Not Found (resource doesn't exist)
- `429` - Rate Limited (too many requests)
- `500` - Internal Server Error
- `503` - Service Unavailable (maintenance)

### 📋 Error Response Format
```json
{
  "error": {
    "code": 400,
    "message": "Invalid transaction format",
    "details": "Transaction signature verification failed",
    "timestamp": "2024-12-16T10:30:00Z",
    "request_id": "req_abc123def456"
  }
}
```

### 🔢 Common Error Codes

| Code | Message | Description | Solution |
|------|---------|-------------|----------|
| `1001` | Invalid address format | Address doesn't match expected format | Use correct address format (`artha1...`) |
| `1002` | Transaction not found | Transaction hash not found | Check transaction hash is correct |
| `1003` | Insufficient balance | Account balance too low | Add funds to account |
| `1004` | Invalid signature | Transaction signature invalid | Re-sign transaction with correct key |
| `1005` | Rate limit exceeded | Too many requests | Wait and retry, or upgrade plan |
| `1006` | Network congestion | Network is busy | Increase gas price or retry later |
| `1007` | Contract execution failed | Smart contract error | Check contract code and parameters |
| `1008` | Invalid gas amount | Gas limit too low or high | Use `eth_estimateGas` for correct amount |

### 🔄 Retry Logic

```javascript
// JavaScript retry example
async function callAPIWithRetry(apiCall, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await apiCall();
    } catch (error) {
      if (error.status === 429) {
        // Rate limited - wait before retry
        const waitTime = Math.pow(2, i) * 1000; // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }
      
      if (error.status >= 500 && i < maxRetries - 1) {
        // Server error - retry
        await new Promise(resolve => setTimeout(resolve, 1000));
        continue;
      }
      
      throw error; // Don't retry client errors
    }
  }
}
```

## 🎯 Best Practices

### ⚡ Performance Tips

1. **Use WebSockets for real-time data** instead of polling
2. **Batch requests** when possible
3. **Cache responses** appropriately
4. **Use pagination** for large datasets
5. **Monitor rate limits** and implement backoff

### 🔐 Security Best Practices

1. **Never expose private keys** in client-side code
2. **Validate all inputs** before sending to API
3. **Use HTTPS only** for all API calls
4. **Implement proper error handling**
5. **Monitor for unusual activity**

### 💰 Cost Optimization

1. **Use read-only calls** when possible (no gas cost)
2. **Estimate gas** before transactions
3. **Batch similar operations**
4. **Use appropriate gas prices**
5. **Cache frequently accessed data**

## 🆘 Support & Resources

### 📚 **Additional Documentation**
- **[🎮 Interactive API Explorer](https://api-docs.arthachain.online)** - Test APIs in your browser
- **[🔧 Postman Collection](https://postman.arthachain.online)** - Ready-to-use API collection
- **[📖 OpenAPI Specification](https://spec.arthachain.online)** - Complete API specification
- **[💡 Code Examples](https://github.com/arthachain/api-examples)** - More code examples

### 💬 **Community Support**
- **[💬 Discord](https://discord.gg/arthachain)** - Live chat with developers
- **[📱 Telegram](https://t.me/arthachain_dev)** - Developer support group
- **[🐙 GitHub](https://github.com/arthachain/blockchain)** - Report issues and contribute

### 📧 **Direct Support**
- **👨‍💻 API Questions**: [api@arthachain.com](mailto:api@arthachain.com)
- **🔐 Security Issues**: [security@arthachain.com](mailto:security@arthachain.com)
- **🏢 Enterprise Support**: [enterprise@arthachain.com](mailto:enterprise@arthachain.com)

---

**🎯 Next**: [🤖 Smart Contracts](./smart-contracts.md) →

**💬 Questions?** Join our [Discord](https://discord.gg/arthachain) - our community loves helping developers! 