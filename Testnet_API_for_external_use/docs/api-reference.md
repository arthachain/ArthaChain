# API Reference

This document provides a comprehensive reference for all available API endpoints in the Artha Chain testnet.

## Base URL

All API endpoints are relative to the base URL of a validator node:
- `http://localhost:3000` (validator1)
- `http://localhost:3001` (validator2)
- `http://localhost:3002` (validator3)
- `http://localhost:3003` (validator4)

## Blocks

### Get Latest Block

Retrieves the most recent block on the blockchain.

**Endpoint:** `GET /api/blocks/latest`

**Response:**
```json
{
  "hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "height": 1024,
  "timestamp": 1650326472,
  "previous_hash": "0xe0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93a2",
  "merkle_root": "0xd5e041084eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9",
  "transactions_count": 12,
  "validator": "validator1",
  "signature": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5",
  "size": 5612
}
```

### Get Block by Hash

Retrieves a block with the specified hash.

**Endpoint:** `GET /api/blocks/:hash`

**Parameters:**
- `hash` (path parameter): The hash of the block to retrieve

**Response:** Same as "Get Latest Block"

### Get Block by Height

Retrieves a block at the specified height.

**Endpoint:** `GET /api/blocks/height/:height`

**Parameters:**
- `height` (path parameter): The height/number of the block to retrieve

**Response:** Same as "Get Latest Block"

## Transactions

### Get Transaction

Retrieves a transaction with the specified hash.

**Endpoint:** `GET /api/transactions/:hash`

**Parameters:**
- `hash` (path parameter): The hash of the transaction to retrieve

**Response:**
```json
{
  "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "sender": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "recipient": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "amount": 100,
  "fee": 1,
  "timestamp": 1650326400,
  "data": "0x",
  "type": "transfer",
  "status": "confirmed",
  "block_hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "block_height": 1024
}
```

### Submit Transaction

Submits a new transaction to the network.

**Endpoint:** `POST /api/transactions`

**Request:**
```json
{
  "sender": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "recipient": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "amount": 100,
  "data": "0x",
  "type": "transfer",
  "signature": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5"
}
```

**Response:**
```json
{
  "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "status": "pending"
}
```

## Accounts

### Get Account

Retrieves account information for the specified address.

**Endpoint:** `GET /api/accounts/:address`

**Parameters:**
- `address` (path parameter): The address of the account to retrieve

**Response:**
```json
{
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "balance": 10000,
  "nonce": 5,
  "storage_hash": "0x5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9"
}
```

### Get Account Transactions

Retrieves transactions for the specified account.

**Endpoint:** `GET /api/accounts/:address/transactions`

**Parameters:**
- `address` (path parameter): The address of the account
- `limit` (query parameter, optional): Maximum number of transactions to return (default: 10)
- `offset` (query parameter, optional): Number of transactions to skip (default: 0)

**Response:**
```json
{
  "transactions": [
    {
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "sender": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "recipient": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "amount": 100,
      "fee": 1,
      "timestamp": 1650326400,
      "type": "transfer",
      "status": "confirmed",
      "block_height": 1024
    },
    // ...more transactions
  ],
  "total": 42
}
```

## Network Status

### Get Status

Retrieves the current status of the blockchain network.

**Endpoint:** `GET /api/status`

**Response:**
```json
{
  "network_id": "artha-testnet",
  "version": "0.5.2",
  "latest_block_height": 1024,
  "latest_block_hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "latest_block_time": 1650326472,
  "syncing": false,
  "peers_count": 3,
  "tps": 15.7,
  "pending_transactions": 2,
  "uptime": 86400
}
```

### Get Peers

Retrieves the list of connected peer nodes.

**Endpoint:** `GET /api/network/peers`

**Response:**
```json
{
  "peers": [
    {
      "id": "validator2",
      "address": "http://localhost:7001",
      "connected_since": 1650240000
    },
    {
      "id": "validator3",
      "address": "http://localhost:7002",
      "connected_since": 1650240100
    },
    {
      "id": "validator4",
      "address": "http://localhost:7003",
      "connected_since": 1650240200
    }
  ]
}
```

## Metrics

### Get Metrics

Retrieves various performance metrics for the node.

**Endpoint:** `GET /api/metrics`

**Response:**
```json
{
  "blocks_produced": 256,
  "blocks_finalized": 1024,
  "transactions_processed": 4096,
  "average_block_time": 15.2,
  "average_tps": 16.8,
  "peer_count": 3,
  "memory_usage": "512MB",
  "disk_usage": "1.2GB",
  "cpu_usage": 45.2
}
```

### Get TPS

Retrieves the current transactions per second rate.

**Endpoint:** `GET /api/metrics/tps`

**Response:**
```json
{
  "current_tps": 15.7,
  "peak_tps": 65.3,
  "average_tps_1m": 12.5,
  "average_tps_10m": 14.8,
  "average_tps_1h": 16.2
}
``` 