# Data Models

This document outlines the data structures returned by the Artha Chain testnet API.

## Block

```json
{
  "header": {
    "height": 42,
    "timestamp": 1652345678,
    "hash": "0x7f2c45fd8c34a1d78e4aa123f517e4d6ae31d1dc812a4b3aef15ca2bc97da153",
    "previous_hash": "0x3d6a45cb7c17f3812a36fc1524ca2890a5f5d8a7c1d6e71238afc245c7b92a1f",
    "merkle_root": "0xf9ce37120b15928c5bd9f95c5f0abaf9b6bc1f353d4f32f01c765cb932a0873a",
    "validator": "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654",
    "signature": "0xa4c92b35c2148dc7fa9cc8f3c489c72213f2ca4b15..."
  },
  "transactions": [
    {
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "sender": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "recipient": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27",
      "amount": 10,
      "gas_price": 1,
      "gas_limit": 21000,
      "timestamp": 1652345600,
      "data": "0x",
      "signature": "0x7c2a65fc3d8a15ef7a49d..."
    }
    // Additional transactions...
  ],
  "size": 1458,
  "transaction_count": 5
}
```

### Block Fields

| Field | Type | Description |
|-------|------|-------------|
| header | Object | Block header information |
| header.height | Integer | Block height/number in the chain |
| header.timestamp | Integer | Unix timestamp when the block was created |
| header.hash | String | Hash of the block |
| header.previous_hash | String | Hash of the previous block |
| header.merkle_root | String | Merkle root of all transactions |
| header.validator | String | Address of the validator who proposed the block |
| header.signature | String | Validator's signature |
| transactions | Array | List of transactions included in the block |
| size | Integer | Size of the block in bytes |
| transaction_count | Integer | Number of transactions in the block |

## Transaction

```json
{
  "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "sender": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "recipient": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27",
  "amount": 10,
  "gas_price": 1,
  "gas_limit": 21000,
  "gas_used": 21000,
  "timestamp": 1652345600,
  "data": "0x",
  "signature": "0x7c2a65fc3d8a15ef7a49d...",
  "block_hash": "0x7f2c45fd8c34a1d78e4aa123f517e4d6ae31d1dc812a4b3aef15ca2bc97da153",
  "block_height": 42,
  "status": "confirmed",
  "nonce": 7
}
```

### Transaction Fields

| Field | Type | Description |
|-------|------|-------------|
| hash | String | Transaction hash |
| sender | String | Address of the transaction sender |
| recipient | String | Address of the transaction recipient |
| amount | Integer | Amount of tokens transferred |
| gas_price | Integer | Price per unit of gas (in smallest token denomination) |
| gas_limit | Integer | Maximum gas units the transaction can consume |
| gas_used | Integer | Actual gas units consumed (only present for confirmed transactions) |
| timestamp | Integer | Unix timestamp when the transaction was created |
| data | String | Additional data included in the transaction (hex-encoded) |
| signature | String | Digital signature of the transaction |
| block_hash | String | Hash of the block containing the transaction (null if pending) |
| block_height | Integer | Height of the block containing the transaction (null if pending) |
| status | String | Transaction status: "pending", "confirmed", or "failed" |
| nonce | Integer | Sender account nonce for this transaction |

## Account

```json
{
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "balance": 9950,
  "nonce": 8,
  "type": "user",
  "storage_used": 0,
  "code_hash": null,
  "last_active_height": 42
}
```

### Account Fields

| Field | Type | Description |
|-------|------|-------------|
| address | String | Account address |
| balance | Integer | Current account balance |
| nonce | Integer | Current account nonce (incremented with each sent transaction) |
| type | String | Account type: "user", "contract", or "validator" |
| storage_used | Integer | Amount of storage used by the account (relevant for contracts) |
| code_hash | String | Hash of contract code (null for non-contract accounts) |
| last_active_height | Integer | Block height when the account was last active |

## NetworkStatus

```json
{
  "node_version": "v0.5.2",
  "network_id": "artha-testnet",
  "current_height": 42,
  "current_hash": "0x7f2c45fd8c34a1d78e4aa123f517e4d6ae31d1dc812a4b3aef15ca2bc97da153",
  "sync_status": "synced",
  "peers": 4,
  "pending_transactions": 2,
  "tps": 8.5,
  "uptime": 3602
}
```

### NetworkStatus Fields

| Field | Type | Description |
|-------|------|-------------|
| node_version | String | Version of the blockchain node software |
| network_id | String | Identifier of the network |
| current_height | Integer | Current block height |
| current_hash | String | Hash of the current block |
| sync_status | String | Node synchronization status: "syncing", "synced", or "stalled" |
| peers | Integer | Number of connected peer nodes |
| pending_transactions | Integer | Number of transactions waiting to be included in a block |
| tps | Float | Current transactions per second rate (average over last minute) |
| uptime | Integer | Node uptime in seconds |

## Peers

```json
{
  "peers": [
    {
      "id": "QmP2p1oLRQbpWzhS7oLfvrnLzbw65YyMwMnDpNnrY9ksQ5",
      "address": "172.17.0.3:30303",
      "connected_since": 1652345000,
      "version": "v0.5.2",
      "is_validator": true
    },
    {
      "id": "QmYZ3P7zxQMfLcMEoy5DvfEYxRixaJ3L7YyL8C5iX1Nz8v",
      "address": "172.17.0.4:30303",
      "connected_since": 1652345120,
      "version": "v0.5.2",
      "is_validator": true
    }
    // Additional peers...
  ],
  "total": 4
}
```

### Peers Fields

| Field | Type | Description |
|-------|------|-------------|
| peers | Array | List of connected peer nodes |
| peers[].id | String | Unique identifier of the peer |
| peers[].address | String | IP address and port of the peer |
| peers[].connected_since | Integer | Unix timestamp when the peer connected |
| peers[].version | String | Version of the peer's node software |
| peers[].is_validator | Boolean | Whether the peer is a validator |
| total | Integer | Total number of connected peers |

## Metrics

```json
{
  "tps": {
    "current": 8.5,
    "average_1m": 7.2,
    "average_10m": 6.8,
    "average_1h": 5.4,
    "peak_24h": 15.3
  },
  "block_time": {
    "current": 5.2,
    "average_1h": 5.0,
    "average_24h": 5.1
  },
  "memory_usage": 128456789,
  "cpu_usage": 12.5,
  "disk_usage": 2345678901,
  "network_in": 1234567,
  "network_out": 2345678
}
```

### Metrics Fields

| Field | Type | Description |
|-------|------|-------------|
| tps | Object | Transactions per second metrics |
| tps.current | Float | Current TPS rate |
| tps.average_1m | Float | Average TPS over the last minute |
| tps.average_10m | Float | Average TPS over the last 10 minutes |
| tps.average_1h | Float | Average TPS over the last hour |
| tps.peak_24h | Float | Peak TPS in the last 24 hours |
| block_time | Object | Block time metrics in seconds |
| block_time.current | Float | Time between the last two blocks |
| block_time.average_1h | Float | Average block time over the last hour |
| block_time.average_24h | Float | Average block time over the last 24 hours |
| memory_usage | Integer | Current memory usage in bytes |
| cpu_usage | Float | Current CPU usage percentage |
| disk_usage | Integer | Current disk usage in bytes |
| network_in | Integer | Inbound network traffic in bytes |
| network_out | Integer | Outbound network traffic in bytes | 