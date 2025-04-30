# Node Portal API

The Artha Chain Testnet Node Portal API provides administrative access to blockchain node operations, configuration, and monitoring. This API is designed for node operators, validators, and network administrators to manage their participation in the network.

## Base URL

All API endpoints are relative to the base admin URL of a validator node:
- `http://localhost:3010/admin` (validator1)
- `http://localhost:3011/admin` (validator2)
- `http://localhost:3012/admin` (validator3)
- `http://localhost:3013/admin` (validator4)

## Authentication

**All endpoints require authentication.**

Authentication is performed using JWT tokens. To obtain a token, use the login endpoint:

**Endpoint:** `POST /admin/auth/login`

**Request:**
```json
{
  "username": "admin",
  "password": "your_secure_password"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_at": "2023-06-05T22:45:22Z"
}
```

For all subsequent requests, include the token in the Authorization header:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Node Management

### Get Node Status

Retrieves the current status of the node.

**Endpoint:** `GET /admin/node/status`

**Response:**
```json
{
  "node_id": "node1",
  "version": "1.0.0",
  "uptime": 1234567,
  "current_height": 10245,
  "synced": true,
  "syncing_progress": 100,
  "peers_connected": 12,
  "validator_status": "active",
  "last_proposed_block": 10240,
  "resources": {
    "cpu_usage": 45.2,
    "memory_usage": 2048,
    "disk_usage": 10240,
    "bandwidth_usage": {
      "in": 1024,
      "out": 2048
    }
  }
}
```

### Restart Node

Initiates a controlled restart of the node.

**Endpoint:** `POST /admin/node/restart`

**Request:**
```json
{
  "reason": "software update",
  "immediate": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Node restart scheduled after the current block",
  "estimated_restart_time": "2023-06-05T17:15:22Z"
}
```

### Update Node Software

Initiates a software update for the node.

**Endpoint:** `POST /admin/node/update`

**Request:**
```json
{
  "version": "1.1.0",
  "restart_after_update": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Software update initiated",
  "update_status": "downloading",
  "estimated_completion_time": "2023-06-05T17:30:22Z"
}
```

### Get Node Logs

Retrieves node logs for a specified time period and log level.

**Endpoint:** `GET /admin/node/logs`

**Query Parameters:**
- `level` (optional): Log level filter (default: "info", options: "debug", "info", "warn", "error")
- `start_time` (optional): Start time for log retrieval (ISO 8601 format)
- `end_time` (optional): End time for log retrieval (ISO 8601 format)
- `limit` (optional): Maximum number of log entries to return (default: 100, max: 1000)
- `component` (optional): Filter logs by component (e.g., "consensus", "p2p", "mempool")

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2023-06-05T16:45:22Z",
      "level": "info",
      "component": "consensus",
      "message": "Proposing block at height 10245",
      "context": {
        "block_hash": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6f2c3b73152abc45d76b23c9",
        "transactions": 42
      }
    },
    {
      "timestamp": "2023-06-05T16:45:12Z",
      "level": "debug",
      "component": "mempool",
      "message": "Added transaction to mempool",
      "context": {
        "tx_hash": "0x7c17f3812a36fc1524ca2890a5f5d8a7c1d6e71238afc245c7b92a1f3d6a45cb"
      }
    }
  ],
  "pagination": {
    "total": 245,
    "returned": 100,
    "has_more": true
  }
}
```

### Export Node Data

Exports node data for backup or analysis.

**Endpoint:** `POST /admin/node/export`

**Request:**
```json
{
  "data_type": "state",
  "format": "json",
  "start_height": 10000,
  "end_height": 10200
}
```

**Response:**
```json
{
  "success": true,
  "message": "Export initiated",
  "export_id": "exp_1234567890",
  "estimated_size": "25MB",
  "estimated_completion_time": "2023-06-05T17:45:22Z",
  "download_url": "/admin/node/export/exp_1234567890"
}
```

## Validator Operations

### Get Validator Status

Retrieves detailed information about the validator operation.

**Endpoint:** `GET /admin/validator/status`

**Response:**
```json
{
  "address": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
  "public_key": "0x048a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6f2c3b73152abc45d76b23c9...",
  "status": "active",
  "total_staked": "5000.0",
  "self_stake": "5000.0",
  "delegated_stake": "7500.0",
  "voting_power": 12500,
  "voting_power_percentage": 2.5,
  "commission_rate": 10.0,
  "uptime": 99.98,
  "missed_blocks": 2,
  "last_signed_block": 10245,
  "consensus_pubkey": "0x9b23e34f712a9088dc34a991b60ed752f29c5a7d...",
  "rank": 5,
  "performance": {
    "blocks_proposed_24h": 120,
    "blocks_signed_24h": 2880,
    "average_block_time": 3.2,
    "rewards_24h": "25.5"
  }
}
```

### Update Validator Configuration

Updates the validator's configuration.

**Endpoint:** `PUT /admin/validator/config`

**Request:**
```json
{
  "commission_rate": 8.5,
  "min_self_delegation": "5000.0",
  "max_total_delegation": "100000.0",
  "metadata": {
    "name": "Validator One",
    "website": "https://validator1.example.com",
    "description": "Reliable validator service with 99.99% uptime",
    "icon_url": "https://validator1.example.com/icon.png"
  },
  "contact": {
    "email": "admin@validator1.example.com",
    "discord": "validator1#1234",
    "twitter": "@validator1"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Validator configuration updated",
  "effective_from_block": 10300,
  "transaction_hash": "0x8d28e44f823a9c5fa4b8f12e72d6adcb325c4bc98975ae53245b563f0425c17a"
}
```

### Register/Unregister Validator

Registers or unregisters the node as a validator.

**Endpoint:** `POST /admin/validator/register`

**Request:**
```json
{
  "stake_amount": "5000.0",
  "commission_rate": 10.0,
  "min_self_delegation": "5000.0",
  "max_total_delegation": "100000.0",
  "metadata": {
    "name": "Validator One",
    "website": "https://validator1.example.com",
    "description": "Reliable validator service with 99.99% uptime",
    "icon_url": "https://validator1.example.com/icon.png"
  },
  "contact": {
    "email": "admin@validator1.example.com",
    "discord": "validator1#1234",
    "twitter": "@validator1"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Validator registration submitted",
  "transaction_hash": "0x9f34e78a2b5c6d8e9f12a3c4b5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4",
  "effective_from_block": 10300
}
```

To unregister:

**Endpoint:** `POST /admin/validator/unregister`

**Request:**
```json
{
  "reason": "Maintenance"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Validator unregistration submitted",
  "transaction_hash": "0xa1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
  "effective_from_block": 10300,
  "unbonding_period_ends": "2023-06-12T00:00:00Z"
}
```

### Get Validator Rewards

Retrieves detailed information about validator rewards.

**Endpoint:** `GET /admin/validator/rewards`

**Query Parameters:**
- `period` (optional): Time period for rewards (default: "24h", options: "24h", "7d", "30d", "all")

**Response:**
```json
{
  "address": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
  "total_rewards": "250.5",
  "pending_rewards": "15.25",
  "claimed_rewards": "235.25",
  "last_claim": "2023-06-01T10:30:00Z",
  "rewards_by_period": {
    "24h": "25.5",
    "7d": "175.2",
    "30d": "250.5"
  },
  "rewards_by_type": {
    "block_proposal": "125.25",
    "block_attestation": "100.2",
    "transaction_fees": "25.05"
  }
}
```

### Claim Validator Rewards

Claims pending validator rewards.

**Endpoint:** `POST /admin/validator/claim-rewards`

**Request:**
```json
{
  "destination_address": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Rewards claim submitted",
  "transaction_hash": "0xb2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
  "claimed_amount": "15.25"
}
```

## Network Management

### Get Network Status

Retrieves the status of the entire network.

**Endpoint:** `GET /admin/network/status`

**Response:**
```json
{
  "network_id": "artha-testnet-1",
  "current_height": 10245,
  "active_validators": 50,
  "total_validators": 55,
  "total_nodes": 120,
  "block_time": 3.2,
  "transactions_per_second": 42.5,
  "total_transactions": 1245678,
  "network_throughput": {
    "transactions_per_block": 120,
    "gas_used_per_block": 5000000
  },
  "latest_upgrade": {
    "version": "1.0.0",
    "height": 10000,
    "time": "2023-06-01T00:00:00Z"
  },
  "upcoming_upgrade": null
}
```

### Get Peers

Retrieves information about connected peers.

**Endpoint:** `GET /admin/network/peers`

**Response:**
```json
{
  "total_peers": 12,
  "connected_peers": [
    {
      "id": "peer1",
      "address": "123.456.789.012:26656",
      "moniker": "validator2",
      "connection_time": "2023-06-01T12:34:56Z",
      "is_validator": true,
      "direction": "outbound",
      "bandwidth": {
        "send_rate": 512,
        "recv_rate": 1024
      }
    },
    {
      "id": "peer2",
      "address": "234.567.890.123:26656",
      "moniker": "validator3",
      "connection_time": "2023-06-01T12:35:12Z",
      "is_validator": true,
      "direction": "inbound",
      "bandwidth": {
        "send_rate": 428,
        "recv_rate": 856
      }
    }
  ],
  "persistent_peers": [
    "peer1@123.456.789.012:26656",
    "peer2@234.567.890.123:26656"
  ]
}
```

### Add/Remove Peer

Adds or removes a peer node.

**Endpoint:** `POST /admin/network/peers`

**Request:**
```json
{
  "action": "add",
  "peer_id": "peer3",
  "address": "345.678.901.234:26656",
  "persistent": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Peer added successfully",
  "peer_info": {
    "id": "peer3",
    "address": "345.678.901.234:26656",
    "persistent": true,
    "status": "connecting"
  }
}
```

To remove:

**Endpoint:** `DELETE /admin/network/peers/:peer_id`

**Response:**
```json
{
  "success": true,
  "message": "Peer removed successfully"
}
```

## Configuration Management

### Get Node Configuration

Retrieves the current node configuration.

**Endpoint:** `GET /admin/config`

**Response:**
```json
{
  "node": {
    "id": "node1",
    "moniker": "validator1",
    "version": "1.0.0"
  },
  "p2p": {
    "laddr": "tcp://0.0.0.0:26656",
    "external_address": "123.456.789.012:26656",
    "seeds": ["seed1@seed1.example.com:26656", "seed2@seed2.example.com:26656"],
    "persistent_peers": ["peer1@123.456.789.012:26656", "peer2@234.567.890.123:26656"],
    "max_num_inbound_peers": 40,
    "max_num_outbound_peers": 10
  },
  "rpc": {
    "laddr": "tcp://0.0.0.0:26657",
    "cors_allowed_origins": ["*"],
    "max_subscription_clients": 100
  },
  "consensus": {
    "create_empty_blocks": true,
    "create_empty_blocks_interval": "0s",
    "timeout_propose": "3s",
    "timeout_prevote": "1s",
    "timeout_precommit": "1s",
    "timeout_commit": "5s"
  },
  "mempool": {
    "size": 5000,
    "cache_size": 10000,
    "max_tx_bytes": 1048576
  },
  "admin_api": {
    "enabled": true,
    "laddr": "tcp://0.0.0.0:26660",
    "rate_limit": 100
  }
}
```

### Update Node Configuration

Updates specific node configuration parameters.

**Endpoint:** `PUT /admin/config`

**Request:**
```json
{
  "p2p": {
    "external_address": "123.456.789.012:26656",
    "persistent_peers": ["peer1@123.456.789.012:26656", "peer2@234.567.890.123:26656", "peer3@345.678.901.234:26656"],
    "max_num_inbound_peers": 50,
    "max_num_outbound_peers": 15
  },
  "mempool": {
    "size": 10000,
    "cache_size": 20000
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "requires_restart": true
}
```

### Get Chain Parameters

Retrieves the current on-chain parameters.

**Endpoint:** `GET /admin/chain/params`

**Response:**
```json
{
  "block_params": {
    "max_bytes": 21000000,
    "max_gas": 10000000
  },
  "evidence_params": {
    "max_age_num_blocks": 100000,
    "max_age_duration": "172800000000000",
    "max_bytes": 1048576
  },
  "validator_params": {
    "min_stake": "1000.0",
    "max_validators": 100,
    "unbonding_time": "1209600000000000"
  },
  "transaction_params": {
    "min_gas_price": "0.000000100",
    "max_tx_size": 1048576
  },
  "governance_params": {
    "voting_period": "1209600000000000",
    "quorum": "0.334",
    "threshold": "0.5",
    "veto_threshold": "0.334"
  }
}
```

## KeyManager

### Get Keys

Retrieves information about keys stored in the node's keystore.

**Endpoint:** `GET /admin/keys`

**Response:**
```json
{
  "keys": [
    {
      "name": "validator",
      "type": "local",
      "address": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
      "pubkey": "0x048a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6f2c3b73152abc45d76b23c9...",
      "pubkey_type": "ed25519"
    }
  ]
}
```

### Create Key

Creates a new key in the node's keystore.

**Endpoint:** `POST /admin/keys`

**Request:**
```json
{
  "name": "operator",
  "key_type": "ed25519",
  "mnemonic": "word1 word2 word3 ... word24",
  "passphrase": "secure_passphrase"
}
```

**Response:**
```json
{
  "success": true,
  "key": {
    "name": "operator",
    "type": "local",
    "address": "0x9b23e34f712a9088dc34a991b60ed752f29c5a7d",
    "pubkey": "0x049b23e34f712a9088dc34a991b60ed752f29c5a7d3152abc45d76b23c9f2c3b7...",
    "pubkey_type": "ed25519"
  },
  "mnemonic": "word1 word2 word3 ... word24"
}
```

### Import Key

Imports an existing key into the node's keystore.

**Endpoint:** `POST /admin/keys/import`

**Request:**
```json
{
  "name": "backup_validator",
  "private_key": "0x7c17f3812a36fc1524ca2890a5f5d8a7c1d6e71238afc245c7b92a1f3d6a45cb",
  "key_type": "ed25519",
  "passphrase": "secure_passphrase"
}
```

**Response:**
```json
{
  "success": true,
  "key": {
    "name": "backup_validator",
    "type": "local",
    "address": "0xa1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
    "pubkey": "0x04a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b01d2e3f4a5b6c7d8e9f0...",
    "pubkey_type": "ed25519"
  }
}
```

### Export Key

Exports a key from the node's keystore.

**Endpoint:** `POST /admin/keys/export`

**Request:**
```json
{
  "name": "validator",
  "passphrase": "secure_passphrase"
}
```

**Response:**
```json
{
  "success": true,
  "key": {
    "name": "validator",
    "type": "local",
    "address": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
    "pubkey": "0x048a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6f2c3b73152abc45d76b23c9...",
    "pubkey_type": "ed25519",
    "private_key": "0x7c17f3812a36fc1524ca2890a5f5d8a7c1d6e71238afc245c7b92a1f3d6a45cb"
  }
}
```

### Delete Key

Deletes a key from the node's keystore.

**Endpoint:** `DELETE /admin/keys/:name`

**Response:**
```json
{
  "success": true,
  "message": "Key deleted successfully"
}
```

## Error Responses

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_PARAMETER | Invalid parameter value provided |
| 401 | UNAUTHORIZED | Authentication required or failed |
| 403 | FORBIDDEN | Not authorized to access this resource |
| 404 | RESOURCE_NOT_FOUND | The requested resource does not exist |
| 409 | CONFLICT | Resource conflict (e.g., duplicate key name) |
| 422 | VALIDATION_FAILED | Request validation failed |
| 429 | RATE_LIMITED | Too many requests, exceeded rate limit |
| 500 | SERVER_ERROR | Internal server error |
| 503 | SERVICE_UNAVAILABLE | Service temporarily unavailable |

### Error Response Example:

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Invalid configuration parameters",
    "details": {
      "mempool.size": "Value must be between 1000 and 100000",
      "p2p.max_num_inbound_peers": "Value must be between 10 and 100"
    }
  }
}
```

## Implementation Notes

- All requests to the admin API require authentication via JWT token
- Configuration changes may require node restart to take effect
- Sensitive operations (key management, validator operations) have additional security requirements
- Rate limits apply to prevent excessive API usage
- Some operations may be asynchronous and return a task ID for status tracking
- The admin API is only available on testnet nodes for development and testing purposes 