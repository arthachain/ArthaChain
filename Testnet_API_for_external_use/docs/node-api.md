# Node API

The Node API provides access to blockchain node information, peer connectivity status, and system performance metrics on the Artha Chain Testnet. This API allows developers to monitor node health, view network topology, query resource usage, and access other node-specific information.

## Base URL

Each validator node exposes the Node API at the following base URLs:
- `http://localhost:3000/api/node` (validator1)
- `http://localhost:3001/api/node` (validator2)
- `http://localhost:3002/api/node` (validator3)
- `http://localhost:3003/api/node` (validator4)

## Authentication

Most node endpoints are public and don't require authentication. Admin-level operations require authentication:

```
Authorization: Bearer your_api_key_here
```

API keys can be obtained from the [Artha Chain Testnet Portal](http://localhost:3000/portal).

## Endpoints

### Node Status

#### GET /api/node/status

Retrieves the current status of the blockchain node.

**Example Response:**
```json
{
  "node_id": "QmX5NLqqHwzm6QjZz7pBoeFwxwTbxp8gGTrSFVTHx5xzDz",
  "version": "1.4.0",
  "status": "online",
  "uptime": 86450,
  "uptime_formatted": "24:00:50",
  "peer_count": 15,
  "synced": true,
  "current_block": 12890,
  "highest_block": 12890,
  "started_at": "2023-06-04T19:15:00Z",
  "last_updated": "2023-06-05T19:15:50Z",
  "chain_id": "artha-testnet-1",
  "consensus_role": "validator",
  "disk_usage": {
    "blockchain_db": "1.2 GB",
    "state_db": "3.5 GB",
    "logs": "250 MB",
    "total": "4.95 GB"
  },
  "memory_usage": {
    "current": "1.8 GB",
    "peak": "2.1 GB",
    "percent": 45
  },
  "cpu_usage": {
    "percent": 12,
    "cores": 4
  },
  "network": {
    "in_bandwidth": "1.2 MB/s",
    "out_bandwidth": "0.8 MB/s"
  }
}
```

### Node Information

#### GET /api/node/info

Retrieves detailed information about the node, including its configuration.

**Example Response:**
```json
{
  "node_id": "QmX5NLqqHwzm6QjZz7pBoeFwxwTbxp8gGTrSFVTHx5xzDz",
  "node_name": "Validator 1",
  "version": "1.4.0",
  "platform": {
    "os": "linux",
    "arch": "x86_64",
    "kernel": "5.4.0-91-generic",
    "go_version": "1.18.5"
  },
  "network": {
    "chain_id": "artha-testnet-1",
    "genesis_hash": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8fd23f56b4c0f76c46354a6d",
    "network_id": "testnet",
    "sync_mode": "full",
    "nat": "none"
  },
  "config": {
    "log_level": "info",
    "max_peers": 50,
    "p2p_port": 30303,
    "rpc_enabled": true,
    "rpc_port": 8545,
    "ws_enabled": true,
    "ws_port": 8546,
    "metrics_enabled": true,
    "metrics_port": 9090,
    "data_dir": "/data/blockchain"
  },
  "features": {
    "evm_compatible": true,
    "wasm_compatible": true,
    "bls_signatures": true,
    "parallelized_execution": true,
    "state_pruning": true,
    "storage_compression": true
  },
  "consensus": {
    "type": "bft",
    "role": "validator",
    "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
    "voting_power": 100
  },
  "start_time": "2023-06-04T19:15:00Z",
  "database": {
    "engine": "leveldb",
    "version": "1.22",
    "data_dir_size": "4.95 GB"
  }
}
```

### Node Peers

#### GET /api/node/peers

Retrieves information about the peers connected to the node.

**Parameters:**
- `limit`: Maximum number of peers to return (optional, default: 50, max: 100)
- `offset`: Pagination offset (optional, default: 0)

**Example Response:**
```json
{
  "peer_count": 15,
  "connected_peers": [
    {
      "id": "QmYyQSiNxTtXP5jbjK5dawHzZ2zZtrshMVN23ffgFLtTQz",
      "address": "54.182.5.123:30303",
      "direction": "outbound",
      "name": "Validator 2",
      "protocols": ["eth/66", "snap/1", "bsc/1"],
      "version": "1.4.0",
      "connected_at": "2023-06-04T19:16:25Z",
      "last_message": "2023-06-05T19:14:55Z",
      "latency": 125,
      "traffic": {
        "bytes_sent": 15482035,
        "bytes_received": 25643892
      }
    },
    {
      "id": "QmZ2zc3G8DdeFkrCZSYZ4fJqTtv5mJsP9L5KD7Qe2zNGKT",
      "address": "65.102.25.124:30303",
      "direction": "inbound",
      "name": "Validator 3",
      "protocols": ["eth/66", "snap/1", "bsc/1"],
      "version": "1.4.0",
      "connected_at": "2023-06-04T19:17:05Z",
      "last_message": "2023-06-05T19:15:15Z",
      "latency": 95,
      "traffic": {
        "bytes_sent": 18752035,
        "bytes_received": 22438657
      }
    }
    // Additional peers...
  ],
  "pagination": {
    "total": 15,
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

#### GET /api/node/peers/{peer_id}

Retrieves detailed information about a specific peer.

**Parameters:**
- `peer_id`: The ID of the peer to get information about (required)

**Example Response:**
```json
{
  "id": "QmYyQSiNxTtXP5jbjK5dawHzZ2zZtrshMVN23ffgFLtTQz",
  "address": "54.182.5.123:30303",
  "direction": "outbound",
  "name": "Validator 2",
  "version": "1.4.0",
  "protocols": [
    {
      "name": "eth",
      "version": "66",
      "info": {
        "difficulty": "0x3fcb82f58",
        "head": "0x7d2a32cb629d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3ea6",
        "network": "artha-testnet-1"
      }
    },
    {
      "name": "snap",
      "version": "1"
    },
    {
      "name": "bsc",
      "version": "1"
    }
  ],
  "capabilities": ["eth/66", "eth/67", "snap/1", "bsc/1"],
  "connection": {
    "established": "2023-06-04T19:16:25Z",
    "duration": "24h 58m 30s",
    "local_address": "127.0.0.1:30303",
    "remote_address": "54.182.5.123:30303",
    "encrypted": true,
    "protocol": "tcp"
  },
  "traffic": {
    "bytes_sent": 15482035,
    "bytes_received": 25643892,
    "rate_sent": "15.5 KB/s",
    "rate_received": "25.6 KB/s",
    "messages_sent": 42587,
    "messages_received": 53264
  },
  "latency": {
    "current": 125,
    "average": 118,
    "min": 85,
    "max": 250
  },
  "sync": {
    "status": "in_sync",
    "height": 12890
  }
}
```

### Node Network

#### GET /api/node/network

Retrieves information about the node's network connectivity and configuration.

**Example Response:**
```json
{
  "node_id": "QmX5NLqqHwzm6QjZz7pBoeFwxwTbxp8gGTrSFVTHx5xzDz",
  "local_addresses": [
    {
      "address": "127.0.0.1",
      "port": 30303,
      "nat": "none",
      "discoverable": true
    },
    {
      "address": "203.0.113.15",
      "port": 30303,
      "nat": "none",
      "discoverable": true
    }
  ],
  "listening": true,
  "reachable": true,
  "connections": {
    "active": 15,
    "max": 50,
    "inbound": 8,
    "outbound": 7
  },
  "bandwidth": {
    "total_sent": "1.25 GB",
    "total_received": "3.45 GB",
    "rate_sent": "15.5 KB/s",
    "rate_received": "25.6 KB/s"
  },
  "discovery": {
    "mode": "dns_and_discovery_v5",
    "bootstrap_nodes": 5,
    "discovered_nodes": 42
  },
  "nat_type": "none",
  "upnp_enabled": false,
  "port_mapping": false,
  "firewalled": false,
  "ipv4_enabled": true,
  "ipv6_enabled": false
}
```

### Node Metrics

#### GET /api/node/metrics

Retrieves performance metrics from the node.

**Parameters:**
- `format`: Response format (`json` or `prometheus`) (optional, default: `json`)
- `include`: Comma-separated list of metric types to include (optional, e.g., `system,blockchain,p2p,consensus`)

**Example Response (JSON format):**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "system": {
    "cpu_usage_percent": 12.5,
    "memory_usage_bytes": 1932735488,
    "memory_usage_percent": 45.8,
    "disk_usage_bytes": 5314572288,
    "disk_usage_percent": 25.3,
    "process_uptime_seconds": 86450,
    "open_files": 248,
    "go_routines": 125
  },
  "blockchain": {
    "block_height": 12890,
    "transactions_total": 345678,
    "pending_transactions": 25,
    "state_size_bytes": 3758096384,
    "block_processing_time_ms": {
      "avg": 125.5,
      "p50": 120.3,
      "p90": 180.7,
      "p99": 250.2
    },
    "blocks_per_minute": 4.2,
    "transactions_per_second": 8.5
  },
  "p2p": {
    "peer_count": 15,
    "bytes_sent_total": 15482035,
    "bytes_received_total": 25643892,
    "messages_sent_total": 42587,
    "messages_received_total": 53264,
    "handshakes_total": 87,
    "handshake_failures_total": 12,
    "peer_disconnect_count": 8
  },
  "consensus": {
    "validator_count": 4,
    "rounds_total": 8596,
    "blocks_proposed": 2212,
    "blocks_finalized": 8596,
    "consensus_time_ms": {
      "avg": 850.3,
      "p50": 820.5,
      "p90": 1250.8,
      "p99": 2100.4
    },
    "missed_rounds": 5,
    "voting_power": 100
  },
  "transactions": {
    "verification_time_ms": {
      "avg": 0.82,
      "p50": 0.75,
      "p90": 1.25,
      "p99": 2.5
    },
    "execution_time_ms": {
      "avg": 15.5,
      "p50": 12.8,
      "p90": 28.3,
      "p99": 75.2
    },
    "gas_used": {
      "avg": 56000,
      "p50": 45000,
      "p90": 120000,
      "p99": 500000
    }
  },
  "api": {
    "requests_total": 15982,
    "requests_per_second": 0.85,
    "response_time_ms": {
      "avg": 52.3,
      "p50": 45.8,
      "p90": 85.2,
      "p99": 150.5
    },
    "errors_total": 85
  }
}
```

### Node Logs

#### GET /api/node/logs

Retrieves the most recent logs from the node.

**Parameters:**
- `level`: Minimum log level to include (`debug`, `info`, `warn`, `error`) (optional, default: `info`)
- `module`: Filter logs by module (optional, e.g., `p2p`, `consensus`, `api`, `txpool`)
- `limit`: Maximum number of log entries to return (optional, default: 100, max: 1000)
- `from_time`: Start time for log entries in ISO 8601 format (optional)
- `to_time`: End time for log entries in ISO 8601 format (optional)
- `search`: Search term to filter log entries (optional)

**Example Response:**
```json
{
  "logs": [
    {
      "timestamp": "2023-06-05T19:15:45Z",
      "level": "INFO",
      "module": "consensus",
      "message": "Proposed block at height 12890"
    },
    {
      "timestamp": "2023-06-05T19:15:40Z",
      "level": "INFO",
      "module": "txpool",
      "message": "Added 5 new transactions to the pool"
    },
    {
      "timestamp": "2023-06-05T19:15:35Z",
      "level": "INFO",
      "module": "blockchain",
      "message": "Imported new chain segment at height 12889"
    },
    {
      "timestamp": "2023-06-05T19:15:30Z",
      "level": "INFO",
      "module": "p2p",
      "message": "New peer connected: QmZ2zc3G8DdeFkrCZSYZ4fJqTtv5mJsP9L5KD7Qe2zNGKT"
    },
    {
      "timestamp": "2023-06-05T19:15:25Z",
      "level": "WARN",
      "module": "api",
      "message": "Rate limit exceeded for IP: 203.0.113.42"
    }
    // Additional log entries...
  ],
  "pagination": {
    "total": 25463,
    "limit": 100,
    "has_more": true
  }
}
```

### Node Configuration

#### GET /api/node/config

Retrieves the current configuration of the node. *Requires authentication.*

**Example Response:**
```json
{
  "node": {
    "data_dir": "/data/blockchain",
    "log_level": "info",
    "log_format": "json",
    "log_to_file": true,
    "log_max_size_mb": 100,
    "log_max_files": 10
  },
  "p2p": {
    "listening_address": "0.0.0.0",
    "port": 30303,
    "max_peers": 50,
    "bootstrap_nodes": [
      "enode://5a1e945c2b2b8bc40f402b5d9e5e854f@203.0.113.1:30303",
      "enode://6b2c843d7a9b8bc80f602b8d9e8e955a@203.0.113.2:30303",
      "enode://7d2a32cb629d88c5ecfb5bf42344821f@203.0.113.3:30303"
    ],
    "discovery_enabled": true,
    "discovery_v5_enabled": true,
    "nat": "none",
    "netrestrict": "",
    "trusted_nodes": []
  },
  "rpc": {
    "enabled": true,
    "address": "0.0.0.0",
    "port": 8545,
    "cors_domains": ["*"],
    "vhosts": ["*"],
    "api_modules": ["eth", "net", "web3", "txpool", "debug"]
  },
  "ws": {
    "enabled": true,
    "address": "0.0.0.0",
    "port": 8546,
    "origins": ["*"],
    "api_modules": ["eth", "net", "web3"]
  },
  "consensus": {
    "type": "bft",
    "validator_address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
    "block_period_seconds": 5,
    "request_timeout_ms": 3000,
    "sync_mode": "full"
  },
  "txpool": {
    "locals": ["0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8"],
    "no_locals": false,
    "journal": "transactions.rlp",
    "rejournal_time": "1h",
    "price_limit": 1,
    "price_bump": 10,
    "account_slots": 16,
    "global_slots": 4096,
    "account_queue": 64,
    "global_queue": 1024,
    "lifetime": "3h"
  },
  "state": {
    "sync_mode": "full",
    "gc_mode": "archive",
    "gc_interval": 100,
    "block_history": 128,
    "state_history": 90
  },
  "metrics": {
    "enabled": true,
    "address": "0.0.0.0",
    "port": 9090,
    "interval": 15,
    "prometheus_enabled": true
  },
  "api": {
    "enabled": true,
    "address": "0.0.0.0",
    "port": 3000,
    "rate_limit": {
      "enabled": true,
      "requests_per_minute": 600
    },
    "cors_enabled": true,
    "cors_origins": ["*"]
  }
}
```

#### PUT /api/node/config

Updates the configuration of the node. Only certain parameters are allowed to be changed at runtime. *Requires authentication.*

**Request Body:**
```json
{
  "p2p": {
    "max_peers": 75
  },
  "log_level": "debug",
  "metrics": {
    "interval": 30
  },
  "api": {
    "rate_limit": {
      "requests_per_minute": 1200
    }
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "updated_parameters": [
    "p2p.max_peers",
    "log_level",
    "metrics.interval",
    "api.rate_limit.requests_per_minute"
  ],
  "requires_restart": false
}
```

### Node Admin Operations

#### POST /api/node/shutdown

Initiates a graceful shutdown of the node. *Requires authentication.*

**Request Body:**
```json
{
  "delay_seconds": 30,
  "reason": "Maintenance"
}
```

**Example Response:**
```json
{
  "success": true,
  "message": "Node shutdown initiated",
  "shutdown_time": "2023-06-05T19:16:20Z"
}
```

#### POST /api/node/restart

Initiates a graceful restart of the node. *Requires authentication.*

**Request Body:**
```json
{
  "delay_seconds": 30,
  "reason": "Configuration update"
}
```

**Example Response:**
```json
{
  "success": true,
  "message": "Node restart initiated",
  "restart_time": "2023-06-05T19:16:20Z"
}
```

#### POST /api/node/ban-peer

Bans a peer from connecting to the node. *Requires authentication.*

**Request Body:**
```json
{
  "peer_id": "QmYyQSiNxTtXP5jbjK5dawHzZ2zZtrshMVN23ffgFLtTQz",
  "duration_hours": 24,
  "reason": "Malicious behavior"
}
```

**Example Response:**
```json
{
  "success": true,
  "message": "Peer banned successfully",
  "peer_id": "QmYyQSiNxTtXP5jbjK5dawHzZ2zZtrshMVN23ffgFLtTQz",
  "expires_at": "2023-06-06T19:15:50Z"
}
```

### Node Version

#### GET /api/node/version

Retrieves version information about the node software.

**Example Response:**
```json
{
  "version": "1.4.0",
  "commit_hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
  "build_date": "2023-05-15T14:30:45Z",
  "go_version": "1.18.5",
  "operating_system": "linux",
  "architecture": "amd64",
  "release_name": "Artha Testnet Node v1.4.0",
  "supported_protocols": {
    "eth": ["66", "67"],
    "snap": ["1"],
    "bsc": ["1"]
  },
  "api_version": "1.2.0"
}
```

### WebSocket Subscriptions

Real-time node updates are available through WebSocket connections:

```
ws://localhost:3000/api/node/ws
```

### Subscribe to Node Updates

**Subscription Message:**
```json
{
  "action": "subscribe",
  "topics": [
    "node:status",
    "node:peers",
    "node:blocks",
    "node:metrics"
  ]
}
```

**Example Messages Received:**

Status update:
```json
{
  "topic": "node:status",
  "type": "status_update",
  "data": {
    "status": "online",
    "peer_count": 16,
    "current_block": 12891,
    "synced": true,
    "memory_usage": {
      "current": "1.9 GB",
      "percent": 47
    },
    "timestamp": "2023-06-05T19:16:00Z"
  }
}
```

New peer connected:
```json
{
  "topic": "node:peers",
  "type": "peer_connected",
  "data": {
    "id": "QmZ9zc3G8DdeFkrCZSYZ4fJqTtv5mJsP9L5KD7Qe2zNGZZ",
    "address": "65.102.25.130:30303",
    "direction": "inbound",
    "name": "Light Client 5",
    "protocols": ["eth/66", "snap/1"],
    "version": "1.4.0",
    "connected_at": "2023-06-05T19:16:05Z"
  }
}
```

Metrics update:
```json
{
  "topic": "node:metrics",
  "type": "metrics_update",
  "data": {
    "timestamp": "2023-06-05T19:16:00Z",
    "system": {
      "cpu_usage_percent": 13.2,
      "memory_usage_percent": 47.1
    },
    "blockchain": {
      "transactions_per_second": 9.2,
      "blocks_per_minute": 4.4
    },
    "p2p": {
      "peer_count": 16
    }
  }
}
```

## Error Responses

All API endpoints use standard HTTP status codes and return error details in JSON format:

```json
{
  "error": {
    "code": 400,
    "message": "Invalid parameter",
    "details": "Peer ID must be a valid multihash"
  },
  "timestamp": "2023-06-05T19:15:30Z",
  "request_id": "req-5d23f56b4c0f"
}
```

Common error status codes:
- `400 Bad Request`: Invalid parameters or request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error

## Rate Limiting

API usage is subject to rate limiting:
- Public endpoints: 300 requests per minute
- Authenticated endpoints: 600 requests per minute
- WebSocket connections: 5 concurrent connections per client

When rate limited, the API responds with HTTP status code 429 and includes a Retry-After header.

## SDK Example

```javascript
// JavaScript example using the Artha Chain SDK
const { NodeClient } = require('artha-chain-sdk');

const nodeClient = new NodeClient('http://localhost:3000/api');

// Get node status
nodeClient.getStatus()
  .then(status => {
    console.log('Node status:', status);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });

// Get peers
nodeClient.getPeers()
  .then(peers => {
    console.log('Connected peers:', peers);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });

// Subscribe to status updates
nodeClient.subscribe('node:status', (status) => {
  console.log('Status update:', status);
});
``` 