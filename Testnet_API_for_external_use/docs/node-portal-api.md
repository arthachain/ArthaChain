# Node Portal API

The Node Portal API provides a comprehensive interface for interacting with individual validator nodes on the Artha Chain Testnet. This API gives developers direct access to node-specific functionality, allowing for detailed node status monitoring, configuration, and management.

## Base URL

Each validator node exposes its Portal API at the following base URLs:
- `http://localhost:3000/api/node` (validator1)
- `http://localhost:3001/api/node` (validator2)
- `http://localhost:3002/api/node` (validator3)
- `http://localhost:3003/api/node` (validator4)

## Authentication

Node Portal API endpoints with management capabilities require authentication:

```
Authorization: Bearer your_api_key_here
```

API keys can be obtained by contacting the testnet administrators. Different access levels are available:
- **Read-only**: Access to status and monitoring endpoints
- **Operator**: Access to node configuration and basic management
- **Administrator**: Full access to all endpoints, including security-sensitive operations

## Endpoints

### Node Status

#### GET /api/node/status

Returns the current status of the node.

**Response:**
```json
{
  "timestamp": "2023-06-05T18:30:45Z",
  "node_id": "validator1",
  "network": "testnet",
  "version": "v1.2.0",
  "uptime_seconds": 345600,
  "is_synced": true,
  "sync_status": {
    "current_height": 10584,
    "highest_known_height": 10584,
    "sync_percentage": 100.0,
    "catching_up": false,
    "latest_block_time": "2023-06-05T18:30:35Z",
    "latest_block_hash": "0x7b8d92384f3b7bbe1b5cb7dbb73b5ca6bb459aee1c40e4413ecaf3f5941c6ab2"
  },
  "validator_status": {
    "is_active": true,
    "voting_power": 25000,
    "total_voting_power": 100000,
    "voting_power_percentage": 25.0,
    "missed_blocks": 0,
    "position_in_validator_set": 1
  },
  "resources": {
    "cpu_load": 32.5,
    "memory_usage_percent": 36.3,
    "disk_usage_percent": 15.0
  }
}
```

### Peer Connections

#### GET /api/node/peers

Returns information about the node's connected peers.

**Response:**
```json
{
  "timestamp": "2023-06-05T18:31:45Z",
  "node_id": "validator1",
  "total_peers": 35,
  "connected_peers": [
    {
      "node_id": "QmW2jVsKvRGW8c1T7K4qhPxNtTkba8r1J4oDGhj3RKjPvX",
      "address": "35.180.12.175:26656",
      "direction": "outbound",
      "connection_time": "2023-06-03T12:25:18Z",
      "country": "Germany",
      "version": "v1.2.0",
      "moniker": "community-node-01",
      "latency_ms": 75,
      "is_validator": false
    },
    {
      "node_id": "QmT6ASpxqCaDJzgEoc9nS5TLvYgKZ4TE5AmfDAJyLWh3vR",
      "address": "54.229.210.9:26656",
      "direction": "inbound",
      "connection_time": "2023-06-04T08:12:45Z", 
      "country": "United States",
      "version": "v1.2.0",
      "moniker": "validator2",
      "latency_ms": 85,
      "is_validator": true
    }
    // Additional peers...
  ],
  "persistent_peers": [
    "QmT6ASpxqCaDJzgEoc9nS5TLvYgKZ4TE5AmfDAJyLWh3vR@54.229.210.9:26656",
    "QmZcAhT5YVkkcGX7TRnVfRJAMUr9MzqvNxmgPYQh8YZVff@13.56.30.158:26656",
    "QmYSahMTyP1ffnfhUN7okjRbS15W4xvwLiuoBM9LsC5Tqz@34.239.105.211:26656",
    "QmXs5tf7VCVJQy3rTdGtCkPsxHnmqbnHBNuGzL5cEBqqt4@52.78.110.240:26656"
  ],
  "recent_disconnects": [
    {
      "node_id": "QmR5Z3GaM6gitJZw5fQ9qDViXvBiDDB8yfEEqNRJALxjsr",
      "address": "175.41.178.23:26656",
      "disconnect_time": "2023-06-05T15:46:22Z",
      "disconnect_reason": "connection reset by peer"
    }
  ]
}
```

### Node Configuration

#### GET /api/node/config

Returns the current configuration of the node. *Requires Operator or Administrator authentication.*

**Response:**
```json
{
  "timestamp": "2023-06-05T18:32:45Z",
  "node_id": "validator1",
  "config": {
    "moniker": "validator1",
    "p2p": {
      "listen_address": "tcp://0.0.0.0:26656",
      "external_address": "tcp://35.180.12.173:26656",
      "seeds": "QmZcAhT5YVkkcGX7TRnVfRJAMUr9MzqvNxmgPYQh8YZVff@13.56.30.158:26656,QmYSahMTyP1ffnfhUN7okjRbS15W4xvwLiuoBM9LsC5Tqz@34.239.105.211:26656",
      "persistent_peers": "QmT6ASpxqCaDJzgEoc9nS5TLvYgKZ4TE5AmfDAJyLWh3vR@54.229.210.9:26656,QmXs5tf7VCVJQy3rTdGtCkPsxHnmqbnHBNuGzL5cEBqqt4@52.78.110.240:26656",
      "max_connections": 50,
      "pex": true,
      "seed_mode": false
    },
    "rpc": {
      "listen_address": "tcp://0.0.0.0:26657",
      "max_open_connections": 900,
      "timeout_broadcast_tx_commit": "10s",
      "cors_allowed_origins": ["*"]
    },
    "mempool": {
      "size": 5000,
      "cache_size": 10000,
      "max_tx_bytes": 1048576
    },
    "consensus": {
      "timeout_propose": "3s",
      "timeout_prevote": "1s",
      "timeout_precommit": "1s",
      "timeout_commit": "5s",
      "skip_timeout_commit": false,
      "create_empty_blocks": true,
      "create_empty_blocks_interval": "0s"
    },
    "tx_index": {
      "indexer": "kv",
      "index_all_tags": true
    },
    "instrumentation": {
      "prometheus": true,
      "prometheus_listen_addr": ":26660",
      "max_open_connections": 3
    }
  }
}
```

#### PUT /api/node/config

Updates the node configuration. *Requires Administrator authentication.*

**Request:**
```json
{
  "config": {
    "mempool": {
      "size": 10000,
      "cache_size": 20000
    },
    "p2p": {
      "persistent_peers": "QmT6ASpxqCaDJzgEoc9nS5TLvYgKZ4TE5AmfDAJyLWh3vR@54.229.210.9:26656,QmXs5tf7VCVJQy3rTdGtCkPsxHnmqbnHBNuGzL5cEBqqt4@52.78.110.240:26656,QmYSahMTyP1ffnfhUN7okjRbS15W4xvwLiuoBM9LsC5Tqz@34.239.105.211:26656",
      "max_connections": 75
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2023-06-05T18:33:45Z",
  "message": "Configuration updated successfully",
  "restart_required": true
}
```

### Logs

#### GET /api/node/logs

Returns the recent logs from the node. *Requires Operator or Administrator authentication.*

**Parameters:**
- `lines`: Number of log lines to return (default: 100, max: 1000)
- `level`: Minimum log level to include (default: "info", options: "debug", "info", "warn", "error")
- `module`: Filter logs by module (optional)

**Response:**
```json
{
  "timestamp": "2023-06-05T18:34:45Z",
  "node_id": "validator1",
  "logs": [
    {
      "timestamp": "2023-06-05T18:34:41Z",
      "level": "info",
      "module": "consensus",
      "message": "Executed block height=10584 txs=72 duration=125ms"
    },
    {
      "timestamp": "2023-06-05T18:34:38Z",
      "level": "info",
      "module": "mempool",
      "message": "Added good transaction size=1.2kB"
    },
    {
      "timestamp": "2023-06-05T18:34:36Z",
      "level": "info",
      "module": "p2p",
      "message": "New peer connection nodeID=QmR9KZJEeejnXrF34SvyCYNdY1p3Mvdnwp7jvwGpCG93Td"
    }
    // Additional log entries...
  ],
  "total_entries": 3,
  "has_more": true
}
```

### Node Management

#### POST /api/node/control/restart

Restarts the node. *Requires Administrator authentication.*

**Request:**
```json
{
  "reason": "Applying configuration changes",
  "scheduled_time": "2023-06-05T19:00:00Z"
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2023-06-05T18:35:45Z",
  "message": "Node restart scheduled for 2023-06-05T19:00:00Z",
  "restart_id": "restart-20230605-1900"
}
```

#### GET /api/node/control/restart/{restart_id}

Checks the status of a scheduled restart. *Requires Administrator authentication.*

**Response:**
```json
{
  "restart_id": "restart-20230605-1900",
  "status": "scheduled",
  "scheduled_time": "2023-06-05T19:00:00Z",
  "reason": "Applying configuration changes",
  "requested_by": "admin",
  "requested_at": "2023-06-05T18:35:45Z"
}
```

#### DELETE /api/node/control/restart/{restart_id}

Cancels a scheduled restart. *Requires Administrator authentication.*

**Response:**
```json
{
  "success": true,
  "timestamp": "2023-06-05T18:36:45Z",
  "message": "Restart cancelled successfully"
}
```

### Block Production

#### GET /api/node/blocks/production

Returns metrics about the blocks produced by this validator node. *Requires Operator or Administrator authentication.*

**Response:**
```json
{
  "timestamp": "2023-06-05T18:37:45Z",
  "node_id": "validator1",
  "total_blocks_produced": 2621,
  "blocks_produced_24h": 120,
  "average_block_time_seconds": 5.2,
  "proposer_priority": 15243,
  "next_proposal_estimated_time": "2023-06-05T19:12:15Z",
  "recent_blocks": [
    {
      "height": 10580,
      "time": "2023-06-05T18:27:15Z",
      "num_txs": 68,
      "size_bytes": 14800,
      "processing_time_ms": 132
    },
    {
      "height": 10560,
      "time": "2023-06-05T18:17:15Z",
      "num_txs": 72,
      "size_bytes": 15240,
      "processing_time_ms": 145
    },
    {
      "height": 10540,
      "time": "2023-06-05T18:07:15Z",
      "num_txs": 65,
      "size_bytes": 14200,
      "processing_time_ms": 128
    }
  ]
}
```

### Validator Performance

#### GET /api/node/validator/performance

Returns detailed performance metrics for this validator node. *Requires Operator or Administrator authentication.*

**Response:**
```json
{
  "timestamp": "2023-06-05T18:38:45Z",
  "node_id": "validator1",
  "address": "0x9b23f56b4c0f76c46354a6d5d0766b87a8cfd9e7",
  "performance": {
    "uptime_percentage": 99.98,
    "responsiveness_percentage": 100.0,
    "missed_blocks_total": 0,
    "missed_blocks_30d": 0,
    "missed_blocks_7d": 0,
    "missed_blocks_24h": 0,
    "average_block_latency_ms": 15,
    "average_vote_latency_ms": 12,
    "average_processing_time_ms": 135,
    "double_sign_events": 0,
    "slashing_events": 0
  },
  "ranking": {
    "performance_rank": 1,
    "uptime_rank": 1,
    "responsiveness_rank": 1,
    "commission_rank": 4
  },
  "historical_performance": [
    {
      "date": "2023-06-04",
      "uptime_percentage": 99.97,
      "responsiveness_percentage": 100.0,
      "missed_blocks": 0,
      "average_block_latency_ms": 16
    },
    {
      "date": "2023-06-03",
      "uptime_percentage": 99.98,
      "responsiveness_percentage": 100.0,
      "missed_blocks": 0,
      "average_block_latency_ms": 15
    },
    {
      "date": "2023-06-02",
      "uptime_percentage": 99.99,
      "responsiveness_percentage": 100.0,
      "missed_blocks": 0,
      "average_block_latency_ms": 14
    }
  ]
}
```

### Network Information

#### GET /api/node/network/info

Returns information about the network the node is connected to.

**Response:**
```json
{
  "timestamp": "2023-06-05T18:39:45Z",
  "node_id": "validator1",
  "network": "testnet",
  "network_id": "artha-testnet-042",
  "earliest_block_height": 1,
  "earliest_block_time": "2023-05-01T00:00:00Z",
  "latest_block_height": 10584,
  "latest_block_time": "2023-06-05T18:30:35Z",
  "network_params": {
    "block_time_target_seconds": 5,
    "unbonding_period_seconds": 1209600,
    "max_validators": 100,
    "min_validator_stake": 10000,
    "base_token": "ARTHA",
    "community_tax": 0.02,
    "block_reward": 5.0,
    "gas_limit_per_block": 30000000
  },
  "genesis_hash": "0x5d23f56b4c0f76c46354a6d5d0766b87a8cfd9e72c34e67f8c2f936b1a83b0e7",
  "node_version_compatibility": {
    "minimum": "v1.0.0",
    "recommended": "v1.2.0",
    "latest": "v1.2.0"
  }
}
```

### Diagnostic Tools

#### POST /api/node/diagnostic/ping

Sends a diagnostic ping to check API responsiveness.

**Response:**
```json
{
  "timestamp": "2023-06-05T18:40:45Z",
  "node_id": "validator1",
  "response_time_ms": 5,
  "server_time": "2023-06-05T18:40:45Z",
  "status": "ok"
}
```

#### GET /api/node/diagnostic/health

Performs a health check on various node components.

**Response:**
```json
{
  "timestamp": "2023-06-05T18:41:45Z",
  "node_id": "validator1",
  "overall_health": "healthy",
  "components": {
    "rpc_service": {
      "status": "healthy",
      "response_time_ms": 12,
      "latency_trend": "stable"
    },
    "p2p_network": {
      "status": "healthy",
      "connected_peers": 35,
      "outbound_connections": 17,
      "inbound_connections": 18
    },
    "consensus_engine": {
      "status": "healthy",
      "last_block_age_seconds": 65,
      "voting_power_present": "95%"
    },
    "mempool": {
      "status": "healthy",
      "transaction_count": 87,
      "size_bytes": 1258000,
      "processing_rate_tps": 14.2
    },
    "database": {
      "status": "healthy",
      "response_time_ms": 8,
      "size_gb": 2.15
    },
    "system_resources": {
      "status": "healthy",
      "cpu_usage_percent": 32.5,
      "memory_usage_percent": 36.3,
      "disk_usage_percent": 15.0,
      "network_bandwidth_usage_percent": 8.5
    }
  },
  "alerts": [
    // Empty array indicates no active alerts
  ],
  "recent_issues": [
    {
      "timestamp": "2023-06-04T15:22:12Z",
      "component": "p2p_network",
      "description": "Temporary network partition detected",
      "resolution": "Auto-recovered after 45 seconds",
      "impact": "low"
    }
  ]
}
```

#### GET /api/node/diagnostic/net-info

Returns detailed network diagnostics information. *Requires Operator or Administrator authentication.*

**Response:**
```json
{
  "timestamp": "2023-06-05T18:42:45Z",
  "node_id": "validator1",
  "listening": true,
  "listeners": [
    "tcp://0.0.0.0:26656"
  ],
  "channels": [
    {
      "id": 0,
      "send_queue_capacity": 1000,
      "send_queue_size": 25,
      "priority": "high",
      "recently_sent_bytes": 15240
    },
    {
      "id": 1,
      "send_queue_capacity": 1000,
      "send_queue_size": 42,
      "priority": "medium",
      "recently_sent_bytes": 24800
    }
  ],
  "connection_status": [
    {
      "peer_id": "QmW2jVsKvRGW8c1T7K4qhPxNtTkba8r1J4oDGhj3RKjPvX",
      "address": "35.180.12.175:26656",
      "direction": "outbound",
      "recent_send_bytes": 12450,
      "recent_recv_bytes": 8750,
      "send_packets": 124,
      "recv_packets": 98,
      "ping_time_ms": 75,
      "channels": [0, 1, 2, 3],
      "connected_since": "2023-06-03T12:25:18Z"
    }
    // Additional peer connection status entries...
  ],
  "network_traffic": {
    "bytes_sent_total": 1250000000,
    "bytes_recv_total": 980000000,
    "bytes_sent_rate": 2150000,
    "bytes_recv_rate": 1250000,
    "packets_sent_total": 12500000,
    "packets_recv_total": 9800000,
    "conn_open_total": 1520,
    "conn_close_total": 1485
  }
}
```

## Error Responses

All API endpoints use standard HTTP status codes and return error details in JSON format:

```json
{
  "error": {
    "code": 400,
    "message": "Invalid request parameters",
    "details": "Parameter 'lines' must be a positive integer"
  },
  "timestamp": "2023-06-05T18:42:45Z",
  "request_id": "req-5d23f56b4c0f76c4"
}
```

Common error status codes:
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Node is temporarily unavailable

## Rate Limiting

API usage is subject to rate limiting:
- Public endpoints: 60 requests per minute
- Authenticated endpoints: 300 requests per minute

When rate limited, the API responds with HTTP status code 429 and includes a Retry-After header.

## Versioning

The Node Portal API is versioned to ensure backward compatibility:
- Current version: v1
- Version can be specified in the URL: `/api/v1/node/...`
- Without a version specifier, the latest version is used

Future versions will be announced with appropriate deprecation notices for older versions.

## WebSocket Subscriptions

Real-time updates are available through WebSocket connections:

```
ws://localhost:3000/api/node/ws
```

Available subscription topics:
- `status`: Node status updates
- `blocks`: New block notifications
- `peers`: Peer connection events
- `health`: Health check alerts

Example subscription message:
```json
{
  "subscribe": "blocks",
  "include_data": true
}
```

Example message received:
```json
{
  "topic": "blocks",
  "timestamp": "2023-06-05T18:45:15Z",
  "data": {
    "height": 10585,
    "hash": "0x9c45f92b020f3b55c8c26db75c82d876ab34f0f1a8cfd9e7762f9db759792a9",
    "proposer": "validator3",
    "num_txs": 75,
    "time": "2023-06-05T18:45:12Z"
  }
}
``` 