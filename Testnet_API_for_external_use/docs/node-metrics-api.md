# Node Metrics API

The Artha Chain Testnet Node Metrics API provides comprehensive performance and health metrics for blockchain nodes. This API enables developers, validators, and network participants to monitor node performance, network status, and blockchain metrics in real-time.

## Base URL

All API endpoints are relative to the base URL of a validator node:
- `http://localhost:3000` (validator1)
- `http://localhost:3001` (validator2)
- `http://localhost:3002` (validator3)
- `http://localhost:3003` (validator4)

## Authentication

Most endpoints in the Node Metrics API are publicly accessible for read operations. Authentication is required for endpoints that modify settings. See the [Authentication](./authentication.md) documentation for details.

## Blockchain Metrics

### Get Blockchain Status

Retrieves the current status of the blockchain network.

**Endpoint:** `GET /api/metrics/blockchain/status`

**Response:**
```json
{
  "block_height": 1024,
  "latest_block_hash": "0x7c17f3812a36fc1524ca2890a5f5d8a7c1d6e71238afc245c7b92a1f3d6a45cb",
  "latest_block_timestamp": "2023-06-01T12:34:56Z",
  "average_block_time": 30.2,
  "active_shards": 4,
  "network_status": "healthy",
  "epoch": 42,
  "total_transactions": 10240,
  "pending_transactions": 15
}
```

### Get Block Production Metrics

Retrieves metrics related to block production across the network.

**Endpoint:** `GET /api/metrics/blockchain/blocks`

**Query Parameters:**
- `period` (optional): Time period for statistics (1h, 24h, 7d, 30d)
- `shard_id` (optional): Filter metrics by specific shard

**Response:**
```json
{
  "blocks_produced": {
    "1h": 120,
    "24h": 2880,
    "7d": 20160,
    "30d": 86400
  },
  "average_block_time": {
    "1h": 30.2,
    "24h": 30.0,
    "7d": 30.1,
    "30d": 30.0
  },
  "missed_blocks": {
    "1h": 0,
    "24h": 12,
    "7d": 42,
    "30d": 156
  },
  "block_sizes": {
    "average": 5240,
    "min": 1024,
    "max": 12500
  },
  "by_shard": [
    {
      "shard_id": 0,
      "blocks_produced": 28800,
      "average_block_time": 30.1
    },
    {
      "shard_id": 1,
      "blocks_produced": 28750,
      "average_block_time": 30.2
    }
  ]
}
```

### Get Transaction Metrics

Retrieves metrics related to transactions processed by the network.

**Endpoint:** `GET /api/metrics/blockchain/transactions`

**Query Parameters:**
- `period` (optional): Time period for statistics (1h, 24h, 7d, 30d)
- `shard_id` (optional): Filter metrics by specific shard

**Response:**
```json
{
  "transactions_processed": {
    "1h": 1200,
    "24h": 28800,
    "7d": 201600,
    "30d": 864000
  },
  "tps": {
    "current": 10.2,
    "average_1h": 10.0,
    "average_24h": 9.5,
    "peak_24h": 25.7
  },
  "gas_used": {
    "1h": 12000000,
    "24h": 288000000,
    "7d": 2016000000,
    "30d": 8640000000
  },
  "average_gas_price": {
    "1h": 1.2,
    "24h": 1.3,
    "7d": 1.25,
    "30d": 1.3
  },
  "transaction_types": {
    "transfer": 70.5,
    "contract_deployment": 5.2,
    "contract_call": 24.3
  },
  "by_shard": [
    {
      "shard_id": 0,
      "transactions_processed": 10240,
      "current_tps": 8.5
    },
    {
      "shard_id": 1,
      "transactions_processed": 9180,
      "current_tps": 7.2
    }
  ]
}
```

### Get Network Economics Metrics

Retrieves economic metrics for the blockchain network.

**Endpoint:** `GET /api/metrics/blockchain/economics`

**Response:**
```json
{
  "total_supply": 100000000,
  "circulating_supply": 75000000,
  "staked_amount": 25000000,
  "transaction_fees_24h": 12500,
  "burned_fees_24h": 6250,
  "rewards_distributed_24h": 4320,
  "staking_apy": 12.5,
  "average_transaction_fee": 0.05,
  "active_validators": 25,
  "active_miners": 75
}
```

## Node Performance Metrics

### Get Node Resource Usage

Retrieves resource usage metrics for a specific node.

**Endpoint:** `GET /api/metrics/nodes/:id/resources`

**Parameters:**
- `id` (path parameter): Node identifier

**Query Parameters:**
- `period` (optional): Time period for statistics (1h, 24h, 7d, 30d)

**Response:**
```json
{
  "id": "validator1",
  "cpu": {
    "current": 45.2,
    "average_1h": 42.5,
    "average_24h": 40.1,
    "peak_24h": 75.3
  },
  "memory": {
    "current": 3.5,
    "average_1h": 3.2,
    "average_24h": 3.3,
    "peak_24h": 4.1,
    "total": 8.0,
    "used_percent": 43.75
  },
  "disk": {
    "usage": 68.2,
    "available": 31.8,
    "total": 100.0,
    "read_rate": 5.2,
    "write_rate": 8.7
  },
  "network": {
    "in_bandwidth": 12500,
    "out_bandwidth": 18700,
    "peers": 15,
    "latency_ms": 125
  }
}
```

### Get Node Consensus Performance

Retrieves consensus participation metrics for a specific node.

**Endpoint:** `GET /api/metrics/nodes/:id/consensus`

**Parameters:**
- `id` (path parameter): Node identifier

**Query Parameters:**
- `period` (optional): Time period for statistics (1h, 24h, 7d, 30d)

**Response:**
```json
{
  "id": "validator1",
  "block_production": {
    "proposed": 1050,
    "produced": 1024,
    "missed": 26,
    "performance": 97.52
  },
  "voting": {
    "total_votes": 10240,
    "correct_votes": 10235,
    "missed_votes": 5,
    "performance": 99.95
  },
  "rewards": {
    "total": 1250,
    "from_blocks": 768,
    "from_votes": 482
  },
  "epoch_performance": [
    {
      "epoch": 42,
      "performance": 99.2,
      "blocks_produced": 120,
      "blocks_missed": 1
    },
    {
      "epoch": 41,
      "performance": 98.3,
      "blocks_produced": 118,
      "blocks_missed": 2
    }
  ]
}
```

### Get Node Network Performance

Retrieves network-related performance metrics for a specific node.

**Endpoint:** `GET /api/metrics/nodes/:id/network`

**Parameters:**
- `id` (path parameter): Node identifier

**Query Parameters:**
- `period` (optional): Time period for statistics (1h, 24h, 7d, 30d)

**Response:**
```json
{
  "id": "validator1",
  "peers": {
    "current": 15,
    "average_24h": 14.2,
    "max": 18,
    "by_region": {
      "NA": 5,
      "EU": 6,
      "APAC": 4
    }
  },
  "latency": {
    "average_ms": 125,
    "min_ms": 80,
    "max_ms": 350,
    "p95_ms": 200
  },
  "bandwidth": {
    "in_current": 12500,
    "out_current": 18700,
    "in_average_24h": 10200,
    "out_average_24h": 15600,
    "in_peak_24h": 25000,
    "out_peak_24h": 35000
  },
  "message_processing": {
    "received_1h": 12500,
    "sent_1h": 18700,
    "failed_1h": 5
  },
  "sync_status": {
    "synced": true,
    "behind_blocks": 0,
    "sync_speed_blocks_per_minute": 0
  }
}
```

## SVDB Metrics

### Get SVDB Storage Metrics

Retrieves storage metrics for the SVDB (State-Value Database).

**Endpoint:** `GET /api/metrics/svdb/storage`

**Query Parameters:**
- `shard_id` (optional): Filter metrics by specific shard

**Response:**
```json
{
  "total_storage": {
    "bytes": 1024000000,
    "formatted": "1.02 GB"
  },
  "storage_by_shard": [
    {
      "shard_id": 0,
      "bytes": 368640000,
      "formatted": "368.64 MB"
    },
    {
      "shard_id": 1,
      "bytes": 327680000, 
      "formatted": "327.68 MB"
    }
  ],
  "storage_growth": {
    "24h": {
      "bytes": 10240000,
      "formatted": "10.24 MB"
    },
    "7d": {
      "bytes": 71680000,
      "formatted": "71.68 MB"
    },
    "30d": {
      "bytes": 307200000,
      "formatted": "307.2 MB"
    }
  },
  "keys_count": 852000,
  "storage_by_type": {
    "account_state": 30.5,
    "contract_code": 15.2,
    "contract_state": 48.3,
    "system": 6.0
  }
}
```

### Get SVDB Performance Metrics

Retrieves performance metrics for the SVDB operations.

**Endpoint:** `GET /api/metrics/svdb/performance`

**Response:**
```json
{
  "operations": {
    "reads_per_second": 1250.5,
    "writes_per_second": 320.8,
    "deletes_per_second": 15.2
  },
  "latency_ms": {
    "read": {
      "average": 2.5,
      "p95": 4.8,
      "p99": 8.2
    },
    "write": {
      "average": 5.1,
      "p95": 12.3,
      "p99": 18.7
    },
    "delete": {
      "average": 4.2,
      "p95": 10.5,
      "p99": 15.8
    }
  },
  "cache": {
    "hit_rate": 92.5,
    "size_bytes": 104857600,
    "formatted_size": "100 MB",
    "items": 125000
  },
  "compaction": {
    "last_time": "2023-06-01T10:15:23Z",
    "duration_seconds": 425,
    "reclaimed_bytes": 52428800,
    "formatted_reclaimed": "50 MB"
  }
}
```

## Network Status

### Get Network Health

Retrieves overall health and status information for the network.

**Endpoint:** `GET /api/metrics/network/health`

**Response:**
```json
{
  "status": "healthy",
  "active_nodes": 100,
  "active_validators": 25,
  "active_miners": 75,
  "active_shards": 4,
  "network_stability": 99.8,
  "average_block_time": 30.2,
  "current_tps": 10.2,
  "peer_connectivity": 98.5,
  "issues": [
    {
      "severity": "warning",
      "description": "Shard 2 experiencing slightly elevated block times",
      "detected_at": "2023-06-01T11:45:32Z"
    }
  ]
}
```

### Get Network Topology

Retrieves information about the current network topology.

**Endpoint:** `GET /api/metrics/network/topology`

**Response:**
```json
{
  "nodes_count": 100,
  "regions": {
    "NA": 35,
    "EU": 40,
    "APAC": 20,
    "Other": 5
  },
  "shard_distribution": [
    {
      "shard_id": 0,
      "nodes": 25,
      "validators": 7,
      "miners": 18
    },
    {
      "shard_id": 1,
      "nodes": 25,
      "validators": 6,
      "miners": 19
    }
  ],
  "connection_graph": {
    "average_connections": 12.5,
    "fully_connected": true,
    "network_diameter": 4
  },
  "nodes_by_version": {
    "0.5.2": 85,
    "0.5.1": 15
  }
}
```

## System Metrics

### Get System Alerts

Retrieves active alerts and warnings for the network.

**Endpoint:** `GET /api/metrics/system/alerts`

**Query Parameters:**
- `severity` (optional): Filter alerts by severity (info, warning, critical)
- `type` (optional): Filter alerts by type (node, network, blockchain, svdb)

**Response:**
```json
{
  "alerts": [
    {
      "id": "alert-12345",
      "severity": "warning",
      "type": "node",
      "title": "Node validator3 high CPU usage",
      "description": "CPU usage has been above 80% for over 30 minutes",
      "created_at": "2023-06-01T11:32:45Z",
      "status": "active",
      "affected_entity": "validator3"
    },
    {
      "id": "alert-12346",
      "severity": "info",
      "type": "network",
      "title": "Network upgrade scheduled",
      "description": "Network upgrade to version 0.6.0 scheduled for June 15, 2023",
      "created_at": "2023-06-01T09:00:00Z",
      "status": "active",
      "affected_entity": "network"
    }
  ],
  "total_alerts": 2,
  "by_severity": {
    "critical": 0,
    "warning": 1,
    "info": 1
  }
}
```

### Get Historical Metrics

Retrieves historical time-series data for specified metrics.

**Endpoint:** `GET /api/metrics/system/historical`

**Query Parameters:**
- `metric` (required): The metric to retrieve (tps, block_time, node_count, gas_price)
- `period` (optional): Time period for data (24h, 7d, 30d, 90d, 1y)
- `resolution` (optional): Data point resolution (1m, 5m, 1h, 1d)
- `shard_id` (optional): Filter metrics by specific shard

**Response:**
```json
{
  "metric": "tps",
  "period": "24h",
  "resolution": "1h",
  "data_points": [
    {
      "timestamp": "2023-06-01T12:00:00Z",
      "value": 12.5
    },
    {
      "timestamp": "2023-06-01T11:00:00Z",
      "value": 10.2
    },
    {
      "timestamp": "2023-06-01T10:00:00Z",
      "value": 11.8
    }
  ],
  "statistics": {
    "average": 11.5,
    "min": 8.2,
    "max": 25.7,
    "std_dev": 2.3
  }
}
```

## Node Metrics Configuration

### Get Metrics Configuration

Retrieves the current metrics collection configuration.

**Endpoint:** `GET /api/metrics/config`

**Authentication Required**

**Response:**
```json
{
  "collection_interval": {
    "system": 60,
    "blockchain": 30,
    "network": 60,
    "svdb": 300
  },
  "storage_retention": {
    "high_resolution": "7d",
    "medium_resolution": "30d",
    "low_resolution": "1y"
  },
  "enabled_metrics": {
    "system": true,
    "blockchain": true,
    "network": true,
    "svdb": true,
    "detailed_resource": true
  },
  "alert_thresholds": {
    "cpu_usage": 80,
    "memory_usage": 85,
    "disk_usage": 90,
    "missed_blocks": 5,
    "peer_count_min": 5
  }
}
```

### Update Metrics Configuration

Updates the metrics collection configuration.

**Endpoint:** `PUT /api/metrics/config`

**Authentication Required**

**Request:**
```json
{
  "collection_interval": {
    "system": 30,
    "blockchain": 15
  },
  "enabled_metrics": {
    "detailed_resource": false
  },
  "alert_thresholds": {
    "cpu_usage": 85,
    "memory_usage": 90
  }
}
```

**Response:**
```json
{
  "status": "updated",
  "updated_fields": [
    "collection_interval.system",
    "collection_interval.blockchain",
    "enabled_metrics.detailed_resource",
    "alert_thresholds.cpu_usage",
    "alert_thresholds.memory_usage"
  ],
  "applied_at": "2023-06-01T12:45:30Z"
}
```

## Error Responses

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_PARAMETER | Invalid parameter value provided |
| 401 | UNAUTHORIZED | Authentication required or failed |
| 403 | FORBIDDEN | Not authorized to access this resource |
| 404 | RESOURCE_NOT_FOUND | The requested resource does not exist |
| 429 | RATE_LIMITED | Too many requests, exceeded rate limit |
| 500 | SERVER_ERROR | Internal server error |

### Error Response Example:

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid metric name provided",
    "details": {
      "parameter": "metric",
      "provided": "unknown_metric",
      "valid_values": ["tps", "block_time", "node_count", "gas_price"]
    }
  }
}
```

## Implementation Notes

- Metrics are collected at regular intervals and cached for efficient retrieval
- Historical metrics use time-series database storage with automatic downsampling
- Rate limits apply to prevent excessive API usage (see [Rate Limiting](./rate-limiting.md))
- Some metrics may be delayed by up to the collection interval time
- All time-based metrics are provided in UTC timezone
- Node performance metrics are only available for active nodes 