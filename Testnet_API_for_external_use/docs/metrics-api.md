# Metrics API

The Metrics API provides comprehensive performance and operational metrics for the Artha Chain Testnet. This API allows developers to monitor blockchain performance, analyze transaction trends, track network health, and gather statistics for reporting and visualization purposes.

## Base URL

Each validator node exposes the Metrics API at the following base URLs:
- `http://localhost:3000/api/metrics` (validator1)
- `http://localhost:3001/api/metrics` (validator2)
- `http://localhost:3002/api/metrics` (validator3)
- `http://localhost:3003/api/metrics` (validator4)

## Authentication

Most metrics endpoints are public and don't require authentication. Some historical or detailed metrics endpoints require authentication:

```
Authorization: Bearer your_api_key_here
```

API keys can be obtained from the [Artha Chain Testnet Portal](http://localhost:3000/portal).

## Global Parameters

All endpoints support the following query parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | string | Response format: `json` (default) or `prometheus` |
| `interval` | string | Time interval for time-series data: `1m`, `5m`, `15m`, `1h`, `6h`, `24h` (default depends on metric) |
| `from` | ISO date | Start time for time-series data (default: 24 hours ago) |
| `to` | ISO date | End time for time-series data (default: current time) |

## Endpoints

### Network Performance

#### GET /api/metrics/performance

Retrieves current performance metrics for the blockchain network.

**Parameters:**
- `interval`: Time interval for metrics calculation (`1m`, `5m`, `15m`, `1h`, `24h`) (optional, default: `5m`)

**Example Response:**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "interval": "5m",
  "transactions": {
    "count": 2563,
    "tps": {
      "current": 8.5,
      "peak": 12.3,
      "average": 8.2
    },
    "confirmation_time_ms": {
      "average": 850,
      "p50": 750,
      "p90": 1200,
      "p99": 2100
    },
    "gas_price_gwei": {
      "average": 5.2,
      "min": 2.0,
      "max": 15.0
    },
    "success_rate": 0.985
  },
  "blocks": {
    "count": 60,
    "time_between_blocks_ms": {
      "average": 5024,
      "min": 4850,
      "max": 5320
    },
    "size_bytes": {
      "average": 35420,
      "min": 5200,
      "max": 120500
    },
    "gas_used_percent": {
      "average": 42.5,
      "min": 15.3,
      "max": 85.7
    }
  },
  "network": {
    "active_validators": 4,
    "active_nodes": 15,
    "peer_count_average": 12.5,
    "consensus_rounds": 60,
    "finality_time_ms": {
      "average": 875,
      "p50": 850,
      "p90": 950,
      "p99": 1200
    }
  },
  "resources": {
    "cpu_usage_percent": {
      "average": 12.5,
      "min": 8.2,
      "max": 18.7
    },
    "memory_usage_percent": {
      "average": 45.8,
      "min": 42.3,
      "max": 48.2
    }
  }
}
```

### Transaction Metrics

#### GET /api/metrics/transactions

Retrieves transaction-related metrics for the specified period.

**Parameters:**
- `period`: Time period for metrics calculation (`hour`, `day`, `week`, `month`) (optional, default: `hour`)
- `detailed`: Include detailed metrics breakdown (optional, default: `false`)

**Example Response:**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "period": "hour",
  "total_transactions": 30580,
  "transactions_per_second": {
    "current": 8.5,
    "peak": 15.3,
    "average": 8.2,
    "min": 3.2
  },
  "transaction_types": {
    "transfer": 18348,
    "contract_call": 8562,
    "contract_creation": 254,
    "other": 3416
  },
  "gas_usage": {
    "total": 42580560000,
    "average_per_tx": 135000,
    "median_per_tx": 75000,
    "max_per_tx": 12500000
  },
  "fees": {
    "total_eth": "5.32",
    "average_eth": "0.000175",
    "median_eth": "0.000125",
    "total_usd_approx": 9576.0,
    "average_usd_approx": 0.315
  },
  "confirmation_times_ms": {
    "average": 850,
    "median": 750,
    "p90": 1200,
    "p99": 2100
  },
  "success_rate": 0.985,
  "failure_reasons": {
    "out_of_gas": 210,
    "revert": 175,
    "nonce_too_low": 45,
    "underpriced": 35,
    "other": 15
  },
  "trending_contracts": [
    {
      "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
      "calls": 1250,
      "unique_callers": 350
    },
    {
      "address": "0x1A2F8fC8c9f481234345677F1AbCdEfC1234567A",
      "calls": 1125,
      "unique_callers": 225
    }
  ]
}
```

### Block Metrics

#### GET /api/metrics/blocks

Retrieves block-related metrics for the specified period.

**Parameters:**
- `period`: Time period for metrics calculation (`hour`, `day`, `week`, `month`) (optional, default: `hour`)

**Example Response:**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "period": "hour",
  "total_blocks": 720,
  "block_time_ms": {
    "average": 5050,
    "min": 4850,
    "max": 8200,
    "median": 5000,
    "p90": 5250,
    "p99": 7500
  },
  "block_size_bytes": {
    "average": 35420,
    "min": 5200,
    "max": 120500,
    "median": 32500
  },
  "gas_usage": {
    "average_limit": 12500000,
    "average_used": 5312500,
    "average_percent_used": 42.5,
    "min_percent_used": 15.3,
    "max_percent_used": 85.7
  },
  "transactions_per_block": {
    "average": 42.5,
    "median": 38.0,
    "min": 0,
    "max": 155
  },
  "uncle_blocks": 3,
  "empty_blocks": 8,
  "full_blocks": 25,
  "validators": {
    "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8": 180,
    "0x2F2b8bc9F3F9bCE73f7F73D9A3E80D890bDe43e4": 178,
    "0x1a2Bc43e789DEf4b5C9388DDed25A8D5d3F7E5Cb": 182,
    "0x3F5cE5FBFe3E9af3971dD833D26bA9b5C936f0bE": 180
  }
}
```

### Gas Metrics

#### GET /api/metrics/gas

Retrieves gas usage and pricing metrics.

**Parameters:**
- `period`: Time period for metrics calculation (`hour`, `day`, `week`, `month`) (optional, default: `hour`)

**Example Response:**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "period": "hour",
  "gas_price_gwei": {
    "current": 5.2,
    "min": 2.0,
    "max": 15.0,
    "average": 5.5,
    "median": 5.0,
    "recommended": {
      "slow": 3.0,
      "standard": 5.0,
      "fast": 8.0,
      "instant": 12.5
    }
  },
  "gas_used": {
    "total": 42580560000,
    "average_per_block": 5312500,
    "average_per_transaction": 135000,
    "utilization_percent": 42.5
  },
  "gas_limit": {
    "average_per_block": 12500000,
    "total": 9000000000000
  },
  "transaction_costs": {
    "eth": {
      "average": "0.000175",
      "median": "0.000125",
      "min": "0.000021",
      "max": "0.002350"
    },
    "usd_approx": {
      "average": 0.315,
      "median": 0.225,
      "min": 0.038,
      "max": 4.23
    }
  },
  "gas_by_transaction_type": {
    "transfer": {
      "average": 21000,
      "total": 385350000
    },
    "contract_call": {
      "average": 185000,
      "total": 1583970000000
    },
    "contract_creation": {
      "average": 1250000,
      "total": 317500000000
    }
  }
}
```

### Validator Metrics

#### GET /api/metrics/validators

Retrieves performance metrics for validators.

**Parameters:**
- `period`: Time period for metrics calculation (`hour`, `day`, `week`, `month`) (optional, default: `hour`)

**Example Response:**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "period": "hour",
  "total_validators": 4,
  "active_validators": 4,
  "total_stake": "400000.00",
  "validators": [
    {
      "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
      "name": "Validator 1",
      "stake": "100000.00",
      "blocks_proposed": 180,
      "blocks_signed": 720,
      "signature_rate": 1.0,
      "uptime": 1.0,
      "reward_eth": "1.25",
      "performance_score": 0.995,
      "latency_ms": {
        "average": 85,
        "min": 70,
        "max": 150
      }
    },
    {
      "address": "0x2F2b8bc9F3F9bCE73f7F73D9A3E80D890bDe43e4",
      "name": "Validator 2",
      "stake": "100000.00",
      "blocks_proposed": 178,
      "blocks_signed": 720,
      "signature_rate": 1.0,
      "uptime": 0.998,
      "reward_eth": "1.24",
      "performance_score": 0.990,
      "latency_ms": {
        "average": 90,
        "min": 75,
        "max": 180
      }
    },
    {
      "address": "0x1a2Bc43e789DEf4b5C9388DDed25A8D5d3F7E5Cb",
      "name": "Validator 3",
      "stake": "100000.00",
      "blocks_proposed": 182,
      "blocks_signed": 720,
      "signature_rate": 1.0,
      "uptime": 1.0,
      "reward_eth": "1.26",
      "performance_score": 0.998,
      "latency_ms": {
        "average": 82,
        "min": 68,
        "max": 145
      }
    },
    {
      "address": "0x3F5cE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
      "name": "Validator 4",
      "stake": "100000.00",
      "blocks_proposed": 180,
      "blocks_signed": 720,
      "signature_rate": 1.0,
      "uptime": 0.999,
      "reward_eth": "1.25",
      "performance_score": 0.992,
      "latency_ms": {
        "average": 88,
        "min": 72,
        "max": 160
      }
    }
  ],
  "consensus_stats": {
    "rounds": 720,
    "average_rounds_per_block": 1.0,
    "max_rounds_per_block": 3,
    "average_signature_collection_time_ms": 750,
    "finality_time_ms": {
      "average": 875,
      "p50": 850,
      "p90": 950,
      "p99": 1200
    }
  }
}
```

### Network Metrics

#### GET /api/metrics/network

Retrieves network-wide metrics and statistics.

**Parameters:**
- `period`: Time period for metrics calculation (`hour`, `day`, `week`, `month`) (optional, default: `hour`)

**Example Response:**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "period": "hour",
  "nodes": {
    "total": 15,
    "validators": 4,
    "full_nodes": 8,
    "light_clients": 3,
    "by_region": {
      "na": 6,
      "eu": 5,
      "asia": 3,
      "other": 1
    },
    "by_version": {
      "1.4.0": 12,
      "1.3.2": 3
    }
  },
  "connections": {
    "total": 78,
    "average_per_node": 10.4,
    "min_per_node": 6,
    "max_per_node": 15
  },
  "bandwidth": {
    "total_mb": 8650,
    "average_mb_per_node": 576.7,
    "average_kb_per_second": 2400.0,
    "by_message_type": {
      "transactions": 4850,
      "blocks": 2100,
      "consensus": 950,
      "state_sync": 650,
      "other": 100
    }
  },
  "latency_ms": {
    "average": 95,
    "min": 65,
    "max": 210,
    "p50": 90,
    "p90": 150,
    "p99": 195
  },
  "propagation_time_ms": {
    "transaction": {
      "average": 125,
      "p50": 110,
      "p90": 180,
      "p99": 250
    },
    "block": {
      "average": 210,
      "p50": 195,
      "p90": 285,
      "p99": 350
    }
  },
  "sync_status": {
    "fully_synced_percent": 0.95,
    "average_behind_blocks": 0.8,
    "max_behind_blocks": 12
  }
}
```

### Account Metrics

#### GET /api/metrics/accounts

Retrieves metrics related to accounts and their activity.

**Parameters:**
- `period`: Time period for metrics calculation (`hour`, `day`, `week`, `month`) (optional, default: `day`)

**Example Response:**
```json
{
  "timestamp": "2023-06-05T19:15:50Z",
  "period": "day",
  "accounts": {
    "total": 12850,
    "new": 285,
    "active": 1520,
    "with_balance": 9850,
    "zero_balance": 3000,
    "with_code": 550
  },
  "balance_distribution": {
    "total_eth": "1250000.00",
    "average_eth": "97.28",
    "median_eth": "0.52",
    "ranges": [
      {
        "range": "0 ETH",
        "count": 3000,
        "percent": 23.35
      },
      {
        "range": "0-0.1 ETH",
        "count": 3540,
        "percent": 27.55
      },
      {
        "range": "0.1-1 ETH",
        "count": 2985,
        "percent": 23.23
      },
      {
        "range": "1-10 ETH",
        "count": 1850,
        "percent": 14.40
      },
      {
        "range": "10-100 ETH",
        "count": 950,
        "percent": 7.39
      },
      {
        "range": "100-1000 ETH",
        "count": 420,
        "percent": 3.27
      },
      {
        "range": "1000+ ETH",
        "count": 105,
        "percent": 0.82
      }
    ]
  },
  "activity": {
    "transactions_per_account": {
      "average": 2.5,
      "median": 1.0,
      "max": 450
    },
    "gas_used_per_account": {
      "average": 225000,
      "median": 75000,
      "max": 45000000
    },
    "top_accounts_by_transactions": [
      "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
      "0x1A2F8fC8c9f481234345677F1AbCdEfC1234567A",
      "0x3c7A661D33F7Ee489cb8E4c86F94a33EFB87BA03"
    ]
  },
  "smart_contracts": {
    "total": 550,
    "new": 35,
    "active": 125,
    "calls": {
      "total": 8562,
      "average_per_contract": 68.5,
      "max_per_contract": 1250
    },
    "creation": {
      "successful": 35,
      "failed": 4
    },
    "top_contracts_by_calls": [
      "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
      "0x1A2F8fC8c9f481234345677F1AbCdEfC1234567A"
    ]
  }
}
```

### Historical Metrics

#### GET /api/metrics/historical

Retrieves historical metrics with support for time series data. *Requires authentication for detailed data.*

**Parameters:**
- `metric`: Metric to retrieve historical data for (`tps`, `gas_price`, `block_time`, `active_accounts`, etc.)
- `start_time`: Start time in ISO 8601 format (required)
- `end_time`: End time in ISO 8601 format (required)
- `resolution`: Data point resolution (`minute`, `hour`, `day`) (optional, default depends on time range)

**Example Response:**
```json
{
  "metric": "tps",
  "start_time": "2023-06-04T19:00:00Z",
  "end_time": "2023-06-05T19:00:00Z",
  "resolution": "hour",
  "data_points": [
    {
      "timestamp": "2023-06-04T19:00:00Z",
      "value": 7.8
    },
    {
      "timestamp": "2023-06-04T20:00:00Z",
      "value": 8.2
    },
    {
      "timestamp": "2023-06-04T21:00:00Z",
      "value": 9.5
    },
    {
      "timestamp": "2023-06-04T22:00:00Z",
      "value": 10.2
    },
    // Additional data points...
    {
      "timestamp": "2023-06-05T18:00:00Z",
      "value": 8.5
    }
  ],
  "statistics": {
    "average": 8.7,
    "min": {
      "value": 4.2,
      "timestamp": "2023-06-05T04:00:00Z"
    },
    "max": {
      "value": 12.8,
      "timestamp": "2023-06-05T14:00:00Z"
    },
    "median": 8.5,
    "trend": "+0.7 (8.8%)"
  }
}
```

### Custom Metrics Dashboard

#### POST /api/metrics/dashboard

Creates a custom metrics dashboard with selected metrics. *Requires authentication.*

**Request Body:**
```json
{
  "name": "Performance Dashboard",
  "description": "Key performance indicators for the Artha Chain",
  "refresh_interval_seconds": 60,
  "metrics": [
    {
      "id": "tps",
      "display_name": "Transactions Per Second",
      "period": "hour",
      "chart_type": "line",
      "include_history": true,
      "history_points": 24
    },
    {
      "id": "gas_price",
      "display_name": "Gas Price (Gwei)",
      "period": "hour",
      "chart_type": "line",
      "include_history": true,
      "history_points": 24
    },
    {
      "id": "block_time",
      "display_name": "Average Block Time",
      "period": "hour",
      "chart_type": "line",
      "include_history": true,
      "history_points": 24
    },
    {
      "id": "active_validators",
      "display_name": "Active Validators",
      "period": "hour",
      "chart_type": "number"
    },
    {
      "id": "success_rate",
      "display_name": "Transaction Success Rate",
      "period": "hour",
      "chart_type": "gauge"
    }
  ]
}
```

**Example Response:**
```json
{
  "dashboard_id": "d8f7e6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l",
  "name": "Performance Dashboard",
  "created_at": "2023-06-05T19:15:50Z",
  "access_url": "http://localhost:3000/portal/dashboards/d8f7e6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l",
  "api_url": "http://localhost:3000/api/metrics/dashboard/d8f7e6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l",
  "refresh_interval_seconds": 60,
  "metrics_count": 5
}
```

#### GET /api/metrics/dashboard/{dashboard_id}

Retrieves data for a previously created custom metrics dashboard. *Requires authentication.*

**Example Response:**
```json
{
  "dashboard_id": "d8f7e6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l",
  "name": "Performance Dashboard",
  "description": "Key performance indicators for the Artha Chain",
  "created_at": "2023-06-05T19:15:50Z",
  "updated_at": "2023-06-05T19:20:50Z",
  "metrics": [
    {
      "id": "tps",
      "display_name": "Transactions Per Second",
      "current_value": 8.5,
      "trend": "+0.3 (3.7%)",
      "history": [
        {
          "timestamp": "2023-06-04T19:00:00Z",
          "value": 7.8
        },
        // Additional data points...
        {
          "timestamp": "2023-06-05T19:00:00Z",
          "value": 8.5
        }
      ]
    },
    {
      "id": "gas_price",
      "display_name": "Gas Price (Gwei)",
      "current_value": 5.2,
      "trend": "-0.3 (-5.5%)",
      "history": [
        {
          "timestamp": "2023-06-04T19:00:00Z",
          "value": 5.5
        },
        // Additional data points...
        {
          "timestamp": "2023-06-05T19:00:00Z",
          "value": 5.2
        }
      ]
    },
    // Additional metrics...
  ]
}
```

### Export Metrics

#### GET /api/metrics/export

Exports metrics data in various formats. *Requires authentication.*

**Parameters:**
- `metrics`: Comma-separated list of metrics to export (required)
- `start_time`: Start time in ISO 8601 format (required)
- `end_time`: End time in ISO 8601 format (required)
- `resolution`: Data point resolution (`minute`, `hour`, `day`) (optional, default depends on time range)
- `format`: Export format (`csv`, `json`) (optional, default: `csv`)

**Example Response (JSON format):**
```json
{
  "export_id": "exp-5d23f56b4c0f",
  "created_at": "2023-06-05T19:15:50Z",
  "parameters": {
    "metrics": ["tps", "gas_price", "block_time"],
    "start_time": "2023-06-04T19:00:00Z",
    "end_time": "2023-06-05T19:00:00Z",
    "resolution": "hour"
  },
  "data": [
    {
      "timestamp": "2023-06-04T19:00:00Z",
      "tps": 7.8,
      "gas_price": 5.5,
      "block_time": 5050
    },
    {
      "timestamp": "2023-06-04T20:00:00Z",
      "tps": 8.2,
      "gas_price": 5.3,
      "block_time": 5025
    },
    // Additional data points...
    {
      "timestamp": "2023-06-05T18:00:00Z",
      "tps": 8.5,
      "gas_price": 5.2,
      "block_time": 5050
    }
  ]
}
```

### WebSocket Subscriptions

Real-time metrics updates are available through WebSocket connections:

```
ws://localhost:3000/api/metrics/ws
```

### Subscribe to Metrics Updates

**Subscription Message:**
```json
{
  "action": "subscribe",
  "topics": [
    "metrics:tps",
    "metrics:gas_price",
    "metrics:block_time",
    "metrics:validator_status"
  ],
  "interval_seconds": 15
}
```

**Example Messages Received:**

TPS update:
```json
{
  "topic": "metrics:tps",
  "timestamp": "2023-06-05T19:16:00Z",
  "value": 8.7,
  "change": "+0.2",
  "period": "current"
}
```

Gas price update:
```json
{
  "topic": "metrics:gas_price",
  "timestamp": "2023-06-05T19:16:00Z",
  "value": 5.1,
  "change": "-0.1",
  "period": "current"
}
```

## Error Responses

All API endpoints use standard HTTP status codes and return error details in JSON format:

```json
{
  "error": {
    "code": 400,
    "message": "Invalid parameter",
    "details": "Metric 'invalid_metric' is not a valid metric identifier"
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
const { MetricsClient } = require('artha-chain-sdk');

const metricsClient = new MetricsClient('http://localhost:3000/api');

// Get current performance metrics
metricsClient.getPerformance({ interval: '5m' })
  .then(performance => {
    console.log('Current TPS:', performance.transactions.tps.current);
    console.log('Block time (ms):', performance.blocks.time_between_blocks_ms.average);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });

// Subscribe to real-time metrics updates
metricsClient.subscribe(['metrics:tps', 'metrics:gas_price'], (update) => {
  console.log(`${update.topic} update:`, update.value);
});

// Get historical TPS data
metricsClient.getHistorical({
  metric: 'tps',
  start_time: '2023-06-04T19:00:00Z',
  end_time: '2023-06-05T19:00:00Z',
  resolution: 'hour'
}).then(data => {
  console.log('Historical TPS data:', data);
});
```
