# WebSocket API

The Artha Chain Testnet WebSocket API provides real-time updates from the blockchain, allowing developers to subscribe to events, transaction confirmations, block creations, and network status changes.

## Base URL

All WebSocket endpoints are relative to the following base URLs:
- `ws://localhost:3000/ws` (validator1)
- `ws://localhost:3001/ws` (validator2)
- `ws://localhost:3002/ws` (validator3)
- `ws://localhost:3003/ws` (validator4)

## Authentication

Some subscription channels may require authentication via an API key. The key can be provided in the initial connection request as a query parameter or in the subscription message.

## Connection

To connect to the WebSocket API, create a standard WebSocket connection to one of the base URLs:

```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:3000/ws');

ws.onopen = () => {
  console.log('Connection established');
  // Send subscription messages after connection is established
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Message received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
  console.log('Connection closed:', event.code, event.reason);
};
```

## Subscription Channels

### 1. Block Updates

Subscribe to receive notifications when new blocks are created.

**Subscription Message:**
```json
{
  "action": "subscribe",
  "channel": "blocks",
  "include_transactions": false
}
```

**Parameters:**
- `include_transactions` (boolean, optional): Whether to include full transaction data in block updates (default: false)

**Response (Confirmation):**
```json
{
  "success": true,
  "message": "Subscribed to blocks channel",
  "subscription_id": "blocks-1234"
}
```

**Event Message:**
```json
{
  "channel": "blocks",
  "event": "new_block",
  "data": {
    "height": 10250,
    "hash": "0x1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd",
    "timestamp": "2023-06-05T16:45:22Z",
    "transactions_count": 15,
    "validator": "0x9b23f56b4c0f76c46354a6d5d0766b87a8cfd9e7",
    "gas_used": 750000,
    "gas_limit": 8000000,
    "size": 24500,
    "transactions": []
  }
}
```

If `include_transactions` is set to `true`, the `transactions` array will contain transaction details.

### 2. Transaction Updates

Subscribe to receive notifications for transaction status changes.

**Subscription Message:**
```json
{
  "action": "subscribe",
  "channel": "transactions",
  "filter": {
    "addresses": ["0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6"],
    "statuses": ["pending", "confirmed", "failed"]
  }
}
```

**Parameters:**
- `filter` (object, optional):
  - `addresses` (array of strings, optional): Only receive updates for transactions involving these addresses
  - `statuses` (array of strings, optional): Only receive updates for these transaction statuses (options: "pending", "confirmed", "failed")

**Response (Confirmation):**
```json
{
  "success": true,
  "message": "Subscribed to transactions channel",
  "subscription_id": "transactions-5678"
}
```

**Event Message:**
```json
{
  "channel": "transactions",
  "event": "transaction_status_change",
  "data": {
    "hash": "0xabcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
    "status": "confirmed",
    "block_height": 10250,
    "block_hash": "0x1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd",
    "timestamp": "2023-06-05T16:45:22Z",
    "from": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
    "to": "0x7b23f56b4c0f76c46354a6d5d0766b87a8cfd9e5",
    "value": "10.5",
    "gas_used": 21000,
    "gas_price": "0.000000025",
    "nonce": 42,
    "confirmations": 1,
    "logs": []
  }
}
```

### 3. Pending Transaction Pool

Subscribe to receive notifications about the pending transaction pool.

**Subscription Message:**
```json
{
  "action": "subscribe",
  "channel": "mempool",
  "filter": {
    "addresses": ["0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6"]
  }
}
```

**Parameters:**
- `filter` (object, optional):
  - `addresses` (array of strings, optional): Only receive updates for transactions involving these addresses

**Response (Confirmation):**
```json
{
  "success": true,
  "message": "Subscribed to mempool channel",
  "subscription_id": "mempool-9012"
}
```

**Event Message:**
```json
{
  "channel": "mempool",
  "event": "transaction_added",
  "data": {
    "hash": "0xabcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
    "timestamp": "2023-06-05T16:44:52Z",
    "from": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
    "to": "0x7b23f56b4c0f76c46354a6d5d0766b87a8cfd9e5",
    "value": "10.5",
    "gas_limit": 21000,
    "gas_price": "0.000000025",
    "nonce": 42
  }
}
```

Other possible events include `transaction_removed` (when a transaction is taken out of the mempool for inclusion in a block or expiration) and `mempool_stats` (periodic updates about the overall mempool status).

### 4. Account Updates

Subscribe to receive notifications when an account's state changes.

**Subscription Message:**
```json
{
  "action": "subscribe",
  "channel": "accounts",
  "addresses": ["0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6"]
}
```

**Parameters:**
- `addresses` (array of strings): The account addresses to monitor

**Response (Confirmation):**
```json
{
  "success": true,
  "message": "Subscribed to accounts channel",
  "subscription_id": "accounts-3456"
}
```

**Event Message:**
```json
{
  "channel": "accounts",
  "event": "balance_change",
  "data": {
    "address": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
    "previous_balance": "10750.25",
    "new_balance": "10739.75",
    "change": "-10.5",
    "transaction_hash": "0xabcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
    "block_height": 10250,
    "timestamp": "2023-06-05T16:45:22Z"
  }
}
```

Other possible events include `nonce_change`, `code_change` (for contract accounts), and `storage_change`.

### 5. Contract Events

Subscribe to receive notifications about events emitted by smart contracts.

**Subscription Message:**
```json
{
  "action": "subscribe",
  "channel": "contract_events",
  "filter": {
    "addresses": ["0x5c23f56b4c0f76c46354a6d5d0766b87a8cfd9e2"],
    "event_signatures": ["Transfer(address,address,uint256)"],
    "topics": [
      "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
      "0x0000000000000000000000008a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6"
    ]
  }
}
```

**Parameters:**
- `filter` (object, optional):
  - `addresses` (array of strings, optional): Only receive events from these contract addresses
  - `event_signatures` (array of strings, optional): Only receive events matching these signatures
  - `topics` (array of strings, optional): Only receive events containing these topics

**Response (Confirmation):**
```json
{
  "success": true,
  "message": "Subscribed to contract_events channel",
  "subscription_id": "contract_events-7890"
}
```

**Event Message:**
```json
{
  "channel": "contract_events",
  "event": "contract_event",
  "data": {
    "contract_address": "0x5c23f56b4c0f76c46354a6d5d0766b87a8cfd9e2",
    "event_name": "Transfer",
    "event_signature": "Transfer(address,address,uint256)",
    "transaction_hash": "0xabcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
    "block_hash": "0x1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd",
    "block_height": 10250,
    "log_index": 2,
    "timestamp": "2023-06-05T16:45:22Z",
    "topics": [
      "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
      "0x0000000000000000000000008a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
      "0x0000000000000000000000006b23f56b4c0f76c46354a6d5d0766b87a8cfd9e4"
    ],
    "data": "0x00000000000000000000000000000000000000000000000000000000000186a0",
    "decoded_data": {
      "from": "0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6",
      "to": "0x6b23f56b4c0f76c46354a6d5d0766b87a8cfd9e4",
      "value": "100000"
    }
  }
}
```

### 6. Validator Updates

Subscribe to receive notifications about validator status changes.

**Subscription Message:**
```json
{
  "action": "subscribe",
  "channel": "validators"
}
```

**Response (Confirmation):**
```json
{
  "success": true,
  "message": "Subscribed to validators channel",
  "subscription_id": "validators-6789"
}
```

**Event Message:**
```json
{
  "channel": "validators",
  "event": "validator_set_change",
  "data": {
    "block_height": 10500,
    "timestamp": "2023-06-05T18:00:00Z",
    "added_validators": [
      {
        "address": "0xc2d3b4a5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1",
        "voting_power": 5000,
        "pubkey": "0x04b3a2c1d0f9e8d7c6b5a4d3c2b1a0f9e8d7c6b5a4d3c2b1a0f9e8d7c6b5a4d3c2b1a0f9e8d7c6b5a4d3c2b1a0f9e8d7c6b5a4d3c2b1a0f9e8d7c6b5a4d3c2b1"
      }
    ],
    "removed_validators": [],
    "updated_validators": [
      {
        "address": "0x9b23f56b4c0f76c46354a6d5d0766b87a8cfd9e7",
        "voting_power": 15000,
        "previous_voting_power": 12500,
        "pubkey": "0x04a8b53d8cfc5510e45f523342beea6584621ec04f2c2ee1578fa266a9e652a5db88e6ff0bd06eec1c1e40c7caf166a559e1f82c8264561e6377de9de82f5b07a2"
      }
    ],
    "total_validators": 8,
    "total_voting_power": 100000
  }
}
```

Other possible events include `validator_status_change` (when a validator is jailed, unjailed, or slashed) and `consensus_round_change` (updates about the consensus process).

### 7. Network Status

Subscribe to receive periodic updates about the overall network status.

**Subscription Message:**
```json
{
  "action": "subscribe",
  "channel": "network_status",
  "interval": 60
}
```

**Parameters:**
- `interval` (integer, optional): How frequently to receive updates, in seconds (default: 60, min: 10, max: 300)

**Response (Confirmation):**
```json
{
  "success": true,
  "message": "Subscribed to network_status channel",
  "subscription_id": "network_status-5432"
}
```

**Event Message:**
```json
{
  "channel": "network_status",
  "event": "status_update",
  "data": {
    "timestamp": "2023-06-05T16:46:00Z",
    "latest_block_height": 10251,
    "latest_block_time": "2023-06-05T16:45:52Z",
    "active_validators": 8,
    "total_transactions": 542897,
    "transactions_per_second": 12.5,
    "average_block_time": 5.2,
    "network_peers": 35,
    "pending_transactions": 143,
    "is_syncing": false,
    "network_version": "v1.2.0",
    "network_upgrade_planned": false
  }
}
```

## Unsubscribing

To stop receiving updates from a subscription channel, send an unsubscribe message:

**Unsubscribe Message:**
```json
{
  "action": "unsubscribe",
  "subscription_id": "blocks-1234"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Unsubscribed from blocks channel",
  "subscription_id": "blocks-1234"
}
```

## Heartbeat

The WebSocket server sends periodic heartbeat messages to keep the connection alive. Clients should respond to these messages to maintain the connection.

**Heartbeat Message (from server):**
```json
{
  "type": "heartbeat",
  "timestamp": "2023-06-05T16:46:30Z"
}
```

**Heartbeat Response (from client):**
```json
{
  "type": "heartbeat_response",
  "timestamp": "2023-06-05T16:46:30Z"
}
```

## Rate Limiting

WebSocket connections and subscriptions are subject to rate limiting. The limits are as follows:
- Maximum of 5 simultaneous WebSocket connections per IP address
- Maximum of 10 subscription channels per connection
- Maximum of 100 addresses monitored across all subscriptions
- Event delivery rate may be throttled during periods of high network activity

## Error Handling

If an error occurs with a subscription request or during connection, an error message will be sent:

**Error Message Example:**
```json
{
  "success": false,
  "error": {
    "code": "invalid_subscription",
    "message": "Invalid subscription parameters provided",
    "details": "The filter contains invalid address format"
  }
}
```

## Connection Lifecycle

1. **Establish connection** - Connect to the WebSocket endpoint
2. **Subscribe to channels** - Send subscription messages for the channels you want to monitor
3. **Process events** - Handle incoming event messages from your subscriptions
4. **Maintain connection** - Respond to heartbeat messages
5. **Unsubscribe** - Unsubscribe from channels when they're no longer needed
6. **Close connection** - Close the WebSocket connection when done

## Best Practices

1. **Reconnection Strategy**: Implement an exponential backoff strategy for reconnecting in case of disconnection
2. **Subscription Management**: Only subscribe to the channels and events you need
3. **Error Handling**: Properly handle error messages and connection issues
4. **Message Processing**: Process messages asynchronously to avoid blocking the WebSocket connection
5. **Heartbeat Monitoring**: Monitor heartbeats to detect connection issues
6. **Connection Pooling**: For applications that require high reliability, consider connecting to multiple validators

## Example Implementation

Here's a simple JavaScript example demonstrating the full lifecycle of a WebSocket connection:

```javascript
class ArthaChainWebSocket {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.subscriptions = {};
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 1000; // Start with 1 second
    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('Connected to Artha Chain WebSocket');
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      
      // Resubscribe to previous subscriptions if reconnecting
      Object.values(this.subscriptions).forEach(sub => {
        this.send(sub.request);
      });
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      // Handle heartbeat
      if (message.type === 'heartbeat') {
        this.send({
          type: 'heartbeat_response',
          timestamp: message.timestamp
        });
        return;
      }
      
      // Handle subscription confirmation
      if (message.success && message.subscription_id) {
        if (this.subscriptions[message.subscription_id]) {
          this.subscriptions[message.subscription_id].active = true;
        }
        return;
      }
      
      // Handle channel events
      if (message.channel && message.event) {
        const callback = this.subscriptions[message.channel]?.callback;
        if (callback) {
          callback(message);
        }
        return;
      }
      
      // Handle errors
      if (message.success === false) {
        console.error('WebSocket error:', message.error);
        return;
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      this.reconnect();
    };
  }

  reconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Maximum reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(30000, this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1));
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }

  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket not connected');
    }
  }

  subscribe(channel, params, callback) {
    const request = {
      action: 'subscribe',
      channel: channel,
      ...params
    };
    
    this.send(request);
    
    const subscriptionId = `${channel}-${Date.now()}`;
    this.subscriptions[subscriptionId] = {
      channel: channel,
      request: request,
      callback: callback,
      active: false
    };
    
    return subscriptionId;
  }

  unsubscribe(subscriptionId) {
    if (!this.subscriptions[subscriptionId]) {
      console.error('Subscription not found:', subscriptionId);
      return;
    }
    
    this.send({
      action: 'unsubscribe',
      subscription_id: subscriptionId
    });
    
    delete this.subscriptions[subscriptionId];
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage example
const ws = new ArthaChainWebSocket('ws://localhost:3000/ws');

// Subscribe to new blocks
ws.subscribe('blocks', { include_transactions: true }, (message) => {
  console.log('New block:', message.data);
});

// Subscribe to transaction updates for a specific address
ws.subscribe('transactions', { 
  filter: { 
    addresses: ['0x8a23f56b4c0f76c46354a6d5d0766b87a8cfd9e6'] 
  } 
}, (message) => {
  console.log('Transaction update:', message.data);
});

// Later, close the connection when done
// ws.close();
``` 