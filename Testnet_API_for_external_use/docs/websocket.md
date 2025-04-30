# WebSocket API

The Artha Chain testnet provides a WebSocket API for real-time updates about blockchain events. This allows you to receive notifications about new blocks, transactions, and other events without polling the REST API.

## Connection

WebSocket connections are established at:
```
ws://localhost:3000/api/ws
```

(Replace `3000` with the appropriate port number for your target validator node)

## Subscription

After establishing a connection, you must subscribe to specific event types:

```json
{
  "action": "subscribe",
  "topics": ["blocks", "transactions"]
}
```

Available subscription topics:
- `blocks`: Receive notifications when new blocks are mined
- `transactions`: Receive notifications for all new transactions
- `transactions:{address}`: Receive notifications for transactions involving a specific address
- `status`: Receive periodic network status updates

You can subscribe to multiple topics in a single request.

## Message Format

### Subscription Confirmation

After subscribing, you'll receive a confirmation message:

```json
{
  "type": "subscription",
  "status": "success",
  "topics": ["blocks", "transactions"]
}
```

### Block Event

When a new block is mined:

```json
{
  "type": "block",
  "data": {
    "hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
    "height": 1024,
    "timestamp": 1650326472,
    "previous_hash": "0xe0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93a2",
    "transactions_count": 12,
    "validator": "validator1"
  }
}
```

### Transaction Event

When a new transaction is added to the blockchain:

```json
{
  "type": "transaction",
  "data": {
    "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
    "sender": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
    "recipient": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
    "amount": 100,
    "type": "transfer",
    "status": "confirmed",
    "block_height": 1024
  }
}
```

### Status Event

Periodic network status updates:

```json
{
  "type": "status",
  "data": {
    "latest_block_height": 1024,
    "tps": 15.7,
    "peers_count": 3,
    "pending_transactions": 2
  }
}
```

## Unsubscribe

To stop receiving updates for specific topics:

```json
{
  "action": "unsubscribe",
  "topics": ["transactions"]
}
```

## Close Connection

To close the WebSocket connection, simply close the socket from your client application.

## Error Handling

If an error occurs with your subscription or connection, you'll receive an error message:

```json
{
  "type": "error",
  "code": 1001,
  "message": "Invalid subscription topic"
}
```

Common error codes:
- `1001`: Invalid subscription topic
- `1002`: Invalid message format
- `1003`: Subscription limit exceeded
- `1004`: Authentication failed
- `1005`: Connection timeout

## Example Usage

### JavaScript (Browser)

```javascript
const ws = new WebSocket('ws://localhost:3000/api/ws');

ws.onopen = function() {
  console.log('Connection established');
  
  // Subscribe to topics
  ws.send(JSON.stringify({
    action: 'subscribe',
    topics: ['blocks', 'transactions']
  }));
};

ws.onmessage = function(event) {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'block':
      console.log('New block:', message.data);
      break;
    case 'transaction':
      console.log('New transaction:', message.data);
      break;
    case 'subscription':
      console.log('Subscription status:', message.status);
      break;
    case 'error':
      console.error('WebSocket error:', message.message);
      break;
  }
};

ws.onclose = function() {
  console.log('Connection closed');
};
```

### Python

```python
import websocket
import json
import threading
import time

def on_message(ws, message):
    data = json.loads(message)
    
    if data['type'] == 'block':
        print(f"New block at height {data['data']['height']}")
    elif data['type'] == 'transaction':
        print(f"New transaction: {data['data']['hash']}")
    elif data['type'] == 'subscription':
        print(f"Subscription: {data['status']}")
    elif data['type'] == 'error':
        print(f"Error: {data['message']}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection established")
    
    # Subscribe to topics
    subscription = {
        "action": "subscribe",
        "topics": ["blocks", "transactions"]
    }
    ws.send(json.dumps(subscription))

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:3000/api/ws",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever()
``` 