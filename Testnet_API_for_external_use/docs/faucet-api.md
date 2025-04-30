# Faucet API

The Artha Chain Testnet Faucet API allows developers to request test tokens for development and testing purposes.

## Base URL

All API endpoints are relative to the base URL of a validator node:
- `http://localhost:3000` (validator1)
- `http://localhost:3001` (validator2)
- `http://localhost:3002` (validator3)
- `http://localhost:3003` (validator4)

## Endpoints

### Request Tokens

Requests test tokens to be sent to a specified address.

**Endpoint:** `POST /api/faucet/request`

**Request:**
```json
{
  "recipient": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "amount": 1000
}
```

**Parameters:**
- `recipient` (required): The recipient address to receive tokens
- `amount` (optional): The amount of tokens to request (default: 1000, max: 10000)

**Response:**
```json
{
  "status": "success",
  "transaction_hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "amount": 1000,
  "recipient": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
}
```

### Check Request Status

Checks the status of a token request using the transaction hash from the request response.

**Endpoint:** `GET /api/faucet/status/:txHash`

**Parameters:**
- `txHash` (path parameter): The transaction hash of the faucet request

**Response:**
```json
{
  "status": "confirmed",
  "transaction_hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "block_number": 1024,
  "confirmation_time": "2023-06-01T12:34:56Z"
}
```

Status can be: "pending", "confirmed", "failed"

### Get Faucet Balance

Retrieves the current balance of the faucet account.

**Endpoint:** `GET /api/faucet/balance`

**Response:**
```json
{
  "address": "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654",
  "balance": 10000000,
  "daily_distribution": 250000,
  "remaining_daily_allowance": 150000
}
```

### Get Request History

Retrieves recent token request history for monitoring or administration.

**Endpoint:** `GET /api/faucet/history`

**Query Parameters:**
- `limit` (optional): Maximum number of records to return (default: 10, max: 100)
- `offset` (optional): Number of records to skip (default: 0)

**Response:**
```json
{
  "requests": [
    {
      "transaction_hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "recipient": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "amount": 1000,
      "timestamp": "2023-06-01T12:30:45Z",
      "status": "confirmed"
    },
    {
      "transaction_hash": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27e0c7f0f9e5b5a14a3bcd7f",
      "recipient": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "amount": 2000,
      "timestamp": "2023-06-01T12:15:32Z",
      "status": "confirmed"
    }
  ],
  "total": 245,
  "limit": 10,
  "offset": 0
}
```

## Error Responses

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_ADDRESS | The recipient address is invalid |
| 400 | INVALID_AMOUNT | The requested amount is invalid or exceeds maximum |
| 404 | TRANSACTION_NOT_FOUND | The specified transaction hash does not exist |
| 429 | RATE_LIMIT_EXCEEDED | The request exceeds the rate limit (see rate limits below) |
| 503 | FAUCET_DEPLETED | The faucet has insufficient funds |

### Error Response Example:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again in 3540 seconds.",
    "details": {
      "retry_after": 3540,
      "request_count": 3,
      "max_requests": 3,
      "time_window": "24h"
    }
  }
}
```

## Rate Limits

To prevent abuse, the following rate limits apply:

- Maximum 5 requests per IP address per day
- Maximum 3 requests per recipient address per day
- Minimum 1 hour between requests for the same recipient
- Maximum request amount of 10,000 tokens per transaction

## Implementation Notes

- The Faucet API automatically creates and broadcasts a transaction on the blockchain
- Requests are processed asynchronously; use the status endpoint to verify completion
- The faucet address is determined by the node configuration and may vary between deployments 