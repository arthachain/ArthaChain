# Wallet API

The Artha Chain Testnet Wallet API allows users to manage their blockchain wallets, including creating addresses, checking balances, and sending transactions.

## Base URL

All API endpoints are relative to the base URL of a validator node:
- `http://localhost:3000` (validator1)
- `http://localhost:3001` (validator2)
- `http://localhost:3002` (validator3)
- `http://localhost:3003` (validator4)

## Authentication

The Wallet API requires authentication for most endpoints. See the [Authentication](./authentication.md) documentation for details.

## Endpoints

### Create Wallet

Creates a new wallet with a generated key pair.

**Endpoint:** `POST /api/wallet/create`

**Response:**
```json
{
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "public_key": "0x7c2a65fc3d8a15ef7a49d512ab8324fa9e903c3f707a1bd30c76542a91c2a36c",
  "private_key": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5"
}
```

**Security Note:** The private key is only returned once during creation. Store it securely.

### Import Wallet

Imports an existing wallet using a private key.

**Endpoint:** `POST /api/wallet/import`

**Request:**
```json
{
  "private_key": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5"
}
```

**Response:**
```json
{
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "public_key": "0x7c2a65fc3d8a15ef7a49d512ab8324fa9e903c3f707a1bd30c76542a91c2a36c",
  "imported": true
}
```

### Get Wallet Balance

Retrieves the current balance for a specific wallet address.

**Endpoint:** `GET /api/wallet/balance/:address`

**Parameters:**
- `address` (path parameter): The wallet address to query

**Response:**
```json
{
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "balance": 10000,
  "pending_balance": 500,
  "last_updated": "2023-06-01T12:34:56Z"
}
```

### List User Wallets

Retrieves all wallets associated with the authenticated user.

**Endpoint:** `GET /api/wallet/list`

**Response:**
```json
{
  "wallets": [
    {
      "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "label": "Main Wallet",
      "balance": 10000
    },
    {
      "address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "label": "Test Wallet",
      "balance": 5000
    }
  ],
  "total": 2
}
```

### Send Transaction

Sends tokens from a wallet to another address.

**Endpoint:** `POST /api/wallet/send`

**Request:**
```json
{
  "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "to": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "amount": 100,
  "gas_price": 1,
  "gas_limit": 21000,
  "data": "0x",
  "private_key": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5"
}
```

**Parameters:**
- `from` (required): Sender's wallet address
- `to` (required): Recipient's wallet address
- `amount` (required): Amount of tokens to send
- `gas_price` (optional): Price per unit of gas (default: network standard)
- `gas_limit` (optional): Maximum gas units (default: 21000)
- `data` (optional): Additional data to include (hex-encoded)
- `private_key` (required): Private key of the sender's wallet

**Response:**
```json
{
  "transaction_hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "status": "pending",
  "estimated_confirmation_time": "30 seconds"
}
```

### Get Transaction History

Retrieves transaction history for a specific wallet address.

**Endpoint:** `GET /api/wallet/transactions/:address`

**Parameters:**
- `address` (path parameter): The wallet address to query
- `limit` (query parameter, optional): Maximum number of transactions to return (default: 10, max: 100)
- `offset` (query parameter, optional): Number of transactions to skip (default: 0)
- `filter` (query parameter, optional): Filter transactions by type (all, sent, received)

**Response:**
```json
{
  "transactions": [
    {
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "to": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "amount": 100,
      "fee": 21,
      "timestamp": "2023-06-01T12:30:45Z",
      "status": "confirmed",
      "block_number": 1024,
      "type": "sent"
    },
    {
      "hash": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27e0c7f0f9e5b5a14a3bcd7f",
      "from": "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654",
      "to": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "amount": 500,
      "fee": 21,
      "timestamp": "2023-06-01T10:15:32Z",
      "status": "confirmed",
      "block_number": 950,
      "type": "received"
    }
  ],
  "total": 45,
  "limit": 10,
  "offset": 0
}
```

### Create Multi-Signature Wallet

Creates a multi-signature wallet requiring multiple approvals for transactions.

**Endpoint:** `POST /api/wallet/multisig/create`

**Request:**
```json
{
  "required_signatures": 2,
  "owners": [
    "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
    "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
    "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654"
  ],
  "label": "Treasury Wallet"
}
```

**Response:**
```json
{
  "address": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27",
  "required_signatures": 2,
  "owners": [
    "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
    "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
    "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654"
  ],
  "label": "Treasury Wallet",
  "creation_transaction": "0x5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9"
}
```

### Export Wallet

Exports wallet information for backup purposes.

**Endpoint:** `POST /api/wallet/export`

**Request:**
```json
{
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "private_key": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5"
}
```

**Response:**
```json
{
  "keystore_file": "eyJhZGRyZXNzIjoiMHhmMzE3Yzc0YzUzNmJkZWJmZmM0NGE5M2QzM2VhZGJjODFhNTlhNGQ5M2U3IiwiaWQiOiI4NzFhNDVjYi03YzE3ZjM4MTJhMzYiLCJ2ZXJzaW9uIjozLCJjaXBoZXIiOiJhZXMtMTI4LWN0ciIsImNpcGhlcnRleHQiOiI3ZjJjNDVmZDhjMzRhMWQ3OGU0YWExMjNmNTE3ZTRkNmFlMzFkMWRjODEyYTRiM2FlZjE1Y2EyYmM5N2RhMTUzIiwiY2lwaGVycGFyYW1zIjp7Iml2IjoiM2Q2YTQ1Y2I3YzE3ZjM4MTJhMzZmYzE1MjRjYTI4OTAifSwibWFjIjoiZjljZTM3MTIwYjE1OTI4YzViZDlmOTVjNWYwYWJhZjliNmJjMWYzNTNkNGYzMmYwMWM3NjVjYjkzMmEwODczYSJ9",
  "mnemonic": "abandon ability able about above absent absorb abstract absurd abuse access accident",
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
}
```

## Error Responses

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_ADDRESS | The specified address is invalid |
| 400 | INVALID_PRIVATE_KEY | The private key format is invalid |
| 400 | INSUFFICIENT_FUNDS | The wallet has insufficient funds for the transaction |
| 401 | UNAUTHORIZED | Authentication required or failed |
| 403 | FORBIDDEN | The authenticated user does not own this wallet |
| 404 | WALLET_NOT_FOUND | The specified wallet does not exist |
| 500 | TRANSACTION_FAILED | The transaction creation failed |

### Error Response Example:

```json
{
  "error": {
    "code": "INSUFFICIENT_FUNDS",
    "message": "Insufficient funds for transaction",
    "details": {
      "required": 150,
      "available": 100,
      "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
    }
  }
}
```

## Implementation Notes

- All wallet operations requiring private keys should be performed client-side when possible
- For server-side operations (like wallet creation), private keys are never stored and are returned only once
- Transactions are broadcast to all validator nodes to ensure rapid propagation
- Wallet data is cached for performance, but always verify balances on-chain for critical operations 