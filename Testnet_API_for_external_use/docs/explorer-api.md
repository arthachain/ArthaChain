# Explorer API

The Artha Chain Testnet Explorer API provides comprehensive access to blockchain data including blocks, transactions, accounts, and network statistics.

## Base URL

All API endpoints are relative to the base URL of a validator node:
- `http://localhost:3000` (validator1)
- `http://localhost:3001` (validator2)
- `http://localhost:3002` (validator3)
- `http://localhost:3003` (validator4)

## Blocks Endpoints

### Get Latest Blocks

Retrieves the most recent blocks on the blockchain.

**Endpoint:** `GET /api/explorer/blocks`

**Query Parameters:**
- `limit` (optional): Maximum number of blocks to return (default: 10, max: 100)
- `offset` (optional): Number of blocks to skip (default: 0)

**Response:**
```json
{
  "blocks": [
    {
      "hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "height": 1024,
      "timestamp": 1650326472,
      "transaction_count": 12,
      "validator": "validator1",
      "size": 5612
    },
    {
      "hash": "0xe0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93a2",
      "height": 1023,
      "timestamp": 1650326442,
      "transaction_count": 8,
      "validator": "validator2",
      "size": 3845
    }
  ],
  "total": 1025,
  "limit": 10,
  "offset": 0
}
```

### Get Block Details

Retrieves detailed information about a specific block.

**Endpoint:** `GET /api/explorer/blocks/:identifier`

**Parameters:**
- `identifier` (path parameter): Block hash or block height

**Response:**
```json
{
  "hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "height": 1024,
  "timestamp": 1650326472,
  "previous_hash": "0xe0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93a2",
  "merkle_root": "0xd5e041084eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9",
  "validator": "validator1",
  "validator_address": "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654",
  "signature": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5",
  "transaction_count": 12,
  "size": 5612,
  "gas_used": 250000,
  "gas_limit": 1000000,
  "transactions": [
    {
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "to": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "amount": 100,
      "gas_used": 21000
    }
    // Additional transactions...
  ]
}
```

## Transaction Endpoints

### Get Transactions

Retrieves recent transactions across the blockchain.

**Endpoint:** `GET /api/explorer/transactions`

**Query Parameters:**
- `limit` (optional): Maximum number of transactions to return (default: 10, max: 100)
- `offset` (optional): Number of transactions to skip (default: 0)
- `filter` (optional): Filter by transaction type (all, transfer, contract_call, contract_creation)

**Response:**
```json
{
  "transactions": [
    {
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "to": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "amount": 100,
      "timestamp": 1650326400,
      "block_hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "block_height": 1024,
      "type": "transfer"
    },
    {
      "hash": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27e0c7f0f9e5b5a14a3bcd7f",
      "from": "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654",
      "to": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "amount": 500,
      "timestamp": 1650326350,
      "block_hash": "0xe0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93a2",
      "block_height": 1023,
      "type": "transfer"
    }
  ],
  "total": 5642,
  "limit": 10,
  "offset": 0
}
```

### Get Transaction Details

Retrieves detailed information about a specific transaction.

**Endpoint:** `GET /api/explorer/transactions/:hash`

**Parameters:**
- `hash` (path parameter): Transaction hash

**Response:**
```json
{
  "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "to": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "amount": 100,
  "fee": 21,
  "timestamp": 1650326400,
  "data": "0x",
  "nonce": 7,
  "signature": "0x7c2a65fc3d8a15ef7a49d512ab8324fa9e903c3f707a1bd30c76542a91c2a36c",
  "gas_price": 1,
  "gas_limit": 21000,
  "gas_used": 21000,
  "status": "confirmed",
  "block_hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "block_height": 1024,
  "type": "transfer",
  "confirmations": 10
}
```

## Account Endpoints

### Get Account Details

Retrieves detailed information about a specific account.

**Endpoint:** `GET /api/explorer/accounts/:address`

**Parameters:**
- `address` (path parameter): Account address

**Response:**
```json
{
  "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "balance": 9950,
  "nonce": 8,
  "type": "user",
  "first_active": 950,
  "last_active": 1024,
  "transaction_count": 25,
  "tokens_sent": 1500,
  "tokens_received": 11450,
  "is_contract": false
}
```

### Get Account Transactions

Retrieves transactions associated with a specific account.

**Endpoint:** `GET /api/explorer/accounts/:address/transactions`

**Parameters:**
- `address` (path parameter): Account address
- `limit` (query parameter, optional): Maximum number of transactions to return (default: 10, max: 100)
- `offset` (query parameter, optional): Number of transactions to skip (default: 0)
- `type` (query parameter, optional): Transaction type filter (all, sent, received)

**Response:**
```json
{
  "transactions": [
    {
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "to": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "amount": 100,
      "timestamp": 1650326400,
      "block_height": 1024,
      "type": "sent"
    },
    {
      "hash": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27e0c7f0f9e5b5a14a3bcd7f",
      "from": "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654",
      "to": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "amount": 500,
      "timestamp": 1650326350,
      "block_height": 1023,
      "type": "received"
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

## Contracts Endpoints

### Get Smart Contract Details

Retrieves detailed information about a deployed smart contract.

**Endpoint:** `GET /api/explorer/contracts/:address`

**Parameters:**
- `address` (path parameter): Contract address

**Response:**
```json
{
  "address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "creator": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "creation_transaction": "0x5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9",
  "creation_block": 950,
  "balance": 1200,
  "transaction_count": 42,
  "code_size": 2456,
  "verified": true,
  "abi": [
    {
      "inputs": [{"name": "recipient", "type": "address"}, {"name": "amount", "type": "uint256"}],
      "name": "transfer",
      "outputs": [{"name": "success", "type": "bool"}],
      "stateMutability": "nonpayable",
      "type": "function"
    }
    // Additional ABI entries...
  ],
  "source_code": "contract Token {\n  function transfer(address recipient, uint256 amount) public returns (bool) {\n    // Implementation\n  }\n}"
}
```

### Get Contract Transactions

Retrieves transactions associated with a smart contract.

**Endpoint:** `GET /api/explorer/contracts/:address/transactions`

**Parameters:**
- `address` (path parameter): Contract address
- `limit` (query parameter, optional): Maximum number of transactions to return (default: 10, max: 100)
- `offset` (query parameter, optional): Number of transactions to skip (default: 0)

**Response:** Same format as account transactions

## Statistics Endpoints

### Get Blockchain Statistics

Retrieves general statistics about the blockchain.

**Endpoint:** `GET /api/explorer/stats`

**Response:**
```json
{
  "block_height": 1024,
  "total_transactions": 5642,
  "average_block_time": 30.2,
  "average_transactions_per_block": 12.5,
  "active_accounts": 156,
  "total_accounts": 245,
  "current_tps": 8.5,
  "average_tps_24h": 7.3,
  "peak_tps": 15.7,
  "contract_count": 27,
  "validators": 4
}
```

### Get Rich List

Retrieves accounts with the highest balances.

**Endpoint:** `GET /api/explorer/accounts/rich-list`

**Query Parameters:**
- `limit` (optional): Maximum number of accounts to return (default: 10, max: 100)
- `offset` (optional): Number of accounts to skip (default: 0)

**Response:**
```json
{
  "accounts": [
    {
      "address": "0x8a23fc45d6712ab8324fa9e903c3f707a1bd30c7654",
      "balance": 1000000,
      "percentage": 10.5,
      "type": "validator"
    },
    {
      "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "balance": 500000,
      "percentage": 5.2,
      "type": "user"
    }
    // Additional accounts...
  ],
  "total": 245,
  "limit": 10,
  "offset": 0,
  "total_supply": 10000000
}
```

## Search

### Search Blockchain

Searches for blocks, transactions, accounts, or contracts by keyword or identifier.

**Endpoint:** `GET /api/explorer/search`

**Query Parameters:**
- `query` (required): Search query (block hash/height, transaction hash, address)

**Response:**
```json
{
  "results": [
    {
      "type": "block",
      "height": 1024,
      "hash": "0x7f9c9456dc9af68229eb5f1d6c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "timestamp": 1650326472
    },
    {
      "type": "transaction",
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "block_height": 1024,
      "timestamp": 1650326400
    },
    {
      "type": "account",
      "address": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
      "balance": 9950
    }
  ],
  "total": 3
}
```

## Error Responses

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_QUERY | The search query is invalid |
| 400 | INVALID_FILTER | The filter value is not supported |
| 404 | BLOCK_NOT_FOUND | The specified block does not exist |
| 404 | TRANSACTION_NOT_FOUND | The specified transaction does not exist |
| 404 | ACCOUNT_NOT_FOUND | The specified account does not exist |
| 404 | CONTRACT_NOT_FOUND | The specified contract does not exist |
| 500 | SERVER_ERROR | An internal server error occurred |

### Error Response Example:

```json
{
  "error": {
    "code": "TRANSACTION_NOT_FOUND",
    "message": "The requested transaction hash does not exist",
    "details": {
      "hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a"
    }
  }
}
```

## Implementation Notes

- For performance reasons, some Explorer API endpoints may return cached data with a short delay (approximately 5-10 seconds behind the actual blockchain state)
- Full transaction details are only available for confirmed transactions
- The Explorer API supports cross-origin resource sharing (CORS) for browser-based applications
- Rate limits apply to prevent excessive API usage (see [Rate Limiting](./rate-limiting.md)) 