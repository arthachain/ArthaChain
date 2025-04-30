# Transaction API

The Transaction API provides a comprehensive interface for creating, submitting, and querying transactions on the Artha Chain Testnet. This API enables developers to interact with the blockchain by sending value transfers, calling smart contracts, deploying new contracts, and monitoring transaction status.

## Base URL

Each validator node exposes the Transaction API at the following base URLs:
- `http://localhost:3000/api/transactions` (validator1)
- `http://localhost:3001/api/transactions` (validator2)
- `http://localhost:3002/api/transactions` (validator3)
- `http://localhost:3003/api/transactions` (validator4)

## Authentication

Authentication is required for transaction submission endpoints:

```
Authorization: Bearer your_api_key_here
```

API keys can be obtained from the [Artha Chain Testnet Portal](http://localhost:3000/portal).

## Endpoints

### Transaction Information

#### GET /api/transactions/{hash}

Retrieves detailed information about a specific transaction.

**Parameters:**
- `hash`: The transaction hash (required)

**Example Response:**
```json
{
  "hash": "0x7d2a32cb629d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3ea6",
  "block_hash": "0xd92a384f3b7bbe1b5cb7dbb73b5ca6bb459aee425d23f56b4c0f76c46354a6d5",
  "block_number": 128956,
  "timestamp": "2023-06-05T19:15:25Z",
  "from": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "to": "0x7d23f56b4c0f76c46354a6d5d0766b87a8cfd9e9",
  "value": "5.25",
  "gas_limit": 21000,
  "gas_price": "0.00000005",
  "gas_used": 21000,
  "nonce": 42,
  "transaction_index": 12,
  "input_data": "0x",
  "status": "success",
  "receipt": {
    "cumulative_gas_used": 1250000,
    "logs": [],
 