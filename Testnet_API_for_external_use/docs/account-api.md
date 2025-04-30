# Account API

The Account API provides comprehensive access to account information, balances, transactions, and related data on the Artha Chain Testnet. This API allows developers to query account states, token balances, transaction history, and perform various account-related operations.

## Base URL

Each validator node exposes the Account API at the following base URLs:
- `http://localhost:3000/api/accounts` (validator1)
- `http://localhost:3001/api/accounts` (validator2)
- `http://localhost:3002/api/accounts` (validator3)
- `http://localhost:3003/api/accounts` (validator4)

## Authentication

Most account endpoints are public and don't require authentication. Endpoints that modify account data or create new accounts require authentication:

```
Authorization: Bearer your_api_key_here
```

API keys can be obtained from the [Artha Chain Testnet Portal](http://localhost:3000/portal).

## Endpoints

### Account Information

#### GET /api/accounts/{address}

Retrieves comprehensive information about an account.

**Parameters:**
- `address`: The account address (required)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "balance": "1258.75",
  "nonce": 42,
  "type": "eoa", // "eoa" (Externally Owned Account) or "contract" 
  "code_hash": null,
  "storage_hash": null,
  "first_seen_at": "2023-05-10T12:15:25Z",
  "last_activity_at": "2023-06-05T19:15:25Z",
  "transaction_count": 128,
  "contract_creation_tx": null
}
```

For contract accounts, additional fields are included:

```json
{
  "address": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
  "balance": "0.5",
  "nonce": 1,
  "type": "contract",
  "code_hash": "0xd92a384f3b7bbe1b5cb7dbb73b5ca6bb459aee425d23f56b4c0f76c46354a6d5",
  "storage_hash": "0xe82a394f3b7bbe1b5cb7dbb73b5ca6bb459aee425d23f56b4c0f76c46354a7c6",
  "first_seen_at": "2023-05-15T14:25:35Z",
  "last_activity_at": "2023-06-04T17:12:15Z",
  "transaction_count": 512,
  "contract_creation_tx": "0x9c4f53db518d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3fd6",
  "contract_metadata": {
    "name": "Token Contract",
    "verified": true,
    "compiler_version": "0.8.19",
    "verification_date": "2023-05-15T15:30:00Z"
  }
}
```

#### GET /api/accounts/{address}/balance

Retrieves the native token balance of an account.

**Parameters:**
- `address`: The account address (required)
- `block`: Optional block number or hash to query historic balance (optional)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "balance": "1258.75",
  "balance_formatted": "1,258.75 ARTHA",
  "timestamp": "2023-06-05T19:30:00Z",
  "block_number": 128970
}
```

### Token Balances

#### GET /api/accounts/{address}/tokens

Retrieves all token balances for an account.

**Parameters:**
- `address`: The account address (required)
- `min_value`: Minimum value to include (optional, default: 0)
- `include_metadata`: Include token metadata (optional, default: true)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "native_balance": "1258.75",
  "token_balances": [
    {
      "token_address": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
      "token_type": "erc20",
      "balance": "1250.75",
      "metadata": {
        "name": "Test Token",
        "symbol": "TST",
        "decimals": 18,
        "logo_url": "https://testnet.artha.network/token-logos/tst.png"
      }
    },
    {
      "token_address": "0x6b2c843d7a9b8bc80f602b8d9e8e955a",
      "token_type": "erc721",
      "balance": "5",
      "token_ids": [
        "12", "45", "78", "102", "155"
      ],
      "metadata": {
        "name": "Test NFT Collection",
        "symbol": "TNFT",
        "logo_url": "https://testnet.artha.network/token-logos/tnft.png"
      }
    }
  ],
  "total_token_count": 2,
  "last_updated": "2023-06-05T19:30:00Z"
}
```

#### GET /api/accounts/{address}/tokens/{token_address}

Retrieves balance and metadata for a specific token.

**Parameters:**
- `address`: The account address (required)
- `token_address`: The token contract address (required)

**Example Response (ERC20):**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "token_address": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
  "token_type": "erc20",
  "balance": "1250.75",
  "balance_formatted": "1,250.75 TST",
  "metadata": {
    "name": "Test Token",
    "symbol": "TST",
    "decimals": 18,
    "total_supply": "1000000",
    "contract_deployment_date": "2023-05-15T14:25:35Z",
    "logo_url": "https://testnet.artha.network/token-logos/tst.png",
    "holder_count": 128
  },
  "last_updated": "2023-06-05T19:30:00Z"
}
```

**Example Response (ERC721):**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "token_address": "0x6b2c843d7a9b8bc80f602b8d9e8e955a",
  "token_type": "erc721",
  "balance": "5",
  "token_ids": [
    {
      "id": "12",
      "metadata_uri": "https://testnet.artha.network/nft-metadata/12.json",
      "metadata": {
        "name": "NFT #12",
        "description": "A test NFT",
        "image": "https://testnet.artha.network/nft-images/12.png",
        "attributes": [
          {"trait_type": "Color", "value": "Blue"},
          {"trait_type": "Shape", "value": "Circle"}
        ]
      }
    },
    {
      "id": "45",
      "metadata_uri": "https://testnet.artha.network/nft-metadata/45.json",
      "metadata": {
        "name": "NFT #45",
        "description": "A test NFT",
        "image": "https://testnet.artha.network/nft-images/45.png",
        "attributes": [
          {"trait_type": "Color", "value": "Red"},
          {"trait_type": "Shape", "value": "Square"}
        ]
      }
    }
    // Additional token IDs...
  ],
  "metadata": {
    "name": "Test NFT Collection",
    "symbol": "TNFT",
    "total_supply": "200",
    "contract_deployment_date": "2023-05-18T16:35:45Z",
    "logo_url": "https://testnet.artha.network/token-logos/tnft.png",
    "holder_count": 45
  },
  "last_updated": "2023-06-05T19:30:00Z"
}
```

### Transactions

#### GET /api/accounts/{address}/transactions

Retrieves transactions associated with an account.

**Parameters:**
- `address`: The account address (required)
- `start_block`: Starting block for the query (optional)
- `end_block`: Ending block for the query (optional)
- `from_date`: Start date in ISO 8601 format (optional)
- `to_date`: End date in ISO 8601 format (optional)
- `type`: Filter by transaction type (`all`, `sent`, `received`, `contract_calls`) (optional, default: `all`)
- `status`: Filter by status (`all`, `success`, `failed`, `pending`) (optional, default: `all`)
- `limit`: Maximum number of transactions to return (optional, default: 20, max: 100)
- `offset`: Pagination offset (optional, default: 0)
- `sort`: Sort order (`asc` or `desc` by block_number) (optional, default: `desc`)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "transactions": [
    {
      "hash": "0x7d2a32cb629d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3ea6",
      "block_number": 128956,
      "timestamp": "2023-06-05T19:15:25Z",
      "from": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
      "to": "0x7d23f56b4c0f76c46354a6d5d0766b87a8cfd9e9",
      "value": "5.25",
      "gas_used": 21000,
      "gas_price": "0.00000005",
      "fee": "0.00105",
      "status": "success",
      "direction": "outgoing",
      "type": "transfer"
    },
    {
      "hash": "0x9c4f53db518d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3fd6",
      "block_number": 128950,
      "timestamp": "2023-06-05T19:12:15Z",
      "from": "0x7d23f56b4c0f76c46354a6d5d0766b87a8cfd9e9",
      "to": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
      "value": "10.5",
      "gas_used": 21000,
      "gas_price": "0.00000005",
      "fee": "0.00105",
      "status": "success",
      "direction": "incoming",
      "type": "transfer"
    }
    // Additional transactions...
  ],
  "pagination": {
    "total": 128,
    "limit": 20,
    "offset": 0,
    "has_more": true
  },
  "summary": {
    "total_sent": "752.25",
    "total_received": "2015.5",
    "net_flow": "1263.25"
  }
}
```

#### GET /api/accounts/{address}/pending-transactions

Retrieves pending transactions associated with an account.

**Parameters:**
- `address`: The account address (required)
- `limit`: Maximum number of transactions to return (optional, default: 20, max: 100)
- `offset`: Pagination offset (optional, default: 0)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "pending_transactions": [
    {
      "hash": "0x5b3a42cb629d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3ec7",
      "timestamp": "2023-06-05T19:16:15Z",
      "from": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
      "to": "0x9b23f56b4c0f76c46354a6d5d0766b87a8cfd9e7",
      "value": "2.75",
      "gas_limit": 21000,
      "gas_price": "0.00000006",
      "nonce": 85,
      "age_seconds": 12.5,
      "type": "transfer",
      "direction": "outgoing",
      "estimated_confirmation_time": "2023-06-05T19:16:30Z"
    }
    // Additional pending transactions...
  ],
  "pagination": {
    "total": 2,
    "limit": 20,
    "offset": 0,
    "has_more": false
  }
}
```

### Internal Transactions

#### GET /api/accounts/{address}/internal-transactions

Retrieves internal transactions (transactions created by smart contracts) associated with an account.

**Parameters:**
- `address`: The account address (required)
- `from_block`: Starting block for the query (optional)
- `to_block`: Ending block for the query (optional)
- `from_date`: Start date in ISO 8601 format (optional)
- `to_date`: End date in ISO 8601 format (optional)
- `limit`: Maximum number of internal transactions to return (optional, default: 20, max: 100)
- `offset`: Pagination offset (optional, default: 0)
- `sort`: Sort order (`asc` or `desc` by block_number) (optional, default: `desc`)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "internal_transactions": [
    {
      "parent_tx_hash": "0x6c1a21db518d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3e95",
      "block_number": 128954,
      "timestamp": "2023-06-05T19:14:55Z",
      "from": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
      "to": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
      "value": "1.5",
      "type": "call",
      "trace_address": "0,2",
      "direction": "incoming"
    },
    {
      "parent_tx_hash": "0x5d2b42db518d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3f25",
      "block_number": 128952,
      "timestamp": "2023-06-05T19:13:45Z",
      "from": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
      "to": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
      "value": "0.75",
      "type": "call",
      "trace_address": "0,1,3",
      "direction": "incoming"
    }
    // Additional internal transactions...
  ],
  "pagination": {
    "total": 15,
    "limit": 20,
    "offset": 0,
    "has_more": false
  }
}
```

### Contract Interactions

#### GET /api/accounts/{address}/contract-interactions

Retrieves interactions with smart contracts for an account.

**Parameters:**
- `address`: The account address (required)
- `contract_address`: Filter by specific contract address (optional)
- `method_signature`: Filter by method signature (optional)
- `from_block`: Starting block for the query (optional)
- `to_block`: Ending block for the query (optional)
- `from_date`: Start date in ISO 8601 format (optional)
- `to_date`: End date in ISO 8601 format (optional)
- `limit`: Maximum number of interactions to return (optional, default: 20, max: 100)
- `offset`: Pagination offset (optional, default: 0)
- `sort`: Sort order (`asc` or `desc` by block_number) (optional, default: `desc`)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "contract_interactions": [
    {
      "tx_hash": "0x6c1a21db518d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3e95",
      "block_number": 128954,
      "timestamp": "2023-06-05T19:14:55Z",
      "contract_address": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
      "contract_name": "Test Token",
      "method_signature": "0xa9059cbb",
      "method_name": "transfer",
      "decoded_parameters": {
        "_to": "0x7d23f56b4c0f76c46354a6d5d0766b87a8cfd9e9",
        "_value": "100.5"
      },
      "status": "success",
      "gas_used": 51253,
      "fee": "0.00256265"
    }
    // Additional contract interactions...
  ],
  "pagination": {
    "total": 42,
    "limit": 20,
    "offset": 0,
    "has_more": true
  },
  "contract_summary": {
    "total_contracts_interacted": 5,
    "most_interacted_contract": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
    "most_used_method": "transfer"
  }
}
```

### Event Logs

#### GET /api/accounts/{address}/event-logs

Retrieves event logs emitted from smart contracts and associated with an account.

**Parameters:**
- `address`: The account address (required)
- `contract_address`: Filter by specific contract address (optional)
- `topic0`: Filter by first topic (event signature) (optional)
- `from_block`: Starting block for the query (optional)
- `to_block`: Ending block for the query (optional)
- `from_date`: Start date in ISO 8601 format (optional)
- `to_date`: End date in ISO 8601 format (optional)
- `limit`: Maximum number of events to return (optional, default: 20, max: 100)
- `offset`: Pagination offset (optional, default: 0)
- `sort`: Sort order (`asc` or `desc` by block_number) (optional, default: `desc`)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "event_logs": [
    {
      "tx_hash": "0x6c1a21db518d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3e95",
      "block_number": 128954,
      "timestamp": "2023-06-05T19:14:55Z",
      "contract_address": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
      "contract_name": "Test Token",
      "log_index": 2,
      "event_signature": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
      "event_name": "Transfer",
      "topics": [
        "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
        "0x0000000000000000000000008c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
        "0x0000000000000000000000007d23f56b4c0f76c46354a6d5d0766b87a8cfd9e9"
      ],
      "data": "0x00000000000000000000000000000000000000000000056bc75e2d63100000",
      "decoded_data": {
        "from": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
        "to": "0x7d23f56b4c0f76c46354a6d5d0766b87a8cfd9e9",
        "value": "100.5"
      }
    }
    // Additional event logs...
  ],
  "pagination": {
    "total": 85,
    "limit": 20,
    "offset": 0,
    "has_more": true
  }
}
```

### Account Analytics

#### GET /api/accounts/{address}/analytics

Retrieves analytics and statistics for an account.

**Parameters:**
- `address`: The account address (required)
- `period`: Time period for analytics (`day`, `week`, `month`, `year`, `all`) (optional, default: `month`)
- `include_token_transfers`: Include token transfer analytics (optional, default: true)
- `include_contract_interactions`: Include contract interaction analytics (optional, default: true)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "period": "month",
  "transaction_analytics": {
    "total_transactions": 128,
    "successful_transactions": 125,
    "failed_transactions": 3,
    "avg_transactions_per_day": 4.27,
    "highest_transaction_day": "2023-06-01",
    "highest_transaction_count": 15,
    "outgoing_transactions": 85,
    "incoming_transactions": 43,
    "total_sent": "752.25",
    "total_received": "2015.5",
    "net_flow": "1263.25",
    "total_gas_spent": "0.35",
    "avg_gas_price": "0.00000005"
  },
  "token_transfer_analytics": {
    "total_token_transfers": 42,
    "tokens_sent": {
      "0x5a1e945c2b2b8bc40f402b5d9e5e854f": "500.25"
    },
    "tokens_received": {
      "0x5a1e945c2b2b8bc40f402b5d9e5e854f": "1750.5"
    },
    "net_token_flow": {
      "0x5a1e945c2b2b8bc40f402b5d9e5e854f": "1250.25"
    }
  },
  "contract_analytics": {
    "total_contract_interactions": 35,
    "unique_contracts": 5,
    "most_interacted_contracts": [
      {
        "address": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
        "name": "Test Token",
        "interaction_count": 25
      },
      {
        "address": "0x6b2c843d7a9b8bc40f402b8d9e8e955a",
        "name": "Test NFT",
        "interaction_count": 8
      }
    ],
    "most_called_methods": [
      {
        "name": "transfer",
        "count": 22
      },
      {
        "name": "approve",
        "count": 6
      }
    ]
  },
  "daily_transaction_chart": [
    {"date": "2023-05-07", "count": 2, "volume": "10.5"},
    {"date": "2023-05-08", "count": 5, "volume": "25.75"},
    // Additional days...
    {"date": "2023-06-05", "count": 6, "volume": "35.25"}
  ],
  "time_of_day_distribution": [
    {"hour": 0, "count": 5},
    {"hour": 1, "count": 2},
    // Additional hours...
    {"hour": 23, "count": 4}
  ]
}
```

### Account Creation

#### POST /api/accounts/create

Creates a new account on the testnet.

**Request Body:**
```json
{
  "generate_private_key": true, 
  "fund_account": true,
  "initial_balance": "10"
}
```

**Example Response:**
```json
{
  "address": "0x9d34f67b5c1f86c56354a7d6d1766b98a9cfd0e7",
  "private_key": "0x2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b",
  "mnemonic": "scatter erosion timber whisper drift wire trial grief harsh mystery love genuine exhaust embark novel",
  "balance": "10.0",
  "transaction_hash": "0x7d2a32cb629d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3ea6",
  "creation_time": "2023-06-05T19:35:00Z",
  "warning": "This is a testnet account. Never use this private key on mainnet or for real assets."
}
```

### Account Validation

#### GET /api/accounts/validate/{address}

Validates an account address format and checks if the account exists on the blockchain.

**Parameters:**
- `address`: The account address to validate (required)

**Example Response:**
```json
{
  "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
  "is_valid_format": true,
  "exists_on_chain": true,
  "account_type": "eoa",
  "checksum_address": "0x8C23F56B4c0F76C46354a6D5d0766B87A8cFD9e8"
}
```

### WebSocket Subscriptions

Real-time account updates are available through WebSocket connections:

```
ws://localhost:3000/api/accounts/ws
```

### Subscribe to Account Updates

**Subscription Message:**
```json
{
  "action": "subscribe",
  "topics": [
    "account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
    "account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8:balance",
    "account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8:transactions",
    "account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8:token:0x5a1e945c2b2b8bc40f402b5d9e5e854f"
  ]
}
```

**Example Messages Received:**

Balance update:
```json
{
  "topic": "account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8:balance",
  "type": "balance_update",
  "data": {
    "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
    "previous_balance": "1258.75",
    "new_balance": "1253.5",
    "delta": "-5.25",
    "transaction_hash": "0x7d2a32cb629d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3ea6",
    "block_number": 128956,
    "timestamp": "2023-06-05T19:15:25Z"
  }
}
```

New transaction:
```json
{
  "topic": "account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8:transactions",
  "type": "new_transaction",
  "data": {
    "hash": "0x7d2a32cb629d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3ea6",
    "block_number": 128956,
    "timestamp": "2023-06-05T19:15:25Z",
    "from": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
    "to": "0x7d23f56b4c0f76c46354a6d5d0766b87a8cfd9e9",
    "value": "5.25",
    "status": "success",
    "gas_used": 21000,
    "gas_price": "0.00000005",
    "fee": "0.00105"
  }
}
```

Token balance update:
```json
{
  "topic": "account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8:token:0x5a1e945c2b2b8bc40f402b5d9e5e854f",
  "type": "token_balance_update",
  "data": {
    "address": "0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8",
    "token_address": "0x5a1e945c2b2b8bc40f402b5d9e5e854f",
    "token_name": "Test Token",
    "token_symbol": "TST",
    "previous_balance": "1350.75",
    "new_balance": "1250.75",
    "delta": "-100.0",
    "transaction_hash": "0x6c1a21db518d88c5ecfb5bf42344821f518ebfb63adba5a3f8e42ee11bae3e95",
    "block_number": 128954,
    "timestamp": "2023-06-05T19:14:55Z"
  }
}
```

## Error Responses

All API endpoints use standard HTTP status codes and return error details in JSON format:

```json
{
  "error": {
    "code": 400,
    "message": "Invalid account address",
    "details": "Account address must be a 0x-prefixed hex string of 42 characters"
  },
  "timestamp": "2023-06-05T19:15:30Z",
  "request_id": "req-5d23f56b4c0f"
}
```

Common error status codes:
- `400 Bad Request`: Invalid parameters or request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Account not found
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
const { AccountClient } = require('artha-chain-sdk');

const accountClient = new AccountClient('http://localhost:3000/api');

// Get account information
accountClient.getAccount('0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8')
  .then(account => {
    console.log('Account:', account);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });

// Get token balances
accountClient.getTokenBalances('0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8')
  .then(balances => {
    console.log('Token balances:', balances);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });

// Subscribe to balance updates
accountClient.subscribe('account:0x8c23f56b4c0f76c46354a6d5d0766b87a8cfd9e8:balance', (update) => {
  console.log('Balance update:', update);
});
``` 