# Smart Contracts API

The Artha Chain Testnet Smart Contracts API allows developers to deploy, interact with, and monitor smart contracts on the Artha Chain testnet.

## Base URL

All API endpoints are relative to the base URL of a validator node:
- `http://localhost:3000` (validator1)
- `http://localhost:3001` (validator2)
- `http://localhost:3002` (validator3)
- `http://localhost:3003` (validator4)

## Authentication

The Smart Contracts API requires authentication for most endpoints. See the [Authentication](./authentication.md) documentation for details.

## Contract Deployment

### Deploy Contract

Deploys a new smart contract to the blockchain.

**Endpoint:** `POST /api/contracts/deploy`

**Request:**
```json
{
  "bytecode": "0x608060405234801561001057600080fd5b50610150806100206000396000f3fe608060405234801561001057600080fd5b50600436106100365760003560e01c80632e64cec11461003b5780636057361d14610059575b600080fd5b610043610075565b60405161005091906100d9565b60405180910390f35b610073600480360381019061006e919061009d565b61007e565b005b60008054905090565b8060008190555050565b60008135905061009781610103565b92915050565b6000602082840312156100b3576100b26100fe565b5b60006100c184828501610088565b91505092915050565b6100d3816100f4565b82525050565b60006020820190506100ee60008301846100ca565b92915050565b6000819050919050565b600080fd5b61010c816100f4565b811461011757600080fd5b5056fea264697066735822122000d6ee56c42df6b7c92bb8d5dc9ecf602a7fd69af6ebabec07788437e347788b64736f6c63430008070033",
  "abi": [
    {
      "inputs": [],
      "name": "retrieve",
      "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [{"internalType": "uint256", "name": "num", "type": "uint256"}],
      "name": "store",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
  ],
  "constructor_args": [],
  "value": 0,
  "gas_limit": 1000000,
  "gas_price": 1,
  "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "private_key": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5"
}
```

**Parameters:**
- `bytecode` (required): The compiled contract bytecode (hex string starting with 0x)
- `abi` (required): The contract ABI (Application Binary Interface)
- `constructor_args` (optional): Arguments to pass to the constructor
- `value` (optional): Amount of tokens to send with the deployment
- `gas_limit` (optional): Maximum gas units for deployment
- `gas_price` (optional): Price per unit of gas
- `from` (required): Address deploying the contract
- `private_key` (required): Private key of the deploying address

**Response:**
```json
{
  "transaction_hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "contract_address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "status": "pending",
  "gas_used": null,
  "block_hash": null,
  "block_number": null
}
```

### Verify Contract

Verifies the source code of a deployed contract for public visibility in the explorer.

**Endpoint:** `POST /api/contracts/verify`

**Request:**
```json
{
  "address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "name": "SimpleStorage",
  "source_code": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.7;\n\ncontract SimpleStorage {\n    uint256 private value;\n\n    function store(uint256 num) public {\n        value = num;\n    }\n\n    function retrieve() public view returns (uint256) {\n        return value;\n    }\n}",
  "compiler_version": "0.8.7",
  "optimization_enabled": true,
  "optimization_runs": 200,
  "constructor_arguments": "0x"
}
```

**Parameters:**
- `address` (required): Contract address to verify
- `name` (required): Contract name
- `source_code` (required): Complete source code
- `compiler_version` (required): Solidity compiler version
- `optimization_enabled` (required): Whether optimizations were enabled
- `optimization_runs` (optional): Number of optimization runs
- `constructor_arguments` (optional): ABI-encoded constructor arguments (hex string)

**Response:**
```json
{
  "status": "success",
  "message": "Contract successfully verified",
  "address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "verified": true
}
```

## Contract Interaction

### Call Contract Method (Read)

Calls a read-only method on a smart contract.

**Endpoint:** `POST /api/contracts/call`

**Request:**
```json
{
  "address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "method": "retrieve",
  "args": [],
  "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
}
```

**Parameters:**
- `address` (required): Contract address
- `method` (required): Method name to call
- `args` (optional): Array of arguments for the method
- `from` (optional): Address making the call (for view functions that need msg.sender)

**Response:**
```json
{
  "result": 42,
  "success": true
}
```

### Send Contract Transaction (Write)

Executes a state-changing method on a smart contract.

**Endpoint:** `POST /api/contracts/execute`

**Request:**
```json
{
  "address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "method": "store",
  "args": [100],
  "value": 0,
  "gas_limit": 100000,
  "gas_price": 1,
  "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
  "private_key": "0x8a4d93e7e0c7f0f9e5b5a14a3bcd7f317c74c536bdebffc44a93d33eadbc81a5"
}
```

**Parameters:**
- `address` (required): Contract address
- `method` (required): Method name to execute
- `args` (optional): Array of arguments for the method
- `value` (optional): Amount of tokens to send with the transaction
- `gas_limit` (optional): Maximum gas units for execution
- `gas_price` (optional): Price per unit of gas
- `from` (required): Address making the call
- `private_key` (required): Private key of the sender's address

**Response:**
```json
{
  "transaction_hash": "0x4a3d52beba35f6cd932ac3b1063fa3b93984c76c27e0c7f0f9e5b5a14a3bcd7f",
  "status": "pending",
  "gas_used": null,
  "block_hash": null,
  "block_number": null
}
```

### Estimate Gas

Estimates the gas required to execute a contract method.

**Endpoint:** `POST /api/contracts/estimate-gas`

**Request:**
```json
{
  "address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
  "method": "store",
  "args": [100],
  "value": 0,
  "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
}
```

**Parameters:**
- `address` (required): Contract address
- `method` (required): Method name to estimate
- `args` (optional): Array of arguments for the method
- `value` (optional): Amount of tokens to send with the transaction
- `from` (required): Address making the call

**Response:**
```json
{
  "gas_estimate": 43842,
  "gas_price": 1,
  "estimated_fee": 43842
}
```

## Contract Management

### Get Contract ABI

Retrieves the ABI of a verified contract.

**Endpoint:** `GET /api/contracts/:address/abi`

**Parameters:**
- `address` (path parameter): Contract address

**Response:**
```json
{
  "abi": [
    {
      "inputs": [],
      "name": "retrieve",
      "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [{"internalType": "uint256", "name": "num", "type": "uint256"}],
      "name": "store",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
  ],
  "verified": true
}
```

### Get Contract Source Code

Retrieves the source code of a verified contract.

**Endpoint:** `GET /api/contracts/:address/source`

**Parameters:**
- `address` (path parameter): Contract address

**Response:**
```json
{
  "name": "SimpleStorage",
  "source_code": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.7;\n\ncontract SimpleStorage {\n    uint256 private value;\n\n    function store(uint256 num) public {\n        value = num;\n    }\n\n    function retrieve() public view returns (uint256) {\n        return value;\n    }\n}",
  "compiler_version": "0.8.7",
  "optimization_enabled": true,
  "optimization_runs": 200,
  "constructor_arguments": "0x",
  "verified": true,
  "verification_date": "2023-06-01T12:34:56Z"
}
```

### Get Contract Events

Retrieves events emitted by a specific contract.

**Endpoint:** `GET /api/contracts/:address/events`

**Parameters:**
- `address` (path parameter): Contract address
- `event_name` (query parameter, optional): Filter by event name
- `from_block` (query parameter, optional): Start block number
- `to_block` (query parameter, optional): End block number (default: "latest")
- `limit` (query parameter, optional): Maximum number of events to return (default: 10, max: 100)
- `offset` (query parameter, optional): Number of events to skip (default: 0)

**Response:**
```json
{
  "events": [
    {
      "event": "Transfer",
      "signature": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
      "transaction_hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
      "block_number": 1024,
      "log_index": 0,
      "timestamp": 1650326400,
      "parameters": {
        "from": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7",
        "to": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
        "value": 100
      }
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

## Code Analysis

### Analyze Contract

Analyzes a contract for potential security vulnerabilities.

**Endpoint:** `POST /api/contracts/analyze`

**Request:**
```json
{
  "source_code": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.7;\n\ncontract SimpleStorage {\n    uint256 private value;\n\n    function store(uint256 num) public {\n        value = num;\n    }\n\n    function retrieve() public view returns (uint256) {\n        return value;\n    }\n}",
  "compiler_version": "0.8.7",
  "analysis_level": "standard"
}
```

**Parameters:**
- `source_code` (required): Contract source code
- `compiler_version` (required): Solidity compiler version
- `analysis_level` (optional): Level of analysis (basic, standard, detailed)

**Response:**
```json
{
  "issues": [
    {
      "severity": "low",
      "title": "Missing Events",
      "description": "Consider emitting events when changing state variables",
      "line_number": 6,
      "code_snippet": "function store(uint256 num) public {",
      "recommendation": "Add an event to track state changes"
    }
  ],
  "metrics": {
    "complexity": "low",
    "size": "small",
    "gas_efficiency": "high"
  },
  "success": true
}
```

## Error Responses

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_BYTECODE | The provided bytecode is invalid |
| 400 | INVALID_ABI | The provided ABI is invalid or inconsistent with the bytecode |
| 400 | COMPILATION_FAILED | Contract compilation failed |
| 400 | EXECUTION_FAILED | Contract method execution failed |
| 404 | CONTRACT_NOT_FOUND | The specified contract address does not exist |
| 404 | METHOD_NOT_FOUND | The specified method does not exist in the contract |
| 401 | UNAUTHORIZED | Authentication required or failed |
| 500 | DEPLOYMENT_FAILED | Contract deployment failed |

### Error Response Example:

```json
{
  "error": {
    "code": "METHOD_NOT_FOUND",
    "message": "Method 'setValues' not found in contract",
    "details": {
      "contract_address": "0xbdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5",
      "method": "setValues",
      "available_methods": ["store", "retrieve"]
    }
  }
}
```

## Implementation Notes

- Smart contracts are deployed using the EVM (Ethereum Virtual Machine) compatible execution environment
- Contract verification uses a multi-step process to validate source code integrity
- Gas estimates are calculated based on the current state of the blockchain
- Contract analysis uses static analysis tools to identify potential issues
- Rate limits apply to prevent excessive API usage (see [Rate Limiting](./rate-limiting.md)) 