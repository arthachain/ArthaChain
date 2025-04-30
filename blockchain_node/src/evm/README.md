# Ethereum Virtual Machine (EVM) Support

This module adds Ethereum Virtual Machine (EVM) support to the blockchain node, allowing it to execute Solidity smart contracts natively.

## Features

- **Native Solidity Support**: Deploy and execute Solidity smart contracts directly on the blockchain.
- **Ethereum Compatibility**: Compatible with Ethereum tooling (Hardhat, Remix, MetaMask, etc.).
- **Lightweight Integration**: EVM integration is modular and can be enabled/disabled via feature flags.
- **Low Overhead**: Minimal impact on the blockchain's core functionality.
- **Storage Integration**: Smart contract state is stored in the blockchain's hybrid storage system.

## Configuration

EVM support is enabled through the `evm` feature flag. Add it to your compiler flags when building the project:

```sh
cargo build --features evm
```

### Configuration Parameters

The EVM integration can be configured in the `config.rs` file:

```rust
// In your blockchain_node's configuration
config.evm = EvmConfig {
    enabled: true,
    chain_id: 1337,
    default_gas_price: 20_000_000_000,  // 20 gwei
    default_gas_limit: 8_000_000,
    enable_precompiles: true,
};

// Configure Ethereum-compatible JSON-RPC
config.api.enable_eth_rpc = true;
config.api.eth_rpc_port = 8545;
```

## Using with Ethereum Tools

### Hardhat

Add the following to your Hardhat configuration:

```js
// hardhat.config.js
module.exports = {
  networks: {
    arthaChain: {
      url: "http://localhost:8545",
      chainId: 1337,
      accounts: ["0xPRIVATE_KEY_HERE"]
    }
  },
};
```

### MetaMask

1. Open MetaMask
2. Click on "Networks" dropdown
3. Select "Add Network"
4. Enter the following information:
   - Network Name: Artha Chain
   - RPC URL: http://localhost:8545
   - Chain ID: 1337
   - Symbol: ARTHA

## Architecture

The EVM implementation consists of several components:

1. **EvmRuntime**: Core EVM execution environment.
2. **EvmBackend**: Adapter between EVM and blockchain storage.
3. **EvmExecutor**: Transaction processing and execution management.
4. **EvmRpcService**: Ethereum-compatible JSON-RPC API.
5. **Precompiles**: Standard Ethereum precompiled contracts.

## Limitations

- Gas estimation may not be perfectly aligned with Ethereum.
- Some advanced Ethereum features may not be fully implemented.
- Performance may differ from dedicated Ethereum nodes.

## Future Improvements

- Add support for EVM events subscription via WebSockets.
- Implement more efficient state storage with Merkle Patricia Tries.
- Add support for additional EIPs (Ethereum Improvement Proposals).
- Improve gas estimation accuracy.
- Add support for debugging and tracing EVM transactions.

## Support

For issues with EVM integration, please file a bug report with the tag "evm". 