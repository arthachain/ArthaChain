// EVM Runtime implementation for our blockchain
// This module provides an Ethereum Virtual Machine (EVM) runtime for executing Solidity smart contracts

mod runtime;
mod backend;
mod precompiles;
mod types;
mod executor;
mod rpc;

// Re-export main components
pub use runtime::EvmRuntime;
pub use types::{EvmAddress, EvmTransaction, EvmExecutionResult, EvmLog, EvmConfig};
pub use executor::EvmExecutor;
pub use rpc::EvmRpcService;

/// Configuration for initializing the EVM runtime
pub const DEFAULT_GAS_PRICE: u64 = 20_000_000_000;   // 20 GWEI
pub const DEFAULT_GAS_LIMIT: u64 = 21_000;           // Standard gas limit for a transfer

/// Conversion rate between native token and EVM gas
pub const NATIVE_TO_GAS_CONVERSION_RATE: u64 = 1;  // 1:1 ratio as a starting point 