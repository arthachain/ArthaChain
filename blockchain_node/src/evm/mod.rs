pub mod advanced_gas_metering;
pub mod backend;
pub mod execution_engine;
pub mod executor;
pub mod opcodes;
pub mod precompiled;
pub mod precompiles;
pub mod rpc;
pub mod runtime;
pub mod types;

// EVM Constants
/// Default gas price (in wei per gas unit)
pub const DEFAULT_GAS_PRICE: u64 = 20_000_000_000; // 20 Gwei

/// Default gas limit for transactions
pub const DEFAULT_GAS_LIMIT: u64 = 21_000; // Standard ETH transfer

/// Block gas limit
pub const BLOCK_GAS_LIMIT: u64 = 30_000_000; // 30M gas per block

/// Maximum code size in bytes
pub const MAX_CODE_SIZE: u64 = 24_576; // 24KB

// Re-export commonly used types
pub use advanced_gas_metering::{
    AdvancedGasConfig, AdvancedGasMeter, Eip1559GasPrice, GasEstimationResult,
};
pub use backend::{EvmAccount, EvmBackend};
pub use executor::EvmExecutor;
pub use runtime::{EvmExecutionContext, EvmRuntime, StepResult};
pub use types::{EvmAddress, EvmConfig, EvmError, EvmExecutionResult, EvmLog, EvmTransaction};
pub use rpc::EvmRpcService;
