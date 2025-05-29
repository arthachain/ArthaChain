//! WASM Smart Contract Runtime Module
//!
//! This module implements WebAssembly smart contract functionality:
//! - Runtime: Core execution environment for WASM contracts
//! - Host: Host functions available to contracts
//! - Types: Common types and error definitions
//! - Executor: High-level contract operation interface
//! - RPC: JSON-RPC interface for contract operations
//! - Upgrade: Contract upgradeability patterns
//! - Verification: Formal verification of contracts
//! - Standards: Contract standards and interfaces
//! - Debug: Debugging infrastructure

use log::info;

mod abi;
mod context;
mod debug;
// mod debug_test; // Disabled due to API mismatch
mod engine;
mod executor;
mod gas;
mod host;
mod host_functions;
mod rpc;
mod runtime;
mod standards;
mod storage;
mod types;
mod upgrade;
mod verification;
mod vm;

// Re-export key components
pub use abi::ContractABI;
pub use context::ContractContext;
pub use debug::{Breakpoint, DebugManager, DebugSession, StackFrame, StackTrace};
pub use executor::{ContractExecutor, WasmExecutor};
pub use gas::{GasCosts, GasMeter};
pub use host::{register_host_functions, HostEnv};
pub use host_functions::{
    crypto_verify, get_block_number, get_caller, storage_delete, storage_read, storage_write,
};
pub use rpc::WasmRpcService;
pub use runtime::{WasmConfig, WasmEnv, WasmExecutionResult, WasmRuntime};
pub use standards::{
    ContractStandard, GovernanceStandard, SecurityStandard, StandardRegistry, StandardType,
    TokenStandard,
};
pub use storage::WasmStorage;
pub use types::{CallContext, CallInfo, ContractResult, WasmError, WasmGasConfig};
pub use upgrade::{ContractVersion, StorageLayout, UpgradeManager, UpgradePattern};
pub use verification::{ContractVerifier, LivenessProperty, SafetyProperty, VerificationResult};
pub use vm::{WasmEnv as VmEnv, WasmVm, WasmVmConfig};

// Gas limits and memory constraints
pub const DEFAULT_GAS_LIMIT: u64 = 10_000_000;
pub const MAX_MEMORY_PAGES: u32 = 100; // 6.4MB (64KB per page)
pub const MAX_CONTRACT_SIZE: usize = 1024 * 1024 * 2; // 2MB

// Gas costs for various operations
pub const GAS_COST_CALL_BASE: u64 = 100;
pub const GAS_COST_STORAGE_READ: u64 = 10;
pub const GAS_COST_STORAGE_WRITE: u64 = 50;
pub const GAS_COST_STORAGE_DELETE: u64 = 30;
pub const GAS_COST_CREATE_CONTRACT: u64 = 10000;

/// Creates a default configuration for WASM execution
pub fn default_config() -> WasmConfig {
    WasmConfig {
        max_memory_pages: MAX_MEMORY_PAGES,
        gas_limit: DEFAULT_GAS_LIMIT,
        create_gas_limit: DEFAULT_GAS_LIMIT,
        gas_costs: default_gas_costs(),
        max_contract_size: MAX_CONTRACT_SIZE,
    }
}

/// Creates default gas costs
pub fn default_gas_costs() -> WasmGasConfig {
    WasmGasConfig {
        call_base: GAS_COST_CALL_BASE,
        storage_read: GAS_COST_STORAGE_READ,
        storage_write: GAS_COST_STORAGE_WRITE,
        storage_delete: GAS_COST_STORAGE_DELETE,
        create_contract: GAS_COST_CREATE_CONTRACT,
    }
}

/// Initializes the WASM runtime
pub fn init() -> anyhow::Result<()> {
    info!("Initializing WASM smart contract runtime...");
    // Any global initialization can be done here
    Ok(())
}

/// WASM error to string conversion
pub fn error_to_string(error: WasmError) -> String {
    match error {
        WasmError::ValidationError(msg) => format!("Validation error: {}", msg),
        WasmError::CompilationError(msg) => format!("Compilation error: {}", msg),
        WasmError::InstantiationError(msg) => format!("Instantiation error: {}", msg),
        WasmError::ExecutionError(msg) => format!("Execution error: {}", msg),
        WasmError::StorageError(msg) => format!("Storage error: {}", msg),
        WasmError::OutOfGas => "Out of gas".to_string(),
        WasmError::ExecutionTimeout => "Execution timeout".to_string(),
        WasmError::MemoryError(msg) => format!("Memory error: {}", msg),
        WasmError::HostError(msg) => format!("Host error: {}", msg),
    }
}

/// Validate WASM bytecode
///
/// Checks if the bytecode is valid WASM and meets basic constraints:
/// - Size is within limits
/// - Contains valid WASM magic bytes
pub fn validate_wasm_bytecode(bytecode: &[u8]) -> Result<(), WasmError> {
    // Check size
    if bytecode.len() > MAX_CONTRACT_SIZE {
        return Err(WasmError::BytecodeTooLarge);
    }

    // Check magic bytes (WASM header is \0asm)
    if bytecode.len() < 8 || &bytecode[0..4] != b"\0asm" {
        return Err(WasmError::InvalidBytecode(
            "Invalid WASM bytecode".to_string(),
        ));
    }

    Ok(())
}
