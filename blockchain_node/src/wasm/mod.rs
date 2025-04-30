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

mod runtime;
mod host;
mod types;
mod executor;
mod upgrade;
mod verification;
mod standards;
mod debug;

// Re-export key components
pub use runtime::{WasmRuntime, WasmEnv, GasMeter};
pub use host::{HostEnv, register_host_functions};
pub use executor::ContractExecutor;
pub use types::{WasmError, ContractResult, ContractContext, CallInfo, WasmGasConfig};
pub use upgrade::{UpgradeManager, UpgradePattern, ContractVersion, StorageLayout};
pub use verification::{ContractVerifier, SafetyProperty, LivenessProperty, VerificationResult};
pub use standards::{StandardRegistry, ContractStandard, StandardType, TokenStandard, GovernanceStandard, SecurityStandard};
pub use debug::{DebugManager, DebugSession, Breakpoint, StackFrame, StackTrace};

// Gas limits and memory constraints
pub const DEFAULT_GAS_LIMIT: u64 = 10_000_000;
pub const MAX_MEMORY_PAGES: u32 = 100;  // 6.4MB (64KB per page)
pub const MAX_CONTRACT_SIZE: usize = 1024 * 1024 * 2;  // 2MB

// Gas costs for various operations
pub const GAS_COST_CALL_BASE: u64 = 100;
pub const GAS_COST_STORAGE_READ: u64 = 10;
pub const GAS_COST_STORAGE_WRITE: u64 = 50;
pub const GAS_COST_STORAGE_DELETE: u64 = 30;
pub const GAS_COST_CREATE_CONTRACT: u64 = 10000;

/// Creates a default configuration for WASM execution
pub fn default_config() -> WasmGasConfig {
    WasmGasConfig {
        gas_limit: DEFAULT_GAS_LIMIT,
        max_memory_pages: MAX_MEMORY_PAGES,
        gas_per_instruction: 1,
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
        return Err(WasmError::InvalidBytecode);
    }
    
    Ok(())
} 