pub mod abi;
pub mod context;
// pub mod debug; // Missing wasmer dep
// pub mod debug_test; // Missing wasmer dep
pub mod engine;
pub mod execution_engine;
pub mod executor;
pub mod gas;
pub mod host;
pub mod host_functions;
pub mod rpc;
pub mod runtime;
pub mod standards;
pub mod storage;
pub mod types;
pub mod upgrade;
pub mod verification;
pub mod vm;
pub mod wallet_bindings;

// Re-export commonly used types
pub use abi::*;
pub use context::*;
// pub use debug::*;
pub use engine::*;
pub use execution_engine::*;
pub use executor::*;
pub use gas::*;
pub use host::*;
pub use host_functions::*;
pub use rpc::*;
pub use runtime::*;
pub use standards::*;
pub use storage::*;
pub use types::*;
pub use upgrade::*;
pub use verification::*;
pub use vm::*;
pub use wallet_bindings::*;

/// Validate WASM bytecode for basic correctness
pub fn validate_wasm_bytecode(bytecode: &[u8]) -> anyhow::Result<()> {
    use wasmparser::Parser;
    
    let parser = Parser::new(0);
    match parser.parse_all(bytecode) {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("WASM validation failed: {}", e)),
    }
}

// Gas cost constants for WASM operations
pub const GAS_COST_BASE: u64 = 1;
pub const GAS_COST_STORAGE_READ: u64 = 200;
pub const GAS_COST_STORAGE_WRITE: u64 = 300;
pub const GAS_COST_STORAGE_DELETE: u64 = 100;
pub const GAS_COST_CONTEXT_READ: u64 = 50;
pub const GAS_COST_FUNCTION_CALL: u64 = 100;
pub const GAS_COST_MEMORY_ACCESS: u64 = 10;
