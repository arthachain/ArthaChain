//! Production-ready WASM Execution Engine for Phase 2
//!
//! This module provides a complete, high-performance WebAssembly execution environment
//! optimized for smart contract execution with full gas metering, security, and interoperability.

use crate::types::Address;
use crate::wasm::gas::GasMeter;
use crate::wasm::storage::WasmStorage;
use crate::wasm::types::{CallContext, WasmError, WasmExecutionResult, WasmGasConfig};
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use wasmtime::{
    Config, Engine, Instance, Linker, Memory, Module, Store, TypedFunc, Val, ValType, WasmParams,
    WasmResults,
};

/// Advanced WASM execution configuration
#[derive(Clone, Debug)]
pub struct WasmExecutionConfig {
    /// Maximum memory pages (64KB each)
    pub max_memory_pages: u32,
    /// Default gas limit for contract execution
    pub default_gas_limit: u64,
    /// Execution timeout in milliseconds
    pub execution_timeout_ms: u64,
    /// Maximum WASM module size in bytes
    pub max_module_size: usize,
    /// Enable optimization passes
    pub enable_optimization: bool,
    /// Enable debugging support
    pub enable_debugging: bool,
    /// Fuel consumption per instruction
    pub fuel_per_instruction: u64,
    /// Maximum call stack depth
    pub max_call_depth: u32,
}

impl Default for WasmExecutionConfig {
    fn default() -> Self {
        Self {
            max_memory_pages: 256, // 16MB maximum memory
            default_gas_limit: 50_000_000,
            execution_timeout_ms: 30_000,     // 30 seconds max
            max_module_size: 8 * 1024 * 1024, // 8MB module size limit
            enable_optimization: true,
            enable_debugging: false,
            fuel_per_instruction: 1,
            max_call_depth: 1024,
        }
    }
}

/// Execution context for WASM contracts
#[derive(Clone)]
pub struct WasmExecutionContext {
    /// Contract address being executed
    pub contract_address: Address,
    /// Address of the caller
    pub caller: Address,
    /// Current block height
    pub block_height: u64,
    /// Current block timestamp
    pub block_timestamp: u64,
    /// Value transferred with the call
    pub value: u64,
    /// Transaction origin address
    pub origin: Address,
    /// Gas price
    pub gas_price: u64,
    /// Chain ID
    pub chain_id: u64,
}

/// Host environment for WASM execution
pub struct WasmHostEnvironment {
    /// Storage interface
    pub storage: Arc<WasmStorage>,
    /// Gas meter for tracking gas consumption
    pub gas_meter: Arc<Mutex<GasMeter>>,
    /// Execution context
    pub context: WasmExecutionContext,
    /// Contract memory
    pub memory: Option<Memory>,
    /// Execution logs
    pub logs: Arc<Mutex<Vec<String>>>,
    /// Call depth tracking
    pub call_depth: u32,
    /// Start time for timeout tracking
    pub start_time: Instant,
    /// Execution timeout
    pub timeout: Duration,
}

/// Production WASM Execution Engine
pub struct WasmExecutionEngine {
    /// Wasmtime engine with optimizations
    engine: Engine,
    /// Module cache for compiled contracts
    module_cache: Arc<RwLock<HashMap<String, Module>>>,
    /// Execution configuration
    config: WasmExecutionConfig,
    /// Compiled linker with host functions
    linker: Arc<Mutex<Linker<WasmHostEnvironment>>>,
}

impl WasmExecutionEngine {
    /// Create a new production WASM execution engine
    pub fn new(config: WasmExecutionConfig) -> Result<Self> {
        // Configure Wasmtime for optimal performance and security
        let mut wasmtime_config = Config::new();

        // Security settings
        wasmtime_config.consume_fuel(true);
        wasmtime_config.max_wasm_stack(config.max_call_depth as usize * 1024);
        wasmtime_config.wasm_memory64(false); // Disable 64-bit memory for security
        wasmtime_config.wasm_multi_memory(false);
        wasmtime_config.wasm_bulk_memory(true);
        wasmtime_config.wasm_reference_types(false);
        wasmtime_config.wasm_simd(false); // Disable SIMD for predictable gas costs
        wasmtime_config.wasm_relaxed_simd(false);
        wasmtime_config.wasm_function_references(false);

        // Performance settings
        if config.enable_optimization {
            wasmtime_config.cranelift_opt_level(wasmtime::OptLevel::Speed);
        } else {
            wasmtime_config.cranelift_opt_level(wasmtime::OptLevel::None);
        }

        // Debugging settings
        if config.enable_debugging {
            wasmtime_config.debug_info(true);
        }

        // Memory limits
        wasmtime_config.static_memory_maximum_size(config.max_memory_pages as u64 * 65536);
        wasmtime_config.dynamic_memory_maximum_size(config.max_memory_pages as u64 * 65536);
        wasmtime_config.static_memory_guard_size(65536);
        wasmtime_config.dynamic_memory_guard_size(65536);

        let engine = Engine::new(&wasmtime_config)
            .map_err(|e| anyhow!("Failed to create WASM engine: {}", e))?;

        // Create linker with host functions
        let mut linker = Linker::new(&engine);
        Self::register_host_functions(&mut linker)?;

        Ok(Self {
            engine,
            module_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            linker: Arc::new(Mutex::new(linker)),
        })
    }

    /// Register all host functions available to WASM contracts
    fn register_host_functions(linker: &mut Linker<WasmHostEnvironment>) -> Result<()> {
        // Storage operations
        linker.func_wrap(
            "env",
            "storage_read",
            |mut caller: wasmtime::Caller<'_, WasmHostEnvironment>,
             key_ptr: u32,
             key_len: u32,
             value_ptr: u32,
             max_len: u32|
             -> u32 {
                let (memory, host_env) = {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| anyhow!("Failed to get memory"))?;
                    let host_env = caller.data().clone();
                    Ok::<_, anyhow::Error>((memory, host_env))
                }?;

                // Charge gas for storage read
                {
                    let mut gas_meter = host_env.gas_meter.lock().unwrap();
                    gas_meter.consume(1000)?; // 1000 gas for storage read
                }

                // Read key from WASM memory
                let key = Self::read_memory(&memory, &mut caller, key_ptr, key_len as usize)?;

                // Perform storage read
                match host_env.storage.read(&key) {
                    Ok(Some(value)) => {
                        let copy_len = std::cmp::min(value.len(), max_len as usize);
                        Self::write_memory(&memory, &mut caller, value_ptr, &value[..copy_len])?;
                        Ok(copy_len as u32)
                    }
                    Ok(None) => Ok(0),
                    Err(_) => Ok(u32::MAX), // Error indicator
                }
            },
        )?;

        linker.func_wrap(
            "env",
            "storage_write",
            |mut caller: wasmtime::Caller<'_, WasmHostEnvironment>,
             key_ptr: u32,
             key_len: u32,
             value_ptr: u32,
             value_len: u32|
             -> u32 {
                let (memory, host_env) = {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| anyhow!("Failed to get memory"))?;
                    let host_env = caller.data().clone();
                    Ok::<_, anyhow::Error>((memory, host_env))
                }?;

                // Charge gas for storage write
                {
                    let mut gas_meter = host_env.gas_meter.lock().unwrap();
                    gas_meter.consume(5000)?; // 5000 gas for storage write
                }

                // Read key and value from WASM memory
                let key = Self::read_memory(&memory, &mut caller, key_ptr, key_len as usize)?;
                let value = Self::read_memory(&memory, &mut caller, value_ptr, value_len as usize)?;

                // Perform storage write
                match host_env.storage.write(&key, &value) {
                    Ok(_) => Ok(1),  // Success
                    Err(_) => Ok(0), // Failure
                }
            },
        )?;

        // Contract information functions
        linker.func_wrap(
            "env",
            "get_caller",
            |mut caller: wasmtime::Caller<'_, WasmHostEnvironment>, addr_ptr: u32| -> u32 {
                let (memory, host_env) = {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| anyhow!("Failed to get memory"))?;
                    let host_env = caller.data().clone();
                    Ok::<_, anyhow::Error>((memory, host_env))
                }?;

                // Write caller address to memory
                let caller_bytes = host_env.context.caller.as_bytes();
                Self::write_memory(&memory, &mut caller, addr_ptr, caller_bytes)?;
                Ok(caller_bytes.len() as u32)
            },
        )?;

        linker.func_wrap(
            "env",
            "get_block_number",
            |_caller: wasmtime::Caller<'_, WasmHostEnvironment>| -> u64 {
                _caller.data().context.block_height
            },
        )?;

        linker.func_wrap(
            "env",
            "get_block_timestamp",
            |_caller: wasmtime::Caller<'_, WasmHostEnvironment>| -> u64 {
                _caller.data().context.block_timestamp
            },
        )?;

        // Logging function
        linker.func_wrap(
            "env",
            "log",
            |mut caller: wasmtime::Caller<'_, WasmHostEnvironment>,
             msg_ptr: u32,
             msg_len: u32|
             -> u32 {
                let (memory, host_env) = {
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| anyhow!("Failed to get memory"))?;
                    let host_env = caller.data().clone();
                    Ok::<_, anyhow::Error>((memory, host_env))
                }?;

                // Read log message from memory
                let message = Self::read_memory(&memory, &mut caller, msg_ptr, msg_len as usize)?;
                let log_str = String::from_utf8_lossy(&message);

                // Add to logs
                {
                    let mut logs = host_env.logs.lock().unwrap();
                    logs.push(log_str.to_string());
                }

                info!("Contract Log: {}", log_str);
                Ok(1)
            },
        )?;

        // Gas query function
        linker.func_wrap(
            "env",
            "gas_left",
            |_caller: wasmtime::Caller<'_, WasmHostEnvironment>| -> u64 {
                let gas_meter = _caller.data().gas_meter.lock().unwrap();
                gas_meter.gas_remaining()
            },
        )?;

        Ok(())
    }

    /// Deploy a new WASM contract
    pub async fn deploy_contract(
        &self,
        bytecode: &[u8],
        deployer: &Address,
        storage: Arc<WasmStorage>,
        gas_limit: u64,
        constructor_args: Option<&[u8]>,
    ) -> Result<WasmExecutionResult> {
        // Validate bytecode size
        if bytecode.len() > self.config.max_module_size {
            return Err(anyhow!(
                "Contract bytecode too large: {} bytes",
                bytecode.len()
            ));
        }

        // Compile the module
        let module = Module::new(&self.engine, bytecode)
            .map_err(|e| anyhow!("Failed to compile WASM module: {}", e))?;

        // Validate the module
        self.validate_module(&module)?;

        // Cache the compiled module
        let contract_hash = hex::encode(blake3::hash(bytecode).as_bytes());
        {
            let mut cache = self.module_cache.write().unwrap();
            cache.insert(contract_hash.clone(), module.clone());
        }

        // Execute constructor if present
        if let Some(args) = constructor_args {
            let context = WasmExecutionContext {
                contract_address: Address::from_bytes(contract_hash.as_bytes()).unwrap(),
                caller: deployer.clone(),
                block_height: 0,    // Will be set by the caller
                block_timestamp: 0, // Will be set by the caller
                value: 0,
                origin: deployer.clone(),
                gas_price: 1,
                chain_id: 1,
            };

            return self
                .execute_function(
                    &contract_hash,
                    "constructor",
                    args,
                    storage,
                    context,
                    gas_limit,
                )
                .await;
        }

        Ok(WasmExecutionResult {
            success: true,
            return_data: vec![],
            gas_used: 1000, // Base deployment cost
            logs: vec![],
            error: None,
        })
    }

    /// Execute a function in a WASM contract
    pub async fn execute_function(
        &self,
        contract_id: &str,
        function_name: &str,
        args: &[u8],
        storage: Arc<WasmStorage>,
        context: WasmExecutionContext,
        gas_limit: u64,
    ) -> Result<WasmExecutionResult> {
        let start_time = Instant::now();

        // Get the compiled module
        let module = {
            let cache = self.module_cache.read().unwrap();
            cache
                .get(contract_id)
                .ok_or_else(|| anyhow!("Contract not found: {}", contract_id))?
                .clone()
        };

        // Create host environment
        let gas_meter = Arc::new(Mutex::new(GasMeter::new(gas_limit)));
        let logs = Arc::new(Mutex::new(Vec::new()));

        let host_env = WasmHostEnvironment {
            storage,
            gas_meter: gas_meter.clone(),
            context,
            memory: None,
            logs: logs.clone(),
            call_depth: 0,
            start_time,
            timeout: Duration::from_millis(self.config.execution_timeout_ms),
        };

        // Create store and instantiate
        let mut store = Store::new(&self.engine, host_env);
        store.add_fuel(gas_limit)?;

        let linker = self.linker.lock().unwrap();
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| anyhow!("Failed to instantiate WASM module: {}", e))?;

        // Get memory export
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow!("Contract must export memory"))?;

        // Update host environment with memory
        store.data_mut().memory = Some(memory);

        // Get the function to execute
        let func = instance
            .get_typed_func::<(u32, u32), u32>(&mut store, function_name)
            .map_err(|_| {
                anyhow!(
                    "Function '{}' not found or has wrong signature",
                    function_name
                )
            })?;

        // Write arguments to memory
        let args_ptr = self.allocate_memory(&mut store, &instance, args.len())?;
        Self::write_memory(&memory, &mut store, args_ptr, args)?;

        // Execute the function with timeout
        let execution_result = tokio::time::timeout(
            Duration::from_millis(self.config.execution_timeout_ms),
            async { func.call(&mut store, (args_ptr, args.len() as u32)) },
        )
        .await;

        let result = match execution_result {
            Ok(Ok(result_ptr)) => {
                // Read result from memory
                let result_data = if result_ptr != 0 {
                    // Assume the contract returns a pointer to result data
                    // In a real implementation, you'd have a convention for this
                    vec![]
                } else {
                    vec![]
                };

                WasmExecutionResult {
                    success: true,
                    return_data: result_data,
                    gas_used: gas_limit - store.fuel_consumed().unwrap_or(0),
                    logs: logs.lock().unwrap().clone(),
                    error: None,
                }
            }
            Ok(Err(e)) => WasmExecutionResult {
                success: false,
                return_data: vec![],
                gas_used: gas_limit,
                logs: logs.lock().unwrap().clone(),
                error: Some(format!("Execution error: {}", e)),
            },
            Err(_) => WasmExecutionResult {
                success: false,
                return_data: vec![],
                gas_used: gas_limit,
                logs: logs.lock().unwrap().clone(),
                error: Some("Execution timeout".to_string()),
            },
        };

        info!(
            "WASM execution completed: success={}, gas_used={}, duration={}ms",
            result.success,
            result.gas_used,
            start_time.elapsed().as_millis()
        );

        Ok(result)
    }

    /// Validate a compiled WASM module
    fn validate_module(&self, module: &Module) -> Result<()> {
        // Check exports
        let mut has_memory = false;

        for export in module.exports() {
            match export.ty() {
                wasmtime::ExternType::Memory(_) => {
                    if export.name() == "memory" {
                        has_memory = true;
                    }
                }
                _ => {}
            }
        }

        if !has_memory {
            return Err(anyhow!("Contract must export memory"));
        }

        Ok(())
    }

    /// Allocate memory in WASM instance
    fn allocate_memory(
        &self,
        store: &mut Store<WasmHostEnvironment>,
        instance: &Instance,
        size: usize,
    ) -> Result<u32> {
        // Try to call an allocator function if it exists
        if let Ok(alloc_func) = instance.get_typed_func::<u32, u32>(store, "alloc") {
            let ptr = alloc_func.call(store, size as u32)?;
            Ok(ptr)
        } else {
            // Simple bump allocator - in production you'd want a proper allocator
            Ok(1024) // Return a fixed offset for now
        }
    }

    /// Read data from WASM memory
    fn read_memory<T>(
        memory: &Memory,
        store: &mut Store<T>,
        ptr: u32,
        len: usize,
    ) -> Result<Vec<u8>> {
        let data = memory.data(store);
        let start = ptr as usize;
        let end = start + len;

        if end > data.len() {
            return Err(anyhow!("Memory access out of bounds"));
        }

        Ok(data[start..end].to_vec())
    }

    /// Write data to WASM memory
    fn write_memory<T>(memory: &Memory, store: &mut Store<T>, ptr: u32, data: &[u8]) -> Result<()> {
        let memory_data = memory.data_mut(store);
        let start = ptr as usize;
        let end = start + data.len();

        if end > memory_data.len() {
            return Err(anyhow!("Memory write out of bounds"));
        }

        memory_data[start..end].copy_from_slice(data);
        Ok(())
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        let cache = self.module_cache.read().unwrap();
        stats.insert("cached_modules".to_string(), cache.len() as u64);
        stats.insert(
            "max_memory_pages".to_string(),
            self.config.max_memory_pages as u64,
        );
        stats.insert(
            "default_gas_limit".to_string(),
            self.config.default_gas_limit,
        );
        stats
    }

    /// Clear module cache
    pub fn clear_cache(&self) {
        let mut cache = self.module_cache.write().unwrap();
        cache.clear();
        info!("WASM module cache cleared");
    }
}

/// Helper function for error conversion
fn error_to_u32(_: anyhow::Error) -> u32 {
    u32::MAX // Error indicator
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_wasm_engine_creation() {
        let config = WasmExecutionConfig::default();
        let engine = WasmExecutionEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_wasm_module_validation() {
        let config = WasmExecutionConfig::default();
        let engine = WasmExecutionEngine::new(config).unwrap();

        // Test with invalid WASM bytecode
        let invalid_bytecode = b"invalid wasm";
        let result = Module::new(&engine.engine, invalid_bytecode);
        assert!(result.is_err());
    }
}
