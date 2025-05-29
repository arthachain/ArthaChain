//! WASM execution engine implementation
//!
//! This module provides the core execution environment for WASM smart contracts
//! using the Wasmer runtime.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use wasmer::{
    imports, AsStoreRef, Function, FunctionEnv, Instance, Memory, Module, RuntimeError, Store,
    TypedFunction, Value, WasmPtr,
};
use wasmer_middlewares::metering::{get_remaining_points, set_remaining_points, MeteringPoints};
use wasmer_vm::trampoline::StoreObjects;

use crate::wasm::storage::WasmStorage;
use crate::wasm::types::{WasmConfig, WasmContractAddress, WasmError, WasmExecutionResult};

/// Gas metering configuration
pub struct GasConfig {
    /// Cost per storage read
    pub storage_read_cost: u64,
    /// Cost per storage write
    pub storage_write_cost: u64,
    /// Base instruction cost
    pub instruction_cost: u64,
    /// Gas limit
    pub gas_limit: u64,
}

impl Default for GasConfig {
    fn default() -> Self {
        Self {
            storage_read_cost: 10,
            storage_write_cost: 100,
            instruction_cost: 1,
            gas_limit: 1_000_000,
        }
    }
}

/// Function environment for WASM execution
pub struct WasmEnv {
    /// Storage interface
    pub storage: Arc<Mutex<WasmStorage>>,
    /// Contract address
    pub contract_address: WasmContractAddress,
    /// Logs collected during execution
    pub logs: Arc<Mutex<Vec<String>>>,
    /// Start time of execution
    pub start_time: Instant,
    /// Execution timeout
    pub timeout: Duration,
    /// Gas configuration
    pub gas_config: GasConfig,
}

/// WASM Execution Engine
pub struct WasmEngine {
    /// Wasmer store
    store: Store,
    /// Contract modules
    modules: std::collections::HashMap<WasmContractAddress, Module>,
    /// WASM configuration
    config: WasmConfig,
}

impl WasmEngine {
    /// Create a new WASM engine
    pub fn new(config: WasmConfig) -> Result<Self, WasmError> {
        let compiler = wasmer::Singlepass::new();
        let store = wasmer::Store::new_with_engine(wasmer::Engine::new(&compiler));

        Ok(Self {
            store,
            modules: std::collections::HashMap::new(),
            config,
        })
    }

    /// Load a contract from WASM bytes
    pub fn load_contract(
        &mut self,
        address: WasmContractAddress,
        wasm_bytes: &[u8],
    ) -> Result<(), WasmError> {
        let module = Module::new(&self.store, wasm_bytes)
            .map_err(|e| WasmError::CompilationError(e.to_string()))?;

        self.modules.insert(address, module);
        Ok(())
    }

    /// Execute a contract method
    pub fn execute(
        &mut self,
        address: &WasmContractAddress,
        method: &str,
        args: &[Value],
        storage: Arc<Mutex<WasmStorage>>,
        gas_limit: Option<u64>,
    ) -> Result<WasmExecutionResult, WasmError> {
        let module = self.modules.get(address).ok_or_else(|| {
            WasmError::InvalidContract(format!("Contract not loaded: {}", address))
        })?;

        let logs = Arc::new(Mutex::new(Vec::new()));
        let start_time = Instant::now();

        let gas_config = GasConfig {
            storage_read_cost: self.config.storage_read_gas_cost,
            storage_write_cost: self.config.storage_write_gas_cost,
            instruction_cost: self.config.execution_gas_cost,
            gas_limit: gas_limit.unwrap_or(self.config.gas_limit),
        };

        let env = FunctionEnv::new(
            &mut self.store,
            WasmEnv {
                storage: storage.clone(),
                contract_address: address.clone(),
                logs: logs.clone(),
                start_time,
                timeout: Duration::from_millis(self.config.execution_timeout),
                gas_config,
            },
        );

        // Create import objects with host functions
        let import_object = imports! {
            "env" => {
                "storage_read" => Function::new_typed_with_env(&mut self.store, &env, storage_read),
                "storage_write" => Function::new_typed_with_env(&mut self.store, &env, storage_write),
                "storage_delete" => Function::new_typed_with_env(&mut self.store, &env, storage_delete),
                "log_message" => Function::new_typed_with_env(&mut self.store, &env, log_message),
                "get_caller" => Function::new_typed_with_env(&mut self.store, &env, get_caller),
                "get_contract_address" => Function::new_typed_with_env(&mut self.store, &env, get_contract_address),
            }
        };

        // Add metering middleware for gas calculation
        let mut module = module.clone();
        let instance = Instance::new(&mut self.store, &module, &import_object)
            .map_err(|e| WasmError::InstantiationError(e.to_string()))?;

        // Set initial gas
        set_remaining_points(&mut self.store, gas_config.gas_limit);

        // Get the function
        let function = instance
            .exports
            .get_function(method)
            .map_err(|_| WasmError::InvalidFunction(format!("Method not found: {}", method)))?;

        // Execute the function
        let result = function.call(&mut self.store, args);

        // Check gas remaining
        let gas_used = gas_config.gas_limit - get_remaining_points(&self.store);

        // Add storage reads/writes to gas used
        let storage_guard = storage.lock().unwrap();
        let storage_gas = storage_guard.get_reads() * gas_config.storage_read_cost
            + storage_guard.get_writes() * gas_config.storage_write_cost;
        let total_gas_used = gas_used + storage_gas;

        // Get logs
        let logs_vec = logs.lock().unwrap().clone();

        match result {
            Ok(ret_values) => {
                let return_data = if ret_values.is_empty() {
                    None
                } else {
                    match &ret_values[0] {
                        Value::I32(val) => Some(val.to_le_bytes().to_vec()),
                        Value::I64(val) => Some(val.to_le_bytes().to_vec()),
                        Value::F32(val) => Some(val.to_le_bytes().to_vec()),
                        Value::F64(val) => Some(val.to_le_bytes().to_vec()),
                        _ => None,
                    }
                };

                Ok(WasmExecutionResult::success(
                    return_data,
                    total_gas_used,
                    logs_vec,
                ))
            }
            Err(e) => {
                if e.to_string().contains("gas limit") {
                    return Ok(WasmExecutionResult::failure(
                        "Gas limit exceeded".to_string(),
                        total_gas_used,
                        logs_vec,
                    ));
                }

                if start_time.elapsed() > Duration::from_millis(self.config.execution_timeout) {
                    return Ok(WasmExecutionResult::failure(
                        "Execution timeout".to_string(),
                        total_gas_used,
                        logs_vec,
                    ));
                }

                Ok(WasmExecutionResult::failure(
                    format!("Execution error: {}", e),
                    total_gas_used,
                    logs_vec,
                ))
            }
        }
    }

    /// Get available contracts
    pub fn get_contracts(&self) -> Vec<WasmContractAddress> {
        self.modules.keys().cloned().collect()
    }

    /// Check if a contract exists
    pub fn has_contract(&self, address: &WasmContractAddress) -> bool {
        self.modules.contains_key(address)
    }
}

// Host functions exposed to WASM contracts

fn storage_read(
    env: FunctionEnv<WasmEnv>,
    key_ptr: WasmPtr<u8>,
    key_len: u32,
    value_ptr: WasmPtr<u8>,
    value_len: u32,
) -> i32 {
    let env = env.as_ref();
    let mut storage = env.storage.lock().unwrap();

    // Check timeout
    if env.start_time.elapsed() > env.timeout {
        return -2; // Timeout error
    }

    // Get memory from instance
    let memory = env.data().memory.unwrap();

    // Read key from memory
    let key = match read_memory_string(&memory, key_ptr, key_len) {
        Ok(k) => k,
        Err(_) => return -1,
    };

    // Read value from storage
    match storage.read(key.as_bytes()) {
        Some(value) => {
            if value.len() > value_len as usize {
                return -3; // Buffer too small
            }

            // Write value to memory
            if let Err(_) = write_memory(&memory, value_ptr, &value) {
                return -1;
            }

            value.len() as i32
        }
        None => 0, // Key not found
    }
}

fn storage_write(
    env: FunctionEnv<WasmEnv>,
    key_ptr: WasmPtr<u8>,
    key_len: u32,
    value_ptr: WasmPtr<u8>,
    value_len: u32,
) -> i32 {
    let env = env.as_ref();
    let mut storage = env.storage.lock().unwrap();

    // Check timeout
    if env.start_time.elapsed() > env.timeout {
        return -2; // Timeout error
    }

    // Get memory from instance
    let memory = env.data().memory.unwrap();

    // Read key from memory
    let key = match read_memory_string(&memory, key_ptr, key_len) {
        Ok(k) => k,
        Err(_) => return -1,
    };

    // Read value from memory
    let value = match read_memory(&memory, value_ptr, value_len) {
        Ok(v) => v,
        Err(_) => return -1,
    };

    // Write to storage
    storage.write(key.as_bytes(), &value);

    0 // Success
}

fn storage_delete(env: FunctionEnv<WasmEnv>, key_ptr: WasmPtr<u8>, key_len: u32) -> i32 {
    let env = env.as_ref();
    let mut storage = env.storage.lock().unwrap();

    // Check timeout
    if env.start_time.elapsed() > env.timeout {
        return -2; // Timeout error
    }

    // Get memory from instance
    let memory = env.data().memory.unwrap();

    // Read key from memory
    let key = match read_memory_string(&memory, key_ptr, key_len) {
        Ok(k) => k,
        Err(_) => return -1,
    };

    // Delete from storage
    storage.delete(key.as_bytes());

    0 // Success
}

fn log_message(env: FunctionEnv<WasmEnv>, msg_ptr: WasmPtr<u8>, msg_len: u32) -> i32 {
    let env = env.as_ref();

    // Check timeout
    if env.start_time.elapsed() > env.timeout {
        return -2; // Timeout error
    }

    // Get memory from instance
    let memory = env.data().memory.unwrap();

    // Read message from memory
    let message = match read_memory_string(&memory, msg_ptr, msg_len) {
        Ok(m) => m,
        Err(_) => return -1,
    };

    // Add to logs
    let mut logs = env.logs.lock().unwrap();
    logs.push(message);

    0 // Success
}

fn get_caller(env: FunctionEnv<WasmEnv>, out_ptr: WasmPtr<u8>, out_len: u32) -> i32 {
    let env = env.as_ref();

    // Check timeout
    if env.start_time.elapsed() > env.timeout {
        return -2; // Timeout error
    }

    // Get memory from instance
    let memory = env.data().memory.unwrap();

    // In a real implementation, we would get the actual caller
    // For now, we use a dummy caller
    let caller = "wasm:0000000000000000000000000000000000000000000000000000000000000000";

    if caller.len() > out_len as usize {
        return -3; // Buffer too small
    }

    // Write caller to memory
    if let Err(_) = write_memory(&memory, out_ptr, caller.as_bytes()) {
        return -1;
    }

    caller.len() as i32
}

fn get_contract_address(env: FunctionEnv<WasmEnv>, out_ptr: WasmPtr<u8>, out_len: u32) -> i32 {
    let env = env.as_ref();

    // Check timeout
    if env.start_time.elapsed() > env.timeout {
        return -2; // Timeout error
    }

    // Get memory from instance
    let memory = env.data().memory.unwrap();

    // Get contract address
    let address = env.contract_address.to_string();

    if address.len() > out_len as usize {
        return -3; // Buffer too small
    }

    // Write address to memory
    if let Err(_) = write_memory(&memory, out_ptr, address.as_bytes()) {
        return -1;
    }

    address.len() as i32
}

// Helper functions for memory access

fn read_memory(memory: &Memory, ptr: WasmPtr<u8>, len: u32) -> Result<Vec<u8>, RuntimeError> {
    let view = memory.view();
    let offset = ptr.offset() as usize;

    if offset + len as usize > view.data_size() {
        return Err(RuntimeError::new("Memory access out of bounds"));
    }

    let mut buffer = vec![0u8; len as usize];
    for i in 0..len as usize {
        buffer[i] = view.data_ptr().add(offset + i).read();
    }

    Ok(buffer)
}

fn read_memory_string(memory: &Memory, ptr: WasmPtr<u8>, len: u32) -> Result<String, RuntimeError> {
    let buffer = read_memory(memory, ptr, len)?;
    String::from_utf8(buffer).map_err(|_| RuntimeError::new("Invalid UTF-8 string"))
}

fn write_memory(memory: &Memory, ptr: WasmPtr<u8>, data: &[u8]) -> Result<(), RuntimeError> {
    let view = memory.view();
    let offset = ptr.offset() as usize;

    if offset + data.len() > view.data_size() {
        return Err(RuntimeError::new("Memory access out of bounds"));
    }

    for (i, &byte) in data.iter().enumerate() {
        view.data_ptr().add(offset + i).write(byte);
    }

    Ok(())
}
