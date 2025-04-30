//! WASM contract runtime
//! 
//! Provides the execution environment for WebAssembly smart contracts.
//! Uses Wasmer for WebAssembly execution with controlled memory and 
//! metered execution.

use std::sync::Arc;
use wasmer::{Instance, Module, Store, Value, Function, FunctionType, Type, Memory, MemoryType, Imports};
use wasmer::{imports, WasmerEnv, Global, GlobalType, Mutability};
use wasmer::AsStoreRef;
use thiserror::Error;
use serde::{Serialize, Deserialize};
use log::{debug, warn, error};

use crate::wasm::types::{WasmContractAddress, CallContext, CallParams, CallResult, WasmError};
use crate::wasm::storage::WasmStorage;
use crate::storage::Storage;

/// Amount of gas charged per Wasm instruction
const GAS_PER_INSTRUCTION: u64 = 1;

/// Maximum memory allowed for a contract in pages (64KB per page)
const MAX_MEMORY_PAGES: u32 = 100; // ~6.4MB

/// Maximum allowed execution steps
const MAX_EXECUTION_STEPS: u64 = 10_000_000; // 10 million steps

/// WebAssembly runtime environment shared with host functions
#[derive(Clone)]
pub struct WasmEnv {
    /// Storage access for the contract
    pub storage: Arc<dyn Storage>,
    /// Gas meter for metered execution
    pub gas_meter: Arc<std::sync::Mutex<GasMeter>>,
    /// Call context (caller, block info, etc.)
    pub context: CallContext,
}

/// Gas meter for tracking gas usage during execution
pub struct GasMeter {
    /// Current gas remaining
    pub remaining: u64,
    /// Maximum allowed gas
    pub limit: u64,
    /// Total gas used so far
    pub used: u64,
}

impl GasMeter {
    /// Create a new gas meter with the given limit
    pub fn new(limit: u64) -> Self {
        Self {
            remaining: limit,
            limit,
            used: 0,
        }
    }
    
    /// Use the specified amount of gas and return an error if exceeds available gas
    pub fn use_gas(&mut self, amount: u64) -> Result<(), WasmError> {
        if amount > self.remaining {
            return Err(WasmError::GasLimitExceeded);
        }
        
        self.remaining = self.remaining.saturating_sub(amount);
        self.used = self.used.saturating_add(amount);
        Ok(())
    }
    
    /// Get the total gas used
    pub fn gas_used(&self) -> u64 {
        self.used
    }
}

/// WASM Contract Runtime
#[derive(Clone)]
pub struct WasmRuntime {
    /// Store for Wasmer modules
    store: Store,
    /// Storage system
    storage: Arc<Storage>,
}

impl WasmRuntime {
    /// Create a new WASM runtime
    pub fn new(storage: Arc<Storage>) -> Self {
        let store = Store::default();
        Self { store, storage }
    }
    
    /// Deploy a new WASM contract to the chain
    pub fn deploy_contract(
        &mut self,
        bytecode: &[u8],
        deployer: &crate::types::Address,
        nonce: u64,
        constructor_args: Option<&[u8]>,
    ) -> Result<WasmContractAddress, WasmError> {
        // Validate the WASM module
        self.validate_bytecode(bytecode)?;
        
        // Create contract address
        let contract_address = WasmContractAddress::new(deployer, nonce);
        
        // Create storage wrapper for this contract
        let wasm_storage = Arc::new(WasmStorage::new(self.storage.clone(), &contract_address));
        
        // Store the bytecode
        wasm_storage.store_bytecode(bytecode)
            .map_err(|e| WasmError::StorageError(format!("Failed to store bytecode: {}", e)))?;
        
        // Compile and instantiate to run constructor if provided
        if let Some(args) = constructor_args {
            let context = CallContext {
                contract_address: contract_address.clone(),
                caller: deployer.clone(),
                block_timestamp: 0, // Will be filled in later
                block_height: 0,    // Will be filled in later
                value: 0,
            };
            
            let params = CallParams {
                function: "constructor".to_string(),
                arguments: args.to_vec(),
                gas_limit: 1_000_000, // Standard gas for constructor
            };
            
            let result = self.execute_contract(&contract_address, &context, &params)?;
            if !result.succeeded {
                return Err(WasmError::ExecutionError(
                    result.error.unwrap_or_else(|| "Constructor failed".to_string())
                ));
            }
        }
        
        Ok(contract_address)
    }
    
    /// Execute a function on a deployed WASM contract
    pub fn execute_contract(
        &mut self,
        contract_address: &WasmContractAddress,
        context: &CallContext,
        params: &CallParams,
    ) -> Result<CallResult, WasmError> {
        // Create storage wrapper for this contract
        let wasm_storage = Arc::new(WasmStorage::new(self.storage.clone(), contract_address));
        
        // Check if contract exists
        if !wasm_storage.contract_exists() {
            return Err(WasmError::ExecutionError(format!(
                "Contract does not exist: {}", contract_address
            )));
        }
        
        // Retrieve the bytecode
        let bytecode = wasm_storage.get_bytecode()
            .map_err(|e| WasmError::StorageError(format!("Failed to load bytecode: {}", e)))?;
        
        // Create gas meter
        let gas_meter = Arc::new(std::sync::Mutex::new(GasMeter::new(params.gas_limit)));
        
        // Create environment
        let env = WasmEnv {
            storage: wasm_storage.clone(),
            gas_meter: gas_meter.clone(),
            context: context.clone(),
        };
        
        // Compile the module
        let module = Module::new(&self.store, bytecode)
            .map_err(|e| WasmError::CompilationError(e.to_string()))?;
        
        // Define imports (host functions the contract can call)
        let import_object = self.create_imports(&env)?;
        
        // Instantiate the module
        let instance = Instance::new(&mut self.store, &module, &import_object)
            .map_err(|e| WasmError::InstantiationError(e.to_string()))?;
        
        // Check if the requested function exists
        let func = instance.exports.get_function(&params.function)
            .map_err(|_| WasmError::FunctionNotFound(params.function.clone()))?;
        
        // Prepare the arguments
        let mut args = Vec::new();
        
        // If there are arguments, we need to pass a pointer to the memory where they are stored
        if !params.arguments.is_empty() {
            let memory = instance.exports.get_memory("memory")
                .map_err(|_| WasmError::MemoryError("Contract has no memory export".to_string()))?;
            
            // Allocate memory in the instance
            let allocate_fn = instance.exports.get_function("allocate")
                .map_err(|_| WasmError::FunctionNotFound("allocate".to_string()))?;
            
            let alloc_result = allocate_fn.call(&mut self.store, &[Value::I32(params.arguments.len() as i32)])
                .map_err(|e| WasmError::ExecutionError(format!("Failed to allocate memory: {}", e)))?;
            
            let ptr = match alloc_result[0] {
                Value::I32(ptr) => ptr as u32,
                _ => return Err(WasmError::ExecutionError("Invalid pointer returned from allocate".to_string())),
            };
            
            // Write arguments to memory
            let view = memory.view(&self.store);
            for (i, byte) in params.arguments.iter().enumerate() {
                view.write(ptr as u64 + i as u64, &[*byte])
                    .map_err(|_| WasmError::MemoryError("Failed to write to memory".to_string()))?;
            }
            
            // Pass pointer and length as arguments
            args.push(Value::I32(ptr as i32));
            args.push(Value::I32(params.arguments.len() as i32));
        }
        
        // Call the function
        let result = func.call(&mut self.store, &args)
            .map_err(|e| WasmError::ExecutionError(format!("Function execution failed: {}", e)));
        
        // Get gas used
        let gas_used = gas_meter.lock().unwrap().gas_used();
        
        match result {
            Ok(values) => {
                // Process return values
                let data = if !values.is_empty() {
                    match &values[0] {
                        Value::I32(ptr) => {
                            if *ptr == 0 {
                                // Null pointer returned, treat as empty result
                                None
                            } else {
                                // Read the data from memory at the returned pointer
                                let memory = instance.exports.get_memory("memory")
                                    .map_err(|_| WasmError::MemoryError("Contract has no memory export".to_string()))?;
                                
                                let view = memory.view(&self.store);
                                
                                // Safety check - ensure pointer is within bounds
                                let memory_size = view.data_size() as u64;
                                if *ptr < 0 || (*ptr as u64) >= memory_size {
                                    return Err(WasmError::MemoryError(
                                        format!("Return pointer out of bounds: {} (memory size: {})", ptr, memory_size)
                                    ));
                                }
                                
                                // First 4 bytes at the pointer contain the length of data
                                let mut length_bytes = [0u8; 4];
                                for i in 0..4 {
                                    if (*ptr as u64 + i as u64) >= memory_size {
                                        return Err(WasmError::MemoryError(
                                            "Length bytes exceed memory bounds".to_string()
                                        ));
                                    }
                                    length_bytes[i] = view.read_byte(*ptr as u64 + i as u64)
                                        .map_err(|_| WasmError::MemoryError("Failed to read from memory".to_string()))?;
                                }
                                
                                let length = u32::from_le_bytes(length_bytes) as usize;
                                
                                // Validate length is reasonable
                                const MAX_RETURN_SIZE: usize = 1024 * 1024; // 1MB max return size
                                if length == 0 {
                                    None
                                } else if length > MAX_RETURN_SIZE {
                                    return Err(WasmError::MemoryError(
                                        format!("Return data too large: {} bytes (max: {})", length, MAX_RETURN_SIZE)
                                    ));
                                } else if (*ptr as u64 + 4 + length as u64) > memory_size {
                                    return Err(WasmError::MemoryError(
                                        "Return data would exceed memory bounds".to_string()
                                    ));
                                } else {
                                    // Read the actual data
                                    let mut data = vec![0u8; length];
                                    for i in 0..length {
                                        data[i] = view.read_byte(*ptr as u64 + 4 + i as u64)
                                            .map_err(|_| WasmError::MemoryError("Failed to read from memory".to_string()))?;
                                    }
                                    
                                    Some(data)
                                }
                            }
                        },
                        _ => None,
                    }
                } else {
                    None
                };
                
                Ok(CallResult {
                    data,
                    error: None,
                    gas_used,
                    succeeded: true,
                })
            },
            Err(e) => Ok(CallResult {
                data: None,
                error: Some(e.to_string()),
                gas_used,
                succeeded: false,
            }),
        }
    }
    
    /// Validate WASM bytecode for security
    fn validate_bytecode(&self, bytecode: &[u8]) -> Result<(), WasmError> {
        // TODO: Add more validation
        
        // Check minimum size
        if bytecode.len() < 8 {
            return Err(WasmError::ValidationError("Bytecode too small".to_string()));
        }
        
        // Check WASM magic number
        if &bytecode[0..4] != b"\0asm" {
            return Err(WasmError::ValidationError("Not a WASM module".to_string()));
        }
        
        // Check WASM version
        let version = u32::from_le_bytes([bytecode[4], bytecode[5], bytecode[6], bytecode[7]]);
        if version != 1 {
            return Err(WasmError::ValidationError(format!("Unsupported WASM version: {}", version)));
        }
        
        // TODO: Validate exports (must have memory)
        // TODO: Check for disallowed imports
        // TODO: Static analysis for infinite loops, etc.
        
        Ok(())
    }
    
    /// Create host function imports for the WASM module
    fn create_imports(&mut self, env: &WasmEnv) -> Result<Imports, WasmError> {
        let env_clone = env.clone();
        
        // Storage read function
        let storage_read = Function::new_with_env(
            &mut self.store,
            env_clone.clone(),
            FunctionType::new(vec![Type::I32, Type::I32], vec![Type::I32]),
            move |mut caller, args, _results| {
                let mut gas_meter = env_clone.gas_meter.lock().unwrap();
                gas_meter.use_gas(10)?; // Base cost for storage read
                
                // Get key pointer and length
                let key_ptr = args[0].unwrap_i32() as u32;
                let key_len = args[1].unwrap_i32() as u32;
                
                // Validate inputs
                if key_len == 0 || key_len > 1024 {
                    return Err(WasmError::MemoryError(format!("Invalid key length: {}", key_len)));
                }
                
                // Read key from memory
                let memory = caller.data().as_store_ref().data;
                let view = memory.view(&caller);
                
                // Verify memory bounds
                let memory_size = view.data_size() as u64;
                if (key_ptr as u64 + key_len as u64) > memory_size {
                    return Err(WasmError::MemoryError("Key exceeds memory bounds".to_string()));
                }
                
                let mut key = vec![0u8; key_len as usize];
                for i in 0..key_len {
                    key[i as usize] = view.read_byte(key_ptr as u64 + i as u64)
                        .map_err(|_| WasmError::MemoryError("Failed to read key from memory".to_string()))?;
                }
                
                // Read from storage
                let value_opt = env_clone.storage.read(&key)
                    .map_err(|e| WasmError::StorageError(format!("Failed to read from storage: {}", e)))?;
                
                match value_opt {
                    Some(value) => {
                        // Additional gas cost based on value size
                        gas_meter.use_gas(value.len() as u64 / 100)?;
                        
                        // Allocate memory for the result
                        // Value format: [length(4 bytes)][data]
                        let total_len = 4 + value.len();
                        
                        // Check if the value is too large to return
                        const MAX_RETURN_SIZE: usize = 1024 * 1024; // 1MB max return size
                        if value.len() > MAX_RETURN_SIZE {
                            return Err(WasmError::MemoryError(
                                format!("Value too large: {} bytes (max: {})", value.len(), MAX_RETURN_SIZE)
                            ));
                        }
                        
                        let allocate_fn = caller.get_export("allocate")
                            .ok_or_else(|| WasmError::FunctionNotFound("allocate".to_string()))?
                            .into_function()
                            .map_err(|_| WasmError::FunctionNotFound("allocate is not a function".to_string()))?;
                        
                        let results = allocate_fn.call(&mut caller, &[Value::I32(total_len as i32)])
                            .map_err(|e| WasmError::ExecutionError(format!("Failed to allocate memory: {}", e)))?;
                        
                        let ptr = match results[0] {
                            Value::I32(ptr) => {
                                if ptr <= 0 {
                                    return Err(WasmError::MemoryError(
                                        format!("Invalid pointer returned from allocate: {}", ptr)
                                    ));
                                }
                                ptr as u32
                            },
                            _ => return Err(WasmError::ExecutionError(
                                "Invalid pointer returned from allocate".to_string()
                            )),
                        };
                        
                        // Verify allocated memory is within bounds
                        if (ptr as u64 + total_len as u64) > memory_size {
                            return Err(WasmError::MemoryError(
                                "Allocated memory exceeds memory bounds".to_string()
                            ));
                        }
                        
                        // Write length as first 4 bytes
                        let len_bytes = (value.len() as u32).to_le_bytes();
                        for i in 0..4 {
                            view.write(ptr as u64 + i as u64, &[len_bytes[i]])
                                .map_err(|_| WasmError::MemoryError("Failed to write to memory".to_string()))?;
                        }
                        
                        // Write actual data
                        for (i, byte) in value.iter().enumerate() {
                            view.write(ptr as u64 + 4 + i as u64, &[*byte])
                                .map_err(|_| WasmError::MemoryError("Failed to write to memory".to_string()))?;
                        }
                        
                        // Return pointer to the result
                        Ok(Some(vec![Value::I32(ptr as i32)]))
                    },
                    None => {
                        // Return 0 to indicate key not found
                        Ok(Some(vec![Value::I32(0)]))
                    }
                }
            },
        );
        
        // Storage write function
        let env_clone = env.clone();
        let storage_write = Function::new_with_env(
            &mut self.store,
            env_clone.clone(),
            FunctionType::new(vec![Type::I32, Type::I32, Type::I32, Type::I32], vec![]),
            move |caller, args, _results| {
                let mut gas_meter = env_clone.gas_meter.lock().unwrap();
                gas_meter.use_gas(20)?; // Base cost for storage write
                
                // Get key pointer and length
                let key_ptr = args[0].unwrap_i32() as u32;
                let key_len = args[1].unwrap_i32() as u32;
                
                // Get value pointer and length
                let value_ptr = args[2].unwrap_i32() as u32;
                let value_len = args[3].unwrap_i32() as u32;
                
                // Read key from memory
                let memory = caller.data().as_store_ref().data;
                let view = memory.view(&caller);
                let mut key = vec![0u8; key_len as usize];
                for i in 0..key_len {
                    key[i as usize] = view.read_byte(key_ptr as u64 + i as u64)
                        .map_err(|_| WasmError::MemoryError("Failed to read key from memory".to_string()))?;
                }
                
                // Read value from memory
                let mut value = vec![0u8; value_len as usize];
                for i in 0..value_len {
                    value[i as usize] = view.read_byte(value_ptr as u64 + i as u64)
                        .map_err(|_| WasmError::MemoryError("Failed to read value from memory".to_string()))?;
                }
                
                // Gas cost proportional to data size
                gas_meter.use_gas(value_len as u64 / 100)?;
                
                // Write to storage
                env_clone.storage.write(&key, &value)
                    .map_err(|e| WasmError::StorageError(format!("Failed to write to storage: {}", e)))?;
                
                Ok(None)
            },
        );
        
        // Storage delete function
        let env_clone = env.clone();
        let storage_delete = Function::new_with_env(
            &mut self.store,
            env_clone.clone(),
            FunctionType::new(vec![Type::I32, Type::I32], vec![]),
            move |caller, args, _results| {
                let mut gas_meter = env_clone.gas_meter.lock().unwrap();
                gas_meter.use_gas(10)?; // Base cost for storage delete
                
                // Get key pointer and length
                let key_ptr = args[0].unwrap_i32() as u32;
                let key_len = args[1].unwrap_i32() as u32;
                
                // Read key from memory
                let memory = caller.data().as_store_ref().data;
                let view = memory.view(&caller);
                let mut key = vec![0u8; key_len as usize];
                for i in 0..key_len {
                    key[i as usize] = view.read_byte(key_ptr as u64 + i as u64)
                        .map_err(|_| WasmError::MemoryError("Failed to read key from memory".to_string()))?;
                }
                
                // Delete from storage
                env_clone.storage.delete(&key)
                    .map_err(|e| WasmError::StorageError(format!("Failed to delete from storage: {}", e)))?;
                
                Ok(None)
            },
        );
        
        // Get blockchain information
        let env_clone = env.clone();
        let get_context = Function::new_with_env(
            &mut self.store,
            env_clone.clone(),
            FunctionType::new(vec![], vec![Type::I64, Type::I64]),
            move |_caller, _args, results| {
                let gas_meter = env_clone.gas_meter.lock().unwrap();
                gas_meter.use_gas(1)?; // Minimal cost
                
                // Return block height and timestamp
                results[0] = Value::I64(env_clone.context.block_height as i64);
                results[1] = Value::I64(env_clone.context.block_timestamp as i64);
                
                Ok(None)
            },
        );
        
        // Get caller information
        let env_clone = env.clone();
        let get_caller = Function::new_with_env(
            &mut self.store,
            env_clone.clone(),
            FunctionType::new(vec![], vec![Type::I32]),
            move |mut caller, _args, _results| {
                let gas_meter = env_clone.gas_meter.lock().unwrap();
                gas_meter.use_gas(1)?; // Minimal cost
                
                // Convert caller address to bytes
                let caller_bytes = env_clone.context.caller.as_bytes();
                
                // Allocate memory for the result
                let allocate_fn = caller.get_export("allocate")
                    .ok_or_else(|| WasmError::FunctionNotFound("allocate".to_string()))?
                    .into_function()
                    .map_err(|_| WasmError::FunctionNotFound("allocate is not a function".to_string()))?;
                
                let results = allocate_fn.call(&mut caller, &[Value::I32(caller_bytes.len() as i32)])
                    .map_err(|e| WasmError::ExecutionError(format!("Failed to allocate memory: {}", e)))?;
                
                let ptr = match results[0] {
                    Value::I32(ptr) => {
                        if ptr <= 0 {
                            return Err(WasmError::MemoryError(
                                format!("Invalid pointer returned from allocate: {}", ptr)
                            ));
                        }
                        ptr as u32
                    },
                    _ => return Err(WasmError::ExecutionError(
                        "Invalid pointer returned from allocate".to_string()
                    )),
                };
                
                // Verify memory bounds
                let memory = caller.data().as_store_ref().data;
                let view = memory.view(&caller);
                let memory_size = view.data_size() as u64;
                
                if (ptr as u64 + caller_bytes.len() as u64) > memory_size {
                    return Err(WasmError::MemoryError(
                        "Allocated memory exceeds memory bounds".to_string()
                    ));
                }
                
                // Write caller address to memory
                for (i, byte) in caller_bytes.iter().enumerate() {
                    view.write(ptr as u64 + i as u64, &[*byte])
                        .map_err(|_| WasmError::MemoryError("Failed to write to memory".to_string()))?;
                }
                
                // Return pointer to the result
                Ok(Some(vec![Value::I32(ptr as i32)]))
            },
        );
        
        // Get value sent with the transaction
        let env_clone = env.clone();
        let get_value = Function::new_with_env(
            &mut self.store,
            env_clone.clone(),
            FunctionType::new(vec![], vec![Type::I64]),
            move |_caller, _args, results| {
                let gas_meter = env_clone.gas_meter.lock().unwrap();
                gas_meter.use_gas(1)?; // Minimal cost
                
                // Return value sent with the transaction
                results[0] = Value::I64(env_clone.context.value as i64);
                
                Ok(None)
            },
        );
        
        // Create the import object
        let import_object = imports! {
            "env" => {
                "storage_read" => storage_read,
                "storage_write" => storage_write, 
                "storage_delete" => storage_delete,
                "get_context" => get_context,
                "get_caller" => get_caller,
                "get_value" => get_value,
            }
        };
        
        Ok(import_object)
    }
} 