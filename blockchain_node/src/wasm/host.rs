//! WASM host functions
//!
//! Provides host functions and utilities that are exposed to WebAssembly contracts,
//! allowing them to interact with the blockchain environment.

use crate::types::Address;
use crate::wasm::types::WasmError;
use std::sync::Arc;
use wasmer::{Memory, Memory32View, WasmPtr};

use crate::wasm::{gas::GasMeter, storage::WasmStorage, types::CallContext};
use wasmer::{Function, FunctionEnv, Store, WasmerEnv};

/// Maximum memory read/write size
pub const MAX_MEMORY_ACCESS_SIZE: usize = 1 * 1024 * 1024; // 1MB

/// Environment for host functions
#[derive(WasmerEnv, Clone)]
pub struct HostEnv {
    /// Storage interface
    pub storage: Arc<WasmStorage>,
    /// Gas meter for tracking gas usage
    pub gas_meter: Arc<GasMeter>,
    /// Call context containing blockchain information
    pub context: CallContext,
}

impl HostEnv {
    /// Create a new host environment
    pub fn new(storage: Arc<WasmStorage>, gas_meter: Arc<GasMeter>, context: CallContext) -> Self {
        Self {
            storage,
            gas_meter,
            context,
        }
    }
}

/// Register host functions for a WASM module
pub fn register_host_functions(
    store: &mut Store,
    env: &FunctionEnv<HostEnv>,
) -> Result<Vec<(String, Function)>, WasmError> {
    let mut functions = Vec::new();

    // Storage functions
    functions.push((
        "storage_read".to_string(),
        Function::new_typed_with_env(store, env, storage_read),
    ));

    functions.push((
        "storage_write".to_string(),
        Function::new_typed_with_env(store, env, storage_write),
    ));

    functions.push((
        "storage_delete".to_string(),
        Function::new_typed_with_env(store, env, storage_delete),
    ));

    // Context functions
    functions.push((
        "get_caller".to_string(),
        Function::new_typed_with_env(store, env, get_caller),
    ));

    functions.push((
        "get_block_number".to_string(),
        Function::new_typed_with_env(store, env, get_block_number),
    ));

    functions.push((
        "get_block_timestamp".to_string(),
        Function::new_typed_with_env(store, env, get_block_timestamp),
    ));

    // Debug functions
    functions.push((
        "debug_log".to_string(),
        Function::new_typed_with_env(store, env, debug_log),
    ));

    Ok(functions)
}

/// Read a byte array from WASM memory
pub fn read_memory_bytes(
    memory: &Memory,
    ptr: WasmPtr<u8>,
    len: u32,
) -> Result<Vec<u8>, WasmError> {
    if len == 0 {
        return Ok(Vec::new());
    }

    if len as usize > MAX_MEMORY_ACCESS_SIZE {
        return Err(WasmError::MemoryError(format!(
            "Requested memory read too large: {} bytes",
            len
        )));
    }

    let view: Memory32View = memory.view();
    let offset = ptr.offset() as usize;

    // Check if the memory access is within bounds
    if offset + (len as usize) > view.data_size() {
        return Err(WasmError::MemoryError(format!(
            "Memory access out of bounds: offset={}, len={}, size={}",
            offset,
            len,
            view.data_size()
        )));
    }

    let mut result = vec![0u8; len as usize];
    for i in 0..len as usize {
        result[i] = unsafe { view.data_unchecked_mut()[offset + i] };
    }

    Ok(result)
}

/// Write a byte array to WASM memory
pub fn write_memory_bytes(memory: &Memory, ptr: WasmPtr<u8>, data: &[u8]) -> Result<(), WasmError> {
    if data.is_empty() {
        return Ok(());
    }

    if data.len() > MAX_MEMORY_ACCESS_SIZE {
        return Err(WasmError::MemoryError(format!(
            "Requested memory write too large: {} bytes",
            data.len()
        )));
    }

    let view = memory.view();
    let offset = ptr.offset() as usize;

    // Check if the memory access is within bounds
    if offset + data.len() > view.data_size() {
        return Err(WasmError::MemoryError(format!(
            "Memory access out of bounds: offset={}, len={}, size={}",
            offset,
            data.len(),
            view.data_size()
        )));
    }

    for (i, &byte) in data.iter().enumerate() {
        unsafe {
            view.data_unchecked_mut()[offset + i] = byte;
        }
    }

    Ok(())
}

/// Read a string from WASM memory
pub fn read_memory_string(
    memory: &Memory,
    ptr: WasmPtr<u8>,
    len: u32,
) -> Result<String, WasmError> {
    let bytes = read_memory_bytes(memory, ptr, len)?;

    String::from_utf8(bytes)
        .map_err(|e| WasmError::MemoryError(format!("Invalid UTF-8 string: {}", e)))
}

/// Write a string to WASM memory
pub fn write_memory_string(memory: &Memory, ptr: WasmPtr<u8>, s: &str) -> Result<(), WasmError> {
    write_memory_bytes(memory, ptr, s.as_bytes())
}

/// Write a 32-bit integer to WASM memory
pub fn write_memory_i32(memory: &Memory, ptr: WasmPtr<u8>, value: i32) -> Result<(), WasmError> {
    let bytes = value.to_le_bytes();
    write_memory_bytes(memory, ptr, &bytes)
}

/// Write a 64-bit integer to WASM memory
pub fn write_memory_i64(memory: &Memory, ptr: WasmPtr<u8>, value: i64) -> Result<(), WasmError> {
    let bytes = value.to_le_bytes();
    write_memory_bytes(memory, ptr, &bytes)
}

/// Create a hash of arbitrary data
pub fn keccak256_hash(data: &[u8]) -> [u8; 32] {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let result = hasher.finalize();

    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

/// Create a cryptographic signature
pub fn crypto_sign(private_key: &[u8], message: &[u8]) -> Result<Vec<u8>, WasmError> {
    // This is a placeholder - in a real implementation, this would use the actual
    // cryptographic signing algorithm used by the blockchain
    Err(WasmError::ExecutionError(
        "Cryptographic signing not implemented for contracts".to_string(),
    ))
}

/// Verify a cryptographic signature
pub fn crypto_verify(
    public_key: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<bool, WasmError> {
    // This is a placeholder - in a real implementation, this would use the actual
    // cryptographic verification algorithm used by the blockchain
    Err(WasmError::ExecutionError(
        "Cryptographic verification not implemented for contracts".to_string(),
    ))
}

/// Convert a hex string to bytes
pub fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, WasmError> {
    hex::decode(hex.trim_start_matches("0x"))
        .map_err(|e| WasmError::ExecutionError(format!("Invalid hex string: {}", e)))
}

/// Convert bytes to a hex string
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(bytes))
}

/// Encode an address to a standard format
pub fn encode_address(address: &Address) -> String {
    address.to_string()
}

/// Decode an address from a standard format
pub fn decode_address(address_str: &str) -> Result<Address, WasmError> {
    Address::from_string(address_str)
        .map_err(|_| WasmError::ExecutionError(format!("Invalid address format: {}", address_str)))
}

// Storage host functions

/// Read data from storage
fn storage_read(
    env: FunctionEnv<HostEnv>,
    key_ptr: u32,
    key_len: u32,
    value_ptr: u32,
    value_len: u32,
) -> i32 {
    // This would involve:
    // 1. Reading the key from WASM memory
    // 2. Looking up the value in storage
    // 3. Writing the value to WASM memory

    let env_ref = env.data();
    let memory = env_ref.memory.clone();

    // Track gas for read operation
    if let Err(e) = env_ref.gas_meter.track_gas(Gas::storage_read(key_len)) {
        log::error!("Storage read out of gas: {}", e);
        return -1;
    }

    // Read the key from WASM memory
    let key = match read_memory_bytes(&memory, WasmPtr::new(key_ptr), key_len) {
        Ok(k) => k,
        Err(e) => {
            log::error!("Failed to read key from memory: {}", e);
            return -1;
        }
    };

    // Look up the value in storage
    let value = match env_ref.storage.get(&key) {
        Ok(Some(v)) => v,
        Ok(None) => {
            // Key not found
            return 0;
        }
        Err(e) => {
            log::error!("Storage read failed: {}", e);
            return -1;
        }
    };

    // Check if the provided buffer is large enough
    if value.len() > value_len as usize {
        // Return the required size
        return value.len() as i32;
    }

    // Write value to WASM memory
    if let Err(e) = write_memory_bytes(&memory, WasmPtr::new(value_ptr), &value) {
        log::error!("Failed to write value to memory: {}", e);
        return -1;
    }

    // Return the number of bytes written
    value.len() as i32
}

/// Write data to storage
fn storage_write(
    env: FunctionEnv<HostEnv>,
    key_ptr: u32,
    key_len: u32,
    value_ptr: u32,
    value_len: u32,
) -> i32 {
    // This would involve:
    // 1. Reading the key from WASM memory
    // 2. Reading the value from WASM memory
    // 3. Writing the key-value pair to storage

    let env_ref = env.data();
    let memory = env_ref.memory.clone();

    // Track gas for write operation
    if let Err(e) = env_ref
        .gas_meter
        .track_gas(Gas::storage_write(key_len, value_len))
    {
        log::error!("Storage write out of gas: {}", e);
        return -1;
    }

    // Read the key from WASM memory
    let key = match read_memory_bytes(&memory, WasmPtr::new(key_ptr), key_len) {
        Ok(k) => k,
        Err(e) => {
            log::error!("Failed to read key from memory: {}", e);
            return -1;
        }
    };

    // Read the value from WASM memory
    let value = match read_memory_bytes(&memory, WasmPtr::new(value_ptr), value_len) {
        Ok(v) => v,
        Err(e) => {
            log::error!("Failed to read value from memory: {}", e);
            return -1;
        }
    };

    // Write to storage
    match env_ref.storage.set(&key, &value) {
        Ok(_) => 1, // Success
        Err(e) => {
            log::error!("Storage write failed: {}", e);
            -1
        }
    }
}

/// Delete data from storage
fn storage_delete(env: FunctionEnv<HostEnv>, key_ptr: u32, key_len: u32) -> i32 {
    // This would involve:
    // 1. Reading the key from WASM memory
    // 2. Deleting the key-value pair from storage

    let env_ref = env.data();
    let memory = env_ref.memory.clone();

    // Track gas for delete operation
    if let Err(e) = env_ref.gas_meter.track_gas(Gas::storage_delete(key_len)) {
        log::error!("Storage delete out of gas: {}", e);
        return -1;
    }

    // Read the key from WASM memory
    let key = match read_memory_bytes(&memory, WasmPtr::new(key_ptr), key_len) {
        Ok(k) => k,
        Err(e) => {
            log::error!("Failed to read key from memory: {}", e);
            return -1;
        }
    };

    // Delete from storage
    match env_ref.storage.delete(&key) {
        Ok(_) => 1, // Success
        Err(e) => {
            log::error!("Storage delete failed: {}", e);
            -1
        }
    }
}

/// Get the caller's address
fn get_caller(env: FunctionEnv<HostEnv>, ptr: u32, len: u32) -> i32 {
    // This would involve:
    // 1. Getting the caller address from the context
    // 2. Writing the address to WASM memory

    let env_ref = env.data();
    let memory = env_ref.memory.clone();

    // Track gas for context operation
    if let Err(e) = env_ref.gas_meter.track_gas(Gas::context_operation()) {
        log::error!("Get caller out of gas: {}", e);
        return -1;
    }

    // Get caller address from context
    let caller = &env_ref.context.caller;
    let caller_bytes = caller.as_bytes();

    // Check if buffer is large enough
    if caller_bytes.len() > len as usize {
        return caller_bytes.len() as i32;
    }

    // Write address to WASM memory
    match write_memory_bytes(&memory, WasmPtr::new(ptr), caller_bytes) {
        Ok(_) => caller_bytes.len() as i32, // Return bytes written
        Err(e) => {
            log::error!("Failed to write caller address to memory: {}", e);
            -1
        }
    }
}

/// Get the current block number
fn get_block_number(env: FunctionEnv<HostEnv>) -> u64 {
    // This would involve getting the block number from the context

    let env_ref = env.data();

    // Track gas for context operation
    if let Err(e) = env_ref.gas_meter.track_gas(Gas::context_operation()) {
        log::error!("Get block number out of gas: {}", e);
        return 0;
    }

    // Return the block number from context
    env_ref.context.block_number
}

/// Get the current block timestamp
fn get_block_timestamp(env: FunctionEnv<HostEnv>) -> u64 {
    // This would involve getting the block timestamp from the context

    let env_ref = env.data();

    // Track gas for context operation
    if let Err(e) = env_ref.gas_meter.track_gas(Gas::context_operation()) {
        log::error!("Get block timestamp out of gas: {}", e);
        return 0;
    }

    // Return the block timestamp from context
    env_ref.context.block_timestamp
}

// Debug host functions

/// Log a debug message from the contract
fn debug_log(env: FunctionEnv<HostEnv>, ptr: u32, len: u32) {
    // This would involve:
    // 1. Reading the message from WASM memory
    // 2. Logging the message

    let env_ref = env.data();
    let memory = env_ref.memory.clone();

    // Skip gas tracking for debug operations

    // Read message from WASM memory
    let message = match read_memory_string(&memory, WasmPtr::new(ptr), len) {
        Ok(msg) => msg,
        Err(e) => {
            log::error!("Failed to read debug message from memory: {}", e);
            return;
        }
    };

    // Log the message with contract address for context
    log::debug!(
        "[Contract {}] {}",
        env_ref.context.contract_address,
        message
    );
}

// Host function interface for WASM contracts
//
// Implements the host functions that are exposed to WASM smart contracts.
// These functions allow the contracts to interact with the blockchain environment.

use crate::wasm::{runtime::WasmEnv, types::HostFunctionCallback};
use wasmer::{Function, FunctionType, ImportObject, Store, Type, Value};

/// Create an import object with all host functions for the WASM module
pub fn create_import_object(store: &Store, env: Arc<WasmEnv>) -> ImportObject {
    let mut import_object = ImportObject::new();

    // Storage functions
    register_function(
        store,
        &mut import_object,
        "env",
        "storage_read",
        storage_read_fn(store, env.clone()),
    );
    register_function(
        store,
        &mut import_object,
        "env",
        "storage_write",
        storage_write_fn(store, env.clone()),
    );
    register_function(
        store,
        &mut import_object,
        "env",
        "storage_delete",
        storage_delete_fn(store, env.clone()),
    );
    register_function(
        store,
        &mut import_object,
        "env",
        "storage_has",
        storage_has_fn(store, env.clone()),
    );

    // Context functions
    register_function(
        store,
        &mut import_object,
        "env",
        "get_caller",
        get_caller_fn(store, env.clone()),
    );
    register_function(
        store,
        &mut import_object,
        "env",
        "get_block_height",
        get_block_height_fn(store, env.clone()),
    );
    register_function(
        store,
        &mut import_object,
        "env",
        "get_block_timestamp",
        get_block_timestamp_fn(store, env.clone()),
    );
    register_function(
        store,
        &mut import_object,
        "env",
        "get_contract_address",
        get_contract_address_fn(store, env.clone()),
    );

    // Memory allocation helpers
    register_function(
        store,
        &mut import_object,
        "env",
        "alloc",
        alloc_fn(store, env.clone()),
    );
    register_function(
        store,
        &mut import_object,
        "env",
        "dealloc",
        dealloc_fn(store, env.clone()),
    );

    import_object
}

/// Register a function in the import object
fn register_function(
    store: &Store,
    import_object: &mut ImportObject,
    namespace: &str,
    name: &str,
    function: Function,
) {
    import_object.register(namespace, name, function);
}

/// Read from storage and return data to WASM
fn storage_read_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![Type::I32, Type::I32], vec![Type::I32]);
    Function::new(store, &signature, move |args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_STORAGE_READ)?;

        // Extract key pointer and length
        let key_ptr = args[0].unwrap_i32() as u32;
        let key_len = args[1].unwrap_i32() as u32;

        // Read key from memory
        let key = env.read_memory(key_ptr, key_len)?;

        // Read from storage
        let value = env
            .storage()
            .get(&env.contract_address(), &key)
            .unwrap_or_default();

        // Write value to memory and return pointer to it
        let ptr = env.write_to_memory(&value)?;

        Ok(vec![Value::I32(ptr as i32)])
    })
}

/// Write to storage from WASM
fn storage_write_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![Type::I32, Type::I32, Type::I32, Type::I32], vec![]);
    Function::new(store, &signature, move |args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_STORAGE_WRITE)?;

        // Extract key and value pointers and lengths
        let key_ptr = args[0].unwrap_i32() as u32;
        let key_len = args[1].unwrap_i32() as u32;
        let val_ptr = args[2].unwrap_i32() as u32;
        let val_len = args[3].unwrap_i32() as u32;

        // Read key and value from memory
        let key = env.read_memory(key_ptr, key_len)?;
        let value = env.read_memory(val_ptr, val_len)?;

        // Write to storage
        env.storage().set(&env.contract_address(), &key, &value);

        Ok(vec![])
    })
}

/// Delete from storage
fn storage_delete_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![Type::I32, Type::I32], vec![]);
    Function::new(store, &signature, move |args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_STORAGE_DELETE)?;

        // Extract key pointer and length
        let key_ptr = args[0].unwrap_i32() as u32;
        let key_len = args[1].unwrap_i32() as u32;

        // Read key from memory
        let key = env.read_memory(key_ptr, key_len)?;

        // Delete from storage
        env.storage().delete(&env.contract_address(), &key);

        Ok(vec![])
    })
}

/// Check if key exists in storage
fn storage_has_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![Type::I32, Type::I32], vec![Type::I32]);
    Function::new(store, &signature, move |args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_STORAGE_READ)?;

        // Extract key pointer and length
        let key_ptr = args[0].unwrap_i32() as u32;
        let key_len = args[1].unwrap_i32() as u32;

        // Read key from memory
        let key = env.read_memory(key_ptr, key_len)?;

        // Check storage
        let exists = env.storage().has(&env.contract_address(), &key);

        Ok(vec![Value::I32(exists as i32)])
    })
}

/// Get the caller address
fn get_caller_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![], vec![Type::I32]);
    Function::new(store, &signature, move |_args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_CONTEXT_READ)?;

        // Get caller address
        let caller = env.caller().to_string();

        // Write to memory and return pointer
        let ptr = env.write_to_memory(caller.as_bytes())?;

        Ok(vec![Value::I32(ptr as i32)])
    })
}

/// Get current block height
fn get_block_height_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![], vec![Type::I64]);
    Function::new(store, &signature, move |_args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_CONTEXT_READ)?;

        // Get block height
        let height = env.block_height();

        Ok(vec![Value::I64(height as i64)])
    })
}

/// Get current block timestamp
fn get_block_timestamp_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![], vec![Type::I64]);
    Function::new(store, &signature, move |_args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_CONTEXT_READ)?;

        // Get block timestamp
        let timestamp = env.block_timestamp();

        Ok(vec![Value::I64(timestamp as i64)])
    })
}

/// Get contract's own address
fn get_contract_address_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![], vec![Type::I32]);
    Function::new(store, &signature, move |_args| {
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_CONTEXT_READ)?;

        // Get contract address
        let address = env.contract_address().to_string();

        // Write to memory and return pointer
        let ptr = env.write_to_memory(address.as_bytes())?;

        Ok(vec![Value::I32(ptr as i32)])
    })
}

/// Memory allocation function for WASM
fn alloc_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![Type::I32], vec![Type::I32]);
    Function::new(store, &signature, move |args| {
        // Charge gas proportional to allocation size
        let size = args[0].unwrap_i32() as u32;
        env.gas_meter()
            .use_gas(crate::wasm::GAS_COST_BASE + (size as u64) / 100)?;

        // Allocate memory
        let ptr = env.alloc(size)?;

        Ok(vec![Value::I32(ptr as i32)])
    })
}

/// Memory deallocation function for WASM
fn dealloc_fn(store: &Store, env: Arc<WasmEnv>) -> Function {
    let signature = FunctionType::new(vec![Type::I32, Type::I32], vec![]);
    Function::new(store, &signature, move |args| {
        env.gas_meter().use_gas(crate::wasm::GAS_COST_BASE)?;

        // Extract pointer and size
        let ptr = args[0].unwrap_i32() as u32;
        let size = args[1].unwrap_i32() as u32;

        // Deallocate memory
        env.dealloc(ptr, size)?;

        Ok(vec![])
    })
}

impl WasmStorage for dyn Storage {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, WasmError> {
        self.get(key)
            .map_err(|e| WasmError::StorageError(e.to_string()))
    }

    fn set(&self, key: &[u8], value: &[u8]) -> Result<(), WasmError> {
        self.set(key, value)
            .map_err(|e| WasmError::StorageError(e.to_string()))
    }

    fn delete(&self, key: &[u8]) -> Result<(), WasmError> {
        self.delete(key)
            .map_err(|e| WasmError::StorageError(e.to_string()))
    }

    fn has(&self, key: &[u8]) -> Result<bool, WasmError> {
        self.has(key)
            .map_err(|e| WasmError::StorageError(e.to_string()))
    }
}
