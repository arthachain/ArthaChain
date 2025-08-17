//! WASM Host Function Implementations
//!
//! This module contains the host functions that are exposed to WASM contracts.
//! These functions allow contracts to interact with the blockchain state, storage,
//! and environment.

use crate::wasm::runtime::WasmEnv;
use crate::wasm::types::WasmError;
use std::sync::{Arc, Mutex};

// Storage functions

/// Read a value from storage
///
/// Arguments:
/// * `key_ptr`: Pointer to the key in WASM memory
/// * `key_len`: Length of the key
///
/// Returns:
/// * If found: (value_ptr << 32) | value_len
/// * If not found: 0
pub fn storage_read(env: &mut WasmEnv, key_ptr: u32, key_len: u32) -> Result<u64, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume_storage_read(key_len as u64)?;

    // Read the key from WASM memory
    let key = env.read_memory(key_ptr, key_len)?;

    // Read from storage
    match env.storage.get(&env.contract_address, &key) {
        Some(value) => {
            // Write value to memory
            let value_ptr = env.write_to_memory(&value)?;

            // Return pointer to value (upper 32 bits) and length (lower 32 bits)
            Ok(((value_ptr as u64) << 32) | (value.len() as u64))
        }
        None => Ok(0), // Not found
    }
}

/// Write a value to storage
///
/// Arguments:
/// * `key_ptr`: Pointer to the key in WASM memory
/// * `key_len`: Length of the key
/// * `value_ptr`: Pointer to the value in WASM memory
/// * `value_len`: Length of the value
///
/// Returns:
/// * 1 if key already existed, 0 if key is new
pub fn storage_write(
    env: &mut WasmEnv,
    key_ptr: u32,
    key_len: u32,
    value_ptr: u32,
    value_len: u32,
) -> Result<u32, WasmError> {
    // Charge gas for the operation
    env.gas_meter
        .consume_storage_write(key_len as u64, value_len as u64)?;

    // Read the key and value from WASM memory
    let key = env.read_memory(key_ptr, key_len)?;
    let value = env.read_memory(value_ptr, value_len)?;

    // Check if key already exists
    let exists = env.storage.has(&env.contract_address, &key);

    // Write to storage
    env.storage.set(&env.contract_address, &key, &value);

    // Return 1 if key existed, 0 if new
    Ok(if exists { 1 } else { 0 })
}

/// Delete a value from storage
///
/// Arguments:
/// * `key_ptr`: Pointer to the key in WASM memory
/// * `key_len`: Length of the key
///
/// Returns:
/// * 1 if key was deleted, 0 if key didn't exist
pub fn storage_delete(env: &mut WasmEnv, key_ptr: u32, key_len: u32) -> Result<u32, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume_storage_delete(key_len as u64)?;

    // Read the key from WASM memory
    let key = env.read_memory(key_ptr, key_len)?;

    // Check if key exists
    let exists = env.storage.has(&env.contract_address, &key);

    // Delete if exists
    if exists {
        env.storage.delete(&env.contract_address, &key);
    }

    // Return 1 if deleted, 0 if not found
    Ok(if exists { 1 } else { 0 })
}

// Context functions

/// Get the caller address
///
/// Arguments:
/// * `result_ptr`: Pointer to write the result in WASM memory
///
/// Returns:
/// * String length written
pub fn get_caller(env: &mut WasmEnv, result_ptr: u32) -> Result<u32, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume(10)?;

    // Get caller address as string
    let caller = env.caller_str.as_ref();

    // Write to memory
    env.write_memory(result_ptr, caller)?;

    // Return length of string written
    Ok(caller.len() as u32)
}

/// Get the current block number (height)
///
/// Returns:
/// * Current block height
pub fn get_block_number(env: &mut WasmEnv) -> Result<u64, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume(5)?;

    // Return block height
    Ok(env.context.block_height)
}

/// Get the current block timestamp
///
/// Returns:
/// * Current block timestamp
pub fn get_block_timestamp(env: &mut WasmEnv) -> Result<u64, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume(5)?;

    // Return block timestamp
    Ok(env.context.block_timestamp)
}

/// Get the contract address
///
/// Arguments:
/// * `result_ptr`: Pointer to write the result in WASM memory
///
/// Returns:
/// * String length written
pub fn get_contract_address(env: &mut WasmEnv, result_ptr: u32) -> Result<u32, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume(10)?;

    // Get contract address as string
    let address = env.contract_address_str.as_ref();

    // Write to memory
    env.write_memory(result_ptr, address)?;

    // Return length of string written
    Ok(address.len() as u32)
}

// Cryptographic functions

/// Compute a cryptographic hash (Keccak-256)
///
/// Arguments:
/// * `data_ptr`: Pointer to the data to hash in WASM memory
/// * `data_len`: Length of the data
/// * `result_ptr`: Pointer to write the result in WASM memory
///
/// Returns:
/// * 32 (length of hash)
pub fn crypto_hash(
    env: &mut WasmEnv,
    data_ptr: u32,
    data_len: u32,
    result_ptr: u32,
) -> Result<u32, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume(50 + data_len as u64)?;

    // Read data from memory
    let data = env.read_memory(data_ptr, data_len)?;

    // Compute hash
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(&data);
    let result = hasher.finalize();

    // Write result to memory
    env.write_memory(result_ptr, &result)?;

    // Return hash length (32 bytes for Keccak-256)
    Ok(32)
}

/// Verify a cryptographic signature
///
/// Arguments:
/// * `message_ptr`: Pointer to the message in WASM memory
/// * `message_len`: Length of the message
/// * `signature_ptr`: Pointer to the signature in WASM memory
/// * `signature_len`: Length of the signature
/// * `public_key_ptr`: Pointer to the public key in WASM memory
/// * `public_key_len`: Length of the public key
///
/// Returns:
/// * 1 if signature is valid, 0 if invalid
pub fn crypto_verify(
    env: &mut WasmEnv,
    message_ptr: u32,
    message_len: u32,
    signature_ptr: u32,
    signature_len: u32,
    public_key_ptr: u32,
    public_key_len: u32,
) -> Result<u32, WasmError> {
    // Charge gas for the operation
    env.gas_meter.consume(200)?;

    // Read data from memory
    let message = env.read_memory(message_ptr, message_len)?;
    let signature = env.read_memory(signature_ptr, signature_len)?;
    let public_key = env.read_memory(public_key_ptr, public_key_len)?;

    // Verify signature using the blockchain's crypto utilities
    match crate::utils::crypto::dilithium_verify(&public_key, &message, &signature) {
        Ok(true) => Ok(1), // Valid signature
        _ => Ok(0),        // Invalid signature
    }
}

/// Log an event from the contract
///
/// Arguments:
/// * `topic_ptr`: Pointer to the topic in WASM memory
/// * `topic_len`: Length of the topic
/// * `data_ptr`: Pointer to the data in WASM memory
/// * `data_len`: Length of the data
pub fn log_event(
    env: &mut WasmEnv,
    topic_ptr: u32,
    topic_len: u32,
    data_ptr: u32,
    data_len: u32,
) -> Result<(), WasmError> {
    // Charge gas for the operation
    env.gas_meter
        .consume(50 + topic_len as u64 + data_len as u64)?;

    // Read data from memory
    let topic = env.read_memory(topic_ptr, topic_len)?;
    let data = env.read_memory(data_ptr, data_len)?;

    // Convert to strings (or hex if not valid UTF-8)
    let topic_str = String::from_utf8(topic.clone()).unwrap_or_else(|_| hex::encode(&topic));
    let data_str = String::from_utf8(data.clone()).unwrap_or_else(|_| hex::encode(&data));

    // Create log message
    let log_message = format!("Event {} - {}", topic_str, data_str);

    // Add to logs
    env.logs.push(log_message);

    Ok(())
}

/// Register all host functions for a WASM module
/// TODO: Temporarily commented out due to wasmer/wasmtime conflict
pub fn register_host_functions(// instance: &mut wasmer::Instance,
    // env: Arc<Mutex<WasmEnv>>,
) -> Result<(), WasmError> {
    // TODO: Temporarily commented out due to wasmer/wasmtime conflict
    /*
    // Define function signatures

    // Storage functions
    instance.exports.register(
        "storage_read",
        move |env_ref: &mut WasmEnv, key_ptr: u32, key_len: u32| -> u64 {
            storage_read(env_ref, key_ptr, key_len).unwrap_or(0)
        },
    )?;

    instance.exports.register(
        "storage_write",
        move |env_ref: &mut WasmEnv,
              key_ptr: u32,
              key_len: u32,
              value_ptr: u32,
              value_len: u32|
              -> u32 {
            storage_write(env_ref, key_ptr, key_len, value_ptr, value_len).unwrap_or(0)
        },
    )?;

    instance.exports.register(
        "storage_delete",
        move |env_ref: &mut WasmEnv, key_ptr: u32, key_len: u32| -> u32 {
            storage_delete(env_ref, key_ptr, key_len).unwrap_or(0)
        },
    )?;

    // Context functions
    instance.exports.register(
        "get_caller",
        move |env_ref: &mut WasmEnv, result_ptr: u32| -> u32 {
            get_caller(env_ref, result_ptr).unwrap_or(0)
        },
    )?;

    instance
        .exports
        .register("get_block_number", move |env_ref: &mut WasmEnv| -> u64 {
            get_block_number(env_ref).unwrap_or(0)
        })?;

    instance
        .exports
        .register("get_block_timestamp", move |env_ref: &mut WasmEnv| -> u64 {
            get_block_timestamp(env_ref).unwrap_or(0)
        })?;

    instance.exports.register(
        "get_contract_address",
        move |env_ref: &mut WasmEnv, result_ptr: u32| -> u32 {
            get_contract_address(env_ref, result_ptr).unwrap_or(0)
        },
    )?;

    // Crypto functions
    instance.exports.register(
        "crypto_hash",
        move |env_ref: &mut WasmEnv, data_ptr: u32, data_len: u32, result_ptr: u32| -> u32 {
            crypto_hash(env_ref, data_ptr, data_len, result_ptr).unwrap_or(0)
        },
    )?;

    instance.exports.register(
        "crypto_verify",
        move |env_ref: &mut WasmEnv,
              message_ptr: u32,
              message_len: u32,
              signature_ptr: u32,
              signature_len: u32,
              public_key_ptr: u32,
              public_key_len: u32|
              -> u32 {
            crypto_verify(
                env_ref,
                message_ptr,
                message_len,
                signature_ptr,
                signature_len,
                public_key_ptr,
                public_key_len,
            )
            .unwrap_or(0)
        },
    )?;

    // Log functions
    instance.exports.register(
        "log_event",
        move |env_ref: &mut WasmEnv,
              topic_ptr: u32,
              topic_len: u32,
              data_ptr: u32,
              data_len: u32| {
            let _ = log_event(env_ref, topic_ptr, topic_len, data_ptr, data_len);
        },
    )?;
    */

    Ok(())
}
