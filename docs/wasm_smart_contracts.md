# WebAssembly Smart Contract Engine

This document describes the WebAssembly (WASM) smart contract execution engine integrated into the blockchain. The WASM VM allows executing smart contracts in a secure, isolated, and deterministic environment with controlled access to blockchain resources.

## Overview

The WASM smart contract engine allows developers to write contracts in any language that compiles to WebAssembly (Rust, AssemblyScript, C/C++, etc.). These contracts are executed in a sandboxed environment with access to specific blockchain functionality through host functions.

Key features of the WASM smart contract engine:

1. **Security**: Strict validation and isolation of contract code
2. **Gas Metering**: Detailed tracking of execution costs for all operations
3. **Storage Access**: Controlled access to blockchain state
4. **Error Handling**: Proper isolation of errors and state rollback on failures
5. **Host Functions**: Blockchain-specific APIs for contracts to interact with the environment

## Architecture

The WASM smart contract system consists of the following components:

- **WasmVm**: Core VM implementation for loading and executing WASM bytecode
- **WasmEnv**: Execution environment providing context and resources
- **Host Functions**: Bridge between the WASM code and blockchain resources
- **Storage Interface**: Contract-specific key-value storage abstraction
- **Gas Metering**: Resource accounting for execution costs
- **Error Handling**: Structured error reporting and recovery mechanisms

## Host Functions

Contracts can access blockchain functionality through the following host functions:

### Storage Functions

- `storage_read(key_ptr: u32, key_len: u32) -> u64`: Read a value from contract storage
- `storage_write(key_ptr: u32, key_len: u32, value_ptr: u32, value_len: u32) -> u32`: Write a value to contract storage
- `storage_delete(key_ptr: u32, key_len: u32) -> u32`: Delete a value from contract storage

### Context Functions

- `get_caller(result_ptr: u32) -> u32`: Get the address of the caller
- `get_block_number() -> u64`: Get the current block height
- `get_block_timestamp() -> u64`: Get the current block timestamp
- `get_contract_address(result_ptr: u32) -> u32`: Get the address of the current contract

### Cryptographic Functions

- `crypto_hash(data_ptr: u32, data_len: u32, result_ptr: u32) -> u32`: Compute a cryptographic hash
- `crypto_verify(message_ptr: u32, message_len: u32, signature_ptr: u32, signature_len: u32, public_key_ptr: u32, public_key_len: u32) -> u32`: Verify a cryptographic signature

### Event Functions

- `log_event(topic_ptr: u32, topic_len: u32, data_ptr: u32, data_len: u32)`: Log an event from the contract

## Gas Metering

Every operation in the WASM execution environment consumes gas based on its computational complexity:

- Base cost for any operation
- Per-byte cost for memory operations
- Per-byte cost for storage operations
- Function call overhead

The gas system ensures that contracts cannot consume excessive resources and provides a standard way to charge for execution.

## Error Handling & State Rollback

Contract execution errors are handled safely, with proper error reporting and state preservation:

1. **Validation Errors**: Detected before execution begins
2. **Execution Errors**: Detected during execution
3. **Out of Gas**: When a contract exceeds its gas limit
4. **Execution Timeout**: When a contract runs too long

When an error occurs during contract execution, all state changes are rolled back, ensuring that failed transactions don't corrupt the blockchain state.

## Security Features

The WASM VM implements several security features:

1. **Module Validation**: WASM bytecode is validated before execution
2. **Import Filtering**: Only allowed host functions can be imported
3. **Memory Limits**: Strict limits on memory allocation
4. **Execution Timeouts**: Contracts cannot run indefinitely
5. **Resource Accounting**: All resources used by a contract are accounted for

## Usage Example

Here's a simple example of a counter contract written in WebAssembly text format (WAT):

```wat
(module
  ;; Import host functions
  (import "env" "storage_read" (func $storage_read (param i32 i32) (result i64)))
  (import "env" "storage_write" (func $storage_write (param i32 i32 i32 i32) (result i32)))

  ;; Memory
  (memory 1)
  (export "memory" (memory 0))

  ;; Constants
  (data (i32.const 0) "counter")  ;; Key for the counter

  ;; Increment the counter
  (func $increment (export "increment") (result i32)
    ;; Get current counter value
    (call $storage_read 
      (i32.const 0)   ;; key pointer ("counter")
      (i32.const 7)   ;; key length
    )
    
    ;; Check if counter exists
    (if (result i32)
      (i64.eqz)       ;; Counter doesn't exist
      (then
        ;; Initialize counter to 1
        (call $storage_write
          (i32.const 0)    ;; key pointer ("counter")
          (i32.const 7)    ;; key length
          (i32.const 100)  ;; value pointer (some arbitrary location)
          (i32.const 4)    ;; value length
        )
        (drop)
        (i32.store (i32.const 100) (i32.const 1))
        (i32.const 1)
      )
      (else
        ;; Extract and increment current value
        (local $value_ptr i32)
        (local $current i32)
        (local.set $value_ptr (i32.wrap_i64 (i64.shr_u (i64.const 32))))
        (local.set $current (i32.load (local.get $value_ptr)))
        (local.set $current (i32.add (local.get $current) (i32.const 1)))
        
        ;; Write back to storage
        (i32.store (i32.const 100) (local.get $current))
        (call $storage_write
          (i32.const 0)    ;; key pointer
          (i32.const 7)    ;; key length
          (i32.const 100)  ;; value pointer
          (i32.const 4)    ;; value length
        )
        (drop)
        (local.get $current)
      )
    )
  )
)
```

## Developing Smart Contracts

Smart contracts can be developed in any language that compiles to WebAssembly. We recommend using Rust with the `wasm-bindgen` and `wasm-pack` tools for a streamlined development experience.

### Contract Lifecycle

1. **Development**: Write contract code in your preferred language
2. **Compilation**: Compile to WebAssembly bytecode
3. **Deployment**: Deploy the contract bytecode to the blockchain
4. **Execution**: Execute contract functions through transactions
5. **Upgrade**: Optionally deploy new versions with improved functionality

### Best Practices

1. **Gas Optimization**: Minimize resource usage to reduce transaction costs
2. **Error Handling**: Properly handle error conditions in your contract
3. **Storage Patterns**: Use efficient storage patterns to minimize costs
4. **Testing**: Thoroughly test contracts before deployment
5. **Security Auditing**: Have contracts audited for security vulnerabilities
6. **Upgradeability**: Consider including upgrade mechanisms for long-lived contracts 