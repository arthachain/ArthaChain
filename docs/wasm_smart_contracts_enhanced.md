# WebAssembly Smart Contracts: Comprehensive Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Virtual Machine Implementation](#virtual-machine-implementation)
3. [Contract Execution Lifecycle](#contract-execution-lifecycle)
4. [Smart Contract Development](#smart-contract-development)
5. [Host Functions API Reference](#host-functions-api-reference)
6. [Storage System](#storage-system)
7. [Gas Metering](#gas-metering)
8. [Security Features](#security-features)
9. [Formal Verification](#formal-verification)
10. [Contract Standards](#contract-standards)
11. [Upgradeability Patterns](#upgradeability-patterns)
12. [Debugging Tools](#debugging-tools)
13. [AI Integration](#ai-integration)
14. [Performance Benchmarks](#performance-benchmarks)
15. [Best Practices](#best-practices)
16. [Advanced Topics](#advanced-topics)
17. [Troubleshooting](#troubleshooting)
18. [Glossary](#glossary)

## Architecture Overview

The ArthaChain WebAssembly (WASM) smart contract system is a sophisticated multi-layered platform designed to provide a secure, efficient, and language-agnostic environment for executing decentralized applications on the blockchain.

### Core Components

The WASM smart contract system is built from the following key components:

```
blockchain_node/src/wasm/
├── abi.rs              # Contract ABI definitions and encoding/decoding
├── context.rs          # Execution context definitions
├── debug.rs            # Debugging infrastructure
├── engine.rs           # Core VM engine implementation
├── executor.rs         # High-level contract execution logic
├── gas.rs              # Gas metering and accounting
├── host.rs             # Host environment implementation
├── host_functions.rs   # Functions callable from contracts
├── mod.rs              # Module entry point and re-exports
├── rpc.rs              # JSON-RPC interface for contract operations
├── runtime.rs          # Runtime environment implementation
├── standards.rs        # Contract standards and interfaces
├── storage.rs          # Contract storage implementation
├── types.rs            # Common type definitions
├── upgrade.rs          # Contract upgrade mechanisms
├── verification.rs     # Formal verification tools
└── vm.rs               # Low-level VM implementation
```

### System Architecture Diagram

```
┌─────────────────────────────────┐
│          RPC Interface          │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│       Contract Executor         │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│        WASM Runtime             │
├─────────────────────────────────┤
│  ┌─────────────┐ ┌────────────┐ │
│  │ WASM Engine │ │Gas Metering│ │
│  └─────────────┘ └────────────┘ │
├─────────────────────────────────┤
│  ┌─────────────┐ ┌────────────┐ │
│  │Host Functions│ │   Storage  │ │
│  └─────────────┘ └────────────┘ │
└─────────────────────────────────┘
                │
┌───────────────▼─────────────────┐
│         Blockchain State        │
└─────────────────────────────────┘
```

### System Integration

The WASM module integrates with other blockchain components through:

1. **Storage Layer**: Persists contract code and state
2. **Ledger**: Provides access to blockchain state
3. **Network Layer**: Enables contract deployment and interaction
4. **Consensus**: Ensures deterministic execution across all validators
5. **Security Manager**: Enforces security policies and access control

The WASM VM provides a strict sandboxed environment for contract execution, ensuring that contracts cannot access unauthorized resources or interfere with the host system.

## Virtual Machine Implementation

The ArthaChain WASM Virtual Machine is implemented in the `vm.rs` module and provides a secure, deterministic environment for executing WebAssembly bytecode.

### WASM VM Core

The core WASM VM is implemented as a struct with several key components:

```rust
pub struct WasmVm {
    /// VM configuration
    config: WasmVmConfig,
    /// Cached modules
    modules: HashMap<String, wasmer::Module>,
    /// Wasmer store
    store: wasmer::Store,
}
```

### VM Configuration

The VM is configured through the `WasmVmConfig` structure which allows fine-tuning of various parameters:

```rust
pub struct WasmVmConfig {
    /// Maximum memory pages (64KB each)
    pub max_memory_pages: u32,
    /// Maximum gas limit per execution
    pub default_gas_limit: u64,
    /// Timeout in milliseconds
    pub execution_timeout_ms: u64,
    /// Maximum WASM module size in bytes
    pub max_module_size: usize,
    /// Features enabled for WASM execution
    pub features: WasmFeatures,
}
```

Default configuration values:
- **Memory Limit**: 100 pages (6.4MB)
- **Gas Limit**: 10,000,000 units
- **Execution Timeout**: 5000ms (5 seconds)
- **Module Size Limit**: 2MB

### Execution Environment

Contract execution occurs within a specialized environment that provides access to contract state and blockchain context:

```rust
pub struct WasmEnv {
    /// Storage interface
    pub storage: WasmStorage,
    /// Gas meter for tracking execution costs
    pub gas_meter: GasMeter,
    /// Contract address
    pub contract_address: Address,
    /// Caller address
    pub caller: Address,
    /// Call context
    pub context: CallContext,
    /// Caller address as string
    pub caller_str: String,
    /// Contract address as string
    pub contract_address_str: String,
    /// Value transferred in the call
    pub value: u64,
    /// Call data (function selector and arguments)
    pub call_data: Vec<u8>,
    /// Logs generated during execution
    pub logs: Vec<String>,
    /// Allocated memory map (pointer -> size)
    pub memory_allocations: HashMap<u32, u32>,
    /// Next available memory pointer
    pub next_memory_ptr: u32,
}
```

### Module Loading and Validation

Before execution, WASM bytecode undergoes rigorous validation to ensure it meets safety requirements:

```rust
pub fn load_module(
    &mut self,
    contract_address: &str,
    bytecode: &[u8],
) -> Result<(), WasmError> {
    // Check module size
    if bytecode.len() > self.config.max_module_size {
        return Err(WasmError::ValidationError(format!(
            "Module size exceeds maximum allowed: {} > {}",
            bytecode.len(),
            self.config.max_module_size
        )));
    }

    // Parse and validate using wasmparser
    let mut validator = Validator::new_with_features(self.config.features);

    // Iterate through and validate each payload
    for payload in Parser::new(0).parse_all(bytecode) {
        let payload = payload.map_err(|e| WasmError::ValidationError(e.to_string()))?;
        validator
            .payload(&payload)
            .map_err(|e| WasmError::ValidationError(e.to_string()))?;
    }

    // Compile the module
    let module = wasmer::Module::new(&self.store, bytecode)
        .map_err(|e| WasmError::CompilationError(e.to_string()))?;

    // Cache the module
    self.modules.insert(contract_address.to_string(), module);

    Ok(())
}
```

### Execution Process

Contract execution involves these steps:

1. **Module Retrieval**: Fetch the pre-loaded and cached module
2. **Import Object Creation**: Set up the environment with host functions
3. **Memory Allocation**: Allocate memory with strict bounds checking
4. **Instance Creation**: Instantiate the module with imports
5. **Function Execution**: Execute the target function with arguments
6. **Result Handling**: Process return values and handle errors

```rust
pub fn execute(
    &self,
    contract_address: &str,
    env: WasmEnv,
    function: &str,
    args: &[wasmer::Value],
) -> Result<WasmExecutionResult, WasmError> {
    // Get module from cache
    let module = self.modules.get(contract_address).ok_or_else(|| {
        WasmError::ExecutionError(format!("Module not loaded: {}", contract_address))
    })?;

    // Create import object with host functions
    let env = Arc::new(Mutex::new(env));
    let mut import_object = wasmer::ImportObject::new();

    // Add memory
    let memory = wasmer::Memory::new(
        &self.store,
        wasmer::MemoryType::new(1, Some(self.config.max_memory_pages)),
    )
    .map_err(|e| WasmError::InstantiationError(e.to_string()))?;

    // Add memory to imports
    import_object.register("env", wasmer::Exports::new());

    // Create instance
    let instance = wasmer::Instance::new(module, &import_object)
        .map_err(|e| WasmError::InstantiationError(e.to_string()))?;

    // Get and execute the function
    let function = instance
        .exports
        .get_function(function)
        .map_err(|_| WasmError::ExecutionError(format!("Function not found: {}", function)))?;

    // Execute with timeout and gas metering
    // Result processing and error handling
    // ...
}
```

### Engine Implementation

The VM is built on the Wasmer runtime with several key features:

1. **Just-In-Time Compilation**: Uses Wasmer JIT compiler for optimal performance
2. **Memory Safety**: Strict bounds-checking for all memory operations
3. **Deterministic Execution**: Guaranteed identical results across all nodes
4. **Resource Limiting**: Memory and execution time constraints
5. **Gas Metering**: Fine-grained gas tracking for fair resource accounting 

## Contract Execution Lifecycle

The complete lifecycle of a smart contract on ArthaChain consists of several distinct phases, from compilation to execution and state updates.

### Contract Deployment Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│    Compile     │────►│    Validate    │────►│    Deploy      │
│    Contract    │     │    Bytecode    │     │    Contract    │
└────────────────┘     └────────────────┘     └────────────────┘
                                                      │
                                                      ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Initialize   │◄────│    Generate    │◄────│     Store      │
│   Contract     │     │    Address     │     │    Bytecode    │
└────────────────┘     └────────────────┘     └────────────────┘
```

#### 1. Compilation
The contract source code is compiled to WebAssembly bytecode using the appropriate toolchain for the chosen programming language.

#### 2. Validation
The WASM bytecode undergoes rigorous validation to ensure it meets all platform requirements:

```rust
// From blockchain_node/src/wasm/runtime.rs
fn validate_bytecode(&self, bytecode: &[u8]) -> Result<(), WasmError> {
    // Check size limits
    if bytecode.len() > self.config.max_module_size {
        return Err(WasmError::ValidationError(
            format!("Module exceeds maximum size: {} > {}", 
                bytecode.len(), self.config.max_module_size)
        ));
    }

    // Parse and validate WASM module
    let validation_result = wasmparser::validate(bytecode);
    if let Err(e) = validation_result {
        return Err(WasmError::ValidationError(
            format!("Invalid WASM module: {}", e)
        ));
    }

    // Check for prohibited instructions
    self.check_prohibited_instructions(bytecode)?;

    // Perform standards validation if requested
    if self.config.validate_standards {
        self.perform_standards_validation(bytecode)?;
    }

    // Perform formal verification if enabled
    if self.config.formal_verification {
        self.perform_formal_verification(bytecode)?;
    }

    Ok(())
}
```

#### 3. Deployment
The validated bytecode is deployed to the blockchain through a transaction:

```rust
// From blockchain_node/src/wasm/runtime.rs
pub fn deploy_contract(
    &mut self,
    bytecode: &[u8],
    deployer: Address,
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
    wasm_storage.store_bytecode(bytecode)?;

    // Run constructor if provided
    if let Some(args) = constructor_args {
        let context = CallContext {
            contract_address: contract_address.clone(),
            caller: deployer.clone(),
            block_timestamp: self.get_block_timestamp(),
            block_height: self.get_block_height(),
            value: 0,
        };

        let params = CallParams {
            function: "constructor".to_string(),
            arguments: args.to_vec(),
            gas_limit: self.config.deployment_gas_limit,
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
```

#### 4. Initialization
If a constructor is provided, it is executed to initialize the contract's state.

### Contract Call Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│    Prepare     │────►│  Load Contract │────►│  Create Call   │
│    Call        │     │    Context     │     │   Environment  │
└────────────────┘     └────────────────┘     └────────────────┘
                                                      │
                                                      ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Update       │◄────│  Process       │◄────│   Execute      │
│   State        │     │  Results       │     │   Function     │
└────────────────┘     └────────────────┘     └────────────────┘
```

#### 1. Call Preparation
Contract calls are initiated through transactions or from other contracts:

```rust
// From blockchain_node/src/wasm/executor.rs
pub fn execute_contract(
    &self,
    contract_address: &str,
    call_data: &[u8],
    sender: &str,
    value: u64,
    gas_limit: u64,
) -> Result<WasmExecutionResult, WasmError> {
    // Execution logic...
}
```

#### 2. Context Creation
The execution context is established with all necessary information:

```rust
// Create execution environment
let state = Arc::new(State::new(&crate::config::Config::default())?);
let mut env = WasmEnv::new(
    state.clone(),
    gas_limit,
    sender,
    contract_address,
    value,
    call_data.to_vec(),
);
```

#### 3. Function Execution
The specified function is executed within the VM:

```rust
// Execute the function
let result = function.call(&args)?;

// Process the result...
```

#### 4. State Update
After successful execution, the contract's state changes are committed to storage:

```rust
// From blockchain_node/src/wasm/storage.rs
pub fn commit_changes(&mut self) -> Result<(), WasmError> {
    for (key, value) in &self.pending_writes {
        self.db.put(key, value)?;
    }
    
    for key in &self.pending_deletes {
        self.db.delete(key)?;
    }
    
    // Clear pending operations
    self.pending_writes.clear();
    self.pending_deletes.clear();
    
    Ok(())
}
```

### Execution Result

The outcome of contract execution is captured in the `WasmExecutionResult` structure:

```rust
pub struct WasmExecutionResult {
    /// Whether execution succeeded
    pub succeeded: bool,
    /// Return data from execution (if any)
    pub return_data: Option<Vec<u8>>,
    /// Gas used during execution
    pub gas_used: u64,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Logs generated during execution
    pub logs: Vec<WasmLog>,
}
```

## Smart Contract Development

ArthaChain provides a comprehensive development environment for creating, testing, and deploying WebAssembly smart contracts.

### Supported Languages

The platform supports multiple programming languages for contract development:

#### Rust

Rust is the primary recommended language for ArthaChain smart contracts due to its memory safety, performance, and rich type system.

**Setup:**
1. Install Rust and Cargo:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Add WASM target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

3. Install the ArthaChain SDK:
   ```bash
   cargo install artha-sdk-cli
   ```

**Example Token Contract:**

```rust
use artha_sdk::prelude::*;

#[artha_contract]
pub struct TokenContract {
    total_supply: u64,
    balances: Map<Address, u64>,
}

#[artha_methods]
impl TokenContract {
    pub fn new(initial_supply: u64) -> Self {
        let mut contract = Self {
            total_supply: initial_supply,
            balances: Map::new(),
        };
        let deployer = env::caller();
        contract.balances.insert(deployer, initial_supply);
        contract
    }
    
    pub fn transfer(&mut self, to: Address, amount: u64) -> Result<(), String> {
        let from = env::caller();
        let from_balance = self.balances.get(&from).unwrap_or(0);
        
        if from_balance < amount {
            return Err("Insufficient balance".to_string());
        }
        
        self.balances.insert(from, from_balance - amount);
        let to_balance = self.balances.get(&to).unwrap_or(0);
        self.balances.insert(to, to_balance + amount);
        
        Ok(())
    }
    
    pub fn balance_of(&self, account: Address) -> u64 {
        self.balances.get(&account).unwrap_or(0)
    }
}
```

**Compiling to WASM:**
```bash
cargo build --target wasm32-unknown-unknown --release
```

This produces a `.wasm` file in the `target/wasm32-unknown-unknown/release/` directory.

#### AssemblyScript

AssemblyScript provides a TypeScript-like experience targeting WebAssembly.

**Setup:**
1. Install Node.js and npm
2. Initialize project:
   ```bash
   npm init -y
   npm install --save-dev assemblyscript
   npx asinit .
   ```

3. Install ArthaChain AssemblyScript SDK:
   ```bash
   npm install --save-dev @arthachain/as-sdk
   ```

**Example AssemblyScript Contract:**

```typescript
import { Storage, Context, Address, u256 } from "@arthachain/as-sdk";

export class TokenContract {
  private balances: Storage.Map<Address, u64>;
  private totalSupply: u64;

  constructor(initialSupply: u64) {
    this.totalSupply = initialSupply;
    this.balances = new Storage.Map<Address, u64>("balances");
    this.balances.set(Context.caller, initialSupply);
  }

  transfer(to: Address, amount: u64): boolean {
    const from = Context.caller;
    const fromBalance = this.balances.get(from, 0);
    
    if (fromBalance < amount) {
      return false;
    }
    
    this.balances.set(from, from_balance - amount);
    const to_balance = this.balances.get(to, 0);
    this.balances.set(to, to_balance + amount);
    
    return true;
  }

  balanceOf(account: Address): u64 {
    return this.balances.get(account, 0);
  }
}
```

**Compiling to WASM:**
```bash
npm run asbuild
```

#### C/C++

For advanced use cases requiring maximum performance or specific optimizations.

**Setup:**
1. Install Emscripten:
   ```bash
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

2. Install ArthaChain C/C++ SDK:
   ```bash
   git clone https://github.com/arthachain/c-sdk
   cd c-sdk
   make install
   ```

**Example C Contract:**

```c
#include "artha_sdk.h"

// State variables
DEFINE_MAP(address, uint64_t, balances);
uint64_t total_supply = 0;

// Initialize contract
EXPORT void constructor(uint64_t initial_supply) {
    total_supply = initial_supply;
    address caller = get_caller();
    balances_set(caller, initial_supply);
}

// Transfer tokens
EXPORT uint8_t transfer(address to, uint64_t amount) {
    address from = get_caller();
    uint64_t from_balance = balances_get(from);
    
    if (from_balance < amount) {
        return 0; // Failure
    }
    
    balances_set(from, from_balance - amount);
    uint64_t to_balance = balances_get(to);
    balances_set(to, to_balance + amount);
    
    return 1; // Success
}

// Get account balance
EXPORT uint64_t balance_of(address account) {
    return balances_get(account);
}
```

**Compiling to WASM:**
```bash
emcc -O3 -s WASM=1 -s STANDALONE_WASM -s EXPORTED_FUNCTIONS="['_constructor', '_transfer', '_balance_of']" -o contract.wasm contract.c
```

### SDK Features

The ArthaChain SDK provides numerous features to simplify contract development:

#### 1. Storage Abstractions

```rust
// Key-Value storage
pub fn storage_read(key: &[u8]) -> Option<Vec<u8>>;
pub fn storage_write(key: &[u8], value: &[u8]);
pub fn storage_delete(key: &[u8]);

// High-level abstractions
pub struct Map<K, V> { /* ... */ }
pub struct Vec<T> { /* ... */ }
pub struct Set<T> { /* ... */ }
```

#### 2. Context Access

```rust
// Access to blockchain context
pub fn caller() -> Address;
pub fn contract_address() -> Address;
pub fn block_height() -> u64;
pub fn block_timestamp() -> u64;
pub fn value() -> u64;
```

#### 3. Event Emission

```rust
// Log event emission
pub fn emit_event(topics: &[&[u8]], data: &[u8]);

// High-level event helpers
pub fn emit<T: Serialize>(event_name: &str, data: &T);
```

#### 4. Cryptographic Utilities

```rust
// Hash functions
pub fn keccak256(data: &[u8]) -> [u8; 32];
pub fn blake3(data: &[u8]) -> [u8; 32];

// Signature verification
pub fn verify_signature(pubkey: &[u8], message: &[u8], signature: &[u8]) -> bool;
```

### Contract ABI

All contracts expose their interface through a standardized ABI:

```rust
pub struct ContractABI {
    /// Contract name
    pub name: String,
    /// Contract version
    pub version: String,
    /// Contract methods
    pub methods: Vec<ABIMethod>,
    /// Contract events
    pub events: Vec<ABIEvent>,
    /// Contract constructor
    pub constructor: Option<ABIConstructor>,
}

pub struct ABIMethod {
    /// Method name
    pub name: String,
    /// Method inputs
    pub inputs: Vec<ABIParam>,
    /// Method outputs
    pub outputs: Vec<ABIParam>,
    /// Method mutability
    pub mutability: Mutability,
}
```

This ABI format enables seamless interaction with contracts through JSON-RPC or SDK interfaces. 

## Host Functions API Reference

The ArthaChain WASM virtual machine exposes a set of host functions that smart contracts can call to interact with the blockchain environment. These functions are implemented in `host.rs` and `host_functions.rs`.

### Storage Functions

#### storage_read

Reads a value from contract storage.

```rust
fn storage_read(env: &mut WasmEnv, key_ptr: u32, key_len: u32) -> Result<u64, WasmError>
```

**Parameters:**
- `key_ptr`: Pointer to the key in WASM memory
- `key_len`: Length of the key

**Returns:**
- If found: `(value_ptr << 32) | value_len` - upper 32 bits contain pointer to value, lower 32 bits contain length
- If not found: 0

**Gas Cost:** 
- Base cost + 5 per byte read

**Example:**
```rust
// In contract code
let key = b"counter";
let value_ptr_and_len = storage_read(key.as_ptr() as u32, key.len() as u32);
let value_len = value_ptr_and_len & 0xFFFFFFFF;
let value_ptr = value_ptr_and_len >> 32;
let value = read_memory(value_ptr, value_len);
```

#### storage_write

Writes a value to contract storage.

```rust
fn storage_write(
    env: &mut WasmEnv,
    key_ptr: u32,
    key_len: u32,
    value_ptr: u32,
    value_len: u32,
) -> Result<u32, WasmError>
```

**Parameters:**
- `key_ptr`: Pointer to the key in WASM memory
- `key_len`: Length of the key
- `value_ptr`: Pointer to the value in WASM memory
- `value_len`: Length of the value

**Returns:**
- 1 if key already existed, 0 if key is new

**Gas Cost:** 
- Base cost + 10 per byte written

**Example:**
```rust
// In contract code
let key = b"counter";
let value = 42u64.to_le_bytes();
storage_write(
    key.as_ptr() as u32,
    key.len() as u32,
    value.as_ptr() as u32,
    value.len() as u32
);
```

#### storage_delete

Deletes a value from contract storage.

```rust
fn storage_delete(env: &mut WasmEnv, key_ptr: u32, key_len: u32) -> Result<u32, WasmError>
```

**Parameters:**
- `key_ptr`: Pointer to the key in WASM memory
- `key_len`: Length of the key

**Returns:**
- 1 if key existed and was deleted, 0 if key didn't exist

**Gas Cost:** 
- Base cost + 5 per byte of key

**Example:**
```rust
// In contract code
let key = b"counter";
storage_delete(key.as_ptr() as u32, key.len() as u32);
```

### Context Functions

#### get_caller

Returns the address of the caller.

```rust
fn get_caller(env: &mut WasmEnv, result_ptr: u32) -> Result<u32, WasmError>
```

**Parameters:**
- `result_ptr`: Pointer where to write the result in WASM memory

**Returns:**
- Length of the caller address string

**Gas Cost:** Fixed

**Example:**
```rust
// In contract code
let mut buffer = [0u8; 64]; // Address is at most 64 bytes
let len = get_caller(buffer.as_ptr() as u32);
let caller = &buffer[0..len as usize];
```

#### get_block_number

Returns the current block height.

```rust
fn get_block_number(env: &mut WasmEnv) -> Result<u64, WasmError>
```

**Returns:**
- Current block height

**Gas Cost:** Fixed

**Example:**
```rust
// In contract code
let block_height = get_block_number();
```

#### get_block_timestamp

Returns the current block timestamp.

```rust
fn get_block_timestamp(env: &mut WasmEnv) -> Result<u64, WasmError>
```

**Returns:**
- Current block timestamp (seconds since Unix epoch)

**Gas Cost:** Fixed

**Example:**
```rust
// In contract code
let timestamp = get_block_timestamp();
```

#### get_contract_address

Returns the address of the current contract.

```rust
fn get_contract_address(env: &mut WasmEnv, result_ptr: u32) -> Result<u32, WasmError>
```

**Parameters:**
- `result_ptr`: Pointer where to write the result in WASM memory

**Returns:**
- Length of the contract address string

**Gas Cost:** Fixed

**Example:**
```rust
// In contract code
let mut buffer = [0u8; 64]; // Address is at most 64 bytes
let len = get_contract_address(buffer.as_ptr() as u32);
let address = &buffer[0..len as usize];
```

### Cryptographic Functions

#### crypto_keccak256

Computes the Keccak-256 hash of data.

```rust
fn crypto_keccak256(
    env: &mut WasmEnv,
    data_ptr: u32,
    data_len: u32,
    result_ptr: u32,
) -> Result<(), WasmError>
```

**Parameters:**
- `data_ptr`: Pointer to the data in WASM memory
- `data_len`: Length of the data
- `result_ptr`: Pointer where to write the result (32 bytes)

**Gas Cost:** 
- Base cost + 1 per 32 bytes of input

**Example:**
```rust
// In contract code
let data = b"Hello, World!";
let mut hash = [0u8; 32];
crypto_keccak256(data.as_ptr() as u32, data.len() as u32, hash.as_ptr() as u32);
```

#### crypto_verify

Verifies a cryptographic signature.

```rust
fn crypto_verify(
    env: &mut WasmEnv,
    pubkey_ptr: u32,
    pubkey_len: u32,
    message_ptr: u32,
    message_len: u32,
    signature_ptr: u32,
    signature_len: u32,
) -> Result<u32, WasmError>
```

**Parameters:**
- `pubkey_ptr`: Pointer to the public key in WASM memory
- `pubkey_len`: Length of the public key
- `message_ptr`: Pointer to the message in WASM memory
- `message_len`: Length of the message
- `signature_ptr`: Pointer to the signature in WASM memory
- `signature_len`: Length of the signature

**Returns:**
- 1 if signature is valid, 0 otherwise

**Gas Cost:** 
- Fixed high cost due to cryptographic verification

**Example:**
```rust
// In contract code
let pubkey = [...]; // Public key bytes
let message = b"Message to verify";
let signature = [...]; // Signature bytes
let is_valid = crypto_verify(
    pubkey.as_ptr() as u32, pubkey.len() as u32,
    message.as_ptr() as u32, message.len() as u32,
    signature.as_ptr() as u32, signature.len() as u32
);
```

### Memory Management

#### alloc

Allocates memory in the WASM module.

```rust
fn alloc(env: &mut WasmEnv, size: u32) -> Result<u32, WasmError>
```

**Parameters:**
- `size`: Number of bytes to allocate

**Returns:**
- Pointer to the allocated memory

**Gas Cost:** 
- Base cost + size / 100

**Example:**
```rust
// In contract code
let size = 1024; // 1KB
let ptr = alloc(size);
```

#### dealloc

Deallocates previously allocated memory.

```rust
fn dealloc(env: &mut WasmEnv, ptr: u32, size: u32) -> Result<(), WasmError>
```

**Parameters:**
- `ptr`: Pointer to the memory to deallocate
- `size`: Size of the memory block

**Gas Cost:** Fixed small cost

**Example:**
```rust
// In contract code
dealloc(ptr, size);
```

### Events and Logging

#### emit_event

Emits an event log.

```rust
fn emit_event(
    env: &mut WasmEnv,
    topics_ptr: u32,
    topics_len: u32,
    data_ptr: u32,
    data_len: u32,
) -> Result<(), WasmError>
```

**Parameters:**
- `topics_ptr`: Pointer to array of topic pointers
- `topics_len`: Number of topics
- `data_ptr`: Pointer to event data
- `data_len`: Length of event data

**Gas Cost:** 
- Base cost + data size

**Example:**
```rust
// In contract code
let topic = b"Transfer";
let topics_ptr = [topic.as_ptr() as u32].as_ptr() as u32;
let data = b"sender:recipient:amount";
emit_event(topics_ptr, 1, data.as_ptr() as u32, data.len() as u32);
```

#### debug_log

Logs a debug message (only available in non-production environments).

```rust
fn debug_log(env: &mut WasmEnv, message_ptr: u32, message_len: u32) -> Result<(), WasmError>
```

**Parameters:**
- `message_ptr`: Pointer to the message in WASM memory
- `message_len`: Length of the message

**Gas Cost:** Free in debug mode, unavailable in production

**Example:**
```rust
// In contract code
let message = b"Debug message";
debug_log(message.as_ptr() as u32, message.len() as u32);
```

### Contract Interaction

#### call_contract

Calls another contract.

```rust
fn call_contract(
    env: &mut WasmEnv,
    address_ptr: u32,
    address_len: u32,
    function_ptr: u32,
    function_len: u32,
    args_ptr: u32,
    args_len: u32,
    value: u64,
    result_ptr: u32,
) -> Result<u64, WasmError>
```

**Parameters:**
- `address_ptr`: Pointer to the contract address
- `address_len`: Length of the contract address
- `function_ptr`: Pointer to the function name
- `function_len`: Length of the function name
- `args_ptr`: Pointer to the serialized arguments
- `args_len`: Length of the arguments
- `value`: Value to send with the call
- `result_ptr`: Pointer where to write the result

**Returns:**
- `(success << 32) | result_len` - upper 32 bits indicate success (1) or failure (0), lower 32 bits contain length of result

**Gas Cost:** 
- Base high cost + remaining gas passed to callee

**Example:**
```rust
// In contract code
let address = b"0x1234567890abcdef";
let function = b"transfer";
let args = b"recipient:100";
let mut result_buffer = [0u8; 1024];
let result = call_contract(
    address.as_ptr() as u32, address.len() as u32,
    function.as_ptr() as u32, function.len() as u32,
    args.as_ptr() as u32, args.len() as u32,
    0, // No value sent
    result_buffer.as_ptr() as u32
);
let success = (result >> 32) != 0;
let result_len = result & 0xFFFFFFFF;
```

## Storage System

The ArthaChain WASM smart contract system provides a robust, efficient storage system for persisting contract state on the blockchain.

### Architecture

The storage system is implemented in `storage.rs` and consists of several key components:

```
┌─────────────────────────────────┐
│       Contract Interface         │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│        WasmStorage              │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│      Storage Backend            │
└─────────────────────────────────┘
```

### WasmStorage

`WasmStorage` is the core component that provides contract-specific storage management:

```rust
pub struct WasmStorage {
    /// Underlying storage backend
    storage: Arc<dyn Storage>,
    /// Contract address
    contract_address: WasmContractAddress,
    /// Pending writes (key -> value)
    pending_writes: HashMap<Vec<u8>, Vec<u8>>,
    /// Pending deletes
    pending_deletes: HashSet<Vec<u8>>,
    /// Cache for reads
    read_cache: LruCache<Vec<u8>, Vec<u8>>,
    /// Contract namespace
    namespace: Vec<u8>,
}
```

The storage system provides:

1. **Namespaced Storage**: Each contract gets its own isolated storage space
2. **Transactional Operations**: Changes can be committed or rolled back atomically 
3. **Caching Layer**: Frequently accessed data is cached for efficiency
4. **Deterministic Behavior**: Identical operations produce identical results across all nodes

### Key Namespacing

To isolate contract storage and prevent conflicts, each contract's storage is namespaced:

```rust
fn namespace_key(&self, key: &[u8]) -> Vec<u8> {
    let mut namespaced_key = self.namespace.clone();
    namespaced_key.extend_from_slice(key);
    namespaced_key
}
```

This ensures that even if two contracts use the same key, they access different storage locations.

### Storage Operations

#### Reading Data

```rust
pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
    // Check pending changes first
    if self.pending_deletes.contains(key) {
        return None;
    }
    
    if let Some(value) = self.pending_writes.get(key) {
        return Some(value.clone());
    }
    
    // Check cache
    if let Some(value) = self.read_cache.get(key) {
        return Some(value.clone());
    }
    
    // Read from underlying storage
    let namespaced_key = self.namespace_key(key);
    match self.storage.get(&namespaced_key) {
        Ok(Some(value)) => {
            // Update cache
            self.read_cache.put(key.to_vec(), value.clone());
            Some(value)
        }
        _ => None,
    }
}
```

#### Writing Data

```rust
pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), WasmError> {
    // Update pending changes
    self.pending_writes.insert(key.to_vec(), value.to_vec());
    self.pending_deletes.remove(key);
    
    // Update cache
    self.read_cache.put(key.to_vec(), value.to_vec());
    
    Ok(())
}
```

#### Deleting Data

```rust
pub fn delete(&mut self, key: &[u8]) -> Result<(), WasmError> {
    // Update pending changes
    self.pending_writes.remove(key);
    self.pending_deletes.insert(key.to_vec());
    
    // Update cache
    self.read_cache.pop(key);
    
    Ok(())
}
```

### High-Level Storage Abstractions

The SDK provides high-level abstractions for common data structures to simplify contract development:

#### Map

A key-value mapping that automatically handles serialization and storage:

```rust
pub struct Map<K, V> {
    prefix: Vec<u8>,
    _phantom: PhantomData<(K, V)>,
}
```

#### Vector

A dynamic array stored in contract storage:

```rust
pub struct Vec<T> {
    prefix: Vec<u8>,
    length_key: Vec<u8>,
    _phantom: PhantomData<T>,
}
```

### Storage Limits and Costs

The storage system enforces limits to prevent abuse:

- **Key Size Limit**: 1024 bytes
- **Value Size Limit**: 128 KB
- **Total Storage Per Contract**: Configurable, default 1 GB
- **Gas Costs**: Proportional to data size (see Gas Metering section)

## Gas Metering

The ArthaChain WASM smart contract system provides a robust, efficient gas metering system to ensure fair resource accounting across all contracts.

### Architecture

The gas metering system is implemented in `gas.rs` and consists of several key components:

```
┌─────────────────────────────────┐
│       Contract Interface         │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│        GasMeter               │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│      Gas Backend               │
└─────────────────────────────────┘
```

### GasMeter

`GasMeter` is the core component that provides contract-specific gas management:

```rust
pub struct GasMeter {
    /// Underlying gas backend
    gas: Arc<dyn Gas>,
    /// Contract address
    contract_address: WasmContractAddress,
    /// Gas limit
    gas_limit: u64,
    /// Gas used
    gas_used: u64,
}
```

The gas metering system provides:

1. **Gas Allocation**: Allocates gas to contracts based on their execution requirements
2. **Gas Tracking**: Tracks gas usage for each contract
3. **Gas Limiting**: Ensures contracts do not exceed their allocated gas limit
4. **Gas Accounting**: Accurately accounts for gas used across all contracts

### Gas Allocation

The gas allocation system is implemented in `gas.rs` and ensures that contracts receive the appropriate amount of gas for their execution requirements:

```rust
pub fn allocate_gas(
    &self,
    contract_address: &WasmContractAddress,
    gas_limit: u64,
) -> Result<u64, WasmError> {
    // Implementation logic...
}
```

### Gas Tracking

The gas tracking system is implemented in `gas.rs` and ensures that gas usage is accurately tracked for each contract:

```rust
pub fn track_gas(
    &self,
    contract_address: &WasmContractAddress,
    gas_used: u64,
) -> Result<(), WasmError> {
    // Implementation logic...
}
```

### Gas Limiting

The gas limiting system is implemented in `gas.rs` and ensures that contracts do not exceed their allocated gas limit:

```rust
pub fn check_gas_limit(
    &self,
    contract_address: &WasmContractAddress,
    gas_used: u64,
) -> Result<(), WasmError> {
    // Implementation logic...
}
```

### Gas Accounting

The gas accounting system is implemented in `gas.rs` and ensures that gas usage is accurately accounted for across all contracts:

```rust
pub fn account_gas(
    &self,
    contract_address: &WasmContractAddress,
    gas_used: u64,
) -> Result<(), WasmError> {
    // Implementation logic...
}
```

### Gas Limits and Costs

The gas metering system enforces limits to prevent abuse:

- **Gas Limit**: Configurable, default 10,000,000 units
- **Gas Costs**: Proportional to gas used (see Gas Metering section)

## Security Features

ArthaChain's WebAssembly smart contract platform incorporates comprehensive security features designed to protect both the blockchain and its users from potential vulnerabilities.

### Sandboxed Execution

The WASM VM provides a strict sandboxed execution environment:

```rust
// From blockchain_node/src/wasm/vm.rs
pub fn create_sandboxed_environment(
    &self, 
    memory_limit: u32,
    call_depth_limit: u32
) -> Result<WasmSandbox, WasmError> {
    // Create isolated sandbox for execution
    let sandbox = WasmSandbox {
        memory_limit,
        call_depth_limit,
        current_call_depth: 0,
        // Other sandbox parameters...
    };
    
    // Apply security restrictions
    sandbox.apply_restrictions()?;
    
    Ok(sandbox)
}
```

Key security aspects of the sandbox:
- **Memory Isolation**: Contract memory is fully isolated from host memory
- **Resource Limits**: Strict constraints on memory, CPU, and storage usage
- **Call Depth Limiting**: Prevents stack overflow attacks via deep recursion
- **Deterministic Execution**: Ensures identical execution across all nodes

### Bytecode Validation

Before execution, all WASM bytecode undergoes rigorous validation:

```rust
// From blockchain_node/src/wasm/verification.rs
pub fn validate_wasm_bytecode(bytecode: &[u8]) -> Result<(), WasmError> {
    // Size limits
    if bytecode.len() > MAX_MODULE_SIZE {
        return Err(WasmError::ValidationError("Module too large".to_string()));
    }
    
    // Parse and validate with wasmparser
    let validation_result = wasmparser::validate(bytecode);
    if let Err(e) = validation_result {
        return Err(WasmError::ValidationError(format!("Invalid WASM: {}", e)));
    }
    
    // Check for prohibited instructions
    check_prohibited_instructions(bytecode)?;
    
    // Additional security checks
    check_import_section(bytecode)?;
    check_export_section(bytecode)?;
    check_function_section(bytecode)?;
    
    Ok(())
}
```

The validation process includes:
- **Size Checks**: Enforces maximum module size (2MB by default)
- **Format Validation**: Ensures conformance to the WebAssembly specification
- **Prohibited Instructions**: Blocks non-deterministic or unsafe instructions
- **Import Analysis**: Restricts imports to approved host functions only
- **Static Analysis**: Detects potentially malicious code patterns

### Prohibited Instructions

Certain WASM instructions are prohibited for security or determinism reasons:

```rust
// From blockchain_node/src/wasm/verification.rs
fn check_prohibited_instructions(bytecode: &[u8]) -> Result<(), WasmError> {
    // Instructions that are prohibited for security or determinism
    const PROHIBITED: &[&str] = &[
        "f32.nearest", "f64.nearest",  // Non-deterministic floating point
        "memory.grow",                 // Dynamic memory growth (must use host function)
        "memory.size",                 // Direct memory size access
        // Other prohibited instructions...
    ];
    
    // Parse the module and check for prohibited instructions
    // Implementation details...
    
    Ok(())
}
```

### Memory Safety

The VM enforces strict memory safety:

```rust
// From blockchain_node/src/wasm/host.rs
pub fn read_memory_bytes(
    memory: &Memory,
    ptr: WasmPtr<u8>,
    len: u32,
) -> Result<Vec<u8>, WasmError> {
    // Bounds check
    if len > MAX_MEMORY_ACCESS_SIZE as u32 {
        return Err(WasmError::MemoryError(format!(
            "Memory access too large: {} > {}",
            len, MAX_MEMORY_ACCESS_SIZE
        )));
    }
    
    // Memory view for safe access
    let view = memory.view::<u8>();
    
    // Safely read memory
    let mut result = Vec::with_capacity(len as usize);
    for i in 0..len {
        match view.get(ptr.offset() + i) {
            Some(cell) => result.push(cell.get()),
            None => {
                return Err(WasmError::MemoryError(
                    "Memory access out of bounds".to_string()
                ))
            }
        }
    }
    
    Ok(result)
}
```

Key memory safety features:
- **Bounds Checking**: All memory accesses are bounds-checked
- **Memory Limit**: Maximum memory pages strictly enforced (6.4MB default)
- **Controlled Allocation**: Memory allocation through safe host functions
- **Access Control**: Memory regions properly isolated between calls

### Execution Timeouts

To prevent infinite loops and DoS attacks, execution is time-limited:

```rust
// From blockchain_node/src/wasm/runtime.rs
pub fn execute_with_timeout(
    &self,
    instance: &wasmer::Instance,
    function: &str,
    args: &[wasmer::Value],
    timeout_ms: u64
) -> Result<Vec<wasmer::Value>, WasmError> {
    // Record start time
    let start = std::time::Instant::now();
    
    // Spawn execution on a separate thread
    let handle = std::thread::spawn(move || {
        // Execute the function
        instance.exports.get_function(function)?.call(args)
    });
    
    // Wait for completion with timeout
    match handle.join_timeout(std::time::Duration::from_millis(timeout_ms)) {
        Ok(result) => result.map_err(|e| WasmError::ExecutionError(e.to_string())),
        Err(_) => {
            // Timeout occurred
            Err(WasmError::ExecutionTimeout(format!(
                "Execution exceeded timeout of {}ms", timeout_ms
            )))
        }
    }
}
```

### Secure Cryptography

The platform provides secure cryptographic primitives:

```rust
// From blockchain_node/src/wasm/host_functions.rs
pub fn crypto_verify(
    env: &mut WasmEnv,
    pubkey_ptr: u32,
    pubkey_len: u32,
    message_ptr: u32,
    message_len: u32,
    signature_ptr: u32,
    signature_len: u32,
) -> Result<u32, WasmError> {
    // Read inputs from memory
    let pubkey = env.read_memory(pubkey_ptr, pubkey_len)?;
    let message = env.read_memory(message_ptr, message_len)?;
    let signature = env.read_memory(signature_ptr, signature_len)?;
    
    // Use platform's secure cryptographic implementation
    match crate::crypto::verify(&pubkey, &message, &signature) {
        Ok(true) => Ok(1),  // Valid signature
        Ok(false) => Ok(0), // Invalid signature
        Err(e) => Err(WasmError::CryptoError(e.to_string())),
    }
}
```

Key cryptographic security features:
- **Standardized Algorithms**: Industry-standard, audited implementations
- **Post-Quantum Ready**: Infrastructure for post-quantum cryptographic algorithms
- **Secure Random Number Generation**: Hardware-backed when available
- **Constant-Time Operations**: Protection against timing attacks

### Reentrancy Protection

The system provides built-in protection against reentrancy attacks:

```rust
// From blockchain_node/src/wasm/runtime.rs
pub fn call_contract(
    &mut self,
    contract_address: &WasmContractAddress,
    function: &str,
    args: &[u8],
    value: u64,
) -> Result<Vec<u8>, WasmError> {
    // Check for reentrancy
    if self.execution_context.is_contract_active(contract_address) {
        return Err(WasmError::ReentrancyError(
            "Reentrant call detected".to_string()
        ));
    }
    
    // Mark contract as active
    self.execution_context.mark_contract_active(contract_address);
    
    // Execute the call
    let result = self.execute_contract(contract_address, function, args, value);
    
    // Mark contract as inactive
    self.execution_context.mark_contract_inactive(contract_address);
    
    result
}
```

### Security Auditing Tools

The platform includes built-in tools for security auditing:

1. **Static Analyzer**: Detects common security patterns
2. **Gas Profiler**: Identifies inefficient code that may be vulnerable to DoS
3. **Call Graph Analyzer**: Maps contract interactions to identify security issues
4. **Storage Access Pattern Analyzer**: Detects improper storage access patterns

### Security Standards

Smart contracts can implement standardized security interfaces:

```rust
// From blockchain_node/src/wasm/standards.rs
pub enum SecurityStandard {
    /// Access control standard
    AccessControl,
    /// Pausable standard
    Pausable,
    /// Reentrancy guard standard
    ReentrancyGuard,
}
```

These standards provide battle-tested implementations of security best practices that developers can easily incorporate into their contracts.

## Formal Verification