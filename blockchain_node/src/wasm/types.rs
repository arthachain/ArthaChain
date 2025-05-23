//! WASM types and structures
//!
//! Defines common types and structures used in the WASM runtime environment

use crate::crypto::hash::{Hash, Hasher};
use crate::types::Address;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// WASM Contract Address - Identifies a smart contract on the blockchain
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct WasmContractAddress(pub String);

impl WasmContractAddress {
    /// Create a new WASM contract address from a deployer address and nonce
    pub fn new(deployer: &Address, nonce: u64) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(deployer.as_bytes());
        hasher.update(&nonce.to_be_bytes());

        // Take the first 20 bytes of the hash as an address (like Ethereum)
        let hash = hasher.finalize();
        let address_bytes = &hash.as_bytes()[0..20];

        // Prefix with "wasm:" to distinguish from other address types
        let address = format!("wasm:{}", hex::encode(address_bytes));
        WasmContractAddress(address)
    }

    /// Create a WASM contract address from a string
    pub fn from_string(s: &str) -> Self {
        WasmContractAddress(s.to_string())
    }

    /// Get the address as bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    /// Get the address as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WasmContractAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// WASM Contract Metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmContractMetadata {
    /// Contract name
    pub name: String,
    /// Contract version
    pub version: String,
    /// Contract author
    pub author: String,
    /// Contract description
    pub description: Option<String>,
    /// Contract ABI (functions and their signatures)
    pub abi: Vec<WasmContractFunction>,
    /// Compiler version
    pub compiler_version: String,
    /// Optimization level used
    pub optimization_level: Option<u8>,
    /// When the contract was deployed
    pub deployed_at: u64,
    /// Address of the account that deployed the contract
    pub deployer: Address,
}

/// Function definition for a WASM contract
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmContractFunction {
    /// Function name
    pub name: String,
    /// Function parameters
    pub params: Vec<WasmContractParam>,
    /// Function return type
    pub returns: Option<WasmValueType>,
    /// Is this function mutable (can change state)
    pub is_mutable: bool,
    /// Is this function view-only (can read state but not change)
    pub is_view: bool,
    /// Is this function payable (can receive tokens)
    pub is_payable: bool,
}

/// Parameter definition for a WASM contract function
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmContractParam {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: WasmValueType,
}

/// Possible value types for WASM contracts
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum WasmValueType {
    /// Unsigned 8-bit integer
    U8,
    /// Unsigned 16-bit integer
    U16,
    /// Unsigned 32-bit integer
    U32,
    /// Unsigned 64-bit integer
    U64,
    /// Unsigned 128-bit integer
    U128,
    /// Signed 8-bit integer
    I8,
    /// Signed 16-bit integer
    I16,
    /// Signed 32-bit integer
    I32,
    /// Signed 64-bit integer
    I64,
    /// Signed 128-bit integer
    I128,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// Boolean
    Bool,
    /// String (UTF-8)
    String,
    /// Binary data
    Bytes,
    /// Array of a specific type
    Array(Box<WasmValueType>),
    /// Map with string keys and specific type values
    Map(Box<WasmValueType>),
    /// Optional value
    Option(Box<WasmValueType>),
    /// Blockchain Address
    Address,
    /// Contract Address
    ContractAddress,
}

/// Error type for WASM operations
#[derive(Error, Debug, Clone)]
pub enum WasmError {
    /// Validation errors for WASM bytecode
    #[error("Invalid WASM bytecode: {0}")]
    InvalidBytecode(String),

    /// Instantiation errors
    #[error("Failed to instantiate WASM module: {0}")]
    InstantiationError(String),

    /// Execution errors
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Gas limit exceeded
    #[error("Gas limit exceeded")]
    GasLimitExceeded,

    /// Storage errors
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Memory access errors
    #[error("Memory access error")]
    MemoryAccessError,

    /// Function not found
    #[error("Function not found: {0}")]
    FunctionNotFound(String),

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Allocation failed
    #[error("Allocation failed")]
    AllocationFailed,

    /// No context available
    #[error("No context available")]
    NoContextAvailable,

    /// Invalid UTF-8 string
    #[error("Invalid UTF-8 string")]
    InvalidUtf8String,

    /// Contract already exists
    #[error("Contract already exists at this address")]
    ContractAlreadyExists,

    /// Contract not found
    #[error("Contract not found at this address")]
    ContractNotFound,

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Bytecode too large
    #[error("Bytecode too large")]
    BytecodeTooLarge,

    /// Memory error
    #[error("Memory error: {0}")]
    MemoryError(String),
}

impl WasmError {
    /// Get the error code for this error
    pub fn as_code(&self) -> u32 {
        match self {
            WasmError::InvalidBytecode(_) => 1,
            WasmError::InstantiationError(_) => 2,
            WasmError::ExecutionError(_) => 3,
            WasmError::GasLimitExceeded => 4,
            WasmError::StorageError(_) => 5,
            WasmError::MemoryAccessError => 6,
            WasmError::FunctionNotFound(_) => 7,
            WasmError::InvalidArgument(_) => 8,
            WasmError::AllocationFailed => 9,
            WasmError::NoContextAvailable => 10,
            WasmError::InvalidUtf8String => 11,
            WasmError::ContractAlreadyExists => 12,
            WasmError::ContractNotFound => 13,
            WasmError::ValidationError(_) => 14,
            WasmError::Internal(_) => 15,
            WasmError::BytecodeTooLarge => 16,
            WasmError::MemoryError(_) => 17,
        }
    }
}

/// Result type for WASM operations
pub type WasmResult<T> = Result<T, WasmError>;

/// Execution context for a contract call
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallContext {
    /// Address of the contract being called
    pub contract_address: Address,

    /// Address of the caller (account or another contract)
    pub caller: Address,

    /// Value attached to the call (in native tokens)
    pub value: u64,

    /// Current block height
    pub block_height: u64,

    /// Current block timestamp (in seconds since epoch)
    pub block_timestamp: u64,

    /// Gas limit for the execution
    pub gas_limit: u64,
}

/// Contract deployment information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContractDeployment {
    /// Contract address (derived from deployer and nonce)
    pub address: Address,

    /// Contract bytecode (validated WASM module)
    pub code: Vec<u8>,

    /// Initial arguments for contract constructor
    pub init_args: Vec<u8>,

    /// Deployer account address
    pub deployer: Address,

    /// Gas limit for deployment
    pub gas_limit: u64,
}

/// Contract execution request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContractExecution {
    /// Contract address to execute
    pub address: Address,

    /// Function name to call
    pub function: String,

    /// Arguments to pass to the function
    pub args: Vec<u8>,

    /// Call context
    pub context: CallContext,
}

/// Result of a contract execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub return_data: Option<Vec<u8>>,
    pub gas_used: u64,
    pub logs: Vec<ContractLog>,
}

/// Contract emitted log entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContractLog {
    /// Contract address that emitted the log
    pub contract_address: WasmContractAddress,

    /// Topic (indexed field) for the log
    pub topics: Vec<Vec<u8>>,

    /// Data payload
    pub data: Vec<u8>,
}

/// Contract metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContractMetadata {
    /// Contract name
    pub name: String,

    /// Contract version
    pub version: String,

    /// Contract author
    pub author: String,

    /// Contract description
    pub description: String,

    /// Deployment timestamp
    pub deployed_at: u64,

    /// Deployer address
    pub deployer: Address,

    /// ABI definition (JSON encoded interface description)
    pub abi: String,
}

/// WASM contract transaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmTransaction {
    /// Transaction sender address
    pub from: Address,

    /// Contract address for calls (None for deployment)
    pub to: Option<WasmContractAddress>,

    /// Value to send with transaction
    pub value: Option<u64>,

    /// Gas limit
    pub gas_limit: u64,

    /// Gas price
    pub gas_price: u64,

    /// Gas used
    pub gas_used: u64,

    /// Nonce
    pub nonce: u64,

    /// Contract bytecode for deployment
    pub data: Option<Vec<u8>>,

    /// Constructor arguments for deployment
    pub constructor_args: Option<Vec<u8>>,

    /// Function name to call
    pub function: Option<String>,

    /// Function arguments
    pub function_args: Option<Vec<u8>>,

    /// Transaction signature
    pub signature: Option<Vec<u8>>,

    /// Transaction hash
    pub hash: Option<[u8; 32]>,
}

impl WasmTransaction {
    /// Create a new deployment transaction
    pub fn new_deployment(
        from: Address,
        bytecode: Vec<u8>,
        constructor_args: Option<Vec<u8>>,
        gas_limit: u64,
    ) -> Self {
        Self {
            from,
            to: None,
            value: None,
            gas_limit,
            gas_price: 1, // Default gas price
            gas_used: 0,
            nonce: 0, // Will be set later
            data: Some(bytecode),
            constructor_args,
            function: None,
            function_args: None,
            signature: None,
            hash: None,
        }
    }

    /// Create a new contract call transaction
    pub fn new_call(
        from: Address,
        to: WasmContractAddress,
        function: String,
        args: Option<Vec<u8>>,
        value: Option<u64>,
        gas_limit: u64,
    ) -> Self {
        Self {
            from,
            to: Some(to),
            value,
            gas_limit,
            gas_price: 1, // Default gas price
            gas_used: 0,
            nonce: 0, // Will be set later
            data: None,
            constructor_args: None,
            function: Some(function),
            function_args: args,
            signature: None,
            hash: None,
        }
    }

    /// Get the sender address
    pub fn get_sender(&self) -> Address {
        self.from.clone()
    }

    /// Sign the transaction
    pub fn sign(&mut self, private_key: &[u8]) -> Result<(), WasmError> {
        // Create transaction hash
        let hash = self.calculate_hash();
        self.hash = Some(hash);

        // In a real implementation, we would sign the hash with the private key
        // For now, we just create a dummy signature
        self.signature = Some(vec![0; 64]);

        Ok(())
    }

    /// Calculate transaction hash
    fn calculate_hash(&self) -> [u8; 32] {
        use sha3::{Digest, Keccak256};
        let mut hasher = Keccak256::new();

        // Add transaction fields to hash
        hasher.update(self.from.as_bytes());
        if let Some(to) = &self.to {
            hasher.update(to.as_bytes());
        }
        if let Some(value) = self.value {
            hasher.update(&value.to_be_bytes());
        }
        hasher.update(&self.gas_limit.to_be_bytes());
        hasher.update(&self.gas_price.to_be_bytes());
        hasher.update(&self.nonce.to_be_bytes());

        if let Some(data) = &self.data {
            hasher.update(data);
        }

        if let Some(args) = &self.constructor_args {
            hasher.update(args);
        }

        if let Some(function) = &self.function {
            hasher.update(function.as_bytes());
        }

        if let Some(args) = &self.function_args {
            hasher.update(args);
        }

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result[..]);
        hash
    }
}

/// Result of WASM execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmExecutionResult {
    /// Success flag
    pub succeeded: bool,

    /// Error message if failed
    pub error: Option<String>,

    /// Gas used during execution
    pub gas_used: u64,

    /// Return data if any
    pub data: Option<Vec<u8>>,

    /// Logs generated during execution
    pub logs: Vec<WasmLog>,

    /// Contract address (for deployment)
    pub contract_address: Option<WasmContractAddress>,
}

impl WasmExecutionResult {
    /// Create a successful execution result
    pub fn success(data: Option<Vec<u8>>, gas_used: u64, logs: Vec<WasmLog>) -> Self {
        Self {
            succeeded: true,
            error: None,
            gas_used,
            data,
            logs,
            contract_address: None,
        }
    }

    /// Create a failed execution result
    pub fn failure(error: String, gas_used: u64, logs: Vec<WasmLog>) -> Self {
        Self {
            succeeded: false,
            error: Some(error),
            gas_used,
            data: None,
            logs,
            contract_address: None,
        }
    }

    /// Create a successful deployment result
    pub fn deployment_success(
        contract_address: WasmContractAddress,
        gas_used: u64,
        logs: Vec<WasmLog>,
    ) -> Self {
        Self {
            succeeded: true,
            error: None,
            gas_used,
            data: None,
            logs,
            contract_address: Some(contract_address),
        }
    }

    /// Create a successful call result
    pub fn call_success(data: Option<Vec<u8>>, gas_used: u64, logs: Vec<WasmLog>) -> Self {
        Self {
            succeeded: true,
            error: None,
            gas_used,
            data,
            logs,
            contract_address: None,
        }
    }

    /// Create a failed call result
    pub fn call_failure(error: String, gas_used: u64, logs: Vec<WasmLog>) -> Self {
        Self {
            succeeded: false,
            error: Some(error),
            gas_used,
            data: None,
            logs,
            contract_address: None,
        }
    }
}

/// Log entry from a WASM contract
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmLog {
    /// Contract address
    pub address: WasmContractAddress,

    /// Log topics (indexed fields)
    pub topics: Vec<Vec<u8>>,

    /// Log data
    pub data: Vec<u8>,
}

/// Call information for contract execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallInfo {
    /// The function name to call
    pub function_name: String,

    /// The arguments to pass to the function (in serialized form)
    pub arguments: Vec<u8>,

    /// Gas limit for this call
    pub gas_limit: u64,
}

/// Contract execution context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContractContext {
    /// The address of the contract being executed
    pub contract_address: String,

    /// The address of the caller
    pub caller: String,

    /// The current block height
    pub block_height: u64,

    /// The current block timestamp
    pub timestamp: u64,
}

/// Storage interface for WASM contracts
pub trait WasmStorage: Send + Sync {
    /// Read a value from storage
    fn read(&self, key: &[u8]) -> WasmResult<Option<Vec<u8>>>;

    /// Write a value to storage
    fn write(&mut self, key: &[u8], value: &[u8]) -> WasmResult<()>;

    /// Delete a key from storage
    fn delete(&mut self, key: &[u8]) -> WasmResult<()>;
}

/// Result of contract execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// The return value from the contract execution (if any)
    pub result: Option<Vec<u8>>,

    /// Gas used during execution
    pub gas_used: u64,

    /// Logs generated during execution
    pub logs: Vec<String>,
}

/// Contract code and metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contract {
    /// The WASM bytecode
    pub bytecode: Vec<u8>,

    /// The contract creator
    pub creator: String,

    /// When the contract was created (block height)
    pub created_at: u64,

    /// Contract hash (used for identity)
    pub hash: String,
}

/// Gas configuration for WASM execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmGasConfig {
    /// Gas per instruction
    pub gas_per_instruction: u64,

    /// Maximum memory pages allowed (64KB per page)
    pub max_memory_pages: u32,

    /// Maximum execution steps
    pub max_execution_steps: u64,

    /// Gas cost for storage read
    pub storage_read_cost: u64,

    /// Gas cost for storage write
    pub storage_write_cost: u64,

    /// Gas cost for storage delete
    pub storage_delete_cost: u64,

    /// Base gas cost for contract calls
    pub call_base_cost: u64,

    /// Gas cost for contract creation
    pub create_contract_cost: u64,

    /// Gas limit
    pub gas_limit: u64,
}
