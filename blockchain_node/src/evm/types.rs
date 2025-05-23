use ethereum_types::{H160, H256, U256};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ethereum-compatible address (20 bytes)
pub type EvmAddress = H160;

/// Configuration for the EVM runtime
#[derive(Clone, Debug)]
pub struct EvmConfig {
    /// Chain ID for EVM transactions
    pub chain_id: u64,
    /// Default gas price (in wei)
    pub default_gas_price: u64,
    /// Default gas limit for transactions
    pub default_gas_limit: u64,
    /// Mapping of precompiled contracts
    pub precompiles: HashMap<EvmAddress, PrecompileFunction>,
}

/// Type for precompiled contract functions
pub type PrecompileFunction = fn(&[u8], u64) -> Result<(Vec<u8>, u64), EvmError>;

/// EVM error types
#[derive(thiserror::Error, Debug)]
pub enum EvmError {
    #[error("Out of gas")]
    OutOfGas,
    #[error("Invalid opcode: {0}")]
    InvalidOpcode(u8),
    #[error("Stack underflow")]
    StackUnderflow,
    #[error("Stack overflow")]
    StackOverflow,
    #[error("Invalid jump destination")]
    InvalidJumpDestination,
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),
    #[error("Execution reverted: {0}")]
    Reverted(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Transaction for the EVM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmTransaction {
    /// Sender address
    pub from: EvmAddress,
    /// Recipient address (None for contract creation)
    pub to: Option<EvmAddress>,
    /// Transaction value in wei
    pub value: U256,
    /// Transaction data (bytecode for contract creation or calldata for contract calls)
    pub data: Vec<u8>,
    /// Gas price in wei
    pub gas_price: U256,
    /// Gas limit for the transaction
    pub gas_limit: U256,
    /// Nonce for the transaction
    pub nonce: U256,
    /// Chain ID
    pub chain_id: Option<u64>,
    /// Transaction signature components (v, r, s)
    pub signature: Option<(u8, H256, H256)>,
}

/// Structure to hold a log entry from EVM execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmLog {
    /// Contract address that generated the log
    pub address: EvmAddress,
    /// Indexed topics (up to 4)
    pub topics: Vec<H256>,
    /// Log data
    pub data: Vec<u8>,
}

/// Result of executing an EVM transaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmExecutionResult {
    /// Success or failure
    pub success: bool,
    /// Gas used during execution
    pub gas_used: u64,
    /// Return data
    pub return_data: Vec<u8>,
    /// Contract address (if created)
    pub contract_address: Option<EvmAddress>,
    /// Logs generated during execution
    pub logs: Vec<EvmLog>,
    /// Error message (if any)
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmAccount {
    pub nonce: u64,
    pub balance: U256,
    pub code: Vec<u8>,
    pub storage: HashMap<H256, H256>,
}

impl EvmConfig {
    /// Create a new EVM configuration with default settings
    pub fn new(chain_id: u64) -> Self {
        Self {
            chain_id,
            default_gas_price: crate::evm::DEFAULT_GAS_PRICE,
            default_gas_limit: crate::evm::DEFAULT_GAS_LIMIT,
            precompiles: HashMap::new(),
        }
    }

    /// Add a precompiled contract
    pub fn add_precompile(&mut self, address: EvmAddress, function: PrecompileFunction) {
        self.precompiles.insert(address, function);
    }
}
