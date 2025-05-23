// Ledger modules will be implemented here
pub mod block;
pub mod state;
pub mod transaction;

// Create an alias for State as BlockchainState to maintain compatibility
pub use state::State as BlockchainState;

// Export BlockExt trait for use by consensus modules
pub use block::BlockExt;

use crate::config::Config;
use crate::ledger::state::State;
use crate::storage::Storage;
use std::sync::Arc;
use thiserror::Error;

/// Transaction processing error
#[derive(Debug, thiserror::Error, Clone)]
pub enum TransactionError {
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Signing failed")]
    SigningFailed,
    #[error("Invalid nonce")]
    InvalidNonce,
    #[error("Duplicate nonce")]
    DuplicateNonce,
    #[error("Insufficient funds")]
    InsufficientFunds,
    #[error("Gas price too low")]
    GasPriceTooLow,
    #[error("Gas limit too low")]
    GasLimitTooLow,
    #[error("Invalid recipient")]
    InvalidRecipient,
    #[error("Contract execution failed: {0}")]
    ContractExecutionFailed(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Invalid public key")]
    InvalidPublicKey,
    #[error("Invalid sender")]
    InvalidSender,
    #[error("Invalid gas price")]
    InvalidGasPrice,
    #[error("Invalid gas limit")]
    InvalidGasLimit,
    #[error("Empty contract code")]
    EmptyContractCode,
    #[error("Invalid amount")]
    InvalidAmount,
    #[error("Stake too small")]
    StakeTooSmall,
    #[error("Transaction expired")]
    Expired,
    #[error("Empty batch")]
    EmptyBatch,
    #[error("From anyhow error: {0}")]
    FromAnyhow(String),
}

/// Errors that can occur during consensus operations
#[derive(Error, Debug, Clone)]
pub enum ConsensusError {
    #[error("Invalid validator signature")]
    InvalidSignature,

    #[error("Insufficient signatures")]
    InsufficientSignatures,

    #[error("Invalid block hash")]
    InvalidBlockHash,

    #[error("Block already finalized")]
    AlreadyFinalized,

    #[error("Invalid consensus state transition")]
    InvalidStateTransition,

    #[error("Missing validators")]
    MissingValidators,

    #[error("Block finality error")]
    FinalityError,

    #[error("Signature combination failed")]
    SignatureCombinationFailed,

    #[error("Consensus operation failed: {0}")]
    OperationFailed(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("{0}")]
    Other(String),

    #[error("Anyhow error: {0}")]
    Anyhow(String),
}

/// Errors that can occur during block validation
#[derive(Error, Debug, Clone)]
pub enum BlockValidationError {
    #[error("Invalid previous hash")]
    InvalidPreviousHash,

    #[error("Invalid timestamp")]
    InvalidTimestamp,

    #[error("Invalid merkle root")]
    InvalidMerkleRoot,

    #[error("Invalid transactions: {0}")]
    InvalidTransactions(TransactionError),

    #[error("Invalid cross-shard reference")]
    InvalidCrossShardRef,

    #[error("Other error: {0}")]
    Other(String),
}

/// Ledger provides a high-level interface to blockchain state and operations
pub struct Ledger {
    /// Current blockchain state
    state: State,
    /// Configuration
    config: Arc<Config>,
    /// Storage backend
    storage: Arc<dyn Storage>,
}

impl Ledger {
    /// Create a new ledger instance
    pub fn new(config: Arc<Config>, storage: Arc<dyn Storage>) -> Self {
        let state = State::new(&*config).expect("Failed to create blockchain state");
        Self {
            state,
            config,
            storage,
        }
    }

    /// Get the current blockchain state
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Get the storage instance
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Get the configuration
    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }
}

// Implement conversion from anyhow::Error to TransactionError
impl From<anyhow::Error> for TransactionError {
    fn from(err: anyhow::Error) -> Self {
        TransactionError::FromAnyhow(err.to_string())
    }
}
