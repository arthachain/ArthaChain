use std::error::Error as StdError;
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BlockchainError {
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    #[error("Mempool is full")]
    MempoolFull,

    #[error("Transaction not found")]
    TransactionNotFound,

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Insufficient funds")]
    InsufficientFunds,

    #[error("Invalid nonce")]
    InvalidNonce,

    #[error("Block not found")]
    BlockNotFound,

    #[error("State error: {0}")]
    StateError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Consensus error: {0}")]
    ConsensusError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<String> for BlockchainError {
    fn from(err: String) -> Self {
        BlockchainError::Other(err)
    }
}

impl From<&str> for BlockchainError {
    fn from(err: &str) -> Self {
        BlockchainError::Other(err.to_string())
    }
}

impl From<anyhow::Error> for BlockchainError {
    fn from(err: anyhow::Error) -> Self {
        BlockchainError::Other(err.to_string())
    }
}

impl From<serde_json::Error> for BlockchainError {
    fn from(err: serde_json::Error) -> Self {
        BlockchainError::SerializationError(err.to_string())
    }
}

impl From<hex::FromHexError> for BlockchainError {
    fn from(err: hex::FromHexError) -> Self {
        BlockchainError::SerializationError(format!("Hex error: {}", err))
    }
}

pub type Result<T> = std::result::Result<T, BlockchainError>;
