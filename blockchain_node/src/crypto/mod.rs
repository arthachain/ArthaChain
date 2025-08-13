use serde::{Deserialize, Serialize};
use std::fmt;

// Core crypto error type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptoError {
    InvalidKey(String),
    InvalidSignature(String),
    EncryptionError(String),
    DecryptionError(String),
    HashError(String),
    ZkpError(String),
    Other(String),
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::InvalidKey(msg) => write!(f, "Invalid key: {}", msg),
            CryptoError::InvalidSignature(msg) => write!(f, "Invalid signature: {}", msg),
            CryptoError::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
            CryptoError::DecryptionError(msg) => write!(f, "Decryption error: {}", msg),
            CryptoError::HashError(msg) => write!(f, "Hash error: {}", msg),
            CryptoError::ZkpError(msg) => write!(f, "ZKP error: {}", msg),
            CryptoError::Other(msg) => write!(f, "Crypto error: {}", msg),
        }
    }
}

impl std::error::Error for CryptoError {}

pub mod hash;
pub mod keys;
pub mod signature;
pub mod zkp;

// Re-export commonly used types and functions
// Note: arkworks_zkp module temporarily disabled for build stability
pub use hash::*;
pub use keys::*;
pub use signature::*;
pub use zkp::*;
