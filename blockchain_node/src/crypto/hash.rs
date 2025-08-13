use blake3;
use serde::{Deserialize, Serialize};
use std::fmt;

// Re-export Hasher for other modules
pub use blake3::Hasher;

/// Hash type used throughout the blockchain
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hash([u8; 32]);

impl Hash {
    /// Create a new hash from a byte array
    pub fn new(data: [u8; 32]) -> Self {
        Hash(data)
    }

    /// Create a quantum-resistant hash from input data using BLAKE3
    pub fn from_data(data: &[u8]) -> Self {
        let hash_result = blake3::hash(data);
        let mut hash_data = [0u8; 32];
        hash_data.copy_from_slice(hash_result.as_bytes());
        Hash(hash_data)
    }

    /// Get the raw bytes of the hash
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl From<[u8; 32]> for Hash {
    fn from(data: [u8; 32]) -> Self {
        Hash(data)
    }
}

impl From<Hash> for [u8; 32] {
    fn from(hash: Hash) -> Self {
        hash.0
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
