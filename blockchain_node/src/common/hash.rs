use anyhow::{anyhow, Result};
use hex;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Hash type representing a 32-byte cryptographic hash
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Hash([u8; 32]);

impl Hash {
    /// Create a new hash from raw bytes
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get raw bytes of the hash
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Create a Hash from a byte slice
    /// Returns error if slice length is not 32 bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(anyhow!(
                "Invalid hash length. Expected 32 bytes, got {}",
                bytes.len()
            ));
        }
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(bytes);
        Ok(Self(hash_bytes))
    }

    /// Create a Hash from a hex string
    pub fn from_hex(hex_str: &str) -> Result<Self> {
        if hex_str.len() != 64 {
            return Err(anyhow!(
                "Invalid hash hex length. Expected 64 chars, got {}",
                hex_str.len()
            ));
        }
        let bytes = hex::decode(hex_str)?;
        Self::from_bytes(&bytes)
    }

    /// Convert to hexadecimal string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// Add conversion from Vec<u8> for compatibility with old code
impl TryFrom<Vec<u8>> for Hash {
    type Error = anyhow::Error;

    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::from_bytes(&bytes)
    }
}

// Add conversion to Vec<u8> for compatibility with old code
impl From<Hash> for Vec<u8> {
    fn from(hash: Hash) -> Self {
        hash.0.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_creation() {
        let bytes = [0u8; 32];
        let hash = Hash::new(bytes);
        assert_eq!(hash.as_bytes(), &bytes);
    }

    #[test]
    fn test_hash_from_bytes() {
        let bytes = [1u8; 32];
        let hash = Hash::from_bytes(&bytes).unwrap();
        assert_eq!(hash.as_bytes(), &bytes);
    }

    #[test]
    fn test_hash_from_hex() {
        let hex_str = "0000000000000000000000000000000000000000000000000000000000000000";
        let hash = Hash::from_hex(hex_str).unwrap();
        assert_eq!(hash.to_hex(), hex_str);
    }

    #[test]
    fn test_hash_display() {
        let bytes = [0u8; 32];
        let hash = Hash::new(bytes);
        assert_eq!(
            format!("{hash}"),
            "0000000000000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_hash_conversion() {
        let bytes = vec![0u8; 32];
        let hash = Hash::try_from(bytes.clone()).unwrap();
        let back_to_vec: Vec<u8> = hash.into();
        assert_eq!(bytes, back_to_vec);
    }
}
