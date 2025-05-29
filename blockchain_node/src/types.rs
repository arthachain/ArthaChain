//! Common blockchain types

use crate::utils::crypto::Hash as CryptoHash;
use anyhow;
use blake3;
use hex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash as StdHash, Hasher};

/// Block header
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Block version
    pub version: u32,
    /// Block shard ID
    pub shard_id: u32,
    /// Block height
    pub height: u64,
    /// Previous block hash
    pub prev_hash: CryptoHash,
    /// Block timestamp
    pub timestamp: u64,
    /// Merkle root of transactions
    pub merkle_root: CryptoHash,
    /// Block state root
    pub state_root: CryptoHash,
    /// Block receipt root
    pub receipt_root: CryptoHash,
    /// Block proposer
    pub proposer: Address,
    /// Block signature
    pub signature: Vec<u8>,
    /// Block gas limit
    pub gas_limit: u64,
    /// Block gas used
    pub gas_used: u64,
    /// Block extra data
    pub extra_data: Vec<u8>,
}

/// Block metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BlockMetadata {
    /// Block size in bytes
    pub size: u64,
    /// Gas used
    pub gas_used: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Validator signatures
    pub signatures: HashMap<String, Vec<u8>>,
}

/// Transaction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transaction {
    /// Transaction sender
    pub from: Address,
    /// Transaction recipient
    pub to: Address,
    /// Transaction value
    pub value: u64,
    /// Gas price
    pub gas_price: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Transaction nonce
    pub nonce: u64,
    /// Transaction data
    pub data: Vec<u8>,
    /// Transaction signature
    pub signature: Vec<u8>,
    /// Transaction hash
    pub hash: CryptoHash,
}

/// Hash type for blockchain data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct Hash(pub Vec<u8>);

impl Hash {
    /// Create a new hash from bytes
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Check if the hash is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the underlying bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }

    pub fn from_hex(hex: &str) -> std::result::Result<Self, anyhow::Error> {
        let bytes = hex::decode(hex).map_err(|e| anyhow::anyhow!("Invalid hex string: {}", e))?;
        Ok(Self(bytes))
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Default for Hash {
    fn default() -> Self {
        Self(vec![0; 32])
    }
}

impl From<Vec<u8>> for Hash {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Address type (20 bytes)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address(pub [u8; 20]);

impl Address {
    pub fn new(bytes: [u8; 20]) -> Self {
        Self(bytes)
    }

    pub fn default() -> Self {
        Self([0u8; 20])
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Create an address from a byte slice
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        if bytes.len() != 20 {
            return Err(anyhow::anyhow!(
                "Invalid address length. Expected 20 bytes, got {}",
                bytes.len()
            ));
        }
        let mut addr = [0u8; 20];
        addr.copy_from_slice(bytes);
        Ok(Self(addr))
    }

    /// Create an address from a hex string
    pub fn from_string(s: &str) -> Result<Self, &'static str> {
        let s = s.trim_start_matches("0x");
        let bytes = hex::decode(s).map_err(|_| "Invalid hex string")?;
        if bytes.len() != 20 {
            return Err("Invalid address length");
        }
        let mut addr = [0u8; 20];
        addr.copy_from_slice(&bytes);
        Ok(Self(addr))
    }

    /// Get address as hex string
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }

    pub fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Default for Address {
    fn default() -> Self {
        Self::default()
    }
}

impl AsRef<[u8]> for Address {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Transaction {
    pub fn new(
        from: Address,
        to: Address,
        value: u64,
        data: Vec<u8>,
        nonce: u64,
        gas_price: u64,
        gas_limit: u64,
    ) -> Self {
        let mut tx = Self {
            hash: CryptoHash::default(),
            from,
            to,
            value,
            data,
            nonce,
            gas_price,
            gas_limit,
            signature: Vec::new(),
        };
        tx.hash = tx.calculate_hash();
        tx
    }

    pub fn calculate_hash(&self) -> CryptoHash {
        let data = self.serialize_for_hash();
        let hash = blake3::hash(&data);
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(hash.as_bytes());
        CryptoHash::new(bytes)
    }

    pub fn serialize_for_hash(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.from.0);
        data.extend_from_slice(&self.to.0);
        data.extend_from_slice(&self.value.to_le_bytes());
        data.extend_from_slice(&self.nonce.to_le_bytes());
        data.extend_from_slice(&self.gas_price.to_le_bytes());
        data.extend_from_slice(&self.gas_limit.to_le_bytes());
        data.extend_from_slice(&self.data);
        data
    }
}

impl BlockHeader {
    /// Create a new block header with basic parameters
    pub fn new(
        version: u32,
        shard_id: u32,
        height: u64,
        prev_hash: CryptoHash,
        timestamp: u64,
        merkle_root: CryptoHash,
        state_root: CryptoHash,
    ) -> Self {
        Self {
            version,
            shard_id,
            height,
            prev_hash,
            timestamp,
            merkle_root,
            state_root,
            receipt_root: CryptoHash::default(),
            proposer: Address::default(),
            signature: Vec::new(),
            gas_limit: 0,
            gas_used: 0,
            extra_data: Vec::new(),
        }
    }

    /// Create a complete block header with all parameters
    pub fn with_all_fields(
        version: u32,
        shard_id: u32,
        height: u64,
        prev_hash: CryptoHash,
        timestamp: u64,
        merkle_root: CryptoHash,
        state_root: CryptoHash,
        receipt_root: CryptoHash,
        proposer: Address,
        signature: Vec<u8>,
        gas_limit: u64,
        gas_used: u64,
        extra_data: Vec<u8>,
    ) -> Self {
        Self {
            version,
            shard_id,
            height,
            prev_hash,
            timestamp,
            merkle_root,
            state_root,
            receipt_root,
            proposer,
            signature,
            gas_limit,
            gas_used,
            extra_data,
        }
    }

    pub fn calculate_hash(&self) -> CryptoHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.shard_id.to_le_bytes());
        hasher.update(&self.height.to_le_bytes());
        hasher.update(self.prev_hash.as_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(self.merkle_root.as_bytes());
        hasher.update(self.state_root.as_bytes());
        hasher.update(self.receipt_root.as_bytes());
        hasher.update(self.proposer.as_bytes());
        hasher.update(&self.signature);
        hasher.update(&self.gas_limit.to_le_bytes());
        hasher.update(&self.gas_used.to_le_bytes());
        hasher.update(&self.extra_data);
        let hash = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(hash.as_bytes());
        CryptoHash::new(bytes)
    }
}

/// Block hash type
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BlockHash(pub CryptoHash);

impl BlockHash {
    pub fn new(hash: CryptoHash) -> Self {
        Self(hash)
    }
}

impl fmt::Display for BlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// State root hash type
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct StateRoot(pub CryptoHash);

impl StateRoot {
    pub fn new(hash: CryptoHash) -> Self {
        Self(hash)
    }
}

impl fmt::Display for StateRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Remove PartialEq and Eq from ValidatorMetrics since it contains f64 fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorMetrics {
    pub total_blocks_proposed: u64,
    pub total_blocks_validated: u64,
    pub total_transactions_processed: u64,
    pub avg_response_time: f64,
    pub uptime: f64,
    pub last_seen: u64,
    pub reputation_score: f64,
}

impl Default for ValidatorMetrics {
    fn default() -> Self {
        Self {
            total_blocks_proposed: 0,
            total_blocks_validated: 0,
            total_transactions_processed: 0,
            avg_response_time: 0.0,
            uptime: 100.0,
            last_seen: 0,
            reputation_score: 0.0,
        }
    }
}

impl PartialEq for ValidatorMetrics {
    fn eq(&self, other: &Self) -> bool {
        (self.total_blocks_proposed == other.total_blocks_proposed) &&
        (self.total_blocks_validated == other.total_blocks_validated) &&
        (self.total_transactions_processed == other.total_transactions_processed) &&
        (self.last_seen == other.last_seen) &&
        // Compare f64 values with some tolerance
        (self.avg_response_time - other.avg_response_time).abs() < f64::EPSILON &&
        (self.uptime - other.uptime).abs() < f64::EPSILON &&
        (self.reputation_score - other.reputation_score).abs() < f64::EPSILON
    }
}

impl From<Transaction> for crate::ledger::transaction::Transaction {
    fn from(tx: Transaction) -> Self {
        use crate::ledger::transaction::TransactionType;
        Self::new(
            TransactionType::Transfer, // Default to Transfer since types::Transaction doesn't have tx_type
            hex::encode(tx.from.0),    // Convert Address to hex string
            hex::encode(tx.to.0),      // Convert Address to hex string
            tx.value,
            tx.nonce,
            tx.gas_price,
            tx.gas_limit,
            tx.data,
        )
    }
}

/// Account identifier type
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AccountId(String);

impl AccountId {
    /// Create a new account ID
    pub fn new(id: String) -> Self {
        Self(id)
    }

    /// Get the string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for AccountId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for AccountId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl fmt::Display for AccountId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Block height type
pub type BlockHeight = u64;

/// Transaction hash type
#[derive(Debug, Clone, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct TransactionHash {
    /// Hash bytes
    hash: Vec<u8>,
    /// String representation (cached)
    string_repr: String,
}

impl TransactionHash {
    /// Create a new transaction hash
    pub fn new(hash: Vec<u8>) -> Self {
        let string_repr = hex::encode(&hash);
        Self { hash, string_repr }
    }

    /// Get the hash bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.hash
    }
}

impl From<Vec<u8>> for TransactionHash {
    fn from(hash: Vec<u8>) -> Self {
        Self::new(hash)
    }
}

impl fmt::Display for TransactionHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.string_repr)
    }
}

impl PartialEq for TransactionHash {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl StdHash for TransactionHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl AsRef<[u8]> for TransactionHash {
    fn as_ref(&self) -> &[u8] {
        &self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_id() {
        let account1 = AccountId::from("account1");
        let account2 = AccountId::from("account1".to_string());
        let account3 = AccountId::from("account2");

        assert_eq!(account1, account2);
        assert_ne!(account1, account3);
        assert_eq!(account1.to_string(), "account1");
    }

    #[test]
    fn test_transaction_hash() {
        let hash1 = TransactionHash::from(vec![1, 2, 3, 4]);
        let hash2 = TransactionHash::from(vec![1, 2, 3, 4]);
        let hash3 = TransactionHash::from(vec![5, 6, 7, 8]);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.to_string(), "01020304");
    }
}
