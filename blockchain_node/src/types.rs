//! Common blockchain types

use crate::crypto::Signature;
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

/// A block in the blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block header containing metadata
    pub header: BlockHeader,
    /// Block transactions (flattened from body)
    pub transactions: Vec<Transaction>,
    /// Block signature
    pub signature: Option<Signature>,
}

impl Block {
    /// Create new block
    pub fn new(previous_hash: Hash, transactions: Vec<Transaction>, difficulty: u64) -> Self {
        let merkle_root = Self::calculate_merkle_root(&transactions).unwrap_or_default();
        let header = BlockHeader {
            version: 1,
            shard_id: 0,
            height: 0,
            prev_hash: CryptoHash::default(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            merkle_root,
            state_root: CryptoHash::default(),
            receipt_root: CryptoHash::default(),
            proposer: Address::default(),
            signature: Vec::new(),
            extra_data: Vec::new(),
            gas_limit: 21000,
            gas_used: 0,
        };

        Self {
            header,
            transactions,
            signature: None,
        }
    }

    fn calculate_merkle_root(transactions: &[Transaction]) -> Result<CryptoHash, anyhow::Error> {
        if transactions.is_empty() {
            return Ok(CryptoHash::default());
        }

        // Simple hash of all transaction hashes concatenated
        let mut hasher = blake3::Hasher::new();
        for tx in transactions {
            hasher.update(tx.hash.as_bytes());
        }
        let hash = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(hash.as_bytes());
        Ok(CryptoHash::new(bytes))
    }
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
    /// Create a new hash from raw bytes
    pub fn new(data: Vec<u8>) -> Self {
        Self(data)
    }

    /// Create hash from data
    pub fn from_data(data: &[u8]) -> Self {
        let hash_bytes = blake3::hash(data).as_bytes().to_vec();
        Self(hash_bytes)
    }

    /// Create hash from hex string
    pub fn from_hex(hex: &str) -> Result<Self, anyhow::Error> {
        let bytes = hex::decode(hex).map_err(|e| anyhow::anyhow!("Invalid hex string: {}", e))?;
        Ok(Self(bytes))
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Get raw bytes as owned vector
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.clone()
    }

    /// Check if hash is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }

    /// Convert to EVM-compatible hex string with 0x prefix
    pub fn to_evm_hex(&self) -> String {
        format!("0x{}", hex::encode(&self.0))
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_evm_hex())
    }
}

impl Default for Hash {
    fn default() -> Self {
        Self(vec![0u8; 32])
    }
}

/// Address type (20 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    /// Get address as EVM-compatible hex string with 0x prefix
    pub fn to_evm_hex(&self) -> String {
        format!("0x{}", hex::encode(&self.0))
    }

    pub fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_evm_hex())
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
    /// Calculate transaction hash
    pub fn hash(&self) -> CryptoHash {
        let mut data = Vec::new();
        data.extend_from_slice(self.from.as_ref());
        data.extend_from_slice(self.to.as_ref());
        data.extend_from_slice(&self.value.to_be_bytes());
        data.extend_from_slice(&self.nonce.to_be_bytes());
        data.extend_from_slice(&self.gas_limit.to_be_bytes());
        data.extend_from_slice(&self.gas_price.to_be_bytes());
        data.extend_from_slice(&self.data);

        let hash = blake3::hash(&data);
        let hash_array = *hash.as_bytes();
        CryptoHash::new(hash_array)
    }

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
        hasher.update(self.prev_hash.as_ref());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(self.merkle_root.as_ref());
        hasher.update(self.state_root.as_ref());
        hasher.update(self.receipt_root.as_ref());
        hasher.update(self.proposer.as_ref());
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

impl From<crate::ledger::transaction::Transaction> for Transaction {
    fn from(tx: crate::ledger::transaction::Transaction) -> Self {
        let from_bytes = hex::decode(&tx.sender).unwrap_or_default();
        let to_bytes = hex::decode(&tx.recipient).unwrap_or_default();

        // Ensure addresses are exactly 20 bytes
        let mut from = [0u8; 20];
        let mut to = [0u8; 20];
        if from_bytes.len() >= 20 {
            from.copy_from_slice(&from_bytes[..20]);
        }
        if to_bytes.len() >= 20 {
            to.copy_from_slice(&to_bytes[..20]);
        }

        Self {
            from: Address(from),
            to: Address(to),
            value: tx.amount,
            gas_price: tx.gas_price,
            gas_limit: tx.gas_limit,
            nonce: tx.nonce,
            data: tx.data,
            signature: tx.signature,
            hash: CryptoHash::default(), // Will be recalculated
        }
    }
}

// Note: Block conversions are complex due to different field sets between types::Block and ledger::block::Block
// For now, using manual conversion in the API layer when needed

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

/// Shard identifier type
pub type ShardId = u64;

/// Node identifier type (re-export from network types)
pub use crate::network::types::NodeId;

/// Transaction hash type
#[derive(Debug, Clone, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct TransactionHash {
    /// Hash bytes
    hash: CryptoHash,
    /// String representation (cached)
    string_repr: String,
}

impl TransactionHash {
    /// Create a new transaction hash
    pub fn new(hash: CryptoHash) -> Self {
        let string_repr = hex::encode(hash.as_bytes());
        Self { hash, string_repr }
    }

    /// Get the hash bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.hash.as_bytes()
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.hash.as_bytes())
    }
}

impl From<Vec<u8>> for TransactionHash {
    fn from(hash: Vec<u8>) -> Self {
        let mut hash_array = [0u8; 32];
        hash_array[..hash.len().min(32)].copy_from_slice(&hash[..hash.len().min(32)]);
        Self::new(CryptoHash::new(hash_array))
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
        self.hash.as_bytes()
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
        assert_eq!(
            hash1.to_string(),
            "0102030400000000000000000000000000000000000000000000000000000000"
        );
    }
}

// DAO-related types
/// Unique identifier for a DAO proposal
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProposalId(pub String);

impl ProposalId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
}

impl fmt::Display for ProposalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Token amount for DAO operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenAmount(pub u64);

impl TokenAmount {
    pub fn new(amount: u64) -> Self {
        Self(amount)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for TokenAmount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for TokenAmount {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl PartialOrd for TokenAmount {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for TokenAmount {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl std::ops::Add for TokenAmount {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl From<Address> for Hash {
    fn from(address: Address) -> Self {
        Hash::new(address.0.to_vec())
    }
}

impl std::ops::Add<&TokenAmount> for TokenAmount {
    type Output = Self;
    fn add(self, other: &TokenAmount) -> Self {
        Self(self.0 + other.0)
    }
}

impl std::ops::Mul<f64> for TokenAmount {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self((self.0 as f64 * rhs) as u64)
    }
}

impl std::ops::AddAssign for TokenAmount {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::SubAssign for TokenAmount {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_sub(rhs.0);
    }
}

impl std::str::FromStr for TokenAmount {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = s.parse::<u64>()?;
        Ok(TokenAmount(v))
    }
}

impl std::iter::Sum for TokenAmount {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        TokenAmount(iter.map(|x| x.0).sum())
    }
}

impl<'a> std::iter::Sum<&'a TokenAmount> for TokenAmount {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        TokenAmount(iter.map(|x| x.0).sum())
    }
}

impl std::ops::Sub for TokenAmount {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

/// Vote weight in DAO governance
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoteWeight(pub u64);

impl VoteWeight {
    pub fn new(weight: u64) -> Self {
        Self(weight)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

impl From<TokenAmount> for VoteWeight {
    fn from(amount: TokenAmount) -> Self {
        Self(amount.0)
    }
}

impl fmt::Display for VoteWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for VoteWeight {
    fn from(v: u64) -> Self {
        VoteWeight(v)
    }
}

impl std::ops::Add for VoteWeight {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        VoteWeight(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for VoteWeight {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

// Gas-related types
/// Gas limit for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GasLimit(pub u64);

impl GasLimit {
    pub fn new(limit: u64) -> Self {
        Self(limit)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

impl From<u64> for GasLimit {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<GasLimit> for u64 {
    fn from(g: GasLimit) -> u64 {
        g.0
    }
}

impl std::ops::Sub for GasLimit {
    type Output = u64;
    fn sub(self, rhs: Self) -> u64 {
        self.0.saturating_sub(rhs.0)
    }
}

/// Gas price for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GasPrice(pub u64);

impl GasPrice {
    pub fn new(price: u64) -> Self {
        Self(price)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
    pub fn base_fee(&self) -> u64 {
        self.0
    }
    pub fn priority_fee(&self) -> u64 {
        0
    }
}

/// Call data for smart contract calls
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallData(pub Vec<u8>);

impl CallData {
    pub fn new(data: Vec<u8>) -> Self {
        Self(data)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Get function hash from first 4 bytes (like Ethereum function selector)
    pub fn function_hash(&self) -> Hash {
        if self.0.len() >= 4 {
            let mut hash_bytes = [0u8; 32];
            hash_bytes[..4].copy_from_slice(&self.0[..4]);
            Hash::new(hash_bytes.to_vec())
        } else {
            Hash::new(vec![0u8; 32])
        }
    }

    /// Estimate computational complexity based on call data size and patterns
    pub fn estimate_complexity(&self) -> u64 {
        let base_complexity = 1000u64;
        let size_factor = self.0.len() as u64;

        // Additional complexity for nested calls, loops, etc.
        let pattern_complexity = self.0.iter().fold(0u64, |acc, &byte| {
            // Simple heuristic: certain byte patterns suggest higher complexity
            match byte {
                0xff => acc + 10,       // Loops, jumps
                0xf0..=0xf4 => acc + 5, // Contract creation/calls
                _ => acc + 1,
            }
        });

        base_complexity + size_factor + pattern_complexity
    }

    /// Get size of call data
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Get hash of the entire call data
    pub fn hash(&self) -> anyhow::Result<Hash> {
        use blake3;
        let hash_bytes = blake3::hash(&self.0);
        Ok(Hash::new(hash_bytes.as_bytes().to_vec()))
    }

    pub fn append(&mut self, other: &CallData) {
        self.0.extend_from_slice(&other.0);
    }

    pub fn value(&self) -> u64 {
        self.0.len() as u64
    }
}

impl Default for CallData {
    fn default() -> Self {
        Self(Vec::new())
    }
}

/// Contract identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContractId(pub String);

impl ContractId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContractId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Address> for ContractId {
    fn from(address: Address) -> Self {
        ContractId(hex::encode(address.0))
    }
}

impl From<u64> for ProposalId {
    fn from(v: u64) -> Self {
        ProposalId(v.to_string())
    }
}

/// Conversion from ledger::Block to types::Block
impl From<crate::ledger::block::Block> for Block {
    fn from(ledger_block: crate::ledger::block::Block) -> Self {
        use crate::utils::crypto::Hash as CryptoHash;

        // Convert ledger header to types header
        let header = BlockHeader {
            version: 1,  // Default version
            shard_id: 0, // Default shard for now
            height: ledger_block.header.height,
            prev_hash: CryptoHash::from_slice(ledger_block.header.previous_hash.as_ref()),
            timestamp: ledger_block.header.timestamp,
            merkle_root: CryptoHash::from_slice(ledger_block.header.merkle_root.as_ref()),
            state_root: CryptoHash::default(), // Will be calculated elsewhere
            receipt_root: CryptoHash::default(), // Will be calculated elsewhere
            proposer: {
                let producer_bytes = ledger_block.header.producer.as_bytes();
                let mut addr_bytes = [0u8; 20];
                let copy_len = producer_bytes.len().min(20);
                addr_bytes[..copy_len].copy_from_slice(&producer_bytes[..copy_len]);
                Address(addr_bytes)
            }, // Convert BLS key to address
            signature: Vec::new(),             // Will be set from block signature
            gas_limit: 21000000,               // Default gas limit
            gas_used: 0,                       // Will be calculated
            extra_data: Vec::new(),            // Empty for now
        };

        // Convert transactions from ledger::block::Transaction to types::Transaction
        let transactions: Vec<Transaction> = ledger_block
            .transactions
            .into_iter()
            .map(|ledger_tx| {
                // Convert Vec<u8> addresses to [u8; 20] addresses
                let from_addr = {
                    let mut addr_bytes = [0u8; 20];
                    let copy_len = ledger_tx.from.len().min(20);
                    addr_bytes[..copy_len].copy_from_slice(&ledger_tx.from[..copy_len]);
                    Address(addr_bytes)
                };
                let to_addr = {
                    let mut addr_bytes = [0u8; 20];
                    let copy_len = ledger_tx.to.len().min(20);
                    addr_bytes[..copy_len].copy_from_slice(&ledger_tx.to[..copy_len]);
                    Address(addr_bytes)
                };

                Transaction {
                    from: from_addr,
                    to: to_addr,
                    value: ledger_tx.amount,
                    gas_price: ledger_tx.fee, // Use fee as gas_price
                    gas_limit: 21000,         // Default gas limit
                    nonce: ledger_tx.nonce,
                    data: ledger_tx.data,
                    signature: ledger_tx
                        .signature
                        .map(|s| s.as_bytes().to_vec())
                        .unwrap_or_default(),
                    hash: CryptoHash::from_slice(ledger_tx.id.as_ref()),
                }
            })
            .collect();

        Block {
            header,
            transactions,
            signature: ledger_block.signature,
        }
    }
}
