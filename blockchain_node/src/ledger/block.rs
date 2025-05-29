use crate::ledger::state::State as BlockchainState;
use crate::ledger::transaction::Transaction;
use crate::ledger::{BlockValidationError, ConsensusError, TransactionError};
use crate::types::Hash;
#[cfg(feature = "crypto")]
use blake2::{Blake2b512, Digest};
use blake3;
use hex;
use serde::{Deserialize, Serialize};
#[cfg(feature = "crypto")]
use sha3::Keccak256;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(feature = "bls")]
use threshold_crypto::{PublicKey as BlsPublicKey, Signature as BlsSignature};

/// Consensus status of a block
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ConsensusStatus {
    /// Block is proposed but not yet validated
    Proposed,
    /// Block has been validated by the proposer
    Validated,
    /// Block has passed pre-commit phase
    PreCommitted,
    /// Block has been committed
    Committed,
    /// Block has been finalized
    Finalized,
    /// Block has been rejected
    Rejected,
}

/// Signature type for validators
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SignatureType {
    /// Ed25519 signature
    Ed25519,
    /// BLS signature
    BLS,
}

/// Represents a block in the blockchain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    /// Block header
    pub header: BlockHeader,
    /// Block body containing transactions
    pub body: BlockBody,
    /// Consensus-related information
    pub consensus: ConsensusInfo,
    /// Whether this is a genesis block
    pub is_genesis: bool,
}

/// Block header containing metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Version of the block
    pub version: u32,
    /// Previous block hash
    pub previous_hash: Hash,
    /// Merkle root of transactions
    pub merkle_root: Hash,
    /// Block timestamp (seconds since UNIX epoch)
    pub timestamp: u64,
    /// Block height
    pub height: u64,
    /// Nonce for mining
    pub nonce: u64,
    /// Block hash (BLAKE2b)
    pub hash: Hash,
    /// Shard ID
    pub shard_id: u32,
    /// Mining difficulty target
    pub difficulty: u64,
    /// ID of the block proposer
    pub proposer_id: String,
}

/// Block body containing actual transaction data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockBody {
    /// Transactions in this block
    pub transactions: Vec<Transaction>,
}

/// Consensus information for SVBFT
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusInfo {
    /// Current consensus status
    pub status: ConsensusStatus,
    /// List of validators who have validated this block
    pub validator_signatures: Vec<ValidatorSignature>,
    /// Timestamp of last status change
    pub status_timestamp: u64,
    /// Social verification data
    pub sv_data: SocialVerificationData,
    /// Sharding information
    pub shard_id: u64,
    /// Cross-shard references
    pub cross_shard_refs: Vec<CrossShardRef>,
}

/// Signature from a validator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidatorSignature {
    /// Validator ID (public key)
    pub validator_id: String,
    /// Signature of the block hash
    pub signature: Vec<u8>,
    /// Timestamp when signed
    pub timestamp: u64,
    /// Signature type (Ed25519 or BLS)
    pub signature_type: SignatureType,
}

/// Reference to a block in another shard
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossShardRef {
    /// Shard ID
    pub shard_id: u64,
    /// Block hash
    pub block_hash: Hash,
    /// Block height
    pub height: u64,
}

/// Social verification data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocialVerificationData {
    /// Overall social verification score
    pub sv_score: f64,
    /// Compute contribution score
    pub compute_score: f64,
    /// Network contribution score
    pub network_score: f64,
    /// Storage contribution score
    pub storage_score: f64,
    /// Engagement score
    pub engagement_score: f64,
    /// AI security score
    pub ai_security_score: f64,
    /// Reputation history (recent blocks)
    pub reputation_history: Vec<f64>,
}

impl SocialVerificationData {
    /// Create new social verification data with default values
    fn new() -> Self {
        Self {
            sv_score: 0.0,
            compute_score: 0.0,
            network_score: 0.0,
            storage_score: 0.0,
            engagement_score: 0.0,
            ai_security_score: 0.0,
            reputation_history: Vec::new(),
        }
    }
}

impl BlockHeader {
    /// Create a new block header
    pub fn new(
        previous_hash: Hash,
        merkle_root: Hash,
        height: u64,
        difficulty: u64,
        proposer_id: String,
        shard_id: u32,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            version: 1,
            previous_hash,
            merkle_root,
            timestamp,
            height,
            nonce: 0,
            hash: Hash::default(),
            shard_id,
            difficulty,
            proposer_id,
        }
    }

    /// Serialize the header for hashing
    pub fn serialize_for_hash(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(
            4 + // version
            32 + // previous_hash
            32 + // merkle_root
            8 + // timestamp
            8 + // height
            8 + // nonce
            4 + // shard_id
            8 + // difficulty
            self.proposer_id.len(),
        );

        buffer.extend_from_slice(&self.version.to_le_bytes());
        buffer.extend_from_slice(self.previous_hash.as_bytes());
        buffer.extend_from_slice(self.merkle_root.as_bytes());
        buffer.extend_from_slice(&self.timestamp.to_le_bytes());
        buffer.extend_from_slice(&self.height.to_le_bytes());
        buffer.extend_from_slice(&self.nonce.to_le_bytes());
        buffer.extend_from_slice(&self.shard_id.to_le_bytes());
        buffer.extend_from_slice(&self.difficulty.to_le_bytes());
        buffer.extend_from_slice(self.proposer_id.as_bytes());

        buffer
    }
}

impl BlockBody {
    /// Create a new block body
    fn new(transactions: Vec<Transaction>) -> Self {
        Self { transactions }
    }
}

impl ConsensusInfo {
    /// Create a new consensus info object
    fn new(shard_id: u64) -> Self {
        Self {
            status: ConsensusStatus::Proposed,
            validator_signatures: Vec::new(),
            status_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sv_data: SocialVerificationData::new(),
            shard_id,
            cross_shard_refs: Vec::new(),
        }
    }

    /// Add a validator signature with specified type
    pub fn add_signature(
        &mut self,
        validator_id: String,
        signature: Vec<u8>,
        sig_type: SignatureType,
    ) -> bool {
        // Check if this validator has already signed
        if self
            .validator_signatures
            .iter()
            .any(|s| s.validator_id == validator_id)
        {
            return false;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.validator_signatures.push(ValidatorSignature {
            validator_id,
            signature,
            timestamp,
            signature_type: sig_type,
        });

        true
    }

    /// Update the consensus status
    pub fn update_status(&mut self, new_status: ConsensusStatus) -> Result<(), ConsensusError> {
        // Validate state transition
        match (self.status.clone(), new_status.clone()) {
            // Valid transitions
            (ConsensusStatus::Proposed, ConsensusStatus::Validated) => {}
            (ConsensusStatus::Validated, ConsensusStatus::PreCommitted) => {}
            (ConsensusStatus::PreCommitted, ConsensusStatus::Committed) => {}
            (ConsensusStatus::Committed, ConsensusStatus::Finalized) => {}
            // Any state can transition to rejected
            (_, ConsensusStatus::Rejected) => {}
            // Invalid transitions
            (_from, _to) => {
                return Err(ConsensusError::InvalidStateTransition);
            }
        }

        self.status = new_status;
        self.status_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(())
    }

    /// Check if we have enough signatures for finality
    pub fn has_finality(&self, required_signatures: usize) -> bool {
        self.validator_signatures.len() >= required_signatures
    }

    /// Add a cross-shard reference
    pub fn add_cross_shard_ref(&mut self, shard_id: u64, block_hash: Hash, height: u64) {
        // Don't add duplicate references
        if self
            .cross_shard_refs
            .iter()
            .any(|r| r.shard_id == shard_id && r.height == height)
        {
            return;
        }

        self.cross_shard_refs.push(CrossShardRef {
            shard_id,
            block_hash,
            height,
        });
    }

    /// Update social verification score based on multiple factors
    pub fn update_sv_score(
        &mut self,
        compute_score: f64,
        network_score: f64,
        storage_score: f64,
        engagement_score: f64,
        ai_security_score: f64,
    ) {
        let sv_data = &mut self.sv_data;

        // Update component scores
        sv_data.compute_score = compute_score;
        sv_data.network_score = network_score;
        sv_data.storage_score = storage_score;
        sv_data.engagement_score = engagement_score;
        sv_data.ai_security_score = ai_security_score;

        // Calculate weighted score
        // Weights can be adjusted based on importance
        const COMPUTE_WEIGHT: f64 = 0.2;
        const NETWORK_WEIGHT: f64 = 0.2;
        const STORAGE_WEIGHT: f64 = 0.2;
        const ENGAGEMENT_WEIGHT: f64 = 0.2;
        const AI_WEIGHT: f64 = 0.2;

        sv_data.sv_score = (compute_score * COMPUTE_WEIGHT)
            + (network_score * NETWORK_WEIGHT)
            + (storage_score * STORAGE_WEIGHT)
            + (engagement_score * ENGAGEMENT_WEIGHT)
            + (ai_security_score * AI_WEIGHT);

        // Add to history (keep last 10 scores)
        sv_data.reputation_history.push(sv_data.sv_score);
        if sv_data.reputation_history.len() > 10 {
            sv_data.reputation_history.remove(0);
        }
    }

    pub fn verify_aggregate_bls_signature(&self, message: &[u8]) -> Result<bool, ConsensusError> {
        // Get BLS signatures and public keys
        let mut signatures = Vec::new();
        let mut public_keys = Vec::new();

        for sig in &self.validator_signatures {
            if sig.signature_type == SignatureType::BLS {
                // Parse signature - convert Vec<u8> to fixed size array
                if sig.signature.len() != 96 {
                    return Err(ConsensusError::InvalidSignature);
                }

                let sig_bytes: [u8; 96] = sig.signature[0..96]
                    .try_into()
                    .map_err(|_| ConsensusError::InvalidSignature)?;

                let bls_sig = BlsSignature::from_bytes(sig_bytes)
                    .map_err(|_| ConsensusError::InvalidSignature)?;
                signatures.push(bls_sig);

                // Parse public key
                let pk_bytes =
                    hex::decode(&sig.validator_id).map_err(|_| ConsensusError::InvalidSignature)?;

                if pk_bytes.len() != 48 {
                    return Err(ConsensusError::InvalidSignature);
                }

                let pk_array: [u8; 48] = pk_bytes[0..48]
                    .try_into()
                    .map_err(|_| ConsensusError::InvalidSignature)?;

                let public_key = BlsPublicKey::from_bytes(pk_array)
                    .map_err(|_| ConsensusError::InvalidSignature)?;
                public_keys.push(public_key);
            }
        }

        // If no BLS signatures, nothing to verify
        if signatures.is_empty() {
            return Ok(true);
        }

        // In threshold_crypto, we need to check each signature individually
        // since there's no direct way to combine signatures without a polynomial
        for (i, sig) in signatures.iter().enumerate() {
            if !public_keys[i].verify(sig, message) {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

impl Block {
    /// Create a new block
    pub fn new(
        previous_hash: Hash,
        transactions: Vec<Transaction>,
        height: u64,
        difficulty: u64,
        proposer_id: String,
        shard_id: u64,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let merkle_root = Self::calculate_merkle_root(&transactions);

        let mut header = BlockHeader {
            version: 1,
            previous_hash,
            merkle_root,
            timestamp,
            height,
            nonce: 0,
            hash: Hash::default(),
            shard_id: shard_id as u32,
            difficulty,
            proposer_id,
        };

        header.hash = Self::hash_header(&header);

        let body = BlockBody::new(transactions);

        // Create consensus info
        let consensus = ConsensusInfo::new(shard_id);

        // Return the complete block
        Self {
            header,
            body,
            consensus,
            is_genesis: false,
        }
    }

    /// Create a genesis block for a shard
    pub fn genesis(shard_id: u64) -> Self {
        let transactions = Vec::new();
        let previous_hash = Hash([0u8; 32].to_vec()); // Zero hash for genesis
        let height = 0;
        let difficulty = 1; // Start with easy difficulty
        let proposer_id = "genesis".to_string();

        let mut block = Self::new(
            previous_hash,
            transactions,
            height,
            difficulty,
            proposer_id,
            shard_id,
        );

        block.is_genesis = true;
        block
    }

    /// Calculate the Merkle root of transactions
    fn calculate_merkle_root(transactions: &[Transaction]) -> Hash {
        if transactions.is_empty() {
            return Hash([0u8; 32].to_vec());
        }

        let mut hashes: Vec<Hash> = transactions
            .iter()
            .map(|tx| {
                let hash_str = tx.hash();
                let hash_bytes = hex::decode(hash_str).unwrap_or_else(|_| vec![0; 32]);
                Hash(hash_bytes)
            })
            .collect();

        while hashes.len() > 1 {
            let mut new_hashes = Vec::new();
            for chunk in hashes.chunks(2) {
                let mut data = Vec::new();
                data.extend_from_slice(chunk[0].as_bytes());
                if chunk.len() > 1 {
                    data.extend_from_slice(chunk[1].as_bytes());
                } else {
                    data.extend_from_slice(chunk[0].as_bytes());
                }
                let hash = blake3::hash(&data);
                new_hashes.push(Hash(hash.as_bytes().to_vec()));
            }
            hashes = new_hashes;
        }

        hashes[0].clone()
    }

    /// Hash a block header
    fn hash_header(header: &BlockHeader) -> Hash {
        let data = header.serialize_for_hash();
        let hash = blake3::hash(&data);
        Hash(hash.as_bytes().to_vec())
    }

    /// Calculate block hash
    pub fn calculate_hash(&self) -> Hash {
        let data = self.header.serialize_for_hash();
        let hash = blake3::hash(&data);
        Hash(hash.as_bytes().to_vec())
    }

    /// Increment the nonce and recalculate hash (for mining)
    pub fn increment_nonce(&mut self) {
        self.header.nonce += 1;
        self.header.hash = self.calculate_hash();
    }

    /// Check if the block meets the difficulty target
    pub fn meets_difficulty(&self) -> bool {
        // Simple difficulty check: first N bits must be zero
        // where N is determined by the difficulty
        // Higher difficulty value means more zero bits required

        let bits_to_check = self.header.difficulty as usize;
        let bytes_to_check = bits_to_check / 8;
        let remaining_bits = bits_to_check % 8;

        // Check full bytes first
        for byte in self.header.hash.as_bytes().iter().take(bytes_to_check) {
            if *byte != 0 {
                return false;
            }
        }

        // Check remaining bits in the next byte
        if remaining_bits > 0 && bytes_to_check < self.header.hash.as_bytes().len() {
            let mask = 0xFF << (8 - remaining_bits);
            if self.header.hash.as_bytes()[bytes_to_check] & mask != 0 {
                return false;
            }
        }

        true
    }

    /// Sign the block using the validator's private key (Ed25519)
    /// Returns true if signature was successfully added
    pub fn sign_as_validator(&mut self, validator_id: String, signature: Vec<u8>) -> bool {
        self.consensus
            .add_signature(validator_id, signature, SignatureType::Ed25519)
    }

    /// Sign the block using BLS signature
    #[cfg(feature = "bls")]
    pub fn sign_as_validator_bls(&mut self, validator_id: String, signature: Vec<u8>) -> bool {
        self.consensus
            .add_signature(validator_id, signature, SignatureType::BLS)
    }

    /// Change the consensus status
    pub fn update_consensus_status(
        &mut self,
        new_status: ConsensusStatus,
    ) -> Result<(), ConsensusError> {
        self.consensus.update_status(new_status)
    }

    /// Total number of validator signatures
    pub fn signature_count(&self) -> usize {
        self.consensus.validator_signatures.len()
    }

    /// Check if the block has reached finality
    pub fn is_finalized(&self, required_signatures: usize) -> bool {
        self.consensus.status == ConsensusStatus::Finalized
            || (self.consensus.status == ConsensusStatus::Committed
                && self.consensus.has_finality(required_signatures))
    }

    /// Get total fees from all transactions
    pub fn total_fees(&self) -> u64 {
        self.body
            .transactions
            .iter()
            .map(|tx| tx.gas_price * tx.gas_limit)
            .sum()
    }

    /// Validate all transactions in the block
    pub fn validate_transactions(&self, state: &BlockchainState) -> Result<(), TransactionError> {
        // Track used nonces to prevent double-spending within the same block
        let mut used_nonces: HashMap<String, HashSet<u64>> = HashMap::new();

        for tx in &self.body.transactions {
            // Basic transaction validation
            tx.validate()?;

            // Check sender's balance
            let sender_balance = state.get_balance(&tx.sender)?;
            let required_amount = tx.amount + tx.fee();
            if sender_balance < required_amount {
                return Err(TransactionError::InsufficientFunds);
            }

            // Check for nonce replay within same block
            let sender_nonces = used_nonces
                .entry(tx.sender.clone())
                .or_insert_with(HashSet::new);
            if !sender_nonces.insert(tx.nonce) {
                return Err(TransactionError::DuplicateNonce);
            }

            // Check nonce sequence
            let expected_nonce = state.get_next_nonce(&tx.sender)?;
            if tx.nonce != expected_nonce {
                return Err(TransactionError::InvalidNonce);
            }
        }

        Ok(())
    }

    /// Validate transactions in parallel (using tokio for async)
    pub async fn validate_transactions_parallel(
        &self,
        state: Arc<BlockchainState>,
    ) -> Result<(), TransactionError> {
        use tokio::task;

        // First validate each transaction independently in parallel
        let validation_tasks: Vec<_> = self
            .body
            .transactions
            .iter()
            .map(|tx| {
                let tx_clone = tx.clone();

                task::spawn(async move {
                    // Run basic transaction validation
                    tx_clone.validate()
                })
            })
            .collect();

        // Wait for all validation tasks to complete
        for task in validation_tasks {
            task.await
                .map_err(|e| TransactionError::Internal(e.to_string()))?
                .map_err(|e| e)?;
        }

        // Some validations must be done sequentially (nonce ordering, balances)
        // Track used nonces to prevent double-spending within the same block
        let mut used_nonces: HashMap<String, HashSet<u64>> = HashMap::new();

        for tx in &self.body.transactions {
            // Check sender's balance
            let sender_balance = state.get_balance(&tx.sender)?;
            let required_amount = tx.amount + tx.fee();
            if sender_balance < required_amount {
                return Err(TransactionError::InsufficientFunds);
            }

            // Check for nonce replay within same block
            let sender_nonces = used_nonces
                .entry(tx.sender.clone())
                .or_insert_with(HashSet::new);
            if !sender_nonces.insert(tx.nonce) {
                return Err(TransactionError::DuplicateNonce);
            }

            // Check nonce sequence
            let expected_nonce = state.get_next_nonce(&tx.sender)?;
            if tx.nonce != expected_nonce {
                return Err(TransactionError::InvalidNonce);
            }
        }

        Ok(())
    }

    /// Update social verification scores
    pub fn update_sv_scores(
        &mut self,
        compute_score: f64,
        network_score: f64,
        storage_score: f64,
        engagement_score: f64,
        ai_security_score: f64,
    ) {
        self.consensus.update_sv_score(
            compute_score,
            network_score,
            storage_score,
            engagement_score,
            ai_security_score,
        );
    }

    /// Hash data using BLAKE2b
    #[cfg(feature = "crypto")]
    pub fn hash_data_blake2b(data: &[u8]) -> Hash {
        let mut hasher = Blake2b512::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result[..32]);
        Hash(hash.to_vec())
    }

    /// Hash data using BLAKE2b (no-crypto fallback)
    #[cfg(not(feature = "crypto"))]
    pub fn hash_data_blake2b(data: &[u8]) -> Hash {
        let mut hash = [0u8; 32];
        for (i, chunk) in data.chunks(32).enumerate() {
            for (j, &byte) in chunk.iter().enumerate() {
                if i * 32 + j < 32 {
                    hash[i * 32 + j] = byte;
                }
            }
        }
        Hash(hash.to_vec())
    }

    /// Hash data using Keccak256
    #[cfg(feature = "crypto")]
    pub fn hash_data_keccak256(data: &[u8]) -> Hash {
        let mut hasher = Keccak256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result[..32]);
        Hash(hash.to_vec())
    }

    /// Hash data using Keccak256 (no crypto feature)
    #[cfg(not(feature = "crypto"))]
    pub fn hash_data_keccak256(data: &[u8]) -> Hash {
        let mut output = [0u8; 32];
        output.copy_from_slice(&data[..32.min(data.len())]);
        Hash(output.to_vec())
    }

    /// Hash data using BLAKE3
    #[cfg(feature = "crypto")]
    pub fn hash_data_blake3(data: &[u8]) -> Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();
        Hash(hash.as_bytes().to_vec())
    }

    /// Hash data using BLAKE3 (no-crypto fallback)
    #[cfg(not(feature = "crypto"))]
    pub fn hash_data_blake3(data: &[u8]) -> Hash {
        Self::hash_data_blake2b(data) // Fallback to BLAKE2b in no-crypto mode
    }

    /// Validate if all transactions in this block are valid
    pub fn validate(&self, state: &BlockchainState) -> Result<(), BlockValidationError> {
        // Validate block hash
        let calculated_hash = self.calculate_hash();
        if calculated_hash != self.header.hash {
            return Err(BlockValidationError::Other(
                "Invalid block hash".to_string(),
            ));
        }

        // Validate transactions
        self.validate_transactions(state)
            .map_err(|e| BlockValidationError::InvalidTransactions(e))?;

        Ok(())
    }

    pub fn from_hash(_hash: &Hash) -> Result<Self, Box<dyn std::error::Error>> {
        // Implementation would typically load from storage
        // For now, return an error
        Err("Not implemented".into())
    }

    pub fn hash(&self) -> Hash {
        self.header.hash.clone()
    }

    /// Encode block data for cryptographic signing
    pub fn encode_for_signing(&self) -> Result<Vec<u8>, anyhow::Error> {
        // Serialize the block header for signing (excluding the hash itself)
        let mut buffer = Vec::new();

        // Add header data without the hash field
        buffer.extend_from_slice(&self.header.version.to_le_bytes());
        buffer.extend_from_slice(self.header.previous_hash.as_bytes());
        buffer.extend_from_slice(self.header.merkle_root.as_bytes());
        buffer.extend_from_slice(&self.header.timestamp.to_le_bytes());
        buffer.extend_from_slice(&self.header.height.to_le_bytes());
        buffer.extend_from_slice(&self.header.nonce.to_le_bytes());
        buffer.extend_from_slice(&self.header.shard_id.to_le_bytes());
        buffer.extend_from_slice(&self.header.difficulty.to_le_bytes());
        buffer.extend_from_slice(self.header.proposer_id.as_bytes());

        // Add body data (transaction hashes)
        for transaction in &self.body.transactions {
            buffer.extend_from_slice(transaction.hash().as_bytes());
        }

        Ok(buffer)
    }
}

// Implement Default trait for Block
impl Default for Block {
    fn default() -> Self {
        Self::genesis(0)
    }
}

/// Extension trait for Block with mining-related operations
pub trait BlockExt {
    /// Get a hex string representation of the block hash
    fn hash_str(&self) -> String;

    /// Get the raw bytes of the block hash
    fn hash_bytes(&self) -> Hash;

    /// Get the bytes used for proof-of-work validation
    fn hash_pow_bytes(&self) -> Hash;

    /// Set the nonce value for mining
    fn set_nonce(&mut self, nonce: u64);
}

impl BlockExt for Block {
    /// Get a hex string representation of the block hash
    fn hash_str(&self) -> String {
        hex::encode(self.header.hash.as_bytes())
    }

    /// Get the raw bytes of the block hash
    fn hash_bytes(&self) -> Hash {
        self.header.hash.clone()
    }

    /// Get the bytes used for proof-of-work validation
    fn hash_pow_bytes(&self) -> Hash {
        let data = self.header.serialize_for_hash();
        let hash = blake3::hash(&data);
        Hash(hash.as_bytes().to_vec())
    }

    /// Set the nonce value for mining
    fn set_nonce(&mut self, nonce: u64) {
        self.header.nonce = nonce;
        self.header.hash = self.calculate_hash();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::transaction::TransactionType;

    #[test]
    fn test_block_creation() {
        let block = Block::new(
            Hash([0; 32].to_vec()),
            Vec::new(),
            0,
            1,
            "proposer1".to_string(),
            1,
        );
        assert_eq!(block.header.height, 0);
    }

    #[test]
    fn test_genesis_block() {
        let genesis = Block::genesis(1);
        assert!(genesis.is_genesis);
    }

    #[test]
    fn test_merkle_root() {
        let transactions = vec![
            Transaction::new(
                TransactionType::Transfer,
                "sender1".to_string(),
                "recipient1".to_string(),
                100,
                1,
                1,    // gas_price
                1000, // gas_limit
                "data1".as_bytes().to_vec(),
            ),
            Transaction::new(
                TransactionType::Transfer,
                "sender2".to_string(),
                "recipient2".to_string(),
                200,
                1,
                1,    // gas_price
                1000, // gas_limit
                "data2".as_bytes().to_vec(),
            ),
        ];
        let merkle_root = Block::calculate_merkle_root(&transactions);
        assert_ne!(merkle_root, Hash([0; 32].to_vec()));
    }

    #[test]
    fn test_block_validation() {
        let _state = BlockchainState::new(&crate::config::Config::new()).unwrap();
        let block = Block::new(
            Hash([0; 32].to_vec()),
            Vec::new(),
            0,
            1,
            "proposer1".to_string(),
            1,
        );
        assert!(block.validate(&_state).is_ok());
    }

    #[test]
    fn test_block_chain() {
        let _state = BlockchainState::new(&crate::config::Config::new()).unwrap();
        let block1 = Block::new(
            Hash([0; 32].to_vec()),
            Vec::new(),
            0,
            1,
            "proposer1".to_string(),
            1,
        );
        let block2 = Block::new(block1.hash(), Vec::new(), 1, 1, "proposer2".to_string(), 1);
        assert_eq!(block2.header.previous_hash, block1.hash());
    }

    #[test]
    fn test_block_serialization() {
        let block = Block::new(
            Hash([0; 32].to_vec()),
            Vec::new(),
            0,
            1,
            "proposer1".to_string(),
            1,
        );
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: Block = serde_json::from_str(&serialized).unwrap();
        assert_eq!(block.header.height, deserialized.header.height);
    }

    #[test]
    fn test_block_with_invalid_transactions() {
        let _state = BlockchainState::new(&crate::config::Config::new()).unwrap();
        let transactions = vec![Transaction::new(
            TransactionType::Transfer,
            "sender1".to_string(),
            "recipient1".to_string(),
            100,
            1,
            1,    // gas_price
            1000, // gas_limit
            "data1".as_bytes().to_vec(),
        )];
        let block = Block::new(
            Hash([0; 32].to_vec()),
            transactions,
            0,
            1,
            "proposer1".to_string(),
            1,
        );
        assert!(block.validate(&_state).is_err());
    }

    #[test]
    fn test_block_hash_consistency() {
        let block = Block::new(
            Hash([0; 32].to_vec()),
            Vec::new(),
            0,
            1,
            "proposer1".to_string(),
            1,
        );
        let hash1 = block.hash();
        let hash2 = block.calculate_hash();
        assert_eq!(hash1, hash2);
    }
}
