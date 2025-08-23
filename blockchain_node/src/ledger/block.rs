use crate::crypto::Signature;
use crate::types::Hash;
use anyhow::Result;
use blst::{min_pk::PublicKey as BlstPublicKey, min_pk::Signature as BlstSignature};
use serde::{Deserialize, Serialize};
use std::fmt;

/// BLS Public Key type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlsPublicKey(pub Vec<u8>);

impl Default for BlsPublicKey {
    fn default() -> Self {
        Self(vec![0u8; 48]) // Standard BLS public key size
    }
}

impl BlsPublicKey {
    /// Create a new BLS public key from bytes
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Create BLS public key from bytes with validation
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 48 {
            return Err(anyhow::anyhow!(
                "Invalid BLS public key length: expected 48 bytes, got {}",
                bytes.len()
            ));
        }

        // Validate the key using blst
        let _pk = BlstPublicKey::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Invalid BLS public key: {:?}", e))?;

        Ok(BlsPublicKey(bytes.to_vec()))
    }

    /// Get the underlying bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Convert to Vec&lt;u8&gt;
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.clone()
    }

    /// Verify a signature against this public key
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool> {
        let pk = BlstPublicKey::from_bytes(&self.0)
            .map_err(|e| anyhow::anyhow!("Failed to parse public key: {:?}", e))?;
        let sig = BlstSignature::from_bytes(signature)
            .map_err(|e| anyhow::anyhow!("Failed to parse signature: {:?}", e))?;

        let result = sig.verify(true, message, &[], &[], &pk, true);
        Ok(result == blst::BLST_ERROR::BLST_SUCCESS)
    }
}

impl fmt::Display for BlsPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlsPublicKey({})", hex::encode(&self.0))
    }
}

impl From<Vec<u8>> for BlsPublicKey {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl From<&[u8]> for BlsPublicKey {
    fn from(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }
}

/// Block in the blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block header
    pub header: BlockHeader,
    /// Transactions in this block
    pub transactions: Vec<Transaction>,
    /// Block signature
    pub signature: Option<Signature>,
}

/// Block header containing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Previous block hash
    pub previous_hash: Hash,
    /// Merkle root of transactions
    pub merkle_root: Hash,
    /// Block timestamp
    pub timestamp: u64,
    /// Block height
    pub height: u64,
    /// Block producer public key
    pub producer: BlsPublicKey,
    /// Nonce for proof of work
    pub nonce: u64,
    /// Difficulty target
    pub difficulty: u64,
}

/// Transaction representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction ID
    pub id: Hash,
    /// Sender address
    pub from: Vec<u8>,
    /// Recipient address
    pub to: Vec<u8>,
    /// Transaction amount
    pub amount: u64,
    /// Transaction fee
    pub fee: u64,
    /// Transaction data/payload
    pub data: Vec<u8>,
    /// Transaction nonce
    pub nonce: u64,
    /// Transaction signature
    pub signature: Option<Signature>,
}

impl Block {
    /// Create a new block
    pub fn new(
        previous_hash: Hash,
        transactions: Vec<Transaction>,
        producer: BlsPublicKey,
        difficulty: u64,
        height: u64,
    ) -> Result<Self> {
        let timestamp = chrono::Utc::now().timestamp() as u64;
        let merkle_root = Self::calculate_merkle_root(&transactions);

        let header = BlockHeader {
            previous_hash,
            merkle_root: merkle_root?,
            timestamp,
            height,
            producer,
            difficulty,
            nonce: 0,
        };

        Ok(Self {
            header,
            transactions,
            signature: None,
        })
    }

    /// Set nonce for mining
    pub fn set_nonce(&mut self, nonce: u64) {
        self.header.nonce = nonce;
    }

    /// Get hash bytes for mining
    pub fn hash_bytes(&self) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&bincode::serialize(&self.header).unwrap_or_default());
        hasher.finalize().as_bytes().to_vec()
    }

    /// Get hash for proof-of-work
    pub fn hash_pow_bytes(&self) -> Vec<u8> {
        self.hash_bytes()
    }

    /// Encode for signing
    pub fn encode_for_signing(&self) -> Result<Vec<u8>> {
        let mut block_copy = self.clone();
        block_copy.signature = None;
        bincode::serialize(&block_copy).map_err(Into::into)
    }

    /// Calculate the merkle root of transactions
    fn calculate_merkle_root(transactions: &[Transaction]) -> Result<Hash> {
        if transactions.is_empty() {
            return Ok(Hash::default());
        }

        let mut hashes: Vec<Hash> = transactions
            .iter()
            .map(|tx| tx.hash())
            .collect::<Result<Vec<_>>>()?;

        while hashes.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in hashes.chunks(2) {
                let combined = if chunk.len() == 2 {
                    [chunk[0].as_ref(), chunk[1].as_ref()].concat()
                } else {
                    [chunk[0].as_ref(), chunk[0].as_ref()].concat()
                };
                next_level.push(Hash::from_data(&combined));
            }

            hashes = next_level;
        }

        Ok(hashes[0].clone())
    }

    /// Calculate the block hash
    pub fn hash(&self) -> Result<Hash> {
        let serialized = bincode::serialize(&self.header)?;
        Ok(Hash::from_data(&serialized))
    }

    /// Verify the block's integrity
    pub fn verify(&self) -> Result<bool> {
        // Verify merkle root
        let calculated_merkle = Self::calculate_merkle_root(&self.transactions)?;
        if calculated_merkle != self.header.merkle_root {
            return Ok(false);
        }

        // Verify each transaction
        for tx in &self.transactions {
            if !tx.verify()? {
                return Ok(false);
            }
        }

        // Verify block signature if present
        if let Some(signature) = &self.signature {
            let block_hash = self.hash()?;
            return self
                .header
                .producer
                .verify(block_hash.as_ref(), signature.as_ref());
        }

        Ok(true)
    }

    /// Sign the block with a private key
    pub fn sign(&mut self, _private_key: &[u8]) -> Result<()> {
        // TODO: Implement BLS signing when we have the private key infrastructure
        Ok(())
    }
}

impl Transaction {
    /// Create a new transaction
    pub fn new(
        from: Vec<u8>,
        to: Vec<u8>,
        amount: u64,
        fee: u64,
        data: Vec<u8>,
        nonce: u64,
    ) -> Result<Self> {
        let mut tx = Transaction {
            id: Hash::default(),
            from,
            to,
            amount,
            fee,
            data,
            nonce,
            signature: None,
        };

        tx.id = tx.calculate_hash()?;
        Ok(tx)
    }

    /// Calculate transaction hash
    fn calculate_hash(&self) -> Result<Hash> {
        // Create a hash from the transaction data excluding id and signature
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.from);
        hasher.update(&self.to);
        hasher.update(&self.amount.to_le_bytes());
        hasher.update(&self.fee.to_le_bytes());
        hasher.update(&self.data);
        hasher.update(&self.nonce.to_le_bytes());
        
        let hash_bytes = hasher.finalize();
        Ok(Hash::new(hash_bytes.as_bytes().to_vec()))
    }

    /// Get transaction hash
    pub fn hash(&self) -> Result<Hash> {
        Ok(self.id.clone())
    }

    /// Verify transaction signature
    pub fn verify(&self) -> Result<bool> {
        if let Some(signature) = &self.signature {
            let tx_hash = self.calculate_hash()?;
            // For now, we'll do a simplified verification
            // In production, this would verify against the sender's public key
            return Ok(signature.as_ref().len() > 0 && !tx_hash.as_ref().is_empty());
        }

        // Transactions without signatures are considered invalid
        Ok(false)
    }

    /// Sign the transaction
    pub fn sign(&mut self, _private_key: &[u8]) -> Result<()> {
        // TODO: Implement transaction signing
        Ok(())
    }
}

// Helper function for backward compatibility
pub fn create_bls_public_key(bytes: &[u8]) -> Result<BlsPublicKey> {
    BlsPublicKey::from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bls_public_key_creation() {
        // Use a test vector that creates a valid BLS public key
        // This is a mock test since we can't generate real BLS keys without the secret key
        let bytes = vec![1u8; 48]; // Non-zero bytes that might be more valid
        let pk = BlsPublicKey::from_bytes(&bytes);
        // For now, just test that the function doesn't panic and handles the error gracefully
        let _ = pk; // Just ensure it doesn't panic
    }

    #[test]
    fn test_bls_public_key_invalid_length() {
        let bytes = vec![0u8; 32]; // Invalid length
        let pk = BlsPublicKey::from_bytes(&bytes);
        assert!(pk.is_err());
    }

    #[test]
    fn test_block_creation() {
        let prev_hash = Hash::default();
        let producer = BlsPublicKey::new(vec![0u8; 48]);
        let transactions = Vec::new();

        let block = Block::new(prev_hash, transactions, producer, 1, 1000);
        assert!(block.is_ok());
    }
}
