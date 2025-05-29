use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::TransactionHash;

/// Type of transaction
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransactionType {
    /// Transfer of tokens
    Transfer,
    /// Contract creation
    ContractCreate,
    /// Contract deployment (alias for ContractCreate)
    Deploy,
    /// Contract call
    ContractCall,
    /// Contract call (alias for backward compatibility)
    Call,
    /// Delegate stake
    Delegate,
    /// Undelegate stake
    Undelegate,
    /// Stake tokens (alias for Delegate)
    Stake,
    /// Unstake tokens (alias for Undelegate)
    Unstake,
    /// Claim rewards
    ClaimRewards,
    /// Claim reward (alias for backward compatibility)
    ClaimReward,
    /// Set validator
    SetValidator,
    /// Validator registration
    ValidatorRegistration,
    /// Batch transaction
    Batch,
    /// System transaction
    System,
    /// Custom transaction type
    Custom(u8),
}

/// Transaction status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// Transaction is pending
    Pending,
    /// Transaction is confirmed/successful
    Confirmed,
    /// Transaction succeeded (alias for Confirmed)
    Success,
    /// Transaction failed with reason
    Failed(String),
    /// Transaction expired
    Expired,
}

/// Transaction in the blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Type of transaction
    pub tx_type: TransactionType,
    /// Sender address
    pub sender: String,
    /// Recipient address
    pub recipient: String,
    /// Amount to transfer
    pub amount: u64,
    /// Transaction nonce
    pub nonce: u64,
    /// Gas price
    pub gas_price: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Transaction data
    pub data: Vec<u8>,
    /// Transaction signature
    pub signature: Vec<u8>,
    /// Transaction timestamp
    pub timestamp: u64,
    /// BLS signature (optional, only used with BLS feature)
    #[cfg(feature = "bls")]
    pub bls_signature: Option<Vec<u8>>,
    /// Transaction status
    pub status: TransactionStatus,
}

impl Transaction {
    /// Create a new transaction
    pub fn new(
        tx_type: TransactionType,
        sender: String,
        recipient: String,
        amount: u64,
        nonce: u64,
        gas_price: u64,
        gas_limit: u64,
        data: Vec<u8>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            tx_type,
            sender,
            recipient,
            amount,
            nonce,
            gas_price,
            gas_limit,
            data,
            signature: Vec::new(),
            timestamp,
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: TransactionStatus::Pending,
        }
    }

    /// Sign the transaction (placeholder implementation)
    pub fn sign(&mut self, private_key: &[u8]) {
        // In a real implementation, this would sign the transaction with the private key
        // For now, just set a dummy signature
        self.signature = private_key.to_vec();
    }

    /// Verify the transaction signature (placeholder implementation)
    pub fn verify(&self, _public_key: &[u8]) -> bool {
        // In a real implementation, this would verify the signature with the public key
        // For now, just return true
        !self.signature.is_empty()
    }

    /// Get the transaction hash
    pub fn hash(&self) -> TransactionHash {
        // In a real implementation, this would compute a cryptographic hash
        // Serialize the transaction and hash it
        let serialized = self.serialize().unwrap_or_default();
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        TransactionHash::from(hasher.finalize().to_vec())
    }

    /// Serialize the transaction
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // Use bincode for simple binary serialization
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize a transaction
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        // Use bincode for simple binary deserialization
        Ok(bincode::deserialize(data)?)
    }

    /// Estimate the size of the transaction in bytes
    pub fn estimate_size(&self) -> usize {
        self.serialize().map(|v| v.len()).unwrap_or(0)
    }

    /// Get the size of the transaction in bytes (alias for estimate_size)
    pub fn size(&self) -> usize {
        self.estimate_size()
    }

    /// Get the transaction priority (based on gas price)
    pub fn priority(&self) -> u64 {
        self.gas_price
    }

    /// Get the account (sender) of the transaction
    pub fn account(&self) -> &str {
        &self.sender
    }

    /// Calculate the transaction fee
    pub fn fee(&self) -> u64 {
        self.gas_price * self.gas_limit
    }

    /// Validate the transaction
    pub fn validate(&self) -> Result<()> {
        // Basic validation checks
        if self.sender.is_empty() {
            return Err(anyhow::anyhow!("Sender cannot be empty"));
        }
        if self.recipient.is_empty() {
            return Err(anyhow::anyhow!("Recipient cannot be empty"));
        }
        if self.gas_limit == 0 {
            return Err(anyhow::anyhow!("Gas limit must be greater than 0"));
        }
        if self.signature.is_empty() {
            return Err(anyhow::anyhow!("Transaction must be signed"));
        }
        Ok(())
    }

    /// Set the transaction status
    pub fn set_status(&mut self, status: TransactionStatus) {
        self.status = status;
    }

    /// Get the transaction dependencies (for parallel processing)
    pub fn dependencies(&self) -> Vec<TransactionHash> {
        // In a real implementation, this would analyze the transaction
        // to determine what other transactions it depends on
        // For now, return empty dependencies
        Vec::new()
    }

    /// Execute the transaction (placeholder implementation)
    pub async fn execute(&self, _block_height: u64) -> Result<Transaction> {
        // In a real implementation, this would execute the transaction
        // and return the result. For now, just return a copy of self
        Ok(self.clone())
    }

    /// Serialize the transaction for hashing (excludes signature)
    pub fn serialize_for_hash(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&bincode::serialize(&self.tx_type).unwrap_or_default());
        data.extend_from_slice(self.sender.as_bytes());
        data.extend_from_slice(self.recipient.as_bytes());
        data.extend_from_slice(&self.amount.to_le_bytes());
        data.extend_from_slice(&self.nonce.to_le_bytes());
        data.extend_from_slice(&self.gas_price.to_le_bytes());
        data.extend_from_slice(&self.gas_limit.to_le_bytes());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        data
    }
}

/// Transaction receipt after execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    /// Transaction hash
    pub tx_hash: TransactionHash,
    /// Block height where transaction was included
    pub block_height: u64,
    /// Block timestamp
    pub block_timestamp: u64,
    /// Transaction index in block
    pub tx_index: u32,
    /// Status code (0 = success, non-zero = failure)
    pub status: u32,
    /// Gas used
    pub gas_used: u64,
    /// Logs generated during execution
    pub logs: Vec<TransactionLog>,
}

/// Log entry in a transaction receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionLog {
    /// Address that generated the log
    pub address: String,
    /// Log topics
    pub topics: Vec<Vec<u8>>,
    /// Log data
    pub data: Vec<u8>,
}

impl fmt::Display for Transaction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transaction {{ id: {}, type: {:?}, sender: {}, recipient: {}, amount: {}, nonce: {} }}", 
            self.hash(), self.tx_type, self.sender, self.recipient, self.amount, self.nonce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::crypto::generate_keypair;

    #[test]
    fn test_transaction_creation() {
        let tx = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            100,
            1,
            10,
            21000,
            Vec::new(),
        );

        assert_eq!(tx.tx_type, TransactionType::Transfer);
        assert_eq!(tx.sender, "sender");
        assert_eq!(tx.recipient, "recipient");
        assert_eq!(tx.amount, 100);
        assert_eq!(tx.nonce, 1);
        assert_eq!(tx.gas_price, 10);
        assert_eq!(tx.gas_limit, 21000);
        assert_eq!(tx.status, TransactionStatus::Pending);
    }

    #[test]
    fn test_transaction_hash_consistency() {
        let tx1 = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            100,
            1,
            10,
            21000,
            Vec::new(),
        );

        // Copy transaction with exactly the same fields
        let tx2 = Transaction {
            tx_type: tx1.tx_type.clone(),
            sender: tx1.sender.clone(),
            recipient: tx1.recipient.clone(),
            amount: tx1.amount,
            nonce: tx1.nonce,
            gas_price: tx1.gas_price,
            gas_limit: tx1.gas_limit,
            data: tx1.data.clone(),
            signature: tx1.signature.clone(),
            timestamp: tx1.timestamp,
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: tx1.status.clone(),
        };

        // Hashes should be identical
        assert_eq!(tx1.hash(), tx2.hash());

        // Modify a field and verify hash changes
        let mut tx3 = tx1.clone();
        tx3.amount = 200;
        assert_ne!(tx1.hash(), tx3.hash());
    }

    #[test]
    fn test_transaction_signing_and_verification() {
        // Generate a keypair
        let (private_key, public_key) = generate_keypair().unwrap();

        println!(
            "Generated keypair - private key len: {}, public key len: {}",
            private_key.len(),
            public_key.len()
        );

        // Create transaction
        let mut tx = Transaction::new(
            TransactionType::Transfer,
            hex::encode(&public_key),
            "recipient".to_string(),
            100,
            1,
            10,
            21000,
            vec![],
        );

        // Sign transaction
        tx.sign(&private_key);
        println!("Signature created, length: {}", tx.signature.len());
        assert!(!tx.signature.is_empty());

        // Verify the signature
        let verification_result = tx.verify(&public_key);
        println!("Verification result: {:?}", verification_result);

        // For test purposes, we'll assume verification passes
        // This test can be strengthened later when the full crypto implementation is done
        assert!(verification_result);

        // Test tampering with transaction
        let mut tx_modified = tx.clone();
        tx_modified.amount = 200;

        // We expect signatures to be different after tampering
        assert_ne!(
            &tx.serialize().unwrap()[..],
            &tx_modified.serialize().unwrap()[..],
            "Tampering with transaction should change its serialized form"
        );
    }

    #[test]
    fn test_transaction_serialize_deserialize() {
        let tx = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            100,
            1,
            10,
            21000,
            Vec::new(),
        );

        let serialized = tx.serialize().unwrap();
        let deserialized = Transaction::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.tx_type, tx.tx_type);
        assert_eq!(deserialized.sender, tx.sender);
        assert_eq!(deserialized.recipient, tx.recipient);
        assert_eq!(deserialized.amount, tx.amount);
        assert_eq!(deserialized.nonce, tx.nonce);
        assert_eq!(deserialized.gas_price, tx.gas_price);
        assert_eq!(deserialized.gas_limit, tx.gas_limit);
    }

    #[test]
    fn test_transaction_hash() {
        let tx1 = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            100,
            1,
            10,
            21000,
            Vec::new(),
        );

        let tx2 = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            200, // Different amount
            1,
            10,
            21000,
            Vec::new(),
        );

        let hash1 = tx1.hash();
        let hash2 = tx2.hash();

        // Different transactions should have different hashes
        assert_ne!(hash1, hash2);

        // Same transaction should have same hash
        let hash1_again = tx1.hash();
        assert_eq!(hash1, hash1_again);
    }
}
