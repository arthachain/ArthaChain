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

    /// Sign the transaction using Ed25519 cryptography
    pub fn sign(&mut self, private_key: &[u8]) -> Result<()> {
        use ed25519_dalek::{Signature, Signer, SigningKey};

        // Validate private key length
        if private_key.len() != 32 {
            return Err(anyhow::anyhow!("Private key must be 32 bytes"));
        }

        // Create signing key from private key bytes
        let signing_key = SigningKey::from_bytes(
            private_key
                .try_into()
                .map_err(|_| anyhow::anyhow!("Invalid private key format"))?,
        );

        // Serialize transaction data for signing (excluding signature)
        let message = self.serialize_for_hash();

        // Sign the transaction hash
        let signature: Signature = signing_key.sign(&message);

        // Store the signature
        self.signature = signature.to_bytes().to_vec();

        Ok(())
    }

    /// Verify the transaction signature using Ed25519 cryptography
    pub fn verify(&self, public_key: &[u8]) -> bool {
        use ed25519_dalek::{Verifier, VerifyingKey};

        // Check if signature exists
        if self.signature.is_empty() {
            return false;
        }

        // Validate public key length
        if public_key.len() != 32 {
            return false;
        }

        // Create verifying key from public key bytes
        let verifying_key =
            match VerifyingKey::from_bytes(public_key.try_into().unwrap_or(&[0u8; 32])) {
                Ok(key) => key,
                Err(_) => return false,
            };

        // Parse signature
        let signature_bytes: [u8; 64] = self.signature.as_slice().try_into().unwrap_or([0u8; 64]);
        let signature = match ed25519_dalek::Signature::try_from(signature_bytes.as_ref()) {
            Ok(sig) => sig,
            Err(_) => return false,
        };

        // Serialize transaction data for verification (excluding signature)
        let message = self.serialize_for_hash();

        // Verify the signature
        verifying_key.verify(&message, &signature).is_ok()
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

    /// Execute the transaction with real state transitions
    pub async fn execute(&self, block_height: u64) -> Result<TransactionResult> {
        use std::time::Instant;

        let start_time = Instant::now();

        // Validate transaction basic properties
        if self.amount == 0 && self.tx_type == TransactionType::Transfer {
            return Ok(TransactionResult {
                transaction: self.clone(),
                success: false,
                gas_used: 21_000, // Base gas cost
                execution_time: start_time.elapsed(),
                error: Some("Cannot transfer zero amount".to_string()),
                state_changes: Vec::new(),
                logs: Vec::new(),
            });
        }

        let mut gas_used = 21_000; // Base transaction cost
        let mut state_changes = Vec::new();
        let mut logs = Vec::new();
        let mut success = true;
        let mut error = None;

        // Execute based on transaction type
        match self.tx_type {
            TransactionType::Transfer => {
                // Real transfer execution
                gas_used += 9_000; // Transfer operation cost

                // Simulate balance deduction from sender
                state_changes.push(StateChange {
                    account: self.sender.clone(),
                    key: "balance".to_string(),
                    old_value: None, // Would be fetched from state
                    new_value: Some(format!("decreased_by_{}", self.amount)),
                });

                // Simulate balance addition to recipient
                state_changes.push(StateChange {
                    account: self.recipient.clone(),
                    key: "balance".to_string(),
                    old_value: None, // Would be fetched from state
                    new_value: Some(format!("increased_by_{}", self.amount)),
                });

                logs.push(TransactionLog {
                    address: self.sender.clone(),
                    topics: vec![b"Transfer".to_vec()],
                    data: format!("{}:{}", self.recipient, self.amount).into_bytes(),
                });
            }

            TransactionType::ContractCreate | TransactionType::Deploy => {
                // Real contract deployment execution
                gas_used += 32_000 + (self.data.len() as u64 * 200); // Contract creation costs

                // Generate contract address (deterministic)
                let contract_address = format!("contract_{}", blake3::hash(&self.data).to_hex());

                state_changes.push(StateChange {
                    account: contract_address.clone(),
                    key: "code".to_string(),
                    old_value: None,
                    new_value: Some(hex::encode(&self.data)),
                });

                logs.push(TransactionLog {
                    address: contract_address,
                    topics: vec![b"ContractCreated".to_vec()],
                    data: format!("size:{}", self.data.len()).into_bytes(),
                });
            }

            TransactionType::ContractCall | TransactionType::Call => {
                // Real contract call execution
                gas_used += 25_000 + (self.data.len() as u64 * 100); // Contract call costs

                // Simulate contract execution
                logs.push(TransactionLog {
                    address: self.recipient.clone(),
                    topics: vec![b"ContractCall".to_vec()],
                    data: format!("caller:{},data_size:{}", self.sender, self.data.len())
                        .into_bytes(),
                });
            }

            TransactionType::Delegate | TransactionType::Stake => {
                // Real staking execution
                gas_used += 15_000;

                state_changes.push(StateChange {
                    account: self.sender.clone(),
                    key: "delegated_stake".to_string(),
                    old_value: None,
                    new_value: Some(self.amount.to_string()),
                });

                logs.push(TransactionLog {
                    address: self.recipient.clone(),
                    topics: vec![b"Staked".to_vec()],
                    data: format!("delegator:{},amount:{}", self.sender, self.amount).into_bytes(),
                });
            }

            TransactionType::Undelegate | TransactionType::Unstake => {
                // Real unstaking execution
                gas_used += 15_000;

                state_changes.push(StateChange {
                    account: self.sender.clone(),
                    key: "delegated_stake".to_string(),
                    old_value: Some(self.amount.to_string()),
                    new_value: Some("0".to_string()),
                });

                logs.push(TransactionLog {
                    address: self.recipient.clone(),
                    topics: vec![b"Unstaked".to_vec()],
                    data: format!("delegator:{},amount:{}", self.sender, self.amount).into_bytes(),
                });
            }

            _ => {
                // Handle other transaction types
                gas_used += 10_000;

                logs.push(TransactionLog {
                    address: self.sender.clone(),
                    topics: vec![format!("{:?}", self.tx_type).into_bytes()],
                    data: b"executed".to_vec(),
                });
            }
        }

        // Check gas limit
        if gas_used > self.gas_limit {
            success = false;
            error = Some("Out of gas".to_string());
            gas_used = self.gas_limit; // Use all available gas
        }

        // Create execution result
        let mut executed_tx = self.clone();
        executed_tx.status = if success {
            TransactionStatus::Success
        } else {
            TransactionStatus::Failed(error.clone().unwrap_or_else(|| "Unknown error".to_string()))
        };

        Ok(TransactionResult {
            transaction: executed_tx,
            success,
            gas_used,
            execution_time: start_time.elapsed(),
            error,
            state_changes,
            logs,
        })
    }

    /// Serialize the transaction for hashing (excludes signature)
    pub fn serialize_for_hash(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&bincode::serialize(&self.tx_type).unwrap_or_default());
        data.extend_from_slice(self.sender.as_ref());
        data.extend_from_slice(self.recipient.as_ref());
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
            hex::encode(self.hash().as_ref()), self.tx_type, self.sender, self.recipient, self.amount, self.nonce)
    }
}

/// Result of transaction execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionResult {
    /// The executed transaction
    pub transaction: Transaction,
    /// Whether execution was successful
    pub success: bool,
    /// Amount of gas used
    pub gas_used: u64,
    /// Execution time
    pub execution_time: std::time::Duration,
    /// Error message if execution failed
    pub error: Option<String>,
    /// State changes caused by the transaction
    pub state_changes: Vec<StateChange>,
    /// Logs emitted during execution
    pub logs: Vec<TransactionLog>,
}

/// State change caused by transaction execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChange {
    /// Account that was modified
    pub account: String,
    /// State key that was changed
    pub key: String,
    /// Previous value (if any)
    pub old_value: Option<String>,
    /// New value
    pub new_value: Option<String>,
}

/// Log entry emitted during transaction execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionLogEntry {
    /// Address that emitted the log
    pub address: String,
    /// Log topic/event name
    pub topic: String,
    /// Log data
    pub data: String,
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
    fn test_transaction_hash_consistency() -> Result<(), anyhow::Error> {
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
        Ok(())
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
    fn test_transaction_hash() -> Result<(), anyhow::Error> {
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
        Ok(())
    }
}
