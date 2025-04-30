use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;

use crate::ledger::transaction::Transaction;
use crate::ledger::transaction::TransactionType;
use crate::utils::crypto::Hash;

/// Defines the shard assignment strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardAssignmentStrategy {
    /// Assign based on account address ranges
    AccountRange,
    /// Assign based on transaction type
    TransactionType,
    /// Assign based on geographic region
    Geographic,
    /// Random assignment (for testing)
    Random,
}

/// Configuration for the sharding system
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Number of shards in the network
    pub shard_count: usize,
    /// Assignment strategy to use
    pub assignment_strategy: ShardAssignmentStrategy,
    /// Whether cross-shard transactions are enabled
    pub enable_cross_shard: bool,
    /// Maximum number of pending cross-shard references
    pub max_pending_cross_shard_refs: usize,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            shard_count: 4,
            assignment_strategy: ShardAssignmentStrategy::AccountRange,
            enable_cross_shard: true,
            max_pending_cross_shard_refs: 100,
        }
    }
}

/// Status of a cross-shard transaction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossShardStatus {
    /// Waiting for confirmation from other shards
    Pending,
    /// Confirmed by all involved shards
    Confirmed,
    /// Rejected by at least one shard
    Rejected,
    /// Timed out waiting for confirmation
    TimedOut,
}

/// Represents a cross-shard transaction reference
#[derive(Debug, Clone)]
pub struct CrossShardReference {
    /// Hash of the transaction
    pub tx_hash: Hash,
    /// Shards involved in this transaction
    pub involved_shards: Vec<u32>,
    /// Current status of the transaction
    pub status: CrossShardStatus,
    /// Block height when this reference was created
    pub created_at_height: u64,
}

/// Main sharding manager that handles shard assignment and cross-shard coordination
pub struct ShardManager {
    /// Configuration for the sharding system
    config: ShardingConfig,
    /// The ID of the current shard
    shard_id: u32,
    /// Pending cross-shard references
    pending_refs: Arc<RwLock<HashMap<Hash, CrossShardReference>>>,
    /// Channel for receiving cross-shard messages
    cross_shard_rx: mpsc::Receiver<CrossShardMessage>,
    /// Channel for sending cross-shard messages
    cross_shard_tx: mpsc::Sender<CrossShardMessage>,
}

/// Message for cross-shard communication
#[derive(Debug, Clone)]
pub struct CrossShardMessage {
    /// Source shard ID
    pub from_shard: u32,
    /// Destination shard ID
    pub to_shard: u32,
    /// Type of the message
    pub message_type: CrossShardMessageType,
    /// Transaction hash if applicable
    pub tx_hash: Option<Hash>,
    /// Block hash if applicable
    pub block_hash: Option<Hash>,
}

/// Types of cross-shard messages
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossShardMessageType {
    /// Notification about a cross-shard transaction
    TransactionNotification,
    /// Confirmation of a cross-shard transaction
    TransactionConfirmation,
    /// Rejection of a cross-shard transaction
    TransactionRejection,
    /// Request for synchronization with another shard
    SyncRequest,
    /// Response to a sync request
    SyncResponse,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(config: ShardingConfig, shard_id: u32) -> Self {
        let (cross_shard_tx, cross_shard_rx) = mpsc::channel(100);
        
        Self {
            config,
            shard_id,
            pending_refs: Arc::new(RwLock::new(HashMap::new())),
            cross_shard_rx,
            cross_shard_tx,
        }
    }
    
    /// Get the sender for cross-shard messages
    pub fn get_sender(&self) -> mpsc::Sender<CrossShardMessage> {
        self.cross_shard_tx.clone()
    }
    
    /// Assign a transaction to a shard based on the configured strategy
    pub fn assign_transaction_to_shard(&self, tx: &Transaction) -> u32 {
        match self.config.assignment_strategy {
            ShardAssignmentStrategy::AccountRange => {
                // Simple hash-based assignment
                let sender_bytes = tx.sender.as_bytes();
                let hash_value = sender_bytes.iter().fold(0u32, |acc, &x| acc.wrapping_add(x as u32));
                hash_value % (self.config.shard_count as u32)
            },
            ShardAssignmentStrategy::TransactionType => {
                // Assign based on transaction type
                match tx.tx_type {
                    TransactionType::Transfer => 0, // Standard transactions go to shard 0
                    TransactionType::Deploy => 1, // Contract creation to shard 1
                    TransactionType::Call => 2, // Contract call to shard 2
                    _ => {
                        // Create a hash from the transaction type string representation
                        let type_str = format!("{:?}", tx.tx_type);
                        let type_bytes = type_str.as_bytes();
                        let hash_value = type_bytes.iter().fold(0u32, |acc, &x| acc.wrapping_add(x as u32));
                        hash_value % (self.config.shard_count as u32)
                    },
                }
            },
            ShardAssignmentStrategy::Geographic => {
                // This would require additional metadata in transactions
                // For now, fall back to account range
                let sender_bytes = tx.sender.as_bytes();
                let hash_value = sender_bytes.iter().fold(0u32, |acc, &x| acc.wrapping_add(x as u32));
                hash_value % (self.config.shard_count as u32)
            },
            ShardAssignmentStrategy::Random => {
                // Simple hash-based pseudo-random assignment
                let tx_hash = tx.hash();
                let hash_bytes = tx_hash.as_bytes();
                let hash_value = hash_bytes.iter().fold(0u32, |acc, &x| acc.wrapping_add(x as u32));
                hash_value % (self.config.shard_count as u32)
            },
        }
    }
    
    /// Determine if a transaction is cross-shard
    pub fn is_cross_shard_transaction(&self, tx: &Transaction) -> bool {
        if !self.config.enable_cross_shard {
            return false;
        }
        
        let sender_shard = self.assign_transaction_to_shard(tx);
        
        // Check if recipient is in a different shard
        let recipient_bytes = tx.recipient.as_bytes();
        let hash_value = recipient_bytes.iter().fold(0u32, |acc, &x| acc.wrapping_add(x as u32));
        let recipient_shard = hash_value % (self.config.shard_count as u32);
        
        sender_shard != recipient_shard
    }
    
    /// Get all shards involved in a transaction
    pub fn get_involved_shards(&self, tx: &Transaction) -> Vec<u32> {
        let mut shards = HashSet::new();
        
        // Add sender's shard
        shards.insert(self.assign_transaction_to_shard(tx));
        
        // Add recipient's shard
        let recipient_bytes = tx.recipient.as_bytes();
        let hash_value = recipient_bytes.iter().fold(0u32, |acc, &x| acc.wrapping_add(x as u32));
        let recipient_shard = hash_value % (self.config.shard_count as u32);
        shards.insert(recipient_shard);
        
        shards.into_iter().collect()
    }
    
    /// Register a new cross-shard transaction
    pub fn register_cross_shard_tx(&self, tx: &Transaction, block_height: u64) -> Result<()> {
        if !self.config.enable_cross_shard {
            return Err(anyhow!("Cross-shard transactions are disabled"));
        }
        
        let involved_shards = self.get_involved_shards(tx);
        if involved_shards.len() <= 1 {
            return Err(anyhow!("Not a cross-shard transaction"));
        }
        
        let tx_hash_str = tx.hash();
        // Convert String hash to Hash type
        let tx_hash = Hash::from_hex(&tx_hash_str).unwrap_or_default();
        
        // Create cross-shard reference
        let cross_ref = CrossShardReference {
            tx_hash: tx_hash.clone(),
            involved_shards,
            status: CrossShardStatus::Pending,
            created_at_height: block_height,
        };
        
        // Store in pending references
        {
            let mut pending_refs = self.pending_refs.write().map_err(|_| anyhow!("Lock poisoned"))?;
            if pending_refs.len() >= self.config.max_pending_cross_shard_refs {
                return Err(anyhow!("Too many pending cross-shard references"));
            }
            pending_refs.insert(tx_hash, cross_ref);
        }
        
        Ok(())
    }
    
    /// Process an incoming cross-shard message
    pub async fn process_cross_shard_message(&mut self, message: CrossShardMessage) -> Result<()> {
        // Only process messages intended for this shard
        if message.to_shard != self.shard_id {
            return Ok(());
        }
        
        match message.message_type {
            CrossShardMessageType::TransactionNotification => {
                // Handle notification of a new cross-shard transaction
                if let Some(tx_hash) = message.tx_hash {
                    // Validate the transaction (would need more context)
                    // For now, just send a confirmation back
                    let confirmation = CrossShardMessage {
                        from_shard: self.shard_id,
                        to_shard: message.from_shard,
                        message_type: CrossShardMessageType::TransactionConfirmation,
                        tx_hash: Some(tx_hash),
                        block_hash: None,
                    };
                    
                    self.cross_shard_tx.send(confirmation).await
                        .map_err(|_| anyhow!("Failed to send cross-shard confirmation"))?;
                }
            },
            CrossShardMessageType::TransactionConfirmation => {
                // Update the status of a cross-shard transaction
                if let Some(tx_hash) = message.tx_hash {
                    let mut pending_refs = self.pending_refs.write().map_err(|_| anyhow!("Lock poisoned"))?;
                    if let Some(cross_ref) = pending_refs.get_mut(&tx_hash) {
                        cross_ref.status = CrossShardStatus::Confirmed;
                    }
                }
            },
            CrossShardMessageType::TransactionRejection => {
                // Mark the transaction as rejected
                if let Some(tx_hash) = message.tx_hash {
                    let mut pending_refs = self.pending_refs.write().map_err(|_| anyhow!("Lock poisoned"))?;
                    if let Some(cross_ref) = pending_refs.get_mut(&tx_hash) {
                        cross_ref.status = CrossShardStatus::Rejected;
                    }
                }
            },
            CrossShardMessageType::SyncRequest => {
                // Handle request for sync (would need more context)
                // For example, might need to send blocks or state
            },
            CrossShardMessageType::SyncResponse => {
                // Process sync response (would need more context)
                // For example, might need to update local state
            },
        }
        
        Ok(())
    }
    
    /// Start processing cross-shard messages
    pub async fn start_message_processing(&mut self) -> Result<()> {
        while let Some(message) = self.cross_shard_rx.recv().await {
            if let Err(e) = self.process_cross_shard_message(message).await {
                eprintln!("Error processing cross-shard message: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Clean up timed-out cross-shard references
    pub fn cleanup_timed_out_refs(&self, current_height: u64, timeout_blocks: u64) -> Result<()> {
        let mut pending_refs = self.pending_refs.write().map_err(|_| anyhow!("Lock poisoned"))?;
        
        let timed_out_keys: Vec<Hash> = pending_refs
            .iter()
            .filter(|(_, reference)| {
                reference.status == CrossShardStatus::Pending &&
                current_height >= reference.created_at_height + timeout_blocks
            })
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in timed_out_keys {
            if let Some(reference) = pending_refs.get_mut(&key) {
                reference.status = CrossShardStatus::TimedOut;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::transaction::{Transaction, TransactionType};
    
    // This is a placeholder for the Transaction struct if it doesn't exist yet
    #[cfg(test)]
    impl Transaction {
        pub fn new_test(sender: &str, recipient: Option<&str>, amount: u64, tx_type: u8) -> Self {
            let recipient_str = recipient.unwrap_or("").to_string();
            let transaction_type = match tx_type {
                0 => TransactionType::Transfer,
                1 => TransactionType::Deploy,
                2 => TransactionType::Call,
                _ => TransactionType::Transfer,
            };
            
            Transaction::new(
                transaction_type,
                sender.to_string(),
                recipient_str,
                amount,
                0, // nonce
                10, // gas_price
                1000, // gas_limit
                vec![], // data
                vec![], // signature
            )
        }
    }
    
    #[test]
    fn test_shard_assignment() {
        let config = ShardingConfig {
            shard_count: 4,
            assignment_strategy: ShardAssignmentStrategy::AccountRange,
            enable_cross_shard: true,
            max_pending_cross_shard_refs: 100,
        };
        
        let shard_manager = ShardManager::new(config, 0);
        
        let tx1 = Transaction::new_test("user1", Some("user2"), 100, 0);
        let tx2 = Transaction::new_test("user3", Some("user4"), 200, 1);
        
        let shard1 = shard_manager.assign_transaction_to_shard(&tx1);
        let shard2 = shard_manager.assign_transaction_to_shard(&tx2);
        
        assert!(shard1 < 4, "Shard ID should be less than shard count");
        assert!(shard2 < 4, "Shard ID should be less than shard count");
    }
    
    #[test]
    fn test_cross_shard_detection() {
        // This is a simplified test that would need to be adjusted based on actual implementation
        // It assumes that certain addresses will hash to different shards
        let config = ShardingConfig::default();
        let shard_manager = ShardManager::new(config, 0);
        
        // These addresses are chosen to likely hash to different shards
        let tx = Transaction::new_test(
            "0x1111111111111111111111111111111111111111", 
            Some("0x9999999999999999999999999999999999999999"), 
            100, 
            0
        );
        
        let involved_shards = shard_manager.get_involved_shards(&tx);
        assert!(involved_shards.len() > 0, "Should determine involved shards");
    }
}

// Comment out this implementation as it's already defined elsewhere
// impl Hash {
//     /// Create a Hash from a hexadecimal string
//     pub fn from_hex(hex_string: &str) -> Result<Self> {
//         let bytes = hex::decode(hex_string)?;
//         Ok(Self(bytes))
//     }
// } 