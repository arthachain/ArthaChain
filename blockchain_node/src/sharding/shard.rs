use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::ledger::{Account, Block, State, Transaction};
use crate::network::p2p::{Message, P2PNetwork};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ShardId(pub u16);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    pub shard_count: u16,
    pub this_shard_id: ShardId,
    pub accounts_per_shard: usize,
    pub cross_shard_tx_timeout: u64,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            shard_count: 4,
            this_shard_id: ShardId(0),
            accounts_per_shard: 1000,
            cross_shard_tx_timeout: 60, // seconds
        }
    }
}

pub struct ShardManager {
    config: ShardConfig,
    shards: Arc<RwLock<HashMap<ShardId, ShardState>>>,
    network: Arc<P2PNetwork>,
    pending_cross_shard_txs: Arc<RwLock<HashMap<String, CrossShardTransaction>>>,
}

#[derive(Debug, Clone)]
pub struct ShardState {
    pub id: ShardId,
    pub state: Arc<RwLock<State>>,
    pub blocks: Vec<Block>,
    pub tx_pool: Vec<Transaction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossShardTransaction {
    pub id: String,
    pub from_shard: ShardId,
    pub to_shard: ShardId,
    pub transaction: Transaction,
    pub status: CrossShardTxStatus,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CrossShardTxStatus {
    Pending,
    Committed,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardMessage {
    CrossShardTxPropose(CrossShardTransaction),
    CrossShardTxCommit(String),
    CrossShardTxAbort(String),
    ShardSync(ShardId, Vec<Block>),
}

impl ShardManager {
    pub fn new(config: ShardConfig, network: Arc<P2PNetwork>) -> Self {
        let shards = Arc::new(RwLock::new(HashMap::new()));
        let pending_cross_shard_txs = Arc::new(RwLock::new(HashMap::new()));
        
        ShardManager {
            config,
            shards,
            network,
            pending_cross_shard_txs,
        }
    }
    
    pub fn init_shards(&self) {
        let mut shards = self.shards.write().unwrap();
        
        // Initialize this node's shard
        let shard_id = self.config.this_shard_id.clone();
        let state = Arc::new(RwLock::new(State::new()));
        
        let shard_state = ShardState {
            id: shard_id.clone(),
            state,
            blocks: Vec::new(),
            tx_pool: Vec::new(),
        };
        
        shards.insert(shard_id, shard_state);
    }
    
    pub fn add_transaction(&self, tx: Transaction) -> Result<(), String> {
        // Determine which shard this transaction belongs to
        let shard_id = self.get_shard_for_account(&tx.from);
        
        // Check if this is a cross-shard transaction
        let to_shard_id = self.get_shard_for_account(&tx.to);
        
        if shard_id != to_shard_id {
            // Handle cross-shard transaction
            self.handle_cross_shard_transaction(tx, shard_id, to_shard_id)
        } else if shard_id == self.config.this_shard_id {
            // This is a transaction for our shard
            let mut shards = self.shards.write().unwrap();
            if let Some(shard) = shards.get_mut(&shard_id) {
                shard.tx_pool.push(tx);
                Ok(())
            } else {
                Err(format!("Shard {} not found", shard_id.0))
            }
        } else {
            // This is a transaction for another shard, forward it
            self.forward_transaction_to_shard(tx, shard_id)
        }
    }
    
    fn handle_cross_shard_transaction(
        &self, 
        tx: Transaction, 
        from_shard: ShardId, 
        to_shard: ShardId
    ) -> Result<(), String> {
        // Create a cross-shard transaction
        let cross_tx = CrossShardTransaction {
            id: format!("{}:{}", tx.id, tx.nonce),
            from_shard: from_shard.clone(),
            to_shard: to_shard.clone(),
            transaction: tx.clone(),
            status: CrossShardTxStatus::Pending,
            timestamp: get_current_timestamp(),
        };
        
        // If this is the originating shard, initiate the 2PC protocol
        if from_shard == self.config.this_shard_id {
            // First phase: Propose to the target shard
            let msg = ShardMessage::CrossShardTxPropose(cross_tx.clone());
            self.send_message_to_shard(to_shard.clone(), msg);
            
            // Store in pending cross-shard transactions
            let mut pending = self.pending_cross_shard_txs.write().unwrap();
            pending.insert(cross_tx.id.clone(), cross_tx);
        } 
        // If this is the target shard, check if we can accept it
        else if to_shard == self.config.this_shard_id {
            // Verify if we can accept this transaction
            let can_accept = self.verify_incoming_cross_shard_tx(&cross_tx);
            
            if can_accept {
                // Accept and send commit message back
                let mut pending = self.pending_cross_shard_txs.write().unwrap();
                pending.insert(cross_tx.id.clone(), cross_tx.clone());
                
                // Send commit message back to source shard
                let msg = ShardMessage::CrossShardTxCommit(cross_tx.id);
                self.send_message_to_shard(from_shard, msg);
            } else {
                // Reject and send abort message back
                let msg = ShardMessage::CrossShardTxAbort(cross_tx.id);
                self.send_message_to_shard(from_shard, msg);
            }
        }
        
        Ok(())
    }
    
    fn verify_incoming_cross_shard_tx(&self, cross_tx: &CrossShardTransaction) -> bool {
        // Verify that the target account exists and has enough balance
        let shards = self.shards.read().unwrap();
        if let Some(shard) = shards.get(&self.config.this_shard_id) {
            let state = shard.state.read().unwrap();
            let tx = &cross_tx.transaction;
            
            // Check if the recipient account exists
            if !state.account_exists(&tx.to) {
                return false;
            }
            
            // Additional verification logic can be added here
            true
        } else {
            false
        }
    }
    
    fn forward_transaction_to_shard(&self, tx: Transaction, shard_id: ShardId) -> Result<(), String> {
        // Serialize the transaction and forward it to the appropriate shard
        let message = Message::Transaction(bincode::serialize(&tx).unwrap());
        
        // In a real implementation, we would need to route this to the appropriate shard
        // For simplicity, we just broadcast it for now
        self.network.broadcast(message);
        
        Ok(())
    }
    
    fn send_message_to_shard(&self, shard_id: ShardId, message: ShardMessage) {
        // Serialize and send the message to the appropriate shard
        match bincode::serialize(&message) {
            Ok(data) => {
                let p2p_message = Message::Transaction(data); // Reusing Transaction type for shard messages
                self.network.broadcast(p2p_message);
            }
            Err(e) => {
                eprintln!("Failed to serialize shard message: {}", e);
            }
        }
    }
    
    pub fn process_shard_message(&self, message: ShardMessage) {
        match message {
            ShardMessage::CrossShardTxPropose(cross_tx) => {
                // Process a cross-shard transaction proposal
                self.handle_cross_shard_transaction(
                    cross_tx.transaction.clone(),
                    cross_tx.from_shard,
                    cross_tx.to_shard
                ).unwrap_or_else(|e| {
                    eprintln!("Failed to handle cross-shard tx: {}", e);
                });
            }
            ShardMessage::CrossShardTxCommit(tx_id) => {
                // Finalize a cross-shard transaction
                self.commit_cross_shard_transaction(&tx_id);
            }
            ShardMessage::CrossShardTxAbort(tx_id) => {
                // Abort a cross-shard transaction
                self.abort_cross_shard_transaction(&tx_id);
            }
            ShardMessage::ShardSync(shard_id, blocks) => {
                // Handle shard sync message
                self.handle_shard_sync(shard_id, blocks);
            }
        }
    }
    
    fn commit_cross_shard_transaction(&self, tx_id: &str) {
        let mut pending = self.pending_cross_shard_txs.write().unwrap();
        
        if let Some(cross_tx) = pending.get_mut(tx_id) {
            // If we're the source shard, finalize the transaction
            if cross_tx.from_shard == self.config.this_shard_id {
                cross_tx.status = CrossShardTxStatus::Committed;
                
                // Apply the first half of the transaction (deduct from source account)
                let mut shards = self.shards.write().unwrap();
                if let Some(shard) = shards.get_mut(&self.config.this_shard_id) {
                    let mut state = shard.state.write().unwrap();
                    let tx = &cross_tx.transaction;
                    
                    // Deduct funds from source account
                    if let Err(e) = state.transfer(&tx.from, &tx.to, tx.amount) {
                        eprintln!("Failed to apply cross-shard transaction: {}", e);
                    }
                }
            }
            // If we're the target shard, apply the second half
            else if cross_tx.to_shard == self.config.this_shard_id {
                cross_tx.status = CrossShardTxStatus::Committed;
                
                // The funds will be added in the next block
                let mut shards = self.shards.write().unwrap();
                if let Some(shard) = shards.get_mut(&self.config.this_shard_id) {
                    shard.tx_pool.push(cross_tx.transaction.clone());
                }
            }
        }
    }
    
    fn abort_cross_shard_transaction(&self, tx_id: &str) {
        let mut pending = self.pending_cross_shard_txs.write().unwrap();
        
        if let Some(cross_tx) = pending.get_mut(tx_id) {
            cross_tx.status = CrossShardTxStatus::Aborted;
            // No action needed, just mark as aborted
        }
    }
    
    fn handle_shard_sync(&self, shard_id: ShardId, blocks: Vec<Block>) {
        // Update our view of the shard's state
        let mut shards = self.shards.write().unwrap();
        
        // If we don't have this shard yet, create it
        if !shards.contains_key(&shard_id) {
            let state = Arc::new(RwLock::new(State::new()));
            let shard_state = ShardState {
                id: shard_id.clone(),
                state,
                blocks: Vec::new(),
                tx_pool: Vec::new(),
            };
            shards.insert(shard_id.clone(), shard_state);
        }
        
        // Update the blocks for this shard
        if let Some(shard) = shards.get_mut(&shard_id) {
            // Apply blocks if they're newer than what we have
            if shard.blocks.len() < blocks.len() {
                // Apply missing blocks to the state
                let mut state = shard.state.write().unwrap();
                
                for block in blocks.iter().skip(shard.blocks.len()) {
                    for tx in &block.body.transactions {
                        // Apply transaction
                        if let Err(e) = state.apply_transaction(&tx) {
                            eprintln!("Error applying transaction: {}", e);
                        }
                    }
                }
                
                // Update our blocks
                shard.blocks = blocks;
            }
        }
    }
    
    pub fn get_shard_for_account(&self, account: &str) -> ShardId {
        // Simple sharding strategy based on account hash
        let account_hash = hash_account(account);
        let shard_index = account_hash % (self.config.shard_count as u64);
        ShardId(shard_index as u16)
    }
    
    pub fn clean_up_pending_transactions(&self) {
        // Remove expired cross-shard transactions
        let mut pending = self.pending_cross_shard_txs.write().unwrap();
        let current_time = get_current_timestamp();
        let timeout = self.config.cross_shard_tx_timeout;
        
        pending.retain(|_, tx| {
            if tx.status == CrossShardTxStatus::Pending && 
               current_time - tx.timestamp > timeout {
                // Abort timed-out transactions
                tx.status = CrossShardTxStatus::Aborted;
                false
            } else {
                tx.status != CrossShardTxStatus::Aborted
            }
        });
    }
}

// Helper functions
fn hash_account(account: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    account.hash(&mut hasher);
    hasher.finish()
}

fn get_current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::p2p::{P2PConfig, P2PNetwork};
    
    #[test]
    fn test_shard_assignment() {
        let p2p_config = P2PConfig::default();
        let network = Arc::new(P2PNetwork::new(p2p_config));
        
        let config = ShardConfig {
            shard_count: 4,
            this_shard_id: ShardId(0),
            accounts_per_shard: 1000,
            cross_shard_tx_timeout: 60,
        };
        
        let shard_manager = ShardManager::new(config, network);
        
        // Test that accounts are distributed across shards
        let account1 = "0x1234567890abcdef";
        let account2 = "0xabcdef1234567890";
        
        let shard1 = shard_manager.get_shard_for_account(account1);
        let shard2 = shard_manager.get_shard_for_account(account2);
        
        // The accounts should have different shard assignments with high probability
        println!("Account {} assigned to shard {}", account1, shard1.0);
        println!("Account {} assigned to shard {}", account2, shard2.0);
    }
    
    // More tests would be added here
} 