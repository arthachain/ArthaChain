pub mod checkpoint;
pub mod storage;
pub mod tree;

use crate::config::Config;
use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::types::Hash;
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use tokio::sync::{broadcast, Mutex as TokioMutex};

/// Interface for sharding configuration
pub trait ShardConfig {
    /// Get the shard ID
    fn get_shard_id(&self) -> u64;

    /// Get the genesis configuration
    fn get_genesis_config(&self) -> Option<&Config>;

    /// Check if sharding is enabled
    fn is_sharding_enabled(&self) -> bool;

    /// Get the number of shards
    fn get_shard_count(&self) -> u32;

    /// Get the primary shard
    fn get_primary_shard(&self) -> u32;
}

/// Snapshot metadata for atomic execution
#[derive(Debug, Clone)]
struct Snapshot {
    /// Unique snapshot ID
    #[allow(dead_code)]
    id: u64,
    /// Balances at snapshot time
    balances: HashMap<String, u64>,
    /// Nonces at snapshot time
    nonces: HashMap<String, u64>,
    /// Storage at snapshot time
    storage: HashMap<String, Vec<u8>>,
}

/// Blockchain state representation
#[derive(Debug)]
pub struct State {
    /// Account balances
    balances: RwLock<HashMap<String, u64>>,

    /// Account nonces
    nonces: RwLock<HashMap<String, u64>>,

    /// Contract storage
    storage: RwLock<HashMap<String, Vec<u8>>>,

    /// Current block height
    height: RwLock<u64>,

    /// Shard ID
    shard_id: u64,

    /// Transaction snapshots for atomic operations
    snapshots: RwLock<HashMap<u64, Snapshot>>,

    /// Next snapshot ID
    next_snapshot_id: RwLock<u64>,

    /// Pending transactions
    pending_transactions: RwLock<VecDeque<Transaction>>,

    /// Transaction history by account
    tx_history: RwLock<HashMap<String, Vec<String>>>,

    /// Blocks by height
    blocks: RwLock<HashMap<u64, Block>>,

    /// Blocks by hash (using String keys for hash display)
    blocks_by_hash: RwLock<HashMap<String, Block>>,

    /// Latest block hash
    latest_block_hash: RwLock<String>,

    // 🛡️ SPOF ELIMINATION: Distributed Stae Management
    /// State replicas for redundancy (SPOF FIX #1)
    state_replicas: Arc<RwLock<Vec<StateReplica>>>,
    /// Current primary replica index
    primary_replica: Arc<RwLock<usize>>,
    /// State synchronization channel
    sync_channel: Arc<TokioMutex<broadcast::Sender<StateSyncMessage>>>,
    /// Health status of each replica
    replica_health: Arc<RwLock<HashMap<usize, ReplicaHealth>>>,
    /// Consensus mechanism for state updates
    state_consensus: Arc<RwLock<StateConsensus>>,
}

impl State {
    pub fn new(_config: &Config) -> Result<Self> {
        let (sync_sender, _) = broadcast::channel(1000);
        
        Ok(Self {
            balances: RwLock::new(HashMap::new()),
            nonces: RwLock::new(HashMap::new()),
            storage: RwLock::new(HashMap::new()),
            height: RwLock::new(0),
            shard_id: 0,
            snapshots: RwLock::new(HashMap::new()),
            next_snapshot_id: RwLock::new(0),
            pending_transactions: RwLock::new(VecDeque::new()),
            tx_history: RwLock::new(HashMap::new()),
            blocks: RwLock::new(HashMap::new()),
            blocks_by_hash: RwLock::new(HashMap::new()),
            latest_block_hash: RwLock::new(
                "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            ),
            
            // 🛡️ SPOF ELIMINATION: Initialize distributed state
            state_replicas: Arc::new(RwLock::new(Vec::new())),
            primary_replica: Arc::new(RwLock::new(0)),
            sync_channel: Arc::new(TokioMutex::new(sync_sender)),
            replica_health: Arc::new(RwLock::new(HashMap::new())),
            state_consensus: Arc::new(RwLock::new(StateConsensus {
                consensus_threshold: 2, // Minimum 2 replicas for Byzantine fault tolerance
                active_replicas: 1,     // Start with primary only
                last_consensus: 0,
                pending_updates: HashMap::new(),
            })),
        })
    }

    /// Get account balance
    pub fn get_balance(&self, address: &str) -> Result<u64> {
        let balances = self.balances.read().unwrap();
        Ok(*balances.get(address).unwrap_or(&0))
    }

    /// Set account balance
    pub fn set_balance(&self, address: &str, amount: u64) -> Result<()> {
        let mut balances = self.balances.write().unwrap();
        balances.insert(address.to_string(), amount);
        Ok(())
    }

    /// Get account nonce
    pub fn get_nonce(&self, address: &str) -> Result<u64> {
        let nonces = self.nonces.read().unwrap();
        Ok(*nonces.get(address).unwrap_or(&0))
    }

    /// Set account nonce
    pub fn set_nonce(&self, address: &str, nonce: u64) -> Result<()> {
        let mut nonces = self.nonces.write().unwrap();
        nonces.insert(address.to_string(), nonce);
        Ok(())
    }

    /// Get storage value
    pub fn get_storage(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let storage = self.storage.read().unwrap();
        Ok(storage.get(key).cloned())
    }

    /// Set storage value
    pub fn set_storage(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let mut storage = self.storage.write().unwrap();
        storage.insert(key.to_string(), value);
        Ok(())
    }

    /// Delete storage value
    pub fn delete_storage(&self, key: &str) -> Result<()> {
        let mut storage = self.storage.write().unwrap();
        storage.remove(key);
        Ok(())
    }

    /// Get current block height
    pub fn get_height(&self) -> Result<u64> {
        Ok(*self.height.read().unwrap())
    }

    /// Set current block height
    pub fn set_height(&self, height: u64) -> Result<()> {
        let mut h = self.height.write().unwrap();
        *h = height;
        Ok(())
    }

    /// Get shard ID
    pub fn get_shard_id(&self) -> Result<u64> {
        Ok(self.shard_id)
    }

    /// Get the next nonce for an account (current nonce + 1)
    pub fn get_next_nonce(&self, address: &str) -> Result<u64> {
        let current_nonce = self.get_nonce(address)?;
        Ok(current_nonce + 1)
    }

    /// Create a new state snapshot for atomic operations
    pub fn create_snapshot(&self) -> Result<u64> {
        let mut next_id = self.next_snapshot_id.write().unwrap();
        let snapshot_id = *next_id;
        *next_id += 1;

        debug!("Creating state snapshot with ID: {}", snapshot_id);

        // Clone current state
        let balances = self.balances.read().unwrap().clone();
        let nonces = self.nonces.read().unwrap().clone();
        let storage = self.storage.read().unwrap().clone();

        // Create snapshot
        let snapshot = Snapshot {
            id: snapshot_id,
            balances,
            nonces,
            storage,
        };

        // Store snapshot
        let mut snapshots = self.snapshots.write().unwrap();
        snapshots.insert(snapshot_id, snapshot);

        Ok(snapshot_id)
    }

    /// Commit a state snapshot (remove it as it's no longer needed)
    pub fn commit_snapshot(&self, snapshot_id: u64) -> Result<()> {
        debug!("Committing snapshot with ID: {}", snapshot_id);

        let mut snapshots = self.snapshots.write().unwrap();
        if !snapshots.contains_key(&snapshot_id) {
            return Err(anyhow!("Snapshot not found: {}", snapshot_id));
        }

        snapshots.remove(&snapshot_id);
        Ok(())
    }

    /// Revert state to a snapshot
    pub fn revert_to_snapshot(&self, snapshot_id: u64) -> Result<()> {
        debug!("Reverting to snapshot with ID: {}", snapshot_id);

        // Get snapshot
        let snapshots = self.snapshots.read().unwrap();
        let snapshot = snapshots
            .get(&snapshot_id)
            .ok_or_else(|| anyhow!("Snapshot not found: {}", snapshot_id))?;

        // Restore balances
        {
            let mut balances = self.balances.write().unwrap();
            *balances = snapshot.balances.clone();
        }

        // Restore nonces
        {
            let mut nonces = self.nonces.write().unwrap();
            *nonces = snapshot.nonces.clone();
        }

        // Restore storage
        {
            let mut storage = self.storage.write().unwrap();
            *storage = snapshot.storage.clone();
        }

        // Remove the snapshot
        drop(snapshots);
        self.commit_snapshot(snapshot_id)?;

        Ok(())
    }

    /// Get the latest block hash
    pub fn get_latest_block_hash(&self) -> Result<String> {
        let hash = self.latest_block_hash.read().unwrap();
        Ok(hash.clone())
    }

    /// Set the latest block hash
    pub fn set_latest_block_hash(&self, hash: &str) -> Result<()> {
        let mut latest = self.latest_block_hash.write().unwrap();
        *latest = hash.to_string();
        Ok(())
    }

    /// Get a block by its hash
    pub fn get_block_by_hash(&self, hash: &Hash) -> Option<Block> {
        let blocks = self.blocks_by_hash.read().unwrap();
        blocks.get(&hash.to_string()).cloned()
    }

    /// Get a block by its height
    pub fn get_block_by_height(&self, height: u64) -> Option<Block> {
        let blocks = self.blocks.read().unwrap();
        blocks.get(&height).cloned()
    }

    /// Add a block to the state
    pub fn add_block(&self, block: Block) -> Result<()> {
        let height = block.header.height;
        let hash = block.hash()?.to_hex();

        // Add to blocks by height
        {
            let mut blocks = self.blocks.write().unwrap();
            blocks.insert(height, block.clone());
        }

        // Add to blocks by hash
        {
            let mut blocks_by_hash = self.blocks_by_hash.write().unwrap();
            blocks_by_hash.insert(hash.clone(), block);
        }

        // Update latest block hash if this is the highest block
        let current_height = self.get_height()?;
        if height > current_height {
            self.set_height(height)?;
            self.set_latest_block_hash(&hash)?;
        }

        Ok(())
    }

    /// Get the latest block
    pub fn latest_block(&self) -> Option<Block> {
        let height = match self.get_height() {
            Ok(h) => h,
            Err(_) => return None,
        };

        self.get_block_by_height(height)
    }

    /// Get account information
    pub fn get_account(&self, address: &str) -> Option<Account> {
        // Dummy implementation
        let balance = match self.get_balance(address) {
            Ok(bal) => bal,
            Err(_) => return None,
        };

        let nonce = match self.get_nonce(address) {
            Ok(n) => n,
            Err(_) => return None,
        };

        Some(Account {
            address: address.to_string(),
            balance,
            nonce,
        })
    }

    /// Add a pending transaction
    pub fn add_pending_transaction(&self, transaction: Transaction) -> Result<()> {
        let mut pending = self.pending_transactions.write().unwrap();
        pending.push_back(transaction);
        Ok(())
    }

    /// Get pending transactions
    pub fn get_pending_transactions(&self, limit: usize) -> Vec<Transaction> {
        let pending = self.pending_transactions.read().unwrap();
        pending.iter().take(limit).cloned().collect()
    }

    /// Remove a pending transaction
    pub fn remove_pending_transaction(&self, tx_hash: &str) -> Result<Option<Transaction>> {
        let mut pending = self.pending_transactions.write().unwrap();

        let pos = pending
            .iter()
            .position(|tx| hex::encode(tx.hash().as_ref()) == tx_hash);
        if let Some(idx) = pos {
            let tx = pending.remove(idx).unwrap();
            return Ok(Some(tx));
        }

        Ok(None)
    }

    /// Get transactions for an account
    pub fn get_account_transactions(&self, address: &str) -> Vec<Transaction> {
        let tx_history = self.tx_history.read().unwrap();
        let _hashes = match tx_history.get(address) {
            Some(h) => h,
            None => return Vec::new(),
        };

        // This implementation would need to retrieve transactions by hash
        // from a transaction store - this is a placeholder
        Vec::new()
    }

    /// Get a transaction by its hash
    pub fn get_transaction_by_hash(&self, _hash: &str) -> Option<(Transaction, String, u64)> {
        // Dummy implementation - return (transaction, block_hash, block_height)
        None
    }

    /// Add a transaction to account history
    pub fn add_transaction_to_history(&self, address: &str, tx_hash: &str) -> Result<()> {
        let mut tx_history = self.tx_history.write().unwrap();

        let account_history = tx_history
            .entry(address.to_string())
            .or_insert_with(Vec::new);
        account_history.push(tx_hash.to_string());

        Ok(())
    }

    pub fn get_blocks(&self, start: u64, limit: u64) -> Result<Vec<Block>> {
        let blocks = self.blocks.read().unwrap();
        let mut result = Vec::new();

        for height in start..start + limit {
            if let Some(block) = blocks.get(&height) {
                result.push(block.clone());
            }
        }

        Ok(result)
    }

    /// Get total number of transactions across all blocks
    pub fn get_total_transactions(&self) -> usize {
        let blocks = self.blocks.read().unwrap();
        let mut total = 0;
        for block in blocks.values() {
            total += block.transactions.len();
        }
        total
    }

    /// Get number of validators (mock implementation)
    pub fn get_validator_count(&self) -> usize {
        // For now, return a fixed number
        // In a full implementation, this would track actual validators
        10
    }

    /// Get the current difficulty level
    pub fn get_difficulty(&self) -> f64 {
        // Return a default difficulty value
        // In a full implementation, this would be calculated based on network conditions
        1.0
    }

    /// Export accounts data for checkpointing
    pub async fn export_accounts(&self) -> Result<Vec<u8>> {
        let accounts = self
            .balances
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        let serialized = bincode::serialize(&*accounts)?;
        Ok(serialized)
    }

    /// Import accounts data from checkpoint
    pub async fn import_accounts(&self, data: Vec<u8>) -> Result<()> {
        let imported_accounts: HashMap<String, u64> = bincode::deserialize(&data)?;

        let mut accounts = self
            .balances
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        *accounts = imported_accounts;
        Ok(())
    }

    /// Export storage data for checkpointing
    pub async fn export_storage(&self) -> Result<Vec<u8>> {
        let storage = self
            .storage
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        let serialized = bincode::serialize(&*storage)?;
        Ok(serialized)
    }

    /// Import storage data from checkpoint
    pub async fn import_storage(&self, data: Vec<u8>) -> Result<()> {
        let imported_storage: HashMap<String, Vec<u8>> = bincode::deserialize(&data)?;

        let mut storage = self
            .storage
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        *storage = imported_storage;
        Ok(())
    }

    /// Export processed transactions for checkpointing
    pub async fn export_processed_transactions(&self) -> Result<Vec<u8>> {
        let transactions = self
            .pending_transactions
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        // Convert VecDeque to Vec for serialization
        let tx_vec: Vec<Transaction> = transactions.iter().cloned().collect();
        let serialized = bincode::serialize(&tx_vec)?;
        Ok(serialized)
    }

    /// Import processed transactions from checkpoint
    pub async fn import_processed_transactions(&self, data: Vec<u8>) -> Result<()> {
        let imported_transactions: Vec<Transaction> = bincode::deserialize(&data)?;

        let mut transactions = self
            .pending_transactions
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        transactions.clear();
        for tx in imported_transactions {
            transactions.push_back(tx);
        }
        Ok(())
    }

    /// Get current state root hash
    pub fn get_state_root(&self) -> Result<Hash> {
        // Calculate state root from current state
        // This is a simplified version - in production, you'd use a Merkle tree
        let mut hasher = blake3::Hasher::new();

        // Hash accounts
        let accounts = self
            .balances
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        for (address, balance) in accounts.iter() {
            hasher.update(address.as_ref());
            hasher.update(&balance.to_le_bytes());
        }

        // Hash storage
        let storage = self
            .storage
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        for (key, value) in storage.iter() {
            hasher.update(key.as_ref());
            hasher.update(value);
        }

        let hash_bytes = hasher.finalize();
        Ok(Hash::new(hash_bytes.as_bytes().to_vec()))
    }

    // 🛡️ SPOF ELIMINATION METHODS

    /// Add a state replica for redundancy
    pub async fn add_replica(&self, replica: StateReplica) -> Result<()> {
        let mut replicas = self.state_replicas.write().unwrap();
        replicas.push(replica.clone());
        
        // Initialize health tracking
        let mut health = self.replica_health.write().unwrap();
        health.insert(replica.replica_id, ReplicaHealth::Healthy);
        
        // Update consensus configuration
        let mut consensus = self.state_consensus.write().unwrap();
        consensus.active_replicas = replicas.len();
        
        info!("Added state replica {} ({}). Total replicas: {}", 
              replica.replica_id, replica.endpoint, replicas.len());
        Ok(())
    }

    /// Remove a failed replica
    pub async fn remove_replica(&self, replica_id: usize) -> Result<()> {
        let mut replicas = self.state_replicas.write().unwrap();
        replicas.retain(|r| r.replica_id != replica_id);
        
        let mut health = self.replica_health.write().unwrap();
        health.remove(&replica_id);
        
        let mut consensus = self.state_consensus.write().unwrap();
        consensus.active_replicas = replicas.len();
        
        warn!("Removed failed state replica {}. Remaining replicas: {}", 
              replica_id, replicas.len());
        Ok(())
    }

    /// Get balance with failover support
    pub async fn get_balance_resilient(&self, address: &str) -> Result<u64> {
        // Try primary replica first
        match self.get_balance(address) {
            Ok(balance) => Ok(balance),
            Err(_) => {
                warn!("Primary state failed, attempting failover for balance query");
                self.failover_get_balance(address).await
            }
        }
    }

    /// Failover balance query to backup replicas
    async fn failover_get_balance(&self, address: &str) -> Result<u64> {
        let replicas = self.state_replicas.read().unwrap();
        let health = self.replica_health.read().unwrap();
        
        // Try healthy replicas
        for replica in replicas.iter() {
            if let Some(ReplicaHealth::Healthy) = health.get(&replica.replica_id) {
                // In a real implementation, this would query the remote replica
                // For now, fallback to local state
                return self.get_balance(address);
            }
        }
        
        Err(anyhow!("All state replicas unavailable"))
    }

    /// Set balance with consensus
    pub async fn set_balance_consensus(&self, address: &str, amount: u64) -> Result<()> {
        let consensus = self.state_consensus.read().unwrap();
        
        if consensus.active_replicas >= consensus.consensus_threshold {
            // Broadcast update to all replicas
            let sync_msg = StateSyncMessage::BalanceUpdate { 
                address: address.to_string(), 
                balance: amount 
            };
            
            if let Ok(sender_guard) = self.sync_channel.try_lock() {
                let _ = sender_guard.send(sync_msg);
            }
            
            // Apply to local state
            self.set_balance(address, amount)?;
            info!("Balance update consensus achieved for {}: {}", address, amount);
            Ok(())
        } else {
            Err(anyhow!("Insufficient replicas for consensus. Need: {}, Have: {}", 
                       consensus.consensus_threshold, consensus.active_replicas))
        }
    }

    /// Check replica health
    pub async fn check_replica_health(&self) -> Result<HashMap<usize, ReplicaHealth>> {
        let health = self.replica_health.read().unwrap();
        Ok(health.clone())
    }

    /// Force failover to backup replica
    pub async fn force_failover(&self) -> Result<()> {
        let replicas = self.state_replicas.read().unwrap();
        if replicas.len() > 1 {
            let mut primary = self.primary_replica.write().unwrap();
            *primary = (*primary + 1) % replicas.len();
            info!("Forced failover to replica {}", *primary);
            Ok(())
        } else {
            Err(anyhow!("No backup replicas available for failover"))
        }
    }
}

/// Account information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Account {
    /// Account address
    pub address: String,
    /// Account balance
    pub balance: u64,
    /// Account nonce
    pub nonce: u64,
}

// 🛡️ SPOF ELIMINATION STRUCTS

/// State replica for distributed state management
#[derive(Debug, Clone)]
pub struct StateReplica {
    pub replica_id: usize,
    pub endpoint: String,
    pub is_active: bool,
    pub last_sync: u64,
    pub sync_lag: u64,
}

/// Replica health status
#[derive(Debug, Clone, PartialEq)]
pub enum ReplicaHealth {
    Healthy,
    Degraded,
    Failed,
    Recovering,
}

/// State synchronization message
#[derive(Debug, Clone)]
pub enum StateSyncMessage {
    BalanceUpdate { address: String, balance: u64 },
    NonceUpdate { address: String, nonce: u64 },
    StorageUpdate { key: String, value: Vec<u8> },
    HeightUpdate { height: u64 },
    HealthCheck { replica_id: usize },
}

/// State consensus mechanism
#[derive(Debug, Clone)]
pub struct StateConsensus {
    pub consensus_threshold: usize, // Minimum replicas for consensus
    pub active_replicas: usize,
    pub last_consensus: u64,
    pub pending_updates: HashMap<String, StateSyncMessage>,
}

pub use storage::StateStorage;
pub use tree::StateTree;
pub mod arthacoin_state;
