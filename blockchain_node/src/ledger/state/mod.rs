pub mod storage;
pub mod tree;

use crate::config::Config;
use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::types::Hash;
use anyhow::{anyhow, Result};
use log::debug;
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

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
}

impl State {
    /// Create a new state instance
    pub fn new(_config: &Config) -> Result<Self> {
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
        let hash = block.hash().to_string();

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
            .position(|tx| tx.hash().to_string() == tx_hash);
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

pub use storage::StateStorage;
pub use tree::StateTree;
