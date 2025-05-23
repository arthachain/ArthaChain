pub mod storage;
pub mod tree;

use crate::config::Config;
use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::types::Hash;
use anyhow::Result;
use std::collections::HashMap;
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

    /// Get the latest block hash
    pub fn get_latest_block_hash(&self) -> Result<String> {
        // This is a dummy implementation - in a real implementation we would track block hashes
        Ok("0000000000000000000000000000000000000000000000000000000000000000".to_string())
    }

    /// Get a block by its hash
    pub fn get_block_by_hash(&self, _hash: &Hash) -> Option<Block> {
        // Implementation here
        None // TODO: Implement actual block retrieval
    }

    /// Get a block by its height
    pub fn get_block_by_height(&self, _height: u64) -> Option<Block> {
        // Dummy implementation
        None
    }

    /// Get the latest block
    pub fn latest_block(&self) -> Option<Block> {
        // Dummy implementation
        None
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

    /// Get pending transactions
    pub fn get_pending_transactions(&self, _limit: usize) -> Vec<Transaction> {
        // Dummy implementation
        Vec::new()
    }

    /// Add a pending transaction
    pub fn add_pending_transaction(&self, _transaction: Transaction) -> Result<()> {
        // Dummy implementation
        Ok(())
    }

    /// Get transactions for an account
    pub fn get_account_transactions(&self, _address: &str) -> Vec<Transaction> {
        // Dummy implementation
        Vec::new()
    }

    /// Get a transaction by its hash
    pub fn get_transaction_by_hash(&self, _hash: &str) -> Option<(Transaction, String, u64)> {
        // Dummy implementation - return (transaction, block_hash, block_height)
        None
    }

    pub fn get_blocks(&self, _start: u64, _limit: u64) -> Result<Vec<Block>> {
        // Implementation here
        Ok(Vec::new()) // TODO: Implement actual block retrieval
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
