use super::{Storage, StorageInit};
use crate::config::Config;
use crate::storage::rocksdb_storage::RocksDbStorage;
use crate::types::Hash;
use async_trait::async_trait;
use blake3;
use sha3;
use sha3::Digest;

use std::collections::HashMap;

use std::sync::{Arc, Mutex};
/// Storage for blockchain data
pub struct BlockchainStorage {
    /// RocksDB storage for on-chain data
    rocksdb: Box<dyn Storage>,
    /// In-memory storage for key-value pairs (used in tests)
    memory: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl BlockchainStorage {
    /// Create a new blockchain storage instance
    pub fn new(_config: &Config) -> anyhow::Result<Self> {
        let rocksdb = Box::new(RocksDbStorage::new());
        let memory = Arc::new(Mutex::new(HashMap::new()));

        Ok(Self { rocksdb, memory })
    }

    /// Calculate hash for data using blake3
    #[allow(dead_code)]
    fn calculate_hash(data: &[u8]) -> Hash {
        let hash = blake3::hash(data);
        Hash::new(hash.as_bytes().to_vec())
    }

    /// Put a key-value pair
    pub fn put(&self, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
        let mut memory = self
            .memory
            .lock()
            .map_err(|e| anyhow::anyhow!(format!("Lock error: {e}")))?;
        memory.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    /// Get a value by key (direct delegate)
    pub async fn get_direct(&self, key: &[u8]) -> crate::storage::Result<Option<Vec<u8>>> {
        self.rocksdb.get(key).await
    }

    /// Delete a key-value pair
    pub fn delete(&self, key: &[u8]) -> anyhow::Result<()> {
        let mut memory = self
            .memory
            .lock()
            .map_err(|e| anyhow::anyhow!(format!("Lock error: {e}")))?;
        memory.remove(key);
        Ok(())
    }

    /// Check if a key exists
    pub fn exists(&self, key: &[u8]) -> anyhow::Result<bool> {
        let memory = self
            .memory
            .lock()
            .map_err(|e| anyhow::anyhow!(format!("Lock error: {e}")))?;
        Ok(memory.contains_key(key))
    }

    /// Get all keys with a prefix
    pub fn get_keys_with_prefix(&self, prefix: &[u8]) -> anyhow::Result<Vec<Vec<u8>>> {
        if prefix.is_empty() {
            return Err(anyhow::anyhow!("Empty prefix provided".to_string(),));
        }

        let memory = self
            .memory
            .lock()
            .map_err(|e| anyhow::anyhow!(format!("Failed to acquire lock: {}", e)))?;

        let mut keys = Vec::new();
        for key in memory.keys() {
            if key.starts_with(prefix) {
                keys.push(key.clone());
            }
        }

        // Sort keys for consistent ordering
        keys.sort();

        Ok(keys)
    }

    pub fn get_block_hash(&self, hash: &[u8]) -> anyhow::Result<Hash> {
        if hash.is_empty() {
            return Err(anyhow::anyhow!("Empty hash provided".to_string()));
        }

        let mut hasher = sha3::Sha3_256::new();
        hasher.update(hash);
        let result = hasher.finalize();

        Ok(Hash::new(result.to_vec()))
    }

    pub fn get_value(&self, _key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

#[async_trait]
impl Storage for BlockchainStorage {
    async fn get(&self, key: &[u8]) -> crate::storage::Result<Option<Vec<u8>>> {
        self.rocksdb.get(key).await
    }

    async fn put(&self, key: &[u8], value: &[u8]) -> crate::storage::Result<()> {
        self.rocksdb.put(key, value).await
    }

    async fn delete(&self, key: &[u8]) -> crate::storage::Result<()> {
        self.rocksdb.delete(key).await
    }

    async fn exists(&self, key: &[u8]) -> crate::storage::Result<bool> {
        self.rocksdb.exists(key).await
    }

    async fn list_keys(&self, _prefix: &[u8]) -> crate::storage::Result<Vec<Vec<u8>>> {
        self.rocksdb.list_keys(_prefix).await
    }

    async fn get_stats(&self) -> crate::storage::Result<crate::storage::StorageStats> {
        self.rocksdb.get_stats().await
    }

    async fn flush(&self) -> crate::storage::Result<()> {
        self.rocksdb.flush().await
    }

    async fn close(&self) -> crate::storage::Result<()> {
        self.rocksdb.close().await
    }
}

#[async_trait]
impl StorageInit for BlockchainStorage {
    async fn init(&self, _config: &crate::storage::StorageConfig) -> crate::storage::Result<()> {
        // Extract the path from config
        Ok(())
    }
}
