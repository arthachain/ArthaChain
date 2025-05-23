use super::RocksDbStorage;
use super::{Result, Storage, StorageBackend, StorageError, StorageInit};
use crate::config::Config;
use crate::types::Hash;
use async_trait::async_trait;
use blake3;
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
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
    pub fn new(_config: &Config) -> Result<Self> {
        let rocksdb = Box::new(RocksDbStorage::new());
        let memory = Arc::new(Mutex::new(HashMap::new()));

        Ok(Self { rocksdb, memory })
    }

    /// Calculate hash for data using blake3
    #[allow(dead_code)]
    fn calculate_hash(data: &[u8]) -> Hash {
        let hash = blake3::hash(data);
        hash.as_bytes().to_vec().try_into().unwrap()
    }

    /// Put a key-value pair
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut memory = self
            .memory
            .lock()
            .map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        memory.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    /// Get a value by key
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let db = self
            .rocksdb
            .as_any()
            .downcast_ref::<RocksDbStorage>()
            .ok_or_else(|| {
                StorageError::Other("Failed to downcast to RocksDbStorage".to_string())
            })?;
        let value = db.get(key).map(|v| v.clone());
        Ok(value)
    }

    /// Delete a key-value pair
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        let mut memory = self
            .memory
            .lock()
            .map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        memory.remove(key);
        Ok(())
    }

    /// Check if a key exists
    pub fn exists(&self, key: &[u8]) -> Result<bool> {
        let memory = self
            .memory
            .lock()
            .map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        Ok(memory.contains_key(key))
    }

    /// Get all keys with a prefix
    pub fn get_keys_with_prefix(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        let memory = self
            .memory
            .lock()
            .map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        let keys = memory
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        Ok(keys)
    }
}

#[async_trait]
impl Storage for BlockchainStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        self.rocksdb.store(data).await
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        self.rocksdb.retrieve(hash).await
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        self.rocksdb.exists(hash).await
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        self.rocksdb.delete(hash).await
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        self.rocksdb.verify(hash, data).await
    }

    async fn close(&self) -> Result<()> {
        self.rocksdb.close().await
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[async_trait]
impl StorageInit for BlockchainStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // Extract the path and convert to PathBuf
        let path_ref = path.as_ref();
        let path_buf = path_ref.as_ref().to_path_buf();

        // Get a mutable reference to the RocksDbStorage
        if let Some(storage) = self.rocksdb.as_any_mut().downcast_mut::<RocksDbStorage>() {
            let box_path = Box::new(path_buf) as Box<dyn AsRef<Path> + Send + Sync>;
            storage.init(box_path).await?;
        } else {
            return Err(StorageError::Other(
                "Failed to get mutable reference to RocksDB storage".to_string(),
            ));
        }

        Ok(())
    }
}

impl StorageBackend for BlockchainStorage {}
