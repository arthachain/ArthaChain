mod rocksdb_storage;
mod svdb_storage;
pub mod transaction;
pub mod hybrid_storage;
pub mod blockchain_storage;

pub use rocksdb_storage::RocksDbStorage;
pub use svdb_storage::SvdbStorage;

use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::path::Path;
use log::{debug, warn};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;
use crate::config::Config;
use hex;
use crate::types::Hash;
use blake3;
use downcast_rs::{Downcast, impl_downcast};

/// Trait for storage operations
#[async_trait]
pub trait Storage: Send + Sync + Downcast {
    async fn store(&self, data: &[u8]) -> Result<Hash>;
    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>>;
    async fn exists(&self, hash: &Hash) -> Result<bool>;
    async fn delete(&self, hash: &Hash) -> Result<()>;
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool>;
    async fn close(&self) -> Result<()>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl_downcast!(Storage);

/// Trait for storage initialization
#[async_trait]
pub trait StorageInit: Send + Sync {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()>;
}

/// Trait for storage backend operations
pub trait StorageBackend: Storage + StorageInit + Send + Sync + 'static {}

/// Storage type for the blockchain
#[derive(Clone, Debug)]
pub enum StorageType {
    /// In-memory storage (for testing)
    Memory,
    /// LevelDB storage
    LevelDB,
    /// RocksDB storage
    RocksDB,
    /// SQL storage
    SQL,
    /// Custom storage implementation
    Custom(String),
    Sled,
}

/// Storage configuration
#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub storage_type: StorageType,
    pub path: String,
    pub max_size: usize,
    pub compression: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            storage_type: StorageType::RocksDB,
            path: "data/storage".to_string(),
            max_size: 1024 * 1024 * 1024, // 1GB
            compression: true,
        }
    }
}

/// Hybrid storage combining RocksDB and SVDB
pub struct HybridStorage {
    /// RocksDB storage for on-chain data
    rocksdb: Box<dyn Storage>,
    
    /// SVDB storage for off-chain data
    svdb: Box<dyn Storage>,
    
    /// Size threshold (in bytes) for deciding between RocksDB and SVDB
    size_threshold: usize,
}

impl HybridStorage {
    /// Create a new hybrid storage
    pub async fn new(rocksdb: Box<dyn Storage>, svdb: Box<dyn Storage>) -> Self {
        Self {
            rocksdb,
            svdb,
            size_threshold: 1024, // Default size threshold
        }
    }
    
    /// Access underlying RocksDB storage
    pub fn rocksdb(&self) -> &Box<dyn Storage> {
        &self.rocksdb
    }
    
    /// Access underlying SVDB storage
    pub fn svdb(&self) -> &Box<dyn Storage> {
        &self.svdb
    }
    
    /// Determine if data should be stored in RocksDB or SVDB
    fn should_use_rocksdb(&self, data: &[u8]) -> bool {
        data.len() < self.size_threshold
    }
}

#[async_trait]
impl Storage for HybridStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        if self.should_use_rocksdb(data) {
            // Store directly in RocksDB
            debug!("Storing '{}' ({} bytes) in RocksDB", data.len(), data.len());
            self.rocksdb.store(data).await
        } else {
            // Store in SVDB and reference in RocksDB
            debug!("Storing '{}' ({} bytes) in SVDB", data.len(), data.len());
            self.svdb.store(data).await
        }
    }
    
    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>> {
        match self.rocksdb.exists(hash).await? {
            true => {
                let data = self.rocksdb.retrieve(hash).await?;
                Ok(data)
            }
            false => {
                self.svdb.retrieve(hash).await
            }
        }
    }
    
    async fn exists(&self, hash: &Hash) -> Result<bool> {
        if self.rocksdb.exists(hash).await? {
            return Ok(true);
        }
        self.svdb.exists(hash).await
    }
    
    async fn delete(&self, hash: &Hash) -> Result<()> {
        if self.rocksdb.exists(hash).await? {
            self.rocksdb.delete(hash).await?;
        }
        if self.svdb.exists(hash).await? {
            self.svdb.delete(hash).await?;
        }
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        if self.rocksdb.exists(hash).await? {
            return self.rocksdb.verify(hash, data).await;
        }
        self.svdb.verify(hash, data).await
    }
    
    async fn close(&self) -> Result<()> {
        self.rocksdb.close().await?;
        self.svdb.close().await?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Transaction for atomic operations across storage systems
pub struct StorageTransaction<'a> {
    hybrid: &'a HybridStorage,
    operations: Vec<StorageOperation>,
    committed: bool,
}

/// Represents a storage operation
enum StorageOperation {
    Store { data: Vec<u8> },
    Delete { key: String },
}

impl<'a> StorageTransaction<'a> {
    /// Create a new storage transaction
    pub fn new(hybrid: &'a HybridStorage) -> Self {
        Self {
            hybrid,
            operations: Vec::new(),
            committed: false,
        }
    }
    
    /// Add store operation to transaction
    pub fn store(&mut self, data: &[u8]) -> &mut Self {
        self.operations.push(StorageOperation::Store {
            data: data.to_vec(),
        });
        self
    }
    
    /// Add delete operation to transaction
    pub fn delete(&mut self, key: &str) -> &mut Self {
        self.operations.push(StorageOperation::Delete {
            key: key.to_string(),
        });
        self
    }
    
    /// Commit the transaction
    pub async fn commit(mut self) -> Result<()> {
        // Perform all operations
        for op in &self.operations {
            match op {
                StorageOperation::Store { data } => {
                    self.hybrid.store(data).await?;
                },
                StorageOperation::Delete { key } => {
                    // Convert key to Hash
                    let hash_bytes = blake3::hash(key.as_bytes()).as_bytes().to_vec();
                    let hash = Hash::new(hash_bytes);
                    let _ = self.hybrid.delete(&hash).await?;
                },
            }
        }
        
        self.committed = true;
        Ok(())
    }
}

impl<'a> Drop for StorageTransaction<'a> {
    fn drop(&mut self) {
        if !self.committed && !self.operations.is_empty() {
            warn!("Storage transaction dropped without commit ({} operations)", 
                  self.operations.len());
        }
    }
}

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
        
        Ok(Self {
            rocksdb,
            memory,
        })
    }
    
    /// Put a key-value pair
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut memory = self.memory.lock().unwrap();
        memory.insert(key.to_vec(), value.to_vec());
        Ok(())
    }
    
    /// Get a value by key
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let memory = self.memory.lock().unwrap();
        let value = memory.get(key).cloned();
        Ok(value)
    }
    
    /// Delete a key-value pair
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        let mut memory = self.memory.lock().unwrap();
        memory.remove(key);
        Ok(())
    }
    
    /// Check if a key exists
    pub fn exists(&self, key: &[u8]) -> Result<bool> {
        let memory = self.memory.lock().unwrap();
        Ok(memory.contains_key(key))
    }
    
    /// Get all keys with a prefix
    pub fn get_keys_with_prefix(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        let memory = self.memory.lock().unwrap();
        let keys = memory.keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        Ok(keys)
    }

    #[allow(dead_code)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[async_trait]
impl Storage for BlockchainStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        let hash_bytes = blake3::hash(data).as_bytes().to_vec();
        let hash = Hash::new(hash_bytes);
        self.put(data, data)?;
        Ok(hash)
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>> {
        match self.get(hash.as_bytes())? {
            Some(data) => Ok(data),
            None => Err(anyhow!("Key not found: {}", hex::encode(hash.as_bytes())))
        }
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        self.exists(hash.as_bytes())
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        self.delete(hash.as_bytes())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        match self.get(data) {
            Ok(Some(stored_data)) => {
                let calculated_hash_bytes = blake3::hash(&stored_data).as_bytes().to_vec();
                let calculated_hash = Hash::new(calculated_hash_bytes);
                Ok(calculated_hash == *hash)
            },
            Ok(None) => Err(anyhow!("Key not found: {}", hex::encode(hash.as_bytes()))),
            Err(e) => Err(e),
        }
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
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
            return Err(anyhow!("Failed to get mutable reference to RocksDB storage"));
        }
        
        Ok(())
    }
}

#[async_trait]
impl StorageBackend for BlockchainStorage {}

#[async_trait]
impl StorageInit for HybridStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // Extract the path and convert to PathBuf
        let path_ref = path.as_ref();
        let path_buf = path_ref.as_ref().to_path_buf();
        
        // Create a boxed path for initialization
        let box_path = Box::new(path_buf) as Box<dyn AsRef<Path> + Send + Sync>;
        
        // Get a mutable reference to the RocksDbStorage
        if let Some(storage) = self.rocksdb.as_any_mut().downcast_mut::<RocksDbStorage>() {
            storage.init(box_path).await?;
        } else {
            return Err(anyhow!("Failed to get mutable reference to RocksDB storage"));
        }
        
        // SVDB doesn't need initialization
        Ok(())
    }
}

#[async_trait]
impl StorageBackend for HybridStorage {} 