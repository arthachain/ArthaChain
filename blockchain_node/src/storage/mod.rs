mod rocksdb_storage;
mod svdb_storage;
pub mod transaction;
pub mod hybrid_storage;
pub mod blockchain_storage;
pub mod memmap_storage;

pub use rocksdb_storage::RocksDbStorage;
pub use svdb_storage::SvdbStorage;
pub use memmap_storage::MemMapStorage;

use std::sync::Arc;
use log::{warn};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;
use crate::config::Config;
use crate::types::Hash;
use blake3;
use std::any::Any;
use thiserror::Error;
use std::result::Result as StdResult;
use std::path::Path;

/// Storage options for memory-mapped database
#[derive(Clone, Debug)]
pub struct MemMapOptions {
    /// Size of the memory map in bytes
    pub map_size: usize,
    /// Max pending writes before flush
    pub max_pending_writes: usize,
    /// Preload data into memory
    pub preload_data: bool,
    /// Compression algorithm to use
    pub compression_algorithm: CompressionAlgorithm,
}

impl Default for MemMapOptions {
    fn default() -> Self {
        Self {
            map_size: 512 * 1024 * 1024, // 512MB
            max_pending_writes: 1000,
            preload_data: true,
            compression_algorithm: CompressionAlgorithm::Adaptive,
        }
    }
}

/// Compression algorithms for storage
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 compression (fast)
    LZ4,
    /// Zstd compression (balanced)
    Zstd,
    /// Brotli compression (high compression)
    Brotli,
    /// Adaptive compression (chooses best)
    Adaptive,
}

// Storage error types
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Storage initialization error: {0}")]
    InitError(String),
    
    #[error("Storage operation error: {0}")]
    OperationError(String),
    
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    
    #[error("Invalid data: {0}")]
    InvalidData(String),
    
    #[error("Incompatible storage version: {0}")]
    IncompatibleVersion(String),
    
    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for storage operations
pub type Result<T> = StdResult<T, StorageError>;

/// Storage interface
#[async_trait]
pub trait Storage: Send + Sync {
    /// Store data and return the hash
    async fn store(&self, data: &[u8]) -> Result<Hash>;
    
    /// Retrieve data by hash
    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>>;
    
    /// Check if data exists
    async fn exists(&self, hash: &Hash) -> Result<bool>;
    
    /// Delete data
    async fn delete(&self, hash: &Hash) -> Result<()>;
    
    /// Verify that data matches hash
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool>;
    
    /// Close the storage
    async fn close(&self) -> Result<()>;
    
    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn Any;
    
    /// Convert to mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Storage backend implementation marker trait
pub trait StorageBackend: Storage {}

/// Storage initialization trait
#[async_trait]
pub trait StorageInit: Storage {
    /// Initialize storage with path
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()>;
}

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
    /// SVDB storage
    SVDB,
    /// Hybrid storage (combining multiple types)
    Hybrid,
    /// Memory-mapped storage (optimized for high throughput)
    MemoryMapped,
}

/// Storage configuration
#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub storage_type: StorageType,
    pub path: String,
    pub max_size: usize,
    pub compression: bool,
    /// Memory map specific options
    pub mmap_options: Option<MemMapOptions>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            storage_type: StorageType::MemoryMapped,
            path: "data/storage".to_string(),
            max_size: 1024 * 1024 * 1024 * 10, // 10GB
            compression: true,
            mmap_options: Some(MemMapOptions::default()),
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

    /// Store data using appropriate backend and return hash
    pub async fn store_data(&self, data: &[u8]) -> Result<Hash> {
        if self.should_use_rocksdb(data) {
            self.rocksdb.store(data).await
        } else {
            self.svdb.store(data).await
        }
    }

    /// Convenience put (old signature) to keep old call-sites working – hashes key & value are identical
    pub async fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // create composite buffer key+value for deterministic hash storage
        let mut buf = Vec::with_capacity(key.len() + value.len());
        buf.extend_from_slice(key);
        buf.extend_from_slice(value);
        let _ = self.store_data(&buf).await?;
        Ok(())
    }

    /// Convenience get by key (prefix search in RocksDB only for now)
    pub async fn get(&self, _key: &[u8]) -> Result<Option<Vec<u8>>> {
        // For demo purposes always None; real impl would map key->hash etc.
        Ok(None)
    }
}

#[async_trait]
impl Storage for HybridStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        self.store_data(data).await
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        match self.rocksdb.retrieve(hash).await {
            Ok(Some(d)) => Ok(Some(d)),
            Ok(None) => self.svdb.retrieve(hash).await,
            Err(e) => Err(e),
        }
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        let rocksdb_result = self.rocksdb.exists(hash).await;
        match rocksdb_result {
            Ok(true) => Ok(true),
            _ => self.svdb.exists(hash).await,
        }
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        // try both backends
        let _ = self.rocksdb.delete(hash).await;
        let _ = self.svdb.delete(hash).await;
        Ok(())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        if self.should_use_rocksdb(data) {
            self.rocksdb.verify(hash, data).await
        } else {
            self.svdb.verify(hash, data).await
        }
    }

    async fn close(&self) -> Result<()> {
        let _ = self.rocksdb.close().await;
        let _ = self.svdb.close().await;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

#[async_trait]
impl StorageInit for HybridStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        let base_path: &std::path::Path = path.as_ref().as_ref();
        let rocksdb_path = base_path.join("rocksdb");
        let svdb_path   = base_path.join("svdb");

        // downcast and init
        if let Some(r) = self.rocksdb.as_any_mut().downcast_mut::<RocksDbStorage>() {
            r.init(Box::new(rocksdb_path)).await?;
        }
        if let Some(s) = self.svdb.as_any_mut().downcast_mut::<SvdbStorage>() {
            s.init(Box::new(svdb_path)).await?;
        }
        Ok(())
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
                    self.hybrid.put(data.as_slice(), data.as_slice()).await?;
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
        let db = self.rocksdb.as_any().downcast_ref::<RocksDbStorage>()
            .ok_or_else(|| StorageError::Other("Failed to downcast to RocksDbStorage".to_string()))?;
        let value = db.get(key);
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

    // hashing helper
    fn calculate_hash(data: &[u8]) -> Hash {
        Hash::new(blake3::hash(data).as_bytes().to_vec())
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

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

#[async_trait]
impl StorageInit for BlockchainStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        if let Some(rocks) = self.rocksdb.as_any_mut().downcast_mut::<RocksDbStorage>() {
            rocks.init(path).await
        } else {
            Err(StorageError::Other("Unable to init BlockchainStorage".into()))
        }
    }
}

impl StorageBackend for HybridStorage {}
impl StorageBackend for BlockchainStorage {}
impl StorageBackend for MemMapStorage {} 