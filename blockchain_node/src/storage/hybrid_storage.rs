use crate::storage::{Storage, StorageInit, StorageError, StorageBackend, RocksDbStorage, SvdbStorage};
use crate::types::Hash;
use async_trait::async_trait;
use std::path::Path;
use std::any::Any;

type Result<T> = std::result::Result<T, StorageError>;

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
    /// Create a new hybrid storage instance
    pub fn new(svdb_url: String, size_threshold: usize) -> Result<Self> {
        let rocksdb = Box::new(RocksDbStorage::new());
        let svdb = Box::new(SvdbStorage::new(svdb_url)?);
        
        Ok(Self {
            rocksdb,
            svdb,
            size_threshold,
        })
    }
}

#[async_trait]
impl Storage for HybridStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        if data.len() > self.size_threshold {
            self.svdb.store(data).await
        } else {
            self.rocksdb.store(data).await
        }
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        match self.rocksdb.retrieve(hash).await {
            Ok(Some(data)) => Ok(Some(data)),
            Ok(None) => self.svdb.retrieve(hash).await,
            Err(_) => self.svdb.retrieve(hash).await,
        }
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        match self.rocksdb.exists(hash).await {
            Ok(true) => Ok(true),
            Ok(false) => self.svdb.exists(hash).await,
            Err(_) => self.svdb.exists(hash).await,
        }
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        // Try to delete from both storages
        let rocksdb_result = self.rocksdb.delete(hash).await;
        let svdb_result = self.svdb.delete(hash).await;

        // Return error only if both failed
        match (rocksdb_result, svdb_result) {
            (Ok(_), _) | (_, Ok(_)) => Ok(()),
            (Err(e1), Err(e2)) => Err(StorageError::Other(format!(
                "Failed to delete from both storages - RocksDB: {}, SVDB: {}", e1, e2
            ))),
        }
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        match self.rocksdb.verify(hash, data).await {
            Ok(true) => Ok(true),
            Ok(false) => self.svdb.verify(hash, data).await,
            Err(_) => self.svdb.verify(hash, data).await,
        }
    }

    async fn close(&self) -> Result<()> {
        // Close both storages
        let rocksdb_result = self.rocksdb.close().await;
        let svdb_result = self.svdb.close().await;

        // Return error only if both failed
        match (rocksdb_result, svdb_result) {
            (Ok(_), _) | (_, Ok(_)) => Ok(()),
            (Err(e1), Err(e2)) => Err(StorageError::Other(format!(
                "Failed to close both storages - RocksDB: {}, SVDB: {}", e1, e2
            ))),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[async_trait]
impl StorageInit for HybridStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // Extract the path and convert to PathBuf
        let path_ref = path.as_ref();
        let path_buf = path_ref.as_ref().to_path_buf();
        
        // Initialize RocksDB storage
        let rocksdb_path = path_buf.join("rocksdb");
        let box_path = Box::new(rocksdb_path) as Box<dyn AsRef<Path> + Send + Sync>;
        self.rocksdb.as_any_mut().downcast_mut::<RocksDbStorage>()
            .ok_or_else(|| StorageError::Other("Failed to get mutable reference to RocksDB storage".to_string()))?
            .init(box_path).await?;
        
        // Initialize SVDB storage
        let svdb_path = path_buf.join("svdb");
        let box_path = Box::new(svdb_path) as Box<dyn AsRef<Path> + Send + Sync>;
        self.svdb.as_any_mut().downcast_mut::<SvdbStorage>()
            .ok_or_else(|| StorageError::Other("Failed to get mutable reference to SVDB storage".to_string()))?
            .init(box_path).await?;
        
        Ok(())
    }
}

impl StorageBackend for HybridStorage {} 