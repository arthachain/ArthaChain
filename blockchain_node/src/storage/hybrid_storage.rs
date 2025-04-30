use std::path::Path;
use async_trait::async_trait;
use crate::types::Hash;
use super::Storage;
use super::StorageInit;
use super::StorageBackend;
use super::rocksdb_storage::RocksDbStorage;
use super::svdb_storage::SvdbStorage;
use anyhow::{Result, Error};
use crate::config::Config;
use downcast_rs::Downcast;

/// Hybrid storage combining RocksDB and SVDB
pub struct HybridStorage {
    /// RocksDB storage for on-chain data
    rocksdb: Box<dyn Storage>,
    
    /// SVDB storage for off-chain data
    svdb: Box<dyn Storage>,
}

impl HybridStorage {
    /// Create a new hybrid storage
    pub fn new(config: Config) -> Self {
        let rocksdb = Box::new(RocksDbStorage::new());
        let svdb_url = config.svdb_url.as_deref().unwrap_or("http://localhost:8000");
        let svdb = Box::new(SvdbStorage::new(svdb_url).expect("Failed to create SVDB storage"));
        
        Self {
            rocksdb,
            svdb,
        }
    }
    
    /// Create a new hybrid storage asynchronously
    pub async fn new_async(
        db_path: impl AsRef<Path>,
        svdb_url: &str,
        _size_threshold: usize,
    ) -> Result<Self, Error> {
        let mut rocksdb = RocksDbStorage::new();
        let mut svdb = SvdbStorage::new(svdb_url)?;
        
        // Initialize and verify connections
        let path = Box::new(db_path.as_ref().to_path_buf());
        rocksdb.init(path.clone()).await?;
        svdb.init(path).await?;
        
        Ok(Self {
            rocksdb: Box::new(rocksdb),
            svdb: Box::new(svdb),
        })
    }
    
    /// Access underlying RocksDB storage
    pub fn rocksdb(&self) -> &dyn Storage {
        self.rocksdb.as_ref()
    }
    
    /// Access underlying SVDB storage
    pub fn svdb(&self) -> &dyn Storage {
        self.svdb.as_ref()
    }
    
    /// Determine if data should be stored in RocksDB or SVDB
    fn should_use_rocksdb(&self, data: &[u8]) -> bool {
        // Use a default threshold of 1024 bytes
        let threshold = 1024;
        
        data.len() < threshold
    }
}

#[async_trait]
impl Storage for HybridStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash, Error> {
        if self.should_use_rocksdb(data) {
            self.rocksdb.store(data).await
        } else {
            self.svdb.store(data).await
        }
    }
    
    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>, Error> {
        // First try to get directly from RocksDB
        match self.rocksdb.retrieve(hash).await {
            Ok(data) => Ok(data),
            Err(_) => {
                // Check if it's in SVDB
                self.svdb.retrieve(hash).await
            }
        }
    }
    
    async fn exists(&self, hash: &Hash) -> Result<bool, Error> {
        // Check if exists directly in RocksDB
        if self.rocksdb.exists(hash).await? {
            return Ok(true);
        }
        
        // Check if exists in SVDB
        self.svdb.exists(hash).await
    }
    
    async fn delete(&self, hash: &Hash) -> Result<(), Error> {
        // Try to delete from both storages
        let _ = self.rocksdb.delete(hash).await;
        let _ = self.svdb.delete(hash).await;
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool, Error> {
        // Try to verify in both storages
        if self.rocksdb.exists(hash).await? {
            return self.rocksdb.verify(hash, data).await;
        }
        if self.svdb.exists(hash).await? {
            return self.svdb.verify(hash, data).await;
        }
        Ok(false)
    }
    
    async fn close(&self) -> Result<(), Error> {
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

#[async_trait]
impl StorageInit for HybridStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // Initialize RocksDB with the provided path
        if let Some(storage) = Downcast::as_any_mut(&mut *self.rocksdb).downcast_mut::<RocksDbStorage>() {
            let path_buf = path.as_ref().as_ref().to_path_buf();
            let box_path = Box::new(path_buf) as Box<dyn AsRef<Path> + Send + Sync>;
            storage.init(box_path).await?;
            // SVDB doesn't need initialization
            Ok(())
        } else {
            Err(Error::msg("Failed to get mutable reference to RocksDB storage"))
        }
    }
}

impl StorageBackend for HybridStorage {} 